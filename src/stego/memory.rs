// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Memory budget prediction + telemetry.
//!
//! Provides peak-RSS prediction (`predict_peak_memory()`, the `ModeId` /
//! `GhostRung` enums, [`ghost_lane_plan`]) and the public API every bridge calls at
//! startup:
//!
//! - `set_memory_budget(Some(bytes))` — bridge tells core the
//!   per-process budget. iOS uses `os_proc_available_memory() / 2`;
//!   Android uses `ActivityManager.getMemoryInfo().availMem / 2`;
//!   WASM hardcodes 800 MB; CLI leaves `None` (no limit).
//! - `set_telemetry_hook(Some(fn))` — optional callback for analytics.
//!   Silent in release if unset. In debug builds, unset hook routes
//!   through `eprintln!` so developers see the trace.
//!
//! The encode paths use the budget to select a ladder rung.
//!
//! Coefficients here are STATIC — the `perf_memory_audit` harness
//! produces real RSS measurements that refine them. Treat the
//! bytes-per-pixel numbers as upper bounds; if actual RSS comes in
//! higher, raise the safety factor before changing the per-component
//! breakdown.

use std::sync::{Mutex, OnceLock};

/// Encode/decode mode identifier used for memory prediction and telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeId {
    GhostNoShadow = 1,
    GhostShadow = 2,
    GhostSi = 3,
    GhostSiShadow = 4,
    Armor = 5,
    Decode = 6,
    /// AV1 whole-video shadow encode (streaming Pass 1 per WV.6).
    /// Peak prediction lives in
    /// `predict_peak_memory_av1_whole_video_shadow` — the bytes_per_pixel
    /// table doesn't apply because peak depends on n_frames + gop_size
    /// (per-GOP working set + accumulators) rather than image area alone.
    Av1WholeVideoShadow = 7,
}

/// Ladder rung of a Ghost encode (`docs/design/image/ghost-embedding-v3.md`,
/// "Memory"): how many embedding lanes are planned at once and whether a lane
/// keeps its trellis back pointers in memory. Every rung writes the same
/// bytes; higher rungs trade wall-clock for peak memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GhostRung {
    /// Rung 0: one lane per worker (all lanes at once if fewer), inline
    /// trellis. Default (no budget: CLI, crate, tests).
    FullParallel,
    /// Rung 1: as many lanes at once as the budget allows (≥ 2), inline.
    CappedParallel,
    /// Rung 2: one lane at a time, inline trellis.
    SingleLane,
    /// Rung 3: one lane at a time, segmented trellis (back pointers
    /// recomputed per segment: `O(√n)` memory, twice the trellis work).
    Segmented,
}

impl GhostRung {
    /// Telemetry code (0..=3).
    pub fn as_u8(self) -> u8 {
        match self {
            GhostRung::FullParallel => 0,
            GhostRung::CappedParallel => 1,
            GhostRung::SingleLane => 2,
            GhostRung::Segmented => 3,
        }
    }
}

/// How a Ghost encode plans its lanes ([`ghost_lane_plan`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GhostLanePlan {
    pub rung: GhostRung,
    /// Lanes planned at the same time.
    pub parallel_lanes: usize,
    /// Back-pointer bytes one lane may keep (above: segmented trellis).
    pub lane_inline_bytes: usize,
    /// Most positions whose costs are held at once (`v31::EncodeOpts::group_positions`).
    pub group_positions: usize,
}

/// Ceiling of the Ghost encoder's own allocations when a budget is set (the
/// apps, WASM): with the app's own ~0.1–0.2 GB this keeps an encode under
/// 1 GB up to ~100 MP (4:2:0). Larger covers need more: the parsed JPEG and
/// the keyed order alone are ~7.2 B per pixel.
pub const GHOST_TARGET_BYTES: usize = 800 << 20;

/// Bytes of a Ghost encode that grow with the pixel count, excluding lanes
/// and cost groups: the parsed JPEG (Y i16 + 4:2:0 chroma, 3 B/px), the keyed
/// order (u32 per AC, ~4 B/px), the output JPEG (~0.25 B/px); side
/// information adds 1 B/px (+ the caller's RGB, 3 B/px), shadows ~2 B/px
/// (wet mask, region orders).
pub fn ghost_fixed_bytes(mode: ModeId, pixels: usize) -> usize {
    let quarter_bytes = match mode {
        ModeId::GhostNoShadow => 29,
        ModeId::GhostSi => 29 + 16,
        ModeId::GhostShadow => 29 + 8,
        ModeId::GhostSiShadow => 29 + 16 + 8,
        _ => 29,
    };
    pixels.saturating_mul(quarter_bytes) / 4 + FIXED_OVERHEAD_BYTES
}

/// Back-pointer table one lane may keep in memory (in-memory Viterbi), on
/// top of the share of the cover's in-flight working set
/// (`v31::lane_inline_share`): a full-length lane (< 2²¹ positions, 64 B each
/// at the wire's h = 9) always fits; longer lanes (short frames) keep what
/// their share allows and recompute the rest.
pub const LANE_INLINE_BYTES: usize = 128 << 20;

/// Rayon workers an encode can use (1 without the `parallel` feature).
pub fn encode_workers() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads().max(1)
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Static safety multiplier applied on top of the per-component sum.
/// ×1.5 chosen as a starting point; the `perf_memory_audit` harness
/// validates against real RSS.
pub(crate) const SAFETY_FACTOR_NUM: usize = 3;
pub(crate) const SAFETY_FACTOR_DEN: usize = 2;

/// Fixed overhead independent of image size: AbsDeltaTable (252 KB),
/// SyndromeMulTables (64 KB), allocator slack, JpegImage metadata,
/// stack frames. Conservative.
pub(crate) const FIXED_OVERHEAD_BYTES: usize = 4 * 1024 * 1024;

/// Predict peak RSS (in bytes) for an encode or decode of an image at
/// `width × height` in `mode` at `rung` with `n_workers` rayon workers.
///
/// Includes the ×1.5 safety factor and the fixed-overhead constant.
/// Returns an upper bound suitable for comparing against `budget`.
///
/// Coefficients trace to the cost table at
/// `docs/design/image/memory-budget-2026-05.md` § 1. They are static;
/// `perf_memory_audit` running on a real device may refine them.
pub fn predict_peak_memory(
    mode: ModeId,
    width: u32,
    height: u32,
    rung: GhostRung,
    n_workers: usize,
) -> usize {
    let pixels = (width as usize).saturating_mul(height as usize);
    let mut raw = if is_ghost_encode(mode) {
        ghost_fixed_bytes(mode, pixels)
    } else {
        pixels.saturating_mul(bytes_per_pixel(mode)).saturating_add(FIXED_OVERHEAD_BYTES)
    };
    if is_ghost_encode(mode) {
        // Lanes: a typical heavy payload (frame ≈ AC/16) on this cover.
        let n_ac = pixels / 64 * 63;
        let m = n_ac / 16;
        let l = crate::stego::ghost::v31::lanes(n_ac, m);
        let nk = n_ac.div_ceil(l);
        let (k, inline) = match rung {
            GhostRung::FullParallel => (n_workers.max(1).min(l), LANE_INLINE_BYTES),
            GhostRung::CappedParallel => (2.min(l), LANE_INLINE_BYTES),
            GhostRung::SingleLane => (1, LANE_INLINE_BYTES),
            GhostRung::Segmented => (1, 0),
        };
        let h = crate::stego::ghost::v31::V31_STC_H;
        let inline = inline.min(crate::stego::ghost::v31::lane_inline_share(n_ac, nk, k, h, n_workers));
        let threads = crate::stego::ghost::v31::lane_threads(k, n_workers);
        raw = raw.saturating_add(k.saturating_mul(crate::stego::ghost::v31::lane_bytes(nk, m.div_ceil(l), h, inline, threads)));
        // One cost group of the lanes in flight (8 B per position).
        raw = raw.saturating_add(k.saturating_mul(nk).saturating_mul(8));
    }
    raw.saturating_mul(SAFETY_FACTOR_NUM) / SAFETY_FACTOR_DEN
}

fn is_ghost_encode(mode: ModeId) -> bool {
    matches!(mode, ModeId::GhostNoShadow | ModeId::GhostShadow | ModeId::GhostSi | ModeId::GhostSiShadow)
}

/// Per-pixel byte budget for the dominant working buffers (Ghost encodes:
/// without the lanes, which [`predict_peak_memory`] adds). See
/// `docs/design/image/memory-budget-2026-05.md` § 1 and, for Ghost v3,
/// `docs/design/image/ghost-embedding-v3.md` "Memory".
fn bytes_per_pixel(mode: ModeId) -> usize {
    match mode {
        // Ghost v3: see `ghost_fixed_bytes` (rounded up); costs are gathered
        // per lane group, not held for the whole image.
        ModeId::GhostNoShadow => 8,
        ModeId::GhostSi => 12,
        ModeId::GhostShadow => 10,
        ModeId::GhostSiShadow => 14,
        ModeId::Armor => {
            // jpeg_to_luma_f64 (8 bytes/pixel) + Spectrum2D f64 complex
            // (16 bytes/pixel during FFT) + SectorLut (~0.3 bytes/pixel).
            // 1600 px ceiling enforced by frontend; CLI users on huge
            // inputs pay this in full.
            25
        }
        ModeId::Decode => {
            // Lighter than encode: no cost surface, no STC embed buffers.
            // The parsed JPEG (~3 B/px) + the largest mode attempt: Ghost
            // v1's nzAC order (~4 B/nzAC), the v1 cost-pool costs, or Armor
            // (u32 order + LLR cache; its geometry stage streams the
            // spectrum and the corrected image with a bounded working set).
            // Measured 6.8-8.1 B/px at 48-110 MP (wrong passphrase, every
            // mode tried).
            10
        }
        ModeId::Av1WholeVideoShadow => {
            // Video peak doesn't fit the per-pixel-only model: it scales
            // with n_frames + gop_size, not just width × height. Callers
            // should use `predict_peak_memory_av1_whole_video_shadow`
            // instead. Returning 0 here so a stray bytes_per_pixel call
            // doesn't silently fabricate a budget number.
            0
        }
    }
}

/// Predict peak RSS for an AV1 whole-video shadow encode under the
/// WV.6 streaming Pass 1 design.
///
/// Peak is bounded by:
///   - one GOP's raw YUV (`width × height × 1.5 × gop_size`) — the
///     `current_gop_yuv` working buffer; capacity preserved across
///     GOPs so it doesn't re-allocate
///   - `per_gop_natural` (~50 KB/frame for OBU bytes + recording)
///     across all `n_frames` frames
///   - `per_gop_harvests` (~60 KB/GOP for cover bits + costs vec)
///     across all GOPs
///   - cascade working set + growing output Vec (small; folded into
///     the safety factor)
///
/// Plus the same ×1.5 safety factor used elsewhere.
///
/// Coefficients are static upper bounds calibrated from observed
/// rav1e + dav1d encode output at 1080p; `perf_memory_audit` may
/// refine them.
pub fn predict_peak_memory_av1_whole_video_shadow(
    width: u32,
    height: u32,
    n_frames: u32,
    gop_size: u32,
) -> usize {
    let pixels = (width as usize).saturating_mul(height as usize);
    let frame_size = pixels.saturating_mul(3) / 2; // tight I420 = 1.5 bpp
    let gs = (gop_size.max(1)) as usize;
    let nf = n_frames as usize;
    let n_gops = nf.div_ceil(gs);

    // current_gop_yuv: peak-time = one full GOP of raw I420.
    let one_gop_yuv = frame_size.saturating_mul(gs);

    // per_gop_natural — OBU bytes + PhasmFrameRecording per frame.
    // ~25 KB OBU + ~50 KB recording = ~75 KB/frame at 1080p; the
    // 100 KB/frame upper bound here gives ~15% headroom.
    const NATURAL_BYTES_PER_FRAME: usize = 100 * 1024;
    let per_gop_natural = nf.saturating_mul(NATURAL_BYTES_PER_FRAME);

    // per_gop_harvests — cover_bits + costs vec per GOP. ~60 KB at
    // 1080p × gop_size=30 with ~5K cover positions/GOP.
    const HARVEST_BYTES_PER_GOP: usize = 100 * 1024;
    let per_gop_harvests = n_gops.saturating_mul(HARVEST_BYTES_PER_GOP);

    let raw = one_gop_yuv
        .saturating_add(per_gop_natural)
        .saturating_add(per_gop_harvests)
        .saturating_add(FIXED_OVERHEAD_BYTES);
    raw.saturating_mul(SAFETY_FACTOR_NUM) / SAFETY_FACTOR_DEN
}

// ===== Public memory-budget API =====

/// Process-wide memory budget in bytes, or `None` for "no limit".
///
/// Bridges set this at startup with the device's available memory / 2
/// (or hardcoded for WASM). The ladder rung selector reads this to
/// decide how aggressively to degrade. Default: `None` (no limit) —
/// what CLI gets when no bridge has called `set_memory_budget`.
///
/// Stored in a `Mutex<Option<usize>>` rather than `AtomicUsize` so we
/// can represent the unset state without a sentinel. Set frequency is
/// O(1 per process); read frequency is O(1 per encode) — Mutex cost is
/// negligible.
static MEMORY_BUDGET: OnceLock<Mutex<Option<usize>>> = OnceLock::new();

fn budget_cell() -> &'static Mutex<Option<usize>> {
    MEMORY_BUDGET.get_or_init(|| Mutex::new(None))
}

/// Set the process-wide memory budget. Pass `None` to disable.
///
/// Bridges should call this once at init. Re-setting is safe but
/// racy with concurrent encodes; do not call mid-encode.
pub fn set_memory_budget(bytes: Option<usize>) {
    *budget_cell().lock().unwrap() = bytes;
}

/// Get the current process-wide memory budget. `None` = no limit.
pub fn get_memory_budget() -> Option<usize> {
    *budget_cell().lock().unwrap()
}

/// Test-only: every lib test that sets the process-global budget or
/// telemetry hook holds this lock, so tests that assert on a plan under
/// their own budget don't run under another test's. (Encodes running
/// concurrently without it only change speed: their output does not depend
/// on the budget — `ghost::v31::tests::encode_bytes_independent_of_budget_and_workers`.)
#[cfg(test)]
pub(crate) fn budget_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static M: Mutex<()> = Mutex::new(());
    M.lock().unwrap_or_else(|p| p.into_inner())
}

/// Heap two `smart_decode` stages may hold at once, beyond the parsed
/// image, when the `parallel` build runs them side by side (lowered to a
/// quarter of the memory budget if one is set): on inputs above
/// `ARMOR_TARGET_DIMENSION`, the v1 cost pass + pools next to Armor.
/// Stages whose estimated peaks ([`DecodeStageHeap`]) add up to more run
/// one after another, so overlap never raises the decode's peak above
/// parse + this cap: typical photos up to about 12–14 MP overlap, a 48 MP cover
/// keeps its serial peak.
pub const DECODE_OVERLAP_BYTES: usize = 192 << 20;

/// Estimated peak heap (beyond the parsed image) of the `smart_decode`
/// stages that may overlap, for one cover. Upper bounds from the counting
/// allocator (`examples/mem_probe`, 12 and 48 MP, Mac, 2026-10-29); `n` =
/// Y coefficients (≈ pixels), `nz` = Y non-zero AC coefficients:
///
/// - the v1 cost order (`legacy_v1_cost_order`): 8 B per position kept
///   + the stable sort's half-length scratch + strip buffers → 20 `nz`;
/// - the v1 cost pools: the held order (8 B) + one-pass priorities of the
///   largest pool (8 B + 4 B per member, half the order) + bitmaps
///   → 14 `nz` + `n` / 4;
/// - Armor (Fortress, STDM, grid sync): the grid sync's luma and its
///   whitened copy (f32, 4 B per pixel each) dominate; the rest is ≤ 3.9 B
///   per coefficient; + one Argon2 (19 MiB) → 8 `n` + 20 MiB.
#[derive(Debug, Clone, Copy)]
pub struct DecodeStageHeap {
    /// The v1 cost order (cost pass + sort).
    pub legacy_order: usize,
    /// The v1 cost pools (order held).
    pub legacy_pools: usize,
    /// Fortress, Armor STDM and geometric recovery.
    pub armor: usize,
}

impl DecodeStageHeap {
    /// Estimates for a cover of `n` Y coefficients, `nz` of them non-zero AC.
    pub fn for_cover(n: usize, nz: usize) -> Self {
        Self {
            legacy_order: nz.saturating_mul(20),
            legacy_pools: nz.saturating_mul(14).saturating_add(n / 4),
            armor: n.saturating_mul(8).saturating_add(20 << 20),
        }
    }
}

/// Whether stages of estimated peaks `a` and `b` may run at once
/// ([`DECODE_OVERLAP_BYTES`], or a quarter of the budget if lower).
pub fn decode_overlap_ok(a: usize, b: usize) -> bool {
    let cap = get_memory_budget().map_or(DECODE_OVERLAP_BYTES, |m| (m / 4).min(DECODE_OVERLAP_BYTES));
    a.saturating_add(b) <= cap
}

/// Telemetry events emitted at key points in the encode/decode
/// lifecycle. Hook in your analytics via `set_telemetry_hook`.
///
/// All variants carry image dimensions and mode so a hook can
/// correlate. Memory predictions are in megabytes for human
/// readability (analytics dashboards rarely want raw byte counts).
#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    /// Emitted at the top of an encode, after the ladder rung is chosen.
    EncodeStarted {
        mode: ModeId,
        width: u32,
        height: u32,
        predicted_peak_mb: u32,
        budget_mb: Option<u32>,
        ladder_rung: u8,
    },
    /// Emitted on successful encode completion.
    EncodeCompleted {
        mode: ModeId,
        width: u32,
        height: u32,
        ladder_rung: u8,
        wall_clock_ms: u32,
    },
    /// Emitted on encode failure (any `StegoError` other than user-cancel).
    EncodeFailed {
        mode: ModeId,
        width: u32,
        height: u32,
        reason: &'static str,
    },
    /// Emitted when the rung selector picks a non-default rung.
    /// Useful for noticing "we're degrading on a 12 MP encode" — usually
    /// indicates the budget is too tight.
    LadderRungSelected {
        mode: ModeId,
        width: u32,
        height: u32,
        predicted_peak_mb: u32,
        budget_mb: Option<u32>,
        rung: u8,
    },
    /// Emitted when the user cancels mid-encode. `ladder_rung` records
    /// which rung was active — useful for "did this take long because
    /// of memory pressure or because the image is huge".
    EncodeCancelled {
        mode: ModeId,
        width: u32,
        height: u32,
        ladder_rung: u8,
        wall_clock_ms_when_cancelled: u32,
    },
}

/// Boxed telemetry hook type. Must be `Send + Sync` so the
/// hook can be invoked from any rayon worker.
pub type TelemetryHook = Box<dyn Fn(&TelemetryEvent) + Send + Sync + 'static>;

static TELEMETRY_HOOK: OnceLock<Mutex<Option<TelemetryHook>>> = OnceLock::new();

fn telemetry_cell() -> &'static Mutex<Option<TelemetryHook>> {
    TELEMETRY_HOOK.get_or_init(|| Mutex::new(None))
}

/// Register a telemetry hook. Pass `None` to unregister.
///
/// The hook fires on every `TelemetryEvent`. It must be cheap (runs on
/// the encode thread) and non-blocking — buffer events and process
/// them on a background thread if your analytics backend is slow.
pub fn set_telemetry_hook(hook: Option<TelemetryHook>) {
    *telemetry_cell().lock().unwrap() = hook;
}

/// Emit a telemetry event. Called from the encode pipeline.
///
/// If a hook is registered, it's invoked synchronously. Otherwise, in
/// debug builds, the event is logged to stderr; in release builds,
/// this is a no-op.
pub(crate) fn emit(event: TelemetryEvent) {
    let cell = telemetry_cell().lock().unwrap();
    if let Some(hook) = cell.as_ref() {
        hook(&event);
    } else {
        #[cfg(debug_assertions)]
        eprintln!("[phasm telemetry] {:?}", event);
    }
}

/// Plan a Ghost encode's lanes for a cover of `width × height` with `n_ac`
/// Y AC coefficients, a frame of `m_bits` and STC height `h`.
///
/// No budget (CLI, crate, tests) → [`GhostRung::FullParallel`], one cost
/// group, lanes in memory as far as the cover's in-flight working set
/// allows (`v31::lane_inline_share`, applied by the planner). With a budget,
/// the encoder's own allocations stay within
/// `min(budget / 1.5, GHOST_TARGET_BYTES)`: the fastest of — in-memory lanes
/// on every worker with one cost pass; the same with more cost passes
/// (smaller groups); checkpointed lanes; fewer lanes at once. Every choice
/// writes the same bytes.
pub fn ghost_lane_plan(mode: ModeId, width: u32, height: u32, n_ac: usize, m_bits: usize, h: usize) -> GhostLanePlan {
    use crate::stego::ghost::v31;
    let workers = encode_workers();
    let l = v31::lanes(n_ac, m_bits.max(1));
    let k_max = workers.min(l).max(1);
    let full = GhostLanePlan {
        rung: GhostRung::FullParallel,
        parallel_lanes: k_max,
        lane_inline_bytes: LANE_INLINE_BYTES,
        group_positions: usize::MAX,
    };
    let Some(budget) = get_memory_budget() else {
        return full;
    };
    let pixels = (width as usize).saturating_mul(height as usize);
    let limit = (budget.saturating_mul(SAFETY_FACTOR_DEN) / SAFETY_FACTOR_NUM).min(GHOST_TARGET_BYTES);
    let avail = limit.saturating_sub(ghost_fixed_bytes(mode, pixels));
    let nk = n_ac.div_ceil(l);
    let bits = m_bits.div_ceil(l);
    // One lane's working set with `k` lanes at once: its table as far as its
    // share allows (`inline(k)`, the plan's `lane_inline_bytes`), or
    // checkpointed; idle workers help recompute.
    let inline = |k: usize| LANE_INLINE_BYTES.min(v31::lane_inline_share(n_ac, nk, k, h, workers));
    let lane_in = |k: usize| v31::lane_bytes(nk, bits, h, inline(k), v31::lane_threads(k, workers));
    let lane_seg = |k: usize| v31::lane_bytes(nk, bits, h, 0, v31::lane_threads(k, workers));
    // Cost-group positions that fit, at least one lane's worth. A group is
    // gathered (8 B per position: sort index + costs, plus one cost strip),
    // then its lanes run `k` at a time next to its costs (4 B per position).
    let strip = crate::stego::cost::uniward::COST_STRIP_WORKING_BYTES;
    let group = |k: usize, lane: usize| -> Option<usize> {
        let lanes_rest = avail.checked_sub(k.saturating_mul(lane))?;
        let gather_rest = avail.checked_sub(strip)?;
        let g = (gather_rest / 8).min(lanes_rest / 4);
        (g >= nk).then_some(g.min(n_ac))
    };
    if let Some(g) = group(k_max, lane_in(k_max)) {
        let lane_inline_bytes = inline(k_max);
        if g >= n_ac {
            return GhostLanePlan { lane_inline_bytes, group_positions: usize::MAX, ..full };
        }
        // Prefer checkpointed lanes on every worker in one or two cost
        // passes over in-memory lanes with many more passes.
        if let Some(gs) = group(k_max, lane_seg(k_max))
            && n_ac.div_ceil(gs) + 1 < n_ac.div_ceil(g)
        {
            return GhostLanePlan { rung: GhostRung::Segmented, parallel_lanes: k_max, lane_inline_bytes: 0, group_positions: gs };
        }
        return GhostLanePlan { rung: GhostRung::FullParallel, parallel_lanes: k_max, lane_inline_bytes, group_positions: g };
    }
    if let Some(g) = group(k_max, lane_seg(k_max)) {
        return GhostLanePlan { rung: GhostRung::Segmented, parallel_lanes: k_max, lane_inline_bytes: 0, group_positions: g };
    }
    for k in (1..k_max).rev() {
        if let Some(g) = group(k, lane_in(k)) {
            let rung = if k > 1 { GhostRung::CappedParallel } else { GhostRung::SingleLane };
            return GhostLanePlan { rung, parallel_lanes: k, lane_inline_bytes: inline(k), group_positions: g };
        }
        if let Some(g) = group(k, lane_seg(k)) {
            return GhostLanePlan { rung: GhostRung::Segmented, parallel_lanes: k, lane_inline_bytes: 0, group_positions: g };
        }
    }
    // Over the target whatever we do (very large covers): the leanest plan.
    GhostLanePlan { rung: GhostRung::Segmented, parallel_lanes: 1, lane_inline_bytes: 0, group_positions: nk }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_grows_linearly_with_pixels() {
        let p1 = predict_peak_memory(ModeId::GhostShadow, 4000, 3000, GhostRung::SingleLane, 6);
        let p2 = predict_peak_memory(ModeId::GhostShadow, 8000, 6000, GhostRung::SingleLane, 6);
        // 4× pixels → between 2× and 5× memory (linear plus lanes, which do
        // not grow with the cover).
        assert!(p2 > p1 * 2, "p2={} should be > 2×p1={}", p2, p1 * 2);
        assert!(p2 < p1 * 5, "p2={} should be < 5×p1={}", p2, p1 * 5);
    }

    #[test]
    fn rung_ordering_monotone() {
        let (w, h) = (8000u32, 6000u32);
        let r0 = predict_peak_memory(ModeId::GhostShadow, w, h, GhostRung::FullParallel, 6);
        let r1 = predict_peak_memory(ModeId::GhostShadow, w, h, GhostRung::CappedParallel, 6);
        let r2 = predict_peak_memory(ModeId::GhostShadow, w, h, GhostRung::SingleLane, 6);
        let r3 = predict_peak_memory(ModeId::GhostShadow, w, h, GhostRung::Segmented, 6);
        assert!(r0 > r1, "rung 0 ({r0}) > rung 1 ({r1})");
        assert!(r1 > r2, "rung 1 ({r1}) > rung 2 ({r2})");
        assert!(r2 > r3, "rung 2 ({r2}) > rung 3 ({r3})");
    }

    /// 12 MP: ~0.1 GB of fixed buffers (parsed JPEG, keyed order); each
    /// in-memory lane adds ~80 MB at h = 9 (64 B of back pointers + 8 B of
    /// vectors per position), plus its cost group.
    #[test]
    fn predict_12mp_in_expected_range() {
        let base = predict_peak_memory(ModeId::GhostNoShadow, 4032, 3024, GhostRung::Segmented, 6) / (1 << 20);
        let full = predict_peak_memory(ModeId::GhostNoShadow, 4032, 3024, GhostRung::FullParallel, 6) / (1 << 20);
        assert!((120..=250).contains(&base), "12 MP segmented {base} MB");
        assert!(full > base + 6 * 60 && full < base + 6 * 160, "12 MP 6 lanes {full} MB (base {base})");
    }

    // ===== M2 API tests =====

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn budget_set_get_roundtrip() {
        let _g = budget_test_lock();
        set_memory_budget(None);
        assert_eq!(get_memory_budget(), None);
        set_memory_budget(Some(1_500_000_000));
        assert_eq!(get_memory_budget(), Some(1_500_000_000));
        set_memory_budget(None);
        assert_eq!(get_memory_budget(), None);
    }

    const N12: usize = 4032 / 8 * (3024 / 8) * 63;

    #[test]
    fn lane_plan_no_budget_is_full_parallel() {
        let _g = budget_test_lock();
        set_memory_budget(None);
        let p = ghost_lane_plan(ModeId::GhostShadow, 4032, 3024, N12, 700_000, 9);
        assert_eq!(p.rung, GhostRung::FullParallel);
        assert_eq!(p.lane_inline_bytes, LANE_INLINE_BYTES);
        assert_eq!(p.group_positions, usize::MAX);
    }

    /// Under a budget the plan fits `min(budget / 1.5, GHOST_TARGET_BYTES)`
    /// (fixed buffers + lanes + one cost group), for any cover size the
    /// target can hold; tiny budgets fall back to one checkpointed lane.
    #[test]
    fn lane_plan_fits_the_budget() {
        use crate::stego::ghost::v31;
        let _g = budget_test_lock();
        for budget in [400usize << 20, 1200 << 20, 1674 << 20, 3 << 30] {
            set_memory_budget(Some(budget));
            let limit = (budget * 2 / 3).min(GHOST_TARGET_BYTES);
            for (w, hgt) in [(4032u32, 3024u32), (8064, 6048), (11_552, 8_672)] {
                let n = (w as usize / 8) * (hgt as usize / 8) * 63;
                for m in [25_000usize, n / 40, n / 6] {
                    let p = ghost_lane_plan(ModeId::GhostNoShadow, w, hgt, n, m, 9);
                    let l = v31::lanes(n, m);
                    let nk = n.div_ceil(l);
                    // As the planner runs a batch of `parallel_lanes` lanes.
                    let workers = encode_workers();
                    let k = p.parallel_lanes.min(workers).min(l);
                    let inline = p.lane_inline_bytes.min(v31::lane_inline_share(n, nk, k, 9, workers));
                    let lane = v31::lane_bytes(nk, m.div_ceil(l), 9, inline, v31::lane_threads(k, workers));
                    // Gather phase (8 B per group position + a cost strip),
                    // then lanes next to the group's costs (4 B per position).
                    let g = p.group_positions.min(n);
                    let used = ghost_fixed_bytes(ModeId::GhostNoShadow, w as usize * hgt as usize)
                        + (g * 8 + crate::stego::cost::uniward::COST_STRIP_WORKING_BYTES)
                            .max(g * 4 + k * lane);
                    let fixed = ghost_fixed_bytes(ModeId::GhostNoShadow, w as usize * hgt as usize);
                    if fixed + lane_bytes_min(nk) < limit {
                        assert!(used <= limit, "budget {budget} {w}x{hgt} m={m}: {p:?} uses {used} > {limit}");
                    }
                    assert!(p.parallel_lanes >= 1 && p.group_positions >= nk.min(n), "{p:?}");
                }
            }
        }
        set_memory_budget(Some(1 << 20));
        let p = ghost_lane_plan(ModeId::GhostNoShadow, 4032, 3024, N12, 700_000, 9);
        assert_eq!((p.rung, p.parallel_lanes), (GhostRung::Segmented, 1));
        set_memory_budget(None);
    }

    /// A checkpointed lane plus a one-lane cost group (test helper bound).
    fn lane_bytes_min(nk: usize) -> usize {
        crate::stego::ghost::v31::lane_bytes(nk, nk / 8, 9, 0, 1) + nk * 8
    }

    #[test]
    fn lane_plan_generous_budget_keeps_full() {
        let _g = budget_test_lock();
        set_memory_budget(Some(8 << 30));
        let p = ghost_lane_plan(ModeId::GhostShadow, 4032, 3024, N12, 700_000, 9);
        assert_eq!(p.rung, GhostRung::FullParallel);
        set_memory_budget(None);
    }

    #[test]
    fn telemetry_hook_invoked_when_set() {
        let _g = budget_test_lock();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        set_telemetry_hook(Some(Box::new(move |_e| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })));
        emit(TelemetryEvent::EncodeStarted {
            mode: ModeId::GhostShadow,
            width: 100,
            height: 100,
            predicted_peak_mb: 50,
            budget_mb: Some(1000),
            ladder_rung: 0,
        });
        emit(TelemetryEvent::EncodeCompleted {
            mode: ModeId::GhostShadow,
            width: 100,
            height: 100,
            ladder_rung: 0,
            wall_clock_ms: 250,
        });
        set_telemetry_hook(None);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn telemetry_hook_unregistered_no_crash() {
        let _g = budget_test_lock();
        set_telemetry_hook(None);
        // Without a hook, emit() is a no-op in release / eprintln in debug.
        // Should never panic regardless.
        emit(TelemetryEvent::EncodeFailed {
            mode: ModeId::Armor,
            width: 200,
            height: 200,
            reason: "test",
        });
    }
}
