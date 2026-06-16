// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Memory budget prediction + telemetry.
//!
//! Provides peak-RSS prediction (`predict_peak_memory()`, the `ModeId` /
//! `GhostShadowRung` enums) and the public API every bridge calls at
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

/// Ladder rung selection for Ghost shadow encode. Rung 0 = default
/// behavior (fast, parallel); higher rungs trade wall-clock for peak
/// memory. The rung selection logic lives in `shadows_encode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GhostShadowRung {
    /// Rung 0: full parallel, cover_wavelets cache on. Default.
    FullParallel,
    /// Rung 1: cap rayon workers to 1, cover_wavelets cache on.
    CappedParallel,
    /// Rung 2: cap rayon workers to 1, drop cover_wavelets cache
    /// (re-stream positions per cascade iter; ~6× slower per iter).
    StreamingNoCache,
    /// Rung 3: rung 2 + drop early original_y/cascade_positions clones.
    MinimalClones,
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
    rung: GhostShadowRung,
    n_workers: usize,
) -> usize {
    let pixels = (width as usize).saturating_mul(height as usize);
    let bpp_base = bytes_per_pixel(mode, rung, n_workers);
    let raw = pixels.saturating_mul(bpp_base).saturating_add(FIXED_OVERHEAD_BYTES);
    raw.saturating_mul(SAFETY_FACTOR_NUM) / SAFETY_FACTOR_DEN
}

/// Per-pixel byte budget for the dominant working buffers, indexed by
/// mode and (where applicable) shadow ladder rung. See cost table at
/// `docs/design/image/memory-budget-2026-05.md` § 1.
fn bytes_per_pixel(mode: ModeId, rung: GhostShadowRung, n_workers: usize) -> usize {
    let n = n_workers.max(1);
    match mode {
        ModeId::GhostNoShadow | ModeId::GhostSi => {
            // luma f64 buffer + cost vec + positions Vec at peak.
            // ~13 bytes/pixel observed at 12 MP in pre-shadow tests.
            13
        }
        ModeId::GhostShadow | ModeId::GhostSiShadow => match rung {
            // Rung 0: 12 (cover_wavelets) + 6 (cover_verify_positions)
            //         + (12 + 6) × n_workers (per-worker stego_wavelets + img clones).
            GhostShadowRung::FullParallel => 18 + 18 * n,
            // Rung 1: same caches, single worker (no per-worker fanout).
            GhostShadowRung::CappedParallel => 18 + 18,
            // Rung 2: drop cover_wavelets + cover_verify_positions caches.
            // Verify + cascade re-stream positions per iter (~6× slower).
            GhostShadowRung::StreamingNoCache => 18,
            // Rung 3: same as rung 2 for now. Future: skip
            // cascade_positions clone (~4 bpp saved at 60 MP) by
            // switching shadow::rebuild_shadow to indices. Predict
            // matches rung 2 so the ladder doesn't promise headroom
            // we haven't shipped yet.
            GhostShadowRung::MinimalClones => 18,
        },
        ModeId::Armor => {
            // jpeg_to_luma_f64 (8 bytes/pixel) + Spectrum2D f64 complex
            // (16 bytes/pixel during FFT) + SectorLut (~0.3 bytes/pixel).
            // 1600 px ceiling enforced by frontend; CLI users on huge
            // inputs pay this in full.
            25
        }
        ModeId::Decode => {
            // Lighter than encode: no cost surface, no STC embed buffers.
            // luma f64 + positions Vec + small scratch.
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

/// Helper: select a Ghost shadow ladder rung given image dimensions
/// and the current budget. Returns `FullParallel` when budget is
/// `None` (no limit), else descends until predicted ≤ budget. Worst
/// case (predicted > budget at every rung) returns `MinimalClones`.
///
/// Called once per encode. Kept in this module so the
/// prediction coefficients and the selection logic live together.
pub fn select_ghost_shadow_rung(
    mode: ModeId,
    width: u32,
    height: u32,
    n_workers: usize,
) -> GhostShadowRung {
    let budget = match get_memory_budget() {
        Some(b) => b,
        None => return GhostShadowRung::FullParallel,
    };
    for rung in [
        GhostShadowRung::FullParallel,
        GhostShadowRung::CappedParallel,
        GhostShadowRung::StreamingNoCache,
        GhostShadowRung::MinimalClones,
    ] {
        if predict_peak_memory(mode, width, height, rung, n_workers) <= budget {
            return rung;
        }
    }
    GhostShadowRung::MinimalClones
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_grows_linearly_with_pixels() {
        let p1 = predict_peak_memory(
            ModeId::GhostShadow, 4000, 3000,
            GhostShadowRung::FullParallel, 6,
        );
        let p2 = predict_peak_memory(
            ModeId::GhostShadow, 8000, 6000,
            GhostShadowRung::FullParallel, 6,
        );
        // 4× pixels → between 3× and 5× memory (linear plus fixed overhead).
        assert!(p2 > p1 * 3, "p2={} should be > 3×p1={}", p2, p1 * 3);
        assert!(p2 < p1 * 5, "p2={} should be < 5×p1={}", p2, p1 * 5);
    }

    #[test]
    fn rung_ordering_monotone() {
        let w = 8000u32;
        let h = 6000u32;
        let r0 = predict_peak_memory(ModeId::GhostShadow, w, h, GhostShadowRung::FullParallel, 6);
        let r1 = predict_peak_memory(ModeId::GhostShadow, w, h, GhostShadowRung::CappedParallel, 6);
        let r2 = predict_peak_memory(ModeId::GhostShadow, w, h, GhostShadowRung::StreamingNoCache, 6);
        let r3 = predict_peak_memory(ModeId::GhostShadow, w, h, GhostShadowRung::MinimalClones, 6);
        assert!(r0 > r1, "rung 0 ({}) > rung 1 ({})", r0, r1);
        assert!(r1 > r2, "rung 1 ({}) > rung 2 ({})", r1, r2);
        // Rung 3 currently predicts same as rung 2 — the
        // cascade_positions skip is a planned follow-on. Track via the
        // explicit follow-up task #669 (filed in M4).
        assert!(r2 >= r3, "rung 2 ({}) >= rung 3 ({})", r2, r3);
    }

    #[test]
    fn predict_60mp_shadow_in_expected_range() {
        // 60 MP (60_000_000 pixels) Ghost shadow at rung 0 with 6 workers.
        // Cost table: 18 + 18×6 = 126 bytes/pixel × 60 MP = 7.56 GB raw.
        // ×1.5 safety = 11.3 GB. This is the predicted peak the ladder
        // tries to avoid by stepping down rungs.
        let peak = predict_peak_memory(
            ModeId::GhostShadow, 10000, 6000,
            GhostShadowRung::FullParallel, 6,
        );
        // 60 MP, 126 bpp, ×1.5 ≈ 11.3 GB. Allow ±10% for fixed overhead.
        let expected_mb = (60_000_000usize * 126 * 3 / 2) / (1024 * 1024);
        let actual_mb = peak / (1024 * 1024);
        assert!(
            actual_mb >= (expected_mb * 9 / 10) && actual_mb <= (expected_mb * 11 / 10),
            "60 MP rung-0 peak {} MB, expected ~{} MB",
            actual_mb, expected_mb,
        );
    }

    // ===== M2 API tests =====

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Each API test takes this lock so concurrent test runners don't
    /// trip over the process-global budget + hook state.
    fn api_lock() -> std::sync::MutexGuard<'static, ()> {
        static M: OnceLock<Mutex<()>> = OnceLock::new();
        M.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn budget_set_get_roundtrip() {
        let _g = api_lock();
        set_memory_budget(None);
        assert_eq!(get_memory_budget(), None);
        set_memory_budget(Some(1_500_000_000));
        assert_eq!(get_memory_budget(), Some(1_500_000_000));
        set_memory_budget(None);
        assert_eq!(get_memory_budget(), None);
    }

    #[test]
    fn select_rung_no_budget_returns_full_parallel() {
        let _g = api_lock();
        set_memory_budget(None);
        let r = select_ghost_shadow_rung(ModeId::GhostShadow, 10000, 6000, 6);
        assert_eq!(r, GhostShadowRung::FullParallel);
    }

    #[test]
    fn select_rung_tight_budget_descends() {
        let _g = api_lock();
        // 60 MP rung 0 needs ~11 GB; rung 3 needs ~0.85 GB.
        // Pass a 1 GB budget — must pick rung 3.
        set_memory_budget(Some(1_073_741_824));
        let r = select_ghost_shadow_rung(ModeId::GhostShadow, 10000, 6000, 6);
        assert_eq!(r, GhostShadowRung::MinimalClones);
        set_memory_budget(None);
    }

    #[test]
    fn select_rung_generous_budget_keeps_full() {
        let _g = api_lock();
        // 12 MP rung 0 needs ~2.3 GB at 6 workers. Pass 8 GB.
        set_memory_budget(Some(8 * 1024 * 1024 * 1024));
        let r = select_ghost_shadow_rung(ModeId::GhostShadow, 4000, 3000, 6);
        assert_eq!(r, GhostShadowRung::FullParallel);
        set_memory_budget(None);
    }

    #[test]
    fn telemetry_hook_invoked_when_set() {
        let _g = api_lock();
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
        let _g = api_lock();
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
