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
    /// Rung 4, the lowest: the keyed order (4 B per AC coefficient) is
    /// written to disk once generated and read back in lane order
    /// (`stego::spill`), with the most lanes and the largest cost groups the
    /// budget then allows. Only when no other rung fits and a spill
    /// directory with room is set ([`crate::stego::spill::set_spill_dir`]).
    SpilledOrder,
}

impl GhostRung {
    /// Telemetry code (0..=4).
    pub fn as_u8(self) -> u8 {
        match self {
            GhostRung::FullParallel => 0,
            GhostRung::CappedParallel => 1,
            GhostRung::SingleLane => 2,
            GhostRung::Segmented => 3,
            GhostRung::SpilledOrder => 4,
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

/// Most the Ghost encoder spends on its cost group and lanes — its working
/// set beyond the buffers every plan holds ([`ghost_fixed_bytes`]) — when a
/// budget is set (the apps, WASM): beyond it, more lanes at once and larger
/// in-memory trellis tables buy little speed. The budget bounds the whole
/// encode ([`ghost_lane_plan`]); this only keeps a generous budget from being
/// spent. A plan always holds one cost group, the whole cover's costs if it
/// fits.
pub const GHOST_TARGET_BYTES: usize = 800 << 20;

/// What a Ghost encode allocates, from its cover and layers: the input of
/// the memory model ([`ghost_fixed_bytes`], [`ghost_plan_peak`],
/// [`ghost_lane_plan`], [`ghost_encode_peak_bytes`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GhostEncodeShape {
    /// Coefficients of the cover's luma grid (blocks × 64, MCU padding included).
    pub y_coeffs: usize,
    /// Its width in blocks (the cost pass's strips).
    pub y_blocks_wide: usize,
    /// Coefficients of its other components (chroma).
    pub chroma_coeffs: usize,
    /// The encode has side information (`encode_si*`).
    pub si: bool,
    /// Frame bits of the primary message.
    pub m_bits: usize,
    /// Shadow messages.
    pub shadows: usize,
    /// Their RS-coded frame bits together.
    pub shadow_bits: usize,
    /// The largest shadow's RS-coded frame bits.
    pub max_shadow_bits: usize,
    /// Length of the cover file (the stego JPEG is written into a buffer of
    /// about that size: `JpegImage::write_reserve`).
    pub source_len: usize,
}

impl GhostEncodeShape {
    /// The shape of an encode into `img` (its own grids and file length) of a
    /// primary frame of `m_bits` and shadows of `shadow_bits` RS-coded bits
    /// each.
    pub fn of_cover(img: &crate::codec::jpeg::JpegImage, si: bool, m_bits: usize, shadow_bits: &[usize]) -> Self {
        let y = img.dct_grid(0);
        let y_coeffs = y.coeffs().len();
        let chroma_coeffs = (1..img.num_components()).map(|c| img.dct_grid(c).coeffs().len()).sum();
        Self {
            y_coeffs,
            y_blocks_wide: y.blocks_wide(),
            chroma_coeffs,
            si,
            m_bits,
            shadows: shadow_bits.len(),
            shadow_bits: shadow_bits.iter().sum(),
            max_shadow_bits: shadow_bits.iter().copied().max().unwrap_or(0),
            source_len: img.source_len(),
        }
    }

    /// A 4:2:0 cover of `width × height` (the Deep Cover writer's, and most
    /// camera JPEGs): 16-pixel MCUs, two chroma blocks per four luma blocks;
    /// a file of [`COVER_FILE_BYTES_PER_PIXEL`].
    pub fn yuv420(width: u32, height: u32, si: bool, m_bits: usize, shadow_bits: &[usize]) -> Self {
        let (mw, mh) = ((width as usize).div_ceil(16), (height as usize).div_ceil(16));
        Self {
            y_coeffs: mw * mh * 4 * 64,
            y_blocks_wide: mw * 2,
            chroma_coeffs: mw * mh * 2 * 64,
            si,
            m_bits,
            shadows: shadow_bits.len(),
            shadow_bits: shadow_bits.iter().sum(),
            max_shadow_bits: shadow_bits.iter().copied().max().unwrap_or(0),
            source_len: cover_file_bound(width, height),
        }
    }

    /// A cover of `width × height` of any chroma subsampling, the most it
    /// can hold: the luma of 16-pixel MCUs (4:2:0's padding), full-size
    /// chroma (4:4:4: twice the luma), a file of
    /// [`COVER_FILE_BYTES_PER_PIXEL`] ([`ghost_encode_peak_bytes`]: the apps
    /// pass camera JPEGs through and do not know their subsampling).
    pub fn any_jpeg(width: u32, height: u32, si: bool, m_bits: usize, shadow_bits: &[usize]) -> Self {
        let s = Self::yuv420(width, height, si, m_bits, shadow_bits);
        Self { chroma_coeffs: s.y_coeffs * 2, ..s }
    }

    /// Y AC coefficients: the positions of the keyed order.
    pub fn n_ac(&self) -> usize {
        self.y_coeffs / 64 * 63
    }

    /// The common shadow width at most: `n / (m + shadow bits)`
    /// (`pipeline::shadows_encode_body`).
    fn shadow_width(&self) -> usize {
        // The widest ladder width the layout tries first.
        let top = self.n_ac() / (self.m_bits + self.shadow_bits).max(1);
        crate::stego::ghost::v31::ladder().take_while(|&w| w <= top).last().unwrap_or(1)
    }

    /// Positions of the largest shadow region, at most: a shadow of `b` bits
    /// takes `b · w` positions at the common width `w`.
    pub fn max_shadow_positions(&self) -> usize {
        self.max_shadow_bits.saturating_mul(self.shadow_width()).min(self.n_ac())
    }
}

/// Bytes a Ghost encode holds through its embedding whatever the plan
/// (`docs/design/image/ghost-embedding-v3.md`, "Memory"): the cover's luma
/// (2 B per coefficient; the chroma is read again for the JPEG write, not
/// held), the side information (1 B per luma coefficient),
/// the keyed order (4 B per AC coefficient), the pair key and the wet mask
/// (1 bit per luma coefficient each), the frame (1 B per bit) and the
/// finished lanes' changes (8 B each, reserved exactly, at most one per two
/// frame bits: 4 B per frame bit) and the shadows' planned values; plus
/// [`FIXED_OVERHEAD_BYTES`]. The cost group and the lanes come on top
/// ([`ghost_plan_peak`]).
pub fn ghost_fixed_bytes(shape: &GhostEncodeShape) -> usize {
    ghost_cover_bytes(shape)
        .saturating_add(shape.n_ac().saturating_mul(4))
        .saturating_add(shape.y_coeffs / 8)
        .saturating_add(shape.m_bits.saturating_mul(5))
        .saturating_add(shadow_held_bytes(shape))
        .saturating_add(FIXED_OVERHEAD_BYTES)
}

/// Peak of the assembly after the lanes (`v31::assemble`): what
/// [`ghost_fixed_bytes`] counts, the lanes' changes and their copy in key
/// order (4 B per frame bit each); the costs and lanes are gone.
fn assemble_phase_bytes(shape: &GhostEncodeShape) -> usize {
    ghost_fixed_bytes(shape).saturating_add(shape.m_bits.saturating_mul(4))
}

/// Size of a cover file per pixel that [`GhostEncodeShape::yuv420`] and
/// [`GhostEncodeShape::any_jpeg`] assume (the stego file is written into a
/// buffer of about the cover's size): camera JPEGs and Deep Cover covers are
/// 0.2-1 B per pixel, a 4:4:4 Q92 noise image 0.8, a quality-100 photo up to
/// ~1.5. A larger cover file adds its excess (×1.125) at the write.
pub const COVER_FILE_BYTES_PER_PIXEL: f64 = 1.5;

fn cover_file_bound(width: u32, height: u32) -> usize {
    ((width as f64) * (height as f64) * COVER_FILE_BYTES_PER_PIXEL) as usize
}

/// Working set of the payload's Brotli compression (quality 11, 4 MiB
/// window: a 32 MiB block from ~1 KB of input, less below).
const BROTLI_WORKING_BYTES: usize = 33 << 20;

/// One Argon2id derivation (19 MiB of memory blocks): the frame key, the
/// structural keys, a shadow's verification.
const ARGON2_WORKING_BYTES: usize = 20 << 20;

/// Peak before the embedding, the payload in its forms (plaintext,
/// compressed, ciphertext, frame, 1 B per frame bit: ≤ 2 B per frame bit
/// together) throughout:
/// - plain (`ghost_encode*`): the payload's Brotli compression (and the
///   shadows' frames: Brotli, Argon2id) before the cover is parsed; then the
///   parsed cover (2 B per coefficient, chroma included) next to an
///   Argon2id derivation (the frame key, the structural key);
/// - side information (`encode_si*`, Deep Cover): the cover is parsed and
///   its side information computed first (1 B per luma coefficient;
///   approximate side information in strips of 50 block rows of f64 luma
///   blocks), and both are held through the compression and derivations.
fn prep_phase_bytes(shape: &GhostEncodeShape) -> usize {
    let grids = shape.y_coeffs.saturating_add(shape.chroma_coeffs).saturating_mul(2);
    let payload = shape.m_bits.saturating_add(shape.shadow_bits).saturating_mul(2);
    let held = if shape.si {
        let strips = shape.y_blocks_wide.saturating_mul(50 * 64 * 8);
        grids.saturating_add(shape.y_coeffs).saturating_add(BROTLI_WORKING_BYTES.max(ARGON2_WORKING_BYTES).max(strips))
    } else {
        BROTLI_WORKING_BYTES.max(grids.saturating_add(ARGON2_WORKING_BYTES))
    };
    held.saturating_add(payload).saturating_add(PREP_OVERHEAD_BYTES)
}

/// The small allocations next to [`prep_phase_bytes`]' (measured < 1 MiB).
const PREP_OVERHEAD_BYTES: usize = 1 << 20;

/// The cover's luma, the side information and the wet mask: what the
/// embedding holds of the cover (`pipeline::drop_chroma`).
fn ghost_cover_bytes(shape: &GhostEncodeShape) -> usize {
    let si = if shape.si { shape.y_coeffs } else { 0 };
    shape.y_coeffs.saturating_mul(2).saturating_add(si).saturating_add(shape.y_coeffs / 8)
}

/// Peak of the JPEG write after the embedding: a fresh parse of the cover
/// (2 B per coefficient, chroma included) next to the embedding's changes
/// (≤ 4 B per frame bit) and the shadows' values (`pipeline::stego_image`),
/// then next to the file's buffer (`JpegImage::write_reserve` of the cover
/// file's length).
fn write_phase_bytes(shape: &GhostEncodeShape) -> usize {
    let grids = shape.y_coeffs.saturating_add(shape.chroma_coeffs).saturating_mul(2);
    let changes = shape.m_bits.saturating_mul(4).saturating_add(shadow_held_bytes(shape));
    let file = crate::codec::jpeg::JpegImage::write_reserve(shape.source_len);
    grids.saturating_add(changes.max(file)).saturating_add(FIXED_OVERHEAD_BYTES)
}

/// What the shadow layers hold while the primary is planned: their planned
/// values (a hash map of 9 B per bucket, at most one change per two frame
/// bits, buckets ≤ 8/7 of a power of two: ≤ 12 B per frame bit) and their
/// regions' parities for the verification (1 bit per position). The regions
/// are in the wet mask, not held (`pipeline::shadows_encode_body`).
fn shadow_held_bytes(shape: &GhostEncodeShape) -> usize {
    let regions = shape.shadow_bits.saturating_mul(shape.shadow_width()).min(shape.n_ac());
    shape.shadow_bits.saturating_mul(12).saturating_add(regions / 8)
}

/// Peak of the shadows' verification after the primary is planned: the
/// cover, the primary's changes (4 B per frame bit), a region's parities
/// unpacked (1 B per position) and its frame key's Argon2id.
fn verify_phase_bytes(shape: &GhostEncodeShape) -> usize {
    if shape.shadows == 0 {
        return 0;
    }
    ghost_cover_bytes(shape)
        .saturating_add(shadow_held_bytes(shape))
        .saturating_add(shape.m_bits.saturating_mul(4))
        .saturating_add(shape.max_shadow_positions())
        .saturating_add(ARGON2_WORKING_BYTES)
        .saturating_add(FIXED_OVERHEAD_BYTES)
}

/// Peak of the shadow layers' planning (before the primary's), one shadow
/// at a time, for the largest region `r` (4 B per position): the order
/// prefix while it is built (`shadow::ac_hash_order_prefix`,
/// [`order_working_bytes`]);
/// the region's costs gathered (8 B per position next to a cost strip); the
/// costs (4 B), trellis input (5 B) and trellis working set; then the
/// stego bits (1 B) and the region's copy (4 B). Earlier shadows hold their
/// planned values and parities ([`shadow_held_bytes`]) and their regions
/// (4 B per position, kept for a retried layout until the primary is
/// planned).
fn shadow_phase_bytes(shape: &GhostEncodeShape, inline_limit: usize, h: usize, workers: usize) -> usize {
    if shape.shadows == 0 {
        return 0;
    }
    let r = shape.max_shadow_positions();
    let limit = inline_limit.min(crate::stego::ghost::v31::lane_inline_cap(shape.n_ac(), h));
    let trellis = crate::stego::stc::trellis::working_bytes(r, shape.shadow_width().max(1), h, limit, workers);
    let order = r.saturating_mul(4).saturating_add(order_working_bytes(r));
    let gather = r.saturating_mul(12).saturating_add(strip_bytes(shape));
    let plan = r.saturating_mul(14).saturating_add(trellis).max(r.saturating_mul(18));
    // The regions of the shadows planned before (kept for a retry).
    let earlier = shape.shadow_bits.saturating_sub(shape.max_shadow_bits).saturating_mul(shape.shadow_width()).min(shape.n_ac());
    ghost_cover_bytes(shape)
        .saturating_add(shadow_held_bytes(shape))
        .saturating_add(earlier.saturating_mul(4))
        .saturating_add(order.max(gather).max(plan))
        .saturating_add(FIXED_OVERHEAD_BYTES)
}

/// A cost pass's working set on this cover.
fn strip_bytes(shape: &GhostEncodeShape) -> usize {
    crate::stego::cost::uniward::cost_strip_working_bytes(shape.y_blocks_wide)
}

/// The shadow hash order's working set while `r` positions of it are built
/// or read again (`shadow::ac_hash_order_visit`): the candidates of a group
/// (16 B each, at most 8 M, in per-chunk vectors reserved with room for
/// their spread: ≤ 17 B) and the histograms (the total and a round of 32
/// per-chunk ones).
fn order_working_bytes(r: usize) -> usize {
    r.min(8 << 20).saturating_mul(17).saturating_add(9 << 20)
}

/// Peak heap of a Ghost encode of `shape` planned as `plan` with `workers`
/// rayon workers: the most of the shadow planning and the primary's
/// embedding — [`ghost_fixed_bytes`], the cost group (the whole cover's
/// costs, 4 B per luma coefficient, when there is one group; else a group's
/// costs, 8 B per position while gathered next to a cost strip and 4 B
/// while its lanes run) and the lanes planned at once
/// ([`crate::stego::ghost::v31::lane_bytes`] each). Everything the encode
/// allocates is in it; the caller's buffers (the cover JPEG, the side
/// information's pixels) are not. Mirrors `v31::plan`'s grouping and
/// batching exactly; an upper bound (measured: `ghost-embedding-v3.md`,
/// "Memory").
pub fn ghost_plan_peak(shape: &GhostEncodeShape, plan: &GhostLanePlan, h: usize, workers: usize) -> usize {
    use crate::stego::ghost::v31;
    let n = shape.n_ac();
    let m = shape.m_bits.max(1);
    let l = v31::lanes(n, m);
    let workers = workers.max(1);
    let groups = ghost_groups(n, l, plan.group_positions);
    let nk = n.div_ceil(l);
    let bits = m.div_ceil(l);
    let lanes = |lanes_in_group: usize| -> usize {
        let nb = plan.parallel_lanes.min(workers).clamp(1, lanes_in_group);
        let inline = plan.lane_inline_bytes.min(v31::lane_inline_share(n, nk, nb, h, workers));
        nb.saturating_mul(v31::lane_bytes(nk, bits, h, inline, v31::lane_threads(nb, workers)))
    };
    let strip = strip_bytes(shape);
    // The lowest rung: the order is on disk during the costs, the lanes and
    // the assembly; a group's slice is read back for its gather (4 B per
    // position), each lane in flight reads it a window at a time.
    let spilled = plan.rung == GhostRung::SpilledOrder;
    let order = if spilled { shape.n_ac().saturating_mul(4) } else { 0 };
    let windows = if spilled { plan.parallel_lanes.min(workers).max(1).saturating_mul(SPILL_WINDOW_BYTES) } else { 0 };
    let costs_and_lanes = if groups.len() == 1 {
        shape.y_coeffs.saturating_mul(4).saturating_add(strip.max(lanes(l)))
    } else {
        let read = |g: usize| if spilled { g.saturating_mul(4) } else { 0 };
        groups
            .iter()
            .map(|&(g, kg)| (g.saturating_mul(8).saturating_add(strip).saturating_add(read(g))).max(g.saturating_mul(4).saturating_add(lanes(kg))))
            .max()
            .unwrap_or(0)
    };
    let primary = ghost_fixed_bytes(shape).saturating_sub(order).saturating_add(costs_and_lanes).saturating_add(windows);
    // Generating the order (in memory) and writing it, before the costs.
    let spill_write = if spilled { ghost_fixed_bytes(shape).saturating_add(SPILL_WRITE_BYTES) } else { 0 };
    let peak = primary
        .max(spill_write)
        .max(prep_phase_bytes(shape))
        .max(assemble_phase_bytes(shape).saturating_sub(order).saturating_add(windows))
        .max(shadow_phase_bytes(shape, plan.lane_inline_bytes, h, workers))
        .max(verify_phase_bytes(shape))
        .max(write_phase_bytes(shape));
    peak.saturating_add(peak / 100 * GHOST_MODEL_MARGIN_PERCENT)
}

/// The spill's write buffer and the samples' flat indices next to the order
/// while it is written (`stego::spill::OrderSpill::write`).
const SPILL_WRITE_BYTES: usize = 2 << 20;

/// A lane's window of the order on disk (`v31::FiWindow`, the trellis
/// chunks' reads are on the stack).
const SPILL_WINDOW_BYTES: usize = 64 << 10;

/// Margin on the modelled peak for what the component sums leave out (small
/// allocations, vector growth, the order in which concurrent lanes allocate):
/// measured peaks came within 0.0-0.7 % of the sums at 12-150 MP.
const GHOST_MODEL_MARGIN_PERCENT: usize = 3;

/// Peak heap (bytes) a Ghost encode allocates for a `width × height` cover,
/// with or without side information (`si`: the `encode_si*` calls), with
/// `shadow_count` shadow messages, for messages of `payload_bytes` together
/// (primary and shadows, compressed payload as `compressed_payload_size`
/// reports it): the plan [`ghost_lane_plan`] picks under the current memory
/// budget, the process's rayon workers and spill directory (with one set and
/// memory tight, the lowest rung's plan: the keyed order on disk,
/// [`GhostRung::SpilledOrder`]), and its [`ghost_plan_peak`].
///
/// For an app to check memory before an encode. It counts everything core
/// allocates during the call — the parse and side information, the
/// payload's compression and key derivations, keyed order, costs, lanes,
/// shadows, the stego JPEG — for a cover whose file is at most
/// [`COVER_FILE_BYTES_PER_PIXEL`]: with `si` a Deep Cover cover (core's
/// writer: 4:2:0, [`GhostEncodeShape::yuv420`]), else a JPEG of any chroma
/// subsampling ([`GhostEncodeShape::any_jpeg`]: full-size chroma, which only
/// the parse and the write hold).
/// It does not count the caller's own buffers: the cover JPEG passed in and,
/// for `si`, the RGB pixels (3 B per pixel), which the caller holds during
/// the call and adds. The split of `payload_bytes` between the layers is not
/// known here; the estimate is the largest over the splits auto-sort allows
/// (the primary carries at least an equal share). An upper bound within
/// about 10 % of the measured peak (`ghost-embedding-v3.md`, "Memory").
pub fn ghost_encode_peak_bytes(width: u32, height: u32, si: bool, shadow_count: usize, payload_bytes: usize) -> usize {
    use crate::stego::frame::{FRAME_OVERHEAD, FRAME_OVERHEAD_EXT};
    let h = crate::stego::ghost::v31::V31_STC_H;
    let workers = encode_workers();
    let frame_bits = |bytes: usize| (bytes + if bytes > u16::MAX as usize { FRAME_OVERHEAD_EXT } else { FRAME_OVERHEAD }) * 8;
    let peak = |primary: usize, shadow_total: usize| {
        // The shadows' bits together, and the largest shadow's: it carries at
        // most the primary's payload (auto-sort) and at most all of theirs.
        let mut bits = vec![0usize; shadow_count];
        if shadow_count > 0 {
            let total = crate::stego::ghost::capacity::shadow_frames_bits(shadow_count, shadow_total);
            let largest = crate::stego::ghost::capacity::shadow_frames_bits(1, primary.min(shadow_total)).min(total);
            bits[0] = largest;
            let rest = total - largest;
            for (k, b) in bits.iter_mut().enumerate().skip(1) {
                *b = rest / (shadow_count - 1) + usize::from(k <= rest % (shadow_count - 1));
            }
        }
        let shape = if si {
            GhostEncodeShape::yuv420(width, height, si, frame_bits(primary), &bits)
        } else {
            GhostEncodeShape::any_jpeg(width, height, si, frame_bits(primary), &bits)
        };
        let plan = ghost_lane_plan(&shape, h);
        ghost_plan_peak(&shape, &plan, h, workers)
    };
    if shadow_count == 0 {
        return peak(payload_bytes, 0);
    }
    // The primary is the largest layer: from an equal share up to all of it.
    let equal = payload_bytes / (shadow_count + 1);
    [equal, payload_bytes / 2, payload_bytes]
        .into_iter()
        .filter(|&p| p >= equal)
        .map(|p| peak(p, payload_bytes - p))
        .max()
        .unwrap_or(0)
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
/// Ghost encodes: [`ghost_plan_peak`] of a typical heavy payload (frame ≈
/// AC / 16, two 1 KB shadows in the shadow modes) on a 4:2:0 cover, with
/// the lanes the rung plans at once and one cost group; truthful, no safety
/// factor (an encode's own plan and peak: [`ghost_lane_plan`],
/// [`ghost_plan_peak`]). Other modes: the per-pixel table with the ×1.5
/// safety factor and the fixed-overhead constant
/// (`docs/design/image/memory-budget-2026-05.md` § 1).
pub fn predict_peak_memory(
    mode: ModeId,
    width: u32,
    height: u32,
    rung: GhostRung,
    n_workers: usize,
) -> usize {
    if is_ghost_encode(mode) {
        let si = matches!(mode, ModeId::GhostSi | ModeId::GhostSiShadow);
        let shadows = if matches!(mode, ModeId::GhostShadow | ModeId::GhostSiShadow) { 2 } else { 0 };
        let one = crate::stego::ghost::capacity::shadow_frames_bits(1, 1024);
        let probe = GhostEncodeShape::yuv420(width, height, si, 0, &vec![one; shadows]);
        let shape = GhostEncodeShape { m_bits: probe.n_ac() / 16, ..probe };
        let n = shape.n_ac();
        let l = crate::stego::ghost::v31::lanes(n, shape.m_bits.max(1));
        let (k, inline, group) = match rung {
            GhostRung::FullParallel => (n_workers.max(1), LANE_INLINE_BYTES, usize::MAX),
            GhostRung::CappedParallel => (2, LANE_INLINE_BYTES, usize::MAX),
            GhostRung::SingleLane => (1, LANE_INLINE_BYTES, usize::MAX),
            GhostRung::Segmented | GhostRung::SpilledOrder => (1, 0, n.div_ceil(l)),
        };
        let plan = GhostLanePlan { rung, parallel_lanes: k, lane_inline_bytes: inline, group_positions: group };
        return ghost_plan_peak(&shape, &plan, crate::stego::ghost::v31::V31_STC_H, n_workers);
    }
    let pixels = (width as usize).saturating_mul(height as usize);
    let raw = pixels.saturating_mul(bytes_per_pixel(mode)).saturating_add(FIXED_OVERHEAD_BYTES);
    raw.saturating_mul(SAFETY_FACTOR_NUM) / SAFETY_FACTOR_DEN
}

fn is_ghost_encode(mode: ModeId) -> bool {
    matches!(mode, ModeId::GhostNoShadow | ModeId::GhostShadow | ModeId::GhostSi | ModeId::GhostSiShadow)
}

/// Per-pixel byte budget for the dominant working buffers of the modes
/// other than Ghost encodes (those: [`ghost_plan_peak`]). See
/// `docs/design/image/memory-budget-2026-05.md` § 1.
fn bytes_per_pixel(mode: ModeId) -> usize {
    match mode {
        ModeId::GhostNoShadow | ModeId::GhostSi | ModeId::GhostShadow | ModeId::GhostSiShadow => 0,
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

/// What a `smart_decode` of a JPEG holds, from its header and one count
/// ([`crate::codec::jpeg::count::coefficient_shape`]), for
/// [`decode_peak_for`].
#[derive(Debug, Clone, Copy)]
pub struct DecodeShape {
    /// Image size in pixels.
    pub width: u32,
    pub height: u32,
    /// Coefficients of every component's MCU-padded grid (the parse: 2 B
    /// each).
    pub coefficients: usize,
    /// Coefficients of the luma's padded grid (`n`).
    pub luma_coefficients: usize,
    /// Blocks per row of the luma's padded grid (the cost strips' width).
    pub luma_blocks_wide: usize,
    /// The luma's non-zero AC coefficients (`nz`).
    pub luma_nonzero_ac: usize,
}

impl DecodeShape {
    /// The shape of the JPEG `image_bytes` (one Huffman pass, no parse), or
    /// `None` if it does not read as a JPEG the parse accepts. A frame too
    /// large to count (over 16384 × 16384 px, or scans that do not decode
    /// after a valid header) is bounded from its header: every luma AC
    /// coefficient non-zero.
    pub fn of_jpeg(image_bytes: &[u8]) -> Option<Self> {
        use crate::codec::jpeg::count;
        let s = count::coefficient_shape(image_bytes).or_else(|_| count::frame_shape(image_bytes)).ok()?;
        Some(Self {
            width: s.frame.width as u32,
            height: s.frame.height as u32,
            coefficients: s.coefficients,
            luma_coefficients: s.luma_coefficients,
            luma_blocks_wide: s.frame.blocks_wide(0),
            luma_nonzero_ac: s.luma_nonzero_ac,
        })
    }

    fn luma_ac(&self) -> usize {
        self.luma_coefficients / 64 * 63
    }
}

/// Peak heap of `smart_decode` on `image_bytes` over every attempt it may
/// make (a wrong passphrase, or an image without a message, tries them
/// all), beyond the caller's copy of the file: an upper bound for the apps'
/// memory check before a decode. Reads the JPEG's header and counts its luma
/// non-zero AC coefficients in one Huffman pass (about a parse's time,
/// luma coefficients / 8 bytes of memory); a file that does not read as a
/// JPEG fails the decode at once (1 MiB).
///
/// Uses the process's memory budget ([`set_memory_budget`]) as the decode
/// will: it decides which attempts run side by side. See
/// [`decode_peak_for`] for the attempts and their working sets.
pub fn decode_peak_bytes(image_bytes: &[u8]) -> usize {
    match DecodeShape::of_jpeg(image_bytes) {
        Some(shape) => decode_peak_for(&shape),
        None => 1 << 20,
    }
}

/// Working set of the v1 cost order's strip pass per pixel column of the
/// luma (`cost::uniward_legacy_v1::compute_positions_streaming_legacy_v1`):
/// a 50-block-row strip's pixels and its two row-filtered copies (≤ 432 rows
/// each) next to the three wavelet subbands (≤ 415 rows), f32 —
/// (3·432 + 3·415)·4 = 10 164 B (measured 10.4 KB at 12 MP, 10.2 KB at 60).
const LEGACY_STRIP_BYTES_PER_COLUMN: usize = 10 << 10;

/// Armor's working set beside its two luma planes: the grid sync's whitening
/// ring, positions and frames (measured 1-9 MB at 12-150 MP).
const ARMOR_EXTRA_BYTES: usize = 16 << 20;

/// Armor's working set beside its luma planes when the rescaled decode
/// runs (a long side under 1600 · 1.1 px: `armor::pipeline::try_rescaled_decode`
/// upscales to 1600 / 1280 — the source luma (f64), the resampled planes and
/// grid of a cover up to 1600 × 1600: ≤ 8 B per source pixel + 46 MB).
const ARMOR_RESCALE_EXTRA_BYTES: usize = 48 << 20;

/// Margin on the decode estimate (as [`GHOST_MODEL_MARGIN_PERCENT`]).
const DECODE_MODEL_MARGIN_PERCENT: usize = 3;

/// Small allocations of a decode next to its large ones (marker segments,
/// Huffman tables, keys, frames).
const DECODE_OVERHEAD_BYTES: usize = 2 << 20;

/// [`decode_peak_bytes`] for a [`DecodeShape`]: the parse (2 B per
/// coefficient, every component) plus the largest of the attempts after it,
/// in `smart_decode`'s order (`stego::smart_decode_inner`, parallel build;
/// the serial build runs the same attempts one at a time, never more). Per
/// attempt, beyond the parse (`n` luma coefficients, `nz` of them non-zero
/// AC; measured on the 12-150 MP samples, `ghost-embedding-v3.md`, "Decode
/// memory"):
///
/// - Ghost primary: an Argon2id key, then the keyed order of the luma AC
///   (4 B per position) with the pair key (its keystream and words, 1/4 B
///   per coefficient) and the three bit planes (3/8 B per position); then
///   v1's non-zero AC set (4 B each, reserved exactly) and its parities
///   (1 B), the widths read after the set is dropped → max(Argon2id,
///   4.4 B per AC + 1/4 B per coefficient, 5 B per non-zero AC);
/// - Ghost shadows (v3 / v3.1, layout v1's legacy tier): an Argon2id key,
///   then hash-order prefixes (≤ 8 M candidates of 17 B, 9 MiB of
///   histograms) and their parities (1 B per position; a shadow's region can
///   be every AC position);
/// - the v1 cost pools (decode-only): the cost pass (8 B per non-zero AC
///   kept beside a strip of [`LEGACY_STRIP_BYTES_PER_COLUMN`] per pixel
///   column), the stable sort (+4 B), then the pools (the order and an
///   Argon2id key, then the largest pool's priorities: 6 B per non-zero AC,
///   1/4 B per coefficient of maps);
/// - Armor (Fortress, v2 / legacy STDM, the grid sync, the rescaled decode):
///   the grid sync's luma and its whitened copy (f32: 8 B per coefficient)
///   and [`ARMOR_EXTRA_BYTES`] ([`ARMOR_RESCALE_EXTRA_BYTES`] when the
///   rescaled decode upscales); its two keys are derived before, at once
///   (`KeyScope::prederive`).
///
/// Images up to `ARMOR_TARGET_DIMENSION` run Fortress (a key and its frames:
/// Argon2id + 1 B per coefficient at most), the Armor chain and Ghost
/// (primary, then shadows) side by side, then the v1 cost pools alone;
/// larger ones Ghost primary alone, the shadows alone, then the v1 cost
/// pools and Armor side by side if [`decode_overlap_ok`] (the budget) allows
/// it, else one after the other. A correct passphrase stops at its mode: this
/// is the most any input takes.
pub fn decode_peak_for(shape: &DecodeShape) -> usize {
    let n = shape.luma_coefficients;
    let n_ac = shape.luma_ac();
    let nz = shape.luma_nonzero_ac;
    let parse = shape.coefficients.saturating_mul(2);

    let primary = ARGON2_WORKING_BYTES
        .max(n_ac.saturating_mul(4).saturating_add(n_ac * 3 / 8).saturating_add(n / 4))
        .max(nz.saturating_mul(5));
    let shadows = ARGON2_WORKING_BYTES.max(order_working_bytes(n_ac).saturating_add(n_ac));
    let strip = shape.luma_blocks_wide.saturating_mul(8).saturating_mul(LEGACY_STRIP_BYTES_PER_COLUMN);
    let order = nz.saturating_mul(8);
    let legacy = order
        .saturating_add(strip)
        .max(nz.saturating_mul(12))
        .max(order.saturating_add(ARGON2_WORKING_BYTES.max(nz.saturating_mul(6).saturating_add(n / 4))));
    let long_side = shape.width.max(shape.height);
    let rescales = (long_side as u64) * 10 < 1600 * 11;
    let armor = (2 * ARGON2_WORKING_BYTES)
        .max(n.saturating_mul(8).saturating_add(if rescales { ARMOR_RESCALE_EXTRA_BYTES } else { ARMOR_EXTRA_BYTES }));
    let heap = DecodeStageHeap::for_cover(n, nz);

    let after_parse = if long_side <= crate::stego::ARMOR_TARGET_DIMENSION {
        let fortress = ARGON2_WORKING_BYTES.saturating_add(n);
        fortress.saturating_add(armor).saturating_add(primary.max(shadows)).max(legacy)
    } else {
        let tail = if decode_overlap_ok(heap.legacy_order.max(heap.legacy_pools), heap.armor) {
            legacy.saturating_add(armor)
        } else {
            legacy.max(armor)
        };
        primary.max(shadows).max(tail)
    };
    let peak = parse.saturating_add(after_parse).saturating_add(DECODE_OVERHEAD_BYTES);
    peak.saturating_add(peak * DECODE_MODEL_MARGIN_PERCENT / 100)
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

/// Plan a Ghost encode's lanes for an encode of `shape` at STC height `h`.
///
/// No budget (CLI, crate, tests) → [`GhostRung::FullParallel`], one cost
/// group, lanes in memory as far as the cover's in-flight working set
/// allows (`v31::lane_inline_share`, applied by the planner). With a budget,
/// the encode's peak ([`ghost_plan_peak`], everything it allocates) stays
/// within `min(budget / 1.5, fixed + GHOST_TARGET_BYTES)` (fixed:
/// [`ghost_fixed_bytes`]): the fastest of — in-memory lanes on every worker
/// with one cost pass; the same with more cost passes (smaller groups);
/// checkpointed lanes; fewer lanes at once. When even one checkpointed lane
/// per cost group is over the budget's own bound (`budget / 1.5`): the same
/// search with the keyed order on disk ([`GhostRung::SpilledOrder`], if a
/// spill directory with room for it is set:
/// [`crate::stego::spill::spill_room`]), else its leanest plan. Otherwise the
/// leanest in-memory plan (its peak is over the limit:
/// [`ghost_encode_peak_bytes`] tells the apps before they start). Every
/// choice writes the same bytes.
///
/// `PHASM_GHOST_FORCE_SPILL` (measurement, the identity runs) keeps the plan
/// but puts its order on disk whenever a spill directory has room.
pub fn ghost_lane_plan(shape: &GhostEncodeShape, h: usize) -> GhostLanePlan {
    let plan = ghost_lane_plan_inner(shape, h);
    if plan.rung != GhostRung::SpilledOrder
        && std::env::var_os("PHASM_GHOST_FORCE_SPILL").is_some()
        && crate::stego::spill::spill_room(spill_bytes(shape))
    {
        return GhostLanePlan { rung: GhostRung::SpilledOrder, ..plan };
    }
    plan
}

/// Size of the spill file of an encode of `shape`: its keyed order.
fn spill_bytes(shape: &GhostEncodeShape) -> u64 {
    shape.n_ac() as u64 * 4
}

fn ghost_lane_plan_inner(shape: &GhostEncodeShape, h: usize) -> GhostLanePlan {
    use crate::stego::ghost::v31;
    let workers = encode_workers();
    let n = shape.n_ac();
    let l = v31::lanes(n, shape.m_bits.max(1));
    let k_max = workers.min(l).max(1);
    let nk = n.div_ceil(l);
    let inline = |k: usize| LANE_INLINE_BYTES.min(v31::lane_inline_share(n, nk, k, h, workers));
    // `spill`: the same plan with the order on disk (the lowest rung).
    let in_memory = |k: usize, g: usize, spill: bool| GhostLanePlan {
        rung: if spill {
            GhostRung::SpilledOrder
        } else if k == k_max {
            GhostRung::FullParallel
        } else if k > 1 {
            GhostRung::CappedParallel
        } else {
            GhostRung::SingleLane
        },
        parallel_lanes: k,
        lane_inline_bytes: inline(k),
        group_positions: g,
    };
    let segmented = |k: usize, g: usize, spill: bool| GhostLanePlan {
        rung: if spill { GhostRung::SpilledOrder } else { GhostRung::Segmented },
        parallel_lanes: k,
        lane_inline_bytes: 0,
        group_positions: g,
    };
    let Some(budget) = get_memory_budget() else {
        return GhostLanePlan { lane_inline_bytes: LANE_INLINE_BYTES, ..in_memory(k_max, usize::MAX, false) };
    };
    // In u64: on wasm32 `budget * 2` saturates usize above 2 GiB.
    let limit = ((budget as u64 * SAFETY_FACTOR_DEN as u64 / SAFETY_FACTOR_NUM as u64).min(usize::MAX as u64) as usize)
        .min(ghost_fixed_bytes(shape).saturating_add(GHOST_TARGET_BYTES));
    let fits = |p: &GhostLanePlan| ghost_plan_peak(shape, p, h, workers) <= limit;
    // The largest cost group (positions; `usize::MAX`: the whole cover) with
    // which `plan(g)` fits, at least one lane's worth. Within groups smaller
    // than the cover the peak grows with the group; one group holds the
    // dense costs instead (4 B per coefficient), so it is tried first.
    let largest_group = |plan: &dyn Fn(usize) -> GhostLanePlan| -> Option<usize> {
        if fits(&plan(usize::MAX)) {
            return Some(usize::MAX);
        }
        if n <= nk || !fits(&plan(nk)) {
            return None;
        }
        let (mut lo, mut hi) = (nk, n - 1); // fits(lo); find the largest
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if fits(&plan(mid)) { lo = mid } else { hi = mid - 1 }
        }
        Some(lo)
    };
    let passes = |g: usize| ghost_groups(n, l, g).len();
    // The fastest plan that fits, the order in memory or (`spill`) on disk.
    let search = |spill: bool| -> Option<GhostLanePlan> {
        if let Some(g) = largest_group(&|g| in_memory(k_max, g, spill)) {
            // Prefer checkpointed lanes on every worker in one or two cost
            // passes over in-memory lanes with many more passes.
            if g != usize::MAX
                && let Some(gs) = largest_group(&|g| segmented(k_max, g, spill))
                && passes(gs) + 1 < passes(g)
            {
                return Some(segmented(k_max, gs, spill));
            }
            return Some(in_memory(k_max, g, spill));
        }
        if let Some(g) = largest_group(&|g| segmented(k_max, g, spill)) {
            return Some(segmented(k_max, g, spill));
        }
        for k in (1..k_max).rev() {
            if let Some(g) = largest_group(&|g| in_memory(k, g, spill)) {
                return Some(in_memory(k, g, spill));
            }
            if let Some(g) = largest_group(&|g| segmented(k, g, spill)) {
                return Some(segmented(k, g, spill));
            }
        }
        None
    };
    if let Some(p) = search(false) {
        return p;
    }
    // Nothing fits `limit`. The order goes to disk only when memory is short
    // — the leanest in-memory plan is over the budget's own bound — not when
    // only `GHOST_TARGET_BYTES` capped the limit (a large cover with a
    // generous budget: the leanest in-memory plan, as before).
    let leanest = segmented(1, nk, false);
    // In u64, as `limit` (wasm32's usize saturates `budget * 2` above 2 GiB).
    let budget_limit = (budget as u64 * SAFETY_FACTOR_DEN as u64 / SAFETY_FACTOR_NUM as u64).min(usize::MAX as u64) as usize;
    let short = ghost_plan_peak(shape, &leanest, h, workers) > budget_limit;
    if short && crate::stego::spill::spill_room(spill_bytes(shape)) {
        return search(true).unwrap_or(segmented(1, nk, true));
    }
    // Over the limit whatever we do (very large covers): the leanest plan.
    leanest
}

/// The cost groups `v31::plan` forms for `l` lanes over `n` positions with
/// groups of at most `group_positions` (at least one lane each):
/// `(positions, lanes)` per group.
pub(crate) fn ghost_groups(n: usize, l: usize, group_positions: usize) -> Vec<(usize, usize)> {
    use crate::stego::ghost::v31::{lane_groups, share};
    lane_groups(n, l, group_positions)
        .into_iter()
        .map(|ks| (share(n, l, ks.end - 1).end - share(n, l, ks.start).start, ks.len()))
        .collect()
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

    /// 12 MP: ~0.1 GB of fixed buffers (luma, keyed order) and a one-lane
    /// cost group when segmented; each in-memory lane adds ~70 MB at h = 9
    /// (64 B of back pointers per position), plus the whole cover's costs.
    #[test]
    fn predict_12mp_in_expected_range() {
        let base = predict_peak_memory(ModeId::GhostNoShadow, 4032, 3024, GhostRung::Segmented, 6) / (1 << 20);
        let full = predict_peak_memory(ModeId::GhostNoShadow, 4032, 3024, GhostRung::FullParallel, 6) / (1 << 20);
        assert!((80..=160).contains(&base), "12 MP segmented {base} MB");
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

    /// A 12 MP 4:2:0 cover, plain, no shadows, a frame of `m` bits.
    fn shape12(m: usize) -> GhostEncodeShape {
        GhostEncodeShape::yuv420(4032, 3024, false, m, &[])
    }

    #[test]
    fn lane_plan_no_budget_is_full_parallel() {
        let _g = budget_test_lock();
        set_memory_budget(None);
        let p = ghost_lane_plan(&shape12(700_000), 9);
        assert_eq!(p.rung, GhostRung::FullParallel);
        assert_eq!(p.lane_inline_bytes, LANE_INLINE_BYTES);
        assert_eq!(p.group_positions, usize::MAX);
    }

    /// Under a budget the plan's peak fits `min(budget / 1.5, fixed +
    /// GHOST_TARGET_BYTES)` whenever the leanest plan does (else it is the
    /// leanest plan), for plain, side-information and shadow encodes of
    /// 12-150 MP; tiny budgets fall back to one checkpointed lane.
    #[test]
    fn lane_plan_fits_the_budget() {
        use crate::stego::ghost::v31;
        let _g = budget_test_lock();
        let workers = encode_workers();
        for budget in [400usize << 20, 1200 << 20, 1674 << 20, 3 << 30] {
            set_memory_budget(Some(budget));
            for (w, hgt) in [(4032u32, 3024u32), (8064, 6048), (11_552, 8_672), (14_976, 9_984)] {
                for (si, shadows) in [(false, &[][..]), (true, &[][..]), (true, &[2000usize, 2000][..])] {
                    let n = GhostEncodeShape::yuv420(w, hgt, si, 0, &[]).n_ac();
                    for m in [800usize, 25_000, n / 40, n / 6] {
                        let shape = GhostEncodeShape::yuv420(w, hgt, si, m, shadows);
                        let p = ghost_lane_plan(&shape, 9);
                        let limit = (budget * 2 / 3).min(ghost_fixed_bytes(&shape) + GHOST_TARGET_BYTES);
                        let l = v31::lanes(n, m);
                        let nk = n.div_ceil(l);
                        let leanest = GhostLanePlan { rung: GhostRung::Segmented, parallel_lanes: 1, lane_inline_bytes: 0, group_positions: nk };
                        let peak = ghost_plan_peak(&shape, &p, 9, workers);
                        if ghost_plan_peak(&shape, &leanest, 9, workers) <= limit {
                            assert!(peak <= limit, "budget {budget} {w}x{hgt} si {si} m={m}: {p:?} peaks at {peak} > {limit}");
                        } else {
                            assert_eq!(p, leanest, "budget {budget} {w}x{hgt} si {si} m={m}");
                        }
                        assert!(p.parallel_lanes >= 1 && p.group_positions >= nk.min(n), "{p:?}");
                    }
                }
            }
        }
        set_memory_budget(Some(1 << 20));
        let p = ghost_lane_plan(&shape12(700_000), 9);
        assert_eq!((p.rung, p.parallel_lanes), (GhostRung::Segmented, 1));
        set_memory_budget(None);
    }

    /// The public estimate is the plan's peak; more payload, side information
    /// or shadows never estimate less; it grows with the cover.
    #[test]
    fn encode_peak_bytes_is_the_plan_peak() {
        let _g = budget_test_lock();
        for budget in [None, Some(1500usize << 20)] {
            set_memory_budget(budget);
            let (w, h) = (9504u32, 6312u32);
            let shape = GhostEncodeShape::yuv420(w, h, true, (1000 + crate::stego::frame::FRAME_OVERHEAD) * 8, &[]);
            let plan = ghost_lane_plan(&shape, 9);
            assert_eq!(ghost_encode_peak_bytes(w, h, true, 0, 1000), ghost_plan_peak(&shape, &plan, 9, encode_workers()));
            let si = ghost_encode_peak_bytes(w, h, true, 0, 1000);
            let plain = ghost_encode_peak_bytes(w, h, false, 0, 1000);
            let shadows = ghost_encode_peak_bytes(w, h, true, 2, 1200);
            assert!(si > plain && shadows > si, "{plain} {si} {shadows}");
            assert!(ghost_encode_peak_bytes(14_976, 9_984, true, 0, 1000) > si);
        }
        set_memory_budget(None);
    }

    #[test]
    fn lane_plan_generous_budget_keeps_full() {
        let _g = budget_test_lock();
        set_memory_budget(Some(8 << 30));
        let p = ghost_lane_plan(&GhostEncodeShape::yuv420(4032, 3024, false, 700_000, &[2000, 2000]), 9);
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

    /// The decode estimate: a file that is not a JPEG fails the decode at
    /// once; a JPEG's estimate holds its parse and grows with the cover.
    #[test]
    fn decode_peak_bytes_bounds_the_parse_and_grows() {
        assert_eq!(decode_peak_bytes(b"not a jpeg"), 1 << 20);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-vectors/image");
        let small = std::fs::read(dir.join("photo_320x240_q75_420.jpg")).unwrap();
        let large = std::fs::read(dir.join("photo_640x480_q75_420.jpg")).unwrap();
        for bytes in [&small, &large] {
            let shape = DecodeShape::of_jpeg(bytes).unwrap();
            let img = crate::codec::jpeg::JpegImage::from_bytes(bytes).unwrap();
            assert_eq!(shape.luma_nonzero_ac, crate::stego::ghost::capacity::count_nonzero_ac(img.dct_grid(0)));
            assert!(decode_peak_bytes(bytes) > shape.coefficients * 2 + ARGON2_WORKING_BYTES);
        }
        assert!(decode_peak_bytes(&large) > decode_peak_bytes(&small));
        // A header claiming 65535 × 65535 px: bounded from the header (every
        // luma AC non-zero), without counting it (no 512 MiB bitmap).
        let mut huge = small.clone();
        let sof = huge.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        huge[sof + 5..sof + 9].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
        assert!(crate::codec::jpeg::count::coefficient_shape(&huge).is_err());
        let shape = DecodeShape::of_jpeg(&huge).unwrap();
        assert_eq!((shape.width, shape.height), (65535, 65535));
        assert_eq!(shape.luma_nonzero_ac, shape.luma_ac());
        assert!(decode_peak_bytes(&huge) >= shape.coefficients.saturating_mul(2));
        // A 150 MP Deep Cover cover (14144 x 10608, 4:2:0, 30 % of the AC
        // non-zero): Armor's luma planes after the parse, under 12 B/px.
        let (w, h) = (14144u32, 10608u32);
        let n = (w as usize / 8) * (h as usize / 8) * 64;
        let shape = DecodeShape {
            width: w,
            height: h,
            coefficients: n * 3 / 2,
            luma_coefficients: n,
            luma_blocks_wide: w as usize / 8,
            luma_nonzero_ac: n / 64 * 63 * 3 / 10,
        };
        let per_px = decode_peak_for(&shape) as f64 / (w as f64 * h as f64);
        assert!((11.0..13.0).contains(&per_px), "{per_px} B/px");
    }

    /// The lowest rung (the order on disk) is chosen only when no other rung
    /// fits the budget and a spill directory has room for the order; it
    /// lowers the estimate by about the order (4 B per AC coefficient).
    #[test]
    fn spill_rung_only_when_nothing_else_fits() {
        use crate::stego::spill::{self, tests::plenty};
        let _guard = budget_test_lock();
        fn tight_disk(_: &std::path::Path) -> Option<u64> {
            Some(256 << 20)
        }
        let before = get_memory_budget();
        let (w, h) = (14_144u32, 10_608u32);
        let dir = spill::tests::test_dir("ladder");
        let rung_and_peak = |budget_mb: usize| {
            set_memory_budget(Some(budget_mb << 20));
            let shape = GhostEncodeShape::yuv420(w, h, true, 648, &[]);
            (ghost_lane_plan(&shape, 9).rung, ghost_encode_peak_bytes(w, h, true, 0, 31))
        };
        // No spill directory: today's ladder at any budget.
        spill::set_spill_dir(None, plenty);
        let (roomy, roomy_peak) = rung_and_peak(4096);
        let (tight, tight_peak) = rung_and_peak(900);
        assert_ne!(roomy, GhostRung::SpilledOrder);
        assert_eq!(tight, GhostRung::Segmented);
        // With one: a budget that fits keeps its plan and estimate.
        spill::set_spill_dir(Some(dir.clone()), plenty);
        assert_eq!(rung_and_peak(4096), (roomy, roomy_peak));
        let (rung, peak) = rung_and_peak(900);
        if cfg!(unix) {
            assert_eq!(rung, GhostRung::SpilledOrder);
            let order = (w as usize / 8) * (h as usize / 8) * 63 * 4;
            assert!(peak + order / 2 < tight_peak && peak + order + order / 10 > tight_peak, "{peak} vs {tight_peak}");
        } else {
            assert_eq!(rung, GhostRung::Segmented);
        }
        // Too little free space for the order: not available.
        spill::set_spill_dir(Some(dir), tight_disk);
        assert_eq!(rung_and_peak(900), (tight, tight_peak));
        spill::set_spill_dir(None, plenty);
        set_memory_budget(before);
    }

    /// The apps' header bound (`ImageMemory.decodeBoundBytes`, Android and iOS;
    /// `docs/design/app/large-samples.md`, "The decode check"), from a frame's
    /// grids alone: 2 B per coefficient + 14.1 B per luma coefficient + 10 KiB
    /// per pixel column (the luma's blocks per row × 8) + 96 MiB, + 10 %.
    fn app_decode_bound(shape: &DecodeShape) -> usize {
        let b = 2 * shape.coefficients
            + shape.luma_coefficients * 141 / 10
            + (10 << 10) * shape.luma_blocks_wide * 8
            + (96 << 20);
        b + b / 10
    }

    /// SOI, a frame (`marker`, `w` × `h`, `sampling` per component), a scan
    /// without its Huffman tables, EOI: the count fails after a valid header,
    /// so `decode_peak_bytes` bounds the frame as every luma AC non-zero.
    fn worst_case_header(w: u16, h: u16, sampling: &[(u8, u8)], marker: u8) -> Vec<u8> {
        let mut f = vec![0xFF, 0xD8, 0xFF, marker, 0, 8 + 3 * sampling.len() as u8, 8];
        f.extend_from_slice(&h.to_be_bytes());
        f.extend_from_slice(&w.to_be_bytes());
        f.push(sampling.len() as u8);
        for (i, (sh, sv)) in sampling.iter().enumerate() {
            f.extend_from_slice(&[i as u8 + 1, sh << 4 | sv, 0]);
        }
        f.extend_from_slice(&[0xFF, 0xDA, 0, 8, 1, 1, 0, 0, 63, 0, 0, 0, 0xFF, 0xD9]);
        f
    }

    /// A contract with the apps: their decode check skips core's estimate
    /// when their header bound ([`app_decode_bound`]) fits, so no decode
    /// estimate may exceed it, for any frame a header can declare (sampling
    /// layouts with 1-5 components; 1-65535 px a side, the ≤ 1600 px cases
    /// where Fortress, Armor and Ghost run side by side among them), any
    /// share of non-zero AC coefficients and any budget (it decides which
    /// stages overlap, [`DECODE_OVERLAP_BYTES`]), and every test-vector JPEG.
    /// A change to the decode model that breaks it must change the apps'
    /// formula (both apps, the same numbers) in the same step.
    #[test]
    fn decode_peak_stays_under_the_apps_header_bound() {
        let _g = budget_test_lock();
        let before = get_memory_budget();
        let layouts: &[&[(u8, u8)]] = &[
            &[(1, 1)],
            &[(2, 2)],
            &[(4, 4)],
            &[(4, 1)],
            &[(2, 2), (1, 1), (1, 1)],
            &[(2, 1), (1, 1), (1, 1)],
            &[(1, 1), (1, 1), (1, 1)],
            &[(1, 1), (2, 2), (2, 2)],
            &[(4, 4), (1, 1), (1, 1)],
            &[(3, 3), (1, 2), (2, 1)],
            &[(1, 1), (1, 1), (1, 1), (1, 1)],
            &[(1, 1), (4, 4), (4, 4), (4, 4)],
            &[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
        ];
        let sides: [u16; 20] =
            [1, 8, 13, 640, 1197, 1599, 1600, 1601, 1759, 1760, 2000, 4032, 9504, 14144, 16383, 16384, 16385, 30000, 40000, 65535];
        let budgets = [None, Some(16 << 20), Some(64 << 20), Some(512 << 20), Some(1 << 30), Some(4 << 30)];
        let mut worst = 0.0f64;
        for layout in layouts {
            for &w in &sides {
                for &h in &sides {
                    for marker in [0xC0u8, 0xC2] {
                        if marker == 0xC2 && (w as usize * 7 + h as usize) % 5 != 0 {
                            continue; // progressive on a fifth of the sizes
                        }
                        let file = worst_case_header(w, h, layout, marker);
                        let shape = DecodeShape::of_jpeg(&file).expect("a valid header");
                        assert_eq!(shape.luma_nonzero_ac, shape.luma_ac(), "{w}x{h} {layout:?}: bounded from the header");
                        let bound = app_decode_bound(&shape);
                        for budget in budgets {
                            set_memory_budget(budget);
                            let peak = decode_peak_bytes(&file);
                            assert!(peak <= bound, "{w}x{h} {layout:?} budget {budget:?}: {peak} > {bound}");
                            worst = worst.max(peak as f64 / bound as f64);
                            // every share of non-zero AC a real file of this frame can have
                            for tenth in 0..10 {
                                let s = DecodeShape { luma_nonzero_ac: shape.luma_ac() / 10 * tenth, ..shape };
                                let peak = decode_peak_for(&s);
                                assert!(peak <= bound, "{w}x{h} {layout:?} budget {budget:?} nz {tenth}/10: {peak} > {bound}");
                                worst = worst.max(peak as f64 / bound as f64);
                            }
                        }
                    }
                }
            }
        }
        // real files: core counts their non-zero AC
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-vectors/image");
        let mut files = Vec::new();
        let mut dirs = vec![dir];
        while let Some(d) = dirs.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    dirs.push(p);
                } else if p.extension().is_some_and(|x| x == "jpg") {
                    files.push(p);
                }
            }
        }
        assert!(files.len() >= 10, "{} test-vector JPEGs", files.len());
        for budget in budgets {
            set_memory_budget(budget);
            for p in &files {
                let bytes = std::fs::read(p).unwrap();
                let Some(shape) = DecodeShape::of_jpeg(&bytes) else { continue };
                let peak = decode_peak_bytes(&bytes);
                assert!(peak <= app_decode_bound(&shape), "{}: {peak}", p.display());
            }
        }
        set_memory_budget(before);
        eprintln!("decode estimate at most {worst:.3} of the apps' header bound");
        assert!(worst < 0.95, "{worst:.3}: the bound's margin is under 5 %");
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
