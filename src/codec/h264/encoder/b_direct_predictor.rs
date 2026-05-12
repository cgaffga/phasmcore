// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! §6E-A6.1 — Spatial-direct MV derivation (encoder side, mirrors
//! what the decoder will compute at the same MB position).
//!
//! Spec reference: ISO/IEC 14496-10
//! - § 8.4.1.2.1 "Derivation process for direct mode prediction"
//! - § 8.4.1.2.2 "Derivation process for luma motion vectors for
//!   B_Skip, B_Direct_16x16 and B_Direct_8x8"
//! - § 8.4.1.3 "Derivation process for luma motion vector
//!   prediction" (the median predictor)
//!
//! Phase 6E-A4 locked `direct_spatial_mv_pred_flag = 1` (matches
//! mobile-encoder defaults; cleaner than temporal direct). This
//! module implements ONLY the spatial-direct path; temporal direct
//! is out of scope.
//!
//! ## Scope of this module
//!
//! What it does:
//! - Derives per-MB `(refIdxL0, refIdxL1)` from neighbour A / B / C
//!   per spec § 8.4.1.2.1 (MIN_POSITIVE of available neighbours,
//!   per list).
//! - Both-`-1` boundary case: forces both refs to 0 with zero MVs.
//! - For each list X with `refIdxLX >= 0`: derives `mvLX` via the
//!   standard median predictor over neighbours (per § 8.4.1.3).
//!
//! What it does NOT do (yet):
//! - Co-located 4x4 sub-MB static / moving check per spec § 8.4.1.2.3.
//!   That check requires access to the L1 reference's MV grid +
//!   mb_type info to determine if the co-located block is "zero
//!   MV" (static) and can be reset to (0,0). For typical M=2 IBPBP
//!   content most B_Direct MBs are in static regions where the
//!   median path also yields (0,0) — so the divergence is rare in
//!   practice. Tracked as a follow-up task for §6E-A6.1 part 3.
//!
//! ## Why this matters for §6E-A6.1
//!
//! When the encoder commits a B_Direct / B_Skip MB, it MUST populate
//! `EncoderMvGrid` with the same MVs the decoder will derive at that
//! MB. Otherwise subsequent non-direct B-MBs (`B_L0_16x16` / etc)
//! that look up B_Direct neighbours for MVD prediction will see
//! REF_IDX_NONE (= "intra, contribute 0 to median") on the encoder
//! side, while the decoder sees real derived MVs and uses them in
//! its median. The MVDs encoded against an "encoder predicted 0"
//! will reconstruct against a "decoder predicted real" → drift.
//!
//! See `docs/design/video/h264/encoder-algorithms/6E-A6-bslice-partitions.md`
//! § 6E-A6.0 deliverable 5 for the architectural rationale.

use super::motion_estimation::MotionVector;
use super::partition_state::{predict_mv_for_partition, EncoderMvGrid, REF_IDX_NONE};
use super::reference_buffer::ColocatedMvGrid;

/// §B-cascade-real Phase 2.7 (#272) — per-MB spatial/temporal direct
/// trace record. Captures the full input state of the derivation +
/// the computed MVs so a separate analysis pass can correlate
/// max-deviation MBs (from `enc.visual_recon` vs reference-decoder output)
/// with their derivation inputs and identify where phasm diverges
/// from spec § 8.4.1.2.2 / § 8.4.1.2.3.
///
/// Gated by env var `PHASM_B_SPATIAL_DIRECT_TRACE`. Drained via
/// [`drain_spatial_direct_traces`].
#[derive(Debug, Clone, Copy)]
pub struct SpatialDirectTrace {
    pub frame_idx: u32,
    pub mb_x: u16,
    pub mb_y: u16,
    /// 0 = spatial-direct, 1 = temporal-direct.
    pub kind: u8,
    pub ref_idx_l0: i8,
    pub ref_idx_l1: i8,
    // Neighbour A (left) state per list.
    pub a_l0_mv_x: i16,
    pub a_l0_mv_y: i16,
    pub a_l0_ref: i8,
    pub a_l1_mv_x: i16,
    pub a_l1_mv_y: i16,
    pub a_l1_ref: i8,
    // Neighbour B (above).
    pub b_l0_mv_x: i16,
    pub b_l0_mv_y: i16,
    pub b_l0_ref: i8,
    pub b_l1_mv_x: i16,
    pub b_l1_mv_y: i16,
    pub b_l1_ref: i8,
    // Neighbour C (above-right with D fallback).
    pub c_l0_mv_x: i16,
    pub c_l0_mv_y: i16,
    pub c_l0_ref: i8,
    pub c_l1_mv_x: i16,
    pub c_l1_mv_y: i16,
    pub c_l1_ref: i8,
    /// Whether C was the C-position (true) or fell back to D (false).
    pub c_was_c: bool,
    // Colocated cell at (mb_x, mb_y) in L1 reference's motion grid.
    pub col_ref_idx_l0: i8,
    pub col_mv_l0_x: i16,
    pub col_mv_l0_y: i16,
    pub col_zero_flag: bool,
    /// True when the colMb grid was supplied; false → check skipped.
    pub col_grid_present: bool,
    // Output MVs.
    pub out_mv_l0_x: i16,
    pub out_mv_l0_y: i16,
    pub out_mv_l1_x: i16,
    pub out_mv_l1_y: i16,
}

static SPATIAL_DIRECT_TRACES: std::sync::OnceLock<std::sync::Mutex<Vec<SpatialDirectTrace>>> =
    std::sync::OnceLock::new();

fn spatial_direct_traces() -> &'static std::sync::Mutex<Vec<SpatialDirectTrace>> {
    SPATIAL_DIRECT_TRACES.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

/// Phase 2.7 — drain accumulated spatial/temporal direct traces.
pub fn drain_spatial_direct_traces() -> Vec<SpatialDirectTrace> {
    let mut guard = spatial_direct_traces().lock().unwrap();
    std::mem::take(&mut *guard)
}

fn push_spatial_direct_trace(trace: SpatialDirectTrace) {
    if !super::mb_decision_b::env_var_os_is_some("PHASM_B_SPATIAL_DIRECT_TRACE") {
        return;
    }
    if let Ok(mut guard) = spatial_direct_traces().lock() {
        guard.push(trace);
    }
}

/// Helper: read the current B-frame's display index from the
/// instrumentation atomic. Falls back to 0 if not set.
fn current_frame_idx() -> u32 {
    super::mb_decision_b::B_INSTRUMENT_FRAME_IDX.load(std::sync::atomic::Ordering::Relaxed)
}

/// Helper: collapse `Option<(mv, ref)>` into `(mv_x, mv_y, ref)` with
/// sentinel `(0, 0, REF_IDX_NONE)` when the neighbour is unavailable
/// / intra.
fn neighbour_to_tuple(n: Option<(MotionVector, i8)>) -> (i16, i16, i8) {
    match n {
        Some((mv, r)) => (mv.mv_x, mv.mv_y, r),
        None => (0, 0, REF_IDX_NONE),
    }
}

/// Result of running spatial-direct MV derivation at one MB.
///
/// Per-list `ref_idx == REF_IDX_NONE (-1)` means "this list is not
/// used at this MB" (decoder won't include this list in MC). The
/// caller is expected to populate `EncoderMvGrid` accordingly:
/// `Some((mv, ref_idx))` for active lists, `None` (i.e. clear that
/// list at the rect) for inactive ones.
///
/// **Phase 2.12 (#275, 2026-05-08)**: extended with per-8×8 sub-block
/// MVs to support spec § 8.4.1.2.2 step 6's per-sub-block colZeroFlag
/// override. A spec-compliant decoder writes per-8×8 different MVs to
/// its motion cache after override; matching that requires per-sub-block
/// MC at the encoder.
///
/// `mv_l0` / `mv_l1` are the median-predictor MV (whole-MB). They
/// represent the SHAPE of the spatial-direct result for legacy
/// callers (RDO SATD evaluator etc.) that don't need sub-block
/// granularity.
///
/// `mv_l0_per_8x8` / `mv_l1_per_8x8` are the per-8×8 sub-block MVs
/// AFTER colZeroFlag override. Index 0 = top-left, 1 = top-right,
/// 2 = bottom-left, 3 = bottom-right (raster order). Each sub-block
/// has either the median MV (if colMb's corresponding 8×8 was moving)
/// or `(0, 0)` (if static AND ref_idx == 0). For backward compat,
/// these arrays are filled with `mv_l0` / `mv_l1` when no override
/// applies — so per-sub-block consumers can blindly iterate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BDirectSpatialResult {
    /// L0 MV + ref_idx for the whole MB. `ref_idx == -1` means L0 is
    /// inactive and `mv` is meaningless. Equals top-left 8×8 sub-block's
    /// post-override MV (matches legacy uniform-MB callers' assumption).
    pub mv_l0: MotionVector,
    pub ref_idx_l0: i8,
    /// L1 MV + ref_idx for the whole MB.
    pub mv_l1: MotionVector,
    pub ref_idx_l1: i8,
    /// Per-8×8 sub-block MVs for L0 list. Index = i8 in raster order
    /// (TL/TR/BL/BR). Always populated; equal to `mv_l0` everywhere
    /// when no per-sub-block override applies.
    pub mv_l0_per_8x8: [MotionVector; 4],
    /// Per-8×8 sub-block MVs for L1 list.
    pub mv_l1_per_8x8: [MotionVector; 4],
}

impl BDirectSpatialResult {
    /// Convenience: this MB uses L0 (predFlagL0 = 1 in spec terms).
    pub fn uses_l0(&self) -> bool {
        self.ref_idx_l0 != REF_IDX_NONE
    }
    /// This MB uses L1 (predFlagL1 = 1 in spec terms).
    pub fn uses_l1(&self) -> bool {
        self.ref_idx_l1 != REF_IDX_NONE
    }
    /// Convenience: bipred (both lists active).
    pub fn is_bipred(&self) -> bool {
        self.uses_l0() && self.uses_l1()
    }
    /// Phase 2.12 (#275) — true if any 8×8 sub-block's MV differs
    /// from `mv_l0` (= colZeroFlag override fired on at least one
    /// sub-block). When false, callers can treat this as legacy
    /// uniform-MB result.
    pub fn has_per_8x8_override_l0(&self) -> bool {
        self.mv_l0_per_8x8.iter().any(|m| *m != self.mv_l0)
    }
    /// Phase 2.12 (#275) — true if any 8×8 sub-block's L1 MV differs
    /// from `mv_l1`.
    pub fn has_per_8x8_override_l1(&self) -> bool {
        self.mv_l1_per_8x8.iter().any(|m| *m != self.mv_l1)
    }
}

/// Spec § 8.4.1.2.1: derive per-list ref_idx for a B_Direct /
/// B_Skip MB at (mb_x, mb_y). Reads neighbour A (left), B (above),
/// C (above-right with D-fallback).
///
/// Returns the MIN_POSITIVE of available neighbours per list, or
/// `REF_IDX_NONE (-1)` if no neighbour has that list. Neighbours
/// off-frame / not-yet-decoded are treated as "unavailable" and
/// don't contribute. Intra neighbours (decoded but no MV) are also
/// "unavailable" per § 8.4.1.2.1.
fn derive_b_direct_ref_idx(grid: &EncoderMvGrid, mb_x: usize, mb_y: usize) -> (i8, i8) {
    // 4×4 anchor of the MB's top-left block.
    let tl_bx = (mb_x * 4) as isize;
    let tl_by = (mb_y * 4) as isize;
    // Neighbour A: left of top-left → (tl_bx - 1, tl_by).
    // Neighbour B: above top-left → (tl_bx, tl_by - 1).
    // Neighbour C: above-right of MB → (tl_bx + 4, tl_by - 1).
    //   D-fallback when C unavailable: (tl_bx - 1, tl_by - 1).
    let a_l0 = grid.get_l0(tl_bx - 1, tl_by);
    let b_l0 = grid.get_l0(tl_bx, tl_by - 1);
    let c_l0 = if grid.is_decoded(tl_bx + 4, tl_by - 1) {
        grid.get_l0(tl_bx + 4, tl_by - 1)
    } else {
        grid.get_l0(tl_bx - 1, tl_by - 1)
    };
    let a_l1 = grid.get_l1(tl_bx - 1, tl_by);
    let b_l1 = grid.get_l1(tl_bx, tl_by - 1);
    let c_l1 = if grid.is_decoded(tl_bx + 4, tl_by - 1) {
        grid.get_l1(tl_bx + 4, tl_by - 1)
    } else {
        grid.get_l1(tl_bx - 1, tl_by - 1)
    };

    let ref_idx_l0 = min_positive_ref_idx(&[a_l0, b_l0, c_l0]);
    let ref_idx_l1 = min_positive_ref_idx(&[a_l1, b_l1, c_l1]);
    (ref_idx_l0, ref_idx_l1)
}

/// Spec § 8.4.1.2.1 MIN_POSITIVE: among the supplied neighbours,
/// take the smallest non-negative ref_idx. If all are unavailable
/// (None / intra), return `REF_IDX_NONE (-1)`.
fn min_positive_ref_idx(neighbours: &[Option<(MotionVector, i8)>]) -> i8 {
    let mut best = REF_IDX_NONE;
    for n in neighbours {
        if let Some((_, r)) = n
            && *r >= 0
            && (best == REF_IDX_NONE || (*r as i32) < (best as i32)) {
                best = *r;
            }
    }
    best
}

/// §6E-A6.1 entry point — full spatial-direct MV derivation for one
/// MB at `(mb_x, mb_y)`. Implements spec § 8.4.1.2.1 + the median
/// path of § 8.4.1.2.2. Co-located static/moving check (colZeroFlag)
/// is gated on `l1_motion_grid` — pass `Some(&grid)` to enable
/// (production B-slice paths), `None` to disable (tests, callers
/// without DPB context).
pub fn derive_b_direct_spatial(
    grid: &EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
) -> BDirectSpatialResult {
    derive_b_direct_spatial_with_col(grid, mb_x, mb_y, None)
}

/// §6E-A6.1 + spec § 8.4.1.2.2 step 6 — spatial-direct MV derivation
/// with co-located static/moving check.
///
/// `l1_motion_grid` is the L1 reference picture's per-MB collocated
/// motion grid (= the next-anchor P-frame's `motion_grid`). When
/// present, the colZeroFlag check is applied: for each list X with
/// `refIdxLX == 0`, if the colMb is "static" (intra → false; else
/// `ref_idx_l0 == 0 && |mv_l0| ≤ 1` quarter-pel units → true), then
/// `mvLX = (0, 0)` regardless of the median predictor.
///
/// **Why this matters**: without colZeroFlag, a static-background
/// B-MB can pick up motion from its neighbour that overlapped a
/// moving subject — the median picks the moving MV and the
/// background is predicted from a shifted reference, producing a
/// "ghost" of the moving subject on the static wall. The colMb
/// check forces (0, 0) when the L1 reference's co-located content
/// is itself static, preventing that ghost.
///
/// **Granularity**: spec is per-4×4 sub-block. Phasm's
/// `ColocatedMvGrid` only stores one cell per MB (top-left 4×4
/// sample, see `EncoderMvGrid::to_colocated_grid`). For 16×16
/// partitions all sub-blocks share one MV anyway, so the MB-level
/// check matches the spec's collapsed result for that case.
pub fn derive_b_direct_spatial_with_col(
    grid: &EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
    l1_motion_grid: Option<&ColocatedMvGrid>,
) -> BDirectSpatialResult {
    let (ref_idx_l0, ref_idx_l1) = derive_b_direct_ref_idx(grid, mb_x, mb_y);

    // Spec § 8.4.1.2.1 boundary case: if BOTH lists have no
    // neighbour info, force both refs to 0 with zero MVs (the
    // "first B after IDR with no usable neighbours" path).
    if ref_idx_l0 == REF_IDX_NONE && ref_idx_l1 == REF_IDX_NONE {
        return BDirectSpatialResult {
            mv_l0: MotionVector::ZERO,
            ref_idx_l0: 0,
            mv_l1: MotionVector::ZERO,
            ref_idx_l1: 0,
            mv_l0_per_8x8: [MotionVector::ZERO; 4],
            mv_l1_per_8x8: [MotionVector::ZERO; 4],
        };
    }

    // Compute the median predictor MV per list (whole-MB).
    // Per spec § 8.4.1.2.2 step 5, this is the MV BEFORE colZeroFlag
    // override. Then per-8×8 override may zero out specific sub-blocks
    // per spec § 8.4.1.2.2 step 6.
    let tl_bx = mb_x * 4;
    let tl_by = mb_y * 4;
    let median_mv_l0 = if ref_idx_l0 != REF_IDX_NONE {
        predict_mv_for_partition(grid, tl_bx, tl_by, /* part_w_4x4 */ 4, ref_idx_l0)
    } else {
        MotionVector::ZERO
    };
    let median_mv_l1 = if ref_idx_l1 != REF_IDX_NONE {
        predict_mv_for_partition_l1(grid, tl_bx, tl_by, ref_idx_l1)
    } else {
        MotionVector::ZERO
    };

    // Phase 2.12 (#275) — per-8×8 colZeroFlag check + override.
    //
    // Spec § 8.4.1.2.2 step 6: colZeroFlag is checked PER 8×8
    // sub-block of the colocated MB. For each of 4 sub-blocks
    // (raster order TL/TR/BL/BR):
    //
    //   if NOT INTRA(colMb sub-block i8) AND ref[L1][0] is NOT long_ref
    //      AND ((l1ref0[xy8]==0 AND |l1mv0[xy4]|<=1 quarter-pel) OR
    //           (l1ref0[xy8]<0 AND l1ref1[xy8]==0 AND |l1mv1[xy4]|<=1)):
    //     // colZeroFlag fires for this sub-block
    //     if ref_l0_b == 0: zero L0 MV at this sub-block
    //     if ref_l1_b == 0: zero L1 MV at this sub-block
    //
    // We compute per-sub-block colZeroFlag bitmap, then build per-8×8
    // MV arrays (zero where override fires, median elsewhere). The
    // result struct's `mv_l0` / `mv_l1` are set to the TOP-LEFT
    // sub-block's value for legacy callers (RDO SATD eval etc.).
    let mut col_zero_flag_per_8x8 = [false; 4];
    if let Some(g) = l1_motion_grid {
        if (mb_x as u32) < g.mb_w && (mb_y as u32) < g.mb_h {
            let mb_bx = mb_x as u32 * 4;
            let mb_by = mb_y as u32 * 4;
            for i8 in 0..4u32 {
                // 8×8 sub-block i8 in raster: TL=0, TR=1, BL=2, BR=3.
                //
                // Phase 2.18 (#287, 2026-05-09) — sample at MB CORNERS
                // (positions (0,0), (3,0), (0,3), (3,3) in colMb's 4×4
                // grid) per the SUB_8X8 branch of the spec § 8.4.1.2.2
                // colocated-MB scan:
                //   xy4 = x8 * 3 + y8 * 3 * b4_stride
                //
                // For colMb encoded as P_8x8 with sub_mb_type∈{1,2,3}
                // (SUB_8x4, SUB_4x8, SUB_4x4), 4×4 cells WITHIN an 8×8
                // sub-block can have different MVs. The spec samples
                // corner-of-each-8x8; phasm previously sampled
                // TL-of-each-8x8 (= positions (0,0), (2,0), (0,2),
                // (2,2)) — DIFFERENT cells in 3 of 4 sub-blocks. When
                // colMb's per-4×4 MV varies within an 8×8, the
                // override decision diverged from the spec → divergent
                // grid → divergent bs at MB-edge deblock → visible
                // cascade at rows 14-15 of B-Skip MBs.
                let sub_bx = mb_bx + (i8 & 1) * 3;
                let sub_by = mb_by + (i8 >> 1) * 3;
                let cell = g.get_4x4(sub_bx, sub_by);
                if cell.ref_idx_l0 < 0 {
                    // Intra colMb sub-block → no override.
                    continue;
                }
                if cell.ref_idx_l0 == 0
                    && cell.mv_l0_x.abs() <= 1
                    && cell.mv_l0_y.abs() <= 1
                {
                    col_zero_flag_per_8x8[i8 as usize] = true;
                }
                // Note: spec also has the L1-ref0-static branch but
                // for our IBPBP M=2 the colMb is a P-frame which only
                // has L0 motion (no L1). The L1-branch never fires.
            }
        }
    }

    // Build per-8×8 sub-block MV arrays: zero where override fires
    // (and ref_idx == 0 for that list), median otherwise.
    let mut mv_l0_per_8x8 = [median_mv_l0; 4];
    let mut mv_l1_per_8x8 = [median_mv_l1; 4];
    for i8 in 0..4 {
        if col_zero_flag_per_8x8[i8] {
            if ref_idx_l0 == 0 {
                mv_l0_per_8x8[i8] = MotionVector::ZERO;
            }
            if ref_idx_l1 == 0 {
                mv_l1_per_8x8[i8] = MotionVector::ZERO;
            }
        }
    }

    // Inactive lists: clear per-sub-block to zero (the ref will be
    // REF_IDX_NONE so MC won't use them anyway).
    if ref_idx_l0 == REF_IDX_NONE {
        mv_l0_per_8x8 = [MotionVector::ZERO; 4];
    }
    if ref_idx_l1 == REF_IDX_NONE {
        mv_l1_per_8x8 = [MotionVector::ZERO; 4];
    }

    // Legacy mv_l0 / mv_l1 fields reflect the TOP-LEFT (i8=0) sub-block's
    // post-override MV. This matches the spec-compliant decoder behaviour
    // for callers that only inspect the MB-level result (e.g., RDO SATD
    // eval) — they see the same per-8×8-aware result as long as the MB is
    // uniform (= no override). For mixed-override MBs the top-left value
    // is the best single-MV approximation (= median if top-left wasn't
    // overridden, zero if it was).
    let mv_l0 = mv_l0_per_8x8[0];
    let mv_l1 = mv_l1_per_8x8[0];

    let result = BDirectSpatialResult {
        mv_l0,
        ref_idx_l0,
        mv_l1,
        ref_idx_l1,
        mv_l0_per_8x8,
        mv_l1_per_8x8,
    };

    // Phase 2.7 trace: keep a backward-compat `col_zero_flag` for
    // the dump (true iff ANY sub-block fired override).
    let col_zero_flag = col_zero_flag_per_8x8.iter().any(|&v| v);
    let _ = col_zero_flag;

    // Phase 2.7 (#272) — emit a trace row when env-gated. Captures
    // the full input state (neighbour MVs/refs for A/B/C on both
    // lists, colocated cell, colZeroFlag) plus the derived output
    // so an analysis pass can correlate max-deviation MBs with the
    // exact derivation inputs and locate where phasm's algorithm
    // diverges from spec § 8.4.1.2.2.
    if super::mb_decision_b::env_var_os_is_some("PHASM_B_SPATIAL_DIRECT_TRACE") {
        let tl_bx_isize = (mb_x * 4) as isize;
        let tl_by_isize = (mb_y * 4) as isize;
        let a_l0 = neighbour_to_tuple(grid.get_l0(tl_bx_isize - 1, tl_by_isize));
        let b_l0 = neighbour_to_tuple(grid.get_l0(tl_bx_isize, tl_by_isize - 1));
        let c_was_c = grid.is_decoded(tl_bx_isize + 4, tl_by_isize - 1);
        let c_l0 = if c_was_c {
            neighbour_to_tuple(grid.get_l0(tl_bx_isize + 4, tl_by_isize - 1))
        } else {
            neighbour_to_tuple(grid.get_l0(tl_bx_isize - 1, tl_by_isize - 1))
        };
        let a_l1 = neighbour_to_tuple(grid.get_l1(tl_bx_isize - 1, tl_by_isize));
        let b_l1 = neighbour_to_tuple(grid.get_l1(tl_bx_isize, tl_by_isize - 1));
        let c_l1 = if c_was_c {
            neighbour_to_tuple(grid.get_l1(tl_bx_isize + 4, tl_by_isize - 1))
        } else {
            neighbour_to_tuple(grid.get_l1(tl_bx_isize - 1, tl_by_isize - 1))
        };
        let (col_ref_idx_l0, col_mv_l0_x, col_mv_l0_y) = match l1_motion_grid {
            Some(g) if (mb_x as u32) < g.mb_w && (mb_y as u32) < g.mb_h => {
                let cell = g.get(mb_x as u32, mb_y as u32);
                (cell.ref_idx_l0, cell.mv_l0_x, cell.mv_l0_y)
            }
            _ => (REF_IDX_NONE, 0, 0),
        };
        push_spatial_direct_trace(SpatialDirectTrace {
            frame_idx: current_frame_idx(),
            mb_x: mb_x as u16,
            mb_y: mb_y as u16,
            kind: 0, // spatial
            ref_idx_l0,
            ref_idx_l1,
            a_l0_mv_x: a_l0.0,
            a_l0_mv_y: a_l0.1,
            a_l0_ref: a_l0.2,
            a_l1_mv_x: a_l1.0,
            a_l1_mv_y: a_l1.1,
            a_l1_ref: a_l1.2,
            b_l0_mv_x: b_l0.0,
            b_l0_mv_y: b_l0.1,
            b_l0_ref: b_l0.2,
            b_l1_mv_x: b_l1.0,
            b_l1_mv_y: b_l1.1,
            b_l1_ref: b_l1.2,
            c_l0_mv_x: c_l0.0,
            c_l0_mv_y: c_l0.1,
            c_l0_ref: c_l0.2,
            c_l1_mv_x: c_l1.0,
            c_l1_mv_y: c_l1.1,
            c_l1_ref: c_l1.2,
            c_was_c,
            col_ref_idx_l0,
            col_mv_l0_x,
            col_mv_l0_y,
            col_zero_flag,
            col_grid_present: l1_motion_grid.is_some(),
            out_mv_l0_x: mv_l0.mv_x,
            out_mv_l0_y: mv_l0.mv_y,
            out_mv_l1_x: mv_l1.mv_x,
            out_mv_l1_y: mv_l1.mv_y,
        });
    }

    result
}

/// §B-direct-fix.v3 — temporal-direct MV derivation per spec
/// § 8.4.1.2.3.
///
/// Unlike spatial-direct (median over neighbours A/B/C, vulnerable
/// to motion-boundary mis-inheritance), temporal-direct uses the
/// L1-reference-frame's colocated MB's L0 MV directly, scaled by
/// the temporal POC distance ratio. No median, no neighbour
/// mixing.
///
/// For IBPBP M=2 (POC distances td=4, tb=2):
///   DistScaleFactor ≈ 128
///   mvL0 ≈ mvCol / 2 (rounded)
///   mvL1 ≈ -mvCol / 2 (rounded)
///
/// On motion-boundary content (e.g., a slow-moving pants MB
/// adjacent to a torso MB with different motion vector),
/// spatial-direct's median picks the wrong neighbour and predicts
/// from a misaligned reference region → visible streak / staircase
/// pixels. Temporal-direct uses the colocated MB's own MV (the
/// pants's own motion, not the neighbour's) → correct reference
/// region → no artifact.
///
/// `l1_motion_grid` is the L1 reference picture's per-MB
/// colocated motion grid (same data spatial-direct uses for the
/// colZeroFlag check). When None, falls back to zero MVs.
///
/// `poc_curr`, `poc_l0`, `poc_l1` are the FULL (un-wrapped) POCs
/// for the current B-frame and its L0 / L1 references. For IBPBP
/// M=2: `poc_l0 = poc_curr - 2`, `poc_l1 = poc_curr + 2`.
///
/// Returns `(zero, zero)` MVs with `ref_idx_l0 = ref_idx_l1 = 0`
/// when:
/// - `l1_motion_grid` is None
/// - colocated MB is intra (`ref_idx_l0 < 0`)
/// - degenerate POC (`td == 0`)
pub fn derive_b_direct_temporal(
    l1_motion_grid: Option<&ColocatedMvGrid>,
    mb_x: usize,
    mb_y: usize,
    poc_curr: i32,
    poc_l0: i32,
    poc_l1: i32,
) -> BDirectSpatialResult {
    let zero = BDirectSpatialResult {
        mv_l0: MotionVector::ZERO,
        ref_idx_l0: 0,
        mv_l1: MotionVector::ZERO,
        ref_idx_l1: 0,
        mv_l0_per_8x8: [MotionVector::ZERO; 4],
        mv_l1_per_8x8: [MotionVector::ZERO; 4],
    };

    // Collect inputs for the optional trace path (Phase 2.7 #272).
    let mut col_ref_idx_l0_trace: i8 = REF_IDX_NONE;
    let mut col_mv_l0_x_trace: i16 = 0;
    let mut col_mv_l0_y_trace: i16 = 0;
    let col_grid_present = l1_motion_grid.is_some();

    let result = (|| {
        let grid = match l1_motion_grid {
            Some(g) => g,
            None => return zero,
        };
        if mb_x as u32 >= grid.mb_w || mb_y as u32 >= grid.mb_h {
            return zero;
        }
        let cell = grid.get(mb_x as u32, mb_y as u32);
        col_ref_idx_l0_trace = cell.ref_idx_l0;
        col_mv_l0_x_trace = cell.mv_l0_x;
        col_mv_l0_y_trace = cell.mv_l0_y;
        if cell.ref_idx_l0 < 0 {
            return zero;
        }

        let mv_col_x = cell.mv_l0_x as i32;
        let mv_col_y = cell.mv_l0_y as i32;

        // §B-direct-fix.v3 2026-05-07 — near-zero colMb override.
        // Spec § 8.4.1.2.2 step 6 (colZeroFlag) exists for spatial-direct
        // but NOT for temporal-direct (§ 8.4.1.2.3 has no equivalent).
        // Empirical: on motion boundaries, P-frame ME picks a spurious
        // sub-pixel MV (|mv|≤1 quarter-pel = ≤0.25 px) for "static" wall
        // MBs at the kid's silhouette — SAD/SATD finds a tiny win by
        // aligning silhouette gradient slightly. Without this override,
        // temporal-direct faithfully scales the noise → mvL0 ≈ ±mvCol/2
        // → wall MB pulls 4-5 px of texture from the adjacent foreground
        // → "pants pixels spill out onto the wall" (user 2026-05-07).
        // Real-world consumer encoders sidestep this via aggressive RDO
        // that picks L0/L1/Bi at motion boundaries; phasm currently picks
        // Direct 99% of B-MBs and so MUST override the noise here.
        // Match the spec colZeroFlag's |mv|≤1 quarter-pel threshold and
        // ref_idx==0 condition (single-ref IBPBP M=2 always sees ref=0).
        if cell.ref_idx_l0 == 0 && mv_col_x.abs() <= 1 && mv_col_y.abs() <= 1 {
            return zero;
        }

        let td = (poc_l1 - poc_l0).clamp(-128, 127);
        let tb = (poc_curr - poc_l0).clamp(-128, 127);
        if td == 0 {
            return zero;
        }

        // Spec § 8.4.1.2.3 eq 8-203 / 8-204 / 8-205:
        //   tx = (16384 + Abs(td / 2)) / td
        //   DistScaleFactor = Clip3(-1024, 1023, (tb*tx + 32) >> 6)
        //   mvLX = (DistScaleFactor * mvCol + 128) >> 8
        let tx = (16384 + (td.abs() / 2)) / td;
        let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);

        let mv_l0 = MotionVector {
            mv_x: ((dsf * mv_col_x + 128) >> 8) as i16,
            mv_y: ((dsf * mv_col_y + 128) >> 8) as i16,
        };
        let mv_l1 = MotionVector {
            mv_x: (mv_l0.mv_x as i32 - mv_col_x) as i16,
            mv_y: (mv_l0.mv_y as i32 - mv_col_y) as i16,
        };

        BDirectSpatialResult {
            mv_l0,
            ref_idx_l0: 0,
            mv_l1,
            ref_idx_l1: 0,
            // Temporal-direct doesn't have a colZeroFlag override
            // per spec § 8.4.1.2.3 (only spatial-direct has it).
            // So per-8×8 MVs are uniform = mv_l0/mv_l1.
            mv_l0_per_8x8: [mv_l0; 4],
            mv_l1_per_8x8: [mv_l1; 4],
        }
    })();

    // Phase 2.7 (#272) — temporal-direct trace. Neighbour fields
    // are unused in the temporal path (no median over A/B/C); they
    // are populated with REF_IDX_NONE sentinels.
    if super::mb_decision_b::env_var_os_is_some("PHASM_B_SPATIAL_DIRECT_TRACE") {
        push_spatial_direct_trace(SpatialDirectTrace {
            frame_idx: current_frame_idx(),
            mb_x: mb_x as u16,
            mb_y: mb_y as u16,
            kind: 1, // temporal
            ref_idx_l0: result.ref_idx_l0,
            ref_idx_l1: result.ref_idx_l1,
            a_l0_mv_x: 0,
            a_l0_mv_y: 0,
            a_l0_ref: REF_IDX_NONE,
            a_l1_mv_x: 0,
            a_l1_mv_y: 0,
            a_l1_ref: REF_IDX_NONE,
            b_l0_mv_x: 0,
            b_l0_mv_y: 0,
            b_l0_ref: REF_IDX_NONE,
            b_l1_mv_x: 0,
            b_l1_mv_y: 0,
            b_l1_ref: REF_IDX_NONE,
            c_l0_mv_x: 0,
            c_l0_mv_y: 0,
            c_l0_ref: REF_IDX_NONE,
            c_l1_mv_x: 0,
            c_l1_mv_y: 0,
            c_l1_ref: REF_IDX_NONE,
            c_was_c: false,
            col_ref_idx_l0: col_ref_idx_l0_trace,
            col_mv_l0_x: col_mv_l0_x_trace,
            col_mv_l0_y: col_mv_l0_y_trace,
            col_zero_flag: false, // not applicable to temporal
            col_grid_present,
            out_mv_l0_x: result.mv_l0.mv_x,
            out_mv_l0_y: result.mv_l0.mv_y,
            out_mv_l1_x: result.mv_l1.mv_x,
            out_mv_l1_y: result.mv_l1.mv_y,
        });
    }

    result
}

/// §6E-A6.1 — public wrapper over [`predict_mv_for_partition_l1`]
/// so encoder.rs (and the §6E-A6.1 walker) can compute L1 MVD
/// predictors without re-implementing the median logic.
///
/// Note: this hardcodes a 16×16 partition shape (no `mb_part_idx`,
/// no per-shape C-block position). For partitioned mb_types
/// (16×8 / 8×16 / 8×8) call [`predict_mv_for_mb_partition_l1`]
/// instead, which mirrors L0's `predict_mv_for_mb_partition` with
/// spec § 8.4.1.3.1 directional shortcuts.
pub fn predict_mv_for_partition_l1_pub(
    grid: &EncoderMvGrid,
    tl_bx: usize,
    tl_by: usize,
    current_ref_idx: i8,
) -> MotionVector {
    predict_mv_for_partition_l1(grid, tl_bx, tl_by, current_ref_idx)
}

/// §B-cascade-real Phase 2 (#267) — L1-list MV predictor for a
/// partitioned mb_type's partition (16×8 / 8×16 / 16×16).
///
/// Mirrors L0's [`super::partition_state::predict_mv_for_mb_partition`]
/// with spec § 8.4.1.3.1 directional shortcuts (B for 16×8 idx=0,
/// A for idx=1, A for 8×16 idx=0, C for idx=1) but reads `grid.get_l1`
/// instead of `grid.get` (= `grid.get_l0`). Required for symmetric
/// L0/L1 MVD emission in B-slice partitioned mb_types 12/13/16/17
/// where partition 0 is single-list (L0 or L1) and partition 1 is
/// Bi: previously the L1 emit used the 16×16-shape `predict_mv_for_partition_l1_pub`
/// which has neither directional shortcuts nor per-partition C-block
/// position, causing PMV asymmetry vs spec-compliant decoders
/// (max|Δ|=199 on motion content per #266 mismatch_y measurement).
pub fn predict_mv_for_mb_partition_l1(
    grid: &EncoderMvGrid,
    tl_bx: usize,
    tl_by: usize,
    part_w_4x4: usize,
    part_h_4x4: usize,
    mb_part_idx: u8,
    current_ref_idx: i8,
) -> MotionVector {
    let x = tl_bx as isize;
    let y = tl_by as isize;
    let a = grid.get_l1(x - 1, y);
    let b = grid.get_l1(x, y - 1);
    // D-fallback only when C is not-yet-decoded / off-frame; in-frame
    // intra C is "available" (contributes mv=(0,0) per spec § 8.4.1.3
    // and fails the ref-match check in directional shortcuts). Mirrors
    // L0 path's logic exactly (task #154 root cause).
    let c_bx = x + part_w_4x4 as isize;
    let c_by = y - 1;
    let c = if grid.is_decoded(c_bx, c_by) {
        grid.get_l1(c_bx, c_by)
    } else {
        grid.get_l1(x - 1, y - 1)
    };

    // Spec § 8.4.1.3.1 directional shortcuts — mirror of the L0
    // path's logic. Only P_16x8 (4×2 in 4×4 units) and P_8x16 (2×4).
    if part_w_4x4 == 4 && part_h_4x4 == 2 {
        if mb_part_idx == 0 {
            if let Some((mv, r)) = b
                && r == current_ref_idx {
                    return mv;
                }
        } else if let Some((mv, r)) = a
            && r == current_ref_idx {
                return mv;
            }
    } else if part_w_4x4 == 2 && part_h_4x4 == 4 {
        if mb_part_idx == 0 {
            if let Some((mv, r)) = a
                && r == current_ref_idx {
                    return mv;
                }
        } else if let Some((mv, r)) = c
            && r == current_ref_idx {
                return mv;
            }
    }

    // Fall through to the general median predictor (mirrors L0's
    // call to `predict_mv_for_partition`).
    let availability = [a.is_some(), b.is_some(), c.is_some()];
    let avail_count: u8 = availability.iter().map(|&v| v as u8).sum();
    if avail_count == 1
        && let Some((mv, _)) = a.or(b).or(c)
    {
        return mv;
    }

    // Single-matching-ref_idx special case.
    let matches: [Option<MotionVector>; 3] = [
        a.and_then(|(mv, r)| if r == current_ref_idx { Some(mv) } else { None }),
        b.and_then(|(mv, r)| if r == current_ref_idx { Some(mv) } else { None }),
        c.and_then(|(mv, r)| if r == current_ref_idx { Some(mv) } else { None }),
    ];
    let match_count: u8 = matches.iter().map(|m| m.is_some() as u8).sum();
    if match_count == 1
        && let Some(mv) = matches.iter().find_map(|m| *m)
    {
        return mv;
    }

    // General median over A/B/C with partition-aware C position.
    // Use the canonical `median3` (sibling to L0's
    // `predict_mv_for_partition`); the inline 3-element-median
    // implementation that originally lived here was a buggy
    // formula returning `p` whenever `p` was the maximum.
    let la = a.map(|(m, _)| m).unwrap_or_default();
    let lb = b.map(|(m, _)| m).unwrap_or_default();
    let lc = c.map(|(m, _)| m).unwrap_or_default();
    MotionVector {
        mv_x: median3(la.mv_x, lb.mv_x, lc.mv_x),
        mv_y: median3(la.mv_y, lb.mv_y, lc.mv_y),
    }
}

/// L1-view median MV predictor for a 16x16 partition. Mirrors
/// `predict_mv_for_partition` exactly but reads `grid.get_l1`
/// instead of `grid.get_l0`. Spec § 8.4.1.3 — same algorithm,
/// per-list neighbour view.
///
/// (A future cleanup might generalize `predict_mv_for_partition`
/// to take a list selector — but the existing function is used
/// hot-path P-side and its signature is referenced from many
/// callers. Adding a parallel l1 variant here keeps the change
/// minimal.)
fn predict_mv_for_partition_l1(
    grid: &EncoderMvGrid,
    tl_bx: usize,
    tl_by: usize,
    current_ref_idx: i8,
) -> MotionVector {
    let x = tl_bx as isize;
    let y = tl_by as isize;
    let part_w_4x4 = 4isize;

    let a = grid.get_l1(x - 1, y);
    let b = grid.get_l1(x, y - 1);
    let c_bx = x + part_w_4x4;
    let c_by = y - 1;
    let c = if grid.is_decoded(c_bx, c_by) {
        grid.get_l1(c_bx, c_by)
    } else {
        grid.get_l1(x - 1, y - 1)
    };

    // Spec § 8.4.1.3: if only one of A/B/C is available, that one
    // is the predictor.
    let availability = [a.is_some(), b.is_some(), c.is_some()];
    let avail_count: u8 = availability.iter().map(|&v| v as u8).sum();
    if avail_count == 1
        && let Some((mv, _)) = a.or(b).or(c)
    {
        return mv;
    }

    // Single-matching-ref_idx special case.
    let matches: [Option<MotionVector>; 3] = [
        a.and_then(|(mv, r)| if r == current_ref_idx { Some(mv) } else { None }),
        b.and_then(|(mv, r)| if r == current_ref_idx { Some(mv) } else { None }),
        c.and_then(|(mv, r)| if r == current_ref_idx { Some(mv) } else { None }),
    ];
    let match_count = matches.iter().filter(|m| m.is_some()).count();
    if match_count == 1
        && let Some(mv) = matches.iter().flatten().next()
    {
        return *mv;
    }

    // General case: componentwise median; unavailable = zero.
    let la = a.map(|(m, _)| m).unwrap_or_default();
    let lb = b.map(|(m, _)| m).unwrap_or_default();
    let lc = c.map(|(m, _)| m).unwrap_or_default();
    MotionVector {
        mv_x: median3(la.mv_x, lb.mv_x, lc.mv_x),
        mv_y: median3(la.mv_y, lb.mv_y, lc.mv_y),
    }
}

/// Three-tap median (i16 components). Local copy — the P-side
/// `partition_state.rs` keeps its `median3` private. Re-export
/// would couple modules unnecessarily for one tiny function.
#[inline]
fn median3(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.max(c)).min(b.max(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(x: i16, y: i16) -> MotionVector {
        MotionVector { mv_x: x, mv_y: y }
    }

    // ─── Boundary / edge cases ─────────────────────────────────

    #[test]
    fn no_neighbours_yields_both_refs_zero_zero_mv() {
        // Empty grid — current MB is at top-left (0, 0). No
        // neighbours at all. Spec says: both refs forced to 0
        // with zero MVs.
        let grid = EncoderMvGrid::new(2, 2);
        let r = derive_b_direct_spatial(&grid, 0, 0);
        assert_eq!(r.ref_idx_l0, 0);
        assert_eq!(r.ref_idx_l1, 0);
        assert_eq!(r.mv_l0, MotionVector::ZERO);
        assert_eq!(r.mv_l1, MotionVector::ZERO);
        assert!(r.is_bipred());
    }

    #[test]
    fn only_l0_neighbour_yields_l0_only() {
        // Left neighbour has L0 only. Current MB is at (1, 0)
        // → 4x4 anchor (4, 0). Left neighbour at (3, 0) — belongs
        // to MB 0, populated.
        let mut grid = EncoderMvGrid::new(2, 2);
        grid.fill_lists(0, 0, 4, 4, Some((mv(10, 20), 0)), None);
        let r = derive_b_direct_spatial(&grid, 1, 0);
        assert_eq!(r.ref_idx_l0, 0);
        assert_eq!(r.ref_idx_l1, REF_IDX_NONE);
        assert!(r.uses_l0());
        assert!(!r.uses_l1());
        // Only one neighbour → that's the predictor.
        assert_eq!(r.mv_l0, mv(10, 20));
    }

    #[test]
    fn only_l1_neighbour_yields_l1_only() {
        let mut grid = EncoderMvGrid::new(2, 2);
        grid.fill_lists(0, 0, 4, 4, None, Some((mv(-5, 15), 0)));
        let r = derive_b_direct_spatial(&grid, 1, 0);
        assert_eq!(r.ref_idx_l0, REF_IDX_NONE);
        assert_eq!(r.ref_idx_l1, 0);
        assert_eq!(r.mv_l1, mv(-5, 15));
    }

    #[test]
    fn bipred_neighbour_yields_both_lists() {
        // Left neighbour has both L0 and L1.
        let mut grid = EncoderMvGrid::new(2, 2);
        grid.fill_lists(
            0, 0, 4, 4,
            Some((mv(7, -3), 0)),
            Some((mv(-2, 9), 0)),
        );
        let r = derive_b_direct_spatial(&grid, 1, 0);
        assert!(r.is_bipred());
        assert_eq!(r.mv_l0, mv(7, -3));
        assert_eq!(r.mv_l1, mv(-2, 9));
    }

    #[test]
    fn three_neighbours_use_componentwise_median() {
        // Top row of MBs (0, 0), (1, 0), (2, 0): set three different
        // L0 MVs. Then derive at (1, 1).
        let mut grid = EncoderMvGrid::new(3, 2);
        grid.fill_lists(0, 0, 4, 4, Some((mv(1, 10), 0)), None);
        grid.fill_lists(4, 0, 4, 4, Some((mv(2, 20), 0)), None);
        grid.fill_lists(8, 0, 4, 4, Some((mv(3, 30), 0)), None);
        // Left neighbour of MB(1, 1) (anchor (4, 4)): cells (3, 4)..
        grid.fill_lists(0, 4, 4, 4, Some((mv(100, 0), 0)), None);
        let r = derive_b_direct_spatial(&grid, 1, 1);
        // A = (100, 0), B = (2, 20), C = (3, 30) (above-right of MB
        // (1, 1) at 4×4 (8, 3)).
        // Median x = 3, median y = 20.
        assert_eq!(r.mv_l0, mv(3, 20));
        assert_eq!(r.ref_idx_l0, 0);
    }

    #[test]
    fn min_positive_picks_smallest_ref_idx() {
        // Spec § 8.4.1.2.1: refIdxLX = MIN_POSITIVE over neighbours.
        // We don't have multi-ref configurations in phasm's ship
        // path (single-ref, refIdx always 0), but the algorithm
        // must still be correct for general cases.
        let neighbours = [
            Some((mv(0, 0), 2)),
            Some((mv(0, 0), 1)),
            Some((mv(0, 0), 5)),
        ];
        assert_eq!(min_positive_ref_idx(&neighbours), 1);
    }

    #[test]
    fn min_positive_skips_intra_neighbours() {
        // None entries (intra / off-frame) don't contribute.
        let neighbours = [
            None,
            Some((mv(0, 0), 0)),
            None,
        ];
        assert_eq!(min_positive_ref_idx(&neighbours), 0);
    }

    #[test]
    fn min_positive_all_intra_returns_neg_one() {
        let neighbours: [Option<(MotionVector, i8)>; 3] = [None, None, None];
        assert_eq!(min_positive_ref_idx(&neighbours), REF_IDX_NONE);
    }

    #[test]
    fn predict_mv_l1_mirrors_l0_predictor_on_l1_data() {
        // Sanity: feeding the same neighbour MV pattern via L1
        // produces the same result the L0 predictor would on L0
        // data. Spec § 8.4.1.3 doesn't depend on which list it's
        // applied to.
        let mut grid_l0 = EncoderMvGrid::new(3, 2);
        grid_l0.fill_lists(0, 0, 4, 4, Some((mv(1, 10), 0)), None);
        grid_l0.fill_lists(4, 0, 4, 4, Some((mv(2, 20), 0)), None);
        grid_l0.fill_lists(8, 0, 4, 4, Some((mv(3, 30), 0)), None);
        grid_l0.fill_lists(0, 4, 4, 4, Some((mv(100, 0), 0)), None);
        let l0_pred = predict_mv_for_partition(&grid_l0, 4, 4, 4, 0);

        let mut grid_l1 = EncoderMvGrid::new(3, 2);
        grid_l1.fill_lists(0, 0, 4, 4, None, Some((mv(1, 10), 0)));
        grid_l1.fill_lists(4, 0, 4, 4, None, Some((mv(2, 20), 0)));
        grid_l1.fill_lists(8, 0, 4, 4, None, Some((mv(3, 30), 0)));
        grid_l1.fill_lists(0, 4, 4, 4, None, Some((mv(100, 0), 0)));
        let l1_pred = predict_mv_for_partition_l1(&grid_l1, 4, 4, 0);

        assert_eq!(l0_pred, l1_pred,
            "L0 and L1 predictors must produce identical results on identical neighbour data");
    }

    #[test]
    fn split_l0_l1_at_different_neighbours_yields_correct_split_result() {
        // Two MB row, each gets a different list at a different
        // neighbour position. Verify L0 and L1 derivation are
        // independent.
        let mut grid = EncoderMvGrid::new(3, 2);
        // Above-left: only L0.
        grid.fill_lists(0, 0, 4, 4, Some((mv(1, 1), 0)), None);
        // Above-middle: only L1.
        grid.fill_lists(4, 0, 4, 4, None, Some((mv(2, 2), 0)));
        // Left-of-current: bipred.
        grid.fill_lists(0, 4, 4, 4, Some((mv(50, 0), 0)), Some((mv(0, 50), 0)));
        // Derive at MB (1, 1) — anchor (4, 4).
        let r = derive_b_direct_spatial(&grid, 1, 1);
        // Both lists should be active (each has at least one
        // neighbour with that list available).
        assert!(r.is_bipred(),
            "expected bipred — L0 from A+B?, L1 from A+B?: got {r:?}");
    }
}
