// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! 4×4-granular MV + ref_idx grid for the encoder, mirroring the
//! parser's `MvPredictorContext`. Populated per-MB as partitions are
//! resolved; read by the median predictor for next-partition and
//! next-MB neighbor lookups.
//!
//! Scope: Phase 6B.3.2a — supports arbitrary partition widths and
//! heights measured in 4×4 blocks.

use super::motion_estimation::MotionVector;

/// Ref-idx sentinel for blocks that are intra-coded or off-frame —
/// the median predictor treats them as "not available".
pub const REF_IDX_NONE: i8 = -1;

/// Snapshot of a macroblock's 4×4 of 4×4-blocks rectangle in the
/// [`EncoderMvGrid`]. Captured before speculative sub-MB writes so
/// the grid can be rolled back if that candidate isn't picked.
#[derive(Debug, Clone)]
pub struct MbMvSnapshot {
    mvs: [MotionVector; 16],
    refs: [i8; 16],
    decs: [bool; 16],
    base_bx: usize,
    base_by: usize,
}

/// Frame-wide 4×4 grid of MVs + ref_idx.
///
/// `decoded[idx]` distinguishes "not yet processed in raster order"
/// (false) from "processed; ref_idx tells if intra/inter" (true). Per
/// spec § 8.4.1.3 / § 6.4.11.7 the D-fallback for neighbour C only
/// applies when C's MB is NOT AVAILABLE (off-frame, different slice,
/// or `decoded == false`). An in-frame intra neighbour is "available"
/// and contributes mv=(0, 0) to the median.
#[derive(Debug, Clone)]
pub struct EncoderMvGrid {
    width_4x4: usize,
    height_4x4: usize,
    mv: Vec<MotionVector>,
    ref_idx: Vec<i8>,
    decoded: Vec<bool>,
}

impl EncoderMvGrid {
    pub fn new(mb_width: usize, mb_height: usize) -> Self {
        let w = mb_width * 4;
        let h = mb_height * 4;
        Self {
            width_4x4: w,
            height_4x4: h,
            mv: vec![MotionVector::default(); w * h],
            ref_idx: vec![REF_IDX_NONE; w * h],
            decoded: vec![false; w * h],
        }
    }

    pub fn width_4x4(&self) -> usize {
        self.width_4x4
    }

    pub fn height_4x4(&self) -> usize {
        self.height_4x4
    }

    /// Clear the grid (all positions not yet decoded). Call at the
    /// start of each P-frame.
    pub fn reset(&mut self) {
        for v in self.mv.iter_mut() {
            *v = MotionVector::default();
        }
        for r in self.ref_idx.iter_mut() {
            *r = REF_IDX_NONE;
        }
        for d in self.decoded.iter_mut() {
            *d = false;
        }
    }

    /// Write a partition's MV+ref_idx into the rectangle
    /// `[bx, bx+w) × [by, by+h)` at 4×4-block granularity. Marks the
    /// slots as decoded (so future neighbour lookups can distinguish
    /// intra from not-yet-processed).
    pub fn fill(
        &mut self,
        bx: usize,
        by: usize,
        w: usize,
        h: usize,
        mv: MotionVector,
        ref_idx: i8,
    ) {
        for dy in 0..h {
            for dx in 0..w {
                let x = bx + dx;
                let y = by + dy;
                if x < self.width_4x4 && y < self.height_4x4 {
                    let idx = y * self.width_4x4 + x;
                    self.mv[idx] = mv;
                    self.ref_idx[idx] = ref_idx;
                    self.decoded[idx] = true;
                }
            }
        }
    }

    /// Snapshot the 4×4-block rectangle spanned by one macroblock so
    /// it can be restored after speculative writes (Phase A.2 scratch
    /// pattern for sub-MB median predictors — see
    /// `docs/design/h264-encoder-quality-plan.md`).
    pub fn snapshot_mb(&self, mb_x: usize, mb_y: usize) -> MbMvSnapshot {
        let base_bx = mb_x * 4;
        let base_by = mb_y * 4;
        let mut mvs = [MotionVector::default(); 16];
        let mut refs = [REF_IDX_NONE; 16];
        let mut decs = [false; 16];
        for dy in 0..4 {
            for dx in 0..4 {
                let x = base_bx + dx;
                let y = base_by + dy;
                if x < self.width_4x4 && y < self.height_4x4 {
                    let idx = y * self.width_4x4 + x;
                    mvs[dy * 4 + dx] = self.mv[idx];
                    refs[dy * 4 + dx] = self.ref_idx[idx];
                    decs[dy * 4 + dx] = self.decoded[idx];
                }
            }
        }
        MbMvSnapshot { mvs, refs, decs, base_bx, base_by }
    }

    /// Undo any writes within the MB rectangle captured by a prior
    /// [`Self::snapshot_mb`] call.
    pub fn restore_mb(&mut self, snap: &MbMvSnapshot) {
        for dy in 0..4 {
            for dx in 0..4 {
                let x = snap.base_bx + dx;
                let y = snap.base_by + dy;
                if x < self.width_4x4 && y < self.height_4x4 {
                    let idx = y * self.width_4x4 + x;
                    self.mv[idx] = snap.mvs[dy * 4 + dx];
                    self.ref_idx[idx] = snap.refs[dy * 4 + dx];
                    self.decoded[idx] = snap.decs[dy * 4 + dx];
                }
            }
        }
    }

    /// Lookup the MV+ref_idx at 4×4 grid position `(bx, by)`. Returns
    /// `None` if out-of-bounds or if the position is marked intra.
    pub fn get(&self, bx: isize, by: isize) -> Option<(MotionVector, i8)> {
        if bx < 0 || by < 0 {
            return None;
        }
        let bx = bx as usize;
        let by = by as usize;
        if bx >= self.width_4x4 || by >= self.height_4x4 {
            return None;
        }
        let idx = by * self.width_4x4 + bx;
        let r = self.ref_idx[idx];
        if r == REF_IDX_NONE {
            None
        } else {
            Some((self.mv[idx], r))
        }
    }

    /// True iff `(bx, by)` has already been processed (fill called).
    /// False for out-of-frame positions and for not-yet-processed
    /// raster-order successors. Used by the median predictor to pick
    /// between D-fallback (C is not-available) and contribute-zero
    /// (C is in-frame intra).
    pub fn is_decoded(&self, bx: isize, by: isize) -> bool {
        if bx < 0 || by < 0 {
            return false;
        }
        let bx = bx as usize;
        let by = by as usize;
        if bx >= self.width_4x4 || by >= self.height_4x4 {
            return false;
        }
        self.decoded[by * self.width_4x4 + bx]
    }
}

/// Three-tap median filter (i16 components).
#[inline]
fn median3(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.max(c)).min(b.max(c))
}

/// MV predictor for an MB-level partition following spec § 8.4.1.3.1.
/// Handles the directional shortcuts for P_16x8 / P_8x16 before falling
/// through to the general median rule.
///
/// Directional shortcuts (§ 8.4.1.3.1):
/// * P_16x8 partition 0 (top):    if refB == curRef → mvB
/// * P_16x8 partition 1 (bottom): if refA == curRef → mvA
/// * P_8x16 partition 0 (left):   if refA == curRef → mvA
/// * P_8x16 partition 1 (right):  if refC == curRef → mvC
///
/// For sub-MB partitions inside P_8x8 (including 8×4, 4×8, 4×4), pass
/// the MB-level 8×8 sub-MB partition dims (part_w_4x4 = part_h_4x4 = 2)
/// so the shortcuts don't fire — sub-MB partitions always use median.
pub fn predict_mv_for_mb_partition(
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
    let a = grid.get(x - 1, y);
    let b = grid.get(x, y - 1);
    // D-fallback only when C is not-yet-decoded / off-frame; in-frame
    // intra C is "available" (contributes mv=(0,0) per spec § 8.4.1.3
    // and fails the ref-match check in directional shortcuts). Task #154.
    let c_bx = x + part_w_4x4 as isize;
    let c_by = y - 1;
    let c = if grid.is_decoded(c_bx, c_by) {
        grid.get(c_bx, c_by)
    } else {
        grid.get(x - 1, y - 1)
    };

    // Directional shortcuts — only P_16x8 (4×2) and P_8x16 (2×4).
    if part_w_4x4 == 4 && part_h_4x4 == 2 {
        if mb_part_idx == 0 {
            if let Some((mv, r)) = b {
                if r == current_ref_idx {
                    return mv;
                }
            }
        } else if let Some((mv, r)) = a {
            if r == current_ref_idx {
                return mv;
            }
        }
    } else if part_w_4x4 == 2 && part_h_4x4 == 4 {
        if mb_part_idx == 0 {
            if let Some((mv, r)) = a {
                if r == current_ref_idx {
                    return mv;
                }
            }
        } else if let Some((mv, r)) = c {
            if r == current_ref_idx {
                return mv;
            }
        }
    }

    predict_mv_for_partition(grid, tl_bx, tl_by, part_w_4x4, current_ref_idx)
}

/// MV predictor for a partition whose top-left 4×4 block sits at
/// absolute frame coords `(tl_bx, tl_by)` and whose width in 4×4
/// blocks is `part_w_4x4`. Follows spec § 8.4.1.3 median-of-A/B/C
/// with D fallback when C is unavailable.
///
/// Note: the § 8.4.1.3.1 directional shortcuts for P_16x8 / P_8x16 are
/// handled by [`predict_mv_for_mb_partition`]. This function implements
/// only the general § 8.4.1.3 median.
pub fn predict_mv_for_partition(
    grid: &EncoderMvGrid,
    tl_bx: usize,
    tl_by: usize,
    part_w_4x4: usize,
    current_ref_idx: i8,
) -> MotionVector {
    let x = tl_bx as isize;
    let y = tl_by as isize;
    let a = grid.get(x - 1, y); // left
    let b = grid.get(x, y - 1); // above
    // Spec § 8.4.1.3 / § 6.4.11.7: D-fallback for C applies when C's
    // 4×4 block is NOT AVAILABLE — off-frame, different slice, OR not
    // yet decoded in raster order (e.g. sub-MB 3 of P_8x8 whose C
    // points to a not-yet-processed MB). An in-frame intra C that IS
    // decoded is still "available" and contributes mv=(0, 0) to the
    // median directly (it returns None from `grid.get` but must not
    // trigger D-fallback). Task #154 root cause.
    let c_bx = x + part_w_4x4 as isize;
    let c_by = y - 1;
    let c = if grid.is_decoded(c_bx, c_by) {
        grid.get(c_bx, c_by) // may be None for in-frame intra
    } else {
        grid.get(x - 1, y - 1) // D-fallback: C not available
    };

    // Spec 8.4.1.3: if only one of A/B/C is available, that one is
    // the predictor.
    let availability = [a.is_some(), b.is_some(), c.is_some()];
    let avail_count: u8 = availability.iter().map(|&v| v as u8).sum();
    if avail_count == 1 {
        if let Some((mv, _)) = a.or(b).or(c) {
            return mv;
        }
    }

    // Single-matching-ref_idx special case.
    let matches: [Option<MotionVector>; 3] = [
        a.and_then(|(mv, r)| if r == current_ref_idx { Some(mv) } else { None }),
        b.and_then(|(mv, r)| if r == current_ref_idx { Some(mv) } else { None }),
        c.and_then(|(mv, r)| if r == current_ref_idx { Some(mv) } else { None }),
    ];
    let match_count = matches.iter().filter(|m| m.is_some()).count();
    if match_count == 1 {
        for m in &matches {
            if let Some(mv) = m {
                return *mv;
            }
        }
    }

    // General case: componentwise median; unavailable = zero.
    let la = a.map(|(m, _)| m).unwrap_or_default();
    let tb = b.map(|(m, _)| m).unwrap_or_default();
    let tr = c.map(|(m, _)| m).unwrap_or_default();
    MotionVector {
        mv_x: median3(la.mv_x, tb.mv_x, tr.mv_x),
        mv_y: median3(la.mv_y, tb.mv_y, tr.mv_y),
    }
}

/// MV predictor for a P_Skip macroblock per spec § 8.4.1.2.1.
///
/// The `(0, 0)` shortcut rules split on the REASON a neighbor is
/// unavailable:
///
/// - mbAddrN is off-frame (or in a different slice) → `(0, 0)`.
/// - mbAddrN is in-frame AND inter with `ref_idx == 0, mv == (0, 0)`
///   → `(0, 0)`.
/// - mbAddrN is in-frame AND intra-coded (ref_idx=-1) → **NOT** a
///   shortcut; fall through to the general AMVP median (where intra
///   neighbors contribute `mv=(0,0)` per § 8.4.1.3).
///
/// Task #154 bug fix: previously this function treated "grid.get
/// returns None" as a blanket off-frame signal and shortcut to
/// `(0, 0)` for intra neighbors too. That produced a spec-divergent
/// MV on P_Skip MBs whose A/B neighbor was intra-coded in-frame —
/// enc-dec parity broke whenever intra-in-P fired next to a
/// P_Skip-able MB. Spec-conformant behaviour is to fall through to
/// the median predictor, yielding a different (often non-zero) MV
/// and different MC. Fix: check bounds explicitly to distinguish
/// off-frame from intra.
pub fn predict_p_skip_mv(grid: &EncoderMvGrid, tl_bx: usize, tl_by: usize) -> MotionVector {
    let x = tl_bx as isize;
    let y = tl_by as isize;
    // Off-frame check: neighbor A is at (x-1, y), neighbor B at
    // (x, y-1). If either coordinate is negative we're at a frame
    // edge — shortcut per § 8.4.1.2.1.
    let a_off_frame = x - 1 < 0;
    let b_off_frame = y - 1 < 0;
    if a_off_frame || b_off_frame {
        return MotionVector { mv_x: 0, mv_y: 0 };
    }
    // Both neighbors are in-frame. grid.get() returns None if the
    // neighbor is intra-coded (ref_idx = REF_IDX_NONE); that case
    // intentionally falls through to the median below.
    let a = grid.get(x - 1, y);
    let b = grid.get(x, y - 1);
    if let Some((mv_a, ref_a)) = a {
        if ref_a == 0 && mv_a.mv_x == 0 && mv_a.mv_y == 0 {
            return MotionVector { mv_x: 0, mv_y: 0 };
        }
    }
    if let Some((mv_b, ref_b)) = b {
        if ref_b == 0 && mv_b.mv_x == 0 && mv_b.mv_y == 0 {
            return MotionVector { mv_x: 0, mv_y: 0 };
        }
    }
    // Fall through to standard AMVP median. Intra neighbors contribute
    // mv=(0,0) at that layer per § 8.4.1.3.
    predict_mv_for_partition(grid, tl_bx, tl_by, 4, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_get_none_when_unset() {
        let g = EncoderMvGrid::new(2, 2);
        assert_eq!(g.get(0, 0), None);
        assert_eq!(g.get(-1, 0), None);
        assert_eq!(g.get(100, 100), None);
    }

    #[test]
    fn grid_fill_and_get_roundtrip() {
        let mut g = EncoderMvGrid::new(2, 2);
        let mv = MotionVector { mv_x: 42, mv_y: -7 };
        g.fill(0, 0, 4, 4, mv, 0);
        for bx in 0..4 {
            for by in 0..4 {
                assert_eq!(g.get(bx, by), Some((mv, 0)));
            }
        }
        // Outside the 4×4 rectangle is still unset.
        assert_eq!(g.get(4, 4), None);
    }

    #[test]
    fn grid_reset_clears() {
        let mut g = EncoderMvGrid::new(1, 1);
        g.fill(0, 0, 4, 4, MotionVector { mv_x: 1, mv_y: 2 }, 0);
        g.reset();
        assert_eq!(g.get(0, 0), None);
    }

    #[test]
    fn predictor_no_neighbors_returns_zero() {
        let g = EncoderMvGrid::new(2, 2);
        let mv = predict_mv_for_partition(&g, 0, 0, 4, 0);
        assert_eq!(mv, MotionVector::ZERO);
    }

    #[test]
    fn predictor_single_left_neighbor_returns_left() {
        let mut g = EncoderMvGrid::new(2, 2);
        g.fill(0, 0, 4, 4, MotionVector { mv_x: 10, mv_y: -5 }, 0);
        // MB (1, 0) in 4×4-block coords = (4, 0). Its left neighbor
        // is (3, 0) — belongs to MB 0, filled above.
        let mv = predict_mv_for_partition(&g, 4, 0, 4, 0);
        assert_eq!(mv, MotionVector { mv_x: 10, mv_y: -5 });
    }

    #[test]
    fn predictor_three_neighbors_componentwise_median() {
        let mut g = EncoderMvGrid::new(3, 2);
        // Top row of MBs (0, 0), (1, 0), (2, 0): set three different MVs.
        g.fill(0, 0, 4, 4, MotionVector { mv_x: 1, mv_y: 10 }, 0); // MB 0 (above for later)
        g.fill(4, 0, 4, 4, MotionVector { mv_x: 2, mv_y: 20 }, 0);
        g.fill(8, 0, 4, 4, MotionVector { mv_x: 3, mv_y: 30 }, 0); // C (above-right)
        // MB (1, 1) at 4×4 coords (4, 4): A = (3, 4), B = (4, 3), C = (8, 3).
        g.fill(0, 4, 4, 4, MotionVector { mv_x: 100, mv_y: 0 }, 0); // A (left)
        let mv = predict_mv_for_partition(&g, 4, 4, 4, 0);
        // Median of (100, 0), (2, 20), (3, 30) componentwise:
        // mv_x = median(100, 2, 3) = 3
        // mv_y = median(0, 20, 30) = 20
        assert_eq!(mv, MotionVector { mv_x: 3, mv_y: 20 });
    }

    // ─── Phase 6C.7 P_Skip MV derivation tests ──────────────

    #[test]
    fn p_skip_mv_no_neighbors_is_zero() {
        let g = EncoderMvGrid::new(2, 2);
        let mv = predict_p_skip_mv(&g, 0, 0);
        assert_eq!(mv, MotionVector::ZERO);
    }

    #[test]
    fn p_skip_mv_left_unavailable_is_zero() {
        let mut g = EncoderMvGrid::new(2, 2);
        // Only B (top) is present; A (left) missing.
        g.fill(0, 0, 4, 4, MotionVector { mv_x: 7, mv_y: 3 }, 0);
        let mv = predict_p_skip_mv(&g, 0, 4); // MB at (0, 1)
        assert_eq!(mv, MotionVector::ZERO);
    }

    #[test]
    fn p_skip_mv_top_unavailable_is_zero() {
        let mut g = EncoderMvGrid::new(2, 2);
        // Only A (left) is present.
        g.fill(0, 0, 4, 4, MotionVector { mv_x: 7, mv_y: 3 }, 0);
        let mv = predict_p_skip_mv(&g, 4, 0); // MB at (1, 0)
        assert_eq!(mv, MotionVector::ZERO);
    }

    #[test]
    fn p_skip_mv_trivial_a_shortcut_to_zero() {
        let mut g = EncoderMvGrid::new(3, 2);
        // A = left of MB (1, 1) at 4×4 (3, 4): ref_idx=0, mv=(0,0).
        g.fill(0, 4, 4, 4, MotionVector { mv_x: 0, mv_y: 0 }, 0);
        // B = top: mv != 0.
        g.fill(4, 0, 4, 4, MotionVector { mv_x: 5, mv_y: 5 }, 0);
        // C not needed here.
        let mv = predict_p_skip_mv(&g, 4, 4);
        assert_eq!(mv, MotionVector::ZERO, "trivial A should shortcut to zero");
    }

    #[test]
    fn p_skip_mv_trivial_b_shortcut_to_zero() {
        let mut g = EncoderMvGrid::new(3, 2);
        g.fill(0, 4, 4, 4, MotionVector { mv_x: 5, mv_y: 5 }, 0); // A non-zero
        g.fill(4, 0, 4, 4, MotionVector { mv_x: 0, mv_y: 0 }, 0); // B trivial
        let mv = predict_p_skip_mv(&g, 4, 4);
        assert_eq!(mv, MotionVector::ZERO);
    }

    #[test]
    fn p_skip_mv_nonzero_neighbors_use_median() {
        let mut g = EncoderMvGrid::new(3, 2);
        g.fill(0, 0, 4, 4, MotionVector { mv_x: 1, mv_y: 10 }, 0);
        g.fill(4, 0, 4, 4, MotionVector { mv_x: 2, mv_y: 20 }, 0);
        g.fill(8, 0, 4, 4, MotionVector { mv_x: 3, mv_y: 30 }, 0);
        g.fill(0, 4, 4, 4, MotionVector { mv_x: 100, mv_y: 1 }, 0);
        let mv = predict_p_skip_mv(&g, 4, 4);
        // Neither A nor B is trivial → fall through to median.
        // A = (100, 1), B = (2, 20), C = (3, 30).
        // Median x = 3, y = 20.
        assert_eq!(mv, MotionVector { mv_x: 3, mv_y: 20 });
    }
}
