// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Intra-prediction mode decision for the H.264 encoder. Phase 6A.3.
//!
//! Reuses the decoder-side prediction generators
//! (`super::super::intra_pred::predict_4x4`, `predict_16x16`,
//! `predict_chroma_8x8`) — the encoder side just orchestrates "for
//! each candidate mode, generate prediction, compute SATD, pick the
//! lowest-cost mode."
//!
//! SATD = Sum of Absolute Hadamard-Transformed Differences.
//! Computed via `super::transform::forward_hadamard_4x4`.
//!
//! Psy-RD bias and CAVLC bit-cost adjustment are deferred — see the
//! algorithm note for the staged plan:
//!   docs/design/h264-encoder-algorithms/intra-prediction.md

use crate::codec::h264::intra_pred::{
    predict_4x4, predict_16x16, predict_chroma_8x8, Intra16x16Mode, Intra4x4Mode,
    IntraChroma8x8Mode, Neighbors16x16, Neighbors4x4, NeighborsChroma8x8,
};

use super::transform::forward_hadamard_4x4;

// ─── SATD helpers ──────────────────────────────────────────────────

/// Compute SATD for a single 4×4 residual block: forward-Hadamard
/// the residual, then sum absolute values of all 16 coefficients.
fn satd_4x4(residual: &[[i32; 4]; 4]) -> u32 {
    let h = forward_hadamard_4x4(residual);
    let mut sum: u32 = 0;
    for row in &h {
        for &v in row {
            sum += v.unsigned_abs();
        }
    }
    sum
}

/// Compute SATD between two 4×4 pixel blocks.
fn satd_4x4_pixels(source: &[[u8; 4]; 4], pred: &[[u8; 4]; 4]) -> u32 {
    let mut residual = [[0i32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            residual[i][j] = source[i][j] as i32 - pred[i][j] as i32;
        }
    }
    satd_4x4(&residual)
}

/// Compute SATD across a 16×16 macroblock as the sum of the 16
/// constituent 4×4 sub-block SATDs.
pub(crate) fn satd_16x16(source: &[[u8; 16]; 16], pred: &[[u8; 16]; 16]) -> u32 {
    let mut total: u32 = 0;
    for by in 0..4 {
        for bx in 0..4 {
            let mut sub_src = [[0u8; 4]; 4];
            let mut sub_prd = [[0u8; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    sub_src[i][j] = source[by * 4 + i][bx * 4 + j];
                    sub_prd[i][j] = pred[by * 4 + i][bx * 4 + j];
                }
            }
            total = total.saturating_add(satd_4x4_pixels(&sub_src, &sub_prd));
        }
    }
    total
}

/// Compute SATD across an 8×8 chroma block (4 × 4×4 sub-block SATDs).
fn satd_8x8(source: &[[u8; 8]; 8], pred: &[[u8; 8]; 8]) -> u32 {
    let mut total: u32 = 0;
    for by in 0..2 {
        for bx in 0..2 {
            let mut sub_src = [[0u8; 4]; 4];
            let mut sub_prd = [[0u8; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    sub_src[i][j] = source[by * 4 + i][bx * 4 + j];
                    sub_prd[i][j] = pred[by * 4 + i][bx * 4 + j];
                }
            }
            total = total.saturating_add(satd_4x4_pixels(&sub_src, &sub_prd));
        }
    }
    total
}

// ─── Intra 4×4 mode decision ──────────────────────────────────────

/// Result of an Intra_4x4 mode-decision pass.
#[derive(Debug, Clone, Copy)]
pub struct ModeDecision4x4 {
    pub mode: Intra4x4Mode,
    pub predicted: [[u8; 4]; 4],
    pub satd: u32,
}

/// Modes that need a top neighbor.
const NEEDS_TOP_4X4: &[Intra4x4Mode] = &[
    Intra4x4Mode::Vertical,
    Intra4x4Mode::DiagonalDownLeft,
    Intra4x4Mode::DiagonalDownRight,
    Intra4x4Mode::VerticalRight,
    Intra4x4Mode::HorizontalDown,
    Intra4x4Mode::VerticalLeft,
];

/// Modes that need a left neighbor.
const NEEDS_LEFT_4X4: &[Intra4x4Mode] = &[
    Intra4x4Mode::Horizontal,
    Intra4x4Mode::DiagonalDownRight,
    Intra4x4Mode::VerticalRight,
    Intra4x4Mode::HorizontalDown,
    Intra4x4Mode::HorizontalUp,
];

/// Modes that need top-left.
const NEEDS_TL_4X4: &[Intra4x4Mode] = &[
    Intra4x4Mode::DiagonalDownRight,
    Intra4x4Mode::VerticalRight,
    Intra4x4Mode::HorizontalDown,
];

/// Modes that need top-right (the upper-right-corner samples).
const NEEDS_TR_4X4: &[Intra4x4Mode] = &[
    Intra4x4Mode::DiagonalDownLeft,
    Intra4x4Mode::VerticalLeft,
];

const ALL_MODES_4X4: [Intra4x4Mode; 9] = [
    Intra4x4Mode::Vertical,
    Intra4x4Mode::Horizontal,
    Intra4x4Mode::Dc,
    Intra4x4Mode::DiagonalDownLeft,
    Intra4x4Mode::DiagonalDownRight,
    Intra4x4Mode::VerticalRight,
    Intra4x4Mode::HorizontalDown,
    Intra4x4Mode::VerticalLeft,
    Intra4x4Mode::HorizontalUp,
];

#[inline]
fn mode_available_4x4(mode: Intra4x4Mode, n: &Neighbors4x4) -> bool {
    if NEEDS_TOP_4X4.contains(&mode) && !n.top_available {
        return false;
    }
    if NEEDS_LEFT_4X4.contains(&mode) && !n.left_available {
        return false;
    }
    if NEEDS_TL_4X4.contains(&mode) && !n.top_left_available {
        return false;
    }
    if NEEDS_TR_4X4.contains(&mode) && !n.top_right_available {
        return false;
    }
    true
}

/// Pick the best Intra_4x4 mode for a 4×4 source block.
///
/// Iterates the 9 candidate modes (filtered by neighbor
/// availability), generates the prediction for each via
/// `predict_4x4`, scores by SATD against the source, and returns
/// the lowest-cost mode along with the predicted block.
///
/// Falls back to `Dc` (always valid — uses 128 fill when neighbors
/// missing) if no other mode is available, which is the spec's
/// guaranteed fallback path.
pub fn choose_intra_4x4_mode(
    neighbors: &Neighbors4x4,
    source: &[[u8; 4]; 4],
) -> ModeDecision4x4 {
    let mut best = ModeDecision4x4 {
        mode: Intra4x4Mode::Dc,
        predicted: predict_4x4(Intra4x4Mode::Dc, neighbors),
        satd: u32::MAX,
    };
    best.satd = satd_4x4_pixels(source, &best.predicted);

    for &mode in &ALL_MODES_4X4 {
        if mode == Intra4x4Mode::Dc {
            continue; // already scored as the baseline
        }
        if !mode_available_4x4(mode, neighbors) {
            continue;
        }
        let predicted = predict_4x4(mode, neighbors);
        let satd = satd_4x4_pixels(source, &predicted);
        if satd < best.satd {
            best = ModeDecision4x4 { mode, predicted, satd };
        }
    }
    best
}

// ─── Intra 16×16 mode decision ────────────────────────────────────

/// Result of an Intra_16x16 mode-decision pass.
#[derive(Debug, Clone, Copy)]
pub struct ModeDecision16x16 {
    pub mode: Intra16x16Mode,
    pub predicted: [[u8; 16]; 16],
    pub satd: u32,
}

#[inline]
fn mode_available_16x16(mode: Intra16x16Mode, n: &Neighbors16x16) -> bool {
    match mode {
        Intra16x16Mode::Vertical => n.top_available,
        Intra16x16Mode::Horizontal => n.left_available,
        Intra16x16Mode::Dc => true,
        Intra16x16Mode::Plane => {
            n.top_available && n.left_available && n.top_left_available
        }
    }
}

const ALL_MODES_16X16: [Intra16x16Mode; 4] = [
    Intra16x16Mode::Vertical,
    Intra16x16Mode::Horizontal,
    Intra16x16Mode::Dc,
    Intra16x16Mode::Plane,
];

pub fn choose_intra_16x16_mode(
    neighbors: &Neighbors16x16,
    source: &[[u8; 16]; 16],
) -> ModeDecision16x16 {
    choose_intra_16x16_mode_psy(neighbors, source, 0)
}

/// Psy-RD-aware 16×16 mode decision.
///
/// `psy_strength` (0..=32, 0 = disabled) biases the chooser toward
/// modes whose prediction has similar AC energy to the source. The
/// bias added to each mode's SATD is:
///
///     psy_strength × |ac_energy(source) − ac_energy(pred)| / 16
///
/// (the ÷16 keeps the magnitude comparable to satd_16x16). Biases
/// predictions that either over- or under-flatten the texture.
pub fn choose_intra_16x16_mode_psy(
    neighbors: &Neighbors16x16,
    source: &[[u8; 16]; 16],
    psy_strength: u32,
) -> ModeDecision16x16 {
    let src_ac = ac_energy_16x16(source);
    let mut best = ModeDecision16x16 {
        mode: Intra16x16Mode::Dc,
        predicted: predict_16x16(Intra16x16Mode::Dc, neighbors),
        satd: u32::MAX,
    };
    let base = satd_16x16(source, &best.predicted);
    best.satd = base.saturating_add(psy_bias(src_ac, ac_energy_16x16(&best.predicted), psy_strength, 16));

    for &mode in &ALL_MODES_16X16 {
        if mode == Intra16x16Mode::Dc {
            continue;
        }
        if !mode_available_16x16(mode, neighbors) {
            continue;
        }
        let predicted = predict_16x16(mode, neighbors);
        let satd = satd_16x16(source, &predicted);
        let adj = satd.saturating_add(psy_bias(
            src_ac,
            ac_energy_16x16(&predicted),
            psy_strength,
            16,
        ));
        if adj < best.satd {
            best = ModeDecision16x16 { mode, predicted, satd: adj };
        }
    }
    best
}

/// Sum of |non-DC Hadamard coefficients| over a 4×4 block (Phase E
/// helper). Real frequency-domain AC energy. Pub-crate so the inter
/// path (partition_decision.rs psy bias) can reuse the same metric.
pub(crate) fn hadamard_ac_sum_4x4(block: &[[u8; 4]; 4]) -> u32 {
    let mut signed = [[0i32; 4]; 4];
    for (dy, row) in block.iter().enumerate() {
        for (dx, &v) in row.iter().enumerate() {
            signed[dy][dx] = v as i32;
        }
    }
    let h = forward_hadamard_4x4(&signed);
    let mut ac: u32 = 0;
    for (dy, row) in h.iter().enumerate() {
        for (dx, &c) in row.iter().enumerate() {
            if dy == 0 && dx == 0 {
                continue;
            }
            ac = ac.saturating_add(c.unsigned_abs());
        }
    }
    ac
}

/// 16×16 Hadamard-AC via 16 tiled 4×4 Hadamards. Used by Phase E.2
/// inter psy bias.
pub(crate) fn hadamard_ac_sum_16x16(block: &[[u8; 16]; 16]) -> u32 {
    let mut ac: u32 = 0;
    for by in 0..4 {
        for bx in 0..4 {
            let mut tile = [[0u8; 4]; 4];
            for dy in 0..4 {
                for dx in 0..4 {
                    tile[dy][dx] = block[by * 4 + dy][bx * 4 + dx];
                }
            }
            ac = ac.saturating_add(hadamard_ac_sum_4x4(&tile));
        }
    }
    ac
}

/// Spatial AC-deviation proxy (sum of |pixel − mean|). Cheaper, and
/// what the intra psy bias still uses. Phase E.3 tested swapping this
/// for Hadamard-AC but it was a visual no-op on P-heavy content
/// (intra fires on only ~1/30 frames); task #156 tracks revisiting.
fn ac_energy_16x16(block: &[[u8; 16]; 16]) -> u32 {
    let mut sum: u32 = 0;
    for row in block {
        for &v in row {
            sum = sum.saturating_add(v as u32);
        }
    }
    let mean = sum / 256;
    let mut ac: u32 = 0;
    for row in block {
        for &v in row {
            let d = v as i32 - mean as i32;
            ac = ac.saturating_add(d.unsigned_abs());
        }
    }
    ac
}

fn ac_energy_4x4(block: &[[u8; 4]; 4]) -> u32 {
    let mut sum: u32 = 0;
    for row in block {
        for &v in row {
            sum = sum.saturating_add(v as u32);
        }
    }
    let mean = sum / 16;
    let mut ac: u32 = 0;
    for row in block {
        for &v in row {
            let d = v as i32 - mean as i32;
            ac = ac.saturating_add(d.unsigned_abs());
        }
    }
    ac
}

/// Compute the psy-RD bias: `strength × |src_ac − pred_ac| / divisor`.
#[inline]
fn psy_bias(src_ac: u32, pred_ac: u32, strength: u32, divisor: u32) -> u32 {
    if strength == 0 {
        return 0;
    }
    let diff = src_ac.abs_diff(pred_ac);
    (diff.saturating_mul(strength)) / divisor.max(1)
}

/// Psy-RD-aware 4×4 mode decision.
pub fn choose_intra_4x4_mode_psy(
    neighbors: &Neighbors4x4,
    source: &[[u8; 4]; 4],
    psy_strength: u32,
) -> ModeDecision4x4 {
    let src_ac = ac_energy_4x4(source);
    let mut best = ModeDecision4x4 {
        mode: Intra4x4Mode::Dc,
        predicted: predict_4x4(Intra4x4Mode::Dc, neighbors),
        satd: u32::MAX,
    };
    let base = satd_4x4_pixels(source, &best.predicted);
    // Divisor 16 matches `choose_intra_16x16_mode_psy`. For equal
    // per-pixel AC deviations, 4×4 ac_energy sums over 16 pixels and
    // 16×16 sums over 256 pixels; dividing each by 16 normalizes the
    // bias-to-SATD ratio so the psy nudge carries the same weight
    // across mode-size choices. Using divisor=1 here (historical) made
    // the bias ~16× more influential on 4×4 decisions than on 16×16,
    // which over-rewarded 4×4 modes whose predictions happen to
    // match source AC energy — a major contributor to poor I-frame
    // quality (see memory/h264_real_world_bug.md).
    best.satd = base.saturating_add(psy_bias(src_ac, ac_energy_4x4(&best.predicted), psy_strength, 16));

    for &mode in &ALL_MODES_4X4 {
        if mode == Intra4x4Mode::Dc {
            continue;
        }
        if !mode_available_4x4(mode, neighbors) {
            continue;
        }
        let predicted = predict_4x4(mode, neighbors);
        let satd = satd_4x4_pixels(source, &predicted);
        let adj =
            satd.saturating_add(psy_bias(src_ac, ac_energy_4x4(&predicted), psy_strength, 16));
        if adj < best.satd {
            best = ModeDecision4x4 { mode, predicted, satd: adj };
        }
    }
    best
}

// ─── Intra Chroma 8×8 mode decision ───────────────────────────────

/// Result of an Intra-chroma mode-decision pass.
#[derive(Debug, Clone, Copy)]
pub struct ModeDecisionChroma {
    pub mode: IntraChroma8x8Mode,
    pub predicted: [[u8; 8]; 8],
    pub satd: u32,
}

#[inline]
fn mode_available_chroma(mode: IntraChroma8x8Mode, n: &NeighborsChroma8x8) -> bool {
    match mode {
        IntraChroma8x8Mode::Vertical => n.top_available,
        IntraChroma8x8Mode::Horizontal => n.left_available,
        IntraChroma8x8Mode::Dc => true,
        IntraChroma8x8Mode::Plane => {
            n.top_available && n.left_available && n.top_left_available
        }
    }
}

const ALL_MODES_CHROMA: [IntraChroma8x8Mode; 4] = [
    IntraChroma8x8Mode::Dc,
    IntraChroma8x8Mode::Horizontal,
    IntraChroma8x8Mode::Vertical,
    IntraChroma8x8Mode::Plane,
];

pub fn choose_intra_chroma_mode(
    neighbors: &NeighborsChroma8x8,
    source: &[[u8; 8]; 8],
) -> ModeDecisionChroma {
    let mut best = ModeDecisionChroma {
        mode: IntraChroma8x8Mode::Dc,
        predicted: predict_chroma_8x8(IntraChroma8x8Mode::Dc, neighbors),
        satd: u32::MAX,
    };
    best.satd = satd_8x8(source, &best.predicted);

    for &mode in &ALL_MODES_CHROMA {
        if mode == IntraChroma8x8Mode::Dc {
            continue;
        }
        if !mode_available_chroma(mode, neighbors) {
            continue;
        }
        let predicted = predict_chroma_8x8(mode, neighbors);
        let satd = satd_8x8(source, &predicted);
        if satd < best.satd {
            best = ModeDecisionChroma { mode, predicted, satd };
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_neighbors_4x4(top: [u8; 8], left: [u8; 4], tl: u8) -> Neighbors4x4 {
        Neighbors4x4 {
            top,
            left,
            top_left: tl,
            top_available: true,
            top_right_available: true,
            left_available: true,
            top_left_available: true,
        }
    }

    fn full_neighbors_16x16(top: [u8; 16], left: [u8; 16], tl: u8) -> Neighbors16x16 {
        Neighbors16x16 {
            top,
            left,
            top_left: tl,
            top_available: true,
            left_available: true,
            top_left_available: true,
        }
    }

    fn full_neighbors_chroma(top: [u8; 8], left: [u8; 8], tl: u8) -> NeighborsChroma8x8 {
        NeighborsChroma8x8 {
            top,
            left,
            top_left: tl,
            top_available: true,
            left_available: true,
            top_left_available: true,
        }
    }

    // ─── SATD ────────────────────────────────────────────────────

    #[test]
    fn satd_zero_residual_zero() {
        assert_eq!(satd_4x4(&[[0i32; 4]; 4]), 0);
    }

    #[test]
    fn satd_constant_residual_only_dc() {
        // Constant residual concentrates all Hadamard energy at (0,0)
        // = 16 × constant. SATD = |16 × c|.
        let residual = [[5i32; 4]; 4];
        assert_eq!(satd_4x4(&residual), 16 * 5);
    }

    #[test]
    fn satd_pixels_round_trip() {
        let src = [[100u8; 4]; 4];
        let prd = [[100u8; 4]; 4];
        assert_eq!(satd_4x4_pixels(&src, &prd), 0);
    }

    // ─── 4×4 mode decision ────────────────────────────────────────

    #[test]
    fn choose_4x4_flat_input_picks_dc() {
        let n = full_neighbors_4x4([100; 8], [100; 4], 100);
        let source = [[100u8; 4]; 4];
        let d = choose_intra_4x4_mode(&n, &source);
        // Several modes produce all-100 output for these neighbors —
        // any of them are tied at SATD 0. DC is the deterministic
        // baseline pick when nothing wins.
        assert_eq!(d.satd, 0, "perfect match should give SATD 0");
    }

    #[test]
    fn choose_4x4_vertical_input_with_only_top_picks_vertical() {
        // Top samples are [10, 20, 30, 40, _, _, _, _] (right 4 unused
        // for vertical). Source is the same column repeated.
        let n = Neighbors4x4 {
            top: [10, 20, 30, 40, 0, 0, 0, 0],
            left: [0; 4],
            top_left: 0,
            top_available: true,
            top_right_available: false,
            left_available: false,
            top_left_available: false,
        };
        let source = [
            [10, 20, 30, 40],
            [10, 20, 30, 40],
            [10, 20, 30, 40],
            [10, 20, 30, 40],
        ];
        let d = choose_intra_4x4_mode(&n, &source);
        assert_eq!(d.mode, Intra4x4Mode::Vertical);
        assert_eq!(d.satd, 0);
    }

    #[test]
    fn choose_4x4_horizontal_input_with_only_left_picks_horizontal() {
        let n = Neighbors4x4 {
            top: [0; 8],
            left: [10, 20, 30, 40],
            top_left: 0,
            top_available: false,
            top_right_available: false,
            left_available: true,
            top_left_available: false,
        };
        let source = [
            [10, 10, 10, 10],
            [20, 20, 20, 20],
            [30, 30, 30, 30],
            [40, 40, 40, 40],
        ];
        let d = choose_intra_4x4_mode(&n, &source);
        assert_eq!(d.mode, Intra4x4Mode::Horizontal);
        assert_eq!(d.satd, 0);
    }

    #[test]
    fn choose_4x4_no_neighbors_falls_back_to_dc() {
        let n = Neighbors4x4 {
            top: [0; 8],
            left: [0; 4],
            top_left: 0,
            top_available: false,
            top_right_available: false,
            left_available: false,
            top_left_available: false,
        };
        let source = [[200u8; 4]; 4];
        let d = choose_intra_4x4_mode(&n, &source);
        assert_eq!(d.mode, Intra4x4Mode::Dc);
        // DC with no neighbors uses 128 default; residual = 200 - 128 = 72
        // per pixel; flat residual SATD = 16 × 72 = 1152.
        assert_eq!(d.satd, 16 * 72);
    }

    #[test]
    fn choose_4x4_deterministic() {
        let n = full_neighbors_4x4([10, 50, 100, 150, 200, 200, 200, 200], [10, 50, 100, 150], 5);
        let source = [
            [12, 48, 105, 145],
            [11, 51, 99, 152],
            [13, 47, 102, 148],
            [10, 50, 101, 149],
        ];
        let a = choose_intra_4x4_mode(&n, &source);
        let b = choose_intra_4x4_mode(&n, &source);
        assert_eq!(a.mode, b.mode);
        assert_eq!(a.satd, b.satd);
    }

    // ─── 16×16 mode decision ──────────────────────────────────────

    #[test]
    fn choose_16x16_flat_picks_dc_satd_zero() {
        let n = full_neighbors_16x16([100; 16], [100; 16], 100);
        let source = [[100u8; 16]; 16];
        let d = choose_intra_16x16_mode(&n, &source);
        assert_eq!(d.satd, 0);
    }

    #[test]
    fn choose_16x16_vertical_only_top_picks_vertical() {
        let mut top = [0u8; 16];
        for i in 0..16 {
            top[i] = (i as u8) * 10;
        }
        let n = Neighbors16x16 {
            top,
            left: [0; 16],
            top_left: 0,
            top_available: true,
            left_available: false,
            top_left_available: false,
        };
        let source = [top; 16];
        let d = choose_intra_16x16_mode(&n, &source);
        assert_eq!(d.mode, Intra16x16Mode::Vertical);
        assert_eq!(d.satd, 0);
    }

    #[test]
    fn choose_16x16_horizontal_only_left_picks_horizontal() {
        let mut left = [0u8; 16];
        for i in 0..16 {
            left[i] = (i as u8) * 10;
        }
        let n = Neighbors16x16 {
            top: [0; 16],
            left,
            top_left: 0,
            top_available: false,
            left_available: true,
            top_left_available: false,
        };
        let mut source = [[0u8; 16]; 16];
        for y in 0..16 {
            for x in 0..16 {
                source[y][x] = left[y];
            }
        }
        let d = choose_intra_16x16_mode(&n, &source);
        assert_eq!(d.mode, Intra16x16Mode::Horizontal);
        assert_eq!(d.satd, 0);
    }

    // ─── Chroma mode decision ─────────────────────────────────────

    #[test]
    fn choose_chroma_flat_picks_dc_satd_zero() {
        let n = full_neighbors_chroma([128; 8], [128; 8], 128);
        let source = [[128u8; 8]; 8];
        let d = choose_intra_chroma_mode(&n, &source);
        assert_eq!(d.satd, 0);
    }

    #[test]
    fn choose_chroma_vertical_only_top_picks_vertical() {
        let top: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        let n = NeighborsChroma8x8 {
            top,
            left: [0; 8],
            top_left: 0,
            top_available: true,
            left_available: false,
            top_left_available: false,
        };
        let source = [top; 8];
        let d = choose_intra_chroma_mode(&n, &source);
        assert_eq!(d.mode, IntraChroma8x8Mode::Vertical);
        assert_eq!(d.satd, 0);
    }
}
