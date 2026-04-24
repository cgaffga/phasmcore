// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! DDCA (Distortion-Drift-Compensated Approximation) for H.264 Phase 2b/2c.
//!
//! Two drift propagation models:
//!
//! **Phase 2b — inter-block intra-prediction drift** (Y plane):
//! Intra-prediction modes read samples from neighbouring 4×4 blocks; a
//! flip at block B perturbs B's reconstruction, which perturbs every
//! forward-decoded block that uses B as its A/B/C neighbour. The
//! [`DRIFT_WEIGHTS`] table + [`forward_drift`] walker accumulate a
//! decayed drift score per flip, which multiplies the base UNIWARD cost.
//!
//! **Phase 2c — inter-frame MV-driven drift**:
//! P-frames reference I-frame samples via motion compensation; if many
//! P-blocks point into the 4×4 pixel region of an I-frame position,
//! flipping that position propagates to all of them. The
//! [`InterFrameRefMap`] accumulates per-4×4-block reference weights
//! from the parsed MV fields of the P-frames between consecutive I-
//! frames, and [`apply_inter_frame_drift`] multiplies the cost by a
//! second drift factor.
//!
//! Both models apply multiplicatively on top of UNIWARD, so the total
//! cost of a luma flip is:
//!   `cost = base_uniward × (1 + w_intra_drift × intra_drift)
//!                        × (1 + w_inter_drift × inter_drift)`

use crate::codec::h264::cavlc::EmbedDomain;
use crate::codec::h264::macroblock::{Macroblock, MbType, BLOCK_INDEX_TO_POS};
use crate::codec::h264::mv::MvField;

use super::h264_uniward::FramePosition;

/// Tunable weights for the DDCA drift multipliers.
#[derive(Debug, Clone, Copy)]
pub struct DdcaParams {
    /// Weight scaling the inter-block intra-prediction drift (Phase 2b).
    pub w_drift: f32,
    /// Attenuation per hop when walking the forward intra chain.
    /// 0.7 means a block three hops downstream contributes at ~0.34× the
    /// weight of a direct neighbour.
    pub decay_per_hop: f32,
    /// Cap on how far forward the intra walker recurses. 3 covers the
    /// same-MB case plus one MB over — enough for the dominant signal.
    pub max_hops: u8,
    /// Weight scaling the inter-frame MV-driven drift (Phase 2c).
    pub w_inter_drift: f32,
    /// Per-P-frame-distance decay for inter-frame drift. The P-frame
    /// immediately after the I-frame contributes at weight 1.0; each
    /// subsequent P-frame scales down by this factor.
    pub inter_frame_decay: f32,
}

impl Default for DdcaParams {
    fn default() -> Self {
        Self {
            w_drift: 0.35,
            decay_per_hop: 0.7,
            max_hops: 3,
            w_inter_drift: 0.25,
            inter_frame_decay: 0.75,
        }
    }
}

/// Drift sensitivity weights per Intra_4x4 mode.
///
/// Indexed as `DRIFT_WEIGHTS[mode][relation]` where `relation`:
/// * 0 = Left neighbour (current block's mode reads from its left 4×4)
/// * 1 = Top neighbour
/// * 2 = Top-right neighbour
/// * 3 = Top-left (corner sample)
///
/// Values derived from the per-mode prediction formulas in
/// [`crate::codec::h264::intra_pred`]: count how many of the 16 output
/// pixels consume each neighbour-axis sample, normalised so a pure
/// copy-style mode (0 Vertical, 1 Horizontal) scores 1.0 on its axis.
///
/// Mode numbering per spec Table 8-2 / `Intra4x4Mode`:
/// 0=Vertical, 1=Horizontal, 2=DC, 3=DiagonalDownLeft, 4=DiagonalDownRight,
/// 5=VerticalRight, 6=HorizontalDown, 7=VerticalLeft, 8=HorizontalUp.
static DRIFT_WEIGHTS: [[f32; 4]; 9] = [
    [0.0, 1.0, 0.0, 0.0], // 0 Vertical: reads top[0..4] only
    [1.0, 0.0, 0.0, 0.0], // 1 Horizontal: reads left[0..4] only
    [0.5, 0.5, 0.0, 0.0], // 2 DC: averages top + left
    [0.0, 0.7, 0.5, 0.0], // 3 DiagDownLeft: top + top-right
    [0.6, 0.6, 0.0, 0.4], // 4 DiagDownRight: top + left + top-left
    [0.3, 0.8, 0.0, 0.3], // 5 VerticalRight: heavy top, some left + TL
    [0.8, 0.3, 0.0, 0.3], // 6 HorizontalDown: heavy left, some top + TL
    [0.0, 0.8, 0.5, 0.0], // 7 VerticalLeft: top + top-right
    [1.0, 0.0, 0.0, 0.0], // 8 HorizontalUp: left only
];

/// Phase 2d — per-mode propagation attenuation.
///
/// How much an error entering a block via the node is *preserved* when
/// passed onward to that block's downstream neighbours. This is distinct
/// from [`DRIFT_WEIGHTS`], which only measures how sensitive the block's
/// *own* prediction is to its neighbours.
///
/// Rationale:
/// * Modes 0 / 1 / 8 (Vertical, Horizontal, Horizontal-Up) directly copy
///   a single neighbour sample into each output — an error preserved at
///   full strength. Factor = 1.0.
/// * Mode 2 (DC) averages up to 8 neighbour samples into a constant, so
///   a single-sample error gets diluted to 1/N of its input. Factor = 0.5
///   to model the "dilution through averaging" effect.
/// * Modes 3 / 4 / 5 / 6 / 7 (diagonals + vertical/horizontal variants)
///   use three-tap (1,2,1)/4 filters over the neighbour samples, which
///   smooth the input but not as aggressively as DC. Factors around
///   0.75..0.95 depending on how much of the input ends up on the output.
///
/// Applied multiplicatively per-hop inside [`forward_drift`]: when the
/// walker visits a block with mode M, the incoming weight attenuates by
/// `PROPAGATION_ATTENUATION[M]` before being propagated to that block's
/// own forward neighbours. The global `decay_per_hop` is kept as a
/// secondary per-hop factor (captures non-mode-specific decay from e.g.
/// deblocking or quantisation rounding).
static PROPAGATION_ATTENUATION: [f32; 9] = [
    1.00, // 0 Vertical: strict copy
    1.00, // 1 Horizontal: strict copy
    0.50, // 2 DC: 1/N averaging dilution
    0.80, // 3 DiagDownLeft: three-tap filter
    0.75, // 4 DiagDownRight: three-tap filter
    0.85, // 5 VerticalRight: mix of copy + three-tap
    0.85, // 6 HorizontalDown: mix of copy + three-tap
    0.80, // 7 VerticalLeft: three-tap filter
    0.95, // 8 HorizontalUp: mostly copy, slight smoothing
];

/// Lookup table: for a candidate block `B` at frame position `(bx, by)`,
/// what block is its forward neighbour that sees `B` via `relation`?
///
/// Returns `Some((nx, ny, neighbour_relation_to_current))` where:
/// * `(nx, ny)` are the neighbour's 4×4-block coordinates in the frame
/// * `neighbour_relation_to_current` is how `B` looks *from* the
///   neighbour's perspective — the index into `DRIFT_WEIGHTS`.
///
/// Covers the three strongest forward-dependent relationships: the
/// block immediately below (which sees `B` as Top), to the right (Left),
/// and diagonally down-right (Top-Left). The top-right propagation is
/// omitted from this table because the forward block at
/// `(bx+1, by-1)` sees `B` as its top-LEFT, which only modes 4, 5, 6
/// consume — captured via the down-right diagonal relation instead.
fn forward_successors(
    bx: usize,
    by: usize,
    width_in_4x4: usize,
    height_in_4x4: usize,
) -> [Option<(usize, usize, usize)>; 3] {
    let mut out: [Option<(usize, usize, usize)>; 3] = [None; 3];
    // Right neighbour: sees B as Left (relation = 0).
    if bx + 1 < width_in_4x4 {
        out[0] = Some((bx + 1, by, 0));
    }
    // Below neighbour: sees B as Top (relation = 1).
    if by + 1 < height_in_4x4 {
        out[1] = Some((bx, by + 1, 1));
    }
    // Down-right diagonal: sees B as TopLeft (relation = 3).
    if bx + 1 < width_in_4x4 && by + 1 < height_in_4x4 {
        out[2] = Some((bx + 1, by + 1, 3));
    }
    out
}

/// Per-frame intra-mode map indexed by 4×4-block frame position.
///
/// Entry values:
/// * `Some(0..=8)` — an I_4x4 block with a resolved intra prediction mode.
/// * `None` — either an I_16x16 block (different prediction scheme) or a
///   P-block / unparsed block. Drift propagation stops at `None` entries.
pub struct IntraModeMap {
    modes: Vec<Option<u8>>,
    width_in_4x4: usize,
    height_in_4x4: usize,
}

impl IntraModeMap {
    /// Build an `IntraModeMap` for an I-slice from the parsed macroblocks.
    /// `mbs` is expected in raster order with length `width_in_mbs * height_in_mbs`.
    pub fn build(mbs: &[Macroblock], width_in_mbs: usize, height_in_mbs: usize) -> Self {
        let width_in_4x4 = width_in_mbs * 4;
        let height_in_4x4 = height_in_mbs * 4;
        let mut modes = vec![None; width_in_4x4 * height_in_4x4];

        for (mb_idx, mb) in mbs.iter().enumerate() {
            let mb_x = mb_idx % width_in_mbs;
            let mb_y = mb_idx / width_in_mbs;
            if !matches!(mb.mb_type, MbType::I4x4) {
                continue;
            }
            let Some(recon) = mb.recon.as_ref() else {
                continue;
            };
            for blk_idx in 0..16 {
                let (bx_in_mb, by_in_mb) = BLOCK_INDEX_TO_POS[blk_idx];
                let bx = mb_x * 4 + bx_in_mb as usize;
                let by = mb_y * 4 + by_in_mb as usize;
                modes[by * width_in_4x4 + bx] = Some(recon.intra4x4_modes[blk_idx]);
            }
        }

        Self {
            modes,
            width_in_4x4,
            height_in_4x4,
        }
    }

    pub fn mode_at(&self, bx: usize, by: usize) -> Option<u8> {
        if bx >= self.width_in_4x4 || by >= self.height_in_4x4 {
            return None;
        }
        self.modes[by * self.width_in_4x4 + bx]
    }

    pub fn width_in_4x4(&self) -> usize {
        self.width_in_4x4
    }

    pub fn height_in_4x4(&self) -> usize {
        self.height_in_4x4
    }
}

/// Apply DDCA drift multipliers to a set of already-computed UNIWARD costs.
///
/// `frame_positions` and `base_costs` are parallel slices of length N
/// (produced by [`crate::stego::cost::h264_uniward::compute_frame_uniward_costs`]).
/// Positions with non-finite base cost are left unchanged. For luma-AC
/// positions (slots 0..=15) the drift walker computes a multiplier and
/// the returned vector is `base_cost[i] * (1 + w_drift * drift[i])`.
///
/// Chroma positions (slots 18..=25) are NOT included in drift propagation
/// — chroma has only 4 intra modes, smaller impact, and 4:2:0 subsampling
/// already dampens their downstream effect. Their base costs pass through
/// unchanged.
pub fn apply_drift_multipliers(
    frame_positions: &[FramePosition<'_>],
    base_costs: &[f32],
    modes: &IntraModeMap,
    width_in_mbs: usize,
    params: &DdcaParams,
) -> Vec<f32> {
    debug_assert_eq!(frame_positions.len(), base_costs.len());

    frame_positions
        .iter()
        .zip(base_costs.iter())
        .map(|(fp, &cost)| {
            if !cost.is_finite() {
                return cost;
            }
            // Luma-only drift propagation; chroma and DC slots bypass.
            if fp.within_mb_block_idx >= 16 {
                return cost;
            }
            // Sign-bit flips at positions with extreme |coeff| already
            // contribute high direct cost via UNIWARD's 2*|coeff|
            // multiplier; an extra drift boost would double-count. Skip
            // them to keep the multiplier bounded.
            if matches!(fp.pos.domain, EmbedDomain::LevelSuffixSign)
                && fp.pos.coeff_value.unsigned_abs() > 4
            {
                return cost;
            }

            let (bx_in_mb, by_in_mb) = BLOCK_INDEX_TO_POS[fp.within_mb_block_idx];
            let mb_x = fp.mb_idx % width_in_mbs;
            let mb_y = fp.mb_idx / width_in_mbs;
            let bx = mb_x * 4 + bx_in_mb as usize;
            let by = mb_y * 4 + by_in_mb as usize;

            let drift = forward_drift(bx, by, modes, params);
            cost * (1.0 + params.w_drift * drift)
        })
        .collect()
}

/// BFS-style walk of the forward intra-prediction dependency chain starting
/// at block `(bx, by)`. At each visited block we look up its mode and add
/// `decay^hops × DRIFT_WEIGHTS[mode][relation]` to the drift score.
///
/// The traversal is bounded by `params.max_hops` and by a visited-set to
/// avoid revisiting blocks that would contribute twice through different
/// paths.
fn forward_drift(bx: usize, by: usize, modes: &IntraModeMap, params: &DdcaParams) -> f32 {
    let width = modes.width_in_4x4();
    let height = modes.height_in_4x4();
    let n = width * height;
    let mut visited = vec![false; n];
    visited[by * width + bx] = true;

    // Queue of (bx, by, hops_from_source, incoming_weight, relation_to_predecessor).
    // `relation_to_predecessor` is how the visited block looks back at its
    // immediate predecessor in the walk — this is what DRIFT_WEIGHTS needs.
    let mut queue: Vec<(usize, usize, u8, f32, usize)> = Vec::with_capacity(8);
    for succ in forward_successors(bx, by, width, height).iter().flatten() {
        queue.push((succ.0, succ.1, 1, 1.0, succ.2));
    }

    let mut drift = 0.0f32;
    while let Some((nx, ny, hops, incoming, relation)) = queue.pop() {
        let idx = ny * width + nx;
        if visited[idx] {
            continue;
        }
        visited[idx] = true;
        let Some(mode) = modes.mode_at(nx, ny) else {
            continue;
        };
        let mode_weight = DRIFT_WEIGHTS[mode as usize][relation];
        drift += incoming * mode_weight;

        if hops >= params.max_hops {
            continue;
        }
        // Phase 2d: attenuate the outgoing weight by both the global
        // per-hop decay AND the mode-specific propagation factor. A chain
        // of DC blocks dilutes ~2× faster than a chain of Vertical blocks
        // at the same hop distance.
        let mode_attenuation = PROPAGATION_ATTENUATION[mode as usize];
        let next_incoming = incoming * params.decay_per_hop * mode_attenuation;
        if next_incoming < 0.01 {
            continue;
        }
        for succ in forward_successors(nx, ny, width, height).iter().flatten() {
            queue.push((succ.0, succ.1, hops + 1, next_incoming, succ.2));
        }
    }

    drift
}

/// Per-I-frame reference weight map used by Phase 2c inter-frame drift.
///
/// For each 4×4 block of the I-frame's pixel grid, stores a cumulative
/// weight representing how much downstream P-frame motion compensation
/// references this block's samples. Built incrementally as the pipeline
/// parses successive P-frames between two I-frames via
/// [`InterFrameRefMap::accumulate_mv_field`].
///
/// Units: the weight for a block is approximately the number of 4×4
/// P-block reference regions that overlap it, scaled by
/// `inter_frame_decay ^ (p_frame_distance - 1)`. A value of 3.0 means
/// "three P-blocks' reference windows land on this region"; a value
/// of 0.5 means "one block, two frames out".
pub struct InterFrameRefMap {
    ref_counts: Vec<f32>,
    width_in_4x4: usize,
    height_in_4x4: usize,
}

impl InterFrameRefMap {
    pub fn new(width_in_4x4: usize, height_in_4x4: usize) -> Self {
        Self {
            ref_counts: vec![0.0; width_in_4x4 * height_in_4x4],
            width_in_4x4,
            height_in_4x4,
        }
    }

    pub fn width_in_4x4(&self) -> usize {
        self.width_in_4x4
    }

    pub fn height_in_4x4(&self) -> usize {
        self.height_in_4x4
    }

    pub fn ref_count(&self, bx: usize, by: usize) -> f32 {
        if bx >= self.width_in_4x4 || by >= self.height_in_4x4 {
            return 0.0;
        }
        self.ref_counts[by * self.width_in_4x4 + bx]
    }

    /// Absorb one P-slice MB's motion vectors into the reference count map.
    ///
    /// `mv_field` is the per-4×4 MV grid for a single P-MB at `(p_mb_x, p_mb_y)`
    /// in MB raster coordinates. `decay` weights the contribution by how far
    /// the P-frame is from the anchor I-frame.
    ///
    /// For each 4×4 P-block with a valid `ref_idx`, translate its MV to the
    /// I-frame's 4×4-block coordinates (integer-pel granularity — sub-pel
    /// interpolation is deferred to a later sub-phase) and distribute the
    /// 4×4 reference area across the 1–4 I-frame blocks it straddles, with
    /// weight proportional to the overlap area.
    pub fn accumulate_mv_field(
        &mut self,
        mv_field: &MvField,
        p_mb_x: usize,
        p_mb_y: usize,
        decay: f32,
    ) {
        for blk_idx in 0..16usize {
            let bx_in_mb = blk_idx % 4;
            let by_in_mb = blk_idx / 4;
            let ref_idx = mv_field.ref_idx[blk_idx];
            if ref_idx < 0 {
                // Intra block inside a P-slice — no MV to consume.
                continue;
            }
            let mv = mv_field.mvs[blk_idx];
            // Current P-block in pixel coords (4×4 block => 4 px per 4×4 unit).
            let p_pixel_x = (p_mb_x * 16 + bx_in_mb * 4) as i32;
            let p_pixel_y = (p_mb_y * 16 + by_in_mb * 4) as i32;
            // MV in quarter-pel; reduce to integer-pel for this v1.
            let mv_px_x = (mv.mv_x as i32) >> 2;
            let mv_px_y = (mv.mv_y as i32) >> 2;
            let ref_pixel_x = p_pixel_x + mv_px_x;
            let ref_pixel_y = p_pixel_y + mv_px_y;

            // Identify the 1..4 I-frame 4×4 blocks this 4×4 reference area
            // overlaps, and add area-weighted contributions.
            let ref_block_x = ref_pixel_x.div_euclid(4);
            let ref_block_y = ref_pixel_y.div_euclid(4);
            let sub_x = ref_pixel_x.rem_euclid(4);
            let sub_y = ref_pixel_y.rem_euclid(4);
            for dy in 0..2i32 {
                for dx in 0..2i32 {
                    let bx2 = ref_block_x + dx;
                    let by2 = ref_block_y + dy;
                    if bx2 < 0
                        || by2 < 0
                        || bx2 as usize >= self.width_in_4x4
                        || by2 as usize >= self.height_in_4x4
                    {
                        continue;
                    }
                    let w = if dx == 0 { 4 - sub_x } else { sub_x };
                    let h = if dy == 0 { 4 - sub_y } else { sub_y };
                    if w <= 0 || h <= 0 {
                        continue;
                    }
                    let area = (w * h) as f32;
                    let weight = (area / 16.0) * decay;
                    let idx = by2 as usize * self.width_in_4x4 + bx2 as usize;
                    self.ref_counts[idx] += weight;
                }
            }
        }
    }
}

/// Apply Phase 2c inter-frame drift multipliers to the already-adjusted
/// costs produced by [`apply_drift_multipliers`] (Phase 2b). For each
/// luma-AC I-frame position, look up the cumulative P-frame reference
/// weight at its 4×4 block position and multiply:
///   `cost_out = cost_in * (1 + w_inter_drift * ref_count)`
///
/// Chroma, DC, and non-finite positions pass through unchanged.
pub fn apply_inter_frame_drift(
    frame_positions: &[FramePosition<'_>],
    base_costs: &[f32],
    ref_map: &InterFrameRefMap,
    width_in_mbs: usize,
    params: &DdcaParams,
) -> Vec<f32> {
    debug_assert_eq!(frame_positions.len(), base_costs.len());

    frame_positions
        .iter()
        .zip(base_costs.iter())
        .map(|(fp, &cost)| {
            if !cost.is_finite() {
                return cost;
            }
            if fp.within_mb_block_idx >= 16 {
                return cost;
            }
            let (bx_in_mb, by_in_mb) = BLOCK_INDEX_TO_POS[fp.within_mb_block_idx];
            let mb_x = fp.mb_idx % width_in_mbs;
            let mb_y = fp.mb_idx / width_in_mbs;
            let bx = mb_x * 4 + bx_in_mb as usize;
            let by = mb_y * 4 + by_in_mb as usize;
            let refs = ref_map.ref_count(bx, by);
            cost * (1.0 + params.w_inter_drift * refs)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::cavlc::EmbeddablePosition;

    fn make_modes(w_mbs: usize, h_mbs: usize, fill: u8) -> IntraModeMap {
        let w4 = w_mbs * 4;
        let h4 = h_mbs * 4;
        IntraModeMap {
            modes: vec![Some(fill); w4 * h4],
            width_in_4x4: w4,
            height_in_4x4: h4,
        }
    }

    #[test]
    fn drift_zero_when_all_modes_are_dc() {
        // Mode 2 DC has 0.5 weight on both Left and Top. Non-zero drift expected.
        let modes = make_modes(2, 2, 2);
        let d = forward_drift(0, 0, &modes, &DdcaParams::default());
        assert!(d > 0.0, "DC modes should propagate some drift");
    }

    #[test]
    fn drift_zero_at_frame_corner_with_no_successors() {
        // Single 4x4 block frame — no successors.
        let w4 = 4;
        let h4 = 4;
        let modes = IntraModeMap {
            modes: vec![None; w4 * h4],
            width_in_4x4: w4,
            height_in_4x4: h4,
        };
        // Bottom-right corner has no right/below/diag neighbours.
        let d = forward_drift(3, 3, &modes, &DdcaParams::default());
        assert_eq!(d, 0.0);
    }

    #[test]
    fn vertical_mode_dominates_top_relation() {
        // Mode 0 (Vertical): full weight through Top relation.
        assert_eq!(DRIFT_WEIGHTS[0][1], 1.0);
        // Mode 0 should NOT propagate via Left.
        assert_eq!(DRIFT_WEIGHTS[0][0], 0.0);
    }

    #[test]
    fn horizontal_mode_dominates_left_relation() {
        assert_eq!(DRIFT_WEIGHTS[1][0], 1.0);
        assert_eq!(DRIFT_WEIGHTS[1][1], 0.0);
    }

    #[test]
    fn dc_chain_attenuates_faster_than_vertical_chain() {
        // Phase 2d: the per-mode propagation factor means a chain of DC
        // blocks should accumulate LESS drift than a chain of Vertical
        // blocks at the same hop depth, because DC (mode 2) dilutes the
        // error through averaging while Vertical (mode 0) passes it
        // through verbatim.
        //
        // Build two 8×8-MB (32×32 4×4-block) frames, one filled with DC
        // modes and one with Vertical modes. Compute drift from the
        // top-left block of each and compare.
        let dc_modes = make_modes(8, 8, 2); // all DC
        let vertical_modes = make_modes(8, 8, 0); // all Vertical

        let params = DdcaParams::default();
        let dc_drift = forward_drift(0, 0, &dc_modes, &params);
        let v_drift = forward_drift(0, 0, &vertical_modes, &params);

        assert!(
            v_drift > dc_drift,
            "Vertical-chain drift ({v_drift}) should exceed DC-chain drift ({dc_drift})"
        );

        // Also verify the propagation factor values themselves.
        assert_eq!(PROPAGATION_ATTENUATION[0], 1.00); // Vertical
        assert_eq!(PROPAGATION_ATTENUATION[1], 1.00); // Horizontal
        assert_eq!(PROPAGATION_ATTENUATION[2], 0.50); // DC
        // DC must be notably smaller than copy-style modes.
        assert!(PROPAGATION_ATTENUATION[2] < PROPAGATION_ATTENUATION[0] * 0.7);
    }

    #[test]
    fn apply_drift_multipliers_leaves_chroma_unchanged() {
        let pos = EmbeddablePosition {
            raw_byte_offset: 0,
            bit_offset: 0,
            domain: EmbedDomain::T1Sign,
            scan_pos: 5,
            coeff_value: 1,
            ep_conflict: false,
            block_idx: 18, // chroma slot
            frame_idx: 0,
            mb_idx: 0,
        };
        let positions = vec![FramePosition {
            pos: &pos,
            mb_idx: 0,
            within_mb_block_idx: 18,
            qp_cb: 26,
            qp_cr: 26,
        }];
        let base_costs = vec![100.0f32];
        let modes = make_modes(1, 1, 0);
        let out = apply_drift_multipliers(&positions, &base_costs, &modes, 1, &DdcaParams::default());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 100.0, "chroma position cost must pass through unchanged");
    }

    #[test]
    fn apply_drift_multipliers_boosts_luma_cost() {
        // Luma position in an all-DC intra frame → drift > 0 → cost increases.
        let pos = EmbeddablePosition {
            raw_byte_offset: 0,
            bit_offset: 0,
            domain: EmbedDomain::T1Sign,
            scan_pos: 5,
            coeff_value: 1,
            ep_conflict: false,
            block_idx: 0, // luma slot 0
            frame_idx: 0,
            mb_idx: 0,
        };
        let positions = vec![FramePosition {
            pos: &pos,
            mb_idx: 0,
            within_mb_block_idx: 0,
            qp_cb: 26,
            qp_cr: 26,
        }];
        let base_costs = vec![100.0f32];
        // 2×2-MB frame, all DC modes.
        let modes = make_modes(2, 2, 2);
        let out = apply_drift_multipliers(&positions, &base_costs, &modes, 2, &DdcaParams::default());
        assert!(out[0] > 100.0, "luma cost should be boosted by drift; got {}", out[0]);
    }

    #[test]
    fn inter_frame_ref_map_accumulates_zero_mv_at_current_block() {
        // A P-block with zero MV references its own position — the
        // reference region aligns exactly with the I-block at the same
        // spatial coordinates. ref_count at that block should be 1.0.
        let mut map = InterFrameRefMap::new(8, 8); // 32×32 pixels
        let mut mv_field = MvField::default();
        for i in 0..16 {
            mv_field.ref_idx[i] = 0; // all active
        }
        // Accumulate as if this is an MB at (1, 1) with zero MVs.
        map.accumulate_mv_field(&mv_field, 1, 1, 1.0);
        // MB at (1, 1) covers 4×4 blocks (4..8, 4..8). Each of the 16
        // blocks of the MB contributes a 4×4 reference area at its own
        // position → ref_count = 1.0 (full 16/16 overlap) per block.
        for by in 4..8 {
            for bx in 4..8 {
                assert!(
                    (map.ref_count(bx, by) - 1.0).abs() < 1e-5,
                    "zero MV should produce unit self-reference, got {} at ({bx},{by})",
                    map.ref_count(bx, by)
                );
            }
        }
        // Neighbour I-blocks outside the MB's region should stay at zero.
        assert_eq!(map.ref_count(0, 0), 0.0);
        assert_eq!(map.ref_count(2, 3), 0.0);
    }

    #[test]
    fn inter_frame_ref_map_decay_applies() {
        // The same reference at decay=0.5 should produce half the weight.
        let mut map = InterFrameRefMap::new(4, 4);
        let mut mv_field = MvField::default();
        for i in 0..16 {
            mv_field.ref_idx[i] = 0;
        }
        map.accumulate_mv_field(&mv_field, 0, 0, 0.5);
        for by in 0..4 {
            for bx in 0..4 {
                assert!((map.ref_count(bx, by) - 0.5).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn inter_frame_drift_boosts_cost_where_references_exist() {
        // Build an I-frame-local position with a small base cost, and a
        // ref map that says this block is referenced by many P-blocks.
        let pos = EmbeddablePosition {
            raw_byte_offset: 0,
            bit_offset: 0,
            domain: EmbedDomain::T1Sign,
            scan_pos: 5,
            coeff_value: 1,
            ep_conflict: false,
            block_idx: 0,
            frame_idx: 0,
            mb_idx: 0,
        };
        let positions = vec![FramePosition {
            pos: &pos,
            mb_idx: 0,
            within_mb_block_idx: 0,
            qp_cb: 26,
            qp_cr: 26,
        }];
        let base_costs = vec![100.0f32];
        let mut ref_map = InterFrameRefMap::new(4, 4);
        // Manually set a ref weight at (0, 0).
        ref_map.ref_counts[0] = 4.0; // "4 P-blocks point here"
        let out = apply_inter_frame_drift(&positions, &base_costs, &ref_map, 1, &DdcaParams::default());
        // Expected: 100 × (1 + 0.25 × 4.0) = 200.
        assert!(
            (out[0] - 200.0).abs() < 1e-3,
            "inter-frame drift should boost ~2x at 4 refs; got {}",
            out[0]
        );
    }

    #[test]
    fn infinite_base_cost_remains_infinite() {
        let pos = EmbeddablePosition {
            raw_byte_offset: 0,
            bit_offset: 0,
            domain: EmbedDomain::T1Sign,
            scan_pos: 0, // DC, would be WET anyway
            coeff_value: 1,
            ep_conflict: false,
            block_idx: 0,
            frame_idx: 0,
            mb_idx: 0,
        };
        let positions = vec![FramePosition {
            pos: &pos,
            mb_idx: 0,
            within_mb_block_idx: 0,
            qp_cb: 26,
            qp_cr: 26,
        }];
        let base_costs = vec![f32::INFINITY];
        let modes = make_modes(1, 1, 0);
        let out = apply_drift_multipliers(&positions, &base_costs, &modes, 1, &DdcaParams::default());
        assert!(out[0].is_infinite());
    }
}
