// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! CSF-weighted cost function for H.264 CAVLC steganography (Phase 1a).
//!
//! Assigns distortion costs to each embeddable bit position based on:
//! 1. Contrast Sensitivity Function (CSF) weights per zigzag position
//! 2. Texture masking (AC energy per 4x4 block)
//! 3. Temporal weighting (GOP position)
//! 4. WET rules (DC, flat blocks, EP byte conflicts)
//!
//! Phase 1b replaces the CSF lookup with full UNIWARD wavelet computation.

use crate::codec::h264::cavlc::{EmbedDomain, EmbeddablePosition};
use crate::codec::h264::slice::SliceType;
use crate::stego::cost::h264_mvd_cost::{
    compute_mvd_cost, mb_luma_residual_energy, MvdCostParams,
};

/// CSF (Contrast Sensitivity Function) weights per 4x4 zigzag scan position.
///
/// Higher frequency = lower visual sensitivity = lower cost = preferred for embedding.
/// DC (position 0) is infinity (never modify).
/// Values calibrated for the H.264 4x4 integer DCT basis functions.
const CSF_WEIGHT_4X4: [f32; 16] = [
    f32::INFINITY, // pos 0: DC — never modify
    8.0,           // pos 1: (0,1) — very visible low-freq
    8.0,           // pos 2: (1,0) — very visible low-freq
    4.0,           // pos 3: (2,0)
    4.0,           // pos 4: (1,1)
    4.0,           // pos 5: (0,2)
    2.0,           // pos 6: (0,3)
    2.0,           // pos 7: (1,2)
    2.0,           // pos 8: (2,1)
    2.0,           // pos 9: (3,0)
    1.0,           // pos 10: (3,1)
    1.0,           // pos 11: (2,2)
    1.0,           // pos 12: (1,3)
    0.5,           // pos 13: (3,2) — barely visible high-freq
    0.5,           // pos 14: (2,3) — barely visible high-freq
    0.25,          // pos 15: (3,3) — least visible
];

/// Default temporal penalty for I-frames (higher = more expensive to modify).
const I_FRAME_PENALTY: f32 = 2.0;

/// Minimum block AC energy to consider embeddable. Blocks below this threshold
/// are flat/smooth and modifications are too visible.
const MIN_AC_ENERGY: f32 = 1.0;

/// Compute costs for a list of embeddable positions within a single frame.
///
/// # Arguments
/// * `positions` — Embeddable positions from CAVLC parsing
/// * `block_ac_energies` — AC energy per 4x4 block (indexed by block_idx),
///   precomputed as `sqrt(sum(coeff[i]^2) for i > 0)` during parsing
/// * `slice_type` — I or P (affects temporal weight)
/// * `gop_position` — Frame index within GOP (0 = I-frame, 1+ = P-frames)
/// * `gop_length` — Total frames in this GOP
///
/// Returns a cost vector (f32) parallel to `positions`. Infinite cost = WET.
pub fn compute_h264_costs(
    positions: &[EmbeddablePosition],
    block_ac_energies: &[f32],
    slice_type: SliceType,
    gop_position: u32,
    gop_length: u32,
) -> Vec<f32> {
    // Phase 2: exponential decay with GOP position. Early-GOP I-frame
    // positions have many dependents (subsequent P-frames reference
    // them via MC), so their distortion compounds — we want to keep
    // them cheaper-looking than late-GOP positions. `alpha = 0.3` lands
    // roughly in the middle of the DDCA literature's 0.1..0.9 range:
    //   gop_pos=0 -> 1.0, gop_pos=1 -> 0.74, gop_pos=5 -> 0.22,
    //   gop_pos=10 -> 0.05.
    // `gop_length` is kept in the signature for API stability but no
    // longer affects the weight directly (the decay is purely a function
    // of distance from the anchor I-frame).
    const TEMPORAL_DECAY_ALPHA: f32 = 0.3;
    let temporal_weight = (-TEMPORAL_DECAY_ALPHA * gop_position as f32).exp();
    let _ = gop_length; // retained for backward-compat callers

    let i_frame_mult = if slice_type.is_intra() {
        I_FRAME_PENALTY
    } else {
        1.0
    };

    positions
        .iter()
        .map(|pos| {
            // NOTE: ep_conflict is NOT marked WET here. It's a function of
            // surrounding raw bytes which differ between cover and stego after
            // neighboring flips — marking it WET would cause encode/decode to
            // disagree on the position list. EP conflicts are handled by a
            // post-STC pass that skips flips which would create new EP bytes.

            // Phase 3b: MVD positions bypass the coefficient cost machinery
            // entirely. They carry their own sentinel `block_idx` and a
            // `scan_pos == 0` (neither meaningful for MV-domain embedding).
            // Cost = `1/(1+0.1·mvd²) · (1+2·optimality) · temporal · i_frame`
            // where optimality is a per-MB residual-energy proxy. See
            // `h264_mvd_cost.rs` for the full formula derivation.
            if pos.domain == EmbedDomain::MvdLsb {
                let mb_residual =
                    mb_luma_residual_energy(block_ac_energies, pos.mb_idx);
                return compute_mvd_cost(
                    pos.coeff_value,
                    mb_residual,
                    temporal_weight,
                    i_frame_mult,
                    &MvdCostParams::default(),
                );
            }

            // WET: I_16x16 luma DC block (sentinel). The Hadamard-DC
            // coefficients carry 16 blocks' worth of luma energy and flipping
            // them cascades into every AC block, so we never embed there.
            if pos.block_idx == u32::MAX {
                return f32::INFINITY;
            }

            // WET: DC position (scan_pos 0)
            if pos.scan_pos == 0 {
                return f32::INFINITY;
            }

            // Phase 1a: only T1Sign domain
            // Phase 1b/2: LevelSuffixMag and LevelSuffixSign also scored
            let domain_mult = match pos.domain {
                EmbedDomain::T1Sign => 2.0, // distortion = 2 (constant)
                EmbedDomain::LevelSuffixMag => 1.0, // distortion = 1
                EmbedDomain::LevelSuffixSign => 2.0 * pos.coeff_value.unsigned_abs() as f32,
                EmbedDomain::MvdLsb => unreachable!("MvdLsb handled above"),
            };

            // CSF base cost (frequency-dependent visibility)
            let csf = CSF_WEIGHT_4X4[pos.scan_pos.min(15) as usize];
            if csf.is_infinite() {
                return f32::INFINITY;
            }

            // Texture masking: high AC energy → lower cost (modifications hidden by texture)
            let ac_energy = if (pos.block_idx as usize) < block_ac_energies.len() {
                block_ac_energies[pos.block_idx as usize]
            } else {
                0.0
            };

            // WET: flat blocks (AC energy below threshold)
            if ac_energy < MIN_AC_ENERGY {
                return f32::INFINITY;
            }

            let texture_mask = 1.0 / (1.0 + ac_energy);

            // Final cost
            csf * domain_mult * texture_mask * temporal_weight * i_frame_mult
        })
        .collect()
}

/// Compute AC energy for a decoded CAVLC block (sum of squares of non-DC coefficients).
///
/// Used to build the `block_ac_energies` vector for cost computation.
pub fn block_ac_energy(coeffs: &[i32; 16]) -> f32 {
    let sum_sq: i64 = coeffs[1..]
        .iter()
        .map(|&c| (c as i64) * (c as i64))
        .sum();
    (sum_sq as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csf_weights_monotonic_high_freq() {
        // Higher zigzag positions (higher frequency) should have lower or equal CSF weight
        for i in 1..15 {
            assert!(
                CSF_WEIGHT_4X4[i] >= CSF_WEIGHT_4X4[i + 1],
                "CSF_WEIGHT_4X4[{i}]={} < CSF_WEIGHT_4X4[{}]={}",
                CSF_WEIGHT_4X4[i],
                i + 1,
                CSF_WEIGHT_4X4[i + 1]
            );
        }
    }

    #[test]
    fn csf_dc_is_wet() {
        assert!(CSF_WEIGHT_4X4[0].is_infinite());
    }

    #[test]
    fn temporal_decay_is_monotonically_decreasing() {
        // Build a valid P-slice position with AC energy in range so only the
        // temporal factor varies across calls. Compare cost at gop_pos 0..10
        // — it should shrink monotonically. Landmark values per the 0.3
        // alpha: gop_pos=5 -> temporal_weight ≈ 0.223.
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
        let positions = vec![pos];
        let ac = vec![10.0];

        let mut prev = f32::INFINITY;
        for gop_pos in 0..=10u32 {
            let costs = compute_h264_costs(&positions, &ac, SliceType::P, gop_pos, 30);
            let c = costs[0];
            assert!(c.is_finite(), "cost must be finite at gop_pos={gop_pos}");
            assert!(c < prev, "cost must strictly decrease, gop_pos={gop_pos} ({c} >= {prev})");
            prev = c;
        }

        // Spot check: cost at gop_pos=5 / cost at gop_pos=0 ≈ exp(-0.3*5) = 0.223.
        let cost_at_0 = compute_h264_costs(&positions, &ac, SliceType::P, 0, 30)[0];
        let cost_at_5 = compute_h264_costs(&positions, &ac, SliceType::P, 5, 30)[0];
        let ratio = cost_at_5 / cost_at_0;
        assert!(
            (ratio - 0.2231).abs() < 0.002,
            "gop_pos=5/0 ratio = {ratio}, expected ~0.223 (e^(-1.5))"
        );
    }

    #[test]
    fn cost_ep_conflict_no_longer_forces_wet() {
        // Historical note: ep_conflict WAS marked WET but that caused
        // encode/decode disagreement (different raw bytes → different
        // ep_conflict values). EP conflicts are now handled at flip time.
        let positions = vec![EmbeddablePosition {
            raw_byte_offset: 0,
            bit_offset: 0,
            domain: EmbedDomain::T1Sign,
            scan_pos: 15,
            coeff_value: 1,
            ep_conflict: true,
            block_idx: 0,
            frame_idx: 0,
            mb_idx: 0,
        }];
        let ac = vec![10.0];
        let costs = compute_h264_costs(&positions, &ac, SliceType::I, 0, 30);
        assert!(costs[0].is_finite(), "ep_conflict should no longer force infinite cost");
    }

    #[test]
    fn cost_dc_position_is_wet() {
        let positions = vec![EmbeddablePosition {
            raw_byte_offset: 0,
            bit_offset: 0,
            domain: EmbedDomain::T1Sign,
            scan_pos: 0, // DC!
            coeff_value: 1,
            ep_conflict: false,
            block_idx: 0,
            frame_idx: 0,
            mb_idx: 0,
        }];
        let ac = vec![10.0];
        let costs = compute_h264_costs(&positions, &ac, SliceType::I, 0, 30);
        assert!(costs[0].is_infinite());
    }

    #[test]
    fn cost_flat_block_is_wet() {
        let positions = vec![EmbeddablePosition {
            raw_byte_offset: 0,
            bit_offset: 0,
            domain: EmbedDomain::T1Sign,
            scan_pos: 15,
            coeff_value: 1,
            ep_conflict: false,
            block_idx: 0,
            frame_idx: 0,
            mb_idx: 0,
        }];
        let ac = vec![0.5]; // below MIN_AC_ENERGY
        let costs = compute_h264_costs(&positions, &ac, SliceType::I, 0, 30);
        assert!(costs[0].is_infinite());
    }

    #[test]
    fn cost_temporal_weight() {
        let pos = EmbeddablePosition {
            raw_byte_offset: 0,
            bit_offset: 0,
            domain: EmbedDomain::T1Sign,
            scan_pos: 15,
            coeff_value: 1,
            ep_conflict: false,
            block_idx: 0,
            frame_idx: 0,
            mb_idx: 0,
        };
        let ac = vec![10.0];

        // I-frame at GOP position 0 (highest temporal cost)
        let cost_i = compute_h264_costs(&[pos.clone()], &ac, SliceType::I, 0, 30)[0];
        // P-frame at GOP position 29 (lowest temporal cost)
        let cost_p = compute_h264_costs(&[pos.clone()], &ac, SliceType::P, 29, 30)[0];

        // I-frame should have higher cost due to temporal weight + I-frame penalty
        assert!(cost_i > cost_p, "I-frame cost {cost_i} should be > P-frame cost {cost_p}");
    }

    #[test]
    fn cost_texture_masking() {
        let pos = EmbeddablePosition {
            raw_byte_offset: 0,
            bit_offset: 0,
            domain: EmbedDomain::T1Sign,
            scan_pos: 15,
            coeff_value: 1,
            ep_conflict: false,
            block_idx: 0,
            frame_idx: 0,
            mb_idx: 0,
        };

        // Low texture (AC=2) should have higher cost than high texture (AC=50)
        let cost_low = compute_h264_costs(&[pos.clone()], &[2.0], SliceType::P, 1, 30)[0];
        let cost_high = compute_h264_costs(&[pos.clone()], &[50.0], SliceType::P, 1, 30)[0];
        assert!(cost_low > cost_high, "low-texture cost {cost_low} should be > high-texture cost {cost_high}");
    }

    #[test]
    fn block_ac_energy_computation() {
        let coeffs = [10, 3, -2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let energy = block_ac_energy(&coeffs);
        // sqrt(9 + 4 + 1) = sqrt(14) ≈ 3.74
        assert!((energy - 3.74).abs() < 0.1);
    }

    #[test]
    fn block_ac_energy_zero_block() {
        let coeffs = [0; 16];
        assert_eq!(block_ac_energy(&coeffs), 0.0);
    }

    #[test]
    fn block_ac_energy_dc_only() {
        let coeffs = [42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(block_ac_energy(&coeffs), 0.0); // DC excluded
    }
}
