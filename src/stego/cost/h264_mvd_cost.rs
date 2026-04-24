// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 3b MVD cost model for H.264 motion-vector-difference suffix-LSB
//! embedding.
//!
//! Replaces the Phase 3a flat cost (`MVD_FLAT_COST = 1.0`) with a model
//! that borrows the AoSO ("Added Optimality Score Offset") literature's
//! intuition:
//!
//! * LSB flips on small MVDs are statistically suspicious — near-zero MVDs
//!   sit on the predictor's "hot spot" where any ±1 shift pops out as a
//!   break in the otherwise tight predictor-fit distribution. Avoid.
//! * LSB flips on large MVDs are quiet — the absolute value |mvd ± 1| is
//!   indistinguishable from |mvd| in any first-order histogram. Prefer.
//! * Flips on MBs that already have high residual energy (poor predictor)
//!   add distortion that the surrounding noise masks. Prefer.
//! * Flips on MBs with low residual energy (well-predicted flat regions)
//!   are the most exposed. Avoid.
//!
//! The combined cost multiplier therefore:
//!
//! * DECREASES as |mvd| grows (larger denominator → smaller cost).
//! * INCREASES as optimality grows (well-predicted → high cost).
//!
//! Where `optimality ∈ [0, 1]` is a proxy for "how close the MB's residual
//! is to zero". For Phase 3b we use `1 / (1 + residual_energy)` computed
//! from the sum of the MB's luma AC total-coeff proxies — a side-channel
//! measurement already wired into the pipeline. A full SAD-based optimality
//! (that requires reference-frame pixel reconstruction) is deferred.
//!
//! Final formula (for a single MVD suffix-LSB position):
//! ```text
//! cost(mvd, residual) =
//!     1.0 / (1.0 + MAG_WEIGHT · mvd²)
//!   · (1.0 + OPTIMALITY_WEIGHT · optimality)
//!   · temporal_weight
//!   · i_frame_mult
//!
//! optimality = 1.0 / (1.0 + residual_energy)
//! ```

/// Tunable weights for the Phase 3b MVD cost model.
///
/// Defaults were picked for the quarter-pel MVD magnitudes observed on real
/// 1080p H.264 Baseline (typical range ±1..±20) and the block_ac_energies
/// proxy we feed in (values typically 0..100 per luma block, 0..~1,600 per
/// MB when summed over slots 0..15).
#[derive(Debug, Clone, Copy)]
pub struct MvdCostParams {
    /// Scale factor on `|mvd|²`. Larger values make the cost drop off faster
    /// with MVD magnitude, concentrating flips on large MVDs.
    pub mag_weight: f32,
    /// Scale factor on `optimality`. Larger values penalise flips on
    /// well-predicted MBs more strongly.
    pub optimality_weight: f32,
}

impl Default for MvdCostParams {
    fn default() -> Self {
        Self {
            mag_weight: 0.1,
            optimality_weight: 2.0,
        }
    }
}

/// Compute the Phase 3b MVD suffix-LSB embedding cost.
///
/// * `mvd_value` — signed MVD component value in quarter-pel units, captured
///   into `EmbeddablePosition::coeff_value` by
///   [`crate::codec::h264::mv::parse_mv_field`]. Must be non-zero (zero MVDs
///   have no suffix and never produce embeddable positions).
/// * `mb_residual_energy` — sum of the MB's 16 luma-AC block energies
///   (entries `[mb_idx*26 .. mb_idx*26+16]` of the pipeline's
///   `block_ac_energies` vector). Captures "how much the predictor failed",
///   used as the optimality proxy.
/// * `temporal_weight` — exp-decay weight from `gop_position` (shared with
///   the coefficient cost path).
/// * `i_frame_mult` — I-frame penalty multiplier (shared).
/// * `params` — tunable weights (see [`MvdCostParams`]).
#[inline]
pub fn compute_mvd_cost(
    mvd_value: i32,
    mb_residual_energy: f32,
    temporal_weight: f32,
    i_frame_mult: f32,
    params: &MvdCostParams,
) -> f32 {
    let mvd_sq = (mvd_value as f32) * (mvd_value as f32);
    let magnitude_factor = 1.0 / (1.0 + params.mag_weight * mvd_sq);
    // `optimality = 1 / (1 + residual)` — a monotonically-decreasing proxy.
    // Well-predicted MBs (residual ≈ 0) → optimality ≈ 1 → high cost.
    // Poorly-predicted MBs (residual large) → optimality ≈ 0 → low cost.
    let optimality = 1.0 / (1.0 + mb_residual_energy.max(0.0));
    let optimality_factor = 1.0 + params.optimality_weight * optimality;
    magnitude_factor * optimality_factor * temporal_weight * i_frame_mult
}

/// Sum the 16 luma-AC block energies for a single MB from the pipeline's
/// per-block energy vector. Slots 0..=15 hold luma AC; 16..=25 are chroma
/// DC / AC. Returns 0.0 when the MB's slots are outside the vector
/// (defensive guard).
#[inline]
pub fn mb_luma_residual_energy(block_ac_energies: &[f32], mb_idx: u32) -> f32 {
    let start = mb_idx as usize * 26;
    let end = start + 16;
    if end <= block_ac_energies.len() {
        block_ac_energies[start..end].iter().copied().sum()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> MvdCostParams {
        MvdCostParams::default()
    }

    #[test]
    fn cost_decreases_as_mvd_magnitude_increases() {
        let p = default_params();
        // Hold everything else constant, sweep mvd magnitude.
        let c1 = compute_mvd_cost(1, 10.0, 1.0, 1.0, &p);
        let c2 = compute_mvd_cost(2, 10.0, 1.0, 1.0, &p);
        let c5 = compute_mvd_cost(5, 10.0, 1.0, 1.0, &p);
        let c10 = compute_mvd_cost(10, 10.0, 1.0, 1.0, &p);
        assert!(c1 > c2, "|mvd|=1 should cost more than |mvd|=2");
        assert!(c2 > c5, "|mvd|=2 should cost more than |mvd|=5");
        assert!(c5 > c10, "|mvd|=5 should cost more than |mvd|=10");
    }

    #[test]
    fn cost_symmetric_in_mvd_sign() {
        let p = default_params();
        let c_pos = compute_mvd_cost(3, 10.0, 1.0, 1.0, &p);
        let c_neg = compute_mvd_cost(-3, 10.0, 1.0, 1.0, &p);
        assert_eq!(c_pos, c_neg);
    }

    #[test]
    fn cost_decreases_as_residual_energy_grows() {
        // Higher residual ⇒ lower optimality ⇒ cheaper to flip.
        let p = default_params();
        let c_low = compute_mvd_cost(3, 0.0, 1.0, 1.0, &p);
        let c_mid = compute_mvd_cost(3, 10.0, 1.0, 1.0, &p);
        let c_high = compute_mvd_cost(3, 100.0, 1.0, 1.0, &p);
        assert!(c_low > c_mid);
        assert!(c_mid > c_high);
    }

    #[test]
    fn cost_factors_temporal_weight_and_i_frame() {
        let p = default_params();
        let base = compute_mvd_cost(3, 10.0, 1.0, 1.0, &p);
        let halved = compute_mvd_cost(3, 10.0, 0.5, 1.0, &p);
        let doubled_iframe = compute_mvd_cost(3, 10.0, 1.0, 2.0, &p);
        // Scaling is exactly multiplicative on both factors.
        assert!((halved - base * 0.5).abs() < 1e-6);
        assert!((doubled_iframe - base * 2.0).abs() < 1e-6);
    }

    #[test]
    fn cost_matches_reference_formula_at_defaults() {
        // mvd=3, residual=10, temporal=1, i_frame=1, mag_weight=0.1,
        // optimality_weight=2.0.
        // optimality = 1 / 11 ≈ 0.09091
        // magnitude_factor = 1 / (1 + 0.1 * 9) = 1 / 1.9 ≈ 0.52632
        // optimality_factor = 1 + 2 * 0.09091 = 1.18182
        // cost = 0.52632 * 1.18182 ≈ 0.62201
        let p = default_params();
        let c = compute_mvd_cost(3, 10.0, 1.0, 1.0, &p);
        assert!((c - 0.62201).abs() < 1e-4, "got {c}");
    }

    #[test]
    fn cost_stays_finite_for_extreme_inputs() {
        let p = default_params();
        // Huge MVD, huge residual — both push cost toward zero but it must stay finite.
        let c = compute_mvd_cost(1000, 1e9, 1.0, 1.0, &p);
        assert!(c.is_finite());
        assert!(c >= 0.0);
    }

    #[test]
    fn cost_zero_residual_gives_maximum_optimality_factor() {
        let p = default_params();
        // residual=0 → optimality=1 → optimality_factor = 1 + 2 = 3.
        let c_hot = compute_mvd_cost(3, 0.0, 1.0, 1.0, &p);
        let magnitude = 1.0 / (1.0 + 0.1 * 9.0); // = 0.5263
        let expected = magnitude * 3.0;
        assert!((c_hot - expected).abs() < 1e-4);
    }

    #[test]
    fn mb_luma_residual_energy_sums_slots_zero_through_fifteen() {
        // 2 MBs × 26 slots; fill MB 0 slots 0..15 with 1.0, slots 16..25 with
        // 99.0 (should NOT be counted). MB 1 all zeros.
        let mut energies = vec![0.0f32; 2 * 26];
        for slot in 0..16 {
            energies[slot] = 1.0;
        }
        for slot in 16..26 {
            energies[slot] = 99.0;
        }
        assert!(
            (mb_luma_residual_energy(&energies, 0) - 16.0).abs() < 1e-5,
            "expected 16.0, got {}",
            mb_luma_residual_energy(&energies, 0)
        );
        assert_eq!(mb_luma_residual_energy(&energies, 1), 0.0);
    }

    #[test]
    fn mb_luma_residual_energy_out_of_bounds_returns_zero() {
        let energies = vec![1.0f32; 10]; // too short for even one MB
        assert_eq!(mb_luma_residual_energy(&energies, 0), 0.0);
        assert_eq!(mb_luma_residual_energy(&energies, 99), 0.0);
    }
}
