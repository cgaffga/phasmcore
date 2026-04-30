// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// UNIWARD-analog per-position cost model for encode-time CABAC stego
// (Phase 6D.4).
//
// Assigns a non-negative `f32` distortion cost to each candidate
// stego position so the STC layer can pick low-cost positions when
// the message demands fewer flips than the cover capacity.
//
// The cost model has three layers, applied multiplicatively:
//
//   1. **Domain base cost**: how visually disruptive a flip in this
//      domain is, intrinsically. Sign flips (CoeffSignBypass /
//      MvdSignBypass) preserve magnitude → low cost. Suffix LSB
//      flips (CoeffSuffixLsb / MvdSuffixLsb) change magnitude by
//      ±1 → higher cost (proportional to magnitude²; UNIWARD-analog).
//
//   2. **Intra drift factor**: how much a coefficient flip leaks
//      into neighboring blocks via intra-prediction. Stubbed at 1.0
//      for 6D.4 (full implementation in 6D.7 once MB-level neighbor
//      info is plumbed through the three-pass driver).
//
//   3. **Inter drift factor**: how many subsequent P-frames reuse
//      this MB as a reference + per-MB usage probability. Stubbed
//      at 1.0 for 6D.4 (needs encoder-side reference list info,
//      6D.7 territory).
//
// The cost model is independently calibrated per domain so STC
// can run a separate plan per domain (per the cross-domain
// orchestration in 6D.5 `DomainCover`).

use super::{EmbedDomain, MvdSlot};

/// Cost weight for a 1-bit `CoeffSignBypass` flip. Conceptually
/// proportional to `2·|coeff|² · drift_factor`, but for sign-only
/// flips the magnitude is preserved, so the visual change comes
/// purely from changing the sign of a single residual sample. Base
/// = 1.0 distortion unit, scaled by drift later.
pub const COEFF_SIGN_BASE_COST: f32 = 1.0;

/// Cost weight per |coeff|² for `CoeffSuffixLsb` flips. Each flip
/// changes magnitude by ±1, so the residual-domain L2 distortion
/// is proportional to `|2·coeff − 1|² ≈ 4·|coeff|²` in the worst
/// case. UNIWARD's distortion model uses `2·|coeff|²` as the
/// reference rate; we follow that convention.
pub const COEFF_SUFFIX_LSB_PER_MAG2: f32 = 2.0;

/// Cost weight for a 1-bit `MvdSignBypass` flip. MV sign flip
/// reflects the partition's motion vector, producing a
/// translation-symmetric prediction error. Visually disruptive on
/// textured content; we set base cost ≈ 4× sign cost (empirical
/// guess matched to UNIWARD's MV-cost paper, refine in 6D.4
/// calibration).
pub const MVD_SIGN_BASE_COST: f32 = 4.0;

/// Cost weight per |mvd|² for `MvdSuffixLsb` flips. Same UNIWARD
/// convention as coefficient suffix LSB, scaled to MV magnitudes
/// (which can be much larger than coefficient values). Base
/// distortion per flip ≈ 2·|mvd|².
pub const MVD_SUFFIX_LSB_PER_MAG2: f32 = 2.0;

/// "Wet cost" — never embed at this position. Used by validity
/// filters and STC's `f32::INFINITY`-cost protection.
pub const WET_COST: f32 = f32::INFINITY;

/// Per-position cost computation context. For 6D.4 this only
/// carries the MB position; future fields (intra neighbor info,
/// inter ref counts) land in 6D.7 when plumbed through the
/// three-pass driver.
///
/// **Default impl is `new(0, 0)` — drift factors initialise to 1.0**,
/// NOT a derived `#[derive(Default)]` (which would zero the factors
/// and silently produce zero-cost plans, making STC pick arbitrary
/// positions). Caught in 6D.9 review.
#[derive(Copy, Clone, Debug)]
pub struct PositionCostCtx {
    pub frame_idx: u32,
    pub mb_addr: u32,
    /// Multiplicative drift factor from intra prediction
    /// sensitivity. Range 1.0..3.0; 1.0 = no extra drift cost.
    /// 6D.4 stub default: 1.0.
    pub intra_drift_factor: f32,
    /// Multiplicative drift factor from inter prediction reuse.
    /// Range 1.0..2.0; 1.0 = end-of-GOP MB (no future references).
    /// 6D.4 stub default: 1.0.
    pub inter_drift_factor: f32,
}

impl Default for PositionCostCtx {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl PositionCostCtx {
    pub fn new(frame_idx: u32, mb_addr: u32) -> Self {
        Self {
            frame_idx, mb_addr,
            intra_drift_factor: 1.0,
            inter_drift_factor: 1.0,
        }
    }

    /// Net drift factor (multiplicative composition of intra + inter).
    #[inline]
    pub fn drift_factor(&self) -> f32 {
        self.intra_drift_factor * self.inter_drift_factor
    }
}

/// Cost of flipping the `CoeffSignBypass` bin for one nonzero
/// coefficient. Independent of magnitude (sign flip preserves
/// magnitude exactly).
#[inline]
pub fn coeff_sign_cost(_coeff: i32, ctx: &PositionCostCtx) -> f32 {
    COEFF_SIGN_BASE_COST * ctx.drift_factor()
}

/// Cost of flipping the `CoeffSuffixLsb` bin for one coefficient.
/// Proportional to `2·|coeff|²` (UNIWARD-analog) since the flip
/// changes |coeff| by ±1, producing residual-domain L2 distortion
/// in the surrounding neighborhood.
#[inline]
pub fn coeff_suffix_lsb_cost(coeff: i32, ctx: &PositionCostCtx) -> f32 {
    let mag = coeff.unsigned_abs() as f32;
    COEFF_SUFFIX_LSB_PER_MAG2 * mag * mag * ctx.drift_factor()
}

/// Cost of flipping the `MvdSignBypass` bin for one MVD slot.
/// Independent of MVD magnitude (sign flip is translation-symmetric).
#[inline]
pub fn mvd_sign_cost(_slot: &MvdSlot, ctx: &PositionCostCtx) -> f32 {
    MVD_SIGN_BASE_COST * ctx.drift_factor()
}

/// Cost of flipping the `MvdSuffixLsb` bin for one MVD slot.
/// Proportional to `2·|mvd|²`.
#[inline]
pub fn mvd_suffix_lsb_cost(slot: &MvdSlot, ctx: &PositionCostCtx) -> f32 {
    let mag = slot.value.unsigned_abs() as f32;
    MVD_SUFFIX_LSB_PER_MAG2 * mag * mag * ctx.drift_factor()
}

/// Compute the cost vector for a sequence of `CoeffSignBypass`
/// candidates, in the same emit order as
/// [`super::enumerate_coeff_sign_positions`].
pub fn coeff_sign_cost_vec(
    scan_coeffs: &[i32],
    start_idx: usize,
    end_idx: usize,
    ctx: &PositionCostCtx,
) -> Vec<f32> {
    let mut sig: Vec<usize> = (start_idx..=end_idx)
        .filter(|&i| scan_coeffs[i] != 0)
        .collect();
    sig.reverse();
    sig.into_iter()
        .map(|i| coeff_sign_cost(scan_coeffs[i], ctx))
        .collect()
}

/// Compute the cost vector for `CoeffSuffixLsb` candidates.
pub fn coeff_suffix_lsb_cost_vec(
    scan_coeffs: &[i32],
    start_idx: usize,
    end_idx: usize,
    ctx: &PositionCostCtx,
) -> Vec<f32> {
    let mut sig: Vec<usize> = (start_idx..=end_idx)
        .filter(|&i| scan_coeffs[i].unsigned_abs() >= 16)
        .collect();
    sig.reverse();
    sig.into_iter()
        .map(|i| coeff_suffix_lsb_cost(scan_coeffs[i], ctx))
        .collect()
}

/// Compute the cost vector for `MvdSignBypass` candidates.
pub fn mvd_sign_cost_vec(slots: &[MvdSlot], ctx: &PositionCostCtx) -> Vec<f32> {
    slots
        .iter()
        .filter(|s| s.value != 0)
        .map(|s| mvd_sign_cost(s, ctx))
        .collect()
}

/// Compute the cost vector for `MvdSuffixLsb` candidates.
pub fn mvd_suffix_lsb_cost_vec(slots: &[MvdSlot], ctx: &PositionCostCtx) -> Vec<f32> {
    slots
        .iter()
        .filter(|s| s.value.unsigned_abs() >= 9)
        .map(|s| mvd_suffix_lsb_cost(s, ctx))
        .collect()
}

/// Dispatch to the matching domain's cost function. Used by Pass 2
/// orchestrators that walk a `DomainCover` and need a single cost
/// vector per domain.
pub fn domain_base_cost(domain: EmbedDomain) -> f32 {
    match domain {
        EmbedDomain::CoeffSignBypass => COEFF_SIGN_BASE_COST,
        EmbedDomain::CoeffSuffixLsb => COEFF_SUFFIX_LSB_PER_MAG2, // per-mag² scale
        EmbedDomain::MvdSignBypass => MVD_SIGN_BASE_COST,
        EmbedDomain::MvdSuffixLsb => MVD_SUFFIX_LSB_PER_MAG2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::stego::Axis;

    fn ctx() -> PositionCostCtx {
        PositionCostCtx::new(0, 0)
    }

    #[test]
    fn coeff_sign_cost_independent_of_magnitude() {
        let c = ctx();
        let small = coeff_sign_cost(1, &c);
        let medium = coeff_sign_cost(50, &c);
        let large = coeff_sign_cost(1000, &c);
        assert_eq!(small, medium);
        assert_eq!(medium, large);
        assert!(small > 0.0);
    }

    #[test]
    fn coeff_suffix_lsb_cost_grows_with_magnitude_squared() {
        let c = ctx();
        let c16 = coeff_suffix_lsb_cost(16, &c);
        let c32 = coeff_suffix_lsb_cost(32, &c);
        let c64 = coeff_suffix_lsb_cost(64, &c);
        // Doubling magnitude → 4× cost.
        assert!((c32 - 4.0 * c16).abs() < 0.01);
        assert!((c64 - 16.0 * c16).abs() < 0.1);
    }

    #[test]
    fn coeff_suffix_lsb_cost_higher_than_sign_for_large_coeffs() {
        let c = ctx();
        let sign = coeff_sign_cost(20, &c);
        let suffix = coeff_suffix_lsb_cost(20, &c);
        assert!(suffix > sign, "suffix LSB cost must exceed sign cost for large coeffs");
        // 2 × 20² = 800, vs sign cost 1 → 800× ratio.
        assert!(suffix / sign > 100.0);
    }

    #[test]
    fn drift_factor_multiplies_costs() {
        let mut c = ctx();
        let baseline = coeff_sign_cost(5, &c);
        c.intra_drift_factor = 2.5;
        let drifted = coeff_sign_cost(5, &c);
        assert!((drifted - 2.5 * baseline).abs() < 0.01);
    }

    #[test]
    fn drift_factor_composes_intra_and_inter() {
        let mut c = ctx();
        c.intra_drift_factor = 2.0;
        c.inter_drift_factor = 1.5;
        assert!((c.drift_factor() - 3.0).abs() < 0.01);
    }

    #[test]
    fn cost_vec_matches_enumerate_order() {
        // Build a scan with mixed signs; verify cost_vec aligns
        // with the position enumeration order (reverse scan).
        let mut scan = vec![0i32; 16];
        scan[0] = 5; scan[3] = -8; scan[7] = 12;
        let costs = coeff_sign_cost_vec(&scan, 0, 15, &ctx());
        // 3 positions; sign cost is constant for sign domain.
        assert_eq!(costs.len(), 3);
        for &c in &costs {
            assert_eq!(c, COEFF_SIGN_BASE_COST);
        }
    }

    #[test]
    fn coeff_suffix_lsb_cost_vec_filters_threshold() {
        let mut scan = vec![0i32; 16];
        scan[0] = 5;     // sub-threshold
        scan[3] = 16;    // threshold
        scan[7] = -32;   // above
        let costs = coeff_suffix_lsb_cost_vec(&scan, 0, 15, &ctx());
        assert_eq!(costs.len(), 2, "only |coeff|>=16 positions");
        // costs[0] is for scan[7]=-32 (reverse scan order):
        // 2 * 32² = 2048
        assert!((costs[0] - 2048.0).abs() < 0.1);
        // costs[1] is for scan[3]=16: 2 * 16² = 512
        assert!((costs[1] - 512.0).abs() < 0.1);
    }

    #[test]
    fn mvd_costs_filter_by_threshold() {
        let slots = vec![
            MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 0 },
            MvdSlot { list: 0, partition: 0, axis: Axis::Y, value: 5 },
            MvdSlot { list: 0, partition: 1, axis: Axis::X, value: -10 },
        ];
        let sign_costs = mvd_sign_cost_vec(&slots, &ctx());
        assert_eq!(sign_costs.len(), 2, "value=0 not eligible for sign");
        let suffix_costs = mvd_suffix_lsb_cost_vec(&slots, &ctx());
        assert_eq!(suffix_costs.len(), 1, "only |mvd|>=9 eligible for suffix");
        // 2 * 10² = 200
        assert!((suffix_costs[0] - 200.0).abs() < 0.1);
    }

    #[test]
    fn domain_base_cost_dispatch() {
        assert_eq!(
            domain_base_cost(EmbedDomain::CoeffSignBypass),
            COEFF_SIGN_BASE_COST,
        );
        assert_eq!(
            domain_base_cost(EmbedDomain::MvdSignBypass),
            MVD_SIGN_BASE_COST,
        );
    }
}
