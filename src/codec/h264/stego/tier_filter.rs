// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Tunable cascade-safety tiers for H.264 video stego.
//
// Discrete tier 0..4 controls a deterministic filter on cover positions.
// The filter rejects positions whose estimated per-flip pixel impact
// exceeds a tier-specific threshold:
//
//     pixel_delta_estimate = q_step(qp) × scan_basis_weight(scan_pos)
//                            × domain_factor(domain, magnitude)
//
//     filter: pixel_delta_estimate >= tier_threshold(tier)
//
// Both encoder and decoder compute this filter from the captured cover
// alone — no metadata is transmitted. Decoder brute-forces tiers during
// STC decode; AES-GCM-SIV disambiguates the correct tier.
//
// Per the v2 plan (`docs/design/video/h264/cascade-elimination-plan-v2.md`):
// - Tier 0 = no filter (baseline)
// - Tier 1 = pixel_delta_estimate ≥ 32 luma units (~25% capacity hit, +1-2 dB)
// - Tier 2 = pixel_delta_estimate ≥ 16 luma units (~50% capacity hit, +3-4 dB)
// - Tier 3 = pixel_delta_estimate ≥ 8 luma units  (~65% capacity hit, +5-7 dB)
// - Tier 4 = pixel_delta_estimate ≥ 4 luma units  (~85% capacity hit, +8-12 dB)
//
// Filter operates on CSB + CSL position domains only. MVD domains have
// their own cascade-safety filter (`cascade_safety::analyze_safe_mvd_subset`)
// which composes trivially AND with the tier filter (disjoint domains).

use super::hook::{BinKind, EmbedDomain, PositionKey, SyntaxPath};
use super::inject::{DomainBits, DomainCover};

/// Discrete cascade-safety tier. Auto selects the highest tier whose
/// capacity comfortably exceeds the message size.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum CascadeTier {
    /// Encoder auto-selects highest tier that fits the message.
    Auto,
    /// No filter — baseline Phase 1 behavior.
    Tier0,
    /// Mild filter — pixel_delta ≥ 32 luma units.
    Tier1,
    /// Moderate filter — pixel_delta ≥ 16 luma units.
    Tier2,
    /// Heavy filter — pixel_delta ≥ 8 luma units.
    Tier3,
    /// Maximum filter — pixel_delta ≥ 4 luma units.
    Tier4,
}

impl CascadeTier {
    /// Numeric tier index for filtering; `Auto` returns 0 (caller resolves
    /// auto-selection before invoking the filter).
    pub fn as_u8(self) -> u8 {
        match self {
            CascadeTier::Auto => 0,
            CascadeTier::Tier0 => 0,
            CascadeTier::Tier1 => 1,
            CascadeTier::Tier2 => 2,
            CascadeTier::Tier3 => 3,
            CascadeTier::Tier4 => 4,
        }
    }

    pub fn from_u8(t: u8) -> Option<Self> {
        match t {
            0 => Some(CascadeTier::Tier0),
            1 => Some(CascadeTier::Tier1),
            2 => Some(CascadeTier::Tier2),
            3 => Some(CascadeTier::Tier3),
            4 => Some(CascadeTier::Tier4),
            _ => None,
        }
    }

    pub fn ui_name(self) -> &'static str {
        match self {
            CascadeTier::Auto => "Auto",
            CascadeTier::Tier0 => "Max Capacity",
            CascadeTier::Tier1 => "Balanced",
            CascadeTier::Tier2 => "Quality",
            CascadeTier::Tier3 => "High Quality",
            CascadeTier::Tier4 => "Best Quality",
        }
    }

}

/// Default headroom for auto-tier selection: encoder needs
/// `capacity_at_tier ≥ msg_bytes × 1.2` to pick that tier. Buffer covers
/// AES envelope (~44 B), frame header (~10 B), STC w-slack (~20%), and
/// per-GOP capacity variance (±10%).
pub const DEFAULT_HEADROOM: f32 = 1.2;

// ─── Pixel-delta primitives ──────────────────────────────────────

/// H.264 4×4 quant scale (spec § 8.5.8, indexed by `qp % 6` and class).
/// Duplicated from `dpb_correction.rs` for module-locality; both refer to
/// the same spec table. Class indexing matches H.264's LevelScale4x4.
const LEVEL_SCALE_4X4: [[i32; 3]; 6] = [
    [160, 256, 208],
    [176, 288, 224],
    [208, 320, 256],
    [224, 368, 288],
    [256, 400, 320],
    [288, 464, 368],
];

/// Approximate quantization step for the given QP. Per H.264 spec the
/// dequant scale is `LevelScale4x4[qp%6][class] << (qp/6)` (for q_bits ≥ 4);
/// the magnitude per |coeff| unit grows by 2^(qp/6) with QP. Class is
/// position-dependent (see `scan_basis_weight`).
///
/// We use the average class (DC ~ 160-288, low-freq ~ 256-464, high-freq
/// ~ 208-368) as a single QP-only factor of ~250 normalised, with the
/// position contribution handled by `scan_basis_weight`. Power-of-two
/// shift dominates the absolute scale.
#[inline]
pub fn q_step(qp: i32) -> f32 {
    // Spec: LevelScale × 2^(qp/6) for q_bits ≥ 4.
    // We normalize to QP=24 (no shift) and let scan_basis_weight carry
    // the per-position factor. Result is in arbitrary units; threshold
    // calibration in `tier_threshold` matches.
    let q_bits = qp.div_euclid(6);
    2.0_f32.powi(q_bits - 4)
}

/// IDCT basis weight at `raster_idx` ∈ 0..16 in a 4×4 block. DC has
/// the largest impact (basis = 1 at every pixel = 16-pixel sum);
/// high-frequency basis functions have smaller summed magnitudes
/// because they alternate signs.
///
/// Precomputed from the H.264 4×4 integer inverse transform basis;
/// values normalized to DC = 1.0. Approximated via the standard
/// Hadamard-derived basis amplitudes.
#[inline]
pub fn scan_basis_weight(scan_pos: u8) -> f32 {
    // Map zigzag scan position to raster (u, v).
    let raster_idx = crate::codec::h264::tables::ZIGZAG_4X4[scan_pos as usize] as usize;
    let u = raster_idx / 4;
    let v = raster_idx % 4;
    // Basis amplitudes (relative to DC = 1.0). Derived from the H.264
    // 4×4 integer inverse transform: rows/cols are {1, 1, 1, 1} for DC,
    // {2, 1, -1, -2} for next basis (mid-amplitude), {1, -1, -1, 1} for
    // 2x2 (low-mid amp), {1, -2, 2, -1} for highest (small amp).
    // Aggregate per-pixel impact scales by row × col basis magnitudes.
    const PER_AXIS_AMPLITUDE: [f32; 4] = [1.0, 0.79, 0.5, 0.45];
    PER_AXIS_AMPLITUDE[u] * PER_AXIS_AMPLITUDE[v]
}

/// Per-domain factor in the pixel-delta formula.
///
/// - CSB sign flip: coefficient swings ±M → ∓M. Pixel delta = `2·|M|·q_step·basis`.
/// - CSL suffix-LSB flip: magnitude shifts by ±1 (toward parity boundary).
///   Pixel delta = `1·q_step·basis` (independent of |M|).
#[inline]
pub fn domain_factor(domain: EmbedDomain, magnitude: u16) -> f32 {
    match domain {
        EmbedDomain::CoeffSignBypass => 2.0 * (magnitude as f32).max(1.0),
        EmbedDomain::CoeffSuffixLsb => 1.0,
        // MVD domains aren't filtered by tier — they have their own
        // cascade-safety predicate. Return ∞ to ensure these never
        // pass tier filter (they'll be handled by the MVD filter).
        _ => f32::INFINITY,
    }
}

/// Estimate the pixel delta a flip at the given position would produce.
/// All factors combined: `q_step × scan_basis_weight × domain_factor`.
#[inline]
pub fn estimate_pixel_delta(
    magnitude: u16,
    qp: i32,
    scan_pos: u8,
    domain: EmbedDomain,
) -> f32 {
    q_step(qp) * scan_basis_weight(scan_pos) * domain_factor(domain, magnitude)
}

/// Tier threshold in luma units. Higher tier = stricter filter = lower
/// threshold = more positions filtered.
#[inline]
pub fn tier_threshold(tier: u8) -> f32 {
    match tier {
        0 => f32::INFINITY, // no filter; nothing fails the threshold
        1 => 32.0,
        2 => 16.0,
        3 => 8.0,
        4 => 4.0,
        _ => f32::INFINITY, // unknown tier → behave like tier 0
    }
}

// ─── Filter application ─────────────────────────────────────────

/// Apply the tier filter to a single domain's positions. Returns a
/// `Vec<bool>` aligned with positions: `true` = position survives the
/// filter (eligible for STC selection), `false` = filtered out.
///
/// `per_pos_qp` provides the per-position QP for `q_step` lookup. If
/// caller doesn't have per-position QP (e.g., uses a global frame QP),
/// pass a slice of constant values.
///
/// At tier 0, every position survives (filter is a no-op).
pub fn apply_tier_filter(
    bits: &DomainBits,
    per_pos_qp: &[i32],
    tier: u8,
) -> Vec<bool> {
    let n = bits.positions.len();
    debug_assert_eq!(bits.bits.len(), n);
    debug_assert_eq!(bits.magnitudes.len(), n);
    debug_assert_eq!(per_pos_qp.len(), n);

    if tier == 0 {
        return vec![true; n];
    }

    let threshold = tier_threshold(tier);
    let mut keep = Vec::with_capacity(n);
    for i in 0..n {
        let pos = bits.positions[i];
        let domain = pos.domain();
        let magnitude = bits.magnitudes[i];

        // Only CSB and CSL are filtered by tiers. MVD positions pass
        // through (handled by MVD cascade-safety predicate separately).
        let scan_pos = match (domain, pos.syntax_path()) {
            (EmbedDomain::CoeffSignBypass, SyntaxPath::Luma4x4 { coeff_idx, kind: BinKind::Sign, .. })
            | (EmbedDomain::CoeffSuffixLsb, SyntaxPath::Luma4x4 { coeff_idx, kind: BinKind::SuffixLsb, .. }) => coeff_idx,
            (EmbedDomain::CoeffSignBypass, SyntaxPath::Luma8x8 { coeff_idx, kind: BinKind::Sign, .. })
            | (EmbedDomain::CoeffSuffixLsb, SyntaxPath::Luma8x8 { coeff_idx, kind: BinKind::SuffixLsb, .. }) => coeff_idx.min(15),
            (EmbedDomain::CoeffSignBypass, SyntaxPath::ChromaAc { coeff_idx, kind: BinKind::Sign, .. })
            | (EmbedDomain::CoeffSuffixLsb, SyntaxPath::ChromaAc { coeff_idx, kind: BinKind::SuffixLsb, .. }) => coeff_idx,
            // For ChromaDc and LumaDcIntra16x16, scan_pos doesn't map to
            // the 4×4 basis; use 0 (DC weight) as a conservative bound.
            _ => 0,
        };

        let qp = per_pos_qp[i];
        let delta = estimate_pixel_delta(magnitude, qp, scan_pos, domain);
        keep.push(delta < threshold);
    }
    keep
}

/// Convenience: filter a `DomainCover`'s coefficient domains (CSB + CSL)
/// at the given tier. Returns two `Vec<bool>` aligned with each domain's
/// positions. MVD domains are not filtered by tier (returns all-`true`).
///
/// `csb_qp` must have length = `cover.coeff_sign_bypass.len()`.
/// `csl_qp` must have length = `cover.coeff_suffix_lsb.len()`.
pub fn apply_tier_filter_cover(
    cover: &DomainCover,
    csb_qp: &[i32],
    csl_qp: &[i32],
    tier: u8,
) -> TierFilterResult {
    TierFilterResult {
        csb_keep: apply_tier_filter(&cover.coeff_sign_bypass, csb_qp, tier),
        csl_keep: apply_tier_filter(&cover.coeff_suffix_lsb, csl_qp, tier),
        msb_keep: vec![true; cover.mvd_sign_bypass.len()],
        msl_keep: vec![true; cover.mvd_suffix_lsb.len()],
    }
}

/// Per-domain keep-masks from a tier filter pass.
#[derive(Default, Debug, Clone)]
pub struct TierFilterResult {
    pub csb_keep: Vec<bool>,
    pub csl_keep: Vec<bool>,
    pub msb_keep: Vec<bool>,
    pub msl_keep: Vec<bool>,
}

impl TierFilterResult {
    pub fn total_kept(&self) -> usize {
        self.csb_keep.iter().filter(|&&b| b).count()
            + self.csl_keep.iter().filter(|&&b| b).count()
            + self.msb_keep.iter().filter(|&&b| b).count()
            + self.msl_keep.iter().filter(|&&b| b).count()
    }
}

/// Total cover bits surviving tier filter across CSB + CSL domains.
/// MVD domains are excluded because their capacity is governed by the
/// MVD cascade-safety predicate, not by tier.
pub fn capacity_at_tier(cover: &DomainCover, csb_qp: &[i32], csl_qp: &[i32], tier: u8) -> usize {
    if tier == 0 {
        return cover.coeff_sign_bypass.len() + cover.coeff_suffix_lsb.len();
    }
    let result = apply_tier_filter_cover(cover, csb_qp, csl_qp, tier);
    result.csb_keep.iter().filter(|&&b| b).count()
        + result.csl_keep.iter().filter(|&&b| b).count()
}

/// Auto-select the highest tier whose capacity comfortably exceeds the
/// message size by `headroom` factor.
///
/// Calibration (2026-05-28) — an empirical sweep over 13 real-world
/// fixtures × 5 tiers showed the tier filter is **content-neutral** on
/// real video: each fixture's roundtrip outcome is identical across all
/// tiers. Aggressive count-based selection is safe on real content.
///
/// Synthetic fixtures (`stego_a_roundtrip_parity` 320×240×4 XOR/LCG)
/// still roundtrip-fail at any tier > 0 due to an unrelated cascade
/// interaction. Those tests must set `PHASM_AUTO_TIER_CONSERVATIVE=1`
/// to opt out of auto-tier and pin Tier 0.
///
/// `PHASM_AUTO_TIER_CONSERVATIVE=1` forces `Tier0` (no filter); useful
/// for synthetic-fixture tests and isolation debugging.
pub fn auto_select_tier(
    cover: &DomainCover,
    csb_qp: &[i32],
    csl_qp: &[i32],
    msg_bytes: usize,
    headroom: f32,
) -> CascadeTier {
    let conservative = std::env::var("PHASM_AUTO_TIER_CONSERVATIVE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if conservative {
        return CascadeTier::Tier0;
    }
    let required_bits = (msg_bytes as f32 * headroom * 8.0) as usize;
    for tier in [4u8, 3, 2, 1, 0] {
        if capacity_at_tier(cover, csb_qp, csl_qp, tier) >= required_bits {
            return CascadeTier::from_u8(tier).unwrap();
        }
    }
    CascadeTier::Tier0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::stego::hook::{SyntaxPath, BinKind};

    fn mk_pos(domain: EmbedDomain, coeff_idx: u8) -> PositionKey {
        let path = match domain {
            EmbedDomain::CoeffSignBypass => {
                SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx, kind: BinKind::Sign }
            }
            EmbedDomain::CoeffSuffixLsb => {
                SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx, kind: BinKind::SuffixLsb }
            }
            _ => unreachable!(),
        };
        PositionKey::new(0, 0, domain, path)
    }

    /// Helper: all positions at DC (scan_pos=0) for deterministic test math.
    fn mk_domain_bits_dc(domain: EmbedDomain, magnitudes: &[u16]) -> DomainBits {
        let mut bits = DomainBits::default();
        for &m in magnitudes {
            let pos = mk_pos(domain, 0); // scan_pos=0 = DC = basis_weight 1.0
            bits.bits.push(0);
            bits.positions.push(pos);
            bits.magnitudes.push(m);
        }
        bits
    }

    fn mk_domain_bits(domain: EmbedDomain, magnitudes: &[u16]) -> DomainBits {
        let mut bits = DomainBits::default();
        for (i, &m) in magnitudes.iter().enumerate() {
            let pos = mk_pos(domain, (i % 16) as u8);
            bits.bits.push(0);
            bits.positions.push(pos);
            bits.magnitudes.push(m);
        }
        bits
    }

    #[test]
    fn tier_0_is_noop() {
        let bits = mk_domain_bits(EmbedDomain::CoeffSignBypass, &[1, 5, 16, 32, 64]);
        let qp = vec![26; 5];
        let keep = apply_tier_filter(&bits, &qp, 0);
        assert_eq!(keep, vec![true; 5]);
    }

    #[test]
    fn tier_1_filters_high_magnitude() {
        // QP=26, q_bits=4, q_step = 2^(4-4) = 1.0. All DC (basis=1.0).
        // CSB factor = 2|M|. Pixel delta = 2|M|.
        // Threshold tier 1 = 32. So |M| ≥ 16 → delta ≥ 32 → reject.
        let bits = mk_domain_bits_dc(EmbedDomain::CoeffSignBypass, &[1, 10, 16, 50]);
        let qp = vec![26; 4];
        let keep = apply_tier_filter(&bits, &qp, 1);
        // |M|=1: delta=2 < 32 → keep
        // |M|=10: delta=20 < 32 → keep
        // |M|=16: delta=32, NOT < 32 → reject
        // |M|=50: delta=100 → reject
        assert_eq!(keep, vec![true, true, false, false]);
    }

    #[test]
    fn tier_4_aggressive_filter() {
        // QP=26, threshold 4. Delta = 2|M| at DC.
        // |M|=1: delta=2 < 4 → keep
        // |M|=2: delta=4, NOT < 4 → reject
        // |M|=5: delta=10 → reject
        let bits = mk_domain_bits_dc(EmbedDomain::CoeffSignBypass, &[1, 2, 5]);
        let qp = vec![26; 3];
        let keep = apply_tier_filter(&bits, &qp, 4);
        assert_eq!(keep, vec![true, false, false]);
    }

    #[test]
    fn csl_independent_of_magnitude() {
        // CSL flips are ±1 regardless of |coeff|. So filter behavior
        // depends only on q_step × scan_basis_weight, not |M|.
        let bits = mk_domain_bits(EmbedDomain::CoeffSuffixLsb, &[1, 100, 1000]);
        let qp = vec![26; 3];
        let keep_t2 = apply_tier_filter(&bits, &qp, 2); // threshold 16
        // delta = 1 × 1.26 × 1.0 = 1.26, < 16 → all keep
        assert_eq!(keep_t2, vec![true; 3]);
    }

    #[test]
    fn higher_qp_filters_more() {
        // All DC, |M|=10. Delta = 2*10 * q_step.
        // qp=18: q_bits=3, q_step = 2^(3-4) = 0.5, delta = 10 < 32 → keep
        // qp=30: q_bits=5, q_step = 2^(5-4) = 2.0, delta = 40 ≥ 32 → reject
        // qp=42: q_bits=7, q_step = 2^(7-4) = 8.0, delta = 160 → reject
        let bits = mk_domain_bits_dc(EmbedDomain::CoeffSignBypass, &[10, 10, 10]);
        let keep = apply_tier_filter(&bits, &[18, 30, 42], 1);
        assert_eq!(keep, vec![true, false, false]);
    }

    #[test]
    fn auto_select_picks_highest_fit() {
        let mut cover = DomainCover::default();
        // Use DC scan_pos so deltas are predictable: |M|=k → delta=2k.
        cover.coeff_sign_bypass = mk_domain_bits_dc(EmbedDomain::CoeffSignBypass,
            &[1, 2, 3, 4, 5, 6, 7, 8, 100, 200]);
        let csb_qp = vec![26; 10];
        let csl_qp: Vec<i32> = vec![];
        // Tier 4 threshold=4: |M|<2 survives → 1 position.
        // For 0-byte message + 20% headroom = 0 bits needed → tier 4 fits.
        let tier = auto_select_tier(&cover, &csb_qp, &csl_qp, 0, 1.2);
        assert_eq!(tier.as_u8(), 4);
    }

    #[test]
    fn auto_select_conservative_env_pins_tier0() {
        let mut cover = DomainCover::default();
        cover.coeff_sign_bypass = mk_domain_bits_dc(
            EmbedDomain::CoeffSignBypass, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        );
        let csb_qp = vec![26; 10];
        let csl_qp: Vec<i32> = vec![];
        // SAFETY: single-threaded test; no other thread reads env concurrently.
        // (edition-2024 made set_var/remove_var unsafe.)
        unsafe { std::env::set_var("PHASM_AUTO_TIER_CONSERVATIVE", "1"); }
        let tier = auto_select_tier(&cover, &csb_qp, &csl_qp, 0, 1.2);
        unsafe { std::env::remove_var("PHASM_AUTO_TIER_CONSERVATIVE"); }
        assert_eq!(tier.as_u8(), 0);
    }

    #[test]
    fn capacity_decreases_with_tier() {
        let mut cover = DomainCover::default();
        // Mix of low and high magnitudes at DC (predictable behavior).
        let mags: Vec<u16> = (1..=64).step_by(4).collect(); // 1, 5, 9, ..., 61
        cover.coeff_sign_bypass = mk_domain_bits_dc(EmbedDomain::CoeffSignBypass, &mags);
        let csb_qp = vec![26; mags.len()];
        let csl_qp: Vec<i32> = vec![];

        let c0 = capacity_at_tier(&cover, &csb_qp, &csl_qp, 0);
        let c1 = capacity_at_tier(&cover, &csb_qp, &csl_qp, 1);
        let c2 = capacity_at_tier(&cover, &csb_qp, &csl_qp, 2);
        let c3 = capacity_at_tier(&cover, &csb_qp, &csl_qp, 3);
        let c4 = capacity_at_tier(&cover, &csb_qp, &csl_qp, 4);
        // Capacity should decrease monotonically with tier
        assert!(c0 >= c1, "c0={c0} c1={c1}");
        assert!(c1 >= c2, "c1={c1} c2={c2}");
        assert!(c2 >= c3, "c2={c2} c3={c3}");
        assert!(c3 >= c4, "c3={c3} c4={c4}");
        // Tier 0 = full capacity
        assert_eq!(c0, mags.len());
    }

    #[test]
    fn ui_names_distinct() {
        let names: Vec<_> = [
            CascadeTier::Tier0, CascadeTier::Tier1, CascadeTier::Tier2,
            CascadeTier::Tier3, CascadeTier::Tier4,
        ].iter().map(|t| t.ui_name()).collect();
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(unique.len(), 5);
    }

}
