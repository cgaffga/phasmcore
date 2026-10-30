// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! 4-domain combined-cover helpers for streaming session 4-domain STC.
//!
//! Defines:
//! - [`CostWeights`]: per-domain global multipliers exposed as session
//!   config. Default ratios were locked by a real-corpus MvdSign
//!   density sweep — see
//!   `docs/design/video/h264/d07-streaming-4domain.md`.
//! - [`DomainBoundaries`]: per-domain length offsets within the
//!   concatenated cover/cost vector. Lets [`split_plan_4domain`]
//!   reverse the [`combine_cover_4domain`] operation.
//! - [`combine_cover_4domain`]: concat 4-domain cover bits + per-
//!   position costs in canonical order (CS → CSL → MVDs → MVDsl).
//!   Per-domain cost is multiplied by the domain weight.
//! - [`split_plan_4domain`]: reverse — take a single combined stego-
//!   bit vector + boundaries, populate a [`DomainPlan`].
//!
//! WET-∞ propagation: per-position cost of `f32::INFINITY` from any
//! domain's Pass-1 cost vector survives the `× domain_weight`
//! multiplication unchanged (since `domain_weight > 0`). STC will
//! never select WET-marked positions regardless of the domain weight.
//!
//! Combined cover length is bounded by
//! `cover.coeff_sign_bypass.bits.len() + cover.coeff_suffix_lsb.bits.len() + cover.mvd_sign_bypass.bits.len() + cover.mvd_suffix_lsb.bits.len()`.

use super::inject::DomainCover;
use super::orchestrate::{DomainCosts, DomainPlan};

/// Per-domain global cost multipliers (`W_domain` of the v2 cost unit
/// `cost = W · S · D`, spec § 4). STC combines these with the per-position
/// costs (J-UNIWARD `S` × damage `D`, from Pass 1) in a single STC plan per
/// GOP over all 4 domains, walked in one keyed order (§ 3).
///
/// **v2 defaults (2026-10): CoeffSign 1, CoeffSuffix 3, MVD domains wet.**
/// The MVD domains stay IN the cover (so admitting them later is a
/// cost-only change, no wire bump) but carry infinite cost: their damage is
/// mostly MV-predictor-chain propagation (median MVP + P_Skip inheritance),
/// which the damage model does not price yet, and under the global order
/// with MVD admitted every flip went to MVD (worst MB ~8 dB, spec § 3). A
/// separate all-wet MVD block would be infeasible (`MessageTooLarge`), which
/// is why the domains are scattered through the order rather than dropped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostWeights {
    pub coeff_sign: f32,
    pub coeff_suffix: f32,
    pub mvd_sign: f32,
    pub mvd_suffix: f32,
}

impl Default for CostWeights {
    /// Ship defaults (v2): CS : CSL = 1 : 3, MVD wet.
    fn default() -> Self {
        Self {
            coeff_sign: 1.0,
            coeff_suffix: 3.0,
            mvd_sign: f32::INFINITY,
            mvd_suffix: f32::INFINITY,
        }
    }
}

impl CostWeights {
    /// Conservative variant — CS+CSL only, MvdSign/MvdSuffix
    /// effectively excluded by WET-∞ weight. Useful as an escalation
    /// if 4-domain stealth degrades vs CS-only baseline:
    /// fall back to "3.5-domain" mode where MVD domains are in the
    /// cover (so a 4-domain-aware classifier sees natural MVD
    /// statistics) but never receive STC overrides.
    pub fn conservative_cs_csl_only() -> Self {
        Self {
            coeff_sign: 1.0,
            coeff_suffix: 3.0,
            mvd_sign: f32::INFINITY,
            mvd_suffix: f32::INFINITY,
        }
    }

    /// Bisect-only helper: MvdSign domain receives all STC overrides,
    /// every other domain is WET-∞ excluded.
    /// Used to isolate whether the OH264 fork's MV-cache mutation
    /// in `phasm_apply_mvd_hooks` is the primary Pass-1↔Pass-2 drift
    /// source. Not for production use.
    pub fn debug_mvd_sign_only() -> Self {
        Self {
            coeff_sign: f32::INFINITY,
            coeff_suffix: f32::INFINITY,
            mvd_sign: 1.0,
            mvd_suffix: f32::INFINITY,
        }
    }

    /// Bisect-only helper: CoeffSign domain receives all STC
    /// overrides, every other domain WET-∞. Counterpart to
    /// `debug_mvd_sign_only` for isolating the CS cascade class.
    pub fn debug_coeff_sign_only() -> Self {
        Self {
            coeff_sign: 1.0,
            coeff_suffix: f32::INFINITY,
            mvd_sign: f32::INFINITY,
            mvd_suffix: f32::INFINITY,
        }
    }

    /// Bisect-only helper: CoeffSuffixLsb only.
    pub fn debug_coeff_suffix_only() -> Self {
        Self {
            coeff_sign: f32::INFINITY,
            coeff_suffix: 1.0,
            mvd_sign: f32::INFINITY,
            mvd_suffix: f32::INFINITY,
        }
    }

    /// Bisect-only helper: MvdSuffixLsb only.
    pub fn debug_mvd_suffix_only() -> Self {
        Self {
            coeff_sign: f32::INFINITY,
            coeff_suffix: f32::INFINITY,
            mvd_sign: f32::INFINITY,
            mvd_suffix: 1.0,
        }
    }

    /// Bisect-only helper: MVD pair only (MSB + MSL). Counterpart to
    /// `conservative_cs_csl_only` — coeff domains WET-∞ excluded, both
    /// MVD domains carry the message. Used in the IPPPP-cascade bisect
    /// to attribute visible damage between coeff and MVD halves.
    pub fn debug_mvd_pair_only() -> Self {
        Self {
            coeff_sign: f32::INFINITY,
            coeff_suffix: f32::INFINITY,
            mvd_sign: 1.0,
            mvd_suffix: 3.0,
        }
    }

    /// Finite-bias variants for IPPPP cascade attribution. All 4
    /// domains stay STC-feasible (no INF cost) but the named pair gets
    /// a 100× discount so STC concentrates flips there. Used when WET-∞
    /// isolation drops below per-GOP chunk_frame overhead.
    pub fn debug_bias_coeff_pair() -> Self {
        Self { coeff_sign: 1.0, coeff_suffix: 3.0, mvd_sign: 1000.0, mvd_suffix: 1000.0 }
    }
    pub fn debug_bias_mvd_pair() -> Self {
        Self { coeff_sign: 100.0, coeff_suffix: 300.0, mvd_sign: 1.0, mvd_suffix: 3.0 }
    }
    pub fn debug_bias_coeff_sign() -> Self {
        Self { coeff_sign: 1.0, coeff_suffix: 100.0, mvd_sign: 1000.0, mvd_suffix: 1000.0 }
    }
    pub fn debug_bias_coeff_suffix() -> Self {
        Self { coeff_sign: 100.0, coeff_suffix: 1.0, mvd_sign: 1000.0, mvd_suffix: 1000.0 }
    }
    pub fn debug_bias_mvd_sign() -> Self {
        Self { coeff_sign: 1000.0, coeff_suffix: 1000.0, mvd_sign: 1.0, mvd_suffix: 100.0 }
    }
    pub fn debug_bias_mvd_suffix() -> Self {
        Self { coeff_sign: 1000.0, coeff_suffix: 1000.0, mvd_sign: 100.0, mvd_suffix: 1.0 }
    }
}

/// Per-domain length offsets within the concatenated combined cover.
/// The concat order is fixed: CS → CSL → MVDs → MVDsl.
///
/// `split_plan_4domain` reads these to slice the combined STC plan
/// back into 4 per-domain bit vectors.
#[derive(Debug, Clone, Copy)]
pub struct DomainBoundaries {
    pub n_coeff_sign: usize,
    pub n_coeff_suffix: usize,
    pub n_mvd_sign: usize,
    pub n_mvd_suffix: usize,
}

impl DomainBoundaries {
    /// Total number of cover positions across all 4 domains.
    pub fn total(&self) -> usize {
        self.n_coeff_sign + self.n_coeff_suffix + self.n_mvd_sign + self.n_mvd_suffix
    }

    /// Start offset of each domain's slice in the combined vector.
    pub fn coeff_sign_range(&self) -> std::ops::Range<usize> {
        0..self.n_coeff_sign
    }
    pub fn coeff_suffix_range(&self) -> std::ops::Range<usize> {
        self.n_coeff_sign..self.n_coeff_sign + self.n_coeff_suffix
    }
    pub fn mvd_sign_range(&self) -> std::ops::Range<usize> {
        let start = self.n_coeff_sign + self.n_coeff_suffix;
        start..start + self.n_mvd_sign
    }
    pub fn mvd_suffix_range(&self) -> std::ops::Range<usize> {
        let start = self.n_coeff_sign + self.n_coeff_suffix + self.n_mvd_sign;
        start..start + self.n_mvd_suffix
    }
}

/// Concatenate the 4-domain cover bits + per-position costs into
/// single combined vectors for STC.
///
/// Canonical order is fixed: CS → CSL → MVDs → MVDsl. The decoder's
/// per-GOP extract (`try_extract_chunk_from_gop` in
/// `streaming_session.rs`) calls this same `combine_cover_4domain` to
/// reproduce the cover vector STC operated on.
///
/// Returns `(combined_cover_bits, combined_costs, boundaries)`.
/// `combined_cover_bits.len() == combined_costs.len() == boundaries.total()`.
///
/// Per-position cost is multiplied by the corresponding domain
/// weight: `cost(D, P) = distortion_cost(D, P) × domain_weight(D)`.
/// WET-∞ positions stay ∞ post-multiplication (weights are > 0).
///
/// **Cost-vector length invariant.** Each per-domain cost vector must
/// match the bit vector length. If a Pass-1 cost vector is shorter
/// than its bit vector (some `pass1` modes leave costs empty for
/// fallback reasons), the missing entries are padded with `1.0`
/// (see `push_domain` below).
pub fn combine_cover_4domain(
    cover: &DomainCover,
    costs: &DomainCosts,
    weights: &CostWeights,
) -> (Vec<u8>, Vec<f32>, DomainBoundaries) {
    // Only the per-domain `bits` are combined (positions/magnitudes are never
    // read here) — delegate to the bits-only core so the shadow cascade can
    // combine over shadow-modified bits WITHOUT cloning the whole 11-byte/
    // position `DomainCover` (§6.cover memory reduction).
    combine_bits_4domain(
        &cover.coeff_sign_bypass.bits,
        &cover.coeff_suffix_lsb.bits,
        &cover.mvd_sign_bypass.bits,
        &cover.mvd_suffix_lsb.bits,
        costs,
        weights,
    )
}

/// Bits-only core of [`combine_cover_4domain`]: pack the 4 domains'
/// (already-extracted) cover bits into the canonical CS → CSL → MVDs → MVDsl
/// order with per-domain cost weighting. Lets the shadow cascade build the
/// combined STC input over shadow-embedded bits (cloned cheaply, 1 byte/pos)
/// while sharing `cover_p1`'s positions/magnitudes (10 bytes/pos) by reference.
/// Bit-identical to the `&DomainCover` path — same `push_domain`, same order.
pub fn combine_bits_4domain(
    coeff_sign_bits: &[u8],
    coeff_suffix_bits: &[u8],
    mvd_sign_bits: &[u8],
    mvd_suffix_bits: &[u8],
    costs: &DomainCosts,
    weights: &CostWeights,
) -> (Vec<u8>, Vec<f32>, DomainBoundaries) {
    let _prof = crate::codec::h264::profile::scope("cost.combine");
    let n_cs = coeff_sign_bits.len();
    let n_csl = coeff_suffix_bits.len();
    let n_mvds = mvd_sign_bits.len();
    let n_mvdsl = mvd_suffix_bits.len();
    let total = n_cs + n_csl + n_mvds + n_mvdsl;

    let mut bits = Vec::with_capacity(total);
    let mut weighted_costs = Vec::with_capacity(total);

    push_domain(
        &mut bits, &mut weighted_costs,
        coeff_sign_bits, &costs.coeff_sign_bypass, weights.coeff_sign,
    );
    push_domain(
        &mut bits, &mut weighted_costs,
        coeff_suffix_bits, &costs.coeff_suffix_lsb, weights.coeff_suffix,
    );
    push_domain(
        &mut bits, &mut weighted_costs,
        mvd_sign_bits, &costs.mvd_sign_bypass, weights.mvd_sign,
    );
    push_domain(
        &mut bits, &mut weighted_costs,
        mvd_suffix_bits, &costs.mvd_suffix_lsb, weights.mvd_suffix,
    );

    debug_assert_eq!(bits.len(), total);
    debug_assert_eq!(weighted_costs.len(), total);

    let boundaries = DomainBoundaries {
        n_coeff_sign: n_cs,
        n_coeff_suffix: n_csl,
        n_mvd_sign: n_mvds,
        n_mvd_suffix: n_mvdsl,
    };
    (bits, weighted_costs, boundaries)
}

fn push_domain(
    out_bits: &mut Vec<u8>,
    out_costs: &mut Vec<f32>,
    src_bits: &[u8],
    src_costs: &[f32],
    weight: f32,
) {
    out_bits.extend_from_slice(src_bits);
    let n = src_bits.len();
    // Pad costs to bit length with 1.0 if shorter; saturate weight
    // multiplication so f32::INFINITY × weight stays ∞.
    for i in 0..n {
        let raw = src_costs.get(i).copied().unwrap_or(1.0);
        out_costs.push(raw * weight);
    }
}

/// A coefficient position is DRY — kept in the v2 STC cover — iff it is an AC
/// position of a 4×4 transform block (`coeff_idx != 0`). Everything else is
/// **syntax-wet** and dropped from the cover entirely (2026-10-25
/// wet-removal; spec § 3/§ 5 amendment): the DC paths (`Luma4x4{0}`,
/// `ChromaAc{0}`, `ChromaDc`, `LumaDcIntra16x16`), `Luma8x8` (no 4×4 cost
/// basis), and both MVD domains (no predictor-chain damage term yet).
///
/// The rule is a pure function of the `SyntaxPath`, so the decoder recomputes
/// it from the walked positions alone — no cost vector, no side channel. That
/// is why cost-based ∞ (tier filter, shadow overlay, the `frame_idx >=
/// n_frames` guard) must stay IN the cover as wet: the decoder cannot tell a
/// cost-∞ dry position from a finite one, so both encoder and decoder keep
/// every syntax-dry position and let the STC syndrome run over all of them.
pub fn coeff_is_dry(key: super::hook::PositionKey) -> bool {
    use super::hook::SyntaxPath;
    match key.syntax_path() {
        SyntaxPath::Luma4x4 { coeff_idx, .. } => coeff_idx != 0,
        SyntaxPath::ChromaAc { coeff_idx, .. } => coeff_idx != 0,
        SyntaxPath::Luma8x8 { .. }
        | SyntaxPath::ChromaDc { .. }
        | SyntaxPath::LumaDcIntra16x16 { .. }
        | SyntaxPath::Mvd { .. } => false,
    }
}

/// The syntax-dry keep-mask over the canonical CS → CSL → MVDs → MVDsl cover
/// of `cover`: `mask[i]` is true for a position kept in the v2 STC cover.
/// MVD positions are always false (both domains dropped). Encoder, decoder
/// and probe build it the same way from the same walked positions.
pub fn canonical_dry_mask(cover: &DomainCover) -> Vec<bool> {
    let mut m = Vec::with_capacity(cover.total_len());
    m.extend(cover.coeff_sign_bypass.positions.iter().map(|&k| coeff_is_dry(k)));
    m.extend(cover.coeff_suffix_lsb.positions.iter().map(|&k| coeff_is_dry(k)));
    m.extend(std::iter::repeat(false).take(cover.mvd_sign_bypass.len()));
    m.extend(std::iter::repeat(false).take(cover.mvd_suffix_lsb.len()));
    m
}

/// v2 embedding order of one GOP's combined cover (spec § 3): position
/// `order[i]` of the canonical CS → CSL → MVDs → MVDsl cover is the `i`-th
/// STC input. The encoder, the decoder and the capacity probe all apply it
/// right after the combine step; the STC itself is unchanged.
#[derive(Debug, Clone)]
pub struct StcOrder(Vec<u32>);

impl StcOrder {
    /// The keyed order of an `n`-position cover (`GopSeeds::order`).
    pub fn new(n: usize, order_seed: &[u8; 32]) -> Self {
        let _prof = crate::codec::h264::profile::scope("v2.order_build");
        Self(crate::stego::cover_order::gop_cover_order(n, order_seed))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// `order[i]` for every STC index `i`.
    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }

    /// Canonical → STC order: `out[i] = canonical[order[i]]`.
    pub fn permute<T: Copy>(&self, canonical: &[T]) -> Vec<T> {
        debug_assert_eq!(canonical.len(), self.0.len());
        self.0.iter().map(|&j| canonical[j as usize]).collect()
    }

    /// STC → canonical order: `out[order[i]] = stc[i]`.
    pub fn unpermute<T: Copy + Default>(&self, stc: &[T]) -> Vec<T> {
        debug_assert_eq!(stc.len(), self.0.len());
        let mut out = vec![T::default(); stc.len()];
        for (i, &j) in self.0.iter().enumerate() {
            out[j as usize] = stc[i];
        }
        out
    }
}

/// One GOP's v2 STC input: the combined cover and costs in STC order, with
/// what is needed to map a plan back.
///
/// **Wet-removal (2026-10-25).** `bits` / `costs` / `order` cover the
/// SYNTAX-DRY positions only ([`canonical_dry_mask`]); the syntax-wet
/// positions are not in the STC cover at all. To map a plan back,
/// [`Self::canonical`] starts from the full canonical cover and overwrites
/// only the dry positions, so a wet position always emits its original cover
/// bit (never a flip). `boundaries` stays the FULL per-domain lengths, so
/// [`split_plan_4domain`] over the expanded vector still produces
/// full-length per-domain plans that line up with the walker's positions.
#[derive(Debug, Clone)]
pub struct GopStcInput {
    /// Dry cover bits in STC order (`len == n_dry`).
    pub bits: Vec<u8>,
    /// Dry weighted costs in STC order (`len == n_dry`).
    pub costs: Vec<f32>,
    /// FULL per-domain lengths of the canonical combined cover.
    pub boundaries: DomainBoundaries,
    /// Keyed order over the DRY positions (`len == n_dry`).
    pub order: StcOrder,
    /// Full canonical cover bits (`len == boundaries.total()`), for the
    /// wet positions on expand-back.
    full_canonical: Vec<u8>,
    /// Dry STC index → canonical position (`len == n_dry`).
    dry_to_canonical: Vec<u32>,
}

impl GopStcInput {
    /// Put a canonical combined cover + costs into STC order, dropping the
    /// syntax-wet positions (`dry_mask[i]` false). `dry_mask` is over the
    /// canonical cover ([`canonical_dry_mask`]); its length must equal
    /// `bits.len()`.
    pub fn from_canonical(
        bits: Vec<u8>,
        costs: Vec<f32>,
        boundaries: DomainBoundaries,
        dry_mask: &[bool],
        order_seed: &[u8; 32],
    ) -> Self {
        debug_assert_eq!(bits.len(), dry_mask.len());
        debug_assert_eq!(bits.len(), costs.len());
        let n_dry = dry_mask.iter().filter(|&&k| k).count();
        let mut dry_bits = Vec::with_capacity(n_dry);
        let mut dry_costs = Vec::with_capacity(n_dry);
        let mut dry_to_canonical = Vec::with_capacity(n_dry);
        for (i, &keep) in dry_mask.iter().enumerate() {
            if keep {
                dry_bits.push(bits[i]);
                dry_costs.push(costs[i]);
                dry_to_canonical.push(i as u32);
            }
        }
        let order = StcOrder::new(dry_bits.len(), order_seed);
        let _prof = crate::codec::h264::profile::scope("v2.order_permute");
        Self {
            bits: order.permute(&dry_bits),
            costs: order.permute(&dry_costs),
            boundaries,
            order,
            full_canonical: bits,
            dry_to_canonical,
        }
    }

    /// An `n_dry`-length STC-order stego vector expanded to the FULL canonical
    /// cover: dry positions take the stego bit, wet positions keep their
    /// original cover bit.
    pub fn canonical(&self, stc_stego_bits: &[u8]) -> Vec<u8> {
        let _prof = crate::codec::h264::profile::scope("v2.order_permute");
        let dry_canonical = self.order.unpermute(stc_stego_bits);
        let mut out = self.full_canonical.clone();
        for (j, &ci) in self.dry_to_canonical.iter().enumerate() {
            out[ci as usize] = dry_canonical[j];
        }
        out
    }

    /// An `n_dry`-length STC-order stego vector as a per-domain plan.
    pub fn split(&self, stc_stego_bits: &[u8]) -> DomainPlan {
        split_plan_4domain(&self.canonical(stc_stego_bits), &self.boundaries)
    }
}

/// [`combine_cover_4domain`] followed by the syntax-dry filter and the v2
/// order (`GopSeeds::order`).
pub fn combine_cover_4domain_v2(
    cover: &DomainCover,
    costs: &DomainCosts,
    weights: &CostWeights,
    order_seed: &[u8; 32],
) -> GopStcInput {
    let (bits, c, b) = combine_cover_4domain(cover, costs, weights);
    GopStcInput::from_canonical(bits, c, b, &canonical_dry_mask(cover), order_seed)
}

/// [`combine_bits_4domain`] followed by the syntax-dry filter and the v2
/// order. `cover` supplies the positions for [`canonical_dry_mask`]; the bit
/// slices (possibly shadow-modified) must line up 1:1 with its domains.
#[allow(clippy::too_many_arguments)]
pub fn combine_bits_4domain_v2(
    coeff_sign_bits: &[u8],
    coeff_suffix_bits: &[u8],
    mvd_sign_bits: &[u8],
    mvd_suffix_bits: &[u8],
    costs: &DomainCosts,
    weights: &CostWeights,
    cover: &DomainCover,
    order_seed: &[u8; 32],
) -> GopStcInput {
    let (bits, c, b) = combine_bits_4domain(
        coeff_sign_bits, coeff_suffix_bits, mvd_sign_bits, mvd_suffix_bits, costs, weights,
    );
    GopStcInput::from_canonical(bits, c, b, &canonical_dry_mask(cover), order_seed)
}

/// The decoder's view: one GOP's walked cover bits in STC order (no costs),
/// over the syntax-dry positions only ([`canonical_dry_mask`]) so `n_cover`
/// and the bit sequence match the encoder's dry cover.
pub fn gop_stc_cover_bits(cover: &DomainCover, order_seed: &[u8; 32]) -> Vec<u8> {
    let mask = canonical_dry_mask(cover);
    let mut dry = Vec::with_capacity(mask.iter().filter(|&&k| k).count());
    let mut i = 0usize;
    for d in [
        &cover.coeff_sign_bypass.bits,
        &cover.coeff_suffix_lsb.bits,
        &cover.mvd_sign_bypass.bits,
        &cover.mvd_suffix_lsb.bits,
    ] {
        for &b in d {
            if mask[i] {
                dry.push(b);
            }
            i += 1;
        }
    }
    StcOrder::new(dry.len(), order_seed).permute(&dry)
}

/// Reverse [`combine_cover_4domain`]: take a single STC-plan output
/// vector (length == `boundaries.total()`) and split it into a
/// [`DomainPlan`] with 4 per-domain stego-bit vectors.
///
/// `total_modifications` and `total_cost` on the returned plan are
/// 0 — fill them with the STC result's values at the call site.
pub fn split_plan_4domain(
    combined_stego_bits: &[u8],
    boundaries: &DomainBoundaries,
) -> DomainPlan {
    debug_assert_eq!(combined_stego_bits.len(), boundaries.total(),
        "split_plan_4domain: combined len {} != boundaries total {}",
        combined_stego_bits.len(), boundaries.total());
    DomainPlan {
        coeff_sign_bypass: combined_stego_bits[boundaries.coeff_sign_range()].to_vec(),
        coeff_suffix_lsb: combined_stego_bits[boundaries.coeff_suffix_range()].to_vec(),
        mvd_sign_bypass: combined_stego_bits[boundaries.mvd_sign_range()].to_vec(),
        mvd_suffix_lsb: combined_stego_bits[boundaries.mvd_suffix_range()].to_vec(),
        total_modifications: 0,
        total_cost: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::stego::inject::DomainBits;
    use crate::codec::h264::stego::hook::{
        BinKind, EmbedDomain, PositionKey, SyntaxPath,
    };

    /// Build a [`DomainBits`] with `bits` and dummy position keys.
    /// Position content doesn't matter for combine/split correctness —
    /// these tests verify bit-level concatenation, not position
    /// fidelity (which is tested by Phase 0 walker-parity gates).
    fn mock_domain_bits(bits: Vec<u8>) -> DomainBits {
        // AC Luma4x4 positions (coeff_idx != 0) are syntax-DRY — kept in the
        // v2 cover. The non-v2 concat tests only read the bits, so the path
        // kind is immaterial there; the v2 tests need dry positions.
        let positions: Vec<PositionKey> = (0..bits.len() as u32)
            .map(|i| {
                PositionKey::new(
                    0, i,
                    EmbedDomain::CoeffSignBypass,
                    SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 1, kind: BinKind::Sign },
                )
            })
            .collect();
        let magnitudes = vec![0u16; bits.len()];
        DomainBits { bits, positions, magnitudes }
    }

    /// A DC (syntax-WET) mock: `LumaDcIntra16x16{0}`, dropped from the v2 cover.
    fn mock_dc_bits(bits: Vec<u8>) -> DomainBits {
        let positions: Vec<PositionKey> = (0..bits.len() as u32)
            .map(|i| {
                PositionKey::new(
                    0, i,
                    EmbedDomain::CoeffSignBypass,
                    SyntaxPath::LumaDcIntra16x16 { coeff_idx: 0, kind: BinKind::Sign },
                )
            })
            .collect();
        DomainBits { bits: bits.clone(), positions, magnitudes: vec![0u16; bits.len()] }
    }

    /// v2 defaults: CS : CSL = 1 : 3, both MVD domains wet (in the cover,
    /// infinite cost) — spec § 3 / § 5.
    #[test]
    fn cost_weights_default_is_v2_mvd_wet() {
        let w = CostWeights::default();
        assert_eq!(w.coeff_sign, 1.0);
        assert_eq!(w.coeff_suffix, 3.0);
        assert!(w.mvd_sign.is_infinite() && w.mvd_sign.is_sign_positive());
        assert!(w.mvd_suffix.is_infinite() && w.mvd_suffix.is_sign_positive());
        assert_eq!(w, CostWeights::conservative_cs_csl_only());
    }

    /// Wet-removal (2026-10-25): the v2 cover is the SYNTAX-DRY positions
    /// only; the keyed order is over them; `split` expands back to the full
    /// canonical cover with wet positions unchanged; the decoder's view
    /// matches. Here the 340 AC coeff positions are dry and the 30 MVD
    /// positions are dropped.
    #[test]
    fn v2_cover_drops_syntax_wet_and_round_trips() {
        let mut cover = DomainCover::default();
        cover.coeff_sign_bypass = mock_domain_bits((0..300).map(|i| (i % 3 == 0) as u8).collect());
        cover.coeff_suffix_lsb = mock_domain_bits((0..40).map(|i| (i % 2) as u8).collect());
        cover.mvd_sign_bypass = mock_domain_bits(vec![1; 25]);
        cover.mvd_suffix_lsb = mock_domain_bits(vec![0, 1, 1, 0, 1]);
        let costs = DomainCosts {
            coeff_sign_bypass: (0..300).map(|i| i as f32).collect(),
            coeff_suffix_lsb: vec![2.0; 40],
            mvd_sign_bypass: vec![1.0; 25],
            mvd_suffix_lsb: vec![1.0; 5],
        };
        let w = CostWeights::default();
        let (canon_bits, _canon_costs, _) = combine_cover_4domain(&cover, &costs, &w);
        let a = combine_cover_4domain_v2(&cover, &costs, &w, &[3u8; 32]);
        let b = combine_cover_4domain_v2(&cover, &costs, &w, &[4u8; 32]);
        // Only the 340 dry positions are in the STC cover; the 30 MVD ones
        // (and any DC/8x8, none here) are gone.
        assert_eq!(a.bits.len(), 340);
        assert_eq!(a.order.len(), 340);
        assert_ne!(a.order.as_slice(), b.order.as_slice());
        // The dropped domains were the only wet ones, so no ∞ is left.
        assert!(a.costs.iter().all(|c| c.is_finite()));
        // Expand-back reproduces the FULL canonical cover (wet = original).
        assert_eq!(a.canonical(&a.bits), canon_bits);
        let plan = a.split(&a.bits);
        assert_eq!(plan.coeff_sign_bypass, cover.coeff_sign_bypass.bits);
        assert_eq!(plan.coeff_suffix_lsb, cover.coeff_suffix_lsb.bits);
        assert_eq!(plan.mvd_sign_bypass, cover.mvd_sign_bypass.bits);
        assert_eq!(plan.mvd_suffix_lsb, cover.mvd_suffix_lsb.bits);
        // The decoder's view is the same dry STC-order vector.
        assert_eq!(gop_stc_cover_bits(&cover, &[3u8; 32]), a.bits);
    }

    /// A DC (syntax-wet) position is dropped from the v2 cover and never
    /// flips: flipping every dry STC bit leaves the DC cover bit untouched.
    #[test]
    fn v2_wet_positions_never_flip() {
        let mut cover = DomainCover::default();
        // 4 AC dry + 2 DC wet in CS; 1 AC dry in CSL; 1 MVD (wet).
        let mut cs = mock_domain_bits(vec![1, 0, 1, 0]);
        cs.extend(mock_dc_bits(vec![1, 1]));
        cover.coeff_sign_bypass = cs;
        cover.coeff_suffix_lsb = mock_domain_bits(vec![0]);
        cover.mvd_sign_bypass = mock_domain_bits(vec![1]);
        let costs = DomainCosts {
            coeff_sign_bypass: vec![1.0; 6],
            coeff_suffix_lsb: vec![1.0],
            mvd_sign_bypass: vec![1.0],
            mvd_suffix_lsb: vec![],
        };
        let w = CostWeights::default();
        let a = combine_cover_4domain_v2(&cover, &costs, &w, &[7u8; 32]);
        assert_eq!(a.bits.len(), 5); // 4 CS-AC + 1 CSL-AC dry
        // Flip every dry STC bit, expand, split.
        let flipped: Vec<u8> = a.bits.iter().map(|b| 1 - b).collect();
        let plan = a.split(&flipped);
        // The two DC positions (indices 4,5 of CS) keep their cover bits.
        assert_eq!(plan.coeff_sign_bypass[4], 1);
        assert_eq!(plan.coeff_sign_bypass[5], 1);
        // The MVD position keeps its cover bit.
        assert_eq!(plan.mvd_sign_bypass, vec![1]);
        // The 4 AC CS bits flipped.
        assert_eq!(&plan.coeff_sign_bypass[..4], &[0, 1, 0, 1]);
    }

    /// Conservative variant excludes MVD domains via WET-∞ weight.
    #[test]
    fn cost_weights_conservative_excludes_mvd() {
        let w = CostWeights::conservative_cs_csl_only();
        assert_eq!(w.coeff_sign, 1.0);
        assert_eq!(w.coeff_suffix, 3.0);
        assert!(w.mvd_sign.is_infinite() && w.mvd_sign.is_sign_positive());
        assert!(w.mvd_suffix.is_infinite() && w.mvd_suffix.is_sign_positive());
    }

    /// Concat in canonical order CS → CSL → MVDs → MVDsl.
    #[test]
    fn combine_cover_concatenates_in_canonical_order() {
        let mut cover = DomainCover::default();
        cover.coeff_sign_bypass = mock_domain_bits(vec![1, 0, 1]); // 3 bits
        cover.coeff_suffix_lsb = mock_domain_bits(vec![0, 1]);     // 2 bits
        cover.mvd_sign_bypass = mock_domain_bits(vec![1]);         // 1 bit
        cover.mvd_suffix_lsb = mock_domain_bits(vec![0, 0]);       // 2 bits

        let costs = DomainCosts {
            coeff_sign_bypass: vec![1.0, 2.0, 3.0],
            coeff_suffix_lsb: vec![4.0, 5.0],
            mvd_sign_bypass: vec![6.0],
            mvd_suffix_lsb: vec![7.0, 8.0],
        };

        let weights = CostWeights { coeff_sign: 1.0, coeff_suffix: 3.0, mvd_sign: 10.0, mvd_suffix: 10.0 };
        let (bits, weighted_costs, b) = combine_cover_4domain(&cover, &costs, &weights);

        // Bits concatenated in canonical order.
        assert_eq!(bits, vec![1, 0, 1, 0, 1, 1, 0, 0]);
        assert_eq!(bits.len(), 8);

        // Costs multiplied by per-domain weight.
        assert_eq!(weighted_costs.len(), 8);
        // CS: weight=1
        assert_eq!(weighted_costs[0], 1.0);
        assert_eq!(weighted_costs[1], 2.0);
        assert_eq!(weighted_costs[2], 3.0);
        // CSL: weight=3
        assert_eq!(weighted_costs[3], 12.0);
        assert_eq!(weighted_costs[4], 15.0);
        // MVDs: weight=10
        assert_eq!(weighted_costs[5], 60.0);
        // MVDsl: weight=10
        assert_eq!(weighted_costs[6], 70.0);
        assert_eq!(weighted_costs[7], 80.0);

        // Boundaries reflect per-domain lengths.
        assert_eq!(b.n_coeff_sign, 3);
        assert_eq!(b.n_coeff_suffix, 2);
        assert_eq!(b.n_mvd_sign, 1);
        assert_eq!(b.n_mvd_suffix, 2);
        assert_eq!(b.total(), 8);
    }

    /// WET-∞ propagates through weight multiplication.
    #[test]
    fn combine_cover_propagates_wet_infinity() {
        let mut cover = DomainCover::default();
        cover.coeff_sign_bypass = mock_domain_bits(vec![1, 0]);
        cover.coeff_suffix_lsb = mock_domain_bits(vec![0]);
        cover.mvd_sign_bypass = mock_domain_bits(vec![1]);
        cover.mvd_suffix_lsb = mock_domain_bits(vec![0]);

        let costs = DomainCosts {
            coeff_sign_bypass: vec![1.0, f32::INFINITY], // WET on CS[1]
            coeff_suffix_lsb: vec![f32::INFINITY],       // WET on CSL[0]
            mvd_sign_bypass: vec![1.0],
            mvd_suffix_lsb: vec![f32::INFINITY],         // WET on MVDsl[0]
        };

        let weights = CostWeights { coeff_sign: 1.0, coeff_suffix: 3.0, mvd_sign: 10.0, mvd_suffix: 10.0 };
        let (_, weighted_costs, _) = combine_cover_4domain(&cover, &costs, &weights);

        // CS[0] = 1.0 × 1 = 1.0
        assert_eq!(weighted_costs[0], 1.0);
        // CS[1] = ∞ × 1 = ∞
        assert!(weighted_costs[1].is_infinite() && weighted_costs[1].is_sign_positive());
        // CSL[0] = ∞ × 3 = ∞
        assert!(weighted_costs[2].is_infinite() && weighted_costs[2].is_sign_positive());
        // MVDs[0] = 1.0 × 10 = 10
        assert_eq!(weighted_costs[3], 10.0);
        // MVDsl[0] = ∞ × 10 = ∞
        assert!(weighted_costs[4].is_infinite() && weighted_costs[4].is_sign_positive());
    }

    /// combine + split = identity on cover bits.
    /// (Split takes the *plan* not the *cover*, but for identity-roundtrip
    /// purposes treating the cover as the plan tests boundary correctness.)
    #[test]
    fn combine_then_split_roundtrip() {
        let mut cover = DomainCover::default();
        cover.coeff_sign_bypass = mock_domain_bits(vec![1, 0, 1, 1, 0]);
        cover.coeff_suffix_lsb = mock_domain_bits(vec![0, 0, 1]);
        cover.mvd_sign_bypass = mock_domain_bits(vec![1, 1]);
        cover.mvd_suffix_lsb = mock_domain_bits(vec![0, 1, 1, 0]);

        let costs = DomainCosts {
            coeff_sign_bypass: vec![1.0; 5],
            coeff_suffix_lsb: vec![1.0; 3],
            mvd_sign_bypass: vec![1.0; 2],
            mvd_suffix_lsb: vec![1.0; 4],
        };

        let weights = CostWeights::default();
        let (bits, _, b) = combine_cover_4domain(&cover, &costs, &weights);
        let plan = split_plan_4domain(&bits, &b);

        assert_eq!(plan.coeff_sign_bypass, cover.coeff_sign_bypass.bits);
        assert_eq!(plan.coeff_suffix_lsb, cover.coeff_suffix_lsb.bits);
        assert_eq!(plan.mvd_sign_bypass, cover.mvd_sign_bypass.bits);
        assert_eq!(plan.mvd_suffix_lsb, cover.mvd_suffix_lsb.bits);
    }

    /// Empty domain (no positions) is handled — produces a zero-length
    /// slice for that domain, doesn't panic.
    #[test]
    fn combine_handles_empty_domains() {
        let mut cover = DomainCover::default();
        cover.coeff_sign_bypass = mock_domain_bits(vec![1, 0]);
        // CSL, MVDs, MVDsl all empty (e.g. all-IDR clip)
        let costs = DomainCosts {
            coeff_sign_bypass: vec![1.0, 1.0],
            coeff_suffix_lsb: vec![],
            mvd_sign_bypass: vec![],
            mvd_suffix_lsb: vec![],
        };

        let (bits, weighted_costs, b) = combine_cover_4domain(
            &cover, &costs, &CostWeights::default(),
        );
        assert_eq!(bits, vec![1, 0]);
        assert_eq!(weighted_costs, vec![1.0, 1.0]);
        assert_eq!(b.total(), 2);
        assert_eq!(b.n_coeff_suffix, 0);
        assert_eq!(b.n_mvd_sign, 0);
        assert_eq!(b.n_mvd_suffix, 0);

        let plan = split_plan_4domain(&bits, &b);
        assert_eq!(plan.coeff_sign_bypass, vec![1, 0]);
        assert!(plan.coeff_suffix_lsb.is_empty());
        assert!(plan.mvd_sign_bypass.is_empty());
        assert!(plan.mvd_suffix_lsb.is_empty());
    }

    /// Pad-with-1.0 fallback for short cost vectors.
    #[test]
    fn combine_pads_short_cost_vectors_with_unit() {
        let mut cover = DomainCover::default();
        cover.coeff_sign_bypass = mock_domain_bits(vec![1, 0, 1]);
        let costs = DomainCosts {
            coeff_sign_bypass: vec![5.0], // SHORT — only 1 cost for 3 bits
            ..Default::default()
        };
        let weights = CostWeights { coeff_sign: 2.0, ..Default::default() };
        let (_, weighted_costs, _) = combine_cover_4domain(&cover, &costs, &weights);
        assert_eq!(weighted_costs.len(), 3);
        assert_eq!(weighted_costs[0], 10.0); // 5.0 × 2
        assert_eq!(weighted_costs[1], 2.0);  // 1.0 × 2 (pad)
        assert_eq!(weighted_costs[2], 2.0);  // 1.0 × 2 (pad)
    }
}
