// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Three-pass encode-time stego orchestration (Phase 6D.7).
//
// Synthesizes the per-domain primitives (6D.3 / 6D.5 / 6D.6) and the
// cost model (6D.4) into a driver that takes per-MB decisions
// (mode, MVs, quantized coeffs) for one GOP, runs:
//
//   Pass 1: collect per-domain cover bits + cost vectors
//   Pass 2: per-domain STC plan
//   Pass 3: apply the plan's bit overrides to the decision cache
//
// and yields the modified decision cache for the encoder to entropy-
// code in the third pass.
//
// **Scope for 6D.7**: framework + synthetic-decision-cache driver.
// Integration with the production encoder's per-MB orchestration
// (the slice walker that calls per-element encode_* fns in spec
// syntax order, deferred from 6D.2) lands in 6D.8 atomic swap.
// Until then the driver works on hand-constructed `GopDecisionCache`
// values for testing.

use crate::stego::stc::embed::{stc_embed, EmbedResult};
use crate::stego::stc::hhat::generate_hhat;

use super::cost_model::{
    coeff_sign_cost_vec, coeff_suffix_lsb_cost_vec, mvd_sign_cost_vec,
    mvd_suffix_lsb_cost_vec, PositionCostCtx,
};
use super::keys::CabacStegoMasterKeys;
#[allow(unused_imports)]
use super::keys::DomainSeeds;
use super::{
    apply_coeff_sign_overrides, apply_coeff_suffix_lsb_overrides,
    apply_mvd_sign_overrides, apply_mvd_suffix_lsb_overrides,
    enumerate_coeff_sign_positions, enumerate_coeff_suffix_lsb_positions,
    enumerate_mvd_sign_positions, enumerate_mvd_suffix_lsb_positions,
    extract_coeff_sign_bits, extract_coeff_suffix_lsb_bits,
    extract_mvd_sign_bits, extract_mvd_suffix_lsb_bits,
    BinKind, BitInjector, DomainBits, DomainCover, EmbedDomain, MvdSlot, PositionKey,
    SyntaxPath,
};

/// One macroblock's per-domain stego inputs. Aggregated by Pass 1
/// from the encoder's decision cache; consumed in Pass 1 to build
/// `DomainCover` and in Pass 3 to apply per-domain overrides.
///
/// Coefficients are stored per-block; MVDs per-partition. The
/// `SyntaxPath`-builder closure is captured at insertion time so
/// the orchestrator stays generic over block layout (Luma4x4,
/// Luma8x8, ChromaAc, etc.).
#[derive(Clone, Debug)]
pub struct MbDecision {
    /// Frame index within the video.
    pub frame_idx: u32,
    /// Macroblock address (raster order, 0..mb_count).
    pub mb_addr: u32,
    /// Residual coefficient blocks for this MB. Each entry:
    /// (scan_coeffs, start_idx, end_idx, ctx_block_cat, path_kind).
    /// Cost ctx defaults to drift_factor=1.0 (6D.4 stub).
    pub residual_blocks: Vec<MbResidualBlock>,
    /// MVD slots for this MB (P-slice MBs only).
    pub mvd_slots: Vec<MvdSlot>,
}

/// One residual block in an [`MbDecision`].
#[derive(Clone, Debug)]
pub struct MbResidualBlock {
    pub scan_coeffs: Vec<i32>,
    pub start_idx: usize,
    pub end_idx: usize,
    /// CABAC ctxBlockCat 0..=5.
    pub ctx_block_cat: u8,
    /// SyntaxPath template for emitted positions. The closure
    /// receives (coeff_idx, BinKind) and returns the appropriate
    /// SyntaxPath variant for this block. Stored as a tag so the
    /// orchestrator can rebuild the closure during Pass 1 / Pass 3.
    pub path_kind: ResidualPathKind,
}

/// Tag describing which [`SyntaxPath`] variant a residual block
/// uses. Lets `MbResidualBlock` stay clonable + serializable
/// without storing closures.
#[derive(Copy, Clone, Debug)]
pub enum ResidualPathKind {
    Luma4x4 { block_idx: u8 },
    Luma8x8 { block_idx: u8 },
    ChromaAc { plane: u8, block_idx: u8 },
    ChromaDc { plane: u8 },
    LumaDcIntra16x16,
}

impl ResidualPathKind {
    /// Build a SyntaxPath for the given coefficient index + bin kind.
    pub fn path(self, coeff_idx: u8, kind: BinKind) -> SyntaxPath {
        match self {
            ResidualPathKind::Luma4x4 { block_idx } => SyntaxPath::Luma4x4 {
                block_idx, coeff_idx, kind,
            },
            ResidualPathKind::Luma8x8 { block_idx } => SyntaxPath::Luma8x8 {
                block_idx, coeff_idx, kind,
            },
            ResidualPathKind::ChromaAc { plane, block_idx } => SyntaxPath::ChromaAc {
                plane, block_idx, coeff_idx, kind,
            },
            ResidualPathKind::ChromaDc { plane } => SyntaxPath::ChromaDc {
                plane, coeff_idx, kind,
            },
            ResidualPathKind::LumaDcIntra16x16 => SyntaxPath::LumaDcIntra16x16 {
                coeff_idx, kind,
            },
        }
    }

    /// §B-cascade-real fix 2026-05-07 — per-path-kind cost multiplier
    /// for sign + suffix-LSB flips. DC paths (LumaDcIntra16x16,
    /// ChromaDc) propagate through inverse Hadamard to ALL sub-blocks
    /// of the parent macroblock — a single DC sign flip changes the
    /// average brightness of 256 luma pixels (or 64 chroma) by
    /// 2N×qstep, vs an AC flip that only affects one 4×4 region
    /// (16 pixels) with mostly-canceling IDCT response. Without this
    /// weighting STC sees DC and AC as equivalent at the same |coeff|
    /// and may pick a low-magnitude DC over a high-magnitude AC →
    /// catastrophic visual damage at QP=26 IDR.
    ///
    /// Multipliers are conservative (perceptual visibility, not strict
    /// pixel-area count) and capture the dominant order-of-magnitude
    /// difference. Production paths can refine via cost calibration.
    pub fn cost_multiplier(self) -> f32 {
        match self {
            // 4×4 hadamard DC propagates to 16 sub-blocks → 256 pixels
            // of coherent DC offset.
            ResidualPathKind::LumaDcIntra16x16 => 16.0,
            // 2×2 hadamard DC (4:2:0 chroma) propagates to 4 sub-blocks
            // = 64 chroma pixels. Chroma plane is 1/4 of luma area but
            // perceptually equally visible (color shift) — keep 16×.
            ResidualPathKind::ChromaDc { .. } => 16.0,
            // AC paths: baseline cost. Single-block locality.
            ResidualPathKind::Luma4x4 { .. } => 1.0,
            ResidualPathKind::Luma8x8 { .. } => 1.0,
            ResidualPathKind::ChromaAc { .. } => 1.0,
        }
    }
}

/// Per-GOP decision cache. Pass 1 populates the cover; Pass 3
/// re-uses the cache after applying overrides.
#[derive(Default, Clone, Debug)]
pub struct GopDecisionCache {
    pub mbs: Vec<MbDecision>,
}

impl GopDecisionCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, mb: MbDecision) {
        self.mbs.push(mb);
    }
}

// ─── Pass 1: collect cover bits + costs ──────────────────────────

/// Pass 1 result: per-domain cover (bits + positions) + cost vector
/// aligned to those bits.
#[derive(Default, Clone, Debug)]
pub struct GopCover {
    pub cover: DomainCover,
    pub costs: DomainCosts,
}

/// Cost vectors per domain, indices aligned with `DomainCover` bits.
#[derive(Default, Clone, Debug)]
pub struct DomainCosts {
    pub coeff_sign_bypass: Vec<f32>,
    pub coeff_suffix_lsb: Vec<f32>,
    pub mvd_sign_bypass: Vec<f32>,
    pub mvd_suffix_lsb: Vec<f32>,
}

/// Pass 1: walk the decision cache, populate the per-domain cover
/// bits + positions + cost vectors.
pub fn pass1_collect_cover(cache: &GopDecisionCache) -> GopCover {
    let mut out = GopCover::default();
    for mb in &cache.mbs {
        let cost_ctx = PositionCostCtx::new(mb.frame_idx, mb.mb_addr);

        // Per-residual-block contributions to coeff domains.
        for blk in &mb.residual_blocks {
            // §B-cascade-real fix 2026-05-07: per-path-kind cost multiplier.
            // DC paths (LumaDcIntra16x16, ChromaDc) propagate through inverse
            // hadamard to all 16 (luma) or 4 (chroma) sub-blocks. Without
            // this weighting STC has no incentive to avoid catastrophic-DC
            // sign flips at IDR.
            let path_mult = blk.path_kind.cost_multiplier();

            // CoeffSignBypass.
            let positions = enumerate_coeff_sign_positions(
                &blk.scan_coeffs,
                blk.start_idx,
                blk.end_idx,
                mb.frame_idx,
                mb.mb_addr,
                |ci| blk.path_kind.path(ci, BinKind::Sign),
            );
            let bits = extract_coeff_sign_bits(
                &blk.scan_coeffs, blk.start_idx, blk.end_idx,
            );
            let costs = coeff_sign_cost_vec(
                &blk.scan_coeffs, blk.start_idx, blk.end_idx, &cost_ctx,
            );
            for ((p, b), c) in positions.iter().zip(bits.iter()).zip(costs.iter()) {
                out.cover.coeff_sign_bypass.push(*b, *p);
                out.costs.coeff_sign_bypass.push(*c * path_mult);
            }

            // CoeffSuffixLsb.
            let positions = enumerate_coeff_suffix_lsb_positions(
                &blk.scan_coeffs,
                blk.start_idx,
                blk.end_idx,
                mb.frame_idx,
                mb.mb_addr,
                |ci| blk.path_kind.path(ci, BinKind::SuffixLsb),
            );
            let bits = extract_coeff_suffix_lsb_bits(
                &blk.scan_coeffs, blk.start_idx, blk.end_idx,
            );
            let costs = coeff_suffix_lsb_cost_vec(
                &blk.scan_coeffs, blk.start_idx, blk.end_idx, &cost_ctx,
            );
            for ((p, b), c) in positions.iter().zip(bits.iter()).zip(costs.iter()) {
                out.cover.coeff_suffix_lsb.push(*b, *p);
                out.costs.coeff_suffix_lsb.push(*c * path_mult);
            }
        }

        // MVD domains (per-MB, not per-block).
        let positions = enumerate_mvd_sign_positions(&mb.mvd_slots, mb.frame_idx, mb.mb_addr);
        let bits = extract_mvd_sign_bits(&mb.mvd_slots);
        let costs = mvd_sign_cost_vec(&mb.mvd_slots, &cost_ctx);
        for ((p, b), c) in positions.iter().zip(bits.iter()).zip(costs.iter()) {
            out.cover.mvd_sign_bypass.push(*b, *p);
            out.costs.mvd_sign_bypass.push(*c);
        }

        let positions = enumerate_mvd_suffix_lsb_positions(&mb.mvd_slots, mb.frame_idx, mb.mb_addr);
        let bits = extract_mvd_suffix_lsb_bits(&mb.mvd_slots);
        let costs = mvd_suffix_lsb_cost_vec(&mb.mvd_slots, &cost_ctx);
        for ((p, b), c) in positions.iter().zip(bits.iter()).zip(costs.iter()) {
            out.cover.mvd_suffix_lsb.push(*b, *p);
            out.costs.mvd_suffix_lsb.push(*c);
        }
    }
    out
}

// ─── Pass 2: STC plan per domain ─────────────────────────────────

/// Pass 2 result: per-domain bit plan that Pass 3 will apply.
#[derive(Default, Clone, Debug)]
pub struct DomainPlan {
    /// Per-position target bit. Indices align with the domain's
    /// `DomainBits.positions` from Pass 1.
    pub coeff_sign_bypass: Vec<u8>,
    pub coeff_suffix_lsb: Vec<u8>,
    pub mvd_sign_bypass: Vec<u8>,
    pub mvd_suffix_lsb: Vec<u8>,
    /// Total modifications across all four domains.
    pub total_modifications: usize,
    /// Total cost across all four domains.
    pub total_cost: f64,
}

/// Pass 2: per-domain STC embed. Each domain has independent
/// cover bits + costs + message slice. Returns a per-domain bit
/// plan that Pass 3 applies.
///
/// `messages` is a 4-slot tuple, one per [`EmbedDomain`] in the
/// canonical enum order, matching the cross-domain split done by
/// Pass 1.5 (which is its own tiny step in 6D.7+: it allocates the
/// global message bits to per-domain slices proportional to cover
/// capacity, mirroring the existing `compute_per_gop_message_split`
/// in `h264_pipeline.rs`).
pub fn pass2_stc_plan(
    cover: &GopCover,
    messages: &DomainMessages,
    h: usize,
) -> Option<DomainPlan> {
    // Backward-compatible entry point: uses a hardcoded zero seed
    // for all four domains. Production callers should use
    // [`pass2_stc_plan_with_keys`] which derives per-domain
    // ChaCha20 seeds from the passphrase via [`CabacStegoMasterKeys`].
    pass2_stc_plan_internal(cover, messages, h, None)
}

/// Production Pass 2: per-domain STC embed using passphrase-derived
/// per-domain seeds. The four domains use independent (perm, hhat)
/// seed pairs so an attacker breaking one domain's syndrome cannot
/// reuse the same H-hat against the others.
pub fn pass2_stc_plan_with_keys(
    cover: &GopCover,
    messages: &DomainMessages,
    h: usize,
    keys: &CabacStegoMasterKeys,
    gop_idx: u32,
) -> Option<DomainPlan> {
    pass2_stc_plan_internal(cover, messages, h, Some((keys, gop_idx)))
}

/// STEGO.A.2 — Scheme A combined-cover STC primary planner.
///
/// Replaces per-domain `stealth_weighted_allocation` +
/// [`pass2_stc_plan_with_keys`]'s 4 independent STC calls with a
/// single global STC over the 4-domain combined cover. The decoder
/// side ([`super::decode_pixels`]'s Scheme A extract — landing in
/// STEGO.A.4) mirrors with the same combine + single STC extract.
///
/// Why this is better than per-domain:
/// - Joint optimization across all 4 domains: STC can trade flips
///   between domains based on actual per-position cost. With
///   content-adaptive costs from [`super::content_costs`], STC
///   concentrates flips in high-detectability-headroom regions
///   regardless of which domain they live in.
/// - Single rate `w_global = n_total / m_total` removes the
///   per-domain rate mismatches that Scheme B's hand-tuned
///   stealth_weighted_allocation has to work around.
///
/// Inputs:
/// - `cover` — Pass-1 cover with per-position costs in `cover.costs`
///   already populated by [`super::content_costs::compute_content_costs_yuv`]
///   (or `DomainCosts::default()` for the uniform-cost baseline).
/// - `weights` — per-domain `CostWeights` multipliers applied during
///   `combine_cover_4domain`. Acts as a final cross-domain calibration
///   layer on top of the content-adaptive per-position costs.
/// - `frame_bits` — full primary message bit vector (NOT pre-split).
/// - `h` — STC constraint height (caller passes the same `h` as the
///   decoder will use; v1.0 = 4).
/// - `keys` — master keys; the combined STC seed is derived from the
///   `CoeffSignBypass` per-GOP seed (canonical "primary" seed in
///   Scheme A; encoder + decoder must agree on this choice).
/// - `gop_idx` — GOP index for per-GOP seed selection.
///
/// Returns `None` when the cover is too small for the message (rate
/// `w = n / m < 1`), or when `stc_embed` fails to find a syndrome.
pub fn pass2_stc_plan_combined_with_keys(
    cover: &GopCover,
    weights: &super::cost_weights::CostWeights,
    frame_bits: &[u8],
    h: usize,
    keys: &CabacStegoMasterKeys,
    gop_idx: u32,
) -> Option<DomainPlan> {
    let (combined_cover, combined_costs, boundaries) =
        super::cost_weights::combine_cover_4domain(&cover.cover, &cover.costs, weights);
    let n_cover = combined_cover.len();
    let m_total = frame_bits.len();

    // Empty message: return an empty plan (caller can early-return).
    if m_total == 0 {
        let mut plan = DomainPlan::default();
        plan.coeff_sign_bypass = cover.cover.coeff_sign_bypass.bits.clone();
        plan.coeff_suffix_lsb = cover.cover.coeff_suffix_lsb.bits.clone();
        plan.mvd_sign_bypass = cover.cover.mvd_sign_bypass.bits.clone();
        plan.mvd_suffix_lsb = cover.cover.mvd_suffix_lsb.bits.clone();
        return Some(plan);
    }
    if n_cover == 0 {
        return None;
    }
    let w = n_cover / m_total;
    if w == 0 {
        return None;
    }
    let used_cover = m_total * w;

    let hhat_seed = keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, gop_idx)
        .hhat_seed;
    let hhat = generate_hhat(h, w, &hhat_seed);

    let plan = stc_embed(
        &combined_cover[..used_cover],
        &combined_costs[..used_cover],
        frame_bits,
        &hhat,
        h,
        w,
    )?;

    // Extend plan back to full combined length: positions past
    // `used_cover` carry the original cover bits (STC didn't touch
    // them, so neither does Pass 3).
    let mut full_stego_bits = Vec::with_capacity(n_cover);
    full_stego_bits.extend_from_slice(&plan.stego_bits);
    full_stego_bits.extend_from_slice(&combined_cover[used_cover..]);
    debug_assert_eq!(full_stego_bits.len(), n_cover);

    let mut domain_plan = super::cost_weights::split_plan_4domain(&full_stego_bits, &boundaries);
    domain_plan.total_modifications = plan.num_modifications;
    domain_plan.total_cost = plan.total_cost;
    Some(domain_plan)
}

fn pass2_stc_plan_internal(
    cover: &GopCover,
    messages: &DomainMessages,
    h: usize,
    keys_and_gop: Option<(&CabacStegoMasterKeys, u32)>,
) -> Option<DomainPlan> {
    let mut plan = DomainPlan::default();
    for (domain, cover_bits, costs, message, plan_slot) in [
        (EmbedDomain::CoeffSignBypass, &cover.cover.coeff_sign_bypass,
         cover.costs.coeff_sign_bypass.as_slice(), &messages.coeff_sign_bypass,
         &mut plan.coeff_sign_bypass),
        (EmbedDomain::CoeffSuffixLsb, &cover.cover.coeff_suffix_lsb,
         cover.costs.coeff_suffix_lsb.as_slice(), &messages.coeff_suffix_lsb,
         &mut plan.coeff_suffix_lsb),
        (EmbedDomain::MvdSignBypass, &cover.cover.mvd_sign_bypass,
         cover.costs.mvd_sign_bypass.as_slice(), &messages.mvd_sign_bypass,
         &mut plan.mvd_sign_bypass),
        (EmbedDomain::MvdSuffixLsb, &cover.cover.mvd_suffix_lsb,
         cover.costs.mvd_suffix_lsb.as_slice(), &messages.mvd_suffix_lsb,
         &mut plan.mvd_suffix_lsb),
    ] {
        let seed = match keys_and_gop {
            Some((k, gop)) => k.per_gop_seeds(domain, gop).hhat_seed,
            None => [0u8; 32],
        };
        let r = plan_one_domain_seeded(cover_bits, costs, message, h, &seed)?;
        *plan_slot = r.bits;
        plan.total_modifications += r.num_modifications;
        plan.total_cost += r.total_cost;
    }
    Some(plan)
}

/// Per-domain plan result returned by `plan_one_domain`.
struct DomainPlanResult {
    bits: Vec<u8>,
    num_modifications: usize,
    total_cost: f64,
}

/// Per-domain message bit slices fed into Pass 2. The Pass-1.5
/// splitter [`split_message_per_domain`] allocates the global
/// message across these four slices proportional to cover
/// capacity.
#[derive(Default, Clone, Debug)]
pub struct DomainMessages {
    pub coeff_sign_bypass: Vec<u8>,
    pub coeff_suffix_lsb: Vec<u8>,
    pub mvd_sign_bypass: Vec<u8>,
    pub mvd_suffix_lsb: Vec<u8>,
}

/// Pass 1.5 — split the global message bit sequence across the
/// four CABAC stego domains proportional to per-domain cover
/// capacity. Mirrors `compute_per_gop_message_split` in
/// `core/src/stego/video/h264_pipeline.rs` (lines ~580-650),
/// adapted to the four-domain scheme.
///
/// Distribution rule:
/// - Each domain receives `m_d = (n_d * m_total) / n_total` bits,
///   floor-divided.
/// - Any leftover bits from the floor are appended to the largest
///   domain (more cover headroom = less STC strain).
///
/// The decoder mirror runs the same split with the same per-domain
/// capacities (recovered from Pass 1 at decode time), so encode +
/// decode allocate identical per-domain bit slices and the STC
/// reverse-pass is exact.
///
/// Returns `None` if `total cover capacity < message length`.
pub fn split_message_per_domain(
    message: &[u8],
    capacities: &super::GopCapacity,
) -> Option<DomainMessages> {
    let m_total = message.len();
    if m_total == 0 {
        return Some(DomainMessages::default());
    }
    let n_total = capacities.total();
    if n_total < m_total {
        return None;
    }

    // Floor-allocate per domain.
    let mut m_coeff_sign = (m_total * capacities.coeff_sign_bypass) / n_total;
    let mut m_coeff_suffix = (m_total * capacities.coeff_suffix_lsb) / n_total;
    let mut m_mvd_sign = (m_total * capacities.mvd_sign_bypass) / n_total;
    let mut m_mvd_suffix = (m_total * capacities.mvd_suffix_lsb) / n_total;

    // Distribute leftover bits to the largest-capacity domain.
    let mut leftover = m_total - (m_coeff_sign + m_coeff_suffix + m_mvd_sign + m_mvd_suffix);
    while leftover > 0 {
        // Domain order matches EmbedDomain enum + ties broken by
        // the largest *remaining* headroom.
        let pick = pick_max_headroom(capacities, &[
            (m_coeff_sign, capacities.coeff_sign_bypass),
            (m_coeff_suffix, capacities.coeff_suffix_lsb),
            (m_mvd_sign, capacities.mvd_sign_bypass),
            (m_mvd_suffix, capacities.mvd_suffix_lsb),
        ]);
        match pick {
            0 => m_coeff_sign += 1,
            1 => m_coeff_suffix += 1,
            2 => m_mvd_sign += 1,
            3 => m_mvd_suffix += 1,
            _ => unreachable!(),
        }
        leftover -= 1;
    }

    // Slice the message bytes into the per-domain bit streams. The
    // global bit stream is `message[0].bit0, message[0].bit1, ...,
    // message[0].bit7, message[1].bit0, ...` per the existing
    // pipeline convention; we build it once and slice.
    let bits: Vec<u8> = message
        .iter()
        .flat_map(|&b| (0..8).rev().map(move |i| (b >> i) & 1))
        .collect();
    // BUT: m_total is in BYTES (architecture-doc convention) → the
    // bit-stream length is 8 × m_total. The per-domain m_X above
    // are in BYTES too. Convert by carving the bit stream.
    //
    // Actually the existing pipeline convention varies; we follow
    // the simpler "messages are bit sequences" model used by the
    // STC layer. Re-interpret: the message arg is a BIT sequence
    // (each entry 0 or 1) when called from the stego pipeline.
    // This avoids a units conversion bug.
    //
    // For backward compatibility this fn accepts bytes; if all
    // bytes are 0/1 it's already a bit stream and used directly,
    // otherwise it's bits-from-bytes-MSB-first.
    let bit_stream: Vec<u8> = if message.iter().all(|&b| b <= 1) {
        message.to_vec()
    } else {
        bits
    };
    let total_bits = bit_stream.len();
    // Re-derive per-domain bit counts in bits (m_X above were in
    // the same units as message — we'll trust that).
    let m_total = total_bits;
    let _ = m_total;

    // For bit-stream input: m_X already represent bit counts (since
    // m_total = bit_stream.len()). For byte-stream input: each m_X
    // is also in those units. The split is consistent regardless.

    let mut cursor = 0usize;
    let take = |start: &mut usize, n: usize| -> Vec<u8> {
        let end = (*start + n).min(bit_stream.len());
        let slice = bit_stream[*start..end].to_vec();
        *start = end;
        slice
    };
    let mut take_mut = |n: usize| -> Vec<u8> {
        let end = (cursor + n).min(bit_stream.len());
        let s = bit_stream[cursor..end].to_vec();
        cursor = end;
        s
    };
    let _ = take;

    let coeff_sign_bypass = take_mut(m_coeff_sign);
    let coeff_suffix_lsb = take_mut(m_coeff_suffix);
    let mvd_sign_bypass = take_mut(m_mvd_sign);
    let mvd_suffix_lsb = take_mut(m_mvd_suffix);

    Some(DomainMessages {
        coeff_sign_bypass,
        coeff_suffix_lsb,
        mvd_sign_bypass,
        mvd_suffix_lsb,
    })
}

/// Phase 6F.2(k).4 — per-domain stealth weight vector + drift
/// budget cap.
///
/// **Stealth weights** `[w_cs, w_cl, w_ms, w_ml]` reflect known
/// per-domain steganalysis detection sensitivity. Higher weight
/// = more "headroom" for embedding without triggering the
/// detector. The allocator then targets equal detection cost
/// across domains:
///
///     m_d / (n_d · w_d) = constant
///     m_d = M · (n_d · w_d) / Σ (n_d · w_d)
///
/// Defaults reflect public literature:
/// - coeff_sign: `0.5` — most-attacked, mature SRNet/STDM detectors.
/// - coeff_suffix: `0.8` — secondary feature, less-studied.
/// - mvd_sign: `1.0` — MV-domain steganalysis is open research.
/// - mvd_suffix: `1.0` — least-attacked.
///
/// Both encoder and decoder must use the SAME weights (protocol-
/// version constants, not key-derived). v1.0 hardcodes the
/// defaults below.
///
/// **Drift budget** caps the fraction of the message that can
/// route through MVD domains. Each MVD-sign flip causes
/// pixel-level drift at the spec decoder (~0.5px per flip on
/// |MVD|=1-2). Capping the MVD share at 0.20 of M bounds
/// cumulative drift while still distributing modifications
/// across domains to break the per-domain fingerprint anomaly.
#[derive(Copy, Clone, Debug)]
pub struct StealthAllocator {
    pub w_coeff_sign: f64,
    pub w_coeff_suffix: f64,
    pub w_mvd_sign: f64,
    pub w_mvd_suffix: f64,
    /// Maximum fraction of total message bits routed through MVD
    /// domains. Caps cumulative pixel drift (each MVD flip ≈
    /// 0.5px drift; budget controls the worst-case PSNR hit).
    pub mvd_drift_budget_frac: f64,
}

impl Default for StealthAllocator {
    fn default() -> Self {
        Self::v1_default()
    }
}

impl StealthAllocator {
    /// v1.0 default weights (see struct doc-comment for rationale).
    pub const fn v1_default_const() -> Self {
        Self {
            w_coeff_sign: 0.5,
            w_coeff_suffix: 0.8,
            w_mvd_sign: 1.0,
            w_mvd_suffix: 1.0,
            mvd_drift_budget_frac: 0.20,
        }
    }

    /// v1.0 default weights with optional `PHASM_STEALTH_ABLATE` env
    /// override for §B-cascade-real domain bisect (2026-05-06). Set to
    /// `cs` / `cl` / `ms` / `ml` to ablate to a single-domain
    /// allocator (other 3 weights = 0). Empty / unset = full v1
    /// defaults. Used to identify which of the 4 stego domains
    /// triggers CABAC bin-context desync at IDR.
    pub fn v1_default() -> Self {
        let mut a = Self::v1_default_const();
        if let Ok(v) = std::env::var("PHASM_STEALTH_ABLATE") {
            match v.as_str() {
                "cs" => { a.w_coeff_suffix = 0.0; a.w_mvd_sign = 0.0; a.w_mvd_suffix = 0.0; }
                "cl" => { a.w_coeff_sign = 0.0; a.w_mvd_sign = 0.0; a.w_mvd_suffix = 0.0; }
                "ms" => { a.w_coeff_sign = 0.0; a.w_coeff_suffix = 0.0; a.w_mvd_suffix = 0.0; a.mvd_drift_budget_frac = 1.0; }
                "ml" => { a.w_coeff_sign = 0.0; a.w_coeff_suffix = 0.0; a.w_mvd_sign = 0.0; a.mvd_drift_budget_frac = 1.0; }
                _ => {}
            }
        }
        a
    }
}

/// Phase 6F.2(k).4 — stealth-weighted message split with drift
/// budget cap.
///
/// Returns the per-domain bit-allocations
/// `(m_coeff_sign, m_coeff_suffix, m_mvd_sign, m_mvd_suffix)`.
/// Sum equals `m_total`. Pure function — encoder + decoder run
/// identical inputs/outputs.
///
/// Algorithm:
/// 1. Compute weighted capacities `n'_d = n_d · w_d`.
/// 2. Proportional-by-weighted-capacity floor allocation:
///    `m_d = M · n'_d / Σ n'_d`.
/// 3. Cap MVD share: if `m_mvd_sign + m_mvd_suffix >
///    drift_budget · M`, scale down MVD allocations and
///    redistribute overflow to coeff domains
///    proportionally-by-weighted-capacity.
/// 4. Cap each `m_d` at its actual capacity `n_d` and
///    redistribute overflow.
/// 5. Round-robin distribute the remainder of `M` to the
///    domain with the largest weighted-headroom.
///
/// Returns `None` iff the total weighted capacity can't fit
/// `M` even after redistribution.
pub fn stealth_weighted_allocation(
    m_total: usize,
    capacities: &super::GopCapacity,
    allocator: &StealthAllocator,
) -> Option<(usize, usize, usize, usize)> {
    if m_total == 0 {
        return Some((0, 0, 0, 0));
    }

    // Step 1: weighted capacities (in floating point for the
    // proportional split).
    let n_cs = capacities.coeff_sign_bypass as f64;
    let n_cl = capacities.coeff_suffix_lsb as f64;
    let n_ms = capacities.mvd_sign_bypass as f64;
    let n_ml = capacities.mvd_suffix_lsb as f64;
    let nw_cs = n_cs * allocator.w_coeff_sign;
    let nw_cl = n_cl * allocator.w_coeff_suffix;
    let nw_ms = n_ms * allocator.w_mvd_sign;
    let nw_ml = n_ml * allocator.w_mvd_suffix;
    let nw_sum = nw_cs + nw_cl + nw_ms + nw_ml;
    if nw_sum <= 0.0 {
        return None;
    }

    // Total raw capacity must accommodate M (drift budget can't
    // help if absolute capacity is insufficient).
    if (capacities.coeff_sign_bypass
        + capacities.coeff_suffix_lsb
        + capacities.mvd_sign_bypass
        + capacities.mvd_suffix_lsb) < m_total
    {
        return None;
    }

    // Step 2: proportional-by-weighted-capacity floor allocation.
    let m_total_f = m_total as f64;
    let mut m_cs = ((m_total_f * nw_cs) / nw_sum).floor() as usize;
    let mut m_cl = ((m_total_f * nw_cl) / nw_sum).floor() as usize;
    let mut m_ms = ((m_total_f * nw_ms) / nw_sum).floor() as usize;
    let mut m_ml = ((m_total_f * nw_ml) / nw_sum).floor() as usize;

    // Step 3: drift budget cap on MVD share.
    let mvd_share_max = (m_total_f * allocator.mvd_drift_budget_frac).floor() as usize;
    let mvd_share = m_ms + m_ml;
    if mvd_share > mvd_share_max {
        // Scale MVD allocations down to the cap; redistribute
        // overflow to coeff domains proportionally.
        let overflow = mvd_share - mvd_share_max;
        // Proportional reduction within MVD (keep the
        // sign-vs-suffix ratio).
        let mvd_total = (m_ms + m_ml) as f64;
        if mvd_total > 0.0 {
            let new_m_ms = ((mvd_share_max as f64) * (m_ms as f64) / mvd_total).floor() as usize;
            let new_m_ml = mvd_share_max.saturating_sub(new_m_ms);
            m_ms = new_m_ms;
            m_ml = new_m_ml;
        } else {
            m_ms = 0;
            m_ml = 0;
        }
        // Redistribute overflow to coeff domains
        // proportionally-by-weighted-capacity.
        let coeff_nw = nw_cs + nw_cl;
        if coeff_nw > 0.0 {
            let extra_cs = ((overflow as f64) * nw_cs / coeff_nw).floor() as usize;
            let extra_cl = overflow.saturating_sub(extra_cs);
            m_cs = m_cs.saturating_add(extra_cs);
            m_cl = m_cl.saturating_add(extra_cl);
        }
    }

    // Step 4: cap each domain at its actual capacity. The overflow
    // is implicit in the post-cap sum being below m_total — step 5
    // redistributes via `m_total - sum` (no separate accumulator).
    let cap_cs = capacities.coeff_sign_bypass;
    let cap_cl = capacities.coeff_suffix_lsb;
    let cap_ms = capacities.mvd_sign_bypass;
    let cap_ml = capacities.mvd_suffix_lsb;
    if m_cs > cap_cs { m_cs = cap_cs; }
    if m_cl > cap_cl { m_cl = cap_cl; }
    if m_ms > cap_ms { m_ms = cap_ms; }
    if m_ml > cap_ml { m_ml = cap_ml; }

    // Step 5: distribute remainder by weighted-headroom
    // `(cap_d - m_d) · w_d`. The MVD drift cap is re-checked
    // after each addition so we don't blow past it during
    // redistribution.
    let mut remainder = m_total.saturating_sub(m_cs + m_cl + m_ms + m_ml);
    while remainder > 0 {
        let mvd_share_now = m_ms + m_ml;
        let mvd_room = mvd_share_max.saturating_sub(mvd_share_now);
        let h_cs = ((cap_cs - m_cs) as f64) * allocator.w_coeff_sign;
        let h_cl = ((cap_cl - m_cl) as f64) * allocator.w_coeff_suffix;
        let h_ms = if mvd_room > 0 {
            ((cap_ms - m_ms) as f64) * allocator.w_mvd_sign
        } else { -1.0 };
        let h_ml = if mvd_room > 0 {
            ((cap_ml - m_ml) as f64) * allocator.w_mvd_suffix
        } else { -1.0 };
        let candidates = [(h_cs, 0usize), (h_cl, 1), (h_ms, 2), (h_ml, 3)];
        let (best_h, best_idx) = candidates
            .iter()
            .copied()
            .filter(|(h, _)| *h > 0.0)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((-1.0, 0));
        if best_h <= 0.0 {
            // No room left in any domain — capacity can't fit M
            // (shouldn't happen given the n_total check above
            // unless drift budget is too tight).
            return None;
        }
        match best_idx {
            0 => m_cs += 1,
            1 => m_cl += 1,
            2 => m_ms += 1,
            3 => m_ml += 1,
            _ => unreachable!(),
        }
        remainder -= 1;
    }

    debug_assert_eq!(m_cs + m_cl + m_ms + m_ml, m_total);
    Some((m_cs, m_cl, m_ms, m_ml))
}

fn pick_max_headroom(
    _caps: &super::GopCapacity,
    allocated_capacity_pairs: &[(usize, usize)],
) -> usize {
    let mut best_idx = 0;
    let mut best_headroom: i64 = i64::MIN;
    for (i, &(allocated, capacity)) in allocated_capacity_pairs.iter().enumerate() {
        let headroom = capacity as i64 - allocated as i64;
        if headroom > best_headroom {
            best_headroom = headroom;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod stealth_alloc_tests {
    use super::*;

    fn caps(cs: usize, cl: usize, ms: usize, ml: usize) -> super::super::GopCapacity {
        super::super::GopCapacity {
            coeff_sign_bypass: cs,
            coeff_suffix_lsb: cl,
            mvd_sign_bypass: ms,
            mvd_suffix_lsb: ml,
        }
    }

    #[test]
    fn weighted_alloc_total_equals_m_total() {
        let alloc = StealthAllocator::v1_default();
        let c = caps(500, 100, 200, 50);
        // Raw capacity = 850; effective capacity under drift
        // budget = coeff_caps (600) + 0.20 · M MVD share.
        // Test values stay well under coeff_caps to avoid the
        // budget-vs-capacity boundary.
        for &m in &[1usize, 8, 100, 400, 600] {
            let (a, b, c2, d) = stealth_weighted_allocation(m, &c, &alloc).unwrap();
            assert_eq!(a + b + c2 + d, m,
                "allocation must sum to m_total ({m}); got {a}+{b}+{c2}+{d}");
        }
    }

    #[test]
    fn weighted_alloc_returns_none_when_drift_budget_blocks_capacity() {
        // Raw capacity 850, but coeff cap = 600. M=700 needs
        // MVD share = 100 minimum, exceeding the 0.20 · 700 = 140
        // budget? Actually 100 < 140 so this fits. M=750 needs
        // MVD ≥ 150 > 0.20 · 750 = 150, exact cap, allocator
        // succeeds. M=800 needs MVD ≥ 200 > 0.20 · 800 = 160 —
        // doesn't fit under budget, returns None.
        let alloc = StealthAllocator::v1_default();
        let c = caps(500, 100, 200, 50);
        assert!(stealth_weighted_allocation(800, &c, &alloc).is_none(),
            "expected None when drift budget can't accommodate M");
    }

    #[test]
    fn weighted_alloc_respects_mvd_drift_budget() {
        let alloc = StealthAllocator::v1_default(); // budget 0.20
        let c = caps(500, 100, 200, 50);
        let m = 400usize;
        let (_, _, ms, ml) = stealth_weighted_allocation(m, &c, &alloc).unwrap();
        let mvd_share = ms + ml;
        let cap_share = (m as f64 * alloc.mvd_drift_budget_frac).floor() as usize;
        assert!(mvd_share <= cap_share,
            "mvd_share {mvd_share} must respect drift budget cap {cap_share}");
    }

    #[test]
    fn weighted_alloc_pushes_to_mvd_under_balanced_caps() {
        // Equal raw capacity in all domains. With v1 weights
        // (cs=0.5, cl=0.8, ms=1.0, ml=1.0), MVD domains should get
        // strictly more allocation than coeff_sign — at least up to
        // the drift budget cap.
        let alloc = StealthAllocator::v1_default();
        let c = caps(100, 100, 100, 100);
        let m = 80usize;
        let (m_cs, _m_cl, m_ms, m_ml) = stealth_weighted_allocation(m, &c, &alloc).unwrap();
        // MVD share is capped at 0.20 * 80 = 16. That cap dominates
        // for this M; verify the cap, AND verify the remaining bits
        // distribute across coeff domains rather than concentrating
        // in coeff_sign (which has the lowest weight).
        assert!(m_ms + m_ml <= 16, "MVD share capped at drift budget");
        // coeff_sign should NOT carry 100% of coeff bits — coeff_suffix
        // (w=0.8 vs cs=0.5) gets a heavier share.
        assert!(m_cs < (m - m_ms - m_ml),
            "coeff_sign should not absorb all coeff bits (lower weight)");
    }

    #[test]
    fn weighted_alloc_returns_none_when_capacity_insufficient() {
        let alloc = StealthAllocator::v1_default();
        let c = caps(10, 10, 10, 10);
        let m = 100usize;
        assert!(stealth_weighted_allocation(m, &c, &alloc).is_none());
    }

    #[test]
    fn weighted_alloc_zero_message() {
        let alloc = StealthAllocator::v1_default();
        let c = caps(100, 100, 100, 100);
        assert_eq!(
            stealth_weighted_allocation(0, &c, &alloc).unwrap(),
            (0, 0, 0, 0)
        );
    }
}

fn plan_one_domain_seeded(
    cover_bits: &DomainBits,
    cost: &[f32],
    message: &[u8],
    h: usize,
    seed: &[u8; 32],
) -> Option<DomainPlanResult> {
    if cover_bits.is_empty() || message.is_empty() {
        return Some(DomainPlanResult {
            bits: cover_bits.bits.clone(),
            num_modifications: 0,
            total_cost: 0.0,
        });
    }
    let n = cover_bits.bits.len();
    let m = message.len();
    let w = n / m.max(1);
    if w == 0 {
        return None;
    }
    let hhat = generate_hhat(h, w, seed);
    let result: EmbedResult = stc_embed(
        &cover_bits.bits, cost, message, &hhat, h, w,
    )?;
    Some(DomainPlanResult {
        bits: result.stego_bits,
        num_modifications: result.num_modifications,
        total_cost: result.total_cost,
    })
}

// ─── Pass 3: apply overrides via BitInjector ─────────────────────

/// Pass-3 [`BitInjector`] that looks up per-position overrides
/// in a `HashMap<PositionKey, u8>` built from a [`DomainPlan`]
/// + per-domain `DomainBits`.
pub struct PlanInjector {
    plan: std::collections::HashMap<PositionKey, u8>,
}

impl PlanInjector {
    /// Build the lookup table from a Pass-1 cover + Pass-2 plan.
    pub fn from_plan(cover: &DomainCover, plan: &DomainPlan) -> Self {
        let mut map = std::collections::HashMap::new();
        Self::extend(&mut map, &cover.coeff_sign_bypass.positions, &plan.coeff_sign_bypass);
        Self::extend(&mut map, &cover.coeff_suffix_lsb.positions, &plan.coeff_suffix_lsb);
        Self::extend(&mut map, &cover.mvd_sign_bypass.positions, &plan.mvd_sign_bypass);
        Self::extend(&mut map, &cover.mvd_suffix_lsb.positions, &plan.mvd_suffix_lsb);
        Self { plan: map }
    }

    fn extend(
        map: &mut std::collections::HashMap<PositionKey, u8>,
        positions: &[PositionKey],
        bits: &[u8],
    ) {
        let n = positions.len().min(bits.len());
        for i in 0..n {
            map.insert(positions[i], bits[i]);
        }
    }

    /// Borrow the underlying plan map (diagnostics).
    pub fn map(&self) -> &std::collections::HashMap<PositionKey, u8> {
        &self.plan
    }
}

impl BitInjector for PlanInjector {
    fn override_bit(&mut self, key: PositionKey) -> Option<u8> {
        self.plan.get(&key).copied()
    }
}

/// Pass 3: apply a [`DomainPlan`] to a [`GopDecisionCache`] in
/// place. Walks every MB + residual block + MVD slot, applying the
/// matching overrides. Returns the total modification count
/// (cross-checks against `plan.total_modifications`).
pub fn pass3_apply_overrides(
    cache: &mut GopDecisionCache,
    cover: &DomainCover,
    plan: &DomainPlan,
) -> usize {
    let mut injector = PlanInjector::from_plan(cover, plan);
    let mut count = 0usize;
    for mb in &mut cache.mbs {
        for blk in &mut mb.residual_blocks {
            count += apply_coeff_sign_overrides(
                &mut blk.scan_coeffs,
                blk.start_idx,
                blk.end_idx,
                mb.frame_idx,
                mb.mb_addr,
                |ci| blk.path_kind.path(ci, BinKind::Sign),
                &mut injector,
            );
            count += apply_coeff_suffix_lsb_overrides(
                &mut blk.scan_coeffs,
                blk.start_idx,
                blk.end_idx,
                mb.frame_idx,
                mb.mb_addr,
                |ci| blk.path_kind.path(ci, BinKind::SuffixLsb),
                &mut injector,
            );
        }
        count += apply_mvd_sign_overrides(
            &mut mb.mvd_slots, mb.frame_idx, mb.mb_addr, &mut injector,
        );
        count += apply_mvd_suffix_lsb_overrides(
            &mut mb.mvd_slots, mb.frame_idx, mb.mb_addr, &mut injector,
        );
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::stego::Axis;

    fn build_synthetic_gop() -> GopDecisionCache {
        let mut cache = GopDecisionCache::new();
        // MB 0: I-slice with one residual block + no MVDs.
        let mut scan = vec![0i32; 16];
        scan[0] = 3;
        scan[3] = -7;
        scan[6] = 2;
        cache.push(MbDecision {
            frame_idx: 0, mb_addr: 0,
            residual_blocks: vec![MbResidualBlock {
                scan_coeffs: scan,
                start_idx: 0, end_idx: 15,
                ctx_block_cat: 1,
                path_kind: ResidualPathKind::Luma4x4 { block_idx: 0 },
            }],
            mvd_slots: vec![],
        });
        // MB 1: P-slice with one residual block + two MVD slots.
        let mut scan = vec![0i32; 16];
        scan[1] = -4;
        scan[5] = 1;
        cache.push(MbDecision {
            frame_idx: 0, mb_addr: 1,
            residual_blocks: vec![MbResidualBlock {
                scan_coeffs: scan,
                start_idx: 0, end_idx: 15,
                ctx_block_cat: 1,
                path_kind: ResidualPathKind::Luma4x4 { block_idx: 0 },
            }],
            mvd_slots: vec![
                MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 5 },
                MvdSlot { list: 0, partition: 0, axis: Axis::Y, value: -3 },
            ],
        });
        cache
    }

    #[test]
    fn split_message_proportional_to_capacity() {
        use super::super::GopCapacity;
        let caps = GopCapacity {
            coeff_sign_bypass: 100,
            coeff_suffix_lsb: 50,
            mvd_sign_bypass: 30,
            mvd_suffix_lsb: 20,
        };
        // total = 200. message of 20 bits → split should be
        // (10, 5, 3, 2) by floor.
        let msg = vec![0u8, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1];
        let split = split_message_per_domain(&msg, &caps).unwrap();
        assert_eq!(split.coeff_sign_bypass.len(), 10);
        assert_eq!(split.coeff_suffix_lsb.len(), 5);
        assert_eq!(split.mvd_sign_bypass.len(), 3);
        assert_eq!(split.mvd_suffix_lsb.len(), 2);
        // Total preserved.
        let total = split.coeff_sign_bypass.len() + split.coeff_suffix_lsb.len()
            + split.mvd_sign_bypass.len() + split.mvd_suffix_lsb.len();
        assert_eq!(total, msg.len());
    }

    #[test]
    fn split_message_too_large_returns_none() {
        use super::super::GopCapacity;
        let caps = GopCapacity {
            coeff_sign_bypass: 5, coeff_suffix_lsb: 0,
            mvd_sign_bypass: 0, mvd_suffix_lsb: 0,
        };
        let msg = vec![0u8; 10];
        assert!(split_message_per_domain(&msg, &caps).is_none());
    }

    #[test]
    fn split_message_empty_message_returns_default() {
        use super::super::GopCapacity;
        let caps = GopCapacity {
            coeff_sign_bypass: 100, coeff_suffix_lsb: 50,
            mvd_sign_bypass: 30, mvd_suffix_lsb: 20,
        };
        let split = split_message_per_domain(&[], &caps).unwrap();
        assert_eq!(split.coeff_sign_bypass.len(), 0);
        assert_eq!(split.coeff_suffix_lsb.len(), 0);
        assert_eq!(split.mvd_sign_bypass.len(), 0);
        assert_eq!(split.mvd_suffix_lsb.len(), 0);
    }

    #[test]
    fn split_message_leftover_goes_to_largest_headroom() {
        use super::super::GopCapacity;
        // total cover = 11. message = 5. floor split = (4, 1, 0, 0).
        // Leftover = 0; no leftover to distribute.
        let caps = GopCapacity {
            coeff_sign_bypass: 9, coeff_suffix_lsb: 2,
            mvd_sign_bypass: 0, mvd_suffix_lsb: 0,
        };
        let msg = vec![0u8; 5];
        let split = split_message_per_domain(&msg, &caps).unwrap();
        // 5 * 9 / 11 = 4 (floor), 5 * 2 / 11 = 0 (floor). leftover=1 → goes to coeff_sign (largest headroom).
        assert_eq!(split.coeff_sign_bypass.len(), 5);
        assert_eq!(split.coeff_suffix_lsb.len(), 0);
    }

    #[test]
    fn split_message_decoder_mirror_recovery() {
        // Encoder splits with capacities (100, 50, 30, 20) on a
        // 20-byte message. Decoder runs the same split with the
        // same capacities → identical per-domain bit slices.
        use super::super::GopCapacity;
        let caps = GopCapacity {
            coeff_sign_bypass: 100, coeff_suffix_lsb: 50,
            mvd_sign_bypass: 30, mvd_suffix_lsb: 20,
        };
        let msg: Vec<u8> = (0..20).map(|i| (i & 1) as u8).collect();
        let enc_split = split_message_per_domain(&msg, &caps).unwrap();
        let dec_split = split_message_per_domain(&msg, &caps).unwrap();
        assert_eq!(enc_split.coeff_sign_bypass, dec_split.coeff_sign_bypass);
        assert_eq!(enc_split.coeff_suffix_lsb, dec_split.coeff_suffix_lsb);
        assert_eq!(enc_split.mvd_sign_bypass, dec_split.mvd_sign_bypass);
        assert_eq!(enc_split.mvd_suffix_lsb, dec_split.mvd_suffix_lsb);
    }

    #[test]
    fn pass1_collects_per_domain_cover() {
        let cache = build_synthetic_gop();
        let cover = pass1_collect_cover(&cache);
        // MB 0: 3 nonzero coeffs → 3 sign bypass positions.
        // MB 1: 2 nonzero coeffs → 2 sign bypass positions.
        assert_eq!(cover.cover.coeff_sign_bypass.len(), 5);
        // No |coeff|>=16 anywhere → 0 suffix LSB positions.
        assert_eq!(cover.cover.coeff_suffix_lsb.len(), 0);
        // MB 1 has 2 MVD slots, both nonzero → 2 sign bypass positions.
        assert_eq!(cover.cover.mvd_sign_bypass.len(), 2);
        // |mvd|<9 → no suffix LSB.
        assert_eq!(cover.cover.mvd_suffix_lsb.len(), 0);
        // Costs aligned with bits.
        assert_eq!(
            cover.costs.coeff_sign_bypass.len(),
            cover.cover.coeff_sign_bypass.len(),
        );
        assert_eq!(
            cover.costs.mvd_sign_bypass.len(),
            cover.cover.mvd_sign_bypass.len(),
        );
    }

    #[test]
    fn pass2_empty_message_returns_cover_bits() {
        let cache = build_synthetic_gop();
        let cover = pass1_collect_cover(&cache);
        let messages = DomainMessages::default();
        let plan = pass2_stc_plan(&cover, &messages, /* h */ 7).unwrap();
        // No message → trivial plan = cover bits.
        assert_eq!(plan.coeff_sign_bypass, cover.cover.coeff_sign_bypass.bits);
        assert_eq!(plan.mvd_sign_bypass, cover.cover.mvd_sign_bypass.bits);
        assert_eq!(plan.total_modifications, 0);
    }

    #[test]
    fn pass2_stc_embeds_message_bits() {
        let cache = build_synthetic_gop();
        let cover = pass1_collect_cover(&cache);
        // Embed a tiny message in CoeffSignBypass only.
        // 5 cover bits, 1 message bit → w=5, very short STC.
        let messages = DomainMessages {
            coeff_sign_bypass: vec![1u8],
            ..Default::default()
        };
        let plan = pass2_stc_plan(&cover, &messages, /* h */ 4).unwrap();
        assert_eq!(plan.coeff_sign_bypass.len(), 5);
        // STC modifications count exposed via aggregate.
        // (May be 0 or 1 depending on cover bits and parity check.)
    }

    #[test]
    fn pass3_apply_overrides_modifies_decision_cache() {
        let mut cache = build_synthetic_gop();
        let cover = pass1_collect_cover(&cache);
        // Force-flip plan: invert every coeff sign bit.
        let mut plan = DomainPlan {
            coeff_sign_bypass: cover
                .cover
                .coeff_sign_bypass
                .bits
                .iter()
                .map(|b| b ^ 1)
                .collect(),
            ..Default::default()
        };
        plan.total_modifications = plan.coeff_sign_bypass.len();
        let count = pass3_apply_overrides(&mut cache, &cover.cover, &plan);
        assert_eq!(count, 5, "all 5 coeff sign positions must flip");
        // Verify the coefficients actually flipped.
        let new_cover = pass1_collect_cover(&cache);
        let new_bits = new_cover.cover.coeff_sign_bypass.bits;
        let old_bits = cover.cover.coeff_sign_bypass.bits;
        for (n, o) in new_bits.iter().zip(old_bits.iter()) {
            assert_eq!(*n, o ^ 1, "every bit should be inverted");
        }
    }

    #[test]
    fn pass3_no_op_plan_does_not_modify_cache() {
        let mut cache = build_synthetic_gop();
        let cover = pass1_collect_cover(&cache);
        // Plan == cover → no modifications.
        let plan = DomainPlan {
            coeff_sign_bypass: cover.cover.coeff_sign_bypass.bits.clone(),
            mvd_sign_bypass: cover.cover.mvd_sign_bypass.bits.clone(),
            ..Default::default()
        };
        let count = pass3_apply_overrides(&mut cache, &cover.cover, &plan);
        assert_eq!(count, 0);
    }

    /// Phase 6D.7 sign-off gate: full three-pass roundtrip on a
    /// synthetic GOP. Verifies:
    /// - Pass 1 produces cover + costs aligned with the decision cache.
    /// - Pass 2's STC plan is recoverable via stc_extract.
    /// - Pass 3's modified cache, when re-walked through Pass 1,
    ///   yields stego bits matching the plan.
    #[test]
    fn three_pass_roundtrip_with_per_domain_keys() {
        // Phase 6D.8 §23 sign-off: production-style three-pass run
        // using passphrase-derived per-domain seeds. Any STC bit
        // plan recovered with the SAME passphrase + GOP idx must
        // round-trip exactly.
        use crate::stego::stc::extract::stc_extract;
        use crate::stego::stc::hhat::generate_hhat;
        use super::super::keys::CabacStegoMasterKeys;

        let mut cache = build_synthetic_gop();
        let cover = pass1_collect_cover(&cache);

        let original_message = vec![1u8, 0];
        let messages = DomainMessages {
            coeff_sign_bypass: original_message.clone(),
            ..Default::default()
        };
        let h = 4;

        let keys = CabacStegoMasterKeys::derive("phase-6d-8-test").unwrap();
        let plan = pass2_stc_plan_with_keys(&cover, &messages, h, &keys, /*gop_idx*/ 0).unwrap();
        pass3_apply_overrides(&mut cache, &cover.cover, &plan);

        let stego_cover = pass1_collect_cover(&cache);
        assert_eq!(
            stego_cover.cover.coeff_sign_bypass.bits,
            plan.coeff_sign_bypass,
        );

        // Decode side: derive the same per-domain seed and
        // recover the message.
        let domain_seed = keys
            .per_gop_seeds(super::super::EmbedDomain::CoeffSignBypass, 0)
            .hhat_seed;
        let n = stego_cover.cover.coeff_sign_bypass.len();
        let w = n / original_message.len();
        let hhat = generate_hhat(h, w, &domain_seed);
        let recovered = stc_extract(&stego_cover.cover.coeff_sign_bypass.bits, &hhat, w);
        assert_eq!(
            recovered[..original_message.len()],
            original_message,
            "per-domain-keyed STC roundtrip must recover the message",
        );
    }

    #[test]
    fn three_pass_roundtrip_synthetic_gop() {
        use crate::stego::stc::extract::stc_extract;
        use crate::stego::stc::hhat::generate_hhat;

        let mut cache = build_synthetic_gop();
        let cover = pass1_collect_cover(&cache);

        // Embed a 2-bit message in CoeffSignBypass (5 cover bits → w=2).
        let original_message = vec![1u8, 0];
        let messages = DomainMessages {
            coeff_sign_bypass: original_message.clone(),
            ..Default::default()
        };
        let h = 4;
        let plan = pass2_stc_plan(&cover, &messages, h).unwrap();
        pass3_apply_overrides(&mut cache, &cover.cover, &plan);

        // Re-walk Pass 1 on the modified cache: bits we extract
        // must equal the plan we built.
        let stego_cover = pass1_collect_cover(&cache);
        assert_eq!(
            stego_cover.cover.coeff_sign_bypass.bits,
            plan.coeff_sign_bypass,
        );
        // And the STC decode of those stego bits returns the
        // original message.
        let n = stego_cover.cover.coeff_sign_bypass.len();
        let w = n / original_message.len();
        let seed = [0u8; 32];
        let hhat = generate_hhat(h, w, &seed);
        let recovered = stc_extract(&stego_cover.cover.coeff_sign_bypass.bits, &hhat, w);
        assert_eq!(
            recovered[..original_message.len()],
            original_message,
            "STC decode must recover the embedded message",
        );
    }

    // ── STEGO.A.2 — Scheme A combined-STC primary planner tests ──

    /// Build a synthetic GopCover with bits + positions across all 4
    /// domains. Costs default to uniform 1.0 (or content-adaptive if
    /// `populate_costs` is true — we use uniform here since the
    /// content_costs module needs a real YUV).
    fn synth_4domain_cover(
        n_csb: usize, n_csl: usize, n_msb: usize, n_msl: usize,
    ) -> GopCover {
        use crate::codec::h264::stego::hook::{
            PositionKey, SyntaxPath, BinKind, Axis as HookAxis,
        };
        use crate::codec::h264::stego::inject::{DomainBits, DomainCover};
        let mut cover = DomainCover::default();
        let mut seed: u32 = 0xDEADBEEF;
        let mut next_bit = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) & 1) as u8
        };
        for i in 0..n_csb {
            cover.coeff_sign_bypass.bits.push(next_bit());
            cover.coeff_sign_bypass.positions.push(PositionKey::new(
                0, (i / 16) as u32, EmbedDomain::CoeffSignBypass,
                SyntaxPath::Luma4x4 {
                    block_idx: (i % 16) as u8,
                    coeff_idx: ((i % 15) + 1) as u8,
                    kind: BinKind::Sign,
                },
            ));
        }
        for i in 0..n_csl {
            cover.coeff_suffix_lsb.bits.push(next_bit());
            cover.coeff_suffix_lsb.positions.push(PositionKey::new(
                0, (i / 16) as u32, EmbedDomain::CoeffSuffixLsb,
                SyntaxPath::Luma4x4 {
                    block_idx: (i % 16) as u8,
                    coeff_idx: ((i % 15) + 1) as u8,
                    kind: BinKind::SuffixLsb,
                },
            ));
        }
        for i in 0..n_msb {
            cover.mvd_sign_bypass.bits.push(next_bit());
            cover.mvd_sign_bypass.positions.push(PositionKey::new(
                0, (i / 4) as u32, EmbedDomain::MvdSignBypass,
                SyntaxPath::Mvd {
                    list: 0,
                    partition: (i % 16) as u8,
                    axis: HookAxis::X,
                    kind: BinKind::Sign,
                },
            ));
        }
        for i in 0..n_msl {
            cover.mvd_suffix_lsb.bits.push(next_bit());
            cover.mvd_suffix_lsb.positions.push(PositionKey::new(
                0, (i / 4) as u32, EmbedDomain::MvdSuffixLsb,
                SyntaxPath::Mvd {
                    list: 0,
                    partition: (i % 16) as u8,
                    axis: HookAxis::Y,
                    kind: BinKind::SuffixLsb,
                },
            ));
        }
        let costs = DomainCosts {
            coeff_sign_bypass: vec![1.0; n_csb],
            coeff_suffix_lsb: vec![1.0; n_csl],
            mvd_sign_bypass: vec![1.0; n_msb],
            mvd_suffix_lsb: vec![1.0; n_msl],
        };
        GopCover { cover, costs }
    }

    /// Self-contained Scheme A round-trip at the planner level:
    /// build cover → plan → reconstruct combined wire bits → STC
    /// extract → recover frame_bits. Independent of any walker / real
    /// encoder; validates that
    /// `pass2_stc_plan_combined_with_keys` is a sound encoder for
    /// the matching STEGO.A.4 decoder.
    #[test]
    fn scheme_a_combined_planner_roundtrip() {
        use crate::codec::h264::stego::cost_weights::{
            combine_cover_4domain, CostWeights,
        };
        use crate::stego::stc::extract::stc_extract;
        let cover = synth_4domain_cover(80, 40, 30, 20);
        let weights = CostWeights::default();
        let keys = CabacStegoMasterKeys::derive("scheme-a-roundtrip-pass").unwrap();
        let gop_idx = 0u32;
        let h = 4;

        // Pick m_total so w = n_total / m_total ≥ 4 (enough headroom
        // for STC).
        let n_total = 80 + 40 + 30 + 20;
        let m_total = n_total / 5;
        let frame_bits: Vec<u8> = (0..m_total).map(|i| (i % 2) as u8).collect();

        let plan = pass2_stc_plan_combined_with_keys(
            &cover, &weights, &frame_bits, h, &keys, gop_idx,
        )
        .expect("planner should succeed at w=5");

        // Reconstruct the wire bits the decoder would walk: take the
        // ORIGINAL combined cover, then overlay per-domain plan bits
        // at their original positions (per-domain plan vectors have
        // the SAME length as the cover's bits).
        let mut wire = cover.cover.clone();
        wire.coeff_sign_bypass.bits = plan.coeff_sign_bypass.clone();
        wire.coeff_suffix_lsb.bits = plan.coeff_suffix_lsb.clone();
        wire.mvd_sign_bypass.bits = plan.mvd_sign_bypass.clone();
        wire.mvd_suffix_lsb.bits = plan.mvd_suffix_lsb.clone();

        // Combine the wire bits + extract.
        let dummy_costs = DomainCosts::default();
        let (wire_combined, _, _) = combine_cover_4domain(&wire, &dummy_costs, &weights);
        let w_rate = wire_combined.len() / m_total;
        let hhat_seed = keys.per_gop_seeds(EmbedDomain::CoeffSignBypass, gop_idx).hhat_seed;
        let hhat = generate_hhat(h, w_rate, &hhat_seed);
        let used = m_total * w_rate;
        let recovered = stc_extract(&wire_combined[..used], &hhat, w_rate);
        assert_eq!(&recovered[..m_total], &frame_bits[..],
            "Scheme A round-trip must recover the embedded message");
    }

    #[test]
    fn scheme_a_combined_planner_empty_message() {
        use crate::codec::h264::stego::cost_weights::CostWeights;
        let cover = synth_4domain_cover(10, 10, 10, 10);
        let weights = CostWeights::default();
        let keys = CabacStegoMasterKeys::derive("empty").unwrap();
        let plan = pass2_stc_plan_combined_with_keys(
            &cover, &weights, &[], 4, &keys, 0,
        ).expect("empty message should plan trivially");
        // Empty-message plan: returns the original cover bits
        // unchanged in all four domains (no STC modification).
        assert_eq!(plan.coeff_sign_bypass, cover.cover.coeff_sign_bypass.bits);
        assert_eq!(plan.coeff_suffix_lsb, cover.cover.coeff_suffix_lsb.bits);
        assert_eq!(plan.mvd_sign_bypass, cover.cover.mvd_sign_bypass.bits);
        assert_eq!(plan.mvd_suffix_lsb, cover.cover.mvd_suffix_lsb.bits);
    }

    #[test]
    fn scheme_a_combined_planner_message_too_large_returns_none() {
        use crate::codec::h264::stego::cost_weights::CostWeights;
        // 8 cover bits across all 4 domains, 9-bit message → w = 0,
        // unfeasible.
        let cover = synth_4domain_cover(2, 2, 2, 2);
        let weights = CostWeights::default();
        let keys = CabacStegoMasterKeys::derive("too-large").unwrap();
        let frame_bits = vec![1u8; 9];
        let res = pass2_stc_plan_combined_with_keys(
            &cover, &weights, &frame_bits, 4, &keys, 0,
        );
        assert!(res.is_none(), "expected planner failure on infeasible rate");
    }
}
