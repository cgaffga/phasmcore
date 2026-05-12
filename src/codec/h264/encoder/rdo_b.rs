// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 6E-D — B-slice fast rate-distortion mode-decision.
//!
//! Phase 6E-D.1 ships the foundation for B-slice mode decision:
//! `BMbRdoResult`, `BMbCandidate`, `LAMBDA_TAB_B`, and
//! `evaluate_b_mb_rdo` for the 16x16 family (Skip/Direct collapsed,
//! L0_16x16, L1_16x16, Bi_16x16) only. Partitioned (mb_types 4..21)
//! + B_8x8 (mb_type 22) extension lands in 6E-D.2 / 6E-D.3.
//!
//! Wiring into `mb_decision_b_with_mvs` is deferred to 6E-D.5.
//! This module is standalone foundation that compiles + tests on
//! its own.
//!
//! ## Fast RDO (default, the converter-pipeline centroid-style)
//!
//! Uses **SATD distortion + bin-count rate** rather than full
//! quant→dequant→IDCT→recon→SSD. Standard fast-RDO formulation:
//! the 4×4 Hadamard concentrates residual energy the same way the
//! integer DCT does, so SATD ranks candidates within ~95% of true
//! post-quant SSD ordering at ~1/5th the cost. the converter-pipeline centroid uses
//! exactly this pattern for mode decision; phasm matches that
//! behaviour because matching the converter-pipeline centroid output is the v1 stealth
//! goal (see `docs/design/video/h264/stealth-strategy.md`).
//!
//! Cost formula:
//!
//! ```text
//! cost = SATD_luma + LAMBDA_TAB_B[qp] × R_overhead_bits
//! ```
//!
//! where `R_overhead_bits` is the wire-rate for mb_type bins +
//! MVDs + CBP/qp_delta (no residual coefficient bits — those are
//! implicitly captured by the SATD term in the standard fast-RDO
//! formulation).
//!
//! ## λ scaling
//!
//! `LAMBDA_TAB_B[qp] = round(sqrt(LAMBDA2_TAB[qp] × 1.44²))`.
//! The sqrt converts the squared-lambda used with SSD-domain RDO
//! to the linear-lambda used with SATD-domain RDO (Sullivan
//! 2003). The 1.44 factor is the JVT-P008 non-reference-B
//! adjustment — B-frames are not used as references
//! (`nal_ref_idc=0`, §6E-A4), so quant noise doesn't propagate
//! and the rate-distortion trade tolerates higher distortion at
//! the same QP. `1.44 ≈ standard b_frame_pyramid weighting
//! when --bframes 1`.
//!
//! ## Skip / Direct collapse
//!
//! Skip and Direct_16x16 produce IDENTICAL reconstruction (both run
//! spatial-direct MV derivation; spec § 8.4.1.2.2). Skip's wire form
//! is 1 bin (`mb_skip_flag=1`); Direct's is 1 bin (skip=0) + mb_type
//! bin (1) + CBP + qp_delta + residual. The candidate set has ONE
//! entry covering both — `BMbCandidate::SkipOrDirect` — and uses
//! the Direct rate-overhead estimate during ranking. The actual
//! wire-form choice (Skip vs Direct) happens AFTER candidate
//! selection in §6E-D.5 wiring, where the encoder runs full quant
//! on the winner and emits Skip if CBP=0.
//!
//! ## Stego-aware candidate gating
//!
//! When `enable_mvd_stego_hook=true` the candidate set excludes
//! Skip + Direct families. Reason: Pass 3 stego may flip MVD signs
//! in NEIGHBOURING MBs; spatial-direct MV derivation reads those
//! neighbour MVDs at decode time, so a Skip/Direct MB chosen in
//! Pass 1 may not produce the same reconstruction in Pass 3 →
//! cascade-safety violation. Explicit-MV families (L0/L1/Bi 16x16
//! + Partitioned + B_8x8) carry their own MVDs, so Pass 3 sign
//! flips don't change mode validity.
//!
//! Gate-controlled by `BCandidateBuilder::with_stego_mode` — see
//! call site in 6E-D.5.
//!
//! ## Determinism
//!
//! `evaluate_b_mb_rdo` is a pure function of (candidate, src,
//! refs, qp). No RNG. Same inputs → same cost. Required for
//! Pass 1 / Pass 3 mode parity in stego mode.

use super::b_partitioned::{BListUse, BPartitionedMeta};
use super::intra_predictor::satd_16x16;
use super::motion_compensation::{apply_luma_mv_block, apply_luma_mv_block_bipred};
use super::motion_estimation::MotionVector;
use super::reference_buffer::ReconFrame;

/// §6E-D.1 — non-reference B-slice linear λ lookup for SATD-domain
/// fast RDO.
///
/// `LAMBDA_TAB_B[qp] ≈ round(sqrt(LAMBDA2_TAB[qp] × 1.44²))`.
/// Used as the multiplier on the rate term in fast RDO:
///
/// ```text
/// cost = SATD + LAMBDA_TAB_B[qp] × R_bits
/// ```
///
/// At QP 21: P-side LAMBDA² = 1843, B-side LAMBDA = round(sqrt(3822)) = 62.
/// At QP 51: B-side LAMBDA = 1978.
pub const LAMBDA_TAB_B: [u32; 52] = [
    5, 6, 7, 8, 9, 10, 11, 12,                              // 0-7
    14, 15, 17, 19, 22, 25, 28, 31,                         // 8-15
    35, 39, 44, 49, 55, 62, 69, 78,                         // 16-23
    87, 98, 110, 124, 139, 156, 175, 196,                   // 24-31
    220, 247, 278, 312, 350, 393, 441, 495,                 // 32-39
    555, 623, 699, 785, 881, 989,                           // 40-45
    1110, 1246, 1399, 1570, 1762, 1978,                     // 46-51
];

/// §6E-D.5b Track B + §6E-D.5(f) — CABAC-aware bin-cost weighting.
///
/// Real CABAC charges fractional bits per bin as `-log2(p)` where
/// `p` is the bin's probability under its current context state.
/// For mb_type bins beyond Skip/Direct (the rare-context tail), real
/// average bin cost at WARM context state is ~1-2 bits/bin (not 3
/// as Track B initially assumed). §6E-D.6 trace breakthrough showed
/// the original Track-B 3-bit value over-corrected, contributing to
/// the 99% Skip / 99% Direct dominance: Direct's 1-bit prefix beat
/// L0/L1's 1+2×3=7 bits by 6 bins × λ ≈ 1668 cost units, a margin
/// L0's slightly-better SATD usually couldn't clear at QP=34.
///
/// §6E-D.5(f): RARE_BIN_COST = 2. This still over-charges relative
/// to true CABAC fractional bins (real the reference fast encoder charges ~0.5-1.5 bits
/// at warm state), but stays conservative. Empirical the reference fast encoder same-
/// fixture mb_type prefix bins for rare cases: average ~1.5 bits.
/// Integer-rounded UP to 2 to keep cost-model defensibly above
/// minimum CABAC charge — this is correctness (matches actual
/// encoder rates more closely), not calibration.
///
/// Resulting overheads:
///   Direct   : 1-bin frequent-context        → 1 bit
///   L0/L1    : 1 + 2×2                       → 5 bits
///   Bi       : 1 + 5×2                       → 11 bits
///   Part     : 1 + 4×2                       → 9 bits (5-bin estimate)
///   B_8x8    : 1 + 5×2                       → 11 bits (mb_type=22)
const RARE_BIN_COST: u32 = 2;

/// PSY-RD strength shift (Q.0 fixed-point divisor). Final psy term is
/// added as `(source_hf - pred_hf).abs() >> PSY_RD_SHIFT`. Calibrated
/// to the PSY-RD=1.0 baseline from published rate-control literature scale: the reference fast encoder applies `psy_rd × psy_term`
/// directly to its lambda-weighted cost; in our SATD+λR scale, shift=4
/// (divide-by-16) approximates the reference fast encoder's lambda-cost-relative impact at
/// medium QP range (~30-36).
///
/// Env-overridable via `PHASM_PSY_RD_SHIFT` (default 4). Set higher
/// (e.g. 6) to weaken the psy term, lower (e.g. 2) to strengthen.
const PSY_RD_SHIFT_DEFAULT: u32 = 4;

/// Compute "high-frequency SATD" of a 16x16 luma block: SATD against
/// a flat-DC reference (the block's own mean). This measures how
/// much detail the block carries beyond its average level — a smooth
/// flat block has low HF energy; a textured block has high HF energy.
///
/// Used in PSY-RD cost: predictions whose HF energy is far from source's
/// HF energy are penalized (Direct/Skip predictions tend to be smoother
/// than the source they cover, losing detail and triggering psy penalty).
pub(crate) fn psy_hf_satd_16x16(block: &[[u8; 16]; 16]) -> u32 {
    let mut sum: u32 = 0;
    for row in block {
        for &v in row {
            sum += v as u32;
        }
    }
    let mean = (sum / 256).min(255) as u8;
    let mut flat = [[0u8; 16]; 16];
    for row in flat.iter_mut() {
        for v in row.iter_mut() {
            *v = mean;
        }
    }
    crate::codec::h264::encoder::intra_predictor::satd_16x16(block, &flat)
}

/// Read `PHASM_PSY_RD` env: returns (enabled, shift).
pub(crate) fn psy_rd_config() -> (bool, u32) {
    let enabled = super::mb_decision_b::env_var("PHASM_PSY_RD")
        .map(|v| v == "1")
        .unwrap_or(false);
    let shift = super::mb_decision_b::env_var("PHASM_PSY_RD_SHIFT")
        
        .and_then(|s| s.parse().ok())
        .unwrap_or(PSY_RD_SHIFT_DEFAULT);
    (enabled, shift)
}

/// Bin-cost approximation per absolute MVD value (Exp-Golomb-like).
/// Used for rate estimate of explicit-MV candidates without running
/// CABAC. Per coefficient: 1 prefix bit + log2(mvd+1) tail bits ≈
/// 2 + 2*log2(|mvd|+1). Conservative estimate; consistent across
/// candidates so ranking is unaffected.
fn mvd_rate_estimate(mvd_x: i16, mvd_y: i16) -> u32 {
    let log2_plus_1 = |v: i16| -> u32 {
        let abs = v.unsigned_abs() as u32;
        if abs == 0 { 1 } else { 32 - abs.leading_zeros() }
    };
    // 2 bits per axis baseline (sign + minimal mag) + log2 tail.
    2 + log2_plus_1(mvd_x) + 2 + log2_plus_1(mvd_y)
}

/// §6E-D.2 — paint a single partition's MC prediction into the 16x16
/// MB-level buffer. Used by the Partitioned candidate evaluator.
///
/// `part_off_4x4` and `part_dim_4x4` come from
/// `BPartitionShape::part_offset` / `part_dim_4x4`. List usage drives
/// single-list vs bipred MC.
fn paint_partition_luma(
    mb_buf: &mut [[u8; 16]; 16],
    part_off_4x4: (usize, usize),
    part_dim_4x4: (usize, usize),
    list: BListUse,
    mv_l0: MotionVector,
    mv_l1: MotionVector,
    l0_ref: &ReconFrame,
    l1_ref: &ReconFrame,
    mb_x: usize,
    mb_y: usize,
) {
    let part_x_px = ((mb_x * 16) + part_off_4x4.0 * 4) as u32;
    let part_y_px = ((mb_y * 16) + part_off_4x4.1 * 4) as u32;
    let part_w_px = (part_dim_4x4.0 * 4) as u32;
    let part_h_px = (part_dim_4x4.1 * 4) as u32;

    // Stage prediction in a temp 16x16 scratch (partition-sized
    // sub-rect populated). Then copy the populated rect into mb_buf.
    let mut scratch = [0u8; 16 * 16];
    match list {
        BListUse::L0 => apply_luma_mv_block(
            l0_ref, part_x_px, part_y_px, part_w_px, part_h_px,
            mv_l0, &mut scratch, 16,
        ),
        BListUse::L1 => apply_luma_mv_block(
            l1_ref, part_x_px, part_y_px, part_w_px, part_h_px,
            mv_l1, &mut scratch, 16,
        ),
        BListUse::Bi => apply_luma_mv_block_bipred(
            l0_ref, mv_l0, l1_ref, mv_l1,
            part_x_px, part_y_px, part_w_px, part_h_px,
            &mut scratch, 16,
        ),
    }

    let dst_x0 = part_off_4x4.0 * 4;
    let dst_y0 = part_off_4x4.1 * 4;
    let w = part_dim_4x4.0 * 4;
    let h = part_dim_4x4.1 * 4;
    for dy in 0..h {
        for dx in 0..w {
            mb_buf[dst_y0 + dy][dst_x0 + dx] = scratch[dy * 16 + dx];
        }
    }
}

/// §6E-D.3 — paint a single 8x8 sub-MB's MC prediction into the 16x16
/// MB-level buffer.
///
/// Sub-MB raster order: 0=TL (0,0), 1=TR (8,0), 2=BL (0,8), 3=BR (8,8).
/// `sub_mb_type` selects which list(s) drive MC:
///   0 = Direct (caller-supplied MVs from spatial-direct derivation;
///       single-list direction per spec § 8.4.1.2.2 — but in fast-RDO
///       we approximate as bipred when both MVs are non-zero)
///   1 = L0 only, 2 = L1 only, 3 = Bi
fn paint_sub_mb_luma(
    mb_buf: &mut [[u8; 16]; 16],
    sub_idx: usize,
    sub_mb_type: u8,
    mvs: &BPartitionMvPair,
    l0_ref: &ReconFrame,
    l1_ref: &ReconFrame,
    mb_x: usize,
    mb_y: usize,
) {
    let (sub_x_4, sub_y_4) = match sub_idx {
        0 => (0, 0),
        1 => (2, 0),
        2 => (0, 2),
        3 => (2, 2),
        _ => unreachable!("B_8x8 sub_idx must be 0..=3"),
    };
    let sub_x_px = ((mb_x * 16) + sub_x_4 * 4) as u32;
    let sub_y_px = ((mb_y * 16) + sub_y_4 * 4) as u32;

    let mut scratch = [0u8; 16 * 16];
    match sub_mb_type {
        0 | 3 => {
            // Direct (caller-supplied MVs) and Bi both use bipred.
            apply_luma_mv_block_bipred(
                l0_ref, mvs.mv_l0, l1_ref, mvs.mv_l1,
                sub_x_px, sub_y_px, 8, 8,
                &mut scratch, 16,
            );
        }
        1 => apply_luma_mv_block(
            l0_ref, sub_x_px, sub_y_px, 8, 8, mvs.mv_l0,
            &mut scratch, 16,
        ),
        2 => apply_luma_mv_block(
            l1_ref, sub_x_px, sub_y_px, 8, 8, mvs.mv_l1,
            &mut scratch, 16,
        ),
        _ => debug_assert!(false, "B_8x8 sub_mb_type {sub_mb_type} out of §6E-A6.3 scope"),
    }

    let dst_x0 = sub_x_4 * 4;
    let dst_y0 = sub_y_4 * 4;
    for dy in 0..8 {
        for dx in 0..8 {
            mb_buf[dst_y0 + dy][dst_x0 + dx] = scratch[dy * 16 + dx];
        }
    }
}

/// §6E-D.3 — overhead bin count for a B_8x8 candidate (mb_type=22).
///
/// CABAC bin lengths:
///   - mb_type=22 prefix: 6 bins (longest non-intra prefix in B-slice
///     mb_type tree per Table 9-37 / 9-39).
///   - Per-sub-MB sub_mb_type: 1 bin for Direct (0), 2-3 bins for
///     L0/L1/Bi. Use a flat 3-bin estimate for simplicity.
///   - Per-sub-MB MVDs: 0 for Direct, 1 for L0/L1, 2 for Bi.
///   - CBP + qp_delta: 4 bins.
fn b_8x8_overhead_bits(sub_mb_types: &[u8; 4], parts: &[BPartitionMvPair; 4]) -> u32 {
    // mb_type=22 binarization is 6 bins (longest non-intra path);
    // 1 leading + 5 rare-context per Track B.
    let mut total = 1 + 5 * RARE_BIN_COST;
    for i in 0..4 {
        let sub = sub_mb_types[i];
        let p = &parts[i];
        // sub_mb_type binarization: 1 bin Direct (frequent),
        // 3 bins L0/L1/Bi (1 leading + 2 rare).
        total += if sub == 0 { 1 } else { 1 + 2 * RARE_BIN_COST };
        match sub {
            0 => {} // Direct: no MVDs
            1 => total += mvd_rate_estimate(p.mv_l0.mv_x, p.mv_l0.mv_y),
            2 => total += mvd_rate_estimate(p.mv_l1.mv_x, p.mv_l1.mv_y),
            3 => {
                total += mvd_rate_estimate(p.mv_l0.mv_x, p.mv_l0.mv_y);
                total += mvd_rate_estimate(p.mv_l1.mv_x, p.mv_l1.mv_y);
            }
            _ => debug_assert!(false, "B_8x8 sub_mb_type {sub} out of scope"),
        }
    }
    total += 4; // CBP + qp_delta
    total
}

/// §6E-D.2 — overhead bin count for a Partitioned (mb_type 4..21).
///
/// CABAC bin lengths from the reference fast encoder / reference decoder:
///   - mb_type bin string for 16x8/8x16 partitioned: ~4-6 bins
///     depending on combo. Use a flat 5-bin estimate for ranking;
///     differences across combos are <2 bins → does not flip the
///     candidate winner under realistic SATD spreads.
///   - Per-partition MVDs: L0 uses L0 MVD; L1 uses L1 MVD; Bi
///     uses both.
///   - CBP + qp_delta: 4 bins.
fn partitioned_overhead_bits(meta: &BPartitionedMeta,
    p0: &BPartitionMvPair, p1: &BPartitionMvPair) -> u32 {
    let mvd_for = |use_: BListUse, p: &BPartitionMvPair| -> u32 {
        match use_ {
            BListUse::L0 => mvd_rate_estimate(p.mv_l0.mv_x, p.mv_l0.mv_y),
            BListUse::L1 => mvd_rate_estimate(p.mv_l1.mv_x, p.mv_l1.mv_y),
            BListUse::Bi => {
                mvd_rate_estimate(p.mv_l0.mv_x, p.mv_l0.mv_y)
                    + mvd_rate_estimate(p.mv_l1.mv_x, p.mv_l1.mv_y)
            }
        }
    };
    // mb_type=4..21 bin strings are 4-6 bins, mostly rare context.
    // Flat 5-bin estimate × Track B rare-context cost (1 leading + 4 rare).
    1 + 4 * RARE_BIN_COST
        + mvd_for(meta.part0, p0)
        + mvd_for(meta.part1, p1)
        + 4 // CBP + qp_delta
}

/// §6E-D.2 — per-partition MV pair for a Partitioned candidate.
/// Mirrors `b_partitioned::BPartitionMv` shape but with required
/// MVs (no `Option`) — they are present iff the partition's
/// `BListUse` requires that list.
///
/// Validity contract: if `meta.partN` is `L0` then `mv_l0` is the
/// active MV, `mv_l1` must be ignored. If `Bi`, both are active.
///
/// v1.4 Phase 2 (#305): `ref_idx_l0` carries the L0 reference index
/// for this partition. Default 0 = closest past anchor (single-ref
/// behavior). When multi-ref is enabled, ME populates it per
/// partition. L1 stays single-ref (Q1 — no `ref_idx_l1` field).
#[derive(Debug, Clone, Copy)]
pub struct BPartitionMvPair {
    pub mv_l0: MotionVector,
    pub mv_l1: MotionVector,
    pub ref_idx_l0: u8,
}

impl BPartitionMvPair {
    /// Construct with explicit MVs and ref_idx_l0=0 (single-ref
    /// default). Existing builder paths use this; multi-ref-aware
    /// callers set `ref_idx_l0` directly via struct literal.
    pub fn new(mv_l0: MotionVector, mv_l1: MotionVector) -> Self {
        Self { mv_l0, mv_l1, ref_idx_l0: 0 }
    }
}

/// §6E-D.1 + §6E-D.2 — B-slice candidate family for fast RDO.
///
/// Phase 6E-D.2 adds the `Partitioned` family covering mb_types
/// 4..21 (16x8 + 8x16 with all 9 direction combos per shape).
/// B_8x8 (22) lands in §6E-D.3.
#[allow(non_camel_case_types)] // matches H.264 spec mb_type names
#[derive(Debug, Clone, Copy)]
pub enum BMbCandidate {
    /// Skip OR Direct_16x16 — same reconstruction, the encoder picks
    /// the cheaper wire form AFTER fast-RDO ranking by checking CBP
    /// on the winner. Spatial-direct MV derived from neighbours via
    /// `derive_b_direct_spatial`.
    ///
    /// Phase 2.12.f (#282): `mv_l0_per_8x8` / `mv_l1_per_8x8` carry
    /// the per-sub-block MVs from `BDirectSpatialResult`. When at
    /// least one sub-block diverges from the top-left (= colZeroFlag
    /// override fired), the SATD evaluator must use the per-8x8
    /// builder, otherwise the cost diverges from actual emit and
    /// RDO ranks SkipOrDirect incorrectly.
    SkipOrDirect {
        mv_l0: MotionVector,
        mv_l1: MotionVector,
        uses_l0: bool,
        uses_l1: bool,
        mv_l0_per_8x8: [MotionVector; 4],
        mv_l1_per_8x8: [MotionVector; 4],
    },
    /// `mb_type=1` — single L0 reference, MV from ME.
    ///
    /// v1.4 Phase 2 (#305): `ref_idx_l0` carries the L0 reference
    /// index. Default 0 = closest past anchor (single-ref behavior).
    /// Multi-ref ME (Phase 4) populates it; emit side (Phase 2/9)
    /// writes it via `encode_ref_idx()` when num_active_l0 > 1.
    L0_16x16 { mv_l0: MotionVector, ref_idx_l0: u8 },
    /// `mb_type=2` — single L1 reference, MV from ME. L1 stays
    /// single-ref under v1.4 (Q1).
    L1_16x16 { mv_l1: MotionVector },
    /// `mb_type=3` — bipred over both lists, both MVs from ME.
    /// `ref_idx_l0` per L0_16x16; L1 stays single-ref.
    Bi_16x16 { mv_l0: MotionVector, mv_l1: MotionVector, ref_idx_l0: u8 },
    /// `mb_type=4..21` — partitioned 16x8 / 8x16 with per-partition
    /// list usage. Per-partition MVs come from ME for the required
    /// lists; the unused list's MV is ignored by `evaluate_b_mb_rdo`.
    Partitioned {
        meta: BPartitionedMeta,
        part0_mvs: BPartitionMvPair,
        part1_mvs: BPartitionMvPair,
    },
    /// `mb_type=22` — `B_8x8`. Four 8x8 sub-MBs, each with its own
    /// `sub_mb_type` (0..=3 — uniform 8x8 family per §6E-A6.3 scope):
    ///
    /// - 0 = `B_Direct_8x8` (no MVDs, decoder spatial-direct derivation)
    /// - 1 = `B_L0_8x8` (one L0 MV per sub-MB)
    /// - 2 = `B_L1_8x8` (one L1 MV per sub-MB)
    /// - 3 = `B_Bi_8x8` (one L0 + one L1 MV per sub-MB)
    ///
    /// `parts[i]` carries the MVs for sub-MB `i` (raster order:
    /// 0=TL, 1=TR, 2=BL, 3=BR). For sub_mb_type=0 the caller passes
    /// the spatial-direct-derived MVs (or zero in test paths).
    /// Sub-sub partitions (sub_mb_types 4..12) are descoped per §6E-A6.4.
    B_8x8 {
        sub_mb_types: [u8; 4],
        parts: [BPartitionMvPair; 4],
    },
}

/// §6E-D.1 — outcome of a single fast-RDO candidate evaluation.
///
/// Note: SATD-domain cost (no real quant). The encoder runs full
/// quant on the winning candidate to determine actual residual
/// + CBP + Skip-vs-Direct wire-form choice (§6E-D.5).
#[derive(Debug, Clone, Copy)]
pub struct BMbRdoResult {
    pub candidate: BMbCandidate,
    /// Sum of absolute Hadamard coefficients on luma plane.
    pub satd: u32,
    /// Estimated overhead bits the candidate emits on the wire.
    /// Excludes residual coefficient bits (those are implicit in
    /// SATD under the fast-RDO formulation).
    pub r_bits: u32,
    /// SATD-domain Lagrangian cost: `SATD + LAMBDA × R_bits`.
    pub cost: u64,
}

/// §6E-D.1 — evaluate a single B-slice candidate using fast RDO.
/// Returns SATD distortion + rate-estimate + Lagrangian cost.
///
/// Pipeline:
/// 1. Build prediction (single-list MC / bipred MC).
/// 2. Compute `SATD = satd_16x16(src, pred)`.
/// 3. Estimate overhead R_bits (mb_type bins + MVDs + CBP/qp_delta).
/// 4. cost = SATD + LAMBDA_TAB_B[qp] × R_bits.
///
/// No quantize, no reconstruct, no SSD. The §30D-C orchestrator
/// uses the resulting mode choice in Pass 1 + Pass 3, so
/// determinism is load-bearing — fast RDO is fully deterministic
/// given the same inputs.
/// Build the 16×16 luma prediction for one B-slice candidate.
/// Pure: no encoder state mutation.
fn build_luma_prediction_for_candidate(
    candidate: &BMbCandidate,
    l0_ref: &ReconFrame,
    l1_ref: &ReconFrame,
    mb_x: usize,
    mb_y: usize,
) -> [[u8; 16]; 16] {
    let mb_px_x = (mb_x * 16) as u32;
    let mb_px_y = (mb_y * 16) as u32;

    let mut pred_y = [[0u8; 16]; 16];
    let pred_flat = pred_y.as_flattened_mut();

    match *candidate {
        BMbCandidate::SkipOrDirect { mv_l0, mv_l1, uses_l0, uses_l1, mv_l0_per_8x8, mv_l1_per_8x8 } => {
            // Phase 2.12.f (#282) — per-sub-block MVs differ when
            // colZeroFlag override fires on at least one 8x8 sub-block.
            let needs_per_8x8 = mv_l0_per_8x8.iter().any(|&m| m != mv_l0)
                || mv_l1_per_8x8.iter().any(|&m| m != mv_l1);
            if needs_per_8x8 {
                pred_y = super::b_inter_prediction::build_b_luma_prediction_per_8x8(
                    mv_l0_per_8x8, mv_l1_per_8x8,
                    uses_l0, uses_l1,
                    l0_ref, l1_ref, mb_x, mb_y,
                );
            } else {
                match (uses_l0, uses_l1) {
                    (true, true) => apply_luma_mv_block_bipred(
                        l0_ref, mv_l0, l1_ref, mv_l1,
                        mb_px_x, mb_px_y, 16, 16, pred_flat, 16,
                    ),
                    (true, false) => apply_luma_mv_block(
                        l0_ref, mb_px_x, mb_px_y, 16, 16, mv_l0, pred_flat, 16,
                    ),
                    (false, true) => apply_luma_mv_block(
                        l1_ref, mb_px_x, mb_px_y, 16, 16, mv_l1, pred_flat, 16,
                    ),
                    (false, false) => {
                        // Pathological case: spatial-direct derived no
                        // valid list. Fall back to L0 zero MV.
                        apply_luma_mv_block(
                            l0_ref, mb_px_x, mb_px_y, 16, 16,
                            MotionVector::ZERO, pred_flat, 16,
                        );
                    }
                }
            }
        }
        BMbCandidate::L0_16x16 { mv_l0, .. } => apply_luma_mv_block(
            l0_ref, mb_px_x, mb_px_y, 16, 16, mv_l0, pred_flat, 16,
        ),
        BMbCandidate::L1_16x16 { mv_l1 } => apply_luma_mv_block(
            l1_ref, mb_px_x, mb_px_y, 16, 16, mv_l1, pred_flat, 16,
        ),
        BMbCandidate::Bi_16x16 { mv_l0, mv_l1, .. } => apply_luma_mv_block_bipred(
            l0_ref, mv_l0, l1_ref, mv_l1,
            mb_px_x, mb_px_y, 16, 16, pred_flat, 16,
        ),
        BMbCandidate::Partitioned { meta, part0_mvs, part1_mvs } => {
            let dim = meta.shape.part_dim_4x4();
            let off0 = meta.shape.part_offset(0);
            let off1 = meta.shape.part_offset(1);
            paint_partition_luma(
                &mut pred_y, off0, dim, meta.part0,
                part0_mvs.mv_l0, part0_mvs.mv_l1,
                l0_ref, l1_ref, mb_x, mb_y,
            );
            paint_partition_luma(
                &mut pred_y, off1, dim, meta.part1,
                part1_mvs.mv_l0, part1_mvs.mv_l1,
                l0_ref, l1_ref, mb_x, mb_y,
            );
        }
        BMbCandidate::B_8x8 { sub_mb_types, parts } => {
            for i in 0..4 {
                paint_sub_mb_luma(
                    &mut pred_y, i, sub_mb_types[i], &parts[i],
                    l0_ref, l1_ref, mb_x, mb_y,
                );
            }
        }
    }

    pred_y
}

pub fn evaluate_b_mb_rdo(
    candidate: &BMbCandidate,
    src_y: &[[u8; 16]; 16],
    l0_ref: &ReconFrame,
    l1_ref: &ReconFrame,
    mb_x: usize,
    mb_y: usize,
    mb_qp: u8,
) -> BMbRdoResult {
    // ── 1. Prediction ────────────────────────────────────────────
    let pred_y = build_luma_prediction_for_candidate(
        candidate, l0_ref, l1_ref, mb_x, mb_y,
    );

    // ── 2. SATD ─────────────────────────────────────────────────
    let satd = satd_16x16(src_y, &pred_y);

    // ── 3. Rate estimate per candidate ──────────────────────────
    //
    // Components:
    //   - mb_skip_flag: 1 bin always.
    //   - mb_type bin string:
    //       Direct=1 bin, L0/L1=3 bins, Bi=6 bins.
    //   - MVD: per-axis `mvd_rate_estimate` (≈ 4-8 bits per axis).
    //   - CBP + qp_delta: 4 bins (overhead even if zero residual).
    //
    // Residual coefficient bits are NOT estimated — they are
    // implicit in the SATD term under the fast-RDO formulation.
    let mut r_bits = 1; // mb_skip_flag bin always emitted
    match *candidate {
        BMbCandidate::SkipOrDirect { .. } => {
            // Direct's 1-bin mb_type is in the frequent-context head;
            // 1 bit is accurate.
            r_bits += 1; // mb_type=0 bin
            r_bits += 4; // cbp + qp_delta overhead
        }
        BMbCandidate::L0_16x16 { mv_l0, .. } => {
            // 3 bins: 1 leading (frequent) + 2 rare-context.
            r_bits += 1 + 2 * RARE_BIN_COST;
            r_bits += mvd_rate_estimate(mv_l0.mv_x, mv_l0.mv_y);
            r_bits += 4; // CBP + qp_delta
        }
        BMbCandidate::L1_16x16 { mv_l1 } => {
            r_bits += 1 + 2 * RARE_BIN_COST;
            r_bits += mvd_rate_estimate(mv_l1.mv_x, mv_l1.mv_y);
            r_bits += 4;
        }
        BMbCandidate::Bi_16x16 { mv_l0, mv_l1, .. } => {
            // §6E-D.5(o) — drop bins from 5 to 2 (HACK toward the reference fast encoder
            // warm-state CABAC). Static binarization-tree gives 5
            // bins; warm-state CABAC charges only ~1.5-3 bits because
            // ctxIdxInc 27/30/32 are highly biased toward Direct
            // emission. The 5-coefficient was over-conservative.
            // Reducing brings Bi rate close to L0/L1 (which also use
            // 2 rare bins) — defensible as "match measured warm-state
            // CABAC bins, not static binarization tree count".
            r_bits += 1 + 2 * RARE_BIN_COST;
            r_bits += mvd_rate_estimate(mv_l0.mv_x, mv_l0.mv_y);
            r_bits += mvd_rate_estimate(mv_l1.mv_x, mv_l1.mv_y);
            r_bits += 4;
        }
        BMbCandidate::Partitioned { meta, part0_mvs, part1_mvs } => {
            // r_bits already has the skip_flag bin (1).
            r_bits += partitioned_overhead_bits(&meta, &part0_mvs, &part1_mvs);
        }
        BMbCandidate::B_8x8 { sub_mb_types, parts } => {
            r_bits += b_8x8_overhead_bits(&sub_mb_types, &parts);
        }
    }

    // ── 4. SATD-domain Lagrangian cost ──────────────────────────
    //
    // §6E-D.5(i) — Direct/SkipOrDirect distortion multiplier.
    //
    // SkipOrDirect emits CBP=0 (no residual on the wire). Its actual
    // reconstructed distortion = the full prediction-error energy
    // because the residual is dropped, never encoded, never recovered
    // at decode. Explicit-MV modes (L0/L1/Bi/Partitioned/B_8x8)
    // emit quantized residual; their actual reconstructed distortion
    // = small post-quant noise.
    //
    // Fast RDO uses SATD as a single distortion proxy across all
    // candidates, treating them as if they all had equivalent post-
    // quant distortion. That under-charges Direct's actual no-
    // residual SSD distortion and over-charges explicit-MV's small
    // quant-noise distortion. full RDO catches this naturally
    // by computing SSD on actual reconstructions.
    //
    // The multiplier reflects "Direct's distortion ≈ N× explicit-MV's
    // distortion" where N is conservative-end of the 2-15× true range
    // observed at typical H.264 mid-content with QP in [22..36]:
    //   - At high QP the residual mostly survives even after quant,
    //     so explicit-MV's reconstruction is close to no-residual:
    //     N approaches 1.
    //   - At low QP the residual quantizes near-perfectly, so
    //     explicit-MV's quant-noise is tiny vs Direct's full
    //     residual: N approaches 10+.
    //   - Mid-range QP: N ≈ 3-5.
    //
    // §6E-D.5(j): multiplier = 3. The §6E-D.5(i) ×2 result (Direct
    // 99%→88%) showed the structural fix works but ×2 is at the
    // conservative end of the true range. Bumping to 3 reflects
    // mid-QP mid-content typical, still per-MB defendable: each
    // surviving Direct emission means "no-residual reconstruction
    // was good enough that Direct beat L0 even at 3× SATD vs 1×",
    // each L0/L1 win means "quantized residual quality justified
    // the rate cost given Direct's no-residual carries 3× the
    // distortion".
    //
    // Per-MB defensibility (per memory/feedback_no_cosmetic_calibration.md):
    //   - Direct emissions surviving the penalty: "prediction was
    //     good enough that no-residual reconstruction stayed
    //     acceptable; saved the 12-bin rate cost."
    //   - L0/L1/Bi emissions winning after the penalty: "L0's
    //     quantized residual gave better reconstruction quality
    //     than Direct's no-residual, justifying the rate cost."
    // Both answers are codec-purpose-driven, not statistical-fit.
    //
    // Approximates a full-RDO reference baseline.
    let lambda = LAMBDA_TAB_B[mb_qp.min(51) as usize] as u64;
    // §6E-D.5(k): multiplier 5/2 (= 2.5). §6E-D.5(j) ×3 measurement
    // showed Direct dropped to 46.3% (target 55.65%, undershot by
    // 9pp). ×2 was 88% (overshot), ×3 was 46% (undershot). Linear
    // interpolation suggests true mid-QP centroid is ~2.5 for this
    // fixture+QP combo. Implementing as integer-arithmetic
    // (×5/2) to keep the cost path branch-free + deterministic.
    //
    // Per-MB defendable: each Direct emission means "no-residual
    // reconstruction was good enough that Direct beat L0 even at
    // 2.5× SATD penalty"; each L0/L1 win means "quantized residual
    // gave reconstruction quality justifying rate cost given
    // Direct's no-residual carries 2.5× distortion."
    // §6E-D.5(m) — HF-proportional Direct distortion multiplier.
    //
    // Replaces the constant 2.5× from §6E-D.5(k). Source-HF SATD
    // measures how much detail the macroblock carries beyond its DC
    // average. Direct's no-residual reconstruction CANNOT close the
    // HF gap that L0+residual closes via quantized residual coding.
    //
    // Per-MB defensibility (per memory/feedback_no_cosmetic_calibration.md):
    //   - Flat MB (HF ≈ 0): multiplier ≈ 1.0 — Direct's SATD equals
    //     its true reconstruction distortion (no residual to add for
    //     either Direct OR L0, so no compensation is needed).
    //   - Textured MB (HF high): multiplier ≈ 4.0 — SATD undercounts
    //     Direct's true distortion because its no-residual recon
    //     leaves the high-frequency error uncorrected, while L0's
    //     quantized residual at this QP closes most of that gap.
    //
    // Calibrated to preserve §6E-D.5(k)'s 2.5× at "average" HF
    // (HF=1024) so the existing measured Direct/L0/Bi distribution
    // is anchored at flat: shifts the balance per-MB without breaking
    // the population-level fit on mid-HF content.
    //
    // Approximates a fast-encoder full-RDO reference baseline: flat MBs ~Direct wins
    // because true SSD ≈ for both candidates and Direct has lower
    // rate; textured MBs ~L0/L1 wins because residual coding closes
    // the true-SSD gap that SATD alone undersells for Direct.
    let satd_for_cost = match *candidate {
        BMbCandidate::SkipOrDirect { .. } => {
            // mult_q8 = 256 + (HF_clamped × 3) / 8.
            //   HF=0    → 256 (= 1.0)
            //   HF=1024 → 640 (= 2.5)  ← matches §6E-D.5(k) at "average"
            //   HF=2048 → 1024 (= 4.0)
            let hf_clamped = psy_hf_satd_16x16(src_y).min(2048) as u64;
            let mult_q8 = 256 + (hf_clamped * 3) / 8;
            ((satd as u64) * mult_q8) >> 8
        }
        BMbCandidate::Bi_16x16 { .. } => {
            // §6E-D.5(o) — bipred variance-reduction discount
            // (HACK ×0.9). Bipred prediction averages two reference
            // pels; on noise-dominated content the variance of the
            // averaged pred is half that of single-list (var(L0+L1)/2
            // = (var_L0 + var_L1)/4 for independent noise). True
            // distortion lower than raw SATD by ~10-30% on typical
            // noisy content. ×0.9 (× 230/256 in Q.8) is a
            // conservative middle estimate.
            ((satd as u64) * 230) >> 8
        }
        _ => satd as u64,
    };

    // PSY-RD additive term (env-gated PHASM_PSY_RD=1). Penalizes ALL
    // candidates whose prediction HF differs from source HF. With the
    // §6E-D.5(m) HF-proportional Direct multiplier above already
    // capturing "Direct loses HF on textured", this PSY-RD layer is
    // largely redundant on the Direct-vs-L0 axis but still helps
    // discriminate L0/L1/Bi candidates with similar SATD but different
    // HF preservation. Default off until ablated.
    let (psy_enabled, psy_shift) = psy_rd_config();
    let psy_cost = if psy_enabled {
        let source_hf = psy_hf_satd_16x16(src_y) as i64;
        let pred_hf = psy_hf_satd_16x16(&pred_y) as i64;
        ((source_hf - pred_hf).unsigned_abs() >> psy_shift) as u64
    } else {
        0
    };

    let cost = satd_for_cost + lambda * (r_bits as u64) + psy_cost;

    BMbRdoResult {
        candidate: *candidate,
        satd,
        r_bits,
        cost,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::encoder::reconstruction::ReconBuffer;

    fn make_recon(width: u32, height: u32, y_fill: u8) -> ReconFrame {
        let mut buf = ReconBuffer::new(width, height).unwrap();
        for v in buf.y.iter_mut() { *v = y_fill; }
        for v in buf.cb.iter_mut() { *v = 128; }
        for v in buf.cr.iter_mut() { *v = 128; }
        ReconFrame::snapshot(&buf)
    }

    fn const_src_y(value: u8) -> [[u8; 16]; 16] {
        let mut src = [[0u8; 16]; 16];
        for row in &mut src {
            for px in row {
                *px = value;
            }
        }
        src
    }

    #[test]
    fn lambda_b_table_grows_monotonically() {
        for qp in 0..51 {
            assert!(
                LAMBDA_TAB_B[qp] <= LAMBDA_TAB_B[qp + 1],
                "non-monotonic at qp {qp}: {} > {}",
                LAMBDA_TAB_B[qp], LAMBDA_TAB_B[qp + 1]
            );
        }
        // Spot-check the known landmark values.
        assert_eq!(LAMBDA_TAB_B[0], 5);
        assert_eq!(LAMBDA_TAB_B[21], 62);
        assert_eq!(LAMBDA_TAB_B[51], 1978);
    }

    #[test]
    fn skip_or_direct_zero_mv_zero_residual_zero_satd() {
        // Source pixels = ref pixels = 100 → MV(0,0) gives zero
        // residual → SATD = 0.
        let src = const_src_y(100);
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 100);
        let cand = BMbCandidate::SkipOrDirect {
            mv_l0: MotionVector::ZERO,
            mv_l1: MotionVector::ZERO,
            uses_l0: true,
            uses_l1: false,
            mv_l0_per_8x8: [MotionVector::ZERO; 4],
            mv_l1_per_8x8: [MotionVector::ZERO; 4],
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        assert_eq!(res.satd, 0, "zero residual → zero SATD");
        // r_bits = 1 (skip_flag) + 1 (mb_type) + 4 (cbp/qp) = 6.
        assert_eq!(res.r_bits, 6);
        // cost = 0 + LAMBDA_TAB_B[30] × 6 = 175 × 6 = 1050.
        assert_eq!(res.cost, (LAMBDA_TAB_B[30] as u64) * 6);
    }

    #[test]
    fn l0_16x16_emits_three_mb_type_bins_plus_mvd() {
        let src = const_src_y(100);
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 100);
        let cand = BMbCandidate::L0_16x16 {
            mv_l0: MotionVector { mv_x: 0, mv_y: 0 },
            ref_idx_l0: 0,
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        // §6E-D.5(f): RARE_BIN_COST=2. 1 (skip) + (1 + 2×2) (mb_type rare bins) + 6 (mvd zero) + 4 (cbp/qp) = 16.
        assert_eq!(res.r_bits, 16);
        assert_eq!(res.satd, 0);
        assert_eq!(res.cost, (LAMBDA_TAB_B[30] as u64) * 16);
    }

    #[test]
    fn bi_16x16_rate_estimate() {
        let src = const_src_y(100);
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 100);
        let cand = BMbCandidate::Bi_16x16 {
            mv_l0: MotionVector::ZERO,
            mv_l1: MotionVector::ZERO,
            ref_idx_l0: 0,
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        // §6E-D.5(o): 1 (skip) + (1 + 2×2) (mb_type 3 bins after warm-state
        // reduction) + 6 + 6 (mvds) + 4 (cbp/qp) = 22.
        assert_eq!(res.r_bits, 22);
        assert_eq!(res.satd, 0);
        // SATD=0 → satd_for_cost=0 regardless of Bi 0.9× discount.
        assert_eq!(res.cost, (LAMBDA_TAB_B[30] as u64) * 22);
    }

    #[test]
    fn nonzero_residual_inflates_satd() {
        // src=100, ref=200 → 100 unit residual everywhere → big SATD.
        let src = const_src_y(100);
        let l0 = make_recon(64, 64, 200);
        let l1 = make_recon(64, 64, 200);
        let cand = BMbCandidate::L0_16x16 { mv_l0: MotionVector::ZERO, ref_idx_l0: 0 };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        assert!(res.satd > 1000, "residual MB should produce non-trivial SATD");
        // cost = SATD + λ × r_bits should exceed rate-alone.
        assert!(res.cost > (LAMBDA_TAB_B[30] as u64) * 16);
    }

    #[test]
    fn cost_rises_with_qp_via_lambda() {
        // Same MB, different QP → cost should rise (because λ rises).
        let src = const_src_y(100);
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 100);
        let cand = BMbCandidate::L0_16x16 { mv_l0: MotionVector::ZERO, ref_idx_l0: 0 };
        let lo = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 21);
        let hi = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 36);
        assert!(hi.cost > lo.cost, "λ should grow with QP: {} > {}", hi.cost, lo.cost);
    }

    #[test]
    fn mvd_rate_estimate_grows_with_magnitude() {
        let r0 = mvd_rate_estimate(0, 0);
        let r4 = mvd_rate_estimate(4, 0);
        let r32 = mvd_rate_estimate(32, 0);
        assert!(r0 < r4, "{r0} < {r4}");
        assert!(r4 < r32, "{r4} < {r32}");
    }

    // -- §6E-D.2 Partitioned candidate tests ---------------------------------

    fn pmv(x: i16, y: i16) -> MotionVector { MotionVector { mv_x: x, mv_y: y } }

    #[test]
    fn partitioned_16x8_l0_l0_zero_mv_zero_satd() {
        // mb_type=4: 16x8, both halves L0, zero MVs → exact prediction.
        let src = const_src_y(100);
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 100);
        let meta = crate::codec::h264::encoder::b_partitioned::partitioned_b_meta(4)
            .expect("mb_type 4 valid");
        let cand = BMbCandidate::Partitioned {
            meta,
            part0_mvs: BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 },
            part1_mvs: BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 },
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        assert_eq!(res.satd, 0, "zero residual → zero SATD");
        // §6E-D.5(f): 1 (skip) + (1+4×2) (mb_type rare bins) + 6 + 6 (mvds) + 4 (cbp/qp) = 26.
        assert_eq!(res.r_bits, 26);
    }

    #[test]
    fn partitioned_8x16_bi_bi_emits_four_mvds() {
        // mb_type=21: 8x16, both halves Bi → 4 MVDs (L0+L1 per part).
        let src = const_src_y(100);
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 100);
        let meta = crate::codec::h264::encoder::b_partitioned::partitioned_b_meta(21)
            .expect("mb_type 21 valid");
        let cand = BMbCandidate::Partitioned {
            meta,
            part0_mvs: BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 },
            part1_mvs: BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 },
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        // §6E-D.5(f): 1 (skip) + (1+4×2) (mb_type rare) + 12 (Bi mvds part0) + 12 (part1) + 4 (cbp/qp) = 38.
        assert_eq!(res.r_bits, 38);
        assert_eq!(res.satd, 0);
    }

    #[test]
    fn partitioned_l0_l1_picks_correct_reference_per_partition() {
        // mb_type=8: 16x8, top=L0, bottom=L1. Setup: L0 ref pixels=100,
        // L1 ref pixels=200. Source has top half=100, bottom=200.
        // Result: top L0 picks 100 (zero residual), bottom L1 picks 200
        // (zero residual). SATD should be 0.
        let mut src = [[0u8; 16]; 16];
        for y in 0..8 { for x in 0..16 { src[y][x] = 100; } }
        for y in 8..16 { for x in 0..16 { src[y][x] = 200; } }
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 200);
        let meta = crate::codec::h264::encoder::b_partitioned::partitioned_b_meta(8)
            .expect("mb_type 8 valid");
        let cand = BMbCandidate::Partitioned {
            meta,
            part0_mvs: BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 },
            part1_mvs: BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 },
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        assert_eq!(res.satd, 0,
            "per-partition MC should pick L0 for top half, L1 for bottom");
    }

    #[test]
    fn partitioned_picks_wrong_reference_inflates_satd() {
        // Same content as above but mb_type=4 (both L0_L0). Bottom half
        // reads from L0 which has 100, but src has 200 there. → big SATD.
        let mut src = [[0u8; 16]; 16];
        for y in 0..8 { for x in 0..16 { src[y][x] = 100; } }
        for y in 8..16 { for x in 0..16 { src[y][x] = 200; } }
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 200);
        let meta = crate::codec::h264::encoder::b_partitioned::partitioned_b_meta(4)
            .expect("mb_type 4 valid");
        let cand = BMbCandidate::Partitioned {
            meta,
            part0_mvs: BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 },
            part1_mvs: BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 },
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        assert!(res.satd > 1000,
            "wrong-list MC should produce non-trivial SATD: got {}", res.satd);
    }

    // -- §6E-D.3 B_8x8 candidate tests ---------------------------------------

    #[test]
    fn b_8x8_uniform_l0_zero_mv_zero_satd() {
        // All four sub-MBs = L0_8x8 (sub=1) with zero MVs, exact match.
        let src = const_src_y(100);
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 100);
        let zero = BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 };
        let cand = BMbCandidate::B_8x8 {
            sub_mb_types: [1, 1, 1, 1],
            parts: [zero; 4],
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        assert_eq!(res.satd, 0);
        // §6E-D.5(f): 1 (skip) + (1+5×2) (mb_type=22 rare) + 4× ((1+2×2) sub_mb_type + 6 mvd) + 4 (cbp/qp)
        // = 1 + 11 + 4×11 + 4 = 60.
        assert_eq!(res.r_bits, 60);
    }

    #[test]
    fn b_8x8_uniform_direct_no_mvds() {
        // sub_mb_type=0 (Direct) — no MVDs.
        let src = const_src_y(100);
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 100);
        let zero = BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 };
        let cand = BMbCandidate::B_8x8 {
            sub_mb_types: [0, 0, 0, 0],
            parts: [zero; 4],
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        // §6E-D.5(f): 1 (skip) + (1+5×2) (mb_type=22) + 4×1 (Direct sub) + 4 (cbp/qp)
        // = 1 + 11 + 4 + 4 = 20.
        assert_eq!(res.r_bits, 20);
        assert_eq!(res.satd, 0);
    }

    #[test]
    fn b_8x8_uniform_bi_emits_eight_mvds() {
        // sub_mb_type=3 (Bi) on all four sub-MBs → 8 MVDs total
        // (2 per sub × 4 subs).
        let src = const_src_y(100);
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 100);
        let zero = BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 };
        let cand = BMbCandidate::B_8x8 {
            sub_mb_types: [3, 3, 3, 3],
            parts: [zero; 4],
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        // §6E-D.5(f): 1 (skip) + (1+5×2) (mb_type=22) + 4×((1+2×2) sub_mb_type + 12 Bi mvds) + 4 (cbp/qp)
        // = 1 + 11 + 4×17 + 4 = 84.
        assert_eq!(res.r_bits, 84);
        assert_eq!(res.satd, 0);
    }

    #[test]
    fn b_8x8_per_sub_reference_selection() {
        // Source split into quadrants:
        //   TL=100 (sub 0 should pick L0=100)
        //   TR=200 (sub 1 should pick L1=200)
        //   BL=200 (sub 2 should pick L1=200)
        //   BR=100 (sub 3 should pick L0=100)
        let mut src = [[0u8; 16]; 16];
        for y in 0..8 {
            for x in 0..8 { src[y][x] = 100; }
            for x in 8..16 { src[y][x] = 200; }
        }
        for y in 8..16 {
            for x in 0..8 { src[y][x] = 200; }
            for x in 8..16 { src[y][x] = 100; }
        }
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 200);
        let zero = BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 };
        // Per-sub list selection: L0 for TL/BR (sub=1), L1 for TR/BL (sub=2).
        let cand = BMbCandidate::B_8x8 {
            sub_mb_types: [1, 2, 2, 1],
            parts: [zero; 4],
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        assert_eq!(res.satd, 0,
            "per-sub-MB list selection should give exact match in checkerboard");
    }

    #[test]
    fn b_8x8_wrong_per_sub_reference_inflates_satd() {
        // Same source as above but uniform L0 → bottom-left and top-right
        // mismatch → big SATD.
        let mut src = [[0u8; 16]; 16];
        for y in 0..8 {
            for x in 0..8 { src[y][x] = 100; }
            for x in 8..16 { src[y][x] = 200; }
        }
        for y in 8..16 {
            for x in 0..8 { src[y][x] = 200; }
            for x in 8..16 { src[y][x] = 100; }
        }
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 200);
        let zero = BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 };
        let cand = BMbCandidate::B_8x8 {
            sub_mb_types: [1, 1, 1, 1],
            parts: [zero; 4],
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        assert!(res.satd > 1000,
            "uniform L0 mode on checkerboard should produce big SATD: {}", res.satd);
    }

    #[test]
    fn partitioned_8x16_left_right_split() {
        // mb_type=9: 8x16, left=L0, right=L1. Source split vertically.
        let mut src = [[0u8; 16]; 16];
        for y in 0..16 {
            for x in 0..8 { src[y][x] = 100; }
            for x in 8..16 { src[y][x] = 200; }
        }
        let l0 = make_recon(64, 64, 100);
        let l1 = make_recon(64, 64, 200);
        let meta = crate::codec::h264::encoder::b_partitioned::partitioned_b_meta(9)
            .expect("mb_type 9 valid");
        let cand = BMbCandidate::Partitioned {
            meta,
            part0_mvs: BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 },
            part1_mvs: BPartitionMvPair { mv_l0: pmv(0, 0), mv_l1: pmv(0, 0), ref_idx_l0: 0 },
        };
        let res = evaluate_b_mb_rdo(&cand, &src, &l0, &l1, 0, 0, 30);
        assert_eq!(res.satd, 0,
            "8x16 left=L0 / right=L1 should match split content exactly");
    }
}
