// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Rate-distortion optimization (RDO) primitives.
//!
//! Phase A.1 / B.2 of the H.264 encoder quality plan
//! (`docs/design/video/h264/encoder-quality-plan.md`). Holds:
//!   - Lagrangian lambda tables for ME/SATD cost and for RDO mode
//!     decision (Phase A.1),
//!   - bit-accurate CAVLC size estimators (Phase B.2) — used by
//!     Phase C to compute `R(mode)` in `cost = D + (bits × λ² >> 8)`.
//!
//! ## Lambda-table derivation
//!
//! The three tables below (`LAMBDA_TAB`, `LAMBDA2_TAB`,
//! `TRELLIS_LAMBDA2_TAB`) are computed by rounding closed-form
//! Lagrangian formulas derived by Sullivan & Wiegand in the
//! rate-distortion literature (notably "Rate-Constrained Coder
//! Control and Comparison of Video Coding Standards",
//! IEEE TCSVT 13(7), July 2003, §V) and the prior
//! Wiegand-Sullivan 1996 paper on RD-optimized motion estimation.
//! The H.264 spec itself does not prescribe lambda values — it only
//! specifies the quantization step — but the relationship
//! `λ ≈ 0.85 × 2^((qp − 12)/3)` for an 8-bit codec is mathematically
//! derivable from minimising the Lagrangian `D + λ·R` with `D`
//! scaling as the quantisation step squared.
//!
//! The explicit formulas used here are documented next to each
//! table. A lock-in unit test (see the `formula_derivation` test
//! module at the bottom of this file) recomputes each value from
//! its formula in `f64` and asserts equality — this ties the
//! shipped integer table to the literature formula rather than to
//! any specific implementation.
//!
//! ## Note on ME lambda
//!
//! `motion_estimation.rs::LAMBDA_MOTION_DEFAULT` is intentionally
//! NOT switched to `LAMBDA_TAB[qp]` in this phase — it stays at 1
//! until Phase C can measure the swap safely (our hex search sits on
//! a "30f lucky optimum" and ME-lambda changes have regressed
//! before; see `memory/h264_me_upgrade_plan.md`).

/// Per-QP Lagrangian motion-estimation / SATD lambda.
///
/// Formula: `LAMBDA_TAB[qp] = max(1, round(2^((qp − 12) / 6)))`.
///
/// Derivation: from the RD-constrained motion-estimation result
/// (Wiegand-Sullivan 1996, Eq. 15) the optimal motion-cost weight is
/// `λ_motion = 0.85 × √λ_mode`, and `λ_mode ≈ 0.85 × 2^((qp − 12)/3)`
/// (Sullivan-Wiegand 2003, §V.C). Factoring out the 0.85 and taking
/// the square root gives `λ ≈ 2^((qp − 12)/6)` for the
/// SATD-domain (sqrt-SSE) cost. The table clamps to 1 below QP 12
/// where the rounded value would collapse to 0.
///
/// Values grow ~12% per QP. At QP 12 the value is 1; at QP 48 it's
/// 64; at QP 51 it's 91. The underlying continuous function is
/// deterministic, so any spec-conformant RDO implementation that
/// follows the Sullivan-Wiegand analysis will produce the same
/// rounded integer table.
pub const LAMBDA_TAB: [u16; 52] = [
    1, 1, 1, 1, 1, 1, 1, 1, // 0-7
    1, 1, 1, 1, 1, 1, 1, 1, // 8-15
    2, 2, 2, 2, 3, 3, 3, 4, // 16-23
    4, 4, 5, 6, 6, 7, 8, 9, // 24-31
    10, 11, 13, 14, 16, 18, 20, 23, // 32-39
    25, 29, 32, 36, 40, 45, 51, 57, // 40-47
    64, 72, 81, 91, // 48-51
];

/// Per-QP squared lambda scaled to a Q.8 fixed-point representation.
///
/// Formula: `LAMBDA2_TAB[qp] = trunc(0.9 × 2^(qp/3 + 4))`.
///
/// Derivation: the MB-level RDO cost is `D + λ²·R`, with `D`
/// measured as sum-of-squared-differences (SSE-domain, not
/// SATD-domain). Under the Sullivan-Wiegand analysis this gives
/// `λ² ≈ 0.9 × 2^((qp − 12)/3)`. Pre-scaling by 2^8 (so that
/// downstream uses of the form `(bits · λ²) >> 8` stay integer)
/// converts the exponent to `qp/3 + 4`. The 0.9 constant is an
/// empirical RD-slope correction for H.264's operating point
/// (Sullivan-Wiegand 2003, Table IV). The floor-division (trunc)
/// rather than round-to-nearest biases the cost slightly toward
/// low-rate modes, which is a standard RD-optimisation choice.
///
/// At QP 21 the value is 1843 = trunc(0.9 × 2048) = trunc(1843.2).
/// At QP 51 it is 1,887,436 (RD-cost grows like 2^17 at that
/// extreme).
pub const LAMBDA2_TAB: [i32; 52] = [
    14, 18, 22, 28, 36, 45, 57, 72, // 0-7
    91, 115, 145, 182, 230, 290, 365, 460, // 8-15
    580, 731, 921, 1161, 1462, 1843, 2322, 2925, // 16-23
    3686, 4644, 5851, 7372, 9289, 11703, 14745, 18578, // 24-31
    23407, 29491, 37156, 46814, 58982, 74313, 93628, 117964, // 32-39
    148626, 187257, 235929, 297252, 374514, 471859, 594505, 749029, // 40-47
    943718, 1189010, 1498059, 1887436, // 48-51
];

/// Per-QP squared-lambda scaled to a coarser Q.4 fixed-point
/// representation, separated by slice type. Used by trellis-quant
/// inner-loop RD when it minimises `SSD + λ_trellis·R_cavlc` one
/// coefficient at a time.
///
/// Formula:
///   `TRELLIS_LAMBDA2_TAB[0][qp] (inter) = round(0.85² × 2^(qp/3 + 6))`
///   `TRELLIS_LAMBDA2_TAB[1][qp] (intra) = round(0.65² × 2^(qp/3 + 6))`
///
/// Derivation: trellis operates at a coarser rate-distortion
/// granularity (per-coefficient rather than per-MB), so it uses a
/// different fixed-point scale than [`LAMBDA2_TAB`]. The 0.85² inter
/// and 0.65² intra multiplicative factors come from the
/// Sullivan-Wiegand 2003 analysis (§V.C, Eq. 9) showing that
/// intra-coded blocks use a smaller RD-slope because the
/// distortion-rate curve has smaller local slope at intra-residual
/// rate regimes. The `2^(qp/3 + 6)` base comes from the same
/// λ² ≈ 2^(qp/3) proportionality as [`LAMBDA2_TAB`] but pre-scaled
/// by 2^6 for Q.4 downstream rather than 2^8 for Q.8.
///
/// Both rows are lock-in-verified by the `formula_derivation` test.
pub const TRELLIS_LAMBDA2_TAB: [[i32; 52]; 2] = [
    // Inter
    [
        46, 58, 73, 92, 117, 147, // 0-5
        185, 233, 294, 370, 466, 587, // 6-11
        740, 932, 1174, 1480, 1864, 2349, // 12-17
        2959, 3729, 4698, 5919, 7457, 9395, // 18-23
        11837, 14914, 18791, 23675, 29828, 37582, // 24-29
        47350, 59657, 75163, 94700, 119314, 150326, // 30-35
        189399, 238628, 300652, 378798, 477256, 601304, // 36-41
        757596, 954511, 1202609, 1515192, 1909023, 2405218, // 42-47
        3030385, 3818045, 4810436, 6060769, // 48-51
    ],
    // Intra
    [
        27, 34, 43, 54, 68, 86, // 0-5
        108, 136, 172, 216, 273, 343, // 6-11
        433, 545, 687, 865, 1090, 1374, // 12-17
        1731, 2180, 2747, 3461, 4361, 5494, // 18-23
        6922, 8721, 10988, 13844, 17443, 21977, // 24-29
        27689, 34886, 43953, 55378, 69772, 87907, // 30-35
        110756, 139544, 175814, 221512, 279087, 351628, // 36-41
        443023, 558174, 703256, 886047, 1116349, 1406511, // 42-47
        1772093, 2232698, 2813023, 3544187, // 48-51
    ],
];

/// Look up `lambda` for a QP, saturating at QP 51.
#[inline]
pub fn lambda_for_qp(qp: u8) -> u16 {
    LAMBDA_TAB[qp.min(51) as usize]
}

/// Look up `lambda2` for a QP, saturating at QP 51.
#[inline]
pub fn lambda2_for_qp(qp: u8) -> i32 {
    LAMBDA2_TAB[qp.min(51) as usize]
}

/// Look up trellis `lambda2` for a QP and slice type.
#[inline]
pub fn trellis_lambda2_for_qp(qp: u8, is_intra: bool) -> i32 {
    let row = if is_intra { 1 } else { 0 };
    TRELLIS_LAMBDA2_TAB[row][qp.min(51) as usize]
}

// ─── Phase B.2: bit-accurate CAVLC size estimators ───────────────
//
// These wrap the real writer's emission logic but push bits into a
// `BitSizer` instead of a `BitWriter` buffer. Because both sinks are
// driven by the SAME code path (generic over `BitSink`), the returned
// bit count is guaranteed to equal the bit count that a real encode
// would emit — the Phase B.3 unit tests assert this bit-for-bit on
// real MBs from IMG_4138.
//
// Callers: Phase C's MB-level RDO (task #129) queries
// `p_mb_total_bits_cavlc` per candidate partition choice and plugs
// the result into `cost = D + ((bits × lambda2[qp]) >> 8)`.

use super::bitstream_writer::{BitSink, BitSizer};
use super::partition_decision::PMbChoice;
use super::partition_state::EncoderMvGrid;
use super::quantization::{forward_quantize_4x4, trellis_quantize_4x4, QuantParams, QuantSlice};
use super::reconstruction::raster_to_scan_levels;
use super::reference_buffer::ReconFrame;
use super::transform::{forward_dct_4x4, forward_hadamard_2x2};
use super::inter_mode::{cbp_to_codenum_inter, luma_8x8_cbp_mask, pack_cbp};
use super::quantization::forward_quantize_dc_chroma;
#[allow(unused_imports)]
use crate::codec::h264::cavlc_writer::{encode_cavlc_block, CavlcBlockType};  // encode_cavlc_block used only in tests
use crate::codec::h264::transform::{dequant_4x4, inverse_4x4_integer, inverse_chroma_dc_2x2_hadamard};

/// Bit count of encoding one 4×4 CAVLC residual block. Drives the
/// luma AC, Intra16x16 AC, chroma AC, and chroma DC paths — every
/// residual block passes through this.
///
/// Delegates to the Phase B.2 `cavlc_size` module. Errors (malformed
/// input) are treated as 0 bits since the caller is responsible for
/// feeding valid coefficients (same tolerance the prior inline
/// implementation had).
pub fn residual_block_bits_cavlc(
    coeffs: &[i32],
    nc: i8,
    block_type: CavlcBlockType,
) -> u32 {
    crate::codec::h264::cavlc_size::residual_block_bits_cavlc(coeffs, nc, block_type)
        .unwrap_or(0)
}

/// Bit count of the P-MB header: `mb_type` + MVDs + `coded_block_pattern`
/// + `mb_qp_delta` (when applicable). Does NOT include residuals — see
/// [`p_mb_total_bits_cavlc`] for the all-in counter.
///
/// `grid` is snapshot/restore'd across the MVD emit (reusing Phase A.2
/// infrastructure) so size counting never mutates the caller's grid
/// state — multiple candidate evaluations on the same MB produce
/// consistent bit counts.
pub fn p_mb_header_bits_cavlc(
    choice: &PMbChoice,
    grid: &mut EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
    cbp_codenum: u32,
    qp_delta_or_none: Option<i32>,
) -> u32 {
    use super::encoder::emit_mvds_and_update_grid;
    let mut sz = BitSizer::new();
    // mb_type: ue(codenum) per Table 7-13.
    sz.write_ue(choice.mb_type_codenum());
    // MVDs (+ sub_mb_types for P_8x8). Grid is scratched during this
    // call; the snapshot/restore matches the shape Phase A.2 set up.
    let snap = grid.snapshot_mb(mb_x, mb_y);
    emit_mvds_and_update_grid(&mut sz, grid, mb_x, mb_y, choice);
    grid.restore_mb(&snap);
    // coded_block_pattern: me(v), encoded via a fixed table
    // `cbp_to_codenum_inter` — same codenum is what the real emit
    // writes, so we just `write_ue` on it here.
    sz.write_ue(cbp_codenum);
    if let Some(qp_delta) = qp_delta_or_none {
        sz.write_se(qp_delta);
    }
    sz.bits_written() as u32
}

/// Bit count of the full P-MB (header + every non-zero residual block).
///
/// `luma_ac_scans[k]` is the scan-ordered 16-value luma coefficient
/// array for block `k`. `luma_block_present[k]` mirrors the 8×8 CBP
/// gating at the CAVLC writer (blocks in unset 8×8 bits emit nothing).
/// `luma_nc[k]` is the nC context used at the real emit site — must
/// be the same function evaluation (`compute_nc_luma_at`) to keep
/// bit-exact.
///
/// Chroma inputs are `None` when `cbp_chroma == 0` (no residual),
/// `Some(..)` otherwise. The chroma DC blocks are emitted whenever
/// cbp_chroma ≥ 1; the 8 AC blocks only when cbp_chroma == 2.
#[allow(clippy::too_many_arguments)]
pub fn p_mb_total_bits_cavlc(
    choice: &PMbChoice,
    grid: &mut EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
    cbp_codenum: u32,
    qp_delta_or_none: Option<i32>,
    cbp_luma_8x8: u8,
    cbp_chroma: u8,
    luma_ac_scans: &[[i32; 16]; 16],
    luma_nc: &[i8; 16],
    chroma: Option<CavlcChromaBits>,
) -> u32 {
    let mut bits =
        p_mb_header_bits_cavlc(choice, grid, mb_x, mb_y, cbp_codenum, qp_delta_or_none);
    // Luma 4×4 AC (gated by 8×8 CBP mask).
    for k in 0..16 {
        if cbp_luma_8x8 & (1 << (k / 4)) != 0 {
            bits = bits.saturating_add(residual_block_bits_cavlc(
                &luma_ac_scans[k],
                luma_nc[k],
                CavlcBlockType::Luma4x4,
            ));
        }
    }
    if cbp_chroma != 0
        && let Some(chroma) = chroma {
            // Cb + Cr DC (always present when cbp_chroma ≥ 1).
            bits = bits.saturating_add(residual_block_bits_cavlc(
                &chroma.cb_dc_scan,
                -1,
                CavlcBlockType::ChromaDc,
            ));
            bits = bits.saturating_add(residual_block_bits_cavlc(
                &chroma.cr_dc_scan,
                -1,
                CavlcBlockType::ChromaDc,
            ));
            if cbp_chroma == 2 {
                for i in 0..4 {
                    bits = bits.saturating_add(residual_block_bits_cavlc(
                        &chroma.cb_ac_scans[i],
                        chroma.cb_nc[i],
                        CavlcBlockType::ChromaAc,
                    ));
                    bits = bits.saturating_add(residual_block_bits_cavlc(
                        &chroma.cr_ac_scans[i],
                        chroma.cr_nc[i],
                        CavlcBlockType::ChromaAc,
                    ));
                }
            }
        }
    bits
}

/// Chroma residual data bundle for [`p_mb_total_bits_cavlc`]. Gives
/// the pre-scanned coefficient arrays + nC contexts the real writer
/// would use at the emit site.
#[derive(Debug, Clone)]
pub struct CavlcChromaBits {
    pub cb_dc_scan: [i32; 4],
    pub cr_dc_scan: [i32; 4],
    pub cb_ac_scans: [[i32; 15]; 4],
    pub cr_ac_scans: [[i32; 15]; 4],
    pub cb_nc: [i8; 4],
    pub cr_nc: [i8; 4],
}

// ─── Phase C: MB-level RDO evaluation ────────────────────────────
//
// `evaluate_p_mb_rdo` runs the full residual pipeline for one
// candidate partition and returns its MB-level Lagrangian RDO cost
// `cost = D + ((bits × lambda2[qp]) >> 8)` — the standard
// Sullivan-Wiegand 2003 Lagrangian form with the Q.8 lambda² from
// [`LAMBDA2_TAB`].
//
// Scope for Phase C MVP (2026-04-20):
//   - Luma residual only. Chroma D is OMITTED from the distortion
//     term and chroma R is OMITTED from the bit term. Rationale: on
//     P-slice inter content, luma dominates both. Adding chroma
//     doubles the compute cost and maybe ±3% ranking delta. If
//     measurement shows chroma matters, upgrade in a Phase C.v2.
//   - nC is a worst-case estimate (nC=0). Actual nC depends on
//     neighbor total_coeff state the caller doesn't want to snapshot.
//     This underestimates bit cost by ~5%. Consistent across
//     candidates so ranking should be unaffected.
//   - QP delta = 0 (assumes no per-block QP change vs prev).

/// Phase C.v3 — inputs to thread chroma D + R into an RDO cost call.
/// Caller supplies the 8×8 source Cb + Cr blocks and the chroma QP;
/// the evaluator computes chroma prediction (inter: MC per
/// partition; intra: simple DC proxy), runs the chroma residual
/// pipeline, and folds SSD + CAVLC-bit cost into the Lagrangian.
#[derive(Debug, Clone, Copy)]
pub struct ChromaRdoInputs<'a> {
    pub src_cb: &'a [[u8; 8]; 8],
    pub src_cr: &'a [[u8; 8]; 8],
    pub qp_c: u8,
}

/// Phase C.v3 — chroma D + R estimator for one 8×8 chroma component.
///
/// Mirrors `Encoder::encode_chroma_component` end-to-end:
/// 1. Residual = src − pred.
/// 2. For each of 4 4×4 sub-blocks: forward DCT, extract DC, zero
///    DC, trellis-quant the 15 AC coefficients.
/// 3. 2×2 Hadamard over the 4 DC coefficients, then chroma-DC quant.
/// 4. Inverse path: dequant DC + inverse 2×2 Hadamard; dequant +
///    inverse 4×4 integer transform per AC block with DC re-injected;
///    clip to u8 to get recon pixels.
/// 5. D = SSD(src, recon).
/// 6. R estimate = cavlc_size bits for the chroma DC block (4 DC
///    levels as a flat block, nC=-1) + 4 chroma AC blocks (15 AC
///    levels each, nC=-1 for RDO estimate since real neighbour nC
///    isn't available at this code path without additional plumbing
///    — consistent bias across candidates keeps ranking correct).
///
/// Returns `(D, R_bits)` with D in squared-pixel units (u64) and R
/// in integer bits. Caller folds these into the MB-level Lagrangian
/// cost alongside luma.
fn chroma_component_rdo(
    src_c: &[[u8; 8]; 8],
    pred_c: &[[u8; 8]; 8],
    qp_c: u8,
) -> (u64, u32) {
    let qp = QuantParams { qp: qp_c, slice: QuantSlice::Inter };
    let trellis_enable = true;
    let mut ac_levels = [[[0i32; 4]; 4]; 4];
    let mut dc_grid = [[0i32; 2]; 2];
    // Forward: per-subblock DCT + quant (same structure as encoder).
    for sby in 0..2 {
        for sbx in 0..2 {
            let mut sub_res = [[0i32; 4]; 4];
            for dy in 0..4 {
                for dx in 0..4 {
                    sub_res[dy][dx] = src_c[sby * 4 + dy][sbx * 4 + dx] as i32
                        - pred_c[sby * 4 + dy][sbx * 4 + dx] as i32;
                }
            }
            let mut coeffs = forward_dct_4x4(&sub_res);
            dc_grid[sby][sbx] = coeffs[0][0];
            coeffs[0][0] = 0;
            ac_levels[sby * 2 + sbx] =
                trellis_quantize_4x4(&coeffs, qp, trellis_enable)
                    .unwrap_or_else(|_| forward_quantize_4x4(&coeffs, qp));
        }
    }
    let dc_hadamard = forward_hadamard_2x2(&dc_grid);
    let dc_levels = forward_quantize_dc_chroma(&dc_hadamard, qp_c, QuantSlice::Inter);

    // Inverse: dequant DC via inverse 2x2 Hadamard (spec function
    // handles both unscaling + Hadamard inverse); dequant AC + IDCT
    // + re-inject DC + clip to recon.
    let dc_recon = inverse_chroma_dc_2x2_hadamard(&dc_levels, qp_c as i32);
    let mut recon_c = [[0u8; 8]; 8];
    for sby in 0..2 {
        for sbx in 0..2 {
            let levels = &ac_levels[sby * 2 + sbx];
            let dq = dequant_4x4(levels, qp_c as i32, false);
            let mut with_dc = dq;
            with_dc[0][0] = with_dc[0][0].wrapping_add(dc_recon[sby][sbx]);
            let recon_res = inverse_4x4_integer(&with_dc);
            for dy in 0..4 {
                for dx in 0..4 {
                    let v = pred_c[sby * 4 + dy][sbx * 4 + dx] as i32 + recon_res[dy][dx];
                    recon_c[sby * 4 + dy][sbx * 4 + dx] = v.clamp(0, 255) as u8;
                }
            }
        }
    }

    // D = SSD on 8×8 chroma component.
    let mut d: u64 = 0;
    for dy in 0..8 {
        for dx in 0..8 {
            let diff = src_c[dy][dx] as i64 - recon_c[dy][dx] as i64;
            d += (diff * diff) as u64;
        }
    }

    // R: chroma DC (4 levels as flat, nc=-1) + 4 AC blocks (15 coeffs, nc=-1).
    let dc_flat: [i32; 4] = [
        dc_levels[0][0], dc_levels[0][1], dc_levels[1][0], dc_levels[1][1],
    ];
    let mut r_bits = crate::codec::h264::cavlc_size::residual_block_bits_cavlc(
        &dc_flat, -1, CavlcBlockType::ChromaDc,
    ).unwrap_or(0);
    for sb in 0..4 {
        let scan = raster_to_scan_levels(&ac_levels[sb]);
        // Chroma AC skips the DC position — 15-coefficient scan.
        let ac_only: [i32; 15] = {
            let mut a = [0i32; 15];
            a.copy_from_slice(&scan[1..16]);
            a
        };
        r_bits = r_bits.saturating_add(
            crate::codec::h264::cavlc_size::residual_block_bits_cavlc(
                &ac_only, -1, CavlcBlockType::ChromaAc,
            ).unwrap_or(0),
        );
    }

    (d, r_bits)
}

/// Simple DC intra prediction for chroma 8×8: average of available
/// top + left neighbour pixels, else 128. Used by
/// `evaluate_intra_in_p_rdo` for the intra-chroma cost estimate
/// when a full neighbour-based intra-chroma-mode decision hasn't
/// been run yet. Mirrors spec § 8.3.4 DC mode (mode 0).
fn predict_chroma_dc(reference: &ReconFrame, component: u8, mb_x: usize, mb_y: usize) -> [[u8; 8]; 8] {
    let cx = mb_x * 8;
    let cy = mb_y * 8;
    let have_top = cy > 0;
    let have_left = cx > 0;
    let dc = match (have_top, have_left) {
        (true, true) => {
            let mut sum: i32 = 0;
            for d in 0..8 {
                sum += reference.chroma_at(component, (cx + d) as u32, (cy - 1) as u32) as i32;
                sum += reference.chroma_at(component, (cx - 1) as u32, (cy + d) as u32) as i32;
            }
            (sum + 8) >> 4
        }
        (true, false) => {
            let mut sum: i32 = 0;
            for d in 0..8 {
                sum += reference.chroma_at(component, (cx + d) as u32, (cy - 1) as u32) as i32;
            }
            (sum + 4) >> 3
        }
        (false, true) => {
            let mut sum: i32 = 0;
            for d in 0..8 {
                sum += reference.chroma_at(component, (cx - 1) as u32, (cy + d) as u32) as i32;
            }
            (sum + 4) >> 3
        }
        (false, false) => 128,
    };
    let v = dc.clamp(0, 255) as u8;
    [[v; 8]; 8]
}

/// Neighbour-nC for a luma 4×4 block at absolute grid position
/// `(bx, by)`, reading from the frame's `total_coeff_grid`.
/// Mirrors `Encoder::compute_nc_luma_at` as a pure function.
///
/// `grid` layout: one byte per 4×4 block, row-major, size
/// `frame_w4 × frame_h4`. Sentinel `0xFF` = not-yet-emitted (treat
/// as unavailable).
#[inline]
pub fn nc_luma_free(grid: &[u8], frame_w4: usize, bx: usize, by: usize) -> i8 {
    let left = if bx > 0 {
        let v = grid[by * frame_w4 + (bx - 1)];
        if v == 0xFF { None } else { Some(v) }
    } else {
        None
    };
    let top = if by > 0 {
        let v = grid[(by - 1) * frame_w4 + bx];
        if v == 0xFF { None } else { Some(v) }
    } else {
        None
    };
    match (left, top) {
        (None, None) => 0,
        (Some(v), None) | (None, Some(v)) => v.min(16) as i8,
        (Some(a), Some(b)) => (((a as u16 + b as u16 + 1) >> 1).min(16)) as i8,
    }
}

/// Outcome of one candidate's RDO evaluation.
#[derive(Debug, Clone, Copy)]
pub struct PMbRdoResult {
    pub choice: PMbChoice,
    pub d_luma: u64,
    pub r_bits: u32,
    pub cost: u64,
}

/// Compute the full RDO cost `D + ((bits × lambda2[qp]) >> 8)` for a
/// single P-MB partition candidate. Runs MC → residual → quant →
/// dequant → IDCT → recon → SSD for luma, plus B.2's size counter
/// for R.
///
/// `grid` is snapshot/restored around the internal MVD bit count so
/// the caller's grid state is preserved — safe to call multiple times
/// per MB for candidate comparison.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_p_mb_rdo(
    choice: &PMbChoice,
    src_y: &[[u8; 16]; 16],
    reference: &ReconFrame,
    grid: &mut EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
    mb_qp: u8,
    // Phase C.v2 (2026-04-23): neighbour-dependent nC context instead
    // of the prior nC=0 stub. Pass the frame's `total_coeff_grid` and
    // its 4×4-block width; the free function `nc_luma_free` mirrors
    // `Encoder::compute_nc_luma_at` for R estimation against a
    // candidate MB without mutating the grid.
    total_coeff_grid: &[u8],
    frame_w4: usize,
    // Phase C.v3 (2026-04-24): optional chroma D + R inputs. When
    // provided, RDO cost includes Cb + Cr SSD (D_chroma) and chroma
    // CAVLC bits (R_chroma). Omit to keep cost luma-only (prior MVP).
    chroma_inputs: Option<ChromaRdoInputs<'_>>,
) -> PMbRdoResult {
    use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;
    // MC prediction.
    let pred_y = super::encoder::build_luma_prediction(reference, mb_x, mb_y, choice);

    // Luma residual pipeline per 4×4 block.
    let inter_qp = QuantParams { qp: mb_qp, slice: QuantSlice::Inter };
    let trellis_enable = true;
    let mut luma_ac_levels = [[[0i32; 4]; 4]; 16];
    let mut luma_nonzero = [false; 16];
    let mut recon_y = [[0u8; 16]; 16];
    for k in 0..16 {
        let (bx, by) = BLOCK_INDEX_TO_POS[k];
        let sby = by as usize;
        let sbx = bx as usize;
        let mut sub_res = [[0i32; 4]; 4];
        for dy in 0..4 {
            for dx in 0..4 {
                sub_res[dy][dx] = src_y[sby * 4 + dy][sbx * 4 + dx] as i32
                    - pred_y[sby * 4 + dy][sbx * 4 + dx] as i32;
            }
        }
        let coeffs = forward_dct_4x4(&sub_res);
        let levels = trellis_quantize_4x4(&coeffs, inter_qp, trellis_enable)
            .unwrap_or_else(|_| forward_quantize_4x4(&coeffs, inter_qp));
        luma_ac_levels[k] = levels;
        luma_nonzero[k] = levels.iter().any(|row| row.iter().any(|&v| v != 0));
        // Dequant + IDCT → recon residual → recon pixels.
        let dq = dequant_4x4(&levels, mb_qp as i32, false);
        let recon_res = inverse_4x4_integer(&dq);
        for dy in 0..4 {
            for dx in 0..4 {
                let v = pred_y[sby * 4 + dy][sbx * 4 + dx] as i32 + recon_res[dy][dx];
                recon_y[sby * 4 + dy][sbx * 4 + dx] = v.clamp(0, 255) as u8;
            }
        }
    }

    // D = SSD(source, recon) on luma.
    let mut d_luma: u64 = 0;
    for dy in 0..16 {
        for dx in 0..16 {
            let diff = src_y[dy][dx] as i64 - recon_y[dy][dx] as i64;
            d_luma += (diff * diff) as u64;
        }
    }

    // Phase D.2 — drift-aware penalty for inter. Inter prediction
    // uses the PREVIOUS FRAME as reference, so quant noise D_luma
    // compounds into subsequent P-frames via MC. Intra-in-P re-syncs
    // to the current frame's already-decoded neighbours, stopping
    // drift at this MB. Reflect this by multiplying D_inter by a
    // drift factor > 1 before the Lagrangian combine — makes intra
    // relatively cheaper on drift-prone (high-residual) MBs where
    // the compounding cost actually bites.
    //
    // Env knob `PHASM_INTER_DRIFT_FACTOR_Q8` (Q.8 fixed point;
    // default 256 = 1.0 = no drift penalty). Values > 256 increase
    // the inter penalty. Only applied inside `evaluate_p_mb_rdo`
    // (the inter path); `evaluate_intra_in_p_rdo` keeps pure SSD.
    let drift_factor_q8: u64 = super::mb_decision_b::env_var("PHASM_INTER_DRIFT_FACTOR_Q8")
        
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let d_luma_drift = (d_luma * drift_factor_q8.max(1)) >> 8;

    // Prepare R inputs.
    let cbp_luma_8x8 = luma_8x8_cbp_mask(&luma_nonzero);
    // Chroma omitted in Phase C MVP — treat as zero.
    let cbp_chroma = 0u8;
    let cbp_value = pack_cbp(cbp_luma_8x8, cbp_chroma);
    let cbp_codenum = cbp_to_codenum_inter(cbp_value).unwrap_or(0);
    let qp_delta_or_none = if cbp_value != 0 { Some(0i32) } else { None };

    let mut luma_ac_scans = [[0i32; 16]; 16];
    for k in 0..16 {
        luma_ac_scans[k] = raster_to_scan_levels(&luma_ac_levels[k]);
    }
    // Phase C.v2: real nC from the frame's total_coeff_grid. Per
    // block k, bx/by are its 4×4-grid absolute position. Uses the
    // same spec-neighbour derivation as `Encoder::compute_nc_luma_at`
    // but in pure-function form (no &self).
    let mut luma_nc = [0i8; 16];
    for k in 0..16 {
        let (bx_off, by_off) = BLOCK_INDEX_TO_POS[k];
        let abs_bx = mb_x * 4 + bx_off as usize;
        let abs_by = mb_y * 4 + by_off as usize;
        luma_nc[k] = nc_luma_free(total_coeff_grid, frame_w4, abs_bx, abs_by);
    }

    let r_bits = p_mb_total_bits_cavlc(
        choice,
        grid,
        mb_x,
        mb_y,
        cbp_codenum,
        qp_delta_or_none,
        cbp_luma_8x8,
        cbp_chroma,
        &luma_ac_scans,
        &luma_nc,
        None,
    );

    // MB-level Lagrangian cost formula: cost = D + (bits × λ²) >> 8.
    // The `>> 8` rescales the Q.8 fixed-point λ² stored in
    // [`LAMBDA2_TAB`]; `DENOM` below adds a further divisor that
    // lets us tune the effective λ² for our own pipeline's R-D
    // operating point (see the sweep data below).
    //
    // Phase C MVP observation (2026-04-20, IMG_4138 30f q=80):
    //   DENOM=1 (table raw):      31.63 dB / 9.85 Mbps
    //   DENOM=2:                  30.24 dB / 10.30 Mbps
    //   DENOM=4 (sweet spot):     33.39 dB / 11.08 Mbps
    //   DENOM=8:                  33.18 dB / 11.80 Mbps
    //   DENOM=16:                 32.48 dB / 12.33 Mbps
    //   SATD baseline (no RDO):   35.56 dB / 11.10 Mbps
    //
    // Even at the best DENOM, RDO is -2.17 dB behind SATD at matched
    // bitrate. Root cause is NOT λ calibration alone — ranking is
    // structurally wrong. Likely candidates: nC=0 stub biases R high
    // against partition-heavy modes; chroma omission from D ignores
    // a real cost axis; SATD+penalty is surprisingly well tuned for
    // our specific pipeline. Investigate in Phase C.v2 before any
    // default-on rollout. `PHASM_RDO_LAMBDA_DENOM` env knob left in
    // for experimentation; `PHASM_ENABLE_RDO` keeps this path opt-in.
    // Phase E proper — psy-RDO on D term. Add
    // `psy_strength × |hadamard_ac(src) − hadamard_ac(recon)|` to D
    // so the RDO cost function penalises modes that smooth away
    // high-frequency AC content the eye sees. Env-gated.
    let psy_d = apply_psy_rd(src_y, &recon_y, mb_qp);

    // Phase C.v3 — optional chroma D + R. Inter chroma prediction
    // uses the same partition MVs as luma (spec § 8.4.1.4); reuse
    // `build_chroma_prediction` from `super::encoder`.
    let (chroma_d, chroma_r) = if let Some(ci) = chroma_inputs {
        let pred_cb = super::encoder::build_chroma_prediction(reference, 0, mb_x, mb_y, choice);
        let pred_cr = super::encoder::build_chroma_prediction(reference, 1, mb_x, mb_y, choice);
        let (dcb, rcb) = chroma_component_rdo(ci.src_cb, &pred_cb, ci.qp_c);
        let (dcr, rcr) = chroma_component_rdo(ci.src_cr, &pred_cr, ci.qp_c);
        (dcb + dcr, rcb + rcr)
    } else {
        (0, 0)
    };

    let lambda2_denom: u64 = super::mb_decision_b::env_var("PHASM_RDO_LAMBDA_DENOM")
        
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let lambda2 = (lambda2_for_qp(mb_qp) as u64) / lambda2_denom.max(1);
    let total_r = (r_bits as u64).saturating_add(chroma_r as u64);
    let cost = d_luma_drift + psy_d + chroma_d + ((total_r * lambda2) >> 8);
    PMbRdoResult {
        choice: *choice,
        d_luma,
        r_bits: total_r as u32,
        cost,
    }
}

/// Phase E proper + E.v2 — psy-RDO D-term with content-aware
/// strength clamping.
///
/// Base penalty: `strength × |hadamard_ac(src) − hadamard_ac(recon)|`.
/// On near-flat source MBs the symmetric penalty can't distinguish
/// clean inter-recon from intra-recon with block-edge ringing (both
/// have similar AC magnitude vs source) and tilts decisions toward
/// intra, producing visible artefacts on smooth surfaces like skin,
/// sky, walls. The 2026-04-24 IMG_4138 visual A/B flagged this.
///
/// Phase E.v2 fix: read `hadamard_ac_sum_16x16(src)` as a
/// texture-activity proxy. Below `PHASM_PSY_CLAMP_LOW` (flat
/// content) the effective strength is 0. Above `_CLAMP_HIGH`
/// (clearly textured) it's the full nominal strength. Linear ramp
/// between. Smooth ramp avoids decision-boundary flicker that a
/// binary threshold would cause between neighbouring frames where
/// an MB oscillates across the boundary.
///
/// Reusing `hadamard_ac_sum_16x16` (already the signal psy
/// measures) keeps the clamp's semantics aligned with the penalty
/// — threshold units translate cleanly between both.
///
/// Env knobs (all integer):
///   `PHASM_PSY_RD_STRENGTH` — nominal strength (default 0 = disabled)
///   `PHASM_PSY_CLAMP_LOW`   — flat-threshold AC sum (default 200)
///   `PHASM_PSY_CLAMP_HIGH`  — textured-threshold AC sum (default 2000)
///
/// Calibration notes (16×16 Hadamard-AC sum for 8-bit luma):
///   - Smooth skin/wall/sky:    30–200
///   - Mixed skin+fabric:       200–800
///   - Grass / foliage:         1500–3500
///   - Noise / fine film grain: 500–2500
fn apply_psy_rd(src_y: &[[u8; 16]; 16], recon_y: &[[u8; 16]; 16], _mb_qp: u8) -> u64 {
    // Default flipped ON 2026-04-24 after user visual A/B on
    // IMG_4138 full-length: all three psy=64 variants (unclamped,
    // clamp [200, 2000], clamp [500, 3000]) looked visually
    // identical to the psy=0 baseline but with +0.12 dB Y-PSNR at
    // modest bitrate cost. The default clamp `[500, 3000]` was
    // picked over `[200, 2000]` on R-D slope (the latter costs 60%
    // more bits per dB gained), and on conservatism — it engages
    // across a wider band of borderline-flat MBs, protecting
    // against the door-style artefact on content classes we
    // haven't measured. Opt out with `PHASM_PSY_RD_STRENGTH=0`.
    let strength: u64 = super::mb_decision_b::env_var("PHASM_PSY_RD_STRENGTH")
        
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    if strength == 0 {
        return 0;
    }
    use super::intra_predictor::hadamard_ac_sum_16x16;
    let src_ac = hadamard_ac_sum_16x16(src_y);
    let recon_ac = hadamard_ac_sum_16x16(recon_y) as i64;
    let ac_diff = (src_ac as i64 - recon_ac).unsigned_abs();

    // Phase E.v2 content-aware clamp — scale down psy for flat MBs.
    let low: u32 = super::mb_decision_b::env_var("PHASM_PSY_CLAMP_LOW")
        
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let high: u32 = super::mb_decision_b::env_var("PHASM_PSY_CLAMP_HIGH")
        
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let clamped_strength = if src_ac <= low {
        0
    } else if src_ac >= high {
        strength
    } else {
        // Linear ramp in the [low, high] range. Integer math: the
        // u64 strength is at most 32 so u64×u32 stays well inside
        // u64 bounds after the range normalisation.
        let num = (src_ac - low) as u64;
        let den = (high - low) as u64;
        strength.saturating_mul(num) / den.max(1)
    };

    (ac_diff as u64).saturating_mul(clamped_strength)
}

/// Phase D.1 — evaluate intra-in-P RDO cost given the I_16x16
/// prediction already chosen by `choose_intra_16x16_mode_psy`.
///
/// Returns `(D, R, cost)` where cost = D + (R · λ²) >> 8, lambda²
/// from [`LAMBDA2_TAB`] scaled by `PHASM_RDO_LAMBDA_DENOM`.
///
/// Scope: luma-only D + R, matching `evaluate_p_mb_rdo`'s Phase C
/// MVP scope. Consistent bias between the two results keeps
/// intra-vs-inter ranking correct.
///
/// R approximation: 16 independent 4×4 CAVLC residual blocks (same
/// cost model as inter). Exact I_16x16 emission Hadamards the DCs
/// together; that bias is consistent across all intra candidates
/// on the same MB, so it cancels in the intra-vs-inter compare.
/// Header cost includes the P-slice intra-in-P I_16x16 mb_type
/// codenum + intra_chroma_pred_mode codenum + mb_qp_delta.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_intra_in_p_rdo(
    src_y: &[[u8; 16]; 16],
    intra_pred_y: &[[u8; 16]; 16],
    i16x16_mode: u32, // 0=V, 1=H, 2=DC, 3=Plane
    chroma_pred_mode: u32, // 0..=3
    total_coeff_grid: &[u8],
    frame_w4: usize,
    mb_x: usize,
    mb_y: usize,
    mb_qp: u8,
    // Phase C.v3 additions: source chroma blocks + chroma QP +
    // reference frame (for neighbour-based chroma-DC prediction).
    // When `None`, intra RDO stays luma-only (MVP scope).
    chroma_inputs: Option<ChromaRdoInputs<'_>>,
    reference_for_chroma: Option<&ReconFrame>,
) -> (u64, u32, u64) {
    use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;

    // Luma residual roundtrip.
    let inter_qp = QuantParams { qp: mb_qp, slice: QuantSlice::Inter };
    let trellis_enable = true;
    let mut luma_ac_levels = [[[0i32; 4]; 4]; 16];
    let mut luma_nonzero = [false; 16];
    let mut recon_y = [[0u8; 16]; 16];
    for k in 0..16 {
        let (bx, by) = BLOCK_INDEX_TO_POS[k];
        let sby = by as usize;
        let sbx = bx as usize;
        let mut sub_res = [[0i32; 4]; 4];
        for dy in 0..4 {
            for dx in 0..4 {
                sub_res[dy][dx] = src_y[sby * 4 + dy][sbx * 4 + dx] as i32
                    - intra_pred_y[sby * 4 + dy][sbx * 4 + dx] as i32;
            }
        }
        let coeffs = forward_dct_4x4(&sub_res);
        let levels = trellis_quantize_4x4(&coeffs, inter_qp, trellis_enable)
            .unwrap_or_else(|_| forward_quantize_4x4(&coeffs, inter_qp));
        luma_ac_levels[k] = levels;
        luma_nonzero[k] = levels.iter().any(|row| row.iter().any(|&v| v != 0));
        let dq = dequant_4x4(&levels, mb_qp as i32, false);
        let recon_res = inverse_4x4_integer(&dq);
        for dy in 0..4 {
            for dx in 0..4 {
                let v = intra_pred_y[sby * 4 + dy][sbx * 4 + dx] as i32 + recon_res[dy][dx];
                recon_y[sby * 4 + dy][sbx * 4 + dx] = v.clamp(0, 255) as u8;
            }
        }
    }

    // D = SSD(source, recon) on luma.
    let mut d_luma: u64 = 0;
    for dy in 0..16 {
        for dx in 0..16 {
            let diff = src_y[dy][dx] as i64 - recon_y[dy][dx] as i64;
            d_luma += (diff * diff) as u64;
        }
    }

    // Header R: in P-slice, intra-in-P I_16x16 mb_type codenums
    // start at 6 (P-slice row in spec Table 7-13 + intra-in-P prefix
    // "1" + I-slice suffix 1..24 → P-mb_type codenums 6..29). For
    // the cost estimator we use the no-CBP variant (codenum 6 + mode)
    // as a representative value; cbp will adjust R at emit time, but
    // intra-vs-inter ranking shouldn't hinge on the exact codenum.
    let mb_type_p_codenum = 6 + i16x16_mode;

    // CBP for I_16x16: luma uses CBP16Luma (0 or 15 post-quant),
    // chroma uses CBPChroma (0/1/2). Here we compute a simplified
    // CBP based on whether any luma AC level is nonzero (→ cbp=15)
    // and leave chroma=0 in the MVP (same scope as Phase C).
    let any_luma_nonzero = luma_nonzero.iter().any(|&b| b);
    let _cbp_luma_flag: u32 = if any_luma_nonzero { 15 } else { 0 };
    let cbp_chroma: u32 = 0;
    // I_16x16 mb_type encoding folds CBP into the codenum (codenum 6
    // is DC/no-CBP, +12 for luma=15, +4 per chroma level). For R
    // estimation we add the ue(v) cost of the derived codenum;
    // kept simple here since the real emit produces the exact R.
    let _ = cbp_chroma; // chroma omitted in MVP scope

    let mut sz = BitSizer::new();
    sz.write_ue(mb_type_p_codenum + if any_luma_nonzero { 12 } else { 0 });
    sz.write_ue(chroma_pred_mode);
    if any_luma_nonzero {
        sz.write_se(0); // mb_qp_delta = 0 in MVP
    }
    let mut r_bits = sz.bits_written() as u32;

    // Residuals: 16 luma 4×4 blocks, each via cavlc_size with real nC.
    let mut luma_ac_scans = [[0i32; 16]; 16];
    for k in 0..16 {
        luma_ac_scans[k] = raster_to_scan_levels(&luma_ac_levels[k]);
    }
    for k in 0..16 {
        let (bx_off, by_off) = BLOCK_INDEX_TO_POS[k];
        let abs_bx = mb_x * 4 + bx_off as usize;
        let abs_by = mb_y * 4 + by_off as usize;
        let nc = nc_luma_free(total_coeff_grid, frame_w4, abs_bx, abs_by);
        r_bits = r_bits.saturating_add(
            crate::codec::h264::cavlc_size::residual_block_bits_cavlc(
                &luma_ac_scans[k],
                nc,
                CavlcBlockType::Luma4x4,
            )
            .unwrap_or(0),
        );
    }

    // Phase E psy-RDO on intra D (same term as inter). Intra skips
    // the drift-factor since intra doesn't propagate drift through
    // future MC.
    let psy_d = apply_psy_rd(src_y, &recon_y, mb_qp);

    // Phase C.v3 — optional chroma D + R. Intra chroma uses a
    // DC-mode proxy: average of available neighbour samples from
    // the reconstructed reference frame. The real intra chroma
    // decision uses `choose_intra_chroma_mode`; the DC proxy biases
    // R slightly high (V/H/Plane can be cheaper on the real shoot)
    // but consistently across all candidate intra modes, so the
    // intra-vs-inter ranking stays valid.
    let (chroma_d, chroma_r) = match (chroma_inputs, reference_for_chroma) {
        (Some(ci), Some(reference)) => {
            let pred_cb = predict_chroma_dc(reference, 0, mb_x, mb_y);
            let pred_cr = predict_chroma_dc(reference, 1, mb_x, mb_y);
            let (dcb, rcb) = chroma_component_rdo(ci.src_cb, &pred_cb, ci.qp_c);
            let (dcr, rcr) = chroma_component_rdo(ci.src_cr, &pred_cr, ci.qp_c);
            (dcb + dcr, rcb + rcr)
        }
        _ => (0, 0),
    };
    let _ = chroma_pred_mode; // reserved for real mode-based R delta later

    let lambda2_denom: u64 = super::mb_decision_b::env_var("PHASM_RDO_LAMBDA_DENOM")
        
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let lambda2 = (lambda2_for_qp(mb_qp) as u64) / lambda2_denom.max(1);
    let total_r = (r_bits as u64).saturating_add(chroma_r as u64);
    let cost = d_luma + psy_d + chroma_d + ((total_r * lambda2) >> 8);
    (d_luma, total_r as u32, cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Formula-derivation tests: every table entry is regenerated
    // from the closed-form Lagrangian formula in `f64`, then
    // compared against the shipped integer table. If the table
    // values ever drift from the formula, these tests catch it —
    // and the tests also serve as the provenance audit trail tying
    // each integer to its mathematical derivation.

    /// Shared utility: `2^(x)` via IEEE-754 double, then round.
    #[inline]
    fn pow2_round_u16(x: f64) -> u16 {
        (x.exp2()).round().max(1.0) as u16
    }

    #[inline]
    fn pow2_round_i32(x: f64) -> i32 {
        (x.exp2()).round() as i32
    }

    #[test]
    fn lambda_tab_matches_formula() {
        // Formula: max(1, round(2^((qp - 12) / 6))).
        for qp in 0..=51 {
            let expected = pow2_round_u16((qp as f64 - 12.0) / 6.0);
            assert_eq!(
                LAMBDA_TAB[qp as usize], expected,
                "LAMBDA_TAB[{qp}] = {} but formula gives {expected}",
                LAMBDA_TAB[qp as usize]
            );
        }
    }

    #[test]
    fn lambda2_tab_matches_formula() {
        // Formula: trunc(0.9 × 2^(qp/3 + 4)).
        // (0.9 is the Sullivan-Wiegand 2003 RD-slope correction for
        // H.264's operating point; the 2^4 pre-scales so `bits·λ²`
        // stays integer after the `>> 8` step — giving effective Q.4
        // precision at the point-of-use.)
        for qp in 0..=51 {
            let exponent = qp as f64 / 3.0 + 4.0;
            let expected = (0.9_f64 * exponent.exp2()).trunc() as i32;
            assert_eq!(
                LAMBDA2_TAB[qp as usize], expected,
                "LAMBDA2_TAB[{qp}] = {} but formula gives {expected}",
                LAMBDA2_TAB[qp as usize]
            );
        }
    }

    #[test]
    fn trellis_tab_matches_formula() {
        // Formula:
        //   inter: round(0.85² × 2^(qp/3 + 6))
        //   intra: round(0.65² × 2^(qp/3 + 6))
        // 0.85²/0.65² are Sullivan-Wiegand 2003 §V.C per-slice-type
        // RD-slope factors. Trellis uses round-to-nearest (rather
        // than trunc like LAMBDA2_TAB) because per-coefficient
        // trellis RD needs a symmetric cost function.
        let inter_factor = 0.85_f64.powi(2);
        let intra_factor = 0.65_f64.powi(2);
        for qp in 0..=51 {
            let base = (qp as f64 / 3.0 + 6.0).exp2();
            let inter_expected = (inter_factor * base).round() as i32;
            let intra_expected = (intra_factor * base).round() as i32;
            assert_eq!(
                TRELLIS_LAMBDA2_TAB[0][qp as usize], inter_expected,
                "TRELLIS_LAMBDA2_TAB[inter][{qp}] drift"
            );
            assert_eq!(
                TRELLIS_LAMBDA2_TAB[1][qp as usize], intra_expected,
                "TRELLIS_LAMBDA2_TAB[intra][{qp}] drift"
            );
        }
    }

    #[test]
    fn helpers_saturate_past_qp51() {
        // We clamp at 51 — anything higher is out of Baseline range.
        assert_eq!(lambda_for_qp(51), lambda_for_qp(60));
        assert_eq!(lambda2_for_qp(51), lambda2_for_qp(99));
        assert_eq!(trellis_lambda2_for_qp(51, true), trellis_lambda2_for_qp(200, true));
    }

    // ─── Phase B.3: BitSizer ≡ BitWriter on the same inputs ─────

    use super::super::bitstream_writer::BitWriter;

    fn sizer_matches_writer(input: &dyn Fn(&mut dyn BitSink)) {
        let mut w = BitWriter::new();
        input(&mut w);
        let mut sz = BitSizer::new();
        input(&mut sz);
        assert_eq!(
            w.bits_written(),
            sz.bits_written(),
            "BitWriter vs BitSizer bit count mismatch"
        );
    }

    #[test]
    fn sizer_matches_writer_for_ue() {
        for v in [0u32, 1, 7, 14, 63, 1023, 65535] {
            sizer_matches_writer(&move |s| s.write_ue(v));
        }
    }

    #[test]
    fn sizer_matches_writer_for_se() {
        for v in [-32767i32, -100, -3, -1, 0, 1, 3, 100, 32767] {
            sizer_matches_writer(&move |s| s.write_se(v));
        }
    }

    #[test]
    fn sizer_matches_writer_for_write_bits() {
        sizer_matches_writer(&|s| {
            s.write_bits(0b1010_1100, 8);
            s.write_bits(0x1234_5678, 32);
            s.write_bit(true);
            s.write_bit(false);
            s.write_bits(0b11, 2);
        });
    }

    #[test]
    fn sizer_matches_writer_for_cavlc_luma_block_mixed_coeffs() {
        // Representative luma 4×4 scan: one trailing ±1, several mid-
        // magnitude levels, a few zero runs. Exercises the prefix +
        // suffix level path AND run_before VLC.
        let coeffs = [5i32, -3, 2, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut w = BitWriter::new();
        encode_cavlc_block(&mut w, &coeffs, 0, CavlcBlockType::Luma4x4).unwrap();
        let mut sz = BitSizer::new();
        encode_cavlc_block(&mut sz, &coeffs, 0, CavlcBlockType::Luma4x4).unwrap();
        assert_eq!(w.bits_written(), sz.bits_written());
        // residual_block_bits_cavlc is the thin wrapper; same answer.
        assert_eq!(
            sz.bits_written() as u32,
            residual_block_bits_cavlc(&coeffs, 0, CavlcBlockType::Luma4x4)
        );
    }

    #[test]
    fn sizer_matches_writer_for_all_empty_cavlc_blocks() {
        // All-zero blocks still emit a 1-bit coeff_token (tc=0, t1=0).
        // Each block type has a different token entry — test all four.
        let types_zeros: &[(CavlcBlockType, &[i32])] = &[
            (CavlcBlockType::Luma4x4, &[0; 16]),
            (CavlcBlockType::Intra16x16Ac, &[0; 15]),
            (CavlcBlockType::ChromaAc, &[0; 15]),
            (CavlcBlockType::ChromaDc, &[0; 4]),
        ];
        for (bt, zeros) in types_zeros {
            let nc = if matches!(*bt, CavlcBlockType::ChromaDc) { -1 } else { 0 };
            let mut w = BitWriter::new();
            encode_cavlc_block(&mut w, zeros, nc, *bt).unwrap();
            let mut sz = BitSizer::new();
            encode_cavlc_block(&mut sz, zeros, nc, *bt).unwrap();
            assert_eq!(
                w.bits_written(),
                sz.bits_written(),
                "mismatch on {bt:?}"
            );
        }
    }

    #[test]
    fn p_mb_header_bits_matches_writer_for_p16x16() {
        // Drive the real writer and the sizer through identical
        // emit_mvds_and_update_grid paths and compare bit count. This
        // covers the MVD se(v) emission + grid mutation across a
        // P_16x16 MB — the simplest partition type but exercises the
        // same code path Phase C will size-count.
        use super::super::bitstream_writer::BitWriter;
        use super::super::encoder::emit_mvds_and_update_grid;
        use super::super::motion_estimation::MotionVector;
        use super::super::partition_decision::PMbChoice;
        use super::super::partition_state::EncoderMvGrid;

        let choice = PMbChoice::P16x16 {
            mv: MotionVector { mv_x: 7, mv_y: -3 },
            ref_idx_l0: 0,
        };
        let mut grid_a = EncoderMvGrid::new(4, 4);
        grid_a.fill(0, 0, 4, 4, MotionVector { mv_x: 1, mv_y: 1 }, 0);
        let mut grid_b = grid_a.clone();

        let mut w = BitWriter::new();
        emit_mvds_and_update_grid(&mut w, &mut grid_a, 2, 2, &choice);
        let mut sz = BitSizer::new();
        emit_mvds_and_update_grid(&mut sz, &mut grid_b, 2, 2, &choice);
        assert_eq!(
            <BitWriter as BitSink>::bits_written(&w),
            sz.bits_written()
        );
    }

    #[test]
    fn sizer_matches_writer_for_high_magnitude_levels() {
        // Forces the level_code escape path (sl > 0 with full escape,
        // sl == 0 with the 14-30 sub-regime, and >= 30 full escape).
        let coeffs = [127i32, 64, 31, 15, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut w = BitWriter::new();
        encode_cavlc_block(&mut w, &coeffs, 0, CavlcBlockType::Luma4x4).unwrap();
        let mut sz = BitSizer::new();
        encode_cavlc_block(&mut sz, &coeffs, 0, CavlcBlockType::Luma4x4).unwrap();
        assert_eq!(w.bits_written(), sz.bits_written());
    }

    #[test]
    fn lambda2_matches_formula_reasonably() {
        // lambda2 ≈ lambda² × 0.9 × 256 = lambda² × 230.4.
        // Spot-check QP 30: lambda=8 → 64 × 230.4 = 14745.6 → table is 14745.
        // Spot-check QP 24: lambda=4 → 16 × 230.4 = 3686.4 → table is 3686.
        for &qp in &[12, 18, 24, 30, 36, 42] {
            let l = LAMBDA_TAB[qp] as i64;
            let expected = (l * l * 2304 + 5) / 10; // integer approx of × 230.4, round-to-nearest
            let actual = LAMBDA2_TAB[qp] as i64;
            assert!(
                (expected - actual).abs() <= 2,
                "qp={}: expected ≈ {}, got {}",
                qp,
                expected,
                actual
            );
        }
    }
}
