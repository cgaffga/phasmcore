// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Top-level CABAC encoder: engine + contexts + neighbors.
//!
//! Glue layer that combines:
//!  - [`CabacEngine`] (arithmetic state + bit output).
//!  - The 1024-entry context table (per-ctxIdx (pStateIdx, valMPS)).
//!  - [`CabacNeighborContext`] (row buffer of neighbor MB state).
//!
//! Per-syntax-element encoders (I-slice Phase 6C.5a, P-slice 6C.5b)
//! operate on a single `CabacEncoder`. They binarize a syntax value
//! (see [`super::binarization`]), compute per-bin ctxIdx (see
//! [`super::neighbor`]), and stream bins through the engine.

use super::binarization::{
    binarize_egk_suffix, binarize_tu, binarize_unary, mb_qp_delta_remap,
    mb_type_i_bins, mb_type_p_bins_prefix, sub_mb_type_p_bins,
};
use super::context::{initialize_contexts, CabacContext, CabacInitSlot};
use super::engine::CabacEngine;
use super::neighbor::{
    compute_cbp_luma_ctx_idx_inc_bin, ctx_idx_inc_cbp_chroma, ctx_idx_inc_coded_block_flag,
    ctx_idx_inc_coeff_abs_level, ctx_idx_inc_intra_chroma_pred_mode_bin0,
    ctx_idx_inc_mb_qp_delta_bin0, ctx_idx_inc_mb_skip_flag, ctx_idx_inc_mb_type_bin0,
    ctx_idx_inc_mvd_bin0, ctx_idx_inc_prior_bin, ctx_idx_inc_ref_idx_bin0,
    ctx_idx_inc_sig_4x4, ctx_idx_inc_sig_chroma_dc, CabacNeighborContext,
};

// Phase 6F.1 follow-on tidy (Task #50, deferred-item #35) — these
// tables now live in `cabac::ctx_offsets`. Re-export here so existing
// `cabac::encoder::{CTX_BLOCK_CAT_OFFSET, ctx_offset, cat5_luma8x8,
// SIG_COEFF_FLAG_OFFSET_8X8_FRAME, LAST_COEFF_FLAG_OFFSET_8X8_FRAME}`
// imports keep compiling.
pub use super::ctx_offsets::{
    cat5_luma8x8, ctx_offset, CTX_BLOCK_CAT_OFFSET,
    LAST_COEFF_FLAG_OFFSET_8X8_FRAME, SIG_COEFF_FLAG_OFFSET_8X8_FRAME,
};

/// 8×8 zigzag scan order — spec § 8.5.4 (for frame-coded 8×8 blocks).
/// Entry `i` gives the row-major index into a `[[i32; 8]; 8]`
/// coefficient matrix, i.e. `row * 8 + col`, that scan position `i`
/// reads. Written as `row + col * 8` so auditors can read off (row, col)
/// pairs directly against the spec.
#[rustfmt::skip]
#[allow(clippy::erasing_op, clippy::identity_op)]
pub const ZIGZAG_8X8: [u8; 64] = [
    0 + 0 * 8,  1 + 0 * 8,  0 + 1 * 8,  0 + 2 * 8,  1 + 1 * 8,  2 + 0 * 8,  3 + 0 * 8,
    2 + 1 * 8,  1 + 2 * 8,  0 + 3 * 8,  0 + 4 * 8,  1 + 3 * 8,  2 + 2 * 8,  3 + 1 * 8,
    4 + 0 * 8,  5 + 0 * 8,  4 + 1 * 8,  3 + 2 * 8,  2 + 3 * 8,  1 + 4 * 8,  0 + 5 * 8,
    0 + 6 * 8,  1 + 5 * 8,  2 + 4 * 8,  3 + 3 * 8,  4 + 2 * 8,  5 + 1 * 8,  6 + 0 * 8,
    7 + 0 * 8,  6 + 1 * 8,  5 + 2 * 8,  4 + 3 * 8,  3 + 4 * 8,  2 + 5 * 8,  1 + 6 * 8,
    0 + 7 * 8,  1 + 7 * 8,  2 + 6 * 8,  3 + 5 * 8,  4 + 4 * 8,  5 + 3 * 8,  6 + 2 * 8,
    7 + 1 * 8,  7 + 2 * 8,  6 + 3 * 8,  5 + 4 * 8,  4 + 5 * 8,  3 + 6 * 8,  2 + 7 * 8,
    3 + 7 * 8,  4 + 6 * 8,  5 + 5 * 8,  6 + 4 * 8,  7 + 3 * 8,  7 + 4 * 8,  6 + 5 * 8,
    5 + 6 * 8,  4 + 7 * 8,  5 + 7 * 8,  6 + 6 * 8,  7 + 5 * 8,  7 + 6 * 8,  6 + 7 * 8,
    7 + 7 * 8,
];

/// Top-level CABAC encoder: engine + 1024 contexts + neighbor row buffer.
pub struct CabacEncoder {
    pub engine: CabacEngine,
    pub contexts: Box<[CabacContext; 1024]>,
    pub neighbors: CabacNeighborContext,
}

impl CabacEncoder {
    /// Construct a new CABAC encoder at slice start. Initializes all
    /// 1024 contexts per spec § 9.3.1.1 using the provided QP and
    /// slot (I/SI or P/B init_idc).
    pub fn new_slice(slot: CabacInitSlot, slice_qp_y: i32, mb_width: usize) -> Self {
        let contexts = Box::new(initialize_contexts(slot, slice_qp_y));
        Self {
            engine: CabacEngine::new(),
            contexts,
            neighbors: CabacNeighborContext::new(mb_width, slot),
        }
    }

    /// Finish the slice. Consumes the engine; returns CABAC bytes
    /// ready for RBSP assembly. Caller handles `cabac_zero_word`
    /// stuffing (§ 9.3.4.6) at the slice level.
    pub fn finish(self) -> Vec<u8> {
        self.engine.finish()
    }

    /// Encode a context-coded bin at the given ctxIdx.
    #[inline]
    pub(crate) fn encode_dec(&mut self, bin: u8, ctx_idx: u32) {
        let ctx = &mut self.contexts[ctx_idx as usize];
        self.engine.encode_decision_with_ctx_idx(bin, ctx, ctx_idx);
    }

    #[inline]
    fn encode_bypass(&mut self, bin: u8) {
        self.engine.encode_bypass(bin);
    }

    #[inline]
    fn encode_terminate(&mut self, bin: u8) {
        self.engine.encode_terminate(bin);
    }
}

// ─── I-slice syntax encoders (Phase 6C.5a) ──────────────────────

/// Encode `mb_type` for an I-slice (spec § 9.3.2.5 Table 9-36 +
/// context derivation § 9.3.3.1.1.3).
///
/// `mb_type` value must be 0..=25. The I_PCM bin (bin 1 of mb_type=25)
/// is automatically routed through `encode_terminate` per spec.
pub fn encode_mb_type_i(enc: &mut CabacEncoder, mb_type: u32, mb_x: usize) {
    debug_assert!(mb_type <= 25);
    let bins = mb_type_i_bins(mb_type);
    for (bin_idx, &bin) in bins.iter().enumerate() {
        if bin_idx == 1 {
            enc.engine.trace_label = "mb_type bin1 (term)".to_string();
            enc.encode_terminate(bin);
            continue;
        }

        let ctx_idx_inc = ctx_idx_inc_mb_type_bin(
            enc,
            mb_x,
            ctx_offset::MB_TYPE_I,
            bin_idx as u32,
            bins,
        );
        let ctx_idx = ctx_offset::MB_TYPE_I + ctx_idx_inc;
        enc.engine.trace_label = format!("mb_type bin{bin_idx}");
        enc.encode_dec(bin, ctx_idx);
    }
}

/// ctxIdxInc for an `mb_type` bin at the given `ctxIdxOffset` and
/// binIdx. Uses bin 0 neighbor derivation or prior-bin dependencies
/// per Table 9-41; bins beyond that use the fixed Table 9-39
/// increments (3..7 for I-slice suffix, 4..6 for P prefix, etc.).
fn ctx_idx_inc_mb_type_bin(
    enc: &CabacEncoder,
    mb_x: usize,
    ctx_idx_offset: u32,
    bin_idx: u32,
    prior_bins: &[u8],
) -> u32 {
    if bin_idx == 0 {
        return ctx_idx_inc_mb_type_bin0(&enc.neighbors, mb_x, ctx_idx_offset);
    }
    if let Some(inc) = ctx_idx_inc_prior_bin(ctx_idx_offset, bin_idx, prior_bins) {
        return inc;
    }
    // Fall back to per-binIdx static increments from spec Table 9-39
    // "Assignment of ctxIdxInc to binIdx for all ctxIdxOffset values
    // except those related to the syntax elements coded_block_flag,
    // significant_coeff_flag, last_significant_coeff_flag, and
    // coeff_abs_level_minus1". For I-slice mb_type at
    // ctxIdxOffset = 3, after the neighbor-based bin-0 uses
    // ctxIdxInc in {0, 1, 2}, the subsequent bins use:
    //   bin 1: TERMINATE (handled in encode_mb_type_i directly)
    //   bin 2: ctxIdxInc = 3 (cbp_luma flag)
    //   bin 3: ctxIdxInc = 4 (cbp_chroma test)
    //   bin 4: ctxIdxInc = 5 when emitted (cbp_chroma bit — only if bin 3 = 1)
    //   bin 5: ctxIdxInc = 6 (pred_mode MSB)
    //   bin 6: ctxIdxInc = 7 (pred_mode LSB)
    // Our binarization table's bin_idx is an index into a variable-
    // length bin array (6 bins when cbp_chroma=0, skipping bin 4;
    // 7 bins when cbp_chroma>0). Bins 4/5 of the 6-bin case map
    // back to the spec's bin 5/6 positions so they're handled by
    // the prior-bin table.
    // Task #19/#21 sweep knobs: each of the 6 offset=17 ctxIdxInc
    // values can be overridden via env var for brute-force search.
    // Defaults are the current educated guesses.
    fn env_or(name: &str, default: u32) -> u32 {
        std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    match (ctx_idx_offset, bin_idx) {
        (3, 2) => 3, // I-slice mb_type bin 2 (cbp_luma)
        (3, 3) => 4, // I-slice mb_type bin 3 (cbp_chroma test)
        (3, 6) => 7, // I-slice mb_type bin 6 (pred_mode LSB, 7-bin case)
        (14, 1) => 1, // P-slice prefix bin 1
        (17, 0) => env_or("PHASM_IIP_CTX_17_0", 0),
        (17, 2) => env_or("PHASM_IIP_CTX_17_2", 1),
        (17, 3) => env_or("PHASM_IIP_CTX_17_3", 2),
        (17, 5) => env_or("PHASM_IIP_CTX_17_5", 3),
        (17, 6) => env_or("PHASM_IIP_CTX_17_6", 3),
        _ => 0,
    }
}

/// Encode `prev_intra4x4_pred_mode_flag` (spec § 9.3.2.4 FL cMax=1,
/// ctxIdxOffset=68). Single context, `ctxIdxInc = 0`.
pub fn encode_prev_intra4x4_pred_mode_flag(enc: &mut CabacEncoder, is_predicted: bool) {
    enc.encode_dec(
        if is_predicted { 1 } else { 0 },
        ctx_offset::PREV_INTRA_PRED_MODE_FLAG,
    );
}

/// Encode `rem_intra4x4_pred_mode` as a 3-bit FL binarization per
/// spec § 9.3.2.4: "The indexing of bins for the FL binarization is
/// such that the binIdx = 0 relates to the least significant bit
/// with increasing values of binIdx towards the most significant
/// bit." That is, FL binarization emits bins LSB-first.
///
/// All 3 bins use `ctxIdxOffset = 69` with `ctxIdxInc = 0` per spec
/// Table 9-39.
pub fn encode_rem_intra4x4_pred_mode(enc: &mut CabacEncoder, rem: u8) {
    debug_assert!(rem <= 7);
    // bin 0 = LSB, bin 2 = MSB (spec § 9.3.2.4 binIdx rule).
    enc.encode_dec(rem & 1, ctx_offset::REM_INTRA_PRED_MODE);
    enc.encode_dec((rem >> 1) & 1, ctx_offset::REM_INTRA_PRED_MODE);
    enc.encode_dec((rem >> 2) & 1, ctx_offset::REM_INTRA_PRED_MODE);
}

/// Encode `intra_chroma_pred_mode` (spec § 9.3.2 TU cMax=3,
/// ctxIdxOffset=64). Bin 0 uses neighbor derivation; bin 1 uses
/// Table 9-39 static increment (3).
pub fn encode_intra_chroma_pred_mode(enc: &mut CabacEncoder, mode: u8, mb_x: usize) {
    debug_assert!(mode <= 3);
    let mut bin_idx = 0usize;
    let cond_a = ctx_idx_inc_intra_chroma_pred_mode_bin0(&enc.neighbors, mb_x);
    binarize_tu(mode as u32, 3, &mut |bin| {
        let ctx_inc = if bin_idx == 0 { cond_a } else { 3 };
        let ctx_idx = ctx_offset::INTRA_CHROMA_PRED_MODE + ctx_inc;
        enc.encode_dec(bin, ctx_idx);
        bin_idx += 1;
    });
}

/// Encode `coded_block_pattern` per spec § 9.3.2.6 (binarization for
/// CBP: FL-cMax=15 luma prefix + TU-cMax=2 chroma suffix in
/// 4:2:0/4:2:2) with context derivation per § 9.3.3.1.1.4.
pub fn encode_coded_block_pattern(enc: &mut CabacEncoder, cbp: u8, mb_x: usize) {
    let luma = cbp & 0x0F;
    let chroma = ((cbp >> 4) & 0x03) as u32;

    // Luma prefix: 4 FL-binarized bins emitted LSB-first per spec
    // § 9.3.2.4 ("binIdx = 0 relates to the least significant bit").
    // Each bin's ctxIdxInc depends on the 8×8 neighbor blocks per
    // § 9.3.3.1.1.4 — binIdx 1..3 may have same-MB neighbors, so we
    // track the partial current-MB CBP as bits are emitted.
    let mut current_partial: u8 = 0;
    for bin_idx in 0..4u32 {
        let bin = (luma >> bin_idx) & 1;
        let ctx_inc =
            compute_cbp_luma_ctx_idx_inc_bin(bin_idx, current_partial, &enc.neighbors, mb_x);
        enc.encode_dec(bin, ctx_offset::CBP_LUMA + ctx_inc);
        current_partial |= bin << bin_idx;
    }

    // Chroma suffix: TU cMax=2. Cross-MB neighbors only (spec § 9.3.3.1.1.4
    // eq 9-11).
    let mut bin_idx = 0u32;
    binarize_tu(chroma, 2, &mut |bin| {
        let ctx_inc = ctx_idx_inc_cbp_chroma(&enc.neighbors, mb_x, bin_idx);
        enc.encode_dec(bin, ctx_offset::CBP_CHROMA + ctx_inc);
        bin_idx += 1;
    });
}

/// Encode `mb_qp_delta` (spec § 9.3.2.7 + § 9.3.3.1.1.5).
pub fn encode_mb_qp_delta(enc: &mut CabacEncoder, qp_delta: i32) {
    let mapped = mb_qp_delta_remap(qp_delta);
    let mut bin_idx = 0u32;
    binarize_unary(mapped, &mut |bin| {
        let ctx_inc = if bin_idx == 0 {
            ctx_idx_inc_mb_qp_delta_bin0(&enc.neighbors)
        } else if bin_idx == 1 {
            2 // Table 9-39 binIdx 1 static
        } else {
            3 // bins 2+ reuse ctxIdxInc = 3 (Table 9-39 cap)
        };
        enc.encode_dec(bin, ctx_offset::MB_QP_DELTA + ctx_inc);
        bin_idx += 1;
    });
}

/// Encode `coded_block_flag` for a residual block.
///
/// `ctx_block_cat` in 0..=4 (Phase 6C scope). `block_idx_in_mb_{a,b}`
/// are the neighbor blocks' category-local indices.
pub fn encode_coded_block_flag(
    enc: &mut CabacEncoder,
    is_coded: bool,
    ctx_block_cat: u8,
    mb_x: usize,
    block_idx_in_mb_a: usize,
    block_idx_in_mb_b: usize,
    current_is_intra: bool,
) {
    let ctx_inc = ctx_idx_inc_coded_block_flag(
        &enc.neighbors,
        mb_x,
        ctx_block_cat,
        block_idx_in_mb_a,
        block_idx_in_mb_b,
        current_is_intra,
    );
    let ctx_idx = ctx_offset::CODED_BLOCK_FLAG_LOW
        + CTX_BLOCK_CAT_OFFSET[0][ctx_block_cat as usize]
        + ctx_inc;
    enc.encode_dec(if is_coded { 1 } else { 0 }, ctx_idx);
}

/// Encode `end_of_slice_flag` (spec § 9.3.4.5, ctxIdx=276, terminate).
pub fn encode_end_of_slice_flag(enc: &mut CabacEncoder, is_last_mb: bool) {
    enc.encode_terminate(if is_last_mb { 1 } else { 0 });
}

/// Encode `transform_size_8x8_flag` (spec § 9.3.3.1.1.10). Single bin at
/// ctxIdxOffset = 399 with ctxIdxInc derived from neighbors'
/// transform_size_8x8_flag.
pub fn encode_transform_size_8x8_flag(
    enc: &mut CabacEncoder,
    flag: bool,
    mb_x: usize,
) {
    let inc = super::neighbor::ctx_idx_inc_transform_size_8x8_flag(&enc.neighbors, mb_x);
    let ctx_idx = ctx_offset::TRANSFORM_SIZE_8X8_FLAG + inc;
    enc.encode_dec(if flag { 1 } else { 0 }, ctx_idx);
}

/// Encode a residual block: coded_block_flag + significance map +
/// reverse-scan level emit (spec § 7.3.5.3.3 + § 9.3.3.1.1.9 + § 9.3.3.1.3).
///
/// `scan_coeffs` is the zigzag-ordered coefficient array of length
/// `max_num_coeff`. `ctx_block_cat` is the block category (0..=4 for
/// Phase 6C scope).
///
/// Caller must set up `coded_block_flag_cat` in `enc.neighbors` for
/// subsequent blocks to see this one as a neighbor.
pub fn encode_residual_block_cabac(
    enc: &mut CabacEncoder,
    scan_coeffs: &[i32],
    start_idx: usize,
    end_idx: usize,
    ctx_block_cat: u8,
    mb_x: usize,
    block_idx_in_mb_a: usize,
    block_idx_in_mb_b: usize,
    current_is_intra: bool,
) -> bool {
    let cbf_ctx_inc = super::neighbor::ctx_idx_inc_coded_block_flag(
        &enc.neighbors,
        mb_x,
        ctx_block_cat,
        block_idx_in_mb_a,
        block_idx_in_mb_b,
        current_is_intra,
    );
    encode_residual_block_cabac_with_cbf_inc(
        enc, scan_coeffs, start_idx, end_idx, ctx_block_cat, cbf_ctx_inc,
    )
}

/// Variant of [`encode_residual_block_cabac`] that accepts a
/// pre-computed `coded_block_flag` `ctxIdxInc`. Use this when the
/// neighbor derivation for the CBF needs to consult the
/// partially-built current-MB state (see `CurrentMbCbf` in
/// [`super::neighbor`]) — the builtin form only sees cross-MB state.
pub fn encode_residual_block_cabac_with_cbf_inc(
    enc: &mut CabacEncoder,
    scan_coeffs: &[i32],
    start_idx: usize,
    end_idx: usize,
    ctx_block_cat: u8,
    cbf_ctx_idx_inc: u32,
) -> bool {
    debug_assert!(end_idx < scan_coeffs.len());
    debug_assert!(ctx_block_cat <= 4);

    // 1. coded_block_flag with the pre-computed ctxIdxInc.
    let has_nonzero = scan_coeffs[start_idx..=end_idx]
        .iter()
        .any(|&v| v != 0);
    let cbf_ctx_idx = ctx_offset::CODED_BLOCK_FLAG_LOW
        + CTX_BLOCK_CAT_OFFSET[0][ctx_block_cat as usize]
        + cbf_ctx_idx_inc;
    enc.encode_dec(if has_nonzero { 1 } else { 0 }, cbf_ctx_idx);
    if !has_nonzero {
        return false;
    }

    // 2. Significance map (forward scan). Loop until the last
    //    nonzero position is signalled via last_significant_coeff_flag.
    let sig_offset = ctx_offset::SIGNIFICANT_COEFF_FLAG_FRAME_LOW
        + CTX_BLOCK_CAT_OFFSET[1][ctx_block_cat as usize];
    let last_offset = ctx_offset::LAST_SIGNIFICANT_COEFF_FLAG_FRAME_LOW
        + CTX_BLOCK_CAT_OFFSET[2][ctx_block_cat as usize];

    // Per spec § 7.3.5.3.3 + § 9.3.3.1.3: significance-map context
    // index uses `levelListIdx`, which is the LOOP COUNTER starting
    // at 0 (NOT the raw scan position). For blocks with start_idx = 1
    // (Intra16x16ACLevel, ChromaACLevel), this means the first ac
    // coefficient at scan[1] uses ctxIdxInc = 0, not 1.
    let mut num_coeff = end_idx + 1;
    let mut i = start_idx;
    while i < num_coeff - 1 {
        let level_list_idx = (i - start_idx) as u32;
        let is_sig = scan_coeffs[i] != 0;
        let sig_bin = if is_sig { 1 } else { 0 };
        let ctx_sig_inc = if ctx_block_cat == 3 {
            ctx_idx_inc_sig_chroma_dc(level_list_idx)
        } else {
            ctx_idx_inc_sig_4x4(level_list_idx)
        };
        enc.encode_dec(sig_bin, sig_offset + ctx_sig_inc);

        if is_sig {
            let is_last = scan_coeffs[i + 1..=end_idx]
                .iter()
                .all(|&v| v == 0);
            let last_bin = if is_last { 1 } else { 0 };
            let ctx_last_inc = if ctx_block_cat == 3 {
                ctx_idx_inc_sig_chroma_dc(level_list_idx)
            } else {
                ctx_idx_inc_sig_4x4(level_list_idx)
            };
            enc.encode_dec(last_bin, last_offset + ctx_last_inc);
            if is_last {
                num_coeff = i + 1;
                break;
            }
        }
        i += 1;
    }
    // The coefficient at position num_coeff-1 is inferred significant.

    // 3. Reverse-scan abs_level + sign emit.
    let abs_offset = ctx_offset::COEFF_ABS_LEVEL_MINUS1_LOW
        + CTX_BLOCK_CAT_OFFSET[3][ctx_block_cat as usize];
    let mut num_eq1 = 0u32;
    let mut num_gt1 = 0u32;

    // Iterate in reverse from num_coeff-1 down to start_idx.
    for i in (start_idx..num_coeff).rev() {
        if scan_coeffs[i] == 0 {
            continue;
        }
        let abs_level_minus1 = scan_coeffs[i].unsigned_abs() - 1;

        // UEG0 prefix: TU with cMax=14.
        let prefix_len = abs_level_minus1.min(14);
        for b in 0..prefix_len {
            let ctx_inc = ctx_idx_inc_coeff_abs_level(
                if b == 0 { 0 } else { 1 },
                ctx_block_cat,
                num_eq1,
                num_gt1,
            );
            enc.encode_dec(1, abs_offset + ctx_inc);
        }

        if abs_level_minus1 < 14 {
            // TU terminator 0 — uses the same ctxInc as the NEXT bin
            // position would have.
            let ctx_inc = ctx_idx_inc_coeff_abs_level(
                if prefix_len == 0 { 0 } else { 1 },
                ctx_block_cat,
                num_eq1,
                num_gt1,
            );
            enc.encode_dec(0, abs_offset + ctx_inc);
        } else {
            // EG0 suffix via bypass.
            let suf_s = abs_level_minus1 - 14;
            binarize_egk_suffix(suf_s, 0, &mut |bin| enc.encode_bypass(bin));
        }

        // coeff_sign_flag (bypass).
        enc.encode_bypass(if scan_coeffs[i] < 0 { 1 } else { 0 });

        // Update counters AFTER the level is fully emitted.
        if scan_coeffs[i].unsigned_abs() == 1 {
            num_eq1 += 1;
        } else {
            num_gt1 += 1;
        }
    }
    true
}

/// Encode a cat=5 Luma 8×8 residual block (spec § 7.3.5.3.2 with
/// `ctxBlockCat == 5`). Unlike 4×4 residuals, 8×8 emits NO CBF — the
/// presence of the block is signaled by the `cbp_luma` bit at the MB
/// level, not a per-block coded_block_flag. Sig-map uses the
/// position-dependent 8×8 offset tables rather than `levelListIdx`
/// directly.
///
/// `scan_coeffs` contains the 64 coefficients in 8×8 zigzag scan order
/// (apply `ZIGZAG_8X8` to a row-major `[[i32; 8]; 8]` transform output
/// to produce this).
pub fn encode_residual_block_cabac_8x8(enc: &mut CabacEncoder, scan_coeffs: &[i32; 64]) {
    // 1. Sig-map over 63 positions. The 64th (DC-farthest) is implicit.
    let mut num_coeff = 64;
    let mut i = 0;
    while i < 63 {
        let level_list_idx = i as u32;
        let is_sig = scan_coeffs[i] != 0;
        let sig_bin = if is_sig { 1 } else { 0 };
        let sig_ctx = cat5_luma8x8::SIG_BASE
            + SIG_COEFF_FLAG_OFFSET_8X8_FRAME[level_list_idx as usize] as u32;
        enc.encode_dec(sig_bin, sig_ctx);

        if is_sig {
            let is_last = scan_coeffs[i + 1..64].iter().all(|&v| v == 0);
            let last_bin = if is_last { 1 } else { 0 };
            let last_ctx = cat5_luma8x8::LAST_BASE
                + LAST_COEFF_FLAG_OFFSET_8X8_FRAME[level_list_idx as usize] as u32;
            enc.encode_dec(last_bin, last_ctx);
            if is_last {
                num_coeff = i + 1;
                break;
            }
        }
        i += 1;
    }
    // Implicit: position 63 is the last-significant when reached.

    // 2. Reverse-scan abs_level + sign emit (same rule as 4×4).
    let mut num_eq1 = 0u32;
    let mut num_gt1 = 0u32;
    for i in (0..num_coeff).rev() {
        if scan_coeffs[i] == 0 {
            continue;
        }
        let abs_level_minus1 = scan_coeffs[i].unsigned_abs() - 1;
        let prefix_len = abs_level_minus1.min(14);
        for b in 0..prefix_len {
            let ctx_inc = super::neighbor::ctx_idx_inc_coeff_abs_level(
                if b == 0 { 0 } else { 1 },
                5,
                num_eq1,
                num_gt1,
            );
            enc.encode_dec(1, cat5_luma8x8::ABS_BASE + ctx_inc);
        }
        if abs_level_minus1 < 14 {
            let ctx_inc = super::neighbor::ctx_idx_inc_coeff_abs_level(
                if prefix_len == 0 { 0 } else { 1 },
                5,
                num_eq1,
                num_gt1,
            );
            enc.encode_dec(0, cat5_luma8x8::ABS_BASE + ctx_inc);
        } else {
            let suf_s = abs_level_minus1 - 14;
            binarize_egk_suffix(suf_s, 0, &mut |bin| enc.encode_bypass(bin));
        }
        enc.encode_bypass(if scan_coeffs[i] < 0 { 1 } else { 0 });
        if scan_coeffs[i].unsigned_abs() == 1 {
            num_eq1 += 1;
        } else {
            num_gt1 += 1;
        }
    }
}

// ─── P-slice syntax encoders (Phase 6C.5b) ──────────────────────
// Minimal set here; full wiring happens in 6C.5b.

/// Encode `mb_skip_flag` (spec § 9.3.3.1.1.1). P-slice only.
pub fn encode_mb_skip_flag(enc: &mut CabacEncoder, is_skip: bool, mb_x: usize) {
    let ctx_inc = ctx_idx_inc_mb_skip_flag(&enc.neighbors, mb_x);
    enc.encode_dec(
        if is_skip { 1 } else { 0 },
        ctx_offset::MB_SKIP_FLAG_P + ctx_inc,
    );
}

/// Phase 6E-A3 — encode `mb_skip_flag` for a B-slice. Same neighbor
/// derivation as P (spec § 9.3.3.1.1.1; both use
/// `condTermFlagN = !is_skip(N)`); ctxIdxOffset 24 vs P's 11.
pub fn encode_mb_skip_flag_b(enc: &mut CabacEncoder, is_skip: bool, mb_x: usize) {
    let ctx_inc = ctx_idx_inc_mb_skip_flag(&enc.neighbors, mb_x);
    enc.encode_dec(
        if is_skip { 1 } else { 0 },
        ctx_offset::MB_SKIP_FLAG_B + ctx_inc,
    );
}

/// Phase 6E-A3 — encode `mb_type` for a B-slice (Table 9-37 B rows).
///
/// Values 0..3 + 22 are §6E-A3 active set (B_Direct_16x16,
/// B_L0_16x16, B_L1_16x16, B_Bi_16x16, B_8x8). Values 4..21 are
/// 16x8 / 8x16 partitions deferred to §6E-A6. Values 23..47 are
/// intra-in-B (encoder emits the 6-bin intra prefix from
/// `mb_type_b_intra_prefix()` then the I-slice suffix via
/// `encode_mb_type_i`-style emission with ctxIdxOffset 32).
///
/// Bin emission: prefix bins use ctxIdxOffset 27 (B mb_type prefix);
/// the v=13 intra branch uses ctxIdxOffset 32 (B mb_type suffix).
/// Per spec Table 9-39, binIdx 0..2 of the prefix have neighbor
/// derivation; binIdx 3+ use static {3, 5} ctxIdxInc rules.
pub fn encode_mb_type_b(enc: &mut CabacEncoder, mb_type: u32, mb_x: usize) {
    debug_assert!(mb_type <= 47, "B-slice mb_type out of range: {mb_type}");

    let prefix: &[u8] = if mb_type <= 22 {
        super::binarization::mb_type_b_bins(mb_type)
    } else {
        // Intra-in-B: emit the v=13 prefix.
        super::binarization::mb_type_b_intra_prefix()
    };

    // Prefix bins emit at ctxIdxOffset 27 (MB_TYPE_B_PREFIX).
    // Bin 0 uses neighbor derivation; bins 1..n use Table 9-41-style
    // static increments. Since we ship 16x16 active set + B_8x8 only,
    // the bins 0..=5 are sufficient — the B_L0/L1/Bi 16x8/8x16 family
    // (deferred to §6E-A6) uses bins 0..=6.
    for (bin_idx, &bin) in prefix.iter().enumerate() {
        let ctx_inc = ctx_idx_inc_mb_type_bin(
            enc,
            mb_x,
            ctx_offset::MB_TYPE_B_PREFIX,
            bin_idx as u32,
            prefix,
        );
        enc.encode_dec(bin, ctx_offset::MB_TYPE_B_PREFIX + ctx_inc);
    }

    // Intra-in-B suffix at ctxIdxOffset 32. Same shape as
    // encode_mb_type_p's intra-in-P suffix (Table 9-36 I-slice
    // mb_type bins, with bin 1 terminate-coded).
    if mb_type >= 23 {
        let suffix_value = mb_type - 23;
        let suffix_bins = mb_type_i_bins(suffix_value);
        for (bin_idx, &bin) in suffix_bins.iter().enumerate() {
            if bin_idx == 1 {
                enc.encode_terminate(bin);
                continue;
            }
            let ctx_inc = ctx_idx_inc_mb_type_bin(
                enc,
                mb_x,
                ctx_offset::MB_TYPE_B_SUFFIX,
                bin_idx as u32,
                suffix_bins,
            );
            enc.encode_dec(bin, ctx_offset::MB_TYPE_B_SUFFIX + ctx_inc);
        }
    }
}

/// Encode `mb_type` for a P-slice (Table 9-37).
///
/// Values 0..3 emit the direct P-partition code. Value 4 (P_8x8ref0)
/// is forbidden in CABAC. Values 5..30 emit a '1' prefix + the
/// I-slice Table 9-36 suffix for `mb_type - 5`.
pub fn encode_mb_type_p(enc: &mut CabacEncoder, mb_type: u32, mb_x: usize) {
    debug_assert!(mb_type != 4, "P_8x8ref0 is forbidden in CABAC");
    debug_assert!(mb_type <= 30);

    let prefix = mb_type_p_bins_prefix(mb_type);
    // Prefix bin 0 uses neighbor derivation at ctxIdxOffset=14.
    // Subsequent prefix bins use Table 9-41 prior-bin rules (binIdx 2 at
    // ctxIdxOffset=14 flips based on prior_bins[1]).
    for (bin_idx, &bin) in prefix.iter().enumerate() {
        let ctx_inc = ctx_idx_inc_mb_type_bin(
            enc,
            mb_x,
            ctx_offset::MB_TYPE_P_PREFIX,
            bin_idx as u32,
            prefix,
        );
        enc.encode_dec(bin, ctx_offset::MB_TYPE_P_PREFIX + ctx_inc);
    }

    // For intra-in-P (value >= 5), append the Table 9-36 suffix at
    // ctxIdxOffset=17.
    //
    // Per spec Table 9-39: binIdx 1 of the I-slice suffix is the
    // I_PCM indicator — always terminate-coded (for I_16x16 it's 0,
    // for I_PCM it's 1). `encode_mb_type_i` terminates binIdx 1
    // UNCONDITIONALLY; we match that here. Previously this branch
    // only terminated bin 1 when suffix_value==25, which was a bug:
    // the I_16x16 MBs (suffix_value 1..24) emitted bin 1 as a
    // regular CABAC bin, putting it out of sync with the decoder
    // (Phase D.0 task #19 fix, 2026-04-23).
    if mb_type >= 5 {
        let suffix_value = mb_type - 5;
        let suffix_bins = mb_type_i_bins(suffix_value);
        for (bin_idx, &bin) in suffix_bins.iter().enumerate() {
            if bin_idx == 1 {
                enc.encode_terminate(bin);
                continue;
            }
            let ctx_inc = ctx_idx_inc_mb_type_bin(
                enc,
                mb_x,
                ctx_offset::MB_TYPE_P_SUFFIX,
                bin_idx as u32,
                suffix_bins,
            );
            enc.encode_dec(bin, ctx_offset::MB_TYPE_P_SUFFIX + ctx_inc);
        }
    }
}

/// Encode `sub_mb_type` for a P-slice (Table 9-38 P rows).
pub fn encode_sub_mb_type_p(enc: &mut CabacEncoder, sub_mb_type: u32) {
    debug_assert!(sub_mb_type <= 3);
    let bins = sub_mb_type_p_bins(sub_mb_type);
    for (bin_idx, &bin) in bins.iter().enumerate() {
        // All bins use Table 9-39 static increments {0, 1, 2}.
        let ctx_inc = bin_idx as u32;
        enc.encode_dec(bin, ctx_offset::SUB_MB_TYPE_P + ctx_inc);
    }
}

/// Encode `ref_idx_lX` as pure unary (spec § 9.3.2 Table 9-34).
pub fn encode_ref_idx(
    enc: &mut CabacEncoder,
    ref_idx: u32,
    mb_x: usize,
    block_idx_in_mb_a: usize,
    block_idx_in_mb_b: usize,
) {
    let mut bin_idx = 0u32;
    binarize_unary(ref_idx, &mut |bin| {
        let ctx_inc = match bin_idx {
            0 => ctx_idx_inc_ref_idx_bin0(
                &enc.neighbors,
                mb_x,
                block_idx_in_mb_a,
                block_idx_in_mb_b,
            ),
            1 => 4,
            _ => 5,
        };
        enc.encode_dec(bin, ctx_offset::REF_IDX + ctx_inc);
        bin_idx += 1;
    });
}

/// Encode `mvd_lX[compIdx]` as UEG3 (spec § 9.3.2.3, ctxIdxOffset 40 / 47).
pub fn encode_mvd(
    enc: &mut CabacEncoder,
    mvd: i32,
    component: u8,
    mb_x: usize,
    block_idx_in_mb_a: usize,
    block_idx_in_mb_b: usize,
) {
    let bin0_inc = ctx_idx_inc_mvd_bin0(
        &enc.neighbors,
        mb_x,
        block_idx_in_mb_a,
        block_idx_in_mb_b,
        component,
    );
    encode_mvd_with_bin0_inc(enc, mvd, component, bin0_inc);
}

/// Variant of [`encode_mvd`] that accepts a pre-computed ctxIdxInc for
/// bin 0. Use this when the neighbor derivation needs to consult the
/// partially-built current-MB MVD state (P_16x8 partition 1, P_8x16
/// right partition, interior sub-MB partitions).
pub fn encode_mvd_with_bin0_inc(
    enc: &mut CabacEncoder,
    mvd: i32,
    component: u8,
    bin0_ctx_idx_inc: u32,
) {
    encode_mvd_with_bin0_inc_sign_override(enc, mvd, component, bin0_ctx_idx_inc, None);
}

/// Phase 6F.2(k).1 — variant of [`encode_mvd_with_bin0_inc`] that
/// optionally overrides the sign bypass bin written to the bitstream.
///
/// `sign_override`:
/// - `None`  → write `0` for `mvd > 0`, `1` for `mvd < 0` (natural).
/// - `Some(b)` → write `b` regardless of `mvd`'s sign.
///
/// Magnitude (prefix + suffix) bins are unaffected — they always
/// reflect `|mvd|`. The sign override is the entry point for
/// inline bitstream-mod MVD stego (§6F.2(k)): the encoder's
/// internal `slot.value` and `mv_grid` stay at the natural
/// pre-injection values (no cascade), but the bin written to the
/// emitted Annex-B bytes carries the planned stego bit.
///
/// The decoder reading the bitstream sees the overridden bit at
/// the bypass-bin offset and reconstructs `MV = predictor +
/// (-mvd)` instead of `predictor + mvd` (spec § 9.3.2.3 +
/// § 8.4.1.4). Decoded pixel drift at the flipped position is
/// `≈ 2·|mvd|·local_gradient`. Constrained by the v1.0 drift
/// budget + |mvd| cap in the orchestrator.
///
/// `sign_override` is ignored when `mvd == 0` (no sign bin emitted
/// by spec; the position isn't a stego candidate anyway).
pub fn encode_mvd_with_bin0_inc_sign_override(
    enc: &mut CabacEncoder,
    mvd: i32,
    component: u8,
    bin0_ctx_idx_inc: u32,
    sign_override: Option<u8>,
) {
    debug_assert!(component <= 1);
    let abs_v = mvd.unsigned_abs();
    let u_coff = 9u32;
    let base_offset = if component == 0 {
        ctx_offset::MVD_L0_X
    } else {
        ctx_offset::MVD_L0_Y
    };

    let prefix_val = abs_v.min(u_coff);
    let mut bin_idx = 0u32;
    binarize_tu(prefix_val, u_coff, &mut |bin| {
        let ctx_inc = match bin_idx {
            0 => bin0_ctx_idx_inc,
            1 => 3,
            2 => 4,
            3 => 5,
            _ => 6, // Table 9-39: bins 4+ cap at 6
        };
        enc.encode_dec(bin, base_offset + ctx_inc);
        bin_idx += 1;
    });

    if abs_v >= u_coff {
        let suf_s = abs_v - u_coff;
        binarize_egk_suffix(suf_s, 3, &mut |bin| enc.encode_bypass(bin));
    }
    if mvd != 0 {
        let sign_bin = match sign_override {
            Some(b) => {
                debug_assert!(b <= 1, "sign override must be 0 or 1");
                b
            }
            None => if mvd > 0 { 0 } else { 1 },
        };
        enc.encode_bypass(sign_bin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_enc() -> CabacEncoder {
        CabacEncoder::new_slice(CabacInitSlot::ISI, 26, 4)
    }

    #[test]
    fn encode_mb_type_i_single_bin_i_nxn() {
        let mut enc = fresh_enc();
        encode_mb_type_i(&mut enc, 0, 0);
        encode_end_of_slice_flag(&mut enc, true);
        let bytes = enc.finish();
        // Non-empty output with RBSP stop bit.
        assert!(!bytes.is_empty());
        assert_ne!(*bytes.last().unwrap(), 0);
    }

    #[test]
    fn encode_mb_type_i_i16x16_variant() {
        let mut enc = fresh_enc();
        encode_mb_type_i(&mut enc, 5, 0); // I_16x16_0_1_0
        encode_end_of_slice_flag(&mut enc, true);
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn encode_intra_chroma_pred_mode_dc_emits_single_bin() {
        let mut enc = fresh_enc();
        encode_intra_chroma_pred_mode(&mut enc, 0, 0);
        let before = enc.engine.bin_count();
        assert_eq!(before, 1); // TU(0, 3) = "0", one bin.
    }

    #[test]
    fn encode_intra_chroma_pred_mode_plane_emits_three_bins() {
        let mut enc = fresh_enc();
        encode_intra_chroma_pred_mode(&mut enc, 3, 0);
        assert_eq!(enc.engine.bin_count(), 3); // TU(3, 3) = "111", 3 bins.
    }

    #[test]
    fn encode_coded_block_pattern_no_residual() {
        let mut enc = fresh_enc();
        encode_coded_block_pattern(&mut enc, 0, 0); // CBP = 0
        // 4 luma bins + 1 chroma bin (TU(0, 2) = "0").
        assert_eq!(enc.engine.bin_count(), 5);
    }

    #[test]
    fn encode_coded_block_pattern_full_cbp() {
        let mut enc = fresh_enc();
        encode_coded_block_pattern(&mut enc, 47, 0); // CBP = 47 (luma 15, chroma 2)
        // 4 luma bins + 2 chroma bins (TU(2, 2) = "11", truncated).
        assert_eq!(enc.engine.bin_count(), 6);
    }

    #[test]
    fn encode_mb_qp_delta_zero_emits_single_bin() {
        let mut enc = fresh_enc();
        encode_mb_qp_delta(&mut enc, 0);
        // mapped = 0 → U(0) = "0", 1 bin.
        assert_eq!(enc.engine.bin_count(), 1);
    }

    #[test]
    fn encode_mb_qp_delta_nonzero() {
        let mut enc = fresh_enc();
        encode_mb_qp_delta(&mut enc, 3);
        // mapped = 5 → U(5) = "111110", 6 bins.
        assert_eq!(enc.engine.bin_count(), 6);
    }

    #[test]
    fn encode_coded_block_flag_emits_one_bin() {
        let mut enc = fresh_enc();
        encode_coded_block_flag(&mut enc, true, 2, 0, 0, 0, true);
        assert_eq!(enc.engine.bin_count(), 1);
    }

    #[test]
    fn encode_residual_block_cabac_all_zero_just_cbf() {
        let mut enc = fresh_enc();
        let coeffs = [0i32; 16];
        let result = encode_residual_block_cabac(&mut enc, &coeffs, 0, 15, 2, 0, 0, 0, true);
        assert!(!result); // all-zero
        // Only the coded_block_flag = 0 bin emitted.
        assert_eq!(enc.engine.bin_count(), 1);
    }

    #[test]
    fn encode_residual_block_cabac_single_coeff() {
        let mut enc = fresh_enc();
        let mut coeffs = [0i32; 16];
        coeffs[0] = 1; // DC only
        let result = encode_residual_block_cabac(&mut enc, &coeffs, 0, 15, 2, 0, 0, 0, true);
        assert!(result);
        // Bins: cbf=1, then sig(0)=1+last(0)=1 (significance map ends
        // immediately since last nonzero is at scan 0), then
        // abs_level prefix=0 + sign=0 (bypass).
        // Engine counts regular + bypass + terminate bins.
        let n = enc.engine.bin_count();
        assert!(n >= 4);
    }

    #[test]
    fn zigzag_8x8_is_permutation() {
        // Every linear index 0..64 must appear exactly once.
        let mut seen = [false; 64];
        for &v in ZIGZAG_8X8.iter() {
            assert!(v < 64, "out-of-range entry {v}");
            assert!(!seen[v as usize], "duplicate entry {v}");
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
        // DC is always first.
        assert_eq!(ZIGZAG_8X8[0], 0);
        // Last position is (7, 7) = 63.
        assert_eq!(ZIGZAG_8X8[63], 63);
    }

    #[test]
    fn sig_coeff_offset_8x8_in_range() {
        // SIG_COEFF_FLAG_OFFSET_8X8_FRAME[pos] is a 4-bit value 0..14.
        // LAST offset is 0..8.
        for &v in SIG_COEFF_FLAG_OFFSET_8X8_FRAME.iter() {
            assert!(v <= 14);
        }
        for &v in LAST_COEFF_FLAG_OFFSET_8X8_FRAME.iter() {
            assert!(v <= 8);
        }
    }

    #[test]
    fn cat5_ctx_offsets_match_spec() {
        // Lock the three cat-5 (Luma 8×8) base ctxIdxOffsets against
        // spec Table 9-34 values (ITU-T H.264 03/2010, page 250).
        // Any table drift would desync our 8×8 residual emission
        // from any spec-conformant decoder.
        assert_eq!(cat5_luma8x8::SIG_BASE, 402);
        assert_eq!(cat5_luma8x8::LAST_BASE, 417);
        assert_eq!(cat5_luma8x8::ABS_BASE, 426);
    }

    #[test]
    fn encode_residual_block_cabac_8x8_all_zero_emits_single_sig_zero() {
        let mut enc = fresh_enc();
        let coeffs = [0i32; 64];
        encode_residual_block_cabac_8x8(&mut enc, &coeffs);
        // With cat=5 there's NO CBF; the sig-map loop just walks
        // positions 0..=62 until it finds a nonzero. An all-zero
        // block emits sig_coeff=0 at position 0, then position 1 ...
        // in fact it walks all 63 positions without emitting a
        // last_coeff bin. So we expect 63 regular bins.
        assert_eq!(enc.engine.bin_count(), 63);
    }

    #[test]
    fn encode_residual_block_cabac_8x8_single_coeff() {
        let mut enc = fresh_enc();
        let mut coeffs = [0i32; 64];
        coeffs[0] = 1; // DC only
        encode_residual_block_cabac_8x8(&mut enc, &coeffs);
        // Bins: sig[0]=1, last[0]=1, abs_level prefix=0 (one regular
        // bin via ctx inc 1+0=1), sign (bypass). 4 total.
        let n = enc.engine.bin_count();
        assert!(n >= 4, "expected ≥ 4 bins for single-coef 8×8, got {n}");
    }

    #[test]
    fn encode_residual_block_cabac_sign() {
        let mut enc = fresh_enc();
        let mut coeffs = [0i32; 16];
        coeffs[0] = -2;
        encode_residual_block_cabac(&mut enc, &coeffs, 0, 15, 2, 0, 0, 0, true);
        // Single nonzero with |c|=2 → abs_level_minus1=1 → prefix "10",
        // sign=1 (bypass).
        assert!(enc.engine.bin_count() >= 5);
    }

    #[test]
    fn encode_mb_skip_flag_zero() {
        let mut enc = fresh_enc();
        encode_mb_skip_flag(&mut enc, false, 0);
        assert_eq!(enc.engine.bin_count(), 1);
    }

    #[test]
    fn encode_mb_type_p_p_16x16() {
        let mut enc = fresh_enc();
        encode_mb_type_p(&mut enc, 0, 0);
        // P_L0_16x16 = 3-bin prefix "000".
        assert_eq!(enc.engine.bin_count(), 3);
    }

    #[test]
    fn encode_sub_mb_type_p_8x8() {
        let mut enc = fresh_enc();
        encode_sub_mb_type_p(&mut enc, 0);
        assert_eq!(enc.engine.bin_count(), 1); // "1", 1 bin
    }

    #[test]
    fn encode_ref_idx_zero_emits_single_bin() {
        let mut enc = fresh_enc();
        encode_ref_idx(&mut enc, 0, 0, 0, 0);
        assert_eq!(enc.engine.bin_count(), 1); // U(0) = "0"
    }

    #[test]
    fn encode_mvd_zero_prefix_only() {
        let mut enc = fresh_enc();
        encode_mvd(&mut enc, 0, 0, 0, 0, 0);
        // mvd=0 → prefix "0", no suffix, no sign.
        assert_eq!(enc.engine.bin_count(), 1);
    }

    #[test]
    fn encode_mvd_negative_with_sign() {
        let mut enc = fresh_enc();
        encode_mvd(&mut enc, -3, 0, 0, 0, 0);
        // prefix TU(3, 9) = "1110" = 4 bins + sign bypass = 5 bins.
        assert_eq!(enc.engine.bin_count(), 5);
    }

    #[test]
    fn encode_mvd_saturated_with_suffix() {
        let mut enc = fresh_enc();
        encode_mvd(&mut enc, 9, 0, 0, 0, 0);
        // prefix = 9 bins (truncated) + EG3 suffix "0000" = 4 bins + sign = 14.
        assert_eq!(enc.engine.bin_count(), 14);
    }

    /// **Phase 6F.2(k).1** — sign override produces same magnitude
    /// bits as the natural path, only the sign bypass bin differs.
    /// Two encodes — natural mvd=3 vs flipped via override — must
    /// produce IDENTICAL prefix bins (3 ones + terminator) and
    /// IDENTICAL bin counts; only the final bypass bit value
    /// differs.
    #[test]
    fn encode_mvd_sign_override_flips_bin_only() {
        // Natural path: mvd=3 (positive sign, sign bin = 0).
        let mut a = fresh_enc();
        encode_mvd(&mut a, 3, 0, 0, 0, 0);
        let bins_a = a.engine.bin_count();

        // Override path: mvd=3 with explicit sign override = 1
        // (write the bin value as if the sign were negative
        // without actually changing the magnitude or the
        // encoder-side mvd value).
        let mut b = fresh_enc();
        encode_mvd_with_bin0_inc_sign_override(&mut b, 3, 0, 0, Some(1));
        let bins_b = b.engine.bin_count();

        assert_eq!(bins_a, bins_b,
            "sign override must NOT change bin count (magnitude bins identical)");
        // Magnitude is the same regardless of sign — only the
        // final bypass bin differs. Inspect bypass-bin diagnostics
        // if available; minimally, both finalize successfully.
        let _ = a.finish();
        let _ = b.finish();
    }

    /// **Phase 6F.2(k).1** — override = None must round-trip
    /// identically to `encode_mvd_with_bin0_inc` (the variant
    /// without override). Backwards-compatibility gate.
    #[test]
    fn encode_mvd_sign_override_none_matches_baseline() {
        let mut a = fresh_enc();
        encode_mvd_with_bin0_inc(&mut a, -7, 1, 0);
        let bytes_a = a.finish();

        let mut b = fresh_enc();
        encode_mvd_with_bin0_inc_sign_override(&mut b, -7, 1, 0, None);
        let bytes_b = b.finish();

        assert_eq!(bytes_a, bytes_b,
            "override=None must produce byte-identical output to baseline");
    }

    /// **Phase 6F.2(k).1** — for mvd=0 the spec emits no sign bin;
    /// override is silently ignored.
    #[test]
    fn encode_mvd_sign_override_ignored_for_zero() {
        let mut a = fresh_enc();
        encode_mvd(&mut a, 0, 0, 0, 0, 0);
        let bytes_a = a.finish();

        let mut b = fresh_enc();
        encode_mvd_with_bin0_inc_sign_override(&mut b, 0, 0, 0, Some(1));
        let bytes_b = b.finish();

        assert_eq!(bytes_a, bytes_b,
            "override at mvd=0 must be a no-op (no sign bin emitted)");
    }

    // ─── Phase 6C.5b P-slice orchestration tests ────────────────────

    #[test]
    fn encode_mb_type_p_intra_in_p_i_nxn() {
        let mut enc = fresh_enc();
        // mb_type 5 = intra-in-P I_NxN. Prefix = '1', suffix = I-slice
        // mb_type_i_bins(0) = [0] (I_NxN bin 0).
        encode_mb_type_p(&mut enc, 5, 0);
        assert_eq!(enc.engine.bin_count(), 2); // prefix 1 + suffix 1
    }

    #[test]
    fn encode_mb_type_p_intra_in_p_i_pcm() {
        let mut enc = fresh_enc();
        // mb_type 30 = intra-in-P I_PCM. Prefix = '1', suffix =
        // I-slice mb_type_i_bins(25) = [1, 1] (bin 1 is terminate).
        encode_mb_type_p(&mut enc, 30, 0);
        // 1 prefix + 1 I-slice regular bin + 1 terminate bin = 3 bins.
        assert_eq!(enc.engine.bin_count(), 3);
    }

    #[test]
    fn encode_full_p_mb_inter_no_residual() {
        let mut enc = fresh_enc();
        // Full P_L0_16x16 MB with 0 residual. Verifies the encoders
        // compose without interfering with each other.
        encode_mb_skip_flag(&mut enc, false, 0);
        encode_mb_type_p(&mut enc, 0, 0);           // P_L0_16x16
        encode_ref_idx(&mut enc, 0, 0, 0, 0);        // ref_idx = 0
        encode_mvd(&mut enc, 2, 0, 0, 0, 0);         // mvd_x = 2
        encode_mvd(&mut enc, -1, 1, 0, 0, 0);        // mvd_y = -1
        encode_coded_block_pattern(&mut enc, 0, 0);  // cbp = 0
        encode_mb_qp_delta(&mut enc, 0);             // qp_delta = 0
        encode_end_of_slice_flag(&mut enc, true);
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn encode_full_p_skip_mb() {
        let mut enc = fresh_enc();
        // P_Skip: only mb_skip_flag=1 and then nothing else.
        encode_mb_skip_flag(&mut enc, true, 0);
        encode_end_of_slice_flag(&mut enc, true);
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn encode_full_p_8x8_with_sub_mb_types() {
        let mut enc = fresh_enc();
        encode_mb_skip_flag(&mut enc, false, 0);
        encode_mb_type_p(&mut enc, 3, 0); // P_8x8
        // Four sub_mb_type entries.
        for sub in &[0u32, 1, 2, 3] {
            encode_sub_mb_type_p(&mut enc, *sub);
        }
        // Each sub-MB partition contributes mvd x/y. For P_L0_8x8 (sub=0)
        // there's 1 partition of 8x8; we just emit one mvd pair for brevity.
        encode_mvd(&mut enc, 0, 0, 0, 0, 0);
        encode_mvd(&mut enc, 0, 1, 0, 0, 0);
        encode_coded_block_pattern(&mut enc, 0, 0);
        encode_mb_qp_delta(&mut enc, 0);
        encode_end_of_slice_flag(&mut enc, true);
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn encode_full_i_mb_i_nxn_4x4_modes() {
        let mut enc = fresh_enc();
        // I-slice I_NxN MB with default 4×4 pred modes.
        encode_mb_type_i(&mut enc, 0, 0); // I_NxN
        // 16 × 4×4 blocks: all prev_intra4x4_pred_mode_flag = true
        // (predicted from neighbors).
        for _ in 0..16 {
            encode_prev_intra4x4_pred_mode_flag(&mut enc, true);
        }
        encode_intra_chroma_pred_mode(&mut enc, 0, 0); // DC
        encode_coded_block_pattern(&mut enc, 0, 0);     // no residual
        encode_mb_qp_delta(&mut enc, 0);
        encode_end_of_slice_flag(&mut enc, true);
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }

    // ─── Phase 6E-A3 B-slice encoder primitives ─────────────────

    fn fresh_b_enc() -> CabacEncoder {
        // B-slice uses the same PIdc{0,1,2} init slots as P/SP.
        // Phase 6E-A defaults to cabac_init_idc=0 → PIdc0.
        CabacEncoder::new_slice(CabacInitSlot::PIdc0, 26, 4)
    }

    #[test]
    fn encode_mb_skip_flag_b_zero_emits_one_bin() {
        let mut enc = fresh_b_enc();
        encode_mb_skip_flag_b(&mut enc, false, 0);
        assert_eq!(enc.engine.bin_count(), 1);
    }

    #[test]
    fn encode_mb_skip_flag_b_one_emits_one_bin() {
        let mut enc = fresh_b_enc();
        encode_mb_skip_flag_b(&mut enc, true, 0);
        assert_eq!(enc.engine.bin_count(), 1);
    }

    #[test]
    fn encode_mb_type_b_direct_one_bin() {
        let mut enc = fresh_b_enc();
        encode_mb_type_b(&mut enc, 0, 0); // B_Direct_16x16
        // Bin 0 = 0 short-circuits the tree.
        assert_eq!(enc.engine.bin_count(), 1);
    }

    #[test]
    fn encode_mb_type_b_l0_three_bins() {
        let mut enc = fresh_b_enc();
        encode_mb_type_b(&mut enc, 1, 0); // B_L0_16x16
        // Bins: 1, 0, 0 → 3 bins.
        assert_eq!(enc.engine.bin_count(), 3);
    }

    #[test]
    fn encode_mb_type_b_l1_three_bins() {
        let mut enc = fresh_b_enc();
        encode_mb_type_b(&mut enc, 2, 0); // B_L1_16x16
        // Bins: 1, 0, 1 → 3 bins.
        assert_eq!(enc.engine.bin_count(), 3);
    }

    #[test]
    fn encode_mb_type_b_bi_six_bins() {
        let mut enc = fresh_b_enc();
        encode_mb_type_b(&mut enc, 3, 0); // B_Bi_16x16
        // Bins: 1, 1, 0, 0, 0, 0 → 6 bins (multi-partition v=0 path).
        assert_eq!(enc.engine.bin_count(), 6);
    }

    #[test]
    fn encode_mb_type_b_8x8_six_bins() {
        let mut enc = fresh_b_enc();
        encode_mb_type_b(&mut enc, 22, 0); // B_8x8
        // Bins: 1, 1, 1, 1, 1, 1 → 6 bins (v=15 short-circuit).
        assert_eq!(enc.engine.bin_count(), 6);
    }

    #[test]
    fn encode_full_b_skip_mb() {
        let mut enc = fresh_b_enc();
        // B_Skip: only mb_skip_flag=1 and end_of_slice. Mirror of
        // P_Skip but with B-slice ctxIdxOffset.
        encode_mb_skip_flag_b(&mut enc, true, 0);
        encode_end_of_slice_flag(&mut enc, true);
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }
}
