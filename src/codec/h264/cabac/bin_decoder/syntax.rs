// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Per-syntax-element CABAC decoders. Mirror of `cabac::encoder` fn-for-fn.
//
// Each decode fn:
//   1. Reads the same bins that the matching `encode_*` fn writes,
//      using the same per-bin ctx_idx derivation and same neighbor
//      state.
//   2. Inverts the binarization to reconstruct the original syntax
//      value.
//   3. Returns the decoded value (Result-wrapped for I/O errors).
//
// Phase 6D.2 chunk 2: simple syntax elements (no stego positions).
// MVD + residual_block_cabac, which DO emit stego positions, land
// in subsequent chunks alongside the position-tracker module.

use crate::codec::h264::cabac::encoder::{
    cat5_luma8x8, ctx_offset, CTX_BLOCK_CAT_OFFSET,
    LAST_COEFF_FLAG_OFFSET_8X8_FRAME, SIG_COEFF_FLAG_OFFSET_8X8_FRAME,
};
use crate::codec::h264::cabac::neighbor::{
    ctx_idx_inc_cbp_chroma, ctx_idx_inc_coded_block_flag, ctx_idx_inc_coeff_abs_level,
    ctx_idx_inc_intra_chroma_pred_mode_bin0, ctx_idx_inc_mb_qp_delta_bin0,
    ctx_idx_inc_mb_skip_flag, ctx_idx_inc_mb_type_bin0, ctx_idx_inc_mvd_bin0,
    ctx_idx_inc_prior_bin, ctx_idx_inc_ref_idx_bin0, ctx_idx_inc_sig_4x4,
    ctx_idx_inc_sig_chroma_dc, compute_cbp_luma_ctx_idx_inc_bin, CabacNeighborContext,
};
use crate::codec::h264::stego::{
    Axis, BinKind, EmbedDomain, PositionKey, PositionLogger, SyntaxPath,
};

use super::decoder::CabacDecoder;
use super::engine::DecodeError;

// ─── Inverse-binarization helpers ──────────────────────────────────

/// Decode a unary value: read regular bins until a 0 appears. Calls
/// `ctx_for_bin(bin_idx)` for each bin's ctxIdx. Returns the count of
/// 1s before the terminating 0. `max_value` caps the loop in case of
/// malformed input.
pub fn decode_unary(
    dec: &mut CabacDecoder<'_>,
    max_value: u32,
    mut ctx_for_bin: impl FnMut(u32) -> u32,
) -> Result<u32, DecodeError> {
    let mut v = 0u32;
    while v <= max_value {
        let ctx_idx = ctx_for_bin(v);
        let bin = dec.decode_dec(ctx_idx)?;
        if bin == 0 {
            return Ok(v);
        }
        v += 1;
    }
    Ok(max_value)
}

/// Decode a Truncated-Unary value with cMax. Read up to `c_max` bins;
/// stop at the first 0 OR after `c_max` ones (saturated).
// Spec § 9.3.2.6 TU decode is described in terms of an accumulator
// `v` that increments for every emitted '1' until the first '0'.
// Mirroring that shape — even though `v == bin_idx` here — keeps the
// reader aligned with the spec's prose.
#[allow(clippy::explicit_counter_loop)]
pub fn decode_tu(
    dec: &mut CabacDecoder<'_>,
    c_max: u32,
    mut ctx_for_bin: impl FnMut(u32) -> u32,
) -> Result<u32, DecodeError> {
    let mut v = 0u32;
    for bin_idx in 0..c_max {
        let ctx_idx = ctx_for_bin(bin_idx);
        let bin = dec.decode_dec(ctx_idx)?;
        if bin == 0 {
            return Ok(v);
        }
        v += 1;
    }
    Ok(c_max)
}

/// Decode an FL value. Spec § 9.3.2.6 binIdx ordering: binIdx=0 is
/// the **MSB** for general FL. The encoder's `binarize_fl` matches
/// this: it shifts MSB-first.
///
/// HOWEVER, several call sites in `cabac::encoder` use a different
/// ordering (LSB-first), described as "FL with binIdx=0 = LSB" in the
/// comments at those sites. Those sites encode bits manually rather
/// than calling `binarize_fl`. The decoder uses an `lsb_first` flag
/// to switch between the two orderings.
pub fn decode_fl(
    dec: &mut CabacDecoder<'_>,
    n_bits: u32,
    lsb_first: bool,
    mut ctx_for_bin: impl FnMut(u32) -> u32,
) -> Result<u32, DecodeError> {
    let mut v = 0u32;
    if lsb_first {
        for bin_idx in 0..n_bits {
            let ctx_idx = ctx_for_bin(bin_idx);
            let bin = dec.decode_dec(ctx_idx)? as u32;
            v |= bin << bin_idx;
        }
    } else {
        for bin_idx in 0..n_bits {
            let ctx_idx = ctx_for_bin(bin_idx);
            let bin = dec.decode_dec(ctx_idx)? as u32;
            v = (v << 1) | bin;
        }
    }
    Ok(v)
}

/// Inverse of `mb_qp_delta_remap`: unsigned mapped value → signed.
#[inline]
pub fn mb_qp_delta_unmap(mapped: u32) -> i32 {
    if mapped == 0 {
        0
    } else if (mapped & 1) == 1 {
        // Odd → positive: mapped = 2v - 1 → v = (mapped + 1) / 2.
        mapped.div_ceil(2) as i32
    } else {
        // Even → negative: mapped = 2v → v = -(mapped / 2).
        -((mapped / 2) as i32)
    }
}

// ─── Simple syntax decoders ─────────────────────────────────────────

/// Decode `end_of_slice_flag` (terminate). Returns `true` if this MB
/// is the last of the slice.
pub fn decode_end_of_slice_flag(dec: &mut CabacDecoder<'_>) -> Result<bool, DecodeError> {
    Ok(dec.decode_terminate()? == 1)
}

/// Decode `mb_skip_flag` (P-slice). Spec § 9.3.3.1.1.1.
pub fn decode_mb_skip_flag(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
) -> Result<bool, DecodeError> {
    let ctx_inc = ctx_idx_inc_mb_skip_flag(&dec.neighbors, mb_x);
    let bin = dec.decode_dec(ctx_offset::MB_SKIP_FLAG_P + ctx_inc)?;
    Ok(bin != 0)
}

/// Phase 6E-A3 — decode `mb_skip_flag` (B-slice). Same neighbor
/// derivation rule as P (spec § 9.3.3.1.1.1; both P and B use
/// `condTermFlagN = !is_skip(N)`), different ctxIdxOffset (24 for
/// B vs 11 for P, spec Table 9-39).
pub fn decode_mb_skip_flag_b(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
) -> Result<bool, DecodeError> {
    let ctx_inc = ctx_idx_inc_mb_skip_flag(&dec.neighbors, mb_x);
    let bin = dec.decode_dec(ctx_offset::MB_SKIP_FLAG_B + ctx_inc)?;
    Ok(bin != 0)
}

/// Decode `mb_qp_delta`. Spec § 9.3.2.7 + § 9.3.3.1.1.5.
pub fn decode_mb_qp_delta(dec: &mut CabacDecoder<'_>) -> Result<i32, DecodeError> {
    let bin0_inc = ctx_idx_inc_mb_qp_delta_bin0(&dec.neighbors);
    let mapped = decode_unary(dec, 53, |bin_idx| {
        let ctx_inc = match bin_idx {
            0 => bin0_inc,
            1 => 2,
            _ => 3,
        };
        ctx_offset::MB_QP_DELTA + ctx_inc
    })?;
    Ok(mb_qp_delta_unmap(mapped))
}

/// Decode `intra_chroma_pred_mode`. Spec § 9.3.2 TU cMax=3.
pub fn decode_intra_chroma_pred_mode(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
) -> Result<u8, DecodeError> {
    let bin0_inc = ctx_idx_inc_intra_chroma_pred_mode_bin0(&dec.neighbors, mb_x);
    let v = decode_tu(dec, 3, |bin_idx| {
        let ctx_inc = if bin_idx == 0 { bin0_inc } else { 3 };
        ctx_offset::INTRA_CHROMA_PRED_MODE + ctx_inc
    })?;
    Ok(v as u8)
}

/// Decode `prev_intra4x4_pred_mode_flag`. Spec § 9.3.3.1.1.6 (single
/// bin, ctxIdxOffset=68, ctxIdxInc=0).
pub fn decode_prev_intra4x4_pred_mode_flag(
    dec: &mut CabacDecoder<'_>,
) -> Result<bool, DecodeError> {
    let bin = dec.decode_dec(ctx_offset::PREV_INTRA_PRED_MODE_FLAG)?;
    Ok(bin != 0)
}

/// Decode `rem_intra4x4_pred_mode` (3-bit FL, LSB-first per spec
/// § 9.3.2.4 binIdx rule).
pub fn decode_rem_intra4x4_pred_mode(dec: &mut CabacDecoder<'_>) -> Result<u8, DecodeError> {
    let v = decode_fl(dec, 3, /* lsb_first */ true, |_| ctx_offset::REM_INTRA_PRED_MODE)?;
    Ok(v as u8)
}

/// Decode `transform_size_8x8_flag` (single regular bin).
pub fn decode_transform_size_8x8_flag(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
) -> Result<bool, DecodeError> {
    let inc = crate::codec::h264::cabac::neighbor::ctx_idx_inc_transform_size_8x8_flag(
        &dec.neighbors,
        mb_x,
    );
    let bin = dec.decode_dec(ctx_offset::TRANSFORM_SIZE_8X8_FLAG + inc)?;
    Ok(bin != 0)
}

/// Decode `ref_idx_lX` (unary; ctxIdxOffset=54).
pub fn decode_ref_idx(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
    block_idx_in_mb_a: usize,
    block_idx_in_mb_b: usize,
) -> Result<u32, DecodeError> {
    let bin0_inc = ctx_idx_inc_ref_idx_bin0(
        &dec.neighbors,
        mb_x,
        block_idx_in_mb_a,
        block_idx_in_mb_b,
    );
    decode_unary(dec, 31, |bin_idx| {
        let ctx_inc = match bin_idx {
            0 => bin0_inc,
            1 => 4,
            _ => 5,
        };
        ctx_offset::REF_IDX + ctx_inc
    })
}

/// Decode `coded_block_pattern` (4-bit luma FL LSB-first + 2-value
/// chroma TU). Returns CBP byte (luma low nibble + chroma high
/// nibble).
pub fn decode_coded_block_pattern(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
) -> Result<u8, DecodeError> {
    // Luma prefix: 4 FL bins, LSB-first, with neighbor ctx_inc that
    // uses the partial CBP being built.
    let mut luma: u8 = 0;
    for bin_idx in 0..4u32 {
        let ctx_inc = compute_cbp_luma_ctx_idx_inc_bin(bin_idx, luma, &dec.neighbors, mb_x);
        let bin = dec.decode_dec(ctx_offset::CBP_LUMA + ctx_inc)?;
        luma |= bin << bin_idx;
    }
    // Chroma suffix: TU cMax=2. ctxIdxInc depends only on neighbors
    // and bin_idx (not on current-MB partial state), so we can
    // pre-compute both bins' ctxIdxInc to avoid the borrow conflict
    // between the helper's `&mut dec` and the closure's `&dec.neighbors`.
    let chroma_inc_bin0 = ctx_idx_inc_cbp_chroma(&dec.neighbors, mb_x, 0);
    let chroma_inc_bin1 = ctx_idx_inc_cbp_chroma(&dec.neighbors, mb_x, 1);
    let chroma = decode_tu(dec, 2, |bin_idx| {
        let ctx_inc = if bin_idx == 0 { chroma_inc_bin0 } else { chroma_inc_bin1 };
        ctx_offset::CBP_CHROMA + ctx_inc
    })?;
    Ok((chroma << 4) as u8 | (luma & 0x0F))
}

/// Decoder mirror of `cabac::encoder::ctx_idx_inc_mb_type_bin`. For
/// bin 0 uses the neighbor derivation; for bin 1 the caller must
/// route to `decode_terminate` (this fn is not consulted); for
/// bins 2+ uses the prior-bin table or the spec Table 9-39 static
/// fallback. **Production-default values only** — the dev-only
/// `PHASM_IIP_CTX_*` env-var sweep knobs from the encoder are NOT
/// honoured here. Production stego encode never sets those vars;
/// sweep-encoded bitstreams are not decodable by this module.
fn ctx_idx_inc_mb_type_bin(
    neighbors: &CabacNeighborContext,
    mb_x: usize,
    ctx_idx_offset: u32,
    bin_idx: u32,
    prior_bins: &[u8],
) -> u32 {
    if bin_idx == 0 {
        return ctx_idx_inc_mb_type_bin0(neighbors, mb_x, ctx_idx_offset);
    }
    if let Some(inc) = ctx_idx_inc_prior_bin(ctx_idx_offset, bin_idx, prior_bins) {
        return inc;
    }
    match (ctx_idx_offset, bin_idx) {
        (3, 2) => 3,
        (3, 3) => 4,
        (3, 6) => 7,
        (14, 1) => 1,
        (17, 0) => 0,
        (17, 2) => 1,
        (17, 3) => 2,
        (17, 5) => 3,
        (17, 6) => 3,
        // §6E-A6.1q.f — B-slice mb_type prefix bins per spec Table
        // 9-41 / ffmpeg. Mirror of encoder.rs:178-203 fix.
        (27, 1) => 3,
        (27, 3) => 5,
        (27, 4) => 5,
        (27, 5) => 5,
        _ => 0,
    }
}

/// Decode `mb_type` for an I-slice. Returns the mb_type value 0..=25.
/// Spec Table 9-36. Bin 1 routes through `decode_terminate` per spec;
/// all other bins are regular-mode coded.
pub fn decode_mb_type_i(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
) -> Result<u32, DecodeError> {
    let mut bins: Vec<u8> = Vec::with_capacity(7);

    // Bin 0.
    let inc0 = ctx_idx_inc_mb_type_bin(&dec.neighbors, mb_x, ctx_offset::MB_TYPE_I, 0, &bins);
    let bin0 = dec.decode_dec(ctx_offset::MB_TYPE_I + inc0)?;
    bins.push(bin0);
    if bin0 == 0 {
        return Ok(0); // I_NxN
    }

    // Bin 1: TERMINATE.
    let bin1 = dec.decode_terminate()?;
    bins.push(bin1);
    if bin1 == 1 {
        return Ok(25); // I_PCM
    }

    // I_16x16 path: bin 2 (cbp_luma flag) + bin 3 (cbp_chroma test).
    let inc2 = ctx_idx_inc_mb_type_bin(&dec.neighbors, mb_x, ctx_offset::MB_TYPE_I, 2, &bins);
    let bin2 = dec.decode_dec(ctx_offset::MB_TYPE_I + inc2)?;
    bins.push(bin2);
    let inc3 = ctx_idx_inc_mb_type_bin(&dec.neighbors, mb_x, ctx_offset::MB_TYPE_I, 3, &bins);
    let bin3 = dec.decode_dec(ctx_offset::MB_TYPE_I + inc3)?;
    bins.push(bin3);

    // bin3 picks 6-bin (=0) vs 7-bin (=1) variant. Read remaining bins.
    let n_remaining = if bin3 == 0 { 2 } else { 3 };
    for bin_idx in 4..(4 + n_remaining) {
        let inc = ctx_idx_inc_mb_type_bin(
            &dec.neighbors, mb_x, ctx_offset::MB_TYPE_I, bin_idx as u32, &bins,
        );
        let bin = dec.decode_dec(ctx_offset::MB_TYPE_I + inc)?;
        bins.push(bin);
    }

    Ok(match_mb_type_i_bins(&bins))
}

/// Look up an mb_type value 1..=24 from the bin string. Returns 0
/// (I_NxN) or 25 (I_PCM) only if the caller passed those — those
/// cases are handled in the caller.
fn match_mb_type_i_bins(bins: &[u8]) -> u32 {
    use crate::codec::h264::cabac::binarization::mb_type_i_bins;
    // Linear search over Table 9-36 — 26 entries, one match.
    for v in 0u32..=25 {
        if mb_type_i_bins(v) == bins {
            return v;
        }
    }
    // Should be unreachable if the encoder + decoder are paired and
    // the bitstream is well-formed.
    debug_assert!(false, "no mb_type_i match for bins {bins:?}");
    0
}

/// Phase 6E-A4 — decode `mb_type` for a B-slice. Returns 0..=22
/// for non-intra mb_types, or 23..=47 for intra-in-B. Spec Table
/// 9-37 B rows; this decoder is the inverse of
/// `encode_mb_type_b`.
///
/// Tree structure:
/// - bin0=0 → mb_type=0 (B_Direct_16x16).
/// - bin0=1, bin1=0, bin2 picks → 1 (B_L0_16x16) or 2 (B_L1_16x16).
/// - bin0=1, bin1=1 → multi-partition. 4 bins build value v=0..15:
///   - v ∈ [0,7]: mb_type = v + 3 (3..10).
///   - v == 13: intra-in-B; emit bin 1 of I-suffix as TERMINATE,
///     decode rest of I mb_type, return 23 + suffix_value.
///   - v == 14: mb_type = 11.
///   - v == 15: mb_type = 22 (B_8x8).
///   - else (v ∈ {8..12}): bin6 selects, mb_type = (v<<1 | bin6) - 4.
pub fn decode_mb_type_b(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
) -> Result<u32, DecodeError> {
    let mut prefix_bins: Vec<u8> = Vec::with_capacity(7);

    // bin0 — direct vs non-direct.
    let inc0 = ctx_idx_inc_mb_type_bin(
        &dec.neighbors, mb_x, ctx_offset::MB_TYPE_B_PREFIX, 0, &prefix_bins,
    );
    let bin0 = dec.decode_dec(ctx_offset::MB_TYPE_B_PREFIX + inc0)?;
    prefix_bins.push(bin0);
    if bin0 == 0 {
        return Ok(0); // B_Direct_16x16
    }

    // bin1 — 16x16 vs multi-partition.
    let inc1 = ctx_idx_inc_mb_type_bin(
        &dec.neighbors, mb_x, ctx_offset::MB_TYPE_B_PREFIX, 1, &prefix_bins,
    );
    let bin1 = dec.decode_dec(ctx_offset::MB_TYPE_B_PREFIX + inc1)?;
    prefix_bins.push(bin1);

    if bin1 == 0 {
        // 16x16 L0/L1: bin2 picks B_L0_16x16 (1) vs B_L1_16x16 (2).
        let inc2 = ctx_idx_inc_mb_type_bin(
            &dec.neighbors, mb_x, ctx_offset::MB_TYPE_B_PREFIX, 2, &prefix_bins,
        );
        let bin2 = dec.decode_dec(ctx_offset::MB_TYPE_B_PREFIX + inc2)?;
        return Ok(1 + bin2 as u32);
    }

    // Multi-partition: read 4 more bins → 4-bit value v.
    let mut v: u32 = 0;
    for bin_idx in 2..6u32 {
        let inc = ctx_idx_inc_mb_type_bin(
            &dec.neighbors, mb_x, ctx_offset::MB_TYPE_B_PREFIX, bin_idx, &prefix_bins,
        );
        let bin = dec.decode_dec(ctx_offset::MB_TYPE_B_PREFIX + inc)?;
        prefix_bins.push(bin);
        v = (v << 1) | bin as u32;
    }

    if v < 8 {
        // v ∈ [0, 7] → mb_type = v + 3 (3..10: B_Bi_16x16 + 16x8/8x16
        // L0/L1/Bi family up to B_L1_L0_16x8).
        return Ok(v + 3);
    }
    if v == 13 {
        // Intra-in-B: decode the I-suffix at ctxIdxOffset=32.
        // Same bin shape as decode_mb_type_i: bin0 short-circuits
        // I_NxN, bin1 is TERMINATE for I_PCM, bins 2..n encode I_16x16.
        let mut suffix: Vec<u8> = Vec::with_capacity(7);
        let s_inc0 = ctx_idx_inc_mb_type_bin(
            &dec.neighbors, mb_x, ctx_offset::MB_TYPE_B_SUFFIX, 0, &suffix,
        );
        let s_bin0 = dec.decode_dec(ctx_offset::MB_TYPE_B_SUFFIX + s_inc0)?;
        suffix.push(s_bin0);
        if s_bin0 == 0 {
            return Ok(23); // 23 + I_NxN(0) = 23
        }
        let s_bin1 = dec.decode_terminate()?;
        suffix.push(s_bin1);
        if s_bin1 == 1 {
            return Ok(48); // 23 + I_PCM(25) = 48
        }
        let s_inc2 = ctx_idx_inc_mb_type_bin(
            &dec.neighbors, mb_x, ctx_offset::MB_TYPE_B_SUFFIX, 2, &suffix,
        );
        let s_bin2 = dec.decode_dec(ctx_offset::MB_TYPE_B_SUFFIX + s_inc2)?;
        suffix.push(s_bin2);
        let s_inc3 = ctx_idx_inc_mb_type_bin(
            &dec.neighbors, mb_x, ctx_offset::MB_TYPE_B_SUFFIX, 3, &suffix,
        );
        let s_bin3 = dec.decode_dec(ctx_offset::MB_TYPE_B_SUFFIX + s_inc3)?;
        suffix.push(s_bin3);
        let n_remaining = if s_bin3 == 0 { 2 } else { 3 };
        for bin_idx in 4..(4 + n_remaining) {
            let inc = ctx_idx_inc_mb_type_bin(
                &dec.neighbors, mb_x, ctx_offset::MB_TYPE_B_SUFFIX, bin_idx as u32, &suffix,
            );
            let bin = dec.decode_dec(ctx_offset::MB_TYPE_B_SUFFIX + inc)?;
            suffix.push(bin);
        }
        let suffix_value = match_mb_type_i_bins(&suffix);
        return Ok(23 + suffix_value);
    }
    if v == 14 {
        return Ok(11); // B_Bi_8x16
    }
    if v == 15 {
        return Ok(22); // B_8x8
    }

    // v ∈ {8, 9, 10, 11, 12}: read bin6, mb_type = (v<<1 | bin6) - 4.
    let inc6 = ctx_idx_inc_mb_type_bin(
        &dec.neighbors, mb_x, ctx_offset::MB_TYPE_B_PREFIX, 6, &prefix_bins,
    );
    let bin6 = dec.decode_dec(ctx_offset::MB_TYPE_B_PREFIX + inc6)?;
    Ok(((v << 1) | bin6 as u32).saturating_sub(4))
}

/// Decode `mb_type` for a P-slice. Returns 0..=30 (skipping 4 which
/// is forbidden in CABAC). Spec Table 9-37 prefix + I-suffix for
/// intra-in-P.
pub fn decode_mb_type_p(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
) -> Result<u32, DecodeError> {
    let mut prefix_bins: Vec<u8> = Vec::with_capacity(3);

    // Prefix bin 0 — disambiguates P-partition (0) vs intra-in-P (1).
    let inc0 = ctx_idx_inc_mb_type_bin(
        &dec.neighbors, mb_x, ctx_offset::MB_TYPE_P_PREFIX, 0, &prefix_bins,
    );
    let bin0 = dec.decode_dec(ctx_offset::MB_TYPE_P_PREFIX + inc0)?;
    prefix_bins.push(bin0);

    if bin0 == 1 {
        // Intra-in-P. Decode the I-suffix at ctxIdxOffset=17.
        // Same shape as decode_mb_type_i, but with a different
        // ctxIdxOffset and starting fresh prior_bins (the I-suffix
        // bins use their own prior_bins, NOT the prefix bin).
        let mut suffix: Vec<u8> = Vec::with_capacity(7);
        let s_inc0 = ctx_idx_inc_mb_type_bin(
            &dec.neighbors, mb_x, ctx_offset::MB_TYPE_P_SUFFIX, 0, &suffix,
        );
        let s_bin0 = dec.decode_dec(ctx_offset::MB_TYPE_P_SUFFIX + s_inc0)?;
        suffix.push(s_bin0);
        if s_bin0 == 0 {
            return Ok(5); // I_NxN suffix value 0 → mb_type 5
        }
        // bin 1 of I-suffix is TERMINATE.
        let s_bin1 = dec.decode_terminate()?;
        suffix.push(s_bin1);
        if s_bin1 == 1 {
            return Ok(30); // I_PCM suffix value 25 → mb_type 30
        }
        // Continue regular bins.
        let s_inc2 = ctx_idx_inc_mb_type_bin(
            &dec.neighbors, mb_x, ctx_offset::MB_TYPE_P_SUFFIX, 2, &suffix,
        );
        let s_bin2 = dec.decode_dec(ctx_offset::MB_TYPE_P_SUFFIX + s_inc2)?;
        suffix.push(s_bin2);
        let s_inc3 = ctx_idx_inc_mb_type_bin(
            &dec.neighbors, mb_x, ctx_offset::MB_TYPE_P_SUFFIX, 3, &suffix,
        );
        let s_bin3 = dec.decode_dec(ctx_offset::MB_TYPE_P_SUFFIX + s_inc3)?;
        suffix.push(s_bin3);
        let n_remaining = if s_bin3 == 0 { 2 } else { 3 };
        for bin_idx in 4..(4 + n_remaining) {
            let inc = ctx_idx_inc_mb_type_bin(
                &dec.neighbors, mb_x, ctx_offset::MB_TYPE_P_SUFFIX, bin_idx as u32, &suffix,
            );
            let bin = dec.decode_dec(ctx_offset::MB_TYPE_P_SUFFIX + inc)?;
            suffix.push(bin);
        }
        let suffix_value = match_mb_type_i_bins(&suffix);
        return Ok(5 + suffix_value);
    }

    // P-partition (bin 0 == 0). Read bins 1, 2 of the prefix and
    // disambiguate via the prefix table.
    let inc1 = ctx_idx_inc_mb_type_bin(
        &dec.neighbors, mb_x, ctx_offset::MB_TYPE_P_PREFIX, 1, &prefix_bins,
    );
    let bin1 = dec.decode_dec(ctx_offset::MB_TYPE_P_PREFIX + inc1)?;
    prefix_bins.push(bin1);
    let inc2 = ctx_idx_inc_mb_type_bin(
        &dec.neighbors, mb_x, ctx_offset::MB_TYPE_P_PREFIX, 2, &prefix_bins,
    );
    let bin2 = dec.decode_dec(ctx_offset::MB_TYPE_P_PREFIX + inc2)?;
    prefix_bins.push(bin2);

    // Match against P-prefix table:
    //   [0,0,0] → 0 (P_L0_16x16)
    //   [0,1,1] → 1 (P_L0_L0_16x8)
    //   [0,1,0] → 2 (P_L0_L0_8x16)
    //   [0,0,1] → 3 (P_8x8)
    Ok(match (bin1, bin2) {
        (0, 0) => 0,
        (1, 1) => 1,
        (1, 0) => 2,
        (0, 1) => 3,
        _ => unreachable!(),
    })
}

/// Decode `sub_mb_type` for a P-slice. Spec Table 9-38 P rows,
/// ctxIdxOffset=21 with static increments {0, 1, 2}.
pub fn decode_sub_mb_type_p(dec: &mut CabacDecoder<'_>) -> Result<u32, DecodeError> {
    // bin 0: ctxIdxInc = 0
    let bin0 = dec.decode_dec(ctx_offset::SUB_MB_TYPE_P)?;
    if bin0 == 1 {
        return Ok(0); // P_L0_8x8
    }
    // bin 1: ctxIdxInc = 1
    let bin1 = dec.decode_dec(ctx_offset::SUB_MB_TYPE_P + 1)?;
    if bin1 == 0 {
        return Ok(1); // P_L0_8x4
    }
    // bin 2: ctxIdxInc = 2
    let bin2 = dec.decode_dec(ctx_offset::SUB_MB_TYPE_P + 2)?;
    Ok(if bin2 == 1 { 2 } else { 3 })
}

/// §6E-A6.3 — decode `sub_mb_type` for a B-slice (uniform 8x8 family,
/// values 0..=3). Spec Table 9-38 B rows, ctxIdxOffset 36; ctxIdxInc
/// from Table 9-39 with bin 2 path-dependent on bin 1.
///
/// **Scope**: §6E-A6.3 ships sub_mb_types 0..=3 (`B_Direct_8x8`,
/// `B_L0_8x8`, `B_L1_8x8`, `B_Bi_8x8`). The remaining 9 sub-sub-
/// partition variants (4..=12, descoped per the x264-medium finding)
/// return `DecodeError::Unsupported` until §6E-A6.4.
pub fn decode_sub_mb_type_b(dec: &mut CabacDecoder<'_>) -> Result<u32, DecodeError> {
    let base = ctx_offset::SUB_MB_TYPE_B;
    // bin 0 (ctxIdxInc 0): 0 → B_Direct_8x8.
    let bin0 = dec.decode_dec(base)?;
    if bin0 == 0 {
        return Ok(0);
    }
    // bin 1 (ctxIdxInc 1): 0 → L0/L1 tail; 1 → Bi/sub-sub family.
    let bin1 = dec.decode_dec(base + 1)?;
    if bin1 == 0 {
        // bin 2 (ctxIdxInc 3): 0 → B_L0_8x8 (1); 1 → B_L1_8x8 (2).
        let bin2 = dec.decode_dec(base + 3)?;
        return Ok(if bin2 == 0 { 1 } else { 2 });
    }
    // bin 2 (ctxIdxInc 2): 0 → B_Bi_8x8 / sub_mb_type 4..=6 family;
    //                     1 → sub_mb_type 7..=12 family (out of scope).
    let bin2 = dec.decode_dec(base + 2)?;
    if bin2 == 1 {
        return Err(DecodeError::Unsupported(
            "B-slice sub_mb_type 7..=12 (sub-sub partitions, lands in §6E-A6.4)",
        ));
    }
    // bin 3 (ctxIdxInc 3): 0 → B_Bi_8x8 (3) / B_L0_8x4 (4); 1 → 5/6.
    let bin3 = dec.decode_dec(base + 3)?;
    // bin 4 (ctxIdxInc 3): completes the 3 + 2*bin3 + bin4 mapping.
    let bin4 = dec.decode_dec(base + 3)?;
    let value = 3 + 2 * (bin3 as u32) + (bin4 as u32);
    if value == 3 {
        Ok(3) // B_Bi_8x8 — only in-scope value of this branch
    } else {
        Err(DecodeError::Unsupported(
            "B-slice sub_mb_type 4..=6 (sub-sub partitions, lands in §6E-A6.4)",
        ))
    }
}

// ─── MVD decoder (with stego-position emission) ─────────────────

/// Per-MB position context passed to decoders that emit stego
/// positions. Carries the (frame, mb_addr) coordinate plus the
/// PositionLogger to emit into.
pub struct PositionCtx<'a> {
    pub frame_idx: u32,
    pub mb_addr: u32,
    pub logger: &'a mut dyn PositionLogger,
}

/// Decode `mvd_lX[compIdx]` (UEG3, ctxIdxOffset 40 / 47). Mirror of
/// `cabac::encoder::encode_mvd`. Returns the signed mvd value.
///
/// Emits stego positions:
/// - `MvdSuffixLsb` at the final bin of the Exp-Golomb suffix
///   (when |mvd| ≥ 9 and a suffix is present).
/// - `MvdSignBypass` at the sign bypass bin (when mvd ≠ 0).
///
/// The two emitted [`SyntaxPath::Mvd`] keys are distinguished by
/// [`BinKind`] (`Sign` vs `SuffixLsb`).
pub fn decode_mvd(
    dec: &mut CabacDecoder<'_>,
    component: u8,
    mb_x: usize,
    block_idx_in_mb_a: usize,
    block_idx_in_mb_b: usize,
    list: u8,
    partition: u8,
    pos_ctx: &mut PositionCtx<'_>,
) -> Result<i32, DecodeError> {
    let bin0_inc = ctx_idx_inc_mvd_bin0(
        &dec.neighbors,
        mb_x,
        block_idx_in_mb_a,
        block_idx_in_mb_b,
        component,
    );
    decode_mvd_with_bin0_inc(dec, component, list, partition, bin0_inc, pos_ctx)
}

/// Variant of [`decode_mvd`] that accepts a pre-computed bin-0
/// ctxIdxInc. Mirror of `encode_mvd_with_bin0_inc`.
pub fn decode_mvd_with_bin0_inc(
    dec: &mut CabacDecoder<'_>,
    component: u8,
    list: u8,
    partition: u8,
    bin0_ctx_idx_inc: u32,
    pos_ctx: &mut PositionCtx<'_>,
) -> Result<i32, DecodeError> {
    debug_assert!(component <= 1);
    let u_coff = 9u32;
    let base_offset = if component == 0 {
        ctx_offset::MVD_L0_X
    } else {
        ctx_offset::MVD_L0_Y
    };

    // TU prefix at base_offset+ctxIdxInc.
    let prefix_val = decode_tu(dec, u_coff, |bin_idx| {
        let ctx_inc = match bin_idx {
            0 => bin0_ctx_idx_inc,
            1 => 3,
            2 => 4,
            3 => 5,
            _ => 6,
        };
        base_offset + ctx_inc
    })?;

    // Suffix: only when prefix saturated at u_coff (= |mvd| ≥ 9).
    let abs_v = if prefix_val < u_coff {
        prefix_val
    } else {
        let axis = if component == 0 { Axis::X } else { Axis::Y };
        let suffix_val = decode_egk_suffix_emit_lsb(
            dec,
            /* k_init */ 3,
            pos_ctx,
            EmbedDomain::MvdSuffixLsb,
            |kind| SyntaxPath::Mvd { list, partition, axis, kind },
        )?;
        u_coff + suffix_val
    };

    // Sign: only when abs_v != 0 (one bypass bin). Emit MvdSignBypass.
    let signed = if abs_v == 0 {
        0i32
    } else {
        let key = PositionKey::new(
            pos_ctx.frame_idx,
            pos_ctx.mb_addr,
            EmbedDomain::MvdSignBypass,
            SyntaxPath::Mvd {
                list,
                partition,
                axis: if component == 0 { Axis::X } else { Axis::Y },
                kind: BinKind::Sign,
            },
        );
        pos_ctx.logger.register(key);
        let sign_bin = dec.decode_bypass()?;
        if sign_bin == 0 { abs_v as i32 } else { -(abs_v as i32) }
    };
    Ok(signed)
}

/// Decode an Exp-Golomb-k bypass suffix mirroring
/// `binarize_egk_suffix`. Returns the suf_s value.
///
/// Emits a PositionKey with the caller-specified domain + SyntaxPath
/// at the **final** suffix bin (the LSB of the suffix at the
/// increased k after the unary terminator). The caller decides
/// what BinKind / SyntaxPath / EmbedDomain to attach.
fn decode_egk_suffix_emit_lsb(
    dec: &mut CabacDecoder<'_>,
    k_init: u32,
    pos_ctx: &mut PositionCtx<'_>,
    lsb_domain: EmbedDomain,
    lsb_path_for_kind: impl FnOnce(BinKind) -> SyntaxPath,
) -> Result<u32, DecodeError> {
    let mut k = k_init;
    let mut prefix_offset = 0u32;
    loop {
        let bin = dec.decode_bypass()?;
        if bin == 0 {
            break;
        }
        prefix_offset += 1u32 << k;
        k += 1;
    }
    let mut suffix_value = 0u32;
    if k > 0 {
        let path = lsb_path_for_kind(BinKind::SuffixLsb);
        for i in (0..k).rev() {
            if i == 0 {
                let key = PositionKey::new(
                    pos_ctx.frame_idx,
                    pos_ctx.mb_addr,
                    lsb_domain,
                    path,
                );
                pos_ctx.logger.register(key);
            }
            let bin = dec.decode_bypass()?;
            suffix_value |= (bin as u32) << i;
        }
    }
    Ok(prefix_offset + suffix_value)
}

// ─── Residual block decoder (with stego-position emission) ──────

/// Decode a residual block at ctxBlockCat 0..=4 (4×4 luma residuals,
/// chroma DC/AC, Intra_16x16 luma DC/AC). Mirror of
/// `cabac::encoder::encode_residual_block_cabac_with_cbf_inc`.
///
/// Returns the decoded coefficient vector indexed by scan position
/// (length = `end_idx + 1`). Positions before `start_idx` are zero.
///
/// Emits stego positions:
/// - `CoeffSuffixLsb` at the LSB bin of each Exp-Golomb-0 suffix
///   (only when |level| ≥ 15, i.e. abs_level_minus1 ≥ 14 saturates).
/// - `CoeffSignBypass` at every nonzero coefficient's sign bypass bin.
///
/// SyntaxPath for emitted positions is built via `path_for_kind`
/// closure: caller decides whether this block is Luma4x4, ChromaAc,
/// ChromaDc, or LumaDcIntra16x16, and supplies the appropriate
/// constructor.
#[allow(clippy::too_many_arguments)]
pub fn decode_residual_block_cabac(
    dec: &mut CabacDecoder<'_>,
    start_idx: usize,
    end_idx: usize,
    ctx_block_cat: u8,
    cbf_ctx_idx_inc: u32,
    pos_ctx: &mut PositionCtx<'_>,
    mut path_for_coeff: impl FnMut(u8, BinKind) -> SyntaxPath,
) -> Result<Vec<i32>, DecodeError> {
    debug_assert!(ctx_block_cat <= 4);
    let mut coeffs = vec![0i32; end_idx + 1];

    // 1. coded_block_flag.
    let cbf_ctx_idx = ctx_offset::CODED_BLOCK_FLAG_LOW
        + CTX_BLOCK_CAT_OFFSET[0][ctx_block_cat as usize]
        + cbf_ctx_idx_inc;
    let cbf = dec.decode_dec(cbf_ctx_idx)?;
    if cbf == 0 {
        return Ok(coeffs);
    }

    // 2. Significance map (forward scan).
    let sig_offset = ctx_offset::SIGNIFICANT_COEFF_FLAG_FRAME_LOW
        + CTX_BLOCK_CAT_OFFSET[1][ctx_block_cat as usize];
    let last_offset = ctx_offset::LAST_SIGNIFICANT_COEFF_FLAG_FRAME_LOW
        + CTX_BLOCK_CAT_OFFSET[2][ctx_block_cat as usize];

    let mut sig_indices: Vec<usize> = Vec::with_capacity(end_idx + 1 - start_idx);
    let mut num_coeff = end_idx + 1;
    let mut i = start_idx;
    while i < num_coeff - 1 {
        let level_list_idx = (i - start_idx) as u32;
        let ctx_sig_inc = if ctx_block_cat == 3 {
            ctx_idx_inc_sig_chroma_dc(level_list_idx)
        } else {
            ctx_idx_inc_sig_4x4(level_list_idx)
        };
        let sig_bin = dec.decode_dec(sig_offset + ctx_sig_inc)?;
        if sig_bin != 0 {
            sig_indices.push(i);
            let ctx_last_inc = if ctx_block_cat == 3 {
                ctx_idx_inc_sig_chroma_dc(level_list_idx)
            } else {
                ctx_idx_inc_sig_4x4(level_list_idx)
            };
            let last_bin = dec.decode_dec(last_offset + ctx_last_inc)?;
            if last_bin != 0 {
                num_coeff = i + 1;
                break;
            }
        }
        i += 1;
    }
    // Position num_coeff-1 is implicitly significant.
    if sig_indices.last().copied().is_none_or(|x| x != num_coeff - 1) {
        sig_indices.push(num_coeff - 1);
    }

    // 3. Reverse-scan abs_level + sign emit.
    let abs_offset = ctx_offset::COEFF_ABS_LEVEL_MINUS1_LOW
        + CTX_BLOCK_CAT_OFFSET[3][ctx_block_cat as usize];
    let mut num_eq1 = 0u32;
    let mut num_gt1 = 0u32;

    // Walk in reverse over sig_indices (which is naturally forward-
    // sorted as we appended in scan order).
    for &i in sig_indices.iter().rev() {
        // TU prefix decode: read up to 14 ones, terminated by 0 OR
        // saturated at 14.
        let mut prefix_len = 0u32;
        loop {
            let ctx_inc = ctx_idx_inc_coeff_abs_level(
                if prefix_len == 0 { 0 } else { 1 },
                ctx_block_cat,
                num_eq1,
                num_gt1,
            );
            let bin = dec.decode_dec(abs_offset + ctx_inc)?;
            if bin == 0 {
                break;
            }
            prefix_len += 1;
            if prefix_len == 14 {
                break;
            }
        }
        // If saturated, decode EG0 suffix (bypass) — emits CoeffSuffixLsb.
        let abs_level_minus1 = if prefix_len < 14 {
            prefix_len
        } else {
            let coeff_idx = i as u8;
            let suffix_val = decode_egk_suffix_emit_lsb(
                dec,
                /* k_init */ 0,
                pos_ctx,
                EmbedDomain::CoeffSuffixLsb,
                |kind| path_for_coeff(coeff_idx, kind),
            )?;
            14 + suffix_val
        };
        let abs_level = abs_level_minus1 + 1;

        // coeff_sign_flag (bypass) — emit CoeffSignBypass.
        let sign_path = path_for_coeff(i as u8, BinKind::Sign);
        let sign_key = PositionKey::new(
            pos_ctx.frame_idx,
            pos_ctx.mb_addr,
            EmbedDomain::CoeffSignBypass,
            sign_path,
        );
        pos_ctx.logger.register(sign_key);
        let sign_bin = dec.decode_bypass()?;
        coeffs[i] = if sign_bin == 0 { abs_level as i32 } else { -(abs_level as i32) };

        if abs_level == 1 {
            num_eq1 += 1;
        } else {
            num_gt1 += 1;
        }
    }

    Ok(coeffs)
}

/// Decode a cat=5 Luma 8×8 residual block (no per-block CBF — caller
/// must decide whether this block is present via cbp_luma).
/// Returns the 64-coefficient zigzag array. Mirror of
/// `cabac::encoder::encode_residual_block_cabac_8x8`.
///
/// Emits stego positions for each nonzero coefficient:
/// - `CoeffSuffixLsb` at the LSB bin of any Exp-Golomb-0 suffix
///   (when |level| ≥ 15).
/// - `CoeffSignBypass` at every coefficient's sign bypass bin.
///
/// `path_for_coeff` closure builds the SyntaxPath::Luma8x8
/// (block_idx is provided by the caller when constructing the path).
pub fn decode_residual_block_cabac_8x8(
    dec: &mut CabacDecoder<'_>,
    pos_ctx: &mut PositionCtx<'_>,
    mut path_for_coeff: impl FnMut(u8, BinKind) -> SyntaxPath,
) -> Result<[i32; 64], DecodeError> {
    let mut coeffs = [0i32; 64];

    // 1. Sig-map over 63 positions. The 64th is implicit-significant.
    let mut sig_indices: Vec<usize> = Vec::with_capacity(64);
    let mut num_coeff = 64;
    let mut i = 0usize;
    while i < 63 {
        let level_list_idx = i;
        let sig_ctx = cat5_luma8x8::SIG_BASE
            + SIG_COEFF_FLAG_OFFSET_8X8_FRAME[level_list_idx] as u32;
        let sig_bin = dec.decode_dec(sig_ctx)?;
        if sig_bin != 0 {
            sig_indices.push(i);
            let last_ctx = cat5_luma8x8::LAST_BASE
                + LAST_COEFF_FLAG_OFFSET_8X8_FRAME[level_list_idx] as u32;
            let last_bin = dec.decode_dec(last_ctx)?;
            if last_bin != 0 {
                num_coeff = i + 1;
                break;
            }
        }
        i += 1;
    }
    if sig_indices.last().copied().is_none_or(|x| x != num_coeff - 1) {
        sig_indices.push(num_coeff - 1);
    }

    // 2. Reverse-scan abs_level + sign emit.
    let mut num_eq1 = 0u32;
    let mut num_gt1 = 0u32;
    for &i in sig_indices.iter().rev() {
        let mut prefix_len = 0u32;
        loop {
            let ctx_inc = ctx_idx_inc_coeff_abs_level(
                if prefix_len == 0 { 0 } else { 1 },
                /* ctx_block_cat */ 5,
                num_eq1,
                num_gt1,
            );
            let bin = dec.decode_dec(cat5_luma8x8::ABS_BASE + ctx_inc)?;
            if bin == 0 {
                break;
            }
            prefix_len += 1;
            if prefix_len == 14 {
                break;
            }
        }
        let abs_level_minus1 = if prefix_len < 14 {
            prefix_len
        } else {
            let coeff_idx = i as u8;
            let suffix_val = decode_egk_suffix_emit_lsb(
                dec,
                /* k_init */ 0,
                pos_ctx,
                EmbedDomain::CoeffSuffixLsb,
                |kind| path_for_coeff(coeff_idx, kind),
            )?;
            14 + suffix_val
        };
        let abs_level = abs_level_minus1 + 1;

        let sign_path = path_for_coeff(i as u8, BinKind::Sign);
        let sign_key = PositionKey::new(
            pos_ctx.frame_idx,
            pos_ctx.mb_addr,
            EmbedDomain::CoeffSignBypass,
            sign_path,
        );
        pos_ctx.logger.register(sign_key);
        let sign_bin = dec.decode_bypass()?;
        coeffs[i] = if sign_bin == 0 { abs_level as i32 } else { -(abs_level as i32) };

        if abs_level == 1 {
            num_eq1 += 1;
        } else {
            num_gt1 += 1;
        }
    }
    Ok(coeffs)
}

/// Decode `coded_block_flag` for a residual block (spec
/// § 9.3.3.1.1.9 + Table 9-39).
pub fn decode_coded_block_flag(
    dec: &mut CabacDecoder<'_>,
    ctx_block_cat: u8,
    mb_x: usize,
    block_idx_in_mb_a: usize,
    block_idx_in_mb_b: usize,
    current_is_intra: bool,
) -> Result<bool, DecodeError> {
    let ctx_inc = ctx_idx_inc_coded_block_flag(
        &dec.neighbors,
        mb_x,
        ctx_block_cat,
        block_idx_in_mb_a,
        block_idx_in_mb_b,
        current_is_intra,
    );
    let ctx_idx = ctx_offset::CODED_BLOCK_FLAG_LOW
        + CTX_BLOCK_CAT_OFFSET[0][ctx_block_cat as usize]
        + ctx_inc;
    let bin = dec.decode_dec(ctx_idx)?;
    Ok(bin != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::cabac::context::CabacInitSlot;
    use crate::codec::h264::cabac::encoder as cenc;
    use crate::codec::h264::cabac::CabacEncoder;

    fn fresh_enc() -> CabacEncoder {
        CabacEncoder::new_slice(CabacInitSlot::ISI, 26, 4)
    }

    fn fresh_p_enc() -> CabacEncoder {
        CabacEncoder::new_slice(CabacInitSlot::PIdc0, 26, 4)
    }

    fn finish_then_decode<'a>(
        enc: CabacEncoder,
        slot: CabacInitSlot,
    ) -> Vec<u8> {
        // Caller terminates and finishes the encoder; we just hand
        // back the bytes. The decoder is constructed by the caller.
        let _ = slot;
        enc.finish()
    }

    fn decoder_from(bytes: &[u8], slot: CabacInitSlot) -> CabacDecoder<'_> {
        CabacDecoder::new_slice(bytes, slot, 26, 4).expect("init")
    }

    #[test]
    fn mb_skip_flag_zero_roundtrip() {
        let mut enc = fresh_p_enc();
        cenc::encode_mb_skip_flag(&mut enc, false, 0);
        enc.engine.encode_terminate(1);
        let bytes = finish_then_decode(enc, CabacInitSlot::PIdc0);
        let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
        let v = decode_mb_skip_flag(&mut dec, 0).unwrap();
        assert!(!v);
    }

    #[test]
    fn mb_skip_flag_one_roundtrip() {
        let mut enc = fresh_p_enc();
        cenc::encode_mb_skip_flag(&mut enc, true, 0);
        enc.engine.encode_terminate(1);
        let bytes = finish_then_decode(enc, CabacInitSlot::PIdc0);
        let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
        let v = decode_mb_skip_flag(&mut dec, 0).unwrap();
        assert!(v);
    }

    #[test]
    fn mb_qp_delta_zero_roundtrip() {
        let mut enc = fresh_enc();
        cenc::encode_mb_qp_delta(&mut enc, 0);
        enc.engine.encode_terminate(1);
        let bytes = finish_then_decode(enc, CabacInitSlot::ISI);
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let v = decode_mb_qp_delta(&mut dec).unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn mb_qp_delta_positive_roundtrip() {
        for &dq in &[1, 3, 5, 10, 25] {
            let mut enc = fresh_enc();
            cenc::encode_mb_qp_delta(&mut enc, dq);
            enc.engine.encode_terminate(1);
            let bytes = finish_then_decode(enc, CabacInitSlot::ISI);
            let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
            let v = decode_mb_qp_delta(&mut dec).unwrap();
            assert_eq!(v, dq, "dq={dq}");
        }
    }

    #[test]
    fn mb_qp_delta_negative_roundtrip() {
        for &dq in &[-1, -3, -5, -10, -25] {
            let mut enc = fresh_enc();
            cenc::encode_mb_qp_delta(&mut enc, dq);
            enc.engine.encode_terminate(1);
            let bytes = finish_then_decode(enc, CabacInitSlot::ISI);
            let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
            let v = decode_mb_qp_delta(&mut dec).unwrap();
            assert_eq!(v, dq, "dq={dq}");
        }
    }

    #[test]
    fn mb_qp_delta_remap_roundtrip() {
        for dq in -26..=25i32 {
            let mapped = crate::codec::h264::cabac::binarization::mb_qp_delta_remap(dq);
            let unmapped = mb_qp_delta_unmap(mapped);
            assert_eq!(unmapped, dq, "dq={dq} mapped={mapped}");
        }
    }

    #[test]
    fn intra_chroma_pred_mode_roundtrip() {
        for mode in 0u8..=3 {
            let mut enc = fresh_enc();
            cenc::encode_intra_chroma_pred_mode(&mut enc, mode, 0);
            enc.engine.encode_terminate(1);
            let bytes = finish_then_decode(enc, CabacInitSlot::ISI);
            let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
            let v = decode_intra_chroma_pred_mode(&mut dec, 0).unwrap();
            assert_eq!(v, mode);
        }
    }

    #[test]
    fn prev_intra4x4_pred_mode_flag_roundtrip() {
        for &flag in &[true, false] {
            let mut enc = fresh_enc();
            cenc::encode_prev_intra4x4_pred_mode_flag(&mut enc, flag);
            enc.engine.encode_terminate(1);
            let bytes = finish_then_decode(enc, CabacInitSlot::ISI);
            let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
            let v = decode_prev_intra4x4_pred_mode_flag(&mut dec).unwrap();
            assert_eq!(v, flag);
        }
    }

    #[test]
    fn rem_intra4x4_pred_mode_roundtrip() {
        for rem in 0u8..8 {
            let mut enc = fresh_enc();
            cenc::encode_rem_intra4x4_pred_mode(&mut enc, rem);
            enc.engine.encode_terminate(1);
            let bytes = finish_then_decode(enc, CabacInitSlot::ISI);
            let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
            let v = decode_rem_intra4x4_pred_mode(&mut dec).unwrap();
            assert_eq!(v, rem, "rem={rem}");
        }
    }

    #[test]
    fn end_of_slice_flag_roundtrip() {
        let mut enc = fresh_enc();
        cenc::encode_end_of_slice_flag(&mut enc, true);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        assert!(decode_end_of_slice_flag(&mut dec).unwrap());
    }

    #[test]
    fn ref_idx_zero_roundtrip() {
        let mut enc = fresh_p_enc();
        cenc::encode_ref_idx(&mut enc, 0, 0, 0, 0);
        enc.engine.encode_terminate(1);
        let bytes = finish_then_decode(enc, CabacInitSlot::PIdc0);
        let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
        let v = decode_ref_idx(&mut dec, 0, 0, 0).unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn ref_idx_nonzero_roundtrip() {
        for &idx in &[1u32, 3, 7] {
            let mut enc = fresh_p_enc();
            cenc::encode_ref_idx(&mut enc, idx, 0, 0, 0);
            enc.engine.encode_terminate(1);
            let bytes = finish_then_decode(enc, CabacInitSlot::PIdc0);
            let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
            let v = decode_ref_idx(&mut dec, 0, 0, 0).unwrap();
            assert_eq!(v, idx);
        }
    }

    #[test]
    fn coded_block_pattern_zero_roundtrip() {
        let mut enc = fresh_enc();
        cenc::encode_coded_block_pattern(&mut enc, 0, 0);
        enc.engine.encode_terminate(1);
        let bytes = finish_then_decode(enc, CabacInitSlot::ISI);
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let v = decode_coded_block_pattern(&mut dec, 0).unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn coded_block_pattern_full_roundtrip() {
        // CBP = 0x2F (luma=0xF, chroma=2 = "all coded").
        let mut enc = fresh_enc();
        cenc::encode_coded_block_pattern(&mut enc, 0x2F, 0);
        enc.engine.encode_terminate(1);
        let bytes = finish_then_decode(enc, CabacInitSlot::ISI);
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let v = decode_coded_block_pattern(&mut dec, 0).unwrap();
        assert_eq!(v, 0x2F);
    }

    #[test]
    fn coded_block_pattern_mixed_roundtrip() {
        for cbp in [0x05u8, 0x0A, 0x13, 0x1C, 0x27] {
            let mut enc = fresh_enc();
            cenc::encode_coded_block_pattern(&mut enc, cbp, 0);
            enc.engine.encode_terminate(1);
            let bytes = finish_then_decode(enc, CabacInitSlot::ISI);
            let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
            let v = decode_coded_block_pattern(&mut dec, 0).unwrap();
            assert_eq!(v, cbp, "cbp=0x{cbp:02x}");
        }
    }

    #[test]
    fn coded_block_flag_roundtrip() {
        for &is_coded in &[false, true] {
            let mut enc = fresh_enc();
            cenc::encode_coded_block_flag(
                &mut enc,
                is_coded,
                /* ctx_block_cat */ 1,
                0, 0, 0,
                /* current_is_intra */ true,
            );
            enc.engine.encode_terminate(1);
            let bytes = finish_then_decode(enc, CabacInitSlot::ISI);
            let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
            let v = decode_coded_block_flag(&mut dec, 1, 0, 0, 0, true).unwrap();
            assert_eq!(v, is_coded);
        }
    }

    #[test]
    fn transform_size_8x8_flag_roundtrip() {
        for &flag in &[false, true] {
            let mut enc = fresh_enc();
            cenc::encode_transform_size_8x8_flag(&mut enc, flag, 0);
            enc.engine.encode_terminate(1);
            let bytes = finish_then_decode(enc, CabacInitSlot::ISI);
            let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
            let v = decode_transform_size_8x8_flag(&mut dec, 0).unwrap();
            assert_eq!(v, flag);
        }
    }

    #[test]
    fn unary_decode_terminates_at_zero() {
        // Manually encode 5 ones + 0 = unary 5.
        let mut enc = CabacEncoder::new_slice(CabacInitSlot::ISI, 26, 4);
        let ctx_idx = ctx_offset::MB_QP_DELTA;
        for _ in 0..5 {
            enc.encode_dec(1, ctx_idx);
        }
        enc.encode_dec(0, ctx_idx);
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let v = decode_unary(&mut dec, 100, |_| ctx_idx).unwrap();
        assert_eq!(v, 5);
    }

    #[test]
    fn tu_decode_saturated() {
        // TU cMax=3 saturated case: emit 3 ones (NO trailing zero) →
        // TU value = c_max = 3.
        let mut enc = CabacEncoder::new_slice(CabacInitSlot::ISI, 26, 4);
        let ctx_idx = ctx_offset::INTRA_CHROMA_PRED_MODE;
        for _ in 0..3 {
            enc.encode_dec(1, ctx_idx);
        }
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let v = decode_tu(&mut dec, 3, |_| ctx_idx).unwrap();
        assert_eq!(v, 3);
    }

    #[test]
    fn mb_type_i_inxn_roundtrip() {
        let mut enc = fresh_enc();
        cenc::encode_mb_type_i(&mut enc, 0, 0);
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let v = decode_mb_type_i(&mut dec, 0).unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn mb_type_i_ipcm_roundtrip() {
        let mut enc = fresh_enc();
        cenc::encode_mb_type_i(&mut enc, 25, 0);
        // I_PCM doesn't continue with end_of_slice; emit terminate(1)
        // through the flush path manually since encode_mb_type_i for
        // I_PCM already terminated bin 1.
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let v = decode_mb_type_i(&mut dec, 0).unwrap();
        assert_eq!(v, 25);
    }

    #[test]
    fn mb_type_i_i16x16_roundtrip_all_values() {
        for mb_type in 1u32..=24 {
            let mut enc = fresh_enc();
            cenc::encode_mb_type_i(&mut enc, mb_type, 0);
            enc.engine.encode_terminate(1);
            let bytes = enc.finish();
            let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
            let v = decode_mb_type_i(&mut dec, 0).unwrap();
            assert_eq!(v, mb_type, "mb_type={mb_type}");
        }
    }

    #[test]
    fn mb_type_p_partition_roundtrip() {
        // P-partition values 0, 1, 2, 3 (skipping 4 which is forbidden).
        for mb_type in [0u32, 1, 2, 3] {
            let mut enc = fresh_p_enc();
            cenc::encode_mb_type_p(&mut enc, mb_type, 0);
            enc.engine.encode_terminate(1);
            let bytes = enc.finish();
            let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
            let v = decode_mb_type_p(&mut dec, 0).unwrap();
            assert_eq!(v, mb_type, "P-partition mb_type={mb_type}");
        }
    }

    #[test]
    fn mb_type_p_intra_in_p_inxn_roundtrip() {
        // Intra-in-P with I_NxN suffix → mb_type = 5.
        let mut enc = fresh_p_enc();
        cenc::encode_mb_type_p(&mut enc, 5, 0);
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
        let v = decode_mb_type_p(&mut dec, 0).unwrap();
        assert_eq!(v, 5);
    }

    #[test]
    fn mb_type_p_intra_in_p_i16x16_roundtrip() {
        // Sample of intra-in-P I_16x16 mb_type values.
        for mb_type in [6u32, 10, 14, 18, 24, 29] {
            let mut enc = fresh_p_enc();
            cenc::encode_mb_type_p(&mut enc, mb_type, 0);
            enc.engine.encode_terminate(1);
            let bytes = enc.finish();
            let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
            let v = decode_mb_type_p(&mut dec, 0).unwrap();
            assert_eq!(v, mb_type, "intra-in-P mb_type={mb_type}");
        }
    }

    #[test]
    fn mb_type_p_intra_in_p_ipcm_roundtrip() {
        // intra-in-P I_PCM is mb_type = 5 + 25 = 30.
        let mut enc = fresh_p_enc();
        cenc::encode_mb_type_p(&mut enc, 30, 0);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
        let v = decode_mb_type_p(&mut dec, 0).unwrap();
        assert_eq!(v, 30);
    }

    // ─── §6E-A4 B-slice mb_type round-trip ──────────────────────

    fn fresh_b_enc() -> CabacEncoder {
        // B-slice uses the same PIdc{0,1,2} init slots as P/SP.
        CabacEncoder::new_slice(CabacInitSlot::PIdc0, 26, 4)
    }

    /// §6E-A4 — B_Direct_16x16 round-trip (single-bin
    /// short-circuit).
    #[test]
    fn mb_type_b_direct_roundtrip() {
        let mut enc = fresh_b_enc();
        cenc::encode_mb_type_b(&mut enc, 0, 0);
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
        let v = decode_mb_type_b(&mut dec, 0).unwrap();
        assert_eq!(v, 0);
    }

    /// §6E-A4 — B_L0_16x16 / B_L1_16x16 round-trip (3-bin tree).
    #[test]
    fn mb_type_b_l0_l1_roundtrip() {
        for mb_type in [1u32, 2] {
            let mut enc = fresh_b_enc();
            cenc::encode_mb_type_b(&mut enc, mb_type, 0);
            enc.engine.encode_terminate(1);
            let bytes = enc.finish();
            let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
            let v = decode_mb_type_b(&mut dec, 0).unwrap();
            assert_eq!(v, mb_type, "B-slice 16x16 mb_type={mb_type}");
        }
    }

    /// §6E-A4 — B_Bi_16x16 round-trip (multi-partition v=0 path,
    /// 6 bins).
    #[test]
    fn mb_type_b_bi_16x16_roundtrip() {
        let mut enc = fresh_b_enc();
        cenc::encode_mb_type_b(&mut enc, 3, 0);
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
        let v = decode_mb_type_b(&mut dec, 0).unwrap();
        assert_eq!(v, 3);
    }

    /// §6E-A4 — B_8x8 round-trip (multi-partition v=15 short-circuit,
    /// 6 bins).
    #[test]
    fn mb_type_b_8x8_roundtrip() {
        let mut enc = fresh_b_enc();
        cenc::encode_mb_type_b(&mut enc, 22, 0);
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
        let v = decode_mb_type_b(&mut dec, 0).unwrap();
        assert_eq!(v, 22);
    }

    /// §6E-A4 — B-slice 16x8 / 8x16 partition family
    /// (mb_types 4..=10) round-trip via the multi-partition v
    /// ∈ [0, 7] path.
    #[test]
    fn mb_type_b_partition_4_to_10_roundtrip() {
        for mb_type in 4u32..=10 {
            let mut enc = fresh_b_enc();
            cenc::encode_mb_type_b(&mut enc, mb_type, 0);
            enc.engine.encode_terminate(1);
            let bytes = enc.finish();
            let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
            let v = decode_mb_type_b(&mut dec, 0).unwrap();
            assert_eq!(v, mb_type, "B-slice partition mb_type={mb_type}");
        }
    }

    #[test]
    fn sub_mb_type_p_all_values() {
        for sub in 0u32..=3 {
            let mut enc = fresh_p_enc();
            cenc::encode_sub_mb_type_p(&mut enc, sub);
            enc.engine.encode_terminate(1);
            let bytes = enc.finish();
            let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
            let v = decode_sub_mb_type_p(&mut dec).unwrap();
            assert_eq!(v, sub, "sub_mb_type={sub}");
        }
    }

    #[test]
    fn mvd_zero_no_sign_position() {
        let mut enc = fresh_p_enc();
        cenc::encode_mvd(&mut enc, 0, /* component */ 0, 0, 0, 0);
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
        let mut recorder = crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let v = decode_mvd(&mut dec, 0, 0, 0, 0, 0, 0, &mut ctx).unwrap();
        assert_eq!(v, 0);
        // mvd=0 → no sign bin emitted, no positions logged.
        assert_eq!(recorder.positions.len(), 0);
    }

    #[test]
    fn mvd_small_emits_only_sign_position() {
        for &mvd in &[1i32, -1, 5, -5, 8, -8] {
            let mut enc = fresh_p_enc();
            cenc::encode_mvd(&mut enc, mvd, 0, 0, 0, 0);
            enc.engine.encode_terminate(1);
            let bytes = enc.finish();
            let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
            let mut recorder =
                crate::codec::h264::stego::PositionRecorder::new();
            let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
            let v = decode_mvd(&mut dec, 0, 0, 0, 0, 0, 0, &mut ctx).unwrap();
            assert_eq!(v, mvd, "mvd={mvd}");
            // |mvd| < 9 → no suffix; only sign bypass emitted.
            assert_eq!(recorder.positions.len(), 1, "mvd={mvd}");
            assert_eq!(
                recorder.positions[0].domain(),
                EmbedDomain::MvdSignBypass,
            );
        }
    }

    #[test]
    fn mvd_saturated_emits_suffix_and_sign_positions() {
        // |mvd| >= 9 → both suffix LSB AND sign bins emitted.
        for &mvd in &[9i32, -9, 17, -17, 100, -100] {
            let mut enc = fresh_p_enc();
            cenc::encode_mvd(&mut enc, mvd, 0, 0, 0, 0);
            enc.engine.encode_terminate(1);
            let bytes = enc.finish();
            let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
            let mut recorder =
                crate::codec::h264::stego::PositionRecorder::new();
            let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
            let v = decode_mvd(&mut dec, 0, 0, 0, 0, 0, 0, &mut ctx).unwrap();
            assert_eq!(v, mvd, "mvd={mvd}");
            // 2 positions: SuffixLsb then SignBypass.
            assert_eq!(recorder.positions.len(), 2, "mvd={mvd}");
            assert_eq!(recorder.positions[0].domain(), EmbedDomain::MvdSuffixLsb);
            assert_eq!(recorder.positions[1].domain(), EmbedDomain::MvdSignBypass);
        }
    }

    #[test]
    fn mvd_position_keys_have_correct_axis_and_kind() {
        // Component 0 (X) and 1 (Y) must produce different SyntaxPath
        // keys.
        let mut enc = fresh_p_enc();
        cenc::encode_mvd(&mut enc, 5, 0, 0, 0, 0);
        cenc::encode_mvd(&mut enc, 5, 1, 0, 0, 0);
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let _ = decode_mvd(&mut dec, 0, 0, 0, 0, 0, 0, &mut ctx).unwrap();
        let _ = decode_mvd(&mut dec, 1, 0, 0, 0, 0, 0, &mut ctx).unwrap();
        assert_eq!(recorder.positions.len(), 2);
        // Two MvdSignBypass keys, but with different axes → distinct.
        assert_ne!(recorder.positions[0], recorder.positions[1]);
        // Both should be MvdSignBypass.
        assert_eq!(recorder.positions[0].domain(), EmbedDomain::MvdSignBypass);
        assert_eq!(recorder.positions[1].domain(), EmbedDomain::MvdSignBypass);
        match recorder.positions[0].syntax_path() {
            SyntaxPath::Mvd { axis: Axis::X, kind: BinKind::Sign, .. } => (),
            other => panic!("expected MVD X+Sign, got {other:?}"),
        }
        match recorder.positions[1].syntax_path() {
            SyntaxPath::Mvd { axis: Axis::Y, kind: BinKind::Sign, .. } => (),
            other => panic!("expected MVD Y+Sign, got {other:?}"),
        }
    }

    #[test]
    fn residual_block_all_zero_emits_only_cbf() {
        // Build a 16-coeff scan with all zeros. Encoder emits CBF=0
        // and stops. Decoder reads CBF=0, returns zero vec, no stego
        // positions emitted.
        let mut enc = fresh_enc();
        let scan = vec![0i32; 16];
        let r = cenc::encode_residual_block_cabac(
            &mut enc, &scan, 0, 15, 1, 0, 0, 0, true,
        );
        assert!(!r, "encoder reports no nonzero");
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 1, 0, 0, true,
        );
        let coeffs = decode_residual_block_cabac(
            &mut dec, 0, 15, 1, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::Luma4x4 {
                block_idx: 0, coeff_idx, kind,
            },
        ).unwrap();
        assert_eq!(coeffs, scan);
        assert_eq!(recorder.positions.len(), 0);
    }

    #[test]
    fn residual_block_single_coeff_roundtrip() {
        // Single nonzero at scan position 0, value 3.
        let mut enc = fresh_enc();
        let mut scan = vec![0i32; 16];
        scan[0] = 3;
        cenc::encode_residual_block_cabac(
            &mut enc, &scan, 0, 15, 1, 0, 0, 0, true,
        );
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 1, 0, 0, true,
        );
        let coeffs = decode_residual_block_cabac(
            &mut dec, 0, 15, 1, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::Luma4x4 {
                block_idx: 0, coeff_idx, kind,
            },
        ).unwrap();
        assert_eq!(coeffs, scan);
        // Single nonzero |level|=3 < 15 → no suffix LSB. Just one
        // sign bypass position emitted.
        assert_eq!(recorder.positions.len(), 1);
        assert_eq!(recorder.positions[0].domain(), EmbedDomain::CoeffSignBypass);
    }

    #[test]
    fn residual_block_negative_coeff_roundtrip() {
        let mut enc = fresh_enc();
        let mut scan = vec![0i32; 16];
        scan[2] = -7;
        cenc::encode_residual_block_cabac(
            &mut enc, &scan, 0, 15, 1, 0, 0, 0, true,
        );
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 1, 0, 0, true,
        );
        let coeffs = decode_residual_block_cabac(
            &mut dec, 0, 15, 1, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::Luma4x4 {
                block_idx: 0, coeff_idx, kind,
            },
        ).unwrap();
        assert_eq!(coeffs, scan);
        assert_eq!(recorder.positions.len(), 1);
    }

    #[test]
    fn residual_block_multiple_coeffs_roundtrip() {
        let mut enc = fresh_enc();
        let mut scan = vec![0i32; 16];
        scan[0] = 5;
        scan[1] = -3;
        scan[3] = 1;
        scan[5] = -2;
        cenc::encode_residual_block_cabac(
            &mut enc, &scan, 0, 15, 1, 0, 0, 0, true,
        );
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 1, 0, 0, true,
        );
        let coeffs = decode_residual_block_cabac(
            &mut dec, 0, 15, 1, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::Luma4x4 {
                block_idx: 0, coeff_idx, kind,
            },
        ).unwrap();
        assert_eq!(coeffs, scan);
        // 4 nonzeros, all |level| < 15 → 4 sign positions, 0 suffix.
        assert_eq!(recorder.positions.len(), 4);
        for key in &recorder.positions {
            assert_eq!(key.domain(), EmbedDomain::CoeffSignBypass);
        }
    }

    #[test]
    fn residual_block_large_coeff_emits_suffix_lsb_position() {
        // |level|=20 → abs_level_minus1=19 ≥ 14 → EG0 suffix path.
        let mut enc = fresh_enc();
        let mut scan = vec![0i32; 16];
        scan[0] = 20;
        cenc::encode_residual_block_cabac(
            &mut enc, &scan, 0, 15, 1, 0, 0, 0, true,
        );
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 1, 0, 0, true,
        );
        let coeffs = decode_residual_block_cabac(
            &mut dec, 0, 15, 1, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::Luma4x4 {
                block_idx: 0, coeff_idx, kind,
            },
        ).unwrap();
        assert_eq!(coeffs, scan);
        // Expect 1 suffix LSB + 1 sign = 2 positions.
        assert_eq!(recorder.positions.len(), 2);
        assert_eq!(recorder.positions[0].domain(), EmbedDomain::CoeffSuffixLsb);
        assert_eq!(recorder.positions[1].domain(), EmbedDomain::CoeffSignBypass);
    }

    #[test]
    fn residual_block_8x8_single_coeff_roundtrip() {
        let mut enc = fresh_enc();
        let mut scan = [0i32; 64];
        scan[7] = 4;
        cenc::encode_residual_block_cabac_8x8(&mut enc, &scan);
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let coeffs = decode_residual_block_cabac_8x8(
            &mut dec, &mut ctx,
            |coeff_idx, kind| SyntaxPath::Luma8x8 {
                block_idx: 0, coeff_idx, kind,
            },
        ).unwrap();
        assert_eq!(coeffs, scan);
        assert_eq!(recorder.positions.len(), 1);
        assert_eq!(recorder.positions[0].domain(), EmbedDomain::CoeffSignBypass);
        match recorder.positions[0].syntax_path() {
            SyntaxPath::Luma8x8 { block_idx: 0, coeff_idx: 7, kind: BinKind::Sign } => (),
            other => panic!("wrong path {other:?}"),
        }
    }

    #[test]
    fn residual_block_chroma_dc_roundtrip() {
        // ctxBlockCat=3 → uses ctx_idx_inc_sig_chroma_dc, different
        // ctx_block_cat_offset entry. Chroma DC is a 2x2 Hadamard
        // (start_idx=0, end_idx=3, 4 coefficients).
        let mut enc = fresh_enc();
        let mut scan = vec![0i32; 4];
        scan[0] = 5;
        scan[2] = -2;
        cenc::encode_residual_block_cabac(
            &mut enc, &scan, 0, 3, 3, 0, 0, 0, true,
        );
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 3, 0, 0, true,
        );
        let coeffs = decode_residual_block_cabac(
            &mut dec, 0, 3, 3, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::ChromaDc {
                plane: 0, coeff_idx, kind,
            },
        ).unwrap();
        assert_eq!(coeffs, scan);
        assert_eq!(recorder.positions.len(), 2);
    }

    #[test]
    fn residual_block_luma_dc_intra16_roundtrip() {
        // ctxBlockCat=0 → LumaDcIntra16x16 (16-coeff Hadamard 4x4 of
        // DC-of-each-4x4 in the macroblock).
        let mut enc = fresh_enc();
        let mut scan = vec![0i32; 16];
        scan[0] = 100; // |level| > 14 → exercises the EG0 suffix path
        scan[3] = -3;
        cenc::encode_residual_block_cabac(
            &mut enc, &scan, 0, 15, 0, 0, 0, 0, true,
        );
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 0, 0, 0, true,
        );
        let coeffs = decode_residual_block_cabac(
            &mut dec, 0, 15, 0, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::LumaDcIntra16x16 {
                coeff_idx, kind,
            },
        ).unwrap();
        assert_eq!(coeffs, scan);
        // |level=100| > 14 → 1 SuffixLsb, plus 2 sign positions.
        assert_eq!(recorder.positions.len(), 3);
        let suffix_count = recorder
            .positions
            .iter()
            .filter(|k| k.domain() == EmbedDomain::CoeffSuffixLsb)
            .count();
        let sign_count = recorder
            .positions
            .iter()
            .filter(|k| k.domain() == EmbedDomain::CoeffSignBypass)
            .count();
        assert_eq!(suffix_count, 1);
        assert_eq!(sign_count, 2);
    }

    #[test]
    fn residual_block_luma_ac_intra16_cat2_roundtrip() {
        // ctxBlockCat=2 → Intra16x16ACLevel. start_idx=1 (DC is
        // separate at cat=0), end_idx=15 (15 AC coeffs). Production
        // encoder emits cat=2 for I_16x16 MBs with non-zero AC.
        let mut enc = fresh_enc();
        let mut scan = vec![0i32; 16];
        scan[1] = 6;
        scan[3] = -2;
        scan[8] = 1;
        cenc::encode_residual_block_cabac(
            &mut enc, &scan, 1, 15, 2, 0, 0, 0, true,
        );
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 2, 0, 0, true,
        );
        let coeffs = decode_residual_block_cabac(
            &mut dec, 1, 15, 2, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::LumaDcIntra16x16 {
                coeff_idx, kind,
            },
        ).unwrap();
        assert_eq!(coeffs, scan);
        assert_eq!(recorder.positions.len(), 3);
        for key in &recorder.positions {
            assert_eq!(key.domain(), EmbedDomain::CoeffSignBypass);
        }
    }

    #[test]
    fn residual_block_chroma_ac_roundtrip() {
        // ctxBlockCat=4 → ChromaAc (start_idx=1, end_idx=15, 15
        // AC coefficients).
        let mut enc = fresh_enc();
        let mut scan = vec![0i32; 16];
        scan[1] = 4;
        scan[7] = -1;
        cenc::encode_residual_block_cabac(
            &mut enc, &scan, 1, 15, 4, 0, 0, 0, true,
        );
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 4, 0, 0, true,
        );
        let coeffs = decode_residual_block_cabac(
            &mut dec, 1, 15, 4, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::ChromaAc {
                plane: 1, block_idx: 0, coeff_idx, kind,
            },
        ).unwrap();
        assert_eq!(coeffs, scan);
        assert_eq!(recorder.positions.len(), 2);
    }

    #[test]
    fn residual_block_8x8_multiple_coeffs_roundtrip() {
        let mut enc = fresh_enc();
        let mut scan = [0i32; 64];
        scan[0] = 3;
        scan[1] = -2;
        scan[5] = 1;
        scan[10] = -4;
        cenc::encode_residual_block_cabac_8x8(&mut enc, &scan);
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let coeffs = decode_residual_block_cabac_8x8(
            &mut dec, &mut ctx,
            |coeff_idx, kind| SyntaxPath::Luma8x8 {
                block_idx: 1, coeff_idx, kind,
            },
        ).unwrap();
        assert_eq!(coeffs, scan);
        // 4 nonzeros, all |level| < 15 → 4 sign positions, 0 suffix.
        assert_eq!(recorder.positions.len(), 4);
    }

    /// Phase 6D.3 sign-off gate: a planned sign-bit pattern survives
    /// the full round-trip through the encoder + decoder when
    /// applied via [`apply_coeff_sign_overrides`], AND the position
    /// keys the decoder logs match the keys
    /// [`enumerate_coeff_sign_positions`] produces on the same input.
    ///
    /// Three properties verified:
    ///   1. Coefficient magnitudes are byte-identical between cover
    ///      and stego (bypass-bin invariant; Phase 6D.0
    ///      cabac-bypass-bin-stego.md Theorem 1).
    ///   2. Decoder-side `PositionRecorder` produces exactly the
    ///      same `PositionKey` sequence (same keys, same order) as
    ///      the encoder-side `enumerate_coeff_sign_positions`.
    ///   3. The decoded coefficient signs match the injected plan.
    #[test]
    fn coeff_sign_bypass_full_roundtrip_with_position_parity() {
        use crate::codec::h264::stego::{
            apply_coeff_sign_overrides, enumerate_coeff_sign_positions,
            BitInjector, PositionRecorder,
        };

        // Original cover: a residual block with 4 mixed-sign coeffs.
        let cover = [5i32, -3, 0, 7, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut scan = cover.to_vec();

        let path_for_coeff = |coeff_idx: u8| SyntaxPath::Luma4x4 {
            block_idx: 0, coeff_idx, kind: BinKind::Sign,
        };

        // Encoder-side enumeration (what the stego planner sees).
        let cover_positions = enumerate_coeff_sign_positions(
            &scan, 0, 15, 0, 0, &path_for_coeff,
        );
        assert_eq!(cover_positions.len(), 4);

        // Non-trivial injection plan: alternate the bit value across
        // positions in emit order.
        struct AlternateInjector { count: u32 }
        impl BitInjector for AlternateInjector {
            fn override_bit(&mut self, _key: PositionKey) -> Option<u8> {
                let bit = (self.count & 1) as u8;
                self.count += 1;
                Some(bit)
            }
        }
        let mut injector = AlternateInjector { count: 0 };
        let _flips = apply_coeff_sign_overrides(
            &mut scan, 0, 15, 0, 0, &path_for_coeff, &mut injector,
        );

        // Encode the modified scan.
        let mut enc = fresh_enc();
        cenc::encode_residual_block_cabac(
            &mut enc, &scan, 0, 15, 1, 0, 0, 0, true,
        );
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();

        // Decode + record positions.
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder = PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 1, 0, 0, true,
        );
        let decoded = decode_residual_block_cabac(
            &mut dec, 0, 15, 1, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::Luma4x4 {
                block_idx: 0, coeff_idx, kind,
            },
        ).unwrap();

        // (1) Coefficient roundtrip: decoded == post-injection scan.
        assert_eq!(decoded, scan, "decoded coeffs must match modified scan");

        // (2) Position-key parity: encoder enumerate == decoder
        //     emission (same keys, same order). 6D.3 sign-off gate.
        assert_eq!(
            recorder.positions, cover_positions,
            "decoder position keys must match encoder enumeration",
        );

        // (3) Magnitude preservation: bypass-bin stego must NOT
        //     change coefficient magnitudes (Theorem 1).
        for i in 0..16 {
            assert_eq!(
                scan[i].unsigned_abs(),
                cover[i].unsigned_abs(),
                "scan[{i}] magnitude must be preserved",
            );
        }
    }

    /// Phase 6D.5 sign-off gate: `MvdSignBypass` end-to-end paired
    /// roundtrip + position parity. Same shape as 6D.3 coeff gate.
    #[test]
    fn mvd_sign_bypass_full_roundtrip_with_position_parity() {
        use crate::codec::h264::stego::{
            apply_mvd_sign_overrides, enumerate_mvd_sign_positions,
            BitInjector, MvdSlot, PositionRecorder,
        };

        // Three MVD slots (e.g. P_8x8 partitions): mixed signs.
        let mut slots = vec![
            MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 5 },
            MvdSlot { list: 0, partition: 1, axis: Axis::X, value: -3 },
            MvdSlot { list: 0, partition: 2, axis: Axis::X, value: 7 },
        ];

        let cover_positions = enumerate_mvd_sign_positions(&slots, 0, 0);
        assert_eq!(cover_positions.len(), 3);

        // Plan: alternating override.
        struct AlternateInjector { count: u32 }
        impl BitInjector for AlternateInjector {
            fn override_bit(&mut self, _key: PositionKey) -> Option<u8> {
                let bit = (self.count & 1) as u8;
                self.count += 1;
                Some(bit)
            }
        }
        let mut injector = AlternateInjector { count: 0 };
        apply_mvd_sign_overrides(&mut slots, 0, 0, &mut injector);

        // Encode each MVD as a stand-alone test (no full MB walker).
        let mut enc = fresh_p_enc();
        for s in &slots {
            cenc::encode_mvd(&mut enc, s.value, /* component */ 0, 0, 0, 0);
        }
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();

        // Decode + record positions.
        let mut dec = decoder_from(&bytes, CabacInitSlot::PIdc0);
        let mut recorder = PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        for s in &slots {
            let v = decode_mvd(&mut dec, 0, 0, 0, 0, s.list, s.partition, &mut ctx).unwrap();
            assert_eq!(v, s.value, "MVD roundtrip for slot {s:?}");
        }

        // Position parity: only sign bypass keys (all values < 9 so
        // no suffix LSB). Encoder enumerate == decoder emission.
        let recorded_signs: Vec<PositionKey> = recorder
            .positions
            .iter()
            .filter(|k| k.domain() == EmbedDomain::MvdSignBypass)
            .copied()
            .collect();
        assert_eq!(recorded_signs, cover_positions);
    }

    /// Phase 6D.6 sign-off gate: `CoeffSuffixLsb` end-to-end paired
    /// roundtrip + position parity. Verifies magnitude changes by ±1
    /// (NOT preserved, unlike sign-only domains), and that the
    /// decoder reads back exactly the suffix LSB bits implied by
    /// the modified magnitudes.
    #[test]
    fn coeff_suffix_lsb_full_roundtrip_with_position_parity() {
        use crate::codec::h264::stego::{
            apply_coeff_suffix_lsb_overrides, enumerate_coeff_suffix_lsb_positions,
            extract_coeff_suffix_lsb_bits, BitInjector, PositionRecorder,
        };

        // Three large coefficients (|coeff| ≥ 16, eligibility threshold).
        let mut scan = vec![0i32; 16];
        scan[0] = 20;
        scan[3] = -25;
        scan[7] = 18;
        // Plus a sub-threshold coeff (not eligible).
        scan[10] = 5;

        let path_for_coeff = |coeff_idx: u8| SyntaxPath::Luma4x4 {
            block_idx: 0, coeff_idx, kind: BinKind::SuffixLsb,
        };

        let cover_positions = enumerate_coeff_suffix_lsb_positions(
            &scan, 0, 15, 0, 0, &path_for_coeff,
        );
        // Three eligible (≥ 16); one sub-threshold.
        assert_eq!(cover_positions.len(), 3);

        // Plan: alternating override bits.
        struct AltInjector { count: u32 }
        impl BitInjector for AltInjector {
            fn override_bit(&mut self, _key: PositionKey) -> Option<u8> {
                let bit = (self.count & 1) as u8;
                self.count += 1;
                Some(bit)
            }
        }
        let mut injector = AltInjector { count: 0 };
        apply_coeff_suffix_lsb_overrides(
            &mut scan, 0, 15, 0, 0, &path_for_coeff, &mut injector,
        );

        // Encode + decode.
        let mut enc = fresh_enc();
        cenc::encode_residual_block_cabac(
            &mut enc, &scan, 0, 15, 1, 0, 0, 0, true,
        );
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder = PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 1, 0, 0, true,
        );
        let decoded = decode_residual_block_cabac(
            &mut dec, 0, 15, 1, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::Luma4x4 {
                block_idx: 0, coeff_idx, kind,
            },
        ).unwrap();

        // Coefficient roundtrip.
        assert_eq!(decoded, scan);

        // Position-key parity for the SuffixLsb domain only.
        let recorded_suffix: Vec<PositionKey> = recorder
            .positions
            .iter()
            .filter(|k| k.domain() == EmbedDomain::CoeffSuffixLsb)
            .copied()
            .collect();
        assert_eq!(recorded_suffix, cover_positions);

        // Bits parity: extracted suffix LSB bits == decoder-side
        // bits read from the modified scan.
        let cover_bits = extract_coeff_suffix_lsb_bits(&scan, 0, 15);
        // Extract decoder-side bits: walk recorded_suffix order +
        // compute LSB(magnitude) of decoded coeffs.
        for (idx, key) in recorded_suffix.iter().enumerate() {
            let coeff_idx = match key.syntax_path() {
                SyntaxPath::Luma4x4 { coeff_idx, .. } => coeff_idx,
                _ => panic!("wrong path"),
            };
            let abs = decoded[coeff_idx as usize].unsigned_abs() as u32;
            let decoder_bit = ((abs & 1) ^ 1) as u8;
            assert_eq!(decoder_bit, cover_bits[idx],
                "suffix LSB at coeff_idx={coeff_idx} mismatch");
        }
    }

    /// Variant of the 6D.3 gate that exercises a custom plan
    /// HashMap<PositionKey, u8> (mirrors the eventual STC bit-plan
    /// lookup pattern). Verifies that the decoder reads back exactly
    /// the bit pattern the injector planted.
    #[test]
    fn coeff_sign_bypass_decoded_bits_match_injected_plan() {
        use crate::codec::h264::stego::{
            apply_coeff_sign_overrides, enumerate_coeff_sign_positions,
            extract_coeff_sign_bits, BitInjector, PositionRecorder,
        };

        let mut scan = vec![0i32; 16];
        scan[1] = 4; scan[5] = -2; scan[9] = 1; scan[13] = -8;

        let path_for_coeff = |coeff_idx: u8| SyntaxPath::Luma4x4 {
            block_idx: 0, coeff_idx, kind: BinKind::Sign,
        };

        let cover_positions = enumerate_coeff_sign_positions(
            &scan, 0, 15, 0, 0, &path_for_coeff,
        );
        // Choose a deterministic per-key bit value.
        let plan: std::collections::HashMap<PositionKey, u8> = cover_positions
            .iter()
            .enumerate()
            .map(|(idx, &k)| (k, ((idx * 3 + 1) & 1) as u8))
            .collect();

        struct PlanInjector(std::collections::HashMap<PositionKey, u8>);
        impl BitInjector for PlanInjector {
            fn override_bit(&mut self, key: PositionKey) -> Option<u8> {
                self.0.get(&key).copied()
            }
        }
        let mut injector = PlanInjector(plan.clone());

        apply_coeff_sign_overrides(
            &mut scan, 0, 15, 0, 0, &path_for_coeff, &mut injector,
        );

        // Encode + decode.
        let mut enc = fresh_enc();
        cenc::encode_residual_block_cabac(
            &mut enc, &scan, 0, 15, 1, 0, 0, 0, true,
        );
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder = PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 0, mb_addr: 0, logger: &mut recorder };
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 1, 0, 0, true,
        );
        let decoded = decode_residual_block_cabac(
            &mut dec, 0, 15, 1, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::Luma4x4 {
                block_idx: 0, coeff_idx, kind,
            },
        ).unwrap();

        // Extract the sign bits the decoder would see, and confirm
        // they match the injected plan exactly.
        let decoded_bits = extract_coeff_sign_bits(&decoded, 0, 15);
        let plan_bits: Vec<u8> = recorder.positions.iter()
            .map(|k| plan[k])
            .collect();
        assert_eq!(decoded_bits, plan_bits);
    }

    #[test]
    fn integration_multi_element_mb_roundtrip() {
        // Encode a sequence of disparate elements that a real I-slice
        // MB walker would emit, then decode them back in the same
        // order. Verifies that:
        //  - per-element decoders chain correctly without state drift,
        //  - PositionRecorder captures the expected total set of
        //    stego positions across multiple emitting decoders,
        //  - the encoder's bin sequence matches the decoder's exactly.
        let mut enc = fresh_enc();

        // 1. mb_type_i = 1 (I_16x16 mode 0, cbp_chroma=0, cbp_luma=0)
        cenc::encode_mb_type_i(&mut enc, 1, 0);
        // 2. intra_chroma_pred_mode = 2
        cenc::encode_intra_chroma_pred_mode(&mut enc, 2, 0);
        // 3. mb_qp_delta = -2
        cenc::encode_mb_qp_delta(&mut enc, -2);
        // 4. coded_block_pattern = 0x07 (luma 0x7, chroma 0)
        cenc::encode_coded_block_pattern(&mut enc, 0x07, 0);
        // 5. residual block: 4x4 luma at scan[2] = -8 (level 8, sign bypass)
        let mut scan = vec![0i32; 16];
        scan[2] = -8;
        cenc::encode_residual_block_cabac(
            &mut enc, &scan, 0, 15, /*cat*/ 1, 0, 0, 0, true,
        );
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();

        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let mut recorder =
            crate::codec::h264::stego::PositionRecorder::new();
        let mut ctx = PositionCtx { frame_idx: 5, mb_addr: 17, logger: &mut recorder };

        assert_eq!(decode_mb_type_i(&mut dec, 0).unwrap(), 1);
        assert_eq!(decode_intra_chroma_pred_mode(&mut dec, 0).unwrap(), 2);
        assert_eq!(decode_mb_qp_delta(&mut dec).unwrap(), -2);
        assert_eq!(decode_coded_block_pattern(&mut dec, 0).unwrap(), 0x07);
        let cbf_inc = ctx_idx_inc_coded_block_flag(
            &dec.neighbors, 0, 1, 0, 0, true,
        );
        let out = decode_residual_block_cabac(
            &mut dec, 0, 15, 1, cbf_inc, &mut ctx,
            |coeff_idx, kind| SyntaxPath::Luma4x4 {
                block_idx: 0, coeff_idx, kind,
            },
        ).unwrap();
        assert_eq!(out, scan);

        // Only the residual decoder emits stego positions in this
        // sequence — it's the only call that received a logger.
        // |level|=8 < 15 → no suffix; just one sign bypass.
        assert_eq!(recorder.positions.len(), 1);
        assert_eq!(recorder.positions[0].domain(), EmbedDomain::CoeffSignBypass);
        assert_eq!(recorder.positions[0].frame_idx(), 5);
        assert_eq!(recorder.positions[0].mb_addr(), 17);
        match recorder.positions[0].syntax_path() {
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 2, kind: BinKind::Sign } => (),
            other => panic!("wrong path {other:?}"),
        }

        // Final terminate sees end_of_slice = true.
        assert!(decode_end_of_slice_flag(&mut dec).unwrap());
    }

    #[test]
    fn fl_lsb_first_decode() {
        // 3-bit FL LSB-first encoding of 5 = 0b101 → emit bits LSB-first
        // = 1, 0, 1.
        let mut enc = CabacEncoder::new_slice(CabacInitSlot::ISI, 26, 4);
        let ctx_idx = ctx_offset::REM_INTRA_PRED_MODE;
        enc.encode_dec(1, ctx_idx); // bit 0 (LSB)
        enc.encode_dec(0, ctx_idx); // bit 1
        enc.encode_dec(1, ctx_idx); // bit 2 (MSB)
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = decoder_from(&bytes, CabacInitSlot::ISI);
        let v = decode_fl(&mut dec, 3, true, |_| ctx_idx).unwrap();
        assert_eq!(v, 5);
    }
}
