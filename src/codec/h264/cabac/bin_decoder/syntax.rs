// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Per-syntax-element CABAC decoders. Each inverts the spec-defined
// binarization (ITU-T H.264 § 9.3) that OpenH264's CABAC emitter
// produces, reading the same bins in the same ctxIdx order. The
// only forward binarizer still living in `cabac::binarization` is the
// Table 9-36 I-slice `mb_type` bin table (`mb_type_i_bins`); the rest
// are spec § 9.3.2 binarizations this module inverts inline.
//
// Each decode fn:
//   1. Reads the bins for one syntax element, using the spec's
//      per-bin ctx_idx derivation and the same neighbor state the
//      emitter used.
//   2. Inverts the binarization to reconstruct the original syntax
//      value.
//   3. Returns the decoded value (Result-wrapped for I/O errors).
//
// The MVD + residual_block_cabac decoders additionally emit stego
// positions (PositionKeys) for the bypass-bin cover domains; see the
// position-tracker module.

use crate::codec::h264::cabac::ctx_offsets::{
    cat5_luma8x8, ctx_offset, CTX_BLOCK_CAT_OFFSET,
    LAST_COEFF_FLAG_OFFSET_8X8_FRAME, SIG_COEFF_FLAG_OFFSET_8X8_FRAME,
};
use crate::codec::h264::cabac::neighbor::{
    ctx_idx_inc_cbp_chroma, ctx_idx_inc_coeff_abs_level,
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
/// the **MSB** for general FL (spec § 9.3.2.6 emits MSB-first).
///
/// HOWEVER, a few syntax elements use the opposite (LSB-first) FL
/// ordering — notably `rem_intra4x4_pred_mode` (spec § 9.3.2.4) and
/// the `coded_block_pattern` luma prefix (spec § 9.3.2.6 / Table
/// 9-39 FL nibble). The decoder uses an `lsb_first` flag to switch
/// between the two orderings.
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

/// Inverse of the forward mb_qp_delta remap: unsigned mapped value → signed.
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

/// Decode `mb_skip_flag` (B-slice). Same neighbor derivation rule
/// as P (spec § 9.3.3.1.1.1; both P and B use
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

/// Decode `ref_idx_lX` (regular unary, terminator-based; ctxIdxOffset=54).
/// Spec § 9.3.2 Table 9-34 row "ref_idx_lX" type U; matches the spec
/// spec § 9.3.2.2 ref_idx binarization. `_c_max` retained on signature for caller
/// uniformity but unused in body (no truncation).
///
/// `current_mb` + `(cur_bx, cur_by)` replaces the prior
/// `(block_idx_in_mb_a, block_idx_in_mb_b)` parameter pair.
/// Caller fills `current_mb` AFTER this returns so subsequent partitions
/// in the same MB read the just-decoded ref_idx for within-MB
/// neighbour lookups (spec § 6.4.11.7).
pub fn decode_ref_idx(
    dec: &mut CabacDecoder<'_>,
    current_mb: &crate::codec::h264::cabac::neighbor::CurrentMbRefIdx,
    mb_x: usize,
    cur_bx: u8,
    cur_by: u8,
    _c_max: u32,
) -> Result<u32, DecodeError> {
    let bin0_inc = crate::codec::h264::cabac::neighbor::compute_ref_idx_ctx_idx_inc_bin0(
        current_mb,
        &dec.neighbors,
        mb_x,
        cur_bx,
        cur_by,
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

/// Derive ctxIdxInc for an `mb_type` bin per spec § 9.3.3.1 / Table
/// 9-39. For bin 0 uses the neighbor derivation; for bin 1 the caller
/// must route to `decode_terminate` (this fn is not consulted); for
/// bins 2+ uses the prior-bin table or the Table 9-39 static
/// fallback.
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
        // B-slice mb_type prefix bins per spec Table 9-41 / the
        // reference decoder.
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

/// Decode `mb_type` for a B-slice. Returns 0..=22 for non-intra
/// mb_types, or 23..=47 for intra-in-B. Inverts the spec Table
/// 9-37 B-slice mb_type binarization.
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

/// Decode `sub_mb_type` for a B-slice (uniform 8x8 family,
/// values 0..=3). Spec Table 9-38 B rows, ctxIdxOffset 36; ctxIdxInc
/// from Table 9-39 with bin 2 path-dependent on bin 1.
///
/// **Scope**: ships sub_mb_types 0..=3 (`B_Direct_8x8`,
/// `B_L0_8x8`, `B_L1_8x8`, `B_Bi_8x8`). The remaining 9
/// sub-partition variants (4..=12) are out of scope and
/// return `DecodeError::Unsupported`.
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
///
/// `nal_idx` identifies which slice NAL in the Annex-B stream
/// contains this MB's bins, plumbed into
/// `PositionLogger::register_with_offset` so the walker can capture
/// per-position byte/bit offsets aligned to NAL boundaries (used by
/// the bitstream-mod splicer).
pub struct PositionCtx<'a> {
    pub frame_idx: u32,
    pub mb_addr: u32,
    pub nal_idx: u32,
    pub logger: &'a mut dyn PositionLogger,
}

/// Variant of [`decode_mvd`] that accepts a pre-computed bin-0
/// ctxIdxInc.
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
        // Capture RBSP bit offset BEFORE the bypass read.
        let off = dec.next_rbsp_bit_offset();
        pos_ctx.logger.register_with_offset(key, off, pos_ctx.nal_idx);
        let sign_bin = dec.decode_bypass()?;
        if sign_bin == 0 { abs_v as i32 } else { -(abs_v as i32) }
    };
    Ok(signed)
}

/// Decode an Exp-Golomb-k bypass suffix (spec § 9.3.2.3 EGk).
/// Returns the suf_s value.
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
                // Capture RBSP bit offset BEFORE the bypass read.
                let off = dec.next_rbsp_bit_offset();
                pos_ctx.logger.register_with_offset(key, off, pos_ctx.nal_idx);
            }
            let bin = dec.decode_bypass()?;
            suffix_value |= (bin as u32) << i;
        }
    }
    Ok(prefix_offset + suffix_value)
}

// ─── Residual block decoder (with stego-position emission) ──────

/// Decode a residual block at ctxBlockCat 0..=4 (4×4 luma residuals,
/// chroma DC/AC, Intra_16x16 luma DC/AC). Inverts the
/// `residual_block_cabac` parse (spec § 9.3.3.1.3).
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
        // Capture RBSP bit offset BEFORE the bypass read.
        let off = dec.next_rbsp_bit_offset();
        pos_ctx.logger.register_with_offset(sign_key, off, pos_ctx.nal_idx);
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
/// Returns the 64-coefficient zigzag array. Inverts the
/// `residual_block_cabac` parse for the 8×8 transform (spec
/// § 9.3.3.1.3).
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
        // Capture RBSP bit offset BEFORE the bypass read.
        let off = dec.next_rbsp_bit_offset();
        pos_ctx.logger.register_with_offset(sign_key, off, pos_ctx.nal_idx);
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
