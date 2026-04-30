// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 CAVLC encoder (Phase 6A.4).
//!
//! Inverts the decoder at `codec::h264::cavlc::decode_cavlc_block`.
//! Given scan-ordered coefficients + nC context + block type, writes
//! the 5-step CAVLC syntax to a `BitWriter`:
//!
//!   1. coeff_token — `(total_coeffs, trailing_ones)` via nC-selected VLC.
//!   2. trailing-one signs — one bit per ±1 trailing coefficient.
//!   3. levels — prefix + suffix with adaptive suffix_length state.
//!   4. total_zeros — VLC (skipped when tc==0 or tc==max_coeffs).
//!   5. run_before — VLC per non-last coefficient.
//!
//! Algorithm note:
//!   docs/design/h264-encoder-algorithms/cavlc-encoder.md
//!
//! The single entry point `encode_cavlc_block` handles all four
//! block variants (luma 4×4, I_16x16 luma AC, chroma AC, chroma DC)
//! via the `max_coeffs` argument.

#[allow(unused_imports)]
use super::encoder::bitstream_writer::{BitSink, BitWriter};  // BitWriter used only in tests
use super::tables::{encode_coeff_token, encode_run_before, encode_total_zeros};
use super::H264Error;

/// Block-type variants for CAVLC encoding. Determines `max_coeffs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CavlcBlockType {
    /// Regular luma 4×4 block: 16 coefficients, DC at scan-pos 0.
    Luma4x4,
    /// Intra_16x16 luma AC block: 15 coefficients, DC emitted
    /// separately via the 4×4 Hadamard DC block.
    Intra16x16Ac,
    /// Chroma AC block (4:2:0): 15 coefficients, DC emitted separately.
    ChromaAc,
    /// Chroma DC block (4:2:0): 4 coefficients, post-Hadamard'd.
    ChromaDc,
}

impl CavlcBlockType {
    /// Number of scan positions for this block type.
    pub fn max_coeffs(self) -> u8 {
        match self {
            Self::Luma4x4 => 16,
            Self::Intra16x16Ac | Self::ChromaAc => 15,
            Self::ChromaDc => 4,
        }
    }
}

/// Encode a single 4×4 CAVLC residual block.
///
/// `coeffs` is a slice of exactly `block_type.max_coeffs()` i32
/// values in scan (zigzag) order. Position 0 is the lowest-frequency
/// coefficient (DC for Luma4x4/ChromaDc; first AC for Intra16x16Ac/
/// ChromaAc), position `max_coeffs - 1` is the highest-frequency.
///
/// `nc` is the neighbor-derived context for the coeff_token table
/// (use `-1` for chroma DC).
pub fn encode_cavlc_block<S: BitSink>(
    writer: &mut S,
    coeffs: &[i32],
    nc: i8,
    block_type: CavlcBlockType,
) -> Result<(), H264Error> {
    let max_coeffs = block_type.max_coeffs();
    if coeffs.len() != max_coeffs as usize {
        return Err(H264Error::CavlcError(format!(
            "encode_cavlc_block: expected {max_coeffs} coeffs, got {}",
            coeffs.len()
        )));
    }

    // Derive CAVLC block metadata from the scan-ordered coefficients.
    let (total_coeffs, trailing_ones, levels, t1_signs, runs, total_zeros) =
        derive_cavlc_fields(coeffs)?;

    // Step 1 — coeff_token.
    let (bits, len) = encode_coeff_token(total_coeffs, trailing_ones, nc).ok_or_else(|| {
        H264Error::CavlcError(format!(
            "encode_coeff_token missing entry: tc={total_coeffs} t1={trailing_ones} nc={nc}"
        ))
    })?;
    writer.write_bits(bits as u32, len);

    if total_coeffs == 0 {
        return Ok(());
    }

    // Step 2 — trailing-one sign bits (highest-freq TS first).
    // `t1_signs` has bit 0 = sign of the **first (highest-freq)** TS.
    for k in 0..trailing_ones {
        let sign_bit = (t1_signs >> k) & 1;
        writer.write_bit(sign_bit != 0);
    }

    // Step 3 — levels. `levels` is ordered highest-freq-first
    // (reverse scan), excluding trailing ones. Its length is
    // `total_coeffs - trailing_ones`.
    //
    // suffix_length state machine mirrors the decoder (spec § 9.2.2.1).
    let mut suffix_length: u8 = if total_coeffs > 10 && trailing_ones < 3 {
        1
    } else {
        0
    };
    // Thresholds for suffix_length growth: sl 1→2 @ |lvl|>3, 2→3 @ >6, etc.
    let thresholds: [u32; 6] = [3, 6, 12, 24, 48, u32::MAX];

    for (k, &level) in levels.iter().enumerate() {
        // Level → level_code. The first non-TS level absorbs a −2 offset
        // (spec § 9.2.2.1) when trailing_ones < 3, because magnitudes
        // ±1 are reserved for trailing-ones.
        let mut mag = level.unsigned_abs() as i64;
        let neg = level < 0;
        if k == 0 && trailing_ones < 3 {
            mag -= 1; // subtract 1, then level_code maps mag→mag*2-2 below
        }
        debug_assert!(mag >= 1, "derive_cavlc_fields should have ensured |level|>=1");

        // Decoder formula (taking abs of the intermediate result):
        //   level_code = (|level_code|+2)/2 if even  →  |level|
        //   level_code = (|level_code|+1)/2 if odd   →  |level| (negated)
        // Inverting: level_code = 2*|l| - 2 (for positive level), or
        //                         2*|l| - 1 (for negative).
        // But we already shifted `mag` by -1 for the first-level case,
        // so let eff_mag = mag (with the -1 already applied). Then
        //   level_code = 2*eff_mag - 2 + (neg as 1)
        //              = 2*(mag) - 2  (positive)
        //              = 2*(mag) - 1  (negative)
        let level_code = 2 * mag - if neg { 1 } else { 2 };
        debug_assert!(level_code >= 0, "level_code underflow at k={k} level={level}");

        // Split into prefix + suffix.
        emit_level_code(writer, level_code as u64, suffix_length)?;

        // Update suffix_length state — based on the ORIGINAL absolute
        // level, not the shifted `mag`.
        let abs_level = level.unsigned_abs();
        if suffix_length == 0 {
            suffix_length = 1;
        }
        if suffix_length < 6 && abs_level > thresholds[suffix_length as usize - 1] {
            suffix_length += 1;
        }
    }

    // Step 4 — total_zeros. Skipped when the block is fully packed.
    if total_coeffs < max_coeffs {
        let (tz_bits, tz_len) = encode_total_zeros(total_zeros, total_coeffs, max_coeffs)
            .ok_or_else(|| {
                H264Error::CavlcError(format!(
                    "encode_total_zeros missing: tz={total_zeros} tc={total_coeffs} max={max_coeffs}"
                ))
            })?;
        writer.write_bits(tz_bits as u32, tz_len);
    }

    // Step 5 — run_before. One per nonzero except the last (lowest-freq).
    let mut zeros_left = total_zeros;
    for i in 0..(total_coeffs as usize - 1) {
        if zeros_left == 0 {
            break;
        }
        let run = runs[i];
        let (rb_bits, rb_len) = encode_run_before(run, zeros_left).ok_or_else(|| {
            H264Error::CavlcError(format!(
                "encode_run_before missing: run={run} zeros_left={zeros_left}"
            ))
        })?;
        writer.write_bits(rb_bits as u32, rb_len);
        zeros_left = zeros_left.saturating_sub(run);
    }

    Ok(())
}

/// Walk scan-ordered coefficients and derive everything CAVLC needs.
///
/// Per H.264 spec § 7.3.5.3.2 pseudocode (line 4097-4101): the decoder
/// places levels in a forward walk from `coeffNum = 0` upward:
/// ```
/// coeffNum = -1
/// for (i = tc-1; i >= 0; i--) {           // REVERSE emit order
///     coeffNum += run[i] + 1
///     coeffLevel[coeffNum] = level[i]     // forward-scan placement
/// }
/// ```
/// So `run[tc-1]` = position of the lowest-scan-pos nonzero (zeros
/// before it), and each earlier `run[i]` is the gap between
/// consecutive nonzeros in forward scan (minus 1).
///
/// `total_zeros` per spec § 9.2.3 = zeros at scan positions **below**
/// the highest-scan-pos nonzero. i.e., zeros in `[0, highest_nz − 1]`
/// that aren't themselves nonzero. Zeros ABOVE the highest-nz are
/// NOT counted.
///
/// `levels` and `t1_signs` are in reverse-scan-emit order (highest-
/// freq first), matching the decoder's level read loop.
fn derive_cavlc_fields(
    coeffs: &[i32],
) -> Result<(u8, u8, Vec<i32>, u8, Vec<u8>, u8), H264Error> {
    // Forward-scan positions of nonzero coefficients.
    let positions: Vec<usize> = coeffs
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0)
        .map(|(i, _)| i)
        .collect();

    let total_coeffs = positions.len() as u8;
    if total_coeffs == 0 {
        return Ok((0, 0, Vec::new(), 0, Vec::new(), 0));
    }

    // Levels in REVERSE scan order (highest-pos first). levels_rev[0]
    // = coeff at the highest scan position.
    let levels_rev: Vec<i32> = positions.iter().rev().map(|&p| coeffs[p]).collect();

    // Trailing ones: ±1 at the highest-freq end, up to 3.
    let mut trailing_ones = 0u8;
    let mut t1_signs = 0u8;
    for (k, &v) in levels_rev.iter().enumerate() {
        if trailing_ones >= 3 || v.abs() != 1 {
            break;
        }
        if v < 0 {
            t1_signs |= 1 << k;
        }
        trailing_ones += 1;
    }

    let levels: Vec<i32> = levels_rev
        .iter()
        .skip(trailing_ones as usize)
        .copied()
        .collect();

    // Runs in emit order. For i in 0..tc-1: run[i] = gap between
    // positions[tc-1-i] (higher-pos nonzero) and positions[tc-2-i]
    // (next lower-pos nonzero). run[tc-1] = positions[0] (zeros
    // before the lowest-pos nonzero).
    let mut runs = Vec::with_capacity(total_coeffs as usize);
    let tc = total_coeffs as usize;
    for i in 0..tc {
        let run_val = if i == tc - 1 {
            positions[0]
        } else {
            let p_hi = positions[tc - 1 - i];
            let p_lo_next = positions[tc - 2 - i];
            p_hi - p_lo_next - 1
        };
        if run_val > 255 {
            return Err(H264Error::CavlcError(format!(
                "run_before overflow: {run_val}"
            )));
        }
        runs.push(run_val as u8);
    }

    let total_zeros: u16 = runs.iter().map(|&r| r as u16).sum();
    if total_zeros > 255 {
        return Err(H264Error::CavlcError(format!(
            "total_zeros overflow: {total_zeros}"
        )));
    }

    Ok((
        total_coeffs,
        trailing_ones,
        levels,
        t1_signs,
        runs,
        total_zeros as u8,
    ))
}

/// Emit a level's (prefix, suffix) pair given `level_code` and the
/// current `suffix_length`. Implements spec § 9.2.2.1 with all its
/// escape-code regions.
fn emit_level_code<S: BitSink>(
    writer: &mut S,
    level_code: u64,
    suffix_length: u8,
) -> Result<(), H264Error> {
    // Three regimes:
    //   - suffix_length > 0: split `level_code` into prefix (unary
    //     of floor(level_code / 2^sl)) and suffix (low sl bits).
    //     If the computed prefix >= 15, we enter the full-escape
    //     path with 12-bit suffix.
    //   - suffix_length == 0 and level_code < 14: prefix = level_code,
    //     no suffix.
    //   - suffix_length == 0 and 14 <= level_code < 30: prefix = 14,
    //     suffix_size = 4, suffix = level_code - 14.
    //   - suffix_length == 0 and level_code >= 30: full escape, prefix
    //     = 15, suffix_size = 12, suffix = level_code - 30.
    //   - suffix_length > 0 and level_code >= 15 << sl: full escape,
    //     prefix = 15, suffix_size = 12, suffix = level_code - (15<<sl).
    //
    // In all escape paths, the escape value encoded in the 12-bit
    // suffix is `(level_code - regime_offset)` and must fit in 12 bits
    // (≤ 4095). Above that, the spec doesn't define an encoding — cap
    // at ±2063 absolute level.
    let lc = level_code;

    if suffix_length == 0 {
        if lc < 14 {
            // Prefix = lc (unary), no suffix.
            emit_unary_prefix(writer, lc as u8);
            return Ok(());
        }
        if lc < 14 + 16 {
            // 14 <= lc < 30: prefix = 14, 4-bit suffix holds lc - 14.
            emit_unary_prefix(writer, 14);
            writer.write_bits((lc - 14) as u32, 4);
            return Ok(());
        }
        // lc >= 30: full escape. prefix = 15, 12-bit suffix holds lc - 30.
        let escape = lc - 30;
        if escape > 0xFFF {
            return Err(H264Error::CavlcError(format!(
                "level_code too large for 12-bit escape: lc={lc}"
            )));
        }
        emit_unary_prefix(writer, 15);
        writer.write_bits(escape as u32, 12);
        return Ok(());
    }

    // suffix_length > 0
    let sl = suffix_length as u32;
    let boundary = 15u64 << sl; // threshold into full-escape regime
    if lc < boundary {
        let prefix = (lc >> sl) as u8;
        let suffix = (lc & ((1u64 << sl) - 1)) as u32;
        emit_unary_prefix(writer, prefix);
        writer.write_bits(suffix, suffix_length);
        return Ok(());
    }
    // Full escape at sl > 0: prefix = 15, 12-bit suffix holds lc - boundary.
    let escape = lc - boundary;
    if escape > 0xFFF {
        return Err(H264Error::CavlcError(format!(
            "level_code too large for 12-bit escape: lc={lc} sl={sl}"
        )));
    }
    emit_unary_prefix(writer, 15);
    writer.write_bits(escape as u32, 12);
    Ok(())
}

/// Emit `n` zero bits followed by a single 1 bit (unary code for n).
fn emit_unary_prefix<S: BitSink>(writer: &mut S, n: u8) {
    for _ in 0..n {
        writer.write_bit(false);
    }
    writer.write_bit(true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::bitstream::{EpByteMap, RbspReader};
    use crate::codec::h264::cavlc::decode_cavlc_block;

    fn identity_ep_map(len: usize) -> EpByteMap {
        EpByteMap {
            rbsp_to_raw: (0..len).collect(),
        }
    }

    fn encode_to_bytes(coeffs: &[i32], nc: i8, bt: CavlcBlockType) -> Vec<u8> {
        let mut w = BitWriter::new();
        encode_cavlc_block(&mut w, coeffs, nc, bt).unwrap();
        // Pad to byte boundary with zeros so decode doesn't hit EOF on
        // an unaligned final block — in a real slice this'd be the
        // next MB's first bits.
        while !w.byte_aligned() {
            w.write_bit(false);
        }
        w.finish()
    }

    /// Round-trip helper: encode → decode → compare scan-order coeffs.
    fn check_round_trip(coeffs: &[i32], nc: i8, bt: CavlcBlockType) {
        let bytes = encode_to_bytes(coeffs, nc, bt);
        // Pad tail so decoder can always read a trailing-zero run_before.
        let mut padded = bytes;
        padded.extend_from_slice(&[0u8; 4]);
        let ep_map = identity_ep_map(padded.len());
        let mut reader = RbspReader::new(&padded);
        let (block, _positions) =
            decode_cavlc_block(&mut reader, nc, &ep_map, &padded, bt.max_coeffs())
                .expect("decode should succeed on valid CAVLC");
        assert_eq!(
            block.total_coeffs as usize,
            coeffs.iter().filter(|c| **c != 0).count(),
            "total_coeffs mismatch for input {coeffs:?}"
        );
        // For positions in range, `block.coeffs[..max]` should equal `coeffs`.
        for i in 0..bt.max_coeffs() as usize {
            assert_eq!(
                block.coeffs[i], coeffs[i],
                "scan pos {i} mismatch for input {coeffs:?} → decoded {:?}",
                &block.coeffs[..bt.max_coeffs() as usize]
            );
        }
    }

    // ─── Real-world DC scan reproductions ────────────────────────

    /// Exact DC scan dumped from MB (0, 0) when encoding
    /// `img4138_32x32_f1.yuv` with forced I_16x16. tc=14, t1=0,
    /// nc=0 — triggers the high-TotalCoeff path that conformant
    /// decoders reject in real content. If our encode→decode
    /// round-trips, the bug is encoder-vs-spec, not
    /// encoder-vs-decoder. If it doesn't, the bug is in our CAVLC
    /// writer or derive_cavlc_fields.
    #[test]
    fn dc_scan_img4138_mb_0_0_round_trip() {
        let coeffs = [
            79i32, 14, 34, 2, -1, 3, 8, 0, 13, -19, -4, 0, 5, -10, -5, -48,
        ];
        check_round_trip(&coeffs, 0, CavlcBlockType::Luma4x4);
    }

    /// Exact DC scan from deterministic pattern MB (0, 0) at CRF=80,
    /// forced I_16x16. tc=5, t1=0, nc=0 — large absolute level
    /// magnitude (-167) at scan pos 0. Tests level coding for big
    /// magnitudes.
    #[test]
    fn dc_scan_deterministic_mb_0_0_round_trip() {
        let coeffs = [
            -167i32, -24, -49, 0, 0, 0, -15, 0, 0, -29, 0, 0, 0, 0, 0, 0,
        ];
        check_round_trip(&coeffs, 0, CavlcBlockType::Luma4x4);
    }

    /// Round-trip various nc values for the same coefficient pattern.
    /// CAVLC uses nC to pick the coeff_token table (5 tables: nc<2,
    /// 2..3, 4..7, 8+, chroma). If our table transcription differs
    /// for the higher-nC tables, we'd see encode→decode disagreement.
    #[test]
    fn nc_table_selection_round_trip_all_tables() {
        // Pick a non-trivial pattern with TC=4, mixed levels.
        let coeffs = [3i32, -2, 5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        for nc in [0i8, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16] {
            check_round_trip(&coeffs, nc, CavlcBlockType::Luma4x4);
        }
    }

    // ─── Empty block ─────────────────────────────────────────────

    #[test]
    fn empty_block_emits_single_bit() {
        // nC=0: coeff_token(tc=0, T1=0) = "1".
        let coeffs = [0i32; 16];
        let bytes = encode_to_bytes(&coeffs, 0, CavlcBlockType::Luma4x4);
        // First bit of first byte should be 1, rest zero-padding.
        assert_eq!(bytes[0] & 0b1000_0000, 0b1000_0000);
    }

    #[test]
    fn empty_luma_round_trip() {
        check_round_trip(&[0i32; 16], 0, CavlcBlockType::Luma4x4);
    }

    // ─── Single trailing one ──────────────────────────────────────

    #[test]
    fn single_plus_one_at_last_scan_pos() {
        // One +1 at scan pos 15. coeff_token(tc=1, T1=1) = "01",
        // T1 sign = "0", total_zeros(tc=1) = "1" (meaning 0 zeros).
        let mut coeffs = [0i32; 16];
        coeffs[15] = 1;
        check_round_trip(&coeffs, 0, CavlcBlockType::Luma4x4);
    }

    #[test]
    fn single_minus_one_at_last_scan_pos() {
        let mut coeffs = [0i32; 16];
        coeffs[15] = -1;
        check_round_trip(&coeffs, 0, CavlcBlockType::Luma4x4);
    }

    // ─── Multiple trailing ones ───────────────────────────────────

    #[test]
    fn two_trailing_ones_round_trip() {
        let mut coeffs = [0i32; 16];
        coeffs[14] = -1;
        coeffs[15] = 1;
        check_round_trip(&coeffs, 0, CavlcBlockType::Luma4x4);
    }

    #[test]
    fn three_trailing_ones_round_trip() {
        let mut coeffs = [0i32; 16];
        coeffs[13] = 1;
        coeffs[14] = -1;
        coeffs[15] = 1;
        check_round_trip(&coeffs, 0, CavlcBlockType::Luma4x4);
    }

    // ─── Levels (non-trailing-one) ────────────────────────────────

    #[test]
    fn small_level_round_trip() {
        let mut coeffs = [0i32; 16];
        coeffs[10] = 3;
        coeffs[15] = 1;
        check_round_trip(&coeffs, 0, CavlcBlockType::Luma4x4);
    }

    #[test]
    fn negative_level_round_trip() {
        let mut coeffs = [0i32; 16];
        coeffs[8] = -5;
        coeffs[15] = 1;
        check_round_trip(&coeffs, 0, CavlcBlockType::Luma4x4);
    }

    #[test]
    fn large_level_round_trip() {
        // Exercises the escape path for level magnitudes > 32 or so.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 100;
        check_round_trip(&coeffs, 0, CavlcBlockType::Luma4x4);
    }

    #[test]
    fn extreme_level_escape_path() {
        // Hits the 12-bit escape for very large magnitudes.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 1000;
        check_round_trip(&coeffs, 0, CavlcBlockType::Luma4x4);
    }

    // ─── Runs ─────────────────────────────────────────────────────

    #[test]
    fn block_with_gaps_round_trip() {
        let mut coeffs = [0i32; 16];
        coeffs[0] = 5;
        coeffs[4] = -3;
        coeffs[8] = 2;
        coeffs[15] = 1;
        check_round_trip(&coeffs, 0, CavlcBlockType::Luma4x4);
    }

    // ─── Different nC contexts ────────────────────────────────────

    #[test]
    fn nc_2_round_trip() {
        // nC=2 selects TABLE_1.
        let mut coeffs = [0i32; 16];
        coeffs[10] = 2;
        coeffs[15] = -1;
        check_round_trip(&coeffs, 2, CavlcBlockType::Luma4x4);
    }

    #[test]
    fn nc_4_round_trip() {
        // nC=4 selects TABLE_2.
        let mut coeffs = [0i32; 16];
        coeffs[5] = 4;
        coeffs[12] = -2;
        coeffs[15] = 1;
        check_round_trip(&coeffs, 4, CavlcBlockType::Luma4x4);
    }

    #[test]
    fn nc_8_round_trip() {
        // nC=8 selects TABLE_3 (fixed 6-bit codes).
        let mut coeffs = [0i32; 16];
        coeffs[0] = -7;
        coeffs[14] = 1;
        coeffs[15] = -1;
        check_round_trip(&coeffs, 8, CavlcBlockType::Luma4x4);
    }

    // ─── Block variants ───────────────────────────────────────────

    #[test]
    fn intra16x16_ac_round_trip() {
        // 15 coefficients (DC stripped).
        let mut coeffs = [0i32; 15];
        coeffs[3] = 2;
        coeffs[14] = 1;
        check_round_trip(&coeffs, 0, CavlcBlockType::Intra16x16Ac);
    }

    #[test]
    fn chroma_ac_round_trip() {
        let mut coeffs = [0i32; 15];
        coeffs[0] = 3;
        coeffs[14] = -1;
        check_round_trip(&coeffs, 1, CavlcBlockType::ChromaAc);
    }

    #[test]
    fn chroma_dc_round_trip() {
        // nC = -1 for chroma DC.
        let mut coeffs = [0i32; 4];
        coeffs[0] = 2;
        coeffs[3] = -1;
        check_round_trip(&coeffs, -1, CavlcBlockType::ChromaDc);
    }

    #[test]
    fn chroma_dc_all_zero_round_trip() {
        check_round_trip(&[0i32; 4], -1, CavlcBlockType::ChromaDc);
    }

    // ─── derive_cavlc_fields unit tests ───────────────────────────

    #[test]
    fn derive_empty_block() {
        let (tc, t1, _lv, sg, _runs, tz) = derive_cavlc_fields(&[0i32; 16]).unwrap();
        assert_eq!(tc, 0);
        assert_eq!(t1, 0);
        assert_eq!(sg, 0);
        assert_eq!(tz, 0);
    }

    #[test]
    fn derive_single_plus_one_at_top() {
        // +1 at scan pos 15 (highest-freq) — total_zeros = 15 per
        // spec § 9.2.3 (zeros at positions 0..14).
        let mut c = [0i32; 16];
        c[15] = 1;
        let (tc, t1, lv, sg, runs, tz) = derive_cavlc_fields(&c).unwrap();
        assert_eq!(tc, 1);
        assert_eq!(t1, 1);
        assert!(lv.is_empty());
        assert_eq!(sg, 0);
        // run[tc-1=0] = position of lowest-pos nonzero = 15.
        assert_eq!(runs, vec![15]);
        assert_eq!(tz, 15);
    }

    #[test]
    fn derive_single_plus_one_at_bottom() {
        // +1 at scan pos 0 (lowest-freq DC) — total_zeros = 0 per
        // spec (no zeros before position 0).
        let mut c = [0i32; 16];
        c[0] = 1;
        let (tc, t1, lv, _sg, runs, tz) = derive_cavlc_fields(&c).unwrap();
        assert_eq!(tc, 1);
        assert_eq!(t1, 1);
        assert!(lv.is_empty());
        assert_eq!(runs, vec![0]);
        assert_eq!(tz, 0);
    }

    #[test]
    fn derive_three_nonzeros_spec_layout() {
        // Block with nonzeros at scan positions 2, 5, 10 (forward).
        // Per spec:
        //   level[0] = coeff at pos 10 (highest)
        //   level[1] = coeff at pos 5
        //   level[2] = coeff at pos 2 (lowest)
        //   run[0] = 10 - 5 - 1 = 4  (zeros between pos 10 and pos 5)
        //   run[1] = 5 - 2 - 1 = 2   (zeros between pos 5 and pos 2)
        //   run[2] = 2              (zeros before pos 2)
        //   total_zeros = 8        (zeros at positions 0..9)
        let mut c = [0i32; 16];
        c[2] = 3;
        c[5] = 2;
        c[10] = -4;
        let (tc, _t1, _lv, _sg, runs, tz) = derive_cavlc_fields(&c).unwrap();
        assert_eq!(tc, 3);
        assert_eq!(runs, vec![4, 2, 2]);
        assert_eq!(tz, 8);
    }
}

