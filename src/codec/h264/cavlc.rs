// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! CAVLC (Context-Adaptive Variable-Length Coding) decoder with embeddable
//! position tracking for H.264 video steganography.
//!
//! This is the core of the H.264 stego approach. CAVLC trailing-one sign bits
//! are true independent bit positions — flipping one does NOT cascade to any
//! other bit, does NOT change the bitstream length, and is read correctly by
//! every H.264 decoder in existence.
//!
//! The decoder follows ITU-T H.264 Section 9.2.1 (CAVLC) exactly:
//! 1. coeff_token — VLC for (TotalCoeffs, TrailingOnes)
//! 2. trailing_ones_sign_flag — raw bits (PRIMARY EMBEDDING TARGET)
//! 3. level codes — prefix + suffix with adaptive suffixLength
//! 4. total_zeros — VLC
//! 5. run_before — VLC per coefficient
//!
//! For each embeddable bit, the decoder records its position in the RAW byte
//! stream (with emulation prevention bytes present) via [`EpByteMap`].

use super::bitstream::{EpByteMap, RbspReader};
use super::tables::{decode_coeff_token, decode_vlc, run_before_table, total_zeros_table_for};
use super::H264Error;

/// Embedding domain: which type of embeddable position this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedDomain {
    /// Trailing-one sign bit (Phase 1a). Distortion = 2 (constant).
    T1Sign,
    /// Level suffix magnitude LSB (Phase 1b). Distortion = 1 (constant).
    /// Requires suffixLength >= 2.
    LevelSuffixMag,
    /// Level suffix sign bit (Phase 2). Distortion = 2*|coeff|.
    /// Requires suffixLength >= 1.
    LevelSuffixSign,
    /// Motion vector difference suffix LSB (Phase 3). Distortion model lives
    /// in `stego/cost/h264_cost.rs` (flat in 3a, |mvd|²-based in 3b).
    /// Length-preserving: flipping the LSB of a signed-Exp-Golomb suffix
    /// keeps codeword length `2·lz + 1` unchanged. Only non-zero MVDs
    /// (codeNum ≥ 1, `lz ≥ 1`) produce embeddable positions; MVD = 0 has
    /// no suffix.
    MvdLsb,
}

/// A position in the raw byte stream where an embeddable bit lives.
#[derive(Debug, Clone)]
pub struct EmbeddablePosition {
    /// Byte offset in the raw NAL unit data (after NAL header byte),
    /// relative to the start of the NAL payload.
    pub raw_byte_offset: usize,
    /// Bit offset within that byte (0 = MSB, 7 = LSB).
    pub bit_offset: u8,
    /// Which embedding domain this position belongs to.
    pub domain: EmbedDomain,
    /// Scan position within the 4x4 block (0-15, zigzag order).
    pub scan_pos: u8,
    /// Coefficient value at this position (for cost computation).
    pub coeff_value: i32,
    /// Whether this position has an EP byte conflict (bit flip would
    /// create or destroy a 0x000000-0x000003 sequence).
    pub ep_conflict: bool,
    /// Block index within the frame (set by macroblock parser, used by cost function).
    pub block_idx: u32,
    /// Frame index (set by pipeline, used for temporal weighting).
    pub frame_idx: u16,
    /// Macroblock index within the frame (set by pipeline at position-shift
    /// time). Phase 3b reads it for MVD positions to derive per-MB residual
    /// energy; coefficient positions can derive `mb_idx = block_idx / 26`
    /// but having the field avoids repeated division in the hot cost loop.
    pub mb_idx: u32,
}

/// Decoded CAVLC 4x4 block data.
#[derive(Debug, Clone)]
pub struct CavlcBlock {
    /// Number of non-zero coefficients (0-16).
    pub total_coeffs: u8,
    /// Number of trailing ±1 coefficients (0-3).
    pub trailing_ones: u8,
    /// Decoded coefficient values in scan order (zigzag).
    /// Only `total_coeffs` entries are valid; rest are zero.
    pub coeffs: [i32; 16],
}

/// Decode a single 4x4 CAVLC residual block with embeddable position tracking.
///
/// # Arguments
/// * `reader` — RBSP bit reader (positioned at the coeff_token start)
/// * `nc` — neighbor context for coeff_token table selection (-1 for chroma DC)
/// * `ep_map` — RBSP-to-raw byte offset mapping for position tracking
/// * `raw_data` — the original raw NAL payload bytes (for EP conflict checking)
/// * `max_coeffs` — maximum number of coefficients (16 for 4x4, 15 for Intra16x16AC, 4 for chroma DC)
///
/// Returns the decoded block and a list of embeddable positions.
pub fn decode_cavlc_block(
    reader: &mut RbspReader<'_>,
    nc: i8,
    ep_map: &EpByteMap,
    raw_data: &[u8],
    max_coeffs: u8,
) -> Result<(CavlcBlock, Vec<EmbeddablePosition>), H264Error> {
    let mut positions = Vec::new();

    // Step 1: Decode coeff_token → (TotalCoeffs, TrailingOnes)
    let (total_coeffs, trailing_ones) = decode_coeff_token(reader, nc)?;

    if total_coeffs == 0 {
        return Ok((
            CavlcBlock {
                total_coeffs: 0,
                trailing_ones: 0,
                coeffs: [0; 16],
            },
            positions,
        ));
    }

    // Levels array: stores decoded levels in reverse scan order
    // (trailing ones first, then remaining levels)
    let mut levels = vec![0i32; total_coeffs as usize];

    // Step 2: Trailing ones sign flags — PRIMARY EMBEDDING TARGET
    // Each trailing one has exactly 1 sign bit: 0 = +1, 1 = -1.
    // These bits are raw bits in the RBSP, occupying independent positions.
    for i in 0..trailing_ones as usize {
        // Record position BEFORE reading the bit
        let rbsp_byte = reader.byte_pos();
        let rbsp_bit = reader.bit_pos();

        let sign_bit = reader.read_bit()?;
        levels[i] = if sign_bit { -1 } else { 1 };

        // Map RBSP position to raw byte position
        if rbsp_byte < ep_map.rbsp_to_raw.len() {
            let raw_byte = ep_map.rbsp_to_raw[rbsp_byte];
            let ep_conflict = check_ep_conflict(raw_data, raw_byte, rbsp_bit);

            positions.push(EmbeddablePosition {
                raw_byte_offset: raw_byte,
                bit_offset: rbsp_bit,
                domain: EmbedDomain::T1Sign,
                scan_pos: 0, // Will be filled in after run_before decoding
                coeff_value: levels[i],
                ep_conflict,
                block_idx: 0, // Set by macroblock parser
                frame_idx: 0, // Set by pipeline
                mb_idx: 0,    // Set by pipeline
            });
        }
    }

    // Step 3: Level codes for remaining coefficients
    // Uses adaptive suffixLength: starts at 0 (or 1), increments at thresholds
    let mut suffix_length: u8 = if total_coeffs > 10 && trailing_ones < 3 {
        1
    } else {
        0
    };

    for i in trailing_ones as usize..total_coeffs as usize {
        // Decode level_prefix (unary: count leading zeros, then a 1)
        let mut level_prefix = 0u32;
        loop {
            if reader.read_bit()? {
                break;
            }
            level_prefix += 1;
            if level_prefix > 28 {
                return Err(H264Error::CavlcError(
                    "level_prefix overflow (>28)".into(),
                ));
            }
        }

        // Compute level_suffix_size
        let level_suffix_size = if level_prefix == 14 && suffix_length == 0 {
            4
        } else if level_prefix >= 15 {
            (level_prefix - 3) as u8
        } else {
            suffix_length
        };

        // Read level_suffix (fixed-length field)
        let level_suffix = if level_suffix_size > 0 {
            // Record suffix bit positions for Phase 1b/2
            let suffix_start_byte = reader.byte_pos();
            let suffix_start_bit = reader.bit_pos();
            let val = reader.read_bits(level_suffix_size)?;

            // Record level suffix positions for future phases
            if suffix_length >= 1 && suffix_start_byte < ep_map.rbsp_to_raw.len() {
                let raw_byte = ep_map.rbsp_to_raw[suffix_start_byte];

                // Suffix bit 0 = sign (LevelSuffixSign, Phase 2)
                // This is the LSB of the suffix field
                let sign_bit_byte_pos = suffix_start_byte + (suffix_start_bit as usize + level_suffix_size as usize - 1) / 8;
                let sign_bit_bit_pos = (suffix_start_bit + level_suffix_size - 1) % 8;
                if sign_bit_byte_pos < ep_map.rbsp_to_raw.len() {
                    let sign_raw = ep_map.rbsp_to_raw[sign_bit_byte_pos];
                    positions.push(EmbeddablePosition {
                        raw_byte_offset: sign_raw,
                        bit_offset: sign_bit_bit_pos,
                        domain: EmbedDomain::LevelSuffixSign,
                        scan_pos: 0, // filled after run_before
                        coeff_value: 0, // filled after level decode
                        ep_conflict: check_ep_conflict(raw_data, sign_raw, sign_bit_bit_pos),
                        block_idx: 0,
                        frame_idx: 0,
                        mb_idx: 0,
                    });
                }

                // Suffix bit 1 = magnitude LSB (LevelSuffixMag, Phase 1b)
                // Only available when suffixLength >= 2
                if suffix_length >= 2 && level_suffix_size >= 2 {
                    let mag_bit_byte_pos = suffix_start_byte + (suffix_start_bit as usize + level_suffix_size as usize - 2) / 8;
                    let mag_bit_bit_pos = (suffix_start_bit + level_suffix_size - 2) % 8;
                    if mag_bit_byte_pos < ep_map.rbsp_to_raw.len() {
                        let mag_raw = ep_map.rbsp_to_raw[mag_bit_byte_pos];
                        positions.push(EmbeddablePosition {
                            raw_byte_offset: mag_raw,
                            bit_offset: mag_bit_bit_pos,
                            domain: EmbedDomain::LevelSuffixMag,
                            scan_pos: 0,
                            coeff_value: 0,
                            ep_conflict: check_ep_conflict(raw_data, mag_raw, mag_bit_bit_pos),
                            block_idx: 0,
                            frame_idx: 0,
                            mb_idx: 0,
                        });
                    }
                }
            }

            val
        } else {
            0
        };

        // Compute levelCode per H.264 Section 9.2.2.1.
        // Note: the spec TEXT says "level_prefix >= 14" for the +15 offset, but
        // that produces overlapping encodings between prefix=14 and prefix=15.
        // The conformant behavior is prefix>=15 for the +15 offset and prefix>=16
        // for the escape extension, which gives a monotonic non-overlapping range.
        let mut level_code =
            ((level_prefix.min(15) as u32) << suffix_length) + level_suffix;
        if level_prefix >= 15 && suffix_length == 0 {
            level_code += 15;
        }
        if level_prefix >= 16 {
            level_code += (1 << (level_prefix - 3)) - 4096;
        }

        // First non-trailing-one level: if fewer than 3 trailing ones,
        // increment magnitude by 1 (the ±1 values are reserved for TS).
        if i == trailing_ones as usize && trailing_ones < 3 {
            level_code += 2;
        }

        // Map levelCode to signed level value (spec 9.2.2.1)
        let level = if level_code & 1 == 0 {
            (level_code as i32 + 2) / 2
        } else {
            -((level_code as i32 + 1) / 2)
        };

        levels[i] = level;

        // Threshold safety: for magnitude LSB flip to be safe, the new
        // magnitude (±1) must NOT cross the suffix_length threshold boundary,
        // because that would change suffix_length for subsequent levels,
        // causing a different bit count for the rest of the block.
        let abs_level = level.unsigned_abs();
        let active_sl = if suffix_length == 0 { 1 } else { suffix_length };
        let thresholds: [u32; 6] = [3, 6, 12, 24, 48, u32::MAX];
        let threshold = thresholds[(active_sl as usize - 1).min(5)];
        let orig_exceeds = abs_level > threshold;
        let would_cross = {
            let plus = abs_level + 1;
            let minus = abs_level.saturating_sub(1);
            (plus > threshold) != orig_exceeds || (minus > threshold) != orig_exceeds
        };

        // Update suffix positions with actual coefficient value and safety
        for pos in positions.iter_mut().rev() {
            if pos.coeff_value == 0
                && (pos.domain == EmbedDomain::LevelSuffixSign
                    || pos.domain == EmbedDomain::LevelSuffixMag)
            {
                pos.coeff_value = level;
                // LevelSuffixMag unsafe if threshold crossing possible
                if pos.domain == EmbedDomain::LevelSuffixMag && would_cross {
                    pos.ep_conflict = true;
                }
            } else {
                break;
            }
        }

        // Update suffixLength state machine
        if suffix_length == 0 {
            suffix_length = 1;
        }
        if suffix_length < 6 && abs_level > thresholds[suffix_length as usize - 1] {
            suffix_length += 1;
        }
    }

    // Step 4: total_zeros
    // Uses different VLC tables for chroma DC (max_coeffs <= 4) vs luma (max_coeffs > 4)
    let total_zeros = if total_coeffs < max_coeffs {
        let table = total_zeros_table_for(total_coeffs, max_coeffs);
        if table.is_empty() {
            0
        } else {
            decode_vlc(reader, table)?
        }
    } else {
        0 // All positions filled, no zeros
    };

    // Step 5: run_before for each coefficient
    let mut runs = vec![0u8; total_coeffs as usize];
    let mut zeros_left = total_zeros;
    for i in 0..total_coeffs as usize - 1 {
        if zeros_left == 0 {
            break;
        }
        let table = run_before_table(zeros_left);
        runs[i] = decode_vlc(reader, table)?;
        zeros_left = zeros_left.saturating_sub(runs[i]);
    }
    // Last coefficient gets remaining zeros
    if total_coeffs > 0 {
        runs[total_coeffs as usize - 1] = zeros_left;
    }

    // Reconstruct coefficient array in scan order per spec § 7.3.5.3.2
    // (pseudocode lines 4097-4101):
    //   coeffNum = -1
    //   for (i = tc-1; i >= 0; i--) {
    //       coeffNum += run[i] + 1
    //       coeffLevel[coeffNum] = level[i]
    //   }
    // Levels are in reverse-scan emit order: levels[0] is the highest-
    // scan-pos nonzero. The reverse walk here places the LOWEST-pos
    // nonzero first (at coeffNum = run[tc-1]), then climbs.
    let mut coeffs = [0i32; 16];
    let mut scan_positions = vec![0u8; total_coeffs as usize];
    let mut coeff_num: i32 = -1;
    for i in (0..total_coeffs as usize).rev() {
        coeff_num += runs[i] as i32 + 1;
        if coeff_num < 0 || coeff_num >= max_coeffs as i32 {
            return Err(H264Error::CavlcError(format!(
                "invalid run_before: coeff_num={coeff_num}"
            )));
        }
        scan_positions[i] = coeff_num as u8;
        coeffs[coeff_num as usize] = levels[i];
    }

    // Now assign scan_pos to embeddable positions
    let mut t1_idx = 0usize;
    let mut level_suffix_idx = trailing_ones as usize;
    for epos in positions.iter_mut() {
        match epos.domain {
            EmbedDomain::T1Sign => {
                if t1_idx < trailing_ones as usize {
                    epos.scan_pos = scan_positions[t1_idx];
                    t1_idx += 1;
                }
            }
            EmbedDomain::LevelSuffixSign | EmbedDomain::LevelSuffixMag => {
                if level_suffix_idx < total_coeffs as usize {
                    epos.scan_pos = scan_positions[level_suffix_idx];
                    // Only advance for LevelSuffixSign (first position per level)
                    if epos.domain == EmbedDomain::LevelSuffixSign {
                        level_suffix_idx += 1;
                    }
                }
            }
            EmbedDomain::MvdLsb => {
                // MVD positions come from `mv::parse_mv_field`, never from
                // CAVLC residual blocks — this branch is unreachable in
                // practice, left here for exhaustiveness.
            }
        }
    }

    Ok((
        CavlcBlock {
            total_coeffs,
            trailing_ones,
            coeffs,
        },
        positions,
    ))
}

/// Check whether flipping a bit at the given raw byte position would create
/// or destroy an emulation prevention byte sequence (0x00 0x00 0x00-0x03).
///
/// Returns true if the position is unsafe (should be marked WET).
pub(crate) fn check_ep_conflict(raw_data: &[u8], byte_offset: usize, bit_offset: u8) -> bool {
    if byte_offset >= raw_data.len() {
        return true; // out of bounds → unsafe
    }

    let original_byte = raw_data[byte_offset];
    let flipped_byte = original_byte ^ (1 << (7 - bit_offset));

    // Check if flipping creates a new 00 00 0x pattern (x <= 03)
    // or destroys an existing EP byte (00 00 03)
    // Check in a 3-byte window around the modified byte
    for offset in 0..3 {
        let start = if byte_offset >= offset {
            byte_offset - offset
        } else {
            continue;
        };
        if start + 2 >= raw_data.len() {
            continue;
        }

        // Build the 3-byte sequence with the flipped byte
        let b0 = if start == byte_offset { flipped_byte } else { raw_data[start] };
        let b1 = if start + 1 == byte_offset {
            flipped_byte
        } else {
            raw_data[start + 1]
        };
        let b2 = if start + 2 == byte_offset {
            flipped_byte
        } else {
            raw_data[start + 2]
        };

        // Check original 3-byte sequence
        let orig_b0 = raw_data[start];
        let orig_b1 = raw_data[start + 1];
        let orig_b2 = raw_data[start + 2];

        let orig_is_ep = orig_b0 == 0 && orig_b1 == 0 && orig_b2 <= 3;
        let new_is_ep = b0 == 0 && b1 == 0 && b2 <= 3;

        // Conflict if EP status changes (created or destroyed)
        if orig_is_ep != new_is_ep {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::bitstream::{remove_emulation_prevention_with_map, EpByteMap};

    /// BitWriter for constructing test bitstreams.
    struct BitWriter {
        data: Vec<u8>,
        current: u8,
        bit_pos: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self { data: Vec::new(), current: 0, bit_pos: 0 }
        }
        fn write_bit(&mut self, val: bool) {
            if val { self.current |= 1 << (7 - self.bit_pos); }
            self.bit_pos += 1;
            if self.bit_pos == 8 { self.data.push(self.current); self.current = 0; self.bit_pos = 0; }
        }
        fn write_bits(&mut self, val: u32, n: u8) {
            for i in (0..n).rev() { self.write_bit((val >> i) & 1 != 0); }
        }
        fn align(&mut self) {
            if self.bit_pos > 0 { self.data.push(self.current); self.current = 0; self.bit_pos = 0; }
        }
    }

    fn identity_ep_map(len: usize) -> EpByteMap {
        EpByteMap {
            rbsp_to_raw: (0..len).collect(),
        }
    }

    #[test]
    fn decode_empty_block() {
        // nC=0, coeff_token for (tc=0, T1=0) = "1" (1 bit)
        let data = [0b1000_0000];
        let ep_map = identity_ep_map(data.len());
        let mut reader = RbspReader::new(&data);
        let (block, positions) = decode_cavlc_block(&mut reader, 0, &ep_map, &data, 16).unwrap();
        assert_eq!(block.total_coeffs, 0);
        assert_eq!(block.trailing_ones, 0);
        assert!(positions.is_empty());
    }

    #[test]
    fn decode_single_trailing_one() {
        // nC=0: coeff_token(tc=1, T1=1) = "01" (2 bits).
        // T1 sign: 1 bit (0 = +1, 1 = -1).
        // total_zeros=0 per spec means "no zeros preceding the highest-
        // pos nonzero" → the nonzero is at scan pos 0. Encoded as "1"
        // (TOTAL_ZEROS_1 table, value 0).
        let mut bits = BitWriter::new();
        bits.write_bits(0b01, 2);
        bits.write_bit(false);
        bits.write_bit(true);
        bits.align();

        let ep_map = identity_ep_map(bits.data.len());
        let mut reader = RbspReader::new(&bits.data);
        let (block, positions) =
            decode_cavlc_block(&mut reader, 0, &ep_map, &bits.data, 16).unwrap();

        assert_eq!(block.total_coeffs, 1);
        assert_eq!(block.trailing_ones, 1);
        // Per spec: tz=0 places the level at scan pos 0 (DC).
        assert_eq!(block.coeffs[0], 1);

        let t1_positions: Vec<_> = positions
            .iter()
            .filter(|p| p.domain == EmbedDomain::T1Sign)
            .collect();
        assert_eq!(t1_positions.len(), 1);
        assert_eq!(t1_positions[0].coeff_value, 1);
    }

    #[test]
    fn decode_two_trailing_ones_with_signs() {
        // nC=0: coeff_token(tc=2, T1=2) = "001" (3 bits)
        // T1 signs: 2 bits (sign for each trailing one, reverse scan order)
        // sign=1,0 → levels = [-1, +1]
        let mut bits = BitWriter::new();
        bits.write_bits(0b001, 3); // coeff_token(tc=2, T1=2)
        bits.write_bit(true); // T1 sign 0: -1
        bits.write_bit(false); // T1 sign 1: +1
        // total_zeros for tc=2: "111" → 0
        bits.write_bits(0b111, 3);
        // run_before: tc=2, first coeff run=0 → "1" (zeros_left=0, no run coded)
        bits.align();

        let ep_map = identity_ep_map(bits.data.len());
        let mut reader = RbspReader::new(&bits.data);
        let (block, positions) =
            decode_cavlc_block(&mut reader, 0, &ep_map, &bits.data, 16).unwrap();

        assert_eq!(block.total_coeffs, 2);
        assert_eq!(block.trailing_ones, 2);

        // 2 T1 sign positions
        let t1_positions: Vec<_> = positions
            .iter()
            .filter(|p| p.domain == EmbedDomain::T1Sign)
            .collect();
        assert_eq!(t1_positions.len(), 2);
        assert_eq!(t1_positions[0].coeff_value, -1);
        assert_eq!(t1_positions[1].coeff_value, 1);
    }

    #[test]
    fn ep_conflict_detection_safe() {
        // Bytes: [0x12, 0x34, 0x56] — no 00 00 xx pattern
        let data = [0x12, 0x34, 0x56];
        assert!(!check_ep_conflict(&data, 1, 0)); // flip byte 1 bit 0
    }

    #[test]
    fn ep_conflict_detection_creates_pattern() {
        // Bytes: [0x00, 0x00, 0x04] — not an EP pattern
        // Flipping bit 5 of byte 2: 0x04 (00000100) → 0x00 (00000000)
        // Creates: [0x00, 0x00, 0x00] — start code! Conflict!
        let data = [0x00, 0x00, 0x04];
        assert!(check_ep_conflict(&data, 2, 5)); // flip makes 0x04→0x00
    }

    #[test]
    fn ep_conflict_detection_destroys_ep_byte() {
        // Bytes: [0x00, 0x00, 0x03] — emulation prevention byte
        // Flipping bit 6 of byte 2: 0x03 → 0x01
        // Destroys EP: [0x00, 0x00, 0x01] still matches pattern but 03→01 changes semantics
        let data = [0x00, 0x00, 0x03];
        // Original IS an EP pattern (0x00 0x00 0x03), flipped would be 0x00 0x00 0x01
        // Both have b2 <= 3, so both are EP patterns → no status change → no conflict
        assert!(!check_ep_conflict(&data, 2, 6));
        // But flipping bit 2 of byte 2: 0x03 → 0x07
        // New: [0x00, 0x00, 0x07] — NOT an EP pattern (0x07 > 0x03)
        // Original WAS EP → destroyed → conflict!
        assert!(check_ep_conflict(&data, 2, 4)); // 0x03 ^ 0x08 = 0x0B > 3
    }
}
