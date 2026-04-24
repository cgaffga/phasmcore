// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 CAVLC VLC lookup tables and scan orders.
//!
//! Tables from ITU-T H.264 specification:
//! - coeff_token (Table 9-5): 4 table sets selected by nC
//! - total_zeros (Tables 9-7, 9-8): indexed by total_coeffs
//! - run_before (Table 9-10): indexed by zeros_left
//! - 4x4 zigzag scan order (Table 8-13)
//!
//! VLC entries are stored as `(code_bits, code_length, value1, value2)`.
//! The decoder reads `code_length` bits and compares against `code_bits`.
//! For efficient lookup, entries within each table are sorted by code length.

use super::bitstream::RbspReader;
use super::H264Error;

// ---------------------------------------------------------------------------
// 4x4 Zigzag Scan Order (H.264 Table 8-13)
// ---------------------------------------------------------------------------

/// 4x4 zigzag scan: maps scan index (0-15) to raster index (row*4+col).
pub const ZIGZAG_4X4: [u8; 16] = [
    0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15,
];

/// Inverse zigzag: maps raster index to scan index.
pub const ZIGZAG_4X4_INV: [u8; 16] = [
    0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15,
];

// ---------------------------------------------------------------------------
// coeff_token VLC Tables (H.264 Table 9-5)
// ---------------------------------------------------------------------------

/// A coeff_token table entry: (code_bits, code_length, total_coeffs, trailing_ones).
#[derive(Debug, Clone, Copy)]
pub struct CoeffTokenEntry {
    pub bits: u16,
    pub len: u8,
    pub total_coeffs: u8,
    pub trailing_ones: u8,
}

/// Decode coeff_token from the bitstream using the nC-selected table.
///
/// Returns `(total_coeffs, trailing_ones)`.
pub fn decode_coeff_token(
    reader: &mut RbspReader<'_>,
    nc: i8,
) -> Result<(u8, u8), H264Error> {
    let table = select_coeff_token_table(nc);

    if nc >= 8 {
        // Table 3: fixed-length 6-bit codes
        let code = reader.read_bits(6)? as u16;
        for entry in table {
            if entry.bits == code && entry.len == 6 {
                return Ok((entry.total_coeffs, entry.trailing_ones));
            }
        }
        return Err(H264Error::CavlcError(format!(
            "invalid coeff_token FLC code: {code:#06b}"
        )));
    }

    // Variable-length: try matching from shortest to longest code
    let mut accumulated = 0u16;
    let mut bits_read = 0u8;

    // Read up to 16 bits (max coeff_token length)
    for _ in 0..16 {
        let bit = reader.read_bit()? as u16;
        accumulated = (accumulated << 1) | bit;
        bits_read += 1;

        // Check all entries with this code length
        for entry in table {
            if entry.len == bits_read && entry.bits == accumulated {
                return Ok((entry.total_coeffs, entry.trailing_ones));
            }
        }
    }

    Err(H264Error::CavlcError(
        "coeff_token: no matching VLC code found".into(),
    ))
}

fn select_coeff_token_table(nc: i8) -> &'static [CoeffTokenEntry] {
    if nc < 0 {
        // chroma DC (nC == -1): use table 4
        &COEFF_TOKEN_CHROMA_DC
    } else if nc < 2 {
        &COEFF_TOKEN_TABLE_0
    } else if nc < 4 {
        &COEFF_TOKEN_TABLE_1
    } else if nc < 8 {
        &COEFF_TOKEN_TABLE_2
    } else {
        &COEFF_TOKEN_TABLE_3
    }
}

macro_rules! ct {
    ($bits:expr, $len:expr, $tc:expr, $t1:expr) => {
        CoeffTokenEntry {
            bits: $bits,
            len: $len,
            total_coeffs: $tc,
            trailing_ones: $t1,
        }
    };
}

/// coeff_token table for nC in [0, 2) — H.264 Table 9-5(a).
/// Data sourced from ITU-T H.264 specification (H.264 Table 9-5(a)).
static COEFF_TOKEN_TABLE_0: [CoeffTokenEntry; 62] = [
    ct!(0b1, 1, 0, 0),
    ct!(0b000101, 6, 1, 0),
    ct!(0b01, 2, 1, 1),
    ct!(0b00000111, 8, 2, 0),
    ct!(0b000100, 6, 2, 1),
    ct!(0b001, 3, 2, 2),
    ct!(0b000000111, 9, 3, 0),
    ct!(0b00000110, 8, 3, 1),
    ct!(0b0000101, 7, 3, 2),
    ct!(0b00011, 5, 3, 3),
    ct!(0b0000000111, 10, 4, 0),
    ct!(0b000000110, 9, 4, 1),
    ct!(0b00000101, 8, 4, 2),
    ct!(0b000011, 6, 4, 3),
    ct!(0b00000000111, 11, 5, 0),
    ct!(0b0000000110, 10, 5, 1),
    ct!(0b000000101, 9, 5, 2),
    ct!(0b0000100, 7, 5, 3),
    ct!(0b0000000001111, 13, 6, 0),
    ct!(0b00000000110, 11, 6, 1),
    ct!(0b0000000101, 10, 6, 2),
    ct!(0b00000100, 8, 6, 3),
    ct!(0b0000000001011, 13, 7, 0),
    ct!(0b0000000001110, 13, 7, 1),
    ct!(0b00000000101, 11, 7, 2),
    ct!(0b000000100, 9, 7, 3),
    ct!(0b0000000001000, 13, 8, 0),
    ct!(0b0000000001010, 13, 8, 1),
    ct!(0b0000000001101, 13, 8, 2),
    ct!(0b0000000100, 10, 8, 3),
    ct!(0b00000000001111, 14, 9, 0),
    ct!(0b00000000001110, 14, 9, 1),
    ct!(0b0000000001001, 13, 9, 2),
    ct!(0b00000000100, 11, 9, 3),
    ct!(0b00000000001011, 14, 10, 0),
    ct!(0b00000000001010, 14, 10, 1),
    ct!(0b00000000001101, 14, 10, 2),
    ct!(0b0000000001100, 13, 10, 3),
    ct!(0b000000000001111, 15, 11, 0),
    ct!(0b000000000001110, 15, 11, 1),
    ct!(0b00000000001001, 14, 11, 2),
    ct!(0b00000000001100, 14, 11, 3),
    ct!(0b000000000001011, 15, 12, 0),
    ct!(0b000000000001010, 15, 12, 1),
    ct!(0b000000000001101, 15, 12, 2),
    ct!(0b00000000001000, 14, 12, 3),
    ct!(0b0000000000001111, 16, 13, 0),
    ct!(0b000000000000001, 15, 13, 1),
    ct!(0b000000000001001, 15, 13, 2),
    ct!(0b000000000001100, 15, 13, 3),
    ct!(0b0000000000001011, 16, 14, 0),
    ct!(0b0000000000001110, 16, 14, 1),
    ct!(0b0000000000001101, 16, 14, 2),
    ct!(0b000000000001000, 15, 14, 3),
    ct!(0b0000000000000111, 16, 15, 0),
    ct!(0b0000000000001010, 16, 15, 1),
    ct!(0b0000000000001001, 16, 15, 2),
    ct!(0b0000000000001100, 16, 15, 3),
    ct!(0b0000000000000100, 16, 16, 0),
    ct!(0b0000000000000110, 16, 16, 1),
    ct!(0b0000000000000101, 16, 16, 2),
    ct!(0b0000000000001000, 16, 16, 3),
];

/// coeff_token table for nC in [2, 4) — H.264 Table 9-5(b).
/// Data sourced from ITU-T H.264 specification.
static COEFF_TOKEN_TABLE_1: [CoeffTokenEntry; 62] = [
    ct!(0b11, 2, 0, 0),
    ct!(0b001011, 6, 1, 0),
    ct!(0b10, 2, 1, 1),
    ct!(0b000111, 6, 2, 0),
    ct!(0b00111, 5, 2, 1),
    ct!(0b011, 3, 2, 2),
    ct!(0b0000111, 7, 3, 0),
    ct!(0b001010, 6, 3, 1),
    ct!(0b001001, 6, 3, 2),
    ct!(0b0101, 4, 3, 3),
    ct!(0b00000111, 8, 4, 0),
    ct!(0b000110, 6, 4, 1),
    ct!(0b000101, 6, 4, 2),
    ct!(0b0100, 4, 4, 3),
    ct!(0b00000100, 8, 5, 0),
    ct!(0b0000110, 7, 5, 1),
    ct!(0b0000101, 7, 5, 2),
    ct!(0b00110, 5, 5, 3),
    ct!(0b000000111, 9, 6, 0),
    ct!(0b00000110, 8, 6, 1),
    ct!(0b00000101, 8, 6, 2),
    ct!(0b001000, 6, 6, 3),
    ct!(0b00000001111, 11, 7, 0),
    ct!(0b000000110, 9, 7, 1),
    ct!(0b000000101, 9, 7, 2),
    ct!(0b000100, 6, 7, 3),
    ct!(0b00000001011, 11, 8, 0),
    ct!(0b00000001110, 11, 8, 1),
    ct!(0b00000001101, 11, 8, 2),
    ct!(0b0000100, 7, 8, 3),
    ct!(0b000000001111, 12, 9, 0),
    ct!(0b00000001010, 11, 9, 1),
    ct!(0b00000001001, 11, 9, 2),
    ct!(0b000000100, 9, 9, 3),
    ct!(0b000000001011, 12, 10, 0),
    ct!(0b000000001110, 12, 10, 1),
    ct!(0b000000001101, 12, 10, 2),
    ct!(0b00000001100, 11, 10, 3),
    ct!(0b000000001000, 12, 11, 0),
    ct!(0b000000001010, 12, 11, 1),
    ct!(0b000000001001, 12, 11, 2),
    ct!(0b00000001000, 11, 11, 3),
    ct!(0b0000000001111, 13, 12, 0),
    ct!(0b0000000001110, 13, 12, 1),
    ct!(0b0000000001101, 13, 12, 2),
    ct!(0b000000001100, 12, 12, 3),
    ct!(0b0000000001011, 13, 13, 0),
    ct!(0b0000000001010, 13, 13, 1),
    ct!(0b0000000001001, 13, 13, 2),
    ct!(0b0000000001100, 13, 13, 3),
    ct!(0b0000000000111, 13, 14, 0),
    ct!(0b00000000001011, 14, 14, 1),
    ct!(0b0000000000110, 13, 14, 2),
    ct!(0b0000000001000, 13, 14, 3),
    ct!(0b00000000001001, 14, 15, 0),
    ct!(0b00000000001000, 14, 15, 1),
    ct!(0b00000000001010, 14, 15, 2),
    ct!(0b0000000000001, 13, 15, 3),
    ct!(0b00000000000111, 14, 16, 0),
    ct!(0b00000000000110, 14, 16, 1),
    ct!(0b00000000000101, 14, 16, 2),
    ct!(0b00000000000100, 14, 16, 3),
];

/// coeff_token table for nC in [4, 8) — H.264 Table 9-5(c).
static COEFF_TOKEN_TABLE_2: [CoeffTokenEntry; 62] = [
    ct!(0b1111, 4, 0, 0),
    ct!(0b001111, 6, 1, 0),
    ct!(0b1110, 4, 1, 1),
    ct!(0b001011, 6, 2, 0),
    ct!(0b01111, 5, 2, 1),
    ct!(0b1101, 4, 2, 2),
    ct!(0b001000, 6, 3, 0),
    ct!(0b01100, 5, 3, 1),
    ct!(0b01110, 5, 3, 2),
    ct!(0b1100, 4, 3, 3),
    ct!(0b0001111, 7, 4, 0),
    ct!(0b01010, 5, 4, 1),
    ct!(0b01011, 5, 4, 2),
    ct!(0b1011, 4, 4, 3),
    ct!(0b0001011, 7, 5, 0),
    ct!(0b01000, 5, 5, 1),
    ct!(0b01001, 5, 5, 2),
    ct!(0b1010, 4, 5, 3),
    ct!(0b0001001, 7, 6, 0),
    ct!(0b001110, 6, 6, 1),
    ct!(0b001101, 6, 6, 2),
    ct!(0b1001, 4, 6, 3),
    ct!(0b0001000, 7, 7, 0),
    ct!(0b001010, 6, 7, 1),
    ct!(0b001001, 6, 7, 2),
    ct!(0b1000, 4, 7, 3),
    ct!(0b00001111, 8, 8, 0),
    ct!(0b0001110, 7, 8, 1),
    ct!(0b0001101, 7, 8, 2),
    ct!(0b01101, 5, 8, 3),
    ct!(0b00001011, 8, 9, 0),
    ct!(0b00001110, 8, 9, 1),
    ct!(0b0001010, 7, 9, 2),
    ct!(0b001100, 6, 9, 3),
    ct!(0b000001111, 9, 10, 0),
    ct!(0b00001010, 8, 10, 1),
    ct!(0b00001101, 8, 10, 2),
    ct!(0b0001100, 7, 10, 3),
    ct!(0b000001011, 9, 11, 0),
    ct!(0b000001110, 9, 11, 1),
    ct!(0b00001001, 8, 11, 2),
    ct!(0b00001100, 8, 11, 3),
    ct!(0b000001000, 9, 12, 0),
    ct!(0b000001010, 9, 12, 1),
    ct!(0b000001101, 9, 12, 2),
    ct!(0b00001000, 8, 12, 3),
    ct!(0b0000001101, 10, 13, 0),
    ct!(0b000000111, 9, 13, 1),
    ct!(0b000001001, 9, 13, 2),
    ct!(0b000001100, 9, 13, 3),
    ct!(0b0000001001, 10, 14, 0),
    ct!(0b0000001100, 10, 14, 1),
    ct!(0b0000001011, 10, 14, 2),
    ct!(0b0000001010, 10, 14, 3),
    ct!(0b0000000101, 10, 15, 0),
    ct!(0b0000001000, 10, 15, 1),
    ct!(0b0000000111, 10, 15, 2),
    ct!(0b0000000110, 10, 15, 3),
    ct!(0b0000000001, 10, 16, 0),
    ct!(0b0000000100, 10, 16, 1),
    ct!(0b0000000011, 10, 16, 2),
    ct!(0b0000000010, 10, 16, 3),
];

/// coeff_token table for nC >= 8 — H.264 Table 9-5(d). Fixed 6-bit codes.
/// Code = `((total_coeffs - 1) << 2) | trailing_ones` for tc >= 1,
/// plus the special tc=0 code.
static COEFF_TOKEN_TABLE_3: [CoeffTokenEntry; 62] = [
    ct!(0b000011, 6, 0, 0),
    ct!(0b000000, 6, 1, 0),
    ct!(0b000001, 6, 1, 1),
    ct!(0b000100, 6, 2, 0),
    ct!(0b000101, 6, 2, 1),
    ct!(0b000110, 6, 2, 2),
    ct!(0b001000, 6, 3, 0),
    ct!(0b001001, 6, 3, 1),
    ct!(0b001010, 6, 3, 2),
    ct!(0b001011, 6, 3, 3),
    ct!(0b001100, 6, 4, 0),
    ct!(0b001101, 6, 4, 1),
    ct!(0b001110, 6, 4, 2),
    ct!(0b001111, 6, 4, 3),
    ct!(0b010000, 6, 5, 0),
    ct!(0b010001, 6, 5, 1),
    ct!(0b010010, 6, 5, 2),
    ct!(0b010011, 6, 5, 3),
    ct!(0b010100, 6, 6, 0),
    ct!(0b010101, 6, 6, 1),
    ct!(0b010110, 6, 6, 2),
    ct!(0b010111, 6, 6, 3),
    ct!(0b011000, 6, 7, 0),
    ct!(0b011001, 6, 7, 1),
    ct!(0b011010, 6, 7, 2),
    ct!(0b011011, 6, 7, 3),
    ct!(0b011100, 6, 8, 0),
    ct!(0b011101, 6, 8, 1),
    ct!(0b011110, 6, 8, 2),
    ct!(0b011111, 6, 8, 3),
    ct!(0b100000, 6, 9, 0),
    ct!(0b100001, 6, 9, 1),
    ct!(0b100010, 6, 9, 2),
    ct!(0b100011, 6, 9, 3),
    ct!(0b100100, 6, 10, 0),
    ct!(0b100101, 6, 10, 1),
    ct!(0b100110, 6, 10, 2),
    ct!(0b100111, 6, 10, 3),
    ct!(0b101000, 6, 11, 0),
    ct!(0b101001, 6, 11, 1),
    ct!(0b101010, 6, 11, 2),
    ct!(0b101011, 6, 11, 3),
    ct!(0b101100, 6, 12, 0),
    ct!(0b101101, 6, 12, 1),
    ct!(0b101110, 6, 12, 2),
    ct!(0b101111, 6, 12, 3),
    ct!(0b110000, 6, 13, 0),
    ct!(0b110001, 6, 13, 1),
    ct!(0b110010, 6, 13, 2),
    ct!(0b110011, 6, 13, 3),
    ct!(0b110100, 6, 14, 0),
    ct!(0b110101, 6, 14, 1),
    ct!(0b110110, 6, 14, 2),
    ct!(0b110111, 6, 14, 3),
    ct!(0b111000, 6, 15, 0),
    ct!(0b111001, 6, 15, 1),
    ct!(0b111010, 6, 15, 2),
    ct!(0b111011, 6, 15, 3),
    ct!(0b111100, 6, 16, 0),
    ct!(0b111101, 6, 16, 1),
    ct!(0b111110, 6, 16, 2),
    ct!(0b111111, 6, 16, 3),
];

/// coeff_token table for chroma DC (nC == -1) — H.264 Table 9-5(e).
/// Data sourced from ITU-T H.264 specification.
static COEFF_TOKEN_CHROMA_DC: [CoeffTokenEntry; 14] = [
    ct!(0b01, 2, 0, 0),
    ct!(0b000111, 6, 1, 0),
    ct!(0b1, 1, 1, 1),
    ct!(0b000100, 6, 2, 0),
    ct!(0b000110, 6, 2, 1),
    ct!(0b001, 3, 2, 2),
    ct!(0b000011, 6, 3, 0),
    ct!(0b0000011, 7, 3, 1),
    ct!(0b0000010, 7, 3, 2),
    ct!(0b000101, 6, 3, 3),
    ct!(0b000010, 6, 4, 0),
    ct!(0b00000011, 8, 4, 1),
    ct!(0b00000010, 8, 4, 2),
    ct!(0b0000000, 7, 4, 3),
];

// ---------------------------------------------------------------------------
// total_zeros VLC Tables (H.264 Table 9-7)
// ---------------------------------------------------------------------------

/// A VLC entry: (code_bits, code_length, value).
#[derive(Debug, Clone, Copy)]
pub struct VlcEntry {
    pub bits: u16,
    pub len: u8,
    pub value: u8,
}

macro_rules! vlc {
    ($bits:expr, $len:expr, $val:expr) => {
        VlcEntry {
            bits: $bits,
            len: $len,
            value: $val,
        }
    };
}

/// Decode a VLC value from the bitstream using the given table.
pub fn decode_vlc(reader: &mut RbspReader<'_>, table: &[VlcEntry]) -> Result<u8, H264Error> {
    let mut accumulated = 0u16;
    let mut bits_read = 0u8;

    for _ in 0..16 {
        let bit = reader.read_bit()? as u16;
        accumulated = (accumulated << 1) | bit;
        bits_read += 1;

        for entry in table {
            if entry.len == bits_read && entry.bits == accumulated {
                return Ok(entry.value);
            }
        }
    }

    Err(H264Error::CavlcError("no matching VLC code found".into()))
}

/// total_zeros tables indexed by total_coeffs (1..=15).
/// Returns the table for `total_coeffs`, or None if out of range.
pub fn total_zeros_table(total_coeffs: u8) -> &'static [VlcEntry] {
    match total_coeffs {
        1 => &TOTAL_ZEROS_1,
        2 => &TOTAL_ZEROS_2,
        3 => &TOTAL_ZEROS_3,
        4 => &TOTAL_ZEROS_4,
        5 => &TOTAL_ZEROS_5,
        6 => &TOTAL_ZEROS_6,
        7 => &TOTAL_ZEROS_7,
        8 => &TOTAL_ZEROS_8,
        9 => &TOTAL_ZEROS_9,
        10 => &TOTAL_ZEROS_10,
        11 => &TOTAL_ZEROS_11,
        12 => &TOTAL_ZEROS_12,
        13 => &TOTAL_ZEROS_13,
        14 => &TOTAL_ZEROS_14,
        15 => &TOTAL_ZEROS_15,
        _ => &[], // tc=0 or tc=16: no total_zeros coded
    }
}

// H.264 Table 9-7: total_zeros for 4x4 blocks, total_coeffs = 1
static TOTAL_ZEROS_1: [VlcEntry; 16] = [
    vlc!(0b1, 1, 0),
    vlc!(0b011, 3, 1),
    vlc!(0b010, 3, 2),
    vlc!(0b0011, 4, 3),
    vlc!(0b0010, 4, 4),
    vlc!(0b00011, 5, 5),
    vlc!(0b00010, 5, 6),
    vlc!(0b000011, 6, 7),
    vlc!(0b000010, 6, 8),
    vlc!(0b0000011, 7, 9),
    vlc!(0b0000010, 7, 10),
    vlc!(0b00000011, 8, 11),
    vlc!(0b00000010, 8, 12),
    vlc!(0b000000011, 9, 13),
    vlc!(0b000000010, 9, 14),
    vlc!(0b000000001, 9, 15),
];

static TOTAL_ZEROS_2: [VlcEntry; 15] = [
    vlc!(0b111, 3, 0),
    vlc!(0b110, 3, 1),
    vlc!(0b101, 3, 2),
    vlc!(0b100, 3, 3),
    vlc!(0b011, 3, 4),
    vlc!(0b0101, 4, 5),
    vlc!(0b0100, 4, 6),
    vlc!(0b0011, 4, 7),
    vlc!(0b0010, 4, 8),
    vlc!(0b00011, 5, 9),
    vlc!(0b00010, 5, 10),
    vlc!(0b000011, 6, 11),
    vlc!(0b000010, 6, 12),
    vlc!(0b000001, 6, 13),
    vlc!(0b000000, 6, 14),
];

static TOTAL_ZEROS_3: [VlcEntry; 14] = [
    vlc!(0b0101, 4, 0),
    vlc!(0b111, 3, 1),
    vlc!(0b110, 3, 2),
    vlc!(0b101, 3, 3),
    vlc!(0b0100, 4, 4),
    vlc!(0b0011, 4, 5),
    vlc!(0b100, 3, 6),
    vlc!(0b011, 3, 7),
    vlc!(0b0010, 4, 8),
    vlc!(0b00011, 5, 9),
    vlc!(0b00010, 5, 10),
    vlc!(0b000001, 6, 11),
    vlc!(0b00001, 5, 12),
    vlc!(0b000000, 6, 13),
];

static TOTAL_ZEROS_4: [VlcEntry; 13] = [
    vlc!(0b00011, 5, 0),
    vlc!(0b111, 3, 1),
    vlc!(0b0101, 4, 2),
    vlc!(0b0100, 4, 3),
    vlc!(0b110, 3, 4),
    vlc!(0b101, 3, 5),
    vlc!(0b100, 3, 6),
    vlc!(0b0011, 4, 7),
    vlc!(0b011, 3, 8),
    vlc!(0b0010, 4, 9),
    vlc!(0b00010, 5, 10),
    vlc!(0b00001, 5, 11),
    vlc!(0b00000, 5, 12),
];

static TOTAL_ZEROS_5: [VlcEntry; 12] = [
    vlc!(0b0101, 4, 0),
    vlc!(0b0100, 4, 1),
    vlc!(0b0011, 4, 2),
    vlc!(0b111, 3, 3),
    vlc!(0b110, 3, 4),
    vlc!(0b101, 3, 5),
    vlc!(0b100, 3, 6),
    vlc!(0b011, 3, 7),
    vlc!(0b0010, 4, 8),
    vlc!(0b00001, 5, 9),
    vlc!(0b0001, 4, 10),
    vlc!(0b00000, 5, 11),
];

static TOTAL_ZEROS_6: [VlcEntry; 11] = [
    vlc!(0b000001, 6, 0),
    vlc!(0b00001, 5, 1),
    vlc!(0b111, 3, 2),
    vlc!(0b110, 3, 3),
    vlc!(0b101, 3, 4),
    vlc!(0b100, 3, 5),
    vlc!(0b011, 3, 6),
    vlc!(0b010, 3, 7),
    vlc!(0b0001, 4, 8),
    vlc!(0b001, 3, 9),
    vlc!(0b000000, 6, 10),
];

static TOTAL_ZEROS_7: [VlcEntry; 10] = [
    vlc!(0b000001, 6, 0),
    vlc!(0b00001, 5, 1),
    vlc!(0b101, 3, 2),
    vlc!(0b100, 3, 3),
    vlc!(0b011, 3, 4),
    vlc!(0b11, 2, 5),
    vlc!(0b010, 3, 6),
    vlc!(0b0001, 4, 7),
    vlc!(0b001, 3, 8),
    vlc!(0b000000, 6, 9),
];

static TOTAL_ZEROS_8: [VlcEntry; 9] = [
    vlc!(0b000001, 6, 0),
    vlc!(0b0001, 4, 1),
    vlc!(0b00001, 5, 2),
    vlc!(0b011, 3, 3),
    vlc!(0b11, 2, 4),
    vlc!(0b10, 2, 5),
    vlc!(0b010, 3, 6),
    vlc!(0b001, 3, 7),
    vlc!(0b000000, 6, 8),
];

static TOTAL_ZEROS_9: [VlcEntry; 8] = [
    vlc!(0b000001, 6, 0),
    vlc!(0b000000, 6, 1),
    vlc!(0b0001, 4, 2),
    vlc!(0b11, 2, 3),
    vlc!(0b10, 2, 4),
    vlc!(0b001, 3, 5),
    vlc!(0b01, 2, 6),
    vlc!(0b00001, 5, 7),
];

static TOTAL_ZEROS_10: [VlcEntry; 7] = [
    vlc!(0b00001, 5, 0),
    vlc!(0b00000, 5, 1),
    vlc!(0b001, 3, 2),
    vlc!(0b11, 2, 3),
    vlc!(0b10, 2, 4),
    vlc!(0b01, 2, 5),
    vlc!(0b0001, 4, 6),
];

static TOTAL_ZEROS_11: [VlcEntry; 6] = [
    vlc!(0b0000, 4, 0),
    vlc!(0b0001, 4, 1),
    vlc!(0b001, 3, 2),
    vlc!(0b010, 3, 3),
    vlc!(0b1, 1, 4),
    vlc!(0b011, 3, 5),
];

static TOTAL_ZEROS_12: [VlcEntry; 5] = [
    vlc!(0b0000, 4, 0),
    vlc!(0b0001, 4, 1),
    vlc!(0b01, 2, 2),
    vlc!(0b1, 1, 3),
    vlc!(0b001, 3, 4),
];

static TOTAL_ZEROS_13: [VlcEntry; 4] = [
    vlc!(0b000, 3, 0),
    vlc!(0b001, 3, 1),
    vlc!(0b1, 1, 2),
    vlc!(0b01, 2, 3),
];

static TOTAL_ZEROS_14: [VlcEntry; 3] = [
    vlc!(0b00, 2, 0),
    vlc!(0b01, 2, 1),
    vlc!(0b1, 1, 2),
];

static TOTAL_ZEROS_15: [VlcEntry; 2] = [
    vlc!(0b0, 1, 0),
    vlc!(0b1, 1, 1),
];

// ---------------------------------------------------------------------------
// run_before VLC Tables (H.264 Table 9-10)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Chroma DC total_zeros VLC Tables (H.264 Table 9-9a, for 4:2:0)
// Used when max_coeffs <= 4 (chroma DC 2x2 blocks)
// ---------------------------------------------------------------------------

/// Chroma DC total_zeros table for total_coeffs = 1 (range 0..3).
static CHROMA_DC_TOTAL_ZEROS_1: [VlcEntry; 4] = [
    vlc!(0b1, 1, 0),
    vlc!(0b01, 2, 1),
    vlc!(0b001, 3, 2),
    vlc!(0b000, 3, 3),
];

/// Chroma DC total_zeros table for total_coeffs = 2 (range 0..2).
static CHROMA_DC_TOTAL_ZEROS_2: [VlcEntry; 3] = [
    vlc!(0b1, 1, 0),
    vlc!(0b01, 2, 1),
    vlc!(0b00, 2, 2),
];

/// Chroma DC total_zeros table for total_coeffs = 3 (range 0..1).
static CHROMA_DC_TOTAL_ZEROS_3: [VlcEntry; 2] = [
    vlc!(0b1, 1, 0),
    vlc!(0b0, 1, 1),
];

/// Select total_zeros table based on total_coeffs AND max_coeffs.
///
/// For chroma DC blocks (max_coeffs <= 4), uses the separate Table 9-9a.
/// For luma blocks (max_coeffs > 4), uses the regular Table 9-7.
pub fn total_zeros_table_for(total_coeffs: u8, max_coeffs: u8) -> &'static [VlcEntry] {
    if max_coeffs <= 4 {
        // Chroma DC (4:2:0): Table 9-9a
        match total_coeffs {
            1 => &CHROMA_DC_TOTAL_ZEROS_1,
            2 => &CHROMA_DC_TOTAL_ZEROS_2,
            3 => &CHROMA_DC_TOTAL_ZEROS_3,
            _ => &[], // tc=0 or tc=4: no total_zeros coded
        }
    } else {
        // Regular luma: Table 9-7
        total_zeros_table(total_coeffs)
    }
}

// ---------------------------------------------------------------------------
// run_before VLC Tables (H.264 Table 9-10)
// ---------------------------------------------------------------------------

/// run_before table indexed by zeros_left (1..=6, 7+ uses the same table).
pub fn run_before_table(zeros_left: u8) -> &'static [VlcEntry] {
    match zeros_left {
        1 => &RUN_BEFORE_1,
        2 => &RUN_BEFORE_2,
        3 => &RUN_BEFORE_3,
        4 => &RUN_BEFORE_4,
        5 => &RUN_BEFORE_5,
        6 => &RUN_BEFORE_6,
        _ => &RUN_BEFORE_7PLUS, // zeros_left >= 7
    }
}

static RUN_BEFORE_1: [VlcEntry; 2] = [
    vlc!(0b1, 1, 0),
    vlc!(0b0, 1, 1),
];

static RUN_BEFORE_2: [VlcEntry; 3] = [
    vlc!(0b1, 1, 0),
    vlc!(0b01, 2, 1),
    vlc!(0b00, 2, 2),
];

static RUN_BEFORE_3: [VlcEntry; 4] = [
    vlc!(0b11, 2, 0),
    vlc!(0b10, 2, 1),
    vlc!(0b01, 2, 2),
    vlc!(0b00, 2, 3),
];

static RUN_BEFORE_4: [VlcEntry; 5] = [
    vlc!(0b11, 2, 0),
    vlc!(0b10, 2, 1),
    vlc!(0b01, 2, 2),
    vlc!(0b001, 3, 3),
    vlc!(0b000, 3, 4),
];

static RUN_BEFORE_5: [VlcEntry; 6] = [
    vlc!(0b11, 2, 0),
    vlc!(0b10, 2, 1),
    vlc!(0b011, 3, 2),
    vlc!(0b010, 3, 3),
    vlc!(0b001, 3, 4),
    vlc!(0b000, 3, 5),
];

static RUN_BEFORE_6: [VlcEntry; 7] = [
    vlc!(0b11, 2, 0),
    vlc!(0b000, 3, 1),
    vlc!(0b001, 3, 2),
    vlc!(0b011, 3, 3),
    vlc!(0b010, 3, 4),
    vlc!(0b101, 3, 5),
    vlc!(0b100, 3, 6),
];

/// For zeros_left >= 7: run_before 0..6 use 3-bit codes, then leading zeros.
/// Max run_before for 4x4 is 14 (when tc=1 & tz=14). Value 15 does not occur.
static RUN_BEFORE_7PLUS: [VlcEntry; 15] = [
    vlc!(0b111, 3, 0),
    vlc!(0b110, 3, 1),
    vlc!(0b101, 3, 2),
    vlc!(0b100, 3, 3),
    vlc!(0b011, 3, 4),
    vlc!(0b010, 3, 5),
    vlc!(0b001, 3, 6),
    vlc!(0b0001, 4, 7),
    vlc!(0b00001, 5, 8),
    vlc!(0b000001, 6, 9),
    vlc!(0b0000001, 7, 10),
    vlc!(0b00000001, 8, 11),
    vlc!(0b000000001, 9, 12),
    vlc!(0b0000000001, 10, 13),
    vlc!(0b00000000001, 11, 14),
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Validate a VLC table (total_zeros / run_before): prefix-free + bits fit len.
#[cfg(test)]
fn validate_vlc_table(name: &str, table: &[VlcEntry]) -> Vec<String> {
    let mut errors = Vec::new();
    for (i, entry) in table.iter().enumerate() {
        if entry.len == 0 || entry.len > 16 {
            errors.push(format!("{name}[{i}]: invalid len={}", entry.len));
            continue;
        }
        let max = if entry.len == 16 { 0xFFFF } else { (1u16 << entry.len) - 1 };
        if entry.bits > max {
            errors.push(format!(
                "{name}[{i}]: bits={:#b} exceeds {}-bit field (value={})",
                entry.bits, entry.len, entry.value
            ));
        }
    }
    for (i, a) in table.iter().enumerate() {
        for (j, b) in table.iter().enumerate() {
            if i >= j || a.len == 0 || b.len == 0 { continue; }
            let (shorter, longer) = if a.len <= b.len { (a, b) } else { (b, a) };
            let shift = longer.len - shorter.len;
            if (longer.bits >> shift) == shorter.bits {
                errors.push(format!(
                    "{name}: value={} len={} {:#b} prefix of value={} len={} {:#b}",
                    shorter.value, shorter.len, shorter.bits,
                    longer.value, longer.len, longer.bits,
                ));
            }
        }
    }
    errors
}

/// Validate a coeff_token table: every entry must have:
/// 1. `bits < (1 << len)` (bits fits in `len` bits)
/// 2. No entry's code is a prefix of any other entry's code
#[cfg(test)]
fn validate_coeff_token_table(
    name: &str,
    table: &[CoeffTokenEntry],
) -> Vec<String> {
    let mut errors = Vec::new();

    // Check 1: bits fit in len bits
    for (i, entry) in table.iter().enumerate() {
        if entry.len == 0 || entry.len > 16 {
            errors.push(format!(
                "{name}[{i}]: invalid len={} for (tc={}, t1={})",
                entry.len, entry.total_coeffs, entry.trailing_ones
            ));
            continue;
        }
        let max_val = if entry.len == 16 {
            0xFFFFu16
        } else {
            (1u16 << entry.len) - 1
        };
        if entry.bits > max_val {
            errors.push(format!(
                "{name}[{i}]: bits={:#b} exceeds {}-bit field for (tc={}, t1={})",
                entry.bits, entry.len, entry.total_coeffs, entry.trailing_ones
            ));
        }
    }

    // Check 2: prefix-freeness
    for (i, a) in table.iter().enumerate() {
        for (j, b) in table.iter().enumerate() {
            if i >= j {
                continue;
            }
            if a.len == 0 || b.len == 0 { continue; }
            let (shorter, longer) = if a.len <= b.len { (a, b) } else { (b, a) };
            let shift = longer.len - shorter.len;
            if (longer.bits >> shift) == shorter.bits {
                errors.push(format!(
                    "{name}: prefix conflict — ({},{}) len={} {:#b} is prefix of ({},{}) len={} {:#b}",
                    shorter.total_coeffs, shorter.trailing_ones, shorter.len, shorter.bits,
                    longer.total_coeffs, longer.trailing_ones, longer.len, longer.bits,
                ));
            }
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// Encoder-side reverse lookups (Phase 6A.4)
// ---------------------------------------------------------------------------

/// Encoder-side lookup for coeff_token. Mirrors `decode_coeff_token`.
///
/// Returns `(bits, len)` for the VLC that encodes `(total_coeffs,
/// trailing_ones)` under the nC-selected table, or `None` if the
/// combination is not in the spec.
pub fn encode_coeff_token(total_coeffs: u8, trailing_ones: u8, nc: i8) -> Option<(u16, u8)> {
    let table = select_coeff_token_table(nc);
    for entry in table {
        if entry.total_coeffs == total_coeffs && entry.trailing_ones == trailing_ones {
            return Some((entry.bits, entry.len));
        }
    }
    None
}

/// Encoder-side lookup for total_zeros. Mirrors `decode_vlc` +
/// `total_zeros_table_for`.
///
/// Returns `(bits, len)` or `None` if out of range. `max_coeffs`
/// selects the chroma-DC table variant when ≤ 4.
pub fn encode_total_zeros(
    total_zeros: u8,
    total_coeffs: u8,
    max_coeffs: u8,
) -> Option<(u16, u8)> {
    let table = total_zeros_table_for(total_coeffs, max_coeffs);
    for entry in table {
        if entry.value == total_zeros {
            return Some((entry.bits, entry.len));
        }
    }
    None
}

/// Encoder-side lookup for run_before. Mirrors `decode_vlc` +
/// `run_before_table`.
pub fn encode_run_before(run_before: u8, zeros_left: u8) -> Option<(u16, u8)> {
    let table = run_before_table(zeros_left);
    for entry in table {
        if entry.value == run_before {
            return Some((entry.bits, entry.len));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_4x4_inverse() {
        for i in 0..16u8 {
            assert_eq!(ZIGZAG_4X4_INV[ZIGZAG_4X4[i as usize] as usize], i);
        }
    }

    #[test]
    fn coeff_token_table3_fixed_length() {
        // nC >= 8: all codes are 6 bits
        for entry in &COEFF_TOKEN_TABLE_3 {
            assert_eq!(entry.len, 6, "Table 3 entry with len != 6: tc={}, t1={}", entry.total_coeffs, entry.trailing_ones);
        }
    }

    #[test]
    fn coeff_token_table3_tc0() {
        // tc=0, T1=0 should be code 0b000011
        let entry = &COEFF_TOKEN_TABLE_3[0];
        assert_eq!(entry.total_coeffs, 0);
        assert_eq!(entry.trailing_ones, 0);
        assert_eq!(entry.bits, 0b000011);
    }

    #[test]
    fn decode_coeff_token_nc0_tc0() {
        // nC=0, tc=0, T1=0: code = "1" (1 bit)
        let data = [0b1000_0000];
        let mut reader = RbspReader::new(&data);
        let (tc, t1) = decode_coeff_token(&mut reader, 0).unwrap();
        assert_eq!(tc, 0);
        assert_eq!(t1, 0);
        assert_eq!(reader.bits_read(), 1);
    }

    #[test]
    fn decode_coeff_token_nc0_tc1_t1_1() {
        // nC=0, tc=1, T1=1: code = "01" (2 bits)
        let data = [0b0100_0000];
        let mut reader = RbspReader::new(&data);
        let (tc, t1) = decode_coeff_token(&mut reader, 0).unwrap();
        assert_eq!(tc, 1);
        assert_eq!(t1, 1);
        assert_eq!(reader.bits_read(), 2);
    }

    #[test]
    fn decode_coeff_token_nc0_tc2_t1_2() {
        // nC=0, tc=2, T1=2: code = "001" (3 bits)
        let data = [0b0010_0000];
        let mut reader = RbspReader::new(&data);
        let (tc, t1) = decode_coeff_token(&mut reader, 0).unwrap();
        assert_eq!(tc, 2);
        assert_eq!(t1, 2);
        assert_eq!(reader.bits_read(), 3);
    }

    #[test]
    fn decode_coeff_token_nc8_fixed() {
        // nC=8: 6-bit FLC. tc=1, T1=1 = code 0b000001
        let data = [0b000001_00];
        let mut reader = RbspReader::new(&data);
        let (tc, t1) = decode_coeff_token(&mut reader, 8).unwrap();
        assert_eq!(tc, 1);
        assert_eq!(t1, 1);
        assert_eq!(reader.bits_read(), 6);
    }

    #[test]
    fn run_before_zeros_left_1() {
        // zeros_left=1: run=0 → "1", run=1 → "0"
        let data = [0b1000_0000];
        let mut reader = RbspReader::new(&data);
        let val = decode_vlc(&mut reader, run_before_table(1)).unwrap();
        assert_eq!(val, 0);

        let data = [0b0000_0000];
        let mut reader = RbspReader::new(&data);
        let val = decode_vlc(&mut reader, run_before_table(1)).unwrap();
        assert_eq!(val, 1);
    }

    #[test]
    fn total_zeros_tc1_tz0() {
        // total_coeffs=1, total_zeros=0: code = "1" (1 bit)
        let data = [0b1000_0000];
        let mut reader = RbspReader::new(&data);
        let val = decode_vlc(&mut reader, total_zeros_table(1)).unwrap();
        assert_eq!(val, 0);
    }

    // -- Table integrity validation --

    #[test]
    fn validate_coeff_token_table_0() {
        let errors = validate_coeff_token_table("TABLE_0", &COEFF_TOKEN_TABLE_0);
        if !errors.is_empty() {
            for e in &errors {
                println!("{e}");
            }
            panic!("TABLE_0 has {} validation errors", errors.len());
        }
    }

    #[test]
    fn validate_coeff_token_table_1() {
        let errors = validate_coeff_token_table("TABLE_1", &COEFF_TOKEN_TABLE_1);
        if !errors.is_empty() {
            for e in &errors {
                println!("{e}");
            }
            panic!("TABLE_1 has {} validation errors", errors.len());
        }
    }

    #[test]
    fn validate_coeff_token_table_2() {
        let errors = validate_coeff_token_table("TABLE_2", &COEFF_TOKEN_TABLE_2);
        if !errors.is_empty() {
            for e in &errors {
                println!("{e}");
            }
            panic!("TABLE_2 has {} validation errors", errors.len());
        }
    }

    #[test]
    fn validate_coeff_token_chroma_dc() {
        let errors = validate_coeff_token_table("CHROMA_DC", &COEFF_TOKEN_CHROMA_DC);
        if !errors.is_empty() {
            for e in &errors { println!("{e}"); }
            panic!("CHROMA_DC has {} validation errors", errors.len());
        }
    }

    #[test]
    fn validate_all_total_zeros_tables() {
        let tables: &[(&str, &[VlcEntry])] = &[
            ("TZ_1", &TOTAL_ZEROS_1),
            ("TZ_2", &TOTAL_ZEROS_2),
            ("TZ_3", &TOTAL_ZEROS_3),
            ("TZ_4", &TOTAL_ZEROS_4),
            ("TZ_5", &TOTAL_ZEROS_5),
            ("TZ_6", &TOTAL_ZEROS_6),
            ("TZ_7", &TOTAL_ZEROS_7),
            ("TZ_8", &TOTAL_ZEROS_8),
            ("TZ_9", &TOTAL_ZEROS_9),
            ("TZ_10", &TOTAL_ZEROS_10),
            ("TZ_11", &TOTAL_ZEROS_11),
            ("TZ_12", &TOTAL_ZEROS_12),
            ("TZ_13", &TOTAL_ZEROS_13),
            ("TZ_14", &TOTAL_ZEROS_14),
            ("TZ_15", &TOTAL_ZEROS_15),
            ("CHROMA_DC_TZ_1", &CHROMA_DC_TOTAL_ZEROS_1),
            ("CHROMA_DC_TZ_2", &CHROMA_DC_TOTAL_ZEROS_2),
            ("CHROMA_DC_TZ_3", &CHROMA_DC_TOTAL_ZEROS_3),
        ];
        let mut all_errors = Vec::new();
        for (name, t) in tables {
            all_errors.extend(validate_vlc_table(name, t));
        }
        if !all_errors.is_empty() {
            for e in &all_errors { println!("{e}"); }
            panic!("total_zeros tables have {} errors", all_errors.len());
        }
    }

    #[test]
    fn validate_all_run_before_tables() {
        let tables: &[(&str, &[VlcEntry])] = &[
            ("RB_1", &RUN_BEFORE_1),
            ("RB_2", &RUN_BEFORE_2),
            ("RB_3", &RUN_BEFORE_3),
            ("RB_4", &RUN_BEFORE_4),
            ("RB_5", &RUN_BEFORE_5),
            ("RB_6", &RUN_BEFORE_6),
            ("RB_7PLUS", &RUN_BEFORE_7PLUS),
        ];
        let mut all_errors = Vec::new();
        for (name, t) in tables {
            all_errors.extend(validate_vlc_table(name, t));
        }
        if !all_errors.is_empty() {
            for e in &all_errors { println!("{e}"); }
            panic!("run_before tables have {} errors", all_errors.len());
        }
    }
}
