// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Huffman coding tables for JPEG entropy decoding and encoding.

use super::bitio::BitReader;
use super::error::{JpegError, Result};

/// Perf-bench reference — replicates the older 8-bit fast table +
/// linear-scan slow path. Doc-hidden, kept around solely so
/// `perf_t37_huffman` can compare wall-clock against the new path
/// in the same binary.
#[doc(hidden)]
pub struct LegacyHuffmanDecodeTable {
    fast: [(u8, u8); 256],
    slow: Vec<(u16, u8, u8)>,
    max_len: u8,
}

#[doc(hidden)]
impl LegacyHuffmanDecodeTable {
    pub fn build(bits: &[u8; 16], huffval: &[u8]) -> Result<Self> {
        let mut fast = [(0u8, 0u8); 256];
        let mut slow = Vec::new();
        let mut max_len = 0u8;
        let mut code: u32 = 0;
        let mut si = 0;
        for length in 1..=16u8 {
            let count = bits[(length - 1) as usize] as usize;
            for _ in 0..count {
                if si >= huffval.len() {
                    return Err(JpegError::InvalidMarkerData("DHT symbol count mismatch"));
                }
                let symbol = huffval[si];
                si += 1;
                max_len = length;
                if length <= 8 {
                    let base = (code << (8 - length)) as usize;
                    let fill = 1usize << (8 - length);
                    for j in 0..fill {
                        fast[base + j] = (symbol, length);
                    }
                } else {
                    slow.push((code as u16, length, symbol));
                }
                code += 1;
            }
            code <<= 1;
        }
        Ok(Self { fast, slow, max_len })
    }

    pub fn decode(&self, reader: &mut BitReader) -> Result<u8> {
        let peek_len = 8.min(self.max_len.max(1));
        let peek = reader.peek_bits(peek_len)?;
        let idx = if self.max_len >= 8 {
            peek as usize
        } else {
            (peek << (8 - self.max_len)) as usize
        };
        let (symbol, length) = self.fast[idx];
        if length > 0 {
            reader.skip_bits(length);
            return Ok(symbol);
        }
        for &(code, length, symbol) in &self.slow {
            let bits = reader.peek_bits(length)?;
            if bits == code {
                reader.skip_bits(length);
                return Ok(symbol);
            }
        }
        Err(JpegError::HuffmanDecode)
    }
}

#[doc(hidden)]
pub fn perf_legacy_huffman_decode_n(
    table: &LegacyHuffmanDecodeTable,
    reader: &mut BitReader,
    n: usize,
) -> Result<u64> {
    let mut checksum: u64 = 0;
    for _ in 0..n {
        let sym = table.decode(reader)?;
        checksum = checksum.wrapping_add(sym as u64);
    }
    Ok(checksum)
}

/// Perf-bench helper. doc-hidden, pub so `perf_t37_huffman`
/// can decode a synthetic bitstream via the new path. The bench
/// compares wall-clock against the older 8-bit-fast / linear-
/// scan path captured by git history; this binary always runs the
/// new 10-bit fast path.
#[doc(hidden)]
pub fn perf_huffman_decode_n(
    table: &HuffmanDecodeTable,
    reader: &mut BitReader,
    n: usize,
) -> Result<u64> {
    let mut checksum: u64 = 0;
    for _ in 0..n {
        let sym = table.decode(reader)?;
        checksum = checksum.wrapping_add(sym as u64);
    }
    Ok(checksum)
}

/// Width of the fast lookup table in bits.
///
/// Bumped from 8 → 10. For typical JPEG content, this drives the
/// slow-path hit rate from ~10 % down to ~1 %. Memory: 1024 × 2 B =
/// 2 KB per table × ~4 tables per decode = 8 KB total — fits L1 on
/// every supported target.
const FAST_BITS: u8 = 10;
const FAST_SIZE: usize = 1 << FAST_BITS;

/// Huffman decode table with two-level lookup.
///
/// **Level 1 (fast)** — `FAST_BITS`-bit indexed lookup. Entries
/// `(symbol, code_length)`. `code_length == 0` is the sentinel
/// meaning "fall through to the slow path".
///
/// **Level 2 (slow, per-length canonical lookup)** — replaces
/// an earlier linear scan of `Vec<(code, length, symbol)>` with a
/// per-length range check + array index, matching libjpeg's
/// `dc/ac_derived_tbl` shape. For each length `L > FAST_BITS`, the
/// canonical Huffman codes at that length live in `[code_min[L],
/// code_max[L]]`; the symbol is `huffval[huffval_offset[L] + peek_L
/// − code_min[L]]`. O(1) per length tried, ≤ 6 lengths to try, so
/// O(1) total per slow-path symbol — vs O(slow.len()) (up to ~50
/// in the earlier code) for the linear scan.
pub struct HuffmanDecodeTable {
    /// FAST_BITS-bit fast lookup, indexed by the top `FAST_BITS`
    /// bits of the bitstream. `length == 0` → use slow path.
    fast: [(u8, u8); FAST_SIZE],
    /// Symbols in canonical-Huffman order. `huffval[huffval_offset[L]
    /// .. huffval_offset[L] + count[L]]` is the symbol set for codes
    /// of length L.
    huffval: Vec<u8>,
    /// Smallest canonical code at length L (indexed by length 0..=16).
    /// Used only for L > FAST_BITS by the slow path.
    code_min: [u32; 17],
    /// Largest canonical code at length L (inclusive). When `count[L]
    /// == 0`, set so that the range check `code_min ≤ x ≤ code_max`
    /// is always false.
    code_max: [u32; 17],
    /// Index into `huffval` where length-L symbols begin.
    huffval_offset: [u16; 17],
    /// Maximum code length in this table. Used to bound the slow-path
    /// length iteration.
    max_len: u8,
}

impl HuffmanDecodeTable {
    /// Build a decode table from JPEG-style counts and symbols.
    ///
    /// `bits`: counts[i] = number of codes of length i+1 (16 entries).
    /// `huffval`: the symbols, in order of increasing code length.
    pub fn build(bits: &[u8; 16], huffval_in: &[u8]) -> Result<Self> {
        let mut fast = [(0u8, 0u8); FAST_SIZE];
        let mut huffval: Vec<u8> = Vec::with_capacity(huffval_in.len());
        let mut code_min = [0u32; 17];
        let mut code_max = [0u32; 17];
        let mut huffval_offset = [0u16; 17];
        let mut max_len = 0u8;

        // Generate canonical Huffman codes per ITU-T T.81 Annex C.
        let mut code: u32 = 0;
        let mut si = 0; // symbol index into huffval_in

        for length in 1..=16u8 {
            let count = bits[(length - 1) as usize] as usize;
            huffval_offset[length as usize] = huffval.len() as u16;
            if count == 0 {
                // Sentinel: u32::MAX for both bounds makes the range
                // check `bits_l >= u32::MAX && bits_l <= u32::MAX`
                // false for any peek (which is ≤ 0xFFFF). Empty
                // lengths never match.
                code_min[length as usize] = u32::MAX;
                code_max[length as usize] = u32::MAX;
            } else {
                code_min[length as usize] = code;
                code_max[length as usize] = code + count as u32 - 1;
            }

            for _ in 0..count {
                if si >= huffval_in.len() {
                    return Err(JpegError::InvalidMarkerData("DHT symbol count mismatch"));
                }
                let symbol = huffval_in[si];
                si += 1;
                huffval.push(symbol);
                max_len = length;

                if length <= FAST_BITS {
                    // Fill fast table: this code, left-shifted to
                    // FAST_BITS bits, covers 2^(FAST_BITS-length) entries.
                    let base = (code << (FAST_BITS - length)) as usize;
                    let fill = 1usize << (FAST_BITS - length);
                    for j in 0..fill {
                        fast[base + j] = (symbol, length);
                    }
                }
                // For length > FAST_BITS we don't fill fast; the
                // slow path's per-length range check picks them up.
                code += 1;
            }
            code <<= 1;
        }

        Ok(Self { fast, huffval, code_min, code_max, huffval_offset, max_len })
    }

    /// Decode one Huffman symbol from the bit stream.
    #[inline]
    pub fn decode(&self, reader: &mut BitReader) -> Result<u8> {
        // Peek up to FAST_BITS bits for the fast-table lookup.
        let peek_len = FAST_BITS.min(self.max_len.max(1));
        let peek = reader.peek_bits(peek_len)?;
        let idx = if self.max_len >= FAST_BITS {
            peek as usize
        } else {
            (peek << (FAST_BITS - self.max_len)) as usize
        };

        let (symbol, length) = self.fast[idx];
        if length > 0 {
            reader.skip_bits(length);
            return Ok(symbol);
        }

        // Slow path: codes longer than FAST_BITS. Per-length canonical
        // range check is O(1) per length; we try up to (max_len -
        // FAST_BITS) lengths (≤ 6 for the 16-bit ceiling).
        self.decode_slow(reader)
    }

    fn decode_slow(&self, reader: &mut BitReader) -> Result<u8> {
        for length in (FAST_BITS + 1)..=self.max_len {
            let bits_l = reader.peek_bits(length)? as u32;
            let lo = self.code_min[length as usize];
            let hi = self.code_max[length as usize];
            if bits_l >= lo && bits_l <= hi {
                reader.skip_bits(length);
                let offset = self.huffval_offset[length as usize] as usize
                    + (bits_l - lo) as usize;
                if offset >= self.huffval.len() {
                    return Err(JpegError::HuffmanDecode);
                }
                return Ok(self.huffval[offset]);
            }
        }
        Err(JpegError::HuffmanDecode)
    }
}

/// Huffman encode table: maps symbol → (code_bits, code_length).
pub struct HuffmanEncodeTable {
    /// For each of the 256 possible symbols: (code, length).
    /// Length 0 means the symbol is not in the table.
    table: [(u16, u8); 256],
}

impl HuffmanEncodeTable {
    /// Build an encode table from JPEG-style counts and symbols.
    pub fn build(bits: &[u8; 16], huffval: &[u8]) -> Self {
        let mut table = [(0u16, 0u8); 256];
        let mut code: u32 = 0;
        let mut si = 0;

        for length in 1..=16u8 {
            let count = bits[(length - 1) as usize] as usize;
            for _ in 0..count {
                if si < huffval.len() {
                    let symbol = huffval[si] as usize;
                    table[symbol] = (code as u16, length);
                    si += 1;
                }
                code += 1;
            }
            code <<= 1;
        }

        Self { table }
    }

    /// Encode a symbol: returns (code_bits, code_length).
    /// Returns `Err` if the symbol has no code in this table.
    pub fn encode(&self, symbol: u8) -> Result<(u16, u8)> {
        let (code, len) = self.table[symbol as usize];
        if len == 0 {
            Err(JpegError::InvalidMarkerData(
                "Huffman table missing code for symbol",
            ))
        } else {
            Ok((code, len))
        }
    }
}

/// Extend a signed value from its JPEG "additional bits" representation.
///
/// Per ITU-T T.81 Table F.1: if the high bit is 0, the value is negative.
pub fn extend_sign(value: u16, bits: u8) -> i16 {
    if bits == 0 {
        return 0;
    }
    let half = 1i32 << (bits - 1);
    if (value as i32) < half {
        // Negative value
        (value as i32 - (1i32 << bits) + 1) as i16
    } else {
        value as i16
    }
}

/// Encode a signed value into JPEG "additional bits" representation.
/// Returns (magnitude_bits, category/size).
pub fn encode_value(value: i16) -> (u16, u8) {
    if value == 0 {
        return (0, 0);
    }
    let abs = value.unsigned_abs();
    let size = 16 - abs.leading_zeros() as u8;
    let bits = if value > 0 {
        value as u16
    } else {
        // For negative values, JPEG uses one's complement
        (value - 1) as u16
    };
    (bits & ((1u16 << size) - 1), size)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Standard JPEG luminance DC Huffman table (ITU-T T.81 Table K.3)
    fn lum_dc_table() -> ([u8; 16], Vec<u8>) {
        let bits = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
        let vals = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        (bits, vals)
    }

    #[test]
    fn build_decode_table() {
        let (bits, vals) = lum_dc_table();
        let table = HuffmanDecodeTable::build(&bits, &vals).unwrap();
        assert!(table.max_len <= 16);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let (bits, vals) = lum_dc_table();
        let enc = HuffmanEncodeTable::build(&bits, &vals);
        let dec = HuffmanDecodeTable::build(&bits, &vals).unwrap();

        // Encode all symbols, then decode and verify
        for &sym in &vals {
            let (code, len) = enc.encode(sym).unwrap();

            // Create a bit stream from this code
            let mut byte_data = vec![0u8; 4];
            // Place code in the top bits of the first bytes
            let shifted = (code as u32) << (32 - len);
            byte_data[0] = (shifted >> 24) as u8;
            byte_data[1] = (shifted >> 16) as u8;
            byte_data[2] = (shifted >> 8) as u8;
            byte_data[3] = shifted as u8;

            // Handle byte-stuffing: if any byte is 0xFF, we need 0x00 after it
            let mut stuffed = Vec::new();
            for &b in &byte_data {
                stuffed.push(b);
                if b == 0xFF {
                    stuffed.push(0x00);
                }
            }

            let mut reader = BitReader::new(&stuffed, 0);
            let decoded = dec.decode(&mut reader).unwrap();
            assert_eq!(decoded, sym, "symbol {sym} round-trip failed");
        }
    }

    #[test]
    fn extend_sign_values() {
        // Category 1: value 0 → -1, value 1 → +1
        assert_eq!(extend_sign(0, 1), -1);
        assert_eq!(extend_sign(1, 1), 1);

        // Category 3: values 0–3 → -7 to -4, values 4–7 → +4 to +7
        assert_eq!(extend_sign(0, 3), -7);
        assert_eq!(extend_sign(3, 3), -4);
        assert_eq!(extend_sign(4, 3), 4);
        assert_eq!(extend_sign(7, 3), 7);

        // Category 0
        assert_eq!(extend_sign(0, 0), 0);
    }

    /// Standard JPEG luminance AC Huffman table (ITU-T T.81 Table
    /// K.5). Contains codes up to 16 bits — exercises the slow path.
    fn lum_ac_table() -> ([u8; 16], Vec<u8>) {
        let bits = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
        let vals = vec![
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13,
            0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42,
            0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a,
            0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35,
            0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a,
            0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67,
            0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84,
            0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
            0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3,
            0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
            0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1,
            0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4,
            0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
        ];
        (bits, vals)
    }

    /// T3.7 — the new 10-bit fast table + per-length slow path must
    /// decode the standard luminance AC table (codes 2-16 bits)
    /// byte-identically to the encode-then-decode reference.
    ///
    /// Exercises the slow path: AC table has many 11-16 bit codes,
    /// so the per-length canonical lookup is hit repeatedly.
    #[test]
    fn ac_table_encode_decode_roundtrip_exercises_slow_path() {
        let (bits, vals) = lum_ac_table();
        let enc = HuffmanEncodeTable::build(&bits, &vals);
        let dec = HuffmanDecodeTable::build(&bits, &vals).unwrap();

        assert!(dec.max_len > FAST_BITS, "AC table should have codes longer than fast-table width");

        for &sym in &vals {
            let (code, len) = enc.encode(sym).unwrap();

            // Place the code in a 32-bit buffer, top-aligned.
            let shifted = (code as u32) << (32 - len);
            let byte_data = [
                (shifted >> 24) as u8,
                (shifted >> 16) as u8,
                (shifted >> 8) as u8,
                shifted as u8,
            ];

            // Byte-stuff 0xFF bytes per JPEG entropy-coded segment rules.
            let mut stuffed = Vec::new();
            for &b in &byte_data {
                stuffed.push(b);
                if b == 0xFF {
                    stuffed.push(0x00);
                }
            }

            let mut reader = BitReader::new(&stuffed, 0);
            let decoded = dec.decode(&mut reader).unwrap();
            assert_eq!(
                decoded, sym,
                "AC symbol {sym:02x} (code={code:b}, len={len}) round-trip failed"
            );
        }
    }

    #[test]
    fn encode_value_roundtrip() {
        for v in -255i16..=255 {
            let (bits, size) = encode_value(v);
            if v == 0 {
                assert_eq!(size, 0);
            } else {
                let recovered = extend_sign(bits, size);
                assert_eq!(recovered, v, "round-trip failed for {v}");
            }
        }
    }
}
