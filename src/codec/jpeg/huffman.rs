// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Huffman coding tables for JPEG entropy decoding and encoding.

use super::bitio::BitReader;
use super::error::{JpegError, Result};

/// Huffman decode table with two-level lookup.
///
/// Level 1: 8-bit fast lookup table (covers most codes).
/// Level 2: slow path for codes longer than 8 bits.
pub struct HuffmanDecodeTable {
    /// Fast lookup: indexed by top 8 bits of the code stream.
    /// Each entry: (symbol, code_length). If code_length == 0, use slow path.
    fast: [(u8, u8); 256],
    /// For codes > 8 bits: (code, length, symbol) sorted by (length, code).
    slow: Vec<(u16, u8, u8)>,
    /// Maximum code length in this table.
    max_len: u8,
}

impl HuffmanDecodeTable {
    /// Build a decode table from JPEG-style counts and symbols.
    ///
    /// `bits`: counts[i] = number of codes of length i+1 (16 entries).
    /// `huffval`: the symbols, in order of increasing code length.
    pub fn build(bits: &[u8; 16], huffval: &[u8]) -> Result<Self> {
        let mut fast = [(0u8, 0u8); 256];
        let mut slow = Vec::new();
        let mut max_len = 0u8;

        // Generate canonical Huffman codes per ITU-T T.81 Annex C
        let mut code: u32 = 0;
        let mut si = 0; // symbol index into huffval

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
                    // Fill fast table: this code, left-shifted to 8 bits,
                    // covers 2^(8-length) entries
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

        Ok(Self {
            fast,
            slow,
            max_len,
        })
    }

    /// Decode one Huffman symbol from the bit stream.
    pub fn decode(&self, reader: &mut BitReader) -> Result<u8> {
        // Peek up to 8 bits for fast table lookup
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

        // Slow path: try longer codes
        self.decode_slow(reader)
    }

    fn decode_slow(&self, reader: &mut BitReader) -> Result<u8> {
        // Read up to max_len bits and try to match
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
