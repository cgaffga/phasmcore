// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Quantization and Huffman table parsing/serialization.
//!
//! Handles DQT (Define Quantization Table) and DHT (Define Huffman Table)
//! marker segments. Supports both 8-bit and 16-bit quantization precision
//! and multiple tables per marker segment.

use super::dct::QuantTable;
use super::error::{JpegError, Result};
use super::zigzag::ZIGZAG_TO_NATURAL;

/// Parse a DQT marker segment body (after the 2-byte length).
///
/// Returns a list of (table_id, QuantTable) pairs. A single DQT segment
/// can contain multiple tables.
pub fn parse_dqt(data: &[u8]) -> Result<Vec<(u8, QuantTable)>> {
    let mut tables = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        if pos >= data.len() {
            break;
        }
        let pq_tq = data[pos];
        pos += 1;
        let precision = pq_tq >> 4;
        let table_id = pq_tq & 0x0F;

        if table_id > 3 {
            return Err(JpegError::InvalidQuantTableId(table_id));
        }

        let mut values = [0u16; 64];
        if precision == 0 {
            // 8-bit values
            if pos + 64 > data.len() {
                return Err(JpegError::UnexpectedEof);
            }
            for zi in 0..64 {
                let ni = ZIGZAG_TO_NATURAL[zi];
                values[ni] = data[pos + zi] as u16;
            }
            pos += 64;
        } else if precision == 1 {
            // 16-bit values
            if pos + 128 > data.len() {
                return Err(JpegError::UnexpectedEof);
            }
            for zi in 0..64 {
                let ni = ZIGZAG_TO_NATURAL[zi];
                values[ni] = u16::from_be_bytes([data[pos + zi * 2], data[pos + zi * 2 + 1]]);
            }
            pos += 128;
        } else {
            return Err(JpegError::InvalidMarkerData("invalid DQT precision"));
        }

        tables.push((table_id, QuantTable::new(values)));
    }

    Ok(tables)
}

/// Write a DQT marker segment (including 0xFFDB marker and length).
pub fn write_dqt(table_id: u8, qt: &QuantTable) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0xFF);
    out.push(0xDB);

    // Check if all values fit in 8 bits
    let precision = if qt.values.iter().all(|&v| v <= 255) { 0u8 } else { 1u8 };
    let data_len = if precision == 0 { 64 } else { 128 };
    let length = 2 + 1 + data_len; // length field + pq_tq + values
    out.push((length >> 8) as u8);
    out.push(length as u8);
    out.push((precision << 4) | (table_id & 0x0F));

    for zi in 0..64 {
        let ni = ZIGZAG_TO_NATURAL[zi];
        if precision == 0 {
            out.push(qt.values[ni] as u8);
        } else {
            out.extend_from_slice(&qt.values[ni].to_be_bytes());
        }
    }

    out
}

/// Parsed Huffman table specification.
#[derive(Debug, Clone)]
pub struct HuffmanSpec {
    /// Table class: 0 = DC, 1 = AC.
    pub class: u8,
    /// Table ID (0–3).
    pub id: u8,
    /// Number of codes of each length (1–16).
    pub bits: [u8; 16],
    /// Symbol values in order of increasing code length.
    pub huffval: Vec<u8>,
}

/// Parse a DHT marker segment body (after the 2-byte length).
///
/// Returns a list of HuffmanSpec. A single DHT segment can contain multiple tables.
pub fn parse_dht(data: &[u8]) -> Result<Vec<HuffmanSpec>> {
    let mut specs = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        if pos >= data.len() {
            break;
        }
        let tc_th = data[pos];
        pos += 1;
        let class = tc_th >> 4;
        let id = tc_th & 0x0F;

        if class > 1 || id > 3 {
            return Err(JpegError::InvalidHuffmanTableId(tc_th));
        }

        if pos + 16 > data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let mut bits = [0u8; 16];
        bits.copy_from_slice(&data[pos..pos + 16]);
        pos += 16;

        let total: usize = bits.iter().map(|&b| b as usize).sum();
        if pos + total > data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let huffval = data[pos..pos + total].to_vec();
        pos += total;

        specs.push(HuffmanSpec {
            class,
            id,
            bits,
            huffval,
        });
    }

    Ok(specs)
}

/// Write a DHT marker segment (including 0xFFC4 marker and length).
pub fn write_dht(spec: &HuffmanSpec) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0xFF);
    out.push(0xC4);

    let total: usize = spec.bits.iter().map(|&b| b as usize).sum();
    let length = 2 + 1 + 16 + total;
    out.push((length >> 8) as u8);
    out.push(length as u8);
    out.push((spec.class << 4) | (spec.id & 0x0F));
    out.extend_from_slice(&spec.bits);
    out.extend_from_slice(&spec.huffval);

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_8bit_dqt() {
        // Build a DQT body: precision=0, id=0, 64 values 1..64 in zigzag order
        let mut body = vec![0x00u8]; // pq=0, tq=0
        for i in 0..64u8 {
            body.push(i + 1);
        }
        let tables = parse_dqt(&body).unwrap();
        assert_eq!(tables.len(), 1);
        let (id, qt) = &tables[0];
        assert_eq!(*id, 0);
        // Zigzag index 0 maps to natural index 0, value should be 1
        assert_eq!(qt.values[0], 1);
        // Zigzag index 1 maps to natural index 1, value should be 2
        assert_eq!(qt.values[1], 2);
        // Zigzag index 2 maps to natural index 8, value should be 3
        assert_eq!(qt.values[8], 3);
    }

    #[test]
    fn dqt_roundtrip() {
        let mut values = [0u16; 64];
        for i in 0..64 {
            values[i] = (i + 1) as u16;
        }
        let qt = QuantTable::new(values);
        let written = write_dqt(0, &qt);
        // Skip marker (2 bytes) and length (2 bytes)
        let body = &written[4..];
        let tables = parse_dqt(body).unwrap();
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].1.values, values);
    }

    #[test]
    fn parse_dht_basic() {
        // Build DHT body: class=0, id=0, standard DC luminance
        let mut body = vec![0x00u8]; // tc=0, th=0
        let bits = [0u8, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
        body.extend_from_slice(&bits);
        let vals: Vec<u8> = (0..12).collect();
        body.extend_from_slice(&vals);

        let specs = parse_dht(&body).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].class, 0);
        assert_eq!(specs[0].id, 0);
        assert_eq!(specs[0].bits, bits);
        assert_eq!(specs[0].huffval, vals);
    }

    #[test]
    fn dht_roundtrip() {
        let spec = HuffmanSpec {
            class: 1,
            id: 0,
            bits: [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125],
            huffval: (0..162).collect(),
        };
        let written = write_dht(&spec);
        let body = &written[4..];
        let specs = parse_dht(body).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].class, spec.class);
        assert_eq!(specs[0].id, spec.id);
        assert_eq!(specs[0].bits, spec.bits);
        assert_eq!(specs[0].huffval, spec.huffval);
    }
}
