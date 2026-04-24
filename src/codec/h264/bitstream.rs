// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264/AVC bitstream utilities: RBSP reader, NAL unit parsing, emulation
//! prevention byte handling with raw-position tracking.
//!
//! The key addition over the HEVC bitstream reader is [`EpByteMap`], which maps
//! RBSP byte indices back to raw (pre-EP-removal) byte indices. This enables the
//! CAVLC parser to record embeddable bit positions in the raw byte stream — the
//! coordinate space where bit flips are applied.

use super::{H264Error, NalType, NalUnit};

// ---------------------------------------------------------------------------
// Emulation Prevention Byte Handling
// ---------------------------------------------------------------------------

/// Mapping from RBSP byte positions to raw NAL byte positions.
///
/// Built during emulation prevention byte removal. For each RBSP byte at
/// index `i`, `rbsp_to_raw[i]` gives the corresponding byte index in the
/// original raw NAL data (after the NAL header byte).
#[derive(Debug, Clone)]
pub struct EpByteMap {
    pub rbsp_to_raw: Vec<usize>,
}

/// Remove emulation prevention bytes (0x03 after 0x00 0x00) from NAL payload.
///
/// Returns the RBSP data with EP bytes stripped.
pub fn remove_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut rbsp = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 3 {
            rbsp.push(0);
            rbsp.push(0);
            i += 3; // skip the 0x03
        } else {
            rbsp.push(data[i]);
            i += 1;
        }
    }
    rbsp
}

/// Remove emulation prevention bytes AND build a position map.
///
/// Returns `(rbsp_data, EpByteMap)` where `EpByteMap.rbsp_to_raw[i]` maps
/// RBSP byte `i` to its raw byte offset in `data`.
///
/// This is critical for steganography: the CAVLC parser reads RBSP data,
/// but bit flips must be applied to the raw NAL bytes (with EP bytes present).
pub fn remove_emulation_prevention_with_map(data: &[u8]) -> (Vec<u8>, EpByteMap) {
    let mut rbsp = Vec::with_capacity(data.len());
    let mut rbsp_to_raw = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 3 {
            rbsp.push(0);
            rbsp_to_raw.push(i);
            rbsp.push(0);
            rbsp_to_raw.push(i + 1);
            i += 3; // skip the 0x03
        } else {
            rbsp.push(data[i]);
            rbsp_to_raw.push(i);
            i += 1;
        }
    }
    (rbsp, EpByteMap { rbsp_to_raw })
}

/// Insert emulation prevention bytes into RBSP to produce valid NAL payload.
pub fn insert_emulation_prevention(rbsp: &[u8]) -> Vec<u8> {
    let mut data = Vec::with_capacity(rbsp.len() + rbsp.len() / 256 + 1);
    let mut zeros = 0u32;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            data.push(0x03);
            zeros = 0;
        }
        data.push(b);
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    data
}

// ---------------------------------------------------------------------------
// RBSP Bit Reader
// ---------------------------------------------------------------------------

/// Bit-level reader for RBSP (Raw Byte Sequence Payload) data.
///
/// Reads bits MSB-first from a byte slice with emulation prevention bytes
/// already removed.
pub struct RbspReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8, // 0-7: bits consumed in current byte
}

impl<'a> RbspReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Current RBSP byte position (for position tracking).
    #[inline]
    pub fn byte_pos(&self) -> usize {
        self.byte_pos
    }

    /// Current bit offset within the current byte (0 = MSB, 7 = LSB).
    #[inline]
    pub fn bit_pos(&self) -> u8 {
        self.bit_pos
    }

    /// Read a single bit (0 or 1).
    #[inline]
    pub fn read_bit(&mut self) -> Result<bool, H264Error> {
        if self.byte_pos >= self.data.len() {
            return Err(H264Error::UnexpectedEof);
        }
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(bit != 0)
    }

    /// Read `n` bits (1-32) as a u32, MSB-first.
    pub fn read_bits(&mut self, n: u8) -> Result<u32, H264Error> {
        debug_assert!(n > 0 && n <= 32);
        let mut val = 0u32;
        for _ in 0..n {
            val = (val << 1) | self.read_bit()? as u32;
        }
        Ok(val)
    }

    /// Read an unsigned Exp-Golomb coded value (ue(v)).
    pub fn read_ue(&mut self) -> Result<u32, H264Error> {
        let mut leading_zeros = 0u32;
        loop {
            if self.read_bit()? {
                break;
            }
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(H264Error::InvalidParameterSet(
                    "exp-golomb overflow".into(),
                ));
            }
        }
        if leading_zeros == 0 {
            return Ok(0);
        }
        let suffix = self.read_bits(leading_zeros as u8)?;
        Ok((1u32 << leading_zeros) - 1 + suffix)
    }

    /// Read a truncated Exp-Golomb coded value (te(v)).
    ///
    /// H.264 Section 9.1.2: if `max_value == 1`, te(v) is a single inverted bit
    /// (0 maps to 1, 1 maps to 0). For max_value > 1, te(v) == ue(v).
    pub fn read_te(&mut self, max_value: u32) -> Result<u32, H264Error> {
        if max_value == 1 {
            // 1-bit FLC, inverted: read 1 bit, return !bit
            let bit = self.read_bit()?;
            Ok(if bit { 0 } else { 1 })
        } else {
            self.read_ue()
        }
    }

    /// Read a signed Exp-Golomb coded value (se(v)).
    pub fn read_se(&mut self) -> Result<i32, H264Error> {
        let ue = self.read_ue()?;
        if ue == 0 {
            Ok(0)
        } else if ue & 1 != 0 {
            Ok(((ue + 1) / 2) as i32)
        } else {
            Ok(-((ue / 2) as i32))
        }
    }

    /// Skip `n` bits.
    pub fn skip_bits(&mut self, n: u32) -> Result<(), H264Error> {
        for _ in 0..n {
            self.read_bit()?;
        }
        Ok(())
    }

    /// True if the reader is at a byte boundary.
    pub fn byte_aligned(&self) -> bool {
        self.bit_pos == 0
    }

    /// Skip to the next byte boundary.
    pub fn align_to_byte(&mut self) {
        if self.bit_pos > 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Total bits consumed so far.
    pub fn bits_read(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Bits remaining in the data.
    pub fn bits_remaining(&self) -> usize {
        let total = self.data.len() * 8;
        total.saturating_sub(self.bits_read())
    }

    /// Check if there is more RBSP data (not just trailing alignment bits).
    pub fn more_rbsp_data(&self) -> bool {
        if self.byte_pos >= self.data.len() {
            return false;
        }
        // Search for the last 1-bit. If it's only the RBSP stop bit with
        // trailing zeros, there's no more data.
        let remaining = self.bits_remaining();
        if remaining == 0 {
            return false;
        }
        // If more than 8 bits remain, there's definitely more data
        if remaining > 8 {
            return true;
        }
        // Check if remaining bits contain more than just a stop bit
        let mut pos = self.byte_pos;
        let mut bit = self.bit_pos;
        let mut last_one_pos = None;
        let mut bit_count = 0;
        while pos < self.data.len() {
            while bit < 8 && pos < self.data.len() {
                if (self.data[pos] >> (7 - bit)) & 1 != 0 {
                    last_one_pos = Some(bit_count);
                }
                bit_count += 1;
                bit += 1;
            }
            bit = 0;
            pos += 1;
        }
        // If the last 1-bit is at a position > 0, there's data before the stop bit
        matches!(last_one_pos, Some(p) if p > 0)
    }
}

// ---------------------------------------------------------------------------
// NAL Unit Parsing
// ---------------------------------------------------------------------------

/// Parse a single H.264 NAL unit from its raw bytes (1-byte header + payload).
///
/// H.264 NAL header (1 byte):
/// ```text
/// +---------------+
/// |0|NRI|  Type   |
/// +-+-+-+-+-+-+-+-+
///  F  2b   5b
/// ```
/// - F: forbidden_zero_bit (must be 0)
/// - NRI: nal_ref_idc (0-3)
/// - Type: nal_unit_type (0-31)
pub fn parse_nal_unit(data: &[u8]) -> Result<NalUnit, H264Error> {
    if data.is_empty() {
        return Err(H264Error::InvalidNalHeader);
    }
    let forbidden_zero = (data[0] >> 7) & 1;
    if forbidden_zero != 0 {
        return Err(H264Error::InvalidNalHeader);
    }
    let nal_ref_idc = (data[0] >> 5) & 0x03;
    let nal_type = NalType(data[0] & 0x1F);

    let rbsp = remove_emulation_prevention(&data[1..]);
    Ok(NalUnit {
        nal_type,
        nal_ref_idc,
        rbsp,
    })
}

/// Parse a single H.264 NAL unit, returning both the NalUnit and an EP byte map
/// for raw position tracking. The `raw_payload` is `&data[1..]` (after NAL header).
pub fn parse_nal_unit_with_map(data: &[u8]) -> Result<(NalUnit, EpByteMap), H264Error> {
    if data.is_empty() {
        return Err(H264Error::InvalidNalHeader);
    }
    let forbidden_zero = (data[0] >> 7) & 1;
    if forbidden_zero != 0 {
        return Err(H264Error::InvalidNalHeader);
    }
    let nal_ref_idc = (data[0] >> 5) & 0x03;
    let nal_type = NalType(data[0] & 0x1F);

    let (rbsp, ep_map) = remove_emulation_prevention_with_map(&data[1..]);
    Ok((
        NalUnit {
            nal_type,
            nal_ref_idc,
            rbsp,
        },
        ep_map,
    ))
}

/// Parse NAL units from length-prefixed data (MP4/ISOBMFF format).
///
/// Each NAL unit is preceded by a `length_size`-byte big-endian length field.
pub fn parse_nal_units_mp4(data: &[u8], length_size: u8) -> Result<Vec<NalUnit>, H264Error> {
    let ls = length_size as usize;
    let mut nalus = Vec::new();
    let mut pos = 0;
    while pos + ls <= data.len() {
        let mut len = 0usize;
        for i in 0..ls {
            len = (len << 8) | data[pos + i] as usize;
        }
        pos += ls;
        if pos + len > data.len() {
            return Err(H264Error::UnexpectedEof);
        }
        if len > 0 {
            nalus.push(parse_nal_unit(&data[pos..pos + len])?);
        }
        pos += len;
    }
    Ok(nalus)
}

/// Parse NAL units from Annex B format (start-code delimited).
pub fn parse_nal_units_annexb(data: &[u8]) -> Result<Vec<NalUnit>, H264Error> {
    let mut nalus = Vec::new();
    let mut i = 0;

    // Skip leading zeros/start code
    while i < data.len() {
        if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            i += 3;
            break;
        }
        if i + 3 < data.len()
            && data[i] == 0
            && data[i + 1] == 0
            && data[i + 2] == 0
            && data[i + 3] == 1
        {
            i += 4;
            break;
        }
        i += 1;
    }

    let mut nal_start = i;
    while i < data.len() {
        if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1
                || (i + 3 < data.len() && data[i + 2] == 0 && data[i + 3] == 1)
            {
                let mut end = i;
                while end > nal_start && data[end - 1] == 0 {
                    end -= 1;
                }
                if end > nal_start {
                    nalus.push(parse_nal_unit(&data[nal_start..end])?);
                }
                if data[i + 2] == 1 {
                    i += 3;
                } else {
                    i += 4;
                }
                nal_start = i;
                continue;
            }
        }
        i += 1;
    }

    if nal_start < data.len() {
        nalus.push(parse_nal_unit(&data[nal_start..])?);
    }

    Ok(nalus)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- RbspReader tests --

    #[test]
    fn rbsp_reader_basic() {
        let data = [0b1010_0110, 0b1100_0000];
        let mut r = RbspReader::new(&data);
        assert!(r.read_bit().unwrap()); // 1
        assert!(!r.read_bit().unwrap()); // 0
        assert!(r.read_bit().unwrap()); // 1
        assert!(!r.read_bit().unwrap()); // 0
        assert_eq!(r.bits_read(), 4);
        assert_eq!(r.byte_pos(), 0);
        assert_eq!(r.bit_pos(), 4);
        assert_eq!(r.read_bits(4).unwrap(), 0b0110);
        assert_eq!(r.bits_read(), 8);
        assert!(r.byte_aligned());
    }

    #[test]
    fn rbsp_reader_ue() {
        // ue(0) = binary 1
        let data = [0b1000_0000];
        let mut r = RbspReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 0);

        // ue(1) = binary 010
        let data = [0b0100_0000];
        let mut r = RbspReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 1);

        // ue(2) = binary 011
        let data = [0b0110_0000];
        let mut r = RbspReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 2);

        // ue(3) = binary 00100
        let data = [0b0010_0000];
        let mut r = RbspReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 3);

        // ue(6) = binary 00111
        let data = [0b0011_1000];
        let mut r = RbspReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 6);
    }

    #[test]
    fn rbsp_reader_se() {
        // se(0) = ue(0) = 1
        let data = [0b1000_0000];
        let mut r = RbspReader::new(&data);
        assert_eq!(r.read_se().unwrap(), 0);

        // se(+1) = ue(1) = 010
        let data = [0b0100_0000];
        let mut r = RbspReader::new(&data);
        assert_eq!(r.read_se().unwrap(), 1);

        // se(-1) = ue(2) = 011
        let data = [0b0110_0000];
        let mut r = RbspReader::new(&data);
        assert_eq!(r.read_se().unwrap(), -1);

        // se(+2) = ue(3) = 00100
        let data = [0b0010_0000];
        let mut r = RbspReader::new(&data);
        assert_eq!(r.read_se().unwrap(), 2);

        // se(-2) = ue(4) = 00101
        let data = [0b0010_1000];
        let mut r = RbspReader::new(&data);
        assert_eq!(r.read_se().unwrap(), -2);
    }

    // -- Emulation Prevention tests --

    #[test]
    fn ep_removal_basic() {
        // 00 00 03 01 → 00 00 01
        let raw = [0x00, 0x00, 0x03, 0x01];
        let rbsp = remove_emulation_prevention(&raw);
        assert_eq!(rbsp, [0x00, 0x00, 0x01]);
    }

    #[test]
    fn ep_removal_no_ep_bytes() {
        let raw = [0x01, 0x02, 0x03, 0x04];
        let rbsp = remove_emulation_prevention(&raw);
        assert_eq!(rbsp, raw);
    }

    #[test]
    fn ep_roundtrip() {
        let original = [0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00, 0x03, 0xFF];
        let with_ep = insert_emulation_prevention(&original);
        let recovered = remove_emulation_prevention(&with_ep);
        assert_eq!(recovered, original);
    }

    #[test]
    fn ep_map_tracks_positions() {
        // raw:  [0x00, 0x00, 0x03, 0x01, 0xFF]
        // rbsp: [0x00, 0x00,       0x01, 0xFF]
        // map:  [0,    1,          3,    4   ]
        let raw = [0x00, 0x00, 0x03, 0x01, 0xFF];
        let (rbsp, map) = remove_emulation_prevention_with_map(&raw);
        assert_eq!(rbsp, [0x00, 0x00, 0x01, 0xFF]);
        assert_eq!(map.rbsp_to_raw, [0, 1, 3, 4]);
    }

    #[test]
    fn ep_map_multiple_ep_bytes() {
        // raw:  [0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x01]
        // rbsp: [0x00, 0x00,       0x00, 0x00,       0x01]
        // map:  [0,    1,          3,    4,           6   ]
        let raw = [0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x01];
        let (rbsp, map) = remove_emulation_prevention_with_map(&raw);
        assert_eq!(rbsp, [0x00, 0x00, 0x00, 0x00, 0x01]);
        assert_eq!(map.rbsp_to_raw, [0, 1, 3, 4, 6]);
    }

    // -- NAL Unit Parsing tests --

    #[test]
    fn parse_h264_sps_nal() {
        // H.264 NAL header: 0x67 = forbidden=0, nal_ref_idc=3, type=7 (SPS)
        let data = [0x67, 0x42, 0x00, 0x1E]; // SPS NAL + some payload
        let nalu = parse_nal_unit(&data).unwrap();
        assert_eq!(nalu.nal_type, NalType::SPS);
        assert_eq!(nalu.nal_ref_idc, 3);
        assert_eq!(nalu.rbsp.len(), 3); // payload after 1-byte header
    }

    #[test]
    fn parse_h264_pps_nal() {
        // 0x68 = forbidden=0, nal_ref_idc=3, type=8 (PPS)
        let data = [0x68, 0xCE, 0x38, 0x80];
        let nalu = parse_nal_unit(&data).unwrap();
        assert_eq!(nalu.nal_type, NalType::PPS);
        assert_eq!(nalu.nal_ref_idc, 3);
    }

    #[test]
    fn parse_h264_idr_nal() {
        // 0x65 = forbidden=0, nal_ref_idc=3, type=5 (IDR)
        let data = [0x65, 0x88, 0x80, 0x40];
        let nalu = parse_nal_unit(&data).unwrap();
        assert_eq!(nalu.nal_type, NalType::SLICE_IDR);
        assert!(nalu.nal_type.is_idr());
        assert!(nalu.nal_type.is_vcl());
    }

    #[test]
    fn parse_h264_non_idr_nal() {
        // 0x41 = forbidden=0, nal_ref_idc=2, type=1 (non-IDR slice)
        let data = [0x41, 0x9A, 0x00];
        let nalu = parse_nal_unit(&data).unwrap();
        assert_eq!(nalu.nal_type, NalType::SLICE);
        assert!(!nalu.nal_type.is_idr());
        assert!(nalu.nal_type.is_vcl());
    }

    #[test]
    fn parse_forbidden_bit_set() {
        let data = [0x80 | 0x67]; // forbidden bit = 1
        assert!(parse_nal_unit(&data).is_err());
    }

    #[test]
    fn parse_nal_with_map() {
        let data = [0x65, 0x00, 0x00, 0x03, 0x01, 0xFF];
        let (nalu, map) = parse_nal_unit_with_map(&data).unwrap();
        assert_eq!(nalu.nal_type, NalType::SLICE_IDR);
        // payload is data[1..] = [0x00, 0x00, 0x03, 0x01, 0xFF]
        // rbsp after EP removal: [0x00, 0x00, 0x01, 0xFF]
        assert_eq!(nalu.rbsp, [0x00, 0x00, 0x01, 0xFF]);
        // map: rbsp[0]→raw[0], rbsp[1]→raw[1], rbsp[2]→raw[3], rbsp[3]→raw[4]
        assert_eq!(map.rbsp_to_raw, [0, 1, 3, 4]);
    }

    #[test]
    fn parse_nal_units_mp4_format() {
        // Two NAL units with 4-byte length prefix
        let mut data = Vec::new();
        // NAL 1: SPS (3 bytes)
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x03]); // length=3
        data.extend_from_slice(&[0x67, 0x42, 0x00]); // SPS NAL
        // NAL 2: PPS (2 bytes)
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x02]); // length=2
        data.extend_from_slice(&[0x68, 0xCE]); // PPS NAL

        let nalus = parse_nal_units_mp4(&data, 4).unwrap();
        assert_eq!(nalus.len(), 2);
        assert_eq!(nalus[0].nal_type, NalType::SPS);
        assert_eq!(nalus[1].nal_type, NalType::PPS);
    }

    #[test]
    fn parse_nal_units_annexb_format() {
        let mut data = Vec::new();
        // Start code + SPS
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]); // 4-byte start code
        data.extend_from_slice(&[0x67, 0x42, 0x00, 0x1E]);
        // Start code + PPS
        data.extend_from_slice(&[0x00, 0x00, 0x01]); // 3-byte start code
        data.extend_from_slice(&[0x68, 0xCE, 0x38]);

        let nalus = parse_nal_units_annexb(&data).unwrap();
        assert_eq!(nalus.len(), 2);
        assert_eq!(nalus[0].nal_type, NalType::SPS);
        assert_eq!(nalus[1].nal_type, NalType::PPS);
    }
}
