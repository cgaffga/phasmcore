// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Bit-level I/O for JPEG entropy-coded data.
//!
//! Provides [`BitReader`] for decoding and [`BitWriter`] for encoding the
//! entropy-coded scan data. Both handle JPEG byte-stuffing (0xFF -> 0xFF 0x00)
//! and operate in MSB-first bit order.

use super::error::{JpegError, Result};

/// Bit-level reader for JPEG entropy-coded data.
///
/// Handles JPEG byte-stuffing (0xFF00 → 0xFF) and marker detection.
/// Bits are read MSB-first from a 32-bit internal buffer.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Bit buffer, MSB-aligned. Valid bits are in the top `bits_left` positions.
    buf: u32,
    bits_left: u8,
    /// Set when a marker (0xFF followed by non-zero byte) is found in the stream.
    marker_found: Option<u8>,
}

impl<'a> BitReader<'a> {
    /// Create a new BitReader over the given byte slice.
    /// `pos` should point to the first byte of entropy-coded data (after SOS header).
    pub fn new(data: &'a [u8], pos: usize) -> Self {
        Self {
            data,
            pos,
            buf: 0,
            bits_left: 0,
            marker_found: None,
        }
    }

    /// Read `count` bits (1–16) and return them right-aligned.
    pub fn read_bits(&mut self, count: u8) -> Result<u16> {
        debug_assert!(count >= 1 && count <= 16);
        while self.bits_left < count {
            self.fill_byte()?;
        }
        self.bits_left -= count;
        let val = (self.buf >> self.bits_left) & ((1u32 << count) - 1);
        Ok(val as u16)
    }

    /// Peek at the top `count` bits without consuming them.
    pub fn peek_bits(&mut self, count: u8) -> Result<u16> {
        debug_assert!(count >= 1 && count <= 16);
        while self.bits_left < count {
            self.fill_byte()?;
        }
        let val = (self.buf >> (self.bits_left - count)) & ((1u32 << count) - 1);
        Ok(val as u16)
    }

    /// Discard `count` bits (must have been peeked already).
    pub fn skip_bits(&mut self, count: u8) {
        debug_assert!(count <= self.bits_left);
        self.bits_left -= count;
    }

    /// Align to the next byte boundary by discarding remaining bits in the current byte.
    pub fn byte_align(&mut self) {
        self.bits_left = 0;
        self.buf = 0;
    }

    /// Current byte position in the underlying data.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Returns the marker byte if a marker was encountered during reading.
    pub fn marker_found(&self) -> Option<u8> {
        self.marker_found
    }

    /// Check if a restart marker (0xFFD0–0xFFD7) is present.
    /// Checks both the `marker_found` flag (set if `fill_byte` already consumed
    /// a RST marker during Huffman decoding) and the next bytes in the stream.
    /// If found, consume the marker and return the marker's low nibble (0–7).
    pub fn check_restart_marker(&mut self) -> Result<Option<u8>> {
        self.byte_align();

        // Case 1: fill_byte already consumed a RST marker during Huffman decoding
        if let Some(m) = self.marker_found {
            if (m & 0xF8) == 0xD0 {
                self.marker_found = None;
                return Ok(Some(m & 0x07));
            }
        }

        // Case 2: RST marker is at the current position in the stream
        // Also skip any fill 0xFF bytes before the marker
        while self.pos + 1 < self.data.len() && self.data[self.pos] == 0xFF {
            let next = self.data[self.pos + 1];
            if next == 0xFF {
                // Fill byte — skip it
                self.pos += 1;
                continue;
            }
            if (next & 0xF8) == 0xD0 {
                let rst = next & 0x07;
                self.pos += 2;
                return Ok(Some(rst));
            }
            break;
        }

        Ok(None)
    }

    fn fill_byte(&mut self) -> Result<()> {
        if self.pos >= self.data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let byte = self.data[self.pos];
        self.pos += 1;

        if byte == 0xFF {
            if self.pos >= self.data.len() {
                return Err(JpegError::UnexpectedEof);
            }
            let next = self.data[self.pos];
            if next == 0x00 {
                // Byte-stuffed 0xFF
                self.pos += 1;
            } else {
                // This is a marker — signal it
                self.marker_found = Some(next);
                self.pos += 1;
                // Treat as zero-fill for remaining reads
                self.buf = (self.buf << 8) | 0xFF;
                self.bits_left += 8;
                return Ok(());
            }
        }

        self.buf = (self.buf << 8) | (byte as u32);
        self.bits_left += 8;
        Ok(())
    }
}

/// Bit-level writer for JPEG entropy-coded data.
///
/// Handles byte-stuffing (0xFF → 0xFF 0x00). MSB-first bit order.
pub struct BitWriter {
    output: Vec<u8>,
    buf: u8,
    bits_used: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            output: Vec::new(),
            buf: 0,
            bits_used: 0,
        }
    }

    /// Write `count` bits (1–16) from the low bits of `value`.
    pub fn write_bits(&mut self, value: u16, count: u8) {
        debug_assert!(count >= 1 && count <= 16);
        // Write bits MSB-first
        for i in (0..count).rev() {
            let bit = (value >> i) & 1;
            self.buf = (self.buf << 1) | (bit as u8);
            self.bits_used += 1;
            if self.bits_used == 8 {
                self.emit_byte(self.buf);
                self.buf = 0;
                self.bits_used = 0;
            }
        }
    }

    /// Pad remaining bits with 1s and flush.
    pub fn flush(mut self) -> Vec<u8> {
        if self.bits_used > 0 {
            // Pad with 1-bits as required by JPEG spec
            let remaining = 8 - self.bits_used;
            self.buf = (self.buf << remaining) | ((1u8 << remaining) - 1);
            self.emit_byte(self.buf);
        }
        self.output
    }

    fn emit_byte(&mut self, byte: u8) {
        self.output.push(byte);
        if byte == 0xFF {
            self.output.push(0x00); // Byte-stuffing
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_basic_bits() {
        // 0xA5 = 1010_0101
        let data = [0xA5];
        let mut r = BitReader::new(&data, 0);
        assert_eq!(r.read_bits(4).unwrap(), 0b1010);
        assert_eq!(r.read_bits(4).unwrap(), 0b0101);
    }

    #[test]
    fn read_cross_byte() {
        // 0xFF00 0x80 → after de-stuffing: 0xFF, 0x80
        let data = [0xFF, 0x00, 0x80];
        let mut r = BitReader::new(&data, 0);
        // Read 12 bits across byte boundary
        assert_eq!(r.read_bits(12).unwrap(), 0xFF8); // 1111_1111_1000
    }

    #[test]
    fn byte_stuffing_decode() {
        // 0xFF 0x00 should yield byte 0xFF
        let data = [0xFF, 0x00];
        let mut r = BitReader::new(&data, 0);
        assert_eq!(r.read_bits(8).unwrap(), 0xFF);
    }

    #[test]
    fn marker_detection() {
        // 0xFF 0xD9 is a marker (EOI), not byte-stuffed data
        let data = [0xAB, 0xFF, 0xD9];
        let mut r = BitReader::new(&data, 0);
        assert_eq!(r.read_bits(8).unwrap(), 0xAB);
        // Next read hits the marker — the 0xFF is read as data but marker is flagged
        let _ = r.read_bits(8);
        assert_eq!(r.marker_found(), Some(0xD9));
    }

    #[test]
    fn write_basic() {
        let mut w = BitWriter::new();
        w.write_bits(0b1010, 4);
        w.write_bits(0b0101, 4);
        let out = w.flush();
        assert_eq!(out, vec![0xA5]);
    }

    #[test]
    fn write_byte_stuffing() {
        let mut w = BitWriter::new();
        w.write_bits(0xFF, 8);
        let out = w.flush();
        assert_eq!(out, vec![0xFF, 0x00]);
    }

    #[test]
    fn write_padding() {
        let mut w = BitWriter::new();
        w.write_bits(0b110, 3);
        // Should pad with 1s: 110_11111 = 0xDF
        let out = w.flush();
        assert_eq!(out, vec![0xDF]);
    }

    #[test]
    fn write_cross_byte() {
        let mut w = BitWriter::new();
        w.write_bits(0b1111_1111_1000, 12);
        // First byte: 0xFF (needs stuffing), then 1000_1111 padded
        let out = w.flush();
        assert_eq!(out, vec![0xFF, 0x00, 0x8F]);
    }

    #[test]
    fn peek_then_skip() {
        let data = [0xA5]; // 1010_0101
        let mut r = BitReader::new(&data, 0);
        assert_eq!(r.peek_bits(4).unwrap(), 0b1010);
        r.skip_bits(4);
        assert_eq!(r.read_bits(4).unwrap(), 0b0101);
    }
}
