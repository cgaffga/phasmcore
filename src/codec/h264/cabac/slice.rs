// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! CABAC slice framing (Phase 6C.6b).
//!
//! Stitches together the pieces produced by the other cabac/ modules
//! into a complete slice RBSP payload:
//!
//!   slice_header (bit-aligned)
//!   cabac_alignment_one_bit * k   ← pad to byte boundary
//!   cabac_body                    ← bytes from CabacEngine::finish()
//!   rbsp_trailing_bits            ← stop bit + zero pad (always 0x80
//!                                    because cabac_body is byte-aligned)
//!   cabac_zero_word * n           ← optional stuffing, deferred to 6C.6d
//!
//! The spec pseudocode that drives this is:
//!   - § 7.3.4  slice_data() — opens with cabac_alignment_one_bit.
//!   - § 7.3.2.10 rbsp_slice_trailing_bits() — appends trailing + zero
//!     words.
//!   - § 9.3.4.5 termination — the last MB's `end_of_slice_flag` uses
//!     `encode_terminate(1)`; the engine's `finish()` then flushes
//!     the arithmetic state and pads to a byte boundary.
//!
//! The `cabac_zero_word` stuffing (§ 9.3.4.6) is **not** emitted here.
//! It's only required when the compressed slice ends up smaller than
//! the spec's minimum-byte-count threshold — rarely the case for the
//! kind of frames we encode. 6C.6d adds the check and stuffing.

use crate::codec::h264::encoder::bitstream_writer::BitWriter;

/// Emit `cabac_alignment_one_bit` bits until the writer is byte-aligned
/// (spec § 7.3.4). No-op if already aligned.
pub fn align_for_cabac(w: &mut BitWriter) {
    while !w.byte_aligned() {
        w.write_bit(true);
    }
}

/// Append `rbsp_trailing_bits()` (spec § 7.3.2.11) to an already
/// byte-aligned RBSP buffer. This is the CABAC-mode case: the CABAC
/// engine's `finish()` output is byte-aligned, so trailing bits are
/// just the single "1" stop bit + zero pad, i.e. one 0x80 byte.
pub fn append_rbsp_trailing_aligned(rbsp: &mut Vec<u8>) {
    rbsp.push(0x80);
}

/// Assemble a CABAC slice RBSP from its three components.
///
/// - `header_bytes`: slice header emitted via BitWriter. Must **not**
///   yet have trailing bits. Any bit-position at the end is fine —
///   this helper finalizes the bit state and aligns via
///   `cabac_alignment_one_bit`.
/// - `cabac_body`: the output of `CabacEngine::finish()` for this slice.
///   Assumed byte-aligned (engine guarantees this).
///
/// Does **not** emit `cabac_zero_word` stuffing — call
/// `append_cabac_zero_words` afterward if spec conformance requires it.
pub fn assemble_cabac_slice_rbsp(mut header_writer: BitWriter, cabac_body: &[u8]) -> Vec<u8> {
    align_for_cabac(&mut header_writer);
    let mut out = header_writer.finish();
    out.extend_from_slice(cabac_body);
    append_rbsp_trailing_aligned(&mut out);
    out
}

/// Append `cabac_zero_word` bytes (0x0000 each, spec § 9.3.4.6) to
/// satisfy the bin-density conformance constraint:
///
///   3 * NumBytesInNALunit >= bin_count - 96 * PicSizeInMbs
///
/// For typical content this is satisfied by default and no stuffing
/// is added. Very compressible sequences (large uniform regions, high
/// QP) can need padding. Each appended word (0x0000) expands to
/// 0x00 0x00 0x03 after emulation-prevention wrapping, adding 3 NAL
/// bytes per 2 RBSP bytes — we approximate NAL byte count as RBSP
/// byte count here, which is conservative enough for the constraint.
pub fn append_cabac_zero_words(rbsp: &mut Vec<u8>, bin_count: u32, pic_size_mbs: u32) {
    // Compute minimum required bytes.
    let threshold = bin_count.saturating_sub(96u32.saturating_mul(pic_size_mbs));
    while 3 * (rbsp.len() as u32) < threshold {
        rbsp.extend_from_slice(&[0x00, 0x00]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_for_cabac_no_op_when_already_aligned() {
        let mut w = BitWriter::new();
        w.write_bits(0xAB, 8); // exactly one byte
        align_for_cabac(&mut w);
        assert!(w.byte_aligned());
        assert_eq!(w.bits_written(), 8);
    }

    #[test]
    fn align_for_cabac_emits_ones_to_byte_boundary() {
        let mut w = BitWriter::new();
        // Write 5 bits — need 3 more to reach 8.
        w.write_bits(0b00000, 5);
        align_for_cabac(&mut w);
        assert!(w.byte_aligned());
        assert_eq!(w.bits_written(), 8);
        let bytes = w.finish();
        // First 5 bits are 0, last 3 bits are 1 → 0b00000_111 = 0x07.
        assert_eq!(bytes, vec![0x07]);
    }

    #[test]
    fn align_for_cabac_three_alignment_bits_all_ones() {
        let mut w = BitWriter::new();
        // 1 bit → need 7 alignment bits.
        w.write_bit(true);
        align_for_cabac(&mut w);
        assert!(w.byte_aligned());
        let bytes = w.finish();
        // 1_1111111 = 0xFF.
        assert_eq!(bytes, vec![0xFF]);
    }

    #[test]
    fn append_rbsp_trailing_aligned_adds_single_0x80_byte() {
        let mut rbsp = vec![0x42, 0x7F];
        append_rbsp_trailing_aligned(&mut rbsp);
        assert_eq!(rbsp, vec![0x42, 0x7F, 0x80]);
    }

    #[test]
    fn assemble_cabac_slice_rbsp_header_body_trailing() {
        // Simulate a 4-bit header (needs 4 alignment bits) + 2-byte
        // CABAC body + trailing.
        let mut hdr = BitWriter::new();
        hdr.write_bits(0b1010, 4);
        let body = [0x12, 0x34];

        let rbsp = assemble_cabac_slice_rbsp(hdr, &body);
        // First byte: 4 header bits + 4 alignment ones = 1010_1111 = 0xAF.
        assert_eq!(rbsp, vec![0xAF, 0x12, 0x34, 0x80]);
    }

    #[test]
    fn assemble_cabac_slice_rbsp_aligned_header() {
        // Header already byte-aligned → no alignment bits added.
        let mut hdr = BitWriter::new();
        hdr.write_bits(0xAB, 8);
        let body = [0xCD];

        let rbsp = assemble_cabac_slice_rbsp(hdr, &body);
        assert_eq!(rbsp, vec![0xAB, 0xCD, 0x80]);
    }

    #[test]
    fn append_cabac_zero_words_no_op_when_bins_fit() {
        let mut rbsp = vec![0u8; 100];
        // 10 bins, 1 MB → threshold = max(0, 10 - 96) = 0 → no padding.
        append_cabac_zero_words(&mut rbsp, 10, 1);
        assert_eq!(rbsp.len(), 100);
    }

    #[test]
    fn append_cabac_zero_words_pads_dense_bin_counts() {
        // Simulate a 10-byte slice with 10000 bins over 1 MB.
        // Threshold = 10000 - 96 = 9904. Needs 3*bytes >= 9904, i.e.,
        // bytes >= 3302. So we pad from 10 to at least 3302 bytes.
        let mut rbsp = vec![0u8; 10];
        append_cabac_zero_words(&mut rbsp, 10000, 1);
        assert!(rbsp.len() >= 3302, "expected padding, got len {}", rbsp.len());
        // All added bytes are 0x00 0x00.
        for i in 10..rbsp.len() {
            assert_eq!(rbsp[i], 0);
        }
        // Should be byte-even (words of 0x0000).
        assert_eq!((rbsp.len() - 10) % 2, 0);
    }

    #[test]
    fn full_cabac_slice_i_mb_then_end_of_slice() {
        use super::super::context::CabacInitSlot;
        use super::super::encoder::{encode_end_of_slice_flag, encode_mb_type_i, CabacEncoder};

        // End-to-end: build a trivial CABAC body with one I_NxN MB and
        // the end_of_slice_flag terminator, then wrap with an empty
        // header for assembly sanity.
        let mut enc = CabacEncoder::new_slice(CabacInitSlot::ISI, 26, 4);
        encode_mb_type_i(&mut enc, 0, 0); // I_NxN single bin
        encode_end_of_slice_flag(&mut enc, true);
        let body = enc.finish();
        assert!(!body.is_empty());

        let mut hdr = BitWriter::new();
        hdr.write_bits(0xAB, 8); // 1-byte stand-in for a byte-aligned header

        let rbsp = assemble_cabac_slice_rbsp(hdr, &body);
        assert_eq!(rbsp[0], 0xAB);
        assert_eq!(rbsp[rbsp.len() - 1], 0x80);
        assert_eq!(rbsp.len(), 2 + body.len());
    }
}
