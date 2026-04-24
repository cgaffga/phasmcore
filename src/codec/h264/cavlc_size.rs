// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase B size-counter helpers for CAVLC.
//!
//! The bit-counter infrastructure itself — the `BitSink` trait and
//! the `BitSizer` sink — lives in `encoder::bitstream_writer`. That
//! trait is already wired through `encode_cavlc_block` so the real
//! emit path and the size-only path walk the same codeword logic;
//! by construction, `BitSizer::bits_written() == 8 * BitWriter_bytes`
//! at every matching call site.
//!
//! This module adds the Phase B.2 top-level entry points: small
//! query-style helpers that the Phase C MB-level RDO pass calls to
//! estimate `R` (rate) for a candidate mode **without** triggering
//! any side-effect in the encoder (no neighbour-grid commits, no
//! skip-run mutation, no reconstruction writes).
//!
//! - `residual_block_bits_cavlc`: bits for one 4×4 CAVLC block given
//!   its scan-ordered coefficients + nC + block type.
//! - `partition_bits_cavlc`: bits for the MVD / ref_idx syntax of
//!   one P-slice partition.
//! - `macroblock_bits_cavlc`: bits for the MB-layer header portion
//!   (mb_type, sub_mb_type, ref_idx, MVDs, CBP, mb_qp_delta) —
//!   residuals are summed separately via `residual_block_bits_cavlc`.
//!
//! All helpers are pure functions of their inputs; they never
//! consult or modify encoder state. Phase B.3 unit tests compare
//! their output against the real `BitWriter`-emitted length.
//!
//! Design doc: `docs/design/h264-encoder-quality-plan.md` § Phase B.

use super::cavlc_writer::{encode_cavlc_block, CavlcBlockType};
use super::encoder::bitstream_writer::{BitSink, BitSizer};

/// Bits emitted by one CAVLC residual block with the given scan-
/// ordered coefficients + nC context + block type. Internally runs
/// the real `encode_cavlc_block` walker against a [`BitSizer`].
///
/// On malformed input (length mismatch vs `block_type.max_coeffs()`
/// or an uncodable escape-level) returns `None`; the real emit
/// path would error for the same reason.
pub fn residual_block_bits_cavlc(
    coeffs: &[i32],
    nc: i8,
    block_type: CavlcBlockType,
) -> Option<u32> {
    let mut sizer = BitSizer::new();
    encode_cavlc_block(&mut sizer, coeffs, nc, block_type).ok()?;
    Some(sizer.bits_written() as u32)
}

/// Bits for one se(v) MVD component (x or y). MVD values are signed
/// integers; se(v) codeword length is `2·⌊log2(codeNum+1)⌋ + 1`
/// where `codeNum = 2·|v|` for v<=0 and `codeNum = 2·v − 1` for v>0
/// (spec § 9.1.1).
pub fn mvd_component_bits(mvd: i32) -> u32 {
    let mut sizer = BitSizer::new();
    sizer.write_se(mvd);
    sizer.bits_written() as u32
}

/// Bits for one P-partition's MVD pair (x, y). No ref_idx (single-
/// reference Baseline/Main assumed; ref_idx is 0 ue(v) = 1 bit, added
/// elsewhere if multi-ref is enabled).
pub fn partition_mvd_bits(mvd_x: i32, mvd_y: i32) -> u32 {
    mvd_component_bits(mvd_x) + mvd_component_bits(mvd_y)
}

/// Bits for the MB-layer header of a P-slice MB given `mb_type`
/// codenum, optional `sub_mb_type` codenums (for P_8x8), the list
/// of MVD pairs in emit order, the `coded_block_pattern` codenum,
/// and the `mb_qp_delta` value.
///
/// `mb_skip_run` is NOT counted here — it's a run-length at the
/// slice level, counted per-slice by the caller. If this MB is
/// itself a P_SKIP, pass `mb_type_codenum = None` to return 0 (the
/// skip is accounted for by a future non-skip MB's `mb_skip_run`
/// ue(v)).
///
/// Returns 0 for P_SKIP, header bit count otherwise.
#[allow(clippy::too_many_arguments)]
pub fn macroblock_header_bits_cavlc(
    mb_type_codenum: u32,
    sub_mb_type_codenums: &[u32],
    mvds: &[(i32, i32)],
    coded_block_pattern_codenum: Option<u32>,
    mb_qp_delta: Option<i32>,
    transform_8x8_flag: Option<bool>,
) -> u32 {
    let mut sizer = BitSizer::new();
    sizer.write_ue(mb_type_codenum);
    for &cn in sub_mb_type_codenums {
        sizer.write_ue(cn);
    }
    for &(x, y) in mvds {
        sizer.write_se(x);
        sizer.write_se(y);
    }
    if let Some(cbp) = coded_block_pattern_codenum {
        sizer.write_ue(cbp);
    }
    if let Some(flag) = transform_8x8_flag {
        sizer.write_bit(flag);
    }
    if let Some(delta) = mb_qp_delta {
        sizer.write_se(delta);
    }
    sizer.bits_written() as u32
}

/// Bits for the mb_skip_run ue(v) that precedes the next non-skip
/// MB in a P-slice.
pub fn mb_skip_run_bits(skip_run: u32) -> u32 {
    let mut sizer = BitSizer::new();
    sizer.write_ue(skip_run);
    sizer.bits_written() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::encoder::bitstream_writer::BitWriter;
    use crate::codec::h264::encoder::bitstream_writer::BitSink as _;

    /// Helper: write some value via BitWriter, return the resulting
    /// bit-length (including partial trailing bits).
    fn writer_bits<F: FnOnce(&mut BitWriter)>(f: F) -> u32 {
        let mut w = BitWriter::new();
        f(&mut w);
        w.bits_written() as u32
    }

    // ── B.3 test fixtures — 5 representative CAVLC residual blocks ──

    /// Test 1: all-zero block.
    #[test]
    fn size_matches_writer_all_zero_luma4x4() {
        let coeffs = [0i32; 16];
        let real = writer_bits(|w| {
            encode_cavlc_block(w, &coeffs, 0, CavlcBlockType::Luma4x4).unwrap()
        });
        let sized = residual_block_bits_cavlc(&coeffs, 0, CavlcBlockType::Luma4x4).unwrap();
        assert_eq!(sized, real, "all-zero luma4x4: size={sized} real={real}");
    }

    /// Test 2: single high-freq trailing ±1 only (min tc=1, t1=1).
    #[test]
    fn size_matches_writer_single_t1_luma4x4() {
        let mut coeffs = [0i32; 16];
        coeffs[15] = 1; // highest-freq = +1
        let real = writer_bits(|w| {
            encode_cavlc_block(w, &coeffs, 0, CavlcBlockType::Luma4x4).unwrap()
        });
        let sized = residual_block_bits_cavlc(&coeffs, 0, CavlcBlockType::Luma4x4).unwrap();
        assert_eq!(sized, real, "single T1 luma4x4: size={sized} real={real}");
    }

    /// Test 3: dense mid-magnitude Intra16x16Ac block.
    #[test]
    fn size_matches_writer_dense_intra16_ac() {
        let coeffs: [i32; 15] = [2, -3, 1, 0, 1, -1, 0, 2, 0, 0, -1, 0, 0, 0, 0];
        let real = writer_bits(|w| {
            encode_cavlc_block(w, &coeffs, 4, CavlcBlockType::Intra16x16Ac).unwrap()
        });
        let sized =
            residual_block_bits_cavlc(&coeffs, 4, CavlcBlockType::Intra16x16Ac).unwrap();
        assert_eq!(sized, real, "dense intra16 AC: size={sized} real={real}");
    }

    /// Test 4: chroma DC block (4-coeff post-Hadamard, nc=-1).
    #[test]
    fn size_matches_writer_chroma_dc() {
        let coeffs: [i32; 4] = [5, -1, 0, 2];
        let real = writer_bits(|w| {
            encode_cavlc_block(w, &coeffs, -1, CavlcBlockType::ChromaDc).unwrap()
        });
        let sized = residual_block_bits_cavlc(&coeffs, -1, CavlcBlockType::ChromaDc).unwrap();
        assert_eq!(sized, real, "chroma DC: size={sized} real={real}");
    }

    /// Test 5: level-escape regime — a large-magnitude coefficient
    /// that forces the suffix_length > 0 escape path.
    #[test]
    fn size_matches_writer_escape_magnitude_luma4x4() {
        let mut coeffs = [0i32; 16];
        coeffs[0] = 63; // large DC, well into the suffix_length growth
        coeffs[1] = -17;
        coeffs[2] = 8;
        coeffs[3] = -4;
        coeffs[4] = 1; // trailing one
        let real = writer_bits(|w| {
            encode_cavlc_block(w, &coeffs, 8, CavlcBlockType::Luma4x4).unwrap()
        });
        let sized = residual_block_bits_cavlc(&coeffs, 8, CavlcBlockType::Luma4x4).unwrap();
        assert_eq!(sized, real, "escape-regime luma4x4: size={sized} real={real}");
    }

    // ── MVD / header bit-counter round-trips ──

    #[test]
    fn mvd_bits_match_writer() {
        for &v in &[0i32, 1, -1, 7, -8, 15, -32, 127, -128, 513, -2047] {
            let real = writer_bits(|w| w.write_se(v));
            assert_eq!(
                mvd_component_bits(v),
                real,
                "mvd={v}: size={} real={real}",
                mvd_component_bits(v)
            );
        }
    }

    #[test]
    fn partition_mvd_bits_match_writer() {
        let real = writer_bits(|w| {
            w.write_se(3);
            w.write_se(-5);
        });
        assert_eq!(partition_mvd_bits(3, -5), real);
    }

    #[test]
    fn header_bits_p16x16_typical() {
        // mb_type=0 (P_L0_16x16), 1 MVD (+2, -1), CBP=15 luma / 0 chroma
        // → codenum 15, mb_qp_delta=0.
        let real = writer_bits(|w| {
            w.write_ue(0);       // mb_type
            w.write_se(2);       // mvd_x
            w.write_se(-1);      // mvd_y
            w.write_ue(15);      // CBP codenum
            w.write_se(0);       // mb_qp_delta
        });
        let sized = macroblock_header_bits_cavlc(
            0,
            &[],
            &[(2, -1)],
            Some(15),
            Some(0),
            None,
        );
        assert_eq!(sized, real);
    }

    #[test]
    fn header_bits_p8x8_with_sub_mb_types() {
        // mb_type=3 (P_8x8), four sub_mb_types = (0, 1, 2, 0),
        // 4 MVDs (one each sub-MB as P_L0_8x8), CBP=3, qp_delta=-2.
        let real = writer_bits(|w| {
            w.write_ue(3);
            w.write_ue(0); w.write_ue(1); w.write_ue(2); w.write_ue(0);
            w.write_se(1); w.write_se(0);
            w.write_se(-2); w.write_se(1);
            w.write_se(0); w.write_se(-1);
            w.write_se(3); w.write_se(2);
            w.write_ue(3);
            w.write_se(-2);
        });
        let sized = macroblock_header_bits_cavlc(
            3,
            &[0, 1, 2, 0],
            &[(1, 0), (-2, 1), (0, -1), (3, 2)],
            Some(3),
            Some(-2),
            None,
        );
        assert_eq!(sized, real);
    }
}
