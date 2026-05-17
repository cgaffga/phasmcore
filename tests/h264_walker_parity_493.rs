// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #493.1 Phase 0 — walker-symmetry parity gates × 8 (engine × domain).
//!
//! For each of the 8 (engine, domain) combinations, verify bit-exact
//! equality between encoder Pass-1 cover capture and the CABAC
//! walker's re-parse of the same emitted Annex-B. ALL positions per
//! domain are compared, not selective — cost-vector-masked walker
//! bugs (positions STC never picks at current cost shape) surface here.
//!
//! Hard prereq for #493.2 (CostWeights framework). If any domain
//! fails, the 4-domain plan's combined-cover-vector decoder symmetry
//! can't be assumed; design needs review.
//!
//! Domains:
//!   - CoeffSign       (CS)
//!   - CoeffSuffixLsb  (CSL)
//!   - MvdSignBypass   (MVDs)
//!   - MvdSuffixLsb    (MVDsl)
//!
//! Engines:
//!   - Pure-Rust H.264 encoder (`pass1_capture_and_emit_clean`)
//!   - OH264 fork (clean encode + `extract_cover_bits_via_decoder`
//!     cross-check vs Rust walker)

#![cfg(feature = "h264-encoder")]

use phasm_core::codec::h264::cabac::bin_decoder::slice::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};

/// Walker options for 4-domain parity gates — `record_mvd: true` is
/// the critical flag (default is false, which leaves MVD cover empty).
/// Phase 0 finding: the 4-domain streaming-session decoder MUST use
/// these options or MvdSign/MvdSuffix domains will silently extract
/// zero cover bits.
fn full_walk_options() -> WalkOptions {
    WalkOptions {
        record_mvd: true,
        ..Default::default()
    }
}
use phasm_core::codec::h264::stego::encode_pixels::pass1_capture_and_emit_clean;
use phasm_core::codec::h264::stego::{DomainCover, PositionKey};

/// Textured YUV synth matching the streaming-session tests. Produces
/// realistic cover-bit density at QP=26 across all 4 domains; smooth
/// gradients produce too few MVD positions to exercise.
fn synth_yuv(width: u32, height: u32, frame_idx: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let half_w = w / 2;
    let half_h = h / 2;
    let mut out = Vec::with_capacity(w * h * 3 / 2);
    // Y plane.
    for j in 0..h {
        for i in 0..w {
            let val = ((i + frame_idx as usize * 2) ^ (j + frame_idx as usize * 3)) as u8;
            out.push(val);
        }
    }
    // U + V planes.
    let mut s: u32 = 0xCAFE_F00D ^ frame_idx;
    for _plane in 0..2 {
        for j in 0..half_h {
            for i in 0..half_w {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let tex = (s >> 16) as u8;
                let pos = (i + j + frame_idx as usize) as u8;
                out.push(tex.wrapping_add(pos));
            }
        }
    }
    out
}

/// Build a multi-frame textured YUV.
fn build_yuv_clip(width: u32, height: u32, n_frames: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for f in 0..n_frames {
        out.extend_from_slice(&synth_yuv(width, height, f));
    }
    out
}

// ───────────────────────── Pure-Rust × 4 domains ─────────────────────

mod pure_rust {
    use super::*;

    /// Encode a fixture clean, capture Pass-1 cover, walk emitted
    /// Annex-B, return both for per-domain comparison.
    fn capture(width: u32, height: u32, n_frames: u32) -> (
        DomainCover,
        DomainCover,
    ) {
        let yuv = build_yuv_clip(width, height, n_frames);
        let (pass1, annex_b) = pass1_capture_and_emit_clean(
            &yuv, width, height, n_frames as usize, /*quality*/ None,
        ).expect("pass1_capture_and_emit_clean");
        let walked = walk_annex_b_for_cover_with_options(&annex_b, full_walk_options())
            .expect("walk_annex_b_for_cover_with_options on clean output");
        (pass1.cover, walked.cover)
    }

    /// Helper: compare bits + positions vectors for one domain.
    /// Reports first diverging index + total length on mismatch.
    fn assert_domain_parity(
        domain: &str,
        pass1_bits: &[u8],
        pass1_positions: &[PositionKey],
        walker_bits: &[u8],
        walker_positions: &[PositionKey],
    ) {
        assert_eq!(
            pass1_bits.len(),
            walker_bits.len(),
            "{domain}: bit count mismatch — pass1 has {}, walker has {}",
            pass1_bits.len(), walker_bits.len(),
        );
        assert_eq!(
            pass1_positions.len(),
            walker_positions.len(),
            "{domain}: position count mismatch — pass1 has {}, walker has {}",
            pass1_positions.len(), walker_positions.len(),
        );
        // Bit-by-bit comparison reporting first divergence.
        for (i, (a, b)) in pass1_bits.iter().zip(walker_bits.iter()).enumerate() {
            assert_eq!(
                a, b,
                "{domain} bit divergence at index {i}/{}: pass1={a} walker={b}",
                pass1_bits.len(),
            );
        }
        for (i, (a, b)) in pass1_positions.iter().zip(walker_positions.iter()).enumerate() {
            assert_eq!(
                a, b,
                "{domain} position divergence at index {i}/{}: pass1={a:?} walker={b:?}",
                pass1_positions.len(),
            );
        }
    }

    #[test]
    fn parity_coeff_sign() {
        // Tiny 4-frame 128×80 textured clip — plenty of CoeffSign
        // cover, fast to encode.
        let (pass1, walker) = capture(128, 80, 4);
        assert_domain_parity(
            "CoeffSign",
            &pass1.coeff_sign_bypass.bits,
            &pass1.coeff_sign_bypass.positions,
            &walker.coeff_sign_bypass.bits,
            &walker.coeff_sign_bypass.positions,
        );
        // Sanity: non-empty cover proves the test exercised something.
        assert!(
            !pass1.coeff_sign_bypass.bits.is_empty(),
            "CoeffSign cover empty — fixture too small to exercise the gate"
        );
    }

    #[test]
    fn parity_coeff_suffix_lsb() {
        let (pass1, walker) = capture(128, 80, 4);
        assert_domain_parity(
            "CoeffSuffixLsb",
            &pass1.coeff_suffix_lsb.bits,
            &pass1.coeff_suffix_lsb.positions,
            &walker.coeff_suffix_lsb.bits,
            &walker.coeff_suffix_lsb.positions,
        );
        // CSL may be sparse on smooth content; report length but don't
        // hard-fail on empty — that's content-dependent, not a parity
        // bug. If CSL empty, the parity equality above is vacuous;
        // try a larger fixture if you need a non-vacuous gate.
        eprintln!(
            "CoeffSuffixLsb cover size: {} bits, {} positions",
            pass1.coeff_suffix_lsb.bits.len(),
            pass1.coeff_suffix_lsb.positions.len(),
        );
    }

    #[test]
    fn parity_mvd_sign_bypass() {
        // MVD domains only fire on P/B frames with non-zero motion.
        // 4-frame clip with textured content (LCG-noise chroma)
        // produces some motion between frames.
        let (pass1, walker) = capture(128, 80, 4);
        assert_domain_parity(
            "MvdSignBypass",
            &pass1.mvd_sign_bypass.bits,
            &pass1.mvd_sign_bypass.positions,
            &walker.mvd_sign_bypass.bits,
            &walker.mvd_sign_bypass.positions,
        );
        eprintln!(
            "MvdSignBypass cover size: {} bits, {} positions",
            pass1.mvd_sign_bypass.bits.len(),
            pass1.mvd_sign_bypass.positions.len(),
        );
    }

    #[test]
    fn parity_mvd_suffix_lsb() {
        let (pass1, walker) = capture(128, 80, 4);
        assert_domain_parity(
            "MvdSuffixLsb",
            &pass1.mvd_suffix_lsb.bits,
            &pass1.mvd_suffix_lsb.positions,
            &walker.mvd_suffix_lsb.bits,
            &walker.mvd_suffix_lsb.positions,
        );
        eprintln!(
            "MvdSuffixLsb cover size: {} bits, {} positions",
            pass1.mvd_suffix_lsb.bits.len(),
            pass1.mvd_suffix_lsb.positions.len(),
        );
    }

    /// Larger fixture: 320×240 × 6f. Exercises more MBs, denser
    /// cover, broader walker coverage. #[ignore]'d by default
    /// to keep CI fast.
    #[test]
    #[ignore = "slow: 320×240 × 6f parity sweep"]
    fn parity_all_domains_larger_fixture() {
        let (pass1, walker) = capture(320, 240, 6);
        assert_domain_parity(
            "CoeffSign(320×240×6)",
            &pass1.coeff_sign_bypass.bits,
            &pass1.coeff_sign_bypass.positions,
            &walker.coeff_sign_bypass.bits,
            &walker.coeff_sign_bypass.positions,
        );
        assert_domain_parity(
            "CoeffSuffixLsb(320×240×6)",
            &pass1.coeff_suffix_lsb.bits,
            &pass1.coeff_suffix_lsb.positions,
            &walker.coeff_suffix_lsb.bits,
            &walker.coeff_suffix_lsb.positions,
        );
        assert_domain_parity(
            "MvdSignBypass(320×240×6)",
            &pass1.mvd_sign_bypass.bits,
            &pass1.mvd_sign_bypass.positions,
            &walker.mvd_sign_bypass.bits,
            &walker.mvd_sign_bypass.positions,
        );
        assert_domain_parity(
            "MvdSuffixLsb(320×240×6)",
            &pass1.mvd_suffix_lsb.bits,
            &pass1.mvd_suffix_lsb.positions,
            &walker.mvd_suffix_lsb.bits,
            &walker.mvd_suffix_lsb.positions,
        );
    }
}

// ───────────────────────── OH264 × 4 domains ─────────────────────────
//
// OH264 path: encode clean via the fork, then cross-check the Rust
// walker's output against `extract_cover_bits_via_decoder` (OH264
// decoder hooks). Both should yield bit-identical per-domain
// sequences for the streaming-session 4-domain decoder to work.

#[cfg(feature = "openh264-backend")]
mod oh264 {
    use super::*;
    use phasm_core::codec::h264::openh264::{
        extract_cover_bits_via_decoder, Encoder,
    };

    /// Encode a fixture with OH264, return Annex-B + OH264 decoder-
    /// hook extraction + Rust walker extraction.
    fn encode_and_extract(width: u32, height: u32, n_frames: u32) -> (
        Vec<u8>,
        phasm_core::codec::h264::openh264::OpenH264CoverBits,
        DomainCover,
    ) {
        let yuv = build_yuv_clip(width, height, n_frames);
        let mut enc = Encoder::new(
            width as i32, height as i32, /*qp*/ 26, /*intra_period*/ n_frames as i32,
        ).expect("OH264 Encoder::new");
        let frame_y = (width * height) as usize;
        let frame_uv = (width * height / 4) as usize;
        let frame_total = frame_y + 2 * frame_uv;
        let mut out = vec![0u8; 4 * 1024 * 1024];
        let mut annex_b = Vec::with_capacity(2 * 1024 * 1024);
        for f in 0..n_frames as usize {
            let base = f * frame_total;
            let (_ftype, n) = enc.encode_frame(
                &yuv[base..base + frame_y],
                &yuv[base + frame_y..base + frame_y + frame_uv],
                &yuv[base + frame_y + frame_uv..base + frame_total],
                (f as i64) * 33,
                &mut out,
            ).expect("OH264 encode_frame");
            annex_b.extend_from_slice(&out[..n]);
        }
        drop(enc);
        let oh264_bits = extract_cover_bits_via_decoder(&annex_b)
            .expect("extract_cover_bits_via_decoder");
        let walked = walk_annex_b_for_cover_with_options(&annex_b, full_walk_options())
            .expect("walk_annex_b_for_cover_with_options on OH264 output");
        (annex_b, oh264_bits, walked.cover)
    }

    /// Per-domain parity for OH264: compare OH264 decoder-hook bit
    /// sequence to Rust walker bit sequence. Positions are encoder-
    /// specific (OH264 keys vs walker raster), so we only compare bit
    /// vectors — positions are validated separately at the override-
    /// map translation layer (Phase B.8 already verified).
    fn assert_domain_bit_parity(
        domain: &str,
        oh264_bits: &[u8],
        walker_bits: &[u8],
    ) {
        assert_eq!(
            oh264_bits.len(),
            walker_bits.len(),
            "{domain}: bit count mismatch — OH264={}, walker={}",
            oh264_bits.len(), walker_bits.len(),
        );
        for (i, (a, b)) in oh264_bits.iter().zip(walker_bits.iter()).enumerate() {
            assert_eq!(
                a, b,
                "{domain} bit divergence at index {i}/{}: OH264={a} walker={b}",
                oh264_bits.len(),
            );
        }
    }

    #[test]
    fn parity_coeff_sign() {
        let (_annex_b, oh264, walker) = encode_and_extract(128, 80, 4);
        assert_domain_bit_parity(
            "OH264 CoeffSign",
            &oh264.coeff_sign_bypass,
            &walker.coeff_sign_bypass.bits,
        );
        assert!(
            !oh264.coeff_sign_bypass.is_empty(),
            "OH264 CoeffSign cover empty"
        );
    }

    #[test]
    fn parity_coeff_suffix_lsb() {
        let (_annex_b, oh264, walker) = encode_and_extract(128, 80, 4);
        assert_domain_bit_parity(
            "OH264 CoeffSuffixLsb",
            &oh264.coeff_suffix_lsb,
            &walker.coeff_suffix_lsb.bits,
        );
        eprintln!(
            "OH264 CoeffSuffixLsb cover size: {} bits",
            oh264.coeff_suffix_lsb.len(),
        );
    }

    #[test]
    fn parity_mvd_sign_bypass() {
        let (_annex_b, oh264, walker) = encode_and_extract(128, 80, 4);
        assert_domain_bit_parity(
            "OH264 MvdSignBypass",
            &oh264.mvd_sign_bypass,
            &walker.mvd_sign_bypass.bits,
        );
        eprintln!(
            "OH264 MvdSignBypass cover size: {} bits",
            oh264.mvd_sign_bypass.len(),
        );
    }

    #[test]
    fn parity_mvd_suffix_lsb() {
        let (_annex_b, oh264, walker) = encode_and_extract(128, 80, 4);
        assert_domain_bit_parity(
            "OH264 MvdSuffixLsb",
            &oh264.mvd_suffix_lsb,
            &walker.mvd_suffix_lsb.bits,
        );
        eprintln!(
            "OH264 MvdSuffixLsb cover size: {} bits",
            oh264.mvd_suffix_lsb.len(),
        );
    }

    #[test]
    #[ignore = "slow: 320×240 × 6f OH264 parity sweep"]
    fn parity_all_domains_larger_fixture() {
        let (_annex_b, oh264, walker) = encode_and_extract(320, 240, 6);
        assert_domain_bit_parity(
            "OH264 CoeffSign(320×240×6)",
            &oh264.coeff_sign_bypass,
            &walker.coeff_sign_bypass.bits,
        );
        assert_domain_bit_parity(
            "OH264 CoeffSuffixLsb(320×240×6)",
            &oh264.coeff_suffix_lsb,
            &walker.coeff_suffix_lsb.bits,
        );
        assert_domain_bit_parity(
            "OH264 MvdSignBypass(320×240×6)",
            &oh264.mvd_sign_bypass,
            &walker.mvd_sign_bypass.bits,
        );
        assert_domain_bit_parity(
            "OH264 MvdSuffixLsb(320×240×6)",
            &oh264.mvd_suffix_lsb,
            &walker.mvd_suffix_lsb.bits,
        );
    }
}
