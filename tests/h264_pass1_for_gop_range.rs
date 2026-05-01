// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §long-form-stego Phase 3 — pass1_capture_4domain_for_gop_range
// produces cover bits + positions that match what a full Pass 1
// would produce when sliced by gop_idx range. Bit-exact gate for
// the Phase 4 H264GopReplayCover adapter.

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::stego::encode_pixels::pass1_capture_4domain_for_gop_range;
use phasm_core::codec::h264::stego::encoder_hook::PositionLoggerHook;
use phasm_core::codec::h264::stego::gop_pattern::{
    iter_encode_order, FrameType, GopPattern,
};
use phasm_core::codec::h264::stego::orchestrate::GopCover;

fn deterministic_yuv(w: u32, h: u32, n_frames: usize) -> Vec<u8> {
    let frame_size = (w * h * 3 / 2) as usize;
    let mut out = Vec::with_capacity(frame_size * n_frames);
    let mut s: u32 = 0x1234_5678;
    for _ in 0..n_frames {
        for _ in 0..frame_size {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            out.push((s >> 16) as u8);
        }
    }
    out
}

fn full_pass1_with_logger(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    pattern: GopPattern,
) -> GopCover {
    let frame_size = (width * height * 3 / 2) as usize;
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = false;
    enc.enable_mvd_stego_hook = true;
    enc.enable_b_frames = pattern.has_b_frames();
    enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
    for meta in iter_encode_order(n_frames, pattern) {
        let d = meta.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        match meta.frame_type {
            FrameType::Idr => { enc.encode_i_frame(frame).unwrap(); }
            FrameType::P => { enc.encode_p_frame(frame).unwrap(); }
            FrameType::B => { enc.encode_b_frame(frame).unwrap(); }
        }
    }
    let mut hook = enc.take_stego_hook().expect("hook present");
    hook.take_cover_if_logger().expect("PositionLoggerHook drain")
}

/// Compare partial-range cover bits + positions against a sliced
/// full-range cover. The partial-range cover should equal the
/// subset of full-range positions whose `frame_idx` falls in
/// `[encode_idx_lo, encode_idx_hi)` — matching bits AND keys.
fn assert_partial_matches_full(
    partial: &GopCover,
    full: &GopCover,
    encode_idx_lo: u32,
    encode_idx_hi: u32,
    label: &str,
) {
    use phasm_core::codec::h264::stego::inject::DomainBits;
    let domains_partial: [&DomainBits; 4] = [
        &partial.cover.coeff_sign_bypass,
        &partial.cover.coeff_suffix_lsb,
        &partial.cover.mvd_sign_bypass,
        &partial.cover.mvd_suffix_lsb,
    ];
    let domains_full: [&DomainBits; 4] = [
        &full.cover.coeff_sign_bypass,
        &full.cover.coeff_suffix_lsb,
        &full.cover.mvd_sign_bypass,
        &full.cover.mvd_suffix_lsb,
    ];
    let names = ["coeff_sign", "coeff_suffix", "mvd_sign", "mvd_suffix"];

    for d in 0..4 {
        let mut full_bits = Vec::new();
        let mut full_keys = Vec::new();
        for (i, p) in domains_full[d].positions.iter().enumerate() {
            let f = p.frame_idx();
            if f >= encode_idx_lo && f < encode_idx_hi {
                full_bits.push(domains_full[d].bits[i]);
                full_keys.push(*p);
            }
        }
        assert_eq!(
            domains_partial[d].bits.len(),
            full_bits.len(),
            "{label} domain={} partial bit-count {} != full-slice bit-count {}",
            names[d],
            domains_partial[d].bits.len(),
            full_bits.len(),
        );
        assert_eq!(
            domains_partial[d].bits, full_bits,
            "{label} domain={} bit values diverge",
            names[d],
        );
        assert_eq!(
            domains_partial[d].positions, full_keys,
            "{label} domain={} PositionKey identity diverges",
            names[d],
        );
    }
}

#[test]
fn partial_range_pass1_matches_full_range_slice_ipppp() {
    let width = 64u32;
    let height = 48u32;
    let n_frames = 9usize;
    let gop_size = 3usize;
    let b_count = 0usize;
    let yuv = deterministic_yuv(width, height, n_frames);

    let pattern = GopPattern::Ipppp { gop: gop_size };
    let full = full_pass1_with_logger(
        &yuv, width, height, n_frames, pattern,
    );

    // Try several gop_start..gop_end ranges.
    let cases = [
        (0usize, 1usize),  // first GOP only
        (1, 2),            // middle GOP only
        (2, 3),            // last GOP only
        (1, 3),            // GOPs 1+2
        (0, 3),            // full range (sanity)
    ];
    for (gop_start, gop_end) in cases.iter().copied() {
        let partial = pass1_capture_4domain_for_gop_range(
            &yuv, width, height, n_frames, gop_size, b_count, Some(26),
            gop_start, gop_end,
        )
        .expect("partial-range Pass 1");
        let lo = (gop_start * gop_size) as u32;
        let hi = (gop_end * gop_size).min(n_frames) as u32;
        let label = format!("gop_range=[{gop_start}..{gop_end}) IPPPP");
        assert_partial_matches_full(&partial, &full, lo, hi, &label);
    }
}

#[test]
fn partial_range_pass1_matches_full_range_slice_ibpbp() {
    let width = 64u32;
    let height = 48u32;
    let n_frames = 9usize;
    let gop_size = 3usize;
    let b_count = 1usize;
    let yuv = deterministic_yuv(width, height, n_frames);

    let pattern = GopPattern::Ibpbp { gop: gop_size, b_count };
    let full = full_pass1_with_logger(
        &yuv, width, height, n_frames, pattern,
    );

    let cases = [
        (0usize, 1usize),
        (1, 2),
        (2, 3),
        (1, 3),
        (0, 3),
    ];
    for (gop_start, gop_end) in cases.iter().copied() {
        let partial = pass1_capture_4domain_for_gop_range(
            &yuv, width, height, n_frames, gop_size, b_count, Some(26),
            gop_start, gop_end,
        )
        .expect("partial-range Pass 1");
        let lo = (gop_start * gop_size) as u32;
        let hi = (gop_end * gop_size).min(n_frames) as u32;
        let label = format!("gop_range=[{gop_start}..{gop_end}) IBPBP");
        assert_partial_matches_full(&partial, &full, lo, hi, &label);
    }
}
