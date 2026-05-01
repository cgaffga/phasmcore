// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §long-form-stego Phase 2 — pass1_count_per_gop_4domain produces
// per-GOP per-domain counts that match a full PositionLoggerHook
// run sliced by gop_idx.

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::stego::encode_pixels::pass1_count_per_gop_4domain;
use phasm_core::codec::h264::stego::encoder_hook::PositionLoggerHook;
use phasm_core::codec::h264::stego::orchestrate::GopCover;
use phasm_core::codec::h264::stego::gop_pattern::{
    iter_encode_order, GopPattern,
};
use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};

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

/// Run a full PositionLoggerHook Pass 1 across the entire YUV.
/// Used as the ground-truth reference: per-GOP counts are derived
/// by partitioning the resulting `GopCover.positions` arrays by
/// `frame_idx` ranges.
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
            phasm_core::codec::h264::stego::gop_pattern::FrameType::Idr => {
                enc.encode_i_frame(frame).expect("i-frame");
            }
            phasm_core::codec::h264::stego::gop_pattern::FrameType::P => {
                enc.encode_p_frame(frame).expect("p-frame");
            }
            phasm_core::codec::h264::stego::gop_pattern::FrameType::B => {
                enc.encode_b_frame(frame).expect("b-frame");
            }
        }
    }
    let mut hook = enc.take_stego_hook().expect("hook present");
    hook.take_cover_if_logger().expect("PositionLoggerHook drain")
}

/// Slice a `GopCover` by gop_idx range, returning per-domain counts
/// of positions whose `frame_idx` falls in `[frame_lo, frame_hi)`.
fn count_in_frame_range(cov: &GopCover, frame_lo: u32, frame_hi: u32) -> [usize; 4] {
    let mut out = [0usize; 4];
    let domains = [
        &cov.cover.coeff_sign_bypass,
        &cov.cover.coeff_suffix_lsb,
        &cov.cover.mvd_sign_bypass,
        &cov.cover.mvd_suffix_lsb,
    ];
    for (d, dom) in domains.iter().enumerate() {
        for p in dom.positions.iter() {
            let f = p.frame_idx();
            if f >= frame_lo && f < frame_hi {
                out[d] += 1;
            }
        }
    }
    out
}

#[test]
fn counting_hook_per_gop_matches_logger_slice_ipppp() {
    // 9 frames at 64x48, gop=3, IPPPP shape (b_count=0). Three
    // GOPs at frame_idx ranges [0,3), [3,6), [6,9).
    let width = 64u32;
    let height = 48u32;
    let n_frames = 9usize;
    let gop_size = 3usize;
    let b_count = 0usize;
    let yuv = deterministic_yuv(width, height, n_frames);

    let counts = pass1_count_per_gop_4domain(
        &yuv, width, height, n_frames, gop_size, b_count, Some(26),
    )
    .expect("count_per_gop");

    let pattern = GopPattern::Ipppp { gop: gop_size };
    let logger_cover = full_pass1_with_logger(
        &yuv, width, height, n_frames, pattern,
    );

    assert_eq!(counts.len(), 3, "expected 3 GOP rows for n=9 gop=3");
    for (g, row) in counts.iter().enumerate() {
        let lo = (g * gop_size) as u32;
        let hi = ((g + 1) * gop_size) as u32;
        let logger_row = count_in_frame_range(&logger_cover, lo, hi);
        assert_eq!(
            row, &logger_row,
            "GOP {g} per-domain counts diverge: counter={:?} logger={:?}",
            row, logger_row,
        );
    }
}

#[test]
fn counting_hook_per_gop_matches_logger_slice_ibpbp() {
    // Same fixture under IBPBP M=2. 9 frames, gop=3, b_count=1.
    // Display order: I_0, B_1, P_2, I_3, B_4, P_5, I_6, B_7, P_8.
    // Encode order (per GOP): I_0, P_2, B_1, I_3, P_5, B_4, ...
    // Per-GOP frame_idx ranges (encode-order):
    //   GOP 0: encode_idx ∈ [0..3)
    //   GOP 1: encode_idx ∈ [3..6)
    //   GOP 2: encode_idx ∈ [6..9)
    let width = 64u32;
    let height = 48u32;
    let n_frames = 9usize;
    let gop_size = 3usize;
    let b_count = 1usize;
    let yuv = deterministic_yuv(width, height, n_frames);

    let counts = pass1_count_per_gop_4domain(
        &yuv, width, height, n_frames, gop_size, b_count, Some(26),
    )
    .expect("count_per_gop");

    let pattern = GopPattern::Ibpbp { gop: gop_size, b_count };
    let logger_cover = full_pass1_with_logger(
        &yuv, width, height, n_frames, pattern,
    );

    assert_eq!(counts.len(), 3);
    for (g, row) in counts.iter().enumerate() {
        let lo = (g * gop_size) as u32;
        let hi = ((g + 1) * gop_size) as u32;
        let logger_row = count_in_frame_range(&logger_cover, lo, hi);
        assert_eq!(
            row, &logger_row,
            "GOP {g} IBPBP per-domain counts diverge: counter={:?} logger={:?}",
            row, logger_row,
        );
    }
}

/// Sanity gate: total counts across all GOPs equal the
/// PositionLoggerHook's drained total. Catches off-by-one errors
/// in the per-GOP boundary detection.
#[test]
fn counting_hook_total_matches_logger_total() {
    let width = 64u32;
    let height = 48u32;
    let n_frames = 6usize;
    let gop_size = 3usize;
    let b_count = 0usize;
    let yuv = deterministic_yuv(width, height, n_frames);

    let counts = pass1_count_per_gop_4domain(
        &yuv, width, height, n_frames, gop_size, b_count, Some(26),
    )
    .expect("count_per_gop");

    let pattern = GopPattern::Ipppp { gop: gop_size };
    let logger = full_pass1_with_logger(
        &yuv, width, height, n_frames, pattern,
    );

    let total_counter: [usize; 4] = counts.iter().fold([0; 4], |mut acc, row| {
        for d in 0..4 {
            acc[d] += row[d];
        }
        acc
    });
    let total_logger = [
        logger.cover.coeff_sign_bypass.positions.len(),
        logger.cover.coeff_suffix_lsb.positions.len(),
        logger.cover.mvd_sign_bypass.positions.len(),
        logger.cover.mvd_suffix_lsb.positions.len(),
    ];
    assert_eq!(
        total_counter, total_logger,
        "total counts diverge: counter={total_counter:?} logger={total_logger:?}",
    );
}
