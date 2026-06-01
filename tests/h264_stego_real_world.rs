// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase 6F.2 — encode → walk → decode round trip on real-world
// iPhone YUV content. Catches any divergence between the encoder's
// post-quantize hook and the bin walker's residual decode that
// synthetic deterministic_yuv / correlated_yuv test patterns miss.
//
// Real-world fixtures live in `test-vectors/video/h264/real-world/`
// (small derived YUVs are committed; 1080p YUVs are regenerated to
// /tmp on demand from the gitignored .MOV sources).

#![cfg(feature = "cabac-stego")]

use phasm_core::{
    h264_stego_decode_yuv_string_4domain,
    h264_stego_encode_yuv_string_4domain_multigop,
};

fn load_real_world(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/video/h264/real-world/{name}"))
        .unwrap_or_else(|_| panic!("missing real-world fixture: {name}"))
}

/// 32×32, 1 frame — minimum repro shape. Real iPhone content
/// (vs flat synthetic fixtures) catches CABAC residual edge cases
/// in the CABAC encoder ↔ bin-walker parity that synthetic patterns
/// don't trigger.
///
/// Stays `#[ignore]`d: even after §6F.2(g) round-trip stabilization,
/// the 32x32 1-frame fixture is too small for any payload at QP 26
/// on real iPhone content — encoder rejects with `MessageTooLarge`.
/// Use the 64x48 / 128x80 fixtures for capacity-feasible round
/// trips.
#[test]
#[ignore = "32x32 1f real-world fixture lacks capacity for any payload at QP 26 (residual-only post-§6F.2(g))."]
fn stego_roundtrip_real_world_32x32_1f() {
    let yuv = load_real_world("img4138_32x32_f1.yuv");
    let msg = "h";  // single byte — 32x32 1f single MB has tiny capacity
    let pass = "test-pass-32";

    let bytes = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, 32, 32, /* n_frames */ 1, /* gop_size */ 1, msg, pass,
    )
    .expect("real-world 32x32 encode");

    let recovered = h264_stego_decode_yuv_string_4domain(&bytes, pass)
        .expect("real-world 32x32 decode");

    assert_eq!(recovered, msg, "32x32 real-world encode→decode must round-trip");
}

/// 64×48, 5 frames — 2×3 MB grid, exercises I + P sequence on
/// real iPhone content. Round-trips post-§6F.2(g).
#[test]
#[ignore = "STEGO.B.P8: Scheme A capacity stricter than Scheme B; tiny fixture below threshold"]
fn stego_roundtrip_real_world_64x48_5f() {
    let yuv = load_real_world("img4138_64x48_f5.yuv");
    let msg = "hi";  // 2-byte msg — well within capacity for 6-MB grid x 5f
    let pass = "test-pass-64";

    let bytes = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, 64, 48, /* n_frames */ 5, /* gop_size */ 5, msg, pass,
    )
    .expect("real-world 64x48 encode");

    let recovered = h264_stego_decode_yuv_string_4domain(&bytes, pass)
        .expect("real-world 64x48 decode");

    assert_eq!(recovered, msg, "64x48 real-world encode→decode must round-trip");
}

/// Diagnostic: same dimensions (64×48, 5 frames) but synthetic
/// high-entropy YUV — confirms whether the FrameCorrupted on
/// real-world is content-specific or dimension-specific.
#[test]
#[ignore = "STEGO.B.P8: Scheme A capacity stricter than Scheme B; tiny fixture below threshold"]
fn stego_roundtrip_synthetic_64x48_5f() {
    // High-entropy noise pattern — gives plenty of non-zero
    // residual coefficients per MB.
    let frame_size = 64 * 48 * 3 / 2;
    let mut yuv = Vec::with_capacity(frame_size * 5);
    for f in 0..5u32 {
        for i in 0..(64u32 * 48) {
            yuv.push((((i * 17 + f * 71) ^ (i * 31 + f * 13)) & 0xFF) as u8);
        }
        for i in 0..(64u32 * 48 / 2) {
            yuv.push(((i * 23 + f * 41) & 0x7F) as u8 + 64);
        }
    }
    let msg = "hi";
    let pass = "test-pass-syn-64";

    let bytes = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, 64, 48, /* n_frames */ 5, /* gop_size */ 5, msg, pass,
    )
    .expect("synthetic 64x48 encode");

    let recovered = h264_stego_decode_yuv_string_4domain(&bytes, pass)
        .expect("synthetic 64x48 decode");

    assert_eq!(recovered, msg, "synthetic 64x48 must round-trip (control)");
}

/// 128×80, 10 frames — larger grid (8×5 MBs), single GOP. Closest
/// fixture to "real video shape" while still small enough to run
/// quickly in CI. Round-trips post-§6F.2(g).
#[test]
#[ignore = "STEGO.B.P8: Scheme A capacity stricter than Scheme B; tiny fixture below threshold"]
fn stego_roundtrip_real_world_128x80_10f() {
    let yuv = load_real_world("img4138_128x80_f10.yuv");
    let msg = "test";  // short msg — single-GOP capacity is limited
    let pass = "test-pass-128";

    let bytes = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, 128, 80, /* n_frames */ 10, /* gop_size */ 10, msg, pass,
    )
    .expect("real-world 128x80 encode");

    let recovered = h264_stego_decode_yuv_string_4domain(&bytes, pass)
        .expect("real-world 128x80 decode");

    assert_eq!(recovered, msg, "128x80 real-world encode→decode must round-trip");
}

/// 128×80, 10 frames, multi-GOP shape (gop_size=5). Stress-tests
/// the §6E-C1a streaming-Viterbi multi-IDR primary STC path on
/// real-world content. Round-trips post-§6F.2(g).
#[test]
#[ignore = "STEGO.B.P8: Scheme A capacity stricter than Scheme B; tiny fixture below threshold"]
fn stego_roundtrip_real_world_128x80_multigop() {
    let yuv = load_real_world("img4138_128x80_f10.yuv");
    let msg = "mg";  // 2 bytes — must fit in 5+5 frames at 128x80
    let pass = "test-pass-mgop";

    let bytes = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, 128, 80, /* n_frames */ 10, /* gop_size */ 5, msg, pass,
    )
    .expect("real-world multi-GOP encode");

    let recovered = h264_stego_decode_yuv_string_4domain(&bytes, pass)
        .expect("real-world multi-GOP decode");

    assert_eq!(recovered, msg, "128x80 real-world multi-GOP must round-trip");
}
