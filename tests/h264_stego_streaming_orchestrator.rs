// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §long-form-stego Phase 5 — streaming orchestrator round-trip.
//
// Verifies h264_stego_encode_yuv_string_4domain_multigop_streaming
// produces a stego stream that decodes via the standard
// h264_stego_decode_yuv_string_4domain. Bytes diverge from the
// in-memory orchestrator (per-GOP fresh encoder re-emits SPS+PPS
// at every GOP IDR, spec-allowed) but cover positions + STC plan
// + stego payload are identical.

#![cfg(feature = "cabac-stego")]

use phasm_core::{
    h264_stego_decode_yuv_string_4domain,
    h264_stego_encode_yuv_string_4domain_multigop_streaming,
};

fn load_real_world(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/video/h264/real-world/{name}"))
        .unwrap_or_else(|_| panic!("missing real-world fixture: {name}"))
}

#[test]
fn streaming_orchestrator_roundtrip_64x48_5f_multigop() {
    // gop_size=5 = single GOP at n=5 (the streaming path's
    // per-GOP loop runs once). Validates the basic pipeline.
    let yuv = load_real_world("img4138_64x48_f5.yuv");
    let bytes = h264_stego_encode_yuv_string_4domain_multigop_streaming(
        &yuv, 64, 48, 5, 5, "x", "stream-roundtrip-64",
    )
    .expect("streaming encode");
    let recovered = h264_stego_decode_yuv_string_4domain(&bytes, "stream-roundtrip-64")
        .expect("streaming decode");
    assert_eq!(recovered, "x", "streaming round-trip preserves payload");
}

#[test]
fn streaming_orchestrator_roundtrip_128x80_10f_2gops() {
    // 10 frames at gop_size=5 → 2 GOPs. Exercises the per-GOP
    // Pass 3 loop with multiple iterations + per-GOP plan slicing.
    let yuv = load_real_world("img4138_128x80_f10.yuv");
    let bytes = h264_stego_encode_yuv_string_4domain_multigop_streaming(
        &yuv, 128, 80, 10, 5, "h", "stream-roundtrip-128",
    )
    .expect("streaming encode 2-GOP");
    let recovered = h264_stego_decode_yuv_string_4domain(&bytes, "stream-roundtrip-128")
        .expect("streaming decode 2-GOP");
    assert_eq!(recovered, "h", "2-GOP streaming round-trip preserves payload");
}
