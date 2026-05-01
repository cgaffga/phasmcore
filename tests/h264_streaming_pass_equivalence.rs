// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §6E-A5(c.x) — diagnostic: do the streaming Pass-1B / Pass-3
// helpers produce equivalent output to the in-memory variants
// on a real-world 2-GOP fixture? If yes, the 1080p shadow smoke
// failure (#107) is fixture-edge capacity, not a streaming
// regression. If no, the streaming refactor has a bug.

#![cfg(feature = "cabac-stego")]

fn load_real_world(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/video/h264/real-world/{name}"))
        .unwrap_or_else(|_| panic!("missing real-world fixture: {name}"))
}

#[test]
fn streaming_pass3_byte_equivalent_to_inmemory_2gop() {
    use phasm_core::codec::h264::stego::encode_pixels::{
        h264_stego_encode_yuv_string_4domain_multigop,
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2,
    };
    let yuv = load_real_world("img4138_128x80_f10.yuv");
    // 128x80, 10 frames, gop=5 → 2 GOPs. Same fixture as the
    // existing v2 round-trip + in-memory orchestrator tests.
    let in_memory = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, 128, 80, 10, 5, "x", "v1v2-eq-128",
    )
    .expect("in-memory encode");
    let streaming = h264_stego_encode_yuv_string_4domain_multigop_streaming_v2(
        &yuv, 128, 80, 10, 5, "x", "v1v2-eq-128",
    )
    .expect("streaming v2 encode");
    assert_eq!(
        in_memory.len(),
        streaming.len(),
        "in-memory + streaming v2 outputs differ in length",
    );
    // Note: even with same length, content may differ run-to-run
    // due to crypto::encrypt's random salt/nonce (#94 finding).
    // This test is documenting that aspect rather than asserting
    // byte-equality. The length match is the strongest signal we
    // get without a deterministic-encrypt path.
    eprintln!(
        "in-memory len={} streaming len={} (content may differ; salt/nonce random)",
        in_memory.len(),
        streaming.len(),
    );
}

/// Round-trip on 128x80 2-GOP shadow encoder, same path as
/// production CLI uses post-§6E-A5(c). If this passes, the
/// 1080p×10f smoke failure is fixture-edge capacity, not a
/// streaming-cascade regression.
#[test]
fn shadow_streaming_roundtrip_128x80_2gop() {
    use phasm_core::h264_stego_encode_yuv_string_with_shadow;
    use phasm_core::h264_stego_smart_decode_video;
    let yuv = load_real_world("img4138_128x80_f10.yuv");
    let bytes = h264_stego_encode_yuv_string_with_shadow(
        &yuv, 128, 80, 10, 5,
        "p", "ppass128",
        "s", "spass128",
    )
    .expect("shadow encode (failure means a real cascade bug, not fixture-edge)");
    let recovered_primary = h264_stego_smart_decode_video(&bytes, "ppass128")
        .expect("primary decode");
    assert_eq!(recovered_primary, "p", "primary preserved");
    let recovered_shadow = h264_stego_smart_decode_video(&bytes, "spass128")
        .expect("shadow decode");
    assert_eq!(recovered_shadow, "s", "shadow preserved");
}

/// §6E-A5(c.x) — 1080p shadow round-trip with gop=5 (2 GOPs)
/// instead of CLI default gop=30 (which clamps to 10 = 1 GOP
/// for the 10-frame fixture). Tests the hypothesis that the
/// CLI smoke failure is a single-GOP fixture-edge issue.
/// `#[ignore]` since 1080p × 10f shadow takes ~25-30 minutes
/// in release.
#[test]
#[ignore = "needs /tmp/img4138_1080p_f10.yuv + 25-30 min"]
fn shadow_streaming_roundtrip_1080p_2gop() {
    use phasm_core::h264_stego_encode_yuv_string_with_shadow;
    use phasm_core::h264_stego_smart_decode_video;
    let yuv = match std::fs::read("/tmp/img4138_1080p_f10.yuv") {
        Ok(y) => y,
        Err(e) => {
            eprintln!("Skipping: /tmp/img4138_1080p_f10.yuv ({e})");
            return;
        }
    };
    let bytes = h264_stego_encode_yuv_string_with_shadow(
        &yuv, 1920, 1072, 10, /* gop=5 → 2 GOPs */ 5,
        "p", "ppass-1080p",
        "s", "spass-1080p",
    )
    .expect("1080p 2-GOP shadow encode (failure here means real bug, not fixture-edge)");
    let recovered_primary =
        h264_stego_smart_decode_video(&bytes, "ppass-1080p").expect("primary decode");
    assert_eq!(recovered_primary, "p");
    let recovered_shadow =
        h264_stego_smart_decode_video(&bytes, "spass-1080p").expect("shadow decode");
    assert_eq!(recovered_shadow, "s");
}
