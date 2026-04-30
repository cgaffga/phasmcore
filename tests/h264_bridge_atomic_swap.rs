// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase 6D.10 atomic-swap smoke gate.
//
// The iOS bridge (`ios-bridge/src/lib.rs`) and Android bridge
// (`android-bridge/src/lib.rs`) expose new H.264 stego entry points
// (`phasm_h264_stego_encode_yuv` / `Java_*_h264StegoEncodeYuv` and the
// matching `*_decode_annexb`) that wrap the §30D-C 4-domain CABAC
// orchestrator from phasm-core.
//
// We can't link the FFI symbols from a Rust integration test (they
// live in different crates with different crate-types), so this test
// exercises the inner functions the bridges call — same code path,
// same arguments, same return shape. The bridge wrappers add only
// pointer/length marshalling and panic guards; if this round-trip
// works, the bridges are aligned with the engine.

#![cfg(feature = "cabac-stego")]

use phasm_core::{
    h264_stego_decode_yuv_string_4domain,
    h264_stego_encode_yuv_string_4domain_multigop,
};

fn load_real_world(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/video/h264/real-world/{name}"))
        .unwrap_or_else(|_| panic!("missing real-world fixture: {name}"))
}

/// Mirrors the bridge call signature exactly:
///
///     phasm_h264_stego_encode_yuv(yuv, w, h, n_frames, gop, msg, pass)
///       → Annex-B Vec<u8>
///     phasm_h264_stego_decode_annexb(annex_b, pass)
///       → message String
///
/// On the real bridge there's pointer marshalling around this; the
/// inner business is identical.
#[test]
fn bridge_atomic_swap_roundtrip_64x48_5f() {
    let yuv = load_real_world("img4138_64x48_f5.yuv");
    // 1-byte payload: under §6E-A.deploy.3's default IBPBP shape the
    // 64x48 5f fixture has reduced residual capacity (B_Skip MBs
    // contribute zero residual cover), so we use the smallest payload
    // that still exercises the FFI round-trip plumbing.
    let msg = "x";
    let pass = "test-pass-bridge-64";

    let annex_b = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, 64, 48, /* n_frames */ 5, /* gop */ 5, msg, pass,
    )
    .expect("bridge-shaped encode");
    assert!(!annex_b.is_empty(), "bridge encode emits Annex-B bytes");

    let recovered = h264_stego_decode_yuv_string_4domain(&annex_b, pass)
        .expect("bridge-shaped decode");
    assert_eq!(recovered, msg, "bridge round-trip preserves message");
}

#[test]
fn bridge_atomic_swap_roundtrip_128x80_10f_multigop() {
    let yuv = load_real_world("img4138_128x80_f10.yuv");
    let msg = "atomic-swap multigop";
    let pass = "test-pass-bridge-128";

    let annex_b = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, 128, 80, /* n_frames */ 10, /* gop */ 5, msg, pass,
    )
    .expect("bridge-shaped encode (multigop)");
    let recovered = h264_stego_decode_yuv_string_4domain(&annex_b, pass)
        .expect("bridge-shaped decode (multigop)");
    assert_eq!(recovered, msg);
}

/// Sanity gate on the input-validation that both bridges replicate:
/// width/height must be 16-aligned, yuv length must match the frame
/// shape, gop_length must be in 1..=n_frames, message bytes are UTF-8.
/// The orchestrator returns `StegoError::InvalidVideo(...)` for these
/// cases, which the bridges map to `INVALID_VIDEO:detail` in the
/// caller-facing error string.
#[test]
fn bridge_atomic_swap_rejects_misaligned_dimensions() {
    // 17×16 — width not multiple of 16.
    let frame_size = (17 * 16 * 3 / 2) as usize;
    let dummy_yuv = vec![128u8; frame_size];
    let r = h264_stego_encode_yuv_string_4domain_multigop(
        &dummy_yuv, 17, 16, 1, 1, "x", "p",
    );
    assert!(
        matches!(r, Err(phasm_core::StegoError::InvalidVideo(_))),
        "expected InvalidVideo for non-16-aligned width, got {r:?}"
    );
}

#[test]
fn bridge_atomic_swap_rejects_yuv_length_mismatch() {
    // Tell the orchestrator we have 5 frames but only provide 1 frame
    // of bytes. Bridges enforce the same check up-front in their
    // pointer marshalling.
    let frame_size = (16 * 16 * 3 / 2) as usize;
    let only_one_frame = vec![128u8; frame_size];
    let r = h264_stego_encode_yuv_string_4domain_multigop(
        &only_one_frame, 16, 16, 5, 5, "x", "p",
    );
    assert!(
        matches!(r, Err(phasm_core::StegoError::InvalidVideo(_))),
        "expected InvalidVideo for short YUV buffer, got {r:?}"
    );
}
