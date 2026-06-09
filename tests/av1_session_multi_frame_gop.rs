// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! D.6.4 — multi-frame-per-GOP session integration tests.
//!
//! Exercises the D.6 multi-frame primitive end-to-end via
//! `Av1StreamingEncodeSession` at `gop_size > 1`. Each test pushes
//! N frames, runs the multi-frame embed path, decodes via
//! `Av1StreamingDecodeSession`, and asserts byte-for-byte payload
//! recovery.
//!
//! Coverage:
//!
//! - Single-GOP at gop_size=4 / 10 / 30: validates inter-frame
//!   compression path on small/medium/large GOPs.
//! - Multi-GOP × gop_size=4 (12 frames → 3 GOPs): validates per-GOP
//!   independence (V6 invariant) under the multi-frame primitive.
//! - Trailing-partial GOP (10 frames at gop_size=4 → 2 full + 1
//!   partial): validates the auto-drain on total_frames_hint reach.
//!
//! D.6.5 (separate test file) adds the bitrate parity gate
//! (gop_size=30 must be substantially smaller than gop_size=1 on
//! the same content).

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;

use phasm_core::codec::av1::stego::session::{
    Av1StreamingDecodeSession, Av1StreamingEncodeParams, Av1StreamingEncodeSession,
};

const W: u32 = 256;
const H: u32 = 144;
const Q: usize = 30;
const SOURCE: &str = "IMG_4138.MOV";

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_n_yuv420_frames(n: usize) -> Vec<Vec<u8>> {
    let src = corpus_root().join(SOURCE);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!("scale={W}:{H}:force_original_aspect_ratio=disable");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args([
            "-frames:v",
            &n.to_string(),
            "-vf",
            &vf,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .expect("ffmpeg launch");
    assert!(out.status.success(), "ffmpeg yuv extract failed");
    let frame_size = (W * H * 3 / 2) as usize;
    assert_eq!(
        out.stdout.len(),
        frame_size * n,
        "extracted {} bytes, expected {} for {} frames",
        out.stdout.len(),
        frame_size * n,
        n
    );
    out.stdout
        .chunks(frame_size)
        .map(|s| s.to_vec())
        .collect()
}

fn set_deterministic_seed() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260605");
    }
}

fn round_trip(
    message: &[u8],
    passphrase: &str,
    gop_size: u32,
    n_frames: usize,
) -> usize {
    set_deterministic_seed();
    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size,
        total_frames_hint: n_frames as u32,
    };
    let yuvs = extract_n_yuv420_frames(n_frames);

    let mut session = Av1StreamingEncodeSession::create(passphrase, message, params).unwrap();
    let expected_chunks = (n_frames as u32).div_ceil(gop_size);
    assert_eq!(
        session.total_chunks() as u32,
        expected_chunks,
        "total_chunks mismatch for {n_frames} frames at gop_size={gop_size}"
    );

    let mut out = Vec::new();
    for yuv in &yuvs {
        session.push_frame(yuv, &mut out).unwrap();
    }
    session.finish(&mut out).unwrap();
    assert!(!out.is_empty(), "session.finish produced empty output");

    let mut decode = Av1StreamingDecodeSession::create(passphrase);
    decode.push_bytes(&out);
    let plaintext = decode.finish().unwrap();
    assert_eq!(
        plaintext, message,
        "decoded plaintext != original message (gop_size={gop_size}, n_frames={n_frames})"
    );

    out.len()
}

#[test]
fn d6_gop_size_4_single_gop_round_trip() {
    let bytes = round_trip(
        b"phasm AV1 D.6.4 gop_size=4 single GOP",
        "d6-gs4-single",
        4,
        4,
    );
    eprintln!("[D.6.4] gop_size=4 single GOP: stego_bytes={bytes}");
}

#[test]
fn d6_gop_size_10_single_gop_round_trip() {
    let bytes = round_trip(
        b"phasm AV1 D.6.4 gop_size=10 single GOP",
        "d6-gs10-single",
        10,
        10,
    );
    eprintln!("[D.6.4] gop_size=10 single GOP: stego_bytes={bytes}");
}

#[test]
fn d6_gop_size_30_single_gop_round_trip() {
    let bytes = round_trip(
        b"phasm AV1 D.6.4 gop_size=30 single GOP",
        "d6-gs30-single",
        30,
        30,
    );
    eprintln!("[D.6.4] gop_size=30 single GOP: stego_bytes={bytes}");
}

#[test]
fn d6_multi_gop_with_gop_size_4_round_trip() {
    // 12 frames at gop_size=4 → 3 GOPs (each carrying 1 chunk of the
    // chunk-split message). Validates the V6 invariant (per-GOP
    // independence) holds under the multi-frame primitive.
    let bytes = round_trip(
        b"phasm AV1 D.6.4 multi-GOP x gop_size=4 round trip - 12 frames split into 3 chunks",
        "d6-multi-gop-gs4",
        4,
        12,
    );
    eprintln!("[D.6.4] gop_size=4 × 3 GOPs: stego_bytes={bytes}");
}

#[test]
fn d6_trailing_partial_gop_round_trip() {
    // 10 frames at gop_size=4 → 3 chunks: 2 full GOPs (4 frames each)
    // + 1 partial trailing GOP (2 frames). push_frame auto-drains the
    // trailing GOP when frames_pushed_total reaches total_frames_hint;
    // finish() then is a no-op for buffered frames.
    let bytes = round_trip(
        b"phasm AV1 D.6.4 trailing partial GOP - 10 frames at gop_size=4 produces 3 chunks",
        "d6-trailing-partial",
        4,
        10,
    );
    eprintln!("[D.6.4] gop_size=4 × 2.5 GOPs: stego_bytes={bytes}");
}
