// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! No-shadow round-trip via the AV1 streaming pull API
//! (`av1_encode_streaming` + `Av1StreamingDecodeSession`).
//!
//! Production gate. Pull-API correctness coverage that replaced the
//! deleted byte-identity gate (#219) and the deleted
//! `av1_session_multi_frame_gop` push-API test when the push API was
//! removed (#225/#187, 2026-06-29).
//!
//! Encodes a real-content YUV through the production pull entry, then
//! decodes the resulting AV1 OBU stream and asserts the recovered
//! payload matches the original. Covers:
//!   - `Av1SliceYuvSource` rewind + per-GOP pull contract
//!   - `av1_encode_streaming` + STC primary-message embed
//!   - Per-GOP `chunk_frame_v3.1` framing
//!   - `Av1StreamingDecodeSession` OBU-walker accumulator + decrypt

use phasm_core::codec::av1::stego::session::Av1StreamingDecodeSession;
use phasm_core::codec::av1::stego::whole_video::{
    av1_encode_streaming, Av1SliceYuvSource,
};
use phasm_core::stego::{crypto, frame, payload};
use phasm_core::Av1StreamingEncodeParams;

/// Pull a corpus YUV from the test-vector clip via ffmpeg. Real
/// content so rav1e's RDO has actual signal to encode — synthetic
/// patterns at small fixtures starve the cover and the encode trips
/// `FrameCorrupted` before the assert runs. Same fixture the
/// byte-identity gate used pre-deletion.
fn corpus_yuv_concat(w: u32, h: u32, n: u32) -> Vec<u8> {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source/IMG_4138.MOV");
    assert!(p.exists(), "corpus fixture missing: {}", p.display());
    let vf = format!("scale={w}:{h}:force_original_aspect_ratio=disable");
    let out = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&p)
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
    assert_eq!(
        out.stdout.len(),
        (w * h * 3 / 2 * n) as usize,
        "ffmpeg produced unexpected byte count"
    );
    out.stdout
}

fn roundtrip_at_fixture(
    w: u32,
    h: u32,
    gop: u32,
    n: u32,
    quantizer: usize,
    primary_text: &str,
    primary_pass: &str,
) {
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer,
        gop_size: gop,
        total_frames_hint: n,
    };
    let yuv = corpus_yuv_concat(w, h, n);

    let primary_payload =
        payload::encode_payload(primary_text, &[]).expect("encode primary payload");
    let (ciphertext, nonce, salt) =
        crypto::encrypt(&primary_payload, primary_pass).expect("encrypt primary");
    let primary_framed =
        frame::build_frame(primary_payload.len(), &salt, &nonce, &ciphertext);

    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let stego = av1_encode_streaming(
        &mut src,
        n,
        params,
        &primary_framed,
        primary_pass,
        None,
    )
    .expect("streaming encode");
    assert!(!stego.is_empty(), "encoder produced no bytes");

    let mut dec = Av1StreamingDecodeSession::create(primary_pass);
    dec.push_bytes(&stego);
    let recovered = dec
        .finish_primary_payload()
        .expect("decode finish_primary_payload");
    assert_eq!(
        recovered.text, primary_text,
        "recovered primary text mismatch"
    );
    assert!(
        recovered.files.is_empty(),
        "no files were encoded — recovered.files must be empty, got {} entry/ies",
        recovered.files.len()
    );
}

/// 3 full GOPs, no partial trailing — exercises the steady-state
/// per-GOP encode + chunk-split path.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull ~30-60 s; run with \
            --ignored --test-threads=1"]
fn pull_no_shadow_roundtrip_3_full_gops() {
    roundtrip_at_fixture(
        256, 144, 4, 12, // w, h, gop, n_frames
        30,              // quantizer (high quality, lots of cover)
        "pull-API no-shadow round-trip — substantive primary message",
        "pull-no-shadow-pass",
    );
}

/// 2 full GOPs + 1 partial 3-frame trailing GOP — validates the
/// trailing-partial-GOP path inside `av1_encode_streaming`.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull ~30-60 s; run with \
            --ignored --test-threads=1"]
fn pull_no_shadow_roundtrip_with_partial_tail() {
    roundtrip_at_fixture(
        256, 144, 4, 11, // n_frames not divisible by gop
        30,
        "pull-API no-shadow with partial-tail GOP",
        "pull-no-shadow-partial",
    );
}
