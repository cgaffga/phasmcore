// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! F.A — session API with shadows. Integration tests for the
//! `Av1StreamingEncodeSession::create_with_shadows` + per-GOP
//! shadow embed + `finish_shadow_first_match` decode path.
//!
//! Per-GOP shadow scope (v0.6): each GOP independently carries the
//! full shadow message; any one GOP recovers it. True whole-video
//! scope per design § 2 is deferred to v0.7+.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;

use phasm_core::codec::av1::stego::session::{
    Av1ShadowSpec, Av1StreamingDecodeSession, Av1StreamingEncodeParams,
    Av1StreamingEncodeSession,
};
use phasm_core::stego::payload;

const Q: usize = 30;

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv(source: &str, width: u32, height: u32, seek_s: f32) -> Vec<u8> {
    let src = corpus_root().join(source);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!("scale={width}:{height}:force_original_aspect_ratio=disable");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args([
            "-frames:v",
            "1",
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
    let expected = (width * height * 3 / 2) as usize;
    assert_eq!(out.stdout.len(), expected);
    out.stdout
}

/// F.A — single-GOP session with 1 shadow. The session encode +
/// decode round-trip both messages via their own passphrase.
#[test]
fn fa_session_single_gop_primary_plus_one_shadow() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260605");
    }
    let primary_msg = b"F.A session primary";
    let primary_pass = "fa-primary";
    let shadow_text = "F.A session shadow at 1088x1920";
    let shadow_payload = payload::encode_payload(shadow_text, &[]).unwrap();
    let shadow_pass = "fa-shadow";

    // 1088×1920 (~1080p portrait) — exercises the post-F.7-diag
    // max_w path. Cover is ~1 M bits → plenty for shadows + primary.
    let (w, h) = (1088u32, 1920u32);
    let yuv = extract_yuv("iphone5_1080p_30fps_h264_high.mov", w, h, 1.0);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: 1,
    };
    let shadows = vec![Av1ShadowSpec {
        passphrase: shadow_pass.into(),
        message: shadow_payload.clone(),
    }];
    let mut enc =
        Av1StreamingEncodeSession::create_with_shadows(primary_pass, primary_msg, params, shadows, 16)
            .expect("create_with_shadows");
    let mut stego = Vec::new();
    enc.push_frame(&yuv, &mut stego).expect("push_frame");
    enc.finish(&mut stego).expect("finish");
    eprintln!(
        "[F.A] single-GOP session w/ 1 shadow: stego {} bytes",
        stego.len()
    );

    // Primary recovers via session finish (primary passphrase).
    let mut dec = Av1StreamingDecodeSession::create(primary_pass);
    dec.push_bytes(&stego);
    // Capture shadow first (borrows &self) before finish() consumes it.
    let recovered_shadow = dec
        .finish_shadow_first_match(shadow_pass)
        .expect("shadow recover");
    let recovered_primary = dec.finish().expect("primary recover");
    assert_eq!(recovered_primary.as_slice(), primary_msg);
    assert_eq!(recovered_shadow.text, shadow_text);
}

/// F.A — multi-GOP session with 1 shadow at small resolution. Three
/// frames pushed; each becomes a GOP carrying the full shadow.
/// Decoder recovers both primary + shadow.
#[test]
fn fa_session_3gop_primary_plus_one_shadow_small() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260605");
    }
    let primary_msg = b"F.A 3-GOP primary - chunk reassembly works";
    let primary_pass = "fa-3gop-primary";
    let shadow_text = "F.A 3-GOP shadow";
    let shadow_payload = payload::encode_payload(shadow_text, &[]).unwrap();
    let shadow_pass = "fa-3gop-shadow";

    let (w, h) = (144u32, 256u32);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: 3,
    };
    let shadows = vec![Av1ShadowSpec {
        passphrase: shadow_pass.into(),
        message: shadow_payload.clone(),
    }];
    let mut enc = Av1StreamingEncodeSession::create_with_shadows(
        primary_pass,
        primary_msg,
        params,
        shadows,
        16,
    )
    .expect("create_with_shadows");
    let mut stego = Vec::new();
    for seek_s in [0.5_f32, 1.0, 1.5] {
        let yuv = extract_yuv("Artlist_CarPlane.mp4", w, h, seek_s);
        enc.push_frame(&yuv, &mut stego).expect("push_frame");
    }
    enc.finish(&mut stego).expect("finish");

    let mut dec = Av1StreamingDecodeSession::create(primary_pass);
    dec.push_bytes(&stego);
    let recovered_shadow = dec
        .finish_shadow_first_match(shadow_pass)
        .expect("shadow recover");
    let recovered_primary = dec.finish().expect("primary recover");
    assert_eq!(recovered_primary.as_slice(), primary_msg);
    assert_eq!(recovered_shadow.text, shadow_text);
}

/// F.A — multi-GOP session with TWO shadows. Each shadow has its own
/// passphrase; both recoverable via their own passphrase. The 1088×1920
/// fixture gives a large per-GOP cover so 2-shadow collisions stay
/// well within parity=32's tolerance.
#[test]
fn fa_session_multi_gop_primary_plus_two_shadows_1080p() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260605");
    }
    let primary_msg = b"F.A 1080p multi-GOP w/ 2 shadows";
    let primary_pass = "fa-1080p-primary";
    let shadow_a_text = "F.A shadow A whole-video span 1080p";
    let shadow_b_text = "F.A shadow B whole-video span 1080p";
    let payload_a = payload::encode_payload(shadow_a_text, &[]).unwrap();
    let payload_b = payload::encode_payload(shadow_b_text, &[]).unwrap();

    let (w, h) = (1088u32, 1920u32);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: 2,
    };
    let shadows = vec![
        Av1ShadowSpec {
            passphrase: "shadow-a".into(),
            message: payload_a.clone(),
        },
        Av1ShadowSpec {
            passphrase: "shadow-b".into(),
            message: payload_b.clone(),
        },
    ];
    let mut enc = Av1StreamingEncodeSession::create_with_shadows(
        primary_pass,
        primary_msg,
        params,
        shadows,
        32,
    )
    .expect("create_with_shadows");
    let mut stego = Vec::new();
    for seek_s in [1.0_f32, 1.5] {
        let yuv = extract_yuv("iphone5_1080p_30fps_h264_high.mov", w, h, seek_s);
        enc.push_frame(&yuv, &mut stego).expect("push_frame");
    }
    enc.finish(&mut stego).expect("finish");
    eprintln!(
        "[F.A] 1080p 2-GOP + 2 shadows: stego {} bytes",
        stego.len()
    );

    let mut dec = Av1StreamingDecodeSession::create(primary_pass);
    dec.push_bytes(&stego);
    let recovered_a = dec
        .finish_shadow_first_match("shadow-a")
        .expect("shadow A recover");
    let recovered_b = dec
        .finish_shadow_first_match("shadow-b")
        .expect("shadow B recover");
    let recovered_primary = dec.finish().expect("primary recover");

    assert_eq!(recovered_primary.as_slice(), primary_msg);
    assert_eq!(recovered_a.text, shadow_a_text);
    assert_eq!(recovered_b.text, shadow_b_text);
}

/// F.A — wrong shadow passphrase returns an error rather than
/// silently returning a "primary" or other shadow's message.
#[test]
fn fa_session_wrong_shadow_passphrase_fails() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260605");
    }
    let (w, h) = (144u32, 256u32);
    let yuv = extract_yuv("Artlist_CarPlane.mp4", w, h, 1.0);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: 1,
    };
    let real_shadow_payload = payload::encode_payload("real shadow content", &[]).unwrap();
    let shadows = vec![Av1ShadowSpec {
        passphrase: "right-shadow".into(),
        message: real_shadow_payload,
    }];
    let mut enc =
        Av1StreamingEncodeSession::create_with_shadows("primary", b"primary", params, shadows, 16)
            .unwrap();
    let mut stego = Vec::new();
    enc.push_frame(&yuv, &mut stego).unwrap();
    enc.finish(&mut stego).unwrap();

    let mut dec = Av1StreamingDecodeSession::create("primary");
    dec.push_bytes(&stego);
    let r = dec.finish_shadow_first_match("wrong-shadow");
    assert!(
        r.is_err(),
        "wrong shadow passphrase must fail; got Ok({:?})",
        r.ok()
    );
}
