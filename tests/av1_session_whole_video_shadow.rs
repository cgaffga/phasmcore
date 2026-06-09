// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! WV.5 — whole-video shadow integration tests.
//!
//! Exercises Av1StreamingEncodeSession in whole-video shadow mode:
//! shadows spread across all GOPs of a clip, recoverable only from
//! the full stream (NOT from any single GOP slab — that's the
//! per-GOP scope's property which we explicitly want NOT to hold
//! here, proving shadow is whole-video-distributed).
//!
//! Coverage:
//!
//! - WV round-trip at gop_size=4 with 1 shadow over 8 frames (2 GOPs).
//!   Asserts both primary and shadow round-trip correctly.
//!
//! Future tests (incremental):
//!
//! - 2 shadows, larger clips, AoSO re-measurement with whole-video
//!   scope vs per-GOP's 0.6775 baseline. Tracked in WV.5 follow-on.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;

use phasm_core::codec::av1::stego::session::{
    Av1ShadowSpec, Av1StreamingDecodeSession, Av1StreamingEncodeParams,
    Av1StreamingEncodeSession,
};
use phasm_core::stego::payload;

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
    out.stdout
        .chunks(frame_size)
        .map(|s| s.to_vec())
        .collect()
}

#[test]
fn wv_round_trip_gop_size_4_one_shadow_8_frames() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260605");
    }

    let primary_text = "phasm AV1 WV primary message";
    let shadow_text = "wv-shadow";
    let primary_pass = "wv-primary-pass";
    let shadow_pass = "wv-shadow-pass";

    let primary_payload = payload::encode_payload(primary_text, &[]).expect("encode primary");
    let shadow_payload = payload::encode_payload(shadow_text, &[]).expect("encode shadow");

    let yuvs = extract_n_yuv420_frames(8);
    assert_eq!(yuvs.len(), 8);

    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 4,
        total_frames_hint: 8,
    };

    let shadows = vec![Av1ShadowSpec {
        passphrase: shadow_pass.to_string(),
        message: shadow_payload.clone(),
    }];

    let mut session = Av1StreamingEncodeSession::create_whole_video_with_shadows(
        primary_pass,
        &primary_payload,
        params,
        shadows,
        16,
    )
    .expect("create_whole_video_with_shadows");

    let mut out = Vec::new();
    for yuv in &yuvs {
        session.push_frame(yuv, &mut out).expect("push_frame");
    }
    // In WV mode push doesn't emit; finish does.
    assert!(out.is_empty(), "WV mode must not emit before finish");
    session.finish(&mut out).expect("finish");
    assert!(!out.is_empty(), "WV finish must emit stego bytes");

    eprintln!("[WV] 8f gop=4 1-shadow stego: {} bytes", out.len());

    // Primary round-trip via the decode session.
    let mut decode = Av1StreamingDecodeSession::create(primary_pass);
    decode.push_bytes(&out);
    let decoded_primary = decode.finish().expect("decode primary");
    assert_eq!(
        decoded_primary, primary_payload,
        "primary round-trip mismatch"
    );

    // Shadow round-trip via WV.4's finish_shadow_whole_video — walks
    // the full accumulator's union cover (NOT per-GOP slabs). This is
    // the load-bearing decoder-symmetry property of whole-video
    // shadow scope.
    let mut decode2 = Av1StreamingDecodeSession::create(primary_pass);
    decode2.push_bytes(&out);
    let shadow_recovered = decode2
        .finish_shadow_whole_video(shadow_pass)
        .expect("decode shadow (whole-video)");
    assert_eq!(
        shadow_recovered.text.as_str(),
        shadow_text,
        "shadow round-trip mismatch"
    );

    // Negative test — the per-GOP decode method must FAIL on a
    // whole-video shadow (proves the bits are spread across GOPs,
    // not replicated per-GOP). This is the scope-distinction
    // invariant.
    let mut decode3 = Av1StreamingDecodeSession::create(primary_pass);
    decode3.push_bytes(&out);
    let per_gop_attempt = decode3.finish_shadow_first_match(shadow_pass);
    assert!(
        per_gop_attempt.is_err(),
        "per-GOP shadow decode must NOT recover a whole-video shadow \
         (proves shadow bits span GOPs); got Ok"
    );
}

#[test]
fn wv_round_trip_gop_size_4_two_shadows_12_frames() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260605");
    }

    let primary_text = "phasm AV1 WV — multi-shadow primary";
    let shadow_a_text = "wv shadow A — early";
    let shadow_b_text = "wv shadow B — later";

    let primary_payload =
        payload::encode_payload(primary_text, &[]).expect("encode primary");
    let shadow_a_payload =
        payload::encode_payload(shadow_a_text, &[]).expect("encode shadow A");
    let shadow_b_payload =
        payload::encode_payload(shadow_b_text, &[]).expect("encode shadow B");

    let yuvs = extract_n_yuv420_frames(12);
    assert_eq!(yuvs.len(), 12);

    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 4,
        total_frames_hint: 12,
    };

    let shadows = vec![
        Av1ShadowSpec {
            passphrase: "wv-pass-A".to_string(),
            message: shadow_a_payload,
        },
        Av1ShadowSpec {
            passphrase: "wv-pass-B".to_string(),
            message: shadow_b_payload,
        },
    ];

    let mut session = Av1StreamingEncodeSession::create_whole_video_with_shadows(
        "wv-primary",
        &primary_payload,
        params,
        shadows,
        16,
    )
    .expect("create_whole_video_with_shadows");

    let mut out = Vec::new();
    for yuv in &yuvs {
        session.push_frame(yuv, &mut out).expect("push_frame");
    }
    session.finish(&mut out).expect("finish");
    eprintln!("[WV] 12f gop=4 2-shadow stego: {} bytes", out.len());

    // Each shadow recoverable independently.
    let mut dec_a = Av1StreamingDecodeSession::create("wv-primary");
    dec_a.push_bytes(&out);
    let rec_a = dec_a
        .finish_shadow_whole_video("wv-pass-A")
        .expect("extract shadow A");
    assert_eq!(rec_a.text.as_str(), shadow_a_text);

    let mut dec_b = Av1StreamingDecodeSession::create("wv-primary");
    dec_b.push_bytes(&out);
    let rec_b = dec_b
        .finish_shadow_whole_video("wv-pass-B")
        .expect("extract shadow B");
    assert_eq!(rec_b.text.as_str(), shadow_b_text);

    // Wrong passphrase must fail.
    let mut dec_x = Av1StreamingDecodeSession::create("wv-primary");
    dec_x.push_bytes(&out);
    assert!(
        dec_x.finish_shadow_whole_video("wrong-pass").is_err(),
        "wrong passphrase must NOT recover any shadow"
    );

    // Primary still round-trips alongside multi-shadow.
    let mut dec_p = Av1StreamingDecodeSession::create("wv-primary");
    dec_p.push_bytes(&out);
    let recovered_primary = dec_p.finish().expect("decode primary");
    assert_eq!(recovered_primary, primary_payload);
}
