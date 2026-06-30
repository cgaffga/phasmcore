// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Whole-video shadow round-trip via the AV1 streaming pull API
//! (`av1_encode_with_shadows_streaming` + `Av1StreamingDecodeSession`).
//!
//! Production gate. Pull-API correctness coverage that replaced the
//! deleted `av1_session_whole_video_shadow` push-API test when the
//! push API was removed (#225/#187, 2026-06-29).
//!
//! Encodes a real-content YUV with a primary + 1 shadow message
//! through the production streaming-pull whole-video shadow entry,
//! then decodes:
//!   - primary via `Av1StreamingDecodeSession::finish_primary_payload`
//!   - shadow  via `finish_shadow_whole_video(shadow_pass)`
//!
//! Both must round-trip. Covers:
//!   - `Av1SliceYuvSource` pull + rewind across the 2-sweep cascade
//!   - `av1_encode_with_shadows_streaming` (Sweep 1 harvest +
//!     Sweep 2 cascade with per-GOP RAM bound)
//!   - Cost-pool position selection for whole-video shadows
//!   - Per-GOP RS parity tier escalation if needed
//!   - Headerless brute-force shadow decode (whole-video union cover)

use phasm_core::codec::av1::stego::session::Av1StreamingDecodeSession;
use phasm_core::codec::av1::stego::whole_video::{
    av1_encode_with_shadows_streaming, Av1SliceYuvSource,
};
use phasm_core::stego::{crypto, frame, payload};
use phasm_core::Av1StreamingEncodeParams;

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

#[test]
#[ignore = "AV1 streaming whole-video shadow encode ~30-60 s; run with \
            --ignored --test-threads=1"]
fn pull_whole_video_one_shadow_roundtrip() {
    // Mirror of the deleted `wv_round_trip_gop_size_4_one_shadow_8_frames`
    // fixture (256×144 × 8 frames, gop=4, 1 shadow) — known good
    // cover-capacity ratio for the WV-shadow flow.
    let (w, h, gop, n) = (256, 144, 4, 8);
    let quantizer = 30;
    let primary_text = "pull-API WV-shadow primary — substantive content";
    let primary_pass = "wv-primary-pass";
    let shadow_text = "wv-shadow message";
    let shadow_pass = "wv-shadow-pass";

    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer,
        gop_size: gop,
        total_frames_hint: n,
    };
    let yuv = corpus_yuv_concat(w, h, n);

    // Frame primary
    let primary_payload =
        payload::encode_payload(primary_text, &[]).expect("encode primary payload");
    let (cipher, nonce, salt) =
        crypto::encrypt(&primary_payload, primary_pass).expect("encrypt primary");
    let primary_framed =
        frame::build_frame(primary_payload.len(), &salt, &nonce, &cipher);

    // Shadow message is payload-encoded going in (so `decode_payload`
    // can extract it back to text+files at recovery). Matches the
    // pre-deletion test pattern.
    let shadow_payload =
        payload::encode_payload(shadow_text, &[]).expect("encode shadow payload");
    let shadows: &[(&str, &[u8])] = &[(shadow_pass, &shadow_payload)];

    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let stego = av1_encode_with_shadows_streaming(
        &mut src,
        n,
        params,
        &primary_framed,
        primary_pass,
        shadows,
        16, // parity floor — matches the pre-deletion test
    )
    .expect("streaming WV-shadow encode");
    assert!(!stego.is_empty(), "encoder produced no bytes");

    // Decode primary
    let mut dec_primary = Av1StreamingDecodeSession::create(primary_pass);
    dec_primary.push_bytes(&stego);
    let recovered_primary = dec_primary
        .finish_primary_payload()
        .expect("decode primary");
    assert_eq!(
        recovered_primary.text, primary_text,
        "recovered primary text mismatch"
    );

    // Decode shadow via whole-video union-cover path
    let dec_shadow = Av1StreamingDecodeSession::create(primary_pass);
    let mut dec_shadow = dec_shadow;
    dec_shadow.push_bytes(&stego);
    let recovered_shadow = dec_shadow
        .finish_shadow_whole_video(shadow_pass)
        .expect("decode shadow (whole-video)");
    assert_eq!(
        recovered_shadow.text, shadow_text,
        "recovered shadow text mismatch"
    );
}
