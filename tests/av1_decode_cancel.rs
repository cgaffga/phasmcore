// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! A decode cancel stops every AV1 decode-session finish with `Cancelled`.
//!
//! The bridges run decodes through `progress::run_decode`, so the session's
//! per-GOP `check_cancelled` reads the Decode channel: a cancel in the decode
//! tab stops the AV1 decode at the next GOP instead of running the whole
//! clip, and never touches an encode on the other channel.
//!
//! Own test binary: the channel flags are process-global.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use phasm_core::codec::av1::stego::orchestrator::Av1StegoError;
use phasm_core::codec::av1::stego::session::Av1StreamingDecodeSession;
use phasm_core::codec::av1::stego::whole_video::{
    av1_encode_with_shadows_streaming, Av1SliceYuvSource, Pass1CacheBudget,
};
use phasm_core::progress::{run_decode, Channel};
use phasm_core::stego::error::StegoError;
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
        .args(["-frames:v", &n.to_string(), "-vf", &vf])
        .args(["-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output()
        .expect("ffmpeg launch");
    assert!(out.status.success(), "ffmpeg yuv extract failed");
    out.stdout
}

#[test]
#[ignore = "AV1 WV-shadow encode + ffmpeg corpus pull; run with --ignored"]
fn a_decode_cancel_stops_every_av1_decode_finish() {
    // The known-good WV-shadow fixture (2 GOPs of 4 frames).
    let (w, h, gop, n) = (256u32, 144u32, 4u32, 8u32);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };
    let yuv = corpus_yuv_concat(w, h, n);
    let (pp, sp) = ("cancel-primary", "cancel-shadow");
    let primary = payload::encode_payload("decode cancel primary", &[]).unwrap();
    let (ct, nonce, salt) = crypto::encrypt(&primary, pp).unwrap();
    let framed = frame::build_frame(primary.len(), &salt, &nonce, &ct);
    let shadow = payload::encode_payload("decode cancel shadow", &[]).unwrap();
    let shadows: &[(&str, &[u8])] = &[(sp, &shadow)];
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let stego = av1_encode_with_shadows_streaming(
        &mut src,
        n,
        params,
        &framed,
        pp,
        shadows,
        16,
        Pass1CacheBudget::from_env_or_default(),
    )
    .expect("encode");
    let session = || {
        let mut d = Av1StreamingDecodeSession::create(pp);
        d.push_bytes(&stego);
        d
    };

    Channel::Decode.cancel();
    let primary = run_decode(|| session().finish());
    let first_match = run_decode(|| session().finish_shadow_first_match(sp));
    let whole_video = run_decode(|| session().finish_shadow_whole_video(sp));
    Channel::Decode.clear_cancel();

    assert!(
        matches!(primary, Err(Av1StegoError::Stego(StegoError::Cancelled))),
        "finish: {:?}",
        primary.map(|v| v.len())
    );
    assert!(matches!(first_match, Err(StegoError::Cancelled)), "first_match: {first_match:?}");
    assert!(matches!(whole_video, Err(StegoError::Cancelled)), "whole_video: {whole_video:?}");

    // Uncancelled, the same streams decode.
    assert!(run_decode(|| session().finish()).is_ok());
    assert_eq!(
        run_decode(|| session().finish_shadow_whole_video(sp)).unwrap().text,
        "decode cancel shadow"
    );
}
