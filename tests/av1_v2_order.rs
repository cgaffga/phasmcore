// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AV1 v2 wire: keyed per-GOP cover order and per-GOP H-hat
//! (spec `docs/design/video/stego-cover-order-and-damage-costs.md` § 3, § 8,
//! § 9).

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

mod common;

use phasm_core::codec::av1::stego::session::Av1StreamingDecodeSession;
use phasm_core::codec::av1::stego::whole_video::{
    av1_encode_streaming, Av1SliceYuvSource, Pass1CacheBudget,
};
use phasm_core::stego::{crypto, frame, payload};
use phasm_core::Av1StreamingEncodeParams;

fn corpus_yuv(w: u32, h: u32, n: u32) -> Vec<u8> {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors/video/h264/real-world/source/IMG_4138.MOV");
    let vf = format!("scale={w}:{h}:force_original_aspect_ratio=disable");
    let out = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&p)
        .args(["-frames:v", &n.to_string(), "-vf", &vf, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output()
        .expect("ffmpeg launch");
    assert!(out.status.success() && out.stdout.len() == (w * h * 3 / 2 * n) as usize);
    out.stdout
}

fn framed(text: &str, pass: &str) -> Vec<u8> {
    let p = payload::encode_payload(text, &[]).unwrap();
    let (ct, nonce, salt) = crypto::encrypt(&p, pass).unwrap();
    frame::build_frame(p.len(), &salt, &nonce, &ct)
}

/// § 9: no legacy decode. A stream written by the v1 encoder (Task 3's
/// fixture, made before the v2 wire) must not decode.
#[test]
fn v1_video_does_not_decode() {
    let path = format!("{}/av1-v2/v1-fixture.obu", common::scratch_dir());
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skipping: v1 fixture missing at {path}");
        return;
    };
    let mut dec = Av1StreamingDecodeSession::create("v1-fixture");
    dec.push_bytes(&bytes);
    assert!(dec.finish_primary_payload().is_err(), "a v1 stream decoded under v2");
}

/// A 2.5-GOP clip whose message reaches the tail GOP: the decoder must
/// derive each GOP's order and H-hat from the same GOP index the encoder
/// used, the partial tail included.
#[test]
#[ignore = "AV1 encode + ffmpeg corpus; --ignored --test-threads=1"]
fn decode_uses_the_encoders_gop_index_for_a_partial_tail() {
    let (w, h, gop, n) = (256u32, 144u32, 8u32, 20u32);
    let yuv = corpus_yuv(w, h, n);
    let p = Av1StreamingEncodeParams { width: w, height: h, quantizer: 30, gop_size: gop, total_frames_hint: n };
    let text: String = (0..1500).map(|i| (b'a' + (i * 7 % 26) as u8) as char).collect();
    let f = framed(&text, "tail");
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let stego = av1_encode_streaming(&mut src, n, p, &f, "tail", Pass1CacheBudget::from_env_or_default(), None)
        .expect("encode");
    let mut dec = Av1StreamingDecodeSession::create("tail");
    dec.push_bytes(&stego);
    assert_eq!(dec.finish_primary_payload().expect("decode").text, text);
}
