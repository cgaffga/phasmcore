// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! D.6.5 — bitrate parity gate.
//!
//! Proves that D.6 multi-frame-per-GOP actually engages inter-frame
//! compression (P-frame reference chaining) and isn't silently
//! degrading to all-keyframes.
//!
//! The contract: encoding the same N frames at `gop_size=N`
//! (1 keyframe + N-1 inter frames) must produce smaller stego
//! bytes than encoding at `gop_size=1` (every frame a keyframe).
//! Gate is set at >=1.3x as a realistic-but-discriminating
//! threshold for 256x144 slow-motion content. Higher resolution +
//! more motion produces larger ratios (3-8x at 1080p with
//! moving content); we test on the smallest content the corpus has,
//! so the gate must tolerate that constraint.
//!
//! Why not a stronger gate: at 256x144 the per-frame header
//! overhead dominates the coefficient bits, so P-frames are only
//! ~30-40% smaller than I-frames even with fully working inter
//! coding. Empirically the ratio sits at ~1.5x on this fixture.
//!
//! If this gate fails (ratio <1.3x), likely causes:
//! - The encode_gop_with_phasm_tee helper isn't actually feeding
//!   P-frames refs (cascade broken - frame 1+ falls back to intra).
//! - The fork's update_rec_buffer path isn't propagating
//!   reconstructions correctly.
//! - low_latency=true was reverted (frames went through B-reorder
//!   and bytes ballooned for a different reason).

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;

use phasm_core::codec::av1::stego::session::{
    Av1StreamingDecodeSession, Av1StreamingEncodeParams, Av1StreamingEncodeSession,
};

const W: u32 = 256;
const H: u32 = 144;
const Q: usize = 30;
const N_FRAMES: usize = 30;
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

fn encode_at_gop_size(yuvs: &[Vec<u8>], gop_size: u32, message: &[u8], pass: &str) -> Vec<u8> {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260605");
    }
    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size,
        total_frames_hint: yuvs.len() as u32,
    };
    let mut session = Av1StreamingEncodeSession::create(pass, message, params).unwrap();
    let mut out = Vec::new();
    for yuv in yuvs {
        session.push_frame(yuv, &mut out).unwrap();
    }
    session.finish(&mut out).unwrap();
    out
}

#[test]
fn d6_gop_size_30_substantially_smaller_than_gop_size_1() {
    let yuvs = extract_n_yuv420_frames(N_FRAMES);
    assert_eq!(yuvs.len(), N_FRAMES, "extracted wrong frame count");

    let message = b"phasm AV1 D.6.5 bitrate parity gate";

    let bytes_gs1 = encode_at_gop_size(&yuvs, 1, message, "d6-gs1-bitrate");
    let bytes_gs30 = encode_at_gop_size(&yuvs, N_FRAMES as u32, message, "d6-gs30-bitrate");

    let ratio = bytes_gs1.len() as f64 / bytes_gs30.len() as f64;
    eprintln!(
        "[D.6.5] {N_FRAMES} frames: gop_size=1 -> {} bytes, gop_size={N_FRAMES} -> {} bytes (ratio {:.2}x)",
        bytes_gs1.len(),
        bytes_gs30.len(),
        ratio,
    );

    // Both must round-trip — bitrate is moot if the payload doesn't
    // survive. Decode gop_size=30 (the multi-frame path); the gop_size=1
    // path is already covered by existing av1_streaming_session_d1
    // tests.
    let mut decode = Av1StreamingDecodeSession::create("d6-gs30-bitrate");
    decode.push_bytes(&bytes_gs30);
    let plaintext = decode.finish().unwrap();
    assert_eq!(plaintext, message, "gop_size={N_FRAMES} round-trip failed");

    // The compression contract: at least 1.3x smaller. At 256x144
    // slow-motion the per-frame header overhead dominates so absolute
    // inter compression caps at ~1.5x; 1.3x catches "all keyframes"
    // regressions without false-failing on legitimate inter coding.
    // Higher-resolution / higher-motion content would hit 3-8x.
    assert!(
        ratio >= 1.3,
        "D.6 inter-frame compression below contract: \
         gop_size=1 produced {}b, gop_size={N_FRAMES} produced {}b ({:.2}x - expected >=1.3x). \
         Likely cause: encode_gop_with_phasm_tee fork helper not propagating refs, \
         or low_latency=true was reverted (B-frame reorder ballooning bytes).",
        bytes_gs1.len(),
        bytes_gs30.len(),
        ratio,
    );
}
