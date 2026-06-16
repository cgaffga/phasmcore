// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

// WV.6.g.1 LINCHPIN — is a per-GOP-standalone encode bit-deterministic across
// two runs? If yes, the streaming shadow can DELETE the per-GOP DecisionCache
// (the dominant whole-clip memory term): Pass 2 re-encodes the re-decoded GOP
// (fresh encoder + #548 reset) and gets the identical cover Pass 1 produced —
// no need to retain Pass-1's mode decisions in RAM (the #533 cache) or spill
// them. The #533 "~200-byte drift between two clean encodes" was a WHOLE-CLIP
// cross-GOP single-session bug; the #548 reset + the per-GOP-standalone
// structure (WV.6.b.2) should make a single GOP reproducible.
//
// This confirms it at 1080p on REAL content (CarPlane — where #533 first bit):
// encode the same GOP twice with the production wire_only path and assert the
// two full bitstreams are byte-identical. (The in-encoder "determinism"
// diagnostic — clean re-encode vs Pass-1 — also prints under PHASM_PERF_TRACE.)

#![cfg(feature = "h264-encoder")]

use phasm_core::codec::h264::openh264_stego::{h264_encode_gop_framed_bits_auto, EncodeOpts};
use phasm_core::codec::h264::stego::CostWeights;

/// Demux the real CarPlane corpus clip to tight-I420 YUV (ffmpeg = fixture prep).
fn real_carplane_yuv(w: u32, h: u32, n: u32) -> Vec<u8> {
    let path = format!("/tmp/phasm_g1_carplane_{w}x{h}_f{n}.yuv");
    let need = (w * h * 3 / 2) as usize * n as usize;
    if let Ok(d) = std::fs::read(&path) {
        if d.len() >= need {
            return d[..need].to_vec();
        }
    }
    let src = format!(
        "{}/test-vectors/video/h264/real-world/source/Artlist_CarPlane.mp4",
        env!("CARGO_MANIFEST_DIR")
    );
    assert!(std::path::Path::new(&src).exists(), "corpus fixture missing: {src}");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &n.to_string(), "-an", "-vf", &format!("scale={w}:{h}")])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&path)
        .status()
        .expect("ffmpeg launch");
    assert!(status.success(), "ffmpeg demux failed");
    let d = std::fs::read(&path).expect("read yuv");
    assert!(d.len() >= need, "ffmpeg produced {} bytes, need {need}", d.len());
    d[..need].to_vec()
}

#[test]
#[ignore = "WV.6.g.1 linchpin: real 1080p GOP encoded twice, ~mins. --ignored --test-threads=1 --nocapture"]
fn wv6_g1_clean_per_gop_encode_is_deterministic() {
    // One 30-frame GOP at 1080p (intra_period 60 > 30 ⇒ a single GOP), real
    // content, production wire_only path.
    unsafe { std::env::set_var("PHASM_USE_WIRE_ONLY", "1") };
    unsafe { std::env::set_var("PHASM_PERF_TRACE", "1") }; // surfaces the in-encoder clean-reencode first_diff

    let (w, h, n) = (1072u32, 1920u32, 30u32);
    let opts = EncodeOpts { qp: 26, intra_period: 60 };
    let weights = CostWeights::default();
    let hhat_seed = [42u8; 32];
    // A modest primary payload (the 1080p cover is huge, so this fits easily).
    let frame_bits: Vec<u8> = (0..2048u32).map(|i| (i & 1) as u8).collect();

    let yuv = real_carplane_yuv(w, h, n);

    eprintln!("=== g.1 Call 1 (clean per-GOP encode) ===");
    let b1 = h264_encode_gop_framed_bits_auto(&yuv, w, h, n, opts, &frame_bits, &hhat_seed, &weights)
        .expect("encode call 1");
    eprintln!("=== g.1 Call 2 (same GOP, fresh encoder + #548 reset) ===");
    let b2 = h264_encode_gop_framed_bits_auto(&yuv, w, h, n, opts, &frame_bits, &hhat_seed, &weights)
        .expect("encode call 2");

    unsafe { std::env::remove_var("PHASM_USE_WIRE_ONLY") };
    unsafe { std::env::remove_var("PHASM_PERF_TRACE") };

    let first_diff = b1.iter().zip(b2.iter()).position(|(a, b)| a != b);
    eprintln!(
        "WV.6.g.1 RESULT — real CarPlane {w}x{h} x{n}f, two clean per-GOP encodes: \
         b1={} B, b2={} B, first_diff={:?} (None ⇒ byte-identical ⇒ DecisionCache deletable)",
        b1.len(), b2.len(), first_diff,
    );
    assert_eq!(
        b1, b2,
        "two clean per-GOP encodes of the same 1080p real GOP diverge \
         (first_diff={first_diff:?}) — the per-GOP encode is NOT bit-deterministic, so the \
         WV.6.g cache-deletion approach is unsound; rethink before building on it."
    );
}
