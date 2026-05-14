// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.2.3 (#404) — cross-encoder PSNR delta gate.
//
// For the same source YUV, encode via both backends:
//   - OpenH264 backend (production v1.0 path)
//   - Pure-Rust 4-domain CABAC (EXPERIMENTAL path, used for the mobile
//     HUD "Experimental encoder" toggle in D.0.2/D.0.3)
//
// Decode each via ffmpeg, measure Y-PSNR against source, assert:
//   1. Each backend's PSNR is above the absolute floor (no regression
//      in either path).
//   2. The PSNR delta between backends is bounded (users switching
//      backends don't see wildly different visual quality).
//
// The two encoders emit bitstream-distinct stego (per CLAUDE.md:
// "different cover sets, different CABAC override domains") so a
// byte-exact comparison is meaningless. PSNR-as-measured-against-the-
// source is the right cross-encoder metric.
//
// `#[ignore]` because pure-Rust at 480p × 8 frames takes ~100 s — too
// slow for default CI, fine for nightly / pre-tag validation.
//
// Run:
//   cargo test --release --features "h264-encoder openh264-backend" \
//     --test openh264_psnr_delta_vs_purerust -- --ignored --nocapture

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend", feature = "cabac-stego"))]

use phasm_core::codec::h264::openh264_stego::{
    openh264_stego_encode_yuv_text, EncodeOpts,
};
use phasm_core::h264_stego_encode_yuv_string_4domain_multigop_streaming_v2;
use std::path::PathBuf;

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn ensure_yuv(source: &str, w: u32, h: u32, n_frames: u32) -> Vec<u8> {
    let yuv_path = format!("/tmp/phasm_oh264_delta_{}_{}x{}_f{}.yuv",
        source.replace(['.', '/'], "_"), w, h, n_frames);
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * n_frames as usize;
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need { return data; }
    }
    let src = corpus_root().join(source);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!("scale={w}:{h}");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &n_frames.to_string(), "-an", "-vf", &vf,
               "-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("ffmpeg launch");
    assert!(status.success(), "ffmpeg scale failed for {source}");
    std::fs::read(&yuv_path).expect("read yuv")
}

fn psnr_y(a: &[u8], b: &[u8]) -> f64 {
    let mut mse = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        mse += d * d;
    }
    mse /= a.len() as f64;
    if mse == 0.0 { 99.0 } else { 10.0 * (255.0_f64 * 255.0 / mse).log10() }
}

/// Decode Annex-B stego via ffmpeg subprocess → raw YUV420p frames.
/// Asserts the decode succeeded with zero concealment events (any
/// concealment means the stego bitstream is broken — gate fails before
/// PSNR is even computed).
fn decode_via_ffmpeg(stego: &[u8], tag: &str, expected_size: usize) -> Vec<u8> {
    let h264_path = std::env::temp_dir().join(format!("phasm_delta_{tag}.h264"));
    let dec_path = std::env::temp_dir().join(format!("phasm_delta_{tag}.yuv"));
    let log_path = std::env::temp_dir().join(format!("phasm_delta_{tag}.log"));
    std::fs::write(&h264_path, stego).expect("write annex-b");
    let log_file = std::fs::File::create(&log_path).expect("create log");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "info", "-framerate", "30", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .stderr(log_file)
        .status()
        .expect("ffmpeg decode");
    assert!(status.success(), "ffmpeg decode failed for {tag}");
    let log = std::fs::read_to_string(&log_path).expect("read log");
    let n_conceal = log.lines().filter(|l| l.contains("concealing")).count();
    assert_eq!(
        n_conceal, 0,
        "[{tag}] ffmpeg reported {n_conceal} concealment events — stego bitstream broken",
    );
    let decoded = std::fs::read(&dec_path).expect("read decoded");
    assert_eq!(decoded.len(), expected_size,
        "[{tag}] decoded size mismatch: got {}, expected {}", decoded.len(), expected_size);
    decoded
}

fn measure_psnr(yuv: &[u8], decoded: &[u8], w: u32, h: u32, n_frames: usize) -> (f64, f64) {
    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w * h) as usize;
    let mut psnrs = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let off = f * frame_size;
        psnrs.push(psnr_y(&yuv[off..off + y_size], &decoded[off..off + y_size]));
    }
    let min = psnrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg = psnrs.iter().sum::<f64>() / n_frames as f64;
    (avg, min)
}

/// Whole-frame Y-PSNR floor for each backend. OH264 hits ~38-42 dB on
/// 480p × 8 frames at QP=26; pure-Rust hits ~30-36 dB. Floor 25 dB
/// catches gross regression in either backend.
const ABSOLUTE_PSNR_FLOOR_DB: f64 = 25.0;

/// Maximum allowed |PSNR_OH264 − PSNR_pure| (avg over frames). The
/// two encoders make independent cover-set decisions so they won't
/// agree byte-for-byte; both should land within 6 dB of each other
/// on the same source. Larger deltas suggest one backend has a real
/// quality regression vs the other.
const PSNR_DELTA_CEILING_DB: f64 = 6.0;

/// Phase C.2.3 cross-encoder PSNR delta gate.
///
/// Encodes the iPhone7 fixture at 480p × 8 frames via both backends,
/// compares each to source via ffmpeg-decoded Y-PSNR, asserts:
///   - Each backend ≥ ABSOLUTE_PSNR_FLOOR_DB (no individual regression)
///   - |delta| ≤ PSNR_DELTA_CEILING_DB (backends comparable in quality)
#[test]
#[ignore]
fn cross_encoder_psnr_delta_iphone7_480p() {
    const W: u32 = 480;
    const H: u32 = 272;
    const N: u32 = 8;
    const GOP: usize = 4;

    let yuv = ensure_yuv("IMG_4138.MOV", W, H, N);
    let frame_size = (W * H * 3 / 2) as usize;
    let expected = frame_size * N as usize;

    // --- OpenH264 backend ---
    let t0 = std::time::Instant::now();
    let stego_oh264 = openh264_stego_encode_yuv_text(
        &yuv, W, H, N,
        EncodeOpts { qp: 26, intra_period: GOP as i32 },
        "C.2.3 delta gate",
        "delta-pass",
    ).expect("oh264 encode");
    let t_oh264 = t0.elapsed().as_secs_f64();

    let dec_oh264 = decode_via_ffmpeg(&stego_oh264, "oh264", expected);
    let (psnr_oh264_avg, psnr_oh264_min) = measure_psnr(&yuv, &dec_oh264, W, H, N as usize);

    // --- Pure-Rust 4-domain CABAC backend ---
    let t0 = std::time::Instant::now();
    let stego_pure = h264_stego_encode_yuv_string_4domain_multigop_streaming_v2(
        &yuv, W, H, N as usize, GOP,
        "C.2.3 delta gate",
        "delta-pass",
    ).expect("pure-rust encode");
    let t_pure = t0.elapsed().as_secs_f64();

    let dec_pure = decode_via_ffmpeg(&stego_pure, "pure", expected);
    let (psnr_pure_avg, psnr_pure_min) = measure_psnr(&yuv, &dec_pure, W, H, N as usize);

    let delta_avg = (psnr_oh264_avg - psnr_pure_avg).abs();

    eprintln!(
        "OH264   : Y-PSNR avg={:.2} min={:.2} dB  enc={:.2}s  ({} KB)",
        psnr_oh264_avg, psnr_oh264_min, t_oh264, stego_oh264.len() / 1024,
    );
    eprintln!(
        "pure-Rust: Y-PSNR avg={:.2} min={:.2} dB  enc={:.2}s  ({} KB)  [{:.0}× slower]",
        psnr_pure_avg, psnr_pure_min, t_pure, stego_pure.len() / 1024,
        t_pure / t_oh264,
    );
    eprintln!("|Δ avg-PSNR| = {:.2} dB  (ceiling {:.1} dB)", delta_avg, PSNR_DELTA_CEILING_DB);

    assert!(
        psnr_oh264_min >= ABSOLUTE_PSNR_FLOOR_DB,
        "OH264 PSNR regression: min {:.2} dB < floor {:.1} dB",
        psnr_oh264_min, ABSOLUTE_PSNR_FLOOR_DB,
    );
    assert!(
        psnr_pure_min >= ABSOLUTE_PSNR_FLOOR_DB,
        "pure-Rust PSNR regression: min {:.2} dB < floor {:.1} dB",
        psnr_pure_min, ABSOLUTE_PSNR_FLOOR_DB,
    );
    assert!(
        delta_avg <= PSNR_DELTA_CEILING_DB,
        "cross-encoder PSNR delta {:.2} dB > ceiling {:.1} dB \
         (OH264 avg={:.2}, pure-Rust avg={:.2}) — one backend has \
         a real quality regression vs the other",
        delta_avg, PSNR_DELTA_CEILING_DB, psnr_oh264_avg, psnr_pure_avg,
    );
}
