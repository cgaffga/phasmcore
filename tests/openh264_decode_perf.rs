// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.5.2 (#412) — decode wall-clock benchmark on OH264 backend.
//
// Companions to C.5.1 (`openh264_perf_benchmark.rs`, encode perf).
// Measures the two relevant decode paths:
//
//   1. PRODUCTION: `extract_cover_bits_via_decoder` — full decode +
//      stego-hook fires, captures the cover-bit stream. This is what
//      `phasm decode` (CLI), iOS bridge, and Android bridge actually
//      invoke. It's the wall-clock cost a user pays at decode time.
//
//   2. PARSE-ONLY: `Decoder::new_parse_only` + `decode_frame` loop —
//      metadata-only fast path. ffmpeg-compatible SPS / PPS / slice-
//      header parse, NO residual decode. Per B.9.2.6 (#378), stego
//      hooks DO NOT fire on this path — `SParserBsInfo` early-out at
//      `decoder_core.cpp:89` exits before residual. So this path
//      cannot be used for stego extraction; it's measured here as a
//      reference for the cost of pure-parse work (e.g. SPS dim probe).
//
// The ratio between the two reveals how much of the full decode is
// CABAC residual + IDCT + recon work that stego extraction requires.
//
// Smoke variant runs in default lane (~0.3s); production fixtures
// `#[ignore]`'d to keep the default suite fast.
//
// Run full corpus:
//   cargo test --release --features "h264-encoder openh264-backend" \
//     --test openh264_decode_perf -- --ignored --nocapture

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use phasm_core::codec::h264::openh264::{extract_cover_bits_via_decoder, Decoder};
use phasm_core::codec::h264::openh264_stego::{
    openh264_stego_encode_yuv_text, EncodeOpts,
};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

/// Local copy of `openh264.rs::split_annex_b_per_frame` so the test
/// doesn't widen the crate's public API for a test-only need. Splits
/// an Annex-B stream at slice-NAL boundaries (NAL types 1 / 5); SPS /
/// PPS NALs accumulate into the next chunk. Bit-for-bit matches the
/// internal helper used by `extract_cover_bits_via_decoder`.
fn split_annex_b_per_frame(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let mut sc_len = 0usize;
        if i + 3 < bytes.len() && bytes[i..i + 4] == [0, 0, 0, 1] {
            sc_len = 4;
        } else if i + 2 < bytes.len() && bytes[i..i + 3] == [0, 0, 1] {
            sc_len = 3;
        }
        if sc_len > 0 && i + sc_len < bytes.len() {
            let nal_type = bytes[i + sc_len] & 0x1F;
            if (nal_type == 1 || nal_type == 5) && !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
        }
        current.push(bytes[i]);
        i += 1;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn ensure_yuv(tag: &str, src: &str, w: u32, h: u32, n_frames: u32) -> Vec<u8> {
    let yuv_path = format!("/tmp/openh264_decperf_{tag}_{w}x{h}_f{n_frames}.yuv");
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * n_frames as usize;
    if let Ok(d) = std::fs::read(&yuv_path) {
        if d.len() >= need { return d; }
    }
    let src_path = corpus_root().join(src);
    assert!(src_path.exists(), "corpus fixture missing: {}", src_path.display());
    let vf = format!("scale={w}:{h},format=yuv420p");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src_path)
        .args(["-frames:v", &n_frames.to_string(),
               "-vf", &vf, "-f", "rawvideo"])
        .arg(&yuv_path)
        .status()
        .expect("ffmpeg launch");
    assert!(status.success(), "ffmpeg scale failed for {src}");
    std::fs::read(&yuv_path).expect("read yuv")
}

/// Time `extract_cover_bits_via_decoder` over `runs` invocations,
/// return the median. Median (not mean) is more robust to occasional
/// CPU-scheduling outliers in CI environments.
fn time_extract(annex_b: &[u8], runs: usize) -> Duration {
    let mut samples: Vec<Duration> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let _bits = extract_cover_bits_via_decoder(annex_b).expect("extract");
        samples.push(t.elapsed());
    }
    samples.sort();
    samples[runs / 2]
}

/// Time the parse-only fast path. Mirrors what
/// `extract_cover_bits_via_decoder` does (per-frame NAL groups via
/// `split_annex_b_per_frame`) but uses `Decoder::new_parse_only`
/// instead of `Decoder::new`. No hooks fire on this path.
fn time_parse_only(annex_b: &[u8], runs: usize) -> Duration {
    let mut samples: Vec<Duration> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let mut dec = Decoder::new_parse_only().expect("parse-only Decoder::new_parse_only");
        for nal_group in split_annex_b_per_frame(annex_b) {
            let _ = dec.decode_frame(&nal_group);
        }
        samples.push(t.elapsed());
    }
    samples.sort();
    samples[runs / 2]
}

fn report(label: &str, dt: Duration, n_frames: u32) {
    let ms = dt.as_secs_f64() * 1000.0;
    let per_frame_ms = ms / n_frames as f64;
    let fps = (n_frames as f64) / dt.as_secs_f64();
    eprintln!("  {:32} {:>8.2} ms total  {:>6.2} ms/frame  {:>7.1} fps",
              label, ms, per_frame_ms, fps);
}

fn run_decode_bench(tag: &str, src: &str, w: u32, h: u32, n_frames: u32, qp: i32, runs: usize) {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());

    let yuv = ensure_yuv(tag, src, w, h, n_frames);
    let stego = openh264_stego_encode_yuv_text(
        &yuv, w, h, n_frames,
        EncodeOpts { qp, intra_period: 60 },
        "decode perf bench",
        "perf-pass",
    ).expect("oh264 stego encode");

    eprintln!("\n[{tag}] {w}×{h}×{n_frames} stego={} KB  decode median over {runs} runs:",
        stego.len() / 1024);

    let t_full = time_extract(&stego, runs);
    report("extract_cover_bits_via_decoder", t_full, n_frames);

    let t_parse = time_parse_only(&stego, runs);
    report("Decoder::new_parse_only loop", t_parse, n_frames);

    let speedup = t_full.as_secs_f64() / t_parse.as_secs_f64();
    eprintln!("  parse-only is {speedup:.2}× faster than stego-extracting full decode");

    // Sanity assertion: parse-only must be faster than full decode
    // (it skips CABAC residual + IDCT + recon). On any sensible
    // architecture the ratio is well above 1×. If this ever fires
    // it means either the parse-only fast-path regressed, or
    // extract_cover_bits_via_decoder somehow got faster than parse
    // — both warrant investigation.
    assert!(
        t_parse < t_full,
        "[{tag}] parse-only ({:.1} ms) is not faster than full extract ({:.1} ms)",
        t_parse.as_secs_f64() * 1000.0,
        t_full.as_secs_f64() * 1000.0,
    );
}

// ─────────────────────────── smoke ────────────────────────────────────

#[test]
fn c5_2_decode_perf_smoke_480p_10f() {
    run_decode_bench("smoke_480p", "IMG_4138.MOV", 480, 272, 10, 22, /*runs=*/ 3);
}

// ─────────────────────────── production ───────────────────────────────

#[test]
#[ignore]
fn c5_2_decode_perf_1080p_30f() {
    run_decode_bench("prod_1080p", "IMG_4138.MOV", 1920, 1072, 30, 26, /*runs=*/ 5);
}

#[test]
#[ignore]
fn c5_2_decode_perf_carplane_1080p_30f() {
    run_decode_bench("carplane_1080p", "Artlist_CarPlane.mp4", 1072, 1920, 30, 26, /*runs=*/ 5);
}

#[test]
#[ignore]
fn c5_2_decode_perf_720p_16f() {
    run_decode_bench("mid_720p", "IMG_4138.MOV", 1280, 720, 16, 22, /*runs=*/ 5);
}

/// Phase C.5.3 (#413) — mobile-relevant decode bench: 480p × 60f.
/// Companion to C.5.3 encode bench in `openh264_perf_benchmark.rs`.
/// 480×272 × 60 frames approximates a typical 2 s mobile share-clip.
#[test]
#[ignore]
fn c5_3_decode_perf_mobile_480p_60f() {
    run_decode_bench("mobile_480p_60f", "IMG_4138.MOV", 480, 272, 60, 24, /*runs=*/ 5);
}
