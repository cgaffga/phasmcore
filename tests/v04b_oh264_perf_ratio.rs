// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// V0.4.B — OH264 stego encode benchmark (hand-run, NOT a CI gate).
//
// Prints the wall-clock cost of the full 4-domain streaming stego encode
// vs a bare (no-stego) OH264 encode on the same fixture + host. This is a
// BENCHMARK, not an assertion gate: both tests are `#[ignore]`'d and run
// only on demand —
//
//   cargo test --release -p phasm-core --features h264-encoder \
//     --test v04b_oh264_perf_ratio -- --ignored --nocapture
//
// History (#832): this file once asserted stego/clean ≤ 10×. That ratio
// was calibrated for the retired single-domain one-shot encoder; the
// production 4-domain streaming path is legitimately multi-pass (Pass-1
// cover walk → STC plan → Pass-2 emit + per-GOP roundtrip-verify), so the
// ratio is ~24-38× *by design*. The ceiling was meaningless, and the test
// ran in no CI lane regardless (the only release lanes invoke specific
// named tests). The assertion was dropped; the measurement remains.
//
// Measured baseline (Apple M-series, release, 480p × 8, qp=22):
//   synthetic : clean ≈ 15 ms, stego ≈ 0.56 s  (~38×)
//   carplane  : clean ≈ 17 ms, stego ≈ 0.40 s  (~24×)
// A debug build runs the stego encode ~25× slower (~16 s) — always
// measure with --release. The production figure that actually matters is
// 1080p × 30 ≈ 1.3 s (CLAUDE.md); add a fixture here to track it.

#![cfg(feature = "h264-encoder")]

mod common;
use common::oh264_stream;

use phasm_core::codec::h264::openh264::{set_frame_num, Encoder};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn try_scale_yuv(source_mp4: &str, w: u32, h: u32, n_frames: u32) -> Option<Vec<u8>> {
    let src = corpus_root().join(source_mp4);
    if !src.exists() {
        return None;
    }
    let yuv_path = format!("/tmp/phasm_v04b_perf_{}x{}_f{}.yuv", w, h, n_frames);
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * (n_frames as usize);
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need {
            return Some(data);
        }
    }
    let vf = format!("scale={}:{}", w, h);
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &n_frames.to_string()])
        .args(["-an", "-vf", &vf])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    std::fs::read(&yuv_path).ok()
}

fn synth_yuv(width: u32, height: u32, n_frames: u32) -> Vec<u8> {
    // Same structured gradient + checkerboard pattern that
    // openh264_cross_arch_determinism.rs uses successfully. Pure-noise
    // YUV trips OH264's content-adaptive limits at small frame counts.
    let w = width as usize;
    let h = height as usize;
    let n = n_frames as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    let frame_size = y_size + 2 * uv_size;
    let mut buf = vec![0u8; frame_size * n];
    for f in 0..n {
        let off = f * frame_size;
        for y in 0..h {
            for x in 0..w {
                let v = (((x + y + f * 8) & 0xff) ^ ((x ^ y) & 0x3f)) as u8;
                buf[off + y * w + x] = v;
            }
        }
        for y in 0..(h / 2) {
            for x in 0..(w / 2) {
                buf[off + y_size + y * (w / 2) + x] =
                    (128 + ((x as i32 - w as i32 / 4) / 8)) as u8;
            }
        }
        for y in 0..(h / 2) {
            for x in 0..(w / 2) {
                buf[off + y_size + uv_size + y * (w / 2) + x] =
                    (128 + ((y as i32 - h as i32 / 4) / 8)) as u8;
            }
        }
    }
    buf
}

fn bench_oh264_clean(yuv: &[u8], w: u32, h: u32, n_frames: u32, qp: i32, intra: i32) -> std::time::Duration {
    let frame_y = (w * h) as usize;
    let frame_uv = (w * h / 4) as usize;
    let frame_total = frame_y + 2 * frame_uv;
    let mut out = vec![0u8; 8 * 1024 * 1024];
    let mut enc = Encoder::new(w as i32, h as i32, qp, intra).expect("clean enc create");
    let t0 = Instant::now();
    for frame in 0..n_frames {
        set_frame_num(frame);
        let base = (frame as usize) * frame_total;
        enc.encode_frame(
            &yuv[base..base + frame_y],
            &yuv[base + frame_y..base + frame_y + frame_uv],
            &yuv[base + frame_y + frame_uv..base + frame_total],
            (frame as i64) * 33,
            &mut out,
        )
        .expect("clean encode");
    }
    t0.elapsed()
}

fn bench_oh264_stego_full(yuv: &[u8], w: u32, h: u32, n_frames: u32, qp: i32, _intra: i32) -> std::time::Duration {
    let t0 = Instant::now();
    let stego = oh264_stream::encode(
        yuv, w, h, n_frames, qp,
        "v04b perf-ratio",
        "v04b-perf-pass",
    ).expect("stego encode");
    let dt = t0.elapsed();
    // Sanity: round-trip works at this perf point.
    let recovered = oh264_stream::decode_text(&stego, "v04b-perf-pass")
        .expect("stego decode");
    assert_eq!(recovered, "v04b perf-ratio");
    dt
}

fn run_perf_bench(yuv: &[u8], w: u32, h: u32, n_frames: u32, qp: i32, label: &str) {
    const INTRA: i32 = 60;

    // Warm up first to avoid first-encode allocation dominating short runs.
    let _warmup_clean = bench_oh264_clean(yuv, w, h, n_frames, qp, INTRA);

    let clean_dt = bench_oh264_clean(yuv, w, h, n_frames, qp, INTRA);
    let stego_dt = bench_oh264_stego_full(yuv, w, h, n_frames, qp, INTRA);

    // Benchmark only — NO assertion. The 4-domain streaming path is
    // multi-pass, so ~24-38× clean is expected (see the module header). The
    // round-trip inside `bench_oh264_stego_full` is the only correctness check.
    let ratio = stego_dt.as_secs_f64() / clean_dt.as_secs_f64().max(1e-6);
    eprintln!(
        "V0.4.B perf-bench [{label}] {w}×{h}×{n_frames} qp={qp}: \
         clean={:.1} ms, stego_full={:.1} ms, ratio={ratio:.2}× (no gate)",
        clean_dt.as_secs_f64() * 1000.0,
        stego_dt.as_secs_f64() * 1000.0,
    );
}

/// Synthetic-fixture benchmark. No corpus dependency.
#[test]
#[ignore = "hand-run benchmark, NOT a CI gate — prints stego-vs-clean encode \
            wall-time (the old ≤10× ratio assert was invalid for the multi-pass \
            4-domain path + ran in no CI lane; #832). Run: cargo test --release \
            --features h264-encoder --test v04b_oh264_perf_ratio -- --ignored \
            --nocapture"]
fn v04b_perf_bench_synthetic_480p() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let yuv = synth_yuv(480, 272, 8);
    run_perf_bench(&yuv, 480, 272, 8, /*qp=*/ 22, "synthetic");
}

/// Real-world fixture benchmark (carplane, gitignored). Skips if absent.
#[test]
#[ignore = "hand-run benchmark, NOT a CI gate — see v04b_perf_bench_synthetic_480p"]
fn v04b_perf_bench_carplane_480p() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let Some(yuv) = try_scale_yuv("Artlist_CarPlane.mp4", 480, 272, 8) else {
        eprintln!("V0.4.B perf-bench carplane SKIP (corpus fixture absent)");
        return;
    };
    run_perf_bench(&yuv, 480, 272, 8, /*qp=*/ 22, "carplane");
}
