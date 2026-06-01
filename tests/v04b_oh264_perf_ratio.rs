// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// V0.4.B — OH264 stego perf-RATIO budget (live lane).
//
// Complements the informational `c814_perf_smoke_480p_10f` benchmark
// in `openh264_perf_benchmark.rs` by adding an actual assertion: the
// full stego pipeline must encode within a generous multiple of clean
// (no-stego) OH264 encoding on the same fixture + same host.
//
// **Why ratio, not absolute wall-time**: absolute milliseconds are
// host-dependent (an M1 Pro vs a 2-core CI runner differ by 5-10×).
// The ratio between two encodes ON THE SAME HOST is far more
// portable — both pay the same per-frame OH264 cost, the ratio
// reflects only the stego pipeline's incremental work.
//
// **Threshold**: ≤ 10× clean. The c814 1080p production gate uses
// ≤ 1.5× clean for "passive overhead" (visual_recon only). The full
// stego path includes Pass 1 cover walk + STC plan + Pass 2/3 emit,
// which structurally multiplies the encode cost. Empirical 480p × 8
// numbers should fit comfortably under 10× on any host where the
// clean encode itself completes in less than a second.
//
// This is a regression gate, not a tightening target. If we ever
// optimize the stego pipeline to 3-5× clean, tighten the assertion.

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use phasm_core::codec::h264::openh264::{set_frame_num, Encoder};
use phasm_core::codec::h264::openh264_stego::{
    openh264_stego_decode_yuv_string, openh264_stego_encode_yuv_text, EncodeOpts,
};
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

fn bench_oh264_stego_full(yuv: &[u8], w: u32, h: u32, n_frames: u32, qp: i32, intra: i32) -> std::time::Duration {
    let opts = EncodeOpts { qp, intra_period: intra };
    let t0 = Instant::now();
    let stego = openh264_stego_encode_yuv_text(
        yuv, w, h, n_frames, opts,
        "v04b perf-ratio",
        "v04b-perf-pass",
    ).expect("stego encode");
    let dt = t0.elapsed();
    // Sanity: round-trip works at this perf point.
    let recovered = openh264_stego_decode_yuv_string(&stego, "v04b-perf-pass")
        .expect("stego decode");
    assert_eq!(recovered, "v04b perf-ratio");
    dt
}

fn run_ratio_check(yuv: &[u8], w: u32, h: u32, n_frames: u32, qp: i32, label: &str) {
    const INTRA: i32 = 60;

    // Warm up first to avoid first-encode allocation dominating short runs.
    let _warmup_clean = bench_oh264_clean(yuv, w, h, n_frames, qp, INTRA);

    let clean_dt = bench_oh264_clean(yuv, w, h, n_frames, qp, INTRA);
    let stego_dt = bench_oh264_stego_full(yuv, w, h, n_frames, qp, INTRA);

    let ratio = stego_dt.as_secs_f64() / clean_dt.as_secs_f64().max(1e-6);
    eprintln!(
        "V0.4.B perf-ratio [{label}] {w}×{h}×{n_frames} qp={qp}: \
         clean={:.1} ms, stego_full={:.1} ms, ratio={ratio:.2}×",
        clean_dt.as_secs_f64() * 1000.0,
        stego_dt.as_secs_f64() * 1000.0,
    );

    // Generous threshold: 10× clean. Realistic measured ratio for the
    // full stego pipeline (Pass 1 + STC + Pass 2/3 emit) at 480p × 8
    // is around 4-7× on M1. The 10× ceiling is what catches a 2× perf
    // regression while letting CI host noise pass.
    const RATIO_CEILING: f64 = 10.0;
    assert!(
        ratio <= RATIO_CEILING,
        "V0.4.B perf-ratio [{label}]: stego/clean = {ratio:.2}× exceeds {RATIO_CEILING:.1}× ceiling. \
         Suspect a stego-path perf regression."
    );
}

/// Live-lane gate using a synthetic fixture. Always runs; no corpus
/// dependency.
#[test]
fn v04b_perf_ratio_synthetic_480p() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let yuv = synth_yuv(480, 272, 8);
    run_ratio_check(&yuv, 480, 272, 8, /*qp=*/ 22, "synthetic");
}

/// Real-world fixture gate (carplane, gitignored). Skips if absent.
#[test]
fn v04b_perf_ratio_carplane_480p() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let Some(yuv) = try_scale_yuv("Artlist_CarPlane.mp4", 480, 272, 8) else {
        eprintln!("V0.4.B perf-ratio carplane SKIP (corpus fixture absent)");
        return;
    };
    run_ratio_check(&yuv, 480, 272, 8, /*qp=*/ 22, "carplane");
}
