// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 6F.3 — §30D-C orchestrator wall-clock benchmark.
//!
//! Encodes an N-frame YUV through the full 3-pass stego pipeline
//! (`h264_stego_encode_yuv_string_4domain_multigop`) at the
//! current HEAD's bitstream-mod MVD + stealth-weighted allocator
//! configuration. Used to compare against the Phase I.benchmark
//! encoder-only baseline (`h264_nframe_bench`).
//!
//! Build with `--features cabac-stego` (without it the example
//! compiles to a stub `main` that just prints a message — required
//! so cargo doesn't fail the public-mirror's
//! `--features video,h264-encoder --examples` build).
//!
//! Usage:
//!   N=10 YUV=/tmp/img4138_1080p_f10.yuv \
//!     target/release/examples/h264_stego_4domain_bench
//!
//! Env knobs: N (default 10), YUV, MSG (default "x" * 8 bytes),
//! PASSPHRASE (default "bench").

#[cfg(not(feature = "cabac-stego"))]
fn main() {
    eprintln!(
        "h264_stego_4domain_bench requires --features cabac-stego"
    );
}

#[cfg(feature = "cabac-stego")]
use phasm_core::h264_stego_encode_yuv_string_4domain_multigop;

#[cfg(feature = "cabac-stego")]
fn main() {
    let w: u32 = std::env::var("W").ok().and_then(|s| s.parse().ok()).unwrap_or(1920);
    let h: u32 = std::env::var("H").ok().and_then(|s| s.parse().ok()).unwrap_or(1072);
    let n_frames: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let gop: usize = std::env::var("GOP").ok().and_then(|s| s.parse().ok()).unwrap_or(n_frames);
    let yuv_path = std::env::var("YUV")
        .unwrap_or_else(|_| format!("/tmp/img4138_1080p_f{n_frames}.yuv"));
    let msg = std::env::var("MSG").unwrap_or_else(|_| "x".repeat(8));
    let passphrase = std::env::var("PASSPHRASE").unwrap_or_else(|_| "bench".into());

    let frame_size = (w * h * 3 / 2) as usize;
    let yuv = std::fs::read(&yuv_path).expect("yuv missing");
    assert!(
        yuv.len() >= n_frames * frame_size,
        "yuv too short: {} vs need {}",
        yuv.len(),
        n_frames * frame_size,
    );
    let yuv_slice = &yuv[..n_frames * frame_size];

    let stego = h264_stego_encode_yuv_string_4domain_multigop(
        yuv_slice, w, h, n_frames, gop, &msg, &passphrase,
    ).expect("stego encode");

    if std::env::var("PHASM_BENCH_PRINT").is_ok() {
        eprintln!(
            "stego(4domain): {}x{} N={} gop={} msg={}B → {} bytes",
            w, h, n_frames, gop, msg.len(), stego.len(),
        );
    }
}
