// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Encoder-only benchmark harness — encodes N frames (1 I + N-1 P) at
//! 1920×1072 and discards the output. No ffmpeg, no reconstruction
//! comparison, no file writes on the hot path. Used with hyperfine for
//! wall-clock baseline numbers.
//!
//! Usage:
//!   N=10 YUV=/tmp/img4138_1080p_f10.yuv \
//!     target/release/examples/h264_nframe_bench
//!
//! Env knobs: N (default 10), YUV (default /tmp/img4138_1080p_f{N}.yuv),
//! Q (default 80), W/H (default 1920/1072).

use phasm_core::codec::h264::encoder::encoder::Encoder;

fn main() {
    let w: u32 = std::env::var("W").ok().and_then(|s| s.parse().ok()).unwrap_or(1920);
    let h: u32 = std::env::var("H").ok().and_then(|s| s.parse().ok()).unwrap_or(1072);
    let n_frames: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let q: u8 = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let yuv_path = std::env::var("YUV")
        .unwrap_or_else(|_| format!("/tmp/img4138_1080p_f{n_frames}.yuv"));

    let frame_size = (w * h * 3 / 2) as usize;
    let pixels = std::fs::read(&yuv_path).expect("yuv missing");
    assert!(
        pixels.len() >= n_frames * frame_size,
        "yuv too short: {} vs need {}",
        pixels.len(),
        n_frames * frame_size
    );

    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    let mut total_bytes: usize = enc.encode_i_frame(&pixels[..frame_size]).unwrap().len();
    for fi in 1..n_frames {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        total_bytes += enc.encode_p_frame(src).unwrap().len();
    }

    // Minimal end-of-run marker so hyperfine can't optimize the bench
    // away; hyperfine never reads this output when redirecting to
    // /dev/null, but the compiler needs to see `total_bytes` consumed
    // so the encode calls stay live.
    if std::env::var("PHASM_BENCH_PRINT").is_ok() {
        eprintln!("bench {w}x{h} N={n_frames} Q={q} → {total_bytes} bytes");
    }
}
