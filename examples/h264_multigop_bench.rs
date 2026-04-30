// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Multi-GOP encoder benchmark — encodes N frames at 1920×1072 with an
//! IDR every `GOP` frames. Hyperfine-friendly (silent stdout). Used by
//! task #72 to validate the SIMD speedup at multi-GOP workload scale.
//!
//! Usage:
//!   N=90 GOP=30 Q=80 YUV=/tmp/img4138_full_1072.yuv \
//!     target/release/examples/h264_multigop_bench
//!
//! Env knobs: N (default 90), GOP (default 30), Q (default 80),
//! YUV (default /tmp/img4138_full_1072.yuv), W/H (default 1920/1072).
//! PHASM_OUT_H264 (optional) — write the bitstream to this path.
//! PHASM_BENCH_PRINT — emit a short summary line on stderr.

use phasm_core::codec::h264::encoder::encoder::Encoder;

fn main() {
    let w: u32 = std::env::var("W").ok().and_then(|s| s.parse().ok()).unwrap_or(1920);
    let h: u32 = std::env::var("H").ok().and_then(|s| s.parse().ok()).unwrap_or(1072);
    let n_frames: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(90);
    let gop: usize = std::env::var("GOP").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let q: u8 = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let yuv_path = std::env::var("YUV")
        .unwrap_or_else(|_| "/tmp/img4138_full_1072.yuv".into());

    let frame_size = (w * h * 3 / 2) as usize;
    let pixels = std::fs::read(&yuv_path).expect("yuv missing");
    assert!(
        pixels.len() >= n_frames * frame_size,
        "yuv too short: {} bytes vs need {} ({}f)",
        pixels.len(),
        n_frames * frame_size,
        n_frames
    );

    let mut enc = Encoder::new(w, h, Some(q)).expect("encoder new");
    enc.set_gop_length(gop as u32);

    let mut bytes: Vec<u8> = Vec::with_capacity(64 * 1024 * 1024);

    for fi in 0..n_frames {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        let is_idr = fi % gop == 0;
        let out = if is_idr {
            enc.encode_i_frame(src).expect("encode_i_frame")
        } else {
            enc.encode_p_frame(src).expect("encode_p_frame")
        };
        bytes.extend_from_slice(&out);
    }

    if let Ok(path) = std::env::var("PHASM_OUT_H264") {
        std::fs::write(&path, &bytes).expect("write h264");
    }

    if std::env::var("PHASM_BENCH_PRINT").is_ok() {
        eprintln!(
            "bench {}x{} N={} GOP={} Q={} → {} bytes",
            w, h, n_frames, gop, q, bytes.len()
        );
    }
}
