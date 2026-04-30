// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! H.264 stego pipeline benchmark — encodes a fixed message into a
//! fixed CAVLC MP4 cover. Used to measure the GOP-parallel speedup
//! from `PHASM_H264_PARALLEL_GOPS=N` (Phase I.1, commit d38c5c7).
//!
//! Usage:
//!   MP4=/tmp/i72_cavlc_90f.mp4 \
//!     target/release/examples/h264_stego_bench
//!
//! Env knobs: MP4 (default /tmp/i72_cavlc_90f.mp4), MSG (default 200-byte
//! payload), PASSPHRASE (default "bench"), PHASM_BENCH_PRINT (verbose).
//! Honors PHASM_H264_PARALLEL_GOPS for MT control.

use phasm_core::stego::video::h264_pipeline::h264_ghost_encode;

fn main() {
    let mp4_path = std::env::var("MP4")
        .unwrap_or_else(|_| "/tmp/i72_cavlc_90f.mp4".into());
    let mp4 = std::fs::read(&mp4_path).expect("read mp4");
    let msg = std::env::var("MSG").unwrap_or_else(|_| "x".repeat(200));
    let passphrase = std::env::var("PASSPHRASE").unwrap_or_else(|_| "bench".into());

    let stego = h264_ghost_encode(&mp4, &msg, &passphrase).expect("encode");

    if std::env::var("PHASM_BENCH_PRINT").is_ok() {
        eprintln!(
            "stego: {} bytes in / {} bytes out (msg {} bytes, parallel_gops={})",
            mp4.len(),
            stego.len(),
            msg.len(),
            std::env::var("PHASM_H264_PARALLEL_GOPS").unwrap_or_else(|_| "default(4)".into())
        );
    }
}
