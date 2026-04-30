// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Reports embedding capacity for an H.264 MP4 (path from arg[1] or
//! `MP4` env). Prints the public `h264_ghost_capacity` byte count and
//! the inferred raw `usable_count` (cover positions across all GOPs)
//! by reversing the capacity formula:
//!   capacity_bytes = (usable_count / 5 - FRAME_OVERHEAD * 8) / 8
//!
//! Used by task #72 to validate capacity at multi-GOP scale.

use phasm_core::stego::video::h264_pipeline::{h264_ghost_capacity, h264_ghost_capacity_max};

const FRAME_OVERHEAD: usize = 50;

fn main() {
    let mp4_path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("MP4").ok())
        .expect("usage: h264_capacity_report <path.mp4>  (or MP4=path env)");
    let mp4 = std::fs::read(&mp4_path).expect("read mp4");
    println!("MP4 size: {} bytes ({:.2} MB)", mp4.len(), mp4.len() as f64 / 1_048_576.0);

    let capacity_bytes = h264_ghost_capacity(&mp4).expect("capacity");
    println!("h264_ghost_capacity     (UX-safe, w=5):     {:>7} bytes  ({} bits)",
        capacity_bytes, capacity_bytes * 8);

    let capacity_max = h264_ghost_capacity_max(&mp4).expect("capacity_max");
    println!("h264_ghost_capacity_max (max, w=1):         {:>7} bytes  ({} bits)",
        capacity_max, capacity_max * 8);

    // Reverse capacity_max formula:
    //   payload_bytes = (usable_count - FRAME_OVERHEAD*8) / 8
    //   -> usable_count = payload_bytes*8 + FRAME_OVERHEAD*8
    let usable_count = capacity_max * 8 + FRAME_OVERHEAD * 8;
    println!();
    println!("usable_count (cover positions across all GOPs): {}", usable_count);
    println!("(Encoder picks largest w in 1..=10 that fits: small msg -> w=10, max msg -> w=1)");

    // For STC w selection at encode: the encoder picks the largest w in
    // 1..=10 such that m_g_c * w fits in n_coeff and m_g_m * w fits in
    // n_mvd for every GOP. For a SHORT message (e.g. 100-byte payload
    // = 800 bits + 400 overhead = 1200 bits / 3 GOPs = 400 bits/gop)
    // and ~10k cover bits per GOP, w_max = floor(10k / 400) = 25, but
    // capped at 10 → encoder picks w=10. See h264_pipeline.rs:741-743.
}
