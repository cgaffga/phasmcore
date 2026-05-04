// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//! IBPBP ffmpeg compliance bisector. Encodes a sequence of frames in
//! IBPBP shape (I, P, B, P, B, ...) and pipes through ffmpeg. Reports
//! per-frame status + first divergence. Mirror of `h264_cabac_n_frame_test`
//! but exercises `encode_b_frame` between P frames.
//!
//! Usage:
//!   h264_ibpbp_diag <yuv> <w> <h> <n_display_frames> [q=75]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: {} <yuv> <w> <h> <n_display_frames> [q=75]", args[0]);
        std::process::exit(2);
    }
    let yuv = &args[1];
    let w: u32 = args[2].parse().unwrap();
    let h: u32 = args[3].parse().unwrap();
    let n: usize = args[4].parse().unwrap();
    let q: u8 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(75);

    let frame_size = (w * h * 3 / 2) as usize;
    let pixels = std::fs::read(yuv).expect("read yuv");
    assert!(pixels.len() >= n * frame_size, "yuv too short");

    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_b_frames = true;

    // Display order: I_0, B_1, P_2, B_3, P_4, B_5, P_6, ...
    // Encode order:  I_0, P_2, B_1, P_4, B_3, P_6, B_5, ...
    // The example reads display-order pixels at fi and dispatches to
    // encode_i / encode_p / encode_b in encode-order.
    let mut bytes = Vec::new();
    let mut frame_kinds: Vec<char> = Vec::with_capacity(n);
    // Encode I_0 first.
    bytes.extend_from_slice(&enc.encode_i_frame(&pixels[0..frame_size]).unwrap());
    frame_kinds.push('I');
    // Encode order for IBPBP: alternating P then B from display index 2 onwards.
    // Display fi=2 → P (anchor), fi=1 → B (between I_0 and P_2)
    // Display fi=4 → P (anchor), fi=3 → B (between P_2 and P_4)
    let mut fi = 2;
    while fi < n {
        let p_src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        bytes.extend_from_slice(&enc.encode_p_frame(p_src).unwrap());
        frame_kinds.push('P');
        // Now emit the B at fi-1 (between previous anchor and this P).
        let b_idx = fi - 1;
        if b_idx > 0 {
            let b_src = &pixels[b_idx * frame_size..(b_idx + 1) * frame_size];
            bytes.extend_from_slice(&enc.encode_b_frame(b_src).unwrap());
            frame_kinds.push('B');
        }
        fi += 2;
    }

    std::fs::write("/tmp/ibpbp_diag.h264", &bytes).unwrap();
    println!(
        "encoded {} NAL frames ({} I, {} P, {} B), {} bytes",
        frame_kinds.len(),
        frame_kinds.iter().filter(|k| **k == 'I').count(),
        frame_kinds.iter().filter(|k| **k == 'P').count(),
        frame_kinds.iter().filter(|k| **k == 'B').count(),
        bytes.len(),
    );

    let dec_out = Command::new("ffmpeg")
        .args([
            "-y", "-loglevel", "error", "-f", "h264",
            "-i", "/tmp/ibpbp_diag.h264",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "/tmp/ibpbp_diag_dec.yuv",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&dec_out.stderr);
    if !stderr.trim().is_empty() {
        eprintln!("---- ffmpeg stderr ----");
        for line in stderr.lines().take(20) {
            eprintln!("  {line}");
        }
        std::process::exit(1);
    }
    println!("ffmpeg accepted IBPBP stream cleanly");
}
