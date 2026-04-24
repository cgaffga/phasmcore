// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
//! CABAC N-frame enc-vs-ffmpeg-dec check at arbitrary resolution, with
//! optional GOP size to exercise multi-GOP IDR refresh.

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: {} <yuv> <w> <h> <n_frames> [q=80] [gop_size=inf]",
            args[0]
        );
        std::process::exit(2);
    }
    let yuv = &args[1];
    let w: u32 = args[2].parse().unwrap();
    let h: u32 = args[3].parse().unwrap();
    let n: usize = args[4].parse().unwrap();
    let q: u8 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(80);
    let gop: usize = args
        .get(6)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let frame_size = (w * h * 3 / 2) as usize;
    let pixels = std::fs::read(yuv).expect("read yuv");
    assert!(pixels.len() >= n * frame_size, "yuv too short");

    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    // Phase 100-E diagnostic: set PHASM_TRANSFORM_8X8=1 to emit High
    // profile SPS/PPS + per-MB transform_size_8x8_flag. At this phase
    // the flag is always 0 so the bitstream semantics stay 4×4; the
    // test just exercises the new emit path.
    if std::env::var_os("PHASM_TRANSFORM_8X8").is_some() {
        enc.enable_transform_8x8 = true;
    }

    let mut bytes = Vec::new();
    let mut recons = Vec::with_capacity(n);
    let mut frame_kinds = Vec::with_capacity(n);

    for fi in 0..n {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        // Force an IDR at frame 0 and at every gop-th frame.
        let is_i = fi == 0 || fi % gop == 0;
        let nal = if is_i {
            enc.encode_i_frame(src).unwrap()
        } else {
            enc.encode_p_frame(src).unwrap()
        };
        bytes.extend_from_slice(&nal);
        recons.push(enc.recon.y.clone());
        frame_kinds.push(if is_i { 'I' } else { 'P' });
    }

    std::fs::write("/tmp/drift_n.h264", &bytes).unwrap();
    let dec_out = Command::new("ffmpeg")
        .args([
            "-y", "-loglevel", "warning", "-f", "h264", "-i", "/tmp/drift_n.h264",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "/tmp/drift_n_dec.yuv",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&dec_out.stderr);
    if !stderr.trim().is_empty() {
        eprintln!(
            "ffmpeg stderr (first 5 lines):\n{}",
            stderr.lines().take(5).collect::<Vec<_>>().join("\n")
        );
    }
    let decoded = std::fs::read("/tmp/drift_n_dec.yuv").unwrap_or_default();
    let y_size = (w * h) as usize;

    println!(
        "{} frames ({} I, {} P), {} bytes, {:.2} Mbps@30",
        n,
        frame_kinds.iter().filter(|k| **k == 'I').count(),
        frame_kinds.iter().filter(|k| **k == 'P').count(),
        bytes.len(),
        bytes.len() as f64 * 8.0 * 30.0 / n as f64 / 1e6
    );

    let mut n_clean = 0;
    let mut first_bad: Option<usize> = None;
    let n_decoded = decoded.len() / frame_size;
    for fi in 0..n.min(n_decoded) {
        let ey = &recons[fi];
        let dy = &decoded[fi * frame_size..fi * frame_size + y_size];
        let diff: usize = ey.iter().zip(dy.iter()).filter(|(a, b)| a != b).count();
        let pct = diff as f64 * 100.0 / y_size as f64;
        if diff == 0 {
            n_clean += 1;
            if fi < 3 || fi == n - 1 || (fi % 10 == 0 && fi < 100) {
                println!("  frame {:3} [{}]: 0 diff ✓", fi, frame_kinds[fi]);
            }
        } else {
            if first_bad.is_none() {
                first_bad = Some(fi);
            }
            println!(
                "  frame {:3} [{}]: {} diff ({:.4}%) ✗",
                fi, frame_kinds[fi], diff, pct
            );
        }
    }
    if n_decoded < n {
        println!("  (ffmpeg decoded {}/{} frames)", n_decoded, n);
    }
    println!("RESULT: {}/{} frames clean", n_clean, n);
    if let Some(fi) = first_bad {
        println!("FIRST DIVERGENCE: frame {}", fi);
        std::process::exit(1);
    }
}
