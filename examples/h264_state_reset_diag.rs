// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Diagnostic: does the encoder carry stale state across IDR boundaries?
//!
//! Two encodes of frames 30..40 from IMG_4138:
//!   A) fresh encoder, f30 = first IDR, f31..39 P (normal startup).
//!   B) same encoder state used for Run-A frames 0..29, then f30 IDR,
//!      f31..39 P (the state the real encoder reaches at a GOP boundary).
//!
//! Compare PSNR of both runs' decoded outputs vs source. If A ≫ B on
//! post-f30 P-frames, we have a state carryover bug. If A ≈ B, content
//! at f30+ is genuinely harder than at f0+.
//!
//! Run with:
//!   cargo build --release --features video,h264-encoder \
//!     --example h264_state_reset_diag
//!   ./target/release/examples/h264_state_reset_diag

use std::process::Command;

use phasm_core::codec::h264::encoder::encoder::Encoder;

fn encode_range(
    enc: &mut Encoder,
    pixels: &[u8],
    frame_size: usize,
    start: usize,
    end: usize,
    gop: usize,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    for f in start..end {
        let s = f * frame_size;
        let is_idr = f == start || (f - start) % gop == 0;
        if is_idr {
            bytes.extend_from_slice(&enc.encode_i_frame(&pixels[s..s + frame_size]).expect("i"));
        } else {
            bytes.extend_from_slice(&enc.encode_p_frame(&pixels[s..s + frame_size]).expect("p"));
        }
    }
    bytes
}

fn psnr_trace(bytes: &[u8], pixels: &[u8], frame_size: usize, w: u32, h: u32, start: usize, end: usize) -> Vec<f64> {
    let h264_path = "/tmp/state_diag.h264";
    std::fs::write(h264_path, bytes).unwrap();
    let decoded = "/tmp/state_diag_decoded.yuv";
    let _ = Command::new("ffmpeg")
        .args([
            "-y", "-loglevel", "error",
            "-f", "h264", "-i", h264_path,
            "-f", "rawvideo", "-pix_fmt", "yuv420p", decoded,
        ])
        .output()
        .expect("ffmpeg decode");
    let d = std::fs::read(decoded).expect("decoded");
    let y_size = (w * h) as usize;
    let mut v = Vec::with_capacity(end - start);
    for i in 0..(end - start) {
        let src_off = (start + i) * frame_size;
        let dec_off = i * frame_size;
        if dec_off + y_size > d.len() {
            v.push(0.0);
            continue;
        }
        let s = &pixels[src_off..src_off + y_size];
        let d = &d[dec_off..dec_off + y_size];
        let mut sqe = 0.0;
        for (a, b) in s.iter().zip(d.iter()) {
            let dd = *a as f64 - *b as f64;
            sqe += dd * dd;
        }
        let mse = sqe / y_size as f64;
        v.push(if mse > 0.0 { 10.0 * (255.0 * 255.0 / mse).log10() } else { 99.99 });
    }
    v
}

fn main() {
    let w: u32 = 1920;
    let h: u32 = 1072;
    let n: usize = 40; // we go f0..f40
    let frame_size = (w * h * 3 / 2) as usize;
    let yuv_path = format!("/tmp/img4138_1080p_f{n}.yuv");
    let need = std::fs::metadata(&yuv_path)
        .map(|m| m.len() as usize != frame_size * n)
        .unwrap_or(true);
    if need {
        let src = format!("{}/Desktop/IMG_4138.MOV", std::env::var("HOME").unwrap());
        let o = Command::new("ffmpeg")
            .args([
                "-y", "-loglevel", "error",
                "-i", &src,
                "-vf", &format!("scale={w}:{h},format=yuv420p"),
                "-frames:v", &n.to_string(),
                "-f", "rawvideo", &yuv_path,
            ])
            .output()
            .expect("ffmpeg transcode");
        if !o.status.success() {
            eprintln!("ffmpeg transcode failed: {}", String::from_utf8_lossy(&o.stderr));
            std::process::exit(1);
        }
    }
    let pixels = std::fs::read(&yuv_path).expect("read yuv");

    // Run A — fresh encoder, starts at f30 as first IDR, encodes f30..40.
    let mut enc_a = Encoder::new(w, h, Some(80)).unwrap();
    let bytes_a = encode_range(&mut enc_a, &pixels, frame_size, 30, 40, 30);
    let psnr_a = psnr_trace(&bytes_a, &pixels, frame_size, w, h, 30, 40);

    // Run B — encoder has seen f0..29 first (drift state), then GOP=30 fires IDR at f30, encodes through f40.
    let mut enc_b = Encoder::new(w, h, Some(80)).unwrap();
    let bytes_b_full = encode_range(&mut enc_b, &pixels, frame_size, 0, 40, 30);
    // Extract only the f30..40 portion of PSNR (but bitstream contains f0..40).
    let psnr_b_all = psnr_trace(&bytes_b_full, &pixels, frame_size, w, h, 0, 40);
    let psnr_b = &psnr_b_all[30..];

    println!("=== State-reset diagnostic ===");
    println!("A: fresh encoder, f30..39 (f30 is first frame).");
    println!("B: encoder has seen f0..29 first, then f30 IDR, f31..39 P.");
    println!();
    println!("frame | A (fresh) | B (after 30 frames) | Δ (A-B)");
    for (i, (a, b)) in psnr_a.iter().zip(psnr_b.iter()).enumerate() {
        let f = 30 + i;
        let delta = a - b;
        println!("  f{f:3} | {a:7.2}   | {b:7.2}             | {delta:+6.2}");
    }
    let avg_a: f64 = psnr_a.iter().sum::<f64>() / psnr_a.len() as f64;
    let avg_b: f64 = psnr_b.iter().sum::<f64>() / psnr_b.len() as f64;
    println!();
    println!("avg A: {:.2} dB", avg_a);
    println!("avg B: {:.2} dB", avg_b);
    println!("avg Δ: {:+.2} dB", avg_a - avg_b);
    if avg_a - avg_b > 1.5 {
        println!("→ STATE CARRYOVER BUG: run B is ≥1.5 dB worse even though f30 is IDR in both.");
    } else {
        println!("→ content-dependent: both runs behave similarly, content at f30+ is just harder.");
    }
}
