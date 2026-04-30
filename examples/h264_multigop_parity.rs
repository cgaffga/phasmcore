// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Multi-GOP enc-vs-dec parity gate. Encodes N frames at 1920×1072 with
//! IDR every GOP frames, decodes the bitstream with ffmpeg, compares
//! per-frame Y plane. Prints PSNR + a one-line PASS/FAIL on the
//! 99.99 dB gate. Used by task #72 to confirm that multi-GOP IDR
//! insertion does not break parity.

use std::process::Command;
use phasm_core::codec::h264::encoder::encoder::Encoder;

fn main() {
    let w: u32 = 1920;
    let h: u32 = 1072;
    let n_frames: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(90);
    let gop: usize = std::env::var("GOP").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let q: u8 = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let yuv_path = std::env::var("YUV")
        .unwrap_or_else(|_| "/tmp/img4138_full_1072.yuv".into());

    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w * h) as usize;
    let pixels = std::fs::read(&yuv_path).expect("yuv missing");
    assert!(
        pixels.len() >= n_frames * frame_size,
        "yuv too short: {} bytes vs need {}",
        pixels.len(),
        n_frames * frame_size
    );

    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    enc.set_gop_length(gop as u32);

    let mut bytes: Vec<u8> = Vec::with_capacity(64 * 1024 * 1024);
    let mut enc_recons: Vec<Vec<u8>> = Vec::with_capacity(n_frames);

    for fi in 0..n_frames {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        let is_idr = fi % gop == 0;
        let out = if is_idr {
            enc.encode_i_frame(src).unwrap()
        } else {
            enc.encode_p_frame(src).unwrap()
        };
        bytes.extend_from_slice(&out);
        enc_recons.push(enc.recon.y.clone());
    }

    let h264_path = std::env::var("PHASM_OUT_H264")
        .unwrap_or_else(|_| "/tmp/multigop_parity.h264".into());
    let dec_path = std::env::var("PHASM_OUT_DEC_YUV")
        .unwrap_or_else(|_| "/tmp/multigop_parity_decoded.yuv".into());
    std::fs::write(&h264_path, &bytes).unwrap();

    let ff = Command::new("ffmpeg").args([
        "-y", "-loglevel", "error",
        "-f", "h264", "-i", h264_path.as_str(),
        "-f", "rawvideo", "-pix_fmt", "yuv420p", dec_path.as_str(),
    ]).output().unwrap();
    if !ff.status.success() {
        eprintln!("ffmpeg failed: {}", String::from_utf8_lossy(&ff.stderr));
        std::process::exit(2);
    }

    let decoded = std::fs::read(&dec_path).unwrap();
    let n_dec = decoded.len() / frame_size;
    if n_dec != n_frames {
        eprintln!("WARN: decoded {n_dec} frames, encoded {n_frames}");
    }
    let n_check = n_dec.min(n_frames);

    let mut min_psnr = f64::INFINITY;
    let mut min_frame: usize = 0;
    let mut frames_pass = 0usize;
    let mut idr_count = 0usize;

    println!("frame | type | enc-vs-dec PSNR | status");
    println!("------+------+-----------------+--------");
    for fi in 0..n_check {
        let ey = &enc_recons[fi];
        let base = fi * frame_size;
        let dy = &decoded[base..base + y_size];
        let mut sqe = 0.0f64;
        for (a, b) in ey.iter().zip(dy.iter()) {
            let d = *a as f64 - *b as f64;
            sqe += d * d;
        }
        let mse = sqe / y_size as f64;
        let psnr = if mse > 0.0 { 10.0 * (255.0 * 255.0 / mse).log10() } else { 99.99 };
        let kind = if fi % gop == 0 { idr_count += 1; "IDR" } else { "P  " };
        let pass = psnr >= 99.99;
        if pass { frames_pass += 1; }
        if psnr < min_psnr { min_psnr = psnr; min_frame = fi; }
        println!("{fi:5} | {kind:>4} | {psnr:13.4} dB | {}",
            if pass { "OK" } else { "FAIL" });
    }
    println!();
    println!("Summary: {}/{} frames at >=99.99 dB ({} IDRs)",
        frames_pass, n_check, idr_count);
    println!("Min PSNR: {:.4} dB at frame {}", min_psnr, min_frame);

    if frames_pass == n_check && n_check == n_frames {
        println!("PARITY GATE: PASS");
        std::process::exit(0);
    } else {
        println!("PARITY GATE: FAIL");
        std::process::exit(1);
    }
}
