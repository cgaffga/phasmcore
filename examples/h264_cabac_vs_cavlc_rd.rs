// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
//! Measure CABAC vs CAVLC rate-distortion at matched QP.
//!
//! For each (yuv, q) pair, encode N frames twice — once with CAVLC,
//! once with CABAC — at identical encoder settings. Decode through
//! ffmpeg (both must decode ffmpeg-clean). Report bits, bitrate, and
//! per-frame Y-PSNR vs source for each mode; compute CABAC savings.

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use std::process::Command;

fn encode(
    yuv_path: &str,
    w: u32,
    h: u32,
    n: usize,
    q: u8,
    mode: EntropyMode,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let frame_size = (w * h * 3 / 2) as usize;
    let pixels = std::fs::read(yuv_path).expect("read yuv");
    assert!(pixels.len() >= n * frame_size);
    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    enc.entropy_mode = mode;
    let mut bytes = enc.encode_i_frame(&pixels[..frame_size]).unwrap();
    let mut recons = vec![enc.recon.y.clone()];
    for fi in 1..n {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        bytes.extend_from_slice(&enc.encode_p_frame(src).unwrap());
        recons.push(enc.recon.y.clone());
    }
    (bytes, recons)
}

fn ffmpeg_decode(bytes: &[u8], label: &str, frame_size: usize) -> Option<Vec<u8>> {
    let in_path = format!("/tmp/rd_{label}.h264");
    let out_path = format!("/tmp/rd_{label}.yuv");
    std::fs::write(&in_path, bytes).unwrap();
    let r = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "h264", "-i", &in_path,
               "-f", "rawvideo", "-pix_fmt", "yuv420p", &out_path])
        .output()
        .ok()?;
    if !r.status.success() {
        eprintln!("ffmpeg failed on {label}: {}", String::from_utf8_lossy(&r.stderr));
        return None;
    }
    let out = std::fs::read(&out_path).ok()?;
    if out.len() < frame_size {
        return None;
    }
    Some(out)
}

fn psnr_y(src_y: &[u8], dec_y: &[u8]) -> f64 {
    assert_eq!(src_y.len(), dec_y.len());
    let sqe: f64 = src_y.iter().zip(dec_y.iter())
        .map(|(s, d)| {
            let d = *s as f64 - *d as f64;
            d * d
        })
        .sum();
    let mse = sqe / src_y.len() as f64;
    if mse == 0.0 { 99.99 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
}

fn run_one(label: &str, yuv: &str, w: u32, h: u32, n: usize, q: u8) {
    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w * h) as usize;
    let src = std::fs::read(yuv).unwrap();

    let mut row = format!("{:>10} @q{:2} ({:3}f)", label, q, n);

    for mode in [EntropyMode::Cavlc, EntropyMode::Cabac] {
        let t = std::time::Instant::now();
        let (bytes, _) = encode(yuv, w, h, n, q, mode);
        let encode_ms = t.elapsed().as_millis();
        let Some(decoded) = ffmpeg_decode(&bytes, &format!("{label}_{mode:?}"), frame_size) else {
            row.push_str(&format!("  {mode:?}: FFMPEG FAIL"));
            continue;
        };

        let mut psnr_sum = 0.0f64;
        let n_dec = decoded.len() / frame_size;
        for fi in 0..n.min(n_dec) {
            let sy = &src[fi * frame_size..fi * frame_size + y_size];
            let dy = &decoded[fi * frame_size..fi * frame_size + y_size];
            psnr_sum += psnr_y(sy, dy);
        }
        let psnr_avg = psnr_sum / n_dec as f64;
        let bitrate = bytes.len() as f64 * 8.0 * 30.0 / n as f64 / 1e6;
        row.push_str(&format!(
            "  {:5}: {:>7} B  {:5.2} Mbps  PSNR {:.2} dB  ({:>4} ms)",
            format!("{:?}", mode),
            bytes.len(),
            bitrate,
            psnr_avg,
            encode_ms,
        ));
    }
    println!("{row}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: {} <yuv> <w> <h> <n_frames> [q_list=30,50,70,80,90]",
            args[0]
        );
        std::process::exit(2);
    }
    let yuv = &args[1];
    let w: u32 = args[2].parse().unwrap();
    let h: u32 = args[3].parse().unwrap();
    let n: usize = args[4].parse().unwrap();
    let qs: Vec<u8> = args
        .get(5)
        .cloned()
        .unwrap_or_else(|| "30,50,70,80,90".to_string())
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    println!(
        "# CABAC vs CAVLC R-D — {w}x{h} {n} frames from {}",
        yuv.rsplit('/').next().unwrap_or(yuv)
    );
    println!("# Columns: mode / bytes / bitrate (Mbps @30fps) / avg Y-PSNR vs source / encode time");
    println!();
    let label = yuv.rsplit('/').next().unwrap_or(yuv);
    for &q in &qs {
        run_one(label, yuv, w, h, n, q);
    }
}
