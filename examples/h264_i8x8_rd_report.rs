// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
//
// Phase 100-I — 8×8 transform R-D + fingerprint report.
//
// For a sweep of quality settings, encode N frames twice at
// matched settings — once with the 8×8 transform disabled
// (`enable_transform_8x8 = false`, Main profile), once with it
// enabled (`enable_transform_8x8 = true`, High profile). Report:
//
//   - bytes, bitrate @ 30 fps,
//   - mean Y-PSNR vs source (via ffmpeg decode),
//   - % MBs that picked the 8×8 transform (flag-ON path only),
//   - ffprobe profile detection.
//
// Usage:
//   h264_i8x8_rd_report <yuv> <w> <h> <n_frames> [gop_size=30]
//
// Expected at high QP / smooth content: ~5–10% bit reduction at
// similar PSNR. If the delta is > 15%, the 4×4 path is probably
// over-spending bits; if PSNR drops > 0.2 dB at equal bitrate, the
// 8×8 path is losing detail.

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use std::process::Command;

struct Run {
    bytes: usize,
    mean_psnr: f64,
    transform_8x8_fraction: f64,
    profile: String,
}

fn encode_one(
    yuv: &str,
    w: u32,
    h: u32,
    n: usize,
    q: u8,
    gop: usize,
    use_8x8: bool,
) -> (Vec<u8>, Vec<Vec<u8>>, Vec<(usize, usize)>) {
    let frame_size = (w * h * 3 / 2) as usize;
    let pixels = std::fs::read(yuv).expect("read yuv");
    assert!(pixels.len() >= n * frame_size, "yuv too short");

    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = use_8x8;

    let mut bytes = Vec::new();
    let mut recons = Vec::with_capacity(n);
    let mut transform_counts = Vec::with_capacity(n);

    for fi in 0..n {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        let is_i = fi == 0 || fi % gop == 0;
        let nal = if is_i {
            enc.encode_i_frame(src).unwrap()
        } else {
            enc.encode_p_frame(src).unwrap()
        };
        bytes.extend_from_slice(&nal);
        recons.push(enc.recon.y.clone());
        transform_counts.push((enc.transform_8x8_mb_count(), enc.total_mb_count()));
    }
    (bytes, recons, transform_counts)
}

fn ffprobe_profile(bytes: &[u8], tag: &str) -> String {
    let path = format!("/tmp/rd_probe_{tag}.h264");
    std::fs::write(&path, bytes).unwrap();
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "stream=profile", "-of",
               "default=noprint_wrappers=1:nokey=1", &path])
        .output();
    match out {
        Ok(r) if r.status.success() => String::from_utf8_lossy(&r.stdout).trim().to_string(),
        _ => "?".to_string(),
    }
}

fn decode_y_plane(bytes: &[u8], tag: &str, w: u32, h: u32, n: usize) -> Option<Vec<u8>> {
    let in_path = format!("/tmp/rd_dec_{tag}.h264");
    let out_path = format!("/tmp/rd_dec_{tag}.yuv");
    std::fs::write(&in_path, bytes).unwrap();
    let r = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "h264", "-i", &in_path,
               "-f", "rawvideo", "-pix_fmt", "yuv420p", &out_path])
        .output()
        .ok()?;
    if !r.status.success() {
        eprintln!("ffmpeg decode failed ({tag}): {}", String::from_utf8_lossy(&r.stderr));
        return None;
    }
    let decoded = std::fs::read(&out_path).ok()?;
    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w * h) as usize;
    if decoded.len() < n * frame_size {
        return None;
    }
    let mut out = Vec::with_capacity(n * y_size);
    for fi in 0..n {
        out.extend_from_slice(&decoded[fi * frame_size..fi * frame_size + y_size]);
    }
    Some(out)
}

fn psnr_y(src: &[u8], dec: &[u8]) -> f64 {
    let sqe: f64 = src.iter().zip(dec.iter())
        .map(|(s, d)| { let d = *s as f64 - *d as f64; d * d })
        .sum();
    let mse = sqe / src.len() as f64;
    if mse == 0.0 { 99.99 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
}

fn run_one_q(yuv: &str, w: u32, h: u32, n: usize, q: u8, gop: usize, use_8x8: bool) -> Option<Run> {
    let (bytes, _, tx_counts) = encode_one(yuv, w, h, n, q, gop, use_8x8);
    let tag = format!("q{q}_{}", if use_8x8 { "on" } else { "off" });
    let profile = ffprobe_profile(&bytes, &tag);
    let dec_y = decode_y_plane(&bytes, &tag, w, h, n)?;
    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w * h) as usize;
    let src = std::fs::read(yuv).ok()?;

    let mut sum_psnr = 0.0;
    for fi in 0..n {
        let base_src = fi * frame_size;
        let src_y = &src[base_src..base_src + y_size];
        let dy = &dec_y[fi * y_size..(fi + 1) * y_size];
        sum_psnr += psnr_y(src_y, dy);
    }
    let mean_psnr = sum_psnr / n as f64;

    let (total_tx8, total_mbs) = tx_counts.iter().fold((0usize, 0usize),
        |(a, b), &(x, y)| (a + x, b + y));
    let fraction = if total_mbs > 0 {
        total_tx8 as f64 / total_mbs as f64
    } else { 0.0 };

    Some(Run { bytes: bytes.len(), mean_psnr, transform_8x8_fraction: fraction, profile })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: {} <yuv> <w> <h> <n_frames> [gop=30]", args[0]);
        std::process::exit(2);
    }
    let yuv = &args[1];
    let w: u32 = args[2].parse().unwrap();
    let h: u32 = args[3].parse().unwrap();
    let n: usize = args[4].parse().unwrap();
    let gop: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(30);

    let q_points: Vec<u8> = vec![40, 50, 60, 70, 80, 90];

    println!("Phase 100-I: 8×8 transform R-D + fingerprint report");
    println!("  input:   {yuv}  ({}×{}, {} frames, GOP={})", w, h, n, gop);
    println!();
    println!(
        "  {:>5} │ {:>11} {:>10} {:>8} {:>8} │ {:>11} {:>10} {:>8} {:>8} {:>8} │ {:>7}",
        "q", "OFF bytes", "OFF Mbps", "OFF dB", "prof",
        "ON bytes", "ON Mbps", "ON dB", "prof", "%tx8×8",
        "bit Δ"
    );
    println!("  {}─┼{}┼{}┼{}",
        "─".repeat(5), "─".repeat(40), "─".repeat(48), "─".repeat(8));

    for &q in &q_points {
        let Some(off) = run_one_q(yuv, w, h, n, q, gop, false) else {
            println!("  q={q}  OFF: encode/decode failed");
            continue;
        };
        let Some(on) = run_one_q(yuv, w, h, n, q, gop, true) else {
            println!("  q={q}  ON:  encode/decode failed");
            continue;
        };
        let mbps_off = off.bytes as f64 * 8.0 * 30.0 / n as f64 / 1e6;
        let mbps_on = on.bytes as f64 * 8.0 * 30.0 / n as f64 / 1e6;
        let bit_delta = (on.bytes as f64 - off.bytes as f64) / off.bytes as f64 * 100.0;
        println!(
            "  {:>5} │ {:>11} {:>10.2} {:>8.2} {:>8} │ {:>11} {:>10.2} {:>8.2} {:>8} {:>7.1}% │ {:>+6.2}%",
            q,
            off.bytes, mbps_off, off.mean_psnr, off.profile,
            on.bytes,  mbps_on,  on.mean_psnr,  on.profile, on.transform_8x8_fraction * 100.0,
            bit_delta
        );
    }

    println!();
    println!("  profile column: expect 'Main' when flag OFF, 'High' when flag ON.");
    println!("  %tx8×8 column : share of encoded MBs that took the 8×8 transform path.");
    println!("  bit Δ column  : (ON / OFF − 1) × 100%. Negative = 8×8 reduces bitrate.");
}
