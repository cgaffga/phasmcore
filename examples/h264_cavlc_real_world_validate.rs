// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Encode IMG_4138.MOV at 1920×1072 for 30 frames (1 GOP) via CAVLC,
//! compare to source per frame, report Y-PSNR trajectory. Produces
//! `/Users/cgaffga/Desktop/IMG_4138_cavlc_full.mp4` + an early-frame
//! PNG sample in /tmp for visual sanity.

use std::process::Command;

use phasm_core::codec::h264::encoder::encoder::Encoder;

fn main() {
    let w: u32 = 1920;
    let h: u32 = 1072;
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let q: u8 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(80);

    let frame_size = (w * h * 3 / 2) as usize;
    let yuv_path = format!("/tmp/img4138_1080p_f{n}.yuv");

    // Regenerate the source YUV if missing or wrong size.
    let need = std::fs::metadata(&yuv_path)
        .map(|m| m.len() as usize != frame_size * n)
        .unwrap_or(true);
    if need {
        let src = format!("{}/Desktop/IMG_4138.MOV", std::env::var("HOME").unwrap());
        let out = Command::new("ffmpeg")
            .args([
                "-y", "-loglevel", "error",
                "-i", &src,
                "-vf", &format!("scale={w}:{h},format=yuv420p"),
                "-frames:v", &n.to_string(),
                "-f", "rawvideo", &yuv_path,
            ])
            .output().expect("ffmpeg transcode");
        if !out.status.success() {
            eprintln!("ffmpeg transcode: {}", String::from_utf8_lossy(&out.stderr));
            std::process::exit(1);
        }
    }
    let pixels = std::fs::read(&yuv_path).expect("read yuv");
    assert_eq!(pixels.len(), frame_size * n);

    eprintln!("encoding {n} frames at {w}x{h}, quality={q}, deblock={}",
        if std::env::var_os("PHASM_DISABLE_DEBLOCK").is_some() { "OFF" } else { "on" });
    let t_start = std::time::Instant::now();
    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    let gop = std::env::var("GOP").ok().and_then(|s| s.parse().ok()).unwrap_or(30usize);
    let mut bytes = enc.encode_i_frame(&pixels[..frame_size]).expect("i");
    for f in 1..n {
        let s = f * frame_size;
        if f % gop == 0 {
            bytes.extend_from_slice(&enc.encode_i_frame(&pixels[s..s + frame_size]).expect("i_periodic"));
        } else {
            bytes.extend_from_slice(&enc.encode_p_frame(&pixels[s..s + frame_size]).expect("p"));
        }
    }
    let encode_time = t_start.elapsed();

    let h264_path = "/tmp/img4138_cavlc.h264";
    std::fs::write(h264_path, &bytes).unwrap();

    // ffmpeg conformance check — log every line of stderr.
    let conform = Command::new("ffmpeg")
        .args([
            "-loglevel", "error",
            "-f", "h264", "-i", h264_path,
            "-f", "null", "-",
        ])
        .output().expect("ffmpeg conformance");
    let ffmpeg_stderr = String::from_utf8_lossy(&conform.stderr).into_owned();
    let ffmpeg_error_count = ffmpeg_stderr
        .lines()
        .filter(|l| l.to_lowercase().contains("error")
            || l.to_lowercase().contains("missing")
            || l.to_lowercase().contains("corrupt"))
        .count();

    // Decode with ffmpeg back to raw YUV for PSNR.
    let decoded_yuv = "/tmp/img4138_cavlc_decoded.yuv";
    let out = Command::new("ffmpeg")
        .args([
            "-y", "-loglevel", "error",
            "-f", "h264", "-i", h264_path,
            "-f", "rawvideo", "-pix_fmt", "yuv420p", decoded_yuv,
        ])
        .output().expect("ffmpeg decode");
    if !out.status.success() {
        eprintln!("ffmpeg decode errors: {}", String::from_utf8_lossy(&out.stderr));
    }
    let decoded = std::fs::read(decoded_yuv).expect("read decoded");

    // Per-frame Y-PSNR.
    let y_size = (w * h) as usize;
    let mut psnrs = Vec::with_capacity(n);
    for f in 0..n {
        let src_y = &pixels[f * frame_size..f * frame_size + y_size];
        let dec_y = &decoded[f * frame_size..f * frame_size + y_size];
        let mut sqe = 0.0;
        for (a, b) in src_y.iter().zip(dec_y.iter()) {
            let d = *a as f64 - *b as f64;
            sqe += d * d;
        }
        let mse = sqe / y_size as f64;
        let psnr = if mse > 0.0 {
            10.0 * (255.0 * 255.0 / mse).log10()
        } else {
            99.99
        };
        psnrs.push(psnr);
    }
    let avg: f64 = psnrs.iter().sum::<f64>() / n as f64;

    // Remux to MP4.
    let mp4 = "/Users/cgaffga/Desktop/IMG_4138_cavlc_full.mp4";
    Command::new("ffmpeg")
        .args([
            "-y", "-loglevel", "error",
            "-f", "h264", "-r", "30", "-i", h264_path,
            "-c:v", "copy", mp4,
        ])
        .output().expect("remux");

    // Dump frame PNGs (frame 0, frame 15, frame 29, frame 30 if exists, frame n-1).
    let sample_frames: Vec<usize> = [0usize, 15, 29, 30, n.saturating_sub(1)]
        .iter().copied().filter(|&f| f < n).collect();
    for &f in &sample_frames {
        let _ = Command::new("ffmpeg")
            .args([
                "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", &format!("{w}x{h}"),
                "-i", decoded_yuv,
                "-vf", &format!("select=eq(n\\,{f})"),
                "-frames:v", "1",
                &format!("/tmp/cavlc_f{f:03}.png"),
            ])
            .output();
    }

    println!("\n=== CAVLC real-world results ===");
    println!("  resolution : {w}x{h}");
    println!("  frames     : {n}");
    println!("  quality    : {q}");
    println!("  encode time: {:.1}s ({:.2} fps)", encode_time.as_secs_f64(), n as f64 / encode_time.as_secs_f64());
    println!("  bytes      : {} ({:.2} Mbps at 30fps)",
        bytes.len(), bytes.len() as f64 * 8.0 * 30.0 / n as f64 / 1_000_000.0);
    println!("  Y-PSNR avg : {avg:.2} dB");
    let verbose = std::env::var_os("PHASM_VERBOSE").is_some();
    for (i, p) in psnrs.iter().enumerate() {
        if verbose || i < 5 || i > n - 3 || i % 5 == 0 {
            println!("    f{i:3}: {p:.2} dB");
        }
    }
    println!("  MP4 out    : {mp4}");
    println!("  frame PNGs : {}", sample_frames.iter().map(|f| format!("/tmp/cavlc_f{f:03}.png")).collect::<Vec<_>>().join(", "));
    println!("\n=== ffmpeg conformance ===");
    if ffmpeg_error_count == 0 && ffmpeg_stderr.trim().is_empty() {
        println!("  ✓ clean — zero stderr");
    } else if ffmpeg_error_count == 0 {
        println!("  ✓ clean — no error lines, but stderr had:");
        for l in ffmpeg_stderr.lines().take(10) { println!("    {l}"); }
    } else {
        println!("  ✗ {ffmpeg_error_count} error/missing/corrupt lines:");
        for l in ffmpeg_stderr.lines().take(20) { println!("    {l}"); }
    }

    // ─── Quality bar: fail loudly if output is sub-H.264 quality ───
    //
    // Reference: x264 at CRF=21, GOP=30 encodes IMG_4138 at ~8.8 Mbps with
    // sustained 38-42 dB Y-PSNR. Anything dramatically worse is a regression.
    //
    // These thresholds are loose enough that known-good outputs pass, and
    // tight enough that regressions (like our current 13 Mbps / 29 dB result
    // at q=80 GOP=30) are caught.
    let mut failures: Vec<String> = Vec::new();
    let bitrate_mbps = (bytes.len() as f64 * 8.0 * 30.0 / n as f64) / 1_000_000.0;
    let min_per_frame_psnr = psnrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_acceptable_mbps = 15.0;
    let min_avg_psnr = 35.0;
    let min_per_frame_psnr_floor = 25.0;
    if bitrate_mbps > max_acceptable_mbps {
        failures.push(format!(
            "bitrate {bitrate_mbps:.2} Mbps > {max_acceptable_mbps:.1} Mbps limit"
        ));
    }
    if avg < min_avg_psnr {
        failures.push(format!(
            "avg Y-PSNR {avg:.2} dB < {min_avg_psnr:.1} dB threshold"
        ));
    }
    if min_per_frame_psnr < min_per_frame_psnr_floor {
        failures.push(format!(
            "worst-frame Y-PSNR {min_per_frame_psnr:.2} dB < {min_per_frame_psnr_floor:.1} dB floor"
        ));
    }
    println!("\n=== quality gate ===");
    if failures.is_empty() {
        println!("  ✓ PASS — bitrate/PSNR within shippable H.264 envelope");
    } else {
        println!("  ✗ FAIL:");
        for f in &failures { println!("    - {f}"); }
        println!("  reference: x264 CRF=21 GOP=30 on same source ≈ 8.8 Mbps @ 38-42 dB");
        std::process::exit(2);
    }
}
