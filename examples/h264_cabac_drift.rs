// Same as h264_nframe_drift but runs CABAC. Prints per-frame
// enc-vs-dec PSNR so we can see where CABAC breaks.

use std::process::Command;
use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};

fn main() {
    let w: u32 = 1920;
    let h: u32 = 1072;
    let n_frames: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let yuv_path = std::env::var("YUV").unwrap_or_else(|_| format!("/tmp/img4138_1080p_f{n_frames}.yuv"));
    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w * h) as usize;
    let pixels = std::fs::read(&yuv_path).expect("yuv missing");
    assert!(pixels.len() >= n_frames * frame_size, "yuv too short");

    let q: u8 = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    let mut bytes = enc.encode_i_frame(&pixels[..frame_size]).unwrap();
    let mut enc_recons = vec![(enc.recon.y.clone(), enc.recon.cb.clone(), enc.recon.cr.clone())];
    for fi in 1..n_frames {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        bytes.extend_from_slice(&enc.encode_p_frame(src).unwrap());
        enc_recons.push((enc.recon.y.clone(), enc.recon.cb.clone(), enc.recon.cr.clone()));
    }

    std::fs::write("/tmp/cabac_drift.h264", &bytes).unwrap();
    let dec_out = Command::new("ffmpeg").args([
        "-y", "-f", "h264", "-i", "/tmp/cabac_drift.h264",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "/tmp/cabac_drift_decoded.yuv"
    ]).output().unwrap();
    if !dec_out.status.success() {
        let err = String::from_utf8_lossy(&dec_out.stderr);
        eprintln!("FFMPEG DECODE FAILED:\n{}", err.lines().take(20).collect::<Vec<_>>().join("\n"));
    }
    let decoded = std::fs::read("/tmp/cabac_drift_decoded.yuv").unwrap_or_default();
    let n_decoded = if decoded.is_empty() { 0 } else { decoded.len() / frame_size };
    println!("encoded {} frames; ffmpeg decoded {}", n_frames, n_decoded);
    println!("bitstream size: {} bytes ({:.2} Mbps @ 30fps)", bytes.len(),
        bytes.len() as f64 * 8.0 / (n_frames as f64 / 30.0) / 1_000_000.0);
    if n_decoded == 0 { return; }
    let n_check = n_decoded.min(n_frames);

    println!();
    println!("frame |  enc-vs-dec |  src-vs-recon  | src-vs-dec  | notes");
    println!("------+-------------+----------------+-------------+--------");
    for fi in 0..n_check {
        let (ey, _, _) = &enc_recons[fi];
        let base = fi * frame_size;
        let dy = &decoded[base..base + y_size];
        let src_y = &pixels[base..base + y_size];

        let mut ed_sqe = 0.0f64; let mut diff = 0usize; let mut maxd = 0i32;
        for (a, b) in ey.iter().zip(dy.iter()) {
            let d = *a as i32 - *b as i32;
            if d != 0 { diff += 1; }
            if d.abs() > maxd { maxd = d.abs(); }
            ed_sqe += (d as f64) * (d as f64);
        }
        let ed_mse = ed_sqe / y_size as f64;
        let ed_psnr = if ed_mse > 0.0 { 10.0 * (255.0 * 255.0 / ed_mse).log10() } else { 99.99 };

        let mut se = 0.0; let mut sd = 0.0;
        for i in 0..y_size {
            let s = src_y[i] as f64;
            se += (s - ey[i] as f64).powi(2);
            sd += (s - dy[i] as f64).powi(2);
        }
        let se_psnr = 10.0 * (255.0 * 255.0 / (se / y_size as f64)).log10();
        let sd_psnr = 10.0 * (255.0 * 255.0 / (sd / y_size as f64)).log10();

        let note = if diff == 0 { String::from("in sync") }
            else { format!("DESYNC {}/{} ({:.2}%) max{}", diff, y_size, diff as f64 * 100.0 / y_size as f64, maxd) };
        println!("{fi:5} | {ed_psnr:8.2} dB | {se_psnr:10.2} dB | {sd_psnr:8.2} dB | {note}");
    }
}
