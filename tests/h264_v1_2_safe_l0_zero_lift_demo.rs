// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase 2.18 follow-on (#287, 2026-05-09) — visual A/B demo for
// SAFE_L0_ZERO lift after the corner-sampling fix in
// `derive_b_direct_spatial_with_col`. Encodes carplane fixture
// through the full stego path with whatever B-RDO config is
// currently set in `encode_pixels.rs::build_encoder()`. Computes
// per-frame Y-PSNR and writes MP4 to ~/Desktop with a label
// chosen via the `PHASM_DEMO_LABEL` env var.
//
// Run twice across the lift to produce a side-by-side comparison:
//
//   # Step 1 — baseline (SAFE_L0_ZERO ship default):
//   PHASM_DEMO_LABEL=A_safe_l0_zero \
//     cargo test --release --features cabac-stego \
//     --test h264_v1_2_safe_l0_zero_lift_demo \
//     -- --ignored --nocapture
//
//   # Step 2 — after editing build_encoder() to set
//   #          BRdoConfig::PRODUCTION_VISUAL:
//   PHASM_DEMO_LABEL=B_production_visual \
//     cargo test --release --features cabac-stego \
//     --test h264_v1_2_safe_l0_zero_lift_demo \
//     -- --ignored --nocapture

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::mp4::build::{
    build_mp4_with_pattern, FrameTiming, MuxerProfile,
};
use phasm_core::{
    h264_stego_encode_yuv_string_4domain_multigop_with_pattern, GopPattern,
};
use std::time::Instant;

fn psnr_y(decoded: &[u8], source: &[u8], w: u32, h: u32) -> f64 {
    let n = (w * h) as usize;
    let mut sse: u64 = 0;
    for i in 0..n {
        let d = decoded[i] as i64 - source[i] as i64;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return 99.0;
    }
    let mse = sse as f64 / n as f64;
    20.0 * (255.0_f64).log10() - 10.0 * mse.log10()
}

#[test]
#[ignore]
fn v1_2_safe_l0_zero_lift_demo_carplane() {
    let label = std::env::var("PHASM_DEMO_LABEL")
        .unwrap_or_else(|_| "unlabelled".to_string());
    let yuv_path = "/tmp/phasm_corpus_artlist_carplane_1072x1920_f30.yuv";
    let yuv = std::fs::read(yuv_path).unwrap_or_else(|e| {
        panic!("missing {yuv_path}: {e}\nRegenerate via the corpus harness.")
    });
    eprintln!("[{label}] loaded {} bytes from {}", yuv.len(), yuv_path);

    let width = 1072u32;
    let height = 1920u32;
    let n_frames = 30usize;
    let pattern = if std::env::var_os("PHASM_DEMO_IPPPP").is_some() {
        GopPattern::Ipppp { gop: 30 }
    } else {
        GopPattern::Ibpbp { gop: 30, b_count: 1 }
    };

    let t0 = Instant::now();
    let stego = h264_stego_encode_yuv_string_4domain_multigop_with_pattern(
        &yuv, width, height, n_frames, pattern,
        "phasm-demo-secret",
        "phasm-v1.2-safe-l0-zero-lift",
    )
    .expect("stego encode");
    eprintln!("[{label}] encoded {} bytes in {:?}", stego.len(), t0.elapsed());

    // Decode via ffmpeg → raw YUV → per-frame Y-PSNR.
    let frame_size = (width * height * 3 / 2) as usize;
    let h264_path = std::env::temp_dir().join(format!("phasm_v12_demo_{label}.h264"));
    let dec_path = std::env::temp_dir().join(format!("phasm_v12_demo_{label}.dec.yuv"));
    std::fs::write(&h264_path, &stego).expect("write h264");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .status()
        .expect("ffmpeg decode");
    assert!(status.success(), "ffmpeg decode failed");
    let decoded = std::fs::read(&dec_path).expect("read decoded yuv");

    let mut total = 0.0;
    let mut min_psnr = 99.0_f64;
    let mut min_frame = 0usize;
    eprintln!("[{label}] per-frame Y-PSNR vs source:");
    for i in 0..n_frames {
        let off = i * frame_size;
        let src_y = &yuv[off..off + (width * height) as usize];
        let dec_y = &decoded[off..off + (width * height) as usize];
        let p = psnr_y(dec_y, src_y, width, height);
        total += p;
        if p < min_psnr {
            min_psnr = p;
            min_frame = i;
        }
        eprintln!("  frame={:>2} Y-PSNR={:>6.2} dB", i, p);
    }
    eprintln!(
        "[{label}] avg Y-PSNR={:.2} dB  min={:.2} dB at frame {}",
        total / n_frames as f64,
        min_psnr,
        min_frame
    );

    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264,
        &stego,
        width, height,
        FrameTiming::FPS_30,
        pattern, n_frames,
    )
    .expect("mp4 mux");

    let home = std::env::var("HOME").expect("HOME not set");
    let desktop = std::path::PathBuf::from(home).join("Desktop");
    let mp4_path = desktop.join(format!("phasm_v12_carplane_{label}.mp4"));
    std::fs::write(&mp4_path, &mp4).expect("write desktop mp4");
    eprintln!("[{label}] mp4 → {}", mp4_path.display());
}
