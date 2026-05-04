// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Fast-iterate PSNR diagnostic for the §B-direct visual bug (#186).
// Encodes /tmp/iphone7_640x384_f10.yuv at IBPBP {gop:5, b_count:1},
// writes Annex-B → MP4 via ffmpeg, decodes back to YUV, computes
// luma PSNR vs source. Skip-Direct only (PHASM_B_RDO unset).
//
// Run with:
//   cargo test --release --features cabac-stego --test h264_iphone_psnr_diag -- --ignored --nocapture

#![cfg(feature = "cabac-stego")]

use phasm_core::{
    h264_stego_encode_yuv_string_4domain_multigop_with_pattern, GopPattern,
};

fn psnr_y(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut mse: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        mse += d * d;
    }
    mse /= a.len() as f64;
    if mse == 0.0 {
        return 99.0;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

#[test]
#[ignore]
fn iphone7_small_ibpbp_psnr() {
    let yuv = std::fs::read("/tmp/iphone7_640x384_f10.yuv")
        .expect("missing /tmp/iphone7_640x384_f10.yuv");
    assert_eq!(yuv.len(), 640 * 384 * 3 / 2 * 10, "fixture size mismatch");

    unsafe {
        std::env::remove_var("PHASM_B_RDO");
        // Deterministic crypto seed so consecutive runs produce
        // byte-identical phasm output → PSNR is reproducible across
        // git-bisect steps.
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
    }

    let stego = h264_stego_encode_yuv_string_4domain_multigop_with_pattern(
        &yuv,
        640,
        384,
        10,
        GopPattern::Ibpbp { gop: 10, b_count: 1 },
        "x",
        "phasm-psnr-diag",
    )
    .expect("phasm encode");

    let h264_path = std::env::temp_dir().join("phasm_psnr_diag.h264");
    let dec_path = std::env::temp_dir().join("phasm_psnr_diag_decoded.yuv");
    std::fs::write(&h264_path, &stego).expect("write h264");

    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate", "30", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .status()
        .expect("ffmpeg decode");
    assert!(status.success(), "ffmpeg decode failed");

    let decoded = std::fs::read(&dec_path).expect("read decoded");
    assert_eq!(decoded.len(), yuv.len(), "decoded size mismatch");

    let frame_size = 640 * 384 * 3 / 2;
    let y_size = 640 * 384;
    let mut psnr_per_frame = Vec::new();
    for f in 0..10 {
        let off = f * frame_size;
        let src_y = &yuv[off..off + y_size];
        let dec_y = &decoded[off..off + y_size];
        let p = psnr_y(src_y, dec_y);
        psnr_per_frame.push(p);
        eprintln!("frame {}: Y-PSNR = {:.2} dB", f, p);
    }
    let avg = psnr_per_frame.iter().sum::<f64>() / 10.0;
    eprintln!("avg Y-PSNR over 10 frames: {:.2} dB", avg);
    eprintln!(
        "(IBPBP encode order: I0 P2 B1 P4 B3 P6 B5 ...; B-frames = display 1, 3, 5, 7, 9)"
    );
}

// §B-direct-fix.v2 (#194): force-mode bisect at 1080p × 10f.
fn run_force_mode_check(force_mode: &str) -> usize {
    run_force_mode_check_residual(force_mode, true)
}

fn run_force_mode_check_residual(force_mode: &str, residual_enabled: bool) -> usize {
    let yuv = std::fs::read("/tmp/iphone7_1920x1072_f10.yuv")
        .expect("missing /tmp/iphone7_1920x1072_f10.yuv");

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        if residual_enabled {
            std::env::set_var("PHASM_B_RESIDUAL", "1");
        } else {
            std::env::remove_var("PHASM_B_RESIDUAL");
        }
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::set_var("PHASM_B_FORCE_MODE", force_mode);
    }

    let stego = h264_stego_encode_yuv_string_4domain_multigop_with_pattern(
        &yuv, 1920, 1072, 10,
        GopPattern::Ibpbp { gop: 10, b_count: 1 },
        "x", "phasm-force-mode",
    ).expect("phasm encode");

    let h264_path = std::env::temp_dir().join(format!("phasm_force_{force_mode}.h264"));
    let log_path = std::env::temp_dir().join(format!("phasm_force_{force_mode}.ffmpeg.log"));
    let dec_path = std::env::temp_dir().join(format!("phasm_force_{force_mode}.dec.yuv"));
    std::fs::write(&h264_path, &stego).expect("write h264");

    let log_file = std::fs::File::create(&log_path).expect("create log");
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "info", "-framerate", "30", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .stderr(log_file)
        .status()
        .expect("ffmpeg decode");

    let log = std::fs::read_to_string(&log_path).expect("read log");
    let n_conceal = log.lines().filter(|l| l.contains("concealing")).count();
    eprintln!("[force={force_mode}] concealment events: {n_conceal}");

    unsafe { std::env::remove_var("PHASM_B_FORCE_MODE"); }
    n_conceal
}

#[test]
#[ignore]
fn force_mode_bisect_1080p() {
    eprintln!("=== B-MB force-mode bisect at 1080p × 10f IBPBP RDO+residual ===");
    let modes = [
        "skip", "direct", "l0_16x16", "l1_16x16", "bi_16x16",
        "partitioned_4", "b_8x8_uniform_l0",
    ];
    let mut hits = vec![];
    for m in modes {
        let n = run_force_mode_check(m);
        if n > 0 { hits.push((m, n)); }
    }
    eprintln!("=== concealing modes (the bug surface): {hits:?} ===");
}

/// Probe whether residual-disabled + RDO-on still triggers concealment.
/// Skip and Direct have no residual emission regardless of `residual_enabled`,
/// so this isolates the bug surface to NON-residual code paths if they fail.
#[test]
#[ignore]
fn no_residual_force_mode_check() {
    eprintln!("=== B-MB modes WITHOUT residual emission at 1080p × 10f ===");
    let modes = ["skip", "direct", "l0_16x16"];
    for m in modes {
        let n = run_force_mode_check_residual(m, false);
        eprintln!("  no_residual force={m}: {n} concealments");
    }
}

/// §B-direct-fix.v2 (#194) — force=l0_16x16 at 640×384 with iPhone7
/// content. PSNR test at 640×384 RDO+residual (RDO-picked mix of
/// modes) passes; this isolates whether forcing 100% L0_16x16 with
/// real-content residual triggers the bug at small fixture too.
/// If yes, ~5sec reproducer for bin-level iteration.
#[test]
#[ignore]
fn force_l0_16x16_640x384() {
    let yuv = std::fs::read("/tmp/iphone7_640x384_f10.yuv")
        .expect("missing /tmp/iphone7_640x384_f10.yuv");
    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::set_var("PHASM_B_FORCE_MODE", "l0_16x16");
    }
    let stego = h264_stego_encode_yuv_string_4domain_multigop_with_pattern(
        &yuv, 640, 384, 10,
        GopPattern::Ibpbp { gop: 10, b_count: 1 },
        "x", "phasm-l0-640",
    ).expect("phasm encode");
    let h264_path = std::env::temp_dir().join("phasm_l0_640.h264");
    let log_path = std::env::temp_dir().join("phasm_l0_640.ffmpeg.log");
    let dec_path = std::env::temp_dir().join("phasm_l0_640.dec.yuv");
    std::fs::write(&h264_path, &stego).expect("write");
    let log_file = std::fs::File::create(&log_path).expect("log");
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "info", "-framerate", "30", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .stderr(log_file)
        .status()
        .expect("ffmpeg");
    let log = std::fs::read_to_string(&log_path).expect("read log");
    let n = log.lines().filter(|l| l.contains("concealing")).count();
    eprintln!("force=l0_16x16 @ 640×384: {n} concealments, h264 size {} bytes", stego.len());
    unsafe { std::env::remove_var("PHASM_B_FORCE_MODE"); }
}

/// Probe with PHASM_B_RDO unset entirely. Forces only Skip via env var,
/// no other B-RDO toggles. Compares against the production no-RDO path
/// (which is known clean per `ibpbp_no_rdo_1080p`). If FORCE=skip with
/// no RDO still fails, the bug is being introduced by PHASM_B_FORCE_MODE
/// itself; if it passes, the bug correlates with PHASM_B_RDO=1.
#[test]
#[ignore]
fn force_skip_no_rdo_1080p() {
    let yuv = std::fs::read("/tmp/iphone7_1920x1072_f10.yuv")
        .expect("missing /tmp/iphone7_1920x1072_f10.yuv");
    unsafe {
        std::env::remove_var("PHASM_B_RDO");
        std::env::remove_var("PHASM_B_RESIDUAL");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::set_var("PHASM_B_FORCE_MODE", "skip");
    }

    let stego = h264_stego_encode_yuv_string_4domain_multigop_with_pattern(
        &yuv, 1920, 1072, 10,
        GopPattern::Ibpbp { gop: 10, b_count: 1 },
        "x", "phasm-force-skip-norods",
    ).expect("phasm encode");

    let h264_path = std::env::temp_dir().join("phasm_force_skip_norods.h264");
    let log_path = std::env::temp_dir().join("phasm_force_skip_norods.ffmpeg.log");
    let dec_path = std::env::temp_dir().join("phasm_force_skip_norods.dec.yuv");
    std::fs::write(&h264_path, &stego).expect("write h264");

    let log_file = std::fs::File::create(&log_path).expect("create log");
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "info", "-framerate", "30", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .stderr(log_file)
        .status()
        .expect("ffmpeg decode");

    let log = std::fs::read_to_string(&log_path).expect("read log");
    let n_conceal = log.lines().filter(|l| l.contains("concealing")).count();
    eprintln!("force=skip + no RDO: {n_conceal} concealments");

    unsafe { std::env::remove_var("PHASM_B_FORCE_MODE"); }
}
