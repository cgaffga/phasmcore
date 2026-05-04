// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §B-direct-fix #187 — automated visual PSNR regression test.
//
// Encodes a real 640×384 iPhone7 fixture via the production stego
// pipeline, decodes via ffmpeg, computes per-frame Y-PSNR vs source,
// asserts the average exceeds a threshold. Catches the class of
// encoder/spec-decoder visual divergence that the 2026-05-03
// iPhone7 visual bug uncovered (where 1291 round-trip lib tests
// passed but ffmpeg-decoded output was scrambled).
//
// Uses `PHASM_DETERMINISTIC_SEED` so the test is reproducible
// across runs (no random crypto salt/nonce → byte-identical output
// → byte-identical PSNR).
//
// `#[ignore]` because it depends on `/tmp/iphone7_640x384_f10.yuv`
// (regen via README) + ~5 sec of ffmpeg work. Run nightly / on
// release branch:
//
//   cargo test --release --features cabac-stego \
//     --test h264_visual_psnr_regression -- --ignored --nocapture

#![cfg(feature = "cabac-stego")]

use phasm_core::{
    h264_stego_encode_yuv_string_4domain_multigop_with_pattern, GopPattern,
};

struct Fixture {
    path: &'static str,
    width: u32,
    height: u32,
    n_frames: usize,
}

const SMALL: Fixture = Fixture {
    path: "/tmp/iphone7_640x384_f10.yuv",
    width: 640,
    height: 384,
    n_frames: 10,
};

const LARGE: Fixture = Fixture {
    path: "/tmp/iphone7_1920x1072_f30.yuv",
    width: 1920,
    height: 1072,
    n_frames: 30,
};

/// Task #209 (2026-05-04) — 60-frame extension of LARGE. The 30f
/// fixture happens to fall short of the headstand motion that
/// triggers the v25 visible speckle bug. f60 covers the full motion
/// arc; the bug fires here when f30 is silent.
const LARGE_F60: Fixture = Fixture {
    path: "/tmp/iphone7_1920x1072_f60.yuv",
    width: 1920,
    height: 1072,
    n_frames: 60,
};

/// Minimum acceptable per-frame Y-PSNR. Below this indicates an
/// encoder/spec-decoder divergence (the class of bug fixed in
/// commit 5b85432). Healthy phasm output on this fixture should
/// exceed 24 dB on every frame; we set the gate at 22 dB to leave
/// ~2 dB headroom for normal encoder evolution.
const PSNR_FLOOR_DB: f64 = 22.0;

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

fn run_one(label: &str, fx: &Fixture, pattern: GopPattern, b_rdo: bool) {
    let yuv = std::fs::read(fx.path)
        .unwrap_or_else(|_| panic!("missing fixture: {}", fx.path));
    assert_eq!(
        yuv.len(),
        (fx.width * fx.height * 3 / 2) as usize * fx.n_frames,
        "fixture size mismatch"
    );

    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        if b_rdo {
            std::env::set_var("PHASM_B_RDO", "1");
            std::env::set_var("PHASM_B_RESIDUAL", "1");
        } else {
            std::env::remove_var("PHASM_B_RDO");
            std::env::remove_var("PHASM_B_RESIDUAL");
        }
    }

    let stego = h264_stego_encode_yuv_string_4domain_multigop_with_pattern(
        &yuv, fx.width, fx.height, fx.n_frames, pattern, "x", "phasm-psnr-regression",
    )
    .expect("phasm encode");

    let h264_path = std::env::temp_dir().join(format!("phasm_psnr_{label}.h264"));
    let dec_path = std::env::temp_dir().join(format!("phasm_psnr_{label}.dec.yuv"));
    let log_path = std::env::temp_dir().join(format!("phasm_psnr_{label}.ffmpeg.log"));
    std::fs::write(&h264_path, &stego).expect("write h264");

    let log_file = std::fs::File::create(&log_path).expect("create log");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "info", "-framerate", "30", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .stderr(log_file)
        .status()
        .expect("ffmpeg decode");
    assert!(status.success(), "ffmpeg decode failed for {label}");

    // §B-direct-fix.v2: ffmpeg's CABAC concealment path is silent at
    // -loglevel error. With -loglevel info it logs one line per
    // broken slice. Parse the log; any concealment is a regression.
    let log = std::fs::read_to_string(&log_path).expect("read ffmpeg log");
    let n_conceal = log.lines().filter(|l| l.contains("concealing")).count();

    let decoded = std::fs::read(&dec_path).expect("read decoded");
    assert_eq!(decoded.len(), yuv.len(), "decoded size mismatch for {label}");

    let frame_size = (fx.width * fx.height * 3 / 2) as usize;
    let y_size = (fx.width * fx.height) as usize;
    let mut psnrs = Vec::new();
    for f in 0..fx.n_frames {
        let off = f * frame_size;
        let p = psnr_y(&yuv[off..off + y_size], &decoded[off..off + y_size]);
        psnrs.push(p);
    }
    let min = psnrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg = psnrs.iter().sum::<f64>() / fx.n_frames as f64;

    eprintln!("[{label}] avg Y-PSNR = {:.2} dB, min = {:.2} dB, ffmpeg concealment events = {n_conceal}", avg, min);
    eprintln!("[{label}] per-frame: {:?}", psnrs.iter().map(|p| format!("{p:.1}")).collect::<Vec<_>>());

    assert!(
        min >= PSNR_FLOOR_DB,
        "[{label}] visual regression: min Y-PSNR {min:.2} dB < floor {PSNR_FLOOR_DB} dB. \
         Per-frame: {:?}",
        psnrs
    );
    assert_eq!(
        n_conceal, 0,
        "[{label}] ffmpeg reported {n_conceal} concealment events — bitstream broken \
         (see {})", log_path.display(),
    );
}

#[test]
#[ignore]
fn ipppp_no_rdo() {
    // Baseline: no B-frames at all. Should always be the cleanest.
    run_one("ipppp", &SMALL, GopPattern::Ipppp { gop: 10 }, /* b_rdo */ false);
}

#[test]
#[ignore]
fn ibpbp_no_rdo() {
    // Production default GopPattern with no-RDO Skip/Direct fallback
    // for B-MBs (post commit 5b85432).
    run_one("ibpbp_no_rdo", &SMALL, GopPattern::Ibpbp { gop: 10, b_count: 1 }, /* b_rdo */ false);
}

#[test]
#[ignore]
fn ibpbp_rdo_with_residual() {
    // Stealth-calibrated path: PHASM_B_RDO=1 + PHASM_B_RESIDUAL=1
    // exercises §6E-D.5(m+o) + §6E-A6.1q.b residual emission.
    run_one(
        "ibpbp_rdo_residual",
        &SMALL,
        GopPattern::Ibpbp { gop: 10, b_count: 1 },
        /* b_rdo */ true,
    );
}

/// §B-direct-fix.v2 (#194): 1080p × 30f IBPBP no-RDO is the production
/// shipping configuration. Verifies the path the iPhone7 demo uses and
/// catches regressions that a 640×384 fixture misses (the bug whose
/// existence prompted this test class). Should always be clean — if
/// THIS regresses, the v1.0 video stego ship config is broken.
#[test]
#[ignore]
fn ibpbp_no_rdo_1080p() {
    run_one(
        "ibpbp_no_rdo_1080p",
        &LARGE,
        GopPattern::Ibpbp { gop: 30, b_count: 1 },
        /* b_rdo */ false,
    );
}

/// §B-direct-fix.v2 (#194) — 1080p × 30f IBPBP RDO+residual.
/// Pre-fix: 14 concealment events (one per B-frame). Post-fix:
/// 0. Locks the §6E-D.5(m+o) calibrated mode-distribution path
/// against future regressions.
#[test]
#[ignore]
fn ibpbp_rdo_with_residual_1080p() {
    run_one(
        "ibpbp_rdo_residual_1080p",
        &LARGE,
        GopPattern::Ibpbp { gop: 30, b_count: 1 },
        /* b_rdo */ true,
    );
}

/// Task #209 (2026-05-04) — 60-frame variant. The user-visible v25
/// bug (flickering triangles + squares on the moving subject) shows
/// up here even though the 30f variant is clean (35-43 dB / 0
/// concealment). The motion arc the bug fires on is in frames 30-59.
#[test]
#[ignore]
fn ibpbp_rdo_with_residual_1080p_f60() {
    run_one(
        "ibpbp_rdo_residual_1080p_f60",
        &LARGE_F60,
        GopPattern::Ibpbp { gop: 30, b_count: 1 },
        /* b_rdo */ true,
    );
}

/// Task #209 — same fixture but RDO disabled. If THIS passes and
/// the RDO variant fails, the bug is in B-RDO+RESIDUAL specifically.
#[test]
#[ignore]
fn ibpbp_no_rdo_1080p_f60() {
    run_one(
        "ibpbp_no_rdo_1080p_f60",
        &LARGE_F60,
        GopPattern::Ibpbp { gop: 30, b_count: 1 },
        /* b_rdo */ false,
    );
}
