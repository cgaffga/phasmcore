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

/// Task #209 (2026-05-04) — fast bisect fixture. 1080p × 10 frames.
/// The bug fires at frame 1 already (first B-frame), so 10 is
/// enough. ~70 sec per run instead of 7 min.
const LARGE_F10: Fixture = Fixture {
    path: "/tmp/iphone7_1920x1072_f10.yuv",
    width: 1920,
    height: 1072,
    n_frames: 10,
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

/// Task #209 (2026-05-04) — visual-localized error metric. A handful
/// of visible speckle pixels in a 2M-pixel frame don't move whole-
/// frame PSNR enough to fail (40+ dB stays clean). Counts pixels with
/// |source − decoded| > `delta_threshold`. Real spec-correct h.264
/// decode at QP=23 should never produce per-pixel deltas > 30; the
/// v25 visible speckle has dozens of pixels with delta > 80.
fn count_bad_pixels(a: &[u8], b: &[u8], delta_threshold: u8) -> u32 {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| (**x as i32 - **y as i32).unsigned_abs() > delta_threshold as u32)
        .count() as u32
}

/// Task #209 — worst-MB PSNR. Slides a 16×16 window over the Y plane
/// and reports the minimum PSNR found at any MB position. Catches
/// localized reconstruction failures that whole-frame PSNR averages
/// out.
fn worst_mb_psnr_y(src: &[u8], dec: &[u8], width: u32, height: u32) -> (f64, u32, u32) {
    let mut worst = f64::INFINITY;
    let mut wx = 0u32;
    let mut wy = 0u32;
    let w = width as usize;
    for mb_y in 0..(height / 16) {
        for mb_x in 0..(width / 16) {
            let mut mse = 0.0_f64;
            for dy in 0..16 {
                for dx in 0..16 {
                    let off = (mb_y as usize * 16 + dy) * w + (mb_x as usize * 16 + dx);
                    let d = src[off] as f64 - dec[off] as f64;
                    mse += d * d;
                }
            }
            mse /= 256.0;
            let p = if mse == 0.0 { 99.0 } else { 10.0 * (255.0_f64 * 255.0 / mse).log10() };
            if p < worst {
                worst = p;
                wx = mb_x;
                wy = mb_y;
            }
        }
    }
    (worst, wx, wy)
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
            // Task #209 (post-#204 BRdoConfig default flip): the field
            // default is now PRODUCTION_VISUAL (RDO+residual ON).
            // Set env vars to "0" explicitly to force them OFF for the
            // no-rdo test variants.
            std::env::set_var("PHASM_B_RDO", "0");
            std::env::set_var("PHASM_B_RESIDUAL", "0");
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
    let mut worst_mb_psnrs = Vec::new();
    let mut bad_pix_counts = Vec::new();
    let mut worst_mb_locs = Vec::new();
    for f in 0..fx.n_frames {
        let off = f * frame_size;
        let p = psnr_y(&yuv[off..off + y_size], &decoded[off..off + y_size]);
        psnrs.push(p);
        let (wmb, wx, wy) = worst_mb_psnr_y(
            &yuv[off..off + y_size],
            &decoded[off..off + y_size],
            fx.width,
            fx.height,
        );
        worst_mb_psnrs.push(wmb);
        worst_mb_locs.push((wx, wy));
        let bad = count_bad_pixels(
            &yuv[off..off + y_size],
            &decoded[off..off + y_size],
            /* delta */ 50,
        );
        bad_pix_counts.push(bad);
    }
    let min = psnrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg = psnrs.iter().sum::<f64>() / fx.n_frames as f64;
    let worst_mb_min = worst_mb_psnrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let total_bad_pix = bad_pix_counts.iter().sum::<u32>();
    let max_bad_pix_per_frame = bad_pix_counts.iter().copied().max().unwrap_or(0);

    eprintln!("[{label}] avg Y-PSNR = {:.2} dB, min frame = {:.2} dB, worst-MB min = {:.2} dB", avg, min, worst_mb_min);
    eprintln!("[{label}] ffmpeg concealment events = {n_conceal}");
    eprintln!("[{label}] total |Δ|>50 pixels across {} frames: {total_bad_pix} (max/frame: {max_bad_pix_per_frame})", fx.n_frames);
    eprintln!("[{label}] per-frame Y-PSNR: {:?}", psnrs.iter().map(|p| format!("{p:.1}")).collect::<Vec<_>>());
    eprintln!("[{label}] per-frame worst-MB PSNR: {:?}", worst_mb_psnrs.iter().map(|p| format!("{p:.1}")).collect::<Vec<_>>());
    eprintln!("[{label}] per-frame |Δ|>50 count: {:?}", bad_pix_counts);
    let worst_frame_idx = worst_mb_psnrs
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let (wx, wy) = worst_mb_locs[worst_frame_idx];
    eprintln!(
        "[{label}] worst-MB hot-spot: frame {worst_frame_idx} at MB ({wx}, {wy}) → PSNR {:.2} dB",
        worst_mb_psnrs[worst_frame_idx]
    );

    assert!(
        min >= PSNR_FLOOR_DB,
        "[{label}] whole-frame Y-PSNR regression: min {min:.2} dB < floor {PSNR_FLOOR_DB} dB",
    );
    assert_eq!(
        n_conceal, 0,
        "[{label}] ffmpeg reported {n_conceal} concealment events — bitstream broken \
         (see {})", log_path.display(),
    );
    // Task #209 — worst-MB PSNR floor. Whole-frame PSNR averages out
    // localized speckle; this catches it. Real spec-correct h.264 at
    // QP=23 maintains 25+ dB on the worst MB. v25 has visible black/blue
    // pixel speckle on motion regions — those MBs PSNR ≪ 25 dB.
    assert!(
        worst_mb_min >= 25.0,
        "[{label}] localized visual regression: worst-MB Y-PSNR {worst_mb_min:.2} dB < 25 dB. \
         Hot-spot frame {worst_frame_idx} at MB ({wx}, {wy})",
    );
    // Task #209 — bad-pixel-count floor. Real spec-correct h.264 at
    // QP=23 should produce single-digit counts of pixels with |Δ|>50
    // across 60 frames (compression artifacts at high-frequency edges).
    // v25 produces 100s.
    assert!(
        max_bad_pix_per_frame < 50,
        "[{label}] localized visual regression: peak {max_bad_pix_per_frame} pixels with |source−decoded|>50 \
         on a single frame (limit 50). Per-frame counts: {bad_pix_counts:?}",
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

/// Task #209 — fast bisect (~70 sec per run). PSNR gate at f10 with
/// production-visual defaults (BRdoConfig::PRODUCTION_VISUAL via
/// build_encoder) is enough to see the bug at frame 1.
#[test]
#[ignore]
fn ibpbp_rdo_with_residual_1080p_f10() {
    run_one(
        "ibpbp_rdo_residual_1080p_f10",
        &LARGE_F10,
        GopPattern::Ibpbp { gop: 10, b_count: 1 },
        /* b_rdo */ true,
    );
}

/// Task #209 — same f10 fixture but RDO disabled (env override).
/// Bisect partner for `ibpbp_rdo_with_residual_1080p_f10`.
#[test]
#[ignore]
fn ibpbp_no_rdo_1080p_f10() {
    run_one(
        "ibpbp_no_rdo_1080p_f10",
        &LARGE_F10,
        GopPattern::Ibpbp { gop: 10, b_count: 1 },
        /* b_rdo */ false,
    );
}
