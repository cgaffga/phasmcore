// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.2 (#401-#404) — visual + PSNR + SSIM ship gate on OH264 backend.
//
//   C.2.1 (#402) — openh264_visual_psnr_regression: one #[test] per
//     fixture so cargo's parallel runner can shard the corpus.
//   C.2.2 (#403) — SSIM measurement alongside PSNR.
//
// Mirrors the shape of `h264_visual_psnr_regression.rs` (pure-Rust)
// but encodes via the OH264 backend. Per fixture:
//   1. Scale corpus MP4 → YUV (cached under `$PHASM_TMP`).
//   2. Encode via `openh264_stego_encode_yuv_text` with a fixed
//      passphrase + message.
//   3. Decode the stego Annex-B via ffmpeg subprocess.
//   4. Compute four quality metrics over the luma plane:
//        - per-frame Y-PSNR (whole-frame)
//        - per-frame Y-SSIM (8×8 sliding window, non-overlapping)
//        - per-frame worst-MB PSNR (catches localized speckle)
//        - per-frame bad-pixel count (|src−dec| > 50)
//   5. Assert against thresholds tuned to leave ~2-3 dB headroom for
//      normal encoder evolution.
//
// Smoke variant `visual_oh264_smoke_iphone7` runs by default at
// 640×368 × 8 frames. Production variants are `#[ignore]` (1080p × 30).
//
// Run full corpus:
//   cargo test --release --features "h264-encoder" \
//     --test openh264_visual_psnr_regression -- --ignored --nocapture
//
// Why per-fixture #[test]: cargo's parallel runner spreads load, and
// per-fixture pass/fail signal localizes regressions (vs a single
// big test that aborts on the first failing fixture). The OH264
// SessionGuard mutex serializes the actual OH264 calls; per-test
// cargo overhead is negligible.

#![cfg(feature = "h264-encoder")]

mod common;
use common::oh264_stream;

// Fixture/loader scaffolding lives in `common::video_fixture` so this gate and
// `oh264_quality_vs_fill.rs` share ONE loader + ONE ffmpeg-decode step. A second
// copy is how two measurements of "the same thing" start disagreeing — the same
// reason the metric kernels were centralised in `common::visual_metrics`.
use common::video_fixture::{
    ensure_yuv, ffmpeg_decode_annex_b, probe_aligned_dims, session_guard,
    VideoFixture as Fixture,
};

// ─────────────────────────── metrics ──────────────────────────────────
// ─── metrics ───
//
// Moved to `common::visual_metrics` so the allocation-shape fidelity
// comparison measures with the SAME code, not a second copy that drifts.
use common::visual_metrics::{count_bad_pixels, psnr_y, ssim_y, worst_mb_psnr_y};

/// Whole-frame Y-PSNR floor. Re-baselined for the production 4-domain
/// streaming path (video-retirement Phase 6). The retired single-domain
/// one-shot embedded in CoeffSign ONLY and measured ~30 dB on the smoke
/// fixture; the production streaming session spreads flips across all 4
/// stego domains (CS → CSL → MVDs → MVDsl, incl. higher-visual-cost MVD
/// signs), so the same fixture now measures ~25.4 dB (min 25.36, SSIM
/// 0.93, 0 concealment, worst-MB 17 dB). Whole-frame PSNR is only the
/// coarse guard here — local visual correctness is enforced by the SSIM
/// (≥0.90), worst-MB-PSNR (≥10 dB), bad-pixel-ceiling, and zero-
/// concealment gates below, all of which still pass with margin. Floor
/// at 24 dB leaves ~1.4 dB headroom on the smoke fixture.
const PSNR_FLOOR_DB: f64 = 24.0;

/// Worst-MB PSNR floor. The worst 16×16 MB in any frame is dominated
/// by high-frequency content (motion blur, edge detail) at production
/// QPs — observed 16-20 dB on natural iPhone footage at QP=26, dipping
/// to 13 dB on CarPlane fast-motion. Cascade leaks and visible-speckle
/// bugs drop this below ~8 dB. Floor at 10 dB catches the bug class
/// without false-positives on legitimate high-frequency content.
const WORST_MB_PSNR_FLOOR_DB: f64 = 10.0;

/// Per-pixel delta threshold for the "bad pixel" metric. At >80 the
/// metric is bug-discriminating: clean OH264 at QP=26 produces
/// single-digit counts even on textured/motion content, while the
/// visible-speckle bug class produces hundreds (each speckle pixel
/// deviates 80+ from source). At ≤50, normal high-frequency content
/// floods the count and the metric loses its signal.
const BAD_PIXEL_DELTA: u8 = 80;

/// Bad-pixel ceiling per frame at `BAD_PIXEL_DELTA`. Clean OH264 +
/// visual_recon produces near-zero; cascade leaks produce 100s.
/// Ceiling 100 leaves comfortable margin for content variation.
const BAD_PIXELS_CEILING_PER_FRAME: u32 = 100;

/// SSIM floor. Wang+2004 SSIM at QP=26 on natural content typically
/// scores 0.94+. Drop below 0.90 indicates a real perceptual artifact.
const SSIM_FLOOR: f64 = 0.90;

// ─────────────────────────── core harness ─────────────────────────────

fn run_metrics(spec: &Fixture, msg: &str, pass: &str) {
    // Recover from poisoning so a single failing fixture doesn't
    // cascade-fail the rest of the corpus — failures are independent
    // measurements; we want to see all of them, not just the first.
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let (w, h) = probe_aligned_dims(spec);
    let yuv = ensure_yuv(spec, w, h);

    let t_enc = std::time::Instant::now();
    let stego = oh264_stream::encode(
        &yuv, w, h, spec.n_frames, spec.qp, msg, pass,
    )
    .expect("oh264 stego encode");
    let enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

    // Decode via ffmpeg subprocess for an independent reference. Intermediates
    // land under `$PHASM_TMP` (`common::scratch_dir`), not `std::env::temp_dir()`
    // — the earlier `temp_dir()` use put hundreds of MB on this machine's
    // internal disk, which is the scarce one (CLAUDE.md "Location & scratch
    // files").
    let clip = ffmpeg_decode_annex_b(&stego, &format!("psnr_{}", spec.name));
    let n_conceal = clip.n_conceal;
    let log_path = clip.log_path.clone();
    let decoded = clip.yuv;
    assert_eq!(
        decoded.len(),
        yuv.len(),
        "decoded size mismatch for {} ({} vs {})",
        spec.name, decoded.len(), yuv.len()
    );

    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w * h) as usize;
    let mut psnrs = Vec::new();
    let mut ssims = Vec::new();
    let mut worst_mb_psnrs = Vec::new();
    let mut bad_pix_counts = Vec::new();
    let mut worst_mb_locs = Vec::new();
    for f in 0..spec.n_frames as usize {
        let off = f * frame_size;
        let src_y = &yuv[off..off + y_size];
        let dec_y = &decoded[off..off + y_size];
        psnrs.push(psnr_y(src_y, dec_y));
        ssims.push(ssim_y(src_y, dec_y, w, h));
        let (wmb, wx, wy) = worst_mb_psnr_y(src_y, dec_y, w, h);
        worst_mb_psnrs.push(wmb);
        worst_mb_locs.push((wx, wy));
        bad_pix_counts.push(count_bad_pixels(src_y, dec_y, BAD_PIXEL_DELTA));
    }

    let psnr_min = psnrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let psnr_avg = psnrs.iter().sum::<f64>() / spec.n_frames as f64;
    let ssim_min = ssims.iter().cloned().fold(f64::INFINITY, f64::min);
    let ssim_avg = ssims.iter().sum::<f64>() / spec.n_frames as f64;
    let worst_mb_min = worst_mb_psnrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_bad_pix = bad_pix_counts.iter().copied().max().unwrap_or(0);

    eprintln!(
        "[{name}] {w}×{h}×{n} qp={qp}: enc={enc:.0}ms  Y-PSNR avg={pa:.2} min={pm:.2}  \
         Y-SSIM avg={sa:.4} min={sm:.4}  worst-MB={mb:.2}dB  max bad/frame={bp}",
        name = spec.name, w = w, h = h, n = spec.n_frames, qp = spec.qp,
        enc = enc_ms, pa = psnr_avg, pm = psnr_min, sa = ssim_avg, sm = ssim_min,
        mb = worst_mb_min, bp = max_bad_pix,
    );

    // Per-frame breakdown only on failure or when verbose.
    let worst_frame = worst_mb_psnrs
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let (wx, wy) = worst_mb_locs[worst_frame];

    assert_eq!(
        n_conceal, 0,
        "[{}] ffmpeg reported {} concealment events — bitstream broken (see {})",
        spec.name, n_conceal, log_path,
    );
    assert!(
        psnr_min >= PSNR_FLOOR_DB,
        "[{}] Y-PSNR regression: min {:.2} dB < floor {} dB (per-frame: {:?})",
        spec.name, psnr_min, PSNR_FLOOR_DB,
        psnrs.iter().map(|p| format!("{:.1}", p)).collect::<Vec<_>>(),
    );
    assert!(
        ssim_min >= SSIM_FLOOR,
        "[{}] SSIM regression: min {:.4} < floor {:.4} (per-frame: {:?})",
        spec.name, ssim_min, SSIM_FLOOR,
        ssims.iter().map(|s| format!("{:.3}", s)).collect::<Vec<_>>(),
    );
    assert!(
        worst_mb_min >= WORST_MB_PSNR_FLOOR_DB,
        "[{}] localized regression: worst-MB Y-PSNR {:.2} dB < {} dB. \
         Hot-spot frame {} at MB ({}, {}).",
        spec.name, worst_mb_min, WORST_MB_PSNR_FLOOR_DB, worst_frame, wx, wy,
    );
    assert!(
        max_bad_pix < BAD_PIXELS_CEILING_PER_FRAME,
        "[{}] localized regression: peak {} pixels with |Δ|>{} on a single frame \
         (limit {}). Per-frame: {:?}",
        spec.name, max_bad_pix, BAD_PIXEL_DELTA, BAD_PIXELS_CEILING_PER_FRAME, bad_pix_counts,
    );
}

// ─────────────────────────── smoke ────────────────────────────────────

/// Default-CI smoke gate. Runs in `cargo test` without --ignored.
/// Single fixture at 640×360 × 8 frames so the whole thing finishes
/// in <10 s including ffmpeg scale + decode.
#[test]
fn visual_oh264_smoke_iphone7() {
    let spec = Fixture {
        name: "iphone7_smoke",
        source_mp4: "IMG_4138.MOV",
        long_side_cap: Some(640),
        n_frames: 8,
        qp: 26,
    };
    run_metrics(&spec, "phasm c2 smoke", "smoke-pass");
}

// ─────────────────────────── production ───────────────────────────────

const PROD_LONG: u32 = 1920;
const PROD_N: u32 = 30;
const PROD_QP: i32 = 26;

#[test]
#[ignore]
fn visual_oh264_iphone7_1080p() {
    run_metrics(
        &Fixture {
            name: "iphone7_1080p",
            source_mp4: "IMG_4138.MOV",
            long_side_cap: Some(PROD_LONG), n_frames: PROD_N, qp: PROD_QP,
        },
        "C.2 visual gate — iPhone 7", "gate-pass",
    );
}

#[test]
#[ignore]
fn visual_oh264_carplane_1080p() {
    run_metrics(
        &Fixture {
            name: "carplane_1080p",
            source_mp4: "Artlist_CarPlane.mp4",
            long_side_cap: Some(PROD_LONG), n_frames: PROD_N, qp: PROD_QP,
        },
        "C.2 visual gate — CarPlane fast-motion", "gate-pass",
    );
}

#[test]
#[ignore]
fn visual_oh264_schoolfight_1080p() {
    run_metrics(
        &Fixture {
            name: "schoolfight_1080p",
            source_mp4: "Artlist_SchoolFight.mp4",
            long_side_cap: Some(PROD_LONG), n_frames: PROD_N, qp: PROD_QP,
        },
        "C.2 visual gate — SchoolFight textured", "gate-pass",
    );
}

#[test]
#[ignore]
fn visual_oh264_asia_bottle_1080p() {
    run_metrics(
        &Fixture {
            name: "asia_bottle_1080p",
            source_mp4: "Artlist_AsiaBottle.mp4",
            long_side_cap: Some(PROD_LONG), n_frames: PROD_N, qp: PROD_QP,
        },
        "C.2 visual gate — AsiaBottle low-motion", "gate-pass",
    );
}

#[test]
#[ignore]
fn visual_oh264_dji_mini2_1080p() {
    run_metrics(
        &Fixture {
            name: "dji_mini2_1080p",
            source_mp4: "dji_mini2_2_7k_24fps_h264_high.mp4",
            long_side_cap: Some(PROD_LONG), n_frames: PROD_N, qp: PROD_QP,
        },
        "C.2 visual gate — DJI Mini2 aerial", "gate-pass",
    );
}
