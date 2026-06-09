// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase B.3.0 — Speed × capacity × stealth benchmark.
//!
//! Sweeps rav1e's `SpeedSettings::from_preset(s)` for s ∈ {0, 4, 7, 10}
//! across a real-world content corpus (DJI drone + 4 Artlist stock +
//! 2 iPhone handheld), measuring:
//!
//!   - Pass-1 wall-clock encode time per frame
//!   - AC_COEFF_SIGN cover-bit count per frame
//!   - GOLOMB_TAIL_LSB cover-bit count per frame
//!   - Y-PSNR natural-vs-source
//!
//! Grounds B.3.2+ cost-function and performance decisions in actual
//! encoder behavior across speed settings and content classes.
//! Per `phase-b3-measure-first.md` § 1 — measurement-first design.
//!
//! # Runtime expectation
//!
//! 28 (fixture, speed) combinations × ~5 frames each at 256×144.
//! Slow path (speed 0): ~100ms / frame, so 5 frames = ~500ms.
//! Fast path (speed 10): ~30ms / frame, so 5 frames = ~150ms.
//! Total: ~30-90 seconds across the matrix on a single machine.
//!
//! # Output
//!
//! Prints a structured table to stderr (use `--nocapture` to see).
//! Each row is one (fixture, speed) combination with its measured
//! metrics. Final summary computes per-speed averages.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use phasm_core::codec::av1::stego::orchestrator::av1_stego_embed;
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
    PHASM_TAG_AC_COEFF_SIGN, PHASM_TAG_GOLOMB_TAIL_LSB,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 144;
const QUANTIZER: usize = 30;
const FRAMES_PER_FIXTURE: usize = 5;
const SPEED_LEVELS: &[u8] = &[0, 4, 7, 10];

struct Fixture {
    label: &'static str,
    source: &'static str,
    duration_s: f32,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        label: "dji_mini2",
        source: "dji_mini2_2_7k_24fps_h264_high.mp4",
        duration_s: 10.0,
    },
    Fixture {
        label: "artlist_carplane",
        source: "Artlist_CarPlane.mp4",
        duration_s: 8.0,
    },
    Fixture {
        label: "artlist_schoolfight",
        source: "Artlist_SchoolFight.mp4",
        duration_s: 8.0,
    },
    Fixture {
        label: "artlist_phonebooth",
        source: "Artlist_PhoneBooth.mp4",
        duration_s: 8.0,
    },
    Fixture {
        label: "artlist_womansubway",
        source: "Artlist_WomanSubway.mp4",
        duration_s: 8.0,
    },
    Fixture {
        label: "iphone_img4138",
        source: "IMG_4138.MOV",
        duration_s: 20.0,
    },
    Fixture {
        label: "iphone5_1080p",
        source: "iphone5_1080p_30fps_h264_high.mov",
        duration_s: 12.0,
    },
];

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv_frame(source: &str, seek_s: f32) -> Option<Vec<u8>> {
    let src = corpus_root().join(source);
    if !src.exists() {
        return None;
    }
    let vf = format!("scale={}:{}:force_original_aspect_ratio=disable", WIDTH, HEIGHT);
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args(["-frames:v", "1", "-vf", &vf, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let expected = (WIDTH * HEIGHT * 3 / 2) as usize;
    if out.stdout.len() != expected {
        return None;
    }
    Some(out.stdout)
}

#[derive(Default, Clone)]
struct CombinationStats {
    encode_ms: Vec<f64>,    // Pass-1 wall time (rav1e natural encode)
    embed_ms: Vec<f64>,     // Embed pipeline wall time (cost + STC + replay + splice)
    ac_counts: Vec<usize>,
    golomb_counts: Vec<usize>,
    y_psnr: Vec<f64>,
}

impl CombinationStats {
    fn mean(v: &[f64]) -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }
    fn mean_usize(v: &[usize]) -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
        }
    }
    fn std_usize(v: &[usize]) -> f64 {
        if v.len() < 2 {
            return 0.0;
        }
        let m = Self::mean_usize(v);
        let var = v
            .iter()
            .map(|&x| (x as f64 - m).powi(2))
            .sum::<f64>()
            / v.len() as f64;
        var.sqrt()
    }
}

fn compute_y_psnr(src: &[u8], rec: &Arc<phasm_rav1e::Frame<u8>>) -> f64 {
    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let p = &rec.planes[0];
    let stride = p.cfg.stride;
    let xorigin = p.cfg.xorigin;
    let yorigin = p.cfg.yorigin;
    let mut sse: u64 = 0;
    for y in 0..h {
        let row_start = (yorigin + y) * stride + xorigin;
        for x in 0..w {
            let src_v = src[y * w + x] as i64;
            let rec_v = p.data[row_start + x] as i64;
            let d = src_v - rec_v;
            sse += (d * d) as u64;
        }
    }
    let mse = sse as f64 / (w * h) as f64;
    if mse <= 1e-9 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// Returns (encode_ms, embed_ms, n_ac, n_golomb, y_psnr).
/// encode_ms = Pass-1 rav1e natural encode wall time.
/// embed_ms  = full embed pipeline (cost compute + STC + replay + splice).
///
/// Together these answer the B.3.0b question: at speed 10, is the
/// encoder or the cost-compute the dominant overhead? If cost compute
/// dominates, encoder swap (3.2.F) buys nothing.
fn encode_one(yuv: &[u8], speed: u8) -> Option<(f64, f64, usize, usize, f64)> {
    let mut ec = EncoderConfig::with_speed_preset(speed);
    ec.width = WIDTH as usize;
    ec.height = HEIGHT as usize;
    ec.bit_depth = 8;
    ec.chroma_sampling = ChromaSampling::Cs420;
    ec.quantizer = QUANTIZER;
    let config = Arc::new(ec);

    let mut sequence = Sequence::new(&config);
    sequence.enable_large_lru = false;
    let mut fi = FrameInvariants::<u8>::new_key_frame(
        config.clone(),
        Arc::new(sequence),
        0,
        Box::new([]),
    );
    fi.enable_segmentation = false;

    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let mut frame = make_frame::<u8>(w, h, ChromaSampling::Cs420);
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    frame.planes[0].copy_from_raw_u8(&yuv[..y_size], w, 1);
    frame.planes[1].copy_from_raw_u8(&yuv[y_size..y_size + uv_size], w / 2, 1);
    frame.planes[2].copy_from_raw_u8(&yuv[y_size + uv_size..y_size + 2 * uv_size], w / 2, 1);

    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame));
    let inter_cfg = make_inter_config(&config);

    let t0 = Instant::now();
    let (natural_bytes, recording) = encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let tile = &recording.tiles[0];
    let n_ac = tile
        .bit_tags
        .iter()
        .filter(|&&t| t == PHASM_TAG_AC_COEFF_SIGN)
        .count();
    let n_golomb = tile
        .bit_tags
        .iter()
        .filter(|&&t| t == PHASM_TAG_GOLOMB_TAIL_LSB)
        .count();
    let n_tier1 = n_ac + n_golomb;

    // Embed-pipeline timing: build a realistic payload (50 bytes) +
    // run av1_stego_embed which does cost-compute + STC + replay
    // + splice. Skip if cover is too small for the test payload.
    let embed_ms = if n_tier1 >= 256 {
        // 50 bytes = 400 bits plaintext; payload frame adds ~24 bytes
        // overhead (len+salt+nonce+CRC); ciphertext same size as
        // plaintext for GCM. Total ~592 bits in STC payload.
        let msg = vec![0xABu8; 50];
        let t1 = Instant::now();
        match av1_stego_embed(
            natural_bytes.clone(),
            recording.clone(),
            &msg,
            "b30b-bench-pass",
        ) {
            Ok(_stego) => t1.elapsed().as_secs_f64() * 1000.0,
            Err(_) => -1.0, // failed embed (e.g., StcInfeasible)
        }
    } else {
        -1.0 // too little cover for the test payload
    };

    let y_psnr = compute_y_psnr(&yuv[..y_size], &recording.reconstructed_planes);

    Some((encode_ms, embed_ms, n_ac, n_golomb, y_psnr))
}

#[test]
fn b3_0_speed_capacity_stealth_bench() {
    eprintln!("[B.3.0] Speed × capacity benchmark");
    eprintln!(
        "[B.3.0] Matrix: {} fixtures × {} speeds × {} frames = {} encodes",
        FIXTURES.len(),
        SPEED_LEVELS.len(),
        FRAMES_PER_FIXTURE,
        FIXTURES.len() * SPEED_LEVELS.len() * FRAMES_PER_FIXTURE,
    );
    eprintln!("[B.3.0] Dims: {}x{} @ QP={}", WIDTH, HEIGHT, QUANTIZER);
    eprintln!("");

    // (fixture_label, speed) -> stats
    let mut results: Vec<(String, u8, CombinationStats)> = Vec::new();
    let mut missing_fixtures: Vec<&str> = Vec::new();

    for fx in FIXTURES {
        let usable_range = fx.duration_s - 1.0;
        let step = usable_range / FRAMES_PER_FIXTURE as f32;
        let seek_points: Vec<f32> = (0..FRAMES_PER_FIXTURE)
            .map(|i| 0.5 + step * i as f32)
            .collect();

        let yuvs: Vec<Vec<u8>> = seek_points
            .iter()
            .filter_map(|&s| extract_yuv_frame(fx.source, s))
            .collect();

        if yuvs.is_empty() {
            missing_fixtures.push(fx.source);
            eprintln!("[B.3.0]   SKIP fixture (not extractable): {}", fx.source);
            continue;
        }
        if yuvs.len() < FRAMES_PER_FIXTURE {
            eprintln!(
                "[B.3.0]   PARTIAL fixture {}: got {}/{} frames",
                fx.label, yuvs.len(), FRAMES_PER_FIXTURE
            );
        }

        for &speed in SPEED_LEVELS {
            let mut stats = CombinationStats::default();
            for yuv in &yuvs {
                if let Some((enc_ms, emb_ms, n_ac, n_gol, psnr)) = encode_one(yuv, speed) {
                    stats.encode_ms.push(enc_ms);
                    if emb_ms >= 0.0 {
                        stats.embed_ms.push(emb_ms);
                    }
                    stats.ac_counts.push(n_ac);
                    stats.golomb_counts.push(n_gol);
                    stats.y_psnr.push(psnr);
                }
            }
            results.push((fx.label.to_string(), speed, stats));
        }
    }

    eprintln!("\n[B.3.0] === PER-COMBINATION RESULTS ===\n");
    eprintln!(
        "{:<22} {:>5} {:>9} {:>9} {:>11} {:>11} {:>8}",
        "fixture", "speed", "enc_ms", "embed_ms", "AC_sign", "GolombTL", "Y-PSNR"
    );
    eprintln!("{}", "-".repeat(82));

    type SpeedBucket = (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>, Vec<f64>);
    let mut by_speed: std::collections::HashMap<u8, SpeedBucket> =
        std::collections::HashMap::new();

    for (label, speed, stats) in &results {
        if stats.encode_ms.is_empty() {
            continue;
        }
        let mean_enc = CombinationStats::mean(&stats.encode_ms);
        let mean_emb = if stats.embed_ms.is_empty() {
            -1.0
        } else {
            CombinationStats::mean(&stats.embed_ms)
        };
        let ac_mean = CombinationStats::mean_usize(&stats.ac_counts);
        let ac_std = CombinationStats::std_usize(&stats.ac_counts);
        let gol_mean = CombinationStats::mean_usize(&stats.golomb_counts);
        let gol_std = CombinationStats::std_usize(&stats.golomb_counts);
        let psnr_mean = CombinationStats::mean(&stats.y_psnr);
        let emb_disp = if mean_emb < 0.0 {
            "  N/A".to_string()
        } else {
            format!("{:>7.1}", mean_emb)
        };
        eprintln!(
            "{:<22} {:>5} {:>9.1} {:>9} {:>5.0}±{:>4.0} {:>5.0}±{:>4.0} {:>8.2}",
            label, speed, mean_enc, emb_disp, ac_mean, ac_std, gol_mean, gol_std, psnr_mean
        );

        let bucket = by_speed.entry(*speed).or_default();
        bucket.0.extend_from_slice(&stats.encode_ms);
        bucket.1.extend_from_slice(&stats.embed_ms);
        bucket.2.extend_from_slice(&stats.ac_counts);
        bucket.3.extend_from_slice(&stats.golomb_counts);
        bucket.4.extend_from_slice(&stats.y_psnr);
    }

    eprintln!("\n[B.3.0] === PER-SPEED SUMMARY (all fixtures combined) ===\n");
    eprintln!(
        "{:>5} {:>9} {:>9} {:>11} {:>11} {:>8}",
        "speed", "enc_ms", "embed_ms", "AC_sign", "GolombTL", "Y-PSNR"
    );
    eprintln!("{}", "-".repeat(60));

    let mut speeds_sorted: Vec<u8> = by_speed.keys().copied().collect();
    speeds_sorted.sort();
    for speed in &speeds_sorted {
        let (enc, emb, ac, gol, psnr) = by_speed.get(speed).unwrap();
        let mean_emb = if emb.is_empty() {
            -1.0
        } else {
            CombinationStats::mean(emb)
        };
        let emb_disp = if mean_emb < 0.0 {
            "  N/A".to_string()
        } else {
            format!("{:>7.1}", mean_emb)
        };
        eprintln!(
            "{:>5} {:>9.1} {:>9} {:>11.0} {:>11.0} {:>8.2}",
            speed,
            CombinationStats::mean(enc),
            emb_disp,
            CombinationStats::mean_usize(ac),
            CombinationStats::mean_usize(gol),
            CombinationStats::mean(psnr),
        );
    }

    // Bottleneck attribution: at each speed, where does wall-clock
    // budget go (encoder vs embed pipeline)? Answers the SVT-AV1-swap
    // question: if embed dominates at speed 10, encoder swap buys 0%.
    eprintln!("\n[B.3.0b] === BOTTLENECK ATTRIBUTION ===\n");
    eprintln!(
        "{:>5} {:>9} {:>9} {:>10} {:>11}",
        "speed", "enc_ms", "embed_ms", "total_ms", "enc/total"
    );
    eprintln!("{}", "-".repeat(50));
    for speed in &speeds_sorted {
        let (enc, emb, _, _, _) = by_speed.get(speed).unwrap();
        let mean_enc = CombinationStats::mean(enc);
        let mean_emb = if emb.is_empty() {
            0.0
        } else {
            CombinationStats::mean(emb)
        };
        let total = mean_enc + mean_emb;
        let enc_frac = if total > 1e-9 { mean_enc / total } else { 0.0 };
        eprintln!(
            "{:>5} {:>9.1} {:>9.1} {:>10.1} {:>10.1}%",
            speed, mean_enc, mean_emb, total, 100.0 * enc_frac
        );
    }

    // Compute scaling factors vs speed 0 baseline.
    if let Some(speed0) = by_speed.get(&0) {
        let baseline_ms = CombinationStats::mean(&speed0.0);
        let baseline_ac = CombinationStats::mean_usize(&speed0.2);
        let baseline_gol = CombinationStats::mean_usize(&speed0.3);
        eprintln!("\n[B.3.0] === SCALING vs SPEED 0 BASELINE ===\n");
        eprintln!(
            "{:>5} {:>10} {:>11} {:>11}",
            "speed", "enc_time×", "AC%", "GolombTL%"
        );
        eprintln!("{}", "-".repeat(42));
        for speed in &[0u8, 4, 7, 10] {
            if let Some((enc, _, ac, gol, _)) = by_speed.get(speed) {
                let time_ratio = CombinationStats::mean(enc) / baseline_ms.max(1e-9);
                let ac_ratio =
                    CombinationStats::mean_usize(ac) / baseline_ac.max(1.0) * 100.0;
                let gol_ratio =
                    CombinationStats::mean_usize(gol) / baseline_gol.max(1.0) * 100.0;
                eprintln!(
                    "{:>5} {:>9.2}× {:>10.0}% {:>10.0}%",
                    speed, time_ratio, ac_ratio, gol_ratio
                );
            }
        }
    }

    if !missing_fixtures.is_empty() {
        eprintln!(
            "\n[B.3.0] NOTE: missing fixtures (not in worktree): {:?}",
            missing_fixtures
        );
    }

    // Sanity assertions: we got SOME data and SPEED 0 ≥ SPEED 10 encode time.
    assert!(!results.is_empty(), "no combinations measured");
    if let (Some(speed0), Some(speed10)) = (by_speed.get(&0), by_speed.get(&10)) {
        let t0 = CombinationStats::mean(&speed0.0);
        let t10 = CombinationStats::mean(&speed10.0);
        assert!(
            t0 >= t10,
            "speed 0 must be slower than speed 10: t0={:.1}ms t10={:.1}ms",
            t0, t10
        );
    }
}
