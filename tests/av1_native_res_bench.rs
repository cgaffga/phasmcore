// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase B.3.0d — Native-resolution encode + embed benchmark.
//!
//! Measures per-frame wall-clock cost (Pass-1 rav1e encode + full
//! embed pipeline) on real-world content at NATIVE RESOLUTION.
//! Replaces the linear-extrapolation guess from B.3.0 with direct
//! measurement.
//!
//! Locked at rav1e speed 10 (production target per B.3.0 finding).
//! 3 frames per fixture (smaller than B.3.0 because per-frame work
//! at 1080p+ is 50-100× larger than at 256×144).

#![cfg(all(feature = "av1-encoder", feature = "av1-backend"))]

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

const QUANTIZER: usize = 30;
const SPEED: u8 = 10;
const FRAMES_PER_FIXTURE: usize = 3;

/// One fixture entry. Width/height in source pixels (rav1e accepts
/// any even dimensions for 4:2:0). All sources confirmed to have
/// both dimensions even via ffprobe.
struct Fixture {
    label: &'static str,
    source: &'static str,
    width: u32,
    height: u32,
    duration_s: f32,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        label: "dji_mini2_2.7K",
        source: "dji_mini2_2_7k_24fps_h264_high.mp4",
        width: 2720,
        height: 1530,
        duration_s: 7.5,
    },
    Fixture {
        label: "artlist_carplane",
        source: "Artlist_CarPlane.mp4",
        width: 1080,
        height: 1920,
        duration_s: 8.0,
    },
    Fixture {
        label: "artlist_schoolfight",
        source: "Artlist_SchoolFight.mp4",
        width: 1280,
        height: 720,
        duration_s: 15.0,
    },
    Fixture {
        label: "artlist_phonebooth",
        source: "Artlist_PhoneBooth.mp4",
        width: 1928,
        height: 1072,
        duration_s: 5.0,
    },
    Fixture {
        label: "artlist_womansubway",
        source: "Artlist_WomanSubway.mp4",
        width: 1280,
        height: 720,
        duration_s: 15.0,
    },
    Fixture {
        label: "iphone_img4138",
        source: "IMG_4138.MOV",
        width: 1920,
        height: 1080,
        duration_s: 20.0,
    },
    Fixture {
        label: "iphone5_1080p",
        source: "iphone5_1080p_30fps_h264_high.mov",
        width: 1920,
        height: 1080,
        duration_s: 12.0,
    },
    Fixture {
        label: "lumix_g9_1080p",
        source: "lumix_g9_1080p_30fps_h264_high.mp4",
        width: 1920,
        height: 1080,
        duration_s: 9.5,
    },
];

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv_frame(source: &str, width: u32, height: u32, seek_s: f32) -> Option<Vec<u8>> {
    let src = corpus_root().join(source);
    if !src.exists() {
        return None;
    }
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args(["-frames:v", "1", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let expected = (width * height * 3 / 2) as usize;
    if out.stdout.len() != expected {
        eprintln!(
            "[native] {} ({}x{}): expected {} bytes got {}",
            source,
            width,
            height,
            expected,
            out.stdout.len()
        );
        return None;
    }
    Some(out.stdout)
}

fn compute_y_psnr(src: &[u8], rec: &Arc<phasm_rav1e::Frame<u8>>, w: usize, h: usize) -> f64 {
    let p = &rec.planes[0];
    let stride = p.cfg.stride;
    let xorigin = p.cfg.xorigin;
    let yorigin = p.cfg.yorigin;
    let mut sse: u64 = 0;
    for y in 0..h {
        let row_start = (yorigin + y) * stride + xorigin;
        for x in 0..w {
            let d = src[y * w + x] as i64 - p.data[row_start + x] as i64;
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

#[derive(Default, Clone)]
struct FrameStats {
    encode_ms: f64,
    embed_ms: f64,
    n_ac: usize,
    n_golomb: usize,
    y_psnr: f64,
}

fn encode_and_embed(yuv: &[u8], width: u32, height: u32) -> Option<FrameStats> {
    let mut ec = EncoderConfig::with_speed_preset(SPEED);
    ec.width = width as usize;
    ec.height = height as usize;
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

    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    let mut frame = make_frame::<u8>(w, h, ChromaSampling::Cs420);
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

    let embed_ms = if n_tier1 >= 256 {
        let msg = vec![0xABu8; 50];
        let t1 = Instant::now();
        match av1_stego_embed(natural_bytes.clone(), recording.clone(), &msg, "native-bench") {
            Ok(_) => t1.elapsed().as_secs_f64() * 1000.0,
            Err(_) => -1.0,
        }
    } else {
        -1.0
    };

    let y_psnr = compute_y_psnr(&yuv[..y_size], &recording.reconstructed_planes, w, h);

    Some(FrameStats {
        encode_ms,
        embed_ms,
        n_ac,
        n_golomb,
        y_psnr,
    })
}

#[test]
fn b3_0d_native_resolution_bench() {
    eprintln!("[B.3.0d] Native-resolution encode+embed benchmark");
    eprintln!(
        "[B.3.0d] {} fixtures × {} frames @ speed {} QP={}",
        FIXTURES.len(),
        FRAMES_PER_FIXTURE,
        SPEED,
        QUANTIZER
    );
    eprintln!("");

    eprintln!(
        "{:<24} {:>11} {:>5} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "fixture", "WxH (Mpx)", "frame", "enc_ms", "embed_ms", "AC", "GolombTL", "Y-PSNR"
    );
    eprintln!("{}", "-".repeat(92));

    let mut per_fixture_totals: Vec<(String, f64, f64, f64, f64, usize, usize, f64)> = Vec::new();
    let bench_start = Instant::now();

    for fx in FIXTURES {
        let usable = fx.duration_s - 1.0;
        let step = usable / FRAMES_PER_FIXTURE as f32;
        let mpix = (fx.width as f64 * fx.height as f64) / 1.0e6;

        let mut sum_enc = 0.0;
        let mut sum_emb = 0.0;
        let mut sum_ac = 0;
        let mut sum_gol = 0;
        let mut sum_psnr = 0.0;
        let mut n_ok = 0usize;

        for i in 0..FRAMES_PER_FIXTURE {
            let seek_s = 0.5 + step * i as f32;
            let yuv = match extract_yuv_frame(fx.source, fx.width, fx.height, seek_s) {
                Some(v) => v,
                None => {
                    eprintln!(
                        "[B.3.0d] SKIP {} frame {} (extract failed)",
                        fx.label, i
                    );
                    continue;
                }
            };
            let frame_start = Instant::now();
            let stats = match encode_and_embed(&yuv, fx.width, fx.height) {
                Some(s) => s,
                None => continue,
            };
            let _wall = frame_start.elapsed();

            eprintln!(
                "{:<24} {:>4}x{:<4} ({:>3.1}M) {:>3} {:>10.0} {:>10.0} {:>8} {:>8} {:>8.2}",
                fx.label,
                fx.width,
                fx.height,
                mpix,
                i,
                stats.encode_ms,
                if stats.embed_ms < 0.0 { -1.0 } else { stats.embed_ms },
                stats.n_ac,
                stats.n_golomb,
                stats.y_psnr,
            );

            sum_enc += stats.encode_ms;
            if stats.embed_ms >= 0.0 {
                sum_emb += stats.embed_ms;
            }
            sum_ac += stats.n_ac;
            sum_gol += stats.n_golomb;
            sum_psnr += stats.y_psnr;
            n_ok += 1;
        }

        if n_ok > 0 {
            let n = n_ok as f64;
            per_fixture_totals.push((
                fx.label.to_string(),
                mpix,
                sum_enc / n,
                sum_emb / n,
                sum_psnr / n,
                sum_ac / n_ok,
                sum_gol / n_ok,
                fx.width as f64 * fx.height as f64,
            ));
        }
    }

    let bench_dur = bench_start.elapsed();
    eprintln!("\n[B.3.0d] Bench wall time: {:.1} sec", bench_dur.as_secs_f64());

    eprintln!("\n[B.3.0d] === PER-FIXTURE MEAN ===\n");
    eprintln!(
        "{:<24} {:>9} {:>10} {:>10} {:>11} {:>10} {:>8}",
        "fixture", "Mpix", "enc_ms", "embed_ms", "total_ms", "AC+Gol", "Y-PSNR"
    );
    eprintln!("{}", "-".repeat(86));
    for (label, mpix, enc, emb, psnr, ac, gol, _pxs) in &per_fixture_totals {
        eprintln!(
            "{:<24} {:>9.2} {:>10.0} {:>10.0} {:>11.0} {:>10} {:>8.2}",
            label,
            mpix,
            enc,
            emb,
            enc + emb,
            ac + gol,
            psnr
        );
    }

    // Scaling analysis: encode_ms / Mpix and embed_ms / Mpix tell
    // us per-megapixel cost. If embed_ms / Mpix is constant across
    // fixtures, embed scales linearly with pixel count. If it grows
    // with resolution, embed is super-linear.
    eprintln!("\n[B.3.0d] === PER-MEGAPIXEL COST ===\n");
    eprintln!(
        "{:<24} {:>9} {:>14} {:>15}",
        "fixture", "Mpix", "enc_ms / Mpix", "embed_ms / Mpix"
    );
    eprintln!("{}", "-".repeat(64));
    for (label, mpix, enc, emb, _psnr, _ac, _gol, _pxs) in &per_fixture_totals {
        let enc_per_mpix = enc / mpix.max(1e-9);
        let emb_per_mpix = emb / mpix.max(1e-9);
        eprintln!(
            "{:<24} {:>9.2} {:>14.0} {:>15.0}",
            label, mpix, enc_per_mpix, emb_per_mpix
        );
    }

    // 5-minute @ 30fps projection per fixture.
    eprintln!("\n[B.3.0d] === 5-MIN @ 30fps PROJECTION ===\n");
    eprintln!(
        "{:<24} {:>11} {:>14} {:>14}",
        "fixture", "ms/frame", "9000 frames", "as hours"
    );
    eprintln!("{}", "-".repeat(67));
    for (label, _mpix, enc, emb, _psnr, _ac, _gol, _pxs) in &per_fixture_totals {
        let per_frame = enc + emb;
        let total_sec = (per_frame / 1000.0) * 9000.0;
        let total_hours = total_sec / 3600.0;
        eprintln!(
            "{:<24} {:>11.0} {:>11.0} sec {:>11.1} hr",
            label, per_frame, total_sec, total_hours
        );
    }

    // Sanity: we got at least 3 fixtures' data.
    assert!(
        per_fixture_totals.len() >= 3,
        "expected ≥ 3 fixtures with data, got {}",
        per_fixture_totals.len()
    );
}
