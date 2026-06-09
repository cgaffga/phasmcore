// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! B.1.5.0 — Cascade measurement spike.
//!
//! For each W4 fixture: encode natural, then for N random AC_SIGN
//! positions, individually flip a single position and measure the
//! resulting per-pixel YUV delta. Output: per-position cascade
//! magnitude (max, mean, MSE, divergence count) + local context
//! score (smoothness derived from natural-frame std-dev around the
//! affected pixel block).
//!
//! **Hypothesis under test:** cascade magnitude varies widely across
//! positions; smooth-region flips have small cascade, textured-region
//! flips have large cascade. If confirmed, this validates the per-
//! position cascade-cost approach in [phase-b15-cascade-safety-v2.md]
//! and provides the threshold calibration data the cost-term flip in
//! B.1.5.5 needs.
//!
//! **Pure measurement.** Does not modify cost compute or commit any
//! side effects. Run with `cargo test ... -- --ignored --nocapture`
//! to inspect output.
//!
//! See [`phase-b15-cascade-safety-v2.md`](../../docs/design/video/av1/phase-b15-cascade-safety-v2.md)
//! § 3 (sub-phase plan) and § 7 (threshold tuning approach).

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use phasm_core::codec::av1::stego::writer::{
    replay_with_overrides, CoverPositions, OverrideMap,
};
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::ec::{PhasmFrameRecording, WriterEncoder};
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
    PHASM_TAG_AC_COEFF_SIGN,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Number of random AC_SIGN positions probed per fixture. 50 keeps
/// total wall-clock under ~2 minutes (each probe ≈ 100 ms decode +
/// ~ms replay).
const N_POSITIONS_PER_FIXTURE: usize = 50;

/// Pixel-divergence threshold for "this pixel was affected by the
/// cascade." 5 luma levels is conservatively above noise floor but
/// catches real cascade impact.
const DIVERGENCE_THRESHOLD: u8 = 5;

/// Seed for reproducible random sampling across runs. Change to
/// resample if the corpus structure changes; otherwise leave fixed
/// so threshold calibration in B.1.5.5 references the same positions.
const SAMPLE_SEED: u64 = 0xB150_0001;

struct Fixture {
    name: &'static str,
    source: &'static str,
    width: u32,
    height: u32,
    seek_s: f32,
    quantizer: usize,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "iphone_img4138",
        source: "IMG_4138.MOV",
        width: 256,
        height: 144,
        seek_s: 1.0,
        quantizer: 30,
    },
    Fixture {
        name: "carplane",
        source: "Artlist_CarPlane.mp4",
        width: 144,
        height: 256,
        seek_s: 2.0,
        quantizer: 30,
    },
    Fixture {
        name: "iphone5_1080p",
        source: "iphone5_1080p_30fps_h264_high.mov",
        width: 256,
        height: 144,
        seek_s: 1.0,
        quantizer: 30,
    },
];

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv420_frame(spec: &Fixture) -> Vec<u8> {
    let src = corpus_root().join(spec.source);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!(
        "scale={}:{}:force_original_aspect_ratio=disable",
        spec.width, spec.height
    );
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(spec.seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args([
            "-frames:v", "1", "-vf", &vf, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
        ])
        .output()
        .expect("ffmpeg launch");
    assert!(out.status.success(), "ffmpeg yuv extract failed");
    out.stdout
}

fn encode_natural_with_recording(
    yuv: &[u8],
    spec: &Fixture,
) -> (Vec<u8>, PhasmFrameRecording<u8>) {
    let config = Arc::new(EncoderConfig {
        width: spec.width as usize,
        height: spec.height as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: spec.quantizer,
        ..Default::default()
    });
    let mut sequence = Sequence::new(&config);
    sequence.enable_large_lru = false;
    let mut fi = FrameInvariants::<u8>::new_key_frame(
        config.clone(),
        Arc::new(sequence),
        0,
        Box::new([]),
    );
    fi.enable_segmentation = false;
    let mut frame =
        make_frame::<u8>(spec.width as usize, spec.height as usize, ChromaSampling::Cs420);
    let w = spec.width as usize;
    let h = spec.height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    frame.planes[0].copy_from_raw_u8(&yuv[..y_size], w, 1);
    frame.planes[1].copy_from_raw_u8(&yuv[y_size..y_size + uv_size], w / 2, 1);
    frame.planes[2].copy_from_raw_u8(
        &yuv[y_size + uv_size..y_size + 2 * uv_size],
        w / 2,
        1,
    );
    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame));
    let inter_cfg = make_inter_config(&config);
    encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg)
}

fn build_ivf_single_frame(obus: &[u8], width: u16, height: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + 12 + obus.len());
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"AV01");
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(obus.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(obus);
    out
}

fn decode_av1_to_yuv(av1_bytes: &[u8], width: u32, height: u32, label: &str) -> Vec<u8> {
    let ivf = build_ivf_single_frame(av1_bytes, width as u16, height as u16);
    let ivf_path = std::env::temp_dir().join(format!(
        "b150_decode_{}_{}.ivf",
        label,
        std::process::id()
    ));
    std::fs::write(&ivf_path, &ivf).expect("write ivf");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&ivf_path)
        .args(["-frames:v", "1", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output()
        .expect("ffmpeg decode launch");
    std::fs::remove_file(&ivf_path).ok();
    assert!(out.status.success(), "ffmpeg decode failed");
    out.stdout
}

/// Build a stego packet that flips exactly ONE bit at the given
/// cover-position cursor. Returns None if the resulting tile group
/// length differs from natural (rare ~5% range-coder trailing-carry
/// case — orchestrator would normally rebuild the OBU, but for the
/// spike we skip those positions to keep the measurement clean).
fn flip_one_and_make_packet(
    natural_packet: &[u8],
    recording: &PhasmFrameRecording<u8>,
    cursor: u64,
    natural_bit: u16,
) -> Option<Vec<u8>> {
    let tile = &recording.tiles[0];
    let mut plan = OverrideMap::new();
    plan.set(cursor, 1 - natural_bit);

    let mut sink = WriterEncoder::new();
    replay_with_overrides(&tile.storage, &tile.bit_positions, &plan, &mut sink);
    let stego_tile_bytes = sink.done();

    if stego_tile_bytes.len() != recording.tile_group_len {
        return None;
    }
    let mut packet = natural_packet.to_vec();
    let dst = &mut packet[recording.tile_group_offset
        ..recording.tile_group_offset + recording.tile_group_len];
    dst.copy_from_slice(&stego_tile_bytes);
    Some(packet)
}

/// Per-pixel std-dev in a `window×window` patch of the luma plane
/// centered at (cx, cy). Higher = textured, lower = smooth.
fn local_smoothness(y_plane: &[u8], width: usize, height: usize, cx: usize, cy: usize, window: usize) -> f64 {
    let half = window / 2;
    let x0 = cx.saturating_sub(half).min(width.saturating_sub(1));
    let y0 = cy.saturating_sub(half).min(height.saturating_sub(1));
    let x1 = (cx + half).min(width);
    let y1 = (cy + half).min(height);
    let mut n = 0u64;
    let mut s = 0u64;
    let mut s2 = 0u64;
    for y in y0..y1 {
        for x in x0..x1 {
            let v = y_plane[y * width + x] as u64;
            n += 1;
            s += v;
            s2 += v * v;
        }
    }
    if n == 0 {
        return 0.0;
    }
    let mean = s as f64 / n as f64;
    let var = (s2 as f64 / n as f64) - mean * mean;
    var.max(0.0).sqrt()
}

#[derive(Debug, Clone, Copy)]
struct CascadeStats {
    max_abs: u8,
    mean_abs: f64,
    mse: f64,
    pixels_diverged: usize,
}

fn measure_cascade(natural_yuv: &[u8], stego_yuv: &[u8], width: u32, height: u32) -> CascadeStats {
    let y_size = (width * height) as usize;
    let mut max_abs: u8 = 0;
    let mut sum_abs: u64 = 0;
    let mut sum_sq: u64 = 0;
    let mut diverged: usize = 0;
    for i in 0..y_size {
        let n = natural_yuv[i] as i32;
        let s = stego_yuv[i] as i32;
        let d = (n - s).unsigned_abs() as u8;
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d as u64;
        sum_sq += (d as u64) * (d as u64);
        if d > DIVERGENCE_THRESHOLD {
            diverged += 1;
        }
    }
    CascadeStats {
        max_abs,
        mean_abs: sum_abs as f64 / y_size as f64,
        mse: sum_sq as f64 / y_size as f64,
        pixels_diverged: diverged,
    }
}

/// Per-position record: (idx, smoothness, coeff_magnitude, cascade_stats).
type SpikeRow = (usize, f64, u16, CascadeStats);

fn run_spike_for_fixture(spec: &Fixture) -> Vec<SpikeRow> {
    eprintln!(
        "[B.1.5.0] ============ {} ({}×{} q={}) ============",
        spec.name, spec.width, spec.height, spec.quantizer
    );
    let source_yuv = extract_yuv420_frame(spec);
    let (natural_packet, recording) = encode_natural_with_recording(&source_yuv, spec);
    let natural_yuv = decode_av1_to_yuv(&natural_packet, spec.width, spec.height, "natural");
    eprintln!(
        "[B.1.5.0] natural packet = {} bytes, decoded YUV = {} bytes",
        natural_packet.len(),
        natural_yuv.len()
    );

    let tile = &recording.tiles[0];
    let positions = CoverPositions::from_recorder(&tile.bit_positions, &tile.bit_tags);
    let ac_positions: Vec<_> = positions
        .iter()
        .filter(|p| p.tag == PHASM_TAG_AC_COEFF_SIGN)
        .copied()
        .collect();
    eprintln!(
        "[B.1.5.0] cover positions: {} total, {} AC_COEFF_SIGN ({:.1}%)",
        positions.len(),
        ac_positions.len(),
        100.0 * ac_positions.len() as f64 / positions.len() as f64
    );

    let mut rng = ChaCha8Rng::seed_from_u64(SAMPLE_SEED);
    let n_sample = N_POSITIONS_PER_FIXTURE.min(ac_positions.len());
    let mut sampled: Vec<_> = ac_positions.clone();
    sampled.shuffle(&mut rng);
    sampled.truncate(n_sample);

    let w_y = spec.width as usize;
    let h_y = spec.height as usize;
    let natural_y = &natural_yuv[..w_y * h_y];

    let mut results: Vec<SpikeRow> = Vec::with_capacity(n_sample);
    let mut skipped_length_delta = 0;

    for (i, p) in sampled.iter().enumerate() {
        let meta = tile.bit_meta[p.cursor as usize];
        let stego_packet = match flip_one_and_make_packet(
            &natural_packet,
            &recording,
            p.cursor,
            p.natural_value,
        ) {
            Some(pkt) => pkt,
            None => {
                skipped_length_delta += 1;
                continue;
            }
        };
        let label = format!("stego_{}_{}", spec.name, i);
        let stego_yuv = decode_av1_to_yuv(&stego_packet, spec.width, spec.height, &label);
        let stats = measure_cascade(&natural_yuv, &stego_yuv, spec.width, spec.height);

        // Local smoothness at the position's block center.
        let cx = meta.plane_px_x as usize + (1usize << meta.tx_width_log2) / 2;
        let cy = meta.plane_px_y as usize + (1usize << meta.tx_height_log2) / 2;
        let smoothness = if meta.plane == 0 {
            local_smoothness(natural_y, w_y, h_y, cx.min(w_y - 1), cy.min(h_y - 1), 32)
        } else {
            // chroma — skip smoothness, use 0 as placeholder
            0.0
        };

        eprintln!(
            "[B.1.5.0] pos {:3} cursor={:6} plane={} px=({:4},{:4}) tx={}x{} tx_type={:2} scan_pos={:3} |coeff|={:5} | \
             smoothness={:6.2} max_abs={:3} mean_abs={:5.2} mse={:7.2} diverged={}",
            i,
            p.cursor,
            meta.plane,
            meta.plane_px_x,
            meta.plane_px_y,
            1usize << meta.tx_width_log2,
            1usize << meta.tx_height_log2,
            meta.tx_type,
            meta.scan_pos,
            meta.coeff_magnitude,
            smoothness,
            stats.max_abs,
            stats.mean_abs,
            stats.mse,
            stats.pixels_diverged,
        );
        results.push((i, smoothness, meta.coeff_magnitude, stats));
    }

    if skipped_length_delta > 0 {
        eprintln!(
            "[B.1.5.0] skipped {} positions (tile-group length delta — rare range-coder trailing-carry case)",
            skipped_length_delta
        );
    }
    results
}

fn print_aggregate(all_results: &[(String, Vec<SpikeRow>)]) {
    eprintln!("\n[B.1.5.0] ============ AGGREGATE HISTOGRAMS ============");
    for (fixture_name, results) in all_results {
        if results.is_empty() {
            eprintln!("[B.1.5.0] {} — no results", fixture_name);
            continue;
        }
        let mut max_abs_v: Vec<u8> = results.iter().map(|(_, _, _, s)| s.max_abs).collect();
        let mut mean_abs_v: Vec<f64> = results.iter().map(|(_, _, _, s)| s.mean_abs).collect();
        let mut diverged_v: Vec<usize> =
            results.iter().map(|(_, _, _, s)| s.pixels_diverged).collect();
        max_abs_v.sort_unstable();
        mean_abs_v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        diverged_v.sort_unstable();

        let p = |sorted_v: &[u8], q: f64| sorted_v[((sorted_v.len() - 1) as f64 * q) as usize];
        let pf = |sorted_v: &[f64], q: f64| sorted_v[((sorted_v.len() - 1) as f64 * q) as usize];
        let pu = |sorted_v: &[usize], q: f64| sorted_v[((sorted_v.len() - 1) as f64 * q) as usize];

        eprintln!(
            "[B.1.5.0] {} (n={}):  max_abs  min={:3} p25={:3} p50={:3} p75={:3} p90={:3} p95={:3} max={:3}",
            fixture_name,
            max_abs_v.len(),
            max_abs_v[0],
            p(&max_abs_v, 0.25),
            p(&max_abs_v, 0.50),
            p(&max_abs_v, 0.75),
            p(&max_abs_v, 0.90),
            p(&max_abs_v, 0.95),
            *max_abs_v.last().unwrap(),
        );
        eprintln!(
            "[B.1.5.0] {} (n={}):  mean_abs min={:6.3} p25={:6.3} p50={:6.3} p75={:6.3} p90={:6.3} p95={:6.3} max={:6.3}",
            fixture_name,
            mean_abs_v.len(),
            mean_abs_v[0],
            pf(&mean_abs_v, 0.25),
            pf(&mean_abs_v, 0.50),
            pf(&mean_abs_v, 0.75),
            pf(&mean_abs_v, 0.90),
            pf(&mean_abs_v, 0.95),
            *mean_abs_v.last().unwrap(),
        );
        eprintln!(
            "[B.1.5.0] {} (n={}):  diverged min={:5} p25={:5} p50={:5} p75={:5} p90={:5} p95={:5} max={:5} (px > {})",
            fixture_name,
            diverged_v.len(),
            diverged_v[0],
            pu(&diverged_v, 0.25),
            pu(&diverged_v, 0.50),
            pu(&diverged_v, 0.75),
            pu(&diverged_v, 0.90),
            pu(&diverged_v, 0.95),
            *diverged_v.last().unwrap(),
            DIVERGENCE_THRESHOLD,
        );

        // Smoothness vs cascade correlation — split into smooth (lower-50%) vs textured (upper-50%) buckets.
        let mut by_smoothness: Vec<_> = results
            .iter()
            .filter(|(_, sm, _, _)| *sm > 0.0)
            .copied()
            .collect();
        by_smoothness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        if by_smoothness.len() >= 4 {
            let mid = by_smoothness.len() / 2;
            let smooth_half = &by_smoothness[..mid];
            let textured_half = &by_smoothness[mid..];
            let mean_smooth = smooth_half.iter().map(|(_, _, _, s)| s.max_abs as f64).sum::<f64>()
                / smooth_half.len() as f64;
            let mean_textured = textured_half.iter().map(|(_, _, _, s)| s.max_abs as f64).sum::<f64>()
                / textured_half.len() as f64;
            eprintln!(
                "[B.1.5.0] {} SMOOTHNESS CORRELATION: smooth-half mean_max_abs={:5.2}  textured-half mean_max_abs={:5.2}  (ratio={:.2}×)",
                fixture_name, mean_smooth, mean_textured, mean_textured / mean_smooth.max(0.001),
            );
        }

        // B.1.5.0.5: coefficient-magnitude vs cascade correlation —
        // the headline correlation for EE-D's threshold calibration.
        // Split by |coeff| quartile (low / mid-low / mid-high / high)
        // and report mean max_abs per bucket.
        let mut by_coeff: Vec<_> = results
            .iter()
            .filter(|(_, _, c, _)| *c > 0)
            .copied()
            .collect();
        by_coeff.sort_by_key(|(_, _, c, _)| *c);
        if by_coeff.len() >= 8 {
            let q = by_coeff.len() / 4;
            let buckets: [&[SpikeRow]; 4] = [
                &by_coeff[..q],
                &by_coeff[q..2 * q],
                &by_coeff[2 * q..3 * q],
                &by_coeff[3 * q..],
            ];
            let bucket_labels = ["low", "mid-low", "mid-high", "high"];
            eprintln!("[B.1.5.0] {} COEFF_MAGNITUDE vs cascade (|coeff| quartiles):", fixture_name);
            for (label, b) in bucket_labels.iter().zip(buckets.iter()) {
                if b.is_empty() {
                    continue;
                }
                let coeff_min = b.iter().map(|(_, _, c, _)| *c).min().unwrap_or(0);
                let coeff_max = b.iter().map(|(_, _, c, _)| *c).max().unwrap_or(0);
                let mean_max_abs =
                    b.iter().map(|(_, _, _, s)| s.max_abs as f64).sum::<f64>() / b.len() as f64;
                let mean_diverged =
                    b.iter().map(|(_, _, _, s)| s.pixels_diverged as f64).sum::<f64>()
                        / b.len() as f64;
                eprintln!(
                    "[B.1.5.0]   {:8} (|coeff|={:3}..={:3}, n={:2}): mean_max_abs={:5.2}  mean_diverged={:7.1}",
                    label,
                    coeff_min,
                    coeff_max,
                    b.len(),
                    mean_max_abs,
                    mean_diverged,
                );
            }
            // Headline ratio: highest-quartile mean_max_abs / lowest-quartile mean_max_abs.
            let low_mean = buckets[0]
                .iter()
                .map(|(_, _, _, s)| s.max_abs as f64)
                .sum::<f64>()
                / buckets[0].len().max(1) as f64;
            let high_mean = buckets[3]
                .iter()
                .map(|(_, _, _, s)| s.max_abs as f64)
                .sum::<f64>()
                / buckets[3].len().max(1) as f64;
            eprintln!(
                "[B.1.5.0] {} HEADLINE: high-coeff-quartile mean_max_abs={:5.2}  low-coeff-quartile mean_max_abs={:5.2}  (ratio={:.2}×)",
                fixture_name, high_mean, low_mean, high_mean / low_mean.max(0.001),
            );
        }

        // B.1.5.0.5 sanity assertion: at least one luma AC position
        // must have a non-zero coeff_magnitude (encoder is populating
        // the field). If this trips, the fork-patch regressed.
        let any_nonzero = results.iter().any(|(_, _, c, _)| *c > 0);
        assert!(
            any_nonzero,
            "[{}] coeff_magnitude is zero on every sampled position — the B.1.5.0.5 fork-patch isn't populating the field. Check phasm-rav1e/src/context/block_unit.rs:2042-2056.",
            fixture_name,
        );
    }
    eprintln!("[B.1.5.0] ============================================");
}

#[test]
#[ignore = "spike — runs N×encode+decode per fixture; invoke with --ignored"]
fn b150_cascade_measurement_spike() {
    let mut all_results: Vec<(String, Vec<SpikeRow>)> = Vec::new();
    for spec in FIXTURES {
        let results = run_spike_for_fixture(spec);
        all_results.push((spec.name.to_string(), results));
    }
    print_aggregate(&all_results);
}
