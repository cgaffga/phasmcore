// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! G.2 Phase 2 Tier 2 — H.264 AUC-vs-utilization sweep (#844).
//!
//! Codec-parity counterpart to `av1_stealth_sweep.rs`. A W6-style self-
//! steganalyzer on the H.264 CoeffSign (`coeff_sign_bypass`) domain,
//! parameterized on per-cover utilization rate `u`.
//!
//! For each `u ∈ {5, 10, 25, 50, 75, 100}%`, the encoder embeds a message
//! sized to `u × reported_capacity` per clip. The detector measures
//! cover-vs-stego separation on two CoeffSign features (positive_ratio +
//! adjacent-agreement), projects onto the mean-difference axis, and
//! computes the Mann-Whitney U AUC. Plotting AUC vs effective utilization
//! identifies the highest `u` that keeps per-clip AUC below the design's
//! 0.55 threshold — the locked `target_utilization` for the
//! balanced-allocation planner.
//!
//! ## Cover vs stego, comparably
//!
//! - **cover** = `h264_walk_cover` (the clean 4-domain cover walk
//!   — natural CoeffSign distribution, zero flips).
//! - **stego** = `StreamingEncodeSession` with the u%-sized message, walked
//!   back. Both use the SAME OH264 4-domain encode, so the only difference
//!   is the STC sign flips. n_cover matches between them (flips never change
//!   the count), which the runner asserts as a comparability check.
//!
//! ## Honesty caveat
//!
//! W6 is a CLASSICAL statistical detector (two 1D features → nearest-mean
//! classifier), not a CNN steganalyst. AUC numbers give a reproducible
//! regression baseline appropriate for the project's stated stealth gate
//! but may understate detectability against a sophisticated adversary. See
//! `av1_stealth_sweep.rs` for the same caveat.
//!
//! ## Usage
//!
//! ```bash
//! h264_stealth_sweep <utilization_pct: 1..=100> <out_json> [n_frames]
//! ```

#![cfg(all(feature = "h264-encoder", feature = "h264-decoder"))]

use std::path::{Path, PathBuf};
use std::process::Command;

use phasm_core::codec::h264::cabac::bin_decoder::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};
use phasm_core::codec::h264::openh264_stego::{h264_walk_cover, EncodeOpts};
use phasm_core::codec::h264::stego::oh264_capacity::h264_video_capacity;
use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::h264::stego::CostWeights;

struct SeekFixture {
    name: &'static str,
    source: &'static str,
    duration_s: f32,
}

// Spans the corpus range: clean mid-texture, motion, grain outlier (high
// cover), and low-texture (low cover).
const FIXTURES: &[SeekFixture] = &[
    SeekFixture { name: "iphone7_1080p", source: "iphone7_1080p_30fps_h264_high.mov", duration_s: 16.0 },
    SeekFixture { name: "carplane",      source: "Artlist_CarPlane.mp4",              duration_s: 6.5 },
    SeekFixture { name: "iphone5_1080p", source: "iphone5_1080p_30fps_h264_high.mov", duration_s: 12.0 },
    SeekFixture { name: "womansubway",   source: "Artlist_WomanSubway.mp4",           duration_s: 12.0 },
];

const SEEK_POINTS_PER_FIXTURE: usize = 8;
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1088; // 16-aligned 1080p
const N_FRAMES_DEFAULT: u32 = 30; // one production GOP
const GOP: u32 = 30;
const QP: i32 = 26;

type Features = [f64; 2];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <utilization_pct: 1..=100> <out_json> [n_frames]", args[0]);
        std::process::exit(2);
    }
    let utilization_pct: u32 = args[1].parse().expect("utilization_pct must be integer");
    let out_json = &args[2];
    let n_frames: u32 = if args.len() >= 4 {
        args[3].parse().expect("n_frames must be integer")
    } else {
        N_FRAMES_DEFAULT
    };
    if !(1..=100).contains(&utilization_pct) {
        eprintln!("utilization_pct must be in [1, 100], got {utilization_pct}");
        std::process::exit(2);
    }

    eprintln!(
        "[h264_stealth_sweep] u={}% ({:.2})  {}x{}x{}f gop={} qp={}  fixtures={} × seeks={}",
        utilization_pct, utilization_pct as f64 / 100.0,
        WIDTH, HEIGHT, n_frames, GOP, QP, FIXTURES.len(), SEEK_POINTS_PER_FIXTURE,
    );

    let opts = EncodeOpts { qp: QP, intra_period: GOP as i32 };
    let mut cover_features: Vec<Features> = Vec::new();
    let mut stego_features: Vec<Features> = Vec::new();
    let mut per_sample: Vec<SampleRecord> = Vec::new();
    let mut skipped_smooth = 0u32;
    let mut skipped_embed_failed = 0u32;
    let mut payload_seed: u64 = 0xC0FFEE;
    let t_start = std::time::Instant::now();

    for fx in FIXTURES {
        let usable_range = (fx.duration_s - (n_frames as f32 / 30.0) - 0.5).max(0.5);
        let step = usable_range / SEEK_POINTS_PER_FIXTURE as f32;
        for i in 0..SEEK_POINTS_PER_FIXTURE {
            let seek_s = 0.3 + step * i as f32;
            let yuv = match extract_yuv_clip(fx.source, seek_s, WIDTH, HEIGHT, n_frames) {
                Some(y) => y,
                None => { skipped_embed_failed += 1; continue; }
            };

            // Clean cover (natural CoeffSign distribution).
            let clean = match h264_walk_cover(&yuv, WIDTH, HEIGHT, n_frames, opts) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("  skip {} @ {:.2}s — count_cover failed: {:?}", fx.name, seek_s, e);
                    skipped_embed_failed += 1;
                    continue;
                }
            };
            let cover_bits = clean.coeff_sign_bypass.bits.clone();
            let n_cover = cover_bits.len();
            const MIN_COVER_BITS: usize = 256;
            if n_cover < MIN_COVER_BITS {
                skipped_smooth += 1;
                continue;
            }

            // Reported 4-domain capacity (tier 0) → size message to u% of it.
            let cap_bytes = h264_video_capacity(&yuv, WIDTH, HEIGHT, n_frames as usize, opts, false)
                .map(|info| info.per_tier_primary_max_message_bytes[0])
                .unwrap_or(0);
            if cap_bytes < 16 {
                skipped_smooth += 1;
                continue;
            }
            let target_bytes = ((cap_bytes * utilization_pct as usize) / 100).clamp(1, cap_bytes);

            payload_seed = payload_seed.wrapping_mul(0x9E3779B97F4A7C15);
            let msg = make_ascii_msg(target_bytes, payload_seed);
            let pass = format!("h264-stealth-{utilization_pct}-{:x}", payload_seed);

            let stego_bytes = match encode_stego(&yuv, WIDTH, HEIGHT, n_frames, GOP, &msg, &pass) {
                Some(b) => b,
                None => {
                    skipped_embed_failed += 1;
                    continue;
                }
            };
            let stego = match walk_annex_b_for_cover_with_options(
                &stego_bytes,
                WalkOptions { record_mvd: true, ..Default::default() },
            ) {
                Ok(w) => w.cover,
                Err(e) => {
                    eprintln!("  skip {} @ {:.2}s — stego walk failed: {:?}", fx.name, seek_s, e);
                    skipped_embed_failed += 1;
                    continue;
                }
            };
            let stego_bits = &stego.coeff_sign_bypass.bits;

            let f_cover = features(&cover_bits);
            let f_stego = features(stego_bits);
            let effective_utilization = (target_bytes as f64) / (cap_bytes as f64);
            cover_features.push(f_cover);
            stego_features.push(f_stego);
            per_sample.push(SampleRecord {
                fixture: fx.name,
                seek_s,
                n_cover,
                n_cover_stego: stego_bits.len(),
                cap_bytes,
                payload_bytes: target_bytes,
                effective_utilization,
                f_cover,
                f_stego,
            });
        }
    }

    let wall_ms = t_start.elapsed().as_millis();
    let n = cover_features.len();
    if n < 6 {
        eprintln!("[h264_stealth_sweep] FATAL: only {n} usable samples (need ≥ 6)");
        std::process::exit(3);
    }

    let mean_c = mean(&cover_features);
    let mean_s = mean(&stego_features);
    let axis = [mean_s[0] - mean_c[0], mean_s[1] - mean_c[1]];
    let project = |f: &Features| f[0] * axis[0] + f[1] * axis[1];
    let cover_proj: Vec<f64> = cover_features.iter().map(project).collect();
    let stego_proj: Vec<f64> = stego_features.iter().map(project).collect();
    let auc_value = auc(&cover_proj, &stego_proj);
    let mean_eff_u =
        per_sample.iter().map(|s| s.effective_utilization).sum::<f64>() / n as f64;

    eprintln!(
        "[h264_stealth_sweep] u={}%  n={}  eff_u={:.3}  cover=({:.4},{:.4})  stego=({:.4},{:.4})  Δ=({:+.4},{:+.4})  AUC={:.4}  wall={}ms",
        utilization_pct, n, mean_eff_u,
        mean_c[0], mean_c[1], mean_s[0], mean_s[1],
        mean_s[0] - mean_c[0], mean_s[1] - mean_c[1], auc_value, wall_ms,
    );

    write_json(out_json, utilization_pct, n, mean_eff_u, skipped_smooth,
        skipped_embed_failed, wall_ms, &mean_c, &mean_s, auc_value, &per_sample)
        .expect("write_json");
    eprintln!("[h264_stealth_sweep] wrote {out_json}");
}

// ─── encode / walk plumbing ────────────────────────────────────────────

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv_clip(source: &str, seek_s: f32, w: u32, h: u32, n: u32) -> Option<Vec<u8>> {
    let src = corpus_root().join(source);
    if !src.exists() {
        eprintln!("  corpus fixture missing: {}", src.display());
        return None;
    }
    let vf = format!("scale={w}:{h}:force_original_aspect_ratio=disable");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"]).arg(&src)
        .args(["-frames:v", &n.to_string(), "-vf", &vf,
               "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output().ok()?;
    if !out.status.success() {
        return None;
    }
    let expected = (w * h * 3 / 2) as usize * n as usize;
    if out.stdout.len() != expected {
        return None; // short read near EOF — drop this seek
    }
    Some(out.stdout)
}

fn encode_stego(yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, msg: &str, pass: &str) -> Option<Vec<u8>> {
    let params = EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: QP, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut enc = StreamingEncodeSession::create(params, msg, pass).ok()?;
    let fb = (w * h * 3 / 2) as usize;
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let cw = (w / 2) as usize;
    let mut out = Vec::new();
    for f in 0..n as usize {
        let off = f * fb;
        let frame = YuvFrameRef {
            y: &yuv[off..off + ys], y_stride: w as usize,
            u: &yuv[off + ys..off + ys + cs], u_stride: cw,
            v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
        };
        enc.push_frame(frame, &mut out).ok()?;
    }
    enc.finish(&mut out).ok()?;
    Some(out)
}

fn make_ascii_msg(len: usize, seed: u64) -> String {
    // Printable-ASCII, 1 byte/char, so byte length == char count == len.
    let mut s = String::with_capacity(len);
    for k in 0..len {
        let x = seed.wrapping_add(k as u64).wrapping_mul(0xCAFEBABE);
        s.push((33 + (x >> 24) as u8 % 94) as char); // '!'..'~'
    }
    s
}

// ─── W6 features + AUC (mirror av1_stealth_sweep) ──────────────────────

fn features(bits: &[u8]) -> Features {
    let n = bits.len();
    if n < 2 {
        return [0.5, 0.5];
    }
    let positive = bits.iter().filter(|&&b| b == 1).count();
    let positive_ratio = positive as f64 / n as f64;
    let mut agree = 0usize;
    for i in 1..n {
        if bits[i] == bits[i - 1] {
            agree += 1;
        }
    }
    let agreement = agree as f64 / (n - 1) as f64;
    [positive_ratio, agreement]
}

fn mean(samples: &[Features]) -> Features {
    let n = samples.len() as f64;
    let mut m = [0.0; 2];
    for f in samples {
        m[0] += f[0];
        m[1] += f[1];
    }
    [m[0] / n, m[1] / n]
}

fn auc(cover: &[f64], stego: &[f64]) -> f64 {
    let mut higher = 0u32;
    let mut tied = 0u32;
    for &s in stego {
        for &c in cover {
            if s > c { higher += 1; } else if s == c { tied += 1; }
        }
    }
    let total = (cover.len() * stego.len()) as f64;
    let raw = (higher as f64 + 0.5 * tied as f64) / total;
    raw.max(1.0 - raw)
}

// ─── JSON output ───────────────────────────────────────────────────────

struct SampleRecord {
    fixture: &'static str,
    seek_s: f32,
    n_cover: usize,
    n_cover_stego: usize,
    cap_bytes: usize,
    payload_bytes: usize,
    effective_utilization: f64,
    f_cover: Features,
    f_stego: Features,
}

#[allow(clippy::too_many_arguments)]
fn write_json(
    path: &str, utilization_pct: u32, n_samples: usize, mean_eff_u: f64,
    skipped_smooth: u32, skipped_embed_failed: u32, wall_ms: u128,
    mean_c: &Features, mean_s: &Features, auc_value: f64, samples: &[SampleRecord],
) -> std::io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"codec\": \"h264\",\n");
    json.push_str(&format!("  \"utilization_pct\": {utilization_pct},\n"));
    json.push_str(&format!("  \"requested_utilization\": {:.4},\n", utilization_pct as f64 / 100.0));
    json.push_str(&format!("  \"mean_effective_utilization\": {mean_eff_u:.4},\n"));
    json.push_str(&format!("  \"n_samples\": {n_samples},\n"));
    json.push_str(&format!("  \"skipped_smooth\": {skipped_smooth},\n"));
    json.push_str(&format!("  \"skipped_embed_failed\": {skipped_embed_failed},\n"));
    json.push_str(&format!("  \"wall_ms\": {wall_ms},\n"));
    json.push_str(&format!("  \"cover_mean\": [{:.6}, {:.6}],\n", mean_c[0], mean_c[1]));
    json.push_str(&format!("  \"stego_mean\": [{:.6}, {:.6}],\n", mean_s[0], mean_s[1]));
    json.push_str(&format!("  \"mean_delta\": [{:.6}, {:.6}],\n", mean_s[0] - mean_c[0], mean_s[1] - mean_c[1]));
    json.push_str(&format!("  \"auc\": {auc_value:.6},\n"));
    json.push_str("  \"samples\": [\n");
    for (i, s) in samples.iter().enumerate() {
        let comma = if i + 1 < samples.len() { "," } else { "" };
        json.push_str(&format!(
            "    {{\"fixture\": \"{}\", \"seek_s\": {:.2}, \"n_cover\": {}, \"n_cover_stego\": {}, \"cap_bytes\": {}, \"payload_bytes\": {}, \"effective_utilization\": {:.4}, \"f_cover\": [{:.6}, {:.6}], \"f_stego\": [{:.6}, {:.6}]}}{}\n",
            s.fixture, s.seek_s, s.n_cover, s.n_cover_stego, s.cap_bytes, s.payload_bytes,
            s.effective_utilization, s.f_cover[0], s.f_cover[1], s.f_stego[0], s.f_stego[1], comma
        ));
    }
    json.push_str("  ]\n}\n");
    std::fs::write(path, json)
}
