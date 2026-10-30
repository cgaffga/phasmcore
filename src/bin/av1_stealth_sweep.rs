// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! G.2 Phase 2 Tier 2 — AUC-vs-utilization sweep using the W6 self-
//! steganalyzer.
//!
//! Mirrors the W6 measurement logic from
//! `core/tests/av1_self_steganalyzer.rs` (positive_ratio + adjacent-
//! agreement features on AC_COEFF_SIGN bits, mean-difference axis
//! projection, Mann-Whitney U AUC) but **parameterized on
//! per-cover utilization rate `u`**.
//!
//! For each `u ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 1.00}`, the encoder
//! consumes `u × n_cover` bits per frame (sizes the plaintext so STC
//! reaches that fraction). The detector then measures cover-vs-stego
//! separation. Plotting AUC vs `u` identifies the highest utilization
//! that keeps detection below the design's 0.55 / 0.6 threshold —
//! which becomes the locked `target_utilization` constant for the
//! balanced-allocation planner (see
//! `docs/design/video/balanced-allocation-v3.md` § 5).
//!
//! ## Honesty caveat
//!
//! The W6 detector is a **classical** statistical detector (two
//! 1D features → nearest-mean classifier). It is NOT a CNN-based
//! steganalyst trained on a large stego/cover corpus. AUC numbers
//! against W6 give a meaningful, reproducible regression baseline
//! and are appropriate for the project's stated stealth gate, but
//! they may understate detectability against a sophisticated
//! adversary. See the `av1_self_steganalyzer.rs` docstring for the
//! W6 model-fitness discussion.
//!
//! ## Usage
//!
//! ```bash
//! av1_stealth_sweep <utilization_pct> <out_json> [width] [height]
//! ```
//!
//! - `utilization_pct` — integer percent: 5, 10, 25, 50, 75, 100.
//! - `out_json` — output path for the JSON measurement.
//! - `width`/`height` — optional frame dimensions (default 256×144 to
//!   mirror W6 baseline). For Tier 2 calibration, use 1920×1080 so
//!   the 24-byte frame-overhead floor doesn't push effective
//!   utilization upward on small fixtures.
//!
//! ## Effective vs requested utilization
//!
//! The JSON output reports `effective_utilization` per sample — the
//! true fraction of `n_cover` bits consumed after the 24-byte frame-
//! overhead floor and the 1..256-byte plaintext clamp. Small fixtures
//! or low requested u may have `effective_utilization > requested`.
//! Plot AUC vs `effective_utilization` (not `requested_utilization`)
//! for calibration.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use phasm_core::codec::av1::stego::decoder::decode_with_recording;
use phasm_core::codec::av1::stego::orchestrator::av1_stego_embed;
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
    PhasmFrameRecording, PHASM_TAG_AC_COEFF_SIGN,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

struct SeekFixture {
    name: &'static str,
    source: &'static str,
    duration_s: f32,
}

// Mirrors the W6 fixture set (3 clips × 10 seek points = 30 frame pairs
// per utilization rate). Smaller corpus than Tier 1 because the
// steganalyzer is per-frame-keyframe.
const FIXTURES: &[SeekFixture] = &[
    SeekFixture {
        name: "iphone_img4138",
        source: "IMG_4138.MOV",
        duration_s: 20.0,
    },
    SeekFixture {
        name: "carplane",
        source: "Artlist_CarPlane.mp4",
        duration_s: 8.0,
    },
    SeekFixture {
        name: "iphone5_1080p",
        source: "iphone5_1080p_30fps_h264_high.mov",
        duration_s: 12.0,
    },
];

const SEEK_POINTS_PER_FIXTURE: usize = 10;
const DEFAULT_WIDTH: u32 = 256;
const DEFAULT_HEIGHT: u32 = 144;
const QUANTIZER: usize = 30;

type Features = [f64; 2];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: {} <utilization_pct: 1..=100> <out_json> [width] [height]",
            args[0]
        );
        std::process::exit(2);
    }
    let utilization_pct: u32 = args[1].parse().expect("utilization_pct must be integer");
    let out_json = &args[2];
    let width: u32 = if args.len() >= 4 {
        args[3].parse().expect("width must be integer")
    } else {
        DEFAULT_WIDTH
    };
    let height: u32 = if args.len() >= 5 {
        args[4].parse().expect("height must be integer")
    } else {
        DEFAULT_HEIGHT
    };
    if !(1..=100).contains(&utilization_pct) {
        eprintln!("utilization_pct must be in [1, 100], got {utilization_pct}");
        std::process::exit(2);
    }

    eprintln!(
        "[stealth_sweep] utilization = {}% ({:.2})  dims = {}x{}  fixtures = {} × seek_points = {}",
        utilization_pct,
        utilization_pct as f64 / 100.0,
        width, height,
        FIXTURES.len(),
        SEEK_POINTS_PER_FIXTURE,
    );

    let mut cover_features: Vec<Features> = Vec::new();
    let mut stego_features: Vec<Features> = Vec::new();
    let mut per_sample: Vec<SampleRecord> = Vec::new();
    let mut skipped_smooth = 0u32;
    let mut skipped_embed_failed = 0u32;
    let mut payload_seed: u64 = 0xC0FFEE;
    let t_start = std::time::Instant::now();

    for fx in FIXTURES {
        let usable_range = fx.duration_s - 1.0;
        let step = usable_range / SEEK_POINTS_PER_FIXTURE as f32;
        for i in 0..SEEK_POINTS_PER_FIXTURE {
            let seek_s = 0.5 + step * i as f32;
            let yuv = extract_yuv_frame(fx.source, seek_s, width, height);
            let (natural, recording) = encode_natural(&yuv, width, height);

            let tile = &recording.tiles[0];
            let n_cover = tile
                .bit_tags
                .iter()
                .filter(|&&t| t == PHASM_TAG_AC_COEFF_SIGN)
                .count();

            // Same MIN_COVER_BITS as W6 — drop smooth frames.
            const MIN_COVER_BITS: usize = 256;
            if n_cover < MIN_COVER_BITS {
                skipped_smooth += 1;
                continue;
            }

            // Size the plaintext so STC reaches utilization_pct% of n_cover.
            // m_bits ≈ n_cover × u; payload_bytes = m_bits / 8;
            // plaintext = payload_bytes − 24 (frame overhead).
            // Note: max(25)/max(1) floors mean effective utilization
            // may exceed requested at small n_cover.
            let target_payload_bytes =
                ((n_cover * utilization_pct as usize / 100) / 8).max(25);
            // Plaintext clamp raised from W6's 256 to 16384 so the
            // Tier 2 sweep can actually reach high utilization on
            // large-cover frames (1080p has n_cover ~ 30k bits → 256
            // bytes capped effective u at 1.2%, useless for the
            // 0.05-0.50 calibration range).
            let plaintext_len = (target_payload_bytes.saturating_sub(24)).max(1).min(16384);
            // Effective utilization after floors + plaintext clamp.
            let effective_payload_bytes = (plaintext_len + 24).min(target_payload_bytes);
            let effective_utilization =
                (effective_payload_bytes * 8) as f64 / n_cover as f64;

            payload_seed = payload_seed.wrapping_mul(0x9E3779B97F4A7C15);
            let mut msg = vec![0u8; plaintext_len];
            for (k, b) in msg.iter_mut().enumerate() {
                let s = payload_seed
                    .wrapping_add(k as u64)
                    .wrapping_mul(0xCAFEBABE);
                *b = (s >> 24) as u8;
            }
            let passphrase = format!("stealth-sweep-{utilization_pct}-{:x}", payload_seed);

            match av1_stego_embed(natural.clone(), recording, &msg, &passphrase) {
                Ok(stego) => {
                    let f_cover = features_from_av1(&natural);
                    let f_stego = features_from_av1(&stego);
                    cover_features.push(f_cover);
                    stego_features.push(f_stego);
                    per_sample.push(SampleRecord {
                        fixture: fx.name,
                        seek_s,
                        n_cover,
                        plaintext_len,
                        effective_utilization,
                        f_cover,
                        f_stego,
                    });
                }
                Err(e) => {
                    skipped_embed_failed += 1;
                    eprintln!(
                        "[stealth_sweep]   skip {} @ {:.2}s — embed failed: {:?}",
                        fx.name, seek_s, e
                    );
                }
            }
        }
    }

    let wall_ms = t_start.elapsed().as_millis();
    let n = cover_features.len();
    if n < 6 {
        eprintln!(
            "[stealth_sweep] FATAL: only {} usable samples (need ≥ 6) — corpus too uniform / smooth",
            n
        );
        std::process::exit(3);
    }

    let mean_c = mean(&cover_features);
    let mean_s = mean(&stego_features);
    let axis = [mean_s[0] - mean_c[0], mean_s[1] - mean_c[1]];
    let project = |f: &Features| f[0] * axis[0] + f[1] * axis[1];
    let cover_proj: Vec<f64> = cover_features.iter().map(project).collect();
    let stego_proj: Vec<f64> = stego_features.iter().map(project).collect();
    let auc_value = auc(&cover_proj, &stego_proj);

    eprintln!(
        "[stealth_sweep] u={}%  n={}  cover_mean=({:.4}, {:.4})  stego_mean=({:.4}, {:.4})  Δ=({:+.4}, {:+.4})  AUC={:.4}  wall={} ms",
        utilization_pct,
        n,
        mean_c[0], mean_c[1],
        mean_s[0], mean_s[1],
        mean_s[0] - mean_c[0], mean_s[1] - mean_c[1],
        auc_value,
        wall_ms,
    );

    write_json(
        out_json,
        utilization_pct,
        n,
        skipped_smooth,
        skipped_embed_failed,
        wall_ms,
        &mean_c,
        &mean_s,
        auc_value,
        &per_sample,
    )
    .expect("write_json");
    eprintln!("[stealth_sweep] wrote {out_json}");
}

// ─── Steganalyzer plumbing (mirrors av1_self_steganalyzer.rs) ──────────

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv_frame(source: &str, seek_s: f32, width: u32, height: u32) -> Vec<u8> {
    let src = corpus_root().join(source);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!(
        "scale={}:{}:force_original_aspect_ratio=disable",
        width, height
    );
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args([
            "-frames:v",
            "1",
            "-vf",
            &vf,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .expect("ffmpeg failed to launch");
    assert!(
        out.status.success(),
        "ffmpeg failed on {}: {}",
        source,
        String::from_utf8_lossy(&out.stderr)
    );
    let expected = (width * height * 3 / 2) as usize;
    assert_eq!(
        out.stdout.len(),
        expected,
        "ffmpeg yuv size mismatch for {} ({}x{})",
        source,
        width,
        height
    );
    out.stdout
}

fn encode_natural(yuv: &[u8], width: u32, height: u32) -> (Vec<u8>, PhasmFrameRecording) {
    let config = Arc::new(EncoderConfig {
        width: width as usize,
        height: height as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: QUANTIZER,
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

    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);

    let mut frame = make_frame::<u8>(w, h, ChromaSampling::Cs420);
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

fn features_from_av1(av1_bytes: &[u8]) -> Features {
    let decoded = decode_with_recording(av1_bytes).expect("decode_with_recording");
    use core_dav1d_sys::DAV1D_PHASM_TAG_AC_COEFF_SIGN;
    let bits: Vec<u8> = decoded
        .iter()
        .filter(|p| p.tag == DAV1D_PHASM_TAG_AC_COEFF_SIGN)
        .map(|p| p.decoded_value)
        .collect();
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
    m[0] /= n;
    m[1] /= n;
    m
}

fn auc(cover: &[f64], stego: &[f64]) -> f64 {
    let mut higher = 0u32;
    let mut tied = 0u32;
    for &s in stego {
        for &c in cover {
            if s > c {
                higher += 1;
            } else if s == c {
                tied += 1;
            }
        }
    }
    let total = (cover.len() * stego.len()) as f64;
    let raw = (higher as f64 + 0.5 * tied as f64) / total;
    raw.max(1.0 - raw)
}

// ─── JSON output ──────────────────────────────────────────────────────

struct SampleRecord {
    fixture: &'static str,
    seek_s: f32,
    n_cover: usize,
    plaintext_len: usize,
    effective_utilization: f64,
    f_cover: Features,
    f_stego: Features,
}

#[allow(clippy::too_many_arguments)]
fn write_json(
    path: &str,
    utilization_pct: u32,
    n_samples: usize,
    skipped_smooth: u32,
    skipped_embed_failed: u32,
    wall_ms: u128,
    mean_c: &Features,
    mean_s: &Features,
    auc_value: f64,
    samples: &[SampleRecord],
) -> std::io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"utilization_pct\": {},\n", utilization_pct));
    json.push_str(&format!(
        "  \"utilization\": {:.4},\n",
        utilization_pct as f64 / 100.0
    ));
    json.push_str(&format!("  \"n_samples\": {},\n", n_samples));
    json.push_str(&format!("  \"skipped_smooth\": {},\n", skipped_smooth));
    json.push_str(&format!(
        "  \"skipped_embed_failed\": {},\n",
        skipped_embed_failed
    ));
    json.push_str(&format!("  \"wall_ms\": {},\n", wall_ms));
    json.push_str(&format!(
        "  \"cover_mean\": [{:.6}, {:.6}],\n",
        mean_c[0], mean_c[1]
    ));
    json.push_str(&format!(
        "  \"stego_mean\": [{:.6}, {:.6}],\n",
        mean_s[0], mean_s[1]
    ));
    json.push_str(&format!(
        "  \"mean_delta\": [{:.6}, {:.6}],\n",
        mean_s[0] - mean_c[0],
        mean_s[1] - mean_c[1]
    ));
    json.push_str(&format!("  \"auc\": {:.6},\n", auc_value));
    json.push_str("  \"samples\": [\n");
    for (i, s) in samples.iter().enumerate() {
        let comma = if i + 1 < samples.len() { "," } else { "" };
        json.push_str(&format!(
            "    {{\"fixture\": \"{}\", \"seek_s\": {:.2}, \"n_cover\": {}, \"plaintext_len\": {}, \"effective_utilization\": {:.4}, \"f_cover\": [{:.6}, {:.6}], \"f_stego\": [{:.6}, {:.6}]}}{}\n",
            s.fixture, s.seek_s, s.n_cover, s.plaintext_len, s.effective_utilization,
            s.f_cover[0], s.f_cover[1], s.f_stego[0], s.f_stego[1], comma
        ));
    }
    json.push_str("  ]\n");
    json.push_str("}\n");
    std::fs::write(path, json)
}
