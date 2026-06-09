// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! W6 — self-steganalyzer v0.3 minimal subset.
//!
//! Distribution-based detector for the AcCoeffSign channel. Builds a
//! small cover/stego corpus from the three W4 fixtures (10 seek
//! points each → 30 cover frames + 30 stego frames with random
//! payloads), extracts two features per sample, trains a nearest-
//! class-mean classifier, reports AUC.
//!
//! Per `docs/design/video/av1/channel-design.md` § 5: the v1.0 ship
//! gate is AUC ≤ 0.6 per channel. v0.3 ships with UNIFORM STC costs,
//! so the AcCoeffSign histogram is expected to leak measurably — the
//! point of W6 in v0.3 isn't to PASS the v1.0 gate (J-UNIWARD costs
//! + cascade-safety land in v0.5+), it's to STAND UP THE
//! MEASUREMENT INFRASTRUCTURE so we ship v0.3 with eyes open about
//! the leakage and have a regression gate for catastrophic drift.
//!
//! The v0.3 assertion is therefore loose (AUC < 0.99) — catches a
//! regression where stego becomes *perfectly* detectable. Tighter
//! gates land alongside the v0.5 cost-model upgrade.
//!
//! # Measured baseline (2026-05-21)
//!
//! On 27 cover / 27 stego pairs across the 3 W4 fixtures × 10 seek
//! points (3 skipped — smooth frames + one byte-splice edge case):
//!
//!     positive_ratio:  cover=0.4977 stego=0.4887 (Δ -0.0090)
//!     agreement_rate:  cover=0.5042 stego=0.5003 (Δ -0.0039)
//!     AUC = 0.6996  — "practically detectable" band per § 5.3
//!
//! v0.3 explicitly does not claim stealth. The number lives here so
//! v0.5 J-UNIWARD experiments can measure improvement quantitatively
//! instead of guessing.
//!
//! # Features extracted
//!
//! For each frame, decode via dav1d hooks, filter to AC_COEFF_SIGN
//! positions, get a bit sequence b_0 .. b_{n-1}, compute:
//!
//!   F1 = positive_ratio       = #(b_i == 1) / n
//!   F2 = adjacent_agreement   = #(b_i == b_{i-1}) / (n-1)
//!
//! Cover frames have natural sign correlation from image structure
//! (smooth gradients → correlated AC signs across adjacent blocks).
//! Stego frames decorrelate signs at flipped positions, lowering F2
//! toward 0.5 and pushing F1 toward 0.5.
//!
//! # Classifier
//!
//! Two-class nearest-mean in the (F1, F2) plane. Discrimination axis
//! = (mean_stego - mean_cover). Each sample is projected onto this
//! axis; AUC is the Mann-Whitney U statistic divided by sample-
//! product. Symmetrized so AUC ∈ [0.5, 1.0] regardless of axis
//! direction.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
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
    /// Approximate clip duration in seconds; seek points are spaced
    /// uniformly inside (0.5, duration - 0.5).
    duration_s: f32,
}

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
const WIDTH: u32 = 256;
const HEIGHT: u32 = 144;
const QUANTIZER: usize = 30;
/// v0.3 ceiling: catch only catastrophic detection (AUC → 1.0).
/// v0.5+ tightens this to AUC ≤ 0.6 alongside J-UNIWARD costs.
const AUC_CEILING_V0_3: f64 = 0.99;

type Features = [f64; 2];

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv_frame(source: &str, seek_s: f32) -> Vec<u8> {
    let src = corpus_root().join(source);
    assert!(src.exists(), "missing corpus source: {}", src.display());
    let vf = format!("scale={}:{}:force_original_aspect_ratio=disable", WIDTH, HEIGHT);
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args(["-frames:v", "1", "-vf", &vf, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output()
        .expect("ffmpeg launch");
    assert!(
        out.status.success(),
        "ffmpeg yuv extract failed on {} @ {}s: {}",
        source,
        seek_s,
        String::from_utf8_lossy(&out.stderr)
    );
    let expected = (WIDTH * HEIGHT * 3 / 2) as usize;
    assert_eq!(out.stdout.len(), expected);
    out.stdout
}

fn encode_natural(yuv: &[u8]) -> (Vec<u8>, PhasmFrameRecording) {
    let config = Arc::new(EncoderConfig {
        width: WIDTH as usize,
        height: HEIGHT as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: QUANTIZER,
        ..Default::default()
    });
    let mut sequence = Sequence::new(&config);
    sequence.enable_large_lru = false;
    let mut fi =
        FrameInvariants::<u8>::new_key_frame(config.clone(), Arc::new(sequence), 0, Box::new([]));
    fi.enable_segmentation = false;
    let mut frame = make_frame::<u8>(WIDTH as usize, HEIGHT as usize, ChromaSampling::Cs420);
    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    // CRITICAL: use `Plane::copy_from_raw_u8` — see feedback memory.
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
    assert!(n > 1, "fixture too small for steganalysis ({} AC bits)", n);
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

/// AUC via Mann-Whitney U: P(stego_proj > cover_proj). Ties count
/// as 0.5. Symmetrized to [0.5, 1.0] so axis direction doesn't
/// matter — the magnitude of separation is what counts.
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

#[test]
fn av1_self_steganalyzer_distribution_baseline() {
    eprintln!(
        "[steganalyzer] corpus: {} fixtures × {} seek points = {} pairs",
        FIXTURES.len(),
        SEEK_POINTS_PER_FIXTURE,
        FIXTURES.len() * SEEK_POINTS_PER_FIXTURE
    );

    let mut cover_features: Vec<Features> = Vec::new();
    let mut stego_features: Vec<Features> = Vec::new();
    let mut skipped = 0u32;

    let mut payload_seed: u64 = 0xC0FFEE;
    for fx in FIXTURES {
        let usable_range = fx.duration_s - 1.0;
        let step = usable_range / SEEK_POINTS_PER_FIXTURE as f32;
        for i in 0..SEEK_POINTS_PER_FIXTURE {
            let seek_s = 0.5 + step * i as f32;
            let yuv = extract_yuv_frame(fx.source, seek_s);
            let (natural, recording) = encode_natural(&yuv);

            // Inspect the cover's AC sign capacity. If the frame is
            // too smooth (e.g., near-uniform sky / dark scene) it has
            // few AC coefficients and can't fit a meaningful stego
            // payload. Skip rather than fail — the v0.3 detector
            // floor needs cross-content samples, not all 30 forced.
            let tile = &recording.tiles[0];
            let n_cover = tile
                .bit_tags
                .iter()
                .filter(|&&t| t == PHASM_TAG_AC_COEFF_SIGN)
                .count();
            // Framing overhead is ~24 bytes = 192 bits payload, plus
            // ≥1 plaintext byte → ≥200-bit minimum. Skip anything
            // tighter than 256 bits to give STC some slack.
            const MIN_COVER_BITS: usize = 256;
            if n_cover < MIN_COVER_BITS {
                skipped += 1;
                eprintln!(
                    "[steganalyzer]   skip {} @ {:.2}s — only {} AC cover bits",
                    fx.name, seek_s, n_cover
                );
                continue;
            }

            // Size the plaintext so total payload fills ~50% of cover
            // (max-detection-signal regime). m_bits ≈ n_cover/2.
            // payload_bytes = m_bits/8; plaintext = payload_bytes - 24
            // (frame overhead: salt 8 + nonce 12 + len 1 + CRC 4 ≈ 24).
            let target_payload_bytes = (n_cover / 2 / 8).max(25);
            let plaintext_len = (target_payload_bytes - 24).max(1).min(256);

            // Deterministic per-sample seeded plaintext bytes — gives
            // each pair a distinct flip pattern through STC.
            payload_seed = payload_seed.wrapping_mul(0x9E3779B97F4A7C15);
            let mut msg = vec![0u8; plaintext_len];
            for (k, b) in msg.iter_mut().enumerate() {
                let s = payload_seed
                    .wrapping_add(k as u64)
                    .wrapping_mul(0xCAFEBABE);
                *b = (s >> 24) as u8;
            }
            let passphrase = format!("self-stega-pass-{:x}", payload_seed);

            // Embed may fail on rare frames where the range-coder
            // length isn't strictly preserved by 50/50 flips (known
            // v0.3 edge case in the byte-splice invariant — affects
            // a small fraction of payload/cover combinations). Skip
            // those samples for the stealth measurement; they don't
            // represent successful stego usage.
            match av1_stego_embed(natural.clone(), recording, &msg, &passphrase) {
                Ok(stego) => {
                    cover_features.push(features_from_av1(&natural));
                    stego_features.push(features_from_av1(&stego));
                }
                Err(e) => {
                    skipped += 1;
                    eprintln!(
                        "[steganalyzer]   skip {} @ {:.2}s — embed failed: {:?}",
                        fx.name, seek_s, e
                    );
                }
            }
        }
    }

    let n = cover_features.len();
    if skipped > 0 {
        eprintln!(
            "[steganalyzer] skipped {} smooth frames (insufficient cover capacity)",
            skipped
        );
    }
    assert!(
        n >= 10,
        "too few usable samples ({}) — corpus is too uniform / seek points landed on smooth frames",
        n
    );
    let mean_c = mean(&cover_features);
    let mean_s = mean(&stego_features);
    eprintln!(
        "[steganalyzer] {} cover / {} stego samples",
        n,
        stego_features.len()
    );
    eprintln!(
        "[steganalyzer] cover  mean F=(pos={:.4}, agree={:.4})",
        mean_c[0], mean_c[1]
    );
    eprintln!(
        "[steganalyzer] stego  mean F=(pos={:.4}, agree={:.4})",
        mean_s[0], mean_s[1]
    );
    eprintln!(
        "[steganalyzer] Δmean   = (pos={:+.4}, agree={:+.4})",
        mean_s[0] - mean_c[0],
        mean_s[1] - mean_c[1]
    );

    // Discrimination axis = stego_mean - cover_mean.
    let axis = [mean_s[0] - mean_c[0], mean_s[1] - mean_c[1]];
    let project = |f: &Features| f[0] * axis[0] + f[1] * axis[1];
    let cover_proj: Vec<f64> = cover_features.iter().map(project).collect();
    let stego_proj: Vec<f64> = stego_features.iter().map(project).collect();

    let auc_value = auc(&cover_proj, &stego_proj);
    eprintln!(
        "[steganalyzer] AUC = {:.4}  (0.5 = random; 0.6 = mildly informative; 0.7+ = detectable; \
         v1.0 gate = ≤ 0.6; v0.3 ceiling = {})",
        auc_value, AUC_CEILING_V0_3
    );

    // v0.3 sanity gate: catch only essentially-perfect detection.
    // v0.5+ tightens this alongside J-UNIWARD costs.
    assert!(
        auc_value < AUC_CEILING_V0_3,
        "self-steganalyzer AUC {:.4} ≥ v0.3 ceiling {:.4} — stego is essentially perfectly detectable. \
         Regression? Re-check phasm-rav1e tagging + STC plumbing.",
        auc_value,
        AUC_CEILING_V0_3
    );
}
