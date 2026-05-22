// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase B.1.4 — AoSO self-steganalyzer adapter for AcCoeffSign.
//!
//! Extends the W6 distribution detector (positive_ratio +
//! adjacent_agreement_rate) with AoSO-style cost-distribution
//! features computed via the av1_uniward J-UNIWARD module.
//!
//! # The intuition
//!
//! W6's 2 features only see sign-value distribution. J-UNIWARD's
//! whole job is to make those invariant under flipping — so W6 AUC
//! drops to ~0.5 once cost-aware flipping is enabled (B.1.2 + B.1.3).
//!
//! AoSO asks a different question: of all the AC sign positions in
//! a frame, what is the distribution of "how much would the encoder
//! have spent to flip each one"? Both cover and stego have AC signs;
//! the costs computed from reconstructed pixels differ between them
//! because:
//!   - Cover: signs match the natural residual at each position
//!   - Stego: ~half the n_used positions had flips driven by STC, so
//!     the residual at THOSE positions is the inverse of natural
//!
//! The cost histogram (mean, std-dev, median, skew) on stego is
//! statistically distinguishable from cover IF the cost-aware flips
//! systematically picked low-cost positions (which they did).
//!
//! # Implementation
//!
//! Cover-side: J-UNIWARD costs computed over the ENCODER's
//! `recording.reconstructed_planes` (the natural reconstruction).
//!
//! Stego-side: J-UNIWARD costs computed over DAV1D's actual decoded
//! YUV via [`decode_with_recording_with_pixels`] — what a real
//! adversary sees. At flipped positions the inverse residual lands
//! in the reconstruction, so stego.recon differs from cover.recon by
//! ~q-step × basis magnitude around each flipped coefficient. The
//! cost histogram on stego.recon reflects this perturbation.
//!
//! # Expected result
//!
//! AUC > 0.52 (W6 baseline after B.1.3 cost-aware flipping) would
//! indicate AoSO catches signal that distribution-only features
//! miss. AUC ≈ 0.5 means our 4 cost features are too coarse and we
//! need finer cost-deviation work in v0.6+.

#![cfg(all(feature = "av1-encoder", feature = "av1-backend"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use phasm_core::codec::av1::stego::decoder::{
    decode_with_recording_with_pixels, DecodedFramePlanes,
};
use phasm_core::codec::av1::stego::orchestrator::av1_stego_embed;
use phasm_core::stego::cost::av1_uniward::{
    compute_av1_uniward_costs, Av1FramePosition, FramePlanes,
};
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

/// 12-feature vector per (cover-or-stego) sample. Three classes:
///
/// W6 distribution (F1-F2):
///   F1 = positive_ratio
///   F2 = adjacent_agreement_rate
///
/// J-UNIWARD-cost AoSO (F3-F6):
///   F3 = mean J-UNIWARD cost
///   F4 = std-dev of cost
///   F5 = median cost
///   F6 = skewness of cost
///
/// DCTR-light residual histogram (F7-F12):
///   F7  = mean |HPF_x| (horizontal first-difference filter)
///   F8  = 90th percentile |HPF_x|
///   F9  = kurtosis of HPF_x
///   F10 = mean |HPF_y| (vertical first-difference filter)
///   F11 = 90th percentile |HPF_y|
///   F12 = kurtosis of HPF_y
///
/// Per Holub/Fridrich 2014 DCTR's design lineage — frame-level
/// histogram statistics of high-pass-filtered reconstruction. Picks
/// up stego flips' perturbation of the residual distribution without
/// relying on J-UNIWARD's wavelet-activity cost (which is symmetric
/// in sign and only catches GLOBAL high-frequency-content asymmetry,
/// not residual shape).
type Features = [f64; 12];

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
        .expect("ffmpeg");
    assert!(out.status.success());
    let expected = (WIDTH * HEIGHT * 3 / 2) as usize;
    assert_eq!(out.stdout.len(), expected);
    out.stdout
}

fn encode_natural(yuv: &[u8]) -> (Vec<u8>, PhasmFrameRecording<u8>) {
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
    let mut fi = FrameInvariants::<u8>::new_key_frame(
        config.clone(),
        Arc::new(sequence),
        0,
        Box::new([]),
    );
    fi.enable_segmentation = false;
    let mut frame = make_frame::<u8>(WIDTH as usize, HEIGHT as usize, ChromaSampling::Cs420);
    let w = WIDTH as usize;
    let h = HEIGHT as usize;
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

/// Pack visible-region YUV from the encoder's reconstructed_planes
/// snapshot. Same helper as orchestrator's pack_visible_planes.
fn pack_visible_planes(rec: &Arc<phasm_rav1e::Frame<u8>>) -> FramePlanes {
    let pack = |idx: usize| -> (Vec<u8>, usize, usize) {
        let p = &rec.planes[idx];
        let w = p.cfg.width;
        let h = p.cfg.height;
        let stride = p.cfg.stride;
        let start = p.cfg.yorigin * stride + p.cfg.xorigin;
        let mut out = Vec::with_capacity(w * h);
        for row in 0..h {
            let row_start = start + row * stride;
            out.extend_from_slice(&p.data[row_start..row_start + w]);
        }
        (out, w, h)
    };
    let (y, luma_w, luma_h) = pack(0);
    let (cb, chroma_w, chroma_h) = pack(1);
    let (cr, _, _) = pack(2);
    FramePlanes {
        y,
        cb,
        cr,
        luma_width: luma_w,
        luma_height: luma_h,
        chroma_width: chroma_w,
        chroma_height: chroma_h,
    }
}

fn decoded_planes_to_frame_planes(p: &DecodedFramePlanes) -> FramePlanes {
    FramePlanes {
        y: p.y.clone(),
        cb: p.cb.clone(),
        cr: p.cr.clone(),
        luma_width: p.luma_width,
        luma_height: p.luma_height,
        chroma_width: p.chroma_width,
        chroma_height: p.chroma_height,
    }
}

/// Compute 6-feature vector.
///
/// - AC bits + meta come from the encoder recording (cover) or from
///   re-decoded stego packet (stego — strict parity guaranteed by
///   B.1.1.b cross-side test).
/// - Reconstructed pixels: encoder.recording.reconstructed_planes for
///   cover, dav1d's actual decoded YUV for stego (Phase B.1.4 — the
///   real-adversary view).
fn features_from_recording(
    av1_bytes: &[u8],
    recording: &PhasmFrameRecording<u8>,
    is_stego: bool,
) -> Features {
    let tile = &recording.tiles[0];

    // Filter to AC_COEFF_SIGN.
    let mut ac_bits: Vec<u8> = Vec::new();
    let mut ac_metas: Vec<Av1FramePosition> = Vec::new();
    for ((&(_, value), &tag), &meta) in tile
        .bit_positions
        .iter()
        .zip(tile.bit_tags.iter())
        .zip(tile.bit_meta.iter())
    {
        if tag == PHASM_TAG_AC_COEFF_SIGN {
            ac_bits.push(value as u8);
            ac_metas.push(Av1FramePosition {
                plane: meta.plane,
                plane_px_x: meta.plane_px_x,
                plane_px_y: meta.plane_px_y,
                tx_width_log2: meta.tx_width_log2,
                tx_height_log2: meta.tx_height_log2,
                tx_type: meta.tx_type,
                scan_pos: meta.scan_pos,
            });
        }
    }

    // For stego: replace bit values with what dav1d actually decoded
    // from the stego packet, and also keep the decoded planes for the
    // J-UNIWARD cost compute below (one decode pass, two outputs).
    let stego_planes: Option<DecodedFramePlanes> = if is_stego {
        let decoded = decode_with_recording_with_pixels(av1_bytes)
            .expect("decode stego with pixels");
        let observed_bits: Vec<u8> = decoded
            .iter()
            .filter(|p| p.tag == core_dav1d_sys::DAV1D_PHASM_TAG_AC_COEFF_SIGN)
            .map(|p| p.decoded_value)
            .collect();
        assert_eq!(
            observed_bits.len(),
            ac_bits.len(),
            "decoder must produce same AC count as encoder bit_meta"
        );
        ac_bits = observed_bits;
        decoded.planes
    } else {
        None
    };

    let n = ac_bits.len();
    assert!(n > 1, "too few AC bits ({n})");

    // F1, F2: existing W6 distribution features.
    let positive = ac_bits.iter().filter(|&&b| b == 1).count();
    let f1 = positive as f64 / n as f64;
    let mut agree = 0usize;
    for i in 1..n {
        if ac_bits[i] == ac_bits[i - 1] {
            agree += 1;
        }
    }
    let f2 = agree as f64 / (n - 1) as f64;

    // F3-F6: new AoSO cost-distribution features.
    //   Cover: encoder's reconstructed_planes.
    //   Stego: dav1d's actual decoded YUV (captured above in the
    //          single decode_with_recording_with_pixels call).
    let frame_planes = match &stego_planes {
        Some(p) => decoded_planes_to_frame_planes(p),
        None => pack_visible_planes(&recording.reconstructed_planes),
    };
    let costs: Vec<f64> = compute_av1_uniward_costs(
        &frame_planes,
        &ac_metas,
        recording.frame_qindex,
    )
    .into_iter()
    .filter(|c| c.is_finite())
    .map(|c| c as f64)
    .collect();

    let f3 = if !costs.is_empty() {
        costs.iter().sum::<f64>() / costs.len() as f64
    } else {
        0.0
    };
    let f4 = if costs.len() > 1 {
        let mean = f3;
        let var = costs.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / costs.len() as f64;
        var.sqrt()
    } else {
        0.0
    };
    let f5 = {
        let mut sorted = costs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if sorted.is_empty() {
            0.0
        } else {
            sorted[sorted.len() / 2]
        }
    };
    let f6 = if f4 > 1e-9 {
        let mean = f3;
        let std = f4;
        costs.iter().map(|c| ((c - mean) / std).powi(3)).sum::<f64>() / costs.len() as f64
    } else {
        0.0
    };

    // F7-F12: DCTR-light residual-histogram features on the SAME
    // reconstruction used above for J-UNIWARD (encoder.recon for
    // cover, dav1d.recon for stego). Two high-pass first-difference
    // filters along x and y; statistics over the resulting residual
    // are mean abs, 90th percentile abs, kurtosis.
    let (f7, f8, f9) = hpf_stats(
        &frame_planes.y,
        frame_planes.luma_width,
        frame_planes.luma_height,
        HpfDirection::Horizontal,
    );
    let (f10, f11, f12) = hpf_stats(
        &frame_planes.y,
        frame_planes.luma_width,
        frame_planes.luma_height,
        HpfDirection::Vertical,
    );

    [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]
}

#[derive(Copy, Clone)]
enum HpfDirection {
    Horizontal,
    Vertical,
}

/// Apply a [-1, +1] high-pass first-difference filter along x or y
/// to the Y plane, then compute (mean |out|, p90 |out|, kurtosis(out))
/// over the full residual map. Mirrors a minimal DCTR-style feature.
fn hpf_stats(
    y: &[u8],
    w: usize,
    h: usize,
    dir: HpfDirection,
) -> (f64, f64, f64) {
    let mut residual: Vec<f64> = match dir {
        HpfDirection::Horizontal => {
            let mut r = Vec::with_capacity((w - 1) * h);
            for row in 0..h {
                let base = row * w;
                for x in 1..w {
                    r.push(y[base + x] as f64 - y[base + x - 1] as f64);
                }
            }
            r
        }
        HpfDirection::Vertical => {
            let mut r = Vec::with_capacity(w * (h - 1));
            for row in 1..h {
                let base = row * w;
                let prev = (row - 1) * w;
                for x in 0..w {
                    r.push(y[base + x] as f64 - y[prev + x] as f64);
                }
            }
            r
        }
    };

    let n = residual.len() as f64;
    let mean = residual.iter().sum::<f64>() / n;
    let mean_abs = residual.iter().map(|v| v.abs()).sum::<f64>() / n;

    let var = residual.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let kurt = if std > 1e-9 {
        residual
            .iter()
            .map(|v| ((v - mean) / std).powi(4))
            .sum::<f64>()
            / n
            - 3.0
    } else {
        0.0
    };

    // 90th percentile of |residual|.
    let mut abs_sorted: Vec<f64> = residual.iter_mut().map(|v| v.abs()).collect();
    abs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p90 = abs_sorted[(0.9 * (abs_sorted.len() - 1) as f64) as usize];

    (mean_abs, p90, kurt)
}

fn mean_features(samples: &[Features]) -> Features {
    let n = samples.len() as f64;
    let mut m = [0.0; 12];
    for f in samples {
        for i in 0..12 {
            m[i] += f[i];
        }
    }
    for i in 0..12 {
        m[i] /= n;
    }
    m
}

/// Compute Mann-Whitney AUC on a 1-d projection of features onto a
/// discrimination axis. Symmetrized to [0.5, 1.0].
fn auc_along(
    cover: &[Features],
    stego: &[Features],
    feature_indices: &[usize],
) -> f64 {
    let mean_c = mean_features(cover);
    let mean_s = mean_features(stego);
    let axis: Vec<f64> = feature_indices
        .iter()
        .map(|&i| mean_s[i] - mean_c[i])
        .collect();
    let project = |f: &Features| -> f64 {
        feature_indices
            .iter()
            .enumerate()
            .map(|(j, &i)| f[i] * axis[j])
            .sum()
    };
    let cover_proj: Vec<f64> = cover.iter().map(project).collect();
    let stego_proj: Vec<f64> = stego.iter().map(project).collect();
    let mut higher = 0u32;
    let mut tied = 0u32;
    for &s in &stego_proj {
        for &c in &cover_proj {
            if s > c {
                higher += 1;
            } else if s == c {
                tied += 1;
            }
        }
    }
    let total = (cover_proj.len() * stego_proj.len()) as f64;
    let raw = (higher as f64 + 0.5 * tied as f64) / total;
    raw.max(1.0 - raw)
}

#[test]
fn b1_4_aoso_detector_with_cost_features() {
    eprintln!(
        "[aoso] corpus: {} fixtures × {} seek points = {} pairs",
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

            // Skip frames with too little AC capacity (mirror W6).
            let tile = &recording.tiles[0];
            let n_cover = tile
                .bit_tags
                .iter()
                .filter(|&&t| t == PHASM_TAG_AC_COEFF_SIGN)
                .count();
            if n_cover < 256 {
                skipped += 1;
                continue;
            }

            let target_payload_bytes = (n_cover / 2 / 8).max(25);
            let plaintext_len = (target_payload_bytes - 24).max(1).min(256);
            payload_seed = payload_seed.wrapping_mul(0x9E3779B97F4A7C15);
            let mut msg = vec![0u8; plaintext_len];
            for (k, b) in msg.iter_mut().enumerate() {
                let s = payload_seed
                    .wrapping_add(k as u64)
                    .wrapping_mul(0xCAFEBABE);
                *b = (s >> 24) as u8;
            }
            let passphrase = format!("aoso-pass-{:x}", payload_seed);

            match av1_stego_embed(natural.clone(), recording.clone(), &msg, &passphrase) {
                Ok(stego) => {
                    cover_features.push(features_from_recording(&natural, &recording, false));
                    stego_features.push(features_from_recording(&stego, &recording, true));
                }
                Err(e) => {
                    skipped += 1;
                    eprintln!("[aoso]   skip {} @ {:.2}s: {:?}", fx.name, seek_s, e);
                }
            }
        }
    }

    let n = cover_features.len();
    eprintln!("[aoso] {} pairs usable ({} skipped)", n, skipped);
    assert!(n >= 10, "too few usable pairs: {}", n);

    // Report AUC per feature class so we can attribute detection
    // power. Three classes:
    //   W6: distribution-only on sign bits (F1+F2)
    //   J-UNIWARD-cost AoSO: wavelet-activity-via-cost (F3-F6)
    //   DCTR-light: high-pass residual histogram (F7-F12)
    let w6_features = [0, 1];
    let juw_features = [2, 3, 4, 5];
    let dctr_features = [6, 7, 8, 9, 10, 11];
    let combined = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

    let auc_w6 = auc_along(&cover_features, &stego_features, &w6_features);
    let auc_juw = auc_along(&cover_features, &stego_features, &juw_features);
    let auc_dctr = auc_along(&cover_features, &stego_features, &dctr_features);
    let auc_combined = auc_along(&cover_features, &stego_features, &combined);

    let mean_c = mean_features(&cover_features);
    let mean_s = mean_features(&stego_features);
    eprintln!("[aoso] cover mean F = {:?}", mean_c);
    eprintln!("[aoso] stego mean F = {:?}", mean_s);
    eprintln!(
        "[aoso] AUC W6-only       (F1+F2)        = {:.4}",
        auc_w6
    );
    eprintln!(
        "[aoso] AUC J-UNIWARD-AoSO (F3..F6)       = {:.4}",
        auc_juw
    );
    eprintln!(
        "[aoso] AUC DCTR-light    (F7..F12)      = {:.4}",
        auc_dctr
    );
    eprintln!(
        "[aoso] AUC combined-all  (F1..F12)      = {:.4}",
        auc_combined
    );

    // v1.0 stealth gate per Phase B.1 design doc:
    //   AUC ≤ 0.7 → ship (acceptable adversary advantage)
    //   AUC > 0.7 → block (signal too strong to ship)
    //
    // Reading: report the STRONGEST single-class AUC (the worst-case
    // for stealth — adversary picks the best detector). v1.0 gate
    // applies to that, NOT combined-all (combined overfits with
    // mean-centroid projection on small corpora).
    let worst_class_auc = auc_w6.max(auc_juw).max(auc_dctr);
    eprintln!(
        "[aoso] STRONGEST-CLASS AUC (vs 0.70 gate) = {:.4}",
        worst_class_auc
    );

    assert!(
        worst_class_auc < 0.70,
        "Strongest-class AUC {:.4} exceeds v1.0 stealth gate (0.70)",
        worst_class_auc
    );
}
