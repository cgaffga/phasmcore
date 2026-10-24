// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! MVD cost-model instrument (2026-10-20).
//!
//! Characterises the per-position cost distributions the 4-domain STC
//! trellis actually sees on a real GOP, so the MVD cost function can be
//! chosen from measurement instead of guessed.
//!
//! ## Why this exists
//!
//! `fill_frame_costs` prices an MVD flip as `1 / (texture + MVD_COST_SIGMA)`
//! — the standard adaptive-steganography heuristic "hide where the content
//! is busy, because changes are masked there". That is right for a
//! *coefficient* change, which perturbs one block's texture. It is backwards
//! for a *motion vector*.
//!
//! When the wire-only emit flips an MVD bit, the encoder's residual is NOT
//! recomputed (by design — encoder state stays clean). The decoder therefore
//! reconstructs `pred(MV') + residual(MV)`, and the reconstruction error is
//!
//! ```text
//!     err = pred(MV) - pred(MV')  ≈  |∇I| · Δ
//! ```
//!
//! i.e. the local spatial gradient times the displacement. `texture` (mean
//! |LH|+|HL|+|HH| over the macroblock) is exactly a gradient-magnitude proxy,
//! so a *high*-texture macroblock makes an MVD flip MORE damaging, not less.
//! Pricing it at `1/texture` tells STC to prefer precisely the macroblocks
//! where the damage is worst.
//!
//! The displacement is also ignored today. A `MvdSignBypass` flip sends the
//! vector from `+d` to `-d`, displacing the block by `2·|d|` quarter-pels —
//! which on a fast-moving block is many whole pixels. A `MvdSuffixLsb` flip
//! moves it by a single quarter-pel. Both currently cost the same and carry
//! the same domain weight (10).
//!
//! Measured consequence on an 18 s 1080p clip: whole 16×16 macroblocks
//! visibly displaced, stego-only PSNR-Y 28.8 dB against a clean encode of
//! identical pixels. See `docs/notes/2026-10-20-video-encode-ux-defects.md`.
//!
//! ## What this prints
//!
//! Per domain: count, and the quantiles of the cost vector STC receives.
//! For the MVD domains it additionally recovers `texture` exactly (the old
//! cost is invertible: `texture = 1/cost - σ`) and reports the joint
//! distribution of texture and MVD magnitude, plus what a
//! gradient×displacement² model would charge. That is the number needed to
//! pick a scale that keeps MVD competitive with the coefficient domains
//! instead of silently pricing it out of the trellis.
//!
//! Run:
//! ```text
//! cargo test -p phasm-core --features h264-encoder --test mvd_cost_model \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "h264-encoder")]

mod common;

use phasm_core::codec::h264::openh264_stego::{gop_clean_cover, EncodeOpts};
use phasm_core::codec::h264::stego::content_costs::compute_content_costs_yuv;
// The cached scale-to-I420 is shared in `common::video_fixture`.
use common::video_fixture::{ensure_scaled_yuv, ScaleRecipe};

/// Mirror of `MVD_COST_SIGMA` in `content_costs.rs`. Used to invert the old
/// cost back to the texture it was derived from.
const MVD_COST_SIGMA: f32 = 1e-3;

/// Decode `n_frames` of a corpus fixture to 16-aligned I420, cached under
/// `$PHASM_TMP`.
fn ensure_yuv(source_mp4: &str, tag: &str, w: u32, h: u32, n_frames: u32) -> Vec<u8> {
    ensure_scaled_yuv(source_mp4, "phasm_mvdcost", tag, w, h, n_frames, ScaleRecipe::PixFmt)
}

/// Quantiles of the finite entries, plus how many were WET (∞).
fn describe(name: &str, costs: &[f32]) {
    let mut finite: Vec<f32> = costs.iter().copied().filter(|c| c.is_finite()).collect();
    let wet = costs.len() - finite.len();
    if finite.is_empty() {
        eprintln!("  {name:<14} n={:<7} wet={wet:<7} (all wet)", costs.len());
        return;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| finite[((finite.len() - 1) as f64 * f) as usize];
    eprintln!(
        "  {name:<14} n={:<7} wet={wet:<7} min={:<12.4} p25={:<12.4} med={:<12.4} \
         p75={:<12.4} p99={:<12.4} max={:.4}",
        costs.len(), finite[0], q(0.25), q(0.50), q(0.75), q(0.99),
        finite[finite.len() - 1],
    );
}

/// The gradient×displacement model this instrument is evaluating.
///
/// `texture` is the macroblock's mean wavelet high-frequency energy (a
/// gradient proxy); `displacement_qpel` is how far the flip moves the block.
/// Error ≈ gradient · displacement, so squared error ≈ (texture · Δpx)².
fn model_cost(texture: f32, displacement_qpel: f32) -> f32 {
    let dpx = displacement_qpel / 4.0;
    let e = texture * dpx;
    e * e
}

fn run(source_mp4: &str, tag: &str, w: u32, h: u32, n_frames: u32, qp: i32) {
    let yuv = ensure_yuv(source_mp4, tag, w, h, n_frames);
    let opts = EncodeOpts { qp, intra_period: n_frames as i32 };
    let cover = gop_clean_cover(&yuv, w, h, n_frames, 0, opts).expect("clean cover");
    let costs = compute_content_costs_yuv(&yuv, w, h, n_frames, &cover, qp).expect("costs");

    eprintln!("\n=== {tag}  {w}x{h} x{n_frames}f  qp={qp} ===");
    eprintln!("--- cost vectors as the STC trellis receives them (pre domain-weight)");
    describe("CoeffSign", &costs.coeff_sign_bypass);
    describe("CoeffSuffix", &costs.coeff_suffix_lsb);
    describe("MvdSign", &costs.mvd_sign_bypass);
    describe("MvdSuffix", &costs.mvd_suffix_lsb);

    // Recover texture exactly: the old MVD cost is 1/(texture + σ).
    let recover = |c: f32| if c.is_finite() && c > 0.0 { 1.0 / c - MVD_COST_SIGMA } else { f32::NAN };

    for (label, costs_v, mags, is_sign) in [
        ("MvdSign", &costs.mvd_sign_bypass, &cover.mvd_sign_bypass.magnitudes, true),
        ("MvdSuffix", &costs.mvd_suffix_lsb, &cover.mvd_suffix_lsb.magnitudes, false),
    ] {
        let n = costs_v.len().min(mags.len());
        if n == 0 {
            eprintln!("--- {label}: no positions");
            continue;
        }
        let mut textures = Vec::with_capacity(n);
        let mut model = Vec::with_capacity(n);
        let mut disps = Vec::with_capacity(n);
        for i in 0..n {
            let t = recover(costs_v[i]);
            if !t.is_finite() {
                continue;
            }
            // Sign flip: +d -> -d, so the block moves 2|d| quarter-pels.
            // Suffix LSB: a single quarter-pel, whatever |d| is.
            let d = if is_sign { 2.0 * mags[i] as f32 } else { 1.0 };
            textures.push(t);
            disps.push(d);
            model.push(model_cost(t, d).max(f32::MIN_POSITIVE));
        }
        if textures.is_empty() {
            eprintln!("--- {label}: all wet");
            continue;
        }
        eprintln!("--- {label}: recovered inputs + candidate model");
        describe("  texture", &textures);
        describe("  displace_qp", &disps);
        describe("  model_cost", &model);

        // The thing that matters: does the OLD cost rank positions the
        // opposite way round from the damage model? Spearman-ish check via
        // the sign of the correlation between old cost and model cost.
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let old: Vec<f32> = (0..n)
            .filter(|&i| recover(costs_v[i]).is_finite())
            .map(|i| costs_v[i])
            .collect();
        let (mo, mm) = (mean(&old), mean(&model));
        let cov: f32 = old.iter().zip(&model).map(|(a, b)| (a - mo) * (b - mm)).sum();
        let va: f32 = old.iter().map(|a| (a - mo) * (a - mo)).sum();
        let vb: f32 = model.iter().map(|b| (b - mm) * (b - mm)).sum();
        let r = if va > 0.0 && vb > 0.0 { cov / (va.sqrt() * vb.sqrt()) } else { f32::NAN };
        eprintln!(
            "  corr(old_cost, model_cost) = {r:+.4}  \
             (negative ⇒ the shipped cost prefers exactly what the damage model penalises)"
        );
    }
}

/// Fast smoke so the instrument itself stays covered by a normal test run.
#[test]
fn mvd_cost_distribution_smoke() {
    run("IMG_4138.MOV", "iphone7_smoke", 640, 368, 8, 26);
}

#[test]
#[ignore = "full 1080p GOP; run with --ignored --nocapture --test-threads=1"]
fn mvd_cost_distribution_1080p() {
    run("IMG_4138.MOV", "iphone7_1080p", 1920, 1072, 30, 26);
}

#[test]
#[ignore = "full 1080p GOP; run with --ignored --nocapture --test-threads=1"]
fn mvd_cost_distribution_carplane_1080p() {
    run("Artlist_CarPlane.mp4", "carplane_1080p", 1920, 1072, 30, 26);
}
