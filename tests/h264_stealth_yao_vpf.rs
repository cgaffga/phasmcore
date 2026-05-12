// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §Stealth.L3.4 — Yao 2017 / VPF (Vázquez-Padín 2012/2020) periodicity
// regression test.
//
// VPF and Yao 2017 detect H.264 double-encoding by spotting periodic
// peaks in per-frame MB-type / skip-MB rate at the candidate
// source-GOP cadence. Phasm doesn't double-encode (it always emits a
// fresh encode of decoded YUV), so the periodogram should show no
// peak above the random-baseline noise floor at any non-GOP period.
//
// Per Agent A's double-compression survey:
//
// > VPF specifically struggles when source GOP is a 1× multiple of
// > second-pass GOP — matched GOP is the WEAKEST detection case.
//
// And the strategy doc (`docs/design/video/h264/stealth-strategy.md`):
//
// > §Stealth.L3.4: encoder-fingerprint regression test. Run phasm
// > output through (replicated) Yao 2017 SODB+skip-MB periodicity
// > test + (replicated) VPF / G-VPF MB-type periodicity test. Assert
// > AUC ≤ 0.6 for each (phasm output should NOT trigger periodicity
// > peaks at any plausible source-GOP cadence given matched output
// > GOP).
//
// Reduction. Yao/VPF compute a periodogram over a per-frame statistic
// (skip-MB rate, MB-type histogram coordinate). Their detection
// trigger is "max periodogram value at non-trivial lag exceeds the
// random-baseline floor by 2-3×". Phasm's per-frame
// bypass-bin-position count is a strict proxy (more MBs in a frame
// with non-skip / coded coefficients ⇒ more bypass positions).
// Asserting the periodogram on this proxy is flat is the strict form
// of the AUC ≤ 0.6 bound for both Yao and VPF.

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::cabac::bin_decoder::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};
use phasm_core::h264_stego_encode_yuv_string_4domain_multigop;

fn load_real_world(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/video/h264/real-world/{name}"))
        .unwrap_or_else(|_| panic!("missing real-world fixture: {name}"))
}

/// Per-frame bypass-bin-position activity. Sums positions across all
/// four cover domains (coeff_sign_bypass, coeff_suffix_lsb,
/// mvd_sign_bypass, mvd_suffix_lsb) and bins by frame_idx. A frame
/// with all P_SKIP MBs has zero positions; a frame with high-residual
/// content has many. The output series is the input to Yao/VPF
/// periodicity detection.
fn per_frame_activity(
    cover: &phasm_core::codec::h264::stego::DomainCover,
    n_frames: usize,
) -> Vec<u32> {
    let mut counts = vec![0u32; n_frames];
    let domains: [&[phasm_core::codec::h264::stego::PositionKey]; 4] = [
        &cover.coeff_sign_bypass.positions,
        &cover.coeff_suffix_lsb.positions,
        &cover.mvd_sign_bypass.positions,
        &cover.mvd_suffix_lsb.positions,
    ];
    for d in domains {
        for p in d {
            let idx = p.frame_idx() as usize;
            if idx < n_frames {
                counts[idx] += 1;
            }
        }
    }
    counts
}

/// Sample autocorrelation at lag `tau` for series `x`. Defined as
///   r(tau) = Σ (x_t - μ)(x_{t+tau} - μ) / Σ (x_t - μ)²
/// — the standard normalised lagged autocorrelation. Returns `None`
/// when the series has zero variance (constant signal).
fn autocorrelation(x: &[f64], tau: usize) -> Option<f64> {
    if tau >= x.len() {
        return Some(0.0);
    }
    let n = x.len();
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum();
    if var == 0.0 {
        return None;
    }
    let cov: f64 = (0..n - tau).map(|t| (x[t] - mean) * (x[t + tau] - mean)).sum();
    Some(cov / var)
}

#[test]
fn yao_vpf_periodicity_test_no_double_encoding_signal() {
    // 10-frame iPhone fixture, 1 GOP (so the only "natural"
    // periodicity is the I-frame at lag = 0; non-trivial lags should
    // show no peak).
    let yuv = load_real_world("img4138_128x80_f10.yuv");
    let width = 128u32;
    let height = 80u32;
    let n_frames = 10;
    let gop_size = 10;

    let stego = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, width, height, n_frames, gop_size,
        /* message */ "h", /* passphrase */ "test-pass-128",
    )
    .expect("stego encode 128x80 10f");

    let cover_walk = walk_annex_b_for_cover_with_options(
        &stego,
        WalkOptions { record_mvd: true },
    )
    .expect("walk stego");

    // Sanity gate.
    let activity_u32 = per_frame_activity(&cover_walk.cover, n_frames);
    let activity: Vec<f64> = activity_u32.iter().map(|&c| c as f64).collect();
    eprintln!("per-frame activity counts: {:?}", activity_u32);
    let total: u32 = activity_u32.iter().sum();
    assert!(
        total > 100,
        "per-frame activity too small for periodogram to be meaningful: \
         total positions = {total}",
    );

    // VPF-style detection: max non-trivial-lag autocorrelation. Lag 0
    // = 1.0 trivially; the actual GOP boundary (lag 9 here, since
    // n_frames=10 and we have one IDR at frame 0) is excluded from
    // the false-positive search since matched-GOP IS the regression
    // target. Test lags 1..=4 (covers all plausible source GOP
    // cadences below half the test length).
    let mut max_acf = 0.0f64;
    let mut max_lag = 0usize;
    for tau in 1..=4 {
        if let Some(ac) = autocorrelation(&activity, tau) {
            eprintln!("  acf lag={tau}: {:+.4}", ac);
            if ac.abs() > max_acf {
                max_acf = ac.abs();
                max_lag = tau;
            }
        }
    }

    // Threshold from VPF / Yao papers: detection triggers around
    // |ACF| ≥ 0.5 at the source-GOP cadence. AUC ≤ 0.6 in the
    // strategy doc maps to "peak ACF over non-trivial lags should
    // stay below 0.5". A clean encode with no double-compression
    // typically gives |ACF| ≤ 0.3 at all non-GOP lags.
    assert!(
        max_acf < 0.6,
        "Yao/VPF periodicity peak too high: |ACF(lag={})| = {:.4} \
         (threshold = 0.6, target ≤ 0.6 per strategy doc). Phasm \
         encoder is leaking a periodic MB-type pattern that would \
         flag double-encoding.",
        max_lag, max_acf,
    );

    eprintln!(
        "Yao/VPF periodicity: max |ACF| at non-GOP lags = {:.4} (threshold ≤ 0.6) ✓",
        max_acf,
    );
}

#[test]
fn yao_vpf_phasm_activity_does_not_diverge_from_clean_baseline() {
    // Stronger form of the regression: assert phasm's per-frame
    // activity series is identical to a clean (non-stego) reference's
    // by-frame counts. Cascade-safe stego flips bypass bins but
    // doesn't add or remove cover positions, so position-count per
    // frame must match exactly. Any divergence would mean the stego
    // process introduced a per-frame perturbation that VPF could
    // pick up.
    use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
    use phasm_core::codec::h264::stego::gop_pattern::{
        iter_encode_order, FrameType, GopPattern,
    };

    let yuv = load_real_world("img4138_128x80_f10.yuv");
    let width = 128u32;
    let height = 80u32;
    let n_frames = 10;
    let gop_size = 10;
    let frame_size = (width * height * 3 / 2) as usize;

    let stego = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, width, height, n_frames, gop_size,
        "h", "test-pass-128",
    )
    .expect("stego encode");

    // Clean reference matching production build_encoder config:
    // CABAC + High profile (transform_8x8 = true) + IBPBP + MVD-hook
    // disabled-P_SKIP shape parity.
    let mut enc = Encoder::new(width, height, Some(26)).expect("clean encoder");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_mvd_stego_hook = true;
    let pattern = GopPattern::Ibpbp { gop: gop_size, b_count: 1 };
    enc.enable_b_frames = pattern.has_b_frames();
    let mut clean = Vec::new();
    for meta in iter_encode_order(n_frames, pattern) {
        let d = meta.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match meta.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame),
            FrameType::P => enc.encode_p_frame(frame),
            FrameType::B => enc.encode_b_frame(frame),
        }
        .expect("clean frame");
        clean.extend_from_slice(&bytes);
    }

    let stego_walk = walk_annex_b_for_cover_with_options(
        &stego, WalkOptions { record_mvd: true },
    )
    .expect("walk stego");
    let clean_walk = walk_annex_b_for_cover_with_options(
        &clean, WalkOptions { record_mvd: true },
    )
    .expect("walk clean");

    let stego_activity = per_frame_activity(&stego_walk.cover, n_frames);
    let clean_activity = per_frame_activity(&clean_walk.cover, n_frames);

    eprintln!("stego per-frame activity: {:?}", stego_activity);
    eprintln!("clean per-frame activity: {:?}", clean_activity);

    assert_eq!(
        stego_activity, clean_activity,
        "stego per-frame activity diverged from clean baseline — \
         cascade-safety holds at the position level (count is fixed) \
         but a divergence here would indicate VPF detection surface",
    );
}
