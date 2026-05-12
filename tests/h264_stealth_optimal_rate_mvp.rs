// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §Stealth.L1+L2.1 — cascade-safety regression test against Li et al.
// 2023 "Optimal-rate-of-MVP" (arXiv:2308.06464).
//
// Per the strategy doc (`docs/design/video/h264/stealth-strategy.md`):
//
// > §Stealth.1.a — cascade-safety CI regression test that runs a
// > clean-room reimplementation of Li et al. 2023 "Optimal-rate-of-MVP"
// > on phasm output, asserts AUC ≤ 0.55. Cheap; defends against future
// > regressions in §6F.2(j) / §6E-A5(d).
//
// Concrete reduction. The Li 2023 1-D feature is: ratio of locally-
// optimal MVPs. Cover = 100% (every encoded MV is the encoder's
// rate-distortion-minimising choice given its predicted candidate set).
// Stego < 100% IFF the stego process moves an MV off-optimum. Phasm's
// §6F.2(j) cascade-safe injection structurally only flips bypass bins
// that don't perturb |MVD| — and the encoder's optimality test is a
// function of |MVD| through the rate term. So if |MVD| is byte-identical
// across all positions between cover and stego, the Li 2023 statistic
// reads pure cover (1.0) by construction.
//
// This regression test asserts the structural invariant directly:
// cover_meta[i].magnitude == stego_meta[i].magnitude for every i.
// AUC for a perfect-tie 1-D feature is 0.5; the strategy doc's
// "AUC ≤ 0.55" bound is met whenever the magnitudes pairwise match.

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::cabac::bin_decoder::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};
use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::stego::gop_pattern::{
    iter_encode_order, FrameType, GopPattern,
};
use phasm_core::h264_stego_encode_yuv_string_4domain_multigop;

fn load_real_world(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/video/h264/real-world/{name}"))
        .unwrap_or_else(|_| panic!("missing real-world fixture: {name}"))
}

/// Encode YUV with the matching production-encoder config but NO stego
/// hook. Mirrors `core/src/codec/h264/stego/encode_pixels.rs:build_encoder`
/// post-§Stealth.L3.1 follow-on (#145): CABAC + transform_8x8 = true →
/// High profile. With no stego hook, no bin overrides happen, so the
/// output is the unmodified bitstream the stego pipeline would produce
/// on a zero-payload encode.
fn encode_clean_reference_high(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
) -> Vec<u8> {
    let frame_size = (width * height * 3 / 2) as usize;
    let quality = Some(26);
    let mut enc = Encoder::new(width, height, quality).expect("clean encoder");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true; // High profile (matches build_encoder)
    enc.enable_mvd_stego_hook = true; // disable P_SKIP for shape parity
    let pattern = GopPattern::Ibpbp { gop: gop_size, b_count: 1 };
    enc.enable_b_frames = pattern.has_b_frames();
    let mut out = Vec::new();
    for meta in iter_encode_order(n_frames, pattern) {
        let d = meta.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match meta.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame).expect("clean i-frame"),
            FrameType::P => enc.encode_p_frame(frame).expect("clean p-frame"),
            FrameType::B => enc.encode_b_frame(frame).expect("clean b-frame"),
        };
        out.extend_from_slice(&bytes);
    }
    out
}

#[test]
fn cascade_safety_invariant_holds_against_optimal_rate_of_mvp() {
    // 64×48 × 5 frames real-world fixture (iPhone 8 footage, smooth
    // hand-held motion → plenty of nonzero MVDs across P-frames).
    let yuv = load_real_world("img4138_64x48_f5.yuv");
    let width = 64u32;
    let height = 48u32;
    let n_frames = 5;
    let gop_size = 5;

    // Match the §6F.2(k).5 stealth measurement's fixture/passphrase
    // pair — known to plan + decode successfully at this small size.
    let stego = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, width, height, n_frames, gop_size,
        /* message */ "h", /* passphrase */ "test-pass-64",
    )
    .expect("stego encode");
    let clean = encode_clean_reference_high(&yuv, width, height, n_frames, gop_size);

    // Walk both with record_mvd to populate cover.mvd_sign_bypass +
    // mvd_meta (per-position |value|, partition, axis).
    let opts = WalkOptions { record_mvd: true };
    let stego_walk = walk_annex_b_for_cover_with_options(&stego, opts).expect("stego walk");
    let clean_walk = walk_annex_b_for_cover_with_options(&clean, opts).expect("clean walk");

    // Sanity gate: at this fixture size the MVD cover must be
    // non-empty, otherwise the test is vacuous.
    assert!(
        !clean_walk.mvd_meta.is_empty(),
        "clean reference produced zero MVD positions — fixture broken or encoder config drift",
    );
    assert_eq!(
        stego_walk.mvd_meta.len(),
        clean_walk.mvd_meta.len(),
        "stego and clean MVD position counts must match (cascade-safe \
         injection flips bypass bits, never adds/removes positions)",
    );

    // Optimal-rate-of-MVP 1-D feature: ratio of MVDs whose magnitude is
    // unchanged from the encoder's rate-optimal choice. With cascade-
    // safe injection the magnitude IS the encoder's rate-optimal value
    // by construction (sign-flip leaves |value| invariant), so every
    // position should match.
    let mut matches = 0usize;
    let total = clean_walk.mvd_meta.len();
    let mut first_mismatch: Option<(usize, u32, u32)> = None;
    for (i, (c, s)) in clean_walk
        .mvd_meta
        .iter()
        .zip(stego_walk.mvd_meta.iter())
        .enumerate()
    {
        if c.magnitude == s.magnitude {
            matches += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = Some((i, c.magnitude, s.magnitude));
        }
    }

    let optimal_ratio = matches as f64 / total as f64;
    eprintln!(
        "Optimal-rate-of-MVP 1-D feature: {}/{} = {:.6} (target = 1.0)",
        matches, total, optimal_ratio,
    );

    assert_eq!(
        matches, total,
        "cascade-safety invariant broken: |MVD| diverged at position {}/{} \
         (clean magnitude = {}, stego magnitude = {})",
        first_mismatch.map(|m| m.0).unwrap_or(0),
        total,
        first_mismatch.map(|m| m.1).unwrap_or(0),
        first_mismatch.map(|m| m.2).unwrap_or(0),
    );

    // AUC for a perfectly-tied 1-D feature is 0.5 (random
    // discrimination). The strategy doc's threshold is AUC ≤ 0.55;
    // exact-match yields 0.5, well below.
    assert!(
        (optimal_ratio - 1.0).abs() < f64::EPSILON,
        "Optimal-rate-of-MVP feature must be exactly 1.0 (got {})",
        optimal_ratio,
    );
}
