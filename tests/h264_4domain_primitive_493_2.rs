// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #493.3 Phase 2 — pure-Rust per-GOP 4-domain primitive integration test.
//!
//! Verifies the new `h264_stego_encode_one_gop_with_chunk_bits_4domain`
//! primitive at the primitive level (decoupled from the streaming
//! session, which is unchanged at Phase 2 — see Phase 4 / #493.5
//! for the streaming-session integration).
//!
//! Test shape:
//!   1. Encode a small fixture with random `chunk_bits` payload
//!      via the 4-domain primitive.
//!   2. Walk the emitted Annex-B with `record_mvd: true` to get
//!      4-domain cover.
//!   3. Combine the walked cover in canonical order (CS → CSL →
//!      MVDs → MVDsl) using the SAME `CostWeights` the encoder used.
//!      (Costs aren't needed for extract; pass dummy `DomainCosts`.)
//!   4. Run `stc_extract` on the combined cover with the same hhat
//!      seed and `w = n_cover / m_total`.
//!   5. Assert the extracted bits match the original `chunk_bits`.
//!
//! This is essentially Phase 4's decode logic done inline at the
//! primitive level — it proves the 4-domain encoder + the same
//! canonical-combine reverse are self-consistent, which is the only
//! foundation property Phase 4's streaming-session integration adds
//! on top of (Phase 4 = "wire this same combine+extract into
//! StreamingDecodeSession::finish").

#![cfg(feature = "h264-encoder")]

use phasm_core::codec::h264::cabac::bin_decoder::slice::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};
use phasm_core::codec::h264::stego::{
    combine_cover_4domain, CostWeights,
};
use phasm_core::codec::h264::stego::encode_pixels::h264_stego_encode_one_gop_with_chunk_bits_4domain;
use phasm_core::codec::h264::stego::orchestrate::DomainCosts;
use phasm_core::stego::stc::extract::stc_extract;
use phasm_core::stego::stc::hhat::generate_hhat;

const STC_H: usize = 4;

/// Walker options for 4-domain cover extraction — `record_mvd: true`
/// is the critical flag (Phase 0 finding).
fn full_walk_options() -> WalkOptions {
    WalkOptions {
        record_mvd: true,
        ..Default::default()
    }
}

/// Same textured synth as #493.1 walker-parity test — produces
/// realistic 4-domain cover density on small fixtures.
fn synth_yuv(width: u32, height: u32, frame_idx: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3 / 2) as usize);
    let w = width as i32;
    let h = height as i32;
    let half_w = w / 2;
    let half_h = h / 2;
    for j in 0..h {
        for i in 0..w {
            let val = ((i + frame_idx as i32 * 2) ^ (j + frame_idx as i32 * 3)) as u8;
            out.push(val);
        }
    }
    let mut s: u32 = 0xCAFE_F00D ^ frame_idx;
    for _plane in 0..2 {
        for j in 0..half_h {
            for i in 0..half_w {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let tex = (s >> 16) as u8;
                let pos = (i + j + frame_idx as i32) as u8;
                out.push(tex.wrapping_add(pos));
            }
        }
    }
    out
}

fn build_yuv_clip(width: u32, height: u32, n_frames: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for f in 0..n_frames {
        out.extend_from_slice(&synth_yuv(width, height, f));
    }
    out
}

/// Run encode → walk → combine → STC extract → compare. Returns the
/// extracted bits for the caller to assert against the original.
fn encode_and_extract(
    width: u32,
    height: u32,
    n_frames: u32,
    chunk_bits: &[u8],
    hhat_seed: &[u8; 32],
    weights: &CostWeights,
) -> Vec<u8> {
    let yuv = build_yuv_clip(width, height, n_frames);
    let annex_b = h264_stego_encode_one_gop_with_chunk_bits_4domain(
        &yuv, width, height, n_frames as usize,
        chunk_bits, hhat_seed, /*quality*/ None,
        weights,
    )
    .expect("4-domain encode");

    let walked = walk_annex_b_for_cover_with_options(&annex_b, full_walk_options())
        .expect("walk_annex_b_for_cover");

    // Decoder-side: combine in canonical order with the same weights
    // (costs are irrelevant for extract — STC extract needs cover +
    // hhat only). Pass empty DomainCosts; the combine helper pads
    // with 1.0 and the result × weight is finite, but unused by
    // stc_extract.
    let dummy_costs = DomainCosts::default();
    let (combined_cover, _, _boundaries) =
        combine_cover_4domain(&walked.cover, &dummy_costs, weights);

    let n_cover = combined_cover.len();
    let m_total = chunk_bits.len();
    assert!(n_cover >= m_total, "n_cover {n_cover} < m_total {m_total} — fixture too small");
    let w = n_cover / m_total;
    assert!(w >= 1, "STC w must be ≥ 1, got {w}");
    let hhat = generate_hhat(STC_H, w, hhat_seed);
    let used_cover = m_total * w;
    let extracted = stc_extract(&combined_cover[..used_cover], &hhat, w);
    extracted[..m_total].to_vec()
}

#[test]
fn four_domain_primitive_roundtrip_tiny() {
    // Smallest viable fixture — 128×80 × 4f. Small chunk_bits to
    // stay well within combined cover capacity.
    let chunk_bits = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
    let hhat_seed = [0x42u8; 32];
    let weights = CostWeights::default();

    let extracted = encode_and_extract(
        128, 80, 4, &chunk_bits, &hhat_seed, &weights,
    );
    assert_eq!(
        extracted, chunk_bits,
        "round-trip mismatch on tiny 4-domain primitive"
    );
}

#[test]
fn four_domain_primitive_roundtrip_default_weights() {
    // Default CostWeights — exercises CS-concentrated allocation
    // (most flips end up in CoeffSign).
    let chunk_bits: Vec<u8> = (0..64).map(|i| (i & 1) as u8).collect();
    let hhat_seed = [0x99u8; 32];
    let weights = CostWeights::default();

    let extracted = encode_and_extract(
        160, 96, 4, &chunk_bits, &hhat_seed, &weights,
    );
    assert_eq!(extracted, chunk_bits, "round-trip mismatch — default weights");
}

#[test]
fn four_domain_primitive_roundtrip_uniform_weights() {
    // Force STC to spread across all 4 domains by using uniform
    // weights (all = 1.0). Verifies the combine/split semantics
    // are correct even when STC genuinely picks multi-domain
    // positions.
    let chunk_bits: Vec<u8> = (0..48).map(|i| ((i * 7 + 3) & 1) as u8).collect();
    let hhat_seed = [0xAAu8; 32];
    let weights = CostWeights {
        coeff_sign: 1.0,
        coeff_suffix: 1.0,
        mvd_sign: 1.0,
        mvd_suffix: 1.0,
    };

    let extracted = encode_and_extract(
        160, 96, 4, &chunk_bits, &hhat_seed, &weights,
    );
    assert_eq!(extracted, chunk_bits, "round-trip mismatch — uniform weights");
}

#[test]
fn four_domain_primitive_rejects_empty_chunk_bits() {
    let yuv = build_yuv_clip(128, 80, 2);
    let weights = CostWeights::default();
    let err = h264_stego_encode_one_gop_with_chunk_bits_4domain(
        &yuv, 128, 80, 2, &[], &[0u8; 32], None, &weights,
    )
    .expect_err("empty chunk_bits should error");
    let msg = format!("{err:?}");
    assert!(msg.contains("chunk_bits") || msg.contains("non-empty"),
        "unexpected error message: {msg}");
}

#[test]
fn four_domain_primitive_rejects_dim_misalignment() {
    let weights = CostWeights::default();
    let yuv = build_yuv_clip(127, 80, 2); // 127 not 16-aligned
    let err = h264_stego_encode_one_gop_with_chunk_bits_4domain(
        &yuv, 127, 80, 2, &[1, 0, 1], &[0u8; 32], None, &weights,
    );
    assert!(err.is_err(), "127-wide should error on alignment check");
}
