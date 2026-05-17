// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §long-form-stego Phase 4 — H264GopReplayCover bit-exact equivalence.
//
// Feed the same STC inputs (message + h-hat + h + w) through
// streaming-Viterbi twice — once with InMemoryCoverFetch wrapping
// the full-clip Pass 1 cover, once with H264GopReplayCover that
// replays Pass 1 per segment. Stego output bytes must match.

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::stego::cover_replay::H264GopReplayCover;
use phasm_core::codec::h264::stego::encoder_hook::PositionLoggerHook;
use phasm_core::codec::h264::stego::gop_pattern::{
    iter_encode_order, FrameType, GopPattern,
};
use phasm_core::codec::h264::stego::hook::EmbedDomain;
use phasm_core::codec::h264::stego::orchestrate::GopCover;
use phasm_core::stego::stc::hhat::generate_hhat;
use phasm_core::stego::stc::streaming_segmented::{
    stc_embed_streaming_segmented, InMemoryCoverFetch,
};

fn deterministic_yuv(w: u32, h: u32, n_frames: usize) -> Vec<u8> {
    let frame_size = (w * h * 3 / 2) as usize;
    let mut out = Vec::with_capacity(frame_size * n_frames);
    let mut s: u32 = 0x1234_5678;
    for _ in 0..n_frames {
        for _ in 0..frame_size {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            out.push((s >> 16) as u8);
        }
    }
    out
}

fn full_pass1(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    pattern: GopPattern,
) -> GopCover {
    use phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig;
    let frame_size = (width * height * 3 / 2) as usize;
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder");
    enc.entropy_mode = EntropyMode::Cabac;
    // #513 fix 2026-05-17: mirror production `build_encoder()` in
    // core/src/codec/h264/stego/encode_pixels.rs:299, which
    // H264GopReplayCover uses internally for its per-segment Pass 1
    // replays. Without these two knobs the test's full-clip Pass 1
    // counts diverge from the replay path's per-GOP counts (clean
    // n=24016 vs replay total 23878 on the 64x48x6 ipppp fixture),
    // and the m*w sanity check in H264GopReplayCover::new bails.
    // Same root cause as #511.
    enc.enable_transform_8x8 = true;
    enc.b_rdo_config = BRdoConfig::PRODUCTION_VISUAL;
    enc.enable_mvd_stego_hook = true;
    enc.enable_b_frames = pattern.has_b_frames();
    enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
    for meta in iter_encode_order(n_frames, pattern) {
        let d = meta.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        match meta.frame_type {
            FrameType::Idr => { enc.encode_i_frame(frame).unwrap(); }
            FrameType::P => { enc.encode_p_frame(frame).unwrap(); }
            FrameType::B => { enc.encode_b_frame(frame).unwrap(); }
        }
    }
    let mut hook = enc.take_stego_hook().expect("hook present");
    hook.take_cover_if_logger().expect("PositionLoggerHook")
}

#[test]
fn gop_replay_cover_matches_in_memory_streaming_ipppp() {
    let width = 64u32;
    let height = 48u32;
    let n_frames = 9usize;
    let gop_size = 3usize;
    let b_count = 0usize;
    let yuv = deterministic_yuv(width, height, n_frames);

    let pattern = GopPattern::Ipppp { gop: gop_size };
    let cover = full_pass1(&yuv, width, height, n_frames, pattern);

    // Pick the coeff_sign_bypass domain (always non-empty on real content).
    let domain = EmbedDomain::CoeffSignBypass;
    let dom_bits = &cover.cover.coeff_sign_bypass.bits;
    let dom_costs = &cover.costs.coeff_sign_bypass;
    let n = dom_bits.len();
    assert!(n >= 8, "test fixture must have ≥8 cover positions");

    // STC params: w=2 gives STC slack vs the cover (m = n/2) so
    // the embedding always finds a finite-cost path even on
    // adversarial real-Pass-1 costs.
    let h = 4usize;
    let w = 2usize;
    let m = n / w;
    let segment_size_in_blocks = ((m as f64).sqrt().ceil() as usize).max(2);
    let mut seed = [0u8; 32];
    seed[..16].copy_from_slice(b"phase4-equivtest");
    let hhat = generate_hhat(h, w, &seed);
    let message: Vec<u8> = (0..m).map(|i| (i & 1) as u8).collect();

    // Path A: InMemoryCoverFetch over the full Pass 1 cover.
    let mut in_mem = InMemoryCoverFetch::new(
        dom_bits, dom_costs, m, w, segment_size_in_blocks,
    )
    .expect("InMemoryCoverFetch::new");
    let mem_result = stc_embed_streaming_segmented(
        &mut in_mem, &message, &hhat, h, w,
    )
    .expect("InMemoryCoverFetch streaming embed");

    // Path B: H264GopReplayCover (replays Pass 1 per fetch_segment).
    let mut replay = H264GopReplayCover::new(
        &yuv, width, height, n_frames, gop_size, b_count, Some(26),
        domain, m, w, segment_size_in_blocks,
    )
    .expect("H264GopReplayCover::new");
    let replay_result = stc_embed_streaming_segmented(
        &mut replay, &message, &hhat, h, w,
    )
    .expect("H264GopReplayCover streaming embed");

    assert_eq!(
        mem_result.stego_bits, replay_result.stego_bits,
        "stego_bits diverge between InMemoryCoverFetch and H264GopReplayCover",
    );
    assert!(
        (mem_result.total_cost - replay_result.total_cost).abs() < 1e-6,
        "total_cost diverges: in-mem={} replay={}",
        mem_result.total_cost, replay_result.total_cost,
    );
    assert_eq!(
        mem_result.num_modifications, replay_result.num_modifications,
        "num_modifications diverges",
    );
}

#[test]
fn gop_replay_cover_total_positions_matches_full_pass1_count() {
    let width = 64u32;
    let height = 48u32;
    let n_frames = 6usize;
    let gop_size = 3usize;
    let b_count = 0usize;
    let yuv = deterministic_yuv(width, height, n_frames);

    let pattern = GopPattern::Ipppp { gop: gop_size };
    let cover = full_pass1(&yuv, width, height, n_frames, pattern);
    let expected = cover.cover.coeff_sign_bypass.bits.len();
    let n = expected;
    let m = n;
    let w = 1usize;

    let replay = H264GopReplayCover::new(
        &yuv, width, height, n_frames, gop_size, b_count, Some(26),
        EmbedDomain::CoeffSignBypass, m, w, 4,
    )
    .expect("H264GopReplayCover::new");
    use phasm_core::stego::stc::streaming_segmented::CoverFetch;
    assert_eq!(
        replay.total_positions(), expected,
        "total_positions {} doesn't match full Pass 1 count {}",
        replay.total_positions(), expected,
    );
}
