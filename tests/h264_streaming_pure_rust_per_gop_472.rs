// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #472.4 — Multi-matrix round-trip for the pure-Rust streaming
//! session (`SessionImpl::PureRust`).
//!
//! Covers `total_frames` × `gop_size` × message-length combinations
//! to verify the per-GOP encode primitive ships chunk_frame data
//! that `StreamingDecodeSession` reassembles correctly.
//!
//! `#[ignore]`'d by default (each matrix encodes the pure-Rust
//! experimental H.264 encoder, which is much slower than OH264 —
//! seconds-to-minutes per matrix).

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingDecodeSession,
    StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::h264::stego::CostWeights;

/// Noisy textured synth matching the lib-test helper in
/// `streaming_session.rs::tests::synth_yuv_frame`. Smooth gradients
/// produce too few CoeffSign cover bits at QP=26; the textured
/// XOR + LCG noise pattern gives a realistic cover budget per GOP.
fn synth_yuv(width: u32, height: u32, frame_idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = Vec::with_capacity((width * height) as usize);
    for j in 0..height {
        for i in 0..width {
            let val = ((i + frame_idx * 2) ^ (j + frame_idx * 3)) as u8;
            y.push(val);
        }
    }
    let half_w = width / 2;
    let half_h = height / 2;
    let mut u = Vec::with_capacity((half_w * half_h) as usize);
    let mut v = Vec::with_capacity((half_w * half_h) as usize);
    let mut s: u32 = 0xCAFE_F00D ^ frame_idx;
    for j in 0..half_h {
        for i in 0..half_w {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let tex = (s >> 16) as u8;
            let pos = (i + j + frame_idx) as u8;
            u.push(tex.wrapping_add(pos));
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let tex2 = (s >> 16) as u8;
            v.push(tex2.wrapping_add(pos));
        }
    }
    (y, u, v)
}

fn run_roundtrip(width: u32, height: u32, gop_size: u32, n_frames: u32, msg: &str, pass: &str) {
    let params = EncodeSessionParams {
        width,
        height,
        fps_num: 30,
        fps_den: 1,
        qp: 26,
        gop_size,
        total_frames_hint: n_frames,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::PureRust,
        cost_weights: CostWeights::default(),
    };
    let mut enc =
        StreamingEncodeSession::create(params, msg, pass).expect("encode session create");
    let mut annex_b = Vec::new();
    for f in 0..n_frames {
        let (y, u, v) = synth_yuv(width, height, f);
        let frame = YuvFrameRef {
            y: &y,
            y_stride: width as usize,
            u: &u,
            u_stride: (width / 2) as usize,
            v: &v,
            v_stride: (width / 2) as usize,
        };
        enc.push_frame(frame, &mut annex_b).expect("push frame");
    }
    enc.finish(&mut annex_b).expect("finish encode");
    assert!(!annex_b.is_empty(), "empty Annex-B output");

    let mut dec = StreamingDecodeSession::create(pass).expect("decode session create");
    dec.push_annex_b(&annex_b).expect("push annex-b");
    let result = dec.finish().expect("finish decode");
    assert_eq!(
        result.text, msg,
        "round-trip mismatch for {width}x{height} × {n_frames}f × GOP={gop_size}: \
         expected {msg:?}, got {:?}",
        result.text
    );
    assert_eq!(result.mode_id, 1, "mode_id should be 1 (Ghost/H.264 stego)");
}

#[test]
#[ignore = "slow: pure-Rust experimental encoder"]
fn pure_rust_streaming_roundtrip_1gop_short_msg() {
    // Single GOP. Smallest valid configuration. total_chunks=1.
    run_roundtrip(320, 240, /*gop=*/ 2, /*n=*/ 2, "hi", "pw");
}

#[test]
#[ignore = "slow: pure-Rust experimental encoder"]
fn pure_rust_streaming_roundtrip_2gop_short_msg() {
    // Two GOPs, exact fill. total_chunks=2.
    run_roundtrip(320, 240, /*gop=*/ 2, /*n=*/ 4, "hi from rust", "pw");
}

#[test]
#[ignore = "slow: pure-Rust experimental encoder"]
fn pure_rust_streaming_roundtrip_partial_tail_gop() {
    // GOP=3, n=4 → first GOP full (3f), second GOP partial (1f).
    // Verifies pure_rust_finish drains the partial-tail GOP.
    run_roundtrip(320, 240, /*gop=*/ 3, /*n=*/ 4, "tail", "pw");
}

#[test]
#[ignore = "slow: pure-Rust experimental encoder"]
fn pure_rust_streaming_roundtrip_3gop() {
    // 3 GOPs of 2 frames each. Exercises cross-GOP chunk indexing
    // (total_chunks=3, chunk_idx ∈ {0,1,2}).
    run_roundtrip(320, 240, /*gop=*/ 2, /*n=*/ 6, "three gops", "pw");
}

#[test]
#[ignore = "slow: pure-Rust experimental encoder"]
fn pure_rust_streaming_roundtrip_longer_message() {
    // Multi-GOP with a payload that's likely to exceed any single
    // chunk's STC capacity — forces split_message_into_chunks to
    // distribute bytes across GOPs.
    let msg = "phasm pure-Rust streaming round-trip: \
               this is a deliberately verbose payload so the per-GOP \
               chunker has to spread bytes across multiple GOPs and \
               the decode session has to reassemble in order.";
    run_roundtrip(320, 240, /*gop=*/ 2, /*n=*/ 6, msg, "pw");
}

#[test]
#[ignore = "slow: pure-Rust experimental encoder"]
fn pure_rust_streaming_roundtrip_larger_frame() {
    // 480×272 × 4f × GOP=2 → 2 GOPs. Same fixture used by
    // Phase 1.1.D probe; confirms the per-GOP encode primitive
    // scales beyond the smallest valid frame.
    run_roundtrip(480, 272, /*gop=*/ 2, /*n=*/ 4, "larger frame", "pw");
}
