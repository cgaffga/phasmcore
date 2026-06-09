// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #530 reproducer — GOP-3 chunk_frame divergence on 1072×1920 OH264
//! streaming. The bug was filed against real iOS/Android recorded
//! video; this synthetic harness exercises the same encoder + walker
//! paths to verify the Phase 4.9 CSL wire-LSB XOR fix closes the
//! mobile-video v1.0 BLOCKER at 1072×1920 IBPBP across 3 GOPs.
//!
//! `#[ignore]`d by default — encodes ~30+ frames of 1072×1920 YUV
//! through the OH264 session (~3-5 s wall).

#![cfg(feature = "h264-encoder")]

use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingDecodeSession,
    StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::h264::stego::CostWeights;

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

fn run_oh264_streaming(
    width: u32,
    height: u32,
    gop_size: u32,
    n_frames: u32,
    msg: &str,
    pass: &str,
) {
    let params = EncodeSessionParams {
        width,
        height,
        fps_num: 30,
        fps_den: 1,
        qp: 26,
        gop_size,
        total_frames_hint: n_frames,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        cost_weights: CostWeights::default(),
        progress_callback: None,
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
        "OH264 round-trip mismatch {width}x{height} × {n_frames}f × GOP={gop_size}: \
         expected {msg:?}, got {:?}",
        result.text
    );
    assert_eq!(result.mode_id, 1, "mode_id should be 1 (Ghost/H.264 stego)");
    eprintln!(
        "OH264 streaming round-trip GREEN: {}x{} × {}f × GOP={} ({} annex_b bytes)",
        width, height, n_frames, gop_size, annex_b.len(),
    );
}

#[test]
#[ignore = "1072×1920 OH264 streaming, ~5s wall"]
fn oh264_streaming_530_repro_1072x1920_3gop() {
    // #530 BLOCKER reproducer: 1072×1920 portrait (carplane shape),
    // 30 frames @ GOP=10 = 3 IDRs/3 GOPs. The Phase 4.9 wire-LSB XOR
    // fix in the fork should make this round-trip green; pre-fix this
    // would diverge at chunk_frame 3 (GOP-3) per #530.
    run_oh264_streaming(
        1072, 1920, /*gop=*/ 10, /*n=*/ 30,
        "phasm v1.0 mobile-video unblock: 3-GOP round-trip works", "pw",
    );
}

#[test]
#[ignore = "Bisect: call encode_yuv_4domain TWICE in sequence under wire_only — outside the streaming session"]
fn oh264_wire_only_two_sequential_calls_bisect() {
    // If this fails the same way as the streaming-multi-GOP test,
    // the bug is in the orchestrator/fork's persistent state across
    // sequential encode_yuv calls — NOT in the streaming session itself.
    use phasm_core::codec::h264::openh264_stego::{
        h264_encode_gop_framed_bits_auto, EncodeOpts,
    };
    use phasm_core::codec::h264::stego::CostWeights;

    unsafe { std::env::set_var("PHASM_USE_WIRE_ONLY", "1") };
    unsafe { std::env::set_var("PHASM_PERF_TRACE", "1") };

    let w: u32 = 320;
    let h: u32 = 240;
    let n: u32 = 2;
    let opts = EncodeOpts { qp: 26, intra_period: 60 };
    let weights = CostWeights::default();
    let hhat_seed: [u8; 32] = [42u8; 32];
    let frame_bits: Vec<u8> = (0..224u8).map(|i| (i & 1)).collect();

    // Build YUV once, encode twice.
    let mut yuv: Vec<u8> = Vec::new();
    for f in 0..n {
        let (y, u, v) = synth_yuv(w, h, f);
        yuv.extend_from_slice(&y);
        yuv.extend_from_slice(&u);
        yuv.extend_from_slice(&v);
    }

    eprintln!("=== Call 1 ===");
    let _b1 = h264_encode_gop_framed_bits_auto(
        &yuv, w, h, n, opts, &frame_bits, &hhat_seed, &weights,
    ).expect("call 1");
    eprintln!("=== Call 2 ===");
    let _b2 = h264_encode_gop_framed_bits_auto(
        &yuv, w, h, n, opts, &frame_bits, &hhat_seed, &weights,
    ).expect("call 2");

    unsafe { std::env::remove_var("PHASM_USE_WIRE_ONLY") };
    unsafe { std::env::remove_var("PHASM_PERF_TRACE") };
}

#[test]
#[ignore = "320×240 OH264 streaming wire_only multi-GOP smallest fail, ~1s wall"]
fn oh264_streaming_530_repro_320x240_3gop_wire_only_tiny() {
    // Smallest multi-GOP wire_only failure repro: 320×240 × 6f at
    // gop=2 → 3 GOPs. Used to isolate the wire_only multi-GOP bug
    // from the 1072×1920 production-resolution noise.
    unsafe { std::env::set_var("PHASM_USE_WIRE_ONLY", "1") };
    run_oh264_streaming(
        320, 240, /*gop=*/ 2, /*n=*/ 6,
        "tiny 3-GOP wire-only", "pw",
    );
    unsafe { std::env::remove_var("PHASM_USE_WIRE_ONLY") };
}

#[test]
#[ignore = "320×240 OH264 streaming wire_only single-GOP control, ~1s wall"]
fn oh264_streaming_530_repro_320x240_1gop_wire_only_tiny() {
    // Single-GOP control: should match the green 4-domain primitive
    // test pattern. If this FAILS, the streaming session itself is
    // breaking wire_only (not multi-GOP specifically).
    unsafe { std::env::set_var("PHASM_USE_WIRE_ONLY", "1") };
    run_oh264_streaming(
        320, 240, /*gop=*/ 6, /*n=*/ 6,
        "tiny single-GOP wire-only", "pw",
    );
    unsafe { std::env::remove_var("PHASM_USE_WIRE_ONLY") };
}

#[test]
#[ignore = "1072×1920 OH264 streaming wire_only single-GOP smoke, ~3s wall"]
fn oh264_streaming_530_repro_1072x1920_1gop_wire_only() {
    // Single-GOP smoke for wire_only: confirms the Phase 4.9 fork
    // CSL XOR fix works end-to-end on 1080p portrait without the
    // multi-GOP streaming-session state complications.
    unsafe { std::env::set_var("PHASM_USE_WIRE_ONLY", "1") };
    run_oh264_streaming(
        1072, 1920, /*gop=*/ 30, /*n=*/ 10,
        "wire-only single-GOP at 1080p", "pw",
    );
    unsafe { std::env::remove_var("PHASM_USE_WIRE_ONLY") };
}

#[test]
#[ignore = "1072×1920 OH264 streaming wire_only multi-GOP, ~5s wall"]
fn oh264_streaming_530_repro_1072x1920_3gop_wire_only() {
    // Same as above but with PHASM_USE_WIRE_ONLY=1 (Pass-2 replay +
    // wire-LSB scratch override path). This is the path the v1.0 ship
    // candidate will run in once the wire_only default flips.
    // SAFETY: set/remove_var are `unsafe` in 2024 edition; this test
    // is single-threaded (gated on --test-threads=1 for StegoSession
    // serialization) so the racy-env warning doesn't apply.
    unsafe { std::env::set_var("PHASM_USE_WIRE_ONLY", "1") };
    run_oh264_streaming(
        1072, 1920, /*gop=*/ 10, /*n=*/ 30,
        "wire-only 3-GOP", "pw",
    );
    unsafe { std::env::remove_var("PHASM_USE_WIRE_ONLY") };
}
