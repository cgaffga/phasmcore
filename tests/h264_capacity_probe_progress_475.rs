// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #475 — progressive capacity-probe estimate via per-GOP callback.
//!
//! Verifies that `StreamingProbeSession::set_progress_callback` fires
//! once per drained GOP (push_frame flushes + final partial flush in
//! finish()), with monotonically non-decreasing `gops_done` and
//! cover_bits arguments, and that the final emit matches the value
//! returned from `finish()`.

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use phasm_core::codec::h264::progress::CapacityProbeCallback;
use phasm_core::codec::h264::stego::CostWeights;
use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingProbeSession, YuvFrameRef,
};
use std::sync::{Arc, Mutex};

fn synth_yuv(width: u32, height: u32, frame_idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = Vec::with_capacity((width * height) as usize);
    for j in 0..height {
        for i in 0..width {
            y.push(((i + frame_idx * 2) ^ (j + frame_idx * 3)) as u8);
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
            v.push(((s >> 16) as u8).wrapping_add(pos));
        }
    }
    (y, u, v)
}

/// Runs both probe scenarios (full GOPs only + final partial GOP)
/// in one test to avoid parallel OH264 session collisions — the
/// session-test mutex is `pub(crate)` so external integration tests
/// can't borrow it.
#[test]
fn capacity_probe_progress_callback_scenarios() {
    capacity_probe_fires_progress_per_gop();
    capacity_probe_partial_final_gop_emits_event();
}

fn capacity_probe_fires_progress_per_gop() {
    const W: u32 = 320;
    const H: u32 = 192;
    const GOP: u32 = 4;
    const N: u32 = 12; // 3 full GOPs

    let log: Arc<Mutex<Vec<(u32, u32, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let log_cb = Arc::clone(&log);
    let cb: CapacityProbeCallback = Arc::new(move |done, total, bits| {
        log_cb.lock().unwrap().push((done, total, bits));
    });

    let params = EncodeSessionParams {
        width: W,
        height: H,
        fps_num: 30,
        fps_den: 1,
        qp: 26,
        gop_size: GOP,
        total_frames_hint: N,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut probe = StreamingProbeSession::create(params)
        .expect("create probe")
        .with_progress_callback(cb);

    for f in 0..N {
        let (y, u, v) = synth_yuv(W, H, f);
        let frame = YuvFrameRef {
            y: &y,
            y_stride: W as usize,
            u: &u,
            u_stride: (W / 2) as usize,
            v: &v,
            v_stride: (W / 2) as usize,
        };
        probe.push_frame(frame).expect("push frame");
    }

    let result = probe.finish().expect("finish probe");
    let events = log.lock().unwrap().clone();

    // Expect at least one event per drained GOP. With N=12, GOP=4
    // we get 3 full GOP drains in push_frame; finish() only emits
    // again if frames_buffered > 0 (zero here).
    assert_eq!(events.len(), 3, "expected 3 progress events, got {:?}", events);

    // gops_done monotonic non-decreasing, total_gops constant at 3,
    // cover_bits monotonic non-decreasing, last event matches final
    // result.
    let mut prev_done = 0u32;
    let mut prev_bits = 0usize;
    for &(done, total, bits) in &events {
        assert!(done >= prev_done, "gops_done regressed: {events:?}");
        assert!(bits >= prev_bits, "cover_bits regressed: {events:?}");
        assert_eq!(total, 3, "total_gops should be ceil(12/4)=3, got {total}");
        prev_done = done;
        prev_bits = bits;
    }
    let last = events.last().expect("at least one event");
    assert_eq!(last.0, result.n_gops, "last event gops_done != result");
    assert_eq!(last.2, result.cover_bits, "last event cover_bits != result");
    assert_eq!(result.n_gops, 3);
}

fn capacity_probe_partial_final_gop_emits_event() {
    // N=10 with GOP=4 → 2 full GOPs drained in push_frame + 1
    // partial GOP drained in finish(). Verify finish() emits its
    // own event so the listener always sees the final number.
    const W: u32 = 320;
    const H: u32 = 192;
    const GOP: u32 = 4;
    const N: u32 = 10;

    let log: Arc<Mutex<Vec<(u32, u32, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let log_cb = Arc::clone(&log);
    let cb: CapacityProbeCallback = Arc::new(move |done, total, bits| {
        log_cb.lock().unwrap().push((done, total, bits));
    });

    let params = EncodeSessionParams {
        width: W, height: H, fps_num: 30, fps_den: 1, qp: 26,
        gop_size: GOP, total_frames_hint: N,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut probe = StreamingProbeSession::create(params)
        .expect("create probe")
        .with_progress_callback(cb);

    for f in 0..N {
        let (y, u, v) = synth_yuv(W, H, f);
        let frame = YuvFrameRef {
            y: &y, y_stride: W as usize,
            u: &u, u_stride: (W / 2) as usize,
            v: &v, v_stride: (W / 2) as usize,
        };
        probe.push_frame(frame).expect("push");
    }
    let result = probe.finish().expect("finish");
    let events = log.lock().unwrap().clone();

    // 2 full GOPs from push_frame + 1 partial-GOP emit from finish()
    assert_eq!(events.len(), 3, "events: {:?}", events);
    assert_eq!(events.last().unwrap().0, result.n_gops);
    assert_eq!(events.last().unwrap().2, result.cover_bits);
    assert_eq!(result.n_gops, 3);
}
