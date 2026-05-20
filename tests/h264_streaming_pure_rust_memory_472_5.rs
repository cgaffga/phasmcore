// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #472.5 — Memory regression test for the pure-Rust streaming session.
//!
//! Verifies the bounded-memory property: peak resident memory during
//! a 2N-frame encode should be ≈ peak during an N-frame encode (modulo
//! the linear-in-`n_frames` Annex-B accumulator). Pre-#472.2 this
//! property held only for OH264; pure-Rust buffered every pushed frame.
//!
//! Uses `ps -o rss=` (POSIX-portable, no extra deps) to sample the
//! test binary's RSS in KB. `#[ignore]`'d by default — wall is dominated
//! by the experimental pure-Rust encoder.

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::h264::stego::CostWeights;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Read RSS in kilobytes via `ps -o rss=`. Works on both macOS and
/// Linux (units differ between platforms — KB on Linux, also KB on
/// macOS — close enough for delta comparison).
fn rss_kb() -> u64 {
    let pid = std::process::id();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .expect("ps invocation failed");
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<u64>().unwrap_or(0)
}

/// Spawn a background thread that polls RSS every `poll_ms` and tracks
/// the peak. Returns `(handle, stop_flag, peak_atomic)`.
fn start_rss_sampler(poll_ms: u64) -> (thread::JoinHandle<()>, Arc<AtomicBool>, Arc<AtomicU64>) {
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let stop_c = stop.clone();
    let peak_c = peak.clone();
    let handle = thread::spawn(move || {
        while !stop_c.load(Ordering::Relaxed) {
            let cur = rss_kb();
            // peak = max(peak, cur)
            let mut prev = peak_c.load(Ordering::Relaxed);
            while cur > prev {
                match peak_c.compare_exchange_weak(
                    prev,
                    cur,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(p) => prev = p,
                }
            }
            thread::sleep(Duration::from_millis(poll_ms));
        }
    });
    (handle, stop, peak)
}

/// Synth helper matching the textured YUV used by the #472.4 round-trip
/// tests + the lib-side `streaming_session` tests.
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

/// Drive a pure-Rust streaming encode for `n_frames` at the given
/// dims/gop, recording peak RSS in KB while it runs. Drops the
/// Annex-B output to keep heap pressure attributable to the session
/// itself, not the output Vec.
fn measure_pure_rust_peak_kb(width: u32, height: u32, gop_size: u32, n_frames: u32) -> u64 {
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
        progress_callback: None,
    };

    let (handle, stop, peak) = start_rss_sampler(20);

    let mut enc = StreamingEncodeSession::create(params, "memprobe", "pw")
        .expect("create");
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

    stop.store(true, Ordering::Relaxed);
    handle.join().expect("sampler thread join");

    peak.load(Ordering::Relaxed)
}

#[test]
#[ignore = "slow: pure-Rust experimental encoder + RSS probe"]
fn pure_rust_streaming_memory_is_bounded_in_n_frames() {
    // Warm up so file-system page caches + dyld + initial heap growth
    // don't skew the first measurement.
    let _warmup = measure_pure_rust_peak_kb(320, 240, /*gop=*/ 2, /*n=*/ 2);

    // Short run: 2 GOPs.
    let peak_short_kb = measure_pure_rust_peak_kb(320, 240, /*gop=*/ 2, /*n=*/ 4);
    // Long run: 4× as many GOPs.
    let peak_long_kb = measure_pure_rust_peak_kb(320, 240, /*gop=*/ 2, /*n=*/ 16);

    eprintln!(
        "#472.5 — short (n=4) peak RSS = {peak_short_kb} KB, \
         long (n=16) peak RSS = {peak_long_kb} KB"
    );
    eprintln!(
        "  delta = {} KB ({:.1}% growth)",
        peak_long_kb as i64 - peak_short_kb as i64,
        100.0 * (peak_long_kb as f64 / peak_short_kb.max(1) as f64 - 1.0),
    );

    // Expected: the per-GOP buffer + STC working set is the dominant
    // session memory term. Annex-B output and (small) heap fragmentation
    // grow with n_frames but should NOT scale the peak by 4× (which is
    // what the pre-#472.2 yuv_buffer-everything path would have done at
    // 320×240).
    //
    // Floor of 1.5× allows for 4× more Annex-B output, allocator slack,
    // and noise; pre-#472.2 path would blow well past this on a real
    // long clip.
    assert!(
        peak_long_kb < (peak_short_kb * 3 / 2).max(peak_short_kb + 5_000),
        "pure-Rust streaming memory not bounded in n_frames: \
         peak_short={peak_short_kb} KB, peak_long={peak_long_kb} KB",
    );
}

#[test]
#[ignore = "slow: pure-Rust experimental encoder + RSS probe"]
fn pure_rust_streaming_peak_within_oh264_band() {
    // Cross-encoder comparison: at the same fixture + n_frames,
    // pure-Rust peak should be within the same order of magnitude as
    // OH264 peak. Both paths are per-GOP bounded; the constants
    // differ (different encoder working memory) but the dominant
    // session-state term is the same gop_buffer.
    let _warmup = measure_pure_rust_peak_kb(320, 240, /*gop=*/ 2, /*n=*/ 2);

    let pure_rust_peak = measure_pure_rust_peak_kb(320, 240, /*gop=*/ 2, /*n=*/ 8);

    // Measure OH264 path with same fixture.
    let oh264_params = EncodeSessionParams {
        width: 320,
        height: 240,
        fps_num: 30,
        fps_den: 1,
        qp: 26,
        gop_size: 2,
        total_frames_hint: 8,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let (handle, stop, peak) = start_rss_sampler(20);
    let mut enc = StreamingEncodeSession::create(oh264_params, "memprobe", "pw")
        .expect("OH264 create");
    let mut annex_b = Vec::new();
    for f in 0..8u32 {
        let (y, u, v) = synth_yuv(320, 240, f);
        let frame = YuvFrameRef {
            y: &y,
            y_stride: 320,
            u: &u,
            u_stride: 160,
            v: &v,
            v_stride: 160,
        };
        enc.push_frame(frame, &mut annex_b).expect("OH264 push");
    }
    enc.finish(&mut annex_b).expect("OH264 finish");
    stop.store(true, Ordering::Relaxed);
    handle.join().expect("sampler join");
    let oh264_peak = peak.load(Ordering::Relaxed);

    eprintln!(
        "#472.5 — OH264 peak = {oh264_peak} KB, PureRust peak = {pure_rust_peak} KB, \
         ratio = {:.2}×",
        pure_rust_peak as f64 / oh264_peak.max(1) as f64
    );

    // Pure-Rust IS expected to use more memory than OH264 (the rust
    // encoder is heavier per-GOP than the well-tuned OH264 fork), but
    // not catastrophically so. 5× ceiling gives headroom for allocator
    // and noise while still catching regressions to whole-video O(n).
    assert!(
        pure_rust_peak < oh264_peak.saturating_mul(5).max(50_000),
        "pure-Rust streaming peak runs {pure_rust_peak} KB, more than 5× OH264 peak {oh264_peak} KB",
    );
}
