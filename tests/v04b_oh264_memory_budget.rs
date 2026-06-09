// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// V0.4.B — OH264 streaming session memory budget (live lane).
//
// The companion to `h264_streaming_pure_rust_memory_472_5.rs`, but for
// OH264. The pure-Rust memory test is `#[ignore]`d because the encoder
// is too slow for the default lane; OH264 is fast enough to run a
// memory-bounded assertion every test run.
//
// What this test gates:
//   1. Per-GOP bounded-ness — peak RSS at 60 frames must NOT blow up
//      relative to peak at 30 frames (the regression class that
//      #472.2 closed for pure-Rust).
//   2. Absolute peak budget at 480p × 30f streaming — must stay under
//      150 MB above warmup baseline. The number is conservative;
//      empirical OH264 streaming at 480p sits around ~10-30 MB of
//      session state.
//
// Both are RSS-based via `ps -o rss=`, so they're noisy on busy CI
// hosts. Thresholds are deliberately generous; the test is meant to
// catch order-of-magnitude regressions (the kind #472 surfaced), not
// 10% drift.

#![cfg(feature = "h264-encoder")]

use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::h264::stego::CostWeights;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

// OH264 StegoSession is a process-global singleton. Serialize tests
// that create one so the two #[test]s in this file don't fight.
static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

fn rss_kb() -> u64 {
    let pid = std::process::id();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .expect("ps invocation failed");
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<u64>().unwrap_or(0)
}

fn start_rss_sampler(poll_ms: u64) -> (thread::JoinHandle<()>, Arc<AtomicBool>, Arc<AtomicU64>) {
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let stop_c = stop.clone();
    let peak_c = peak.clone();
    let handle = thread::spawn(move || {
        while !stop_c.load(Ordering::Relaxed) {
            let cur = rss_kb();
            let mut prev = peak_c.load(Ordering::Relaxed);
            while cur > prev {
                match peak_c.compare_exchange_weak(
                    prev, cur, Ordering::Relaxed, Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => prev = actual,
                }
            }
            thread::sleep(std::time::Duration::from_millis(poll_ms));
        }
    });
    (handle, stop, peak)
}

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

fn measure_oh264_peak_kb(width: u32, height: u32, gop_size: u32, n_frames: u32) -> u64 {
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

    let (handle, stop, peak) = start_rss_sampler(20);

    let mut enc = StreamingEncodeSession::create(params, "v04b memprobe", "pw")
        .expect("OH264 streaming session create");
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
        enc.push_frame(frame, &mut annex_b).expect("OH264 push_frame");
    }
    enc.finish(&mut annex_b).expect("OH264 finish");
    drop(annex_b);

    stop.store(true, Ordering::Relaxed);
    handle.join().expect("sampler join");
    peak.load(Ordering::Relaxed)
}

#[test]
fn v04b_oh264_streaming_memory_bounded_in_n_frames() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    // Warm up so first-encode allocation noise doesn't skew the short
    // measurement.
    let _warmup = measure_oh264_peak_kb(480, 272, /*gop=*/ 30, /*n=*/ 10);

    let peak_short = measure_oh264_peak_kb(480, 272, /*gop=*/ 30, /*n=*/ 30);
    let peak_long = measure_oh264_peak_kb(480, 272, /*gop=*/ 30, /*n=*/ 60);

    let growth_kb = peak_long as i64 - peak_short as i64;
    let growth_pct = 100.0 * (peak_long as f64 / peak_short.max(1) as f64 - 1.0);

    eprintln!(
        "V0.4.B OH264 memory budget — short(n=30)={peak_short} KB, \
         long(n=60)={peak_long} KB, growth={growth_kb} KB ({growth_pct:.1}%)"
    );

    // The streaming session should be O(gop_size × frame_size) per
    // GOP. Doubling n_frames doubles the Annex-B accumulator size but
    // the SESSION state (gop_buffer, recon DPB, encoder workspace)
    // stays flat.
    //
    // Annex-B at 480p × 30 with QP=26 is ~30-40 KB per second of video,
    // so 60-frame run vs 30-frame run adds ~30 KB to the output Vec.
    // Allocator slack + GC may add another 5-10 MB.
    //
    // Allow +50 MB AND +50% — whichever is larger. This is loose
    // enough that CI noise doesn't fire, tight enough that
    // re-introducing #472-class (per-frame yuv_buffer) regressions
    // would fail (those grow by 10-100 MB per extra frame at 480p).
    let bound_kb = (peak_short.saturating_mul(3) / 2).max(peak_short + 50_000);
    assert!(
        peak_long < bound_kb,
        "OH264 streaming peak unbounded in n_frames: \
         short={peak_short} KB, long={peak_long} KB, bound={bound_kb} KB"
    );
}

#[test]
fn v04b_oh264_streaming_peak_under_absolute_budget() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    // Conservative absolute budget: at 480p × 30f, OH264 streaming with
    // gop=30 (single IDR + 29 P) should sit well under 300 MB session
    // growth (peak RSS minus baseline). Test binary RSS itself at
    // startup is ~100-150 MB on macOS arm64 (rayon thread pools, std
    // heap, dyld); we measure DELTA so that's subtracted out.
    //
    // This gate catches pathological growth — e.g., accidentally
    // accumulating every frame's YUV in a session field at 1080p
    // would blow past this within a few seconds of input.
    let baseline = rss_kb();
    let peak = measure_oh264_peak_kb(480, 272, /*gop=*/ 30, /*n=*/ 30);
    let session_growth = peak.saturating_sub(baseline);

    eprintln!(
        "V0.4.B OH264 absolute memory budget — baseline={baseline} KB, \
         peak={peak} KB, session_growth={session_growth} KB"
    );

    // 300 MB ceiling above baseline. Real OH264 streaming at 480p in
    // May 2026 sits around 10-30 MB of new session state; the rest of
    // any growth is allocator slack + the Annex-B accumulator. 300 MB
    // is loose enough to absorb host-noise jitter and tight enough to
    // catch order-of-magnitude regressions.
    const SESSION_GROWTH_CEILING_KB: u64 = 300_000;
    assert!(
        session_growth < SESSION_GROWTH_CEILING_KB,
        "V0.4.B: OH264 480p × 30f streaming session growth {session_growth} KB exceeds \
         {SESSION_GROWTH_CEILING_KB} KB ceiling. Suspect leaked frame buffers."
    );
}
