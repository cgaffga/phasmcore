// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #474.6 — empirical calibration of progress-phase weights.
//!
//! Runs an OH264 streaming encode + decode at 1080p × 30 frames ×
//! GOP=30 with progress callbacks that record arrival timestamps,
//! then prints per-phase wall-clock (absolute ms + % of total). The
//! output is consumed by the iOS `PhasmProgressEngine.encodeWeights`
//! / `decodeWeights` tables and the Android equivalents — the
//! mobile smoothing engine needs phase weights that reflect actual
//! wall-clock so the displayed fraction stays faithful (no jump to
//! 100% followed by a long stall, no creep to 30% followed by a
//! sudden completion).
//!
//! Run with:
//!   cargo test -p phasm-core --features h264-encoder,openh264-backend \
//!     --test h264_progress_phase_calibration -- --ignored --nocapture
//!
//! `#[ignore]`d by default — encodes ~6-10 s of wall time. Synthetic
//! YUV; real-content fixtures would be more representative but
//! synthetic is fine for first-pass weights (#474.6's goal is to
//! get the order-of-magnitude right, not to fit a model). Revisit
//! with on-device traces if the user reports new pacing issues.

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use phasm_core::codec::h264::progress::{
    DecodePhase, DecodeProgressCallback, EncodePhase, EncodeProgressCallback,
};
use phasm_core::codec::h264::stego::CostWeights;
use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingDecodeSession,
    StreamingEncodeSession, YuvFrameRef,
};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Phase bucket — collapses per-frame / per-GOP fanouts into one
/// of the 5 progress codes the mobile engine knows about.
fn encode_bucket(p: &EncodePhase) -> u32 {
    match p {
        EncodePhase::Setup => 0,
        EncodePhase::Pass1Capture { .. } => 1,
        EncodePhase::StcPlan { .. } => 2,
        EncodePhase::Pass2Replay { .. } => 3,
        EncodePhase::Mux => 4,
        EncodePhase::Done => 5,
    }
}

fn decode_bucket(p: &DecodePhase) -> u32 {
    match p {
        DecodePhase::Demux => 0,
        DecodePhase::Walker { .. } => 1,
        DecodePhase::StcExtract => 2,
        DecodePhase::ShadowExtract { .. } => 3,
        DecodePhase::Decrypt => 4,
        DecodePhase::Done => 5,
    }
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

#[test]
#[ignore = "encodes ~6-10 s wall; run on demand for weight calibration"]
fn calibrate_progress_phase_weights_oh264_1080p_30f() {
    // OH264 requires 16-aligned dimensions; iOS/Android apps pad 1080p
    // to 1088 before pushing. Match that here.
    const W: u32 = 1920;
    const H: u32 = 1088;
    // 2 GOPs — matches typical 2-second mobile clip @ 30fps and lets
    // the per-GOP events provide some inter-GOP progress granularity.
    const GOP: u32 = 30;
    const N: u32 = 60;
    const MSG: &str = "calibration probe — progress phase weights";
    const PASS: &str = "calibrate";

    // Encode-side recording. Each entry is (bucket, ms_since_start).
    let enc_start = Instant::now();
    let enc_log: Arc<Mutex<Vec<(u32, f64)>>> = Arc::new(Mutex::new(Vec::new()));
    let enc_log_cb = Arc::clone(&enc_log);
    let enc_cb: EncodeProgressCallback = Arc::new(move |phase| {
        let ms = enc_start.elapsed().as_secs_f64() * 1000.0;
        enc_log_cb.lock().unwrap().push((encode_bucket(&phase), ms));
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
        progress_callback: Some(enc_cb),
    };
    let mut enc = StreamingEncodeSession::create(params, MSG, PASS).expect("encode create");
    let mut annex_b = Vec::new();
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
        enc.push_frame(frame, &mut annex_b).expect("push frame");
    }
    enc.finish(&mut annex_b).expect("finish encode");
    let enc_total_ms = enc_start.elapsed().as_secs_f64() * 1000.0;

    // Decode-side recording.
    let dec_start = Instant::now();
    let dec_log: Arc<Mutex<Vec<(u32, f64)>>> = Arc::new(Mutex::new(Vec::new()));
    let dec_log_cb = Arc::clone(&dec_log);
    let dec_cb: DecodeProgressCallback = Arc::new(move |phase| {
        let ms = dec_start.elapsed().as_secs_f64() * 1000.0;
        dec_log_cb.lock().unwrap().push((decode_bucket(&phase), ms));
    });
    let mut dec = StreamingDecodeSession::create(PASS)
        .expect("decode create")
        .with_progress_callback(dec_cb);
    dec.push_annex_b(&annex_b).expect("push annex-b");
    let result = dec.finish().expect("finish decode");
    let dec_total_ms = dec_start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(result.text, MSG, "round-trip text mismatch");

    print_phase_durations("ENCODE", &enc_log.lock().unwrap(), enc_total_ms, 5);
    print_phase_durations("DECODE", &dec_log.lock().unwrap(), dec_total_ms, 5);
}

/// Compute per-phase deltas: each event's timestamp is the START of
/// the NEXT phase, so phase `i`'s duration = `events[i+1].ms -
/// events[i].ms`. The final `Done` event closes the last bucket.
/// Buckets `0..done_code` are summed and reported as % of the
/// captured timeline (which is `done_code`'s timestamp).
fn print_phase_durations(label: &str, log: &[(u32, f64)], wall_ms: f64, done_code: u32) {
    if log.is_empty() {
        eprintln!("[{}] no progress events", label);
        return;
    }

    // Events fire AT THE END of their named work. So the duration of
    // event[i]'s phase = event[i].ts - event[i-1].ts (or just
    // event[0].ts for the first event, measured from t=0). Bucket
    // by event[i].code, NOT event[i-1].code.
    let mut bucket_ms = vec![0.0f64; (done_code + 1) as usize];
    let mut prev = 0.0f64;
    for &(code, t) in log.iter() {
        let dur = (t - prev).max(0.0);
        if (code as usize) < bucket_ms.len() {
            bucket_ms[code as usize] += dur;
        }
        prev = t;
    }
    // Anchor on Done timestamp if present, else wall.
    let captured_ms = log
        .iter()
        .rev()
        .find(|(c, _)| *c == done_code)
        .map(|(_, t)| *t)
        .unwrap_or(wall_ms);

    eprintln!();
    eprintln!(
        "═══════ {} CALIBRATION  ({} events, captured = {:.0} ms, wall = {:.0} ms) ═══════",
        label,
        log.len(),
        captured_ms,
        wall_ms,
    );
    let names_enc = ["Setup", "Pass1Capture", "StcPlan", "Pass2Replay", "Mux"];
    let names_dec = ["Demux", "Walker", "StcExtract", "ShadowExtract", "Decrypt"];
    let names: &[&str] = if label == "ENCODE" { &names_enc } else { &names_dec };

    let total: f64 = bucket_ms.iter().take(done_code as usize).sum();
    for (i, ms) in bucket_ms.iter().take(done_code as usize).enumerate() {
        let pct = if total > 0.0 { ms / total * 100.0 } else { 0.0 };
        eprintln!(
            "  [{}] {:<14}  {:>9.1} ms   {:>5.1}%",
            i, names[i], ms, pct
        );
    }
    // Weight-table-friendly summary: print as a comma-separated list of fractions.
    eprintln!();
    let weights: Vec<String> = bucket_ms
        .iter()
        .take(done_code as usize)
        .map(|ms| format!("{:.3}", if total > 0.0 { ms / total } else { 0.0 }))
        .collect();
    eprintln!("  weights: [{}]", weights.join(", "));
    eprintln!();
}
