// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! G.2.P4 — `Av1StreamingEncodeSession::plan_safe_balanced` integration.
//!
//! Exercises the caller-loop pattern end-to-end:
//!
//! 1. Create session
//! 2. Loop: planner returns `NeedMoreSamples` → probe → append samples → retry
//! 3. Planner returns `Installed` → push frames → finish
//! 4. Decode round-trips

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;

use phasm_core::codec::av1::stego::session::{
    Av1PlanOutcome, Av1StreamingDecodeSession, Av1StreamingEncodeParams,
    Av1StreamingEncodeSession,
};
use phasm_core::stego::calibration::AllocationCalibration;

const W: u32 = 256;
const H: u32 = 144;
const Q: usize = 30;
const N_GOPS: usize = 3;
const SEEK_S: f32 = 1.0;

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv420_frame(seek_s: f32) -> Vec<u8> {
    let src = corpus_root().join("IMG_4138.MOV");
    assert!(src.exists(), "corpus fixture missing");
    let vf = format!("scale={W}:{H}:force_original_aspect_ratio=disable");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args([
            "-frames:v",
            "1",
            "-vf",
            &vf,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .expect("ffmpeg");
    assert!(out.status.success(), "ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr));
    out.stdout
}

fn params_3_gop() -> Av1StreamingEncodeParams {
    Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: N_GOPS as u32,
    }
}

/// Synthetic caps for stratified samples. Returns enough capacity
/// per GOP that even a sizeable message can fit at 5% utilization.
fn synth_sample_caps(positions: &[usize]) -> Vec<(usize, usize)> {
    // 80 KB per GOP — well above what 256×144 actually has, but the
    // planner doesn't care where the numbers come from; it just
    // does math on them. Real callers feed in real probe measurements.
    positions.iter().map(|&i| (i, 80_000usize)).collect()
}

#[test]
fn plan_safe_balanced_caller_loop_round_trip() {
    let msg = b"phasm AV1 G.2.P4 plan_safe_balanced round-trip";
    let pass = "g2p4-pass";
    let params = params_3_gop();
    let mut session = Av1StreamingEncodeSession::create(pass, msg, params).expect("create");

    let cal = AllocationCalibration::AV1_1080P_QP30;
    let mut samples: Vec<(usize, usize)> = Vec::new();
    let mut iterations = 0;
    let installed_window: u32;

    loop {
        iterations += 1;
        assert!(iterations < 20, "planner did not converge");
        let outcome = session
            .plan_safe_balanced(&samples, params.gop_size as usize, &cal)
            .expect("planner");
        match outcome {
            Av1PlanOutcome::Installed { window } => {
                installed_window = window;
                break;
            }
            Av1PlanOutcome::NeedMoreSamples { positions, .. } => {
                // K_min on N_GOPS=3 clamps to 3. So we'll get all 3 GOPs
                // requested up front; full coverage in one shot.
                let new_samples = synth_sample_caps(&positions);
                samples.extend(new_samples);
            }
            Av1PlanOutcome::MessageTooLarge => {
                panic!("MessageTooLarge unexpected for trivial payload");
            }
        }
    }

    // For N_GOPS=3 and W_floor_abs=8 → W = min(8, 3) = 3.
    assert!(installed_window >= 1 && installed_window <= N_GOPS as u32);

    // Encode loop.
    let yuv = extract_yuv420_frame(SEEK_S);
    let mut stego = Vec::new();
    for _ in 0..N_GOPS {
        session.push_frame(&yuv, &mut stego).expect("push_frame");
    }
    session.finish(&mut stego).expect("finish");

    // Decode side.
    let mut dec = Av1StreamingDecodeSession::create(pass);
    dec.push_bytes(&stego);
    let plain = dec.finish().expect("decode");
    assert_eq!(plain, msg);
}

#[test]
fn plan_safe_balanced_message_too_large_at_full_coverage() {
    // Tiny synth caps (100 bytes each), massive message — even with
    // all 3 GOPs probed there's no way to fit. Must surface
    // MessageTooLarge.
    let msg = vec![0u8; 1_000_000];
    let pass = "g2p4-too-large";
    let params = params_3_gop();
    let mut session =
        Av1StreamingEncodeSession::create(pass, &msg, params).expect("create");
    let cal = AllocationCalibration::AV1_1080P_QP30;

    // Skip the K_min loop, jump straight to "full coverage" by supplying
    // all 3 GOPs with tiny caps.
    let samples: Vec<(usize, usize)> = (0..N_GOPS).map(|i| (i, 100usize)).collect();
    let outcome = session
        .plan_safe_balanced(&samples, params.gop_size as usize, &cal)
        .expect("planner");
    assert_eq!(outcome, Av1PlanOutcome::MessageTooLarge);
}

#[test]
fn plan_safe_balanced_empty_samples_requests_kmin() {
    let msg = b"x";
    let pass = "g2p4-empty";
    let params = params_3_gop();
    let mut session = Av1StreamingEncodeSession::create(pass, msg, params).expect("create");
    let cal = AllocationCalibration::AV1_1080P_QP30;

    let outcome = session
        .plan_safe_balanced(&[], params.gop_size as usize, &cal)
        .expect("planner");
    match outcome {
        Av1PlanOutcome::NeedMoreSamples { positions, total_target } => {
            // K_min=8 clamps to N_GOPS=3.
            assert_eq!(total_target, 3);
            assert!(positions.len() <= 3);
            assert!(positions.iter().all(|&p| p < N_GOPS));
        }
        other => panic!("expected NeedMoreSamples on first call, got {other:?}"),
    }
}

#[test]
fn plan_safe_balanced_rejects_after_push_frame() {
    let msg = b"x";
    let pass = "g2p4-after-push";
    let params = params_3_gop();
    let mut session = Av1StreamingEncodeSession::create(pass, msg, params).expect("create");
    let cal = AllocationCalibration::AV1_1080P_QP30;

    let yuv = extract_yuv420_frame(SEEK_S);
    let mut stego = Vec::new();
    session.push_frame(&yuv, &mut stego).expect("push_frame");

    // Now planner should reject — chunk_idx > 0 means we're past planning.
    let err = session.plan_safe_balanced(&[], params.gop_size as usize, &cal);
    assert!(err.is_err(), "planner must reject after first push_frame");
}
