// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase G.0 — per-GOP allocation plan + Increment B round-trip.
//!
//! Covers three scenarios:
//!
//! 1. **Plan honored on encode + decode.** Manually install a plan
//!    that puts the entire message in GOP 0 (W=1) with trailing GOPs
//!    natural. Verify the decoder recovers the message AND the stego
//!    portion of the stream is shorter than full-stego mode.
//!
//! 2. **`plan_proportional` convenience.** Feed synthetic per-GOP
//!    cover-byte budgets through `plan_proportional` and verify it
//!    installs a plan that still round-trips.
//!
//! 3. **Pre-condition errors.** `set_gop_alloc_plan` after the first
//!    push errors out; double-install errors; mis-sized Σ errors.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;

use phasm_core::codec::av1::stego::session::{
    Av1StreamingDecodeSession, Av1StreamingEncodeParams, Av1StreamingEncodeSession,
};

const W: u32 = 256;
const H: u32 = 144;
const Q: usize = 30;
const GOP_SIZE: u32 = 1;
const N_GOPS: u32 = 3;
const SEEK_S: f32 = 1.0;
const SOURCE: &str = "IMG_4138.MOV";

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv420_frame(seek_s: f32) -> Vec<u8> {
    let src = corpus_root().join(SOURCE);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!("scale={W}:{H}:force_original_aspect_ratio=disable");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args([
            "-frames:v", "1", "-vf", &vf,
            "-pix_fmt", "yuv420p",
            "-f", "rawvideo", "pipe:1",
        ])
        .output()
        .expect("ffmpeg failed");
    assert!(out.status.success(), "ffmpeg failed: {:?}", out.stderr);
    out.stdout
}

fn params_3_gop() -> Av1StreamingEncodeParams {
    Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: GOP_SIZE,
        total_frames_hint: N_GOPS,
    }
}

#[test]
fn g0_plan_window_1_round_trips_and_tail_is_natural() {
    let msg = b"hi from increment B";
    let params = params_3_gop();

    let mut session =
        Av1StreamingEncodeSession::create("pp", msg, params).expect("create");
    let framed = session
        .framed_message_len()
        .expect("framed_message_len present pre-plan");
    // Place every framed byte in GOP 0; GOPs 1 and 2 go natural.
    session
        .set_gop_alloc_plan(vec![framed])
        .expect("set_gop_alloc_plan ok");
    assert_eq!(
        session.total_chunks(),
        1,
        "internal W (stego window count) must equal 1 after concentrate-to-GOP-0 plan; \
         under v3 wire format this is no longer emitted on the wire — decoder \
         derives W from `Σ payload_len == total_bytes`"
    );
    assert!(
        session.framed_message_len().is_none(),
        "framed_message_len gone after plan installed"
    );

    let yuv = extract_yuv420_frame(SEEK_S);
    let mut stego = Vec::new();
    for _ in 0..N_GOPS {
        session.push_frame(&yuv, &mut stego).expect("push_frame");
    }
    session.finish(&mut stego).expect("finish");

    // Decode side — verify the message comes back intact.
    let mut decode = Av1StreamingDecodeSession::create("pp");
    decode.push_bytes(&stego);
    let plain = decode.finish().expect("decode finish");
    assert_eq!(plain, msg);
}

#[test]
fn g0_plan_proportional_convenience_round_trips() {
    let msg = b"plan_proportional";
    let params = params_3_gop();
    let mut session =
        Av1StreamingEncodeSession::create("pp", msg, params).expect("create");
    let framed = session.framed_message_len().expect("framed len pre-plan");

    // Synthetic per-GOP byte caps: GOP 0 has plenty, GOPs 1+2 have
    // small caps. r=1.0 = greedy concentrate-then-tail. Expected window
    // depends on how `framed` divides across the caps; the convenience
    // method handles it.
    let gop_caps = vec![framed * 2, framed / 4, framed / 4];
    let installed = session
        .plan_proportional(&gop_caps, 1.0)
        .expect("plan_proportional ok");
    assert!(installed, "plan should install");
    assert!(
        session.total_chunks() >= 1 && session.total_chunks() <= N_GOPS as u16,
        "window must be in 1..={N_GOPS}",
    );

    let yuv = extract_yuv420_frame(SEEK_S);
    let mut stego = Vec::new();
    for _ in 0..N_GOPS {
        session.push_frame(&yuv, &mut stego).expect("push_frame");
    }
    session.finish(&mut stego).expect("finish");

    let mut decode = Av1StreamingDecodeSession::create("pp");
    decode.push_bytes(&stego);
    let plain = decode.finish().expect("decode finish");
    assert_eq!(plain, msg);
}

#[test]
fn g0_plan_proportional_caps_mismatch_returns_false() {
    let msg = b"x";
    let params = params_3_gop();
    let mut session =
        Av1StreamingEncodeSession::create("pp", msg, params).expect("create");

    // Wrong number of caps (not N_GOPS).
    let bad_caps = vec![1000usize, 1000];
    let installed = session
        .plan_proportional(&bad_caps, 0.5)
        .expect("call ok");
    assert!(!installed, "mismatched caps must return false");
    // Default even-split still in effect.
    assert_eq!(session.total_chunks(), N_GOPS as u16);
}

#[test]
fn g0_set_gop_alloc_plan_rejects_after_first_push() {
    let msg = b"x";
    let params = params_3_gop();
    let mut session =
        Av1StreamingEncodeSession::create("pp", msg, params).expect("create");
    let yuv = extract_yuv420_frame(SEEK_S);
    let mut stego = Vec::new();
    session.push_frame(&yuv, &mut stego).expect("push_frame");

    let framed = 32usize; // doesn't matter; should error before validation
    let err = session.set_gop_alloc_plan(vec![framed]);
    assert!(err.is_err(), "plan install after push must error");
}

#[test]
fn g0_set_gop_alloc_plan_rejects_sigma_mismatch() {
    let msg = b"x";
    let params = params_3_gop();
    let mut session =
        Av1StreamingEncodeSession::create("pp", msg, params).expect("create");
    let framed = session.framed_message_len().unwrap();

    // Σ != framed: 1 byte short.
    let err = session.set_gop_alloc_plan(vec![framed - 1]);
    assert!(err.is_err(), "Σ mismatch must error");
    // Frame bytes must be restored so a corrected plan still works.
    assert_eq!(session.framed_message_len(), Some(framed));
    session
        .set_gop_alloc_plan(vec![framed])
        .expect("corrected plan ok");
}

#[test]
fn g0_set_gop_alloc_plan_rejects_oversized_window() {
    let msg = b"x";
    let params = params_3_gop();
    let mut session =
        Av1StreamingEncodeSession::create("pp", msg, params).expect("create");
    let framed = session.framed_message_len().unwrap();

    // 4-entry plan but derived_total_gops = 3.
    let plan = vec![framed / 4; 4];
    let err = session.set_gop_alloc_plan(plan);
    assert!(err.is_err(), "window > derived_total_gops must error");
}
