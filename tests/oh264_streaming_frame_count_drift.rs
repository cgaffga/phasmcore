// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #850 regression — H.264 streaming encode must tolerate a frame-count
//! hint that differs from the real decoded count.
//!
//! The mobile bridges derive `total_frames_hint` from `duration × fps`
//! (iOS `round(dur×fps)`, Android `(durationUs×fps)/1e6`) and stream
//! frames one at a time, so the true count is only known once the
//! decoder hits EOS — and it routinely differs by a frame or two (a clip
//! with a leading empty edit, variable frame durations, …). Before #850
//! the encode session hard-failed on ANY mismatch:
//!   * MORE frames than the hint  → `oh264_push_frame` "pushed more
//!     frames than total_frames_hint expected" → surfaces on iOS/Android
//!     as the generic "This video format isn't supported." (the
//!     Artlist_CarPlane.mp4 report).
//!   * FEWER frames than the hint → `oh264_finish` "total_frames_hint was
//!     too high".
//!
//! The DECODER was already tolerant: `StreamingDecodeSession::finish`
//! reads GOP 0's declared `total_bytes` and accumulates over however many
//! GOPs the real stream contains, stopping early and ignoring a plain
//! tail. So the asserts were over-strict safety checks, not correctness
//! requirements. #850 relaxes them; this test pins the round-trip in both
//! drift directions.
//!
//! `#[ignore]`d by default — each case runs the real OpenH264 encoder
//! over synthetic YUV (~1-3 s wall) and the global `StegoSession` wants
//! `--test-threads=1`. Run explicitly:
//!   cargo test -p phasm-core --features h264-encoder --test \
//!     oh264_streaming_frame_count_drift -- --ignored --test-threads=1

#![cfg(feature = "h264-encoder")]

use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingDecodeSession,
    StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::h264::stego::CostWeights;

/// Same deterministic textured YUV generator the #530 repro uses — a
/// per-frame XOR luma ramp plus an LCG-textured chroma plane (enough
/// high-frequency content for the STC to find cover positions).
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

/// Encode `actual_frames` of synthetic YUV through a session created with
/// `total_frames_hint = hint_frames` (deliberately ≠ actual), then decode
/// and assert the message round-trips.
///
/// `concentrate` installs a front-loaded plan `[framed_len, 0, …]` so the
/// whole message lives in GOP 0 — the realistic mobile shape (the #841
/// concentrate+tail planner) and the only way an OVER-estimate
/// (hint > actual) can complete, since the bare uniform-spread reserves a
/// share for GOPs that never materialise.
fn run_drift(
    width: u32,
    height: u32,
    gop_size: u32,
    hint_frames: u32,
    actual_frames: u32,
    concentrate: bool,
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
        total_frames_hint: hint_frames,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut enc =
        StreamingEncodeSession::create(params, msg, pass).expect("encode session create");

    if concentrate {
        // total_chunks = ceil(hint / gop); concentrate the entire framed
        // message into GOP 0 (the rest of the plan is a 0-byte plain tail).
        let total_chunks = hint_frames.div_ceil(gop_size) as usize;
        let framed = enc
            .framed_message_len()
            .expect("OH264 session exposes framed length");
        let mut plan = vec![0usize; total_chunks];
        plan[0] = framed;
        enc.set_gop_alloc_plan(plan).expect("install concentrate plan");
    }

    let mut annex_b = Vec::new();
    for f in 0..actual_frames {
        let (y, u, v) = synth_yuv(width, height, f);
        let frame = YuvFrameRef {
            y: &y,
            y_stride: width as usize,
            u: &u,
            u_stride: (width / 2) as usize,
            v: &v,
            v_stride: (width / 2) as usize,
        };
        enc.push_frame(frame, &mut annex_b).unwrap_or_else(|e| {
            panic!("push frame {f} (hint={hint_frames}, actual={actual_frames}): {e:?}")
        });
    }
    enc.finish(&mut annex_b).unwrap_or_else(|e| {
        panic!("finish (hint={hint_frames}, actual={actual_frames}): {e:?}")
    });
    assert!(!annex_b.is_empty(), "empty Annex-B output");

    let mut dec = StreamingDecodeSession::create(pass).expect("decode session create");
    dec.push_annex_b(&annex_b).expect("push annex-b");
    let result = dec.finish().expect("finish decode");
    assert_eq!(
        result.text, msg,
        "frame-count-drift round-trip mismatch (hint={hint_frames}, actual={actual_frames}, \
         concentrate={concentrate}): expected {msg:?}, got {:?}",
        result.text
    );
    assert_eq!(result.mode_id, 1, "mode_id should be 1 (Ghost/H.264 stego)");
    eprintln!(
        "frame-count-drift GREEN: {width}x{height} gop={gop_size} hint={hint_frames} \
         actual={actual_frames} concentrate={concentrate} ({} annex_b bytes)",
        annex_b.len()
    );
}

#[test]
#[ignore = "OH264 encode ~2s; run with --ignored --test-threads=1"]
fn underestimate_decoder_yields_more_frames_than_hint() {
    // The Artlist_CarPlane.mp4 class: the duration×fps estimate undershoots
    // the real decoded count, so the bridge pushes MORE frames than the
    // hint's GOP budget. Pre-#850 this hard-failed in `oh264_push_frame`.
    // hint=20 → 2 GOPs; push 30 frames → a 3rd (plain-tail) GOP. No plan:
    // the message spreads over the 2 hint GOPs and completes there; the
    // surplus GOP is a plain tail the decoder skips.
    run_drift(
        320, 240, /*gop=*/ 10, /*hint=*/ 20, /*actual=*/ 30,
        /*concentrate=*/ false,
        "carplane: decoder yielded more frames than the hint predicted", "pw",
    );
}

#[test]
#[ignore = "OH264 encode ~1s; run with --ignored --test-threads=1"]
fn overestimate_decoder_yields_fewer_frames_than_hint() {
    // The opposite drift: the estimate overshoots (e.g. a leading empty
    // edit inflates the reported duration), so the bridge pushes FEWER
    // frames than predicted. Pre-#850 this hard-failed in `oh264_finish`
    // ("total_frames_hint was too high"). hint=30 → 3 GOPs planned; push
    // only 10 frames → 1 GOP. The concentrate plan puts the whole message
    // in GOP 0, which materialises, so the encode completes.
    run_drift(
        320, 240, /*gop=*/ 10, /*hint=*/ 30, /*actual=*/ 10,
        /*concentrate=*/ true,
        "leading-edit: decoder yielded fewer frames than the hint", "pw",
    );
}

#[test]
#[ignore = "OH264 encode ~1s; run with --ignored --test-threads=1"]
fn exact_count_control_unchanged() {
    // Control: hint == actual exercises the unchanged path (no relaxed
    // assert fires), proving the #850 change is a pure tolerance superset
    // and the byte-for-byte wire output (→ the #835 determinism SHA) is
    // untouched when the estimate is exact.
    run_drift(
        320, 240, /*gop=*/ 10, /*hint=*/ 20, /*actual=*/ 20,
        /*concentrate=*/ false,
        "control: hint equals the real frame count", "pw",
    );
}
