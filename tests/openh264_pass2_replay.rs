// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// #533 Phase 1 (Option A: Pass-2 replay architecture).
//
// Bit-identity gate scaffolding. This file exists to land alongside
// the Phase 1 ABI plumbing. Real test logic (CAPTURE → REPLAY →
// byte-equal recon) wires up in Phase 3 once REPLAY pass mode is
// implemented inside the fork. All tests are #[ignore] until then.
//
// What the eventual gate proves: encoding the same source twice —
// once in CAPTURE pass (Pass-1, populates DecisionCache), then in
// REPLAY pass (Pass-2, fetches decisions from the cache) — produces
// bit-identical recon pixels frame-for-frame. Bit-identity is the
// strongest statement we can make: no drift, no rounding, no
// rate/distortion divergence. If REPLAY matches CAPTURE byte-for-
// byte on a clean cover (no stego flips), it will match on a flipped
// cover too because the decisions are forced, not re-derived.

#![cfg(feature = "h264-encoder")]

use phasm_core::codec::h264::openh264::{
    abi_version, header_abi_version, set_frame_num, set_pass_mode, Encoder, FrameType, MbDecision,
    PassMode, StegoHandlers, StegoSession,
};
use phasm_core::codec::h264::pass2_cache::DecisionCache;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Mutex;

/// Process-wide guard: only one test may hold a StegoSession at a
/// time (SESSION_ALIVE in openh264.rs is process-global). cargo
/// test runs the binary's tests in parallel by default, so we
/// serialise here.
static SESSION_GUARD: Mutex<()> = Mutex::new(());

/// ABI sanity: fork's runtime version matches the bindings' compile-
/// time version. If this fails, someone bumped one side without
/// re-pinning the SHA in `core/openh264-sys/build.rs`.
#[test]
fn abi_versions_match() {
    let runtime = abi_version();
    let compile = header_abi_version();
    assert_eq!(
        runtime, compile,
        "ABI version mismatch: fork={runtime:#x}, bindings={compile:#x}. \
         Either the fork SHA pin or the bindings header is stale."
    );
}

/// PASSTHROUGH is the default — registering with no capture / replay
/// closures and not calling `set_pass_mode` means the encoder runs
/// unchanged. Smoke-test that the session lifecycle works for the
/// pass-mode API.
#[test]
fn set_pass_mode_smoke() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let handlers = StegoHandlers::default();
    let _session = StegoSession::register(handlers).expect("register");

    // Default is PASSTHROUGH. Cycle through all three modes without
    // encoding to confirm the FFI accepts them.
    set_pass_mode(PassMode::Passthrough);
    set_pass_mode(PassMode::Capture);
    set_pass_mode(PassMode::Replay);
    set_pass_mode(PassMode::Passthrough);
}

/// Phase 2 (#536): encode one IDR with pass_mode=CAPTURE and assert
/// the registered capture closure fires once per MB. For a 320×240
/// frame, that's 20 × 15 = 300 MBs.
#[test]
fn pass1_capture_populates_cache_320x240() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 26;
    let expected_mbs = (WIDTH / 16) * (HEIGHT / 16);

    let cache = Rc::new(RefCell::new(DecisionCache::with_capacity(expected_mbs)));
    let capture_cache = Rc::clone(&cache);

    let handlers = StegoHandlers {
        capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
            capture_cache.borrow_mut().insert(*d);
        })),
        ..Default::default()
    };
    let _session = StegoSession::register(handlers).expect("register");

    set_pass_mode(PassMode::Capture);
    set_frame_num(0);

    let (y_in, u_in, v_in) = synth_yuv_frame(WIDTH, HEIGHT, 0);
    let mut out = vec![0u8; 256 * 1024];
    let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("encoder create");
    let (ftype, _n) = enc
        .encode_frame(&y_in, &u_in, &v_in, 0, &mut out)
        .expect("encode_frame");
    assert_eq!(ftype, FrameType::Idr);

    let got = cache.borrow().len();
    assert_eq!(
        got, expected_mbs,
        "Pass-1 capture should fire once per MB ({} expected, got {})",
        expected_mbs, got
    );

    // Reset to default so subsequent tests aren't affected.
    set_pass_mode(PassMode::Passthrough);
}

/// Synthetic YUV generator matching the encoder lib tests.
fn synth_yuv_frame(width: usize, height: usize, frame_idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; width * height];
    let mut u = vec![128u8; (width / 2) * (height / 2)];
    let mut v = vec![128u8; (width / 2) * (height / 2)];
    let n = |fi: u32, p: u8, i: usize, j: usize| -> u8 {
        (((fi as usize).wrapping_mul(7).wrapping_add(p as usize) ^ (i * 3) ^ (j * 5)) & 0xff) as u8
    };
    for j in 0..height {
        for i in 0..width {
            y[j * width + i] = n(frame_idx, 0, i, j);
        }
    }
    for j in 0..height / 2 {
        for i in 0..width / 2 {
            u[j * (width / 2) + i] = n(frame_idx, 1, i, j);
            v[j * (width / 2) + i] = n(frame_idx, 2, i, j);
        }
    }
    (y, u, v)
}

/// Phase 3 Stage 1 (#537): proves the REPLAY-mode entry-check fires
/// the replay callback during a P-frame encode. Counts invocations,
/// expects one call per MB on the P-frame. Cache miss (callback
/// returns 0) ⇒ encoder falls back to normal mode decision.
#[test]
#[ignore = "Stale post-Bug-1 hotfix: phasm_replay_inter_override returns 0 in REPLAY (svc_base_layer_md.cpp:2068), so the REPLAY callback no longer fires per MB. Kept for the day we revive cache-application."]
fn pass2_replay_callback_fires_per_mb_on_p_frame() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 26;
    let expected_mbs = (WIDTH / 16) * (HEIGHT / 16);

    let calls = Rc::new(RefCell::new(0u32));
    let replay_calls = Rc::clone(&calls);

    // Replay callback that always returns 0 (cache miss) — encoder
    // should fall through to normal mode decision. We just count.
    let handlers = StegoHandlers {
        replay_mb_decision: Some(Box::new(move |_fnum, _mx, _my| -> Option<MbDecision> {
            *replay_calls.borrow_mut() += 1;
            None
        })),
        ..Default::default()
    };
    let _session = StegoSession::register(handlers).expect("register");

    let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("encoder create");

    // Frame 0 IDR — PASSTHROUGH (no callback fires on intra path yet).
    set_pass_mode(PassMode::Passthrough);
    set_frame_num(0);
    let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
    let mut out = vec![0u8; 256 * 1024];
    let (ftype, _) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out).expect("frame 0");
    assert_eq!(ftype, FrameType::Idr);

    // Frame 1 P — REPLAY mode. Expect one callback per MB on the
    // inter path.
    set_pass_mode(PassMode::Replay);
    set_frame_num(1);
    let (y1, u1, v1) = synth_yuv_frame(WIDTH, HEIGHT, 1);
    let (ftype, _) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out).expect("frame 1");
    assert_eq!(ftype, FrameType::P);

    let got = *calls.borrow();
    assert_eq!(
        got, expected_mbs as u32,
        "REPLAY callback should fire once per MB on a P-frame ({} expected, got {})",
        expected_mbs, got
    );

    set_pass_mode(PassMode::Passthrough);
}

/// Phase 3 Stage 2A (#542): CAPTURE a P-frame, then REPLAY the same
/// source with the captured cache. Asserts (a) REPLAY encode succeeds
/// (the new MB_TYPE_16x16 dispatch doesn't crash on real cached MVs),
/// (b) cache contained at least one P_16x16 entry so Stage 2A path was
/// genuinely exercised, and (c) reports whether CAPTURE vs REPLAY bytes
/// match (informational — full bit-identity requires Stages 2B/2C/2D
/// for partitioned/intra coverage and stays #[ignore] until then).
#[test]
#[ignore = "Stale post-Bug-1 hotfix: REPLAY mb_decision callback no longer fires (svc_base_layer_md.cpp:2068 early return). Kept for the day we revive cache-application."]
fn pass2_replay_stage2a_p16x16_exercises_dispatch() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 26;
    const MB_TYPE_16X16: u16 = 0x08;
    const MB_TYPE_SKIP: u16 = 0x100;
    const MB_TYPE_INTRA_ANY: u16 = 0x01 | 0x02 | 0x04 | 0x400 | 0x200;

    // Phase 1 — CAPTURE pass: encode IDR PASSTHROUGH, then P CAPTURE.
    let cache_rc = Rc::new(RefCell::new(DecisionCache::with_capacity(1024)));
    let p_cap_bytes: Vec<u8>;
    let (n_skip, n_p16, n_intra, n_other);
    {
        let cap_cache = Rc::clone(&cache_rc);
        let handlers = StegoHandlers {
            capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
                cap_cache.borrow_mut().insert(*d);
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-cap");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-cap");
        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut out = vec![0u8; 256 * 1024];
        let (ft0, _) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out).expect("cap-idr");
        assert_eq!(ft0, FrameType::Idr);

        set_pass_mode(PassMode::Capture);
        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame(WIDTH, HEIGHT, 1);
        let mut out_p = vec![0u8; 256 * 1024];
        let (ft1, n_p) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out_p).expect("cap-p");
        assert_eq!(ft1, FrameType::P);
        p_cap_bytes = out_p[..n_p].to_vec();
        set_pass_mode(PassMode::Passthrough);

        // Snapshot mb_type distribution while we still hold the session.
        let mut skip = 0usize;
        let mut p16 = 0usize;
        let mut intra = 0usize;
        let mut other = 0usize;
        let mb_w = WIDTH / 16;
        let mb_h = HEIGHT / 16;
        for my in 0..mb_h as u16 {
            for mx in 0..mb_w as u16 {
                if let Some(d) = cache_rc.borrow().get(1, mx, my) {
                    let t = d.ui_mb_type;
                    if t & MB_TYPE_SKIP != 0 {
                        skip += 1;
                    } else if t & MB_TYPE_INTRA_ANY != 0 {
                        intra += 1;
                    } else if t & MB_TYPE_16X16 != 0 {
                        p16 += 1;
                    } else {
                        other += 1;
                    }
                }
            }
        }
        n_skip = skip;
        n_p16 = p16;
        n_intra = intra;
        n_other = other;
    }
    let total_p_mbs = n_skip + n_p16 + n_intra + n_other;
    eprintln!(
        "Stage 2A capture: P-frame {} MBs → {} Skip / {} P_16x16 / {} intra / {} other",
        total_p_mbs, n_skip, n_p16, n_intra, n_other
    );
    assert_eq!(
        total_p_mbs,
        (WIDTH / 16) * (HEIGHT / 16),
        "Capture should fire once per P-frame MB"
    );

    // Phase 2 — REPLAY pass: re-encode same source, lookup from cache.
    let p_rep_bytes: Vec<u8>;
    let replay_calls: u32;
    {
        let rep_cache = Rc::clone(&cache_rc);
        let calls_cell = Rc::new(RefCell::new(0u32));
        let calls_cap = Rc::clone(&calls_cell);
        let handlers = StegoHandlers {
            replay_mb_decision: Some(Box::new(move |fnum, mx, my| {
                *calls_cap.borrow_mut() += 1;
                rep_cache.borrow().get(fnum, mx, my)
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-rep");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-rep");
        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut out = vec![0u8; 256 * 1024];
        let (ft0, _) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out).expect("rep-idr");
        assert_eq!(ft0, FrameType::Idr);

        set_pass_mode(PassMode::Replay);
        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame(WIDTH, HEIGHT, 1);
        let mut out_p = vec![0u8; 256 * 1024];
        let (ft1, n_p) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out_p).expect("rep-p");
        assert_eq!(ft1, FrameType::P);
        p_rep_bytes = out_p[..n_p].to_vec();
        set_pass_mode(PassMode::Passthrough);
        replay_calls = *calls_cell.borrow();
    }

    // (a) Encode succeeded — Stage 2A dispatch didn't crash on real MVs.
    assert!(!p_rep_bytes.is_empty(), "REPLAY produced empty P-frame");

    // (b) Replay callback fired once per MB (the REPLAY entry-check ran).
    assert_eq!(
        replay_calls,
        total_p_mbs as u32,
        "Replay callback should fire once per MB ({} expected, got {})",
        total_p_mbs,
        replay_calls
    );

    // (c) Bit-identity informational. Full byte-equal only achievable
    //     once Stages 2B/2C/2D close the partitioned + intra branches.
    if p_cap_bytes == p_rep_bytes {
        eprintln!("Stage 2A: CAPTURE == REPLAY P-frame bytes (byte-identical)");
    } else {
        eprintln!(
            "Stage 2A: CAPTURE vs REPLAY P-frame divergence — cap={}B rep={}B",
            p_cap_bytes.len(),
            p_rep_bytes.len()
        );
        for (i, (a, b)) in p_cap_bytes.iter().zip(p_rep_bytes.iter()).enumerate() {
            if a != b {
                eprintln!("  first byte diff at offset {}: cap={:#x} rep={:#x}", i, a, b);
                break;
            }
        }
        eprintln!(
            "  (expected while {} intra + {} other MBs fall through; closes with Stages 2B/2C/2D)",
            n_intra, n_other
        );
    }
}

/// Phase 3 Stage 2B.b (#544): a multi-frame CAPTURE→REPLAY round-trip
/// that exercises partitioned mb_types (16x8 / 8x16 / 8x8). Each
/// P-frame's REPLAY output must be byte-identical to its CAPTURE
/// output. Reports the per-frame mb_type histogram for diagnostics.
///
/// The architectural fix that closed this gate:
/// 1. Moved the REPLAY override from the top of `WelsMdInterMb` to
///    `WelsMdInterSecondaryModesEnc` between `WelsMdInterMbRefinement`
///    and `WelsMdInterEncode`. Aux mode-decision functions (BGD/SCDP/
///    JudgePskip/P16x16/FirstIntraMode/Refinement) run normally with
///    natural inputs, so their state updates (BGD/SCDP/SAD stats /
///    iSadPredSkip / pSadCost) fire correctly for downstream frames.
/// 2. Fixed the MC pre-shift contract — `McLuma_c` / `McChroma_c` both
///    require the caller to pre-shift pSrc by the integer MV (only the
///    sub-pel part of MV is used internally). The original Stage 2A/2B
///    code passed unshifted pRefLuma / pRefCb / pRefCr, which silently
///    worked for MV=(0,0) on synth-noise fixtures but broke as soon as
///    ME found non-zero MVs at frame 2+. WelsMdPSkipEnc shows the
///    correct pattern.
#[test]
fn pass2_replay_stage2b_multi_frame_byte_identity() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 26;
    const N_P_FRAMES: u32 = 4;
    const MB_TYPE_16X16: u16 = 0x08;
    const MB_TYPE_16X8: u16 = 0x10;
    const MB_TYPE_8X16: u16 = 0x20;
    const MB_TYPE_8X8: u16 = 0x40;
    const MB_TYPE_SKIP: u16 = 0x100;
    const MB_TYPE_INTRA_ANY: u16 = 0x01 | 0x02 | 0x04 | 0x400 | 0x200;
    const SUB_MB_TYPE_8X8: u8 = 0x01;

    // Phase 1 — CAPTURE: encode IDR + N P-frames, fill cache, record bytes.
    let cache_rc = Rc::new(RefCell::new(DecisionCache::with_capacity(2048)));
    let p_bytes_cap: Vec<Vec<u8>>;
    {
        let cap_cache = Rc::clone(&cache_rc);
        let handlers = StegoHandlers {
            capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
                cap_cache.borrow_mut().insert(*d);
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-cap");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-cap");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut idr_out = vec![0u8; 256 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("cap-idr");

        set_pass_mode(PassMode::Capture);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = synth_yuv_frame(WIDTH, HEIGHT, fi);
            let mut out = vec![0u8; 256 * 1024];
            let (ftype, n) = enc
                .encode_frame(&y, &u, &v, fi as i64, &mut out)
                .expect("cap-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_cap = p_bytes;
    }

    // Histogram of cached mb_types across all P-frames.
    let mb_w = WIDTH / 16;
    let mb_h = HEIGHT / 16;
    let total_p_mbs = mb_w * mb_h * N_P_FRAMES as usize;
    let mut n_skip = 0usize;
    let mut n_p16 = 0usize;
    let mut n_p16x8 = 0usize;
    let mut n_p8x16 = 0usize;
    let mut n_p8x8_uniform = 0usize;
    let mut n_p8x8_mixed = 0usize;
    let mut n_intra = 0usize;
    let mut n_other = 0usize;
    for fi in 1..=N_P_FRAMES {
        for my in 0..mb_h as u16 {
            for mx in 0..mb_w as u16 {
                if let Some(d) = cache_rc.borrow().get(fi, mx, my) {
                    let t = d.ui_mb_type;
                    if t & MB_TYPE_SKIP != 0 {
                        n_skip += 1;
                    } else if t & MB_TYPE_INTRA_ANY != 0 {
                        n_intra += 1;
                    } else if t & MB_TYPE_16X8 != 0 {
                        n_p16x8 += 1;
                    } else if t & MB_TYPE_8X16 != 0 {
                        n_p8x16 += 1;
                    } else if t & MB_TYPE_8X8 != 0 {
                        let uniform = d.sub_mb_type.iter().all(|&s| s == SUB_MB_TYPE_8X8);
                        if uniform {
                            n_p8x8_uniform += 1;
                        } else {
                            n_p8x8_mixed += 1;
                        }
                    } else if t & MB_TYPE_16X16 != 0 {
                        n_p16 += 1;
                    } else {
                        n_other += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "Stage 2B histogram (P-frames × {} = {} MBs): Skip={} P_16x16={} 16x8={} 8x16={} 8x8_uniform={} 8x8_mixed={} intra={} other={}",
        N_P_FRAMES, total_p_mbs,
        n_skip, n_p16, n_p16x8, n_p8x16, n_p8x8_uniform, n_p8x8_mixed, n_intra, n_other
    );

    // Phase 2 — REPLAY: re-encode same sequence, lookup from cache.
    // Optionally bisect: disable specific Stage 2B branches by returning
    // None for them so they fall through to natural mode-decision.
    let disable_p16x8 = std::env::var("PHASM_BISECT_NO_16X8").is_ok();
    let disable_p8x16 = std::env::var("PHASM_BISECT_NO_8X16").is_ok();
    let disable_p8x8  = std::env::var("PHASM_BISECT_NO_8X8").is_ok();
    let p_bytes_rep: Vec<Vec<u8>>;
    {
        let rep_cache = Rc::clone(&cache_rc);
        let handlers = StegoHandlers {
            replay_mb_decision: Some(Box::new(move |fnum, mx, my| {
                rep_cache.borrow().get(fnum, mx, my).and_then(|d| {
                    let t = d.ui_mb_type;
                    if disable_p16x8 && (t & MB_TYPE_16X8) != 0
                        && (t & (MB_TYPE_INTRA_ANY | MB_TYPE_SKIP)) == 0 {
                        return None;
                    }
                    if disable_p8x16 && (t & MB_TYPE_8X16) != 0
                        && (t & (MB_TYPE_INTRA_ANY | MB_TYPE_SKIP)) == 0 {
                        return None;
                    }
                    if disable_p8x8 && (t & MB_TYPE_8X8) != 0
                        && (t & (MB_TYPE_INTRA_ANY | MB_TYPE_SKIP)) == 0 {
                        return None;
                    }
                    Some(d)
                })
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-rep");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-rep");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut idr_out = vec![0u8; 256 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("rep-idr");

        set_pass_mode(PassMode::Replay);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = synth_yuv_frame(WIDTH, HEIGHT, fi);
            let mut out = vec![0u8; 256 * 1024];
            let (ftype, n) = enc
                .encode_frame(&y, &u, &v, fi as i64, &mut out)
                .expect("rep-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_rep = p_bytes;
    }

    // Per-frame byte-identity: every P-frame in REPLAY must match CAPTURE.
    // If this fails, either a Stage 2B branch has a bug or some MB falls
    // through to natural decision which diverged (the latter shouldn't
    // happen for inter-only fixtures with no flips).
    for (i, (cap, rep)) in p_bytes_cap.iter().zip(p_bytes_rep.iter()).enumerate() {
        if cap == rep {
            eprintln!("Stage 2B: P-frame {} byte-identical ({} bytes)", i + 1, cap.len());
        } else {
            eprintln!(
                "Stage 2B: P-frame {} DIVERGENCE — cap={}B rep={}B",
                i + 1,
                cap.len(),
                rep.len()
            );
            for (j, (a, b)) in cap.iter().zip(rep.iter()).enumerate() {
                if a != b {
                    eprintln!("  first byte diff at offset {}: cap={:#x} rep={:#x}", j, a, b);
                    break;
                }
            }
        }
    }
    for (i, (cap, rep)) in p_bytes_cap.iter().zip(p_bytes_rep.iter()).enumerate() {
        assert_eq!(cap, rep, "Stage 2B P-frame {} divergence", i + 1);
    }
}

/// Phase 3 Stage 2C (#545): I-slice (IDR) CAPTURE→REPLAY byte-identity.
///
/// Encodes an IDR in CAPTURE mode (all 300 MBs are intra: I_16x16 or
/// I_4x4), then re-encodes the same source in REPLAY mode. Verifies
/// that on a clean (no-flip) cover, Pass-2's natural intra
/// mode-decision matches Pass-1's. This is the state-sync hypothesis:
///
///   The IDR is the FIRST frame in the encoder's lifetime — there are
///   no inter-frame state dependencies, no rate-control history,
///   no pVaa carryover. Given identical source pixels and identical
///   slice config, `WelsMdIntraMb` (and the I_16x16/I_4x4 sub-RDO it
///   drives) is deterministic.  If state-sync holds, the cache is
///   reference-only for I-slice MBs — no override needed.
///
/// Passes today → state-sync covers clean cover; Stage 2C.full
/// (forced intra pred-mode override) deferred to v1.1+ for stego-flip
/// scenarios where Pass-2 RDO might diverge from cache.
/// Fails today → override implementation is the next blocker.
#[test]
fn pass2_replay_stage2c_idr_intra_byte_identity() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 26;
    const MB_TYPE_INTRA16X16: u16 = 0x02;
    const MB_TYPE_INTRA4X4: u16 = 0x01;
    const MB_TYPE_INTRA8X8: u16 = 0x04;
    const MB_TYPE_INTRA_PCM: u16 = 0x200;
    const MB_TYPE_INTRA_ANY: u16 =
        MB_TYPE_INTRA16X16 | MB_TYPE_INTRA4X4 | MB_TYPE_INTRA8X8 | MB_TYPE_INTRA_PCM;

    // Phase 1 — CAPTURE the IDR (all I-slice MBs).
    let cache_rc = Rc::new(RefCell::new(DecisionCache::with_capacity(1024)));
    let idr_bytes_cap: Vec<u8>;
    {
        let cap_cache = Rc::clone(&cache_rc);
        let handlers = StegoHandlers {
            capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
                cap_cache.borrow_mut().insert(*d);
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-cap");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-cap");

        set_pass_mode(PassMode::Capture);
        set_frame_num(0);
        let (y, u, v) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut out = vec![0u8; 256 * 1024];
        let (ftype, n) = enc.encode_frame(&y, &u, &v, 0, &mut out).expect("cap-idr");
        assert_eq!(ftype, FrameType::Idr);
        idr_bytes_cap = out[..n].to_vec();
        set_pass_mode(PassMode::Passthrough);
    }

    // Verify every captured MB is intra (sanity: it IS an I-slice).
    let mb_w = WIDTH / 16;
    let mb_h = HEIGHT / 16;
    let mut n_intra16 = 0usize;
    let mut n_intra4x4 = 0usize;
    let mut n_intra_pcm = 0usize;
    let mut n_non_intra = 0usize;
    for my in 0..mb_h as u16 {
        for mx in 0..mb_w as u16 {
            if let Some(d) = cache_rc.borrow().get(0, mx, my) {
                let t = d.ui_mb_type;
                if t & MB_TYPE_INTRA16X16 != 0 {
                    n_intra16 += 1;
                } else if t & MB_TYPE_INTRA4X4 != 0 {
                    n_intra4x4 += 1;
                } else if t & MB_TYPE_INTRA_PCM != 0 {
                    n_intra_pcm += 1;
                } else if t & MB_TYPE_INTRA_ANY == 0 {
                    n_non_intra += 1;
                }
            }
        }
    }
    eprintln!(
        "Stage 2C IDR cache: I_16x16={} I_4x4={} I_PCM={} non-intra={}",
        n_intra16, n_intra4x4, n_intra_pcm, n_non_intra
    );
    assert_eq!(
        n_non_intra, 0,
        "I-slice should have only intra MBs but cache shows {} non-intra",
        n_non_intra
    );

    // Phase 2 — REPLAY the same source.
    let idr_bytes_rep: Vec<u8>;
    {
        let rep_cache = Rc::clone(&cache_rc);
        let handlers = StegoHandlers {
            replay_mb_decision: Some(Box::new(move |fnum, mx, my| {
                rep_cache.borrow().get(fnum, mx, my)
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-rep");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-rep");

        set_pass_mode(PassMode::Replay);
        set_frame_num(0);
        let (y, u, v) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut out = vec![0u8; 256 * 1024];
        let (ftype, n) = enc.encode_frame(&y, &u, &v, 0, &mut out).expect("rep-idr");
        assert_eq!(ftype, FrameType::Idr);
        idr_bytes_rep = out[..n].to_vec();
        set_pass_mode(PassMode::Passthrough);
    }

    if idr_bytes_cap == idr_bytes_rep {
        eprintln!(
            "Stage 2C: IDR byte-identical CAPTURE↔REPLAY ({} bytes, all intra)",
            idr_bytes_cap.len()
        );
    } else {
        eprintln!(
            "Stage 2C: IDR DIVERGENCE — cap={}B rep={}B",
            idr_bytes_cap.len(),
            idr_bytes_rep.len()
        );
        for (i, (a, b)) in idr_bytes_cap.iter().zip(idr_bytes_rep.iter()).enumerate() {
            if a != b {
                eprintln!("  first byte diff at offset {}: cap={:#x} rep={:#x}", i, a, b);
                break;
            }
        }
    }
    assert_eq!(idr_bytes_cap, idr_bytes_rep, "Stage 2C IDR divergence");
}

// `bit_identity_capture_then_replay` scaffold removed in #545 — the Phase
// 1 placeholder is fully superseded by the Stage 2B P-frame + Stage 2C
// I-slice byte-identity tests above, which between them cover the entire
// mb_type universe on the synth-noise fixture.

/// Phase 4.6 (#538) — wire-only architecture diagnostic counters.
///
/// Diagnostic-only test: reads the fork-side atomic counters via FFI
/// after one IDR encode under wire-only flag ON, prints the counts via
/// --nocapture. Asserts nothing other than the FFI counters are
/// readable; the actual end-to-end assertion lives in
/// pass2_replay_wire_only_forged_flip_cascade_safe below.
///
/// Expected output (post-Phase 4.6 fix):
///   set_calls  ≈ 16 (per FLIP_COUNT)
///   set_writes ≈ 16 (none rejected)
///   apply_hits > 0  (luma fixes reach the wire; 10/16 typical on
///                    synth 320x240 — remaining 6 may land on chroma
///                    where Cb↔Cr collision in the scratch key is a
///                    known TODO for v1.1).
#[test]
fn pass2_replay_wire_only_diagnostic_counters() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 26;
    const FLIP_COUNT: u32 = 16;

    unsafe {
        core_openh264_sys::phasm_diag_reset_counters();
        core_openh264_sys::phasm_set_use_wire_only_overrides(1);
    }

    let dispatches = Rc::new(RefCell::new(0u32));
    let dispatches_cb = Rc::clone(&dispatches);
    let handlers = StegoHandlers {
        enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
            if pos.domain != 1 || pos.frame_num != 0 {
                return None;
            }
            *dispatches_cb.borrow_mut() += 1;
            let n = *dispatches_cb.borrow();
            if n <= FLIP_COUNT {
                Some(1 - original)
            } else {
                None
            }
        })),
        ..Default::default()
    };
    let _sess = StegoSession::register(handlers).expect("diag register");
    let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("diag enc");
    set_pass_mode(PassMode::Passthrough);
    set_frame_num(0);
    let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
    let mut out = vec![0u8; 256 * 1024];
    let (_ft, _n) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out).expect("diag idr");

    let set_calls         = unsafe { core_openh264_sys::phasm_diag_get_set_calls() };
    let set_writes        = unsafe { core_openh264_sys::phasm_diag_get_set_writes() };
    let set_rejected      = unsafe { core_openh264_sys::phasm_diag_get_set_rejected_oob() };
    let apply_calls       = unsafe { core_openh264_sys::phasm_diag_get_apply_calls() };
    let apply_hits        = unsafe { core_openh264_sys::phasm_diag_get_apply_hits() };
    let reset_calls       = unsafe { core_openh264_sys::phasm_diag_get_reset_calls() };
    let total_dispatches  = *dispatches.borrow();

    eprintln!("[wire-only diag]");
    eprintln!("  callback dispatches      : {}", total_dispatches);
    eprintln!("  phasm_set_bypass_override:");
    eprintln!("    total calls            : {}", set_calls);
    eprintln!("    slot writes            : {}", set_writes);
    eprintln!("    rejected (out-of-bound): {}", set_rejected);
    eprintln!("  phasm_apply_bypass_bin_override:");
    eprintln!("    total calls            : {}", apply_calls);
    eprintln!("    slot hits (non-zero)   : {}", apply_hits);
    eprintln!("  phasm_reset_bypass_overrides:");
    eprintln!("    total calls            : {}", reset_calls);

    unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }

    // Diagnostic-only — counters readable, fix-validated end-to-end by
    // the assert-flavoured test below. apply_hits > 0 confirms scratch
    // populate↔emit alignment for at least the luma path; full 16/16
    // hit count is the v1.1 target (depends on chroma fixes).
}

///
/// What the test asserts:
///   1. Baseline encode (flag OFF, no callback) → idr_clean, p_clean.
///   2. Forged-flip encode (flag ON, callback flips first N CoeffSign
///      fires) → idr_stego, p_stego.
///   3. idr_clean != idr_stego  (wire-only override propagates to bitstream).
///   4. p_clean == p_stego      (cascade-safety: clean reference frame
///                               for inter prediction under flag ON).
///
/// Without 4.5.d's raster→scan reconciliation this test wouldn't pass
/// for CoeffSign — overrides would silently no-op for AC sites. With
/// 4.5.d shipped, the flip lands on the wire reliably for both DC
/// (HOOK-A) and AC sub-blocks.
#[test]
fn pass2_replay_wire_only_forged_flip_cascade_safe() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 26;
    const FLIP_COUNT: u32 = 16; // first 16 CoeffSign fires get flipped

    // ----------------------------------------------------------------
    // Baseline: flag OFF, no callback → unmodified bitstream.
    // ----------------------------------------------------------------
    let idr_clean: Vec<u8>;
    let p_clean: Vec<u8>;
    {
        // Ensure flag is OFF (it's the default but be explicit).
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }

        let handlers = StegoHandlers::default();
        let _sess = StegoSession::register(handlers).expect("baseline register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60)
            .expect("baseline enc");
        set_pass_mode(PassMode::Passthrough);

        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut out_idr = vec![0u8; 256 * 1024];
        let (ft0, n0) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out_idr)
            .expect("baseline idr");
        assert_eq!(ft0, FrameType::Idr);
        idr_clean = out_idr[..n0].to_vec();

        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame(WIDTH, HEIGHT, 1);
        let mut out_p = vec![0u8; 256 * 1024];
        let (ft1, n1) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out_p)
            .expect("baseline p");
        assert_eq!(ft1, FrameType::P);
        p_clean = out_p[..n1].to_vec();
    }

    // ----------------------------------------------------------------
    // Wire-only: flag ON, callback flips the first FLIP_COUNT CoeffSign
    // fires. Only fires on the IDR (frame_num=0); the P-frame callback
    // returns None so no flips fire there.
    // ----------------------------------------------------------------
    let idr_stego: Vec<u8>;
    let p_stego: Vec<u8>;
    let total_dispatches: u32;
    let actual_flips: u32;
    {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(1); }
        let flag_state = unsafe { core_openh264_sys::phasm_get_use_wire_only_overrides() };
        assert_eq!(flag_state, 1, "wire-only flag failed to set via FFI");

        let dispatches = Rc::new(RefCell::new(0u32));
        let flips = Rc::new(RefCell::new(0u32));
        let dispatches_cb = Rc::clone(&dispatches);
        let flips_cb = Rc::clone(&flips);

        let handlers = StegoHandlers {
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                // Only flip CoeffSign (domain=1) on IDR (frame_num=0).
                // PhasmStegoDomain::CoeffSign == 1.
                if pos.domain != 1 || pos.frame_num != 0 {
                    return None;
                }
                *dispatches_cb.borrow_mut() += 1;
                let n = *dispatches_cb.borrow();
                if n <= FLIP_COUNT {
                    *flips_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("wire-only register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60)
            .expect("wire-only enc");
        set_pass_mode(PassMode::Passthrough);

        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut out_idr = vec![0u8; 256 * 1024];
        let (ft0, n0) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out_idr)
            .expect("wire-only idr");
        assert_eq!(ft0, FrameType::Idr);
        idr_stego = out_idr[..n0].to_vec();

        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame(WIDTH, HEIGHT, 1);
        let mut out_p = vec![0u8; 256 * 1024];
        let (ft1, n1) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out_p)
            .expect("wire-only p");
        assert_eq!(ft1, FrameType::P);
        p_stego = out_p[..n1].to_vec();

        total_dispatches = *dispatches.borrow();
        actual_flips = *flips.borrow();

        // Always restore default before the test exits.
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
    }

    // ----------------------------------------------------------------
    // Assertions.
    // ----------------------------------------------------------------
    assert!(
        total_dispatches > 0,
        "callback never fired — IDR had no CoeffSign emit sites? \
         (callback dispatches: {}, flips: {})",
        total_dispatches, actual_flips
    );
    assert_eq!(
        actual_flips, FLIP_COUNT,
        "expected exactly {} CoeffSign flips on IDR, got {}",
        FLIP_COUNT, actual_flips
    );

    assert_ne!(
        idr_clean, idr_stego,
        "IDR bytes are byte-identical despite {} forged CoeffSign flips — \
         wire-only override is not propagating to the bitstream. \
         Baseline IDR: {} bytes; forged IDR: {} bytes.",
        actual_flips, idr_clean.len(), idr_stego.len()
    );

    assert_eq!(
        p_clean, p_stego,
        "P-frame DIVERGES between baseline and wire-only encode. This \
         means the IDR flips polluted the encoder's reference frame \
         (pDecPic), breaking cascade-safety — the wire-only mode is \
         supposed to keep encoder state clean. Baseline P: {} bytes; \
         forged P: {} bytes.",
        p_clean.len(), p_stego.len()
    );
}

/// #549 Phase D test gap closure (2026-05-19): the
/// `pass2_replay_wire_only_forged_flip_cascade_safe` test above only
/// covers PHASM_DOMAIN_COEFF_SIGN (dom=1) cascade-safety. The 4-domain
/// production stego flips bits in all 4 domains, and the #549 failure
/// observed mode-decision drift on frame=2 that couldn't be explained
/// by CS flips alone (since the synthetic CS test passes at 16 flips).
/// This companion test forges PHASM_DOMAIN_COEFF_SUFFIX_LSB (dom=0)
/// flips on the IDR and asserts the next P-frame's bitstream is
/// byte-identical to baseline. Failure = CSL flips leak into encoder
/// reference frame, violating wire_only=1's design contract.
///
/// CSL fires only when |coeff|>=15 (vendor/phasm-openh264/codec/encoder/
/// core/src/wels_stego.cpp:339), which is rare at QP=26 on synthetic
/// content. Test runs at QP=18 (less aggressive quantization) to
/// produce more |coeff|>=16 sites and exercise the CSL hook.
#[test]
fn pass2_replay_wire_only_forged_csl_flip_cascade_safe() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 18; // lower QP → more |coeff|>=16 sites → more CSL fires
    const FLIP_COUNT: u32 = 16;

    // Baseline: flag OFF, no callback.
    let idr_clean: Vec<u8>;
    let p_clean: Vec<u8>;
    {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
        let handlers = StegoHandlers::default();
        let _sess = StegoSession::register(handlers).expect("baseline register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("baseline enc");
        set_pass_mode(PassMode::Passthrough);

        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut out_idr = vec![0u8; 256 * 1024];
        let (ft0, n0) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out_idr).expect("baseline idr");
        assert_eq!(ft0, FrameType::Idr);
        idr_clean = out_idr[..n0].to_vec();

        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame(WIDTH, HEIGHT, 1);
        let mut out_p = vec![0u8; 256 * 1024];
        let (ft1, n1) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out_p).expect("baseline p");
        assert_eq!(ft1, FrameType::P);
        p_clean = out_p[..n1].to_vec();
    }

    // Wire-only: flag ON, flip first FLIP_COUNT CoeffSuffixLsb fires on IDR.
    let idr_stego: Vec<u8>;
    let p_stego: Vec<u8>;
    let total_dispatches: u32;
    let actual_flips: u32;
    let apply_hits: u64;
    {
        unsafe {
            core_openh264_sys::phasm_diag_reset_counters();
            core_openh264_sys::phasm_set_use_wire_only_overrides(1);
        }

        let dispatches = Rc::new(RefCell::new(0u32));
        let flips = Rc::new(RefCell::new(0u32));
        let dispatches_cb = Rc::clone(&dispatches);
        let flips_cb = Rc::clone(&flips);

        let handlers = StegoHandlers {
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                // Only flip CoeffSuffixLsb (domain=0) on IDR (frame_num=0).
                // PhasmStegoDomain::CoeffSuffixLsb == 0.
                if pos.domain != 0 || pos.frame_num != 0 {
                    return None;
                }
                *dispatches_cb.borrow_mut() += 1;
                let n = *dispatches_cb.borrow();
                if n <= FLIP_COUNT {
                    *flips_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("wire-only csl register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("wire-only csl enc");
        set_pass_mode(PassMode::Passthrough);

        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut out_idr = vec![0u8; 256 * 1024];
        let (ft0, n0) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out_idr).expect("wire-only csl idr");
        assert_eq!(ft0, FrameType::Idr);
        idr_stego = out_idr[..n0].to_vec();

        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame(WIDTH, HEIGHT, 1);
        let mut out_p = vec![0u8; 256 * 1024];
        let (ft1, n1) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out_p).expect("wire-only csl p");
        assert_eq!(ft1, FrameType::P);
        p_stego = out_p[..n1].to_vec();

        total_dispatches = *dispatches.borrow();
        actual_flips = *flips.borrow();
        apply_hits = unsafe { core_openh264_sys::phasm_diag_get_apply_hits() };

        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
    }

    eprintln!(
        "[CSL cascade-safety] CSL dispatches={} flips={} apply_hits={} | clean IDR {} B / P {} B | stego IDR {} B / P {} B",
        total_dispatches, actual_flips, apply_hits, idr_clean.len(), p_clean.len(), idr_stego.len(), p_stego.len()
    );

    // If no CSL hooks fired at all, synthetic fixture didn't produce
    // |coeff|>=16 sites — test inconclusive, lower QP further to repro.
    if total_dispatches == 0 {
        panic!("CSL dispatches=0: synthetic fixture at QP={} produced no |coeff|>=16 sites — \
                cannot exercise CoeffSuffixLsb hook. Try a lower QP or different content.", QP);
    }

    assert!(actual_flips > 0, "no CSL flips fired despite {} dispatches", total_dispatches);

    // Sanity check: CSL flips should actually reach the wire (apply_hits > 0)
    // and the IDR bytes should differ between baseline and stego.
    assert!(
        apply_hits > 0,
        "{} CSL flips dispatched to scratch but zero apply_hits — flips \
         never reached the wire. Scratch key (populate↔emit) mismatch \
         for CSL?",
        actual_flips
    );
    assert_ne!(
        idr_clean, idr_stego,
        "IDR bytes byte-identical despite {} CSL apply_hits — flips not \
         propagating to bitstream. Baseline IDR: {} bytes; stego IDR: {} bytes.",
        apply_hits, idr_clean.len(), idr_stego.len()
    );

    // The actual cascade-safety assertion: P-frame byte-identical
    // between baseline and CSL-flipped encode. If this fails, CSL
    // overrides leak into encoder reference frame.
    assert_eq!(
        p_clean, p_stego,
        "P-frame DIVERGES after {} CSL flips on IDR. CSL wire-only override \
         is leaking into encoder reference frame (pDecPic). This breaks \
         cascade-safety and is the suspected #549 root cause. \
         Baseline P: {} bytes; CSL-flipped P: {} bytes.",
        actual_flips, p_clean.len(), p_stego.len()
    );
}

/// Translated synthetic frame: keeps the same static pattern with a
/// horizontal shift proportional to frame_idx. Produces meaningful
/// inter-frame motion → encoder picks P_16x16/P_16x8/etc. with
/// non-zero MVDs, exercising HOOK-H1..H7 hooks. Required for the
/// MvdSign / MvdSuffixLsb cascade-safety tests below since
/// `synth_yuv_frame` produces uncorrelated noise per frame (encoder
/// picks SKIP or intra-in-P, no MVDs fire).
fn synth_yuv_frame_translated(width: usize, height: usize, frame_idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; width * height];
    let mut u = vec![128u8; (width / 2) * (height / 2)];
    let mut v = vec![128u8; (width / 2) * (height / 2)];
    let shift = (frame_idx as usize) * 4; // 4 px / frame horizontal shift
    // Use a static pattern (no frame_idx in the noise hash) and shift the
    // sample coordinate so the encoder sees a translating image.
    let n = |p: u8, i: usize, j: usize| -> u8 {
        (((p as usize).wrapping_add(1) ^ (i * 3) ^ (j * 5)) & 0xff) as u8
    };
    for j in 0..height {
        for i in 0..width {
            let src_i = (i + shift) % width;
            y[j * width + i] = n(0, src_i, j);
        }
    }
    for j in 0..height / 2 {
        for i in 0..width / 2 {
            let src_i = (i + shift / 2) % (width / 2);
            u[j * (width / 2) + i] = n(1, src_i, j);
            v[j * (width / 2) + i] = n(2, src_i, j);
        }
    }
    (y, u, v)
}

/// #549 Phase D test gap closure (2026-05-19): cascade-safety for
/// PHASM_DOMAIN_MVD_SIGN (dom=2). MVDs only exist in inter (P/B)
/// frames, so this test uses a 3-frame I+P1+P2 pattern: flip MvdSign
/// overrides on P1, then assert P2's bitstream is byte-identical to
/// baseline. Failure = MvdSign flips leak into P1's reference frame,
/// breaking P2's prediction.
///
/// Uses `synth_yuv_frame_translated` so the encoder sees meaningful
/// inter-frame motion and actually picks P modes with non-zero MVDs.
///
/// Background: C.8.7 (#440) originally added an MvdSign MC dual-pass
/// (cascade-break via pVisualRecPic mirroring). Phase 5 (#539) DELETED
/// that dual-pass because wire_only=1 was supposed to make it
/// unnecessary. If wire_only=1 actually leaks for MvdSign, the C.8.7
/// deletion exposed a regression — this test would catch it.
#[test]
fn pass2_replay_wire_only_forged_mvdsign_flip_cascade_safe() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 26;
    const FLIP_COUNT: u32 = 16;

    // Baseline: 3 frames, flag OFF.
    let idr_clean: Vec<u8>;
    let p1_clean: Vec<u8>;
    let p2_clean: Vec<u8>;
    {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
        let handlers = StegoHandlers::default();
        let _sess = StegoSession::register(handlers).expect("baseline register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("baseline enc");
        set_pass_mode(PassMode::Passthrough);

        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame_translated(WIDTH, HEIGHT, 0);
        let mut out0 = vec![0u8; 256 * 1024];
        let (ft0, n0) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out0).expect("baseline idr");
        assert_eq!(ft0, FrameType::Idr);
        idr_clean = out0[..n0].to_vec();

        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame_translated(WIDTH, HEIGHT, 1);
        let mut out1 = vec![0u8; 256 * 1024];
        let (ft1, n1) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out1).expect("baseline p1");
        assert_eq!(ft1, FrameType::P);
        p1_clean = out1[..n1].to_vec();

        set_frame_num(2);
        let (y2, u2, v2) = synth_yuv_frame_translated(WIDTH, HEIGHT, 2);
        let mut out2 = vec![0u8; 256 * 1024];
        let (ft2, n2) = enc.encode_frame(&y2, &u2, &v2, 2, &mut out2).expect("baseline p2");
        assert_eq!(ft2, FrameType::P);
        p2_clean = out2[..n2].to_vec();
    }

    // Stego: flag ON, flip MvdSign on P1 (frame_num=1).
    let idr_stego: Vec<u8>;
    let p1_stego: Vec<u8>;
    let p2_stego: Vec<u8>;
    let total_dispatches: u32;
    let actual_flips: u32;
    let apply_hits: u64;
    {
        unsafe {
            core_openh264_sys::phasm_diag_reset_counters();
            core_openh264_sys::phasm_set_use_wire_only_overrides(1);
        }

        let dispatches = Rc::new(RefCell::new(0u32));
        let flips = Rc::new(RefCell::new(0u32));
        let dispatches_cb = Rc::clone(&dispatches);
        let flips_cb = Rc::clone(&flips);

        let handlers = StegoHandlers {
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                // Only flip MvdSign (domain=2) on P1 (frame_num=1).
                if pos.domain != 2 || pos.frame_num != 1 {
                    return None;
                }
                *dispatches_cb.borrow_mut() += 1;
                let n = *dispatches_cb.borrow();
                if n <= FLIP_COUNT {
                    *flips_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("wire-only mvdsign register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("wire-only mvdsign enc");
        set_pass_mode(PassMode::Passthrough);

        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame_translated(WIDTH, HEIGHT, 0);
        let mut out0 = vec![0u8; 256 * 1024];
        let (ft0, n0) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out0).expect("stego idr");
        assert_eq!(ft0, FrameType::Idr);
        idr_stego = out0[..n0].to_vec();

        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame_translated(WIDTH, HEIGHT, 1);
        let mut out1 = vec![0u8; 256 * 1024];
        let (ft1, n1) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out1).expect("stego p1");
        assert_eq!(ft1, FrameType::P);
        p1_stego = out1[..n1].to_vec();

        set_frame_num(2);
        let (y2, u2, v2) = synth_yuv_frame_translated(WIDTH, HEIGHT, 2);
        let mut out2 = vec![0u8; 256 * 1024];
        let (ft2, n2) = enc.encode_frame(&y2, &u2, &v2, 2, &mut out2).expect("stego p2");
        assert_eq!(ft2, FrameType::P);
        p2_stego = out2[..n2].to_vec();

        total_dispatches = *dispatches.borrow();
        actual_flips = *flips.borrow();
        apply_hits = unsafe { core_openh264_sys::phasm_diag_get_apply_hits() };

        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
    }

    eprintln!(
        "[MvdSign cascade-safety] dispatches={} flips={} apply_hits={} | clean IDR {} / P1 {} / P2 {} | stego IDR {} / P1 {} / P2 {} bytes",
        total_dispatches, actual_flips, apply_hits,
        idr_clean.len(), p1_clean.len(), p2_clean.len(),
        idr_stego.len(), p1_stego.len(), p2_stego.len()
    );

    if total_dispatches == 0 {
        panic!("MvdSign dispatches=0: synthetic fixture produced no non-zero MVDs on P1 — \
                cannot exercise MvdSign hook. Try different synth content or motion.");
    }
    assert!(actual_flips > 0, "no MvdSign flips fired despite {} dispatches", total_dispatches);

    // IDR has no MVDs, must be byte-identical between baseline and stego.
    assert_eq!(
        idr_clean, idr_stego,
        "IDR DIFFERS between baseline and stego — but MvdSign callback \
         filter (pos.frame_num==1) means no flips should fire on IDR. \
         This indicates a leak or the callback fired on IDR unexpectedly."
    );

    // P1 should differ (the MvdSign flips landed on P1's wire).
    assert!(
        apply_hits > 0,
        "{} MvdSign flips dispatched but zero apply_hits — flips never \
         reached the wire (scratch key mismatch?).",
        actual_flips
    );
    assert_ne!(
        p1_clean, p1_stego,
        "P1 byte-identical despite {} MvdSign apply_hits — flips did not \
         propagate to bitstream.",
        apply_hits
    );

    // THE assertion: P2 must be byte-identical. If it differs, MvdSign
    // override leaked into P1's encoder reference (pDecPic), perturbing
    // P2's prediction. This would be the #549 cascade-leak root cause.
    assert_eq!(
        p2_clean, p2_stego,
        "P2 DIVERGES after {} MvdSign flips on P1. MvdSign wire-only \
         override is leaking into encoder reference frame (pDecPic) for P1, \
         breaking cascade-safety. SUSPECTED #549 ROOT CAUSE. \
         Baseline P2: {} bytes; MvdSign-flipped P2: {} bytes.",
        actual_flips, p2_clean.len(), p2_stego.len()
    );
}

/// #549 cascade-safety, MvdSuffixLsb (domain=3) companion. Same pattern
/// as the MvdSign test above — 3-frame I+P+P with translated motion so
/// the encoder generates non-zero MVDs whose suffix-LSB bins are
/// reachable. The MVD suffix bins are bypass-coded just like the sign
/// bins, so the same wire_only=1 scratch-override mechanism applies.
///
/// What this proves (if it passes): per-domain MvdSuffixLsb wire flips
/// at low count on synthetic content do not leak into the encoder
/// reference frame.
#[test]
fn pass2_replay_wire_only_forged_mvdsuffixlsb_flip_cascade_safe() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 26;
    const FLIP_COUNT: u32 = 16;

    let idr_clean: Vec<u8>;
    let p1_clean: Vec<u8>;
    let p2_clean: Vec<u8>;
    {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
        let handlers = StegoHandlers::default();
        let _sess = StegoSession::register(handlers).expect("baseline register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("baseline enc");
        set_pass_mode(PassMode::Passthrough);

        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame_translated(WIDTH, HEIGHT, 0);
        let mut out0 = vec![0u8; 256 * 1024];
        let (ft0, n0) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out0).expect("baseline idr");
        assert_eq!(ft0, FrameType::Idr);
        idr_clean = out0[..n0].to_vec();

        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame_translated(WIDTH, HEIGHT, 1);
        let mut out1 = vec![0u8; 256 * 1024];
        let (ft1, n1) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out1).expect("baseline p1");
        assert_eq!(ft1, FrameType::P);
        p1_clean = out1[..n1].to_vec();

        set_frame_num(2);
        let (y2, u2, v2) = synth_yuv_frame_translated(WIDTH, HEIGHT, 2);
        let mut out2 = vec![0u8; 256 * 1024];
        let (ft2, n2) = enc.encode_frame(&y2, &u2, &v2, 2, &mut out2).expect("baseline p2");
        assert_eq!(ft2, FrameType::P);
        p2_clean = out2[..n2].to_vec();
    }

    let idr_stego: Vec<u8>;
    let p1_stego: Vec<u8>;
    let p2_stego: Vec<u8>;
    let total_dispatches: u32;
    let actual_flips: u32;
    let apply_hits: u64;
    {
        unsafe {
            core_openh264_sys::phasm_diag_reset_counters();
            core_openh264_sys::phasm_set_use_wire_only_overrides(1);
        }

        let dispatches = Rc::new(RefCell::new(0u32));
        let flips = Rc::new(RefCell::new(0u32));
        let dispatches_cb = Rc::clone(&dispatches);
        let flips_cb = Rc::clone(&flips);

        let handlers = StegoHandlers {
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                // Only flip MvdSuffixLsb (domain=3) on P1 (frame_num=1).
                if pos.domain != 3 || pos.frame_num != 1 {
                    return None;
                }
                *dispatches_cb.borrow_mut() += 1;
                let n = *dispatches_cb.borrow();
                if n <= FLIP_COUNT {
                    *flips_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("wire-only msl register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("wire-only msl enc");
        set_pass_mode(PassMode::Passthrough);

        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame_translated(WIDTH, HEIGHT, 0);
        let mut out0 = vec![0u8; 256 * 1024];
        let (ft0, n0) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out0).expect("stego idr");
        assert_eq!(ft0, FrameType::Idr);
        idr_stego = out0[..n0].to_vec();

        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame_translated(WIDTH, HEIGHT, 1);
        let mut out1 = vec![0u8; 256 * 1024];
        let (ft1, n1) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out1).expect("stego p1");
        assert_eq!(ft1, FrameType::P);
        p1_stego = out1[..n1].to_vec();

        set_frame_num(2);
        let (y2, u2, v2) = synth_yuv_frame_translated(WIDTH, HEIGHT, 2);
        let mut out2 = vec![0u8; 256 * 1024];
        let (ft2, n2) = enc.encode_frame(&y2, &u2, &v2, 2, &mut out2).expect("stego p2");
        assert_eq!(ft2, FrameType::P);
        p2_stego = out2[..n2].to_vec();

        total_dispatches = *dispatches.borrow();
        actual_flips = *flips.borrow();
        apply_hits = unsafe { core_openh264_sys::phasm_diag_get_apply_hits() };

        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
    }

    eprintln!(
        "[MvdSuffixLsb cascade-safety] dispatches={} flips={} apply_hits={} | clean IDR {} / P1 {} / P2 {} | stego IDR {} / P1 {} / P2 {} bytes",
        total_dispatches, actual_flips, apply_hits,
        idr_clean.len(), p1_clean.len(), p2_clean.len(),
        idr_stego.len(), p1_stego.len(), p2_stego.len()
    );

    if total_dispatches == 0 {
        panic!("MvdSuffixLsb dispatches=0: synthetic fixture produced no MVDs \
                with non-trivial suffix bins on P1 — cannot exercise hook. \
                Real-content motion would generate them; consider raising \
                motion amplitude in synth_yuv_frame_translated.");
    }
    assert!(actual_flips > 0, "no MvdSuffixLsb flips fired despite {} dispatches", total_dispatches);

    assert_eq!(
        idr_clean, idr_stego,
        "IDR DIFFERS between baseline and stego — but MvdSuffixLsb \
         filter (pos.frame_num==1) means no flips should fire on IDR. \
         This indicates a leak or the callback fired on IDR unexpectedly."
    );

    assert!(
        apply_hits > 0,
        "{} MvdSuffixLsb flips dispatched but zero apply_hits — flips \
         never reached the wire (scratch key mismatch?).",
        actual_flips
    );
    assert_ne!(
        p1_clean, p1_stego,
        "P1 byte-identical despite {} MvdSuffixLsb apply_hits — flips did \
         not propagate to bitstream.",
        apply_hits
    );

    // THE assertion: P2 must be byte-identical. If it differs, the
    // MvdSuffixLsb override leaked into the encoder reference frame
    // for P1, perturbing P2's prediction.
    assert_eq!(
        p2_clean, p2_stego,
        "P2 DIVERGES after {} MvdSuffixLsb flips on P1. Wire-only \
         override is leaking into encoder reference frame (pDecPic) \
         for P1, breaking cascade-safety. SUSPECTED #549 ROOT CAUSE. \
         Baseline P2: {} bytes; MSL-flipped P2: {} bytes.",
        actual_flips, p2_clean.len(), p2_stego.len()
    );
}

/// #549 cascade-safety high-load stress test. The existing
/// `pass2_replay_wire_only_forged_flip_cascade_safe` test flips only 16
/// CoeffSign positions on an IDR with ~1376 candidate slots. The #549
/// real-content failure observed 134 overrides — almost 10× higher
/// load. If wire_only=1's encoder-recon-clean invariant degrades
/// non-linearly with flip count, low-count synthetic tests miss it.
///
/// This variant flips EVERY CoeffSign emit site on the IDR (no flip
/// cap). If P-frame still byte-identical, cascade-safety is robust;
/// if it diverges, the bug scales with override volume.
#[test]
fn pass2_replay_wire_only_forged_flip_cascade_safe_high_load() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 26;

    let idr_clean: Vec<u8>;
    let p_clean: Vec<u8>;
    {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
        let handlers = StegoHandlers::default();
        let _sess = StegoSession::register(handlers).expect("baseline register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("baseline enc");
        set_pass_mode(PassMode::Passthrough);

        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut out0 = vec![0u8; 256 * 1024];
        let (ft0, n0) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out0).expect("baseline idr");
        assert_eq!(ft0, FrameType::Idr);
        idr_clean = out0[..n0].to_vec();

        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame(WIDTH, HEIGHT, 1);
        let mut out1 = vec![0u8; 256 * 1024];
        let (ft1, n1) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out1).expect("baseline p");
        assert_eq!(ft1, FrameType::P);
        p_clean = out1[..n1].to_vec();
    }

    let idr_stego: Vec<u8>;
    let p_stego: Vec<u8>;
    let total_dispatches: u32;
    let actual_flips: u32;
    let apply_hits: u64;
    {
        unsafe {
            core_openh264_sys::phasm_diag_reset_counters();
            core_openh264_sys::phasm_set_use_wire_only_overrides(1);
        }

        let dispatches = Rc::new(RefCell::new(0u32));
        let flips = Rc::new(RefCell::new(0u32));
        let dispatches_cb = Rc::clone(&dispatches);
        let flips_cb = Rc::clone(&flips);

        // Pseudo-random per-position flip: ~50% of sites get a sign
        // inversion. Static seed for determinism.
        let handlers = StegoHandlers {
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                if pos.domain != 1 || pos.frame_num != 0 {
                    return None;
                }
                *dispatches_cb.borrow_mut() += 1;
                // Hash-mix position to get a deterministic ~50% bit.
                let h = (pos.mb_x as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add((pos.mb_y as u32).wrapping_mul(40503))
                    .wrapping_add(pos.coeff_idx as u32);
                if h & 1 == 1 {
                    *flips_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("high-load register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("high-load enc");
        set_pass_mode(PassMode::Passthrough);

        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut out0 = vec![0u8; 256 * 1024];
        let (ft0, n0) = enc.encode_frame(&y0, &u0, &v0, 0, &mut out0).expect("stego idr");
        assert_eq!(ft0, FrameType::Idr);
        idr_stego = out0[..n0].to_vec();

        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame(WIDTH, HEIGHT, 1);
        let mut out1 = vec![0u8; 256 * 1024];
        let (ft1, n1) = enc.encode_frame(&y1, &u1, &v1, 1, &mut out1).expect("stego p");
        assert_eq!(ft1, FrameType::P);
        p_stego = out1[..n1].to_vec();

        total_dispatches = *dispatches.borrow();
        actual_flips = *flips.borrow();
        apply_hits = unsafe { core_openh264_sys::phasm_diag_get_apply_hits() };

        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
    }

    eprintln!(
        "[CS HIGH-LOAD cascade-safety] dispatches={} flips={} apply_hits={} | clean IDR {} / P {} | stego IDR {} / P {} bytes",
        total_dispatches, actual_flips, apply_hits,
        idr_clean.len(), p_clean.len(),
        idr_stego.len(), p_stego.len()
    );

    assert!(total_dispatches > 100, "expected >100 dispatches, got {}", total_dispatches);
    assert!(actual_flips > 50, "expected >50 flips, got {}", actual_flips);
    assert_ne!(idr_clean, idr_stego, "IDR byte-identical despite {} flips", actual_flips);

    // THE assertion: P byte-identical even under ~50% flip density.
    assert_eq!(
        p_clean, p_stego,
        "P-frame DIVERGES after {} CoeffSign flips ({} dispatches). \
         High-load cascade-safety BROKEN — wire_only=1 leaks encoder \
         recon at scale. Implicates #549 root cause.",
        actual_flips, total_dispatches
    );
}

/// #549 multi-domain cascade-safety. Each per-domain test passes
/// individually at high flip count (CS high-load = 16k flips clean,
/// CSL/MvdSign/MSL all clean at low count). This test flips ALL FOUR
/// domains simultaneously on the IDR + P1 of a 3-frame I+P+P sequence
/// (with motion content to generate MVDs and high QP variance for CSL
/// boundaries). If P2 still byte-identical, the bug is real-content-
/// specific (1080p partition shapes / B-frame mode mix / streaming
/// orchestrator), not a wire_only=1 invariant violation.
#[test]
fn pass2_replay_wire_only_forged_4domain_combined_cascade_safe() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 22; // lower QP → more CSL boundary fires

    let encode_3 = |use_overrides: bool,
                    handlers: StegoHandlers|
     -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        unsafe {
            core_openh264_sys::phasm_set_use_wire_only_overrides(if use_overrides { 1 } else { 0 });
        }
        let _sess = StegoSession::register(handlers).expect("register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc");
        set_pass_mode(PassMode::Passthrough);

        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame_translated(WIDTH, HEIGHT, 0);
        let mut o0 = vec![0u8; 512 * 1024];
        let (ft0, n0) = enc.encode_frame(&y0, &u0, &v0, 0, &mut o0).expect("idr");
        assert_eq!(ft0, FrameType::Idr);
        let f0 = o0[..n0].to_vec();

        set_frame_num(1);
        let (y1, u1, v1) = synth_yuv_frame_translated(WIDTH, HEIGHT, 1);
        let mut o1 = vec![0u8; 512 * 1024];
        let (ft1, n1) = enc.encode_frame(&y1, &u1, &v1, 1, &mut o1).expect("p1");
        assert_eq!(ft1, FrameType::P);
        let f1 = o1[..n1].to_vec();

        set_frame_num(2);
        let (y2, u2, v2) = synth_yuv_frame_translated(WIDTH, HEIGHT, 2);
        let mut o2 = vec![0u8; 512 * 1024];
        let (ft2, n2) = enc.encode_frame(&y2, &u2, &v2, 2, &mut o2).expect("p2");
        assert_eq!(ft2, FrameType::P);
        let f2 = o2[..n2].to_vec();

        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
        (f0, f1, f2)
    };

    // Baseline: no callback, no flips.
    let (idr_clean, p1_clean, p2_clean) =
        encode_3(false, StegoHandlers::default());

    // Stego: flip ALL FOUR domains on IDR (frame=0) and P1 (frame=1).
    // P2 (frame=2) gets no flips; its bitstream must be byte-identical
    // to baseline if wire_only=1's encoder-recon-clean invariant holds.
    unsafe { core_openh264_sys::phasm_diag_reset_counters(); }

    let per_dom: [Rc<RefCell<u64>>; 4] = [
        Rc::new(RefCell::new(0)),
        Rc::new(RefCell::new(0)),
        Rc::new(RefCell::new(0)),
        Rc::new(RefCell::new(0)),
    ];
    let per_dom_cb = per_dom.clone();

    let handlers = StegoHandlers {
        enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
            // Only flip on frames 0 and 1.
            if pos.frame_num > 1 {
                return None;
            }
            if pos.domain >= 4 {
                return None;
            }
            // Hash-mix for deterministic ~50% flip density per position.
            let h = (pos.mb_x as u32)
                .wrapping_mul(2654435761)
                .wrapping_add((pos.mb_y as u32).wrapping_mul(40503))
                .wrapping_add((pos.coeff_idx as u32).wrapping_mul(83492791))
                .wrapping_add(pos.domain as u32);
            if h & 1 == 1 {
                *per_dom_cb[pos.domain as usize].borrow_mut() += 1;
                Some(1 - original)
            } else {
                None
            }
        })),
        ..Default::default()
    };

    let (idr_stego, p1_stego, p2_stego) = encode_3(true, handlers);
    let apply_hits = unsafe { core_openh264_sys::phasm_diag_get_apply_hits() };

    eprintln!(
        "[4-DOM cascade-safety] per-dom flips: CSL={} CS={} MvdSign={} MSL={} | apply_hits={}",
        *per_dom[0].borrow(),
        *per_dom[1].borrow(),
        *per_dom[2].borrow(),
        *per_dom[3].borrow(),
        apply_hits,
    );
    eprintln!(
        "  clean IDR {} / P1 {} / P2 {} | stego IDR {} / P1 {} / P2 {} bytes",
        idr_clean.len(), p1_clean.len(), p2_clean.len(),
        idr_stego.len(), p1_stego.len(), p2_stego.len(),
    );

    let total_flips: u64 = per_dom.iter().map(|c| *c.borrow()).sum();
    assert!(total_flips > 50, "expected >50 total flips, got {}", total_flips);

    assert_ne!(idr_clean, idr_stego, "IDR byte-identical despite {} flips", total_flips);
    assert_ne!(p1_clean, p1_stego, "P1 byte-identical despite {} flips", total_flips);

    // THE assertion: P2 byte-identical. No flips fire on P2, so it
    // must match baseline unless P1's flips polluted the encoder's
    // P1-reconstruction (which P2 uses as a reference).
    assert_eq!(
        p2_clean, p2_stego,
        "P2 DIVERGES after multi-domain flips on IDR+P1. \
         (CSL={}, CS={}, MvdSign={}, MSL={}). Wire-only invariant \
         broken under 4-domain combined load. SUSPECTED #549 ROOT CAUSE.",
        *per_dom[0].borrow(),
        *per_dom[1].borrow(),
        *per_dom[2].borrow(),
        *per_dom[3].borrow(),
    );
}

/// #549 root-cause localisation: Pass 1 (CAPTURE) vs Pass 2 (REPLAY)
/// byte-identity WITH bypass-bin flips active in both passes.
///
/// Existing Stage 2B test proves Pass 1 ≡ Pass 2 WITHOUT flips. The
/// production failure happens only when the orchestrator applies flips
/// in both passes. This test mirrors that pattern on synthetic content.
///
/// If Pass 1 bytes != Pass 2 bytes here, the bug reproduces on
/// synthetic and we can shrink to a unit-testable fixture. The most
/// likely root cause then: `phasm_replay_inter_override` does its own
/// MC via direct `pMcLumaFunc`, bypassing Pass 1's normal
/// `MeRefineFracPixel` path. For the same final MV they may produce
/// different prediction pixels → different residual → mb_type drift.
///
/// If Pass 1 bytes == Pass 2 bytes here, the bug is real-content-only
/// (1080p partition shapes / multi-ref / motion patterns) and the
/// next move is to reproduce + instrument at scale.
#[test]
fn pass1_capture_vs_pass2_replay_byte_identity_with_flips() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const QP: i32 = 26;
    const N_P_FRAMES: u32 = 4;

    let cache_rc = Rc::new(RefCell::new(DecisionCache::with_capacity(2048)));

    // Deterministic hash-based flip predicate — fires the SAME way in
    // both passes because it depends only on (pos.domain, pos.mb_x,
    // pos.mb_y, pos.coeff_idx, pos.frame_num).
    let should_flip = |pos: &phasm_core::codec::h264::openh264::Position| -> bool {
        // Only flip CoeffSign (most common domain).
        if pos.domain != 1 {
            return false;
        }
        let h = (pos.mb_x as u32)
            .wrapping_mul(2654435761)
            .wrapping_add((pos.mb_y as u32).wrapping_mul(40503))
            .wrapping_add((pos.coeff_idx as u32).wrapping_mul(83492791))
            .wrapping_add(pos.frame_num.wrapping_mul(2147483647));
        // ~10% flip density — sparse enough that most MBs are
        // unaffected, dense enough to seed cascade if there is one.
        (h % 10) == 0
    };

    // ---------------------------------------------------------------
    // Pass 1: CAPTURE mode for P-frames + enc_pre_emit applies flips.
    // ---------------------------------------------------------------
    unsafe {
        core_openh264_sys::phasm_diag_reset_counters();
        core_openh264_sys::phasm_set_use_wire_only_overrides(1);
    }

    let p_bytes_pass1: Vec<Vec<u8>>;
    let p1_flips_total: u32;
    {
        let cap_cache = Rc::clone(&cache_rc);
        let flip_count = Rc::new(RefCell::new(0u32));
        let flip_count_cb = Rc::clone(&flip_count);
        let handlers = StegoHandlers {
            capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
                cap_cache.borrow_mut().insert(*d);
            })),
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                if should_flip(pos) {
                    *flip_count_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-pass1");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-pass1");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut idr_out = vec![0u8; 256 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("pass1-idr");

        set_pass_mode(PassMode::Capture);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = synth_yuv_frame(WIDTH, HEIGHT, fi);
            let mut out = vec![0u8; 256 * 1024];
            let (ftype, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out)
                .expect("pass1-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_pass1 = p_bytes;
        p1_flips_total = *flip_count.borrow();
    }

    // ---------------------------------------------------------------
    // Pass 2: REPLAY mode for P-frames + enc_pre_emit applies SAME
    // flips (same predicate, deterministic).
    // ---------------------------------------------------------------
    let p_bytes_pass2: Vec<Vec<u8>>;
    let p2_flips_total: u32;
    {
        let rep_cache = Rc::clone(&cache_rc);
        let flip_count = Rc::new(RefCell::new(0u32));
        let flip_count_cb = Rc::clone(&flip_count);
        let handlers = StegoHandlers {
            replay_mb_decision: Some(Box::new(move |fnum, mx, my| {
                rep_cache.borrow().get(fnum, mx, my)
            })),
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                if should_flip(pos) {
                    *flip_count_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-pass2");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-pass2");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut idr_out = vec![0u8; 256 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("pass2-idr");

        set_pass_mode(PassMode::Replay);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = synth_yuv_frame(WIDTH, HEIGHT, fi);
            let mut out = vec![0u8; 256 * 1024];
            let (ftype, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out)
                .expect("pass2-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_pass2 = p_bytes;
        p2_flips_total = *flip_count.borrow();
    }

    unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }

    eprintln!(
        "[Pass1-vs-Pass2 byte-identity] pass1_flips={} pass2_flips={} | per-frame sizes: pass1={:?} pass2={:?}",
        p1_flips_total, p2_flips_total,
        p_bytes_pass1.iter().map(|v| v.len()).collect::<Vec<_>>(),
        p_bytes_pass2.iter().map(|v| v.len()).collect::<Vec<_>>(),
    );

    // Same predicate, same position-space → flip counts must match.
    assert_eq!(
        p1_flips_total, p2_flips_total,
        "Flip-count mismatch ({} vs {}) means the position space \
         differs between passes — pre-existing bug.",
        p1_flips_total, p2_flips_total
    );
    assert!(p1_flips_total > 0, "no flips fired in pass 1");

    // THE assertion: Pass 1 bytes == Pass 2 bytes, frame by frame.
    for (i, (p1, p2)) in p_bytes_pass1.iter().zip(p_bytes_pass2.iter()).enumerate() {
        if p1 == p2 {
            eprintln!("  P-frame {} byte-identical ({} bytes)", i + 1, p1.len());
            continue;
        }
        eprintln!(
            "  P-frame {} DIVERGENCE: pass1={}B pass2={}B",
            i + 1, p1.len(), p2.len()
        );
        for (j, (a, b)) in p1.iter().zip(p2.iter()).enumerate() {
            if a != b {
                eprintln!("    first byte diff at offset {}: pass1={:#x} pass2={:#x}", j, a, b);
                break;
            }
        }
    }
    for (i, (p1, p2)) in p_bytes_pass1.iter().zip(p_bytes_pass2.iter()).enumerate() {
        assert_eq!(
            p1, p2,
            "P-frame {} DIVERGES between Pass 1 (CAPTURE) and Pass 2 (REPLAY) \
             despite same-position deterministic flip predicate fired in both. \
             #549 ROOT-CAUSE LOCALISED to REPLAY-mode encoder state divergence.",
            i + 1
        );
    }
}

/// #549 root-cause localisation at 1080p. Same as the 320×240 variant
/// above, but at full production resolution. Synthetic content but
/// 100× more MBs per frame → exercises any scale-dependent code paths
/// (SIMD branches, threading, larger MV ranges, more partition
/// candidates) that small fixtures miss.
///
/// If 320×240 passes and 1080p fails: bug is scale-dependent.
/// If both pass: bug is real-content-feature-specific.
#[test]
fn pass1_capture_vs_pass2_replay_byte_identity_with_flips_1080p() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 1920;
    const HEIGHT: usize = 1080;
    const QP: i32 = 26;
    const N_P_FRAMES: u32 = 4;

    let cache_rc = Rc::new(RefCell::new(DecisionCache::with_capacity(32_000)));

    let should_flip = |pos: &phasm_core::codec::h264::openh264::Position| -> bool {
        if pos.domain != 1 {
            return false;
        }
        let h = (pos.mb_x as u32)
            .wrapping_mul(2654435761)
            .wrapping_add((pos.mb_y as u32).wrapping_mul(40503))
            .wrapping_add((pos.coeff_idx as u32).wrapping_mul(83492791))
            .wrapping_add(pos.frame_num.wrapping_mul(2147483647));
        (h % 100) == 0 // ~1% sparse — keep flip count manageable
    };

    unsafe {
        core_openh264_sys::phasm_diag_reset_counters();
        core_openh264_sys::phasm_set_use_wire_only_overrides(1);
    }

    let p_bytes_pass1: Vec<Vec<u8>>;
    let p1_flips_total: u32;
    {
        let cap_cache = Rc::clone(&cache_rc);
        let flip_count = Rc::new(RefCell::new(0u32));
        let flip_count_cb = Rc::clone(&flip_count);
        let handlers = StegoHandlers {
            capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
                cap_cache.borrow_mut().insert(*d);
            })),
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                if should_flip(pos) {
                    *flip_count_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-pass1-1080p");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-pass1-1080p");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut idr_out = vec![0u8; 8 * 1024 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("pass1-1080p-idr");

        set_pass_mode(PassMode::Capture);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = synth_yuv_frame(WIDTH, HEIGHT, fi);
            let mut out = vec![0u8; 8 * 1024 * 1024];
            let (ftype, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out)
                .expect("pass1-1080p-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_pass1 = p_bytes;
        p1_flips_total = *flip_count.borrow();
    }

    let p_bytes_pass2: Vec<Vec<u8>>;
    let p2_flips_total: u32;
    {
        let rep_cache = Rc::clone(&cache_rc);
        let flip_count = Rc::new(RefCell::new(0u32));
        let flip_count_cb = Rc::clone(&flip_count);
        let handlers = StegoHandlers {
            replay_mb_decision: Some(Box::new(move |fnum, mx, my| {
                rep_cache.borrow().get(fnum, mx, my)
            })),
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                if should_flip(pos) {
                    *flip_count_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-pass2-1080p");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-pass2-1080p");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = synth_yuv_frame(WIDTH, HEIGHT, 0);
        let mut idr_out = vec![0u8; 8 * 1024 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("pass2-1080p-idr");

        set_pass_mode(PassMode::Replay);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = synth_yuv_frame(WIDTH, HEIGHT, fi);
            let mut out = vec![0u8; 8 * 1024 * 1024];
            let (ftype, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out)
                .expect("pass2-1080p-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_pass2 = p_bytes;
        p2_flips_total = *flip_count.borrow();
    }

    unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }

    eprintln!(
        "[Pass1-vs-Pass2 1080p byte-identity] pass1_flips={} pass2_flips={} | per-frame sizes: pass1={:?} pass2={:?}",
        p1_flips_total, p2_flips_total,
        p_bytes_pass1.iter().map(|v| v.len()).collect::<Vec<_>>(),
        p_bytes_pass2.iter().map(|v| v.len()).collect::<Vec<_>>(),
    );

    assert_eq!(p1_flips_total, p2_flips_total,
        "1080p flip-count mismatch ({} vs {})", p1_flips_total, p2_flips_total);
    assert!(p1_flips_total > 0, "no flips fired in 1080p pass 1");

    for (i, (p1, p2)) in p_bytes_pass1.iter().zip(p_bytes_pass2.iter()).enumerate() {
        if p1 == p2 {
            eprintln!("  1080p P-frame {} byte-identical ({} bytes)", i + 1, p1.len());
            continue;
        }
        eprintln!(
            "  1080p P-frame {} DIVERGENCE: pass1={}B pass2={}B",
            i + 1, p1.len(), p2.len()
        );
        for (j, (a, b)) in p1.iter().zip(p2.iter()).enumerate() {
            if a != b {
                eprintln!("    first byte diff at offset {}: pass1={:#x} pass2={:#x}", j, a, b);
                break;
            }
        }
    }
    for (i, (p1, p2)) in p_bytes_pass1.iter().zip(p_bytes_pass2.iter()).enumerate() {
        assert_eq!(
            p1, p2,
            "1080p P-frame {} DIVERGES — scale-dependent #549 reproduction on synthetic.",
            i + 1
        );
    }
}

/// #549 root-cause localisation on REAL content. The 1080p synthetic
/// test passes — synthetic noise forces all-intra P-frames, never
/// exercising `phasm_replay_inter_override`. Real motion content
/// picks INTER MBs whose REPLAY path goes through the direct
/// `pMcLumaFunc` call (svc_base_layer_md.cpp:2086).
///
/// Reads the cached 1072×1920 × 12f carplane YUV (already populated by
/// the #549 streaming regression test) and runs the same Pass1-vs-Pass2
/// byte-identity check against the OH264 fork directly — no streaming
/// session, no 4-domain wrapper, no chunk_frame protocol. Just raw fork
/// behavior.
///
/// If this fails: bug is at the OH264 fork level for real motion.
/// If this passes: bug is in `h264_encode_gop_framed_bits_auto`
/// or the streaming session wrapper, NOT in the fork.
#[test]
fn pass1_capture_vs_pass2_replay_byte_identity_real_carplane() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 1072;
    const HEIGHT: usize = 1920;
    const QP: i32 = 26;
    const N_P_FRAMES: u32 = 11;
    const N_FRAMES: u32 = N_P_FRAMES + 1;

    let yuv_path = "/tmp/phasm_oh264_real_content_549_Artlist_CarPlane_mp4_1072x1920_f12.yuv";
    let yuv = match std::fs::read(yuv_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("real-content fixture not cached at {yuv_path}: {e} — run the #549 regression test once to populate it. Skipping.");
            return;
        }
    };
    let frame_size = WIDTH * HEIGHT * 3 / 2;
    assert!(yuv.len() >= frame_size * (N_FRAMES as usize),
        "fixture too short: {} bytes, need {}", yuv.len(), frame_size * (N_FRAMES as usize));

    let read_frame = |fi: u32| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let base = (fi as usize) * frame_size;
        let y = yuv[base..base + WIDTH * HEIGHT].to_vec();
        let uv_base = base + WIDTH * HEIGHT;
        let uv_size = (WIDTH * HEIGHT) / 4;
        let u = yuv[uv_base..uv_base + uv_size].to_vec();
        let v = yuv[uv_base + uv_size..uv_base + 2 * uv_size].to_vec();
        (y, u, v)
    };

    let cache_rc = Rc::new(RefCell::new(DecisionCache::with_capacity(32_000)));

    let should_flip = |pos: &phasm_core::codec::h264::openh264::Position| -> bool {
        if pos.domain != 1 {
            return false;
        }
        let h = (pos.mb_x as u32)
            .wrapping_mul(2654435761)
            .wrapping_add((pos.mb_y as u32).wrapping_mul(40503))
            .wrapping_add((pos.coeff_idx as u32).wrapping_mul(83492791))
            .wrapping_add(pos.frame_num.wrapping_mul(2147483647));
        (h % 100) == 0
    };

    unsafe {
        core_openh264_sys::phasm_diag_reset_counters();
        core_openh264_sys::phasm_set_use_wire_only_overrides(1);
    }

    let p_bytes_pass1: Vec<Vec<u8>>;
    let p1_flips_total: u32;
    {
        let cap_cache = Rc::clone(&cache_rc);
        let flip_count = Rc::new(RefCell::new(0u32));
        let flip_count_cb = Rc::clone(&flip_count);
        let handlers = StegoHandlers {
            capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
                cap_cache.borrow_mut().insert(*d);
            })),
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                if should_flip(pos) {
                    *flip_count_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-pass1-real");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-pass1-real");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = read_frame(0);
        let mut idr_out = vec![0u8; 8 * 1024 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("pass1-real-idr");

        set_pass_mode(PassMode::Capture);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = read_frame(fi);
            let mut out = vec![0u8; 8 * 1024 * 1024];
            let (ftype, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out)
                .expect("pass1-real-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_pass1 = p_bytes;
        p1_flips_total = *flip_count.borrow();
    }

    let p_bytes_pass2: Vec<Vec<u8>>;
    let p2_flips_total: u32;
    {
        let rep_cache = Rc::clone(&cache_rc);
        let flip_count = Rc::new(RefCell::new(0u32));
        let flip_count_cb = Rc::clone(&flip_count);
        let handlers = StegoHandlers {
            replay_mb_decision: Some(Box::new(move |fnum, mx, my| {
                rep_cache.borrow().get(fnum, mx, my)
            })),
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                if should_flip(pos) {
                    *flip_count_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-pass2-real");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-pass2-real");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = read_frame(0);
        let mut idr_out = vec![0u8; 8 * 1024 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("pass2-real-idr");

        set_pass_mode(PassMode::Replay);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = read_frame(fi);
            let mut out = vec![0u8; 8 * 1024 * 1024];
            let (ftype, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out)
                .expect("pass2-real-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_pass2 = p_bytes;
        p2_flips_total = *flip_count.borrow();
    }

    unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }

    eprintln!(
        "[Pass1-vs-Pass2 REAL carplane byte-identity] pass1_flips={} pass2_flips={} | per-frame sizes pass1={:?} pass2={:?}",
        p1_flips_total, p2_flips_total,
        p_bytes_pass1.iter().map(|v| v.len()).collect::<Vec<_>>(),
        p_bytes_pass2.iter().map(|v| v.len()).collect::<Vec<_>>(),
    );

    // Note: flip-count mismatch IS the bug signature on real content
    // (encoder emits different CS positions between passes due to mode
    // drift). Don't assert equality — just record and continue to the
    // per-frame byte comparison.
    if p1_flips_total != p2_flips_total {
        eprintln!(
            "  [signature] flip-count differs: pass1={} pass2={} (delta={}). \
             This itself confirms encoder structural divergence between passes.",
            p1_flips_total, p2_flips_total,
            (p2_flips_total as i64) - (p1_flips_total as i64),
        );
    }
    assert!(p1_flips_total > 0, "no flips fired in real-content pass 1");

    let mut first_divergent_frame: Option<usize> = None;
    for (i, (p1, p2)) in p_bytes_pass1.iter().zip(p_bytes_pass2.iter()).enumerate() {
        if p1 == p2 {
            eprintln!("  real P-frame {} byte-identical ({} bytes)", i + 1, p1.len());
            continue;
        }
        if first_divergent_frame.is_none() {
            first_divergent_frame = Some(i);
        }
        eprintln!(
            "  real P-frame {} DIVERGENCE: pass1={}B pass2={}B",
            i + 1, p1.len(), p2.len()
        );
        for (j, (a, b)) in p1.iter().zip(p2.iter()).enumerate() {
            if a != b {
                eprintln!("    first byte diff at offset {}: pass1={:#x} pass2={:#x}", j, a, b);
                break;
            }
        }
    }
    if let Some(idx) = first_divergent_frame {
        panic!("Real-content P-frame {} DIVERGES — #549 reproduced at the OH264 fork level WITHOUT streaming wrapper.",
            idx + 1);
    }
}

/// #549 small-scale repro at 480×272 × 3 frames. Same harness as the
/// 1072×1920 carplane test, but on the fast-cycle 480p downscale.
/// If this also reproduces, future bisect rounds get a ~10× speed-up.
///
/// Bug-present canary: asserts `diverged` to flag the bug. Post-Bug-5
/// fix Pass 1 ≡ Pass 2 again, so the canary trips by design. Keep
/// #[ignore] since the bug it watches for is closed; un-ignore if a
/// regression revives any of Bug 1-5.
#[test]
#[ignore = "Bug-present canary for closed #549 cluster — keep for future regression watch"]
fn pass1_capture_vs_pass2_replay_byte_identity_real_carplane_480p_minimal() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 480;
    const HEIGHT: usize = 272;
    const QP: i32 = 26;
    const N_P_FRAMES: u32 = 2; // P1 + P2 — divergence appears at P2 per the 1080p run
    const N_FRAMES: u32 = N_P_FRAMES + 1;

    let yuv_path = "/tmp/phasm_oh264_real_content_549_Artlist_CarPlane_mp4_480x272_f12.yuv";
    let yuv = match std::fs::read(yuv_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("480p fixture not cached at {yuv_path}: {e} — skipping.");
            return;
        }
    };
    let frame_size = WIDTH * HEIGHT * 3 / 2;
    assert!(yuv.len() >= frame_size * (N_FRAMES as usize),
        "fixture too short: {} bytes, need {}", yuv.len(), frame_size * (N_FRAMES as usize));

    let read_frame = |fi: u32| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let base = (fi as usize) * frame_size;
        let y = yuv[base..base + WIDTH * HEIGHT].to_vec();
        let uv_base = base + WIDTH * HEIGHT;
        let uv_size = (WIDTH * HEIGHT) / 4;
        let u = yuv[uv_base..uv_base + uv_size].to_vec();
        let v = yuv[uv_base + uv_size..uv_base + 2 * uv_size].to_vec();
        (y, u, v)
    };

    let cache_rc = Rc::new(RefCell::new(DecisionCache::with_capacity(8192)));

    let should_flip = |pos: &phasm_core::codec::h264::openh264::Position| -> bool {
        if pos.domain != 1 {
            return false;
        }
        let h = (pos.mb_x as u32)
            .wrapping_mul(2654435761)
            .wrapping_add((pos.mb_y as u32).wrapping_mul(40503))
            .wrapping_add((pos.coeff_idx as u32).wrapping_mul(83492791))
            .wrapping_add(pos.frame_num.wrapping_mul(2147483647));
        (h % 50) == 0
    };

    unsafe {
        core_openh264_sys::phasm_diag_reset_counters();
        core_openh264_sys::phasm_set_use_wire_only_overrides(1);
    }

    let p_bytes_pass1: Vec<Vec<u8>>;
    let p1_flips: u32;
    {
        let cap_cache = Rc::clone(&cache_rc);
        let flip_count = Rc::new(RefCell::new(0u32));
        let flip_count_cb = Rc::clone(&flip_count);
        let handlers = StegoHandlers {
            capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
                cap_cache.borrow_mut().insert(*d);
            })),
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                if should_flip(pos) {
                    *flip_count_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-pass1-480p");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-pass1-480p");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = read_frame(0);
        let mut idr_out = vec![0u8; 1024 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("pass1-480p-idr");

        set_pass_mode(PassMode::Capture);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = read_frame(fi);
            let mut out = vec![0u8; 1024 * 1024];
            let (ftype, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out)
                .expect("pass1-480p-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_pass1 = p_bytes;
        p1_flips = *flip_count.borrow();
    }

    let p_bytes_pass2: Vec<Vec<u8>>;
    let p2_flips: u32;
    {
        let rep_cache = Rc::clone(&cache_rc);
        let flip_count = Rc::new(RefCell::new(0u32));
        let flip_count_cb = Rc::clone(&flip_count);
        let handlers = StegoHandlers {
            replay_mb_decision: Some(Box::new(move |fnum, mx, my| {
                rep_cache.borrow().get(fnum, mx, my)
            })),
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                if should_flip(pos) {
                    *flip_count_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-pass2-480p");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-pass2-480p");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = read_frame(0);
        let mut idr_out = vec![0u8; 1024 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("pass2-480p-idr");

        set_pass_mode(PassMode::Replay);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = read_frame(fi);
            let mut out = vec![0u8; 1024 * 1024];
            let (ftype, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out)
                .expect("pass2-480p-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_pass2 = p_bytes;
        p2_flips = *flip_count.borrow();
    }

    unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }

    eprintln!(
        "[Pass1-vs-Pass2 REAL 480p minimal] pass1_flips={} pass2_flips={} delta={} | sizes pass1={:?} pass2={:?}",
        p1_flips, p2_flips, (p2_flips as i64) - (p1_flips as i64),
        p_bytes_pass1.iter().map(|v| v.len()).collect::<Vec<_>>(),
        p_bytes_pass2.iter().map(|v| v.len()).collect::<Vec<_>>(),
    );

    let mut diverged = false;
    for (i, (p1, p2)) in p_bytes_pass1.iter().zip(p_bytes_pass2.iter()).enumerate() {
        if p1 == p2 {
            eprintln!("  480p P-frame {} byte-identical ({} bytes)", i + 1, p1.len());
        } else {
            diverged = true;
            eprintln!(
                "  480p P-frame {} DIVERGENCE: pass1={}B pass2={}B",
                i + 1, p1.len(), p2.len()
            );
            for (j, (a, b)) in p1.iter().zip(p2.iter()).enumerate() {
                if a != b {
                    eprintln!("    first byte diff at offset {}: pass1={:#x} pass2={:#x}", j, a, b);
                    break;
                }
            }
        }
    }
    assert!(diverged, "480p × 2f did NOT reproduce — bug needs larger / longer fixture");
}

/// #549 hypothesis test: is REPLAY's cache-application path the bug,
/// or is it something else in REPLAY mode itself?
///
/// Same harness as the 480p minimal repro, but in Pass 2 the
/// `replay_mb_decision` callback returns None for every MB (forced
/// cache miss). The fork's REPLAY path should then fall through to
/// normal mode decision for every MB, which should produce the same
/// output as Pass 1's CAPTURE mode (same RDO, same overrides applied,
/// same input).
///
/// If Pass 1 == Pass 2 after forcing cache miss: bug is in the
/// cache-APPLICATION code (`phasm_replay_inter_override`, intra REPLAY
/// fallback, or related).
///
/// If Pass 1 != Pass 2 after forcing cache miss: bug is in the REPLAY
/// pass mode itself, NOT cache application (some other state mutation
/// the encoder does when pass_mode == REPLAY regardless of cache hits).
#[test]
fn pass1_capture_vs_pass2_replay_with_forced_cache_miss() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 480;
    const HEIGHT: usize = 272;
    const QP: i32 = 26;
    const N_P_FRAMES: u32 = 2;
    const N_FRAMES: u32 = N_P_FRAMES + 1;

    let yuv_path = "/tmp/phasm_oh264_real_content_549_Artlist_CarPlane_mp4_480x272_f12.yuv";
    let yuv = match std::fs::read(yuv_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("480p fixture not cached at {yuv_path}: {e} — skipping.");
            return;
        }
    };
    let frame_size = WIDTH * HEIGHT * 3 / 2;
    assert!(yuv.len() >= frame_size * (N_FRAMES as usize));

    let read_frame = |fi: u32| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let base = (fi as usize) * frame_size;
        let y = yuv[base..base + WIDTH * HEIGHT].to_vec();
        let uv_base = base + WIDTH * HEIGHT;
        let uv_size = (WIDTH * HEIGHT) / 4;
        let u = yuv[uv_base..uv_base + uv_size].to_vec();
        let v = yuv[uv_base + uv_size..uv_base + 2 * uv_size].to_vec();
        (y, u, v)
    };

    let should_flip = |pos: &phasm_core::codec::h264::openh264::Position| -> bool {
        if pos.domain != 1 {
            return false;
        }
        let h = (pos.mb_x as u32)
            .wrapping_mul(2654435761)
            .wrapping_add((pos.mb_y as u32).wrapping_mul(40503))
            .wrapping_add((pos.coeff_idx as u32).wrapping_mul(83492791))
            .wrapping_add(pos.frame_num.wrapping_mul(2147483647));
        (h % 50) == 0
    };

    unsafe {
        core_openh264_sys::phasm_diag_reset_counters();
        core_openh264_sys::phasm_set_use_wire_only_overrides(1);
    }

    // Pass 1: CAPTURE — same as before.
    let p_bytes_pass1: Vec<Vec<u8>>;
    {
        let handlers = StegoHandlers {
            // Capture but don't store — we only care about Pass 1 bytes here.
            capture_mb_decision: Some(Box::new(move |_d: &MbDecision| {})),
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                if should_flip(pos) { Some(1 - original) } else { None }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-pass1-miss");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-pass1-miss");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = read_frame(0);
        let mut idr_out = vec![0u8; 1024 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("pass1-miss-idr");

        set_pass_mode(PassMode::Capture);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = read_frame(fi);
            let mut out = vec![0u8; 1024 * 1024];
            let (ftype, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out).expect("pass1-miss-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_pass1 = p_bytes;
    }

    // Pass 2: REPLAY but forced cache miss on every fetch.
    let p_bytes_pass2: Vec<Vec<u8>>;
    {
        let handlers = StegoHandlers {
            // ALWAYS returns None — fork should fall through to normal RDO.
            replay_mb_decision: Some(Box::new(move |_fnum, _mx, _my| None)),
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                if should_flip(pos) { Some(1 - original) } else { None }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-pass2-miss");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-pass2-miss");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = read_frame(0);
        let mut idr_out = vec![0u8; 1024 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("pass2-miss-idr");

        set_pass_mode(PassMode::Replay);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = read_frame(fi);
            let mut out = vec![0u8; 1024 * 1024];
            let (ftype, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out).expect("pass2-miss-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_pass2 = p_bytes;
    }

    unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }

    eprintln!(
        "[Pass1-CAPTURE vs Pass2-REPLAY-FORCED-MISS] sizes pass1={:?} pass2={:?}",
        p_bytes_pass1.iter().map(|v| v.len()).collect::<Vec<_>>(),
        p_bytes_pass2.iter().map(|v| v.len()).collect::<Vec<_>>(),
    );

    let mut diverged = false;
    for (i, (p1, p2)) in p_bytes_pass1.iter().zip(p_bytes_pass2.iter()).enumerate() {
        if p1 == p2 {
            eprintln!("  forced-miss P-frame {} byte-identical ({} bytes)", i + 1, p1.len());
        } else {
            diverged = true;
            eprintln!(
                "  forced-miss P-frame {} DIVERGENCE: pass1={}B pass2={}B",
                i + 1, p1.len(), p2.len()
            );
            for (j, (a, b)) in p1.iter().zip(p2.iter()).enumerate() {
                if a != b {
                    eprintln!("    first byte diff at offset {}: pass1={:#x} pass2={:#x}", j, a, b);
                    break;
                }
            }
        }
    }

    // The diagnostic: if forced-miss MATCHES Pass 1, bug is in cache application.
    // If forced-miss STILL DIVERGES, bug is in REPLAY mode itself.
    if !diverged {
        eprintln!(
            "  DIAGNOSIS: forced-miss matches Pass 1 → bug is in REPLAY's cache application code \
            (phasm_replay_inter_override / intra fallback). Patching the cache-application path \
            should fix #549."
        );
    } else {
        eprintln!(
            "  DIAGNOSIS: forced-miss still DIVERGES from Pass 1 → bug is in REPLAY pass mode \
            ITSELF, not cache application. Some state mutation runs in REPLAY mode regardless \
            of cache hits. Look for PhasmStegoGetPassMode() == REPLAY guards beyond \
            phasm_replay_inter_override."
        );
    }
}

/// #549 NEW hypothesis (2026-05-19, post-fork-patch revert): asymmetric
/// Pass setup. The streaming orchestrator runs Pass 1 with EMPTY
/// override_map (no flips fire) and Pass 2 with POPULATED override_map
/// (flips fire at emit). My earlier Pass1-vs-Pass2 test used the SAME
/// flip predicate in both passes — symmetric setup, no repro of the
/// orchestrator's behavior.
///
/// This test mirrors the orchestrator pattern: Pass 1 fires NO flips,
/// Pass 2 fires deterministic flips. If Pass 2 diverges from Pass 1
/// MORE than just the cascade from the flipped bits, we have a synthetic
/// repro of the asymmetric-pass-setup bug.
///
/// What "more than cascade" means: bypass-bin overrides cause local
/// CABAC cascade (a few bytes shifted). But mode-decision drift, where
/// Pass 2 visits DIFFERENT MB positions / partition shapes than Pass 1,
/// shows up as the walker on Pass 2 finding +N cover positions Pass 1
/// didn't have. That's the #549 fingerprint.
#[test]
fn pass1_no_flips_vs_pass2_with_flips_real_carplane_480p() {
    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 480;
    const HEIGHT: usize = 272;
    const QP: i32 = 26;
    const N_P_FRAMES: u32 = 2;
    const N_FRAMES: u32 = N_P_FRAMES + 1;

    let yuv_path = "/tmp/phasm_oh264_real_content_549_Artlist_CarPlane_mp4_480x272_f12.yuv";
    let yuv = match std::fs::read(yuv_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("480p fixture not cached at {yuv_path}: {e} — skipping.");
            return;
        }
    };
    let frame_size = WIDTH * HEIGHT * 3 / 2;
    assert!(yuv.len() >= frame_size * (N_FRAMES as usize));

    let read_frame = |fi: u32| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let base = (fi as usize) * frame_size;
        let y = yuv[base..base + WIDTH * HEIGHT].to_vec();
        let uv_base = base + WIDTH * HEIGHT;
        let uv_size = (WIDTH * HEIGHT) / 4;
        let u = yuv[uv_base..uv_base + uv_size].to_vec();
        let v = yuv[uv_base + uv_size..uv_base + 2 * uv_size].to_vec();
        (y, u, v)
    };

    let should_flip = |pos: &phasm_core::codec::h264::openh264::Position| -> bool {
        if pos.domain != 1 { return false; }
        let h = (pos.mb_x as u32)
            .wrapping_mul(2654435761)
            .wrapping_add((pos.mb_y as u32).wrapping_mul(40503))
            .wrapping_add((pos.coeff_idx as u32).wrapping_mul(83492791))
            .wrapping_add(pos.frame_num.wrapping_mul(2147483647));
        (h % 50) == 0
    };

    let cache_rc = Rc::new(RefCell::new(DecisionCache::with_capacity(8192)));

    unsafe {
        core_openh264_sys::phasm_diag_reset_counters();
        core_openh264_sys::phasm_set_use_wire_only_overrides(1);
    }

    // Pass 1: CAPTURE, NO flips (orchestrator pattern).
    let p_bytes_pass1: Vec<Vec<u8>>;
    {
        let cap_cache = Rc::clone(&cache_rc);
        let handlers = StegoHandlers {
            capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
                cap_cache.borrow_mut().insert(*d);
            })),
            enc_pre_emit: Some(Box::new(move |_pos, _original| -> Option<i32> {
                None // NO flips in Pass 1
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-pass1-noflip");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-pass1-noflip");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = read_frame(0);
        let mut idr_out = vec![0u8; 1024 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("pass1-noflip-idr");

        set_pass_mode(PassMode::Capture);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = read_frame(fi);
            let mut out = vec![0u8; 1024 * 1024];
            let (ftype, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out).expect("pass1-noflip-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_pass1 = p_bytes;
    }

    // Pass 2: REPLAY, WITH flips (orchestrator pattern).
    let p_bytes_pass2: Vec<Vec<u8>>;
    let pass2_flips: u32;
    {
        let rep_cache = Rc::clone(&cache_rc);
        let flips = Rc::new(RefCell::new(0u32));
        let flips_cb = Rc::clone(&flips);
        let handlers = StegoHandlers {
            replay_mb_decision: Some(Box::new(move |fnum, mx, my| {
                rep_cache.borrow().get(fnum, mx, my)
            })),
            enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
                if should_flip(pos) {
                    *flips_cb.borrow_mut() += 1;
                    Some(1 - original)
                } else {
                    None
                }
            })),
            ..Default::default()
        };
        let _sess = StegoSession::register(handlers).expect("register-pass2-flip");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc-pass2-flip");

        set_pass_mode(PassMode::Passthrough);
        set_frame_num(0);
        let (y0, u0, v0) = read_frame(0);
        let mut idr_out = vec![0u8; 1024 * 1024];
        enc.encode_frame(&y0, &u0, &v0, 0, &mut idr_out).expect("pass2-flip-idr");

        set_pass_mode(PassMode::Replay);
        let mut p_bytes: Vec<Vec<u8>> = Vec::new();
        for fi in 1..=N_P_FRAMES {
            set_frame_num(fi);
            let (y, u, v) = read_frame(fi);
            let mut out = vec![0u8; 1024 * 1024];
            let (ftype, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out).expect("pass2-flip-p");
            assert_eq!(ftype, FrameType::P);
            p_bytes.push(out[..n].to_vec());
        }
        set_pass_mode(PassMode::Passthrough);
        p_bytes_pass2 = p_bytes;
        pass2_flips = *flips.borrow();
    }

    unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }

    eprintln!(
        "[asymmetric Pass 1(no-flips)/Pass 2(with {} flips)] sizes pass1={:?} pass2={:?}",
        pass2_flips,
        p_bytes_pass1.iter().map(|v| v.len()).collect::<Vec<_>>(),
        p_bytes_pass2.iter().map(|v| v.len()).collect::<Vec<_>>(),
    );

    // Pass 1 has no flips, Pass 2 has flips. Bytes WILL differ in cascade
    // from flips. The interesting question is: by HOW MUCH?
    // Acceptable: O(pass2_flips) bytes diff = local CABAC cascade.
    // Pathological: significantly more than that = mode-decision drift.
    for (i, (p1, p2)) in p_bytes_pass1.iter().zip(p_bytes_pass2.iter()).enumerate() {
        let len_diff = (p2.len() as i64) - (p1.len() as i64);
        if p1 == p2 {
            eprintln!("  P-frame {} byte-identical despite {} flips (must have all hit positions with same bit value)", i + 1, pass2_flips);
        } else {
            eprintln!(
                "  P-frame {} differs: pass1={}B pass2={}B (delta={:+}B) — expected ~{} from cascade",
                i + 1, p1.len(), p2.len(), len_diff, pass2_flips,
            );
        }
    }
    // No assertion — this is purely diagnostic. The OUTPUT shows whether
    // the bug exists on synthetic content with the asymmetric setup.
}

/// #549 closed-loop walker test: encode Pass 1 → walk → flip ONE bit →
/// encode Pass 2 → walk → assert walker sees the flipped bit at the
/// expected position. Mirrors the orchestrator's flow in miniature.
///
/// If walker sees the flip: encoder + walker are symmetric, orchestrator
/// failure is at higher layer (plan derivation, STC, chunk_frame).
/// If walker doesn't see the flip: encoder/walker asymmetry — fork bug.
#[test]
fn pass2_walker_sees_single_cs_flip_real_carplane_480p() {
    use phasm_core::codec::h264::openh264::extract_cover_bits_via_decoder;

    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 480;
    const HEIGHT: usize = 272;
    const QP: i32 = 26;
    const N_P_FRAMES: u32 = 2;
    const N_FRAMES: u32 = N_P_FRAMES + 1;

    let yuv_path = "/tmp/phasm_oh264_real_content_549_Artlist_CarPlane_mp4_480x272_f12.yuv";
    let yuv = match std::fs::read(yuv_path) {
        Ok(d) => d,
        Err(e) => { eprintln!("fixture missing: {e} — skipping."); return; }
    };
    let frame_size = WIDTH * HEIGHT * 3 / 2;
    if yuv.len() < frame_size * (N_FRAMES as usize) {
        eprintln!("fixture too short — skipping");
        return;
    }

    let read_frame = |fi: u32| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let base = (fi as usize) * frame_size;
        let y = yuv[base..base + WIDTH * HEIGHT].to_vec();
        let uv_base = base + WIDTH * HEIGHT;
        let uv_size = (WIDTH * HEIGHT) / 4;
        let u = yuv[uv_base..uv_base + uv_size].to_vec();
        let v = yuv[uv_base + uv_size..uv_base + 2 * uv_size].to_vec();
        (y, u, v)
    };

    let cache_rc = Rc::new(RefCell::new(DecisionCache::with_capacity(8192)));

    // Helper: encode 3 frames in CAPTURE-then-REPLAY pattern, with
    // optional flip predicate. Returns full Annex-B (IDR + 2 P-frames).
    let encode_with = |pass: PassMode,
                       handlers: StegoHandlers|
     -> Vec<u8> {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(1); }
        let _sess = StegoSession::register(handlers).expect("register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc");
        set_pass_mode(PassMode::Passthrough);
        let mut all = Vec::new();
        for fi in 0..N_FRAMES {
            set_frame_num(fi);
            if fi == 1 { set_pass_mode(pass); }
            let (y, u, v) = read_frame(fi);
            let mut out = vec![0u8; 1024 * 1024];
            let (_, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out).expect("encode");
            all.extend_from_slice(&out[..n]);
        }
        set_pass_mode(PassMode::Passthrough);
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
        all
    };

    // Pass 1: CAPTURE, no flips.
    unsafe { core_openh264_sys::phasm_diag_reset_counters(); }
    let cap_cache = Rc::clone(&cache_rc);
    let pass1_handlers = StegoHandlers {
        capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
            cap_cache.borrow_mut().insert(*d);
        })),
        ..Default::default()
    };
    let pass1_bytes = encode_with(PassMode::Capture, pass1_handlers);
    let p1_set_writes = unsafe { core_openh264_sys::phasm_diag_get_set_writes() };
    let p1_set_calls  = unsafe { core_openh264_sys::phasm_diag_get_set_calls() };
    let p1_apply_hits = unsafe { core_openh264_sys::phasm_diag_get_apply_hits() };
    let p1_apply_calls= unsafe { core_openh264_sys::phasm_diag_get_apply_calls() };
    eprintln!("[closed-loop] pass1 diag: set_calls={p1_set_calls} set_writes={p1_set_writes} apply_calls={p1_apply_calls} apply_hits={p1_apply_hits}");
    let cover_p1 = extract_cover_bits_via_decoder(&pass1_bytes)
        .expect("walk pass 1");

    eprintln!("[closed-loop] pass1 walk: CS={} CSL={} MvdSign={} MSL={}",
        cover_p1.coeff_sign_bypass.len(),
        cover_p1.coeff_suffix_lsb.len(),
        cover_p1.mvd_sign_bypass.len(),
        cover_p1.mvd_suffix_lsb.len(),
    );

    // Build override map: flip CS bit at index 100 (middle of the array).
    let flip_idx = (cover_p1.coeff_sign_bypass.len() / 2).min(100);
    if cover_p1.coeff_sign_bypass.is_empty() {
        eprintln!("no CS positions in pass1 walk — fixture too small. Skipping.");
        return;
    }
    let original_bit = cover_p1.coeff_sign_bypass[flip_idx];
    eprintln!("[closed-loop] planning flip at CS index {flip_idx}: original={original_bit} → {}", 1 - original_bit);

    // Pass 2: REPLAY, with a single deterministic flip on the Nth CS fire.
    let rep_cache = Rc::clone(&cache_rc);
    let cs_counter = Rc::new(RefCell::new(0usize));
    let cs_counter_cb = Rc::clone(&cs_counter);
    let pass2_handlers = StegoHandlers {
        replay_mb_decision: Some(Box::new(move |fnum, mx, my| {
            rep_cache.borrow().get(fnum, mx, my)
        })),
        enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
            if pos.domain != 1 { return None; }
            let idx = *cs_counter_cb.borrow();
            *cs_counter_cb.borrow_mut() += 1;
            if idx == flip_idx {
                Some(1 - original)
            } else {
                None
            }
        })),
        ..Default::default()
    };
    unsafe { core_openh264_sys::phasm_diag_reset_counters(); }
    let pass2_bytes = encode_with(PassMode::Replay, pass2_handlers);
    let p2_set_writes = unsafe { core_openh264_sys::phasm_diag_get_set_writes() };
    let p2_set_calls  = unsafe { core_openh264_sys::phasm_diag_get_set_calls() };
    let p2_apply_hits = unsafe { core_openh264_sys::phasm_diag_get_apply_hits() };
    let p2_apply_calls= unsafe { core_openh264_sys::phasm_diag_get_apply_calls() };
    let cs_counter_final = *cs_counter.borrow();
    eprintln!("[closed-loop] pass2 diag: set_calls={p2_set_calls} set_writes={p2_set_writes} apply_calls={p2_apply_calls} apply_hits={p2_apply_hits} cs_counter={cs_counter_final}");
    let cover_p2 = extract_cover_bits_via_decoder(&pass2_bytes)
        .expect("walk pass 2");

    eprintln!("[closed-loop] pass2 walk: CS={} CSL={} MvdSign={} MSL={}",
        cover_p2.coeff_sign_bypass.len(),
        cover_p2.coeff_suffix_lsb.len(),
        cover_p2.mvd_sign_bypass.len(),
        cover_p2.mvd_suffix_lsb.len(),
    );

    // Count walker-side diffs.
    if cover_p1.coeff_sign_bypass.len() != cover_p2.coeff_sign_bypass.len() {
        eprintln!(
            "[closed-loop] CS LENGTH MISMATCH between pass1 and pass2 walks: {} vs {} — encoder visited DIFFERENT position set",
            cover_p1.coeff_sign_bypass.len(),
            cover_p2.coeff_sign_bypass.len(),
        );
    }
    let n = cover_p1.coeff_sign_bypass.len().min(cover_p2.coeff_sign_bypass.len());
    let mut walker_diffs = 0usize;
    let mut first_diff: Option<usize> = None;
    for i in 0..n {
        if cover_p1.coeff_sign_bypass[i] != cover_p2.coeff_sign_bypass[i] {
            walker_diffs += 1;
            if first_diff.is_none() { first_diff = Some(i); }
        }
    }
    eprintln!(
        "[closed-loop] CS walker diffs pass1 vs pass2: {} of {} (first at idx {:?})",
        walker_diffs, n, first_diff,
    );

    // The crux assertion: walker should see EXACTLY 1 diff at flip_idx
    // (the position we flipped), and 0 diffs elsewhere.
    if walker_diffs == 1 && first_diff == Some(flip_idx) {
        eprintln!("[closed-loop] WALKER SYMMETRIC ✓ — encoder + walker agree on flip");
    } else if walker_diffs == 0 {
        eprintln!("[closed-loop] WALKER MISSED FLIP — flip didn't reach the wire or was reverted");
    } else {
        eprintln!(
            "[closed-loop] WALKER SAW MULTIPLE DIFFS ({} diffs, first at {:?}, planned flip at {}) — \
             single flip caused walker to see multiple bit changes. This is the orchestrator bug.",
            walker_diffs, first_diff, flip_idx,
        );
    }
}

/// #549 closed-loop walker test, MULTI-FLIP variant. Mirrors the orchestrator
/// pattern with N flips (N=139 on real carplane). Confirms whether the
/// `phasm_replay_inter_override` patch holds at production flip count.
///
/// If walker sees exactly N diffs at the flipped indices: patch is sufficient,
/// orchestrator bug is in higher-level code (STC, chunk_frame, mux).
/// If walker sees >> N diffs: patch insufficient at scale, there's another
/// asymmetric-pass bug.
#[test]
fn pass2_walker_sees_N_cs_flips_real_carplane_480p() {
    use phasm_core::codec::h264::openh264::extract_cover_bits_via_decoder;

    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 480;
    const HEIGHT: usize = 272;
    const QP: i32 = 26;
    const N_P_FRAMES: u32 = 11;
    const N_FRAMES: u32 = N_P_FRAMES + 1;
    const FLIP_EVERY: usize = 10; // higher density: every 10th CS fire

    let yuv_path = "/tmp/phasm_oh264_real_content_549_Artlist_CarPlane_mp4_480x272_f12.yuv";
    let yuv = match std::fs::read(yuv_path) {
        Ok(d) => d,
        Err(e) => { eprintln!("fixture missing: {e} — skipping."); return; }
    };
    let frame_size = WIDTH * HEIGHT * 3 / 2;
    if yuv.len() < frame_size * (N_FRAMES as usize) {
        eprintln!("fixture too short — skipping"); return;
    }
    let read_frame = |fi: u32| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let base = (fi as usize) * frame_size;
        let y = yuv[base..base + WIDTH * HEIGHT].to_vec();
        let uv_base = base + WIDTH * HEIGHT;
        let uv_size = (WIDTH * HEIGHT) / 4;
        let u = yuv[uv_base..uv_base + uv_size].to_vec();
        let v = yuv[uv_base + uv_size..uv_base + 2 * uv_size].to_vec();
        (y, u, v)
    };

    let cache_rc = Rc::new(RefCell::new(DecisionCache::with_capacity(8192)));

    let encode_with = |pass: PassMode, handlers: StegoHandlers| -> Vec<u8> {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(1); }
        let _sess = StegoSession::register(handlers).expect("register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc");
        set_pass_mode(PassMode::Passthrough);
        let mut all = Vec::new();
        for fi in 0..N_FRAMES {
            set_frame_num(fi);
            if fi == 1 { set_pass_mode(pass); }
            let (y, u, v) = read_frame(fi);
            let mut out = vec![0u8; 1024 * 1024];
            let (_, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out).expect("encode");
            all.extend_from_slice(&out[..n]);
        }
        set_pass_mode(PassMode::Passthrough);
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
        all
    };

    // Pass 1: CAPTURE, no flips
    let cap_cache = Rc::clone(&cache_rc);
    let p1_handlers = StegoHandlers {
        capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
            cap_cache.borrow_mut().insert(*d);
        })),
        ..Default::default()
    };
    let p1_bytes = encode_with(PassMode::Capture, p1_handlers);
    let cover_p1 = extract_cover_bits_via_decoder(&p1_bytes).expect("walk p1");
    eprintln!("[N-flip] pass1 walk: CS={}", cover_p1.coeff_sign_bypass.len());

    // Pass 2: REPLAY, every Nth CS gets flipped
    let rep_cache = Rc::clone(&cache_rc);
    let cs_counter = Rc::new(RefCell::new(0usize));
    let n_flipped = Rc::new(RefCell::new(0usize));
    let cs_counter_cb = Rc::clone(&cs_counter);
    let n_flipped_cb = Rc::clone(&n_flipped);
    let p2_handlers = StegoHandlers {
        replay_mb_decision: Some(Box::new(move |fnum, mx, my| {
            rep_cache.borrow().get(fnum, mx, my)
        })),
        enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
            if pos.domain != 1 { return None; }
            let idx = *cs_counter_cb.borrow();
            *cs_counter_cb.borrow_mut() += 1;
            if idx % FLIP_EVERY == 0 && idx > 0 {
                *n_flipped_cb.borrow_mut() += 1;
                Some(1 - original)
            } else {
                None
            }
        })),
        ..Default::default()
    };
    unsafe { core_openh264_sys::phasm_diag_reset_counters(); }
    let p2_bytes = encode_with(PassMode::Replay, p2_handlers);
    let p2_set_writes = unsafe { core_openh264_sys::phasm_diag_get_set_writes() };
    let p2_apply_hits = unsafe { core_openh264_sys::phasm_diag_get_apply_hits() };
    let n_flips = *n_flipped.borrow();
    eprintln!("[N-flip] pass2: set_writes={p2_set_writes} apply_hits={p2_apply_hits} planned_flips={n_flips}");
    let cover_p2 = extract_cover_bits_via_decoder(&p2_bytes).expect("walk p2");
    eprintln!("[N-flip] pass2 walk: CS={} CSL={} MvdSign={} MSL={}",
        cover_p2.coeff_sign_bypass.len(),
        cover_p2.coeff_suffix_lsb.len(),
        cover_p2.mvd_sign_bypass.len(),
        cover_p2.mvd_suffix_lsb.len(),
    );

    if cover_p1.coeff_sign_bypass.len() != cover_p2.coeff_sign_bypass.len() {
        eprintln!("[N-flip] CS LENGTH MISMATCH: {} vs {} — encoder visited DIFFERENT positions in pass2",
            cover_p1.coeff_sign_bypass.len(), cover_p2.coeff_sign_bypass.len());
    }
    let n = cover_p1.coeff_sign_bypass.len().min(cover_p2.coeff_sign_bypass.len());
    let mut walker_diffs = 0;
    for i in 0..n {
        if cover_p1.coeff_sign_bypass[i] != cover_p2.coeff_sign_bypass[i] {
            walker_diffs += 1;
        }
    }
    eprintln!("[N-flip] CS walker diffs: {} (planned flips: {}, ratio: {:.2}x)",
        walker_diffs, n_flips,
        if n_flips > 0 { walker_diffs as f64 / n_flips as f64 } else { 0.0 });
    if walker_diffs <= n_flips + 5 {
        eprintln!("[N-flip] WALKER ROUGHLY SYMMETRIC — patch holds at N={n_flips} flips ✓");
    } else {
        eprintln!("[N-flip] WALKER CASCADE — {} diffs from {} flips. Patch INSUFFICIENT at scale.",
            walker_diffs, n_flips);
    }
}

/// #549 Bug 3 isolation: extend closed-loop walker test to flip ALL FOUR
/// domains (CS, CSL, MvdSign, MvdSuffixLsb) at the same density. If this
/// shows walker drift on the CS axis when MVD domains are flipped, the
/// bug is fork-level MVD cascade. If it stays symmetric, the bug is in
/// the orchestrator wrapper code, not the fork.
#[test]
fn pass2_walker_sees_N_4domain_flips_real_carplane_480p() {
    use phasm_core::codec::h264::openh264::extract_cover_bits_via_decoder;

    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 480;
    const HEIGHT: usize = 272;
    const QP: i32 = 26;
    const N_P_FRAMES: u32 = 11;
    const N_FRAMES: u32 = N_P_FRAMES + 1;
    const FLIP_EVERY: usize = 50;

    let yuv_path = "/tmp/phasm_oh264_real_content_549_Artlist_CarPlane_mp4_480x272_f12.yuv";
    let yuv = match std::fs::read(yuv_path) {
        Ok(d) => d,
        Err(e) => { eprintln!("fixture missing: {e} — skipping."); return; }
    };
    let frame_size = WIDTH * HEIGHT * 3 / 2;
    if yuv.len() < frame_size * (N_FRAMES as usize) {
        eprintln!("fixture too short — skipping"); return;
    }
    let read_frame = |fi: u32| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let base = (fi as usize) * frame_size;
        let y = yuv[base..base + WIDTH * HEIGHT].to_vec();
        let uv_base = base + WIDTH * HEIGHT;
        let uv_size = (WIDTH * HEIGHT) / 4;
        let u = yuv[uv_base..uv_base + uv_size].to_vec();
        let v = yuv[uv_base + uv_size..uv_base + 2 * uv_size].to_vec();
        (y, u, v)
    };

    let cache_rc = Rc::new(RefCell::new(DecisionCache::with_capacity(8192)));

    let encode_with = |pass: PassMode, handlers: StegoHandlers| -> Vec<u8> {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(1); }
        let _sess = StegoSession::register(handlers).expect("register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc");
        set_pass_mode(PassMode::Passthrough);
        let mut all = Vec::new();
        for fi in 0..N_FRAMES {
            set_frame_num(fi);
            if fi == 1 { set_pass_mode(pass); }
            let (y, u, v) = read_frame(fi);
            let mut out = vec![0u8; 1024 * 1024];
            let (_, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out).expect("encode");
            all.extend_from_slice(&out[..n]);
        }
        set_pass_mode(PassMode::Passthrough);
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
        all
    };

    let cap_cache = Rc::clone(&cache_rc);
    let p1_handlers = StegoHandlers {
        capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
            cap_cache.borrow_mut().insert(*d);
        })),
        ..Default::default()
    };
    let p1_bytes = encode_with(PassMode::Capture, p1_handlers);
    let cover_p1 = extract_cover_bits_via_decoder(&p1_bytes).expect("walk p1");
    eprintln!("[4-dom N-flip] pass1 walk: CS={} CSL={} MvdSign={} MSL={}",
        cover_p1.coeff_sign_bypass.len(),
        cover_p1.coeff_suffix_lsb.len(),
        cover_p1.mvd_sign_bypass.len(),
        cover_p1.mvd_suffix_lsb.len(),
    );

    // Per-domain counter; flip every FLIP_EVERY-th occurrence per domain.
    let rep_cache = Rc::clone(&cache_rc);
    let per_dom_counter = Rc::new(RefCell::new([0usize; 4]));
    let per_dom_flips = Rc::new(RefCell::new([0usize; 4]));
    let per_dom_counter_cb = Rc::clone(&per_dom_counter);
    let per_dom_flips_cb = Rc::clone(&per_dom_flips);
    let p2_handlers = StegoHandlers {
        replay_mb_decision: Some(Box::new(move |fnum, mx, my| {
            rep_cache.borrow().get(fnum, mx, my)
        })),
        enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
            let d = pos.domain as usize;
            if d >= 4 { return None; }
            let idx = per_dom_counter_cb.borrow()[d];
            per_dom_counter_cb.borrow_mut()[d] += 1;
            if idx % FLIP_EVERY == 0 && idx > 0 {
                per_dom_flips_cb.borrow_mut()[d] += 1;
                Some(1 - original)
            } else {
                None
            }
        })),
        ..Default::default()
    };
    let p2_bytes = encode_with(PassMode::Replay, p2_handlers);
    let cover_p2 = extract_cover_bits_via_decoder(&p2_bytes).expect("walk p2");
    let flips = *per_dom_flips.borrow();
    eprintln!("[4-dom N-flip] pass2 flips per-domain: CSL={} CS={} MvdSign={} MSL={} | total={}",
        flips[0], flips[1], flips[2], flips[3], flips.iter().sum::<usize>());
    eprintln!("[4-dom N-flip] pass2 walk: CS={} CSL={} MvdSign={} MSL={}",
        cover_p2.coeff_sign_bypass.len(),
        cover_p2.coeff_suffix_lsb.len(),
        cover_p2.mvd_sign_bypass.len(),
        cover_p2.mvd_suffix_lsb.len(),
    );

    // Compare counts per domain
    let cs_delta = cover_p2.coeff_sign_bypass.len() as i64 - cover_p1.coeff_sign_bypass.len() as i64;
    let csl_delta = cover_p2.coeff_suffix_lsb.len() as i64 - cover_p1.coeff_suffix_lsb.len() as i64;
    let mvds_delta = cover_p2.mvd_sign_bypass.len() as i64 - cover_p1.mvd_sign_bypass.len() as i64;
    let msl_delta = cover_p2.mvd_suffix_lsb.len() as i64 - cover_p1.mvd_suffix_lsb.len() as i64;
    eprintln!("[4-dom N-flip] domain count delta p2-p1: CS={cs_delta:+} CSL={csl_delta:+} MvdSign={mvds_delta:+} MSL={msl_delta:+}");

    if cs_delta == 0 && csl_delta == 0 && mvds_delta == 0 && msl_delta == 0 {
        eprintln!("[4-dom N-flip] STRUCTURALLY SYMMETRIC — encoder visited same positions in both passes ✓");
    } else {
        eprintln!("[4-dom N-flip] STRUCTURAL DRIFT — encoder visited DIFFERENT positions. \
            This is the orchestrator gate failure mode reproduced on the closed-loop test.");
    }

    // Walker diff count on CS only (most numerous)
    let n_cs = cover_p1.coeff_sign_bypass.len().min(cover_p2.coeff_sign_bypass.len());
    let mut cs_walker_diffs = 0;
    for i in 0..n_cs {
        if cover_p1.coeff_sign_bypass[i] != cover_p2.coeff_sign_bypass[i] {
            cs_walker_diffs += 1;
        }
    }
    eprintln!("[4-dom N-flip] CS walker diffs (p1 vs p2): {} of {} (CS-domain flips: {})",
        cs_walker_diffs, n_cs, flips[1]);
}

/// #549 Bug 3 per-domain isolation: flip ONLY MvdSign (dom=2). If this
/// causes CS positions to drift in Pass 2 vs Pass 1, MvdSign is the
/// cascade source. (CS-only N-flip test already showed zero CS drift
/// up to 11k flips — so any drift seen here is MvdSign-caused.)
#[test]
fn pass2_walker_sees_mvdsign_only_real_carplane_480p() {
    use phasm_core::codec::h264::openh264::extract_cover_bits_via_decoder;

    let _lock = SESSION_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const WIDTH: usize = 480;
    const HEIGHT: usize = 272;
    const QP: i32 = 26;
    const N_P_FRAMES: u32 = 11;
    const N_FRAMES: u32 = N_P_FRAMES + 1;

    let yuv_path = "/tmp/phasm_oh264_real_content_549_Artlist_CarPlane_mp4_480x272_f12.yuv";
    let yuv = match std::fs::read(yuv_path) {
        Ok(d) => d,
        Err(e) => { eprintln!("fixture missing: {e} — skipping."); return; }
    };
    let frame_size = WIDTH * HEIGHT * 3 / 2;
    if yuv.len() < frame_size * (N_FRAMES as usize) {
        eprintln!("fixture too short — skipping"); return;
    }
    let read_frame = |fi: u32| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let base = (fi as usize) * frame_size;
        let y = yuv[base..base + WIDTH * HEIGHT].to_vec();
        let uv_base = base + WIDTH * HEIGHT;
        let uv_size = (WIDTH * HEIGHT) / 4;
        let u = yuv[uv_base..uv_base + uv_size].to_vec();
        let v = yuv[uv_base + uv_size..uv_base + 2 * uv_size].to_vec();
        (y, u, v)
    };

    let cache_rc = Rc::new(RefCell::new(DecisionCache::with_capacity(8192)));
    let encode_with = |pass: PassMode, handlers: StegoHandlers| -> Vec<u8> {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(1); }
        let _sess = StegoSession::register(handlers).expect("register");
        let mut enc = Encoder::new(WIDTH as i32, HEIGHT as i32, QP, 60).expect("enc");
        set_pass_mode(PassMode::Passthrough);
        let mut all = Vec::new();
        for fi in 0..N_FRAMES {
            set_frame_num(fi);
            if fi == 1 { set_pass_mode(pass); }
            let (y, u, v) = read_frame(fi);
            let mut out = vec![0u8; 1024 * 1024];
            let (_, n) = enc.encode_frame(&y, &u, &v, fi as i64, &mut out).expect("encode");
            all.extend_from_slice(&out[..n]);
        }
        set_pass_mode(PassMode::Passthrough);
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0); }
        all
    };

    // Pass 1: CAPTURE no flips
    let cap_cache = Rc::clone(&cache_rc);
    let p1_handlers = StegoHandlers {
        capture_mb_decision: Some(Box::new(move |d: &MbDecision| {
            cap_cache.borrow_mut().insert(*d);
        })),
        ..Default::default()
    };
    let p1_bytes = encode_with(PassMode::Capture, p1_handlers);
    let cover_p1 = extract_cover_bits_via_decoder(&p1_bytes).expect("walk p1");

    // Pass 2: REPLAY with ONLY MvdSign (dom=2) flips, every Nth
    let rep_cache = Rc::clone(&cache_rc);
    let mvd_counter = Rc::new(RefCell::new(0usize));
    let mvd_flips = Rc::new(RefCell::new(0usize));
    let mvd_counter_cb = Rc::clone(&mvd_counter);
    let mvd_flips_cb = Rc::clone(&mvd_flips);
    let p2_handlers = StegoHandlers {
        replay_mb_decision: Some(Box::new(move |fnum, mx, my| {
            rep_cache.borrow().get(fnum, mx, my)
        })),
        enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
            if pos.domain != 2 { return None; } // 2=MvdSign
            let idx = *mvd_counter_cb.borrow();
            *mvd_counter_cb.borrow_mut() += 1;
            if idx % 50 == 0 && idx > 0 {
                *mvd_flips_cb.borrow_mut() += 1;
                Some(1 - original)
            } else {
                None
            }
        })),
        ..Default::default()
    };
    let p2_bytes = encode_with(PassMode::Replay, p2_handlers);
    let cover_p2 = extract_cover_bits_via_decoder(&p2_bytes).expect("walk p2");
    let n_flips = *mvd_flips.borrow();

    eprintln!("[MvdSign-only] p1: CS={} CSL={} MvdSign={} MSL={}",
        cover_p1.coeff_sign_bypass.len(), cover_p1.coeff_suffix_lsb.len(),
        cover_p1.mvd_sign_bypass.len(), cover_p1.mvd_suffix_lsb.len());
    eprintln!("[MvdSign-only] p2: CS={} CSL={} MvdSign={} MSL={} | flipped: {} MvdSign positions",
        cover_p2.coeff_sign_bypass.len(), cover_p2.coeff_suffix_lsb.len(),
        cover_p2.mvd_sign_bypass.len(), cover_p2.mvd_suffix_lsb.len(),
        n_flips);

    let cs_delta = cover_p2.coeff_sign_bypass.len() as i64 - cover_p1.coeff_sign_bypass.len() as i64;
    let csl_delta = cover_p2.coeff_suffix_lsb.len() as i64 - cover_p1.coeff_suffix_lsb.len() as i64;
    let mvds_delta = cover_p2.mvd_sign_bypass.len() as i64 - cover_p1.mvd_sign_bypass.len() as i64;
    let msl_delta = cover_p2.mvd_suffix_lsb.len() as i64 - cover_p1.mvd_suffix_lsb.len() as i64;
    eprintln!("[MvdSign-only] domain count delta p2-p1: CS={cs_delta:+} CSL={csl_delta:+} MvdSign={mvds_delta:+} MSL={msl_delta:+}");

    if cs_delta == 0 && csl_delta == 0 {
        eprintln!("[MvdSign-only] CS+CSL DOMAIN COUNTS UNCHANGED — MvdSign flips do NOT cascade into coefficient mode-decision ✓");
    } else {
        eprintln!("[MvdSign-only] CS+CSL DRIFT — MvdSign flips DO cascade. Confirmed Bug 3 source.");
    }
}





