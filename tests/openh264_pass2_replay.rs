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

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

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
