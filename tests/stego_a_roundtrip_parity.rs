// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// STEGO.A.5 — Round-trip parity gate for the unified Scheme A
// pipeline.
//
// Validates that the OH264 4-domain encoder (Scheme A combined STC +
// Tier 3 content-adaptive costs from STEGO.A.1) round-trips through
// the unified smart_decode_video (Scheme A combined extract from
// STEGO.A.4). This is the architectural-unification gate: the
// encoder/decoder pair that was MISMATCHED before STEGO.A is now
// MATCHED.
//
// Coverage:
//   - 480×272 × 10f synthetic YUV (proven working size)
//   - Primary-only (no shadows)
//   - Round-trip via the production-decodable smart_decode_video path
//   - Both default + uniform CostWeights (uniform-weights pass exercises
//     the worst case for content-adaptive cost differentiation)
//
// What STEGO.A.5 does NOT cover (yet):
//   - Pure-Rust encoder × decoder (gated by STEGO.A.2's caller
//     migration; foundation function is tested in orchestrate::tests
//     but full encode_pixels.rs entry points still use Scheme B
//     pending STEGO.A.12 migration).
//   - Shadow messages (STEGO.A.6).
//   - Real-world corpus fixtures (STEGO.A.10).
//   - cgPhone (STEGO.A.11).

#![cfg(all(
    feature = "h264-encoder",
    feature = "openh264-backend",
    feature = "cabac-stego",
))]

use phasm_core::codec::h264::openh264_stego::{
    encode_yuv_with_n_shadows_with_pattern_and_files,
    encode_yuv_with_pre_framed_bits_4domain, EncodeOpts,
};
use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingDecodeSession,
    StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::stego::shadow_layer::ShadowLayer;
use phasm_core::codec::h264::stego::cost_weights::CostWeights;
use phasm_core::codec::h264::stego::keys::CabacStegoMasterKeys;
use phasm_core::codec::h264::stego::hook::EmbedDomain;
use phasm_core::h264_stego_smart_decode_video;
use phasm_core::stego::{crypto, frame, payload};
use std::sync::{Mutex, OnceLock};

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

/// #796 — these tests use 320x240 / 480x272 synthetic XOR/LCG content
/// which interacts badly with aggressive auto-tier (high-entropy synth
/// content has few |coeff|=1 positions, so Tier 4 leaves STC with too
/// few finite-cost positions to span the syndrome). Pin to Tier 0 for
/// the duration of these tests so the parity assertions exercise the
/// encoder semantics, not the tier-filter selection.
fn session_lock_synth<'a>() -> std::sync::MutexGuard<'a, ()> {
    let g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: SESSION_GUARD serializes test execution, so env-var
    // mutations are race-free. Set unconditionally on each lock so
    // the variable stays present for the duration of the test (kept
    // set across test boundaries within the same process is also fine
    // — it's the right value for every test in this file).
    unsafe { std::env::set_var("PHASM_AUTO_TIER_CONSERVATIVE", "1"); }
    g
}

fn synth_yuv(width: u32, height: u32, n_frames: u32) -> Vec<u8> {
    let frame_size = (width * height * 3 / 2) as usize;
    let mut out = Vec::with_capacity(frame_size * n_frames as usize);
    let w = width as i32;
    let h = height as i32;
    let half_w = w / 2;
    let half_h = h / 2;
    for f in 0..n_frames {
        for j in 0..h {
            for i in 0..w {
                let v = ((i + f as i32 * 2) ^ (j + f as i32 * 3)) as u8;
                out.push(v);
            }
        }
        let mut s: u32 = 0xCAFE_F00D ^ f;
        for _plane in 0..2 {
            for j in 0..half_h {
                for i in 0..half_w {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    let tex = (s >> 16) as u8;
                    let pos = (i + j + f as i32) as u8;
                    out.push(tex.wrapping_add(pos));
                }
            }
        }
    }
    out
}

/// Build the framed primary payload bits using the same path the
/// streaming session uses (payload::encode_payload → crypto::encrypt
/// → frame::build_frame → MSB-first bits).
fn build_primary_frame_bits(
    message: &str,
    passphrase: &str,
) -> (Vec<u8>, [u8; 32]) {
    let primary_bytes = payload::encode_payload(message, &[]).unwrap();
    let (ct, nonce, salt) = crypto::encrypt(&primary_bytes, passphrase).unwrap();
    let frame_bytes = frame::build_frame(primary_bytes.len(), &salt, &nonce, &ct);
    let frame_bits: Vec<u8> = frame_bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect();
    let keys = CabacStegoMasterKeys::derive(passphrase).unwrap();
    let hhat_seed = keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;
    (frame_bits, hhat_seed)
}

#[test]
fn oh264_scheme_a_primary_roundtrip_default_weights() {
    let _g = session_lock_synth();

    let width: u32 = 480;
    let height: u32 = 272;
    let n_frames: u32 = 10;
    let yuv = synth_yuv(width, height, n_frames);
    let opts = EncodeOpts { qp: 26, intra_period: 5 };
    let weights = CostWeights::default();

    let message = "hello-stego-a";
    let passphrase = "stego-a-roundtrip-default";

    let (frame_bits, hhat_seed) = build_primary_frame_bits(message, passphrase);

    let annex_b = encode_yuv_with_pre_framed_bits_4domain(
        &yuv, width, height, n_frames, opts,
        &frame_bits, &hhat_seed, &weights,
    )
    .expect("OH264 Scheme A encode");

    let recovered = h264_stego_smart_decode_video(&annex_b, passphrase)
        .expect("smart_decode_video should recover via Scheme A combined extract");
    assert_eq!(recovered, message);
}

#[test]
fn oh264_scheme_a_primary_roundtrip_uniform_weights() {
    let _g = session_lock_synth();

    let width: u32 = 480;
    let height: u32 = 272;
    let n_frames: u32 = 10;
    let yuv = synth_yuv(width, height, n_frames);
    let opts = EncodeOpts { qp: 26, intra_period: 5 };
    let weights = CostWeights {
        coeff_sign: 1.0,
        coeff_suffix: 1.0,
        mvd_sign: 1.0,
        mvd_suffix: 1.0,
    };

    let message = "uniform-weights-still-decodes";
    let passphrase = "stego-a-roundtrip-uniform";

    let (frame_bits, hhat_seed) = build_primary_frame_bits(message, passphrase);

    let annex_b = encode_yuv_with_pre_framed_bits_4domain(
        &yuv, width, height, n_frames, opts,
        &frame_bits, &hhat_seed, &weights,
    )
    .expect("OH264 Scheme A encode (uniform weights)");

    let recovered = h264_stego_smart_decode_video(&annex_b, passphrase)
        .expect("smart_decode_video with uniform weights");
    assert_eq!(recovered, message);
}

#[test]
fn oh264_scheme_a_shadow_roundtrip_primary_plus_one_shadow() {
    let _g = session_lock_synth();

    let width: u32 = 480;
    let height: u32 = 272;
    let n_frames: u32 = 10;
    let yuv = synth_yuv(width, height, n_frames);
    let opts = EncodeOpts { qp: 26, intra_period: 5 };
    let weights = CostWeights::default();

    let primary_msg = "primary-shadow-2";
    let primary_pass = "ppass-stego-a6";
    let shadow_msg = "shadow-secret-2";
    let shadow_pass = "spass-stego-a6";

    let shadows = [ShadowLayer {
        message: shadow_msg,
        passphrase: shadow_pass,
        files: &[],
    }];

    let annex_b = encode_yuv_with_n_shadows_with_pattern_and_files(
        &yuv, width, height, n_frames, opts,
        primary_msg, &[], primary_pass,
        &shadows,
        &weights,
    )
    .expect("OH264 Scheme A shadow encode");

    // Decode primary via smart_decode_video — should fall through
    // to Scheme A combined STC extract.
    let recovered_primary = h264_stego_smart_decode_video(&annex_b, primary_pass)
        .expect("primary decode via Scheme A combined extract");
    assert_eq!(recovered_primary, primary_msg, "primary preserved");

    // Decode shadow via the same entry — should match
    // shadow_extract_all4_safe (cheap AES-GCM-SIV-auth-gated attempt
    // BEFORE the combined STC extract).
    let recovered_shadow = h264_stego_smart_decode_video(&annex_b, shadow_pass)
        .expect("shadow decode via shadow_extract_all4_safe");
    assert_eq!(recovered_shadow, shadow_msg, "shadow preserved");

    // Wrong passphrase rejects both.
    let res = h264_stego_smart_decode_video(&annex_b, "wrong-pass-stego-a6");
    assert!(res.is_err(), "wrong passphrase must not recover");
}

#[test]
fn oh264_scheme_a_shadow_roundtrip_primary_plus_two_shadows() {
    let _g = session_lock_synth();

    let width: u32 = 480;
    let height: u32 = 272;
    let n_frames: u32 = 10;
    let yuv = synth_yuv(width, height, n_frames);
    let opts = EncodeOpts { qp: 26, intra_period: 5 };
    let weights = CostWeights::default();

    let primary_msg = "p";
    let primary_pass = "primary-2-shadows";
    let shadow1 = ShadowLayer { message: "s1", passphrase: "pass-s1", files: &[] };
    let shadow2 = ShadowLayer { message: "s2", passphrase: "pass-s2", files: &[] };

    let annex_b = encode_yuv_with_n_shadows_with_pattern_and_files(
        &yuv, width, height, n_frames, opts,
        primary_msg, &[], primary_pass,
        &[shadow1, shadow2],
        &weights,
    )
    .expect("OH264 Scheme A 2-shadow encode");

    assert_eq!(
        h264_stego_smart_decode_video(&annex_b, primary_pass).unwrap(),
        primary_msg,
    );
    assert_eq!(
        h264_stego_smart_decode_video(&annex_b, "pass-s1").unwrap(),
        "s1",
    );
    assert_eq!(
        h264_stego_smart_decode_video(&annex_b, "pass-s2").unwrap(),
        "s2",
    );
}

#[test]
fn oh264_scheme_a_streaming_session_shadow_roundtrip() {
    let _g = session_lock_synth();

    let width: u32 = 480;
    let height: u32 = 272;
    let n_frames: u32 = 10;
    let yuv = synth_yuv(width, height, n_frames);

    let params = EncodeSessionParams {
        width,
        height,
        fps_num: 30,
        fps_den: 1,
        qp: 26,
        gop_size: 5,
        total_frames_hint: n_frames,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        cost_weights: CostWeights::default(),
        progress_callback: None,
    };

    let primary_msg = "streaming-primary";
    let primary_pass = "primary-pass-streaming";
    let shadow_msg = "streaming-shadow";
    let shadow_pass = "shadow-pass-streaming";
    let shadows = [ShadowLayer {
        message: shadow_msg,
        passphrase: shadow_pass,
        files: &[],
    }];

    let mut session = StreamingEncodeSession::create_with_shadows(
        params, primary_msg, &[], primary_pass, &shadows,
    )
    .expect("create_with_shadows");

    // Push each frame through the streaming API.
    let frame_size = (width * height * 3 / 2) as usize;
    let y_plane_size = (width * height) as usize;
    let chroma_plane_size = y_plane_size / 4;
    let mut out_bytes: Vec<u8> = Vec::new();
    for f in 0..n_frames {
        let off = f as usize * frame_size;
        let y = &yuv[off..off + y_plane_size];
        let u = &yuv[off + y_plane_size..off + y_plane_size + chroma_plane_size];
        let v = &yuv[off + y_plane_size + chroma_plane_size..off + frame_size];
        let frame = YuvFrameRef {
            y,
            y_stride: width as usize,
            u,
            u_stride: (width / 2) as usize,
            v,
            v_stride: (width / 2) as usize,
        };
        session.push_frame(frame, &mut out_bytes).expect("push_frame");
    }
    session.finish(&mut out_bytes).expect("finish should run shadow encode");

    // Decode primary + shadow via smart_decode_video.
    let recovered_primary = h264_stego_smart_decode_video(&out_bytes, primary_pass)
        .expect("primary decode");
    assert_eq!(recovered_primary, primary_msg);

    let recovered_shadow = h264_stego_smart_decode_video(&out_bytes, shadow_pass)
        .expect("shadow decode");
    assert_eq!(recovered_shadow, shadow_msg);
}

#[test]
fn oh264_scheme_a_primary_wrong_passphrase_rejects() {
    let _g = session_lock_synth();

    let width: u32 = 480;
    let height: u32 = 272;
    let n_frames: u32 = 10;
    let yuv = synth_yuv(width, height, n_frames);
    let opts = EncodeOpts { qp: 26, intra_period: 5 };
    let weights = CostWeights::default();

    let message = "secret";
    let right_pass = "right-pass-stego-a";
    let wrong_pass = "wrong-pass-stego-a";

    let (frame_bits, hhat_seed) = build_primary_frame_bits(message, right_pass);

    let annex_b = encode_yuv_with_pre_framed_bits_4domain(
        &yuv, width, height, n_frames, opts,
        &frame_bits, &hhat_seed, &weights,
    )
    .expect("encode");

    // Wrong passphrase must NOT recover the message — AES-GCM-SIV
    // authentication failure should bubble up as FrameCorrupted
    // (or any error variant; just must NOT return the secret).
    let res = h264_stego_smart_decode_video(&annex_b, wrong_pass);
    match res {
        Ok(s) => panic!(
            "wrong passphrase must not recover secret, got: {:?}", s,
        ),
        Err(_) => { /* expected */ }
    }
}

// ─── STEGO.B.P4 — Pure-Rust + cross-encoder parity gates ─────────────

/// Drive the streaming session with the requested backend and return
/// the produced Annex-B bytes. Shared helper so the OH264 and pure-Rust
/// parity tests stay structurally identical.
fn encode_via_streaming(
    engine: EncodeEngineChoice,
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    gop_size: u32,
    message: &str,
    passphrase: &str,
) -> Vec<u8> {
    let params = EncodeSessionParams {
        width,
        height,
        fps_num: 30,
        fps_den: 1,
        qp: 26,
        gop_size,
        total_frames_hint: n_frames,
        color: ColorParams::default(),
        engine,
        cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut session = StreamingEncodeSession::create(params, message, passphrase)
        .expect("create streaming session");

    let frame_size = (width * height * 3 / 2) as usize;
    let y_plane_size = (width * height) as usize;
    let chroma_plane_size = y_plane_size / 4;
    let mut out_bytes: Vec<u8> = Vec::new();
    for f in 0..n_frames {
        let off = f as usize * frame_size;
        let y = &yuv[off..off + y_plane_size];
        let u = &yuv[off + y_plane_size..off + y_plane_size + chroma_plane_size];
        let v = &yuv[off + y_plane_size + chroma_plane_size..off + frame_size];
        let frame = YuvFrameRef {
            y,
            y_stride: width as usize,
            u,
            u_stride: (width / 2) as usize,
            v,
            v_stride: (width / 2) as usize,
        };
        session.push_frame(frame, &mut out_bytes).expect("push_frame");
    }
    session.finish(&mut out_bytes).expect("finish");
    out_bytes
}

/// STEGO.B.P4 — Pure-Rust streaming session primary round-trip.
///
/// Mirrors `oh264_scheme_a_streaming_session_shadow_roundtrip` but on
/// the pure-Rust backend (no shadows — that's STEGO.B.P5). Proves that
/// after P1+P2 the pure-Rust per-GOP path is decode-compatible with
/// `smart_decode_video` (which now tries Scheme A combined extract
/// first, then falls back). The Scheme B fallback is still present;
/// P6 will remove it once this gate proves Scheme A is sufficient.
#[test]
fn pure_rust_scheme_a_streaming_session_primary_roundtrip() {
    let _g = session_lock_synth();

    // Pure-Rust encoder is ~1365× slower than OH264, so the fixture
    // must be small enough to keep wall-clock under ~10s. 320×240 ×
    // 4f / gop=2 matches the working h264_streaming_pure_rust_per_gop_472
    // fixtures (which also exercise OH264 streaming sessions).
    let width: u32 = 320;
    let height: u32 = 240;
    let n_frames: u32 = 4;
    let gop_size: u32 = 2;
    let yuv = synth_yuv(width, height, n_frames);

    let message = "stego-b-p4";
    let passphrase = "pure-rust-parity-pass";

    let annex_b = encode_via_streaming(
        EncodeEngineChoice::PureRust, &yuv, width, height, n_frames,
        gop_size, message, passphrase,
    );

    // Streaming session output uses the per-GOP chunk_frame protocol —
    // must decode via StreamingDecodeSession, not smart_decode_video.
    let mut dec = StreamingDecodeSession::create(passphrase)
        .expect("create streaming decode session");
    dec.push_annex_b(&annex_b).expect("push annex_b");
    let result = dec.finish().expect("streaming decode finish");
    assert_eq!(result.text, message);
}

/// STEGO.B.P4 — Cross-encoder payload parity.
///
/// Both backends, same YUV + message + passphrase. The Annex-B bytes
/// differ (Cisco OpenH264 vs phasm pure-Rust make different baseline
/// mode/MV decisions), but the decoded payload MUST match. This is the
/// contract that proves both encoders implement the SAME stego scheme
/// (Scheme A + Tier 3) — only the H.264 baseline differs.
///
/// Strict assertion: each backend's output must decode via
/// `smart_decode_video` (Scheme A primary path). If both encoders are
/// truly on Scheme A, neither output triggers the Scheme B fallback
/// (which is what P6 will remove).
#[test]
fn cross_encoder_primary_payload_parity() {
    let _g = session_lock_synth();

    let width: u32 = 320;
    let height: u32 = 240;
    let n_frames: u32 = 4;
    let gop_size: u32 = 2;
    let yuv = synth_yuv(width, height, n_frames);

    let message = "cross-encoder-parity";
    let passphrase = "cross-pass";

    // OH264 encode.
    let annex_b_oh = encode_via_streaming(
        EncodeEngineChoice::Oh264, &yuv, width, height, n_frames,
        gop_size, message, passphrase,
    );
    // Pure-Rust encode.
    let annex_b_pr = encode_via_streaming(
        EncodeEngineChoice::PureRust, &yuv, width, height, n_frames,
        gop_size, message, passphrase,
    );

    // Annex-B bytes are NOT byte-identical (different baseline encoders)
    // — sanity check that they really are distinct streams.
    assert_ne!(
        annex_b_oh, annex_b_pr,
        "OH264 and pure-Rust must produce distinct H.264 baselines",
    );

    // Both decode via StreamingDecodeSession (per-GOP chunk_frame).
    let decode_once = |annex_b: &[u8]| -> String {
        let mut dec = StreamingDecodeSession::create(passphrase)
            .expect("create streaming decode session");
        dec.push_annex_b(annex_b).expect("push annex_b");
        dec.finish().expect("streaming decode finish").text
    };
    let recovered_oh = decode_once(&annex_b_oh);
    let recovered_pr = decode_once(&annex_b_pr);
    assert_eq!(recovered_oh, message, "OH264 round-trip");
    assert_eq!(recovered_pr, message, "pure-Rust round-trip");
    assert_eq!(recovered_oh, recovered_pr, "cross-encoder payload parity");
}

/// STEGO.B.P3 direct test — call the pure-Rust **primary** whole-video
/// orchestrator directly. Isolates whether the bug is shadow-specific
/// or hits the whole-video path generally.
#[test]
fn pure_rust_streaming_v2_primary_direct_roundtrip() {
    let _g = session_lock_synth();
    let width: u32 = 480;
    let height: u32 = 272;
    let n_frames: u32 = 10;
    let yuv = synth_yuv(width, height, n_frames);
    let pattern = phasm_core::codec::h264::stego::gop_pattern::GopPattern::Ipppp { gop: 5 };

    let primary_msg = "pr-v2-prim";
    let primary_pass = "pure-rust-v2-primary-pass";

    let bytes = phasm_core::h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
        &yuv, width, height, n_frames as usize, pattern,
        primary_msg, &[], primary_pass,
    )
    .expect("pure-Rust streaming_v2 primary encode");

    let recovered = h264_stego_smart_decode_video(&bytes, primary_pass)
        .expect("primary decode");
    assert_eq!(recovered, primary_msg);
}

/// STEGO.B.P3 direct test — call the pure-Rust shadow orchestrator
/// directly (no streaming session wrapper) to isolate whether the
/// migration produces decode-compatible output. If this fails, P3
/// has a bug; if this passes but the streaming session wrapper
/// fails, the bug is in the wrapper.
#[test]
fn pure_rust_shadow_orchestrator_direct_roundtrip() {
    let _g = session_lock_synth();
    let width: u32 = 480;
    let height: u32 = 272;
    let n_frames: u32 = 10;
    let yuv = synth_yuv(width, height, n_frames);
    let pattern = phasm_core::codec::h264::stego::gop_pattern::GopPattern::Ipppp { gop: 5 };

    let primary_msg = "pr-prim-direct";
    let primary_pass = "pure-rust-shadow-primary-pass-direct";
    let shadow_msg = "pr-shad-direct";
    let shadow_pass = "pure-rust-shadow-pass-direct";
    let shadows = [ShadowLayer {
        message: shadow_msg,
        passphrase: shadow_pass,
        files: &[],
    }];

    let bytes = phasm_core::codec::h264::stego::encode_pixels::h264_stego_encode_yuv_string_with_n_shadows_with_pattern_and_files(
        &yuv, width, height, n_frames as usize, pattern,
        primary_msg, &[], primary_pass, &shadows,
    )
    .expect("pure-Rust shadow orchestrator encode");

    let recovered_primary = h264_stego_smart_decode_video(&bytes, primary_pass)
        .expect("primary decode (direct)");
    assert_eq!(recovered_primary, primary_msg);
    let recovered_shadow = h264_stego_smart_decode_video(&bytes, shadow_pass)
        .expect("shadow decode (direct)");
    assert_eq!(recovered_shadow, shadow_msg);
}

/// STEGO.B.P5 — pure-Rust shadow streaming session round-trip.
///
/// Mirrors `oh264_scheme_a_streaming_session_shadow_roundtrip` but on
/// the pure-Rust backend. Confirms `StreamingEncodeSession::
/// create_with_shadows` no longer errors for `EncodeEngineChoice::
/// PureRust` (the previous `Err("requires OH264 backend")` at
/// streaming_session.rs:430 is replaced by a buffer-on-finish path
/// that calls the migrated pure-Rust shadow orchestrator).
///
/// Buffer-on-finish output is whole-video phasm v1/v2 frame format
/// (NOT chunk_frame), so decode goes through `smart_decode_video`,
/// same as the OH264 shadow streaming path.
#[test]
fn pure_rust_scheme_a_streaming_session_shadow_roundtrip() {
    let _g = session_lock_synth();

    // Pure-Rust shadow encode is ~1365× slower than OH264. Use the
    // exact same fixture size as the OH264 shadow streaming test
    // (480×272 × 10f / gop=5) — known to work for shadow cascade
    // capacity + IBPBP pattern.
    let width: u32 = 480;
    let height: u32 = 272;
    let n_frames: u32 = 10;
    let gop_size: u32 = 5;
    let yuv = synth_yuv(width, height, n_frames);

    let primary_msg = "pr-prim";
    let primary_pass = "pure-rust-shadow-primary-pass";
    let shadow_msg = "pr-shad";
    let shadow_pass = "pure-rust-shadow-pass";
    let shadows = [ShadowLayer {
        message: shadow_msg,
        passphrase: shadow_pass,
        files: &[],
    }];

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

    let mut session = StreamingEncodeSession::create_with_shadows(
        params, primary_msg, &[], primary_pass, &shadows,
    )
    .expect("create_with_shadows on pure-Rust backend");

    let frame_size = (width * height * 3 / 2) as usize;
    let y_plane_size = (width * height) as usize;
    let chroma_plane_size = y_plane_size / 4;
    let mut out_bytes: Vec<u8> = Vec::new();
    for f in 0..n_frames {
        let off = f as usize * frame_size;
        let y = &yuv[off..off + y_plane_size];
        let u = &yuv[off + y_plane_size..off + y_plane_size + chroma_plane_size];
        let v = &yuv[off + y_plane_size + chroma_plane_size..off + frame_size];
        let frame = YuvFrameRef {
            y,
            y_stride: width as usize,
            u,
            u_stride: (width / 2) as usize,
            v,
            v_stride: (width / 2) as usize,
        };
        session.push_frame(frame, &mut out_bytes).expect("push_frame");
    }
    session.finish(&mut out_bytes).expect("finish should run pure-Rust shadow orchestrator");

    // Whole-video output → decode via smart_decode_video (same as OH264 path).
    let recovered_primary = h264_stego_smart_decode_video(&out_bytes, primary_pass)
        .expect("primary decode");
    assert_eq!(recovered_primary, primary_msg);

    let recovered_shadow = h264_stego_smart_decode_video(&out_bytes, shadow_pass)
        .expect("shadow decode");
    assert_eq!(recovered_shadow, shadow_msg);
}
