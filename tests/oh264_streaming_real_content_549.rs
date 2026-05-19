// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #549 — real-content streaming-session round-trip ship gate.
//!
//! This is the gate that #541 / Phase 7 should have caught and didn't.
//!
//! The synthetic-YUV lib test `oh264_streaming_530_repro_1072x1920_3gop`
//! passes with `plan vs walker diffs=0` because pure XOR noise rarely
//! triggers the encoder's INTER→INTRA mode-decision crossover under stego
//! coefficient flips. Real motion content (carplane) DOES trigger it:
//! Pass 1 picks P_inter for most MBs, Pass 2 sees flipped-sign coefficients
//! that push the inter cost above the intra threshold, and chooses
//! Intra-in-P for some MBs. Result: Pass 2 emits a different mb_type
//! structure than Pass 1 captured → walker reads different cover bits
//! than the STC plan flipped → decode finds zero chunk_idx matches.
//!
//! Phase A: lock the regression. Phase B: bisect. Phase C: identify
//! missing DecisionCache coverage. Phase D: fix.
//!
//! The test consumes the cached corpus YUV at
//! `/tmp/phasm_oh264_corpus_carplane_1080p_1072x1920_f30.yuv` (or the
//! Artlist_CarPlane.mp4 source it derives from). Encode via
//! `StreamingEncodeSession` (4-domain, OH264 default, wire_only=1 by
//! default — same path the CLI takes), then `StreamingDecodeSession`.
//!
//! Variants for Phase B bisect:
//!   * `_30f` — full 30 frame × 1 GOP fixture (matches CLI repro)
//!   * `_12f` — minimum repro candidate, 12 frame × 1 GOP
//!   * `_6f`  — even smaller
//!   * `_wire_only_off` — same content with PHASM_USE_WIRE_ONLY=0
//!   * `_iphone7_30f` — second real-content fixture

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingDecodeSession,
    StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::h264::stego::CostWeights;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

/// Demux a corpus MP4 to YUV via ffmpeg at the requested resolution +
/// frame count. Caches to /tmp so re-runs are cheap.
fn ensure_real_yuv(source_mp4: &str, w: u32, h: u32, n_frames: u32) -> Vec<u8> {
    let yuv_path = format!(
        "/tmp/phasm_oh264_real_content_549_{}_{}x{}_f{}.yuv",
        source_mp4.replace(['/', '.'], "_"),
        w, h, n_frames
    );
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * (n_frames as usize);
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need {
            return data[..need].to_vec();
        }
    }
    let src = corpus_root().join(source_mp4);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!("scale={w}:{h}");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &n_frames.to_string()])
        .args(["-an", "-vf", &vf])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("ffmpeg launch");
    assert!(status.success(), "ffmpeg scale failed for {source_mp4}");
    let data = std::fs::read(&yuv_path).expect("read yuv");
    assert!(
        data.len() >= need,
        "ffmpeg produced {} bytes, need {} ({}x{} x 1.5 x {})",
        data.len(), need, w, h, n_frames,
    );
    data[..need].to_vec()
}

struct RealContentSpec {
    name: &'static str,
    source_mp4: &'static str,
    width: u32,
    height: u32,
    n_frames: u32,
    gop_size: u32,
    msg: &'static str,
    pass: &'static str,
    cost_weights: CostWeights,
}

fn run_real_content_streaming(spec: &RealContentSpec) {
    // Recover the mutex even if a sibling test poisoned it — we still
    // want each variant's failure reported individually.
    let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());

    let yuv_bytes = ensure_real_yuv(spec.source_mp4, spec.width, spec.height, spec.n_frames);

    let params = EncodeSessionParams {
        width: spec.width,
        height: spec.height,
        fps_num: 30,
        fps_den: 1,
        qp: 26,
        gop_size: spec.gop_size,
        total_frames_hint: spec.n_frames,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        cost_weights: spec.cost_weights,
    };
    let mut enc =
        StreamingEncodeSession::create(params, spec.msg, spec.pass).expect("encode session");

    let frame_bytes = (spec.width as usize) * (spec.height as usize) * 3 / 2;
    let chroma_w = (spec.width as usize) / 2;
    let chroma_h = (spec.height as usize) / 2;
    let y_plane_size = (spec.width as usize) * (spec.height as usize);
    let chroma_plane_size = chroma_w * chroma_h;

    let mut annex_b = Vec::new();
    for f in 0..spec.n_frames {
        let off = (f as usize) * frame_bytes;
        let y = &yuv_bytes[off..off + y_plane_size];
        let u = &yuv_bytes[off + y_plane_size..off + y_plane_size + chroma_plane_size];
        let v = &yuv_bytes[off + y_plane_size + chroma_plane_size
            ..off + y_plane_size + 2 * chroma_plane_size];
        let frame = YuvFrameRef {
            y, y_stride: spec.width as usize,
            u, u_stride: chroma_w,
            v, v_stride: chroma_w,
        };
        enc.push_frame(frame, &mut annex_b).expect("push frame");
    }
    enc.finish(&mut annex_b).expect("finish encode");
    assert!(!annex_b.is_empty(), "{}: empty Annex-B output", spec.name);

    let mut dec =
        StreamingDecodeSession::create(spec.pass).expect("decode session");
    dec.push_annex_b(&annex_b).expect("push annex-b");
    let result = dec.finish().unwrap_or_else(|e| {
        panic!(
            "{}: DECODE FAILED on real-content YUV — this is the #549 regression. \
             Underlying error: {e:?}. \
             Inspect the encode trace with PHASM_PERF_TRACE=1 for `plan vs walker diffs` \
             at the first divergent MB; expect drift starting at the first P-frame (frame=2 \
             in encode order) when stego coefficient flips push some inter MBs into intra fallback.",
            spec.name,
        )
    });
    assert_eq!(
        result.text, spec.msg,
        "{}: message mismatch — recovered {:?} vs expected {:?}",
        spec.name, result.text, spec.msg,
    );
    assert_eq!(result.mode_id, 1, "{}: mode_id should be 1 (Ghost)", spec.name);

    eprintln!(
        "#549 real-content GREEN: {} {}x{} × {}f × GOP={} ({} annex_b bytes)",
        spec.name, spec.width, spec.height, spec.n_frames, spec.gop_size, annex_b.len(),
    );
}

/// PRIMARY SHIP GATE (production resolution + length).
///
/// Real-content carplane (high motion, real coefficient distribution),
/// 1072×1920 × 30f × 1 GOP. Matches CLI repro at gop_size=30. Green
/// post-2026-05-19 (#549 Bug 5 closed). Stays #[ignore] to keep
/// default cargo-test wall-clock short; the lighter 480p_12f variant
/// below is un-ignored as the default-suite gate.
#[test]
#[ignore = "Heavy ship-gate; default suite runs the 480p_12f variant"]
fn oh264_streaming_real_carplane_1080p_30f() {
    run_real_content_streaming(&RealContentSpec {
        name: "carplane_1080p_30f",
        source_mp4: "Artlist_CarPlane.mp4",
        width: 1072, height: 1920,
        n_frames: 30, gop_size: 30,
        msg: "phasm v1.0 ship gate: real-content streaming round-trip",
        pass: "pw",
        cost_weights: CostWeights::default(),
    });
}

/// Smaller repro — try to find minimum frame count that still triggers
/// the drift. If this passes but 30f fails, the bug is cumulative
/// (Pass 2 mode-decision drift compounds across P-frames).
#[test]
#[ignore = "Phase B.2 bisect — 12f minimum repro"]
fn oh264_streaming_real_carplane_1080p_12f() {
    run_real_content_streaming(&RealContentSpec {
        name: "carplane_1080p_12f",
        source_mp4: "Artlist_CarPlane.mp4",
        width: 1072, height: 1920,
        n_frames: 12, gop_size: 12,
        msg: "12f repro",
        pass: "pw",
        cost_weights: CostWeights::default(),
    });
}

/// Even smaller — 6 frames is the minimum that still has 5 P-frames
/// after the IDR, enough surface for mode-decision drift to manifest.
#[test]
#[ignore = "Phase B.2 bisect — 6f minimum repro"]
fn oh264_streaming_real_carplane_1080p_6f() {
    run_real_content_streaming(&RealContentSpec {
        name: "carplane_1080p_6f",
        source_mp4: "Artlist_CarPlane.mp4",
        width: 1072, height: 1920,
        n_frames: 6, gop_size: 6,
        msg: "6f repro",
        pass: "pw",
        cost_weights: CostWeights::default(),
    });
}

/// 3-frame minimum: I + P + P. Tests whether 1 P-frame after IDR is
/// enough to trigger drift on real content, OR whether the drift only
/// shows up at second-or-later P-frame (cumulative effect).
#[test]
#[ignore = "Phase B.2 bisect — 3f minimum repro (IPP)"]
fn oh264_streaming_real_carplane_1080p_3f() {
    run_real_content_streaming(&RealContentSpec {
        name: "carplane_1080p_3f",
        source_mp4: "Artlist_CarPlane.mp4",
        width: 1072, height: 1920,
        n_frames: 3, gop_size: 3,
        msg: "3f IPP repro",
        pass: "pw",
        cost_weights: CostWeights::default(),
    });
}

/// Phase B.3 — same 3f IPP repro but with PHASM_USE_WIRE_ONLY=0
/// (legacy dual_recon path). If this PASSES while the default
/// wire_only=1 version FAILS, the bug is exclusively in the wire_only
/// DecisionCache REPLAY drift. If it ALSO fails, the bug is broader
/// (e.g. MVD override mutates encoder state in both modes).
#[test]
#[ignore = "Phase B.3 bisect — wire_only=0 on 3f IPP"]
fn oh264_streaming_real_carplane_1080p_3f_wire_only_off() {
    // SAFETY: --test-threads=1 enforced for OH264 session serialisation.
    unsafe { std::env::set_var("PHASM_USE_WIRE_ONLY", "0") };
    let result = std::panic::catch_unwind(|| {
        run_real_content_streaming(&RealContentSpec {
            name: "carplane_1080p_3f_wire_only_off",
            source_mp4: "Artlist_CarPlane.mp4",
            width: 1072, height: 1920,
            n_frames: 3, gop_size: 3,
            msg: "3f IPP wire_only=0 repro",
            pass: "pw",
            cost_weights: CostWeights::default(),
        });
    });
    unsafe { std::env::remove_var("PHASM_USE_WIRE_ONLY") };
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}

/// Phase B.3 — 12f repro under wire_only=0. Larger surface; if 3f
/// wire_only=0 PASSES but this FAILS, dual_recon cascade-break starts
/// losing accuracy past 3 P-frames.
#[test]
#[ignore = "Phase B.3 bisect — wire_only=0 on 12f"]
fn oh264_streaming_real_carplane_1080p_12f_wire_only_off() {
    unsafe { std::env::set_var("PHASM_USE_WIRE_ONLY", "0") };
    let result = std::panic::catch_unwind(|| {
        run_real_content_streaming(&RealContentSpec {
            name: "carplane_1080p_12f_wire_only_off",
            source_mp4: "Artlist_CarPlane.mp4",
            width: 1072, height: 1920,
            n_frames: 12, gop_size: 12,
            msg: "12f wire_only=0 repro",
            pass: "pw",
            cost_weights: CostWeights::default(),
        });
    });
    unsafe { std::env::remove_var("PHASM_USE_WIRE_ONLY") };
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}

/// 2-frame: I + P. If this fails, drift starts at the very first P
/// after IDR.
#[test]
#[ignore = "Phase B.2 bisect — 2f IP smallest possible"]
fn oh264_streaming_real_carplane_1080p_2f() {
    run_real_content_streaming(&RealContentSpec {
        name: "carplane_1080p_2f",
        source_mp4: "Artlist_CarPlane.mp4",
        width: 1072, height: 1920,
        n_frames: 2, gop_size: 2,
        msg: "2f IP repro",
        pass: "pw",
        cost_weights: CostWeights::default(),
    });
}

/// Second real-content fixture — iphone7 IMG_4138 1080p. Confirms the
/// drift isn't carplane-specific.
#[test]
#[ignore = "Phase B.5 second fixture — iphone7 real content"]
fn oh264_streaming_real_iphone7_1080p_12f() {
    run_real_content_streaming(&RealContentSpec {
        name: "iphone7_1080p_12f",
        source_mp4: "IMG_4138.MOV",
        // IMG_4138 is landscape — invert dims
        width: 1920, height: 1072,
        n_frames: 12, gop_size: 12,
        msg: "iphone7 real-content repro",
        pass: "pw",
        cost_weights: CostWeights::default(),
    });
}

/// Phase C — walker parity: confirm `record_mvd: true` adds MVD
/// positions but DOES NOT change CS positions on real-content stream.
/// If CS positions or bits differ between the two walks, candidate 2
/// (walker option) is the bug. If identical, eliminate candidate 2.
#[test]
#[ignore = "Phase C — walker CS-position parity with/without record_mvd"]
fn phase_c_walker_cs_parity_on_real_carplane() {
    use phasm_core::codec::h264::cabac::bin_decoder::{
        walk_annex_b_for_cover_with_options, WalkOptions,
    };
    use phasm_core::codec::h264::openh264_stego::{
        openh264_stego_encode_yuv_text, EncodeOpts,
    };
    let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());
    let yuv = ensure_real_yuv("Artlist_CarPlane.mp4", 1072, 1920, 12);
    let opts = EncodeOpts { qp: 26, intra_period: 12 };
    let bitstream = openh264_stego_encode_yuv_text(
        &yuv, 1072, 1920, 12, opts, "walker parity probe", "pw",
    ).expect("baseline encode");

    let walk_default = walk_annex_b_for_cover_with_options(
        &bitstream, WalkOptions::default(),
    ).expect("walk default");
    let walk_with_mvd = walk_annex_b_for_cover_with_options(
        &bitstream, WalkOptions { record_mvd: true, ..Default::default() },
    ).expect("walk record_mvd");

    let cs_default = &walk_default.cover.coeff_sign_bypass;
    let cs_with_mvd = &walk_with_mvd.cover.coeff_sign_bypass;

    eprintln!(
        "CS positions: default={} record_mvd={}",
        cs_default.positions.len(), cs_with_mvd.positions.len(),
    );
    eprintln!(
        "CS bits:      default={} record_mvd={}",
        cs_default.bits.len(), cs_with_mvd.bits.len(),
    );
    assert_eq!(
        cs_default.positions.len(), cs_with_mvd.positions.len(),
        "CS position count mismatch with vs without record_mvd",
    );
    assert_eq!(
        cs_default.bits, cs_with_mvd.bits,
        "CS bits differ with vs without record_mvd",
    );
    for i in 0..cs_default.positions.len() {
        assert_eq!(
            cs_default.positions[i].raw(),
            cs_with_mvd.positions[i].raw(),
            "CS position[{i}] raw mismatch",
        );
    }
    eprintln!("#549 Phase C walker parity GREEN: CS positions identical with/without record_mvd");
}

/// Phase C — encoder Pass 1 byte-identity between 1-domain and 4-domain
/// callers on the SAME YUV. If the baseline bitstreams differ, the bug
/// is at Pass 1 (registration/walker affects encoder output). If they
/// match, Pass 1 is clean and the bug is in Pass 2 or after.
///
/// Both functions call the SAME `encode_once` with `dual_recon=false`
/// + `PassMode::Passthrough` (in wire_only=0 mode for 4-domain). The
/// only structural diff is the `phasm_reset_encoder_session_state()`
/// call + the `record_mvd: true` walker option afterwards. Neither
/// should affect Pass 1's encoder output.
#[test]
#[ignore = "Phase C — Pass 1 byte-identity 1-domain vs 4-domain"]
fn phase_c_pass1_bytewise_parity() {
    use phasm_core::codec::h264::openh264_stego::{
        encode_yuv_with_pre_framed_bits_4domain, EncodeOpts,
    };
    use phasm_core::codec::h264::stego::CostWeights;
    // Default wire_only=1 path — DecisionCache CAPTURE→REPLAY enabled.
    unsafe { std::env::set_var("PHASM_USE_WIRE_ONLY", "1") };
    let result = std::panic::catch_unwind(|| {
        let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());
        let yuv = ensure_real_yuv("Artlist_CarPlane.mp4", 1072, 1920, 12);
        let opts = EncodeOpts { qp: 26, intra_period: 12 };
        let frame_bits: Vec<u8> = (0..480u32).map(|i| (i & 1) as u8).collect();
        let hhat_seed = [42u8; 32];
        let weights = CostWeights::default();

        // First call.
        let result_a = encode_yuv_with_pre_framed_bits_4domain(
            &yuv, 1072, 1920, 12, opts, &frame_bits, &hhat_seed, &weights,
        ).expect("first 4-domain encode");
        // Second call — identical input. Bug repro #548 tests this on
        // synth YUV; here we use REAL content.
        let result_b = encode_yuv_with_pre_framed_bits_4domain(
            &yuv, 1072, 1920, 12, opts, &frame_bits, &hhat_seed, &weights,
        ).expect("second 4-domain encode");

        let a_len = result_a.len();
        let b_len = result_b.len();
        let first_diff = (0..a_len.min(b_len)).find(|&i| result_a[i] != result_b[i]);
        eprintln!(
            "two sequential 4-domain encodes on real carplane: a={a_len} b={b_len} first_diff={first_diff:?}",
        );
        assert_eq!(a_len, b_len, "sequential 4-domain calls produced different-sized output");
        assert_eq!(result_a, result_b, "sequential 4-domain calls produced different bytes (first_diff={first_diff:?})");
        eprintln!("#549 Phase C two-sequential-call parity GREEN");
    });
    unsafe { std::env::remove_var("PHASM_USE_WIRE_ONLY") };
    if let Err(p) = result { std::panic::resume_unwind(p); }
}

/// DEFAULT-SUITE SHIP GATE for #549 real-content streaming round-trip.
/// 480×272 (16-aligned 480p) × 12f — ~500 ms wall-clock, exercises
/// all four override domains end-to-end on real motion content.
/// Green post-2026-05-19 (#549 Bug 5 closed). Requires
/// `core/test-vectors/video/h264/real-world/source/Artlist_CarPlane.mp4`
/// (gitignored — provide locally for CI).
#[test]
fn oh264_streaming_real_carplane_480p_12f() {
    run_real_content_streaming(&RealContentSpec {
        name: "carplane_480p_12f",
        source_mp4: "Artlist_CarPlane.mp4",
        width: 480, height: 272,
        n_frames: 12, gop_size: 12,
        msg: "480p fast bisect",
        pass: "pw",
        cost_weights: CostWeights::default(),
    });
}

// ─────────────── Phase B.4 — per-domain isolation ─────────────────────
//
// Hypothesis: the bug is exclusively in MVD overrides (MvdSign/MvdSuffix).
// The legacy single-domain CS-only path (`openh264_stego_encode_yuv_text`)
// works on real content; the 4-domain streaming session does not.
// Difference: 4-domain adds CSL + MVDs + MVDsl overrides on top of CS.
//
// These tests use STREAMING SESSION + 4-domain cover layout but constrain
// the STC plan to a single domain via WET-∞ on the others. If
// `coeff_sign_only` PASSES while `default()` FAILS, the bug is in one
// of the 3 added domains. Then drill: CS+CSL only, CS+MVDs only.

/// Phase B.4 — streaming session with CS-only STC plan (CSL/MVDs/MVDsl
/// receive ∞ cost so STC never selects them). Cover composition is the
/// same 4-domain layout (cover positions for all 4 domains are walked
/// and combined), but no overrides fire on the 3 added domains.
///
/// If this PASSES on real content while `default()` FAILS, the bug
/// is exclusively in the MVD/CSL override-side path mutating encoder
/// state in a way that changes mode decisions.
#[test]
#[ignore = "Phase B.4 bisect harness (obsolete post-Bug-5 fix); restricted-cover plans hit MessageTooLarge with the original test message because accurate post-fix cover counts revealed pre-fix over-counting"]
fn oh264_streaming_real_carplane_1080p_12f_cs_only_plan() {
    run_real_content_streaming(&RealContentSpec {
        name: "carplane_1080p_12f_cs_only_plan",
        source_mp4: "Artlist_CarPlane.mp4",
        width: 1072, height: 1920,
        n_frames: 12, gop_size: 12,
        msg: "CS-only STC plan on 4-domain cover",
        pass: "pw",
        cost_weights: CostWeights::debug_coeff_sign_only(),
    });
}

/// Phase B.4 — CS+CSL only (MVDs/MVDsl WET-∞). If this PASSES while
/// `default()` FAILS, MVD overrides specifically are the cascade
/// source. If it FAILS, CSL is also a contributor (less likely given
/// CSL has only 522 diffs in trace).
#[test]
#[ignore = "Phase B.4 bisect harness (obsolete post-Bug-5 fix); see _cs_only_plan note"]
fn oh264_streaming_real_carplane_1080p_12f_cs_csl_only() {
    run_real_content_streaming(&RealContentSpec {
        name: "carplane_1080p_12f_cs_csl_only",
        source_mp4: "Artlist_CarPlane.mp4",
        width: 1072, height: 1920,
        n_frames: 12, gop_size: 12,
        msg: "CS+CSL only STC plan",
        pass: "pw",
        cost_weights: CostWeights::conservative_cs_csl_only(),
    });
}

/// Phase B.4 — MVD-sign only STC plan. If this fails standalone, then
/// MVD overrides alone (no CS interaction) cause encoder drift on real
/// content. Pure smoking-gun isolation test for the MV-cache mutation
/// hypothesis.
#[test]
#[ignore = "Phase B.4 bisect harness (obsolete post-Bug-5 fix); see _cs_only_plan note"]
fn oh264_streaming_real_carplane_1080p_12f_mvd_sign_only() {
    run_real_content_streaming(&RealContentSpec {
        name: "carplane_1080p_12f_mvd_sign_only",
        source_mp4: "Artlist_CarPlane.mp4",
        width: 1072, height: 1920,
        n_frames: 12, gop_size: 12,
        msg: "MvdSign-only STC plan",
        pass: "pw",
        cost_weights: CostWeights::debug_mvd_sign_only(),
    });
}

/// Phase B.4 — sanity: legacy 1-domain `openh264_stego_encode_yuv_text`
/// on the same 12f real carplane YUV. Goes through the OLD encode path
/// (`encode_yuv_with_pre_framed_bits`, NOT the 4-domain function).
/// If this PASSES while every 4-domain variant FAILS on the same YUV,
/// it pinpoints the bug to `encode_yuv_with_pre_framed_bits_4domain`
/// (or something it triggers in the fork that the 1-domain function
/// doesn't).
#[test]
#[ignore = "Phase B.4 — sanity: legacy 1-domain path on same real YUV"]
fn oh264_legacy_1domain_real_carplane_1080p_12f() {
    use phasm_core::codec::h264::openh264_stego::{
        openh264_stego_decode_yuv_string, openh264_stego_encode_yuv_text,
        EncodeOpts,
    };
    let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());
    let yuv = ensure_real_yuv("Artlist_CarPlane.mp4", 1072, 1920, 12);
    let msg = "1-domain sanity on real content";
    let pass = "pw";
    let opts = EncodeOpts { qp: 26, intra_period: 12 };
    let stego = openh264_stego_encode_yuv_text(&yuv, 1072, 1920, 12, opts, msg, pass)
        .expect("1-domain legacy encode");
    let recovered = openh264_stego_decode_yuv_string(&stego, pass)
        .expect("1-domain legacy decode");
    assert_eq!(recovered, msg, "legacy 1-domain failed on real content (would be new bug)");
    eprintln!(
        "#549 legacy 1-domain GREEN on real-content carplane 1072×1920 × 12f ({} bytes)",
        stego.len(),
    );
}

/// Phase B.4 — CS-only STC plan + wire_only=0 (legacy dual_recon path).
/// If this PASSES while default `cs_only_plan` test FAILS, the wire_only
/// REPLAY path drifts even when no MVD overrides are written. If this
/// FAILS, the 4-domain cover *enumeration* itself (cover positions
/// walked differently in 4-domain mode vs 1-domain mode) is broken.
#[test]
#[ignore = "Phase B.4 bisect harness (obsolete post-Bug-5 fix); see _cs_only_plan note"]
fn oh264_streaming_real_carplane_1080p_12f_cs_only_wire_off() {
    unsafe { std::env::set_var("PHASM_USE_WIRE_ONLY", "0") };
    let result = std::panic::catch_unwind(|| {
        run_real_content_streaming(&RealContentSpec {
            name: "carplane_1080p_12f_cs_only_wire_off",
            source_mp4: "Artlist_CarPlane.mp4",
            width: 1072, height: 1920,
            n_frames: 12, gop_size: 12,
            msg: "CS-only + wire_only=0",
            pass: "pw",
            cost_weights: CostWeights::debug_coeff_sign_only(),
        });
    });
    unsafe { std::env::remove_var("PHASM_USE_WIRE_ONLY") };
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}
