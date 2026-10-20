// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
//
// #810 — real-world capacity data collection. Measures CAP2.1 reported
// capacity vs the real encoder's true max on ONE corpus fixture, at its
// NATIVE (16-aligned) resolution and full length, driven by env vars so a
// shell driver can fan out one process per video in parallel (the OH264
// encoder has C-global state, so cross-process is the only safe
// parallelism). Emits a single `RESULT ...` line on stderr.
//
// Driver: cap_realworld_810.sh. Run:
//   cargo test --test cap_realworld_810 --features h264-encoder,cabac-stego \
//     --release --no-run     # build once
//   ./cap_realworld_810.sh   # fan out
//
// Env: PHASM_CAP_SRC (filename), PHASM_CAP_TAG, PHASM_CAP_W, PHASM_CAP_H,
//      PHASM_CAP_N, PHASM_CAP_GOP.

#![cfg(feature = "h264-encoder")]

mod common;
use phasm_core::{
    h264_shadow_capacity_for_n,
    h264_video_capacity,
    ColorParams, CostWeights, EncodeEngineChoice, EncodeSessionParams,
    StreamingDecodeSession, StreamingEncodeSession, StreamingProbeSession, YuvFrameRef,
};
use phasm_core::stego::chunk_frame::CAP2_DEFAULT_R_TARGET;
use phasm_core::stego::calibration::AllocationCalibration;
use phasm_core::codec::h264::streaming_session::H264PlanOutcome;
use phasm_core::codec::h264::openh264_stego::{h264_walk_cover, EncodeOpts};
use phasm_core::codec::h264::cabac::bin_decoder::{walk_annex_b_for_cover_with_options, WalkOptions};

fn env_u32(k: &str) -> u32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

/// CAP2.5 — the public N-aware shadow-capacity helper the mobile bridges
/// call once per shadow-count change to size each segment of the segmented
/// shadow bar (closed-form on `shadow_pool_bits`, no re-probe). Black-box
/// property checks: zero edges, monotone non-increasing in N, and a real
/// √N drop past the collision knee.
#[test]
fn cap_shadow_capacity_for_n_properties() {
    let pool = 2_000_000usize; // ~ a real 1080p 3-domain injectable pool

    assert_eq!(h264_shadow_capacity_for_n(pool, 0), 0, "n=0 → no shadow capacity");
    assert_eq!(h264_shadow_capacity_for_n(0, 4), 0, "empty pool → 0, no panic");

    // Non-increasing in N (n=1 and n=2 tie — the collision denominator
    // clamps to 1 below 2 messages — then it falls as sharers are added).
    let mut prev = usize::MAX;
    for n in 1..=8usize {
        let cap = h264_shadow_capacity_for_n(pool, n);
        assert!(cap <= prev, "per-shadow cap must be non-increasing (n={n}: {cap} > {prev})");
        prev = cap;
    }

    // A real √N falloff: 8 sharers get materially less room than 2.
    let c2 = h264_shadow_capacity_for_n(pool, 2);
    let c8 = h264_shadow_capacity_for_n(pool, 8);
    assert!(c8 < c2, "8 shadows must give strictly less per-shadow room than 2 ({c8} !< {c2})");
    assert!(c2 > 0, "a 2 MB pool must afford a non-trivial 2-shadow budget");
}

/// Crop (NOT scale) to the 16-aligned native resolution → preserves the
/// content's real detail / inf-cost fraction. Cached in /tmp for re-runs.
fn crop_yuv(src: &str, w: u32, h: u32, n: u32) -> Option<Vec<u8>> {
    let src_path = corpus_root().join(src);
    if !src_path.exists() {
        return None;
    }
    let key: String = src.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let yuv_path = format!("{}/phasm_cap810_{}_{}x{}_f{}.yuv", common::scratch_dir(), key, w, h, n);
    let need = (w * h * 3 / 2) as usize * n as usize;
    if let Ok(d) = std::fs::read(&yuv_path) {
        if d.len() >= need {
            return Some(d[..need].to_vec());
        }
    }
    let vf = format!("crop={}:{}:0:0", w, h);
    let ok = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&src_path)
        .args(["-frames:v", &n.to_string(), "-an", "-vf", &vf,
               "-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path).status().ok()?.success();
    if !ok {
        return None;
    }
    let d = std::fs::read(&yuv_path).ok()?;
    if d.len() < need {
        return None;
    }
    Some(d[..need].to_vec())
}

/// `encode_bytes`, but surfacing the real failure.
///
/// The `Option` form below collapses three distinct failures — session
/// create, a `push_frame`, and `finish` — into a bare `None`, so an
/// `.expect()` on it reports only *that* an encode failed, never which stage
/// or why. Anything asserting encode success should use this instead; the
/// `Option` form stays for the call sites that genuinely only need the bool.
fn encode_bytes_res(
    yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, msg: &str, pass: &str,
) -> Result<Vec<u8>, String> {
    let params = EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut enc = StreamingEncodeSession::create(params, msg, pass)
        .map_err(|e| format!("session create: {e:?}"))?;
    let fb = (w * h * 3 / 2) as usize;
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let cw = (w / 2) as usize;
    let mut out = Vec::new();
    for f in 0..n as usize {
        let off = f * fb;
        let frame = YuvFrameRef {
            y: &yuv[off..off + ys], y_stride: w as usize,
            u: &yuv[off + ys..off + ys + cs], u_stride: cw,
            v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
        };
        enc.push_frame(frame, &mut out)
            .map_err(|e| format!("push_frame #{f}: {e:?}"))?;
    }
    enc.finish(&mut out).map_err(|e| format!("finish: {e:?}"))?;
    Ok(out)
}

fn encode_bytes(
    yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, msg: &str, pass: &str,
) -> Option<Vec<u8>> {
    encode_bytes_res(yuv, w, h, n, gop, msg, pass).ok()
}

/// Probe pass alone: the per-GOP cap vector `plan_proportional` consumes.
///
/// Split out from [`encode_bytes_planned`] because the caps depend only on
/// (YUV, params) — NOT on the message — so a K-trial success-rate measurement
/// probes once and re-plans per trial instead of paying the probe K times.
/// `plan_target` mirrors production's `set_plan_target` early-exit; `None`
/// measures every GOP (what a full-scan production run yields).
fn probe_per_gop_caps(
    yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, plan_target: Option<(usize, f64)>,
) -> Result<Vec<usize>, String> {
    let params = EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut probe = StreamingProbeSession::create(params)
        .map_err(|e| format!("probe create: {e:?}"))?;
    if let Some((framed, r)) = plan_target {
        probe.set_plan_target(framed, r);
    }
    let fb = (w * h * 3 / 2) as usize;
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let cw = (w / 2) as usize;
    for f in 0..n as usize {
        let off = f * fb;
        probe
            .push_frame(YuvFrameRef {
                y: &yuv[off..off + ys], y_stride: w as usize,
                u: &yuv[off + ys..off + ys + cs], u_stride: cw,
                v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
            })
            .map_err(|e| format!("probe push #{f}: {e:?}"))?;
    }
    let (_res, caps) = probe.finish_with_per_gop().map_err(|e| format!("probe finish: {e:?}"))?;
    Ok(caps)
}

/// Encode with an ALREADY-MEASURED cap vector installed via
/// `plan_proportional`. Returns the bytes and whether the plan was installed
/// (`false` = silently degraded to uniform spread).
fn encode_bytes_with_plan(
    yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, msg: &str, pass: &str,
    caps: &[usize], r_target: f64,
) -> Result<(Vec<u8>, bool), String> {
    let params = EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut enc = StreamingEncodeSession::create(params, msg, pass)
        .map_err(|e| format!("session create: {e:?}"))?;
    let planned = enc
        .plan_proportional(caps, r_target)
        .map_err(|e| format!("plan_proportional: {e:?}"))?;
    let fb = (w * h * 3 / 2) as usize;
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let cw = (w / 2) as usize;
    let mut out = Vec::new();
    for f in 0..n as usize {
        let off = f * fb;
        enc.push_frame(
            YuvFrameRef {
                y: &yuv[off..off + ys], y_stride: w as usize,
                u: &yuv[off + ys..off + ys + cs], u_stride: cw,
                v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
            },
            &mut out,
        )
        .map_err(|e| format!("push_frame #{f}: {e:?}"))?;
    }
    enc.finish(&mut out).map_err(|e| format!("finish: {e:?}"))?;
    Ok((out, planned))
}

/// CLI-path encode: probe → per-GOP caps → `plan_proportional` → sequential
/// encode, mirroring `core/cli/src/video_encode.rs:563`.
///
/// **This is the CLI planner, not the one the apps use.** Both bridges *expose*
/// `plan_proportional`, but iOS `PhasmStegoTranscoder.swift:552` and Android
/// `PhasmStegoTranscoder.kt:207` both call `plan_safe_balanced` (#841) —
/// a different allocator with a corpus-measured calibration and a 1.1 safety
/// margin. For that path see `encode_bytes_app_planner` below.
///
/// This distinction is load-bearing and was the trap in this file. The plain
/// `encode_bytes` above installs NO allocation plan, so the session falls back
/// to uniform spread, whose ceiling is ≈ `n_gops × min(cap)`. The capacity
/// `h264_video_capacity` reports is `Σ(cap) − overhead`, which only the
/// PLANNED path can reach. Comparing a reported capacity against an unplanned
/// encode therefore compares two different ceilings.
///
/// Returns the stego bytes, the probe's per-GOP caps, and whether a plan was
/// actually installed (`plan_proportional` returns `false` and silently
/// degrades to uniform spread when the message exceeds Σ caps or the GOP
/// shapes disagree — a caller that ignores this gets the low ceiling without
/// being told).
/// Kept as the executable record of the CLI's probe→plan→encode sequence even
/// when no test currently drives it; the doc comment above is referenced from
/// the gates.
#[allow(dead_code)]
fn encode_bytes_planned(
    yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, msg: &str, pass: &str, r_target: f64,
) -> Result<(Vec<u8>, Vec<usize>, bool), String> {
    let params = EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut enc = StreamingEncodeSession::create(params.clone(), msg, pass)
        .map_err(|e| format!("session create: {e:?}"))?;

    let fb = (w * h * 3 / 2) as usize;
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let cw = (w / 2) as usize;
    let frame_at = |f: usize| {
        let off = f * fb;
        YuvFrameRef {
            y: &yuv[off..off + ys], y_stride: w as usize,
            u: &yuv[off + ys..off + ys + cs], u_stride: cw,
            v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
        }
    };

    // Probe pass, including the same `set_plan_target` early-exit production
    // uses (it stops measuring once no further GOP can change the plan).
    let mut probe = StreamingProbeSession::create(params)
        .map_err(|e| format!("probe create: {e:?}"))?;
    if let Some(framed) = enc.framed_message_len() {
        probe.set_plan_target(framed, r_target);
    }
    for f in 0..n as usize {
        probe.push_frame(frame_at(f)).map_err(|e| format!("probe push #{f}: {e:?}"))?;
    }
    let (_probe_result, caps) =
        probe.finish_with_per_gop().map_err(|e| format!("probe finish: {e:?}"))?;

    let planned = enc
        .plan_proportional(&caps, r_target)
        .map_err(|e| format!("plan_proportional: {e:?}"))?;

    let mut out = Vec::new();
    for f in 0..n as usize {
        enc.push_frame(frame_at(f), &mut out)
            .map_err(|e| format!("push_frame #{f}: {e:?}"))?;
    }
    enc.finish(&mut out).map_err(|e| format!("finish: {e:?}"))?;
    Ok((out, caps, planned))
}

/// What `plan_safe_balanced` did, so a gate can tell an installed plan from
/// the apps' silent downgrade.
#[derive(Debug, Clone, Copy, PartialEq)]
enum AppPlan {
    /// Plan installed; `window` leading GOPs carry stego.
    Installed(u32),
    /// The planner asked for MORE samples although every GOP had already been
    /// measured. Both apps log this and then encode **with no plan at all**
    /// (iOS `PhasmStegoTranscoder.swift:294`, Android `.kt:213` — "uniform
    /// fallback"), which drops the session onto the uniform-share+carry
    /// ceiling of ≈ `n_gops × min(cap)`. On heterogeneous content that is far
    /// below the reported Σ-based capacity, so the downgrade converts a
    /// displayed-capacity message into a mid-encode `MessageTooLarge`.
    UniformFallback,
}

/// APP-path encode: probe every GOP → `plan_safe_balanced` (#841) → encode.
///
/// This is what iOS and Android actually run. Differences from the CLI path
/// that matter for capacity:
///
/// * the allocator is `stego::balanced_allocation::plan_safe_balanced`, not
///   `allocate_chunks_concentrate_tail`;
/// * it is calibrated (`AllocationCalibration::H264_1080P_QP26`, corpus-measured
///   in `docs/design/video/calibration-h264-tier1-2026-06.md`) where the CLI's
///   `CAP2_DEFAULT_R_TARGET` is documented in-source as "Provisional, NOT yet
///   calibrated";
/// * it applies `safety_margin = 1.1`, i.e. it refuses up front unless the
///   safe total clears the message by 10% — the pre-encode headroom the CLI
///   path has none of;
/// * a refusal is therefore INSTANT (pre-encode) rather than arriving from
///   `finish()` after the user has waited out a full encode.
///
/// Both apps hand it the complete sample set (they probe every GOP), so that
/// is what this mirrors. `Err` strings are prefixed `planner:` for a
/// pre-encode refusal and `finish:`/`push_frame:` for a mid-encode one, which
/// the gate classifies separately.
#[allow(clippy::too_many_arguments)]
fn encode_bytes_app_planner(
    yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, msg: &str, pass: &str,
    caps: &[usize],
) -> Result<(Vec<u8>, AppPlan), String> {
    let params = EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut enc = StreamingEncodeSession::create(params, msg, pass)
        .map_err(|e| format!("session create: {e:?}"))?;

    // The apps probe every GOP and pass the lot; `caps` is that sample set.
    let samples: Vec<(usize, usize)> =
        caps.iter().copied().enumerate().collect();
    let cal = AllocationCalibration::H264_1080P_QP26;
    let plan = match enc
        .plan_safe_balanced(&samples, gop as usize, &cal)
        .map_err(|e| format!("planner: {e:?}"))?
    {
        H264PlanOutcome::Installed { window } => AppPlan::Installed(window),
        H264PlanOutcome::MessageTooLarge => {
            return Err("planner: MessageTooLarge".to_string())
        }
        // Mirror the apps exactly: warn and carry on with NO plan.
        H264PlanOutcome::NeedMoreSamples { .. } => AppPlan::UniformFallback,
    };

    let fb = (w * h * 3 / 2) as usize;
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let cw = (w / 2) as usize;
    let mut out = Vec::new();
    for f in 0..n as usize {
        let off = f * fb;
        enc.push_frame(
            YuvFrameRef {
                y: &yuv[off..off + ys], y_stride: w as usize,
                u: &yuv[off + ys..off + ys + cs], u_stride: cw,
                v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
            },
            &mut out,
        )
        .map_err(|e| format!("push_frame #{f}: {e:?}"))?;
    }
    enc.finish(&mut out).map_err(|e| format!("finish: {e:?}"))?;
    Ok((out, plan))
}


/// Like `encode_bytes` but with explicit per-domain `CostWeights` (for the
/// per-domain isolation cascade stress — forces all flips into one domain).
fn encode_bytes_w(
    yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, msg: &str, pass: &str,
    weights: CostWeights,
) -> Option<Vec<u8>> {
    let params = EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: weights,
        progress_callback: None,
    };
    let mut enc = StreamingEncodeSession::create(params, msg, pass).ok()?;
    let fb = (w * h * 3 / 2) as usize;
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let cw = (w / 2) as usize;
    let mut out = Vec::new();
    for f in 0..n as usize {
        let off = f * fb;
        let frame = YuvFrameRef {
            y: &yuv[off..off + ys], y_stride: w as usize,
            u: &yuv[off + ys..off + ys + cs], u_stride: cw,
            v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
        };
        enc.push_frame(frame, &mut out).ok()?;
    }
    enc.finish(&mut out).ok()?;
    Some(out)
}

/// Decode + recover the embedded text (StreamingDecodeSession).
fn decode_text(annex_b: &[u8], pass: &str) -> Option<String> {
    let mut dec = StreamingDecodeSession::create(pass).ok()?;
    dec.push_annex_b(annex_b).ok()?;
    dec.finish().ok().map(|d| d.text)
}

/// Deterministic ~incompressible printable-ASCII (LCG) — measures real STC
/// capacity, not Brotli compressibility.
fn incompressible(len: usize) -> String {
    let mut s = String::with_capacity(len);
    let mut st: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..len {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s.push((0x21u8 + ((st >> 33) as usize % 0x5d) as u8) as char);
    }
    s
}

/// The framed length of `incompressible(len)` — what every capacity ceiling in
/// this file is actually denominated in.
///
/// `incompressible` is a misnomer kept for continuity: it emits printable
/// ASCII over a 93-symbol alphabet (~6.54 bits/char), and `stego::payload`
/// brotli-compresses the payload whenever that shrinks it
/// (`payload.rs:206 try_compress`). So a raw text message of N chars frames to
/// ≈ 0.83 N bytes, and a gate that sets `len = reported × frac` is really
/// testing `frac × 0.83` of the reported capacity. That understated every
/// size in this file by ~17% until 2026-10-18.
///
/// `h264_video_capacity` reports a FRAMED-byte budget (Σ per-GOP caps −
/// `FRAME_OVERHEAD`), so framed bytes are the right denominator. An
/// incompressible payload — a file attachment, say — frames ~1:1 and is the
/// case the reported number must hold for.
fn framed_len_of(len: usize, w: u32, h: u32, n: u32, gop: u32, pass: &str) -> Option<usize> {
    let params = EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let msg = incompressible(len);
    StreamingEncodeSession::create(params, &msg, pass)
        .ok()
        .and_then(|e| e.framed_message_len())
}

/// Largest raw `incompressible(..)` length whose FRAMED size is ≤
/// `target_framed`. Bisects; `framed_len_of` is monotonic in `len` because
/// `incompressible(n)` is a fixed prefix of one LCG stream.
///
/// This is what lets a gate say "100% of the reported capacity" and mean it.
#[allow(clippy::too_many_arguments)]
fn raw_len_for_framed(
    target_framed: usize, w: u32, h: u32, n: u32, gop: u32, pass: &str,
) -> usize {
    if target_framed == 0 {
        return 0;
    }
    // Upper bracket: raw never frames to less than ~0.7x, so 2x target is safe.
    let mut lo = 1usize;
    let mut hi = target_framed.saturating_mul(2).max(64);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        match framed_len_of(mid, w, h, n, gop, pass) {
            Some(f) if f <= target_framed => lo = mid,
            _ => hi = mid,
        }
    }
    lo
}

/// #804 — confirm lumix (J2-flaky) still encodes a sane amount post-verify
/// (not a 0-capacity regression), and report its round-trip-safe max.
#[test]
#[ignore]
fn lumix_safe_cap_probe() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 150;
    const GOP: u32 = 30;
    for src in [
        "lumix_g9_1080p_30fps_h264_high.mp4",
        "iphone7_1080p_30fps_h264_high.mov",
    ] {
        let Some(yuv) = crop_yuv(src, W, H, N) else {
            eprintln!("SKIP {src}");
            continue;
        };
        let short: String = src.chars().take(12).collect();
        // gallop the round-trip-safe max from a low start
        let tmax = true_max_from(&yuv, W, H, N, GOP, 300);
        eprintln!(
            "{short}: round-trip-safe max = {tmax} B  (~{} B/GOP over {} GOPs)",
            tmax / (N / GOP) as usize,
            N / GOP,
        );
        assert!(tmax > 0, "{short}: zero capacity — verify loop over-rejects (BUG)");
    }
}

/// #804 §14 SHIP GATE — the verify loop must CLOSE the silent-loss zone. For a
/// payload-size sweep on the J2-flaky fixture (lumix) that previously
/// encoded-OK-but-decoded-WRONG, EVERY encode-success must now round-trip
/// (encode either fits-and-decodes, or gracefully MessageTooLarge — never
/// silent loss). This is the whole point of `drain_one_gop`'s verify+carry.
///
///     cargo test -p phasm-core --features h264-encoder,cabac-stego --release \
///         --test cap_realworld_810 cap_no_silent_loss -- --ignored --nocapture
#[test]
#[ignore]
fn cap_no_silent_loss() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 150;
    const GOP: u32 = 30;
    let pass = "cap810nsl";
    let mut any = false;
    for src in [
        "lumix_g9_1080p_30fps_h264_high.mp4", // J2-flaky in calibration
        "iphone7_1080p_30fps_h264_high.mov",  // control
    ] {
        let Some(yuv) = crop_yuv(src, W, H, N) else {
            eprintln!("SKIP {src}");
            continue;
        };
        any = true;
        let short: String = src.chars().take(12).collect();
        let (mut tested, mut encoded) = (0usize, 0usize);
        for size in (1800..=7800).step_by(600) {
            tested += 1;
            let msg = incompressible(size);
            if let Some(stego) = encode_bytes(&yuv, W, H, N, GOP, &msg, pass) {
                encoded += 1;
                let dec = decode_text(&stego, pass);
                assert_eq!(
                    dec.as_deref(),
                    Some(msg.as_str()),
                    "{short} size {size}: ENCODED but did NOT round-trip — SILENT LOSS \
                     (drain_one_gop's verify+carry loop failed to catch it)"
                );
            }
        }
        eprintln!("{short}: {encoded}/{tested} sizes encoded — ALL round-tripped (no silent loss)");
    }
    assert!(any, "no fixtures available");
}

/// #804 perf estimate — encode-time round-trip VERIFICATION adds a decode pass
/// (and rare re-encodes on failure). Measure the decode/encode ratio on a
/// 1080p clip so we can size the verification overhead before building it.
///
///     cargo test -p phasm-core --features h264-encoder,cabac-stego --release \
///         --test cap_realworld_810 verification_cost_estimate -- --ignored --nocapture
#[test]
#[ignore]
fn verification_cost_estimate() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 90; // 3 GOPs
    const GOP: u32 = 30;
    let Some(yuv) = crop_yuv("iphone7_1080p_30fps_h264_high.mov", W, H, N) else {
        eprintln!("SKIP (no iphone7 fixture)");
        return;
    };
    let pass = "cap810v";
    let msg = incompressible(2000); // ~667 B/GOP — a representative mid-fill

    // Warm the fixture/codec once (first encode pays one-time setup).
    let _ = encode_bytes(&yuv, W, H, N, GOP, &incompressible(64), pass);

    let t = std::time::Instant::now();
    let stego = encode_bytes(&yuv, W, H, N, GOP, &msg, pass).expect("encode");
    let enc_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = std::time::Instant::now();
    let dec = decode_text(&stego, pass);
    let dec_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(dec.as_deref(), Some(msg.as_str()), "round-trip");

    let n_gops = (N / GOP) as f64;
    eprintln!("\n=== #804 verification cost — iphone7 {N}f / {} GOPs ===", n_gops as u32);
    eprintln!("encode      : {enc_ms:>8.0} ms  ({:.0} ms/GOP)", enc_ms / n_gops);
    eprintln!("decode/verify: {dec_ms:>8.0} ms  ({:.0} ms/GOP)", dec_ms / n_gops);
    eprintln!(
        "verify overhead = {:.0}% of encode (only on HIGH-FILL GOPs near Σ; \
         0% for typical small payloads).",
        100.0 * dec_ms / enc_ms,
    );
}

/// Generic galloping max-finder: largest `len` for which `fits(len)` holds,
/// starting near `start`. Used by the §14 gate for both encode-only and
/// round-trip ceilings.
fn gallop_max(start: usize, fits: &dyn Fn(usize) -> bool) -> usize {
    let start = start.max(1);
    if !fits(start) {
        let (mut lo, mut hi) = (0usize, start);
        while lo + 8 < hi {
            let m = (lo + hi) / 2;
            if fits(m) { lo = m; } else { hi = m; }
        }
        return lo;
    }
    let (mut lo, mut hi) = (start, start + start / 2 + 64);
    while hi < (1 << 20) && fits(hi) {
        lo = hi;
        hi = hi + hi / 2 + 64;
    }
    while lo + 8 < hi {
        let m = (lo + hi) / 2;
        if fits(m) { lo = m; } else { hi = m; }
    }
    lo
}

/// §14 GATE (CAP2.2 proportional) — does each GOP round-trip at its OWN STC
/// cap, or only at the even-split's uniform `min`? The cascade audit (#815)
/// verified UNIFORM fills; proportional allocation fills each GOP up to its
/// own cap. If `enc_max > rt_max` for any GOP there is a **silent-loss zone**
/// (encodes but corrupts on decode) and proportional must cap at `rt_max`, not
/// the STC cap. Empty zone everywhere ⇒ proportional is round-trip-safe → wire
/// it in. Each GOP is probed in isolation (1-GOP clip = full message in one
/// chunk), which is representative: every GOP is IDR-led, the cascade is
/// within-GOP, and per-GOP STC is independent.
///
///     cargo test -p phasm-core --features h264-encoder,cabac-stego --release \
///         --test cap_realworld_810 cap_proportional_roundtrip_gate -- --ignored --nocapture
#[test]
#[ignore]
fn cap_proportional_roundtrip_gate() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 150; // 5 GOPs
    const GOP: u32 = 30;
    let Some(yuv) = crop_yuv("iphone7_1080p_30fps_h264_high.mov", W, H, N) else {
        eprintln!("SKIP (no iphone7 fixture)");
        return;
    };
    let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
    let fs = (W * H * 3 / 2) as usize;
    let n_gops = (N / GOP) as usize;
    let pass = "cap810g";

    // CALIBRATION, not pass/fail. Proportional fills each GOP to its REPORTED
    // STC cap. Where `rep_cap > rt_max` the reported cap OVER-REPORTS the
    // round-trip ceiling (per-GOP J2 message-jitter, #814) — proportional would
    // silently lose data there. The even-split masks this by under-filling
    // high-cap GOPs (binding on the min). So this gate MEASURES the worst
    // per-GOP over-report → the minimum derating margin proportional needs
    // (calibrated for real on the corpus in #806). `hroom = rt_max − rep_cap`
    // (negative ⇒ over-report); `zone = enc_max − rt_max` is the cascade/J2
    // band above the operating point (info).
    eprintln!("\n=== §14 proportional derating-margin calibration — iphone7 {N}f / {n_gops} GOPs ===");
    eprintln!(
        "{:<4} | {:>7} | {:>7} | {:>7} | {:>6} | {:>5}",
        "gop", "rep_cap", "rt_max", "enc_max", "hroom", "zone",
    );
    let mut worst_over = 0.0f64; // max (rep_cap − rt_max)/rep_cap
    for g in 0..n_gops {
        let gy = &yuv[g * GOP as usize * fs..((g + 1) * GOP as usize).min(N as usize) * fs];
        let gf = (gy.len() / fs) as u32;
        let rep_cap = h264_video_capacity(gy, W, H, gf as usize, opts, false)
            .map(|i| i.per_tier_primary_max_message_bytes[0])
            .unwrap_or(0);
        if rep_cap == 0 {
            eprintln!("{g:<4} | rep_cap=0 (skip)");
            continue;
        }
        let enc_fits =
            |len: usize| encode_bytes(gy, W, H, gf, GOP, &incompressible(len), pass).is_some();
        let rt_fits = |len: usize| {
            let m = incompressible(len);
            encode_bytes(gy, W, H, gf, GOP, &m, pass)
                .and_then(|b| decode_text(&b, pass))
                .as_deref()
                == Some(m.as_str())
        };
        let enc_max = gallop_max(rep_cap, &enc_fits);
        let rt_max = gallop_max(rep_cap, &rt_fits);
        let hroom = rt_max as i64 - rep_cap as i64;
        let zone = enc_max.saturating_sub(rt_max);
        let over = ((rep_cap as f64 - rt_max as f64).max(0.0)) / rep_cap as f64;
        worst_over = worst_over.max(over);
        eprintln!(
            "{g:<4} | {rep_cap:>7} | {rt_max:>7} | {enc_max:>7} | {hroom:>6} | {zone:>5}{}",
            if hroom < 0 { "  <-- OVER-REPORT" } else { "" }
        );
    }
    eprintln!(
        "\nworst per-GOP over-report = {:.1}% → proportional MUST derate per-GOP caps by at \
         least this (real margin calibrated on the corpus in #806).",
        worst_over * 100.0,
    );
    // Regression sanity: the reported cap is only SLIGHTLY optimistic (J2-level,
    // single-digit %). If it ever blows past 15% the per-GOP capacity primitive
    // has regressed (a real cascade leak, not jitter) — fail loudly.
    assert!(
        worst_over <= 0.15,
        "per-GOP over-report {:.1}% exceeds the 15% J2 sanity budget — the reported cap is \
         wildly optimistic; a cascade leak has regressed, investigate before any spreading.",
        worst_over * 100.0,
    );
}

/// CAP2.2 #806/#811 — corpus derating-margin calibration. Proportional fills
/// each GOP toward `(1 − margin)·rep_cap`; the salt is RANDOM per encode, so
/// the per-GOP round-trip ceiling is a DISTRIBUTION over salts (J2). A single
/// check under-estimates the margin → unsafe. So measure the round-trip
/// **success RATE over K random salts** at each candidate margin, across a
/// diverse corpus. The smallest margin whose worst-GOP success rate is 100%
/// is the safe derating (the cap on the §4 soft-ceiling fill rate, and the
/// Σ-derate the v3 estimator reports).
///
///     cargo test -p phasm-core --features h264-encoder,cabac-stego --release \
///         --test cap_realworld_810 cap_proportional_margin_corpus -- --ignored --nocapture
#[test]
#[ignore]
fn cap_proportional_margin_corpus() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 150; // 5 GOPs / fixture
    const GOP: u32 = 30;
    const K: usize = 8; // random-salt trials per (gop, margin)
    let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
    let fs = (W * H * 3 / 2) as usize;
    let n_gops = (N / GOP) as usize;
    let pass = "cap810m";
    let fixtures = [
        "iphone7_1080p_30fps_h264_high.mov",
        "Artlist_CarPlane.mp4",
        "lumix_g9_1080p_30fps_h264_high.mp4",
        "iphone5_1080p_30fps_h264_high.mov",
    ];
    let margins = [0.00f64, 0.05, 0.10, 0.15];

    eprintln!("\n=== CAP2 derating-margin CORPUS calibration (K={K} salts/level) ===");
    eprintln!("worst-GOP round-trip success rate at each derating margin:\n");
    eprintln!("{:<26} | {:>5} | {:>5} | {:>5} | {:>5}", "fixture", "0%", "5%", "10%", "15%");
    // worst (min) success rate per margin across ALL gops of ALL fixtures.
    let mut global_min = [1.0f64; 4];
    let mut any = false;
    for src in fixtures {
        let Some(yuv) = crop_yuv(src, W, H, N) else {
            eprintln!("{:<26} | SKIP", src.chars().take(26).collect::<String>());
            continue;
        };
        any = true;
        let mut fx_min = [1.0f64; 4];
        for g in 0..n_gops {
            let gy = &yuv[g * GOP as usize * fs..((g + 1) * GOP as usize).min(N as usize) * fs];
            let gf = (gy.len() / fs) as u32;
            let rep_cap = h264_video_capacity(gy, W, H, gf as usize, opts, false)
                .map(|i| i.per_tier_primary_max_message_bytes[0])
                .unwrap_or(0);
            if rep_cap < 64 {
                continue;
            }
            for (mi, &m) in margins.iter().enumerate() {
                let level = ((rep_cap as f64) * (1.0 - m)) as usize;
                let ok = (0..K)
                    .filter(|_| {
                        let msg = incompressible(level);
                        encode_bytes(gy, W, H, gf, GOP, &msg, pass)
                            .and_then(|b| decode_text(&b, pass))
                            .as_deref()
                            == Some(msg.as_str())
                    })
                    .count();
                let rate = ok as f64 / K as f64;
                fx_min[mi] = fx_min[mi].min(rate);
                global_min[mi] = global_min[mi].min(rate);
            }
        }
        eprintln!(
            "{:<26} | {:>4.0}% | {:>4.0}% | {:>4.0}% | {:>4.0}%",
            src.chars().take(26).collect::<String>(),
            fx_min[0] * 100.0, fx_min[1] * 100.0, fx_min[2] * 100.0, fx_min[3] * 100.0,
        );
    }
    eprintln!(
        "\nCORPUS worst-GOP success rate: 0%={:.0}%  5%={:.0}%  10%={:.0}%  15%={:.0}%",
        global_min[0] * 100.0, global_min[1] * 100.0, global_min[2] * 100.0, global_min[3] * 100.0,
    );
    let safe_margin = margins
        .iter()
        .zip(global_min.iter())
        .find(|pair| *pair.1 >= 1.0)
        .map(|pair| *pair.0);
    match safe_margin {
        Some(m) => eprintln!(
            "→ smallest derating margin with 100% worst-GOP round-trip = {:.0}% (CAP2.2 r_target \
             soft-ceiling cap + estimator Σ-derate).",
            m * 100.0
        ),
        None => eprintln!(
            "→ even 15% derating leaves round-trip failures — need a wider sweep or the J2 \
             floor is content-pathological; investigate before wiring proportional."
        ),
    }
    assert!(any, "no corpus fixtures available");
}

/// True max-encodable, galloping from `start` (the reported estimate, which
/// is close to true) so we spend ~a dozen full-length encodes instead of
/// exp-searching from scratch.
fn true_max_from(yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, start: usize) -> usize {
    let pass = "cap810";
    // ROUND-TRIP true: a size counts only if it encodes AND decodes back
    // (so the carry path's decodability is part of the measurement).
    let fits = |len: usize| -> bool {
        let msg = incompressible(len);
        match encode_bytes(yuv, w, h, n, gop, &msg, pass) {
            Some(bytes) => decode_text(&bytes, pass).as_deref() == Some(msg.as_str()),
            None => false,
        }
    };
    let start = start.max(1);
    if !fits(start) {
        // reported over-reports true → bisect down [0, start].
        let mut lo = 0usize;
        let mut hi = start;
        while lo + 8 < hi {
            let mid = (lo + hi) / 2;
            if fits(mid) { lo = mid; } else { hi = mid; }
        }
        return lo;
    }
    // reported fits → gallop up (1.5×) for a failing ceiling, then bisect.
    let mut lo = start;
    let mut hi = start + start / 2 + 64;
    while hi < (1 << 21) && fits(hi) {
        lo = hi;
        hi = hi + hi / 2 + 64;
    }
    while lo + 8 < hi {
        let mid = (lo + hi) / 2;
        if fits(mid) { lo = mid; } else { hi = mid; }
    }
    lo
}

#[test]
#[ignore]
fn cap_realworld_one() {
    let src = std::env::var("PHASM_CAP_SRC").unwrap_or_default();
    let tag = std::env::var("PHASM_CAP_TAG").unwrap_or_else(|_| src.clone());
    let w = env_u32("PHASM_CAP_W");
    let h = env_u32("PHASM_CAP_H");
    let n = env_u32("PHASM_CAP_N");
    let gop = env_u32("PHASM_CAP_GOP").max(1);
    if src.is_empty() || w == 0 || h == 0 || n == 0 {
        eprintln!("RESULT {:<14} SKIP (bad env)", tag);
        return;
    }
    let Some(yuv) = crop_yuv(&src, w, h, n) else {
        eprintln!("RESULT {:<14} SKIP (no fixture/ffmpeg)", tag);
        return;
    };
    let reported = h264_video_capacity(
        &yuv, w, h, n as usize, EncodeOpts { qp: 26, intra_period: gop as i32 }, false,
    ).map(|i| i.per_tier_primary_max_message_bytes[0]).unwrap_or(0);
    let true_max = true_max_from(&yuv, w, h, n, gop, reported);
    let ratio = if true_max > 0 { reported as f64 / true_max as f64 } else { f64::NAN };
    let n_gops = (n + gop - 1) / gop;
    eprintln!(
        "RESULT {:<14} {:>4}x{:<4} {:>4}f gop{:<3} {:>3}gops | reported={:>6}  true={:>6}  ratio={:.2}",
        tag, w, h, n, gop, n_gops, reported, true_max, ratio,
    );
}

/// #810 — iphone7 round-trip-vs-encode threshold. The re-baseline showed
/// iphone7 reports ~4591 B but only ~711 B round-trips (6.46× over). Sweep
/// FIXED payloads (one message per size, deterministic) to find the ENCODE
/// threshold (MessageTooLarge) vs the DECODE threshold (round-trip breaks),
/// separating cascade-leak (encode OK / decode fails) from capacity. 150f /
/// 5 GOPs for speed — the cascade leak is per-GOP (flip count), so the
/// per-GOP threshold is representative of the full clip.
#[test]
#[ignore]
fn iphone7_roundtrip_threshold() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 150;
    const GOP: u32 = 30;
    let Some(yuv) = crop_yuv("iphone7_1080p_30fps_h264_high.mov", W, H, N) else {
        eprintln!("SKIP (no iphone7 fixture)");
        return;
    };
    let reported = h264_video_capacity(
        &yuv, W, H, N as usize, EncodeOpts { qp: 26, intra_period: GOP as i32 }, false,
    )
    .map(|i| i.per_tier_primary_max_message_bytes[0])
    .unwrap_or(0);
    let n_gops = (N + GOP - 1) / GOP;
    eprintln!(
        "\n=== iphone7 {}x{} x{}f ({} gops) — CAP2.1 reported={} ===",
        W, H, N, n_gops, reported,
    );
    eprintln!("payload | B/gop | encode      | decode");
    for &p in &[50usize, 100, 150, 200, 300, 450, 650, 900, 1300, 1900, 2700] {
        let msg = incompressible(p);
        let (enc, dec) = match encode_bytes(&yuv, W, H, N, GOP, &msg, "ip7diag") {
            Some(bytes) => (
                "OK",
                match decode_text(&bytes, "ip7diag") {
                    Some(t) if t == msg => "OK",
                    Some(_) => "WRONG",
                    None => "DEC-ERR",
                },
            ),
            None => ("MsgTooLarge", "—"),
        };
        eprintln!("{:>7} | {:>5} | {:<11} | {}", p, p as u32 / n_gops, enc, dec);
    }
}

/// CASCADE.V2 A.1.5 — decision-gate bisect. Encode iphone7 ONE GOP (30
/// frames = full per-GOP cascade depth) across payloads bracketing the
/// ~260 B/GOP decode ceiling, with PHASM_PERF_TRACE=1 so the 4-domain
/// driver's [symmetry]/[selfextract]/[diff#] diagnostic fires per GOP.
///
/// Read the driver lines, NOT just enc/dec:
///   [selfextract] frame_bits_diff=0  on a DEC-ERR payload
///       => the encoder's own Pass-2 walk recovers the planted bits, so the
///          cover (C') is FAITHFUL and the failure is decoder/slab-side.
///          Encoder replay (A.2-A.8) would NOT fix it — redirect.
///   [selfextract] frame_bits_diff>0  on a DEC-ERR payload
///       => the emitted cover DIVERGED from the STC plan (the cascade).
///          [symmetry] per-domain p1-vs-p2 counts + [diff#] tag/mb/frame
///          name the first diverging block category = the recompute to
///          replay first. Validates/narrows the L1+L2+L3 architecture.
///
/// Run:
///   PHASM_PERF_TRACE=1 cargo test --test cap_realworld_810 \
///     --features h264-encoder,cabac-stego --release \
///     iphone7_cascade_bisect -- --ignored --nocapture
#[test]
#[ignore]
fn iphone7_cascade_bisect() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 30; // ONE GOP — full per-GOP cascade depth, isolates the leak
    const GOP: u32 = 30;
    let Some(yuv) = crop_yuv("iphone7_1080p_30fps_h264_high.mov", W, H, N) else {
        eprintln!("SKIP (no iphone7 fixture)");
        return;
    };
    eprintln!("\n=== A.1.5 iphone7 {W}x{H} x{N}f (1 GOP) cascade bisect ===");
    eprintln!("(set PHASM_PERF_TRACE=1 to see [selfextract]/[symmetry]/[diff#])");
    for &p in &[200usize, 320, 380, 440] {
        let msg = incompressible(p);
        eprintln!("\n######## A.1.5 payload={p} B (1 GOP) ########");
        let (enc, dec) = match encode_bytes(&yuv, W, H, N, GOP, &msg, "ip7casc") {
            Some(bytes) => (
                "OK",
                match decode_text(&bytes, "ip7casc") {
                    Some(t) if t == msg => "OK",
                    Some(_) => "WRONG",
                    None => "DEC-ERR",
                },
            ),
            None => ("MsgTooLarge", "—"),
        };
        eprintln!("######## A.1.5 payload={p}: encode={enc} decode={dec} ########");
    }
}

/// CASCADE.V2 A.1.5 — multi-GOP companion to `iphone7_cascade_bisect`.
/// The 1-GOP bisect proved the per-GOP encoder cover is FAITHFUL even at
/// 440 B/GOP (selfextract=0, decode OK). Yet the 5-GOP/150f sweep fails to
/// decode at 1900 B. This runs the real 5-GOP path at a passing (1300 B)
/// and the failing (1900 B) payload with PHASM_PERF_TRACE=1. If ALL 5 GOPs
/// show selfextract frame_bits_diff=0 on the FAILING payload, the encoder
/// is faithful per-GOP and the leak is 100% in the multi-GOP decode/slab
/// path (slab splitting / m_total brute-force / chunk reassembly), NOT an
/// encoder cascade. Run as for iphone7_cascade_bisect (name swapped).
#[test]
#[ignore]
fn iphone7_cascade_bisect_multigop() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 150; // 5 GOPs — the regime where decode actually fails
    const GOP: u32 = 30;
    let Some(yuv) = crop_yuv("iphone7_1080p_30fps_h264_high.mov", W, H, N) else {
        eprintln!("SKIP (no iphone7 fixture)");
        return;
    };
    let n_gops = (N + GOP - 1) / GOP;
    eprintln!("\n=== A.1.5 iphone7 {W}x{H} x{N}f ({n_gops} GOPs) multigop bisect ===");
    for &p in &[1300usize, 1900] {
        let msg = incompressible(p);
        eprintln!(
            "\n######## A.1.5 MULTIGOP payload={p} B (~{} B/GOP) ########",
            p as u32 / n_gops,
        );
        let (enc, dec) = match encode_bytes(&yuv, W, H, N, GOP, &msg, "ip7mg") {
            Some(bytes) => (
                "OK",
                match decode_text(&bytes, "ip7mg") {
                    Some(t) if t == msg => "OK",
                    Some(_) => "WRONG",
                    None => "DEC-ERR",
                },
            ),
            None => ("MsgTooLarge", "—"),
        };
        eprintln!("######## A.1.5 MULTIGOP payload={p}: encode={enc} decode={dec} ########");
    }
}

/// CASCADE.V2 A.1.5 — ROOT-CAUSE confirmation. The multigop bisect gave
/// OPPOSITE results on two identical runs (run1 DEC-ERR, run2 OK) and the
/// "clean" Pass-1 cover size varied run-to-run AND payload-to-payload —
/// impossible unless the OH264 encode is contaminated by leaked C-global
/// state across calls. This isolates that on the PRODUCTION path (NO trace,
/// so no diagnostic re-encodes churn state): encode the SAME clip + message
/// + passphrase TWICE in one process and diff the bytes. identical=false ⇒
/// the encoder is non-deterministic call-to-call → the multi-GOP decode
/// failure is intermittent corruption from incomplete state reset (#548
/// class), NOT a flip cascade and NOT a decoder bug.
///
/// Run (NO PHASM_PERF_TRACE):
///   cargo test --test cap_realworld_810 \
///     --features h264-encoder,cabac-stego --release \
///     iphone7_encode_determinism -- --ignored --nocapture
#[test]
#[ignore]
fn iphone7_encode_determinism() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 150;
    const GOP: u32 = 30;
    let Some(yuv) = crop_yuv("iphone7_1080p_30fps_h264_high.mov", W, H, N) else {
        eprintln!("SKIP (no iphone7 fixture)");
        return;
    };
    let msg = incompressible(1900);
    eprintln!("\n=== A.1.5 iphone7 encode determinism (2 identical encodes, one process) ===");
    let a = encode_bytes(&yuv, W, H, N, GOP, &msg, "ip7det");
    let b = encode_bytes(&yuv, W, H, N, GOP, &msg, "ip7det");
    match (a, b) {
        (Some(a), Some(b)) => {
            let same = a == b;
            let first = (0..a.len().min(b.len())).find(|&i| a[i] != b[i]);
            // FNV-1a so two SEPARATE process runs can be compared: if hash#1
            // is identical across processes, the FIRST encode is byte-stable
            // => non-determinism is pure within-process state leak (not
            // threading / uninitialized memory, which would vary hash#1 too).
            let fnv = |v: &[u8]| -> u64 {
                v.iter().fold(0xcbf29ce484222325u64, |h, &b| {
                    (h ^ b as u64).wrapping_mul(0x100000001b3)
                })
            };
            eprintln!(
                "encode#1 bytes={} encode#2 bytes={} IDENTICAL={} first_byte_diff={:?}",
                a.len(), b.len(), same, first,
            );
            eprintln!("hash#1={:016x} hash#2={:016x}", fnv(&a), fnv(&b));
            let da = decode_text(&a, "ip7det").as_deref() == Some(msg.as_str());
            let db = decode_text(&b, "ip7det").as_deref() == Some(msg.as_str());
            eprintln!("decode#1_ok={da} decode#2_ok={db}");
            eprintln!(
                "=== VERDICT: {} ===",
                if same { "encoder DETERMINISTic call-to-call" }
                else { "encoder NON-DETERMINISTIC call-to-call (state leak — root cause)" },
            );
        }
        _ => eprintln!("encode failed (MessageTooLarge or error)"),
    }
}

/// CASCADE.V2 A.1.6 — find a DETERMINISTIC repro of the intra-call cascade.
/// The encoder is deterministic (iphone7_encode_determinism with a fixed
/// seed => byte-identical, cross-process). The intermittent decode failures
/// are the random AES salt picking different STC flip positions; some flip
/// sets trip a payload-dependent intra-call cascade (Pass-1 plan != Pass-2
/// emit), some don't. Sweep deterministic crypto seeds at a fixed payload to
/// pin a seed that reproducibly FAILS — then re-run that seed with
/// PHASM_PERF_TRACE=1 to read [selfextract]/[symmetry]/[diff#] and bisect.
///
/// Run: cargo test --test cap_realworld_810 \
///   --features h264-encoder,cabac-stego --release \
///   iphone7_cascade_seed_sweep -- --ignored --nocapture
#[test]
#[ignore]
fn iphone7_cascade_seed_sweep() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 150;
    const GOP: u32 = 30;
    let Some(yuv) = crop_yuv("iphone7_1080p_30fps_h264_high.mov", W, H, N) else {
        eprintln!("SKIP (no iphone7 fixture)");
        return;
    };
    let msg = incompressible(1900);
    eprintln!("\n=== A.1.6 iphone7 5-GOP seed sweep @1900 B (find a reproducible cascade) ===");
    let mut fails = vec![];
    for seed in 1u64..=20 {
        // SAFETY: single-threaded test; set before each encode so the crypto
        // layer (crypto.rs:293) reads this seed instead of OsRng.
        unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", seed.to_string()); }
        let r = match encode_bytes(&yuv, W, H, N, GOP, &msg, "ip7seed") {
            Some(bytes) => match decode_text(&bytes, "ip7seed") {
                Some(t) if t == msg => "OK",
                Some(_) => "WRONG",
                None => "DEC-ERR",
            },
            None => "MsgTooLarge",
        };
        if r != "OK" { fails.push(seed); }
        eprintln!("seed={seed:>3} payload=1900 => {r}");
    }
    eprintln!("=== failing seeds: {fails:?} (use one with PHASM_PERF_TRACE=1 to bisect) ===");
}

/// #814 — per-GOP over-report diagnostic (asia_bottle, 1 GOP). The
/// post-CASCADE.V2 re-sweep found CAP2.1 OVER-reports asia_bottle
/// (reported 3550 > true round-trip 3293, ratio 1.08). This isolates WHY
/// on the per-GOP unit (fast): prints all 5 per-tier CAP2.1 capacities,
/// then the REAL round-trip success rate over K random-salt encodes at
/// sizes bracketing the tier-0 report, under BOTH auto-tier and forced
/// tier-0. Reading:
///   - FORCED-tier0 fails below the tier-0 report  => fixed-seed/message
///     jitter (J): the CAP2_TRIAL_SEED probe (×5/6) over-estimates THIS
///     passphrase's real embeddable. Root-cause fix = §15 passphrase-hhat
///     probe (drop the representative seed + derating).  [== #811 item 3]
///   - AUTO fails where FORCED-tier0 succeeds       => auto_select_tier
///     picks a higher (lower-capacity) tier near the margin (S1): a
///     SEPARATE structural bug (raw-cover-bit proxy vs STC payload).
///
/// Run:
///   cargo test --test cap_realworld_810 \
///     --features h264-encoder,cabac-stego --release \
///     asia_bottle_overreport_diag -- --ignored --nocapture
#[test]
#[ignore]
fn asia_bottle_overreport_diag() {
    const W: u32 = 1072;
    const H: u32 = 1920;
    const N: u32 = 30; // 1 GOP — isolates the per-GOP cap fast
    const GOP: u32 = 30;
    const K: usize = 10; // random-salt encodes per size
    let Some(yuv) = crop_yuv("Artlist_AsiaBottle.mp4", W, H, N) else {
        eprintln!("SKIP asia_bottle_overreport_diag (no fixture)");
        return;
    };
    let info = h264_video_capacity(
        &yuv, W, H, N as usize, EncodeOpts { qp: 26, intra_period: GOP as i32 }, false,
    )
    .expect("capacity");
    let per_tier = info.per_tier_primary_max_message_bytes;
    eprintln!("\n=== #814 asia_bottle 1-GOP per-tier CAP2.1 (reported msg bytes, incl ×5/6 derating) ===");
    for (t, &v) in per_tier.iter().enumerate() {
        eprintln!("  tier {t}: {v} B");
    }
    let cap0 = per_tier[0].max(1);
    let pass = "cap810";

    // K random-salt round-trips at `len` → OK count (msg is fixed per len;
    // only the AES salt/nonce/ct vary call-to-call = message-bit jitter J2).
    let rate = |len: usize| -> usize {
        let msg = incompressible(len);
        (0..K)
            .filter(|_| {
                encode_bytes(&yuv, W, H, N, GOP, &msg, pass)
                    .and_then(|b| decode_text(&b, pass))
                    .as_deref()
                    == Some(msg.as_str())
            })
            .count()
    };

    let base = cap0 as i64;
    let sizes: Vec<usize> = [-200i64, -120, -60, -20, 0, 40, 100]
        .iter()
        .map(|d| (base + d).max(1) as usize)
        .collect();

    for (label, force0) in [("AUTO-tier", false), ("FORCED tier0", true)] {
        // SAFETY: single-threaded test; env read inside the encode path.
        if force0 {
            unsafe { std::env::set_var("PHASM_TIER_OVERRIDE", "0"); }
        } else {
            unsafe { std::env::remove_var("PHASM_TIER_OVERRIDE"); }
        }
        eprintln!("\n--- {label}: round-trip OK/{K} over random salts (tier-0 report cap0={cap0} B) ---");
        for &s in &sizes {
            eprintln!("  {:>5} B : {:>2}/{}", s, rate(s), K);
        }
    }
    unsafe { std::env::remove_var("PHASM_TIER_OVERRIDE"); }
}

/// #814 — full-clip (production auto-tier) round-trip success-RATE sweep.
/// The single-bisect `true` in `cap_realworld_one` is severely stochastic
/// (random AES salt per `fits()` → message-bit jitter): the post-CASCADE.V2
/// re-sweep galloped to 3293 for asia_bottle, a fresh run to 4734 — SAME
/// auto measurement, 1.4× apart. This replaces the noisy point estimate
/// with a success RATE over K random salts at fixed sizes bracketing the
/// reported capacity, on the FULL clip + production auto-tier — the honest
/// "does the report over-promise?" gate. A reported value that round-trips
/// K/K is safe; <K/K at or below reported is a real over-report.
///
/// Run:
///   cargo test --test cap_realworld_810 \
///     --features h264-encoder,cabac-stego --release \
///     asia_bottle_fullclip_success_rate -- --ignored --nocapture
#[test]
#[ignore]
fn asia_bottle_fullclip_success_rate() {
    const W: u32 = 1072;
    const H: u32 = 1920;
    const N: u32 = 145; // full clip, 5 GOPs — production aggregate (carry)
    const GOP: u32 = 30;
    const K: usize = 12; // random-salt encodes per size
    let Some(yuv) = crop_yuv("Artlist_AsiaBottle.mp4", W, H, N) else {
        eprintln!("SKIP asia_bottle_fullclip_success_rate (no fixture)");
        return;
    };
    let reported = h264_video_capacity(
        &yuv, W, H, N as usize, EncodeOpts { qp: 26, intra_period: GOP as i32 }, false,
    )
    .map(|i| i.per_tier_primary_max_message_bytes[0])
    .unwrap_or(0);
    eprintln!("\n=== #814 asia_bottle FULL-CLIP auto-tier round-trip success rate (reported={reported} B) ===");
    let pass = "cap810";
    let rate = |len: usize| -> usize {
        let msg = incompressible(len);
        (0..K)
            .filter(|_| {
                encode_bytes(&yuv, W, H, N, GOP, &msg, pass)
                    .and_then(|b| decode_text(&b, pass))
                    .as_deref()
                    == Some(msg.as_str())
            })
            .count()
    };
    let r = reported as i64;
    let sizes: Vec<usize> = [-600i64, -300, -100, 0, 200, 600, 1000, 1400]
        .iter()
        .map(|d| (r + d).max(1) as usize)
        .collect();
    eprintln!("  payload | round-trip OK / {K}  (reported={reported})");
    for &s in &sizes {
        let tag = if s as i64 == r { "  <- reported" } else { "" };
        eprintln!("  {:>6} B : {:>2}/{}{}", s, rate(s), K, tag);
    }
}

/// #814 — CRITICAL: split graceful MessageTooLarge from SILENT decode
/// failure. The full-clip success-rate sweep showed reported=3550 only
/// round-trips 4/12 (reliable ceiling ~3250). This decides whether those
/// misses are GRACEFUL (encode refuses → MessageTooLarge → user shortens,
/// no data loss) or SILENT CORRUPTION (encode OK but decode wrong/err →
/// the user loses their message = a CASCADE.V2 residual correctness bug,
/// e.g. the still-pending #812 CoeffSuffixLsb ctxIdxInc desync at the
/// ~710 B/GOP regime, well above A.1.10's ~380 B/GOP fix point).
///
/// Run:
///   cargo test --test cap_realworld_810 \
///     --features h264-encoder,cabac-stego --release \
///     asia_bottle_enc_dec_split -- --ignored --nocapture
#[test]
#[ignore]
fn asia_bottle_enc_dec_split() {
    const W: u32 = 1072;
    const H: u32 = 1920;
    const N: u32 = 145;
    const GOP: u32 = 30;
    const K: usize = 12;
    let Some(yuv) = crop_yuv("Artlist_AsiaBottle.mp4", W, H, N) else {
        eprintln!("SKIP asia_bottle_enc_dec_split (no fixture)");
        return;
    };
    let pass = "cap810";
    eprintln!("\n=== #814 asia_bottle enc/dec split (auto-tier, K={K} random salts) ===");
    eprintln!("  payload | MsgTooLarge (graceful) | DECODE-FAIL (SILENT LOSS) | OK");
    for &s in &[3050usize, 3250, 3450, 3550, 3750, 4150] {
        let msg = incompressible(s);
        let (mut too_large, mut dec_fail, mut ok) = (0usize, 0usize, 0usize);
        for _ in 0..K {
            match encode_bytes(&yuv, W, H, N, GOP, &msg, pass) {
                None => too_large += 1,
                Some(b) => match decode_text(&b, pass) {
                    Some(t) if t == msg => ok += 1,
                    _ => dec_fail += 1,
                },
            }
        }
        let flag = if dec_fail > 0 { "  <== SILENT LOSS" } else { "" };
        eprintln!(
            "  {:>6} B | {:>2} | {:>2} | {:>2}{}",
            s, too_large, dec_fail, ok, flag,
        );
    }
}

/// #814 / §15 regression-gate probe — CORPUS-WIDE enc/dec split. Env-driven
/// (`PHASM_CAP_SRC/TAG/W/H/N/GOP`) so `cap_enc_dec_split.sh` fans out one
/// process per fixture (OH264 C-global state ⇒ cross-process only). For each
/// fixture: probe reported capacity, then sweep sizes around it with K
/// random salts, classifying each encode as MsgTooLarge (graceful) /
/// DECODE-FAIL (SILENT LOSS) / OK. The headline `SILENT_MAX` column is the
/// max DECODE-FAIL over all sizes: any nonzero = a CASCADE.V2 residual
/// (encode>decode) for that fixture = correctness bug. All-zero corpus-wide
/// confirms §15's premise — the cascade is closed, so capacity over-reports
/// are always graceful, never data loss.
#[test]
#[ignore]
fn cap_enc_dec_split_one() {
    let src = std::env::var("PHASM_CAP_SRC").unwrap_or_default();
    let tag = std::env::var("PHASM_CAP_TAG").unwrap_or_else(|_| src.clone());
    let w = env_u32("PHASM_CAP_W");
    let h = env_u32("PHASM_CAP_H");
    let n = env_u32("PHASM_CAP_N");
    let gop = env_u32("PHASM_CAP_GOP").max(1);
    const K: usize = 8;
    if src.is_empty() || w == 0 || h == 0 || n == 0 {
        eprintln!("RESULT {tag:<14} SKIP (bad env)");
        return;
    }
    let Some(yuv) = crop_yuv(&src, w, h, n) else {
        eprintln!("RESULT {tag:<14} SKIP (no fixture/ffmpeg)");
        return;
    };
    let reported = h264_video_capacity(
        &yuv, w, h, n as usize, EncodeOpts { qp: 26, intra_period: gop as i32 }, false,
    )
    .map(|i| i.per_tier_primary_max_message_bytes[0])
    .unwrap_or(0);
    let pass = "cap810";
    let mut silent_max = 0usize;
    let mut detail = String::new();
    for &frac in &[0.85f64, 0.95, 1.00, 1.10, 1.25] {
        let s = ((reported as f64 * frac) as usize).max(1);
        let msg = incompressible(s);
        let (mut tl, mut df, mut ok) = (0usize, 0usize, 0usize);
        for _ in 0..K {
            match encode_bytes(&yuv, w, h, n, gop, &msg, pass) {
                None => tl += 1,
                Some(b) => match decode_text(&b, pass) {
                    Some(t) if t == msg => ok += 1,
                    _ => df += 1,
                },
            }
        }
        silent_max = silent_max.max(df);
        detail.push_str(&format!(" {s}:{tl}/{df}/{ok}"));
    }
    // Columns per size: MsgTooLarge/DECODE-FAIL/OK (out of K).
    eprintln!(
        "RESULT {tag:<14} reported={reported:>6} SILENT_MAX={silent_max} K={K} |{detail}",
    );
}

/// #815/#816 — per-domain cascade stress. Concentrates STC flips into ONE
/// bypass-bin domain via finite 50× cost bias (NOT ∞: pure ∞-isolation makes
/// the syndrome trellis unembeddable — 3 ∞ domains leave all-∞ trellis
/// windows, so nothing encodes and no decode is tested). 50× bias fills the
/// target domain FIRST — maxing the sparse magnitude domains (|coeff|≥15,
/// |mvd|≥9) completely — then spills, so high payloads still embed. Sweeps
/// payload over K random salts on the cascade-prone iphone7 fixture (the
/// A.1.10 trigger), classifying each encode MsgTooLarge / DECODE-FAIL (silent
/// loss) / OK. Any DECODE-FAIL with flips concentrated in domain D ⇒ a
/// residual cascade in D. The four domains:
///   - CoeffSign / MvdSign: signs feed NO CABAC ctxIdxInc → expected clean.
///   - CoeffSuffixLsb: |level|≥15, EG length flip-invariant → expected clean
///     (the #812 structural argument; this is its empirical confirmation).
///   - MvdSuffixLsb: |mvd| feeds next-MB mvd ctxIdxInc (threshold 32) →
///     A.1.10-FIXED; this re-confirms it at max flip density.
/// Headline: any DECODE-FAIL > 0 = a residual cascade in THAT domain.
///
/// Run:
///   cargo test --test cap_realworld_810 \
///     --features h264-encoder,cabac-stego --release \
///     per_domain_cascade_stress -- --ignored --nocapture
#[test]
#[ignore]
fn per_domain_cascade_stress() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 150; // 5 GOPs — the multi-GOP regime where A.1.10 bit
    const GOP: u32 = 30;
    const K: usize = 8;
    let Some(yuv) = crop_yuv("iphone7_1080p_30fps_h264_high.mov", W, H, N) else {
        eprintln!("SKIP per_domain_cascade_stress (no iphone7 fixture)");
        return;
    };
    let pass = "casc_stress";
    // Finite 50× bias (NOT ∞): STC fills the target domain FIRST — maxing the
    // sparse magnitude domains (|coeff|≥15, |mvd|≥9) completely — then spills
    // to the others, so high payloads still embed. (Pure ∞-isolation makes the
    // syndrome trellis unembeddable: 3 ∞ domains leave all-∞ trellis windows.)
    let b = |cs: f32, csl: f32, ms: f32, msl: f32| CostWeights {
        coeff_sign: cs, coeff_suffix: csl, mvd_sign: ms, mvd_suffix: msl,
    };
    let domains: [(&str, CostWeights); 4] = [
        ("CoeffSign     ", b(1.0, 50.0, 50.0, 50.0)),
        ("CoeffSuffixLsb", b(50.0, 1.0, 50.0, 50.0)),
        ("MvdSign       ", b(50.0, 50.0, 1.0, 50.0)),
        ("MvdSuffixLsb  ", b(50.0, 50.0, 50.0, 1.0)),
    ];
    // Sweep spans the small MVD-domain ceilings and the larger coeff ceilings.
    let sizes = [40usize, 100, 250, 600, 1200, 2400];
    let mut any_silent = false;
    for (name, w) in domains {
        eprintln!("\n--- domain {name} (all flips forced here) ---");
        eprintln!("  payload | MsgTooLarge | DECODE-FAIL | OK  (/{K})");
        for &s in &sizes {
            let msg = incompressible(s);
            let (mut tl, mut df, mut ok) = (0usize, 0usize, 0usize);
            for _ in 0..K {
                match encode_bytes_w(&yuv, W, H, N, GOP, &msg, pass, w) {
                    None => tl += 1,
                    Some(b) => match decode_text(&b, pass) {
                        Some(t) if t == msg => ok += 1,
                        _ => df += 1,
                    },
                }
            }
            if df > 0 { any_silent = true; }
            let flag = if df > 0 { "  <== SILENT LOSS" } else { "" };
            eprintln!("  {:>6} B | {:>2} | {:>2} | {:>2}{}", s, tl, df, ok, flag);
        }
    }
    assert!(
        !any_silent,
        "per-domain cascade stress: a domain produced silent decode failure (encode OK, decode wrong) — residual cascade",
    );
    eprintln!("\n=== per_domain_cascade_stress: ALL DOMAINS CLEAN (zero silent loss) ===");
}

/// #763 — OH264 bypass-bin entropy re-baseline (PRODUCTION path). The
/// existing `h264_stego_stealth_measurement` measures these per-domain
/// stealth metrics via the pure-Rust encoder, which is currently
/// #802-blocked (`MessageTooLarge` on tiny covers). This re-baselines them
/// on the OH264 streaming path that actually ships: per domain, the clean
/// cover bits (`h264_walk_cover`) vs the walked stego bits —
/// flip rate + Shannon entropy. The stego carrier (bypass bins) must stay
/// ~uniform and look like the clean cover (ΔH≈0) so the modifications are
/// statistically undetectable at the bin level. Confirms the context-only
/// cascade fix (A.1.10) did not perturb bin statistics.
///
/// Run:
///   cargo test --test cap_realworld_810 \
///     --features h264-encoder,cabac-stego --release \
///     oh264_bypass_entropy_rebaseline -- --ignored --nocapture
#[test]
#[ignore]
fn oh264_bypass_entropy_rebaseline() {
    const GOP: u32 = 30;
    // FULL-LENGTH native-res real-world fixtures (16-aligned crop of the
    // originals, NOT downscaled/truncated): (file, w, h, full_n_frames).
    let fixtures: [(&str, u32, u32, u32); 3] = [
        ("iphone7_1080p_30fps_h264_high.mov", 1920, 1072, 510), // 17 GOPs
        ("Artlist_AsiaBottle.mp4", 1072, 1920, 145),            //  5 GOPs
        ("dji_mini2_2_7k_24fps_h264_high.mp4", 2720, 1520, 189),//  7 GOPs
    ];
    fn entropy(bits: &[u8]) -> f64 {
        if bits.is_empty() {
            return 0.0;
        }
        let ones = bits.iter().filter(|&&b| b != 0).count() as f64;
        let n = bits.len() as f64;
        let p1 = ones / n;
        let p0 = 1.0 - p1;
        let t = |p: f64| if p > 0.0 { -p * p.log2() } else { 0.0 };
        t(p0) + t(p1)
    }
    let mut ran_any = false;
    let mut worst_min_h = 1.0f64;
    let mut failures: Vec<String> = Vec::new();
    for (file, w, h, n) in fixtures {
        let Some(yuv) = crop_yuv(file, w, h, n) else {
            eprintln!("SKIP {file} (no fixture)");
            continue;
        };
        ran_any = true;
        let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
        let clean = h264_walk_cover(&yuv, w, h, n, opts).expect("clean cover");
        let reported = h264_video_capacity(&yuv, w, h, n as usize, opts, false)
            .map(|i| i.per_tier_primary_max_message_bytes[0])
            .unwrap_or(0);
        let payload = (reported * 4 / 5).max(16); // 80% of reported → comfortably fits
        let msg = incompressible(payload);
        let pass = "stealth_rebase";
        // Must encode on the PRODUCTION PLANNED path.
        //
        // `reported` is the Σ-based number (`Σ per-GOP cap − FRAME_OVERHEAD`),
        // and Σ is reachable ONLY when an allocation plan is installed — what
        // the CLI and both mobile bridges do via `plan_proportional`. The bare
        // `encode_bytes` helper installs no plan, so it runs uniform-share +
        // carry, whose ceiling is far below Σ on heterogeneous content. That
        // mismatch — not any capacity inaccuracy — is why this test sat red:
        // iphone7's weakest GOP holds 320 B across 17 GOPs, so the unplanned
        // path cannot take 12132 B, while the planned path takes the very same
        // 12132 B 8/8 over random salts
        // (`cap_reported_capacity_success_rate_on_the_cli_planner`).
        let caps = match probe_per_gop_caps(&yuv, w, h, n, GOP, None) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{file}: PROBE FAILED: {e}");
                failures.push(format!("{file}: probe failed: {e}"));
                continue;
            }
        };
        let sigma: usize = caps.iter().sum();
        let min_cap = caps.iter().copied().min().unwrap_or(0);
        eprintln!(
            "{file}: reported={reported} B, payload={payload} B (80%), \
             Σcaps={sigma}, min cap={min_cap}, {} GOPs",
            caps.len(),
        );
        let stego_bytes = match encode_bytes_with_plan(
            &yuv, w, h, n, GOP, &msg, pass, &caps, CAP2_DEFAULT_R_TARGET,
        ) {
            Ok((b, true)) => b,
            Ok((_, false)) => {
                // `plan_proportional` degrades silently to uniform spread; if
                // that happens the measurement below would be of a different
                // path than intended, so refuse it rather than report it.
                eprintln!("{file}: NO PLAN INSTALLED");
                failures.push(format!(
                    "{file}: plan_proportional declined the plan (payload {payload} B vs \
                     Σcaps {sigma}) — refusing to measure the degraded path",
                ));
                continue;
            }
            Err(e) => {
                // Collect and continue rather than panicking: a first-fixture
                // panic hid every later fixture, which is how this went
                // un-diagnosed. J2 message/salt jitter (§16) can still refuse
                // a given draw near the margin; at 80% every corpus fixture
                // measured 8/8, so a failure here is a real regression.
                eprintln!("{file}: ENCODE FAILED: {e}");
                failures.push(format!(
                    "{file}: encode failed ({e}); payload {payload} B, \
                     reported {reported} B, Σcaps {sigma}, min cap {min_cap}",
                ));
                continue;
            }
        };
        if decode_text(&stego_bytes, pass).as_deref() != Some(msg.as_str()) {
            eprintln!("{file}: ROUND-TRIP FAILED");
            failures.push(format!("{file}: stego round-trip mismatch"));
            continue;
        }
        let stego = walk_annex_b_for_cover_with_options(
            &stego_bytes,
            WalkOptions { record_mvd: true, ..Default::default() },
        )
        .expect("stego walk")
        .cover;

        let n_gops = n.div_ceil(GOP);
        eprintln!(
            "\n=== #763 OH264 bypass-entropy (FULL-LENGTH {file} {w}×{h}×{n}f = {n_gops} GOPs, payload {payload} B) ===",
        );
        eprintln!("  domain             |      n | flips |  rate | H_clean | H_stego |     ΔH");
        let domains: [(&str, &Vec<u8>, &Vec<u8>); 4] = [
            ("coeff_sign_bypass ", &clean.coeff_sign_bypass.bits, &stego.coeff_sign_bypass.bits),
            ("coeff_suffix_lsb  ", &clean.coeff_suffix_lsb.bits, &stego.coeff_suffix_lsb.bits),
            ("mvd_sign_bypass   ", &clean.mvd_sign_bypass.bits, &stego.mvd_sign_bypass.bits),
            ("mvd_suffix_lsb    ", &clean.mvd_suffix_lsb.bits, &stego.mvd_suffix_lsb.bits),
        ];
        let mut min_h_stego = 1.0f64;
        for (name, c, s) in domains {
            let cn = c.len().min(s.len());
            if cn == 0 {
                eprintln!("  {name} | (empty)");
                continue;
            }
            let flips = (0..cn).filter(|&i| c[i] != s[i]).count();
            let (hc, hs) = (entropy(&c[..cn]), entropy(&s[..cn]));
            eprintln!(
                "  {name} | {cn:6} | {flips:5} | {:.3} | {hc:.4}  | {hs:.4}  | {:+.4}",
                flips as f64 / cn as f64,
                hs - hc,
            );
            if cn >= 200 {
                min_h_stego = min_h_stego.min(hs);
            }
        }
        worst_min_h = worst_min_h.min(min_h_stego);
    }
    assert!(ran_any, "no fixtures available");
    assert!(
        failures.is_empty(),
        "#763 rebaseline: {} full-length fixture(s) failed:\n  {}",
        failures.len(),
        failures.join("\n  "),
    );
    // Collapse sanity floor across all fixtures: a well-sampled stego bypass
    // domain must stay near-uniform (a real leak would skew the bin marginal).
    // Tight stealth thresholds live in the (#802-blocked pure-Rust) harness.
    assert!(
        worst_min_h >= 0.90,
        "a well-sampled stego bypass domain collapsed to H={worst_min_h:.4} (<0.90) — detectable bin skew",
    );
    eprintln!("\n  => every well-sampled stego bypass domain ≥0.90 bits (near-uniform) across all FULL-LENGTH fixtures — no detectable bin skew");
}

/// #809 — profile the per-GOP capacity probe (the live mobile path). Times
/// the full `h264_video_capacity` vs `h264_walk_cover`
/// (encode+walk only, no STC trials) on iphone7 at a few realistic lengths,
/// so the STC-trial share (the part the K-message floor multiplies, and the
/// 5-tier-redundancy / parallelism wins target) is visible. Grounds #809
/// before any optimization.
///
/// Run:
///   cargo test --test cap_realworld_810 \
///     --features h264-encoder,cabac-stego --release \
///     probe_perf_profile -- --ignored --nocapture
#[test]
#[ignore]
fn probe_perf_profile() {
    use std::time::Instant;
    const W: u32 = 1920;
    const H: u32 = 1072;
    eprintln!("\n=== #809 probe perf profile (iphone7 {W}×{H}, qp=26, gop=30) ===");
    eprintln!("  frames | GOPs | encode+walk | full probe | STC-trial share | reported");
    for &n in &[60u32, 150, 300] {
        let Some(yuv) = crop_yuv("iphone7_1080p_30fps_h264_high.mov", W, H, n) else {
            eprintln!("SKIP (no iphone7 fixture)");
            return;
        };
        let opts = EncodeOpts { qp: 26, intra_period: 30 };
        let t0 = Instant::now();
        let _ = h264_walk_cover(&yuv, W, H, n, opts).expect("count_cover");
        let t_enc = t0.elapsed();
        let t1 = Instant::now();
        let info = h264_video_capacity(&yuv, W, H, n as usize, opts, false).expect("probe");
        let t_probe = t1.elapsed();
        let stc = t_probe.saturating_sub(t_enc);
        eprintln!(
            "  {:>6} | {:>4} | {:>10.2?} | {:>10.2?} | {:>14.2?} | {}",
            n,
            n / 30,
            t_enc,
            t_probe,
            stc,
            info.primary_max_message_bytes,
        );
    }
    eprintln!("  (encode+walk ≈ 1 whole-video encode; STC-trial share = 5 tiers × binary-search × Viterbi per GOP)");
}

/// PERMANENT regression gate for CASCADE.V2 §A.1.10 — the MvdSuffixLsb
/// `ctxIdxInc` desync (a wire_only MVD-suffix override shifted the decoded
/// |MVD| ±1, which feeds the neighbour mvd bin0 ctxIdxInc; the encoder kept
/// the clean magnitude in pCurMb->sMvd[] → CABAC context desync → the
/// iphone7 "cascade ceiling": decode failed above ~260 B/GOP). Fix lives in
/// the OH264 fork (svc_set_mb_syn_cabac.cpp WelsCabacMbMvd).
///
/// These deterministic seeds reproducibly FAILED pre-fix (8/20 of seeds
/// 1..20 at this exact config). With the fix they all round-trip. NOT
/// `#[ignore]` — this must stay green. Skips gracefully without the fixture
/// (e.g. CI mirrors without test-vectors).
///
/// NOTE: the OH264 encoder is deterministic *per platform*; the specific
/// triggering seeds are macOS-arm64 values. The fix is content-agnostic, so
/// any platform where these decode confirms the cascade is closed; if ported
/// to a platform where none of these happen to trigger, widen the sweep.
#[test]
fn iphone7_mvd_ctxidxinc_regression_gate() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 150; // 5 GOPs — the multi-GOP regime where the desync bit
    const GOP: u32 = 30;
    let Some(yuv) = crop_yuv("iphone7_1080p_30fps_h264_high.mov", W, H, N) else {
        eprintln!("SKIP iphone7_mvd_ctxidxinc_regression_gate (no fixture)");
        return;
    };
    let msg = incompressible(1900); // ~380 B/GOP — was DEC-ERR pre-fix
    // SAFETY: this is the only non-ignored test in this binary, so the
    // default `cargo test` run executes it alone (no parallel sibling races
    // on PHASM_DETERMINISTIC_SEED).
    for seed in [7u64, 9, 17, 20] {
        unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", seed.to_string()); }
        let bytes = encode_bytes(&yuv, W, H, N, GOP, &msg, "ip7gate")
            .unwrap_or_else(|| panic!("seed {seed}: encode failed (MessageTooLarge?)"));
        let got = decode_text(&bytes, "ip7gate");
        assert_eq!(
            got.as_deref(), Some(msg.as_str()),
            "CASCADE.V2 regression: iphone7 seed={seed} @1900B failed to round-trip — \
             the MvdSuffixLsb ctxIdxInc desync (§A.1.10) has returned",
        );
    }
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED"); }
}

/// §15.3 + §16.3 — the ratified regression gate that still did not exist: a
/// success **RATE** at the reported capacity, on the CLI planned path.
///
/// Scope: this measures `plan_proportional`, which is what
/// `core/cli/src/video_encode.rs:563` installs. The mobile apps install
/// `plan_safe_balanced` (#841) instead — see
/// `cap_reported_capacity_success_rate_on_the_app_planner` below for that
/// surface. The two allocators differ in calibration and in whether any
/// pre-encode headroom is required, so neither gate substitutes for the other.
///
/// Why a rate and not one round-trip. §16 established that STC embeddability
/// near a GOP's cap is a random variable over the AES salt (random salt →
/// random nonce/ct → random message bits → random whether the min-cost
/// embedding path is forced through an ∞-cost position). That is the **J2**
/// axis, and it is salt-, not passphrase-driven: `CabacStegoMasterKeys::derive`
/// is passphrase-only Argon2id, so hhat is salt-independent. Consequences the
/// doc spells out and this harness obeys:
///
///  - A single bisect is too noisy to call a ceiling — the SAME asia_bottle
///    measurement gave 3293 and 4734 (1.4× apart), and success rate is
///    non-monotonic in size (1690 B → 4/10 but 1730/1790 B → 10/10).
///    So: K salts per size, report the rate.
///  - Over-reports are GRACEFUL by construction (cascade audit §A.1.11,
///    13/13 `SILENT_MAX=0` corpus-wide). The INVARIANT worth asserting is
///    therefore `DECODE-FAIL == 0`, never a capacity ratio.
///  - The reported number is an intentional **upper bound** (§17.0: exactness
///    is an encode-time invariant, not a display one; `CapacityProbeResult::
///    primary_max_message_bytes` is documented as such and the app renders it
///    with the #807 upper-bound framing). A size that refuses with
///    `MessageTooLarge` is the design working, not a defect.
///
/// So this gate is calibration data for #806 (does clip-level reliability need
/// a per-GOP K-message floor, per §17.1's "verify first") plus the one hard
/// correctness assertion. Columns per size: MsgTooLarge / DECODE-FAIL / OK.
///
/// Probes once per fixture and re-plans per trial (caps don't depend on the
/// message), so the cost is K encodes per size, not K probe+encodes.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_reported_capacity_success_rate -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_reported_capacity_success_rate_on_the_cli_planner() {
    const GOP: u32 = 30;
    const K: usize = 8;
    let fixtures: [(&str, u32, u32, u32); 3] = [
        ("iphone7_1080p_30fps_h264_high.mov", 1920, 1072, 510), // 17 GOPs
        ("Artlist_AsiaBottle.mp4", 1072, 1920, 145),            //  5 GOPs
        ("dji_mini2_2_7k_24fps_h264_high.mp4", 2720, 1520, 189),//  7 GOPs
    ];
    let pass = "cap_rate";
    let mut silent_loss: Vec<String> = Vec::new();
    let mut rows: Vec<String> = Vec::new();

    for (file, w, h, n) in fixtures {
        let Some(yuv) = crop_yuv(file, w, h, n) else {
            eprintln!("SKIP {file} (no fixture)");
            continue;
        };
        let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
        let reported = h264_video_capacity(&yuv, w, h, n as usize, opts, false)
            .map(|i| i.per_tier_primary_max_message_bytes[0])
            .unwrap_or(0);
        let caps = match probe_per_gop_caps(&yuv, w, h, n, GOP, None) {
            Ok(c) => c,
            Err(e) => {
                silent_loss.push(format!("{file}: probe failed: {e}"));
                continue;
            }
        };
        let sigma: usize = caps.iter().sum();
        let min_cap = caps.iter().copied().min().unwrap_or(0);
        let n_gops = caps.len();
        eprintln!(
            "\n=== {file} {w}×{h}×{n}f = {n_gops} GOPs ===\n  \
             reported(Σ-based upper bound) = {reported} B | Σcaps = {sigma} | \
             min cap = {min_cap} | n×min = {} (the UNPLANNED uniform ceiling)",
            n_gops * min_cap,
        );
        eprintln!("  {:>5} | {:>8} | {:>3} {:>3} {:>3}  (of K={K})", "frac", "bytes", "TL", "DF", "OK");

        for &frac in &[1.00f64, 0.95, 0.90, 0.80] {
            // `reported` is a FRAMED-byte budget, so pick the raw length that
            // frames to the target rather than using it as a raw length (see
            // `raw_len_for_framed` — doing the latter understated every size
            // here by ~17%).
            let target_framed = ((reported as f64 * frac) as usize).max(16);
            let len = raw_len_for_framed(target_framed, w, h, n, GOP, pass).max(16);
            let framed = framed_len_of(len, w, h, n, GOP, pass).unwrap_or(0);
            let msg = incompressible(len);
            let (mut tl, mut df, mut ok) = (0usize, 0usize, 0usize);
            let mut degraded = false;
            for _ in 0..K {
                match encode_bytes_with_plan(
                    &yuv, w, h, n, GOP, &msg, pass, &caps, CAP2_DEFAULT_R_TARGET,
                ) {
                    Err(_) => tl += 1, // MessageTooLarge (or any refusal) — graceful
                    Ok((b, installed)) => {
                        if !installed {
                            degraded = true;
                        }
                        if decode_text(&b, pass).as_deref() == Some(msg.as_str()) {
                            ok += 1;
                        } else {
                            df += 1;
                        }
                    }
                }
            }
            eprintln!(
                "  {:>4.0}% | {framed:>8} | {tl:>3} {df:>3} {ok:>3}   (raw {len}){}",
                frac * 100.0,
                if degraded { "   [NO PLAN — degraded to uniform]" } else { "" },
            );
            rows.push(format!(
                "{file} @{:.0}% (framed {framed}): TL={tl} DF={df} OK={ok}",
                frac * 100.0,
            ));
            if df > 0 {
                silent_loss.push(format!(
                    "{file} @{:.0}% (framed {framed} B): {df}/{K} SILENT DECODE FAILURES \
                     — encode accepted a message the decoder could not recover",
                    frac * 100.0,
                ));
            }
        }
    }

    eprintln!("\n=== summary ===");
    for r in &rows {
        eprintln!("  {r}");
    }
    // The ONE invariant: over-reports must be graceful refusals, never silent
    // loss. Capacity ratios are calibration data (#806), not pass/fail.
    assert!(
        silent_loss.is_empty(),
        "silent data loss on the CLI planned path:\n  {}",
        silent_loss.join("\n  "),
    );
}

/// The same §15.3 success-rate measurement on the planner the **apps** use.
///
/// `cap_reported_capacity_success_rate_on_the_cli_planner` above measures
/// `plan_proportional`. Neither shipping app calls it: iOS
/// `PhasmStegoTranscoder.swift:552` and Android `PhasmStegoTranscoder.kt:207`
/// both call `plan_safe_balanced` (#841) with
/// `AllocationCalibration::H264_1080P_QP26`. So the number a phone user is
/// shown has never been measured against the allocator that phone runs — this
/// gate is that measurement.
///
/// Three things it is built to separate, because they are three different
/// user experiences at the same displayed capacity:
///
///  - **PRE** — the planner refuses before encoding (its `safety_margin = 1.1`
///    is not cleared, or `MessageTooLarge` at full corpus coverage). The user
///    is told instantly. This is the design working, and is strictly better
///    UX than the CLI path's equivalent.
///  - **MID** — the plan installed but `push_frame`/`finish` threw
///    `MessageTooLarge` anyway. Same refusal, except the user waited out a
///    whole encode first (tens of seconds at 2.7K) to get it.
///  - **DF** — encode succeeded and the decoder could not recover the message.
///    Silent loss. The one hard assertion, exactly as in the CLI gate.
///
/// It also flags `AppPlan::UniformFallback`: the planner returning
/// `NeedMoreSamples` although every GOP was already sampled, which both apps
/// answer by encoding with no plan at all. That lands the session on the
/// uniform-share ceiling (≈ `n_gops × min(cap)`), which on heterogeneous
/// content is far under the reported Σ-based number — so if this ever engages
/// in the field it manufactures exactly the false `MessageTooLarge` class
/// #850 is about. Measured here rather than asserted, because whether it can
/// engage at full coverage is the open question.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_reported_capacity_success_rate_on_the_app_planner -- --ignored \
///     --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_reported_capacity_success_rate_on_the_app_planner() {
    const GOP: u32 = 30;
    const K: usize = 8;
    let fixtures: [(&str, u32, u32, u32); 3] = [
        ("iphone7_1080p_30fps_h264_high.mov", 1920, 1072, 510), // 17 GOPs
        ("Artlist_AsiaBottle.mp4", 1072, 1920, 145),            //  5 GOPs
        ("dji_mini2_2_7k_24fps_h264_high.mp4", 2720, 1520, 189),//  7 GOPs
    ];
    let pass = "cap_rate_app";
    let mut silent_loss: Vec<String> = Vec::new();
    let mut rows: Vec<String> = Vec::new();

    for (file, w, h, n) in fixtures {
        let Some(yuv) = crop_yuv(file, w, h, n) else {
            eprintln!("SKIP {file} (no fixture)");
            continue;
        };
        let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
        let reported = h264_video_capacity(&yuv, w, h, n as usize, opts, false)
            .map(|i| i.per_tier_primary_max_message_bytes[0])
            .unwrap_or(0);
        // Full sample set — the apps probe every GOP, so `None` (no early
        // exit) is the faithful mirror.
        let caps = match probe_per_gop_caps(&yuv, w, h, n, GOP, None) {
            Ok(c) => c,
            Err(e) => {
                silent_loss.push(format!("{file}: probe failed: {e}"));
                continue;
            }
        };
        let sigma: usize = caps.iter().sum();
        let min_cap = caps.iter().copied().min().unwrap_or(0);
        let n_gops = caps.len();
        // Ask the planner what it accepts rather than restating its arithmetic
        // here. The previous hardcoded `Σ×0.5/1.1` described the pre-fix rule
        // and silently became a wrong annotation on passing results the moment
        // the planner changed.
        let ceiling = {
            let samples: Vec<(usize, usize)> =
                caps.iter().copied().enumerate().collect();
            phasm_core::stego::balanced_allocation::max_message_len(
                &samples, n_gops, &AllocationCalibration::H264_1080P_QP26,
            )
        };
        eprintln!(
            "\n=== {file} {w}×{h}×{n}f = {n_gops} GOPs (APP planner, #841) ===\n  \
             reported(Σ-based upper bound) = {reported} B | Σcaps = {sigma} | \
             min cap = {min_cap} | n×min = {} ({:.0}% of displayed)\n  \
             planner accept ceiling = {} ({:.0}% of displayed), from \
             `max_message_len` — the planner's OWN rule, not a formula \
             restated here\n  \
             (pre-fix this was Σ×0.5/1.1 = {} and `uniform_split` overflowed \
             the weakest GOP above n×min)",
            n_gops * min_cap,
            100.0 * (n_gops * min_cap) as f64 / reported.max(1) as f64,
            ceiling,
            100.0 * ceiling as f64 / reported.max(1) as f64,
            (sigma as f64 * 0.5 / 1.1) as usize,
        );
        eprintln!(
            "  {:>5} | {:>8} | {:>3} {:>3} {:>3} {:>3}  (of K={K})",
            "frac", "framed", "PRE", "MID", "DF", "OK",
        );

        // Sweeps BELOW the planner's 45.5% accept ceiling too, because the
        // question this gate has to answer is not "does the displayed number
        // work" (it does not) but "where does the app path actually start
        // working". `min cap × n_gops / reported` is the predicted knee — the
        // uniform split hands every GOP `framed / W`, so the weakest GOP
        // overflows as soon as `framed / W > min(cap)`.
        for &frac in &[1.00f64, 0.80, 0.55, 0.45, 0.35, 0.25] {
            let target_framed = ((reported as f64 * frac) as usize).max(16);
            let len = raw_len_for_framed(target_framed, w, h, n, GOP, pass).max(16);
            let framed = framed_len_of(len, w, h, n, GOP, pass).unwrap_or(0);
            let msg = incompressible(len);
            let (mut pre, mut mid, mut df, mut ok) = (0usize, 0usize, 0usize, 0usize);
            let mut fallback = false;
            let mut windows: Vec<u32> = Vec::new();
            for _ in 0..K {
                match encode_bytes_app_planner(&yuv, w, h, n, GOP, &msg, pass, &caps) {
                    Err(e) if e.starts_with("planner:") => pre += 1,
                    Err(_) => mid += 1,
                    Ok((b, plan)) => {
                        match plan {
                            AppPlan::Installed(win) => windows.push(win),
                            AppPlan::UniformFallback => fallback = true,
                        }
                        if decode_text(&b, pass).as_deref() == Some(msg.as_str()) {
                            ok += 1;
                        } else {
                            df += 1;
                        }
                    }
                }
            }
            let wtxt = if windows.is_empty() {
                String::new()
            } else {
                let lo = windows.iter().copied().min().unwrap_or(0);
                let hi = windows.iter().copied().max().unwrap_or(0);
                if lo == hi { format!("   W={lo}/{n_gops}") } else { format!("   W={lo}..{hi}/{n_gops}") }
            };
            eprintln!(
                "  {:>4.0}% | {framed:>8} | {pre:>3} {mid:>3} {df:>3} {ok:>3}{wtxt}{}",
                frac * 100.0,
                if fallback { "   [NeedMoreSamples AT FULL COVERAGE -> no plan]" } else { "" },
            );
            rows.push(format!(
                "{file} @{:.0}% (framed {framed}): PRE={pre} MID={mid} DF={df} OK={ok}{}",
                frac * 100.0,
                if fallback { " (uniform fallback engaged)" } else { "" },
            ));
            if df > 0 {
                silent_loss.push(format!(
                    "{file} @{:.0}% (framed {framed} B): {df}/{K} SILENT DECODE FAILURES — \
                     the app planner accepted a message the decoder could not recover",
                    frac * 100.0,
                ));
            }
        }
    }

    eprintln!("\n=== summary (app planner, #841) ===");
    for r in &rows {
        eprintln!("  {r}");
    }
    // Same single invariant as the CLI gate: a refusal at any size is the
    // upper-bound design working (§17.0); silent loss never is.
    assert!(
        silent_loss.is_empty(),
        "silent data loss on the APP planned path:\n  {}",
        silent_loss.join("\n  "),
    );
}

/// Ask `plan_safe_balanced` the only question that matters for the HUD: **what
/// is the largest message it will accept, as a fraction of the capacity the
/// app displays?**
///
/// The app shows `result.primaryMaxBytes`
/// (`EncodeCoordinator.swift:475`) — the Σ-based upper bound from
/// `h264_video_capacity`, with no derate anywhere between the probe and the
/// label. The planner that then has to honour that number applies, at full
/// sample coverage:
///
/// ```text
///   fpc            = 0                       (census: k == n_gops)
///   cap_gop_lower  = mu                      (so the t-bound vanishes)
///   cap_gop_safe   = mu × pessimism_factor   (× 0.5)
///   total_safe     = cap_gop_safe × W,  W ≤ n_gops
///   accept  ⟺  total_safe ≥ msg × safety_margin   (× 1.1)
/// ```
///
/// Σcaps = mu × n_gops, so the accept ceiling is `Σ × 0.5 / 1.1 ≈ 0.4545 × Σ`
/// — a constant, independent of content, resolution and GOP count. This gate
/// measures it rather than asserting the algebra, because the point is the
/// *ratio to the displayed number*, and that is what a user hits.
///
/// Deterministic: the planner reads only the probed caps, so unlike the
/// success-rate gates there is no salt axis here and one bisect is exact. It
/// then does ONE encode+decode at the measured ceiling to confirm the number
/// the planner *does* accept is genuinely embeddable — i.e. that the planner
/// is over-conservative rather than merely wrong.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_app_planner_accept_ceiling -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_app_planner_accept_ceiling_vs_displayed_capacity() {
    const GOP: u32 = 30;
    let fixtures: [(&str, u32, u32, u32); 3] = [
        ("iphone7_1080p_30fps_h264_high.mov", 1920, 1072, 510),
        ("Artlist_AsiaBottle.mp4", 1072, 1920, 145),
        ("dji_mini2_2_7k_24fps_h264_high.mp4", 2720, 1520, 189),
    ];
    let pass = "cap_ceiling";
    let cal = AllocationCalibration::H264_1080P_QP26;

    // Does the app planner accept `len` raw message bytes on these caps?
    let accepts = |yuv: &[u8], w: u32, h: u32, n: u32, caps: &[usize], len: usize| -> bool {
        let params = EncodeSessionParams {
            width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: GOP,
            total_frames_hint: n, color: ColorParams::default(),
            engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
            progress_callback: None,
        };
        let msg = incompressible(len);
        let Ok(mut enc) = StreamingEncodeSession::create(params, &msg, pass) else {
            return false;
        };
        let samples: Vec<(usize, usize)> = caps.iter().copied().enumerate().collect();
        matches!(
            enc.plan_safe_balanced(&samples, GOP as usize, &cal),
            Ok(H264PlanOutcome::Installed { .. })
        )
    };

    // Diagnostic: the planner's actual inputs/outputs at a given raw length,
    // so the accept ceiling can be reconciled with the algebra instead of
    // guessed at. Returns (framed_len, window) when the plan installs.
    let diag = |w: u32, h: u32, n: u32, caps: &[usize], len: usize|
        -> (Option<usize>, Option<u32>) {
        let params = EncodeSessionParams {
            width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: GOP,
            total_frames_hint: n, color: ColorParams::default(),
            engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
            progress_callback: None,
        };
        let msg = incompressible(len);
        let Ok(mut enc) = StreamingEncodeSession::create(params, &msg, pass) else {
            return (None, None);
        };
        let framed = enc.framed_message_len();
        let samples: Vec<(usize, usize)> = caps.iter().copied().enumerate().collect();
        let win = match enc.plan_safe_balanced(&samples, GOP as usize, &cal) {
            Ok(H264PlanOutcome::Installed { window }) => Some(window),
            _ => None,
        };
        (framed, win)
    };

    let mut rows: Vec<String> = Vec::new();
    let mut unembeddable: Vec<String> = Vec::new();

    for (file, w, h, n) in fixtures {
        let Some(yuv) = crop_yuv(file, w, h, n) else {
            eprintln!("SKIP {file} (no fixture)");
            continue;
        };
        let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
        let reported = h264_video_capacity(&yuv, w, h, n as usize, opts, false)
            .map(|i| i.per_tier_primary_max_message_bytes[0])
            .unwrap_or(0);
        let caps = match probe_per_gop_caps(&yuv, w, h, n, GOP, None) {
            Ok(c) => c,
            Err(e) => { eprintln!("{file}: probe failed: {e}"); continue; }
        };
        let sigma: usize = caps.iter().sum();
        let n_gops = caps.len();
        let mu = sigma as f64 / n_gops as f64;
        let analytic = (sigma as f64 * cal.pessimism_factor / cal.safety_margin) as usize;

        // Bisect the accept ceiling over [0, reported].
        let (mut lo, mut hi) = (0usize, reported + 1); // lo accepts, hi refuses
        if accepts(&yuv, w, h, n, &caps, reported) {
            lo = reported;
            hi = reported + 1;
        } else {
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if accepts(&yuv, w, h, n, &caps, mid) { lo = mid } else { hi = mid }
            }
        }
        let ceiling = lo;

        eprintln!(
            "\n=== {file} {w}×{h}×{n}f ===\n  \
             n_gops={n_gops} Σcaps={sigma} mu={mu:.0}\n  \
             DISPLAYED (app HUD, primaryMaxBytes) = {reported} B\n  \
             app-planner ACCEPT CEILING           = {ceiling} B  \
             ({:.1}% of displayed)\n  \
             analytic Σ×0.5/1.1                   = {analytic} B",
            100.0 * ceiling as f64 / reported.max(1) as f64,
        );
        let (framed_at_ceiling, win_at_ceiling) = diag(w, h, n, &caps, ceiling);
        eprintln!(
            "  at ceiling: framed={:?} W={:?}  (mu×pessimism={:.1}/GOP, \
             total_safe@W=n_gops={:.0}, required=framed×1.1)",
            framed_at_ceiling, win_at_ceiling,
            mu * cal.pessimism_factor,
            mu * cal.pessimism_factor * n_gops as f64,
        );
        rows.push(format!(
            "{file}: displayed={reported} ceiling={ceiling} ({:.1}%) analytic={analytic}",
            100.0 * ceiling as f64 / reported.max(1) as f64,
        ));

        // The ceiling must actually embed — otherwise the planner is not
        // merely conservative, it is also wrong, which is a different bug.
        if ceiling >= 16 {
            let msg = incompressible(ceiling);
            match encode_bytes_app_planner(&yuv, w, h, n, GOP, &msg, pass, &caps) {
                Ok((b, _)) => {
                    if decode_text(&b, pass).as_deref() == Some(msg.as_str()) {
                        eprintln!("  ceiling round-trip: OK");
                    } else {
                        unembeddable.push(format!("{file}: ceiling {ceiling} B DECODE-FAIL"));
                    }
                }
                Err(e) => unembeddable.push(format!(
                    "{file}: ceiling {ceiling} B accepted by the planner then failed: {e}"
                )),
            }
        }
    }

    eprintln!("\n=== summary ===");
    for r in &rows { eprintln!("  {r}"); }
    eprintln!(
        "\nThe ceiling is a fixed fraction of Σ (pessimism 0.5 / margin 1.1),\n\
         so it does not converge toward the displayed number on any content."
    );
    assert!(
        unembeddable.is_empty(),
        "the app planner accepted a message it could not embed:\n  {}",
        unembeddable.join("\n  "),
    );
}


/// The 13-fixture corpus census of `min(cap) / mu` — the predicate that
/// decides whether `plan_safe_balanced` can over-accept.
///
/// **Why this is probe-only.** The accept ceiling of the app planner at full
/// coverage is `0.4545 × Σcaps` (derived in §18), and `uniform_split` hands
/// every GOP in the window `framed / W` with no reference to the cap vector.
/// So the planner is safe iff its ceiling sits at or below the weakest GOP's
/// share, i.e.
///
/// ```text
///     0.4545 × Σ  ≤  n_gops × min(cap)      ⟺      min(cap)/mu ≥ 0.4545
/// ```
///
/// That is algebra, not a fit — it needs no encodes and no salts. What a
/// corpus *can* tell us is how often it bites, which is a prevalence
/// question and the thing that weights a cap-aware allocator against a
/// narrow feasibility clamp. One probe pass per fixture (~20 ms/frame at
/// 1080p), no stego encode, no decode.
///
/// The first three rows reuse the exact dims/frame counts of the two
/// success-rate gates, so their `min/mu`, `Σ` and ceiling must reproduce the
/// numbers those gates printed — a self-check on the census.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_corpus_min_mu_census -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_corpus_min_mu_census() {
    const GOP: u32 = 30;
    /// `pessimism_factor / safety_margin` = 0.5 / 1.1 — the app planner's
    /// accept fraction of Σ at full coverage (`fpc = 0` ⇒ lower bound = mu).
    const CEILING_FRAC: f64 = 0.5 / 1.1;

    // (tag, source, crop_w, crop_h, n_frames). The first three match the
    // success-rate gates exactly; the rest are whole-GOP windows at each
    // fixture's native aspect, 16-aligned.
    let fixtures: [(&str, &str, u32, u32, u32); 13] = [
        ("iphone7",       "iphone7_1080p_30fps_h264_high.mov",   1920, 1072, 510),
        ("asia_bottle",   "Artlist_AsiaBottle.mp4",              1072, 1920, 145),
        ("dji",           "dji_mini2_2_7k_24fps_h264_high.mp4",  2720, 1520, 189),
        // Carries rotation=-90 side data, so ffmpeg autorotates it to
        // PORTRAIT 1080x1920 — a 1920x1072 crop is rejected as "too big".
        ("iphone5",       "iphone5_1080p_30fps_h264_high.mov",   1072, 1920, 360),
        ("lumix",         "lumix_g9_1080p_30fps_h264_high.mp4",  1920, 1072, 270),
        ("img4138",       "IMG_4138.MOV",                        1920, 1072, 600),
        ("carplane",      "Artlist_CarPlane.mp4",                1072, 1920, 180),
        ("handbag",       "Artlist_Handbag.mp4",                 1072, 1904, 180),
        ("horse_flag",    "Artlist_HorseFlag.mp4",               1920, 1072, 120),
        ("phone_booth",   "Artlist_PhoneBooth.mp4",              1920, 1072, 120),
        ("pirate_battle", "Artlist_PirateBattle.mp4",            1920, 1072, 120),
        ("school_fight",  "Artlist_SchoolFight.mp4",             1280,  720, 360),
        ("woman_subway",  "Artlist_WomanSubway.mp4",             1280,  720, 360),
    ];

    struct Row {
        tag: &'static str,
        n_gops: usize,
        sigma: usize,
        min_cap: usize,
        mu: f64,
        ratio: f64,
        reported: usize,
        ceiling: f64,
        n_min: usize,
        probe_ms: u128,
    }
    let mut rows: Vec<Row> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    for (tag, file, w, h, n) in fixtures {
        let Some(yuv) = crop_yuv(file, w, h, n) else {
            skipped.push(format!("{tag} ({file}): no fixture / ffmpeg failed"));
            continue;
        };
        let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
        let reported = h264_video_capacity(&yuv, w, h, n as usize, opts, false)
            .map(|i| i.per_tier_primary_max_message_bytes[0])
            .unwrap_or(0);
        let t0 = std::time::Instant::now();
        let caps = match probe_per_gop_caps(&yuv, w, h, n, GOP, None) {
            Ok(c) => c,
            Err(e) => {
                skipped.push(format!("{tag}: probe failed: {e}"));
                continue;
            }
        };
        let probe_ms = t0.elapsed().as_millis();
        let sigma: usize = caps.iter().sum();
        let min_cap = caps.iter().copied().min().unwrap_or(0);
        let n_gops = caps.len();
        let mu = sigma as f64 / n_gops as f64;
        eprintln!(
            "  probed {tag:<14} {w}x{h}x{n}f  {n_gops} GOPs  caps={caps:?}",
        );
        rows.push(Row {
            tag,
            n_gops,
            sigma,
            min_cap,
            mu,
            ratio: if mu > 0.0 { min_cap as f64 / mu } else { 0.0 },
            reported,
            ceiling: CEILING_FRAC * sigma as f64,
            n_min: n_gops * min_cap,
            probe_ms,
        });
    }

    eprintln!("\n=== corpus census: min(cap)/mu vs the app planner's accept ceiling ===");
    eprintln!("  accept ceiling = {CEILING_FRAC:.4} x Sigma   |   safe iff min/mu >= {CEILING_FRAC:.4}\n");
    eprintln!(
        "  {:<14} {:>5} {:>9} {:>8} {:>9} {:>7} {:>9} {:>9} {:>8}  {}",
        "fixture", "GOPs", "Sigma", "min", "mu", "min/mu", "ceiling", "n x min", "probe", "verdict",
    );
    for r in &rows {
        let over = r.ceiling > r.n_min as f64;
        eprintln!(
            "  {:<14} {:>5} {:>9} {:>8} {:>9.0} {:>7.3} {:>9.0} {:>9} {:>7}s  {}",
            r.tag, r.n_gops, r.sigma, r.min_cap, r.mu, r.ratio, r.ceiling, r.n_min,
            format!("{:.1}", r.probe_ms as f64 / 1000.0),
            if over {
                format!("OVER-ACCEPTS by {} B ({:.0}% of displayed)",
                    r.ceiling as usize - r.n_min,
                    100.0 * (r.ceiling - r.n_min as f64) / r.reported.max(1) as f64)
            } else {
                "safe".to_string()
            },
        );
    }

    let n = rows.len();
    let bad: Vec<&Row> = rows.iter().filter(|r| r.ceiling > r.n_min as f64).collect();
    eprintln!("\n  prevalence: {}/{} fixtures over-accept (min/mu < {CEILING_FRAC:.4})", bad.len(), n);
    if !bad.is_empty() {
        eprintln!("    affected: {}", bad.iter().map(|r| r.tag).collect::<Vec<_>>().join(", "));
    }
    if n > 0 {
        let mut ratios: Vec<f64> = rows.iter().map(|r| r.ratio).collect();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = ratios[n / 2];
        eprintln!(
            "    min/mu  min={:.3}  median={:.3}  max={:.3}",
            ratios[0], median, ratios[n - 1],
        );
        // What fraction of the DISPLAYED number is actually safe to accept,
        // per fixture, under today's uniform split: n x min / reported.
        let mut usable: Vec<f64> = rows.iter()
            .map(|r| r.n_min as f64 / r.reported.max(1) as f64).collect();
        usable.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "    safe fraction of displayed (n x min / reported): min={:.3} median={:.3} max={:.3}",
            usable[0], usable[n / 2], usable[n - 1],
        );
    }
    if !skipped.is_empty() {
        eprintln!("\n  SKIPPED:");
        for s in &skipped { eprintln!("    {s}"); }
    }

    assert!(n >= 3, "census needs >= 3 probed fixtures, got {n}");
}


/// Encode with the CANDIDATE allocator: `allocate_chunks_proportional`
/// (cap-clamped, largest-remainder) installed via `set_gop_alloc_plan`.
///
/// This is the option-1 shape. It differs from both shipped planners:
/// `plan_safe_balanced` throws the cap vector away (`uniform_split`), and
/// `plan_proportional` is misnamed — it installs
/// `allocate_chunks_concentrate_tail`. Only this one gives
/// `alloc[i] ≤ cap[i]` by construction, so the binding limit is `Σcap`
/// rather than `n_gops × min(cap)` and no shrink-carry is needed.
///
/// Returns the installed plan alongside the bytes so a caller can report the
/// realised shape.
fn encode_bytes_proportional(
    yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, msg: &str, pass: &str,
    caps: &[usize], r_target: f64,
) -> Result<(Vec<u8>, Vec<usize>), String> {
    let params = EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut enc = StreamingEncodeSession::create(params, msg, pass)
        .map_err(|e| format!("session create: {e:?}"))?;
    let framed = enc.framed_message_len()
        .ok_or_else(|| "no framed length (not OH264)".to_string())?;
    let plan = phasm_core::stego::chunk_frame::allocate_chunks_proportional(
        caps, framed, r_target,
    ).ok_or_else(|| format!(
        "planner: framed {framed} > Sigma {}", caps.iter().sum::<usize>(),
    ))?;
    enc.set_gop_alloc_plan(plan.clone())
        .map_err(|e| format!("planner: set_gop_alloc_plan: {e:?}"))?;
    let fb = (w * h * 3 / 2) as usize;
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let cw = (w / 2) as usize;
    let mut out = Vec::new();
    for f in 0..n as usize {
        let off = f * fb;
        enc.push_frame(
            YuvFrameRef {
                y: &yuv[off..off + ys], y_stride: w as usize,
                u: &yuv[off + ys..off + ys + cs], u_stride: cw,
                v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
            },
            &mut out,
        ).map_err(|e| format!("push_frame #{f}: {e:?}"))?;
    }
    enc.finish(&mut out).map_err(|e| format!("finish: {e:?}"))?;
    Ok((out, plan))
}

/// Corpus success-rate gate for the CANDIDATE proportional allocator, as a
/// fraction of **Σcaps** (not of the displayed number, so the ceiling the
/// allocator claims to bind on is the x-axis).
///
/// What this answers that the two existing gates cannot: both of them
/// measure a shipped planner whose accept region is far below Σ
/// (`plan_safe_balanced` at 0.4545 × Σ, `plan_proportional` via
/// concentrate-tail). The question for option 1 is how much of Σ a
/// cap-clamped split can actually *deliver* end-to-end — encode, then decode
/// back to the exact plaintext.
///
/// Columns: PRE (allocator refused before encoding), MID (accepted then
/// failed mid-encode — the failure class option 1 is meant to delete),
/// DF (encoded but decoded wrong or not at all — the silent-loss class that
/// must stay 0), OK.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_proportional_success_rate_on_the_corpus -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_proportional_success_rate_on_the_corpus() {
    const GOP: u32 = 30;
    const K: usize = 4;
    let fixtures: [(&str, &str, u32, u32, u32); 13] = [
        ("iphone7",       "iphone7_1080p_30fps_h264_high.mov",   1920, 1072, 510),
        ("asia_bottle",   "Artlist_AsiaBottle.mp4",              1072, 1920, 145),
        ("dji",           "dji_mini2_2_7k_24fps_h264_high.mp4",  2720, 1520, 189),
        ("iphone5",       "iphone5_1080p_30fps_h264_high.mov",   1072, 1920, 360),
        ("lumix",         "lumix_g9_1080p_30fps_h264_high.mp4",  1920, 1072, 270),
        ("img4138",       "IMG_4138.MOV",                        1920, 1072, 600),
        ("carplane",      "Artlist_CarPlane.mp4",                1072, 1920, 180),
        ("handbag",       "Artlist_Handbag.mp4",                 1072, 1904, 180),
        ("horse_flag",    "Artlist_HorseFlag.mp4",               1920, 1072, 120),
        ("phone_booth",   "Artlist_PhoneBooth.mp4",              1920, 1072, 120),
        ("pirate_battle", "Artlist_PirateBattle.mp4",            1920, 1072, 120),
        ("school_fight",  "Artlist_SchoolFight.mp4",             1280,  720, 360),
        ("woman_subway",  "Artlist_WomanSubway.mp4",             1280,  720, 360),
    ];
    let fracs: [f64; 4] = [0.80, 0.60, 0.45, 0.30];
    let mut silent_loss: Vec<String> = Vec::new();
    let mut unembeddable: Vec<String> = Vec::new();
    let mut summary: Vec<String> = Vec::new();

    for (tag, file, w, h, n) in fixtures {
        let Some(yuv) = crop_yuv(file, w, h, n) else {
            eprintln!("SKIP {tag} ({file}) — no fixture");
            continue;
        };
        let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
        let reported = h264_video_capacity(&yuv, w, h, n as usize, opts, false)
            .map(|i| i.per_tier_primary_max_message_bytes[0])
            .unwrap_or(0);
        let caps = match probe_per_gop_caps(&yuv, w, h, n, GOP, None) {
            Ok(c) => c,
            Err(e) => { silent_loss.push(format!("{tag}: probe failed: {e}")); continue; }
        };
        let sigma: usize = caps.iter().sum();
        let min_cap = caps.iter().copied().min().unwrap_or(0);
        let n_gops = caps.len();
        let mu = sigma as f64 / n_gops as f64;
        eprintln!(
            "\n=== {tag} {w}x{h}x{n}f = {n_gops} GOPs | Sigma={sigma} min={min_cap} \
             min/mu={:.3} reported={reported} ===",
            if mu > 0.0 { min_cap as f64 / mu } else { 0.0 },
        );
        eprintln!("  {:>6} | {:>8} | {:>3} {:>3} {:>3} {:>3}  (of K={K})",
            "of Sig", "framed", "PRE", "MID", "DF", "OK");

        let mut best_ok_frac = 0.0f64;
        for &frac in &fracs {
            let target_framed = ((sigma as f64 * frac) as usize).max(16);
            let len = raw_len_for_framed(target_framed, w, h, n, GOP, "cap_prop").max(16);
            let framed = framed_len_of(len, w, h, n, GOP, "cap_prop").unwrap_or(0);
            let (mut pre, mut mid, mut df, mut ok) = (0usize, 0usize, 0usize, 0usize);
            for k in 0..K {
                // Salt held FIXED across fractions, deliberately. Folding the
                // fraction into the passphrase (as this first did) gives every
                // size a different salt AND a different message, so a
                // size-vs-size comparison varies two things at once — which is
                // how iphone5 came out 0/4 at 60% of Sigma while passing 4/4 at
                // both 80% and 45%, a shape the allocator provably cannot
                // produce (see
                // `per_gop_allocation_is_monotone_in_message_size_across_the_soft_ceiling`).
                // `incompressible` is prefix-stable, so with the salt fixed the
                // shorter message is a strict prefix of the longer one and size
                // is the only variable.
                let pass = format!("cap_prop_{tag}_{k}");
                let msg = incompressible(len);
                match encode_bytes_proportional(
                    &yuv, w, h, n, GOP, &msg, &pass, &caps, CAP2_DEFAULT_R_TARGET,
                ) {
                    Err(e) if e.starts_with("planner:") => pre += 1,
                    Err(e) => {
                        mid += 1;
                        unembeddable.push(format!(
                            "{tag} @ {:.0}% of Sigma (framed {framed}): accepted then {e}",
                            frac * 100.0,
                        ));
                    }
                    Ok((bytes, _plan)) => {
                        match decode_text(&bytes, &pass) {
                            Some(t) if t == msg => ok += 1,
                            other => {
                                df += 1;
                                silent_loss.push(format!(
                                    "{tag} @ {:.0}% of Sigma: encoded but decode {}",
                                    frac * 100.0,
                                    if other.is_none() { "FAILED" } else { "MISMATCHED" },
                                ));
                            }
                        }
                    }
                }
            }
            eprintln!("  {:>5.0}% | {:>8} | {:>3} {:>3} {:>3} {:>3}",
                frac * 100.0, framed, pre, mid, df, ok);
            if ok == K && best_ok_frac == 0.0 {
                best_ok_frac = frac;
            }
        }
        summary.push(format!(
            "  {:<14} K/K at {:>3.0}% of Sigma  ({} B of Sigma {}, {:.0}% of displayed {})",
            tag, best_ok_frac * 100.0,
            (sigma as f64 * best_ok_frac) as usize, sigma,
            100.0 * sigma as f64 * best_ok_frac / reported.max(1) as f64, reported,
        ));
    }

    eprintln!("\n=== proportional: highest fraction of Sigma with {K}/{K} round-trips ===");
    for l in &summary { eprintln!("{l}"); }

    if !silent_loss.is_empty() {
        eprintln!("\nSILENT LOSS ({}):", silent_loss.len());
        for l in &silent_loss { eprintln!("  {l}"); }
    }
    if !unembeddable.is_empty() {
        eprintln!("\nACCEPTED-THEN-FAILED ({}):", unembeddable.len());
        for l in &unembeddable { eprintln!("  {l}"); }
    }

    // The one hard invariant: never encode something that decodes wrong.
    assert!(
        silent_loss.is_empty(),
        "proportional allocator produced silent loss:\n  {}",
        silent_loss.join("\n  "),
    );
}


/// Pins the closed-form framed-length formula against a real session.
///
/// `core/src/bin/h264_alloc_shape_sweep.rs` sizes its payloads by computing
/// the framed length directly rather than by creating a
/// `StreamingEncodeSession`. Session create runs `crypto::encrypt` *and*
/// `CabacStegoMasterKeys::derive` — two Argon2id derivations, ~38 ms per
/// create at the sizes asserted below — so a ~13-probe bisect burns ~0.5 s
/// per sample in the KDF alone. Worth removing from a hot loop, but the
/// decisive reason is exactness: the formula IS the framing.
///
/// The formula mirrors `stego::frame::build_frame`'s layout
/// `[len 2|6][salt 16][nonce 12][ct P+16][crc32 4]`:
///
/// ```text
///     framed = compressed_payload_size(msg) + 50     (P <= u16::MAX)
///     framed = compressed_payload_size(msg) + 54     (v2 header)
/// ```
///
/// A duplicated constant that nothing checks is a constant that drifts, so
/// this test asserts the formula against `framed_message_len()` for real
/// messages on both sides of the v2 header threshold. NOT `#[ignore]`d — it
/// is cheap (a few session creates, no encoding) and it guards an instrument
/// whose payload sizing silently decides what every AUC number means.
#[test]
fn alloc_shape_framing_matches_session() {
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 30;
    const GOP: u32 = 30;

    // Straddle the v2 header threshold: `is_v2` triggers on the COMPRESSED
    // payload length exceeding u16::MAX, and `incompressible` yields ~1.0x,
    // so ~70 KB of it clears 65535.
    for raw in [16usize, 100, 1000, 8306, 40_000, 70_000] {
        let msg = incompressible(raw);
        let p = phasm_core::stego::compressed_payload_size(&msg, &[]);
        let predicted = p + if p > u16::MAX as usize { 54 } else { 50 };

        let params = EncodeSessionParams {
            width: W, height: H, fps_num: 30, fps_den: 1, qp: 26, gop_size: GOP,
            total_frames_hint: N, color: ColorParams::default(),
            engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
            progress_callback: None,
        };
        let sess = StreamingEncodeSession::create(params, &msg, "framing-pin")
            .expect("session create");
        let actual = sess.framed_message_len().expect("framed len (OH264)");

        assert_eq!(
            predicted, actual,
            "framed-length formula drifted at raw={raw} (compressed P={p}): \
             predicted {predicted}, session says {actual}. \
             If stego::frame::build_frame's layout changed, update both this \
             test and h264_alloc_shape_sweep::framed_len_of_raw.",
        );
    }
}


/// Is the planner budgeting against a tier the encoder never uses?
///
/// The probe measures per-GOP capacity at **tier 0** — `drain_gop_probe`
/// passes `full_tiers = false` ("#809 — progress-probe path: tier 0 only
/// (5× faster)"), and `h264_video_capacity` fills tiers 1..4 from tier 0 when
/// asked the same way, on the stated assumption that higher tiers are
/// "capacity-neutral on real content".
///
/// The encode does NOT embed at tier 0. `embed_gop_roundtrip_safe` calls
/// `consume_gop_emit(..., CascadeTier::Auto, DEFAULT_HEADROOM)`, so the tier
/// is resolved per GOP from the chunk size. A higher tier is a STRICTER
/// cascade-safety filter, i.e. fewer usable positions.
///
/// If that assumption fails on some content, the planner allocates against a
/// budget the encoder never has — which would show up exactly as the corpus
/// sweep's shape: handbag round-tripping only at 30% of Σ, a 3.3× shortfall
/// far beyond the ~1.4× J2 spread §16 measured.
///
/// This prints true per-tier capacity (`full_tiers = true`) against tier 0
/// for the fixtures the sweep found worst, and asserts nothing — it is a
/// diagnostic, and the ratio is the finding.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_per_tier_vs_tier0_census -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_per_tier_vs_tier0_census() {
    const GOP: u32 = 30;
    // The four that failed to reach 80% of Sigma, plus iphone7 and dji as
    // controls (both reached 80%).
    let fixtures: [(&str, &str, u32, u32, u32); 6] = [
        ("handbag",      "Artlist_Handbag.mp4",                1072, 1904, 180),
        ("horse_flag",   "Artlist_HorseFlag.mp4",              1920, 1072, 120),
        ("lumix",        "lumix_g9_1080p_30fps_h264_high.mp4", 1920, 1072, 270),
        ("school_fight", "Artlist_SchoolFight.mp4",            1280,  720, 360),
        ("iphone7",      "iphone7_1080p_30fps_h264_high.mov",  1920, 1072, 510),
        ("dji",          "dji_mini2_2_7k_24fps_h264_high.mp4", 2720, 1520, 189),
    ];
    eprintln!(
        "\n=== per-tier capacity vs tier 0 (what the planner budgets against) ===\n  \
         {:<14} {:>9} {:>9} {:>9} {:>9} {:>9}   {}",
        "fixture", "tier0", "tier1", "tier2", "tier3", "tier4", "worst/tier0",
    );
    for (tag, file, w, h, n) in fixtures {
        let Some(yuv) = crop_yuv(file, w, h, n) else {
            eprintln!("  SKIP {tag}");
            continue;
        };
        let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
        // full_tiers = true: compute the REAL per-tier breakdown instead of
        // copying tier 0 into 1..4.
        let Some(info) = h264_video_capacity(&yuv, w, h, n as usize, opts, true).ok() else {
            eprintln!("  {tag}: capacity probe failed");
            continue;
        };
        let t = info.per_tier_primary_max_message_bytes;
        let t0 = t[0].max(1);
        let worst = t.iter().copied().min().unwrap_or(0);
        eprintln!(
            "  {:<14} {:>9} {:>9} {:>9} {:>9} {:>9}   {:.3}",
            tag, t[0], t[1], t[2], t[3], t[4],
            worst as f64 / t0 as f64,
        );
    }
    eprintln!(
        "\n  If any tier sits well below tier 0, the probe's tier-0 basis\n  \
         over-reports what `CascadeTier::Auto` can actually embed."
    );
}


/// Attribute the probe-vs-encode capacity gap across its three axes.
///
/// `cap2_stc_trial_max_payload` reports the largest payload embeddable for a
/// SYNTHETIC payload under the FIXED `CAP2_TRIAL_SEED` hhat, with NO
/// round-trip verification, times a 5/6 derate. The encode requires the REAL
/// payload under the REAL hhat AND `verify_chunk_v3` to decode it back. Three
/// differences, any of which can flip STC feasibility near the limit.
///
/// Per GOP, this feeds the probe's own reported cap back through the real
/// encode path (`h264_diag_verified_gop_consumed`) under four combinations and
/// reports `consumed / cap`:
///
/// - `verif` — synthetic payload + trial seed. Same STC instance the probe
///   measured, so a shortfall here is the VERIFICATION axis alone.
/// - `msg`   — real `incompressible` message + trial seed. Versus `verif`,
///   the MESSAGE-bit axis (J2).
/// - `hhat`  — synthetic payload + a different seed. Versus `verif`, the
///   PARITY-CHECK axis (J1).
/// - `floorK`— minimum over K real message draws, trial seed. The practical
///   message-independent floor, which is what a HUD number must be if
///   "accepted in the UI" is to imply "will encode".
///
/// All four use `chunk_idx = 0` because the probe always frames with
/// `build_first_chunk_frame_v3_1`; comparing against a smaller non-first
/// header would bias the ratio.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_probe_vs_verified_axes -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_probe_vs_verified_axes() {
    use phasm_core::codec::h264::streaming_session::h264_diag_verified_gop_consumed;
    const GOP: u32 = 30;
    const K: usize = 4;
    const TRIAL_SEED: [u8; 32] = [0x2c; 32]; // CAP2_TRIAL_SEED
    const OTHER_SEED: [u8; 32] = [0x5a; 32];

    /// The probe's synthetic trial payload (`cap2_trial_payload`).
    fn trial_payload(len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let mut st: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..len {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push((st >> 33) as u8);
        }
        v
    }
    fn msg_payload(len: usize, draw: u64) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let mut st: u64 = 0xDEAD_BEEF_0000_0001 ^ draw.wrapping_mul(0x9E3779B97F4A7C15);
        for _ in 0..len {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push((st >> 33) as u8);
        }
        v
    }

    // handbag is the worst case (30% of Sigma); horse_flag and lumix also
    // fell short; dji reached 80% as a control.
    let fixtures: [(&str, &str, u32, u32, u32); 4] = [
        ("handbag",    "Artlist_Handbag.mp4",                1072, 1904, 180),
        ("horse_flag", "Artlist_HorseFlag.mp4",              1920, 1072, 120),
        ("lumix",      "lumix_g9_1080p_30fps_h264_high.mp4", 1920, 1072, 270),
        ("dji",        "dji_mini2_2_7k_24fps_h264_high.mp4", 2720, 1520, 189),
    ];
    let weights = CostWeights::default();

    for (tag, file, w, h, n) in fixtures {
        let Some(yuv) = crop_yuv(file, w, h, n) else {
            eprintln!("SKIP {tag}");
            continue;
        };
        let Ok(caps) = probe_per_gop_caps(&yuv, w, h, n, GOP, None) else {
            eprintln!("{tag}: probe failed");
            continue;
        };
        let frame_sz = (w * h * 3 / 2) as usize;
        eprintln!(
            "\n=== {tag} ({} GOPs) — consumed/cap under the encoder's real test ===\n  \
             {:>4} {:>8}  {:>7} {:>7} {:>7} {:>7}",
            caps.len(), "gop", "cap", "verif", "msg", "hhat", "floorK",
        );
        let mut worst_floor = 1.0f64;
        for (g, &cap) in caps.iter().enumerate() {
            if cap == 0 { continue; }
            let f0 = g * GOP as usize;
            let f1 = ((g + 1) * GOP as usize).min(n as usize);
            let gop_yuv = &yuv[f0 * frame_sz..f1 * frame_sz];
            let gn = (f1 - f0) as u32;
            let run = |payload: &[u8], seed: &[u8; 32]| -> f64 {
                h264_diag_verified_gop_consumed(
                    gop_yuv, w, h, gn, 26, GOP, payload, 0, cap as u32, seed, &weights,
                ).map(|c| c as f64 / cap as f64).unwrap_or(f64::NAN)
            };
            let r_verif = run(&trial_payload(cap), &TRIAL_SEED);
            let r_msg = run(&msg_payload(cap, 0), &TRIAL_SEED);
            let r_hhat = run(&trial_payload(cap), &OTHER_SEED);
            let r_floor = (0..K)
                .map(|k| run(&msg_payload(cap, k as u64), &TRIAL_SEED))
                .fold(f64::INFINITY, f64::min);
            worst_floor = worst_floor.min(r_floor);
            eprintln!(
                "  {:>4} {:>8}  {:>7.3} {:>7.3} {:>7.3} {:>7.3}",
                g, cap, r_verif, r_msg, r_hhat, r_floor,
            );
        }
        eprintln!("  worst per-GOP floor over K={K} draws: {worst_floor:.3} of reported cap");
    }
    eprintln!(
        "\n  verif < 1.0  => the probe over-reports for its OWN instance (verification).\n  \
         msg/hhat < verif => the message-bit / parity-check axis adds to it.\n  \
         floorK is what a message-independent HUD number would have to use."
    );
}


/// Attribute the verify-fail: WHICH domain leaks, and is it desync or values?
///
/// `cap_probe_vs_verified_axes` found GOPs whose reported cap does not verify
/// — handbag GOP 2 at 0.587, lumix GOP 6 at 0.676 — deterministically across
/// payload and parity-check variation. §15 is explicit about what that means:
/// "If a clip ever round-trips *below* reported, that's a CASCADE.V2 bug to
/// fix at the root — never a margin to widen." So this locates the root.
///
/// Method: emit ONE attempt at the probe's own reported cap (no shrink loop,
/// so the FAILING bitstream is observable), then walk it and compare the
/// 4-domain cover against the clean walk of the same GOP.
///
/// - A domain whose POSITION COUNT differs ⇒ CABAC desync: a flip changed a
///   later bin's `ctxIdxInc`, so the decoder walks a different number of bins.
///   That is the A.1.10 signature (`|mvd|` feeding next-MB mvd ctxIdxInc).
/// - Equal counts, differing VALUES ⇒ a plain bit-level leak, not desync.
///
/// Flip counts are reported too: a domain the STC never touched cannot be the
/// cause, so the leak must lie in a domain with flips > 0.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_cascade_leak_attribution -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_cascade_leak_attribution() {
    use phasm_core::codec::h264::streaming_session::h264_diag_gop_emit_once;
    const GOP: u32 = 30;
    const TRIAL_SEED: [u8; 32] = [0x2c; 32];

    fn trial_payload(len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let mut st: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..len {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push((st >> 33) as u8);
        }
        v
    }

    // (tag, file, w, h, n_frames, the GOP that failed to verify)
    let targets: [(&str, &str, u32, u32, u32, usize); 2] = [
        ("handbag", "Artlist_Handbag.mp4",                1072, 1904, 180, 2),
        ("lumix",   "lumix_g9_1080p_30fps_h264_high.mp4", 1920, 1072, 270, 6),
    ];
    let weights = CostWeights::default();

    for (tag, file, w, h, n, gop_idx) in targets {
        let Some(yuv) = crop_yuv(file, w, h, n) else {
            eprintln!("SKIP {tag}");
            continue;
        };
        let Ok(caps) = probe_per_gop_caps(&yuv, w, h, n, GOP, None) else {
            eprintln!("{tag}: probe failed");
            continue;
        };
        let Some(&cap) = caps.get(gop_idx) else {
            eprintln!("{tag}: no GOP {gop_idx}");
            continue;
        };
        let frame_sz = (w * h * 3 / 2) as usize;
        let f0 = gop_idx * GOP as usize;
        let f1 = ((gop_idx + 1) * GOP as usize).min(n as usize);
        let gop_yuv = &yuv[f0 * frame_sz..f1 * frame_sz];
        let gn = (f1 - f0) as u32;

        let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
        let clean = match h264_walk_cover(gop_yuv, w, h, gn, opts) {
            Ok(c) => c,
            Err(e) => { eprintln!("{tag}: clean walk failed: {e:?}"); continue; }
        };
        let payload = trial_payload(cap);
        let (stego_bs, verified) = match h264_diag_gop_emit_once(
            gop_yuv, w, h, gn, 26, GOP, &payload, 0, cap as u32, &TRIAL_SEED, &weights,
        ) {
            Ok(r) => r,
            Err(e) => { eprintln!("{tag}: emit failed: {e:?}"); continue; }
        };
        eprintln!("\n=== {tag} GOP {gop_idx} | reported cap {cap} B | verified={verified} ===");
        if stego_bs.is_empty() {
            eprintln!("  emit refused outright (MessageTooLarge) — STC could not place it");
            continue;
        }
        let stego = match walk_annex_b_for_cover_with_options(
            &stego_bs, WalkOptions { record_mvd: true, ..Default::default() },
        ) {
            Ok(wk) => wk.cover,
            Err(e) => {
                eprintln!("  STEGO WALK FAILED OUTRIGHT: {e:?}");
                eprintln!("  => the emitted bitstream is not walkable; desync is total.");
                continue;
            }
        };
        eprintln!("  {:<16} {:>9} {:>9} {:>7}  {}", "domain", "clean n", "stego n", "flips", "verdict");
        let doms: [(&str, &Vec<u8>, &Vec<u8>); 4] = [
            ("CoeffSign",      &clean.coeff_sign_bypass.bits, &stego.coeff_sign_bypass.bits),
            ("CoeffSuffixLsb", &clean.coeff_suffix_lsb.bits,  &stego.coeff_suffix_lsb.bits),
            ("MvdSign",        &clean.mvd_sign_bypass.bits,   &stego.mvd_sign_bypass.bits),
            ("MvdSuffixLsb",   &clean.mvd_suffix_lsb.bits,    &stego.mvd_suffix_lsb.bits),
        ];
        for (name, c, sg) in doms {
            let flips = c.iter().zip(sg.iter()).filter(|(a, b)| a != b).count();
            let verdict = if c.len() != sg.len() {
                format!("DESYNC ({:+})", sg.len() as i64 - c.len() as i64)
            } else if flips == 0 {
                "untouched".to_string()
            } else {
                "counts match".to_string()
            };
            eprintln!("  {:<16} {:>9} {:>9} {:>7}  {}", name, c.len(), sg.len(), flips, verdict);
        }
    }
    eprintln!(
        "\n  A DESYNC row is the cascade: that domain's flips moved a later\n  \
         bin's ctxIdxInc. 'untouched' rules a domain out as the cause."
    );
}


/// Does the structural cost mask's ∞ pattern match full J-UNIWARD at the
/// geometry and GOP index the corpus sweep actually used?
///
/// `cap_cascade_leak_attribution` showed the emit refused handbag GOP 2 and
/// lumix GOP 6 with `MessageTooLarge` — STC could not place the payload at
/// all, so nothing reached `verify_chunk_v3`. That is a FEASIBILITY
/// disagreement, not a verification leak: the probe decides feasibility under
/// `CapCostSource::StructuralMask` while the emit decides it under full
/// J-UNIWARD costs (`produce_gop_cover`).
///
/// `h264_gop_cost_mask_divergence`'s own contract: "**Every entry must be
/// zero** … if the tail ever does fire, the patterns diverge here and the test
/// fails loudly instead of the probe quietly misreporting capacity."
///
/// `cap2_mask_equivalence` pins that — but it scales EVERY fixture to
/// 1920×1072 and uses N=30, i.e. landscape geometry and GOP 0 only. The
/// failures are handbag at 1072×1904 PORTRAIT GOP 2, and lumix at GOP 6.
/// Neither is inside that coverage, so this re-checks at the sweep's own
/// geometry, per GOP, and prints the tier-0 cap under both cost sources.
///
/// `cap_mask` > `cap_full` is the over-report: the probe promises what the
/// emit cannot place.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_mask_divergence_at_sweep_geometry -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_mask_divergence_at_sweep_geometry() {
    use phasm_core::codec::h264::openh264_stego::{
        h264_gop_capacity_per_tier_with_costs, h264_gop_cost_mask_divergence,
    };
    const GOP: u32 = 30;
    // handbag at its PORTRAIT sweep geometry (never covered by the
    // equivalence gate); lumix at the same landscape dims the gate uses, so
    // geometry-vs-GOP-index can be separated.
    let targets: [(&str, &str, u32, u32, u32); 2] = [
        ("handbag", "Artlist_Handbag.mp4",                1072, 1904, 180),
        ("lumix",   "lumix_g9_1080p_30fps_h264_high.mp4", 1920, 1072, 270),
    ];
    let weights = CostWeights::default();
    let mut any_divergence = false;

    for (tag, file, w, h, n) in targets {
        let Some(yuv) = crop_yuv(file, w, h, n) else {
            eprintln!("SKIP {tag}");
            continue;
        };
        let frame_sz = (w * h * 3 / 2) as usize;
        let n_gops = (n as usize).div_ceil(GOP as usize);
        eprintln!(
            "\n=== {tag} {w}x{h} ({n_gops} GOPs) ===\n  \
             {:>4} {:>9} {:>9}  {:>26}",
            "gop", "cap_mask", "cap_full", "divergence [CS,CSL,MS,MSL]",
        );
        for g in 0..n_gops {
            let f0 = g * GOP as usize;
            let f1 = ((g + 1) * GOP as usize).min(n as usize);
            if f0 >= f1 { break; }
            let gop_yuv = &yuv[f0 * frame_sz..f1 * frame_sz];
            let gn = (f1 - f0) as u32;
            let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
            let cap_mask = h264_gop_capacity_per_tier_with_costs(
                gop_yuv, w, h, gn, opts, &weights, false, false,
            ).map(|t| t[0]).unwrap_or(0);
            let cap_full = h264_gop_capacity_per_tier_with_costs(
                gop_yuv, w, h, gn, opts, &weights, false, true,
            ).map(|t| t[0]).unwrap_or(0);
            let div = h264_gop_cost_mask_divergence(gop_yuv, w, h, gn, opts)
                .unwrap_or([usize::MAX; 4]);
            if div.iter().any(|&d| d != 0) { any_divergence = true; }
            let flag = if cap_mask > cap_full {
                format!("  <== OVER-REPORT {:.3}", cap_full as f64 / cap_mask.max(1) as f64)
            } else {
                String::new()
            };
            eprintln!(
                "  {:>4} {:>9} {:>9}  {:>26}{}",
                g, cap_mask, cap_full, format!("{div:?}"), flag,
            );
        }
    }
    eprintln!(
        "\n  Non-zero divergence ⇒ the structural mask's \u{221e} pattern does NOT match\n  \
         full J-UNIWARD at this geometry, so the probe's feasibility answer is\n  \
         not the emit's. any_divergence = {any_divergence}"
    );
}


/// Is the reported cap hitting the un-derated `hard_max` early return?
///
/// `cap2_stc_trial_max_payload` has two exits. The normal one applies the
/// 5/6 `CAP2_SAFETY_NUM/DEN` derate:
///
/// ```text
///     lo.saturating_mul(CAP2_SAFETY_NUM) / CAP2_SAFETY_DEN
/// ```
///
/// The exponential probe's exit does NOT:
///
/// ```text
///     if hi >= hard_max { if embeddable(hard_max) { return hard_max; } ... }
/// ```
///
/// So a GOP whose capacity is bounded by the COVER (`hard_max = n_cover/8 −
/// header`) rather than by the cost pattern is reported 1.2× higher than the
/// derate intends. This checks whether any real GOP takes that path:
/// `cap == hard_max` means it did.
///
/// Counter-argument to rule out first: `hard_max` implies
/// `w = n_cover/m_total ≈ 1`, i.e. STC with no redundancy, which almost never
/// embeds an arbitrary syndrome — so the branch may be unreachable in
/// practice and the defect latent rather than active. `n_cover` here is the
/// 4-domain combined cover length, matching `combine_cover_4domain`.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_hard_max_derate_bypass_check -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_hard_max_derate_bypass_check() {
    const GOP: u32 = 30;
    // CHUNK_FRAME_FIRST_HEADER_LEN_V3_1
    const HEADER: usize = 8;
    let targets: [(&str, &str, u32, u32, u32); 3] = [
        ("handbag", "Artlist_Handbag.mp4",                1072, 1904, 180),
        ("lumix",   "lumix_g9_1080p_30fps_h264_high.mp4", 1920, 1072, 270),
        ("dji",     "dji_mini2_2_7k_24fps_h264_high.mp4", 2720, 1520, 189),
    ];
    let mut any_bypass = false;
    for (tag, file, w, h, n) in targets {
        let Some(yuv) = crop_yuv(file, w, h, n) else { eprintln!("SKIP {tag}"); continue };
        let Ok(caps) = probe_per_gop_caps(&yuv, w, h, n, GOP, None) else { continue };
        let frame_sz = (w * h * 3 / 2) as usize;
        eprintln!(
            "\n=== {tag} ===\n  {:>4} {:>8} {:>10} {:>9} {:>9}  {}",
            "gop", "cap", "n_cover", "hard_max", "cap*6/5", "bypass?",
        );
        for (g, &cap) in caps.iter().enumerate() {
            let f0 = g * GOP as usize;
            let f1 = ((g + 1) * GOP as usize).min(n as usize);
            if f0 >= f1 { break; }
            let gop_yuv = &yuv[f0 * frame_sz..f1 * frame_sz];
            let gn = (f1 - f0) as u32;
            let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
            let Ok(cover) = h264_walk_cover(gop_yuv, w, h, gn, opts) else { continue };
            let n_cover = cover.coeff_sign_bypass.bits.len()
                + cover.coeff_suffix_lsb.bits.len()
                + cover.mvd_sign_bypass.bits.len()
                + cover.mvd_suffix_lsb.bits.len();
            let hard_max = (n_cover / 8).saturating_sub(HEADER);
            let bypass = cap == hard_max && cap > 0;
            if bypass { any_bypass = true; }
            eprintln!(
                "  {:>4} {:>8} {:>10} {:>9} {:>9}  {}",
                g, cap, n_cover, hard_max, cap * 6 / 5,
                if bypass { "YES — un-derated" } else { "no" },
            );
        }
    }
    eprintln!("\n  any GOP took the un-derated hard_max exit: {any_bypass}");
}


/// Do the probe and the emit walk the SAME cover?
///
/// Everything else is eliminated: tier is capacity-neutral, the structural
/// mask matches full J-UNIWARD (`divergence [0,0,0,0]`), the reproduction used
/// the identical `CAP2_TRIAL_SEED`, verification is never reached (the emit
/// returns `MessageTooLarge`), and no GOP takes the un-derated `hard_max`
/// exit. So the two `stc_embed` calls differ only in cover or in the `w`
/// derived from it.
///
/// The probe walks `clean_encode_gop`'s output; the emit walks
/// `produce_gop_cover`'s wire-only Pass-1 (coefficient capture + fork-state
/// reset). Feasibility is decided on `w = n_cover / m_total`, so ANY length
/// difference means the probe's answer does not bind the emit.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_probe_vs_emit_cover_lengths -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_probe_vs_emit_cover_lengths() {
    use phasm_core::codec::h264::streaming_session::h264_diag_emit_cover_lens;
    const GOP: u32 = 30;
    let targets: [(&str, &str, u32, u32, u32, usize); 2] = [
        ("handbag", "Artlist_Handbag.mp4",                1072, 1904, 180, 2),
        ("lumix",   "lumix_g9_1080p_30fps_h264_high.mp4", 1920, 1072, 270, 6),
    ];
    let mut mismatch = false;
    for (tag, file, w, h, n, gop_idx) in targets {
        let Some(yuv) = crop_yuv(file, w, h, n) else { eprintln!("SKIP {tag}"); continue };
        let frame_sz = (w * h * 3 / 2) as usize;
        let f0 = gop_idx * GOP as usize;
        let f1 = ((gop_idx + 1) * GOP as usize).min(n as usize);
        let gop_yuv = &yuv[f0 * frame_sz..f1 * frame_sz];
        let gn = (f1 - f0) as u32;
        let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };

        let Ok(clean) = h264_walk_cover(gop_yuv, w, h, gn, opts) else {
            eprintln!("{tag}: clean walk failed"); continue
        };
        let probe_lens = [
            clean.coeff_sign_bypass.bits.len(),
            clean.coeff_suffix_lsb.bits.len(),
            clean.mvd_sign_bypass.bits.len(),
            clean.mvd_suffix_lsb.bits.len(),
        ];
        let emit_lens = match h264_diag_emit_cover_lens(gop_yuv, w, h, gn, 26, GOP) {
            Ok(l) => l,
            Err(e) => { eprintln!("{tag}: emit cover failed: {e:?}"); continue }
        };
        let names = ["CoeffSign", "CoeffSuffixLsb", "MvdSign", "MvdSuffixLsb"];
        eprintln!("\n=== {tag} GOP {gop_idx} ===\n  {:<16} {:>10} {:>10} {:>9}",
            "domain", "probe", "emit", "delta");
        for i in 0..4 {
            let d = emit_lens[i] as i64 - probe_lens[i] as i64;
            if d != 0 { mismatch = true; }
            eprintln!("  {:<16} {:>10} {:>10} {:>+9}", names[i], probe_lens[i], emit_lens[i], d);
        }
        let (pt, et): (usize, usize) = (probe_lens.iter().sum(), emit_lens.iter().sum());
        eprintln!("  {:<16} {:>10} {:>10} {:>+9}   n_cover drives w = n_cover/m_total",
            "TOTAL", pt, et, et as i64 - pt as i64);
    }
    eprintln!("\n  cover length mismatch between probe and emit: {mismatch}");
}


/// Do the probe and the emit walk the same cover BY CONTENT, not just length?
///
/// `cap_probe_vs_emit_cover_lengths` showed equal lengths. Costs derive from
/// the magnitudes, so equal lengths with differing magnitudes still produces a
/// different cost vector and a different ∞ pattern — which would make the
/// probe's feasibility answer not bind the emit.
///
/// The docs say this identity is UNPROVEN, not established:
/// `oh264-parallel-gop-encode.md` (2026-06-29) — "`produce_gop_cover` does a
/// full Capture-session Pass-1 (NOT the proven-safe bare `clean_encode_gop`),
/// so B-full.6a's isolation proof does NOT cover it", with B-lite.1 (#890)
/// open. That post-dates `capacity-spreading-plan.md` §16's claim that the
/// probe and encode "call the identical helpers; covers are bit-identical at
/// tier 0" — the two contradict.
///
/// Precedent for this failure mode: #796, where the capacity path and the
/// encode used different walkers and lumix 480p reported 935 B while the
/// encoder accepted 29 B (32×).
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_probe_vs_emit_cover_content -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_probe_vs_emit_cover_content() {
    use phasm_core::codec::h264::streaming_session::h264_diag_emit_cover_checksums;
    const GOP: u32 = 30;

    fn fnv_u8(xs: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &x in xs { h ^= x as u64; h = h.wrapping_mul(0x100_0000_01b3); }
        h
    }
    fn fnv_u16(xs: &[u16]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &x in xs { h ^= x as u64; h = h.wrapping_mul(0x100_0000_01b3); }
        h
    }

    let targets: [(&str, &str, u32, u32, u32, usize); 3] = [
        ("handbag", "Artlist_Handbag.mp4",                1072, 1904, 180, 2),
        ("lumix",   "lumix_g9_1080p_30fps_h264_high.mp4", 1920, 1072, 270, 6),
        ("dji",     "dji_mini2_2_7k_24fps_h264_high.mp4", 2720, 1520, 189, 1),
    ];
    let names = ["CoeffSign", "CoeffSuffixLsb", "MvdSign", "MvdSuffixLsb"];
    let mut any_diff = false;

    for (tag, file, w, h, n, g) in targets {
        let Some(yuv) = crop_yuv(file, w, h, n) else { eprintln!("SKIP {tag}"); continue };
        let frame_sz = (w * h * 3 / 2) as usize;
        let f0 = g * GOP as usize;
        let f1 = ((g + 1) * GOP as usize).min(n as usize);
        let gop_yuv = &yuv[f0 * frame_sz..f1 * frame_sz];
        let gn = (f1 - f0) as u32;
        let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };

        let Ok(clean) = h264_walk_cover(gop_yuv, w, h, gn, opts) else {
            eprintln!("{tag}: clean walk failed"); continue
        };
        let probe = [
            (fnv_u8(&clean.coeff_sign_bypass.bits), fnv_u16(&clean.coeff_sign_bypass.magnitudes)),
            (fnv_u8(&clean.coeff_suffix_lsb.bits), fnv_u16(&clean.coeff_suffix_lsb.magnitudes)),
            (fnv_u8(&clean.mvd_sign_bypass.bits), fnv_u16(&clean.mvd_sign_bypass.magnitudes)),
            (fnv_u8(&clean.mvd_suffix_lsb.bits), fnv_u16(&clean.mvd_suffix_lsb.magnitudes)),
        ];
        let emit = match h264_diag_emit_cover_checksums(gop_yuv, w, h, gn, 26, GOP) {
            Ok(c) => c,
            Err(e) => { eprintln!("{tag}: emit cover failed: {e:?}"); continue }
        };
        eprintln!("\n=== {tag} GOP {g} ===\n  {:<16} {:>8} {:>8}", "domain", "bits", "mags");
        for i in 0..4 {
            let bd = if probe[i].0 == emit[i].0 { "same" } else { "DIFFER" };
            let md = if probe[i].1 == emit[i].1 { "same" } else { "DIFFER" };
            if probe[i].0 != emit[i].0 || probe[i].1 != emit[i].1 { any_diff = true; }
            eprintln!("  {:<16} {:>8} {:>8}", names[i], bd, md);
        }
    }
    eprintln!(
        "\n  probe cover != emit cover (by content): {any_diff}\n  \
         DIFFER in `mags` alone still changes the cost vector, since cost\n  \
         scales with |coeff| — so the ∞ pattern and feasibility can differ."
    );
}


/// THE ROOT CAUSE: embeddability is not monotone in payload size, so the
/// "safety derate" guarantees nothing.
///
/// `cap2_stc_trial_max_payload` bisects for the largest embeddable `lo`, then
/// returns `lo × 5/6`. It proves `embeddable(lo)`. It never tests
/// `embeddable(lo × 5/6)` — and it cannot infer it, because each size is a
/// DIFFERENT message, hence a different syndrome. §16: "Success rate is
/// non-monotonic in size … 1690 isn't a ceiling — it's one 'hard' message."
///
/// Every elimination so far points here. Probe and emit agree on cover
/// (content-identical), costs (`divergence [0,0,0,0]`), hhat seed, tier
/// (forced 0 reproduces), `w`, and framing. The only thing that differs
/// between "probe says 2006 fits" and "emit refuses 2006" is that the probe
/// never asked about 2006.
///
/// This sweeps sizes across the reported cap and prints which embed. A
/// FAIL below a PASS is the proof: capacity is not a scalar with monotone
/// feasibility beneath it, so no derate — and no corpus-fitted margin — can
/// make "under the number" mean "will encode". It also explains the corpus
/// sweep's shape directly (iphone5 0/4 at 60% of Σ while 4/4 at 80%).
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_embeddability_is_non_monotone_in_size -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_embeddability_is_non_monotone_in_size() {
    use phasm_core::codec::h264::streaming_session::h264_diag_gop_emit_once;
    const GOP: u32 = 30;
    const TRIAL_SEED: [u8; 32] = [0x2c; 32];

    fn trial_payload(len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let mut st: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..len {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push((st >> 33) as u8);
        }
        v
    }

    let targets: [(&str, &str, u32, u32, u32, usize); 2] = [
        ("handbag", "Artlist_Handbag.mp4",                1072, 1904, 180, 2),
        ("lumix",   "lumix_g9_1080p_30fps_h264_high.mp4", 1920, 1072, 270, 6),
    ];
    let weights = CostWeights::default();
    let mut non_monotone = false;

    for (tag, file, w, h, n, g) in targets {
        let Some(yuv) = crop_yuv(file, w, h, n) else { eprintln!("SKIP {tag}"); continue };
        let Ok(caps) = probe_per_gop_caps(&yuv, w, h, n, GOP, None) else { continue };
        let Some(&cap) = caps.get(g) else { continue };
        let frame_sz = (w * h * 3 / 2) as usize;
        let f0 = g * GOP as usize;
        let f1 = ((g + 1) * GOP as usize).min(n as usize);
        let gop_yuv = &yuv[f0 * frame_sz..f1 * frame_sz];
        let gn = (f1 - f0) as u32;

        // Sweep from 40% of the reported cap up past it to the pre-derate
        // value (cap × 6/5), which is what the bisect actually proved.
        let hi = cap * 6 / 5;
        let step = (hi / 16).max(1);
        eprintln!("\n=== {tag} GOP {g} | reported cap {cap} (pre-derate ~{hi}) ===");
        let mut results: Vec<(usize, bool)> = Vec::new();
        let mut size = (cap * 2 / 5).max(step);
        while size <= hi {
            let p = trial_payload(size);
            let ok = h264_diag_gop_emit_once(
                gop_yuv, w, h, gn, 26, GOP, &p, 0, size as u32, &TRIAL_SEED, &weights,
            ).map(|(bs, _v)| !bs.is_empty()).unwrap_or(false);
            results.push((size, ok));
            size += step;
        }
        for (sz, ok) in &results {
            let marker = if *sz == cap { "  <= reported cap" } else { "" };
            eprintln!("  {:>7} B  {}{}", sz, if *ok { "embeds" } else { "REFUSED" }, marker);
        }
        // Non-monotone iff some REFUSED size sits below some embedding size.
        if let Some(max_ok) = results.iter().filter(|(_, o)| *o).map(|(s, _)| *s).max() {
            if results.iter().any(|(s, o)| !*o && *s < max_ok) {
                non_monotone = true;
                eprintln!("  => NON-MONOTONE: a smaller payload is refused while {max_ok} B embeds");
            }
        }
    }
    eprintln!(
        "\n  embeddability non-monotone in size: {non_monotone}\n  \
         If true, no derate can make 'under the reported number' safe, because\n  \
         shrinking the payload swaps the syndrome rather than relaxing it."
    );
}


/// Is embeddability-vs-size monotone as the RULE or the EXCEPTION?
///
/// Two GOPs gave opposite answers: lumix GOP 6 is cleanly monotone (embeds to
/// 190 B, refuses from 205 B, reported 213 — a ~6% over-report), while handbag
/// GOP 2 is reproducibly non-monotone (refuses 1852–2152 B yet embeds 2302 B),
/// identically with `PHASM_TIER_OVERRIDE=0`, so `CascadeTier::Auto` is not the
/// cause. Generalising from the worse of two was wrong, and which case
/// dominates decides the fix:
///
/// - mostly MONOTONE ⇒ the probe merely over-reports. Fix at the root by
///   verifying the bisect's returned value and stepping down until it holds.
///   No statistics, no retry — §15-compliant.
/// - substantially NON-MONOTONE ⇒ no reported number can be safe, and §18.4's
///   bounded fresh-salt retry is the only lever that matches.
///
/// Stratified: up to 4 GOPs per fixture, 8 sizes from 40% to 120% of the
/// reported cap. "monotone" means a threshold exists — no refusal below the
/// largest embedding size. For monotone GOPs `thresh/cap` is the honest
/// derate; a value ≥ 1.0 means the report was already safe.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_monotonicity_census -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_monotonicity_census() {
    use phasm_core::codec::h264::streaming_session::h264_diag_gop_emit_once;
    const GOP: u32 = 30;
    const TRIAL_SEED: [u8; 32] = [0x2c; 32];
    const GOPS_PER_FIXTURE: usize = 4;
    const N_SIZES: usize = 8;

    fn trial_payload(len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let mut st: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..len {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push((st >> 33) as u8);
        }
        v
    }

    let fixtures: [(&str, &str, u32, u32, u32); 13] = [
        ("iphone7",       "iphone7_1080p_30fps_h264_high.mov",   1920, 1072, 510),
        ("asia_bottle",   "Artlist_AsiaBottle.mp4",              1072, 1920, 145),
        ("dji",           "dji_mini2_2_7k_24fps_h264_high.mp4",  2720, 1520, 189),
        ("iphone5",       "iphone5_1080p_30fps_h264_high.mov",   1072, 1920, 360),
        ("lumix",         "lumix_g9_1080p_30fps_h264_high.mp4",  1920, 1072, 270),
        ("img4138",       "IMG_4138.MOV",                        1920, 1072, 600),
        ("carplane",      "Artlist_CarPlane.mp4",                1072, 1920, 180),
        ("handbag",       "Artlist_Handbag.mp4",                 1072, 1904, 180),
        ("horse_flag",    "Artlist_HorseFlag.mp4",               1920, 1072, 120),
        ("phone_booth",   "Artlist_PhoneBooth.mp4",              1920, 1072, 120),
        ("pirate_battle", "Artlist_PirateBattle.mp4",            1920, 1072, 120),
        ("school_fight",  "Artlist_SchoolFight.mp4",             1280,  720, 360),
        ("woman_subway",  "Artlist_WomanSubway.mp4",             1280,  720, 360),
    ];
    let weights = CostWeights::default();
    let (mut n_mono, mut n_non) = (0usize, 0usize);
    let mut ratios: Vec<(f64, String)> = Vec::new();

    for (tag, file, w, h, n) in fixtures {
        let Some(yuv) = crop_yuv(file, w, h, n) else { eprintln!("SKIP {tag}"); continue };
        let Ok(caps) = probe_per_gop_caps(&yuv, w, h, n, GOP, None) else { continue };
        let frame_sz = (w * h * 3 / 2) as usize;
        let n_gops = caps.len();
        // Stratified GOP pick: midpoints of GOPS_PER_FIXTURE strata.
        let picks: Vec<usize> = (0..GOPS_PER_FIXTURE.min(n_gops))
            .map(|i| (2 * i + 1) * n_gops / (2 * GOPS_PER_FIXTURE.min(n_gops)))
            .collect();
        for g in picks {
            let Some(&cap) = caps.get(g) else { continue };
            if cap < 64 { continue }
            let f0 = g * GOP as usize;
            let f1 = ((g + 1) * GOP as usize).min(n as usize);
            if f0 >= f1 { continue }
            let gop_yuv = &yuv[f0 * frame_sz..f1 * frame_sz];
            let gn = (f1 - f0) as u32;
            let mut res: Vec<(usize, bool)> = Vec::new();
            for i in 0..N_SIZES {
                // 40% .. 120% of cap
                let size = cap * (40 + 80 * i / (N_SIZES - 1)) / 100;
                if size == 0 { continue }
                let p = trial_payload(size);
                let ok = h264_diag_gop_emit_once(
                    gop_yuv, w, h, gn, 26, GOP, &p, 0, size as u32, &TRIAL_SEED, &weights,
                ).map(|(bs, _)| !bs.is_empty()).unwrap_or(false);
                res.push((size, ok));
            }
            let max_ok = res.iter().filter(|(_, o)| *o).map(|(s, _)| *s).max();
            let monotone = match max_ok {
                None => true, // nothing embedded; vacuously threshold = 0
                Some(mx) => !res.iter().any(|(s, o)| !*o && *s < mx),
            };
            let thresh = max_ok.unwrap_or(0);
            let ratio = thresh as f64 / cap as f64;
            if monotone { n_mono += 1; ratios.push((ratio, format!("{tag}/g{g}"))); }
            else { n_non += 1; }
            eprintln!(
                "  {:<14} g{:<3} cap={:<7} thresh={:<7} ratio={:.2}  {}",
                tag, g, cap, thresh, ratio,
                if monotone { "monotone" } else { "NON-MONOTONE" },
            );
        }
    }

    eprintln!("\n=== monotonicity census ===");
    eprintln!("  monotone GOPs:     {n_mono}");
    eprintln!("  NON-monotone GOPs: {n_non}");
    if !ratios.is_empty() {
        ratios.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let n = ratios.len();
        eprintln!(
            "  monotone thresh/cap: min={:.2} ({})  median={:.2}  max={:.2}",
            ratios[0].0, ratios[0].1, ratios[n / 2].0, ratios[n - 1].0,
        );
        eprintln!("  => a derate of {:.0}% would cover the worst monotone GOP here",
            (1.0 - ratios[0].0) * 100.0);
    }
    eprintln!(
        "  Mostly monotone ⇒ fix the probe (verify its returned value).\n  \
         Substantially non-monotone ⇒ §18.4 fresh-salt retry is required."
    );
}


/// How much of the cover is ∞-cost (wet), and what redundancy does the
/// trellis EFFECTIVELY get?
///
/// `w = n_cover / m_total` counts every position, but ∞-cost positions cannot
/// be flipped — they pin their syndrome bit. The effective redundancy is
/// `n_finite / m_total`. If the wet fraction is large, nominal `w` overstates
/// it badly, and at low effective redundancy sporadic syndrome infeasibility
/// is the expected behaviour rather than an anomaly.
///
/// That would explain the measured shape without anything exotic: 20 of 51
/// GOPs show INTERIOR refusals (a size failing while both smaller and larger
/// sizes embed), the rate varies per GOP, and no derate on the SIZE axis helps
/// because size is the wrong axis.
///
/// Reports, per GOP: the wet fraction overall and per domain, the nominal `w`
/// at the reported cap, and the effective `w`. `eff_w` in single digits while
/// `nom_w` is in the dozens is the finding.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_wet_fraction_census -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_wet_fraction_census() {
    use phasm_core::codec::h264::streaming_session::h264_diag_gop_cost_profile;
    const GOP: u32 = 30;
    const HEADER: usize = 8;

    // The GOPs the monotonicity census classified, mixing non-monotone and
    // clean so the wet fraction can be correlated against behaviour.
    let targets: [(&str, &str, u32, u32, u32, &[usize], &[usize]); 5] = [
        // (tag, file, w, h, n, NON-monotone gops, clean gops)
        ("handbag",     "Artlist_Handbag.mp4",                1072, 1904, 180, &[2, 5], &[0, 3]),
        ("lumix",       "lumix_g9_1080p_30fps_h264_high.mp4", 1920, 1072, 270, &[1],    &[3, 5, 7]),
        ("horse_flag",  "Artlist_HorseFlag.mp4",              1920, 1072, 120, &[2, 3], &[0, 1]),
        ("pirate_battle","Artlist_PirateBattle.mp4",          1920, 1072, 120, &[],     &[0, 1, 2, 3]),
        ("dji",         "dji_mini2_2_7k_24fps_h264_high.mp4", 2720, 1520, 189, &[0],    &[2, 4, 6]),
    ];
    let names = ["CS", "CSL", "MS", "MSL"];
    let mut rows: Vec<(f64, f64, bool, String)> = Vec::new();

    for (tag, file, w, h, n, bad, good) in targets {
        let Some(yuv) = crop_yuv(file, w, h, n) else { eprintln!("SKIP {tag}"); continue };
        let Ok(caps) = probe_per_gop_caps(&yuv, w, h, n, GOP, None) else { continue };
        let frame_sz = (w * h * 3 / 2) as usize;
        eprintln!("\n=== {tag} ===");
        eprintln!("  {:>4} {:>5} {:>8} {:>10} {:>9} {:>7} {:>7}   per-domain wet%",
            "gop", "class", "cap", "n_cover", "wet%", "nom_w", "eff_w");
        for (&g, is_bad) in bad.iter().map(|g| (g, true)).chain(good.iter().map(|g| (g, false))) {
            let Some(&cap) = caps.get(g) else { continue };
            if cap == 0 { continue }
            let f0 = g * GOP as usize;
            let f1 = ((g + 1) * GOP as usize).min(n as usize);
            if f0 >= f1 { continue }
            let gop_yuv = &yuv[f0 * frame_sz..f1 * frame_sz];
            let gn = (f1 - f0) as u32;
            let prof = match h264_diag_gop_cost_profile(gop_yuv, w, h, gn, 26, GOP) {
                Ok(p) => p,
                Err(e) => { eprintln!("  g{g}: {e:?}"); continue }
            };
            let n_cover: usize = prof.iter().map(|(l, _)| *l).sum();
            let n_inf: usize = prof.iter().map(|(_, i)| *i).sum();
            let n_fin = n_cover.saturating_sub(n_inf);
            let m_total = 8 * (HEADER + cap);
            let nom_w = n_cover / m_total.max(1);
            let eff_w = n_fin / m_total.max(1);
            let wet = 100.0 * n_inf as f64 / n_cover.max(1) as f64;
            let per_dom: Vec<String> = prof.iter().zip(names.iter())
                .map(|((l, i), nm)| format!("{nm}={:.0}%", 100.0 * *i as f64 / (*l).max(1) as f64))
                .collect();
            eprintln!(
                "  {:>4} {:>5} {:>8} {:>10} {:>8.1}% {:>7} {:>7}   {}",
                g, if is_bad { "BAD" } else { "ok" }, cap, n_cover, wet, nom_w, eff_w,
                per_dom.join(" "),
            );
            rows.push((wet, eff_w as f64, is_bad, format!("{tag}/g{g}")));
        }
    }

    eprintln!("\n=== does the wet fraction separate BAD from ok? ===");
    let bad: Vec<_> = rows.iter().filter(|r| r.2).collect();
    let ok: Vec<_> = rows.iter().filter(|r| !r.2).collect();
    let mean = |v: &Vec<&(f64, f64, bool, String)>, f: fn(&(f64, f64, bool, String)) -> f64| {
        if v.is_empty() { f64::NAN } else { v.iter().map(|r| f(r)).sum::<f64>() / v.len() as f64 }
    };
    eprintln!("  NON-monotone GOPs: n={} mean wet={:.1}%  mean eff_w={:.1}",
        bad.len(), mean(&bad, |r| r.0), mean(&bad, |r| r.1));
    eprintln!("  clean GOPs:        n={} mean wet={:.1}%  mean eff_w={:.1}",
        ok.len(), mean(&ok, |r| r.0), mean(&ok, |r| r.1));
    eprintln!(
        "\n  High wet% with eff_w far below nom_w ⇒ the capacity arithmetic\n  \
         treats unusable positions as capacity (a real bug, wrong axis for any\n  \
         size-based derate). If wet% is low and similar in both groups, the\n  \
         redundancy is ample and infeasibility is intrinsic to the syndrome."
    );
}


/// At a size that IS refused, does changing only the message bits fix it?
///
/// This is the decisive test for whether pre-encode salt vetting can work. The
/// size, cover, costs, hhat and `w` are all held fixed; only the payload bytes
/// change — which is exactly what a fresh AEAD salt does to the frame (random
/// salt → random nonce/ct → different syndrome, same length).
///
/// - Most draws embed ⇒ infeasibility is a property of the (size, message)
///   instance, vetting candidate salts with `stc_embed` before encoding is
///   cheap and effective, and the expected number of draws is ~1/p.
/// - No draw embeds ⇒ the refusal is structural at that size and no amount of
///   salt selection helps; the cause is elsewhere.
///
/// Sizes are taken from the measured refusals: handbag GOP 2 refused 1252,
/// 1852, 2002 and 2152 B with the fixed trial payload; lumix GOP 6 refused
/// 205 B upward.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_same_size_different_message -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_same_size_different_message() {
    use phasm_core::codec::h264::streaming_session::h264_diag_gop_emit_once;
    const GOP: u32 = 30;
    const TRIAL_SEED: [u8; 32] = [0x2c; 32];
    const K: usize = 8;

    /// Draw `draw` of a pseudo-random payload of exactly `len` bytes. Draw 0
    /// reproduces the probe's own `cap2_trial_payload`, so the known-refused
    /// result must reappear as a control.
    fn payload(len: usize, draw: u64) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let mut st: u64 = if draw == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            0x9E37_79B9_7F4A_7C15 ^ draw.wrapping_mul(0xA076_1D64_78BD_642F)
        };
        for _ in 0..len {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push((st >> 33) as u8);
        }
        v
    }

    // (tag, file, w, h, n, gop, sizes known to refuse with draw 0)
    let targets: [(&str, &str, u32, u32, u32, usize, &[usize]); 2] = [
        ("handbag", "Artlist_Handbag.mp4",                1072, 1904, 180, 2, &[1252, 1852, 2002, 2152]),
        ("lumix",   "lumix_g9_1080p_30fps_h264_high.mp4", 1920, 1072, 270, 6, &[205, 220, 250]),
    ];
    let weights = CostWeights::default();

    for (tag, file, w, h, n, g, sizes) in targets {
        let Some(yuv) = crop_yuv(file, w, h, n) else { eprintln!("SKIP {tag}"); continue };
        let frame_sz = (w * h * 3 / 2) as usize;
        let f0 = g * GOP as usize;
        let f1 = ((g + 1) * GOP as usize).min(n as usize);
        let gop_yuv = &yuv[f0 * frame_sz..f1 * frame_sz];
        let gn = (f1 - f0) as u32;
        eprintln!("\n=== {tag} GOP {g} — same size, {K} different messages ===");
        eprintln!("  {:>7}  {:>8}  {}", "size", "embeds", "per-draw (draw 0 = the probe's trial payload)");
        for &size in sizes {
            let mut marks = String::new();
            let mut n_ok = 0usize;
            for k in 0..K {
                let p = payload(size, k as u64);
                let ok = h264_diag_gop_emit_once(
                    gop_yuv, w, h, gn, 26, GOP, &p, 0, size as u32, &TRIAL_SEED, &weights,
                ).map(|(bs, _)| !bs.is_empty()).unwrap_or(false);
                if ok { n_ok += 1; marks.push('+'); } else { marks.push('.'); }
            }
            eprintln!("  {:>7}  {:>5}/{K}   {}", size, n_ok, marks);
        }
    }
    eprintln!(
        "\n  '+' = embeds, '.' = refused. Draw 0 must be '.' (it is the\n  \
         known-refused control). Any '+' at the same size proves the refusal\n  \
         is instance-specific, so vetting a candidate salt with `stc_embed`\n  \
         before encoding would avoid it at Viterbi cost, not encode cost."
    );
}


/// A/B timing harness for the per-GOP shrink path.
///
/// Measures the two GOPs that shrink DEEPLY — handbag GOP 2 needed four
/// ~12.5% steps (0.875⁴ = 0.587 of its cap) and lumix GOP 6 three — so the
/// loop dominates rather than the one-off `produce_gop_cover`.
///
/// Exists because my first perf claim was an artefact: I compared corpus-gate
/// runs made with `RAYON_NUM_THREADS=4` and `=8` under different background
/// load, and read the 8023 s → 4288 s difference as a win. It was the thread
/// count. This isolates the loop and must be run back-to-back, same thread
/// count, nothing else active.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_shrink_path_timing -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_shrink_path_timing() {
    use phasm_core::codec::h264::streaming_session::h264_diag_verified_gop_consumed;
    use std::time::Instant;
    const GOP: u32 = 30;
    const TRIAL_SEED: [u8; 32] = [0x2c; 32];
    const REPS: usize = 3;

    fn trial_payload(len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        let mut st: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..len {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push((st >> 33) as u8);
        }
        v
    }

    let targets: [(&str, &str, u32, u32, u32, usize); 2] = [
        ("handbag", "Artlist_Handbag.mp4",                1072, 1904, 180, 2),
        ("lumix",   "lumix_g9_1080p_30fps_h264_high.mp4", 1920, 1072, 270, 6),
    ];
    let weights = CostWeights::default();
    let mut grand = 0u128;

    for (tag, file, w, h, n, g) in targets {
        let Some(yuv) = crop_yuv(file, w, h, n) else { eprintln!("SKIP {tag}"); continue };
        let Ok(caps) = probe_per_gop_caps(&yuv, w, h, n, GOP, None) else { continue };
        let Some(&cap) = caps.get(g) else { continue };
        let frame_sz = (w * h * 3 / 2) as usize;
        let f0 = g * GOP as usize;
        let f1 = ((g + 1) * GOP as usize).min(n as usize);
        let gop_yuv = &yuv[f0 * frame_sz..f1 * frame_sz];
        let gn = (f1 - f0) as u32;
        let payload = trial_payload(cap);

        let mut per: Vec<u128> = Vec::new();
        let mut consumed_last = 0usize;
        for _ in 0..REPS {
            let t = Instant::now();
            let c = h264_diag_verified_gop_consumed(
                gop_yuv, w, h, gn, 26, GOP, &payload, 0, cap as u32, &TRIAL_SEED, &weights,
            ).unwrap_or(0);
            per.push(t.elapsed().as_millis());
            consumed_last = c;
        }
        let total: u128 = per.iter().sum();
        let min = per.iter().copied().min().unwrap_or(0);
        grand += total;
        eprintln!(
            "  {:<9} g{:<2} cap={:<6} consumed={:<6} ({:.3})  reps={:?} ms  min={min} total={total}",
            tag, g, cap, consumed_last, consumed_last as f64 / cap as f64, per,
        );
    }
    eprintln!("\n  GRAND TOTAL {grand} ms   (compare across A/B runs; min per GOP is the\n  \
               noise-resistant figure)");
}


/// Does pointing the HUD at the planner's accept ceiling actually make the
/// displayed number honest?
///
/// Three quantities per fixture:
/// - `sigma`  — what the HUD showed before: `Σ caps − FRAME_OVERHEAD`.
/// - `achiev` — what it shows now: `max_message_len` − FRAME_OVERHEAD, i.e.
///   the planner's own accept ceiling.
/// - `reliable` — the largest fraction of Σ that round-tripped 4/4 in
///   `cap_proportional_success_rate_on_the_corpus` (hardcoded from that run,
///   since re-measuring it costs ~70 min).
///
/// `achiev < sigma` closes the "HUD promises what the planner rejects" gap.
/// `achiev > reliable` means it is still not a reliability promise — the
/// remaining gap is the interior-infeasibility problem no mechanism tested
/// today addressed, and this test exists so that limit stays visible instead
/// of being quietly assumed away.
///
/// Run:
///   cargo test --test cap_realworld_810 --features h264-encoder --release \
///     cap_hud_number_vs_reliable -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn cap_hud_number_vs_reliable() {
    const GOP: u32 = 30;
    // (tag, file, w, h, n, reliable fraction of Sigma from propgate run 3)
    let fixtures: [(&str, &str, u32, u32, u32, f64); 13] = [
        ("iphone7",       "iphone7_1080p_30fps_h264_high.mov",   1920, 1072, 510, 0.80),
        ("asia_bottle",   "Artlist_AsiaBottle.mp4",              1072, 1920, 145, 0.80),
        ("dji",           "dji_mini2_2_7k_24fps_h264_high.mp4",  2720, 1520, 189, 0.80),
        ("iphone5",       "iphone5_1080p_30fps_h264_high.mov",   1072, 1920, 360, 0.80),
        ("lumix",         "lumix_g9_1080p_30fps_h264_high.mp4",  1920, 1072, 270, 0.60),
        ("img4138",       "IMG_4138.MOV",                        1920, 1072, 600, 0.80),
        ("carplane",      "Artlist_CarPlane.mp4",                1072, 1920, 180, 0.80),
        ("handbag",       "Artlist_Handbag.mp4",                 1072, 1904, 180, 0.30),
        ("horse_flag",    "Artlist_HorseFlag.mp4",               1920, 1072, 120, 0.45),
        ("phone_booth",   "Artlist_PhoneBooth.mp4",              1920, 1072, 120, 0.80),
        ("pirate_battle", "Artlist_PirateBattle.mp4",            1920, 1072, 120, 0.80),
        ("school_fight",  "Artlist_SchoolFight.mp4",             1280,  720, 360, 0.60),
        ("woman_subway",  "Artlist_WomanSubway.mp4",             1280,  720, 360, 0.80),
    ];
    let cal = AllocationCalibration::H264_1080P_QP26;
    let mut still_over = 0usize;
    let mut rows = 0usize;
    eprintln!(
        "\n  {:<14} {:>9} {:>9} {:>9}  {:>7} {:>8}",
        "fixture", "sigma-50", "achiev", "reliable", "a/sig", "a/reli",
    );
    for (tag, file, w, h, n, rel) in fixtures {
        let Some(yuv) = crop_yuv(file, w, h, n) else { eprintln!("  SKIP {tag}"); continue };
        let Ok(caps) = probe_per_gop_caps(&yuv, w, h, n, GOP, None) else { continue };
        let sigma: usize = caps.iter().sum();
        let samples: Vec<(usize, usize)> = caps.iter().copied().enumerate().collect();
        let achiev = phasm_core::stego::balanced_allocation::max_message_len(
            &samples, caps.len(), &cal,
        ).saturating_sub(50);
        let old = sigma.saturating_sub(50);
        let reliable = (sigma as f64 * rel) as usize;
        rows += 1;
        if achiev > reliable { still_over += 1; }
        eprintln!(
            "  {:<14} {:>9} {:>9} {:>9}  {:>6.2}x {:>7.2}x{}",
            tag, old, achiev, reliable,
            achiev as f64 / old.max(1) as f64,
            achiev as f64 / reliable.max(1) as f64,
            if achiev > reliable { "  <= still over-promises" } else { "" },
        );
    }
    eprintln!(
        "\n  {still_over}/{rows} fixtures: the NEW displayed number still exceeds the\n  \
         corpus-measured reliable level. Pointing the HUD at the planner closes\n  \
         the sigma-vs-planner gap; it does NOT close the planner-vs-reliable gap."
    );
}
