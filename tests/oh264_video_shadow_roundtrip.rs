// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! WV.6 round-trip GATE — OH264 video primary + shadow round-trip.
//!
//! Encode a synthetic clip with a primary AND one shadow through the
//! per-GOP streaming shadow path (`h264_encode_with_shadows_streaming`
//! driven by a `SliceYuvSource`), then decode
//! BOTH via `StreamingDecodeSession` (primary passphrase recovers the
//! primary; shadow passphrase recovers the shadow). This is the gate the
//! WV.6.a refactor (per-GOP primary in shadow mode,
//! `docs/design/video/h264/oh264-wv6-streaming-shadow-unification.md`) must
//! keep green: today the primary is a whole-video `frame_bits` STC; after
//! WV.6.a it is a per-GOP `chunk_frame`. Either way both messages must come
//! back. (No prior OH264 *video* shadow round-trip test existed — the older
//! shadow_roundtrip.rs covers IMAGE shadows only.)
//!
//! `#[ignore]`d by default — the whole-video shadow encode runs the OH264
//! encoder over synthetic YUV (~3-6 s) and wants `--test-threads=1`. Run:
//!   cargo test -p phasm-core --features h264-encoder --test \
//!     oh264_video_shadow_roundtrip -- --ignored --test-threads=1

#![cfg(feature = "h264-encoder")]

use phasm_core::codec::h264::openh264_stego::{
    h264_encode_with_shadows_streaming, h264_gop_capacity_per_tier_with_costs, EncodeOpts,
    SliceYuvSource,
};
use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingDecodeSession,
    StreamingProbeSession, YuvFrameRef,
};
use phasm_core::stego::payload::compressed_payload_size;
use phasm_core::codec::h264::stego::CostWeights;
use phasm_core::stego::balanced_allocation::{max_message_len, plan_safe_balanced, PlanOutcome};
use phasm_core::stego::calibration::AllocationCalibration;
use phasm_core::stego::shadow_layer::ShadowLayer;

/// Same deterministic textured YUV generator as the #530 repro.
fn synth_yuv(width: u32, height: u32, frame_idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = Vec::with_capacity((width * height) as usize);
    for j in 0..height {
        for i in 0..width {
            y.push(((i + frame_idx * 2) ^ (j + frame_idx * 3)) as u8);
        }
    }
    let (half_w, half_h) = (width / 2, height / 2);
    let mut u = Vec::with_capacity((half_w * half_h) as usize);
    let mut v = Vec::with_capacity((half_w * half_h) as usize);
    let mut s: u32 = 0xCAFE_F00D ^ frame_idx;
    for j in 0..half_h {
        for i in 0..half_w {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let pos = (i + j + frame_idx) as u8;
            u.push(((s >> 16) as u8).wrapping_add(pos));
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(((s >> 16) as u8).wrapping_add(pos));
        }
    }
    (y, u, v)
}

fn run_shadow_roundtrip(
    width: u32,
    height: u32,
    gop_size: u32,
    n_frames: u32,
    primary_msg: &str,
    primary_pass: &str,
    shadow_msg: &str,
    shadow_pass: &str,
    // D2: when true, MEASURE this clip's per-GOP capacities and hand them to
    // the encoder so the primary is planned proportionally instead of split
    // evenly. The plan changes which GOP carries which slice of the payload,
    // so it is a wire-format change in the only sense that matters here —
    // the decoder must still reassemble the payload from the chunk headers.
    plan_from_measured_caps: bool,
) {
    // Assemble the whole-clip tight-I420 buffer (same synthetic content as
    // before) and drive the per-GOP streaming shadow encoder through a
    // `SliceYuvSource` — the byte-identical stand-in for the bridge's real
    // pull source. WV.6.g retired the push session; the streaming pull path
    // is the only shadow-video encode entry now.
    let frame_bytes = (width * height * 3 / 2) as usize;
    let mut yuv = Vec::with_capacity(frame_bytes * n_frames as usize);
    for f in 0..n_frames {
        let (y, u, v) = synth_yuv(width, height, f);
        yuv.extend_from_slice(&y);
        yuv.extend_from_slice(&u);
        yuv.extend_from_slice(&v);
    }
    // Primary must be the LARGEST payload so the n-shadow auto-sort doesn't
    // swap it with a shadow (mirrors shadow_roundtrip.rs's image gate).
    let shadows = [ShadowLayer {
        message: shadow_msg,
        passphrase: shadow_pass,
        files: &[],
    }];
    let opts = EncodeOpts { qp: 26, intra_period: gop_size as i32 };

    // D2 — the measured per-GOP cap vector, or none at all.
    let caps: Option<Vec<usize>> = if plan_from_measured_caps {
        let frame_size = (width * height * 3 / 2) as usize;
        let n_gops = n_frames.div_ceil(gop_size);
        let measured: Vec<usize> = (0..n_gops)
            .map(|g| {
                let f0 = (g * gop_size) as usize;
                let f1 = (((g + 1) * gop_size).min(n_frames)) as usize;
                let per_tier = h264_gop_capacity_per_tier_with_costs(
                    &yuv[f0 * frame_size..f1 * frame_size],
                    width,
                    height,
                    (f1 - f0) as u32,
                    opts,
                    &CostWeights::default(),
                    /* full_tiers */ false,
                    /* full_juniward_costs */ false,
                )
                .expect("per-GOP capacity");
                per_tier[0]
            })
            .collect();
        eprintln!("D2 measured per-GOP caps: {measured:?} (Σ {})", measured.iter().sum::<usize>());
        assert!(
            measured.iter().all(|&c| c > 0),
            "a zero cap means the probe failed, not that the plan works: {measured:?}",
        );
        // Confirm the planner actually produces a plan for this payload, so a
        // green round-trip cannot be the even-split fallback in disguise.
        let samples: Vec<(usize, usize)> =
            measured.iter().copied().enumerate().collect();
        match plan_safe_balanced(
            &samples,
            n_gops as usize,
            primary_msg.len(),
            gop_size as usize,
            &AllocationCalibration::H264_1080P_QP26,
        ) {
            PlanOutcome::Plan { plan, .. } => {
                eprintln!("D2 plan: {plan:?}");
            }
            other => panic!(
                "the planner refused a {} B primary on caps {measured:?}: {other:?} —                  this test would then be measuring the even-split fallback",
                primary_msg.len(),
            ),
        }
        Some(measured)
    } else {
        None
    };

    let mut src = SliceYuvSource::new(&yuv, width, height, n_frames, gop_size);
    let annex_b = h264_encode_with_shadows_streaming(
        &mut src,
        width,
        height,
        n_frames,
        opts,
        primary_msg,
        &[],
        primary_pass,
        &shadows,
        &CostWeights::default(),
        caps.as_deref(),
        None,
    )
    .expect("streaming shadow encode");
    assert!(!annex_b.is_empty(), "empty Annex-B");

    // Primary passphrase → primary message.
    let mut dp = StreamingDecodeSession::create(primary_pass).expect("primary decode session");
    dp.push_annex_b(&annex_b).expect("push annex-b (primary)");
    let rp = dp.finish().expect("primary decode");
    assert_eq!(rp.text, primary_msg, "primary message mismatch");

    // Shadow passphrase → shadow message.
    let mut ds = StreamingDecodeSession::create(shadow_pass).expect("shadow decode session");
    ds.push_annex_b(&annex_b).expect("push annex-b (shadow)");
    let rs = ds.finish().expect("shadow decode");
    assert_eq!(rs.text, shadow_msg, "shadow message mismatch");

    eprintln!(
        "OH264 video shadow round-trip GREEN: {width}x{height} x {n_frames}f gop={gop_size} \
         (primary {} B, shadow {} B, {} annex_b bytes)",
        primary_msg.len(),
        shadow_msg.len(),
        annex_b.len()
    );
}

#[test]
#[ignore = "OH264 shadow encode ~5s; run with --ignored --test-threads=1"]
fn oh264_video_primary_plus_one_shadow_roundtrip() {
    // 320x240 x 20f @ gop=10 → 2 GOPs. Primary longer than the shadow.
    run_shadow_roundtrip(
        320, 240, /*gop=*/ 10, /*n=*/ 20,
        "primary message, longer than the shadow", "primary-pass",
        "shdw", "s",
        /*plan_from_measured_caps=*/ false,
    );
}

/// D1 regression (2026-10-20) — `h264_encode_with_shadows_streaming` must
/// REFUSE an empty shadow slice instead of quietly reassembling the whole
/// decoded clip and delegating to the whole-clip encoder.
///
/// That fallback was reachable from both mobile bridges: the encode
/// coordinators routed any clip with a FILE ATTACHMENT here even with zero
/// shadows, because the no-shadow session create FFI had no files parameter.
/// It cost ~1.68 GB of buffered YUV on an 18 s 1080p clip (the kernel killed
/// the app) and it allocated the payload with the legacy even split, whose
/// ceiling is `n_gops * min(cap)` rather than the proportional planner's
/// `~0.91 * sum(cap)` that the capacity bar displays — so an attachment the
/// green bar promised would fit came back `MessageTooLarge`.
///
/// Both bridges now carry files through
/// `StreamingEncodeSession::create_with_files`. Nothing should ask this entry
/// point for a primary-only encode again, and if something does it must say so
/// rather than silently taking the expensive, differently-allocated route.
///
/// Fast: the guard fires before any encode work, so this needs no `#[ignore]`.
#[test]
fn d1_streaming_shadow_encode_refuses_empty_shadows() {
    const W: u32 = 64;
    const H: u32 = 64;
    const N: u32 = 4;
    let frame_bytes = (W * H * 3 / 2) as usize;
    let mut yuv = Vec::with_capacity(frame_bytes * N as usize);
    for f in 0..N {
        let (y, u, v) = synth_yuv(W, H, f);
        yuv.extend_from_slice(&y);
        yuv.extend_from_slice(&u);
        yuv.extend_from_slice(&v);
    }
    let opts = EncodeOpts { qp: 26, intra_period: 2 };
    let mut src = SliceYuvSource::new(&yuv, W, H, N, 2);
    let err = h264_encode_with_shadows_streaming(
        &mut src, W, H, N, opts,
        "primary only", &[], "primary-pass",
        &[], // no shadows — the misuse this guard catches
        &CostWeights::default(),
        None,
        None,
    )
    .expect_err("empty shadows must be refused, not silently rerouted");
    let msg = err.to_string();
    assert!(
        msg.contains("at least one shadow"),
        "the refusal must name the cause and point at the right API, got: {msg}"
    );
    assert!(
        msg.contains("create_with_files"),
        "the refusal must point at the planned O(GOP) alternative, got: {msg}"
    );
}

/// D2 — the same clip, but with the PRIMARY planned against measured per-GOP
/// capacities instead of split evenly. This is the end-to-end half of the D2
/// fix: `d2_shadow_primary_plan.rs` shows the plan reaches a capacity the even
/// split cannot, and this shows the plan's different payload distribution
/// still decodes. Both halves are needed — a plan that lifts capacity and
/// desyncs the decoder would pass the first gate alone.
///
/// Why it cannot desync by construction: `chunk_frame` v3.1 carries the total
/// byte count in the first chunk and a length in each later one, and the
/// decoder accumulates until `accumulated == total_bytes`. Any split across
/// GOPs is therefore decode-compatible. This test is what makes that claim
/// an observation rather than an argument.
#[test]
#[ignore = "OH264 shadow encode + per-GOP capacity probe ~10s; run with --ignored --test-threads=1"]
fn d2_oh264_video_shadow_roundtrip_with_proportional_primary_plan() {
    run_shadow_roundtrip(
        320, 240, /*gop=*/ 10, /*n=*/ 20,
        "primary message, longer than the shadow", "primary-pass",
        "shdw", "s",
        /*plan_from_measured_caps=*/ true,
    );
}

/// Printable ASCII from a per-`seed` LCG. Prefix-stable in `len`, so a
/// longer text is the shorter one extended — which keeps the framed size
/// (near-)monotone in `len` for the search below.
fn lcg_text(len: usize, seed: u64) -> String {
    let mut st = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..len)
        .map(|_| {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (0x21u8 + ((st >> 33) % 0x5d) as u8) as char
        })
        .collect()
}

/// The longest `lcg_text(.., seed)` whose FRAMED payload — the unit every
/// shadow-capacity figure is denominated in, i.e. after the Brotli pass in
/// `stego::payload` — is ≤ `target`. Random ASCII frames to ~0.83× its raw
/// length, so sizing by `.len()` would test ~17% below the cap and prove
/// nothing (the D9 lesson).
fn text_with_framed_size(target: usize, seed: u64) -> String {
    let fits = |l: usize| compressed_payload_size(&lcg_text(l, seed), &[]) <= target;
    let (mut lo, mut hi) = (1usize, target * 2 + 64);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(mid) { lo = mid } else { hi = mid - 1 }
    }
    // Brotli is only near-monotone in length: walk up past any local dip.
    let mut best = lo;
    for l in lo + 1..lo + 64 {
        if fits(l) {
            best = l;
        }
    }
    lcg_text(best, seed)
}

/// Encode `primary` + `shadows` (one passphrase each) and report whether the
/// primary AND every shadow decode back exactly. `Err` = refused at encode.
fn shadows_round_trip(
    yuv: &[u8],
    w: u32,
    h: u32,
    nf: u32,
    gop: u32,
    primary: &str,
    shadows: &[(String, String)],
) -> Result<bool, String> {
    let layers: Vec<ShadowLayer> = shadows
        .iter()
        .map(|(m, p)| ShadowLayer { message: m, passphrase: p, files: &[] })
        .collect();
    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };
    let mut src = SliceYuvSource::new(yuv, w, h, nf, gop);
    let annex_b = h264_encode_with_shadows_streaming(
        &mut src, w, h, nf, opts, primary, &[], "d10-primary", &layers,
        &CostWeights::default(), None, None,
    )
    .map_err(|e| format!("{e:?}"))?;
    let decode = |pass: &str| -> Option<String> {
        let mut d = StreamingDecodeSession::create(pass).ok()?;
        d.push_annex_b(&annex_b).ok()?;
        d.finish().ok().map(|r| r.text)
    };
    if decode("d10-primary").as_deref() != Some(primary) {
        return Ok(false);
    }
    Ok(shadows.iter().all(|(m, p)| decode(p).as_deref() == Some(m.as_str())))
}

/// D10 + D19 (2026-10-23) — N = 2, 3 and 4 shadows, each at EXACTLY the
/// per-shadow capacity the probe reports, all decode.
///
/// D10 raised the shadow pool from CoeffSign + CoeffSuffixLsb + MvdSign to the
/// full eligible set, adding the cascade-safe MvdSuffixLsb positions the
/// encoder always drew on, so the per-shadow figure `√(1024·pool/(N−1))` rose
/// with it. For ONE shadow that bound sits orders of magnitude below the pool
/// and nothing is tight. With N ≥ 2 it is the collision budget the RS parity
/// is meant to absorb, and the encoder does not pre-flight it — it emits,
/// verifies, and refuses with `ShadowEmbedFailed` — so the displayed number is
/// the user's only guide. Each shadow's FRAMED payload is sized to the
/// reported cap and both must come back.
///
/// N = 3 and 4 joined with D19. Before it, N = 3 was refused already at 85%
/// of its reported cap and N = 4 lower still: the √ formula assumes one
/// 128-byte parity block, while the shadow frame is RS-coded in 255-byte
/// blocks carrying `parity_len` bytes EACH
/// (`d10_instrument_n_shadow_cap_fraction_sweep` is that measurement). The
/// figure is now capped by the collision model at
/// `SHADOW_RELIABILITY_TARGET` (0.95).
///
/// That makes the cap a ≥ 95% event, not a certainty, and every run is a
/// fresh draw: the shadow ciphertext is salted and nonced from `thread_rng`
/// on each encode (`crypto::encrypt`), so which overwritten bits come out
/// wrong changes run to run even with these fixed messages and passphrases.
/// Asserting one draw per N would fail about one run in eight over N = 2, 3,
/// 4, so each N gets ONE redraw, and only after an encoder refusal
/// (`ShadowEmbedFailed` — the self-verify caught it, nothing was emitted);
/// two refusals in a row is ≤ 0.25% per N at the target and near-certain at
/// the pre-D19 figures. A shadow that encodes but then fails to decode is
/// never retried: the encoder verifies before it emits, so that is a bug.
/// For the rate itself, run `d19_instrument_success_rate_at_new_cap`.
///
/// Also PRINTS, without asserting, whether +5% / +10% over the cap still
/// decodes. Not asserted: collision survival above the budget is a
/// probability, not a contract.
#[test]
#[ignore = "probe + 3 x (1-2 shadow encodes + 2 headroom runs), ~6 min; run with --ignored --test-threads=1"]
fn d10_n_shadows_at_the_reported_per_shadow_capacity_roundtrip() {
    const W: u32 = 320;
    const H: u32 = 192;
    const GOP: u32 = 30;
    const NF: u32 = 60;
    let frames: Vec<_> = (0..NF).map(|f| synth_yuv(W, H, f)).collect();

    let mut probe = StreamingProbeSession::create(EncodeSessionParams {
        width: W,
        height: H,
        fps_num: 30,
        fps_den: 1,
        qp: 26,
        gop_size: GOP,
        total_frames_hint: NF,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        cost_weights: CostWeights::default(),
        progress_callback: None,
    })
    .expect("probe session");
    for (y, u, v) in &frames {
        probe
            .push_frame(YuvFrameRef {
                y,
                y_stride: W as usize,
                u,
                u_stride: (W / 2) as usize,
                v,
                v_stride: (W / 2) as usize,
            })
            .expect("probe push");
    }
    let probed = probe.finish().expect("probe finish");

    let mut yuv = Vec::new();
    for (y, u, v) in &frames {
        yuv.extend_from_slice(y);
        yuv.extend_from_slice(u);
        yuv.extend_from_slice(v);
    }

    let ns: &[u32] = &[2, 3, 4];
    for &n in ns {
        let cap = probed.shadow_max_message_bytes(n);
        assert!(cap > 0, "n={n}: zero per-shadow capacity — the probe found no pool");
        let build = |target: usize| -> Vec<(String, String)> {
            (0..n)
                .map(|k| {
                    let m = text_with_framed_size(target, 0xD10 + k as u64 * 7919);
                    (m, format!("d10-shadow-{k}"))
                })
                .collect()
        };

        let at_cap = build(cap);
        for (m, _) in &at_cap {
            let framed = compressed_payload_size(m, &[]);
            assert!(
                framed <= cap && framed + 4 >= cap,
                "n={n}: sized a shadow to {framed} framed B, not the {cap} B cap",
            );
        }
        let mut verdict = shadows_round_trip(&yuv, W, H, NF, GOP, "d10 primary", &at_cap);
        if matches!(&verdict, Err(e) if e.contains("ShadowEmbedFailed")) {
            eprintln!("D10 n={n}: refused at the {cap} B cap on the first draw — one redraw (fresh salts)");
            verdict = shadows_round_trip(&yuv, W, H, NF, GOP, "d10 primary", &at_cap);
        }
        match verdict {
            Ok(true) => eprintln!("D10 n={n}: {n} shadows x {cap} B framed (the reported cap) — ALL DECODE"),
            Ok(false) => panic!("n={n}: a shadow at the reported {cap} B cap failed to decode"),
            Err(e) => panic!("n={n}: encode refused shadows at the reported {cap} B cap (twice if ShadowEmbedFailed): {e}"),
        }

        for pct in [105usize, 110] {
            let target = cap * pct / 100;
            let verdict = match shadows_round_trip(&yuv, W, H, NF, GOP, "d10 primary", &build(target)) {
                Ok(true) => "all decode".to_string(),
                Ok(false) => "DECODE FAIL".to_string(),
                Err(e) => format!("refused: {e}"),
            };
            eprintln!("D10 n={n}: headroom {pct}% of cap ({target} B framed): {verdict}");
        }
    }
}

/// Instrument (prints, asserts nothing): for N = 3 and 4 shadows, which
/// fraction of the reported per-shadow cap actually embeds. Built to settle
/// whether the N-aware shadow figure over-promises independently of D10:
/// D10 moved the pool by a few percent, so a refusal well below the old cap
/// cannot be D10's doing. It is the measurement behind D19.
///
/// First run (2026-10-23, pool 1 725 747 bits): N = 3 cap 3541 B refused at
/// 97/93/85%, decoded at 75/60/45%.
#[test]
#[ignore = "instrument: many shadow encodes, ~10-20 min; run with --ignored --test-threads=1 --nocapture"]
fn d10_instrument_n_shadow_cap_fraction_sweep() {
    const W: u32 = 320;
    const H: u32 = 192;
    const GOP: u32 = 30;
    const NF: u32 = 60;
    let frames: Vec<_> = (0..NF).map(|f| synth_yuv(W, H, f)).collect();
    let mut probe = StreamingProbeSession::create(EncodeSessionParams {
        width: W, height: H, fps_num: 30, fps_den: 1, qp: 26, gop_size: GOP,
        total_frames_hint: NF, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    })
    .expect("probe session");
    for (y, u, v) in &frames {
        probe
            .push_frame(YuvFrameRef {
                y, y_stride: W as usize, u, u_stride: (W / 2) as usize, v, v_stride: (W / 2) as usize,
            })
            .expect("probe push");
    }
    let probed = probe.finish().expect("probe finish");
    eprintln!("SWEEP pool = {} bits", probed.shadow_pool_bits);
    let mut yuv = Vec::new();
    for (y, u, v) in &frames {
        yuv.extend_from_slice(y);
        yuv.extend_from_slice(u);
        yuv.extend_from_slice(v);
    }
    for n in [3u32, 4] {
        let cap = probed.shadow_max_message_bytes(n);
        for pct in [97usize, 93, 85, 75, 60, 45] {
            let target = cap * pct / 100;
            let shadows: Vec<(String, String)> = (0..n)
                .map(|k| (text_with_framed_size(target, 0xD10 + k as u64 * 7919), format!("sw-{k}")))
                .collect();
            let verdict = match shadows_round_trip(&yuv, W, H, NF, GOP, "sweep primary", &shadows) {
                Ok(true) => "all decode".to_string(),
                Ok(false) => "DECODE FAIL".to_string(),
                Err(e) => format!("refused: {e}"),
            };
            eprintln!("SWEEP n={n} cap={cap} B: {pct}% ({target} B framed) -> {verdict}");
        }
    }
}

/// Instrument (prints, asserts nothing): the SUCCESS RATE of two shadows at
/// the reported per-shadow cap, over K independent passphrase sets.
///
/// Shadow positions follow each passphrase and shadow bits follow the random
/// AES salt, so one round-trip at the cap is one draw, not a property. Built
/// after the D10 gate passed at N = 2 while a per-block collision model
/// (Poisson byte errors per 255-byte RS block vs `parity_len/2` correctable)
/// put N = 2 at the cap close to a coin flip.
///
/// First run (2026-10-23, synthetic 320×192×60, cap 5080 B): 8/8 at 100%,
/// 8/8 at 90%, 8/8 at 80% — while +5% and +10% were refused in the D10
/// instrument. The cliff is sharp and sits just above the cap, so the
/// per-block Poisson model is too pessimistic at N = 2 (it predicted ~0.9),
/// and D19's corrected formula has to be calibrated on measured rates.
#[test]
#[ignore = "instrument: K shadow encodes at the cap, ~15 min; run with --ignored --test-threads=1 --nocapture"]
fn d10_instrument_two_shadow_success_rate_at_cap() {
    const W: u32 = 320;
    const H: u32 = 192;
    const GOP: u32 = 30;
    const NF: u32 = 60;
    const K: u64 = 8;
    let frames: Vec<_> = (0..NF).map(|f| synth_yuv(W, H, f)).collect();
    let mut probe = StreamingProbeSession::create(EncodeSessionParams {
        width: W, height: H, fps_num: 30, fps_den: 1, qp: 26, gop_size: GOP,
        total_frames_hint: NF, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    })
    .expect("probe session");
    for (y, u, v) in &frames {
        probe
            .push_frame(YuvFrameRef {
                y, y_stride: W as usize, u, u_stride: (W / 2) as usize, v, v_stride: (W / 2) as usize,
            })
            .expect("probe push");
    }
    let probed = probe.finish().expect("probe finish");
    let mut yuv = Vec::new();
    for (y, u, v) in &frames {
        yuv.extend_from_slice(y);
        yuv.extend_from_slice(u);
        yuv.extend_from_slice(v);
    }
    let cap = probed.shadow_max_message_bytes(2);
    for pct in [100usize, 90, 80] {
        let target = cap * pct / 100;
        let mut ok = 0;
        for trial in 0..K {
            let shadows: Vec<(String, String)> = (0..2u64)
                .map(|k| {
                    (
                        text_with_framed_size(target, 0x2_0000 + trial * 31 + k),
                        format!("rate-{trial}-{k}"),
                    )
                })
                .collect();
            let verdict = shadows_round_trip(&yuv, W, H, NF, GOP, "rate primary", &shadows);
            if matches!(verdict, Ok(true)) {
                ok += 1;
            }
            eprintln!("RATE n=2 {pct}% of {cap} B, trial {trial}: {verdict:?}");
        }
        eprintln!("RATE n=2 {pct}% of cap ({target} B framed): {ok}/{K} embed + decode");
    }
}

/// A primary that cannot fit must be REFUSED, never emitted truncated.
///
/// The streaming shadow path's per-GOP drain shrinks a GOP's ask until it
/// fits — down to zero — and carries the rest forward. The last GOP is asked
/// for the whole remainder, but if even that shrinks, nothing after it can
/// take the leftover, and neither sweep B nor the shadow-only verify checks
/// that the primary cursor reached the end. Before this test the encode then
/// returned `Ok` with a primary the decoder can never reassemble (its first
/// chunk announces `total_bytes`; fewer arrive). That is silent loss of the
/// message the user typed first.
#[test]
#[ignore = "OH264 shadow encode ~1 min; run with --ignored --test-threads=1"]
fn oversized_primary_is_refused_not_emitted_truncated() {
    const W: u32 = 320;
    const H: u32 = 192;
    const GOP: u32 = 30;
    const NF: u32 = 60;
    let mut yuv = Vec::new();
    for f in 0..NF {
        let (y, u, v) = synth_yuv(W, H, f);
        yuv.extend_from_slice(&y);
        yuv.extend_from_slice(&u);
        yuv.extend_from_slice(&v);
    }
    // Far beyond this clip's primary capacity (a few hundred bytes).
    let primary = text_with_framed_size(40_000, 0xBADF00D);
    let shadows = vec![("short shadow".to_string(), "oversize-shadow".to_string())];
    match shadows_round_trip(&yuv, W, H, NF, GOP, &primary, &shadows) {
        Err(e) => {
            assert!(
                e.contains("MessageTooLarge"),
                "an oversized primary must be refused as MessageTooLarge, got: {e}",
            );
        }
        Ok(true) => panic!("a 40 KB primary round-tripped through a ~0.5 KB clip?"),
        Ok(false) => panic!(
            "SILENT PRIMARY LOSS: the encoder returned Ok for a primary it could not fit, \
             and the stream does not decode to it"
        ),
    }
}

/// The primary dry run predicts sweep B's drain EXACTLY — checked where it
/// matters: a primary close to the shown ceiling plus a maximum-size shadow,
/// so the shadows' fenced slots force the drain to shrink and carry.
///
/// Meant for a DEBUG run: `h264_encode_with_shadows_streaming`
/// `debug_assert`s that sweep B's final primary cursor equals the dry run's at
/// every tier it attempts, so a divergence panics here. In any build the
/// outcome must not be silent — `Ok` must decode both messages; a refusal
/// must be `MessageTooLarge` (primary) or `ShadowEmbedFailed` (shadow).
///   cargo test --features h264-encoder --test oh264_video_shadow_roundtrip \
///     primary_dry_run -- --ignored --test-threads=1 --nocapture
#[test]
#[ignore = "debug-mode equivalence check, several shadow encodes; run with --ignored --test-threads=1"]
fn primary_dry_run_matches_sweep_b_near_capacity() {
    const W: u32 = 320;
    const H: u32 = 192;
    const GOP: u32 = 30;
    const NF: u32 = 60;
    let frames: Vec<_> = (0..NF).map(|f| synth_yuv(W, H, f)).collect();
    let mut probe = StreamingProbeSession::create(EncodeSessionParams {
        width: W, height: H, fps_num: 30, fps_den: 1, qp: 26, gop_size: GOP,
        total_frames_hint: NF, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    })
    .expect("probe session");
    for (y, u, v) in &frames {
        probe
            .push_frame(YuvFrameRef {
                y, y_stride: W as usize, u, u_stride: (W / 2) as usize, v, v_stride: (W / 2) as usize,
            })
            .expect("probe push");
    }
    let (probed, caps) = probe.finish_with_per_gop().expect("probe finish");
    let samples: Vec<(usize, usize)> = caps.iter().copied().enumerate().collect();
    let ceiling = max_message_len(&samples, caps.len(), &AllocationCalibration::H264_1080P_QP26);
    let shadow_cap = probed.shadow_max_message_bytes(1);
    eprintln!("DRY caps {caps:?} ceiling {ceiling} B, 1-shadow cap {shadow_cap} B");
    assert!(ceiling > 0 && shadow_cap > 0);

    let mut yuv = Vec::new();
    for (y, u, v) in &frames {
        yuv.extend_from_slice(y);
        yuv.extend_from_slice(u);
        yuv.extend_from_slice(v);
    }
    let shadow = text_with_framed_size(shadow_cap, 0xD7);
    let opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
    // Straddle the boundary the shadow loading moves: with a cap-sized shadow
    // this clip refuses already at 70% of the shown ceiling (first run), so
    // the sizes that make sweep B run AND shrink-and-carry sit below that.
    for pct in [20usize, 35, 45, 55, 62, 70] {
        let primary = text_with_framed_size(ceiling * pct / 100, 0x1D + pct as u64);
        let layers = [ShadowLayer { message: &shadow, passphrase: "dry-shadow", files: &[] }];
        let mut src = SliceYuvSource::new(&yuv, W, H, NF, GOP);
        let out = h264_encode_with_shadows_streaming(
            &mut src, W, H, NF, opts, &primary, &[], "dry-primary", &layers,
            &CostWeights::default(), Some(&caps), None,
        );
        match out {
            Ok(annex_b) => {
                let dec = |pass: &str| {
                    let mut d = StreamingDecodeSession::create(pass).expect("decode session");
                    d.push_annex_b(&annex_b).expect("push");
                    d.finish().ok().map(|r| r.text)
                };
                assert_eq!(dec("dry-primary").as_deref(), Some(primary.as_str()),
                    "{pct}%: encode reported Ok but the primary does not decode");
                assert_eq!(dec("dry-shadow").as_deref(), Some(shadow.as_str()),
                    "{pct}%: encode reported Ok but the shadow does not decode");
                eprintln!("DRY {pct}% of ceiling ({} B framed): Ok, both decode",
                    compressed_payload_size(&primary, &[]));
            }
            Err(e) => {
                let s = format!("{e:?}");
                assert!(s.contains("MessageTooLarge") || s.contains("ShadowEmbedFailed"),
                    "{pct}%: unexpected error {s}");
                eprintln!("DRY {pct}% of ceiling: refused ({s})");
            }
        }
    }
}

/// D19 instrument (prints, asserts nothing): success RATE of N = 3 and N = 4
/// shadows at the per-shadow capacity the probe now reports — the
/// collision-model figure, held to `SHADOW_RELIABILITY_TARGET` (0.95) — over
/// K independent passphrase sets. Before D19 the shown figures (3541 / 2859 B
/// on this clip) were refused from 85% up.
#[test]
#[ignore = "instrument: 16 shadow encodes, ~20 min; run with --ignored --test-threads=1 --nocapture"]
fn d19_instrument_success_rate_at_new_cap() {
    const W: u32 = 320;
    const H: u32 = 192;
    const GOP: u32 = 30;
    const NF: u32 = 60;
    const K: u64 = 8;
    let frames: Vec<_> = (0..NF).map(|f| synth_yuv(W, H, f)).collect();
    let mut probe = StreamingProbeSession::create(EncodeSessionParams {
        width: W, height: H, fps_num: 30, fps_den: 1, qp: 26, gop_size: GOP,
        total_frames_hint: NF, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    })
    .expect("probe session");
    for (y, u, v) in &frames {
        probe
            .push_frame(YuvFrameRef {
                y, y_stride: W as usize, u, u_stride: (W / 2) as usize, v, v_stride: (W / 2) as usize,
            })
            .expect("probe push");
    }
    let probed = probe.finish().expect("probe finish");
    let mut yuv = Vec::new();
    for (y, u, v) in &frames {
        yuv.extend_from_slice(y);
        yuv.extend_from_slice(u);
        yuv.extend_from_slice(v);
    }
    for n in [3u32, 4] {
        let cap = probed.shadow_max_message_bytes(n);
        let mut ok = 0;
        for trial in 0..K {
            let shadows: Vec<(String, String)> = (0..n as u64)
                .map(|k| {
                    (
                        text_with_framed_size(cap, 0x19_0000 + trial * 97 + k),
                        format!("d19-{n}-{trial}-{k}"),
                    )
                })
                .collect();
            let verdict = shadows_round_trip(&yuv, W, H, NF, GOP, "d19 primary", &shadows);
            if matches!(verdict, Ok(true)) {
                ok += 1;
            }
            eprintln!("D19 n={n} cap {cap} B, trial {trial}: {verdict:?}");
        }
        eprintln!("D19 n={n} at the new cap ({cap} B framed): {ok}/{K} embed + decode");
    }
}

