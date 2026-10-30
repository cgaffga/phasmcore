// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Tier 3 — H.264 allocation-SHAPE stealth sweep (spread vs concentrate).
//!
//! `calibration.rs` marks the spread-vs-concentrate AUC as **Tier 3, still
//! design-default** — i.e. never measured. This binary measures it.
//!
//! ## Why the Tier 2 sweep does not already answer this
//!
//! [`h264_stealth_sweep`](./h264_stealth_sweep.rs) varies *utilization* `u`
//! (how much payload per cover) on a **single GOP** (`N_FRAMES_DEFAULT = 30`)
//! and installs **no allocation plan**. Its result — AUC flat at ~0.51 from
//! u=5% to u=100% — says detectability is rate-insensitive for one GOP taking
//! the default uniform spread.
//!
//! Allocation shape is a different axis and is meaningless at `n_gops = 1`.
//! The shape worry is specifically that concentrating payload into
//! high-capacity GOPs makes *those* GOPs individually hot even when the clip
//! aggregate is unchanged — and a whole-clip feature vector averages exactly
//! that away. So this binary differs on three points:
//!
//! 1. **Multi-GOP** (default 150 frames = 5 production GOPs).
//! 2. **Per-GOP features.** Positions carry `frame_idx`
//!    ([`PositionKey::frame_idx`]), so cover/stego bits are segmented into
//!    `frame_idx / gop_size` buckets and each GOP is its own AUC sample.
//! 3. **Plan-parameterized**, over the three shipped shapes plus the
//!    candidate — all four at the SAME payload, so shape is the only variable.
//!
//! ## Arms
//!
//! | arm | allocator | who ships it |
//! |---|---|---|
//! | `none` | online uniform spread (no plan installed) | both apps' silent fallback on `NeedMoreSamples` |
//! | `balanced` | `plan_safe_balanced` → `uniform_split` | iOS + Android today (#841) |
//! | `concentrate` | `plan_proportional` → `allocate_chunks_concentrate_tail` | the CLI today |
//! | `proportional` | `allocate_chunks_proportional` → `set_gop_alloc_plan` | **the candidate** (cap-clamped, binds on Σ not n×min) |
//!
//! ## The sharp test
//!
//! Whole-clip AUC is reported for comparability with Tier 2, but the
//! decision-relevant number is **`auc_hot`**: the AUC restricted to GOPs in
//! the top quartile of payload share. A concentrating allocator is only
//! dangerous if its hot GOPs are individually detectable. `share_gini` and
//! `share_max` quantify how unequal each arm's split actually came out, so a
//! flat AUC can be read against whether the arm concentrated at all.
//!
//! ## Honesty caveat
//!
//! W6 is a CLASSICAL detector (two 1D features → mean-difference axis →
//! Mann-Whitney U), not a CNN steganalyst — the same caveat the Tier 2 docs
//! carry. A low W6 AUC is a reproducible regression baseline, not proof of
//! undetectability against a strong adversary.
//!
//! ## Usage
//!
//! ```bash
//! h264_alloc_shape_sweep <arm: none|balanced|concentrate|proportional> \
//!     <payload_pct_of_sigma: 1..=100> <out_json> [n_frames]
//! ```

#![cfg(all(feature = "h264-encoder", feature = "h264-decoder"))]

use std::path::PathBuf;
use std::process::Command;

use phasm_core::codec::h264::cabac::bin_decoder::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};
use phasm_core::codec::h264::openh264_stego::{h264_walk_cover, EncodeOpts};
use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, H264PlanOutcome,
    StreamingEncodeSession, StreamingProbeSession, YuvFrameRef,
};
use phasm_core::codec::h264::stego::CostWeights;
use phasm_core::stego::calibration::AllocationCalibration;
use phasm_core::stego::chunk_frame::{allocate_chunks_proportional, CAP2_DEFAULT_R_TARGET};

struct SeekFixture {
    name: &'static str,
    source: &'static str,
    duration_s: f32,
}

/// Same four clips as the Tier 2 sweep, so the two measurements are
/// comparable on content. Spans the cover-yield range.
const FIXTURES: &[SeekFixture] = &[
    SeekFixture { name: "iphone7_1080p", source: "iphone7_1080p_30fps_h264_high.mov", duration_s: 16.0 },
    SeekFixture { name: "carplane",      source: "Artlist_CarPlane.mp4",              duration_s: 6.5 },
    SeekFixture { name: "iphone5_1080p", source: "iphone5_1080p_30fps_h264_high.mov", duration_s: 12.0 },
    SeekFixture { name: "womansubway",   source: "Artlist_WomanSubway.mp4",           duration_s: 12.0 },
];

const SEEK_POINTS_PER_FIXTURE: usize = 6;
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1088;
const N_FRAMES_DEFAULT: u32 = 150; // 5 production GOPs
const GOP: u32 = 30;
const QP: i32 = 26;

type Features = [f64; 2];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Arm { None_, Balanced, Concentrate, Proportional }

impl Arm {
    fn parse(s: &str) -> Option<Arm> {
        match s {
            "none" => Some(Arm::None_),
            "balanced" => Some(Arm::Balanced),
            "concentrate" => Some(Arm::Concentrate),
            "proportional" => Some(Arm::Proportional),
            _ => None,
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Arm::None_ => "none",
            Arm::Balanced => "balanced",
            Arm::Concentrate => "concentrate",
            Arm::Proportional => "proportional",
        }
    }
}

/// One GOP's cover/stego feature pair — the AUC sample unit.
///
/// `share` is the MEASURED flip share, not the planned byte share. The
/// planned vector is a reconstruction of each allocator's intended split
/// (and for the `none` arm the online spread may shrink-carry away from it),
/// whereas flips are read directly off the cover/stego bit pair. Flips are
/// also the physically right quantity: W6 reads sign statistics, so what
/// makes a GOP hot is how many signs moved in it, not how many payload bytes
/// were nominally assigned. `planned_bytes` is kept alongside so plan-vs-
/// reality can be compared.
struct GopSample {
    fixture: &'static str,
    seek_s: f32,
    gop_idx: usize,
    n_cover: usize,
    planned_bytes: usize,
    flips: usize,
    flip_rate: f64,
    share: f64,
    f_cover: Features,
    f_stego: Features,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <arm: none|balanced|concentrate|proportional> \
             <payload_pct_of_sigma: 1..=100> <out_json> [n_frames]",
            args[0]
        );
        std::process::exit(2);
    }
    let Some(arm) = Arm::parse(&args[1]) else {
        eprintln!("unknown arm {:?}", args[1]);
        std::process::exit(2);
    };
    let payload_pct: u32 = args[2].parse().expect("payload_pct must be integer");
    let out_json = &args[3];
    let n_frames: u32 = if args.len() >= 5 {
        args[4].parse().expect("n_frames must be integer")
    } else {
        N_FRAMES_DEFAULT
    };
    if !(1..=100).contains(&payload_pct) {
        eprintln!("payload_pct must be in [1, 100], got {payload_pct}");
        std::process::exit(2);
    }

    eprintln!(
        "[alloc_shape] arm={}  payload={}% of Sigma  {}x{}x{}f gop={} ({} GOPs) qp={}  \
         fixtures={} x seeks={}",
        arm.as_str(), payload_pct, WIDTH, HEIGHT, n_frames, GOP,
        (n_frames + GOP - 1) / GOP, QP, FIXTURES.len(), SEEK_POINTS_PER_FIXTURE,
    );

    let opts = EncodeOpts { qp: QP, intra_period: GOP as i32 };
    let mut samples: Vec<GopSample> = Vec::new();
    let mut skipped_extract = 0u32;
    let mut skipped_cover = 0u32;
    let mut skipped_probe = 0u32;
    let mut refused_plan = 0u32;
    let mut failed_encode = 0u32;
    let mut payload_seed: u64 = 0xC0FFEE;
    let t_start = std::time::Instant::now();

    for fx in FIXTURES {
        let usable_range = (fx.duration_s - (n_frames as f32 / 30.0) - 0.5).max(0.5);
        let step = usable_range / SEEK_POINTS_PER_FIXTURE as f32;
        for i in 0..SEEK_POINTS_PER_FIXTURE {
            let seek_s = 0.3 + step * i as f32;
            let Some(yuv) = extract_yuv_clip(fx.source, seek_s, WIDTH, HEIGHT, n_frames) else {
                skipped_extract += 1;
                continue;
            };

            // ── clean cover, segmented per GOP ──
            let clean = match h264_walk_cover(&yuv, WIDTH, HEIGHT, n_frames, opts) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("  skip {} @ {:.2}s — cover walk: {:?}", fx.name, seek_s, e);
                    skipped_cover += 1;
                    continue;
                }
            };
            let cover_by_gop = segment_by_gop(
                &clean.coeff_sign_bypass.bits,
                &clean.coeff_sign_bypass.positions,
                GOP,
            );

            // ── per-GOP caps (the vector every planner consumes) ──
            let Some(caps) = probe_caps(&yuv, WIDTH, HEIGHT, n_frames, GOP) else {
                skipped_probe += 1;
                continue;
            };
            let sigma: usize = caps.iter().sum();
            if sigma < 64 {
                skipped_probe += 1;
                continue;
            }

            // ── payload sized to payload_pct of Sigma in FRAMED bytes ──
            // Framed, not raw: the payload is brotli-compressed when that
            // shrinks it (`payload.rs::try_compress`), so a raw length would
            // land well under the intended fraction.
            payload_seed = payload_seed.wrapping_mul(0x9E3779B97F4A7C15);
            let pass = format!("alloc-shape-{}-{:x}", payload_pct, payload_seed);
            let target_framed = ((sigma as f64) * payload_pct as f64 / 100.0) as usize;
            let target_framed = target_framed.clamp(16, sigma);
            let raw_len = raw_len_for_framed(target_framed, payload_seed);
            let msg = make_ascii_msg(raw_len, payload_seed);

            // ── encode under this arm's allocation shape ──
            let (stego_bytes, plan) = match encode_arm(
                arm, &yuv, WIDTH, HEIGHT, n_frames, GOP, &msg, &pass, &caps,
            ) {
                EncodeOutcome::Ok { bytes, plan } => (bytes, plan),
                EncodeOutcome::PlanRefused(why) => {
                    eprintln!("  {} @ {:.2}s — plan refused: {}", fx.name, seek_s, why);
                    refused_plan += 1;
                    continue;
                }
                EncodeOutcome::EncodeFailed(why) => {
                    eprintln!("  {} @ {:.2}s — encode failed: {}", fx.name, seek_s, why);
                    failed_encode += 1;
                    continue;
                }
            };

            let stego = match walk_annex_b_for_cover_with_options(
                &stego_bytes,
                WalkOptions { record_mvd: true, ..Default::default() },
            ) {
                Ok(w) => w.cover,
                Err(e) => {
                    eprintln!("  skip {} @ {:.2}s — stego walk: {:?}", fx.name, seek_s, e);
                    failed_encode += 1;
                    continue;
                }
            };
            let stego_by_gop = segment_by_gop(
                &stego.coeff_sign_bypass.bits,
                &stego.coeff_sign_bypass.positions,
                GOP,
            );

            // ── one AUC sample per GOP ──
            // Shares are normalised WITHIN a clip: a GOP is "hot" relative to
            // the other GOPs of its own clip, since that is the comparison an
            // allocator controls. Normalising across the whole corpus would
            // instead rank clips by absolute capacity.
            let first_sample = samples.len();
            let mut gop_flips: Vec<usize> = Vec::new();
            let n_gops = caps.len().min(cover_by_gop.len()).min(stego_by_gop.len());
            for g in 0..n_gops {
                let cb = &cover_by_gop[g];
                let sb = &stego_by_gop[g];
                // Comparability: flips never change the position count, so a
                // GOP whose cover and stego lengths disagree means the walk
                // desynced — drop it rather than compare unlike things.
                if cb.len() < 256 || cb.len() != sb.len() {
                    continue;
                }
                let planned = plan.get(g).copied().unwrap_or(0);
                let flips = cb.iter().zip(sb.iter()).filter(|(a, b)| a != b).count();
                gop_flips.push(flips);
                samples.push(GopSample {
                    fixture: fx.name,
                    seek_s,
                    gop_idx: g,
                    n_cover: cb.len(),
                    planned_bytes: planned,
                    flips,
                    flip_rate: flips as f64 / cb.len() as f64,
                    share: 0.0, // filled below, once the clip's flip total is known
                    f_cover: features(cb),
                    f_stego: features(sb),
                });
            }
            let clip_flips: usize = gop_flips.iter().sum();
            if clip_flips > 0 {
                for sm in &mut samples[first_sample..] {
                    sm.share = sm.flips as f64 / clip_flips as f64;
                }
            }
        }
    }

    let wall_ms = t_start.elapsed().as_millis();
    let n = samples.len();
    if n < 8 {
        eprintln!("[alloc_shape] FATAL: only {n} usable GOP samples (need >= 8)");
        eprintln!(
            "  skipped: extract={skipped_extract} cover={skipped_cover} probe={skipped_probe} \
             plan_refused={refused_plan} encode_failed={failed_encode}"
        );
        std::process::exit(3);
    }

    let auc_all = auc_over(&samples, |_| true);
    // The sharp test: AUC among the most-loaded GOPs only.
    let mut shares: Vec<f64> = samples.iter().map(|s| s.share).collect();
    shares.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q75 = shares[(shares.len() * 3 / 4).min(shares.len() - 1)];
    let auc_hot = auc_over(&samples, |s| s.share >= q75);
    let n_hot = samples.iter().filter(|s| s.share >= q75).count();

    let share_max = shares[shares.len() - 1];
    let share_gini = gini(&shares);
    let mean_share = shares.iter().sum::<f64>() / n as f64;

    let mean_c = mean(&samples.iter().map(|s| s.f_cover).collect::<Vec<_>>());
    let mean_s = mean(&samples.iter().map(|s| s.f_stego).collect::<Vec<_>>());

    eprintln!(
        "\n[alloc_shape] arm={}  payload={}%  n_gop_samples={}  \
         AUC_all={:.4}  AUC_hot={:.4} (n={})",
        arm.as_str(), payload_pct, n, auc_all, auc_hot, n_hot,
    );
    let mean_flip_rate =
        samples.iter().map(|s| s.flip_rate).sum::<f64>() / n as f64;
    let max_flip_rate =
        samples.iter().map(|s| s.flip_rate).fold(0.0f64, f64::max);
    eprintln!(
        "  flip share (MEASURED): mean={:.4} max={:.4} gini={:.4}  \
         (gini 0 = perfectly even across a clip's GOPs, 1 = all in one GOP)",
        mean_share, share_max, share_gini,
    );
    eprintln!(
        "  flip rate per GOP: mean={:.5} max={:.5}  (flips / cover positions)",
        mean_flip_rate, max_flip_rate,
    );
    eprintln!(
        "  features cover=({:.5},{:.5}) stego=({:.5},{:.5}) delta=({:+.5},{:+.5})",
        mean_c[0], mean_c[1], mean_s[0], mean_s[1],
        mean_s[0] - mean_c[0], mean_s[1] - mean_c[1],
    );
    eprintln!(
        "  skipped: extract={skipped_extract} cover={skipped_cover} probe={skipped_probe} \
         plan_refused={refused_plan} encode_failed={failed_encode}  wall={wall_ms}ms"
    );

    write_json(
        out_json, arm, payload_pct, n, n_hot, auc_all, auc_hot, share_max, share_gini,
        mean_share, &mean_c, &mean_s, wall_ms, skipped_extract, skipped_cover,
        skipped_probe, refused_plan, failed_encode, &samples,
    ).expect("write_json");
    eprintln!("[alloc_shape] wrote {out_json}");
}

// ─── arms ──────────────────────────────────────────────────────────────

enum EncodeOutcome {
    Ok { bytes: Vec<u8>, plan: Vec<usize> },
    PlanRefused(String),
    EncodeFailed(String),
}

fn session_params(w: u32, h: u32, n: u32, gop: u32) -> EncodeSessionParams {
    EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: QP, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_arm(
    arm: Arm, yuv: &[u8], w: u32, h: u32, n: u32, gop: u32,
    msg: &str, pass: &str, caps: &[usize],
) -> EncodeOutcome {
    let mut enc = match StreamingEncodeSession::create(session_params(w, h, n, gop), msg, pass) {
        Ok(e) => e,
        Err(e) => return EncodeOutcome::EncodeFailed(format!("session create: {e:?}")),
    };
    let framed = enc.framed_message_len().unwrap_or(0);
    let n_gops = caps.len();

    // Install (or don't) this arm's allocation plan. `plan` is also the
    // per-GOP payload vector the AUC samples are weighted by; the `none` arm
    // has no plan, so it reports the uniform share it will actually take.
    let plan: Vec<usize> = match arm {
        Arm::None_ => {
            let base = framed / n_gops.max(1);
            let rem = framed % n_gops.max(1);
            (0..n_gops).map(|i| base + usize::from(i < rem)).collect()
        }
        Arm::Balanced => {
            let samples: Vec<(usize, usize)> =
                caps.iter().copied().enumerate().collect();
            let cal = AllocationCalibration::H264_1080P_QP26;
            match enc.plan_safe_balanced(&samples, gop as usize, &cal) {
                Ok(H264PlanOutcome::Installed { window }) => {
                    let w = (window as usize).min(n_gops).max(1);
                    let base = framed / w;
                    let rem = framed % w;
                    (0..n_gops)
                        .map(|i| if i < w { base + usize::from(i < rem) } else { 0 })
                        .collect()
                }
                Ok(H264PlanOutcome::MessageTooLarge) =>
                    return EncodeOutcome::PlanRefused("balanced: MessageTooLarge".into()),
                Ok(H264PlanOutcome::NeedMoreSamples { .. }) =>
                    return EncodeOutcome::PlanRefused("balanced: NeedMoreSamples".into()),
                Err(e) => return EncodeOutcome::PlanRefused(format!("balanced: {e:?}")),
            }
        }
        Arm::Concentrate => {
            match enc.plan_proportional(caps, CAP2_DEFAULT_R_TARGET) {
                Ok(true) => {}
                Ok(false) =>
                    return EncodeOutcome::PlanRefused("concentrate: not installed".into()),
                Err(e) => return EncodeOutcome::PlanRefused(format!("concentrate: {e:?}")),
            }
            // Mirror of the installed shape: concentrate_tail fills each GOP
            // sequentially to floor(r*cap), leaving a plain tail.
            concentrate_tail_shape(caps, framed, CAP2_DEFAULT_R_TARGET)
        }
        Arm::Proportional => {
            let Some(p) = allocate_chunks_proportional(caps, framed, CAP2_DEFAULT_R_TARGET) else {
                return EncodeOutcome::PlanRefused(
                    format!("proportional: framed {framed} > Sigma {}", caps.iter().sum::<usize>()),
                );
            };
            if let Err(e) = enc.set_gop_alloc_plan(p.clone()) {
                return EncodeOutcome::PlanRefused(format!("proportional: {e:?}"));
            }
            p
        }
    };

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
        if let Err(e) = enc.push_frame(frame, &mut out) {
            return EncodeOutcome::EncodeFailed(format!("push_frame #{f}: {e:?}"));
        }
    }
    if let Err(e) = enc.finish(&mut out) {
        return EncodeOutcome::EncodeFailed(format!("finish: {e:?}"));
    }
    EncodeOutcome::Ok { bytes: out, plan }
}

/// Reproduce `allocate_chunks_concentrate_tail`'s shape for reporting.
/// Sequential fill to `floor(r*cap)`, widening to full cap once the message
/// exceeds the soft ceiling (the `use_full` branch).
fn concentrate_tail_shape(caps: &[usize], framed: usize, r: f64) -> Vec<usize> {
    let soft: usize = caps.iter().map(|&c| (r * c as f64) as usize).sum();
    let use_full = framed > soft;
    let mut plan = vec![0usize; caps.len()];
    let mut left = framed;
    for (i, &c) in caps.iter().enumerate() {
        if left == 0 { break; }
        let room = if use_full { c } else { (r * c as f64) as usize };
        let take = room.min(left);
        plan[i] = take;
        left -= take;
    }
    plan
}

// ─── per-GOP segmentation ──────────────────────────────────────────────

/// Split a domain's flat cover bits into per-GOP buckets using each
/// position's `frame_idx`. `PositionKey` packs frame_idx in its low 24 bits,
/// so `gop = frame_idx / gop_size` recovers the bucket exactly.
fn segment_by_gop(
    bits: &[u8],
    positions: &[phasm_core::codec::h264::stego::hook::PositionKey],
    gop_size: u32,
) -> Vec<Vec<u8>> {
    let n = bits.len().min(positions.len());
    let mut max_gop = 0usize;
    for p in &positions[..n] {
        max_gop = max_gop.max((p.frame_idx() / gop_size) as usize);
    }
    let mut out = vec![Vec::new(); max_gop + 1];
    for i in 0..n {
        let g = (positions[i].frame_idx() / gop_size) as usize;
        out[g].push(bits[i]);
    }
    out
}

// ─── payload sizing (framed, not raw) ──────────────────────────────────

/// Framed length of a `make_ascii_msg(raw)` payload, computed in closed form.
///
/// **Prefer this over `StreamingEncodeSession::create`.** That path runs
/// `crypto::encrypt` *and* `CabacStegoMasterKeys::derive` — two Argon2id
/// derivations, ~38 ms per create as measured by
/// `alloc_shape_framing_matches_session`. A ~13-probe bisect per sample is
/// then ~0.5 s of pure KDF: real waste in a hot loop, though not this
/// binary's dominant cost (the cover walk, probe and encode passes are).
/// The stronger reason is exactness — this is the framing, not a model of it.
///
/// The frame layout (`stego::frame::build_frame`) is
/// `[len 2|6] [salt 16] [nonce 12] [ciphertext P+16] [crc32 4]`, so the
/// framed length is a pure function of the compressed payload size `P` and
/// carries no key material:
///
/// ```text
///     framed = P + 50        (P <= u16::MAX)
///     framed = P + 54        (v2 header, P > u16::MAX)
/// ```
///
/// `compressed_payload_size` is the same brotli path the session takes
/// (`payload::encode_payload`), so this is exact, not an estimate — the
/// `alloc_shape_framing_matches_session` test pins it against a real session.
fn framed_len_of_raw(raw: usize, seed: u64) -> usize {
    let msg = make_ascii_msg(raw, seed);
    let p = phasm_core::stego::compressed_payload_size(&msg, &[]);
    p + if p > u16::MAX as usize { 54 } else { 50 }
}

/// Bisect the raw length whose FRAMED length is closest to `target_framed`
/// without exceeding it. Needed because `make_ascii_msg` emits printable
/// ASCII over 94 symbols (~6.55 bits/char), which brotli shrinks ~17% — so
/// using the target directly as a raw length understates every payload.
fn raw_len_for_framed(target_framed: usize, seed: u64) -> usize {
    let probe = |raw: usize| framed_len_of_raw(raw, seed);
    if probe(target_framed) >= target_framed {
        // Already at or over — shrink.
        let (mut lo, mut hi) = (16usize, target_framed.max(16));
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if probe(mid) <= target_framed { lo = mid; } else { hi = mid - 1; }
        }
        return lo.max(16);
    }
    // Framing compressed it — grow until it overshoots, then bisect.
    let mut hi = target_framed.max(16);
    for _ in 0..8 {
        let next = hi.saturating_mul(2);
        if probe(next) > target_framed { hi = next; break; }
        hi = next;
    }
    let mut lo = target_framed.max(16);
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if probe(mid) <= target_framed { lo = mid; } else { hi = mid - 1; }
    }
    lo.max(16)
}

fn make_ascii_msg(len: usize, seed: u64) -> String {
    let mut s = String::with_capacity(len);
    for k in 0..len {
        let x = seed.wrapping_add(k as u64).wrapping_mul(0xCAFEBABE);
        s.push((33 + (x >> 24) as u8 % 94) as char);
    }
    s
}

// ─── probe ─────────────────────────────────────────────────────────────

fn probe_caps(yuv: &[u8], w: u32, h: u32, n: u32, gop: u32) -> Option<Vec<usize>> {
    let mut probe = StreamingProbeSession::create(session_params(w, h, n, gop)).ok()?;
    let fb = (w * h * 3 / 2) as usize;
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let cw = (w / 2) as usize;
    for f in 0..n as usize {
        let off = f * fb;
        probe.push_frame(YuvFrameRef {
            y: &yuv[off..off + ys], y_stride: w as usize,
            u: &yuv[off + ys..off + ys + cs], u_stride: cw,
            v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
        }).ok()?;
    }
    let (_res, caps) = probe.finish_with_per_gop().ok()?;
    Some(caps)
}

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv_clip(source: &str, seek_s: f32, w: u32, h: u32, n: u32) -> Option<Vec<u8>> {
    let src = corpus_root().join(source);
    if !src.exists() {
        eprintln!("  corpus fixture missing: {}", src.display());
        return None;
    }
    let vf = format!("scale={w}:{h}:force_original_aspect_ratio=disable");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"]).arg(&src)
        .args(["-frames:v", &n.to_string(), "-vf", &vf,
               "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output().ok()?;
    if !out.status.success() {
        return None;
    }
    let expected = (w * h * 3 / 2) as usize * n as usize;
    if out.stdout.len() != expected {
        return None; // short read near EOF
    }
    Some(out.stdout)
}

// ─── W6 features + AUC (mirrors h264_stealth_sweep) ────────────────────

fn features(bits: &[u8]) -> Features {
    let n = bits.len();
    if n < 2 {
        return [0.5, 0.5];
    }
    let positive = bits.iter().filter(|&&b| b == 1).count();
    let positive_ratio = positive as f64 / n as f64;
    let mut agree = 0usize;
    for i in 1..n {
        if bits[i] == bits[i - 1] { agree += 1; }
    }
    [positive_ratio, agree as f64 / (n - 1) as f64]
}

fn mean(samples: &[Features]) -> Features {
    let n = samples.len().max(1) as f64;
    let mut m = [0.0; 2];
    for f in samples { m[0] += f[0]; m[1] += f[1]; }
    [m[0] / n, m[1] / n]
}

/// Mean-difference-axis projection + Mann-Whitney U AUC, over the subset of
/// samples passing `keep`. The axis is fitted on that same subset, matching
/// the Tier 2 estimator (an optimistic-for-the-detector choice, so a low AUC
/// is the conservative direction).
fn auc_over(samples: &[GopSample], keep: impl Fn(&GopSample) -> bool) -> f64 {
    let sel: Vec<&GopSample> = samples.iter().filter(|s| keep(s)).collect();
    if sel.len() < 4 {
        return f64::NAN;
    }
    let mc = mean(&sel.iter().map(|s| s.f_cover).collect::<Vec<_>>());
    let ms = mean(&sel.iter().map(|s| s.f_stego).collect::<Vec<_>>());
    let axis = [ms[0] - mc[0], ms[1] - mc[1]];
    let project = |f: &Features| f[0] * axis[0] + f[1] * axis[1];
    let cover: Vec<f64> = sel.iter().map(|s| project(&s.f_cover)).collect();
    let stego: Vec<f64> = sel.iter().map(|s| project(&s.f_stego)).collect();
    auc(&cover, &stego)
}

fn auc(cover: &[f64], stego: &[f64]) -> f64 {
    let mut higher = 0u64;
    let mut tied = 0u64;
    for &s in stego {
        for &c in cover {
            if s > c { higher += 1; } else if s == c { tied += 1; }
        }
    }
    let total = (cover.len() * stego.len()) as f64;
    if total == 0.0 { return f64::NAN; }
    let raw = (higher as f64 + 0.5 * tied as f64) / total;
    // Detector may point either way; report the informative side.
    if raw < 0.5 { 1.0 - raw } else { raw }
}

/// Gini coefficient of the per-GOP payload shares — 0 = perfectly even
/// split, 1 = everything in one GOP. Quantifies how much an arm actually
/// concentrated, so a flat AUC can be read against whether there was any
/// concentration to detect.
fn gini(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 { return 0.0; }
    let sum: f64 = sorted.iter().sum();
    if sum <= 0.0 { return 0.0; }
    let mut acc = 0.0;
    for (i, v) in sorted.iter().enumerate() {
        acc += (2.0 * (i as f64 + 1.0) - n as f64 - 1.0) * v;
    }
    acc / (n as f64 * sum)
}

// ─── JSON ──────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn write_json(
    path: &str, arm: Arm, payload_pct: u32, n: usize, n_hot: usize,
    auc_all: f64, auc_hot: f64, share_max: f64, share_gini: f64, mean_share: f64,
    mean_c: &Features, mean_s: &Features, wall_ms: u128,
    skipped_extract: u32, skipped_cover: u32, skipped_probe: u32,
    refused_plan: u32, failed_encode: u32, samples: &[GopSample],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "{{")?;
    writeln!(f, "  \"measurement\": \"h264_alloc_shape_sweep\",")?;
    writeln!(f, "  \"tier\": 3,")?;
    writeln!(f, "  \"arm\": \"{}\",", arm.as_str())?;
    writeln!(f, "  \"payload_pct_of_sigma\": {payload_pct},")?;
    writeln!(f, "  \"n_gop_samples\": {n},")?;
    writeln!(f, "  \"n_hot_samples\": {n_hot},")?;
    writeln!(f, "  \"auc_all\": {auc_all:.6},")?;
    writeln!(f, "  \"auc_hot\": {auc_hot:.6},")?;
    writeln!(f, "  \"share_mean\": {mean_share:.6},")?;
    writeln!(f, "  \"share_max\": {share_max:.6},")?;
    writeln!(f, "  \"share_gini\": {share_gini:.6},")?;
    writeln!(f, "  \"share_metric\": \"measured_flips_normalised_per_clip\",")?;
    writeln!(f, "  \"mean_cover\": [{:.8}, {:.8}],", mean_c[0], mean_c[1])?;
    writeln!(f, "  \"mean_stego\": [{:.8}, {:.8}],", mean_s[0], mean_s[1])?;
    writeln!(f, "  \"wall_ms\": {wall_ms},")?;
    writeln!(f, "  \"skipped\": {{\"extract\": {skipped_extract}, \"cover\": {skipped_cover}, \
        \"probe\": {skipped_probe}, \"plan_refused\": {refused_plan}, \
        \"encode_failed\": {failed_encode}}},")?;
    writeln!(f, "  \"samples\": [")?;
    for (i, s) in samples.iter().enumerate() {
        writeln!(
            f,
            "    {{\"fixture\": \"{}\", \"seek_s\": {:.3}, \"gop\": {}, \"n_cover\": {}, \
             \"planned_bytes\": {}, \"flips\": {}, \"flip_rate\": {:.8}, \
             \"share\": {:.6}, \
             \"f_cover\": [{:.8}, {:.8}], \"f_stego\": [{:.8}, {:.8}]}}{}",
            s.fixture, s.seek_s, s.gop_idx, s.n_cover, s.planned_bytes,
            s.flips, s.flip_rate, s.share,
            s.f_cover[0], s.f_cover[1], s.f_stego[0], s.f_stego[1],
            if i + 1 == samples.len() { "" } else { "," },
        )?;
    }
    writeln!(f, "  ]")?;
    writeln!(f, "}}")?;
    Ok(())
}
