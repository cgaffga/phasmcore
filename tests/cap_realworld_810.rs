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

use phasm_core::{
    h264_shadow_capacity_for_n,
    h264_video_capacity,
    ColorParams, CostWeights, EncodeEngineChoice, EncodeSessionParams,
    StreamingDecodeSession, StreamingEncodeSession, YuvFrameRef,
};
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
    let yuv_path = format!("/tmp/phasm_cap810_{}_{}x{}_f{}.yuv", key, w, h, n);
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

fn encode_bytes(yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, msg: &str, pass: &str) -> Option<Vec<u8>> {
    let params = EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
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
        let stego_bytes = encode_bytes(&yuv, w, h, n, GOP, &msg, pass).expect("oh264 stego");
        assert_eq!(
            decode_text(&stego_bytes, pass).as_deref(),
            Some(msg.as_str()),
            "{file}: stego round-trip",
        );
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
