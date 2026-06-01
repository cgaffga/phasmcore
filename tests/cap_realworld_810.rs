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
//   cargo test --test cap_realworld_810 --features openh264-backend,cabac-stego \
//     --release --no-run     # build once
//   ./cap_realworld_810.sh   # fan out
//
// Env: PHASM_CAP_SRC (filename), PHASM_CAP_TAG, PHASM_CAP_W, PHASM_CAP_H,
//      PHASM_CAP_N, PHASM_CAP_GOP.

#![cfg(all(feature = "openh264-backend", feature = "cabac-stego"))]

use phasm_core::{
    h264_stego_capacity_4domain_oh264,
    ColorParams, CostWeights, EncodeEngineChoice, EncodeSessionParams,
    StreamingDecodeSession, StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::h264::openh264_stego::EncodeOpts;

fn env_u32(k: &str) -> u32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
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
    let reported = h264_stego_capacity_4domain_oh264(
        &yuv, w, h, n as usize, EncodeOpts { qp: 26, intra_period: gop as i32 },
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
    let reported = h264_stego_capacity_4domain_oh264(
        &yuv, W, H, N as usize, EncodeOpts { qp: 26, intra_period: GOP as i32 },
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
///     --features openh264-backend,cabac-stego --release \
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
///     --features openh264-backend,cabac-stego --release \
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
///   --features openh264-backend,cabac-stego --release \
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
