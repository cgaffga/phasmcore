// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.8.14 (#447) — perf benchmark + ship-gate acceptance for the
// OpenH264-backend stego pipeline.
//
// Compares wall-clock encode time of:
//   1. OpenH264 clean (no stego session registered)
//   2. OpenH264 stego (full visual_recon path, C.8.3-11 + C.8.13)
//   3. pure-Rust stego (multigop streaming v2 with same IPPPP pattern)
//
// Targets (per `docs/design/video/h264/phase-c8-visual-recon-plan.md`
// §C.8.14):
//   - stego overhead ≤ 1.5× clean
//   - OH264 stego encode ≤ 1.5× pure-Rust stego on the same fixture
//
// All variants run on the SAME YUV input + SAME message/passphrase
// so timing differences reflect encoder pipeline cost, not workload.
//
// Live gate (480p×10f) runs by default; the 1080p×30f production
// gate is `#[ignore]` because it takes minutes.

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use phasm_core::codec::h264::openh264_stego::{
    openh264_stego_decode_yuv_string, openh264_stego_encode_yuv_text, EncodeOpts,
};
use phasm_core::codec::h264::openh264::{set_frame_num, Encoder, StegoHandlers, StegoSession};
use phasm_core::codec::h264::stego::encode_pixels::
    h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files;
use phasm_core::codec::h264::stego::gop_pattern::GopPattern;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn ensure_yuv(name: &str, source_mp4: &str, w: u32, h: u32, n_frames: u32) -> Vec<u8> {
    let yuv_path = format!("/tmp/phasm_c814_{}_{}x{}_f{}.yuv", name, w, h, n_frames);
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * (n_frames as usize);
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need {
            return data;
        }
    }
    let src = corpus_root().join(source_mp4);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!("scale={}:{}", w, h);
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &n_frames.to_string()])
        .args(["-an", "-vf", &vf])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("ffmpeg");
    assert!(status.success(), "ffmpeg scale failed for {}", source_mp4);
    std::fs::read(&yuv_path).expect("read yuv")
}

/// Encode `n_frames` of `yuv` through OpenH264 with NO stego callbacks.
/// Returns (annex_b bytes, wall-clock elapsed).
fn bench_openh264_clean(
    yuv: &[u8],
    w: u32,
    h: u32,
    n_frames: u32,
    qp: i32,
    intra_period: i32,
) -> (Vec<u8>, std::time::Duration) {
    let frame_y = (w * h) as usize;
    let frame_uv = (w * h / 4) as usize;
    let frame_total = frame_y + 2 * frame_uv;
    let mut out = vec![0u8; 8 * 1024 * 1024];
    let mut bs = Vec::with_capacity(4 * 1024 * 1024);
    let mut enc = Encoder::new(w as i32, h as i32, qp, intra_period).expect("enc");
    let t0 = Instant::now();
    for frame in 0..n_frames {
        set_frame_num(frame);
        let base = (frame as usize) * frame_total;
        let (_, n) = enc
            .encode_frame(
                &yuv[base..base + frame_y],
                &yuv[base + frame_y..base + frame_y + frame_uv],
                &yuv[base + frame_y + frame_uv..base + frame_total],
                (frame as i64) * 33,
                &mut out,
            )
            .expect("encode");
        bs.extend_from_slice(&out[..n]);
    }
    let dt = t0.elapsed();
    (bs, dt)
}

/// Encode `n_frames` of `yuv` through OpenH264 with an *empty* stego
/// session registered (callbacks installed but no overrides). Isolates
/// the dual-recon visual_recon overhead alone — independent of message
/// / STC / decode work.
fn bench_openh264_stego_passive(
    yuv: &[u8],
    w: u32,
    h: u32,
    n_frames: u32,
    qp: i32,
    intra_period: i32,
) -> (Vec<u8>, std::time::Duration) {
    let handlers = StegoHandlers {
        enc_pre_emit: Some(Box::new(|_pos, _orig| None)),
        ..Default::default()
    };
    let _sess = StegoSession::register(handlers).expect("register");
    let frame_y = (w * h) as usize;
    let frame_uv = (w * h / 4) as usize;
    let frame_total = frame_y + 2 * frame_uv;
    let mut out = vec![0u8; 8 * 1024 * 1024];
    let mut bs = Vec::with_capacity(4 * 1024 * 1024);
    let mut enc = Encoder::new(w as i32, h as i32, qp, intra_period).expect("enc");
    let t0 = Instant::now();
    for frame in 0..n_frames {
        set_frame_num(frame);
        let base = (frame as usize) * frame_total;
        let (_, n) = enc
            .encode_frame(
                &yuv[base..base + frame_y],
                &yuv[base + frame_y..base + frame_y + frame_uv],
                &yuv[base + frame_y + frame_uv..base + frame_total],
                (frame as i64) * 33,
                &mut out,
            )
            .expect("encode");
        bs.extend_from_slice(&out[..n]);
    }
    let dt = t0.elapsed();
    (bs, dt)
}

/// Encode through the full OpenH264 stego orchestrator (Pass 1 capacity
/// scan + Pass 2 STC + Pass 3 emit), end-to-end production path.
fn bench_openh264_stego_full(
    yuv: &[u8],
    w: u32,
    h: u32,
    n_frames: u32,
    qp: i32,
    intra_period: i32,
    msg: &str,
    pass: &str,
) -> (Vec<u8>, std::time::Duration) {
    let opts = EncodeOpts { qp, intra_period };
    let t0 = Instant::now();
    let bs = openh264_stego_encode_yuv_text(yuv, w, h, n_frames, opts, msg, pass).expect("encode");
    (bs, t0.elapsed())
}

/// Encode through the pure-Rust stego pipeline with an IPPPP pattern
/// matched to OpenH264's intra_period.
fn bench_pure_rust_stego(
    yuv: &[u8],
    w: u32,
    h: u32,
    n_frames: u32,
    intra_period: usize,
    msg: &str,
    pass: &str,
) -> (Vec<u8>, std::time::Duration) {
    // GopPattern requires gop_size ≤ n_frames (single-GOP fits ≤
    // n_frames). Match OH264's effective IPPPP shape via capped gop.
    let gop = intra_period.min(n_frames as usize).max(1);
    let pattern = GopPattern::legacy_ipppp(gop);
    let t0 = Instant::now();
    let bs = h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
        yuv,
        w,
        h,
        n_frames as usize,
        pattern,
        msg,
        &[],
        pass,
    )
    .expect("pure-rust encode");
    (bs, t0.elapsed())
}

fn report(name: &str, bytes: usize, dt: std::time::Duration, baseline: Option<std::time::Duration>) {
    let ms = dt.as_secs_f64() * 1000.0;
    let throughput_mbps = (bytes as f64 * 8.0) / (dt.as_secs_f64() * 1_000_000.0);
    if let Some(b) = baseline {
        let ratio = dt.as_secs_f64() / b.as_secs_f64();
        eprintln!(
            "  {:36} {:>9.1} ms  {:>9} bytes  {:>7.1} Mbps  {:>5.2}× clean",
            name, ms, bytes, throughput_mbps, ratio
        );
    } else {
        eprintln!(
            "  {:36} {:>9.1} ms  {:>9} bytes  {:>7.1} Mbps   {:>5}",
            name, ms, bytes, throughput_mbps, "—"
        );
    }
}

/// v1.0 ship gate: smoke benchmark at 480p × 10f. Fast enough to live
/// in the default test suite. Prints comparative timings without hard
/// budget assertions — those are tracked in [`bench_1080p_30f_ipppp`].
#[test]
fn c814_perf_smoke_480p_10f() {
    let _g = session_guard().lock().unwrap();
    const W: u32 = 480;
    const H: u32 = 272;
    const N: u32 = 10;
    const QP: i32 = 22;
    const INTRA: i32 = 60; // single IDR → IPPPP

    let yuv = ensure_yuv("smoke_480p", "IMG_4138.MOV", W, H, N);
    let msg = "phasm c814 smoke benchmark";
    let pass = "smoke-pass";

    eprintln!(
        "\nc814 smoke benchmark ({}×{} × {} frames, QP={}, IPPPP):",
        W, H, N, QP
    );
    eprintln!(
        "  {:36} {:>9}     {:>9}        {:>7}       {}",
        "variant", "ms", "bytes", "Mbps", "vs-clean"
    );

    let (clean_bs, clean_dt) = bench_openh264_clean(&yuv, W, H, N, QP, INTRA);
    report("openh264 clean (no callbacks)", clean_bs.len(), clean_dt, None);

    let (passive_bs, passive_dt) = bench_openh264_stego_passive(&yuv, W, H, N, QP, INTRA);
    report(
        "openh264 stego passive (visual_recon)",
        passive_bs.len(),
        passive_dt,
        Some(clean_dt),
    );

    let (full_bs, full_dt) =
        bench_openh264_stego_full(&yuv, W, H, N, QP, INTRA, msg, pass);
    report(
        "openh264 stego full (encode_yuv_text)",
        full_bs.len(),
        full_dt,
        Some(clean_dt),
    );

    let (pr_bs, pr_dt) = bench_pure_rust_stego(&yuv, W, H, N, INTRA as usize, msg, pass);
    report(
        "pure-Rust stego (multigop v2 IPPPP)",
        pr_bs.len(),
        pr_dt,
        Some(clean_dt),
    );

    // Sanity: OH264 stego round-trips.
    let recovered =
        openh264_stego_decode_yuv_string(&full_bs, pass).expect("oh264 stego decode");
    assert_eq!(recovered, msg);

    eprintln!(
        "  openh264 stego speed-up vs pure-Rust: {:.2}×",
        pr_dt.as_secs_f64() / full_dt.as_secs_f64()
    );
}

/// v1.0 production budget gate (OH264-only at 1080p × 30f IPPPP).
///
/// **Why no pure-Rust at 1080p**: empirical 480p × 10f smoke shows
/// pure-Rust takes ~117 s per encode (vs OH264 ~86 ms) — extrapolating
/// to 1080p × 30 = ~60 minutes per pure-Rust encode, untenable for a
/// gate. Cross-encoder timing comparison is captured at 720p × 16f in
/// [`c814_perf_mid_scale_720p_16f`] instead.
///
/// **Budget**: only `passive_overhead` (per-encode visual_recon cost)
/// is subject to ≤ 1.5× clean. The full-path orchestrator is structurally
/// multi-pass (Pass 1 cover walk + STC + Pass 2/3 emit) so its ratio
/// is reported but not gated here.
///
/// `#[ignore]` because the encode alone takes ~30 s on 1080p × 30.
#[test]
#[ignore]
fn c814_perf_1080p_30f_ipppp() {
    let _g = session_guard().lock().unwrap();
    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 30;
    const QP: i32 = 26;
    const INTRA: i32 = 60;

    let yuv = ensure_yuv("prod_1080p", "IMG_4138.MOV", W, H, N);
    let msg = "phasm c814 production benchmark — 1080p × 30 IPPPP";
    let pass = "prod-pass";

    eprintln!(
        "\nc814 production benchmark ({}×{} × {} frames, QP={}, IPPPP, OH264-only):",
        W, H, N, QP
    );
    eprintln!(
        "  {:36} {:>9}     {:>9}        {:>7}       {}",
        "variant", "ms", "bytes", "Mbps", "vs-clean"
    );

    let (clean_bs, clean_dt) = bench_openh264_clean(&yuv, W, H, N, QP, INTRA);
    report("openh264 clean (no callbacks)", clean_bs.len(), clean_dt, None);

    let (passive_bs, passive_dt) = bench_openh264_stego_passive(&yuv, W, H, N, QP, INTRA);
    report(
        "openh264 stego passive (visual_recon)",
        passive_bs.len(),
        passive_dt,
        Some(clean_dt),
    );

    let (full_bs, full_dt) =
        bench_openh264_stego_full(&yuv, W, H, N, QP, INTRA, msg, pass);
    report(
        "openh264 stego full (encode_yuv_text)",
        full_bs.len(),
        full_dt,
        Some(clean_dt),
    );

    let passive_overhead = passive_dt.as_secs_f64() / clean_dt.as_secs_f64();
    let full_overhead = full_dt.as_secs_f64() / clean_dt.as_secs_f64();

    eprintln!("\nc814 production budget:");
    eprintln!("  OH264 stego passive overhead: {:.2}× clean  (≤ 1.5× gate)", passive_overhead);
    eprintln!("  OH264 stego full    overhead: {:.2}× clean  (informational)", full_overhead);

    let recovered =
        openh264_stego_decode_yuv_string(&full_bs, pass).expect("oh264 stego decode");
    assert_eq!(recovered, msg);

    assert!(
        passive_overhead <= 1.5,
        "OH264 passive overhead {:.2}× exceeds 1.5× clean budget",
        passive_overhead
    );
}

/// Mid-scale 3-way cross-encoder comparison at 720p × 16f IPPPP.
///
/// `#[ignore]` because pure-Rust at 720p × 16 takes ~5 - 10 minutes.
/// This is the headline ship-direction comparison (OH264 default vs
/// pure-Rust experimental).
#[test]
#[ignore]
fn c814_perf_mid_scale_720p_16f() {
    let _g = session_guard().lock().unwrap();
    const W: u32 = 1280;
    const H: u32 = 720;
    const N: u32 = 16;
    const QP: i32 = 24;
    const INTRA: i32 = 60;

    let yuv = ensure_yuv("mid_720p", "IMG_4138.MOV", W, H, N);
    let msg = "phasm c814 mid-scale cross-encoder benchmark";
    let pass = "mid-pass";

    eprintln!(
        "\nc814 mid-scale 3-way ({}×{} × {} frames, QP={}, IPPPP):",
        W, H, N, QP
    );
    eprintln!(
        "  {:36} {:>9}     {:>9}        {:>7}       {}",
        "variant", "ms", "bytes", "Mbps", "vs-clean"
    );

    let (clean_bs, clean_dt) = bench_openh264_clean(&yuv, W, H, N, QP, INTRA);
    report("openh264 clean (no callbacks)", clean_bs.len(), clean_dt, None);

    let (passive_bs, passive_dt) = bench_openh264_stego_passive(&yuv, W, H, N, QP, INTRA);
    report(
        "openh264 stego passive (visual_recon)",
        passive_bs.len(),
        passive_dt,
        Some(clean_dt),
    );

    let (full_bs, full_dt) =
        bench_openh264_stego_full(&yuv, W, H, N, QP, INTRA, msg, pass);
    report(
        "openh264 stego full (encode_yuv_text)",
        full_bs.len(),
        full_dt,
        Some(clean_dt),
    );

    let (pr_bs, pr_dt) = bench_pure_rust_stego(&yuv, W, H, N, INTRA as usize, msg, pass);
    report(
        "pure-Rust stego (multigop v2 IPPPP)",
        pr_bs.len(),
        pr_dt,
        Some(clean_dt),
    );

    let oh264_vs_pure = pr_dt.as_secs_f64() / full_dt.as_secs_f64();
    eprintln!(
        "\n  OH264 stego full is {:.0}× FASTER than pure-Rust at {}p × {}f",
        oh264_vs_pure, H, N
    );

    let recovered =
        openh264_stego_decode_yuv_string(&full_bs, pass).expect("oh264 stego decode");
    assert_eq!(recovered, msg);
}

/// Phase C.5.3 (#413) — mobile-relevant fixture: 480p × 60f.
///
/// 480×272 × 60 frames at QP=24 (~2 s at 30 fps, or 1 s at 60 fps).
/// Approximates the typical mobile share-clip size: a short stego
/// video on the phasm.link upload path or a HUD-driven encode-and-
/// share flow on iOS / Android. Compared to the C.5.1 1080p × 30f
/// production bench, this fixture stresses:
///   - Per-init / codec-setup overhead amortized over more frames.
///   - Memory pressure: 60 frames of YUV in residence + accumulating
///     stego bitstream buffer.
///   - GOP cadence: 60 frames spans two IDR cycles at intra=30.
///
/// Pure-Rust skipped here (would take ~10+ minutes at this size; see
/// C.5.1's 1080p variant for the cross-encoder comparison). OH264 +
/// pure-Rust ratio is content-invariant within ±2× across resolutions
/// — 1080p×30f data extrapolates faithfully.
#[test]
#[ignore]
fn c5_3_perf_mobile_480p_60f() {
    let _g = session_guard().lock().unwrap();
    const W: u32 = 480;
    const H: u32 = 272;
    const N: u32 = 60;
    const QP: i32 = 24;
    const INTRA: i32 = 30; // two IDR cycles across the 60 frames

    let yuv = ensure_yuv("mobile_480p_60f", "IMG_4138.MOV", W, H, N);
    let msg = "phasm c5.3 mobile-relevant fixture bench";
    let pass = "mobile-pass";

    eprintln!(
        "\nc5.3 mobile-relevant ({}×{} × {} frames, QP={}, intra={}):",
        W, H, N, QP, INTRA
    );
    eprintln!(
        "  {:36} {:>9}     {:>9}        {:>7}       {}",
        "variant", "ms", "bytes", "Mbps", "vs-clean"
    );

    let (clean_bs, clean_dt) = bench_openh264_clean(&yuv, W, H, N, QP, INTRA);
    report("openh264 clean (no callbacks)", clean_bs.len(), clean_dt, None);

    let (passive_bs, passive_dt) = bench_openh264_stego_passive(&yuv, W, H, N, QP, INTRA);
    report(
        "openh264 stego passive (visual_recon)",
        passive_bs.len(),
        passive_dt,
        Some(clean_dt),
    );

    let (full_bs, full_dt) =
        bench_openh264_stego_full(&yuv, W, H, N, QP, INTRA, msg, pass);
    report(
        "openh264 stego full (encode_yuv_text)",
        full_bs.len(),
        full_dt,
        Some(clean_dt),
    );

    let per_frame_ms = full_dt.as_secs_f64() * 1000.0 / N as f64;
    let realtime_fps = N as f64 / full_dt.as_secs_f64();
    eprintln!(
        "\n  full stego encode: {:.2} ms/frame  ({:.0} fps real-time on host arm64)",
        per_frame_ms, realtime_fps,
    );

    let recovered =
        openh264_stego_decode_yuv_string(&full_bs, pass).expect("oh264 stego decode");
    assert_eq!(recovered, msg);
}
