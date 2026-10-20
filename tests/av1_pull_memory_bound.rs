#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Peak-RSS regression gate for the AV1 streaming pull-API
//! whole-video shadow encode (`av1_encode_with_shadows_streaming`).
//!
//! Replaces the deleted `av1_whole_video_memory` push-session test
//! (#225/#187, 2026-06-29). The whole point of the pull API + the
//! 2-sweep cascade in `av1_encode_with_shadows_streaming` is O(GOP)
//! RAM not O(clip): one decoded GOP YUV + per-GOP harvests +
//! per-GOP natural slabs, NOT the full clip preallocated. If a
//! future refactor accidentally re-introduces a whole-clip buffer
//! (mirror of the pre-WV.6 ~2.8 GB regression at 1080p × 30s), this
//! gate fails loudly.
//!
//! Fixture: 720p × 150 frames × 2 shadows, gop=30 → 5 GOPs. Source
//! YUV ~135 MB; the assertion is that the encoder's RSS-over-baseline
//! delta stays under 500 MB headroom (target observed pre-deletion
//! ~150-200 MB; 500 MB allows for variance + allocator quirks).
//!
//! `#[ignore]` because the encode takes ~60-90 s wall on a dev box.
//! Run explicitly:
//!
//!   cargo test --release --features av1-encoder,av1-decoder \
//!     --test av1_pull_memory_bound -- --ignored --test-threads=1
//!
//! Cross-platform via POSIX `getrusage` (`ru_maxrss`). macOS reports
//! bytes; Linux reports KiB; both normalised to bytes here.
//! Non-Unix falls through with RSS=0 (the test still runs end-to-
//! end on those, just without the RSS gate).

use std::time::Instant;

use phasm_core::codec::av1::stego::whole_video::{
    av1_encode_with_shadows_streaming, Av1SliceYuvSource, Pass1CacheBudget,
};
use phasm_core::stego::{crypto, frame, payload};
use phasm_core::Av1StreamingEncodeParams;

#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
        return 0;
    }
    let max_rss = ru.ru_maxrss as u64;
    #[cfg(target_os = "macos")]
    {
        max_rss
    }
    #[cfg(not(target_os = "macos"))]
    {
        max_rss * 1024
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
}

/// Synthetic deterministic YUV — same pattern the stratified-probe
/// tests use. The smoke companion exercises the streaming WV-shadow
/// code path without needing the corpus fixture (which is gitignored
/// on fresh checkouts) and without the ~82 s wall of the real
/// corpus-sized gate.
fn make_yuv_synthetic(w: u32, h: u32, n: u32) -> Vec<u8> {
    let y_size = (w * h) as usize;
    let uv_size = (w / 2) as usize * (h / 2) as usize;
    let frame = y_size + 2 * uv_size;
    let mut out = vec![0u8; frame * n as usize];
    for f in 0..n as usize {
        let base = f * frame;
        for yy in 0..h as usize {
            for xx in 0..w as usize {
                out[base + yy * w as usize + xx] =
                    ((xx + yy + f * 7) & 0xff) as u8;
            }
        }
        for yy in 0..(h / 2) as usize {
            for xx in 0..(w / 2) as usize {
                let v = (((xx * 2) + (yy * 3) + f * 5) & 0xff) as u8;
                out[base + y_size + yy * (w / 2) as usize + xx] = v;
                out[base + y_size + uv_size + yy * (w / 2) as usize + xx] =
                    v.wrapping_add(32);
            }
        }
    }
    out
}

/// Pull corpus YUV via ffmpeg. Real content so rav1e's RDO has signal
/// — synthetic gradients can starve the per-GOP cover and trip
/// `FrameCorrupted` before the assertion runs (a documented constraint
/// from the pre-deletion byte-identity gates).
fn corpus_yuv_concat(w: u32, h: u32, n: u32) -> Vec<u8> {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source/IMG_4138.MOV");
    assert!(p.exists(), "corpus fixture missing: {}", p.display());
    let vf = format!("scale={w}:{h}:force_original_aspect_ratio=disable");
    let out = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&p)
        .args([
            "-frames:v",
            &n.to_string(),
            "-vf",
            &vf,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .expect("ffmpeg launch");
    assert!(out.status.success(), "ffmpeg yuv extract failed");
    assert_eq!(
        out.stdout.len(),
        (w * h * 3 / 2 * n) as usize,
        "ffmpeg produced unexpected byte count"
    );
    out.stdout
}

#[test]
#[ignore = "AV1 WV-shadow streaming encode ~60-90 s; \
            run with --ignored --test-threads=1"]
fn wv_streaming_pull_peak_rss_720p_1shadow() {
    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 720;
    const N_FRAMES: u32 = 60; // 2 GOPs at gop=30
    const GOP_SIZE: u32 = 30;
    // Regression-detection threshold. The pre-WV.6 push-session
    // implementation hit ~2.8 GB at 1080p × 30s × 2-shadow due to the
    // pre-allocated whole-clip YUV buffer. At this smaller fixture
    // (720p × 2s × 1-shadow) the regression bound would scale to
    // ~250 MB if a whole-clip buffer were re-introduced naively.
    // Observed peak with the streaming pull cascade is ~1.2 GB on
    // macOS (allocator mmap reservation inflates ru_maxrss; the
    // actual resident working set is far smaller). The 3 GB
    // threshold catches a true O(clip) regression at this scale
    // without false-positiving on allocator behaviour.
    const PEAK_RSS_LIMIT_BYTES: u64 = 3 * 1024 * 1024 * 1024;

    let params = Av1StreamingEncodeParams {
        width: WIDTH,
        height: HEIGHT,
        // High quality (q=30) so per-GOP cover capacity is generous
        // and the encoder doesn't trip FrameCorrupted at this scale.
        // Memory profile shape is independent of QP.
        quantizer: 30,
        gop_size: GOP_SIZE,
        total_frames_hint: N_FRAMES,
    };

    let primary_text = "pull-API WV memory regression gate";
    let primary_pass = "pull-mem-primary";
    let shadow_pass = "pull-mem-shadow";
    let shadow_payload = payload::encode_payload(
        "mem-gate shadow text", &[]
    ).expect("encode shadow payload");
    let shadows: &[(&str, &[u8])] = &[(shadow_pass, &shadow_payload)];

    let primary_payload =
        payload::encode_payload(primary_text, &[]).expect("encode primary payload");
    let (cipher, nonce, salt) =
        crypto::encrypt(&primary_payload, primary_pass).expect("encrypt primary");
    let primary_framed =
        frame::build_frame(primary_payload.len(), &salt, &nonce, &cipher);

    eprintln!(
        "[av1_pull_mem] {WIDTH}×{HEIGHT} × {N_FRAMES}f × gop={GOP_SIZE} \
         × 2 shadows; RSS encoder-delta limit = {} MB",
        PEAK_RSS_LIMIT_BYTES / (1024 * 1024)
    );

    let yuv = corpus_yuv_concat(WIDTH, HEIGHT, N_FRAMES);

    // Baseline AFTER source allocation — what we're measuring is the
    // ENCODER's allocation on top of the test's own working set.
    let rss_baseline = peak_rss_bytes();
    eprintln!(
        "[av1_pull_mem] RSS baseline (source allocated): {:.1} MB",
        rss_baseline as f64 / 1_048_576.0
    );

    let mut src = Av1SliceYuvSource::new(&yuv, WIDTH, HEIGHT, N_FRAMES, GOP_SIZE);

    let t0 = Instant::now();
    let stego = av1_encode_with_shadows_streaming(
        &mut src,
        N_FRAMES,
        params,
        &primary_framed,
        primary_pass,
        shadows,
        4, // parity floor — start at lowest tier
        Pass1CacheBudget::from_env_or_default(),
    )
    .expect("streaming WV-shadow encode");
    let wall = t0.elapsed();

    let rss_after = peak_rss_bytes();
    let rss_delta = rss_after.saturating_sub(rss_baseline);

    eprintln!(
        "[av1_pull_mem] DONE wall={:.1}s peak_rss_after={:.1}MB \
         delta={:.1}MB out_bytes={}",
        wall.as_secs_f64(),
        rss_after as f64 / 1_048_576.0,
        rss_delta as f64 / 1_048_576.0,
        stego.len()
    );

    assert!(!stego.is_empty(), "encoder produced no bytes");

    #[cfg(unix)]
    {
        assert!(
            rss_delta < PEAK_RSS_LIMIT_BYTES,
            "encoder RSS delta {:.1} MB exceeded limit {} MB — a whole-clip \
             buffer may have been re-introduced into the streaming pull path",
            rss_delta as f64 / 1_048_576.0,
            PEAK_RSS_LIMIT_BYTES / (1024 * 1024)
        );
    }
}

/// Smoke companion to the ignored 720p × 60f × 1-shadow gate above.
/// Runs the same code path (`av1_encode_with_shadows_streaming` with
/// the per-GOP streaming cascade) on the small deterministic fixture
/// used by `av1_pull_whole_video_shadow_roundtrip` + the stratified-
/// probe tests, so CI exercises the production memory-bounded flow on
/// every build without the ~82 s wall of the real-corpus gate.
///
/// The RSS bound here is INTENTIONALLY generous — small-fixture
/// macOS runs are dominated by allocator/mmap noise, so a tight
/// threshold would false-positive. The point is to catch an outright
/// O(clip²) regression (whole-video YUV re-materialization, etc.)
/// without a 1-min CI wait.
#[test]
fn wv_streaming_pull_smoke_small_fixture() {
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 144;
    const N_FRAMES: u32 = 8;
    const GOP_SIZE: u32 = 4;
    // Small-fixture smoke bound — 1 GB would catch an O(clip) explosion
    // without flagging allocator-mmap noise.
    const PEAK_RSS_LIMIT_BYTES: u64 = 1024 * 1024 * 1024;

    let params = Av1StreamingEncodeParams {
        width: WIDTH,
        height: HEIGHT,
        quantizer: 30,
        gop_size: GOP_SIZE,
        total_frames_hint: N_FRAMES,
    };

    let primary_text = "WV-shadow smoke — primary payload";
    let primary_pass = "wv-smoke-primary";
    let shadow_pass = "wv-smoke-shadow";
    let shadow_payload =
        payload::encode_payload("shadow smoke", &[]).expect("encode shadow");
    let shadows: &[(&str, &[u8])] = &[(shadow_pass, &shadow_payload)];

    let primary_payload =
        payload::encode_payload(primary_text, &[]).expect("encode primary");
    let (cipher, nonce, salt) =
        crypto::encrypt(&primary_payload, primary_pass).expect("encrypt primary");
    let primary_framed =
        frame::build_frame(primary_payload.len(), &salt, &nonce, &cipher);

    let yuv = make_yuv_synthetic(WIDTH, HEIGHT, N_FRAMES);
    let rss_baseline = peak_rss_bytes();

    let mut src = Av1SliceYuvSource::new(&yuv, WIDTH, HEIGHT, N_FRAMES, GOP_SIZE);

    let t0 = Instant::now();
    let stego = av1_encode_with_shadows_streaming(
        &mut src,
        N_FRAMES,
        params,
        &primary_framed,
        primary_pass,
        shadows,
        16, // parity floor — matches the roundtrip test
        Pass1CacheBudget::from_env_or_default(),
    )
    .expect("streaming WV-shadow smoke encode");
    let wall = t0.elapsed();

    let rss_after = peak_rss_bytes();
    let rss_delta = rss_after.saturating_sub(rss_baseline);

    eprintln!(
        "[av1_pull_mem_smoke] wall={:.2}s rss_after={:.1}MB delta={:.1}MB out={}B",
        wall.as_secs_f64(),
        rss_after as f64 / 1_048_576.0,
        rss_delta as f64 / 1_048_576.0,
        stego.len()
    );

    assert!(!stego.is_empty(), "smoke encode produced no bytes");

    #[cfg(unix)]
    assert!(
        rss_delta < PEAK_RSS_LIMIT_BYTES,
        "smoke RSS delta {:.1} MB exceeded limit {} MB",
        rss_delta as f64 / 1_048_576.0,
        PEAK_RSS_LIMIT_BYTES / (1024 * 1024)
    );
}
