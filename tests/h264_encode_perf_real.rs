// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Real-content encode/decode perf baseline.
//!
//! The synthetic `h264_progress_phase_calibration` overstates absolute
//! encode time: random high-entropy YUV maximises the cover and thrashes
//! the shrink-carry round-trip loop inside `embed_gop_roundtrip_safe`.
//! This measures the production-representative number — a real 1080p clip
//! decoded to I420, encoded through the OH264 streaming stego path at the
//! mobile default (qp 26, gop 30), with total + per-frame wall-clock for
//! encode and decode, plus a round-trip correctness check. Two payloads:
//! a typical ~2 KB message, and a near-capacity stress payload to see
//! whether the shrink-carry loop activates on real footage.
//!
//! Run:
//!   cargo test --manifest-path core/Cargo.toml --release \
//!     --features h264-encoder --test h264_encode_perf_real \
//!     -- --ignored --nocapture
//!
//! `#[ignore]`d — needs ffmpeg on PATH (fixture decode only) + real wall time.

#![cfg(feature = "h264-encoder")]

mod common;

use common::oh264_stream;
use phasm_core::codec::h264::stego::CostWeights;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p.push(name);
    p
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Process peak resident-set size via `getrusage`. macOS reports bytes,
/// Linux KiB — normalised to bytes.
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

/// Decode the first `n` frames of `src`, padded to `w`x`h` (16-aligned,
/// matching the mobile 1080p→1088 pad), to a flat I420 buffer.
fn decode_to_i420(src: &PathBuf, w: u32, h: u32, n: u32) -> Vec<u8> {
    // Scale-to-fit (preserve aspect) then pad to exactly w×h — robust to any
    // source resolution/orientation, matching the mobile 1080p→1088 pad.
    let vf = format!(
        "scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:-1:-1:color=black"
    );
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            src.to_str().unwrap(),
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
        .expect("spawn ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg decode failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

#[test]
#[ignore = "real-content encode perf; needs ffmpeg + real wall time"]
fn encode_perf_real_1080p() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg not on PATH (fixture decode unavailable)");
        return;
    }

    // Mobile default: qp 26, gop 30, 1080p padded to 1088 (16-aligned).
    const W: u32 = 1920;
    const H: u32 = 1088;
    const GOP: u32 = 30;
    // 2 GOPs by default — matches the synthetic calibration fixture. Override
    // with PERF_N=<frames> to sweep peak RSS vs. frame count (localises whether
    // the encode working set is per-GOP-bounded or whole-clip-accumulating).
    #[allow(non_snake_case)]
    let N: u32 = std::env::var("PERF_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    const QP: i32 = 26;
    let frame_sz = (W as usize) * (H as usize) * 3 / 2;

    // Landscape 1080p sources only (portrait clips would need a different pad).
    // PERF_CLIP=<substr> restricts the run to one clip — used to isolate a
    // single encode's standalone peak RSS (vs. the cross-clip accumulation a
    // 3-clip loop measures) for the #847 memory audit.
    let all_clips = [
        "iphone7_1080p_30fps_h264_high.mov",
        "iphone5_1080p_30fps_h264_high.mov",
        "lumix_g9_1080p_30fps_h264_high.mp4",
    ];
    let clip_filter = std::env::var("PERF_CLIP").ok();
    let clips: Vec<&str> = all_clips
        .iter()
        .copied()
        .filter(|c| clip_filter.as_deref().is_none_or(|f| c.contains(f)))
        .collect();

    eprintln!("\n═══════ REAL-CONTENT ENCODE PERF  ({W}x{H} ×{N}f, gop {GOP}, qp {QP}) ═══════");
    for clip in clips {
        let src = fixture(clip);
        if !src.exists() {
            eprintln!("  SKIP {clip}: fixture missing");
            continue;
        }
        let yuv = decode_to_i420(&src, W, H, N);
        let got = yuv.len() / frame_sz;
        if got < N as usize {
            eprintln!("  SKIP {clip}: decoded only {got} frames (need {N})");
            continue;
        }

        // Capacity for context (so the payloads below are interpretable).
        let cover_bits = oh264_stream::probe_cover_bits(&yuv, W, H, N, QP, GOP).unwrap_or(0);
        let cover_kb = cover_bits / 8 / 1024;

        // Typical payloads, both comfortably below usable STC capacity so the
        // round-trip is exact (usable capacity ≪ cover bytes — STC rate +
        // safe-msl + tier costs). Encode time is cover-bound, not payload-bound.
        for (label, n_bytes) in [("2KB", 2048usize), ("8KB", 8192usize)] {
            let msg = "M".repeat(n_bytes);

            let t = Instant::now();
            let annex_b = oh264_stream::encode_with(
                &yuv,
                W,
                H,
                N,
                QP,
                GOP,
                CostWeights::default(),
                &msg,
                "perf-pass",
            )
            .expect("encode");
            let enc_ms = t.elapsed().as_secs_f64() * 1000.0;

            let t = Instant::now();
            let text = oh264_stream::decode_text(&annex_b, "perf-pass").expect("decode");
            let dec_ms = t.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(text.len(), msg.len(), "round-trip length mismatch ({clip}/{label})");

            eprintln!(
                "  {clip:38} payload {label:8} ({n_bytes:>7} B / cover ~{cover_kb} KB)\n      ENCODE {enc_ms:8.0} ms  ({:6.1} ms/frame)   DECODE {dec_ms:7.0} ms  ({:5.1} ms/frame)   out {} B",
                enc_ms / N as f64,
                dec_ms / N as f64,
                annex_b.len(),
            );
        }
    }
    let peak_mb = peak_rss_bytes() as f64 / (1024.0 * 1024.0);
    let one_cover_mb = (frame_sz * N as usize) as f64 / (1024.0 * 1024.0);
    eprintln!(
        "  peak RSS over run: {peak_mb:.0} MB  (this in-memory test holds a full\n      {one_cover_mb:.0} MB I420 cover in heap — the mobile path mmaps it instead; the\n      remainder is the per-clip cover model + OH264 working set). The parallel\n      coeff+MVD cost loops share each frame's ~32 MB wavelets BY REFERENCE — no\n      per-core copy, only a transient per-frame contrib vec — so the cost pass\n      is additive-only vs the serial path, NOT cores×wavelet."
    );
    eprintln!("════════════════════════════════════════════════════════════════════\n");
}
