// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
//
// #796 Mode A diagnostic — quantify the gap between
// `h264_stego_capacity_4domain` (pure-Rust whole-video walker) and
// the OH264 streaming-session per-GOP cover.
//
// For each failing fixture, prints side-by-side:
//   - pure-Rust capacity_4domain primary_max_message_bytes
//   - OH264 baseline encode + walk → CSB+CSL+MSB position count → bytes
//   - max message that StreamingEncodeSession actually accepts (binary-search)
//
// The 3-way gap is the bug. With this data we can decide whether to:
//   (a) recalibrate `h264_stego_capacity_4domain` to use OH264 walker
//   (b) make StreamingEncodeSession use pure-Rust pass1 for its STC w
//   (c) something else

#![cfg(all(feature = "openh264-backend", feature = "cabac-stego"))]

use phasm_core::{
    h264_stego_capacity_4domain, h264_stego_capacity_4domain_oh264,
    ColorParams, CostWeights, EncodeEngineChoice, EncodeSessionParams,
    StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::h264::openh264_stego::EncodeOpts;
use std::sync::{Mutex, OnceLock};

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn try_scale_yuv(source_mp4: &str, tag: &str, w: u32, h: u32, n_frames: u32) -> Option<Vec<u8>> {
    let src = corpus_root().join(source_mp4);
    if !src.exists() {
        return None;
    }
    let yuv_path = format!("/tmp/phasm_796_diag_{}_{}x{}_f{}.yuv", tag, w, h, n_frames);
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * (n_frames as usize);
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need {
            return Some(data[..need].to_vec());
        }
    }
    let vf = format!("scale={}:{}", w, h);
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &n_frames.to_string()])
        .args(["-an", "-vf", &vf])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let data = std::fs::read(&yuv_path).ok()?;
    if data.len() < need {
        return None;
    }
    Some(data[..need].to_vec())
}

/// Try encoding with a specific message size to see if StreamingEncodeSession accepts it.
fn try_streaming_encode(
    yuv: &[u8],
    w: u32, h: u32, n_frames: u32, gop: u32,
    msg_size: usize,
) -> bool {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let msg: String = std::iter::repeat('x').take(msg_size).collect();
    let pass = "diag-pass";
    let params = EncodeSessionParams {
        width: w, height: h,
        fps_num: 30, fps_den: 1,
        qp: 26, gop_size: gop,
        total_frames_hint: n_frames,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let Ok(mut enc) = StreamingEncodeSession::create(params, &msg, pass) else { return false; };

    let frame_bytes = (w as usize) * (h as usize) * 3 / 2;
    let chroma_w = (w as usize) / 2;
    let chroma_h = (h as usize) / 2;
    let y_size = (w as usize) * (h as usize);
    let chroma_size = chroma_w * chroma_h;

    let mut annex_b = Vec::new();
    for f in 0..n_frames {
        let off = (f as usize) * frame_bytes;
        let y_plane = &yuv[off..off + y_size];
        let u_plane = &yuv[off + y_size..off + y_size + chroma_size];
        let v_plane = &yuv[off + y_size + chroma_size..off + y_size + 2 * chroma_size];
        let frame = YuvFrameRef {
            y: y_plane, y_stride: w as usize,
            u: u_plane, u_stride: chroma_w,
            v: v_plane, v_stride: chroma_w,
        };
        if enc.push_frame(frame, &mut annex_b).is_err() {
            return false;
        }
    }
    enc.finish(&mut annex_b).is_ok()
}

/// Binary search for the largest message size that StreamingEncodeSession accepts.
fn measured_max_msg(yuv: &[u8], w: u32, h: u32, n_frames: u32, gop: u32, hi_hint: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = hi_hint.max(64);
    // Confirm hi is too big.
    if try_streaming_encode(yuv, w, h, n_frames, gop, hi) {
        return hi; // surprise — fits at the upper bound
    }
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if try_streaming_encode(yuv, w, h, n_frames, gop, mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

struct Fixture { tag: &'static str, source: &'static str }

const FIXTURES: &[Fixture] = &[
    Fixture { tag: "lumix",        source: "lumix_g9_1080p_30fps_h264_high.mp4" },
    Fixture { tag: "dji",          source: "dji_mini2_2_7k_24fps_h264_high.mp4" },
    Fixture { tag: "school_fight", source: "Artlist_SchoolFight.mp4" },
    Fixture { tag: "woman_subway", source: "Artlist_WomanSubway.mp4" },
    // Controls — known-OK fixtures.
    Fixture { tag: "iphone5",      source: "iphone5_1080p_30fps_h264_high.mov" },
    Fixture { tag: "carplane",     source: "Artlist_CarPlane.mp4" },
];

#[test]
fn diag_capacity_vs_encoder_mismatch_30f() {
    diag_run(30, 30);
}

#[test]
fn diag_capacity_vs_encoder_mismatch_8f() {
    diag_run(8, 8);
}

fn diag_run(n_frames: u32, gop: u32) {
    const W: u32 = 480;
    const H: u32 = 272;

    eprintln!("\n=== #796 Mode A diagnostic — {W}x{H}x{n_frames} gop={gop} ===");
    eprintln!(
        "{:<14} | {:>9} | {:>9} | {:>9} | {:>5} | note",
        "fixture", "pure-Rust", "OH264 new", "measured", "gap%",
    );

    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };

    for fx in FIXTURES {
        let Some(yuv) = try_scale_yuv(fx.source, fx.tag, W, H, n_frames) else {
            eprintln!("{:<14} | (source missing — skipped)", fx.tag);
            continue;
        };

        let pure_rust = h264_stego_capacity_4domain(
            &yuv, W, H, n_frames as usize, gop as usize,
        ).map(|i| i.primary_max_message_bytes).unwrap_or(0);

        let oh264 = h264_stego_capacity_4domain_oh264(
            &yuv, W, H, n_frames as usize, opts, false,
        ).map(|i| i.primary_max_message_bytes).unwrap_or(0);

        let measured = measured_max_msg(&yuv, W, H, n_frames, gop, pure_rust);

        let gap_pct = if oh264 > 0 {
            100.0 * (oh264 as f64 - measured as f64) / oh264 as f64
        } else if measured == 0 { 0.0 } else { 100.0 };

        let note = if measured == 0 && oh264 > 0 {
            "decode/cascade bug (Mode B): OH264 reports OK, encoder rejects"
        } else if measured == 0 {
            "encoder rejects ALL message sizes"
        } else if oh264 <= measured {
            "OK (OH264 capacity ≤ encoder-accepted)"
        } else if gap_pct.abs() < 20.0 {
            "OK (OH264 within 20% of encoder)"
        } else {
            "OH264 still over-reports"
        };

        eprintln!(
            "{:<14} | {:>9} | {:>9} | {:>9} | {:>4.1}% | {}",
            fx.tag, pure_rust, oh264, measured, gap_pct, note,
        );
    }
}
