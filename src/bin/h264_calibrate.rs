// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! G.2 Phase 2 Tier 1 calibration measurement binary — H.264 (#844).
//!
//! Codec-parity counterpart to `av1_calibrate.rs`. Runs the H.264
//! `StreamingProbeSession` over a single clip at the PRODUCTION GOP
//! size and emits per-GOP cap-byte measurements + aggregate stats as
//! JSON. A separate aggregation step combines per-clip JSON into the
//! corpus-wide `AllocationCalibration` constants for the
//! balanced-allocation planner (see
//! `docs/design/video/balanced-allocation-v3.md` § 5 and
//! `docs/design/video/calibration-spike-2026-06.md`).
//!
//! ## Usage
//!
//! ```bash
//! h264_calibrate <clip_path> <qp> <gop_size> <max_frames> <out_json> [width] [height]
//! ```
//!
//! - `clip_path` — absolute path to source MP4/MOV.
//! - `qp` — OH264 quantizer (26 = mobile production default).
//! - `gop_size` — frames per GOP (30 = mobile production default).
//! - `max_frames` — cap on frames to probe. 900 (= 30 s at 30 fps =
//!   30 GOPs at gop_size=30) gives a usable per-clip CV sample.
//! - `out_json` — output path for the JSON measurement file.
//! - `width`/`height` — optional ffmpeg scale target. If omitted,
//!   ffprobe detects native resolution. Both are rounded to the
//!   nearest multiple of 16 (H.264 macroblock alignment; the probe
//!   rejects non-16-aligned dims).
//!
//! ## Probe semantics — DIFFERS from av1_calibrate
//!
//! Unlike `av1_calibrate` (locked at `gop_size=1`, per-FRAME upper
//! bound), the H.264 `StreamingProbeSession` runs the REAL per-GOP
//! baseline encode at the production `gop_size`. The reported per-GOP
//! cap bytes are therefore the actual quantity the planner samples
//! (`plan_safe_balanced`'s `(gop_idx, cap_bytes)`), inter-frame
//! accurate — NOT an upper bound. This is the codec-parity Tier 1 the
//! AV1 spike deferred (calibration-spike-2026-06.md follow-on #6).

#![cfg(all(feature = "h264-encoder", feature = "h264-decoder"))]

use std::path::Path;
use std::process::Command;

use phasm_core::codec::h264::stego::CostWeights;
use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingProbeSession, YuvFrameRef,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "usage: {} <clip_path> <qp> <gop_size> <max_frames> <out_json> [width] [height]",
            args[0]
        );
        std::process::exit(2);
    }
    let clip_path = &args[1];
    let qp: i32 = args[2].parse().expect("qp must be integer");
    let gop_size: u32 = args[3].parse().expect("gop_size must be integer");
    let max_frames: u32 = args[4].parse().expect("max_frames must be integer");
    let out_json = &args[5];
    let (width, height) = if args.len() >= 8 {
        (
            args[6].parse::<u32>().expect("width must be integer"),
            args[7].parse::<u32>().expect("height must be integer"),
        )
    } else {
        detect_native_dims(clip_path)
    };

    let (width, height) = round16_dims(width, height);

    eprintln!(
        "[h264_calibrate] clip={} qp={} gop_size={} max_frames={} dims={}x{}",
        clip_path, qp, gop_size, max_frames, width, height
    );

    let frame_size = expected_i420_size(width, height);
    let mut probe = StreamingProbeSession::create(EncodeSessionParams {
        width,
        height,
        fps_num: 30,
        fps_den: 1,
        qp,
        gop_size,
        total_frames_hint: max_frames,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        // Production cost weights — these drive the per-GOP STC plan, so the
        // measured per-GOP cap bytes match what mobile/CLI encode produces.
        cost_weights: CostWeights::default(),
        progress_callback: None,
    })
    .expect("probe.create");

    let mut child = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(clip_path)
        .args([
            "-frames:v",
            &max_frames.to_string(),
            "-vf",
            &format!("scale={}:{}:force_original_aspect_ratio=disable", width, height),
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("ffmpeg spawn");

    let mut stdout = child.stdout.take().expect("ffmpeg stdout");
    let mut buf = vec![0u8; frame_size];
    // Tight I420 plane geometry for the YuvFrameRef.
    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let c_w = w / 2;
    let uv_size = c_w * (h / 2);
    let mut frame_idx: u32 = 0;
    let t_start = std::time::Instant::now();
    loop {
        match read_exact_or_eof(&mut stdout, &mut buf) {
            Ok(true) => {
                let frame = YuvFrameRef {
                    y: &buf[..y_size],
                    y_stride: w,
                    u: &buf[y_size..y_size + uv_size],
                    u_stride: c_w,
                    v: &buf[y_size + uv_size..y_size + 2 * uv_size],
                    v_stride: c_w,
                };
                probe.push_frame(frame).expect("probe.push_frame");
                frame_idx += 1;
                if frame_idx >= max_frames {
                    break;
                }
            }
            Ok(false) => break, // EOF
            Err(e) => {
                eprintln!("[h264_calibrate] ffmpeg stdout error: {e}");
                break;
            }
        }
    }
    let _ = child.wait();
    let probe_wall_ms = t_start.elapsed().as_millis();

    let (result, per_gop_caps) = probe.finish_with_per_gop().expect("probe.finish");
    let n_gops = result.n_gops as usize;

    eprintln!(
        "[h264_calibrate] probed {} frames -> {} GOPs in {} ms",
        frame_idx, n_gops, probe_wall_ms
    );

    // Per-GOP cap bytes (tier 0) — the exact quantity the planner samples.
    let stats = compute_stats(&per_gop_caps);

    write_json(
        out_json,
        clip_path,
        width,
        height,
        qp,
        gop_size,
        max_frames,
        probe_wall_ms,
        result.cover_bits,
        result.shadow_pool_bits,
        result.primary_max_message_bytes(),
        &per_gop_caps,
        &stats,
    )
    .expect("write_json");
    eprintln!("[h264_calibrate] wrote {out_json}");
}

fn round16_dims(w: u32, h: u32) -> (u32, u32) {
    // Round to nearest multiple of 16 (H.264 MB alignment). Nearest, not
    // floor: 1080 -> 1088 (standard H.264 1080p coded size), 1072 would
    // drop content.
    let r16 = |x: u32| ((x + 8) / 16) * 16;
    (r16(w).max(16), r16(h).max(16))
}

fn expected_i420_size(w: u32, h: u32) -> usize {
    (w as usize) * (h as usize) * 3 / 2
}

fn detect_native_dims(clip: &str) -> (u32, u32) {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
        ])
        .arg(clip)
        .output()
        .expect("ffprobe failed");
    if !out.status.success() {
        panic!(
            "ffprobe failed on {clip}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 2 {
        panic!("ffprobe output unexpected: {s}");
    }
    (
        parts[0].parse().expect("width parse"),
        parts[1].parse().expect("height parse"),
    )
}

fn read_exact_or_eof<R: std::io::Read>(r: &mut R, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut done = 0;
    while done < buf.len() {
        match r.read(&mut buf[done..]) {
            Ok(0) => return Ok(done > 0 && done == buf.len()),
            Ok(n) => done += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

#[derive(Debug)]
struct Stats {
    n: usize,
    sum: u64,
    mean: f64,
    stddev: f64,
    cv: f64,
    min: usize,
    p1: usize,
    p5: usize,
    p50: usize,
    p95: usize,
    p99: usize,
    max: usize,
}

fn compute_stats(v: &[usize]) -> Stats {
    if v.is_empty() {
        return Stats {
            n: 0,
            sum: 0,
            mean: 0.0,
            stddev: 0.0,
            cv: 0.0,
            min: 0,
            p1: 0,
            p5: 0,
            p50: 0,
            p95: 0,
            p99: 0,
            max: 0,
        };
    }
    let n = v.len();
    let sum: u64 = v.iter().map(|x| *x as u64).sum();
    let mean = sum as f64 / n as f64;
    let var = v
        .iter()
        .map(|x| {
            let d = *x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let stddev = var.sqrt();
    let cv = if mean > 0.0 { stddev / mean } else { 0.0 };

    let mut sorted: Vec<usize> = v.to_vec();
    sorted.sort_unstable();
    let pct = |p: f64| -> usize {
        let idx = ((p / 100.0) * (n - 1) as f64).round() as usize;
        sorted[idx.min(n - 1)]
    };

    Stats {
        n,
        sum,
        mean,
        stddev,
        cv,
        min: sorted[0],
        p1: pct(1.0),
        p5: pct(5.0),
        p50: pct(50.0),
        p95: pct(95.0),
        p99: pct(99.0),
        max: sorted[n - 1],
    }
}

#[allow(clippy::too_many_arguments)]
fn write_json(
    path: &str,
    clip_path: &str,
    width: u32,
    height: u32,
    qp: i32,
    gop_size: u32,
    max_frames: u32,
    probe_wall_ms: u128,
    cover_bits: usize,
    shadow_pool_bits: usize,
    primary_max_message_bytes: usize,
    per_gop_caps: &[usize],
    s: &Stats,
) -> std::io::Result<()> {
    let parent = Path::new(path).parent();
    if let Some(p) = parent {
        std::fs::create_dir_all(p)?;
    }
    let clip_name = Path::new(clip_path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| clip_path.to_string());

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"clip\": \"{}\",\n", clip_name));
    json.push_str("  \"codec\": \"h264\",\n");
    json.push_str(&format!("  \"width\": {},\n", width));
    json.push_str(&format!("  \"height\": {},\n", height));
    json.push_str(&format!("  \"qp\": {},\n", qp));
    json.push_str(&format!("  \"gop_size\": {},\n", gop_size));
    json.push_str(&format!("  \"max_frames\": {},\n", max_frames));
    json.push_str(&format!("  \"probe_wall_ms\": {},\n", probe_wall_ms));
    json.push_str(&format!("  \"n_gops\": {},\n", s.n));
    json.push_str(&format!("  \"cover_bits_total\": {},\n", cover_bits));
    json.push_str(&format!("  \"shadow_pool_bits_total\": {},\n", shadow_pool_bits));
    json.push_str(&format!(
        "  \"primary_max_message_bytes\": {},\n",
        primary_max_message_bytes
    ));
    // Stats are on PER-GOP cap bytes (tier 0) — the planner's sample unit.
    json.push_str(&format!("  \"sum_cap_bytes\": {},\n", s.sum));
    json.push_str(&format!("  \"mean_cap_bytes_per_gop\": {:.2},\n", s.mean));
    json.push_str(&format!("  \"stddev\": {:.2},\n", s.stddev));
    json.push_str(&format!("  \"cv\": {:.4},\n", s.cv));
    json.push_str(&format!("  \"min\": {},\n", s.min));
    json.push_str(&format!("  \"p1\": {},\n", s.p1));
    json.push_str(&format!("  \"p5\": {},\n", s.p5));
    json.push_str(&format!("  \"p50\": {},\n", s.p50));
    json.push_str(&format!("  \"p95\": {},\n", s.p95));
    json.push_str(&format!("  \"p99\": {},\n", s.p99));
    json.push_str(&format!("  \"max\": {},\n", s.max));
    // Derived: per-frame cap bits = cap_bytes * 8 / gop_size (the
    // `cover_bits_per_frame_floor` unit the planner's table_pred uses).
    let min_per_frame_bits = if gop_size > 0 {
        (s.min as f64) * 8.0 / (gop_size as f64)
    } else {
        0.0
    };
    json.push_str(&format!(
        "  \"min_cap_bits_per_frame\": {:.1},\n",
        min_per_frame_bits
    ));
    json.push_str("  \"per_gop_cap_bytes\": [");
    for (i, v) in per_gop_caps.iter().enumerate() {
        if i > 0 {
            json.push_str(", ");
        }
        json.push_str(&v.to_string());
    }
    json.push_str("]\n");
    json.push_str("}\n");

    std::fs::write(path, json)
}
