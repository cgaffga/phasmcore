// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! G.2 Phase 2 calibration measurement binary.
//!
//! Runs the AV1 streaming probe over a single clip and emits per-frame
//! cover-bit measurements + aggregate distribution stats as JSON. The
//! aggregator (a separate step) combines per-clip JSON into the
//! corpus-wide calibration constants for the balanced-allocation
//! planner (see `docs/design/video/balanced-allocation-v3.md` § 5).
//!
//! ## Usage
//!
//! ```bash
//! av1_calibrate <clip_path> <qp> <max_frames> <out_json> [width] [height]
//! ```
//!
//! - `clip_path` — absolute path to source MP4/MOV.
//! - `qp` — rav1e quantizer (typically 30 = production default).
//! - `max_frames` — cap on frames to probe (keeps wall-clock bounded;
//!   a 5-min clip at 30 fps would otherwise be 9000 probes). Typical
//!   value: 900 (= 30 sec at 30 fps).
//! - `out_json` — output path for the JSON measurement file.
//! - `width`/`height` — optional ffmpeg scale target. If omitted,
//!   ffprobe is used to detect native resolution (rounded to nearest
//!   multiple of 8).
//!
//! ## Probe semantics
//!
//! Runs `Av1StreamingProbeSession` at `gop_size=1` (current probe API
//! constraint). Each frame is therefore treated as an independent
//! key-frame; per-frame cover bits over-estimate the inter-frame yield
//! of a real GOP encode. For Phase 1 calibration this gives the
//! per-frame upper bound; a Phase 2 follow-on can extend the probe to
//! the real gop_size for inter-frame-accurate measurements.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::Path;
use std::process::Command;

use phasm_core::codec::av1::stego::capacity::Av1StreamingProbeSession;
use phasm_core::codec::av1::stego::session::Av1StreamingEncodeParams;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: {} <clip_path> <qp> <max_frames> <out_json> [width] [height]",
            args[0]
        );
        std::process::exit(2);
    }
    let clip_path = &args[1];
    let qp: usize = args[2].parse().expect("qp must be integer");
    let max_frames: u32 = args[3].parse().expect("max_frames must be integer");
    let out_json = &args[4];
    let (width, height) = if args.len() >= 7 {
        (
            args[5].parse::<u32>().expect("width must be integer"),
            args[6].parse::<u32>().expect("height must be integer"),
        )
    } else {
        detect_native_dims(clip_path)
    };

    let (width, height) = round8_dims(width, height);

    eprintln!(
        "[av1_calibrate] clip={} qp={} max_frames={} dims={}x{}",
        clip_path, qp, max_frames, width, height
    );

    let frame_size = expected_i420_size(width, height);
    let mut probe = Av1StreamingProbeSession::create(Av1StreamingEncodeParams {
        width,
        height,
        quantizer: qp,
        gop_size: 1,
        total_frames_hint: max_frames,
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
    let mut frame_idx: u32 = 0;
    let t_start = std::time::Instant::now();
    loop {
        match read_exact_or_eof(&mut stdout, &mut buf) {
            Ok(true) => {
                probe.push_frame(&buf).expect("probe.push_frame");
                frame_idx += 1;
                if frame_idx >= max_frames {
                    break;
                }
            }
            Ok(false) => break, // EOF
            Err(e) => {
                eprintln!("[av1_calibrate] ffmpeg stdout error: {e}");
                break;
            }
        }
    }
    let _ = child.wait();
    let probe_wall_ms = t_start.elapsed().as_millis();

    let result = probe.finish();
    let n_frames = result.per_gop_cover_bits.len();
    let cover_bits = result.per_gop_cover_bits.clone();

    eprintln!(
        "[av1_calibrate] probed {} frames in {} ms",
        n_frames, probe_wall_ms
    );

    // Stats.
    let stats = compute_stats(&cover_bits);

    write_json(
        out_json,
        clip_path,
        width,
        height,
        qp,
        max_frames,
        probe_wall_ms,
        &cover_bits,
        &stats,
    )
    .expect("write_json");
    eprintln!("[av1_calibrate] wrote {out_json}");
}

fn round8_dims(w: u32, h: u32) -> (u32, u32) {
    ((w / 8) * 8, (h / 8) * 8)
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
    qp: usize,
    max_frames: u32,
    probe_wall_ms: u128,
    per_frame_cover_bits: &[usize],
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
    json.push_str(&format!("  \"width\": {},\n", width));
    json.push_str(&format!("  \"height\": {},\n", height));
    json.push_str(&format!("  \"qp\": {},\n", qp));
    json.push_str(&format!("  \"max_frames\": {},\n", max_frames));
    json.push_str(&format!("  \"probe_wall_ms\": {},\n", probe_wall_ms));
    json.push_str(&format!("  \"n_frames\": {},\n", s.n));
    json.push_str(&format!("  \"sum_cover_bits\": {},\n", s.sum));
    json.push_str(&format!("  \"mean_cover_bits_per_frame\": {:.2},\n", s.mean));
    json.push_str(&format!("  \"stddev\": {:.2},\n", s.stddev));
    json.push_str(&format!("  \"cv\": {:.4},\n", s.cv));
    json.push_str(&format!("  \"min\": {},\n", s.min));
    json.push_str(&format!("  \"p1\": {},\n", s.p1));
    json.push_str(&format!("  \"p5\": {},\n", s.p5));
    json.push_str(&format!("  \"p50\": {},\n", s.p50));
    json.push_str(&format!("  \"p95\": {},\n", s.p95));
    json.push_str(&format!("  \"p99\": {},\n", s.p99));
    json.push_str(&format!("  \"max\": {},\n", s.max));
    json.push_str("  \"per_frame_cover_bits\": [");
    for (i, v) in per_frame_cover_bits.iter().enumerate() {
        if i > 0 {
            json.push_str(", ");
        }
        json.push_str(&v.to_string());
    }
    json.push_str("]\n");
    json.push_str("}\n");

    std::fs::write(path, json)
}
