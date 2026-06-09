// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.4.3 (#408) — spatial per-MB map gate on OH264 backend.
//
// Defense-in-depth companion to the marginal-histogram L3 gate
// (`openh264_stealth_distribution_gate.rs`, C.4.1). For each fixture:
//
//   1. Encode CLEAN OH264 (no stego) via low-level `Encoder` API.
//   2. Encode STEGO OH264 via the regular stego pipeline.
//   3. Run `scripts/h264_mb_grid.py` on both Annex-B outputs to dump
//      per-MB-position `(qp, type_char)` tokens in raster order.
//   4. Diff the two grids cell-by-cell; assert mismatch count below
//      threshold.
//
// The architectural invariant we're protecting:
//   Phase C.8 `visual_recon` keeps the encoder reference state
//   (`pDecPic`) clean throughout the encode. Stego flips happen at
//   CABAC emission time, AFTER mode-decision has locked in mb_type /
//   partition shape / MV direction. Mode-decision sees clean pixels,
//   so per-MB mode choice MUST equal clean OH264's per-MB choice.
//
// If that invariant ever silently breaks (a fork-side regression that
// lets stego flips reach the encoder reference path), the marginal
// gate may still pass within ε while this spatial gate fires with a
// clear "N MBs differ" signal pointing at the regression.
//
// Expected steady-state: mismatch_count == 0 on every fixture. We
// assert strict zero; non-zero is always a real regression — there's
// no legitimate reason for clean and stego to disagree at the per-MB
// mode-decision level given the visual_recon invariant.
//
// Run smoke (default lane):
//   cargo test --release --features "h264-encoder" \
//     --test openh264_spatial_mb_map_gate
//
// Run full corpus:
//   cargo test --release --features "h264-encoder" \
//     --test openh264_spatial_mb_map_gate -- --ignored --nocapture

#![cfg(feature = "h264-encoder")]

mod common;
use common::oh264_stream;

use phasm_core::codec::h264::openh264::Encoder;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn script_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("..");
    p.push("scripts");
    p.push("h264_mb_grid.py");
    p
}

struct Fixture {
    name: &'static str,
    source_mp4: &'static str,
    n_frames: u32,
    qp: i32,
    intra_period: i32,
}

/// Same dim-rule as `regen_openh264_baseline.sh` — longest side =
/// 1920, both axes 16-aligned. Matches the marginal-gate baseline.
fn probe_baseline_dims(spec: &Fixture) -> (u32, u32) {
    let src = corpus_root().join(spec.source_mp4);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let out = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height",
               "-of", "csv=p=0"])
        .arg(&src)
        .output()
        .expect("ffprobe");
    let s = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<&str> = s.trim().split(',').collect();
    let iw: u32 = parts[0].parse().expect("source width");
    let ih: u32 = parts[1].parse().expect("source height");
    if iw >= ih {
        (1920, ((1920 * ih) / iw / 16) * 16)
    } else {
        (((1920 * iw) / ih / 16) * 16, 1920)
    }
}

fn ensure_yuv(spec: &Fixture, w: u32, h: u32) -> Vec<u8> {
    let yuv_path = format!(
        "/tmp/openh264_baseline_{}_{}x{}_f{}.yuv",
        spec.name, w, h, spec.n_frames
    );
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * spec.n_frames as usize;
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need { return data; }
    }
    let src = corpus_root().join(spec.source_mp4);
    let vf = format!("scale={w}:{h},format=yuv420p");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &spec.n_frames.to_string(),
               "-vf", &vf, "-f", "rawvideo"])
        .arg(&yuv_path)
        .status()
        .expect("ffmpeg launch");
    assert!(status.success(), "ffmpeg scale failed for {}", spec.source_mp4);
    std::fs::read(&yuv_path).expect("read yuv")
}

/// Encode a YUV frame stream via the low-level `Encoder` API with no
/// stego callbacks registered (mirrors `examples/openh264_clean_encode.rs`).
fn encode_clean(yuv: &[u8], w: u32, h: u32, n_frames: u32, qp: i32, gop_size: i32) -> Vec<u8> {
    let luma_plane = (w * h) as usize;
    let chroma_plane = ((w / 2) * (h / 2)) as usize;
    let frame_bytes = luma_plane + 2 * chroma_plane;

    let mut enc = Encoder::new(w as i32, h as i32, qp, gop_size)
        .expect("openh264 clean Encoder::new");
    let mut nal_buf = vec![0u8; frame_bytes * 2];
    let mut out = Vec::with_capacity(frame_bytes / 4 * n_frames as usize);

    for i in 0..n_frames {
        let off = i as usize * frame_bytes;
        let y = &yuv[off..off + luma_plane];
        let u = &yuv[off + luma_plane..off + luma_plane + chroma_plane];
        let v = &yuv[off + luma_plane + chroma_plane..off + frame_bytes];
        let ts = (i as i64) * 33;
        let (_ft, n) = enc
            .encode_frame(y, u, v, ts, &mut nal_buf)
            .expect("openh264 clean encode_frame");
        out.extend_from_slice(&nal_buf[..n]);
    }
    out
}

/// Run `scripts/h264_mb_grid.py` on an Annex-B file → grid text.
fn extract_mb_grid(annex_b: &[u8], tag: &str) -> String {
    let h264_path = std::env::temp_dir().join(format!("oh264_spatial_{tag}.h264"));
    let grid_path = std::env::temp_dir().join(format!("oh264_spatial_{tag}.grid.txt"));
    std::fs::write(&h264_path, annex_b).expect("write annex-b");
    let status = std::process::Command::new("python3")
        .arg(script_path())
        .arg(&h264_path)
        .arg(&grid_path)
        .status()
        .expect("invoke mb_grid script");
    assert!(status.success(), "mb_grid script failed for {tag}");
    std::fs::read_to_string(&grid_path).expect("read grid")
}

/// Parse grid text → flat per-frame `Vec<Vec<String>>` (frame -> tokens).
/// Drops `# frame ...` headers; preserves token order within each frame.
fn parse_grid(text: &str) -> Vec<Vec<String>> {
    let mut frames: Vec<Vec<String>> = Vec::new();
    let mut current: Option<Vec<String>> = None;
    for line in text.lines() {
        if line.starts_with("# frame") {
            if let Some(f) = current.take() {
                frames.push(f);
            }
            current = Some(Vec::new());
        } else if let Some(ref mut f) = current {
            for tok in line.split_whitespace() {
                f.push(tok.to_string());
            }
        }
    }
    if let Some(f) = current {
        frames.push(f);
    }
    frames
}

fn run_spatial_gate(spec: &Fixture) {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());

    let (w, h) = probe_baseline_dims(spec);
    let yuv = ensure_yuv(spec, w, h);

    // Stego encode (production 4-domain streaming session). Single GOP:
    // every fixture is 10 frames with intra_period 30, so the whole clip
    // is one GOP — pass `n_frames` for gop_size.
    let stego = oh264_stream::encode(
        &yuv, w, h, spec.n_frames,
        spec.qp,
        "C.4.3 spatial gate", "spatial-pass",
    ).expect("oh264 stego encode");

    // Clean encode (no stego callbacks).
    let clean = encode_clean(&yuv, w, h, spec.n_frames, spec.qp, spec.intra_period);

    let clean_grid = parse_grid(&extract_mb_grid(&clean, &format!("{}_clean", spec.name)));
    let stego_grid = parse_grid(&extract_mb_grid(&stego, &format!("{}_stego", spec.name)));

    assert_eq!(
        clean_grid.len(), stego_grid.len(),
        "[{}] frame-count mismatch: clean has {} frames, stego has {}",
        spec.name, clean_grid.len(), stego_grid.len(),
    );

    let mut total_mb = 0_u32;
    let mut mismatches: Vec<(usize, usize, String, String)> = Vec::new();
    for (frame_idx, (cf, sf)) in clean_grid.iter().zip(stego_grid.iter()).enumerate() {
        assert_eq!(
            cf.len(), sf.len(),
            "[{}] frame {} MB-count mismatch: clean={}, stego={}",
            spec.name, frame_idx, cf.len(), sf.len(),
        );
        total_mb += cf.len() as u32;
        for (mb_idx, (c, s)) in cf.iter().zip(sf.iter()).enumerate() {
            if c != s {
                mismatches.push((frame_idx, mb_idx, c.clone(), s.clone()));
            }
        }
    }

    let n_mismatch = mismatches.len();
    eprintln!(
        "[{}] {}×{}×{} qp={} spatial gate: {} MBs total, {} mismatched ({:.3}%)",
        spec.name, w, h, spec.n_frames, spec.qp,
        total_mb, n_mismatch, n_mismatch as f64 / total_mb as f64 * 100.0,
    );

    if n_mismatch != 0 {
        // Show first few mismatches to help diagnose.
        eprintln!("  first up to 5 mismatches (frame, mb_idx, clean -> stego):");
        for (fi, mi, c, s) in mismatches.iter().take(5) {
            eprintln!("    frame {fi:>2} mb {mi:>5}: {c} -> {s}");
        }
    }
    assert_eq!(
        n_mismatch, 0,
        "[{}] visual_recon invariant violated: {} per-MB mode-decision mismatches \
         between clean and stego OH264 (expected 0). Stego flips are leaking into \
         the encoder reference path.",
        spec.name, n_mismatch,
    );
}

// ─────────────────────────── smoke ────────────────────────────────────

#[test]
fn spatial_oh264_smoke_img4138() {
    run_spatial_gate(&Fixture {
        name: "img4138_smoke",
        source_mp4: "IMG_4138.MOV",
        n_frames: 10, qp: 26, intra_period: 30,
    });
}

// ─────────────────────────── production ───────────────────────────────

#[test]
#[ignore]
fn spatial_oh264_img4138_1080p() {
    run_spatial_gate(&Fixture {
        name: "img4138_1080p",
        source_mp4: "IMG_4138.MOV",
        n_frames: 10, qp: 26, intra_period: 30,
    });
}

#[test]
#[ignore]
fn spatial_oh264_img4273_1080p() {
    run_spatial_gate(&Fixture {
        name: "img4273_1080p",
        source_mp4: "IMG_4273.MOV",
        n_frames: 10, qp: 26, intra_period: 30,
    });
}

#[test]
#[ignore]
fn spatial_oh264_carplane_1080p() {
    run_spatial_gate(&Fixture {
        name: "carplane_1080p",
        source_mp4: "Artlist_CarPlane.mp4",
        n_frames: 10, qp: 26, intra_period: 30,
    });
}

#[test]
#[ignore]
fn spatial_oh264_dji_mini2_1080p() {
    run_spatial_gate(&Fixture {
        name: "dji_mini2_1080p",
        source_mp4: "dji_mini2_2_7k_24fps_h264_high.mp4",
        n_frames: 10, qp: 26, intra_period: 30,
    });
}

#[test]
#[ignore]
fn spatial_oh264_lumix_g9_1080p() {
    run_spatial_gate(&Fixture {
        name: "lumix_g9_1080p",
        source_mp4: "lumix_g9_1080p_30fps_h264_high.mp4",
        n_frames: 10, qp: 26, intra_period: 30,
    });
}
