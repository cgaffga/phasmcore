// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.1 (#397-#400) — corpus round-trip on OpenH264-backend.
//
// One #[test] per fixture so cargo's parallel runner spreads the
// encode workload across cores. Each fixture:
//   1. Scales corpus MP4 → YUV at largest 16-aligned dim (cached at /tmp).
//   2. Encodes via OH264 stego pipeline with a passphrase-driven message.
//   3. Decodes back via openh264_stego_decode_yuv_string → assert recovery.
//   4. Muxes the stego annex-B to mp4 via ffmpeg, writing to
//      ~/Desktop/phasm_oh264_<fixture>_stego.mp4 for visual review.
//
// Smoke variant (`corpus_oh264_smoke_iphone7`) runs by default at
// 640×368×8 frames. Production variants are `#[ignore]` (1080p × 30).
//
// Run:
//   cargo test --release --features "h264-encoder openh264-backend" \
//       --test openh264_corpus_validation -- --ignored --nocapture

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use phasm_core::codec::h264::openh264_stego::{
    openh264_stego_decode_yuv_string, openh264_stego_encode_yuv_text, EncodeOpts,
};
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

fn desktop_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME env var");
    let mut p = PathBuf::from(home);
    p.push("Desktop");
    p
}

struct Fixture {
    name: &'static str,
    source_mp4: &'static str,
    /// Optional cap on the longer side (pixels). Source aspect is
    /// preserved; both axes are rounded down to the nearest multiple
    /// of 16 (H.264 MB alignment). `None` = use source dims with only
    /// the 16-align rounding.
    long_side_cap: Option<u32>,
    n_frames: u32,
    qp: i32,
}

/// Probe source W×H via ffprobe, then return 16-aligned dims that
/// preserve the source aspect ratio. If `long_side_cap` is set, the
/// longer axis is bounded by it (uniformly scaling down).
fn probe_aligned_dims(spec: &Fixture) -> (u32, u32) {
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
    let trimmed = s.trim();
    let parts: Vec<&str> = trimmed.split(',').collect();
    let src_w: u32 = parts[0].parse().expect("source width");
    let src_h: u32 = parts[1].parse().expect("source height");

    let (mut tw, mut th) = (src_w, src_h);
    if let Some(cap) = spec.long_side_cap {
        let long = src_w.max(src_h) as f64;
        if (long as u32) > cap {
            let scale = cap as f64 / long;
            tw = (src_w as f64 * scale) as u32;
            th = (src_h as f64 * scale) as u32;
        }
    }
    // Round DOWN to nearest multiple of 16 (H.264 MB align).
    let aligned_w = (tw / 16) * 16;
    let aligned_h = (th / 16) * 16;
    (aligned_w.max(16), aligned_h.max(16))
}

fn ensure_yuv(spec: &Fixture, w: u32, h: u32) -> Vec<u8> {
    let yuv_path = format!(
        "/tmp/phasm_oh264_corpus_{}_{}x{}_f{}.yuv",
        spec.name, w, h, spec.n_frames
    );
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * (spec.n_frames as usize);
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need {
            return data;
        }
    }
    let src = corpus_root().join(spec.source_mp4);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!("scale={}:{}", w, h);
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &spec.n_frames.to_string()])
        .args(["-an", "-vf", &vf])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("ffmpeg launch");
    assert!(status.success(), "ffmpeg scale failed for {}", spec.source_mp4);
    std::fs::read(&yuv_path).expect("read yuv")
}

/// Wrap a stego Annex-B stream into an mp4 via ffmpeg copy-mux.
/// Writes to ~/Desktop/phasm_oh264_<name>_stego.mp4.
fn mux_to_desktop(stego_bytes: &[u8], name: &str) -> PathBuf {
    let h264_path = std::env::temp_dir().join(format!("phasm_oh264_demo_{}.h264", name));
    let mp4_path = desktop_dir().join(format!("phasm_oh264_{}_stego.mp4", name));
    std::fs::write(&h264_path, stego_bytes).expect("write annex-b");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error",
               "-f", "h264", "-r", "30",
               "-i"])
        .arg(&h264_path)
        .args(["-c:v", "copy"])
        .arg(&mp4_path)
        .status()
        .expect("ffmpeg mux launch");
    assert!(status.success(), "ffmpeg mux failed for {}", name);
    let _ = std::fs::remove_file(&h264_path);
    mp4_path
}

fn run_roundtrip(spec: &Fixture, msg: &str, pass: &str, render_demo: bool) {
    let _g = session_guard().lock().unwrap();
    let (w, h) = probe_aligned_dims(spec);
    let yuv = ensure_yuv(spec, w, h);
    let opts = EncodeOpts { qp: spec.qp, intra_period: 60 };

    let t_enc = std::time::Instant::now();
    let stego = openh264_stego_encode_yuv_text(
        &yuv, w, h, spec.n_frames, opts, msg, pass,
    )
    .expect("oh264 stego encode");
    let enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

    let t_dec = std::time::Instant::now();
    let recovered =
        openh264_stego_decode_yuv_string(&stego, pass).expect("oh264 stego decode");
    let dec_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(recovered, msg, "round-trip message mismatch for {}", spec.name);

    let kb = stego.len() as f64 / 1024.0;
    eprintln!(
        "corpus_oh264 {:18} {}×{}×{} qp={}: encode={:>7.1}ms decode={:>6.1}ms stego={:>7.1} KB",
        spec.name, w, h, spec.n_frames, spec.qp, enc_ms, dec_ms, kb
    );

    if render_demo {
        let mp4 = mux_to_desktop(&stego, spec.name);
        eprintln!("  demo → {}", mp4.display());
    }
}

// ----------------------------- smoke ---------------------------------

/// v1.0 ship gate: smoke round-trip on a single corpus fixture.
/// Runs in default test suite.
#[test]
fn corpus_oh264_smoke_iphone7() {
    let spec = Fixture {
        name: "iphone7_smoke",
        source_mp4: "IMG_4138.MOV",
        long_side_cap: Some(640),
        n_frames: 8,
        qp: 22,
    };
    run_roundtrip(&spec, "phasm c1 smoke", "smoke-pass", /*demo=*/ false);
}

// ----------------------- production fixtures -------------------------

const PROD_LONG: u32 = 1920;
const PROD_N: u32 = 30;
const PROD_QP: i32 = 26;

#[test]
#[ignore]
fn corpus_oh264_iphone7_1080p() {
    run_roundtrip(
        &Fixture {
            name: "iphone7_1080p",
            source_mp4: "IMG_4138.MOV",
            long_side_cap: Some(PROD_LONG), n_frames: PROD_N, qp: PROD_QP,
        },
        "phasm OH264 demo — IMG_4138 iPhone 7 1080p",
        "demo-pass",
        true,
    );
}

#[test]
#[ignore]
fn corpus_oh264_carplane_1080p() {
    run_roundtrip(
        &Fixture {
            name: "carplane_1080p",
            source_mp4: "Artlist_CarPlane.mp4",
            long_side_cap: Some(PROD_LONG), n_frames: PROD_N, qp: PROD_QP,
        },
        "phasm OH264 demo — CarPlane fast-motion 1080p",
        "demo-pass",
        true,
    );
}

#[test]
#[ignore]
fn corpus_oh264_schoolfight_1080p() {
    run_roundtrip(
        &Fixture {
            name: "schoolfight_1080p",
            source_mp4: "Artlist_SchoolFight.mp4",
            long_side_cap: Some(PROD_LONG), n_frames: PROD_N, qp: PROD_QP,
        },
        "phasm OH264 demo — SchoolFight textured 1080p",
        "demo-pass",
        true,
    );
}

#[test]
#[ignore]
fn corpus_oh264_asia_bottle_1080p() {
    run_roundtrip(
        &Fixture {
            name: "asia_bottle_1080p",
            source_mp4: "Artlist_AsiaBottle.mp4",
            long_side_cap: Some(PROD_LONG), n_frames: PROD_N, qp: PROD_QP,
        },
        "phasm OH264 demo — AsiaBottle low-motion 1080p",
        "demo-pass",
        true,
    );
}

#[test]
#[ignore]
fn corpus_oh264_dji_mini2_1080p() {
    run_roundtrip(
        &Fixture {
            name: "dji_mini2_1080p",
            source_mp4: "dji_mini2_2_7k_24fps_h264_high.mp4",
            long_side_cap: Some(PROD_LONG), n_frames: PROD_N, qp: PROD_QP,
        },
        "phasm OH264 demo — DJI Mini2 aerial 1080p",
        "demo-pass",
        true,
    );
}
