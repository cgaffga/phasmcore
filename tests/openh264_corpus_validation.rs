// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.1 (#397-#400) — corpus round-trip on OpenH264-backend.
//
// One #[test] per fixture so cargo's parallel runner spreads the
// encode workload across cores. Each fixture:
//   1. Scales corpus MP4 → YUV at largest 16-aligned dim (cached under
//      `$PHASM_TMP`).
//   2. Encodes via OH264 stego pipeline with a passphrase-driven message.
//   3. Decodes back via openh264_stego_decode_yuv_string → assert recovery.
//   4. Muxes the stego annex-B to mp4 via ffmpeg, writing to
//      ~/Desktop/phasm_oh264_<fixture>_stego.mp4 for visual review.
//
// Smoke variant (`corpus_oh264_smoke_iphone7`) runs by default at
// 640×368×8 frames. Production variants are `#[ignore]` (1080p × 30).
//
// Run:
//   cargo test --release --features "h264-encoder" \
//       --test openh264_corpus_validation -- --ignored --nocapture

#![cfg(feature = "h264-encoder")]

mod common;
use common::oh264_stream;

// Fixture/loader scaffolding (session guard, corpus root, ffprobe dims,
// cached scale-to-I420) is shared in `common::video_fixture`. This suite keeps
// its own `phasm_oh264_corpus_*` YUV cache via `ensure_scaled_yuv`.
use common::video_fixture::{
    ensure_scaled_yuv, probe_aligned_dims, session_guard, ScaleRecipe,
    VideoFixture as Fixture,
};
use std::path::PathBuf;

fn desktop_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME env var");
    let mut p = PathBuf::from(home);
    p.push("Desktop");
    p
}

fn ensure_yuv(spec: &Fixture, w: u32, h: u32) -> Vec<u8> {
    ensure_scaled_yuv(
        spec.source_mp4, "phasm_oh264_corpus", spec.name,
        w, h, spec.n_frames, ScaleRecipe::PixFmt,
    )
}

/// Wrap a stego Annex-B stream into an mp4 via ffmpeg copy-mux.
/// Writes to ~/Desktop/phasm_oh264_<name>_stego.mp4.
fn mux_to_desktop(stego_bytes: &[u8], name: &str) -> PathBuf {
    let h264_path = common::scratch_path(&format!("phasm_oh264_demo_{}.h264", name));
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
    let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());
    let (w, h) = probe_aligned_dims(spec);
    let yuv = ensure_yuv(spec, w, h);

    let t_enc = std::time::Instant::now();
    let stego = oh264_stream::encode(&yuv, w, h, spec.n_frames, spec.qp, msg, pass)
        .expect("oh264 stego encode");
    let enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

    let t_dec = std::time::Instant::now();
    let recovered = oh264_stream::decode_text(&stego, pass).expect("oh264 stego decode");
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
