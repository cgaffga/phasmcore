// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.8.13 (#446) — production orchestrator integration tests.
//
// Video-retirement Phase 6 (2026-06): migrated off the retired
// single-domain one-shot `openh264_stego_*_yuv_*` helpers onto the
// production 4-domain streaming session (`common::oh264_stream`). The
// file-attachment round-trip goes through the production decode
// (mux → `h264_ghost_decode`) because `StreamingDecodeSession` is
// text-only.
//
// Each round-trip test runs independently with a per-binary `SESSION_GUARD`
// mutex so the single-instance OH264 session doesn't conflict with other
// openh264 integration suites.

#![cfg(feature = "h264-encoder")]

mod common;
use common::oh264_stream;

use phasm_core::stego::payload::FileEntry;
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

fn ensure_yuv(name: &str, source_mp4: &str, w: u32, h: u32, n_frames: u32) -> Vec<u8> {
    let yuv_path = format!("/tmp/phasm_c813_{}_{}x{}_f{}.yuv", name, w, h, n_frames);
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * (n_frames as usize);
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need {
            return data;
        }
    }
    let src = corpus_root().join(source_mp4);
    if !src.exists() {
        panic!("corpus fixture missing: {}", src.display());
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
        .expect("ffmpeg");
    assert!(status.success(), "ffmpeg scale failed for {}", source_mp4);
    std::fs::read(&yuv_path).expect("read yuv")
}

/// v1.0 ship gate: the streaming capacity probe returns a non-trivial
/// cover estimate on the smoke fixture.
#[test]
fn c813_capacity_smoke() {
    let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());
    let yuv = ensure_yuv("iphone7_smoke", "IMG_4138.MOV", 640, 368, 8);
    let cover_bits = oh264_stream::probe_cover_bits(&yuv, 640, 368, 8, 22, 8).expect("capacity");
    eprintln!("c813 capacity iphone7_smoke 640x368x8 qp=22: cover_bits={cover_bits}");
    assert!(
        cover_bits >= 10_000,
        "iphone7_smoke expected > 10k cover, got {cover_bits}"
    );
}

#[test]
fn c813_roundtrip_text_only_smoke() {
    let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());
    let yuv = ensure_yuv("iphone7_smoke", "IMG_4138.MOV", 640, 368, 8);
    let msg = "hello phasm h264-encoder";
    let pass = "test-passphrase";

    let bytes = oh264_stream::encode(&yuv, 640, 368, 8, 22, msg, pass).expect("encode");
    eprintln!("c813 text-only smoke: {} bytes encoded", bytes.len());

    let recovered = oh264_stream::decode_text(&bytes, pass).expect("decode");
    assert_eq!(recovered, msg);
}

#[test]
fn c813_roundtrip_wrong_passphrase_fails() {
    let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());
    let yuv = ensure_yuv("iphone7_smoke", "IMG_4138.MOV", 640, 368, 8);
    let bytes = oh264_stream::encode(&yuv, 640, 368, 8, 22, "secret", "right").expect("encode");

    let res = oh264_stream::decode_text(&bytes, "wrong");
    assert!(res.is_err(), "decode with wrong passphrase must fail");
}

#[test]
fn c813_roundtrip_with_file_attachment() {
    let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());
    let yuv = ensure_yuv("iphone7_smoke", "IMG_4138.MOV", 640, 368, 8);
    let files = vec![FileEntry {
        filename: "note.txt".into(),
        content: b"file attachment via h264-encoder C.8.13".to_vec(),
    }];
    let pass = "with-files";
    let msg = "cover text";

    let bytes =
        oh264_stream::encode_with_files(&yuv, 640, 368, 8, 22, msg, &files, pass).expect("encode");
    eprintln!("c813 with-file: {} bytes encoded", bytes.len());

    let payload = oh264_stream::decode_full(&bytes, pass).expect("decode");
    assert_eq!(payload.text, msg);
    assert_eq!(payload.files.len(), 1);
    assert_eq!(payload.files[0].filename, "note.txt");
    assert_eq!(payload.files[0].content, files[0].content);
}

#[test]
fn c813_roundtrip_iphone7_1080p() {
    let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());
    let yuv = ensure_yuv("iphone7_landscape", "IMG_4138.MOV", 1920, 1072, 12);
    let msg = "phasm 1080p iphone7 stego round-trip";
    let pass = "production-pass";

    let bytes = oh264_stream::encode(&yuv, 1920, 1072, 12, 26, msg, pass).expect("encode");
    let recovered = oh264_stream::decode_text(&bytes, pass).expect("decode");
    assert_eq!(recovered, msg);
    eprintln!("c813 iphone7_1080p: round-trip green ({} stego bytes)", bytes.len());
}
