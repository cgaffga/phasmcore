// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.8.13 (#446) — production orchestrator integration tests.
//
// Video-retirement Phase 6 (2026-06): migrated off the retired
// single-domain one-shot `openh264_stego_*_yuv_*` helpers onto the
// production 4-domain streaming session (`common::oh264_stream`).
//
// Two file-attachment round-trips, deliberately both:
//   - `c813_roundtrip_with_file_attachment` — the raw Annex-B session
//     (`StreamingDecodeSession::finish`), which the streaming decode FFI
//     reaches directly.
//   - `d15_bridge_path_roundtrip_with_file_attachment` — the whole-FILE
//     path (`mux → h264_ghost_decode`) behind `phasm_h264_decode` /
//     `NativeBridge.h264Decode`. That layer dropped `files` on the floor
//     until 2026-10-20 even though the session underneath had them, so it
//     needs a gate of its own.
//
// Each round-trip test runs independently with a per-binary `SESSION_GUARD`
// mutex so the single-instance OH264 session doesn't conflict with other
// openh264 integration suites.

#![cfg(feature = "h264-encoder")]

mod common;
use common::oh264_stream;

use phasm_core::stego::payload::FileEntry;
// Session guard + the cached scale-to-I420 are shared in `common::video_fixture`.
use common::video_fixture::{ensure_scaled_yuv, session_guard, ScaleRecipe};

fn ensure_yuv(name: &str, source_mp4: &str, w: u32, h: u32, n_frames: u32) -> Vec<u8> {
    ensure_scaled_yuv(source_mp4, "phasm_c813", name, w, h, n_frames, ScaleRecipe::PixFmt)
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

/// D15 (2026-10-20) — file attachments must survive the **whole-file**
/// decode surface the mobile bridges use, not just the raw Annex-B
/// session.
///
/// `phasm_h264_decode` (iOS) and `NativeBridge.h264Decode` (Android) both
/// call `h264_ghost_decode_path` → `h264_ghost_decode`, which demuxes the
/// MP4 and runs `StreamingDecodeSession` internally. That internal hop
/// used to rebuild its `PayloadData` with `files: Vec::new()`, so a video
/// that demonstrably carried an attachment decoded as text-only on both
/// phones — the defect the user hit. This test drives the exact chain:
///
///   `create_with_files` → Annex-B → HandBrake-profile MP4 mux →
///   `h264_ghost_decode`
///
/// and asserts the attachment comes back byte-identical.
#[test]
fn d15_bridge_path_roundtrip_with_file_attachment() {
    use phasm_core::codec::mp4::build::{build_mp4, FrameTiming, MuxerProfile};

    let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());
    let (w, h, n_frames) = (640u32, 368u32, 8u32);
    let yuv = ensure_yuv("iphone7_smoke", "IMG_4138.MOV", w, h, n_frames);

    // Non-UTF-8 content on purpose: a real attachment is binary, and the
    // text-only surface would have silently lossy-converted it.
    let files = vec![FileEntry {
        filename: "attachment.bin".into(),
        content: (0u8..=255u8).cycle().take(1024).collect(),
    }];
    let msg = "D15 — files must reach the phone";
    let pass = "d15-bridge-path";

    let annex_b = oh264_stream::encode_with_files(&yuv, w, h, n_frames, 22, msg, &files, pass)
        .expect("encode_with_files");

    // Mux exactly as the bridges do (`phasm_h264_mux_annexb_to_mp4` /
    // `NativeBridge.h264MuxAnnexbToMp4` → HandbrakeX264 profile).
    let mp4 = build_mp4(
        MuxerProfile::HandbrakeX264,
        &annex_b,
        w,
        h,
        FrameTiming { fps_num: 30, fps_den: 1 },
    )
    .expect("mux annex-b → mp4");

    // The bridge-facing entry point.
    let payload = phasm_core::stego::video::h264_ghost_decode(&mp4, pass)
        .expect("h264_ghost_decode (the phasm_h264_decode / h264Decode entry)");

    assert_eq!(payload.text, msg);
    assert_eq!(
        payload.files.len(),
        1,
        "whole-file decode dropped the attachment (D15 regression)"
    );
    assert_eq!(payload.files[0].filename, "attachment.bin");
    assert_eq!(payload.files[0].content, files[0].content);
    eprintln!(
        "d15 bridge-path: text + {} byte attachment recovered from {} byte MP4",
        payload.files[0].content.len(),
        mp4.len()
    );
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
