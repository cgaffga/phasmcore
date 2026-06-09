// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! VP.M.4 — `build_mp4_av1_with_audio` integration test.
//!
//! Feeds a synthetic AV1 OBU stream + a real source MP4 carrying an
//! AAC audio track into the muxer and verifies:
//!
//! 1. The output is still a valid MP4 (ftyp + mdat + moov, av01
//!    sample entry present).
//! 2. The output contains an audio track (`soun` handler type, `mp4a`
//!    sample entry).
//! 3. mdat carries both the AV1 video samples AND the audio samples,
//!    so payload size grows when audio is included.
//! 4. The video-only fallback path still works.

#![cfg(feature = "video")]

use std::path::PathBuf;

use phasm_core::codec::mp4::{
    build::{build_mp4_av1, build_mp4_av1_with_audio, FrameTiming},
    is_mp4,
};

/// Path to a real-world source MP4 with an AAC audio track. The Artlist
/// fixture is one of the standard H.264 + AAC test vectors used elsewhere
/// in the suite.
fn aac_source_mp4_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors/video/h264/real-world/source/Artlist_SchoolFight.mp4")
}

/// Build a synthetic AV1 OBU stream with one SH + N FRAME OBUs.
fn make_synthetic_av1_stream(n_frames: usize) -> Vec<u8> {
    fn make_obu(obu_type: u8, payload_len: usize) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(((obu_type & 0x0F) << 3) | 0b010);
        let mut n = payload_len as u64;
        loop {
            let mut byte = (n & 0x7F) as u8;
            n >>= 7;
            if n != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if n == 0 {
                break;
            }
        }
        out.extend(std::iter::repeat(0u8).take(payload_len));
        out
    }
    const OBU_SEQUENCE_HEADER: u8 = 1;
    const OBU_FRAME: u8 = 6;

    let mut stream = Vec::new();
    stream.extend_from_slice(&make_obu(OBU_SEQUENCE_HEADER, 12));
    for _ in 0..n_frames {
        stream.extend_from_slice(&make_obu(OBU_FRAME, 100));
    }
    stream
}

/// Walk `bytes` as top-level MP4 boxes and return their fourCC strings.
fn list_top_level_boxes(bytes: &[u8]) -> Vec<[u8; 4]> {
    let mut out = Vec::new();
    let mut cursor = 0;
    while cursor + 8 <= bytes.len() {
        let size32 = u32::from_be_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]);
        let mut fourcc = [0u8; 4];
        fourcc.copy_from_slice(&bytes[cursor + 4..cursor + 8]);
        let total_len = if size32 == 1 {
            if cursor + 16 > bytes.len() {
                break;
            }
            u64::from_be_bytes([
                bytes[cursor + 8],
                bytes[cursor + 9],
                bytes[cursor + 10],
                bytes[cursor + 11],
                bytes[cursor + 12],
                bytes[cursor + 13],
                bytes[cursor + 14],
                bytes[cursor + 15],
            ]) as usize
        } else if size32 == 0 {
            bytes.len() - cursor
        } else {
            size32 as usize
        };
        let absolute_end = cursor + total_len;
        if absolute_end > bytes.len() {
            break;
        }
        out.push(fourcc);
        cursor = absolute_end;
    }
    out
}

fn contains_fourcc(bytes: &[u8], fourcc: &[u8; 4]) -> bool {
    bytes.windows(4).any(|w| w == fourcc)
}

#[test]
fn build_mp4_av1_with_audio_produces_valid_mp4() {
    let source_path = aac_source_mp4_path();
    let Ok(source) = std::fs::read(&source_path) else {
        eprintln!(
            "skipping: AAC source MP4 missing at {:?}",
            source_path
        );
        return;
    };
    let stream = make_synthetic_av1_stream(3);

    let timing = FrameTiming { fps_num: 30, fps_den: 1 };
    let mp4 = build_mp4_av1_with_audio(&stream, 1920, 1080, timing, &source)
        .expect("build_mp4_av1_with_audio");

    assert!(is_mp4(&mp4), "output must be detected as MP4");

    let fourccs = list_top_level_boxes(&mp4);
    let fourccs_slices: Vec<&[u8]> = fourccs.iter().map(|f| f.as_slice()).collect();
    assert!(fourccs_slices.contains(&b"ftyp".as_slice()), "ftyp present");
    assert!(fourccs_slices.contains(&b"mdat".as_slice()), "mdat present");
    assert!(fourccs_slices.contains(&b"moov".as_slice()), "moov present");
    assert!(
        contains_fourcc(&mp4, b"av01"),
        "av01 video sample entry must appear"
    );
    assert!(
        contains_fourcc(&mp4, b"av1C"),
        "av1C config box must appear"
    );
}

#[test]
fn build_mp4_av1_with_audio_carries_audio_track() {
    let source_path = aac_source_mp4_path();
    let Ok(source) = std::fs::read(&source_path) else {
        eprintln!(
            "skipping: AAC source MP4 missing at {:?}",
            source_path
        );
        return;
    };
    let stream = make_synthetic_av1_stream(5);
    let timing = FrameTiming { fps_num: 30, fps_den: 1 };
    let mp4 = build_mp4_av1_with_audio(&stream, 1280, 720, timing, &source)
        .expect("build_mp4_av1_with_audio");

    // The source carries an AAC audio track, so the output MUST contain
    // both a video handler (vide) AND an audio handler (soun).
    assert!(contains_fourcc(&mp4, b"vide"), "video handler must appear");
    assert!(contains_fourcc(&mp4, b"soun"), "audio handler must appear");
    // AAC audio is wrapped in an `mp4a` sample entry with an `esds` box.
    assert!(
        contains_fourcc(&mp4, b"mp4a"),
        "mp4a audio sample entry must appear (source has AAC)"
    );
    assert!(
        contains_fourcc(&mp4, b"esds"),
        "esds AAC config box must appear"
    );
}

#[test]
fn build_mp4_av1_with_audio_grows_mdat_vs_video_only() {
    let source_path = aac_source_mp4_path();
    let Ok(source) = std::fs::read(&source_path) else {
        eprintln!(
            "skipping: AAC source MP4 missing at {:?}",
            source_path
        );
        return;
    };
    let stream = make_synthetic_av1_stream(10);
    let timing = FrameTiming { fps_num: 30, fps_den: 1 };

    let mp4_video_only = build_mp4_av1(&stream, 1280, 720, timing)
        .expect("video-only build_mp4_av1");
    let mp4_with_audio = build_mp4_av1_with_audio(&stream, 1280, 720, timing, &source)
        .expect("audio build_mp4_av1_with_audio");

    // The audio payload + extra audio trak inside moov must make the
    // audio-bearing mux strictly larger.
    assert!(
        mp4_with_audio.len() > mp4_video_only.len(),
        "audio mux ({}) should be strictly larger than video-only mux ({})",
        mp4_with_audio.len(),
        mp4_video_only.len()
    );
}

#[test]
fn build_mp4_av1_with_audio_video_only_fallback_when_source_has_no_audio() {
    // Use the AV1 video-only output of build_mp4_av1 itself as the
    // "source" — it has no audio track. The result should still be a
    // valid MP4, just video-only (no `soun` handler).
    let stream = make_synthetic_av1_stream(3);
    let timing = FrameTiming { fps_num: 30, fps_den: 1 };
    let source_no_audio = build_mp4_av1(&stream, 640, 480, timing)
        .expect("video-only stub source");

    let mp4 = build_mp4_av1_with_audio(&stream, 640, 480, timing, &source_no_audio)
        .expect("build_mp4_av1_with_audio with audio-less source");
    assert!(is_mp4(&mp4), "output must be MP4");
    assert!(contains_fourcc(&mp4, b"av01"), "still has av01");
    // No audio track in the source, so the output must not have one.
    assert!(
        !contains_fourcc(&mp4, b"soun"),
        "no audio in source → no soun in output"
    );
}
