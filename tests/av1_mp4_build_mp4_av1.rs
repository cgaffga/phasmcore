// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! VP.M.3 — integration tests for `build_mp4_av1`.
//!
//! Synthesize an AV1 OBU byte stream → run `build_mp4_av1` → verify
//! the output is a well-formed MP4 with the expected structure
//! (ftyp + mdat + moov + av01 sample entry + av1C config).

#![cfg(feature = "video")]

use phasm_core::codec::mp4::{
    build::{build_mp4_av1, FrameTiming},
    is_mp4,
};

/// Build a synthetic AV1 OBU with a 1-byte header (no extension,
/// has_size=1) followed by a ULEB128 length and `payload_len` payload
/// bytes (all zeros).
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

/// Walk `bytes` as a sequence of MP4 boxes at the top level, returning
/// `(fourcc, content_range)` for each. Box header layout:
/// `size:u32 BE + fourcc[4] + (size:u64 BE if size==1) + content`.
fn list_top_level_boxes(bytes: &[u8]) -> Vec<([u8; 4], (usize, usize))> {
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
        let (content_start, total_len) = if size32 == 1 {
            // 64-bit extended size.
            if cursor + 16 > bytes.len() {
                break;
            }
            let size64 = u64::from_be_bytes([
                bytes[cursor + 8],
                bytes[cursor + 9],
                bytes[cursor + 10],
                bytes[cursor + 11],
                bytes[cursor + 12],
                bytes[cursor + 13],
                bytes[cursor + 14],
                bytes[cursor + 15],
            ]);
            (cursor + 16, size64 as usize)
        } else if size32 == 0 {
            // Size 0 = "extends to end of file".
            (cursor + 8, bytes.len() - cursor)
        } else {
            (cursor + 8, size32 as usize)
        };
        let absolute_end = cursor + total_len;
        if absolute_end > bytes.len() {
            break;
        }
        out.push((fourcc, (content_start, absolute_end)));
        cursor = absolute_end;
    }
    out
}

#[test]
fn build_mp4_av1_single_keyframe_produces_valid_mp4() {
    let mut stream = Vec::new();
    stream.extend_from_slice(&make_obu(OBU_SEQUENCE_HEADER, 12));
    stream.extend_from_slice(&make_obu(OBU_FRAME, 200));

    let timing = FrameTiming {
        fps_num: 30,
        fps_den: 1,
    };
    let mp4 = build_mp4_av1(&stream, 1920, 1080, timing).expect("build_mp4_av1");

    assert!(is_mp4(&mp4), "output must be detected as MP4");

    let boxes = list_top_level_boxes(&mp4);
    let fourccs: Vec<&[u8]> = boxes.iter().map(|(fc, _)| fc.as_slice()).collect();
    assert!(fourccs.contains(&b"ftyp".as_slice()), "ftyp box must be present");
    assert!(fourccs.contains(&b"mdat".as_slice()), "mdat box must be present");
    assert!(fourccs.contains(&b"moov".as_slice()), "moov box must be present");

    // Box order: ftyp → mdat → moov (HandBrake convention).
    assert_eq!(fourccs[0], b"ftyp");
    assert_eq!(fourccs[1], b"mdat");
    assert_eq!(fourccs[2], b"moov");
}

#[test]
fn build_mp4_av1_3_frame_gop_video_payload_sized_correctly() {
    let sh = make_obu(OBU_SEQUENCE_HEADER, 8);
    // 3 frames, varied sizes for distinct sample sizes.
    let f1 = make_obu(OBU_FRAME, 100);
    let f2 = make_obu(OBU_FRAME, 150);
    let f3 = make_obu(OBU_FRAME, 200);
    let mut stream = Vec::new();
    stream.extend_from_slice(&sh);
    stream.extend_from_slice(&f1);
    stream.extend_from_slice(&f2);
    stream.extend_from_slice(&f3);

    let timing = FrameTiming {
        fps_num: 30,
        fps_den: 1,
    };
    let mp4 = build_mp4_av1(&stream, 1280, 720, timing).expect("build_mp4_av1");

    let boxes = list_top_level_boxes(&mp4);
    let (_, mdat_range) = boxes
        .iter()
        .find(|(fc, _)| fc == b"mdat")
        .expect("mdat box present");

    // muxer-sh-in-band-fix: the first sample is sync → SH bytes are
    // PREPENDED to f1's bytes in-band (av1C still carries a redundant
    // copy too — the av1C bytes live in moov, not mdat, so they don't
    // affect the mdat payload size). mdat payload = sh ‖ f1 ‖ f2 ‖ f3.
    let mdat_payload_len = mdat_range.1 - mdat_range.0;
    let expected = sh.len() + f1.len() + f2.len() + f3.len();
    assert_eq!(
        mdat_payload_len, expected,
        "mdat payload must equal SH ‖ f1 ‖ f2 ‖ f3 (SH in-band on the sync sample)"
    );
}

#[test]
fn build_mp4_av1_rejects_empty_stream() {
    let timing = FrameTiming {
        fps_num: 30,
        fps_den: 1,
    };
    let result = build_mp4_av1(&[], 1920, 1080, timing);
    assert!(result.is_err(), "empty stream must be rejected");
}

#[test]
fn build_mp4_av1_rejects_no_sequence_header() {
    // FRAME without preceding SH.
    let stream = make_obu(OBU_FRAME, 100);
    let timing = FrameTiming {
        fps_num: 30,
        fps_den: 1,
    };
    let result = build_mp4_av1(&stream, 1920, 1080, timing);
    assert!(
        result.is_err(),
        "stream without sequence_header_obu must be rejected"
    );
}

#[test]
fn build_mp4_av1_rejects_zero_dimensions() {
    let mut stream = Vec::new();
    stream.extend_from_slice(&make_obu(OBU_SEQUENCE_HEADER, 8));
    stream.extend_from_slice(&make_obu(OBU_FRAME, 100));
    let timing = FrameTiming {
        fps_num: 30,
        fps_den: 1,
    };
    assert!(build_mp4_av1(&stream, 0, 1080, timing).is_err());
    assert!(build_mp4_av1(&stream, 1920, 0, timing).is_err());
}

#[test]
fn build_mp4_av1_av01_sample_entry_appears_in_moov() {
    let mut stream = Vec::new();
    stream.extend_from_slice(&make_obu(OBU_SEQUENCE_HEADER, 8));
    stream.extend_from_slice(&make_obu(OBU_FRAME, 100));

    let timing = FrameTiming {
        fps_num: 30,
        fps_den: 1,
    };
    let mp4 = build_mp4_av1(&stream, 640, 480, timing).expect("build_mp4_av1");

    // The 'av01' fourcc should appear inside the moov box (in the
    // stbl/stsd entry).
    let mut found_av01 = false;
    for window in mp4.windows(4) {
        if window == b"av01" {
            found_av01 = true;
            break;
        }
    }
    assert!(found_av01, "av01 sample entry fourcc must appear in mp4 output");

    // Same for av1C config box.
    let mut found_av1c = false;
    for window in mp4.windows(4) {
        if window == b"av1C" {
            found_av1c = true;
            break;
        }
    }
    assert!(found_av1c, "av1C config box fourcc must appear in mp4 output");
}

#[test]
fn build_mp4_av1_multi_gop_stss_records_keyframes() {
    // 2 GOPs of 3 frames each: SH F F F SH F F F → 6 samples, 2 sync.
    let mut stream = Vec::new();
    stream.extend_from_slice(&make_obu(OBU_SEQUENCE_HEADER, 8));
    for _ in 0..3 {
        stream.extend_from_slice(&make_obu(OBU_FRAME, 50));
    }
    stream.extend_from_slice(&make_obu(OBU_SEQUENCE_HEADER, 8));
    for _ in 0..3 {
        stream.extend_from_slice(&make_obu(OBU_FRAME, 60));
    }

    let timing = FrameTiming {
        fps_num: 30,
        fps_den: 1,
    };
    let mp4 = build_mp4_av1(&stream, 256, 144, timing).expect("build_mp4_av1");

    // The mp4 must contain an stss box.
    let mut found_stss = false;
    for window in mp4.windows(4) {
        if window == b"stss" {
            found_stss = true;
            break;
        }
    }
    assert!(found_stss, "stss sync-sample box must appear");
}
