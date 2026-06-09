// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! VP.M.2 — integration tests for the AV1 OBU → MP4 sample splitter.
//!
//! Lives in core/tests/ instead of inline `mod tests` so it gets its
//! own compilation unit, bypassing pre-existing h264 stego compile
//! breakage in the lib test harness on this AV1 branch.

#![cfg(feature = "video")]

use phasm_core::codec::mp4::av1_obu_split::{
    split_av1_into_samples, OBU_FRAME, OBU_FRAME_HEADER, OBU_METADATA, OBU_SEQUENCE_HEADER,
    OBU_TEMPORAL_DELIMITER, OBU_TILE_GROUP,
};

/// Build a synthetic AV1 OBU with a 1-byte header (no extension,
/// has_size=1) followed by a ULEB128 length and `payload_len` payload
/// bytes (all zeros). Useful for testing the splitter without
/// depending on rav1e output.
fn make_obu(obu_type: u8, payload_len: usize) -> Vec<u8> {
    let mut out = Vec::new();
    // Header: forbidden=0, type=obu_type (4 bits), ext=0, has_size=1, reserved=0.
    out.push(((obu_type & 0x0F) << 3) | 0b010);
    // Encode payload_len as ULEB128.
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
    // Payload (zeros).
    out.extend(std::iter::repeat(0u8).take(payload_len));
    out
}

#[test]
fn empty_stream_returns_empty_split() {
    let result = split_av1_into_samples(&[]).unwrap();
    assert!(result.sequence_header_obu.is_empty());
    assert!(result.samples.is_empty());
    assert!(result.sync.is_empty());
}

#[test]
fn single_keyframe_sequence_header_plus_frame() {
    let mut stream = Vec::new();
    let sh = make_obu(OBU_SEQUENCE_HEADER, 8);
    let frame = make_obu(OBU_FRAME, 100);
    stream.extend_from_slice(&sh);
    stream.extend_from_slice(&frame);

    let result = split_av1_into_samples(&stream).unwrap();
    assert_eq!(result.sequence_header_obu, sh, "av1C must carry the SH OBU");
    assert_eq!(result.samples.len(), 1, "one frame = one sample");
    assert_eq!(result.samples[0], frame, "sample is the frame_obu bytes");
    assert_eq!(result.sync, vec![true], "first frame after SH is a keyframe");
}

#[test]
fn multi_frame_ipppp_gop_one_sh_one_keyframe() {
    // SH + 5 frames: I P P P P. Only first frame is sync.
    let mut stream = Vec::new();
    let sh = make_obu(OBU_SEQUENCE_HEADER, 12);
    stream.extend_from_slice(&sh);
    let mut frames = Vec::new();
    for i in 0..5 {
        let f = make_obu(OBU_FRAME, 50 + i * 10);
        stream.extend_from_slice(&f);
        frames.push(f);
    }

    let result = split_av1_into_samples(&stream).unwrap();
    assert_eq!(result.sequence_header_obu, sh);
    assert_eq!(result.samples.len(), 5);
    assert_eq!(result.sync, vec![true, false, false, false, false]);
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(&result.samples[i], f, "sample {i} bytes");
    }
}

#[test]
fn multi_gop_each_starts_with_sh_marks_keyframe() {
    // SH + frame + SH + frame + frame → 3 samples, sync = [true, true, false].
    let mut stream = Vec::new();
    let sh1 = make_obu(OBU_SEQUENCE_HEADER, 8);
    let f1 = make_obu(OBU_FRAME, 30);
    let sh2 = make_obu(OBU_SEQUENCE_HEADER, 8);
    let f2 = make_obu(OBU_FRAME, 40);
    let f3 = make_obu(OBU_FRAME, 50);
    stream.extend_from_slice(&sh1);
    stream.extend_from_slice(&f1);
    stream.extend_from_slice(&sh2);
    stream.extend_from_slice(&f2);
    stream.extend_from_slice(&f3);

    let result = split_av1_into_samples(&stream).unwrap();
    // av1C carries only the FIRST sequence_header_obu.
    assert_eq!(result.sequence_header_obu, sh1);
    assert_eq!(result.samples.len(), 3);
    assert_eq!(result.sync, vec![true, true, false]);
}

#[test]
fn temporal_delimiters_stripped() {
    // TD + SH + TD + frame → 1 sample, TDs absent from output.
    let mut stream = Vec::new();
    stream.extend_from_slice(&make_obu(OBU_TEMPORAL_DELIMITER, 0));
    let sh = make_obu(OBU_SEQUENCE_HEADER, 8);
    stream.extend_from_slice(&sh);
    stream.extend_from_slice(&make_obu(OBU_TEMPORAL_DELIMITER, 0));
    let f = make_obu(OBU_FRAME, 20);
    stream.extend_from_slice(&f);

    let result = split_av1_into_samples(&stream).unwrap();
    assert_eq!(result.samples.len(), 1);
    assert_eq!(result.samples[0], f, "sample bytes must NOT include TD OBU");
    assert_eq!(result.sync, vec![true]);
}

#[test]
fn frame_header_plus_tile_group_combine_into_one_sample() {
    // SH + FH + TG → 1 sample = FH + TG (separate frame+tile mode).
    let mut stream = Vec::new();
    let sh = make_obu(OBU_SEQUENCE_HEADER, 8);
    let fh = make_obu(OBU_FRAME_HEADER, 12);
    let tg = make_obu(OBU_TILE_GROUP, 80);
    stream.extend_from_slice(&sh);
    stream.extend_from_slice(&fh);
    stream.extend_from_slice(&tg);

    let result = split_av1_into_samples(&stream).unwrap();
    assert_eq!(result.samples.len(), 1);
    let mut expected = Vec::new();
    expected.extend_from_slice(&fh);
    expected.extend_from_slice(&tg);
    assert_eq!(result.samples[0], expected);
    assert_eq!(result.sync, vec![true]);
}

#[test]
fn truncated_stream_stops_cleanly() {
    // SH + truncated FRAME → splitter stops at the bad OBU.
    let mut stream = Vec::new();
    stream.extend_from_slice(&make_obu(OBU_SEQUENCE_HEADER, 8));
    let f = make_obu(OBU_FRAME, 50);
    stream.extend_from_slice(&f[..f.len() / 2]); // Truncate the FRAME mid-payload.

    let result = split_av1_into_samples(&stream).unwrap();
    // SH parsed cleanly; the truncated FRAME OBU is rejected
    // (insufficient bytes) → no samples emitted. Callers should
    // validate `samples.len() > 0` before muxing.
    assert!(result.sequence_header_obu.len() > 0);
    assert!(result.samples.is_empty());
}

#[test]
fn metadata_obus_attach_to_current_sample() {
    // SH + frame + METADATA + frame → 2 samples; metadata attached
    // to second sample (or first if metadata precedes the frame).
    let mut stream = Vec::new();
    let sh = make_obu(OBU_SEQUENCE_HEADER, 8);
    let f1 = make_obu(OBU_FRAME, 30);
    let md = make_obu(OBU_METADATA, 16);
    let f2 = make_obu(OBU_FRAME, 40);
    stream.extend_from_slice(&sh);
    stream.extend_from_slice(&f1);
    stream.extend_from_slice(&md);
    stream.extend_from_slice(&f2);

    let result = split_av1_into_samples(&stream).unwrap();
    assert_eq!(result.samples.len(), 2);
    assert_eq!(result.sync, vec![true, false]);
    // Metadata sat AFTER f1's sample was closed; it appended to f1
    // (the "current sample" at the time MD was seen). Then f2 starts
    // a new sample.
    let mut expected_first = Vec::new();
    expected_first.extend_from_slice(&f1);
    expected_first.extend_from_slice(&md);
    assert_eq!(result.samples[0], expected_first, "metadata sits with the preceding frame");
    assert_eq!(result.samples[1], f2);
}
