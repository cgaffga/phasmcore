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
    // muxer-sh-in-band-fix: sync samples now carry the SH bytes in-band
    // (av1C is redundant — both are populated). The single keyframe
    // sample = SH bytes followed by frame_obu bytes.
    let mut expected = Vec::new();
    expected.extend_from_slice(&sh);
    expected.extend_from_slice(&frame);
    assert_eq!(
        result.samples[0], expected,
        "sync sample bytes = SH ‖ frame_obu (in-band SH)"
    );
    assert_eq!(result.sync, vec![true], "first frame after SH is a keyframe");
}

#[test]
fn multi_frame_ipppp_gop_one_sh_one_keyframe() {
    // SH + 5 frames: I P P P P. Only first frame is sync (and carries
    // the SH in-band per muxer-sh-in-band-fix); the other 4 carry only
    // their frame_obu bytes.
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
    // Sample 0 is sync → SH-prepended; samples 1..4 are unchanged.
    let mut sync_expected = Vec::new();
    sync_expected.extend_from_slice(&sh);
    sync_expected.extend_from_slice(&frames[0]);
    assert_eq!(result.samples[0], sync_expected, "sync sample carries SH in-band");
    for i in 1..5 {
        assert_eq!(&result.samples[i], &frames[i], "non-sync sample {i} bytes unchanged");
    }
}

#[test]
fn multi_gop_each_starts_with_sh_marks_keyframe() {
    // SH + frame + SH + frame + frame → 3 samples, sync = [true, true, false].
    // muxer-sh-in-band-fix: BOTH sync samples carry their respective SH
    // bytes in-band (every IDR re-emits the SH for seekability), but
    // av1C still carries only the FIRST one.
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
    // Sample 0: SH1 ‖ f1 (sync). Sample 1: SH2 ‖ f2 (sync, mid-clip
    // SH ensures decoders can seek here without prior state). Sample
    // 2: f3 (no SH).
    let mut s0 = Vec::new();
    s0.extend_from_slice(&sh1);
    s0.extend_from_slice(&f1);
    assert_eq!(result.samples[0], s0);
    let mut s1 = Vec::new();
    s1.extend_from_slice(&sh2);
    s1.extend_from_slice(&f2);
    assert_eq!(result.samples[1], s1);
    assert_eq!(result.samples[2], f3);
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
    // muxer-sh-in-band-fix: TDs still stripped, but SH is in-band.
    let mut expected = Vec::new();
    expected.extend_from_slice(&sh);
    expected.extend_from_slice(&f);
    assert_eq!(
        result.samples[0], expected,
        "sample bytes = SH ‖ frame_obu; TDs are dropped"
    );
    assert_eq!(result.sync, vec![true]);
}

#[test]
fn frame_header_plus_tile_group_combine_into_one_sample() {
    // SH + FH + TG → 1 sample = SH ‖ FH ‖ TG (separate frame+tile mode,
    // SH in-band per muxer-sh-in-band-fix).
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
    expected.extend_from_slice(&sh);
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
    // to first sample (the "current sample" at the time MD was seen).
    // muxer-sh-in-band-fix: sample 0 also carries the SH in-band.
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
    // Sample 0 (sync): SH ‖ f1 ‖ md. Sample 1: f2.
    let mut expected_first = Vec::new();
    expected_first.extend_from_slice(&sh);
    expected_first.extend_from_slice(&f1);
    expected_first.extend_from_slice(&md);
    assert_eq!(
        result.samples[0], expected_first,
        "sync sample = SH in-band ‖ frame_obu ‖ metadata_obu"
    );
    assert_eq!(result.samples[1], f2);
}
