// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! End-to-end integration test for §Stealth.L4.1: phasm's CABAC stego
//! encoder produces an Annex-B byte stream → wrap in HandBrake/x264-class
//! MP4 via `mp4::build::build_mp4` → demux back → confirm structural
//! invariants.
//!
//! This validates that what the real encoder produces is consumable by the
//! new MP4 builder without further preprocessing.

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::mp4::{
    build::{build_mp4, FrameTiming, MuxerProfile},
    demux::demux,
};
use phasm_core::h264_stego_encode_yuv_string;

/// Generate a high-texture synthetic YUV using the same LCG pattern as the
/// in-tree `deterministic_yuv` helper (encode_pixels.rs:3722). Full-range
/// luma values produce plenty of nonzero quantised residuals at QP=26 so
/// the I-frame stego cover has the capacity for short messages. CABAC +
/// High + 8×8 needs 16-aligned dimensions. Returns YUV420p planar bytes.
fn synthetic_yuv(width: u32, height: u32, n_frames: usize) -> Vec<u8> {
    let frame_size = (width * height * 3 / 2) as usize;
    let mut out = Vec::with_capacity(frame_size * n_frames);
    let mut s: u32 = 0x1234_5678;
    for _ in 0..n_frames {
        for _ in 0..frame_size {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            out.push((s >> 16) as u8);
        }
    }
    out
}

#[test]
fn handbrake_mp4_round_trips_phasm_encoder_output() {
    // 32×32, single frame — fastest path through the encoder while
    // exercising the full SPS+PPS+IDR NAL set.
    let width: u32 = 128;
    let height: u32 = 128;
    let n_frames = 4;
    let yuv = synthetic_yuv(width, height, n_frames);

    let annex_b = h264_stego_encode_yuv_string(
        &yuv,
        width,
        height,
        n_frames,
        "x",
        "passphrase",
    )
    .expect("phasm encoder produces Annex-B output");

    assert!(!annex_b.is_empty(), "encoder produced bytes");

    let mp4 = build_mp4(
        MuxerProfile::HandbrakeX264,
        &annex_b,
        width,
        height,
        FrameTiming::FPS_30,
    )
    .expect("HandBrake MP4 build succeeds on real phasm output");

    // Re-demux and confirm structural invariants.
    let parsed = demux(&mp4).expect("demux of HandBrake-built MP4 succeeds");
    let video_idx = parsed.video_track_idx.expect("video track present");
    let track = &parsed.tracks[video_idx];

    assert!(track.is_h264(), "video track is H.264");
    assert_eq!(track.width, width);
    assert_eq!(track.height, height);
    assert_eq!(
        track.samples.len(),
        n_frames,
        "one MP4 sample per access unit"
    );
    // h264_stego_encode_yuv_string emits I-frames only, so every sample
    // is sync.
    for (i, s) in track.samples.iter().enumerate() {
        assert!(s.is_sync, "sample {i} is sync (I-frame)");
    }

    // avcC carries SPS+PPS so the decoder can reconstruct without
    // looking at mdat for parameter sets.
    let avcc = track
        .avcc_data
        .as_ref()
        .expect("avcC present in stsd");
    assert!(!avcc.sps_nalus.is_empty(), "avcC carries at least one SPS");
    assert!(!avcc.pps_nalus.is_empty(), "avcC carries at least one PPS");
    assert_eq!(avcc.length_size_minus1, 3, "4-byte NAL length prefix");

    // §Stealth.L4.2 SEI injection: the file must contain the canonical
    // x264 plaintext banner (in the leading IDR sample's mdat data).
    let banner = b"x264 - core";
    let banner_present = mp4.windows(banner.len()).any(|w| w == banner);
    assert!(
        banner_present,
        "x264 SEI plaintext banner is in the muxed file"
    );

    // §Stealth.L3.1 VUI: the SPS in avcC must parse with VUI present and
    // carry BT.709 colour primaries. (parse_sps doesn't expose VUI
    // fields directly, but a malformed VUI would prevent the SPS from
    // parsing at all.)
    use phasm_core::codec::h264::bitstream::remove_emulation_prevention;
    use phasm_core::codec::h264::sps::parse_sps;
    let sps_nal = &avcc.sps_nalus[0];
    // avcC stores SPS bytes including the 1-byte NAL header AND any
    // emulation-prevention 0x03 bytes inserted by the encoder. parse_sps
    // wants pure RBSP — strip header, strip EP.
    let sps_rbsp = remove_emulation_prevention(&sps_nal[1..]);
    let sps = parse_sps(&sps_rbsp).expect("avcC SPS parses cleanly");
    // §Stealth.L3.1 follow-on (#145): stego encoder default flipped to
    // High profile (CABAC + transform_8x8). avcC carries that SPS
    // verbatim, so the muxed file's SPS must report profile_idc=100.
    assert_eq!(
        sps.profile_idc, 100,
        "phasm stego encoder targets High profile (libx264-medium centroid)",
    );
}

#[test]
fn handbrake_mp4_strips_aud_sps_pps_from_per_frame_samples() {
    let width: u32 = 128;
    let height: u32 = 128;
    let n_frames = 4;
    let yuv = synthetic_yuv(width, height, n_frames);
    let annex_b = h264_stego_encode_yuv_string(
        &yuv, width, height, n_frames, "x", "p",
    )
    .expect("encode");
    let mp4 = build_mp4(
        MuxerProfile::HandbrakeX264,
        &annex_b,
        width,
        height,
        FrameTiming::FPS_30,
    )
    .expect("build");

    let parsed = demux(&mp4).expect("demux");
    let track = &parsed.tracks[parsed.video_track_idx.unwrap()];
    let sample = &track.samples[0];

    // Walk the length-prefixed NAL stream in this sample and confirm
    // no AUD (type 9), no SPS (type 7), no PPS (type 8) NALs remain.
    let bytes = &sample.data;
    let mut pos = 0;
    while pos + 4 <= bytes.len() {
        let len = u32::from_be_bytes([
            bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3],
        ]) as usize;
        pos += 4;
        assert!(pos + len <= bytes.len(), "NAL length within sample bounds");
        let nal_type = bytes[pos] & 0x1F;
        assert_ne!(nal_type, 7, "SPS must NOT appear in mdat sample");
        assert_ne!(nal_type, 8, "PPS must NOT appear in mdat sample");
        assert_ne!(nal_type, 9, "AUD must NOT appear in mdat sample");
        pos += len;
    }
    assert_eq!(pos, bytes.len(), "sample length is exact sum of NAL chunks");
}
