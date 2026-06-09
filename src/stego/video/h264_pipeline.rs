// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 video Ghost decode dispatch.
//!
//! Production H.264 video stego is OpenH264-encode + pure-Rust-walker decode
//! (`StreamingDecodeSession`). This module is the thin decode entry point that
//! the mobile bridges (`phasm_h264_decode` / `h264Decode`, via
//! [`h264_ghost_decode_path`]) and the CLI reach: it demuxes the MP4, validates
//! the avcC SPS/PPS, and routes CABAC (High-profile) input to the streaming
//! walker.
//!
//! The legacy #77 CAVLC bitstream-mod stego subsystem (the in-place trailing-one
//! sign-bit encoder + its whole-video CAVLC decoder, plus the pre-streaming
//! CABAC-v2 fallback) was retired in the video-stack retirement — production
//! never produced or consumed it. See `docs/design/video/_RETIREMENT-PLAN.md`
//! § "Phase 4".

use crate::codec::h264::bitstream::{self};
use crate::codec::h264::sps::{self, Pps, Sps};
use crate::codec::mp4;
use crate::stego::error::StegoError;

/// Decode a message from an H.264 stego MP4.
///
/// Demuxes + validates the container, then routes CABAC (High-profile) input —
/// the only format production emits — to the streaming walker
/// (`StreamingDecodeSession`). Returns `PayloadData` (text + any file
/// attachments) for parity with the rest of the decode surface.
///
/// Requires the `h264-decoder` feature; without it (the no-walker container-only
/// `video` build) this returns an `InvalidVideo` error.
pub fn h264_ghost_decode(
    mp4_bytes: &[u8],
    passphrase: &str,
) -> Result<crate::stego::payload::PayloadData, StegoError> {
    // 1. Demux and validate.
    let mp4_file = mp4::demux::demux(mp4_bytes)?;
    let (_sps, pps, length_size) = extract_h264_params(&mp4_file)?;

    if pps.entropy_coding_mode_flag {
        // CABAC (High profile) — the production wire format. Route to the
        // streaming walker. CLI's `decode_h264_cabac` reaches the same
        // walker directly; mobile bridges (iOS + Android) reach this
        // function via `h264_ghost_decode_path`.
        #[cfg(feature = "h264-decoder")]
        {
            if let Some(payload) =
                try_streaming_decode_mp4(&mp4_file, length_size, passphrase)
            {
                return Ok(payload);
            }
            return Err(StegoError::DecryptionFailed);
        }
        #[cfg(not(feature = "h264-decoder"))]
        {
            let _ = length_size;
            return Err(StegoError::InvalidVideo(
                "H.264 CABAC not supported for decode \
                 (build with --features h264-decoder to enable)"
                    .into(),
            ));
        }
    }

    // Non-CABAC (Baseline CAVLC) input. The legacy CAVLC stego decoder was
    // retired — production never emits CAVLC stego — so there is nothing to
    // recover here.
    Err(StegoError::InvalidVideo(
        "H.264 Baseline CAVLC stego decode is retired; \
         expected a CABAC (High-profile) Phasm stego video"
            .into(),
    ))
}

/// Decode a message from a stego H.264 MP4 at `path`. Streaming: the file
/// is mmap'd read-only so large videos never fully load into heap.
pub fn h264_ghost_decode_path(
    path: &std::path::Path,
    passphrase: &str,
) -> Result<crate::stego::payload::PayloadData, StegoError> {
    let file = std::fs::File::open(path)
        .map_err(|e| StegoError::InvalidVideo(format!("open failed: {e}")))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| StegoError::InvalidVideo(format!("mmap failed: {e}")))?;
    h264_ghost_decode(&mmap, passphrase)
}

/// Decode an MP4 via the engine-agnostic streaming walker
/// (`StreamingDecodeSession`). MP4 keeps SPS/PPS in avcC (not inline), so we
/// prepend the SPS+PPS pair before every IDR slice to give the streaming
/// session the per-GOP slabs it expects. Returns `None` on any structural
/// failure or wrong passphrase.
#[cfg(feature = "h264-decoder")]
fn try_streaming_decode_mp4(
    mp4_file: &mp4::Mp4File,
    length_size: u8,
    passphrase: &str,
) -> Option<crate::stego::payload::PayloadData> {
    use crate::codec::h264::streaming_session::StreamingDecodeSession;

    let track_idx = mp4_file.video_track_idx?;
    let track = &mp4_file.tracks[track_idx];
    let avcc = track.avcc_data.as_ref()?;

    let mut annex_b: Vec<u8> = Vec::new();
    let start_code: [u8; 4] = [0, 0, 0, 1];
    let push_params = |buf: &mut Vec<u8>| {
        for sps_bytes in &avcc.sps_nalus {
            if sps_bytes.is_empty() {
                continue;
            }
            buf.extend_from_slice(&start_code);
            buf.extend_from_slice(sps_bytes);
        }
        for pps_bytes in &avcc.pps_nalus {
            if pps_bytes.is_empty() {
                continue;
            }
            buf.extend_from_slice(&start_code);
            buf.extend_from_slice(pps_bytes);
        }
    };
    let ls = length_size as usize;
    for sample in &track.samples {
        let data = &sample.data;
        let mut p = 0usize;
        let mut sample_idr_emitted_params = false;
        while p + ls <= data.len() {
            let mut nal_len = 0usize;
            for i in 0..ls {
                nal_len = (nal_len << 8) | data[p + i] as usize;
            }
            p += ls;
            if nal_len == 0 || p + nal_len > data.len() {
                return None;
            }
            let nal_bytes = &data[p..p + nal_len];
            let nal_type = nal_bytes.first().map(|b| b & 0x1F).unwrap_or(0);
            if nal_type == 5 && !sample_idr_emitted_params {
                push_params(&mut annex_b);
                sample_idr_emitted_params = true;
            }
            annex_b.extend_from_slice(&start_code);
            annex_b.extend_from_slice(nal_bytes);
            p += nal_len;
        }
    }
    if annex_b.is_empty() {
        return None;
    }

    let mut session = StreamingDecodeSession::create(passphrase).ok()?;
    session.push_annex_b(&annex_b).ok()?;
    let result = session.finish().ok()?;
    Some(crate::stego::payload::PayloadData {
        text: result.text,
        files: Vec::new(),
    })
}

/// Parse SPS + PPS out of the MP4 track's avcC configuration record, returning
/// them together with the NAL length-prefix size. Shared validation entry for
/// the decode dispatch.
fn extract_h264_params(mp4_file: &mp4::Mp4File) -> Result<(Sps, Pps, u8), StegoError> {
    let track_idx = mp4_file
        .video_track_idx
        .ok_or(StegoError::InvalidVideo("no video track".into()))?;
    let track = &mp4_file.tracks[track_idx];

    if !track.is_h264() {
        return Err(StegoError::InvalidVideo(format!(
            "video track codec {:?} is not H.264",
            std::str::from_utf8(&track.codec).unwrap_or("????")
        )));
    }

    let avcc = track
        .avcc_data
        .as_ref()
        .ok_or(StegoError::InvalidVideo("no avcC configuration".into()))?;

    let length_size = avcc.length_size_minus1 + 1;

    // Parse SPS from avcC.
    if avcc.sps_nalus.is_empty() {
        return Err(StegoError::InvalidVideo("no SPS in avcC".into()));
    }
    // SPS NAL units in avcC include the NAL header byte.
    let sps_nalu = &avcc.sps_nalus[0];
    let sps_rbsp = if !sps_nalu.is_empty() && (sps_nalu[0] & 0x1F) == 7 {
        // Has NAL header — strip it and remove EP bytes.
        bitstream::remove_emulation_prevention(&sps_nalu[1..])
    } else {
        bitstream::remove_emulation_prevention(sps_nalu)
    };
    let sps_parsed = sps::parse_sps(&sps_rbsp)?;

    // Parse PPS from avcC.
    if avcc.pps_nalus.is_empty() {
        return Err(StegoError::InvalidVideo("no PPS in avcC".into()));
    }
    let pps_nalu = &avcc.pps_nalus[0];
    let pps_rbsp = if !pps_nalu.is_empty() && (pps_nalu[0] & 0x1F) == 8 {
        bitstream::remove_emulation_prevention(&pps_nalu[1..])
    } else {
        bitstream::remove_emulation_prevention(pps_nalu)
    };
    let pps_parsed = sps::parse_pps(&pps_rbsp)?;

    Ok((sps_parsed, pps_parsed, length_size))
}
