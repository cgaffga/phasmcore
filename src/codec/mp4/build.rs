// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phasm-owned MP4 builder.
//!
//! **H.264.** Wraps a phasm-encoded Annex-B bitstream in an MP4 container
//! whose atom signature matches a [`MuxerProfile`]. The one profile is
//! [`MuxerProfile::LavfOpenH264`] (module [`lavf`]): the container a
//! libavformat transcoder writes around a libopenh264 encode, consistent
//! with the OpenH264 bitstream inside it (see
//! `docs/design/video/h264/openh264-stealth-report.md`, "Container layer").
//! A HandBrake/x264 profile that put an x264 SEI banner and a HandBrake
//! `©too` around that bitstream (every claim of which the SPS / PPS /
//! slices contradict) was removed on 2026-10-29.
//!
//! **AV1.** [`build_mp4_av1`] and its variants write the `av01` container
//! with the box helpers in this file (`ftyp → mdat → moov`, one chunk per
//! GOP, `mvhd` timescale 1000, `fps × 512` media timescale).
//!
//! Distinct from [`super::mux`], which patches an existing source MP4 by
//! copying its `moov` and substituting modified samples — that path makes
//! sense when the source file is also H.264 (HEVC sources can't be
//! shadow-patched). The builders here write a fresh MP4 from scratch given
//! the target profile and an Annex-B (or OBU) byte stream.

use super::{AvccData, Mp4Error};
use crate::codec::h264::NalType;
#[cfg(feature = "h264-encoder")]
use crate::codec::h264::stego::gop_pattern::{iter_encode_order, FrameType, GopPattern};

// A child of this module (it uses the builder's private helpers), kept
// outside a `build/` directory, which the repository ignores.
#[path = "build_lavf.rs"]
mod lavf;

/// Container target profile: the byte-exact atom signature of one
/// converter-pipeline metaclass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuxerProfile {
    /// The container a libavformat-based transcoder writes around a
    /// libopenh264 encode: consistent with the OpenH264 bitstream inside it.
    /// The CLI default since 2026-10-29. See [`lavf`] for the field-by-field
    /// rationale.
    LavfOpenH264,
}

/// Per-frame timing info supplied by the caller. One entry per access unit
/// in the input Annex-B stream. Required because the Annex-B stream itself
/// has no frame-rate metadata.
#[derive(Debug, Clone, Copy)]
pub struct FrameTiming {
    /// Frame rate numerator (e.g. 30 for 30 fps, 30000 for 29.97).
    pub fps_num: u32,
    /// Frame rate denominator (e.g. 1 for integer fps, 1001 for 29.97).
    pub fps_den: u32,
}

impl FrameTiming {
    /// 30 fps integer (the phasm video-stego default; matches the
    /// H.264 stealth-measurement and CABAC v2 fixtures).
    pub const FPS_30: Self = Self { fps_num: 30, fps_den: 1 };
}

/// Build an MP4 file from a phasm-encoded H.264 Annex-B byte stream and a
/// target [`MuxerProfile`]. Returns the complete MP4 bytes.
///
/// `width` / `height` are the DISPLAY dimensions in pixels: for a stream
/// whose SPS crops the MB grid (1920×1080 coded as 1088 lines) that is the
/// cropped size, as in a phone MP4. A size that disagrees with the SPS is
/// refused (`Mp4Error::InvalidBox`).
///
/// The Annex-B stream must contain at least one SPS NAL and one PPS NAL —
/// they are extracted into the `avcC` decoder configuration record and
/// then dropped from the per-frame mdat samples.
pub fn build_mp4(
    profile: MuxerProfile,
    annex_b: &[u8],
    width: u32,
    height: u32,
    timing: FrameTiming,
) -> Result<Vec<u8>, Mp4Error> {
    if width == 0 || height == 0 {
        return Err(Mp4Error::InvalidBox(format!(
            "invalid dimensions {width}x{height}"
        )));
    }
    if timing.fps_num == 0 || timing.fps_den == 0 {
        return Err(Mp4Error::InvalidBox(format!(
            "invalid frame rate {}/{}",
            timing.fps_num, timing.fps_den
        )));
    }

    let nals = parse_annexb_nals(annex_b);
    if nals.is_empty() {
        return Err(Mp4Error::InvalidBox("no NAL units in input".into()));
    }

    let avcc = AvccData::from_annexb(annex_b)
        .ok_or_else(|| Mp4Error::InvalidBox("missing SPS or PPS in input".into()))?;

    // Group NALs into access units. An access unit ends when the NEXT
    // NAL is an AUD or when the next slice starts a new picture (we use
    // AUD-or-VCL-after-VCL as the boundary — phasm always emits AUD).
    let access_units = group_access_units(&nals);
    if access_units.is_empty() {
        return Err(Mp4Error::InvalidBox("no access units in input".into()));
    }

    match profile {
        MuxerProfile::LavfOpenH264 => {
            lavf::build(annex_b, &access_units, &avcc, width, height, timing, None, None, 0)
        }
    }
}

/// §Stealth.L4.2 follow-on — B-frame-aware variant of [`build_mp4`].
///
/// When the input Annex-B stream is encoded in IBPBP (or any pattern with
/// B-frames), access units appear in encode order — but the decoder must
/// display frames in display order. The standard MP4 mechanism is the
/// `ctts` (composition time-to-sample) box: per-sample CTS-DTS offset.
///
/// `pattern` describes the encode-order pattern (e.g.
/// `GopPattern::iphone_default()` = `Ibpbp{ gop: 30, b_count: 1 }`) and
/// is used to derive each sample's display index. The number of access
/// units in `annex_b` must equal the number of frames the pattern emits
/// for `n_display_frames`.
///
/// For pure-`Ipppp` patterns (no B-frames) this is equivalent to
/// [`build_mp4`] — no `ctts` / `edts/elst` boxes are added.
#[cfg(feature = "h264-encoder")]
pub fn build_mp4_with_pattern(
    profile: MuxerProfile,
    annex_b: &[u8],
    width: u32,
    height: u32,
    timing: FrameTiming,
    pattern: GopPattern,
    n_display_frames: usize,
) -> Result<Vec<u8>, Mp4Error> {
    if width == 0 || height == 0 {
        return Err(Mp4Error::InvalidBox(format!(
            "invalid dimensions {width}x{height}"
        )));
    }
    if timing.fps_num == 0 || timing.fps_den == 0 {
        return Err(Mp4Error::InvalidBox(format!(
            "invalid frame rate {}/{}",
            timing.fps_num, timing.fps_den
        )));
    }

    let nals = parse_annexb_nals(annex_b);
    if nals.is_empty() {
        return Err(Mp4Error::InvalidBox("no NAL units in input".into()));
    }
    let avcc = AvccData::from_annexb(annex_b)
        .ok_or_else(|| Mp4Error::InvalidBox("missing SPS or PPS in input".into()))?;
    let access_units = group_access_units(&nals);
    if access_units.is_empty() {
        return Err(Mp4Error::InvalidBox("no access units in input".into()));
    }

    // Build encode-order metadata for each sample.
    let encode_order: Vec<_> = iter_encode_order(n_display_frames, pattern).collect();
    if encode_order.len() != access_units.len() {
        return Err(Mp4Error::InvalidBox(format!(
            "pattern emits {} frames but input has {} access units",
            encode_order.len(),
            access_units.len(),
        )));
    }

    // Per-sample composition-time offset = display_idx - encode_idx, in
    // STTS_DELTA_PER_FRAME units. Same for IPPPP (always 0).
    let any_b_frame = encode_order.iter().any(|f| f.frame_type == FrameType::B);
    let composition_offsets: Option<Vec<i32>> = if any_b_frame {
        Some(
            encode_order
                .iter()
                .map(|f| {
                    let display = f.display_idx as i32;
                    let encode = f.encode_idx as i32;
                    (display - encode) * (shared::STTS_DELTA_PER_FRAME as i32)
                })
                .collect(),
        )
    } else {
        None
    };

    match profile {
        MuxerProfile::LavfOpenH264 => lavf::build(
            annex_b,
            &access_units,
            &avcc,
            width,
            height,
            timing,
            composition_offsets.as_deref(),
            None,
            0,
        ),
    }
}

/// §Stealth.L4.5 — variant of [`build_mp4_with_pattern`] that also
/// passes through an audio track from the source MP4. The video container
/// shape is unchanged; the audio is added as a second track inside `moov`
/// with its samples appended to `mdat` after the video samples.
///
/// # Audio handling
///
/// The source's first audio track is copied. The codec configuration
/// record (mp4a/esds for AAC, dops for Opus, etc.) passes through
/// unchanged inside the source's `stsd` box. Sample timing
/// (`stts`/`stsc`/`stsz`) is preserved; only `stco`/`co64` chunk offsets
/// are rewritten to point at the new MP4's mdat positions. Track IDs are
/// renumbered so video stays at `track_id=1` and audio takes
/// `track_id=2`. The track header fields that describe the source file
/// rather than the audio — creation / modification times, the handler's
/// name, data references, user data and metadata boxes — are cleared
/// (`audio_iso`); a track that cannot be cleared fails the mux.
///
/// `source_mp4` is the original MP4 byte stream (or any MP4 carrying
/// the desired audio track — typically the same input the YUV came
/// from). When the source has no audio track, the result is identical
/// to [`build_mp4_with_pattern`].
#[cfg(feature = "h264-encoder")]
#[allow(clippy::too_many_arguments)]
pub fn build_mp4_with_pattern_audio(
    profile: MuxerProfile,
    annex_b: &[u8],
    width: u32,
    height: u32,
    timing: FrameTiming,
    pattern: GopPattern,
    n_display_frames: usize,
    source_mp4: &[u8],
) -> Result<Vec<u8>, Mp4Error> {
    build_mp4_with_pattern_audio_lead_in(
        profile, annex_b, width, height, timing, pattern, n_display_frames, source_mp4, 0,
    )
}

/// [`build_mp4_with_pattern_audio`] for a source whose video starts
/// `lead_in_us` after the start of its timeline, i.e. after the
/// passed-through audio's time zero (a leading empty edit on the source's
/// video track, as stock-footage exports carry).
///
/// [`MuxerProfile::LavfOpenH264`] carries the lead-in as a leading empty
/// edit, rounded to whole frames — what a libavformat transcode of such a
/// source writes. The frames keep a uniform `stts`. Every frame lands on its
/// source time against the audio, which keeps its own edit list, and the
/// frame count is unchanged.
///
/// `lead_in_us == 0` is exactly [`build_mp4_with_pattern_audio`].
#[cfg(feature = "h264-encoder")]
#[allow(clippy::too_many_arguments)]
pub fn build_mp4_with_pattern_audio_lead_in(
    profile: MuxerProfile,
    annex_b: &[u8],
    width: u32,
    height: u32,
    timing: FrameTiming,
    pattern: GopPattern,
    n_display_frames: usize,
    source_mp4: &[u8],
    lead_in_us: u64,
) -> Result<Vec<u8>, Mp4Error> {
    if width == 0 || height == 0 {
        return Err(Mp4Error::InvalidBox(format!(
            "invalid dimensions {width}x{height}"
        )));
    }
    if timing.fps_num == 0 || timing.fps_den == 0 {
        return Err(Mp4Error::InvalidBox(format!(
            "invalid frame rate {}/{}",
            timing.fps_num, timing.fps_den
        )));
    }
    let nals = parse_annexb_nals(annex_b);
    if nals.is_empty() {
        return Err(Mp4Error::InvalidBox("no NAL units in input".into()));
    }
    let avcc = AvccData::from_annexb(annex_b)
        .ok_or_else(|| Mp4Error::InvalidBox("missing SPS or PPS in input".into()))?;
    let access_units = group_access_units(&nals);
    if access_units.is_empty() {
        return Err(Mp4Error::InvalidBox("no access units in input".into()));
    }

    let encode_order: Vec<_> = iter_encode_order(n_display_frames, pattern).collect();
    if encode_order.len() != access_units.len() {
        return Err(Mp4Error::InvalidBox(format!(
            "pattern emits {} frames but input has {} access units",
            encode_order.len(),
            access_units.len(),
        )));
    }
    let any_b_frame = encode_order.iter().any(|f| f.frame_type == FrameType::B);
    let composition_offsets: Option<Vec<i32>> = if any_b_frame {
        Some(
            encode_order
                .iter()
                .map(|f| {
                    let display = f.display_idx as i32;
                    let encode = f.encode_idx as i32;
                    (display - encode) * (shared::STTS_DELTA_PER_FRAME as i32)
                })
                .collect(),
        )
    } else {
        None
    };

    let audio = AudioPassthrough::extract_first(source_mp4)?;

    match profile {
        MuxerProfile::LavfOpenH264 => lavf::build(
            annex_b,
            &access_units,
            &avcc,
            width,
            height,
            timing,
            composition_offsets.as_deref(),
            audio.as_ref(),
            lead_in_us,
        ),
    }
}

/// Extracted audio track from a source MP4, ready to splice into a
/// new mux. `None` when the source has no audio.
pub struct AudioPassthrough<'a> {
    /// Original audio samples in source order. Lifetime tied to the
    /// caller's source MP4 byte slice.
    samples: Vec<&'a [u8]>,
    /// Source's `trak` box bytes, as the source has them. The output gets
    /// `audio_iso::passthrough_audio_trak` of them (ISO shape, output
    /// movie timescale, source identity cleared) with `tkhd.track_id`
    /// rewritten to 2 and `stco`/`co64` to our new mdat positions.
    trak_raw: Vec<u8>,
    /// The source file's `mvhd.time_scale`: the unit of `tkhd.duration` and
    /// the edit list inside `trak_raw` (see `audio_iso`).
    source_movie_timescale: u32,
}

impl<'a> AudioPassthrough<'a> {
    /// Extract the first audio track from `source_mp4`. Returns `None`
    /// when no audio track is present (caller proceeds with
    /// video-only mux).
    fn extract_first(source_mp4: &'a [u8]) -> Result<Option<Self>, Mp4Error> {
        let parsed = super::demux::demux(source_mp4)?;
        // Find first track with handler_type == "soun".
        let audio_idx = parsed
            .tracks
            .iter()
            .position(|t| &t.handler_type == b"soun");
        let audio_idx = match audio_idx {
            Some(i) => i,
            None => return Ok(None),
        };
        let track = &parsed.tracks[audio_idx];
        if track.samples.is_empty() {
            return Ok(None);
        }

        // Pull sample bytes by slicing into the source — sample.offset
        // is the absolute byte offset and sample.size is its length.
        let mut samples: Vec<&[u8]> = Vec::with_capacity(track.samples.len());
        for s in &track.samples {
            let start = s.offset as usize;
            let end = start + s.size as usize;
            if end > source_mp4.len() {
                return Err(Mp4Error::UnexpectedEof);
            }
            samples.push(&source_mp4[start..end]);
        }

        Ok(Some(AudioPassthrough {
            samples,
            trak_raw: track.trak_raw.clone(),
            source_movie_timescale: super::audio_iso::movie_timescale(source_mp4)
                .unwrap_or(shared::MVHD_TIMESCALE),
        }))
    }
}

// ─── Constants shared with the lavf profile ──────────────────────────

/// Container constants the `lavf` profile's movie/track header builders
/// share with this module.
mod shared {
    /// `mvhd.time_scale` (1000 = ms).
    pub const MVHD_TIMESCALE: u32 = 1000;

    /// Ticks per frame the composition offsets are computed in.
    #[cfg(feature = "h264-encoder")]
    pub const STTS_DELTA_PER_FRAME: u32 = 512;

    /// Video handler name.
    pub const HDLR_VIDE_NAME: &str = "VideoHandler";
}

fn build_mvhd(duration_ms: u64, next_track_id: u32) -> Vec<u8> {
    let mut content = Vec::new();
    // version=0, flags=0
    content.extend_from_slice(&[0, 0, 0, 0]);
    // creation_time / modification_time (zeroed for determinism)
    content.extend_from_slice(&[0; 4]);
    content.extend_from_slice(&[0; 4]);
    // time_scale
    content.extend_from_slice(&shared::MVHD_TIMESCALE.to_be_bytes());
    // duration (32-bit since version=0)
    content.extend_from_slice(&(duration_ms as u32).to_be_bytes());
    // rate = 1.0 (16.16 fixed)
    content.extend_from_slice(&0x0001_0000u32.to_be_bytes());
    // volume = 1.0 (8.8 fixed)
    content.extend_from_slice(&0x0100u16.to_be_bytes());
    // reserved (2 + 8 bytes = 10 zero bytes)
    content.extend_from_slice(&[0; 10]);
    // unity matrix
    content.extend_from_slice(&UNITY_MATRIX);
    // pre_defined (24 zero bytes)
    content.extend_from_slice(&[0; 24]);
    // next_track_ID
    content.extend_from_slice(&next_track_id.to_be_bytes());
    wrap_box(b"mvhd", &content)
}

const UNITY_MATRIX: [u8; 36] = [
    0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
];

fn build_hdlr_vide() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&[0, 0, 0, 0]); // version=0, flags=0
    content.extend_from_slice(&[0; 4]); // pre_defined
    content.extend_from_slice(b"vide");
    content.extend_from_slice(&[0; 12]); // reserved
    content.extend_from_slice(shared::HDLR_VIDE_NAME.as_bytes());
    content.push(0); // null terminator
    wrap_box(b"hdlr", &content)
}

fn build_vmhd() -> Vec<u8> {
    let mut content = Vec::new();
    // version=0, flags=1 (no_lean_ahead)
    content.extend_from_slice(&[0, 0, 0, 1]);
    // graphicsmode=0, opcolor=(0,0,0)
    content.extend_from_slice(&[0; 8]);
    wrap_box(b"vmhd", &content)
}

fn build_dinf() -> Vec<u8> {
    let mut url = Vec::new();
    url.extend_from_slice(&[0, 0, 0, 1]); // version=0, flags=1 (self-contained)
    let url_box = wrap_box(b"url ", &url);
    let mut dref_content = Vec::new();
    dref_content.extend_from_slice(&[0, 0, 0, 0]); // version+flags
    dref_content.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    dref_content.extend_from_slice(&url_box);
    let dref = wrap_box(b"dref", &dref_content);
    wrap_box(b"dinf", &dref)
}

/// Run-length `stts` over per-sample deltas: all-equal deltas give a single
/// run.
fn build_stts(deltas: &[u32]) -> Vec<u8> {
    let mut runs: Vec<(u32, u32)> = Vec::new(); // (sample_count, delta)
    for &d in deltas {
        match runs.last_mut() {
            Some((n, last)) if *last == d => *n += 1,
            _ => runs.push((1, d)),
        }
    }
    let mut content = Vec::with_capacity(8 + 8 * runs.len());
    content.extend_from_slice(&[0, 0, 0, 0]); // version+flags
    content.extend_from_slice(&(runs.len() as u32).to_be_bytes());
    for (n, d) in runs {
        content.extend_from_slice(&n.to_be_bytes());
        content.extend_from_slice(&d.to_be_bytes());
    }
    wrap_box(b"stts", &content)
}

fn build_stsc(chunk_sizes: &[u32]) -> Vec<u8> {
    // Compress runs of equal-sized chunks into a single stsc entry.
    // One chunk per GOP, so equal-size GOPs collapse to a single entry —
    // the most common shape.
    let mut entries: Vec<(u32, u32)> = Vec::new(); // (first_chunk, spc)
    for (i, &spc) in chunk_sizes.iter().enumerate() {
        if entries.last().is_none_or(|&(_, prev_spc)| prev_spc != spc) {
            entries.push((i as u32 + 1, spc));
        }
    }

    let mut content = Vec::new();
    content.extend_from_slice(&[0, 0, 0, 0]); // version+flags
    content.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for (first_chunk, spc) in entries {
        content.extend_from_slice(&first_chunk.to_be_bytes());
        content.extend_from_slice(&spc.to_be_bytes());
        content.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index
    }
    wrap_box(b"stsc", &content)
}

fn build_stco(chunk_offsets: &[u64]) -> Vec<u8> {
    // One offset per chunk (one chunk per GOP). Use stco when
    // all offsets fit in 32 bits; otherwise co64.
    let needs_co64 = chunk_offsets.iter().any(|&o| o > u32::MAX as u64);
    let mut content = Vec::new();
    content.extend_from_slice(&[0, 0, 0, 0]); // version+flags
    content.extend_from_slice(&(chunk_offsets.len() as u32).to_be_bytes());
    if needs_co64 {
        for &o in chunk_offsets {
            content.extend_from_slice(&o.to_be_bytes());
        }
        wrap_box(b"co64", &content)
    } else {
        for &o in chunk_offsets {
            content.extend_from_slice(&(o as u32).to_be_bytes());
        }
        wrap_box(b"stco", &content)
    }
}

fn build_stss(sync_samples: &[u32]) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&[0, 0, 0, 0]); // version+flags
    content.extend_from_slice(&(sync_samples.len() as u32).to_be_bytes());
    for &s in sync_samples {
        content.extend_from_slice(&s.to_be_bytes());
    }
    wrap_box(b"stss", &content)
}

// ──────────────────────────────────────────────────────────────────────
//  VP.M.3 — AV1-in-MP4 orchestrator
// ──────────────────────────────────────────────────────────────────────

/// VP.M.3 — top-level entry point: wrap an AV1 OBU byte stream in the
/// MP4 a libavformat transcode with the librav1e encoder writes (see
/// `build_lavf.rs`, "AV1"), video only, at the nominal `timing` rate.
///
/// **Per VP decision D2 (locked 2026-06-06)**: single cross-platform
/// Rust muxer. No platform-specific paths (no AVAssetWriter, no
/// MediaMuxer).
///
/// [`super::av1_obu_split::split_av1_into_samples`] partitions the stream
/// into temporal-unit samples (delimiters dropped, the sequence header in
/// each sync sample) and sync flags; the sequence header gives the `av1C`
/// fields and `colr`.
///
/// **Errors**: `InvalidBox` if dimensions are zero, frame rate is
/// zero, OBU stream contains no sequence_header_obu, or OBU stream
/// contains no frame samples.
pub fn build_mp4_av1(
    av1_obus: &[u8],
    width: u32,
    height: u32,
    timing: FrameTiming,
) -> Result<Vec<u8>, Mp4Error> {
    build_mp4_av1_inner(av1_obus, width, height, timing, None, None)
}

/// VP.M.4 — same as [`build_mp4_av1`] but also passes through the
/// first audio track from `source_mp4`. The source's audio `trak` box
/// is copied with `track_id` renumbered to 2, `stco`/`co64` rewritten to
/// point at the audio sample positions in the new mdat, and the fields
/// that identify the source file (creation / modification times, handler
/// name, data references, user data and metadata) cleared, as on the
/// H.264 path. Codec-agnostic — AAC, Opus, anything the source carries
/// works the same way because we never touch the audio codec-config
/// record (`mp4a`/`esds`, `Opus`/`dops`, etc.).
///
/// When `source_mp4` has no audio track the result is identical to
/// [`build_mp4_av1`].
pub fn build_mp4_av1_with_audio(
    av1_obus: &[u8],
    width: u32,
    height: u32,
    timing: FrameTiming,
    source_mp4: &[u8],
) -> Result<Vec<u8>, Mp4Error> {
    let audio = AudioPassthrough::extract_first(source_mp4)?;
    build_mp4_av1_inner(av1_obus, width, height, timing, audio.as_ref(), None)
}

/// [`build_mp4_av1`] / [`build_mp4_av1_with_audio`] with the video track
/// timed from the source's frame presentation times (µs, one per frame, in
/// presentation order — e.g. the decode pass's output times).
///
/// The untimed muxers stamp every frame at the nominal rate from 0: a
/// 29.97 / 23.976 source passed as an integer rate runs 0.1 % fast against
/// the passed-through audio, a variable-rate clip loses its cadence, and a
/// lead-in (a leading empty edit in the source) is lost. Here the clip's
/// constant rate, its cadence in whole frame slots and the lead-in (an
/// empty edit) are kept, as the libavformat transcode keeps them — see
/// `lavf::av1_timeline`.
///
/// When `frame_times_us` does not have one time per frame (or fewer than
/// two), the result is exactly the untimed muxer's at `timing`.
pub fn build_mp4_av1_timed(
    av1_obus: &[u8],
    width: u32,
    height: u32,
    timing: FrameTiming,
    frame_times_us: &[i64],
    source_mp4: Option<&[u8]>,
) -> Result<Vec<u8>, Mp4Error> {
    let audio = match source_mp4 {
        Some(src) => AudioPassthrough::extract_first(src)?,
        None => None,
    };
    build_mp4_av1_inner(av1_obus, width, height, timing, audio.as_ref(), Some(frame_times_us))
}

/// [`build_mp4_av1_timed`] with the source given as a file path, mapped
/// read-only, so a long clip never loads into the heap just to copy its
/// audio. The source's audio track is copied as-is, edit list included —
/// re-containerising it instead (Android's `MediaMuxer`) drops a leading
/// empty edit, and the audio then plays early against the video.
///
/// `None` muxes video only. A source that cannot be opened or mapped is an
/// error; the caller decides whether to retry without audio.
#[cfg(feature = "video")]
pub fn build_mp4_av1_timed_from_path(
    av1_obus: &[u8],
    width: u32,
    height: u32,
    timing: FrameTiming,
    frame_times_us: &[i64],
    source_path: Option<&std::path::Path>,
) -> Result<Vec<u8>, Mp4Error> {
    let Some(path) = source_path else {
        return build_mp4_av1_timed(av1_obus, width, height, timing, frame_times_us, None);
    };
    let file = std::fs::File::open(path)
        .map_err(|e| Mp4Error::InvalidBox(format!("cannot open source {}: {e}", path.display())))?;
    // SAFETY: read-only map of a file the app owns for the length of this
    // call; nothing writes it while the mux runs.
    let source = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| Mp4Error::InvalidBox(format!("cannot map source {}: {e}", path.display())))?;
    build_mp4_av1_timed(av1_obus, width, height, timing, frame_times_us, Some(&source))
}

fn build_mp4_av1_inner(
    av1_obus: &[u8],
    width: u32,
    height: u32,
    timing: FrameTiming,
    audio: Option<&AudioPassthrough<'_>>,
    frame_times_us: Option<&[i64]>,
) -> Result<Vec<u8>, Mp4Error> {
    use super::av1_obu_split::{parse_sequence_header_obu, split_av1_into_samples};

    if width == 0 || height == 0 {
        return Err(Mp4Error::InvalidBox(format!(
            "invalid dimensions {width}x{height}"
        )));
    }
    if timing.fps_num == 0 || timing.fps_den == 0 {
        return Err(Mp4Error::InvalidBox(format!(
            "invalid frame rate {}/{}",
            timing.fps_num, timing.fps_den
        )));
    }

    let split = split_av1_into_samples(av1_obus)?;
    if split.sequence_header_obu.is_empty() {
        return Err(Mp4Error::InvalidBox(
            "no sequence_header_obu in stream".into(),
        ));
    }
    if split.samples.is_empty() {
        return Err(Mp4Error::InvalidBox(
            "no frame samples in stream".into(),
        ));
    }
    let sh = parse_sequence_header_obu(&split.sequence_header_obu);
    lavf::build_av1(split.samples, &split.sync, sh.as_ref(), width, height, timing, frame_times_us, audio)
}

// ─── Annex-B parsing ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ParsedNal {
    /// Payload start offset in the original byte stream (after the start code).
    start: usize,
    /// Payload end offset (exclusive).
    end: usize,
    nal_type: NalType,
}

fn parse_annexb_nals(bytes: &[u8]) -> Vec<ParsedNal> {
    let starts = find_start_codes(bytes);
    let mut out = Vec::with_capacity(starts.len());
    for (k, &(payload_start, _sc_len)) in starts.iter().enumerate() {
        let payload_end = if k + 1 < starts.len() {
            // Trim the next NAL's start-code prefix off the back.
            let next = starts[k + 1];
            next.0 - next.1
        } else {
            bytes.len()
        };
        if payload_start >= payload_end {
            continue;
        }
        let nal_type = NalType(bytes[payload_start] & 0x1F);
        out.push(ParsedNal { start: payload_start, end: payload_end, nal_type });
    }
    out
}

/// Returns (payload_start_offset, start_code_len_bytes) for each NAL.
fn find_start_codes(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 <= bytes.len() {
        if bytes[i] == 0 && bytes[i + 1] == 0 {
            if i + 4 <= bytes.len() && bytes[i + 2] == 0 && bytes[i + 3] == 1 {
                out.push((i + 4, 4));
                i += 4;
                continue;
            } else if bytes[i + 2] == 1 {
                out.push((i + 3, 3));
                i += 3;
                continue;
            }
        }
        i += 1;
    }
    out
}

#[derive(Debug, Clone)]
struct AccessUnit {
    /// (start_offset, end_offset, nal_type) for each NAL in this AU,
    /// in input order.
    nals: Vec<(usize, usize, NalType)>,
    /// True when this AU contains an IDR slice.
    is_sync: bool,
}

/// Group NALs into access units. Boundaries are detected by:
///   1. AUD NAL (type 9) — explicit boundary at the AUD's position.
///   2. First VCL NAL (types 1-5) in a row after a non-VCL run, when the
///      previous AU already had a VCL.
///
/// Phasm always emits AUD-prefixed access units (encoder.rs:1196 etc), so
/// rule (1) carries the load.
fn group_access_units(nals: &[ParsedNal]) -> Vec<AccessUnit> {
    let mut out: Vec<AccessUnit> = Vec::new();
    let mut current = AccessUnit { nals: Vec::new(), is_sync: false };
    let mut current_has_vcl = false;

    for n in nals {
        let is_vcl = n.nal_type.is_vcl();
        let is_aud = n.nal_type == NalType::AUD;
        let starts_new = is_aud
            || (is_vcl && current_has_vcl);
        if starts_new && !current.nals.is_empty() {
            out.push(std::mem::replace(
                &mut current,
                AccessUnit { nals: Vec::new(), is_sync: false },
            ));
            current_has_vcl = false;
        }
        current.nals.push((n.start, n.end, n.nal_type));
        if is_vcl {
            current_has_vcl = true;
            if n.nal_type.is_idr() {
                current.is_sync = true;
            }
        }
    }
    if !current.nals.is_empty() {
        out.push(current);
    }

    // Drop any access unit that has no VCL NAL — phasm doesn't emit
    // parameter-set-only AUs in practice but the guard keeps the mux
    // valid for malformed inputs.
    out.retain(|au| au.nals.iter().any(|&(_, _, t)| t.is_vcl()));
    out
}

// ─── Box helpers ──────────────────────────────────────────────────────

fn wrap_box(box_type: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let total_size = (8 + content.len()) as u32;
    let mut out = Vec::with_capacity(total_size as usize);
    out.extend_from_slice(&total_size.to_be_bytes());
    out.extend_from_slice(box_type);
    out.extend_from_slice(content);
    out
}

// ─── §Stealth.L4.5 audio passthrough helpers ──────────────────────────
//
// These walk a source `trak` box and rewrite specific sub-fields in
// place, without parsing or re-emitting the rest of the structure.
// Same approach as `super::mux::patch_*_in_moov` — preserves opaque
// fields (esds for AAC, dops for Opus, etc.) byte-exact.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::mp4::Av1cData;

    // ──────────────────────────────────────────────────────────────
    //  VP.M.1 — AV1 box builder unit tests
    // ──────────────────────────────────────────────────────────────

    /// Helper: synthesize a 16-byte fake `sequence_header_obu` for
    /// av1C ConfigOBU tests. Not a valid AV1 SH, just placeholder bytes.
    fn fake_sh_obu() -> Vec<u8> {
        vec![0x0A, 0x0E, 0x00, 0x00, 0x00, 0x42, 0xAB, 0xBF, 0xC3, 0xFB, 0xB3, 0xFE, 0x40, 0x00, 0x00, 0x00]
    }

    #[test]
    fn av1c_serialization_default_4_2_0_8bit() {
        let sh = fake_sh_obu();
        let av1c = Av1cData::default_yuv420_8bit(sh.clone());
        let bytes = av1c.to_bytes();

        // 4 header bytes + ConfigOBUs length.
        assert_eq!(bytes.len(), 4 + sh.len());

        // Byte 0: marker(1)=1, version(7)=1 → 0b1000_0001 = 0x81.
        assert_eq!(bytes[0], 0x81);

        // Byte 1: seq_profile(3)=0, seq_level_idx_0(5)=8 → 0b000_01000 = 0x08.
        assert_eq!(bytes[1], 0x08);

        // Byte 2: bits for 4:2:0 8-bit:
        //   seq_tier_0=0, high_bitdepth=0, twelve_bit=0, monochrome=0,
        //   chroma_subsampling_x=1, chroma_subsampling_y=1, chroma_sample_position=0
        //   → 0b0000_1100 = 0x0C.
        assert_eq!(bytes[2], 0x0C);

        // Byte 3: reserved(3) | initial_presentation_delay_present(1)=0 | reserved(4) = 0.
        assert_eq!(bytes[3], 0x00);

        // Trailing ConfigOBUs.
        assert_eq!(&bytes[4..], sh.as_slice());
    }

    #[test]
    fn av1c_serialization_high_tier_10bit_4_2_2() {
        // Non-default profile for coverage of all the bit-field branches.
        let av1c = Av1cData {
            configuration_version: 1,
            seq_profile: 2,                 // 2 = Professional
            seq_level_idx_0: 16,            // Level 6.0
            seq_tier_0: 1,                  // High tier
            high_bitdepth: true,
            twelve_bit: false,              // → 10-bit
            monochrome: false,
            chroma_subsampling_x: true,
            chroma_subsampling_y: false,    // 4:2:2
            chroma_sample_position: 2,
            config_obus: Vec::new(),
        };
        let bytes = av1c.to_bytes();
        assert_eq!(bytes.len(), 4);
        assert_eq!(bytes[0], 0x81);                                  // marker=1, version=1
        assert_eq!(bytes[1], (2 << 5) | 16);                         // seq_profile=2, level=16
        assert_eq!(bytes[2], 0x80 | 0x40 | 0x08 | 0x02);             // tier=1, hi_bd=1, ss_x=1, csp=2
        assert_eq!(bytes[3], 0x00);
    }


    /// Build the minimum viable Annex-B stream: SPS + PPS + AUD + IDR.
    fn build_minimal_annexb() -> Vec<u8> {
        // SPS NAL (header 0x67 = nal_ref_idc=3, type=7), profile=High(100),
        // constraints=0, level=30, then minimal SPS body bytes (we don't
        // need a fully decodable SPS for the mux test — just parseable
        // profile/level bytes).
        let sps_payload = vec![
            0x67, 0x64, 0x00, 0x1E, // header + profile + constraint + level
            0xAC, 0xD9, 0x40, 0x40, 0x3C, 0x80,
        ];
        let pps_payload = vec![0x68, 0xEB, 0xE3, 0xCB, 0x22, 0xC0];
        let aud_payload = vec![0x09, 0x10]; // AUD with primary_pic_type=0
        let idr_payload = vec![
            0x65, 0x88, 0x84, 0x00, 0x33, 0xFF, 0xFE, // IDR slice (truncated)
        ];

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0, 0, 0, 1]);
        bytes.extend(&sps_payload);
        bytes.extend_from_slice(&[0, 0, 0, 1]);
        bytes.extend(&pps_payload);
        bytes.extend_from_slice(&[0, 0, 0, 1]);
        bytes.extend(&aud_payload);
        bytes.extend_from_slice(&[0, 0, 0, 1]);
        bytes.extend(&idr_payload);
        bytes
    }

    #[test]
    fn build_mp4_demuxes_back_to_one_video_track() {
        use crate::codec::mp4::demux::demux;
        let annex_b = build_minimal_annexb();
        let mp4 = build_mp4(
            MuxerProfile::LavfOpenH264,
            &annex_b,
            1920,
            1080,
            FrameTiming::FPS_30,
        )
        .unwrap();

        let parsed = demux(&mp4).expect("demux must succeed");
        assert_eq!(parsed.tracks.len(), 1);
        let video_idx = parsed.video_track_idx.expect("video track");
        let track = &parsed.tracks[video_idx];
        assert!(track.is_h264(), "track is recognised as H.264");
        assert_eq!(track.width, 1920);
        assert_eq!(track.height, 1080);
        // One access unit (one IDR sample).
        assert_eq!(track.samples.len(), 1);
        assert!(track.samples[0].is_sync, "IDR sample is sync");
        // mdhd timescale: 30 × 512 = 15360 @ 30 fps.
        assert_eq!(track.timescale, 15360);
    }

    #[test]
    fn group_access_units_splits_on_aud() {
        // Stream: AUD SPS PPS IDR | AUD SLICE | AUD SLICE
        // Result: 3 access units, only the first is sync (IDR).
        let mut bytes = Vec::new();
        // AU 1
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x10]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x67, 0x64, 0x00, 0x1E, 0xAC, 0xD9, 0x40, 0x40, 0x3C, 0x80]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xEB, 0xE3, 0xCB]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00]);
        // AU 2
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x30]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x41, 0x9A, 0x00]);
        // AU 3
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x30]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x41, 0x9A, 0x00]);

        let nals = parse_annexb_nals(&bytes);
        let aus = group_access_units(&nals);
        assert_eq!(aus.len(), 3);
        assert!(aus[0].is_sync);
        assert!(!aus[1].is_sync);
        assert!(!aus[2].is_sync);
    }

    #[test]
    fn build_mp4_strips_aud_and_parameter_sets_from_mdat() {
        // After mux, the IDR sample contains the IDR slice NAL only (no SEI is
        // injected). SPS, PPS, AUD are stripped. Walk the NAL list and verify
        // NO type-6/7/8/9 NALs remain.
        let annex_b = build_minimal_annexb();
        let mp4 = build_mp4(
            MuxerProfile::LavfOpenH264,
            &annex_b,
            1920,
            1080,
            FrameTiming::FPS_30,
        )
        .unwrap();

        let mdat_payload = find_box(&mp4, b"mdat");

        let mut pos = 0;
        let mut saw_idr = false;
        while pos + 4 <= mdat_payload.len() {
            let len = u32::from_be_bytes([
                mdat_payload[pos],
                mdat_payload[pos + 1],
                mdat_payload[pos + 2],
                mdat_payload[pos + 3],
            ]) as usize;
            pos += 4;
            assert!(pos + len <= mdat_payload.len());
            let nal_type = mdat_payload[pos] & 0x1F;
            assert_ne!(nal_type, 7, "SPS must NOT appear in mdat");
            assert_ne!(nal_type, 8, "PPS must NOT appear in mdat");
            assert_ne!(nal_type, 9, "AUD must NOT appear in mdat");
            assert_ne!(nal_type, 6, "no SEI is injected into mdat");
            if nal_type == 5 {
                saw_idr = true;
            }
            pos += len;
        }
        assert_eq!(pos, mdat_payload.len(), "mdat parses exactly to end");
        assert!(saw_idr, "IDR slice NAL is present in mdat");
    }

    #[test]
    fn build_mp4_rejects_missing_parameter_sets() {
        // Annex-B with only an IDR (no SPS/PPS) → must fail with a clear
        // error.
        let bytes = vec![0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00];
        let r = build_mp4(
            MuxerProfile::LavfOpenH264,
            &bytes,
            1920,
            1080,
            FrameTiming::FPS_30,
        );
        assert!(r.is_err());
    }

    #[test]
    fn build_mp4_rejects_zero_dimensions() {
        let annex_b = build_minimal_annexb();
        let r = build_mp4(
            MuxerProfile::LavfOpenH264,
            &annex_b,
            0,
            1080,
            FrameTiming::FPS_30,
        );
        assert!(r.is_err());
    }

    #[test]
    fn build_stsc_collapses_uniform_runs() {
        // Three identical-size GOPs → 1 stsc entry.
        let stsc = build_stsc(&[30, 30, 30]);
        // 8-byte header + 4 (version+flags) + 4 (entry_count) + 1×12
        // (one entry of 3 u32s) = 28 bytes total.
        assert_eq!(stsc.len(), 8 + 4 + 4 + 12);
        // entry_count = 1
        assert_eq!(&stsc[12..16], &1u32.to_be_bytes());
        assert_eq!(&stsc[16..20], &1u32.to_be_bytes()); // first_chunk
        assert_eq!(&stsc[20..24], &30u32.to_be_bytes()); // samples_per_chunk
        assert_eq!(&stsc[24..28], &1u32.to_be_bytes()); // sd_idx
    }

    #[test]
    fn build_stsc_splits_mixed_runs() {
        // GOP sizes [30, 30, 25, 30] → 3 stsc entries.
        let stsc = build_stsc(&[30, 30, 25, 30]);
        assert_eq!(stsc.len(), 8 + 4 + 4 + 3 * 12);
        // entry_count = 3
        assert_eq!(&stsc[12..16], &3u32.to_be_bytes());
        assert_eq!(&stsc[16..20], &1u32.to_be_bytes()); // first_chunk = 1
        assert_eq!(&stsc[20..24], &30u32.to_be_bytes()); // spc = 30
        assert_eq!(&stsc[28..32], &3u32.to_be_bytes()); // first_chunk = 3
        assert_eq!(&stsc[32..36], &25u32.to_be_bytes()); // spc = 25
        assert_eq!(&stsc[40..44], &4u32.to_be_bytes()); // first_chunk = 4
        assert_eq!(&stsc[44..48], &30u32.to_be_bytes()); // spc = 30
    }

        #[cfg(feature = "h264-encoder")]
        #[test]
    fn build_mp4_with_pattern_rejects_au_count_mismatch() {
        // Pattern asks for 5 frames but input has 1 → error.
        use crate::codec::h264::stego::gop_pattern::GopPattern;
        let annex_b = build_minimal_annexb();
        let r = build_mp4_with_pattern(
            MuxerProfile::LavfOpenH264,
            &annex_b,
            1920,
            1080,
            FrameTiming::FPS_30,
            GopPattern::Ibpbp { gop: 5, b_count: 1 },
            5,
        );
        assert!(r.is_err());
    }

    #[test]
    fn build_mp4_rejects_zero_fps() {
        let annex_b = build_minimal_annexb();
        let r = build_mp4(
            MuxerProfile::LavfOpenH264,
            &annex_b,
            1920,
            1080,
            FrameTiming { fps_num: 0, fps_den: 1 },
        );
        assert!(r.is_err());
    }

    /// `build_minimal_annexb`'s IDR followed by `n - 1` AUD-delimited P
    /// access units.
    #[cfg(feature = "h264-encoder")]
    fn build_annexb_frames(n: usize) -> Vec<u8> {
        let mut bytes = build_minimal_annexb();
        for _ in 1..n {
            bytes.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x30]); // AUD
            bytes.extend_from_slice(&[0, 0, 0, 1, 0x41, 0x9A, 0x02, 0x10]); // P slice (truncated)
        }
        bytes
    }

    /// The `stts` entries `(sample_count, delta)` of the first `stts` box.
    #[cfg(feature = "h264-encoder")]
    fn stts_entries(mp4: &[u8]) -> Vec<(u32, u32)> {
        let at = mp4.windows(4).position(|w| w == b"stts").expect("stts box");
        let rd = |o: usize| u32::from_be_bytes(mp4[o..o + 4].try_into().unwrap());
        let n = rd(at + 8) as usize;
        (0..n).map(|i| (rd(at + 12 + 8 * i), rd(at + 16 + 8 * i))).collect()
    }

    // ─── LavfOpenH264 profile ────────────────────────────────────────

    /// Samples of the first track as NAL-unit lists (4-byte length prefixes).
    fn sample_nals(mp4: &[u8]) -> Vec<Vec<Vec<u8>>> {
        let parsed = super::super::demux::demux(mp4).unwrap();
        let v = parsed.tracks.iter().find(|t| &t.handler_type == b"vide").unwrap();
        v.samples
            .iter()
            .map(|s| {
                let d = &mp4[s.offset as usize..(s.offset + s.size as u64) as usize];
                let mut out = Vec::new();
                let mut i = 0;
                while i < d.len() {
                    let n = u32::from_be_bytes(d[i..i + 4].try_into().unwrap()) as usize;
                    out.push(d[i + 4..i + 4 + n].to_vec());
                    i += 4 + n;
                }
                out
            })
            .collect()
    }

    fn find_box<'a>(mp4: &'a [u8], typ: &[u8; 4]) -> &'a [u8] {
        let at = mp4.windows(4).position(|w| w == typ).unwrap_or_else(|| panic!("no {typ:?}"));
        let size = u32::from_be_bytes(mp4[at - 4..at].try_into().unwrap()) as usize;
        &mp4[at + 4..at - 4 + size]
    }

    /// The profile carries exactly the encoder's NAL units: every NAL of the
    /// input but AUD / SPS / PPS, and nothing added (in particular no SEI).
    #[cfg(feature = "h264-encoder")]
    #[test]
    fn lavf_profile_keeps_every_nal_but_aud_and_parameter_sets() {
        use crate::codec::h264::stego::gop_pattern::GopPattern;
        let annex_b = build_annexb_frames(7);
        let pat = GopPattern::Ipppp { gop: 30 };
        let lavf = build_mp4_with_pattern(MuxerProfile::LavfOpenH264, &annex_b, 1920, 1080, FrameTiming::FPS_30, pat, 7).unwrap();
        let want: Vec<Vec<Vec<u8>>> = group_access_units(&parse_annexb_nals(&annex_b))
            .iter()
            .map(|au| {
                au.nals
                    .iter()
                    .filter(|&&(_, _, t)| t != NalType::AUD && t != NalType::SPS && t != NalType::PPS)
                    .map(|&(a, b, _)| annex_b[a..b].to_vec())
                    .collect()
            })
            .collect();
        assert_eq!(sample_nals(&lavf), want);
    }

    #[test]
    fn lavf_profile_has_no_x264_or_handbrake_marker() {
        // The 16-byte UUID of an x264 version SEI (`user_data_unregistered`).
        const X264_SEI_UUID: [u8; 16] = [
            0xDC, 0x45, 0xE9, 0xBD, 0xE6, 0xD9, 0x48, 0xB7,
            0x96, 0x2C, 0xD8, 0x20, 0xD9, 0x23, 0xEE, 0xEF,
        ];
        let mp4 = build_mp4(MuxerProfile::LavfOpenH264, &build_minimal_annexb(), 1920, 1080, FrameTiming::FPS_30).unwrap();
        let has = |n: &[u8]| mp4.windows(n.len()).any(|w| w == n);
        assert!(!has(b"x264"), "x264 banner");
        assert!(!has(&X264_SEI_UUID), "x264 SEI UUID");
        assert!(!has(b"HandBrake"), "HandBrake tag");
        assert!(has(lavf::LAVF_TOOL.as_bytes()), "muxer tag");
        assert!(has(lavf::LAVC_COMPRESSOR.as_bytes()), "compressor name");
        assert_eq!(&mp4[32..40], &[0, 0, 0, 8, b'f', b'r', b'e', b'e'], "free before mdat");
        assert_eq!(&mp4[44..48], b"mdat");
    }

    #[test]
    fn lavf_profile_headers() {
        let mp4 = build_mp4(MuxerProfile::LavfOpenH264, &build_minimal_annexb(), 1920, 1080, FrameTiming::FPS_30).unwrap();
        let tkhd = find_box(&mp4, b"tkhd");
        assert_eq!(&tkhd[..4], &[0, 0, 0, 3], "tkhd flags enabled | in movie");
        let elst = find_box(&mp4, b"elst");
        // One frame at 30 fps: 512 / 15360 s → 34 ms (rounded up), media time 0.
        assert_eq!(elst, &[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 34, 0, 0, 0, 0, 0, 1, 0, 0]);
        let mdhd = find_box(&mp4, b"mdhd");
        assert_eq!(&mdhd[12..16], &15360u32.to_be_bytes());
        // avcC: High profile → the chroma-format / bit-depth extension.
        let avcc = find_box(&mp4, b"avcC");
        assert_eq!(&avcc[avcc.len() - 4..], &[0xFD, 0xF8, 0xF8, 0x00]);
        // btrt: (bufferSizeDB 0, max = average, average 0).
        let btrt = find_box(&mp4, b"btrt");
        assert_eq!(&btrt[..4], &[0; 4]);
        assert_ne!(&btrt[4..8], &[0; 4]);
        assert_eq!(&btrt[8..12], &[0; 4]);
        // A single IDR: every sample is a sync sample → no stss.
        assert!(!mp4.windows(4).any(|w| w == b"stss"));
    }

    #[cfg(feature = "h264-encoder")]
    #[test]
    fn lavf_profile_fractional_rate_uses_the_rate_as_timescale() {
        use crate::codec::h264::stego::gop_pattern::GopPattern;
        let annex_b = build_annexb_frames(5);
        let t = FrameTiming { fps_num: 30000, fps_den: 1001 };
        let mp4 = build_mp4_with_pattern(MuxerProfile::LavfOpenH264, &annex_b, 1920, 1080, t, GopPattern::Ipppp { gop: 30 }, 5).unwrap();
        assert_eq!(&find_box(&mp4, b"mdhd")[12..16], &30000u32.to_be_bytes());
        assert_eq!(stts_entries(&mp4), vec![(5, 1001)]);
        assert!(mp4.windows(4).any(|w| w == b"stss"), "P frames → stss");
    }

    #[cfg(feature = "h264-encoder")]
    #[test]
    fn lavf_profile_rejects_b_frames() {
        use crate::codec::h264::stego::gop_pattern::GopPattern;
        let annex_b = build_annexb_frames(5);
        let pat = GopPattern::Ibpbp { gop: 30, b_count: 1 };
        let n = crate::codec::h264::stego::gop_pattern::iter_encode_order(5, pat).count();
        let annex_b = if n == 5 { annex_b } else { build_annexb_frames(n) };
        assert!(build_mp4_with_pattern(MuxerProfile::LavfOpenH264, &annex_b, 1920, 1080, FrameTiming::FPS_30, pat, 5).is_err());
    }

    /// Runs of contiguous samples are chunked below 1 MiB.
    #[test]
    fn lavf_profile_chunks_below_one_mebibyte() {
        let mut annex_b = build_minimal_annexb();
        for _ in 1..8 {
            annex_b.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x30]);
            annex_b.extend_from_slice(&[0, 0, 0, 1, 0x41, 0x9A]);
            annex_b.extend(std::iter::repeat_n(0x55u8, 300_000));
        }
        let mp4 = build_mp4(MuxerProfile::LavfOpenH264, &annex_b, 1920, 1080, FrameTiming::FPS_30).unwrap();
        let stsc = find_box(&mp4, b"stsc");
        let rd = |o: usize| u32::from_be_bytes(stsc[o..o + 4].try_into().unwrap());
        let entries: Vec<(u32, u32)> = (0..rd(4) as usize).map(|i| (rd(8 + 12 * i), rd(12 + 12 * i))).collect();
        // IDR (tiny) + 3 × 300 kB = 0.9 MB; a 4th would reach 1.2 MB.
        assert_eq!(entries, vec![(1, 4), (2, 3), (3, 1)]);
    }

    /// A source lead-in becomes a leading empty edit in whole frames; the
    /// frames keep one uniform `stts` run.
    #[cfg(feature = "h264-encoder")]
    #[test]
    fn lavf_lead_in_is_an_empty_edit() {
        use crate::codec::h264::stego::gop_pattern::GopPattern;
        let annex_b = build_annexb_frames(5);
        let source = build_mp4(MuxerProfile::LavfOpenH264, &build_minimal_annexb(), 1920, 1080, FrameTiming::FPS_30).unwrap();
        // 110 ms at 30 fps → 3.3 frames → 3 frames = 1536 ticks of 15360.
        let mp4 = build_mp4_with_pattern_audio_lead_in(
            MuxerProfile::LavfOpenH264, &annex_b, 1920, 1080, FrameTiming::FPS_30,
            GopPattern::Ipppp { gop: 30 }, 5, &source, 110_000,
        )
        .unwrap();
        assert_eq!(stts_entries(&mp4), vec![(5, 512)]);
        let elst = find_box(&mp4, b"elst");
        let rd = |o: usize| u32::from_be_bytes(elst[o..o + 4].try_into().unwrap());
        assert_eq!(rd(4), 2);
        assert_eq!((rd(8), rd(12) as i32), (100, -1), "empty edit, 1536 ticks → 100 ms");
        assert_eq!((rd(20), rd(24)), (167, 0), "5 frames → 166.7 ms, rounded up");
        let tkhd = find_box(&mp4, b"tkhd");
        assert_eq!(u32::from_be_bytes(tkhd[20..24].try_into().unwrap()), 267, "(1536 + 2560) ticks → 266.7 ms");
    }
}
