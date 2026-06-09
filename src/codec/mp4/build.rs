// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phasm-owned MP4 builder for H.264 Annex-B output.
//!
//! Wraps a phasm-encoded H.264 Annex-B bitstream in an MP4 container whose
//! top-level atom signature matches a chosen [`MuxerProfile`]. The default
//! `HandbrakeX264` profile lands phasm output inside the HandBrake/x264
//! converter-pipeline metaclass — the largest single container metaclass on
//! the open internet (Agent B 2026-05-02 + strategy doc
//! `docs/design/video/h264/stealth-strategy.md`).
//!
//! Distinct from [`super::mux`], which patches an existing source MP4 by
//! copying its `moov` and substituting modified samples — that path makes
//! sense when the source file is also H.264 (HEVC sources can't be
//! shadow-patched). The builder here writes a fresh MP4 from scratch given
//! the target profile and an Annex-B byte stream.
//!
//! # Scope
//!
//! Shipped (§Stealth.L4.1):
//! - HandBrake/x264 container signature: `ftyp` major+compatibles,
//!   `mvhd.time_scale=1000`, `hdlr.name="VideoHandler"`, `udta/©too`
//!   plaintext "HandBrake X.X.X yyyymmddhhss".
//! - Top-level atom order `ftyp → mdat → moov` (HandBrake without
//!   `+faststart`).
//! - `avcC` from extracted SPS+PPS NAL units; lengthSizeMinusOne=3.
//! - One sample per access unit; AUD NALs are stripped from `mdat`
//!   (HandBrake/x264 convention — Apple/Android keep AUDs).
//! - One chunk per GOP (run-length compressed `stsc`, per-GOP `stco`).
//! - IPPPP timing (uniform `stts`, no `ctts`).
//!
//! Shipped (§Stealth.L4.2):
//! - x264 SEI `user_data_unregistered` NAL with the canonical UUID +
//!   plaintext "x264 - core 164 …" banner injected into the leading IDR
//!   sample. Strongest single L4 plaintext fingerprint per Agent B
//!   survey §Tier B.
//! - B-frame timing — [`build_mp4_with_pattern`] emits a per-sample
//!   `ctts` box (version=1 with signed offsets) and an `edts/elst`
//!   pre-roll edit when the input GOP pattern carries B-frames. Plain
//!   `build_mp4` keeps the IPPPP / I-only path (no `ctts`, no `edts`).
//!
//! Shipped (§Stealth.L3.1):
//! - VUI emission on every CABAC SPS path (encoder side, see
//!   `core/src/codec/h264/encoder/bitstream_writer.rs`). The new
//!   builder reads SPS+PPS straight from the input Annex-B and stores
//!   them verbatim in `avcC`, so VUI presence flows through end-to-end.
//!
//! Deferred (held for follow-on sessions):
//! - Byte-exact `avcC` byte equality vs reference x264 SPS+PPS —
//!   currently blocked on stego encoder still emitting Main profile
//!   (§Stealth.L3.1 follow-on / I_8x8 walker support).
//! - Audio track passthrough — separate task.

use super::{write_u32, write_u64, Av1cData, AvccData, Mp4Error};
use crate::codec::h264::NalType;
#[cfg(feature = "h264-encoder")]
use crate::codec::h264::stego::gop_pattern::{iter_encode_order, FrameType, GopPattern};

/// Container target profile. Each variant carries the byte-exact atom
/// signature of one converter-pipeline metaclass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuxerProfile {
    /// HandBrake / FFmpeg + libx264 (the v1.0 default — largest single
    /// internet container metaclass, smallest engineering gap from phasm
    /// encoder defaults).
    HandbrakeX264,
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
/// `width` / `height` are the encoded picture dimensions in pixels.
///
/// The Annex-B stream must contain at least one SPS NAL and one PPS NAL —
/// they are extracted into the `avcC` decoder configuration record and
/// then dropped from the per-frame mdat samples (HandBrake/x264
/// convention).
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
        MuxerProfile::HandbrakeX264 => build_handbrake_x264(
            annex_b,
            &access_units,
            &avcc,
            width,
            height,
            timing,
            None,
            None,
        ),
    }
}

/// §Stealth.L4.2 follow-on — B-frame-aware variant of [`build_mp4`].
///
/// When the input Annex-B stream is encoded in IBPBP (or any pattern with
/// B-frames), access units appear in encode order — but the decoder must
/// display frames in display order. The standard MP4 mechanism is the
/// `ctts` (composition time-to-sample) box: per-sample CTS-DTS offset.
/// Real HandBrake/x264 output also emits an `edts/elst` leading empty
/// edit so the first displayed frame's PTS lands at zero.
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
    // STTS_DELTA_PER_FRAME units. Same for IPPPP (always 0) — but we
    // still emit ctts when any B-frame is present to match HandBrake/
    // x264 convention.
    let any_b_frame = encode_order.iter().any(|f| f.frame_type == FrameType::B);
    let composition_offsets: Option<Vec<i32>> = if any_b_frame {
        Some(
            encode_order
                .iter()
                .map(|f| {
                    let display = f.display_idx as i32;
                    let encode = f.encode_idx as i32;
                    (display - encode) * (handbrake::STTS_DELTA_PER_FRAME as i32)
                })
                .collect(),
        )
    } else {
        None
    };

    match profile {
        MuxerProfile::HandbrakeX264 => build_handbrake_x264(
            annex_b,
            &access_units,
            &avcc,
            width,
            height,
            timing,
            composition_offsets.as_deref(),
            None,
        ),
    }
}

/// §Stealth.L4.5 — variant of [`build_mp4_with_pattern`] that also
/// passes through an audio track from the source MP4. The HandBrake-
/// class video container shape is unchanged; the audio is added as a
/// second track inside `moov` with its samples appended to `mdat`
/// after the video samples.
///
/// # Audio handling
///
/// The source's first audio track is copied verbatim. The codec
/// configuration record (mp4a/esds for AAC, dops for Opus, etc.)
/// passes through unchanged inside the source's `stsd` box. Sample
/// timing (`stts`/`stsc`/`stsz`) is preserved; only `stco`/`co64`
/// chunk offsets are rewritten to point at the new MP4's mdat
/// positions. Track IDs are renumbered so video stays at `track_id=1`
/// and audio takes `track_id=2`.
///
/// `source_mp4` is the original MP4 byte stream (or any MP4 carrying
/// the desired audio track — typically the same input the YUV came
/// from). When the source has no audio track, the result is identical
/// to [`build_mp4_with_pattern`].
#[cfg(feature = "h264-encoder")]
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
                    (display - encode) * (handbrake::STTS_DELTA_PER_FRAME as i32)
                })
                .collect(),
        )
    } else {
        None
    };

    let audio = AudioPassthrough::extract_first(source_mp4)?;

    match profile {
        MuxerProfile::HandbrakeX264 => build_handbrake_x264(
            annex_b,
            &access_units,
            &avcc,
            width,
            height,
            timing,
            composition_offsets.as_deref(),
            audio.as_ref(),
        ),
    }
}

/// Extracted audio track from a source MP4, ready to splice into a
/// new HandBrake-class mux. `None` when the source has no audio.
pub struct AudioPassthrough<'a> {
    /// Original audio samples in source order. Lifetime tied to the
    /// caller's source MP4 byte slice.
    samples: Vec<&'a [u8]>,
    /// Source's `trak` box bytes — copied verbatim except for
    /// `tkhd.track_id` (rewritten to 2) and `stco`/`co64` (rewritten
    /// to point at our new mdat positions).
    trak_raw: Vec<u8>,
    /// Source's media-timescale duration in `mdhd.time_scale` units.
    duration_media: u64,
    /// Audio sample-table chunk layout from the source. Each entry is
    /// `(first_sample_index_0based, samples_in_chunk)`.
    chunks: Vec<(usize, usize)>,
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

        // Reconstruct chunk layout from stsc — same logic the patch
        // path in mux.rs uses. We need this to recompute stco offsets
        // when the audio samples land at fresh positions in our mdat.
        let chunks = reconstruct_chunk_layout(&track.trak_raw, track.samples.len())?;

        Ok(Some(AudioPassthrough {
            samples,
            trak_raw: track.trak_raw.clone(),
            duration_media: track.duration,
            chunks,
        }))
    }
}

// ─── HandBrake/x264 profile ──────────────────────────────────────────

/// HandBrake target signature constants. Pulled from
/// `docs/research/mp4-container-forensics-survey.md` §Q2 + Gloe 2014
/// Tab. 2-4 + Altinisik 2022 §IV.
mod handbrake {
    /// `ftyp` major brand.
    pub const FTYP_MAJOR: &[u8; 4] = b"isom";
    /// `ftyp` minor version.
    pub const FTYP_MINOR: u32 = 0x0000_0200;
    /// `ftyp` compatible brands, in order.
    pub const FTYP_COMPATIBLE: &[&[u8; 4]] =
        &[b"isom", b"iso2", b"avc1", b"mp41"];

    /// `mvhd.time_scale` — HandBrake convention.
    pub const MVHD_TIMESCALE: u32 = 1000;

    /// Per-track media `time_scale`, parametrised by frame rate. HandBrake
    /// emits 12800 @ 25 fps, 15360 @ 30 fps, 60000 @ 29.97. The general
    /// rule is "fps × 512" so a single-frame `stts` delta of 512 ticks
    /// gives a clean per-frame duration. (Per-fps caps documented in
    /// Agent B survey §Q2 "FFmpeg + libx264".)
    pub fn mdhd_timescale(fps_num: u32, fps_den: u32) -> u32 {
        // Frame rate × 512 → integer ticks per frame.
        // 30 fps → 15360 ticks; 25 → 12800; 29.97 (30000/1001) →
        // 30000 × 512 / 1001 = 15344. Round-half-to-even.
        let num = (fps_num as u64) * 512;
        let den = fps_den as u64;
        ((num + den / 2) / den) as u32
    }

    /// Per-frame media-timescale duration in `mdhd_timescale` units.
    pub const STTS_DELTA_PER_FRAME: u32 = 512;

    /// Plaintext `udta/©too` — version + build date marker. Matches the
    /// structure HandBrake emits ("HandBrake N.N.N (yyyymmddhhss)") but
    /// uses a stable phasm-side version pin so the same phasm build
    /// produces the same string. HandBrake's own version pool is wide
    /// enough that any pinned version lands inside the metaclass.
    pub const UDTA_TOO: &str = "HandBrake 1.7.3 2024010100";

    /// Video handler name.
    pub const HDLR_VIDE_NAME: &str = "VideoHandler";

    /// x264 SEI `user_data_unregistered` UUID — appears in every
    /// libx264-generated H.264 bitstream. The 16-byte UUID identifies the
    /// payload as the x264 settings/version blob; analysts (and most
    /// detectors) match on the UUID + the leading "x264 - core" prefix.
    /// See Agent B survey §Q2 + x264 source `encoder/encoder.c`
    /// (`x264_sei_version_write`).
    pub const X264_SEI_UUID: [u8; 16] = [
        0xDC, 0x45, 0xE9, 0xBD, 0xE6, 0xD9, 0x48, 0xB7,
        0x96, 0x2C, 0xD8, 0x20, 0xD9, 0x23, 0xEE, 0xEF,
    ];

    /// x264 SEI plaintext core string — the well-known "x264 - core …"
    /// banner. The exact suffix (build hash + options) is wide-pool in
    /// the wild; pinning it to a believable libx264 release leaves
    /// phasm output indistinguishable from any other libx264 transcode
    /// at the SEI plaintext level. Includes a NUL terminator (x264
    /// emits the C string null-terminated).
    pub const X264_SEI_PLAINTEXT: &str = concat!(
        "x264 - core 164 r3107 a8b68eb",
        " - H.264/MPEG-4 AVC codec",
        " - Copyleft 2003-2024",
        " - http://www.videolan.org/x264.html",
        " - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113",
        " me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1",
        " me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0",
        " deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2",
        " threads=6 lookahead_threads=1 sliced_threads=0 nr=0",
        " decimate=1 interlaced=0 bluray_compat=0",
        " constrained_intra=0 bframes=3 b_pyramid=2",
        " b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0",
        " weightp=2 keyint=250 keyint_min=25 scenecut=40",
        " intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1",
        " crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4",
        " ip_ratio=1.40 aq=1:1.00",
        "\0",
    );
}

#[allow(clippy::too_many_arguments)]
fn build_handbrake_x264(
    annex_b: &[u8],
    access_units: &[AccessUnit],
    avcc: &AvccData,
    width: u32,
    height: u32,
    timing: FrameTiming,
    composition_offsets: Option<&[i32]>,
    audio: Option<&AudioPassthrough<'_>>,
) -> Result<Vec<u8>, Mp4Error> {
    let mdhd_timescale = handbrake::mdhd_timescale(timing.fps_num, timing.fps_den);
    let frame_count = access_units.len() as u32;
    let track_duration_media: u64 =
        (frame_count as u64) * (handbrake::STTS_DELTA_PER_FRAME as u64);
    // Movie-header duration uses `MVHD_TIMESCALE` (1000 = ms).
    let movie_duration_ms: u64 = (frame_count as u64) * 1000 * (timing.fps_den as u64)
        / (timing.fps_num as u64);

    // Pre-synthesise the x264 SEI NAL — prepended to the FIRST IDR
    // sample's NAL list. x264/HandBrake emits this exactly once per
    // bitstream (at the leading IDR), and the plaintext "x264 - core"
    // suffix is the strongest single L4 fingerprint (Agent B survey
    // §"Tier B"). The phasm encoder does NOT emit this NAL itself —
    // injection is mux-time only.
    let x264_sei_nal = build_x264_sei_user_data_unregistered();
    let mut sei_consumed = false;

    // Build mdat sample data: one sample per access unit, NAL-length-
    // prefixed. AUD NALs are stripped (HandBrake convention). SPS/PPS
    // NALs are stripped (they live in avcC). On the first IDR sample,
    // prepend the synthesised x264 SEI before the slice NAL so analysts
    // see the canonical libx264 banner at the start of the bitstream.
    let mut sample_data: Vec<Vec<u8>> = Vec::with_capacity(access_units.len());
    for au in access_units {
        let mut buf = Vec::new();
        if au.is_sync && !sei_consumed {
            let len = x264_sei_nal.len() as u32;
            buf.extend_from_slice(&len.to_be_bytes());
            buf.extend_from_slice(&x264_sei_nal);
            sei_consumed = true;
        }
        for &(nal_start, nal_end, nal_type) in &au.nals {
            if nal_type == NalType::AUD {
                continue; // HandBrake/x264 strips AUDs in the AVC1 mdat.
            }
            // For IDR samples: drop SPS/PPS too (they live in avcC).
            if nal_type == NalType::SPS || nal_type == NalType::PPS {
                continue;
            }
            let nal = &annex_b[nal_start..nal_end];
            // 4-byte big-endian length prefix.
            let len = nal.len() as u32;
            buf.extend_from_slice(&len.to_be_bytes());
            buf.extend_from_slice(nal);
        }
        sample_data.push(buf);
    }

    let sample_sizes: Vec<u32> = sample_data.iter().map(|s| s.len() as u32).collect();
    let video_payload: u64 = sample_sizes.iter().map(|&s| s as u64).sum();
    let audio_payload: u64 = audio
        .map(|a| a.samples.iter().map(|s| s.len() as u64).sum::<u64>())
        .unwrap_or(0);
    let total_mdat_payload: u64 = video_payload + audio_payload;

    // ─── Compute layout ─────────────────────────────────────────
    //
    // HandBrake (no +faststart) layout: ftyp → mdat → moov.
    // mdat carries video samples first, then audio samples (when
    // present) — matches what `ffmpeg -c:a copy` interleaves into a
    // libx264 file at low frame counts.

    let ftyp = build_ftyp_handbrake();
    let ftyp_len = ftyp.len() as u64;

    let mdat_header_len: u64 = if total_mdat_payload + 8 > u32::MAX as u64 { 16 } else { 8 };
    let mdat_total_size = mdat_header_len + total_mdat_payload;
    let first_sample_offset = ftyp_len + mdat_header_len;

    // Per-video-sample absolute offsets in the output file.
    let mut sample_offsets: Vec<u64> = Vec::with_capacity(sample_data.len());
    let mut cursor = first_sample_offset;
    for &size in &sample_sizes {
        sample_offsets.push(cursor);
        cursor += size as u64;
    }
    // Per-audio-sample absolute offsets — sit right after the video.
    let mut audio_sample_offsets: Vec<u64> = Vec::new();
    if let Some(a) = audio {
        for s in &a.samples {
            audio_sample_offsets.push(cursor);
            cursor += s.len() as u64;
        }
    }

    // Sync sample indices (1-based per spec): each access unit whose
    // first VCL NAL is IDR.
    let sync_samples: Vec<u32> = access_units
        .iter()
        .enumerate()
        .filter_map(|(i, au)| if au.is_sync { Some(i as u32 + 1) } else { None })
        .collect();

    // ─── Patch the audio trak (when present) ────────────────────
    // Compute new chunk-offset table from the audio sample layout +
    // source's chunk grouping (stsc), then rewrite the source's
    // stco/co64 in-place inside its trak_raw clone. tkhd.track_id is
    // also rewritten to 2 (video has 1).
    let audio_trak_patched: Option<Vec<u8>> = audio.map(|a| {
        let mut trak = a.trak_raw.clone();
        let chunk_offsets: Vec<u64> = a
            .chunks
            .iter()
            .map(|&(first_idx, _spc)| {
                audio_sample_offsets
                    .get(first_idx)
                    .copied()
                    .unwrap_or(0)
            })
            .collect();
        // Patch stco/co64. Failures are non-fatal — we just leave
        // the source's offsets in place (they'd be wrong but the
        // file would at least decode video). Caller's smoke test
        // catches misshapen audio, then we'd return an error. For
        // now, propagate the error.
        patch_trak_stco(&mut trak, &chunk_offsets)?;
        patch_trak_track_id(&mut trak, 2)?;
        Ok::<Vec<u8>, Mp4Error>(trak)
    }).transpose()?;

    let audio_duration_movie: Option<u64> = audio.map(|a| {
        // Convert audio's mdhd-timescale duration to movie-timescale
        // (1000) so mvhd.duration covers the longest track.
        let mdhd_ts = read_audio_mdhd_timescale(&a.trak_raw).unwrap_or(48000) as u64;
        a.duration_media * (handbrake::MVHD_TIMESCALE as u64) / mdhd_ts
    });

    // ─── Build moov ─────────────────────────────────────────────
    let movie_duration_final = audio_duration_movie
        .map(|ad| ad.max(movie_duration_ms))
        .unwrap_or(movie_duration_ms);
    let moov = build_moov_handbrake(MoovParams {
        width,
        height,
        movie_duration_ms: movie_duration_final,
        track_duration_media,
        mdhd_timescale,
        sample_sizes: &sample_sizes,
        sample_offsets: &sample_offsets,
        sync_samples: &sync_samples,
        avcc,
        composition_offsets,
        audio_trak: audio_trak_patched.as_deref(),
    });

    // ─── Assemble output ────────────────────────────────────────
    let mut out = Vec::with_capacity(ftyp.len() + mdat_total_size as usize + moov.len());
    out.extend_from_slice(&ftyp);

    // mdat header
    if mdat_header_len == 16 {
        write_u32(&mut out, 1); // size = 1 → extended
        out.extend_from_slice(b"mdat");
        write_u64(&mut out, mdat_total_size);
    } else {
        write_u32(&mut out, mdat_total_size as u32);
        out.extend_from_slice(b"mdat");
    }
    for sample in &sample_data {
        out.extend_from_slice(sample);
    }
    if let Some(a) = audio {
        for s in &a.samples {
            out.extend_from_slice(s);
        }
    }
    out.extend_from_slice(&moov);

    Ok(out)
}

fn build_ftyp_handbrake() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(handbrake::FTYP_MAJOR);
    content.extend_from_slice(&handbrake::FTYP_MINOR.to_be_bytes());
    for &brand in handbrake::FTYP_COMPATIBLE {
        content.extend_from_slice(brand);
    }
    wrap_box(b"ftyp", &content)
}

struct MoovParams<'a> {
    width: u32,
    height: u32,
    movie_duration_ms: u64,
    track_duration_media: u64,
    mdhd_timescale: u32,
    sample_sizes: &'a [u32],
    sample_offsets: &'a [u64],
    sync_samples: &'a [u32],
    avcc: &'a AvccData,
    /// Per-sample composition-time offset (PTS - DTS) in
    /// `mdhd_timescale` units. `Some` when B-frames are present;
    /// `None` for IPPPP / I-only inputs.
    composition_offsets: Option<&'a [i32]>,
    /// §Stealth.L4.5 — patched audio `trak` box bytes ready to
    /// splice. Already has track_id=2 + new stco/co64 offsets. None
    /// → video-only mux.
    audio_trak: Option<&'a [u8]>,
}

fn build_moov_handbrake(p: MoovParams<'_>) -> Vec<u8> {
    let next_track_id: u32 = if p.audio_trak.is_some() { 3 } else { 2 };
    let mvhd = build_mvhd(p.movie_duration_ms, next_track_id);
    let trak = build_video_trak_handbrake(&p);
    let udta = build_udta_handbrake();

    let mut moov = Vec::new();
    moov.extend_from_slice(&mvhd);
    moov.extend_from_slice(&trak);
    if let Some(audio_trak) = p.audio_trak {
        moov.extend_from_slice(audio_trak);
    }
    moov.extend_from_slice(&udta);
    wrap_box(b"moov", &moov)
}

fn build_mvhd(duration_ms: u64, next_track_id: u32) -> Vec<u8> {
    let mut content = Vec::new();
    // version=0, flags=0
    content.extend_from_slice(&[0, 0, 0, 0]);
    // creation_time / modification_time (zeroed for determinism)
    content.extend_from_slice(&[0; 4]);
    content.extend_from_slice(&[0; 4]);
    // time_scale
    content.extend_from_slice(&handbrake::MVHD_TIMESCALE.to_be_bytes());
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

fn build_video_trak_handbrake(p: &MoovParams<'_>) -> Vec<u8> {
    let tkhd = build_tkhd_video(p.movie_duration_ms, p.width, p.height);
    let edts = build_edts_handbrake(p);
    let mdia = build_mdia_video_handbrake(p);
    let mut trak = Vec::new();
    trak.extend_from_slice(&tkhd);
    if let Some(edts) = edts {
        trak.extend_from_slice(&edts);
    }
    trak.extend_from_slice(&mdia);
    wrap_box(b"trak", &trak)
}

/// `edts/elst` for B-frame-aware streams.
///
/// **2026-05-05 fix**: real x264-medium output emits NO `edts/elst`
/// box. The original §Stealth.L4.2 follow-on (#146) emitted one with
/// `media_time = -min(ctts)` thinking it matched x264; in fact x264
/// relies on `ctts` version=1 signed offsets alone to encode B-frame
/// PTS reordering, with all DTS values non-negative and all PTS = DTS
/// + ctts non-negative. The leading-empty-edit phasm was emitting
/// instead instructed players to start playback past the IDR, dropping
/// the IDR from display output and producing a 1-frame
/// presentation-time shift that read as ~5–10 dB Y-PSNR vs source.
///
/// See `docs/design/video/_archive/h264/encoder-quality-perf-gap-2026-05-04.md`.
///
/// Kept around as `_unused` for diff readability — caller never invokes.
#[allow(dead_code)]
fn build_edts_handbrake(_p: &MoovParams<'_>) -> Option<Vec<u8>> {
    // x264 doesn't emit edts/elst. We don't either.
    None
}

fn build_tkhd_video(duration_ms: u64, width: u32, height: u32) -> Vec<u8> {
    let mut content = Vec::new();
    // version=0, flags=0x000007 (track_enabled | in_movie | in_preview)
    content.extend_from_slice(&[0, 0x00, 0x00, 0x07]);
    content.extend_from_slice(&[0; 4]); // creation_time
    content.extend_from_slice(&[0; 4]); // modification_time
    content.extend_from_slice(&1u32.to_be_bytes()); // track_ID
    content.extend_from_slice(&[0; 4]); // reserved
    content.extend_from_slice(&(duration_ms as u32).to_be_bytes());
    content.extend_from_slice(&[0; 8]); // reserved
    content.extend_from_slice(&[0; 2]); // layer
    content.extend_from_slice(&[0; 2]); // alternate_group
    content.extend_from_slice(&[0; 2]); // volume
    content.extend_from_slice(&[0; 2]); // reserved
    content.extend_from_slice(&UNITY_MATRIX);
    // width / height as 16.16 fixed-point
    content.extend_from_slice(&(width << 16).to_be_bytes());
    content.extend_from_slice(&(height << 16).to_be_bytes());
    wrap_box(b"tkhd", &content)
}

fn build_mdia_video_handbrake(p: &MoovParams<'_>) -> Vec<u8> {
    let mdhd = build_mdhd(p.track_duration_media, p.mdhd_timescale);
    let hdlr = build_hdlr_vide();
    let minf = build_minf_video_handbrake(p);
    let mut mdia = Vec::new();
    mdia.extend_from_slice(&mdhd);
    mdia.extend_from_slice(&hdlr);
    mdia.extend_from_slice(&minf);
    wrap_box(b"mdia", &mdia)
}

fn build_mdhd(duration: u64, timescale: u32) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&[0, 0, 0, 0]); // version=0, flags=0
    content.extend_from_slice(&[0; 4]); // creation_time
    content.extend_from_slice(&[0; 4]); // modification_time
    content.extend_from_slice(&timescale.to_be_bytes());
    content.extend_from_slice(&(duration as u32).to_be_bytes());
    // language: ISO-639-2 packed; "und" = 0x55C4
    content.extend_from_slice(&[0x55, 0xC4]);
    content.extend_from_slice(&[0; 2]); // pre_defined
    wrap_box(b"mdhd", &content)
}

fn build_hdlr_vide() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&[0, 0, 0, 0]); // version=0, flags=0
    content.extend_from_slice(&[0; 4]); // pre_defined
    content.extend_from_slice(b"vide");
    content.extend_from_slice(&[0; 12]); // reserved
    content.extend_from_slice(handbrake::HDLR_VIDE_NAME.as_bytes());
    content.push(0); // null terminator
    wrap_box(b"hdlr", &content)
}

fn build_minf_video_handbrake(p: &MoovParams<'_>) -> Vec<u8> {
    let vmhd = build_vmhd();
    let dinf = build_dinf();
    let stbl = build_stbl_video_handbrake(p);
    let mut minf = Vec::new();
    minf.extend_from_slice(&vmhd);
    minf.extend_from_slice(&dinf);
    minf.extend_from_slice(&stbl);
    wrap_box(b"minf", &minf)
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

fn build_stbl_video_handbrake(p: &MoovParams<'_>) -> Vec<u8> {
    let chunks = chunk_layout_per_gop(p.sample_sizes.len() as u32, p.sync_samples);
    let chunk_offsets = chunk_offsets_for_layout(&chunks, p.sample_offsets);

    let stsd = build_stsd_avc1(p.width, p.height, p.avcc);
    let stts = build_stts_uniform(p.sample_sizes.len() as u32);
    let ctts = p.composition_offsets.map(build_ctts);
    let stsc = build_stsc(&chunks);
    let stsz = build_stsz(p.sample_sizes);
    let stco = build_stco(&chunk_offsets);
    let stss = build_stss(p.sync_samples);

    // §Stealth.L4.3 — stbl child order matches HandBrake/libavformat:
    // stsd → stts → ctts (B-frames only) → stss → stsc → stsz → stco.
    // Yang/EVA decision trees treat stbl `PathOrder` as a fingerprint
    // input; the legacy `stss-after-stsz` order phasm shipped placed it
    // outside the libx264 metaclass.
    let mut stbl = Vec::new();
    stbl.extend_from_slice(&stsd);
    stbl.extend_from_slice(&stts);
    if let Some(ctts) = ctts {
        stbl.extend_from_slice(&ctts);
    }
    stbl.extend_from_slice(&stss);
    stbl.extend_from_slice(&stsc);
    stbl.extend_from_slice(&stsz);
    stbl.extend_from_slice(&stco);
    wrap_box(b"stbl", &stbl)
}

/// `ctts` (composition time-to-sample) box, version=1 with signed
/// offsets. One entry per sample (no run-length compression here —
/// HandBrake/x264 typically emits per-sample for IBPBP M=2 since
/// adjacent samples don't share offsets). offsets are in
/// `mdhd_timescale` units.
fn build_ctts(offsets: &[i32]) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&[1, 0, 0, 0]); // version=1, flags=0
    content.extend_from_slice(&(offsets.len() as u32).to_be_bytes());
    for &off in offsets {
        content.extend_from_slice(&1u32.to_be_bytes()); // sample_count = 1
        content.extend_from_slice(&off.to_be_bytes()); // signed i32
    }
    wrap_box(b"ctts", &content)
}

/// One chunk == one GOP. Chunk i contains all samples whose 1-based index
/// is in the half-open run `[sync_samples[i], sync_samples[i+1])` (with
/// the last chunk extending to `sample_count`). Returns `(chunk_index_0
/// → samples_in_chunk)` for every chunk in encode order. If
/// `sync_samples` is empty (no IDRs — degenerate), falls back to one
/// chunk holding all samples.
fn chunk_layout_per_gop(sample_count: u32, sync_samples: &[u32]) -> Vec<u32> {
    if sync_samples.is_empty() || sample_count == 0 {
        return vec![sample_count];
    }
    let mut chunks = Vec::with_capacity(sync_samples.len());
    for w in sync_samples.windows(2) {
        chunks.push(w[1] - w[0]);
    }
    chunks.push(sample_count + 1 - sync_samples[sync_samples.len() - 1]);
    chunks
}

/// First-sample file offset for each chunk, given per-sample offsets and
/// the chunk-size layout. The first sample of chunk `c` is sample index
/// `sum(chunks[0..c])` (0-based).
fn chunk_offsets_for_layout(chunks: &[u32], sample_offsets: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(chunks.len());
    let mut idx = 0usize;
    for &spc in chunks {
        if idx < sample_offsets.len() {
            out.push(sample_offsets[idx]);
        } else {
            out.push(0);
        }
        idx += spc as usize;
    }
    out
}

fn build_stsd_avc1(width: u32, height: u32, avcc: &AvccData) -> Vec<u8> {
    let avc1 = build_avc1(width, height, avcc);
    let mut content = Vec::new();
    content.extend_from_slice(&[0, 0, 0, 0]); // version+flags
    content.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    content.extend_from_slice(&avc1);
    wrap_box(b"stsd", &content)
}

fn build_avc1(width: u32, height: u32, avcc: &AvccData) -> Vec<u8> {
    // VisualSampleEntry layout per ISO/IEC 14496-12 §8.5.2.
    let mut content = Vec::new();
    content.extend_from_slice(&[0; 6]); // reserved
    content.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    content.extend_from_slice(&[0; 2]); // pre_defined
    content.extend_from_slice(&[0; 2]); // reserved
    content.extend_from_slice(&[0; 12]); // pre_defined[3]
    content.extend_from_slice(&(width as u16).to_be_bytes());
    content.extend_from_slice(&(height as u16).to_be_bytes());
    content.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // horizresolution = 72 dpi
    content.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // vertresolution = 72 dpi
    content.extend_from_slice(&[0; 4]); // reserved
    content.extend_from_slice(&1u16.to_be_bytes()); // frame_count
    content.extend_from_slice(&[0; 32]); // compressorname (32-byte Pascal string, all zeros)
    content.extend_from_slice(&0x0018u16.to_be_bytes()); // depth = 24
    content.extend_from_slice(&[0xFF, 0xFF]); // pre_defined = -1

    let avcc_box = wrap_box(b"avcC", &avcc.to_bytes());
    content.extend_from_slice(&avcc_box);

    wrap_box(b"avc1", &content)
}

/// VP.M.1 — stsd builder for AV1 video tracks. Mirror of
/// [`build_stsd_avc1`] but emits an `av01` sample entry wrapping an
/// `av1C` config box. Used by [`build_mp4_av1`] (VP.M.2+).
#[allow(dead_code)] // Used by build_mp4_av1 once VP.M.2 lands.
fn build_stsd_av01(width: u32, height: u32, av1c: &Av1cData) -> Vec<u8> {
    let av01 = build_av01(width, height, av1c);
    let mut content = Vec::new();
    content.extend_from_slice(&[0, 0, 0, 0]); // version+flags
    content.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    content.extend_from_slice(&av01);
    wrap_box(b"stsd", &content)
}

/// VP.M.1 — `av01` VisualSampleEntry builder. The ISO/IEC 14496-12
/// VisualSampleEntry layout (§ 8.5.2) is codec-agnostic — same 78
/// bytes as `avc1` — the only difference is the fourcc and the
/// trailing codec-specific config box (`av1C` for AV1 vs `avcC` for
/// H.264).
///
/// Per AV1-ISOBMFF § 2.2.1, the `av01` sample entry MAY also include
/// optional `colr` / `pasp` boxes. VP.M.1 emits only `av1C` for the
/// MVP; color VUI passthrough (VP.9) adds `colr` later.
#[allow(dead_code)] // Used by build_mp4_av1 once VP.M.2 lands.
fn build_av01(width: u32, height: u32, av1c: &Av1cData) -> Vec<u8> {
    // VisualSampleEntry layout per ISO/IEC 14496-12 §8.5.2.
    let mut content = Vec::new();
    content.extend_from_slice(&[0; 6]); // reserved
    content.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    content.extend_from_slice(&[0; 2]); // pre_defined
    content.extend_from_slice(&[0; 2]); // reserved
    content.extend_from_slice(&[0; 12]); // pre_defined[3]
    content.extend_from_slice(&(width as u16).to_be_bytes());
    content.extend_from_slice(&(height as u16).to_be_bytes());
    content.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // horizresolution = 72 dpi
    content.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // vertresolution = 72 dpi
    content.extend_from_slice(&[0; 4]); // reserved
    content.extend_from_slice(&1u16.to_be_bytes()); // frame_count
    content.extend_from_slice(&[0; 32]); // compressorname (32-byte Pascal string, all zeros)
    content.extend_from_slice(&0x0018u16.to_be_bytes()); // depth = 24
    content.extend_from_slice(&[0xFF, 0xFF]); // pre_defined = -1

    let av1c_box = wrap_box(b"av1C", &av1c.to_bytes());
    content.extend_from_slice(&av1c_box);

    wrap_box(b"av01", &content)
}

fn build_stts_uniform(sample_count: u32) -> Vec<u8> {
    // One run: sample_count samples each with delta = STTS_DELTA_PER_FRAME.
    let mut content = Vec::new();
    content.extend_from_slice(&[0, 0, 0, 0]); // version+flags
    content.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    content.extend_from_slice(&sample_count.to_be_bytes());
    content.extend_from_slice(&handbrake::STTS_DELTA_PER_FRAME.to_be_bytes());
    wrap_box(b"stts", &content)
}

fn build_stsc(chunk_sizes: &[u32]) -> Vec<u8> {
    // Compress runs of equal-sized chunks into a single stsc entry.
    // HandBrake convention: one chunk per GOP, so equal-size GOPs
    // collapse to a single entry — the most common shape.
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

fn build_stsz(sample_sizes: &[u32]) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&[0, 0, 0, 0]); // version+flags
    content.extend_from_slice(&0u32.to_be_bytes()); // sample_size = 0 → variable
    content.extend_from_slice(&(sample_sizes.len() as u32).to_be_bytes());
    for &s in sample_sizes {
        content.extend_from_slice(&s.to_be_bytes());
    }
    wrap_box(b"stsz", &content)
}

fn build_stco(chunk_offsets: &[u64]) -> Vec<u8> {
    // One offset per chunk (HandBrake: one chunk per GOP). Use stco when
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

fn build_udta_handbrake() -> Vec<u8> {
    // udta/©too — QuickTime classic user-data string format
    // (§ "User-Data Atoms"): length(u16 BE, NOT counting these 4
    // header bytes) + language(u16 BE, 0 = unspecified) + UTF-8 string
    // bytes. Real HandBrake matches this layout exactly.
    let mut too_content = Vec::new();
    let s = handbrake::UDTA_TOO.as_bytes();
    too_content.extend_from_slice(&(s.len() as u16).to_be_bytes());
    too_content.extend_from_slice(&0u16.to_be_bytes()); // language = unspecified
    too_content.extend_from_slice(s);
    let too_box = wrap_box(&[0xA9, b't', b'o', b'o'], &too_content);
    wrap_box(b"udta", &too_box)
}

// ──────────────────────────────────────────────────────────────────────
//  VP.M.3 — AV1-in-MP4 orchestrator
// ──────────────────────────────────────────────────────────────────────

/// VP.M.3 — top-level entry point: encode an AV1 OBU byte stream
/// into a valid AV1-in-MP4 (av01) file ready for distribution. Used
/// by [`crate::codec::av1`] consumers to wrap rav1e output.
///
/// **Per VP decision D2 (locked 2026-06-06)**: single cross-platform
/// Rust muxer. No platform-specific paths (no AVAssetWriter, no
/// MediaMuxer).
///
/// **Scope**: video-only mux. Audio passthrough (when input video
/// has audio tracks) is the iOS/Android app layer's responsibility
/// (VP.3 / VP.6).
///
/// **Pipeline**:
///
/// 1. [`super::av1_obu_split::split_av1_into_samples`] partitions the
///    OBU stream into a `sequence_header_obu` + per-frame samples +
///    sync flags.
/// 2. The SH OBU populates `av1C` via
///    [`Av1cData::default_yuv420_8bit`] (assumes 4:2:0 8-bit BT.709 —
///    VP.M's MVP; VP.8 will add proper SH parsing).
/// 3. Sample byte sizes + offsets are computed assuming
///    ftyp → mdat → moov layout (HandBrake convention).
/// 4. The full mp4 byte stream is assembled and returned.
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
    build_mp4_av1_inner(av1_obus, width, height, timing, None)
}

/// VP.M.4 — same as [`build_mp4_av1`] but also passes through the
/// first audio track from `source_mp4`. The source's audio `trak` box
/// is copied verbatim modulo `track_id` (renumbered to 2) and
/// `stco`/`co64` (rewritten to point at the audio sample positions in
/// the new mdat). Codec-agnostic — AAC, Opus, anything the source
/// carries works the same way because we never touch the audio
/// codec-config record (`mp4a`/`esds`, `Opus`/`dops`, etc.).
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
    build_mp4_av1_inner(av1_obus, width, height, timing, audio.as_ref())
}

fn build_mp4_av1_inner(
    av1_obus: &[u8],
    width: u32,
    height: u32,
    timing: FrameTiming,
    audio: Option<&AudioPassthrough<'_>>,
) -> Result<Vec<u8>, Mp4Error> {
    use super::av1_obu_split::split_av1_into_samples;

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

    // 1. Split + extract SH.
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
    // VP.8: try the real parser first; fall back to default profile
    // (4:2:0 8-bit BT.709, level 4.0) if the parser doesn't recognize
    // the OBU syntax. Default is correct for phasm-rav1e's standard
    // output; the parser improves accuracy for non-standard streams
    // (different profile / bit depth / chroma config).
    let av1c = Av1cData::from_sequence_header_obu(split.sequence_header_obu.clone())
        .unwrap_or_else(|| Av1cData::default_yuv420_8bit(split.sequence_header_obu));

    // 2. Sample layout — video first, then audio (when present).
    let sample_sizes: Vec<u32> = split.samples.iter().map(|s| s.len() as u32).collect();
    let video_payload: u64 = sample_sizes.iter().map(|&s| s as u64).sum();
    let audio_payload: u64 = audio
        .map(|a| a.samples.iter().map(|s| s.len() as u64).sum::<u64>())
        .unwrap_or(0);
    let total_mdat_payload: u64 = video_payload + audio_payload;

    // 3. ftyp + mdat header sizing.
    let ftyp = build_ftyp_av1();
    let mdat_header_len: u64 = if total_mdat_payload + 8 > u32::MAX as u64 { 16 } else { 8 };
    let first_sample_offset = ftyp.len() as u64 + mdat_header_len;

    // 4. Per-video-sample absolute file offsets.
    let mut sample_offsets: Vec<u64> = Vec::with_capacity(sample_sizes.len());
    let mut cursor = first_sample_offset;
    for &size in &sample_sizes {
        sample_offsets.push(cursor);
        cursor += size as u64;
    }
    // Per-audio-sample absolute offsets — sit right after the video.
    let mut audio_sample_offsets: Vec<u64> = Vec::new();
    if let Some(a) = audio {
        for s in &a.samples {
            audio_sample_offsets.push(cursor);
            cursor += s.len() as u64;
        }
    }

    // 5. Sync sample indices (1-based per ISO BMFF stss).
    let sync_samples: Vec<u32> = split
        .sync
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| if s { Some(i as u32 + 1) } else { None })
        .collect();

    // 6. Patch the audio trak when present — rewrite stco/co64 + track_id.
    // Same mechanism the §Stealth.L4.5 H.264 audio passthrough uses
    // (see `build_handbrake_x264`).
    let audio_trak_patched: Option<Vec<u8>> = audio.map(|a| {
        let mut trak = a.trak_raw.clone();
        let chunk_offsets: Vec<u64> = a
            .chunks
            .iter()
            .map(|&(first_idx, _spc)| {
                audio_sample_offsets
                    .get(first_idx)
                    .copied()
                    .unwrap_or(0)
            })
            .collect();
        patch_trak_stco(&mut trak, &chunk_offsets)?;
        patch_trak_track_id(&mut trak, 2)?;
        Ok::<Vec<u8>, Mp4Error>(trak)
    }).transpose()?;

    // 7. moov.
    let mdhd_timescale = handbrake::mdhd_timescale(timing.fps_num, timing.fps_den);
    let frame_count = sample_sizes.len() as u32;
    let track_duration_media: u64 =
        (frame_count as u64) * (handbrake::STTS_DELTA_PER_FRAME as u64);
    let movie_duration_ms_video: u64 =
        (frame_count as u64) * 1000 * (timing.fps_den as u64) / (timing.fps_num as u64);
    // mvhd.duration must cover the longest track. Convert the audio
    // track's mdhd-timescale duration into movie-timescale (1000 ms/s)
    // and take the max.
    let movie_duration_ms = match audio {
        Some(a) => {
            let mdhd_ts = read_audio_mdhd_timescale(&a.trak_raw).unwrap_or(48000) as u64;
            let audio_ms = a.duration_media * (handbrake::MVHD_TIMESCALE as u64) / mdhd_ts;
            movie_duration_ms_video.max(audio_ms)
        }
        None => movie_duration_ms_video,
    };

    let moov = build_moov_av1(MoovParamsAv1 {
        width,
        height,
        movie_duration_ms,
        track_duration_media,
        mdhd_timescale,
        sample_sizes: &sample_sizes,
        sample_offsets: &sample_offsets,
        sync_samples: &sync_samples,
        av1c: &av1c,
        audio_trak: audio_trak_patched.as_deref(),
    });

    // 8. Assemble.
    let total_size = ftyp.len() as u64
        + mdat_header_len
        + total_mdat_payload
        + moov.len() as u64;
    let mut out = Vec::with_capacity(total_size as usize);
    out.extend_from_slice(&ftyp);

    // mdat header — 8-byte for ≤ ~4 GiB payload, 16-byte extended for larger.
    if mdat_header_len == 8 {
        out.extend_from_slice(&((total_mdat_payload + 8) as u32).to_be_bytes());
        out.extend_from_slice(b"mdat");
    } else {
        out.extend_from_slice(&1u32.to_be_bytes()); // size=1 signals extended u64 length
        out.extend_from_slice(b"mdat");
        out.extend_from_slice(&(total_mdat_payload + 16).to_be_bytes());
    }

    // Video samples, then audio samples.
    for s in &split.samples {
        out.extend_from_slice(s);
    }
    if let Some(a) = audio {
        for s in &a.samples {
            out.extend_from_slice(s);
        }
    }

    // moov.
    out.extend_from_slice(&moov);

    Ok(out)
}

/// `ftyp` for AV1-in-MP4. Major brand = `isom`; compatible brands
/// include `av01` to advertise AV1 capability per
/// AV1-ISOBMFF § 2.1. Modern decoders use the major brand for
/// container compatibility checks; av01 in compatible_brands
/// signals "this file contains AV1 video".
fn build_ftyp_av1() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(b"isom"); // major_brand
    content.extend_from_slice(&0x0000_0200u32.to_be_bytes()); // minor_version
    for brand in &[b"isom", b"iso6", b"av01", b"mp41"] {
        content.extend_from_slice(*brand);
    }
    wrap_box(b"ftyp", &content)
}

/// Parallel of [`MoovParams`] for AV1 — carries an `av1C` config
/// instead of `avcC`, drops `composition_offsets` (AV1 stego is IPPPP
/// always; no B-frames). VP.M.4: gained `audio_trak` for source-MP4
/// audio passthrough.
struct MoovParamsAv1<'a> {
    width: u32,
    height: u32,
    movie_duration_ms: u64,
    track_duration_media: u64,
    mdhd_timescale: u32,
    sample_sizes: &'a [u32],
    sample_offsets: &'a [u64],
    sync_samples: &'a [u32],
    av1c: &'a Av1cData,
    /// VP.M.4 — pre-patched audio `trak` box bytes (track_id=2 +
    /// rewritten stco/co64 offsets). None → video-only mux.
    audio_trak: Option<&'a [u8]>,
}

fn build_moov_av1(p: MoovParamsAv1<'_>) -> Vec<u8> {
    // next_track_ID: 2 video-only; 3 with audio (track 1 = video,
    // track 2 = audio, next allocates 3).
    let next_track_id: u32 = if p.audio_trak.is_some() { 3 } else { 2 };
    let mvhd = build_mvhd(p.movie_duration_ms, next_track_id);
    let trak = build_video_trak_av1(&p);
    let udta = build_udta_handbrake(); // Reuse — UDTA is codec-agnostic.

    let mut moov = Vec::new();
    moov.extend_from_slice(&mvhd);
    moov.extend_from_slice(&trak);
    if let Some(audio_trak) = p.audio_trak {
        moov.extend_from_slice(audio_trak);
    }
    moov.extend_from_slice(&udta);
    wrap_box(b"moov", &moov)
}

fn build_video_trak_av1(p: &MoovParamsAv1<'_>) -> Vec<u8> {
    let tkhd = build_tkhd_video(p.movie_duration_ms, p.width, p.height);
    let mdia = build_mdia_video_av1(p);
    let mut trak = Vec::new();
    trak.extend_from_slice(&tkhd);
    trak.extend_from_slice(&mdia);
    wrap_box(b"trak", &trak)
}

fn build_mdia_video_av1(p: &MoovParamsAv1<'_>) -> Vec<u8> {
    let mdhd = build_mdhd(p.track_duration_media, p.mdhd_timescale);
    let hdlr = build_hdlr_vide();
    let minf = build_minf_video_av1(p);
    let mut mdia = Vec::new();
    mdia.extend_from_slice(&mdhd);
    mdia.extend_from_slice(&hdlr);
    mdia.extend_from_slice(&minf);
    wrap_box(b"mdia", &mdia)
}

fn build_minf_video_av1(p: &MoovParamsAv1<'_>) -> Vec<u8> {
    let vmhd = build_vmhd();
    let dinf = build_dinf();
    let stbl = build_stbl_av01(p);
    let mut minf = Vec::new();
    minf.extend_from_slice(&vmhd);
    minf.extend_from_slice(&dinf);
    minf.extend_from_slice(&stbl);
    wrap_box(b"minf", &minf)
}

/// AV1 stbl — same shape as [`build_stbl_video_handbrake`] but emits
/// an `av01` sample entry, no `ctts` (AV1 stego is IPPPP / no
/// B-frames), and uses [`chunk_layout_per_gop`] keyed off the AV1
/// sync sample list.
fn build_stbl_av01(p: &MoovParamsAv1<'_>) -> Vec<u8> {
    let chunks = chunk_layout_per_gop(p.sample_sizes.len() as u32, p.sync_samples);
    let chunk_offsets = chunk_offsets_for_layout(&chunks, p.sample_offsets);

    let stsd = build_stsd_av01(p.width, p.height, p.av1c);
    let stts = build_stts_uniform(p.sample_sizes.len() as u32);
    let stsc = build_stsc(&chunks);
    let stsz = build_stsz(p.sample_sizes);
    let stco = build_stco(&chunk_offsets);
    let stss = build_stss(p.sync_samples);

    // stbl child order matches the H.264 path (HandBrake/libavformat
    // convention): stsd → stts → stss → stsc → stsz → stco.
    let mut stbl = Vec::new();
    stbl.extend_from_slice(&stsd);
    stbl.extend_from_slice(&stts);
    stbl.extend_from_slice(&stss);
    stbl.extend_from_slice(&stsc);
    stbl.extend_from_slice(&stsz);
    stbl.extend_from_slice(&stco);
    wrap_box(b"stbl", &stbl)
}

// ─── SEI synthesis ────────────────────────────────────────────────────

/// Synthesise an x264-class `user_data_unregistered` SEI NAL carrying the
/// 16-byte UUID + plaintext core string. Returns the NAL bytes including
/// the 1-byte NAL header (no Annex-B start code, no length prefix —
/// caller adds whichever envelope it needs).
///
/// SEI NAL layout (ITU-T H.264 §7.3.2.3):
///   nal_header (1 byte)               = 0x06   (type=6, ref_idc=0)
///   ┌─ for each SEI message: ─────┐
///   │  payload_type bytes          │   user_data_unregistered = 5
///   │  payload_size bytes          │   length of the payload that follows
///   │  payload (UUID + data)       │
///   └─────────────────────────────┘
///   rbsp_trailing_bits             (single 1 bit + zero-pad to byte)
///
/// The payload_type/payload_size encoding is "0xFF * (n / 0xFF) + (n %
/// 0xFF)" — successive 0xFF bytes accumulate, then a final byte < 0xFF
/// gives the remainder. Most short SEIs (≤ 0xFE bytes) emit one byte.
///
/// Emulation-prevention is needed but extremely rare in this payload:
/// our plaintext has no `00 00 ≤01` runs, but to stay correct under any
/// future plaintext, we apply EP after the NAL header. The output is the
/// final NALU bytes.
fn build_x264_sei_user_data_unregistered() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&handbrake::X264_SEI_UUID);
    payload.extend_from_slice(handbrake::X264_SEI_PLAINTEXT.as_bytes());

    let mut rbsp = Vec::new();
    rbsp.push(0x06); // NAL header: nal_ref_idc=0, nal_unit_type=6 (SEI)

    // payload_type = 5 (user_data_unregistered) — fits in one byte.
    rbsp.push(5);

    // payload_size encoding (sect. 7.3.2.3.1):
    let mut remaining = payload.len();
    while remaining >= 0xFF {
        rbsp.push(0xFF);
        remaining -= 0xFF;
    }
    rbsp.push(remaining as u8);

    rbsp.extend_from_slice(&payload);

    // RBSP trailing bits: rbsp_stop_one_bit (1) + zero-pad to byte.
    // Since the payload above is byte-aligned, this is a single 0x80.
    rbsp.push(0x80);

    add_emulation_prevention(&rbsp)
}

/// Insert emulation-prevention `0x03` bytes into RBSP per H.264 §7.4.1.1.
/// Required for any byte stream that may contain `00 00 ≤02` patterns.
/// Operates on the NAL payload (does NOT touch the leading header byte
/// at index 0).
fn add_emulation_prevention(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len());
    if !rbsp.is_empty() {
        out.push(rbsp[0]);
    }
    let mut zero_run = 0usize;
    for &b in &rbsp[1..] {
        if zero_run >= 2 && b <= 0x03 {
            out.push(0x03);
            zero_run = 0;
        }
        out.push(b);
        if b == 0 {
            zero_run += 1;
        } else {
            zero_run = 0;
        }
    }
    out
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

fn find_subbox_offset(buf: &[u8], target: &[u8; 4]) -> Option<usize> {
    fn recurse(buf: &[u8], start: usize, end: usize, target: &[u8; 4]) -> Option<usize> {
        let mut found = None;
        let _ = super::iterate_boxes(buf, start, end, |h, content_start, _| {
            if found.is_some() {
                return Ok(());
            }
            let box_start = content_start - h.header_len as usize;
            if h.box_type == *target {
                found = Some(box_start);
            } else if matches!(
                &h.box_type,
                b"trak" | b"mdia" | b"minf" | b"stbl" | b"edts"
            ) {
                let inner_end = box_start + h.size as usize;
                if let Some(inner) = recurse(buf, content_start, inner_end, target) {
                    found = Some(inner);
                }
            }
            Ok(())
        });
        found
    }
    recurse(buf, 0, buf.len(), target)
}

/// Reconstruct `(first_sample_idx_0based, samples_per_chunk)` for
/// every chunk in an audio `trak`'s sample table. Mirrors the same
/// derivation `super::mux::compute_chunk_first_samples` does, but
/// reads from a standalone trak buffer instead of the moov.
fn reconstruct_chunk_layout(
    trak: &[u8],
    n_samples: usize,
) -> Result<Vec<(usize, usize)>, Mp4Error> {
    if n_samples == 0 {
        return Ok(Vec::new());
    }
    // Read num_chunks from stco or co64.
    let (num_chunks, is_co64) = if let Some(off) = find_subbox_offset(trak, b"stco") {
        let h = super::parse_box_header(trak, off)?;
        let cs = off + h.header_len as usize;
        (super::read_u32(trak, cs + 4)? as usize, false)
    } else if let Some(off) = find_subbox_offset(trak, b"co64") {
        let h = super::parse_box_header(trak, off)?;
        let cs = off + h.header_len as usize;
        (super::read_u32(trak, cs + 4)? as usize, true)
    } else {
        return Ok(Vec::new());
    };
    let _ = is_co64;

    let stsc_entries = if let Some(off) = find_subbox_offset(trak, b"stsc") {
        let h = super::parse_box_header(trak, off)?;
        let cs = off + h.header_len as usize;
        super::demux::parse_stsc(trak, cs)?
    } else {
        Vec::new()
    };

    let mut samples_per_chunk = vec![0u32; num_chunks];
    if stsc_entries.is_empty() {
        samples_per_chunk.fill(1);
    } else {
        for (i, entry) in stsc_entries.iter().enumerate() {
            let first_chunk = entry.0 as usize;
            let spc = entry.1;
            let next_first = if i + 1 < stsc_entries.len() {
                stsc_entries[i + 1].0 as usize
            } else {
                num_chunks + 1
            };
            for chunk_idx in first_chunk..next_first {
                if (1..=num_chunks).contains(&chunk_idx) {
                    samples_per_chunk[chunk_idx - 1] = spc;
                }
            }
        }
    }

    let mut chunks = Vec::with_capacity(num_chunks);
    let mut sample_idx = 0usize;
    for &spc in &samples_per_chunk {
        let count = spc as usize;
        chunks.push((sample_idx, count));
        sample_idx += count;
    }
    Ok(chunks)
}

/// Rewrite `stco` (32-bit) or `co64` (64-bit) chunk-offset entries
/// inside a `trak` buffer in place. `new_chunk_offsets` must have
/// exactly `count(stco/co64.entries)` entries — the source's chunk
/// count is preserved.
fn patch_trak_stco(trak: &mut [u8], new_chunk_offsets: &[u64]) -> Result<(), Mp4Error> {
    if let Some(off) = find_subbox_offset(trak, b"stco") {
        let h = super::parse_box_header(trak, off)?;
        let cs = off + h.header_len as usize;
        let count = super::read_u32(trak, cs + 4)? as usize;
        if count != new_chunk_offsets.len() {
            return Err(Mp4Error::InvalidBox(format!(
                "stco entry count mismatch in audio trak: {} vs {}",
                count,
                new_chunk_offsets.len()
            )));
        }
        if new_chunk_offsets.iter().any(|&o| o > u32::MAX as u64) {
            return Err(Mp4Error::InvalidBox(
                "audio trak stco offsets exceed 32-bit; co64 upgrade not supported".into(),
            ));
        }
        for (i, &offset) in new_chunk_offsets.iter().enumerate() {
            let pos = cs + 8 + i * 4;
            trak[pos..pos + 4].copy_from_slice(&(offset as u32).to_be_bytes());
        }
        return Ok(());
    }
    if let Some(off) = find_subbox_offset(trak, b"co64") {
        let h = super::parse_box_header(trak, off)?;
        let cs = off + h.header_len as usize;
        let count = super::read_u32(trak, cs + 4)? as usize;
        if count != new_chunk_offsets.len() {
            return Err(Mp4Error::InvalidBox(format!(
                "co64 entry count mismatch in audio trak: {} vs {}",
                count,
                new_chunk_offsets.len()
            )));
        }
        for (i, &offset) in new_chunk_offsets.iter().enumerate() {
            let pos = cs + 8 + i * 8;
            trak[pos..pos + 8].copy_from_slice(&offset.to_be_bytes());
        }
        return Ok(());
    }
    Err(Mp4Error::InvalidBox("audio trak missing stco/co64".into()))
}

/// Rewrite `tkhd.track_id` inside a `trak` buffer in place. Used to
/// renumber the audio track to 2 (video stays at 1).
fn patch_trak_track_id(trak: &mut [u8], new_track_id: u32) -> Result<(), Mp4Error> {
    let off = find_subbox_offset(trak, b"tkhd")
        .ok_or_else(|| Mp4Error::InvalidBox("audio trak missing tkhd".into()))?;
    let h = super::parse_box_header(trak, off)?;
    let cs = off + h.header_len as usize;
    let version = trak[cs];
    let track_id_pos = if version == 1 {
        cs + 4 + 16 // ver+flags(4) + creation(8) + modification(8)
    } else {
        cs + 4 + 8 // ver+flags(4) + creation(4) + modification(4)
    };
    trak[track_id_pos..track_id_pos + 4].copy_from_slice(&new_track_id.to_be_bytes());
    Ok(())
}

/// Read `mdhd.time_scale` from a `trak` buffer (used to convert the
/// audio track's media-timescale duration to mvhd-timescale ms when
/// computing the new movie duration).
fn read_audio_mdhd_timescale(trak: &[u8]) -> Option<u32> {
    let off = find_subbox_offset(trak, b"mdhd")?;
    let h = super::parse_box_header(trak, off).ok()?;
    let cs = off + h.header_len as usize;
    let version = trak[cs];
    let ts_pos = if version == 1 {
        cs + 4 + 16 // ver+flags + creation(8) + modification(8)
    } else {
        cs + 4 + 8 // ver+flags + creation(4) + modification(4)
    };
    super::read_u32(trak, ts_pos).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::mp4::{is_mp4, parse_box_header};

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

    #[test]
    fn build_av01_box_well_formed() {
        let av1c = Av1cData::default_yuv420_8bit(fake_sh_obu());
        let av01 = build_av01(1920, 1080, &av1c);

        // First 4 bytes = box size (big-endian u32).
        let box_size =
            u32::from_be_bytes([av01[0], av01[1], av01[2], av01[3]]) as usize;
        assert_eq!(box_size, av01.len(), "size field must match actual length");

        // Next 4 bytes = "av01" fourcc.
        assert_eq!(&av01[4..8], b"av01", "fourcc must be av01");

        // VisualSampleEntry header is 78 bytes after fourcc (per § 8.5.2).
        // At byte 8 we have: 6 reserved + 2 data_ref_idx + 2 pre_defined
        // + 2 reserved + 12 pre_defined[3] = 24 codec-agnostic
        // before width. Width is bytes 32-33 of the box. Big-endian u16.
        let width =
            u16::from_be_bytes([av01[32], av01[33]]) as u32;
        let height =
            u16::from_be_bytes([av01[34], av01[35]]) as u32;
        assert_eq!(width, 1920, "width must be encoded at expected offset");
        assert_eq!(height, 1080, "height must be encoded at expected offset");

        // av1C box must appear after the 78-byte VisualSampleEntry
        // header (which lives at bytes [8..86]).
        assert_eq!(&av01[86 + 4..86 + 8], b"av1C", "av1C box must follow VisualSampleEntry header");
    }

    #[test]
    fn build_stsd_av01_well_formed() {
        let av1c = Av1cData::default_yuv420_8bit(fake_sh_obu());
        let stsd = build_stsd_av01(1280, 720, &av1c);

        // Box header: size + "stsd".
        let box_size =
            u32::from_be_bytes([stsd[0], stsd[1], stsd[2], stsd[3]]) as usize;
        assert_eq!(box_size, stsd.len());
        assert_eq!(&stsd[4..8], b"stsd", "fourcc must be stsd");

        // Then version+flags (4 bytes of zero), then entry_count = 1.
        assert_eq!(&stsd[8..12], &[0, 0, 0, 0], "version+flags");
        assert_eq!(
            u32::from_be_bytes([stsd[12], stsd[13], stsd[14], stsd[15]]),
            1,
            "entry_count must be 1"
        );

        // Then the av01 entry starts at byte 16.
        assert_eq!(&stsd[16 + 4..16 + 8], b"av01", "av01 entry must follow stsd header");
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
    fn ftyp_handbrake_is_byte_exact() {
        let ftyp = build_ftyp_handbrake();
        // 8-byte header + 8 content bytes (major + minor) + 16 compatibles
        assert_eq!(ftyp.len(), 8 + 8 + 16);
        // size + 'ftyp'
        assert_eq!(&ftyp[0..4], &(ftyp.len() as u32).to_be_bytes());
        assert_eq!(&ftyp[4..8], b"ftyp");
        // major_brand
        assert_eq!(&ftyp[8..12], b"isom");
        // minor_version (HandBrake = 0x0000_0200)
        assert_eq!(&ftyp[12..16], &[0x00, 0x00, 0x02, 0x00]);
        // compatibles: isom, iso2, avc1, mp41
        assert_eq!(&ftyp[16..20], b"isom");
        assert_eq!(&ftyp[20..24], b"iso2");
        assert_eq!(&ftyp[24..28], b"avc1");
        assert_eq!(&ftyp[28..32], b"mp41");
    }

    #[test]
    fn build_mp4_round_trips_minimal_annexb() {
        let annex_b = build_minimal_annexb();
        let mp4 = build_mp4(
            MuxerProfile::HandbrakeX264,
            &annex_b,
            1920,
            1080,
            FrameTiming::FPS_30,
        )
        .expect("build_mp4 succeeds");

        assert!(is_mp4(&mp4), "output is recognized as MP4");

        // Top-level layout: ftyp → mdat → moov.
        let h0 = parse_box_header(&mp4, 0).unwrap();
        assert_eq!(h0.box_type, *b"ftyp");
        let h1 = parse_box_header(&mp4, h0.size as usize).unwrap();
        assert_eq!(h1.box_type, *b"mdat");
        let moov_off = h0.size as usize + h1.size as usize;
        let h2 = parse_box_header(&mp4, moov_off).unwrap();
        assert_eq!(h2.box_type, *b"moov");
        assert_eq!(moov_off + h2.size as usize, mp4.len());
    }

    #[test]
    fn build_mp4_demuxes_back_to_one_video_track() {
        use crate::codec::mp4::demux::demux;
        let annex_b = build_minimal_annexb();
        let mp4 = build_mp4(
            MuxerProfile::HandbrakeX264,
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
        // mdhd timescale: 30 × 512 = 15360 (HandBrake @ 30 fps).
        assert_eq!(track.timescale, 15360);
    }

    #[test]
    fn handbrake_mdhd_timescale_30fps() {
        assert_eq!(handbrake::mdhd_timescale(30, 1), 15360);
    }

    #[test]
    fn stbl_child_order_matches_handbrake() {
        // §Stealth.L4.3 — stbl children must appear in
        // stsd → stts → stss → stsc → stsz → stco order (no ctts on
        // IPPPP). Yang/EVA decision trees fingerprint this ordering.
        let annex_b = build_minimal_annexb();
        let mp4 = build_mp4(
            MuxerProfile::HandbrakeX264,
            &annex_b,
            1920,
            1080,
            FrameTiming::FPS_30,
        )
        .unwrap();

        // Walk into moov → trak → mdia → minf → stbl and collect child
        // 4-CCs in order.
        let order = collect_stbl_child_order(&mp4);
        let expected: &[&[u8; 4]] = &[b"stsd", b"stts", b"stss", b"stsc", b"stsz", b"stco"];
        assert_eq!(
            order.iter().map(|s| *s).collect::<Vec<_>>(),
            expected.iter().map(|s| **s).collect::<Vec<_>>(),
        );
    }

        #[cfg(feature = "h264-encoder")]
        #[test]
    fn stbl_child_order_with_ctts_inserts_after_stts() {
        // With B-frames present, stbl order is
        // stsd → stts → ctts → stss → stsc → stsz → stco.
        use crate::codec::h264::stego::gop_pattern::GopPattern;
        // 5-frame Annex-B (encode order I P B P B). Synthesise NAL
        // sequence by hand — content doesn't have to decode, just
        // parse as 5 access units.
        let mut bytes = Vec::new();
        // AU 0 — IDR (with SPS+PPS prefix).
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x10]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x67, 0x64, 0x00, 0x1E, 0xAC, 0xD9, 0x40, 0x40, 0x3C, 0x80]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xEB, 0xE3, 0xCB]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00]);
        // AUs 1-4 — non-IDR slices.
        for _ in 0..4 {
            bytes.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x30]);
            bytes.extend_from_slice(&[0, 0, 0, 1, 0x41, 0x9A, 0x00]);
        }
        let mp4 = build_mp4_with_pattern(
            MuxerProfile::HandbrakeX264,
            &bytes,
            1920,
            1080,
            FrameTiming::FPS_30,
            GopPattern::Ibpbp { gop: 5, b_count: 1 },
            5,
        )
        .unwrap();
        let order = collect_stbl_child_order(&mp4);
        let expected: &[&[u8; 4]] =
            &[b"stsd", b"stts", b"ctts", b"stss", b"stsc", b"stsz", b"stco"];
        assert_eq!(
            order.iter().map(|s| *s).collect::<Vec<_>>(),
            expected.iter().map(|s| **s).collect::<Vec<_>>(),
        );
    }

    /// §Stealth.L4.2.fix regression test (2026-05-05).
    ///
    /// Phasm MUST NOT emit `edts/elst` boxes — real x264-medium output
    /// has none, and a leading-empty-edit shifts the IDR's presentation
    /// time to a negative value (or past the start) that ffmpeg trims
    /// from display, causing a 1-frame frame-pairing offset that read
    /// as a 7+ dB Y-PSNR cliff in stego measurements. CTTS version=1
    /// signed offsets handle B-frame PTS reordering without any edit
    /// list. Lock this in: scan the entire MP4 byte stream for the
    /// `elst` 4-CC and assert it never appears, even with B-frames in
    /// the GOP pattern.
    #[cfg(feature = "h264-encoder")]
    #[test]
    fn ibpbp_mp4_does_not_emit_edts_elst() {
        use crate::codec::h264::stego::gop_pattern::GopPattern;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x10]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x67, 0x64, 0x00, 0x1E, 0xAC, 0xD9, 0x40, 0x40, 0x3C, 0x80]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xEB, 0xE3, 0xCB]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00]);
        for _ in 0..4 {
            bytes.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x30]);
            bytes.extend_from_slice(&[0, 0, 0, 1, 0x41, 0x9A, 0x00]);
        }
        let mp4 = build_mp4_with_pattern(
            MuxerProfile::HandbrakeX264,
            &bytes,
            1920,
            1080,
            FrameTiming::FPS_30,
            GopPattern::Ibpbp { gop: 5, b_count: 1 },
            5,
        )
        .unwrap();
        // Search for 'edts' and 'elst' 4-CC anywhere in the byte
        // stream. Both must be absent for a spec-correct B-frame
        // container that doesn't trim the IDR.
        for needle in [b"edts", b"elst"] {
            let mut found = false;
            for window in mp4.windows(4) {
                if window == needle.as_slice() {
                    found = true;
                    break;
                }
            }
            assert!(
                !found,
                "MP4 contains '{}' box — phasm must not emit edts/elst (see §Stealth.L4.2.fix)",
                std::str::from_utf8(needle).unwrap(),
            );
        }
    }

    /// Walk the box tree to find moov/trak/mdia/minf/stbl and return its
    /// children's 4-CCs in order. Helper for stbl-ordering tests.
    fn collect_stbl_child_order(mp4: &[u8]) -> Vec<[u8; 4]> {
        let mut order = Vec::new();
        let _ = crate::codec::mp4::iterate_boxes(mp4, 0, mp4.len(), |h, content_start, _| {
            if h.box_type == *b"moov" {
                let _ = crate::codec::mp4::iterate_boxes(
                    mp4,
                    content_start,
                    content_start + h.size as usize - h.header_len as usize,
                    |h2, cs2, _| {
                        if h2.box_type == *b"trak" {
                            walk_trak_to_stbl(mp4, cs2, h2.size as usize, &mut order);
                        }
                        Ok(())
                    },
                );
            }
            Ok(())
        });
        order
    }

    fn walk_trak_to_stbl(mp4: &[u8], content_start: usize, size: usize, out: &mut Vec<[u8; 4]>) {
        let end = content_start + size - 8;
        let _ = crate::codec::mp4::iterate_boxes(mp4, content_start, end, |h, cs, _| {
            if h.box_type == *b"mdia" {
                let _ = crate::codec::mp4::iterate_boxes(
                    mp4,
                    cs,
                    cs + h.size as usize - h.header_len as usize,
                    |h2, cs2, _| {
                        if h2.box_type == *b"minf" {
                            let _ = crate::codec::mp4::iterate_boxes(
                                mp4,
                                cs2,
                                cs2 + h2.size as usize - h2.header_len as usize,
                                |h3, cs3, _| {
                                    if h3.box_type == *b"stbl" {
                                        let _ = crate::codec::mp4::iterate_boxes(
                                            mp4,
                                            cs3,
                                            cs3 + h3.size as usize - h3.header_len as usize,
                                            |h4, _, _| {
                                                out.push(h4.box_type);
                                                Ok(())
                                            },
                                        );
                                    }
                                    Ok(())
                                },
                            );
                        }
                        Ok(())
                    },
                );
            }
            Ok(())
        });
    }

    #[test]
    fn udta_too_length_field_is_string_length_only() {
        // §Stealth.L4.3 — QuickTime user-data string layout:
        // length(u16, NOT counting itself or language) + language(u16) +
        // string. Fixed a latent off-by-4 that shipped with L4.1.
        let udta = build_udta_handbrake();
        // Outer udta: header(8) + ©too child.
        // ©too: header(8) + content (length(2) + lang(2) + string).
        let s = handbrake::UDTA_TOO.as_bytes();
        let too_content_len = 2 + 2 + s.len();
        assert_eq!(udta.len(), 8 + 8 + too_content_len);
        // Length field at udta[16..18] (after both 8-byte headers).
        let len_field = u16::from_be_bytes([udta[16], udta[17]]);
        assert_eq!(len_field as usize, s.len(), "length field = string length");
    }

    #[test]
    fn handbrake_mdhd_timescale_25fps() {
        assert_eq!(handbrake::mdhd_timescale(25, 1), 12800);
    }

    #[test]
    fn handbrake_mdhd_timescale_29_97fps() {
        // 29.97 → 30000/1001 → 30000×512 / 1001 = 15344.6… → round to 15345
        // (round-half-to-even, but 15344.65 is closer to 15345).
        let t = handbrake::mdhd_timescale(30000, 1001);
        // Allow ±1 tolerance — the rounding rule is documented but not
        // load-bearing at this precision.
        assert!((t as i64 - 15345).abs() <= 1, "got {t}");
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
        // After mux, the IDR sample contains: synthesised x264 SEI NAL +
        // the IDR slice NAL. SPS, PPS, AUD are stripped. Walk the NAL
        // list and verify NO type-7/8/9 NALs remain.
        let annex_b = build_minimal_annexb();
        let mp4 = build_mp4(
            MuxerProfile::HandbrakeX264,
            &annex_b,
            1920,
            1080,
            FrameTiming::FPS_30,
        )
        .unwrap();

        let h0 = parse_box_header(&mp4, 0).unwrap();
        let h1 = parse_box_header(&mp4, h0.size as usize).unwrap();
        assert_eq!(h1.box_type, *b"mdat");
        let mdat_payload =
            &mp4[h0.size as usize + 8..h0.size as usize + h1.size as usize];

        let mut pos = 0;
        let mut saw_sei = false;
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
            if nal_type == 6 {
                saw_sei = true;
            }
            if nal_type == 5 {
                saw_idr = true;
            }
            pos += len;
        }
        assert_eq!(pos, mdat_payload.len(), "mdat parses exactly to end");
        assert!(saw_sei, "x264 SEI NAL is present in mdat");
        assert!(saw_idr, "IDR slice NAL is present in mdat");
    }

    #[test]
    fn build_mp4_rejects_missing_parameter_sets() {
        // Annex-B with only an IDR (no SPS/PPS) → must fail with a clear
        // error.
        let bytes = vec![0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00];
        let r = build_mp4(
            MuxerProfile::HandbrakeX264,
            &bytes,
            1920,
            1080,
            FrameTiming::FPS_30,
        );
        assert!(r.is_err());
    }

    #[test]
    fn x264_sei_nal_starts_with_sei_header_and_carries_uuid() {
        let nal = build_x264_sei_user_data_unregistered();
        assert_eq!(nal[0], 0x06, "SEI NAL header byte");
        assert_eq!(nal[1], 5, "payload_type = user_data_unregistered (5)");

        // Skip past payload_size byte(s) — payload_size encoding is
        // 0xFF * (n / 0xFF) + (n % 0xFF). For our string this is at
        // most a couple of 0xFF bytes plus a remainder.
        let mut pos = 2;
        while nal[pos] == 0xFF {
            pos += 1;
        }
        pos += 1; // remainder byte

        // The UUID should follow immediately.
        assert_eq!(
            &nal[pos..pos + 16],
            &handbrake::X264_SEI_UUID,
            "x264 SEI UUID present at expected offset"
        );
    }

    #[test]
    fn x264_sei_plaintext_present_in_nal() {
        let nal = build_x264_sei_user_data_unregistered();
        // Search for the leading "x264 - core" plaintext anywhere in
        // the NAL bytes — emulation-prevention can split runs but
        // never inserts inside non-zero text, so the literal substring
        // survives.
        let banner = b"x264 - core";
        let found = nal.windows(banner.len()).any(|w| w == banner);
        assert!(found, "x264 banner is plaintext-readable inside SEI NAL");
    }

    #[test]
    fn build_mp4_injects_exactly_one_sei_at_first_idr() {
        // Two-IDR stream: SEI must appear in sample 0 only.
        let mut bytes = Vec::new();
        // Sample 0 (IDR)
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x10]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x67, 0x64, 0x00, 0x1E, 0xAC, 0xD9, 0x40, 0x40, 0x3C, 0x80]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xEB, 0xE3, 0xCB]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00]);
        // Sample 1 (also IDR — second access unit)
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x10]);
        bytes.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88, 0x84, 0x01]);

        let mp4 = build_mp4(
            MuxerProfile::HandbrakeX264,
            &bytes,
            1920,
            1080,
            FrameTiming::FPS_30,
        )
        .unwrap();

        // Count occurrences of the x264 UUID across the entire file —
        // expected: exactly 1 (in the leading IDR sample).
        let mut count = 0;
        for w in mp4.windows(handbrake::X264_SEI_UUID.len()) {
            if w == handbrake::X264_SEI_UUID {
                count += 1;
            }
        }
        assert_eq!(count, 1, "x264 SEI UUID appears exactly once in mp4 file");

        // And that one occurrence is in sample 0, not sample 1.
        use crate::codec::mp4::demux::demux;
        let parsed = demux(&mp4).unwrap();
        let track = &parsed.tracks[parsed.video_track_idx.unwrap()];
        let s0 = &track.samples[0].data;
        let s1 = &track.samples[1].data;
        let in_s0 = s0
            .windows(handbrake::X264_SEI_UUID.len())
            .any(|w| w == handbrake::X264_SEI_UUID);
        let in_s1 = s1
            .windows(handbrake::X264_SEI_UUID.len())
            .any(|w| w == handbrake::X264_SEI_UUID);
        assert!(in_s0, "SEI UUID is in sample 0");
        assert!(!in_s1, "SEI UUID is NOT in sample 1");
    }

    #[test]
    fn emulation_prevention_inserts_03_after_two_zeros() {
        // Input with a 00 00 01 sequence in the body.
        let rbsp = vec![0x42, 0x00, 0x00, 0x01, 0xAA];
        let out = add_emulation_prevention(&rbsp);
        // Expected: 0x42 0x00 0x00 0x03 0x01 0xAA
        assert_eq!(out, vec![0x42, 0x00, 0x00, 0x03, 0x01, 0xAA]);
    }

    #[test]
    fn emulation_prevention_does_not_touch_header_byte() {
        // If the input header byte is 0x00, EP must NOT insert an 0x03
        // before it (the header is exempt from EP).
        let rbsp = vec![0x06, 0x00, 0x00, 0x00, 0xFF];
        let out = add_emulation_prevention(&rbsp);
        // After header: zero run of 3, but EP only inserts when value
        // ≤ 0x03 follows two zeros, so position [1..3] = 00 00 then 00
        // gets EP inserted because next byte (0x00) is ≤ 0x03.
        assert_eq!(out, vec![0x06, 0x00, 0x00, 0x03, 0x00, 0xFF]);
    }

    #[test]
    fn build_mp4_rejects_zero_dimensions() {
        let annex_b = build_minimal_annexb();
        let r = build_mp4(
            MuxerProfile::HandbrakeX264,
            &annex_b,
            0,
            1080,
            FrameTiming::FPS_30,
        );
        assert!(r.is_err());
    }

    #[test]
    fn chunk_layout_per_gop_uniform_size() {
        // 6 samples, sync at indices 1 and 4 → GOP sizes [3, 3]
        let chunks = chunk_layout_per_gop(6, &[1, 4]);
        assert_eq!(chunks, vec![3, 3]);
    }

    #[test]
    fn chunk_layout_per_gop_mixed_size() {
        // 10 samples, sync at indices 1, 4, 8 → GOP sizes [3, 4, 3]
        let chunks = chunk_layout_per_gop(10, &[1, 4, 8]);
        assert_eq!(chunks, vec![3, 4, 3]);
    }

    #[test]
    fn chunk_layout_per_gop_no_sync_falls_back() {
        let chunks = chunk_layout_per_gop(5, &[]);
        assert_eq!(chunks, vec![5]);
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

    #[test]
    fn build_ctts_writes_per_sample_signed_offsets() {
        // 5 samples, IBPBP M=2: ctts = [0, +1, -1, +1, -1] × ticks.
        let offsets: Vec<i32> = vec![0, 512, -512, 512, -512];
        let ctts = build_ctts(&offsets);
        // Header (8) + version+flags (4) + entry_count (4) + 5 × 8 bytes
        assert_eq!(ctts.len(), 8 + 4 + 4 + 5 * 8);
        // Version byte = 1.
        assert_eq!(ctts[8], 1);
        // entry_count = 5.
        assert_eq!(&ctts[12..16], &5u32.to_be_bytes());
        // First entry: sample_count=1, offset=0.
        assert_eq!(&ctts[16..20], &1u32.to_be_bytes());
        assert_eq!(&ctts[20..24], &0i32.to_be_bytes());
        // Third entry: sample_count=1, offset=-512.
        assert_eq!(&ctts[32..36], &1u32.to_be_bytes());
        assert_eq!(&ctts[36..40], &(-512i32).to_be_bytes());
    }

        #[cfg(feature = "h264-encoder")]
        #[test]
    fn build_mp4_with_ipppp_pattern_emits_no_ctts_no_edts() {
        // IPPPP (no B-frames) → no ctts, no edts.
        use crate::codec::h264::stego::gop_pattern::GopPattern;
        let annex_b = build_minimal_annexb();
        let mp4 = build_mp4_with_pattern(
            MuxerProfile::HandbrakeX264,
            &annex_b,
            1920,
            1080,
            FrameTiming::FPS_30,
            GopPattern::Ipppp { gop: 1 },
            1,
        )
        .unwrap();
        assert!(!mp4.windows(4).any(|w| w == b"ctts"), "no ctts box");
        assert!(!mp4.windows(4).any(|w| w == b"edts"), "no edts box");
    }

        #[cfg(feature = "h264-encoder")]
        #[test]
    fn build_mp4_with_pattern_rejects_au_count_mismatch() {
        // Pattern asks for 5 frames but input has 1 → error.
        use crate::codec::h264::stego::gop_pattern::GopPattern;
        let annex_b = build_minimal_annexb();
        let r = build_mp4_with_pattern(
            MuxerProfile::HandbrakeX264,
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
            MuxerProfile::HandbrakeX264,
            &annex_b,
            1920,
            1080,
            FrameTiming { fps_num: 0, fps_den: 1 },
        );
        assert!(r.is_err());
    }
}
