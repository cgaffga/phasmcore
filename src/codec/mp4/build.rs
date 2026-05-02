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
//! `docs/design/h264-stealth-strategy.md`).
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

use super::{write_u32, write_u64, AvccData, Mp4Error};
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
        ),
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

fn build_handbrake_x264(
    annex_b: &[u8],
    access_units: &[AccessUnit],
    avcc: &AvccData,
    width: u32,
    height: u32,
    timing: FrameTiming,
    composition_offsets: Option<&[i32]>,
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
    let total_mdat_payload: u64 = sample_sizes.iter().map(|&s| s as u64).sum();

    // ─── Compute layout ─────────────────────────────────────────
    //
    // HandBrake (no +faststart) layout: ftyp → mdat → moov.
    // We need the absolute byte offset of the first sample in mdat
    // before we can finalise stco. ftyp size is fixed; mdat header
    // is 8 or 16 bytes depending on payload size.

    let ftyp = build_ftyp_handbrake();
    let ftyp_len = ftyp.len() as u64;

    let mdat_header_len: u64 = if total_mdat_payload + 8 > u32::MAX as u64 { 16 } else { 8 };
    let mdat_total_size = mdat_header_len + total_mdat_payload;
    let first_sample_offset = ftyp_len + mdat_header_len;

    // Per-sample absolute offsets in the output file.
    let mut sample_offsets: Vec<u64> = Vec::with_capacity(sample_data.len());
    let mut cursor = first_sample_offset;
    for &size in &sample_sizes {
        sample_offsets.push(cursor);
        cursor += size as u64;
    }

    // Sync sample indices (1-based per spec): each access unit whose
    // first VCL NAL is IDR.
    let sync_samples: Vec<u32> = access_units
        .iter()
        .enumerate()
        .filter_map(|(i, au)| if au.is_sync { Some(i as u32 + 1) } else { None })
        .collect();

    // ─── Build moov ─────────────────────────────────────────────
    let moov = build_moov_handbrake(MoovParams {
        width,
        height,
        movie_duration_ms,
        track_duration_media,
        mdhd_timescale,
        sample_sizes: &sample_sizes,
        sample_offsets: &sample_offsets,
        sync_samples: &sync_samples,
        avcc,
        composition_offsets,
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
}

fn build_moov_handbrake(p: MoovParams<'_>) -> Vec<u8> {
    let mvhd = build_mvhd(p.movie_duration_ms, /* next_track_id */ 2);
    let trak = build_video_trak_handbrake(&p);
    let udta = build_udta_handbrake();

    let mut moov = Vec::new();
    moov.extend_from_slice(&mvhd);
    moov.extend_from_slice(&trak);
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

/// `edts/elst` for B-frame-aware streams. HandBrake emits a single
/// `elst` entry pointing at `media_time = b_offset` (the PTS of the
/// first DISPLAYED frame, in track-media units) so the player skips
/// the encode-order pre-roll. `None` when no composition offsets are
/// supplied (no B-frames → no `edts` needed).
fn build_edts_handbrake(p: &MoovParams<'_>) -> Option<Vec<u8>> {
    let offsets = p.composition_offsets?;
    if offsets.is_empty() {
        return None;
    }
    // Find the first displayed frame's PTS in mdhd-timescale units.
    // For closed IBPBP M=2: display_idx 0 = encode_idx 0 (the IDR), so
    // PTS_of_first_display = 0. The "pre-roll" is the difference
    // between the latest (max) DTS+ctts and the encode-order range.
    // Practically, HandBrake sets media_time = (-min(ctts)) so the
    // smallest negative composition shift becomes a leading silent
    // edit.
    let min_ctts = offsets.iter().copied().min().unwrap_or(0);
    if min_ctts >= 0 {
        return None; // no negative offsets → no pre-roll needed
    }
    // Convert min_ctts (in mdhd-timescale) to mvhd-timescale (1000) for
    // segment_duration; media_time stays in mdhd-timescale.
    let pre_roll_media: u64 = (-min_ctts) as u64;
    let pre_roll_movie: u64 = pre_roll_media * (handbrake::MVHD_TIMESCALE as u64) /
        (p.mdhd_timescale as u64);
    let track_duration_movie: u64 = p.movie_duration_ms; // already movie-timescale

    // Single edit: segment_duration covers full track, media_time =
    // pre_roll_media so the playhead starts at the first displayed
    // frame.
    let mut content = Vec::new();
    content.extend_from_slice(&[0, 0, 0, 0]); // version=0, flags=0
    content.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    content.extend_from_slice(&((track_duration_movie + pre_roll_movie) as u32).to_be_bytes());
    content.extend_from_slice(&(pre_roll_media as u32).to_be_bytes()); // media_time
    content.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // media_rate = 1.0
    let elst = wrap_box(b"elst", &content);
    Some(wrap_box(b"edts", &elst))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::mp4::{is_mp4, parse_box_header};

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
