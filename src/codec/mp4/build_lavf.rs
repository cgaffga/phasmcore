// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! [`MuxerProfile::LavfOpenH264`](super::MuxerProfile::LavfOpenH264): the MP4
//! a libavformat-based transcoder writes around a libopenh264 encode.
//!
//! # Why this profile
//!
//! phasm's H.264 bitstream comes from a fork of OpenH264, and it carries
//! OpenH264's traits whatever the container says: no B-frames, one reference
//! frame, `log2_max_frame_num` 16 and a 16-bit POC LSB, High profile without
//! the 8x8 transform, no SEI, no AUD. The earlier `HandbrakeX264` profile was
//! built (2026-05-02) for the retired pure-Rust encoder, whose structure was
//! close to x264's, and it wrapped the OpenH264 stream in an x264 SEI banner
//! (`bframes=3 ref=3 8x8dct=1 keyint=250 …`) and a HandBrake `©too`. Every
//! one of those claims is contradicted by the SPS, PPS and slice structure
//! next to it. A file is only as plausible as its least consistent field, so
//! the container now says what the bitstream already says: OpenH264, muxed
//! by the most common tool that produces OpenH264-in-MP4 — the libavformat
//! mp4 muxer with libavcodec's libopenh264 wrapper (the default H.264
//! encoder of distribution builds that ship without x264). The phasm CLI
//! already needs that tool installed to decode its input, so its output
//! looking like that tool's output is the unsurprising reading.
//!
//! # What it writes (verified box by box against that muxer's output,
//! version 62.12.101, on the corpus clips)
//!
//! - `ftyp` `isom`/512 + `isom iso2 avc1 mp41`, an 8-byte `free`, then
//!   `mdat`, then `moov` (no fast-start). Past 4 GiB the `free` slot becomes
//!   the 64-bit `mdat` header, as that muxer does.
//! - `mvhd`/`tkhd`/`mdhd` creation and modification times 0, movie timescale
//!   1000, `tkhd` flags 3 (enabled, in movie); the audio track in alternate
//!   group 1 at full volume.
//! - Video media timescale = the frame rate's denominator-side `1/fps` time
//!   base doubled until it reaches 10000 (30 fps → 15360, 29.97 → 30000 with
//!   a 1001 delta, 24 → 12288, 25 → 12800).
//! - An `edts/elst` on every track: one entry `(duration, 0)`, plus a leading
//!   empty edit when the video starts after the audio (the lead-in rounded to
//!   whole frames, as an encoder with a `1/fps` time base has it).
//! - `hdlr` "VideoHandler" / "SoundHandler"; `avc1` compressor name
//!   `Lavc… libopenh264`; `avcC` with the High-profile extension bytes;
//!   `btrt` (0, average, 0) for the video (the encoder publishes CPB
//!   properties with no average bitrate), (0, average, average) for copied
//!   audio, whose `esds` bitrates and ES_ID are rewritten the same way.
//! - Samples interleaved by decode time across tracks, video first on ties,
//!   and chunked as runs of contiguous samples under 1 MiB.
//! - AAC audio gets the `roll` sample group (distance -1). Audio samples
//!   that start after the source's edit list ends (never presented) are
//!   dropped, as that muxer's demuxer drops them; the new edit spans the
//!   samples kept.
//! - `moov/udta/meta/ilst/©too` = `Lavf…`.
//!
//! The SPS of the OpenH264 fork carries no colour description, so no `colr`
//! is written (the muxer writes one only for a tagged stream), and no `pasp`
//! (no sample aspect ratio). AUDs are dropped and SPS/PPS live in `avcC`
//! only, which is what the libopenh264 wrapper emits with global headers.
//!
//! The bitstream is untouched: the samples are the encoder's NAL units with
//! only AUD/SPS/PPS removed (they live in `avcC`), and nothing injected.
//!
//! # AV1 ([`build_av1`])
//!
//! phasm's AV1 comes from a fork of rav1e, so the AV1 file is the same
//! muxer around libavcodec's librav1e wrapper (checked against that
//! muxer's AV1 output, version 62.12.101, and its source for what depends
//! on the encoder wrapper). It differs from the H.264 file in:
//!
//! - `ftyp` compatible brands `isom av01 iso2 mp41`.
//! - `av01` compressor name `Lavc… librav1e`; `av1C` = rav1e's container
//!   sequence header, which the wrapper hands over verbatim: level 31, no
//!   config OBUs (the sequence header rides in every sync sample).
//! - `fiel` (progressive), which the muxer writes for AV1; `colr` (nclx)
//!   when the sequence header carries a specified colour description (the
//!   wrapper always passes one, and phasm's encoder signals BT.709).
//! - `btrt` (0, average, average): librav1e publishes no CPB properties.
//! - No `sdtp`: librav1e marks no frame disposable.
//! - Timing from the source's frame times, each rounded to a slot of the
//!   encoder time base ([`av1_timeline`]); samples are temporal units with
//!   the temporal delimiters dropped.

use super::{
    build_dinf, build_hdlr_vide, build_mvhd, build_stco, build_stsc, build_stss, build_stts,
    build_vmhd, AccessUnit, AudioPassthrough, FrameTiming, UNITY_MATRIX,
};
use crate::codec::mp4::av1_obu_split::Av1SequenceHeaderInfo;
use crate::codec::h264::NalType;
use crate::codec::mp4::{AvccData, Mp4Error};

/// libavformat identification, `moov/udta/meta/ilst/©too`.
pub(super) const LAVF_TOOL: &str = "Lavf62.12.101";
/// libavcodec identification + encoder name, the `avc1` compressor name.
/// The three version strings here move together.
pub(super) const LAVC_COMPRESSOR: &str = "Lavc62.28.101 libopenh264";
/// The `av01` compressor name: the same libavcodec, librav1e encoder.
pub(super) const LAVC_COMPRESSOR_AV1: &str = "Lavc62.28.101 librav1e";

const MOVIE_TIMESCALE: u32 = 1000;
/// A chunk grows while it stays below this many bytes.
const MAX_CHUNK_BYTES: u64 = 1 << 20;
const SOUND_HANDLER: &[u8] = b"SoundHandler";
/// ISO-639-2 "und", packed.
const LANG_UND: u16 = 0x55C4;
/// ISO-639-2 "eng", packed (QuickTime language code 0).
const LANG_ENG: u16 = 0x15C7;

// ─── Box writing ──────────────────────────────────────────────────────

fn bx(typ: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 16);
    let size = payload.len() as u64 + 8;
    if size <= u32::MAX as u64 {
        out.extend_from_slice(&(size as u32).to_be_bytes());
        out.extend_from_slice(typ);
    } else {
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(typ);
        out.extend_from_slice(&(size + 8).to_be_bytes());
    }
    out.extend_from_slice(payload);
    out
}

fn concat(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

/// `ticks` of `timescale` in movie time (ms), rounded up.
fn to_movie_ceil(ticks: u64, timescale: u32) -> u64 {
    ((ticks as u128 * MOVIE_TIMESCALE as u128).div_ceil(timescale as u128)) as u64
}

/// `ticks` of `timescale` in movie time (ms), rounded down.
fn to_movie_floor(ticks: u64, timescale: u32) -> u64 {
    (ticks as u128 * MOVIE_TIMESCALE as u128 / timescale as u128) as u64
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Video `(mdhd timescale, per-frame stts delta)` for a `fps_num/fps_den`
/// frame rate: the `fps_den/fps_num` time base, reduced, with its
/// denominator doubled until it reaches 10000.
pub(super) fn video_timescale(timing: FrameTiming) -> (u32, u32) {
    let (num, den) = (timing.fps_num as u64, timing.fps_den as u64);
    let g = gcd(num, den).max(1);
    let (mut ts, mut delta) = (num / g, den / g);
    while ts < 10_000 {
        ts *= 2;
        delta *= 2;
    }
    (ts as u32, delta.min(u32::MAX as u64) as u32)
}

fn mdhd(duration: u64, timescale: u32, language: u16) -> Vec<u8> {
    let mut p = Vec::with_capacity(32);
    if duration < i32::MAX as u64 {
        p.extend_from_slice(&[0, 0, 0, 0]);
        p.extend_from_slice(&[0; 8]); // creation, modification
        p.extend_from_slice(&timescale.to_be_bytes());
        p.extend_from_slice(&(duration as u32).to_be_bytes());
    } else {
        p.extend_from_slice(&[1, 0, 0, 0]);
        p.extend_from_slice(&[0; 16]);
        p.extend_from_slice(&timescale.to_be_bytes());
        p.extend_from_slice(&duration.to_be_bytes());
    }
    p.extend_from_slice(&language.to_be_bytes());
    p.extend_from_slice(&[0; 2]); // pre_defined
    bx(b"mdhd", &p)
}

/// `tkhd`: flags 3 (enabled, in movie). Video: alternate group 0, volume 0,
/// the display size. Audio: alternate group 1, volume 1.0, size 0.
fn tkhd(track_id: u32, duration_ms: u64, audio: bool, width: u32, height: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity(92);
    let v1 = duration_ms >= i32::MAX as u64;
    p.extend_from_slice(&[v1 as u8, 0, 0, 3]);
    if v1 {
        p.extend_from_slice(&[0; 16]);
        p.extend_from_slice(&track_id.to_be_bytes());
        p.extend_from_slice(&[0; 4]);
        p.extend_from_slice(&duration_ms.to_be_bytes());
    } else {
        p.extend_from_slice(&[0; 8]);
        p.extend_from_slice(&track_id.to_be_bytes());
        p.extend_from_slice(&[0; 4]);
        p.extend_from_slice(&(duration_ms as u32).to_be_bytes());
    }
    p.extend_from_slice(&[0; 8]); // reserved
    p.extend_from_slice(&[0; 2]); // layer
    p.extend_from_slice(&(audio as u16).to_be_bytes()); // alternate_group
    p.extend_from_slice(&(if audio { 0x0100u16 } else { 0 }).to_be_bytes()); // volume
    p.extend_from_slice(&[0; 2]);
    p.extend_from_slice(&UNITY_MATRIX);
    p.extend_from_slice(&(width << 16).to_be_bytes());
    p.extend_from_slice(&(height << 16).to_be_bytes());
    bx(b"tkhd", &p)
}

/// One `elst` entry: (segment duration in movie time, media time, rate 1.0).
type Edit = (u64, i64);

fn edts(edits: &[Edit]) -> Vec<u8> {
    let v1 = edits
        .iter()
        .any(|&(d, m)| d > u32::MAX as u64 || m > i32::MAX as i64 || m < i32::MIN as i64);
    let mut p = Vec::with_capacity(8 + 20 * edits.len());
    p.extend_from_slice(&[v1 as u8, 0, 0, 0]);
    p.extend_from_slice(&(edits.len() as u32).to_be_bytes());
    for &(d, m) in edits {
        if v1 {
            p.extend_from_slice(&d.to_be_bytes());
            p.extend_from_slice(&m.to_be_bytes());
        } else {
            p.extend_from_slice(&(d as u32).to_be_bytes());
            p.extend_from_slice(&(m as i32).to_be_bytes());
        }
        p.extend_from_slice(&0x0001_0000u32.to_be_bytes());
    }
    bx(b"edts", &bx(b"elst", &p))
}

fn hdlr_soun() -> Vec<u8> {
    let mut p = vec![0u8; 8];
    p.extend_from_slice(b"soun");
    p.extend_from_slice(&[0; 12]);
    p.extend_from_slice(SOUND_HANDLER);
    p.push(0);
    bx(b"hdlr", &p)
}

fn smhd() -> Vec<u8> {
    bx(b"smhd", &[0; 8])
}

/// `stsz`: one shared size when every sample has it, else a table.
fn stsz(sizes: &[u32]) -> Vec<u8> {
    let mut p = vec![0u8; 4];
    match sizes.first() {
        Some(&s) if sizes.iter().all(|&x| x == s) => {
            p.extend_from_slice(&s.max(1).to_be_bytes());
            p.extend_from_slice(&(sizes.len() as u32).to_be_bytes());
        }
        _ => {
            p.extend_from_slice(&0u32.to_be_bytes());
            p.extend_from_slice(&(sizes.len() as u32).to_be_bytes());
            for &s in sizes {
                p.extend_from_slice(&s.to_be_bytes());
            }
        }
    }
    bx(b"stsz", &p)
}

fn btrt(buffer: u32, max: u32, avg: u32) -> Vec<u8> {
    bx(b"btrt", &concat(&[&buffer.to_be_bytes(), &max.to_be_bytes(), &avg.to_be_bytes()]))
}

/// Average bitrate the muxer computes: bytes × 8 over the track's media
/// duration.
fn avg_bitrate(bytes: u64, timescale: u32, duration: u64) -> u32 {
    if duration == 0 {
        return 0;
    }
    (bytes as u128 * 8 * timescale as u128 / duration as u128).min(u32::MAX as u128) as u32
}

/// `avcC`: the decoder configuration record, plus — for every profile but
/// Baseline / Main / Extended — the chroma-format / bit-depth extension and
/// a zero SPS-extension count, which that muxer appends when it builds the
/// record from Annex-B parameter sets.
fn avcc_box(avcc: &AvccData) -> Vec<u8> {
    let mut p = avcc.to_bytes();
    if !matches!(avcc.profile, 66 | 77 | 88) {
        let (chroma, luma, chroma_depth) = avcc
            .sps_nalus
            .first()
            .filter(|s| s.len() > 1)
            .and_then(|s| {
                let rbsp = crate::codec::h264::bitstream::remove_emulation_prevention(&s[1..]);
                crate::codec::h264::sps::parse_sps(&rbsp).ok()
            })
            .map(|s| (s.chroma_format_idc, s.bit_depth_luma, s.bit_depth_chroma))
            .unwrap_or((1, 8, 8));
        p.push(0xFC | (chroma & 3));
        p.push(0xF8 | (luma.saturating_sub(8) & 7));
        p.push(0xF8 | (chroma_depth.saturating_sub(8) & 7));
        p.push(0); // numOfSequenceParameterSetExt
    }
    bx(b"avcC", &p)
}

/// The 78-byte VisualSampleEntry head the muxer writes, `compressor` as
/// the Pascal-string compressor name.
fn visual_sample_entry(width: u32, height: u32, compressor: &str) -> Vec<u8> {
    let mut e = Vec::with_capacity(160);
    e.extend_from_slice(&[0; 6]); // reserved
    e.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    e.extend_from_slice(&[0; 16]); // version, revision, vendor, temporal + spatial quality
    e.extend_from_slice(&(width as u16).to_be_bytes());
    e.extend_from_slice(&(height as u16).to_be_bytes());
    e.extend_from_slice(&0x0048_0000u32.to_be_bytes());
    e.extend_from_slice(&0x0048_0000u32.to_be_bytes());
    e.extend_from_slice(&[0; 4]); // data size
    e.extend_from_slice(&1u16.to_be_bytes()); // frame count
    let mut name = [0u8; 32];
    let n = compressor.len().min(31);
    name[0] = n as u8;
    name[1..1 + n].copy_from_slice(&compressor.as_bytes()[..n]);
    e.extend_from_slice(&name);
    e.extend_from_slice(&0x0018u16.to_be_bytes()); // depth
    e.extend_from_slice(&[0xFF, 0xFF]); // color table id -1
    e
}

fn stsd_avc1(width: u32, height: u32, avcc: &AvccData, btrt_box: &[u8]) -> Vec<u8> {
    let mut e = visual_sample_entry(width, height, LAVC_COMPRESSOR);
    e.extend_from_slice(&avcc_box(avcc));
    e.extend_from_slice(btrt_box);
    bx(b"stsd", &concat(&[&[0, 0, 0, 0], &1u32.to_be_bytes(), &bx(b"avc1", &e)]))
}

/// `moov/udta/meta/ilst/©too`, as the muxer writes its identification.
fn udta() -> Vec<u8> {
    let mut hdlr = vec![0u8; 8];
    hdlr.extend_from_slice(b"mdir");
    hdlr.extend_from_slice(b"appl");
    hdlr.extend_from_slice(&[0; 8]);
    hdlr.push(0);
    let data = bx(b"data", &concat(&[&1u32.to_be_bytes(), &0u32.to_be_bytes(), LAVF_TOOL.as_bytes()]));
    let ilst = bx(b"ilst", &bx(&[0xA9, b't', b'o', b'o'], &data));
    let meta = bx(b"meta", &concat(&[&[0, 0, 0, 0], &bx(b"hdlr", &hdlr), &ilst]));
    bx(b"udta", &meta)
}

// ─── Box reading (the passed-through audio track) ─────────────────────

/// Top-level boxes of `buf` as (type, payload). `None` on a size that does
/// not fit.
fn children(buf: &[u8]) -> Option<Vec<([u8; 4], &[u8])>> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while buf.len() - off >= 8 {
        let size32 = u32::from_be_bytes(buf[off..off + 4].try_into().ok()?) as usize;
        let typ: [u8; 4] = buf[off + 4..off + 8].try_into().ok()?;
        let (header, size) = match size32 {
            0 => (8, buf.len() - off),
            1 => {
                let large = u64::from_be_bytes(buf.get(off + 8..off + 16)?.try_into().ok()?);
                (16, usize::try_from(large).ok()?)
            }
            n => (8, n),
        };
        if size < header || size > buf.len() - off {
            return None;
        }
        out.push((typ, &buf[off + header..off + size]));
        off += size;
    }
    Some(out)
}

fn child<'a>(boxes: &[([u8; 4], &'a [u8])], typ: &[u8; 4]) -> Option<&'a [u8]> {
    boxes.iter().find(|(t, _)| t == typ).map(|&(_, p)| p)
}

fn be32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn be64(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_be_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// An `mp4` language: QuickTime's "unspecified" and Macintosh language codes
/// become ISO-639-2 (code 0 is English), as a QuickTime → ISO remux maps
/// them.
fn iso_language(code: u16) -> u16 {
    match code {
        0 => LANG_ENG,
        0x7FFF => LANG_UND,
        c if c < 0x400 => LANG_UND,
        c => c,
    }
}

/// How the copied audio sits on the movie timeline.
enum AudioEdits {
    /// Starts with the movie, `media_time` ticks into its media.
    Plain { media_time: u64 },
    /// An empty edit of `delay_ms` first.
    Delayed { delay_ms: u64, media_time: u64 },
    /// Anything else, kept as the source had it (movie-timescale durations).
    Verbatim(Vec<Edit>),
}

impl AudioEdits {
    #[cfg(test)]
    fn parse(elst: Option<&[u8]>) -> Option<Self> {
        Self::parse_presented(elst).map(|(e, _)| e)
    }

    /// [`Self::parse`], plus the presented segment's duration (movie time)
    /// for a `Plain` / `Delayed` edit list.
    fn parse_presented(elst: Option<&[u8]>) -> Option<(Self, Option<u64>)> {
        let Some(e) = elst else {
            return Some((Self::Plain { media_time: 0 }, None));
        };
        let v1 = *e.first()? == 1;
        let n = be32(e, 4)? as usize;
        let mut entries = Vec::with_capacity(n);
        let mut at = 8;
        for _ in 0..n {
            let (d, m) = if v1 {
                let r = (be64(e, at)?, be64(e, at + 8)? as i64);
                at += 16;
                r
            } else {
                let r = (be32(e, at)? as u64, be32(e, at + 4)? as i32 as i64);
                at += 8;
                r
            };
            let rate = be32(e, at)?;
            at += 4;
            entries.push((d, m, rate));
        }
        let delay: u64 = entries.iter().take_while(|e| e.1 == -1).map(|e| e.0).sum();
        let rest: Vec<_> = entries.iter().skip_while(|e| e.1 == -1).collect();
        Some(match *rest.as_slice() {
            [] if entries.is_empty() => (Self::Plain { media_time: 0 }, None),
            [&(d, m, 0x0001_0000)] if m >= 0 && delay == 0 => (Self::Plain { media_time: m as u64 }, Some(d)),
            [&(d, m, 0x0001_0000)] if m >= 0 => {
                (Self::Delayed { delay_ms: delay, media_time: m as u64 }, Some(d))
            }
            _ => (Self::Verbatim(entries.iter().map(|&(d, m, _)| (d, m)).collect()), None),
        })
    }

    /// Leading delay (ms) and the media time presented at it.
    fn start(&self) -> (u64, u64) {
        match *self {
            Self::Plain { media_time } => (0, media_time),
            Self::Delayed { delay_ms, media_time } => (delay_ms, media_time),
            Self::Verbatim(ref v) => {
                let delay = v.iter().take_while(|e| e.1 == -1).map(|e| e.0).sum();
                let m = v.iter().find(|e| e.1 >= 0).map_or(0, |e| e.1 as u64);
                (delay, m)
            }
        }
    }

    /// `(elst entries, tkhd duration)` for `media_duration` ticks.
    fn output(&self, media_duration: u64, timescale: u32) -> (Vec<Edit>, u64) {
        match *self {
            Self::Plain { media_time } => {
                let seg = to_movie_ceil(media_duration.saturating_sub(media_time), timescale);
                (vec![(seg, media_time as i64)], seg)
            }
            Self::Delayed { delay_ms, media_time } => {
                let seg = to_movie_ceil(media_duration.saturating_sub(media_time), timescale);
                (vec![(delay_ms, -1), (seg, media_time as i64)], delay_ms + seg)
            }
            Self::Verbatim(ref v) => (v.clone(), v.iter().map(|e| e.0).sum()),
        }
    }
}

/// An ES descriptor length, always in the 4-byte form.
fn put_descr(out: &mut Vec<u8>, tag: u8, size: usize) {
    out.push(tag);
    for i in (1..=3).rev() {
        out.push(((size >> (7 * i)) as u8 & 0x7F) | 0x80);
    }
    out.push(size as u8 & 0x7F);
}

/// Read one descriptor at `at`: (tag, payload range).
fn read_descr(b: &[u8], at: usize) -> Option<(u8, usize, usize)> {
    let tag = *b.get(at)?;
    let mut len = 0usize;
    let mut p = at + 1;
    for _ in 0..4 {
        let c = *b.get(p)?;
        p += 1;
        len = (len << 7) | (c & 0x7F) as usize;
        if c & 0x80 == 0 {
            break;
        }
    }
    (p + len <= b.len()).then_some((tag, p, p + len))
}

/// `(objectTypeIndication, DecoderSpecificInfo)` from an `esds` payload.
fn parse_esds(esds: &[u8]) -> Option<(u8, Option<Vec<u8>>)> {
    let (tag, s, e) = read_descr(esds, 4)?;
    if tag != 0x03 {
        return None;
    }
    let es = &esds[s..e];
    let flags = *es.get(2)?;
    let mut at = 3;
    if flags & 0x80 != 0 {
        at += 2;
    }
    if flags & 0x40 != 0 {
        at += 1 + *es.get(at)? as usize;
    }
    if flags & 0x20 != 0 {
        at += 2;
    }
    while at < es.len() {
        let (t, ds, de) = read_descr(es, at)?;
        if t == 0x04 {
            let dc = &es[ds..de];
            let oti = *dc.first()?;
            let dsi = (dc.len() > 13)
                .then(|| read_descr(dc, 13))
                .flatten()
                .filter(|&(t5, _, _)| t5 == 0x05)
                .map(|(_, a, b)| dc[a..b].to_vec());
            return Some((oti, dsi));
        }
        at = de;
    }
    None
}

/// The muxer's `esds`: ES_ID = the track id, the stream's bitrates, the
/// decoder-specific info, SL config 2.
fn esds(track_id: u16, oti: u8, dsi: Option<&[u8]>, max: u32, avg: u32) -> Vec<u8> {
    let dsi_total = dsi.map_or(0, |d| 5 + d.len());
    let mut p = vec![0u8; 4];
    put_descr(&mut p, 0x03, 3 + 5 + 13 + dsi_total + 5 + 1);
    p.extend_from_slice(&track_id.to_be_bytes());
    p.push(0);
    put_descr(&mut p, 0x04, 13 + dsi_total);
    p.push(oti);
    p.push(0x15); // audio stream, reserved bit
    p.extend_from_slice(&[0, 0, 0]); // bufferSizeDB
    p.extend_from_slice(&max.to_be_bytes());
    p.extend_from_slice(&avg.to_be_bytes());
    if let Some(d) = dsi {
        put_descr(&mut p, 0x05, d.len());
        p.extend_from_slice(d);
    }
    put_descr(&mut p, 0x06, 1);
    p.push(0x02);
    bx(b"esds", &p)
}

/// The copied audio track, parsed out of the ISO-normalized, scrubbed `trak`
/// that `audio_iso` makes of the source's.
struct AudioTrack<'s> {
    samples: &'s [&'s [u8]],
    durations: Vec<u32>,
    timescale: u32,
    language: u16,
    edits: AudioEdits,
    /// The single sample entry box (`mp4a`, …) as the source had it.
    entry_typ: [u8; 4],
    entry: Vec<u8>,
    /// Source `stss` when not every sample is a sync sample.
    stss: Option<Vec<u8>>,
    /// Source sample groups, kept for codecs other than AAC.
    groups: Vec<Vec<u8>>,
}

const AUDIO_SAMPLE_ENTRY_LEN: usize = 28;

impl<'s> AudioTrack<'s> {
    fn parse(trak: &[u8], samples: &'s [&'s [u8]]) -> Result<Self, Mp4Error> {
        let bad = |what: &str| Mp4Error::InvalidBox(format!("audio track: {what}"));
        let top = children(trak).ok_or_else(|| bad("trak layout"))?;
        let trak_body = child(&top, b"trak").ok_or_else(|| bad("no trak"))?;
        let tb = children(trak_body).ok_or_else(|| bad("trak children"))?;
        let elst = child(&tb, b"edts")
            .and_then(children)
            .and_then(|e| child(&e, b"elst").map(|p| p.to_vec()));
        let mdia = children(child(&tb, b"mdia").ok_or_else(|| bad("no mdia"))?)
            .ok_or_else(|| bad("mdia children"))?;
        let mdhd = child(&mdia, b"mdhd").ok_or_else(|| bad("no mdhd"))?;
        let (timescale, language) = if mdhd.first() == Some(&1) {
            (be32(mdhd, 20), mdhd.get(32..34))
        } else {
            (be32(mdhd, 12), mdhd.get(20..22))
        };
        let timescale = timescale.filter(|&t| t > 0).ok_or_else(|| bad("mdhd timescale"))?;
        let language = language
            .map(|l| u16::from_be_bytes([l[0], l[1]]))
            .ok_or_else(|| bad("mdhd language"))?;
        let minf = children(child(&mdia, b"minf").ok_or_else(|| bad("no minf"))?)
            .ok_or_else(|| bad("minf children"))?;
        let stbl = children(child(&minf, b"stbl").ok_or_else(|| bad("no stbl"))?)
            .ok_or_else(|| bad("stbl children"))?;

        let stsd = child(&stbl, b"stsd").ok_or_else(|| bad("no stsd"))?;
        let entries = stsd.get(8..).and_then(children).ok_or_else(|| bad("stsd entries"))?;
        let [(entry_typ, entry)] = entries.as_slice() else {
            return Err(bad("more than one sample description"));
        };

        let stts = child(&stbl, b"stts").ok_or_else(|| bad("no stts"))?;
        let runs = be32(stts, 4).ok_or_else(|| bad("stts"))? as usize;
        let mut durations = Vec::with_capacity(samples.len());
        for r in 0..runs {
            let count = be32(stts, 8 + 8 * r).ok_or_else(|| bad("stts"))? as usize;
            let delta = be32(stts, 12 + 8 * r).ok_or_else(|| bad("stts"))?;
            if durations.len() + count > samples.len() {
                return Err(bad("stts covers more samples than the track has"));
            }
            durations.extend(std::iter::repeat_n(delta, count));
        }
        if durations.len() != samples.len() {
            return Err(bad("stts does not cover every sample"));
        }

        let (edits, presented_ms) = AudioEdits::parse_presented(elst.as_deref()).ok_or_else(|| bad("elst"))?;
        // Samples that start after the source's edit ends were never
        // presented; a remux drops them (its demuxer applies the edit list),
        // and its own edit then spans exactly the samples it kept.
        let keep = match presented_ms {
            Some(seg) => {
                let start = edits.start().1 as i128;
                let end = seg as i128 * timescale as i128; // movie ms × media ts
                let mut t: i128 = 0;
                let mut n = 0usize;
                for &d in &durations {
                    if (t - start) * MOVIE_TIMESCALE as i128 >= end {
                        break;
                    }
                    t += d as i128;
                    n += 1;
                }
                if n == 0 { durations.len() } else { n }
            }
            None => durations.len(),
        };
        let all = durations.len();
        durations.truncate(keep);
        let samples = &samples[..keep];

        // Sync samples among those kept, written only when not all are.
        let stss = child(&stbl, b"stss").and_then(|s| {
            let n = be32(s, 4)? as usize;
            let kept: Vec<u32> =
                (0..n).filter_map(|i| be32(s, 8 + 4 * i)).filter(|&i| i as usize <= keep).collect();
            (kept.len() != keep).then(|| super::build_stss(&kept))
        });
        // Other codecs' sample groups, as the source had them — only while
        // they still describe every sample.
        let groups = if keep == all {
            stbl.iter().filter(|(t, _)| t == b"sgpd" || t == b"sbgp").map(|(t, p)| bx(t, p)).collect()
        } else {
            Vec::new()
        };

        Ok(Self {
            samples,
            durations,
            timescale,
            language: iso_language(language),
            edits,
            entry_typ: *entry_typ,
            entry: entry.to_vec(),
            stss,
            groups,
        })
    }

    fn media_duration(&self) -> u64 {
        self.durations.iter().map(|&d| d as u64).sum()
    }

    fn bytes(&self) -> u64 {
        self.samples.iter().map(|s| s.len() as u64).sum()
    }

    /// AAC in an `mp4a` entry (any of the MPEG-4 / MPEG-2 AAC object types).
    fn esds(&self) -> Option<(u8, Option<Vec<u8>>)> {
        if &self.entry_typ != b"mp4a" || self.entry.len() < AUDIO_SAMPLE_ENTRY_LEN {
            return None;
        }
        let kids = children(&self.entry[AUDIO_SAMPLE_ENTRY_LEN..])?;
        parse_esds(child(&kids, b"esds")?)
    }

    fn is_aac(&self) -> bool {
        self.esds().is_some_and(|(oti, _)| matches!(oti, 0x40 | 0x66..=0x68))
    }

    /// The sample entry as the muxer writes it: `esds` rewritten with this
    /// track's bitrates, then `btrt`. Entries it cannot read keep their
    /// fields and only gain the `btrt`.
    fn sample_entry(&self, avg: u32) -> Vec<u8> {
        let rate = btrt(0, avg, avg);
        if let Some((oti, dsi)) = self.esds() {
            let oti = if matches!(oti, 0x66..=0x68) { 0x40 } else { oti };
            let head = &self.entry[..AUDIO_SAMPLE_ENTRY_LEN];
            let body = concat(&[head, &esds(2, oti, dsi.as_deref(), avg, avg), &rate]);
            return bx(&self.entry_typ, &body);
        }
        let mut body = self.entry.clone();
        let has_btrt = self
            .entry
            .get(AUDIO_SAMPLE_ENTRY_LEN..)
            .and_then(children)
            .is_some_and(|k| child(&k, b"btrt").is_some());
        if !has_btrt {
            body.extend_from_slice(&rate);
        }
        bx(&self.entry_typ, &body)
    }

    /// The `roll` sample group, distance -1, over every sample.
    fn aac_roll(&self) -> [Vec<u8>; 2] {
        let sgpd = bx(
            b"sgpd",
            &concat(&[&[1, 0, 0, 0], b"roll", &2u32.to_be_bytes(), &1u32.to_be_bytes(), &(-1i16).to_be_bytes()]),
        );
        let sbgp = bx(
            b"sbgp",
            &concat(&[
                &[0, 0, 0, 0],
                b"roll",
                &1u32.to_be_bytes(),
                &(self.samples.len() as u32).to_be_bytes(),
                &1u32.to_be_bytes(),
            ]),
        );
        [sgpd, sbgp]
    }
}

// ─── Layout ───────────────────────────────────────────────────────────

/// Per-track chunk table built while samples are laid out.
#[derive(Default)]
struct Chunks {
    offsets: Vec<u64>,
    samples_per_chunk: Vec<u32>,
    open_bytes: u64,
}

/// Whether video sample `v` (at `v_ticks` of `v_ts`) is muxed before audio
/// sample at `a_num / (a_ts × 1000)` seconds. Video first on equal times.
fn video_first(v_ticks: u64, v_ts: u32, a_num: i128, a_ts: u32) -> bool {
    let lhs = v_ticks as i128 * a_ts as i128 * MOVIE_TIMESCALE as i128;
    let rhs = a_num * v_ts as i128;
    lhs <= rhs
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build(
    annex_b: &[u8],
    access_units: &[AccessUnit],
    avcc: &AvccData,
    width: u32,
    height: u32,
    timing: FrameTiming,
    composition_offsets: Option<&[i32]>,
    audio: Option<&AudioPassthrough<'_>>,
    lead_in_us: u64,
) -> Result<Vec<u8>, Mp4Error> {
    if let Some((sps_w, sps_h)) = avcc.sps_display_size()
        && (sps_w, sps_h) != (width, height)
    {
        return Err(Mp4Error::InvalidBox(format!(
            "track size {width}x{height} != SPS display size {sps_w}x{sps_h} \
             (pass the cropped display size, not the coded MB-grid size)"
        )));
    }
    if composition_offsets.is_some() {
        // OpenH264 has no B-frames; a reordered stream under its name would
        // contradict itself.
        return Err(Mp4Error::InvalidBox(
            "LavfOpenH264 profile: B-frame streams are not OpenH264 output".into(),
        ));
    }

    // Video samples: the encoder's NAL units, length-prefixed, without the
    // AUDs (the libopenh264 wrapper emits none) and the parameter sets (they
    // live in avcC).
    let samples: Vec<Vec<u8>> = access_units
        .iter()
        .map(|au| {
            let mut buf = Vec::new();
            for &(s, e, t) in &au.nals {
                if matches!(t, NalType::AUD | NalType::SPS | NalType::PPS) {
                    continue;
                }
                buf.extend_from_slice(&((e - s) as u32).to_be_bytes());
                buf.extend_from_slice(&annex_b[s..e]);
            }
            buf
        })
        .collect();
    let (timescale, v_delta) = video_timescale(timing);
    // A late video start, in whole frames.
    let lead_frames = ((lead_in_us as u128 * timing.fps_num as u128
        + timing.fps_den as u128 * 500_000)
        / (timing.fps_den as u128 * 1_000_000)) as u64;
    let video = VideoTrack {
        deltas: vec![v_delta; samples.len()],
        samples,
        sync: access_units
            .iter()
            .enumerate()
            .filter_map(|(i, au)| au.is_sync.then_some(i as u32 + 1))
            .collect(),
        timescale,
        lead_ticks: lead_frames * v_delta as u64,
    };
    // The libopenh264 wrapper publishes CPB properties without an average
    // bitrate, so the muxer writes `btrt` (0, average, 0).
    mux(&[b"isom", b"iso2", b"avc1", b"mp41"], video, width, height, audio, |avg| {
        stsd_avc1(width, height, avcc, &btrt(0, avg, 0))
    })
}

// ─── AV1 (librav1e) ───────────────────────────────────────────────────

/// AV1 video timing as the libavformat transcode has it: `(media timescale,
/// one stts delta per sample, lead-in ticks)`.
///
/// The encoder's time base is `1/rate` ([`video_timescale`]), and each frame
/// keeps its source time rounded to the nearest slot of it: a gap in the
/// source stays a multi-slot delta, jitter under half a slot disappears,
/// and the last frame lasts one slot. The rate is the clip's constant rate
/// when its frame times have one (the closer of an integer rate N and an
/// NTSC rate M·1000/1001 to the mean period, every frame within a quarter
/// period of its slot), else `nominal`, as the tool takes the stream's
/// nominal rate for a variable-rate source. Two frames that round to the
/// same slot take consecutive ones (the tool would drop one; the stego
/// stream must keep every frame). The first frame's slot is the lead-in, a
/// leading empty edit when the video starts after the audio.
///
/// Without one time per frame (or with fewer than two), `nominal` and a
/// uniform delta from 0. Integer and basic f64 arithmetic only, so every
/// platform derives the same timeline.
pub(super) fn av1_timeline(nominal: FrameTiming, n: usize, frame_times_us: Option<&[i64]>) -> (u32, Vec<u32>, u64) {
    let Some(times) = frame_times_us.filter(|t| t.len() == n && n >= 2) else {
        let (ts, delta) = video_timescale(nominal);
        return (ts, vec![delta; n], 0);
    };
    let rate = constant_rate(times).unwrap_or(nominal);
    let (ts, delta) = video_timescale(rate);
    let (num, den) = (rate.fps_num as i128, rate.fps_den as i128);
    // Nearest slot of `1/rate` to `us`, halves away from zero.
    let slot = |us: i64| -> i64 { ((us.max(0) as i128 * num * 2 + den * 1_000_000) / (den * 2_000_000)) as i64 };
    let mut slots: Vec<i64> = Vec::with_capacity(n);
    for &t in times {
        let s = slot(t);
        slots.push(match slots.last() {
            Some(&p) if s <= p => p + 1,
            _ => s,
        });
    }
    let mut deltas: Vec<u32> = slots
        .windows(2)
        .map(|w| ((w[1] - w[0]) as u64 * delta as u64).min(u32::MAX as u64) as u32)
        .collect();
    deltas.push(delta);
    (ts, deltas, slots[0] as u64 * delta as u64)
}

/// The constant rate a clip's frame times follow, if they follow one.
fn constant_rate(times: &[i64]) -> Option<FrameTiming> {
    let n = times.len();
    let mut t = times.to_vec();
    for i in 1..n {
        if t[i] <= t[i - 1] {
            t[i] = t[i - 1] + 1;
        }
    }
    let rel: Vec<i64> = t.iter().map(|&x| x - t[0]).collect();
    let period = rel[n - 1] as f64 / (n - 1) as f64;
    // (rate, exact period in µs) for the two families.
    let mut candidates: Vec<(FrameTiming, f64)> = Vec::new();
    let int_fps = (1_000_000.0 / period).round();
    if (1.0..=1000.0).contains(&int_fps) {
        candidates.push((FrameTiming { fps_num: int_fps as u32, fps_den: 1 }, 1_000_000.0 / int_fps));
    }
    let ntsc_m = (1_001_000.0 / period).round();
    if (1.0..=1000.0).contains(&ntsc_m) {
        candidates.push((FrameTiming { fps_num: ntsc_m as u32 * 1000, fps_den: 1001 }, 1_001_000.0 / ntsc_m));
    }
    candidates
        .into_iter()
        .min_by(|a, b| (period - a.1).abs().total_cmp(&(period - b.1).abs()))
        .filter(|&(_, p)| {
            (period - p).abs() <= p * 0.0005
                && rel.iter().enumerate().all(|(i, &r)| (r as f64 - i as f64 * p).abs() <= p / 4.0)
        })
        .map(|(rate, _)| rate)
}

/// The `av1C` the librav1e wrapper stores: rav1e's container sequence
/// header, copied verbatim (it already starts with the record marker, so
/// the muxer does not rebuild it). rav1e writes level 31, tier 0, chroma
/// sample position 0, no initial presentation delay and no config OBUs (the
/// sequence header travels in the sync samples), and derives the
/// subsampling bits from its chroma format: 4:0:0 is (1, 0).
pub(super) fn av1c_rav1e(sh: Option<&Av1SequenceHeaderInfo>) -> [u8; 4] {
    let (profile, high, twelve, mono, sx, sy) = sh
        .map(|s| (s.seq_profile, s.high_bitdepth, s.twelve_bit, s.monochrome, s.chroma_subsampling_x, s.chroma_subsampling_y))
        .unwrap_or((0, false, false, false, true, true));
    let (sx, sy) = if mono { (true, false) } else { (sx, sy) };
    [
        0x81,
        (profile & 7) << 5 | 31,
        (high as u8) << 6 | (twelve as u8) << 5 | (mono as u8) << 4 | (sx as u8) << 3 | (sy as u8) << 2,
        0,
    ]
}

/// `colr` (nclx) as the muxer writes it: only when primaries, transfer and
/// matrix are all specified, which for a librav1e encode is when the
/// sequence header's colour description says so.
pub(super) fn colr_nclx(sh: Option<&Av1SequenceHeaderInfo>) -> Option<Vec<u8>> {
    let sh = sh?;
    let (cp, tc, mc) = sh.color_description?;
    if cp == 2 || tc == 2 || mc == 2 {
        return None;
    }
    Some(bx(
        b"colr",
        &concat(&[
            b"nclx",
            &(cp as u16).to_be_bytes(),
            &(tc as u16).to_be_bytes(),
            &(mc as u16).to_be_bytes(),
            &[(sh.full_range as u8) << 7],
        ]),
    ))
}

fn stsd_av01(width: u32, height: u32, sh: Option<&Av1SequenceHeaderInfo>, btrt_box: &[u8]) -> Vec<u8> {
    let mut e = visual_sample_entry(width, height, LAVC_COMPRESSOR_AV1);
    e.extend_from_slice(&bx(b"av1C", &av1c_rav1e(sh)));
    // Progressive: the transcoder sets the field order from the decoded
    // frames, and the muxer writes `fiel` for every codec but H.264 / MPEG-4
    // / DNxHD.
    e.extend_from_slice(&bx(b"fiel", &[1, 0]));
    if let Some(colr) = colr_nclx(sh) {
        e.extend_from_slice(&colr);
    }
    e.extend_from_slice(btrt_box);
    bx(b"stsd", &concat(&[&[0, 0, 0, 0], &1u32.to_be_bytes(), &bx(b"av01", &e)]))
}

/// The MP4 around a librav1e encode: `samples` are the temporal units
/// (delimiters dropped, the sequence header in each sync sample), `sync`
/// one flag per sample, `sh` the stream's sequence header, `frame_times_us`
/// the source's frame presentation times (see [`av1_timeline`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_av1(
    samples: Vec<Vec<u8>>,
    sync: &[bool],
    sh: Option<&Av1SequenceHeaderInfo>,
    width: u32,
    height: u32,
    timing: FrameTiming,
    frame_times_us: Option<&[i64]>,
    audio: Option<&AudioPassthrough<'_>>,
) -> Result<Vec<u8>, Mp4Error> {
    let (timescale, deltas, lead_ticks) = av1_timeline(timing, samples.len(), frame_times_us);
    let video = VideoTrack {
        samples,
        sync: sync.iter().enumerate().filter_map(|(i, &s)| s.then_some(i as u32 + 1)).collect(),
        timescale,
        deltas,
        lead_ticks,
    };
    // librav1e publishes no CPB properties: `btrt` (0, average, average).
    mux(&[b"isom", b"av01", b"iso2", b"mp41"], video, width, height, audio, |avg| {
        stsd_av01(width, height, sh, &btrt(0, avg, avg))
    })
}

// ─── Shared track layout ──────────────────────────────────────────────

/// The video track, its codec-specific parts prepared by the caller.
struct VideoTrack {
    /// Sample bytes, in decode order.
    samples: Vec<Vec<u8>>,
    /// 1-based numbers of the sync samples.
    sync: Vec<u32>,
    /// Media timescale and one `stts` delta per sample.
    timescale: u32,
    deltas: Vec<u32>,
    /// A late start against the audio, in media ticks.
    lead_ticks: u64,
}

/// Lays out `video` (and the passed-through `audio`) as the muxer does:
/// `ftyp` with the `compatible` brands, `free`, `mdat`, `moov`. `stsd` builds
/// the video sample description from the track's average bitrate.
fn mux(
    compatible: &[&[u8; 4]],
    video: VideoTrack,
    width: u32,
    height: u32,
    audio: Option<&AudioPassthrough<'_>>,
    stsd: impl Fn(u32) -> Vec<u8>,
) -> Result<Vec<u8>, Mp4Error> {
    let VideoTrack { samples: video, sync, timescale: v_ts, deltas, lead_ticks } = video;
    // Decode time of each video sample, lead-in included.
    let mut v_dts: Vec<u64> = Vec::with_capacity(video.len());
    let mut t = lead_ticks;
    for &d in &deltas {
        v_dts.push(t);
        t += d as u64;
    }
    let v_media = t - lead_ticks;

    let audio_track = match audio {
        Some(a) => {
            let trak = crate::codec::mp4::audio_iso::passthrough_audio_trak(
                &a.trak_raw,
                a.source_movie_timescale,
                MOVIE_TIMESCALE,
            )?;
            Some(AudioTrack::parse(&trak, &a.samples)?)
        }
        None => None,
    };

    // ─── mdat layout: interleave by decode time, chunk contiguous runs ──
    let video_bytes: u64 = video.iter().map(|s| s.len() as u64).sum();
    let audio_bytes = audio_track.as_ref().map_or(0, |a| a.bytes());
    let payload = video_bytes + audio_bytes;
    // `free` + 8-byte `mdat` header, or the 16-byte header in their place.
    let large = payload + 8 > u32::MAX as u64;
    let mut brands: Vec<&[u8]> = vec![b"isom", &[0, 0, 2, 0]];
    brands.extend(compatible.iter().map(|b| &b[..]));
    let ftyp = bx(b"ftyp", &concat(&brands));
    let first = ftyp.len() as u64 + 16;

    let mut order: Vec<(bool, usize)> = Vec::with_capacity(video.len() + audio_track.as_ref().map_or(0, |a| a.samples.len()));
    {
        let (a_delay_ms, a_media_start, a_ts, a_durs): (u64, u64, u32, &[u32]) = match &audio_track {
            Some(a) => {
                let (d, m) = a.edits.start();
                (d, m, a.timescale, &a.durations)
            }
            None => (0, 0, 1, &[]),
        };
        let (mut vi, mut ai) = (0usize, 0usize);
        let mut a_dts: u64 = 0;
        while vi < video.len() || ai < a_durs.len() {
            let take_video = if ai >= a_durs.len() {
                true
            } else if vi >= video.len() {
                false
            } else {
                // Audio decode time, seconds × a_ts × 1000.
                let a_num = a_delay_ms as i128 * a_ts as i128
                    + (a_dts as i128 - a_media_start as i128) * MOVIE_TIMESCALE as i128;
                video_first(v_dts[vi], v_ts, a_num, a_ts)
            };
            if take_video {
                order.push((true, vi));
                vi += 1;
            } else {
                order.push((false, ai));
                a_dts += a_durs[ai] as u64;
                ai += 1;
            }
        }
    }

    let (mut vc, mut ac) = (Chunks::default(), Chunks::default());
    let mut cursor = first;
    let mut last_video: Option<bool> = None;
    for &(is_video, i) in &order {
        let size = if is_video {
            video[i].len() as u64
        } else {
            audio_track.as_ref().map_or(0, |a| a.samples[i].len() as u64)
        };
        let c = if is_video { &mut vc } else { &mut ac };
        if last_video == Some(is_video) && c.open_bytes + size < MAX_CHUNK_BYTES {
            *c.samples_per_chunk.last_mut().unwrap() += 1;
            c.open_bytes += size;
        } else {
            c.offsets.push(cursor);
            c.samples_per_chunk.push(1);
            c.open_bytes = size;
        }
        last_video = Some(is_video);
        cursor += size;
    }

    // ─── Video trak ─────────────────────────────────────────────────────
    let v_seg = to_movie_ceil(v_media, v_ts);
    let lead_ms = to_movie_floor(lead_ticks, v_ts);
    let (v_edits, v_tkhd_dur) = if lead_ms > 0 {
        (vec![(lead_ms, -1), (v_seg, 0)], to_movie_ceil(lead_ticks + v_media, v_ts))
    } else {
        (vec![(v_seg, 0)], v_seg)
    };
    let sizes: Vec<u32> = video.iter().map(|s| s.len() as u32).collect();
    let v_stbl = {
        let mut s = stsd(avg_bitrate(video_bytes, v_ts, v_media));
        s.extend_from_slice(&build_stts(&deltas));
        if sync.len() != video.len() {
            s.extend_from_slice(&build_stss(&sync));
        }
        s.extend_from_slice(&build_stsc(&vc.samples_per_chunk));
        s.extend_from_slice(&stsz(&sizes));
        s.extend_from_slice(&build_stco(&vc.offsets));
        bx(b"stbl", &s)
    };
    let v_trak = bx(
        b"trak",
        &concat(&[
            &tkhd(1, v_tkhd_dur, false, width, height),
            &edts(&v_edits),
            &bx(
                b"mdia",
                &concat(&[
                    &mdhd(v_media, v_ts, LANG_UND),
                    &build_hdlr_vide(),
                    &bx(b"minf", &concat(&[&build_vmhd(), &build_dinf(), &v_stbl])),
                ]),
            ),
        ]),
    );

    // ─── Audio trak ─────────────────────────────────────────────────────
    let (a_trak, a_tkhd_dur) = match &audio_track {
        Some(a) => {
            let media = a.media_duration();
            let (edits, dur) = a.edits.output(media, a.timescale);
            let avg = avg_bitrate(a.bytes(), a.timescale, media);
            let sizes: Vec<u32> = a.samples.iter().map(|s| s.len() as u32).collect();
            let mut s = bx(b"stsd", &concat(&[&[0, 0, 0, 0], &1u32.to_be_bytes(), &a.sample_entry(avg)]));
            s.extend_from_slice(&build_stts(&a.durations));
            if let Some(stss) = &a.stss {
                s.extend_from_slice(stss);
            }
            s.extend_from_slice(&build_stsc(&ac.samples_per_chunk));
            s.extend_from_slice(&stsz(&sizes));
            s.extend_from_slice(&build_stco(&ac.offsets));
            if a.is_aac() {
                for g in a.aac_roll() {
                    s.extend_from_slice(&g);
                }
            } else {
                for g in &a.groups {
                    s.extend_from_slice(g);
                }
            }
            let trak = bx(
                b"trak",
                &concat(&[
                    &tkhd(2, dur, true, 0, 0),
                    &edts(&edits),
                    &bx(
                        b"mdia",
                        &concat(&[
                            &mdhd(media, a.timescale, a.language),
                            &hdlr_soun(),
                            &bx(b"minf", &concat(&[&smhd(), &build_dinf(), &bx(b"stbl", &s)])),
                        ]),
                    ),
                ]),
            );
            (Some(trak), dur)
        }
        None => (None, 0),
    };

    let movie_dur = v_tkhd_dur.max(a_tkhd_dur);
    let mut moov = build_mvhd(movie_dur, if a_trak.is_some() { 3 } else { 2 });
    moov.extend_from_slice(&v_trak);
    if let Some(t) = &a_trak {
        moov.extend_from_slice(t);
    }
    moov.extend_from_slice(&udta());
    let moov = bx(b"moov", &moov);

    // ─── Assemble ───────────────────────────────────────────────────────
    let mut out = Vec::with_capacity((first + payload) as usize + moov.len());
    out.extend_from_slice(&ftyp);
    if large {
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(b"mdat");
        out.extend_from_slice(&(payload + 16).to_be_bytes());
    } else {
        out.extend_from_slice(&8u32.to_be_bytes());
        out.extend_from_slice(b"free");
        out.extend_from_slice(&((payload + 8) as u32).to_be_bytes());
        out.extend_from_slice(b"mdat");
    }
    for &(is_video, i) in &order {
        if is_video {
            out.extend_from_slice(&video[i]);
        } else if let Some(a) = &audio_track {
            out.extend_from_slice(a.samples[i]);
        }
    }
    out.extend_from_slice(&moov);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(profile: u8, high: bool, mono: bool, sub: (bool, bool), colour: Option<(u8, u8, u8)>, full: bool) -> Av1SequenceHeaderInfo {
        Av1SequenceHeaderInfo {
            max_frame_width: 64,
            max_frame_height: 64,
            seq_profile: profile,
            seq_level_idx_0: 8,
            seq_tier_0: 0,
            high_bitdepth: high,
            twelve_bit: false,
            monochrome: mono,
            chroma_subsampling_x: sub.0,
            chroma_subsampling_y: sub.1,
            chroma_sample_position: 1,
            color_description: colour,
            full_range: full,
        }
    }

    #[test]
    fn av1c_is_rav1es_container_record() {
        // Level 31 and chroma sample position 0 whatever the stream says.
        assert_eq!(av1c_rav1e(Some(&sh(0, false, false, (true, true), None, false))), [0x81, 0x1F, 0x0C, 0x00]);
        assert_eq!(av1c_rav1e(None), [0x81, 0x1F, 0x0C, 0x00]);
        // 10-bit 4:4:4 (profile 1).
        assert_eq!(av1c_rav1e(Some(&sh(1, true, false, (false, false), None, false))), [0x81, 0x3F, 0x40, 0x00]);
        // 4:0:0: rav1e writes subsampling (1, 0).
        assert_eq!(av1c_rav1e(Some(&sh(0, false, true, (true, true), None, false))), [0x81, 0x1F, 0x18, 0x00]);
    }

    #[test]
    fn colr_only_for_a_specified_colour_description() {
        assert_eq!(
            colr_nclx(Some(&sh(0, false, false, (true, true), Some((1, 1, 1)), false))),
            Some(bx(b"colr", b"nclx\0\x01\0\x01\0\x01\0"))
        );
        assert_eq!(
            colr_nclx(Some(&sh(0, false, false, (true, true), Some((9, 16, 9)), true))),
            Some(bx(b"colr", b"nclx\0\x09\0\x10\0\x09\x80"))
        );
        // Absent, or any of the three unspecified (2): no colr.
        assert_eq!(colr_nclx(Some(&sh(0, false, false, (true, true), None, false))), None);
        assert_eq!(colr_nclx(Some(&sh(0, false, false, (true, true), Some((2, 2, 2)), false))), None);
        assert_eq!(colr_nclx(Some(&sh(0, false, false, (true, true), Some((1, 2, 1)), false))), None);
        assert_eq!(colr_nclx(None), None);
    }

    #[test]
    fn both_compressor_names_are_the_same_libavcodec() {
        let lavc = |s: &str| s.split(' ').next().unwrap().to_string();
        assert_eq!(lavc(LAVC_COMPRESSOR), lavc(LAVC_COMPRESSOR_AV1));
        assert!(LAVF_TOOL.starts_with("Lavf"));
    }

    #[test]
    fn video_timescale_matches_the_muxer() {
        let t = |n, d| video_timescale(FrameTiming { fps_num: n, fps_den: d });
        assert_eq!(t(30, 1), (15360, 512));
        assert_eq!(t(24, 1), (12288, 512));
        assert_eq!(t(25, 1), (12800, 512));
        assert_eq!(t(60, 1), (15360, 256));
        assert_eq!(t(30000, 1001), (30000, 1001));
        assert_eq!(t(24000, 1001), (24000, 1001));
        assert_eq!(t(60000, 2002), (30000, 1001)); // reduced first
    }

    #[test]
    fn movie_time_rounding() {
        assert_eq!(to_movie_ceil(25600, 12288), 2084);
        assert_eq!(to_movie_floor(1024, 12288), 83);
        assert_eq!(to_movie_ceil(46080, 15360), 3000);
    }

    #[test]
    fn languages_become_iso() {
        assert_eq!(iso_language(0x7FFF), LANG_UND);
        assert_eq!(iso_language(0), LANG_ENG);
        assert_eq!(iso_language(LANG_UND), LANG_UND);
        assert_eq!(iso_language(0x15C7), 0x15C7);
    }

    /// The `esds` the muxer wrote for a copied mono AAC-LC track (corpus
    /// iPhone 7 clip): ES_ID 2, bitrates (0, 90537, 90537), ASC `12 08`.
    #[test]
    fn esds_matches_the_muxer_bytes() {
        let want = "000000336573647300000000038080802200020004808080144015000000000161a9000161a905808080021208068080800102";
        let got = esds(2, 0x40, Some(&[0x12, 0x08]), 0x161a9, 0x161a9);
        let hex: String = got.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, want);
        // And it parses back.
        let (oti, dsi) = parse_esds(&got[8..]).unwrap();
        assert_eq!((oti, dsi.as_deref()), (0x40, Some(&[0x12u8, 0x08][..])));
    }

    #[test]
    fn udta_matches_the_muxer_layout() {
        let u = udta();
        // udta(meta(full box, hdlr mdir/appl, ilst(©too(data UTF-8 "Lavf…")))).
        assert_eq!(&u[4..8], b"udta");
        let hex: String = u[..74].iter().map(|b| format!("{b:02x}")).collect();
        assert!(hex.starts_with("00000062756474610000005a6d657461000000000000002168646c7200000000000000006d6469726170706c"));
        assert!(u.ends_with(LAVF_TOOL.as_bytes()));
    }

    #[test]
    fn audio_edits_parse() {
        let elst = |v: &[(u32, i32)]| {
            let mut p = vec![0u8; 4];
            p.extend_from_slice(&(v.len() as u32).to_be_bytes());
            for &(d, m) in v {
                p.extend_from_slice(&d.to_be_bytes());
                p.extend_from_slice(&m.to_be_bytes());
                p.extend_from_slice(&0x0001_0000u32.to_be_bytes());
            }
            p
        };
        assert!(matches!(AudioEdits::parse(None), Some(AudioEdits::Plain { media_time: 0 })));
        assert!(matches!(
            AudioEdits::parse(Some(&elst(&[(2300, 1024)]))),
            Some(AudioEdits::Plain { media_time: 1024 })
        ));
        assert!(matches!(
            AudioEdits::parse(Some(&elst(&[(83, -1), (2000, 0)]))),
            Some(AudioEdits::Delayed { delay_ms: 83, media_time: 0 })
        ));
        assert!(matches!(
            AudioEdits::parse(Some(&elst(&[(1000, 0), (1000, 5000)]))),
            Some(AudioEdits::Verbatim(_))
        ));
        // Plain: segment = media after the priming, rounded up.
        let (e, d) = AudioEdits::Plain { media_time: 1024 }.output(102_454, 44_100);
        assert_eq!((e, d), (vec![(2300, 1024)], 2300));
    }
}
