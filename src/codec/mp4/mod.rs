// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! MP4/ISOBMFF container demux and mux.
//!
//! Zero-dependency MP4 parser and writer for extracting and replacing
//! HEVC NAL units in video files. Handles the ISO Base Media File Format
//! box hierarchy, sample tables, and HEVC decoder configuration records.
//!
//! # Overview
//!
//! - [`demux`] — Parse MP4 file → [`Mp4File`] with tracks, samples, hvcC data
//! - [`mux`] — Rebuild MP4 with modified video samples, corrected sample tables
//!
//! # Limitations
//!
//! - Only handles single-fragment MP4 files (no fragmented MP4/fMP4)
//! - Requires `mdat` box to be after `moov` for muxing (standard for camera output)

pub mod demux;
pub mod mux;

use std::fmt;

/// MP4 container error.
#[derive(Debug, Clone)]
pub enum Mp4Error {
    /// Unexpected end of file while parsing.
    UnexpectedEof,
    /// Invalid or malformed box structure.
    InvalidBox(String),
    /// No video track found in the MP4 file.
    NoVideoTrack,
    /// Video track codec is not supported (not HEVC or H.264).
    UnsupportedCodec,
    /// File appears truncated (mdat overshoots file, moov missing or incomplete).
    TruncatedFile,
    /// Fragmented MP4 (moof+mdat) is not supported — only flat MP4 with moov+mdat.
    FragmentedMp4,
}

impl fmt::Display for Mp4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of MP4 data"),
            Self::InvalidBox(s) => write!(f, "invalid MP4 box: {s}"),
            Self::NoVideoTrack => write!(f, "no video track in MP4 file"),
            Self::UnsupportedCodec => write!(f, "video track codec not supported (need HEVC or H.264)"),
            Self::TruncatedFile => write!(f, "MP4 file appears truncated (incomplete recording?)"),
            Self::FragmentedMp4 => write!(f, "fragmented MP4 (fMP4/DASH) is not supported"),
        }
    }
}

impl std::error::Error for Mp4Error {}

/// Check if a byte slice starts with an MP4/ISOBMFF ftyp box.
///
/// Looks for the `ftyp` four-character code at bytes 4..8. This is the
/// standard way to identify ISO Base Media File Format files (MP4, MOV, etc.).
pub fn is_mp4(data: &[u8]) -> bool {
    data.len() >= 8 && &data[4..8] == b"ftyp"
}

/// Parsed MP4 file structure.
#[derive(Debug, Clone)]
pub struct Mp4File {
    /// Raw ftyp box bytes.
    pub ftyp: Vec<u8>,
    /// All tracks in the file (video, audio, subtitle, etc.).
    pub tracks: Vec<Track>,
    /// Index of the HEVC video track in `tracks`, if found.
    pub video_track_idx: Option<usize>,
}

/// A single track (video, audio, subtitle, etc.).
#[derive(Debug, Clone)]
pub struct Track {
    /// Track ID from tkhd box.
    pub track_id: u32,
    /// Handler type from hdlr box: `b"vide"`, `b"soun"`, `b"sbtl"`, etc.
    pub handler_type: [u8; 4],
    /// Codec four-character code from stsd: `b"hev1"`, `b"hvc1"`, `b"mp4a"`, etc.
    pub codec: [u8; 4],
    /// Video width (from tkhd or stsd visual sample entry). 0 for non-video.
    pub width: u32,
    /// Video height (from tkhd or stsd visual sample entry). 0 for non-video.
    pub height: u32,
    /// Media timescale from mdhd.
    pub timescale: u32,
    /// Track duration in media timescale units from mdhd.
    pub duration: u64,
    /// All samples in this track.
    pub samples: Vec<Sample>,
    /// HEVC decoder configuration (only for hev1/hvc1 tracks).
    pub hvcc_data: Option<HvccData>,
    /// H.264/AVC decoder configuration (only for avc1/avc3 tracks).
    pub avcc_data: Option<AvccData>,
    /// Raw bytes of the stsd box (for non-video track passthrough during mux).
    pub stsd_raw: Vec<u8>,
    /// Raw bytes of the complete trak box (for non-video track passthrough).
    pub trak_raw: Vec<u8>,
}

/// A single sample (frame) within a track.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Byte offset of this sample in the original MP4 file.
    pub offset: u64,
    /// Size of this sample in bytes.
    pub size: u32,
    /// True for sync (I-frame) samples (from stss box, or all-sync if stss absent).
    pub is_sync: bool,
    /// Raw sample data. Populated by demux from the mdat region.
    pub data: Vec<u8>,
}

/// HEVC Decoder Configuration Record (hvcC box payload).
///
/// Contains the parameter set NAL units (VPS, SPS, PPS) and the NAL length
/// size used in the sample data.
#[derive(Debug, Clone)]
pub struct HvccData {
    /// Configuration version (always 1).
    pub configuration_version: u8,
    /// NAL unit length field size minus 1 (typically 3, meaning 4-byte lengths).
    pub length_size_minus1: u8,
    /// Video Parameter Set NAL units.
    pub vps_nalus: Vec<Vec<u8>>,
    /// Sequence Parameter Set NAL units.
    pub sps_nalus: Vec<Vec<u8>>,
    /// Picture Parameter Set NAL units.
    pub pps_nalus: Vec<Vec<u8>>,
}

/// AVC (H.264) Decoder Configuration Record (avcC box payload).
///
/// Contains the parameter set NAL units (SPS, PPS) and the NAL length
/// size used in the sample data.
#[derive(Debug, Clone)]
pub struct AvccData {
    /// Configuration version (always 1).
    pub configuration_version: u8,
    /// AVC profile indication.
    pub profile: u8,
    /// Profile compatibility flags.
    pub profile_compat: u8,
    /// AVC level indication.
    pub level: u8,
    /// NAL unit length field size minus 1 (typically 3 → 4-byte lengths).
    pub length_size_minus1: u8,
    /// Sequence Parameter Set NAL units.
    pub sps_nalus: Vec<Vec<u8>>,
    /// Picture Parameter Set NAL units.
    pub pps_nalus: Vec<Vec<u8>>,
}

impl AvccData {
    /// Construct an `AvccData` from an Annex-B H.264 byte stream by
    /// finding the first SPS + first PPS NAL and extracting the three
    /// profile/level bytes from the SPS body. Used to pack our
    /// encoder's output into an MP4 container (`avcC` extradata box).
    pub fn from_annexb(bytes: &[u8]) -> Option<Self> {
        let mut sps: Option<Vec<u8>> = None;
        let mut pps: Option<Vec<u8>> = None;
        for nal in iter_annexb_nals(bytes) {
            if nal.is_empty() {
                continue;
            }
            let nal_type = nal[0] & 0x1F;
            if nal_type == 7 && sps.is_none() {
                sps = Some(nal.to_vec());
            } else if nal_type == 8 && pps.is_none() {
                pps = Some(nal.to_vec());
            }
            if sps.is_some() && pps.is_some() {
                break;
            }
        }
        let sps = sps?;
        let pps = pps?;
        // SPS layout (after the 1-byte NAL header):
        //   profile_idc (u8), constraint flags (u8), level_idc (u8), ...
        if sps.len() < 4 {
            return None;
        }
        Some(AvccData {
            configuration_version: 1,
            profile: sps[1],
            profile_compat: sps[2],
            level: sps[3],
            length_size_minus1: 3,
            sps_nalus: vec![sps],
            pps_nalus: vec![pps],
        })
    }

    /// Serialize this `AvccData` into the raw bytes of an MP4 `avcC`
    /// box payload (NOT including the outer 4-byte size + 4-byte
    /// "avcC" fourcc header — caller wraps that).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32);
        out.push(self.configuration_version);
        out.push(self.profile);
        out.push(self.profile_compat);
        out.push(self.level);
        // Reserved top 6 bits + 2-bit lengthSizeMinusOne.
        out.push(0xFC | (self.length_size_minus1 & 0x03));
        // Reserved top 3 bits + 5-bit numOfSequenceParameterSets.
        out.push(0xE0 | (self.sps_nalus.len() as u8 & 0x1F));
        for sps in &self.sps_nalus {
            out.push((sps.len() >> 8) as u8);
            out.push((sps.len() & 0xFF) as u8);
            out.extend_from_slice(sps);
        }
        out.push(self.pps_nalus.len() as u8);
        for pps in &self.pps_nalus {
            out.push((pps.len() >> 8) as u8);
            out.push((pps.len() & 0xFF) as u8);
            out.extend_from_slice(pps);
        }
        out
    }
}

/// Iterate over Annex-B NAL unit payloads (start-code-prefixed).
/// Yields `&[u8]` slices into `bytes` for each NAL's body (excluding
/// the start-code prefix).
fn iter_annexb_nals(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut starts: Vec<usize> = Vec::new();
    let mut i = 0;
    while i + 3 <= bytes.len() {
        if bytes[i] == 0 && bytes[i + 1] == 0 {
            if i + 4 <= bytes.len() && bytes[i + 2] == 0 && bytes[i + 3] == 1 {
                starts.push(i + 4);
                i += 4;
                continue;
            } else if bytes[i + 2] == 1 {
                starts.push(i + 3);
                i += 3;
                continue;
            }
        }
        i += 1;
    }
    let mut pairs = Vec::new();
    for (k, &s) in starts.iter().enumerate() {
        let end = if k + 1 < starts.len() {
            // The end is just before the next start code.
            let next = starts[k + 1];
            next - (if next >= 4 && bytes[next - 4..next] == [0, 0, 0, 1] { 4 } else { 3 })
        } else {
            bytes.len()
        };
        pairs.push((s, end));
    }
    pairs.into_iter().map(move |(s, e)| &bytes[s..e])
}

impl Track {
    /// True if this is an H.264/AVC video track.
    pub fn is_h264(&self) -> bool {
        self.codec == *b"avc1" || self.codec == *b"avc3"
    }

    /// True if this is an HEVC/H.265 video track.
    pub fn is_hevc(&self) -> bool {
        self.codec == *b"hev1" || self.codec == *b"hvc1"
    }
}

// ─── Internal helpers ────────────────────────────────────────────────

/// Four-character code as `[u8; 4]`.
pub(crate) fn fourcc(data: &[u8], offset: usize) -> Result<[u8; 4], Mp4Error> {
    if offset + 4 > data.len() {
        return Err(Mp4Error::UnexpectedEof);
    }
    Ok([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
}

/// Read big-endian u16.
pub(crate) fn read_u16(data: &[u8], offset: usize) -> Result<u16, Mp4Error> {
    if offset + 2 > data.len() {
        return Err(Mp4Error::UnexpectedEof);
    }
    Ok(u16::from_be_bytes([data[offset], data[offset + 1]]))
}

/// Read big-endian u32.
pub(crate) fn read_u32(data: &[u8], offset: usize) -> Result<u32, Mp4Error> {
    if offset + 4 > data.len() {
        return Err(Mp4Error::UnexpectedEof);
    }
    Ok(u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Read big-endian u64.
pub(crate) fn read_u64(data: &[u8], offset: usize) -> Result<u64, Mp4Error> {
    if offset + 8 > data.len() {
        return Err(Mp4Error::UnexpectedEof);
    }
    Ok(u64::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]))
}

/// Write big-endian u32 into a buffer.
pub(crate) fn write_u32(buf: &mut Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_be_bytes());
}

/// Write big-endian u64 into a buffer.
pub(crate) fn write_u64(buf: &mut Vec<u8>, val: u64) {
    buf.extend_from_slice(&val.to_be_bytes());
}

/// Parsed box header: type code, total box size (including header), header length,
/// and the offset where box content begins.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BoxHeader {
    pub box_type: [u8; 4],
    /// Total size of the box including the header.
    pub size: u64,
    /// Length of the header itself (8 for 32-bit size, 16 for 64-bit extended size).
    pub header_len: u8,
}

/// Parse a box header at the given offset. Returns the header and advances past it.
pub(crate) fn parse_box_header(data: &[u8], offset: usize) -> Result<BoxHeader, Mp4Error> {
    if offset + 8 > data.len() {
        return Err(Mp4Error::UnexpectedEof);
    }
    let size32 = read_u32(data, offset)?;
    let box_type = fourcc(data, offset + 4)?;

    if size32 == 1 {
        // 64-bit extended size
        if offset + 16 > data.len() {
            return Err(Mp4Error::UnexpectedEof);
        }
        let size64 = read_u64(data, offset + 8)?;
        Ok(BoxHeader {
            box_type,
            size: size64,
            header_len: 16,
        })
    } else if size32 == 0 {
        // Box extends to end of file
        let size = (data.len() - offset) as u64;
        Ok(BoxHeader {
            box_type,
            size,
            header_len: 8,
        })
    } else {
        Ok(BoxHeader {
            box_type,
            size: size32 as u64,
            header_len: 8,
        })
    }
}

/// Iterate over child boxes within a given range, calling a visitor for each.
pub(crate) fn iterate_boxes<F>(
    data: &[u8],
    start: usize,
    end: usize,
    mut visitor: F,
) -> Result<(), Mp4Error>
where
    F: FnMut(&BoxHeader, usize, &[u8]) -> Result<(), Mp4Error>,
{
    let mut pos = start;
    while pos < end {
        if pos + 8 > end {
            break; // Not enough data for another box header
        }
        let header = parse_box_header(data, pos)?;
        if header.size < 8 {
            return Err(Mp4Error::InvalidBox(format!(
                "box {:?} has invalid size {}",
                std::str::from_utf8(&header.box_type).unwrap_or("????"),
                header.size
            )));
        }
        let box_end = pos + header.size as usize;
        if box_end > end {
            return Err(Mp4Error::UnexpectedEof);
        }
        let content_start = pos + header.header_len as usize;
        visitor(&header, content_start, &data[pos..box_end])?;
        pos = box_end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avcc_from_annexb_extracts_profile_level() {
        // Synthetic Annex-B stream: [0,0,0,1] start + SPS NAL + [0,0,0,1] + PPS NAL.
        // SPS NAL: header byte 0x67 (nal_ref_idc=3, nal_type=7) +
        //          profile_idc (0x42 = Baseline), constraint (0x00), level (0x1E = 30).
        let sps = vec![0x67, 0x42, 0x00, 0x1E, 0xAC, 0xD9];
        let pps = vec![0x68, 0xEB, 0xE3, 0xCB];
        let mut bytes = vec![0, 0, 0, 1];
        bytes.extend(&sps);
        bytes.extend([0, 0, 0, 1]);
        bytes.extend(&pps);

        let avcc = AvccData::from_annexb(&bytes).expect("should find SPS + PPS");
        assert_eq!(avcc.profile, 0x42);
        assert_eq!(avcc.profile_compat, 0x00);
        assert_eq!(avcc.level, 0x1E);
        assert_eq!(avcc.length_size_minus1, 3);
        assert_eq!(avcc.sps_nalus.len(), 1);
        assert_eq!(avcc.pps_nalus.len(), 1);
        assert_eq!(avcc.sps_nalus[0], sps);
        assert_eq!(avcc.pps_nalus[0], pps);
    }

    #[test]
    fn avcc_from_annexb_missing_sps_or_pps_returns_none() {
        // Only SPS, no PPS.
        let bytes = vec![0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1E];
        assert!(AvccData::from_annexb(&bytes).is_none());
    }

    #[test]
    fn avcc_to_bytes_roundtrip() {
        // Serialize a known AvccData and verify the first bytes match
        // spec layout exactly.
        let avcc = AvccData {
            configuration_version: 1,
            profile: 0x42,
            profile_compat: 0x40,
            level: 0x1E,
            length_size_minus1: 3,
            sps_nalus: vec![vec![0x67, 0x42, 0x40, 0x1E]],
            pps_nalus: vec![vec![0x68, 0xCE, 0x38, 0x80]],
        };
        let bytes = avcc.to_bytes();
        assert_eq!(bytes[0], 1); // configuration_version
        assert_eq!(bytes[1], 0x42); // profile
        assert_eq!(bytes[2], 0x40); // profile_compat
        assert_eq!(bytes[3], 0x1E); // level
        assert_eq!(bytes[4], 0xFF); // reserved (0xFC) | lengthSizeMinus1 (3)
        assert_eq!(bytes[5], 0xE1); // reserved (0xE0) | numSPS (1)
        assert_eq!(bytes[6], 0); // SPS length hi
        assert_eq!(bytes[7], 4); // SPS length lo
        assert_eq!(&bytes[8..12], &[0x67, 0x42, 0x40, 0x1E]);
        assert_eq!(bytes[12], 1); // numPPS
        assert_eq!(bytes[13], 0); // PPS length hi
        assert_eq!(bytes[14], 4); // PPS length lo
        assert_eq!(&bytes[15..19], &[0x68, 0xCE, 0x38, 0x80]);
    }

    #[test]
    fn test_parse_box_header_32bit() {
        // 32-bit size box: size=20, type='ftyp'
        let data = [
            0x00, 0x00, 0x00, 0x14, // size = 20
            b'f', b't', b'y', b'p', // type
            0x00, 0x00, 0x00, 0x00, // content...
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let h = parse_box_header(&data, 0).unwrap();
        assert_eq!(h.box_type, *b"ftyp");
        assert_eq!(h.size, 20);
        assert_eq!(h.header_len, 8);
    }

    #[test]
    fn test_parse_box_header_64bit() {
        // 64-bit extended size box: size32=1, type='mdat', size64=0x100
        let data = [
            0x00, 0x00, 0x00, 0x01, // size = 1 → extended
            b'm', b'd', b'a', b't', // type
            0x00, 0x00, 0x00, 0x00, // extended size high
            0x00, 0x00, 0x01, 0x00, // extended size low = 256
            0x00, 0x00, 0x00, 0x00, // content padding
        ];
        let h = parse_box_header(&data, 0).unwrap();
        assert_eq!(h.box_type, *b"mdat");
        assert_eq!(h.size, 256);
        assert_eq!(h.header_len, 16);
    }

    #[test]
    fn test_parse_box_header_to_eof() {
        // size=0 means extends to end of data
        let data = [
            0x00, 0x00, 0x00, 0x00, // size = 0 → to EOF
            b'm', b'd', b'a', b't', // type
            0xAA, 0xBB, 0xCC, 0xDD, // some content
        ];
        let h = parse_box_header(&data, 0).unwrap();
        assert_eq!(h.box_type, *b"mdat");
        assert_eq!(h.size, 12); // entire data length
        assert_eq!(h.header_len, 8);
    }

    #[test]
    fn test_iterate_boxes() {
        // Two boxes: ftyp(16 bytes) + free(12 bytes)
        let mut data = Vec::new();
        // ftyp box: size=16
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x10]);
        data.extend_from_slice(b"ftyp");
        data.extend_from_slice(&[0x69, 0x73, 0x6F, 0x6D]); // isom
        data.extend_from_slice(&[0x00, 0x00, 0x02, 0x00]); // minor version
        // free box: size=12
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x0C]);
        data.extend_from_slice(b"free");
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

        let mut types = Vec::new();
        iterate_boxes(&data, 0, data.len(), |header, _content_start, _box_data| {
            types.push(header.box_type);
            Ok(())
        })
        .unwrap();
        assert_eq!(types, vec![*b"ftyp", *b"free"]);
    }

    #[test]
    fn test_read_helpers() {
        let data = [0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03];
        assert_eq!(read_u16(&data, 0).unwrap(), 1);
        assert_eq!(read_u32(&data, 2).unwrap(), 2);
    }
}
