// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! MP4 demuxer — parse an MP4 file into tracks, samples, and HEVC parameter sets.
//!
//! Parses the ISOBMFF box hierarchy to extract:
//! - File type (ftyp)
//! - All tracks with metadata (trak/tkhd/mdia/mdhd/hdlr)
//! - Sample tables (stts, stsc, stsz, stco/co64, stss)
//! - HEVC decoder configuration (hvcC → VPS/SPS/PPS)
//! - Sample data from mdat

use super::{
    fourcc, iterate_boxes, parse_box_header, read_u16, read_u32, read_u64, AvccData, HvccData,
    Mp4Error, Mp4File, Sample, Track,
};

/// Parse an entire MP4 file into an [`Mp4File`] structure.
///
/// Extracts all tracks, identifies the HEVC video track, parses sample tables,
/// and reads sample data from the mdat box.
pub fn demux(data: &[u8]) -> Result<Mp4File, Mp4Error> {
    let mut ftyp: Option<Vec<u8>> = None;
    let mut moov_range: Option<(usize, usize)> = None;
    let mut has_moof = false;

    // First pass: locate top-level boxes.
    // Use tolerant iteration — some files have mdat boxes whose declared size
    // overshoots the actual file (truncated recordings, e.g. DJI Mavic).
    // We skip oversized boxes rather than failing, so we can still find moov.
    let file_end = data.len();
    let mut pos = 0usize;
    while pos < file_end {
        if pos + 8 > file_end {
            break;
        }
        let header = parse_box_header(data, pos)?;
        if header.size < 8 {
            break;
        }
        let box_end = pos + header.size as usize;
        let _content_start = pos + header.header_len as usize;

        match &header.box_type {
            b"ftyp" => {
                let safe_end = box_end.min(file_end);
                ftyp = Some(data[pos..safe_end].to_vec());
            }
            b"moov" => {
                let safe_end = box_end.min(file_end);
                moov_range = Some((pos, safe_end));
            }
            b"moof" => {
                has_moof = true;
            }
            _ => {} // skip mdat, free, wide, etc.
        }

        if box_end > file_end {
            // Box overshoots file — skip to end (file may be truncated)
            break;
        }
        pos = box_end;
    }

    // Detect fragmented MP4 (DASH/HLS — uses moof+mdat instead of moov+mdat)
    if has_moof {
        return Err(Mp4Error::FragmentedMp4);
    }

    let ftyp = ftyp.ok_or_else(|| Mp4Error::InvalidBox("missing ftyp box".into()))?;
    let (moov_start, moov_end) = match moov_range {
        Some(r) => r,
        None => {
            // No moov box found — file is likely truncated (e.g. interrupted drone recording).
            // Attempt recovery by scanning the raw mdat for HEVC NAL units.
            return recover_truncated(data, &ftyp);
        }
    };

    // Parse moov → extract tracks
    let moov_header = parse_box_header(data, moov_start)?;
    let moov_content_start = moov_start + moov_header.header_len as usize;
    let tracks = parse_moov(data, moov_content_start, moov_end)?;

    // Find HEVC or H.264 video track
    let video_track_idx = tracks.iter().position(|t| {
        t.handler_type == *b"vide" && (t.is_hevc() || t.is_h264())
    });

    Ok(Mp4File {
        ftyp,
        tracks,
        video_track_idx,
    })
}

/// Parse the moov box contents → list of tracks.
fn parse_moov(data: &[u8], start: usize, end: usize) -> Result<Vec<Track>, Mp4Error> {
    let mut tracks = Vec::new();

    iterate_boxes(data, start, end, |header, content_start, box_data| {
        if header.box_type == *b"trak" {
            let trak_start = content_start - header.header_len as usize;
            let trak_end = trak_start + header.size as usize;
            let track = parse_trak(data, content_start, trak_end, box_data)?;
            tracks.push(track);
        }
        Ok(())
    })?;

    Ok(tracks)
}

/// Parse a single trak box → Track.
fn parse_trak(
    data: &[u8],
    start: usize,
    end: usize,
    trak_raw_data: &[u8],
) -> Result<Track, Mp4Error> {
    let mut track = Track {
        track_id: 0,
        handler_type: [0; 4],
        codec: [0; 4],
        width: 0,
        height: 0,
        timescale: 0,
        duration: 0,
        samples: Vec::new(),
        hvcc_data: None,
            avcc_data: None,
        stsd_raw: Vec::new(),
        trak_raw: trak_raw_data.to_vec(),
    };

    // Intermediate sample table data collected during parsing
    let mut sample_sizes: Vec<u32> = Vec::new();
    let mut sample_offsets: Vec<u64> = Vec::new();
    let mut sync_samples: Option<Vec<u32>> = None; // None = all samples are sync
    let mut stsc_entries: Vec<(u32, u32, u32)> = Vec::new(); // (first_chunk, samples_per_chunk, _desc_idx)

    // Parse trak children recursively
    parse_trak_children(
        data,
        start,
        end,
        &mut track,
        &mut sample_sizes,
        &mut sample_offsets,
        &mut sync_samples,
        &mut stsc_entries,
    )?;

    // Build sample list from sample table data
    build_samples(
        data,
        &mut track,
        &sample_sizes,
        &sample_offsets,
        &sync_samples,
        &stsc_entries,
    )?;

    Ok(track)
}

/// Recursively parse trak children (tkhd, mdia/mdhd/hdlr/minf/stbl).
#[allow(clippy::too_many_arguments)]
fn parse_trak_children(
    data: &[u8],
    start: usize,
    end: usize,
    track: &mut Track,
    sample_sizes: &mut Vec<u32>,
    sample_offsets: &mut Vec<u64>,
    sync_samples: &mut Option<Vec<u32>>,
    stsc_entries: &mut Vec<(u32, u32, u32)>,
) -> Result<(), Mp4Error> {
    iterate_boxes(data, start, end, |header, content_start, _box_data| {
        match &header.box_type {
            b"tkhd" => parse_tkhd(data, content_start, track)?,
            b"mdia" => {
                let child_end = content_start - header.header_len as usize + header.size as usize;
                parse_trak_children(
                    data,
                    content_start,
                    child_end,
                    track,
                    sample_sizes,
                    sample_offsets,
                    sync_samples,
                    stsc_entries,
                )?;
            }
            b"mdhd" => parse_mdhd(data, content_start, track)?,
            b"hdlr" => parse_hdlr(data, content_start, track)?,
            b"minf" | b"stbl" => {
                let child_end = content_start - header.header_len as usize + header.size as usize;
                // Save the media handler type before descending — QuickTime MOVs
                // have a data handler hdlr inside minf that would overwrite it.
                let saved_handler = track.handler_type;
                parse_trak_children(
                    data,
                    content_start,
                    child_end,
                    track,
                    sample_sizes,
                    sample_offsets,
                    sync_samples,
                    stsc_entries,
                )?;
                // Restore media handler if minf's data handler hdlr overwrote it
                if saved_handler != [0; 4] {
                    track.handler_type = saved_handler;
                }
            }
            b"stsd" => {
                let box_start = content_start - header.header_len as usize;
                let box_end = box_start + header.size as usize;
                track.stsd_raw = data[box_start..box_end].to_vec();
                parse_stsd(data, content_start, track)?;
            }
            b"stsz" => *sample_sizes = parse_stsz(data, content_start)?,
            b"stco" => *sample_offsets = parse_stco(data, content_start)?,
            b"co64" => *sample_offsets = parse_co64(data, content_start)?,
            b"stss" => *sync_samples = Some(parse_stss(data, content_start)?),
            b"stsc" => *stsc_entries = parse_stsc(data, content_start)?,
            b"stts" => {} // We don't need decode timestamps for steganography
            _ => {}       // Skip unknown boxes
        }
        Ok(())
    })
}

// ─── Box parsers ─────────────────────────────────────────────────────

/// Parse tkhd (Track Header) box → track_id, width, height.
fn parse_tkhd(data: &[u8], start: usize, track: &mut Track) -> Result<(), Mp4Error> {
    if start >= data.len() {
        return Err(Mp4Error::UnexpectedEof);
    }
    let version = data[start];
    let offset = if version == 1 {
        // Version 1: 1(ver) + 3(flags) + 8(creation) + 8(modification) + 4(track_id) + 4(reserved) + 8(duration)
        start + 4 // skip version+flags
    } else {
        // Version 0: 1(ver) + 3(flags) + 4(creation) + 4(modification) + 4(track_id) + 4(reserved) + 4(duration)
        start + 4
    };

    if version == 1 {
        // v1: skip 8+8, then track_id at offset+16
        track.track_id = read_u32(data, offset + 16)?;
        // Width/height are fixed-point 16.16 at the end of tkhd
        // v1: offset + 16(creation+modification) + 4(track_id) + 4(reserved) + 8(duration) + 8(reserved) + ...
        // Total: 4 + 8+8+4+4+8 + 8 + 2+2+2+2 + 36(matrix) = 4+8+8+4+4+8+8+2+2+2+2+36 = 88
        let wh_offset = start + 4 + 8 + 8 + 4 + 4 + 8 + 8 + 2 + 2 + 2 + 2 + 36;
        track.width = read_u32(data, wh_offset)? >> 16;
        track.height = read_u32(data, wh_offset + 4)? >> 16;
    } else {
        // v0: offset, then creation(4) + modification(4) + track_id(4)
        track.track_id = read_u32(data, offset + 8)?;
        // v0: 4(ver+flags) + 4+4+4+4+4 + 8 + 2+2+2+2 + 36 = 4+4+4+4+4+4+8+2+2+2+2+36 = 76
        let wh_offset = start + 4 + 4 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 2 + 2 + 36;
        track.width = read_u32(data, wh_offset)? >> 16;
        track.height = read_u32(data, wh_offset + 4)? >> 16;
    }
    Ok(())
}

/// Parse mdhd (Media Header) box → timescale, duration.
fn parse_mdhd(data: &[u8], start: usize, track: &mut Track) -> Result<(), Mp4Error> {
    if start >= data.len() {
        return Err(Mp4Error::UnexpectedEof);
    }
    let version = data[start];
    if version == 1 {
        // v1: 4(ver+flags) + 8(creation) + 8(modification) + 4(timescale) + 8(duration)
        track.timescale = read_u32(data, start + 4 + 8 + 8)?;
        track.duration = read_u64(data, start + 4 + 8 + 8 + 4)?;
    } else {
        // v0: 4(ver+flags) + 4(creation) + 4(modification) + 4(timescale) + 4(duration)
        track.timescale = read_u32(data, start + 4 + 4 + 4)?;
        track.duration = read_u32(data, start + 4 + 4 + 4 + 4)? as u64;
    }
    Ok(())
}

/// Parse hdlr (Handler Reference) box → handler_type.
fn parse_hdlr(data: &[u8], start: usize, track: &mut Track) -> Result<(), Mp4Error> {
    // hdlr: version(1) + flags(3) + pre_defined(4) + handler_type(4) + ...
    track.handler_type = fourcc(data, start + 4 + 4)?;
    Ok(())
}

/// Parse stsd (Sample Description) box → codec type + hvcC data.
fn parse_stsd(data: &[u8], start: usize, track: &mut Track) -> Result<(), Mp4Error> {
    // stsd: version(1) + flags(3) + entry_count(4)
    let entry_count = read_u32(data, start + 4)?;
    if entry_count == 0 {
        return Ok(());
    }

    // Parse first sample entry
    let entry_start = start + 8;
    let entry_header = parse_box_header(data, entry_start)?;
    track.codec = entry_header.box_type;

    // Visual sample entries share a common fixed part (ISO 14496-12 §12.1.3):
    // reserved(6) + data_ref_idx(2) + pre_defined(2) + reserved(2) +
    // pre_defined(12) + width(2) + height(2) + horizresolution(4) +
    // vertresolution(4) + reserved(4) + frame_count(2) + compressorname(32) +
    // depth(2) + pre_defined(2) = 78 bytes after the box header.
    // Children (hvcC/avcC, etc.) follow the fixed part.
    let is_video = track.codec == *b"hev1"
        || track.codec == *b"hvc1"
        || track.codec == *b"avc1"
        || track.codec == *b"avc3";

    if is_video {
        let children_start = entry_start + entry_header.header_len as usize + 78;
        let entry_end = entry_start + entry_header.size as usize;

        // Read dimensions from the sample entry (more reliable than tkhd for coded size)
        let dim_offset = entry_start + entry_header.header_len as usize + 6 + 2 + 2 + 2 + 12;
        if dim_offset + 4 <= data.len() {
            track.width = read_u16(data, dim_offset)? as u32;
            track.height = read_u16(data, dim_offset + 2)? as u32;
        }

        if children_start < entry_end {
            iterate_boxes(data, children_start, entry_end, |h, cs, _| {
                if h.box_type == *b"hvcC" {
                    track.hvcc_data = Some(parse_hvcc(data, cs)?);
                } else if h.box_type == *b"avcC" {
                    track.avcc_data = Some(parse_avcc(data, cs)?);
                }
                Ok(())
            })?;
        }
    }

    Ok(())
}

/// Parse HEVCDecoderConfigurationRecord (hvcC box content).
fn parse_hvcc(data: &[u8], start: usize) -> Result<HvccData, Mp4Error> {
    // HEVCDecoderConfigurationRecord:
    // configurationVersion(1) + general_profile_space/tier/profile(1) +
    // general_profile_compatibility_flags(4) + general_constraint_indicator_flags(6) +
    // general_level_idc(1) + min_spatial_segmentation_idc(2) +
    // parallelism_type(1) + chroma_format(1) + bit_depth_luma(1) + bit_depth_chroma(1) +
    // avg_frame_rate(2) + const_frame_rate/num_temporal_layers/temporal_id_nesting/length_size(1) +
    // num_of_arrays(1)
    // = 23 bytes before the arrays

    if start + 23 > data.len() {
        return Err(Mp4Error::UnexpectedEof);
    }

    let configuration_version = data[start];
    let length_size_minus1 = data[start + 21] & 0x03;
    let num_arrays = data[start + 22];

    let mut vps_nalus = Vec::new();
    let mut sps_nalus = Vec::new();
    let mut pps_nalus = Vec::new();

    let mut pos = start + 23;
    for _ in 0..num_arrays {
        if pos + 3 > data.len() {
            return Err(Mp4Error::UnexpectedEof);
        }
        let nal_type = data[pos] & 0x3F;
        let num_nalus = read_u16(data, pos + 1)? as usize;
        pos += 3;

        for _ in 0..num_nalus {
            if pos + 2 > data.len() {
                return Err(Mp4Error::UnexpectedEof);
            }
            let nalu_len = read_u16(data, pos)? as usize;
            pos += 2;
            if pos + nalu_len > data.len() {
                return Err(Mp4Error::UnexpectedEof);
            }
            let nalu_data = data[pos..pos + nalu_len].to_vec();
            pos += nalu_len;

            match nal_type {
                32 => vps_nalus.push(nalu_data), // VPS_NUT
                33 => sps_nalus.push(nalu_data), // SPS_NUT
                34 => pps_nalus.push(nalu_data), // PPS_NUT
                _ => {}                          // SEI, etc. — skip
            }
        }
    }

    Ok(HvccData {
        configuration_version,
        length_size_minus1,
        vps_nalus,
        sps_nalus,
        pps_nalus,
    })
}

/// Parse AVCDecoderConfigurationRecord (avcC box content).
///
/// ISO 14496-15, Section 5.3.3.1.2:
/// ```text
/// configurationVersion(1) + AVCProfileIndication(1) +
/// profile_compatibility(1) + AVCLevelIndication(1) +
/// reserved_6bits_lengthSizeMinusOne_2bits(1) +
/// reserved_3bits_numOfSPS_5bits(1) + SPS array + numOfPPS(1) + PPS array
/// ```
fn parse_avcc(data: &[u8], start: usize) -> Result<AvccData, Mp4Error> {
    if start + 6 > data.len() {
        return Err(Mp4Error::UnexpectedEof);
    }

    let configuration_version = data[start];
    let profile = data[start + 1];
    let profile_compat = data[start + 2];
    let level = data[start + 3];
    let length_size_minus1 = data[start + 4] & 0x03;
    let num_sps = (data[start + 5] & 0x1F) as usize;

    let mut sps_nalus = Vec::new();
    let mut pps_nalus = Vec::new();
    let mut pos = start + 6;

    // Read SPS NAL units (2-byte length prefix each)
    for _ in 0..num_sps {
        if pos + 2 > data.len() {
            return Err(Mp4Error::UnexpectedEof);
        }
        let nalu_len = read_u16(data, pos)? as usize;
        pos += 2;
        if pos + nalu_len > data.len() {
            return Err(Mp4Error::UnexpectedEof);
        }
        sps_nalus.push(data[pos..pos + nalu_len].to_vec());
        pos += nalu_len;
    }

    // Read PPS NAL units
    if pos >= data.len() {
        return Err(Mp4Error::UnexpectedEof);
    }
    let num_pps = data[pos] as usize;
    pos += 1;

    for _ in 0..num_pps {
        if pos + 2 > data.len() {
            return Err(Mp4Error::UnexpectedEof);
        }
        let nalu_len = read_u16(data, pos)? as usize;
        pos += 2;
        if pos + nalu_len > data.len() {
            return Err(Mp4Error::UnexpectedEof);
        }
        pps_nalus.push(data[pos..pos + nalu_len].to_vec());
        pos += nalu_len;
    }

    Ok(AvccData {
        configuration_version,
        profile,
        profile_compat,
        level,
        length_size_minus1,
        sps_nalus,
        pps_nalus,
    })
}

/// Parse stsz (Sample Size) box → list of sample sizes.
fn parse_stsz(data: &[u8], start: usize) -> Result<Vec<u32>, Mp4Error> {
    // stsz: version(1) + flags(3) + sample_size(4) + sample_count(4) [+ per-sample sizes]
    let sample_size = read_u32(data, start + 4)?;
    let sample_count = read_u32(data, start + 8)? as usize;

    if sample_size != 0 {
        // Fixed sample size — all samples have the same size
        Ok(vec![sample_size; sample_count])
    } else {
        // Variable sizes
        let mut sizes = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            sizes.push(read_u32(data, start + 12 + i * 4)?);
        }
        Ok(sizes)
    }
}

/// Parse stco (Chunk Offset) box → list of chunk offsets (32-bit).
fn parse_stco(data: &[u8], start: usize) -> Result<Vec<u64>, Mp4Error> {
    // stco: version(1) + flags(3) + entry_count(4) + offsets(4 each)
    let count = read_u32(data, start + 4)? as usize;
    let mut offsets = Vec::with_capacity(count);
    for i in 0..count {
        offsets.push(read_u32(data, start + 8 + i * 4)? as u64);
    }
    Ok(offsets)
}

/// Parse co64 (Chunk Offset 64-bit) box → list of chunk offsets.
fn parse_co64(data: &[u8], start: usize) -> Result<Vec<u64>, Mp4Error> {
    // co64: version(1) + flags(3) + entry_count(4) + offsets(8 each)
    let count = read_u32(data, start + 4)? as usize;
    let mut offsets = Vec::with_capacity(count);
    for i in 0..count {
        offsets.push(read_u64(data, start + 8 + i * 8)?);
    }
    Ok(offsets)
}

/// Parse stss (Sync Sample) box → list of sync sample numbers (1-based).
fn parse_stss(data: &[u8], start: usize) -> Result<Vec<u32>, Mp4Error> {
    // stss: version(1) + flags(3) + entry_count(4) + sample_numbers(4 each)
    let count = read_u32(data, start + 4)? as usize;
    let mut syncs = Vec::with_capacity(count);
    for i in 0..count {
        syncs.push(read_u32(data, start + 8 + i * 4)?);
    }
    Ok(syncs)
}

/// Parse stsc (Sample-to-Chunk) box → list of (first_chunk, samples_per_chunk, desc_index).
pub(super) fn parse_stsc(data: &[u8], start: usize) -> Result<Vec<(u32, u32, u32)>, Mp4Error> {
    // stsc: version(1) + flags(3) + entry_count(4) + entries(12 each)
    let count = read_u32(data, start + 4)? as usize;
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let offset = start + 8 + i * 12;
        let first_chunk = read_u32(data, offset)?;
        let samples_per_chunk = read_u32(data, offset + 4)?;
        let desc_index = read_u32(data, offset + 8)?;
        entries.push((first_chunk, samples_per_chunk, desc_index));
    }
    Ok(entries)
}

/// Build the final Sample list from parsed sample table data.
///
/// Maps chunk offsets + sample-to-chunk + sample sizes → per-sample byte offsets,
/// then reads sample data from the file.
fn build_samples(
    data: &[u8],
    track: &mut Track,
    sample_sizes: &[u32],
    chunk_offsets: &[u64],
    sync_samples: &Option<Vec<u32>>,
    stsc_entries: &[(u32, u32, u32)],
) -> Result<(), Mp4Error> {
    if sample_sizes.is_empty() || chunk_offsets.is_empty() {
        return Ok(());
    }

    // Build sync sample lookup set (1-based sample numbers)
    let sync_set: Option<std::collections::HashSet<u32>> =
        sync_samples.as_ref().map(|s| s.iter().copied().collect());

    // Expand stsc entries to get samples_per_chunk for every chunk
    let num_chunks = chunk_offsets.len();
    let mut samples_per_chunk = vec![0u32; num_chunks];

    if stsc_entries.is_empty() {
        // Fallback: assume 1 sample per chunk
        samples_per_chunk.fill(1);
    } else {
        for (i, entry) in stsc_entries.iter().enumerate() {
            let first_chunk = entry.0 as usize; // 1-based
            let spc = entry.1;
            let next_first = if i + 1 < stsc_entries.len() {
                stsc_entries[i + 1].0 as usize
            } else {
                num_chunks + 1
            };
            for chunk_idx in first_chunk..next_first {
                if chunk_idx <= num_chunks && chunk_idx >= 1 {
                    samples_per_chunk[chunk_idx - 1] = spc;
                }
            }
        }
    }

    // Build per-sample offsets from chunk offsets + per-chunk sample counts + sizes
    let mut samples = Vec::with_capacity(sample_sizes.len());
    let mut sample_idx = 0usize;

    for (chunk_idx, &chunk_offset) in chunk_offsets.iter().enumerate() {
        let spc = samples_per_chunk[chunk_idx] as usize;
        let mut offset_in_chunk = 0u64;

        for _ in 0..spc {
            if sample_idx >= sample_sizes.len() {
                break;
            }
            let size = sample_sizes[sample_idx];
            let abs_offset = chunk_offset + offset_in_chunk;
            let sample_num = (sample_idx + 1) as u32; // 1-based

            let is_sync = match &sync_set {
                Some(set) => set.contains(&sample_num),
                None => true, // No stss → all samples are sync
            };

            // Read sample data from file
            let start = abs_offset as usize;
            let end = start + size as usize;
            let sample_data = if end <= data.len() {
                data[start..end].to_vec()
            } else {
                // Sample data extends past file — truncated file
                return Err(Mp4Error::TruncatedFile);
            };

            samples.push(Sample {
                offset: abs_offset,
                size,
                is_sync,
                data: sample_data,
            });

            offset_in_chunk += size as u64;
            sample_idx += 1;
        }
    }

    track.samples = samples;
    Ok(())
}

// ─── Truncated file recovery ──────────────────────────────────────────

/// Attempt to recover a truncated MP4 file that has mdat but no moov box.
///
/// This happens with interrupted recordings (e.g. DJI Mavic drones) where the
/// camera wrote raw H.265 data to mdat but never finalized the moov atom.
///
/// Strategy:
/// 1. Find the mdat region from the top-level box scan
/// 2. Try length-prefixed NAL scanning (MP4 format); fall back to Annex B start codes
/// 3. Extract VPS/SPS/PPS and group VCL NAL units into access units (samples)
/// 4. Build a synthetic Track with enough metadata for the pipeline
fn recover_truncated(data: &[u8], ftyp: &[u8]) -> Result<Mp4File, Mp4Error> {
    let (mdat_start, mdat_end) = find_mdat_region(data)?;
    let mdat_data = &data[mdat_start..mdat_end];

    // Try MP4-style length-prefixed NAL units first (4-byte prefix)
    let length_size: u8 = 4;
    if validate_nal_lengths(mdat_data, length_size) {
        return recover_from_length_prefixed(ftyp, mdat_data, mdat_start, length_size);
    }

    // Fall back to Annex B start code scanning (some truncated files use raw H.265 streams)
    recover_from_annexb(ftyp, mdat_data, mdat_start)
}

/// Recovery using Annex B start code scanning (raw H.265 streams).
///
/// Some truncated files store NAL units with 3- or 4-byte start codes
/// instead of MP4-style length prefixes. This scans for start codes and
/// reconstructs access units.
fn recover_from_annexb(
    ftyp: &[u8],
    mdat_data: &[u8],
    mdat_start: usize,
) -> Result<Mp4File, Mp4Error> {
    let length_size: u8 = 4;

    // Find all Annex B start code positions (00 00 00 01 or 00 00 01)
    let mut sc_positions: Vec<usize> = Vec::new();
    let mut i = 0;
    while i + 3 < mdat_data.len() {
        if mdat_data[i] == 0 && mdat_data[i + 1] == 0 {
            if mdat_data[i + 2] == 1 {
                sc_positions.push(i + 3);
                i += 3;
                continue;
            } else if i + 3 < mdat_data.len() && mdat_data[i + 2] == 0 && mdat_data[i + 3] == 1 {
                sc_positions.push(i + 4);
                i += 4;
                continue;
            }
        }
        i += 1;
    }

    if sc_positions.len() < 2 {
        return Err(Mp4Error::TruncatedFile);
    }

    let mut vps_nalus: Vec<Vec<u8>> = Vec::new();
    let mut sps_nalus: Vec<Vec<u8>> = Vec::new();
    let mut pps_nalus: Vec<Vec<u8>> = Vec::new();
    let mut samples: Vec<Sample> = Vec::new();

    // Build access units from NALs between start codes.
    // Convert to length-prefixed format for consistent sample data.
    let mut au_buffer: Vec<u8> = Vec::new();
    let mut au_buffer_start: usize = 0;
    let mut au_is_irap: bool = false;
    let mut au_has_vcl: bool = false;

    for (idx, &nal_start) in sc_positions.iter().enumerate() {
        // NAL end: just before the next start code (trim trailing zeros)
        let nal_end = if idx + 1 < sc_positions.len() {
            let next_sc_nal = sc_positions[idx + 1];
            let mut end = next_sc_nal;
            while end > nal_start && mdat_data[end - 1] == 0 {
                end -= 1;
            }
            end
        } else {
            mdat_data.len()
        };

        if nal_end <= nal_start || nal_end - nal_start < 2 {
            continue;
        }

        let nal_data = &mdat_data[nal_start..nal_end];
        let forbidden_zero = (nal_data[0] >> 7) & 1;
        if forbidden_zero != 0 {
            continue;
        }
        let nal_type_val = (nal_data[0] >> 1) & 0x3F;

        // Collect parameter sets
        match nal_type_val {
            32 => {
                if !vps_nalus.iter().any(|v| v.as_slice() == nal_data) {
                    vps_nalus.push(nal_data.to_vec());
                }
            }
            33 => {
                if !sps_nalus.iter().any(|v| v.as_slice() == nal_data) {
                    sps_nalus.push(nal_data.to_vec());
                }
            }
            34 => {
                if !pps_nalus.iter().any(|v| v.as_slice() == nal_data) {
                    pps_nalus.push(nal_data.to_vec());
                }
            }
            _ => {}
        }

        let is_vcl = nal_type_val <= 31;
        let is_irap = (16..=23).contains(&nal_type_val);
        let is_first_slice = if is_vcl && nal_data.len() >= 3 {
            (nal_data[2] >> 7) & 1 == 1
        } else {
            is_vcl
        };

        // Access unit boundary: first VCL slice of a new picture
        if is_vcl && is_first_slice && au_has_vcl {
            if !au_buffer.is_empty() {
                let file_offset = (mdat_start + au_buffer_start) as u64;
                samples.push(Sample {
                    offset: file_offset,
                    size: au_buffer.len() as u32,
                    is_sync: au_is_irap,
                    data: au_buffer.clone(),
                });
            }
            au_buffer.clear();
            au_buffer_start = nal_start;
            au_is_irap = false;
            au_has_vcl = false;
        }

        if au_buffer.is_empty() {
            au_buffer_start = nal_start;
        }

        // Write length-prefixed NAL to access unit buffer
        let nal_len = nal_data.len() as u32;
        au_buffer.extend_from_slice(&nal_len.to_be_bytes());
        au_buffer.extend_from_slice(nal_data);

        if is_vcl {
            au_has_vcl = true;
        }
        if is_irap {
            au_is_irap = true;
        }
    }

    // Flush final access unit
    if !au_buffer.is_empty() && au_has_vcl {
        let file_offset = (mdat_start + au_buffer_start) as u64;
        samples.push(Sample {
            offset: file_offset,
            size: au_buffer.len() as u32,
            is_sync: au_is_irap,
            data: au_buffer,
        });
    }

    build_recovered_mp4(ftyp, vps_nalus, sps_nalus, pps_nalus, samples, length_size)
}

/// Recovery using MP4-style length-prefixed NAL units.
fn recover_from_length_prefixed(
    ftyp: &[u8],
    mdat_data: &[u8],
    mdat_start: usize,
    length_size: u8,
) -> Result<Mp4File, Mp4Error> {
    let _ls = length_size as usize;

    // Walk NAL units, collect parameter sets and build access units
    let mut vps_nalus: Vec<Vec<u8>> = Vec::new();
    let mut sps_nalus: Vec<Vec<u8>> = Vec::new();
    let mut pps_nalus: Vec<Vec<u8>> = Vec::new();

    // Each access unit: (start_offset_in_mdat, total_size, is_irap)
    let mut samples: Vec<Sample> = Vec::new();

    // Current access unit accumulator
    let mut au_start: Option<usize> = None; // byte offset in mdat_data
    let mut au_size: usize = 0;
    let mut au_is_irap: bool = false;

    let ls = length_size as usize;
    let mut pos: usize = 0;

    while pos + ls <= mdat_data.len() {
        // Read NAL length
        let nal_len = read_nal_length(mdat_data, pos, ls);

        // Sanity check: length must not exceed remaining data
        if nal_len == 0 || pos + ls + nal_len > mdat_data.len() {
            break;
        }

        let nal_data_start = pos + ls;
        let nal_data = &mdat_data[nal_data_start..nal_data_start + nal_len];

        // Need at least 2 bytes for HEVC NAL header
        if nal_len < 2 {
            break;
        }

        // Parse NAL header
        let forbidden_zero = (nal_data[0] >> 7) & 1;
        if forbidden_zero != 0 {
            break; // Corrupted data
        }
        let nal_type_val = (nal_data[0] >> 1) & 0x3F;

        // Validate NAL type is in valid HEVC range (0-63)
        if nal_type_val > 63 {
            break;
        }

        let total_nal_size = ls + nal_len; // length prefix + NAL body

        match nal_type_val {
            // VPS (32)
            32 => {
                if !vps_nalus.iter().any(|v| v.as_slice() == nal_data) {
                    vps_nalus.push(nal_data.to_vec());
                }
                // VPS/SPS/PPS before an IRAP starts a new access unit
                if au_start.is_some() {
                    // Flush current AU
                    flush_access_unit(
                        &mut samples, mdat_start, au_start.unwrap(), au_size, au_is_irap,
                        mdat_data,
                    );
                    au_start = None;
                    au_size = 0;
                    au_is_irap = false;
                }
                if au_start.is_none() {
                    au_start = Some(pos);
                }
                au_size += total_nal_size;
            }
            // SPS (33)
            33 => {
                if !sps_nalus.iter().any(|v| v.as_slice() == nal_data) {
                    sps_nalus.push(nal_data.to_vec());
                }
                if au_start.is_none() {
                    au_start = Some(pos);
                }
                au_size += total_nal_size;
            }
            // PPS (34)
            34 => {
                if !pps_nalus.iter().any(|v| v.as_slice() == nal_data) {
                    pps_nalus.push(nal_data.to_vec());
                }
                if au_start.is_none() {
                    au_start = Some(pos);
                }
                au_size += total_nal_size;
            }
            // VCL NAL units (0-31)
            0..=31 => {
                // Check first_slice_segment_in_pic_flag (bit 0 of byte 2 of the NAL body,
                // which is the first byte after the 2-byte NAL header).
                let is_first_slice = if nal_len >= 3 {
                    (nal_data[2] >> 7) & 1 == 1
                } else {
                    true // Single-slice assumption
                };

                let is_irap = (16..=23).contains(&nal_type_val);

                if is_first_slice {
                    // This starts a new picture. If we had an AU accumulating, flush it.
                    if au_start.is_some() && au_size > 0 {
                        // Only flush if the previous AU contained VCL data
                        // (parameter sets alone don't count as a sample)
                        let prev_has_vcl = samples_has_vcl_data(
                            mdat_data, au_start.unwrap(), au_size, ls,
                        );
                        if prev_has_vcl {
                            flush_access_unit(
                                &mut samples, mdat_start, au_start.unwrap(), au_size,
                                au_is_irap, mdat_data,
                            );
                            au_start = Some(pos);
                            au_size = 0;
                            au_is_irap = false;
                        }
                        // If prev had no VCL, the parameter sets belong to this new picture
                    }
                    if au_start.is_none() {
                        au_start = Some(pos);
                    }
                    if is_irap {
                        au_is_irap = true;
                    }
                } else if au_start.is_none() {
                    // Continuation slice without a picture start — skip
                    pos += total_nal_size;
                    continue;
                }

                au_size += total_nal_size;
            }
            // Non-VCL, non-parameter-set NAL units (AUD, SEI, filler, etc.)
            _ => {
                // AUD (35) signals an access unit boundary
                if nal_type_val == 35
                    && au_start.is_some() && au_size > 0 {
                        let prev_has_vcl = samples_has_vcl_data(
                            mdat_data, au_start.unwrap(), au_size, ls,
                        );
                        if prev_has_vcl {
                            flush_access_unit(
                                &mut samples, mdat_start, au_start.unwrap(), au_size,
                                au_is_irap, mdat_data,
                            );
                            au_start = None;
                            au_size = 0;
                            au_is_irap = false;
                        }
                    }
                if au_start.is_none() {
                    au_start = Some(pos);
                }
                au_size += total_nal_size;
            }
        }

        pos += total_nal_size;
    }

    // Flush final access unit
    if let Some(start) = au_start
        && au_size > 0 && samples_has_vcl_data(mdat_data, start, au_size, ls) {
            flush_access_unit(&mut samples, mdat_start, start, au_size, au_is_irap, mdat_data);
        }

    build_recovered_mp4(ftyp, vps_nalus, sps_nalus, pps_nalus, samples, length_size)
}

/// Build an Mp4File from recovered parameter sets and samples.
///
/// Shared by both `recover_from_annexb` and `recover_from_length_prefixed`.
fn build_recovered_mp4(
    ftyp: &[u8],
    vps_nalus: Vec<Vec<u8>>,
    sps_nalus: Vec<Vec<u8>>,
    pps_nalus: Vec<Vec<u8>>,
    samples: Vec<Sample>,
    length_size: u8,
) -> Result<Mp4File, Mp4Error> {
    if vps_nalus.is_empty() || sps_nalus.is_empty() || pps_nalus.is_empty() {
        return Err(Mp4Error::TruncatedFile);
    }
    if samples.is_empty() {
        return Err(Mp4Error::TruncatedFile);
    }

    let (width, height) = extract_dimensions_from_sps(&sps_nalus[0]).unwrap_or((0, 0));

    let hvcc = HvccData {
        configuration_version: 1,
        length_size_minus1: length_size - 1,
        vps_nalus,
        sps_nalus,
        pps_nalus,
    };

    let track = Track {
        track_id: 1,
        handler_type: *b"vide",
        codec: *b"hvc1",
        width,
        height,
        timescale: 30000,
        duration: (samples.len() as u64) * 1000,
        samples,
        hvcc_data: Some(hvcc),
            avcc_data: None,
        stsd_raw: Vec::new(),
        trak_raw: Vec::new(),
    };

    Ok(Mp4File {
        ftyp: ftyp.to_vec(),
        tracks: vec![track],
        video_track_idx: Some(0),
    })
}

/// Find the mdat box region in the file, returning (content_start, content_end).
fn find_mdat_region(data: &[u8]) -> Result<(usize, usize), Mp4Error> {
    let file_end = data.len();
    let mut pos = 0usize;
    while pos < file_end {
        if pos + 8 > file_end {
            break;
        }
        let header = parse_box_header(data, pos)?;
        if header.size < 8 {
            break;
        }
        if header.box_type == *b"mdat" {
            let content_start = pos + header.header_len as usize;
            // For truncated files, mdat may declare a size larger than the file.
            // Use the smaller of declared end and actual file end.
            let content_end = (pos + header.size as usize).min(file_end);
            return Ok((content_start, content_end));
        }
        let box_end = pos + header.size as usize;
        if box_end > file_end {
            break;
        }
        pos = box_end;
    }
    Err(Mp4Error::TruncatedFile)
}

/// Read a big-endian NAL length from the given position.
fn read_nal_length(data: &[u8], pos: usize, length_size: usize) -> usize {
    let mut len = 0usize;
    for i in 0..length_size {
        len = (len << 8) | data[pos + i] as usize;
    }
    len
}

/// Validate that the first few NAL length-prefixed units in mdat look reasonable.
///
/// Checks that consecutive length+NAL pairs chain together without gaps and that
/// NAL type bytes look like valid HEVC.
fn validate_nal_lengths(mdat_data: &[u8], length_size: u8) -> bool {
    let ls = length_size as usize;
    let mut pos = 0usize;
    let mut valid_count = 0u32;

    for _ in 0..8 {
        // Check first 8 NAL units
        if pos + ls > mdat_data.len() {
            break;
        }
        let len = read_nal_length(mdat_data, pos, ls);
        if len == 0 || len > 50_000_000 || pos + ls + len > mdat_data.len() {
            break;
        }
        let nal_start = pos + ls;
        if nal_start >= mdat_data.len() {
            break;
        }
        // Check forbidden_zero_bit and NAL type
        let byte0 = mdat_data[nal_start];
        if (byte0 >> 7) & 1 != 0 {
            break;
        }
        let nal_type = (byte0 >> 1) & 0x3F;
        if nal_type > 63 {
            break;
        }
        valid_count += 1;
        pos += ls + len;
    }

    // Need at least 2 valid NAL units to consider this a valid MP4-style mdat
    valid_count >= 2
}

/// Check if a region of mdat data contains any VCL NAL units.
fn samples_has_vcl_data(mdat_data: &[u8], start: usize, size: usize, length_size: usize) -> bool {
    let end = start + size;
    let mut pos = start;
    while pos + length_size <= end {
        let len = read_nal_length(mdat_data, pos, length_size);
        if len == 0 || pos + length_size + len > end {
            break;
        }
        let nal_byte0 = mdat_data[pos + length_size];
        let nal_type = (nal_byte0 >> 1) & 0x3F;
        if nal_type <= 31 {
            return true;
        }
        pos += length_size + len;
    }
    false
}

/// Flush the current access unit as a Sample.
fn flush_access_unit(
    samples: &mut Vec<Sample>,
    mdat_file_start: usize,
    au_offset_in_mdat: usize,
    au_size: usize,
    is_irap: bool,
    mdat_data: &[u8],
) {
    let file_offset = (mdat_file_start + au_offset_in_mdat) as u64;
    let sample_data = mdat_data[au_offset_in_mdat..au_offset_in_mdat + au_size].to_vec();
    samples.push(Sample {
        offset: file_offset,
        size: au_size as u32,
        is_sync: is_irap,
        data: sample_data,
    });
}

/// Extract video dimensions from raw SPS NAL data.
///
/// Performs minimal SPS parsing: skip profile/level info, read pic_width/pic_height.
/// Returns (0, 0) if parsing fails rather than propagating the error.
fn extract_dimensions_from_sps(sps_nal_data: &[u8]) -> Option<(u32, u32)> {
    // Phase 4a: switched from HEVC bitstream to H.264 bitstream for the RBSP
    // primitives (they're spec-identical). This lets MP4 dimension extraction
    // keep working when the HEVC parser is archived behind `hevc-archive`.
    use crate::codec::h264::bitstream::{remove_emulation_prevention, RbspReader};

    if sps_nal_data.len() < 4 {
        return None;
    }

    // SPS NAL body: 2 bytes NAL header, then RBSP
    let rbsp = remove_emulation_prevention(&sps_nal_data[2..]);
    let mut r = RbspReader::new(&rbsp);

    // sps_video_parameter_set_id: u(4)
    r.read_bits(4).ok()?;
    // sps_max_sub_layers_minus1: u(3)
    let max_sub_layers_minus1 = r.read_bits(3).ok()? as u8;
    // sps_temporal_id_nesting_flag: u(1)
    r.read_bits(1).ok()?;

    // profile_tier_level(1, max_sub_layers_minus1)
    skip_profile_tier_level(&mut r, max_sub_layers_minus1).ok()?;

    // sps_seq_parameter_set_id: ue(v)
    r.read_ue().ok()?;
    // chroma_format_idc: ue(v)
    let chroma = r.read_ue().ok()?;
    if chroma == 3 {
        // separate_colour_plane_flag: u(1)
        r.read_bits(1).ok()?;
    }
    // pic_width_in_luma_samples: ue(v)
    let width = r.read_ue().ok()?;
    // pic_height_in_luma_samples: ue(v)
    let height = r.read_ue().ok()?;

    Some((width, height))
}

/// Skip the profile_tier_level syntax in the SPS bitstream.
fn skip_profile_tier_level(
    r: &mut crate::codec::h264::bitstream::RbspReader<'_>,
    max_sub_layers_minus1: u8,
) -> Result<(), Mp4Error> {
    // general_profile_space(2) + general_tier_flag(1) + general_profile_idc(5)
    r.read_bits(8).map_err(|_| Mp4Error::UnexpectedEof)?;
    // general_profile_compatibility_flags: u(32)
    r.read_bits(32).map_err(|_| Mp4Error::UnexpectedEof)?;
    // general_constraint_indicator_flags: u(48) = 6 bytes
    r.read_bits(32).map_err(|_| Mp4Error::UnexpectedEof)?;
    r.read_bits(16).map_err(|_| Mp4Error::UnexpectedEof)?;
    // general_level_idc: u(8)
    r.read_bits(8).map_err(|_| Mp4Error::UnexpectedEof)?;

    // sub_layer_profile_present_flag[i], sub_layer_level_present_flag[i]
    let mut sub_layer_profile_present = [false; 8];
    let mut sub_layer_level_present = [false; 8];
    for i in 0..max_sub_layers_minus1 as usize {
        sub_layer_profile_present[i] = r.read_bit().map_err(|_| Mp4Error::UnexpectedEof)?;
        sub_layer_level_present[i] = r.read_bit().map_err(|_| Mp4Error::UnexpectedEof)?;
    }
    // reserved zero bits
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            r.read_bits(2).map_err(|_| Mp4Error::UnexpectedEof)?;
        }
    }
    // sub-layer profiles and levels
    for i in 0..max_sub_layers_minus1 as usize {
        if sub_layer_profile_present[i] {
            // profile_space(2) + tier(1) + profile_idc(5) + compat(32) + constraint(48) + level(8) = 88 bits
            r.skip_bits(88).map_err(|_| Mp4Error::UnexpectedEof)?;
        }
        if sub_layer_level_present[i] {
            r.read_bits(8).map_err(|_| Mp4Error::UnexpectedEof)?;
        }
    }

    Ok(())
}

// ─── Streaming API ────────────────────────────────────────────────────

use std::io::{Read, Seek, SeekFrom};

/// Parse MP4 metadata from a `Read + Seek` source without reading sample data.
///
/// Returns an `Mp4File` where each `Sample` has `offset` and `size` set but
/// `data` is empty. Use [`read_sample_data`] to read individual samples.
pub fn demux_streaming<R: Read + Seek>(reader: &mut R) -> Result<Mp4File, Mp4Error> {
    // Read the entire file into memory for box parsing, then drop it.
    // The moov box is typically < 1 MB even for large files.
    // We only keep metadata; sample data is read on-demand.
    let start_pos = reader.seek(SeekFrom::Start(0)).map_err(|_| Mp4Error::UnexpectedEof)?;
    let file_size = reader.seek(SeekFrom::End(0)).map_err(|_| Mp4Error::UnexpectedEof)?;
    reader.seek(SeekFrom::Start(start_pos)).map_err(|_| Mp4Error::UnexpectedEof)?;

    // Find ftyp and moov boxes by scanning top-level box headers
    let mut ftyp: Option<Vec<u8>> = None;
    let mut moov_data: Option<Vec<u8>> = None;
    let mut pos = 0u64;

    while pos < file_size {
        reader.seek(SeekFrom::Start(pos)).map_err(|_| Mp4Error::UnexpectedEof)?;
        let mut header_buf = [0u8; 16];
        let n = reader.read(&mut header_buf[..8]).map_err(|_| Mp4Error::UnexpectedEof)?;
        if n < 8 {
            break;
        }
        let size32 = u32::from_be_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]]);
        let box_type = [header_buf[4], header_buf[5], header_buf[6], header_buf[7]];
        let (box_size, header_len): (u64, u64) = if size32 == 1 {
            let n2 = reader.read(&mut header_buf[8..16]).map_err(|_| Mp4Error::UnexpectedEof)?;
            if n2 < 8 {
                return Err(Mp4Error::UnexpectedEof);
            }
            let s64 = u64::from_be_bytes([
                header_buf[8], header_buf[9], header_buf[10], header_buf[11],
                header_buf[12], header_buf[13], header_buf[14], header_buf[15],
            ]);
            (s64, 16)
        } else if size32 == 0 {
            (file_size - pos, 8)
        } else {
            (size32 as u64, 8)
        };

        if box_size < 8 {
            break;
        }

        let _content_size = (box_size - header_len) as usize;
        match &box_type {
            b"ftyp" => {
                // Read the whole ftyp box (header + content)
                let mut buf = vec![0u8; box_size as usize];
                reader.seek(SeekFrom::Start(pos)).map_err(|_| Mp4Error::UnexpectedEof)?;
                reader.read_exact(&mut buf).map_err(|_| Mp4Error::UnexpectedEof)?;
                ftyp = Some(buf);
            }
            b"moov" => {
                // Read the whole moov box (header + content) for parsing
                let mut buf = vec![0u8; box_size as usize];
                reader.seek(SeekFrom::Start(pos)).map_err(|_| Mp4Error::UnexpectedEof)?;
                reader.read_exact(&mut buf).map_err(|_| Mp4Error::UnexpectedEof)?;
                moov_data = Some(buf);
            }
            _ => {} // skip mdat, free, etc.
        }

        pos += box_size;
    }

    let ftyp = ftyp.ok_or_else(|| Mp4Error::InvalidBox("missing ftyp box".into()))?;
    let moov_buf = moov_data.ok_or_else(|| Mp4Error::InvalidBox("missing moov box".into()))?;

    // Parse moov to extract tracks (metadata only — no sample data)
    let moov_header = parse_box_header(&moov_buf, 0)?;
    let moov_content_start = moov_header.header_len as usize;
    let moov_end = moov_header.size as usize;
    let tracks = parse_moov_metadata(&moov_buf, moov_content_start, moov_end)?;

    let video_track_idx = tracks.iter().position(|t| {
        t.handler_type == *b"vide" && (t.is_hevc() || t.is_h264())
    });

    Ok(Mp4File {
        ftyp,
        tracks,
        video_track_idx,
    })
}

/// Read sample data for a single sample from a `Read + Seek` source.
pub fn read_sample_data<R: Read + Seek>(
    reader: &mut R,
    sample: &Sample,
) -> Result<Vec<u8>, Mp4Error> {
    reader.seek(SeekFrom::Start(sample.offset)).map_err(|_| Mp4Error::UnexpectedEof)?;
    let mut buf = vec![0u8; sample.size as usize];
    reader.read_exact(&mut buf).map_err(|_| Mp4Error::UnexpectedEof)?;
    Ok(buf)
}

/// Parse moov box contents for metadata-only tracks (no sample data).
fn parse_moov_metadata(data: &[u8], start: usize, end: usize) -> Result<Vec<Track>, Mp4Error> {
    let mut tracks = Vec::new();

    iterate_boxes(data, start, end, |header, content_start, box_data| {
        if header.box_type == *b"trak" {
            let trak_end = content_start - header.header_len as usize + header.size as usize;
            let track = parse_trak_metadata(data, content_start, trak_end, box_data)?;
            tracks.push(track);
        }
        Ok(())
    })?;

    Ok(tracks)
}

/// Parse a single trak box for metadata only (samples have offset+size but no data).
fn parse_trak_metadata(
    data: &[u8],
    start: usize,
    end: usize,
    trak_raw_data: &[u8],
) -> Result<Track, Mp4Error> {
    let mut track = Track {
        track_id: 0,
        handler_type: [0; 4],
        codec: [0; 4],
        width: 0,
        height: 0,
        timescale: 0,
        duration: 0,
        samples: Vec::new(),
        hvcc_data: None,
            avcc_data: None,
        stsd_raw: Vec::new(),
        trak_raw: trak_raw_data.to_vec(),
    };

    let mut sample_sizes: Vec<u32> = Vec::new();
    let mut sample_offsets: Vec<u64> = Vec::new();
    let mut sync_samples: Option<Vec<u32>> = None;
    let mut stsc_entries: Vec<(u32, u32, u32)> = Vec::new();

    parse_trak_children(
        data, start, end, &mut track,
        &mut sample_sizes, &mut sample_offsets, &mut sync_samples, &mut stsc_entries,
    )?;

    // Build sample list with metadata only (no data read)
    build_samples_metadata(
        &mut track, &sample_sizes, &sample_offsets, &sync_samples, &stsc_entries,
    )?;

    Ok(track)
}

/// Build Sample list with offset/size/sync but empty data vectors.
fn build_samples_metadata(
    track: &mut Track,
    sample_sizes: &[u32],
    chunk_offsets: &[u64],
    sync_samples: &Option<Vec<u32>>,
    stsc_entries: &[(u32, u32, u32)],
) -> Result<(), Mp4Error> {
    if sample_sizes.is_empty() || chunk_offsets.is_empty() {
        return Ok(());
    }

    let sync_set: Option<std::collections::HashSet<u32>> =
        sync_samples.as_ref().map(|s| s.iter().copied().collect());

    let num_chunks = chunk_offsets.len();
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
                if chunk_idx <= num_chunks && chunk_idx >= 1 {
                    samples_per_chunk[chunk_idx - 1] = spc;
                }
            }
        }
    }

    let mut samples = Vec::with_capacity(sample_sizes.len());
    let mut sample_idx = 0usize;

    for (chunk_idx, &chunk_offset) in chunk_offsets.iter().enumerate() {
        let spc = samples_per_chunk[chunk_idx] as usize;
        let mut offset_in_chunk = 0u64;

        for _ in 0..spc {
            if sample_idx >= sample_sizes.len() {
                break;
            }
            let size = sample_sizes[sample_idx];
            let abs_offset = chunk_offset + offset_in_chunk;
            let sample_num = (sample_idx + 1) as u32;

            let is_sync = match &sync_set {
                Some(set) => set.contains(&sample_num),
                None => true,
            };

            samples.push(Sample {
                offset: abs_offset,
                size,
                is_sync,
                data: Vec::new(), // No data — read on demand
            });

            offset_in_chunk += size as u64;
            sample_idx += 1;
        }
    }

    track.samples = samples;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a box with 32-bit size.
    fn make_box(box_type: &[u8; 4], content: &[u8]) -> Vec<u8> {
        let size = (8 + content.len()) as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&size.to_be_bytes());
        buf.extend_from_slice(box_type);
        buf.extend_from_slice(content);
        buf
    }

    /// Helper: build a fullbox (version + flags + content).
    fn make_fullbox(box_type: &[u8; 4], version: u8, flags: u32, content: &[u8]) -> Vec<u8> {
        let mut inner = Vec::new();
        inner.push(version);
        inner.extend_from_slice(&(flags & 0x00FF_FFFF).to_be_bytes()[1..]);
        inner.extend_from_slice(content);
        make_box(box_type, &inner)
    }

    #[test]
    fn test_parse_stsz_variable() {
        // stsz: version=0, flags=0, sample_size=0 (variable), count=3, sizes=[100, 200, 150]
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        content.extend_from_slice(&0u32.to_be_bytes()); // sample_size = 0
        content.extend_from_slice(&3u32.to_be_bytes()); // count = 3
        content.extend_from_slice(&100u32.to_be_bytes());
        content.extend_from_slice(&200u32.to_be_bytes());
        content.extend_from_slice(&150u32.to_be_bytes());

        let sizes = parse_stsz(&content, 0).unwrap();
        assert_eq!(sizes, vec![100, 200, 150]);
    }

    #[test]
    fn test_parse_stsz_fixed() {
        // stsz with fixed sample_size
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        content.extend_from_slice(&512u32.to_be_bytes()); // sample_size = 512
        content.extend_from_slice(&4u32.to_be_bytes()); // count = 4

        let sizes = parse_stsz(&content, 0).unwrap();
        assert_eq!(sizes, vec![512, 512, 512, 512]);
    }

    #[test]
    fn test_parse_stco() {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        content.extend_from_slice(&2u32.to_be_bytes()); // count = 2
        content.extend_from_slice(&1000u32.to_be_bytes());
        content.extend_from_slice(&5000u32.to_be_bytes());

        let offsets = parse_stco(&content, 0).unwrap();
        assert_eq!(offsets, vec![1000, 5000]);
    }

    #[test]
    fn test_parse_co64() {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        content.extend_from_slice(&1u32.to_be_bytes()); // count = 1
        content.extend_from_slice(&0x0000_0001_0000_0000u64.to_be_bytes()); // > 4GB offset

        let offsets = parse_co64(&content, 0).unwrap();
        assert_eq!(offsets, vec![0x0000_0001_0000_0000]);
    }

    #[test]
    fn test_parse_stss() {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        content.extend_from_slice(&3u32.to_be_bytes()); // count = 3
        content.extend_from_slice(&1u32.to_be_bytes()); // sample 1
        content.extend_from_slice(&31u32.to_be_bytes()); // sample 31
        content.extend_from_slice(&61u32.to_be_bytes()); // sample 61

        let syncs = parse_stss(&content, 0).unwrap();
        assert_eq!(syncs, vec![1, 31, 61]);
    }

    #[test]
    fn test_parse_stsc() {
        let mut content = Vec::new();
        content.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        content.extend_from_slice(&2u32.to_be_bytes()); // count = 2
        // Entry 1: first_chunk=1, samples_per_chunk=10, desc_index=1
        content.extend_from_slice(&1u32.to_be_bytes());
        content.extend_from_slice(&10u32.to_be_bytes());
        content.extend_from_slice(&1u32.to_be_bytes());
        // Entry 2: first_chunk=5, samples_per_chunk=5, desc_index=1
        content.extend_from_slice(&5u32.to_be_bytes());
        content.extend_from_slice(&5u32.to_be_bytes());
        content.extend_from_slice(&1u32.to_be_bytes());

        let entries = parse_stsc(&content, 0).unwrap();
        assert_eq!(entries, vec![(1, 10, 1), (5, 5, 1)]);
    }

    #[test]
    fn test_parse_hvcc() {
        // Minimal HEVCDecoderConfigurationRecord
        let mut content = Vec::new();
        content.push(1); // configurationVersion
        content.push(0); // general_profile_space/tier/profile_idc
        content.extend_from_slice(&[0; 4]); // general_profile_compatibility_flags
        content.extend_from_slice(&[0; 6]); // general_constraint_indicator_flags
        content.push(0); // general_level_idc
        content.extend_from_slice(&[0xF0, 0x00]); // min_spatial_segmentation_idc (4 reserved + 12 bits)
        content.push(0xFC); // parallelismType (6 reserved + 2 bits)
        content.push(0xFC); // chromaFormat (6 reserved + 2 bits)
        content.push(0xF8); // bitDepthLuma (5 reserved + 3 bits)
        content.push(0xF8); // bitDepthChroma (5 reserved + 3 bits)
        content.extend_from_slice(&[0x00, 0x00]); // avgFrameRate
        content.push(0x0F); // const/temporal/nesting/lengthSizeMinusOne = 3

        // 3 arrays: VPS, SPS, PPS
        content.push(3); // numOfArrays

        // VPS array
        content.push(0x20 | 32); // array_completeness=1, nal_type=32 (VPS)
        content.extend_from_slice(&1u16.to_be_bytes()); // numNalus=1
        let vps_data = [0x40, 0x01, 0x0C, 0x01]; // fake VPS NAL body
        content.extend_from_slice(&(vps_data.len() as u16).to_be_bytes());
        content.extend_from_slice(&vps_data);

        // SPS array
        content.push(0x20 | 33); // nal_type=33 (SPS)
        content.extend_from_slice(&1u16.to_be_bytes());
        let sps_data = [0x42, 0x01, 0x01, 0x02, 0x20];
        content.extend_from_slice(&(sps_data.len() as u16).to_be_bytes());
        content.extend_from_slice(&sps_data);

        // PPS array
        content.push(0x20 | 34); // nal_type=34 (PPS)
        content.extend_from_slice(&1u16.to_be_bytes());
        let pps_data = [0x44, 0x01, 0xC1];
        content.extend_from_slice(&(pps_data.len() as u16).to_be_bytes());
        content.extend_from_slice(&pps_data);

        let hvcc = parse_hvcc(&content, 0).unwrap();
        assert_eq!(hvcc.configuration_version, 1);
        assert_eq!(hvcc.length_size_minus1, 3);
        assert_eq!(hvcc.vps_nalus.len(), 1);
        assert_eq!(hvcc.sps_nalus.len(), 1);
        assert_eq!(hvcc.pps_nalus.len(), 1);
        assert_eq!(hvcc.vps_nalus[0], vps_data);
        assert_eq!(hvcc.sps_nalus[0], sps_data);
        assert_eq!(hvcc.pps_nalus[0], pps_data);
    }

    #[test]
    fn test_build_samples_simple() {
        // 1 chunk, 3 samples, all sync (no stss)
        let sample_data = [0xAA; 10]; // 10 bytes at offset 100
        let mut file_data = vec![0u8; 200];
        file_data[100..110].copy_from_slice(&sample_data);
        file_data[110..115].fill(0xBB); // sample 2: 5 bytes
        file_data[115..120].fill(0xCC); // sample 3: 5 bytes

        let mut track = Track {
            track_id: 1,
            handler_type: *b"vide",
            codec: *b"hvc1",
            width: 1920,
            height: 1080,
            timescale: 30000,
            duration: 90000,
            samples: Vec::new(),
            hvcc_data: None,
            avcc_data: None,
            stsd_raw: Vec::new(),
            trak_raw: Vec::new(),
        };

        let sizes = vec![10, 5, 5];
        let offsets = vec![100u64];
        let stsc = vec![(1u32, 3u32, 1u32)]; // 1 chunk, 3 samples per chunk

        build_samples(&file_data, &mut track, &sizes, &offsets, &None, &stsc).unwrap();

        assert_eq!(track.samples.len(), 3);
        assert_eq!(track.samples[0].offset, 100);
        assert_eq!(track.samples[0].size, 10);
        assert!(track.samples[0].is_sync); // No stss → all sync
        assert_eq!(track.samples[0].data, vec![0xAA; 10]);
        assert_eq!(track.samples[1].offset, 110);
        assert_eq!(track.samples[1].size, 5);
        assert_eq!(track.samples[2].offset, 115);
    }

    #[test]
    fn test_build_samples_with_stss() {
        // 3 samples, only sample 1 is sync
        let mut file_data = vec![0u8; 200];
        file_data[50..54].fill(0x11);
        file_data[54..58].fill(0x22);
        file_data[58..62].fill(0x33);

        let mut track = Track {
            track_id: 1,
            handler_type: *b"vide",
            codec: *b"hvc1",
            width: 0,
            height: 0,
            timescale: 0,
            duration: 0,
            samples: Vec::new(),
            hvcc_data: None,
            avcc_data: None,
            stsd_raw: Vec::new(),
            trak_raw: Vec::new(),
        };

        let sizes = vec![4, 4, 4];
        let offsets = vec![50u64];
        let sync = Some(vec![1u32]); // Only sample 1 is sync
        let stsc = vec![(1, 3, 1)];

        build_samples(&file_data, &mut track, &sizes, &offsets, &sync, &stsc).unwrap();

        assert!(track.samples[0].is_sync);
        assert!(!track.samples[1].is_sync);
        assert!(!track.samples[2].is_sync);
    }

    /// Build a minimal but valid MP4 file for integration testing.
    fn build_test_mp4() -> Vec<u8> {
        let mut mp4 = Vec::new();

        // ftyp box
        let ftyp = make_box(b"ftyp", b"isom\x00\x00\x02\x00isom");
        mp4.extend_from_slice(&ftyp);

        // We need to know the mdat offset to set stco correctly.
        // Build moov first to know its size, then fix offsets.

        // Sample data: 3 samples of 8 bytes each, starting at mdat content offset
        let sample1 = [0x00, 0x00, 0x00, 0x04, 0x28, 0x01, 0xAF, 0x09]; // length-prefixed NAL
        let sample2 = [0x00, 0x00, 0x00, 0x04, 0x02, 0x01, 0xD0, 0x10];
        let sample3 = [0x00, 0x00, 0x00, 0x04, 0x02, 0x01, 0xD0, 0x11];

        // ─── Build moov ───

        // tkhd v0: version(1) + flags(3) + creation(4) + modification(4) + track_id(4) +
        // reserved(4) + duration(4) + reserved(8) + layer(2) + alt_group(2) + volume(2) +
        // reserved(2) + matrix(36) + width(4) + height(4) = 84 bytes content
        let mut tkhd_content = Vec::new();
        tkhd_content.extend_from_slice(&[0, 0, 0, 0]); // creation_time
        tkhd_content.extend_from_slice(&[0, 0, 0, 0]); // modification_time
        tkhd_content.extend_from_slice(&1u32.to_be_bytes()); // track_id
        tkhd_content.extend_from_slice(&[0; 4]); // reserved
        tkhd_content.extend_from_slice(&90u32.to_be_bytes()); // duration
        tkhd_content.extend_from_slice(&[0; 8]); // reserved
        tkhd_content.extend_from_slice(&[0; 2]); // layer
        tkhd_content.extend_from_slice(&[0; 2]); // alternate_group
        tkhd_content.extend_from_slice(&[0; 2]); // volume
        tkhd_content.extend_from_slice(&[0; 2]); // reserved
        tkhd_content.extend_from_slice(&[     // 3x3 identity matrix (36 bytes)
            0x00,0x01,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00, 0x00,0x01,0x00,0x00, 0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x40,0x00,0x00,0x00,
        ]);
        tkhd_content.extend_from_slice(&((1920u32) << 16).to_be_bytes()); // width 16.16
        tkhd_content.extend_from_slice(&((1080u32) << 16).to_be_bytes()); // height 16.16
        let tkhd = make_fullbox(b"tkhd", 0, 3, &tkhd_content);

        // mdhd v0
        let mut mdhd_content = Vec::new();
        mdhd_content.extend_from_slice(&[0; 4]); // creation_time
        mdhd_content.extend_from_slice(&[0; 4]); // modification_time
        mdhd_content.extend_from_slice(&30000u32.to_be_bytes()); // timescale
        mdhd_content.extend_from_slice(&90000u32.to_be_bytes()); // duration
        mdhd_content.extend_from_slice(&[0x55, 0xC4]); // language (und)
        mdhd_content.extend_from_slice(&[0; 2]); // pre_defined
        let mdhd = make_fullbox(b"mdhd", 0, 0, &mdhd_content);

        // hdlr
        let mut hdlr_content = Vec::new();
        hdlr_content.extend_from_slice(&[0; 4]); // pre_defined
        hdlr_content.extend_from_slice(b"vide"); // handler_type
        hdlr_content.extend_from_slice(&[0; 12]); // reserved
        hdlr_content.push(0); // name (null-terminated)
        let hdlr = make_fullbox(b"hdlr", 0, 0, &hdlr_content);

        // stsd with hvc1 sample entry + hvcC
        let mut hvcc_content = Vec::new();
        hvcc_content.push(1); // configurationVersion
        hvcc_content.push(0); // profile stuff
        hvcc_content.extend_from_slice(&[0; 4]); // compat flags
        hvcc_content.extend_from_slice(&[0; 6]); // constraint flags
        hvcc_content.push(0); // level_idc
        hvcc_content.extend_from_slice(&[0xF0, 0x00]); // min_spatial_seg
        hvcc_content.push(0xFC); // parallelism
        hvcc_content.push(0xFC); // chroma
        hvcc_content.push(0xF8); // luma depth
        hvcc_content.push(0xF8); // chroma depth
        hvcc_content.extend_from_slice(&[0, 0]); // avg_frame_rate
        hvcc_content.push(0x0F); // length_size=4 (3+1)
        hvcc_content.push(1); // numOfArrays (VPS only for simplicity)
        hvcc_content.push(0x20); // array_completeness=0, nal_unit_type=32 (VPS)
        hvcc_content.extend_from_slice(&1u16.to_be_bytes()); // 1 NAL
        let vps = [0x40, 0x01, 0x0C];
        hvcc_content.extend_from_slice(&(vps.len() as u16).to_be_bytes());
        hvcc_content.extend_from_slice(&vps);
        let hvcc_box = make_box(b"hvcC", &hvcc_content);

        // Visual sample entry (hvc1): 6(reserved) + 2(data_ref) + 16(pre_defined+reserved) +
        // 2(width) + 2(height) + 4(hres) + 4(vres) + 4(reserved) + 2(frame_count) +
        // 32(compressorname) + 2(depth) + 2(pre_defined) + children = 78 + header
        let mut vse_content = Vec::new();
        vse_content.extend_from_slice(&[0; 6]); // reserved
        vse_content.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
        vse_content.extend_from_slice(&[0; 2]); // pre_defined
        vse_content.extend_from_slice(&[0; 2]); // reserved
        vse_content.extend_from_slice(&[0; 12]); // pre_defined
        vse_content.extend_from_slice(&1920u16.to_be_bytes()); // width
        vse_content.extend_from_slice(&1080u16.to_be_bytes()); // height
        vse_content.extend_from_slice(&0x00480000u32.to_be_bytes()); // horizresolution 72dpi
        vse_content.extend_from_slice(&0x00480000u32.to_be_bytes()); // vertresolution
        vse_content.extend_from_slice(&[0; 4]); // reserved
        vse_content.extend_from_slice(&1u16.to_be_bytes()); // frame_count
        vse_content.extend_from_slice(&[0; 32]); // compressorname
        vse_content.extend_from_slice(&0x0018u16.to_be_bytes()); // depth
        vse_content.extend_from_slice(&[0xFF, 0xFF]); // pre_defined = -1
        vse_content.extend_from_slice(&hvcc_box);
        let vse = make_box(b"hvc1", &vse_content);

        let mut stsd_content = Vec::new();
        stsd_content.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        stsd_content.extend_from_slice(&vse);
        let stsd = make_fullbox(b"stsd", 0, 0, &stsd_content);

        // stts: 1 entry, 3 samples, delta=1000
        let mut stts_content = Vec::new();
        stts_content.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        stts_content.extend_from_slice(&3u32.to_be_bytes()); // sample_count
        stts_content.extend_from_slice(&1000u32.to_be_bytes()); // sample_delta
        let stts = make_fullbox(b"stts", 0, 0, &stts_content);

        // stsc: 1 entry, all in chunk 1
        let mut stsc_content = Vec::new();
        stsc_content.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        stsc_content.extend_from_slice(&1u32.to_be_bytes()); // first_chunk
        stsc_content.extend_from_slice(&3u32.to_be_bytes()); // samples_per_chunk
        stsc_content.extend_from_slice(&1u32.to_be_bytes()); // desc_index
        let stsc = make_fullbox(b"stsc", 0, 0, &stsc_content);

        // stsz: 3 samples, each 8 bytes
        let mut stsz_content = Vec::new();
        stsz_content.extend_from_slice(&0u32.to_be_bytes()); // sample_size (variable)
        stsz_content.extend_from_slice(&3u32.to_be_bytes()); // sample_count
        stsz_content.extend_from_slice(&8u32.to_be_bytes());
        stsz_content.extend_from_slice(&8u32.to_be_bytes());
        stsz_content.extend_from_slice(&8u32.to_be_bytes());
        let stsz = make_fullbox(b"stsz", 0, 0, &stsz_content);

        // stco: placeholder (will be patched)
        let mut stco_content = Vec::new();
        stco_content.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        stco_content.extend_from_slice(&0u32.to_be_bytes()); // placeholder offset
        let stco = make_fullbox(b"stco", 0, 0, &stco_content);

        // stss: sample 1 is sync
        let mut stss_content = Vec::new();
        stss_content.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        stss_content.extend_from_slice(&1u32.to_be_bytes()); // sample 1
        let stss = make_fullbox(b"stss", 0, 0, &stss_content);

        // Assemble stbl
        let mut stbl_content = Vec::new();
        stbl_content.extend_from_slice(&stsd);
        stbl_content.extend_from_slice(&stts);
        stbl_content.extend_from_slice(&stsc);
        stbl_content.extend_from_slice(&stsz);
        stbl_content.extend_from_slice(&stco);
        stbl_content.extend_from_slice(&stss);
        let stbl = make_box(b"stbl", &stbl_content);

        // dinf + dref (required by spec)
        let dref = make_fullbox(b"dref", 0, 0, &{
            let mut d = Vec::new();
            d.extend_from_slice(&1u32.to_be_bytes()); // entry_count
            let url = make_fullbox(b"url ", 0, 1, &[]); // self-contained
            d.extend_from_slice(&url);
            d
        });
        let dinf = make_box(b"dinf", &dref);

        // vmhd
        let vmhd = make_fullbox(b"vmhd", 0, 1, &[0; 8]);

        // minf
        let mut minf_content = Vec::new();
        minf_content.extend_from_slice(&vmhd);
        minf_content.extend_from_slice(&dinf);
        minf_content.extend_from_slice(&stbl);
        let minf = make_box(b"minf", &minf_content);

        // mdia
        let mut mdia_content = Vec::new();
        mdia_content.extend_from_slice(&mdhd);
        mdia_content.extend_from_slice(&hdlr);
        mdia_content.extend_from_slice(&minf);
        let mdia = make_box(b"mdia", &mdia_content);

        // trak
        let mut trak_content = Vec::new();
        trak_content.extend_from_slice(&tkhd);
        trak_content.extend_from_slice(&mdia);
        let trak = make_box(b"trak", &trak_content);

        // mvhd v0
        let mut mvhd_content = Vec::new();
        mvhd_content.extend_from_slice(&[0; 4]); // creation
        mvhd_content.extend_from_slice(&[0; 4]); // modification
        mvhd_content.extend_from_slice(&1000u32.to_be_bytes()); // timescale
        mvhd_content.extend_from_slice(&3000u32.to_be_bytes()); // duration
        mvhd_content.extend_from_slice(&0x00010000u32.to_be_bytes()); // rate 1.0
        mvhd_content.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
        mvhd_content.extend_from_slice(&[0; 10]); // reserved
        mvhd_content.extend_from_slice(&[     // identity matrix
            0x00,0x01,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00, 0x00,0x01,0x00,0x00, 0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x40,0x00,0x00,0x00,
        ]);
        mvhd_content.extend_from_slice(&[0; 24]); // pre_defined
        mvhd_content.extend_from_slice(&2u32.to_be_bytes()); // next_track_id
        let mvhd = make_fullbox(b"mvhd", 0, 0, &mvhd_content);

        // moov
        let mut moov_content = Vec::new();
        moov_content.extend_from_slice(&mvhd);
        moov_content.extend_from_slice(&trak);
        let moov = make_box(b"moov", &moov_content);

        mp4.extend_from_slice(&moov);

        // mdat: 8(header) + 24(3 samples × 8 bytes)
        let mut mdat_content = Vec::new();
        mdat_content.extend_from_slice(&sample1);
        mdat_content.extend_from_slice(&sample2);
        mdat_content.extend_from_slice(&sample3);
        let mdat = make_box(b"mdat", &mdat_content);

        // Now we know the mdat data offset: ftyp.len() + moov.len() + 8(mdat header)
        let mdat_data_offset = (ftyp.len() + moov.len() + 8) as u32;

        mp4.extend_from_slice(&mdat);

        // Patch the stco offset — find the stco box and write the correct offset
        // The stco entry is at a known position. Find "stco" in the buffer.
        let stco_needle = b"stco";
        for i in 0..mp4.len() - 4 {
            if &mp4[i..i + 4] == stco_needle {
                // stco box type is at i. Content starts at i+4 (version+flags).
                // entry_count at i+4+4=i+8, first offset at i+12
                let offset_pos = i + 4 + 4 + 4; // version+flags(4) + entry_count(4)
                mp4[offset_pos..offset_pos + 4]
                    .copy_from_slice(&mdat_data_offset.to_be_bytes());
                break;
            }
        }

        mp4
    }

    #[test]
    fn test_demux_minimal_mp4() {
        let mp4_data = build_test_mp4();
        let mp4 = demux(&mp4_data).unwrap();

        // Check ftyp
        assert!(!mp4.ftyp.is_empty());

        // Check video track found
        assert!(mp4.video_track_idx.is_some());
        let idx = mp4.video_track_idx.unwrap();
        let track = &mp4.tracks[idx];

        assert_eq!(track.track_id, 1);
        assert_eq!(track.handler_type, *b"vide");
        assert_eq!(track.codec, *b"hvc1");
        assert_eq!(track.width, 1920);
        assert_eq!(track.height, 1080);
        assert_eq!(track.timescale, 30000);
        assert_eq!(track.duration, 90000);

        // Check samples
        assert_eq!(track.samples.len(), 3);
        assert_eq!(track.samples[0].size, 8);
        assert!(track.samples[0].is_sync);
        assert!(!track.samples[1].is_sync);
        assert!(!track.samples[2].is_sync);

        // Check sample data was read correctly
        assert_eq!(
            track.samples[0].data,
            [0x00, 0x00, 0x00, 0x04, 0x28, 0x01, 0xAF, 0x09]
        );

        // Check hvcC
        let hvcc = track.hvcc_data.as_ref().unwrap();
        assert_eq!(hvcc.configuration_version, 1);
        assert_eq!(hvcc.length_size_minus1, 3);
        assert_eq!(hvcc.vps_nalus.len(), 1);
        assert_eq!(hvcc.vps_nalus[0], [0x40, 0x01, 0x0C]);
    }

    #[test]
    fn test_demux_no_video_track() {
        // ftyp + moov with no tracks
        let ftyp = make_box(b"ftyp", b"isom\x00\x00\x02\x00");

        let mvhd_content = vec![0u8; 100]; // simplified mvhd
        let mvhd = make_fullbox(b"mvhd", 0, 0, &mvhd_content);
        let moov = make_box(b"moov", &mvhd);
        let mdat = make_box(b"mdat", &[]);

        let mut mp4 = Vec::new();
        mp4.extend_from_slice(&ftyp);
        mp4.extend_from_slice(&moov);
        mp4.extend_from_slice(&mdat);

        let result = demux(&mp4).unwrap();
        assert!(result.video_track_idx.is_none());
        assert!(result.tracks.is_empty());
    }

    /// Build a fake HEVC NAL unit (length-prefixed, 4-byte prefix).
    fn make_nal(nal_type: u8, payload: &[u8]) -> Vec<u8> {
        // NAL header: byte0 = forbidden(0) | type(6 bits) | layer_msb(1 bit)
        //             byte1 = layer_lsb(5 bits) | temporal_id+1(3 bits)
        let byte0 = (nal_type << 1) & 0x7E; // forbidden=0, type shifted, layer_msb=0
        let byte1 = 0x01u8; // layer_lsb=0, temporal_id_plus1=1
        let nal_body_len = 2 + payload.len();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(nal_body_len as u32).to_be_bytes()); // 4-byte length prefix
        buf.push(byte0);
        buf.push(byte1);
        buf.extend_from_slice(payload);
        buf
    }

    #[test]
    #[ignore] // requires real DJI file
    fn test_debug_dji_structure() {
        let path = "/Users/cgaffga/Desktop/HEVC-H.265-samples/hevc_4k60P_main_dji_mavic3.mov";
        let data = if let Ok(d) = std::fs::read(path) { d } else { eprintln!("[SKIP] file not found"); return; };
        eprintln!("File size: {} bytes ({:.1} MB)", data.len(), data.len() as f64 / 1048576.0);

        // Scan top-level boxes
        let mut pos = 0usize;
        while pos + 8 <= data.len() {
            let header = parse_box_header(&data, pos).unwrap();
            let box_type = std::str::from_utf8(&header.box_type).unwrap_or("????");
            let box_end = pos as u64 + header.size;
            eprintln!("  pos={pos}: box={box_type:?} size={} header_len={} end={}", header.size, header.header_len, box_end);
            if header.size < 8 { break; }
            let next = pos + header.size as usize;
            if next > data.len() {
                eprintln!("    (overshoots file by {} bytes)", next - data.len());
                // Try to see what's inside mdat
                if header.box_type == *b"mdat" {
                    let content_start = pos + header.header_len as usize;
                    let content_end = data.len().min(pos + header.size as usize);
                    let mdat_data = &data[content_start..content_end];
                    eprintln!("    mdat content: {} bytes", mdat_data.len());
                    // Check first 32 bytes
                    let show = mdat_data.len().min(32);
                    eprint!("    first {} bytes: ", show);
                    for b in &mdat_data[..show] {
                        eprint!("{:02x} ", b);
                    }
                    eprintln!();

                    // Try Annex B parsing
                    eprintln!("    Trying Annex B scan...");
                    let mut sc_count = 0;
                    for i in 0..mdat_data.len().min(100_000_000).saturating_sub(3) {
                        if mdat_data[i] == 0 && mdat_data[i+1] == 0 && mdat_data[i+2] == 1 {
                            let nal_pos = i + 3;
                            if nal_pos < mdat_data.len() {
                                let nal_type = (mdat_data[nal_pos] >> 1) & 0x3F;
                                if sc_count < 20 {
                                    let prefix = if i > 0 && mdat_data[i-1] == 0 { "00 00 00 01" } else { "00 00 01" };
                                    eprintln!("    SC[{sc_count}] pos={i}: {prefix} -> NAL type={nal_type}");
                                }
                            }
                            sc_count += 1;
                        }
                    }
                    eprintln!("    Total start codes found (first 100MB): {sc_count}");
                }
                break;
            }
            pos = next;
        }
    }

    #[test]
    fn test_recover_truncated_basic() {
        // Build a truncated MP4: ftyp + mdat (with HEVC NAL units) but no moov.
        let mut data = Vec::new();

        // ftyp box
        let ftyp = make_box(b"ftyp", b"isom\x00\x00\x02\x00");
        data.extend_from_slice(&ftyp);

        // Build mdat content with HEVC NAL units:
        //   VPS + SPS + PPS + IDR slice (access unit 1, sync)
        //   TRAIL_R slice (access unit 2, non-sync)
        let mut mdat_content = Vec::new();

        // VPS (NAL type 32)
        mdat_content.extend_from_slice(&make_nal(32, &[0x0C, 0x01, 0x00, 0x00]));
        // SPS (NAL type 33) — minimal payload with dimensions.
        // We don't need a fully valid SPS for the recovery test, just enough to
        // pass the NAL type check. Dimensions extraction may return (0,0) for
        // a synthetic SPS, which is acceptable.
        mdat_content.extend_from_slice(&make_nal(33, &[0x01, 0x02, 0x20, 0x00, 0x00]));
        // PPS (NAL type 34)
        mdat_content.extend_from_slice(&make_nal(34, &[0xC1, 0x00]));
        // IDR_W_RADL (NAL type 19) — first_slice_segment_in_pic_flag=1 (MSB of byte 2)
        mdat_content.extend_from_slice(&make_nal(19, &[0x80, 0x00, 0xAA, 0xBB]));
        // TRAIL_R (NAL type 1) — first_slice_segment_in_pic_flag=1
        mdat_content.extend_from_slice(&make_nal(1, &[0x80, 0x00, 0xCC, 0xDD]));

        let mdat = make_box(b"mdat", &mdat_content);
        data.extend_from_slice(&mdat);

        // demux should succeed via recovery path
        let mp4 = demux(&data).unwrap();

        assert!(mp4.video_track_idx.is_some());
        let idx = mp4.video_track_idx.unwrap();
        let track = &mp4.tracks[idx];

        assert_eq!(track.handler_type, *b"vide");
        assert_eq!(track.codec, *b"hvc1");

        // Should have 2 samples (2 access units)
        assert_eq!(track.samples.len(), 2);
        assert!(track.samples[0].is_sync); // IDR
        assert!(!track.samples[1].is_sync); // TRAIL_R

        // Check hvcC was built
        let hvcc = track.hvcc_data.as_ref().unwrap();
        assert_eq!(hvcc.length_size_minus1, 3);
        assert_eq!(hvcc.vps_nalus.len(), 1);
        assert_eq!(hvcc.sps_nalus.len(), 1);
        assert_eq!(hvcc.pps_nalus.len(), 1);
    }

    #[test]
    fn test_recover_truncated_no_parameter_sets() {
        // ftyp + mdat with only VCL NALs (no VPS/SPS/PPS) → should fail
        let mut data = Vec::new();
        let ftyp = make_box(b"ftyp", b"isom\x00\x00\x02\x00");
        data.extend_from_slice(&ftyp);

        let mut mdat_content = Vec::new();
        mdat_content.extend_from_slice(&make_nal(19, &[0x80, 0x00, 0xAA])); // IDR
        mdat_content.extend_from_slice(&make_nal(1, &[0x80, 0x00, 0xBB])); // TRAIL

        let mdat = make_box(b"mdat", &mdat_content);
        data.extend_from_slice(&mdat);

        let result = demux(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_recover_truncated_oversized_mdat() {
        // Simulate a truncated file: mdat declares a larger size than actual data.
        let mut data = Vec::new();
        let ftyp = make_box(b"ftyp", b"isom\x00\x00\x02\x00");
        data.extend_from_slice(&ftyp);

        // Build NAL data
        let mut nal_data = Vec::new();
        nal_data.extend_from_slice(&make_nal(32, &[0x0C, 0x01, 0x00, 0x00])); // VPS
        nal_data.extend_from_slice(&make_nal(33, &[0x01, 0x02, 0x20, 0x00, 0x00])); // SPS
        nal_data.extend_from_slice(&make_nal(34, &[0xC1, 0x00])); // PPS
        nal_data.extend_from_slice(&make_nal(19, &[0x80, 0x00, 0xAA, 0xBB])); // IDR

        // Write mdat header with inflated size (claim 1MB but only write nal_data bytes)
        let fake_mdat_size: u32 = 1_000_000;
        data.extend_from_slice(&fake_mdat_size.to_be_bytes());
        data.extend_from_slice(b"mdat");
        data.extend_from_slice(&nal_data);

        // File ends here — mdat is "truncated"
        let mp4 = demux(&data).unwrap();
        assert!(mp4.video_track_idx.is_some());
        let track = &mp4.tracks[mp4.video_track_idx.unwrap()];
        assert_eq!(track.samples.len(), 1);
        assert!(track.samples[0].is_sync);
    }

    /// Build an Annex B NAL unit with 4-byte start code.
    fn make_annexb_nal(nal_type: u8, payload: &[u8]) -> Vec<u8> {
        let byte0 = (nal_type << 1) & 0x7E;
        let byte1 = 0x01u8;
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]); // 4-byte start code
        buf.push(byte0);
        buf.push(byte1);
        buf.extend_from_slice(payload);
        buf
    }

    #[test]
    fn test_recover_truncated_annexb() {
        // Build a truncated MP4 with Annex B NAL units inside mdat.
        // validate_nal_lengths should fail (start codes aren't valid length prefixes)
        // so it should fall back to recover_from_annexb.
        let mut data = Vec::new();
        let ftyp = make_box(b"ftyp", b"isom\x00\x00\x02\x00");
        data.extend_from_slice(&ftyp);

        // Build mdat content with Annex B start codes
        let mut mdat_content = Vec::new();
        mdat_content.extend_from_slice(&make_annexb_nal(32, &[0x0C, 0x01, 0x00, 0x00])); // VPS
        mdat_content.extend_from_slice(&make_annexb_nal(33, &[0x01, 0x02, 0x20, 0x00, 0x00])); // SPS
        mdat_content.extend_from_slice(&make_annexb_nal(34, &[0xC1, 0x00])); // PPS
        // IDR (first_slice_segment_in_pic_flag=1)
        mdat_content.extend_from_slice(&make_annexb_nal(19, &[0x80, 0x00, 0xAA, 0xBB]));
        // TRAIL_R (first_slice_segment_in_pic_flag=1 → new access unit)
        mdat_content.extend_from_slice(&make_annexb_nal(1, &[0x80, 0x00, 0xCC, 0xDD]));

        let mdat = make_box(b"mdat", &mdat_content);
        data.extend_from_slice(&mdat);

        let mp4 = demux(&data).unwrap();
        assert!(mp4.video_track_idx.is_some());
        let track = &mp4.tracks[mp4.video_track_idx.unwrap()];

        assert_eq!(track.handler_type, *b"vide");
        assert_eq!(track.codec, *b"hvc1");
        assert_eq!(track.samples.len(), 2);
        assert!(track.samples[0].is_sync);  // IDR
        assert!(!track.samples[1].is_sync); // TRAIL_R

        let hvcc = track.hvcc_data.as_ref().unwrap();
        assert_eq!(hvcc.vps_nalus.len(), 1);
        assert_eq!(hvcc.sps_nalus.len(), 1);
        assert_eq!(hvcc.pps_nalus.len(), 1);
    }

    #[test]
    fn test_recover_truncated_annexb_3byte_startcode() {
        // Test with 3-byte start codes (00 00 01) instead of 4-byte.
        let mut data = Vec::new();
        let ftyp = make_box(b"ftyp", b"isom\x00\x00\x02\x00");
        data.extend_from_slice(&ftyp);

        let mut mdat_content = Vec::new();
        // Use 3-byte start codes
        let make_3byte_nal = |nal_type: u8, payload: &[u8]| -> Vec<u8> {
            let byte0 = (nal_type << 1) & 0x7E;
            let mut buf = vec![0x00, 0x00, 0x01];
            buf.push(byte0);
            buf.push(0x01); // byte1
            buf.extend_from_slice(payload);
            buf
        };

        mdat_content.extend_from_slice(&make_3byte_nal(32, &[0x0C, 0x01, 0x00, 0x00]));
        mdat_content.extend_from_slice(&make_3byte_nal(33, &[0x01, 0x02, 0x20, 0x00, 0x00]));
        mdat_content.extend_from_slice(&make_3byte_nal(34, &[0xC1, 0x00]));
        mdat_content.extend_from_slice(&make_3byte_nal(19, &[0x80, 0x00, 0xAA, 0xBB]));

        let mdat = make_box(b"mdat", &mdat_content);
        data.extend_from_slice(&mdat);

        let mp4 = demux(&data).unwrap();
        assert!(mp4.video_track_idx.is_some());
        let track = &mp4.tracks[mp4.video_track_idx.unwrap()];
        assert_eq!(track.samples.len(), 1);
        assert!(track.samples[0].is_sync);
    }

    #[test]
    fn test_recover_truncated_multiple_irap() {
        // Multiple IDR access units (simulates a GOP boundary).
        let mut data = Vec::new();
        let ftyp = make_box(b"ftyp", b"isom\x00\x00\x02\x00");
        data.extend_from_slice(&ftyp);

        let mut mdat_content = Vec::new();
        // First GOP: VPS+SPS+PPS+IDR
        mdat_content.extend_from_slice(&make_nal(32, &[0x0C, 0x01, 0x00, 0x00]));
        mdat_content.extend_from_slice(&make_nal(33, &[0x01, 0x02, 0x20, 0x00, 0x00]));
        mdat_content.extend_from_slice(&make_nal(34, &[0xC1, 0x00]));
        mdat_content.extend_from_slice(&make_nal(19, &[0x80, 0x00, 0xAA, 0xBB]));
        // Inter frames
        mdat_content.extend_from_slice(&make_nal(1, &[0x80, 0x00, 0xCC, 0xDD]));
        mdat_content.extend_from_slice(&make_nal(1, &[0x80, 0x00, 0xEE, 0xFF]));
        // Second GOP: VPS+SPS+PPS+IDR
        mdat_content.extend_from_slice(&make_nal(32, &[0x0C, 0x01, 0x00, 0x00]));
        mdat_content.extend_from_slice(&make_nal(33, &[0x01, 0x02, 0x20, 0x00, 0x00]));
        mdat_content.extend_from_slice(&make_nal(34, &[0xC1, 0x00]));
        mdat_content.extend_from_slice(&make_nal(19, &[0x80, 0x00, 0x11, 0x22]));

        let mdat = make_box(b"mdat", &mdat_content);
        data.extend_from_slice(&mdat);

        let mp4 = demux(&data).unwrap();
        let track = &mp4.tracks[mp4.video_track_idx.unwrap()];

        // Should have 4 samples: IDR, TRAIL, TRAIL, IDR
        assert_eq!(track.samples.len(), 4);
        assert!(track.samples[0].is_sync);   // First IDR
        assert!(!track.samples[1].is_sync);  // TRAIL
        assert!(!track.samples[2].is_sync);  // TRAIL
        assert!(track.samples[3].is_sync);   // Second IDR
    }

    #[test]
    fn test_recover_no_mdat() {
        // File with ftyp only, no mdat → should return TruncatedFile error.
        let ftyp = make_box(b"ftyp", b"isom\x00\x00\x02\x00");
        let result = demux(&ftyp);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_nal_lengths() {
        // Valid length-prefixed NALs
        let mut valid = Vec::new();
        valid.extend_from_slice(&make_nal(32, &[0x0C, 0x01])); // VPS: 4 + 4 = 8 bytes
        valid.extend_from_slice(&make_nal(33, &[0x01, 0x02])); // SPS: 4 + 4 = 8 bytes
        assert!(validate_nal_lengths(&valid, 4));

        // Invalid: random garbage
        let garbage = [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00];
        assert!(!validate_nal_lengths(&garbage, 4));

        // Invalid: Annex B start codes (should fail length validation)
        let mut annexb = Vec::new();
        annexb.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]); // looks like length=1
        annexb.extend_from_slice(&[0x40, 0x01]); // VPS NAL header
        // Length prefix "00 00 00 01" = 1, but only 2 bytes of NAL data
        // validate checks 8 NALs — second will likely fail
        assert!(!validate_nal_lengths(&annexb, 4));
    }
}
