// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! MP4 muxer — rebuild an MP4 file with modified video samples.
//!
//! Takes a demuxed [`Mp4File`], the original file bytes, and a list of
//! (sample_index, new_data) replacements. Produces a valid MP4 with:
//! - Updated mdat containing modified + unmodified sample data
//! - Corrected stsz entries for changed sample sizes
//! - Recomputed stco/co64 chunk offsets
//! - All non-video tracks preserved byte-for-byte

use super::{
    iterate_boxes, parse_box_header, read_u32, write_u32, write_u64, Mp4Error, Mp4File,
};
use std::collections::HashMap;

/// Rebuild an MP4 file with modified video samples.
///
/// # Arguments
///
/// - `original` — Original MP4 file bytes (used for non-modified data)
/// - `mp4` — Parsed MP4 structure from [`super::demux::demux`]
/// - `modified_samples` — List of `(sample_index, new_data)` pairs for the video track.
///   `sample_index` is 0-based into `mp4.tracks[video_track_idx].samples`.
///
/// # Returns
///
/// Complete MP4 file bytes with modifications applied.
pub fn mux(
    original: &[u8],
    mp4: &Mp4File,
    modified_samples: &[(usize, Vec<u8>)],
) -> Result<Vec<u8>, Mp4Error> {
    let video_idx = mp4
        .video_track_idx
        .ok_or(Mp4Error::NoVideoTrack)?;
    let video_track = &mp4.tracks[video_idx];

    // Build modification lookup: sample_index → new data
    let mods: HashMap<usize, &[u8]> = modified_samples
        .iter()
        .map(|(idx, data)| (*idx, data.as_slice()))
        .collect();

    // ─── Phase 1: Compute new sample sizes for the video track ───

    let new_sample_sizes: Vec<u32> = video_track
        .samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if let Some(new_data) = mods.get(&i) {
                new_data.len() as u32
            } else {
                s.size
            }
        })
        .collect();

    // ─── Phase 2: Build new mdat ───
    // Write all track samples in their original interleaved order.
    // We sort all samples (across all tracks) by their original file offset,
    // then write them sequentially. This preserves interleaving.

    // Collect all samples with track index and sample index
    struct SampleRef {
        track_idx: usize,
        sample_idx: usize,
        original_offset: u64,
        original_size: u32,
    }

    let mut all_samples: Vec<SampleRef> = Vec::new();
    for (ti, track) in mp4.tracks.iter().enumerate() {
        for (si, sample) in track.samples.iter().enumerate() {
            all_samples.push(SampleRef {
                track_idx: ti,
                sample_idx: si,
                original_offset: sample.offset,
                original_size: sample.size,
            });
        }
    }
    // Sort by original file offset to preserve interleaving
    all_samples.sort_by_key(|s| s.original_offset);

    // Write mdat content and record new offsets
    let mut mdat_content = Vec::new();
    // Track → sample_idx → new absolute offset
    let mut new_offsets: HashMap<(usize, usize), u64> = HashMap::new();

    // We'll compute absolute offsets later once we know the moov size.
    // For now, record relative offsets within mdat content.
    for sr in &all_samples {
        let rel_offset = mdat_content.len();
        new_offsets.insert((sr.track_idx, sr.sample_idx), rel_offset as u64);

        if sr.track_idx == video_idx {
            if let Some(new_data) = mods.get(&sr.sample_idx) {
                mdat_content.extend_from_slice(new_data);
            } else {
                // Copy original sample data
                let start = sr.original_offset as usize;
                let end = start + sr.original_size as usize;
                if end > original.len() {
                    return Err(Mp4Error::UnexpectedEof);
                }
                mdat_content.extend_from_slice(&original[start..end]);
            }
        } else {
            // Non-video track: copy original data
            let start = sr.original_offset as usize;
            let end = start + sr.original_size as usize;
            if end > original.len() {
                return Err(Mp4Error::UnexpectedEof);
            }
            mdat_content.extend_from_slice(&original[start..end]);
        }
    }

    // ─── Phase 3: Build output file ───
    // Layout: ftyp + moov (with patched stbl tables) + mdat

    let mut output = Vec::new();

    // Write ftyp
    output.extend_from_slice(&mp4.ftyp);

    // Build the new moov box. We need to:
    // 1. For non-video tracks: copy trak_raw unchanged, but patch stco/co64
    // 2. For the video track: rebuild trak with updated stsz + stco/co64
    //
    // Strategy: Copy the original moov and patch in-place.
    // But it's cleaner to find the moov in original, copy it, then patch.

    // Find original moov box location
    let (moov_start, moov_size) = find_top_level_box(original, b"moov")?;
    let mut moov_data = original[moov_start..moov_start + moov_size].to_vec();

    // The mdat header is 8 bytes (or 16 for 64-bit size)
    let mdat_header_size: usize = if mdat_content.len() + 8 > u32::MAX as usize {
        16 // Need 64-bit extended size
    } else {
        8
    };

    // Absolute offset where mdat content starts in the output file
    let mdat_content_offset = mp4.ftyp.len() + moov_data.len() + mdat_header_size;

    // Convert relative offsets to absolute
    let abs_offsets: HashMap<(usize, usize), u64> = new_offsets
        .iter()
        .map(|(&key, &rel)| (key, rel + mdat_content_offset as u64))
        .collect();

    // ─── Phase 4: Patch moov ───
    // For each track, update stsz and stco/co64 entries inside the moov copy.

    for (ti, track) in mp4.tracks.iter().enumerate() {
        if track.samples.is_empty() {
            continue;
        }

        // Compute new chunk offsets for this track using the original stsc + stco
        // structure from the moov box.
        let chunk_first_samples = compute_chunk_first_samples(&moov_data, track)?;

        // Find and patch stco/co64 in moov_data for this track
        let is_video = ti == video_idx;

        // Patch stsz for video track
        if is_video {
            patch_stsz_in_moov(&mut moov_data, track, &new_sample_sizes)?;
        }

        // Patch stco/co64 for all tracks
        let chunk_offsets: Vec<u64> = chunk_first_samples
            .iter()
            .map(|&(sample_idx, _)| {
                *abs_offsets
                    .get(&(ti, sample_idx))
                    .unwrap_or(&0)
            })
            .collect();

        patch_stco_in_moov(&mut moov_data, track, &chunk_offsets)?;
    }

    output.extend_from_slice(&moov_data);

    // Write mdat box
    if mdat_header_size == 16 {
        // 64-bit extended size
        write_u32(&mut output, 1); // size = 1 → extended
        output.extend_from_slice(b"mdat");
        write_u64(&mut output, (mdat_content.len() + 16) as u64);
    } else {
        write_u32(&mut output, (mdat_content.len() + 8) as u32);
        output.extend_from_slice(b"mdat");
    }
    output.extend_from_slice(&mdat_content);

    Ok(output)
}

// ─── Streaming mux ─────────────────────────────────────────────────────

use std::io::{Read, Seek, SeekFrom, Write};

/// Rebuild an MP4 file using streaming I/O.
///
/// Reads the original moov + sample data from `reader` on demand (using seek),
/// patches sample tables, then writes ftyp + patched moov + mdat to `writer`.
/// Modified samples come from `modified_samples`; unmodified samples are
/// copied from `reader` via seek.
///
/// Memory: only one sample in memory at a time (plus moov metadata).
pub fn mux_streaming<R: Read + Seek, W: Write>(
    reader: &mut R,
    mp4: &Mp4File,
    modified_samples: &[(usize, Vec<u8>)],
    writer: &mut W,
) -> Result<(), Mp4Error> {
    let video_idx = mp4
        .video_track_idx
        .ok_or(Mp4Error::NoVideoTrack)?;
    let video_track = &mp4.tracks[video_idx];

    let mods: HashMap<usize, &[u8]> = modified_samples
        .iter()
        .map(|(idx, data)| (*idx, data.as_slice()))
        .collect();

    // Compute new sample sizes for the video track
    let new_sample_sizes: Vec<u32> = video_track
        .samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if let Some(new_data) = mods.get(&i) {
                new_data.len() as u32
            } else {
                s.size
            }
        })
        .collect();

    // Collect all samples sorted by original file offset (preserves interleaving)
    struct SampleRef {
        track_idx: usize,
        sample_idx: usize,
        original_offset: u64,
        original_size: u32,
    }

    let mut all_samples: Vec<SampleRef> = Vec::new();
    for (ti, track) in mp4.tracks.iter().enumerate() {
        for (si, sample) in track.samples.iter().enumerate() {
            all_samples.push(SampleRef {
                track_idx: ti,
                sample_idx: si,
                original_offset: sample.offset,
                original_size: sample.size,
            });
        }
    }
    all_samples.sort_by_key(|s| s.original_offset);

    // First pass: compute mdat content size to know offsets for moov patching
    let mut mdat_content_size: u64 = 0;
    let mut sample_rel_offsets: Vec<((usize, usize), u64)> = Vec::new();
    for sr in &all_samples {
        sample_rel_offsets.push(((sr.track_idx, sr.sample_idx), mdat_content_size));
        if sr.track_idx == video_idx {
            if let Some(new_data) = mods.get(&sr.sample_idx) {
                mdat_content_size += new_data.len() as u64;
            } else {
                mdat_content_size += sr.original_size as u64;
            }
        } else {
            mdat_content_size += sr.original_size as u64;
        }
    }

    let rel_offsets: HashMap<(usize, usize), u64> = sample_rel_offsets.into_iter().collect();

    // Read moov from original file
    let moov_range = find_moov_streaming(reader)?;
    reader.seek(SeekFrom::Start(moov_range.0)).map_err(|_| Mp4Error::UnexpectedEof)?;
    let moov_size = (moov_range.1 - moov_range.0) as usize;
    let mut moov_data = vec![0u8; moov_size];
    reader.read_exact(&mut moov_data).map_err(|_| Mp4Error::UnexpectedEof)?;

    let mdat_header_size: usize = if mdat_content_size + 8 > u32::MAX as u64 { 16 } else { 8 };
    let mdat_content_offset = mp4.ftyp.len() as u64 + moov_data.len() as u64 + mdat_header_size as u64;

    // Convert relative offsets to absolute
    let abs_offsets: HashMap<(usize, usize), u64> = rel_offsets
        .iter()
        .map(|(&key, &rel)| (key, rel + mdat_content_offset))
        .collect();

    // Patch moov: stsz + stco/co64 for each track
    for (ti, track) in mp4.tracks.iter().enumerate() {
        if track.samples.is_empty() {
            continue;
        }

        let chunk_first_samples = compute_chunk_first_samples(&moov_data, track)?;
        let is_video = ti == video_idx;

        if is_video {
            patch_stsz_in_moov(&mut moov_data, track, &new_sample_sizes)?;
        }

        let chunk_offsets: Vec<u64> = chunk_first_samples
            .iter()
            .map(|&(sample_idx, _)| {
                *abs_offsets.get(&(ti, sample_idx)).unwrap_or(&0)
            })
            .collect();

        patch_stco_in_moov(&mut moov_data, track, &chunk_offsets)?;
    }

    // Write ftyp
    writer.write_all(&mp4.ftyp).map_err(|_| Mp4Error::InvalidBox("write failed".into()))?;

    // Write patched moov
    writer.write_all(&moov_data).map_err(|_| Mp4Error::InvalidBox("write failed".into()))?;

    // Write mdat header
    if mdat_header_size == 16 {
        let mut hdr = Vec::with_capacity(16);
        write_u32(&mut hdr, 1);
        hdr.extend_from_slice(b"mdat");
        write_u64(&mut hdr, mdat_content_size + 16);
        writer.write_all(&hdr).map_err(|_| Mp4Error::InvalidBox("write failed".into()))?;
    } else {
        let mut hdr = Vec::with_capacity(8);
        write_u32(&mut hdr, (mdat_content_size + 8) as u32);
        hdr.extend_from_slice(b"mdat");
        writer.write_all(&hdr).map_err(|_| Mp4Error::InvalidBox("write failed".into()))?;
    }

    // Write mdat content: stream samples from reader or modified data
    let mut buf = Vec::new();
    for sr in &all_samples {
        if sr.track_idx == video_idx
            && let Some(new_data) = mods.get(&sr.sample_idx) {
                writer.write_all(new_data).map_err(|_| Mp4Error::InvalidBox("write failed".into()))?;
                continue;
            }

        // Read original sample data from reader via seek
        let size = sr.original_size as usize;
        buf.resize(size, 0);
        reader.seek(SeekFrom::Start(sr.original_offset)).map_err(|_| Mp4Error::UnexpectedEof)?;
        reader.read_exact(&mut buf).map_err(|_| Mp4Error::UnexpectedEof)?;
        writer.write_all(&buf).map_err(|_| Mp4Error::InvalidBox("write failed".into()))?;
    }

    Ok(())
}

/// Find the moov box location in a file via streaming search.
/// Returns (start_offset, end_offset) of the moov box.
fn find_moov_streaming<R: Read + Seek>(reader: &mut R) -> Result<(u64, u64), Mp4Error> {
    reader.seek(SeekFrom::Start(0)).map_err(|_| Mp4Error::UnexpectedEof)?;
    let file_size = reader.seek(SeekFrom::End(0)).map_err(|_| Mp4Error::UnexpectedEof)?;
    reader.seek(SeekFrom::Start(0)).map_err(|_| Mp4Error::UnexpectedEof)?;

    let mut pos: u64 = 0;
    let mut header_buf = [0u8; 16];

    while pos < file_size {
        if pos + 8 > file_size {
            break;
        }
        reader.seek(SeekFrom::Start(pos)).map_err(|_| Mp4Error::UnexpectedEof)?;
        reader.read_exact(&mut header_buf[..8]).map_err(|_| Mp4Error::UnexpectedEof)?;

        let mut box_size = u32::from_be_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]]) as u64;
        let box_type = [header_buf[4], header_buf[5], header_buf[6], header_buf[7]];

        if box_size == 1 {
            // 64-bit extended size
            if pos + 16 > file_size {
                break;
            }
            reader.read_exact(&mut header_buf[8..16]).map_err(|_| Mp4Error::UnexpectedEof)?;
            box_size = u64::from_be_bytes([
                header_buf[8], header_buf[9], header_buf[10], header_buf[11],
                header_buf[12], header_buf[13], header_buf[14], header_buf[15],
            ]);
        }

        if box_size < 8 || pos + box_size > file_size {
            break;
        }

        if box_type == *b"moov" {
            return Ok((pos, pos + box_size));
        }

        pos += box_size;
    }

    Err(Mp4Error::InvalidBox("missing moov box".into()))
}

/// Find a top-level box in the MP4 data. Returns (offset, size).
fn find_top_level_box(data: &[u8], box_type: &[u8; 4]) -> Result<(usize, usize), Mp4Error> {
    let mut pos = 0;
    while pos < data.len() {
        if pos + 8 > data.len() {
            break;
        }
        let header = parse_box_header(data, pos)?;
        if header.box_type == *box_type {
            return Ok((pos, header.size as usize));
        }
        let next = pos + header.size as usize;
        if next <= pos || next > data.len() {
            break; // Prevent infinite loop or oversized box
        }
        pos = next;
    }
    Err(Mp4Error::InvalidBox(format!(
        "missing {} box",
        std::str::from_utf8(box_type).unwrap_or("????")
    )))
}

/// Compute (sample_index, samples_in_chunk) for the first sample of each chunk.
///
/// Reads the stco/co64 entry count and stsc entries from the moov box to
/// reconstruct the original chunk structure. This is more reliable than
/// heuristic offset-discontinuity detection, which fails for single-track
/// files where all samples are contiguous in the mdat.
fn compute_chunk_first_samples(
    moov_data: &[u8],
    track: &super::Track,
) -> Result<Vec<(usize, usize)>, Mp4Error> {
    if track.samples.is_empty() {
        return Ok(Vec::new());
    }

    let trak_range = find_trak_range(moov_data, track.track_id)?;

    // Read chunk count from stco or co64
    let num_chunks = if let Some(stco_off) =
        find_box_in_range(moov_data, trak_range.0, trak_range.1, b"stco")
    {
        let header = parse_box_header(moov_data, stco_off)?;
        let cs = stco_off + header.header_len as usize;
        read_u32(moov_data, cs + 4)? as usize
    } else if let Some(co64_off) =
        find_box_in_range(moov_data, trak_range.0, trak_range.1, b"co64")
    {
        let header = parse_box_header(moov_data, co64_off)?;
        let cs = co64_off + header.header_len as usize;
        read_u32(moov_data, cs + 4)? as usize
    } else {
        return Ok(Vec::new());
    };

    // Read stsc entries
    let stsc_entries = if let Some(stsc_off) =
        find_box_in_range(moov_data, trak_range.0, trak_range.1, b"stsc")
    {
        let header = parse_box_header(moov_data, stsc_off)?;
        let cs = stsc_off + header.header_len as usize;
        super::demux::parse_stsc(moov_data, cs)?
    } else {
        Vec::new()
    };

    // Expand stsc → samples_per_chunk for every chunk (same logic as demux)
    let mut samples_per_chunk = vec![0u32; num_chunks];
    if stsc_entries.is_empty() {
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
                if chunk_idx >= 1 && chunk_idx <= num_chunks {
                    samples_per_chunk[chunk_idx - 1] = spc;
                }
            }
        }
    }

    // Compute (first_sample_idx, num_samples_in_chunk) for each chunk
    let mut chunks = Vec::with_capacity(num_chunks);
    let mut sample_idx = 0usize;
    for &spc in &samples_per_chunk {
        let count = spc as usize;
        chunks.push((sample_idx, count));
        sample_idx += count;
    }

    Ok(chunks)
}

/// Patch the stsz box inside moov_data for the video track.
///
/// Finds the stsz box that belongs to this track by matching the trak box
/// via track_raw bytes, then overwrites the sample size entries.
fn patch_stsz_in_moov(
    moov_data: &mut [u8],
    track: &super::Track,
    new_sizes: &[u32],
) -> Result<(), Mp4Error> {
    // Find this track's trak box within moov by matching track_id in tkhd.
    // Then find the stsz box within that trak.
    let trak_range = find_trak_range(moov_data, track.track_id)?;

    // Find stsz within the trak range
    if let Some(stsz_offset) = find_box_in_range(moov_data, trak_range.0, trak_range.1, b"stsz") {
        let header = parse_box_header(moov_data, stsz_offset)?;
        let content_start = stsz_offset + header.header_len as usize;

        // stsz: version+flags(4) + sample_size(4) + sample_count(4) + entries...
        let sample_size_field = read_u32(moov_data, content_start + 4)?;

        if sample_size_field != 0 {
            // Fixed-size stsz — if sizes changed, we can't patch in-place without
            // restructuring. For now, this is an error (HEVC typically uses variable sizes).
            // Check if all new sizes are equal.
            let all_same = new_sizes.windows(2).all(|w| w[0] == w[1]);
            if all_same && !new_sizes.is_empty() {
                // Update the fixed sample_size field
                let val = new_sizes[0];
                moov_data[content_start + 4..content_start + 8]
                    .copy_from_slice(&val.to_be_bytes());
                return Ok(());
            }
            // If sizes differ, we would need to expand the box — not supported in patch mode.
            // In practice, HEVC video always uses variable stsz.
            return Err(Mp4Error::InvalidBox(
                "cannot convert fixed-size stsz to variable in-place".into(),
            ));
        }

        let count = read_u32(moov_data, content_start + 8)? as usize;
        if count != new_sizes.len() {
            return Err(Mp4Error::InvalidBox(format!(
                "stsz sample count mismatch: {} vs {}",
                count,
                new_sizes.len()
            )));
        }

        // Overwrite per-sample sizes
        for (i, &size) in new_sizes.iter().enumerate() {
            let pos = content_start + 12 + i * 4;
            moov_data[pos..pos + 4].copy_from_slice(&size.to_be_bytes());
        }
    }

    Ok(())
}

/// Patch stco or co64 box inside moov_data for a given track.
fn patch_stco_in_moov(
    moov_data: &mut [u8],
    track: &super::Track,
    new_chunk_offsets: &[u64],
) -> Result<(), Mp4Error> {
    let trak_range = find_trak_range(moov_data, track.track_id)?;

    // Try stco first
    if let Some(stco_offset) = find_box_in_range(moov_data, trak_range.0, trak_range.1, b"stco") {
        let header = parse_box_header(moov_data, stco_offset)?;
        let content_start = stco_offset + header.header_len as usize;
        let count = read_u32(moov_data, content_start + 4)? as usize;

        // Check if any offset exceeds 32-bit range
        let needs_co64 = new_chunk_offsets.iter().any(|&o| o > u32::MAX as u64);
        if needs_co64 {
            // Would need to replace stco with co64, changing box size.
            // For files under 4GB this won't happen. Log and error.
            return Err(Mp4Error::InvalidBox(
                "mdat exceeds 4GB, stco→co64 upgrade not supported in-place".into(),
            ));
        }

        if count != new_chunk_offsets.len() {
            return Err(Mp4Error::InvalidBox(format!(
                "stco entry count mismatch: {} vs {}",
                count,
                new_chunk_offsets.len()
            )));
        }

        for (i, &offset) in new_chunk_offsets.iter().enumerate() {
            let pos = content_start + 8 + i * 4;
            moov_data[pos..pos + 4].copy_from_slice(&(offset as u32).to_be_bytes());
        }

        return Ok(());
    }

    // Try co64
    if let Some(co64_offset) = find_box_in_range(moov_data, trak_range.0, trak_range.1, b"co64") {
        let header = parse_box_header(moov_data, co64_offset)?;
        let content_start = co64_offset + header.header_len as usize;
        let count = read_u32(moov_data, content_start + 4)? as usize;

        if count != new_chunk_offsets.len() {
            return Err(Mp4Error::InvalidBox(format!(
                "co64 entry count mismatch: {} vs {}",
                count,
                new_chunk_offsets.len()
            )));
        }

        for (i, &offset) in new_chunk_offsets.iter().enumerate() {
            let pos = content_start + 8 + i * 8;
            moov_data[pos..pos + 8].copy_from_slice(&offset.to_be_bytes());
        }

        return Ok(());
    }

    // No chunk offset box found — track might have no samples
    Ok(())
}

/// Find the byte range of a trak box within moov_data by matching track_id in tkhd.
///
/// Returns `(start, end)` byte offsets within `moov_data`.
fn find_trak_range(moov_data: &[u8], target_track_id: u32) -> Result<(usize, usize), Mp4Error> {
    let moov_header = parse_box_header(moov_data, 0)?;
    let moov_content_start = moov_header.header_len as usize;
    let moov_content_end = moov_header.size as usize;

    let mut found: Option<(usize, usize)> = None;

    iterate_boxes(
        moov_data,
        moov_content_start,
        moov_content_end,
        |header, content_start, _box_data| {
            if header.box_type == *b"trak" && found.is_none() {
                let trak_start = content_start - header.header_len as usize;
                let trak_end = trak_start + header.size as usize;

                // Look for tkhd inside this trak to get the track_id
                if let Some(track_id) = extract_track_id(moov_data, content_start, trak_end)
                    && track_id == target_track_id {
                        found = Some((trak_start, trak_end));
                    }
            }
            Ok(())
        },
    )?;

    found.ok_or_else(|| {
        Mp4Error::InvalidBox(format!("trak with track_id={target_track_id} not found in moov"))
    })
}

/// Extract track_id from the tkhd box within a trak range.
fn extract_track_id(data: &[u8], start: usize, end: usize) -> Option<u32> {
    let mut result = None;
    let _ = iterate_boxes(data, start, end, |header, content_start, _| {
        if header.box_type == *b"tkhd" {
            let version = data[content_start];
            let track_id = if version == 1 {
                // v1: ver+flags(4) + creation(8) + modification(8) + track_id(4)
                read_u32(data, content_start + 4 + 16).ok()
            } else {
                // v0: ver+flags(4) + creation(4) + modification(4) + track_id(4)
                read_u32(data, content_start + 4 + 8).ok()
            };
            result = track_id;
        }
        Ok(())
    });
    result
}

/// Find a box with the given type within a byte range (recursive search).
///
/// Returns the absolute offset of the box header within `data`, or None.
fn find_box_in_range(
    data: &[u8],
    start: usize,
    end: usize,
    box_type: &[u8; 4],
) -> Option<usize> {
    let mut found = None;
    let _ = iterate_boxes(data, start, end, |header, content_start, _| {
        if found.is_some() {
            return Ok(());
        }
        let box_start = content_start - header.header_len as usize;
        if header.box_type == *box_type {
            found = Some(box_start);
        } else {
            // Recurse into container boxes
            let child_end = box_start + header.size as usize;
            match &header.box_type {
                b"trak" | b"mdia" | b"minf" | b"stbl" | b"moov" => {
                    if let Some(inner) = find_box_in_range(data, content_start, child_end, box_type)
                    {
                        found = Some(inner);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    });
    found
}

#[cfg(test)]
mod tests {
    use super::super::demux::demux;
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

    /// Helper: build a fullbox.
    fn make_fullbox(box_type: &[u8; 4], version: u8, flags: u32, content: &[u8]) -> Vec<u8> {
        let mut inner = Vec::new();
        inner.push(version);
        inner.extend_from_slice(&(flags & 0x00FF_FFFF).to_be_bytes()[1..]);
        inner.extend_from_slice(content);
        make_box(box_type, &inner)
    }

    /// Build a minimal test MP4 (same as demux tests).
    fn build_test_mp4() -> Vec<u8> {
        // Reuse the test MP4 builder from demux tests
        let mut mp4 = Vec::new();

        let ftyp = make_box(b"ftyp", b"isom\x00\x00\x02\x00isom");
        mp4.extend_from_slice(&ftyp);

        let sample1 = [0x00, 0x00, 0x00, 0x04, 0x28, 0x01, 0xAF, 0x09];
        let sample2 = [0x00, 0x00, 0x00, 0x04, 0x02, 0x01, 0xD0, 0x10];
        let sample3 = [0x00, 0x00, 0x00, 0x04, 0x02, 0x01, 0xD0, 0x11];

        // tkhd
        let mut tkhd_content = Vec::new();
        tkhd_content.extend_from_slice(&[0; 4]);
        tkhd_content.extend_from_slice(&[0; 4]);
        tkhd_content.extend_from_slice(&1u32.to_be_bytes());
        tkhd_content.extend_from_slice(&[0; 4]);
        tkhd_content.extend_from_slice(&90u32.to_be_bytes());
        tkhd_content.extend_from_slice(&[0; 8]);
        tkhd_content.extend_from_slice(&[0; 2]);
        tkhd_content.extend_from_slice(&[0; 2]);
        tkhd_content.extend_from_slice(&[0; 2]);
        tkhd_content.extend_from_slice(&[0; 2]);
        tkhd_content.extend_from_slice(&[
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
        ]);
        tkhd_content.extend_from_slice(&((1920u32) << 16).to_be_bytes());
        tkhd_content.extend_from_slice(&((1080u32) << 16).to_be_bytes());
        let tkhd = make_fullbox(b"tkhd", 0, 3, &tkhd_content);

        let mut mdhd_content = Vec::new();
        mdhd_content.extend_from_slice(&[0; 4]);
        mdhd_content.extend_from_slice(&[0; 4]);
        mdhd_content.extend_from_slice(&30000u32.to_be_bytes());
        mdhd_content.extend_from_slice(&90000u32.to_be_bytes());
        mdhd_content.extend_from_slice(&[0x55, 0xC4]);
        mdhd_content.extend_from_slice(&[0; 2]);
        let mdhd = make_fullbox(b"mdhd", 0, 0, &mdhd_content);

        let mut hdlr_content = Vec::new();
        hdlr_content.extend_from_slice(&[0; 4]);
        hdlr_content.extend_from_slice(b"vide");
        hdlr_content.extend_from_slice(&[0; 12]);
        hdlr_content.push(0);
        let hdlr = make_fullbox(b"hdlr", 0, 0, &hdlr_content);

        let mut hvcc_content = Vec::new();
        hvcc_content.push(1);
        hvcc_content.push(0);
        hvcc_content.extend_from_slice(&[0; 4]);
        hvcc_content.extend_from_slice(&[0; 6]);
        hvcc_content.push(0);
        hvcc_content.extend_from_slice(&[0xF0, 0x00]);
        hvcc_content.push(0xFC);
        hvcc_content.push(0xFC);
        hvcc_content.push(0xF8);
        hvcc_content.push(0xF8);
        hvcc_content.extend_from_slice(&[0, 0]);
        hvcc_content.push(0x0F);
        hvcc_content.push(1);
        hvcc_content.push(0x20);
        hvcc_content.extend_from_slice(&1u16.to_be_bytes());
        let vps = [0x40, 0x01, 0x0C];
        hvcc_content.extend_from_slice(&(vps.len() as u16).to_be_bytes());
        hvcc_content.extend_from_slice(&vps);
        let hvcc_box = make_box(b"hvcC", &hvcc_content);

        let mut vse_content = Vec::new();
        vse_content.extend_from_slice(&[0; 6]);
        vse_content.extend_from_slice(&1u16.to_be_bytes());
        vse_content.extend_from_slice(&[0; 2]);
        vse_content.extend_from_slice(&[0; 2]);
        vse_content.extend_from_slice(&[0; 12]);
        vse_content.extend_from_slice(&1920u16.to_be_bytes());
        vse_content.extend_from_slice(&1080u16.to_be_bytes());
        vse_content.extend_from_slice(&0x00480000u32.to_be_bytes());
        vse_content.extend_from_slice(&0x00480000u32.to_be_bytes());
        vse_content.extend_from_slice(&[0; 4]);
        vse_content.extend_from_slice(&1u16.to_be_bytes());
        vse_content.extend_from_slice(&[0; 32]);
        vse_content.extend_from_slice(&0x0018u16.to_be_bytes());
        vse_content.extend_from_slice(&[0xFF, 0xFF]);
        vse_content.extend_from_slice(&hvcc_box);
        let vse = make_box(b"hvc1", &vse_content);

        let mut stsd_content = Vec::new();
        stsd_content.extend_from_slice(&1u32.to_be_bytes());
        stsd_content.extend_from_slice(&vse);
        let stsd = make_fullbox(b"stsd", 0, 0, &stsd_content);

        let mut stts_content = Vec::new();
        stts_content.extend_from_slice(&1u32.to_be_bytes());
        stts_content.extend_from_slice(&3u32.to_be_bytes());
        stts_content.extend_from_slice(&1000u32.to_be_bytes());
        let stts = make_fullbox(b"stts", 0, 0, &stts_content);

        let mut stsc_content = Vec::new();
        stsc_content.extend_from_slice(&1u32.to_be_bytes());
        stsc_content.extend_from_slice(&1u32.to_be_bytes());
        stsc_content.extend_from_slice(&3u32.to_be_bytes());
        stsc_content.extend_from_slice(&1u32.to_be_bytes());
        let stsc = make_fullbox(b"stsc", 0, 0, &stsc_content);

        let mut stsz_content = Vec::new();
        stsz_content.extend_from_slice(&0u32.to_be_bytes());
        stsz_content.extend_from_slice(&3u32.to_be_bytes());
        stsz_content.extend_from_slice(&8u32.to_be_bytes());
        stsz_content.extend_from_slice(&8u32.to_be_bytes());
        stsz_content.extend_from_slice(&8u32.to_be_bytes());
        let stsz = make_fullbox(b"stsz", 0, 0, &stsz_content);

        let mut stco_content = Vec::new();
        stco_content.extend_from_slice(&1u32.to_be_bytes());
        stco_content.extend_from_slice(&0u32.to_be_bytes());
        let stco = make_fullbox(b"stco", 0, 0, &stco_content);

        let mut stss_content = Vec::new();
        stss_content.extend_from_slice(&1u32.to_be_bytes());
        stss_content.extend_from_slice(&1u32.to_be_bytes());
        let stss = make_fullbox(b"stss", 0, 0, &stss_content);

        let mut stbl_content = Vec::new();
        stbl_content.extend_from_slice(&stsd);
        stbl_content.extend_from_slice(&stts);
        stbl_content.extend_from_slice(&stsc);
        stbl_content.extend_from_slice(&stsz);
        stbl_content.extend_from_slice(&stco);
        stbl_content.extend_from_slice(&stss);
        let stbl = make_box(b"stbl", &stbl_content);

        let dref = make_fullbox(b"dref", 0, 0, &{
            let mut d = Vec::new();
            d.extend_from_slice(&1u32.to_be_bytes());
            let url = make_fullbox(b"url ", 0, 1, &[]);
            d.extend_from_slice(&url);
            d
        });
        let dinf = make_box(b"dinf", &dref);
        let vmhd = make_fullbox(b"vmhd", 0, 1, &[0; 8]);

        let mut minf_content = Vec::new();
        minf_content.extend_from_slice(&vmhd);
        minf_content.extend_from_slice(&dinf);
        minf_content.extend_from_slice(&stbl);
        let minf = make_box(b"minf", &minf_content);

        let mut mdia_content = Vec::new();
        mdia_content.extend_from_slice(&mdhd);
        mdia_content.extend_from_slice(&hdlr);
        mdia_content.extend_from_slice(&minf);
        let mdia = make_box(b"mdia", &mdia_content);

        let mut trak_content = Vec::new();
        trak_content.extend_from_slice(&tkhd);
        trak_content.extend_from_slice(&mdia);
        let trak = make_box(b"trak", &trak_content);

        let mut mvhd_content = Vec::new();
        mvhd_content.extend_from_slice(&[0; 4]);
        mvhd_content.extend_from_slice(&[0; 4]);
        mvhd_content.extend_from_slice(&1000u32.to_be_bytes());
        mvhd_content.extend_from_slice(&3000u32.to_be_bytes());
        mvhd_content.extend_from_slice(&0x00010000u32.to_be_bytes());
        mvhd_content.extend_from_slice(&0x0100u16.to_be_bytes());
        mvhd_content.extend_from_slice(&[0; 10]);
        mvhd_content.extend_from_slice(&[
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
        ]);
        mvhd_content.extend_from_slice(&[0; 24]);
        mvhd_content.extend_from_slice(&2u32.to_be_bytes());
        let mvhd = make_fullbox(b"mvhd", 0, 0, &mvhd_content);

        let mut moov_content = Vec::new();
        moov_content.extend_from_slice(&mvhd);
        moov_content.extend_from_slice(&trak);
        let moov = make_box(b"moov", &moov_content);

        mp4.extend_from_slice(&moov);

        let mut mdat_content = Vec::new();
        mdat_content.extend_from_slice(&sample1);
        mdat_content.extend_from_slice(&sample2);
        mdat_content.extend_from_slice(&sample3);
        let mdat = make_box(b"mdat", &mdat_content);

        let mdat_data_offset = (ftyp.len() + moov.len() + 8) as u32;
        mp4.extend_from_slice(&mdat);

        // Patch stco offset
        let stco_needle = b"stco";
        for i in 0..mp4.len() - 4 {
            if &mp4[i..i + 4] == stco_needle {
                let offset_pos = i + 4 + 4 + 4;
                mp4[offset_pos..offset_pos + 4].copy_from_slice(&mdat_data_offset.to_be_bytes());
                break;
            }
        }

        mp4
    }

    #[test]
    fn test_round_trip_no_modifications() {
        let original = build_test_mp4();
        let mp4 = demux(&original).unwrap();

        // Mux with no modifications
        let output = mux(&original, &mp4, &[]).unwrap();

        // Re-demux and verify structure
        let mp4_out = demux(&output).unwrap();
        let idx = mp4_out.video_track_idx.unwrap();
        let track = &mp4_out.tracks[idx];

        assert_eq!(track.samples.len(), 3);
        assert_eq!(track.samples[0].size, 8);
        assert_eq!(track.samples[1].size, 8);
        assert_eq!(track.samples[2].size, 8);
        assert!(track.samples[0].is_sync);
        assert!(!track.samples[1].is_sync);

        // Sample data should match original
        let orig_track = &mp4.tracks[mp4.video_track_idx.unwrap()];
        for i in 0..3 {
            assert_eq!(track.samples[i].data, orig_track.samples[i].data);
        }
    }

    #[test]
    fn test_mux_with_same_size_modification() {
        let original = build_test_mp4();
        let mp4 = demux(&original).unwrap();

        // Modify sample 0 with same-size data
        let new_data = vec![0x00, 0x00, 0x00, 0x04, 0xFF, 0xEE, 0xDD, 0xCC];
        let output = mux(&original, &mp4, &[(0, new_data.clone())]).unwrap();

        // Re-demux and check
        let mp4_out = demux(&output).unwrap();
        let track = &mp4_out.tracks[mp4_out.video_track_idx.unwrap()];

        assert_eq!(track.samples[0].data, new_data);
        assert_eq!(track.samples[0].size, 8);
        // Other samples unchanged
        let orig_track = &mp4.tracks[mp4.video_track_idx.unwrap()];
        assert_eq!(track.samples[1].data, orig_track.samples[1].data);
        assert_eq!(track.samples[2].data, orig_track.samples[2].data);
    }

    #[test]
    fn test_mux_with_different_size_modification() {
        let original = build_test_mp4();
        let mp4 = demux(&original).unwrap();

        // Modify sample 1 with larger data (10 bytes instead of 8)
        let new_data = vec![0x00, 0x00, 0x00, 0x06, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        let output = mux(&original, &mp4, &[(1, new_data.clone())]).unwrap();

        let mp4_out = demux(&output).unwrap();
        let track = &mp4_out.tracks[mp4_out.video_track_idx.unwrap()];

        assert_eq!(track.samples[1].data, new_data);
        assert_eq!(track.samples[1].size, 10);
        // stsz was updated
        assert_eq!(track.samples[0].size, 8);
        assert_eq!(track.samples[2].size, 8);

        // Other samples' data still correct
        let orig_track = &mp4.tracks[mp4.video_track_idx.unwrap()];
        assert_eq!(track.samples[0].data, orig_track.samples[0].data);
        assert_eq!(track.samples[2].data, orig_track.samples[2].data);
    }

    #[test]
    fn test_compute_chunk_first_samples() {
        // Test via round-trip: build_test_mp4 has 1 chunk with 3 samples
        let original = build_test_mp4();
        let mp4 = demux(&original).unwrap();

        let (moov_start, moov_size) = find_top_level_box(&original, b"moov").unwrap();
        let moov_data = &original[moov_start..moov_start + moov_size];

        let track = &mp4.tracks[mp4.video_track_idx.unwrap()];
        let chunks = compute_chunk_first_samples(moov_data, track).unwrap();

        // Our test MP4 has 1 chunk with 3 samples
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], (0, 3));
    }
}
