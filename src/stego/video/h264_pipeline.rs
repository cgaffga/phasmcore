// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 CAVLC video Ghost encode/decode pipeline.
//!
//! Embeds encrypted messages by flipping trailing-one sign bits in CAVLC-coded
//! H.264 Baseline video. Zero bitrate change — the output MP4 is byte-identical
//! to the input except for the flipped bits.
//!
//! Pipeline:
//! 1. Demux MP4, parse avcC → SPS/PPS
//! 2. For each frame: parse CAVLC macroblocks, collect embeddable T1 sign positions
//! 3. Compute CSF costs, permute positions, STC embed
//! 4. Apply bit flips to the original byte stream
//! 5. Output: copy-and-patch (original bytes + flipped bits)

use crate::codec::h264::bitstream::{self};
use crate::codec::h264::cavlc::{EmbedDomain, EmbeddablePosition};
use crate::codec::h264::macroblock::{self, Macroblock, NeighborContext};
use crate::codec::h264::slice::{self, SliceType};
use crate::codec::h264::sps::{self, Pps, Sps};
use crate::codec::h264::NalType;
use crate::codec::mp4;
use crate::det_math::det_powi_f64;
use crate::stego::cost::{h264_cost, h264_uniward};
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload;
use crate::stego::permute::{self, CoeffPos};
use crate::stego::stc::{embed as stc_embed_mod, extract as stc_extract_mod, hhat};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// STC constraint height (same as image Ghost).
const STC_H: usize = 7;

/// Default max in-flight GOPs for the Phase I.1 GOP-parallel encoder.
///
/// Per-GOP scan transient is ~150 MB at 1080p / 30-frame GOP; the
/// product N_threads × per_gop_mb sets peak working set. Default 4
/// targets a 600 MB working set — fine on desktop and 8 GB iPhone Pro
/// devices, but tight on 4 GB phones (~300 MB jetsam ceiling). Mobile
/// bridges should override via the `PHASM_H264_PARALLEL_GOPS` env var
/// (recommended: 2 on 4 GB phones, 8-12 on desktop).
///
/// Compiles to a no-op (always 1) when the `parallel` Cargo feature is
/// disabled — WASM and feature-off CLI builds.
#[allow(dead_code)] // Used only under #[cfg(feature = "parallel")] in max_parallel_gops().
const MAX_PARALLEL_GOPS_DEFAULT: usize = 4;

/// Resolve the runtime cap on parallel GOP encodes.
fn max_parallel_gops() -> usize {
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
    #[cfg(feature = "parallel")]
    {
        std::env::var("PHASM_H264_PARALLEL_GOPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(MAX_PARALLEL_GOPS_DEFAULT)
    }
}

/// One bit-flip to apply to the output mp4 bytes. Computed in the
/// parallel encode phase, applied serially after the chunk join (with
/// EP-byte safety check at apply time).
#[derive(Debug, Clone, Copy)]
struct FlipOp {
    abs_offset: usize,
    mask: u8,
}

/// Phase 3c: built per-domain STC inputs (permuted cover bits, costs, and a
/// reverse index from permuted-position to the original `all_positions`
/// index).
struct DomainStcBuild {
    cover_bits: Vec<u8>,
    stc_costs: Vec<f32>,
    permuted_to_orig: Vec<usize>,
}

/// Build STC inputs (cover bits, costs, permutation mapping) for a single
/// embedding domain. Used by both the encode and decode paths so the two
/// sides agree on position ordering.
fn build_domain_stc(
    usable: &[(usize, &EmbeddablePosition)],
    all_positions: &[EmbeddablePosition],
    costs: &[f32],
    perm_seed: &[u8; 32],
    track: &mp4::Track,
    mp4_bytes: &[u8],
) -> DomainStcBuild {
    let mut coeff_positions: Vec<(usize, CoeffPos)> = Vec::with_capacity(usable.len());
    for &(orig_idx, _pos) in usable {
        let cost = costs[orig_idx];
        if cost.is_finite() {
            coeff_positions.push((
                orig_idx,
                CoeffPos {
                    flat_idx: coeff_positions.len() as u32,
                    cost,
                },
            ));
        }
    }

    let mut stc_positions: Vec<CoeffPos> =
        coeff_positions.iter().map(|(_, cp)| cp.clone()).collect();
    permute::permute_positions(&mut stc_positions, perm_seed);

    let mut cover_bits = Vec::with_capacity(stc_positions.len());
    let mut stc_costs = Vec::with_capacity(stc_positions.len());
    let mut permuted_to_orig: Vec<usize> = Vec::with_capacity(stc_positions.len());

    for spos in &stc_positions {
        let (orig_idx, _) = coeff_positions[spos.flat_idx as usize];
        let epos = &all_positions[orig_idx];
        let sample = &track.samples[epos.frame_idx as usize];
        let abs_offset = sample.offset as usize + epos.raw_byte_offset;
        if abs_offset < mp4_bytes.len() {
            let current_bit = (mp4_bytes[abs_offset] >> (7 - epos.bit_offset)) & 1;
            cover_bits.push(current_bit);
        } else {
            cover_bits.push(0);
        }
        stc_costs.push(spos.cost);
        permuted_to_orig.push(orig_idx);
    }

    DomainStcBuild {
        cover_bits,
        stc_costs,
        permuted_to_orig,
    }
}

/// Embed a message into an H.264 CAVLC MP4 file.
///
/// Returns the stego MP4 bytes (same size as input — zero bitrate change).
/// Allocates a single `Vec<u8>` the size of the input; for large videos on
/// memory-constrained devices, use [`h264_ghost_encode_inplace`] against an
/// mmap or [`h264_ghost_encode_path`] which does the mmap for you.
pub fn h264_ghost_encode(
    mp4_bytes: &[u8],
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    let mut output = mp4_bytes.to_vec();
    h264_ghost_encode_inplace(&mut output, message, passphrase)?;
    Ok(output)
}

/// Streaming variant: embed the message by modifying `mp4_bytes` in place.
/// The caller is responsible for providing a mutable buffer containing the
/// cover MP4 — typically a `MmapMut` over the output file (see
/// [`h264_ghost_encode_path`]). Peak heap stays sub-linear in file size
/// (only position lists + parser state, not the video bytes).
pub fn h264_ghost_encode_inplace(
    mp4_bytes: &mut [u8],
    message: &str,
    passphrase: &str,
) -> Result<(), StegoError> {
    // 1. Demux and validate (owned `Mp4File`; no borrow into `mp4_bytes`).
    let mp4_file = mp4::demux::demux(&*mp4_bytes)?;
    let (sps, pps, length_size) = extract_h264_params(&mp4_file)?;

    if pps.entropy_coding_mode_flag {
        return Err(StegoError::InvalidVideo(
            "H.264 CABAC not supported; input must be Baseline CAVLC".into(),
        ));
    }

    let track_idx = mp4_file
        .video_track_idx
        .ok_or(StegoError::InvalidVideo("no video track".into()))?;
    let track = &mp4_file.tracks[track_idx];

    // 2. Phase 1 — capacity scan (per-GOP, lightweight on metadata).
    //
    // Phase I.0.5/68: per-GOP streaming. The whole-clip scan-and-accumulate
    // pattern OOMs on long video (~500 GB metadata for a 90-min 1080p movie).
    // Per-GOP streaming caps peak working set at ~150 MB on 1080p / 30-frame
    // GOP videos. Multi-GOP message-spread is the #68 enhancement: capacity
    // scales with video length, and the message spreads across all GOPs for
    // stealth. See docs/design/video/h264/encoder-memory-analysis.md.
    let capacities = scan_gop_capacities(track, &sps, &pps, length_size, &*mp4_bytes)?;
    if capacities.is_empty() {
        return Err(StegoError::InvalidVideo(
            "no GOPs found in video track".into(),
        ));
    }
    let total_n_coeff: usize = capacities.iter().map(|c| c.n_coeff).sum();
    let total_n_mvd: usize = capacities.iter().map(|c| c.n_mvd).sum();
    if total_n_coeff == 0 && total_n_mvd == 0 {
        return Err(StegoError::InvalidVideo(
            "no embeddable positions found in video".into(),
        ));
    }

    // 3. Prepare payload (encrypted + framed).
    let payload_bytes = payload::encode_payload(message, &[])?;
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
    let frame_bits: Vec<u8> = frame_bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect();
    let m_total = frame_bits.len();

    if m_total > total_n_coeff + total_n_mvd {
        return Err(StegoError::MessageTooLarge);
    }

    // 4. Compute per-GOP message bit allocation. Both encoder and decoder
    //    run `compute_per_gop_message_split(capacities, m_total, …)` with
    //    the same inputs so they agree on the split.
    let (gop_m_coeff, gop_m_mvd) =
        compute_per_gop_message_split(&capacities, m_total, total_n_coeff, total_n_mvd);

    // 5. Pick the largest shared `w` such that every GOP fits its allocated
    //    bits. Larger `w` = better stealth (fewer flips); smallest needed
    //    `w` is 1. We try from 10 down to 1 and pick the highest that fits
    //    all GOPs. Decoder mirrors: try 1..=10 ascending and accept the
    //    first that decodes (encoder picks the highest, so by symmetry the
    //    decoder finds the right one).
    let mut w_chosen: Option<usize> = None;
    for w_try in (1..=10usize).rev() {
        let mut ok = true;
        for (i, cap) in capacities.iter().enumerate() {
            if gop_m_coeff[i] * w_try > cap.n_coeff || gop_m_mvd[i] * w_try > cap.n_mvd {
                ok = false;
                break;
            }
        }
        if ok {
            w_chosen = Some(w_try);
            break;
        }
    }
    let w = w_chosen.ok_or(StegoError::MessageTooLarge)?;

    // 6. Master Argon2 keys — derived once per encode. Per-GOP keys mixed
    //    cheaply via SHA-256 inside the loop below.
    let coeff_master = crypto::derive_structural_key(passphrase)?;
    let coeff_master_perm: [u8; 32] = coeff_master[..32].try_into().unwrap();
    let coeff_master_hhat: [u8; 32] = coeff_master[32..].try_into().unwrap();
    let mvd_master = crypto::derive_h264_mvd_structural_key(passphrase)?;
    let mvd_master_perm: [u8; 32] = mvd_master[..32].try_into().unwrap();
    let mvd_master_hhat: [u8; 32] = mvd_master[32..].try_into().unwrap();

    // Pre-compute per-GOP frame_bits offsets so each GOP knows its slice
    // independently of the others (decouples sequential dependency for MT).
    let mut gop_frame_offsets: Vec<usize> = Vec::with_capacity(capacities.len());
    {
        let mut acc = 0usize;
        for i in 0..capacities.len() {
            gop_frame_offsets.push(acc);
            acc += gop_m_coeff[i] + gop_m_mvd[i];
        }
    }

    // 7. Phase 2 — per-GOP scan + STC, batched across `max_parallel_gops`
    //    workers. Each worker COLLECTS its flip ops into a Vec<FlipOp>
    //    rather than mutating mp4_bytes directly, so multiple workers can
    //    read the (immutable) mp4_bytes in parallel without aliasing on
    //    the mutable side. After each chunk, the main thread serially
    //    applies the collected flips (with EP-byte safety check).
    //
    //    Per-batch peak working set: max_parallel_gops × ~150 MB scan
    //    transient. Default 4 = 600 MB on 1080p, fine on desktop, tight
    //    on 4 GB phones — `PHASM_H264_PARALLEL_GOPS=2` env override
    //    recommended for low-memory devices.
    let max_parallel = max_parallel_gops();
    let mut num_flips = 0usize;
    let mut num_ep_skipped = 0usize;

    for chunk in capacities.chunks(max_parallel) {
        // Re-borrow mp4_bytes as &[u8] for the parallel scan section.
        // The reborrow ends at the end of this block; mp4_bytes returns
        // to mutable status for the apply phase below.
        let mp4_immut: &[u8] = &*mp4_bytes;

        // Process each GOP in the chunk: scan + STC + collect FlipOps.
        // Returns Result<Vec<FlipOp>> per GOP; first error short-circuits.
        let chunk_offset = chunk.as_ptr() as usize - capacities.as_ptr() as usize;
        let chunk_start_idx = chunk_offset / std::mem::size_of::<GopCapacity>();

        let work = |idx_in_chunk: usize, cap: &GopCapacity| -> Result<Vec<FlipOp>, StegoError> {
            let i = chunk_start_idx + idx_in_chunk;
            compute_gop_flips(
                track,
                &sps,
                &pps,
                length_size,
                mp4_immut,
                cap,
                gop_frame_offsets[i],
                gop_m_coeff[i],
                gop_m_mvd[i],
                w,
                &frame_bits,
                &coeff_master_perm,
                &coeff_master_hhat,
                &mvd_master_perm,
                &mvd_master_hhat,
            )
        };

        #[cfg(feature = "parallel")]
        let chunk_flips: Result<Vec<Vec<FlipOp>>, StegoError> = chunk
            .par_iter()
            .enumerate()
            .map(|(idx_in_chunk, cap)| work(idx_in_chunk, cap))
            .collect();
        #[cfg(not(feature = "parallel"))]
        let chunk_flips: Result<Vec<Vec<FlipOp>>, StegoError> = chunk
            .iter()
            .enumerate()
            .map(|(idx_in_chunk, cap)| work(idx_in_chunk, cap))
            .collect();

        // Reborrow ends; we can use mp4_bytes mutably again.
        let chunk_flips = chunk_flips?;

        // 7c. Apply all chunk flips serially with EP-byte safety check.
        //     EP check reads adjacent bytes in mp4_bytes — must be
        //     serial w.r.t. flips, since adjacent bytes may have been
        //     just-flipped by a prior op in the same chunk.
        for flips in chunk_flips {
            for op in flips {
                if op.abs_offset >= mp4_bytes.len() {
                    continue;
                }
                if would_change_ep_pattern(mp4_bytes, op.abs_offset, op.mask) {
                    num_ep_skipped += 1;
                    continue;
                }
                mp4_bytes[op.abs_offset] ^= op.mask;
                num_flips += 1;
            }
        }
        // chunk's per-GOP scan transients have all been dropped; ready for
        // the next chunk.
    }

    // If we skipped any flips, the STC syndrome is broken by that many bits —
    // decode will fail authentication. Surface this as an explicit error rather
    // than producing a silently un-decodeable stego file. Empirically this has
    // never fired on our test vectors.
    if num_ep_skipped > 0 {
        return Err(StegoError::InvalidVideo(format!(
            "H.264 encode aborted: {num_ep_skipped} flip(s) would have created \
             or destroyed an emulation-prevention byte sequence. This indicates \
             coefficient data landed next to 0x00 0x00 bytes in the bitstream; \
             re-encoding the cover at a different QP usually fixes it."
        )));
    }

    let _ = num_flips; // diagnostic only — caller doesn't consume it

    Ok(())
}

// ── Streaming path-based wrappers ─────────────────────────────────────────
//
// These mmap the input / output files so the Rust heap never holds the
// whole video. The mobile bridges and the CLI use these instead of the
// byte-based variants when they have a file path handy.

/// Embed a message into the H.264 CAVLC MP4 at `input_path`, writing the
/// stego to `output_path`. Streaming: `input_path` is mmap'd read-only for
/// parsing, then the file at `output_path` is opened read+write and
/// modified in place. Peak heap stays sub-linear in file size.
///
/// If `input_path == output_path`, the file is edited in place (no copy).
pub fn h264_ghost_encode_path(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
    message: &str,
    passphrase: &str,
) -> Result<(), StegoError> {
    if input_path != output_path {
        std::fs::copy(input_path, output_path)
            .map_err(|e| StegoError::InvalidVideo(format!("copy failed: {e}")))?;
    }
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(output_path)
        .map_err(|e| StegoError::InvalidVideo(format!("open output failed: {e}")))?;
    let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file) }
        .map_err(|e| StegoError::InvalidVideo(format!("mmap failed: {e}")))?;
    h264_ghost_encode_inplace(&mut mmap, message, passphrase)?;
    mmap.flush()
        .map_err(|e| StegoError::InvalidVideo(format!("mmap flush failed: {e}")))?;
    Ok(())
}

/// Decode a message from a stego H.264 MP4 at `path`. Streaming: the file
/// is mmap'd read-only.
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

/// Estimate embedding capacity for the H.264 MP4 at `path` (streaming).
pub fn h264_ghost_capacity_path(path: &std::path::Path) -> Result<usize, StegoError> {
    let file = std::fs::File::open(path)
        .map_err(|e| StegoError::InvalidVideo(format!("open failed: {e}")))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| StegoError::InvalidVideo(format!("mmap failed: {e}")))?;
    h264_ghost_capacity(&mmap)
}

/// Pack 16 MSB-first bits into a big-endian u16. Used at decode to read the
/// frame's length prefix out of the extracted message bit stream without
/// having to build a full byte vec first.
fn read_u16_be_from_bits(bits: &[u8]) -> u16 {
    debug_assert!(bits.len() >= 16);
    let mut v = 0u16;
    for i in 0..16 {
        v = (v << 1) | (bits[i] & 1) as u16;
    }
    v
}

/// Pack 32 MSB-first bits into a big-endian u32 (v2 frame's extended length).
fn read_u32_be_from_bits(bits: &[u8]) -> u32 {
    debug_assert!(bits.len() >= 32);
    let mut v = 0u32;
    for i in 0..32 {
        v = (v << 1) | (bits[i] & 1) as u32;
    }
    v
}

/// Apply one domain's STC embed result to `output`: for every permuted
/// position where `stego_bits` differs from `cover_bits`, flip the bit in
/// `output` (guarded against emulation-prevention-byte corruption). Shared
/// by the coefficient and MVD domain passes at encode.
#[allow(dead_code)] // Pre-§30D-C primitive; superseded by inline injection hooks.
fn apply_domain_flips(
    output: &mut [u8],
    stego_bits: &[u8],
    build: &DomainStcBuild,
    all_positions: &[EmbeddablePosition],
    track: &mp4::Track,
    num_flips: &mut usize,
    num_ep_skipped: &mut usize,
) {
    for (perm_idx, &stego_bit) in stego_bits.iter().enumerate() {
        if stego_bit == build.cover_bits[perm_idx] {
            continue;
        }
        let orig_idx = build.permuted_to_orig[perm_idx];
        let epos = &all_positions[orig_idx];
        let sample = &track.samples[epos.frame_idx as usize];
        let abs_offset = sample.offset as usize + epos.raw_byte_offset;
        if abs_offset >= output.len() {
            continue;
        }
        let mask = 1u8 << (7 - epos.bit_offset);
        if would_change_ep_pattern(output, abs_offset, mask) {
            *num_ep_skipped += 1;
            continue;
        }
        output[abs_offset] ^= mask;
        *num_flips += 1;
    }
}

/// Check whether flipping the bits in `mask` at `output[byte_offset]` would
/// add or remove a 3-byte sequence `00 00 0x` (x in 0..=3) anywhere in the
/// 5-byte window covering that byte. Used as a last-chance safety net at flip
/// application time to prevent silent stream corruption.
#[inline]
fn would_change_ep_pattern(output: &[u8], byte_offset: usize, mask: u8) -> bool {
    let new_byte = output[byte_offset] ^ mask;
    // Three 3-byte windows contain byte_offset: windows starting at
    // byte_offset-2, byte_offset-1, and byte_offset (when in range).
    for offset in 0..3usize {
        let Some(start) = byte_offset.checked_sub(offset) else { continue };
        if start + 3 > output.len() {
            continue;
        }
        let b0 = if start == byte_offset { new_byte } else { output[start] };
        let b1 = if start + 1 == byte_offset { new_byte } else { output[start + 1] };
        let b2 = if start + 2 == byte_offset { new_byte } else { output[start + 2] };

        let orig_is_ep = output[start] == 0 && output[start + 1] == 0 && output[start + 2] <= 3;
        let new_is_ep = b0 == 0 && b1 == 0 && b2 <= 3;
        if orig_is_ep != new_is_ep {
            return true;
        }
    }
    false
}

/// Decode a message from an H.264 CAVLC stego MP4 file.
///
/// Returns `PayloadData` (text + any file attachments) for parity with
/// `video_ghost_decode`. Use `.text` for plain-message use cases.
pub fn h264_ghost_decode(
    mp4_bytes: &[u8],
    passphrase: &str,
) -> Result<crate::stego::payload::PayloadData, StegoError> {
    // 1. Demux and validate
    let mp4_file = mp4::demux::demux(mp4_bytes)?;
    let (sps, pps, length_size) = extract_h264_params(&mp4_file)?;

    if pps.entropy_coding_mode_flag {
        // DECODE-FAST (2026-05-24) — try the modern StreamingDecodeSession
        // first. It does per-GOP brute-force, which is ~order-of-magnitude
        // cheaper than the legacy whole-video search in
        // `decode_cabac_via_chunk_6g` on full-length 1080p clips. CLI's
        // `decode_h264_cabac` already does this; mobile bridges (iOS +
        // Android) reach this function via `h264_ghost_decode_path` and
        // were stuck on the legacy path before this change.
        #[cfg(feature = "cabac-stego")]
        {
            if let Some(payload) =
                try_streaming_decode_mp4(&mp4_file, length_size, passphrase)
            {
                return Ok(payload);
            }
            return decode_cabac_via_chunk_6g(&mp4_file, length_size, passphrase);
        }
        #[cfg(not(feature = "cabac-stego"))]
        return Err(StegoError::InvalidVideo(
            "H.264 CABAC not supported for decode \
             (build with --features cabac-stego to enable)".into(),
        ));
    }

    let track_idx = mp4_file
        .video_track_idx
        .ok_or(StegoError::InvalidVideo("no video track".into()))?;
    let track = &mp4_file.tracks[track_idx];

    // 2. Phase 1 — capacity scan (per-GOP).
    let capacities = scan_gop_capacities(track, &sps, &pps, length_size, mp4_bytes)?;
    if capacities.is_empty() {
        return Err(StegoError::InvalidVideo("no GOPs found".into()));
    }
    let total_n_coeff: usize = capacities.iter().map(|c| c.n_coeff).sum();
    let total_n_mvd: usize = capacities.iter().map(|c| c.n_mvd).sum();
    let total_n = total_n_coeff + total_n_mvd;
    if total_n == 0 {
        return Err(StegoError::InvalidVideo("no embeddable positions".into()));
    }

    // 3. Master keys (shared with encode-side).
    let coeff_master = crypto::derive_structural_key(passphrase)?;
    let coeff_master_perm: [u8; 32] = coeff_master[..32].try_into().unwrap();
    let coeff_master_hhat: [u8; 32] = coeff_master[32..].try_into().unwrap();
    let mvd_master = crypto::derive_h264_mvd_structural_key(passphrase)?;
    let mvd_master_perm: [u8; 32] = mvd_master[..32].try_into().unwrap();
    let mvd_master_hhat: [u8; 32] = mvd_master[32..].try_into().unwrap();

    // 4. Phase 2a — scan GOP 0 once, cache its STC builds for cheap w-loop.
    //
    // #69 design: instead of rescanning every GOP per w iteration (~10×N
    // total work), we cache GOP 0's covers once (~10 KB), iterate w cheaply
    // on those cached covers, and only stream subsequent GOPs once we find
    // the right w. Average decode work drops from ~11N to ~2N scans.
    //
    // Memory bound: Phase 2a leaves the GOP 0 build alive for the duration
    // of decode (~1 MB at 1080p). Phase 2c per-GOP transient (~150 MB) only
    // exists for one GOP at a time, dropped between iterations.
    let gop0_builds = scan_and_build_gop_stc(
        track,
        &sps,
        &pps,
        length_size,
        mp4_bytes,
        &capacities[0],
        &coeff_master_perm,
        &coeff_master_hhat,
        &mvd_master_perm,
        &mvd_master_hhat,
    )?;

    // 5. Phase 2b — try each w ∈ [10..1] descending. For each, do a cheap
    //    extract from the cached GOP 0 build, validate, then Phase 2c
    //    streams remaining GOPs.
    'w_loop: for w in (1..=10usize).rev() {
        let n_c0 = gop0_builds.coeff.cover_bits.len();
        let n_m0 = gop0_builds.mvd.cover_bits.len();
        let m_c0_max = n_c0 / w;
        let m_m0_max = n_m0 / w;
        if m_c0_max == 0 && m_m0_max == 0 {
            continue 'w_loop;
        }

        let coeff_extract_0 = if m_c0_max > 0 {
            let hhat_c = hhat::generate_hhat(STC_H, w, &gop0_builds.coeff_hhat_seed);
            stc_extract_mod::stc_extract(
                &gop0_builds.coeff.cover_bits[..m_c0_max * w],
                &hhat_c,
                w,
            )
        } else {
            Vec::new()
        };
        let mvd_extract_0 = if m_m0_max > 0 {
            let hhat_m = hhat::generate_hhat(STC_H, w, &gop0_builds.mvd_hhat_seed);
            stc_extract_mod::stc_extract(
                &gop0_builds.mvd.cover_bits[..m_m0_max * w],
                &hhat_m,
                w,
            )
        } else {
            Vec::new()
        };

        // 5a. Peek frame length from GOP 0's coefficient extract head.
        if coeff_extract_0.len() < 16 {
            continue 'w_loop;
        }
        let length_prefix = read_u16_be_from_bits(&coeff_extract_0[0..16]);
        let (total_payload_bytes, frame_overhead) = if length_prefix != 0 {
            (length_prefix as usize, frame::FRAME_OVERHEAD)
        } else {
            if coeff_extract_0.len() < 48 {
                continue 'w_loop;
            }
            let ext_len = read_u32_be_from_bits(&coeff_extract_0[16..48]);
            (ext_len as usize, frame::FRAME_OVERHEAD_EXT)
        };
        let m_total = (frame_overhead + total_payload_bytes) * 8;
        // Plausibility filter: a wrong w gives random bits → m_total often
        // exceeds the total cover capacity. Bail without streaming the
        // remaining GOPs in that case.
        if m_total == 0 || m_total > total_n {
            continue 'w_loop;
        }

        // 5b. Reconstruct the per-GOP split (same formula encoder used).
        let (gop_m_c, gop_m_m) =
            compute_per_gop_message_split(&capacities, m_total, total_n_coeff, total_n_mvd);

        // 5c. Validate this w fits the implied split for ALL GOPs, given the
        //     Pass-1 capacity counts.
        let mut fits = true;
        for (i, cap) in capacities.iter().enumerate() {
            if gop_m_c[i] * w > cap.n_coeff || gop_m_m[i] * w > cap.n_mvd {
                fits = false;
                break;
            }
        }
        if !fits {
            continue 'w_loop;
        }
        if gop_m_c[0] > coeff_extract_0.len() || gop_m_m[0] > mvd_extract_0.len() {
            continue 'w_loop;
        }

        // 5d. Phase 2c — stream remaining GOPs at this w. Each GOP's
        //     metadata transient (~150 MB) drops before the next.
        let mut message_bits = Vec::with_capacity(m_total);
        message_bits.extend_from_slice(&coeff_extract_0[..gop_m_c[0]]);
        message_bits.extend_from_slice(&mvd_extract_0[..gop_m_m[0]]);

        let mut gop_failed = false;
        for (i, cap) in capacities.iter().enumerate().skip(1) {
            if gop_m_c[i] == 0 && gop_m_m[i] == 0 {
                continue;
            }
            let builds = match scan_and_build_gop_stc(
                track,
                &sps,
                &pps,
                length_size,
                mp4_bytes,
                cap,
                &coeff_master_perm,
                &coeff_master_hhat,
                &mvd_master_perm,
                &mvd_master_hhat,
            ) {
                Ok(b) => b,
                Err(_) => {
                    gop_failed = true;
                    break;
                }
            };
            let n_c = builds.coeff.cover_bits.len();
            let n_m = builds.mvd.cover_bits.len();
            if gop_m_c[i] * w > n_c || gop_m_m[i] * w > n_m {
                // Pass-2 cover smaller than Pass-1 capacity — shouldn't
                // happen since both apply the same filter chain. Bail.
                gop_failed = true;
                break;
            }
            let m_c_max = n_c / w;
            let m_m_max = n_m / w;
            let coeff_extract = if m_c_max > 0 {
                let hhat_c = hhat::generate_hhat(STC_H, w, &builds.coeff_hhat_seed);
                stc_extract_mod::stc_extract(&builds.coeff.cover_bits[..m_c_max * w], &hhat_c, w)
            } else {
                Vec::new()
            };
            let mvd_extract = if m_m_max > 0 {
                let hhat_m = hhat::generate_hhat(STC_H, w, &builds.mvd_hhat_seed);
                stc_extract_mod::stc_extract(&builds.mvd.cover_bits[..m_m_max * w], &hhat_m, w)
            } else {
                Vec::new()
            };
            if gop_m_c[i] > coeff_extract.len() || gop_m_m[i] > mvd_extract.len() {
                gop_failed = true;
                break;
            }
            message_bits.extend_from_slice(&coeff_extract[..gop_m_c[i]]);
            message_bits.extend_from_slice(&mvd_extract[..gop_m_m[i]]);
            // builds drops here, freeing this GOP's ~150 MB transient.
        }
        if gop_failed {
            continue 'w_loop;
        }

        // 5e. Parse + decrypt + decode payload.
        let message_bytes: Vec<u8> = message_bits
            .chunks(8)
            .map(|chunk| {
                let mut byte = 0u8;
                for (i, &bit) in chunk.iter().enumerate() {
                    byte |= bit << (7 - i);
                }
                byte
            })
            .collect();

        if let Ok(parsed) = frame::parse_frame(&message_bytes)
            && let Ok(plaintext) =
                crypto::decrypt(&parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce)
            && let Ok(payload_data) = payload::decode_payload(&plaintext)
        {
            return Ok(payload_data);
        }
    }

    Err(StegoError::DecryptionFailed)
}

/// Compute the STC flip operations for a single GOP without mutating
/// `mp4_bytes`. Used by the GOP-parallel encode loop (Phase I.1) — the
/// outer loop collects per-GOP `Vec<FlipOp>` from N workers in parallel,
/// then serially applies them after the chunk join.
///
/// All the same per-GOP work as the previous serial path
/// (`scan_frame_range` -> `compute_all_costs` -> filter usable
/// -> per-GOP keys -> `build_domain_stc` -> `stc_embed`) but the final
/// step COLLECTS flips into a `Vec<FlipOp>` instead of XOR'ing
/// `mp4_bytes` directly. The EP-byte safety check is deferred to the
/// serial apply phase.
#[allow(clippy::too_many_arguments)]
fn compute_gop_flips(
    track: &mp4::Track,
    sps: &Sps,
    pps: &Pps,
    length_size: u8,
    mp4_bytes: &[u8],
    cap: &GopCapacity,
    frame_offset: usize,
    m_g_c: usize,
    m_g_m: usize,
    w: usize,
    frame_bits: &[u8],
    coeff_master_perm: &[u8; 32],
    coeff_master_hhat: &[u8; 32],
    mvd_master_perm: &[u8; 32],
    mvd_master_hhat: &[u8; 32],
) -> Result<Vec<FlipOp>, StegoError> {
    if m_g_c == 0 && m_g_m == 0 {
        return Ok(Vec::new());
    }

    let (positions, ac_energies, uniward_costs) = scan_frame_range(
        track,
        sps,
        pps,
        length_size,
        mp4_bytes,
        cap.range.first,
        cap.range.last,
    )?;
    let costs = compute_all_costs(&positions, &ac_energies, &uniward_costs, track);

    let usable: Vec<_> = positions
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            let is_coeff = matches!(
                p.domain,
                EmbedDomain::T1Sign | EmbedDomain::LevelSuffixMag | EmbedDomain::LevelSuffixSign
            );
            (is_coeff && p.scan_pos > 0) || p.domain == EmbedDomain::MvdLsb
        })
        .collect();
    let (coeff_usable, mvd_usable): (Vec<_>, Vec<_>) = usable
        .iter()
        .partition(|(_, p)| p.domain != EmbedDomain::MvdLsb);

    let gop_id = cap.range.gop_idx;
    let coeff_perm_seed =
        crypto::derive_per_gop_seed_from_master(coeff_master_perm, gop_id, b"coeff-perm");
    let coeff_hhat_seed =
        crypto::derive_per_gop_seed_from_master(coeff_master_hhat, gop_id, b"coeff-hhat");
    let mvd_perm_seed =
        crypto::derive_per_gop_seed_from_master(mvd_master_perm, gop_id, b"mvd-perm");
    let mvd_hhat_seed =
        crypto::derive_per_gop_seed_from_master(mvd_master_hhat, gop_id, b"mvd-hhat");

    let coeff_build = build_domain_stc(
        &coeff_usable,
        &positions,
        &costs,
        &coeff_perm_seed,
        track,
        mp4_bytes,
    );
    let mvd_build = build_domain_stc(
        &mvd_usable,
        &positions,
        &costs,
        &mvd_perm_seed,
        track,
        mp4_bytes,
    );

    let mut flips = Vec::new();

    if m_g_c > 0 {
        let n_used_c = m_g_c * w;
        if coeff_build.cover_bits.len() < n_used_c {
            return Err(StegoError::MessageTooLarge);
        }
        let hhat_c = hhat::generate_hhat(STC_H, w, &coeff_hhat_seed);
        let embed_c = stc_embed_mod::stc_embed(
            &coeff_build.cover_bits[..n_used_c],
            &coeff_build.stc_costs[..n_used_c],
            &frame_bits[frame_offset..frame_offset + m_g_c],
            &hhat_c,
            STC_H,
            w,
        )
        .ok_or(StegoError::MessageTooLarge)?;
        collect_domain_flips(&embed_c.stego_bits, &coeff_build, &positions, track, &mut flips);
    }

    if m_g_m > 0 {
        let n_used_m = m_g_m * w;
        if mvd_build.cover_bits.len() < n_used_m {
            return Err(StegoError::MessageTooLarge);
        }
        let hhat_m = hhat::generate_hhat(STC_H, w, &mvd_hhat_seed);
        let embed_m = stc_embed_mod::stc_embed(
            &mvd_build.cover_bits[..n_used_m],
            &mvd_build.stc_costs[..n_used_m],
            &frame_bits[frame_offset + m_g_c..frame_offset + m_g_c + m_g_m],
            &hhat_m,
            STC_H,
            w,
        )
        .ok_or(StegoError::MessageTooLarge)?;
        collect_domain_flips(&embed_m.stego_bits, &mvd_build, &positions, track, &mut flips);
    }

    Ok(flips)
}

/// Append flip ops for one domain's STC embed result to `flips`. Pure
/// — no mp4 mutation. Mirrors `apply_domain_flips` shape but pushes to
/// a Vec instead of XOR'ing. EP-byte safety check is deferred to the
/// serial apply phase in the encoder.
fn collect_domain_flips(
    stego_bits: &[u8],
    build: &DomainStcBuild,
    all_positions: &[EmbeddablePosition],
    track: &mp4::Track,
    flips: &mut Vec<FlipOp>,
) {
    for (perm_idx, &stego_bit) in stego_bits.iter().enumerate() {
        if stego_bit == build.cover_bits[perm_idx] {
            continue;
        }
        let orig_idx = build.permuted_to_orig[perm_idx];
        let epos = &all_positions[orig_idx];
        let sample = &track.samples[epos.frame_idx as usize];
        let abs_offset = sample.offset as usize + epos.raw_byte_offset;
        let mask = 1u8 << (7 - epos.bit_offset);
        flips.push(FlipOp { abs_offset, mask });
    }
}

/// All the per-GOP STC infrastructure needed to extract message bits.
/// Built once per GOP (decode side); cached for GOP 0 across all w iterations.
struct GopStcBuilds {
    coeff: DomainStcBuild,
    mvd: DomainStcBuild,
    coeff_hhat_seed: [u8; 32],
    mvd_hhat_seed: [u8; 32],
}

/// Scan one GOP and build its per-domain STC inputs. Used by both encode
/// (drops the result after STC + apply_flips) and decode (caches GOP 0,
/// streams the rest).
///
/// Memory: holds positions + ac_energies + uniward_costs + costs + both
/// `DomainStcBuild`s during the call. Peak ~150 MB at 1080p / 30-frame
/// GOP. The returned `GopStcBuilds` keeps only the two builds (~1 MB).
fn scan_and_build_gop_stc(
    track: &mp4::Track,
    sps: &Sps,
    pps: &Pps,
    length_size: u8,
    mp4_bytes: &[u8],
    cap: &GopCapacity,
    coeff_master_perm: &[u8; 32],
    coeff_master_hhat: &[u8; 32],
    mvd_master_perm: &[u8; 32],
    mvd_master_hhat: &[u8; 32],
) -> Result<GopStcBuilds, StegoError> {
    let (positions, ac_energies, uniward_costs) = scan_frame_range(
        track,
        sps,
        pps,
        length_size,
        mp4_bytes,
        cap.range.first,
        cap.range.last,
    )?;
    let costs = compute_all_costs(&positions, &ac_energies, &uniward_costs, track);

    let usable: Vec<_> = positions
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            let is_coeff = matches!(
                p.domain,
                EmbedDomain::T1Sign | EmbedDomain::LevelSuffixMag | EmbedDomain::LevelSuffixSign
            );
            (is_coeff && p.scan_pos > 0) || p.domain == EmbedDomain::MvdLsb
        })
        .collect();
    let (coeff_usable, mvd_usable): (Vec<_>, Vec<_>) = usable
        .iter()
        .partition(|(_, p)| p.domain != EmbedDomain::MvdLsb);

    let gop_id = cap.range.gop_idx;
    let coeff_perm_seed =
        crypto::derive_per_gop_seed_from_master(coeff_master_perm, gop_id, b"coeff-perm");
    let coeff_hhat_seed =
        crypto::derive_per_gop_seed_from_master(coeff_master_hhat, gop_id, b"coeff-hhat");
    let mvd_perm_seed =
        crypto::derive_per_gop_seed_from_master(mvd_master_perm, gop_id, b"mvd-perm");
    let mvd_hhat_seed =
        crypto::derive_per_gop_seed_from_master(mvd_master_hhat, gop_id, b"mvd-hhat");

    let coeff = build_domain_stc(
        &coeff_usable,
        &positions,
        &costs,
        &coeff_perm_seed,
        track,
        mp4_bytes,
    );
    let mvd = build_domain_stc(
        &mvd_usable,
        &positions,
        &costs,
        &mvd_perm_seed,
        track,
        mp4_bytes,
    );

    Ok(GopStcBuilds {
        coeff,
        mvd,
        coeff_hhat_seed,
        mvd_hhat_seed,
    })
    // positions / ac_energies / uniward_costs / costs / usable / *_usable
    // / *_perm_seed all drop here.
}

/// Estimate the embedding capacity of an H.264 CAVLC MP4 file.
///
/// Returns the maximum payload bytes that can be embedded.
pub fn h264_ghost_capacity(mp4_bytes: &[u8]) -> Result<usize, StegoError> {
    let mp4_file = mp4::demux::demux(mp4_bytes)?;
    let (sps, pps, length_size) = extract_h264_params(&mp4_file)?;

    if pps.entropy_coding_mode_flag {
        return Err(StegoError::InvalidVideo("CABAC not supported".into()));
    }

    let track_idx = mp4_file
        .video_track_idx
        .ok_or(StegoError::InvalidVideo("no video track".into()))?;
    let track = &mp4_file.tracks[track_idx];

    // Phase I.0.5/68: capacity sums over all GOPs (matching the multi-GOP
    // encode/decode that walks per-GOP).
    let capacities = scan_gop_capacities(track, &sps, &pps, length_size, mp4_bytes)?;
    let usable_count: usize = capacities
        .iter()
        .map(|c| c.n_coeff + c.n_mvd)
        .sum();

    // STC h=7: 1 message bit per `w` cover positions, `w ∈ [1, 10]`.
    // The multi-GOP encoder picks the largest `w` that fits all GOPs
    // (see compute_per_gop_message_split). Conservative default
    // assumes `w = 5` — true capacity is up to 2× this for short
    // messages on long videos (typical case picks w=10). Power users
    // can call `h264_ghost_capacity_max` for the optimistic bound.
    let message_bits = usable_count / 5;
    let payload_bytes = message_bits.saturating_sub(frame::FRAME_OVERHEAD * 8) / 8;
    Ok(payload_bytes)
}

/// Maximum embedding capacity: the largest message the encoder can
/// possibly fit, at STC `w = 1` (no stealth amplification, 1 message
/// bit per cover position). This is **5× the default
/// `h264_ghost_capacity`** which assumes the conservative `w = 5`.
///
/// STC relationship: each message bit consumes `w` cover positions.
/// Lower `w` = MORE message bits per cover (max capacity, min
/// stealth). Higher `w` = FEWER message bits per cover (max stealth,
/// fewer flips). The encoder picks the LARGEST `w ∈ [1, 10]` that
/// still fits the message — so a small message gets `w = 10` (max
/// stealth) while a max-sized message lands at `w = 1`. Capacity at
/// `w = 1` is therefore the upper bound on what the encoder can
/// embed at all.
///
/// Useful for power-user tooling (CLI) that wants to display the
/// "real" maximum rather than the UX-safe pessimistic default.
///
/// Mobile UX should keep using `h264_ghost_capacity` (the pessimistic
/// `usable / 5`) — under-promising and over-delivering is the safer
/// gate. If the user types a message between the two values the
/// encoder will succeed, just at a lower `w` (= less stealth).
pub fn h264_ghost_capacity_max(mp4_bytes: &[u8]) -> Result<usize, StegoError> {
    let mp4_file = mp4::demux::demux(mp4_bytes)?;
    let (sps, pps, length_size) = extract_h264_params(&mp4_file)?;

    if pps.entropy_coding_mode_flag {
        return Err(StegoError::InvalidVideo("CABAC not supported".into()));
    }

    let track_idx = mp4_file
        .video_track_idx
        .ok_or(StegoError::InvalidVideo("no video track".into()))?;
    let track = &mp4_file.tracks[track_idx];

    let capacities = scan_gop_capacities(track, &sps, &pps, length_size, mp4_bytes)?;
    let usable_count: usize = capacities
        .iter()
        .map(|c| c.n_coeff + c.n_mvd)
        .sum();

    // Max: w = 1 → 1 message bit per cover position.
    let message_bits = usable_count;
    let payload_bytes = message_bits.saturating_sub(frame::FRAME_OVERHEAD * 8) / 8;
    Ok(payload_bytes)
}

/// Path-based wrapper for `h264_ghost_capacity_max` — mirrors the
/// `h264_ghost_capacity_path` streaming API.
pub fn h264_ghost_capacity_max_path(path: &std::path::Path) -> Result<usize, StegoError> {
    let file = std::fs::File::open(path)
        .map_err(|e| StegoError::InvalidVideo(format!("open failed: {e}")))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| StegoError::InvalidVideo(format!("mmap failed: {e}")))?;
    h264_ghost_capacity_max(&mmap)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract SPS, PPS, and NAL length size from an H.264 MP4 file.
/// Phase 6D.8 chunk 7 — route CABAC MP4 inputs through the
/// chunk-6G decoder. Demux → extract SPS+PPS NALs from avcC →
/// gather per-sample length-prefixed NAL bytes → assemble a
/// flat `Vec<NalUnit>` → call `h264_stego_decode_nalus_string`
/// → wrap result as `PayloadData` for legacy-API parity.
///
/// Decoupled from the legacy CAVLC pipeline: the cabac-stego
/// route does not share scan/STC code with the bitstream-mod
/// pipeline. Two different stego modes living side-by-side.
/// DECODE-FAST (2026-05-24) — assemble an Annex-B bitstream from the
/// already-demuxed MP4 (avcC SPS+PPS + per-sample length-prefixed NALs
/// rewritten with `00 00 00 01` start codes) and try
/// `StreamingDecodeSession`. Returns `Some(payload)` on a clean
/// chunk_frame match, `None` if the file isn't streaming-session
/// output (e.g. pre-streaming OH264 stego, or pure-Rust experimental
/// path) so the caller can fall back to the legacy decoder.
#[cfg(feature = "cabac-stego")]
fn try_streaming_decode_mp4(
    mp4_file: &mp4::Mp4File,
    length_size: u8,
    passphrase: &str,
) -> Option<crate::stego::payload::PayloadData> {
    use crate::codec::h264::streaming_session::StreamingDecodeSession;

    let track_idx = mp4_file.video_track_idx?;
    let track = &mp4_file.tracks[track_idx];
    let avcc = track.avcc_data.as_ref()?;

    // `split_annex_b_into_gops` looks for SPS boundaries to find each
    // GOP. MP4 keeps SPS/PPS in avcC (not inline), so we prepend the
    // SPS+PPS pair before every IDR slice (nal_type 5) we emit. That
    // gives the streaming session the per-GOP slabs it expects.
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

#[cfg(feature = "cabac-stego")]
fn decode_cabac_via_chunk_6g(
    mp4_file: &mp4::Mp4File,
    length_size: u8,
    passphrase: &str,
) -> Result<crate::stego::payload::PayloadData, StegoError> {
    use crate::codec::h264::bitstream::{parse_nal_unit, parse_nal_units_mp4};
    use crate::codec::h264::stego::decode_pixels::
        h264_stego_decode_nalus_string;

    let track_idx = mp4_file
        .video_track_idx
        .ok_or(StegoError::InvalidVideo("no video track".into()))?;
    let track = &mp4_file.tracks[track_idx];
    let avcc = track.avcc_data.as_ref().ok_or(
        StegoError::InvalidVideo("no avcC configuration".into())
    )?;

    // Build a NalUnit list: SPS + PPS from avcC, then sample NALs.
    let mut nalus: Vec<crate::codec::h264::NalUnit> = Vec::new();
    for sps_bytes in &avcc.sps_nalus {
        if sps_bytes.is_empty() {
            continue;
        }
        let nu = parse_nal_unit(sps_bytes)
            .map_err(|e| StegoError::InvalidVideo(format!("avcC SPS: {e}")))?;
        nalus.push(nu);
    }
    for pps_bytes in &avcc.pps_nalus {
        if pps_bytes.is_empty() {
            continue;
        }
        let nu = parse_nal_unit(pps_bytes)
            .map_err(|e| StegoError::InvalidVideo(format!("avcC PPS: {e}")))?;
        nalus.push(nu);
    }
    for sample in &track.samples {
        if sample.data.is_empty() {
            continue;
        }
        let sample_nalus = parse_nal_units_mp4(&sample.data, length_size)
            .map_err(|e| StegoError::InvalidVideo(
                format!("sample NAL parse: {e}")
            ))?;
        nalus.extend(sample_nalus);
    }

    let text = h264_stego_decode_nalus_string(&nalus, passphrase)?;
    Ok(crate::stego::payload::PayloadData {
        text,
        files: Vec::new(),
    })
}

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

    // Parse SPS from avcC
    if avcc.sps_nalus.is_empty() {
        return Err(StegoError::InvalidVideo("no SPS in avcC".into()));
    }
    // SPS NAL units in avcC include the NAL header byte
    let sps_nalu = &avcc.sps_nalus[0];
    let sps_rbsp = if !sps_nalu.is_empty() && (sps_nalu[0] & 0x1F) == 7 {
        // Has NAL header — strip it and remove EP bytes
        bitstream::remove_emulation_prevention(&sps_nalu[1..])
    } else {
        bitstream::remove_emulation_prevention(sps_nalu)
    };
    let sps_parsed = sps::parse_sps(&sps_rbsp)?;

    // Parse PPS from avcC
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

/// Scan all frames in the video track, collecting embeddable positions.
///
/// Returns `(positions, ac_energies, uniward_costs)` where `uniward_costs[i]`
/// is `Some(c)` when position `i` lives in an I-slice and has a J-UNIWARD cost
/// computed from reconstructed Y-plane pixels, and `None` when the position is
/// in a P/B slice (falls back to CSF cost in [`compute_all_costs`]).
/// Phase 2c pending-drift state. Collected for the current I-frame while
/// we're still seeing its subsequent P-frames' motion vectors; finalized
/// when the next I-frame (or end-of-scan) arrives, at which point the
/// inter-frame drift multiplier is applied and the costs land in
/// `uniward_costs`.
struct PendingIFrameDrift {
    ref_map: crate::stego::cost::h264_ddca::InterFrameRefMap,
    i_frame_idx: usize,
    /// One entry per I-frame position: (cost_target_idx, frame_block_x,
    /// frame_block_y, cost_after_uniward_plus_intra_drift).
    entries: Vec<(usize, usize, usize, f32)>,
}

/// Retrospective analysis tool — not for production API.
/// Exposes `scan_all_frames` + everything needed to re-run the same filter that
/// `h264_ghost_encode`/`decode` apply, so external tools (e.g. the
/// `h264_stealth_retrospective` example) can inspect the cover position pool
/// without duplicating the whole pipeline. Scans the WHOLE video (last frame
/// inclusive) — the production encode/decode in Phase I.0.5 only scans the
/// first GOP, but this analysis tool deliberately retains the full-video view.
pub fn scan_frames_for_stealth_analysis(
    mp4_bytes: &[u8],
) -> Result<(Vec<EmbeddablePosition>, Vec<f32>, Vec<Option<f32>>), StegoError> {
    let mp4_file = mp4::demux::demux(mp4_bytes)?;
    let (sps, pps, length_size) = extract_h264_params(&mp4_file)?;
    if pps.entropy_coding_mode_flag {
        return Err(StegoError::InvalidVideo(
            "H.264 CABAC not supported; input must be Baseline CAVLC".into(),
        ));
    }
    let track_idx = mp4_file
        .video_track_idx
        .ok_or(StegoError::InvalidVideo("no video track".into()))?;
    let track = &mp4_file.tracks[track_idx];
    let last_frame = track.samples.len().saturating_sub(1);
    scan_frame_range(track, &sps, &pps, length_size, mp4_bytes, 0, last_frame)
}

/// Phase I.0.5/68 — multi-GOP support.
///
/// One GOP's coordinates within `track.samples`: `[first..=last]` inclusive.
/// `gop_idx` is the position of the GOP within the track (0 for the first
/// GOP, 1 for the second, …).
#[derive(Debug, Clone, Copy)]
struct GopRange {
    gop_idx: u32,
    first: usize,
    last: usize, // inclusive
}

/// Walk `track.samples` and partition it into GOPs by IDR boundaries.
///
/// A GOP starts at a sync (IDR) frame and includes all subsequent non-sync
/// frames until the next sync frame (or end of video). If the first sample
/// is not sync (rare / malformed), the function still treats `[0..]` as the
/// first GOP for safety.
fn discover_gops(track: &mp4::Track) -> Vec<GopRange> {
    let n = track.samples.len();
    if n == 0 {
        return Vec::new();
    }
    let mut gops = Vec::new();
    let mut current_first = 0usize;
    let mut next_idx = 0u32;
    for i in 1..n {
        if track.samples[i].is_sync {
            gops.push(GopRange {
                gop_idx: next_idx,
                first: current_first,
                last: i - 1,
            });
            next_idx += 1;
            current_first = i;
        }
    }
    gops.push(GopRange {
        gop_idx: next_idx,
        first: current_first,
        last: n - 1,
    });
    gops
}

/// Per-GOP usable-position counts (after the encode-side filter:
/// coeff scan_pos > 0 OR MvdLsb).
#[derive(Debug, Clone, Copy)]
struct GopCapacity {
    range: GopRange,
    n_coeff: usize,
    n_mvd: usize,
}

/// Phase 1 of the per-GOP streaming pipeline: count usable positions per GOP.
///
/// Calls `scan_frame_range` for each GOP individually, then drops the
/// per-GOP metadata before moving to the next. Peak working set during this
/// pass equals one GOP's worth of metadata (~150 MB at 1080p / 30-frame GOP).
///
/// Counts apply the SAME filter chain as `build_domain_stc`: scan_pos > 0
/// for coeff domain (or `MvdLsb`), AND finite cost from `compute_all_costs`.
/// This ensures `n_g_c` and `n_g_m` here match the actual cover-bits count
/// the encoder will see in Phase 2 (`build_domain_stc` drops infinite-cost
/// positions). Without the cost filter the counts overestimate and the
/// encoder allocates message bits that overflow the actual cover.
///
/// We pay for a full scan here (including UNIWARD + DDCA + cost computation)
/// even though Phase 2 re-scans each GOP again. Empirically the wasted cost
/// is small relative to the encode-on-mobile budget. Optimize if profiling
/// later shows this dominates wall time.
fn scan_gop_capacities(
    track: &mp4::Track,
    sps: &Sps,
    pps: &Pps,
    length_size: u8,
    mp4_bytes: &[u8],
) -> Result<Vec<GopCapacity>, StegoError> {
    let gops = discover_gops(track);
    let mut caps = Vec::with_capacity(gops.len());
    for range in gops {
        let (positions, ac_energies, uniward_costs) = scan_frame_range(
            track,
            sps,
            pps,
            length_size,
            mp4_bytes,
            range.first,
            range.last,
        )?;
        let costs = compute_all_costs(&positions, &ac_energies, &uniward_costs, track);
        let mut n_coeff = 0usize;
        let mut n_mvd = 0usize;
        for (i, p) in positions.iter().enumerate() {
            // Mirror `build_domain_stc`'s filter chain: scan_pos > 0 for
            // coeff (or MvdLsb), AND finite cost (`build_domain_stc` drops
            // infinite-cost positions before STC).
            if !costs[i].is_finite() {
                continue;
            }
            let is_coeff = matches!(
                p.domain,
                EmbedDomain::T1Sign | EmbedDomain::LevelSuffixMag | EmbedDomain::LevelSuffixSign
            );
            if is_coeff && p.scan_pos > 0 {
                n_coeff += 1;
            } else if p.domain == EmbedDomain::MvdLsb {
                n_mvd += 1;
            }
        }
        caps.push(GopCapacity {
            range,
            n_coeff,
            n_mvd,
        });
        // positions / energies / costs drop here.
    }
    Ok(caps)
}

/// Compute per-GOP message bit allocation given total capacities + a target
/// total message length `m_total`. Returns `(gop_m_coeff, gop_m_mvd)` —
/// per-GOP coefficient + MVD bit counts. Uses proportional split:
/// `m_g_c = m_c_total * n_g_c / sum(n_g_c)`. Rounding remainder is absorbed
/// by the last GOP. Both encoder and decoder run this with the same
/// `(capacities, m_total)` so they agree on the split.
fn compute_per_gop_message_split(
    capacities: &[GopCapacity],
    m_total: usize,
    total_n_coeff: usize,
    total_n_mvd: usize,
) -> (Vec<usize>, Vec<usize>) {
    let total_n = total_n_coeff + total_n_mvd;
    let m_c_total = if total_n > 0 {
        m_total * total_n_coeff / total_n
    } else {
        0
    };
    let m_m_total = m_total.saturating_sub(m_c_total);
    let mut gop_m_c = Vec::with_capacity(capacities.len());
    let mut gop_m_m = Vec::with_capacity(capacities.len());
    let mut allocated_m_c = 0usize;
    let mut allocated_m_m = 0usize;
    for cap in capacities {
        let m_g_c = if total_n_coeff > 0 {
            m_c_total * cap.n_coeff / total_n_coeff
        } else {
            0
        };
        let m_g_m = if total_n_mvd > 0 {
            m_m_total * cap.n_mvd / total_n_mvd
        } else {
            0
        };
        gop_m_c.push(m_g_c);
        gop_m_m.push(m_g_m);
        allocated_m_c += m_g_c;
        allocated_m_m += m_g_m;
    }
    // Absorb rounding remainder into the last GOP that has the relevant pool.
    if let Some(idx) = capacities.iter().rposition(|c| c.n_coeff > 0) {
        gop_m_c[idx] += m_c_total.saturating_sub(allocated_m_c);
    }
    if let Some(idx) = capacities.iter().rposition(|c| c.n_mvd > 0) {
        gop_m_m[idx] += m_m_total.saturating_sub(allocated_m_m);
    }
    (gop_m_c, gop_m_m)
}

fn scan_frame_range(
    track: &mp4::Track,
    sps: &Sps,
    pps: &Pps,
    length_size: u8,
    _mp4_bytes: &[u8],
    first_frame_inclusive: usize,
    last_frame_inclusive: usize,
) -> Result<(Vec<EmbeddablePosition>, Vec<f32>, Vec<Option<f32>>), StegoError> {
    let mut all_positions = Vec::new();
    let mut all_ac_energies = Vec::new();
    let mut uniward_costs: Vec<Option<f32>> = Vec::new();
    let mut global_block_idx = 0u32;
    // Phase 2c state — one pending I-frame at a time.
    let mut pending_drift: Option<PendingIFrameDrift> = None;
    let ddca_params = crate::stego::cost::h264_ddca::DdcaParams::default();

    let height_in_mbs = if sps.frame_mbs_only_flag {
        sps.pic_height_in_map_units
    } else {
        sps.pic_height_in_map_units * 2
    };

    let n_frames = last_frame_inclusive
        .saturating_add(1)
        .saturating_sub(first_frame_inclusive);
    for (frame_idx, sample) in track
        .samples
        .iter()
        .enumerate()
        .skip(first_frame_inclusive)
        .take(n_frames)
    {
        if sample.data.is_empty() {
            continue;
        }

        // Parse NAL units from sample data, finding slice NALs with EP byte maps
        let ls = length_size as usize;
        let mut nal_pos = 0usize;
        while nal_pos + ls <= sample.data.len() {
            let mut nal_len = 0usize;
            for i in 0..ls {
                nal_len = (nal_len << 8) | sample.data[nal_pos + i] as usize;
            }
            let nal_data_start = nal_pos + ls;
            let nal_data_end = (nal_data_start + nal_len).min(sample.data.len());
            nal_pos = nal_data_start + nal_len;

            if nal_len == 0 || nal_data_start >= sample.data.len() {
                continue;
            }

            let nal_data = &sample.data[nal_data_start..nal_data_end];
            if nal_data.is_empty() {
                continue;
            }

            let nal_type = NalType(nal_data[0] & 0x1F);
            let nal_ref_idc = (nal_data[0] >> 5) & 0x03;

            if !nal_type.is_slice() {
                continue;
            }

            // Build EP byte map for this NAL's payload (after header byte)
            let raw_payload = &nal_data[1..];
            let (rbsp, ep_map) = bitstream::remove_emulation_prevention_with_map(raw_payload);

            // Parse slice header from RBSP
            let slice_hdr =
                slice::parse_slice_header(&rbsp, sps, pps, nal_type, nal_ref_idc)
                    .map_err(|e| {
                        StegoError::InvalidVideo(format!("frame {frame_idx} slice header: {e}"))
                    })?;

            // Parse macroblocks
            let _data_byte = slice_hdr.data_bit_offset / 8;
            let _data_bit = (slice_hdr.data_bit_offset % 8) as u8;

            let mut reader = bitstream::RbspReader::new(&rbsp);
            // Skip to data_bit_offset
            reader.skip_bits(slice_hdr.data_bit_offset as u32)
                .map_err(|e| StegoError::InvalidVideo(format!("skip to data: {e}")))?;

            let mut neighbor_ctx = NeighborContext::new(sps.pic_width_in_mbs, height_in_mbs);
            // Phase 2: per-slice MV predictor context for P-slice MV parsing.
            // Created unconditionally (tiny memory cost) so parse_macroblock
            // can thread it through for any slice; intra slices leave it empty.
            let mut mv_ctx = crate::codec::h264::mv::MvPredictorContext::new(
                sps.pic_width_in_mbs,
                height_in_mbs,
            );
            let mut current_qp = slice_hdr.slice_qp;

            // Parse mb_skip_run for P-slices, then macroblocks
            let total_mbs = sps.pic_size_in_mbs;
            let mut mb_idx = slice_hdr.first_mb_in_slice;

            let _slice_start_mb = mb_idx;
            let mut _parse_ok_count = 0usize;
            let mut _parse_skip_count = 0usize;
            let mut _parse_fail_info: Option<(u32, String)> = None;

            // Phase 1b: on I-slices, capture reconstruction data + per-MB
            // positional info so we can compute J-UNIWARD costs after the
            // slice parses. P/B slices skip this and stay on CSF.
            let is_intra_slice = matches!(slice_hdr.slice_type, SliceType::I | SliceType::SI);
            let capture_recon = is_intra_slice;
            // Per-MB reconstruction buffer, indexed by mb_idx in frame
            // coordinates. Pre-filled with default (no-recon) macroblocks so
            // reconstructor sees a dense frame even when only one slice is
            // coded.
            let mut frame_mbs: Vec<Macroblock> = if is_intra_slice {
                (0..total_mbs as usize)
                    .map(|_| default_macroblock())
                    .collect()
            } else {
                Vec::new()
            };
            // QP per MB (same layout as frame_mbs). Seeded from the slice QP
            // so untouched MBs contribute a sensible default.
            let mut frame_qps: Vec<i32> = if is_intra_slice {
                vec![slice_hdr.slice_qp; total_mbs as usize]
            } else {
                Vec::new()
            };
            // (mb_idx_in_frame, positions_start, positions_end) for every
            // successfully-parsed MB in this slice. Used to build the
            // FramePosition list for UNIWARD scoring.
            let mut slice_parsed_mbs: Vec<(usize, usize, usize)> = Vec::new();

            while mb_idx < total_mbs {
                // P-slice: parse mb_skip_run
                if slice_hdr.slice_type == SliceType::P || slice_hdr.slice_type == SliceType::SP {
                    if reader.bits_remaining() < 2 {
                        break;
                    }
                    let skip_run = reader.read_ue()
                        .map_err(|e| StegoError::InvalidVideo(format!("mb_skip_run: {e}")))?;

                    // Skip `skip_run` macroblocks (no data, no positions)
                    for _ in 0..skip_run {
                        if mb_idx >= total_mbs {
                            break;
                        }
                        // Update neighbor context for skip MBs (all TotalCoeffs = 0)
                        let sx = mb_idx % sps.pic_width_in_mbs;
                        let sy = mb_idx / sps.pic_width_in_mbs;
                        for blk in 0..16 {
                            let (bpx, bpy) = crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS[blk];
                            let bx = (sx * 4 + bpx as u32) as usize;
                            let by = (sy * 4 + bpy as u32) as usize;
                            neighbor_ctx.set_luma_tc(bx, by, 0);
                        }
                        // 26 slots per MB: 16 luma + 2 chroma DC + 4 Cb AC + 4 Cr AC.
                        // Skip MBs contribute zero energy; append 26 zero entries so
                        // the energy array stays aligned with global_block_idx.
                        all_ac_energies.extend([0.0; 26]);
                        global_block_idx += 26;
                        mb_idx += 1;
                        _parse_skip_count += 1;
                    }

                    if mb_idx >= total_mbs || skip_run > 0 && reader.bits_remaining() < 2 {
                        break;
                    }
                    if skip_run > 0 {
                        // After skip run, if more MBs remain, parse the next MB
                        // (the mb_skip_run terminated with a non-skip MB)
                    }
                }

                if mb_idx >= total_mbs || reader.bits_remaining() < 2 {
                    break;
                }

                // IMPORTANT: compute mb_x/mb_y from CURRENT mb_idx, AFTER any
                // mb_skip_run advanced it. Doing this at the top of the loop
                // leaves stale coordinates when skip_run > 0, which corrupts
                // nC neighbor context lookups inside parse_macroblock.
                let mb_x = mb_idx % sps.pic_width_in_mbs;
                let mb_y = mb_idx / sps.pic_width_in_mbs;

                let _mb_bit_pos_before = reader.bits_read();

                // Parse macroblock (with recon capture on I-slices for UNIWARD;
                // MV tracking for P-slices).
                let mb_result = macroblock::parse_macroblock_with_recon(
                    &mut reader,
                    slice_hdr.slice_type,
                    mb_x,
                    mb_y,
                    sps,
                    pps,
                    &mut neighbor_ctx,
                    &ep_map,
                    raw_payload,
                    &mut current_qp,
                    slice_hdr.num_ref_idx_l0_active,
                    capture_recon,
                    Some(&mut mv_ctx),
                );

                match mb_result {
                    Ok(mb) => {
                        _parse_ok_count += 1;
                        // Compute AC energies for this MB's blocks per the 26-slot
                        // scheme: 0..15 luma AC, 16..17 chroma DC, 18..21 Cb AC,
                        // 22..25 Cr AC. total_coeffs*2.0 is the rough proxy.
                        for blk_idx in 0..16 {
                            let tc = mb.luma_total_coeffs[blk_idx];
                            all_ac_energies.push(tc as f32 * 2.0);
                        }
                        // Slots 16 + 17: chroma DC (Cb, Cr). No separate DC
                        // total_coeffs is tracked; use 0.0 (DC slots are WET in
                        // UNIWARD anyway, and CSF rarely reads them for the
                        // domains we embed in).
                        all_ac_energies.push(0.0);
                        all_ac_energies.push(0.0);
                        // Slots 18..21: Cb AC (blk 0..3). chroma_total_coeffs[0..4]
                        // is the Cb side of the 8-element array.
                        for blk_idx in 0..4 {
                            let tc = mb.chroma_total_coeffs[blk_idx];
                            all_ac_energies.push(tc as f32 * 2.0);
                        }
                        // Slots 22..25: Cr AC (blk 0..3). chroma_total_coeffs[4..8]
                        // is the Cr side.
                        for blk_idx in 4..8 {
                            let tc = mb.chroma_total_coeffs[blk_idx];
                            all_ac_energies.push(tc as f32 * 2.0);
                        }

                        let pos_start = all_positions.len();

                        // Position in sample.data = nal_data_start + 1 (NAL header) + raw_byte_offset
                        let positions_moved: Vec<_> = mb.positions.to_vec();
                        let mb_idx_val = global_block_idx / 26;
                        for mut pos in positions_moved {
                            pos.frame_idx = frame_idx as u16;
                            pos.mb_idx = mb_idx_val;
                            // Sentinels mark non-coefficient positions:
                            //   u32::MAX      — I_16x16 luma DC (WET)
                            //   u32::MAX - 1  — Phase 3a MVD suffix LSB
                            // Both stay unshifted so cost functions see them unchanged.
                            if pos.block_idx != u32::MAX
                                && pos.block_idx != crate::codec::h264::mv::MVD_BLOCK_IDX_SENTINEL
                            {
                                pos.block_idx += global_block_idx;
                            }
                            // Adjust to absolute position within sample data
                            pos.raw_byte_offset += nal_data_start + 1;
                            all_positions.push(pos);
                            uniward_costs.push(None);
                        }

                        let pos_end = all_positions.len();

                        if is_intra_slice {
                            // Keep the parsed MB alive for reconstruction.
                            // `frame_mbs` was sized to total_mbs above.
                            frame_qps[mb_idx as usize] = current_qp;
                            frame_mbs[mb_idx as usize] = mb;
                            slice_parsed_mbs.push((mb_idx as usize, pos_start, pos_end));
                        } else if let Some(pending) = pending_drift.as_mut() {
                            // Phase 2c: accumulate this P-MB's MV field into
                            // the pending I-frame's reference map. Decay by
                            // how many P-frames have elapsed since the
                            // anchor I-frame (frame_idx > pending.i_frame_idx
                            // always when we're in this branch).
                            if let Some(mv_field) = mb.mv_field.as_ref() {
                                let hops = frame_idx.saturating_sub(pending.i_frame_idx + 1) as i32;
                                // det_powi_f64 instead of f32::powi — the latter lowers to
                                // @llvm.powi.f64 with implementation-defined rounding on WASM.
                                // det_powi_f64 uses bit-exact IEEE 754 multiply (exp-by-squaring).
                                let decay =
                                    det_powi_f64(ddca_params.inter_frame_decay as f64, hops) as f32;
                                let mb_x_usize =
                                    (mb_idx % sps.pic_width_in_mbs) as usize;
                                let mb_y_usize =
                                    (mb_idx / sps.pic_width_in_mbs) as usize;
                                pending.ref_map.accumulate_mv_field(
                                    mv_field,
                                    mb_x_usize,
                                    mb_y_usize,
                                    decay,
                                );
                            }
                        }

                        global_block_idx += 26;
                    }
                    Err(_e) => {
                        if _parse_fail_info.is_none() {
                            _parse_fail_info = Some((mb_idx, format!("{:?} (at bit {})", _e, _mb_bit_pos_before)));
                        }
                        // Parse error → bitstream drift. Stop parsing this slice.
                        break;
                    }
                }

                mb_idx += 1;
            }

            if std::env::var("PHASM_H264_DEBUG").is_ok() {
                eprintln!("[h264-parse] frame={} slice_type={:?} qp={} parsed={}+skip={}/{} (start={}) bits_left={} num_ref_l0={}",
                    frame_idx, slice_hdr.slice_type, slice_hdr.slice_qp, _parse_ok_count, _parse_skip_count, total_mbs - _slice_start_mb, _slice_start_mb, reader.bits_remaining(), slice_hdr.num_ref_idx_l0_active);
                if let Some((failed_mb, err)) = _parse_fail_info {
                    eprintln!("  FIRST FAIL at MB {}: {}", failed_mb, err);
                }
            }

            // Phase 1b — I-slice only: reconstruct the Y plane from the parsed
            // MBs, run the J-UNIWARD wavelet cost over the positions that
            // belong to this slice, and write the scores back into the
            // uniward_costs slots built up during MB parsing. P-slice
            // positions keep `None` and fall through to CSF in compute_all_costs.
            if is_intra_slice && !slice_parsed_mbs.is_empty() {
                let width = sps.pic_width_in_mbs as usize * 16;
                let height = height_in_mbs as usize * 16;
                let planes =
                    crate::codec::h264::reconstruct::reconstruct_i_frame_planes(
                        &frame_mbs,
                        sps.pic_width_in_mbs as usize,
                        height_in_mbs as usize,
                    );

                let mut frame_pos: Vec<h264_uniward::FramePosition> = Vec::new();
                let mut cost_target_idx: Vec<usize> = Vec::new();
                for &(mb_idx_in_frame, pos_start, pos_end) in &slice_parsed_mbs {
                    // Pull qp_cb/qp_cr from the MB's recon (captured at parse
                    // time via derive_chroma_qp). Fallback 26 keeps the path
                    // well-behaved if recon is absent for any reason.
                    let (qp_cb, qp_cr) = frame_mbs
                        .get(mb_idx_in_frame)
                        .and_then(|mb| mb.recon.as_ref())
                        .map_or((26, 26), |r| (r.qp_cb, r.qp_cr));
                    for global_pos_idx in pos_start..pos_end {
                        let pos = &all_positions[global_pos_idx];
                        // within_mb_block_idx = pos.block_idx mod 26 under the
                        // Phase 2 slot scheme. u32::MAX is the sentinel for
                        // I_16x16 luma DC (skip — they get INF from CSF
                        // fallback since we don't add to uniward_costs).
                        if pos.block_idx == u32::MAX
                            || pos.block_idx == crate::codec::h264::mv::MVD_BLOCK_IDX_SENTINEL
                        {
                            continue;
                        }
                        let within_mb = (pos.block_idx % 26) as usize;
                        frame_pos.push(h264_uniward::FramePosition {
                            pos,
                            mb_idx: mb_idx_in_frame,
                            within_mb_block_idx: within_mb,
                            qp_cb,
                            qp_cr,
                        });
                        cost_target_idx.push(global_pos_idx);
                    }
                }

                let frame_planes = h264_uniward::FramePlanes {
                    y: &planes.y,
                    cb: &planes.cb,
                    cr: &planes.cr,
                    width,
                    height,
                };
                let scores = h264_uniward::compute_frame_uniward_costs(
                    &frame_planes,
                    &frame_pos,
                    &frame_qps,
                );

                // Phase 2b — DDCA inter-block intra-prediction drift.
                // Build an intra-mode map from the parsed MBs and apply a
                // drift multiplier on top of the UNIWARD base cost so STC
                // preferentially avoids flipping at the head of a long
                // intra-prediction chain.
                let mode_map = crate::stego::cost::h264_ddca::IntraModeMap::build(
                    &frame_mbs,
                    sps.pic_width_in_mbs as usize,
                    height_in_mbs as usize,
                );
                let adjusted = crate::stego::cost::h264_ddca::apply_drift_multipliers(
                    &frame_pos,
                    &scores,
                    &mode_map,
                    sps.pic_width_in_mbs as usize,
                    &ddca_params,
                );

                // Phase 2c — stage the costs into the pending-drift state
                // instead of writing to uniward_costs directly. We'll apply
                // the inter-frame drift multiplier after we've seen the
                // motion vectors of every P-frame that references this
                // I-frame (i.e., up until the next I-slice or end of scan).
                //
                // First finalise any previously-pending I-frame — its P-frame
                // MV stream has ended (we're about to start the new I-frame's
                // drift accumulation).
                finalize_pending_drift(&mut pending_drift, &mut uniward_costs, &ddca_params);

                let width_in_4x4 = sps.pic_width_in_mbs as usize * 4;
                let height_in_4x4 = height_in_mbs as usize * 4;
                let mut entries: Vec<(usize, usize, usize, f32)> =
                    Vec::with_capacity(cost_target_idx.len());
                for (i, &cost) in adjusted.iter().enumerate() {
                    let fp = &frame_pos[i];
                    // Map the 26-slot within-MB index onto a 4×4 frame-grid
                    // coordinate. Only luma (0..15) gets valid coordinates;
                    // chroma positions carry a sentinel so drift lookup is
                    // guaranteed to return 0.0 and leave the cost intact.
                    let (bx, by) = if fp.within_mb_block_idx < 16 {
                        let (dx, dy) = crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS
                            [fp.within_mb_block_idx];
                        let mb_x = fp.mb_idx % sps.pic_width_in_mbs as usize;
                        let mb_y = fp.mb_idx / sps.pic_width_in_mbs as usize;
                        (mb_x * 4 + dx as usize, mb_y * 4 + dy as usize)
                    } else {
                        // Out-of-range sentinel -> ref_count returns 0 (no boost).
                        (usize::MAX, usize::MAX)
                    };
                    entries.push((cost_target_idx[i], bx, by, cost));
                }
                pending_drift = Some(PendingIFrameDrift {
                    ref_map: crate::stego::cost::h264_ddca::InterFrameRefMap::new(
                        width_in_4x4,
                        height_in_4x4,
                    ),
                    i_frame_idx: frame_idx,
                    entries,
                });
                // planes / frame_mbs / frame_qps drop on scope exit.
            }
        }
    }

    // Phase 2c: finalise the last pending I-frame's drift now that we've
    // seen every P-frame that could reference it.
    finalize_pending_drift(&mut pending_drift, &mut uniward_costs, &ddca_params);

    Ok((all_positions, all_ac_energies, uniward_costs))
}

/// Finalise a `PendingIFrameDrift`, applying the inter-frame drift
/// multiplier to each staged cost and writing the result into
/// `uniward_costs`. Sets `pending_drift` back to `None`. No-op when
/// pending is already `None`.
fn finalize_pending_drift(
    pending_drift: &mut Option<PendingIFrameDrift>,
    uniward_costs: &mut [Option<f32>],
    params: &crate::stego::cost::h264_ddca::DdcaParams,
) {
    let Some(pending) = pending_drift.take() else {
        return;
    };
    for (cost_idx, bx, by, base_cost) in pending.entries {
        if bx == usize::MAX || by == usize::MAX {
            // Chroma / sentinel — keep as-is.
            uniward_costs[cost_idx] = Some(base_cost);
            continue;
        }
        let refs = pending.ref_map.ref_count(bx, by);
        let adjusted = base_cost * (1.0 + params.w_inter_drift * refs);
        uniward_costs[cost_idx] = Some(adjusted);
    }
}

/// Build a default macroblock used to pad a slice's reconstruction buffer
/// when not all MBs were parsed (edge slices, parse errors, etc.). The
/// reconstructor treats `recon = None` as a 128-fill, which is the sensible
/// fallback for blocks we can't decode.
fn default_macroblock() -> Macroblock {
    Macroblock {
        mb_type: crate::codec::h264::macroblock::MbType::P16x16,
        mb_qp_delta: 0,
        coded_block_pattern: 0,
        luma_total_coeffs: [0; 16],
        chroma_total_coeffs: [0; 8],
        positions: Vec::new(),
        recon: None,
        mv_field: None,
    }
}

/// Compute costs for all positions, using per-frame slice type and GOP position.
///
/// `ac_energies` is indexed by the position's global `block_idx` (one entry per
/// 4x4 block in every parsed MB, 24 per MB). The inner cost function does the
/// lookup itself, so we pass the full array through without slicing.
fn compute_all_costs(
    positions: &[EmbeddablePosition],
    ac_energies: &[f32],
    uniward_costs: &[Option<f32>],
    track: &mp4::Track,
) -> Vec<f32> {
    let gop_length = estimate_gop_length(track);

    positions
        .iter()
        .enumerate()
        .map(|(i, pos)| {
            // Prefer the Phase 1b UNIWARD cost if available (I-slice positions
            // with reconstructed pixels). Fall back to the Phase 1a CSF cost
            // for P-frame positions and as a safety net when reconstruction
            // data is missing.
            if let Some(uw) = uniward_costs.get(i).copied().flatten()
                && uw.is_finite() && uw > 0.0 {
                    return uw;
                }

            let frame_idx = pos.frame_idx as usize;
            let is_sync = frame_idx < track.samples.len() && track.samples[frame_idx].is_sync;
            let slice_type = if is_sync { SliceType::I } else { SliceType::P };

            let mut gop_pos = 0u32;
            for i in (0..=frame_idx).rev() {
                if i < track.samples.len() && track.samples[i].is_sync {
                    gop_pos = (frame_idx - i) as u32;
                    break;
                }
            }

            h264_cost::compute_h264_costs(
                std::slice::from_ref(pos),
                ac_energies,
                slice_type,
                gop_pos,
                gop_length,
            )[0]
        })
        .collect()
}

fn estimate_gop_length(track: &mp4::Track) -> u32 {
    let sync_count = track.samples.iter().filter(|s| s.is_sync).count();
    if sync_count <= 1 {
        track.samples.len() as u32
    } else {
        (track.samples.len() as u32) / (sync_count as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_gop_length_single_iframe() {
        let track = mp4::Track {
            track_id: 1,
            handler_type: *b"vide",
            codec: *b"avc1",
            width: 320,
            height: 240,
            timescale: 30000,
            duration: 60000,
            samples: vec![
                mp4::Sample { offset: 0, size: 100, is_sync: true, data: vec![] },
                mp4::Sample { offset: 100, size: 50, is_sync: false, data: vec![] },
                mp4::Sample { offset: 150, size: 50, is_sync: false, data: vec![] },
            ],
            hvcc_data: None,
            avcc_data: None,
            stsd_raw: vec![],
            trak_raw: vec![],
        };
        // 1 sync sample → GOP length = total samples = 3
        assert_eq!(estimate_gop_length(&track), 3);
    }

    #[test]
    fn estimate_gop_length_two_iframes() {
        let track = mp4::Track {
            track_id: 1,
            handler_type: *b"vide",
            codec: *b"avc1",
            width: 320,
            height: 240,
            timescale: 30000,
            duration: 120000,
            samples: vec![
                mp4::Sample { offset: 0, size: 100, is_sync: true, data: vec![] },
                mp4::Sample { offset: 100, size: 50, is_sync: false, data: vec![] },
                mp4::Sample { offset: 150, size: 50, is_sync: false, data: vec![] },
                mp4::Sample { offset: 200, size: 100, is_sync: true, data: vec![] },
                mp4::Sample { offset: 300, size: 50, is_sync: false, data: vec![] },
                mp4::Sample { offset: 350, size: 50, is_sync: false, data: vec![] },
            ],
            hvcc_data: None,
            avcc_data: None,
            stsd_raw: vec![],
            trak_raw: vec![],
        };
        // 2 sync samples, 6 total → GOP length = 3
        assert_eq!(estimate_gop_length(&track), 3);
    }
}
