// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! D.0.7 — per-GOP chunk framing for streaming H.264 stego.
//!
//! Multi-GOP streaming stego splits the (already-encrypted+CRC'd)
//! payload bytes across GOP boundaries. Each GOP carries one chunk
//! wrapped in a 4-byte header so the decoder can reassemble them in
//! order without relying on GOP-iteration order.
//!
//! ## Wire format
//!
//! ```text
//!   chunk_index   : u16  big-endian   (0-based)
//!   total_chunks  : u16  big-endian   (1..=u16::MAX)
//!   payload bytes : variable, length implicit from carrier capacity
//! ```
//!
//! The payload length is NOT in the chunk header — it's recovered
//! from the STC m_total brute-force at decode time (see
//! `openh264_stego::try_decode_at`). Inner CRC inside the encrypted
//! payload validates correctness once all chunks are reassembled.
//!
//! ## Why chunks
//!
//! Whole-video STC can't stream: the planner needs the full cover
//! before producing any stego output. Per-GOP STC streams: each GOP
//! plans + emits independently. But the message has to span GOPs to
//! preserve total capacity, hence chunks.
//!
//! ## Compatibility
//!
//! A single-GOP video produces exactly one chunk with `chunk_index=0`
//! `total_chunks=1`. The chunk header still exists in this case —
//! the legacy whole-video format (no chunk header) is a different
//! on-the-wire format and is NOT decode-compatible with the streaming
//! format. Mobile encodes ALWAYS use the streaming format starting
//! v1.0 (D.0.7); legacy `phasm_h264_encode` is retired from mobile.

use crate::stego::error::StegoError;

/// Bytes consumed by the chunk header.
pub const CHUNK_HEADER_LEN: usize = 4;

/// Maximum chunks a streaming payload can span. 16-bit index +
/// 16-bit total. Practical limit for v1.0: a 2-hour 4K video at
/// GOP=30 is ~7200 GOPs — well within u16.
pub const MAX_CHUNKS: u16 = u16::MAX;

/// Build the on-wire bytes for one chunk.
///
/// Caller is responsible for sizing `payload` to fit the carrier's
/// STC capacity (capacity_bytes - CHUNK_HEADER_LEN). On encode this
/// is enforced by the per-GOP capacity probe before building.
///
/// # Errors
/// * `MessageTooLarge` if `chunk_index >= total_chunks` or
///   `total_chunks == 0`.
pub fn build_chunk_frame(
    chunk_index: u16,
    total_chunks: u16,
    payload: &[u8],
) -> Result<Vec<u8>, StegoError> {
    if total_chunks == 0 {
        return Err(StegoError::InvalidVideo(
            "chunk_frame: total_chunks must be > 0".into(),
        ));
    }
    if chunk_index >= total_chunks {
        return Err(StegoError::InvalidVideo(format!(
            "chunk_frame: chunk_index {chunk_index} >= total_chunks {total_chunks}"
        )));
    }
    let mut out = Vec::with_capacity(CHUNK_HEADER_LEN + payload.len());
    out.extend_from_slice(&chunk_index.to_be_bytes());
    out.extend_from_slice(&total_chunks.to_be_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

/// Parse the chunk header off the front of `bytes`. Returns
/// `(chunk_index, total_chunks, payload_slice)` on success.
///
/// Validates: `bytes.len() >= CHUNK_HEADER_LEN`, `total_chunks > 0`,
/// `chunk_index < total_chunks`. The payload slice may be empty if
/// a GOP ended up carrying no message bytes (degenerate, only used
/// in tests).
pub fn parse_chunk_frame(bytes: &[u8]) -> Option<(u16, u16, &[u8])> {
    if bytes.len() < CHUNK_HEADER_LEN {
        return None;
    }
    let chunk_index = u16::from_be_bytes([bytes[0], bytes[1]]);
    let total_chunks = u16::from_be_bytes([bytes[2], bytes[3]]);
    if total_chunks == 0 || chunk_index >= total_chunks {
        return None;
    }
    Some((chunk_index, total_chunks, &bytes[CHUNK_HEADER_LEN..]))
}

/// Split `message_bytes` into `total_chunks` evenly-sized pieces.
/// The final chunk may be smaller if the message length doesn't
/// divide evenly. Returns the chunk payload slices (NOT yet framed
/// with `build_chunk_frame` — caller does that per chunk after
/// determining the per-GOP capacity).
///
/// # Errors
/// * `MessageTooLarge` if `total_chunks == 0` or `total_chunks >
///   MAX_CHUNKS`.
pub fn split_message_into_chunks(
    message_bytes: &[u8],
    total_chunks: u16,
) -> Result<Vec<Vec<u8>>, StegoError> {
    if total_chunks == 0 {
        return Err(StegoError::InvalidVideo(
            "split_message_into_chunks: total_chunks must be > 0".into(),
        ));
    }
    let n = total_chunks as usize;
    let len = message_bytes.len();
    // Ceiling-divide so the last chunk absorbs the remainder.
    let chunk_size = len.div_ceil(n);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * chunk_size;
        let end = ((i + 1) * chunk_size).min(len);
        if start >= len {
            // Pad with empty chunks if message is shorter than n_chunks
            // (rare; only happens if caller picked too many chunks for
            // a small message). The empty chunks still need headers so
            // total_chunks is consistent across GOPs.
            out.push(Vec::new());
        } else {
            out.push(message_bytes[start..end].to_vec());
        }
    }
    Ok(out)
}

/// Reassemble parsed chunks into the original message bytes.
/// Input: vector of `(chunk_index, payload)` pairs collected from
/// per-GOP decode passes. Caller has already validated that
/// `total_chunks` is consistent across all chunks.
///
/// Returns `None` if:
/// - Any chunk_index is duplicated
/// - Any chunk_index >= total_chunks
/// - total_chunks unique chunks aren't all present
///
/// On success returns the concatenated payload.
pub fn assemble_chunks(
    mut chunks: Vec<(u16, Vec<u8>)>,
    total_chunks: u16,
) -> Option<Vec<u8>> {
    if total_chunks == 0 || chunks.len() != total_chunks as usize {
        return None;
    }
    chunks.sort_by_key(|(idx, _)| *idx);
    let mut out = Vec::new();
    for (i, (idx, payload)) in chunks.iter().enumerate() {
        if *idx as usize != i {
            // Missing or duplicate chunk_index.
            return None;
        }
        out.extend_from_slice(payload);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_constants_match_wire_layout() {
        assert_eq!(CHUNK_HEADER_LEN, 4);
    }

    #[test]
    fn build_parse_roundtrip_single_chunk() {
        let payload = b"hello world";
        let framed = build_chunk_frame(0, 1, payload).unwrap();
        assert_eq!(framed.len(), CHUNK_HEADER_LEN + payload.len());
        let (idx, total, slice) = parse_chunk_frame(&framed).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(total, 1);
        assert_eq!(slice, payload);
    }

    #[test]
    fn build_rejects_zero_total_chunks() {
        assert!(build_chunk_frame(0, 0, b"data").is_err());
    }

    #[test]
    fn build_rejects_out_of_range_index() {
        assert!(build_chunk_frame(5, 5, b"data").is_err());
        assert!(build_chunk_frame(10, 5, b"data").is_err());
    }

    #[test]
    fn parse_rejects_short_buffer() {
        assert!(parse_chunk_frame(&[]).is_none());
        assert!(parse_chunk_frame(&[0, 0, 0]).is_none()); // only 3 bytes
    }

    #[test]
    fn parse_rejects_zero_total_chunks() {
        let bad = vec![0, 0, 0, 0, 1, 2, 3]; // chunk_idx=0, total=0, payload
        assert!(parse_chunk_frame(&bad).is_none());
    }

    #[test]
    fn parse_rejects_index_oob() {
        // chunk_idx=5, total=3 → invalid.
        let bad = vec![0, 5, 0, 3, 1, 2, 3];
        assert!(parse_chunk_frame(&bad).is_none());
    }

    #[test]
    fn split_evenly_when_divisible() {
        let msg = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let chunks = split_message_into_chunks(&msg, 4).unwrap();
        assert_eq!(chunks.len(), 4);
        for c in &chunks { assert_eq!(c.len(), 2); }
        assert_eq!(chunks[0], &[1, 2]);
        assert_eq!(chunks[3], &[7, 8]);
    }

    #[test]
    fn split_uneven_distributes_remainder_to_last() {
        let msg = vec![1u8, 2, 3, 4, 5];
        let chunks = split_message_into_chunks(&msg, 2).unwrap();
        assert_eq!(chunks.len(), 2);
        // div_ceil(5, 2) = 3, so chunks are [1,2,3] and [4,5].
        assert_eq!(chunks[0], &[1, 2, 3]);
        assert_eq!(chunks[1], &[4, 5]);
    }

    #[test]
    fn split_with_more_chunks_than_bytes_pads_empty() {
        let msg = vec![1u8, 2];
        let chunks = split_message_into_chunks(&msg, 5).unwrap();
        assert_eq!(chunks.len(), 5);
        // div_ceil(2, 5) = 1, so first 2 chunks have 1 byte each, rest empty.
        assert_eq!(chunks[0], &[1]);
        assert_eq!(chunks[1], &[2]);
        assert!(chunks[2].is_empty());
        assert!(chunks[3].is_empty());
        assert!(chunks[4].is_empty());
    }

    #[test]
    fn split_rejects_zero_chunks() {
        assert!(split_message_into_chunks(&[1, 2, 3], 0).is_err());
    }

    #[test]
    fn assemble_roundtrip_in_order() {
        let original = b"the quick brown fox";
        let chunks = split_message_into_chunks(original, 3).unwrap();
        let collected: Vec<(u16, Vec<u8>)> =
            chunks.into_iter().enumerate().map(|(i, c)| (i as u16, c)).collect();
        let reassembled = assemble_chunks(collected, 3).unwrap();
        assert_eq!(reassembled.as_slice(), original);
    }

    #[test]
    fn assemble_roundtrip_out_of_order() {
        let original = b"the quick brown fox";
        let chunks = split_message_into_chunks(original, 3).unwrap();
        let mut collected: Vec<(u16, Vec<u8>)> =
            chunks.into_iter().enumerate().map(|(i, c)| (i as u16, c)).collect();
        // Shuffle: [2, 0, 1]
        collected.swap(0, 2);
        let reassembled = assemble_chunks(collected, 3).unwrap();
        assert_eq!(reassembled.as_slice(), original);
    }

    #[test]
    fn assemble_detects_missing_chunk() {
        let collected = vec![(0u16, vec![1, 2]), (2u16, vec![5, 6])];
        // Only 2 chunks supplied for total=3 → mismatched count returns None.
        assert!(assemble_chunks(collected, 3).is_none());
    }

    #[test]
    fn assemble_detects_duplicate_chunk() {
        let collected = vec![
            (0u16, vec![1, 2]),
            (0u16, vec![1, 2]),
            (2u16, vec![5, 6]),
        ];
        // chunk_index=1 missing, 0 duplicated.
        assert!(assemble_chunks(collected, 3).is_none());
    }

    #[test]
    fn assemble_detects_oob_chunk_index() {
        let collected = vec![(0u16, vec![1, 2]), (1u16, vec![3, 4]), (5u16, vec![5, 6])];
        // chunk_index=5 but total=3 → OOB.
        assert!(assemble_chunks(collected, 3).is_none());
    }
}
