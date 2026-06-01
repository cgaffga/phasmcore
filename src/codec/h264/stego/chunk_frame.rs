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
//!   payload_len   : u16  big-endian   (inline,   0 ..= 65534)   → header 6 B
//!               OR  0xFFFF (u16) + u32 big-endian actual length
//!                                     (extended, len ≥ 65535)    → header 10 B
//!   payload bytes : exactly payload_len bytes
//! ```
//!
//! ### Why `payload_len` is mandatory (#800, 2026-05-29)
//!
//! The decoder recovers the per-GOP STC `m_total` by brute-force,
//! computing `w = ⌊n_cover / m_total⌋` for each candidate. That floor
//! is **many-to-one**: a contiguous range of `m_total` values share
//! one `w`, and because STC extract is convolutional, the chunk header
//! (the leading syndrome bits) parses *identically* across that whole
//! w-class. With no length field the decoder stopped at the class
//! **minimum** `m_total`, returning a **truncated** payload that passed
//! the header parse but failed the outer `parse_frame` → `FrameCorrupted`
//! (it only round-tripped when the true `m_total` happened to be the
//! class minimum). The explicit length makes [`parse_chunk_frame`]
//! length-strict, so short candidates are rejected and the brute-force
//! lands on the exact `m_total`. `m_total` is always byte-aligned, so
//! it sits on the decoder's 8-bit search grid.
//!
//! The `u16`→`u32` escape mirrors the outer frame's v1/v2 length idiom
//! (`stego::frame`). The sentinel is **`0xFFFF`, not `0x0000`**: empty
//! chunks (`payload_len = 0`) are routine — a long video produces more
//! GOPs than a short message has bytes, padding the tail with empty
//! chunks — so `0` must stay a valid inline value. `0xFFFF`
//! unconditionally means "a u32 follows" (no peek-disambiguation); the
//! only cost is that a chunk of *exactly* 65535 bytes uses the extended
//! form.
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
//! There is **no backward compatibility** across the #800 header
//! change — no shipped-to-users H.264 stego predates it, so pre-#800
//! streaming encodes simply fail to decode (cleanly: the misread
//! `payload_len` fails the decoder's `w`/length sanity checks).

use crate::stego::error::StegoError;

/// Bytes consumed by the chunk header in the common (inline-length) form:
/// `chunk_index(2) + total_chunks(2) + payload_len(2)`. Also the smallest
/// possible chunk frame (an empty-payload chunk), hence the decoder's
/// minimum `m_total`.
pub const CHUNK_HEADER_LEN: usize = 6;

/// Bytes consumed by the chunk header in the extended (escaped-length)
/// form: base 4 + `0xFFFF` sentinel(2) + `u32` length(4). Used only when
/// a single GOP carries ≥ 65535 payload bytes.
pub const CHUNK_HEADER_LEN_MAX: usize = 10;

/// `payload_len` u16 value reserved to signal the extended form (a `u32`
/// length follows). Inline lengths are therefore `0 ..= 65534`.
pub const LEN_SENTINEL: u16 = 0xFFFF;

/// Maximum chunks a streaming payload can span. 16-bit index +
/// 16-bit total. Practical limit for v1.0: a 2-hour 4K video at
/// GOP=30 is ~7200 GOPs — well within u16.
pub const MAX_CHUNKS: u16 = u16::MAX;

/// Build the on-wire bytes for one chunk.
///
/// Writes the inline `u16` payload length for `payload.len() <= 65534`,
/// or the `0xFFFF` + `u32` extended form otherwise (see module docs).
///
/// Caller is responsible for sizing `payload` to fit the carrier's
/// STC capacity (capacity_bytes - CHUNK_HEADER_LEN). On encode this
/// is enforced by the per-GOP capacity probe before building.
///
/// # Errors
/// * `InvalidVideo` if `chunk_index >= total_chunks`, `total_chunks == 0`,
///   or `payload.len() > u32::MAX`.
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
    if payload.len() > u32::MAX as usize {
        return Err(StegoError::InvalidVideo(format!(
            "chunk_frame: payload {} exceeds u32::MAX", payload.len()
        )));
    }
    // Extended form when the length can't be represented inline as a
    // non-sentinel u16 (i.e. >= LEN_SENTINEL).
    let extended = payload.len() >= LEN_SENTINEL as usize;
    let header_len = if extended { CHUNK_HEADER_LEN_MAX } else { CHUNK_HEADER_LEN };
    let mut out = Vec::with_capacity(header_len + payload.len());
    out.extend_from_slice(&chunk_index.to_be_bytes());
    out.extend_from_slice(&total_chunks.to_be_bytes());
    if extended {
        out.extend_from_slice(&LEN_SENTINEL.to_be_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    } else {
        out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    }
    out.extend_from_slice(payload);
    Ok(out)
}

/// Parse the chunk header off the front of `bytes`. Returns
/// `(chunk_index, total_chunks, payload_slice)` on success.
///
/// **Length-strict** (#800): the returned slice is exactly
/// `payload_len` bytes; if `bytes` is shorter than `header + payload_len`
/// this returns `None`. The streaming decoder relies on this to reject
/// under-extracted (too-small `m_total`) candidates during its
/// brute-force, so it lands on the encoder's exact `m_total`.
///
/// Validates: enough bytes for the header (and the declared payload),
/// `total_chunks > 0`, `chunk_index < total_chunks`, and that an
/// extended-form length genuinely needed the escape (`>= LEN_SENTINEL`).
/// The payload slice may be empty if a GOP carried no message bytes
/// (routine on long-video / short-message — the tail GOPs are padded
/// with empty chunks).
pub fn parse_chunk_frame(bytes: &[u8]) -> Option<(u16, u16, &[u8])> {
    if bytes.len() < CHUNK_HEADER_LEN {
        return None;
    }
    let chunk_index = u16::from_be_bytes([bytes[0], bytes[1]]);
    let total_chunks = u16::from_be_bytes([bytes[2], bytes[3]]);
    if total_chunks == 0 || chunk_index >= total_chunks {
        return None;
    }
    let len_field = u16::from_be_bytes([bytes[4], bytes[5]]);
    let (payload_len, header_len) = if len_field == LEN_SENTINEL {
        if bytes.len() < CHUNK_HEADER_LEN_MAX {
            return None;
        }
        let v = u32::from_be_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        // Extended form must encode a length that couldn't be inline,
        // otherwise the bytes aren't a well-formed extended header.
        if v < LEN_SENTINEL as usize {
            return None;
        }
        (v, CHUNK_HEADER_LEN_MAX)
    } else {
        (len_field as usize, CHUNK_HEADER_LEN)
    };
    let end = header_len.checked_add(payload_len)?;
    if bytes.len() < end {
        return None;
    }
    Some((chunk_index, total_chunks, &bytes[header_len..end]))
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
        // #800: base header is idx(2)+total(2)+len(2)=6; extended adds
        // the 0xFFFF sentinel(2)+u32(4) = 10.
        assert_eq!(CHUNK_HEADER_LEN, 6);
        assert_eq!(CHUNK_HEADER_LEN_MAX, 10);
        assert_eq!(LEN_SENTINEL, 0xFFFF);
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
    fn build_parse_roundtrip_empty_chunk() {
        // Empty trailing chunks are routine (long video / short message).
        // payload_len=0 must stay a valid inline value (the reason the
        // sentinel is 0xFFFF, not 0x0000).
        let framed = build_chunk_frame(2, 5, b"").unwrap();
        assert_eq!(framed.len(), CHUNK_HEADER_LEN);
        let (idx, total, slice) = parse_chunk_frame(&framed).unwrap();
        assert_eq!(idx, 2);
        assert_eq!(total, 5);
        assert!(slice.is_empty());
    }

    #[test]
    fn build_parse_inline_boundary() {
        // 65534 stays inline (6-byte header); 65535 escapes to extended
        // (10-byte header) because 0xFFFF is the sentinel.
        let inline = vec![0xABu8; (LEN_SENTINEL as usize) - 1]; // 65534
        let f_inline = build_chunk_frame(0, 1, &inline).unwrap();
        assert_eq!(f_inline.len(), CHUNK_HEADER_LEN + inline.len());
        let (_, _, s) = parse_chunk_frame(&f_inline).unwrap();
        assert_eq!(s, &inline[..]);

        let escaped = vec![0xCDu8; LEN_SENTINEL as usize]; // 65535
        let f_escaped = build_chunk_frame(0, 1, &escaped).unwrap();
        assert_eq!(f_escaped.len(), CHUNK_HEADER_LEN_MAX + escaped.len());
        let (_, _, s2) = parse_chunk_frame(&f_escaped).unwrap();
        assert_eq!(s2.len(), escaped.len());
        assert_eq!(s2, &escaped[..]);
    }

    #[test]
    fn build_parse_roundtrip_extended_large() {
        // > u16 payload uses the extended u32 length form.
        let payload = vec![0x5Au8; 70_000];
        let framed = build_chunk_frame(0, 1, &payload).unwrap();
        assert_eq!(framed.len(), CHUNK_HEADER_LEN_MAX + payload.len());
        let (idx, total, slice) = parse_chunk_frame(&framed).unwrap();
        assert_eq!((idx, total), (0, 1));
        assert_eq!(slice.len(), 70_000);
        assert_eq!(slice, &payload[..]);
    }

    #[test]
    fn parse_is_length_strict_rejects_truncation() {
        // The #800 core invariant: a frame missing even one payload byte
        // must NOT parse. This is what makes the decoder skip too-small
        // m_total candidates in the w-class and land on the exact one.
        let payload = b"the exact length matters";
        let framed = build_chunk_frame(0, 1, payload).unwrap();
        // Drop the final payload byte → buffer < header + payload_len.
        let truncated = &framed[..framed.len() - 1];
        assert!(parse_chunk_frame(truncated).is_none());
    }

    #[test]
    fn parse_ignores_trailing_bytes_returns_exact_payload() {
        // A buffer longer than header+payload_len returns exactly
        // payload_len bytes (a larger m_total in the same w-class still
        // recovers the correct payload — robustness, though the
        // smallest-first brute-force lands on the exact m_total first).
        let payload = b"abc";
        let mut framed = build_chunk_frame(0, 1, payload).unwrap();
        framed.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]); // garbage tail
        let (idx, total, slice) = parse_chunk_frame(&framed).unwrap();
        assert_eq!((idx, total), (0, 1));
        assert_eq!(slice, payload);
    }

    #[test]
    fn parse_rejects_false_sentinel_with_inline_value() {
        // A 0xFFFF sentinel followed by a u32 < LEN_SENTINEL is malformed
        // (such a length would have been encoded inline) → reject.
        let bad = vec![0, 0, 0, 1, 0xFF, 0xFF, 0, 0, 0, 10, 1, 2, 3];
        assert!(parse_chunk_frame(&bad).is_none());
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
