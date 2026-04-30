// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 6E-C2 polish — Reed-Solomon parity for the primary message.
//!
//! Wraps the bit-flip-tolerant RS code from
//! `crate::stego::armor::ecc` for use as the primary-side absorption
//! mechanism. The polish architecture (single-iteration with
//! primary-emit cover) overlays shadow bits ON TOP of the primary
//! STC plan in Pass 3-final, breaking primary's STC syndrome at
//! shadow override positions. Primary RS absorbs those breaks; the
//! decoder STC-extracts as usual, then RS-decodes to recover the
//! framed message.
//!
//! See `docs/design/h264-shadow-messages.md` § "§6E-C2 polish" for
//! the full architecture.
//!
//! ## Status (this commit — utility only)
//!
//! This module ships the RS encode/decode wrappers and unit tests.
//! No production wiring yet — the encoder/decoder primary path
//! continues to call `frame::build_frame` / `frame::parse_frame`
//! directly. Wiring lands in the commit that introduces shadow
//! override-on-top final emit (commit 4 of the polish sequence),
//! once syndrome breaks actually start happening.

use crate::stego::armor::ecc::{
    rs_decode_blocks_with_parity, rs_encode_blocks_with_parity,
    rs_encoded_len_with_parity, RsDecodeError, RsDecodeStats,
};

/// Parity tier ladder for primary RS. Tier 0 = no RS (no-op
/// pass-through); higher tiers absorb up to `tier / 2` byte errors
/// per RS block.
///
/// The ladder mirrors `SHADOW_PARITY_TIERS` from
/// `crate::stego::shadow_layer` with an added `0` tier for the
/// no-shadow / no-RS fast path.
pub const PRIMARY_PARITY_TIERS: [usize; 7] = [0, 4, 8, 16, 32, 64, 128];

/// RS-encode the framed primary bytes. With `parity_len == 0` this
/// is a pass-through (no allocation beyond `Vec::from(frame_bytes)`).
///
/// Block-based — splits long frames into RS blocks of size
/// `(255 - parity_len)` data + `parity_len` parity. Decoder must
/// know `frame_bytes.len()` to recover (passed alongside the
/// encoded bytes).
pub fn primary_rs_encode(frame_bytes: &[u8], parity_len: usize) -> Vec<u8> {
    if parity_len == 0 {
        return frame_bytes.to_vec();
    }
    rs_encode_blocks_with_parity(frame_bytes, parity_len)
}

/// RS-decode the received bytes back to `frame_bytes`. With
/// `parity_len == 0` this is a pass-through (the input is already
/// the framed bytes).
///
/// `data_len` is the original `frame_bytes.len()` — the decoder
/// brute-forces this in the same way it brute-forces `m_total`.
pub fn primary_rs_decode(
    received: &[u8],
    data_len: usize,
    parity_len: usize,
) -> Result<(Vec<u8>, RsDecodeStats), RsDecodeError> {
    if parity_len == 0 {
        if received.len() != data_len {
            return Err(RsDecodeError);
        }
        return Ok((received.to_vec(), RsDecodeStats::default()));
    }
    rs_decode_blocks_with_parity(received, data_len, parity_len)
}

/// Total encoded byte length for a given `data_len` + `parity_len`.
/// At `parity_len == 0` this is `data_len` (pass-through).
pub fn primary_rs_encoded_len(data_len: usize, parity_len: usize) -> usize {
    if parity_len == 0 {
        return data_len;
    }
    rs_encoded_len_with_parity(data_len, parity_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(len: usize) -> Vec<u8> {
        (0..len).map(|i| ((i * 37 + 13) & 0xff) as u8).collect()
    }

    #[test]
    fn round_trip_parity_zero_is_identity() {
        let frame = make_frame(64);
        let encoded = primary_rs_encode(&frame, 0);
        assert_eq!(encoded, frame);
        let (decoded, _) = primary_rs_decode(&encoded, frame.len(), 0).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn round_trip_at_each_parity_tier() {
        let frame = make_frame(96);
        for &parity in &PRIMARY_PARITY_TIERS {
            let encoded = primary_rs_encode(&frame, parity);
            assert_eq!(
                encoded.len(),
                primary_rs_encoded_len(frame.len(), parity),
                "encoded len mismatch at parity {parity}"
            );
            let (decoded, stats) =
                primary_rs_decode(&encoded, frame.len(), parity).unwrap();
            assert_eq!(decoded, frame, "round-trip mismatch at parity {parity}");
            assert_eq!(stats.total_errors, 0, "unexpected errors at parity {parity}");
        }
    }

    #[test]
    fn rs_absorbs_byte_errors_up_to_capacity() {
        let frame = make_frame(80);
        // parity 16 → t_max = 8 byte errors per block (single block here).
        let parity = 16;
        let mut encoded = primary_rs_encode(&frame, parity);
        // Inject 6 byte errors (under capacity).
        for &pos in &[0_usize, 7, 13, 25, 47, 71] {
            encoded[pos] ^= 0xa5;
        }
        let (decoded, stats) = primary_rs_decode(&encoded, frame.len(), parity)
            .expect("RS decode should succeed under capacity");
        assert_eq!(decoded, frame);
        assert_eq!(stats.total_errors, 6);
    }

    #[test]
    fn rs_fails_when_errors_exceed_capacity() {
        let frame = make_frame(80);
        let parity = 8; // t_max = 4 byte errors per block.
        let mut encoded = primary_rs_encode(&frame, parity);
        // Inject 5 byte errors (1 over capacity).
        for &pos in &[0_usize, 11, 22, 33, 44] {
            encoded[pos] ^= 0xa5;
        }
        let result = primary_rs_decode(&encoded, frame.len(), parity);
        assert!(result.is_err(), "expected RS decode failure over capacity");
    }

    #[test]
    fn parity_zero_decode_rejects_length_mismatch() {
        let frame = make_frame(64);
        let mut encoded = primary_rs_encode(&frame, 0);
        encoded.push(0); // length mismatch.
        assert!(primary_rs_decode(&encoded, frame.len(), 0).is_err());
    }

    #[test]
    fn rs_handles_multi_block_frames() {
        // N_MAX = 255; with parity 16 → k_max = 239 data bytes per
        // block. Use a 500-byte frame to force 3 blocks.
        let frame = make_frame(500);
        let parity = 16;
        let encoded = primary_rs_encode(&frame, parity);
        assert_eq!(
            encoded.len(),
            primary_rs_encoded_len(frame.len(), parity)
        );
        let (decoded, stats) = primary_rs_decode(&encoded, frame.len(), parity).unwrap();
        assert_eq!(decoded, frame);
        assert_eq!(stats.num_blocks, 3);
    }
}
