// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Media-agnostic shadow-layer primitives.
//!
//! Shadow messages provide plausible deniability: multiple messages
//! can be hidden in a single cover (image, video, …), each with a
//! different passphrase. The frame format, parity tier ladder, RS
//! encoding shape, and capacity arithmetic are the same regardless
//! of medium. Per-medium specifics (which positions to embed at,
//! how to write into them, how to walk the cover at decode time)
//! live in the medium-specific modules:
//!
//! - **Image** (Y-channel JPEG nzAC, cost-pool + hash priority):
//!   `core/src/stego/ghost/shadow.rs`
//! - **Video H.264** (4 bypass-bin domains, hash priority + additive
//!   bias for N>1): `core/src/codec/h264/stego/shadow.rs`
//!
//! ## Frame format — v1 + v2 sentinel dispatch (matches `frame.rs`)
//!
//! Single unified shadow frame format with size-driven v1/v2
//! selection, mirroring the primary frame format in `frame.rs`:
//!
//! **v1 (small, ≤ 65535-byte plaintext)** — 46-byte overhead:
//! ```text
//! [plaintext_len: 2B u16 BE] [salt: 16B] [nonce: 12B] [ciphertext: N+16B]
//! ```
//!
//! **v2 (large, > 65535-byte plaintext)** — 50-byte overhead:
//! ```text
//! [sentinel: 2B = 0x0000] [plaintext_len: 4B u32 BE]
//! [salt: 16B] [nonce: 12B] [ciphertext: N+16B]
//! ```
//!
//! Encoder picks v1 or v2 by plaintext size. Parser dispatches on
//! the first two bytes: `header_u16 == 0` + `data.len() >= 6` +
//! `v2_len > u16::MAX` selects v2; otherwise v1. The
//! `v2_len > u16::MAX` tie-breaker protects the legitimate
//! `plaintext_len == 0` v1 case (same logic as `frame::parse_frame`).
//!
//! **Backward compatibility**: image stego v0.x decoders use the v1
//! parse logic. v1 byte layout is unchanged. v2 frames cleanly fail
//! v0.x decoders (parser reads `[0x00, 0x00]` as `plaintext_len = 0`,
//! salt/nonce slots misalign, AES-GCM-SIV tag rejects → smart_decode
//! falls through). Video stego is pre-release (opt-in in v0.2.9), so
//! v2 lands greenfield — no v0.x video shadow stego exists in the
//! wild that depended on the old u32-only WIDE format.
//!
//! No magic byte beyond the v2 sentinel; AES-256-GCM-SIV
//! authentication is the only validator. Decoders brute-force
//! `(parity, fdl)` combinations using a first-block-peek heuristic
//! to derive `fdl` from the plaintext-length prefix once the first
//! 255-byte RS block decodes. Brute-force fdl range covers v1 only
//! (v2 frames are always ≥ 65586 bytes — handled exclusively by the
//! peek path).

use crate::stego::crypto::{NONCE_LEN, SALT_LEN};
use crate::stego::error::StegoError;
use crate::stego::payload::FileEntry;

/// One shadow layer's input — message + passphrase + optional file
/// attachments. The encoder takes a slice of these (size-descending
/// for primary-vs-shadow ordering by message size).
pub struct ShadowLayer<'a> {
    pub message: &'a str,
    pub passphrase: &'a str,
    pub files: &'a [FileEntry],
}

/// v1 frame overhead (`plaintext_len(2) + salt(16) + nonce(12) +
/// tag(16) = 46 bytes`). Picked when `plaintext_len ≤ u16::MAX`.
pub const SHADOW_FRAME_OVERHEAD_V1: usize = 2 + SALT_LEN + NONCE_LEN + 16;

/// v2 frame overhead (`sentinel(2) + plaintext_len(4) + salt(16) +
/// nonce(12) + tag(16) = 50 bytes`). Picked when `plaintext_len >
/// u16::MAX`. Sentinel = `[0x00, 0x00]` distinguishes from v1.
pub const SHADOW_FRAME_OVERHEAD_V2: usize = 2 + 4 + SALT_LEN + NONCE_LEN + 16;

/// Back-compat alias — points at v1 (the smallest, most common
/// shape). Callers that need the exact overhead for a given
/// plaintext size should use [`shadow_frame_overhead_for`].
pub const SHADOW_FRAME_OVERHEAD: usize = SHADOW_FRAME_OVERHEAD_V1;

/// RS parity tiers. Brute-forced at decode.
pub const SHADOW_PARITY_TIERS: [usize; 6] = [4, 8, 16, 32, 64, 128];

/// Maximum RS-encoded shadow frame bytes — bumped to 16 MB to
/// accommodate v2 frames with plausible large attachments
/// (e.g., embedded photos as shadows in long videos). Decoder
/// brute-force scan is bounded; this is the safety upper bound.
/// Image v1 frames are still naturally capped at ~64 KB by the u16
/// length field; v2 is what this larger bound exists for.
pub const MAX_SHADOW_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Compute the per-format header overhead for a given plaintext
/// length. Use when computing capacity / fdl ranges where the
/// caller already knows the plaintext size.
pub fn shadow_frame_overhead_for(plaintext_len: usize) -> usize {
    if plaintext_len > u16::MAX as usize {
        SHADOW_FRAME_OVERHEAD_V2
    } else {
        SHADOW_FRAME_OVERHEAD_V1
    }
}

/// Maximum shadow plaintext bytes that fit in `available_bytes` of
/// pre-RS frame space, accounting for v1 vs v2 dispatch. Picks
/// whichever format gives more capacity at that buffer size.
/// Use this when computing user-facing "max message size" — the
/// encoder will automatically pick the right format at build time.
pub fn max_shadow_plaintext_bytes(available_bytes: usize) -> usize {
    let v1_max = available_bytes
        .saturating_sub(SHADOW_FRAME_OVERHEAD_V1)
        .min(u16::MAX as usize);
    let v2_max = available_bytes.saturating_sub(SHADOW_FRAME_OVERHEAD_V2);
    v1_max.max(v2_max)
}

/// Build the shadow inner frame (before RS encoding). Picks v1 or
/// v2 layout based on `plaintext_len` (matches the `frame.rs`
/// primary-frame pattern).
pub fn build_shadow_frame(
    plaintext_len: usize,
    salt: &[u8; SALT_LEN],
    nonce: &[u8; NONCE_LEN],
    ciphertext: &[u8],
) -> Vec<u8> {
    assert!(
        plaintext_len <= u32::MAX as usize,
        "shadow frame plaintext exceeds u32::MAX",
    );
    debug_assert_eq!(
        ciphertext.len(),
        plaintext_len + 16,
        "ciphertext length must equal plaintext_len + 16 (AES-GCM-SIV tag)",
    );

    let is_v2 = plaintext_len > u16::MAX as usize;
    let header_len = if is_v2 { 6 } else { 2 };
    let mut fr = Vec::with_capacity(header_len + SALT_LEN + NONCE_LEN + ciphertext.len());

    if is_v2 {
        fr.extend_from_slice(&0u16.to_be_bytes());
        fr.extend_from_slice(&(plaintext_len as u32).to_be_bytes());
    } else {
        fr.extend_from_slice(&(plaintext_len as u16).to_be_bytes());
    }
    fr.extend_from_slice(salt);
    fr.extend_from_slice(nonce);
    fr.extend_from_slice(ciphertext);
    fr
}

/// Parsed shadow frame — output of [`parse_shadow_frame`].
/// `plaintext_len` is u32 to cover both v1 (≤ u16::MAX) and v2 (full
/// u32 range). `header_overhead` tells callers which layout was
/// decoded (46 = v1, 50 = v2) — used by the brute-force consistency
/// gate to cross-check the candidate `fdl`.
pub struct ParsedShadowFrame {
    pub plaintext_len: u32,
    pub salt: [u8; SALT_LEN],
    pub nonce: [u8; NONCE_LEN],
    pub ciphertext: Vec<u8>,
    /// Bytes consumed by the layout (46 for v1, 50 for v2). Equals
    /// `total_frame_len - plaintext_len - 16(tag)`.
    pub header_overhead: usize,
}

/// Parse a shadow inner frame (after RS decoding). Dispatches v1
/// vs v2 on the first 2-6 bytes (same logic as
/// `frame::parse_frame`).
pub fn parse_shadow_frame(data: &[u8]) -> Result<ParsedShadowFrame, StegoError> {
    if data.len() < SHADOW_FRAME_OVERHEAD_V1 {
        return Err(StegoError::FrameCorrupted);
    }

    let header_u16 = u16::from_be_bytes([data[0], data[1]]);
    let (plaintext_len, header_overhead): (usize, usize) = if header_u16 == 0 && data.len() >= 6 {
        let v2_len = u32::from_be_bytes([data[2], data[3], data[4], data[5]]) as usize;
        if v2_len > u16::MAX as usize {
            (v2_len, SHADOW_FRAME_OVERHEAD_V2)
        } else {
            (0, SHADOW_FRAME_OVERHEAD_V1)
        }
    } else {
        (header_u16 as usize, SHADOW_FRAME_OVERHEAD_V1)
    };

    let expected_len = header_overhead + plaintext_len;
    if data.len() < expected_len {
        return Err(StegoError::FrameCorrupted);
    }

    let header_len = header_overhead - SALT_LEN - NONCE_LEN - 16;
    let mut salt = [0u8; SALT_LEN];
    salt.copy_from_slice(&data[header_len..header_len + SALT_LEN]);
    let mut nonce = [0u8; NONCE_LEN];
    nonce.copy_from_slice(&data[header_len + SALT_LEN..header_len + SALT_LEN + NONCE_LEN]);
    let ciphertext = data[header_len + SALT_LEN + NONCE_LEN..expected_len].to_vec();
    Ok(ParsedShadowFrame {
        plaintext_len: plaintext_len as u32,
        salt,
        nonce,
        ciphertext,
        header_overhead,
    })
}

/// Peek at the first 6 bytes of a candidate RS-decoded shadow frame
/// to derive `fdl` (total frame length) without doing the full
/// parse. Returns `None` if the data is too short or the format
/// can't be determined. Caller is responsible for plausibility
/// bounds (`fdl <= max_fdl`).
///
/// This is the first half of the decoder fast-path: peek → compute
/// fdl → run one RS-decode at that fdl → AES-GCM-SIV verify. The
/// consistency gate inside [`parse_shadow_frame`] catches mismatch
/// when the brute-force tries the wrong fdl.
pub fn peek_shadow_fdl(data: &[u8]) -> Option<usize> {
    if data.len() < 2 {
        return None;
    }
    let header_u16 = u16::from_be_bytes([data[0], data[1]]);
    if header_u16 == 0 && data.len() >= 6 {
        let v2_len = u32::from_be_bytes([data[2], data[3], data[4], data[5]]) as usize;
        if v2_len > u16::MAX as usize {
            return Some(SHADOW_FRAME_OVERHEAD_V2 + v2_len);
        }
        // header_u16 == 0 + v2_len <= u16::MAX = legitimate v1 with
        // plaintext_len = 0. Empty shadows are pointless but
        // structurally valid.
        return Some(SHADOW_FRAME_OVERHEAD_V1);
    }
    Some(SHADOW_FRAME_OVERHEAD_V1 + header_u16 as usize)
}

/// Compute the maximum `frame_data_len` (bytes before RS encoding)
/// that fits in `max_rs_bytes` of available LSB capacity at the
/// given parity length.
pub fn compute_max_shadow_fdl(max_rs_bytes: usize, parity_len: usize) -> usize {
    let k = 255usize.saturating_sub(parity_len);
    if k == 0 || max_rs_bytes == 0 {
        return 0;
    }
    let full_blocks = max_rs_bytes / 255;
    let remainder = max_rs_bytes % 255;
    let mut max_data = full_blocks * k;
    if remainder > parity_len {
        max_data += remainder - parity_len;
    }
    max_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_v1_roundtrip_small() {
        let salt = [1u8; SALT_LEN];
        let nonce = [2u8; NONCE_LEN];
        let ciphertext = vec![0xAAu8; 20];
        let fr = build_shadow_frame(4, &salt, &nonce, &ciphertext);
        assert_eq!(fr.len(), SHADOW_FRAME_OVERHEAD_V1 + 4);
        let parsed = parse_shadow_frame(&fr).unwrap();
        assert_eq!(parsed.plaintext_len, 4);
        assert_eq!(parsed.salt, salt);
        assert_eq!(parsed.nonce, nonce);
        assert_eq!(parsed.ciphertext, ciphertext);
        assert_eq!(parsed.header_overhead, SHADOW_FRAME_OVERHEAD_V1);
    }

    #[test]
    fn frame_v1_roundtrip_max_u16() {
        // Boundary: plaintext_len == u16::MAX stays v1
        let salt = [11u8; SALT_LEN];
        let nonce = [12u8; NONCE_LEN];
        let ciphertext = vec![0xCCu8; u16::MAX as usize + 16];
        let fr = build_shadow_frame(u16::MAX as usize, &salt, &nonce, &ciphertext);
        assert_eq!(fr.len(), SHADOW_FRAME_OVERHEAD_V1 + u16::MAX as usize);
        let parsed = parse_shadow_frame(&fr).unwrap();
        assert_eq!(parsed.plaintext_len, u16::MAX as u32);
        assert_eq!(parsed.header_overhead, SHADOW_FRAME_OVERHEAD_V1);
    }

    #[test]
    fn frame_v2_roundtrip_first_byte_above_u16() {
        // Boundary: plaintext_len == u16::MAX + 1 switches to v2
        let salt = [21u8; SALT_LEN];
        let nonce = [22u8; NONCE_LEN];
        let pt_len = u16::MAX as usize + 1;
        let ciphertext = vec![0xDDu8; pt_len + 16];
        let fr = build_shadow_frame(pt_len, &salt, &nonce, &ciphertext);
        assert_eq!(fr.len(), SHADOW_FRAME_OVERHEAD_V2 + pt_len);
        assert_eq!(&fr[..2], &[0x00, 0x00], "v2 sentinel must be 0x0000");
        let parsed = parse_shadow_frame(&fr).unwrap();
        assert_eq!(parsed.plaintext_len, pt_len as u32);
        assert_eq!(parsed.salt, salt);
        assert_eq!(parsed.nonce, nonce);
        assert_eq!(parsed.ciphertext, ciphertext);
        assert_eq!(parsed.header_overhead, SHADOW_FRAME_OVERHEAD_V2);
    }

    #[test]
    fn frame_v2_roundtrip_large() {
        // 200 KB plaintext — comfortably v2
        let salt = [31u8; SALT_LEN];
        let nonce = [32u8; NONCE_LEN];
        let pt_len = 200_000usize;
        let ciphertext = vec![0xEEu8; pt_len + 16];
        let fr = build_shadow_frame(pt_len, &salt, &nonce, &ciphertext);
        let parsed = parse_shadow_frame(&fr).unwrap();
        assert_eq!(parsed.plaintext_len, pt_len as u32);
        assert_eq!(parsed.header_overhead, SHADOW_FRAME_OVERHEAD_V2);
        assert_eq!(parsed.ciphertext.len(), pt_len + 16);
    }

    #[test]
    fn frame_v1_with_zero_plaintext_len_low_salt() {
        // Legitimate v1 edge case: plaintext_len = 0 with a salt
        // whose first 4 bytes interpret as u32 ≤ u16::MAX. The v2
        // disambiguator's `v2_len > u16::MAX` tie-breaker correctly
        // falls back to v1 with plaintext_len = 0.
        let salt = [0u8; SALT_LEN]; // first 4 bytes = 0x00000000
        let nonce = [43u8; NONCE_LEN];
        let ciphertext = vec![0u8; 16]; // tag only, no plaintext
        let fr = build_shadow_frame(0, &salt, &nonce, &ciphertext);
        assert_eq!(fr.len(), SHADOW_FRAME_OVERHEAD_V1);
        let parsed = parse_shadow_frame(&fr).unwrap();
        assert_eq!(parsed.plaintext_len, 0);
        assert_eq!(parsed.header_overhead, SHADOW_FRAME_OVERHEAD_V1);
        assert_eq!(parsed.salt, salt);
    }

    #[test]
    fn frame_v1_zero_plaintext_high_salt_is_known_edge_case() {
        // **Known edge case** (identical behavior to `frame.rs`
        // primary-frame parser): a v1 frame with plaintext_len=0
        // AND salt[0..4] interpreting as u32 > u16::MAX will be
        // mis-dispatched as v2 by the sentinel disambiguator. The
        // resulting `expected_len` won't match `data.len()` and
        // parse fails with `FrameCorrupted`. Documented here so
        // future readers don't think this is a bug.
        //
        // In practice the encoder never emits plaintext_len=0
        // (empty shadow messages are rejected upstream), so this
        // code path is unreachable from real callers.
        let salt = [0x42u8; SALT_LEN]; // 0x42424242 > u16::MAX
        let nonce = [43u8; NONCE_LEN];
        let ciphertext = vec![0u8; 16];
        let fr = build_shadow_frame(0, &salt, &nonce, &ciphertext);
        // Built frame is structurally valid v1, but the parser
        // mis-dispatches → returns FrameCorrupted.
        assert!(parse_shadow_frame(&fr).is_err());
    }

    #[test]
    fn overheads_match_spec() {
        assert_eq!(SHADOW_FRAME_OVERHEAD_V1, 46);
        assert_eq!(SHADOW_FRAME_OVERHEAD_V2, 50);
        assert_eq!(SHADOW_FRAME_OVERHEAD_V2 - SHADOW_FRAME_OVERHEAD_V1, 4);
        assert_eq!(SHADOW_FRAME_OVERHEAD, SHADOW_FRAME_OVERHEAD_V1);
    }

    #[test]
    fn shadow_frame_overhead_for_sizes() {
        assert_eq!(shadow_frame_overhead_for(0), SHADOW_FRAME_OVERHEAD_V1);
        assert_eq!(shadow_frame_overhead_for(1), SHADOW_FRAME_OVERHEAD_V1);
        assert_eq!(
            shadow_frame_overhead_for(u16::MAX as usize),
            SHADOW_FRAME_OVERHEAD_V1
        );
        assert_eq!(
            shadow_frame_overhead_for(u16::MAX as usize + 1),
            SHADOW_FRAME_OVERHEAD_V2
        );
        assert_eq!(shadow_frame_overhead_for(1_000_000), SHADOW_FRAME_OVERHEAD_V2);
    }

    #[test]
    fn peek_v1() {
        let salt = [1u8; SALT_LEN];
        let nonce = [2u8; NONCE_LEN];
        let ciphertext = vec![0xAAu8; 200 + 16];
        let fr = build_shadow_frame(200, &salt, &nonce, &ciphertext);
        // Pass at least the header bytes to peek.
        let fdl = peek_shadow_fdl(&fr).unwrap();
        assert_eq!(fdl, SHADOW_FRAME_OVERHEAD_V1 + 200);
    }

    #[test]
    fn peek_v2() {
        let salt = [1u8; SALT_LEN];
        let nonce = [2u8; NONCE_LEN];
        let pt_len = 100_000usize;
        let ciphertext = vec![0xBBu8; pt_len + 16];
        let fr = build_shadow_frame(pt_len, &salt, &nonce, &ciphertext);
        let fdl = peek_shadow_fdl(&fr[..6]).unwrap();
        assert_eq!(fdl, SHADOW_FRAME_OVERHEAD_V2 + pt_len);
    }

    #[test]
    fn peek_too_short() {
        // Only 1 byte — can't determine format
        assert!(peek_shadow_fdl(&[0x00]).is_none());
        // 2 bytes is enough for v1
        assert!(peek_shadow_fdl(&[0x00, 0x05]).is_some());
        // 2 bytes of 0x0000 + insufficient for v2 — falls back to v1 with len=0
        assert_eq!(peek_shadow_fdl(&[0x00, 0x00]).unwrap(), SHADOW_FRAME_OVERHEAD_V1);
    }

    /// v2 frame fed to a hypothetical v0.x decoder (one that does
    /// `u16::from_be_bytes` blindly): reads `plaintext_len = 0`,
    /// expects 46-byte total frame. Frame is actually 50+ bytes →
    /// `data.len() >= 46` succeeds, but salt/nonce slots are
    /// misaligned. Downstream AES-GCM-SIV tag check rejects.
    ///
    /// This test simulates that path: parse a v2 frame using the
    /// OLD u16-only logic and confirm the salt slot ends up wrong.
    #[test]
    fn v0x_decoder_fails_cleanly_on_v2_frame() {
        let salt = [42u8; SALT_LEN];
        let nonce = [43u8; NONCE_LEN];
        let pt_len = 100_000usize;
        let ciphertext = vec![0xCCu8; pt_len + 16];
        let fr = build_shadow_frame(pt_len, &salt, &nonce, &ciphertext);

        // Simulate the old u16-only parser:
        let old_plaintext_len = u16::from_be_bytes([fr[0], fr[1]]) as usize;
        // Old parser sees plaintext_len = 0 (the v2 sentinel)
        assert_eq!(old_plaintext_len, 0);
        let old_expected_len = SHADOW_FRAME_OVERHEAD_V1 + old_plaintext_len; // 46
        // Buffer is large enough, so the old length check passes:
        assert!(fr.len() >= old_expected_len);
        // But the salt slot would read 4 bytes too early (bytes 2..18
        // instead of 6..22), so the "salt" the old parser extracts
        // is actually [u32_len_high_bytes, salt[0..14]]:
        let old_salt = &fr[2..2 + SALT_LEN];
        assert_ne!(
            old_salt, &salt[..],
            "old decoder's salt slot is misaligned vs the real salt",
        );
        // Result: downstream AES-GCM-SIV would derive the wrong key
        // (wrong salt → wrong Argon2 output) and reject the tag.
        // Clean failure, no garbage decode.
    }

    #[test]
    fn fdl_capacity_arithmetic() {
        assert_eq!(compute_max_shadow_fdl(0, 4), 0);
        assert_eq!(compute_max_shadow_fdl(255, 4), 251);
        assert_eq!(compute_max_shadow_fdl(510, 4), 502);
        // remainder shorter than parity: rounds down to full blocks only.
        assert_eq!(compute_max_shadow_fdl(258, 4), 251);
    }
}
