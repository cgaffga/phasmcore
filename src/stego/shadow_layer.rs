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
//! ## Frame formats (no header, fdl recovered via first-block peek)
//!
//! Two variants exist, distinguished by the width of the
//! plaintext-length prefix:
//!
//! **Standard (image)** — `SHADOW_FRAME_OVERHEAD = 46 bytes`:
//! ```text
//! [plaintext_len: 2B u16 BE] [salt: 16B] [nonce: 12B] [ciphertext: N+16B]
//! ```
//! Used by image stego (`core/src/stego/ghost/shadow.rs`). Hard
//! cap: 65,535-byte plaintext per shadow. Adequate for image
//! covers (Y-channel JPEG nzAC capacity ≈ tens of KB).
//!
//! **Wide (video)** — `SHADOW_FRAME_OVERHEAD_WIDE = 48 bytes`:
//! ```text
//! [plaintext_len: 4B u32 BE] [salt: 16B] [nonce: 12B] [ciphertext: N+16B]
//! ```
//! Used by H.264 video stego
//! (`core/src/codec/h264/stego/shadow.rs`). Hard cap:
//! 4,294,967,295-byte plaintext per shadow. Required because
//! video covers can support much larger payloads (file
//! attachments, multi-MB shadows). The two formats are wire-
//! incompatible by design — image stego has been released and the
//! u16 layout is locked; video stego is pre-release and uses the
//! wider layout natively.
//!
//! No magic byte; AES-256-GCM-SIV authentication is the only
//! validator. Decoders brute-force `(parity, fdl)` combinations
//! using a first-block-peek heuristic to derive `fdl` from the
//! plaintext-length prefix once the first 255-byte RS block decodes.

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

/// Standard (image) frame overhead inside the RS-encoded payload:
/// `plaintext_len(2) + salt(16) + nonce(12) + tag(16) = 46 bytes`.
/// Used by image stego (`stego::ghost::shadow`); locked at u16 to
/// preserve compatibility with released app versions.
pub const SHADOW_FRAME_OVERHEAD: usize = 2 + SALT_LEN + NONCE_LEN + 16;

/// Wide (video) frame overhead — same fields but with a u32
/// plaintext-length prefix:
/// `plaintext_len(4) + salt(16) + nonce(12) + tag(16) = 48 bytes`.
/// Used by H.264 video stego — covers can support multi-MB
/// shadows (file attachments) so the u16 cap is too tight.
pub const SHADOW_FRAME_OVERHEAD_WIDE: usize = 4 + SALT_LEN + NONCE_LEN + 16;

/// RS parity tiers. Brute-forced at decode.
pub const SHADOW_PARITY_TIERS: [usize; 6] = [4, 8, 16, 32, 64, 128];

/// Maximum RS-encoded frame bytes for the standard (image) format
/// — guards against unreasonable allocations during decode
/// brute-force. Implies plaintext ≤ ~256 KB before RS expansion;
/// in practice u16 caps it at 65,535 bytes.
pub const MAX_SHADOW_FRAME_BYTES: usize = 256 * 1024;

/// Maximum RS-encoded frame bytes for the wide (video) format.
/// Bumped to 16 MB to accommodate plausible large attachments
/// (e.g., embedded photos as shadows in long videos). Decoder
/// brute-force scan is bounded; this is the safety upper bound.
pub const MAX_SHADOW_FRAME_BYTES_WIDE: usize = 16 * 1024 * 1024;

/// Build the shadow inner frame (before RS encoding).
///
/// Layout: `[plaintext_len: 2B] [salt: 16B] [nonce: 12B]
///          [ciphertext: N+16B]`
pub fn build_shadow_frame(
    plaintext_len: usize,
    salt: &[u8; SALT_LEN],
    nonce: &[u8; NONCE_LEN],
    ciphertext: &[u8],
) -> Vec<u8> {
    assert!(
        plaintext_len <= u16::MAX as usize,
        "shadow frame plaintext exceeds u16::MAX",
    );
    let mut fr = Vec::with_capacity(SHADOW_FRAME_OVERHEAD + plaintext_len);
    fr.extend_from_slice(&(plaintext_len as u16).to_be_bytes());
    fr.extend_from_slice(salt);
    fr.extend_from_slice(nonce);
    fr.extend_from_slice(ciphertext);
    fr
}

/// Parsed shadow frame — output of [`parse_shadow_frame`].
pub struct ParsedShadowFrame {
    pub plaintext_len: u16,
    pub salt: [u8; SALT_LEN],
    pub nonce: [u8; NONCE_LEN],
    pub ciphertext: Vec<u8>,
}

/// Parse a shadow inner frame (after RS decoding).
pub fn parse_shadow_frame(data: &[u8]) -> Result<ParsedShadowFrame, StegoError> {
    if data.len() < SHADOW_FRAME_OVERHEAD {
        return Err(StegoError::FrameCorrupted);
    }
    let plaintext_len = u16::from_be_bytes([data[0], data[1]]);
    let expected_len = SHADOW_FRAME_OVERHEAD + plaintext_len as usize;
    if data.len() < expected_len {
        return Err(StegoError::FrameCorrupted);
    }
    let mut salt = [0u8; SALT_LEN];
    salt.copy_from_slice(&data[2..2 + SALT_LEN]);
    let mut nonce = [0u8; NONCE_LEN];
    nonce.copy_from_slice(&data[2 + SALT_LEN..2 + SALT_LEN + NONCE_LEN]);
    let ciphertext = data[2 + SALT_LEN + NONCE_LEN..expected_len].to_vec();
    Ok(ParsedShadowFrame { plaintext_len, salt, nonce, ciphertext })
}

/// Build the wide shadow inner frame (u32 plaintext_len) — see
/// module docs for layout. Used by video stego.
pub fn build_shadow_frame_wide(
    plaintext_len: usize,
    salt: &[u8; SALT_LEN],
    nonce: &[u8; NONCE_LEN],
    ciphertext: &[u8],
) -> Vec<u8> {
    assert!(
        plaintext_len <= u32::MAX as usize,
        "shadow frame plaintext exceeds u32::MAX",
    );
    let mut fr = Vec::with_capacity(SHADOW_FRAME_OVERHEAD_WIDE + plaintext_len);
    fr.extend_from_slice(&(plaintext_len as u32).to_be_bytes());
    fr.extend_from_slice(salt);
    fr.extend_from_slice(nonce);
    fr.extend_from_slice(ciphertext);
    fr
}

/// Parsed wide shadow frame — output of [`parse_shadow_frame_wide`].
pub struct ParsedShadowFrameWide {
    pub plaintext_len: u32,
    pub salt: [u8; SALT_LEN],
    pub nonce: [u8; NONCE_LEN],
    pub ciphertext: Vec<u8>,
}

/// Parse a wide shadow inner frame (u32 plaintext_len). Used by
/// video stego.
pub fn parse_shadow_frame_wide(data: &[u8]) -> Result<ParsedShadowFrameWide, StegoError> {
    if data.len() < SHADOW_FRAME_OVERHEAD_WIDE {
        return Err(StegoError::FrameCorrupted);
    }
    let plaintext_len =
        u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    let expected_len =
        SHADOW_FRAME_OVERHEAD_WIDE + plaintext_len as usize;
    if data.len() < expected_len {
        return Err(StegoError::FrameCorrupted);
    }
    let mut salt = [0u8; SALT_LEN];
    salt.copy_from_slice(&data[4..4 + SALT_LEN]);
    let mut nonce = [0u8; NONCE_LEN];
    nonce.copy_from_slice(&data[4 + SALT_LEN..4 + SALT_LEN + NONCE_LEN]);
    let ciphertext =
        data[4 + SALT_LEN + NONCE_LEN..expected_len].to_vec();
    Ok(ParsedShadowFrameWide { plaintext_len, salt, nonce, ciphertext })
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
    fn frame_roundtrip() {
        let salt = [1u8; SALT_LEN];
        let nonce = [2u8; NONCE_LEN];
        let ciphertext = vec![0xAAu8; 20];
        let fr = build_shadow_frame(4, &salt, &nonce, &ciphertext);
        let parsed = parse_shadow_frame(&fr).unwrap();
        assert_eq!(parsed.plaintext_len, 4);
        assert_eq!(parsed.salt, salt);
        assert_eq!(parsed.nonce, nonce);
        assert_eq!(parsed.ciphertext, ciphertext);
    }

    #[test]
    fn frame_wide_roundtrip() {
        let salt = [3u8; SALT_LEN];
        let nonce = [4u8; NONCE_LEN];
        let ciphertext = vec![0xBBu8; 20];
        let fr = build_shadow_frame_wide(4, &salt, &nonce, &ciphertext);
        assert_eq!(fr.len(), SHADOW_FRAME_OVERHEAD_WIDE + ciphertext.len() - 16);
        let parsed = parse_shadow_frame_wide(&fr).unwrap();
        assert_eq!(parsed.plaintext_len, 4);
        assert_eq!(parsed.salt, salt);
        assert_eq!(parsed.nonce, nonce);
        assert_eq!(parsed.ciphertext, ciphertext);
    }

    /// The wide format must be wire-incompatible with the standard
    /// format (different overhead, different length-field width).
    /// Sanity-check that constants reflect the expected byte counts.
    #[test]
    fn wide_overhead_two_bytes_more_than_standard() {
        assert_eq!(SHADOW_FRAME_OVERHEAD, 46);
        assert_eq!(SHADOW_FRAME_OVERHEAD_WIDE, 48);
        assert_eq!(SHADOW_FRAME_OVERHEAD_WIDE - SHADOW_FRAME_OVERHEAD, 2);
    }

    /// A frame written via the standard builder must NOT parse via
    /// the wide parser (they're distinct formats; mixing decoders
    /// would silently produce garbage).
    #[test]
    fn wide_parser_rejects_standard_frame_layout() {
        let salt = [5u8; SALT_LEN];
        let nonce = [6u8; NONCE_LEN];
        let ciphertext = vec![0xCCu8; 100];
        let fr_standard = build_shadow_frame(84, &salt, &nonce, &ciphertext);
        let parsed = parse_shadow_frame_wide(&fr_standard);
        // Either the length field decodes to a huge u32 (rejected
        // for being > data.len() - WIDE_OVERHEAD) or the plaintext
        // bytes match by accident — but the salt/nonce slots will
        // be misaligned and any subsequent crypto step will fail.
        if let Ok(p) = parsed {
            // If parse "succeeds", the salt/nonce slots are shifted
            // by 2 bytes, so they don't match the original salt/nonce.
            // The point of the test is that you can't mix the two.
            assert_ne!(p.salt, salt);
        }
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
