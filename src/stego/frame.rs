// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Payload frame construction and parsing.
//!
//! The frame is the binary container that wraps the encrypted message before
//! embedding into DCT coefficients. Both Ghost and Armor modes use the same
//! frame format:
//!
//! ```text
//! [2 bytes ] plaintext length (big-endian u16)
//! [16 bytes] Argon2 salt (for Tier-2 key derivation)
//! [12 bytes] AES-GCM-SIV nonce
//! [N bytes ] ciphertext (plaintext_len + 16 bytes for auth tag)
//! [4 bytes ] CRC-32 of everything above
//! ```
//!
//! Total frame size = 50 + plaintext_len bytes.
//!
//! Note: The mode byte was removed from the frame format to eliminate known
//! plaintext and improve stealth. Mode detection is handled by trial decoding
//! in `smart_decode`.

use crate::stego::crypto::{NONCE_LEN, SALT_LEN};
use crate::stego::error::StegoError;

/// Ghost mode identifier byte.
pub const MODE_GHOST: u8 = 0x01;
/// Armor mode identifier byte.
pub const MODE_ARMOR: u8 = 0x02;

/// Fixed overhead: length(2) + salt(16) + nonce(12) + tag(16) + crc(4) = 50 bytes.
/// Plus the ciphertext length equals plaintext length (AES-GCM stream cipher).
/// So total frame = 34 + plaintext_len + 16(tag) = 50 + plaintext_len.
pub const FRAME_OVERHEAD: usize = 2 + SALT_LEN + NONCE_LEN + 16 + 4; // 50

/// Maximum payload frame size in bytes.
/// The u16 length prefix supports up to 65,535 bytes of plaintext.
/// 65,535 (max plaintext from u16) + 50 (overhead) = 65,585 bytes.
/// The actual usable capacity is determined by each image's embedding capacity.
pub const MAX_FRAME_BYTES: usize = 65_535 + FRAME_OVERHEAD; // 65,585

/// Maximum payload frame size in bits.
pub const MAX_FRAME_BITS: usize = MAX_FRAME_BYTES * 8;

/// Build a payload frame from encrypted components.
///
/// Frame layout:
/// ```text
/// [2 bytes ] plaintext length (BE u16)
/// [16 bytes] Argon2 salt
/// [12 bytes] AES-GCM nonce
/// [N bytes ] AES-GCM ciphertext (includes 16-byte auth tag)
/// [4 bytes ] CRC-32 of everything above
/// ```
pub fn build_frame(
    plaintext_len: u16,
    salt: &[u8; SALT_LEN],
    nonce: &[u8; NONCE_LEN],
    ciphertext: &[u8],
) -> Vec<u8> {
    debug_assert_eq!(ciphertext.len(), plaintext_len as usize + 16, "ciphertext length mismatch");

    let mut frame = Vec::with_capacity(2 + SALT_LEN + NONCE_LEN + ciphertext.len() + 4);

    frame.extend_from_slice(&plaintext_len.to_be_bytes());
    frame.extend_from_slice(salt);
    frame.extend_from_slice(nonce);
    frame.extend_from_slice(ciphertext);

    let crc = crc32fast::hash(&frame);
    frame.extend_from_slice(&crc.to_be_bytes());

    frame
}

/// Parsed payload frame.
///
/// Contains all fields needed to decrypt the embedded message:
/// the original plaintext length, the Argon2 salt and AES-GCM-SIV nonce
/// for key derivation/decryption, and the ciphertext (including auth tag).
pub struct ParsedFrame {
    /// Original plaintext length in bytes (before encryption).
    pub plaintext_len: u16,
    /// Argon2 salt for Tier-2 encryption key derivation.
    pub salt: [u8; SALT_LEN],
    /// AES-GCM-SIV nonce.
    pub nonce: [u8; NONCE_LEN],
    /// Ciphertext including 16-byte authentication tag.
    pub ciphertext: Vec<u8>,
}

/// Parse a payload frame, verifying the CRC.
///
/// The input `data` may be larger than the actual frame (e.g. zero-padded).
/// The actual frame length is determined from the embedded `plaintext_len` field.
///
/// Returns `Err(StegoError::FrameCorrupted)` if the CRC check fails or the
/// frame is truncated.
pub fn parse_frame(data: &[u8]) -> Result<ParsedFrame, StegoError> {
    // Need at least 2 bytes to read plaintext_len.
    if data.len() < 2 {
        return Err(StegoError::FrameCorrupted);
    }

    // Read plaintext_len to compute the actual frame size.
    let plaintext_len = u16::from_be_bytes([data[0], data[1]]);
    let ciphertext_len = plaintext_len as usize + 16; // AES-GCM-SIV auth tag
    let total_frame_len = 2 + SALT_LEN + NONCE_LEN + ciphertext_len + 4;

    // Reject frames that exceed the maximum supported size.
    if total_frame_len > MAX_FRAME_BYTES {
        return Err(StegoError::FrameCorrupted);
    }

    if data.len() < total_frame_len {
        return Err(StegoError::FrameCorrupted);
    }

    // Verify CRC at the correct position within the frame.
    let payload = &data[..total_frame_len - 4];
    let crc_bytes = &data[total_frame_len - 4..total_frame_len];
    let stored_crc = u32::from_be_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
    let computed_crc = crc32fast::hash(payload);
    if stored_crc != computed_crc {
        return Err(StegoError::FrameCorrupted);
    }

    // Parse fields.
    let mut salt = [0u8; SALT_LEN];
    salt.copy_from_slice(&payload[2..2 + SALT_LEN]);

    let mut nonce = [0u8; NONCE_LEN];
    nonce.copy_from_slice(&payload[2 + SALT_LEN..2 + SALT_LEN + NONCE_LEN]);

    let ciphertext = payload[2 + SALT_LEN + NONCE_LEN..].to_vec();

    Ok(ParsedFrame {
        plaintext_len,
        salt,
        nonce,
        ciphertext,
    })
}

/// Compact frame overhead for Fortress empty-passphrase mode.
/// Salt and nonce are omitted (derived from constants on both sides).
/// Layout: length(2) + ciphertext(N+16) + crc(4) = 22 + plaintext_len.
pub const FORTRESS_COMPACT_FRAME_OVERHEAD: usize = 2 + 16 + 4; // 22

/// Build a compact fortress frame (no salt, no nonce embedded).
///
/// Frame layout:
/// ```text
/// [2 bytes ] plaintext length (BE u16)
/// [N bytes ] AES-GCM ciphertext (includes 16-byte auth tag)
/// [4 bytes ] CRC-32 of everything above
/// ```
pub fn build_fortress_compact_frame(
    plaintext_len: u16,
    ciphertext: &[u8],
) -> Vec<u8> {
    let mut frame = Vec::with_capacity(2 + ciphertext.len() + 4);

    frame.extend_from_slice(&plaintext_len.to_be_bytes());
    frame.extend_from_slice(ciphertext);

    let crc = crc32fast::hash(&frame);
    frame.extend_from_slice(&crc.to_be_bytes());

    frame
}

/// Parse a compact fortress frame, verifying the CRC.
///
/// Returns a `ParsedFrame` with `salt` and `nonce` set to the known Fortress
/// empty-passphrase constants (the caller should use them for decryption).
pub fn parse_fortress_compact_frame(data: &[u8]) -> Result<ParsedFrame, StegoError> {
    use crate::stego::crypto::{FORTRESS_EMPTY_SALT, FORTRESS_EMPTY_NONCE};

    if data.len() < 2 {
        return Err(StegoError::FrameCorrupted);
    }

    let plaintext_len = u16::from_be_bytes([data[0], data[1]]);
    let ciphertext_len = plaintext_len as usize + 16;
    let total_frame_len = 2 + ciphertext_len + 4;

    if total_frame_len > MAX_FRAME_BYTES {
        return Err(StegoError::FrameCorrupted);
    }

    if data.len() < total_frame_len {
        return Err(StegoError::FrameCorrupted);
    }

    let payload = &data[..total_frame_len - 4];
    let crc_bytes = &data[total_frame_len - 4..total_frame_len];
    let stored_crc = u32::from_be_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
    let computed_crc = crc32fast::hash(payload);
    if stored_crc != computed_crc {
        return Err(StegoError::FrameCorrupted);
    }

    let ciphertext = payload[2..].to_vec();

    Ok(ParsedFrame {
        plaintext_len,
        salt: FORTRESS_EMPTY_SALT,
        nonce: FORTRESS_EMPTY_NONCE,
        ciphertext,
    })
}

/// Convert bytes to a bit vector (MSB first within each byte).
pub fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(bytes.len() * 8);
    for &byte in bytes {
        for bit_pos in (0..8).rev() {
            bits.push((byte >> bit_pos) & 1);
        }
    }
    bits
}

/// Convert a bit vector (MSB first) back to bytes.
/// Pads the last byte with zero bits if `bits.len()` is not a multiple of 8.
pub fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((bits.len() + 7) / 8);
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            byte |= (bit & 1) << (7 - i);
        }
        bytes.push(byte);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_parse_roundtrip() {
        let salt = [1u8; SALT_LEN];
        let nonce = [2u8; NONCE_LEN];
        // plaintext_len=2, so ciphertext must be 2+16=18 bytes (AES-GCM tag).
        let ciphertext = vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
                              0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
                              0x77, 0x88, 0x99, 0x00, 0xAA, 0xBB];
        let frame = build_frame(2, &salt, &nonce, &ciphertext);
        let parsed = parse_frame(&frame).unwrap();

        assert_eq!(parsed.plaintext_len, 2);
        assert_eq!(parsed.salt, salt);
        assert_eq!(parsed.nonce, nonce);
        assert_eq!(parsed.ciphertext, ciphertext);
    }

    #[test]
    fn corrupted_crc_detected() {
        let salt = [0u8; SALT_LEN];
        let nonce = [0u8; NONCE_LEN];
        // plaintext_len=4, ciphertext=4+16=20 bytes.
        let ciphertext = vec![0u8; 20];
        let mut frame = build_frame(4, &salt, &nonce, &ciphertext);
        // Corrupt last byte (CRC).
        let len = frame.len();
        frame[len - 1] ^= 0xFF;
        assert!(matches!(parse_frame(&frame), Err(StegoError::FrameCorrupted)));
    }

    #[test]
    fn corrupted_length_detected() {
        let salt = [0u8; SALT_LEN];
        let nonce = [0u8; NONCE_LEN];
        // plaintext_len=4, ciphertext=4+16=20 bytes.
        let ciphertext = vec![0u8; 20];
        let mut frame = build_frame(4, &salt, &nonce, &ciphertext);
        // Corrupt the plaintext_len field (byte 0) without updating CRC.
        frame[0] = 0xFF;
        assert!(matches!(parse_frame(&frame), Err(StegoError::FrameCorrupted)));
    }

    #[test]
    fn bytes_bits_roundtrip() {
        let original = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let bits = bytes_to_bits(&original);
        assert_eq!(bits.len(), 32);
        let recovered = bits_to_bytes(&bits);
        assert_eq!(recovered, original);
    }

    #[test]
    fn truncated_data_rejected() {
        // Too short to even read plaintext_len
        assert!(matches!(parse_frame(&[0x00]), Err(StegoError::FrameCorrupted)));
        assert!(matches!(parse_frame(&[]), Err(StegoError::FrameCorrupted)));
    }

    #[test]
    fn frame_no_mode_byte() {
        // Verify the frame format has no mode byte -- the frame starts with
        // plaintext_len (2 bytes), so a frame for plaintext_len=4 should
        // start with [0x00, 0x04].
        let salt = [3u8; SALT_LEN];
        let nonce = [4u8; NONCE_LEN];
        let ciphertext = vec![0x55u8; 20]; // plaintext_len=4
        let frame = build_frame(4, &salt, &nonce, &ciphertext);

        // First two bytes are plaintext_len in big-endian
        assert_eq!(frame[0], 0x00);
        assert_eq!(frame[1], 0x04);

        // Total size: 2 + 16 + 12 + 20 + 4 = 54
        assert_eq!(frame.len(), 2 + SALT_LEN + NONCE_LEN + 20 + 4);

        // Roundtrip should work
        let parsed = parse_frame(&frame).unwrap();
        assert_eq!(parsed.plaintext_len, 4);
        assert_eq!(parsed.salt, salt);
        assert_eq!(parsed.nonce, nonce);
        assert_eq!(parsed.ciphertext, ciphertext);
    }

    #[test]
    fn frame_with_zero_length_data() {
        let salt = [0u8; SALT_LEN];
        let nonce = [0u8; NONCE_LEN];
        // plaintext_len=0, ciphertext=0+16=16 bytes (auth tag only)
        let ciphertext = vec![0u8; 16];
        let frame = build_frame(0, &salt, &nonce, &ciphertext);
        let parsed = parse_frame(&frame).unwrap();
        assert_eq!(parsed.plaintext_len, 0);
        assert_eq!(parsed.ciphertext.len(), 16);
    }

    #[test]
    fn bits_to_bytes_partial_byte() {
        // 5 bits should produce 1 byte, padded with zeros
        let bits = vec![1u8, 0, 1, 1, 0];
        let bytes = bits_to_bytes(&bits);
        assert_eq!(bytes.len(), 1);
        // 10110_000 = 0xB0
        assert_eq!(bytes[0], 0xB0);
    }

    // --- Compact fortress frame tests ---

    #[test]
    fn compact_frame_build_parse_roundtrip() {
        // plaintext_len=4, ciphertext=4+16=20 bytes (AES-GCM tag).
        let ciphertext = vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
                              0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
                              0x77, 0x88, 0x99, 0x00, 0xAA, 0xBB,
                              0xCC, 0xDD];
        let frame = build_fortress_compact_frame(4, &ciphertext);

        // Total size: 2 + 20 + 4 = 26
        assert_eq!(frame.len(), 2 + 20 + 4);

        let parsed = parse_fortress_compact_frame(&frame).unwrap();
        assert_eq!(parsed.plaintext_len, 4);
        assert_eq!(parsed.ciphertext, ciphertext);
        // Salt and nonce should be the fixed constants
        assert_eq!(parsed.salt, crate::stego::crypto::FORTRESS_EMPTY_SALT);
        assert_eq!(parsed.nonce, crate::stego::crypto::FORTRESS_EMPTY_NONCE);
    }

    #[test]
    fn compact_frame_smaller_than_full() {
        // Same plaintext_len=4, ciphertext=20 bytes
        let salt = [1u8; SALT_LEN];
        let nonce = [2u8; NONCE_LEN];
        let ciphertext = vec![0u8; 20];

        let full_frame = build_frame(4, &salt, &nonce, &ciphertext);
        let compact_frame = build_fortress_compact_frame(4, &ciphertext);

        // Full: 2 + 16 + 12 + 20 + 4 = 54
        // Compact: 2 + 20 + 4 = 26
        assert_eq!(full_frame.len() - compact_frame.len(), SALT_LEN + NONCE_LEN);
        assert_eq!(full_frame.len() - compact_frame.len(), 28);
    }

    #[test]
    fn compact_frame_corrupted_crc_detected() {
        let ciphertext = vec![0u8; 20];
        let mut frame = build_fortress_compact_frame(4, &ciphertext);
        let len = frame.len();
        frame[len - 1] ^= 0xFF;
        assert!(matches!(parse_fortress_compact_frame(&frame), Err(StegoError::FrameCorrupted)));
    }

    #[test]
    fn compact_frame_truncated_rejected() {
        assert!(matches!(parse_fortress_compact_frame(&[0x00]), Err(StegoError::FrameCorrupted)));
        assert!(matches!(parse_fortress_compact_frame(&[]), Err(StegoError::FrameCorrupted)));
    }

    #[test]
    fn compact_frame_zero_length() {
        // plaintext_len=0, ciphertext=0+16=16 bytes (auth tag only)
        let ciphertext = vec![0u8; 16];
        let frame = build_fortress_compact_frame(0, &ciphertext);
        let parsed = parse_fortress_compact_frame(&frame).unwrap();
        assert_eq!(parsed.plaintext_len, 0);
        assert_eq!(parsed.ciphertext.len(), 16);
    }

    #[test]
    fn compact_frame_overhead_is_28_less() {
        assert_eq!(
            FRAME_OVERHEAD - FORTRESS_COMPACT_FRAME_OVERHEAD,
            28,
            "Compact frame saves exactly 28 bytes (salt + nonce)"
        );
    }
}
