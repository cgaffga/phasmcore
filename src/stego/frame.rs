use crate::stego::crypto::{NONCE_LEN, SALT_LEN};
use crate::stego::error::StegoError;

/// Ghost mode identifier byte.
const MODE_GHOST: u8 = 0x01;

/// Fixed overhead: mode(1) + length(2) + salt(16) + nonce(12) + tag(16) + crc(4) = 51 bytes.
/// Plus the ciphertext length equals plaintext length (AES-GCM stream cipher).
/// So total frame = 35 + plaintext_len + 16(tag) = 51 + plaintext_len.
pub const FRAME_OVERHEAD: usize = 1 + 2 + SALT_LEN + NONCE_LEN + 16 + 4; // 51

/// Maximum payload frame size in bytes (supports up to ~1KB messages).
/// 1024 (max message) + 51 (overhead) = 1075 bytes.
pub const MAX_FRAME_BYTES: usize = 1075;

/// Maximum payload frame size in bits.
pub const MAX_FRAME_BITS: usize = MAX_FRAME_BYTES * 8;

/// Build a payload frame from encrypted components.
///
/// Frame layout:
/// ```text
/// [1 byte  ] mode = 0x01 (Ghost)
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
    let mut frame = Vec::with_capacity(1 + 2 + SALT_LEN + NONCE_LEN + ciphertext.len() + 4);

    frame.push(MODE_GHOST);
    frame.extend_from_slice(&plaintext_len.to_be_bytes());
    frame.extend_from_slice(salt);
    frame.extend_from_slice(nonce);
    frame.extend_from_slice(ciphertext);

    let crc = crc32fast::hash(&frame);
    frame.extend_from_slice(&crc.to_be_bytes());

    frame
}

/// Parsed payload frame.
pub struct ParsedFrame {
    pub plaintext_len: u16,
    pub salt: [u8; SALT_LEN],
    pub nonce: [u8; NONCE_LEN],
    pub ciphertext: Vec<u8>,
}

/// Parse a payload frame, verifying the CRC and mode byte.
///
/// The input `data` may be larger than the actual frame (e.g. zero-padded).
/// The actual frame length is determined from the embedded `plaintext_len` field.
pub fn parse_frame(data: &[u8]) -> Result<ParsedFrame, StegoError> {
    // Need at least 3 bytes to read mode + plaintext_len.
    if data.len() < 3 {
        return Err(StegoError::FrameCorrupted);
    }

    // Read plaintext_len to compute the actual frame size.
    let plaintext_len = u16::from_be_bytes([data[1], data[2]]);
    let ciphertext_len = plaintext_len as usize + 16; // AES-GCM auth tag
    let total_frame_len = 1 + 2 + SALT_LEN + NONCE_LEN + ciphertext_len + 4;

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
    let mode = payload[0];
    if mode != MODE_GHOST {
        return Err(StegoError::UnknownFrameMode(mode));
    }

    let mut salt = [0u8; SALT_LEN];
    salt.copy_from_slice(&payload[3..3 + SALT_LEN]);

    let mut nonce = [0u8; NONCE_LEN];
    nonce.copy_from_slice(&payload[3 + SALT_LEN..3 + SALT_LEN + NONCE_LEN]);

    let ciphertext = payload[3 + SALT_LEN + NONCE_LEN..].to_vec();

    Ok(ParsedFrame {
        plaintext_len,
        salt,
        nonce,
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
    fn wrong_mode_rejected() {
        let salt = [0u8; SALT_LEN];
        let nonce = [0u8; NONCE_LEN];
        // plaintext_len=4, ciphertext=4+16=20 bytes.
        let ciphertext = vec![0u8; 20];
        let mut frame = build_frame(4, &salt, &nonce, &ciphertext);
        // Change mode byte and recompute CRC.
        frame[0] = 0xFF;
        let crc = crc32fast::hash(&frame[..frame.len() - 4]);
        let len = frame.len();
        frame[len - 4..].copy_from_slice(&crc.to_be_bytes());
        assert!(matches!(parse_frame(&frame), Err(StegoError::UnknownFrameMode(0xFF))));
    }

    #[test]
    fn bytes_bits_roundtrip() {
        let original = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let bits = bytes_to_bits(&original);
        assert_eq!(bits.len(), 32);
        let recovered = bits_to_bytes(&bits);
        assert_eq!(recovered, original);
    }
}
