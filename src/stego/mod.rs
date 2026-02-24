//! Steganographic encoding and decoding pipelines.
//!
//! This module provides two embedding modes:
//!
//! - **Ghost** (`ghost_encode` / `ghost_decode`): Stealth mode using J-UNIWARD
//!   cost function and Syndrome-Trellis Coding (STC) to minimize statistical
//!   detectability. Best for images that will not be recompressed.
//!
//! - **Armor** (`armor_encode` / `armor_decode`): Robustness mode using STDM
//!   (Spread Transform Dither Modulation) with Reed-Solomon error correction to
//!   survive JPEG recompression. Trades capacity for survivability.
//!
//! Both modes share the same payload frame format, encryption (AES-256-GCM-SIV),
//! and key derivation (Argon2id two-tier). The `smart_decode` function
//! auto-detects which mode was used.

pub mod error;
pub mod cost;
pub mod stc;
pub mod crypto;
pub mod frame;
pub mod permute;
pub mod capacity;
mod pipeline;
pub mod armor;

pub use error::StegoError;

/// Maximum pixel dimension (width or height) for encode.
/// Images exceeding this are downsampled by the frontend before reaching Rust.
pub const MAX_DIMENSION: u32 = 8192;

/// Maximum total pixel count for encode (width × height).
pub const MAX_PIXELS: u32 = 16_000_000;

/// Minimum pixel dimension (width or height) for encode.
/// Images below this are rejected with an error message.
pub const MIN_ENCODE_DIMENSION: u32 = 200;

/// Validate image dimensions for encoding.
///
/// Returns `Ok(())` if the dimensions are within acceptable bounds.
/// Called at the start of both `ghost_encode` and `armor_encode`.
///
/// # Errors
/// - [`StegoError::ImageTooSmall`] if either dimension < 200px.
/// - [`StegoError::ImageTooLarge`] if either dimension > 8192px or total pixels > 16M.
pub fn validate_encode_dimensions(width: u32, height: u32) -> Result<(), StegoError> {
    if width < MIN_ENCODE_DIMENSION || height < MIN_ENCODE_DIMENSION {
        return Err(StegoError::ImageTooSmall);
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION || width * height > MAX_PIXELS {
        return Err(StegoError::ImageTooLarge);
    }
    Ok(())
}
pub use pipeline::ghost_encode;
pub use pipeline::ghost_decode;
pub use capacity::estimate_capacity as ghost_capacity;
pub use armor::pipeline::{armor_encode, armor_decode, DecodeQuality, ArmorCapacityInfo, armor_capacity_info};
pub use armor::capacity::estimate_armor_capacity as armor_capacity;

#[cfg(test)]
mod dimension_tests {
    use super::*;

    #[test]
    fn valid_dimensions() {
        assert!(validate_encode_dimensions(800, 600).is_ok());
        assert!(validate_encode_dimensions(3000, 4000).is_ok());
    }

    #[test]
    fn boundary_min() {
        assert!(validate_encode_dimensions(200, 200).is_ok());
        assert!(validate_encode_dimensions(199, 200).is_err());
        assert!(validate_encode_dimensions(200, 199).is_err());
    }

    #[test]
    fn boundary_max_dimension() {
        assert!(validate_encode_dimensions(8192, 1000).is_ok());
        assert!(validate_encode_dimensions(1000, 8192).is_ok());
        assert!(validate_encode_dimensions(8193, 1000).is_err());
        assert!(validate_encode_dimensions(1000, 8193).is_err());
    }

    #[test]
    fn too_many_pixels() {
        // 5000 * 3201 = 16_005_000 > 16M
        assert!(validate_encode_dimensions(5000, 3201).is_err());
        // 4000 * 4000 = 16M exactly — OK
        assert!(validate_encode_dimensions(4000, 4000).is_ok());
    }

    #[test]
    fn error_variants() {
        match validate_encode_dimensions(100, 300) {
            Err(StegoError::ImageTooSmall) => {}
            other => panic!("expected ImageTooSmall, got {other:?}"),
        }
        match validate_encode_dimensions(9000, 1000) {
            Err(StegoError::ImageTooLarge) => {}
            other => panic!("expected ImageTooLarge, got {other:?}"),
        }
    }
}

/// Unified decode: auto-detects Ghost or Armor mode from the embedded frame.
///
/// Tries Ghost first, then Armor. Returns the decoded message and quality info.
pub fn smart_decode(stego_bytes: &[u8], passphrase: &str) -> Result<(String, DecodeQuality), StegoError> {
    let mut saw_decryption_failed = false;

    // Try Ghost first
    match ghost_decode(stego_bytes, passphrase) {
        Ok(text) => return Ok((text, DecodeQuality::ghost())),
        Err(StegoError::DecryptionFailed) => {
            saw_decryption_failed = true;
            // Could be wrong passphrase for Ghost — still try Armor
        }
        Err(StegoError::FrameCorrupted) => {
            // Likely not Ghost — try Armor
        }
        Err(e) => {
            // Fundamental error (bad JPEG, too small, etc.) — try Armor anyway
            // in case Ghost fails for mode-specific reasons
            match armor_decode(stego_bytes, passphrase) {
                Ok((text, quality)) => return Ok((text, quality)),
                Err(_) => return Err(e), // Return original Ghost error
            }
        }
    }

    // Try Armor
    match armor_decode(stego_bytes, passphrase) {
        Ok((text, quality)) => Ok((text, quality)),
        Err(StegoError::DecryptionFailed) => Err(StegoError::DecryptionFailed),
        Err(e) => {
            // Only report DecryptionFailed if at least one mode actually
            // reached the decryption stage. Otherwise the image likely has
            // no hidden message at all (or is too corrupted to decode).
            if saw_decryption_failed {
                Err(StegoError::DecryptionFailed)
            } else {
                Err(e)
            }
        }
    }
}
