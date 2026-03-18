// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

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
pub mod payload;
mod pipeline;
pub mod armor;
pub mod progress;
pub mod side_info;
pub mod shadow;
pub mod optimizer;
pub mod quality;

pub use error::StegoError;
pub use quality::EncodeQuality;
pub use optimizer::{optimize_cover, OptimizerConfig, OptimizerMode};

/// Maximum pixel dimension (width or height) for encode.
/// Images exceeding this are downsampled by the frontend before reaching Rust.
pub const MAX_DIMENSION: u32 = 16384;

/// Maximum total pixel count for encode (width × height).
/// 200 MP covers all current cameras including flagship 200 MP sensors.
/// Memory-optimized: strip-based UNIWARD (~170 MB/strip), compact positions
/// (8 bytes each), segmented STC Viterbi. Total ~1 GB for 200 MP.
pub const MAX_PIXELS: u32 = 200_000_000;

/// Minimum pixel dimension (width or height) for encode.
/// Images below this are rejected with an error message.
pub const MIN_ENCODE_DIMENSION: u32 = 200;

/// Target pixel dimension (longest side) for Armor/Fortress pre-resize.
/// Images larger than this are downsampled by the frontend before encoding
/// in Armor mode, so that the 8×8 block grid survives platform recompression
/// (e.g. WhatsApp resizes to ~1600px on the longest side).
pub const ARMOR_TARGET_DIMENSION: u32 = 1600;

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
    if width > MAX_DIMENSION || height > MAX_DIMENSION || width.checked_mul(height).map_or(true, |p| p > MAX_PIXELS) {
        return Err(StegoError::ImageTooLarge);
    }
    Ok(())
}
pub use pipeline::{ghost_encode, ghost_decode, ghost_encode_with_files, ghost_encode_si, ghost_encode_si_with_files, GHOST_DECODE_STEPS, GHOST_ENCODE_STEPS};
pub use pipeline::{ghost_encode_with_quality, ghost_encode_with_files_quality, ghost_encode_si_with_quality, ghost_encode_si_with_files_quality};
pub use pipeline::{ghost_encode_with_shadows, ghost_encode_si_with_shadows, ghost_shadow_decode, ShadowLayer, GHOST_ENCODE_WITH_SHADOWS_STEPS};
pub use pipeline::{ghost_encode_with_shadows_quality, ghost_encode_si_with_shadows_quality};
pub use shadow::shadow_capacity;
pub use capacity::estimate_shadow_capacity;
pub use capacity::estimate_capacity as ghost_capacity;
pub use capacity::estimate_capacity_si as ghost_capacity_si;
pub use capacity::estimate_capacity_with_shadows as ghost_capacity_with_shadows;
pub use armor::pipeline::{armor_encode, armor_encode_with_quality, armor_decode, DecodeQuality, ArmorCapacityInfo, armor_capacity_info};
pub use armor::capacity::estimate_armor_capacity as armor_capacity;
pub use payload::{PayloadData, FileEntry, compressed_payload_size};

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
        assert!(validate_encode_dimensions(16384, 1000).is_ok());
        assert!(validate_encode_dimensions(1000, 16384).is_ok());
        assert!(validate_encode_dimensions(16385, 1000).is_err());
        assert!(validate_encode_dimensions(1000, 16385).is_err());
    }

    #[test]
    fn too_many_pixels() {
        // 14143 * 14143 = 200_024_449 > 200M
        assert!(validate_encode_dimensions(14143, 14143).is_err());
        // 14142 * 14142 = 199_996_164 < 200M — OK
        assert!(validate_encode_dimensions(14142, 14142).is_ok());
    }

    #[test]
    fn error_variants() {
        match validate_encode_dimensions(100, 300) {
            Err(StegoError::ImageTooSmall) => {}
            other => panic!("expected ImageTooSmall, got {other:?}"),
        }
        match validate_encode_dimensions(16385, 1000) {
            Err(StegoError::ImageTooLarge) => {}
            other => panic!("expected ImageTooLarge, got {other:?}"),
        }
    }
}

/// Unified decode: auto-detects Ghost or Armor mode from the embedded frame.
///
/// Tries Ghost first, then Armor. Returns the decoded payload and quality info.
///
/// When the `parallel` feature is enabled, Ghost and Armor decodes run
/// concurrently via `rayon::join`, roughly halving decode latency on
/// multi-core devices.
pub fn smart_decode(stego_bytes: &[u8], passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let result = smart_decode_inner(stego_bytes, passphrase);
    progress::finish();
    result
}

/// Serial smart_decode implementation (default path and WASM).
///
/// Tries Armor first (default mode, most common), then Ghost.
/// Progress steps: 1 (fortress) + ~21 (phase1) + ~21 (phase2) + 1 (phase3)
///   + GHOST_DECODE_STEPS (102: 100 UNIWARD + 2 STC/decrypt).
/// Actual total is set by try_armor_decode once candidate count is known.
#[cfg(not(feature = "parallel"))]
fn smart_decode_inner(stego_bytes: &[u8], passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    progress::init(0); // reset; try_armor_decode sets real total

    progress::check_cancelled()?;

    let mut saw_decryption_failed = false;

    // Try Armor first (default mode, most likely)
    match armor_decode(stego_bytes, passphrase) {
        Ok((payload, quality)) => return Ok((payload, quality)),
        Err(StegoError::DecryptionFailed) => {
            saw_decryption_failed = true;
            // Could be wrong passphrase for Armor — still try Ghost
        }
        Err(StegoError::FrameCorrupted) => {
            // Likely not Armor — try Ghost
        }
        Err(e) => {
            // Fundamental error (bad JPEG, too small, etc.) — try Ghost anyway
            // in case Armor fails for mode-specific reasons
            match ghost_decode(stego_bytes, passphrase) {
                Ok(payload) => return Ok((payload, DecodeQuality::ghost())),
                Err(_) => return Err(e), // Return original Armor error
            }
        }
    }

    // Try Ghost — extend progress total instead of resetting to avoid the
    // bar jumping back to 0%.  The bar continues smoothly from the Armor
    // phase into the Ghost phase.
    let (armor_done, _) = progress::get();
    progress::set_total(armor_done + GHOST_DECODE_STEPS as u32);
    let ghost_result = ghost_decode(stego_bytes, passphrase);
    match ghost_result {
        Ok(payload) => return Ok((payload, DecodeQuality::ghost())),
        Err(StegoError::DecryptionFailed) => {
            saw_decryption_failed = true;
        }
        Err(_) => {}
    }

    // Try Ghost shadow (Y-channel direct LSB + RS)
    match pipeline::ghost_shadow_decode(stego_bytes, passphrase) {
        Ok(payload) => return Ok((payload, DecodeQuality::ghost())),
        Err(StegoError::DecryptionFailed) => {
            saw_decryption_failed = true;
        }
        Err(_) => {}
    }

    if saw_decryption_failed {
        Err(StegoError::DecryptionFailed)
    } else {
        Err(StegoError::FrameCorrupted)
    }
}

/// Parallel smart_decode: three-way concurrent decode via rayon.
///
/// Parses the JPEG once and shares `&JpegImage` across threads.
/// Runs Fortress, STDM+Phase3, and Ghost in parallel.
/// Preference order: Fortress > Armor STDM > Ghost.
#[cfg(feature = "parallel")]
fn smart_decode_inner(stego_bytes: &[u8], passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    use crate::jpeg::JpegImage;
    use crate::stego::armor::fortress;
    use crate::stego::armor::pipeline::armor_decode_no_fortress;

    // In parallel mode all three branches advance the same global counter
    // concurrently.  We init with 0 (indeterminate) — try_armor_decode will
    // set a real total once it knows the candidate count.  The cap in
    // advance() prevents step from ever exceeding total.
    progress::init(0);
    progress::check_cancelled()?;

    let img = JpegImage::from_bytes(stego_bytes)?;

    let (fortress_result, (stdm_result, (ghost_result, shadow_result))) = rayon::join(
        || {
            if img.num_components() > 0 {
                fortress::fortress_decode(&img, passphrase)
            } else {
                Err(StegoError::FrameCorrupted)
            }
        },
        || rayon::join(
            || armor_decode_no_fortress(&img, stego_bytes, passphrase),
            || rayon::join(
                || ghost_decode(stego_bytes, passphrase),
                || pipeline::ghost_shadow_decode_from_image(&img, passphrase),
            ),
        ),
    );

    // Prefer Fortress (fastest, most robust).
    if let Ok((payload, quality)) = fortress_result {
        return Ok((payload, quality));
    }

    // Try Armor STDM + Phase 3.
    if let Ok((payload, quality)) = stdm_result {
        return Ok((payload, quality));
    }

    // Try Ghost.
    if let Ok(payload) = ghost_result {
        return Ok((payload, DecodeQuality::ghost()));
    }

    // Try Ghost shadow.
    if let Ok(payload) = shadow_result {
        return Ok((payload, DecodeQuality::ghost()));
    }

    // All failed — determine the best error to report.
    let saw_decryption_failed = matches!(&fortress_result, Err(StegoError::DecryptionFailed))
        || matches!(&stdm_result, Err(StegoError::DecryptionFailed))
        || matches!(&ghost_result, Err(StegoError::DecryptionFailed))
        || matches!(&shadow_result, Err(StegoError::DecryptionFailed));

    if saw_decryption_failed {
        return Err(StegoError::DecryptionFailed);
    }

    Err(stdm_result.unwrap_err())
}
