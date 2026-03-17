// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Ghost mode capacity estimation.
//!
//! Estimates the maximum plaintext message size that can be embedded in a
//! given JPEG cover image using Ghost mode. The estimate accounts for:
//! - Number of usable (non-WET) AC coefficients
//! - Minimum capacity ratio to ensure reliable STC embedding
//! - Frame overhead (length, salt, nonce, auth tag, CRC)

use crate::jpeg::JpegImage;
use crate::jpeg::dct::DctGrid;
use crate::stego::frame::{FRAME_OVERHEAD, FRAME_OVERHEAD_EXT};
use crate::stego::error::StegoError;
use crate::stego::shadow;

/// Minimum ratio of usable (non-WET) AC coefficients to message bits
/// for standard J-UNIWARD. A ratio below this produces detectable artifacts.
const MIN_CAPACITY_RATIO: f64 = 5.0;

/// Minimum ratio for SI-UNIWARD ("Deep Cover").
///
/// SI-UNIWARD lowers embedding costs by exploiting quantization rounding errors,
/// allowing a higher embedding rate at the same steganalysis detectability.
/// Literature shows ~1.5-2× capacity gain; 3.5 is a conservative estimate (~43%
/// more capacity than J-UNIWARD's ratio of 5.0).
const MIN_CAPACITY_RATIO_SI: f64 = 3.5;

/// Count non-zero AC coefficients in the DctGrid (fast capacity estimation).
///
/// This is equivalent to counting usable positions from a CostMap, since
/// J-UNIWARD assigns finite costs to all non-zero AC coefficients. This
/// avoids the expensive UNIWARD cost computation, making capacity estimation
/// instantaneous even for very large images.
fn count_nonzero_ac(grid: &DctGrid) -> usize {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let mut count = 0usize;
    for br in 0..bt {
        for bc in 0..bw {
            let blk = grid.block(br, bc);
            for k in 1..64 { // skip DC at index 0
                if blk[k] != 0 {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Convert usable position count + capacity ratio → plaintext byte capacity.
fn capacity_from_usable(usable: usize, ratio: f64) -> usize {
    let max_frame_bits = (usable as f64 / ratio) as usize;
    let max_frame_bytes = max_frame_bits / 8;

    if max_frame_bytes <= FRAME_OVERHEAD {
        return 0;
    }

    let capacity = max_frame_bytes - FRAME_OVERHEAD;
    if capacity > u16::MAX as usize {
        // v2 frame needs 4 extra bytes for the extended length header.
        max_frame_bytes.saturating_sub(FRAME_OVERHEAD_EXT)
    } else {
        capacity
    }
}

/// Estimate Ghost mode capacity (standard J-UNIWARD).
///
/// Conservative estimate: divides usable coefficient count by
/// [`MIN_CAPACITY_RATIO`] (5.0) to ensure the STC has sufficient slack for
/// low-distortion embedding, then subtracts the frame overhead.
///
/// # Errors
/// Returns [`StegoError::NoLuminanceChannel`] if the image has no Y component
/// or its quantization table is missing.
pub fn estimate_capacity(img: &JpegImage) -> Result<usize, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let usable = count_nonzero_ac(img.dct_grid(0));
    Ok(capacity_from_usable(usable, MIN_CAPACITY_RATIO))
}

/// Estimate Ghost mode capacity with SI-UNIWARD ("Deep Cover").
///
/// When the input image is non-JPEG (PNG, HEIC, RAW), SI-UNIWARD exploits
/// quantization rounding errors to embed at lower distortion per bit. This
/// allows a higher embedding rate at the same steganalysis risk, resulting
/// in ~43% more capacity than standard J-UNIWARD.
///
/// Uses [`MIN_CAPACITY_RATIO_SI`] (3.5) — conservative relative to the
/// literature's 1.5-2× improvement at equal detectability.
///
/// No raw pixels needed: the position count is identical, only the per-bit
/// distortion budget changes.
pub fn estimate_capacity_si(img: &JpegImage) -> Result<usize, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let usable = count_nonzero_ac(img.dct_grid(0));
    Ok(capacity_from_usable(usable, MIN_CAPACITY_RATIO_SI))
}

/// Estimate shadow layer capacity for a JPEG image.
///
/// Shadow uses direct LSB embedding in the Y (luminance) channel with
/// cost-pool position selection. Capacity is based on Y nzAC count.
pub fn estimate_shadow_capacity(img: &JpegImage) -> Result<usize, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let y_nzac = count_nonzero_ac(img.dct_grid(0));
    Ok(shadow::shadow_capacity(y_nzac))
}

/// Estimate Ghost primary capacity accounting for shadow position overhead.
///
/// Subtracts shadow positions from usable Y NZAC before computing primary
/// capacity. Uses conservative RS parity (16) for the estimate since the
/// escalation cascade frequently bumps parity above the initial 4.
pub fn estimate_capacity_with_shadows(
    img: &JpegImage,
    shadow_count: usize,
    shadow_total_bytes: usize,
    is_si: bool,
) -> Result<usize, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let y_nzac = count_nonzero_ac(img.dct_grid(0));
    let ratio = if is_si { MIN_CAPACITY_RATIO_SI } else { MIN_CAPACITY_RATIO };

    // Conservative RS parity for capacity estimate (cascade often escalates)
    let parity = 16usize;
    // Compute RS-encoded shadow bytes (per shadow: frame_overhead + payload, RS-encoded)
    let shadow_frame_overhead = 46usize; // plaintext_len(2) + salt(16) + nonce(12) + tag(16)
    let total_shadow_frame_bytes = shadow_count * shadow_frame_overhead + shadow_total_bytes;
    // RS encoding expands data: for each 255-byte block, parity bytes added
    let k = 255 - parity; // 239 data bytes per block
    let full_blocks = (total_shadow_frame_bytes + k - 1) / k;
    let shadow_rs_bytes = full_blocks * 255;
    let shadow_bits = shadow_rs_bytes * 8;

    // Subtract shadow positions from available pool
    let effective_nzac = y_nzac.saturating_sub(shadow_bits);
    Ok(capacity_from_usable(effective_nzac, ratio))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_reasonable_for_photo() {
        let data = std::fs::read("test-vectors/photo_320x240_q75_420.jpg").unwrap();
        let img = JpegImage::from_bytes(&data).unwrap();
        let cap = estimate_capacity(&img).unwrap();
        // 320×240 at 4:2:0 → 40×30=1200 Y blocks → 75,600 AC positions.
        // Even with many zeros, should have >100 bytes capacity.
        assert!(cap > 100, "capacity {cap} is too low for 320x240");
        // But shouldn't be unreasonably high.
        assert!(cap < 5000, "capacity {cap} is suspiciously high");
    }

    #[test]
    fn si_capacity_higher_than_standard() {
        let data = std::fs::read("test-vectors/photo_320x240_q75_420.jpg").unwrap();
        let img = JpegImage::from_bytes(&data).unwrap();
        let cap_j = estimate_capacity(&img).unwrap();
        let cap_si = estimate_capacity_si(&img).unwrap();
        // SI should give ~43% more capacity (ratio 3.5 vs 5.0)
        assert!(
            cap_si > cap_j,
            "SI capacity ({cap_si}) should exceed J-UNIWARD capacity ({cap_j})"
        );
        // Verify the ratio is approximately 5.0/3.5 ≈ 1.43
        let ratio = cap_si as f64 / cap_j as f64;
        assert!(
            ratio > 1.3 && ratio < 1.6,
            "SI/J ratio {ratio:.2} should be ~1.43"
        );
    }
}
