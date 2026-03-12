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
use crate::stego::cost::uniward::compute_uniward;
use crate::stego::frame::FRAME_OVERHEAD;
use crate::stego::error::StegoError;

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

/// Count usable (non-WET, non-DC) AC coefficients in a cost map.
fn count_usable_positions(cost_map: &super::cost::CostMap) -> usize {
    let mut usable = 0usize;
    let bt = cost_map.blocks_tall();
    let bw = cost_map.blocks_wide();
    for br in 0..bt {
        for bc in 0..bw {
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    if cost_map.get(br, bc, i, j).is_finite() {
                        usable += 1;
                    }
                }
            }
        }
    }
    usable
}

/// Convert usable position count + capacity ratio → plaintext byte capacity.
fn capacity_from_usable(usable: usize, ratio: f64) -> usize {
    let max_frame_bits = (usable as f64 / ratio) as usize;
    let max_frame_bytes = max_frame_bits / 8;

    if max_frame_bytes <= FRAME_OVERHEAD {
        return 0;
    }

    (max_frame_bytes - FRAME_OVERHEAD).min(u16::MAX as usize)
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
    let grid = img.dct_grid(0);
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;

    let cost_map = compute_uniward(grid, qt);
    let usable = count_usable_positions(&cost_map);

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
    let grid = img.dct_grid(0);
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;

    let cost_map = compute_uniward(grid, qt);
    let usable = count_usable_positions(&cost_map);

    Ok(capacity_from_usable(usable, MIN_CAPACITY_RATIO_SI))
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
