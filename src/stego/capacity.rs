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

/// Minimum ratio of usable (non-WET) AC coefficients to message bits.
/// A ratio below this makes the STC likely to fail or produce detectable artifacts.
const MIN_CAPACITY_RATIO: f64 = 5.0;

/// Estimate the maximum plaintext message size (in bytes) that can be embedded
/// in the given cover JPEG image using Ghost mode.
///
/// The estimate is conservative: it divides the usable coefficient count by
/// [`MIN_CAPACITY_RATIO`] (5.0) to ensure the STC has sufficient slack for
/// low-distortion embedding, then subtracts the frame overhead.
///
/// # Arguments
/// - `img`: A parsed JPEG image.
///
/// # Returns
/// The estimated capacity in bytes, or 0 if the image is too small.
///
/// # Errors
/// Returns [`StegoError::NoLuminanceChannel`] if the image has no Y component
/// or its quantization table is missing.
pub fn estimate_capacity(img: &JpegImage) -> Result<usize, StegoError> {
    let grid = img.dct_grid(0);
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;

    let cost_map = compute_uniward(grid, qt);

    // Count usable (non-WET) AC coefficients.
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

    // Maximum embeddable bits = usable / MIN_CAPACITY_RATIO.
    // Frame overhead must be subtracted.
    let max_frame_bits = (usable as f64 / MIN_CAPACITY_RATIO) as usize;
    let max_frame_bytes = max_frame_bits / 8;

    if max_frame_bytes <= FRAME_OVERHEAD {
        return Ok(0);
    }

    // Subtract overhead (length + salt + nonce + tag + crc) to get plaintext capacity.
    // Cap at u16::MAX since the frame format uses a 2-byte length prefix.
    let capacity = (max_frame_bytes - FRAME_OVERHEAD).min(u16::MAX as usize);

    Ok(capacity)
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
}
