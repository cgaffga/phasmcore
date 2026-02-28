// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Armor mode capacity estimation.

use crate::jpeg::JpegImage;
use crate::stego::armor::ecc;
use crate::stego::armor::selection::compute_stability_map;
use crate::stego::armor::spreading::SPREAD_LEN;
use crate::stego::error::StegoError;
use crate::stego::frame::FRAME_OVERHEAD;

/// Estimate the maximum plaintext message size (in bytes) that can be embedded
/// in the given cover JPEG image using Armor mode.
///
/// The estimate accounts for:
/// - SPREAD_LEN coefficients per embedded bit (STDM spreading)
/// - Reed-Solomon parity overhead (64 bytes per 191-byte block)
/// - Frame overhead (length, salt, nonce, auth tag, CRC)
///
/// Note: Phase 2 adaptive robustness (higher RS parity, repetition coding,
/// adaptive delta) uses spare capacity automatically when the message is small
/// relative to the image. This does not reduce the maximum capacity reported
/// here -- it only improves robustness for messages well below this limit.
///
/// # Errors
/// Returns [`StegoError::NoLuminanceChannel`] if the image has no Y component
/// or its quantization table is missing.
pub fn estimate_armor_capacity(img: &JpegImage) -> Result<usize, StegoError> {
    let grid = img.dct_grid(0);
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img
        .quant_table(qt_id)
        .ok_or(StegoError::NoLuminanceChannel)?;

    let cost_map = compute_stability_map(grid, qt);

    // Count stable (non-WET) AC positions
    let bt = cost_map.blocks_tall();
    let bw = cost_map.blocks_wide();
    let mut stable_count = 0usize;
    for br in 0..bt {
        for bc in 0..bw {
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    if cost_map.get(br, bc, i, j).is_finite() {
                        stable_count += 1;
                    }
                }
            }
        }
    }

    // Each SPREAD_LEN stable positions carry 1 bit
    let num_units = stable_count / SPREAD_LEN;

    // Subtract header overhead (56 units for 1-byte mean-QT header)
    let qf_header_units = super::embedding::HEADER_UNITS;
    if num_units <= qf_header_units {
        return Ok(0);
    }
    let payload_units = num_units - qf_header_units;

    let embeddable_bytes = payload_units / 8;

    if embeddable_bytes == 0 {
        return Ok(0);
    }

    // RS encoding expands each block by PARITY_LEN bytes.
    // For a single shortened RS block: encoded = data + 64 parity
    // So max data = embeddable_bytes - 64 (for a single block)
    // For multi-block: solve iteratively
    let parity = ecc::parity_len();
    if embeddable_bytes <= parity {
        return Ok(0);
    }

    // How many frame bytes can we fit?
    // frame_bytes + ceil(frame_bytes / 191) * 64 <= embeddable_bytes
    // Approximate: frame_bytes * (1 + 64/191) <= embeddable_bytes
    // frame_bytes <= embeddable_bytes * 191 / 255
    let max_frame_bytes = (embeddable_bytes as f64 * 191.0 / 255.0).floor() as usize;

    if max_frame_bytes <= FRAME_OVERHEAD {
        return Ok(0);
    }

    // Verify the RS-encoded size actually fits.
    // Cap at u16::MAX since the frame format uses a 2-byte length prefix.
    let capacity = (max_frame_bytes - FRAME_OVERHEAD).min(u16::MAX as usize);
    let frame_len = capacity + FRAME_OVERHEAD;
    let rs_len = ecc::rs_encoded_len(frame_len);
    if rs_len > embeddable_bytes {
        // Reduce by 1 to ensure fit
        if capacity == 0 {
            return Ok(0);
        }
        return Ok(capacity - 1);
    }

    Ok(capacity)
}
