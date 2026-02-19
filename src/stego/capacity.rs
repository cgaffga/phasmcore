use crate::jpeg::JpegImage;
use crate::stego::cost::uerd::compute_uerd;
use crate::stego::frame::FRAME_OVERHEAD;
use crate::stego::error::StegoError;

/// Minimum ratio of usable (non-WET) AC coefficients to message bits.
/// A ratio below this makes the STC likely to fail or produce detectable artifacts.
const MIN_CAPACITY_RATIO: f64 = 5.0;

/// Estimate the maximum plaintext message size (in bytes) that can be embedded
/// in the given cover JPEG image using Ghost mode.
pub fn estimate_capacity(img: &JpegImage) -> Result<usize, StegoError> {
    let grid = img.dct_grid(0);
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;

    let cost_map = compute_uerd(grid, qt);

    // Count usable (non-WET) AC coefficients.
    let total_ac = cost_map.total_blocks() * 63;
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

    // Subtract overhead (mode + length + salt + nonce + tag + crc) to get plaintext capacity.
    let capacity = max_frame_bytes - FRAME_OVERHEAD;

    // Also report for diagnostics:
    // total_ac, usable, capacity
    let _ = total_ac; // suppress unused warning in release

    Ok(capacity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_reasonable_for_photo() {
        let data = std::fs::read("../test-vectors/photo_320x240_q75_420.jpg").unwrap();
        let img = JpegImage::from_bytes(&data).unwrap();
        let cap = estimate_capacity(&img).unwrap();
        // 320×240 at 4:2:0 → 40×30=1200 Y blocks → 75,600 AC positions.
        // Even with many zeros, should have >100 bytes capacity.
        assert!(cap > 100, "capacity {cap} is too low for 320x240");
        // But shouldn't be unreasonably high.
        assert!(cap < 5000, "capacity {cap} is suspiciously high");
    }
}
