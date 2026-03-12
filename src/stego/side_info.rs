// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Side information for SI-UNIWARD (Side-Informed UNIWARD).
//!
//! When the encoder has access to the original uncompressed pixels (e.g. PNG,
//! HEIC, or RAW input), it can compute the quantization rounding errors — the
//! difference between the continuous DCT coefficients and their rounded integer
//! values. These errors reveal which coefficients are "close to the boundary"
//! between two quantization levels and can be cheaply flipped.
//!
//! SI-UNIWARD uses this to:
//! 1. **Lower embedding costs** for coefficients with large rounding errors
//!    (cheap to push across the boundary).
//! 2. **Choose modification direction** toward the pre-quantization value
//!    (minimizing perceptual distortion).
//!
//! The result: ~1.5-2× capacity at the same detection risk, or equivalently
//! the same capacity with significantly lower distortion.
//!
//! The decoder is completely unchanged — it reads LSBs regardless of which
//! direction the modification went.

use crate::jpeg::dct::DctGrid;
use crate::jpeg::pixels::{dct_block_unquantized, rgb_to_luma_blocks};

/// Per-coefficient rounding errors from quantization.
///
/// Each error is in [-0.5, +0.5] and represents how far the continuous
/// (unquantized) DCT coefficient was from its rounded integer value.
/// Positive error means the pre-quantization value was above the integer;
/// negative means below.
pub struct SideInfo {
    /// Rounding errors in DctGrid flat order (block_idx * 64 + row * 8 + col).
    pub rounding_errors: Vec<f64>,
    /// Number of 8×8 blocks horizontally.
    pub blocks_wide: usize,
    /// Number of 8×8 blocks vertically.
    pub blocks_tall: usize,
}

/// Minimum cost for SI-modulated coefficients.
///
/// When |rounding_error| ≈ 0.5 ("1/2-coefficients"), the modulated cost
/// approaches zero. Clamping to this floor prevents zero-cost embedding
/// at quantization midpoints, which is a known detectable artifact.
const MIN_SI_COST: f32 = 1e-6;

impl SideInfo {
    /// Compute side information from raw RGB pixels and the cover JPEG.
    ///
    /// For each Y-channel 8×8 block:
    /// 1. Forward DCT on the original (pre-JPEG) pixels
    /// 2. Divide by quantization table (without rounding)
    /// 3. error = unquantized_value − cover_integer_coefficient
    ///
    /// Errors are clamped to [-0.5, +0.5] for robustness against minor
    /// floating-point differences between the platform's JPEG encoder and
    /// our forward DCT implementation.
    pub fn compute(
        raw_rgb: &[u8],
        pixel_width: u32,
        pixel_height: u32,
        cover_grid: &DctGrid,
        qt_values: &[u16; 64],
    ) -> Self {
        let bw = cover_grid.blocks_wide();
        let bh = cover_grid.blocks_tall();
        let total_coeffs = bw * bh * 64;
        let mut errors = vec![0.0f64; total_coeffs];

        // Convert raw RGB → Y luma blocks
        let luma_blocks = rgb_to_luma_blocks(raw_rgb, pixel_width, pixel_height);

        for br in 0..bh {
            for bc in 0..bw {
                let block_idx = br * bw + bc;

                // Get the luma block (may have fewer blocks if dimensions differ)
                let luma_bw = ((pixel_width as usize) + 7) / 8;
                if br >= ((pixel_height as usize) + 7) / 8 || bc >= luma_bw {
                    continue; // out of range — leave errors at 0
                }
                let luma_idx = br * luma_bw + bc;
                let luma_block = &luma_blocks[luma_idx];

                // Forward DCT + divide by QT (no rounding)
                let unquantized = dct_block_unquantized(luma_block, qt_values);

                // Compute and clamp rounding errors
                let cover_block: [i16; 64] = {
                    let slice = cover_grid.block(br, bc);
                    slice.try_into().unwrap()
                };

                for k in 0..64 {
                    let error = unquantized[k] - cover_block[k] as f64;
                    // Clamp to [-0.5, +0.5] for robustness
                    errors[block_idx * 64 + k] = error.clamp(-0.5, 0.5);
                }
            }
        }

        SideInfo {
            rounding_errors: errors,
            blocks_wide: bw,
            blocks_tall: bh,
        }
    }

    /// Get the rounding error at a flat index.
    #[inline]
    pub fn error_at(&self, flat_idx: usize) -> f64 {
        self.rounding_errors[flat_idx]
    }
}

/// Modulate J-UNIWARD costs using SI rounding errors.
///
/// For each AC coefficient with finite cost:
/// - `modulated_cost = ρ × (1 - 2|e|)` where `e` is the rounding error
/// - Larger |e| → lower cost (closer to quantization boundary → cheaper to flip)
/// - |e| = 0 → cost unchanged (exactly on the integer → no benefit)
/// - |e| = 0.5 → cost clamped to `MIN_SI_COST` (avoid zero-cost artifact)
///
/// Special cases:
/// - DC coefficients remain WET (infinite cost)
/// - |coeff| = 1 positions: cost is NOT modulated (anti-shrinkage forces
///   the direction, so the rounding error doesn't help choose direction)
pub fn modulate_costs_si(
    cost_map: &mut super::cost::CostMap,
    side_info: &SideInfo,
    cover_grid: &DctGrid,
) {
    let bw = cost_map.blocks_wide();
    let bh = cost_map.blocks_tall();

    for br in 0..bh {
        for bc in 0..bw {
            let block_idx = br * bw + bc;
            for i in 0..8 {
                for j in 0..8 {
                    // Skip DC
                    if i == 0 && j == 0 {
                        continue;
                    }

                    let cost = cost_map.get(br, bc, i, j);
                    if !cost.is_finite() {
                        continue; // WET position — leave as-is
                    }

                    // Skip |coeff| == 1: anti-shrinkage forces direction,
                    // SI modulation doesn't help
                    let coeff = cover_grid.get(br, bc, i, j);
                    if coeff.abs() == 1 {
                        continue;
                    }

                    let flat_idx = block_idx * 64 + i * 8 + j;
                    let error = side_info.error_at(flat_idx);
                    let abs_error = error.abs();

                    // modulated = ρ × (1 - 2|e|)
                    // When |e| = 0.5: modulated = 0 → clamp to MIN_SI_COST
                    let factor = (1.0 - 2.0 * abs_error) as f32;
                    let modulated = (cost * factor).max(MIN_SI_COST);
                    cost_map.set(br, bc, i, j, modulated);
                }
            }
        }
    }
}

/// Determine the modification direction for a coefficient using SI rounding error.
///
/// Returns the modified coefficient value (coeff ± 1).
///
/// Rules:
/// - |coeff| == 1: ALWAYS away from zero (anti-shrinkage, prevents coeff → 0)
/// - |coeff| > 1 with side info: toward the pre-quantization value
///   (error > 0 → precover was above → go up; error < 0 → go down)
/// - |coeff| > 1 without side info: nsF5 convention (toward zero)
/// - coeff == 0: should never be called (filtered out as WET)
#[inline]
pub fn si_modify_coefficient(coeff: i16, rounding_error: f64) -> i16 {
    if coeff == 1 {
        2 // anti-shrinkage: away from zero
    } else if coeff == -1 {
        -2 // anti-shrinkage: away from zero
    } else if rounding_error > 0.0 {
        coeff + 1 // precover was above → go up
    } else {
        coeff - 1 // precover was at or below → go down
    }
}

/// Standard nsF5 modification direction (no side info).
///
/// - |coeff| == 1: away from zero
/// - |coeff| > 1: toward zero
#[inline]
pub fn nsf5_modify_coefficient(coeff: i16) -> i16 {
    if coeff == 1 {
        2
    } else if coeff == -1 {
        -2
    } else if coeff > 1 {
        coeff - 1
    } else if coeff < -1 {
        coeff + 1
    } else {
        coeff // zero: should never happen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::pixels::dct_block;

    // --- T1: dct_block_unquantized matches dct_block ---

    fn standard_qt() -> [u16; 64] {
        [
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99,
        ]
    }

    #[test]
    fn t1_unquantized_rounds_to_quantized() {
        // Various test patterns
        let patterns: Vec<[f64; 64]> = vec![
            // Flat gray
            [128.0; 64],
            // Gradient
            {
                let mut p = [0.0f64; 64];
                for i in 0..64 {
                    p[i] = 50.0 + (i as f64) * 3.0;
                }
                p
            },
            // High contrast
            {
                let mut p = [0.0f64; 64];
                for i in 0..64 {
                    p[i] = if i % 2 == 0 { 20.0 } else { 230.0 };
                }
                p
            },
        ];

        let qt = standard_qt();
        for pixels in &patterns {
            let quantized = dct_block(pixels, &qt);
            let unquantized = dct_block_unquantized(pixels, &qt);
            for i in 0..64 {
                assert_eq!(
                    quantized[i],
                    unquantized[i].round() as i16,
                    "Mismatch at index {i}: quantized={}, unquantized={}",
                    quantized[i],
                    unquantized[i]
                );
            }
        }
    }

    // --- T2: Rounding errors in range ---

    #[test]
    fn t2_rounding_errors_in_range() {
        let qt = standard_qt();
        // Test with multiple pixel patterns
        for seed in 0..10u8 {
            let mut pixels = [0.0f64; 64];
            for i in 0..64 {
                pixels[i] = ((seed as f64 * 37.0 + i as f64 * 13.0) % 256.0).abs();
            }
            let quantized = dct_block(&pixels, &qt);
            let unquantized = dct_block_unquantized(&pixels, &qt);
            for i in 0..64 {
                let error = unquantized[i] - quantized[i] as f64;
                assert!(
                    error >= -0.50001 && error <= 0.50001,
                    "seed={seed}, index={i}: error={error}"
                );
            }
        }
    }

    // --- T3: Half-coefficient clamping ---

    #[test]
    fn t3_half_coefficient_cost_not_zero() {
        // When |error| = 0.5, modulated cost must NOT be zero
        let factor = (1.0 - 2.0 * 0.5_f64) as f32; // = 0.0
        let cost = 1.0f32;
        let modulated = (cost * factor).max(MIN_SI_COST);
        assert!(modulated > 0.0, "1/2-coefficient must not have zero cost");
        assert_eq!(modulated, MIN_SI_COST);
    }

    // --- T4: Asymmetric cost modulation ---

    #[test]
    fn t4_si_cost_scales_with_rounding_error() {
        // Larger |error| → lower cost
        let cost = 1.0f32;
        let small_error = 0.1f64;
        let large_error = 0.4f64;

        let small_modulated = (cost * (1.0 - 2.0 * small_error) as f32).max(MIN_SI_COST);
        let large_modulated = (cost * (1.0 - 2.0 * large_error) as f32).max(MIN_SI_COST);

        assert!(
            large_modulated < small_modulated,
            "larger error should give lower cost: small={small_modulated}, large={large_modulated}"
        );
    }

    #[test]
    fn t4_si_costs_never_exceed_original() {
        // Modulated costs should always be <= original
        for error_pct in 0..=50 {
            let error = error_pct as f64 / 100.0;
            let cost = 5.0f32;
            let factor = (1.0 - 2.0 * error) as f32;
            let modulated = (cost * factor).max(MIN_SI_COST);
            assert!(
                modulated <= cost + 1e-6,
                "modulated={modulated} > original={cost} at error={error}"
            );
        }
    }

    // --- T5: Anti-shrinkage preserved ---

    #[test]
    fn t5_anti_shrinkage_preserved() {
        // |coeff| = 1 must always go away from zero
        assert_eq!(si_modify_coefficient(1, -0.4), 2);
        assert_eq!(si_modify_coefficient(1, 0.4), 2);
        assert_eq!(si_modify_coefficient(1, 0.0), 2);
        assert_eq!(si_modify_coefficient(-1, -0.4), -2);
        assert_eq!(si_modify_coefficient(-1, 0.4), -2);
        assert_eq!(si_modify_coefficient(-1, 0.0), -2);
    }

    // --- T6: Direction selection ---

    #[test]
    fn t6_direction_follows_rounding_error() {
        // Positive error → precover above → go up
        assert_eq!(si_modify_coefficient(5, 0.3), 6);
        assert_eq!(si_modify_coefficient(-5, 0.3), -4); // toward zero = up

        // Negative error → precover below → go down
        assert_eq!(si_modify_coefficient(5, -0.3), 4);
        assert_eq!(si_modify_coefficient(-5, -0.3), -6); // away from zero = down

        // Zero error → down (the else branch)
        assert_eq!(si_modify_coefficient(5, 0.0), 4);
        assert_eq!(si_modify_coefficient(-5, 0.0), -6);
    }

    // --- T6b: nsF5 direction ---

    #[test]
    fn t6b_nsf5_toward_zero() {
        assert_eq!(nsf5_modify_coefficient(5), 4);
        assert_eq!(nsf5_modify_coefficient(-5), -4);
        assert_eq!(nsf5_modify_coefficient(2), 1);
        assert_eq!(nsf5_modify_coefficient(-2), -1);
        // Anti-shrinkage
        assert_eq!(nsf5_modify_coefficient(1), 2);
        assert_eq!(nsf5_modify_coefficient(-1), -2);
    }
}
