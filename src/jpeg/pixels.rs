// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Pixel-domain conversion for the Y (luminance) channel.
//!
//! Provides IDCT (coefficient → pixel) and forward DCT (pixel → coefficient)
//! transforms for geometry-resilient steganography. Only operates on the
//! Y channel — no RGB conversion needed since the DFT template is embedded
//! in luminance.

use std::sync::OnceLock;

use crate::jpeg::JpegImage;

/// Pre-computed 8×8 cosine table for IDCT/DCT.
/// `COSINE[u][x] = cos((2*x + 1) * u * PI / 16)`
static COSINE: OnceLock<[[f64; 8]; 8]> = OnceLock::new();

/// Normalization constants: C(0) = 1/sqrt(8), C(u>0) = 1/2.
static NORM: OnceLock<[f64; 8]> = OnceLock::new();

fn cosine_table() -> &'static [[f64; 8]; 8] {
    COSINE.get_or_init(|| {
        let mut table = [[0.0f64; 8]; 8];
        for u in 0..8 {
            for x in 0..8 {
                table[u][x] = crate::det_math::det_cos((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0);
            }
        }
        table
    })
}

fn norm_table() -> &'static [f64; 8] {
    NORM.get_or_init(|| {
        let mut n = [0.5f64; 8];
        n[0] = 1.0 / (8.0f64).sqrt();
        n
    })
}

/// Dequantize + 8×8 IDCT → 64 spatial-domain pixel values.
///
/// Input: quantized DCT coefficients in natural (row-major) order.
/// Output: pixel values (approximately 0–255 after +128 shift).
pub fn idct_block(quantized: &[i16; 64], qt: &[u16; 64]) -> [f64; 64] {
    let cos = cosine_table();
    let c = norm_table();

    // Dequantize
    let mut f = [0.0f64; 64];
    for i in 0..64 {
        f[i] = quantized[i] as f64 * qt[i] as f64;
    }

    // Separable IDCT: columns then rows.
    // Step 1: IDCT on columns (transform over rows for each column).
    let mut temp = [0.0f64; 64];
    for col in 0..8 {
        for y in 0..8 {
            let mut sum = 0.0;
            for v in 0..8 {
                sum += c[v] * f[v * 8 + col] * cos[v][y];
            }
            temp[y * 8 + col] = sum;
        }
    }

    // Step 2: IDCT on rows.
    let mut pixels = [0.0f64; 64];
    for row in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0;
            for u in 0..8 {
                sum += c[u] * temp[row * 8 + u] * cos[u][x];
            }
            pixels[row * 8 + x] = sum + 128.0;
        }
    }

    pixels
}

/// 8×8 forward DCT + quantize → 64 DCT coefficients.
///
/// Input: pixel values (expected ~0–255).
/// Output: quantized DCT coefficients in natural (row-major) order.
pub fn dct_block(pixels: &[f64; 64], qt: &[u16; 64]) -> [i16; 64] {
    let cos = cosine_table();
    let c = norm_table();

    // Level shift: subtract 128
    let mut shifted = [0.0f64; 64];
    for i in 0..64 {
        shifted[i] = pixels[i] - 128.0;
    }

    // Separable forward DCT: rows then columns.
    // Step 1: DCT on rows.
    let mut temp = [0.0f64; 64];
    for row in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0;
            for x in 0..8 {
                sum += shifted[row * 8 + x] * cos[u][x];
            }
            temp[row * 8 + u] = c[u] * sum;
        }
    }

    // Step 2: DCT on columns.
    let mut coeffs = [0.0f64; 64];
    for col in 0..8 {
        for v in 0..8 {
            let mut sum = 0.0;
            for y in 0..8 {
                sum += temp[y * 8 + col] * cos[v][y];
            }
            coeffs[v * 8 + col] = c[v] * sum;
        }
    }

    // Quantize
    let mut quantized = [0i16; 64];
    for i in 0..64 {
        quantized[i] = (coeffs[i] / qt[i] as f64).round() as i16;
    }
    quantized
}

/// Convert entire Y-channel DctGrid → row-major f64 pixel array.
///
/// Returns (pixels, width_in_pixels, height_in_pixels) where dimensions
/// are the full block-aligned size (multiples of 8).
pub fn jpeg_to_luma_f64(img: &JpegImage) -> (Vec<f64>, usize, usize) {
    let grid = img.dct_grid(0);
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).expect("Y quant table must exist");

    let bw = grid.blocks_wide();
    let bh = grid.blocks_tall();
    let width = bw * 8;
    let height = bh * 8;
    let mut pixels = vec![0.0f64; width * height];

    for br in 0..bh {
        for bc in 0..bw {
            let block = grid.block(br, bc);
            let quantized: [i16; 64] = block.try_into().unwrap();
            let block_pixels = idct_block(&quantized, &qt.values);

            for row in 0..8 {
                for col in 0..8 {
                    let py = br * 8 + row;
                    let px = bc * 8 + col;
                    pixels[py * width + px] = block_pixels[row * 8 + col];
                }
            }
        }
    }

    (pixels, width, height)
}

/// Write f64 pixel array back into Y-channel DctGrid.
///
/// Performs forward DCT + quantization on each 8×8 block.
pub fn luma_f64_to_jpeg(pixels: &[f64], width: usize, height: usize, img: &mut JpegImage) {
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt_values = img.quant_table(qt_id).expect("Y quant table must exist").values;
    let grid = img.dct_grid_mut(0);
    let bw = grid.blocks_wide();
    let bh = grid.blocks_tall();

    assert_eq!(width, bw * 8);
    assert_eq!(height, bh * 8);

    for br in 0..bh {
        for bc in 0..bw {
            let mut block_pixels = [0.0f64; 64];
            for row in 0..8 {
                for col in 0..8 {
                    let py = br * 8 + row;
                    let px = bc * 8 + col;
                    block_pixels[row * 8 + col] = pixels[py * width + px];
                }
            }

            let quantized = dct_block(&block_pixels, &qt_values);
            let block = grid.block_mut(br, bc);
            block.copy_from_slice(&quantized);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct_dct_roundtrip() {
        // Create some DCT coefficients
        let mut quantized = [0i16; 64];
        quantized[0] = 100; // DC
        quantized[1] = 10;
        quantized[8] = -5;
        quantized[9] = 3;

        // Unity quantization table
        let qt = [1u16; 64];

        let pixels = idct_block(&quantized, &qt);
        let recovered = dct_block(&pixels, &qt);

        for i in 0..64 {
            assert!(
                (quantized[i] - recovered[i]).abs() <= 1,
                "Mismatch at index {i}: expected {}, got {}",
                quantized[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn dc_only_block_produces_flat_pixels() {
        let mut quantized = [0i16; 64];
        quantized[0] = 16; // DC coefficient
        let qt = [1u16; 64];

        let pixels = idct_block(&quantized, &qt);

        // All pixels should be approximately the same value
        // DC contribution = C(0)_col * C(0)_row * F[0][0] = (1/sqrt(8))^2 * 16 = 16/8 = 2
        let expected = 128.0 + 16.0 / 8.0;
        let dc_val = pixels[0];
        for i in 0..64 {
            assert!(
                (pixels[i] - dc_val).abs() < 1e-10,
                "Pixel {i} = {}, expected uniform {}",
                pixels[i],
                dc_val
            );
        }
        assert!((dc_val - expected).abs() < 1e-10);
    }

    #[test]
    fn idct_dct_roundtrip_with_quant() {
        // Typical JPEG quantization table
        let qt = [
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99u16,
        ];

        let mut quantized = [0i16; 64];
        quantized[0] = 50;
        quantized[1] = -3;
        quantized[8] = 2;

        let pixels = idct_block(&quantized, &qt);
        let recovered = dct_block(&pixels, &qt);

        for i in 0..64 {
            assert!(
                (quantized[i] - recovered[i]).abs() <= 1,
                "Mismatch at index {i}: expected {}, got {}",
                quantized[i],
                recovered[i]
            );
        }
    }
}
