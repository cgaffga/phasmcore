// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Bilinear resampling for undoing geometric transforms.
//!
//! Inverse-maps each output pixel through the estimated affine transform
//! to find its source coordinates, then bilinear-interpolates from the
//! four nearest source pixels.

use crate::stego::armor::template::AffineTransform;

/// Inverse-map bilinear resampling to undo a geometric transform.
///
/// For each output pixel, computes source coordinates by applying the
/// inverse transform (rotate by -θ, scale by 1/s), then bilinear
/// interpolates. Out-of-bounds pixels default to 128.0 (mid-gray).
pub fn resample_bilinear(
    pixels: &[f64],
    src_w: usize,
    src_h: usize,
    transform: &AffineTransform,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f64> {
    let mut result = vec![128.0f64; dst_w * dst_h];

    // Inverse transform: rotate by -θ, scale by 1/s
    let (sin_t, cos_t) = crate::det_math::det_sincos(transform.rotation_rad);
    let inv_scale = if transform.scale.abs() > 1e-12 { 1.0 / transform.scale } else { 1.0 };

    // Centers of source and destination
    let src_cx = src_w as f64 / 2.0;
    let src_cy = src_h as f64 / 2.0;
    let dst_cx = dst_w as f64 / 2.0;
    let dst_cy = dst_h as f64 / 2.0;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Offset from destination center
            let x = dx as f64 - dst_cx;
            let y = dy as f64 - dst_cy;

            // Inverse rotation (rotate by -θ)
            let xr = x * cos_t + y * sin_t;
            let yr = -x * sin_t + y * cos_t;

            // Inverse scale
            let xs = xr * inv_scale;
            let ys = yr * inv_scale;

            // Map to source coordinates
            let sx = xs + src_cx;
            let sy = ys + src_cy;

            // Bilinear interpolation
            result[dy * dst_w + dx] = bilinear_sample(pixels, src_w, src_h, sx, sy);
        }
    }

    result
}

/// Sample a pixel from the image using bilinear interpolation.
///
/// Returns 128.0 (mid-gray) for out-of-bounds coordinates.
fn bilinear_sample(pixels: &[f64], w: usize, h: usize, x: f64, y: f64) -> f64 {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let get = |px: i64, py: i64| -> f64 {
        if px >= 0 && px < w as i64 && py >= 0 && py < h as i64 {
            pixels[py as usize * w + px as usize]
        } else {
            128.0
        }
    };

    let v00 = get(x0, y0);
    let v10 = get(x1, y0);
    let v01 = get(x0, y1);
    let v11 = get(x1, y1);

    v00 * (1.0 - fx) * (1.0 - fy)
        + v10 * fx * (1.0 - fy)
        + v01 * (1.0 - fx) * fy
        + v11 * fx * fy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_transform_preserves_image() {
        let w = 16;
        let h = 16;
        let pixels: Vec<f64> = (0..w * h).map(|i| (i as f64) * 1.5 + 10.0).collect();

        let identity = AffineTransform {
            rotation_rad: 0.0,
            scale: 1.0,
        };

        let result = resample_bilinear(&pixels, w, h, &identity, w, h);

        // Interior pixels should match closely (edges may differ due to interpolation)
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = y * w + x;
                assert!(
                    (pixels[idx] - result[idx]).abs() < 0.01,
                    "Mismatch at ({x},{y}): expected {}, got {}",
                    pixels[idx],
                    result[idx]
                );
            }
        }
    }

    #[test]
    fn rotation_180_roundtrip() {
        let w = 16;
        let h = 16;
        let pixels: Vec<f64> = (0..w * h).map(|i| (i % 7) as f64 * 30.0 + 50.0).collect();

        // Rotate by 180° then rotate by 180° again
        let rot180 = AffineTransform {
            rotation_rad: std::f64::consts::PI,
            scale: 1.0,
        };

        let rotated = resample_bilinear(&pixels, w, h, &rot180, w, h);
        let roundtrip = resample_bilinear(&rotated, w, h, &rot180, w, h);

        // Interior pixels should match
        for y in 2..h - 2 {
            for x in 2..w - 2 {
                let idx = y * w + x;
                assert!(
                    (pixels[idx] - roundtrip[idx]).abs() < 1.0,
                    "Mismatch at ({x},{y}): expected {}, got {}",
                    pixels[idx],
                    roundtrip[idx]
                );
            }
        }
    }

    #[test]
    fn scale_2x_then_half() {
        let w = 16;
        let h = 16;
        let pixels: Vec<f64> = (0..w * h).map(|i| 128.0 + (i as f64 * 0.5).sin() * 40.0).collect();

        let scale2 = AffineTransform {
            rotation_rad: 0.0,
            scale: 2.0,
        };
        let scale_half = AffineTransform {
            rotation_rad: 0.0,
            scale: 0.5,
        };

        let scaled_up = resample_bilinear(&pixels, w, h, &scale2, w, h);
        let roundtrip = resample_bilinear(&scaled_up, w, h, &scale_half, w, h);

        // Center region should roughly match
        for y in 4..h - 4 {
            for x in 4..w - 4 {
                let idx = y * w + x;
                assert!(
                    (pixels[idx] - roundtrip[idx]).abs() < 5.0,
                    "Large mismatch at ({x},{y}): expected {}, got {}",
                    pixels[idx],
                    roundtrip[idx]
                );
            }
        }
    }
}
