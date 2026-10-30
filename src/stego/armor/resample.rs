// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Bilinear resampling for undoing geometric transforms.
//!
//! Inverse-maps each output pixel through the estimated affine transform
//! to find its source coordinates, then bilinear-interpolates from the
//! four nearest source pixels.
//!
//! `CorrectedLuma` produces the same corrected image one block row at a time
//! (the decoder's bounded-memory path); `resample_bilinear` is the whole-image
//! form. Both go through `InverseMap::source` and `bilinear`, so they agree
//! bit for bit.

use std::collections::VecDeque;

use crate::codec::jpeg::dct::DctGrid;
use crate::codec::jpeg::pixels::{dct_block_f64, luma_block_row_f64};
use crate::stego::armor::template::AffineTransform;

/// The inverse transform: destination pixel → source coordinates.
#[derive(Clone, Copy, Debug)]
struct InverseMap {
    sin_t: f64,
    cos_t: f64,
    inv_scale: f64,
    src_cx: f64,
    src_cy: f64,
    dst_cx: f64,
    dst_cy: f64,
}

impl InverseMap {
    fn new(transform: &AffineTransform, src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Self {
        // Inverse transform: rotate by -θ, scale by 1/s
        let (sin_t, cos_t) = crate::det_math::det_sincos(transform.rotation_rad);
        let inv_scale = if transform.scale.abs() > 1e-12 { 1.0 / transform.scale } else { 1.0 };
        InverseMap {
            sin_t,
            cos_t,
            inv_scale,
            // Centers of source and destination
            src_cx: src_w as f64 / 2.0,
            src_cy: src_h as f64 / 2.0,
            dst_cx: dst_w as f64 / 2.0,
            dst_cy: dst_h as f64 / 2.0,
        }
    }

    /// Source coordinates of destination pixel (dx, dy).
    #[inline]
    fn source(&self, dx: usize, dy: usize) -> (f64, f64) {
        // Offset from destination center
        let x = dx as f64 - self.dst_cx;
        let y = dy as f64 - self.dst_cy;

        // Inverse rotation (rotate by -θ)
        let xr = x * self.cos_t + y * self.sin_t;
        let yr = -x * self.sin_t + y * self.cos_t;

        // Inverse scale
        let xs = xr * self.inv_scale;
        let ys = yr * self.inv_scale;

        // Map to source coordinates
        (xs + self.src_cx, ys + self.src_cy)
    }
}

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
    let map = InverseMap::new(transform, src_w, src_h, dst_w, dst_h);

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let (sx, sy) = map.source(dx, dy);
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
    bilinear(|px, py| pixels[py * w + px], w, h, x, y)
}

/// Bilinear interpolation at (x, y); `get(px, py)` is called only for
/// in-bounds pixels, out-of-bounds ones read 128.0 (mid-gray).
#[inline]
fn bilinear<G: Fn(usize, usize) -> f64>(get: G, w: usize, h: usize, x: f64, y: f64) -> f64 {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let get = |px: i64, py: i64| -> f64 {
        if px >= 0 && px < w as i64 && py >= 0 && py < h as i64 {
            get(px as usize, py as usize)
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

/// The luma of a geometry-corrected image, one block row at a time.
///
/// Block row `br` of `next_block_row` equals block row `br` of the Y grid
/// `luma_f64_to_jpeg(resample_bilinear(jpeg_to_luma_f64(img), T))` leaves,
/// bit for bit, without the luma image, the resampled image or a second grid
/// in memory: the source rows a block row samples are decoded on demand
/// (IDCT of their block rows) into a band that slides with the output, so the
/// working set is the band (the rows one output block row spans in the source
/// — a few hundred for the near-identity transforms template estimation
/// yields) plus one block row of output. The source grid is only read.
pub struct CorrectedLuma<'a> {
    grid: &'a DctGrid,
    qt: [u16; 64],
    map: InverseMap,
    w: usize,
    h: usize,
    /// Decoded source block rows `band_first ..` (each `8 * w` pixels).
    band: VecDeque<Vec<f64>>,
    band_first: usize,
    spare: Vec<Vec<f64>>,
}

impl<'a> CorrectedLuma<'a> {
    /// The corrected image has the grid's (block-aligned) size, like the
    /// whole-image path, which resamples `w × h` onto `w × h`.
    pub fn new(grid: &'a DctGrid, qt: &[u16; 64], transform: &AffineTransform) -> Self {
        let w = grid.blocks_wide() * 8;
        let h = grid.blocks_tall() * 8;
        CorrectedLuma {
            grid,
            qt: *qt,
            map: InverseMap::new(transform, w, h, w, h),
            w,
            h,
            band: VecDeque::new(),
            band_first: 0,
            spare: Vec::new(),
        }
    }

    pub fn blocks_wide(&self) -> usize {
        self.grid.blocks_wide()
    }

    pub fn blocks_tall(&self) -> usize {
        self.grid.blocks_tall()
    }

    /// Source pixel rows (inclusive) bilinear sampling reads for output block
    /// row `br`, or `None` if every sample lands outside the image. The
    /// source y of a pixel is monotone in its x and in its y (every step of
    /// `InverseMap::source` is a monotone rounding), so the extremes are at
    /// the row ends; all 8 rows are checked anyway.
    fn source_rows(&self, br: usize) -> Option<(usize, usize)> {
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        for dy in br * 8..br * 8 + 8 {
            for dx in [0, self.w - 1] {
                let y0 = self.map.source(dx, dy).1.floor() as i64;
                lo = lo.min(y0);
                hi = hi.max(y0.saturating_add(1));
            }
        }
        let lo = lo.max(0);
        let hi = hi.min(self.h as i64 - 1);
        (lo <= hi).then_some((lo as usize, hi as usize))
    }

    fn decode_block_row(&mut self, sbr: usize) -> Vec<f64> {
        let mut rows = self.spare.pop().unwrap_or_else(|| vec![0.0; 8 * self.w]);
        luma_block_row_f64(self.grid, &self.qt, sbr, &mut rows);
        rows
    }

    /// Make the band hold exactly source block rows `blo ..= bhi`.
    fn ensure_band(&mut self, blo: usize, bhi: usize) {
        let band_end = self.band_first + self.band.len(); // exclusive
        if self.band.is_empty() || blo >= band_end || bhi < self.band_first {
            self.spare.extend(self.band.drain(..));
            self.band_first = blo;
        }
        while self.band_first < blo && !self.band.is_empty() {
            self.spare.push(self.band.pop_front().unwrap());
            self.band_first += 1;
        }
        while self.band_first + self.band.len() > bhi + 1 {
            self.spare.push(self.band.pop_back().unwrap());
        }
        if self.band.is_empty() {
            self.band_first = blo;
        }
        while self.band_first > blo {
            let rows = self.decode_block_row(self.band_first - 1);
            self.band.push_front(rows);
            self.band_first -= 1;
        }
        while self.band_first + self.band.len() <= bhi {
            let rows = self.decode_block_row(self.band_first + self.band.len());
            self.band.push_back(rows);
        }
    }

    /// Quantized coefficients of output block row `br` (`out.len()` = blocks wide).
    pub fn block_row(&mut self, br: usize, out: &mut [[i16; 64]]) {
        assert_eq!(out.len(), self.grid.blocks_wide());
        if let Some((lo, hi)) = self.source_rows(br) {
            self.ensure_band(lo / 8, hi / 8);
        }
        let (w, h, map, qt) = (self.w, self.h, self.map, self.qt);
        let band = &self.band;
        let band_first = self.band_first;
        let get = |px: usize, py: usize| -> f64 {
            // `source_rows` put every in-bounds row this block row reads in
            // the band; indexing panics otherwise (a bug, not bad input).
            band[py / 8 - band_first][(py % 8) * w + px]
        };
        let do_block = |(bc, block): (usize, &mut [i16; 64])| {
            let mut block_pixels = [0.0f64; 64];
            for row in 0..8 {
                for col in 0..8 {
                    let (sx, sy) = map.source(bc * 8 + col, br * 8 + row);
                    block_pixels[row * 8 + col] = bilinear(get, w, h, sx, sy);
                }
            }
            *block = dct_block_f64(&block_pixels, &qt);
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out.par_iter_mut().enumerate().for_each(do_block);
        }
        #[cfg(not(feature = "parallel"))]
        out.iter_mut().enumerate().for_each(do_block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrected_luma_matches_the_whole_image_path() {
        use crate::codec::jpeg::pixels::{dct_block_f64, luma_block_row_f64};
        let (bw, bt) = (9usize, 7usize);
        let (w, h) = (bw * 8, bt * 8);
        let mut grid = DctGrid::new(bw, bt);
        let mut state = 12345u32;
        for br in 0..bt {
            for bc in 0..bw {
                for k in 0..64 {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let span = if k == 0 { 120 } else { 24 / (1 + k as i32 / 8) };
                    let v = (state >> 8) as i32 % (2 * span + 1) - span;
                    grid.set(br, bc, k / 8, k % 8, v as i16);
                }
            }
        }
        let qt: [u16; 64] = std::array::from_fn(|k| 2 + (k as u16 % 13));
        let mut luma = vec![0.0f64; w * h];
        for br in 0..bt {
            luma_block_row_f64(&grid, &qt, br, &mut luma[br * 8 * w..(br + 1) * 8 * w]);
        }
        let transforms = [
            (0.0, 1.0),
            (0.0021, 0.9987),
            (-0.03, 1.02),
            (0.5, 1.0),
            (3.0, 1.0),
            (0.1, 0.5),
            (-0.2, 2.0),
        ];
        for &(rot, scale) in &transforms {
            let t = AffineTransform { rotation_rad: rot, scale };
            let resampled = resample_bilinear(&luma, w, h, &t, w, h);
            let mut corrected = CorrectedLuma::new(&grid, &qt, &t);
            // Forward, then backward, to exercise the band in both directions.
            let order: Vec<usize> = (0..bt).chain((0..bt).rev()).collect();
            for &br in &order {
                let mut out = vec![[0i16; 64]; bw];
                corrected.block_row(br, &mut out);
                for bc in 0..bw {
                    let mut px = [0.0f64; 64];
                    for row in 0..8 {
                        for col in 0..8 {
                            px[row * 8 + col] = resampled[(br * 8 + row) * w + bc * 8 + col];
                        }
                    }
                    assert_eq!(out[bc], dct_block_f64(&px, &qt), "rot {rot} scale {scale} block ({br},{bc})");
                }
            }
        }
    }

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
