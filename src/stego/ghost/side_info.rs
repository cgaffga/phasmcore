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

use crate::codec::jpeg::dct::DctGrid;
use crate::codec::jpeg::pixels::dct_block_unquantized;

/// Per-coefficient rounding errors from quantization.
///
/// Each error is in [-0.5, +0.5] and represents how far the continuous
/// (unquantized) DCT coefficient was from its rounded integer value.
/// Positive error means the pre-quantization value was above the integer;
/// negative means below.
///
/// Stored as i8 [-127, +127] via `error * 254`, giving ~0.004 resolution.
/// This is 8x smaller than f64 and 4x smaller than f32, saving 85 MB (12MP)
/// or 341 MB (48MP) compared to the original f64 representation.
pub struct SideInfo {
    /// Rounding errors in DctGrid flat order (block_idx * 64 + row * 8 + col).
    /// Encoded as i8: value = (error * 254).round().clamp(-127, 127).
    rounding_errors: Vec<i8>,
    /// Number of 8x8 blocks horizontally.
    pub blocks_wide: usize,
    /// Number of 8x8 blocks vertically.
    pub blocks_tall: usize,
    /// True when the errors are the cover encoder's own rounding (the cover
    /// was written by phasm's libjpeg-turbo-compatible writer, verified
    /// coefficient by coefficient). Then `|e| = ½` ties are exact (stored as
    /// ±127) and every other error is stored as `e · 252` (≤ ±126).
    exact: bool,
}

/// Encode a rounding error [-0.5, +0.5] to i8 [-127, +127].
#[inline]
fn encode_error(error: f64) -> i8 {
    (error * 254.0).round().clamp(-127.0, 127.0) as i8
}

/// Decode an i8 error back to approximate f32.
#[inline]
fn decode_error(val: i8) -> f32 {
    val as f32 / 254.0
}

/// Minimum cost for SI-modulated coefficients.
///
/// When |rounding_error| ~ 0.5 ("1/2-coefficients"), the modulated cost
/// approaches zero. Clamping to this floor prevents zero-cost embedding
/// at quantization midpoints, which is a known detectable artifact.
///
/// Test-only: the production SI cost (`ghost::v31`, floor + 0.3 factor
/// floor) is applied inline by the encoder; the tests below pin the formula.
#[cfg(test)]
const MIN_SI_COST: f32 = 1e-6;

impl SideInfo {
    /// Compute side information from raw RGB pixels and the cover JPEG.
    ///
    /// For each Y-channel 8x8 block:
    /// 1. Forward DCT on the original (pre-JPEG) pixels
    /// 2. Divide by quantization table (without rounding)
    /// 3. error = unquantized_value - cover_integer_coefficient
    ///
    /// Errors are clamped to [-0.5, +0.5] for robustness against minor
    /// floating-point differences between the platform's JPEG encoder and
    /// our forward DCT implementation.
    ///
    /// Luma blocks are computed in strips of 50 block-rows to limit transient
    /// memory (~12.9 MB per strip instead of ~97.5 MB for all blocks at once
    /// on a 12MP image).
    pub fn compute(
        raw_rgb: &[u8],
        pixel_width: u32,
        pixel_height: u32,
        cover_grid: &DctGrid,
        qt_values: &[u16; 64],
    ) -> Result<Self, crate::StegoError> {
        check_pixels(raw_rgb, pixel_width, pixel_height)?;
        let w = pixel_width as usize;
        let h = pixel_height as usize;

        let bw = cover_grid.blocks_wide();
        let bh = cover_grid.blocks_tall();
        let total_coeffs = bw * bh * 64;

        // Exact side information when the cover's luma is what phasm's
        // cover writer (= libjpeg-turbo islow) makes of these pixels.
        if let Some(si) = Self::compute_exact(raw_rgb, pixel_width, pixel_height, cover_grid, qt_values) {
            return Ok(si);
        }

        let mut errors = vec![0i8; total_coeffs];
        // Pixels that do not belong to this cover (other dimensions, e.g. a
        // frontend that downscaled the JPEG but not the pixels) carry no side
        // information: embed without it rather than with misaligned errors.
        if !Self::geometry_matches(w, h, bw, bh) {
            return Ok(SideInfo { rounding_errors: errors, blocks_wide: bw, blocks_tall: bh, exact: false });
        }

        let luma_bw = w.div_ceil(8);
        let luma_bh = h.div_ceil(8);

        // Process luma blocks in strips to limit transient memory.
        // Each strip holds at most STRIP_ROWS block-rows of luma data.
        const STRIP_ROWS: usize = 50;
        for strip_start in (0..bh).step_by(STRIP_ROWS) {
            let strip_end = (strip_start + STRIP_ROWS).min(bh);
            let luma_strip = rgb_to_luma_blocks_strip(
                raw_rgb, pixel_width, pixel_height, strip_start, strip_end,
            );

            for br in strip_start..strip_end {
                for bc in 0..bw {
                    let block_idx = br * bw + bc;

                    // Skip if outside the raw pixel grid
                    if br >= luma_bh || bc >= luma_bw {
                        continue; // leave errors at 0
                    }

                    let local_idx = (br - strip_start) * luma_bw + bc;
                    let luma_block = &luma_strip[local_idx];

                    // Forward DCT + divide by QT (no rounding)
                    let unquantized = dct_block_unquantized(luma_block, qt_values);

                    // Compute and clamp rounding errors, encode to i8
                    let cover_block: [i16; 64] = {
                        let slice = cover_grid.block(br, bc);
                        slice.try_into().unwrap()
                    };

                    for k in 0..64 {
                        let error = (unquantized[k] - cover_block[k] as f64).clamp(-0.5, 0.5);
                        errors[block_idx * 64 + k] = encode_error(error);
                    }
                }
            }
            // luma_strip dropped here -- only one strip in memory at a time
        }

        Ok(SideInfo {
            rounding_errors: errors,
            blocks_wide: bw,
            blocks_tall: bh,
            exact: false,
        })
    }

    /// The side information an embed into `cover_grid` uses: [`Self::compute`],
    /// or `None` (plain Ghost v3) when the pixels are a JPEG decoded on this
    /// block grid ([`is_decoded_jpeg_on_grid`]). Errors as `compute` does on
    /// an empty or short pixel buffer.
    pub fn for_embed(
        raw_rgb: &[u8],
        pixel_width: u32,
        pixel_height: u32,
        cover_grid: &DctGrid,
        qt_values: &[u16; 64],
    ) -> Result<Option<Self>, crate::StegoError> {
        check_pixels(raw_rgb, pixel_width, pixel_height)?;
        if Self::geometry_matches(pixel_width as usize, pixel_height as usize, cover_grid.blocks_wide(), cover_grid.blocks_tall())
            && is_decoded_jpeg_on_grid(raw_rgb, pixel_width, pixel_height)
        {
            return Ok(None);
        }
        Self::compute(raw_rgb, pixel_width, pixel_height, cover_grid, qt_values).map(Some)
    }

    /// Luma block grid `bw × bh` can hold a `w × h` image (MCU 8 or 16).
    fn geometry_matches(w: usize, h: usize, bw: usize, bh: usize) -> bool {
        let (wib, hib) = (w.div_ceil(8), h.div_ceil(8));
        (wib..=wib + 1).contains(&bw) && (hib..=hib + 1).contains(&bh)
    }

    /// Exact side information, or `None` unless every luma coefficient of
    /// `cover_grid` equals what libjpeg-turbo's islow path (phasm's cover
    /// writer, [`crate::codec::jpeg::cover_writer`]) computes from
    /// `raw_rgb` with this quantization table. Luma does not depend on the
    /// chroma subsampling, so any libjpeg-turbo islow cover with these tables
    /// qualifies.
    pub fn compute_exact(
        raw_rgb: &[u8],
        pixel_width: u32,
        pixel_height: u32,
        cover_grid: &DctGrid,
        qt_values: &[u16; 64],
    ) -> Option<Self> {
        use crate::codec::jpeg::cover_writer::{for_each_luma_block, Geometry};
        let (w, h) = (pixel_width as usize, pixel_height as usize);
        let (bw, bh) = (cover_grid.blocks_wide(), cover_grid.blocks_tall());
        if w == 0 || h == 0 || raw_rgb.len() < w * h * 3 || !Self::geometry_matches(w, h, bw, bh) {
            return None;
        }
        let g = Geometry::new(w, h);
        let mut errors = vec![0i8; bw * bh * 64];
        let ok = for_each_luma_block(raw_rgb, &g, qt_values, |br, bc, raw, q| {
            let cover = cover_grid.block(br, bc);
            let base = (br * bw + bc) * 64;
            for k in 0..64 {
                if cover[k] != q[k] {
                    return false;
                }
                errors[base + k] = exact_error_code(raw[k], q[k], qt_values[k]);
            }
            true
        });
        // Dummy blocks (beyond the image, right/bottom MCU padding) keep 0:
        // they have no precover.
        ok.then_some(SideInfo { rounding_errors: errors, blocks_wide: bw, blocks_tall: bh, exact: true })
    }

    /// Whether [`Self::compute_exact`] would accept these pixels for this
    /// cover, without building the side information (no allocation; stops
    /// at the first coefficient that differs). The capacity bands use it:
    /// the Deep Cover capacity bonus applies exactly when an `encode_si*`
    /// call with the same arguments embeds with exact side information.
    pub fn is_exact_for(
        raw_rgb: &[u8],
        pixel_width: u32,
        pixel_height: u32,
        cover_grid: &DctGrid,
        qt_values: &[u16; 64],
    ) -> bool {
        use crate::codec::jpeg::cover_writer::{for_each_luma_block, Geometry};
        let (w, h) = (pixel_width as usize, pixel_height as usize);
        let (bw, bh) = (cover_grid.blocks_wide(), cover_grid.blocks_tall());
        if w == 0 || h == 0 || raw_rgb.len() < w * h * 3 || !Self::geometry_matches(w, h, bw, bh) {
            return false;
        }
        for_each_luma_block(raw_rgb, &Geometry::new(w, h), qt_values, |br, bc, _, q| {
            cover_grid.block(br, bc) == &q[..]
        })
    }

    /// Exact side information from [`exact_error_code`] values (flat luma
    /// order, `bw × bh` blocks), as the Deep Cover writer produces them.
    pub(crate) fn exact_from_codes(rounding_errors: Vec<i8>, blocks_wide: usize, blocks_tall: usize) -> Self {
        SideInfo { rounding_errors, blocks_wide, blocks_tall, exact: true }
    }

    /// Get the rounding error at a flat index, decoded from i8 to f32.
    #[inline]
    pub fn error_at(&self, flat_idx: usize) -> f32 {
        let v = self.rounding_errors[flat_idx];
        if self.exact {
            if v.unsigned_abs() == 127 { v.signum() as f32 * 0.5 } else { v as f32 / 252.0 }
        } else {
            decode_error(v)
        }
    }

    /// Whether the errors are the cover encoder's own (see [`Self::compute_exact`]).
    #[inline]
    pub fn is_exact(&self) -> bool {
        self.exact
    }

    /// Exact `|e| = ½` tie: both neighbours are equally far from the
    /// precover (only meaningful when [`Self::is_exact`]).
    #[inline]
    pub fn is_tie(&self, flat_idx: usize) -> bool {
        self.exact && self.rounding_errors[flat_idx].unsigned_abs() == 127
    }
}

/// `ImageTooSmall` unless `raw_rgb` holds `w · h` RGB pixels.
fn check_pixels(raw_rgb: &[u8], w: u32, h: u32) -> Result<(), crate::StegoError> {
    let expected = (w as usize).checked_mul(h as usize).and_then(|n| n.checked_mul(3)).unwrap_or(0);
    if w == 0 || h == 0 || raw_rgb.len() < expected {
        return Err(crate::StegoError::ImageTooSmall);
    }
    Ok(())
}

/// The pixels are a JPEG decoded on the block grid a cover written from them
/// uses (same size, or a grid-preserving transform), i.e. **not** a precover:
/// their DCT sits on the source's quantization lattice, the cover is an
/// aligned double compression, and side information — exact or approximate —
/// would make the comb-filling changes cheap (histogram attack P_E 0.04 at
/// 0.20 bpnzAC, `eval/runs/2026-10-31-wand-ghost/RESULTS.md` arm D).
/// Encodes then embed without side information and the bands use nzAC / 10
/// ([`super::capacity::SiKind::DecodedJpeg`]).
/// The test: [`crate::codec::jpeg::history`].
pub fn is_decoded_jpeg_on_grid(raw_rgb: &[u8], pixel_width: u32, pixel_height: u32) -> bool {
    crate::codec::jpeg::history::is_aligned_jpeg_decode(raw_rgb, pixel_width, pixel_height)
}

/// i8 code of the exact rounding error of one coefficient: islow output
/// `raw` (libjpeg scale, divisor `8q`), quantized value `c`. Ties
/// (`|raw − 8qc| = 4q`) → ±127, otherwise `round(e · 252)` (≤ ±126).
#[inline]
pub(crate) fn exact_error_code(raw: i16, c: i16, q: u16) -> i8 {
    let d = (q as i32) << 3;
    let r = raw as i32 - d * c as i32; // e = r / d, |r| ≤ d/2
    if 2 * r.abs() >= d {
        return if r > 0 { 127 } else { -127 };
    }
    // round(r · 252 / d) half away from zero, in integers.
    let num = r * 252;
    let v = (num.abs() + d / 2) / d;
    (v.min(126) * num.signum()) as i8
}

/// Convert a horizontal strip of RGB pixels to Y (luminance) 8x8 blocks.
///
/// Only converts block rows in `[br_start, br_end)`, returning them in
/// raster order with `luma_bw` blocks per row. This avoids allocating
/// ALL luma blocks at once (97.5 MB for 12MP, 390 MB for 48MP).
///
/// Uses BT.601: `Y = 0.299*R + 0.587*G + 0.114*B`.
/// Handles non-multiple-of-8 dimensions by edge-replicating.
fn rgb_to_luma_blocks_strip(
    rgb: &[u8],
    width: u32,
    height: u32,
    br_start: usize,
    br_end: usize,
) -> Vec<[f64; 64]> {
    let w = width as usize;
    let h = height as usize;
    let luma_bw = w.div_ceil(8);
    let luma_bh = h.div_ceil(8);

    let strip_br_end = br_end.min(luma_bh);
    let strip_rows = strip_br_end.saturating_sub(br_start);

    let mut blocks = Vec::with_capacity(strip_rows * luma_bw);

    for br in br_start..strip_br_end {
        for bc in 0..luma_bw {
            let mut block = [0.0f64; 64];
            for row in 0..8 {
                for col in 0..8 {
                    let py = (br * 8 + row).min(h - 1);
                    let px = (bc * 8 + col).min(w - 1);
                    let idx = (py * w + px) * 3;
                    let r = rgb[idx] as f64;
                    let g = rgb[idx + 1] as f64;
                    let b = rgb[idx + 2] as f64;
                    block[row * 8 + col] = 0.299 * r + 0.587 * g + 0.114 * b;
                }
            }
            blocks.push(block);
        }
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::jpeg::pixels::dct_block;

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
                    (-0.50001..=0.50001).contains(&error),
                    "seed={seed}, index={i}: error={error}"
                );
            }
        }
    }

    // --- T3: Half-coefficient clamping ---

    #[test]
    fn t3_half_coefficient_cost_not_zero() {
        // When |error| = 0.5, modulated cost must NOT be zero
        let factor = 1.0f32 - 2.0 * 0.5_f32; // = 0.0
        let cost = 1.0f32;
        let modulated = (cost * factor).max(MIN_SI_COST);
        assert!(modulated > 0.0, "1/2-coefficient must not have zero cost");
        assert_eq!(modulated, MIN_SI_COST);
    }

    // --- T4: Asymmetric cost modulation ---

    #[test]
    fn t4_si_cost_scales_with_rounding_error() {
        // Larger |error| -> lower cost
        let cost = 1.0f32;
        let small_error = 0.1f32;
        let large_error = 0.4f32;

        let small_modulated = (cost * (1.0f32 - 2.0 * small_error)).max(MIN_SI_COST);
        let large_modulated = (cost * (1.0f32 - 2.0 * large_error)).max(MIN_SI_COST);

        assert!(
            large_modulated < small_modulated,
            "larger error should give lower cost: small={small_modulated}, large={large_modulated}"
        );
    }

    #[test]
    fn t4_si_costs_never_exceed_original() {
        // Modulated costs should always be <= original
        for error_pct in 0..=50 {
            let error = error_pct as f32 / 100.0;
            let cost = 5.0f32;
            let factor = 1.0f32 - 2.0 * error;
            let modulated = (cost * factor).max(MIN_SI_COST);
            assert!(
                modulated <= cost + 1e-6,
                "modulated={modulated} > original={cost} at error={error}"
            );
        }
    }

    // --- T5: Anti-shrinkage preserved ---

    // --- T6: Direction selection ---

    // --- T6b: nsF5 direction ---

    // --- T7: i8 encode/decode roundtrip ---

    #[test]
    fn t7_i8_encode_decode_precision() {
        // Test that encode_error/decode_error roundtrip has <1% error
        // for the cost modulation factor (1 - 2|e|).
        for i in 0..=100 {
            let error = (i as f64 - 50.0) / 100.0; // [-0.5, +0.5]
            let encoded = encode_error(error);
            let decoded = decode_error(encoded);

            // Check the modulation factor precision
            let original_factor = 1.0 - 2.0 * error.abs();
            let decoded_factor = 1.0f32 - 2.0 * decoded.abs();
            let factor_error = (original_factor as f32 - decoded_factor).abs();
            assert!(
                factor_error < 0.02, // <2% error on factor
                "error={error}, encoded={encoded}, decoded={decoded}, factor_error={factor_error}"
            );
        }
    }

    #[test]
    fn t7_i8_sign_preserved() {
        // Sign must be exact for si_modify_coefficient direction
        assert!(decode_error(encode_error(0.3)) > 0.0);
        assert!(decode_error(encode_error(-0.3)) < 0.0);
        assert_eq!(decode_error(encode_error(0.0)), 0.0);
    }

    // --- T8: strip-based luma matches full computation ---

    #[test]
    fn t8_strip_luma_matches_full() {
        use crate::codec::jpeg::pixels::rgb_to_luma_blocks;

        // Create a small test image (24x16 = 3x2 blocks)
        let width = 24u32;
        let height = 16u32;
        let mut rgb = vec![0u8; (width * height * 3) as usize];
        for i in 0..rgb.len() {
            rgb[i] = ((i * 37 + 13) % 256) as u8;
        }

        let full_blocks = rgb_to_luma_blocks(&rgb, width, height);
        let luma_bw = (width as usize).div_ceil(8); // 3

        // Get strip for all rows at once
        let strip_all = rgb_to_luma_blocks_strip(&rgb, width, height, 0, 2);
        assert_eq!(strip_all.len(), full_blocks.len());
        for (i, (a, b)) in full_blocks.iter().zip(strip_all.iter()).enumerate() {
            for k in 0..64 {
                assert!(
                    (a[k] - b[k]).abs() < 1e-10,
                    "block {i}, coeff {k}: full={}, strip={}",
                    a[k], b[k]
                );
            }
        }

        // Get strips one row at a time
        let strip0 = rgb_to_luma_blocks_strip(&rgb, width, height, 0, 1);
        let strip1 = rgb_to_luma_blocks_strip(&rgb, width, height, 1, 2);
        assert_eq!(strip0.len(), luma_bw);
        assert_eq!(strip1.len(), luma_bw);
        for bc in 0..luma_bw {
            for k in 0..64 {
                assert!(
                    (full_blocks[bc][k] - strip0[bc][k]).abs() < 1e-10,
                    "row 0, block {bc}, coeff {k}"
                );
                assert!(
                    (full_blocks[luma_bw + bc][k] - strip1[bc][k]).abs() < 1e-10,
                    "row 1, block {bc}, coeff {k}"
                );
            }
        }
    }
}
