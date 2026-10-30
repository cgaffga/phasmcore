// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Ghost mode capacity and stealth bands.
//!
//! The capacity policy (`docs/design/image/ghost-embedding-v3.md`, "Capacity
//! policy") is stated in **frame bits per non-zero Y AC coefficient**
//! (bpnzAC), the unit the detectability measurements use:
//!
//! | band | frame bits | exact SI (Deep Cover) | meaning |
//! |---|---|---|---|
//! | comfort | ≤ max / 4, and ≤ 2240 · √(nzAC / 44 000) | ≤ max / 4, and ≤ 2800 · √(nzAC / 44 000) | near chance for every detector measured |
//! | caution | ≤ max / 2 | ≤ max / 2 | |
//! | heavy | ≤ max = nzAC / 5 (0.20 bpnzAC) | ≤ max = nzAC / 4 (0.25 bpnzAC) | detectable by a matched CNN at the max |
//!
//! The square-root cap on the comfort band keeps its *absolute* size growing
//! only with √cover on large photos (the square-root law: a fixed rate gets
//! more detectable as the cover grows; 280 B of frame at BOSSbase size,
//! ≈ 2.6 KB at 12 MP). It is max / 4 at the BOSSbase reference size, so it
//! scales with the maximum.
//!
//! **Deep Cover bonus**: when the embed uses *exact* side information — the
//! cover is what phasm's cover writer makes of the raw pixels, checked
//! coefficient by coefficient ([`SideInfo::is_exact_for`], the same check
//! `encode_si*` runs) — the maximum is nzAC / 4 and every band scales with
//! it (SRNet QF92: exact SI at nzAC / 4 P_E 0.450 vs plain v3 at nzAC / 5
//! 0.366; `docs/design/image/deep-cover-exact-si.md`). Approximate side
//! information (any other cover) and plain JPEG input keep nzAC / 5. Core
//! decides from the pixels; no caller flag can claim the bonus.
//!
//! Everything is reported in **plaintext bytes** of the (compressed)
//! payload: frame bytes minus the frame overhead. With shadows, the shadows'
//! RS-coded frames are taken off every band (the bands bound the total), and
//! a single shadow may hold half the maximum — the exact-SI maximum on a
//! Deep Cover cover.
//!
//! These are the policy the apps enforce; the core encoder itself enforces
//! only technical feasibility (frame ≤ one bit per Y AC coefficient).

use crate::codec::jpeg::JpegImage;
use crate::codec::jpeg::dct::DctGrid;
use crate::stego::error::StegoError;
use crate::stego::frame::{FRAME_OVERHEAD, FRAME_OVERHEAD_EXT};
use crate::stego::shadow;
use super::side_info::SideInfo;

/// Maximum frame bits = nzAC / `MAX_RATE_DIV` (0.20 bpnzAC).
pub const MAX_RATE_DIV: usize = 5;

/// Maximum frame bits with exact side information (Deep Cover):
/// nzAC / `MAX_RATE_DIV_EXACT_SI` (0.25 bpnzAC).
pub const MAX_RATE_DIV_EXACT_SI: usize = 4;

/// Square-root-law cap on the comfort band: frame bits
/// `COMFORT_SQRT_BITS · √(nzAC / COMFORT_SQRT_NZAC)` (0.05 bpnzAC at the
/// BOSSbase mean nzAC), scaled by `MAX_RATE_DIV / rate_div` with the maximum.
const COMFORT_SQRT_BITS: f64 = 2240.0;
const COMFORT_SQRT_NZAC: f64 = 44_000.0;

/// Stealth bands of a cover, in plaintext bytes of the primary payload.
/// Always `comfort ≤ caution ≤ max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GhostBands {
    /// The capacity bar's 100 %.
    pub max_bytes: usize,
    /// Green up to here.
    pub comfort_bytes: usize,
    /// Medium/Low stealth rating boundary.
    pub caution_bytes: usize,
}

/// Count non-zero AC coefficients of the Y grid.
pub fn count_nonzero_ac(grid: &DctGrid) -> usize {
    grid.coeffs()
        .chunks_exact(64)
        .map(|b| b[1..].iter().filter(|&&c| c != 0).count())
        .sum()
}

/// Maximum-rate divisor for a cover: [`MAX_RATE_DIV_EXACT_SI`] when the
/// embed has exact side information, else [`MAX_RATE_DIV`].
pub fn rate_div(exact_si: bool) -> usize {
    if exact_si { MAX_RATE_DIV_EXACT_SI } else { MAX_RATE_DIV }
}

/// Band thresholds in frame bits for a cover with `nzac` non-zero AC
/// coefficients and maximum nzAC / `rate_div`: `(max, comfort, caution)`.
pub fn band_frame_bits(nzac: usize, rate_div: usize) -> (usize, usize, usize) {
    let max = nzac / rate_div;
    let sqrt_bits = COMFORT_SQRT_BITS * MAX_RATE_DIV as f64 / rate_div as f64;
    let sqrt_cap = (sqrt_bits * (nzac as f64 / COMFORT_SQRT_NZAC).sqrt()) as usize;
    let comfort = (max / 4).min(sqrt_cap);
    let caution = (max / 2).max(comfort);
    (max, comfort, caution)
}

/// Plaintext bytes a frame of `bits` carries.
pub fn plaintext_for_frame_bits(bits: usize) -> usize {
    let frame_bytes = bits / 8;
    if frame_bytes <= FRAME_OVERHEAD {
        return 0;
    }
    let capacity = frame_bytes - FRAME_OVERHEAD;
    if capacity > u16::MAX as usize {
        // Extended frame: 4 extra bytes for the 32-bit length.
        frame_bytes.saturating_sub(FRAME_OVERHEAD_EXT)
    } else {
        capacity
    }
}

/// RS-coded bits of `count` shadow frames carrying `total_bytes` of
/// plaintext together (upper bound: every shadow pays its own partial RS
/// block).
pub fn shadow_frames_bits(count: usize, total_bytes: usize) -> usize {
    if count == 0 {
        return 0;
    }
    let data = total_bytes + count * crate::stego::shadow_layer::SHADOW_FRAME_OVERHEAD_V1;
    let k = 255 - shadow::SHADOW_STC_PARITY;
    let blocks = data.div_ceil(k) + count - 1;
    (data + blocks * shadow::SHADOW_STC_PARITY) * 8
}

fn y_nzac(img: &JpegImage) -> Result<usize, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    Ok(count_nonzero_ac(img.dct_grid(0)))
}

/// Whether `encode_si*` with these pixels embeds into `img` with exact side
/// information. Errors exactly where `encode_si*` would on the pixel buffer
/// (empty or shorter than `width · height · 3`); pixels that belong to
/// another cover are simply not exact.
pub fn has_exact_side_info(img: &JpegImage, raw_rgb: &[u8], width: u32, height: u32) -> Result<bool, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let expected = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(3)).unwrap_or(0);
    if width == 0 || height == 0 || raw_rgb.len() < expected {
        return Err(StegoError::ImageTooSmall);
    }
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    Ok(SideInfo::is_exact_for(raw_rgb, width, height, img.dct_grid(0), &qt.values))
}

/// Bands of a cover with `nzac` non-zero AC coefficients at maximum
/// nzAC / `rate_div`, next to the given shadows.
fn bands_for(nzac: usize, rate_div: usize, shadow_count: usize, shadow_total_bytes: usize) -> GhostBands {
    let (max, comfort, caution) = band_frame_bits(nzac, rate_div);
    let sb = shadow_frames_bits(shadow_count, shadow_total_bytes);
    let p = |bits: usize| plaintext_for_frame_bits(bits.saturating_sub(sb));
    let max_bytes = p(max);
    let caution_bytes = p(caution).min(max_bytes);
    let comfort_bytes = p(comfort).min(caution_bytes);
    GhostBands { max_bytes, comfort_bytes, caution_bytes }
}

/// Stealth bands of the primary payload for a cover carrying `shadow_count`
/// shadows of `shadow_total_bytes` plaintext together (0, 0 = no shadows),
/// embedded without side information (JPEG input).
pub fn ghost_bands(img: &JpegImage, shadow_count: usize, shadow_total_bytes: usize) -> Result<GhostBands, StegoError> {
    Ok(bands_for(y_nzac(img)?, MAX_RATE_DIV, shadow_count, shadow_total_bytes))
}

/// [`ghost_bands`] for an `encode_si*` embed with these raw pixels: the
/// Deep Cover maximum (nzAC / 4) when the side information is exact
/// ([`has_exact_side_info`]), else the same as [`ghost_bands`].
pub fn ghost_bands_si(
    img: &JpegImage,
    raw_rgb: &[u8],
    width: u32,
    height: u32,
    shadow_count: usize,
    shadow_total_bytes: usize,
) -> Result<GhostBands, StegoError> {
    let div = rate_div(has_exact_side_info(img, raw_rgb, width, height)?);
    Ok(bands_for(y_nzac(img)?, div, shadow_count, shadow_total_bytes))
}

/// Ghost capacity (the bands' maximum), plaintext bytes.
///
/// # Errors
/// [`StegoError::NoLuminanceChannel`] if the image has no Y component.
pub fn estimate_capacity(img: &JpegImage) -> Result<usize, StegoError> {
    Ok(ghost_bands(img, 0, 0)?.max_bytes)
}

/// Ghost capacity of an `encode_si*` embed with these raw pixels ("Deep
/// Cover"): nzAC / 4 frame bits with exact side information, else the same
/// as [`estimate_capacity`].
pub fn estimate_capacity_si(img: &JpegImage, raw_rgb: &[u8], width: u32, height: u32) -> Result<usize, StegoError> {
    Ok(ghost_bands_si(img, raw_rgb, width, height, 0, 0)?.max_bytes)
}

/// Largest single shadow (plaintext bytes) at maximum nzAC / `rate_div`: a
/// shadow frame of half the maximum, so the primary — the largest message,
/// by the auto-sort — still fits next to it.
pub(crate) fn shadow_capacity_for(nzac: usize, rate_div: usize) -> usize {
    shadow::shadow_plaintext_for_bits(band_frame_bits(nzac, rate_div).0 / 2)
}

/// Largest single shadow message (plaintext bytes), JPEG input.
pub fn estimate_shadow_capacity(img: &JpegImage) -> Result<usize, StegoError> {
    Ok(shadow_capacity_for(y_nzac(img)?, MAX_RATE_DIV))
}

/// [`estimate_shadow_capacity`] for an `encode_si*` embed with these raw
/// pixels: half the Deep Cover maximum with exact side information.
pub fn estimate_shadow_capacity_si(img: &JpegImage, raw_rgb: &[u8], width: u32, height: u32) -> Result<usize, StegoError> {
    let div = rate_div(has_exact_side_info(img, raw_rgb, width, height)?);
    Ok(shadow_capacity_for(y_nzac(img)?, div))
}

/// Primary capacity (plaintext bytes) next to `shadow_count` shadows of
/// `shadow_total_bytes` together, without side information. `is_si` does not
/// change the result: the Deep Cover bonus needs the pixels
/// ([`ghost_bands_si`]); a flag cannot claim it.
pub fn estimate_capacity_with_shadows(
    img: &JpegImage,
    shadow_count: usize,
    shadow_total_bytes: usize,
    _is_si: bool,
) -> Result<usize, StegoError> {
    Ok(ghost_bands(img, shadow_count, shadow_total_bytes)?.max_bytes)
}

/// Deep Cover bands of a cover phasm's writer just made (exact side
/// information by construction).
pub(crate) fn ghost_bands_exact(img: &JpegImage, shadow_count: usize, shadow_total_bytes: usize) -> Result<GhostBands, StegoError> {
    Ok(bands_for(y_nzac(img)?, MAX_RATE_DIV_EXACT_SI, shadow_count, shadow_total_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(name: &str) -> JpegImage {
        JpegImage::from_bytes(&std::fs::read(format!("test-vectors/image/{name}")).unwrap()).unwrap()
    }

    #[test]
    fn capacity_reasonable_for_photo() {
        let img = load("photo_320x240_q75_420.jpg");
        let cap = estimate_capacity(&img).unwrap();
        assert!(cap > 100, "capacity {cap} is too low for 320x240");
        assert!(cap < 5000, "capacity {cap} is suspiciously high");
        // The maximum is nzAC / 5 frame bits.
        let nz = count_nonzero_ac(img.dct_grid(0));
        assert_eq!(cap, nz / 5 / 8 - FRAME_OVERHEAD);
    }

    #[test]
    fn band_fractions_small_cover() {
        // BOSSbase-sized: the square-root cap does not bind.
        let (max, comfort, caution) = band_frame_bits(40_000, MAX_RATE_DIV);
        assert_eq!((max, comfort, caution), (8000, 2000, 4000));
        // Exact SI: nzAC / 4, every band scaled with it.
        assert_eq!(band_frame_bits(40_000, MAX_RATE_DIV_EXACT_SI), (10_000, 2500, 5000));
        assert_eq!(rate_div(false), MAX_RATE_DIV);
        assert_eq!(rate_div(true), MAX_RATE_DIV_EXACT_SI);
    }

    #[test]
    fn square_root_cap_on_large_covers() {
        // 12 MP photo (nzAC ≈ 3.83 M): comfort ≈ 2240 · √87 ≈ 20.9 kbit.
        let (max, comfort, caution) = band_frame_bits(3_833_089, MAX_RATE_DIV);
        assert_eq!(max, 766_617);
        assert_eq!(caution, max / 2);
        assert!((20_800..=21_000).contains(&comfort), "{comfort}");
        // Exact SI: the cap scales with the maximum (×5/4), so the bar keeps
        // its proportions.
        let (xmax, xcomfort, xcaution) = band_frame_bits(3_833_089, MAX_RATE_DIV_EXACT_SI);
        assert_eq!(xmax, 958_272);
        assert_eq!(xcaution, xmax / 2);
        assert!(xcomfort.abs_diff(comfort * 5 / 4) <= 1, "{xcomfort} vs {comfort}");
        // Crossover near nzAC ≈ 45.6 k, for both maxima.
        for div in [MAX_RATE_DIV, MAX_RATE_DIV_EXACT_SI] {
            let (m, c, _) = band_frame_bits(46_000, div);
            assert!(c < m / 4);
            let (m, c, _) = band_frame_bits(45_000, div);
            assert_eq!(c, m / 4);
        }
    }

    #[test]
    fn bands_are_ordered_and_shadows_reduce_them() {
        let img = load("photo_640x480_q75_420.jpg");
        let b = ghost_bands(&img, 0, 0).unwrap();
        assert!(b.comfort_bytes <= b.caution_bytes && b.caution_bytes <= b.max_bytes);
        assert!(b.comfort_bytes > 0);
        let s = ghost_bands(&img, 2, 100).unwrap();
        assert!(s.max_bytes < b.max_bytes && s.comfort_bytes < b.comfort_bytes);
        assert_eq!(estimate_capacity_with_shadows(&img, 2, 100, true).unwrap(), s.max_bytes);
        // A tiny cover: everything zero, still ordered.
        let t = ghost_bands(&load("tiny_8x8_q95.jpg"), 0, 0).unwrap();
        assert_eq!(t, GhostBands { max_bytes: 0, comfort_bytes: 0, caution_bytes: 0 });
    }

    #[test]
    fn shadow_bits_bound_the_real_frames() {
        for (count, total) in [(1usize, 10usize), (2, 300), (3, 1000), (1, 60_000)] {
            let per = total / count;
            let real: usize = (0..count).map(|_| shadow::shadow_bits_for_plaintext(per)).sum();
            assert!(shadow_frames_bits(count, total) >= real, "{count} {total}");
        }
    }
}
