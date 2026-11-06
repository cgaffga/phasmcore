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
//! | caution | ≤ min(max / 2, 2 · comfort) | ≤ min(max / 2, 2 · comfort) | |
//! | heavy | ≤ max = nzAC / 5 (0.20 bpnzAC) | ≤ max = nzAC / 4 (0.25 bpnzAC) | detectable by a matched CNN at the max |
//!
//! The square-root cap on the comfort band keeps its *absolute* size growing
//! only with √cover on large photos (the square-root law: a fixed rate gets
//! more detectable as the cover grows; 280 B of frame at BOSSbase size,
//! ≈ 2.6 KB at 12 MP). It is max / 4 at the BOSSbase reference size, so it
//! scales with the maximum.
//!
//! **Caution cap (R3.8)**: on large photos max / 2 is far above the comfort
//! band and is detectable (tile-pooled SRNet, 190 Wikimedia photos ≥ 8 MP (median 9.8 MP),
//! photo-level P_E, 0.5 = chance: comfort 0.48, caution at max / 2 0.35-0.37,
//! max 0.21-0.26; `docs/notes/2026-10-30-app-fix-list.md` R3.8). The caution
//! band is therefore capped at 2 · comfort (P_E 0.47, chance). Where the
//! square-root cap does not bind, comfort = max / 4 and 2 · comfort = max / 2,
//! so small photos keep caution = max / 2.
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
//! **Decoded-JPEG cut**: pixels that are a JPEG decoded on the cover's grid
//! ([`super::side_info::is_decoded_jpeg_on_grid`]) make an aligned
//! double-compression cover. The encode drops the side information, and the
//! maximum is nzAC / 10 (0.10 bpnzAC): the comb of such a cover keeps plain
//! ±1 changes detectable at higher rates (histogram attack P_E 0.460 at 0.10,
//! 0.298 at 0.20; plain v3 on a single-compressed JPEG ≈ 0.47;
//! `eval/runs/2026-10-31-dc-guard/RESULTS.md`).
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

/// Maximum frame bits when the pixels are a JPEG decoded on the cover's grid:
/// nzAC / `MAX_RATE_DIV_DECODED_JPEG` (0.10 bpnzAC).
pub const MAX_RATE_DIV_DECODED_JPEG: usize = 10;

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
    // R3.8: at most twice the comfort band (equal to max / 2 on small covers).
    let caution = (max / 2).min(comfort * 2).max(comfort);
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

/// How an `encode_si*` embed with these pixels uses side information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiKind {
    /// The cover is the writer's output for the pixels: exact side
    /// information, maximum nzAC / 4.
    Exact,
    /// Any other cover: approximate side information, maximum nzAC / 5.
    Approximate,
    /// The pixels are a JPEG decoded on this grid: no side information,
    /// maximum nzAC / 10.
    DecodedJpeg,
}

impl SiKind {
    /// The bands' maximum is nzAC / `rate_div()`.
    pub fn rate_div(self) -> usize {
        match self {
            SiKind::Exact => MAX_RATE_DIV_EXACT_SI,
            SiKind::Approximate => MAX_RATE_DIV,
            SiKind::DecodedJpeg => MAX_RATE_DIV_DECODED_JPEG,
        }
    }
}

/// Classifies an `encode_si*` embed with these pixels into `img`. Errors
/// exactly where `encode_si*` would on the pixel buffer (empty or shorter
/// than `width · height · 3`); pixels that belong to another cover are
/// simply not exact.
pub fn si_kind(img: &JpegImage, raw_rgb: &[u8], width: u32, height: u32) -> Result<SiKind, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let expected = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(3)).unwrap_or(0);
    if width == 0 || height == 0 || raw_rgb.len() < expected {
        return Err(StegoError::ImageTooSmall);
    }
    // A decoded JPEG re-written on its own grid is exact for the writer but
    // no precover: the encode drops the side information (`SideInfo::for_embed`).
    if super::side_info::is_decoded_jpeg_on_grid(raw_rgb, width, height) {
        return Ok(SiKind::DecodedJpeg);
    }
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    Ok(if SideInfo::is_exact_for(raw_rgb, width, height, img.dct_grid(0), &qt.values) {
        SiKind::Exact
    } else {
        SiKind::Approximate
    })
}

/// Whether `encode_si*` with these pixels embeds into `img` with exact side
/// information (the cover is the writer's output for them, and they are not a
/// JPEG decoded on this grid). Errors as [`si_kind`].
pub fn has_exact_side_info(img: &JpegImage, raw_rgb: &[u8], width: u32, height: u32) -> Result<bool, StegoError> {
    Ok(si_kind(img, raw_rgb, width, height)? == SiKind::Exact)
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
/// Deep Cover maximum (nzAC / 4) when the side information is exact, nzAC / 10
/// when the pixels are a JPEG decoded on this grid, else the same as
/// [`ghost_bands`] ([`si_kind`]).
pub fn ghost_bands_si(
    img: &JpegImage,
    raw_rgb: &[u8],
    width: u32,
    height: u32,
    shadow_count: usize,
    shadow_total_bytes: usize,
) -> Result<GhostBands, StegoError> {
    let div = si_kind(img, raw_rgb, width, height)?.rate_div();
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
/// Cover"): nzAC / 4 frame bits with exact side information, nzAC / 10 for a
/// JPEG decoded on this grid, else the same
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
/// pixels: half the maximum of [`ghost_bands_si`].
pub fn estimate_shadow_capacity_si(img: &JpegImage, raw_rgb: &[u8], width: u32, height: u32) -> Result<usize, StegoError> {
    let div = si_kind(img, raw_rgb, width, height)?.rate_div();
    Ok(shadow_capacity_for(y_nzac(img)?, div))
}

/// The Ghost capacity bar's two numbers in one call (A#31): the stealth
/// bands next to `shadow_count` shadows of `shadow_total_bytes` together, and
/// the largest single shadow message, as [`ghost_bands_si`] and
/// [`estimate_shadow_capacity_si`] would return them for the same pixels
/// (`si` = `(raw_rgb, width, height)`; `None` = JPEG input, the plain
/// [`ghost_bands`] / [`estimate_shadow_capacity`]). One JPEG parse, one
/// side-information classification, and the apps cross the bridge once with
/// the pixel buffer instead of twice.
pub fn ghost_bands_and_shadow_capacity(
    img: &JpegImage,
    si: Option<(&[u8], u32, u32)>,
    shadow_count: usize,
    shadow_total_bytes: usize,
) -> Result<(GhostBands, usize), StegoError> {
    let div = match si {
        Some((raw_rgb, width, height)) => si_kind(img, raw_rgb, width, height)?.rate_div(),
        None => MAX_RATE_DIV,
    };
    let nzac = y_nzac(img)?;
    Ok((bands_for(nzac, div, shadow_count, shadow_total_bytes), shadow_capacity_for(nzac, div)))
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

/// Bands of a cover phasm's writer just made from pixels already classified
/// (the writer's cover is exact by construction unless the pixels are a JPEG
/// decoded on its grid).
pub(crate) fn ghost_bands_for_kind(img: &JpegImage, kind: SiKind, shadow_count: usize, shadow_total_bytes: usize) -> Result<GhostBands, StegoError> {
    Ok(bands_for(y_nzac(img)?, kind.rate_div(), shadow_count, shadow_total_bytes))
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
        // R3.8: caution is 2 x comfort, far below max / 2 here.
        assert_eq!(caution, comfort * 2);
        assert!(caution < max / 2);
        assert!((20_800..=21_000).contains(&comfort), "{comfort}");
        // Exact SI: the cap scales with the maximum (×5/4), so the bar keeps
        // its proportions.
        let (xmax, xcomfort, xcaution) = band_frame_bits(3_833_089, MAX_RATE_DIV_EXACT_SI);
        assert_eq!(xmax, 958_272);
        assert_eq!(xcaution, xcomfort * 2);
        assert!(xcaution < xmax / 2);
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
    fn caution_is_capped_at_twice_comfort() {
        // 512x512-like (nzAC ~ 40 k): the cap does not bind, caution = max / 2.
        for div in [MAX_RATE_DIV, MAX_RATE_DIV_EXACT_SI] {
            let (max, comfort, caution) = band_frame_bits(40_000, div);
            assert_eq!(caution, max / 2);
            assert_eq!(caution, comfort * 2);
        }
        // 3840x2560-like (nzAC ~ 3.8 M): caution = 2 x comfort < max / 2.
        for div in [MAX_RATE_DIV, MAX_RATE_DIV_EXACT_SI] {
            let (max, comfort, caution) = band_frame_bits(3_833_089, div);
            assert_eq!(caution, comfort * 2);
            assert!(caution < max / 2);
        }
        // Invariant comfort <= caution <= max over a wide nzAC sweep, for
        // every rate divisor, and caution == min(max / 2, 2 x comfort) up to
        // integer rounding.
        for div in [MAX_RATE_DIV, MAX_RATE_DIV_EXACT_SI, MAX_RATE_DIV_DECODED_JPEG] {
            for nzac in [0usize, 1, 7, 100, 5_000, 40_000, 45_000, 46_000, 300_000, 1_000_000, 3_833_089, 10_000_000] {
                let (max, comfort, caution) = band_frame_bits(nzac, div);
                assert!(comfort <= caution && caution <= max, "{nzac} {div}");
                assert_eq!(caution, (max / 2).min(comfort * 2).max(comfort), "{nzac} {div}");
            }
        }
    }

    #[test]
    fn bands_are_ordered_and_shadows_reduce_them() {
        let img = load("photo_640x480_q75_420.jpg");
        let b = ghost_bands(&img, 0, 0).unwrap();
        assert!(b.comfort_bytes <= b.caution_bytes && b.caution_bytes <= b.max_bytes);
        assert!(b.comfort_bytes > 0);
        let s = ghost_bands(&img, 2, 100).unwrap();
        assert!(s.comfort_bytes <= s.caution_bytes && s.caution_bytes <= s.max_bytes);
        assert!(s.max_bytes < b.max_bytes && s.comfort_bytes < b.comfort_bytes);
        assert_eq!(estimate_capacity_with_shadows(&img, 2, 100, true).unwrap(), s.max_bytes);
        // A tiny cover: everything zero, still ordered.
        let t = ghost_bands(&load("tiny_8x8_q95.jpg"), 0, 0).unwrap();
        assert_eq!(t, GhostBands { max_bytes: 0, comfort_bytes: 0, caution_bytes: 0 });
    }

    #[test]
    fn bands_and_shadow_capacity_in_one_call_match_the_separate_calls() {
        use crate::stego::ghost::deep_cover::deep_cover_jpeg;
        // JPEG input: the plain calls.
        let img = load("photo_640x480_q75_420.jpg");
        let (b, s) = ghost_bands_and_shadow_capacity(&img, None, 2, 100).unwrap();
        assert_eq!(b, ghost_bands(&img, 2, 100).unwrap());
        assert_eq!(s, estimate_shadow_capacity(&img).unwrap());
        // Deep Cover: exact pixels, then pixels that are not this cover's,
        // then a malformed buffer (the same error as the separate calls).
        let (w, h) = (320u32, 240u32);
        let mut st = 7u32;
        let px: Vec<u8> = (0..(w * h * 3) as usize)
            .map(|i| {
                st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                (((i % 251) as u32 / 2 + (st >> 25)) as u8).saturating_add(30)
            })
            .collect();
        let cover = JpegImage::from_bytes(&deep_cover_jpeg(&px, w, h, 92).unwrap()).unwrap();
        // Pixels of another size are no precover of this cover: plain figures.
        let (ow, oh) = (w + 8, h);
        let other: Vec<u8> = (0..(ow * oh * 3) as usize).map(|i| (i % 253) as u8).collect();
        for (raw, rw, rh) in [(&px, w, h), (&other, ow, oh)] {
            for (count, total) in [(0, 0), (1, 200), (3, 900)] {
                let (b, s) = ghost_bands_and_shadow_capacity(&cover, Some((raw, rw, rh)), count, total).unwrap();
                assert_eq!(b, ghost_bands_si(&cover, raw, rw, rh, count, total).unwrap());
                assert_eq!(s, estimate_shadow_capacity_si(&cover, raw, rw, rh).unwrap());
            }
        }
        let exact = ghost_bands_and_shadow_capacity(&cover, Some((&px, w, h)), 0, 0).unwrap();
        let approx = ghost_bands_and_shadow_capacity(&cover, Some((&other, ow, oh)), 0, 0).unwrap();
        assert!(exact.0.max_bytes > approx.0.max_bytes && exact.1 > approx.1, "exact SI is the larger cover: {exact:?} vs {approx:?}");
        assert!(matches!(
            ghost_bands_and_shadow_capacity(&cover, Some((&px[..px.len() - 3], w, h)), 0, 0),
            Err(StegoError::ImageTooSmall)
        ));
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
