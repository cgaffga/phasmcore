// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! The cover every frontend encodes: one rule table for iOS, Android, the web
//! and the CLI (`docs/design/app/encode-cover-pipeline.md`, decided
//! 2026-11-02). The platform keeps the picked file's bytes (`original`) and,
//! when [`cover_needs_pixels`] says so, decodes it to upright 8-bit sRGB
//! pixels; [`build_cover`] does everything else: pass-through, lossless
//! clean, resample, crop, the cover writer, metadata, and which pixels the
//! side-informed encode gets.
//!
//! | Mode | Source / settings | Cover | SI pixels |
//! |---|---|---|---|
//! | Ghost | JPEG source, original size, default quality, wand off | the original bytes | none |
//! | Ghost | … wand on | lossless upright + metadata strip | none |
//! | Ghost | JPEG source made smaller, or any non-JPEG source | writer from the (resampled) pixels | the pixels |
//! | Ghost | same size, AVIF source or pixels the decoded-JPEG guard flags (the quality chip on a JPEG, a decoded JPEG as PNG) | writer from the realigned pixels: 4 px cropped top and left, resampled back to the same size | the realigned pixels |
//! | Armor | any | `armor_cover_dimensions` → resample → crop → writer at QF65 | none |
//!
//! With the wand off, a written cover carries the source's metadata, cleaned
//! ([`crate::codec::jpeg::exif`]); with it on, none. Every written Ghost
//! cover returns its pixels, and core decides from them what the `*_si`
//! calls do ([`super::ghost::capacity::si_kind`]): exact side information
//! and nzAC/4 for a precover. A JPEG or AVIF decoded and re-written on its
//! own grid would be an aligned double compression, which the decoded-JPEG
//! guard ([`crate::codec::jpeg::history`]) answers with no side information
//! and nzAC/10 (fix list R3.7); since 2026-11-21 such a cover is realigned
//! instead ([`realign`]) and gets exact side information like any precover.
//! The guard stays in the `*_si` calls for pixels that still show a lattice
//! (rare: a decode at QF ≤ 30, mostly flat blocks).

use crate::codec::jpeg::error::JpegError;
use crate::codec::jpeg::exif::cover_with_source_metadata;
use crate::codec::jpeg::lossless::upright_and_clean;
use crate::codec::resample::{fit_long_side, resample_rgb, resample_rgb_window};
use crate::stego::armor::cover_dims::{armor_cover_dimensions, ArmorCoverPreset};
use crate::stego::error::StegoError;
use crate::stego::ghost::deep_cover::{deep_cover_jpeg, DEEP_COVER_DEFAULT_QUALITY};
use crate::stego::ghost::side_info::is_decoded_jpeg_on_grid;
use crate::stego::{MAX_DIMENSION, MAX_PIXELS};

/// IJG quality of every Armor cover (measured: other qualities lose
/// survival, `eval/runs/2026-10-31-armor-v2/RESULTS.md`).
pub const ARMOR_COVER_QUALITY: u8 = 65;

/// Which embedding the cover is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverMode {
    Ghost,
    Armor,
}

/// The user's settings that shape the cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoverRequest {
    pub mode: CoverMode,
    /// The magic wand: metadata stripped (and a generated name, app side).
    pub wand: bool,
    /// Ghost quality chip; `None` = default (pass-through for a JPEG source,
    /// IJG 92 for a written cover).
    pub ghost_quality: Option<u8>,
    /// Ghost resolution chip: long side at most this; `None` = original size.
    pub ghost_max_side: Option<u32>,
    /// Armor channel preset.
    pub armor_preset: ArmorCoverPreset,
}

/// The cover to encode.
#[derive(Debug, Clone)]
pub struct CoverOut {
    pub jpeg: Vec<u8>,
    /// Pixels the Ghost encode and capacity calls must get (`*_si`), when
    /// they are not the caller's own input pixels (a resample or the
    /// realignment of a same-size cover made them; same size or not, always
    /// these); see [`CoverOut::si_pixels_of`]. `None` with `si_is_input`
    /// false = the plain calls.
    pub si_pixels: Option<Vec<u8>>,
    /// The `*_si` pixels are exactly the pixels the caller passed to
    /// [`build_cover`] (a same-size Ghost cover that needed no
    /// realignment): `si_pixels` is then `None` and the caller reuses its
    /// own buffer, so core returns no copy (48 MP: 145 MB).
    pub si_is_input: bool,
    pub width: u32,
    pub height: u32,
    /// `jpeg` is the picked file, byte for byte.
    pub pass_through: bool,
    /// Armor's Instagram crop: the kept rectangle `[x, y, w, h]` in source
    /// pixels.
    pub crop: Option<[u32; 4]>,
    /// The `*_si` calls with `si_pixels` embed with exact side information
    /// (Deep Cover, nzAC/4): core's own classification, the same answer as
    /// [`super::ghost::capacity::si_kind`]. `false` without `si_pixels`
    /// and for pixels that are still a JPEG decoded on the cover's grid
    /// after the realignment (rare), whose `*_si` calls are plain Ghost at
    /// nzAC/10. What a "Deep Cover" label must show (not
    /// `si_pixels.is_some()`).
    pub deep_cover: bool,
}

impl CoverOut {
    /// The pixels for the `*_si` calls (`None` = the plain calls): the
    /// returned resample, or `input`, the pixels that were passed to
    /// [`build_cover`], when `si_is_input`.
    pub fn si_pixels_of<'a>(&'a self, input: Option<&'a [u8]>) -> Option<&'a [u8]> {
        if self.si_is_input { input } else { self.si_pixels.as_deref() }
    }
}

/// Width and height of a JPEG core reads as it is (8-bit SOF0 / SOF2), from
/// its headers only.
fn jpeg_source_dims(b: &[u8]) -> Option<(u32, u32)> {
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 {
        return None;
    }
    let mut p = 2;
    while p + 4 <= b.len() {
        if b[p] != 0xFF {
            p += 1;
            continue;
        }
        let m = b[p + 1];
        if m == 0xFF {
            p += 1;
            continue;
        }
        if m == 0x00 || m == 0x01 || (0xD0..=0xD9).contains(&m) {
            p += 2;
            continue;
        }
        let len = u16::from_be_bytes([b[p + 2], b[p + 3]]) as usize;
        if m == 0xC0 || m == 0xC2 {
            let s = b.get(p + 4..p + 9)?;
            if s[0] != 8 {
                return None;
            }
            let (h, w) = (u16::from_be_bytes([s[1], s[2]]) as u32, u16::from_be_bytes([s[3], s[4]]) as u32);
            return (w > 0 && h > 0).then_some((w, h));
        }
        if matches!(m, 0xC1 | 0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF | 0xDA) {
            return None;
        }
        p += 2 + len;
    }
    None
}

/// The size of `w` x `h` within the encode limits (16384 px a side, 200 MP).
fn within_limits(w: u32, h: u32) -> (u32, u32) {
    let side = (MAX_DIMENSION as f64 / w.max(h) as f64).min(1.0);
    let area = (MAX_PIXELS as f64 / (w as f64 * h as f64)).sqrt().min(1.0);
    let s = side.min(area);
    if s >= 1.0 {
        (w, h)
    } else {
        (((w as f64 * s).floor() as u32).max(1), ((h as f64 * s).floor() as u32).max(1))
    }
}

/// The Ghost cover size for a `w` x `h` source: the resolution chip and the
/// encode limits.
fn ghost_size(w: u32, h: u32, req: &CoverRequest) -> (u32, u32) {
    let (lw, lh) = within_limits(w, h);
    match req.ghost_max_side {
        Some(m) if m > 0 => fit_long_side(lw, lh, m),
        _ => (lw, lh),
    }
}

/// The Ghost cover is the JPEG source itself (pass-through or lossless clean).
fn ghost_keeps_jpeg(original: &[u8], req: &CoverRequest) -> bool {
    req.mode == CoverMode::Ghost
        && req.ghost_quality.is_none()
        && jpeg_source_dims(original).is_some_and(|(w, h)| ghost_size(w, h, req) == (w, h))
}

/// Whether [`build_cover`] needs the source's upright sRGB pixels for this
/// request (the platform decodes them only then).
pub fn cover_needs_pixels(original: &[u8], req: &CoverRequest) -> bool {
    !ghost_keeps_jpeg(original, req)
}

fn crop_rgb(rgb: &[u8], stride: u32, x: u32, y: u32, w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for row in y..y + h {
        let s = ((row * stride + x) * 3) as usize;
        out.extend_from_slice(&rgb[s..s + (w * 3) as usize]);
    }
    out
}

fn with_metadata(cover: Vec<u8>, original: &[u8], req: &CoverRequest) -> Result<Vec<u8>, StegoError> {
    if req.wand { Ok(cover) } else { Ok(cover_with_source_metadata(&cover, original)?) }
}

/// Pixels the realignment ([`realign`]) crops off the top and the left: half
/// a block, the farthest a grid can be from the cover's.
pub const REALIGN_CROP: u32 = 4;

/// Whether `original` is an AVIF file: an ISO-BMFF `ftyp` box first, with
/// `avif` (still image) or `avis` (image sequence) as its major or a
/// compatible brand.
pub fn is_avif_source(original: &[u8]) -> bool {
    let word = |i: usize| original.get(i..i + 4);
    let be32 = |i: usize| word(i).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as u64);
    if word(4) != Some(b"ftyp".as_slice()) {
        return false;
    }
    // Size 1: a 64-bit size follows the type; 0: the box runs to the end.
    let (head, size) = match be32(0) {
        Some(1) => (16, be32(8).zip(be32(12)).map_or(0, |(hi, lo)| hi << 32 | lo)),
        Some(0) => (8, original.len() as u64),
        Some(s) => (8, s),
        None => return false,
    };
    let end = size.min(original.len() as u64) as usize;
    let brand = |i: usize| i + 4 <= end && matches!(word(i), Some(b) if b == b"avif" || b == b"avis");
    // Major brand, minor version (skipped), compatible brands.
    brand(head) || (head + 8..end).step_by(4).any(brand)
}

/// A same-size Ghost cover is realigned when its source is AVIF or the
/// decoded-JPEG guard fires on its pixels (`aligned`). AV1 and JPEG keep
/// 8×8 transforms on the grid the cover writer uses, so their decodes are
/// an aligned double compression: the guard drops the side information and
/// caps the cover at nzAC/10 (histogram attack, AVIF 10-bit: P_E 0.190 at
/// 0.10 bpnzAC), and even an AVIF decode it does not flag hides part of its
/// nzAC on that grid. Realigned, the cover is a precover with exact side
/// information (`docs/design/image/deep-cover-exact-si.md`, "Realigned
/// covers").
fn needs_realign(original: &[u8], aligned: bool) -> bool {
    aligned || is_avif_source(original)
}

/// The realigned pixels of a same-size Ghost cover: [`REALIGN_CROP`] px
/// cropped off the top and the left and the rest resampled back to
/// `w × h` with [`resample_rgb`] (scale `w / (w − 4)`), so the cover keeps
/// the photo's size while the source's block grid moves off the cover's
/// (by 4 px at the top-left, falling to 0 at the bottom-right corner).
/// `None` when the image is too small to crop.
fn realign(rgb: &[u8], w: u32, h: u32) -> Result<Option<Vec<u8>>, StegoError> {
    let c = REALIGN_CROP;
    if w <= c || h <= c {
        return Ok(None);
    }
    Ok(Some(resample_rgb_window(rgb, w, h, [c, c, w - c, h - c], w, h)?))
}

/// The cover for `original` (the picked file's bytes) and `req`; `pixels` =
/// its upright 8-bit sRGB pixels `(rgb, width, height)`, required when
/// [`cover_needs_pixels`] is true.
///
/// # Errors
/// The cover writer's and the JPEG parser's errors; `InvalidJpeg` when the
/// pixels are needed but missing or malformed.
pub fn build_cover(original: &[u8], pixels: Option<(&[u8], u32, u32)>, req: &CoverRequest) -> Result<CoverOut, StegoError> {
    if ghost_keeps_jpeg(original, req) {
        let (w, h) = jpeg_source_dims(original).unwrap_or((0, 0));
        if !req.wand {
            return Ok(CoverOut {
                jpeg: original.to_vec(),
                si_pixels: None,
                si_is_input: false,
                width: w,
                height: h,
                pass_through: true,
                crop: None,
                deep_cover: false,
            });
        }
        let jpeg = upright_and_clean(original, true)?;
        let (cw, ch) = jpeg_source_dims(&jpeg).unwrap_or((w, h));
        return Ok(CoverOut {
            jpeg,
            si_pixels: None,
            si_is_input: false,
            width: cw,
            height: ch,
            pass_through: false,
            crop: None,
            deep_cover: false,
        });
    }
    let missing = || StegoError::InvalidJpeg(JpegError::InvalidMarkerData("the cover needs the source pixels"));
    let (rgb, w, h) = pixels.ok_or_else(missing)?;
    let need = (w as usize).checked_mul(h as usize).and_then(|n| n.checked_mul(3));
    if w == 0 || h == 0 || need != Some(rgb.len()) {
        return Err(missing());
    }
    match req.mode {
        CoverMode::Ghost => {
            let (tw, th) = ghost_size(w, h, req);
            // New pixels: a resample to the chip's size (or the encode
            // limits), or, at the same size, the realignment of a source
            // whose block grid is the cover's ([`needs_realign`]). Otherwise
            // the writer reads the caller's pixels (no copy).
            let (new_px, aligned) = if (tw, th) != (w, h) {
                (Some(resample_rgb(rgb, w, h, tw, th)?), None)
            } else {
                let aligned = is_decoded_jpeg_on_grid(rgb, w, h);
                let realigned = if needs_realign(original, aligned) { realign(rgb, w, h)? } else { None };
                (realigned, Some(aligned))
            };
            let px: &[u8] = new_px.as_deref().unwrap_or(rgb);
            let quality = req.ghost_quality.unwrap_or(DEEP_COVER_DEFAULT_QUALITY);
            let jpeg = with_metadata(deep_cover_jpeg(px, tw, th, quality)?, original, req)?;
            // The pixels go to the `*_si` calls in every case; core
            // classifies them (exact side information for a precover, none
            // and nzAC/10 for pixels that are still a JPEG decoded on the
            // cover's grid). A cover from the caller's own pixels returns no
            // copy: `si_is_input` says those are the ones (fix list R3.13
            // follow-up).
            let deep_cover = match (&new_px, aligned) {
                (None, Some(aligned)) => !aligned,
                _ => !is_decoded_jpeg_on_grid(px, tw, th),
            };
            let si_is_input = new_px.is_none();
            Ok(CoverOut { jpeg, si_pixels: new_px, si_is_input, width: tw, height: th, pass_through: false, crop: None, deep_cover })
        }
        CoverMode::Armor => {
            let d = armor_cover_dimensions(w, h, req.armor_preset);
            // No copy of the caller's pixels when no resample is needed.
            let resampled = if (d.scaled_width, d.scaled_height) == (w, h) {
                None
            } else {
                Some(resample_rgb(rgb, w, h, d.scaled_width, d.scaled_height)?)
            };
            let scaled: &[u8] = resampled.as_deref().unwrap_or(rgb);
            let cropped = (d.width, d.height) != (d.scaled_width, d.scaled_height);
            let cropped_px = cropped.then(|| crop_rgb(scaled, d.scaled_width, d.crop_x, d.crop_y, d.width, d.height));
            let px: &[u8] = cropped_px.as_deref().unwrap_or(scaled);
            let jpeg = with_metadata(deep_cover_jpeg(px, d.width, d.height, ARMOR_COVER_QUALITY)?, original, req)?;
            // The kept rectangle in source pixels.
            let crop = cropped.then(|| {
                let sx = |v: u32| ((v as u64 * w as u64) / d.scaled_width.max(1) as u64) as u32;
                let sy = |v: u32| ((v as u64 * h as u64) / d.scaled_height.max(1) as u64) as u32;
                [sx(d.crop_x), sy(d.crop_y), sx(d.width), sy(d.height)]
            });
            Ok(CoverOut { jpeg, si_pixels: None, si_is_input: false, width: d.width, height: d.height, pass_through: false, crop, deep_cover: false })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::jpeg::JpegImage;

    /// Photo-like test pixels: a gentle gradient under independent noise
    /// (an exactly periodic pattern repeats its blocks' DCT values and looks
    /// like a quantization lattice to the decoded-JPEG guard).
    fn textured(w: u32, h: u32) -> Vec<u8> {
        let mut s: u32 = 11;
        let mut n = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 25) as u32
        };
        (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                let g = (x * 60 / w + y * 60 / h) as u32;
                [(40 + g + n()) as u8, (30 + g / 2 + n()) as u8, (50 + n() / 2) as u8]
            })
            .collect()
    }

    fn req(mode: CoverMode) -> CoverRequest {
        CoverRequest { mode, wand: false, ghost_quality: None, ghost_max_side: None, armor_preset: ArmorCoverPreset::Standard }
    }

    fn dims(jpeg: &[u8]) -> (u32, u32) {
        jpeg_source_dims(jpeg).unwrap()
    }

    #[test]
    fn ghost_jpeg_source_passes_through_or_is_cleaned_losslessly() {
        let (w, h) = (640, 480);
        let px = textured(w, h);
        let src = deep_cover_jpeg(&px, w, h, 90).unwrap();
        let r = req(CoverMode::Ghost);
        assert!(!cover_needs_pixels(&src, &r));
        let out = build_cover(&src, None, &r).unwrap();
        assert!(out.pass_through && out.si_pixels.is_none() && !out.deep_cover);
        assert_eq!(out.jpeg, src);
        let wand = build_cover(&src, None, &CoverRequest { wand: true, ..r }).unwrap();
        assert!(!wand.pass_through && wand.si_pixels.is_none());
        let (a, b) = (JpegImage::from_bytes(&src).unwrap(), JpegImage::from_bytes(&wand.jpeg).unwrap());
        assert_eq!(a.dct_grid(0).coeffs(), b.dct_grid(0).coeffs(), "lossless: same coefficients");
    }

    /// A minimal ISO-BMFF head: an `ftyp` box with these brands (libavif
    /// writes major `avif`, compatible `avif mif1 miaf`), then an empty
    /// `meta` box header.
    fn avif_bytes(major: &[u8; 4], compatible: &[&[u8; 4]]) -> Vec<u8> {
        let mut b = ((16 + 4 * compatible.len()) as u32).to_be_bytes().to_vec();
        b.extend_from_slice(b"ftyp");
        b.extend_from_slice(major);
        b.extend_from_slice(&[0; 4]);
        compatible.iter().for_each(|c| b.extend_from_slice(*c));
        b.extend_from_slice(b"\0\0\0\x08meta");
        b
    }

    /// Asserts what a realigned same-size Ghost cover must be: the source's
    /// size, new pixels returned (not the caller's), no lattice on them, exact
    /// side information with the Deep Cover maximum nzAC/4 (1.25 × the plain
    /// nzAC/5), and an SI encode that round-trips. Returns the SI pixels.
    fn assert_realigned(out: &CoverOut, px: &[u8], w: u32, h: u32) -> Vec<u8> {
        use crate::stego::ghost::capacity::{count_nonzero_ac, plaintext_for_frame_bits, si_kind, SiKind, MAX_RATE_DIV, MAX_RATE_DIV_EXACT_SI};
        assert_eq!((out.width, out.height), (w, h), "same size");
        assert_eq!(dims(&out.jpeg), (w, h));
        assert!(!out.si_is_input, "new pixels, not the caller's");
        let si = out.si_pixels.as_deref().expect("the realigned pixels");
        assert_eq!(out.si_pixels_of(Some(px)), Some(si));
        assert_eq!(si.len(), (w * h * 3) as usize);
        assert_ne!(si, px);
        assert!(!crate::codec::jpeg::history::is_aligned_jpeg_decode(si, w, h), "{:?}", crate::codec::jpeg::history::grid_evidence(si, w, h));
        let img = JpegImage::from_bytes(&out.jpeg).unwrap();
        assert!(out.deep_cover);
        assert_eq!(si_kind(&img, si, w, h).unwrap(), SiKind::Exact);
        let nz = count_nonzero_ac(img.dct_grid(0));
        let (dc, plain) = (crate::stego::ghost_bands_si(&img, si, w, h, 0, 0).unwrap(), crate::stego::ghost_bands(&img, 0, 0).unwrap());
        assert_eq!(dc.max_bytes, plaintext_for_frame_bits(nz / MAX_RATE_DIV_EXACT_SI));
        assert_eq!(plain.max_bytes, plaintext_for_frame_bits(nz / MAX_RATE_DIV));
        let ratio = dc.max_bytes as f64 / plain.max_bytes as f64;
        assert!((1.24..=1.26).contains(&ratio), "Deep Cover / plain = {ratio:.3}");
        assert_eq!(crate::stego::ghost_capacity_si(&img, si, w, h).unwrap(), dc.max_bytes);
        let msg = "realigned";
        let (st, _) = crate::stego::ghost_encode_si_with_files_quality(&out.jpeg, si, w, h, msg, &[], "pw").unwrap();
        assert_eq!(crate::stego::ghost_decode(&st, "pw").unwrap().text, msg);
        si.to_vec()
    }

    #[test]
    fn avif_sources_are_recognised_by_their_brands() {
        assert!(is_avif_source(&avif_bytes(b"avif", &[b"avif", b"mif1", b"miaf"])));
        assert!(is_avif_source(&avif_bytes(b"avis", &[b"msf1", b"iso8"])), "image sequence");
        assert!(is_avif_source(&avif_bytes(b"mif1", &[b"miaf", b"avif"])), "compatible brand");
        assert!(!is_avif_source(&avif_bytes(b"heic", &[b"mif1", b"heic"])), "HEIC");
        assert!(!is_avif_source(&avif_bytes(b"mif1", &[b"miaf", b"MA1B"])));
        // "avif" as the minor version is not a brand.
        let mut v = avif_bytes(b"mif1", &[b"miaf"]);
        v[12..16].copy_from_slice(b"avif");
        assert!(!is_avif_source(&v));
        // A brand past the box's end does not count; size 0 runs to the end.
        let mut short = avif_bytes(b"mif1", &[b"miaf", b"avif"]);
        short[3] = 20;
        assert!(!is_avif_source(&short));
        short[3] = 0;
        assert!(is_avif_source(&short));
        // A 64-bit box size.
        let mut big = vec![0, 0, 0, 1];
        big.extend_from_slice(b"ftyp");
        big.extend_from_slice(&28u64.to_be_bytes());
        big.extend_from_slice(b"mif1\0\0\0\0avif");
        assert!(is_avif_source(&big));
        for b in [&b""[..], b"\0\0\0\x18ftyp", b"\xFF\xD8\xFF\xE0", b"\x89PNG\r\n\x1a\n"] {
            assert!(!is_avif_source(b));
        }
    }

    /// The decided fix (fix list "Aligned-JPEG guard fires on AVIF
    /// decodes", 2026-11-21): a same-size Ghost cover of an AVIF source is
    /// realigned, whether or not its decode shows a lattice.
    #[test]
    fn avif_sources_get_a_realigned_same_size_cover_with_exact_si() {
        let (w, h) = (640, 480);
        let avif = avif_bytes(b"avif", &[b"avif", b"mif1", b"miaf"]);
        // An aligned decode (AV1 keeps 8x8 transforms on the cover's grid;
        // a JPEG decode stands in for it here): the guard fires on the
        // input, not on the cover's pixels.
        let decoded = image::load_from_memory(&deep_cover_jpeg(&textured(w, h), w, h, 75).unwrap()).unwrap().to_rgb8().into_raw();
        assert!(crate::codec::jpeg::history::is_aligned_jpeg_decode(&decoded, w, h));
        let out = build_cover(&avif, Some((&decoded, w, h)), &req(CoverMode::Ghost)).unwrap();
        let si = assert_realigned(&out, &decoded, w, h);
        // Exactly: crop 4 px top and left, resample back with core's resampler.
        let crop = crop_rgb(&decoded, w, REALIGN_CROP, REALIGN_CROP, w - REALIGN_CROP, h - REALIGN_CROP);
        assert_eq!(si, resample_rgb(&crop, w - REALIGN_CROP, h - REALIGN_CROP, w, h).unwrap());
        assert_eq!(out.jpeg, deep_cover_jpeg(&si, w, h, DEEP_COVER_DEFAULT_QUALITY).unwrap());
        // An AVIF decode the guard does not flag is realigned too.
        let px = textured(w, h);
        let out = build_cover(&avif, Some((&px, w, h)), &req(CoverMode::Ghost)).unwrap();
        assert_realigned(&out, &px, w, h);
        // The quality chip and the wand change only the writer's quality and
        // the metadata.
        let q = build_cover(&avif, Some((&px, w, h)), &CoverRequest { ghost_quality: Some(85), wand: true, ..req(CoverMode::Ghost) }).unwrap();
        assert_eq!(q.si_pixels, out.si_pixels);
        assert_eq!(q.jpeg, deep_cover_jpeg(q.si_pixels.as_ref().unwrap(), w, h, 85).unwrap());
        // A resolution chip resamples once, from the source (no realignment).
        let r = build_cover(&avif, Some((&decoded, w, h)), &CoverRequest { ghost_max_side: Some(320), ..req(CoverMode::Ghost) }).unwrap();
        assert_eq!(r.si_pixels.unwrap(), resample_rgb(&decoded, w, h, 320, 240).unwrap());
        // Armor is unchanged.
        let a = build_cover(&avif, Some((&px, w, h)), &req(CoverMode::Armor)).unwrap();
        assert_eq!(a.jpeg, build_cover(b"\x89PNG\r\n\x1a\n", Some((&px, w, h)), &req(CoverMode::Armor)).unwrap().jpeg);
    }

    /// The quality chip on a JPEG at its original size re-writes a decoded
    /// JPEG on its own grid. Before 2026-11-21 the guard took the side
    /// information away (plain v3, nzAC/10, fix list R3.7); now the pixels
    /// are realigned and get exact side information.
    #[test]
    fn ghost_quality_chip_on_a_jpeg_is_realigned() {
        let (w, h) = (640, 480);
        let src = deep_cover_jpeg(&textured(w, h), w, h, 90).unwrap();
        // The platform's decode of the JPEG source.
        let px = image::load_from_memory(&src).unwrap().to_rgb8().into_raw();
        assert!(crate::codec::jpeg::history::is_aligned_jpeg_decode(&px, w, h));
        let r = CoverRequest { ghost_quality: Some(85), ..req(CoverMode::Ghost) };
        assert!(cover_needs_pixels(&src, &r));
        assert!(build_cover(&src, None, &r).is_err(), "pixels required");
        let out = build_cover(&src, Some((&px, w, h)), &r).unwrap();
        assert_realigned(&out, &px, w, h);
        // Decoded-JPEG pixels from any other source (a PNG export) as well.
        let png = build_cover(b"\x89PNG\r\n\x1a\n", Some((&px, w, h)), &req(CoverMode::Ghost)).unwrap();
        assert_realigned(&png, &px, w, h);
        // Pixels the platform already shrank (a JPEG over the encode limits)
        // are a resample: exact SI, no realignment.
        let small = resample_rgb(&px, w, h, 320, 240).unwrap();
        let out = build_cover(&src, Some((&small, 320, 240)), &r).unwrap();
        assert!(out.si_is_input && out.deep_cover);
        assert_eq!(out.jpeg, deep_cover_jpeg(&small, 320, 240, 85).unwrap());
    }

    /// Where the realignment cannot remove the lattice (a decode at QF 30:
    /// mostly flat blocks whose shifted edges are quantized DC steps), the
    /// guard still decides in the `*_si` calls: no side information, nzAC/10,
    /// no Deep Cover label.
    #[test]
    fn a_lattice_left_after_the_realignment_still_gets_the_guard() {
        use crate::stego::ghost::capacity::{si_kind, SiKind};
        let (w, h) = (640, 480);
        let flat: Vec<u8> = (0..w * h).flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let v = (60 + (x / 40) * 7 + (y / 30) * 5) as u8;
            [v, v, v]
        }).collect();
        let px = image::load_from_memory(&deep_cover_jpeg(&flat, w, h, 30).unwrap()).unwrap().to_rgb8().into_raw();
        let out = build_cover(b"\x89PNG\r\n\x1a\n", Some((&px, w, h)), &req(CoverMode::Ghost)).unwrap();
        let si = out.si_pixels_of(Some(&px)).unwrap();
        let img = JpegImage::from_bytes(&out.jpeg).unwrap();
        let kind = si_kind(&img, si, w, h).unwrap();
        assert_eq!(out.deep_cover, kind == SiKind::Exact, "label and classification agree");
        if crate::codec::jpeg::history::is_aligned_jpeg_decode(si, w, h) {
            assert_eq!(kind, SiKind::DecodedJpeg);
        }
    }

    #[test]
    fn ghost_resample_and_non_jpeg_sources_get_si_pixels() {
        let (w, h) = (1000, 750);
        let px = textured(w, h);
        let src = deep_cover_jpeg(&px, w, h, 95).unwrap();
        let r = CoverRequest { ghost_max_side: Some(600), ..req(CoverMode::Ghost) };
        assert!(cover_needs_pixels(&src, &r));
        let out = build_cover(&src, Some((&px, w, h)), &r).unwrap();
        assert_eq!((out.width, out.height), (600, 450));
        assert!(!out.si_is_input);
        let si = out.si_pixels.as_ref().unwrap();
        assert_eq!(si.len(), 600 * 450 * 3);
        let img = JpegImage::from_bytes(&out.jpeg).unwrap();
        assert_eq!(crate::stego::ghost::capacity::si_kind(&img, si, 600, 450).unwrap(), crate::stego::ghost::capacity::SiKind::Exact);
        assert!(out.deep_cover);
        // A PNG / HEIC source (not a JPEG): always written, always SI.
        let png = b"\x89PNG\r\n\x1a\n".to_vec();
        let out = build_cover(&png, Some((&px, w, h)), &req(CoverMode::Ghost)).unwrap();
        assert_eq!(out.si_pixels_of(Some(&px)), Some(&px[..]));
        assert!(out.si_is_input && out.si_pixels.is_none());
        assert_eq!(out.jpeg, deep_cover_jpeg(&px, w, h, DEEP_COVER_DEFAULT_QUALITY).unwrap());
    }

    #[test]
    fn armor_covers_use_the_preset_and_qf65() {
        let (w, h) = (2000, 3000); // 2:3 portrait
        let px = textured(w, h);
        let png = b"\x89PNG\r\n\x1a\n".to_vec();
        let std = build_cover(&png, Some((&px, w, h)), &req(CoverMode::Armor)).unwrap();
        assert_eq!(std.height, 1600);
        assert!(std.si_pixels.is_none());
        // The scale-friendly trim takes at most 4 px off the short side.
        assert!(std.width % 5 == 0 && std.width + 4 >= 2000 * 1600 / 3000, "{}", std.width);
        let qt = |j: &[u8]| JpegImage::from_bytes(j).unwrap().quant_table(0).unwrap().values;
        assert_eq!(qt(&std.jpeg), qt(&deep_cover_jpeg(&textured(256, 256), 256, 256, 65).unwrap()), "QF65");
        let ig = build_cover(&png, Some((&px, w, h)), &CoverRequest { armor_preset: ArmorCoverPreset::Instagram, ..req(CoverMode::Armor) }).unwrap();
        assert_eq!((ig.width, ig.height), (1080, 1350));
        let c = ig.crop.unwrap();
        assert!(c[1] > 0 && c[3] < h, "portrait cropped vertically: {c:?}");
        assert_eq!(dims(&ig.jpeg), (1080, 1350));
    }

    #[test]
    fn wand_off_keeps_the_source_metadata_wand_on_drops_it() {
        // A JPEG source carrying an EXIF block, written at a smaller size.
        let (w, h) = (800, 600);
        let px = textured(w, h);
        let plain = deep_cover_jpeg(&px, w, h, 95).unwrap();
        let tiff = b"MM\0*\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01\0\x06\0\0\0\0\0\0";
        let mut src = vec![0xFF, 0xD8, 0xFF, 0xE1];
        src.extend_from_slice(&((tiff.len() + 8) as u16).to_be_bytes());
        src.extend_from_slice(b"Exif\0\0");
        src.extend_from_slice(tiff);
        src.extend_from_slice(&plain[2..]);
        let r = CoverRequest { ghost_max_side: Some(400), ..req(CoverMode::Ghost) };
        let off = build_cover(&src, Some((&px, w, h)), &r).unwrap();
        assert!(crate::codec::jpeg::exif::source_exif(&off.jpeg).is_some());
        let on = build_cover(&src, Some((&px, w, h)), &CoverRequest { wand: true, ..r }).unwrap();
        assert!(crate::codec::jpeg::exif::source_exif(&on.jpeg).is_none());
    }

    /// The in-app cameras (iOS and Android `CameraExif`) save a PNG whose
    /// `eXIf` chunk is this block (pinned byte for byte in `CameraExifTest` /
    /// `camera_pixels_test.swift`): the cover keeps it with the wand off and
    /// none of it with the wand on.
    #[test]
    fn in_app_camera_png_exif_reaches_the_cover_unless_the_wand_is_on() {
        use crate::codec::jpeg::exif::source_exif;
        let hex = "4d4d002a000000080004010f0002000000070000003e01100002000000080000004601120003000000010001000087690004000000\
                   010000004e00000000476f6f676c650000506978656c203900000290030002000000140000006c90110002000000070000008000\
                   000000323032363a31313a30332030393a30353a3037002b30313a30300000";
        let tiff: Vec<u8> = (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect();
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&(tiff.len() as u32).to_be_bytes());
        png.extend_from_slice(b"eXIf");
        png.extend_from_slice(&tiff);
        png.extend_from_slice(&[0; 4]); // CRC (core's reader does not check it)
        png.extend_from_slice(b"\0\0\0\0IDAT\0\0\0\0");
        assert_eq!(source_exif(&png).as_deref(), Some(&tiff[..]));

        let (w, h) = (640, 480);
        let px = textured(w, h);
        for mode in [CoverMode::Ghost, CoverMode::Armor] {
            let off = build_cover(&png, Some((&px, w, h)), &req(mode)).unwrap();
            let kept = source_exif(&off.jpeg).expect("wand off keeps the capture's EXIF");
            let has = |needle: &[u8]| kept.windows(needle.len()).any(|x| x == needle);
            assert!(has(b"2026:11:03 09:05:07\0") && has(b"+01:00\0") && has(b"Google\0") && has(b"Pixel 9\0"), "{mode:?}");
            let on = build_cover(&png, Some((&px, w, h)), &CoverRequest { wand: true, ..req(mode) }).unwrap();
            assert!(source_exif(&on.jpeg).is_none(), "{mode:?}: the wand strips everything");
            // Only metadata differs: the quantisation tables (and the pixels) are the same.
            let (a, b) = (JpegImage::from_bytes(&off.jpeg).unwrap(), JpegImage::from_bytes(&on.jpeg).unwrap());
            assert_eq!(a.quant_table(0).unwrap().values, b.quant_table(0).unwrap().values);
        }
    }
}

