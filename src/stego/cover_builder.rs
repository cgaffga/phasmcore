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
//! | Ghost | JPEG source, original size, quality chip | writer at that quality (a re-encode) | the pixels (core: decoded JPEG, no SI, nzAC/10) |
//! | Ghost | JPEG source made smaller, or any non-JPEG source | writer from the (resampled) pixels | the pixels |
//! | Armor | any | `armor_cover_dimensions` → resample → crop → writer at QF65 | none |
//!
//! With the wand off, a written cover carries the source's metadata, cleaned
//! ([`crate::codec::jpeg::exif`]); with it on, none. Every written Ghost
//! cover returns its pixels, and core decides from them what the `*_si`
//! calls do ([`super::ghost::capacity::si_kind`]): exact side information
//! and nzAC/4 for a precover; for a JPEG decoded and re-written on its own
//! grid (the quality chip at the original size) the decoded-JPEG guard
//! ([`crate::codec::jpeg::history`]) drops the side information (plain v3
//! encode) and caps the bands at nzAC/10 (fix list R3.7: before, those
//! covers got the plain calls and nzAC/5, though an aligned double
//! compression measured hist P_E 0.298 at 0.20 bpnzAC).

use crate::codec::jpeg::error::JpegError;
use crate::codec::jpeg::exif::cover_with_source_metadata;
use crate::codec::jpeg::lossless::upright_and_clean;
use crate::codec::resample::{fit_long_side, resample_rgb};
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
    /// they are not the caller's own input pixels (a resample made them);
    /// see [`CoverOut::si_pixels_of`]. `None` with `si_is_input` false =
    /// the plain calls.
    pub si_pixels: Option<Vec<u8>>,
    /// The `*_si` pixels are exactly the pixels the caller passed to
    /// [`build_cover`] (a same-size Ghost cover): `si_pixels` is then
    /// `None` and the caller reuses its own buffer, so core returns no
    /// copy (48 MP: 145 MB).
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
    /// and for a decoded JPEG re-written on its grid (the quality chip at
    /// the original size), whose `*_si` calls are plain Ghost at nzAC/10.
    /// What a "Deep Cover" label must show (not `si_pixels.is_some()`).
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
            // Same size: the writer reads the caller's pixels (no copy).
            let resampled = if (tw, th) == (w, h) { None } else { Some(resample_rgb(rgb, w, h, tw, th)?) };
            let px: &[u8] = resampled.as_deref().unwrap_or(rgb);
            let quality = req.ghost_quality.unwrap_or(DEEP_COVER_DEFAULT_QUALITY);
            let jpeg = with_metadata(deep_cover_jpeg(px, tw, th, quality)?, original, req)?;
            // The pixels go to the `*_si` calls in every case; core
            // classifies them. A JPEG source written at its own size (the
            // quality chip) is a re-encode of a decoded JPEG: core's guard
            // sees the source's quantization lattice on the cover's grid and
            // embeds without side information at nzAC/10. A resample (by
            // the chip, or by the platform for a JPEG over the encode
            // limits) is a valid precover: exact side information.
            // A same-size cover returns no copy: `si_is_input` says the
            // caller's own pixels are the ones (fix list R3.13 follow-up).
            let deep_cover = !is_decoded_jpeg_on_grid(px, tw, th);
            let si_is_input = resampled.is_none();
            Ok(CoverOut { jpeg, si_pixels: resampled, si_is_input, width: tw, height: th, pass_through: false, crop: None, deep_cover })
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

    #[test]
    fn ghost_quality_chip_on_a_jpeg_gets_the_decoded_jpeg_treatment() {
        use crate::stego::ghost::capacity::{count_nonzero_ac, plaintext_for_frame_bits, si_kind, SiKind, MAX_RATE_DIV_DECODED_JPEG};
        let (w, h) = (640, 480);
        let src = deep_cover_jpeg(&textured(w, h), w, h, 90).unwrap();
        // The platform's decode of the JPEG source.
        let px = image::load_from_memory(&src).unwrap().to_rgb8().into_raw();
        let r = CoverRequest { ghost_quality: Some(85), ..req(CoverMode::Ghost) };
        assert!(cover_needs_pixels(&src, &r));
        let out = build_cover(&src, Some((&px, w, h)), &r).unwrap();
        assert_eq!(dims(&out.jpeg), (w, h));
        assert!(build_cover(&src, None, &r).is_err(), "pixels required");
        // R3.7: the pixels come back, so the apps' `*_si` calls let core
        // classify the cover: a decoded JPEG, nzAC/10, no side information.
        assert!(out.si_is_input && out.si_pixels.is_none(), "same size: the caller's pixels, no copy");
        let si = out.si_pixels_of(Some(&px)).expect("pixels for the *_si calls");
        assert_eq!(si, &px[..]);
        let img = JpegImage::from_bytes(&out.jpeg).unwrap();
        assert_eq!(si_kind(&img, si, w, h).unwrap(), SiKind::DecodedJpeg);
        assert!(!out.deep_cover, "no Deep Cover label");
        let bands = crate::stego::ghost_bands_si(&img, si, w, h, 0, 0).unwrap();
        let nz = count_nonzero_ac(img.dct_grid(0));
        assert_eq!(bands.max_bytes, plaintext_for_frame_bits(nz / MAX_RATE_DIV_DECODED_JPEG));
        assert!(bands.max_bytes < crate::stego::ghost_bands(&img, 0, 0).unwrap().max_bytes, "below the plain nzAC/5");
        // ... and the SI encode is plain Ghost v3, byte for byte.
        let msg = "quality chip";
        let (plain, si_stego) = {
            let _seed = crate::stego::crypto::DeterministicSeedGuard::set("307");
            let plain = crate::stego::ghost_encode(&out.jpeg, msg, "pw").unwrap();
            let (si_stego, _) = crate::stego::ghost_encode_si_with_files_quality(&out.jpeg, si, w, h, msg, &[], "pw").unwrap();
            (plain, si_stego)
        };
        assert_eq!(si_stego, plain, "no side information used");
        assert_eq!(crate::stego::ghost_decode(&si_stego, "pw").unwrap().text, msg);
        // Pixels the platform already shrank (a JPEG over the encode limits)
        // are a resample: exact SI.
        let small = resample_rgb(&px, w, h, 320, 240).unwrap();
        let out = build_cover(&src, Some((&small, 320, 240)), &r).unwrap();
        let img = JpegImage::from_bytes(&out.jpeg).unwrap();
        assert_eq!(si_kind(&img, out.si_pixels_of(Some(&small)).unwrap(), 320, 240).unwrap(), SiKind::Exact);
        assert!(out.deep_cover);
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

