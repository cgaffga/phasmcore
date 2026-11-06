// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! The source's pixels as core's `build_cover` wants them: upright 8-bit
//! sRGB, 3 bytes a pixel (`docs/design/app/encode-cover-pipeline.md` §1, the
//! same contract as the apps).
//!
//! - **Orientation**: the EXIF `Orientation` tag is applied to the pixels.
//! - **Colour**: an embedded ICC profile (Display P3, Adobe RGB, ProPhoto,
//!   a grey gamma, …) is converted to sRGB (relative colorimetric, through
//!   `moxcms`, the colour engine the `image` crate itself links), the way iOS
//!   draws into an sRGB context and the browsers' canvas does. A profile
//!   that is sRGB in all but name (every test colour within 1 level) is left
//!   alone, so a plain sRGB file keeps its exact pixels. Without a profile
//!   the pixels are taken as sRGB. A profile that is unreadable or does not
//!   match the pixels (CMYK, Lab) is ignored, with a note.
//! - **Alpha**: dropped. The colour channels are kept as stored
//!   (unpremultiplied, as PNG / WebP / TIFF store them), not composited over
//!   any background: the rule `encode-cover-pipeline.md` states ("alpha
//!   dropped").

use moxcms::{ColorProfile, DataColorSpace, Layout, ProfileText, RenderingIntent, TransformOptions};
use image::{ColorType, DynamicImage};

/// Rows converted per transform call (bounds the 16-bit scratch buffers).
const ROWS_PER_CHUNK: usize = 64;

/// Upright 8-bit sRGB pixels of `img` (already turned upright), whose file
/// carried the ICC profile `icc`. Also returns a note for the user when the
/// colours were converted or a profile was ignored.
pub fn to_srgb8(img: DynamicImage, icc: Option<&[u8]>) -> (Vec<u8>, Option<String>) {
    let Some(icc) = icc else {
        return (img.into_rgb8().into_raw(), None);
    };
    let profile = match ColorProfile::new_from_slice(icc) {
        Ok(p) => p,
        Err(_) => return (img.into_rgb8().into_raw(), Some("unreadable ICC profile ignored (pixels taken as sRGB)".into())),
    };
    let grey = !img.color().has_color();
    let layout = match (profile.color_space, grey) {
        (DataColorSpace::Rgb, false) => Layout::Rgb,
        (DataColorSpace::Gray, true) => Layout::Gray,
        (cs, _) => {
            let note = format!("{cs:?} ICC profile does not match the decoded pixels; ignored (pixels taken as sRGB)");
            return (img.into_rgb8().into_raw(), Some(note));
        }
    };
    let options = TransformOptions {
        rendering_intent: RenderingIntent::RelativeColorimetric,
        prefer_fixed_point: false,
        ..TransformOptions::default()
    };
    let Ok(transform) = profile.create_transform_16bit(layout, &ColorProfile::new_srgb(), Layout::Rgb, options) else {
        return (img.into_rgb8().into_raw(), Some("unsupported ICC profile ignored (pixels taken as sRGB)".into()));
    };
    let convert = |src: &[u16], dst: &mut [u16]| transform.transform(src, dst).is_ok();
    let channels = if layout == Layout::Gray { 1 } else { 3 };
    if is_srgb_already(channels, &convert) {
        return (img.into_rgb8().into_raw(), None);
    }

    let width = img.width() as usize;
    let wide = matches!(
        img.color(),
        ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16 | ColorType::Rgb32F | ColorType::Rgba32F
    );
    // The source samples as 16 bit (8-bit sources widened chunk by chunk).
    let src16: Vec<u16>;
    let src8: Vec<u8>;
    match (wide, channels) {
        (true, 1) => (src16, src8) = (img.into_luma16().into_raw(), Vec::new()),
        (true, _) => (src16, src8) = (img.into_rgb16().into_raw(), Vec::new()),
        (false, 1) => (src16, src8) = (Vec::new(), img.into_luma8().into_raw()),
        (false, _) => (src16, src8) = (Vec::new(), img.into_rgb8().into_raw()),
    }
    let total = if wide { src16.len() } else { src8.len() };
    let mut out = Vec::with_capacity(total / channels * 3);
    let (mut inbuf, mut outbuf) = (Vec::new(), Vec::new());
    let step = ROWS_PER_CHUNK * width * channels;
    let mut start = 0;
    while start < total {
        let end = (start + step).min(total);
        inbuf.clear();
        if wide {
            inbuf.extend_from_slice(&src16[start..end]);
        } else {
            inbuf.extend(src8[start..end].iter().map(|&v| v as u16 * 257));
        }
        outbuf.resize((end - start) / channels * 3, 0);
        if !convert(&inbuf, &mut outbuf) {
            // Cannot happen for matching buffer sizes; fall back unconverted.
            let fallback = if wide {
                src16.iter().map(|&v| to8(v)).collect::<Vec<u8>>()
            } else {
                src8.clone()
            };
            let rgb = if channels == 1 { fallback.iter().flat_map(|&g| [g, g, g]).collect() } else { fallback };
            return (rgb, Some("ICC conversion failed; pixels taken as sRGB".into()));
        }
        out.extend(outbuf.iter().map(|&v| to8(v)));
        start = end;
    }
    let name = profile_name(&profile);
    (out, Some(format!("colours converted to sRGB from the embedded ICC profile ({name})")))
}

/// 16-bit sample to 8 bit, rounded.
fn to8(v: u16) -> u8 {
    ((v as u32 * 255 + 32_767) / 65_535) as u8
}

/// True when the transform maps every test colour (a grey ramp and a 9-level
/// RGB lattice, as 8-bit values) to itself within one level: the profile is
/// sRGB, perhaps under another name or with a sampled curve.
fn is_srgb_already(channels: usize, convert: &dyn Fn(&[u16], &mut [u16]) -> bool) -> bool {
    let levels: Vec<u8> = (0..=8).map(|i| (i * 255 / 8) as u8).collect();
    let mut samples: Vec<[u8; 3]> = (0..=255u8).map(|g| [g, g, g]).collect();
    if channels == 3 {
        for &r in &levels {
            for &g in &levels {
                for &b in &levels {
                    samples.push([r, g, b]);
                }
            }
        }
    }
    let src: Vec<u16> = samples.iter().flat_map(|s| s[..channels].iter().map(|&v| v as u16 * 257)).collect();
    let mut dst = vec![0u16; samples.len() * 3];
    if !convert(&src, &mut dst) {
        return false;
    }
    samples
        .iter()
        .zip(dst.as_chunks::<3>().0)
        .all(|(s, d)| (0..3).all(|c| (to8(d[c]) as i16 - s[c] as i16).abs() <= 1))
}

fn profile_name(p: &ColorProfile) -> String {
    let text = match &p.description {
        Some(ProfileText::PlainString(s)) => s.clone(),
        Some(ProfileText::Localizable(v)) => v.first().map(|s| s.value.clone()).unwrap_or_default(),
        Some(ProfileText::Description(d)) => d.ascii_string.clone(),
        None => String::new(),
    };
    let text = text.trim_matches(|c: char| c == '\0' || c.is_whitespace()).to_string();
    if text.is_empty() { "unnamed".into() } else { text }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgb, RgbImage};

    fn photo(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| Rgb([(x * 7 % 256) as u8, (y * 5 % 256) as u8, ((x + y) * 3 % 256) as u8]))
    }

    fn icc(p: &ColorProfile) -> Vec<u8> {
        p.encode().expect("encode ICC")
    }

    #[test]
    fn no_profile_and_srgb_profile_keep_the_pixels() {
        let img = photo(40, 30);
        let raw = img.as_raw().clone();
        assert_eq!(to_srgb8(DynamicImage::ImageRgb8(img.clone()), None), (raw.clone(), None));
        let srgb = icc(&ColorProfile::new_srgb());
        assert_eq!(to_srgb8(DynamicImage::ImageRgb8(img), Some(&srgb)), (raw, None));
    }

    #[test]
    fn display_p3_is_converted_to_srgb() {
        // P3 pure red lies outside sRGB: it clips to sRGB red; a P3 mid grey
        // stays grey (same white point and transfer curve).
        let img = RgbImage::from_fn(2, 1, |x, _| if x == 0 { Rgb([255, 0, 0]) } else { Rgb([128, 128, 128]) });
        let p3 = icc(&ColorProfile::new_display_p3());
        let (rgb, note) = to_srgb8(DynamicImage::ImageRgb8(img), Some(&p3));
        assert!(note.unwrap().contains("converted"));
        assert_eq!(&rgb[..3], &[255, 0, 0]);
        assert!(rgb[3..].iter().all(|&v| (v as i16 - 128).abs() <= 1), "{rgb:?}");
        // A less saturated P3 colour moves: P3 (200, 100, 50) is more vivid
        // than sRGB (200, 100, 50), so its sRGB red goes up.
        let img = RgbImage::from_pixel(1, 1, Rgb([200, 100, 50]));
        let (rgb, _) = to_srgb8(DynamicImage::ImageRgb8(img), Some(&p3));
        assert!(rgb[0] > 205 && rgb[1] < 100, "{rgb:?}");
    }

    #[test]
    fn sixteen_bit_and_grey_sources() {
        let img = image::ImageBuffer::<Rgb<u16>, _>::from_pixel(3, 2, Rgb([65535u16, 0, 0]));
        let p3 = icc(&ColorProfile::new_display_p3());
        let (rgb, _) = to_srgb8(DynamicImage::ImageRgb16(img), Some(&p3));
        assert_eq!(rgb.len(), 3 * 2 * 3);
        assert_eq!(&rgb[..3], &[255, 0, 0]);
        // A grey image with a grey profile becomes three equal channels.
        let g = GrayImage::from_fn(5, 4, |x, y| Luma([(x * 40 + y * 10) as u8]));
        let gray22 = icc(&ColorProfile::new_gray_with_gamma(2.2));
        let (rgb, _) = to_srgb8(DynamicImage::ImageLuma8(g), Some(&gray22));
        assert_eq!(rgb.len(), 5 * 4 * 3);
        assert!(rgb.chunks(3).all(|p| p[0] == p[1] && p[1] == p[2]));
    }

    #[test]
    fn alpha_is_dropped_without_compositing() {
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([10, 200, 30, 0]));
        let (rgb, _) = to_srgb8(DynamicImage::ImageRgba8(img), None);
        assert_eq!(rgb, [10, 200, 30].repeat(4));
    }

    #[test]
    fn a_broken_or_mismatched_profile_is_ignored() {
        let img = photo(8, 8);
        let raw = img.as_raw().clone();
        let (rgb, note) = to_srgb8(DynamicImage::ImageRgb8(img.clone()), Some(b"not an icc profile"));
        assert_eq!(rgb, raw);
        assert!(note.unwrap().contains("ignored"));
        let gray = icc(&ColorProfile::new_gray_with_gamma(2.2));
        let (rgb, note) = to_srgb8(DynamicImage::ImageRgb8(img), Some(&gray));
        assert_eq!(rgb, raw);
        assert!(note.unwrap().contains("ignored"));
    }
}
