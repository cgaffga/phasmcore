// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use phasm_core::{
    build_cover, cover_needs_pixels, ArmorCoverPreset, CoverMode, CoverRequest, ARMOR_COVER_QUALITY,
    ARMOR_TARGET_DIMENSION, DEEP_COVER_DEFAULT_QUALITY,
};
use std::io::Cursor;
use std::path::Path;

/// Result of preparing an image for encoding.
pub struct PreparedImage {
    pub jpeg_bytes: Vec<u8>,
    /// Ghost: the pixels the side-informed calls get (`build_cover`'s
    /// `si_pixels`); None = the plain calls. Always None for Armor.
    pub raw_pixels: Option<Vec<u8>>,
    /// Dimensions of the cover that gets encoded.
    pub width: u32,
    pub height: u32,
    /// Dimensions of the input (after EXIF orientation).
    pub source_width: u32,
    pub source_height: u32,
    /// The quality phasm wrote the cover at; None when the JPEG was used as
    /// it is (possibly cleaned losslessly, see `lossless_clean`).
    pub reencoded_qf: Option<u8>,
    /// A JPEG input was cleaned without re-encoding (`--optimize`: metadata
    /// dropped, EXIF orientation applied in the DCT domain).
    pub lossless_clean: bool,
    /// Ghost: a JPEG input was re-encoded at its own size (`--qf` without a
    /// real downscale): an aligned double-compressed cover, embedded
    /// without side information.
    pub jpeg_reencoded_same_size: bool,
    /// What happened to the source's colours (ICC conversion, an ignored
    /// profile), for a note; None when the pixels were taken as they are.
    pub color_note: Option<String>,
}

impl PreparedImage {
    /// An Armor cover larger than `ARMOR_TARGET_DIMENSION` still decodes, but
    /// does not survive recompression or resizing.
    pub fn exceeds_armor_target(&self) -> bool {
        self.width.max(self.height) > ARMOR_TARGET_DIMENSION
    }
}

/// `--armor-preset` values: the channel the Armor message is meant for.
pub fn parse_armor_preset(s: &str) -> Result<ArmorCoverPreset, String> {
    match s.to_lowercase().as_str() {
        "standard" | "whatsapp" | "1600" => Ok(ArmorCoverPreset::Standard),
        "telegram" | "1280" => Ok(ArmorCoverPreset::Telegram),
        "instagram" | "1080" => Ok(ArmorCoverPreset::Instagram),
        other => Err(format!("unknown Armor preset {other}: use standard, telegram or instagram")),
    }
}

/// The request core's `build_cover` gets for the CLI flags.
///
/// - Ghost: `--qf` is the quality chip, `--resize` the resolution chip.
/// - Armor: the app cover of `--armor-preset` (QF65). `--qf` never changes
///   it (`encode` rejects `--qf` with Armor; it is ignored here).
/// - Armor with `--full-res` or `--resize` (CLI-only, not recompression
///   robust): core's Ghost size rules (original size within the encode
///   limits, or the long side at most `--resize`) at QF65; a JPEG that needs
///   no resize under `--full-res` is used as it is (cleaned losslessly with
///   `--optimize`).
/// - `--optimize` is the apps' magic wand: no metadata on the cover. Without
///   it a JPEG used as it is keeps all its metadata, and a cover phasm writes
///   carries the source's EXIF, cleaned (`encode` warns about both).
fn cover_request(
    original: &[u8],
    mode: &str,
    qf: Option<u8>,
    resize: Option<u32>,
    optimize: bool,
    armor_full_res: bool,
    armor_preset: ArmorCoverPreset,
) -> CoverRequest {
    let base = CoverRequest { mode: CoverMode::Ghost, wand: optimize, ghost_quality: None, ghost_max_side: None, armor_preset };
    match mode {
        "ghost" => CoverRequest { ghost_quality: qf, ghost_max_side: resize, ..base },
        _ if !armor_full_res && resize.is_none() => CoverRequest { mode: CoverMode::Armor, ..base },
        _ => {
            let sized = CoverRequest { ghost_max_side: resize, ..base };
            let written = resize.is_some() || cover_needs_pixels(original, &sized);
            CoverRequest { ghost_quality: written.then_some(ARMOR_COVER_QUALITY), ..sized }
        }
    }
}

/// Prepare an image for steganographic encoding: core's `build_cover`, the
/// same rules as the apps (`docs/design/app/encode-cover-pipeline.md`). The
/// CLI only decodes the input to upright 8-bit sRGB pixels when core asks
/// for them (`cover_needs_pixels`): EXIF orientation applied, an embedded
/// ICC profile converted to sRGB, alpha dropped ([`crate::color`]).
///
/// - Ghost, JPEG, original size, no `--qf`: the file as it is (with
///   `--optimize`: cleaned losslessly, orientation applied in the DCT domain).
/// - Ghost, JPEG, `--qf` at its own size: a plain re-encode, no side
///   information (a decoded JPEG is no precover).
/// - Ghost, JPEG made smaller (`--resize`), or any other format: the Deep
///   Cover cover phasm writes (IJG 92 or `--qf`), with its pixels for
///   SI-UNIWARD.
/// - Armor: `armor_cover_dimensions` for `--armor-preset` (long side at most
///   1600 px, 1280 px for Telegram, width at most 1080 px for Instagram, the
///   free side center-cropped to a multiple of 5 px) → core's resampler →
///   crop → QF65, even when no resize is needed (the pre-settle). Armor
///   survives recompression only from that cover; `--full-res` keeps the
///   original size (decodes, but is not robust).
pub fn prepare_image(
    path: &Path,
    mode: &str,
    qf: Option<u8>,
    resize: Option<u32>,
    optimize: bool,
    armor_full_res: bool,
    armor_preset: ArmorCoverPreset,
) -> Result<PreparedImage, CliError> {
    let original = std::fs::read(path)?;
    let format = detect_format(path, &original)?;
    let req = cover_request(&original, mode, qf, resize, optimize, armor_full_res, armor_preset);
    let needs_pixels = cover_needs_pixels(&original, &req);
    let source = if needs_pixels { Some(decode_upright_srgb(&original, format)?) } else { None };
    let out = build_cover(&original, source.as_ref().map(|s| (&s.rgb[..], s.width, s.height)), &req)?;
    let ghost = mode == "ghost";
    let (source_width, source_height, color_note, source_rgb) = match source {
        Some(s) => (s.width, s.height, s.color_note, Some(s.rgb)),
        None => (out.width, out.height, None, None),
    };
    // The SI pixels are the source's own when core returns no copy.
    let raw_pixels = if !ghost {
        None
    } else if out.si_is_input {
        source_rgb
    } else {
        out.si_pixels
    };
    let default_quality = if req.mode == CoverMode::Armor { ARMOR_COVER_QUALITY } else { DEEP_COVER_DEFAULT_QUALITY };
    Ok(PreparedImage {
        jpeg_bytes: out.jpeg,
        // A written Ghost cover without exact side information: a JPEG
        // re-encoded at its own size (core's decoded-JPEG guard caps it).
        jpeg_reencoded_same_size: ghost && needs_pixels && !out.deep_cover,
        raw_pixels,
        width: out.width,
        height: out.height,
        source_width,
        source_height,
        reencoded_qf: needs_pixels.then(|| req.ghost_quality.unwrap_or(default_quality)),
        lossless_clean: !needs_pixels && optimize,
        color_note,
    })
}

/// A source decoded for `build_cover`.
struct SourcePixels {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    color_note: Option<String>,
}

/// The upright 8-bit sRGB pixels of the input (see [`crate::color`]).
fn decode_upright_srgb(bytes: &[u8], format: ImageFormat) -> Result<SourcePixels, CliError> {
    let mut decoder = ImageReader::with_format(Cursor::new(bytes), format)
        .into_decoder()
        .map_err(CliError::ImageLoad)?;
    // Phone photos store their rotation as an EXIF tag, and re-encoding drops
    // it, so turn it into pixels here or the cover comes out sideways.
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let icc = decoder.icc_profile().ok().flatten();
    let mut img = DynamicImage::from_decoder(decoder).map_err(CliError::ImageLoad)?;
    img.apply_orientation(orientation);
    let (width, height) = (img.width(), img.height());
    let (rgb, color_note) = crate::color::to_srgb8(img, icc.as_deref());
    Ok(SourcePixels { rgb, width, height, color_note })
}

fn detect_format(path: &Path, bytes: &[u8]) -> Result<ImageFormat, CliError> {
    // Try extension first, then magic bytes
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext.to_lowercase().as_str() {
            "jpg" | "jpeg" => return Ok(ImageFormat::Jpeg),
            "png" => return Ok(ImageFormat::Png),
            "webp" => return Ok(ImageFormat::WebP),
            "bmp" => return Ok(ImageFormat::Bmp),
            "tiff" | "tif" => return Ok(ImageFormat::Tiff),
            "gif" => return Ok(ImageFormat::Gif),
            "heic" | "heif" | "avif" | "cr2" | "cr3" | "nef" | "arw" | "raf" | "rw2" | "dng" => {
                return Err(CliError::UnsupportedFormat(ext.to_string()));
            }
            _ => {}
        }
    }
    // Fallback to magic bytes
    image::guess_format(bytes).map_err(CliError::ImageLoad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, ImageEncoder, Rgb, RgbImage};
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::var("PHASM_TMP")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir())
            .join(format!("phasm-cli-image-conv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// A textured w×h JPEG (Armor needs texture for capacity). The left
    /// quarter is dark so a rotation can be told apart from the original.
    fn textured_jpeg(w: u32, h: u32) -> Vec<u8> {
        let mut s: u32 = 0x1234_5678;
        let img = RgbImage::from_fn(w, h, |x, _| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = (s >> 24) as u8 / 4;
            let base = if x < w / 4 { 20 } else { 150 };
            Rgb([base + n, base + n, base + n])
        });
        phasm_core::deep_cover_jpeg(img.as_raw(), w, h, 90).unwrap()
    }

    /// Insert an EXIF APP1 segment carrying Orientation = `value` after SOI.
    fn with_exif_orientation(jpeg: &[u8], value: u16) -> Vec<u8> {
        let mut tiff = b"MM\x00\x2a\x00\x00\x00\x08\x00\x01".to_vec();
        tiff.extend_from_slice(&[0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01]);
        tiff.extend_from_slice(&value.to_be_bytes());
        tiff.extend_from_slice(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        let mut payload = b"Exif\x00\x00".to_vec();
        payload.extend_from_slice(&tiff);
        let mut out = jpeg[..2].to_vec();
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&payload);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    /// `CliError` has no `Debug`, so `unwrap` is unavailable.
    fn prep(p: &Path, mode: &str, qf: Option<u8>, resize: Option<u32>, optimize: bool, full_res: bool) -> PreparedImage {
        prepare_image(p, mode, qf, resize, optimize, full_res, ArmorCoverPreset::Standard).unwrap_or_else(|e| panic!("{e}"))
    }

    fn write(name: &str, bytes: &[u8]) -> PathBuf {
        let p = scratch(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    fn jpeg_dims(bytes: &[u8]) -> (u32, u32) {
        image::load_from_memory(bytes).unwrap().dimensions()
    }

    #[test]
    fn armor_shrinks_a_large_cover_to_the_target_at_qf65() {
        let p = write("large.jpg", &textured_jpeg(2400, 1800));
        let c = prep(&p, "armor", None, None, false, false);
        assert_eq!((c.width, c.height), (1600, 1200));
        assert_eq!((c.source_width, c.source_height), (2400, 1800));
        assert_eq!(c.reencoded_qf, Some(65));
        assert_eq!(jpeg_dims(&c.jpeg_bytes), (1600, 1200));
        assert!(!c.exceeds_armor_target());
    }

    #[test]
    fn armor_cover_follows_the_core_size_rule() {
        // 3:2 -> 1600 x 1067 scaled, short side cropped to a multiple of 5.
        let p = write("odd.jpg", &textured_jpeg(2400, 1600));
        for (preset, dims) in [
            (ArmorCoverPreset::Standard, (1600, 1065)),
            (ArmorCoverPreset::Telegram, (1280, 850)),
            (ArmorCoverPreset::Instagram, (1080, 720)),
        ] {
            let c = prepare_image(&p, "armor", None, None, false, false, preset).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!((c.width, c.height), dims, "{preset:?}");
            assert_eq!(jpeg_dims(&c.jpeg_bytes), dims);
            let d = phasm_core::armor_cover_dimensions(2400, 1600, preset);
            assert_eq!((d.width, d.height), dims);
        }
        // An explicit --resize keeps the plain long-side resize.
        let c = prep(&p, "armor", None, Some(1600), false, false);
        assert_eq!((c.width, c.height), (1600, 1067));
        // Instagram crops a 2:3 portrait to 4:5 (the middle of 1080 x 1620).
        let p = write("portrait.jpg", &textured_jpeg(1600, 2400));
        let c = prepare_image(&p, "armor", None, None, false, false, ArmorCoverPreset::Instagram)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((c.width, c.height), (1080, 1350));
        assert_eq!(jpeg_dims(&c.jpeg_bytes), (1080, 1350));
    }

    #[test]
    fn armor_re_encodes_a_small_cover_too() {
        // The apps pre-settle every Armor cover at QF65, resized or not.
        let original = textured_jpeg(800, 600);
        let p = write("small.jpg", &original);
        let c = prep(&p, "armor", None, None, false, false);
        assert_eq!((c.width, c.height), (800, 600));
        assert_eq!(c.reencoded_qf, Some(65));
        assert_ne!(c.jpeg_bytes, original);
    }

    #[test]
    fn armor_full_res_passes_the_file_through() {
        let original = textured_jpeg(2400, 1800);
        let p = write("fullres.jpg", &original);
        let c = prep(&p, "armor", None, None, false, true);
        assert_eq!(c.jpeg_bytes, original);
        assert_eq!((c.width, c.height), (2400, 1800));
        assert_eq!(c.reencoded_qf, None);
        assert!(c.exceeds_armor_target());
    }

    #[test]
    fn ghost_still_passes_a_jpeg_through() {
        let original = textured_jpeg(2400, 1800);
        let p = write("ghost.jpg", &original);
        let c = prep(&p, "ghost", None, None, false, false);
        assert_eq!(c.jpeg_bytes, original);
        assert!(c.raw_pixels.is_none());
    }

    #[test]
    fn ghost_qf_on_a_jpeg_is_a_plain_re_encode() {
        // A same-size re-encode of a decoded JPEG is no Deep Cover precover
        // (aligned double compression): core gets the pixels so its
        // decoded-JPEG guard applies, and encodes plain Ghost.
        let p = write("ghost-qf.jpg", &textured_jpeg(640, 480));
        let c = prep(&p, "ghost", Some(85), None, false, false);
        assert!(c.raw_pixels.is_some());
        assert_eq!(c.reencoded_qf, Some(85));
        assert!(c.jpeg_reencoded_same_size);
        assert_eq!(jpeg_dims(&c.jpeg_bytes), (640, 480));
        // A --resize that does not shrink adds nothing: same answer.
        let c = prep(&p, "ghost", Some(85), Some(4000), false, false);
        assert!(c.raw_pixels.is_some() && c.jpeg_reencoded_same_size);
    }

    #[test]
    fn ghost_resize_that_does_not_shrink_passes_the_jpeg_through() {
        let original = textured_jpeg(640, 480);
        let p = write("ghost-noresize.jpg", &original);
        let c = prep(&p, "ghost", None, Some(4000), false, false);
        assert_eq!(c.jpeg_bytes, original);
        assert!(c.raw_pixels.is_none() && c.reencoded_qf.is_none());
    }

    #[test]
    fn ghost_real_downscale_of_a_jpeg_keeps_deep_cover() {
        // A resampled JPEG is a valid precover: pixels, and core finds the
        // side information exact (the bonus applies).
        let p = write("ghost-down.jpg", &textured_jpeg(1280, 960));
        let c = prep(&p, "ghost", None, Some(800), false, false);
        assert_eq!((c.width, c.height), (800, 600));
        let px = c.raw_pixels.as_deref().expect("pixels");
        assert!(!phasm_core::is_aligned_jpeg_decode(px, c.width, c.height));
        let img = phasm_core::JpegImage::from_bytes(&c.jpeg_bytes).unwrap();
        let plain = phasm_core::ghost_bands(&img, 0, 0).unwrap();
        let si = phasm_core::ghost_bands_si(&img, px, c.width, c.height, 0, 0).unwrap();
        assert!(si.max_bytes > plain.max_bytes);
    }

    #[test]
    fn ghost_optimize_on_a_jpeg_cleans_it_losslessly() {
        // Orientation 6 (stored 640x480, shown 480x640): turned upright in
        // the DCT domain, metadata dropped, no re-encode, no pixels.
        let src = textured_jpeg(640, 480);
        let p = write("ghost-opt.jpg", &with_exif_orientation(&src, 6));
        let c = prep(&p, "ghost", None, None, true, false);
        assert!(c.lossless_clean && c.raw_pixels.is_none() && c.reencoded_qf.is_none());
        assert_eq!(jpeg_dims(&c.jpeg_bytes), (480, 640));
        assert!(!c.jpeg_bytes.windows(6).any(|w| w == b"Exif\0\0"));
        let img = phasm_core::JpegImage::from_bytes(&c.jpeg_bytes).unwrap();
        let orig = phasm_core::JpegImage::from_bytes(&src).unwrap();
        // Same coefficients, moved: the non-zero AC count is unchanged.
        let nz = |i: &phasm_core::JpegImage| i.dct_grid(0).coeffs().chunks(64).map(|b| b[1..].iter().filter(|&&v| v != 0).count()).sum::<usize>();
        assert_eq!(nz(&img), nz(&orig));
        // And the round trip works on it.
        let (stego, _) = phasm_core::ghost_encode_with_quality(&c.jpeg_bytes, "lossless", "pw").unwrap();
        assert_eq!(phasm_core::ghost_decode(&stego, "pw").unwrap().text, "lossless");
    }

    #[test]
    fn exif_rotation_is_applied_when_the_cover_is_re_encoded() {
        // Orientation 6 = rotate 90° clockwise to display: stored 2400×1800
        // shows as 1800×2400, and the stored left edge (dark) ends up on top.
        let p = write("rotated.jpg", &with_exif_orientation(&textured_jpeg(2400, 1800), 6));
        let c = prep(&p, "armor", None, None, false, false);
        assert_eq!((c.width, c.height), (1200, 1600));
        assert_eq!((c.source_width, c.source_height), (1800, 2400));
        let img = image::load_from_memory(&c.jpeg_bytes).unwrap().to_luma8();
        let mean = |y0: u32, y1: u32| {
            let rows = (y0..y1).flat_map(|y| (0..img.width()).map(move |x| (x, y)));
            let (sum, n) = rows.fold((0u64, 0u64), |(s, n), (x, y)| (s + img.get_pixel(x, y)[0] as u64, n + 1));
            sum / n
        };
        assert!(mean(0, 300) + 60 < mean(1300, 1600), "the dark edge should be on top");
    }

    #[test]
    fn the_default_armor_cover_round_trips() {
        let p = write("roundtrip.jpg", &textured_jpeg(2400, 1800));
        let c = prep(&p, "armor", None, None, false, false);
        let (stego, _) =
            phasm_core::armor_encode_with_quality(&c.jpeg_bytes, "cli round trip", "pw").unwrap();
        let (payload, quality) = phasm_core::smart_decode(&stego, "pw").unwrap();
        assert_eq!(payload.text, "cli round trip");
        assert_eq!(quality.mode, 2); // Armor (Fortress is an Armor sub-mode)
        assert_eq!(jpeg_dims(&stego), (1600, 1200));
    }

    /// A textured RGB image as PNG bytes (`(png, rgb)`).
    fn textured_png(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
        let mut s: u32 = 0x0BAD_5EED;
        let img = RgbImage::from_fn(w, h, |x, y| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = (s >> 26) as u8;
            Rgb([(x * 90 / w) as u8 + n + 40, (y * 90 / h) as u8 + n + 30, n * 2 + 60])
        });
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(img.as_raw(), w, h, image::ExtendedColorType::Rgb8)
            .unwrap();
        (png, img.into_raw())
    }

    fn core_cover(original: &[u8], px: &[u8], w: u32, h: u32, req: CoverRequest) -> phasm_core::CoverOut {
        phasm_core::build_cover(original, Some((px, w, h)), &req).unwrap()
    }

    #[test]
    fn png_covers_are_build_covers_output_for_the_same_pixels() {
        let (png, px) = textured_png(1200, 900);
        let p = write("cover.png", &png);
        let base = CoverRequest {
            mode: CoverMode::Ghost,
            wand: false,
            ghost_quality: None,
            ghost_max_side: None,
            armor_preset: ArmorCoverPreset::Standard,
        };
        // Ghost: Deep Cover at IJG 92 with the pixels for SI.
        let c = prep(&p, "ghost", None, None, false, false);
        let want = core_cover(&png, &px, 1200, 900, base);
        assert_eq!(c.jpeg_bytes, want.jpeg);
        assert_eq!(c.raw_pixels.as_deref(), want.si_pixels_of(Some(&px)));
        assert_eq!(c.reencoded_qf, Some(92));
        // Ghost with --qf / --resize: the quality and resolution chips.
        let c = prep(&p, "ghost", Some(85), Some(800), false, false);
        let want = core_cover(&png, &px, 1200, 900, CoverRequest { ghost_quality: Some(85), ghost_max_side: Some(800), ..base });
        assert_eq!(c.jpeg_bytes, want.jpeg);
        assert_eq!((c.width, c.height), (800, 600));
        // Armor, every preset, with and without the wand.
        for preset in [ArmorCoverPreset::Standard, ArmorCoverPreset::Telegram, ArmorCoverPreset::Instagram] {
            for wand in [false, true] {
                let c = prepare_image(&p, "armor", None, None, wand, false, preset).unwrap_or_else(|e| panic!("{e}"));
                let want = core_cover(&png, &px, 1200, 900, CoverRequest { mode: CoverMode::Armor, wand, armor_preset: preset, ..base });
                assert_eq!(c.jpeg_bytes, want.jpeg, "{preset:?} wand {wand}");
                assert!(c.raw_pixels.is_none());
                assert_eq!(c.reencoded_qf, Some(65));
            }
        }
    }

    #[test]
    fn qf_never_changes_the_armor_cover() {
        let p = write("armor-qf.jpg", &textured_jpeg(2400, 1800));
        let plain = prep(&p, "armor", None, None, false, false);
        let with_qf = prep(&p, "armor", Some(95), None, false, false);
        assert_eq!(plain.jpeg_bytes, with_qf.jpeg_bytes);
        assert_eq!(with_qf.reencoded_qf, Some(65));
        // --resize / --full-res write at QF65 too (a JPEG over the size).
        let c = prep(&p, "armor", Some(95), Some(2000), false, false);
        assert_eq!(c.reencoded_qf, Some(65));
        assert_eq!((c.width, c.height), (2000, 1500));
    }

    #[test]
    fn armor_full_res_writes_a_non_jpeg_at_qf65() {
        let (png, _) = textured_png(900, 700);
        let p = write("fullres.png", &png);
        let c = prep(&p, "armor", None, None, false, true);
        assert_eq!((c.width, c.height), (900, 700));
        assert_eq!(c.reencoded_qf, Some(65));
        assert!(c.raw_pixels.is_none());
    }

    #[test]
    fn metadata_is_kept_without_optimize_and_dropped_with_it() {
        // A JPEG with EXIF (orientation only would not be "identifying": add
        // a camera model).
        let mut tiff = b"MM\x00\x2a\x00\x00\x00\x08\x00\x01".to_vec();
        tiff.extend_from_slice(&[0x01, 0x10, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, b'C', b'a', b'm', 0]);
        tiff.extend_from_slice(&[0, 0, 0, 0]);
        let mut payload = b"Exif\x00\x00".to_vec();
        payload.extend_from_slice(&tiff);
        let src = textured_jpeg(1800, 1200);
        let mut j = src[..2].to_vec();
        j.extend_from_slice(&[0xFF, 0xE1]);
        j.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        j.extend_from_slice(&payload);
        j.extend_from_slice(&src[2..]);
        let p = write("meta.jpg", &j);
        let kept = |c: &PreparedImage| crate::metadata::identifying_metadata(&c.jpeg_bytes);
        // Ghost pass-through: everything; Armor (written): the cleaned EXIF.
        assert_eq!(kept(&prep(&p, "ghost", None, None, false, false)), vec!["camera / lens make and model"]);
        assert_eq!(kept(&prep(&p, "armor", None, None, false, false)), vec!["camera / lens make and model"]);
        assert!(kept(&prep(&p, "ghost", None, None, true, false)).is_empty());
        assert!(kept(&prep(&p, "armor", None, None, true, false)).is_empty());
    }
}
