// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use image::codecs::jpeg::JpegEncoder;
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageEncoder, ImageFormat, ImageReader};
use phasm_core::{optimize_cover, OptimizerConfig, OptimizerMode, ARMOR_TARGET_DIMENSION};
use std::io::Cursor;
use std::path::Path;

/// Default JPEG quality factors when re-encoding is needed.
const DEFAULT_QF_GHOST: u8 = 92;
const DEFAULT_QF_ARMOR: u8 = 65;

/// Result of preparing an image for encoding.
pub struct PreparedImage {
    pub jpeg_bytes: Vec<u8>,
    /// Some when raw pixels are available → SI-UNIWARD for Ghost.
    pub raw_pixels: Option<Vec<u8>>,
    /// Dimensions of the cover that gets encoded.
    pub width: u32,
    pub height: u32,
    /// Dimensions of the input (after EXIF orientation).
    pub source_width: u32,
    pub source_height: u32,
    /// The quality the cover was re-encoded at; None when the JPEG passed
    /// through untouched.
    pub reencoded_qf: Option<u8>,
}

impl PreparedImage {
    /// An Armor cover larger than `ARMOR_TARGET_DIMENSION` still decodes, but
    /// does not survive recompression or resizing.
    pub fn exceeds_armor_target(&self) -> bool {
        self.width.max(self.height) > ARMOR_TARGET_DIMENSION
    }
}

/// Prepare an image for steganographic encoding.
///
/// - Armor (unless `armor_full_res`) → the cover the apps make: longest edge
///   at most `ARMOR_TARGET_DIMENSION`, re-encoded at QF65 even when no
///   resize is needed (the pre-settle). Armor survives recompression only
///   from that cover; at full resolution it decodes but is not robust.
/// - JPEG with no modifications → pass through (standard J-UNIWARD)
/// - JPEG + resize/optimize/qf → decode to RGB → process → JPEG compress → SI-UNIWARD
/// - Non-JPEG → decode to RGB → process → JPEG compress → SI-UNIWARD
pub fn prepare_image(
    path: &Path,
    mode: &str,
    qf: Option<u8>,
    resize: Option<u32>,
    optimize: bool,
    armor_full_res: bool,
) -> Result<PreparedImage, CliError> {
    let file_bytes = std::fs::read(path)?;
    let format = detect_format(path, &file_bytes)?;

    let resize = if mode == "armor" && !armor_full_res && resize.is_none() {
        Some(ARMOR_TARGET_DIMENSION)
    } else {
        resize
    };

    let is_jpeg = matches!(format, ImageFormat::Jpeg);
    let needs_processing = !is_jpeg || resize.is_some() || optimize || qf.is_some();

    if is_jpeg && !needs_processing {
        // Pass through — standard J-UNIWARD (no raw pixels)
        let (width, height) = ImageReader::with_format(Cursor::new(&file_bytes), format)
            .into_dimensions()
            .map_err(CliError::ImageLoad)?;
        return Ok(PreparedImage {
            jpeg_bytes: file_bytes,
            raw_pixels: None,
            width,
            height,
            source_width: width,
            source_height: height,
            reencoded_qf: None,
        });
    }

    // Decode to RGB pixels
    let mut img = load_image(&file_bytes, format)?;
    let (source_width, source_height) = (img.width(), img.height());

    // Resize if requested
    if let Some(max_dim) = resize {
        let (w, h) = (img.width(), img.height());
        if w > max_dim || h > max_dim {
            img = img.resize(max_dim, max_dim, image::imageops::FilterType::Lanczos3);
        }
    }

    let width = img.width();
    let height = img.height();
    let mut rgb_pixels = img.into_rgb8().into_raw();

    // Optimize if requested
    if optimize {
        let opt_mode = match mode {
            "ghost" => OptimizerMode::Ghost,
            _ => OptimizerMode::Armor,
        };
        let seed = [0u8; 32]; // deterministic for reproducibility
        rgb_pixels = optimize_cover(
            &rgb_pixels,
            width,
            height,
            &OptimizerConfig {
                strength: 0.85,
                seed,
                mode: opt_mode,
            },
        );
    }

    // JPEG compress
    let effective_qf = qf.unwrap_or(if mode == "ghost" {
        DEFAULT_QF_GHOST
    } else {
        DEFAULT_QF_ARMOR
    });
    let jpeg_bytes = compress_to_jpeg(&rgb_pixels, width, height, effective_qf)?;

    Ok(PreparedImage {
        jpeg_bytes,
        raw_pixels: Some(rgb_pixels),
        width,
        height,
        source_width,
        source_height,
        reencoded_qf: Some(effective_qf),
    })
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
    image::guess_format(bytes).map_err(|e| CliError::ImageLoad(e))
}

fn load_image(bytes: &[u8], format: ImageFormat) -> Result<DynamicImage, CliError> {
    let mut decoder = ImageReader::with_format(Cursor::new(bytes), format)
        .into_decoder()
        .map_err(CliError::ImageLoad)?;
    // Phone photos store their rotation as an EXIF tag, and re-encoding drops
    // it, so turn it into pixels here or the cover comes out sideways.
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).map_err(CliError::ImageLoad)?;
    img.apply_orientation(orientation);
    Ok(img)
}

fn compress_to_jpeg(rgb: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>, CliError> {
    let mut buf = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut buf, quality);
    encoder
        .write_image(rgb, width, height, image::ExtendedColorType::Rgb8)
        .map_err(|e| CliError::ImageLoad(e))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, Rgb, RgbImage};
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
        compress_to_jpeg(img.as_raw(), w, h, 90).unwrap_or_else(|e| panic!("{e}"))
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
        prepare_image(p, mode, qf, resize, optimize, full_res).unwrap_or_else(|e| panic!("{e}"))
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
}
