// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageEncoder, ImageFormat, ImageReader};
use phasm_core::{optimize_cover, OptimizerConfig, OptimizerMode};
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
    pub width: u32,
    pub height: u32,
}

/// Prepare an image for steganographic encoding.
///
/// - JPEG with no modifications → pass through (standard J-UNIWARD)
/// - JPEG + resize/optimize/qf → decode to RGB → process → JPEG compress → SI-UNIWARD
/// - Non-JPEG → decode to RGB → process → JPEG compress → SI-UNIWARD
pub fn prepare_image(
    path: &Path,
    mode: &str,
    qf: Option<u8>,
    resize: Option<u32>,
    optimize: bool,
) -> Result<PreparedImage, CliError> {
    let file_bytes = std::fs::read(path)?;
    let format = detect_format(path, &file_bytes)?;

    let is_jpeg = matches!(format, ImageFormat::Jpeg);
    let needs_processing = !is_jpeg || resize.is_some() || optimize || qf.is_some();

    if is_jpeg && !needs_processing {
        // Pass through — standard J-UNIWARD (no raw pixels)
        return Ok(PreparedImage {
            jpeg_bytes: file_bytes,
            raw_pixels: None,
            width: 0,
            height: 0,
        });
    }

    // Decode to RGB pixels
    let mut img = load_image(&file_bytes, format)?;

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
    let reader = ImageReader::with_format(Cursor::new(bytes), format);
    reader.decode().map_err(CliError::ImageLoad)
}

fn compress_to_jpeg(rgb: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>, CliError> {
    let mut buf = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut buf, quality);
    encoder
        .write_image(rgb, width, height, image::ExtendedColorType::Rgb8)
        .map_err(|e| CliError::ImageLoad(e))?;
    Ok(buf)
}
