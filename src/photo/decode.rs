// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AVIF -> interleaved RGB through dav1d: photo attachments shown (Android, which has no AVIF
//! decoder before Android 12; iOS decodes AVIF natively), and AVIF covers (the built-in and the
//! large sample photos, any AVIF the user picks) decoded the same way on every platform, so the
//! cover pixels and therefore the Deep Cover cover are identical everywhere
//! (`docs/design/app/large-samples.md`, "Decoding").
//!
//! dav1d applies AV1 film grain itself (bit-exact with its C reference on every SIMD path), and its
//! output does not depend on the thread count, so the pixels depend only on the file.

use super::avif::{first_frame, read_avif, AvifImage, Nclx};
use super::yuv::{yuv_to_rgb, Layout};
use super::PhotoError;
use crate::codec::av1::stego::decoder::{decode_still_view, plain_decode_threads, Av1DecodeError};

/// The largest image shown (an attachment is someone else's file: no decompression bombs).
/// Photo attachments are at most 1280 px (fit ladder); 16 MP still shows camera photos sent as files.
pub const MAX_DECODE_PIXELS: u64 = 16_777_216;

/// A decoded image: interleaved 8-bit RGB, `width` x `height`.
#[derive(Debug)]
pub struct DecodedPhoto {
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Decode the primary image of an AVIF file for display (photo attachments): at most
/// [`MAX_DECODE_PIXELS`], colour converted by the matrix the file names, transforms and ICC
/// profiles not applied.
pub fn decode_avif(file: &[u8]) -> Result<DecodedPhoto, PhotoError> {
    let img = read_avif(file)?;
    decode_image(&img, MAX_DECODE_PIXELS)
}

/// The size [`decode_avif_rgb`] would decode, without decoding: the `ispe` of the primary item.
/// The same refusals as [`decode_avif_rgb`] (except the pixel cap), so a caller can check memory
/// first and send the files core refuses to the platform decoder.
pub fn avif_rgb_size(file: &[u8]) -> Result<(u32, u32), PhotoError> {
    let img = read_avif(file)?;
    check_exact_srgb(&img)?;
    Ok((img.width, img.height))
}

/// Decode an AVIF file to upright 8-bit sRGB pixels for a cover, at most `max_pixels` (the apps
/// pass core's [`crate::MAX_PIXELS`], 200 MP). Only files whose stored picture is the picture as
/// shown, in sRGB, are decoded: a grid, an alpha item, a transform (`irot` /
/// `imir` / `clap`), an ICC profile, primaries or a transfer other than sRGB / BT.709 (or
/// unspecified), an unusual matrix, or more than 8 bits are [`PhotoError::UnsupportedAvif`], and the
/// apps decode those with the platform (which applies transforms and colour management). Over
/// `max_pixels`: [`PhotoError::TooLarge`], checked on the container's size before anything is
/// allocated and again by dav1d (`frame_size_limit`).
pub fn decode_avif_rgb(file: &[u8], max_pixels: u64) -> Result<DecodedPhoto, PhotoError> {
    let img = read_avif(file)?;
    check_exact_srgb(&img)?;
    decode_image(&img, max_pixels)
}

/// The refusals of [`decode_avif_rgb`]: anything whose decoded samples are not the sRGB picture
/// as shown.
fn check_exact_srgb(img: &AvifImage) -> Result<(), PhotoError> {
    if img.transformed {
        return Err(PhotoError::UnsupportedAvif("transformed image (irot / imir / clap)"));
    }
    if img.icc {
        return Err(PhotoError::UnsupportedAvif("ICC colour profile"));
    }
    if img.high_bit_depth {
        return Err(PhotoError::UnsupportedAvif("more than 8 bits per sample"));
    }
    if let Some(c) = img.nclx {
        // primaries BT.709 (= sRGB) or unspecified; transfer sRGB or unspecified; matrices the
        // converter implements exactly (identity, BT.709, unspecified, BT.470BG, BT.601)
        let primaries = matches!(c.primaries, 1 | 2);
        let transfer = matches!(c.transfer, 2 | 13);
        let matrix = matches!(c.matrix, 0 | 1 | 2 | 5 | 6);
        if !(primaries && transfer && matrix) {
            return Err(PhotoError::UnsupportedAvif("colour other than sRGB"));
        }
    }
    Ok(())
}

fn decode_image(img: &AvifImage, max_pixels: u64) -> Result<DecodedPhoto, PhotoError> {
    if img.width as u64 * img.height as u64 > max_pixels {
        return Err(PhotoError::TooLarge);
    }
    // one picture: whatever follows the first frame is not decoded
    let av1 = first_frame(&img.av1)?;
    // without a colour box: the AV1 defaults (unspecified matrix as BT.601, studio range)
    let nclx = img.nclx.unwrap_or(Nclx { primaries: 2, transfer: 2, matrix: 6, full_range: false });
    let limit = max_pixels.min(u32::MAX as u64) as u32;
    // YUV -> RGB straight from dav1d's picture: no copy of the planes (peak about 6 B/px at
    // 4:4:4 instead of 9)
    let decoded = decode_still_view(av1, limit, plain_decode_threads(), |p| {
        let layout = match p.layout {
            0 => Layout::Mono,
            1 => Layout::I420,
            2 => Layout::I422,
            _ => Layout::I444,
        };
        let rgb = yuv_to_rgb(layout, p.width, p.height, p.y, p.y_stride, p.u, p.v, p.c_stride, nclx.matrix, nclx.full_range);
        DecodedPhoto { rgb, width: p.width as u32, height: p.height as u32 }
    })
    .map_err(|e| match e {
        Av1DecodeError::UnsupportedPicture { layout: -2, .. } => PhotoError::TooLarge,
        e => PhotoError::Decode(format!("{e:?}")),
    })?;
    if decoded.width as u64 * decoded.height as u64 > max_pixels {
        return Err(PhotoError::TooLarge);
    }
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photo::avif::{write_avif, NCLX_SRGB_BT601_FULL};
    use sha2::{Digest, Sha256};

    /// 320 x 240 crop of the built-in grassy-field sample, avifenc 1.3.0 / libaom 3.13.3, q60,
    /// 4:4:4, 8-bit, full range, BT.601 matrix.
    const GRASS: &[u8] = include_bytes!("../../test-vectors/image/grass_320x240_q60_444.avif");

    fn hex(d: impl AsRef<[u8]>) -> String {
        d.as_ref().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// SHA-256 of each plane's visible samples, read straight from the view.
    fn plane_hashes(p: &crate::codec::av1::stego::decoder::StillView<'_>) -> [String; 3] {
        let (cw, ch) = match p.layout {
            1 => (p.width.div_ceil(2), p.height.div_ceil(2)),
            2 => (p.width.div_ceil(2), p.height),
            _ => (p.width, p.height),
        };
        let hash = |plane: &[u8], stride: usize, w: usize, h: usize| {
            let mut s = Sha256::new();
            for r in 0..h {
                s.update(&plane[r * stride..r * stride + w]);
            }
            hex(s.finalize())
        };
        [
            hash(p.y, p.y_stride, p.width, p.height),
            hash(p.u, p.c_stride, cw, ch),
            hash(p.v, p.c_stride, cw, ch),
        ]
    }

    #[test]
    fn planes_match_upstream_dav1d_at_every_thread_count() {
        // the reference: avifdec 1.3.0 (dav1d 1.5.3) `--depth 8` to y4m, plane by plane
        let reference = [
            "8cee28641fc50c49887e2a911a45906e29e93452c332535107b773d517ff179e",
            "218d3c07a7c23083c88885b8afbbcad310e6f3f3ff89fd0230891667360632cf",
            "1deabc0162c81517b86e6054f530b7bb31aef499336068c4aa11966cfc94c6de",
        ];
        let img = read_avif(GRASS).unwrap();
        let av1 = first_frame(&img.av1).unwrap();
        for threads in [1, 2, 4, 8] {
            let hashes = decode_still_view(av1, 1 << 20, threads, |p| {
                assert_eq!((p.layout, p.width, p.height), (3, 320, 240));
                plane_hashes(p)
            })
            .unwrap();
            assert_eq!(hashes, reference, "{threads} threads");
        }
    }

    #[test]
    fn cover_decode_of_a_444_file() {
        assert_eq!(avif_rgb_size(GRASS).unwrap(), (320, 240));
        let d = decode_avif_rgb(GRASS, crate::MAX_PIXELS as u64).unwrap();
        assert_eq!((d.width, d.height, d.rgb.len()), (320, 240, 320 * 240 * 3));
        // pinned: integer conversion, the same bytes on every target. The same math on avifdec's
        // planes gives this hash, and libavif's own conversion is within 1 of it (99.96 % equal).
        assert_eq!(hex(Sha256::digest(&d.rgb)), "311f3671b403f8e89d4d4f5c7ea982f521b57530fa993d956e7c7eea15872417");
        // the attachment decode is the same conversion
        assert_eq!(decode_avif(GRASS).unwrap().rgb, d.rgb);
        // the cap counts pixels, before anything is decoded
        assert_eq!(decode_avif_rgb(GRASS, 320 * 240 - 1).unwrap_err(), PhotoError::TooLarge);
        assert!(decode_avif_rgb(GRASS, 320 * 240).is_ok());
    }

    #[test]
    fn cover_decode_refuses_what_it_cannot_show_exactly() {
        // a stub AV1 stream: the refusals happen on the container, before any decode
        let av1 = [0x12, 0x00, 0x0a, 0x03, 0b0001_1000, 0x00, 0x00, 0x32, 0x02, 0xaa, 0xbb];
        let plain = write_avif(&av1, 64, 48, NCLX_SRGB_BT601_FULL).unwrap();
        assert_eq!(avif_rgb_size(&plain).unwrap(), (64, 48));
        for nclx in [
            Nclx { primaries: 9, ..NCLX_SRGB_BT601_FULL },  // BT.2020
            Nclx { primaries: 12, ..NCLX_SRGB_BT601_FULL }, // Display P3
            Nclx { transfer: 16, ..NCLX_SRGB_BT601_FULL },  // PQ
            Nclx { transfer: 18, ..NCLX_SRGB_BT601_FULL },  // HLG
            Nclx { matrix: 8, ..NCLX_SRGB_BT601_FULL },     // YCgCo
        ] {
            let file = write_avif(&av1, 64, 48, nclx).unwrap();
            assert!(matches!(avif_rgb_size(&file), Err(PhotoError::UnsupportedAvif(_))), "{nclx:?}");
            assert!(matches!(decode_avif_rgb(&file, 1 << 20), Err(PhotoError::UnsupportedAvif(_))), "{nclx:?}");
        }
        for ok in [
            Nclx { primaries: 2, transfer: 2, matrix: 2, full_range: false },
            Nclx { primaries: 1, transfer: 13, matrix: 1, full_range: true },
            Nclx { primaries: 1, transfer: 13, matrix: 0, full_range: true },
        ] {
            assert!(avif_rgb_size(&write_avif(&av1, 64, 48, ok).unwrap()).is_ok(), "{ok:?}");
        }
        let mut deep = plain.clone();
        let at = deep.windows(4).position(|w| w == b"av1C").unwrap() + 4;
        deep[at + 2] |= 0x40;
        assert!(matches!(avif_rgb_size(&deep), Err(PhotoError::UnsupportedAvif(_))));
    }

    /// The large sample photos (single-item AVIF up to 150 MP, 4:4:4): core's dav1d decodes them,
    /// with the plane hashes printed for a comparison with avifdec, and the time.
    /// `PHASM_LARGE_AVIF=a.avif:b.avif cargo test --release --features av1-decoder
    /// large_avif_files -- --ignored --nocapture`
    #[test]
    #[ignore = "needs PHASM_LARGE_AVIF"]
    fn large_avif_files() {
        let paths = std::env::var("PHASM_LARGE_AVIF").expect("PHASM_LARGE_AVIF=<file>[:<file>...]");
        for path in paths.split(':') {
            let file = std::fs::read(path).unwrap();
            let img = read_avif(&file).unwrap();
            let av1 = first_frame(&img.av1).unwrap();
            let t = std::time::Instant::now();
            let hashes = decode_still_view(av1, crate::MAX_PIXELS, plain_decode_threads(), |p| plane_hashes(p)).unwrap();
            let planes = t.elapsed();
            let t = std::time::Instant::now();
            let d = decode_avif_rgb(&file, crate::MAX_PIXELS as u64).unwrap();
            let rgb = t.elapsed();
            eprintln!(
                "{path}: {}x{} threads {} planes {hashes:?} decode+hash {planes:?}, decode_avif_rgb {rgb:?}, rgb {}",
                d.width,
                d.height,
                plain_decode_threads(),
                hex(Sha256::digest(&d.rgb))
            );
        }
    }
}
