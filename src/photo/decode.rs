// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AVIF -> interleaved RGB through dav1d, for showing a photo attachment (Android, which has no
//! AVIF decoder before Android 12; iOS decodes AVIF natively).

use super::avif::{first_frame, read_avif, Nclx};
use super::yuv::{yuv_to_rgb, Layout};
use super::PhotoError;
use crate::codec::av1::stego::decoder::decode_still_planes;

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

/// Decode the primary image of an AVIF file.
pub fn decode_avif(file: &[u8]) -> Result<DecodedPhoto, PhotoError> {
    let img = read_avif(file)?;
    if img.width as u64 * img.height as u64 > MAX_DECODE_PIXELS {
        return Err(PhotoError::TooLarge);
    }
    // one picture: whatever follows the first frame is not decoded
    let av1 = first_frame(&img.av1)?;
    let p = decode_still_planes(av1, MAX_DECODE_PIXELS as u32).map_err(|e| match e {
        crate::codec::av1::stego::decoder::Av1DecodeError::UnsupportedPicture { layout: -2, .. } => PhotoError::TooLarge,
        e => PhotoError::Decode(format!("{e:?}")),
    })?;
    if p.width as u64 * p.height as u64 > MAX_DECODE_PIXELS {
        return Err(PhotoError::TooLarge);
    }
    let layout = match p.layout {
        0 => Layout::Mono,
        1 => Layout::I420,
        2 => Layout::I422,
        _ => Layout::I444,
    };
    // without a colour box: the AV1 defaults (unspecified matrix as BT.601, studio range)
    let nclx = img.nclx.unwrap_or(Nclx { primaries: 2, transfer: 2, matrix: 6, full_range: false });
    let cw = match layout {
        Layout::I420 | Layout::I422 => p.width.div_ceil(2),
        _ => p.width,
    };
    let rgb = yuv_to_rgb(layout, p.width, p.height, &p.y, p.width, &p.u, &p.v, cw, nclx.matrix, nclx.full_range);
    Ok(DecodedPhoto { rgb, width: p.width as u32, height: p.height as u32 })
}
