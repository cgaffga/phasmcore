// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Photo attachments ("photo in photo", Ghost): a photo shrunk and AV1-compressed into an AVIF
//! file that fits what is left of a cover's budget, and AVIF files decoded back to RGB for display.
//!
//! The rules (resolution ladder, quality floor, encoder setup) come from the 2026-11-07 measurement:
//! `docs/design/app/photo-attachments.md`, `eval/runs/2026-11-07-photo-attach/`. The attachment is a
//! plain file entry of the payload (`photo-1.avif`), so the wire format does not change and every
//! decoder that saves attachments can open it.
//!
//! - [`avif`]: the container, written and read (always built).
//! - [`yuv`]: RGB <-> YCbCr (always built).
//! - [`fit`]: the fitting search over libaom (`photo-encoder`).
//! - [`decode`]: AVIF -> RGB through dav1d (`av1-decoder`).

pub mod avif;
pub mod yuv;
#[cfg(feature = "photo-encoder")]
pub mod fit;
#[cfg(feature = "av1-decoder")]
pub mod decode;

/// Why a photo could not be fitted or an AVIF file not shown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhotoError {
    /// The pixel buffer does not match the size, or a side is 0.
    InvalidImage,
    /// Not even the smallest size at the lowest quality fits the budget.
    DoesNotFit,
    /// The caller's progress callback asked to stop.
    Cancelled,
    /// libaom failed (the shim's code).
    Encoder(i32),
    /// Not an AVIF file this module reads (the apps then show it as a plain file).
    UnsupportedAvif(&'static str),
    /// The image is larger than a photo attachment is allowed to be when shown.
    TooLarge,
    /// dav1d could not decode the image.
    Decode(String),
}

impl core::fmt::Display for PhotoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidImage => write!(f, "invalid image buffer"),
            Self::DoesNotFit => write!(f, "the photo does not fit, even at the smallest size"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Encoder(c) => write!(f, "AV1 encoder failed ({c})"),
            Self::UnsupportedAvif(why) => write!(f, "unsupported AVIF: {why}"),
            Self::TooLarge => write!(f, "image too large to show"),
            Self::Decode(e) => write!(f, "AV1 decode failed: {e}"),
        }
    }
}

impl std::error::Error for PhotoError {}
