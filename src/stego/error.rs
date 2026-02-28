// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Error types for the steganography pipeline.
//!
//! [`StegoError`] covers all failure modes from JPEG parsing through
//! encryption and frame extraction.

use core::fmt;

/// Errors that can occur during steganographic encoding or decoding.
#[derive(Debug)]
pub enum StegoError {
    /// The cover image could not be parsed as a valid JPEG.
    InvalidJpeg(crate::jpeg::error::JpegError),
    /// The image is too small or has too few usable coefficients.
    ImageTooSmall,
    /// The image dimensions exceed the maximum allowed (8192px / 16MP).
    ImageTooLarge,
    /// The message is too large for the cover image's embedding capacity.
    MessageTooLarge,
    /// CRC check failed on the extracted payload frame.
    FrameCorrupted,
    /// AES-GCM decryption failed (wrong passphrase or corrupted data).
    DecryptionFailed,
    /// The extracted plaintext is not valid UTF-8.
    InvalidUtf8,
    /// The cover image has no luminance component.
    NoLuminanceChannel,
    /// The operation was cancelled by the user.
    Cancelled,
}

impl fmt::Display for StegoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJpeg(e) => write!(f, "invalid JPEG: {e}"),
            Self::ImageTooSmall => write!(f, "image too small for embedding"),
            Self::ImageTooLarge => write!(f, "image too large (max 8192px / 16MP)"),
            Self::MessageTooLarge => write!(f, "message too large for this image"),
            Self::FrameCorrupted => write!(f, "payload frame CRC mismatch"),
            Self::DecryptionFailed => write!(f, "decryption failed (wrong passphrase?)"),
            Self::InvalidUtf8 => write!(f, "extracted text is not valid UTF-8"),
            Self::NoLuminanceChannel => write!(f, "image has no luminance channel"),
            Self::Cancelled => write!(f, "operation cancelled by user"),
        }
    }
}

impl std::error::Error for StegoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidJpeg(e) => Some(e),
            _ => None,
        }
    }
}

impl From<crate::jpeg::error::JpegError> for StegoError {
    fn from(e: crate::jpeg::error::JpegError) -> Self {
        Self::InvalidJpeg(e)
    }
}
