// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use phasm_core::{JpegError, StegoError};
use std::fmt;

/// Exit codes for the CLI.
pub const EXIT_OK: i32 = 0;
pub const EXIT_GENERAL: i32 = 1;
pub const EXIT_STEGO: i32 = 2;
pub const EXIT_WRONG_PASSPHRASE: i32 = 3;
pub const EXIT_MESSAGE_TOO_LARGE: i32 = 4;
pub const EXIT_UNSUPPORTED_FORMAT: i32 = 5;

/// CLI error type wrapping all possible failure modes.
pub enum CliError {
    Io(std::io::Error),
    Stego(StegoError),
    ImageLoad(image::ImageError),
    UnsupportedFormat(String),
    InvalidArgs(String),
    NoMessage,
    PassphraseMismatch,
}

impl CliError {
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::Io(_) => EXIT_GENERAL,
            CliError::Stego(StegoError::DecryptionFailed) => EXIT_WRONG_PASSPHRASE,
            CliError::Stego(StegoError::MessageTooLarge) => EXIT_MESSAGE_TOO_LARGE,
            CliError::Stego(_) => EXIT_STEGO,
            CliError::ImageLoad(_) => EXIT_GENERAL,
            CliError::UnsupportedFormat(_) => EXIT_UNSUPPORTED_FORMAT,
            CliError::InvalidArgs(_) => EXIT_GENERAL,
            CliError::NoMessage => EXIT_GENERAL,
            CliError::PassphraseMismatch => EXIT_GENERAL,
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::Io(e) => write!(f, "I/O error: {e}"),
            CliError::Stego(StegoError::DecryptionFailed) => {
                write!(f, "Decryption failed — wrong passphrase or not a stego image")
            }
            CliError::Stego(StegoError::MessageTooLarge) => {
                write!(f, "Message too large for this image")
            }
            CliError::Stego(StegoError::ImageTooSmall) => {
                write!(f, "Image too small for steganography")
            }
            CliError::Stego(StegoError::ImageTooLarge) => {
                write!(f, "Image exceeds maximum supported dimensions")
            }
            CliError::Stego(StegoError::Cancelled) => write!(f, "Operation cancelled"),
            CliError::Stego(e) => write!(f, "Stego error: {e:?}"),
            CliError::ImageLoad(e) => write!(f, "Failed to load image: {e}"),
            CliError::UnsupportedFormat(ext) => {
                write!(f, "Unsupported image format: {ext}. Convert to JPEG/PNG/WebP/BMP/TIFF/GIF first.")
            }
            CliError::InvalidArgs(msg) => write!(f, "{msg}"),
            CliError::NoMessage => {
                write!(f, "No message provided. Use -m <text> or pipe via stdin.")
            }
            CliError::PassphraseMismatch => write!(f, "Passphrases do not match"),
        }
    }
}

impl From<std::io::Error> for CliError {
    fn from(e: std::io::Error) -> Self {
        CliError::Io(e)
    }
}

impl From<StegoError> for CliError {
    fn from(e: StegoError) -> Self {
        CliError::Stego(e)
    }
}

impl From<image::ImageError> for CliError {
    fn from(e: image::ImageError) -> Self {
        CliError::ImageLoad(e)
    }
}

impl From<JpegError> for CliError {
    fn from(e: JpegError) -> Self {
        CliError::Stego(StegoError::InvalidJpeg(e))
    }
}
