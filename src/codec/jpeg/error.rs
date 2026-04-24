// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Error types for JPEG parsing and encoding.

use std::fmt;

/// Errors that can occur during JPEG parsing or encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JpegError {
    /// Input data is too short or truncated.
    UnexpectedEof,
    /// Missing SOI (0xFFD8) at start of data.
    InvalidSoi,
    /// Missing EOI (0xFFD9) — non-fatal for some files.
    MissingEoi,
    /// Encountered an unsupported JPEG marker (progressive, arithmetic, etc.).
    UnsupportedMarker(u8),
    /// A marker segment has invalid or inconsistent length/content.
    InvalidMarkerData(&'static str),
    /// Huffman decode error (invalid code encountered in scan data).
    HuffmanDecode,
    /// Quantization table ID out of range (0–3).
    InvalidQuantTableId(u8),
    /// Huffman table ID out of range or missing.
    InvalidHuffmanTableId(u8),
    /// Component ID referenced in SOS not found in SOF.
    UnknownComponentId(u8),
    /// Image dimensions or sampling factors are invalid.
    InvalidDimensions,
    /// 12-bit precision is not supported.
    UnsupportedPrecision(u8),
}

impl fmt::Display for JpegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of JPEG data"),
            Self::InvalidSoi => write!(f, "missing SOI marker (not a JPEG)"),
            Self::MissingEoi => write!(f, "missing EOI marker"),
            Self::UnsupportedMarker(m) => write!(f, "unsupported JPEG marker: 0xFF{m:02X}"),
            Self::InvalidMarkerData(msg) => write!(f, "invalid marker data: {msg}"),
            Self::HuffmanDecode => write!(f, "Huffman decode error"),
            Self::InvalidQuantTableId(id) => write!(f, "invalid quantization table ID: {id}"),
            Self::InvalidHuffmanTableId(id) => write!(f, "invalid Huffman table ID: {id}"),
            Self::UnknownComponentId(id) => write!(f, "unknown component ID in SOS: {id}"),
            Self::InvalidDimensions => write!(f, "invalid image dimensions or sampling factors"),
            Self::UnsupportedPrecision(p) => write!(f, "unsupported sample precision: {p}-bit"),
        }
    }
}

impl std::error::Error for JpegError {}

pub type Result<T> = std::result::Result<T, JpegError>;
