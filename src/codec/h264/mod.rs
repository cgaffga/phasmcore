// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264/AVC bitstream parsing.
//!
//! Parses NAL units, parameter sets (SPS/PPS), slice headers, and CAVLC-coded
//! macroblock residual data. Designed for steganography: tracks exact bit
//! positions of embeddable trailing-one sign bits in the raw byte stream.

pub mod bitstream;
pub mod tables;
pub mod sps;
pub mod slice;
pub mod cavlc;
#[cfg(feature = "h264-encoder")]
pub mod cavlc_writer;
#[cfg(feature = "h264-encoder")]
pub mod cavlc_size;
pub mod macroblock;
pub mod transform;
pub mod intra_pred;
pub mod intra_pred_8x8;
pub mod reconstruct;
pub mod mv;
pub mod fingerprint;

// Phase 6 — pure-Rust H.264 encoder + its dedicated CABAC entropy coder.
// Gated behind the `h264-encoder` Cargo feature (OFF by default) so the
// GitHub-release CLI binary never accidentally bundles an encoder
// subject to Via LA AVC patent obligations. Mobile bridges enable the
// feature explicitly. See
// `docs/design/video/h264/video-steganography.md` § Phase 6.
#[cfg(feature = "h264-encoder")]
pub mod cabac;
#[cfg(feature = "h264-encoder")]
pub mod encoder;
// Phase 6F.1 — encode-time stego primitives previously lived under
// `encoder/stego`. The decoder's bin walker also consumes them
// (`crate::codec::h264::stego::hook::PositionKey` etc.), and the
// encoder struct embeds an `Option<Box<dyn StegoMbHook>>` field
// for hook plumbing — so this module must compile whenever the
// `h264-encoder` feature is on. The PUBLIC stego API in `lib.rs`
// stays gated by `cabac-stego` (consumer-facing intent unchanged).
//
// Phase 6F.4 (cross-platform smoke 2026-04-30) — gate widened from
// `cabac-stego` to `h264-encoder` after iOS/Android bridges
// (`features = ["parallel", "video", "h264-encoder"]`) failed to
// compile post-6F.1 because the bin_decoder + encoder.rs reference
// `stego::*` types unconditionally.
#[cfg(feature = "h264-encoder")]
pub mod stego;

/// High-level Rust API for the cgaffga/phasm-openh264 fork's stego
/// hooks. Gated by the `openh264-backend` Cargo feature (Phase B);
/// pulls in `core-openh264-sys` for the raw FFI bindings.
#[cfg(feature = "openh264-backend")]
pub mod openh264;

/// Per-MB decision cache for the Pass-2 replay architecture (#533).
/// Pass-1 streams decisions into the cache via `StegoSession`'s
/// `capture_mb_decision` closure; Pass-2 fetches them via
/// `replay_mb_decision`. Scope is per-GOP — caller drops the cache
/// at GOP boundaries to keep memory bounded on long clips.
#[cfg(feature = "openh264-backend")]
pub mod pass2_cache;

/// Phase C.8.13 — production stego orchestrator on top of the OpenH264
/// backend. Single-domain (CoeffSign) STC encode + brute-force decode
/// over walker-aligned cover, with passphrase-derived seeds. Relies on
/// the C.8.3-11 dual-recon cascade-break to keep mode-decision stable
/// across baseline ↔ stego encodes.
#[cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]
pub mod openh264_stego;

/// D.0.7 — streaming H.264 stego session API. Engine-agnostic surface
/// that mobile bridges expose via FFI; internally dispatches to the
/// OH264 backend or the pure-Rust encoder based on `EncodeEngineChoice`.
/// Per-GOP STC, bounded memory, arbitrary clip length. Design memo
/// at `docs/design/video/h264/d07-streaming-sessions.md`.
#[cfg(feature = "h264-encoder")]
pub mod streaming_session;

/// #474 — engine-agnostic encode + decode progress event vocabulary.
/// Optional callback on the streaming session params; mobile bridges
/// marshal events across FFI into per-platform smoothing engines that
/// drive the HUD progress bar + ETA text. Design memo at
/// `docs/design/video/h264/progress-indicator.md`.
#[cfg(feature = "h264-encoder")]
pub mod progress;

use std::fmt;

/// H.264/AVC parsing error.
#[derive(Debug, Clone)]
pub enum H264Error {
    /// Unexpected end of bitstream data.
    UnexpectedEof,
    /// Invalid NAL unit header.
    InvalidNalHeader,
    /// Invalid or malformed parameter set.
    InvalidParameterSet(String),
    /// H.264 feature not supported for steganography.
    Unsupported(String),
    /// CAVLC decoding failure.
    CavlcError(String),
}

impl fmt::Display for H264Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of H.264 data"),
            Self::InvalidNalHeader => write!(f, "invalid NAL unit header"),
            Self::InvalidParameterSet(s) => write!(f, "invalid parameter set: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported H.264 feature: {s}"),
            Self::CavlcError(s) => write!(f, "CAVLC decode error: {s}"),
        }
    }
}

impl std::error::Error for H264Error {}

/// H.264 NAL unit type (ITU-T H.264 Table 7-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NalType(pub u8);

impl NalType {
    /// Coded slice of a non-IDR picture.
    pub const SLICE: Self = Self(1);
    /// Coded slice data partition A.
    pub const SLICE_DPA: Self = Self(2);
    /// Coded slice data partition B.
    pub const SLICE_DPB: Self = Self(3);
    /// Coded slice data partition C.
    pub const SLICE_DPC: Self = Self(4);
    /// Coded slice of an IDR picture.
    pub const SLICE_IDR: Self = Self(5);
    /// Supplemental enhancement information.
    pub const SEI: Self = Self(6);
    /// Sequence parameter set.
    pub const SPS: Self = Self(7);
    /// Picture parameter set.
    pub const PPS: Self = Self(8);
    /// Access unit delimiter.
    pub const AUD: Self = Self(9);
    /// End of sequence.
    pub const END_SEQ: Self = Self(10);
    /// End of stream.
    pub const END_STREAM: Self = Self(11);
    /// Filler data.
    pub const FILLER: Self = Self(12);
    /// SPS extension.
    pub const SPS_EXT: Self = Self(13);

    /// True for VCL NAL types (contain coded slice data).
    pub fn is_vcl(self) -> bool {
        self.0 >= 1 && self.0 <= 5
    }

    /// True for IDR (Instantaneous Decoder Refresh) slices.
    pub fn is_idr(self) -> bool {
        self.0 == 5
    }

    /// True for any slice type (non-IDR or IDR).
    pub fn is_slice(self) -> bool {
        self.0 >= 1 && self.0 <= 5
    }
}

impl fmt::Display for NalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self.0 {
            1 => "SLICE",
            2 => "SLICE_DPA",
            3 => "SLICE_DPB",
            4 => "SLICE_DPC",
            5 => "IDR",
            6 => "SEI",
            7 => "SPS",
            8 => "PPS",
            9 => "AUD",
            10 => "END_SEQ",
            11 => "END_STREAM",
            12 => "FILLER",
            13 => "SPS_EXT",
            n => return write!(f, "NAL({n})"),
        };
        write!(f, "{name}")
    }
}

/// Parsed H.264 NAL unit.
#[derive(Debug, Clone)]
pub struct NalUnit {
    /// NAL unit type.
    pub nal_type: NalType,
    /// nal_ref_idc: 0 = non-reference, 1-3 = reference priority.
    pub nal_ref_idc: u8,
    /// Raw Byte Sequence Payload (emulation prevention bytes removed).
    pub rbsp: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nal_type_is_vcl() {
        assert!(!NalType(0).is_vcl());
        assert!(NalType::SLICE.is_vcl());
        assert!(NalType::SLICE_IDR.is_vcl());
        assert!(!NalType::SPS.is_vcl());
        assert!(!NalType::PPS.is_vcl());
        assert!(!NalType::SEI.is_vcl());
    }

    #[test]
    fn nal_type_is_idr() {
        assert!(NalType::SLICE_IDR.is_idr());
        assert!(!NalType::SLICE.is_idr());
        assert!(!NalType::SPS.is_idr());
    }

    #[test]
    fn nal_type_display() {
        assert_eq!(format!("{}", NalType::SPS), "SPS");
        assert_eq!(format!("{}", NalType::SLICE_IDR), "IDR");
        assert_eq!(format!("{}", NalType(99)), "NAL(99)");
    }

    #[test]
    fn error_display() {
        let e = H264Error::UnexpectedEof;
        assert!(format!("{e}").contains("unexpected"));
        let e = H264Error::CavlcError("bad block".into());
        assert!(format!("{e}").contains("bad block"));
    }
}
