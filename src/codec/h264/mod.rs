// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264/AVC bitstream parsing.
//!
//! Parses NAL units, parameter sets (SPS/PPS), and slice headers. The CABAC
//! bin-walker ([`cabac`]) handles High-profile residual + MV decode for stego.
//! The legacy CAVLC parser cluster (`cavlc` / `macroblock` / `mv` /
//! `reconstruct` / `intra_pred` / `intra_pred_8x8` / `fingerprint`) was removed
//! with the CAVLC bitstream-mod stego subsystem — see
//! `docs/design/video/_RETIREMENT-PLAN.md` § "Phase 4".

pub mod bitstream;
pub mod tables;
pub mod sps;
pub mod slice;
pub mod transform;
pub mod b_partition_meta;

// H.264 CABAC entropy coder: the bin-walker (decode) + shared tables.
#[cfg(feature = "h264-decoder")]
pub mod cabac;
// Encode-time stego primitives. The decoder's bin walker also consumes
// the shared position/hook types (`stego::hook::PositionKey` etc.), so
// this module compiles whenever the decode feature is on.
#[cfg(feature = "h264-decoder")]
pub mod stego;

/// High-level Rust API for the cgaffga/phasm-openh264 fork's stego
/// hooks. Gated by the `h264-encoder` Cargo feature; pulls in
/// `core-openh264-sys` for the raw FFI bindings.
#[cfg(feature = "h264-encoder")]
pub mod openh264;

/// Per-MB decision cache for the Pass-2 replay architecture (#533).
/// Pass-1 streams decisions into the cache via `StegoSession`'s
/// `capture_mb_decision` closure; Pass-2 fetches them via
/// `replay_mb_decision`. Scope is per-GOP — caller drops the cache
/// at GOP boundaries to keep memory bounded on long clips.
#[cfg(feature = "h264-encoder")]
pub mod pass2_cache;

/// Production stego ENCODE orchestrator on top of the OpenH264 backend.
/// The live primitive is the 2-pass `h264_encode_gop_framed_bits_auto
/// [_with_tier]`: a single combined-cover STC over all four bypass-bin
/// domains (CS → CSL → MVDs → MVDsl), with passphrase-derived seeds. OH264
/// reaches the wire via `wire_only` scratch-table overrides at CABAC emit,
/// so the encoder reconstruction stays clean by construction (this is what
/// keeps mode-decision stable across baseline ↔ stego encodes). DECODE is
/// not here — it lives in `streaming_session::StreamingDecodeSession`.
#[cfg(feature = "h264-encoder")]
pub mod openh264_stego;

/// Streaming H.264 stego session API: the OH264 encode session
/// (mobile/CLI FFI) + the engine-agnostic streaming DECODE session.
/// Per-GOP STC, bounded memory, arbitrary clip length. Design memo at
/// `docs/design/video/h264/d07-streaming-sessions.md`.
#[cfg(feature = "h264-decoder")]
pub mod streaming_session;

/// Engine-agnostic encode + decode progress event vocabulary.
/// Optional callback on the streaming session params; mobile bridges
/// marshal events across FFI into per-platform smoothing engines that
/// drive the HUD progress bar + ETA text. Design memo at
/// `docs/design/video/h264/progress-indicator.md`.
#[cfg(feature = "h264-decoder")]
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
