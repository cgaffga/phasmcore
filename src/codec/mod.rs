// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Media format codecs — pure I/O parsers with no steganographic logic.
//!
//! - [`jpeg`] — JPEG coefficient codec (zero-dependency, byte-for-byte round-trip)
//! - [`mp4`] — ISO BMFF / QuickTime MP4/MOV container demuxer and muxer

pub mod jpeg;

#[cfg(feature = "video")]
pub mod h264;
#[cfg(feature = "video")]
pub mod mp4;

// AV1 stego: skeleton landed 2026-05-20 (W3.2). Now (W3.D.4.2) gated
// on EITHER av1-encoder (rav1e) OR av1-decoder (dav1d), since the
// stego module contains both encode-side (writer.rs) and decode-side
// (decoder.rs) pieces. Files within are individually cfg'd on the
// specific feature they need.
#[cfg(any(feature = "av1-encoder", feature = "av1-decoder"))]
pub mod av1;
