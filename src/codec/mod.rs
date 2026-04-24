// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Media format codecs — pure I/O parsers with no steganographic logic.
//!
//! - [`jpeg`] — JPEG coefficient codec (zero-dependency, byte-for-byte round-trip)
//! - [`hevc`] — H.265/HEVC bitstream parser (NAL units, SPS, PPS, slices, CTUs)
//! - [`cabac`] — Context-Adaptive Binary Arithmetic Coding engine
//! - [`mp4`] — ISO BMFF / QuickTime MP4/MOV container demuxer and muxer

pub mod jpeg;

// Phase 4a: HEVC modules are archived. Gate behind `hevc-archive` (off by
// default). The H.264 pipeline is the production path.
#[cfg(feature = "video")]
pub mod h264;
#[cfg(feature = "video")]
pub mod mp4;
