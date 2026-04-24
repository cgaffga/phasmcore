// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Video steganography.
//!
//! Phase 4a: the production pipeline is H.264 Baseline CAVLC
//! ([`h264_pipeline`]). The legacy HEVC/H.265 pipeline + its supporting
//! `capacity`, `gop`, `cbib`, and `pipeline` modules are archived behind
//! the `hevc-archive` feature flag (off by default). Source stays in tree
//! for future reference; re-enable with `--features hevc-archive`.


pub mod h264_pipeline;

// H.264 (production) re-exports.
pub use h264_pipeline::{
    h264_ghost_encode, h264_ghost_encode_inplace,
    h264_ghost_decode, h264_ghost_capacity,
    h264_ghost_encode_path, h264_ghost_decode_path, h264_ghost_capacity_path,
};
pub use crate::codec::mp4::is_mp4;

// HEVC (archived) re-exports — only visible when `hevc-archive` is enabled.
