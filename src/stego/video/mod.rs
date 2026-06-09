// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Video steganography.
//!
//! The production video pipeline is H.264 ([`h264_pipeline`]). The legacy
//! HEVC/H.265 pipeline was removed 2026-06-04 — see
//! `docs/design/video/_RETIREMENT-PLAN.md`.

pub mod h264_pipeline;

// H.264 decode dispatch re-exports. (Production H.264 video encode is the
// OpenH264 streaming session; this module is decode-only since the legacy
// CAVLC encode/capacity pipeline was retired — see
// `docs/design/video/_RETIREMENT-PLAN.md` § "Phase 4".)
pub use h264_pipeline::{h264_ghost_decode, h264_ghost_decode_path};
pub use crate::codec::mp4::is_mp4;
