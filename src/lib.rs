// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! # phasm-core
//!
//! Pure-Rust steganography engine for hiding encrypted text messages in JPEG
//! photos. Provides two embedding modes:
//!
//! - **Ghost** (stealth): J-UNIWARD cost function + STC coding to resist
//!   statistical steganalysis. Optimizes for undetectability.
//! - **Armor** (robust): STDM embedding + Reed-Solomon ECC to survive
//!   JPEG recompression. Optimizes for message survivability.
//!
//! All processing is client-side. The JPEG coefficient codec (`jpeg` module)
//! is zero-dependency (std only). The steganography layer (`stego` module)
//! uses AES-256-GCM-SIV encryption and Argon2id key derivation.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use phasm_core::{ghost_encode, ghost_decode};
//!
//! let cover_jpeg = std::fs::read("photo.jpg").unwrap();
//! let stego = ghost_encode(&cover_jpeg, "secret message", "passphrase").unwrap();
//! let decoded = ghost_decode(&stego, "passphrase").unwrap();
//! assert_eq!(decoded.text, "secret message");
//! ```

pub mod codec;
pub mod det_math;
pub mod stego;

// Backward-compatible re-exports: `phasm_core::jpeg::*` still works
pub use codec::jpeg as jpeg;

pub use codec::jpeg::error::{JpegError, Result as JpegResult};
pub use codec::jpeg::dct::{DctGrid, QuantTable};
pub use codec::jpeg::frame::FrameInfo;
pub use codec::jpeg::JpegImage;
pub use stego::{ghost_encode, ghost_decode, ghost_encode_with_files, ghost_encode_si, ghost_encode_si_with_files, ghost_capacity, ghost_capacity_si, StegoError, GHOST_DECODE_STEPS, GHOST_ENCODE_STEPS};
pub use stego::{ghost_encode_with_quality, ghost_encode_with_files_quality, ghost_encode_si_with_quality, ghost_encode_si_with_files_quality};
pub use stego::{ghost_encode_with_shadows, ghost_encode_si_with_shadows, ghost_shadow_decode, ShadowLayer, GHOST_ENCODE_WITH_SHADOWS_STEPS, shadow_capacity, estimate_shadow_capacity, ghost_capacity_with_shadows};
pub use stego::{ghost_encode_with_shadows_quality, ghost_encode_si_with_shadows_quality};
pub use stego::{armor_encode, armor_encode_with_quality, armor_decode, armor_capacity, armor_capacity_info, smart_decode, DecodeQuality, ArmorCapacityInfo};
pub use stego::EncodeQuality;
pub use stego::{validate_encode_dimensions, MAX_DIMENSION, MAX_PIXELS, MIN_ENCODE_DIMENSION, ARMOR_TARGET_DIMENSION};
pub use stego::{PayloadData, FileEntry, compressed_payload_size};
pub use stego::progress;
pub use stego::memory;
pub use stego::{
    get_memory_budget, predict_peak_memory, select_ghost_shadow_rung,
    set_memory_budget, set_telemetry_hook,
    GhostShadowRung, ModeId, TelemetryEvent, TelemetryHook,
};
pub use stego::{optimize_cover, OptimizerConfig, OptimizerMode};

// H.264 decode dispatch re-exports. (Production H.264 video encode is the
// OpenH264 streaming session below; the legacy CAVLC encode/capacity pipeline
// was retired — see `docs/design/video/_RETIREMENT-PLAN.md` § "Phase 4".)
#[cfg(feature = "video")]
pub use stego::video::{h264_ghost_decode, h264_ghost_decode_path, is_mp4};

// Streaming H.264 stego session API (engine-agnostic). The encoder
// transcoder (#77) replaces VideoToolbox / MediaCodec on mobile for the
// input-video → H.264 step. Mobile bridges + CLI consume these directly;
// they own the per-GOP state machine + chunk_frame wire format. Design
// memo: `docs/design/video/h264/d07-streaming-sessions.md`.
// Streaming DECODE session — available standalone (decode-only builds:
// App Clip / decode WASM decode phasm H.264 stego without OpenH264).
#[cfg(feature = "h264-decoder")]
pub use codec::h264::streaming_session::{DecodeSessionResult, StreamingDecodeSession};
// Streaming ENCODE session + capacity probe — OpenH264 encoder only.
#[cfg(feature = "h264-encoder")]
pub use codec::h264::streaming_session::{
    CapacityProbeResult, ColorParams, EncodeEngineChoice, EncodeSessionParams,
    StreamingEncodeSession, StreamingProbeSession, YuvFrameRef,
};
#[cfg(feature = "h264-decoder")]
pub use codec::h264::stego::CostWeights;

// #474 — video stego progress event vocabulary. Mobile bridges re-export
// these to wire C/JNI callbacks; see `docs/design/video/h264/progress-indicator.md`.
#[cfg(feature = "h264-decoder")]
pub use codec::h264::progress::{
    decode_phase_codes, encode_phase_codes, DecodePhase, DecodeProgressCallback, EncodePhase,
    EncodeProgressCallback, PROGRESS_MIN_INTERVAL, ProgressThrottle,
};

// OH264 video-stego capacity surface (relocated from the retired pure-Rust
// `encode_pixels` in the video-retirement Phase 3). Production capacity API
// consumed by the mobile bridges + CLI for the streaming session.
#[cfg(feature = "h264-encoder")]
pub use codec::h264::stego::oh264_capacity::{
    h264_resolve_auto_tier, h264_shadow_capacity_for_n,
    h264_video_capacity, H264StegoCapacityInfo,
};
#[cfg(feature = "h264-decoder")]
pub use codec::h264::stego::gop_pattern::{FrameType, GopPattern};
// Cascade-safety tier types for CLI / bridges.
#[cfg(feature = "h264-decoder")]
pub use codec::h264::stego::tier_filter::{CascadeTier, DEFAULT_HEADROOM as CASCADE_DEFAULT_HEADROOM};

// AV1 streaming sessions + capacity. Mirror of the H.264 streaming
// surface above.
//
// Re-exports require `av1-encoder` (which also implies `av1-decoder` at
// the CLI / bridge feature layer). The `av1-decoder`-only configuration
// would naturally re-export the decode session alone, but
// `codec::av1::stego::session` currently has unconditional
// `phasm_rav1e::*` use statements at module top — so the av1-decoder-only
// build is broken upstream regardless of these re-exports. Tracked as
// follow-on; in practice CLI builds always pair encoder + decoder, like
// the H.264 surface above.
#[cfg(feature = "av1-encoder")]
pub use codec::av1::stego::session::{
    Av1ShadowSpec, Av1StreamingDecodeSession, Av1StreamingEncodeParams,
    Av1StreamingEncodeSession,
};
#[cfg(feature = "av1-encoder")]
pub use codec::av1::stego::capacity::{
    av1_capacity, av1_shadow_capacity, Av1CapacityInfo, Av1PerDomainBits,
    Av1ShadowCapacityInfo, Av1StreamingProbeSession,
};
#[cfg(feature = "av1-encoder")]
pub use codec::av1::stego::orchestrator::Av1StegoError;

/// Detected video codec inside an MP4 container.
///
/// `Hevc` is still reported even though the HEVC pipeline was removed —
/// detection only needs MP4-level codec bytes, not a full parser. Callers
/// dispatch on `H264` (OpenH264 streaming session) or `Av1`
/// (Av1StreamingDecodeSession); `Hevc` / `Unknown` should be surfaced as
/// "not a phasm stego" to the user.
#[cfg(feature = "video")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264/AVC — `avc1` / `avc3` sample entry.
    H264,
    /// HEVC/H.265 — `hev1` / `hvc1` sample entry. Detected but not
    /// supported for stego (HEVC pipeline removed, see
    /// docs/design/video/_RETIREMENT-PLAN.md).
    Hevc,
    /// AV1 — `av01` sample entry. Stego-supported via the
    /// `Av1StreamingEncodeSession` / `Av1StreamingDecodeSession`
    /// pipeline (phasm-rav1e encode + phasm-dav1d decode).
    Av1,
    /// MP4 but no recognised H.264 / HEVC / AV1 track.
    Unknown,
}

/// Detect the video codec of an MP4 byte buffer. Returns `Unknown` if the
/// input is not an MP4 or has no recognised video track.
#[cfg(feature = "video")]
pub fn detect_video_codec(mp4_bytes: &[u8]) -> VideoCodec {
    if !is_mp4(mp4_bytes) {
        return VideoCodec::Unknown;
    }
    let Ok(file) = codec::mp4::demux::demux(mp4_bytes) else {
        return VideoCodec::Unknown;
    };
    let Some(idx) = file.video_track_idx else {
        return VideoCodec::Unknown;
    };
    let track = &file.tracks[idx];
    if track.is_h264() {
        VideoCodec::H264
    } else if track.is_hevc() {
        VideoCodec::Hevc
    } else if track.is_av1() {
        VideoCodec::Av1
    } else {
        VideoCodec::Unknown
    }
}
