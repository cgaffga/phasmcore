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

// H.264 (production) re-exports.
#[cfg(feature = "video")]
pub use stego::video::{
    h264_ghost_encode, h264_ghost_encode_inplace,
    h264_ghost_decode, h264_ghost_capacity, h264_ghost_capacity_max,
    h264_ghost_encode_path, h264_ghost_decode_path,
    h264_ghost_capacity_path, h264_ghost_capacity_max_path,
    is_mp4,
};

// H.264 phase-6 encoder transcoder (#77) — replaces VideoToolbox /
// MediaCodec on mobile for the input-video → Baseline-CAVLC step.
// `StreamingEncoder` is the per-frame stateful API used by the mobile
// bridges; `transcode_yuv_to_baseline_cavlc_h264` is the one-shot
// convenience wrapper for tests / CLI / batch contexts.
// D.0.7 — streaming H.264 stego session API (engine-agnostic).
// Mobile bridges + CLI consume these directly; they own the per-GOP
// state machine + chunk_frame wire format. Design memo:
// `docs/design/video/h264/d07-streaming-sessions.md`.
#[cfg(feature = "h264-encoder")]
pub use codec::h264::streaming_session::{
    CapacityProbeResult, ColorParams, DecodeSessionResult, EncodeEngineChoice,
    EncodeSessionParams, StreamingDecodeSession, StreamingEncodeSession,
    StreamingProbeSession, YuvFrameRef,
};
#[cfg(feature = "h264-encoder")]
pub use codec::h264::stego::CostWeights;

// #474 — video stego progress event vocabulary. Mobile bridges re-export
// these to wire C/JNI callbacks; see `docs/design/video/h264/progress-indicator.md`.
#[cfg(feature = "h264-encoder")]
pub use codec::h264::progress::{
    decode_phase_codes, encode_phase_codes, DecodePhase, DecodeProgressCallback, EncodePhase,
    EncodeProgressCallback, PROGRESS_MIN_INTERVAL, ProgressThrottle,
};

#[cfg(feature = "h264-encoder")]
pub use codec::h264::encoder::baseline_transcode::{
    BaselineTranscodeConfig, StreamingEncoder, transcode_yuv_to_baseline_cavlc_h264,
};

// Phase 6D.8 chunk 5 — encode-time CABAC stego public API. UTF-8
// string message + passphrase + raw I420 YUV → Annex-B byte stream.
// I-frame-only single-GOP scope until §30 MVD wiring lands.
// Mobile bridges + CLI route through this once chunk 7 atomic-swaps
// the legacy CAVLC pipeline gates.
#[cfg(feature = "cabac-stego")]
pub use codec::h264::stego::encode_pixels::{
    h264_stego_encode_i_frames_only, h264_stego_encode_i_then_p_frames,
    h264_stego_encode_yuv_string, h264_stego_encode_yuv_string_4domain,
    h264_stego_encode_yuv_string_4domain_multigop,
    h264_stego_encode_yuv_string_4domain_multigop_streaming_v2,
    h264_stego_encode_yuv_string_4domain_multigop_with_pattern,
    h264_stego_encode_yuv_string_with_n_shadows,
    h264_stego_encode_yuv_string_with_n_shadows_with_pattern,
    h264_stego_encode_yuv_string_with_shadow,
    h264_stego_encode_yuv_string_with_shadow_with_pattern,
    h264_stego_shadow_capacity, H264ShadowCapacityInfo,
    // Task #96 — combined primary + shadow capacity surface.
    // #796 — OH264-accurate variant for the production streaming session.
    h264_stego_capacity_4domain, H264StegoCapacityInfo,
    h264_stego_capacity_4domain_oh264,
    h264_resolve_auto_tier_oh264,
    // Task #97 — file-attachment-aware encode entries.
    h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_files,
    h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files,
    h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files_with_tier,
    h264_stego_encode_yuv_string_with_n_shadows_with_pattern_and_files,
};
#[cfg(feature = "cabac-stego")]
pub use codec::h264::stego::gop_pattern::{FrameType, GopPattern};
// D'.5 — Track 1 cascade-safety tier types for CLI / bridges.
#[cfg(feature = "cabac-stego")]
pub use codec::h264::stego::tier_filter::{CascadeTier, DEFAULT_HEADROOM as CASCADE_DEFAULT_HEADROOM};

// Phase 6D.8 chunk 6G + §30D-C — decode entry points. Walks the
// encoded Annex-B + passphrase → recovered UTF-8 string. The
// `_4domain` variant pairs with §30D-C's 3-pass encoder + uses
// fill-MVD-first allocation; the basic variant pairs with chunk
// 5 / §30C residual-only encoders.
#[cfg(feature = "cabac-stego")]
pub use codec::h264::stego::decode_pixels::{
    h264_stego_decode_yuv_string, h264_stego_decode_yuv_string_4domain,
    h264_stego_shadow_decode, h264_stego_smart_decode_video,
    // Task #97 — _with_payload variant returns PayloadData (text +
    // attached files), not just text.
    h264_stego_smart_decode_video_with_payload,
};

// HEVC (archived) re-exports — only available with the `hevc-archive` feature.

/// Detected video codec inside an MP4 container.
///
/// The `Hevc` variant is still reported even when the HEVC pipeline is
/// archived — detection only needs MP4-level codec bytes, not the full
/// HEVC parser. Callers should check for `VideoCodec::H264` as the only
/// supported stego target.
#[cfg(feature = "video")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264/AVC — uses CAVLC-based encoding (Baseline/Main CAVLC).
    H264,
    /// HEVC/H.265. Detected but not supported for stego in current builds;
    /// the pipeline is archived behind the `hevc-archive` feature flag.
    Hevc,
    /// MP4 but no recognised H.264 or HEVC track.
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
    } else {
        VideoCodec::Unknown
    }
}
