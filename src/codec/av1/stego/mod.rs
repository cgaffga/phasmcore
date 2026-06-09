// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AV1 stego — Writer-trait-based Pass 1 record + Pass 2 cached
//! replay orchestration.
//!
//! See `docs/design/video/av1/streaming-session.md` for the pass
//! architecture and the WriterRecorder + WriterStego pipeline.
//!
//! # WriterStego skeleton state
//!
//! [`writer::WriterStego`] is currently a structural skeleton:
//! - Defines the struct shape that satisfies both [`Writer`] and
//!   [`StorageBackend`] (`WriterRecorder::replay` takes
//!   `&mut dyn StorageBackend`, NOT `&mut dyn Writer`, so the stego
//!   writer must impl BOTH to act as a live encode interceptor AND
//!   a cached-replay sink).
//! - Stub impls forward all calls to an inner `WriterEncoder` with
//!   TODO markers where Tier 1 override logic plugs in.
//!
//! Real Tier 1 override logic lands alongside:
//! - STC plan integration (`crate::stego::stc`)
//! - Cover-position canonical order enumeration (see
//!   `docs/design/video/av1/streaming-session.md`)
//! - cascade-safety filter integration (see
//!   `docs/design/video/av1/cascade-safety.md`)

#[cfg(feature = "av1-encoder")]
pub mod writer;

// phasm-core decode-side wrapper around the dav1d fork hooks.
// Takes encoded AV1 bytes, drives dav1d with a recording bit_hook
// registered, returns DecodedCoverPositions (mirror of writer.rs's
// CoverPositions encoder-side wrapper).
#[cfg(feature = "av1-decoder")]
pub mod decoder;

// Production av1_stego_embed + av1_stego_extract that wire rav1e
// encode + dav1d decode + STC + crypto into the end-to-end
// hide-a-message flow. Module compiles when EITHER feature is on;
// individual functions are feature-gated so a decode-only build
// (av1-decoder without av1-encoder, as shipped on the phasmcore
// public mirror) exposes only av1_stego_extract.
#[cfg(any(feature = "av1-encoder", feature = "av1-decoder"))]
pub mod orchestrator;

// Cascade-safety v2 forward modeling kernels (deblock + CDEF
// approximation). Consumed by the cost-compute path's three-tier
// dispatch; for the ~10-15% of cover positions in the ambiguous
// |coeff| middle band, the L2 cache here produces a per-tuple
// post-cascade impulse pattern from L1's basis + frame-level
// loop-filter state. See
// `docs/design/video/av1/phase-b15-cascade-safety-v2.md`.
#[cfg(feature = "av1-encoder")]
pub mod cascade_kernel;

// Av1StreamingEncodeSession / Av1StreamingDecodeSession. Per-GOP
// chunk-framed wire format; started as single-frame-per-GOP
// internally (legacy `av1_stego_embed` is the v=1 special case),
// then scaled to N-frame GOPs once the multi-frame primitive landed
// in the phasm-rav1e fork. See
// `docs/design/video/av1/phase-c-streaming-session-v6.md`.
#[cfg(any(feature = "av1-encoder", feature = "av1-decoder"))]
pub mod session;

// Capacity probe + closed-form capacity math. Two-API + real-probe
// shape lifted from H.264. See
// `docs/design/video/av1/phase-c-capacity-api.md`.
#[cfg(feature = "av1-encoder")]
pub mod capacity;

// Shadow messages — multi-message plausible deniability. Data
// structures + ChaCha20 position priority + embed + extract +
// multi-shadow cascade + session integration. See
// `docs/design/video/av1/phase-c-shadows.md`.
#[cfg(any(feature = "av1-encoder", feature = "av1-decoder"))]
pub mod shadow;

// YUV-level convenience entry points for mobile bridges (iOS /
// Android). Wraps the existing `av1_stego_embed` /
// `av1_stego_extract` flow with the keyframe-encode bookkeeping the
// bridge would otherwise have to replicate (and that lives only in
// integration tests today).
#[cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]
pub mod yuv_io;

// Whole-video shadow encode flow. Ported from OH264's
// `h264_encode_with_shadows`; AV1-simplified because
// `replay_with_overrides` is wire-clean by construction (no
// provisional Pass 2 needed). See
// `docs/design/video/av1/phase-c-wv-whole-video-shadow.md`. Needs
// both features because the encode path self-verifies via the
// decoder hook (`harvest_cover_bits_from_stego`).
#[cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]
pub mod whole_video;
