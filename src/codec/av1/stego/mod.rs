// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AV1 stego — Writer-trait-based Pass 1 record + Pass 2 cached
//! replay orchestration.
//!
//! See `docs/design/video/av1/streaming-session.md` § 1 (pass
//! architecture) and § 3 (WriterRecorder + WriterStego pipeline).
//!
//! # W3.2 SKELETON STATE (2026-05-20)
//!
//! [`writer::WriterStego`] is currently a structural skeleton:
//! - Defines the struct shape that satisfies both [`Writer`] and
//!   [`StorageBackend`] (audit B-S4 finding: `WriterRecorder::replay`
//!   takes `&mut dyn StorageBackend`, NOT `&mut dyn Writer`, so the
//!   stego writer must impl BOTH to act as a live encode interceptor
//!   AND a cached-replay sink).
//! - Stub impls forward all calls to an inner `WriterEncoder` with
//!   TODO markers where Tier 1 override logic plugs in.
//!
//! Real Tier 1 override logic lands in W3.3+ alongside:
//! - STC plan integration (`crate::stego::stc`)
//! - Cover-position canonical order enumeration (per
//!   `docs/design/video/av1/streaming-session.md` § 5)
//! - cascade-safety filter integration (per
//!   `docs/design/video/av1/cascade-safety.md`)

#[cfg(feature = "av1-encoder")]
pub mod writer;

// W3.D.4.2: phasm-core decode-side wrapper around the W3.D fork
// hooks. Takes encoded AV1 bytes, drives dav1d with a recording
// bit_hook registered, returns DecodedCoverPositions (mirror of
// writer.rs's CoverPositions encoder-side wrapper).
#[cfg(feature = "av1-backend")]
pub mod decoder;

// W3.10.5: production av1_stego_embed + av1_stego_extract that wire
// rav1e encode + dav1d decode + STC + crypto into the end-to-end
// hide-a-message flow. Module compiles when EITHER feature is on;
// individual functions are feature-gated so a decode-only build
// (av1-backend without av1-encoder, as shipped on the phasmcore
// public mirror) exposes only av1_stego_extract.
#[cfg(any(feature = "av1-encoder", feature = "av1-backend"))]
pub mod orchestrator;
