// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 6 — pure-Rust H.264 encoder.
//!
//! **Clean-room invariant**: this subtree is derived from the H.264
//! spec (ITU-T Rec. H.264 | ISO/IEC 14496-10) plus published academic
//! literature. Code MUST cite spec sections, not other implementations.
//! The `core/src/codec/h264/openh264.rs` file is the explicit exception
//! — it binds against the OpenH264 BSD-2 fork intentionally.
//!
//! Module skeleton (Phase 6.0a). Most types are empty placeholders that
//! get filled in as the per-module sessions land — see
//! `docs/design/video/h264/encoder-algorithms/README.md` for the per-module
//! note + implementation tracking, and
//! `docs/design/video/h264/video-steganography.md` § Phase 6 for the full
//! sub-phase plan.
//!
//! ## Module structure
//!
//! - [`bitstream_writer`] — NAL / SPS / PPS / slice header / MB layer
//!   bit-level writer. Reuses Exp-Golomb helpers from the existing
//!   parser side.
//! - [`transform`] — forward 4×4 + 8×8 integer DCT, forward Hadamard
//!   4×4 DC, plus the inverses needed by the reconstruction loop.
//!   Forward direction is new; inverse is shared with Phase 1b.
//! - [`quantization`] — forward quantization with trellis-quant
//!   refinement.
//! - [`intra_predictor`] — all 9 Intra_4x4 modes, 4 Intra_16x16,
//!   chroma intra, plus SATD-based mode decision with psy-RD bias.
//! - [`cavlc_writer`] — CAVLC encoder (luma 4×4, chroma DC + AC,
//!   I_16x16 DC + AC). Reverses the existing Phase 1a decoder.
//! - [`reconstruction`] — inverse quant + IDCT + add prediction +
//!   neighbor pixel buffer for next-block intra-pred lookup.
//! - [`deblocking_filter`] — boundary strength + alpha/beta tables +
//!   Filter4 / Filter2 + cross-MB filtering per ITU-T H.264 § 8.7.
//! - [`rate_control`] — per-MB variance + AQ-mode 1 + CRF-style frame
//!   QP target + the `quality: u8` public API + source-quality
//!   estimator.
//! - [`motion_estimation`] — Phase 6B: hexagonal + UMH + sub-pel
//!   refinement.
//! - [`motion_compensation`] — Phase 6B: 6-tap quarter-pel luma +
//!   bilinear chroma.
//! - [`reference_buffer`] — Phase 6B: DPB + reference list + GOP
//!   structure.
//! - [`encoder`] — top-level encoder driver: pixel input → MB loop →
//!   bytes out. Holds the encoder state machine.
//!
//! ## Status
//!
//! Skeleton only. Every public function in this module currently
//! returns a `EncoderError::NotImplemented` or panics with `todo!()`.
//! Real implementations land in Phase 6A through 6E.

pub mod b_direct_predictor;
pub mod b_inter_prediction;
pub mod b_partitioned;
pub mod baseline_transcode;
pub mod bitstream_writer;
pub mod deblocking_filter;
#[allow(clippy::module_inception)] // Encoder driver lives in encoder/encoder.rs by design.
pub mod encoder;
pub mod i4x4_encode;
pub mod inter_mode;
pub mod intra_8x8_encode;
pub mod intra_predictor;
pub mod lookahead;
pub mod mb_decision_b;
pub mod mb_tree;
pub mod motion_compensation;
pub mod motion_estimation;
pub mod partition_decision;
pub mod partition_state;
pub mod poc;
pub mod quantization;
pub mod rate_control;
pub mod rdo;
pub mod rdo_b;
pub mod reconstruction;
pub mod reference_buffer;
pub mod reorder;
pub mod simd;
// Phase 6F.1 — `stego` was relocated to `h264::stego`
// (peer of encoder/) so the bin-decoder doesn't need to import
// from encoder::*. Module declaration moved to
// `core/src/codec/h264/mod.rs`. This empty marker line preserves
// commit history readability.
pub mod transform;
pub mod transform_8x8;
// CAVLC writer lives next door at `core/src/codec/h264/cavlc_writer.rs`
// once Phase 6A.4 lands — keeping it one level up keeps it sibling to
// the existing Phase 1a `cavlc.rs` decoder.

/// Top-level error type for the H.264 encoder.
#[derive(Debug, Clone)]
pub enum EncoderError {
    /// Returned by every method during the Phase 6.0 skeleton phase.
    /// Implementing modules replace this with proper variants.
    NotImplemented(&'static str),
    /// Caller passed invalid parameters (e.g. non-MB-aligned dimensions).
    InvalidInput(String),
}

impl core::fmt::Display for EncoderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EncoderError::NotImplemented(what) => {
                write!(f, "H.264 encoder: not yet implemented — {what}")
            }
            EncoderError::InvalidInput(what) => {
                write!(f, "H.264 encoder: invalid input — {what}")
            }
        }
    }
}

impl std::error::Error for EncoderError {}
