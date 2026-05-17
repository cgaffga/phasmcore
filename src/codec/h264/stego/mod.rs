// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 6D — encode-time CABAC stego scaffolding.
//!
//! This module ships the TYPE + TRAIT contracts that the rest of
//! Phase 6D builds on. Concrete implementations land in 6D.3+ and
//! the encoder integration happens alongside the first concrete hook
//! to avoid wiring call sites without test coverage.
//!
//! See `docs/design/video/h264/encoder-algorithms/stego-encode-time-architecture.md`
//! for the architectural decisions A1–A7 and overall design,
//! and `cabac-bypass-bin-stego.md` for the bin-by-bin invariant proofs.

pub mod cascade_safety;
pub mod chunk_frame;
pub mod cost_model;
pub mod cost_weights;
pub mod cover_replay;
pub mod decode_pixels;
pub mod encode_pixels;
pub mod encoder_hook;
pub mod gop_pattern;
pub mod hook;
pub mod inject;
pub mod keys;
pub mod orchestrate;
pub mod per_gop_plan;
pub mod primary_rs;
pub mod provisional_emit;
pub mod shadow;
pub mod validate;

pub use cost_model::PositionCostCtx;
pub use cost_weights::{
    combine_cover_4domain, split_plan_4domain, CostWeights, DomainBoundaries,
};
pub use orchestrate::ResidualPathKind;

pub use hook::{
    Axis, BinKind, BitInjector, EmbedDomain, GopCapacity, NullLogger,
    PositionCounter, PositionKey, PositionLogger, PositionRecorder, SyntaxPath,
};
pub use inject::{
    apply_coeff_sign_overrides, apply_coeff_suffix_lsb_overrides,
    apply_mvd_sign_overrides, apply_mvd_suffix_lsb_overrides,
    enumerate_coeff_sign_positions, enumerate_coeff_suffix_lsb_positions,
    enumerate_mvd_sign_positions, enumerate_mvd_suffix_lsb_positions,
    extract_coeff_sign_bits, extract_coeff_suffix_lsb_bits,
    extract_mvd_sign_bits, extract_mvd_suffix_lsb_bits,
    DomainBits, DomainCover, MvdSlot,
};
