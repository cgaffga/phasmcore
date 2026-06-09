// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 CABAC stego module root (encode-time CABAC stego).
//!
//! This is the live stego root: it re-exports the decode-shared
//! types + helpers used by BOTH the OpenH264 encode path and the
//! pure-Rust CABAC bin-walker decode path — the 4-domain combine
//! (`combine_cover_4domain` / `split_plan_4domain` / `CostWeights` /
//! `DomainBoundaries`), `EmbedDomain` + `PositionKey`, and the
//! `inject` enumerate/extract/record primitives. The submodules
//! below carry cost (`content_costs` / `cost_weights`), STC framing,
//! `keys`, `tier_filter`, `cascade_safety`, and `shadow`;
//! `dpb_correction` + `oh264_capacity` are gated behind
//! `h264-encoder`.
//!
//! See `docs/design/video/_archive/h264/encoder-algorithms/stego-encode-time-architecture.md`
//! for the original architectural decisions A1–A7 and design rationale.

// Decode + shared — available under `h264-decoder`.
pub mod shadow_capacity;
pub mod cascade_safety;
// chunk_frame moved to `crate::stego::chunk_frame` so AV1 can share
// it (AV1 D.1+ wire format == H.264 chunk_frame v2/extended). The
// shared module carries main's extended form (#800 0xFFFF sentinel
// + u32 length) AND AV1's mandatory payload_len.
pub use crate::stego::chunk_frame;
pub mod content_costs;
pub mod cost_weights;
pub mod gop_pattern;
pub mod hook;
pub mod inject;
pub mod keys;
pub mod orchestrate;
pub mod shadow;
pub mod tier_filter;

// OH264 encode-side helper — `h264-encoder` only.
#[cfg(feature = "h264-encoder")]
pub mod dpb_correction;
// OH264 capacity surface + N-shadow input prep — `h264-encoder` only.
// Relocated from `encode_pixels` in the pure-Rust video retirement so the
// OH264 capacity API survives the encoder deletion.
#[cfg(feature = "h264-encoder")]
pub mod oh264_capacity;

pub use cost_weights::{
    combine_cover_4domain, split_plan_4domain, CostWeights, DomainBoundaries,
};
// `ResidualPathKind` is decode-shared — the walker (`bin_decoder`)
// consumes it. Defined unconditionally in `orchestrate`,
// so the re-export is ungated (was gated `h264-encoder`, which broke
// standalone `video,h264-decoder` decode builds).
pub use orchestrate::ResidualPathKind;

pub use hook::{
    Axis, BinKind, EmbedDomain, GopCapacity, NullLogger, PositionKey,
    PositionLogger, SyntaxPath,
};
pub use inject::{
    enumerate_mvd_sign_positions, enumerate_mvd_suffix_lsb_positions,
    extract_mvd_sign_bits, extract_mvd_suffix_lsb_bits,
    record_residual_block_into_cover, DomainBits, DomainCover, MvdSlot,
};
