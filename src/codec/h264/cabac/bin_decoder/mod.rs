// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 CABAC bin-level decoder for stego decode.
//!
//! Decode direction only — the spec § 9.3.3.2 arithmetic engine run
//! backwards. It decodes the same syntax elements the bitstream
//! carries, in the same order, using the same context tables. The
//! forward CABAC encoder was removed in the video-retirement;
//! production stego bytes are emitted by the OpenH264 fork, and the
//! walker reads back the bins embedded at the bypass-bin emit site.
//!
//! ## Scope
//!
//! This module implements **just enough CABAC parsing to locate stego
//! positions**. We do NOT reconstruct pixels, run intra/inter
//! prediction, or deblock — the goal is purely to walk the bitstream
//! and emit a [`PositionKey`](crate::codec::h264::stego::PositionKey)
//! for every bypass bin in our four target domains
//! ([`EmbedDomain`](crate::codec::h264::stego::EmbedDomain)).
//!
//! Total scope is ~3500 LOC vs ~12K for a full reference decoder.
//!
//! ## Module structure:
//!   - [`engine`] — `CabacDecodeEngine`, the spec § 9.3.3.2 arithmetic decoder
//!   - [`syntax`] — per-syntax-element decoders
//!   - [`positions`] — position tracker that emits `PositionKey`s during parse
//!   - [`slice`] — top-level slice walker
//!
//! See `docs/design/video/_archive/h264/encoder-algorithms/stego-encode-time-architecture.md`
//! § A3 for scope rationale and § 12 ("slice boundary handling")
//! for the single-slice constraint that this decoder honours.

pub mod bitstream_mod;
pub mod decoder;
pub mod engine;
pub mod positions;
pub mod slice;
pub mod syntax;

pub use decoder::CabacDecoder;
pub use engine::{CabacDecodeEngine, DecodeError};
pub use positions::PositionRecorder;
pub use bitstream_mod::{
    cabac_data_byte_offset, locate_nal_units_annexb, repack_emulation_prevention,
    strip_emulation_prevention, strip_emulation_prevention_with_map,
    EmulationPrevByteMap, LocateError, NalLocation,
};
pub use slice::{
    walk_annex_b_for_cover, walk_annex_b_for_cover_with_options,
    walk_nalus_for_cover_with_options,
    walk_annex_b_streaming, walk_nalus_streaming_with_options,
    CoverWalkOutput, GopContext, StreamingWalkOutput, WalkAction,
    WalkError, WalkOptions,
};
