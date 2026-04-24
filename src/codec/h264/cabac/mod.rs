// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 CABAC entropy coding. Phase 6C.
//!
//! **Independent of the archived HEVC CABAC** under
//! `core/src/codec/cabac/`. H.264 CABAC has different tables, different
//! context models, different binarization patterns, different syntax
//! semantics. Sharing the arithmetic engine is tempting but in practice
//! creates subtle "we fixed a bug in only one of them" hazards. Each
//! module owns its own implementation.
//!
//! Phase 6C sub-sub-phases:
//!   - 6C.1 CABAC arithmetic engine (state, put, bypass, terminate, renorm, finish)
//!   - 6C.2 H.264 CABAC tables (RANGE_TAB_LPS, TRANS_IDX_*, context-init)
//!   - 6C.3 Binarization (unary, truncated unary, Exp-Golomb, fixed-length, custom)
//!   - 6C.4 Context selection logic (depends on neighbor values)
//!   - 6C.5 Per-syntax-element encoders (mb_skip_flag, mb_type, sub_mb_type,
//!          intra modes, ref_idx, mvd, qp_delta, CBP, residual block)
//!   - 6C.6 Integration (PPS wiring, slice framing, feature parity)
//!   - 6C.7 Post-CABAC polish (including 8×8 DCT from deferred #11)
//!
//! ## Critical lessons from the HEVC CABAC saga (DO NOT repeat):
//!
//! - **Byte-level write_out** from day one, NOT bit-level PutBit. Phase
//!   6C.1 follows the HM / HEVC-CABAC pattern: `low` is 32-bit left-
//!   justified, carry propagation operates on buffered 0xFF bytes.
//! - **`finish()` is co-designed with the slice writer**. RBSP stop bit
//!   shares the same bit buffer as CABAC's last byte. The final
//!   `encode_terminate(1)` call triggers the flush.
//! - **Diff-check every table against the spec transcription**. Hash-
//!   based integrity (Phase 6C.2) + monotonicity checks (in 6C.1
//!   already) catch any single-entry typo cheaply.
//! - **No "modify someone else's CABAC stream" anywhere**. Phase 6C
//!   only generates from scratch. We do not mix bytes.
//!
//! See `docs/design/h264-encoder-algorithms/cabac-engine.md` (and
//! sibling notes) for the per-module algorithm notes.

pub mod binarization;
pub mod context;
pub mod encoder;
pub mod engine;
pub mod neighbor;
pub mod slice;
pub mod tables;

pub use context::CabacContext;
pub use encoder::CabacEncoder;
pub use engine::CabacEngine;
