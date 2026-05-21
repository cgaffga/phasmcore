// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AV1 video stego — Phasm AV1 stego implementation.
//!
//! Built on top of the `cgaffga/phasm-rav1e` fork (vendored at
//! `vendor/phasm-rav1e` on the `phasm-stego` branch). Implements
//! the Option E per-GOP two-pass architecture from
//! `docs/design/video/av1/streaming-session.md` § 1.
//!
//! # Module structure
//!
//! - [`stego`] — WriterStego + STC plan + cover-position enumeration
//!   (Pass 1 record + Pass 2 cached replay via rav1e's exposed
//!   `Writer` + `StorageBackend` traits).
//!
//! Phase A workstream W3. v0.3-AV1 ship target: Tier 1 AcCoeffSign
//! only, with the channel-design.md § 6 phased build plan adding
//! Tier 2 channels in v0.5+ and Tier 3 (CdefIdx) in v1.0-AV1.

pub mod stego;
