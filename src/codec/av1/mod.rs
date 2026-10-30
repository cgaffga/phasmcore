// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AV1 video stego — Phasm AV1 stego implementation.
//!
//! Built on top of the `cgaffga/phasm-rav1e` fork (`phasm-stego`
//! branch; vendored at `vendor/phasm-rav1e` in the monorepo, patched in
//! from GitHub elsewhere: see BUILDING.md). Implements
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

// Fork guard. `av1-encoder` needs the phasm rav1e fork (the `phasm_stego`
// hooks), which phasm-core gets by patching the crates.io package `rav1e`
// to github.com/cgaffga/phasm-rav1e at a pinned commit. Without the patch,
// or when cargo drops it ("patch `rav1e` was not used", e.g. after the
// fork's version moved off the `=0.8.0` that Cargo.toml requires), cargo
// silently compiles upstream rav1e from crates.io instead. This import of
// the fork's marker module is then the error to read; its message is on
// the line itself because rustc prints that line. See BUILDING.md,
// "The rav1e fork".
#[cfg(feature = "av1-encoder")]
#[allow(unused_imports)]
use phasm_rav1e::phasm_stego as _; // av1-encoder needs the phasm rav1e fork: build with `make build-video` or see BUILDING.md

pub mod stego;
