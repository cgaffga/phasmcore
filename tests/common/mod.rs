// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Shared integration-test helpers. Each file under `tests/` is compiled
//! as its own crate, so code shared across tests goes here and is
//! included with `mod common;` + `use common::*;` in each test file.

#![allow(dead_code)] // not every test file uses every helper

pub mod h264_oracle;
pub mod oh264_stream;
pub mod ship_gate;
pub mod video_fixture;
pub mod visual_metrics;

/// Base directory for test scratch files — decoded-YUV caches, generated
/// fixtures, intermediate bitstreams.
///
/// Resolution: `$PHASM_TMP` if set, else `/tmp`.
///
/// **Why this exists.** The video gates cache multi-hundred-megabyte decoded
/// YUV per fixture/resolution so repeat runs skip ffmpeg. Hardcoding `/tmp`
/// put ~14 GB on this machine's internal disk, which is the scarce one —
/// found 2026-10-18 when the tree was relocated to an external disk. The
/// variable keeps that redirectable **per project**, which matters because the
/// alternative (exporting `TMPDIR`) is machine-global and would drag every
/// other project on the machine onto the external disk too.
///
/// `/tmp` is the fallback so a fresh clone, CI, or any machine without the
/// variable set behaves exactly as before.
pub fn scratch_dir() -> String {
    std::env::var("PHASM_TMP").unwrap_or_else(|_| "/tmp".to_string())
}

/// `scratch_dir()` joined with `name`. Convenience for the common case of one
/// cache file per (fixture, resolution, frame-count).
pub fn scratch_path(name: &str) -> String {
    format!("{}/{}", scratch_dir(), name)
}
