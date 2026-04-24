// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Shared integration-test helpers. Each file under `tests/` is compiled
//! as its own crate, so code shared across tests goes here and is
//! included with `mod common;` + `use common::*;` in each test file.

#![allow(dead_code)] // not every test file uses every helper

pub mod h264_oracle;
