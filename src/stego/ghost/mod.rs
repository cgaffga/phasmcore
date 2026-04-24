// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Ghost mode — stealth steganography for JPEG images.
//!
//! Uses J-UNIWARD cost function + Syndrome-Trellis Coding (STC) to embed
//! messages with minimal statistical detectability. Includes SI-UNIWARD
//! "Deep Cover" for non-JPEG input and shadow messages for plausible deniability.

pub mod capacity;
pub mod optimizer;
pub mod permute;
pub mod pipeline;
pub mod quality;
pub mod shadow;
pub mod side_info;
