// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Syndrome-Trellis Coding (STC) for minimal-distortion embedding.
//!
//! STC encodes a message into a cover sequence by finding the minimum-cost
//! modification pattern via the Viterbi algorithm. The trellis has `2^h`
//! states, where `h` is the constraint length (typically 7, giving 128 states).
//!
//! The submatrix H-hat defines the parity-check structure. Each group of `w`
//! cover elements produces one message bit. The encoder finds the stego
//! sequence that matches all message bits with minimum total distortion cost.
//!
//! References:
//! - Filler, Judas, Fridrich. "Minimizing Additive Distortion in Steganography
//!   Using Syndrome-Trellis Codes", IEEE TIFS, 2011.

pub mod hhat;
pub mod embed;
pub mod extract;
