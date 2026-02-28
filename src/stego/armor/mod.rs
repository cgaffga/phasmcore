// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Armor mode: robust steganography resistant to JPEG recompression.
//!
//! Armor mode trades undetectability for robustness by using:
//! - **STDM** (Spread Transform Dither Modulation) for embedding, which
//!   distributes each bit across multiple coefficients via spreading vectors
//! - **Reed-Solomon** error correction to recover from bit errors introduced
//!   by recompression
//! - **Stability analysis** to select coefficient positions that survive
//!   quality factor changes
//!
//! **Phase 2 adaptive robustness:** When the message is small relative to the
//! image capacity, the encoder automatically maximizes robustness by:
//! - Increasing RS parity (up to 240/255 symbols, from fixed 64/255)
//! - Repeating the RS-encoded bitstream r times across spare capacity
//! - Using soft majority voting with STDM log-likelihood ratios on decode
//! - Increasing the STDM delta for larger decision regions
//!
//! Phase 2 activates transparently when r >= 3. Phase 1 images (r <= 1) are
//! decoded using the original path for full backward compatibility.
//!
//! The pipeline: encrypt -> frame -> RS encode -> [repeat r×] -> STDM embed.

pub mod ecc;
pub mod selection;
pub mod spreading;
pub mod embedding;
pub mod repetition;
pub mod capacity;
pub mod fft2d;
pub mod template;
pub mod resample;
pub mod dft_payload;
pub mod fortress;
pub mod pipeline;
