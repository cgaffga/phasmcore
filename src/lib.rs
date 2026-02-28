// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! # phasm-core
//!
//! Pure-Rust steganography engine for hiding encrypted text messages in JPEG
//! photos. Provides two embedding modes:
//!
//! - **Ghost** (stealth): J-UNIWARD cost function + STC coding to resist
//!   statistical steganalysis. Optimizes for undetectability.
//! - **Armor** (robust): STDM embedding + Reed-Solomon ECC to survive
//!   JPEG recompression. Optimizes for message survivability.
//!
//! All processing is client-side. The JPEG coefficient codec (`jpeg` module)
//! is zero-dependency (std only). The steganography layer (`stego` module)
//! uses AES-256-GCM-SIV encryption and Argon2id key derivation.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use phasm_core::{ghost_encode, ghost_decode};
//!
//! let cover_jpeg = std::fs::read("photo.jpg").unwrap();
//! let stego = ghost_encode(&cover_jpeg, "secret message", "passphrase").unwrap();
//! let decoded = ghost_decode(&stego, "passphrase").unwrap();
//! assert_eq!(decoded, "secret message");
//! ```

pub mod det_math;
pub mod jpeg;
pub mod stego;

pub use jpeg::error::{JpegError, Result as JpegResult};
pub use jpeg::dct::{DctGrid, QuantTable};
pub use jpeg::frame::FrameInfo;
pub use jpeg::JpegImage;
pub use stego::{ghost_encode, ghost_decode, ghost_encode_with_files, ghost_capacity, StegoError};
pub use stego::{armor_encode, armor_decode, armor_capacity, armor_capacity_info, smart_decode, DecodeQuality, ArmorCapacityInfo};
pub use stego::{validate_encode_dimensions, MAX_DIMENSION, MAX_PIXELS, MIN_ENCODE_DIMENSION, ARMOR_TARGET_DIMENSION};
pub use stego::{PayloadData, FileEntry, compressed_payload_size};
pub use stego::progress;
