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
//! The pipeline: encrypt -> frame -> RS encode -> STDM embed per bit.

pub mod ecc;
pub mod selection;
pub mod spreading;
pub mod embedding;
pub mod capacity;
pub mod pipeline;
