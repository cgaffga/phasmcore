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

/// Parameters for the STC codec.
pub struct StcParams {
    /// Submatrix height (constraint length). h=7 gives 128 Viterbi states.
    pub h: usize,
    /// Submatrix width: floor(n / m) where n = cover length, m = message length.
    pub w: usize,
}
