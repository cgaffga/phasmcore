// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Media-agnostic coefficient position permutation.
//!
//! Provides the [`CoeffPos`] type and a portable Fisher-Yates shuffle using
//! a ChaCha20 PRNG seeded from the passphrase. Both the JPEG image pipeline
//! and (future) video pipeline use this to establish a shared pseudo-random
//! embedding order between encoder and decoder.
//!
//! # Cross-platform portability
//!
//! The Fisher-Yates shuffle uses `u32` for `gen_range` (not `usize`) to ensure
//! identical permutations on all platforms. `usize` is 32-bit on WASM but
//! 64-bit on native, which causes `rand::Rng::gen_range` to consume different
//! amounts of PRNG entropy per step — producing completely different shuffles.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

// Re-export JPEG-specific select_and_permute at the original path so that
// existing code (`crate::stego::permute::select_and_permute`) continues to work.
pub use crate::stego::ghost::permute::select_and_permute;

/// A coefficient position: (flat_index, cost).
///
/// Compact representation: `u32` flat index (supports up to ~268M blocks = 4 Gpx)
/// and `f32` cost (sufficient for cost ranking). Total: 8 bytes per position
/// (down from 16 bytes with `usize` + `f64`).
#[derive(Clone)]
pub struct CoeffPos {
    /// Flat index into the coefficient grid.
    /// For JPEG: block_index * 64 + row * 8 + col.
    /// For HEVC: TU-relative index (defined by video pipeline).
    pub flat_idx: u32,
    /// Embedding cost at this position (f32 — sufficient for ranking).
    pub cost: f32,
}

/// Apply Fisher-Yates shuffle using `u32` for portable cross-platform behavior.
fn shuffle_portable(positions: &mut [CoeffPos], seed: &[u8; 32]) {
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let n = positions.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=(i as u32)) as usize;
        positions.swap(i, j);
    }
}

/// Apply a portable Fisher-Yates permutation to an externally collected
/// position vector.
///
/// Used by the streaming UNIWARD path (image) where positions are collected
/// during strip-based cost computation, and by the video pipeline where
/// positions are collected from HEVC TUs.
pub fn permute_positions(positions: &mut [CoeffPos], seed: &[u8; 32]) {
    shuffle_portable(positions, seed);
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_deterministic() {
        let mut a = vec![
            CoeffPos { flat_idx: 0, cost: 1.0 },
            CoeffPos { flat_idx: 1, cost: 2.0 },
            CoeffPos { flat_idx: 2, cost: 3.0 },
            CoeffPos { flat_idx: 3, cost: 4.0 },
        ];
        let mut b = a.clone();
        let seed = [42u8; 32];
        permute_positions(&mut a, &seed);
        permute_positions(&mut b, &seed);
        let a_idx: Vec<_> = a.iter().map(|p| p.flat_idx).collect();
        let b_idx: Vec<_> = b.iter().map(|p| p.flat_idx).collect();
        assert_eq!(a_idx, b_idx);
    }

    #[test]
    fn different_seeds_produce_different_order() {
        let mut a = (0..20).map(|i| CoeffPos { flat_idx: i, cost: 1.0 }).collect::<Vec<_>>();
        let mut b = a.clone();
        permute_positions(&mut a, &[1u8; 32]);
        permute_positions(&mut b, &[2u8; 32]);
        let a_idx: Vec<_> = a.iter().map(|p| p.flat_idx).collect();
        let b_idx: Vec<_> = b.iter().map(|p| p.flat_idx).collect();
        assert_ne!(a_idx, b_idx);
    }
}
