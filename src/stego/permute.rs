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
//! 64-bit on native, which causes `gen_range` to consume different
//! amounts of PRNG entropy per step — producing completely different shuffles.
//! The draws use `rand` 0.8's sampler ([`crate::stego::rng_compat`]).

use crate::stego::rng_compat::Rand08;
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
        let j = rng.gen_range08(0..=(i as u32)) as usize;
        positions.swap(i, j);
    }
}

/// The same portable Fisher-Yates shuffle on bare flat indices (4 bytes per
/// position instead of `CoeffPos`'s 8): `shuffle_u32(idx)` permutes exactly
/// like `permute_positions` permutes `CoeffPos`es with those indices.
pub fn shuffle_u32(v: &mut [u32], seed: &[u8; 32]) {
    let mut rng = shuffle_rng(seed);
    let mut hi = v.len();
    while hi > 1 {
        let lo = hi.saturating_sub(SHUFFLE_CHUNK).max(1);
        shuffle_u32_steps(v, &mut rng, lo, hi);
        hi = lo;
    }
}

/// `ChaCha20Rng`'s stream; on wasm32 computed with SIMD128
/// ([`crate::stego::chacha_stream`]; `ppv_lite86` has no wasm SIMD backend).
#[cfg(target_arch = "wasm32")]
type ShuffleRng = crate::stego::chacha_stream::ChaCha20Stream;
#[cfg(not(target_arch = "wasm32"))]
type ShuffleRng = ChaCha20Rng;

fn shuffle_rng(seed: &[u8; 32]) -> ShuffleRng {
    ShuffleRng::from_seed(*seed)
}

/// Fisher-Yates steps per call of [`shuffle_u32_steps`].
const SHUFFLE_CHUNK: usize = 1 << 16;

/// The steps `i = hi − 1, …, lo` of [`shuffle_u32`]. A function of its own,
/// not inlined on wasm32: V8 runs a function in baseline code (Liftoff)
/// until its next call, with no on-stack replacement for Wasm, so a shuffle
/// of 12 M positions in one loop never left baseline code.
#[cfg_attr(target_arch = "wasm32", inline(never))]
fn shuffle_u32_steps(v: &mut [u32], rng: &mut ShuffleRng, lo: usize, hi: usize) {
    for i in (lo..hi).rev() {
        let j = rng.gen_range08(0..=(i as u32)) as usize;
        v.swap(i, j);
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
