// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Spreading vector generation for STDM embedding.
//!
//! Generates unit-norm pseudo-random spreading vectors from a ChaCha20 PRNG
//! seeded by the structural key. Each vector maps one message bit to a group
//! of L DCT coefficients.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

/// Number of DCT coefficients per embedding unit (spreading vector length).
pub const SPREAD_LEN: usize = 8;

/// Generate `count` spreading vectors of length `SPREAD_LEN` from a seed.
///
/// Each vector is drawn uniformly from [-1, 1)^L and then normalized to unit
/// length. This ensures the projection operation in STDM is well-conditioned.
pub fn generate_spreading_vectors(seed: &[u8; 32], count: usize) -> Vec<[f64; SPREAD_LEN]> {
    spreading_vectors_range(seed, 0, count)
}

/// ChaCha20 keystream words one vector consumes.
///
/// `gen_range(-1.0..1.0)` on f64 draws exactly one `u64` (two words) per
/// value and never rejects: rand maps the draw to `v * 2.0 - 1.0` with
/// `v` in `[0, 1 - 2^-52]`, which is exact and at most `1 - 2^-51 < 1`.
/// So vector `i` is a pure function of keystream words `16 i .. 16 i + 16`
/// (one ChaCha20 block), and any vector can be generated on its own.
const WORDS_PER_VECTOR: u128 = 2 * SPREAD_LEN as u128;

/// Vectors `start .. start + count` of the sequence `generate_spreading_vectors`
/// produces, bit for bit, without generating the ones before them. Lets the
/// decoder regenerate vectors chunk by chunk instead of holding all of them.
pub fn spreading_vectors_range(seed: &[u8; 32], start: usize, count: usize) -> Vec<[f64; SPREAD_LEN]> {
    let mut rng = ChaCha20Rng::from_seed(*seed);
    rng.set_word_pos(start as u128 * WORDS_PER_VECTOR);
    let mut vectors = Vec::with_capacity(count);

    for _ in 0..count {
        let mut v = [0.0f64; SPREAD_LEN];

        // Generate random values and normalize
        for val in v.iter_mut() {
            // Uniform random in [-1, 1], then normalize afterward
            *val = rng.gen_range(-1.0..1.0);
        }

        // Normalize to unit length
        // sqrt() is IEEE 754 correctly-rounded — deterministic across all platforms
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for val in v.iter_mut() {
                *val /= norm;
            }
        } else {
            // Extremely unlikely, but set first element to 1
            v[0] = 1.0;
        }

        vectors.push(v);
    }

    vectors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_are_unit_norm() {
        let vectors = generate_spreading_vectors(&[42u8; 32], 100);
        for (i, v) in vectors.iter().enumerate() {
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "vector {i} has norm {norm}"
            );
        }
    }

    #[test]
    fn deterministic() {
        let a = generate_spreading_vectors(&[7u8; 32], 50);
        let b = generate_spreading_vectors(&[7u8; 32], 50);
        for (va, vb) in a.iter().zip(b.iter()) {
            for (ea, eb) in va.iter().zip(vb.iter()) {
                assert_eq!(ea.to_bits(), eb.to_bits());
            }
        }
    }

    #[test]
    fn different_seeds_differ() {
        let a = generate_spreading_vectors(&[1u8; 32], 10);
        let b = generate_spreading_vectors(&[2u8; 32], 10);
        assert_ne!(
            a.iter().map(|v| v[0].to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v[0].to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn range_matches_the_sequence() {
        let seed = [9u8; 32];
        let all = generate_spreading_vectors(&seed, 3000);
        for &(start, count) in &[(0usize, 1usize), (1, 5), (3, 64), (777, 100), (2999, 1), (1024, 1976)] {
            let part = spreading_vectors_range(&seed, start, count);
            for (k, v) in part.iter().enumerate() {
                for (a, b) in v.iter().zip(all[start + k].iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "vector {}", start + k);
                }
            }
        }
    }

    #[test]
    fn correct_count() {
        let vectors = generate_spreading_vectors(&[0u8; 32], 37);
        assert_eq!(vectors.len(), 37);
    }
}
