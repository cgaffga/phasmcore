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
/// Each vector is drawn from a standard normal distribution (via Box-Muller)
/// and then normalized to unit length. This ensures the projection operation
/// in STDM is well-conditioned.
pub fn generate_spreading_vectors(seed: &[u8; 32], count: usize) -> Vec<[f64; SPREAD_LEN]> {
    let mut rng = ChaCha20Rng::from_seed(*seed);
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
    fn correct_count() {
        let vectors = generate_spreading_vectors(&[0u8; 32], 37);
        assert_eq!(vectors.len(), 37);
    }
}
