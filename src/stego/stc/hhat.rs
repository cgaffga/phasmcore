// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H-hat submatrix generation for STC.
//!
//! The H-hat matrix defines the parity-check structure of the STC code.
//! It has `h` rows and `w` columns, where each column is a packed `u32`
//! with odd Hamming weight (ensuring every column contributes to the syndrome).
//!
//! The matrix is generated deterministically from a seed so encoder and
//! decoder produce identical matrices.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

/// Generate the H-hat submatrix for STC.
///
/// Returns `h` rows, each containing `w` column entries packed as `u32`.
/// Each column has odd Hamming weight (guaranteed by flipping the top bit if needed).
/// The matrix is seeded deterministically from `seed` so encoder and decoder agree.
pub fn generate_hhat(h: usize, w: usize, seed: &[u8; 32]) -> Vec<Vec<u32>> {
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let mask = (1u32 << h) - 1; // only bottom h bits matter

    let mut cols: Vec<u32> = Vec::with_capacity(w);
    for _ in 0..w {
        let mut val = rng.r#gen::<u32>() & mask;
        // Ensure odd Hamming weight so every column contributes to the syndrome.
        if val.count_ones() % 2 == 0 {
            val ^= 1; // flip lowest bit
        }
        cols.push(val);
    }

    // Transpose into h rows of w entries for easier row-wise access.
    let mut rows = vec![vec![0u32; w]; h];
    for (c, &col_val) in cols.iter().enumerate() {
        for r in 0..h {
            if col_val & (1 << r) != 0 {
                rows[r][c] = 1;
            }
        }
    }

    rows
}

/// Return the column vector at position `col` as a packed u32 (bottom h bits).
pub fn column_packed(hhat: &[Vec<u32>], col: usize) -> u32 {
    let h = hhat.len();
    let mut val = 0u32;
    for r in 0..h {
        if hhat[r][col] != 0 {
            val |= 1 << r;
        }
    }
    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let seed = [42u8; 32];
        let a = generate_hhat(7, 100, &seed);
        let b = generate_hhat(7, 100, &seed);
        assert_eq!(a, b);
    }

    #[test]
    fn odd_weight_columns() {
        let seed = [7u8; 32];
        let hhat = generate_hhat(7, 200, &seed);
        for c in 0..200 {
            let col = column_packed(&hhat, c);
            assert!(
                col.count_ones() % 2 == 1,
                "column {c} has even weight: {col:#b}"
            );
        }
    }

    #[test]
    fn different_seeds_differ() {
        let a = generate_hhat(7, 50, &[1u8; 32]);
        let b = generate_hhat(7, 50, &[2u8; 32]);
        assert_ne!(a, b);
    }
}
