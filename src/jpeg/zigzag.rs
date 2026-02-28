// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Zigzag scan order mapping between JPEG coefficient order and natural order.

/// Maps zigzag index (0–63) to natural row-major index (0–63).
///
/// JPEG stores DCT coefficients in zigzag order. This table converts
/// a zigzag position to the corresponding (row * 8 + col) position.
pub const ZIGZAG_TO_NATURAL: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// Maps natural row-major index (0–63) to zigzag index (0–63).
///
/// Inverse of [`ZIGZAG_TO_NATURAL`].
pub const NATURAL_TO_ZIGZAG: [usize; 64] = {
    let mut table = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        table[ZIGZAG_TO_NATURAL[i]] = i;
        i += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        for i in 0..64 {
            assert_eq!(NATURAL_TO_ZIGZAG[ZIGZAG_TO_NATURAL[i]], i);
            assert_eq!(ZIGZAG_TO_NATURAL[NATURAL_TO_ZIGZAG[i]], i);
        }
    }

    #[test]
    fn known_positions() {
        // DC coefficient: zigzag 0 → natural 0 (top-left)
        assert_eq!(ZIGZAG_TO_NATURAL[0], 0);
        // Zigzag 1 → natural 1 (row 0, col 1)
        assert_eq!(ZIGZAG_TO_NATURAL[1], 1);
        // Zigzag 2 → natural 8 (row 1, col 0)
        assert_eq!(ZIGZAG_TO_NATURAL[2], 8);
        // Last zigzag position → natural 63 (bottom-right)
        assert_eq!(ZIGZAG_TO_NATURAL[63], 63);
    }

    #[test]
    fn all_indices_covered() {
        let mut seen = [false; 64];
        for &idx in &ZIGZAG_TO_NATURAL {
            assert!(!seen[idx], "duplicate natural index {idx}");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }
}
