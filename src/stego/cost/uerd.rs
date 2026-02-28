// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! UERD (Uniform Embedding Revisited Distortion) cost function for steganography.

use crate::jpeg::dct::{DctGrid, QuantTable};
use super::CostMap;

const EPSILON: f64 = 1e-10;

/// Compute UERD embedding costs for a single component's DCT grid.
///
/// UERD (Uniform Embedding Revisited Distortion) assigns lower cost to
/// coefficients in textured blocks and at higher frequencies — positions
/// where ±1 changes are least detectable.
///
/// DC coefficients and zero-valued AC coefficients receive `WET_COST`
/// (effectively infinite) so the STC will never modify them.
pub fn compute_uerd(grid: &DctGrid, qt: &QuantTable) -> CostMap {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let mut map = CostMap::new(bw, bt);

    for br in 0..bt {
        for bc in 0..bw {
            let blk = grid.block(br, bc);

            // Block energy: sum of squared AC coefficients (DC excluded).
            let energy: f64 = blk[1..]
                .iter()
                .map(|&c| (c as f64) * (c as f64))
                .sum();

            for i in 0..8 {
                for j in 0..8 {
                    // DC coefficient — never modify.
                    if i == 0 && j == 0 {
                        continue; // already WET_COST from CostMap::new
                    }

                    let coeff = blk[i * 8 + j];

                    // Zero AC coefficient — never modify (prevents shrinkage alignment issues).
                    if coeff == 0 {
                        continue; // already WET_COST
                    }

                    let q = qt.values[i * 8 + j] as f64;
                    // Higher frequency → larger factor → larger denominator → lower cost.
                    let freq_factor = (i + j + 1) as f64;
                    let cost = q / (energy * freq_factor + EPSILON);

                    map.set(br, bc, i, j, cost as f32);
                }
            }
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stego::cost::WET_COST;

    fn make_qt_uniform(val: u16) -> QuantTable {
        QuantTable::new([val; 64])
    }

    #[test]
    fn dc_is_wet() {
        let mut grid = DctGrid::new(1, 1);
        grid.set(0, 0, 0, 0, 100);
        grid.set(0, 0, 1, 0, 10);
        let map = compute_uerd(&grid, &make_qt_uniform(16));
        assert_eq!(map.get(0, 0, 0, 0), WET_COST);
    }

    #[test]
    fn zero_ac_is_wet() {
        let mut grid = DctGrid::new(1, 1);
        grid.set(0, 0, 0, 0, 100);
        grid.set(0, 0, 1, 0, 10);
        // (0, 1) is zero — should be WET
        let map = compute_uerd(&grid, &make_qt_uniform(16));
        assert_eq!(map.get(0, 0, 0, 1), WET_COST);
    }

    #[test]
    fn textured_block_cheaper_than_smooth() {
        let qt = make_qt_uniform(16);

        // Textured block: large AC coefficients → high energy → lower cost
        let mut textured = DctGrid::new(2, 1);
        for i in 0..8 {
            for j in 0..8 {
                if i == 0 && j == 0 {
                    textured.set(0, 0, i, j, 100);
                } else {
                    textured.set(0, 0, i, j, 20);
                }
            }
        }
        // Smooth block: small AC coefficients → low energy → higher cost
        for i in 0..8 {
            for j in 0..8 {
                if i == 0 && j == 0 {
                    textured.set(0, 1, i, j, 100);
                } else {
                    textured.set(0, 1, i, j, 1);
                }
            }
        }

        let map = compute_uerd(&textured, &qt);
        let cost_textured = map.get(0, 0, 1, 0);
        let cost_smooth = map.get(0, 1, 1, 0);
        assert!(
            cost_textured < cost_smooth,
            "textured {cost_textured} should be < smooth {cost_smooth}"
        );
    }

    #[test]
    fn high_freq_cheaper_than_low_freq() {
        let qt = make_qt_uniform(16);
        let mut grid = DctGrid::new(1, 1);
        grid.set(0, 0, 0, 0, 100);
        // Set nonzero coefficients at both positions
        grid.set(0, 0, 1, 0, 10); // low freq (i+j=1)
        grid.set(0, 0, 7, 7, 10); // high freq (i+j=14)

        let map = compute_uerd(&grid, &qt);
        let cost_low = map.get(0, 0, 1, 0);
        let cost_high = map.get(0, 0, 7, 7);
        assert!(
            cost_high < cost_low,
            "high-freq {cost_high} should be < low-freq {cost_low}"
        );
    }
}
