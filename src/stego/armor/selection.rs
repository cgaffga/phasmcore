// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Frequency-based coefficient selection for Armor embedding.
//!
//! Selects DCT coefficient positions suitable for robust STDM embedding
//! based on zigzag frequency index. Low-to-mid frequency AC positions
//! (zigzag 1..=MAX_ARMOR_ZIGZAG) are selected; DC and high-frequency
//! positions are excluded.

use crate::jpeg::dct::DctGrid;
use crate::jpeg::zigzag::NATURAL_TO_ZIGZAG;
use crate::stego::cost::CostMap;
use super::embedding::MAX_ARMOR_ZIGZAG;

/// Cost assigned to stable positions (low, uniform cost for permutation compatibility).
const STABLE_COST: f32 = 1.0;

/// Compute a stability map for the given DCT grid.
///
/// Includes all AC coefficient positions with zigzag index 1..=MAX_ARMOR_ZIGZAG.
/// Returns a `CostMap` where selected positions have `STABLE_COST` and
/// DC/high-frequency positions have `WET_COST`.
pub fn compute_stability_map(grid: &DctGrid, _qt: &crate::jpeg::dct::QuantTable) -> CostMap {
    compute_stability_map_freq_only(grid)
}

/// Frequency-only selection: include all zigzag 1..=MAX_ARMOR_ZIGZAG positions.
fn compute_stability_map_freq_only(grid: &DctGrid) -> CostMap {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let mut cost_map = CostMap::new(bw, bt);

    for br in 0..bt {
        for bc in 0..bw {
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 { continue; }
                    let freq_idx = i * 8 + j;
                    let zz = NATURAL_TO_ZIGZAG[freq_idx];
                    if zz >= 1 && zz <= MAX_ARMOR_ZIGZAG {
                        cost_map.set(br, bc, i, j, STABLE_COST);
                    }
                }
            }
        }
    }

    cost_map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::dct::{DctGrid, QuantTable};
    use crate::stego::cost::WET_COST;

    #[test]
    fn stability_map_dc_is_wet_lowfreq_ac_is_stable() {
        // Small grid (2x2 = 4 blocks).
        let mut grid = DctGrid::new(2, 2);
        let qt = QuantTable::new([8; 64]);

        grid.set(0, 0, 0, 0, 100);

        let cost_map = compute_stability_map(&grid, &qt);
        for br in 0..2 {
            for bc in 0..2 {
                // DC should be WET
                assert_eq!(
                    cost_map.get(br, bc, 0, 0),
                    WET_COST,
                    "DC ({br},{bc},0,0) should be WET"
                );
                // Low-freq AC (0,1) = zigzag 1 -> STABLE (frequency-only mode)
                assert_eq!(
                    cost_map.get(br, bc, 0, 1),
                    STABLE_COST,
                    "AC ({br},{bc},0,1) should be STABLE (zigzag 1)"
                );
                // (1,0) = zigzag 2 -> STABLE
                assert_eq!(
                    cost_map.get(br, bc, 1, 0),
                    STABLE_COST,
                    "AC ({br},{bc},1,0) should be STABLE (zigzag 2)"
                );
            }
        }
    }

    #[test]
    fn stability_map_excludes_high_freq() {
        // Small grid (1x1 = 1 block).
        let mut grid = DctGrid::new(1, 1);
        let qt = QuantTable::new([8; 64]);

        grid.set(0, 0, 1, 0, 50);
        grid.set(0, 0, 0, 1, 1);

        let cost_map = compute_stability_map(&grid, &qt);

        // Low-freq AC positions should be STABLE (frequency-only selection)
        assert_eq!(cost_map.get(0, 0, 1, 0), STABLE_COST); // zigzag 2
        assert_eq!(cost_map.get(0, 0, 0, 1), STABLE_COST); // zigzag 1
        // Mid-freq AC should also be STABLE (zigzag 11, 12 within 1..=15)
        assert_eq!(cost_map.get(0, 0, 3, 1), STABLE_COST, "zigzag 11 should be STABLE");
        assert_eq!(cost_map.get(0, 0, 2, 2), STABLE_COST, "zigzag 12 should be STABLE");
        // High-freq AC (7,7) = zigzag 63 -> WET (excluded, beyond MAX_ARMOR_ZIGZAG)
        assert_eq!(cost_map.get(0, 0, 7, 7), WET_COST);
        // DC should be WET
        assert_eq!(cost_map.get(0, 0, 0, 0), WET_COST);
    }
}
