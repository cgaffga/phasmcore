//! TCM (Transport Channel Model) coefficient stability testing.
//!
//! Simulates JPEG recompression at multiple quality factors to identify
//! DCT coefficients that are likely to survive. Used by Armor mode to
//! select robust embedding positions.

use crate::jpeg::dct::{DctGrid, QuantTable};
use crate::stego::cost::CostMap;

/// Quality factors to test stability against.
const TEST_QFS: &[u8] = &[65, 70, 75, 80, 85];

/// Minimum fraction of QFs a coefficient must survive to be considered stable.
const STABILITY_THRESHOLD: f64 = 0.4;

/// Minimum absolute coefficient value to be considered for embedding.
/// A value of 1 includes all non-zero coefficients (same pool as Ghost mode).
/// Zero coefficients are always excluded.
const MIN_COEFF_ABS: i16 = 1;

/// Cost assigned to stable positions (low, uniform cost for permutation compatibility).
const STABLE_COST: f64 = 1.0;

/// Whether to apply TCM stability testing. When false, only the coefficient
/// magnitude filter is used (faster, more positions, relies on STDM + RS robustness).
const USE_TCM: bool = false;

/// Standard JPEG luminance quantization table (ITU-T T.81, Annex K).
const STD_LUMA_QT: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
];

/// Standard JPEG chrominance quantization table (ITU-T T.81, Annex K).
const STD_CHROMA_QT: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];

/// Compute a scaled quantization table for a given quality factor.
///
/// Uses the IJG (Independent JPEG Group) quality factor formula:
/// - QF < 50: scale = 5000 / QF
/// - QF >= 50: scale = 200 - 2 * QF
/// - q_value = max(1, (std_value * scale + 50) / 100)
fn scale_quant_table(std_qt: &[u16; 64], qf: u8) -> [u16; 64] {
    let scale = if qf < 50 {
        5000u32 / qf as u32
    } else {
        200u32 - 2 * qf as u32
    };

    let mut qt = [0u16; 64];
    for i in 0..64 {
        let val = (std_qt[i] as u32 * scale + 50) / 100;
        qt[i] = val.clamp(1, 255) as u16;
    }
    qt
}

/// Detect whether the original quant table is closer to standard luma or chroma.
fn is_chroma_table(qt: &QuantTable) -> bool {
    // Compute sum-of-absolute-differences to both standard tables
    let luma_diff: u32 = qt
        .values
        .iter()
        .zip(STD_LUMA_QT.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
        .sum();
    let chroma_diff: u32 = qt
        .values
        .iter()
        .zip(STD_CHROMA_QT.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
        .sum();
    chroma_diff < luma_diff
}

/// Simulate recompression of a single coefficient at a given QF.
///
/// Returns the coefficient value after one dequantize→requantize cycle.
fn simulate_recompress(coeff: i16, q_orig: u16, q_target: u16) -> i16 {
    // Dequantize: recover approximate DCT value
    let dct_value = coeff as i32 * q_orig as i32;
    // Requantize at target QF
    let recompressed = if dct_value >= 0 {
        (dct_value + q_target as i32 / 2) / q_target as i32
    } else {
        (dct_value - q_target as i32 / 2) / q_target as i32
    };
    // Dequantize again at target QF, then requantize at original QF
    let dct_value2 = recompressed * q_target as i32;
    if dct_value2 >= 0 {
        ((dct_value2 + q_orig as i32 / 2) / q_orig as i32) as i16
    } else {
        ((dct_value2 - q_orig as i32 / 2) / q_orig as i32) as i16
    }
}

/// Compute a stability map for the given DCT grid and quantization table.
///
/// Tests each AC coefficient position against recompression at multiple QFs.
/// Returns a `CostMap` where stable positions have `STABLE_COST` and
/// unstable/DC/zero positions have `WET_COST`.
pub fn compute_stability_map(grid: &DctGrid, qt: &QuantTable) -> CostMap {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let mut cost_map = CostMap::new(bw, bt);

    // Determine the standard reference table
    let std_qt = if is_chroma_table(qt) {
        &STD_CHROMA_QT
    } else {
        &STD_LUMA_QT
    };

    // Precompute target QTs
    let target_qts: Vec<[u16; 64]> = TEST_QFS.iter().map(|&qf| scale_quant_table(std_qt, qf)).collect();
    let num_qfs = target_qts.len() as f64;
    let min_stable = (STABILITY_THRESHOLD * num_qfs).ceil() as usize;

    for br in 0..bt {
        for bc in 0..bw {
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 {
                        continue; // skip DC
                    }

                    if !USE_TCM {
                        // Include ALL AC positions for deterministic encode/decode sync.
                        // STDM modifies coefficients, so filtering by value would
                        // produce different position lists on encode vs decode.
                        cost_map.set(br, bc, i, j, STABLE_COST);
                        continue;
                    }

                    let coeff = grid.get(br, bc, i, j);
                    if coeff.unsigned_abs() < MIN_COEFF_ABS as u16 {
                        continue; // skip zero and near-zero coefficients
                    }

                    let freq_idx = i * 8 + j;
                    let q_orig = qt.values[freq_idx];

                    // Count how many QFs preserve this coefficient
                    let mut stable_count = 0usize;
                    for target_qt in &target_qts {
                        let q_target = target_qt[freq_idx];
                        let recompressed = simulate_recompress(coeff, q_orig, q_target);
                        if recompressed == coeff {
                            stable_count += 1;
                        }
                    }

                    if stable_count >= min_stable {
                        cost_map.set(br, bc, i, j, STABLE_COST);
                    }
                    // else: stays WET_COST (default)
                }
            }
        }
    }

    cost_map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stego::cost::WET_COST;

    #[test]
    fn scale_quant_table_qf75() {
        let qt = scale_quant_table(&STD_LUMA_QT, 75);
        // At QF 75, scale = 200 - 150 = 50.
        // For position [0,0] = 16: (16 * 50 + 50) / 100 = 8
        assert_eq!(qt[0], 8);
        // For position [0,1] = 11: (11 * 50 + 50) / 100 = 6
        assert_eq!(qt[1], 6);
    }

    #[test]
    fn scale_quant_table_qf50() {
        let qt = scale_quant_table(&STD_LUMA_QT, 50);
        // At QF 50, scale = 200 - 100 = 100. Values should equal std table.
        for i in 0..64 {
            assert_eq!(qt[i], STD_LUMA_QT[i], "mismatch at position {i}");
        }
    }

    #[test]
    fn scale_quant_table_qf25() {
        let qt = scale_quant_table(&STD_LUMA_QT, 25);
        // At QF 25, scale = 5000/25 = 200.
        // For position [0,0] = 16: (16 * 200 + 50) / 100 = 32
        assert_eq!(qt[0], 32);
    }

    #[test]
    fn simulate_recompress_stable() {
        // Large coefficient with small quant step difference should survive
        let coeff: i16 = 10;
        let q_orig: u16 = 8;
        let q_target: u16 = 8; // same quant step → always stable
        assert_eq!(simulate_recompress(coeff, q_orig, q_target), coeff);
    }

    #[test]
    fn simulate_recompress_unstable() {
        // Small coefficient with large quant step change likely becomes 0
        let coeff: i16 = 1;
        let q_orig: u16 = 8;
        let q_target: u16 = 50;
        let result = simulate_recompress(coeff, q_orig, q_target);
        // 1 * 8 = 8. 8 / 50 = 0. 0 * 50 / 8 = 0 → not stable
        assert_ne!(result, coeff);
    }

    #[test]
    fn stability_map_dc_is_wet_ac_is_stable() {
        let mut grid = DctGrid::new(2, 2);
        let qt = QuantTable::new([8; 64]);

        // Set only DC and leave AC as zero
        grid.set(0, 0, 0, 0, 100);

        let cost_map = compute_stability_map(&grid, &qt);
        // With USE_TCM=false, all AC positions are STABLE (for encode/decode sync)
        // Only DC (0,0) is WET
        for br in 0..2 {
            for bc in 0..2 {
                // DC should be WET
                assert_eq!(
                    cost_map.get(br, bc, 0, 0),
                    WET_COST,
                    "DC ({br},{bc},0,0) should be WET"
                );
                // AC should be STABLE
                assert_eq!(
                    cost_map.get(br, bc, 1, 0),
                    STABLE_COST,
                    "AC ({br},{bc},1,0) should be STABLE"
                );
            }
        }
    }

    #[test]
    fn stability_map_includes_all_ac() {
        let mut grid = DctGrid::new(1, 1);
        let qt = QuantTable::new([8; 64]);

        // Set some coefficients (values don't matter with USE_TCM=false)
        grid.set(0, 0, 1, 0, 50);
        grid.set(0, 0, 0, 1, 1);

        let cost_map = compute_stability_map(&grid, &qt);

        // With USE_TCM=false, ALL AC positions are STABLE regardless of value
        assert_eq!(cost_map.get(0, 0, 1, 0), STABLE_COST);
        assert_eq!(cost_map.get(0, 0, 0, 1), STABLE_COST);
        // Even zero AC positions should be STABLE
        assert_eq!(cost_map.get(0, 0, 3, 3), STABLE_COST);
        // DC should be WET
        assert_eq!(cost_map.get(0, 0, 0, 0), WET_COST);
    }

    #[test]
    fn is_chroma_detection() {
        let luma_qt = QuantTable::new(STD_LUMA_QT);
        assert!(!is_chroma_table(&luma_qt));

        let chroma_qt = QuantTable::new(STD_CHROMA_QT);
        assert!(is_chroma_table(&chroma_qt));
    }
}
