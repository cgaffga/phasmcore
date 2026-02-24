//! TCM (Transport Channel Model) coefficient stability testing.
//!
//! Simulates JPEG recompression at multiple quality factors to identify
//! DCT coefficients that are likely to survive. Used by Armor mode to
//! select robust embedding positions.

use crate::jpeg::dct::{DctGrid, QuantTable};
use crate::jpeg::zigzag::NATURAL_TO_ZIGZAG;
use crate::stego::cost::CostMap;
use super::embedding::MAX_ARMOR_ZIGZAG;

/// Quality factors to test stability against.
const TEST_QFS: &[u8] = &[65, 70, 75, 80, 85];

/// Minimum fraction of QFs a coefficient must survive to be considered stable.
const STABILITY_THRESHOLD: f64 = 0.4;

/// Minimum absolute coefficient value to be considered for embedding.
const MIN_COEFF_ABS: i16 = 1;

/// Cost assigned to stable positions (low, uniform cost for permutation compatibility).
const STABLE_COST: f64 = 1.0;

/// Whether to apply TCM stability testing. When true, uses pixel-domain
/// simulation to identify coefficients that survive recompression.
/// DISABLED: TCM breaks encode/decode position agreement under recompression.
/// Coefficient values change after recompression, causing TCM to select
/// different positions on decode, misaligning the entire bitstream.
/// Frequency-only selection guarantees agreement since it only checks zigzag index.
const USE_TCM: bool = false;

/// Minimum stable positions needed to use TCM results.
/// Below this, fall back to frequency-only selection.
/// Needs at least HEADER_UNITS * SPREAD_LEN + some payload capacity.
const MIN_TCM_POSITIONS: usize = 400 * 8; // ~400 units (enough for header + small payload)

/// Standard JPEG luminance quantization table (ITU-T T.81, Annex K).
pub(crate) const STD_LUMA_QT: [u16; 64] = [
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
pub(crate) const STD_CHROMA_QT: [u16; 64] = [
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
pub(crate) fn scale_quant_table(std_qt: &[u16; 64], qf: u8) -> [u16; 64] {
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
pub(crate) fn is_chroma_table(qt: &QuantTable) -> bool {
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

/// Compute a stability map for the given DCT grid and quantization table.
///
/// Tests each AC coefficient position against recompression at multiple QFs.
/// Returns a `CostMap` where stable positions have `STABLE_COST` and
/// unstable/DC/zero positions have `WET_COST`.
pub fn compute_stability_map(grid: &DctGrid, qt: &QuantTable) -> CostMap {
    if USE_TCM {
        // Try TCM first; fall back to frequency-only if too few positions.
        let tcm_map = compute_stability_map_tcm(grid, qt);
        let tcm_count = count_stable(&tcm_map);
        if tcm_count >= MIN_TCM_POSITIONS {
            return tcm_map;
        }
        // TCM produced too few positions — fall back to frequency-only.
    }
    compute_stability_map_freq_only(grid)
}

/// Count stable (non-WET) positions in a cost map.
fn count_stable(cost_map: &CostMap) -> usize {
    let mut count = 0;
    for br in 0..cost_map.blocks_tall() {
        for bc in 0..cost_map.blocks_wide() {
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 { continue; }
                    if cost_map.get(br, bc, i, j).is_finite() {
                        count += 1;
                    }
                }
            }
        }
    }
    count
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

/// TCM-based stability map with pixel-domain simulation.
fn compute_stability_map_tcm(grid: &DctGrid, qt: &QuantTable) -> CostMap {
    use crate::jpeg::pixels;

    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let mut cost_map = CostMap::new(bw, bt);

    let std_qt = if is_chroma_table(qt) {
        &STD_CHROMA_QT
    } else {
        &STD_LUMA_QT
    };

    let target_qts: Vec<[u16; 64]> = TEST_QFS.iter().map(|&qf| scale_quant_table(std_qt, qf)).collect();
    let num_qfs = target_qts.len() as f64;
    let min_stable = (STABILITY_THRESHOLD * num_qfs).ceil() as usize;

    for br in 0..bt {
        for bc in 0..bw {
            let block_slice = grid.block(br, bc);
            let quantized: [i16; 64] = block_slice.try_into().unwrap();

            let mut survival_count = [0usize; 64];

            for target_qt_vals in &target_qts {
                let mut px = pixels::idct_block(&quantized, &qt.values);
                for p in px.iter_mut() {
                    *p = p.clamp(0.0, 255.0);
                }
                let recompressed = pixels::dct_block(&px, target_qt_vals);
                for freq_idx in 0..64 {
                    let dct_value = recompressed[freq_idx] as i32 * target_qt_vals[freq_idx] as i32;
                    let q_orig = qt.values[freq_idx] as i32;
                    let settled = if dct_value >= 0 {
                        ((dct_value + q_orig / 2) / q_orig) as i16
                    } else {
                        ((dct_value - q_orig / 2) / q_orig) as i16
                    };
                    if settled == quantized[freq_idx] {
                        survival_count[freq_idx] += 1;
                    }
                }
            }

            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 { continue; }

                    let freq_idx = i * 8 + j;
                    let zz = NATURAL_TO_ZIGZAG[freq_idx];
                    if zz < 1 || zz > MAX_ARMOR_ZIGZAG { continue; }

                    let coeff = quantized[freq_idx];
                    if coeff.unsigned_abs() < MIN_COEFF_ABS as u16 { continue; }

                    if survival_count[freq_idx] >= min_stable {
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
    fn stability_map_dc_is_wet_lowfreq_ac_is_stable() {
        // Small grid (2×2 = 4 blocks × 10 freqs = 40 positions).
        // Below MIN_TCM_POSITIONS threshold → falls back to frequency-only.
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
                // Low-freq AC (0,1) = zigzag 1 → STABLE (frequency-only mode)
                assert_eq!(
                    cost_map.get(br, bc, 0, 1),
                    STABLE_COST,
                    "AC ({br},{bc},0,1) should be STABLE (zigzag 1)"
                );
                // (1,0) = zigzag 2 → STABLE
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
        // Small grid (1x1 = 1 block) -> falls back to frequency-only.
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

    #[test]
    fn tcm_small_coefficients_excluded() {
        // Test the TCM function directly (bypasses the fallback logic).
        let mut grid = DctGrid::new(1, 1);
        let qt = QuantTable::new([8; 64]);

        // Moderate DC to avoid pixel clamping
        grid.set(0, 0, 0, 0, 50);
        // Coefficients with |coeff| < MIN_COEFF_ABS (1) should be excluded.
        // Coefficient == 0 is excluded (zero check), coeff >= 1 may survive.
        grid.set(0, 0, 0, 1, 0);  // zigzag 1, zero coeff -> excluded
        grid.set(0, 0, 1, 0, 10); // zigzag 2, |coeff| >= 1 -> potentially stable

        let cost_map = compute_stability_map_tcm(&grid, &qt);

        assert_eq!(cost_map.get(0, 0, 0, 1), WET_COST, "coeff=0 should be excluded by TCM");
        assert_eq!(cost_map.get(0, 0, 1, 0), STABLE_COST, "coeff=10 should be stable in TCM");
    }

    #[test]
    fn is_chroma_detection() {
        let luma_qt = QuantTable::new(STD_LUMA_QT);
        assert!(!is_chroma_table(&luma_qt));

        let chroma_qt = QuantTable::new(STD_CHROMA_QT);
        assert!(is_chroma_table(&chroma_qt));
    }
}
