// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! STDM (Spread Transform Dither Modulation) embedding and extraction.
//!
//! Embeds one message bit per embedding unit of L coefficients using
//! dither-quantized projections onto spreading vectors. The quantization
//! step `delta` controls robustness vs. distortion.

use super::spreading::SPREAD_LEN;
use crate::jpeg::zigzag::NATURAL_TO_ZIGZAG;

/// Fixed bootstrap delta for the header region.
///
/// This constant is used to embed/extract the mean-QT byte at the start
/// of the embedding stream. It must be robust enough to survive recompression
/// (56 units with 7× redundancy).
pub const BOOTSTRAP_DELTA: f64 = 100.0;

/// Maximum zigzag position for frequency-restricted embedding.
/// Positions 1..=MAX_ARMOR_ZIGZAG are used; higher frequencies are excluded
/// to prevent pixel clamping issues during recompression.
pub const MAX_ARMOR_ZIGZAG: usize = 15;

/// Mean of actual QT values at zigzag positions 1..=MAX_ARMOR_ZIGZAG.
pub fn compute_mean_qt(qt_values: &[u16; 64]) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for nat_idx in 0..64 {
        let zz = NATURAL_TO_ZIGZAG[nat_idx];
        if zz >= 1 && zz <= MAX_ARMOR_ZIGZAG {
            sum += qt_values[nat_idx] as f64;
            count += 1;
        }
    }
    if count == 0 {
        return 10.0; // fallback
    }
    sum / count as f64
}

/// Encode mean QT as a header byte: round(mean * 4).clamp(1, 255).
pub fn encode_mean_qt(mean_qt: f64) -> u8 {
    (mean_qt * 4.0).round().clamp(1.0, 255.0) as u8
}

/// Decode header byte back to mean QT: byte / 4.0.
pub fn decode_mean_qt(header_byte: u8) -> f64 {
    header_byte as f64 / 4.0
}

/// Number of header bytes (mean QT byte).
pub const HEADER_BYTES: usize = 1;

/// Number of embedding units for the header.
/// 1 byte × 8 bits × 7 copies = 56 units.
pub const HEADER_UNITS: usize = HEADER_BYTES * 8 * HEADER_COPIES;

/// Number of copies for header majority voting.
pub const HEADER_COPIES: usize = 7;

/// Compute delta from mean QT value and repetition factor.
/// Uses adaptive multipliers scaled by r for larger decision regions.
pub fn compute_delta_from_mean_qt(mean_qt: f64, r: usize) -> f64 {
    let mult = if r >= 7 {
        8.0
    } else if r >= 5 {
        7.0
    } else if r >= 3 {
        6.0
    } else if r >= 2 {
        4.0
    } else {
        3.0 // Phase 1 base: was 2.0, now 3.0
    };
    mult * mean_qt
}

/// Embed a single bit into a group of coefficients using STDM.
///
/// - `coeffs`: L coefficient values (will be modified in place)
/// - `v`: unit-norm spreading vector of length L
/// - `bit`: the message bit to embed (0 or 1)
/// - `delta`: quantization step size
pub fn stdm_embed(coeffs: &mut [f64; SPREAD_LEN], v: &[f64; SPREAD_LEN], bit: u8, delta: f64) {
    debug_assert!(bit <= 1);

    // Project onto spreading vector
    let p: f64 = coeffs.iter().zip(v.iter()).map(|(&c, &vi)| c * vi).sum();

    // Dither-quantize to encode the bit
    let q = quantize_for_bit(p, delta, bit);

    // Distribute the change along the spreading vector
    let dp = q - p;
    for i in 0..SPREAD_LEN {
        coeffs[i] += dp * v[i];
    }
}

/// Extract a single bit from a group of coefficients using STDM.
///
/// - `coeffs`: L coefficient values
/// - `v`: unit-norm spreading vector of length L (same as used for embedding)
/// - `delta`: quantization step size (same as used for embedding)
///
/// Returns the extracted bit (0 or 1).
#[cfg(test)]
pub fn stdm_extract(coeffs: &[f64; SPREAD_LEN], v: &[f64; SPREAD_LEN], delta: f64) -> u8 {
    let p: f64 = coeffs.iter().zip(v.iter()).map(|(&c, &vi)| c * vi).sum();

    // Determine which quantizer lattice is closest
    let half_delta = delta / 2.0;
    let m = (p / half_delta).round() as i64;
    m.rem_euclid(2) as u8
}

/// Quantize `p` to the nearest point in the Q_b lattice.
///
/// - Q_0: centers at {n * delta} for integer n
/// - Q_1: centers at {(n + 0.5) * delta} for integer n
fn quantize_for_bit(p: f64, delta: f64, bit: u8) -> f64 {
    if bit == 0 {
        (p / delta).round() * delta
    } else {
        ((p / delta - 0.5).round() + 0.5) * delta
    }
}

/// Extract a bit with soft confidence (log-likelihood ratio).
///
/// Positive LLR → bit 0 more likely, negative → bit 1 more likely.
/// |LLR| magnitude indicates confidence.
pub fn stdm_extract_soft(coeffs: &[f64; SPREAD_LEN], v: &[f64; SPREAD_LEN], delta: f64) -> f64 {
    let p: f64 = coeffs.iter().zip(v.iter()).map(|(&c, &vi)| c * vi).sum();

    // Distance to nearest Q_0 lattice point
    let q0 = (p / delta).round() * delta;
    let d0 = (p - q0).abs();

    // Distance to nearest Q_1 lattice point
    let q1 = ((p / delta - 0.5).round() + 0.5) * delta;
    let d1 = (p - q1).abs();

    // LLR: positive favors bit 0, negative favors bit 1
    d1 - d0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spreading_vec() -> [f64; SPREAD_LEN] {
        // Simple normalized vector
        let raw = [1.0, 0.5, -0.3, 0.7, -0.2, 0.4, 0.6, -0.1];
        let norm: f64 = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut v = [0.0; SPREAD_LEN];
        for i in 0..SPREAD_LEN {
            v[i] = raw[i] / norm;
        }
        v
    }

    #[test]
    fn embed_extract_roundtrip_bit0() {
        let v = make_spreading_vec();
        let delta = 10.0;
        let mut coeffs = [20.0, 15.0, -8.0, 30.0, -5.0, 10.0, 25.0, -3.0];

        stdm_embed(&mut coeffs, &v, 0, delta);
        let extracted = stdm_extract(&coeffs, &v, delta);
        assert_eq!(extracted, 0);
    }

    #[test]
    fn embed_extract_roundtrip_bit1() {
        let v = make_spreading_vec();
        let delta = 10.0;
        let mut coeffs = [20.0, 15.0, -8.0, 30.0, -5.0, 10.0, 25.0, -3.0];

        stdm_embed(&mut coeffs, &v, 1, delta);
        let extracted = stdm_extract(&coeffs, &v, delta);
        assert_eq!(extracted, 1);
    }

    #[test]
    fn embed_extract_many_bits() {
        let v = make_spreading_vec();
        let delta = 8.0;

        for bit in 0..=1 {
            for base in [-50.0, -10.0, 0.0, 10.0, 50.0] {
                let mut coeffs = [base; SPREAD_LEN];
                stdm_embed(&mut coeffs, &v, bit, delta);
                let extracted = stdm_extract(&coeffs, &v, delta);
                assert_eq!(extracted, bit, "failed for bit={bit}, base={base}");
            }
        }
    }

    #[test]
    fn survives_small_perturbation() {
        let v = make_spreading_vec();
        let delta = 16.0; // Large delta for more robustness

        for bit in 0..=1 {
            let mut coeffs = [20.0, -10.0, 5.0, 30.0, -15.0, 8.0, 12.0, -6.0];
            stdm_embed(&mut coeffs, &v, bit, delta);

            // Add small perturbation (simulating quantization noise)
            for c in coeffs.iter_mut() {
                *c += 0.3;
            }

            let extracted = stdm_extract(&coeffs, &v, delta);
            assert_eq!(extracted, bit, "failed for bit={bit} after perturbation");
        }
    }

    #[test]
    fn quantize_for_bit_correct() {
        let delta = 10.0;

        // For bit=0, should quantize to nearest multiple of delta
        assert!((quantize_for_bit(7.0, delta, 0) - 10.0).abs() < 1e-10);
        assert!((quantize_for_bit(3.0, delta, 0) - 0.0).abs() < 1e-10);
        assert!((quantize_for_bit(-7.0, delta, 0) - -10.0).abs() < 1e-10);

        // For bit=1, should quantize to nearest half-multiple of delta
        assert!((quantize_for_bit(3.0, delta, 1) - 5.0).abs() < 1e-10);
        assert!((quantize_for_bit(8.0, delta, 1) - 5.0).abs() < 1e-10);
        assert!((quantize_for_bit(12.0, delta, 1) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn compute_mean_qt_reasonable() {
        // Standard luma QT at QF 75 (approximate)
        let qt = [8, 6, 5, 8, 12, 20, 26, 31,
                   6, 6, 7, 10, 13, 29, 30, 28,
                   7, 7, 8, 12, 20, 29, 35, 28,
                   7, 9, 11, 15, 26, 44, 40, 31,
                   9, 11, 19, 28, 34, 55, 52, 39,
                   12, 18, 28, 32, 41, 52, 57, 46,
                   25, 32, 39, 44, 52, 61, 60, 51,
                   36, 46, 48, 49, 56, 50, 52, 50];
        let mean = compute_mean_qt(&qt);
        // Mean of low-freq AC positions should be reasonable
        assert!(mean > 5.0 && mean < 30.0, "mean_qt={mean}");
    }

    #[test]
    fn mean_qt_encode_decode_roundtrip() {
        for qt_val in [5.0, 10.0, 15.5, 25.0, 50.0, 63.0] {
            let encoded = encode_mean_qt(qt_val);
            let decoded = decode_mean_qt(encoded);
            assert!((decoded - qt_val).abs() < 0.5, "roundtrip failed: {qt_val} -> {encoded} -> {decoded}");
        }
    }

    #[test]
    fn soft_extract_sign_matches_hard_extract() {
        let v = make_spreading_vec();
        let delta = 10.0;

        for bit in 0..=1 {
            let mut coeffs = [20.0, 15.0, -8.0, 30.0, -5.0, 10.0, 25.0, -3.0];
            stdm_embed(&mut coeffs, &v, bit, delta);

            let llr = stdm_extract_soft(&coeffs, &v, delta);
            let hard_bit = stdm_extract(&coeffs, &v, delta);

            // Positive LLR → bit 0, negative → bit 1
            let soft_bit = if llr >= 0.0 { 0u8 } else { 1u8 };
            assert_eq!(soft_bit, hard_bit, "bit={bit}, llr={llr}");
            assert_eq!(soft_bit, bit, "bit={bit}, llr={llr}");
        }
    }

    #[test]
    fn soft_extract_confidence_decreases_with_noise() {
        let v = make_spreading_vec();
        let delta = 16.0;
        let mut coeffs = [20.0, -10.0, 5.0, 30.0, -15.0, 8.0, 12.0, -6.0];
        stdm_embed(&mut coeffs, &v, 0, delta);

        let llr_clean = stdm_extract_soft(&coeffs, &v, delta);
        assert!(llr_clean > 0.0, "should favor bit 0");

        // Add noise
        let mut noisy = coeffs;
        for c in noisy.iter_mut() {
            *c += 2.0;
        }
        let llr_noisy = stdm_extract_soft(&noisy, &v, delta);
        // Confidence may decrease or stay, but should still likely favor bit 0
        assert!(llr_clean.abs() >= llr_noisy.abs() - 1.0, "noise should not increase confidence dramatically");
    }

    #[test]
    fn header_units_constant_correct() {
        assert_eq!(HEADER_UNITS, HEADER_BYTES * 8 * HEADER_COPIES);
        assert_eq!(HEADER_UNITS, 56);
    }

    #[test]
    fn delta_increases_with_r() {
        let mean_qt = 10.0;
        let d1 = compute_delta_from_mean_qt(mean_qt, 1);
        let d2 = compute_delta_from_mean_qt(mean_qt, 2);
        let d3 = compute_delta_from_mean_qt(mean_qt, 3);
        let d5 = compute_delta_from_mean_qt(mean_qt, 5);
        let d7 = compute_delta_from_mean_qt(mean_qt, 7);

        assert!(d2 > d1, "r=2 should increase delta");
        assert!(d3 > d2, "r=3 should increase delta more");
        assert!(d5 > d3, "r=5 should increase delta further");
        assert!(d7 > d5, "r=7 should increase delta even more");

        // Verify exact multipliers
        assert!((d1 - 30.0).abs() < 1e-10, "r=1: 3.0 * 10.0 = 30.0, got {d1}");
        assert!((d2 - 40.0).abs() < 1e-10, "r=2: 4.0 * 10.0 = 40.0, got {d2}");
        assert!((d3 - 60.0).abs() < 1e-10, "r=3: 6.0 * 10.0 = 60.0, got {d3}");
        assert!((d5 - 70.0).abs() < 1e-10, "r=5: 7.0 * 10.0 = 70.0, got {d5}");
        assert!((d7 - 80.0).abs() < 1e-10, "r=7: 8.0 * 10.0 = 80.0, got {d7}");
    }
}
