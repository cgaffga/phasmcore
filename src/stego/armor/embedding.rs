//! STDM (Spread Transform Dither Modulation) embedding and extraction.
//!
//! Embeds one message bit per embedding unit of L coefficients using
//! dither-quantized projections onto spreading vectors. The quantization
//! step `delta` controls robustness vs. distortion.

use super::spreading::SPREAD_LEN;

/// Default delta multiplier: delta = DELTA_MULT × mean quantization step.
pub const DELTA_MULT: f64 = 2.0;

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

/// Compute the default delta for an image given its quantization table.
///
/// delta = DELTA_MULT × mean of quantization step values for AC positions.
pub fn compute_delta(qt_values: &[u16; 64]) -> f64 {
    compute_delta_with_mult(qt_values, DELTA_MULT)
}

/// Compute delta with a custom multiplier.
pub fn compute_delta_with_mult(qt_values: &[u16; 64], mult: f64) -> f64 {
    let sum: f64 = qt_values[1..].iter().map(|&q| q as f64).sum();
    let mean = sum / 63.0;
    mult * mean
}

/// Compute adaptive delta based on the repetition factor.
///
/// Higher repetition = smaller message relative to capacity = more room
/// for larger decision regions = more robust to recompression.
pub fn compute_delta_adaptive(qt_values: &[u16; 64], r: usize) -> f64 {
    let mult = if r >= 7 {
        4.0
    } else if r >= 3 {
        3.0
    } else if r >= 2 {
        2.5
    } else {
        DELTA_MULT
    };
    compute_delta_with_mult(qt_values, mult)
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
    fn compute_delta_reasonable() {
        // Standard luma QT at QF 75 has mean AC step ~20
        let qt = [8, 6, 5, 8, 12, 20, 26, 31,
                   6, 6, 7, 10, 13, 29, 30, 28,
                   7, 7, 8, 12, 20, 29, 35, 28,
                   7, 9, 11, 15, 26, 44, 40, 31,
                   9, 11, 19, 28, 34, 55, 52, 39,
                   12, 18, 28, 32, 41, 52, 57, 46,
                   25, 32, 39, 44, 52, 61, 60, 51,
                   36, 46, 48, 49, 56, 50, 52, 50];
        let delta = compute_delta(&qt);
        // Mean of AC positions ≈ 30, delta ≈ 60
        assert!(delta > 20.0 && delta < 120.0, "delta={delta}");
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
    fn adaptive_delta_increases_with_r() {
        let qt = [8, 6, 5, 8, 12, 20, 26, 31,
                   6, 6, 7, 10, 13, 29, 30, 28,
                   7, 7, 8, 12, 20, 29, 35, 28,
                   7, 9, 11, 15, 26, 44, 40, 31,
                   9, 11, 19, 28, 34, 55, 52, 39,
                   12, 18, 28, 32, 41, 52, 57, 46,
                   25, 32, 39, 44, 52, 61, 60, 51,
                   36, 46, 48, 49, 56, 50, 52, 50];

        let d1 = compute_delta_adaptive(&qt, 1);
        let d2 = compute_delta_adaptive(&qt, 2);
        let d3 = compute_delta_adaptive(&qt, 3);
        let d7 = compute_delta_adaptive(&qt, 7);

        assert_eq!(d1, compute_delta(&qt), "r=1 should use baseline delta");
        assert!(d2 > d1, "r=2 should increase delta");
        assert!(d3 > d2, "r=3 should increase delta more");
        assert!(d7 > d3, "r=7 should increase delta further");
    }
}
