// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! STDM (Spread Transform Dither Modulation) embedding and extraction.
//!
//! Embeds one message bit per embedding unit of L coefficients using
//! dither-quantized projections onto spreading vectors. The quantization
//! step `delta` controls robustness vs. distortion.

use super::spreading::SPREAD_LEN;
use crate::codec::jpeg::zigzag::NATURAL_TO_ZIGZAG;

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
        if (1..=MAX_ARMOR_ZIGZAG).contains(&zz) {
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

/// Legacy delta multipliers for r = 1, r = 2, r 3-4, r 5-6, r >= 7
/// (the table of [`compute_delta_from_mean_qt`]).
pub const LEGACY_MULTS: [f64; 5] = [3.0, 4.0, 6.0, 7.0, 8.0];

/// Armor v2 delta multipliers for r = 1, r = 2, r 3-4, r 5-6, r >= 7
/// (x max(mean QT, V2_MEAN_QT_FLOOR), DCT units). Falls with r: repetition
/// absorbs more raw errors (eval/runs/2026-10-31-armor-v2, tune2).
pub const V2_MULTS: [f64; 5] = [5.0, 4.0, 3.5, 3.0, 2.5];

/// Armor v2 rounding-aware embed tolerance (fraction of delta).
pub const V2_ROUNDING_TOL: f64 = 0.05;

/// Armor v2 header magic (7 copies precede the 7 mean-QT copies).
pub const V2_MAGIC: u8 = 0xA2;

/// Armor v2 header units: 7 x magic byte + 7 x mean-QT byte.
pub const V2_HEADER_UNITS: usize = 2 * 8 * HEADER_COPIES;

/// Armor v2 header step, in DCT units (dequantized coefficients).
pub const V2_HEADER_DELTA: f64 = 60.0;

/// Armor v2 delta floor: delta = mult x max(mean QT, floor), so a
/// high-quality cover (small QT) still gets a step that survives a QF75-class
/// channel. 11.75 = the header value of the IJG QF65 table (mean 11.67).
pub const V2_MEAN_QT_FLOOR: f64 = 11.75;

/// Multiplier from a 5-entry table (r = 1, 2, 3-4, 5-6, >= 7).
pub fn mult_for_r(mults: &[f64; 5], r: usize) -> f64 {
    if r >= 7 {
        mults[4]
    } else if r >= 5 {
        mults[3]
    } else if r >= 3 {
        mults[2]
    } else if r >= 2 {
        mults[1]
    } else {
        mults[0]
    }
}

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
    let p = super::embedding_simd::spread_dot_product(coeffs, v);

    // Dither-quantize to encode the bit
    let q = quantize_for_bit(p, delta, bit);

    // Distribute the change along the spreading vector
    let dp = q - p;
    for i in 0..SPREAD_LEN {
        coeffs[i] += dp * v[i];
    }
}

/// How an STDM embed turns the continuous solution into integer indices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StdmRounding {
    /// Round each coefficient independently (the legacy behaviour).
    Naive,
    /// Search the floor/ceil choice of all `SPREAD_LEN` coefficients
    /// (2^8 = 256 combinations): among the combinations whose projection lands
    /// within `tol * delta` of the target lattice point take the one with the
    /// least DCT-domain change, else the one closest to the target.
    Search { tol: f64 },
}

/// STDM embed on integer JPEG indices with an arbitrary projection domain.
///
/// The projection is `p = sum_k v_k * w_k * x_k` (`w_k = 1`: the legacy
/// index domain; `w_k = q_k`: Armor v2, dequantized DCT units). The target is
/// the nearest lattice point of `bit` at step `delta`; the continuous
/// minimum-norm solution moves along `v` in the projection domain, then the
/// indices are rounded per `rounding`. `q` (quantization steps) only weighs
/// the distortion. Returns `(projection error after rounding, squared change
/// in DCT units)`.
pub fn stdm_embed_int(
    idx: &mut [i16; SPREAD_LEN],
    w: &[f64; SPREAD_LEN],
    q: &[f64; SPREAD_LEN],
    v: &[f64; SPREAD_LEN],
    bit: u8,
    delta: f64,
    rounding: StdmRounding,
) -> (f64, f64) {
    const LIM: f64 = 1023.0;
    let mut p = 0.0;
    for k in 0..SPREAD_LEN {
        p += v[k] * w[k] * idx[k] as f64;
    }
    let t = quantize_for_bit(p, delta, bit);
    let dp = t - p;
    let mut cont = [0.0f64; SPREAD_LEN];
    for k in 0..SPREAD_LEN {
        cont[k] = (idx[k] as f64 + dp * v[k] / w[k]).clamp(-LIM, LIM);
    }
    let orig = *idx;
    let choose = |x: &[f64; SPREAD_LEN]| -> (f64, f64) {
        let mut pp = 0.0;
        let mut d = 0.0;
        for k in 0..SPREAD_LEN {
            pp += v[k] * w[k] * x[k];
            let c = (x[k] - orig[k] as f64) * q[k];
            d += c * c;
        }
        (pp - t, d)
    };
    match rounding {
        StdmRounding::Naive => {
            let mut x = [0.0f64; SPREAD_LEN];
            for k in 0..SPREAD_LEN {
                x[k] = cont[k].round();
                idx[k] = x[k] as i16;
            }
            choose(&x)
        }
        StdmRounding::Search { tol } => {
            let lo: [f64; SPREAD_LEN] = std::array::from_fn(|k| cont[k].floor());
            let mut best_in: Option<(f64, f64, u32)> = None; // (dist, err, mask) within tol
            let mut best_err: (f64, f64, u32) = (f64::INFINITY, 0.0, 0);
            let lim = tol * delta;
            for mask in 0u32..(1 << SPREAD_LEN) {
                let x: [f64; SPREAD_LEN] = std::array::from_fn(|k| {
                    let up = (mask >> k) & 1 == 1;
                    if up && lo[k] < cont[k] { lo[k] + 1.0 } else { lo[k] }
                });
                // floor == cont (integer): skip the duplicated "up" choice.
                if (0..SPREAD_LEN).any(|k| (mask >> k) & 1 == 1 && lo[k] == cont[k]) {
                    continue;
                }
                let (e, d) = choose(&x);
                if e.abs() <= lim && best_in.is_none_or(|(bd, _, _)| d < bd) {
                    best_in = Some((d, e, mask));
                }
                if e.abs() < best_err.0 {
                    best_err = (e.abs(), d, mask);
                }
            }
            let mask = best_in.map_or(best_err.2, |b| b.2);
            let x: [f64; SPREAD_LEN] = std::array::from_fn(|k| {
                if (mask >> k) & 1 == 1 && lo[k] < cont[k] { lo[k] + 1.0 } else { lo[k] }
            });
            for k in 0..SPREAD_LEN {
                idx[k] = x[k] as i16;
            }
            choose(&x)
        }
    }
}

/// Soft STDM extraction on integer indices with projection weights `w`
/// (`w_k = 1`: legacy index domain, `w_k = q_k`: Armor v2).
pub fn stdm_extract_soft_w(idx: &[f64; SPREAD_LEN], w: &[f64; SPREAD_LEN], v: &[f64; SPREAD_LEN], delta: f64) -> f64 {
    let mut c = [0.0f64; SPREAD_LEN];
    for k in 0..SPREAD_LEN {
        c[k] = idx[k] * w[k];
    }
    stdm_extract_soft(&c, v, delta)
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
    let p = super::embedding_simd::spread_dot_product(coeffs, v);

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
    let p = super::embedding_simd::spread_dot_product(coeffs, v);

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
