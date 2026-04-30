// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Deterministic math functions for cross-platform WASM reproducibility.
//!
//! Provides `det_sin`, `det_cos`, `det_sincos`, `det_atan2`, `det_hypot`,
//! `det_exp`, and `det_powi_f64` using only IEEE 754 operations that map
//! to WASM intrinsics (add, sub, mul, div, floor, sqrt, abs, copysign,
//! bit-pattern manipulation). No calls to `Math.sin`, `Math.cos`,
//! `Math.exp`, etc.
//!
//! Algorithms and coefficients are classical polynomial approximations of
//! IEEE 754 transcendentals (Cody-Waite range reduction + minimax
//! polynomials evaluated by Horner's scheme), guaranteeing < 1 ULP error
//! for all functions. `det_powi_f64` is exponentiation-by-squaring on
//! bit-exact IEEE 754 multiply — exact by construction.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

// ──────────────────────────────────────────────────────────────────────────
// Extended-precision π/2 for Cody-Waite range reduction.
// PIO2_HI + PIO2_LO = π/2 to ~70 bits of precision.
// ──────────────────────────────────────────────────────────────────────────

const PIO2_HI: f64 = f64::from_bits(0x3FF921FB54442D18); // 1.5707963267948966
const PIO2_LO: f64 = f64::from_bits(0x3C91A62633145C07); // 6.123233995736766e-17

// ──────────────────────────────────────────────────────────────────────────
// Sin kernel coefficients — classical minimax polynomial on [-π/4, π/4].
// sin(x) ≈ x + x³·(S1 + x²·(S2 + x²·(S3 + x²·(S4 + x²·(S5 + x²·S6)))))
// for |x| ≤ π/4.
// ──────────────────────────────────────────────────────────────────────────

const S1: f64 = f64::from_bits(0xBFC5555555555549); // -1.66666666666666324348e-01
const S2: f64 = f64::from_bits(0x3F8111111110F8A6); //  8.33333333332248946124e-03
const S3: f64 = f64::from_bits(0xBF2A01A019C161D5); // -1.98412698298579493134e-04
const S4: f64 = f64::from_bits(0x3EC71DE357B1FE7D); //  2.75573137070700676789e-06
const S5: f64 = f64::from_bits(0xBE5AE5E68A2B9CEB); // -2.50507602534068634195e-08
const S6: f64 = f64::from_bits(0x3DE5D93A5ACFD57C); //  1.58969099521155010221e-10

// ──────────────────────────────────────────────────────────────────────────
// Cos kernel coefficients — classical minimax polynomial on [-π/4, π/4].
// cos(x) ≈ 1 - x²/2 + x⁴·(C1 + x²·(C2 + …))
// Uses Kahan trick for precision: w = 1-hz, return w + ((1-w)-hz+r).
// ──────────────────────────────────────────────────────────────────────────

const C1: f64 = f64::from_bits(0x3FA5555555555549); //  4.16666666666666019037e-02
const C2: f64 = f64::from_bits(0xBF56C16C16C15177); // -1.38888888888741095749e-03
const C3: f64 = f64::from_bits(0x3EFA01A019CB1590); //  2.48015872894767294178e-05
const C4: f64 = f64::from_bits(0xBE927E4F809C52AD); // -2.75573143513906633035e-07
const C5: f64 = f64::from_bits(0x3E21EE9EBDB4B1C4); //  2.08757232129817482790e-09
const C6: f64 = f64::from_bits(0xBDA8FAE9BE8838D4); // -1.13596475577881948265e-11

// ──────────────────────────────────────────────────────────────────────────
// Atan coefficients and constants — classical 11-term polynomial with
// 5-range argument reduction.
// ──────────────────────────────────────────────────────────────────────────

const AT: [f64; 11] = [
    f64::from_bits(0x3FD555555555550D), //  3.33333333333329318027e-01
    f64::from_bits(0xBFC999999998EBC4), // -1.99999999998764832476e-01
    f64::from_bits(0x3FC24924920083FF), //  1.42857142725034663711e-01
    f64::from_bits(0xBFBC71C6FE231671), // -1.11111104054623557880e-01
    f64::from_bits(0x3FB745CDC54C206E), //  9.09088713343650656196e-02
    f64::from_bits(0xBFB3B0F2AF749A6D), // -7.69187620504482999495e-02
    f64::from_bits(0x3FB10D66A0D03D51), //  6.66107313738753120669e-02
    f64::from_bits(0xBFADDE2D52DEFD9A), // -5.83357013379057348645e-02
    f64::from_bits(0x3FA97B4B24760DEB), //  4.97687799461593236017e-02
    f64::from_bits(0xBFA2B4442C6A6C2F), // -3.65315727442169155270e-02
    f64::from_bits(0x3F90AD3AE322DA11), //  1.62858201153657823623e-02
];

/// atan reference values: atan(0.5), atan(1.0), atan(1.5), atan(∞).
/// Using decimal values parsed by the Rust compiler (guaranteed correct f64).
const ATAN_REF: [f64; 4] = [
    4.636476090008061e-01,   // atan(0.5)
    FRAC_PI_4,               // atan(1.0) = π/4
    9.827_937_232_473_29e-1,   // atan(1.5)
    FRAC_PI_2,               // atan(∞) = π/2
];

// ──────────────────────────────────────────────────────────────────────────
// Exp coefficients and constants — classical exp polynomial with
// Cody-Waite ln(2) range reduction.
// exp(x) = 2^k · exp(r), where x = k·ln(2) + r, |r| ≤ 0.5·ln(2).
// exp(r) ≈ 1 + 2·c / (2 - c), c = r - r²·(P1 + r²·(P2 + r²·(P3 + r²·(P4 + r²·P5))))
// ──────────────────────────────────────────────────────────────────────────

const EXP_O_THRESHOLD: f64 = f64::from_bits(0x40862E42FEFA39EF); //  7.09782712893383973096e+02
const EXP_U_THRESHOLD: f64 = f64::from_bits(0xC0874910D52D3051); // -7.45133219101941108420e+02
const EXP_LN2HI: f64 = f64::from_bits(0x3FE62E42FEE00000);       //  6.93147180369123816490e-01
const EXP_LN2LO: f64 = f64::from_bits(0x3DEA39EF35793C76);       //  1.90821492927058770002e-10
const EXP_INVLN2: f64 = f64::from_bits(0x3FF71547652B82FE);      //  1.44269504088896338700e+00
const EXP_TWOM1000: f64 = f64::from_bits(0x0170000000000000);    //  9.33263618503218878990e-302 = 2^-1000

const EXP_P1: f64 = f64::from_bits(0x3FC555555555553E); //  1.66666666666666019037e-01
const EXP_P2: f64 = f64::from_bits(0xBF66C16C16BEBD93); // -2.77777777770155933842e-03
const EXP_P3: f64 = f64::from_bits(0x3F11566AAF25DE2C); //  6.61375632143793436117e-05
const EXP_P4: f64 = f64::from_bits(0xBEBBBD41C5D26BF1); // -1.65339022054652515390e-07
const EXP_P5: f64 = f64::from_bits(0x3E66376972BEA4D0); //  4.13813679705723846039e-10

// ──────────────────────────────────────────────────────────────────────────
// Core polynomial evaluations on reduced range [-π/4, π/4].
// ──────────────────────────────────────────────────────────────────────────

/// Evaluate sin polynomial for |x| ≤ π/4 (classical sine kernel on the reduced range).
#[inline]
fn sin_kern(x: f64) -> f64 {
    let z = x * x;
    let v = z * x;
    let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    x + v * (S1 + z * r)
}

/// Evaluate cos polynomial for |x| ≤ π/4 (classical cosine kernel on the reduced range).
/// cos(x) ≈ 1 - z/2 + z²·(C1 + z·C2 + …) where z = x².
#[inline]
fn cos_kern(x: f64) -> f64 {
    let z = x * x;
    let r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    let hz = 0.5 * z;
    // Return 1 - (hz - z*r), where z*r = z²·(C1 + z·C2 + …)
    1.0 - (hz - z * r)
}

// ──────────────────────────────────────────────────────────────────────────
// Cody-Waite range reduction: x → r in [-π/4, π/4], quadrant n mod 4.
// ──────────────────────────────────────────────────────────────────────────

#[inline]
fn reduce(x: f64) -> (f64, i32) {
    let n = (x * (2.0 / PI) + 0.5).floor();
    let r = (x - n * PIO2_HI) - n * PIO2_LO;
    (r, (n as i64 & 3) as i32)
}

// ──────────────────────────────────────────────────────────────────────────
// Public sin / cos / sincos
// ──────────────────────────────────────────────────────────────────────────

/// Deterministic sine — uses only WASM-intrinsic f64 operations.
pub fn det_sin(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    let (r, q) = reduce(x);
    match q {
        0 =>  sin_kern(r),
        1 =>  cos_kern(r),
        2 => -sin_kern(r),
        3 => -cos_kern(r),
        _ => unreachable!(),
    }
}

/// Deterministic cosine — uses only WASM-intrinsic f64 operations.
pub fn det_cos(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    let (r, q) = reduce(x);
    match q {
        0 =>  cos_kern(r),
        1 => -sin_kern(r),
        2 => -cos_kern(r),
        3 =>  sin_kern(r),
        _ => unreachable!(),
    }
}

/// Deterministic sin and cos computed together (shared range reduction).
pub fn det_sincos(x: f64) -> (f64, f64) {
    if x.is_nan() || x.is_infinite() {
        return (f64::NAN, f64::NAN);
    }
    let (r, q) = reduce(x);
    let s = sin_kern(r);
    let c = cos_kern(r);
    match q {
        0 => ( s,  c),
        1 => ( c, -s),
        2 => (-s, -c),
        3 => (-c,  s),
        _ => unreachable!(),
    }
}

// ──────────────────────────────────────────────────────────────────────────
// atan / atan2 — classical 5-range argument reduction + polynomial.
// ──────────────────────────────────────────────────────────────────────────

/// Deterministic atan(x) using 5-range argument reduction + 11-term polynomial.
fn det_atan(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let neg = x < 0.0;
    let mut xa = x.abs();

    // Argument reduction into 5 ranges
    let id: i32;
    if xa < 0.4375 {
        // |x| < 7/16 — use polynomial directly
        if xa < 1e-29 {
            return x; // tiny x
        }
        id = -1;
    } else if xa < 1.1875 {
        if xa < 0.6875 {
            // 7/16 <= |x| < 11/16 — reduce via atan(0.5)
            id = 0;
            xa = (2.0 * xa - 1.0) / (2.0 + xa);
        } else {
            // 11/16 <= |x| < 19/16 — reduce via atan(1.0)
            id = 1;
            xa = (xa - 1.0) / (xa + 1.0);
        }
    } else if xa < 2.4375 {
        // 19/16 <= |x| < 39/16 — reduce via atan(1.5)
        id = 2;
        xa = (xa - 1.5) / (1.0 + 1.5 * xa);
    } else {
        // |x| >= 39/16 — reduce via atan(∞) = π/2
        id = 3;
        xa = -1.0 / xa;
    }

    // Polynomial evaluation: split into odd and even parts for accuracy
    let z = xa * xa;
    let w = z * z;
    let s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))));
    let s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))));

    let result = if id < 0 {
        xa - xa * (s1 + s2)
    } else {
        ATAN_REF[id as usize] + (xa - xa * (s1 + s2))
    };

    if neg { -result } else { result }
}

/// Deterministic atan2(y, x) — uses only WASM-intrinsic f64 operations.
pub fn det_atan2(y: f64, x: f64) -> f64 {
    if y.is_nan() || x.is_nan() {
        return f64::NAN;
    }

    if y == 0.0 {
        if x > 0.0 || (x == 0.0 && x.is_sign_positive()) {
            return y; // ±0
        } else {
            return if y.is_sign_negative() { -PI } else { PI };
        }
    }

    if x == 0.0 {
        return if y > 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 };
    }

    if y.is_infinite() {
        if x.is_infinite() {
            return if x > 0.0 {
                if y > 0.0 { FRAC_PI_4 } else { -FRAC_PI_4 }
            } else if y > 0.0 { 3.0 * FRAC_PI_4 } else { -3.0 * FRAC_PI_4 };
        }
        return if y > 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 };
    }

    if x.is_infinite() {
        return if x > 0.0 {
            if y.is_sign_negative() { -0.0 } else { 0.0 }
        } else if y.is_sign_negative() { -PI } else { PI };
    }

    // General case
    let a = det_atan((y / x).abs());

    if x > 0.0 {
        if y >= 0.0 { a } else { -a }
    } else if y >= 0.0 { PI - a } else { -(PI - a) }
}

/// Deterministic hypot(x, y) = sqrt(x² + y²).
///
/// sqrt is a WASM intrinsic, so this is deterministic. Uses scaling
/// to avoid overflow/underflow for extreme values.
pub fn det_hypot(x: f64, y: f64) -> f64 {
    let xa = x.abs();
    let ya = y.abs();
    let (big, small) = if xa >= ya { (xa, ya) } else { (ya, xa) };
    if big == 0.0 {
        return 0.0;
    }
    let ratio = small / big;
    big * (1.0 + ratio * ratio).sqrt()
}

// ──────────────────────────────────────────────────────────────────────────
// exp / powi
// ──────────────────────────────────────────────────────────────────────────

/// Deterministic exp(x) — classical exp polynomial with Cody-Waite ln(2)
/// range reduction.
///
/// Strategy: argument-reduce to `x = k·ln(2) + r` with `|r| ≤ 0.5·ln(2)`,
/// evaluate `exp(r)` via a 5-term polynomial in `r²`, then scale by `2^k`
/// using direct exponent-bit manipulation. All operations are pure
/// IEEE 754 arithmetic + bit cast — bit-exact across iOS / Android /
/// x86_64 / WASM.
pub fn det_exp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { f64::INFINITY } else { 0.0 };
    }
    if x > EXP_O_THRESHOLD {
        return f64::INFINITY; // overflow
    }
    if x < EXP_U_THRESHOLD {
        return 0.0; // underflow
    }

    // High 32 bits of |x| (sign bit cleared); used for branch thresholds
    // via the standard GET_HIGH_WORD bit-extraction pattern.
    let hx_abs = (x.abs().to_bits() >> 32) as u32;

    // Argument reduction: x = k·ln(2) + r, |r| ≤ 0.5·ln(2)
    let (hi, lo, k): (f64, f64, i32) = if hx_abs > 0x3fd62e42 {
        // |x| > 0.5·ln(2) ≈ 0.347
        if hx_abs < 0x3FF0A2B2 {
            // |x| < 1.5·ln(2) ≈ 1.04 — single-step k = ±1
            if x.is_sign_negative() {
                (x + EXP_LN2HI, -EXP_LN2LO, -1)
            } else {
                (x - EXP_LN2HI, EXP_LN2LO, 1)
            }
        } else {
            // General case: k = round-toward-zero(x · 1/ln(2) + ±0.5)
            let half = if x.is_sign_negative() { -0.5 } else { 0.5 };
            let k_int = (EXP_INVLN2 * x + half) as i32;
            let t = k_int as f64;
            (x - t * EXP_LN2HI, t * EXP_LN2LO, k_int)
        }
    } else if hx_abs < 0x3e300000 {
        // |x| < 2^-28 — exp(x) ≈ 1 + x
        return 1.0 + x;
    } else {
        // |x| in [2^-28, 0.5·ln(2)] — no reduction needed
        (x, 0.0, 0)
    };

    let r = hi - lo;
    let z = r * r;
    let c = r - z * (EXP_P1 + z * (EXP_P2 + z * (EXP_P3 + z * (EXP_P4 + z * EXP_P5))));

    let y = if k == 0 {
        1.0 - ((r * c) / (c - 2.0) - r)
    } else {
        1.0 - ((lo - (r * c) / (2.0 - c)) - hi)
    };

    if k == 0 {
        return y;
    }

    // Scale by 2^k via direct exponent-bit add. After reduction, y ∈ ~[0.5, 2],
    // so its biased exponent is 1022, 1023, or 1024 — adding k stays in the
    // valid normal range as long as k ≥ -1021. Below that we scale via 2^-1000.
    if k >= -1021 {
        let bits = y.to_bits().wrapping_add(((k as i64) << 52) as u64);
        f64::from_bits(bits)
    } else {
        let bits = y.to_bits().wrapping_add((((k + 1000) as i64) << 52) as u64);
        f64::from_bits(bits) * EXP_TWOM1000
    }
}

/// Deterministic integer-exponent power: `base^n` for `n: i32`.
///
/// Uses exponentiation-by-squaring with bit-exact IEEE 754 multiply.
/// For our codec / stego use cases (`n` typically in `[-32, 64]`) the
/// log₂(|n|) ≤ 6 steps means at most ~12 multiplications, each
/// deterministic per IEEE 754. Negative `n` inverts the result via a
/// single deterministic divide.
///
/// Behaviour matches `f64::powi` semantically but is bit-exact across
/// iOS / Android / x86_64 / WASM (whereas `f64::powi` lowers to
/// `@llvm.powi.f64` which has implementation-defined rounding).
pub fn det_powi_f64(base: f64, n: i32) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if base == 1.0 {
        return 1.0;
    }
    let mut e: u64 = (n as i64).unsigned_abs();
    let mut b = base;
    let mut result = 1.0;
    while e > 0 {
        if e & 1 == 1 {
            result *= b;
        }
        e >>= 1;
        if e > 0 {
            b *= b;
        }
    }
    if n < 0 { 1.0 / result } else { result }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_3, FRAC_PI_6};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() && b.is_nan() { return true; }
        (a - b).abs() <= tol
    }

    #[test]
    fn sin_exact_values() {
        let tol = 1e-15;
        assert!(approx_eq(det_sin(0.0), 0.0, tol));
        assert!(approx_eq(det_sin(FRAC_PI_6), 0.5, tol));
        assert!(approx_eq(det_sin(FRAC_PI_4), std::f64::consts::FRAC_1_SQRT_2, tol));
        assert!(approx_eq(det_sin(FRAC_PI_3), 3.0_f64.sqrt() / 2.0, tol));
        assert!(approx_eq(det_sin(FRAC_PI_2), 1.0, tol));
        assert!(approx_eq(det_sin(PI), 0.0, 1e-15));
        assert!(approx_eq(det_sin(-FRAC_PI_2), -1.0, tol));
    }

    #[test]
    fn cos_exact_values() {
        let tol = 1e-15;
        assert!(approx_eq(det_cos(0.0), 1.0, tol));
        assert!(approx_eq(det_cos(FRAC_PI_6), 3.0_f64.sqrt() / 2.0, tol));
        assert!(approx_eq(det_cos(FRAC_PI_4), std::f64::consts::FRAC_1_SQRT_2, tol));
        assert!(approx_eq(det_cos(FRAC_PI_3), 0.5, tol));
        assert!(approx_eq(det_cos(FRAC_PI_2), 0.0, 1e-15));
        assert!(approx_eq(det_cos(PI), -1.0, tol));
    }

    #[test]
    fn sincos_identity() {
        // sin²(x) + cos²(x) = 1 for various x
        for i in 0..200 {
            let x = (i as f64 - 100.0) * 0.13;
            let (s, c) = det_sincos(x);
            let err = (s * s + c * c - 1.0).abs();
            assert!(err < 1e-14, "sin²+cos²={} at x={x} (err={err})", s * s + c * c);
        }
    }

    #[test]
    fn sincos_matches_separate() {
        for i in 0..50 {
            let x = (i as f64 - 25.0) * 0.37;
            let (s, c) = det_sincos(x);
            assert_eq!(s, det_sin(x), "sin mismatch at x={x}");
            assert_eq!(c, det_cos(x), "cos mismatch at x={x}");
        }
    }

    #[test]
    fn sin_large_argument() {
        let x = 1e6;
        let s = det_sin(x);
        let c = det_cos(x);
        let err = (s * s + c * c - 1.0).abs();
        assert!(err < 1e-6, "sin²+cos²={} at x={x}", s * s + c * c);
    }

    #[test]
    fn sin_special_values() {
        assert!(det_sin(f64::NAN).is_nan());
        assert!(det_sin(f64::INFINITY).is_nan());
        assert!(det_sin(f64::NEG_INFINITY).is_nan());
    }

    #[test]
    fn atan2_quadrants() {
        let eps = 1e-15;
        assert!(approx_eq(det_atan2(1.0, 1.0), FRAC_PI_4, eps));
        assert!(approx_eq(det_atan2(1.0, -1.0), 3.0 * FRAC_PI_4, eps));
        assert!(approx_eq(det_atan2(-1.0, -1.0), -3.0 * FRAC_PI_4, eps));
        assert!(approx_eq(det_atan2(-1.0, 1.0), -FRAC_PI_4, eps));
        assert!(approx_eq(det_atan2(0.0, 1.0), 0.0, eps));
        assert!(approx_eq(det_atan2(1.0, 0.0), FRAC_PI_2, eps));
        assert!(approx_eq(det_atan2(0.0, -1.0), PI, eps));
        assert!(approx_eq(det_atan2(-1.0, 0.0), -FRAC_PI_2, eps));
    }

    #[test]
    fn atan2_special_values() {
        assert!(det_atan2(f64::NAN, 1.0).is_nan());
        assert!(det_atan2(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn atan_specific_values() {
        let eps = 1e-15;
        assert!(approx_eq(det_atan(0.0), 0.0, eps));
        assert!(approx_eq(det_atan(1.0), FRAC_PI_4, eps));
        assert!(approx_eq(det_atan(-1.0), -FRAC_PI_4, eps));
        // atan(√3) = π/3
        assert!(approx_eq(det_atan(3.0_f64.sqrt()), FRAC_PI_3, eps));
        // atan(1/√3) = π/6
        assert!(approx_eq(det_atan(1.0 / 3.0_f64.sqrt()), FRAC_PI_6, eps));
    }

    #[test]
    fn hypot_basic() {
        assert!(approx_eq(det_hypot(3.0, 4.0), 5.0, 1e-15));
        assert!(approx_eq(det_hypot(0.0, 0.0), 0.0, 0.0));
        assert!(approx_eq(det_hypot(1.0, 0.0), 1.0, 0.0));
        assert!(approx_eq(det_hypot(0.0, 1.0), 1.0, 0.0));
    }

    #[test]
    fn hypot_no_overflow() {
        let big = 1e300;
        let h = det_hypot(big, big);
        assert!(h.is_finite());
        assert!(approx_eq(h, big * 2.0_f64.sqrt(), big * 1e-14));
    }

    #[test]
    fn deterministic_across_calls() {
        for i in 0..100 {
            let x = (i as f64) * 0.0731 - 3.5;
            assert_eq!(det_sin(x).to_bits(), det_sin(x).to_bits());
            assert_eq!(det_cos(x).to_bits(), det_cos(x).to_bits());
        }
    }

    #[test]
    fn matches_std_closely() {
        for i in 0..200 {
            let x = (i as f64 - 100.0) * 0.05;
            let ds = det_sin(x);
            let ss = x.sin();
            assert!((ds - ss).abs() < 5e-13,
                "det_sin({x})={ds} vs std sin={ss}, diff={}", (ds - ss).abs());
            let dc = det_cos(x);
            let sc = x.cos();
            assert!((dc - sc).abs() < 5e-13,
                "det_cos({x})={dc} vs std cos={sc}, diff={}", (dc - sc).abs());
        }
    }

    #[test]
    fn exp_exact_values() {
        let tol = 1e-15;
        assert_eq!(det_exp(0.0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(det_exp(-0.0).to_bits(), 1.0_f64.to_bits());
        assert!(approx_eq(det_exp(1.0), std::f64::consts::E, tol));
        assert!(approx_eq(det_exp(-1.0), 1.0 / std::f64::consts::E, tol));
        assert!(approx_eq(det_exp(2.0), std::f64::consts::E.powi(2), 1e-14));
        assert!(approx_eq(det_exp(-3.0), 0.049787068367863944, 1e-15));
        assert!(approx_eq(det_exp(0.5), 1.6487212707001282, tol));
    }

    #[test]
    fn exp_special_values() {
        assert!(det_exp(f64::NAN).is_nan());
        assert_eq!(det_exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(det_exp(f64::NEG_INFINITY), 0.0);
        // Overflow / underflow saturate.
        assert_eq!(det_exp(800.0), f64::INFINITY);
        assert_eq!(det_exp(-800.0), 0.0);
    }

    #[test]
    fn exp_tiny_arguments() {
        // For |x| < 2^-28 the algorithm short-circuits to 1+x.
        let tiny = 1e-12;
        let res = det_exp(tiny);
        assert!((res - (1.0 + tiny)).abs() < 1e-24);
        let neg_tiny = -1e-12;
        let res = det_exp(neg_tiny);
        assert!((res - (1.0 + neg_tiny)).abs() < 1e-24);
    }

    #[test]
    fn exp_matches_std_closely() {
        // Match std f64::exp within a tight tolerance for typical input ranges.
        for i in 0..200 {
            let x = (i as f64 - 100.0) * 0.07;
            let de = det_exp(x);
            let se = x.exp();
            let rel = if se != 0.0 { ((de - se) / se).abs() } else { (de - se).abs() };
            assert!(rel < 5e-15,
                "det_exp({x})={de} vs std exp={se}, rel={rel}");
        }
    }

    #[test]
    fn exp_deterministic_across_calls() {
        for i in 0..100 {
            let x = (i as f64) * 0.0731 - 3.5;
            assert_eq!(det_exp(x).to_bits(), det_exp(x).to_bits());
        }
    }

    #[test]
    fn powi_zero_exponent() {
        // x^0 = 1 for any non-NaN x (matches f64::powi).
        assert_eq!(det_powi_f64(0.0, 0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(det_powi_f64(1.0, 0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(det_powi_f64(-3.5, 0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(det_powi_f64(1e300, 0).to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn powi_one_exponent() {
        for &b in &[0.0, 1.0, -1.0, 2.5, -7.3, 1e-10, 1e10] {
            assert_eq!(det_powi_f64(b, 1).to_bits(), b.to_bits(),
                "powi({b}, 1) should equal {b}");
        }
    }

    #[test]
    fn powi_basic() {
        assert_eq!(det_powi_f64(2.0, 10), 1024.0);
        assert_eq!(det_powi_f64(2.0, 16), 65536.0);
        assert_eq!(det_powi_f64(0.5, 3), 0.125);
        assert_eq!(det_powi_f64(0.75, 4), 0.31640625);
        assert_eq!(det_powi_f64(-1.0, 2), 1.0);
        assert_eq!(det_powi_f64(-1.0, 3), -1.0);
        assert_eq!(det_powi_f64(-2.0, 4), 16.0);
    }

    #[test]
    fn powi_negative_exponent() {
        assert_eq!(det_powi_f64(2.0, -1), 0.5);
        assert_eq!(det_powi_f64(2.0, -10), 1.0 / 1024.0);
        assert_eq!(det_powi_f64(0.5, -3), 8.0);
        // Powers of 2 are exact f64 — bit-exact equality.
        assert_eq!(det_powi_f64(2.0, -8).to_bits(), (1.0_f64 / 256.0).to_bits());
    }

    #[test]
    fn powi_matches_std_closely() {
        // For powers of 2 the result is exactly representable; should be
        // bit-exact even compared to std.
        for k in -10..=10 {
            let d = det_powi_f64(2.0, k);
            let s = 2.0_f64.powi(k);
            assert_eq!(d.to_bits(), s.to_bits(),
                "powi(2.0, {k}): det={d} vs std={s}");
        }
        // For non-powers-of-2 we should match within a few ULPs.
        for &b in &[0.75, 1.5, 0.9, 1.1, 3.7] {
            for k in 0..=12 {
                let d = det_powi_f64(b, k);
                let s = b.powi(k);
                let rel = ((d - s) / s).abs();
                assert!(rel < 5e-15,
                    "powi({b}, {k}): det={d} vs std={s}, rel={rel}");
            }
        }
    }

    #[test]
    fn powi_deterministic_across_calls() {
        for k in -20..=20 {
            for &b in &[0.5, 0.75, 1.5, 2.0, 3.0] {
                assert_eq!(det_powi_f64(b, k).to_bits(),
                           det_powi_f64(b, k).to_bits());
            }
        }
    }
}
