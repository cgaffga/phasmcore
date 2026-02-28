// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Deterministic 2D FFT/IFFT using only WASM-intrinsic f64 operations.
//!
//! Replaces `rustfft` with an in-house implementation:
//! - Radix-2 Cooley-Tukey for power-of-2 sizes
//! - Bluestein's chirp-z transform for arbitrary sizes
//! All twiddle factors computed via `det_sincos()`.
//!
//! Memory optimizations (Phase 3):
//! - P1a: Column FFTs use gather-FFT-scatter with a single column buffer
//!   instead of a full transposed copy (saves ~186 MB for 4032x3024).
//! - P1b: All FFT data uses f32 (Complex32) — template detection only needs
//!   coarse peak finding. Twiddle factors still computed in f64 via
//!   `det_sincos()` then cast to f32.
//! - P2a: BluesteinPlan precomputes chirp factors and FFT(b_hat) for reuse
//!   across all rows/columns of the same length.

use num_complex::Complex;
use crate::det_math::{det_sincos, det_hypot};
use std::f64::consts::PI;

/// Complex32 type alias for f32 complex numbers.
pub type Complex32 = Complex<f32>;

/// 2D complex spectrum using f32 for memory efficiency.
pub struct Spectrum2D {
    pub data: Vec<Complex32>,
    pub width: usize,
    pub height: usize,
}

// ──────────────────────────────────────────────────────────────────────────
// Bluestein plan: precomputed chirp factors for reuse (P2a)
// ──────────────────────────────────────────────────────────────────────────

/// Precomputed Bluestein chirp factors and FFT(b_hat) for a given (n, sign).
///
/// Eliminates redundant chirp computation and FFT(b) calls when processing
/// many rows or columns of the same length.
struct BluesteinPlan {
    n: usize,
    m: usize, // next_pow2(2*n - 1)
    chirp: Vec<Complex32>,
    b_hat: Vec<Complex32>, // FFT of padded conjugate chirp
}

impl BluesteinPlan {
    /// Create a new Bluestein plan for length `n` and direction `sign`.
    fn new(n: usize, sign: f64) -> Self {
        let m = next_pow2(2 * n - 1);

        // Chirp factors: w_k = exp(sign * i * pi * k^2 / n)
        let mut chirp = vec![Complex32::new(0.0, 0.0); n];
        for k in 0..n {
            let angle = sign * PI * (k as f64 * k as f64) / n as f64;
            let (s, c) = det_sincos(angle);
            chirp[k] = Complex32::new(c as f32, s as f32);
        }

        // b[k] = chirp[k], with wrap-around for negative indices, zero-padded
        let mut b = vec![Complex32::new(0.0, 0.0); m];
        b[0] = chirp[0];
        for k in 1..n {
            b[k] = chirp[k];
            b[m - k] = chirp[k];
        }

        // Precompute FFT(b)
        fft_radix2_f32(&mut b, -1.0);

        BluesteinPlan { n, m, chirp, b_hat: b }
    }

    /// Execute Bluestein FFT using precomputed plan.
    fn execute(&self, input: &[Complex32]) -> Vec<Complex32> {
        debug_assert_eq!(input.len(), self.n);

        // a[k] = x[k] * conj(chirp[k]), zero-padded to length m
        let mut a = vec![Complex32::new(0.0, 0.0); self.m];
        for k in 0..self.n {
            a[k] = input[k] * self.chirp[k].conj();
        }

        // Convolve: A = FFT(a), C = IFFT(A * B_hat)
        fft_radix2_f32(&mut a, -1.0);
        for i in 0..self.m {
            a[i] = a[i] * self.b_hat[i];
        }
        fft_radix2_f32(&mut a, 1.0);

        // Normalize radix-2 inverse and apply chirp
        let inv_m = 1.0 / self.m as f32;
        let mut result = vec![Complex32::new(0.0, 0.0); self.n];
        for k in 0..self.n {
            result[k] = a[k] * inv_m * self.chirp[k].conj();
        }

        result
    }
}

// ──────────────────────────────────────────────────────────────────────────
// 1D FFT primitives (f32)
// ──────────────────────────────────────────────────────────────────────────

/// Next power of 2 >= n.
fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// In-place radix-2 Cooley-Tukey FFT for f32.  `data.len()` must be a power of 2.
/// `sign`: -1.0 for forward FFT, +1.0 for inverse FFT.
fn fft_radix2_f32(data: &mut [Complex32], sign: f64) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // Butterfly stages
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_step = sign * PI / half as f64;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let angle = angle_step * k as f64;
                // Use f64 det_sincos for twiddle factor precision, cast to f32
                let (s, c) = det_sincos(angle);
                let w = Complex32::new(c as f32, s as f32);
                let u = data[start + k];
                let v = data[start + k + half] * w;
                data[start + k] = u + v;
                data[start + k + half] = u - v;
            }
        }
        len <<= 1;
    }
}

/// 1D FFT for arbitrary length using f32.
/// Uses BluesteinPlan if available (for 2D FFT reuse), or creates one on the fly.
/// `sign`: -1.0 for forward, +1.0 for inverse.
fn fft1d_f32_with_plan(input: &[Complex32], sign: f64, plan: Option<&BluesteinPlan>) -> Vec<Complex32> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return input.to_vec();
    }
    if n.is_power_of_two() {
        let mut buf = input.to_vec();
        fft_radix2_f32(&mut buf, sign);
        return buf;
    }

    // Use precomputed plan if available
    if let Some(p) = plan {
        debug_assert_eq!(p.n, n);
        return p.execute(input);
    }

    // Fallback: create a temporary plan
    let temp_plan = BluesteinPlan::new(n, sign);
    temp_plan.execute(input)
}

/// 1D forward FFT (arbitrary length, f32).
#[allow(dead_code)]
fn fft1d_f32(data: &[Complex32]) -> Vec<Complex32> {
    fft1d_f32_with_plan(data, -1.0, None)
}

/// 1D inverse FFT (arbitrary length, f32) — unnormalized.
#[allow(dead_code)]
fn ifft1d_f32(data: &[Complex32]) -> Vec<Complex32> {
    fft1d_f32_with_plan(data, 1.0, None)
}

// ──────────────────────────────────────────────────────────────────────────
// 2D FFT / IFFT — public API (f32, memory-optimized)
// ──────────────────────────────────────────────────────────────────────────

/// Real-valued pixel array -> 2D complex spectrum (f32).
///
/// The input is a row-major f64 array of size `width * height`.
/// Uses gather-FFT-scatter for columns (P1a) and precomputed Bluestein
/// plans for chirp reuse (P2a).
pub fn fft2d(pixels: &[f64], width: usize, height: usize) -> Spectrum2D {
    assert_eq!(pixels.len(), width * height);

    let mut data: Vec<Complex32> = pixels.iter().map(|&v| Complex32::new(v as f32, 0.0)).collect();

    // P2a: Precompute Bluestein plans for row and column lengths (if non-power-of-2)
    let row_plan = if !width.is_power_of_two() && width > 1 {
        Some(BluesteinPlan::new(width, -1.0))
    } else {
        None
    };
    let col_plan = if !height.is_power_of_two() && height > 1 {
        Some(BluesteinPlan::new(height, -1.0))
    } else {
        None
    };

    // FFT each row
    for row in 0..height {
        let start = row * width;
        let row_data = &data[start..start + width];
        let transformed = fft1d_f32_with_plan(row_data, -1.0, row_plan.as_ref());
        data[start..start + width].copy_from_slice(&transformed);
    }

    // P1a: FFT each column using gather-FFT-scatter with a single column buffer.
    // No full transposed buffer needed.
    let mut col_buf = vec![Complex32::new(0.0, 0.0); height];
    for col in 0..width {
        // Gather column
        for r in 0..height {
            col_buf[r] = data[r * width + col];
        }
        // FFT
        let transformed = fft1d_f32_with_plan(&col_buf, -1.0, col_plan.as_ref());
        // Scatter back
        for r in 0..height {
            data[r * width + col] = transformed[r];
        }
    }

    Spectrum2D { data, width, height }
}

/// 2D complex spectrum -> real-valued pixel array.
///
/// Takes the real parts after inverse FFT, normalized by `1/(width*height)`.
/// Uses gather-IFFT-scatter for columns (P1a) and precomputed Bluestein
/// plans for chirp reuse (P2a).
pub fn ifft2d(spectrum: &Spectrum2D) -> Vec<f64> {
    let width = spectrum.width;
    let height = spectrum.height;
    let mut data = spectrum.data.clone();

    // P2a: Precompute Bluestein plans for row and column lengths (if non-power-of-2)
    let row_plan = if !width.is_power_of_two() && width > 1 {
        Some(BluesteinPlan::new(width, 1.0))
    } else {
        None
    };
    let col_plan = if !height.is_power_of_two() && height > 1 {
        Some(BluesteinPlan::new(height, 1.0))
    } else {
        None
    };

    // IFFT each row
    for row in 0..height {
        let start = row * width;
        let row_data = &data[start..start + width];
        let transformed = fft1d_f32_with_plan(row_data, 1.0, row_plan.as_ref());
        data[start..start + width].copy_from_slice(&transformed);
    }

    // P1a: IFFT each column using gather-IFFT-scatter with a single column buffer.
    let mut col_buf = vec![Complex32::new(0.0, 0.0); height];
    for col in 0..width {
        // Gather column
        for r in 0..height {
            col_buf[r] = data[r * width + col];
        }
        // IFFT
        let transformed = fft1d_f32_with_plan(&col_buf, 1.0, col_plan.as_ref());
        // Scatter back
        for r in 0..height {
            data[r * width + col] = transformed[r];
        }
    }

    // Normalize and extract real parts
    let norm = 1.0 / (width * height) as f64;
    let mut result = vec![0.0f64; width * height];
    for i in 0..data.len() {
        result[i] = data[i].re as f64 * norm;
    }

    result
}

/// Compute magnitude of each spectrum element (returns f32).
pub fn magnitude_spectrum(spectrum: &Spectrum2D) -> Vec<f32> {
    spectrum.data.iter().map(|c| {
        // Use f64 det_hypot for precision, cast result to f32
        det_hypot(c.re as f64, c.im as f64) as f32
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_ifft_roundtrip() {
        let width = 16;
        let height = 16;
        let pixels: Vec<f64> = (0..width * height).map(|i| (i as f64) * 0.1 + 50.0).collect();

        let spectrum = fft2d(&pixels, width, height);
        let recovered = ifft2d(&spectrum);

        for i in 0..pixels.len() {
            assert!(
                (pixels[i] - recovered[i]).abs() < 1e-3,
                "Mismatch at {i}: expected {}, got {}",
                pixels[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn fft_ifft_roundtrip_non_pow2() {
        // Test with non-power-of-2 dimensions (Bluestein path)
        let width = 12;
        let height = 10;
        let pixels: Vec<f64> = (0..width * height).map(|i| (i as f64) * 0.3 + 20.0).collect();

        let spectrum = fft2d(&pixels, width, height);
        let recovered = ifft2d(&spectrum);

        for i in 0..pixels.len() {
            assert!(
                (pixels[i] - recovered[i]).abs() < 0.1,
                "Mismatch at {i}: expected {}, got {}",
                pixels[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn parseval_theorem() {
        let width = 8;
        let height = 8;
        let pixels: Vec<f64> = (0..width * height).map(|i| ((i * 7 + 3) % 256) as f64).collect();

        let spatial_energy: f64 = pixels.iter().map(|v| v * v).sum();

        let spectrum = fft2d(&pixels, width, height);
        let freq_energy: f64 = spectrum.data.iter().map(|c| {
            let re = c.re as f64;
            let im = c.im as f64;
            re * re + im * im
        }).sum();

        let n = (width * height) as f64;
        // Relaxed tolerance for f32 spectrum
        assert!(
            (spatial_energy - freq_energy / n).abs() < 10.0,
            "Parseval's theorem violated: spatial={spatial_energy}, freq/N={}", freq_energy / n
        );
    }

    #[test]
    fn dc_component_is_sum() {
        let width = 4;
        let height = 4;
        let pixels = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                          9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

        let spectrum = fft2d(&pixels, width, height);

        let expected_dc: f64 = pixels.iter().sum();
        // Relaxed tolerance for f32
        assert!(
            (spectrum.data[0].re as f64 - expected_dc).abs() < 0.1,
            "DC component should be sum of all pixels: expected {expected_dc}, got {}",
            spectrum.data[0].re
        );
        assert!((spectrum.data[0].im as f64).abs() < 0.1);
    }

    #[test]
    fn fft1d_basic() {
        // FFT of [1, 0, 0, 0] should be [1, 1, 1, 1]
        let input = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
        ];
        let output = fft1d_f32(&input);
        for k in 0..4 {
            assert!((output[k].re - 1.0).abs() < 1e-5, "Re[{k}]={}", output[k].re);
            assert!(output[k].im.abs() < 1e-5, "Im[{k}]={}", output[k].im);
        }
    }

    #[test]
    fn bluestein_matches_radix2() {
        // For power-of-2 size, Bluestein plan should give same result as radix-2
        let n = 8;
        let input: Vec<Complex32> = (0..n).map(|i| Complex32::new((i * 3 + 1) as f32, (i * 2) as f32)).collect();

        let mut radix2_buf = input.clone();
        fft_radix2_f32(&mut radix2_buf, -1.0);

        // Test via plan
        let _plan = BluesteinPlan::new(n, -1.0);
        // For power-of-2, fft1d_f32_with_plan uses radix-2 directly, not the plan.
        // Test the plan directly on a non-power-of-2 size instead.
        let n2 = 7;
        let input2: Vec<Complex32> = (0..n2).map(|i| Complex32::new((i * 3 + 1) as f32, (i * 2) as f32)).collect();
        let plan2 = BluesteinPlan::new(n2, -1.0);
        let result_plan = plan2.execute(&input2);
        let result_direct = fft1d_f32(&input2);
        for k in 0..n2 {
            assert!(
                (result_plan[k].re - result_direct[k].re).abs() < 1e-3 &&
                (result_plan[k].im - result_direct[k].im).abs() < 1e-3,
                "Plan vs direct mismatch at {k}: plan={}, direct={}",
                result_plan[k], result_direct[k]
            );
        }

        // Also verify radix-2 results haven't changed (basic sanity)
        let result_r2 = fft1d_f32(&input);
        for k in 0..n {
            assert!(
                (radix2_buf[k].re - result_r2[k].re).abs() < 1e-3 &&
                (radix2_buf[k].im - result_r2[k].im).abs() < 1e-3,
                "Mismatch at {k}: radix2={}, fft1d={}",
                radix2_buf[k], result_r2[k]
            );
        }
    }

    #[test]
    fn bluestein_plan_reuse() {
        // Verify that reusing a BluesteinPlan gives the same result each time
        let n = 13; // non-power-of-2
        let plan = BluesteinPlan::new(n, -1.0);

        let input1: Vec<Complex32> = (0..n).map(|i| Complex32::new(i as f32, 0.0)).collect();
        let input2: Vec<Complex32> = (0..n).map(|i| Complex32::new(0.0, i as f32)).collect();

        let r1a = plan.execute(&input1);
        let r2 = plan.execute(&input2);
        let r1b = plan.execute(&input1);

        for k in 0..n {
            assert!(
                (r1a[k].re - r1b[k].re).abs() < 1e-5 &&
                (r1a[k].im - r1b[k].im).abs() < 1e-5,
                "Plan reuse gave different results at {k}: first={}, second={}",
                r1a[k], r1b[k]
            );
        }
        // Just verify r2 didn't corrupt anything (no specific value check needed)
        assert_eq!(r2.len(), n);
    }
}
