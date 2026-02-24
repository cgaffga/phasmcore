//! Deterministic 2D FFT/IFFT using only WASM-intrinsic f64 operations.
//!
//! Replaces `rustfft` with an in-house implementation:
//! - Radix-2 Cooley-Tukey for power-of-2 sizes
//! - Bluestein's chirp-z transform for arbitrary sizes
//! All twiddle factors computed via `det_sincos()`.

use num_complex::Complex64;
use crate::det_math::{det_sincos, det_hypot};
use std::f64::consts::PI;

/// 2D complex spectrum.
pub struct Spectrum2D {
    pub data: Vec<Complex64>,
    pub width: usize,
    pub height: usize,
}

// ──────────────────────────────────────────────────────────────────────────
// 1D FFT primitives
// ──────────────────────────────────────────────────────────────────────────

/// Next power of 2 >= n.
fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// In-place radix-2 Cooley-Tukey FFT.  `data.len()` must be a power of 2.
/// `sign`: -1.0 for forward FFT, +1.0 for inverse FFT.
fn fft_radix2(data: &mut [Complex64], sign: f64) {
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
                let (s, c) = det_sincos(angle);
                let w = Complex64::new(c, s);
                let u = data[start + k];
                let v = data[start + k + half] * w;
                data[start + k] = u + v;
                data[start + k + half] = u - v;
            }
        }
        len <<= 1;
    }
}

/// 1D FFT for arbitrary length via Bluestein's algorithm.
/// `sign`: -1.0 for forward, +1.0 for inverse.
fn fft_bluestein(input: &[Complex64], sign: f64) -> Vec<Complex64> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return input.to_vec();
    }
    if n.is_power_of_two() {
        let mut buf = input.to_vec();
        fft_radix2(&mut buf, sign);
        return buf;
    }

    let m = next_pow2(2 * n - 1);

    // Chirp factors: w_k = exp(sign * i * π * k² / n)
    let mut chirp = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n {
        let angle = sign * PI * (k as f64 * k as f64) / n as f64;
        let (s, c) = det_sincos(angle);
        chirp[k] = Complex64::new(c, s);
    }

    // a[k] = x[k] * conj(chirp[k]),  zero-padded to length m
    let mut a = vec![Complex64::new(0.0, 0.0); m];
    for k in 0..n {
        a[k] = input[k] * chirp[k].conj();
    }

    // b[k] = chirp[k], with wrap-around for negative indices, zero-padded
    let mut b = vec![Complex64::new(0.0, 0.0); m];
    b[0] = chirp[0];
    for k in 1..n {
        b[k] = chirp[k];
        b[m - k] = chirp[k]; // b[-k] = chirp[k] (symmetric since chirp[k]=chirp[-k mod n])
    }

    // Convolve via FFT: A = FFT(a), B = FFT(b), C = IFFT(A·B)
    fft_radix2(&mut a, -1.0);
    fft_radix2(&mut b, -1.0);
    for i in 0..m {
        a[i] = a[i] * b[i];
    }
    fft_radix2(&mut a, 1.0);

    // Normalize radix-2 inverse and apply chirp
    let inv_m = 1.0 / m as f64;
    let mut result = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n {
        result[k] = a[k] * inv_m * chirp[k].conj();
    }

    result
}

/// 1D forward FFT (arbitrary length).
fn fft1d(data: &[Complex64]) -> Vec<Complex64> {
    fft_bluestein(data, -1.0)
}

/// 1D inverse FFT (arbitrary length) — unnormalized.
/// Caller must divide by n.
fn ifft1d(data: &[Complex64]) -> Vec<Complex64> {
    fft_bluestein(data, 1.0)
}

// ──────────────────────────────────────────────────────────────────────────
// 2D FFT / IFFT — public API (same as before)
// ──────────────────────────────────────────────────────────────────────────

/// Real-valued pixel array → 2D complex spectrum.
///
/// The input is a row-major f64 array of size `width * height`.
pub fn fft2d(pixels: &[f64], width: usize, height: usize) -> Spectrum2D {
    assert_eq!(pixels.len(), width * height);

    let mut data: Vec<Complex64> = pixels.iter().map(|&v| Complex64::new(v, 0.0)).collect();

    // FFT each row
    for row in 0..height {
        let start = row * width;
        let row_data = &data[start..start + width];
        let transformed = fft1d(row_data);
        data[start..start + width].copy_from_slice(&transformed);
    }

    // Transpose (row-major → col-major for column FFTs)
    let mut transposed = vec![Complex64::new(0.0, 0.0); width * height];
    for r in 0..height {
        for c in 0..width {
            transposed[c * height + r] = data[r * width + c];
        }
    }

    // FFT each column (now stored as rows in transposed)
    for col in 0..width {
        let start = col * height;
        let col_data = &transposed[start..start + height];
        let transformed = fft1d(col_data);
        transposed[start..start + height].copy_from_slice(&transformed);
    }

    // Transpose back
    for r in 0..height {
        for c in 0..width {
            data[r * width + c] = transposed[c * height + r];
        }
    }

    Spectrum2D { data, width, height }
}

/// 2D complex spectrum → real-valued pixel array.
///
/// Takes the real parts after inverse FFT, normalized by `1/(width*height)`.
pub fn ifft2d(spectrum: &Spectrum2D) -> Vec<f64> {
    let width = spectrum.width;
    let height = spectrum.height;
    let mut data = spectrum.data.clone();

    // IFFT each row
    for row in 0..height {
        let start = row * width;
        let row_data = &data[start..start + width];
        let transformed = ifft1d(row_data);
        data[start..start + width].copy_from_slice(&transformed);
    }

    // Transpose
    let mut transposed = vec![Complex64::new(0.0, 0.0); width * height];
    for r in 0..height {
        for c in 0..width {
            transposed[c * height + r] = data[r * width + c];
        }
    }

    // IFFT each column
    for col in 0..width {
        let start = col * height;
        let col_data = &transposed[start..start + height];
        let transformed = ifft1d(col_data);
        transposed[start..start + height].copy_from_slice(&transformed);
    }

    // Transpose back and normalize
    let norm = 1.0 / (width * height) as f64;
    let mut result = vec![0.0f64; width * height];
    for r in 0..height {
        for c in 0..width {
            result[r * width + c] = transposed[c * height + r].re * norm;
        }
    }

    result
}

/// Compute magnitude of each spectrum element.
pub fn magnitude_spectrum(spectrum: &Spectrum2D) -> Vec<f64> {
    spectrum.data.iter().map(|c| det_hypot(c.re, c.im)).collect()
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
                (pixels[i] - recovered[i]).abs() < 1e-10,
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
                (pixels[i] - recovered[i]).abs() < 1e-8,
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
        let freq_energy: f64 = spectrum.data.iter().map(|c| c.norm_sqr()).sum();

        let n = (width * height) as f64;
        assert!(
            (spatial_energy - freq_energy / n).abs() < 1e-6,
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
        assert!(
            (spectrum.data[0].re - expected_dc).abs() < 1e-10,
            "DC component should be sum of all pixels: expected {expected_dc}, got {}",
            spectrum.data[0].re
        );
        assert!(spectrum.data[0].im.abs() < 1e-10);
    }

    #[test]
    fn fft1d_basic() {
        // FFT of [1, 0, 0, 0] should be [1, 1, 1, 1]
        let input = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let output = fft1d(&input);
        for k in 0..4 {
            assert!((output[k].re - 1.0).abs() < 1e-12, "Re[{k}]={}", output[k].re);
            assert!(output[k].im.abs() < 1e-12, "Im[{k}]={}", output[k].im);
        }
    }

    #[test]
    fn bluestein_matches_radix2() {
        // For power-of-2 size, Bluestein should give same result as radix-2
        let n = 8;
        let input: Vec<Complex64> = (0..n).map(|i| Complex64::new((i * 3 + 1) as f64, (i * 2) as f64)).collect();

        let mut radix2_buf = input.clone();
        fft_radix2(&mut radix2_buf, -1.0);

        let bluestein_result = fft_bluestein(&input, -1.0);

        for k in 0..n {
            assert!(
                (radix2_buf[k].re - bluestein_result[k].re).abs() < 1e-8 &&
                (radix2_buf[k].im - bluestein_result[k].im).abs() < 1e-8,
                "Mismatch at {k}: radix2={}, bluestein={}",
                radix2_buf[k], bluestein_result[k]
            );
        }
    }
}
