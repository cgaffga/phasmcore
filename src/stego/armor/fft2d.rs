//! 2D FFT/IFFT wrapper over `rustfft`.
//!
//! Implements 2D DFT via row-then-column 1D FFTs, avoiding the `fft2d` crate
//! for fewer dependencies and more control.

use num_complex::Complex64;
use rustfft::FftPlanner;

/// 2D complex spectrum.
pub struct Spectrum2D {
    pub data: Vec<Complex64>,
    pub width: usize,
    pub height: usize,
}

/// Real-valued pixel array → 2D complex spectrum.
///
/// The input is a row-major f64 array of size `width * height`.
pub fn fft2d(pixels: &[f64], width: usize, height: usize) -> Spectrum2D {
    assert_eq!(pixels.len(), width * height);

    let mut data: Vec<Complex64> = pixels.iter().map(|&v| Complex64::new(v, 0.0)).collect();

    let mut planner = FftPlanner::new();

    // FFT each row
    let fft_row = planner.plan_fft_forward(width);
    let mut scratch = vec![Complex64::new(0.0, 0.0); fft_row.get_inplace_scratch_len()];
    for row in 0..height {
        let start = row * width;
        fft_row.process_with_scratch(&mut data[start..start + width], &mut scratch);
    }

    // Transpose (row-major → col-major for column FFTs)
    let mut transposed = vec![Complex64::new(0.0, 0.0); width * height];
    for r in 0..height {
        for c in 0..width {
            transposed[c * height + r] = data[r * width + c];
        }
    }

    // FFT each column (now stored as rows in transposed)
    let fft_col = planner.plan_fft_forward(height);
    let mut scratch = vec![Complex64::new(0.0, 0.0); fft_col.get_inplace_scratch_len()];
    for col in 0..width {
        let start = col * height;
        fft_col.process_with_scratch(&mut transposed[start..start + height], &mut scratch);
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

    let mut planner = FftPlanner::new();

    // IFFT each row
    let ifft_row = planner.plan_fft_inverse(width);
    let mut scratch = vec![Complex64::new(0.0, 0.0); ifft_row.get_inplace_scratch_len()];
    for row in 0..height {
        let start = row * width;
        ifft_row.process_with_scratch(&mut data[start..start + width], &mut scratch);
    }

    // Transpose
    let mut transposed = vec![Complex64::new(0.0, 0.0); width * height];
    for r in 0..height {
        for c in 0..width {
            transposed[c * height + r] = data[r * width + c];
        }
    }

    // IFFT each column
    let ifft_col = planner.plan_fft_inverse(height);
    let mut scratch = vec![Complex64::new(0.0, 0.0); ifft_col.get_inplace_scratch_len()];
    for col in 0..width {
        let start = col * height;
        ifft_col.process_with_scratch(&mut transposed[start..start + height], &mut scratch);
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
    spectrum.data.iter().map(|c| c.norm()).collect()
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
    fn parseval_theorem() {
        // Energy in spatial domain should equal energy in frequency domain (scaled)
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
}
