// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! DFT template generation, embedding, detection, and transform estimation.
//!
//! Embeds K=32 peaks at passphrase-derived positions in the DFT magnitude
//! spectrum. These peaks survive geometric transforms (rotation, scaling, crop)
//! and allow the decoder to estimate and undo the transform.

use crate::stego::armor::fft2d::Complex32;
use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha20Rng;

use crate::stego::armor::fft2d::Spectrum2D;
use crate::stego::crypto::derive_template_key;

/// Number of template peaks.
const K: usize = 32;

/// Embedding strength: peaks are `ALPHA * local_magnitude`.
const ALPHA: f32 = 0.4;

/// Minimum radius factor (fraction of min dimension).
const R_MIN_FACTOR: f64 = 0.05;

/// Maximum radius factor (fraction of min dimension).
const R_MAX_FACTOR: f64 = 0.25;

/// Detection threshold: sigma above local noise.
const DETECTION_THRESHOLD: f64 = 3.0;

/// Minimum peaks needed for transform estimation.
const MIN_PEAKS_FOR_ESTIMATION: usize = 8;

/// Search radius in frequency bins for peak detection.
const SEARCH_RADIUS: i32 = 5;

/// A template peak position and amplitude.
#[derive(Debug, Clone)]
pub struct TemplatePeak {
    /// Frequency coordinate u (can be fractional for generation, integer bin for embedding).
    pub u: f64,
    /// Frequency coordinate v.
    pub v: f64,
    /// Embedding amplitude (set during embedding).
    pub amplitude: f64,
}

/// A detected peak with expected vs actual position.
#[derive(Debug, Clone)]
pub struct DetectedPeak {
    pub expected_u: f64,
    pub expected_v: f64,
    pub detected_u: f64,
    pub detected_v: f64,
    pub confidence: f64,
}

/// Estimated affine transform parameters.
#[derive(Debug, Clone)]
pub struct AffineTransform {
    pub rotation_rad: f64,
    pub scale: f64,
}

/// Generate K=32 peak positions from passphrase via ChaCha20 PRNG.
///
/// Peak positions are in the mid-frequency band (between R_MIN and R_MAX
/// from the center) to avoid both DC and high-frequency noise.
pub fn generate_template_peaks(passphrase: &str, width: usize, height: usize) -> Vec<TemplatePeak> {
    let key = derive_template_key(passphrase);
    let mut rng = ChaCha20Rng::from_seed(key);

    let min_dim = width.min(height) as f64;
    let r_min = R_MIN_FACTOR * min_dim;
    let r_max = R_MAX_FACTOR * min_dim;

    let mut peaks = Vec::with_capacity(K);
    for _ in 0..K {
        // Generate random angle and radius in mid-frequency band
        let angle: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let radius: f64 = rng.gen_range(r_min..r_max);

        let (sin_a, cos_a) = crate::det_math::det_sincos(angle);
        let u = radius * cos_a;
        let v = radius * sin_a;

        peaks.push(TemplatePeak {
            u,
            v,
            amplitude: 0.0, // Set during embedding
        });
    }

    peaks
}

/// Add peaks to DFT magnitude, preserving phase + Hermitian symmetry.
///
/// For each peak at (u,v), adds `ALPHA * local_magnitude` along the existing
/// phase direction. Mirrors at conjugate position (-u,-v) for Hermitian symmetry
/// so the IFFT result remains real-valued.
///
/// P1b: Works with f32 spectrum data.
pub fn embed_template(spectrum: &mut Spectrum2D, peaks: &[TemplatePeak]) {
    let w = spectrum.width;
    let h = spectrum.height;
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;

    for peak in peaks {
        // Convert centered frequency to spectrum index
        let su = (cx + peak.u).round() as isize;
        let sv = (cy + peak.v).round() as isize;

        // Skip out-of-bounds peaks
        if su < 0 || su >= w as isize || sv < 0 || sv >= h as isize {
            continue;
        }
        let idx = sv as usize * w + su as usize;

        // Compute local magnitude for scaling
        let local_mag = local_mean_magnitude(spectrum, su as usize, sv as usize);
        let add_mag = ALPHA * local_mag.max(1.0);

        // Add along existing phase direction
        let existing = spectrum.data[idx];
        let enorm = crate::det_math::det_hypot(existing.re as f64, existing.im as f64) as f32;
        let phase = if enorm > 1e-6 {
            Complex32::new(existing.re / enorm, existing.im / enorm)
        } else {
            Complex32::new(1.0, 0.0)
        };
        spectrum.data[idx] += phase * add_mag;

        // Hermitian conjugate: (-u, -v) maps to (w-su, h-sv)
        let cu = (w as isize - su) % w as isize;
        let cv = (h as isize - sv) % h as isize;
        if cu >= 0 && cv >= 0 {
            let conj_idx = cv as usize * w + cu as usize;
            if conj_idx != idx {
                let conj_existing = spectrum.data[conj_idx];
                let cnorm = crate::det_math::det_hypot(conj_existing.re as f64, conj_existing.im as f64) as f32;
                let conj_phase = if cnorm > 1e-6 {
                    Complex32::new(conj_existing.re / cnorm, conj_existing.im / cnorm)
                } else {
                    Complex32::new(1.0, 0.0)
                };
                spectrum.data[conj_idx] += conj_phase * add_mag;
            }
        }
    }
}

/// Search for peaks in DFT magnitude near expected positions.
///
/// For each expected peak, searches a `SEARCH_RADIUS`-bin neighborhood
/// for the maximum magnitude. Computes confidence as (peak - noise_mean) / noise_std.
///
/// P1b: Works with f32 magnitude spectrum, computes stats in f64 for precision.
pub fn detect_template(spectrum: &Spectrum2D, peaks: &[TemplatePeak]) -> Vec<DetectedPeak> {
    let w = spectrum.width;
    let h = spectrum.height;
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;

    let magnitudes = super::fft2d::magnitude_spectrum(spectrum);

    let mut detected = Vec::new();

    for peak in peaks {
        let su = (cx + peak.u).round() as isize;
        let sv = (cy + peak.v).round() as isize;

        if su < 0 || su >= w as isize || sv < 0 || sv >= h as isize {
            continue;
        }

        // Search for maximum in neighborhood
        let mut best_mag = 0.0f64;
        let mut best_u = su;
        let mut best_v = sv;
        let mut noise_sum = 0.0f64;
        let mut noise_sq_sum = 0.0f64;
        let mut noise_count = 0usize;

        for dv in -SEARCH_RADIUS..=SEARCH_RADIUS {
            for du in -SEARCH_RADIUS..=SEARCH_RADIUS {
                let nu = su + du as isize;
                let nv = sv + dv as isize;
                if nu < 0 || nu >= w as isize || nv < 0 || nv >= h as isize {
                    continue;
                }
                let mag = magnitudes[nv as usize * w + nu as usize] as f64;

                if mag > best_mag {
                    best_mag = mag;
                    best_u = nu;
                    best_v = nv;
                }

                // Collect noise stats from the ring (exclude center 1-bin radius)
                if du.abs() > 1 || dv.abs() > 1 {
                    noise_sum += mag;
                    noise_sq_sum += mag * mag;
                    noise_count += 1;
                }
            }
        }

        if noise_count < 2 {
            continue;
        }

        let noise_mean = noise_sum / noise_count as f64;
        let noise_var = (noise_sq_sum / noise_count as f64) - noise_mean * noise_mean;
        // sqrt() is IEEE 754 correctly-rounded — deterministic across all platforms
        let noise_std = noise_var.max(0.0).sqrt().max(1e-12);

        let confidence = (best_mag - noise_mean) / noise_std;

        if confidence >= DETECTION_THRESHOLD {
            detected.push(DetectedPeak {
                expected_u: peak.u,
                expected_v: peak.v,
                detected_u: best_u as f64 - cx,
                detected_v: best_v as f64 - cy,
                confidence,
            });
        }
    }

    detected
}

/// Least-squares rotation+scale estimation from peak displacement.
///
/// Given pairs of (expected, detected) peak positions, estimates the
/// rotation angle and scale factor. Returns None if fewer than
/// `MIN_PEAKS_FOR_ESTIMATION` peaks were detected.
///
/// Uses the closed-form solution:
/// ```text
/// a = Σ(u·u' + v·v') / Σ(u² + v²)
/// b = Σ(u·v' - v·u') / Σ(u² + v²)
/// θ = atan2(b, a),  s = sqrt(a² + b²)
/// ```
pub fn estimate_transform(detected: &[DetectedPeak]) -> Option<AffineTransform> {
    if detected.len() < MIN_PEAKS_FOR_ESTIMATION {
        return None;
    }

    let mut num_a = 0.0f64;
    let mut num_b = 0.0f64;
    let mut denom = 0.0f64;

    for peak in detected {
        let u = peak.expected_u;
        let v = peak.expected_v;
        let u_prime = peak.detected_u;
        let v_prime = peak.detected_v;

        num_a += u * u_prime + v * v_prime;
        num_b += u * v_prime - v * u_prime;
        denom += u * u + v * v;
    }

    if denom < 1e-12 {
        return None;
    }

    let a = num_a / denom;
    let b = num_b / denom;

    let rotation_rad = crate::det_math::det_atan2(b, a);
    // sqrt() is IEEE 754 correctly-rounded — deterministic across all platforms
    let scale = (a * a + b * b).sqrt();

    Some(AffineTransform {
        rotation_rad,
        scale,
    })
}

/// Compute mean magnitude in a 3x3 neighborhood around (u, v).
///
/// P1b: Works with f32 spectrum data, returns f32.
fn local_mean_magnitude(spectrum: &Spectrum2D, u: usize, v: usize) -> f32 {
    let w = spectrum.width;
    let h = spectrum.height;
    let mut sum = 0.0f64;
    let mut count = 0;

    for dv in -1i32..=1 {
        for du in -1i32..=1 {
            let nu = u as i32 + du;
            let nv = v as i32 + dv;
            if nu >= 0 && nu < w as i32 && nv >= 0 && nv < h as i32 {
                let c = spectrum.data[nv as usize * w + nu as usize];
                sum += crate::det_math::det_hypot(c.re as f64, c.im as f64);
                count += 1;
            }
        }
    }

    if count > 0 { (sum / count as f64) as f32 } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stego::armor::fft2d;

    #[test]
    fn generate_deterministic() {
        let p1 = generate_template_peaks("test_pass", 256, 256);
        let p2 = generate_template_peaks("test_pass", 256, 256);
        assert_eq!(p1.len(), K);
        assert_eq!(p2.len(), K);
        for i in 0..K {
            assert_eq!(p1[i].u, p2[i].u);
            assert_eq!(p1[i].v, p2[i].v);
        }
    }

    #[test]
    fn different_passphrases_differ() {
        let p1 = generate_template_peaks("pass1", 256, 256);
        let p2 = generate_template_peaks("pass2", 256, 256);
        // At least some peaks should differ
        let mut all_same = true;
        for i in 0..K {
            if (p1[i].u - p2[i].u).abs() > 1e-10 || (p1[i].v - p2[i].v).abs() > 1e-10 {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "Different passphrases should produce different peaks");
    }

    #[test]
    fn peaks_in_mid_frequency_band() {
        let w = 256;
        let h = 256;
        let peaks = generate_template_peaks("test", w, h);
        let min_dim = w.min(h) as f64;
        let r_min = R_MIN_FACTOR * min_dim;
        let r_max = R_MAX_FACTOR * min_dim;

        for (i, peak) in peaks.iter().enumerate() {
            let r = (peak.u * peak.u + peak.v * peak.v).sqrt();
            assert!(
                r >= r_min - 0.01 && r <= r_max + 0.01,
                "Peak {i} at radius {r} outside band [{r_min}, {r_max}]"
            );
        }
    }

    #[test]
    fn embed_detect_roundtrip() {
        let width = 128;
        let height = 128;
        // Create a synthetic image with some content
        let pixels: Vec<f64> = (0..width * height)
            .map(|i| {
                let x = (i % width) as f64;
                let y = (i / width) as f64;
                128.0 + 50.0 * (x * 0.1).sin() * (y * 0.05).cos()
            })
            .collect();

        let mut spectrum = fft2d::fft2d(&pixels, width, height);
        let peaks = generate_template_peaks("embed_test", width, height);
        embed_template(&mut spectrum, &peaks);

        let detected = detect_template(&spectrum, &peaks);

        assert!(
            detected.len() >= K / 2,
            "Should detect at least half of {} peaks, got {}",
            K,
            detected.len()
        );
    }

    #[test]
    fn transform_estimation_identity() {
        // If detected positions match expected, transform should be identity
        let detected: Vec<DetectedPeak> = (0..16)
            .map(|i| {
                let angle = i as f64 * std::f64::consts::TAU / 16.0;
                let r = 20.0;
                let u = r * angle.cos();
                let v = r * angle.sin();
                DetectedPeak {
                    expected_u: u,
                    expected_v: v,
                    detected_u: u,
                    detected_v: v,
                    confidence: 10.0,
                }
            })
            .collect();

        let transform = estimate_transform(&detected).unwrap();
        assert!(
            transform.rotation_rad.abs() < 0.01,
            "Expected ~0 rotation, got {}",
            transform.rotation_rad.to_degrees()
        );
        assert!(
            (transform.scale - 1.0).abs() < 0.01,
            "Expected ~1.0 scale, got {}",
            transform.scale
        );
    }

    #[test]
    fn transform_estimation_rotation() {
        // Simulate 15 degree rotation
        let angle_deg = 15.0;
        let angle_rad = angle_deg * std::f64::consts::PI / 180.0;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let detected: Vec<DetectedPeak> = (0..20)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 20.0;
                let r = 25.0;
                let u = r * a.cos();
                let v = r * a.sin();
                // Rotated positions
                let u_rot = u * cos_a - v * sin_a;
                let v_rot = u * sin_a + v * cos_a;
                DetectedPeak {
                    expected_u: u,
                    expected_v: v,
                    detected_u: u_rot,
                    detected_v: v_rot,
                    confidence: 10.0,
                }
            })
            .collect();

        let transform = estimate_transform(&detected).unwrap();
        assert!(
            (transform.rotation_rad - angle_rad).abs() < 0.01,
            "Expected {angle_deg} deg rotation, got {} deg",
            transform.rotation_rad.to_degrees()
        );
        assert!(
            (transform.scale - 1.0).abs() < 0.01,
            "Expected ~1.0 scale, got {}",
            transform.scale
        );
    }

    #[test]
    fn transform_estimation_scale() {
        // Simulate 0.8x scale
        let scale_factor = 0.8;

        let detected: Vec<DetectedPeak> = (0..16)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 16.0;
                let r = 20.0;
                let u = r * a.cos();
                let v = r * a.sin();
                DetectedPeak {
                    expected_u: u,
                    expected_v: v,
                    detected_u: u * scale_factor,
                    detected_v: v * scale_factor,
                    confidence: 10.0,
                }
            })
            .collect();

        let transform = estimate_transform(&detected).unwrap();
        assert!(
            transform.rotation_rad.abs() < 0.01,
            "Expected ~0 rotation, got {} deg",
            transform.rotation_rad.to_degrees()
        );
        assert!(
            (transform.scale - scale_factor).abs() < 0.01,
            "Expected {scale_factor} scale, got {}",
            transform.scale
        );
    }

    #[test]
    fn too_few_peaks_returns_none() {
        let detected: Vec<DetectedPeak> = (0..3)
            .map(|i| DetectedPeak {
                expected_u: i as f64,
                expected_v: i as f64,
                detected_u: i as f64,
                detected_v: i as f64,
                confidence: 10.0,
            })
            .collect();

        assert!(estimate_transform(&detected).is_none());
    }
}
