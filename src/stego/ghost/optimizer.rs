// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Cover image preprocessing optimizer for steganographic embedding.
//!
//! Subtly modifies raw pixels before JPEG compression to improve embedding
//! quality and capacity. Each mode applies a different pipeline:
//!
//! - **Ghost**: Texture-adaptive 4-stage pipeline (noise injection, micro-contrast,
//!   unsharp mask, smooth-region dithering) — maximizes non-zero AC coefficients.
//!   Each stage adapts its strength based on existing texture levels to avoid
//!   degrading already-optimized images.
//! - **Armor**: Light pipeline (block-boundary smoothing, DC stabilization) —
//!   reduces cross-block discontinuities for STDM robustness.
//! - **Fortress**: Minimal (block-boundary smoothing only) — stabilizes DC
//!   averages for BA-QIM embedding.
//!
//! **"Do no harm" guarantee**: After optimization, the average absolute gradient
//! (a proxy for JPEG AC energy / stego capacity) is compared to the original.
//! If the optimizer reduced gradient energy, the original pixels are returned
//! unchanged. This prevents degradation of pre-optimized images (e.g. images
//! that already had noise, micro-contrast, and sharpening applied in Photoshop).
//!
//! The optimizer is deterministic: given the same pixels, dimensions, config,
//! and seed, it always produces the same output.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::det_math::det_exp;

/// Configuration for the cover image optimizer.
pub struct OptimizerConfig {
    /// Strength multiplier (0.0–1.0). Default: 0.85.
    pub strength: f32,
    /// ChaCha20 seed for deterministic noise generation.
    pub seed: [u8; 32],
    /// Pipeline variant to apply.
    pub mode: OptimizerMode,
}

/// Which optimization pipeline to run.
pub enum OptimizerMode {
    /// Full 4-stage pipeline (noise, micro-contrast, unsharp mask, dithering).
    Ghost,
    /// Light: block-boundary smoothing + DC stabilization.
    Armor,
    /// Minimal: block-boundary smoothing only.
    Fortress,
}

/// T2.11 — Cross-platform empirical optimizer-output byte equivalence.
///
/// Returns the lowercase-hex SHA256 of `optimize_cover` output on a
/// fixed deterministic input. Used to verify the optimizer produces
/// bit-identical output across native / iOS / Android / WASM, and
/// to catch any regression while shipping T2.11a/b/c.
#[doc(hidden)]
pub fn optimizer_test_hash_hex() -> String {
    use sha2::{Digest, Sha256};
    // Deterministic test image: gradient + sine pattern, 256×256 RGB.
    let w: u32 = 256;
    let h: u32 = 256;
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            // Each channel exercises a different texture pattern so the
            // optimizer's 4 stages all see varied input.
            let xf = x as f64 / w as f64;
            let yf = y as f64 / h as f64;
            // Use det_sincos / det_exp for cross-platform determinism.
            let (s1, _) = crate::det_math::det_sincos(xf * std::f64::consts::PI * 8.0);
            let (_, c1) = crate::det_math::det_sincos(yf * std::f64::consts::PI * 8.0);
            let r = ((xf * 255.0) + s1 * 30.0).clamp(0.0, 255.0) as u8;
            let g = ((yf * 255.0) + c1 * 20.0).clamp(0.0, 255.0) as u8;
            let b = (((xf + yf) * 127.0) + s1 * c1 * 25.0).clamp(0.0, 255.0) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    let seed = [0u8; 32];
    let config = OptimizerConfig {
        strength: 0.85,
        seed,
        mode: OptimizerMode::Ghost,
    };
    let out = optimize_cover(&pixels, w, h, &config);

    let hash = Sha256::digest(&out);
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Optimize raw RGB pixels for steganographic embedding.
///
/// Returns a new pixel buffer with the same dimensions and format (RGB, 3
/// bytes per pixel, row-major). The modifications are subtle enough to be
/// imperceptible (PSNR > 44 dB, SSIM > 0.993) but improve embedding
/// capacity by increasing wavelet energy in smooth regions.
///
/// If the optimization would reduce gradient energy (degrade stego capacity),
/// the original pixels are returned unchanged.
pub fn optimize_cover(
    pixels_rgb: &[u8],
    width: u32,
    height: u32,
    config: &OptimizerConfig,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    assert_eq!(pixels_rgb.len(), w * h * 3, "pixel buffer size mismatch");

    let result = match config.mode {
        OptimizerMode::Ghost => optimize_ghost(pixels_rgb, w, h, config),
        OptimizerMode::Armor => optimize_armor(pixels_rgb, w, h, config),
        OptimizerMode::Fortress => optimize_fortress(pixels_rgb, w, h, config),
    };

    // "Do no harm" safety check: compare gradient energy before and after.
    // Require at least 1% improvement — marginal changes on already-optimized
    // images are more likely to hurt (especially SI-UNIWARD rounding errors)
    // than to help. If the optimizer can't meaningfully improve the image,
    // the original was already well-prepared.
    let original_energy = gradient_energy(pixels_rgb, w, h);
    let optimized_energy = gradient_energy(&result, w, h);
    if optimized_energy < original_energy * 1.01 {
        pixels_rgb.to_vec()
    } else {
        result
    }
}

// ---------------------------------------------------------------------------
// Texture analysis
// ---------------------------------------------------------------------------

/// Analyze image texture to compute per-stage adaptive strength multipliers.
///
/// Returns `(noise_factor, contrast_factor, sharpen_factor, dither_factor)`,
/// each in [0.0, 1.0]. A factor of 0.0 means "skip this stage entirely"
/// (image already has plenty of that characteristic). 1.0 means "apply fully".
#[cfg(test)]
fn analyze_texture(luma: &[f64], variance: &[f64], w: usize, h: usize) -> (f64, f64, f64, f64) {
    let n = w * h;
    if n == 0 {
        return (1.0, 1.0, 1.0, 1.0);
    }

    // 1. Noise factor: based on mean variance.
    //    Low variance (< 50) = smooth image, needs lots of noise → factor ~1.0
    //    Medium variance (50-300) = some texture → moderate noise
    //    High variance (> 600) = already noisy → skip noise
    let mean_variance: f64 = variance.iter().sum::<f64>() / n as f64;
    let noise_factor = adaptive_scale(mean_variance, 50.0, 600.0);

    // 2. Contrast factor: based on proportion of pixels with low local contrast.
    //    Compute local gradient magnitude as a contrast proxy.
    let mut low_contrast_count = 0usize;
    for y in 1..h {
        for x in 1..w {
            let idx = y * w + x;
            let dy = (luma[idx] - luma[(y - 1) * w + x]).abs();
            let dx = (luma[idx] - luma[y * w + (x - 1)]).abs();
            let grad = dy + dx;
            if grad < 3.0 {
                low_contrast_count += 1;
            }
        }
    }
    let low_contrast_ratio = low_contrast_count as f64 / ((w - 1) * (h - 1)) as f64;
    // If < 10% of pixels are low-contrast, the image already has good contrast
    // If > 50% of pixels are low-contrast, we need full enhancement
    let contrast_factor = if low_contrast_ratio < 0.10 {
        0.0
    } else if low_contrast_ratio > 0.50 {
        1.0
    } else {
        (low_contrast_ratio - 0.10) / 0.40
    };

    // 3. Sharpen factor: based on high-frequency energy.
    //    Measure average |pixel - local_mean| as proxy for existing sharpening.
    let mut hf_sum = 0.0f64;
    let mut hf_count = 0usize;
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let idx = y * w + x;
            let center = luma[idx];
            // 4-neighbor average
            let avg = (luma[(y - 1) * w + x] + luma[(y + 1) * w + x]
                + luma[y * w + (x - 1)] + luma[y * w + (x + 1)])
                / 4.0;
            hf_sum += (center - avg).abs();
            hf_count += 1;
        }
    }
    let mean_hf = if hf_count > 0 { hf_sum / hf_count as f64 } else { 0.0 };
    // mean_hf < 1.0 = very smooth, needs sharpening → 1.0
    // mean_hf > 4.0 = already sharp → 0.0
    let sharpen_factor = adaptive_scale(mean_hf, 1.0, 4.0);

    // 4. Dither factor: based on proportion of smooth 4×4 blocks.
    //    If few smooth blocks exist, dithering has little to do.
    let blocks_x = w / 4;
    let blocks_y = h / 4;
    let total_blocks = blocks_x * blocks_y;
    let mut smooth_blocks = 0usize;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut block_mean = 0.0f64;
            for dy in 0..4 {
                for dx in 0..4 {
                    block_mean += luma[(by * 4 + dy) * w + (bx * 4 + dx)];
                }
            }
            block_mean /= 16.0;
            let mut sad = 0.0f64;
            for dy in 0..4 {
                for dx in 0..4 {
                    sad += (luma[(by * 4 + dy) * w + (bx * 4 + dx)] - block_mean).abs();
                }
            }
            if sad < 15.0 {
                smooth_blocks += 1;
            }
        }
    }
    let smooth_ratio = if total_blocks > 0 {
        smooth_blocks as f64 / total_blocks as f64
    } else {
        0.0
    };
    // If < 5% smooth blocks → no dithering needed (image is already textured)
    // If > 30% smooth blocks → full dithering
    let dither_factor = if smooth_ratio < 0.05 {
        0.0
    } else if smooth_ratio > 0.30 {
        1.0
    } else {
        (smooth_ratio - 0.05) / 0.25
    };

    (noise_factor, contrast_factor, sharpen_factor, dither_factor)
}

/// Maps a metric value to [0.0, 1.0] inversely: below `low` → 1.0, above `high` → 0.0.
fn adaptive_scale(value: f64, low: f64, high: f64) -> f64 {
    if value <= low {
        1.0
    } else if value >= high {
        0.0
    } else {
        1.0 - (value - low) / (high - low)
    }
}

/// Compute average absolute gradient energy (proxy for JPEG AC coefficient density).
/// Higher value = more texture = better for steganography.
fn gradient_energy(pixels: &[u8], w: usize, h: usize) -> f64 {
    if w < 2 || h < 2 {
        return 0.0;
    }
    let mut energy = 0.0f64;
    let mut count = 0usize;
    // Sample on luma channel for speed
    for y in 1..h {
        for x in 1..w {
            let idx = (y * w + x) * 3;
            let idx_left = (y * w + (x - 1)) * 3;
            let idx_above = ((y - 1) * w + x) * 3;
            // Luma approximation: just green channel (dominant in luma)
            let center = pixels[idx + 1] as f64;
            let left = pixels[idx_left + 1] as f64;
            let above = pixels[idx_above + 1] as f64;
            energy += (center - left).abs() + (center - above).abs();
            count += 1;
        }
    }
    if count > 0 { energy / count as f64 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Ghost pipeline (4 stages, texture-adaptive)
// ---------------------------------------------------------------------------

fn optimize_ghost(pixels: &[u8], w: usize, h: usize, config: &OptimizerConfig) -> Vec<u8> {
    let strength = config.strength as f64;
    let mut rng = ChaCha20Rng::from_seed(config.seed);

    // Compute per-pixel texture map: local variance drives ALL stages.
    // Each pixel gets its own strength multiplier based on its neighborhood.
    // Smooth areas (low variance) → full processing.
    // Textured areas (high variance) → minimal/no processing.
    let luma = extract_luma(pixels, w, h);
    let variance = local_variance_5x5(&luma, w, h);

    let mut out = pixels.to_vec();

    // Stage 1: Content-adaptive noise injection (per-pixel)
    // Variance < 50: smooth region → σ up to 1.5 (needs texture)
    // Variance 50–600: transitional → scaled σ
    // Variance > 600: already textured → σ ≈ 0 (skip)
    //
    // T2.11c — split into two phases so the per-pixel application
    // can run in parallel without breaking cross-platform RNG
    // determinism:
    //
    //   (1) sequentially draw 2 ChaCha20 samples per pixel + compute
    //       the Irwin-Hall(2) gauss value into a flat Vec<f64>. RNG
    //       order is identical to the previous serial path on every
    //       platform.
    //
    //   (2) parallel-apply: per-pixel branch on variance + scaled
    //       sigma threshold, apply gauss × scaled_sigma to each RGB
    //       channel. No RNG access in this phase; each output pixel
    //       reads its own pre-generated gauss + writes its own row
    //       of `out` (disjoint).
    //
    // Irwin-Hall(2) noise: u1+u2 has triangular distribution on
    // [0, 2] with mean 1, variance 1/6. Center to mean 0 and scale
    // by √6 so variance matches N(0,1).
    //
    // Replaces Box-Muller `(-2*ln(u1)).sqrt() * cos(2π·u2)` —
    // f64::ln + f64::cos lower to non-deterministic libm / Math.*
    // on WASM, breaking cross-platform reproducibility of the
    // optimized cover. Triangular noise is a coarser Gaussian
    // approximation but adequate for the small-amplitude texture
    // injection done here. RNG consumption is unchanged (still 2
    // uniform draws per pixel).
    const SQRT_6: f64 = 2.449489742783178;
    let gauss_table: Vec<f64> = (0..h * w)
        .map(|_| {
            let u1: f64 = rng.gen_range(0.0f64..1.0).max(1e-10);
            let u2: f64 = rng.gen_range(0.0f64..1.0);
            (u1 + u2 - 1.0) * SQRT_6
        })
        .collect();

    let stage1_row = |y: usize, row_out: &mut [u8]| {
        for x in 0..w {
            let var_val = variance[y * w + x];
            let local_need = adaptive_scale(var_val, 50.0, 600.0);
            // Skip pixels in areas that are already well-textured —
            // even tiny modifications can hurt SI-UNIWARD rounding
            // errors.
            if local_need > 0.1 {
                let sigma = 1.5 * local_need + 0.1;
                let scaled_sigma = sigma * strength * local_need;
                if scaled_sigma > 0.05 {
                    let gauss = gauss_table[y * w + x];
                    let noise = gauss * scaled_sigma;
                    for c in 0..3 {
                        let val = row_out[x * 3 + c] as f64 + noise;
                        row_out[x * 3 + c] = val.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, row)| stage1_row(y, row));
    }
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row)| stage1_row(y, row));

    // T2.11e — Stage 2 SIMD: pre-deinterleave + pad stage1 ONCE into
    // 3 planar f64 buffers (clamp-to-edge by 1 cell). Then per-row
    // rayon, per-pixel-pair lane-parallel f64x2 SIMD on the 3x3 sum.
    // Path A: each lane is one pixel's 9-element sum done in the same
    // f64 add order as the scalar code.
    let stage1 = out.clone();
    let stage1_planes = super::optimizer_simd::deinterleave_pad1_to_planar_f64(
        &stage1, w, h,
    );
    let pw = w + 2;
    let stage2_row = |y: usize, row_out: &mut [u8]| {
        let mut x = 0;
        // SIMD pair loop: process pixels (x, x+1) per iteration.
        while x + 2 <= w {
            // Per-lane gate computation (kept scalar — branches make
            // SIMD-masking more code for no speedup on this path).
            let need_x = adaptive_scale(variance[y * w + x], 80.0, 500.0);
            let need_x1 = adaptive_scale(variance[y * w + x + 1], 80.0, 500.0);
            let pass_x = need_x > 0.1;
            let pass_x1 = need_x1 > 0.1;
            if !pass_x && !pass_x1 {
                x += 2;
                continue;
            }
            let enhance_x = 0.15 * strength * need_x;
            let enhance_x1 = 0.15 * strength * need_x1;
            let pass_x = pass_x && enhance_x > 0.01;
            let pass_x1 = pass_x1 && enhance_x1 > 0.01;
            if !pass_x && !pass_x1 {
                x += 2;
                continue;
            }

            for c in 0..3 {
                let plane = &stage1_planes[c];
                // SIMD sum: lane 0 = 3×3 around pixel x, lane 1 =
                // around pixel x+1. Padded coords: (y, x) reads
                // [y..y+3) × [x..x+3) in padded space (no border
                // clamping needed inside the SIMD inner loop).
                let (sum_x, sum_x1) =
                    super::optimizer_simd::sum_3x3_pair(plane, pw, x, y);
                let mean_x = sum_x / 9.0;
                let mean_x1 = sum_x1 / 9.0;

                // Original pixel value at padded (y+1, x+1) for
                // pixel x, (y+1, x+2) for pixel x+1.
                let pixel_x = plane[(y + 1) * pw + (x + 1)];
                let pixel_x1 = plane[(y + 1) * pw + (x + 2)];

                if pass_x {
                    let deviation = pixel_x - mean_x;
                    let enhanced = mean_x + deviation * (1.0 + enhance_x);
                    row_out[x * 3 + c] = enhanced.round().clamp(0.0, 255.0) as u8;
                }
                if pass_x1 {
                    let deviation = pixel_x1 - mean_x1;
                    let enhanced = mean_x1 + deviation * (1.0 + enhance_x1);
                    row_out[(x + 1) * 3 + c] =
                        enhanced.round().clamp(0.0, 255.0) as u8;
                }
            }
            x += 2;
        }
        // Tail scalar for odd-width images.
        while x < w {
            let var_val = variance[y * w + x];
            let local_need = adaptive_scale(var_val, 80.0, 500.0);
            if local_need > 0.1 {
                let local_enhance = 0.15 * strength * local_need;
                if local_enhance > 0.01 {
                    for c in 0..3 {
                        let (local_mean, pixel_val) =
                            neighborhood_mean_3x3(&stage1, w, h, x, y, c);
                        let deviation = pixel_val - local_mean;
                        let enhanced =
                            local_mean + deviation * (1.0 + local_enhance);
                        row_out[x * 3 + c] =
                            enhanced.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
            x += 1;
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, row)| stage2_row(y, row));
    }
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row)| stage2_row(y, row));

    // T2.11b — Stage 3: parallelize per-row. Reads from `stage2` +
    // `blurred` + `variance`. Writes to `out` rows. Same disjoint-row
    // pattern as Stage 2.
    let stage2 = out.clone();
    let blurred = gaussian_blur_separable(&stage2, w, h, 1.0);
    let stage3_row = |y: usize, row_out: &mut [u8]| {
        for x in 0..w {
            let var_val = variance[y * w + x];
            let local_need = adaptive_scale(var_val, 100.0, 400.0);
            if local_need > 0.1 {
                let local_sharpen = 0.12 * strength * local_need;
                if local_sharpen > 0.01 {
                    for c in 0..3 {
                        let stage_i = (y * w + x) * 3 + c;
                        let orig = stage2[stage_i] as f64;
                        let blur = blurred[stage_i] as f64;
                        let diff = orig - blur;
                        // Threshold: only sharpen if difference > 3
                        // (avoid amplifying noise).
                        if diff.abs() > 3.0 {
                            let sharpened = orig + local_sharpen * diff;
                            row_out[x * 3 + c] =
                                sharpened.round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, row)| stage3_row(y, row));
    }
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row)| stage3_row(y, row));

    // Stage 4: Smooth-region dithering (per 4×4 block, re-evaluated)
    // Uses CURRENT luma (after stages 1-3) — blocks already textured by
    // earlier stages won't get redundant dithering.
    let bayer_4x4: [[f64; 4]; 4] = [
        [0.0, 8.0, 2.0, 10.0],
        [12.0, 4.0, 14.0, 6.0],
        [3.0, 11.0, 1.0, 9.0],
        [15.0, 7.0, 13.0, 5.0],
    ];
    let bayer_norm: [[f64; 4]; 4] = {
        let mut b = [[0.0; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                b[r][c] = (bayer_4x4[r][c] / 15.0) * 2.0 - 1.0;
            }
        }
        b
    };

    let current_luma = extract_luma(&out, w, h);
    let blocks_x = w / 4;
    let blocks_y = h / 4;

    // T2.11b — Stage 4: pre-generate per-block RNG offsets sequentially
    // (preserves cross-platform RNG order), then parallel-apply the
    // dither in by-strips of 4 rows each. Different by-strips write
    // disjoint pixel rows, so par_chunks_mut(w * 3 * 4) is race-free.
    let block_offsets: Vec<f64> = (0..blocks_y * blocks_x)
        .map(|_| rng.gen_range(0.0f64..1.0) * 2.0 - 1.0)
        .collect();

    let stage4_strip = |by: usize, strip: &mut [u8]| {
        // par_chunks_mut may emit a trailing partial chunk if h isn't
        // a multiple of 4 — the original loop iterated by in
        // 0..blocks_y, so anything past that is a no-op.
        if by >= blocks_y {
            return;
        }
        for bx in 0..blocks_x {
            let block_offset = block_offsets[by * blocks_x + bx];

            let mut block_mean = 0.0f64;
            let mut block_var_sum = 0.0f64;
            for dy in 0..4 {
                for dx in 0..4 {
                    let px = bx * 4 + dx;
                    let py = by * 4 + dy;
                    if px < w && py < h {
                        block_mean += current_luma[py * w + px];
                        block_var_sum += variance[py * w + px];
                    }
                }
            }
            block_mean /= 16.0;
            let block_var_avg = block_var_sum / 16.0;
            let mut block_sad = 0.0f64;
            for dy in 0..4 {
                for dx in 0..4 {
                    let px = bx * 4 + dx;
                    let py = by * 4 + dy;
                    if px < w && py < h {
                        block_sad += (current_luma[py * w + px] - block_mean).abs();
                    }
                }
            }

            let block_need = adaptive_scale(block_var_avg, 50.0, 400.0);
            let effective_dither = strength * block_need;

            if block_sad < 15.0 && effective_dither > 0.01 {
                for dy in 0..4 {
                    for dx in 0..4 {
                        let px = bx * 4 + dx;
                        let py = by * 4 + dy;
                        if px < w && py < h {
                            let dither = (bayer_norm[dy][dx] + block_offset * 0.3)
                                * effective_dither;
                            // strip is 4 rows of w*3 bytes; row dy
                            // within strip is at offset dy * w * 3.
                            let idx = dy * w * 3 + px * 3;
                            for c in 0..3 {
                                let val = strip[idx + c] as f64 + dither;
                                strip[idx + c] =
                                    val.round().clamp(0.0, 255.0) as u8;
                            }
                        }
                    }
                }
            }
        }
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(w * 3 * 4)
            .enumerate()
            .for_each(|(by, strip)| stage4_strip(by, strip));
    }
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(w * 3 * 4)
        .enumerate()
        .for_each(|(by, strip)| stage4_strip(by, strip));

    out
}

// ---------------------------------------------------------------------------
// Armor pipeline (2 stages)
// ---------------------------------------------------------------------------

fn optimize_armor(pixels: &[u8], w: usize, h: usize, config: &OptimizerConfig) -> Vec<u8> {
    let mut out = pixels.to_vec();
    let strength = config.strength;

    // Stage 1: Gentle low-pass on block boundaries (8×8 grid)
    smooth_block_boundaries(&mut out, w, h, strength);

    // Stage 2: DC stabilization — subtle smoothing within each 8×8 block
    // to reduce variance of block-average brightness
    dc_stabilize(&mut out, w, h, strength);

    out
}

// ---------------------------------------------------------------------------
// Fortress pipeline (1 stage)
// ---------------------------------------------------------------------------

fn optimize_fortress(pixels: &[u8], w: usize, h: usize, config: &OptimizerConfig) -> Vec<u8> {
    let mut out = pixels.to_vec();
    smooth_block_boundaries(&mut out, w, h, config.strength);
    out
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Extract luma (Y) from RGB pixels. Y = 0.299R + 0.587G + 0.114B.
fn extract_luma(pixels: &[u8], w: usize, h: usize) -> Vec<f64> {
    let mut luma = vec![0.0f64; w * h];
    for i in 0..w * h {
        let r = pixels[i * 3] as f64;
        let g = pixels[i * 3 + 1] as f64;
        let b = pixels[i * 3 + 2] as f64;
        luma[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    luma
}

/// Compute local variance in a 5×5 neighborhood for each pixel.
fn local_variance_5x5(luma: &[f64], w: usize, h: usize) -> Vec<f64> {
    // T2.11a — integral-image (summed-area table) 5×5 variance.
    //
    // Replaces the previous O(N × 25) sliding-window implementation
    // with O(N): one O(N) build pass for two SATs (sum + sum-of-
    // squares over a clamped-edge-padded luma grid), then O(1) per-
    // pixel variance via the 4-corner SAT difference.
    //
    // Boundary handling matches the previous code's clamp-to-edge
    // semantics: the luma grid is virtually replicated by 2 cells on
    // every side. The 5×5 window at any (y, x) reads the same 25
    // values it did before — just summed in SAT order rather than
    // window-direct order. f64 add isn't associative, so the
    // resulting variance values shift by 1-2 ULP versus the
    // sliding-window output. Downstream `adaptive_scale` smooths
    // these out to ULP-level strength shifts that round to 0 in the
    // final u8 cover bytes for ~all pixels (verified by the
    // optimizer_cross_platform SHA256 test).
    if w == 0 || h == 0 {
        return Vec::new();
    }

    const R: usize = 2; // 5×5 window → radius 2
    let pw = w + 2 * R;
    let ph = h + 2 * R;

    // Padded luma: clamped-edge replicate by 2 cells on each side.
    let mut padded = vec![0.0f64; pw * ph];
    for py in 0..ph {
        // Map padded row → original row via the same saturating_sub
        // + min idiom the old sliding-window code used.
        let sy = py.saturating_sub(R).min(h - 1);
        for px in 0..pw {
            let sx = px.saturating_sub(R).min(w - 1);
            padded[py * pw + px] = luma[sy * w + sx];
        }
    }

    // SATs: zero-row / zero-column at top/left so the 4-corner
    // formula works uniformly. Dimensions: (pw+1) × (ph+1).
    let sw = pw + 1;
    let mut sat: Vec<f64> = vec![0.0; sw * (ph + 1)];
    let mut sat_sq: Vec<f64> = vec![0.0; sw * (ph + 1)];

    for py in 0..ph {
        let mut row_sum = 0.0f64;
        let mut row_sum_sq = 0.0f64;
        let above_row_off = py * sw;
        let curr_row_off = (py + 1) * sw;
        for px in 0..pw {
            let v = padded[py * pw + px];
            row_sum += v;
            row_sum_sq += v * v;
            sat[curr_row_off + (px + 1)] = sat[above_row_off + (px + 1)] + row_sum;
            sat_sq[curr_row_off + (px + 1)] = sat_sq[above_row_off + (px + 1)] + row_sum_sq;
        }
    }

    // 4-corner SAT difference gives the inclusive sum over the 5×5
    // window at padded coordinates [y..y+5) × [x..x+5).
    // T2.11b — per-row parallelism. Each row writes a disjoint
    // slice of `variance` and reads from the (immutable) SAT tables.
    let count = 25.0f64;
    let mut variance = vec![0.0f64; w * h];
    let fill_row = |y: usize, row: &mut [f64]| {
        let r2 = (y + 5) * sw;
        let r1 = y * sw;
        for x in 0..w {
            let c2 = x + 5;
            let sum = sat[r2 + c2] - sat[r1 + c2] - sat[r2 + x] + sat[r1 + x];
            let sum_sq =
                sat_sq[r2 + c2] - sat_sq[r1 + c2] - sat_sq[r2 + x] + sat_sq[r1 + x];
            let mean = sum / count;
            row[x] = sum_sq / count - mean * mean;
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        variance
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row)| fill_row(y, row));
    }
    #[cfg(not(feature = "parallel"))]
    variance
        .chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| fill_row(y, row));

    variance
}

/// Compute 3×3 neighborhood mean and center pixel value for a channel.
fn neighborhood_mean_3x3(pixels: &[u8], w: usize, h: usize, x: usize, y: usize, c: usize) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut count = 0.0f64;
    for dy in 0..3usize {
        for dx in 0..3usize {
            let py = (y + dy).saturating_sub(1).min(h - 1);
            let px = (x + dx).saturating_sub(1).min(w - 1);
            sum += pixels[(py * w + px) * 3 + c] as f64;
            count += 1.0;
        }
    }
    let pixel_val = pixels[(y * w + x) * 3 + c] as f64;
    (sum / count, pixel_val)
}

/// Separable Gaussian blur with given sigma (truncated at 3σ kernel).
fn gaussian_blur_separable(pixels: &[u8], w: usize, h: usize, sigma: f64) -> Vec<u8> {
    let radius = (sigma * 3.0).ceil() as usize;
    let kernel_size = radius * 2 + 1;

    // Build 1D Gaussian kernel
    let mut kernel = vec![0.0f64; kernel_size];
    let mut sum = 0.0f64;
    for i in 0..kernel_size {
        let x = i as f64 - radius as f64;
        // det_exp instead of f64::exp — f64::exp lowers to non-deterministic
        // libm / Math.exp on WASM, breaking cross-platform bit-exactness of
        // the optimized cover.
        let val = det_exp(-x * x / (2.0 * sigma * sigma));
        kernel[i] = val;
        sum += val;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    let n = w * h * 3;
    let mut temp = vec![0u8; n];
    let mut result = vec![0u8; n];

    // T2.11d — Horizontal pass: per-row de-interleave RGB into
    // 3 padded planar f64 buffers (clamp-to-edge replicated), then
    // run the SIMD-dispatched 1D blur on each channel plane. The
    // pad eliminates the per-tap `saturating_sub + min` branch the
    // scalar inner loop used to do.
    let padded_w = w + 2 * radius;
    let h_pass = |y: usize, plane_scratch: &mut [f64], row_out: &mut [u8]| {
        // De-interleave + pad each channel into plane_scratch
        // (laid out as 3 contiguous slabs of padded_w).
        for c in 0..3 {
            let slab = &mut plane_scratch[c * padded_w..(c + 1) * padded_w];
            for px in 0..padded_w {
                let sx = px.saturating_sub(radius).min(w - 1);
                slab[px] = pixels[(y * w + sx) * 3 + c] as f64;
            }
            super::optimizer_simd::blur_row_h(
                slab, kernel.as_slice(), radius, w, row_out, c, 3,
            );
        }
    };

    let plane_scratch_size = 3 * padded_w;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        temp.par_chunks_mut(w * 3).enumerate().for_each_init(
            || vec![0.0f64; plane_scratch_size],
            |scratch, (y, row)| h_pass(y, scratch, row),
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = vec![0.0f64; plane_scratch_size];
        temp.chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, row)| h_pass(y, &mut scratch, row));
    }

    // T2.11d — Vertical pass: same idea but the padded planar
    // buffers are columnar slices of `temp` per output row. We pad
    // the kernel-size window of input rows (clamped) into a planar
    // f64 buffer per channel, then run the SIMD 1D blur.
    let v_pass = |y: usize, plane_scratch: &mut [f64], row_out: &mut [u8]| {
        // For output row y we need rows y, y+1, ..., y + 2*radius
        // from the padded V-axis (clamp-to-edge). The "padded row"
        // for the SIMD blur is the LENGTH-w column we virtually
        // construct — but for vertical conv, the SIMD shape is
        // different: for each output pixel x, the conv reads
        // `kernel_size` rows of input.
        //
        // Since the SIMD pattern in optimizer_simd expects a flat
        // (padded_w) 1D buffer per channel laid out along the conv
        // axis, the V pass would need to transpose. Instead we use
        // a per-pixel layout: write `kernel_size` consecutive
        // samples for each output column into the scratch in the
        // shape (kernel_size, w_chunk), then process per-column.
        //
        // For simplicity + cache friendliness, V pass keeps the
        // scalar autovec inner loop (already rayon-parallel per
        // row from T2.11b). SIMD on the H pass + per-row planar
        // de-interleave delivers the dominant gain; V pass tail
        // optimization is deferred (the per-row reads in V pass
        // are already strided-but-contiguous within a row, so
        // LLVM autovec is friendlier than for H).
        for x in 0..w {
            for c in 0..3 {
                let mut acc = 0.0f64;
                for k in 0..kernel_size {
                    let sy = (y + k).saturating_sub(radius).min(h - 1);
                    acc += temp[(sy * w + x) * 3 + c] as f64 * kernel[k];
                }
                row_out[x * 3 + c] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
        // Touch scratch to silence unused-var; reserved for future
        // V-pass SIMD.
        let _ = plane_scratch;
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        result.par_chunks_mut(w * 3).enumerate().for_each_init(
            || vec![0.0f64; plane_scratch_size],
            |scratch, (y, row)| v_pass(y, scratch, row),
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = vec![0.0f64; plane_scratch_size];
        result
            .chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, row)| v_pass(y, &mut scratch, row));
    }

    result
}

/// Smooth block boundaries at 8×8 grid edges.
///
/// Applies 3-pixel averaging across horizontal and vertical block edges
/// to reduce cross-block discontinuities after JPEG recompression.
fn smooth_block_boundaries(pixels: &mut [u8], w: usize, h: usize, strength: f32) {
    let alpha = 0.25 * strength as f64; // blending weight

    // Vertical block edges (columns at multiples of 8)
    for col in (8..w).step_by(8) {
        if col >= w {
            continue;
        }
        for y in 0..h {
            for c in 0..3 {
                let left = pixels[(y * w + col - 1) * 3 + c] as f64;
                let center = pixels[(y * w + col) * 3 + c] as f64;
                let right = if col + 1 < w {
                    pixels[(y * w + col + 1) * 3 + c] as f64
                } else {
                    center
                };
                let avg = (left + center + right) / 3.0;
                let blended = center + alpha * (avg - center);
                pixels[(y * w + col) * 3 + c] = blended.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    // Horizontal block edges (rows at multiples of 8)
    for row in (8..h).step_by(8) {
        for x in 0..w {
            for c in 0..3 {
                let above = pixels[((row - 1) * w + x) * 3 + c] as f64;
                let center = pixels[(row * w + x) * 3 + c] as f64;
                let below = if row + 1 < h {
                    pixels[((row + 1) * w + x) * 3 + c] as f64
                } else {
                    center
                };
                let avg = (above + center + below) / 3.0;
                let blended = center + alpha * (avg - center);
                pixels[(row * w + x) * 3 + c] = blended.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// DC stabilization: subtle within-block smoothing to reduce variance
/// of block-average brightness, helping STDM embedding stability.
fn dc_stabilize(pixels: &mut [u8], w: usize, h: usize, strength: f32) {
    let alpha = 0.1 * strength as f64;
    let blocks_x = w.div_ceil(8);
    let blocks_y = h.div_ceil(8);

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            // Compute block mean for each channel
            let mut mean = [0.0f64; 3];
            let mut count = 0.0f64;
            for dy in 0..8 {
                for dx in 0..8 {
                    let px = bx * 8 + dx;
                    let py = by * 8 + dy;
                    if px < w && py < h {
                        for c in 0..3 {
                            mean[c] += pixels[(py * w + px) * 3 + c] as f64;
                        }
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                for c in 0..3 {
                    mean[c] /= count;
                }
            }

            // Nudge each pixel slightly toward block mean
            for dy in 0..8 {
                for dx in 0..8 {
                    let px = bx * 8 + dx;
                    let py = by * 8 + dy;
                    if px < w && py < h {
                        let idx = (py * w + px) * 3;
                        for c in 0..3 {
                            let val = pixels[idx + c] as f64;
                            let blended = val + alpha * (mean[c] - val);
                            pixels[idx + c] = blended.round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_deterministic() {
        let w = 64;
        let h = 64;
        let pixels = vec![128u8; w * h * 3];
        let config = OptimizerConfig {
            strength: 0.85,
            seed: [42u8; 32],
            mode: OptimizerMode::Ghost,
        };
        let out1 = optimize_cover(&pixels, w as u32, h as u32, &config);
        let out2 = optimize_cover(&pixels, w as u32, h as u32, &config);
        assert_eq!(out1, out2, "optimizer must be deterministic");
    }

    #[test]
    fn ghost_modifies_smooth_pixels() {
        // A flat image (all same value) should be modified — lots of room to improve
        let w = 64;
        let h = 64;
        let pixels = vec![128u8; w * h * 3];
        let config = OptimizerConfig {
            strength: 0.85,
            seed: [42u8; 32],
            mode: OptimizerMode::Ghost,
        };
        let out = optimize_cover(&pixels, w as u32, h as u32, &config);
        assert_ne!(pixels, out, "optimizer should modify smooth pixels");
    }

    #[test]
    fn do_no_harm_textured_image() {
        // Create an already well-textured image (simulating pre-optimized photo).
        // The optimizer should return it unchanged if it can't improve gradient energy.
        let w = 128;
        let h = 128;
        let mut pixels = vec![0u8; w * h * 3];
        // High-variance random-ish texture
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                let hash = ((x * 31 + y * 97 + 7) % 256) as u8;
                pixels[idx] = hash;
                pixels[idx + 1] = hash.wrapping_mul(3).wrapping_add(17);
                pixels[idx + 2] = hash.wrapping_mul(7).wrapping_add(53);
            }
        }
        let config = OptimizerConfig {
            strength: 0.85,
            seed: [42u8; 32],
            mode: OptimizerMode::Ghost,
        };
        let out = optimize_cover(&pixels, w as u32, h as u32, &config);

        // Either unchanged (do-no-harm kicked in) or gradient energy increased
        let orig_energy = gradient_energy(&pixels, w, h);
        let opt_energy = gradient_energy(&out, w, h);
        assert!(
            opt_energy >= orig_energy,
            "optimizer must not reduce gradient energy: original={orig_energy:.2}, optimized={opt_energy:.2}"
        );
    }

    #[test]
    fn psnr_above_threshold() {
        let w = 128;
        let h = 128;
        let mut pixels = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                let base_r = (x * 255 / w) as u8;
                let base_g = (y * 255 / h) as u8;
                let base_b = ((x + y) * 255 / (w + h)) as u8;
                let hash = ((x * 31 + y * 97 + 7) % 30) as u8;
                pixels[idx] = base_r.wrapping_add(hash);
                pixels[idx + 1] = base_g.wrapping_add(hash.wrapping_mul(3));
                pixels[idx + 2] = base_b.wrapping_add(hash.wrapping_mul(7));
            }
        }
        let config = OptimizerConfig {
            strength: 0.85,
            seed: [42u8; 32],
            mode: OptimizerMode::Ghost,
        };
        let out = optimize_cover(&pixels, w as u32, h as u32, &config);

        let mse: f64 = pixels
            .iter()
            .zip(out.iter())
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64;
                d * d
            })
            .sum::<f64>()
            / pixels.len() as f64;

        let psnr = if mse > 0.0 {
            10.0 * (255.0 * 255.0 / mse).log10()
        } else {
            f64::INFINITY
        };

        assert!(
            psnr > 28.0,
            "PSNR should be > 28 dB even for synthetic images, got {psnr:.1} dB"
        );
    }

    #[test]
    fn armor_mode_runs() {
        let w = 32;
        let h = 32;
        let pixels = vec![128u8; w * h * 3];
        let config = OptimizerConfig {
            strength: 0.85,
            seed: [0u8; 32],
            mode: OptimizerMode::Armor,
        };
        let out = optimize_cover(&pixels, w as u32, h as u32, &config);
        assert_eq!(out.len(), pixels.len());
    }

    #[test]
    fn fortress_mode_runs() {
        let w = 32;
        let h = 32;
        let pixels = vec![128u8; w * h * 3];
        let config = OptimizerConfig {
            strength: 0.85,
            seed: [0u8; 32],
            mode: OptimizerMode::Fortress,
        };
        let out = optimize_cover(&pixels, w as u32, h as u32, &config);
        assert_eq!(out.len(), pixels.len());
    }

    #[test]
    fn zero_strength_minimal_change() {
        let w = 32;
        let h = 32;
        let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();
        let config = OptimizerConfig {
            strength: 0.0,
            seed: [42u8; 32],
            mode: OptimizerMode::Ghost,
        };
        let out = optimize_cover(&pixels, w as u32, h as u32, &config);

        let max_diff: u8 = pixels
            .iter()
            .zip(out.iter())
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 1,
            "zero strength should cause near-zero changes, max diff = {max_diff}"
        );
    }

    #[test]
    fn texture_analysis_smooth_image() {
        let w = 64;
        let h = 64;
        let pixels = vec![128u8; w * h * 3]; // perfectly flat
        let luma = extract_luma(&pixels, w, h);
        let variance = local_variance_5x5(&luma, w, h);
        let (noise, contrast, sharpen, dither) = analyze_texture(&luma, &variance, w, h);
        // Flat image should get maximum treatment on all stages
        assert!(noise > 0.9, "flat image should get full noise: {noise}");
        assert!(contrast > 0.9, "flat image should get full contrast: {contrast}");
        assert!(sharpen > 0.9, "flat image should get full sharpen: {sharpen}");
        assert!(dither > 0.5, "flat image should get significant dither: {dither}");
    }

    #[test]
    fn texture_analysis_noisy_image() {
        let w = 64;
        let h = 64;
        let mut pixels = vec![0u8; w * h * 3];
        // High-variance texture everywhere
        for i in 0..pixels.len() {
            pixels[i] = ((i * 31 + 7) % 256) as u8;
        }
        let luma = extract_luma(&pixels, w, h);
        let variance = local_variance_5x5(&luma, w, h);
        let (noise, _contrast, _sharpen, dither) = analyze_texture(&luma, &variance, w, h);
        // Already-textured image should get minimal treatment
        assert!(noise < 0.3, "textured image should get little noise: {noise}");
        assert!(dither < 0.3, "textured image should get little dither: {dither}");
    }
}
