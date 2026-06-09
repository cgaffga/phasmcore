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

/// Cross-platform empirical optimizer-output byte equivalence.
///
/// Returns the lowercase-hex SHA256 of `optimize_cover` output on a
/// fixed deterministic input. Used to verify the optimizer produces
/// bit-identical output across native / iOS / Android / WASM, and
/// to catch any regression in the optimizer kernels.
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
///
/// M9.5.C1 (2026-05-22) — Takes `&[u8]` luma + `&[u16]` variance to
/// match the type-narrowed buffers. Internal math stays f64.
#[cfg(test)]
fn analyze_texture(luma: &[u8], variance: &[u16], w: usize, h: usize) -> (f64, f64, f64, f64) {
    let n = w * h;
    if n == 0 {
        return (1.0, 1.0, 1.0, 1.0);
    }

    // 1. Noise factor: based on mean variance.
    //    Low variance (< 50) = smooth image, needs lots of noise → factor ~1.0
    //    Medium variance (50-300) = some texture → moderate noise
    //    High variance (> 600) = already noisy → skip noise
    let mean_variance: f64 = variance.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let noise_factor = adaptive_scale(mean_variance, 50.0, 600.0);

    // 2. Contrast factor: based on proportion of pixels with low local contrast.
    //    Compute local gradient magnitude as a contrast proxy.
    let mut low_contrast_count = 0usize;
    for y in 1..h {
        for x in 1..w {
            let idx = y * w + x;
            let dy = (luma[idx] as f64 - luma[(y - 1) * w + x] as f64).abs();
            let dx = (luma[idx] as f64 - luma[y * w + (x - 1) as usize] as f64).abs();
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
            let center = luma[idx] as f64;
            // 4-neighbor average
            let avg = (luma[(y - 1) * w + x] as f64
                + luma[(y + 1) * w + x] as f64
                + luma[y * w + (x - 1)] as f64
                + luma[y * w + (x + 1)] as f64)
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
                    block_mean += luma[(by * 4 + dy) * w + (bx * 4 + dx)] as f64;
                }
            }
            block_mean /= 16.0;
            let mut sad = 0.0f64;
            for dy in 0..4 {
                for dx in 0..4 {
                    sad += (luma[(by * 4 + dy) * w + (bx * 4 + dx)] as f64 - block_mean).abs();
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

// Per-strip pipeline (memory-bound refactor, see
// docs/design/image/memory-budget-2026-05.md).
//
// The optimizer previously ran as a "stage-major" pipeline: Stage 1
// over all rows in parallel, Stage 2 over all rows in parallel, etc.
// Each stage allocated full-image working buffers (variance 460 MB at
// 60 MP, stage1 clone 173 MB, blurred 173 MB, current_luma 460 MB) and
// rayon workers shuffled between stages, blowing cache between every
// transition.
//
// The strip refactor inverts the loop: each rayon worker takes a strip
// and runs ALL 4 STAGES on it. Per-strip working set (~10-20 MB at 60
// MP / 64-row strips) fits L2/L3 and stays hot across all stages. The
// 460 MB shared `variance` allocation disappears entirely — each
// strip computes its own per-strip variance from its own per-strip
// luma, and both get dropped at strip end.
//
// Strip overlap: 4 rows above/below the output range, clamped at
// image boundary. Cascading kernel reach analysis:
//   variance 5×5 → ±2 of input luma
//   Stage 2 3×3  → ±1 of Stage 1 output → needs Stage 1 correct on ±1
//   Stage 3 blur σ=1, radius=3 → ±3 of Stage 2 output → needs Stage
//                                2 correct on ±3 → needs Stage 1
//                                correct on ±4
//   Stage 4: per 4×4 block, no inter-row read past block bounds
//   Variance for output rows: needs luma on ±2 rows of OVERLAP
//   Max needed overlap = max(2 luma, 4 stage cascade) = 4 rows.
//
// Output bytes match the stage-major byte-exact baseline:
//   - Per-row ChaCha: `set_word_pos(4 * abs_y * w)` keeps the
//     same RNG stream byte the original serial pre-pass produced for
//     that row's pixels.
//   - Per-strip Stage 4 ChaCha: `set_word_pos(h*w*4 + by*blocks_x*2)`
//     matches the post-Stage-1 RNG position.
//   - Variance computed per-strip differs from full-image only at
//     the work-buffer edge rows (OVERLAP region), which we DON'T
//     output. For output rows, per-strip variance == full-image
//     variance (5×5 kernel within work buffer, OVERLAP=4 ≥ kernel
//     radius 2).
//
// STRIP_HEIGHT must be 4-aligned because Stage 4 processes 4×4
// blocks and we partition by block-rows.
//
// Tuning (2026-05-22): 256 rows bench-best on M-series 8-core. At
// 64 rows the per-strip overhead (input copy + Vec allocation +
// 12.5% overlap recompute over the 8-row OVERLAP/strip_h ratio)
// dominated the cache locality win. At 256 the overlap ratio drops
// to 3% and per-strip work amortises better. Working set per strip
// at 60 MP / 9520 cols × 256 rows × ~7 bytes/pixel ≈ 17 MB — fits
// L3 on Apple Silicon (24 MB per cluster).
const STRIP_HEIGHT: usize = 256;
const STRIP_OVERLAP: usize = 4;
const SQRT_6: f64 = 2.449489742783178;

fn optimize_ghost(pixels: &[u8], w: usize, h: usize, config: &OptimizerConfig) -> Vec<u8> {
    let strength = config.strength as f64;
    let seed = config.seed;

    let blocks_x = w / 4;
    let blocks_y = h / 4;

    // Allocate the output buffer once and write strip outputs directly
    // into disjoint `par_chunks_mut` slices — avoids a `collect()`
    // intermediate that would hold every strip's output Vec
    // simultaneously (~180 MB additional peak at 60 MP).
    let mut out = pixels.to_vec();

    // Per-strip closure: runs the full optimize pipeline on one
    // strip's worth of rows, writing directly into `out_chunk` (the
    // strip's slice of the shared output buffer).
    let process_strip = |strip_idx: usize, out_chunk: &mut [u8]| {
        let y_start = strip_idx * STRIP_HEIGHT;
        let y_end = ((strip_idx + 1) * STRIP_HEIGHT).min(h);
        let strip_h = y_end - y_start;

        let work_y_start = y_start.saturating_sub(STRIP_OVERLAP);
        let work_y_end = (y_end + STRIP_OVERLAP).min(h);
        let work_h = work_y_end - work_y_start;

        // Local-row offset of the OUTPUT range within the work buffer
        // (== STRIP_OVERLAP for interior strips, 0 for strip 0).
        let output_local_start = y_start - work_y_start;

        // Copy input strip rows (with overlap) into the working buffer.
        let mut work = pixels[work_y_start * w * 3..work_y_end * w * 3].to_vec();

        // Per-strip luma + variance. Computed from the working buffer
        // luma; matches full-image variance for OUTPUT rows because
        // STRIP_OVERLAP=4 ≥ variance 5×5 radius=2.
        let work_variance = {
            let work_luma = extract_luma(&work, w, work_h);
            local_variance_5x5(&work_luma, w, work_h)
        };

        // === Stage 1: per-row noise injection ===
        // Per-row RNG: `set_word_pos(4 * abs_y * w)` so byte order
        // matches the original serial pre-pass.
        for local_y in 0..work_h {
            let abs_y = work_y_start + local_y;
            let mut row_rng = ChaCha20Rng::from_seed(seed);
            row_rng.set_word_pos((4u128) * (abs_y as u128) * (w as u128));

            let row_off = local_y * w * 3;
            for x in 0..w {
                // Always draw — matches the original unconditional pre-pass.
                let u1: f64 = row_rng.gen_range(0.0f64..1.0).max(1e-10);
                let u2: f64 = row_rng.gen_range(0.0f64..1.0);

                let var_val = work_variance[local_y * w + x] as f64;
                let local_need = adaptive_scale(var_val, 50.0, 600.0);
                if local_need > 0.1 {
                    let sigma = 1.5 * local_need + 0.1;
                    let scaled_sigma = sigma * strength * local_need;
                    if scaled_sigma > 0.05 {
                        let gauss = (u1 + u2 - 1.0) * SQRT_6;
                        let noise = gauss * scaled_sigma;
                        for c in 0..3 {
                            let idx = row_off + x * 3 + c;
                            let val = work[idx] as f64 + noise;
                            work[idx] = val.round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }

        // === Stage 2: 3×3 unsharp ===
        // Needs pre-Stage-2 snapshot. Per-strip clone is ~2 MB at 60 MP
        // (was a 173 MB full-image clone under the stage-major pipeline).
        let work_stage1 = work.clone();
        for local_y in 0..work_h {
            let row_off = local_y * w * 3;
            for x in 0..w {
                let var_val = work_variance[local_y * w + x] as f64;
                let local_need = adaptive_scale(var_val, 80.0, 500.0);
                if local_need > 0.1 {
                    let local_enhance = 0.15 * strength * local_need;
                    if local_enhance > 0.01 {
                        for c in 0..3 {
                            // neighborhood_mean_3x3 clamps at work-buffer
                            // edges; for OUTPUT rows with OVERLAP≥1 the
                            // kernel stays inside the work buffer.
                            let (local_mean, pixel_val) =
                                neighborhood_mean_3x3(&work_stage1, w, work_h, x, local_y, c);
                            let deviation = pixel_val - local_mean;
                            let enhanced = local_mean + deviation * (1.0 + local_enhance);
                            work[row_off + x * 3 + c] =
                                enhanced.round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }
        drop(work_stage1);

        // === Stage 3: blur + unsharp mask ===
        let blurred = gaussian_blur_separable(&work, w, work_h, 1.0);
        for local_y in 0..work_h {
            let row_off = local_y * w * 3;
            for x in 0..w {
                let var_val = work_variance[local_y * w + x] as f64;
                let local_need = adaptive_scale(var_val, 100.0, 400.0);
                if local_need > 0.1 {
                    let local_sharpen = 0.12 * strength * local_need;
                    if local_sharpen > 0.01 {
                        for c in 0..3 {
                            let idx = row_off + x * 3 + c;
                            let orig = work[idx] as f64;
                            let blur = blurred[idx] as f64;
                            let diff = orig - blur;
                            if diff.abs() > 3.0 {
                                let sharpened = orig + local_sharpen * diff;
                                work[idx] = sharpened.round().clamp(0.0, 255.0) as u8;
                            }
                        }
                    }
                }
            }
        }
        drop(blurred);

        // === Stage 4: per-block dither ===
        // current_luma re-extracted from post-Stage-3 strip work buffer.
        let work_current_luma = extract_luma(&work, w, work_h);

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

        // Block-row range within the strip's OUTPUT range. STRIP_HEIGHT
        // is 4-aligned so y_start % 4 == 0. y_end might not be 4-aligned
        // on the trailing strip (h not a multiple of 4); clamp to
        // `blocks_y` to drop the partial bottom block, matching the
        // stage-major Stage 4 behavior.
        let block_y_start = y_start / 4;
        let block_y_end = (y_end / 4).min(blocks_y);

        for by in block_y_start..block_y_end {
            // Per-strip Stage 4 RNG byte order:
            //   word_pos = h * w * 4 (post-Stage-1) + by * blocks_x * 2
            let mut row_rng = ChaCha20Rng::from_seed(seed);
            row_rng.set_word_pos(
                (h as u128) * (w as u128) * 4u128
                    + (by as u128) * (blocks_x as u128) * 2u128,
            );

            let block_local_y = by * 4 - work_y_start;

            for bx in 0..blocks_x {
                let block_offset = row_rng.gen_range(0.0f64..1.0) * 2.0 - 1.0;

                let mut block_mean = 0.0f64;
                let mut block_var_sum = 0.0f64;
                for dy in 0..4 {
                    for dx in 0..4 {
                        let px = bx * 4 + dx;
                        let py_local = block_local_y + dy;
                        let py_abs = by * 4 + dy;
                        if px < w && py_abs < h {
                            block_mean += work_current_luma[py_local * w + px] as f64;
                            block_var_sum += work_variance[py_local * w + px] as f64;
                        }
                    }
                }
                block_mean /= 16.0;
                let block_var_avg = block_var_sum / 16.0;
                let mut block_sad = 0.0f64;
                for dy in 0..4 {
                    for dx in 0..4 {
                        let px = bx * 4 + dx;
                        let py_local = block_local_y + dy;
                        let py_abs = by * 4 + dy;
                        if px < w && py_abs < h {
                            block_sad += (work_current_luma[py_local * w + px] as f64 - block_mean).abs();
                        }
                    }
                }

                let block_need = adaptive_scale(block_var_avg, 50.0, 400.0);
                let effective_dither = strength * block_need;

                if block_sad < 15.0 && effective_dither > 0.01 {
                    for dy in 0..4 {
                        for dx in 0..4 {
                            let px = bx * 4 + dx;
                            let py_local = block_local_y + dy;
                            let py_abs = by * 4 + dy;
                            if px < w && py_abs < h {
                                let dither = (bayer_norm[dy][dx] + block_offset * 0.3)
                                    * effective_dither;
                                let idx = py_local * w * 3 + px * 3;
                                for c in 0..3 {
                                    let val = work[idx + c] as f64 + dither;
                                    work[idx + c] = val.round().clamp(0.0, 255.0) as u8;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Write OUTPUT rows directly into the caller-provided
        // `out_chunk` (skip overlap).
        let out_start = output_local_start * w * 3;
        let out_end = out_start + strip_h * w * 3;
        out_chunk[..strip_h * w * 3].copy_from_slice(&work[out_start..out_end]);
    };

    // Parallel over strips via `par_chunks_mut` on the shared output
    // buffer. Each rayon worker holds one strip's working set (~10-20
    // MB at 60 MP) which fits L2/L3 cache and stays hot across all 4
    // stages — the cache-locality win this refactor is designed to
    // deliver. Writing directly into the output buffer avoids a
    // `collect()` intermediate that would peak at ~180 MB extra at
    // 60 MP.
    let strip_byte_size = STRIP_HEIGHT * w * 3;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(strip_byte_size)
            .enumerate()
            .for_each(|(strip_idx, out_chunk)| process_strip(strip_idx, out_chunk));
    }
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(strip_byte_size)
        .enumerate()
        .for_each(|(strip_idx, out_chunk)| process_strip(strip_idx, out_chunk));

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
///
/// Returns `Vec<u8>` (not `Vec<f64>`): Y values are in [0, 255] integer
/// range anyway; the f64 storage was 8× memory for zero precision
/// benefit downstream (variance integerizes luma in its inner loop;
/// Stage 4 block stats accumulate in f32/f64 on demand). Saves 60 MB at
/// 60 MP full-image (or ~5 MB per strip).
fn extract_luma(pixels: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut luma = vec![0u8; w * h];
    for i in 0..w * h {
        let r = pixels[i * 3] as f32;
        let g = pixels[i * 3 + 1] as f32;
        let b = pixels[i * 3 + 2] as f32;
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        luma[i] = y.round().clamp(0.0, 255.0) as u8;
    }
    luma
}

/// Compute local variance in a 5×5 neighborhood for each pixel.
///
/// Direct sliding-window sum (a summed-area-table variant was tried and
/// reverted). Type narrowed to `&[u8]` luma in, `Vec<u16>` variance out:
/// sliding-window sum/sum_sq computed in u32 (max sum² ≈ 40.6M fits u26
/// < u32 max). Max possible variance for 8-bit luma in a 5×5 window is
/// ~16,256 (8-bit Bernoulli, half-zero/half-255), fits u14 with
/// headroom. Saves 460 MB → 120 MB full-image / ~5.5 MB → 1.3 MB per
/// strip.
///
/// Integer formula: `25 * sum_sq - sum²` is the numerator of
/// `25² · variance`. Divide by 625 to recover variance. All inputs
/// non-negative; numerator is exactly non-negative (Cauchy-Schwarz),
/// so u32 arithmetic is safe and bit-exact across platforms.
///
/// Helper is called per-strip from `process_strip` inside
/// `optimize_ghost`'s outer strip par_iter; kept serial here so
/// inner par_chunks_mut doesn't oversubscribe the rayon pool.
fn local_variance_5x5(luma: &[u8], w: usize, h: usize) -> Vec<u16> {
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let mut variance = vec![0u16; w * h];

    let fill_row = |y: usize, row: &mut [u16]| {
        for x in 0..w {
            let mut sum: u32 = 0;
            let mut sum_sq: u32 = 0;
            for dy in 0..5usize {
                let py = (y + dy).saturating_sub(2).min(h - 1);
                let row_off = py * w;
                for dx in 0..5usize {
                    let px = (x + dx).saturating_sub(2).min(w - 1);
                    let v = luma[row_off + px] as u32;
                    sum += v;
                    sum_sq += v * v;
                }
            }
            // Exact integer variance scaled by 25². See doc comment.
            let numerator = 25 * sum_sq - sum * sum;
            row[x] = (numerator / 625) as u16;
        }
    };

    // Inner par_chunks_mut. Nested rayon here (called from
    // `process_strip` inside the outer strip par_iter) is fine in
    // production — the optimizer runs once per encode, no contention.
    // The serial-within-strip version was 25-30% slower on M-series
    // 8-core; inner parallelism closes that gap while keeping the strip
    // refactor's memory savings.
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
///
/// Kernel + plane scratch narrowed to f32 so the SIMD blur dispatch
/// (`optimizer_simd::blur_row_h`) uses native f32x4 lanes (NEON FMA on
/// aarch64, native WASM SIMD128 f32x4, SSE2 f32x4 on x86) — 2× the lane
/// count of the earlier f64x2 path. `det_exp` is still used for the
/// kernel builder so the kernel values themselves are deterministic
/// across platforms; the f64 intermediate is cast to f32 before
/// normalisation.
fn gaussian_blur_separable(pixels: &[u8], w: usize, h: usize, sigma: f64) -> Vec<u8> {
    let radius = (sigma * 3.0).ceil() as usize;
    let kernel_size = radius * 2 + 1;

    // Build 1D Gaussian kernel. The kernel is small (7 entries for
    // sigma=1) and only built once per blur call; computing in f64
    // (deterministic via det_exp) then casting to f32 for storage
    // keeps the kernel-build precision while making the inner-loop
    // arithmetic f32.
    let mut kernel = vec![0.0f32; kernel_size];
    let mut sum = 0.0f64;
    for i in 0..kernel_size {
        let x = i as f64 - radius as f64;
        let val = det_exp(-x * x / (2.0 * sigma * sigma));
        kernel[i] = val as f32;
        sum += val;
    }
    let inv_sum = 1.0 / sum as f32;
    for k in kernel.iter_mut() {
        *k *= inv_sum;
    }

    let n = w * h * 3;
    let mut temp = vec![0u8; n];
    let mut result = vec![0u8; n];

    // Horizontal pass: per-row de-interleave RGB into 3 padded planar
    // f32 buffers (clamp-to-edge replicated), then run the SIMD-
    // dispatched 1D blur on each channel plane. The pad eliminates
    // the per-tap `saturating_sub + min` branch.
    let padded_w = w + 2 * radius;
    let h_pass = |y: usize, plane_scratch: &mut [f32], row_out: &mut [u8]| {
        for c in 0..3 {
            let slab = &mut plane_scratch[c * padded_w..(c + 1) * padded_w];
            for px in 0..padded_w {
                let sx = px.saturating_sub(radius).min(w - 1);
                slab[px] = pixels[(y * w + sx) * 3 + c] as f32;
            }
            super::optimizer_simd::blur_row_h(
                slab, kernel.as_slice(), radius, w, row_out, c, 3,
            );
        }
    };

    // Inner par_chunks_mut. See local_variance_5x5 comment above.
    let plane_scratch_size = 3 * padded_w;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        temp.par_chunks_mut(w * 3).enumerate().for_each_init(
            || vec![0.0f32; plane_scratch_size],
            |scratch, (y, row)| h_pass(y, scratch, row),
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = vec![0.0f32; plane_scratch_size];
        temp.chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, row)| h_pass(y, &mut scratch, row));
    }

    // Vertical pass: scalar in f32. Per-row reads of `temp` are
    // strided (RGB-interleaved) which LLVM autovec handles. SIMD
    // on the V axis would need column transposition; deferred.
    let v_pass = |y: usize, _plane_scratch: &mut [f32], row_out: &mut [u8]| {
        for x in 0..w {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for k in 0..kernel_size {
                    let sy = (y + k).saturating_sub(radius).min(h - 1);
                    acc += temp[(sy * w + x) * 3 + c] as f32 * kernel[k];
                }
                row_out[x * 3 + c] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        result.par_chunks_mut(w * 3).enumerate().for_each_init(
            || vec![0.0f32; plane_scratch_size],
            |scratch, (y, row)| v_pass(y, scratch, row),
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut v_scratch = vec![0.0f32; plane_scratch_size];
        result
            .chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, row)| v_pass(y, &mut v_scratch, row));
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
