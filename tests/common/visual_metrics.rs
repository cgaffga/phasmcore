// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Luma visual-quality metrics shared by the video visual gates.
//!
//! Moved verbatim out of `openh264_visual_psnr_regression.rs` (Phase C.2,
//! #401-#404) so more than one gate can measure the same way. The
//! allocation-shape fidelity comparison needs exactly these four metrics at
//! per-GOP granularity, and a second copy would be a second set of numbers
//! that silently drift apart.
//!
//! All four operate on a single Y plane; callers slice the frame out of a
//! yuv420p buffer themselves.

#![allow(dead_code)] // not every gate uses every metric

/// Whole-frame Y-PSNR. 99.0 when MSE==0 (identical).
pub fn psnr_y(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut mse = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        mse += d * d;
    }
    mse /= a.len() as f64;
    if mse == 0.0 { 99.0 } else { 10.0 * (255.0_f64 * 255.0 / mse).log10() }
}

/// Worst-16×16-MB PSNR. Slides a 16×16 window over the Y plane and
/// returns the minimum PSNR found at any MB position, plus its (mb_x,
/// mb_y) location. Catches localized reconstruction failures that
/// whole-frame PSNR averages out.
pub fn worst_mb_psnr_y(src: &[u8], dec: &[u8], width: u32, height: u32) -> (f64, u32, u32) {
    let mut worst = f64::INFINITY;
    let mut wx = 0u32;
    let mut wy = 0u32;
    let w = width as usize;
    for mb_y in 0..(height / 16) {
        for mb_x in 0..(width / 16) {
            let mut mse = 0.0_f64;
            for dy in 0..16 {
                for dx in 0..16 {
                    let off = (mb_y as usize * 16 + dy) * w + (mb_x as usize * 16 + dx);
                    let d = src[off] as f64 - dec[off] as f64;
                    mse += d * d;
                }
            }
            mse /= 256.0;
            let p = if mse == 0.0 { 99.0 } else { 10.0 * (255.0_f64 * 255.0 / mse).log10() };
            if p < worst {
                worst = p;
                wx = mb_x;
                wy = mb_y;
            }
        }
    }
    (worst, wx, wy)
}

/// Count pixels with |src−dec| > `delta_threshold`. Real spec-correct
/// h.264 at QP≤26 produces single-digit counts; localized bugs produce
/// 100s.
pub fn count_bad_pixels(a: &[u8], b: &[u8], delta_threshold: u8) -> u32 {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| (**x as i32 - **y as i32).unsigned_abs() > delta_threshold as u32)
        .count() as u32
}

/// C.2.2 (#403) — luma SSIM with 8×8 non-overlapping windows.
///
/// Wang et al. 2004 formulation. Constants per the reference:
///   C1 = (0.01 × 255)² ≈ 6.5025
///   C2 = (0.03 × 255)² ≈ 58.5225
/// Returns the mean SSIM across all 8×8 windows. 1.0 = identical,
/// ≥0.95 = excellent perceptual quality on stego output at QP=26.
pub fn ssim_y(a: &[u8], b: &[u8], width: u32, height: u32) -> f64 {
    assert_eq!(a.len(), b.len());
    const C1: f64 = 6.5025;
    const C2: f64 = 58.5225;
    const WIN: usize = 8;
    let w = width as usize;
    let n_wx = (width as usize) / WIN;
    let n_wy = (height as usize) / WIN;
    if n_wx == 0 || n_wy == 0 {
        return 1.0;
    }
    let mut sum = 0.0_f64;
    let mut count = 0_u32;
    let n = (WIN * WIN) as f64;
    for wy in 0..n_wy {
        for wx in 0..n_wx {
            let mut sum_a = 0.0_f64;
            let mut sum_b = 0.0_f64;
            for dy in 0..WIN {
                for dx in 0..WIN {
                    let off = (wy * WIN + dy) * w + (wx * WIN + dx);
                    sum_a += a[off] as f64;
                    sum_b += b[off] as f64;
                }
            }
            let mu_a = sum_a / n;
            let mu_b = sum_b / n;
            let mut var_a = 0.0_f64;
            let mut var_b = 0.0_f64;
            let mut cov = 0.0_f64;
            for dy in 0..WIN {
                for dx in 0..WIN {
                    let off = (wy * WIN + dy) * w + (wx * WIN + dx);
                    let da = a[off] as f64 - mu_a;
                    let db = b[off] as f64 - mu_b;
                    var_a += da * da;
                    var_b += db * db;
                    cov += da * db;
                }
            }
            var_a /= n;
            var_b /= n;
            cov /= n;
            let num = (2.0 * mu_a * mu_b + C1) * (2.0 * cov + C2);
            let den = (mu_a * mu_a + mu_b * mu_b + C1) * (var_a + var_b + C2);
            sum += num / den;
            count += 1;
        }
    }
    sum / count as f64
}

// ─────────────────────────── thresholds ───────────────────────────────
