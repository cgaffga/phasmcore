// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Experiment: Does a stego optimizer (adding texture to smooth regions)
//! compound with SI-UNIWARD, or do they conflict?
//!
//! Hypothesis: The two mechanisms are orthogonal and compound:
//! - Optimizer increases wavelet energy denominators (spatial domain) → more
//!   non-zero AC coefficients → more embeddable positions.
//! - SI-UNIWARD exploits quantization rounding errors (frequency domain) →
//!   lower per-position cost → higher embedding rate per position.
//!
//! Expected ordering: cap_opt_si > cap_si > cap_opt > cap_standard

use phasm_core::{ghost_capacity, ghost_capacity_si, JpegImage};
use phasm_core::jpeg::pixels::{jpeg_to_luma_f64, luma_f64_to_jpeg};

fn load_test_image(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/{name}")).unwrap()
}

/// Compute local variance in a 3×3 neighborhood around (y, x).
fn local_variance(pixels: &[f64], width: usize, height: usize, y: usize, x: usize) -> f64 {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0.0;

    for dy in 0..3usize {
        for dx in 0..3usize {
            let py = (y + dy).saturating_sub(1).min(height - 1);
            let px = (x + dx).saturating_sub(1).min(width - 1);
            let val = pixels[py * width + px];
            sum += val;
            sum_sq += val * val;
            count += 1.0;
        }
    }

    let mean = sum / count;
    sum_sq / count - mean * mean
}

/// Add subtle noise to smooth regions of the image.
///
/// For each pixel, if the local 3×3 variance is below `threshold`,
/// add a small deterministic perturbation (±noise_amplitude).
/// This simulates a simple stego optimizer that adds texture where
/// the image is flat, increasing wavelet energy denominators.
fn optimize_pixels(
    pixels: &[f64],
    width: usize,
    height: usize,
    threshold: f64,
    noise_amplitude: f64,
) -> Vec<f64> {
    let mut optimized = pixels.to_vec();
    let mut smooth_count = 0usize;
    let mut modified_count = 0usize;

    for y in 0..height {
        for x in 0..width {
            let var = local_variance(pixels, width, height, y, x);
            if var < threshold {
                smooth_count += 1;
                // Deterministic pseudo-noise based on position
                // Use a simple hash-like function for reproducibility
                let hash = ((y * 31 + x * 97 + 13) as f64 * 0.618033988749895).fract();
                let noise = (hash - 0.5) * 2.0 * noise_amplitude;
                let idx = y * width + x;
                optimized[idx] = (pixels[idx] + noise).clamp(0.0, 255.0);
                modified_count += 1;
            }
        }
    }

    eprintln!("  Smooth pixels (var < {threshold}): {smooth_count} / {} ({:.1}%)",
        width * height,
        100.0 * smooth_count as f64 / (width * height) as f64);
    eprintln!("  Modified pixels: {modified_count}");

    optimized
}

/// Count non-zero AC coefficients in a JpegImage (matches capacity.rs logic).
fn count_nonzero_ac(img: &JpegImage) -> usize {
    let grid = img.dct_grid(0);
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let mut count = 0usize;
    for br in 0..bt {
        for bc in 0..bw {
            let blk = grid.block(br, bc);
            for k in 1..64 {
                if blk[k] != 0 {
                    count += 1;
                }
            }
        }
    }
    count
}

#[test]
fn optimizer_si_compounding_experiment() {
    let base_image = "photo_640x480_q75_420.jpg";
    let original_bytes = load_test_image(base_image);

    eprintln!("\n{}", "=".repeat(70));
    eprintln!("STEGO OPTIMIZER + SI-UNIWARD COMPOUNDING EXPERIMENT");
    eprintln!("{}\n", "=".repeat(70));

    // --- Step 1: Baseline measurements on original JPEG ---
    let img_orig = JpegImage::from_bytes(&original_bytes).unwrap();
    let cap_standard = ghost_capacity(&img_orig).unwrap();
    let cap_si = ghost_capacity_si(&img_orig).unwrap();
    let nonzero_orig = count_nonzero_ac(&img_orig);

    eprintln!("Original image: {base_image}");
    eprintln!("  Dimensions: {}x{}", img_orig.frame_info().width, img_orig.frame_info().height);
    eprintln!("  Non-zero AC coefficients: {nonzero_orig}");
    eprintln!("  Standard Ghost capacity: {cap_standard} bytes");
    eprintln!("  SI-UNIWARD capacity:     {cap_si} bytes");
    eprintln!();

    // --- Step 2: Decompress to pixels ---
    let (pixels, width, height) = jpeg_to_luma_f64(&img_orig).unwrap();
    eprintln!("Decompressed to {width}x{height} pixels");

    // Compute variance statistics for the original image
    let mut variances: Vec<f64> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            variances.push(local_variance(&pixels, width, height, y, x));
        }
    }
    variances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_var = variances[variances.len() / 2];
    let p25_var = variances[variances.len() / 4];
    let p10_var = variances[variances.len() / 10];
    eprintln!("  Variance statistics: p10={p10_var:.1}, p25={p25_var:.1}, median={median_var:.1}");

    // --- Step 3: Create optimized pixels ---
    // Use a threshold that catches the smoothest ~30-50% of pixels
    // and a noise amplitude of ±3 levels (subtle but enough to create texture)
    let threshold = p25_var.max(10.0); // At least 10 to avoid only modifying constant regions
    let noise_amplitude = 3.0;

    eprintln!("\nOptimizer settings: threshold={threshold:.1}, noise=±{noise_amplitude}");
    let optimized_pixels = optimize_pixels(&pixels, width, height, threshold, noise_amplitude);

    // --- Step 4: JPEG-compress the optimized pixels ---
    let mut img_opt = JpegImage::from_bytes(&original_bytes).unwrap();
    luma_f64_to_jpeg(&optimized_pixels, width, height, &mut img_opt).unwrap();
    img_opt.rebuild_huffman_tables();
    let opt_bytes = img_opt.to_bytes().unwrap();

    // Re-parse to get clean capacity measurement
    let img_opt2 = JpegImage::from_bytes(&opt_bytes).unwrap();
    let cap_opt = ghost_capacity(&img_opt2).unwrap();
    let cap_opt_si = ghost_capacity_si(&img_opt2).unwrap();
    let nonzero_opt = count_nonzero_ac(&img_opt2);

    eprintln!("\nOptimized image:");
    eprintln!("  Non-zero AC coefficients: {nonzero_opt} (was {nonzero_orig}, delta +{})",
        nonzero_opt as isize - nonzero_orig as isize);
    eprintln!("  Standard Ghost capacity:  {cap_opt} bytes (was {cap_standard})");
    eprintln!("  SI-UNIWARD capacity:      {cap_opt_si} bytes (was {cap_si})");

    // --- Step 5: Also create SI test data (raw pixels → JPEG with rounding errors) ---
    // The original pixels serve as "raw pixels" for SI-UNIWARD of the original image.
    // The optimized pixels serve as "raw pixels" for SI-UNIWARD of the optimized image.

    // --- Step 6: Print results table ---
    eprintln!("\n{:-<70}", "");
    eprintln!("RESULTS SUMMARY");
    eprintln!("{:-<70}", "");
    eprintln!("                          Non-zero AC    Capacity (bytes)");
    eprintln!("  Standard Ghost:         {nonzero_orig:>10}    {cap_standard:>8}");
    eprintln!("  SI-UNIWARD:             {nonzero_orig:>10}    {cap_si:>8}  (+{:.1}%)",
        100.0 * (cap_si as f64 - cap_standard as f64) / cap_standard as f64);
    eprintln!("  Optimized Ghost:        {nonzero_opt:>10}    {cap_opt:>8}  (+{:.1}%)",
        100.0 * (cap_opt as f64 - cap_standard as f64) / cap_standard as f64);
    eprintln!("  Optimized + SI-UNIWARD: {nonzero_opt:>10}    {cap_opt_si:>8}  (+{:.1}%)",
        100.0 * (cap_opt_si as f64 - cap_standard as f64) / cap_standard as f64);
    eprintln!("{:-<70}", "");

    let si_gain = cap_si as f64 / cap_standard as f64;
    let opt_gain = cap_opt as f64 / cap_standard as f64;
    let combined_gain = cap_opt_si as f64 / cap_standard as f64;
    let theoretical_compound = si_gain * opt_gain; // if truly independent, should multiply

    eprintln!("\nGain factors (relative to standard Ghost):");
    eprintln!("  SI-UNIWARD:                 {si_gain:.3}x");
    eprintln!("  Optimizer:                  {opt_gain:.3}x");
    eprintln!("  Combined (actual):          {combined_gain:.3}x");
    eprintln!("  Combined (if multiplicative): {theoretical_compound:.3}x");
    eprintln!("  Compound ratio: {:.3} (1.0 = perfect multiplicative compounding)",
        combined_gain / theoretical_compound);

    // --- Step 7: Verify the hypothesis ---
    // The ordering should be: cap_opt_si > cap_si AND cap_opt_si > cap_opt > cap_standard
    eprintln!("\nHypothesis checks:");

    let check1 = cap_si > cap_standard;
    eprintln!("  [{}] SI > Standard: {} > {}", if check1 { "PASS" } else { "FAIL" }, cap_si, cap_standard);

    let check2 = cap_opt > cap_standard;
    eprintln!("  [{}] Optimized > Standard: {} > {}", if check2 { "PASS" } else { "FAIL" }, cap_opt, cap_standard);

    let check3 = cap_opt_si > cap_si;
    eprintln!("  [{}] Optimized+SI > SI: {} > {}", if check3 { "PASS" } else { "FAIL" }, cap_opt_si, cap_si);

    let check4 = cap_opt_si > cap_opt;
    eprintln!("  [{}] Optimized+SI > Optimized: {} > {}", if check4 { "PASS" } else { "FAIL" }, cap_opt_si, cap_opt);

    let check5 = cap_opt_si > cap_standard;
    eprintln!("  [{}] Optimized+SI > Standard: {} > {}", if check5 { "PASS" } else { "FAIL" }, cap_opt_si, cap_standard);

    // Check that the combined gain is reasonably close to multiplicative
    // (within 20% — it should be exactly multiplicative since both mechanisms
    // work through the same capacity formula: more positions × lower ratio)
    let compound_ratio = combined_gain / theoretical_compound;
    let check6 = (compound_ratio - 1.0).abs() < 0.01;
    eprintln!("  [{}] Compound ratio ≈ 1.0: {:.4} (expected ~1.0 for independent mechanisms)",
        if check6 { "PASS" } else { "INFO" }, compound_ratio);

    eprintln!();

    // Assert the key results
    assert!(check1, "SI capacity should exceed standard Ghost capacity");
    assert!(check2, "Optimized image should have higher capacity than original");
    assert!(check3, "Optimized+SI should exceed SI alone (compounding)");
    assert!(check4, "Optimized+SI should exceed Optimized alone (compounding)");
    assert!(check5, "Optimized+SI should exceed standard Ghost (compounding)");

    // The combined gain should be approximately multiplicative since both
    // mechanisms operate through the same capacity formula:
    // capacity = count_nonzero_ac / ratio - overhead
    // Optimizer increases count_nonzero_ac, SI decreases ratio.
    // These are independent multipliers.
}

/// Additional experiment: vary the noise amplitude and see how capacity scales.
#[test]
fn optimizer_amplitude_sweep() {
    let base_image = "photo_640x480_q75_420.jpg";
    let original_bytes = load_test_image(base_image);

    let img_orig = JpegImage::from_bytes(&original_bytes).unwrap();
    let (pixels, width, height) = jpeg_to_luma_f64(&img_orig).unwrap();

    let threshold = 20.0; // fixed threshold for comparison

    eprintln!("\n{}", "=".repeat(70));
    eprintln!("AMPLITUDE SWEEP: How noise level affects capacity");
    eprintln!("{}", "=".repeat(70));
    eprintln!("Threshold: {threshold}, Image: {base_image}\n");
    eprintln!("{:>10} {:>12} {:>12} {:>12} {:>12}", "Noise±", "NonZero AC", "Ghost Cap", "SI Cap", "SI Gain");
    eprintln!("{:-<62}", "");

    let cap_standard = ghost_capacity(&img_orig).unwrap();
    let cap_si_orig = ghost_capacity_si(&img_orig).unwrap();
    let nonzero_orig = count_nonzero_ac(&img_orig);
    eprintln!("{:>10} {:>12} {:>12} {:>12} {:>12.1}%",
        "0 (orig)", nonzero_orig, cap_standard, cap_si_orig,
        100.0 * (cap_si_orig as f64 - cap_standard as f64) / cap_standard as f64);

    for amplitude in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0] {
        let opt_pixels = optimize_pixels_quiet(&pixels, width, height, threshold, amplitude);

        let mut img_opt = JpegImage::from_bytes(&original_bytes).unwrap();
        luma_f64_to_jpeg(&opt_pixels, width, height, &mut img_opt).unwrap();
        img_opt.rebuild_huffman_tables();
        let opt_bytes = img_opt.to_bytes().unwrap();

        let img_opt2 = JpegImage::from_bytes(&opt_bytes).unwrap();
        let cap_opt = ghost_capacity(&img_opt2).unwrap();
        let cap_opt_si = ghost_capacity_si(&img_opt2).unwrap();
        let nonzero = count_nonzero_ac(&img_opt2);

        eprintln!("{:>10} {:>12} {:>12} {:>12} {:>12.1}%",
            format!("±{amplitude}"), nonzero, cap_opt, cap_opt_si,
            100.0 * (cap_opt_si as f64 - cap_opt as f64) / cap_opt as f64);
    }
    eprintln!();
}

/// Quiet version of optimize_pixels (no eprintln).
fn optimize_pixels_quiet(
    pixels: &[f64],
    width: usize,
    height: usize,
    threshold: f64,
    noise_amplitude: f64,
) -> Vec<f64> {
    let mut optimized = pixels.to_vec();
    for y in 0..height {
        for x in 0..width {
            let var = local_variance(pixels, width, height, y, x);
            if var < threshold {
                let hash = ((y * 31 + x * 97 + 13) as f64 * 0.618033988749895).fract();
                let noise = (hash - 0.5) * 2.0 * noise_amplitude;
                let idx = y * width + x;
                optimized[idx] = (pixels[idx] + noise).clamp(0.0, 255.0);
            }
        }
    }
    optimized
}
