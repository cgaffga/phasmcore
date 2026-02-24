//! WhatsApp survival research experiments.
//!
//! Scientific experiments testing various strategies for surviving WhatsApp's
//! image processing pipeline (recompression at QF ~75-80 with non-IJG quantization tables).
//!
//! Run all experiments:
//!   cargo test -p phasm-core --release -- --ignored whatsapp_ --nocapture

use phasm_core::{JpegImage, QuantTable};
use phasm_core::jpeg::pixels::{idct_block, dct_block};

// --- Standard quantization tables (ITU-T T.81, Annex K) ---

const STD_LUMA_QT: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
];

/// Zigzag scan order: position i in zigzag order maps to this natural index.
const ZIGZAG_TO_NATURAL: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

const NATURAL_TO_ZIGZAG: [usize; 64] = {
    let mut table = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        table[ZIGZAG_TO_NATURAL[i]] = i;
        i += 1;
    }
    table
};

fn scale_quant_table(std_qt: &[u16; 64], qf: u8) -> [u16; 64] {
    let scale = if qf < 50 {
        5000u32 / qf as u32
    } else {
        200u32 - 2 * qf as u32
    };
    let mut qt = [0u16; 64];
    for i in 0..64 {
        let val = (std_qt[i] as u32 * scale + 50) / 100;
        qt[i] = val.clamp(1, 255) as u16;
    }
    qt
}

fn load_test_vector(name: &str) -> Vec<u8> {
    std::fs::read(format!("../test-vectors/{name}")).unwrap()
}

fn load_real_photo(name: &str) -> Vec<u8> {
    std::fs::read(format!("tests/real_photos/{name}")).unwrap()
}

// =============================================================================
// EXPERIMENT 1: DC Coefficient Stability Under Recompression
// =============================================================================

/// Measures how stable DC coefficients are under pixel-domain recompression.
/// DC coefficients represent the average brightness of each 8x8 block.
///
/// Hypothesis: DC coefficients are MORE stable than AC because they represent
/// block averages, which are less affected by quantization changes.
///
/// Run: cargo test -p phasm-core --release -- --ignored whatsapp_dc_stability --nocapture
#[test]
#[ignore]
fn whatsapp_dc_stability() {
    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let target_qfs: &[u8] = &[95, 85, 80, 75, 70, 53];

    println!();
    println!("EXPERIMENT 1: DC COEFFICIENT STABILITY");
    println!("======================================");
    println!("Measures bit flip rate of DC and AC coefficients under recompression.");
    println!();
    println!("{:<16} {:>4} {:>10} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "Image", "QF", "DC_total", "DC_flipped", "DC_BER%", "AC_BER%", "Sign_AC_BER%", "Low_AC_BER%");
    println!("{}", "-".repeat(105));

    for (img_name, cover_bytes) in &test_images {
        let img = match JpegImage::from_bytes(cover_bytes) {
            Ok(img) => img,
            Err(e) => { println!("{}: SKIP ({})", img_name, e); continue; }
        };

        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let orig_qt = img.quant_table(qt_id).unwrap().clone();
        let grid = img.dct_grid(0);
        let bw = grid.blocks_wide();
        let bt = grid.blocks_tall();

        for &target_qf in target_qfs {
            let target_qt_values = scale_quant_table(&STD_LUMA_QT, target_qf);

            let mut dc_total = 0u32;
            let mut dc_flipped = 0u32;
            let mut ac_total = 0u32;
            let mut ac_flipped = 0u32;
            let mut sign_ac_total = 0u32;
            let mut sign_ac_flipped = 0u32;
            let mut low_ac_total = 0u32;  // zigzag 1-15
            let mut low_ac_flipped = 0u32;

            for br in 0..bt {
                for bc in 0..bw {
                    let block_slice = grid.block(br, bc);
                    let quantized: [i16; 64] = block_slice.try_into().unwrap();

                    // Pixel-domain round-trip
                    let mut px = idct_block(&quantized, &orig_qt.values);
                    for p in px.iter_mut() {
                        *p = p.clamp(0.0, 255.0);
                    }
                    let recompressed = dct_block(&px, &target_qt_values);

                    // DC coefficient
                    dc_total += 1;
                    if quantized[0] != recompressed[0] {
                        dc_flipped += 1;
                    }

                    // AC coefficients
                    for freq_idx in 1..64 {
                        let zz = NATURAL_TO_ZIGZAG[freq_idx];
                        let orig = quantized[freq_idx];
                        let recomp = recompressed[freq_idx];

                        ac_total += 1;
                        if orig != recomp {
                            ac_flipped += 1;
                        }

                        // Sign stability (only for non-zero coefficients)
                        if orig != 0 {
                            sign_ac_total += 1;
                            let orig_sign = orig > 0;
                            let recomp_sign = recomp > 0;
                            if orig_sign != recomp_sign {
                                sign_ac_flipped += 1;
                            }
                        }

                        // Low-frequency AC (zigzag 1-15)
                        if zz >= 1 && zz <= 15 {
                            low_ac_total += 1;
                            if orig != recomp {
                                low_ac_flipped += 1;
                            }
                        }
                    }
                }
            }

            let dc_ber = if dc_total > 0 { dc_flipped as f64 / dc_total as f64 * 100.0 } else { 0.0 };
            let ac_ber = if ac_total > 0 { ac_flipped as f64 / ac_total as f64 * 100.0 } else { 0.0 };
            let sign_ber = if sign_ac_total > 0 { sign_ac_flipped as f64 / sign_ac_total as f64 * 100.0 } else { 0.0 };
            let low_ber = if low_ac_total > 0 { low_ac_flipped as f64 / low_ac_total as f64 * 100.0 } else { 0.0 };

            println!("{:<16} {:>4} {:>10} {:>10} {:>9.2}% {:>9.2}% {:>11.2}% {:>11.2}%",
                img_name, target_qf, dc_total, dc_flipped, dc_ber, ac_ber, sign_ber, low_ber);
        }
        println!();
    }
}

// =============================================================================
// EXPERIMENT 2: Block Average Brightness Stability
// =============================================================================

/// Measures how stable the average brightness of 8x8 blocks is under recompression.
///
/// Hypothesis: Block averages (derived from DC) are extremely stable because
/// the average of 64 pixels is almost unaffected by quantization of individual
/// frequency components.
///
/// Run: cargo test -p phasm-core --release -- --ignored whatsapp_block_avg_stability --nocapture
#[test]
#[ignore]
fn whatsapp_block_avg_stability() {
    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let target_qfs: &[u8] = &[95, 85, 80, 75, 70, 53];

    println!();
    println!("EXPERIMENT 2: BLOCK AVERAGE BRIGHTNESS STABILITY");
    println!("=================================================");
    println!("Measures average brightness change per 8x8 block under recompression.");
    println!();
    println!("{:<16} {:>4} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "Image", "QF", "Blocks", "MeanAbsDiff", "MaxAbsDiff", "StdDev", "BitsOK@4");
    println!("{}", "-".repeat(88));

    for (img_name, cover_bytes) in &test_images {
        let img = match JpegImage::from_bytes(cover_bytes) {
            Ok(img) => img,
            Err(e) => { println!("{}: SKIP ({})", img_name, e); continue; }
        };

        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let orig_qt = img.quant_table(qt_id).unwrap().clone();
        let grid = img.dct_grid(0);
        let bw = grid.blocks_wide();
        let bt = grid.blocks_tall();

        for &target_qf in target_qfs {
            let target_qt_values = scale_quant_table(&STD_LUMA_QT, target_qf);

            let mut diffs: Vec<f64> = Vec::new();

            for br in 0..bt {
                for bc in 0..bw {
                    let block_slice = grid.block(br, bc);
                    let quantized: [i16; 64] = block_slice.try_into().unwrap();

                    // Compute block average from original
                    let orig_pixels = idct_block(&quantized, &orig_qt.values);
                    let orig_avg: f64 = orig_pixels.iter().sum::<f64>() / 64.0;

                    // Pixel-domain round-trip
                    let mut px = orig_pixels;
                    for p in px.iter_mut() {
                        *p = p.clamp(0.0, 255.0);
                    }
                    let recomp_coeff = dct_block(&px, &target_qt_values);
                    let recomp_pixels = idct_block(&recomp_coeff, &target_qt_values);
                    let recomp_avg: f64 = recomp_pixels.iter().sum::<f64>() / 64.0;

                    diffs.push((orig_avg - recomp_avg).abs());
                }
            }

            let n = diffs.len();
            let mean_diff = diffs.iter().sum::<f64>() / n as f64;
            let max_diff = diffs.iter().cloned().fold(0.0f64, f64::max);
            let variance = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / n as f64;
            let std_dev = variance.sqrt();

            // How many blocks have diff < 4.0 (meaning a QIM step of 8 would survive)?
            let bits_ok_at_4 = diffs.iter().filter(|&&d| d < 4.0).count();
            let bits_ok_pct = bits_ok_at_4 as f64 / n as f64 * 100.0;

            println!("{:<16} {:>4} {:>8} {:>11.3} {:>11.3} {:>11.3} {:>8} ({:.0}%)",
                img_name, target_qf, n, mean_diff, max_diff, std_dev,
                bits_ok_at_4, bits_ok_pct);
        }
        println!();
    }
}

// =============================================================================
// EXPERIMENT 3: STDM with Extreme Deltas
// =============================================================================

/// Tests STDM embedding survival at very large delta values (10x-50x mean_qt).
///
/// Hypothesis: With delta = 20x mean_qt (~200), the decision boundary margin is
/// so large that even cross-encoder recompression cannot flip bits.
///
/// Run: cargo test -p phasm-core --release -- --ignored whatsapp_extreme_delta --nocapture
#[test]
#[ignore]
fn whatsapp_extreme_delta() {
    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let target_qfs: &[u8] = &[85, 80, 75, 70];
    let delta_multipliers: &[f64] = &[3.0, 8.0, 15.0, 20.0, 30.0, 50.0];

    println!();
    println!("EXPERIMENT 3: STDM WITH EXTREME DELTAS");
    println!("========================================");
    println!("Measures bit error rate of STDM embedding at various delta multipliers.");
    println!("Delta = multiplier * mean_qt. Larger delta = more robust but more artifacts.");
    println!();
    println!("{:<16} {:>4} {:>6} {:>10} {:>8} {:>10} {:>10} {:>12}",
        "Image", "QF", "DeltaX", "Delta_val", "Units", "Errors", "BER%", "PSNR_est_dB");
    println!("{}", "-".repeat(88));

    for (img_name, cover_bytes) in &test_images {
        let img = match JpegImage::from_bytes(cover_bytes) {
            Ok(img) => img,
            Err(e) => { println!("{}: SKIP ({})", img_name, e); continue; }
        };

        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let orig_qt = img.quant_table(qt_id).unwrap().clone();
        let grid = img.dct_grid(0);
        let bw = grid.blocks_wide();
        let bt = grid.blocks_tall();

        // Compute mean_qt (zigzag 1-15)
        let mut sum_qt = 0.0f64;
        let mut count_qt = 0usize;
        for nat_idx in 0..64 {
            let zz = NATURAL_TO_ZIGZAG[nat_idx];
            if zz >= 1 && zz <= 15 {
                sum_qt += orig_qt.values[nat_idx] as f64;
                count_qt += 1;
            }
        }
        let mean_qt = sum_qt / count_qt as f64;

        // Collect all eligible coefficient positions (zigzag 1-15)
        struct CoeffPos { br: usize, bc: usize, i: usize, j: usize }
        let mut positions: Vec<CoeffPos> = Vec::new();
        for br in 0..bt {
            for bc in 0..bw {
                for i in 0..8 {
                    for j in 0..8 {
                        let freq_idx = i * 8 + j;
                        let zz = NATURAL_TO_ZIGZAG[freq_idx];
                        if zz >= 1 && zz <= 15 {
                            positions.push(CoeffPos { br, bc, i, j });
                        }
                    }
                }
            }
        }

        // For each (delta_mult, target_qf), simulate STDM embed + pixel-domain recomp + extract
        for &delta_mult in delta_multipliers {
            let delta = delta_mult * mean_qt;

            for &target_qf in target_qfs {
                let target_qt_values = scale_quant_table(&STD_LUMA_QT, target_qf);

                // Create a mutable copy for embedding
                let mut img_copy = img.clone();

                // Simple embedding: embed alternating 0/1 bits using a simple
                // spreading approach (single coefficient per bit for simplicity)
                let num_bits = positions.len();
                let mut embedded_bits = vec![0u8; num_bits];

                // Embed using STDM-like quantization on individual coefficients
                // (simplified: no spreading, just quantize each coefficient)
                for (idx, pos) in positions.iter().enumerate() {
                    let bit = (idx % 2) as u8;
                    embedded_bits[idx] = bit;

                    let coeff = img_copy.dct_grid(0).get(pos.br, pos.bc, pos.i, pos.j) as f64;
                    let quantized = if bit == 0 {
                        (coeff / delta).round() * delta
                    } else {
                        ((coeff / delta - 0.5).round() + 0.5) * delta
                    };
                    img_copy.dct_grid_mut(0).set(pos.br, pos.bc, pos.i, pos.j, quantized.round() as i16);
                }

                // Pre-clamp pass (like armor does)
                for br in 0..bt {
                    for bc in 0..bw {
                        let block_slice = img_copy.dct_grid(0).block(br, bc);
                        let quantized: [i16; 64] = block_slice.try_into().unwrap();
                        let mut px = idct_block(&quantized, &orig_qt.values);
                        for p in px.iter_mut() { *p = p.clamp(0.0, 255.0); }
                        let settled = dct_block(&px, &orig_qt.values);
                        img_copy.dct_grid_mut(0).block_mut(br, bc).copy_from_slice(&settled);
                    }
                }

                // Now simulate pixel-domain recompression at target QF
                for br in 0..bt {
                    for bc in 0..bw {
                        let block_slice = img_copy.dct_grid(0).block(br, bc);
                        let quantized: [i16; 64] = block_slice.try_into().unwrap();
                        let mut px = idct_block(&quantized, &orig_qt.values);
                        for p in px.iter_mut() { *p = p.clamp(0.0, 255.0); }
                        let recompressed = dct_block(&px, &target_qt_values);
                        img_copy.dct_grid_mut(0).block_mut(br, bc).copy_from_slice(&recompressed);
                    }
                }

                // Extract bits
                let mut errors = 0u32;
                for (idx, pos) in positions.iter().enumerate() {
                    let coeff = img_copy.dct_grid(0).get(pos.br, pos.bc, pos.i, pos.j) as f64;
                    let half_delta = delta / 2.0;
                    let m = (coeff / half_delta).round() as i64;
                    let extracted = m.rem_euclid(2) as u8;
                    if extracted != embedded_bits[idx] {
                        errors += 1;
                    }
                }

                let ber = errors as f64 / num_bits as f64 * 100.0;

                // Rough PSNR estimate: delta change per coefficient ~ delta / 2
                // Mean squared error per pixel ~ (delta/2)^2 / 64 (distributed across block)
                let avg_change = delta / 2.0;
                let mse_per_pixel = (avg_change * avg_change) / 64.0;
                let psnr_est = if mse_per_pixel > 0.0 {
                    10.0 * (255.0 * 255.0 / mse_per_pixel).log10()
                } else {
                    99.0
                };

                println!("{:<16} {:>4} {:>5.0}x {:>9.1} {:>8} {:>10} {:>9.2}% {:>11.1}",
                    img_name, target_qf, delta_mult, delta, num_bits, errors, ber, psnr_est);
            }
        }
        println!();
    }
}

// =============================================================================
// EXPERIMENT 4: Real Encoder Recompression BER
// =============================================================================

/// Tests STDM bit survival using real external encoders (sips/libjpeg-turbo/MozJPEG).
///
/// This is the most realistic test: embed bits, write to JPEG, recompress with
/// a real encoder, read back, extract bits.
///
/// Run: cargo test -p phasm-core --release -- --ignored whatsapp_real_encoder_ber --nocapture
#[test]
#[ignore]
fn whatsapp_real_encoder_ber() {
    use std::process::Command;

    let cover_bytes = load_real_photo("istockphoto-612x612-baseline.jpg");
    let img = JpegImage::from_bytes(&cover_bytes).unwrap();

    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let orig_qt = img.quant_table(qt_id).unwrap().clone();
    let grid = img.dct_grid(0);
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();

    // Compute mean_qt
    let mut sum_qt = 0.0f64;
    let mut count_qt = 0usize;
    for nat_idx in 0..64 {
        let zz = NATURAL_TO_ZIGZAG[nat_idx];
        if zz >= 1 && zz <= 15 {
            sum_qt += orig_qt.values[nat_idx] as f64;
            count_qt += 1;
        }
    }
    let mean_qt = sum_qt / count_qt as f64;

    // Collect eligible positions
    struct CoeffPos { br: usize, bc: usize, i: usize, j: usize, freq_idx: usize }
    let mut positions: Vec<CoeffPos> = Vec::new();
    for br in 0..bt {
        for bc in 0..bw {
            for i in 0..8 {
                for j in 0..8 {
                    let freq_idx = i * 8 + j;
                    let zz = NATURAL_TO_ZIGZAG[freq_idx];
                    if zz >= 1 && zz <= 15 {
                        positions.push(CoeffPos { br, bc, i, j, freq_idx });
                    }
                }
            }
        }
    }

    let delta_multipliers: &[f64] = &[8.0, 15.0, 20.0, 30.0];
    let qfs: &[u8] = &[85, 80, 75, 70];

    println!();
    println!("EXPERIMENT 4: REAL ENCODER BER (istock_612x408)");
    println!("================================================");
    println!("Image: istock_612x408, mean_qt = {:.2}", mean_qt);
    println!("Embeds individual QIM bits, writes JPEG, recompresses with real encoders, extracts.");
    println!();

    struct EncoderInfo {
        name: &'static str,
        recompress: Box<dyn Fn(&[u8], u8) -> Option<Vec<u8>>>,
    }

    let encoders: Vec<EncoderInfo> = vec![
        EncoderInfo {
            name: "sips (AppleJPEG)",
            recompress: Box::new(|bytes: &[u8], qf: u8| -> Option<Vec<u8>> {
                let dir = std::env::temp_dir();
                let pid = std::process::id();
                let input = dir.join(format!("phasm_exp4_in_{pid}.jpg"));
                let output = dir.join(format!("phasm_exp4_out_{pid}.jpg"));
                std::fs::write(&input, bytes).ok()?;
                let out = Command::new("/usr/bin/sips")
                    .arg("-s").arg("format").arg("jpeg")
                    .arg("-s").arg("formatOptions").arg(qf.to_string())
                    .arg(&input)
                    .arg("--out").arg(&output)
                    .output().ok()?;
                let _ = std::fs::remove_file(&input);
                if !out.status.success() { let _ = std::fs::remove_file(&output); return None; }
                let result = std::fs::read(&output).ok();
                let _ = std::fs::remove_file(&output);
                result
            }),
        },
        EncoderInfo {
            name: "libjpeg-turbo",
            recompress: Box::new(|bytes: &[u8], qf: u8| -> Option<Vec<u8>> {
                let dir = std::env::temp_dir();
                let pid = std::process::id();
                let input = dir.join(format!("phasm_exp4_ljt_in_{pid}.jpg"));
                let ppm = dir.join(format!("phasm_exp4_ljt_{pid}.ppm"));
                let output = dir.join(format!("phasm_exp4_ljt_out_{pid}.jpg"));
                std::fs::write(&input, bytes).ok()?;
                let djpeg = Command::new("/opt/homebrew/bin/djpeg")
                    .arg("-ppm").arg("-outfile").arg(&ppm).arg(&input)
                    .output().ok()?;
                let _ = std::fs::remove_file(&input);
                if !djpeg.status.success() { return None; }
                let cjpeg = Command::new("/opt/homebrew/bin/cjpeg")
                    .arg("-quality").arg(qf.to_string())
                    .arg("-baseline")
                    .arg("-outfile").arg(&output).arg(&ppm)
                    .output().ok()?;
                let _ = std::fs::remove_file(&ppm);
                if !cjpeg.status.success() { let _ = std::fs::remove_file(&output); return None; }
                let result = std::fs::read(&output).ok();
                let _ = std::fs::remove_file(&output);
                result
            }),
        },
        EncoderInfo {
            name: "MozJPEG",
            recompress: Box::new(|bytes: &[u8], qf: u8| -> Option<Vec<u8>> {
                let dir = std::env::temp_dir();
                let pid = std::process::id();
                let input = dir.join(format!("phasm_exp4_moz_in_{pid}.jpg"));
                let ppm = dir.join(format!("phasm_exp4_moz_{pid}.ppm"));
                let output = dir.join(format!("phasm_exp4_moz_out_{pid}.jpg"));
                std::fs::write(&input, bytes).ok()?;
                let djpeg = Command::new("/opt/homebrew/bin/djpeg")
                    .arg("-ppm").arg("-outfile").arg(&ppm).arg(&input)
                    .output().ok()?;
                let _ = std::fs::remove_file(&input);
                if !djpeg.status.success() { return None; }
                let cjpeg = Command::new("/opt/homebrew/opt/mozjpeg/bin/cjpeg")
                    .arg("-quality").arg(qf.to_string())
                    .arg("-baseline")
                    .arg("-outfile").arg(&output).arg(&ppm)
                    .output().ok()?;
                let _ = std::fs::remove_file(&ppm);
                if !cjpeg.status.success() { let _ = std::fs::remove_file(&output); return None; }
                let result = std::fs::read(&output).ok();
                let _ = std::fs::remove_file(&output);
                result
            }),
        },
    ];

    println!("{:<18} {:>6} {:>4} {:>8} {:>8} {:>10} {:>12}",
        "Encoder", "DeltaX", "QF", "Bits", "Errors", "BER%", "SurvivesR27?");
    println!("{}", "-".repeat(78));

    for encoder in &encoders {
        for &delta_mult in delta_multipliers {
            let delta = delta_mult * mean_qt;

            // Embed into a fresh copy
            let mut img_embed = img.clone();
            let mut embedded_bits = vec![0u8; positions.len()];

            for (idx, pos) in positions.iter().enumerate() {
                let bit = (idx % 2) as u8;
                embedded_bits[idx] = bit;
                let coeff = img_embed.dct_grid(0).get(pos.br, pos.bc, pos.i, pos.j) as f64;
                let quantized = if bit == 0 {
                    (coeff / delta).round() * delta
                } else {
                    ((coeff / delta - 0.5).round() + 0.5) * delta
                };
                img_embed.dct_grid_mut(0).set(pos.br, pos.bc, pos.i, pos.j, quantized.round() as i16);
            }

            // Pre-clamp
            for br in 0..bt {
                for bc in 0..bw {
                    let block_slice = img_embed.dct_grid(0).block(br, bc);
                    let quantized: [i16; 64] = block_slice.try_into().unwrap();
                    let mut px = idct_block(&quantized, &orig_qt.values);
                    for p in px.iter_mut() { *p = p.clamp(0.0, 255.0); }
                    let settled = dct_block(&px, &orig_qt.values);
                    img_embed.dct_grid_mut(0).block_mut(br, bc).copy_from_slice(&settled);
                }
            }

            // Write to JPEG
            img_embed.rebuild_huffman_tables();
            let stego_bytes = img_embed.to_bytes().unwrap();

            for &qf in qfs {
                let recompressed = match (encoder.recompress)(&stego_bytes, qf) {
                    Some(r) => r,
                    None => {
                        println!("{:<18} {:>5.0}x {:>4} {:>8} {:>8} {:>10} {:>12}",
                            encoder.name, delta_mult, qf, "-", "-", "ENC_ERR", "-");
                        continue;
                    }
                };

                // Parse recompressed image and extract bits
                let recomp_img = match JpegImage::from_bytes(&recompressed) {
                    Ok(img) => img,
                    Err(_) => {
                        println!("{:<18} {:>5.0}x {:>4} {:>8} {:>8} {:>10} {:>12}",
                            encoder.name, delta_mult, qf, "-", "-", "PARSE_ERR", "-");
                        continue;
                    }
                };

                let recomp_grid = recomp_img.dct_grid(0);
                let recomp_qt_id = recomp_img.frame_info().components[0].quant_table_id as usize;
                let recomp_qt = recomp_img.quant_table(recomp_qt_id).unwrap();

                // The recompressed image may have different QT, so we need to
                // dequantize with the recomp QT to get the actual DCT values,
                // then apply our QIM extraction
                let mut errors = 0u32;
                let mut total = 0u32;
                for (idx, pos) in positions.iter().enumerate() {
                    // Get the recompressed quantized coefficient
                    let recomp_coeff_quantized = recomp_grid.get(pos.br, pos.bc, pos.i, pos.j);
                    // Dequantize to actual DCT value
                    let dct_value = recomp_coeff_quantized as f64 * recomp_qt.values[pos.freq_idx] as f64;
                    // Re-quantize with original QT (since delta was computed from original)
                    let coeff_in_orig_units = dct_value / orig_qt.values[pos.freq_idx] as f64;

                    let half_delta = delta / 2.0;
                    let m = (coeff_in_orig_units / half_delta).round() as i64;
                    let extracted = m.rem_euclid(2) as u8;

                    total += 1;
                    if extracted != embedded_bits[idx] {
                        errors += 1;
                    }
                }

                let ber = errors as f64 / total as f64 * 100.0;

                // Would r=27 repetition + RS survive?
                // At BER p, majority voting with r=27 gives post-voting BER ~
                // If p < 0.5, post-vote BER ≈ binomial(27, floor(27/2), p)
                // Rough: if BER < 20%, r=27 should correct to near-zero
                let survives = if ber < 15.0 { "LIKELY YES" } else if ber < 25.0 { "MAYBE" } else { "NO" };

                println!("{:<18} {:>5.0}x {:>4} {:>8} {:>8} {:>9.2}% {:>12}",
                    encoder.name, delta_mult, qf, total, errors, ber, survives);
            }
        }
        println!();
    }
}

// =============================================================================
// EXPERIMENT 5: Pre-Conditioned Recompression
// =============================================================================

/// Tests the "errorless" approach: pre-compress with target QT, then embed,
/// then recompress with the same/similar QT.
///
/// Hypothesis: If we pre-compress with QF 80 (IJG tables), then recompression
/// at QF 80 is nearly idempotent. Even with different encoder QTs, the
/// coefficients are already "settled" near the lattice.
///
/// Run: cargo test -p phasm-core --release -- --ignored whatsapp_precondition --nocapture
#[test]
#[ignore]
fn whatsapp_precondition() {
    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    // Pre-condition QFs and target recompression QFs
    let precond_qfs: &[u8] = &[70, 75, 80, 85];
    let target_qfs: &[u8] = &[85, 80, 75, 70];

    println!();
    println!("EXPERIMENT 5: PRE-CONDITIONED RECOMPRESSION");
    println!("============================================");
    println!("Pre-compress at one QF, then measure coefficient change at another QF.");
    println!("Measures: what fraction of coefficients survive unchanged?");
    println!();
    println!("{:<16} {:>6} {:>6} {:>10} {:>10} {:>10} {:>12}",
        "Image", "PreQF", "TargQF", "LowAC_Tot", "Changed", "ChgRate%", "Same2x_Rate%");
    println!("{}", "-".repeat(80));

    for (img_name, cover_bytes) in &test_images {
        let img = match JpegImage::from_bytes(cover_bytes) {
            Ok(img) => img,
            Err(e) => { println!("{}: SKIP ({})", img_name, e); continue; }
        };

        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let orig_qt = img.quant_table(qt_id).unwrap().clone();
        let grid = img.dct_grid(0);
        let bw = grid.blocks_wide();
        let bt = grid.blocks_tall();

        for &precond_qf in precond_qfs {
            let precond_qt = scale_quant_table(&STD_LUMA_QT, precond_qf);

            // Pre-condition: IDCT with orig QT -> clamp -> DCT with precond QT
            let mut precond_coeffs: Vec<Vec<[i16; 64]>> = Vec::new();
            for br in 0..bt {
                let mut row = Vec::new();
                for bc in 0..bw {
                    let block_slice = grid.block(br, bc);
                    let quantized: [i16; 64] = block_slice.try_into().unwrap();
                    let mut px = idct_block(&quantized, &orig_qt.values);
                    for p in px.iter_mut() { *p = p.clamp(0.0, 255.0); }
                    let precond = dct_block(&px, &precond_qt);
                    row.push(precond);
                }
                precond_coeffs.push(row);
            }

            for &target_qf in target_qfs {
                let target_qt = scale_quant_table(&STD_LUMA_QT, target_qf);

                let mut total_low_ac = 0u32;
                let mut changed_count = 0u32;

                // Also test: if we recompress TWICE at target, does it stabilize?
                let mut same_after_2x = 0u32;

                for br in 0..bt {
                    for bc in 0..bw {
                        let precond = &precond_coeffs[br][bc];

                        // Recompress once: IDCT with precond QT -> clamp -> DCT with target QT
                        let mut px = idct_block(precond, &precond_qt);
                        for p in px.iter_mut() { *p = p.clamp(0.0, 255.0); }
                        let recomp1 = dct_block(&px, &target_qt);

                        // Recompress twice
                        let mut px2 = idct_block(&recomp1, &target_qt);
                        for p in px2.iter_mut() { *p = p.clamp(0.0, 255.0); }
                        let recomp2 = dct_block(&px2, &target_qt);

                        for freq_idx in 0..64 {
                            let zz = NATURAL_TO_ZIGZAG[freq_idx];
                            if zz < 1 || zz > 15 { continue; }

                            total_low_ac += 1;
                            if precond[freq_idx] != recomp1[freq_idx] {
                                changed_count += 1;
                            }
                            // Is recomp1 == recomp2? (idempotent after 1 recomp)
                            if recomp1[freq_idx] == recomp2[freq_idx] {
                                same_after_2x += 1;
                            }
                        }
                    }
                }

                let change_rate = changed_count as f64 / total_low_ac as f64 * 100.0;
                let idem_rate = same_after_2x as f64 / total_low_ac as f64 * 100.0;

                println!("{:<16} {:>6} {:>6} {:>10} {:>10} {:>9.2}% {:>11.2}%",
                    img_name, precond_qf, target_qf, total_low_ac, changed_count,
                    change_rate, idem_rate);
            }
        }
        println!();
    }
}

// =============================================================================
// EXPERIMENT 6: Relative Block Brightness Embedding
// =============================================================================

/// Tests embedding in RELATIVE brightness between adjacent blocks.
///
/// Hypothesis: While absolute block brightness may shift during recompression,
/// the RELATIVE ordering of adjacent blocks is extremely stable. If block A
/// is brighter than block B before recompression, it will almost certainly
/// still be brighter after.
///
/// Run: cargo test -p phasm-core --release -- --ignored whatsapp_relative_brightness --nocapture
#[test]
#[ignore]
fn whatsapp_relative_brightness() {
    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let target_qfs: &[u8] = &[95, 85, 80, 75, 70, 53];

    println!();
    println!("EXPERIMENT 6: RELATIVE BLOCK BRIGHTNESS STABILITY");
    println!("===================================================");
    println!("Tests whether A>B relationships between adjacent blocks survive recompression.");
    println!();
    println!("{:<16} {:>4} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "Image", "QF", "Pairs", "Flipped", "FlipRate%", "MinGap_OK", "Gap>4_OK%");
    println!("{}", "-".repeat(88));

    for (img_name, cover_bytes) in &test_images {
        let img = match JpegImage::from_bytes(cover_bytes) {
            Ok(img) => img,
            Err(e) => { println!("{}: SKIP ({})", img_name, e); continue; }
        };

        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let orig_qt = img.quant_table(qt_id).unwrap().clone();
        let grid = img.dct_grid(0);
        let bw = grid.blocks_wide();
        let bt = grid.blocks_tall();

        for &target_qf in target_qfs {
            let target_qt = scale_quant_table(&STD_LUMA_QT, target_qf);

            // Compute block DC values before and after recompression
            let mut orig_dc: Vec<Vec<f64>> = Vec::new();
            let mut recomp_dc: Vec<Vec<f64>> = Vec::new();

            for br in 0..bt {
                let mut orig_row = Vec::new();
                let mut recomp_row = Vec::new();
                for bc in 0..bw {
                    let block_slice = grid.block(br, bc);
                    let quantized: [i16; 64] = block_slice.try_into().unwrap();

                    // Original DC value (dequantized)
                    let orig_dc_val = quantized[0] as f64 * orig_qt.values[0] as f64;
                    orig_row.push(orig_dc_val);

                    // Recompressed DC value
                    let mut px = idct_block(&quantized, &orig_qt.values);
                    for p in px.iter_mut() { *p = p.clamp(0.0, 255.0); }
                    let recomp = dct_block(&px, &target_qt);
                    let recomp_dc_val = recomp[0] as f64 * target_qt[0] as f64;
                    recomp_row.push(recomp_dc_val);
                }
                orig_dc.push(orig_row);
                recomp_dc.push(recomp_row);
            }

            // Test horizontal adjacent pairs
            let mut total_pairs = 0u32;
            let mut flipped = 0u32;
            let mut gap_ok = 0u32;      // pairs where |gap| > 4 that survived
            let mut gap_total = 0u32;   // pairs where |gap| > 4

            for br in 0..bt {
                for bc in 0..bw.saturating_sub(1) {
                    total_pairs += 1;

                    let orig_diff = orig_dc[br][bc] - orig_dc[br][bc + 1];
                    let recomp_diff = recomp_dc[br][bc] - recomp_dc[br][bc + 1];

                    // Did the ordering flip?
                    if (orig_diff > 0.0 && recomp_diff < 0.0) || (orig_diff < 0.0 && recomp_diff > 0.0) {
                        flipped += 1;
                    }

                    // Track pairs with significant gap
                    if orig_diff.abs() > 4.0 * orig_qt.values[0] as f64 {
                        gap_total += 1;
                        if (orig_diff > 0.0 && recomp_diff > 0.0) || (orig_diff < 0.0 && recomp_diff < 0.0) || orig_diff.abs() < 1e-10 {
                            gap_ok += 1;
                        }
                    }
                }
            }

            // Also test vertical pairs
            for br in 0..bt.saturating_sub(1) {
                for bc in 0..bw {
                    total_pairs += 1;

                    let orig_diff = orig_dc[br][bc] - orig_dc[br + 1][bc];
                    let recomp_diff = recomp_dc[br][bc] - recomp_dc[br + 1][bc];

                    if (orig_diff > 0.0 && recomp_diff < 0.0) || (orig_diff < 0.0 && recomp_diff > 0.0) {
                        flipped += 1;
                    }

                    if orig_diff.abs() > 4.0 * orig_qt.values[0] as f64 {
                        gap_total += 1;
                        if (orig_diff > 0.0 && recomp_diff > 0.0) || (orig_diff < 0.0 && recomp_diff < 0.0) || orig_diff.abs() < 1e-10 {
                            gap_ok += 1;
                        }
                    }
                }
            }

            let flip_rate = flipped as f64 / total_pairs as f64 * 100.0;
            let gap_ok_rate = if gap_total > 0 { gap_ok as f64 / gap_total as f64 * 100.0 } else { 0.0 };

            println!("{:<16} {:>4} {:>10} {:>10} {:>9.2}% {:>12} {:>11.2}%",
                img_name, target_qf, total_pairs, flipped, flip_rate,
                format!("{}/{}", gap_ok, gap_total), gap_ok_rate);
        }
        println!();
    }
}

// =============================================================================
// EXPERIMENT 7: Large-Region Average Embedding
// =============================================================================

/// Tests embedding in averages of large pixel regions (16x16, 32x32, 64x64).
///
/// Hypothesis: Averages of large regions are extremely stable under recompression
/// because they integrate over many pixels, averaging out quantization noise.
/// A 32x32 region (4 blocks) average should barely change at all.
///
/// Run: cargo test -p phasm-core --release -- --ignored whatsapp_large_region_avg --nocapture
#[test]
#[ignore]
fn whatsapp_large_region_avg() {
    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let target_qfs: &[u8] = &[85, 80, 75, 70];
    let region_sizes: &[usize] = &[8, 16, 32, 64];

    println!();
    println!("EXPERIMENT 7: LARGE REGION AVERAGE STABILITY");
    println!("=============================================");
    println!("Measures average brightness stability of NxN pixel regions.");
    println!();
    println!("{:<16} {:>4} {:>6} {:>8} {:>12} {:>12} {:>12} {:>14}",
        "Image", "QF", "Region", "Count", "MeanAbsDiff", "MaxAbsDiff", "StdDev", "QIM_step8_OK%");
    println!("{}", "-".repeat(98));

    for (img_name, cover_bytes) in &test_images {
        let img = match JpegImage::from_bytes(cover_bytes) {
            Ok(img) => img,
            Err(e) => { println!("{}: SKIP ({})", img_name, e); continue; }
        };

        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let orig_qt = img.quant_table(qt_id).unwrap().clone();
        let grid = img.dct_grid(0);
        let bw = grid.blocks_wide();
        let bt = grid.blocks_tall();
        let w = bw * 8;
        let h = bt * 8;

        // Get all pixel values (original)
        let mut orig_pixels = vec![0.0f64; w * h];
        for br in 0..bt {
            for bc in 0..bw {
                let block_slice = grid.block(br, bc);
                let quantized: [i16; 64] = block_slice.try_into().unwrap();
                let px = idct_block(&quantized, &orig_qt.values);
                for row in 0..8 {
                    for col in 0..8 {
                        orig_pixels[(br * 8 + row) * w + (bc * 8 + col)] = px[row * 8 + col].clamp(0.0, 255.0);
                    }
                }
            }
        }

        for &target_qf in target_qfs {
            let target_qt = scale_quant_table(&STD_LUMA_QT, target_qf);

            // Get recompressed pixel values
            let mut recomp_pixels = vec![0.0f64; w * h];
            for br in 0..bt {
                for bc in 0..bw {
                    let block_slice = grid.block(br, bc);
                    let quantized: [i16; 64] = block_slice.try_into().unwrap();
                    let mut px = idct_block(&quantized, &orig_qt.values);
                    for p in px.iter_mut() { *p = p.clamp(0.0, 255.0); }
                    let recomp = dct_block(&px, &target_qt);
                    let rpx = idct_block(&recomp, &target_qt);
                    for row in 0..8 {
                        for col in 0..8 {
                            recomp_pixels[(br * 8 + row) * w + (bc * 8 + col)] = rpx[row * 8 + col].clamp(0.0, 255.0);
                        }
                    }
                }
            }

            for &region_size in region_sizes {
                if region_size > w.min(h) { continue; }

                let regions_x = w / region_size;
                let regions_y = h / region_size;
                let num_regions = regions_x * regions_y;

                let mut diffs: Vec<f64> = Vec::with_capacity(num_regions);

                for ry in 0..regions_y {
                    for rx in 0..regions_x {
                        let mut orig_sum = 0.0f64;
                        let mut recomp_sum = 0.0f64;
                        let n = (region_size * region_size) as f64;

                        for dy in 0..region_size {
                            for dx in 0..region_size {
                                let y = ry * region_size + dy;
                                let x = rx * region_size + dx;
                                orig_sum += orig_pixels[y * w + x];
                                recomp_sum += recomp_pixels[y * w + x];
                            }
                        }

                        let orig_avg = orig_sum / n;
                        let recomp_avg = recomp_sum / n;
                        diffs.push((orig_avg - recomp_avg).abs());
                    }
                }

                let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
                let max_diff = diffs.iter().cloned().fold(0.0f64, f64::max);
                let variance = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64;
                let std_dev = variance.sqrt();
                let ok_at_8 = diffs.iter().filter(|&&d| d < 4.0).count();
                let ok_pct = ok_at_8 as f64 / diffs.len() as f64 * 100.0;

                println!("{:<16} {:>4} {:>4}x{:<1} {:>8} {:>11.4} {:>11.4} {:>11.4} {:>13.1}%",
                    img_name, target_qf, region_size, region_size, num_regions,
                    mean_diff, max_diff, std_dev, ok_pct);
            }
        }
        println!();
    }
}

// =============================================================================
// EXPERIMENT 8: End-to-End Armor with Aggressive Delta (Full Pipeline)
// =============================================================================

/// Tests the full Armor encode/decode pipeline using real encoders at various QFs,
/// with the EXISTING codebase (no modifications).
///
/// This establishes the current baseline for comparison against proposed changes.
///
/// Run: cargo test -p phasm-core --release -- --ignored whatsapp_e2e_baseline --nocapture
#[test]
#[ignore]
fn whatsapp_e2e_baseline() {
    use phasm_core::{armor_encode, armor_decode};
    use std::process::Command;

    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
        ("photo_640x480", load_test_vector("photo_640x480_q75_420.jpg")),
    ];

    let passphrase = "whatsapp-survival-experiment-2026";
    let messages: Vec<String> = vec![
        "Hello".to_string(),
        "Meeting at noon".to_string(),
        "The package is under the third bench from the left in Central Park".to_string(),
    ];

    // Encoder: sips (most representative of WhatsApp iOS)
    let qfs: &[u8] = &[85, 80, 75, 70];

    println!();
    println!("EXPERIMENT 8: END-TO-END ARMOR BASELINE (sips recompression)");
    println!("=============================================================");
    println!("Tests current Armor pipeline against sips recompression at WhatsApp QFs.");
    println!();
    println!("{:<16} {:>4} {:>4} {:>8} {:>6} {:>6} {:>4} {:>6}",
        "Image", "Msg", "QF", "Result", "Integ", "RSErr", "r", "Parity");
    println!("{}", "-".repeat(64));

    let mut total = 0u32;
    let mut survived = 0u32;

    for (img_name, cover_bytes) in &test_images {
        for message in &messages {
            let msg_len = message.len();

            let stego = match armor_encode(cover_bytes, message, passphrase) {
                Ok(s) => s,
                Err(e) => {
                    println!("{:<16} {:>4} {:>4} {:>8}", img_name, msg_len, "-", format!("ENC:{}", e));
                    continue;
                }
            };

            for &qf in qfs {
                total += 1;

                // Recompress with sips
                let dir = std::env::temp_dir();
                let pid = std::process::id();
                let input = dir.join(format!("phasm_e2e_in_{pid}_{qf}.jpg"));
                let output = dir.join(format!("phasm_e2e_out_{pid}_{qf}.jpg"));
                std::fs::write(&input, &stego).unwrap();

                let sips_out = Command::new("/usr/bin/sips")
                    .arg("-s").arg("format").arg("jpeg")
                    .arg("-s").arg("formatOptions").arg(qf.to_string())
                    .arg(&input)
                    .arg("--out").arg(&output)
                    .output();

                let _ = std::fs::remove_file(&input);

                let recompressed = match sips_out {
                    Ok(out) if out.status.success() => {
                        let result = std::fs::read(&output).unwrap_or_default();
                        let _ = std::fs::remove_file(&output);
                        result
                    }
                    _ => {
                        let _ = std::fs::remove_file(&output);
                        println!("{:<16} {:>4} {:>4} {:>8}", img_name, msg_len, qf, "SIPS_ERR");
                        continue;
                    }
                };

                match armor_decode(&recompressed, passphrase) {
                    Ok((decoded, quality)) => {
                        let ok = decoded == *message;
                        if ok { survived += 1; }
                        println!("{:<16} {:>4} {:>4} {:>8} {:>5}% {:>6} {:>4} {:>6}",
                            img_name, msg_len, qf,
                            if ok { "YES" } else { "CORRUPT" },
                            quality.integrity_percent,
                            quality.rs_errors_corrected,
                            quality.repetition_factor,
                            quality.parity_len);
                    }
                    Err(_) => {
                        println!("{:<16} {:>4} {:>4} {:>8}", img_name, msg_len, qf, "FAIL");
                    }
                }
            }
        }
        println!();
    }

    println!("{}", "-".repeat(64));
    println!("Total: {survived}/{total} ({:.1}%)",
        if total > 0 { survived as f64 / total as f64 * 100.0 } else { 0.0 });
}

// =============================================================================
// EXPERIMENT 9: Coefficient Sign Stability Across Encoders
// =============================================================================

/// Detailed analysis of coefficient SIGN stability using real external encoders.
///
/// Run: cargo test -p phasm-core --release -- --ignored whatsapp_sign_stability_real --nocapture
#[test]
#[ignore]
fn whatsapp_sign_stability_real() {
    use std::process::Command;

    let cover_bytes = load_real_photo("istockphoto-612x612-baseline.jpg");
    let img = JpegImage::from_bytes(&cover_bytes).unwrap();

    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let orig_qt = img.quant_table(qt_id).unwrap().clone();
    let grid = img.dct_grid(0);
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();

    // Write original to temp
    let dir = std::env::temp_dir();
    let pid = std::process::id();

    let qfs: &[u8] = &[85, 80, 75, 70];

    println!();
    println!("EXPERIMENT 9: SIGN STABILITY WITH REAL ENCODERS (istock_612x408)");
    println!("=================================================================");
    println!("Measures coefficient sign flip rate using real external encoders.");
    println!();

    // Recompressors
    struct Recompressor {
        name: &'static str,
        func: Box<dyn Fn(&[u8], u8) -> Option<Vec<u8>>>,
    }

    let recompressors: Vec<Recompressor> = vec![
        Recompressor {
            name: "sips",
            func: Box::new(|bytes: &[u8], qf: u8| -> Option<Vec<u8>> {
                let dir = std::env::temp_dir();
                let pid = std::process::id();
                let input = dir.join(format!("phasm_sign_sips_in_{pid}.jpg"));
                let output = dir.join(format!("phasm_sign_sips_out_{pid}.jpg"));
                std::fs::write(&input, bytes).ok()?;
                let out = Command::new("/usr/bin/sips")
                    .arg("-s").arg("format").arg("jpeg")
                    .arg("-s").arg("formatOptions").arg(qf.to_string())
                    .arg(&input).arg("--out").arg(&output)
                    .output().ok()?;
                let _ = std::fs::remove_file(&input);
                if !out.status.success() { let _ = std::fs::remove_file(&output); return None; }
                let r = std::fs::read(&output).ok();
                let _ = std::fs::remove_file(&output);
                r
            }),
        },
        Recompressor {
            name: "libjpeg-turbo",
            func: Box::new(|bytes: &[u8], qf: u8| -> Option<Vec<u8>> {
                let dir = std::env::temp_dir();
                let pid = std::process::id();
                let input = dir.join(format!("phasm_sign_ljt_in_{pid}.jpg"));
                let ppm = dir.join(format!("phasm_sign_ljt_{pid}.ppm"));
                let output = dir.join(format!("phasm_sign_ljt_out_{pid}.jpg"));
                std::fs::write(&input, bytes).ok()?;
                let d = Command::new("/opt/homebrew/bin/djpeg")
                    .arg("-ppm").arg("-outfile").arg(&ppm).arg(&input).output().ok()?;
                let _ = std::fs::remove_file(&input);
                if !d.status.success() { return None; }
                let c = Command::new("/opt/homebrew/bin/cjpeg")
                    .arg("-quality").arg(qf.to_string()).arg("-baseline")
                    .arg("-outfile").arg(&output).arg(&ppm).output().ok()?;
                let _ = std::fs::remove_file(&ppm);
                if !c.status.success() { let _ = std::fs::remove_file(&output); return None; }
                let r = std::fs::read(&output).ok();
                let _ = std::fs::remove_file(&output);
                r
            }),
        },
        Recompressor {
            name: "MozJPEG",
            func: Box::new(|bytes: &[u8], qf: u8| -> Option<Vec<u8>> {
                let dir = std::env::temp_dir();
                let pid = std::process::id();
                let input = dir.join(format!("phasm_sign_moz_in_{pid}.jpg"));
                let ppm = dir.join(format!("phasm_sign_moz_{pid}.ppm"));
                let output = dir.join(format!("phasm_sign_moz_out_{pid}.jpg"));
                std::fs::write(&input, bytes).ok()?;
                let d = Command::new("/opt/homebrew/bin/djpeg")
                    .arg("-ppm").arg("-outfile").arg(&ppm).arg(&input).output().ok()?;
                let _ = std::fs::remove_file(&input);
                if !d.status.success() { return None; }
                let c = Command::new("/opt/homebrew/opt/mozjpeg/bin/cjpeg")
                    .arg("-quality").arg(qf.to_string()).arg("-baseline")
                    .arg("-outfile").arg(&output).arg(&ppm).output().ok()?;
                let _ = std::fs::remove_file(&ppm);
                if !c.status.success() { let _ = std::fs::remove_file(&output); return None; }
                let r = std::fs::read(&output).ok();
                let _ = std::fs::remove_file(&output);
                r
            }),
        },
    ];

    // Write original JPEG
    let orig_path = dir.join(format!("phasm_sign_orig_{pid}.jpg"));
    std::fs::write(&orig_path, &cover_bytes).unwrap();

    println!("{:<16} {:>4} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Encoder", "QF", "NonZero_AC", "SignFlips", "SignBER%", "|c|>2_Flips", "|c|>2_BER%");
    println!("{}", "-".repeat(88));

    for recomp in &recompressors {
        for &qf in qfs {
            let recompressed = match (recomp.func)(&cover_bytes, qf) {
                Some(r) => r,
                None => { println!("{:<16} {:>4} ENCODER_ERROR", recomp.name, qf); continue; }
            };

            let recomp_img = match JpegImage::from_bytes(&recompressed) {
                Ok(img) => img,
                Err(_) => { println!("{:<16} {:>4} PARSE_ERROR", recomp.name, qf); continue; }
            };

            let recomp_grid = recomp_img.dct_grid(0);
            let recomp_qt_id = recomp_img.frame_info().components[0].quant_table_id as usize;
            let recomp_qt = recomp_img.quant_table(recomp_qt_id).unwrap();

            let mut nonzero_total = 0u32;
            let mut sign_flips = 0u32;
            let mut big_total = 0u32;   // |coeff| > 2
            let mut big_flips = 0u32;

            let rbw = recomp_grid.blocks_wide();
            let rbt = recomp_grid.blocks_tall();

            for br in 0..bt.min(rbt) {
                for bc in 0..bw.min(rbw) {
                    for i in 0..8 {
                        for j in 0..8 {
                            let freq_idx = i * 8 + j;
                            let zz = NATURAL_TO_ZIGZAG[freq_idx];
                            if zz < 1 || zz > 15 { continue; }

                            let orig_c = grid.get(br, bc, i, j);
                            let recomp_c = recomp_grid.get(br, bc, i, j);

                            if orig_c != 0 {
                                nonzero_total += 1;
                                if (orig_c > 0 && recomp_c <= 0) || (orig_c < 0 && recomp_c >= 0) {
                                    sign_flips += 1;
                                }

                                if orig_c.abs() > 2 {
                                    big_total += 1;
                                    if (orig_c > 0 && recomp_c <= 0) || (orig_c < 0 && recomp_c >= 0) {
                                        big_flips += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let sign_ber = if nonzero_total > 0 { sign_flips as f64 / nonzero_total as f64 * 100.0 } else { 0.0 };
            let big_ber = if big_total > 0 { big_flips as f64 / big_total as f64 * 100.0 } else { 0.0 };

            println!("{:<16} {:>4} {:>12} {:>12} {:>11.2}% {:>12} {:>11.2}%",
                recomp.name, qf, nonzero_total, sign_flips, sign_ber,
                format!("{}/{}", big_flips, big_total), big_ber);
        }
        println!();
    }

    let _ = std::fs::remove_file(&orig_path);
}
