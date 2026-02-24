//! Social media recompression survival test suite.
//!
//! Validates that Armor mode survives JPEG recompression at various quality
//! factors (simulating social media platform processing), and confirms that
//! Ghost mode does NOT survive (expected behavior — honest limitation).

use phasm_core::{armor_encode, armor_decode, armor_capacity, ghost_encode, ghost_decode, JpegImage, QuantTable};
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

const STD_CHROMA_QT: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];

/// Platform quality factors for recompression simulation.
const PLATFORM_QFS: &[(u8, &str)] = &[
    (95, "High Quality"),
    (85, "Twitter/X"),
    (80, "WhatsApp"),
    (75, "Facebook"),
    (70, "Instagram"),
    (53, "WeChat"),
];

/// Message lengths to test across the matrix.
const MESSAGE_LENGTHS: &[usize] = &[10, 20, 50, 80, 100, 150, 200, 500, 1000];

// --- Helpers ---

fn load_test_vector(name: &str) -> Vec<u8> {
    std::fs::read(format!("../test-vectors/{name}")).unwrap()
}

fn load_real_photo(name: &str) -> Vec<u8> {
    std::fs::read(format!("tests/real_photos/{name}")).unwrap()
}

/// Compute a scaled quantization table for a given quality factor.
///
/// Uses the IJG (Independent JPEG Group) quality factor formula:
/// - QF < 50: scale = 5000 / QF
/// - QF >= 50: scale = 200 - 2 * QF
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

/// Detect whether a quant table is closer to standard chroma than luma.
fn is_chroma_table(qt: &QuantTable) -> bool {
    let luma_diff: u32 = qt.values.iter()
        .zip(STD_LUMA_QT.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
        .sum();
    let chroma_diff: u32 = qt.values.iter()
        .zip(STD_CHROMA_QT.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
        .sum();
    chroma_diff < luma_diff
}

/// Simulate JPEG recompression at a target quality factor.
///
/// Performs dequantize→requantize on every DCT coefficient using the
/// target QF's quantization tables, then writes back to JPEG bytes.
/// Returns `None` if the stego bytes can't be parsed (e.g. Huffman decode error).
fn simulate_recompression(stego_bytes: &[u8], target_qf: u8) -> Option<Vec<u8>> {
    let mut img = match JpegImage::from_bytes(stego_bytes) {
        Ok(img) => img,
        Err(_) => return None,
    };
    let frame = img.frame_info().clone();

    // For each component, compute the target quant table and requantize
    for comp_idx in 0..img.num_components() {
        let qt_id = frame.components[comp_idx].quant_table_id as usize;
        let orig_qt = img.quant_table(qt_id).unwrap().clone();

        // Pick the right standard table based on whether this is luma or chroma
        let std_qt = if is_chroma_table(&orig_qt) {
            &STD_CHROMA_QT
        } else {
            &STD_LUMA_QT
        };
        let target_values = scale_quant_table(std_qt, target_qf);

        let grid = img.dct_grid(comp_idx);
        let bw = grid.blocks_wide();
        let bt = grid.blocks_tall();

        // Dequantize with original table, requantize with target table
        for br in 0..bt {
            for bc in 0..bw {
                for i in 0..8 {
                    for j in 0..8 {
                        let freq_idx = i * 8 + j;
                        let coeff = img.dct_grid(comp_idx).get(br, bc, i, j);
                        let q_orig = orig_qt.values[freq_idx];
                        let q_target = target_values[freq_idx];

                        // Dequantize: approximate DCT value
                        let dct_value = coeff as i32 * q_orig as i32;
                        // Requantize at target QF
                        let recompressed = if dct_value >= 0 {
                            (dct_value + q_target as i32 / 2) / q_target as i32
                        } else {
                            (dct_value - q_target as i32 / 2) / q_target as i32
                        };
                        img.dct_grid_mut(comp_idx).set(br, bc, i, j, recompressed as i16);
                    }
                }
            }
        }

        // Replace the quant table with the target table
        img.set_quant_table(qt_id, QuantTable::new(target_values));
    }

    // Rebuild Huffman tables for the modified coefficients
    img.rebuild_huffman_tables();
    Some(img.to_bytes().unwrap())
}

/// Generate a reproducible message of exact byte length.
fn generate_message(len: usize) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789 ";
    let mut msg = String::with_capacity(len);
    for i in 0..len {
        msg.push(CHARS[i % CHARS.len()] as char);
    }
    msg
}

// --- Tests ---

/// Full recompression matrix: multiple images × message lengths × quality factors.
///
/// Slow test — run with: cargo test -p phasm-core -- --ignored recompression_survival
#[test]
#[ignore]
fn armor_recompression_matrix() {
    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("fractal_100x75", load_test_vector("fractal_100x75_q85_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let passphrase = "recompression-test-suite";

    // Results table header
    println!();
    println!("{:<16} {:>7} {:>4}  {:>8}  {:>10} {:>9} {:>5}",
        "Image", "MsgLen", "QF", "Survived", "Integrity", "RS_Errs", "RepF");
    println!("{}", "-".repeat(72));

    let mut total = 0u32;
    let mut survived = 0u32;

    for (img_name, cover_bytes) in &test_images {
        let img = match JpegImage::from_bytes(cover_bytes) {
            Ok(img) => img,
            Err(e) => {
                println!("{:<16} SKIPPED ({})", img_name, e);
                continue;
            }
        };
        let cap = armor_capacity(&img).unwrap();

        for &msg_len in MESSAGE_LENGTHS {
            if msg_len > cap {
                continue; // Skip if message too long for this image
            }

            let message = generate_message(msg_len);
            let stego = match armor_encode(cover_bytes, &message, passphrase) {
                Ok(s) => s,
                Err(_) => continue, // capacity exceeded with overhead
            };

            for &(qf, _platform) in PLATFORM_QFS {
                total += 1;
                let recompressed = match simulate_recompression(&stego, qf) {
                    Some(r) => r,
                    None => {
                        println!("{:<16} {:>7} {:>4}  {:>8}  {:>9}  {:>9} {:>5}",
                            img_name, msg_len, qf, "PARSE_ERR", "-", "-", "-");
                        continue;
                    }
                };
                match armor_decode(&recompressed, passphrase) {
                    Ok((decoded, quality)) => {
                        let ok = decoded == message;
                        if ok {
                            survived += 1;
                        }
                        println!("{:<16} {:>7} {:>4}  {:>8}  {:>9}% {:>9} {:>5}",
                            img_name, msg_len, qf,
                            if ok { "YES" } else { "CORRUPT" },
                            quality.integrity_percent,
                            quality.rs_errors_corrected,
                            quality.repetition_factor);
                    }
                    Err(_) => {
                        println!("{:<16} {:>7} {:>4}  {:>8}  {:>9}  {:>9} {:>5}",
                            img_name, msg_len, qf, "FAIL", "-", "-", "-");
                    }
                }
            }
        }
    }

    println!("{}", "-".repeat(72));
    println!("Total: {survived}/{total} survived ({:.1}%)",
        if total > 0 { survived as f64 / total as f64 * 100.0 } else { 0.0 });
}

/// Ghost mode recompression sensitivity test.
///
/// Documents Ghost mode behavior under recompression. Ghost optimizes for
/// stealth, not robustness. It may survive when the target QF produces the
/// same quant tables as the original (e.g., QF 75 on a QF 75 image), but
/// should fail when the quant tables differ significantly.
#[test]
#[ignore]
fn ghost_recompression_sensitivity() {
    let cover = load_test_vector("photo_320x240_q75_420.jpg");
    let passphrase = "ghost-recomp-test";
    // Test QFs that differ from the original QF 75 (where Ghost should fail)
    // and QF 75 itself (where it may survive since quant tables are identical)
    let test_qfs: &[u8] = &[95, 85, 80, 75, 70, 53];
    let test_msgs: &[usize] = &[20, 100];

    println!();
    println!("Ghost mode recompression sensitivity");
    println!("{:<10} {:>4}  {:>15}", "MsgLen", "QF", "Result");
    println!("{}", "-".repeat(35));

    let mut survived = 0u32;
    let mut total = 0u32;

    for &msg_len in test_msgs {
        let message = generate_message(msg_len);
        let stego = ghost_encode(&cover, &message, passphrase).unwrap();

        for &qf in test_qfs {
            total += 1;
            let recompressed = match simulate_recompression(&stego, qf) {
                Some(r) => r,
                None => {
                    println!("{:<10} {:>4}  {:>15}", msg_len, qf, "PARSE_ERR");
                    continue;
                }
            };
            let result = ghost_decode(&recompressed, passphrase);
            let label = match &result {
                Ok(decoded) if *decoded == message => {
                    survived += 1;
                    "SURVIVED"
                }
                Ok(_) => "CORRUPT",
                Err(_) => "FAIL (expected)",
            };
            println!("{:<10} {:>4}  {:>15}", msg_len, qf, label);
        }
    }

    println!("{}", "-".repeat(35));
    println!("Ghost survived: {survived}/{total}");

    // Ghost should fail at QFs significantly different from the original.
    // QF 53 (WeChat-level) should definitely destroy Ghost embeddings.
    for &msg_len in test_msgs {
        let message = generate_message(msg_len);
        let stego = ghost_encode(&cover, &message, passphrase).unwrap();
        let recompressed = simulate_recompression(&stego, 53)
            .expect("simulate_recompression should succeed for test vector");
        let result = ghost_decode(&recompressed, passphrase);
        assert!(
            result.is_err() || result.as_ref().unwrap() != &message,
            "Ghost unexpectedly survived aggressive recompression (QF 53) with {msg_len}-char message"
        );
    }
}

/// Simulate pixel-domain JPEG recompression at a target quality factor.
///
/// This is what real JPEG re-encoders do:
/// DCT coefficients → IDCT → pixel values (clamped to [0,255]) → forward DCT → quantize.
/// The pixel clamping step is the key source of information loss beyond simple requantization.
/// Returns `None` if the stego bytes can't be parsed.
fn simulate_pixel_recompression(stego_bytes: &[u8], target_qf: u8) -> Option<Vec<u8>> {
    let mut img = match JpegImage::from_bytes(stego_bytes) {
        Ok(img) => img,
        Err(_) => return None,
    };
    let frame = img.frame_info().clone();

    for comp_idx in 0..img.num_components() {
        let qt_id = frame.components[comp_idx].quant_table_id as usize;
        let orig_qt = img.quant_table(qt_id).unwrap().clone();

        let std_qt = if is_chroma_table(&orig_qt) { &STD_CHROMA_QT } else { &STD_LUMA_QT };
        let target_values = scale_quant_table(std_qt, target_qf);

        let grid = img.dct_grid(comp_idx);
        let bw = grid.blocks_wide();
        let bt = grid.blocks_tall();

        for br in 0..bt {
            for bc in 0..bw {
                // Extract block coefficients
                let block_slice = img.dct_grid(comp_idx).block(br, bc);
                let quantized: [i16; 64] = block_slice.try_into().unwrap();

                // IDCT with original quant table → pixel values
                let mut pixels = idct_block(&quantized, &orig_qt.values);

                // Clamp to valid pixel range (this is what causes information loss!)
                for p in pixels.iter_mut() {
                    *p = p.clamp(0.0, 255.0);
                }

                // Forward DCT + quantize with target table
                let recompressed = dct_block(&pixels, &target_values);

                // Write back
                img.dct_grid_mut(comp_idx).block_mut(br, bc).copy_from_slice(&recompressed);
            }
        }

        img.set_quant_table(qt_id, QuantTable::new(target_values));
    }

    img.rebuild_huffman_tables();
    Some(img.to_bytes().unwrap())
}

/// Platform profile for ImageMagick-based recompression simulation.
struct PlatformProfile {
    name: &'static str,
    quality: u8,
    max_dimension: Option<u32>,  // None = no resize
    sharpen: Option<&'static str>,  // ImageMagick -unsharp args
    // Note: progressive field omitted because our JPEG parser only handles
    // baseline (SOF0), not progressive (SOF2). All platform simulations
    // force `-interlace none` to produce baseline JPEGs we can decode.
}

/// Platform profiles for ImageMagick simulation.
/// Based on research in docs/research/social-media-image-processing.md.
const PLATFORMS: &[PlatformProfile] = &[
    PlatformProfile { name: "Twitter/X", quality: 85, max_dimension: Some(4096), sharpen: None },
    PlatformProfile { name: "Facebook", quality: 75, max_dimension: Some(2048), sharpen: Some("0x0.5+0.5+0") },
    PlatformProfile { name: "WhatsApp", quality: 80, max_dimension: Some(1600), sharpen: None },
    PlatformProfile { name: "Instagram", quality: 75, max_dimension: Some(1080), sharpen: Some("0x0.5+0.7+0") },
    PlatformProfile { name: "WeChat", quality: 53, max_dimension: Some(1440), sharpen: None },
    PlatformProfile { name: "Signal", quality: 85, max_dimension: Some(1920), sharpen: None },
    PlatformProfile { name: "Telegram", quality: 82, max_dimension: Some(2560), sharpen: None },
    PlatformProfile { name: "Discord", quality: 80, max_dimension: Some(4096), sharpen: None },
    PlatformProfile { name: "Snapchat", quality: 60, max_dimension: Some(1920), sharpen: None },
];

/// Simulate platform recompression using ImageMagick.
///
/// Shells out to `/opt/ImageMagick/bin/convert` to perform realistic platform
/// processing: resize to max dimension, apply quality factor, optional sharpening.
/// Forces baseline encoding (`-interlace none`) because our parser only handles SOF0.
/// Returns `None` if ImageMagick is not available or the conversion fails.
fn simulate_platform(stego_bytes: &[u8], profile: &PlatformProfile) -> Option<Vec<u8>> {
    use std::process::Command;

    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let input_path = dir.join(format!("phasm_recomp_input_{pid}.jpg"));
    let output_path = dir.join(format!("phasm_recomp_output_{pid}.jpg"));

    std::fs::write(&input_path, stego_bytes).ok()?;

    let mut args: Vec<String> = vec![input_path.to_string_lossy().to_string()];

    // Resize if needed (only shrink, never enlarge)
    if let Some(max_dim) = profile.max_dimension {
        args.push("-resize".to_string());
        args.push(format!("{max_dim}x{max_dim}>"));
    }

    // Quality
    args.push("-quality".to_string());
    args.push(profile.quality.to_string());

    // Sharpen
    if let Some(unsharp) = profile.sharpen {
        args.push("-unsharp".to_string());
        args.push(unsharp.to_string());
    }

    // Force baseline encoding — our parser only handles SOF0, not progressive SOF2
    args.push("-interlace".to_string());
    args.push("none".to_string());

    // Strip metadata (like platforms do)
    args.push("-strip".to_string());

    args.push(output_path.to_string_lossy().to_string());

    let status = Command::new("/opt/ImageMagick/bin/convert")
        .args(&args)
        .status()
        .ok()?;

    if !status.success() {
        // Clean up input file on failure
        let _ = std::fs::remove_file(&input_path);
        return None;
    }

    let result = std::fs::read(&output_path).ok();

    // Clean up temp files
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&output_path);

    result
}

/// Baseline: Armor encode/decode without recompression.
///
/// Quick sanity test (not ignored) — asserts 100% integrity and exact match.
/// Catches regressions in the basic Armor pipeline.
#[test]
fn armor_no_recompression_baseline() {
    let cover = load_test_vector("photo_320x240_q75_420.jpg");
    let img = JpegImage::from_bytes(&cover).unwrap();
    let cap = armor_capacity(&img).unwrap();
    let passphrase = "baseline-test-key";

    for &msg_len in MESSAGE_LENGTHS {
        if msg_len > cap {
            continue;
        }

        let message = generate_message(msg_len);
        let stego = match armor_encode(&cover, &message, passphrase) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let (decoded, quality) = armor_decode(&stego, passphrase)
            .unwrap_or_else(|e| panic!("Armor decode failed for {msg_len}-char message: {e}"));
        assert_eq!(decoded, message, "Message mismatch for {msg_len}-char message");
        assert_eq!(quality.integrity_percent, 100,
            "Expected 100% integrity without recompression for {msg_len}-char message");
        assert_eq!(quality.rs_errors_corrected, 0,
            "Expected 0 RS errors without recompression for {msg_len}-char message");
    }
}

/// Pixel-domain recompression matrix: multiple images x message lengths x quality factors.
///
/// Like `armor_recompression_matrix` but uses pixel-domain round-trip simulation
/// (IDCT → clamp → DCT) which is more realistic than coefficient-domain requantization.
///
/// Slow test — run with: cargo test -p phasm-core -- --ignored armor_pixel_recompression_matrix
#[test]
#[ignore]
fn armor_pixel_recompression_matrix() {
    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("fractal_100x75", load_test_vector("fractal_100x75_q85_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let passphrase = "pixel-recompression-test-suite";

    // Results table header
    println!();
    println!("PIXEL-DOMAIN recompression matrix");
    println!("{:<16} {:>7} {:>4}  {:>8}  {:>10} {:>9} {:>5}",
        "Image", "MsgLen", "QF", "Survived", "Integrity", "RS_Errs", "RepF");
    println!("{}", "-".repeat(72));

    let mut total = 0u32;
    let mut survived = 0u32;

    for (img_name, cover_bytes) in &test_images {
        let img = match JpegImage::from_bytes(cover_bytes) {
            Ok(img) => img,
            Err(e) => {
                println!("{:<16} SKIPPED ({})", img_name, e);
                continue;
            }
        };
        let cap = armor_capacity(&img).unwrap();

        for &msg_len in MESSAGE_LENGTHS {
            if msg_len > cap {
                continue;
            }

            let message = generate_message(msg_len);
            let stego = match armor_encode(cover_bytes, &message, passphrase) {
                Ok(s) => s,
                Err(_) => continue,
            };

            for &(qf, _platform) in PLATFORM_QFS {
                total += 1;
                let recompressed = match simulate_pixel_recompression(&stego, qf) {
                    Some(r) => r,
                    None => {
                        println!("{:<16} {:>7} {:>4}  {:>8}  {:>9}  {:>9} {:>5}",
                            img_name, msg_len, qf, "PARSE_ERR", "-", "-", "-");
                        continue;
                    }
                };
                match armor_decode(&recompressed, passphrase) {
                    Ok((decoded, quality)) => {
                        let ok = decoded == message;
                        if ok {
                            survived += 1;
                        }
                        println!("{:<16} {:>7} {:>4}  {:>8}  {:>9}% {:>9} {:>5}",
                            img_name, msg_len, qf,
                            if ok { "YES" } else { "CORRUPT" },
                            quality.integrity_percent,
                            quality.rs_errors_corrected,
                            quality.repetition_factor);
                    }
                    Err(_) => {
                        println!("{:<16} {:>7} {:>4}  {:>8}  {:>9}  {:>9} {:>5}",
                            img_name, msg_len, qf, "FAIL", "-", "-", "-");
                    }
                }
            }
        }
    }

    println!("{}", "-".repeat(72));
    println!("PIXEL total: {survived}/{total} survived ({:.1}%)",
        if total > 0 { survived as f64 / total as f64 * 100.0 } else { 0.0 });
}

/// Full platform simulation matrix using ImageMagick.
///
/// Tests Armor survival against realistic platform processing (resize, quality,
/// sharpening, metadata stripping) via external ImageMagick convert command.
/// Gracefully skips if ImageMagick is not available at `/opt/ImageMagick/bin/convert`.
///
/// Slow test — run with: cargo test -p phasm-core -- --ignored armor_platform_simulation_matrix
#[test]
#[ignore]
fn armor_platform_simulation_matrix() {
    // Check if ImageMagick is available before running the full matrix
    let im_available = std::process::Command::new("/opt/ImageMagick/bin/convert")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !im_available {
        println!();
        println!("SKIPPED: ImageMagick not found at /opt/ImageMagick/bin/convert");
        println!("Install ImageMagick to run platform simulation tests.");
        return;
    }

    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("fractal_100x75", load_test_vector("fractal_100x75_q85_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    // Fewer message lengths to keep runtime manageable
    let platform_msg_lengths: &[usize] = &[20, 100, 500];
    let passphrase = "platform-simulation-test-suite";

    // Results table header
    println!();
    println!("PLATFORM SIMULATION matrix (ImageMagick)");
    println!("{:<16} {:>7} {:<12} {:>3}  {:>8}  {:>10} {:>9} {:>5}",
        "Image", "MsgLen", "Platform", "QF", "Survived", "Integrity", "RS_Errs", "RepF");
    println!("{}", "-".repeat(88));

    let mut total = 0u32;
    let mut survived = 0u32;

    for (img_name, cover_bytes) in &test_images {
        let img = match JpegImage::from_bytes(cover_bytes) {
            Ok(img) => img,
            Err(e) => {
                println!("{:<16} SKIPPED ({})", img_name, e);
                continue;
            }
        };
        let cap = armor_capacity(&img).unwrap();

        for &msg_len in platform_msg_lengths {
            if msg_len > cap {
                continue;
            }

            let message = generate_message(msg_len);
            let stego = match armor_encode(cover_bytes, &message, passphrase) {
                Ok(s) => s,
                Err(_) => continue,
            };

            for profile in PLATFORMS {
                total += 1;
                let recompressed = match simulate_platform(&stego, profile) {
                    Some(r) => r,
                    None => {
                        println!("{:<16} {:>7} {:<12} {:>3}  {:>8}  {:>9}  {:>9} {:>5}",
                            img_name, msg_len, profile.name, profile.quality,
                            "IM_ERR", "-", "-", "-");
                        continue;
                    }
                };
                match armor_decode(&recompressed, passphrase) {
                    Ok((decoded, quality)) => {
                        let ok = decoded == message;
                        if ok {
                            survived += 1;
                        }
                        println!("{:<16} {:>7} {:<12} {:>3}  {:>8}  {:>9}% {:>9} {:>5}",
                            img_name, msg_len, profile.name, profile.quality,
                            if ok { "YES" } else { "CORRUPT" },
                            quality.integrity_percent,
                            quality.rs_errors_corrected,
                            quality.repetition_factor);
                    }
                    Err(_) => {
                        println!("{:<16} {:>7} {:<12} {:>3}  {:>8}  {:>9}  {:>9} {:>5}",
                            img_name, msg_len, profile.name, profile.quality,
                            "FAIL", "-", "-", "-");
                    }
                }
            }
        }
    }

    println!("{}", "-".repeat(88));
    println!("PLATFORM total: {survived}/{total} survived ({:.1}%)",
        if total > 0 { survived as f64 / total as f64 * 100.0 } else { 0.0 });
}
