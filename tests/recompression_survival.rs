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

// =============================================================================
// Social Media Mode Experiment
// =============================================================================

/// Preprocessing configuration for the social media experiment.
struct PreprocessConfig {
    name: &'static str,
    max_long_edge: u32,
    quality: u8,
}

const SOCIAL_MEDIA_PREPROCESS: &[PreprocessConfig] = &[
    PreprocessConfig { name: "SocMedia/QF70", max_long_edge: 1600, quality: 70 },
    PreprocessConfig { name: "SocMedia/QF75", max_long_edge: 1600, quality: 75 },
];

/// Pre-process a cover image for social media mode using ImageMagick.
///
/// Resizes to max_long_edge (shrink only) and recompresses at the target QF.
/// Forces baseline JPEG. Returns the preprocessed bytes, or None on failure.
fn preprocess_for_social_media(cover_bytes: &[u8], config: &PreprocessConfig) -> Option<Vec<u8>> {
    use std::process::Command;

    let pid = std::process::id();
    let tid: u64 = config.quality as u64 * 10000 + config.max_long_edge as u64;
    let dir = std::env::temp_dir();
    let input_path = dir.join(format!("phasm_sm_input_{pid}_{tid}.jpg"));
    let output_path = dir.join(format!("phasm_sm_output_{pid}_{tid}.jpg"));

    std::fs::write(&input_path, cover_bytes).ok()?;

    let dim = config.max_long_edge;
    let status = Command::new("/opt/ImageMagick/bin/convert")
        .args([
            &input_path.to_string_lossy().to_string(),
            "-resize", &format!("{dim}x{dim}>"),
            "-quality", &config.quality.to_string(),
            "-interlace", "none",
            &output_path.to_string_lossy().to_string(),
        ])
        .status()
        .ok()?;

    if !status.success() {
        let _ = std::fs::remove_file(&input_path);
        return None;
    }

    let result = std::fs::read(&output_path).ok();
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&output_path);
    result
}

/// Social Media Mode A/B experiment.
///
/// Compares Armor survival rates:
///   A (Baseline): encode original cover → platform recompression → decode
///   B (Social Media): preprocess cover (resize 1600px + recompress QF 70/75) → encode → platform recomp → decode
///
/// Run with: cargo test -p phasm-core -- --ignored armor_social_media_experiment --nocapture
#[test]
#[ignore]
fn armor_social_media_experiment() {
    // Check ImageMagick availability
    let im_available = std::process::Command::new("/opt/ImageMagick/bin/convert")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !im_available {
        println!("\nSKIPPED: ImageMagick not found at /opt/ImageMagick/bin/convert");
        return;
    }

    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("fractal_100x75", load_test_vector("fractal_100x75_q85_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let experiment_msg_lengths: &[usize] = &[10, 20, 50, 100];
    let passphrase = "social-media-experiment-2026";

    // Track results per mode for summary
    struct ModeStats {
        name: String,
        total: u32,
        survived: u32,
    }
    let mut all_stats: Vec<ModeStats> = Vec::new();

    // Per-platform stats for the final breakdown
    struct PlatformModeStats {
        mode_name: String,
        platform_name: String,
        total: u32,
        survived: u32,
    }
    let mut platform_stats: Vec<PlatformModeStats> = Vec::new();

    // =========================================================================
    // Phase 1: Capacity diagnostic report
    // =========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          SOCIAL MEDIA MODE — A/B EXPERIMENT                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("CAPACITY DIAGNOSTIC REPORT");
    println!("{:<16} {:<14} {:>10} {:>10} {:>8}",
        "Image", "Mode", "Dimensions", "FileSize", "Capacity");
    println!("{}", "-".repeat(62));

    for (img_name, cover_bytes) in &test_images {
        // Original
        if let Ok(img) = JpegImage::from_bytes(cover_bytes) {
            let fi = img.frame_info();
            let cap = armor_capacity(&img).unwrap_or(0);
            println!("{:<16} {:<14} {:>4}x{:<5} {:>8} {:>8}",
                img_name, "Original", fi.width, fi.height,
                format!("{}K", cover_bytes.len() / 1024), format!("{cap}B"));
        }

        // Preprocessed variants
        for config in SOCIAL_MEDIA_PREPROCESS {
            if let Some(pp_bytes) = preprocess_for_social_media(cover_bytes, config) {
                if let Ok(img) = JpegImage::from_bytes(&pp_bytes) {
                    let fi = img.frame_info();
                    let cap = armor_capacity(&img).unwrap_or(0);
                    println!("{:<16} {:<14} {:>4}x{:<5} {:>8} {:>8}",
                        img_name, config.name, fi.width, fi.height,
                        format!("{}K", pp_bytes.len() / 1024), format!("{cap}B"));
                }
            }
        }
    }

    // =========================================================================
    // Phase 2: Full experiment matrix
    // =========================================================================
    println!();
    println!("EXPERIMENT MATRIX");
    println!("{:<14} {:<16} {:>4} {:<12} {:>3}  {:>8} {:>6} {:>6} {:>4}",
        "Mode", "Image", "Msg", "Platform", "QF", "Result", "Integ", "RSErr", "r");
    println!("{}", "-".repeat(95));

    // --- Mode A: Baseline ---
    {
        let mode_name = "Baseline";
        let mut mode_total = 0u32;
        let mut mode_survived = 0u32;

        for (img_name, cover_bytes) in &test_images {
            let img = match JpegImage::from_bytes(cover_bytes) {
                Ok(img) => img,
                Err(_) => continue,
            };
            let cap = armor_capacity(&img).unwrap_or(0);

            for &msg_len in experiment_msg_lengths {
                if msg_len > cap { continue; }

                let message = generate_message(msg_len);
                let stego = match armor_encode(cover_bytes, &message, passphrase) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                for profile in PLATFORMS {
                    mode_total += 1;
                    let recompressed = match simulate_platform(&stego, profile) {
                        Some(r) => r,
                        None => {
                            println!("{:<14} {:<16} {:>4} {:<12} {:>3}  {:>8} {:>6} {:>6} {:>4}",
                                mode_name, img_name, msg_len, profile.name, profile.quality,
                                "IM_ERR", "-", "-", "-");
                            continue;
                        }
                    };
                    let (survived_str, integ, rs_err, rep_f) = match armor_decode(&recompressed, passphrase) {
                        Ok((decoded, quality)) => {
                            let ok = decoded == message;
                            if ok { mode_survived += 1; }
                            // Track per-platform stats
                            let ps = platform_stats.iter_mut().find(|p| p.mode_name == mode_name && p.platform_name == profile.name);
                            if let Some(ps) = ps {
                                ps.total += 1;
                                if ok { ps.survived += 1; }
                            } else {
                                platform_stats.push(PlatformModeStats {
                                    mode_name: mode_name.to_string(),
                                    platform_name: profile.name.to_string(),
                                    total: 1,
                                    survived: if ok { 1 } else { 0 },
                                });
                            }
                            (
                                if ok { "YES" } else { "CORRUPT" },
                                format!("{}%", quality.integrity_percent),
                                format!("{}", quality.rs_errors_corrected),
                                format!("{}", quality.repetition_factor),
                            )
                        }
                        Err(_) => {
                            let ps = platform_stats.iter_mut().find(|p| p.mode_name == mode_name && p.platform_name == profile.name);
                            if let Some(ps) = ps { ps.total += 1; }
                            else {
                                platform_stats.push(PlatformModeStats {
                                    mode_name: mode_name.to_string(),
                                    platform_name: profile.name.to_string(),
                                    total: 1, survived: 0,
                                });
                            }
                            ("FAIL", "-".to_string(), "-".to_string(), "-".to_string())
                        }
                    };
                    println!("{:<14} {:<16} {:>4} {:<12} {:>3}  {:>8} {:>6} {:>6} {:>4}",
                        mode_name, img_name, msg_len, profile.name, profile.quality,
                        survived_str, integ, rs_err, rep_f);
                }
            }
        }

        all_stats.push(ModeStats { name: mode_name.to_string(), total: mode_total, survived: mode_survived });
    }

    // --- Mode B: Social Media Mode (for each QF) ---
    for config in SOCIAL_MEDIA_PREPROCESS {
        let mode_name = config.name;
        let mut mode_total = 0u32;
        let mut mode_survived = 0u32;

        for (img_name, cover_bytes) in &test_images {
            // Preprocess the cover
            let pp_bytes = match preprocess_for_social_media(cover_bytes, config) {
                Some(b) => b,
                None => {
                    println!("{:<14} {:<16} PREPROCESS FAILED", mode_name, img_name);
                    continue;
                }
            };

            let img = match JpegImage::from_bytes(&pp_bytes) {
                Ok(img) => img,
                Err(_) => continue,
            };
            let cap = armor_capacity(&img).unwrap_or(0);

            for &msg_len in experiment_msg_lengths {
                if msg_len > cap { continue; }

                let message = generate_message(msg_len);
                let stego = match armor_encode(&pp_bytes, &message, passphrase) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                for profile in PLATFORMS {
                    mode_total += 1;
                    let recompressed = match simulate_platform(&stego, profile) {
                        Some(r) => r,
                        None => {
                            println!("{:<14} {:<16} {:>4} {:<12} {:>3}  {:>8} {:>6} {:>6} {:>4}",
                                mode_name, img_name, msg_len, profile.name, profile.quality,
                                "IM_ERR", "-", "-", "-");
                            continue;
                        }
                    };
                    let (survived_str, integ, rs_err, rep_f) = match armor_decode(&recompressed, passphrase) {
                        Ok((decoded, quality)) => {
                            let ok = decoded == message;
                            if ok { mode_survived += 1; }
                            let ps = platform_stats.iter_mut().find(|p| p.mode_name == mode_name && p.platform_name == profile.name);
                            if let Some(ps) = ps {
                                ps.total += 1;
                                if ok { ps.survived += 1; }
                            } else {
                                platform_stats.push(PlatformModeStats {
                                    mode_name: mode_name.to_string(),
                                    platform_name: profile.name.to_string(),
                                    total: 1,
                                    survived: if ok { 1 } else { 0 },
                                });
                            }
                            (
                                if ok { "YES" } else { "CORRUPT" },
                                format!("{}%", quality.integrity_percent),
                                format!("{}", quality.rs_errors_corrected),
                                format!("{}", quality.repetition_factor),
                            )
                        }
                        Err(_) => {
                            let ps = platform_stats.iter_mut().find(|p| p.mode_name == mode_name && p.platform_name == profile.name);
                            if let Some(ps) = ps { ps.total += 1; }
                            else {
                                platform_stats.push(PlatformModeStats {
                                    mode_name: mode_name.to_string(),
                                    platform_name: profile.name.to_string(),
                                    total: 1, survived: 0,
                                });
                            }
                            ("FAIL", "-".to_string(), "-".to_string(), "-".to_string())
                        }
                    };
                    println!("{:<14} {:<16} {:>4} {:<12} {:>3}  {:>8} {:>6} {:>6} {:>4}",
                        mode_name, img_name, msg_len, profile.name, profile.quality,
                        survived_str, integ, rs_err, rep_f);
                }
            }
        }

        all_stats.push(ModeStats { name: mode_name.to_string(), total: mode_total, survived: mode_survived });
    }

    // =========================================================================
    // Phase 3: Summary
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("SUMMARY — Overall Survival Rates");
    println!("{:<16} {:>8} {:>8} {:>8}", "Mode", "Survived", "Total", "Rate");
    println!("{}", "-".repeat(44));
    for stat in &all_stats {
        let rate = if stat.total > 0 { stat.survived as f64 / stat.total as f64 * 100.0 } else { 0.0 };
        println!("{:<16} {:>8} {:>8} {:>7.1}%", stat.name, stat.survived, stat.total, rate);
    }

    println!();
    println!("SUMMARY — Per-Platform Breakdown");
    println!("{:<12} {:>16} {:>16} {:>16}", "Platform",
        all_stats.get(0).map(|s| s.name.as_str()).unwrap_or("-"),
        all_stats.get(1).map(|s| s.name.as_str()).unwrap_or("-"),
        all_stats.get(2).map(|s| s.name.as_str()).unwrap_or("-"));
    println!("{}", "-".repeat(64));

    for profile in PLATFORMS {
        let rates: Vec<String> = all_stats.iter().map(|mode_stat| {
            if let Some(ps) = platform_stats.iter().find(|p| p.mode_name == mode_stat.name && p.platform_name == profile.name) {
                if ps.total > 0 {
                    format!("{}/{} ({:.0}%)", ps.survived, ps.total,
                        ps.survived as f64 / ps.total as f64 * 100.0)
                } else {
                    "-".to_string()
                }
            } else {
                "-".to_string()
            }
        }).collect();

        println!("{:<12} {:>16} {:>16} {:>16}",
            profile.name,
            rates.get(0).map(|s| s.as_str()).unwrap_or("-"),
            rates.get(1).map(|s| s.as_str()).unwrap_or("-"),
            rates.get(2).map(|s| s.as_str()).unwrap_or("-"));
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
}

// =============================================================================
// Real Encoder Experiment (libjpeg-turbo, MozJPEG, AppleJPEG)
// =============================================================================

/// Encoder profile for real JPEG encoder tools.
struct EncoderProfile {
    name: &'static str,
    /// Shell out: decompress stego → recompress at target QF.
    /// Returns the recompressed JPEG bytes, or None on failure.
    recompress_fn: fn(stego_bytes: &[u8], qf: u8) -> Option<Vec<u8>>,
    /// Quality factors to test.
    qfs: &'static [u8],
}

/// Recompress using libjpeg-turbo (djpeg → cjpeg pipeline).
/// Twitter, Discord, Telegram Android, WhatsApp Android.
fn recompress_libjpeg_turbo(stego_bytes: &[u8], qf: u8) -> Option<Vec<u8>> {
    use std::process::Command;
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let input = dir.join(format!("phasm_ljt_in_{pid}.jpg"));
    let ppm = dir.join(format!("phasm_ljt_{pid}.ppm"));
    let output = dir.join(format!("phasm_ljt_out_{pid}.jpg"));

    std::fs::write(&input, stego_bytes).ok()?;

    // Decompress
    let djpeg_out = Command::new("/opt/homebrew/bin/djpeg")
        .arg("-ppm")
        .arg("-outfile").arg(&ppm)
        .arg(&input)
        .output().ok()?;
    if !djpeg_out.status.success() {
        let _ = std::fs::remove_file(&input);
        return None;
    }

    // Recompress with libjpeg-turbo cjpeg (baseline by default)
    let cjpeg_out = Command::new("/opt/homebrew/bin/cjpeg")
        .arg("-quality").arg(qf.to_string())
        .arg("-baseline")
        .arg("-outfile").arg(&output)
        .arg(&ppm)
        .output().ok()?;

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&ppm);

    if !cjpeg_out.status.success() {
        let _ = std::fs::remove_file(&output);
        return None;
    }

    let result = std::fs::read(&output).ok();
    let _ = std::fs::remove_file(&output);
    result
}

/// Recompress using MozJPEG (djpeg → cjpeg pipeline).
/// Facebook, Instagram.
fn recompress_mozjpeg(stego_bytes: &[u8], qf: u8) -> Option<Vec<u8>> {
    use std::process::Command;
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let input = dir.join(format!("phasm_moz_in_{pid}.jpg"));
    let ppm = dir.join(format!("phasm_moz_{pid}.ppm"));
    let output = dir.join(format!("phasm_moz_out_{pid}.jpg"));

    std::fs::write(&input, stego_bytes).ok()?;

    // Use libjpeg-turbo djpeg for decompression (MozJPEG's djpeg also works)
    let djpeg_out = Command::new("/opt/homebrew/bin/djpeg")
        .arg("-ppm")
        .arg("-outfile").arg(&ppm)
        .arg(&input)
        .output().ok()?;
    if !djpeg_out.status.success() {
        let _ = std::fs::remove_file(&input);
        return None;
    }

    // Recompress with MozJPEG cjpeg (-baseline forces non-progressive)
    let cjpeg_out = Command::new("/opt/homebrew/opt/mozjpeg/bin/cjpeg")
        .arg("-quality").arg(qf.to_string())
        .arg("-baseline")
        .arg("-outfile").arg(&output)
        .arg(&ppm)
        .output().ok()?;

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&ppm);

    if !cjpeg_out.status.success() {
        let _ = std::fs::remove_file(&output);
        return None;
    }

    let result = std::fs::read(&output).ok();
    let _ = std::fs::remove_file(&output);
    result
}

/// Recompress using Apple's sips (CoreImage/AppleJPEG framework).
/// iMessage, WhatsApp iOS, Telegram iOS.
fn recompress_sips(stego_bytes: &[u8], qf: u8) -> Option<Vec<u8>> {
    use std::process::Command;
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let input = dir.join(format!("phasm_sips_in_{pid}.jpg"));
    let output = dir.join(format!("phasm_sips_out_{pid}.jpg"));

    std::fs::write(&input, stego_bytes).ok()?;

    // sips recompresses directly (pixel-domain round-trip through CoreImage)
    let sips_out = Command::new("/usr/bin/sips")
        .arg("-s").arg("format").arg("jpeg")
        .arg("-s").arg("formatOptions").arg(qf.to_string())
        .arg(&input)
        .arg("--out").arg(&output)
        .output().ok()?;

    let _ = std::fs::remove_file(&input);

    if !sips_out.status.success() {
        let _ = std::fs::remove_file(&output);
        return None;
    }

    let result = std::fs::read(&output).ok();
    let _ = std::fs::remove_file(&output);
    result
}

/// Recompress using our internal pixel-domain simulation.
/// Provides a pure-Rust comparison baseline.
fn recompress_internal(stego_bytes: &[u8], qf: u8) -> Option<Vec<u8>> {
    simulate_pixel_recompression(stego_bytes, qf)
}

/// Real encoder experiment: tests Armor survival against three real JPEG
/// encoder families (libjpeg-turbo, MozJPEG, AppleJPEG) plus internal sim.
///
/// Run with: cargo test -p phasm-core -- --ignored armor_real_encoder_experiment --nocapture
#[test]
#[ignore]
fn armor_real_encoder_experiment() {
    // Check encoder availability
    let ljt_available = std::process::Command::new("/opt/homebrew/bin/cjpeg")
        .arg("-version").output().map(|o| o.status.success() || !o.stderr.is_empty()).unwrap_or(false);
    let moz_available = std::process::Command::new("/opt/homebrew/opt/mozjpeg/bin/cjpeg")
        .arg("-version").output().map(|o| o.status.success() || !o.stderr.is_empty()).unwrap_or(false);
    let sips_available = std::process::Command::new("/usr/bin/sips")
        .arg("--help").output().map(|o| o.status.success()).unwrap_or(false);

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║     REAL ENCODER EXPERIMENT — Armor Robustness Fixes               ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Encoders:                                                         ║");
    println!("║  • libjpeg-turbo: {} (Twitter, Discord, Telegram Android)      ║",
        if ljt_available { "AVAILABLE" } else { "MISSING  " });
    println!("║  • MozJPEG:       {} (Facebook, Instagram)                     ║",
        if moz_available { "AVAILABLE" } else { "MISSING  " });
    println!("║  • AppleJPEG:     {} (iMessage, WhatsApp iOS, Telegram iOS)    ║",
        if sips_available { "AVAILABLE" } else { "MISSING  " });
    println!("║  • Internal sim:  AVAILABLE (pure-Rust baseline)                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let encoders: Vec<EncoderProfile> = vec![
        EncoderProfile {
            name: "Internal",
            recompress_fn: recompress_internal,
            qfs: &[95, 85, 80, 75, 70, 53],
        },
        EncoderProfile {
            name: "libjpeg-turbo",
            recompress_fn: recompress_libjpeg_turbo,
            qfs: &[95, 85, 80, 75, 70, 53],
        },
        EncoderProfile {
            name: "MozJPEG",
            recompress_fn: recompress_mozjpeg,
            qfs: &[95, 85, 80, 75, 70, 53],
        },
        EncoderProfile {
            name: "AppleJPEG",
            recompress_fn: recompress_sips,
            qfs: &[95, 85, 80, 75, 70, 53],
        },
    ];

    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("fractal_100x75", load_test_vector("fractal_100x75_q85_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let msg_lengths: &[usize] = &[10, 20, 50, 100, 200, 500];
    let passphrase = "real-encoder-experiment-2026";

    // Capacity report
    println!();
    println!("CAPACITY REPORT (with robustness fixes)");
    println!("{:<16} {:>10} {:>10}", "Image", "Dimensions", "Capacity");
    println!("{}", "-".repeat(40));
    for (img_name, cover_bytes) in &test_images {
        if let Ok(img) = JpegImage::from_bytes(cover_bytes) {
            let fi = img.frame_info();
            let cap = armor_capacity(&img).unwrap_or(0);
            println!("{:<16} {:>4}x{:<5} {:>8}B", img_name, fi.width, fi.height, cap);
        }
    }

    // Per-encoder summary tracking
    struct EncoderStats {
        name: String,
        total: u32,
        survived: u32,
    }
    let mut encoder_summaries: Vec<EncoderStats> = Vec::new();

    // Per-encoder × QF tracking
    struct QfStats {
        encoder: String,
        qf: u8,
        total: u32,
        survived: u32,
    }
    let mut qf_summaries: Vec<QfStats> = Vec::new();

    // Per-encoder × image tracking
    struct ImgStats {
        encoder: String,
        image: String,
        total: u32,
        survived: u32,
    }
    let mut img_summaries: Vec<ImgStats> = Vec::new();

    // Full matrix
    println!();
    println!("FULL EXPERIMENT MATRIX");
    println!("{:<14} {:<16} {:>4} {:>3}  {:>8} {:>6} {:>6} {:>4} {:>6}",
        "Encoder", "Image", "Msg", "QF", "Result", "Integ", "RSErr", "r", "Parity");
    println!("{}", "-".repeat(88));

    for encoder in &encoders {
        if encoder.name == "libjpeg-turbo" && !ljt_available { continue; }
        if encoder.name == "MozJPEG" && !moz_available { continue; }
        if encoder.name == "AppleJPEG" && !sips_available { continue; }

        let mut enc_total = 0u32;
        let mut enc_survived = 0u32;

        for (img_name, cover_bytes) in &test_images {
            let img = match JpegImage::from_bytes(cover_bytes) {
                Ok(img) => img,
                Err(_) => continue,
            };
            let cap = armor_capacity(&img).unwrap_or(0);

            for &msg_len in msg_lengths {
                if msg_len > cap { continue; }

                let message = generate_message(msg_len);
                let stego = match armor_encode(cover_bytes, &message, passphrase) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                for &qf in encoder.qfs {
                    enc_total += 1;

                    let recompressed = match (encoder.recompress_fn)(&stego, qf) {
                        Some(r) => r,
                        None => {
                            println!("{:<14} {:<16} {:>4} {:>3}  {:>8} {:>6} {:>6} {:>4} {:>6}",
                                encoder.name, img_name, msg_len, qf,
                                "ENC_ERR", "-", "-", "-", "-");
                            // Track
                            let qs = qf_summaries.iter_mut().find(|s| s.encoder == encoder.name && s.qf == qf);
                            if let Some(qs) = qs { qs.total += 1; }
                            else { qf_summaries.push(QfStats { encoder: encoder.name.to_string(), qf, total: 1, survived: 0 }); }
                            let is = img_summaries.iter_mut().find(|s| s.encoder == encoder.name && s.image == *img_name);
                            if let Some(is) = is { is.total += 1; }
                            else { img_summaries.push(ImgStats { encoder: encoder.name.to_string(), image: img_name.to_string(), total: 1, survived: 0 }); }
                            continue;
                        }
                    };

                    let (result_str, integ, rs_err, rep_f, parity) = match armor_decode(&recompressed, passphrase) {
                        Ok((decoded, quality)) => {
                            let ok = decoded == message;
                            if ok { enc_survived += 1; }
                            // Track QF
                            let qs = qf_summaries.iter_mut().find(|s| s.encoder == encoder.name && s.qf == qf);
                            if let Some(qs) = qs { qs.total += 1; if ok { qs.survived += 1; } }
                            else { qf_summaries.push(QfStats { encoder: encoder.name.to_string(), qf, total: 1, survived: if ok { 1 } else { 0 } }); }
                            // Track image
                            let is = img_summaries.iter_mut().find(|s| s.encoder == encoder.name && s.image == *img_name);
                            if let Some(is) = is { is.total += 1; if ok { is.survived += 1; } }
                            else { img_summaries.push(ImgStats { encoder: encoder.name.to_string(), image: img_name.to_string(), total: 1, survived: if ok { 1 } else { 0 } }); }
                            (
                                if ok { "YES" } else { "CORRUPT" },
                                format!("{}%", quality.integrity_percent),
                                format!("{}", quality.rs_errors_corrected),
                                format!("{}", quality.repetition_factor),
                                format!("{}", quality.parity_len),
                            )
                        }
                        Err(_) => {
                            let qs = qf_summaries.iter_mut().find(|s| s.encoder == encoder.name && s.qf == qf);
                            if let Some(qs) = qs { qs.total += 1; }
                            else { qf_summaries.push(QfStats { encoder: encoder.name.to_string(), qf, total: 1, survived: 0 }); }
                            let is = img_summaries.iter_mut().find(|s| s.encoder == encoder.name && s.image == *img_name);
                            if let Some(is) = is { is.total += 1; }
                            else { img_summaries.push(ImgStats { encoder: encoder.name.to_string(), image: img_name.to_string(), total: 1, survived: 0 }); }
                            ("FAIL", "-".to_string(), "-".to_string(), "-".to_string(), "-".to_string())
                        }
                    };

                    println!("{:<14} {:<16} {:>4} {:>3}  {:>8} {:>6} {:>6} {:>4} {:>6}",
                        encoder.name, img_name, msg_len, qf,
                        result_str, integ, rs_err, rep_f, parity);
                }
            }
        }

        encoder_summaries.push(EncoderStats {
            name: encoder.name.to_string(),
            total: enc_total,
            survived: enc_survived,
        });
    }

    // =========================================================================
    // Summary tables
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("SUMMARY — Overall by Encoder");
    println!("{:<16} {:>8} {:>8} {:>8}", "Encoder", "Survived", "Total", "Rate");
    println!("{}", "-".repeat(44));
    for es in &encoder_summaries {
        let rate = if es.total > 0 { es.survived as f64 / es.total as f64 * 100.0 } else { 0.0 };
        println!("{:<16} {:>8} {:>8} {:>7.1}%", es.name, es.survived, es.total, rate);
    }

    println!();
    println!("SUMMARY — By Encoder × Quality Factor");
    print!("{:>4}", "QF");
    for es in &encoder_summaries {
        print!(" {:>16}", es.name);
    }
    println!();
    println!("{}", "-".repeat(4 + encoder_summaries.len() * 17));
    for &qf in &[95u8, 85, 80, 75, 70, 53] {
        print!("{:>4}", qf);
        for es in &encoder_summaries {
            if let Some(qs) = qf_summaries.iter().find(|s| s.encoder == es.name && s.qf == qf) {
                let rate = if qs.total > 0 { qs.survived as f64 / qs.total as f64 * 100.0 } else { 0.0 };
                print!(" {:>5}/{:<4} ({:>4.0}%)", qs.survived, qs.total, rate);
            } else {
                print!(" {:>16}", "-");
            }
        }
        println!();
    }

    println!();
    println!("SUMMARY — By Encoder × Image");
    print!("{:<16}", "Image");
    for es in &encoder_summaries {
        print!(" {:>16}", es.name);
    }
    println!();
    println!("{}", "-".repeat(16 + encoder_summaries.len() * 17));
    for (img_name, _) in &test_images {
        print!("{:<16}", img_name);
        for es in &encoder_summaries {
            if let Some(is) = img_summaries.iter().find(|s| s.encoder == es.name && s.image == *img_name) {
                let rate = if is.total > 0 { is.survived as f64 / is.total as f64 * 100.0 } else { 0.0 };
                print!(" {:>5}/{:<4} ({:>4.0}%)", is.survived, is.total, rate);
            } else {
                print!(" {:>16}", "-");
            }
        }
        println!();
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
}

// =============================================================================
// Real Encoder Experiment V2 — WITH 1600px long-edge resize
// =============================================================================

/// Resize a JPEG to max long-edge using sips+djpeg, outputting PPM.
/// PPM is used instead of BMP because cjpeg accepts PPM natively.
/// Returns the PPM file path, or None if resize not needed / failed.
fn resize_to_ppm(stego_bytes: &[u8], max_long_edge: u32) -> Option<std::path::PathBuf> {
    // Check if resize is needed
    let img = JpegImage::from_bytes(stego_bytes).ok()?;
    let fi = img.frame_info();
    let long_edge = fi.width.max(fi.height) as u32;
    if long_edge <= max_long_edge {
        return None; // No resize needed
    }

    use std::process::Command;
    let pid = std::process::id();
    let dir = std::env::temp_dir();
    let input = dir.join(format!("phasm_rsz_in_{pid}.jpg"));
    let resized_jpg = dir.join(format!("phasm_rsz_tmp_{pid}.jpg"));
    let ppm_out = dir.join(format!("phasm_rsz_{pid}.ppm"));

    std::fs::write(&input, stego_bytes).ok()?;

    // Step 1: sips resize to temp JPEG (high quality to minimize loss)
    let sips_out = Command::new("/usr/bin/sips")
        .arg("-Z").arg(max_long_edge.to_string())
        .arg("-s").arg("format").arg("jpeg")
        .arg("-s").arg("formatOptions").arg("98")
        .arg(&input)
        .arg("--out").arg(&resized_jpg)
        .output().ok()?;

    let _ = std::fs::remove_file(&input);

    if !sips_out.status.success() {
        let _ = std::fs::remove_file(&resized_jpg);
        return None;
    }

    // Step 2: djpeg to PPM (lossless decompression for cjpeg input)
    let djpeg_out = Command::new("/opt/homebrew/bin/djpeg")
        .arg("-ppm")
        .arg("-outfile").arg(&ppm_out)
        .arg(&resized_jpg)
        .output().ok()?;

    let _ = std::fs::remove_file(&resized_jpg);

    if !djpeg_out.status.success() {
        let _ = std::fs::remove_file(&ppm_out);
        return None;
    }

    Some(ppm_out)
}

/// Platform-realistic recompression: optional resize + encoder-specific recompression.
/// Simulates: decode → resize (if long edge > max_dim) → encode at target QF.
///
/// For libjpeg-turbo/MozJPEG: resize to BMP (lossless), then cjpeg the BMP.
/// For AppleJPEG: sips handles resize + recompress in one step.
/// For Internal: resize via sips to temp JPEG, then pixel-domain sim.
fn platform_recompress(
    stego_bytes: &[u8],
    qf: u8,
    max_long_edge: u32,
    encoder: &str,
) -> Option<Vec<u8>> {
    use std::process::Command;
    let pid = std::process::id();
    let dir = std::env::temp_dir();

    // Check if resize needed
    let img = JpegImage::from_bytes(stego_bytes).ok()?;
    let fi = img.frame_info();
    let long_edge = fi.width.max(fi.height) as u32;
    let needs_resize = long_edge > max_long_edge;

    match encoder {
        "AppleJPEG" => {
            // sips handles resize + recompress in one step
            let input = dir.join(format!("phasm_v2_sips_in_{pid}.jpg"));
            let output = dir.join(format!("phasm_v2_sips_out_{pid}.jpg"));
            std::fs::write(&input, stego_bytes).ok()?;

            let mut cmd = Command::new("/usr/bin/sips");
            if needs_resize {
                cmd.arg("-Z").arg(max_long_edge.to_string());
            }
            cmd.arg("-s").arg("format").arg("jpeg")
                .arg("-s").arg("formatOptions").arg(qf.to_string())
                .arg(&input)
                .arg("--out").arg(&output);

            let out = cmd.output().ok()?;
            let _ = std::fs::remove_file(&input);
            if !out.status.success() {
                let _ = std::fs::remove_file(&output);
                return None;
            }
            let result = std::fs::read(&output).ok();
            let _ = std::fs::remove_file(&output);
            result
        }
        "Internal" => {
            if needs_resize {
                // Resize via sips to temp JPEG at high quality, then pixel-domain sim
                let input = dir.join(format!("phasm_v2_int_in_{pid}.jpg"));
                let resized = dir.join(format!("phasm_v2_int_rsz_{pid}.jpg"));
                std::fs::write(&input, stego_bytes).ok()?;

                let out = Command::new("/usr/bin/sips")
                    .arg("-Z").arg(max_long_edge.to_string())
                    .arg("-s").arg("format").arg("jpeg")
                    .arg("-s").arg("formatOptions").arg("95")
                    .arg(&input)
                    .arg("--out").arg(&resized)
                    .output().ok()?;
                let _ = std::fs::remove_file(&input);
                if !out.status.success() {
                    let _ = std::fs::remove_file(&resized);
                    return None;
                }
                let resized_bytes = std::fs::read(&resized).ok()?;
                let _ = std::fs::remove_file(&resized);
                simulate_pixel_recompression(&resized_bytes, qf)
            } else {
                simulate_pixel_recompression(stego_bytes, qf)
            }
        }
        "libjpeg-turbo" | "MozJPEG" => {
            let cjpeg_path = if encoder == "MozJPEG" {
                "/opt/homebrew/opt/mozjpeg/bin/cjpeg"
            } else {
                "/opt/homebrew/bin/cjpeg"
            };

            if needs_resize {
                // Resize to PPM (lossless), then compress PPM with cjpeg
                let ppm_path = resize_to_ppm(stego_bytes, max_long_edge)?;
                let output = dir.join(format!("phasm_v2_{}_out_{pid}.jpg",
                    if encoder == "MozJPEG" { "moz" } else { "ljt" }));

                let out = Command::new(cjpeg_path)
                    .arg("-quality").arg(qf.to_string())
                    .arg("-baseline")
                    .arg("-outfile").arg(&output)
                    .arg(&ppm_path)
                    .output().ok()?;
                let _ = std::fs::remove_file(&ppm_path);
                if !out.status.success() {
                    let _ = std::fs::remove_file(&output);
                    return None;
                }
                let result = std::fs::read(&output).ok();
                let _ = std::fs::remove_file(&output);
                result
            } else {
                // No resize — standard djpeg → cjpeg pipeline
                let input = dir.join(format!("phasm_v2_{}_in_{pid}.jpg",
                    if encoder == "MozJPEG" { "moz" } else { "ljt" }));
                let ppm = dir.join(format!("phasm_v2_{}_ppm_{pid}.ppm",
                    if encoder == "MozJPEG" { "moz" } else { "ljt" }));
                let output = dir.join(format!("phasm_v2_{}_out_{pid}.jpg",
                    if encoder == "MozJPEG" { "moz" } else { "ljt" }));

                std::fs::write(&input, stego_bytes).ok()?;

                let djpeg_out = Command::new("/opt/homebrew/bin/djpeg")
                    .arg("-ppm")
                    .arg("-outfile").arg(&ppm)
                    .arg(&input)
                    .output().ok()?;
                let _ = std::fs::remove_file(&input);
                if !djpeg_out.status.success() {
                    return None;
                }

                let cjpeg_out = Command::new(cjpeg_path)
                    .arg("-quality").arg(qf.to_string())
                    .arg("-baseline")
                    .arg("-outfile").arg(&output)
                    .arg(&ppm)
                    .output().ok()?;
                let _ = std::fs::remove_file(&ppm);
                if !cjpeg_out.status.success() {
                    let _ = std::fs::remove_file(&output);
                    return None;
                }
                let result = std::fs::read(&output).ok();
                let _ = std::fs::remove_file(&output);
                result
            }
        }
        _ => None,
    }
}

/// Real encoder experiment V2: with 1600px long-edge resize.
///
/// Simulates realistic platform processing: resize to ≤1600px long edge,
/// then recompress with each encoder family at various quality factors.
///
/// Run with: cargo test -p phasm-core -- --ignored armor_real_encoder_v2 --nocapture
#[test]
#[ignore]
fn armor_real_encoder_v2() {
    let max_long_edge = 1600u32;

    let ljt_available = std::process::Command::new("/opt/homebrew/bin/cjpeg")
        .arg("-version").output().map(|o| o.status.success() || !o.stderr.is_empty()).unwrap_or(false);
    let moz_available = std::process::Command::new("/opt/homebrew/opt/mozjpeg/bin/cjpeg")
        .arg("-version").output().map(|o| o.status.success() || !o.stderr.is_empty()).unwrap_or(false);
    let sips_available = std::process::Command::new("/usr/bin/sips")
        .arg("--help").output().map(|o| o.status.success()).unwrap_or(false);

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  REAL ENCODER EXPERIMENT V2 — WITH 1600px RESIZE                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║  Resize: long edge > {max_long_edge}px → downscale to {max_long_edge}px (via sips)             ║");
    println!("║  Encoders:                                                           ║");
    println!("║  • libjpeg-turbo: {} (Twitter, Discord, Telegram Android)        ║",
        if ljt_available { "AVAILABLE" } else { "MISSING  " });
    println!("║  • MozJPEG:       {} (Facebook, Instagram)                       ║",
        if moz_available { "AVAILABLE" } else { "MISSING  " });
    println!("║  • AppleJPEG:     {} (iMessage, WhatsApp iOS, Telegram iOS)      ║",
        if sips_available { "AVAILABLE" } else { "MISSING  " });
    println!("║  • Internal sim:  AVAILABLE (pixel-domain baseline)                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");

    let encoder_names: Vec<&str> = {
        let mut v = vec!["Internal"];
        if ljt_available { v.push("libjpeg-turbo"); }
        if moz_available { v.push("MozJPEG"); }
        if sips_available { v.push("AppleJPEG"); }
        v
    };

    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let qfs: &[u8] = &[95, 85, 80, 75, 70, 53];
    let msg_lengths: &[usize] = &[10, 20, 50, 100, 200, 500];
    let passphrase = "real-encoder-v2-resize-2026";

    // Report dimensions + resize info
    println!();
    println!("IMAGE DIMENSIONS");
    println!("{:<16} {:>10} {:>10} {:>10} {:>10}", "Image", "Original", "Resized", "Cap(orig)", "Cap(rsz)");
    println!("{}", "-".repeat(60));
    for (img_name, cover_bytes) in &test_images {
        if let Ok(img) = JpegImage::from_bytes(cover_bytes) {
            let fi = img.frame_info();
            let orig_dims = format!("{}x{}", fi.width, fi.height);
            let cap_orig = armor_capacity(&img).unwrap_or(0);

            let long_edge = fi.width.max(fi.height) as u32;
            if long_edge > max_long_edge {
                // Compute resized capacity
                let pid = std::process::id();
                let dir = std::env::temp_dir();
                let tmp_in = dir.join(format!("phasm_cap_{pid}.jpg"));
                let tmp_out = dir.join(format!("phasm_cap_rsz_{pid}.jpg"));
                std::fs::write(&tmp_in, cover_bytes).unwrap();
                let _ = std::process::Command::new("/usr/bin/sips")
                    .arg("-Z").arg(max_long_edge.to_string())
                    .arg("-s").arg("format").arg("jpeg")
                    .arg("-s").arg("formatOptions").arg("95")
                    .arg(&tmp_in).arg("--out").arg(&tmp_out)
                    .output();
                let _ = std::fs::remove_file(&tmp_in);
                let rsz_bytes = std::fs::read(&tmp_out).unwrap_or_default();
                let _ = std::fs::remove_file(&tmp_out);
                if let Ok(rsz_img) = JpegImage::from_bytes(&rsz_bytes) {
                    let rsz_fi = rsz_img.frame_info();
                    let rsz_dims = format!("{}x{}", rsz_fi.width, rsz_fi.height);
                    let cap_rsz = armor_capacity(&rsz_img).unwrap_or(0);
                    println!("{:<16} {:>10} {:>10} {:>8}B {:>8}B",
                        img_name, orig_dims, rsz_dims, cap_orig, cap_rsz);
                }
            } else {
                println!("{:<16} {:>10} {:>10} {:>8}B {:>8}",
                    img_name, orig_dims, "(no rsz)", cap_orig, "-");
            }
        }
    }

    // Stats tracking
    struct Stats { name: String, total: u32, survived: u32 }
    struct QfS { enc: String, qf: u8, total: u32, survived: u32 }
    struct ImgS { enc: String, img: String, total: u32, survived: u32 }
    let mut enc_stats: Vec<Stats> = Vec::new();
    let mut qf_stats: Vec<QfS> = Vec::new();
    let mut img_stats: Vec<ImgS> = Vec::new();

    fn bump(stats: &mut Vec<QfS>, enc: &str, qf: u8, ok: bool) {
        if let Some(s) = stats.iter_mut().find(|s| s.enc == enc && s.qf == qf) {
            s.total += 1; if ok { s.survived += 1; }
        } else {
            stats.push(QfS { enc: enc.to_string(), qf, total: 1, survived: if ok { 1 } else { 0 } });
        }
    }
    fn bump_img(stats: &mut Vec<ImgS>, enc: &str, img: &str, ok: bool) {
        if let Some(s) = stats.iter_mut().find(|s| s.enc == enc && s.img == img) {
            s.total += 1; if ok { s.survived += 1; }
        } else {
            stats.push(ImgS { enc: enc.to_string(), img: img.to_string(), total: 1, survived: if ok { 1 } else { 0 } });
        }
    }

    // Full matrix
    println!();
    println!("FULL EXPERIMENT MATRIX (with {max_long_edge}px resize)");
    println!("{:<14} {:<16} {:>4} {:>3}  {:>8} {:>6} {:>6} {:>4} {:>6}",
        "Encoder", "Image", "Msg", "QF", "Result", "Integ", "RSErr", "r", "Parity");
    println!("{}", "-".repeat(88));

    for &enc_name in &encoder_names {
        let mut et = 0u32;
        let mut es = 0u32;

        for (img_name, cover_bytes) in &test_images {
            let img = match JpegImage::from_bytes(cover_bytes) {
                Ok(img) => img,
                Err(_) => continue,
            };
            let cap = armor_capacity(&img).unwrap_or(0);

            for &msg_len in msg_lengths {
                if msg_len > cap { continue; }

                let message = generate_message(msg_len);
                let stego = match armor_encode(cover_bytes, &message, passphrase) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                for &qf in qfs {
                    et += 1;
                    let recompressed = match platform_recompress(&stego, qf, max_long_edge, enc_name) {
                        Some(r) => r,
                        None => {
                            println!("{:<14} {:<16} {:>4} {:>3}  {:>8} {:>6} {:>6} {:>4} {:>6}",
                                enc_name, img_name, msg_len, qf, "ENC_ERR", "-", "-", "-", "-");
                            bump(&mut qf_stats, enc_name, qf, false);
                            bump_img(&mut img_stats, enc_name, img_name, false);
                            continue;
                        }
                    };

                    match armor_decode(&recompressed, passphrase) {
                        Ok((decoded, quality)) => {
                            let ok = decoded == message;
                            if ok { es += 1; }
                            bump(&mut qf_stats, enc_name, qf, ok);
                            bump_img(&mut img_stats, enc_name, img_name, ok);
                            println!("{:<14} {:<16} {:>4} {:>3}  {:>8} {:>6} {:>6} {:>4} {:>6}",
                                enc_name, img_name, msg_len, qf,
                                if ok { "YES" } else { "CORRUPT" },
                                format!("{}%", quality.integrity_percent),
                                format!("{}", quality.rs_errors_corrected),
                                format!("{}", quality.repetition_factor),
                                format!("{}", quality.parity_len));
                        }
                        Err(_) => {
                            bump(&mut qf_stats, enc_name, qf, false);
                            bump_img(&mut img_stats, enc_name, img_name, false);
                            println!("{:<14} {:<16} {:>4} {:>3}  {:>8} {:>6} {:>6} {:>4} {:>6}",
                                enc_name, img_name, msg_len, qf, "FAIL", "-", "-", "-", "-");
                        }
                    }
                }
            }
        }

        enc_stats.push(Stats { name: enc_name.to_string(), total: et, survived: es });
    }

    // Summaries
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("SUMMARY — Overall by Encoder (with {max_long_edge}px resize)");
    println!("{:<16} {:>8} {:>8} {:>8}", "Encoder", "Survived", "Total", "Rate");
    println!("{}", "-".repeat(44));
    for s in &enc_stats {
        let rate = if s.total > 0 { s.survived as f64 / s.total as f64 * 100.0 } else { 0.0 };
        println!("{:<16} {:>8} {:>8} {:>7.1}%", s.name, s.survived, s.total, rate);
    }

    println!();
    println!("SUMMARY — By Encoder × Quality Factor");
    print!("{:>4}", "QF");
    for s in &enc_stats { print!(" {:>16}", s.name); }
    println!();
    println!("{}", "-".repeat(4 + enc_stats.len() * 17));
    for &qf in qfs {
        print!("{:>4}", qf);
        for s in &enc_stats {
            if let Some(q) = qf_stats.iter().find(|q| q.enc == s.name && q.qf == qf) {
                let rate = if q.total > 0 { q.survived as f64 / q.total as f64 * 100.0 } else { 0.0 };
                print!(" {:>5}/{:<4} ({:>4.0}%)", q.survived, q.total, rate);
            } else {
                print!(" {:>16}", "-");
            }
        }
        println!();
    }

    println!();
    println!("SUMMARY — By Encoder × Image");
    print!("{:<16}", "Image");
    for s in &enc_stats { print!(" {:>16}", s.name); }
    println!();
    println!("{}", "-".repeat(16 + enc_stats.len() * 17));
    for (img_name, _) in &test_images {
        print!("{:<16}", img_name);
        for s in &enc_stats {
            if let Some(i) = img_stats.iter().find(|i| i.enc == s.name && i.img == *img_name) {
                let rate = if i.total > 0 { i.survived as f64 / i.total as f64 * 100.0 } else { 0.0 };
                print!(" {:>5}/{:<4} ({:>4.0}%)", i.survived, i.total, rate);
            } else {
                print!(" {:>16}", "-");
            }
        }
        println!();
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
}
