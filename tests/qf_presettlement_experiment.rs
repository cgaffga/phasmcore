//! QF Pre-Settlement Experiment
//!
//! Tests whether pre-compressing a cover image at QF ~70 before Armor encoding
//! improves robustness against subsequent recompression (e.g., by WhatsApp).
//!
//! Hypothesis: If we pre-compress the cover at QF ~70 (below WhatsApp's QF ~75),
//! the DCT coefficients are already "settled" on the coarse quantization grid,
//! so recompression at QF >= 70 causes minimal further change to the embedded data.
//!
//! Run all experiments:
//!   cargo test -p phasm-core --test qf_presettlement_experiment -- --ignored --nocapture
//!
//! Run in release mode (faster):
//!   cargo test -p phasm-core --release --test qf_presettlement_experiment -- --ignored --nocapture

use phasm_core::{armor_encode, armor_capacity, smart_decode, JpegImage};
use std::process::Command;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_test_vector(name: &str) -> Vec<u8> {
    std::fs::read(format!("../test-vectors/{name}")).unwrap()
}

fn load_real_photo(name: &str) -> Vec<u8> {
    std::fs::read(format!("tests/real_photos/{name}")).unwrap()
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

/// Pre-compress (settle) an image at a given quality factor using sips.
/// Returns the settled JPEG bytes, or None on failure.
fn presettle_with_sips(jpeg_bytes: &[u8], qf: u8) -> Option<Vec<u8>> {
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir();
    let input = dir.join(format!("phasm_presettle_in_{pid}_{ts}.jpg"));
    let output = dir.join(format!("phasm_presettle_out_{pid}_{ts}.jpg"));

    std::fs::write(&input, jpeg_bytes).ok()?;

    let out = Command::new("/usr/bin/sips")
        .arg("-s").arg("format").arg("jpeg")
        .arg("-s").arg("formatOptions").arg(qf.to_string())
        .arg(&input)
        .arg("--out").arg(&output)
        .output().ok()?;

    let _ = std::fs::remove_file(&input);

    if !out.status.success() {
        let _ = std::fs::remove_file(&output);
        return None;
    }

    let result = std::fs::read(&output).ok();
    let _ = std::fs::remove_file(&output);
    result
}

/// Recompress a JPEG at a given quality factor using sips (AppleJPEG).
/// Returns the recompressed bytes, or None on failure.
fn recompress_with_sips(jpeg_bytes: &[u8], qf: u8) -> Option<Vec<u8>> {
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir();
    let input = dir.join(format!("phasm_recomp_in_{pid}_{ts}.jpg"));
    let output = dir.join(format!("phasm_recomp_out_{pid}_{ts}.jpg"));

    std::fs::write(&input, jpeg_bytes).ok()?;

    let out = Command::new("/usr/bin/sips")
        .arg("-s").arg("format").arg("jpeg")
        .arg("-s").arg("formatOptions").arg(qf.to_string())
        .arg(&input)
        .arg("--out").arg(&output)
        .output().ok()?;

    let _ = std::fs::remove_file(&input);

    if !out.status.success() {
        let _ = std::fs::remove_file(&output);
        return None;
    }

    let result = std::fs::read(&output).ok();
    let _ = std::fs::remove_file(&output);
    result
}

// ---------------------------------------------------------------------------
// EXPERIMENT 1: QF Pre-Settlement Main Matrix
// ---------------------------------------------------------------------------

/// Tests Armor encode/decode survival when the cover image is pre-settled at
/// various quality factors before encoding, then recompressed at QF 75
/// (simulating WhatsApp).
///
/// For each pre-settle QF in [70, 75, 80, 85, 92, 98], plus "None" (no pre-settle):
///   1. Pre-compress cover at that QF using sips
///   2. Armor-encode a message into the pre-settled cover
///   3. Recompress the stego image at QF 75 using sips
///   4. Attempt to decode
///
/// Run: cargo test -p phasm-core --release --test qf_presettlement_experiment -- --ignored presettlement_main_matrix --nocapture
#[test]
#[ignore]
fn presettlement_main_matrix() {
    // Verify sips is available
    let sips_ok = Command::new("/usr/bin/sips")
        .arg("--help")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !sips_ok {
        println!("\nSKIPPED: sips not available (macOS only)");
        return;
    }

    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("photo_640x480", load_test_vector("photo_640x480_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let presettle_qfs: &[Option<u8>] = &[
        None,           // No pre-settlement (baseline)
        Some(98),
        Some(92),
        Some(85),
        Some(80),
        Some(75),
        Some(70),
    ];

    let recomp_qf: u8 = 75; // WhatsApp-like
    let passphrase = "qf-presettlement-experiment-2026";

    let messages: Vec<&str> = vec![
        "Hello",                    // 5 bytes
        "Meeting at noon",          // 15 bytes
        "The package is ready",     // 20 bytes
    ];

    println!();
    println!("================================================================");
    println!("  QF PRE-SETTLEMENT EXPERIMENT — Main Matrix");
    println!("================================================================");
    println!("Hypothesis: Pre-compressing the cover at QF <= target recomp QF");
    println!("            settles coefficients, improving Armor robustness.");
    println!();
    println!("Recompression QF: {} (sips/AppleJPEG)", recomp_qf);
    println!("================================================================");

    // Print capacity report first
    println!();
    println!("CAPACITY REPORT");
    println!("{:<16} {:>8} {:>10} {:>10} {:>8}",
        "Image", "PreQF", "Dimensions", "FileSize", "Capacity");
    println!("{}", "-".repeat(58));

    for (img_name, cover_bytes) in &test_images {
        for &presettle_qf in presettle_qfs {
            let (settled_bytes, qf_label) = match presettle_qf {
                None => (cover_bytes.clone(), "None".to_string()),
                Some(qf) => {
                    match presettle_with_sips(cover_bytes, qf) {
                        Some(b) => (b, format!("{}", qf)),
                        None => {
                            println!("{:<16} {:>8} {:>10} {:>10} {:>8}",
                                img_name, qf, "SIPS_ERR", "-", "-");
                            continue;
                        }
                    }
                }
            };

            if let Ok(img) = JpegImage::from_bytes(&settled_bytes) {
                let fi = img.frame_info();
                let cap = armor_capacity(&img).unwrap_or(0);
                println!("{:<16} {:>8} {:>4}x{:<5} {:>8} {:>6}B",
                    img_name, qf_label,
                    fi.width, fi.height,
                    format!("{}K", settled_bytes.len() / 1024),
                    cap);
            }
        }
    }

    // Main experiment matrix
    println!();
    println!("RESULTS MATRIX");
    println!("{:<16} {:>6} {:>4} {:>8} {:>8} {:>8} {:>6} {:>6} {:>4} {:>6}",
        "Image", "PreQF", "Msg", "StegoSz", "RecompSz", "Result", "Integ", "RSErr", "r", "Parity");
    println!("{}", "-".repeat(92));

    // Track summary stats per presettle QF
    struct QfStats {
        qf_label: String,
        total: u32,
        survived: u32,
    }
    let mut qf_stats: Vec<QfStats> = Vec::new();

    for (img_name, cover_bytes) in &test_images {
        for &presettle_qf in presettle_qfs {
            let qf_label = match presettle_qf {
                None => "None".to_string(),
                Some(qf) => format!("{}", qf),
            };

            let settled_bytes = match presettle_qf {
                None => cover_bytes.clone(),
                Some(qf) => {
                    match presettle_with_sips(cover_bytes, qf) {
                        Some(b) => b,
                        None => continue,
                    }
                }
            };

            // Check capacity
            let cap = match JpegImage::from_bytes(&settled_bytes) {
                Ok(img) => armor_capacity(&img).unwrap_or(0),
                Err(_) => continue,
            };

            for &message in &messages {
                let msg_len = message.len();
                if msg_len > cap {
                    println!("{:<16} {:>6} {:>4} {:>8} {:>8} {:>8} {:>6} {:>6} {:>4} {:>6}",
                        img_name, &qf_label, msg_len, "-", "-", "NO_CAP", "-", "-", "-", "-");
                    // Still count in stats
                    let stat = qf_stats.iter_mut().find(|s| s.qf_label == qf_label);
                    if let Some(s) = stat { s.total += 1; }
                    else { qf_stats.push(QfStats { qf_label: qf_label.clone(), total: 1, survived: 0 }); }
                    continue;
                }

                // Encode
                let stego = match armor_encode(&settled_bytes, message, passphrase) {
                    Ok(s) => s,
                    Err(e) => {
                        println!("{:<16} {:>6} {:>4} {:>8} {:>8} {:>8} {:>6} {:>6} {:>4} {:>6}",
                            img_name, &qf_label, msg_len, "-", "-",
                            &format!("ENC:{}", &format!("{e}")[..6.min(format!("{e}").len())]),
                            "-", "-", "-", "-");
                        let stat = qf_stats.iter_mut().find(|s| s.qf_label == qf_label);
                        if let Some(s) = stat { s.total += 1; }
                        else { qf_stats.push(QfStats { qf_label: qf_label.clone(), total: 1, survived: 0 }); }
                        continue;
                    }
                };
                let stego_size = stego.len();

                // Recompress
                let recompressed = match recompress_with_sips(&stego, recomp_qf) {
                    Some(r) => r,
                    None => {
                        println!("{:<16} {:>6} {:>4} {:>7}K {:>8} {:>8} {:>6} {:>6} {:>4} {:>6}",
                            img_name, &qf_label, msg_len,
                            stego_size / 1024, "-", "REC_ERR", "-", "-", "-", "-");
                        let stat = qf_stats.iter_mut().find(|s| s.qf_label == qf_label);
                        if let Some(s) = stat { s.total += 1; }
                        else { qf_stats.push(QfStats { qf_label: qf_label.clone(), total: 1, survived: 0 }); }
                        continue;
                    }
                };
                let recomp_size = recompressed.len();

                let stat = qf_stats.iter_mut().find(|s| s.qf_label == qf_label);
                if stat.is_none() {
                    qf_stats.push(QfStats { qf_label: qf_label.clone(), total: 0, survived: 0 });
                }
                let stat = qf_stats.iter_mut().find(|s| s.qf_label == qf_label).unwrap();
                stat.total += 1;

                // Decode
                match smart_decode(&recompressed, passphrase) {
                    Ok((decoded, quality)) => {
                        let ok = decoded == message;
                        if ok { stat.survived += 1; }
                        println!("{:<16} {:>6} {:>4} {:>7}K {:>7}K {:>8} {:>5}% {:>6} {:>4} {:>6}",
                            img_name, &qf_label, msg_len,
                            stego_size / 1024, recomp_size / 1024,
                            if ok { "YES" } else { "CORRUPT" },
                            quality.integrity_percent,
                            quality.rs_errors_corrected,
                            quality.repetition_factor,
                            quality.parity_len);
                    }
                    Err(_) => {
                        println!("{:<16} {:>6} {:>4} {:>7}K {:>7}K {:>8} {:>6} {:>6} {:>4} {:>6}",
                            img_name, &qf_label, msg_len,
                            stego_size / 1024, recomp_size / 1024,
                            "FAIL", "-", "-", "-", "-");
                    }
                }
            }
        }
        println!();
    }

    // Summary
    println!("================================================================");
    println!("SUMMARY — Survival Rate by Pre-Settle QF (recomp at QF {})", recomp_qf);
    println!("{:>8} {:>8} {:>8} {:>8}", "PreQF", "Survived", "Total", "Rate");
    println!("{}", "-".repeat(36));
    for stat in &qf_stats {
        let rate = if stat.total > 0 { stat.survived as f64 / stat.total as f64 * 100.0 } else { 0.0 };
        println!("{:>8} {:>8} {:>8} {:>7.1}%", stat.qf_label, stat.survived, stat.total, rate);
    }
    println!("================================================================");
}

// ---------------------------------------------------------------------------
// EXPERIMENT 2: Best Pre-Settle QF Against Multiple Recompression QFs
// ---------------------------------------------------------------------------

/// For the best-performing pre-settle QF(s) from Experiment 1, tests survival
/// against recompression at multiple platform QFs:
///   - QF 71 (Facebook-like)
///   - QF 75 (WhatsApp-like)
///   - QF 80 (WhatsApp HD / Discord)
///   - QF 85 (Twitter/X / Signal)
///
/// Also tests the "no pre-settle" baseline for comparison.
///
/// Run: cargo test -p phasm-core --release --test qf_presettlement_experiment -- --ignored presettlement_multi_platform --nocapture
#[test]
#[ignore]
fn presettlement_multi_platform() {
    let sips_ok = Command::new("/usr/bin/sips")
        .arg("--help")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !sips_ok {
        println!("\nSKIPPED: sips not available (macOS only)");
        return;
    }

    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_320x240", load_test_vector("photo_320x240_q75_420.jpg")),
        ("photo_640x480", load_test_vector("photo_640x480_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    // Pre-settle at the most promising QFs + baseline
    let presettle_qfs: &[Option<u8>] = &[
        None,
        Some(70),
        Some(75),
        Some(80),
    ];

    // Platform recompression QFs
    let recomp_qfs: &[(u8, &str)] = &[
        (71, "Facebook"),
        (75, "WhatsApp"),
        (80, "WhatsApp HD"),
        (85, "Twitter/X"),
    ];

    let passphrase = "presettlement-multiplatform-2026";
    let messages: Vec<&str> = vec![
        "Hello",
        "Meeting at noon",
        "The package is ready",
    ];

    println!();
    println!("================================================================");
    println!("  QF PRE-SETTLEMENT EXPERIMENT — Multi-Platform");
    println!("================================================================");
    println!("Tests best pre-settle QFs against multiple platform recomp QFs.");
    println!("================================================================");
    println!();

    // Track stats per (presettle_qf, recomp_qf)
    struct CellStats {
        pre_qf: String,
        rec_qf: u8,
        total: u32,
        survived: u32,
    }
    let mut cell_stats: Vec<CellStats> = Vec::new();

    println!("{:<16} {:>6} {:>4} {:>3} {:>10} {:>8} {:>6} {:>6} {:>4} {:>6}",
        "Image", "PreQF", "Msg", "RQF", "Platform", "Result", "Integ", "RSErr", "r", "Parity");
    println!("{}", "-".repeat(96));

    for (img_name, cover_bytes) in &test_images {
        for &presettle_qf in presettle_qfs {
            let qf_label = match presettle_qf {
                None => "None".to_string(),
                Some(qf) => format!("{}", qf),
            };

            let settled_bytes = match presettle_qf {
                None => cover_bytes.clone(),
                Some(qf) => {
                    match presettle_with_sips(cover_bytes, qf) {
                        Some(b) => b,
                        None => continue,
                    }
                }
            };

            let cap = match JpegImage::from_bytes(&settled_bytes) {
                Ok(img) => armor_capacity(&img).unwrap_or(0),
                Err(_) => continue,
            };

            for &message in &messages {
                let msg_len = message.len();
                if msg_len > cap { continue; }

                let stego = match armor_encode(&settled_bytes, message, passphrase) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                for &(rec_qf, platform) in recomp_qfs {
                    let recompressed = match recompress_with_sips(&stego, rec_qf) {
                        Some(r) => r,
                        None => {
                            println!("{:<16} {:>6} {:>4} {:>3} {:>10} {:>8} {:>6} {:>6} {:>4} {:>6}",
                                img_name, &qf_label, msg_len, rec_qf, platform,
                                "REC_ERR", "-", "-", "-", "-");
                            continue;
                        }
                    };

                    // Track stats
                    let cell = cell_stats.iter_mut()
                        .find(|c| c.pre_qf == qf_label && c.rec_qf == rec_qf);
                    if cell.is_none() {
                        cell_stats.push(CellStats {
                            pre_qf: qf_label.clone(),
                            rec_qf,
                            total: 0,
                            survived: 0,
                        });
                    }
                    let cell = cell_stats.iter_mut()
                        .find(|c| c.pre_qf == qf_label && c.rec_qf == rec_qf).unwrap();
                    cell.total += 1;

                    match smart_decode(&recompressed, passphrase) {
                        Ok((decoded, quality)) => {
                            let ok = decoded == message;
                            if ok { cell.survived += 1; }
                            println!("{:<16} {:>6} {:>4} {:>3} {:>10} {:>8} {:>5}% {:>6} {:>4} {:>6}",
                                img_name, &qf_label, msg_len, rec_qf, platform,
                                if ok { "YES" } else { "CORRUPT" },
                                quality.integrity_percent,
                                quality.rs_errors_corrected,
                                quality.repetition_factor,
                                quality.parity_len);
                        }
                        Err(_) => {
                            println!("{:<16} {:>6} {:>4} {:>3} {:>10} {:>8} {:>6} {:>6} {:>4} {:>6}",
                                img_name, &qf_label, msg_len, rec_qf, platform,
                                "FAIL", "-", "-", "-", "-");
                        }
                    }
                }
            }
        }
        println!();
    }

    // Summary: survival rate heatmap (pre-settle QF x recomp QF)
    println!("================================================================");
    println!("SUMMARY — Survival Heatmap (Pre-Settle QF x Recomp QF)");
    println!();
    print!("{:>8}", "PreQF");
    for &(rec_qf, platform) in recomp_qfs {
        print!(" {:>14}", format!("QF{} {}", rec_qf, platform));
    }
    println!();
    println!("{}", "-".repeat(8 + recomp_qfs.len() * 15));

    for presettle_qf in presettle_qfs {
        let qf_label = match presettle_qf {
            None => "None".to_string(),
            Some(qf) => format!("{}", qf),
        };
        print!("{:>8}", qf_label);
        for &(rec_qf, _) in recomp_qfs {
            if let Some(cell) = cell_stats.iter().find(|c| c.pre_qf == qf_label && c.rec_qf == rec_qf) {
                let rate = if cell.total > 0 { cell.survived as f64 / cell.total as f64 * 100.0 } else { 0.0 };
                print!(" {:>4}/{:<4}({:>3.0}%)", cell.survived, cell.total, rate);
            } else {
                print!(" {:>14}", "-");
            }
        }
        println!();
    }
    println!("================================================================");
}

// ---------------------------------------------------------------------------
// EXPERIMENT 3: Double Pre-Settlement (settle twice at same QF)
// ---------------------------------------------------------------------------

/// Tests whether settling TWICE at the same QF provides additional benefit.
/// Theory: A single round-trip through sips may not fully converge the
/// coefficients. A second pass might bring them closer to the fixed point.
///
/// Run: cargo test -p phasm-core --release --test qf_presettlement_experiment -- --ignored presettlement_double_settle --nocapture
#[test]
#[ignore]
fn presettlement_double_settle() {
    let sips_ok = Command::new("/usr/bin/sips")
        .arg("--help")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !sips_ok {
        println!("\nSKIPPED: sips not available (macOS only)");
        return;
    }

    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_640x480", load_test_vector("photo_640x480_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let settle_qf: u8 = 70;
    let recomp_qf: u8 = 75;
    let passphrase = "double-settle-2026";
    let messages: Vec<&str> = vec!["Hello", "Meeting at noon", "The package is ready"];

    println!();
    println!("================================================================");
    println!("  QF PRE-SETTLEMENT EXPERIMENT — Double Settlement");
    println!("================================================================");
    println!("Settle QF: {}   Recomp QF: {} (sips)", settle_qf, recomp_qf);
    println!("Tests: None, 1x settle, 2x settle, 3x settle");
    println!("================================================================");
    println!();

    struct Stats {
        label: String,
        total: u32,
        survived: u32,
    }
    let mut stats: Vec<Stats> = Vec::new();

    let settle_passes: &[(u8, &str)] = &[
        (0, "None"),
        (1, "1x"),
        (2, "2x"),
        (3, "3x"),
    ];

    println!("{:<16} {:>8} {:>4} {:>8} {:>8} {:>6} {:>6} {:>4} {:>6}",
        "Image", "Settle", "Msg", "StegoSz", "Result", "Integ", "RSErr", "r", "Parity");
    println!("{}", "-".repeat(82));

    for (img_name, cover_bytes) in &test_images {
        for &(passes, label) in settle_passes {
            // Apply N rounds of settlement
            let mut settled = cover_bytes.clone();
            let mut settle_ok = true;
            for _ in 0..passes {
                match presettle_with_sips(&settled, settle_qf) {
                    Some(b) => settled = b,
                    None => { settle_ok = false; break; }
                }
            }
            if !settle_ok { continue; }

            let cap = match JpegImage::from_bytes(&settled) {
                Ok(img) => armor_capacity(&img).unwrap_or(0),
                Err(_) => continue,
            };

            for &message in &messages {
                let msg_len = message.len();
                if msg_len > cap { continue; }

                let stego = match armor_encode(&settled, message, passphrase) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let stego_size = stego.len();

                let recompressed = match recompress_with_sips(&stego, recomp_qf) {
                    Some(r) => r,
                    None => continue,
                };

                let stat = stats.iter_mut().find(|s| s.label == label);
                if stat.is_none() {
                    stats.push(Stats { label: label.to_string(), total: 0, survived: 0 });
                }
                let stat = stats.iter_mut().find(|s| s.label == label).unwrap();
                stat.total += 1;

                match smart_decode(&recompressed, passphrase) {
                    Ok((decoded, quality)) => {
                        let ok = decoded == message;
                        if ok { stat.survived += 1; }
                        println!("{:<16} {:>8} {:>4} {:>7}K {:>8} {:>5}% {:>6} {:>4} {:>6}",
                            img_name, label, msg_len,
                            stego_size / 1024,
                            if ok { "YES" } else { "CORRUPT" },
                            quality.integrity_percent,
                            quality.rs_errors_corrected,
                            quality.repetition_factor,
                            quality.parity_len);
                    }
                    Err(_) => {
                        println!("{:<16} {:>8} {:>4} {:>7}K {:>8} {:>6} {:>6} {:>4} {:>6}",
                            img_name, label, msg_len,
                            stego_size / 1024,
                            "FAIL", "-", "-", "-", "-");
                    }
                }
            }
        }
        println!();
    }

    println!("================================================================");
    println!("SUMMARY — Double Settlement");
    println!("{:>8} {:>8} {:>8} {:>8}", "Settle", "Survived", "Total", "Rate");
    println!("{}", "-".repeat(36));
    for stat in &stats {
        let rate = if stat.total > 0 { stat.survived as f64 / stat.total as f64 * 100.0 } else { 0.0 };
        println!("{:>8} {:>8} {:>8} {:>7.1}%", stat.label, stat.survived, stat.total, rate);
    }
    println!("================================================================");
}

// ---------------------------------------------------------------------------
// EXPERIMENT 4: Pre-Settlement with Multiple Encoders
// ---------------------------------------------------------------------------

/// Tests pre-settlement with sips (AppleJPEG) but recompression using
/// libjpeg-turbo and MozJPEG in addition to sips, to check whether
/// pre-settlement benefits transfer across encoder families.
///
/// Run: cargo test -p phasm-core --release --test qf_presettlement_experiment -- --ignored presettlement_cross_encoder --nocapture
#[test]
#[ignore]
fn presettlement_cross_encoder() {
    let sips_ok = Command::new("/usr/bin/sips")
        .arg("--help")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !sips_ok {
        println!("\nSKIPPED: sips not available (macOS only)");
        return;
    }

    let ljt_ok = Command::new("/opt/homebrew/bin/cjpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success() || !o.stderr.is_empty())
        .unwrap_or(false);

    let moz_ok = Command::new("/opt/homebrew/opt/mozjpeg/bin/cjpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success() || !o.stderr.is_empty())
        .unwrap_or(false);

    println!();
    println!("================================================================");
    println!("  QF PRE-SETTLEMENT EXPERIMENT — Cross-Encoder");
    println!("================================================================");
    println!("Pre-settle with sips, recompress with multiple encoders.");
    println!("Encoders: sips {}, libjpeg-turbo {}, MozJPEG {}",
        if sips_ok { "OK" } else { "MISSING" },
        if ljt_ok { "OK" } else { "MISSING" },
        if moz_ok { "OK" } else { "MISSING" });
    println!("================================================================");
    println!();

    struct Encoder {
        name: &'static str,
        available: bool,
        recompress: fn(&[u8], u8) -> Option<Vec<u8>>,
    }

    fn recomp_ljt(bytes: &[u8], qf: u8) -> Option<Vec<u8>> {
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir();
        let input = dir.join(format!("phasm_ce_ljt_in_{pid}_{ts}.jpg"));
        let ppm = dir.join(format!("phasm_ce_ljt_{pid}_{ts}.ppm"));
        let output = dir.join(format!("phasm_ce_ljt_out_{pid}_{ts}.jpg"));

        std::fs::write(&input, bytes).ok()?;
        let d = Command::new("/opt/homebrew/bin/djpeg")
            .arg("-ppm").arg("-outfile").arg(&ppm).arg(&input)
            .output().ok()?;
        let _ = std::fs::remove_file(&input);
        if !d.status.success() { return None; }
        let c = Command::new("/opt/homebrew/bin/cjpeg")
            .arg("-quality").arg(qf.to_string())
            .arg("-baseline")
            .arg("-outfile").arg(&output).arg(&ppm)
            .output().ok()?;
        let _ = std::fs::remove_file(&ppm);
        if !c.status.success() { let _ = std::fs::remove_file(&output); return None; }
        let r = std::fs::read(&output).ok();
        let _ = std::fs::remove_file(&output);
        r
    }

    fn recomp_moz(bytes: &[u8], qf: u8) -> Option<Vec<u8>> {
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir();
        let input = dir.join(format!("phasm_ce_moz_in_{pid}_{ts}.jpg"));
        let ppm = dir.join(format!("phasm_ce_moz_{pid}_{ts}.ppm"));
        let output = dir.join(format!("phasm_ce_moz_out_{pid}_{ts}.jpg"));

        std::fs::write(&input, bytes).ok()?;
        let d = Command::new("/opt/homebrew/bin/djpeg")
            .arg("-ppm").arg("-outfile").arg(&ppm).arg(&input)
            .output().ok()?;
        let _ = std::fs::remove_file(&input);
        if !d.status.success() { return None; }
        let c = Command::new("/opt/homebrew/opt/mozjpeg/bin/cjpeg")
            .arg("-quality").arg(qf.to_string())
            .arg("-baseline")
            .arg("-outfile").arg(&output).arg(&ppm)
            .output().ok()?;
        let _ = std::fs::remove_file(&ppm);
        if !c.status.success() { let _ = std::fs::remove_file(&output); return None; }
        let r = std::fs::read(&output).ok();
        let _ = std::fs::remove_file(&output);
        r
    }

    fn recomp_sips(bytes: &[u8], qf: u8) -> Option<Vec<u8>> {
        recompress_with_sips_static(bytes, qf)
    }

    // Need a non-method version for fn pointer
    fn recompress_with_sips_static(jpeg_bytes: &[u8], qf: u8) -> Option<Vec<u8>> {
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir();
        let input = dir.join(format!("phasm_ce_sips_in_{pid}_{ts}.jpg"));
        let output = dir.join(format!("phasm_ce_sips_out_{pid}_{ts}.jpg"));
        std::fs::write(&input, jpeg_bytes).ok()?;
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
    }

    let encoders: Vec<Encoder> = vec![
        Encoder { name: "sips", available: sips_ok, recompress: recomp_sips },
        Encoder { name: "libjpeg-turbo", available: ljt_ok, recompress: recomp_ljt },
        Encoder { name: "MozJPEG", available: moz_ok, recompress: recomp_moz },
    ];

    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_640x480", load_test_vector("photo_640x480_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let presettle_qfs: &[Option<u8>] = &[None, Some(70), Some(75)];
    let recomp_qf: u8 = 75;
    let passphrase = "cross-encoder-2026";
    let messages: Vec<&str> = vec!["Hello", "Meeting at noon", "The package is ready"];

    // Track stats per (encoder, presettle_qf)
    struct CellStats {
        encoder: String,
        pre_qf: String,
        total: u32,
        survived: u32,
    }
    let mut cell_stats: Vec<CellStats> = Vec::new();

    println!("{:<14} {:>6} {:<16} {:>4} {:>8} {:>6} {:>6} {:>4}",
        "Encoder", "PreQF", "Image", "Msg", "Result", "Integ", "RSErr", "r");
    println!("{}", "-".repeat(78));

    for encoder in &encoders {
        if !encoder.available {
            println!("{:<14} SKIPPED (not available)", encoder.name);
            continue;
        }

        for (img_name, cover_bytes) in &test_images {
            for &presettle_qf in presettle_qfs {
                let qf_label = match presettle_qf {
                    None => "None".to_string(),
                    Some(qf) => format!("{}", qf),
                };

                let settled_bytes = match presettle_qf {
                    None => cover_bytes.clone(),
                    Some(qf) => match presettle_with_sips(cover_bytes, qf) {
                        Some(b) => b,
                        None => continue,
                    },
                };

                let cap = match JpegImage::from_bytes(&settled_bytes) {
                    Ok(img) => armor_capacity(&img).unwrap_or(0),
                    Err(_) => continue,
                };

                for &message in &messages {
                    if message.len() > cap { continue; }

                    let stego = match armor_encode(&settled_bytes, message, passphrase) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };

                    let recompressed = match (encoder.recompress)(&stego, recomp_qf) {
                        Some(r) => r,
                        None => continue,
                    };

                    let cell = cell_stats.iter_mut()
                        .find(|c| c.encoder == encoder.name && c.pre_qf == qf_label);
                    if cell.is_none() {
                        cell_stats.push(CellStats {
                            encoder: encoder.name.to_string(),
                            pre_qf: qf_label.clone(),
                            total: 0,
                            survived: 0,
                        });
                    }
                    let cell = cell_stats.iter_mut()
                        .find(|c| c.encoder == encoder.name && c.pre_qf == qf_label).unwrap();
                    cell.total += 1;

                    match smart_decode(&recompressed, passphrase) {
                        Ok((decoded, quality)) => {
                            let ok = decoded == message;
                            if ok { cell.survived += 1; }
                            println!("{:<14} {:>6} {:<16} {:>4} {:>8} {:>5}% {:>6} {:>4}",
                                encoder.name, &qf_label, img_name, message.len(),
                                if ok { "YES" } else { "CORRUPT" },
                                quality.integrity_percent,
                                quality.rs_errors_corrected,
                                quality.repetition_factor);
                        }
                        Err(_) => {
                            println!("{:<14} {:>6} {:<16} {:>4} {:>8} {:>6} {:>6} {:>4}",
                                encoder.name, &qf_label, img_name, message.len(),
                                "FAIL", "-", "-", "-");
                        }
                    }
                }
            }
        }
        println!();
    }

    // Summary heatmap: encoder x presettle QF
    println!("================================================================");
    println!("SUMMARY — Survival by Encoder x Pre-Settle QF (recomp at QF {})", recomp_qf);
    println!();
    print!("{:<14}", "Encoder");
    for pqf in presettle_qfs {
        let label = match pqf { None => "None".to_string(), Some(qf) => format!("QF{}", qf) };
        print!(" {:>14}", label);
    }
    println!();
    println!("{}", "-".repeat(14 + presettle_qfs.len() * 15));

    for encoder in &encoders {
        if !encoder.available { continue; }
        print!("{:<14}", encoder.name);
        for pqf in presettle_qfs {
            let label = match pqf { None => "None".to_string(), Some(qf) => format!("{}", qf) };
            if let Some(cell) = cell_stats.iter().find(|c| c.encoder == encoder.name && c.pre_qf == label) {
                let rate = if cell.total > 0 { cell.survived as f64 / cell.total as f64 * 100.0 } else { 0.0 };
                print!(" {:>4}/{:<4}({:>3.0}%)", cell.survived, cell.total, rate);
            } else {
                print!(" {:>14}", "-");
            }
        }
        println!();
    }
    println!("================================================================");
}

// ---------------------------------------------------------------------------
// EXPERIMENT 5: Capacity vs. Message Length Sweep
// ---------------------------------------------------------------------------

/// Tests how pre-settlement affects success rate across message lengths.
/// Uses QF 70 pre-settlement vs. no pre-settlement, with recomp at QF 75.
///
/// Run: cargo test -p phasm-core --release --test qf_presettlement_experiment -- --ignored presettlement_message_sweep --nocapture
#[test]
#[ignore]
fn presettlement_message_sweep() {
    let sips_ok = Command::new("/usr/bin/sips")
        .arg("--help")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !sips_ok {
        println!("\nSKIPPED: sips not available (macOS only)");
        return;
    }

    let test_images: Vec<(&str, Vec<u8>)> = vec![
        ("photo_640x480", load_test_vector("photo_640x480_q75_420.jpg")),
        ("istock_612x408", load_real_photo("istockphoto-612x612-baseline.jpg")),
        ("real_1290x1715", load_real_photo("637586123-baseline.jpg")),
    ];

    let msg_lengths: &[usize] = &[5, 10, 15, 20, 30, 50, 80, 100, 150, 200, 300, 500];
    let presettle_qfs: &[Option<u8>] = &[None, Some(70)];
    let recomp_qf: u8 = 75;
    let passphrase = "msg-sweep-2026";

    println!();
    println!("================================================================");
    println!("  QF PRE-SETTLEMENT EXPERIMENT — Message Length Sweep");
    println!("================================================================");
    println!("Tests decode survival across message lengths.");
    println!("Pre-settle: None vs QF70.  Recomp: QF{} (sips)", recomp_qf);
    println!("================================================================");
    println!();

    // Track per (image, presettle, msg_len)
    struct LenStats {
        pre_qf: String,
        msg_len: usize,
        total: u32,
        survived: u32,
    }
    let mut len_stats: Vec<LenStats> = Vec::new();

    println!("{:<16} {:>6} {:>5} {:>8} {:>6} {:>6} {:>4}",
        "Image", "PreQF", "Msg", "Result", "Integ", "RSErr", "r");
    println!("{}", "-".repeat(60));

    for (img_name, cover_bytes) in &test_images {
        for &presettle_qf in presettle_qfs {
            let qf_label = match presettle_qf {
                None => "None".to_string(),
                Some(qf) => format!("{}", qf),
            };

            let settled_bytes = match presettle_qf {
                None => cover_bytes.clone(),
                Some(qf) => match presettle_with_sips(cover_bytes, qf) {
                    Some(b) => b,
                    None => continue,
                },
            };

            let cap = match JpegImage::from_bytes(&settled_bytes) {
                Ok(img) => armor_capacity(&img).unwrap_or(0),
                Err(_) => continue,
            };

            for &msg_len in msg_lengths {
                if msg_len > cap { continue; }

                let message = generate_message(msg_len);
                let stego = match armor_encode(&settled_bytes, &message, passphrase) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                let recompressed = match recompress_with_sips(&stego, recomp_qf) {
                    Some(r) => r,
                    None => continue,
                };

                let ls = len_stats.iter_mut()
                    .find(|s| s.pre_qf == qf_label && s.msg_len == msg_len);
                if ls.is_none() {
                    len_stats.push(LenStats {
                        pre_qf: qf_label.clone(),
                        msg_len,
                        total: 0,
                        survived: 0,
                    });
                }
                let ls = len_stats.iter_mut()
                    .find(|s| s.pre_qf == qf_label && s.msg_len == msg_len).unwrap();
                ls.total += 1;

                match smart_decode(&recompressed, passphrase) {
                    Ok((decoded, quality)) => {
                        let ok = decoded == message;
                        if ok { ls.survived += 1; }
                        println!("{:<16} {:>6} {:>5} {:>8} {:>5}% {:>6} {:>4}",
                            img_name, &qf_label, msg_len,
                            if ok { "YES" } else { "CORRUPT" },
                            quality.integrity_percent,
                            quality.rs_errors_corrected,
                            quality.repetition_factor);
                    }
                    Err(_) => {
                        println!("{:<16} {:>6} {:>5} {:>8} {:>6} {:>6} {:>4}",
                            img_name, &qf_label, msg_len,
                            "FAIL", "-", "-", "-");
                    }
                }
            }
        }
        println!();
    }

    // Summary
    println!("================================================================");
    println!("SUMMARY — Survival by Message Length x Pre-Settle QF");
    println!();
    print!("{:>6}", "MsgLen");
    for pqf in presettle_qfs {
        let label = match pqf { None => "None".to_string(), Some(qf) => format!("QF{}", qf) };
        print!(" {:>14}", label);
    }
    println!();
    println!("{}", "-".repeat(6 + presettle_qfs.len() * 15));

    for &msg_len in msg_lengths {
        print!("{:>6}", msg_len);
        for pqf in presettle_qfs {
            let label = match pqf { None => "None".to_string(), Some(qf) => format!("{}", qf) };
            if let Some(ls) = len_stats.iter().find(|s| s.pre_qf == label && s.msg_len == msg_len) {
                let rate = if ls.total > 0 { ls.survived as f64 / ls.total as f64 * 100.0 } else { 0.0 };
                print!(" {:>4}/{:<4}({:>3.0}%)", ls.survived, ls.total, rate);
            } else {
                print!(" {:>14}", "-");
            }
        }
        println!();
    }
    println!("================================================================");
}
