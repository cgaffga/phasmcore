// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Tests for progressive JPEG (SOF2) parsing support.
//!
//! Verifies that progressive JPEG files (e.g., from WhatsApp) can be parsed,
//! their coefficients extracted, and written back as baseline JPEG with the
//! same coefficient values.

use phasm_core::JpegImage;
use std::path::Path;

fn read_test_image(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn try_read_test_image(name: &str) -> Option<Vec<u8>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors")
        .join(name);
    std::fs::read(&path).ok()
}

#[test]
fn progressive_whatsapp_parses() {
    let Some(data) = try_read_test_image("progressive_whatsapp_1200x1600.jpg") else {
        eprintln!("skipped: test vector not found"); return;
    };
    let img = JpegImage::from_bytes(&data).unwrap();

    let fi = img.frame_info();
    assert_eq!(fi.width, 1200, "expected width 1200");
    assert_eq!(fi.height, 1600, "expected height 1600");
    assert_eq!(fi.components.len(), 3, "expected 3 components (YCbCr)");
}

#[test]
fn progressive_whatsapp_has_quant_tables() {
    let Some(data) = try_read_test_image("progressive_whatsapp_1200x1600.jpg") else {
        eprintln!("skipped: test vector not found"); return;
    };
    let img = JpegImage::from_bytes(&data).unwrap();

    // Should have at least one quant table
    let qt0 = img.quant_table(0);
    assert!(qt0.is_some(), "expected quant table 0");
    let qt = qt0.unwrap();
    assert!(qt.values.iter().any(|&v| v > 0), "quant table should have nonzero values");
}

#[test]
fn progressive_whatsapp_correct_grid_sizes() {
    let Some(data) = try_read_test_image("progressive_whatsapp_1200x1600.jpg") else {
        eprintln!("skipped: test vector not found"); return;
    };
    let img = JpegImage::from_bytes(&data).unwrap();

    let fi = img.frame_info();

    // Verify grid sizes match frame info
    for comp_idx in 0..fi.components.len() {
        let grid = img.dct_grid(comp_idx);
        let expected_bw = fi.blocks_wide(comp_idx);
        let expected_bt = fi.blocks_tall(comp_idx);
        assert_eq!(
            grid.blocks_wide(), expected_bw,
            "component {} blocks_wide mismatch", comp_idx
        );
        assert_eq!(
            grid.blocks_tall(), expected_bt,
            "component {} blocks_tall mismatch", comp_idx
        );
    }
}

#[test]
fn progressive_whatsapp_has_nonzero_coefficients() {
    let Some(data) = try_read_test_image("progressive_whatsapp_1200x1600.jpg") else {
        eprintln!("skipped: test vector not found"); return;
    };
    let img = JpegImage::from_bytes(&data).unwrap();

    // The image should have significant nonzero coefficients
    let grid = img.dct_grid(0); // luminance
    let mut nonzero_dc = 0u32;
    let mut nonzero_ac = 0u32;

    for br in 0..grid.blocks_tall() {
        for bc in 0..grid.blocks_wide() {
            let dc = grid.get(br, bc, 0, 0);
            if dc != 0 {
                nonzero_dc += 1;
            }
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 { continue; }
                    if grid.get(br, bc, i, j) != 0 {
                        nonzero_ac += 1;
                    }
                }
            }
        }
    }

    let total_blocks = grid.total_blocks();
    eprintln!("Luminance: {} blocks, {} nonzero DC ({:.1}%), {} nonzero AC ({:.1}%)",
        total_blocks, nonzero_dc,
        nonzero_dc as f64 / total_blocks as f64 * 100.0,
        nonzero_ac,
        nonzero_ac as f64 / (total_blocks as f64 * 63.0) * 100.0,
    );

    // A real photo should have many nonzero DC coefficients
    assert!(nonzero_dc > total_blocks as u32 / 2,
        "expected most DC coefficients to be nonzero, got {}/{}", nonzero_dc, total_blocks);
    // A real photo should have many nonzero AC coefficients
    assert!(nonzero_ac > 100,
        "expected many nonzero AC coefficients, got {}", nonzero_ac);
}

#[test]
fn progressive_whatsapp_write_baseline_roundtrip() {
    let Some(data) = try_read_test_image("progressive_whatsapp_1200x1600.jpg") else {
        eprintln!("skipped: test vector not found"); return;
    };
    let img = JpegImage::from_bytes(&data).unwrap();

    // Write as baseline
    let baseline_bytes = img.to_bytes().unwrap();

    // Verify it starts with SOI
    assert_eq!(baseline_bytes[0], 0xFF);
    assert_eq!(baseline_bytes[1], 0xD8);

    // Re-parse the baseline output
    let img2 = JpegImage::from_bytes(&baseline_bytes).unwrap();

    let fi1 = img.frame_info();
    let fi2 = img2.frame_info();
    assert_eq!(fi1.width, fi2.width);
    assert_eq!(fi1.height, fi2.height);
    assert_eq!(fi1.components.len(), fi2.components.len());

    // Verify ALL coefficients match between progressive parse and baseline re-parse
    for comp_idx in 0..fi1.components.len() {
        let grid1 = img.dct_grid(comp_idx);
        let grid2 = img2.dct_grid(comp_idx);

        assert_eq!(grid1.blocks_wide(), grid2.blocks_wide(),
            "component {} blocks_wide mismatch after roundtrip", comp_idx);
        assert_eq!(grid1.blocks_tall(), grid2.blocks_tall(),
            "component {} blocks_tall mismatch after roundtrip", comp_idx);

        let mut diff_count = 0u32;
        let mut first_diff = None;

        for br in 0..grid1.blocks_tall() {
            for bc in 0..grid1.blocks_wide() {
                for i in 0..8 {
                    for j in 0..8 {
                        let v1 = grid1.get(br, bc, i, j);
                        let v2 = grid2.get(br, bc, i, j);
                        if v1 != v2 {
                            diff_count += 1;
                            if first_diff.is_none() {
                                first_diff = Some((comp_idx, br, bc, i, j, v1, v2));
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(diff_count, 0,
            "component {}: {} coefficient differences after roundtrip. First: {:?}",
            comp_idx, diff_count, first_diff);
    }
}

#[test]
fn progressive_baseline_output_is_valid_jpeg() {
    let Some(data) = try_read_test_image("progressive_whatsapp_1200x1600.jpg") else {
        eprintln!("skipped: test vector not found"); return;
    };
    let img = JpegImage::from_bytes(&data).unwrap();

    let baseline_bytes = img.to_bytes().unwrap();

    // Verify it's a valid baseline JPEG by checking structure
    assert!(baseline_bytes.len() > 100, "output too small");
    assert_eq!(baseline_bytes[0], 0xFF);
    assert_eq!(baseline_bytes[1], 0xD8); // SOI

    // Should end with EOI
    let len = baseline_bytes.len();
    assert_eq!(baseline_bytes[len - 2], 0xFF);
    assert_eq!(baseline_bytes[len - 1], 0xD9); // EOI

    // Should contain SOF0 (baseline), not SOF2 (progressive)
    let mut found_sof0 = false;
    let mut found_sof2 = false;
    let mut pos = 2;
    while pos + 1 < baseline_bytes.len() {
        if baseline_bytes[pos] == 0xFF {
            let m = baseline_bytes[pos + 1];
            if m == 0xC0 { found_sof0 = true; }
            if m == 0xC2 { found_sof2 = true; }
            if m == 0xDA { break; } // SOS — stop scanning
        }
        pos += 1;
    }
    assert!(found_sof0, "baseline output should contain SOF0");
    assert!(!found_sof2, "baseline output should NOT contain SOF2");
}

#[test]
fn progressive_double_roundtrip() {
    // Parse progressive -> write baseline -> parse baseline -> write baseline
    // Coefficients should be identical at every step.
    let Some(data) = try_read_test_image("progressive_whatsapp_1200x1600.jpg") else {
        eprintln!("skipped: test vector not found"); return;
    };
    let img1 = JpegImage::from_bytes(&data).unwrap();
    let baseline1 = img1.to_bytes().unwrap();

    let img2 = JpegImage::from_bytes(&baseline1).unwrap();
    let baseline2 = img2.to_bytes().unwrap();

    // The two baseline outputs should be identical
    assert_eq!(baseline1, baseline2,
        "double roundtrip: baseline outputs differ (len {} vs {})",
        baseline1.len(), baseline2.len());
}

/// Verify that existing baseline test images are not affected by the progressive code path.
#[test]
fn baseline_images_still_work() {
    let images = [
        "red_64x64_q75_444.jpg",
        "blue_64x64_q90_420.jpg",
        "green_64x64_q50_422.jpg",
        "gray_64x64_q75.jpg",
        "tiny_8x8_q95.jpg",
        "nonaligned_13x13_q80.jpg",
        "fractal_100x75_q85_420.jpg",
        "photo_320x240_q75_420.jpg",
    ];

    for name in &images {
        let data = read_test_image(name);
        let img = JpegImage::from_bytes(&data).unwrap();
        let output = img.to_bytes().unwrap();
        assert_eq!(data, output,
            "baseline roundtrip failed for {}: len {} vs {}", name, data.len(), output.len());
    }
}
