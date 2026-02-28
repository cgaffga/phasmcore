// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! JPEG codec round-trip tests verifying byte-for-byte decode/re-encode fidelity.

use phasm_core::JpegImage;
use std::path::Path;

fn read_test_image(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

#[test]
fn roundtrip_red_444() {
    let data = read_test_image("red_64x64_q75_444.jpg");
    let img = JpegImage::from_bytes(&data).unwrap();
    let output = img.to_bytes().unwrap();
    assert_eq!(data, output, "red 4:4:4 round-trip failed (len: {} vs {})", data.len(), output.len());
}

#[test]
fn roundtrip_blue_420() {
    let data = read_test_image("blue_64x64_q90_420.jpg");
    let img = JpegImage::from_bytes(&data).unwrap();
    let output = img.to_bytes().unwrap();
    assert_eq!(data, output, "blue 4:2:0 round-trip failed");
}

#[test]
fn roundtrip_green_422() {
    let data = read_test_image("green_64x64_q50_422.jpg");
    let img = JpegImage::from_bytes(&data).unwrap();
    let output = img.to_bytes().unwrap();
    assert_eq!(data, output, "green 4:2:2 round-trip failed");
}

#[test]
fn roundtrip_grayscale() {
    let data = read_test_image("gray_64x64_q75.jpg");
    let img = JpegImage::from_bytes(&data).unwrap();
    let output = img.to_bytes().unwrap();
    assert_eq!(data, output, "grayscale round-trip failed");
}

#[test]
fn roundtrip_tiny_8x8() {
    let data = read_test_image("tiny_8x8_q95.jpg");
    let img = JpegImage::from_bytes(&data).unwrap();
    let output = img.to_bytes().unwrap();
    assert_eq!(data, output, "tiny 8x8 round-trip failed");
}

#[test]
fn roundtrip_non_aligned() {
    let data = read_test_image("nonaligned_13x13_q80.jpg");
    let img = JpegImage::from_bytes(&data).unwrap();
    let output = img.to_bytes().unwrap();
    assert_eq!(data, output, "non-aligned 13x13 round-trip failed");
}

#[test]
fn roundtrip_fractal() {
    let data = read_test_image("fractal_100x75_q85_420.jpg");
    let img = JpegImage::from_bytes(&data).unwrap();
    let output = img.to_bytes().unwrap();
    assert_eq!(data, output, "fractal round-trip failed");
}

#[test]
fn modify_single_coefficient() {
    // Use the fractal image which has richer Huffman tables (non-trivial content)
    let data = read_test_image("fractal_100x75_q85_420.jpg");
    let mut img = JpegImage::from_bytes(&data).unwrap();

    // Read original DC value
    let original_dc = img.dct_grid(0).get(0, 0, 0, 0);

    // Find a nonzero AC coefficient and modify it by ±1 (steganography-style)
    let grid = img.dct_grid(0);
    let mut target_val = 0i16;
    let mut target_pos = (0usize, 0usize, 1usize, 1usize);
    'outer: for br in 0..grid.blocks_tall() {
        for bc in 0..grid.blocks_wide() {
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 {
                        continue; // skip DC
                    }
                    let v = grid.get(br, bc, i, j);
                    if v != 0 {
                        target_val = v;
                        target_pos = (br, bc, i, j);
                        break 'outer;
                    }
                }
            }
        }
    }
    assert_ne!(target_val, 0, "should find a nonzero AC coefficient");

    let new_val = if target_val > 0 { target_val - 1 } else { target_val + 1 };
    let (br, bc, i, j) = target_pos;
    img.dct_grid_mut(0).set(br, bc, i, j, new_val);

    let modified_bytes = img.to_bytes().unwrap();

    // Re-read the modified image
    let img2 = JpegImage::from_bytes(&modified_bytes).unwrap();
    assert_eq!(img2.dct_grid(0).get(br, bc, i, j), new_val);
    // DC of block (0,0) should be unchanged
    assert_eq!(img2.dct_grid(0).get(0, 0, 0, 0), original_dc);
    // Other coefficients should be unchanged
    let grid1 = img.dct_grid(0);
    let grid2 = img2.dct_grid(0);
    let mut diff_count = 0;
    for tbr in 0..grid1.blocks_tall() {
        for tbc in 0..grid1.blocks_wide() {
            for ti in 0..8 {
                for tj in 0..8 {
                    if grid1.get(tbr, tbc, ti, tj) != grid2.get(tbr, tbc, ti, tj) {
                        diff_count += 1;
                    }
                }
            }
        }
    }
    assert_eq!(diff_count, 0, "only the modified coefficient should differ (re-read reflects change)");
}

#[test]
fn frame_info_correct() {
    let data = read_test_image("blue_64x64_q90_420.jpg");
    let img = JpegImage::from_bytes(&data).unwrap();
    let fi = img.frame_info();

    assert_eq!(fi.width, 64);
    assert_eq!(fi.height, 64);
    assert_eq!(fi.components.len(), 3);
    assert_eq!(fi.max_h_sampling, 2);
    assert_eq!(fi.max_v_sampling, 2);
}

#[test]
fn grayscale_single_component() {
    let data = read_test_image("gray_64x64_q75.jpg");
    let img = JpegImage::from_bytes(&data).unwrap();

    assert_eq!(img.num_components(), 1);
    let fi = img.frame_info();
    assert_eq!(fi.components.len(), 1);
    assert_eq!(fi.max_h_sampling, 1);
    assert_eq!(fi.max_v_sampling, 1);
}

#[test]
fn roundtrip_photo() {
    let data = read_test_image("photo_320x240_q75_420.jpg");
    let img = JpegImage::from_bytes(&data).unwrap();
    let output = img.to_bytes().unwrap();
    assert_eq!(data, output, "photo round-trip failed (len: {} vs {})", data.len(), output.len());
}

#[test]
fn quant_tables_present() {
    let data = read_test_image("red_64x64_q75_444.jpg");
    let img = JpegImage::from_bytes(&data).unwrap();

    // Should have at least one quant table
    let qt0 = img.quant_table(0);
    assert!(qt0.is_some());

    // Values should be reasonable (not all zero)
    let qt = qt0.unwrap();
    assert!(qt.values.iter().any(|&v| v > 0));
}
