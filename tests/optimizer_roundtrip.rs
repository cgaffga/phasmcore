// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Integration tests for the cover image optimizer.
//!
//! Tests the full pipeline: optimize raw pixels → JPEG compress → stego encode
//! → decode → verify message integrity.

use phasm_core::jpeg::pixels::{jpeg_to_luma_f64, luma_f64_to_jpeg};
use phasm_core::{
    armor_encode, ghost_capacity, ghost_encode, optimize_cover, smart_decode,
    JpegImage, OptimizerConfig, OptimizerMode,
};

fn load_test_image(name: &str) -> Option<Vec<u8>> {
    std::fs::read(format!("test-vectors/{name}")).ok()
}

/// Optimize luma pixels and produce a new JPEG.
fn optimize_and_compress(
    original_bytes: &[u8],
    mode: OptimizerMode,
    strength: f32,
) -> Vec<u8> {
    let img = JpegImage::from_bytes(original_bytes).unwrap();
    let (luma, width, height) = jpeg_to_luma_f64(&img).unwrap();

    // Convert luma f64 to RGB u8 (grayscale → all channels equal)
    let mut rgb = vec![0u8; width * height * 3];
    for i in 0..width * height {
        let v = luma[i].round().clamp(0.0, 255.0) as u8;
        rgb[i * 3] = v;
        rgb[i * 3 + 1] = v;
        rgb[i * 3 + 2] = v;
    }

    let config = OptimizerConfig {
        strength,
        seed: [42u8; 32],
        mode,
    };
    let optimized_rgb = optimize_cover(&rgb, width as u32, height as u32, &config);

    // Convert back to luma f64 and write to JPEG
    let optimized_luma: Vec<f64> = (0..width * height)
        .map(|i| optimized_rgb[i * 3] as f64)
        .collect();

    let mut img_opt = JpegImage::from_bytes(original_bytes).unwrap();
    luma_f64_to_jpeg(&optimized_luma, width, height, &mut img_opt).unwrap();
    img_opt.rebuild_huffman_tables();
    img_opt.to_bytes().unwrap()
}

#[test]
fn ghost_roundtrip_with_optimizer() {
    let Some(original_bytes) = load_test_image("photo_640x480_q75_420.jpg") else {
        eprintln!("Skipping: test vector not found");
        return;
    };

    let opt_bytes = optimize_and_compress(&original_bytes, OptimizerMode::Ghost, 0.85);

    // Encode a message into the optimized image
    let message = "Hello from optimized cover!";
    let passphrase = "testpass";
    let stego = ghost_encode(&opt_bytes, message, passphrase).unwrap();

    // Decode and verify
    let (payload, _quality) = smart_decode(&stego, passphrase).unwrap();
    assert_eq!(payload.text, message);
}

#[test]
fn ghost_capacity_increase() {
    let Some(original_bytes) = load_test_image("photo_640x480_q75_420.jpg") else {
        eprintln!("Skipping: test vector not found");
        return;
    };

    let img_orig = JpegImage::from_bytes(&original_bytes).unwrap();
    let cap_original = ghost_capacity(&img_orig).unwrap();

    let opt_bytes = optimize_and_compress(&original_bytes, OptimizerMode::Ghost, 0.85);
    let img_opt = JpegImage::from_bytes(&opt_bytes).unwrap();
    let cap_optimized = ghost_capacity(&img_opt).unwrap();

    let increase_pct = 100.0 * (cap_optimized as f64 - cap_original as f64) / cap_original as f64;
    eprintln!(
        "Ghost capacity: original={cap_original}, optimized={cap_optimized}, increase={increase_pct:.1}%"
    );

    assert!(
        cap_optimized > cap_original,
        "optimized capacity ({cap_optimized}) should exceed original ({cap_original})"
    );
    assert!(
        increase_pct > 10.0,
        "capacity should increase by > 10%, got {increase_pct:.1}%"
    );
}

#[test]
fn armor_roundtrip_with_optimizer() {
    let Some(original_bytes) = load_test_image("photo_640x480_q75_420.jpg") else {
        eprintln!("Skipping: test vector not found");
        return;
    };

    let opt_bytes = optimize_and_compress(&original_bytes, OptimizerMode::Armor, 0.85);

    let message = "Armor optimized!";
    let passphrase = "armorpass";
    let stego = armor_encode(&opt_bytes, message, passphrase).unwrap();

    let (payload, _quality) = smart_decode(&stego, passphrase).unwrap();
    assert_eq!(payload.text, message);
}

#[test]
fn psnr_real_image() {
    let Some(original_bytes) = load_test_image("photo_640x480_q75_420.jpg") else {
        eprintln!("Skipping: test vector not found");
        return;
    };

    let img = JpegImage::from_bytes(&original_bytes).unwrap();
    let (luma_orig, width, height) = jpeg_to_luma_f64(&img).unwrap();

    // Create RGB from luma
    let mut rgb = vec![0u8; width * height * 3];
    for i in 0..width * height {
        let v = luma_orig[i].round().clamp(0.0, 255.0) as u8;
        rgb[i * 3] = v;
        rgb[i * 3 + 1] = v;
        rgb[i * 3 + 2] = v;
    }

    let config = OptimizerConfig {
        strength: 0.85,
        seed: [42u8; 32],
        mode: OptimizerMode::Ghost,
    };
    let optimized_rgb = optimize_cover(&rgb, width as u32, height as u32, &config);

    // Compute PSNR on luma channel
    let mse: f64 = (0..width * height)
        .map(|i| {
            let orig = rgb[i * 3] as f64;
            let opt = optimized_rgb[i * 3] as f64;
            let d = orig - opt;
            d * d
        })
        .sum::<f64>()
        / (width * height) as f64;

    let psnr = if mse > 0.0 {
        10.0 * (255.0 * 255.0 / mse).log10()
    } else {
        f64::INFINITY
    };

    eprintln!("PSNR (real photo, luma): {psnr:.1} dB");

    assert!(
        psnr > 44.0,
        "PSNR should be > 44 dB for real photos, got {psnr:.1} dB"
    );
}
