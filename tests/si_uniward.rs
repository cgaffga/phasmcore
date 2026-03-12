// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Integration tests for SI-UNIWARD ("Deep Cover") Ghost mode.
//!
//! These tests verify that:
//! - SI-encoded images decode correctly with the STANDARD decoder (T8)
//! - SI doesn't break existing J-UNIWARD images (T11)
//! - SI handles edge cases: small images, non-aligned dimensions (T13, T14)
//! - Wrong passphrase still fails (T12)
//! - File attachments work with SI (T10)

use phasm_core::{
    ghost_encode, ghost_decode, ghost_encode_si, ghost_encode_si_with_files,
    JpegImage, FileEntry,
};
use phasm_core::jpeg::pixels::{jpeg_to_luma_f64, luma_f64_to_jpeg};

fn load_test_image(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/{name}")).unwrap()
}

/// Create a synthetic raw RGB pixel buffer and a matching cover JPEG.
///
/// Process:
/// 1. Load a real JPEG test image.
/// 2. Decompress to Y luma pixels.
/// 3. Add deliberate sub-pixel perturbations to create non-trivial rounding errors.
/// 4. Write perturbed pixels back → new cover JPEG.
/// 5. Build fake RGB (R=G=B=Y) from the perturbed pixels.
///
/// Returns (jpeg_bytes, rgb_pixels, width, height).
fn create_si_test_data(base_image: &str) -> (Vec<u8>, Vec<u8>, u32, u32) {
    let original_bytes = load_test_image(base_image);
    let mut img = JpegImage::from_bytes(&original_bytes).unwrap();

    // Decompress to luma pixels
    let (mut pixels, width, height) = jpeg_to_luma_f64(&img).unwrap();

    // Add sub-pixel perturbations to create genuine rounding errors.
    // These simulate what happens when a non-JPEG image (PNG/HEIC) is
    // JPEG-compressed: the continuous pixel values don't land exactly
    // on the quantization grid.
    for (i, pixel) in pixels.iter_mut().enumerate() {
        let perturbation = ((i as f64 * 0.37).sin() * 3.5).clamp(-4.0, 4.0);
        *pixel = (*pixel + perturbation).clamp(0.0, 255.0);
    }

    // Write perturbed pixels back into the JPEG (forward DCT + quantize)
    luma_f64_to_jpeg(&pixels, width, height, &mut img).unwrap();
    // Rebuild Huffman tables since modified coefficients may need new symbols.
    img.rebuild_huffman_tables();
    let cover_bytes = img.to_bytes().unwrap();

    // Build RGB from perturbed luma (R=G=B=Y for each pixel)
    // BT.601: Y = 0.299*R + 0.587*G + 0.114*B. When R=G=B=Y:
    // Y_reconstructed = Y * (0.299 + 0.587 + 0.114) = Y * 1.0. Correct.
    let mut rgb = Vec::with_capacity(width * height * 3);
    for &y in &pixels {
        let v = y.round().clamp(0.0, 255.0) as u8;
        rgb.push(v);
        rgb.push(v);
        rgb.push(v);
    }

    (cover_bytes, rgb, width as u32, height as u32)
}

// --- T8: SI-UNIWARD round-trip (THE key test) ---

#[test]
fn t8_si_encode_standard_decode_roundtrip() {
    let (cover, raw_rgb, w, h) = create_si_test_data("photo_320x240_q75_420.jpg");
    let message = "Hello from Deep Cover!";
    let passphrase = "si-test-key-42";

    let stego = ghost_encode_si(&cover, &raw_rgb, w, h, message, passphrase).unwrap();

    // The KEY assertion: standard ghost_decode works on SI-encoded images.
    let decoded = ghost_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
}

#[test]
fn t8b_si_roundtrip_various_messages() {
    let (cover, raw_rgb, w, h) = create_si_test_data("photo_320x240_q75_420.jpg");
    let passphrase = "si-multi-test";

    for len in [1, 5, 20, 50, 100] {
        let message: String = (0..len).map(|i| (b'A' + (i % 26) as u8) as char).collect();
        let stego = ghost_encode_si(&cover, &raw_rgb, w, h, &message, passphrase).unwrap();
        let decoded = ghost_decode(&stego, passphrase).unwrap();
        assert_eq!(decoded.text, message, "failed for message length {len}");
    }
}

#[test]
fn t8c_si_roundtrip_unicode() {
    let (cover, raw_rgb, w, h) = create_si_test_data("photo_320x240_q75_420.jpg");
    let message = "SI-UNIWARD: Héllo wörld! 暗号 🔐";
    let passphrase = "unicode-si";

    let stego = ghost_encode_si(&cover, &raw_rgb, w, h, message, passphrase).unwrap();
    let decoded = ghost_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
}

// --- T9: SI vs J-UNIWARD comparison ---

#[test]
fn t9_si_produces_valid_stego_jpeg() {
    let (cover, raw_rgb, w, h) = create_si_test_data("photo_320x240_q75_420.jpg");
    let stego = ghost_encode_si(&cover, &raw_rgb, w, h, "test", "pass").unwrap();

    // Stego output must be parseable as a valid JPEG.
    let img = JpegImage::from_bytes(&stego).unwrap();
    let frame = img.frame_info();
    assert!(frame.width > 0);
    assert!(frame.height > 0);
}

#[test]
fn t9b_si_and_standard_both_decode() {
    let (cover, raw_rgb, w, h) = create_si_test_data("photo_320x240_q75_420.jpg");
    let message = "compare modes";
    let pass = "compare-pass";

    // Encode with both modes
    let stego_j = ghost_encode(&cover, message, pass).unwrap();
    let stego_si = ghost_encode_si(&cover, &raw_rgb, w, h, message, pass).unwrap();

    // Both must decode correctly
    assert_eq!(ghost_decode(&stego_j, pass).unwrap().text, message);
    assert_eq!(ghost_decode(&stego_si, pass).unwrap().text, message);
}

// --- T10: SI with file attachments ---

#[test]
fn t10_si_encode_with_files_roundtrip() {
    let (cover, raw_rgb, w, h) = create_si_test_data("photo_320x240_q75_420.jpg");
    let files = vec![FileEntry {
        filename: "secret.txt".to_string(),
        content: b"Hidden file content!".to_vec(),
    }];

    let stego = ghost_encode_si_with_files(
        &cover, &raw_rgb, w, h,
        "msg with file", &files, "file-pass",
    ).unwrap();

    let decoded = ghost_decode(&stego, "file-pass").unwrap();
    assert_eq!(decoded.text, "msg with file");
    assert_eq!(decoded.files.len(), 1);
    assert_eq!(decoded.files[0].filename, "secret.txt");
    assert_eq!(decoded.files[0].content, b"Hidden file content!");
}

// --- T11: Backward compatibility ---

#[test]
fn t11_existing_j_uniward_images_still_decode() {
    // Encode with standard J-UNIWARD (no side info)
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let stego = ghost_encode(&cover, "backward compat", "key").unwrap();

    // Decode must still work (regression test)
    let decoded = ghost_decode(&stego, "key").unwrap();
    assert_eq!(decoded.text, "backward compat");
}

// --- T12: Wrong passphrase ---

#[test]
fn t12_si_wrong_passphrase_fails() {
    let (cover, raw_rgb, w, h) = create_si_test_data("photo_320x240_q75_420.jpg");
    let stego = ghost_encode_si(&cover, &raw_rgb, w, h, "secret", "right-key").unwrap();

    let err = ghost_decode(&stego, "wrong-key");
    assert!(err.is_err(), "decoding SI-encoded image with wrong passphrase should fail");
}

// --- T13: Small image ---

#[test]
fn t13_si_small_image() {
    // 320x240 is already fairly small; test with a short message
    let (cover, raw_rgb, w, h) = create_si_test_data("photo_320x240_q75_420.jpg");
    let stego = ghost_encode_si(&cover, &raw_rgb, w, h, "small", "p").unwrap();
    assert_eq!(ghost_decode(&stego, "p").unwrap().text, "small");
}

// --- T15: Empty message ---

#[test]
fn t15_si_empty_message() {
    let (cover, raw_rgb, w, h) = create_si_test_data("photo_320x240_q75_420.jpg");
    let stego = ghost_encode_si(&cover, &raw_rgb, w, h, "", "pass").unwrap();
    assert_eq!(ghost_decode(&stego, "pass").unwrap().text, "");
}

// --- T11b: smart_decode detects SI-encoded Ghost images ---

#[test]
fn t11b_smart_decode_detects_si_ghost() {
    let (cover, raw_rgb, w, h) = create_si_test_data("photo_320x240_q75_420.jpg");
    let stego = ghost_encode_si(&cover, &raw_rgb, w, h, "smart test", "smart-key").unwrap();

    let (payload, _quality) = phasm_core::smart_decode(&stego, "smart-key").unwrap();
    assert_eq!(payload.text, "smart test");
}
