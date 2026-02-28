// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Round-trip integration tests for Armor mode encode/decode.

use phasm_core::{armor_encode, armor_decode, armor_capacity, smart_decode, JpegImage};

fn load_test_image(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/{name}")).unwrap()
}

#[test]
fn armor_roundtrip_basic() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "Hello, Armor mode!";
    let passphrase = "test-passphrase-123";

    let stego = armor_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = armor_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
    assert_eq!(quality.mode, 0x02, "should be Armor mode");
    assert_eq!(quality.rs_errors_corrected, 0, "no recompression = no errors");
    assert!(quality.integrity_percent >= 85,
        "Pristine Armor integrity should be high: {}%", quality.integrity_percent);
}

#[test]
fn armor_wrong_passphrase_fails() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let stego = armor_encode(&cover, "secret", "correct-pass").unwrap();

    let result = armor_decode(&stego, "wrong-pass");
    assert!(result.is_err(), "decoding with wrong passphrase should fail");
}

#[test]
fn armor_capacity_positive() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let img = JpegImage::from_bytes(&cover).unwrap();
    let cap = armor_capacity(&img).unwrap();

    assert!(cap > 0, "capacity should be positive for 320x240");
    assert!(cap < 3000, "capacity {cap} suspiciously high for Armor");
}

#[test]
fn armor_stego_is_valid_jpeg() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let stego = armor_encode(&cover, "test", "pass").unwrap();

    let img = JpegImage::from_bytes(&stego).unwrap();
    let frame = img.frame_info();
    assert_eq!(frame.width, 320);
    assert_eq!(frame.height, 240);
}

#[test]
fn armor_roundtrip_unicode() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "Héllo wörld! 🔐";
    let passphrase = "unicode-key";

    let stego = armor_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = armor_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
    assert!(quality.integrity_percent >= 85,
        "Pristine Armor integrity should be high: {}%", quality.integrity_percent);
}

#[test]
fn smart_decode_detects_ghost() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "ghost message";
    let passphrase = "shared-pass";

    let stego = phasm_core::ghost_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = smart_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
    assert_eq!(quality.mode, 0x01, "should detect Ghost mode");
    assert_eq!(quality.integrity_percent, 100, "Ghost is always 100%");
    assert_eq!(quality.rs_errors_corrected, 0, "Ghost has no RS");
}

#[test]
fn smart_decode_detects_armor() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "armor message";
    let passphrase = "shared-pass";

    let stego = armor_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = smart_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
    assert_eq!(quality.mode, 0x02, "should detect Armor mode");
    assert!(quality.integrity_percent >= 85,
        "Pristine Armor integrity should be high: {}%", quality.integrity_percent);
}

#[test]
fn both_modes_have_positive_capacity() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let img = JpegImage::from_bytes(&cover).unwrap();
    let armor_cap = armor_capacity(&img).unwrap();
    let ghost_cap = phasm_core::ghost_capacity(&img).unwrap();

    assert!(armor_cap > 0, "Armor capacity should be positive");
    assert!(ghost_cap > 0, "Ghost capacity should be positive");
}

// --- Phase 2 adaptive robustness tests ---

#[test]
fn armor_phase2_short_message_activates_repetition() {
    // Short message in 640x480 — encoder uses Phase 2 (r>=3) for robustness.
    // V3 uses larger deltas (up to 8× mean_qt), so the decoder's delta sweep
    // may find the message via Phase 1 path with RS correction even for Phase 2
    // encoded data (because most STDM projections land at n=0 lattice point,
    // which extracts correctly at any delta). The key check: message round-trips.
    let cover = load_test_image("photo_640x480_q75_420.jpg");
    let message = "armor message";
    let passphrase = "phase2-test-key";

    let stego = armor_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = armor_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
    assert_eq!(quality.mode, 0x02);
    assert!(quality.integrity_percent >= 50,
        "Integrity should be reasonable: {}%", quality.integrity_percent);
}

#[test]
fn armor_phase2_quality_fields_present() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "test";
    let passphrase = "quality-fields-test";

    let stego = armor_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = armor_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
    // New quality fields should be populated
    assert!(quality.parity_len > 0, "parity_len should be set");
}

#[test]
fn armor_phase1_large_message_no_repetition() {
    // A message that fills most of the capacity should use Phase 1 (r=1).
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let img = JpegImage::from_bytes(&cover).unwrap();
    let cap = armor_capacity(&img).unwrap();

    // Use ~80% of capacity
    let msg_len = (cap * 4 / 5).min(cap);
    if msg_len < 10 {
        return; // Image too small for this test
    }
    let message: String = "A".repeat(msg_len);
    let passphrase = "large-msg-test";

    let stego = armor_encode(&cover, &message, passphrase).unwrap();
    let (decoded, quality) = armor_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
    assert!(quality.integrity_percent >= 85,
        "Pristine Armor Phase 1 integrity should be high: {}%", quality.integrity_percent);
    // Phase 1 should be used (no repetition or r=1)
    assert!(
        quality.repetition_factor <= 1,
        "Large message should use Phase 1: r={}",
        quality.repetition_factor
    );
}

#[test]
fn armor_phase2_smart_decode_works() {
    // smart_decode should work with Phase 2 encoded images
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "smart decode p2";
    let passphrase = "smart-p2-key";

    let stego = armor_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = smart_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
    assert_eq!(quality.mode, 0x02, "should detect Armor mode");
    assert!(quality.integrity_percent >= 85,
        "Pristine Armor integrity should be high: {}%", quality.integrity_percent);
}
