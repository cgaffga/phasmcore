// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Round-trip integration tests for Ghost shadow messages.
//!
//! Shadow uses Y-channel direct LSB embedding with headerless brute-force
//! decoding and Reed-Solomon error correction. Dynamic w + ∞-cost protection
//! gives near-zero BER on shadows when w >= 2.

use phasm_core::{
    ghost_encode, ghost_decode, ghost_encode_with_shadows,
    ghost_shadow_decode, smart_decode, ShadowLayer, StegoError,
    estimate_shadow_capacity, JpegImage,
};

fn load_test_image(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/{name}")).unwrap()
}

/// Use the larger test image for shadow tests.
fn load_shadow_test_image() -> Vec<u8> {
    load_test_image("photo_640x480_q75_420.jpg")
}

#[test]
fn shadow_single_roundtrip() {
    let cover = load_shadow_test_image();
    let primary_msg = "innocent";
    let primary_pass = "primary-pass";
    let shadow_msg = "secret";
    let shadow_pass = "shadow-pass";

    let stego = ghost_encode_with_shadows(
        &cover,
        primary_msg,
        &[],
        primary_pass,
        &[ShadowLayer {
            message: shadow_msg.to_string(),
            passphrase: shadow_pass.to_string(),
            files: vec![],
        }],
        None,
    ).unwrap();

    // Primary passphrase decodes primary message.
    let primary_decoded = ghost_decode(&stego, primary_pass).unwrap();
    assert_eq!(primary_decoded.text, primary_msg);

    // Shadow passphrase decodes shadow message.
    let shadow_decoded = ghost_shadow_decode(&stego, shadow_pass).unwrap();
    assert_eq!(shadow_decoded.text, shadow_msg);
}

#[test]
fn shadow_two_layers() {
    let cover = load_shadow_test_image();
    // Primary must be largest to avoid auto-sort swapping with a shadow.
    let primary_msg = "primary message that is longer than both shadows";
    let primary_pass = "primary";
    let shadow1_msg = "shadow one";
    let shadow1_pass = "shadow-1";
    let shadow2_msg = "shadow two";
    let shadow2_pass = "shadow-2";

    let stego = ghost_encode_with_shadows(
        &cover,
        primary_msg,
        &[],
        primary_pass,
        &[
            ShadowLayer {
                message: shadow1_msg.to_string(),
                passphrase: shadow1_pass.to_string(),
                files: vec![],
            },
            ShadowLayer {
                message: shadow2_msg.to_string(),
                passphrase: shadow2_pass.to_string(),
                files: vec![],
            },
        ],
        None,
    ).unwrap();

    let primary_decoded = ghost_decode(&stego, primary_pass).unwrap();
    assert_eq!(primary_decoded.text, primary_msg);

    let shadow1_decoded = ghost_shadow_decode(&stego, shadow1_pass).unwrap();
    assert_eq!(shadow1_decoded.text, shadow1_msg);

    let shadow2_decoded = ghost_shadow_decode(&stego, shadow2_pass).unwrap();
    assert_eq!(shadow2_decoded.text, shadow2_msg);
}

#[test]
fn shadow_wrong_pass() {
    let cover = load_shadow_test_image();

    let stego = ghost_encode_with_shadows(
        &cover,
        "primary",
        &[],
        "primary-pass",
        &[ShadowLayer {
            message: "shadow".to_string(),
            passphrase: "shadow-pass".to_string(),
            files: vec![],
        }],
        None,
    ).unwrap();

    let result = ghost_shadow_decode(&stego, "unknown-pass");
    assert!(result.is_err(), "unknown passphrase should fail");
}

#[test]
fn shadow_smart_decode() {
    let cover = load_shadow_test_image();
    let shadow_msg = "via smart";
    let shadow_pass = "shadow-smart";

    let stego = ghost_encode_with_shadows(
        &cover,
        "primary",
        &[],
        "primary-pass",
        &[ShadowLayer {
            message: shadow_msg.to_string(),
            passphrase: shadow_pass.to_string(),
            files: vec![],
        }],
        None,
    ).unwrap();

    // smart_decode with shadow passphrase should find the shadow message.
    let (decoded, _quality) = smart_decode(&stego, shadow_pass).unwrap();
    assert_eq!(decoded.text, shadow_msg);
}

#[test]
fn shadow_primary_unchanged() {
    let cover = load_shadow_test_image();
    let primary_msg = "test unchanged";
    let primary_pass = "test-pass";

    // Encode without shadows.
    let stego_plain = ghost_encode(&cover, primary_msg, primary_pass).unwrap();
    let decoded_plain = ghost_decode(&stego_plain, primary_pass).unwrap();

    // Encode with shadows.
    let stego_shadow = ghost_encode_with_shadows(
        &cover,
        primary_msg,
        &[],
        primary_pass,
        &[ShadowLayer {
            message: "shadow".to_string(),
            passphrase: "shadow-pass".to_string(),
            files: vec![],
        }],
        None,
    ).unwrap();
    let decoded_shadow = ghost_decode(&stego_shadow, primary_pass).unwrap();

    // Primary message is identical in both cases.
    assert_eq!(decoded_plain.text, decoded_shadow.text);
    assert_eq!(decoded_shadow.text, primary_msg);
}

#[test]
fn shadow_too_large() {
    // Use the smaller 320x240 image.
    let cover = load_test_image("photo_320x240_q75_420.jpg");

    let img = JpegImage::from_bytes(&cover).unwrap();
    let cap = estimate_shadow_capacity(&img).unwrap();
    // Generate incompressible data using a simple LCG to defeat Brotli compression.
    // Compressed size ≈ raw size, so this will exceed shadow capacity.
    let mut rng_state = 0x12345678u64;
    let oversized_msg: String = (0..(cap + 100))
        .map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            char::from(b'!' + ((rng_state >> 33) % 94) as u8)
        })
        .collect();

    let result = ghost_encode_with_shadows(
        &cover,
        "primary",
        &[],
        "primary-pass",
        &[ShadowLayer {
            message: oversized_msg,
            passphrase: "shadow-pass".to_string(),
            files: vec![],
        }],
        None,
    );

    assert!(result.is_err(), "oversized shadow should fail");
}

#[test]
fn shadow_empty_passphrase() {
    let cover = load_shadow_test_image();

    let stego = ghost_encode_with_shadows(
        &cover,
        "primary",
        &[],
        "primary-pass",
        &[ShadowLayer {
            message: "hi".to_string(),
            passphrase: "".to_string(),
            files: vec![],
        }],
        None,
    ).unwrap();

    let shadow_decoded = ghost_shadow_decode(&stego, "").unwrap();
    assert_eq!(shadow_decoded.text, "hi");
}

#[test]
fn shadow_duplicate_passphrase() {
    let cover = load_shadow_test_image();

    // Primary and shadow have the same passphrase.
    let result = ghost_encode_with_shadows(
        &cover,
        "primary",
        &[],
        "same-pass",
        &[ShadowLayer {
            message: "shadow".to_string(),
            passphrase: "same-pass".to_string(),
            files: vec![],
        }],
        None,
    );

    assert!(
        matches!(result, Err(StegoError::DuplicatePassphrase)),
        "duplicate passphrases should be rejected"
    );

    // Two shadows with the same passphrase.
    let result2 = ghost_encode_with_shadows(
        &cover,
        "primary",
        &[],
        "primary-pass",
        &[
            ShadowLayer {
                message: "s1".to_string(),
                passphrase: "dup".to_string(),
                files: vec![],
            },
            ShadowLayer {
                message: "s2".to_string(),
                passphrase: "dup".to_string(),
                files: vec![],
            },
        ],
        None,
    );

    assert!(
        matches!(result2, Err(StegoError::DuplicatePassphrase)),
        "duplicate shadow passphrases should be rejected"
    );
}

#[test]
fn shadow_no_shadows() {
    let cover = load_shadow_test_image();
    let msg = "test no shadows";
    let pass = "pass";

    let stego = ghost_encode_with_shadows(&cover, msg, &[], pass, &[], None).unwrap();
    let decoded = ghost_decode(&stego, pass).unwrap();
    assert_eq!(decoded.text, msg);
}

#[test]
fn shadow_capacity_estimate() {
    let cover = load_shadow_test_image();
    let img = JpegImage::from_bytes(&cover).unwrap();
    let cap = estimate_shadow_capacity(&img).unwrap();

    // With 100% pool: capacity should be large.
    // 640x480 has ~300K nzAC positions, 100% pool -> many KB.
    assert!(cap > 1000, "shadow capacity should be > 1KB, got {cap}");

    // Grayscale: still has Y channel, so V2 should have capacity.
    let gray = load_test_image("gray_64x64_q75.jpg");
    let gray_img = JpegImage::from_bytes(&gray).unwrap();
    let gray_cap = estimate_shadow_capacity(&gray_img).unwrap();
    assert!(gray_cap >= 0, "grayscale capacity should be non-negative");
}

#[test]
fn shadow_coexistence() {
    // Verify that shadow + primary coexist correctly with ∞-cost protection.
    let cover = load_shadow_test_image();
    let primary_msg = "primary message for coexistence test";
    let shadow_msg = "shadow message for coexistence test";

    let stego = ghost_encode_with_shadows(
        &cover,
        primary_msg,
        &[],
        "coex-primary",
        &[ShadowLayer {
            message: shadow_msg.to_string(),
            passphrase: "coex-shadow".to_string(),
            files: vec![],
        }],
        None,
    ).unwrap();

    // Both must decode correctly.
    let primary = ghost_decode(&stego, "coex-primary").unwrap();
    assert_eq!(primary.text, primary_msg);

    let shadow = ghost_shadow_decode(&stego, "coex-shadow").unwrap();
    assert_eq!(shadow.text, shadow_msg);
}

#[test]
fn shadow_dynamic_w_small_message() {
    // Small primary message should get high w, reducing modifications.
    // Auto-sort may route the larger shadow to STC and the smaller primary
    // to shadow channel, so use smart_decode which tries both paths.
    let cover = load_shadow_test_image();
    let primary_msg = "hi"; // very small
    let shadow_msg = "secret shadow";

    let stego = ghost_encode_with_shadows(
        &cover,
        primary_msg,
        &[],
        "dyn-primary",
        &[ShadowLayer {
            message: shadow_msg.to_string(),
            passphrase: "dyn-shadow".to_string(),
            files: vec![],
        }],
        None,
    ).unwrap();

    // smart_decode finds the message regardless of internal routing.
    let (primary, _) = smart_decode(&stego, "dyn-primary").unwrap();
    assert_eq!(primary.text, primary_msg);

    let (shadow, _) = smart_decode(&stego, "dyn-shadow").unwrap();
    assert_eq!(shadow.text, shadow_msg);
}

#[test]
fn shadow_backward_compat_no_shadow() {
    // Encode without shadows (new short STC + dynamic w), decode should work.
    let cover = load_shadow_test_image();
    let msg = "backward compat test";
    let pass = "compat-pass";

    let stego = ghost_encode(&cover, msg, pass).unwrap();
    let decoded = ghost_decode(&stego, pass).unwrap();
    assert_eq!(decoded.text, msg);
}
