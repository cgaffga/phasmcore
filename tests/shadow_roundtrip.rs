// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Round-trip integration tests for Ghost shadow messages (plausible deniability).

use phasm_core::{
    ghost_encode, ghost_decode, ghost_encode_with_shadows,
    ghost_shadow_decode, smart_decode, ShadowLayer, StegoError,
    estimate_shadow_capacity, JpegImage,
};

fn load_test_image(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/{name}")).unwrap()
}

/// Use the larger test image that has enough chrominance capacity for shadows.
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
    // Two shadows need enough chrominance capacity for both.
    // Use the large original photo which has ample Cb+Cr positions.
    let path = "test-vectors/photo_original_3557.jpg";
    if !std::path::Path::new(path).exists() {
        eprintln!("skipping shadow_two_layers: {path} not found");
        return;
    }
    let cover = std::fs::read(path).unwrap();
    let primary_msg = "public";
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
    // Use the smaller 320x240 image and a message that exceeds shadow capacity.
    // Shadow capacity = usable_chroma_positions / R(7) / 8 - frame_overhead.
    // A 320x240 4:2:0 image has limited chrominance, so a ~4KB message should overflow.
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let oversized_msg = "X".repeat(4096);

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

    // 640x480 4:2:0 should have some shadow capacity from Cb+Cr.
    assert!(cap > 0, "shadow capacity should be positive, got {cap}");

    // Grayscale should have 0 shadow capacity.
    let gray = load_test_image("gray_64x64_q75.jpg");
    let gray_img = JpegImage::from_bytes(&gray).unwrap();
    let gray_cap = estimate_shadow_capacity(&gray_img).unwrap();
    assert_eq!(gray_cap, 0, "grayscale should have 0 shadow capacity");
}
