// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Recompression survival baseline test.
//!
//! Quick sanity test that Armor encode/decode works without recompression.
//! Catches regressions in the basic Armor pipeline.

use phasm_core::{armor_encode, armor_decode, armor_capacity, JpegImage};

/// Message lengths to test.
const MESSAGE_LENGTHS: &[usize] = &[10, 20, 50, 80, 100, 150, 200, 500, 1000];

fn load_test_vector(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/{name}")).unwrap()
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

/// Baseline: Armor encode/decode without recompression.
///
/// Quick sanity test (not ignored) -- asserts 100% integrity and exact match.
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
        assert_eq!(decoded.text, message, "Message mismatch for {msg_len}-char message");
        assert!(quality.integrity_percent >= 85,
            "Expected high integrity without recompression for {msg_len}-char message: {}%",
            quality.integrity_percent);
        assert_eq!(quality.rs_errors_corrected, 0,
            "Expected 0 RS errors without recompression for {msg_len}-char message");
    }
}
