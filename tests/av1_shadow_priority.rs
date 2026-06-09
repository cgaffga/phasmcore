// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! F.1 + F.2 — shadow priority + shared infrastructure verification.
//!
//! F.1: position-priority unit tests live in-module
//! (`core/src/codec/av1/stego/shadow.rs`).
//!
//! F.2: integration tests that the shared `crate::stego::shadow_layer`
//! (WIDE u32 BE shadow frame) + `crate::stego::armor::ecc`
//! (Reed-Solomon) + `crate::stego::crypto::derive_shadow_structural_key`
//! infrastructure is reachable + behaves correctly for AV1's intended
//! use. F.3+ depend on these.

#![cfg(any(feature = "av1-encoder", feature = "av1-decoder"))]

use phasm_core::codec::av1::stego::shadow::{
    build_av1_shadow_frame, parse_av1_shadow_frame, priority_slots, AV1_SHADOW_FRAME_OVERHEAD,
    AV1_SHADOW_PARITY_TIERS, MAX_AV1_SHADOW_FRAME_BYTES,
};

#[test]
fn f2_shadow_frame_round_trip_various_sizes() {
    // Verify the WIDE u32 BE shadow frame layout round-trips clean
    // across the size envelope shadow needs to carry:
    // - 1 byte: degenerate single-char message
    // - 100 bytes: typical text message
    // - 4 KiB: small file attachment
    // - 256 KiB: large file attachment (well below MAX 16 MiB)
    for size in [1usize, 100, 4 * 1024, 256 * 1024] {
        let plaintext_len = size;
        let salt = [0xABu8; 16];
        let nonce = [0xCDu8; 12];
        // Build_av1_shadow_frame expects ciphertext including the 16-byte
        // AES-GCM auth tag, so ciphertext.len() = plaintext_len + 16.
        let ciphertext = vec![0xEEu8; plaintext_len + 16];
        let framed = build_av1_shadow_frame(plaintext_len, &salt, &nonce, &ciphertext);
        assert!(
            framed.len() >= AV1_SHADOW_FRAME_OVERHEAD + plaintext_len,
            "framed too small for plaintext_len={plaintext_len}"
        );
        let parsed = parse_av1_shadow_frame(&framed)
            .unwrap_or_else(|e| panic!("parse_av1_shadow_frame failed at size {size}: {e:?}"));
        assert_eq!(parsed.plaintext_len as usize, plaintext_len);
        assert_eq!(parsed.salt, salt);
        assert_eq!(parsed.nonce, nonce);
        assert_eq!(parsed.ciphertext, ciphertext);
    }
}

#[test]
fn f2_shadow_frame_overhead_constant_is_48() {
    // WIDE: 4-byte plaintext_len + 16 salt + 12 nonce + 16 GCM tag = 48.
    // Used by F.3's capacity math + F.4's brute-force lower bound.
    assert_eq!(
        AV1_SHADOW_FRAME_OVERHEAD, 48,
        "AV1_SHADOW_FRAME_OVERHEAD must match the WIDE shadow frame layout"
    );
}

#[test]
fn f2_parity_tiers_match_h264_shape() {
    // SHADOW_PARITY_TIERS = [4, 8, 16, 32, 64, 128] — F.4's brute-force
    // decode iterates these. Larger tiers = more error tolerance but
    // less payload. Same ladder H.264 ships (`shadow.rs`-side
    // SHADOW_PARITY_TIERS).
    assert_eq!(AV1_SHADOW_PARITY_TIERS, [4, 8, 16, 32, 64, 128]);
}

#[test]
fn f2_max_shadow_frame_bytes_accommodates_video_attachments() {
    // 16 MiB. Phasm's "5-min video could be some MBs ZIP file"
    // constraint per CLAUDE.md → ≥ 4 MiB minimum. WIDE format gives
    // us 16 MiB headroom.
    assert!(
        MAX_AV1_SHADOW_FRAME_BYTES >= 4 * 1024 * 1024,
        "shadow frame ceiling {MAX_AV1_SHADOW_FRAME_BYTES} < 4 MiB target"
    );
}

#[test]
fn f2_rs_round_trip_at_each_parity_tier() {
    // Phase F.4 decode iterates SHADOW_PARITY_TIERS to find the right
    // parity. Verify clean round-trip at each tier without error
    // injection (the trivial case — F.4 tests inject corrupt bits to
    // exercise RS error correction).
    use phasm_core::stego::armor::ecc::{rs_decode_with_parity, rs_encode_blocks_with_parity};

    let payload = b"phasm AV1 shadow F.2 RS round-trip exhaustive parity tier sweep";
    for &parity_len in &AV1_SHADOW_PARITY_TIERS {
        let encoded = rs_encode_blocks_with_parity(payload, parity_len);
        let block_len = payload.len() + parity_len;
        assert_eq!(
            encoded.len(),
            block_len,
            "encoded length mismatch for parity={parity_len}"
        );
        let (decoded, errors) = rs_decode_with_parity(&encoded, payload.len(), parity_len)
            .expect("rs_decode_with_parity clean round-trip");
        assert_eq!(decoded.as_slice(), payload);
        assert_eq!(errors, 0, "no errors injected, decoder reported {errors}");
    }
}

#[test]
fn f2_rs_corrects_below_threshold_errors() {
    // Each parity tier should correct up to floor(parity/2) byte errors.
    // F.4's primary purpose is enabling shadow-bit corruption tolerance
    // (collisions with other shadows, single-bit decode noise, etc.).
    use phasm_core::stego::armor::ecc::{rs_decode_with_parity, rs_encode_blocks_with_parity};

    let payload = vec![0x42u8; 200];
    let parity_len = 32usize;
    let max_correctable = parity_len / 2; // 16

    for n_errors in [0usize, 1, 8, max_correctable] {
        let mut encoded = rs_encode_blocks_with_parity(&payload, parity_len);
        // Corrupt the first n_errors bytes.
        for i in 0..n_errors {
            encoded[i] = encoded[i].wrapping_add(0x80);
        }
        let (decoded, errs) =
            rs_decode_with_parity(&encoded, payload.len(), parity_len)
                .unwrap_or_else(|e| panic!("RS failed at {n_errors} errors: {e:?}"));
        assert_eq!(decoded, payload);
        assert_eq!(errs, n_errors);
    }
}

#[test]
fn f2_derive_shadow_key_differs_per_passphrase() {
    use phasm_core::stego::crypto::derive_shadow_structural_key;
    let a = derive_shadow_structural_key("passphrase-A").unwrap();
    let b = derive_shadow_structural_key("passphrase-B").unwrap();
    // Both 32 bytes.
    assert_ne!(
        a.as_slice(),
        b.as_slice(),
        "shadow keys must differ across passphrases"
    );
    // Same passphrase → same key (determinism).
    let a2 = derive_shadow_structural_key("passphrase-A").unwrap();
    assert_eq!(a.as_slice(), a2.as_slice());
}

#[test]
fn f1_priority_slots_re_exported_and_usable() {
    // Integration-level smoke check: the F.1 `priority_slots` is
    // pub-re-exported from av1::stego::shadow and accepts the
    // expected signature. F.3+ build on this.
    let slots = priority_slots(50_000, &[0u8; 32]);
    assert_eq!(slots.len(), 50_000);
    assert!(slots[0].priority <= slots[slots.len() - 1].priority);
}
