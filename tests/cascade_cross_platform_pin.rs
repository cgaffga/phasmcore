// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! M6 phase gate: parallel and no-parallel feature builds produce the
//! same Ghost shadow encode bytes for the same input.
//!
//! Pre-M6 the serial cascade (`#[cfg(not(feature = "parallel"))]`)
//! used a greedy "first success in CASCADE order" algorithm, while the
//! parallel branch used "collect all per parity, pick largest fraction".
//! That meant CLI builds without `--features parallel` could produce
//! different stego bytes than iOS / Android / WASM (all `parallel`) for
//! cascade-triggering inputs. M6 unifies the serial branch to the
//! parallel algorithm — this test pins that.
//!
//! Run this under BOTH:
//!   cargo test --test cascade_cross_platform_pin                (no parallel)
//!   cargo test --features parallel --test cascade_cross_platform_pin
//!
//! If either run produces a hash that doesn't match
//! `EXPECTED_SHADOW_ENCODE_HASH` below, the two paths have diverged.

use phasm_core::{ghost_encode_with_shadows, ShadowLayer};
use sha2::{Digest, Sha256};

const DETERMINISTIC_SEED: &str = "42";

/// Pinned cover: tracked in repo at `core/test-vectors/image/`.
fn load_cover() -> Vec<u8> {
    std::fs::read("test-vectors/image/photo_640x480_q75_420.jpg").unwrap()
}

/// Encode a fixed shadow-bearing payload with PHASM_DETERMINISTIC_SEED
/// set so the AES-GCM-SIV salt + nonce are deterministic. Returns the
/// SHA256 hex of the stego bytes.
fn encode_and_hash() -> String {
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", DETERMINISTIC_SEED); }
    let cover = load_cover();
    let stego = ghost_encode_with_shadows(
        &cover,
        "primary message that is longer than each shadow",
        &[],
        "primary-pass-m6",
        &[
            ShadowLayer {
                message: "shadow one".to_string(),
                passphrase: "shadow-pass-m6-1".to_string(),
                files: vec![],
            },
            ShadowLayer {
                message: "shadow two".to_string(),
                passphrase: "shadow-pass-m6-2".to_string(),
                files: vec![],
            },
        ],
        None,
    )
    .expect("ghost_encode_with_shadows");
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED"); }

    let mut h = Sha256::new();
    h.update(&stego);
    let digest = h.finalize();
    let mut s = String::with_capacity(64);
    for b in digest.iter() {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// The expected hash. If this test fails on the first run after the
/// commit, the hash needs to be re-pinned to whatever the test
/// produces. After re-pinning, the test must be GREEN under both
/// `cargo test` (no parallel) and `cargo test --features parallel`.
const EXPECTED_SHADOW_ENCODE_HASH: &str =
    "4a149432465e390e9a23e343e4be7d9cf66ed354c793a4a5df142dc96262691f";

#[test]
fn shadow_encode_hash_matches_cross_platform_pin() {
    let hash = encode_and_hash();
    if EXPECTED_SHADOW_ENCODE_HASH.starts_with("TBD") {
        // First run: print the actual hash so we can pin it.
        // Subsequent runs assert.
        panic!(
            "M6 hash pin not yet set. Run produced:\n  {}\n\
             Update EXPECTED_SHADOW_ENCODE_HASH in cascade_cross_platform_pin.rs \
             to this value, then re-run under BOTH `cargo test` and \
             `cargo test --features parallel` — both must GREEN.",
            hash
        );
    }
    assert_eq!(
        hash, EXPECTED_SHADOW_ENCODE_HASH,
        "Ghost shadow encode SHA256 changed. Either: (a) M6 serial/parallel \
         cascade unification regressed; (b) some other encoder change shifted \
         output. If (b) is intended, re-pin the hash.",
    );
}
