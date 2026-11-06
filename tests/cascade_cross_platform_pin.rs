// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Cross-platform encode pins: parallel and no-parallel feature builds (and
//! aarch64 / x86_64 / WASM) produce the same Ghost encode bytes for the
//! same input — a Ghost v3 primary, and a primary with two shadows.
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

use phasm_core::{ghost_encode, ghost_encode_with_shadows, ShadowLayer};
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
///
/// History:
/// - `4a149432465e390e…` — v1 J-UNIWARD costs (phasm-core ≤ 0.4.x).
/// - `fc67d9eec489ce2a…` (2026-10-25) — v2 costs (whole-window border
///   pricing, window aligned with the impact; `docs/design/image/
///   juniward-cost.md`). Every primary-STC cost changed, so the flips —
///   and the bytes — did. Same hash with and without `parallel`.
/// - `d9451b982931a4ee…` (2026-10-25) — shadow layout v2 (keyed-region
///   STC, `docs/design/image/shadow-messages.md`): the two shadows now
///   embed by STC (width 10) instead of direct LSB, and the primary's wet
///   set is their regions. Identical serial / `parallel`, and on aarch64
///   (NEON) and x86_64 (Rosetta, SSE2). (Never released.)
/// - `0f2802224f275acc…` (2026-10-27) — Ghost v3
///   (`docs/design/image/ghost-embedding-v3.md`): ternary STC over all Y AC
///   coefficients (STC height 9), shadows on keyed regions of the all-AC set.
/// - `bd1c113c411094bc…` (2026-11-03) — Ghost v3.1 (keyed pairs,
///   mixed-width STCs, costs on the unrounded image; the shadow layout is
///   unchanged, its costs and the primary moved).
const EXPECTED_SHADOW_ENCODE_HASH: &str =
    "bd1c113c411094bc4cb25f517767186611e0909b95e0367361b92657b47e7264";

/// Ghost v3.1 primary, no shadows (same seed, cover and passphrase as the
/// shadow pin's primary).
/// v3: `0efed1f4b986dae6…`; v3.1 (2026-11-03) below.
const EXPECTED_PRIMARY_ENCODE_HASH: &str =
    "a49f8969aa7437356f927646d28d92283e65f1fa522fc0909eee67d3dcb7d656";

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn primary_encode_hash_matches_cross_platform_pin() {
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", DETERMINISTIC_SEED); }
    let stego = ghost_encode(&load_cover(), &"primary message, pinned ".repeat(20), "primary-pass-m6")
        .expect("ghost_encode");
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED"); }
    assert_eq!(hex(&stego), EXPECTED_PRIMARY_ENCODE_HASH, "Ghost v3.1 primary encode changed; re-pin only if intended");
}

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
