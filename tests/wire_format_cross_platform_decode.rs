// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Wire-format cross-platform decode gate.
//!
//! ## Why this exists
//!
//! M9.5.C1/C2 (2026-05-22) relaxed the optimizer's per-pixel
//! cross-platform bit-exactness in exchange for memory + perf wins.
//! Different platforms now produce different OPTIMIZED COVER bytes
//! from the same input.
//!
//! The contract that **still must hold**: stego JPEGs produced on
//! any platform decode correctly on any other platform. The
//! pre-flight audit (#678) confirmed no code path derives keys,
//! salts, or seeds from optimizer output bytes — the encryption
//! salt is per-encode `rand::thread_rng()` embedded in the stego
//! frame, structural keys are `passphrase + FIXED_SALT`. So
//! different platforms produce different stego JPEG BYTES but the
//! decoder can extract the message from any of them.
//!
//! This test validates that contract by checking in stego JPEG
//! fixtures encoded on a known platform and asserting decode
//! succeeds on the CURRENT platform regardless of arch. As CI matrix
//! adds aarch64 / x86_64 / Rosetta 2 / WASM runners, the SAME
//! checked-in fixtures get decoded on each — proving the wire format
//! truly is cross-platform.
//!
//! ## How to add a fixture from a new platform
//!
//! 1. On the new platform, run the helper at the bottom of this file
//!    via `cargo test wire_format_print_fixture_bytes -- --nocapture
//!    --ignored`. It encodes a known message + passphrase with the
//!    `PHASM_DETERMINISTIC_SEED=42` env set so the random salt+nonce
//!    are reproducible.
//! 2. Save the printed bytes to `test-vectors/image/wire-format/
//!    ghost_<arch>_<date>.jpg`.
//! 3. Add the path + expected metadata to `FIXTURES` below.

use phasm_core::stego::{ghost_decode, ghost_encode, smart_decode};
// `mode` field on DecodeQuality is u8 — 1 = Ghost, 2 = Armor.
const MODE_GHOST: u8 = 1;

const TEST_PASSPHRASE: &str = "wire-format-cross-platform-test";
const TEST_MESSAGE: &str = "Hello from the wire-format gate!";

/// Each fixture: (path relative to phasm root, expected mode_id from smart_decode).
///
/// `mode_id == 1` is Ghost; `mode_id == 2` is Armor. Fixtures are
/// checked into `test-vectors/image/wire-format/` and committed to
/// the repo; the CI matrix decodes them on every supported arch.
const FIXTURES: &[(&str, &str, u8)] = &[
    // Format: (relative_path, expected_message, expected_mode_id)
    //
    // 2026-05-22: aarch64 fixture captured on M-series Apple Silicon
    // (Mac Studio M2 Max). PHASM_DETERMINISTIC_SEED=42 fixed the
    // per-encode random salt+nonce so the stego bytes are reproducible.
    // Cover: photo_640x480_q75_420.jpg (640×480 Q75 4:2:0 JPEG).
    // Captured stego size: 45,936 bytes (Ghost mode).
    (
        "test-vectors/image/wire-format/ghost_aarch64_2026_05_22.jpg",
        TEST_MESSAGE,
        MODE_GHOST,
    ),
];

#[test]
fn wire_format_fixtures_decode_on_this_arch() {
    if FIXTURES.is_empty() {
        eprintln!(
            "[INFO] No wire-format fixtures committed yet. Run\n  \
             cargo test --test wire_format_cross_platform_decode \
             wire_format_capture_aarch64_fixture -- --ignored --nocapture\n\
             on the dev machine to generate one."
        );
        return;
    }
    for (path, expected_msg, expected_mode_id) in FIXTURES {
        let stego_bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("could not read fixture {path}: {e}"));
        let (payload, quality) = smart_decode(&stego_bytes, TEST_PASSPHRASE)
            .unwrap_or_else(|e| panic!("decode failed for fixture {path}: {e:?}"));
        assert_eq!(
            payload.text, *expected_msg,
            "fixture {path}: decoded message mismatch"
        );
        assert_eq!(
            quality.mode, *expected_mode_id,
            "fixture {path}: detected mode mismatch"
        );
    }
}

/// Sanity check that the local encode+decode round-trip works on the
/// current platform. Catches in-process regressions independently of
/// the checked-in fixtures.
#[test]
fn wire_format_local_roundtrip() {
    let cover_path = "test-vectors/image/photo_640x480_q75_420.jpg";
    let cover_bytes = match std::fs::read(cover_path) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("[skip] test-vectors/image/jpeg_compatibility_test.jpg not present");
            return;
        }
    };

    // Encode with a fixed deterministic seed so the test is
    // reproducible. The seed env affects the per-encode random salt
    // + nonce only — same input + same env → same stego bytes on
    // the same platform.
    // SAFETY: setting env in a single-threaded test is safe; std test
    // harness gives each test its own thread but env is process-wide.
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42") };
    let stego = ghost_encode(&cover_bytes, TEST_MESSAGE, TEST_PASSPHRASE)
        .expect("ghost_encode should succeed");
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };

    let payload = ghost_decode(&stego, TEST_PASSPHRASE)
        .expect("ghost_decode should succeed on locally-encoded stego");
    assert_eq!(
        payload.text, TEST_MESSAGE,
        "local round-trip message mismatch"
    );

    let (smart_payload, smart_quality) =
        smart_decode(&stego, TEST_PASSPHRASE).expect("smart_decode should succeed");
    assert_eq!(smart_payload.text, TEST_MESSAGE);
    assert_eq!(
        smart_quality.mode, MODE_GHOST,
        "smart_decode should detect Ghost mode"
    );
}

/// Capture helper — encodes the standard test message with the
/// fixed seed and prints the bytes (length + first/last 64 bytes
/// for sanity, full bytes to a file if PHASM_FIXTURE_OUT is set).
///
/// Marked `#[ignore]` because it's a fixture-generation tool, not a
/// test. Run with:
///
///     PHASM_FIXTURE_OUT=test-vectors/image/wire-format/ghost_aarch64_2026_05_22.jpg \
///     cargo test --test wire_format_cross_platform_decode \
///     wire_format_capture_aarch64_fixture -- --ignored --nocapture
#[test]
#[ignore = "fixture generation helper; run with PHASM_FIXTURE_OUT"]
fn wire_format_capture_aarch64_fixture() {
    let cover_path = "test-vectors/image/photo_640x480_q75_420.jpg";
    let cover_bytes = std::fs::read(cover_path).expect("cover JPEG must exist");

    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42") };
    let stego = ghost_encode(&cover_bytes, TEST_MESSAGE, TEST_PASSPHRASE)
        .expect("ghost_encode should succeed");
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };

    eprintln!("=== Wire-format fixture capture ===");
    eprintln!("arch         : {}", std::env::consts::ARCH);
    eprintln!("passphrase   : {}", TEST_PASSPHRASE);
    eprintln!("message      : {}", TEST_MESSAGE);
    eprintln!("stego bytes  : {} bytes", stego.len());

    if let Ok(out_path) = std::env::var("PHASM_FIXTURE_OUT") {
        std::fs::write(&out_path, &stego).expect("write fixture");
        eprintln!("Wrote fixture to {out_path}");
        eprintln!("Add to FIXTURES table in test source:");
        eprintln!(
            "  (\"{out_path}\", TEST_MESSAGE, 1),"
        );
    } else {
        eprintln!("Set PHASM_FIXTURE_OUT=<path> to write the fixture to disk.");
    }
}
