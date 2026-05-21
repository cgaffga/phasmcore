// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Cross-platform empirical FFT-output byte equivalence (T2.3).
//!
//! The deterministic 256×256 FFT (`fft2d_test_deterministic_bytes`) is
//! seeded with a fixed LCG so every IEEE 754-conformant f32
//! implementation MUST produce identical output bytes.
//!
//! This test asserts the SHA256 of the output matches a hardcoded
//! value first computed on aarch64 NEON. The same binary is also run
//! under x86_64 SSE via Rosetta 2 and inside WASM SIMD128 under V8 to
//! confirm Path A's structural bit-equivalence claim empirically.
//!
//! If the hash diverges on any platform → the SIMD butterfly has
//! introduced a per-lane non-IEEE-754 op (e.g. FMA fusion, reduction
//! reordering) and Path A is broken.

use phasm_core::stego::armor::fft2d::fft2d_test_hash_hex;

/// Hash recorded on aarch64 NEON, 2026-05-21.
/// Re-run on every supported arch to confirm bit-equivalence.
const EXPECTED_HASH: &str =
    "17995f36918a37049160f1bd9a393f20676506c464eceafe0f731efbc83db31e";

#[test]
fn fft2d_bytes_are_platform_invariant() {
    let actual = fft2d_test_hash_hex();
    eprintln!("FFT2D 256x256 hash on this platform: {}", actual);
    if EXPECTED_HASH == "REPLACE_ME_WITH_AARCH64_HASH" {
        // Bootstrap mode: not yet recorded. Print + fail with hint.
        panic!(
            "EXPECTED_HASH not yet pinned. \
            Set EXPECTED_HASH = \"{}\" in core/tests/fft2d_cross_platform.rs.",
            actual
        );
    }
    assert_eq!(
        actual, EXPECTED_HASH,
        "FFT2D bytes diverged on this platform — SIMD butterfly broke Path A invariant"
    );
}
