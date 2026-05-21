// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Cross-platform empirical optimizer-output byte equivalence (T2.11).
//!
//! The deterministic 256×256 RGB input fed through `optimize_cover`
//! (Ghost pipeline) is hashed via SHA256; every supported architecture
//! MUST produce the same hash.
//!
//! The hash is pinned in two stages:
//!
//!   1. PRE-T2.11 baseline (recorded before any T2.11 changes) — catches
//!      regressions on platforms that should already be bit-identical.
//!   2. POST-T2.11 baseline (recorded after each T2.11 sub-task lands) —
//!      catches per-arch divergence inside the new code (integral
//!      image, rayon parallelism, noise pre-generation).
//!
//! Determinism constraint: optimizer output MUST be bit-identical
//! across native aarch64 / x86_64 / iOS / Android / WASM SIMD128.
//! Same input + seed = same output bytes, period.

use phasm_core::stego::optimizer_test_hash_hex;

/// Hash recorded on aarch64 NEON, pre-T2.11a (2026-05-21).
const EXPECTED_HASH: &str =
    "0e6fab9f4b10b76298c043e3d20edeb330092a2ef7de93395e9b39a1591dcc50";

#[test]
fn optimizer_output_is_platform_invariant() {
    let actual = optimizer_test_hash_hex();
    eprintln!("optimizer 256×256 RGB Ghost hash on this platform: {}", actual);
    if EXPECTED_HASH == "REPLACE_ME_WITH_AARCH64_HASH" {
        panic!(
            "EXPECTED_HASH not yet pinned. \
            Set EXPECTED_HASH = \"{}\" in core/tests/optimizer_cross_platform.rs.",
            actual
        );
    }
    assert_eq!(
        actual, EXPECTED_HASH,
        "optimize_cover output diverged on this platform — T2.11 broke Path A invariant"
    );
}
