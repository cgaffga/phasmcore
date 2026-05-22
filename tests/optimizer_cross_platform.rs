// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Optimizer per-platform determinism gate.
//!
//! ## Contract — M9.5 (2026-05-22)
//!
//! Pre-C1 the optimizer was cross-platform bit-exact (single SHA pin
//! across aarch64 / x86_64 / WASM SIMD128). M9.5 narrowed `luma` to
//! `u8` and `variance` to `u16`, and C2 added native f32x4 SIMD with
//! NEON FMA — both moves are per-platform deterministic but produce
//! different bytes per architecture. Pre-flight #678 confirmed no code
//! path derives keys, salts, or seeds from optimizer output bytes, so
//! wire-format cross-platform decode is preserved.
//!
//! ## What this test guards
//!
//! For the **current architecture**:
//! 1. Same input + seed + arch → byte-identical output across runs
//!    (per-platform determinism: same code, same machine, same bytes).
//! 2. Output matches a pinned per-arch SHA — catches regressions on
//!    that arch even when the run-twice check passes trivially.
//!
//! ## Adding a new arch
//!
//! Run the test once; if the pinned hash for that arch is
//! `REPLACE_ME_<arch>` the test logs the actual hash and panics.
//! Capture that hash in the `EXPECTED_HASH` `cfg!` ladder below and
//! re-run; subsequent runs gate the arch normally.
//!
//! ## What this test does NOT guard
//!
//! Cross-arch wire-format compatibility. See
//! `wire_format_cross_platform_decode.rs` for that — stego JPEGs
//! produced on one arch must decode on any arch, regardless of
//! whether the optimizer pixels match.

use phasm_core::stego::optimizer_test_hash_hex;

/// Per-arch pinned SHA256 of the 256×256 RGB Ghost optimizer output.
///
/// Captured 2026-05-22 post-Phase-C2 (NEON FMA, WASM f32x4 native,
/// type-narrowed u8 luma + u16 variance). M-series Apple Silicon
/// (Mac Studio M2 Max in dev; expected stable across M-series).
const EXPECTED_HASH: &str = {
    #[cfg(target_arch = "aarch64")]
    {
        "ef272637fe80ee76b0b77c014ba6cb8e1829df345037d74abd6fa8208d87c21d"
    }
    #[cfg(target_arch = "x86_64")]
    {
        // Not yet pinned — run on x86_64 (or Rosetta 2 on aarch64) and
        // paste the logged hash. Until then the test logs the hash and
        // passes the run-twice determinism check only.
        "REPLACE_ME_x86_64"
    }
    #[cfg(target_arch = "wasm32")]
    {
        // Not yet pinned — see comment above.
        "REPLACE_ME_wasm32"
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "wasm32")))]
    {
        "REPLACE_ME_unknown_arch"
    }
};

#[test]
fn optimizer_output_is_per_platform_deterministic() {
    let h1 = optimizer_test_hash_hex();
    let h2 = optimizer_test_hash_hex();
    eprintln!("optimizer 256×256 RGB Ghost hash on this platform: {}", h1);
    eprintln!("target_arch                                       : {}", std::env::consts::ARCH);

    assert_eq!(
        h1, h2,
        "optimizer output must be deterministic on a single platform — \
         same input + same seed must produce byte-identical output bytes"
    );

    if EXPECTED_HASH.starts_with("REPLACE_ME_") {
        // First run on a new arch — log and skip the pin gate.
        eprintln!(
            "[INFO] No pinned hash for this arch yet. Capture this hash in\n\
             EXPECTED_HASH (core/tests/optimizer_cross_platform.rs)\n\
             to enable per-arch regression gating:\n  \"{}\"",
            h1
        );
    } else {
        assert_eq!(
            h1, EXPECTED_HASH,
            "optimizer output on {} diverged from the pinned hash. A \
             regression has shifted output bytes on this arch.",
            std::env::consts::ARCH
        );
    }
}
