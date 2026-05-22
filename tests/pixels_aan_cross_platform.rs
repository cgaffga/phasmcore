// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Cross-platform empirical AAN DCT/IDCT byte equivalence (T3.1.C.3).
//!
//! Phasm's integer LL&M 8×8 DCT/IDCT (Loeffler-Ligtenberg-Moschytz,
//! libjpeg-turbo `jfdctint.c` / `jidctint.c` "slow integer" form) is
//! IS-deterministic by construction: integer arithmetic, no f64, no
//! FMA, no SIMD rounding-mode dependence. This test pins a SHA256 of
//! the output bytes on a deterministic 12-block fixture and asserts
//! the hash matches across every supported architecture.
//!
//! Active code paths covered:
//!
//! | Arch    | Build           | fdct path               | idct path             |
//! |---------|-----------------|-------------------------|-----------------------|
//! | aarch64 | --features aan-dct | NEON (Phase C.1)        | NEON (Phase C.2)       |
//! | x86_64  | --features aan-dct | scalar (AVX2 in Phase D) | scalar (AVX2 in Phase D) |
//! | wasm32  | --features aan-dct | scalar (SIMD128 in Phase E) | scalar (SIMD128 in Phase E) |
//!
//! All three MUST produce identical hashes. If a divergence appears
//! later (when AVX2 / SIMD128 paths land), the new SIMD intrinsic
//! has broken IS-determinism — Phase F gate fails until fixed.
//!
//! Re-run under Rosetta 2 (x86_64 on aarch64) and inside WASM
//! SIMD128 via Node/V8 to confirm equivalence empirically.

use phasm_core::codec::jpeg::pixels_aan::aan_test_hash_hex;

/// Hash recorded on aarch64 NEON, 2026-05-21. Phases C.1 + C.2 active.
const EXPECTED_HASH: &str =
    "8affcdb98021d23b1c1ef536f782610f229fac187360f02722bb3470db8446d5";

#[test]
fn aan_dct_idct_bytes_are_platform_invariant() {
    let actual = aan_test_hash_hex();
    eprintln!("AAN DCT/IDCT 12-block hash on this platform: {}", actual);
    if EXPECTED_HASH == "REPLACE_ME_WITH_AARCH64_HASH" {
        panic!(
            "EXPECTED_HASH not yet pinned. \
            Set EXPECTED_HASH = \"{}\" in core/tests/pixels_aan_cross_platform.rs.",
            actual
        );
    }
    assert_eq!(
        actual, EXPECTED_HASH,
        "AAN DCT/IDCT bytes diverged on this platform — integer LL&M IS-determinism broken"
    );
}
