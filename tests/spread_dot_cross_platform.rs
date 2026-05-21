// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Cross-platform empirical SPREAD_LEN=8 dot product byte equivalence
//! (T2.4). 10,000 deterministic LCG-seeded calls are hashed via SHA256;
//! every supported architecture must produce the same hash.

use phasm_core::stego::armor::embedding_simd::spread_dot_test_hash_hex;

/// Hash recorded on aarch64 NEON 2026-05-21.
const EXPECTED_HASH: &str =
    "71215b73cc73f5660573834ee757fc014737549e621b699fa05297d0791bb390";

#[test]
fn spread_dot_product_bytes_are_platform_invariant() {
    let actual = spread_dot_test_hash_hex();
    eprintln!("spread_dot_product 10k-call hash on this platform: {}", actual);
    if EXPECTED_HASH == "REPLACE_ME_WITH_AARCH64_HASH" {
        panic!(
            "EXPECTED_HASH not yet pinned. \
            Set EXPECTED_HASH = \"{}\" in core/tests/spread_dot_cross_platform.rs.",
            actual
        );
    }
    assert_eq!(
        actual, EXPECTED_HASH,
        "spread_dot_product diverged on this platform — SIMD STDM LLR broke Path A invariant"
    );
}
