// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Cross-platform empirical STC Viterbi kernel byte equivalence
//! (T3.6.E).
//!
//! T3.6 introduced a restructured 128-state Viterbi step + explicit
//! NEON / SSE2 / WASM SIMD128 intrinsic kernels. All four code paths
//! (scalar, NEON, SSE, SIMD128) MUST produce byte-identical
//! `prev_cost` traces + `packed_bp` back-pointers on the
//! deterministic fixture in `viterbi_test_hash_hex`.
//!
//! The hash below was recorded on aarch64-NEON, then verified
//! byte-for-byte identical on:
//!
//! - x86_64-SSE2 via Rosetta:
//!     `cargo test --target x86_64-apple-darwin --test viterbi_cross_platform`
//! - wasm32-SIMD128 via wasmtime:
//!     `cargo build --release --target wasm32-wasip1 --example viterbi_hash_check`
//!     `wasmtime target/wasm32-wasip1/release/examples/viterbi_hash_check.wasm`
//! - wasm32 scalar (no `+simd128`): same wasmtime command after a
//!   clean rebuild without the SIMD128 rustflag — confirms the
//!   restructured scalar fallback is byte-identical to every SIMD ISA.

use phasm_core::stego::stc::embed::viterbi_test_hash_hex;

const EXPECTED_HASH: &str =
    "48629651a69d9b9121d1c5db7c42ac535e1132361dbacfa2f153b6d1e631fd4d";

#[test]
fn viterbi_bytes_are_platform_invariant() {
    let actual = viterbi_test_hash_hex();
    eprintln!("Viterbi cross-platform hash: {}", actual);
    if EXPECTED_HASH == "REPLACE_ME_WITH_AARCH64_HASH" {
        panic!(
            "EXPECTED_HASH not yet pinned. \
            Set EXPECTED_HASH = \"{}\" in \
            core/tests/viterbi_cross_platform.rs.",
            actual
        );
    }
    assert_eq!(
        actual, EXPECTED_HASH,
        "Viterbi kernel diverged on this platform — scalar / NEON / \
        SSE2 / SIMD128 must produce byte-identical state traces + \
        back-pointer u128s. The restructured kernel preserves \
        f64-IEEE-754 determinism: same per-lane add + < compare + \
        select on every path."
    );
}
