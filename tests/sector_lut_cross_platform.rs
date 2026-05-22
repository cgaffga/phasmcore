// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Cross-platform empirical SectorLut byte equivalence (T3.4.D).
//!
//! The Armor DFT-ring sector lookup table classifies each (dy, dx)
//! frequency bin into one of NUM_SECTORS=256 sectors using
//! `det_hypot` + `det_atan2`. Both are IEEE-754-deterministic (no
//! transcendentals from `f64::sin/cos/atan2`, all polynomial
//! approximations in `det_math`). The bin classification + indexing
//! is integer arithmetic, also deterministic.
//!
//! The LUT layout (number of bins per sector + their (idx, conj_idx)
//! pairs in dy-outer / dx-inner order) MUST be byte-identical on
//! every supported target. Otherwise Armor stego JPEG output would
//! diverge across platforms — wire-format break.
//!
//! Hash recorded on aarch64-NEON 2026-05-22. Same hash also produced
//! by:
//!   x86_64-SSE2 via Rosetta:
//!     `cargo test --target x86_64-apple-darwin --test sector_lut_cross_platform`
//!   wasm32-SIMD128 via wasmtime:
//!     `cargo build --release --target wasm32-wasip1 --example sector_lut_hash_check`
//!     `wasmtime target/wasm32-wasip1/release/examples/sector_lut_hash_check.wasm`

use phasm_core::stego::armor::dft_payload::lut_cross_platform_hash_hex;

const EXPECTED_HASH: &str =
    "16cbee4b8d204bb5726e2fad8e3e504b84ac09eab53e9fb38808e3027fcb1131";

#[test]
fn sector_lut_bytes_are_platform_invariant() {
    let actual = lut_cross_platform_hash_hex();
    eprintln!("SectorLut cross-platform hash: {}", actual);
    if EXPECTED_HASH == "REPLACE_ME_WITH_AARCH64_HASH" {
        panic!(
            "EXPECTED_HASH not yet pinned. \
            Set EXPECTED_HASH = \"{}\" in \
            core/tests/sector_lut_cross_platform.rs.",
            actual
        );
    }
    assert_eq!(
        actual, EXPECTED_HASH,
        "SectorLut bytes diverged on this platform — det_hypot / \
        det_atan2 / integer-arithmetic classification expected to be \
        IS-deterministic across aarch64-NEON, x86_64-SSE2, wasm32-\
        SIMD128, and scalar. Wire-format break."
    );
}
