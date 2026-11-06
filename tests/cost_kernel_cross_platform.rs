// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Cross-platform empirical UNIWARD cost-kernel byte equivalence
//! (T3.3.E).
//!
//! Phases A→D of T3.3 introduced a precomputed |delta| table +
//! pair-wise SIMD-friendly sum order for `compute_coefficient_cost`.
//! The same pair-wise sum order is used by ALL four code paths:
//!
//! | Arch    | Build                  | Active cost-kernel path |
//! |---------|------------------------|-------------------------|
//! | aarch64 | default                | NEON 2-lane f64         |
//! | x86_64  | default (+sse4.1)      | SSE2 2-lane f64         |
//! | wasm32  | +simd128 (phasm.app)   | SIMD128 2-lane f64      |
//! | any     | no SIMD target_feature | scalar pair-wise        |
//!
//! All four MUST produce identical bit patterns on the deterministic
//! 8×8-block fixture used by `cost_kernel_test_hash_hex`. The pinned
//! hash below was recorded on aarch64-NEON, then verified byte-for-
//! byte identical on:
//!
//! - x86_64-SSE2 via Rosetta:
//!     `cargo test --target x86_64-apple-darwin --test cost_kernel_cross_platform`
//! - wasm32-SIMD128 via wasmtime:
//!     `cargo build --release --target wasm32-wasip1 --example cost_hash_check`
//!     `wasmtime target/wasm32-wasip1/release/examples/cost_hash_check.wasm`
//! - wasm32 scalar (no `+simd128`): same wasmtime command after a
//!   clean rebuild without the SIMD128 rustflag — confirms the
//!   pair-wise sum order in the scalar fallback is byte-identical to
//!   every SIMD path (not just LLVM autovec hiding the difference).
//!
//! Wire-format protection: this test pins the f64 cost values, which
//! are stored as f32 in the actual CostMap. The pair-wise reordering
//! drifts vs the legacy per-cell sum order by ~f64 ULP (8 orders of
//! magnitude below f32 LSB), so all f32-cast costs are bit-identical
//! between legacy and any of these four paths — stego JPEG bytes
//! unchanged. The hash here guards the STRONGER claim that the f64
//! internals are also identical across SIMD ISAs.

use phasm_core::stego::cost::uniward::{cost_kernel_test_hash_hex, cost_kernel_test_hash_hex_legacy_v1};

/// v2 cost function (phasm-core 0.5, 2026-10-25): whole-window pricing over
/// symmetric-padded wavelets, window aligned with the impact
/// (`docs/design/image/juniward-cost.md`). The fixture now also exercises
/// the padded decomposition: the outer block ring's windows read margin
/// samples.
///
/// History:
/// - `059a4a4707b21fa1…` (2026-05-22, aarch64 NEON, T3.3 phases A–D): the
///   v1 kernel. Still pinned below as `EXPECTED_HASH_LEGACY_V1` — the
///   legacy path the shadow decoder falls back to must stay bit-identical
///   to what 0.4.x shipped.
/// - `52b3740787c356b97d664d896d037379e5c847293e64ec1b6141252c2fc974e9` (2026-10-25): v2. Moved because the cost definition
///   changed (both flaws fixed); not a determinism regression. Verified
///   identical on aarch64 NEON, x86_64 SSE2 (Rosetta) and wasm32 SIMD128.
/// - `14156ec68d394b4b1b92fdcb796faded67ed0e97f957ca22e2b7733e354d44b5` (2026-11-03, Ghost v3.1): the
///   costs read the unrounded decompressed image (an f64 IDCT instead of the
///   rounded integer one, fix list R3.17), as reference J-UNIWARD does.
const EXPECTED_HASH: &str = "14156ec68d394b4b1b92fdcb796faded67ed0e97f957ca22e2b7733e354d44b5";

/// The 0.4.x (v1) kernel, kept verbatim in `uniward_legacy_v1` for
/// decoding the shadows of older images. Must never move.
const EXPECTED_HASH_LEGACY_V1: &str =
    "059a4a4707b21fa135e803a27c8bdad0dc01d9b1f39a030ec1ac4e923e62a8f2";

#[test]
fn cost_kernel_bytes_are_platform_invariant() {
    let actual = cost_kernel_test_hash_hex();
    eprintln!("Cost kernel cross-platform hash: {}", actual);
    assert_eq!(
        actual, EXPECTED_HASH,
        "cost kernel f64 bytes diverged on this platform — \
        cross-platform determinism broken (the pair-wise sum order \
        in uniward::cost_kernel and the padded decomposition must \
        produce identical output on every SIMD ISA)"
    );
}

#[test]
fn legacy_v1_cost_kernel_is_unchanged() {
    let actual = cost_kernel_test_hash_hex_legacy_v1();
    eprintln!("Legacy v1 cost kernel hash: {}", actual);
    assert_eq!(
        actual, EXPECTED_HASH_LEGACY_V1,
        "the legacy v1 cost kernel no longer reproduces phasm-core 0.4.x — \
        shadows of images made by older releases would stop decoding"
    );
}
