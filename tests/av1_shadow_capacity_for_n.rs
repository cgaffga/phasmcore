#![cfg(feature = "av1-encoder")]

// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Gates for `av1_shadow_capacity_for_n` — the N-aware shadow byte
//! cap helper the mobile HUD calls on shadow-count change (so the
//! shadow-segment bar can resize without re-probing the clip).
//!
//! Three invariants:
//! 1. **Matches the probe-result method at N=1** — the free fn and
//!    `Av1CapacityProbeResult::shadow_max_message_bytes` must agree
//!    bit-for-bit, because the method delegates to the free fn.
//! 2. **Strictly decreases as N grows** — more shadows → less
//!    per-shadow capacity under the collision formula
//!    `m_max_bits ≤ sqrt(1024·C/max(1, N-1))`.
//! 3. **N=0 returns 0** — degenerate guard; prevents div-by-zero or
//!    nonsensical caps when the UI hasn't picked a shadow count yet.

use phasm_core::av1_shadow_capacity_for_n;
use phasm_core::codec::av1::stego::capacity::Av1CapacityProbeResult;

/// Build a probe result with a given `cover_bits` total. Other fields
/// default to zero / empty — only `cover_bits` matters for the shadow
/// cap math.
fn probe_result_with_cover(cover_bits: usize) -> Av1CapacityProbeResult {
    Av1CapacityProbeResult {
        cover_bits,
        ac_sign_bits: 0,
        golomb_tail_bits: 0,
        n_gops: 0,
        per_gop_cover_bits: Vec::new(),
    }
}

#[test]
fn av1_shadow_capacity_for_n_matches_probe_result_at_n1() {
    // Pool large enough to clear the (128 parity + FRAME_OVERHEAD) floor.
    for cover_bits in [1_000_000usize, 2_500_000, 10_000_000, 100_000_000] {
        let free = av1_shadow_capacity_for_n(cover_bits, 1);
        let method = probe_result_with_cover(cover_bits).shadow_max_message_bytes(1);
        assert_eq!(
            free, method,
            "free fn and probe-method must agree at N=1 for cover_bits={cover_bits}"
        );
    }
}

#[test]
fn av1_shadow_capacity_for_n_matches_probe_result_across_n() {
    // Agreement must hold for every N, not just N=1 — the method
    // delegates to the free fn for ALL callers.
    let cover_bits = 10_000_000usize;
    for n in [1u32, 2, 3, 4, 8, 16] {
        let free = av1_shadow_capacity_for_n(cover_bits, n);
        let method = probe_result_with_cover(cover_bits).shadow_max_message_bytes(n);
        assert_eq!(
            free, method,
            "free fn ↔ probe-method at N={n}, cover_bits={cover_bits}"
        );
    }
}

#[test]
fn av1_shadow_capacity_for_n_shrinks_as_n_grows() {
    // Collision formula is strictly monotonic in N once N > 1.
    let cover_bits = 10_000_000usize;
    let c1 = av1_shadow_capacity_for_n(cover_bits, 1);
    let c2 = av1_shadow_capacity_for_n(cover_bits, 2);
    let c4 = av1_shadow_capacity_for_n(cover_bits, 4);
    let c8 = av1_shadow_capacity_for_n(cover_bits, 8);
    // N=1 and N=2 share the same denominator `max(1, N-1) = 1` in the
    // closed-form so they're EQUAL by design — the strict decrease
    // kicks in from N=2 onward (denom grows). This matches H.264's
    // behaviour and the pre-existing method semantics.
    assert_eq!(
        c1, c2,
        "N=1 and N=2 share denom=max(1, N-1)=1 — must be equal (c1={c1}, c2={c2})"
    );
    assert!(c2 > c4, "N=2 ({c2}) must exceed N=4 ({c4})");
    assert!(c4 > c8, "N=4 ({c4}) must exceed N=8 ({c8})");
}

#[test]
fn av1_shadow_capacity_for_n_n0_returns_zero() {
    // Degenerate guard — the UI may call this before the user has
    // picked a shadow count. Returning 0 is safer than panicking on
    // a `denom = 0` div.
    assert_eq!(
        av1_shadow_capacity_for_n(10_000_000, 0),
        0,
        "N=0 must return 0, not panic"
    );
    assert_eq!(
        av1_shadow_capacity_for_n(0, 0),
        0,
        "N=0 + empty cover must return 0, not panic"
    );
}

#[test]
fn av1_shadow_capacity_for_n_empty_cover_returns_zero() {
    // Clip too small to carry any shadow at all — must saturate to 0.
    for n in [1u32, 2, 4, 8] {
        assert_eq!(
            av1_shadow_capacity_for_n(0, n),
            0,
            "empty cover at N={n} must return 0"
        );
    }
}
