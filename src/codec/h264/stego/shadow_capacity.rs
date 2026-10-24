// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Shadow-capacity math — the collision-limited per-shadow ceiling.
//!
//! Pure arithmetic (no encoder dependency). Shared by the OH264 capacity
//! surface (`oh264_capacity`, encode-side, `h264-encoder`) and the
//! streaming DECODE-session capacity probe
//! (`streaming_session::CapacityProbeResult::shadow_max_message_bytes`).
//! Lives in a decode-available module so both the OpenH264 build and a
//! standalone `h264-decoder` build resolve it.

/// Collision-limited shadow-capacity formula.
///
/// `cover_size_bits` = total shadow-eligible bits (CoeffSign +
/// CoeffSuffixLsb + MvdSign + cascade-safe MvdSuffixLsb). Multiple shadows write LSBs
/// into one shared priority-ordered pool; the birthday-paradox collision
/// rate over `N` messages bounds the per-message ceiling at
/// √(1024·pool/(N−1)), capped by the raw pool. Worst-case RS parity
/// (128 B) is then subtracted, and the v1/v2 frame envelope is applied
/// by `max_shadow_plaintext_bytes`.
///
/// ## The pool INTENTIONALLY counts primary-wet positions — do not "fix" this
///
/// Verified 2026-10-19. A raw-cover-count capacity figure is usually a bug
/// (the caller must exclude ∞-cost positions), and this one looks like one:
/// `injectable_cover_bits` is a plain `bits.len()` sum, and 46-60% of those
/// positions carry `f32::INFINITY` in the primary cost vector
/// (`content_costs::compute_cost_mask`) on the real-world corpus. It is
/// nevertheless **correct**, because ∞-cost is a statement about the PRIMARY
/// embedder, not about the position:
///
/// - The primary uses STC, so its cost vector gates which positions the
///   Viterbi may flip. ∞ ⇒ never flipped.
/// - Shadows use **direct LSB writes**, not STC, and select positions by hash
///   priority alone. `shadow::for_each_eligible_position` — the single source
///   of truth for shadow eligibility — takes `(cover, safe_csl, safe_msl)` and
///   **no cost vector at all**. A DC coefficient's sign bin is just as
///   writable at the CABAC bypass-emit site as an AC one.
/// - The ∞ overlay runs the OPPOSITE way round: `shadow::overlay_infinity_costs`
///   marks the chosen SHADOW positions ∞ so the primary's STC leaves them
///   alone.
///
/// Empirically: ~46-60% of the top-N slots a real shadow encode consumes are
/// primary-wet, and `oh264_video_shadow_roundtrip` is green — impossible if
/// those slots were unusable, since RS absorbs ~128 B of parity, not 50%
/// symbol noise. Narrowing the pool to finite-cost positions would understate
/// shadow capacity by ~√2 and surface a smaller number than the encoder can
/// actually carry.
///
/// The pool also includes the cascade-safe `MvdSuffixLsb` positions, which
/// production shadow selection draws on too (1.5-4.1% of the real pool).
/// They were left out until 2026-10-23 as a deliberate under-report (D10);
/// the pool is now counted by `for_each_eligible_position` itself. See
/// `GopCapacity::injectable_cover_bits`.
///
/// ## D19 (2026-10-23): capped by what N shadows actually survive
///
/// The √ form above assumes ONE parity budget of 128 bytes for the whole
/// frame. The frame is really RS-coded in 255-byte blocks carrying
/// `parity_len` bytes EACH (`rs_encode_blocks_with_parity`), each block
/// correcting `parity_len / 2` byte errors, and the earliest-written shadow
/// takes collisions from every later one. Correctable errors therefore grow
/// with the frame, and the reliable per-shadow size scales roughly with
/// `pool / (N−1)`, not with its square root. So the √ figure over-promises
/// for small pools and many shadows — measured on a 1.7 M-bit pool: three
/// shadows refused from 85% of the old figure up, four likewise — and
/// under-states large pools.
///
/// The returned figure is `min(√ figure, reliable figure)`, where the
/// reliable figure is the largest payload [`shadow_success_probability`]
/// puts at ≥ [`SHADOW_RELIABILITY_TARGET`]. Only over-promises are removed.
/// The √ figure still bounds large pools on purpose: shadow bits are raw LSB
/// writes, about half on primary-wet positions, so a bigger shadow is a
/// bigger stealth hit and takes primary capacity (D2) and selection memory —
/// raising that envelope is a separate decision, not a collision fix.
/// `N = 1` has no inter-shadow collisions and keeps the √ figure.
pub(crate) fn shadow_max_message_bytes_from_cover_bits(
    cover_size_bits: usize,
    n_shadows: usize,
) -> usize {
    use crate::stego::shadow_layer::max_shadow_plaintext_bytes;
    if n_shadows == 0 {
        return 0;
    }
    let denom = n_shadows.saturating_sub(1).max(1);
    let m_max_bits_squared = 1024usize.saturating_mul(cover_size_bits) / denom;
    // f64::sqrt is a correctly-rounded IEEE-754 op (deterministic across
    // platforms, unlike sin/cos) — and capacity is a display estimate, not
    // a key-derived value, so cross-platform bit-identity isn't required.
    let m_max_bits = (m_max_bits_squared as f64).sqrt() as usize;
    let m_max_bits = m_max_bits.min(cover_size_bits);
    let m_max_bytes = m_max_bits / 8;
    let sqrt_figure = max_shadow_plaintext_bytes(m_max_bytes.saturating_sub(128));
    if n_shadows < 2 {
        return sqrt_figure;
    }
    // Largest payload in [0, sqrt_figure] the collision model accepts.
    // Success falls monotonically with size, so bisect.
    let ok = |payload: usize| {
        shadow_success_probability(cover_size_bits, n_shadows, payload)
            >= SHADOW_RELIABILITY_TARGET
    };
    if ok(sqrt_figure) {
        return sqrt_figure;
    }
    let (mut lo, mut hi) = (0usize, sqrt_figure);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if ok(mid) { lo = mid } else { hi = mid - 1 }
    }
    lo
}

/// Success probability the shown per-shadow capacity is held to (D19).
///
/// The encoder self-verifies and refuses rather than emitting an
/// undecodable shadow, so a miss is a refusal, never silent loss — but it
/// is a refusal after a full encode, so the bar should rarely promise one.
pub(crate) const SHADOW_RELIABILITY_TARGET: f64 = 0.95;

/// Probability that `n_shadows` shadows of `payload` plaintext bytes each,
/// sharing a pool of `pool_bits` eligible positions, ALL decode (D19).
///
/// Model, part by part what the encoder does:
///  * frame = `payload` + the v1/v2 shadow envelope, RS-coded in 255-byte
///    blocks with `p` parity bytes each, correcting `p/2` byte errors;
///  * each shadow's positions are a hash-priority order — a uniformly
///    random subset of the pool, a higher tier taking a longer prefix;
///  * shadows are stamped in order, a later one overwriting an earlier one,
///    and an overwritten bit is wrong with probability 1/2;
///  * the encoder takes the first tier of 4, 8, …, 128 at which every shadow
///    verifies, so success is the best tier's.
///
/// Per block, byte errors are Binomial(block length, q) with
/// `q = 1 − (1 − e)^8` and `e = ½·(1 − (1 − bits/pool)^later)`. Checked
/// against a Monte Carlo of the same mechanics within 1–4 points, and the
/// Monte Carlo against 15 real encodes (`d19_shadow_collision_mc.rs`).
/// Arithmetic only (`+ × ÷ powi`), so it is reproducible across platforms.
pub(crate) fn shadow_success_probability(pool_bits: usize, n_shadows: usize, payload: usize) -> f64 {
    use crate::stego::shadow_layer::{
        SHADOW_FRAME_OVERHEAD_V1, SHADOW_FRAME_OVERHEAD_V2, SHADOW_PARITY_TIERS,
    };
    const RS_N: usize = 255;
    if n_shadows == 0 || pool_bits == 0 {
        return 0.0;
    }
    let overhead = if payload > u16::MAX as usize {
        SHADOW_FRAME_OVERHEAD_V2
    } else {
        SHADOW_FRAME_OVERHEAD_V1
    };
    let frame = payload + overhead;
    let mut best = 0.0f64;
    for &p in SHADOW_PARITY_TIERS.iter() {
        let k = RS_N - p;
        let blocks = frame.div_ceil(k);
        let rs_bytes = frame + blocks * p;
        let bits = 8 * rs_bytes;
        if bits > pool_bits {
            continue; // not enough positions for even one shadow at this tier
        }
        let t = p / 2;
        let last_len = (frame - (blocks - 1) * k) + p;
        let frac = bits as f64 / pool_bits as f64;
        let mut prob = 1.0f64;
        for later in 0..n_shadows {
            // `later` = shadows stamped after this one; the last has none.
            if later == 0 {
                continue;
            }
            let overwritten = 1.0 - (1.0 - frac).powi(later as i32);
            let q = 1.0 - (1.0 - overwritten / 2.0).powi(8);
            let full = binomial_cdf(t, RS_N, q);
            prob *= full.powi((blocks - 1) as i32) * binomial_cdf(t, last_len, q);
        }
        if prob > best {
            best = prob;
        }
    }
    best
}

/// `P(X ≤ t)` for `X ~ Binomial(n, q)`, by the pmf ratio recurrence — no
/// `ln`/`exp`, so platform-deterministic.
fn binomial_cdf(t: usize, n: usize, q: f64) -> f64 {
    if q <= 0.0 || t >= n {
        return 1.0;
    }
    if q >= 1.0 {
        return 0.0;
    }
    let ratio = q / (1.0 - q);
    let mut pmf = (1.0 - q).powi(n as i32);
    let mut cdf = pmf;
    for x in 0..t {
        pmf *= ratio * (n - x) as f64 / (x + 1) as f64;
        cdf += pmf;
    }
    cdf.min(1.0)
}

#[cfg(test)]
mod d19_tests {
    use super::*;

    /// The analytic model against the reference values it was validated with
    /// (the Python reference, itself within 1–4 points of the Monte Carlo).
    #[test]
    fn success_probability_matches_the_validated_reference() {
        let pool = 1_725_747;
        for &(n, payload, want) in &[
            (2usize, 5080usize, 0.955f64),
            (2, 5588, 0.673),
            (3, 2655, 0.929),
            (3, 3009, 0.399),
            (4, 1715, 0.981),
            (4, 2144, 0.153),
        ] {
            let got = shadow_success_probability(pool, n, payload);
            assert!(
                (got - want).abs() < 0.01,
                "n={n} {payload} B: model {got:.3}, reference {want:.3}",
            );
        }
    }

    /// The fix only ever LOWERS the shown figure, never below the reliability
    /// target, and N = 1 is untouched.
    #[test]
    fn capacity_only_removes_over_promises() {
        for &pool in &[500_000usize, 1_725_747, 5_000_000, 20_000_000] {
            let sqrt = |n: usize| {
                let d = n.saturating_sub(1).max(1);
                let mb = (((1024 * pool / d) as f64).sqrt() as usize).min(pool) / 8;
                crate::stego::shadow_layer::max_shadow_plaintext_bytes(mb.saturating_sub(128))
            };
            assert_eq!(shadow_max_message_bytes_from_cover_bits(pool, 1), sqrt(1));
            let mut prev = usize::MAX;
            for n in 2..=8 {
                let cap = shadow_max_message_bytes_from_cover_bits(pool, n);
                assert!(cap <= sqrt(n), "pool {pool} n={n}: {cap} above the old figure");
                assert!(cap <= prev, "pool {pool}: non-monotone in N at {n}");
                if cap > 0 {
                    assert!(
                        shadow_success_probability(pool, n, cap) >= SHADOW_RELIABILITY_TARGET
                            || cap == sqrt(n),
                        "pool {pool} n={n}: shown {cap} below target",
                    );
                }
                prev = cap;
            }
        }
        // Measured over-promise on the 1.7 M-bit clip is gone.
        assert!(shadow_max_message_bytes_from_cover_bits(1_725_747, 3) < 2700);
        assert!(shadow_max_message_bytes_from_cover_bits(1_725_747, 4) < 1800);
    }
}

