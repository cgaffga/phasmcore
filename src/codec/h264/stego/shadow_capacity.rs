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
/// `cover_size_bits` = total injectable bits across the 3 shadow domains
/// (CoeffSign + CoeffSuffixLsb + MvdSign). Multiple shadows write LSBs
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
/// (Conservative in the other direction, left as-is deliberately: the pool
/// omits the cascade-safe `MvdSuffixLsb` positions that production shadow
/// selection *does* include — 1.5-4.1% of the real pool. See
/// `GopCapacity::injectable_cover_bits`.)
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
    max_shadow_plaintext_bytes(m_max_bytes.saturating_sub(128))
}
