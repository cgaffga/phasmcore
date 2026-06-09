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
