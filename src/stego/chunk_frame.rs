// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Per-GOP chunk framing for streaming video stego — wire format v3
//! (codec-agnostic, locked 2026-06-08) + v3.1 (#888, 2026-06-29): adds an
//! explicit `m_total: u32` checksum per chunk header that kills the decode
//! brute-force (~10–20×). H.264 emits/reads v3.1; AV1 migrates in step 6.
//! See `docs/design/video/h264/chunk-frame-v3.1-decode.md`.
//!
//! Multi-GOP streaming stego splits the (already-encrypted+CRC'd)
//! payload bytes across GOP boundaries. The first stego GOP carries
//! a clip-level `total_bytes` header; subsequent stego GOPs carry only
//! a `payload_len` length field. Decoder reads chunks in GOP order
//! and stops when `Σ payload_len == total_bytes`.
//!
//! **Used by both H.264 and AV1 video stego.** Wire bytes are
//! identical across codecs; only the per-codec STC + cover-bit harvest
//! differs. Sharing one module guarantees the chunk_frame layout
//! cannot drift between codecs.
//!
//! Full design: `docs/design/video/chunk-frame-v3.md`.
//!
//! ## Wire format
//!
//! ### First chunk (GOP 0 — message-bearing)
//!
//! ```text
//!   total_bytes : u32  big-endian   (1..=u32::MAX; full AEAD-encrypted payload size)
//!   payload_len : u16  big-endian   (inline,   0 ..= 65534)   → header  6 B
//!             OR  0xFFFF (u16) + u32 big-endian actual length
//!                                   (extended, len ≥ 65535)    → header 10 B
//!   payload bytes : exactly `payload_len` bytes
//! ```
//!
//! ### Subsequent chunks (GOP i, 1 ≤ i < W)
//!
//! ```text
//!   payload_len : u16  big-endian   (inline)                   → header  2 B
//!             OR  0xFFFF (u16) + u32 big-endian actual length
//!                                   (extended, len ≥ 65535)    → header  6 B
//!   payload bytes : exactly `payload_len` bytes
//! ```
//!
//! ### Tail GOPs (i ≥ W)
//!
//! Fully natural — no chunk_frame, no STC, no cover hooks fire. The
//! decoder never STC-extracts these (stops at `accumulated == total_bytes`).
//!
//! ## Why `payload_len` is mandatory (length-strict invariant)
//!
//! The decoder recovers the per-GOP STC `m_total` by brute-force,
//! computing `w = ⌊n_cover / m_total⌋` for each candidate. That floor
//! is **many-to-one**: a contiguous range of `m_total` values share
//! one `w`, and because STC extract is convolutional, the chunk header
//! (the leading syndrome bits) parses *identically* across that whole
//! w-class. Without a length field the decoder would land on the class
//! **minimum** `m_total`, returning a truncated payload that passes the
//! header parse but fails AEAD downstream. The explicit `payload_len`
//! makes the v3 parsers length-strict, so short candidates are rejected
//! and the brute-force lands on the encoder's exact `m_total`.
//!
//! ## No version byte, no magic, no backwards compatibility
//!
//! v3 ships as a clean break. No pre-v3 stego exists in the wild
//! (zero installed v2 base in the production app at cutover).
//! Wrong-passphrase fast-fail is via the `total_bytes` sanity check
//! on the first-chunk parse + AEAD-on-payload at the outer frame.
//!
//! ## Brute-force fast-reject
//!
//! v2 used an explicit `chunk_idx` value-match (1/65536 reject per
//! candidate) as the fast-reject filter. v3 has no `chunk_idx` — the
//! equivalent power comes from the **w-canonicality filter**:
//! the encoder picks `w_e = max(1, n_cover / ((header + payload_len) × 8))`
//! deterministically; the decoder inverts this and rejects any
//! iteration where `w != w_canonical(cand_payload_len)`. Same
//! ~1/65536 reject magnitude as v2, zero wire bytes. Implementation
//! is per-codec (different STC search dimensions); see
//! `codec/av1/stego/orchestrator.rs::extract_*_v3_match` and the H.264
//! analogue.

use crate::stego::error::StegoError;

/// `payload_len` u16 value reserved to signal the extended form (a `u32`
/// length follows). Inline lengths are therefore `0 ..= 65534`. Same
/// sentinel for first and subsequent chunks (and v2 if anything still
/// remembered it).
pub const LEN_SENTINEL: u16 = 0xFFFF;

/// Default per-GOP soft rate ceiling for the concentrate+tail
/// allocator (`allocate_chunks_concentrate_tail`). **Provisional, NOT yet
/// calibrated** — the real value will be set by sweeping per-GOP
/// embedding rate vs. an ML steganalysis detector and picking chance+margin.
///
/// 0.5 (vs the absolute-max 1.0) is the conservative interim: it fills each
/// stego GOP to ~half its probed cap, which (a) leaves headroom so a verify
/// shrink-carry stays inside the stego window instead of overflowing into the
/// plain tail (→ graceful `MessageTooLarge` rather than a wider window), and
/// (b) is stealth-leaning per the 2026-06-01 product decision. A future `fast`
/// opt-in raises it (tighter window, more perf, higher rate).
pub const CAP2_DEFAULT_R_TARGET: f64 = 0.5;

/// Split `message_bytes` into `total_chunks` evenly-sized pieces. The
/// final chunk may be smaller if the message length doesn't divide
/// evenly. Returns the chunk payload slices ready for the v3.1 builders
/// ([`build_first_chunk_frame_v3_1`] on chunk 0, [`build_chunk_frame_v3_1`]
/// on subsequent).
///
/// Wire-format-independent: just message splitting. v3.1 uses this same
/// helper unchanged from the v2 era. Future planners (balanced safe
/// allocation, etc.) may compute their own per-GOP byte counts and
/// build chunks directly from a slice without this helper.
///
/// # Errors
/// * `InvalidVideo` if `total_chunks == 0`.
pub fn split_message_into_chunks(
    message_bytes: &[u8],
    total_chunks: u16,
) -> Result<Vec<Vec<u8>>, StegoError> {
    if total_chunks == 0 {
        return Err(StegoError::InvalidVideo(
            "split_message_into_chunks: total_chunks must be > 0".into(),
        ));
    }
    let n = total_chunks as usize;
    let len = message_bytes.len();
    // Ceiling-divide so the last chunk absorbs the remainder.
    let chunk_size = len.div_ceil(n);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * chunk_size;
        let end = ((i + 1) * chunk_size).min(len);
        if start >= len {
            // Pad with empty chunks if message is shorter than n_chunks
            // (rare; only happens if caller picked too many chunks for
            // a small message).
            out.push(Vec::new());
        } else {
            out.push(message_bytes[start..end].to_vec());
        }
    }
    Ok(out)
}

/// Rate-bounded proportional chunk allocation (the carry-free
/// successor to `split_message_into_chunks`' even-split).
///
/// Given each GOP's accurate byte capacity (`gop_caps`, from the STC-trial /
/// the v3 estimator) and the message length, return the per-GOP byte counts.
/// Two properties the even-split lacks:
///
/// - **Achieves Σ, doesn't bind on `min`.** `alloc[i] = ⌊M·basis[i]/Σbasis⌋`;
///   since the basis is per-GOP (capacity or rate-target), no GOP is asked to
///   hold more than it can — a weak GOP simply takes less. The binding limit is
///   `M ≤ Σ cap`, not `n_gops × min cap` (so `None` ⇔ genuinely over capacity).
///   This is why proportional needs **no carry**: with `M ≤ Σcap`, `M/Σ ≤ 1`,
///   so `alloc[i] ≤ cap[i]` automatically.
/// - **Rate-bounded for stealth (`r_target`).** Each GOP is first capped at
///   `⌊r_target·cap⌋` so a sub-capacity message spreads thin (low flip density).
///   **Soft ceiling:** a message larger than `r_target·Σcap` falls back to the
///   full-capacity basis — graceful stealth degradation exactly where stealth is
///   already compromised (near the absolute max).
///
/// The leftover from integer rounding is distributed by the **largest-remainder
/// method** (respecting each GOP's true cap), so `Σ alloc == message_len` exactly
/// and the split is deterministic. Returns `None` iff `message_len > Σ cap`.
///
/// NOTE: this fills high-capacity GOPs nearer their STC budget than
/// the even-split did. Wiring it into the encoder must be gated on a round-trip
/// re-verification at these higher per-GOP fills (the cascade-decode ceiling)
/// — the allocator math here is encoder-independent and side-effect free.
pub fn allocate_chunks_proportional(
    gop_caps: &[usize],
    message_len: usize,
    r_target: f64,
) -> Option<Vec<usize>> {
    let n = gop_caps.len();
    if n == 0 {
        return (message_len == 0).then(Vec::new);
    }
    if message_len == 0 {
        return Some(vec![0; n]);
    }
    let sigma_cap: usize = gop_caps.iter().sum();
    if message_len > sigma_cap {
        return None; // genuinely over the Σ ceiling
    }

    // Rate-bounded per-GOP targets (stealth). Soft ceiling: if the message
    // doesn't fit under the rate target, widen the basis to full capacity.
    let r = r_target.clamp(0.0, 1.0);
    let targets: Vec<usize> = gop_caps.iter().map(|&c| (r * c as f64) as usize).collect();
    let sigma_target: usize = targets.iter().sum();
    let use_targets = sigma_target > 0 && message_len <= sigma_target;
    let basis: &[usize] = if use_targets { &targets } else { gop_caps };
    let sigma_basis: usize = basis.iter().sum();
    if sigma_basis == 0 {
        return None;
    }

    // Floor proportional + fractional remainders (u128 to avoid overflow on
    // long clips).
    let mut alloc = vec![0usize; n];
    let mut frac = vec![0u128; n];
    let mut assigned = 0usize;
    for i in 0..n {
        let num = (message_len as u128) * (basis[i] as u128);
        alloc[i] = (num / sigma_basis as u128) as usize;
        frac[i] = num % sigma_basis as u128;
        assigned += alloc[i];
    }

    // Largest-remainder: hand the leftover bytes to the highest fractional
    // remainders first, but never past a GOP's TRUE capacity.
    let mut leftover = message_len - assigned;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| frac[b].cmp(&frac[a]).then(a.cmp(&b)));
    for &i in order.iter().cycle().take(n * 2) {
        if leftover == 0 {
            break;
        }
        if alloc[i] < gop_caps[i] {
            alloc[i] += 1;
            leftover -= 1;
        }
    }
    // Final sweep in case the remainder basis was tight (targets near cap): fill
    // any remaining bytes wherever capacity is left. Guaranteed to drain since
    // message_len ≤ Σcap.
    if leftover > 0 {
        for i in 0..n {
            while leftover > 0 && alloc[i] < gop_caps[i] {
                alloc[i] += 1;
                leftover -= 1;
            }
        }
    }
    debug_assert_eq!(leftover, 0, "leftover must drain when message_len ≤ Σcap");
    Some(alloc)
}

/// Rate-bounded **concentrate + plain-tail** allocation. Unlike
/// [`allocate_chunks_proportional`] (which spreads the message uniformly over
/// *all* GOPs, so `r_target` cancels in the proportions and is inert), this
/// fills GOPs **sequentially** to `min(remaining, ⌊r_target·cap⌋)` and stops
/// once the message is exhausted — leaving a **plain tail** of zero-allocation
/// GOPs. `r_target` is therefore *functional* here: it is the per-GOP rate
/// ceiling, so a lower `r_target` forces the message across MORE leading GOPs
/// at LOWER density (a wider stego window, a shorter plain tail).
///
/// Returns `(plan, window)` where `plan[i]` is GOP `i`'s byte count
/// (`plan[window..]` are all 0) and `window = W` is the count of leading stego
/// GOPs. `Σ plan == message_len` exactly.
///
/// **Soft ceiling:** if `message_len > Σ⌊r·cap⌋` (a large message near the
/// absolute max), the per-GOP cap widens to the FULL `cap[i]` — graceful
/// stealth degradation exactly where stealth is already compromised. Returns
/// `None` iff `message_len > Σ cap` (genuinely over capacity).
///
/// Pure integer arithmetic except the per-element `(r·cap).floor()` cast (same
/// deterministic pattern as `allocate_chunks_proportional`); cross-platform safe.
pub fn allocate_chunks_concentrate_tail(
    gop_caps: &[usize],
    message_len: usize,
    r_target: f64,
) -> Option<(Vec<usize>, usize)> {
    let n = gop_caps.len();
    if n == 0 {
        return (message_len == 0).then(|| (Vec::new(), 0));
    }
    if message_len == 0 {
        return Some((vec![0; n], 0));
    }
    let sigma_cap: usize = gop_caps.iter().sum();
    if message_len > sigma_cap {
        return None; // genuinely over the Σ ceiling
    }

    // Per-GOP rate ceiling ⌊r·cap⌋. Soft ceiling: if the message doesn't fit
    // under Σ⌊r·cap⌋, widen every GOP's cap to its FULL capacity.
    let r = r_target.clamp(0.0, 1.0);
    let ceiled: Vec<usize> = gop_caps
        .iter()
        .map(|&c| ((r * c as f64) as usize).min(c))
        .collect();
    let sigma_ceiled: usize = ceiled.iter().sum();
    let use_full = message_len > sigma_ceiled;

    // Sequential fill: each GOP takes up to its effective cap; the rest are
    // plain. `window` is the index after the last GOP that took any bytes.
    let mut plan = vec![0usize; n];
    let mut remaining = message_len;
    let mut window = 0usize;
    for i in 0..n {
        if remaining == 0 {
            break;
        }
        let cap_i = if use_full { gop_caps[i] } else { ceiled[i] };
        let take = remaining.min(cap_i);
        plan[i] = take;
        remaining -= take;
        if take > 0 {
            window = i + 1;
        }
    }
    // Guaranteed to drain: under the rate branch Σ⌊r·cap⌋ ≥ message_len; under
    // the soft-ceiling branch Σcap ≥ message_len.
    debug_assert_eq!(remaining, 0, "concentrate fill must drain when message_len ≤ basis sum");
    Some((plan, window))
}

// ═════════════════════════════════════════════════════════════════════════
// Wire format v3.1 (codec-agnostic). Locked 2026-06-08 as v3; #888 added the
// explicit u32 m_total checksum and migrated both H.264 and AV1 — v3 is gone.
// No backward compatibility: neither codec shipped a v3 installed base.
// ═════════════════════════════════════════════════════════════════════════
//
// Full spec + m_total reject mechanism: see the `chunk_frame v3.1` note below
// and docs/design/video/h264/chunk-frame-v3.1-decode.md (#888).
//
// ## First chunk (GOP 0) wire layout
//   inline (≤ 65 534 byte payload):
//     u32 BE total_bytes (4) + u32 BE m_total (4) + u16 BE payload_len (2)
//                                                       → 10-byte header
//   extended (≥ 65 535 byte payload):
//     u32 BE total_bytes (4) + u32 BE m_total (4) + 0xFFFF u16 (2)
//       + u32 BE payload_len (4)                        → 14-byte header
//
// ## Subsequent chunks (GOP i > 0) wire layout
//   inline:   u32 BE m_total (4) + u16 BE payload_len (2)   → 6-byte header
//   extended: u32 BE m_total (4) + 0xFFFF u16 (2)
//               + u32 BE payload_len (4)                    → 10-byte header
//
// ## Decoder stop condition
//   Stop when `Σ payload_len == total_bytes` (W is implicit). Tail
//   GOPs `i ≥ W` are fully natural — no chunk_frame, no STC, no cover
//   hooks fire.

// ── chunk_frame v3.1 ──────────────────────────────────────────────────────
//
// v3.1 adds an explicit `m_total: u32` (framed message **bit** count) to every
// chunk header — immediately after `total_bytes` (first chunk) / at the head
// (subsequent). The decoder reads it in the cheap prefix extract and rejects
// any STC candidate whose loop `m_total` != the stored value: an exact 1/2³²
// reject that collapses the v3 brute-force survivors (canonicality
// floor-collisions + LEN_SENTINEL bypass) to a single full extract per slab.
// See docs/design/video/h264/chunk-frame-v3.1-decode.md (#888).
//
// `m_total` is a *checksum*, not free data: it always equals
// `(header_len + payload_len) * 8`, so the builder derives it and the parser
// re-validates it. Stored as a fixed `u32` with NO sentinel/extended form — its
// width *is* its reject strength (u16's 1/2¹⁶ is too weak on large/4K slabs).
// `payload_len` keeps its v3 `u16` + `0xFFFF`→`u32` escape.

/// v3.1 first-chunk header, inline: `u32 total_bytes + u32 m_total + u16 payload_len`.
pub const CHUNK_FRAME_FIRST_HEADER_LEN_V3_1: usize = 10;
/// v3.1 first-chunk header, extended: base 10 + sentinel 2 + u32 = 14 bytes.
pub const CHUNK_FRAME_FIRST_HEADER_LEN_V3_1_MAX: usize = 14;
/// v3.1 subsequent-chunk header, inline: `u32 m_total + u16 payload_len`.
pub const CHUNK_FRAME_NEXT_HEADER_LEN_V3_1: usize = 6;
/// v3.1 subsequent-chunk header, extended: base 6 + sentinel 2 + u32 = 10 bytes.
pub const CHUNK_FRAME_NEXT_HEADER_LEN_V3_1_MAX: usize = 10;

/// `m_total` for a framed chunk = framed message **bit** count =
/// `(header_len + payload_len) * 8`. `None` if it would overflow `u32`
/// (only for absurd >536 MB chunks — far beyond any cover capacity).
#[inline]
fn v3_1_m_total(header_len: usize, payload_len: usize) -> Option<u32> {
    let bits = (header_len as u64)
        .checked_add(payload_len as u64)?
        .checked_mul(8)?;
    u32::try_from(bits).ok()
}

/// Build v3.1 first-chunk wire bytes. `m_total` is derived + written here, so
/// callers pass only `total_bytes` + `payload`, exactly as v3.
///
/// # Errors
/// * `InvalidVideo` if `total_bytes == 0`, `payload.len() > u32::MAX`, or the
///   derived `m_total` would overflow `u32`.
pub fn build_first_chunk_frame_v3_1(
    total_bytes: u32,
    payload: &[u8],
) -> Result<Vec<u8>, StegoError> {
    if total_bytes == 0 {
        return Err(StegoError::InvalidVideo(
            "chunk_frame_v3_1: total_bytes must be > 0".into(),
        ));
    }
    if payload.len() > u32::MAX as usize {
        return Err(StegoError::InvalidVideo(format!(
            "chunk_frame_v3_1: payload {} exceeds u32::MAX",
            payload.len()
        )));
    }
    let extended = payload.len() >= LEN_SENTINEL as usize;
    let header_len = if extended {
        CHUNK_FRAME_FIRST_HEADER_LEN_V3_1_MAX
    } else {
        CHUNK_FRAME_FIRST_HEADER_LEN_V3_1
    };
    let m_total = v3_1_m_total(header_len, payload.len()).ok_or_else(|| {
        StegoError::InvalidVideo("chunk_frame_v3_1: m_total exceeds u32::MAX".into())
    })?;
    let mut out = Vec::with_capacity(header_len + payload.len());
    out.extend_from_slice(&total_bytes.to_be_bytes());
    out.extend_from_slice(&m_total.to_be_bytes());
    if extended {
        out.extend_from_slice(&LEN_SENTINEL.to_be_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    } else {
        out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    }
    out.extend_from_slice(payload);
    Ok(out)
}

/// Build v3.1 subsequent-chunk wire bytes (no clip-level `total_bytes`).
///
/// # Errors
/// * `InvalidVideo` if `payload.len() > u32::MAX` or `m_total` overflows `u32`.
pub fn build_chunk_frame_v3_1(payload: &[u8]) -> Result<Vec<u8>, StegoError> {
    if payload.len() > u32::MAX as usize {
        return Err(StegoError::InvalidVideo(format!(
            "chunk_frame_v3_1: payload {} exceeds u32::MAX",
            payload.len()
        )));
    }
    let extended = payload.len() >= LEN_SENTINEL as usize;
    let header_len = if extended {
        CHUNK_FRAME_NEXT_HEADER_LEN_V3_1_MAX
    } else {
        CHUNK_FRAME_NEXT_HEADER_LEN_V3_1
    };
    let m_total = v3_1_m_total(header_len, payload.len()).ok_or_else(|| {
        StegoError::InvalidVideo("chunk_frame_v3_1: m_total exceeds u32::MAX".into())
    })?;
    let mut out = Vec::with_capacity(header_len + payload.len());
    out.extend_from_slice(&m_total.to_be_bytes());
    if extended {
        out.extend_from_slice(&LEN_SENTINEL.to_be_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    } else {
        out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    }
    out.extend_from_slice(payload);
    Ok(out)
}

/// Parse a v3.1 first-chunk header → `(total_bytes, m_total, payload_slice)`.
/// Length-strict, and re-validates the checksum invariant
/// `m_total == (header_len + payload_len) * 8` — a mismatch returns `None`.
pub fn parse_first_chunk_frame_v3_1(bytes: &[u8]) -> Option<(u32, u32, &[u8])> {
    if bytes.len() < CHUNK_FRAME_FIRST_HEADER_LEN_V3_1 {
        return None;
    }
    let total_bytes = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if total_bytes == 0 {
        return None;
    }
    let m_total = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let len_field = u16::from_be_bytes([bytes[8], bytes[9]]);
    let (payload_len, header_len) = if len_field == LEN_SENTINEL {
        if bytes.len() < CHUNK_FRAME_FIRST_HEADER_LEN_V3_1_MAX {
            return None;
        }
        let v = u32::from_be_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
        if v < LEN_SENTINEL as usize {
            return None;
        }
        (v, CHUNK_FRAME_FIRST_HEADER_LEN_V3_1_MAX)
    } else {
        (len_field as usize, CHUNK_FRAME_FIRST_HEADER_LEN_V3_1)
    };
    if v3_1_m_total(header_len, payload_len)? != m_total {
        return None;
    }
    let end = header_len.checked_add(payload_len)?;
    if bytes.len() < end {
        return None;
    }
    Some((total_bytes, m_total, &bytes[header_len..end]))
}

/// Parse a v3.1 subsequent-chunk header → `(m_total, payload_slice)`. Same
/// length-strict + checksum-invariant validation as the first-chunk parser.
pub fn parse_chunk_frame_v3_1(bytes: &[u8]) -> Option<(u32, &[u8])> {
    if bytes.len() < CHUNK_FRAME_NEXT_HEADER_LEN_V3_1 {
        return None;
    }
    let m_total = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let len_field = u16::from_be_bytes([bytes[4], bytes[5]]);
    let (payload_len, header_len) = if len_field == LEN_SENTINEL {
        if bytes.len() < CHUNK_FRAME_NEXT_HEADER_LEN_V3_1_MAX {
            return None;
        }
        let v = u32::from_be_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        if v < LEN_SENTINEL as usize {
            return None;
        }
        (v, CHUNK_FRAME_NEXT_HEADER_LEN_V3_1_MAX)
    } else {
        (len_field as usize, CHUNK_FRAME_NEXT_HEADER_LEN_V3_1)
    };
    if v3_1_m_total(header_len, payload_len)? != m_total {
        return None;
    }
    let end = header_len.checked_add(payload_len)?;
    if bytes.len() < end {
        return None;
    }
    Some((m_total, &bytes[header_len..end]))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- chunk_frame v3.1 (#888) ----

    #[test]
    fn v3_1_first_roundtrip_inline() {
        let payload = vec![0xABu8; 200];
        let total = 4096u32;
        let framed = build_first_chunk_frame_v3_1(total, &payload).unwrap();
        // layout: total(4) | m_total(4) | payload_len(2) | payload
        assert_eq!(framed.len(), CHUNK_FRAME_FIRST_HEADER_LEN_V3_1 + payload.len());
        assert_eq!(&framed[0..4], &total.to_be_bytes());
        let m_field = u32::from_be_bytes([framed[4], framed[5], framed[6], framed[7]]);
        assert_eq!(
            m_field as usize,
            (CHUNK_FRAME_FIRST_HEADER_LEN_V3_1 + payload.len()) * 8
        );
        let (rt_total, rt_m, rt_payload) = parse_first_chunk_frame_v3_1(&framed).unwrap();
        assert_eq!(rt_total, total);
        assert_eq!(rt_m, m_field);
        assert_eq!(rt_payload, &payload[..]);
    }

    #[test]
    fn v3_1_first_roundtrip_extended() {
        let payload = vec![0xCDu8; LEN_SENTINEL as usize + 50]; // forces extended form
        let total = 1_000_000u32;
        let framed = build_first_chunk_frame_v3_1(total, &payload).unwrap();
        assert_eq!(
            framed.len(),
            CHUNK_FRAME_FIRST_HEADER_LEN_V3_1_MAX + payload.len()
        );
        let (rt_total, rt_m, rt_payload) = parse_first_chunk_frame_v3_1(&framed).unwrap();
        assert_eq!(rt_total, total);
        assert_eq!(
            rt_m as usize,
            (CHUNK_FRAME_FIRST_HEADER_LEN_V3_1_MAX + payload.len()) * 8
        );
        assert_eq!(rt_payload, &payload[..]);
    }

    #[test]
    fn v3_1_subsequent_roundtrip_inline() {
        let payload = vec![0x12u8; 137];
        let framed = build_chunk_frame_v3_1(&payload).unwrap();
        assert_eq!(framed.len(), CHUNK_FRAME_NEXT_HEADER_LEN_V3_1 + payload.len());
        let (rt_m, rt_payload) = parse_chunk_frame_v3_1(&framed).unwrap();
        assert_eq!(
            rt_m as usize,
            (CHUNK_FRAME_NEXT_HEADER_LEN_V3_1 + payload.len()) * 8
        );
        assert_eq!(rt_payload, &payload[..]);
    }

    #[test]
    fn v3_1_subsequent_roundtrip_extended() {
        let payload = vec![0x34u8; LEN_SENTINEL as usize];
        let framed = build_chunk_frame_v3_1(&payload).unwrap();
        assert_eq!(
            framed.len(),
            CHUNK_FRAME_NEXT_HEADER_LEN_V3_1_MAX + payload.len()
        );
        let (_rt_m, rt_payload) = parse_chunk_frame_v3_1(&framed).unwrap();
        assert_eq!(rt_payload, &payload[..]);
    }

    #[test]
    fn v3_1_parse_rejects_corrupt_m_total() {
        let payload = vec![0x55u8; 64];
        let mut framed = build_first_chunk_frame_v3_1(4096, &payload).unwrap();
        framed[5] ^= 0x01; // flip a bit in the m_total field → checksum invariant fails
        assert!(parse_first_chunk_frame_v3_1(&framed).is_none());
    }

    #[test]
    fn v3_1_parse_rejects_short_and_zero_total() {
        assert!(parse_first_chunk_frame_v3_1(&[0u8; 4]).is_none()); // too short
        let payload = vec![0u8; 16];
        let mut framed = build_first_chunk_frame_v3_1(7, &payload).unwrap();
        framed[0..4].copy_from_slice(&0u32.to_be_bytes()); // total_bytes = 0
        assert!(parse_first_chunk_frame_v3_1(&framed).is_none());
    }

    #[test]
    fn v3_1_length_strict_trailing_garbage() {
        let payload = vec![0x99u8; 32];
        let mut framed = build_first_chunk_frame_v3_1(4096, &payload).unwrap();
        framed.extend_from_slice(&[0xFFu8; 20]); // trailing garbage
        let (_t, _m, rt_payload) = parse_first_chunk_frame_v3_1(&framed).unwrap();
        assert_eq!(rt_payload.len(), payload.len()); // length-strict: only declared len
        assert_eq!(rt_payload, &payload[..]);
    }

    // ---- allocate_chunks_proportional ----

    #[test]
    fn alloc_sums_to_message_and_respects_caps() {
        let caps = [100usize, 200, 50, 300, 10];
        for &m in &[0usize, 1, 37, 200, 659, 660] {
            let a = allocate_chunks_proportional(&caps, m, 1.0).expect("fits");
            assert_eq!(a.iter().sum::<usize>(), m, "Σ alloc == message (m={m})");
            for (i, (&got, &cap)) in a.iter().zip(caps.iter()).enumerate() {
                assert!(got <= cap, "alloc[{i}]={got} > cap {cap} (m={m})");
            }
        }
    }

    #[test]
    fn alloc_none_over_sigma_capacity() {
        let caps = [100usize, 200, 50, 300, 10]; // Σ = 660
        assert!(allocate_chunks_proportional(&caps, 661, 1.0).is_none());
        assert!(allocate_chunks_proportional(&caps, 660, 1.0).is_some());
    }

    #[test]
    fn alloc_does_not_bind_on_min_gop() {
        // Even-split would bind on the weak GOP (10) → n×min = 5×10 = 50.
        // Proportional achieves the full Σ = 660. A 600 B message must fit.
        let caps = [100usize, 200, 50, 300, 10];
        let a = allocate_chunks_proportional(&caps, 600, 1.0).expect("Σ fits");
        assert_eq!(a.iter().sum::<usize>(), 600);
        // The weak GOP takes proportionally less, not an equal 120.
        assert!(a[4] <= caps[4], "weak GOP within its cap");
        assert!(a[3] > a[4], "strong GOP carries more than the weak one");
    }

    #[test]
    fn alloc_is_proportional_to_capacity() {
        // Equal caps → ~equal split. 100 over 4×100 → 25 each.
        let a = allocate_chunks_proportional(&[100, 100, 100, 100], 100, 1.0).unwrap();
        assert_eq!(a, vec![25, 25, 25, 25]);
        // 2:1 cap ratio → ~2:1 allocation.
        let a = allocate_chunks_proportional(&[200, 100], 150, 1.0).unwrap();
        assert_eq!(a.iter().sum::<usize>(), 150);
        assert!(a[0] > a[1], "larger-cap GOP gets more ({a:?})");
    }

    #[test]
    fn alloc_rate_target_spreads_thin_then_soft_ceiling() {
        let caps = [1000usize; 4]; // Σcap = 4000, Σ(0.25·cap) = 1000
        // Small message under the rate target: stays within ⌊0.25·cap⌋ = 250.
        let a = allocate_chunks_proportional(&caps, 400, 0.25).unwrap();
        assert_eq!(a.iter().sum::<usize>(), 400);
        assert!(a.iter().all(|&x| x <= 250), "within rate target: {a:?}");
        // Message above the rate target → soft-ceiling to full-cap basis.
        let a = allocate_chunks_proportional(&caps, 2000, 0.25).unwrap();
        assert_eq!(a.iter().sum::<usize>(), 2000);
        assert!(a.iter().any(|&x| x > 250), "soft ceiling widened the basis: {a:?}");
    }

    #[test]
    fn alloc_edge_cases() {
        assert_eq!(allocate_chunks_proportional(&[], 0, 1.0), Some(vec![]));
        assert_eq!(allocate_chunks_proportional(&[], 5, 1.0), None);
        assert_eq!(allocate_chunks_proportional(&[100, 100], 0, 1.0), Some(vec![0, 0]));
        // A zero-cap GOP never receives bytes.
        let a = allocate_chunks_proportional(&[0, 100], 50, 1.0).unwrap();
        assert_eq!(a, vec![0, 50]);
    }

    // ── concentrate+tail allocator ───────────────────────────────────────

    #[test]
    fn concentrate_fills_then_tails() {
        // r=1.0: fill each GOP to its full cap, sequentially, then plain tail.
        let caps = [100usize, 200, 50, 300, 10];
        let (plan, w) = allocate_chunks_concentrate_tail(&caps, 120, 1.0).unwrap();
        assert_eq!(plan, vec![100, 20, 0, 0, 0]);
        assert_eq!(w, 2);
        assert_eq!(plan.iter().sum::<usize>(), 120);
    }

    #[test]
    fn concentrate_window_is_count_of_stego_gops() {
        let caps = [100usize, 200, 50, 300, 10];
        for m in [1usize, 99, 100, 101, 300, 660] {
            let (plan, w) = allocate_chunks_concentrate_tail(&caps, m, 1.0).unwrap();
            // window == index after the last nonzero entry, and the tail is all 0.
            let expected_w = plan.iter().rposition(|&x| x > 0).map_or(0, |p| p + 1);
            assert_eq!(w, expected_w, "m={m} plan={plan:?}");
            assert!(plan[w..].iter().all(|&x| x == 0), "tail not plain: {plan:?}");
            assert_eq!(plan.iter().sum::<usize>(), m);
        }
    }

    #[test]
    fn concentrate_r_target_widens_window() {
        // THE functional-r_target test: lower r ⇒ each GOP capped lower ⇒
        // message spreads across MORE leading GOPs (wider window).
        let caps = [1000usize; 4];
        let (plan_hi, w_hi) = allocate_chunks_concentrate_tail(&caps, 400, 1.0).unwrap();
        assert_eq!(plan_hi, vec![400, 0, 0, 0]);
        assert_eq!(w_hi, 1);
        let (plan_lo, w_lo) = allocate_chunks_concentrate_tail(&caps, 400, 0.25).unwrap();
        // ⌊0.25·1000⌋ = 250 per GOP → 250 + 150.
        assert_eq!(plan_lo, vec![250, 150, 0, 0]);
        assert_eq!(w_lo, 2);
        assert!(w_lo > w_hi, "lower r_target must widen the stego window");
        assert!(plan_lo.iter().all(|&x| x <= 250), "rate-capped: {plan_lo:?}");
    }

    #[test]
    fn concentrate_soft_ceiling() {
        // Message above Σ⌊r·cap⌋ widens to full caps (graceful degradation).
        let caps = [1000usize; 4]; // Σ⌊0.25·cap⌋ = 1000
        let (plan, w) = allocate_chunks_concentrate_tail(&caps, 2000, 0.25).unwrap();
        assert_eq!(plan, vec![1000, 1000, 0, 0]);
        assert_eq!(w, 2);
        assert_eq!(plan.iter().sum::<usize>(), 2000);
    }

    #[test]
    fn concentrate_none_over_sigma() {
        let caps = [100usize, 200, 50];
        assert_eq!(allocate_chunks_concentrate_tail(&caps, 351, 1.0), None);
        // exactly Σcap fits (fills everything, window == n).
        let (plan, w) = allocate_chunks_concentrate_tail(&caps, 350, 1.0).unwrap();
        assert_eq!(plan, vec![100, 200, 50]);
        assert_eq!(w, 3);
    }

    #[test]
    fn concentrate_edge_cases() {
        assert_eq!(allocate_chunks_concentrate_tail(&[], 0, 1.0), Some((vec![], 0)));
        assert_eq!(allocate_chunks_concentrate_tail(&[], 5, 1.0), None);
        assert_eq!(
            allocate_chunks_concentrate_tail(&[100, 100], 0, 1.0),
            Some((vec![0, 0], 0))
        );
        // A leading zero-cap GOP is skipped; window starts at the first GOP
        // that actually takes bytes.
        let (plan, w) = allocate_chunks_concentrate_tail(&[0, 100], 50, 1.0).unwrap();
        assert_eq!(plan, vec![0, 50]);
        assert_eq!(w, 2);
    }

    #[test]
    fn split_evenly_when_divisible() {
        let msg = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let chunks = split_message_into_chunks(&msg, 4).unwrap();
        assert_eq!(chunks.len(), 4);
        for c in &chunks { assert_eq!(c.len(), 2); }
        assert_eq!(chunks[0], &[1, 2]);
        assert_eq!(chunks[3], &[7, 8]);
    }

    #[test]
    fn split_uneven_distributes_remainder_to_last() {
        let msg = vec![1u8, 2, 3, 4, 5];
        let chunks = split_message_into_chunks(&msg, 2).unwrap();
        assert_eq!(chunks.len(), 2);
        // div_ceil(5, 2) = 3, so chunks are [1,2,3] and [4,5].
        assert_eq!(chunks[0], &[1, 2, 3]);
        assert_eq!(chunks[1], &[4, 5]);
    }

    #[test]
    fn split_with_more_chunks_than_bytes_pads_empty() {
        let msg = vec![1u8, 2];
        let chunks = split_message_into_chunks(&msg, 5).unwrap();
        assert_eq!(chunks.len(), 5);
        // div_ceil(2, 5) = 1, so first 2 chunks have 1 byte each, rest empty.
        assert_eq!(chunks[0], &[1]);
        assert_eq!(chunks[1], &[2]);
        assert!(chunks[2].is_empty());
        assert!(chunks[3].is_empty());
        assert!(chunks[4].is_empty());
    }

    #[test]
    fn split_rejects_zero_chunks() {
        assert!(split_message_into_chunks(&[1, 2, 3], 0).is_err());
    }

}
