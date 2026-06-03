// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! D.0.7 — per-GOP chunk framing for streaming H.264 stego.
//!
//! Multi-GOP streaming stego splits the (already-encrypted+CRC'd)
//! payload bytes across GOP boundaries. Each GOP carries one chunk
//! wrapped in a 4-byte header so the decoder can reassemble them in
//! order without relying on GOP-iteration order.
//!
//! ## Wire format
//!
//! ```text
//!   chunk_index   : u16  big-endian   (0-based)
//!   total_chunks  : u16  big-endian   (1..=u16::MAX)
//!   payload_len   : u16  big-endian   (inline,   0 ..= 65534)   → header 6 B
//!               OR  0xFFFF (u16) + u32 big-endian actual length
//!                                     (extended, len ≥ 65535)    → header 10 B
//!   payload bytes : exactly payload_len bytes
//! ```
//!
//! ### Why `payload_len` is mandatory (#800, 2026-05-29)
//!
//! The decoder recovers the per-GOP STC `m_total` by brute-force,
//! computing `w = ⌊n_cover / m_total⌋` for each candidate. That floor
//! is **many-to-one**: a contiguous range of `m_total` values share
//! one `w`, and because STC extract is convolutional, the chunk header
//! (the leading syndrome bits) parses *identically* across that whole
//! w-class. With no length field the decoder stopped at the class
//! **minimum** `m_total`, returning a **truncated** payload that passed
//! the header parse but failed the outer `parse_frame` → `FrameCorrupted`
//! (it only round-tripped when the true `m_total` happened to be the
//! class minimum). The explicit length makes [`parse_chunk_frame`]
//! length-strict, so short candidates are rejected and the brute-force
//! lands on the exact `m_total`. `m_total` is always byte-aligned, so
//! it sits on the decoder's 8-bit search grid.
//!
//! The `u16`→`u32` escape mirrors the outer frame's v1/v2 length idiom
//! (`stego::frame`). The sentinel is **`0xFFFF`, not `0x0000`**: empty
//! chunks (`payload_len = 0`) are routine — so `0` must stay a valid
//! inline value. `0xFFFF` unconditionally means "a u32 follows" (no
//! peek-disambiguation); the only cost is that a chunk of *exactly* 65535
//! bytes uses the extended form.
//!
//! Tail handling: under the CAP2.3 concentrate+tail allocator the message
//! occupies GOPs `[0, W)` and `total_chunks = W < n_gops`; the `[W, n_gops)`
//! tail is emitted as PLAIN GOPs (no chunk at all) and the decoder stops at
//! `W`. (The no-plan / pre-protocol path instead pads the tail with
//! empty-payload chunks and `total_chunks = n_gops` — hence empty chunks
//! staying routine.)
//!
//! ## Why chunks
//!
//! Whole-video STC can't stream: the planner needs the full cover
//! before producing any stego output. Per-GOP STC streams: each GOP
//! plans + emits independently. But the message has to span GOPs to
//! preserve total capacity, hence chunks.
//!
//! ## Compatibility
//!
//! A single-GOP video produces exactly one chunk with `chunk_index=0`
//! `total_chunks=1`. The chunk header still exists in this case —
//! the legacy whole-video format (no chunk header) is a different
//! on-the-wire format and is NOT decode-compatible with the streaming
//! format. Mobile encodes ALWAYS use the streaming format starting
//! v1.0 (D.0.7); legacy `phasm_h264_encode` is retired from mobile.
//! There is **no backward compatibility** across the #800 header
//! change — no shipped-to-users H.264 stego predates it, so pre-#800
//! streaming encodes simply fail to decode (cleanly: the misread
//! `payload_len` fails the decoder's `w`/length sanity checks).

use crate::stego::error::StegoError;

/// Bytes consumed by the chunk header in the common (inline-length) form:
/// `chunk_index(2) + total_chunks(2) + payload_len(2)`. Also the smallest
/// possible chunk frame (an empty-payload chunk), hence the decoder's
/// minimum `m_total`.
pub const CHUNK_HEADER_LEN: usize = 6;

/// Bytes consumed by the chunk header in the extended (escaped-length)
/// form: base 4 + `0xFFFF` sentinel(2) + `u32` length(4). Used only when
/// a single GOP carries ≥ 65535 payload bytes.
pub const CHUNK_HEADER_LEN_MAX: usize = 10;

/// `payload_len` u16 value reserved to signal the extended form (a `u32`
/// length follows). Inline lengths are therefore `0 ..= 65534`.
pub const LEN_SENTINEL: u16 = 0xFFFF;

/// Maximum chunks a streaming payload can span. 16-bit index +
/// 16-bit total. Practical limit for v1.0: a 2-hour 4K video at
/// GOP=30 is ~7200 GOPs — well within u16.
pub const MAX_CHUNKS: u16 = u16::MAX;

/// CAP2.3 §7 — default per-GOP soft rate ceiling for the concentrate+tail
/// allocator (`allocate_chunks_concentrate_tail`). **Provisional, NOT yet
/// calibrated** — the real value is set in CAP2.4 (#806) by sweeping per-GOP
/// embedding rate vs. an ML steganalysis detector and picking chance+margin.
///
/// 0.5 (vs the absolute-max 1.0) is the conservative interim: it fills each
/// stego GOP to ~half its probed cap, which (a) leaves headroom so a verify
/// shrink-carry stays inside the stego window instead of overflowing into the
/// plain tail (→ graceful `MessageTooLarge` rather than a wider window), and
/// (b) is stealth-leaning per the 2026-06-01 product decision. A future `fast`
/// opt-in raises it (tighter window, more perf, higher rate).
pub const CAP2_DEFAULT_R_TARGET: f64 = 0.5;

/// Build the on-wire bytes for one chunk.
///
/// Writes the inline `u16` payload length for `payload.len() <= 65534`,
/// or the `0xFFFF` + `u32` extended form otherwise (see module docs).
///
/// Caller is responsible for sizing `payload` to fit the carrier's
/// STC capacity (capacity_bytes - CHUNK_HEADER_LEN). On encode this
/// is enforced by the per-GOP capacity probe before building.
///
/// # Errors
/// * `InvalidVideo` if `chunk_index >= total_chunks`, `total_chunks == 0`,
///   or `payload.len() > u32::MAX`.
pub fn build_chunk_frame(
    chunk_index: u16,
    total_chunks: u16,
    payload: &[u8],
) -> Result<Vec<u8>, StegoError> {
    if total_chunks == 0 {
        return Err(StegoError::InvalidVideo(
            "chunk_frame: total_chunks must be > 0".into(),
        ));
    }
    if chunk_index >= total_chunks {
        return Err(StegoError::InvalidVideo(format!(
            "chunk_frame: chunk_index {chunk_index} >= total_chunks {total_chunks}"
        )));
    }
    if payload.len() > u32::MAX as usize {
        return Err(StegoError::InvalidVideo(format!(
            "chunk_frame: payload {} exceeds u32::MAX", payload.len()
        )));
    }
    // Extended form when the length can't be represented inline as a
    // non-sentinel u16 (i.e. >= LEN_SENTINEL).
    let extended = payload.len() >= LEN_SENTINEL as usize;
    let header_len = if extended { CHUNK_HEADER_LEN_MAX } else { CHUNK_HEADER_LEN };
    let mut out = Vec::with_capacity(header_len + payload.len());
    out.extend_from_slice(&chunk_index.to_be_bytes());
    out.extend_from_slice(&total_chunks.to_be_bytes());
    if extended {
        out.extend_from_slice(&LEN_SENTINEL.to_be_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    } else {
        out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    }
    out.extend_from_slice(payload);
    Ok(out)
}

/// Parse the chunk header off the front of `bytes`. Returns
/// `(chunk_index, total_chunks, payload_slice)` on success.
///
/// **Length-strict** (#800): the returned slice is exactly
/// `payload_len` bytes; if `bytes` is shorter than `header + payload_len`
/// this returns `None`. The streaming decoder relies on this to reject
/// under-extracted (too-small `m_total`) candidates during its
/// brute-force, so it lands on the encoder's exact `m_total`.
///
/// Validates: enough bytes for the header (and the declared payload),
/// `total_chunks > 0`, `chunk_index < total_chunks`, and that an
/// extended-form length genuinely needed the escape (`>= LEN_SENTINEL`).
/// The payload slice may be empty if a GOP carried no message bytes
/// (routine on long-video / short-message — the tail GOPs are padded
/// with empty chunks).
pub fn parse_chunk_frame(bytes: &[u8]) -> Option<(u16, u16, &[u8])> {
    if bytes.len() < CHUNK_HEADER_LEN {
        return None;
    }
    let chunk_index = u16::from_be_bytes([bytes[0], bytes[1]]);
    let total_chunks = u16::from_be_bytes([bytes[2], bytes[3]]);
    if total_chunks == 0 || chunk_index >= total_chunks {
        return None;
    }
    let len_field = u16::from_be_bytes([bytes[4], bytes[5]]);
    let (payload_len, header_len) = if len_field == LEN_SENTINEL {
        if bytes.len() < CHUNK_HEADER_LEN_MAX {
            return None;
        }
        let v = u32::from_be_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        // Extended form must encode a length that couldn't be inline,
        // otherwise the bytes aren't a well-formed extended header.
        if v < LEN_SENTINEL as usize {
            return None;
        }
        (v, CHUNK_HEADER_LEN_MAX)
    } else {
        (len_field as usize, CHUNK_HEADER_LEN)
    };
    let end = header_len.checked_add(payload_len)?;
    if bytes.len() < end {
        return None;
    }
    Some((chunk_index, total_chunks, &bytes[header_len..end]))
}

/// Split `message_bytes` into `total_chunks` evenly-sized pieces.
/// The final chunk may be smaller if the message length doesn't
/// divide evenly. Returns the chunk payload slices (NOT yet framed
/// with `build_chunk_frame` — caller does that per chunk after
/// determining the per-GOP capacity).
///
/// # Errors
/// * `MessageTooLarge` if `total_chunks == 0` or `total_chunks >
///   MAX_CHUNKS`.
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
            // a small message). The empty chunks still need headers so
            // total_chunks is consistent across GOPs.
            out.push(Vec::new());
        } else {
            out.push(message_bytes[start..end].to_vec());
        }
    }
    Ok(out)
}

/// CAP2.2 §4 — **rate-bounded proportional** chunk allocation (the carry-free
/// successor to `split_message_into_chunks`' even-split).
///
/// Given each GOP's accurate byte capacity (`gop_caps`, from the §5 STC-trial /
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
/// NOTE (§14 gate): this fills high-capacity GOPs nearer their STC budget than
/// the even-split did. Wiring it into the encoder must be gated on a round-trip
/// re-verification at these higher per-GOP fills (the cascade-decode ceiling,
/// CASCADE.V2) — the allocator math here is encoder-independent and side-effect
/// free.
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

/// CAP2.3 §4 — rate-bounded **concentrate + plain-tail** allocation. Unlike
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

/// Reassemble parsed chunks into the original message bytes.
/// Input: vector of `(chunk_index, payload)` pairs collected from
/// per-GOP decode passes. Caller has already validated that
/// `total_chunks` is consistent across all chunks.
///
/// Returns `None` if:
/// - Any chunk_index is duplicated
/// - Any chunk_index >= total_chunks
/// - total_chunks unique chunks aren't all present
///
/// On success returns the concatenated payload.
pub fn assemble_chunks(
    mut chunks: Vec<(u16, Vec<u8>)>,
    total_chunks: u16,
) -> Option<Vec<u8>> {
    if total_chunks == 0 || chunks.len() != total_chunks as usize {
        return None;
    }
    chunks.sort_by_key(|(idx, _)| *idx);
    let mut out = Vec::new();
    for (i, (idx, payload)) in chunks.iter().enumerate() {
        if *idx as usize != i {
            // Missing or duplicate chunk_index.
            return None;
        }
        out.extend_from_slice(payload);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- CAP2.2 allocate_chunks_proportional ----

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

    // ── CAP2.3 §4 concentrate+tail allocator ────────────────────────────

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
    fn header_constants_match_wire_layout() {
        // #800: base header is idx(2)+total(2)+len(2)=6; extended adds
        // the 0xFFFF sentinel(2)+u32(4) = 10.
        assert_eq!(CHUNK_HEADER_LEN, 6);
        assert_eq!(CHUNK_HEADER_LEN_MAX, 10);
        assert_eq!(LEN_SENTINEL, 0xFFFF);
    }

    #[test]
    fn build_parse_roundtrip_single_chunk() {
        let payload = b"hello world";
        let framed = build_chunk_frame(0, 1, payload).unwrap();
        assert_eq!(framed.len(), CHUNK_HEADER_LEN + payload.len());
        let (idx, total, slice) = parse_chunk_frame(&framed).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(total, 1);
        assert_eq!(slice, payload);
    }

    #[test]
    fn build_parse_roundtrip_empty_chunk() {
        // Empty trailing chunks are routine (long video / short message).
        // payload_len=0 must stay a valid inline value (the reason the
        // sentinel is 0xFFFF, not 0x0000).
        let framed = build_chunk_frame(2, 5, b"").unwrap();
        assert_eq!(framed.len(), CHUNK_HEADER_LEN);
        let (idx, total, slice) = parse_chunk_frame(&framed).unwrap();
        assert_eq!(idx, 2);
        assert_eq!(total, 5);
        assert!(slice.is_empty());
    }

    #[test]
    fn build_parse_inline_boundary() {
        // 65534 stays inline (6-byte header); 65535 escapes to extended
        // (10-byte header) because 0xFFFF is the sentinel.
        let inline = vec![0xABu8; (LEN_SENTINEL as usize) - 1]; // 65534
        let f_inline = build_chunk_frame(0, 1, &inline).unwrap();
        assert_eq!(f_inline.len(), CHUNK_HEADER_LEN + inline.len());
        let (_, _, s) = parse_chunk_frame(&f_inline).unwrap();
        assert_eq!(s, &inline[..]);

        let escaped = vec![0xCDu8; LEN_SENTINEL as usize]; // 65535
        let f_escaped = build_chunk_frame(0, 1, &escaped).unwrap();
        assert_eq!(f_escaped.len(), CHUNK_HEADER_LEN_MAX + escaped.len());
        let (_, _, s2) = parse_chunk_frame(&f_escaped).unwrap();
        assert_eq!(s2.len(), escaped.len());
        assert_eq!(s2, &escaped[..]);
    }

    #[test]
    fn build_parse_roundtrip_extended_large() {
        // > u16 payload uses the extended u32 length form.
        let payload = vec![0x5Au8; 70_000];
        let framed = build_chunk_frame(0, 1, &payload).unwrap();
        assert_eq!(framed.len(), CHUNK_HEADER_LEN_MAX + payload.len());
        let (idx, total, slice) = parse_chunk_frame(&framed).unwrap();
        assert_eq!((idx, total), (0, 1));
        assert_eq!(slice.len(), 70_000);
        assert_eq!(slice, &payload[..]);
    }

    #[test]
    fn parse_is_length_strict_rejects_truncation() {
        // The #800 core invariant: a frame missing even one payload byte
        // must NOT parse. This is what makes the decoder skip too-small
        // m_total candidates in the w-class and land on the exact one.
        let payload = b"the exact length matters";
        let framed = build_chunk_frame(0, 1, payload).unwrap();
        // Drop the final payload byte → buffer < header + payload_len.
        let truncated = &framed[..framed.len() - 1];
        assert!(parse_chunk_frame(truncated).is_none());
    }

    #[test]
    fn parse_ignores_trailing_bytes_returns_exact_payload() {
        // A buffer longer than header+payload_len returns exactly
        // payload_len bytes (a larger m_total in the same w-class still
        // recovers the correct payload — robustness, though the
        // smallest-first brute-force lands on the exact m_total first).
        let payload = b"abc";
        let mut framed = build_chunk_frame(0, 1, payload).unwrap();
        framed.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]); // garbage tail
        let (idx, total, slice) = parse_chunk_frame(&framed).unwrap();
        assert_eq!((idx, total), (0, 1));
        assert_eq!(slice, payload);
    }

    #[test]
    fn parse_rejects_false_sentinel_with_inline_value() {
        // A 0xFFFF sentinel followed by a u32 < LEN_SENTINEL is malformed
        // (such a length would have been encoded inline) → reject.
        let bad = vec![0, 0, 0, 1, 0xFF, 0xFF, 0, 0, 0, 10, 1, 2, 3];
        assert!(parse_chunk_frame(&bad).is_none());
    }

    #[test]
    fn build_rejects_zero_total_chunks() {
        assert!(build_chunk_frame(0, 0, b"data").is_err());
    }

    #[test]
    fn build_rejects_out_of_range_index() {
        assert!(build_chunk_frame(5, 5, b"data").is_err());
        assert!(build_chunk_frame(10, 5, b"data").is_err());
    }

    #[test]
    fn parse_rejects_short_buffer() {
        assert!(parse_chunk_frame(&[]).is_none());
        assert!(parse_chunk_frame(&[0, 0, 0]).is_none()); // only 3 bytes
    }

    #[test]
    fn parse_rejects_zero_total_chunks() {
        let bad = vec![0, 0, 0, 0, 1, 2, 3]; // chunk_idx=0, total=0, payload
        assert!(parse_chunk_frame(&bad).is_none());
    }

    #[test]
    fn parse_rejects_index_oob() {
        // chunk_idx=5, total=3 → invalid.
        let bad = vec![0, 5, 0, 3, 1, 2, 3];
        assert!(parse_chunk_frame(&bad).is_none());
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

    #[test]
    fn assemble_roundtrip_in_order() {
        let original = b"the quick brown fox";
        let chunks = split_message_into_chunks(original, 3).unwrap();
        let collected: Vec<(u16, Vec<u8>)> =
            chunks.into_iter().enumerate().map(|(i, c)| (i as u16, c)).collect();
        let reassembled = assemble_chunks(collected, 3).unwrap();
        assert_eq!(reassembled.as_slice(), original);
    }

    #[test]
    fn assemble_roundtrip_out_of_order() {
        let original = b"the quick brown fox";
        let chunks = split_message_into_chunks(original, 3).unwrap();
        let mut collected: Vec<(u16, Vec<u8>)> =
            chunks.into_iter().enumerate().map(|(i, c)| (i as u16, c)).collect();
        // Shuffle: [2, 0, 1]
        collected.swap(0, 2);
        let reassembled = assemble_chunks(collected, 3).unwrap();
        assert_eq!(reassembled.as_slice(), original);
    }

    #[test]
    fn assemble_detects_missing_chunk() {
        let collected = vec![(0u16, vec![1, 2]), (2u16, vec![5, 6])];
        // Only 2 chunks supplied for total=3 → mismatched count returns None.
        assert!(assemble_chunks(collected, 3).is_none());
    }

    #[test]
    fn assemble_detects_duplicate_chunk() {
        let collected = vec![
            (0u16, vec![1, 2]),
            (0u16, vec![1, 2]),
            (2u16, vec![5, 6]),
        ];
        // chunk_index=1 missing, 0 duplicated.
        assert!(assemble_chunks(collected, 3).is_none());
    }

    #[test]
    fn assemble_detects_oob_chunk_index() {
        let collected = vec![(0u16, vec![1, 2]), (1u16, vec![3, 4]), (5u16, vec![5, 6])];
        // chunk_index=5 but total=3 → OOB.
        assert!(assemble_chunks(collected, 3).is_none());
    }
}
