// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 video shadow messages (1 shadow, fixed parity).
//!
//! Shadow messages provide plausible deniability for H.264 video stego.
//! Multiple messages can be hidden in a single video, each with a
//! different passphrase. Under coercion the user reveals primary;
//! shadows remain undetectable.
//!
//! ## Layer model — asymmetric
//!
//! - **Primary** uses STC (existing 4-domain orchestrator).
//! - **Shadows** use direct LSB writes + Reed-Solomon error
//!   correction.
//!
//! ## Position selection (wire v2, damage-aware)
//!
//! Spec: `docs/design/video/stego-cover-order-and-damage-costs.md` § 6.
//! A shadow decoder has only the stego video and its passphrase, so it
//! cannot compute the J-UNIWARD detectability (that needs natural pixels).
//! It CAN compute the damage `D = E · T · X` (`stego::damage`): position,
//! transform, QP, clamped `|c|`, MB intra flags, frame position in the
//! key-frame segment and MB location are all syntax. So:
//!
//! 1. **Eligible**: CoeffSignBypass and CoeffSuffixLsb positions the damage
//!    model prices (AC coefficients of Luma4x4 / Intra16x16-AC / ChromaAc)
//!    with `log2(D / D_ref) ≤ SHADOW_MAX_LOG2`. **Excluded**: every DC
//!    (Luma4x4 coefficient 0, chroma DC, Intra16x16 luma DC — a DC sign
//!    flip shifts a whole block / MB), 8×8, every position whose D is above
//!    the threshold, and both MVD domains (wet for the primary; MVD damage is
//!    MV-predictor propagation the model does not price).
//! 2. **Order**: low-damage buckets first ([`damage::shadow_bucket`]), the
//!    existing keyed priority `ChaCha20(shadow_perm_seed, position_key)`
//!    within a bucket.
//!
//! **Invariance**: every feature is unchanged by every layer's flips —
//! sign flips keep `|c|`; CoeffSuffixLsb flips move `|c|` by ±1 only at
//! `|c| ≥ 16` (never below 16) and `|c|` is clamped at 15; MB types and QPs
//! are untouched by the wire-only overrides. So the encoder (over the
//! primary-emit cover) and the decoder (over the final cover) rank the same
//! positions identically.
//!
//! ## Coexistence with primary STC
//!
//! Three rules:
//!
//! 1. **Shadow LSBs are written first** into the per-domain
//!    `cover.bits` arrays at shadow-selected positions. These bits
//!    are the RS-encoded shadow frame.
//! 2. **Primary STC sees `f32::INFINITY` cost at shadow positions**.
//!    Viterbi never flips ∞-cost positions → primary doesn't
//!    modify shadow positions → shadow bits survive.
//! 3. **Inter-shadow collisions** absorb into RS parity.
//!
//! ## Decoder
//!
//! Consumes a pre-walked `DomainCover`: the per-GOP
//! `StreamingDecodeSession` walks each GOP's Annex-B slab (via
//! `walk_annex_b_for_cover_with_options`) into a `DomainCover`, and
//! shadow extract runs over that cover. It brute-forces the 6 parity
//! tiers over hash-priority-selected positions across all 4 domains.
//! Each tier: take top-N positions globally (no biasing), extract
//! LSBs, RS-decode + first-block peek for `fdl`, AES-GCM-SIV
//! validate. First success wins. This is the decoder's tier-1
//! attempt — tried before the Scheme A combined STC extract.

use super::{DomainCover, EmbedDomain};
use super::PositionKey;
use super::damage;
use crate::stego::armor::ecc;
use crate::stego::crypto::{self};
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload::{self, FileEntry, PayloadData};
use crate::stego::progress;
// 2026-05-21 — Video shadow now uses the unified v1/v2 dispatch
// shadow frame (see `crate::stego::shadow_layer` and the
// `frame.rs` primary-frame pattern). v1 (46-byte overhead, u16
// plaintext_len) for small payloads; v2 (50-byte overhead,
// `[0x0000][u32]` sentinel) for plaintexts > u16::MAX. The previous
// u32-only WIDE format was a wire-format change that landed
// pre-release (v0.2.9 ships video stego opt-in) so no back-compat
// burden. Image stego shares the same unified API.
use crate::stego::shadow_layer::{
    build_shadow_frame, compute_max_shadow_fdl, parse_shadow_frame, peek_shadow_fdl,
    MAX_SHADOW_FRAME_BYTES, SHADOW_FRAME_OVERHEAD, SHADOW_PARITY_TIERS,
};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Locator for one shadow position inside a 4-domain `DomainCover`.
/// Indexes the per-domain `bits[]` / `positions[]` / `costs[]`
/// vectors at `(domain, intra_index)`.
#[derive(Debug, Clone, Copy)]
pub struct ShadowSlot {
    pub domain: EmbedDomain,
    pub intra_index: usize,
    /// Keyed priority within the damage bucket.
    pub priority: u32,
    /// v2 damage bucket ([`damage::shadow_bucket`]); low first.
    pub bucket: u8,
}

/// State for one shadow layer during encoding.
#[derive(Debug)]
pub struct ShadowState {
    /// Selected shadow positions across all 4 domains, sorted by
    /// hash priority (lowest first, top-N total).
    pub positions: Vec<ShadowSlot>,
    /// Desired LSB bits at those positions — the RS-encoded frame.
    pub bits: Vec<u8>,
    /// Total bits = `RS-encoded shadow frame length × 8`.
    pub n_total: usize,
    /// RS parity length used for encode (decoder brute-forces).
    pub parity_len: usize,
    /// Unencoded frame byte count (before RS encoding).
    pub frame_data_len: usize,
}

/// Try one (rs_bytes, fdl, parity, passphrase) candidate. Returns
/// `Some(Ok(payload))` on success, `None` on any failure (RS,
/// frame parse, or AES-GCM-SIV authentication).
///
/// Takes the pre-byte-packed cover (computed once per
/// `shadow_extract*` call) rather than re-converting bits to
/// bytes on every brute-force iteration. Format-aware O(1) gate via
/// [`peek_shadow_fdl`]: dispatches v1 vs v2 on the decoded prefix,
/// computes expected total frame length, rejects every wrong-fdl
/// candidate before running AES.
fn try_single_fdl(
    rs_bytes: &[u8],
    fdl: usize,
    parity_len: usize,
    passphrase: &str,
) -> Option<Result<PayloadData, StegoError>> {
    let rs_encoded_len = ecc::rs_encoded_len_with_parity(fdl, parity_len);
    if rs_encoded_len > rs_bytes.len() {
        return None;
    }
    let decoded = match ecc::rs_decode_blocks_with_parity(
        &rs_bytes[..rs_encoded_len],
        fdl,
        parity_len,
    ) {
        Ok((data, _)) => data,
        Err(_) => return None,
    };
    // O(1) format-aware consistency gate. peek_shadow_fdl reads the
    // first 2-6 bytes, dispatches v1 (u16 len at bytes 0..2) vs v2
    // (sentinel 0x0000 at bytes 0..2 + u32 len at bytes 2..6), and
    // returns the total frame length the producer encoded. If that
    // doesn't equal the brute-force fdl candidate, reject without
    // running parse + AES. (2026-05-21 unification; supersedes the
    // earlier u32-only gate.)
    let expected_total = peek_shadow_fdl(&decoded)?;
    if expected_total != fdl {
        return None;
    }
    let fr = parse_shadow_frame(&decoded).ok()?;
    match crypto::decrypt(&fr.ciphertext, passphrase, &fr.salt, &fr.nonce) {
        Ok(plaintext) => {
            let len = fr.plaintext_len as usize;
            if len > plaintext.len() {
                return None;
            }
            Some(payload::decode_payload(&plaintext[..len]))
        }
        Err(_) => None,
    }
}

/// First-block peek: decode the first 255 RS bytes to read the
/// `plaintext_len` prefix and derive the exact `fdl`. Returns the
/// candidate `fdl` if it's plausible (>= k, within capacity).
///
/// 2026-05-21 unification — delegates v1/v2 dispatch to the shared
/// [`peek_shadow_fdl`] helper in `shadow_layer`. Same dispatch
/// logic for image + video shadow.
fn peek_fdl_from_first_block(
    rs_bytes: &[u8],
    parity_len: usize,
    max_fdl: usize,
) -> Option<usize> {
    let k = 255usize.saturating_sub(parity_len);
    if k < 2 || rs_bytes.len() < 255 {
        return None;
    }
    let (data, _) =
        ecc::rs_decode_blocks_with_parity(&rs_bytes[..255], k, parity_len).ok()?;
    let fdl = peek_shadow_fdl(&data)?;
    if fdl >= k && fdl <= max_fdl {
        Some(fdl)
    } else {
        None
    }
}

/// WV.6.g.5 — shared shadow-decode tail. Given the **priority-ordered** cover
/// LSBs (one bit per `priority_slots` entry, in that ascending priority order)
/// and the passphrase, bit-pack once and run the parity-tier × fdl
/// peek/brute-force RS-decode search. Returns the recovered payload or
/// [`StegoError::FrameCorrupted`].
///
/// This is the cover-shape-independent half of [`shadow_extract`]: the
/// whole-clip extractor feeds it `bits[priority_slots(whole_cover)]`, and the
/// WV.6.g.5 streaming verify feeds it the **emitted** clip's priority-ordered
/// bits captured per-GOP (no whole-clip walk). Both must produce identical
/// decode decisions — which holds because the decode depends only on the
/// priority-ordered bit stream, not on how it was assembled.
pub(crate) fn decode_shadow_from_priority_lsbs(
    all_lsbs: &[u8],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    // Bit-pack the full LSB stream ONCE (not per parity tier nor per fdl
    // candidate); try_single_fdl + peek slice into this buffer.
    let max_rs_bytes = all_lsbs.len() / 8;
    let all_rs_bytes = frame::bits_to_bytes(&all_lsbs[..max_rs_bytes * 8]);

    for &parity_len in &SHADOW_PARITY_TIERS {
        progress::check_cancelled()?;
        let k = 255usize.saturating_sub(parity_len);
        // Need ≥4 bytes recovered from first RS block to read the
        // u32 BE plaintext_len prefix. With k<4 the peek can't function;
        // skip the tier (also unreachable for SHADOW_PARITY_TIERS — even
        // parity=128 gives k=127).
        if k < 4 {
            continue;
        }
        let max_fdl = compute_max_shadow_fdl(max_rs_bytes, parity_len)
            .min(MAX_SHADOW_FRAME_BYTES);
        if SHADOW_FRAME_OVERHEAD > max_fdl {
            continue;
        }

        // Peek path (works for fdl ≥ k, covers most real shadows
        // including the >250-byte payloads that brute-force can't reach).
        let peeked = peek_fdl_from_first_block(&all_rs_bytes, parity_len, max_fdl);
        if let Some(fdl) = peeked
            && let Some(result) = try_single_fdl(&all_rs_bytes, fdl, parity_len, passphrase)
        {
            return result;
        }

        let small_max = (k - 1).min(max_fdl);
        if SHADOW_FRAME_OVERHEAD > small_max {
            continue;
        }
        for fdl in SHADOW_FRAME_OVERHEAD..=small_max {
            // Skip the peek-tried fdl in the brute-force fallback.
            // If peek returned Some(fdl) and try_single_fdl rejected it,
            // re-trying gains nothing.
            if Some(fdl) == peeked {
                continue;
            }
            // Up to ~250 RS decodes per tier: the last stretch of a video
            // decode, all inside one blocking call.
            progress::check_cancelled()?;
            if let Some(result) = try_single_fdl(&all_rs_bytes, fdl, parity_len, passphrase) {
                return result;
            }
        }
    }

    Err(StegoError::FrameCorrupted)
}

// ─── Shadow position selection (v2, damage-aware) ────────────────

/// THE single source of truth for shadow-eligible cover positions, their
/// damage bucket and their iteration order. Invokes `f(domain,
/// intra_index, key, bucket)` for every eligible position, CoeffSignBypass
/// then CoeffSuffixLsb, each in `positions[]` order (`intra_index` is the
/// `enumerate()` index). Eligibility and buckets per the module doc (§ 6),
/// from `cover.mb_syntax` via [`damage::DamageModel`]. [`priority_slots`],
/// the streaming selector, the streaming verify, the streaming decoder and
/// the capacity probe's pool count all route through this, so their orders
/// can never diverge — a divergence would silently break shadow decode.
pub(crate) fn for_each_eligible_position(
    cover: &DomainCover,
    f: impl FnMut(EmbedDomain, usize, &PositionKey, u8),
) {
    let model = damage::DamageModel::from_cover(cover);
    for_each_eligible_position_with(cover, &model, f);
}

/// [`for_each_eligible_position`] with a prebuilt model of `cover`.
pub(crate) fn for_each_eligible_position_with(
    cover: &DomainCover,
    model: &damage::DamageModel,
    mut f: impl FnMut(EmbedDomain, usize, &PositionKey, u8),
) {
    let _prof = crate::codec::h264::profile::scope("v2.shadow_eligible");
    for (domain, bits) in [
        (EmbedDomain::CoeffSignBypass, &cover.coeff_sign_bypass),
        (EmbedDomain::CoeffSuffixLsb, &cover.coeff_suffix_lsb),
    ] {
        for (intra_index, key) in bits.positions.iter().enumerate() {
            let mag = bits.magnitudes.get(intra_index).copied().unwrap_or(0);
            if let Some(bucket) = model
                .damage(*key, mag, damage::SHADOW_MAG_CAP)
                .and_then(damage::shadow_bucket)
            {
                f(domain, intra_index, key, bucket);
            }
        }
    }
}

/// Every eligible position, in the shadow total order: `(bucket, keyed
/// priority)` ascending, ties (equal priority within a bucket) in
/// [`for_each_eligible_position`] order — i.e. `(bucket, priority,
/// domain, intra_index)`.
pub(super) fn priority_slots(cover: &DomainCover, perm_seed: &[u8; 32]) -> Vec<ShadowSlot> {
    let mut rng = ChaCha20Rng::from_seed(*perm_seed);
    let mut slots = Vec::new();
    for_each_eligible_position(cover, |domain, intra_index, key, bucket| {
        rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
        slots.push(ShadowSlot { domain, intra_index, priority: rng.next_u32(), bucket });
    });
    slots.sort_by_key(|s| (s.bucket, s.priority));
    slots
}

/// One retained slot inside [`StreamingTopN`], ordered by the
/// `priority_slots` total order: `(bucket, priority, domain as u8,
/// intra_index)` ascending. `BinaryHeap` is a *max*-heap, so the natural
/// `Ord` below makes it evict the largest tuple — retaining the smallest
/// `capacity`, which is exactly the top-N `priority_slots` keeps after its
/// ascending sort. The tuple is unique across all positions (`domain` +
/// `intra_index` identify one), so the order is strict and the retained set
/// is independent of push order.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(any(test, feature = "h264-encoder")), allow(dead_code))] // used by the encoder-side streaming shadow path
struct TopNEntry {
    bucket: u8,
    priority: u32,
    domain_rank: u8,
    intra_index: usize,
    domain: EmbedDomain,
}

#[cfg_attr(not(any(test, feature = "h264-encoder")), allow(dead_code))]
impl TopNEntry {
    #[inline]
    fn order_key(&self) -> (u8, u32, u8, usize) {
        (self.bucket, self.priority, self.domain_rank, self.intra_index)
    }
}

impl Ord for TopNEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.order_key().cmp(&other.order_key())
    }
}
impl PartialOrd for TopNEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Streaming top-N shadow position selector — the O(N)-memory equivalent of
/// `priority_slots(..)[..n]`, for the streaming shadow path (no whole-clip
/// `Vec<ShadowSlot>` materialised). Push every eligible `(domain,
/// intra_index, key, bucket)` — from [`for_each_eligible_position`] — in any
/// order; [`StreamingTopN::into_sorted`] emits exactly the slots, in the
/// same order, that `priority_slots(..)[..n]` would: the priority of a
/// position is `ChaCha20(perm_seed).seek(key.raw()*2).next_u32()`, position-
/// local and order-independent, and the bucket is a function of the
/// position's syntax alone.
#[cfg_attr(not(any(test, feature = "h264-encoder")), allow(dead_code))]
pub(crate) struct StreamingTopN {
    rng: ChaCha20Rng,
    capacity: usize,
    heap: BinaryHeap<TopNEntry>,
}

#[cfg_attr(not(any(test, feature = "h264-encoder")), allow(dead_code))]
impl StreamingTopN {
    pub(crate) fn new(perm_seed: &[u8; 32], capacity: usize) -> Self {
        Self {
            rng: ChaCha20Rng::from_seed(*perm_seed),
            capacity,
            heap: BinaryHeap::with_capacity(capacity.saturating_add(1)),
        }
    }

    /// Offer one eligible cover position (`intra_index` = its index within
    /// its domain's `positions[]`, `bucket` its damage bucket).
    pub(crate) fn push(
        &mut self,
        domain: EmbedDomain,
        intra_index: usize,
        key: PositionKey,
        bucket: u8,
    ) {
        if self.capacity == 0 {
            return;
        }
        self.rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
        let entry = TopNEntry {
            bucket,
            priority: self.rng.next_u32(),
            domain_rank: domain as u8,
            intra_index,
            domain,
        };
        if self.heap.len() < self.capacity {
            self.heap.push(entry);
        } else if let Some(&max) = self.heap.peek() {
            if entry < max {
                self.heap.pop();
                self.heap.push(entry);
            }
        }
    }

    /// Drain into the final `Vec<ShadowSlot>`, ascending in the
    /// `priority_slots` total order.
    pub(crate) fn into_sorted(self) -> Vec<ShadowSlot> {
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|e| ShadowSlot {
                domain: e.domain,
                intra_index: e.intra_index,
                priority: e.priority,
                bucket: e.bucket,
            })
            .collect()
    }
}

/// One retained slot inside [`ShadowBitTopN`] — the [`TopNEntry`] order plus
/// the cover **bit** riding along. Ord/Eq ignore `bit`.
#[derive(Clone, Copy)]
#[cfg_attr(not(any(test, feature = "h264-encoder")), allow(dead_code))]
struct BitTopNEntry {
    bucket: u8,
    priority: u32,
    domain_rank: u8,
    intra_index: usize,
    bit: u8,
}
#[cfg_attr(not(any(test, feature = "h264-encoder")), allow(dead_code))]
impl BitTopNEntry {
    #[inline]
    fn order_key(&self) -> (u8, u32, u8, usize) {
        (self.bucket, self.priority, self.domain_rank, self.intra_index)
    }
}
impl PartialEq for BitTopNEntry {
    fn eq(&self, other: &Self) -> bool {
        self.order_key() == other.order_key()
    }
}
impl Eq for BitTopNEntry {}
impl Ord for BitTopNEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.order_key().cmp(&other.order_key())
    }
}
impl PartialOrd for BitTopNEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Bit-carrying twin of [`StreamingTopN`] for the streaming verify: fed the
/// **emitted** cover's eligible positions one GOP at a time (each with its
/// emitted bit), keeps the top `capacity` in the `priority_slots` order, and
/// [`into_sorted_bits`](Self::into_sorted_bits) drains the retained bits in
/// that order — exactly the `all_lsbs` prefix
/// [`decode_shadow_from_priority_lsbs`] consumes.
#[cfg_attr(not(any(test, feature = "h264-encoder")), allow(dead_code))]
pub(crate) struct ShadowBitTopN {
    rng: ChaCha20Rng,
    capacity: usize,
    heap: BinaryHeap<BitTopNEntry>,
}

#[cfg_attr(not(any(test, feature = "h264-encoder")), allow(dead_code))]
impl ShadowBitTopN {
    pub(crate) fn new(perm_seed: &[u8; 32], capacity: usize) -> Self {
        Self {
            rng: ChaCha20Rng::from_seed(*perm_seed),
            capacity,
            heap: BinaryHeap::with_capacity(capacity.saturating_add(1)),
        }
    }

    /// Offer one eligible emitted position with its cover `bit`.
    pub(crate) fn push(
        &mut self,
        domain: EmbedDomain,
        intra_index: usize,
        key: PositionKey,
        bucket: u8,
        bit: u8,
    ) {
        if self.capacity == 0 {
            return;
        }
        self.rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
        let entry = BitTopNEntry {
            bucket,
            priority: self.rng.next_u32(),
            domain_rank: domain as u8,
            intra_index,
            bit,
        };
        if self.heap.len() < self.capacity {
            self.heap.push(entry);
        } else if let Some(&max) = self.heap.peek() {
            if entry < max {
                self.heap.pop();
                self.heap.push(entry);
            }
        }
    }

    /// Drain the retained bits in ascending `priority_slots` order.
    pub(crate) fn into_sorted_bits(self) -> Vec<u8> {
        self.heap.into_sorted_vec().into_iter().map(|e| e.bit).collect()
    }
}

/// WV.6.g.4.2b — the cover-INDEPENDENT half of [`prepare_shadow`]: build the
/// shadow's RS-encoded frame (the bits to embed) for a parity tier, decoupled
/// from cover selection. Returns `(rs_bits, frame_data_len)`; `n_total =
/// rs_bits.len()`. The streaming Sweep B pairs these bits with the
/// `StreamingTopN` selection (the cover-side half) instead of re-running
/// `priority_slots` — which needs the whole cover it no longer holds.
///
/// Length is deterministic (Brotli + AES-GCM-SIV add fixed-length overhead), so
/// `n_total` is the same across runs; only the ciphertext bytes vary with the
/// random salt+nonce (pinned under `PHASM_DETERMINISTIC_SEED`).
pub fn build_shadow_rs_frame(
    shadow_pass: &str,
    message: &str,
    files: &[FileEntry],
    parity_len: usize,
) -> Result<(Vec<u8>, usize), StegoError> {
    let payload_bytes = payload::encode_payload(message, files)?;
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, shadow_pass)?;
    let frame_bytes = build_shadow_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
    let frame_data_len = frame_bytes.len();
    let rs_bytes = ecc::rs_encode_blocks_with_parity(&frame_bytes, parity_len);
    let rs_bits = frame::bytes_to_bits(&rs_bytes);
    Ok((rs_bits, frame_data_len))
}

/// Select a shadow's positions over `cover` (the top `n_total` of
/// [`priority_slots`]) and pair them with its RS-encoded frame.
pub fn prepare_shadow(
    cover: &DomainCover,
    shadow_pass: &str,
    message: &str,
    files: &[FileEntry],
    parity_len: usize,
) -> Result<ShadowState, StegoError> {
    let (rs_bits, frame_data_len) =
        build_shadow_rs_frame(shadow_pass, message, files, parity_len)?;
    let n_total = rs_bits.len();

    let perm_seed = crypto::derive_shadow_structural_key(shadow_pass)?;
    let slots = priority_slots(cover, &perm_seed);

    if slots.len() < n_total {
        return Err(StegoError::MessageTooLarge);
    }

    let positions = slots.into_iter().take(n_total).collect();

    Ok(ShadowState {
        positions,
        bits: rs_bits,
        n_total,
        parity_len,
        frame_data_len,
    })
}

/// 4-domain shadow LSB injection — write shadow bits into the
/// per-domain cover bit arrays. Run BEFORE primary STC plans so
/// primary's Viterbi sees shadow bits as natural cover bits;
/// combined with [`overlay_infinity_costs`] the primary plan
/// keeps shadow bits at shadow positions.
pub fn embed_shadow_lsb(
    coeff_sign_bypass_bits: &mut [u8],
    coeff_suffix_lsb_bits: &mut [u8],
    mvd_sign_bypass_bits: &mut [u8],
    mvd_suffix_lsb_bits: &mut [u8],
    state: &ShadowState,
) {
    for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
        let bit = state.bits[i];
        match slot.domain {
            EmbedDomain::CoeffSignBypass => {
                coeff_sign_bypass_bits[slot.intra_index] = bit;
            }
            EmbedDomain::CoeffSuffixLsb => {
                coeff_suffix_lsb_bits[slot.intra_index] = bit;
            }
            EmbedDomain::MvdSignBypass => {
                mvd_sign_bypass_bits[slot.intra_index] = bit;
            }
            EmbedDomain::MvdSuffixLsb => {
                mvd_suffix_lsb_bits[slot.intra_index] = bit;
            }
        }
    }
}

/// 4-domain ∞-cost overlay — set `f32::INFINITY` at each shadow
/// position in the corresponding per-domain cost vector. Primary
/// STC's Viterbi avoids flipping ∞-cost positions, preserving the
/// shadow bits injected via [`embed_shadow_lsb`].
pub fn overlay_infinity_costs(
    coeff_sign_bypass_cost: &mut [f32],
    coeff_suffix_lsb_cost: &mut [f32],
    mvd_sign_bypass_cost: &mut [f32],
    mvd_suffix_lsb_cost: &mut [f32],
    state: &ShadowState,
) {
    for slot in state.positions.iter().take(state.n_total) {
        match slot.domain {
            EmbedDomain::CoeffSignBypass => {
                coeff_sign_bypass_cost[slot.intra_index] = f32::INFINITY;
            }
            EmbedDomain::CoeffSuffixLsb => {
                coeff_suffix_lsb_cost[slot.intra_index] = f32::INFINITY;
            }
            EmbedDomain::MvdSignBypass => {
                mvd_sign_bypass_cost[slot.intra_index] = f32::INFINITY;
            }
            EmbedDomain::MvdSuffixLsb => {
                mvd_suffix_lsb_cost[slot.intra_index] = f32::INFINITY;
            }
        }
    }
}

/// 4-domain plan stamp — defensive override of `DomainPlan` bits
/// at shadow positions with the shadow's RS-encoded LSBs.
/// Primary STC's plan should already carry the shadow bits at
/// shadow positions (via cover-bit injection + ∞-cost overlay
/// before STC plans); this is a defensive guard against any
/// future plan-layer drift.
pub fn apply_shadow_to_plan(
    coeff_sign_bypass: &mut [u8],
    coeff_suffix_lsb: &mut [u8],
    mvd_sign_bypass: &mut [u8],
    mvd_suffix_lsb: &mut [u8],
    state: &ShadowState,
) {
    for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
        let bit = state.bits[i];
        match slot.domain {
            EmbedDomain::CoeffSignBypass => coeff_sign_bypass[slot.intra_index] = bit,
            EmbedDomain::CoeffSuffixLsb => coeff_suffix_lsb[slot.intra_index] = bit,
            EmbedDomain::MvdSignBypass => mvd_sign_bypass[slot.intra_index] = bit,
            EmbedDomain::MvdSuffixLsb => mvd_suffix_lsb[slot.intra_index] = bit,
        }
    }
}

/// Translate a [`ShadowState`]'s positions from
/// `source_cover` indexing to `(mvd_target, coeff_target)` indexing
/// via [`PositionKey`] lookup.
///
/// MVD-domain slots use `mvd_target.mvd_*.positions`; coeff-domain
/// slots use `coeff_target.coeff_*.positions`. Slots whose
/// `PositionKey` doesn't appear in the corresponding target are
/// dropped (rare boundary drift case; cascade absorbs).
///
/// The returned state retains the same `parity_len` /
/// `frame_data_len`. `n_total` is updated to reflect the dropped
/// slots; `positions` and `bits` are aligned (slot[i] ↔ bits[i]).
///
/// For backends where the MVD cover and coeff cover come from the
/// same walk (e.g. OH264 wire_only — encoder state stays clean so a
/// single 4-domain cover serves both), pass the same `DomainCover`
/// reference for both `mvd_target` and `coeff_target`.
pub fn translate_shadow_state(
    state: &ShadowState,
    source_cover: &DomainCover,
    mvd_target: &DomainCover,
    coeff_target: &DomainCover,
) -> ShadowState {
    use std::collections::HashMap;
    let build_map = |positions: &[PositionKey]| -> HashMap<PositionKey, usize> {
        positions.iter().enumerate().map(|(i, &k)| (k, i)).collect()
    };
    let target_csb = build_map(&coeff_target.coeff_sign_bypass.positions);
    let target_csl = build_map(&coeff_target.coeff_suffix_lsb.positions);
    let target_msb = build_map(&mvd_target.mvd_sign_bypass.positions);
    let target_msl = build_map(&mvd_target.mvd_suffix_lsb.positions);

    let mut out_positions = Vec::with_capacity(state.positions.len());
    let mut out_bits = Vec::with_capacity(state.bits.len());

    for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
        if i >= state.bits.len() {
            break;
        }
        let pk_opt = match slot.domain {
            EmbedDomain::CoeffSignBypass =>
                source_cover.coeff_sign_bypass.positions.get(slot.intra_index),
            EmbedDomain::CoeffSuffixLsb =>
                source_cover.coeff_suffix_lsb.positions.get(slot.intra_index),
            EmbedDomain::MvdSignBypass =>
                source_cover.mvd_sign_bypass.positions.get(slot.intra_index),
            EmbedDomain::MvdSuffixLsb =>
                source_cover.mvd_suffix_lsb.positions.get(slot.intra_index),
        };
        let pk = match pk_opt {
            Some(&k) => k,
            None => continue,
        };
        let target_idx = match slot.domain {
            EmbedDomain::CoeffSignBypass => target_csb.get(&pk).copied(),
            EmbedDomain::CoeffSuffixLsb => target_csl.get(&pk).copied(),
            EmbedDomain::MvdSignBypass => target_msb.get(&pk).copied(),
            EmbedDomain::MvdSuffixLsb => target_msl.get(&pk).copied(),
        };
        if let Some(target_idx) = target_idx {
            out_positions.push(ShadowSlot {
                domain: slot.domain,
                intra_index: target_idx,
                priority: slot.priority,
                bucket: slot.bucket,
            });
            out_bits.push(state.bits[i]);
        }
    }

    let n_total = out_bits.len();
    ShadowState {
        positions: out_positions,
        bits: out_bits,
        n_total,
        parity_len: state.parity_len,
        frame_data_len: state.frame_data_len,
    }
}

// ─── Single-cover over primary-emit cover ───────────────────────
//
// This architecture (locked 2026-04-28) selects shadow
// positions over the PRIMARY-EMIT COVER — the cover obtained by
// running a primary-only provisional emit and walking the bytes
// via the streaming walker
// (`walk_annex_b_for_cover_with_options`). On the live OH264 path
// this provisional emit + walk is done inline in
// `openh264_stego.rs` (the pure-Rust `provisional_emit::pass3_emit_provisional`
// helper this once referenced was deleted with the pure-Rust
// encoder in the 2026-06 video-retirement).
//
// This matches the cover the DECODER sees: decoder priority-sorts
// shadow positions over the FINAL EMIT cover, which differs from
// the primary-emit cover only at shadow-override positions (where
// shadow LSBs flip primary's bit values, but don't change set
// membership beyond rare boundary cases that cascade absorbs).

/// Shadow preparation over the primary-emit cover (the cover the decoder
/// sees, up to the shadow flips themselves — which move no feature the
/// selection reads, module doc).
pub fn prepare_shadow_over_emit_cover(
    primary_emit_cover: &DomainCover,
    shadow_pass: &str,
    message: &str,
    files: &[FileEntry],
    parity_len: usize,
) -> Result<ShadowState, StegoError> {
    prepare_shadow(primary_emit_cover, shadow_pass, message, files, parity_len)
}

/// Whole-cover shadow extract: read the cover bits in the `priority_slots`
/// order and run the parity-tier × fdl search.
pub fn shadow_extract(cover: &DomainCover, passphrase: &str) -> Result<PayloadData, StegoError> {
    if cover.total_len() == 0 {
        return Err(StegoError::FrameCorrupted);
    }

    let perm_seed = crypto::derive_shadow_structural_key(passphrase)?;
    let slots = priority_slots(cover, &perm_seed);

    let all_lsbs: Vec<u8> = slots
        .iter()
        .map(|slot| match slot.domain {
            EmbedDomain::CoeffSignBypass => cover.coeff_sign_bypass.bits[slot.intra_index],
            EmbedDomain::CoeffSuffixLsb => cover.coeff_suffix_lsb.bits[slot.intra_index],
            EmbedDomain::MvdSignBypass => cover.mvd_sign_bypass.bits[slot.intra_index],
            EmbedDomain::MvdSuffixLsb => cover.mvd_suffix_lsb.bits[slot.intra_index],
        })
        .collect();

    // WV.6.g.5 — the decode tail is shared with the streaming verify.
    decode_shadow_from_priority_lsbs(&all_lsbs, passphrase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::stego::damage::{MbSyntax, MbSyntaxMap, MB_INTRA, MB_KEY};
    use crate::codec::h264::stego::hook::{BinKind, SyntaxPath};

    // ─── v2 — damage-aware selection; StreamingTopN / ShadowBitTopN ≡
    //     priority_slots(..)[..n] ──────────────────────────────────────

    const MB_W: u32 = 20;
    const MB_H: u32 = 12;
    const FRAMES: u32 = 6;

    /// A key frame then P frames, every MB of the key frame intra, a few
    /// intra MBs in the P frames.
    fn syntax() -> MbSyntaxMap {
        let mut mbs = Vec::new();
        for f in 0..FRAMES {
            for a in 0..MB_W * MB_H {
                let intra = f == 0 || a % 37 == 0;
                mbs.push(MbSyntax {
                    frame_idx: f,
                    mb_addr: a,
                    qp_y: 26,
                    qp_c: 26,
                    flags: if intra { MB_INTRA } else { 0 } | if f == 0 { MB_KEY } else { 0 },
                });
            }
        }
        MbSyntaxMap { mb_w: MB_W, mb_h: MB_H, mbs }
    }

    fn key(domain: EmbedDomain, i: usize) -> PositionKey {
        let kind = if domain == EmbedDomain::CoeffSuffixLsb { BinKind::SuffixLsb } else { BinKind::Sign };
        PositionKey::new(
            (i as u32) % FRAMES,
            (i as u32 * 7 + 1) % (MB_W * MB_H),
            domain,
            SyntaxPath::Luma4x4 { block_idx: (i % 16) as u8, coeff_idx: (i % 16) as u8, kind },
        )
    }

    /// Cover with `k` positions per domain (distinct keys, varied |c| and
    /// bits), MB syntax attached.
    fn build_cover(k: usize) -> DomainCover {
        let mut c = DomainCover::default();
        for i in 0..k {
            let mag = 1 + (i % 19) as u16;
            c.coeff_sign_bypass.push_with_magnitude((i & 1) as u8, key(EmbedDomain::CoeffSignBypass, i), mag);
            c.coeff_suffix_lsb.push_with_magnitude(((i ^ 7) & 1) as u8, key(EmbedDomain::CoeffSuffixLsb, i), 16 + mag);
            c.mvd_sign_bypass.push(((i ^ 13) & 1) as u8, key(EmbedDomain::MvdSignBypass, i));
            c.mvd_suffix_lsb.push(((i ^ 21) & 1) as u8, key(EmbedDomain::MvdSuffixLsb, i));
        }
        c.mb_syntax = syntax();
        c
    }

    fn eligible(cover: &DomainCover) -> Vec<(EmbedDomain, usize, PositionKey, u8)> {
        let mut v = Vec::new();
        for_each_eligible_position(cover, |d, i, k, b| v.push((d, i, *k, b)));
        v
    }

    fn assert_slots_eq(got: &[ShadowSlot], want: &[ShadowSlot]) {
        assert_eq!(got.len(), want.len(), "slot count mismatch");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                (g.domain, g.intra_index, g.priority, g.bucket),
                (w.domain, w.intra_index, w.priority, w.bucket),
                "slot mismatch at rank {i}"
            );
        }
    }

    /// § 6 eligibility: no DC, no MVD, no unpriced position; buckets
    /// non-decreasing in the priority order; the bucket rank puts the
    /// low-damage positions first.
    #[test]
    fn eligibility_excludes_dc_mvd_and_orders_by_bucket() {
        let cover = build_cover(400);
        let el = eligible(&cover);
        assert!(!el.is_empty());
        for (d, _, k, _) in &el {
            assert!(matches!(d, EmbedDomain::CoeffSignBypass | EmbedDomain::CoeffSuffixLsb));
            match k.syntax_path() {
                SyntaxPath::Luma4x4 { coeff_idx, .. } => assert_ne!(coeff_idx, 0, "DC must be excluded"),
                p => panic!("unexpected path {p:?}"),
            }
        }
        let slots = priority_slots(&cover, &[9u8; 32]);
        assert_eq!(slots.len(), el.len());
        assert!(slots.windows(2).all(|w| (w[0].bucket, w[0].priority) <= (w[1].bucket, w[1].priority)));
        let buckets: std::collections::BTreeSet<u8> = slots.iter().map(|s| s.bucket).collect();
        assert!(buckets.len() > 1, "fixture should span several damage buckets");
        // Key-frame edge-constant flips at the top MB row are the floods:
        // they must not be eligible.
        let model = damage::DamageModel::from_cover(&cover);
        let flood = PositionKey::new(
            0, 0, EmbedDomain::CoeffSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 1, kind: BinKind::Sign },
        );
        assert!(model.damage(flood, 1, damage::SHADOW_MAG_CAP).and_then(damage::shadow_bucket).is_none());
    }

    /// No synthetic MB syntax ⇒ nothing is priced ⇒ nothing is eligible
    /// (fail closed rather than fall back to a cost-blind rule).
    #[test]
    fn cover_without_syntax_has_no_eligible_positions() {
        let mut cover = build_cover(50);
        cover.mb_syntax = MbSyntaxMap::default();
        assert!(eligible(&cover).is_empty());
    }

    /// Invariance (§ 6): the primary's and the other shadows' flips change no
    /// feature the selection reads — flipping every sign bit, and moving
    /// every CSL |c| by ±1 within |c| ≥ 16, leaves the slots unchanged.
    #[test]
    fn selection_is_invariant_under_every_layers_flips() {
        let cover = build_cover(300);
        let mut flipped = cover.clone();
        for b in flipped.coeff_sign_bypass.bits.iter_mut() {
            *b ^= 1;
        }
        for (i, m) in flipped.coeff_suffix_lsb.magnitudes.iter_mut().enumerate() {
            *m = if i % 2 == 0 { *m + 1 } else { (*m - 1).max(16) };
        }
        // Sign positions see the |c| of a coefficient whose suffix LSB moved.
        for m in flipped.coeff_sign_bypass.magnitudes.iter_mut() {
            if *m >= 16 {
                *m += 1;
            }
        }
        let seed = [0x77u8; 32];
        assert_slots_eq(&priority_slots(&flipped, &seed), &priority_slots(&cover, &seed));
    }

    #[test]
    fn streaming_topn_matches_priority_slots() {
        let cover = build_cover(300);
        let seed = [0x5Au8; 32];
        let full = priority_slots(&cover, &seed);
        let n = full.len() / 2;
        let mut sel = StreamingTopN::new(&seed, n);
        // Offer the eligible set in reverse order to prove order-independence.
        for (d, i, k, b) in eligible(&cover).into_iter().rev() {
            sel.push(d, i, k, b);
        }
        assert_slots_eq(&sel.into_sorted(), &full[..n]);
    }

    #[test]
    fn streaming_topn_capacity_exceeds_eligible_returns_all_sorted() {
        let cover = build_cover(40);
        let seed = [0x11u8; 32];
        let full = priority_slots(&cover, &seed);
        let mut sel = StreamingTopN::new(&seed, 100_000);
        for (d, i, k, b) in eligible(&cover) {
            sel.push(d, i, k, b);
        }
        assert_slots_eq(&sel.into_sorted(), &full);
    }

    #[test]
    fn streaming_topn_zero_capacity_is_empty() {
        let cover = build_cover(8);
        let mut sel = StreamingTopN::new(&[0u8; 32], 0);
        for (d, i, k, b) in eligible(&cover) {
            sel.push(d, i, k, b);
        }
        assert!(sel.into_sorted().is_empty());
    }

    /// The streaming verify's retained bits equal what `shadow_extract` reads
    /// off `priority_slots(..)[..n]`.
    #[test]
    fn shadow_bit_topn_matches_priority_slots_bits() {
        let cover = build_cover(300);
        let seed = [0xC3u8; 32];
        let full = priority_slots(&cover, &seed);
        let n = full.len() / 3;
        let bit = |d: EmbedDomain, i: usize| match d {
            EmbedDomain::CoeffSignBypass => cover.coeff_sign_bypass.bits[i],
            EmbedDomain::CoeffSuffixLsb => cover.coeff_suffix_lsb.bits[i],
            EmbedDomain::MvdSignBypass => cover.mvd_sign_bypass.bits[i],
            EmbedDomain::MvdSuffixLsb => cover.mvd_suffix_lsb.bits[i],
        };
        let want: Vec<u8> = full[..n].iter().map(|s| bit(s.domain, s.intra_index)).collect();
        let mut sel = ShadowBitTopN::new(&seed, n);
        for (d, i, k, b) in eligible(&cover).into_iter().rev() {
            sel.push(d, i, k, b, bit(d, i));
        }
        assert_eq!(sel.into_sorted_bits(), want);
    }
}
