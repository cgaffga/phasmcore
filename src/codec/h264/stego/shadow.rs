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
//! ## Position selection
//!
//! Shadow positions are selected across the **3 injectable bypass-bin
//! domains** by hash priority:
//!
//! - **CoeffSignBypass** — sign-bin overrides applied at the CABAC
//!   bypass-bin emit site (`wire_only`).
//! - **CoeffSuffixLsb** — magnitude-LSB ±1 flips at eligible
//!   coeffs (|coeff|≥16); cascade-absorbed for the rare boundary
//!   case where a flip drops a coefficient out of the suffix-LSB
//!   set.
//! - **MvdSignBypass** — sign-bin override at MVD bypass-emit
//!   (decoupled from `slot.value` so MC + median predictor see the
//!   encoder's natural MV — no cascade).
//!
//! **MvdSuffixLsb is NOT injectable** (magnitude-LSB flip changes
//! |MVD| → cascades through the median MV predictor). Pass 1 logs
//! MvdSuffixLsb positions in the cover but Pass 3 never overrides
//! them in the bitstream. Stamping shadow bits at MvdSuffixLsb
//! positions would put non-injectable slots in the shadow's RS frame
//! — the decoder reads the natural value, not the shadow bit, so
//! ~50% of those slots become noise → RS exhausts every parity tier.
//! That is why `priority_slots_all4` is restricted to the 3
//! injectable domains.
//!
//! Selection is by hash priority alone —
//! `ChaCha20(shadow_perm_seed, position_key)`. No locally-adaptive
//! bias (N=1 has no inter-shadow load to balance).
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
use crate::stego::armor::ecc;
use crate::stego::crypto::{self};
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload::{self, FileEntry, PayloadData};
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

use rand::{RngCore, SeedableRng};
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
    pub priority: u32,
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
            if let Some(result) = try_single_fdl(&all_rs_bytes, fdl, parity_len, passphrase) {
                return result;
            }
        }
    }

    Err(StegoError::FrameCorrupted)
}

// ─── 4-domain helpers (cascade-equipped shadow) ──────────────────
//
// These are the production shadow path. They span all 4 bypass-bin
// domains (CoeffSignBypass + CoeffSuffixLsb + MvdSignBypass +
// MvdSuffixLsb). The earlier experimental sign-only residual
// variants were retired in the 2026-06 cleanup.

/// Hash-priority sort across the bypass-bin domains
/// with optional per-domain cascade-safety masks.
///
/// **Without masks (`safe_csl = safe_msl = None`)**: includes the 3
/// always-injectable domains — CoeffSignBypass + CoeffSuffixLsb +
/// MvdSignBypass — and EXCLUDES MvdSuffixLsb entirely (the default).
///
/// **With masks**: MvdSuffixLsb included where `safe_msl[i] = true`;
/// CoeffSuffixLsb additionally filtered by `safe_csl[i]` when
/// supplied. `safe_msl` is the output of
/// `cascade_safety::derive_msl_safe_from_msb`. `safe_csl` filters
/// the |coeff|=16→15 boundary case (true iff |coeff|≥17).
///
/// Encoder + decoder MUST call with identical inputs to stay in
/// lockstep — both sides recompute the masks from their own meta
/// (cover_p1 on encode, walker meta on decode), and the safe-set
/// analysis is invariant under sign-flips and safe-set magnitude
/// flips by construction.
/// THE single source of truth for shadow-eligible cover positions + their
/// iteration order. Invokes `f(domain, intra_index, key)` for every eligible
/// position in the canonical domain order:
///   CoeffSignBypass (all — sign-only, no cascade),
///   CoeffSuffixLsb (cascade-aware `safe_csl` filter; `None` ⇒ all),
///   MvdSignBypass (all — sign-only bitstream-mod override),
///   MvdSuffixLsb (ONLY when `safe_msl` is `Some`, filtered by it).
/// `intra_index` is the position's index within its domain's `positions[]`
/// (the same `enumerate()` index everywhere). [`priority_slots`], the
/// streaming verify (`streaming_shadow_verify`), and the streaming decoder
/// (`StreamingDecodeSession::try_shadow_streaming`) all route through this so
/// their priority order can NEVER diverge — a divergence would silently break
/// shadow decode. (`sweep_b_emit`'s selection loop must mirror the same rule;
/// it stays inline only because its iteration is interleaved with STC cost
/// gating.)
pub(crate) fn for_each_eligible_position(
    cover: &DomainCover,
    safe_csl: Option<&[bool]>,
    safe_msl: Option<&[bool]>,
    mut f: impl FnMut(EmbedDomain, usize, &PositionKey),
) {
    for (intra_index, key) in cover.coeff_sign_bypass.positions.iter().enumerate() {
        f(EmbedDomain::CoeffSignBypass, intra_index, key);
    }
    for (intra_index, key) in cover.coeff_suffix_lsb.positions.iter().enumerate() {
        if let Some(mask) = safe_csl
            && (intra_index >= mask.len() || !mask[intra_index])
        {
            continue;
        }
        f(EmbedDomain::CoeffSuffixLsb, intra_index, key);
    }
    for (intra_index, key) in cover.mvd_sign_bypass.positions.iter().enumerate() {
        f(EmbedDomain::MvdSignBypass, intra_index, key);
    }
    if let Some(mask) = safe_msl {
        for (intra_index, key) in cover.mvd_suffix_lsb.positions.iter().enumerate() {
            if intra_index >= mask.len() || !mask[intra_index] {
                continue;
            }
            f(EmbedDomain::MvdSuffixLsb, intra_index, key);
        }
    }
}

pub(super) fn priority_slots(
    cover: &DomainCover,
    perm_seed: &[u8; 32],
    safe_csl: Option<&[bool]>,
    safe_msl: Option<&[bool]>,
) -> Vec<ShadowSlot> {
    let mut rng = ChaCha20Rng::from_seed(*perm_seed);
    let mut slots = Vec::new();
    for_each_eligible_position(cover, safe_csl, safe_msl, |domain, intra_index, key| {
        rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
        slots.push(ShadowSlot { domain, intra_index, priority: rng.next_u32() });
    });
    slots.sort_by_key(|s| s.priority);
    slots
}

/// One retained slot inside [`StreamingTopN`], ordered by the
/// `priority_slots` total order: `(priority, domain as u8,
/// intra_index)` ascending. `BinaryHeap` is a *max*-heap, so the
/// natural `Ord` below makes it evict the largest tuple — retaining
/// the smallest `capacity`, which is exactly the top-N
/// `priority_slots` keeps after its ascending sort.
///
/// The full tuple is unique across all cover positions (`domain` +
/// `intra_index` identify the position; only `priority` can collide),
/// so the order is *strict* — there are no genuine ties to resolve,
/// and the retained set is independent of push order.
// `dead_code` until WV.6.g.4 wires the streaming cover into the
// encode path; covered now by the equivalence tests below.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct TopNEntry {
    priority: u32,
    domain_rank: u8,
    intra_index: usize,
    domain: EmbedDomain,
}

impl Ord for TopNEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.priority, self.domain_rank, self.intra_index).cmp(&(
            other.priority,
            other.domain_rank,
            other.intra_index,
        ))
    }
}
impl PartialOrd for TopNEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Streaming top-N shadow position selector — the O(N)-memory
/// equivalent of `priority_slots(..)[..n]`, for the WV.6.g streaming
/// shadow path (no whole-clip `Vec<ShadowSlot>` materialised).
///
/// `priority_slots` builds the *entire* slot vector (one entry per
/// eligible cover position across all 4 domains), sorts it by
/// `(priority, domain, intra_index)`, and the caller keeps the first
/// `n`. At long-clip scale that vector is itself O(clip). This
/// selector keeps only the `n` lowest-priority slots seen so far in a
/// bounded max-heap: push every eligible `(domain, intra_index, key)`
/// in any order; once the heap exceeds `n`, the current maximum is
/// evicted. [`StreamingTopN::into_sorted`] then emits exactly the
/// same `n` slots, in the same order, that `priority_slots(..)[..n]`
/// would.
///
/// **Bit-identical to `priority_slots`** because the priority of a
/// position is `ChaCha20(perm_seed).seek(key.raw()*2).next_u32()` —
/// position-local and order-independent — and the total order's
/// tie-break `(domain as u8, intra_index)` exactly reproduces
/// `priority_slots`' stable-sort insertion order (CSB < CSL < MSB <
/// MSL, intra ascending). The caller is responsible for offering the
/// same eligible set: apply the `safe_csl` / `safe_msl` filters and
/// the "MvdSuffixLsb only when masked" rule that `priority_slots`
/// applies internally, and pass each position's `enumerate()` index
/// within its domain as `intra_index`.
// `dead_code` until WV.6.g.4 wires this into the streaming encode
// path (g.2 ships the primitive + its equivalence gate first).
#[allow(dead_code)]
pub(crate) struct StreamingTopN {
    rng: ChaCha20Rng,
    capacity: usize,
    heap: BinaryHeap<TopNEntry>,
}

#[allow(dead_code)] // see StreamingTopN: methods wired in WV.6.g.4
impl StreamingTopN {
    pub(crate) fn new(perm_seed: &[u8; 32], capacity: usize) -> Self {
        Self {
            rng: ChaCha20Rng::from_seed(*perm_seed),
            capacity,
            heap: BinaryHeap::with_capacity(capacity.saturating_add(1)),
        }
    }

    /// Offer one eligible cover position. `intra_index` is the
    /// position's index within its domain's `positions[]` vector (the
    /// same `enumerate()` index `priority_slots` uses); `key` is its
    /// [`PositionKey`]. Computes the position-local priority and
    /// retains the entry iff it ranks within the current top-`capacity`.
    pub(crate) fn push(&mut self, domain: EmbedDomain, intra_index: usize, key: PositionKey) {
        if self.capacity == 0 {
            return;
        }
        self.rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
        let entry = TopNEntry {
            priority: self.rng.next_u32(),
            domain_rank: domain as u8,
            intra_index,
            domain,
        };
        if self.heap.len() < self.capacity {
            self.heap.push(entry);
        } else if let Some(&max) = self.heap.peek() {
            // Heap full: evict the current max iff this entry ranks
            // strictly lower. The tuple is unique so `<` is a strict
            // total order — no tie can flip the retained set.
            if entry < max {
                self.heap.pop();
                self.heap.push(entry);
            }
        }
    }

    /// Drain into the final `Vec<ShadowSlot>`, ascending in the
    /// `priority_slots` total order — bit-identical to
    /// `priority_slots(..)[..min(capacity, eligible)]` over the same
    /// eligible set.
    pub(crate) fn into_sorted(self) -> Vec<ShadowSlot> {
        self.heap
            .into_sorted_vec() // ascending by `TopNEntry: Ord`
            .into_iter()
            .map(|e| ShadowSlot {
                domain: e.domain,
                intra_index: e.intra_index,
                priority: e.priority,
            })
            .collect()
    }
}

/// One retained slot inside [`ShadowBitTopN`] — same `(priority, domain_rank,
/// intra_index)` total order as [`TopNEntry`], plus the cover **bit** at that
/// position riding along as payload. Ord/Eq deliberately ignore `bit` (it is
/// not part of the position identity), so the heap eviction keeps the same
/// top-N positions [`StreamingTopN`] would.
// `dead_code` until WV.6.g.5c wires the streaming verify; covered now by the
// bit-equivalence gates below.
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct BitTopNEntry {
    priority: u32,
    domain_rank: u8,
    intra_index: usize,
    bit: u8,
}
impl BitTopNEntry {
    #[inline]
    fn order_key(&self) -> (u32, u8, usize) {
        (self.priority, self.domain_rank, self.intra_index)
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

/// WV.6.g.5 — bit-carrying twin of [`StreamingTopN`] for the streaming verify.
///
/// The per-tier verify must recover, from the **emitted** clip, the same
/// priority-ordered LSB stream the whole-clip `shadow_extract` reads — without
/// re-walking the whole clip. This selector is fed the emitted cover's eligible
/// positions one GOP at a time (each with its emitted bit), keeps the top
/// `capacity` by the identical `priority_slots` total order as `StreamingTopN`,
/// and [`into_sorted_bits`](Self::into_sorted_bits) drains the retained **bits**
/// in ascending priority order.
///
/// Result is bit-identical to `bits[priority_slots(emitted_cover, seed, None,
/// safe_msl)[..min(capacity, eligible)]]` — i.e. exactly the prefix of
/// `all_lsbs` that [`decode_shadow_from_priority_lsbs`] consumes. `capacity` is
/// the shadow's `n_total` at the largest parity tier (≥ every cascade tier's
/// frame length), so the retained bits cover the embedded frame at any tier.
#[allow(dead_code)] // see BitTopNEntry: wired into the verify in WV.6.g.5c
pub(crate) struct ShadowBitTopN {
    rng: ChaCha20Rng,
    capacity: usize,
    heap: BinaryHeap<BitTopNEntry>,
}

#[allow(dead_code)] // see BitTopNEntry: wired into the verify in WV.6.g.5c
impl ShadowBitTopN {
    pub(crate) fn new(perm_seed: &[u8; 32], capacity: usize) -> Self {
        Self {
            rng: ChaCha20Rng::from_seed(*perm_seed),
            capacity,
            heap: BinaryHeap::with_capacity(capacity.saturating_add(1)),
        }
    }

    /// Offer one eligible emitted position with its cover `bit`. Same
    /// position-local priority + top-`capacity` retention as
    /// [`StreamingTopN::push`]; the caller applies the identical per-domain
    /// eligibility filters `priority_slots` applies internally.
    pub(crate) fn push(
        &mut self,
        domain: EmbedDomain,
        intra_index: usize,
        key: PositionKey,
        bit: u8,
    ) {
        if self.capacity == 0 {
            return;
        }
        self.rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
        let entry = BitTopNEntry {
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

    /// Drain the retained bits in ascending `priority_slots` order — the
    /// `all_lsbs` prefix for [`decode_shadow_from_priority_lsbs`].
    pub(crate) fn into_sorted_bits(self) -> Vec<u8> {
        self.heap
            .into_sorted_vec() // ascending by `BitTopNEntry: Ord`
            .into_iter()
            .map(|e| e.bit)
            .collect()
    }
}

/// 4-domain shadow preparation with optional per-domain
/// cascade-safety masks. None = backwards-compat (the default — 3
/// always-injectable domains). Some(mask) = include
/// the corresponding suffix domain at safe positions only.
///
/// The masks must be aligned with `cover.coeff_suffix_lsb.positions`
/// and `cover.mvd_suffix_lsb.positions` respectively. Mask shorter
/// than the position vector → trailing positions treated as unsafe.
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

pub fn prepare_shadow(
    cover: &DomainCover,
    shadow_pass: &str,
    message: &str,
    files: &[FileEntry],
    parity_len: usize,
    safe_csl: Option<&[bool]>,
    safe_msl: Option<&[bool]>,
) -> Result<ShadowState, StegoError> {
    let (rs_bits, frame_data_len) =
        build_shadow_rs_frame(shadow_pass, message, files, parity_len)?;
    let n_total = rs_bits.len();

    let perm_seed = crypto::derive_shadow_structural_key(shadow_pass)?;
    let slots = priority_slots(cover, &perm_seed, safe_csl, safe_msl);

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

/// Cascade-safety-aware shadow preparation over a
/// single primary-emit cover. Threads `safe_csl` /
/// `safe_msl` through to `prepare_shadow` so the encoder
/// can include MvdSuffixLsb (and the brittle CoeffSuffixLsb
/// |coeff|=16 boundary case) at cascade-safe positions only.
pub fn prepare_shadow_over_emit_cover(
    primary_emit_cover: &DomainCover,
    shadow_pass: &str,
    message: &str,
    files: &[FileEntry],
    parity_len: usize,
    safe_csl: Option<&[bool]>,
    safe_msl: Option<&[bool]>,
) -> Result<ShadowState, StegoError> {
    prepare_shadow(
        primary_emit_cover,
        shadow_pass,
        message,
        files,
        parity_len,
        safe_csl,
        safe_msl,
    )
}

/// Cascade-safety-aware 4-domain shadow extract.
/// Decoder uses this when it has computed
/// `safe_csl` / `safe_msl` from the walked cover meta. Encoder +
/// decoder must use IDENTICAL mask inputs to land on the same
/// priority order.
pub fn shadow_extract(
    cover: &DomainCover,
    passphrase: &str,
    safe_csl: Option<&[bool]>,
    safe_msl: Option<&[bool]>,
) -> Result<PayloadData, StegoError> {
    if cover.total_len() == 0 {
        return Err(StegoError::FrameCorrupted);
    }

    let perm_seed = crypto::derive_shadow_structural_key(passphrase)?;
    let slots = priority_slots(cover, &perm_seed, safe_csl, safe_msl);

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
    // All Phase 6E-C1b "residual-only" + non-safe `_all4`/`_over_emit`
    // wrapper tests were retired alongside the dead functions they
    // exercised. The live `_safe` / `_all4`-mutator shadow paths are
    // covered by integration tests in the H.264 stego test suite.

    use super::*;
    use crate::codec::h264::stego::hook::{BinKind, SyntaxPath};

    // ─── WV.6.g.2 — StreamingTopN ≡ priority_slots(..)[..n] ───────
    //
    // The streaming shadow selector must pick BIT-IDENTICALLY to the
    // whole-clip `priority_slots(..)[..n]` it replaces — otherwise the
    // streaming path would place shadow bits at different cover
    // positions and existing stego files would stop decoding. These
    // tests prove the equivalence (and its push-order independence)
    // over a synthetic 4-domain cover with distinct keys per position.

    /// Distinct `PositionKey` per `(domain, i)` so every position gets
    /// its own `raw()` (hence its own ChaCha20 priority draw).
    fn pos_key(domain: EmbedDomain, i: usize) -> PositionKey {
        PositionKey::new(
            (i as u32) & 0xFFFF,
            (i as u32).wrapping_mul(7).wrapping_add(1),
            domain,
            SyntaxPath::Luma4x4 {
                block_idx: (i % 16) as u8,
                coeff_idx: (i % 16) as u8,
                kind: BinKind::Sign,
            },
        )
    }

    /// 4-domain cover with `k` positions in each domain.
    fn build_cover(k: usize) -> DomainCover {
        let mut c = DomainCover::default();
        for i in 0..k {
            c.coeff_sign_bypass.push(0, pos_key(EmbedDomain::CoeffSignBypass, i));
            c.coeff_suffix_lsb.push(0, pos_key(EmbedDomain::CoeffSuffixLsb, i));
            c.mvd_sign_bypass.push(0, pos_key(EmbedDomain::MvdSignBypass, i));
            c.mvd_suffix_lsb.push(0, pos_key(EmbedDomain::MvdSuffixLsb, i));
        }
        c
    }

    fn assert_slots_eq(got: &[ShadowSlot], want: &[ShadowSlot]) {
        assert_eq!(got.len(), want.len(), "slot count mismatch");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g.domain, w.domain, "domain mismatch at rank {i}");
            assert_eq!(g.intra_index, w.intra_index, "intra_index mismatch at rank {i}");
            assert_eq!(g.priority, w.priority, "priority mismatch at rank {i}");
        }
    }

    #[test]
    fn streaming_topn_matches_priority_slots_no_mask() {
        let cover = build_cover(64);
        let seed = [0x5Au8; 32];
        // No masks ⇒ CSB + CSL(all) + MSB, MvdSuffixLsb EXCLUDED.
        let full = priority_slots(&cover, &seed, None, None);
        let n = 40usize;
        let want = &full[..n];

        let mut sel = StreamingTopN::new(&seed, n);
        // Offer the SAME eligible set in an order UNRELATED to
        // priority_slots' insertion order (intra descending,
        // domains interleaved) to prove order-independence. No MSL.
        for i in (0..64).rev() {
            sel.push(EmbedDomain::MvdSignBypass, i, cover.mvd_sign_bypass.positions[i]);
            sel.push(EmbedDomain::CoeffSignBypass, i, cover.coeff_sign_bypass.positions[i]);
            sel.push(EmbedDomain::CoeffSuffixLsb, i, cover.coeff_suffix_lsb.positions[i]);
        }
        assert_slots_eq(&sel.into_sorted(), want);
    }

    #[test]
    fn streaming_topn_matches_priority_slots_with_masks() {
        // The d.6 cascade-safe path: CSL filtered by |coeff|≥17 and
        // MvdSuffixLsb included only where derive_msl_safe says so.
        let cover = build_cover(50);
        let seed = [0xC3u8; 32];
        let safe_csl: Vec<bool> = (0..50).map(|i| i % 3 != 0).collect();
        let safe_msl: Vec<bool> = (0..50).map(|i| i % 2 == 1).collect();
        let full = priority_slots(&cover, &seed, Some(&safe_csl), Some(&safe_msl));
        let n = full.len() / 2;
        let want = &full[..n];

        let mut sel = StreamingTopN::new(&seed, n);
        // Caller applies the same per-domain filters priority_slots
        // applies internally, and offers each position's enumerate()
        // index as intra_index.
        for i in 0..50 {
            sel.push(EmbedDomain::CoeffSignBypass, i, cover.coeff_sign_bypass.positions[i]);
            if safe_csl[i] {
                sel.push(EmbedDomain::CoeffSuffixLsb, i, cover.coeff_suffix_lsb.positions[i]);
            }
            sel.push(EmbedDomain::MvdSignBypass, i, cover.mvd_sign_bypass.positions[i]);
            if safe_msl[i] {
                sel.push(EmbedDomain::MvdSuffixLsb, i, cover.mvd_suffix_lsb.positions[i]);
            }
        }
        assert_slots_eq(&sel.into_sorted(), want);
    }

    #[test]
    fn streaming_topn_capacity_exceeds_eligible_returns_all_sorted() {
        let cover = build_cover(8);
        let seed = [0x11u8; 32];
        let full = priority_slots(&cover, &seed, None, None); // 24 slots
        let n = 1000usize; // >> eligible
        let want = &full[..full.len().min(n)];

        let mut sel = StreamingTopN::new(&seed, n);
        for i in 0..8 {
            sel.push(EmbedDomain::CoeffSignBypass, i, cover.coeff_sign_bypass.positions[i]);
            sel.push(EmbedDomain::CoeffSuffixLsb, i, cover.coeff_suffix_lsb.positions[i]);
            sel.push(EmbedDomain::MvdSignBypass, i, cover.mvd_sign_bypass.positions[i]);
        }
        assert_slots_eq(&sel.into_sorted(), want);
    }

    #[test]
    fn streaming_topn_zero_capacity_is_empty() {
        let cover = build_cover(4);
        let seed = [0u8; 32];
        let mut sel = StreamingTopN::new(&seed, 0);
        for i in 0..4 {
            sel.push(EmbedDomain::CoeffSignBypass, i, cover.coeff_sign_bypass.positions[i]);
        }
        assert!(sel.into_sorted().is_empty());
    }

    // ─── WV.6.g.5b — ShadowBitTopN bits ≡ priority_slots(..)[..n] bits ──
    //
    // The streaming verify reads the EMITTED clip's priority-ordered LSBs via
    // ShadowBitTopN (per-GOP, bounded) instead of walking the whole clip. These
    // prove the retained BITS equal what `shadow_extract` would read off
    // `priority_slots(cover, seed, None, safe_msl)[..n]` — the `all_lsbs` prefix
    // fed to `decode_shadow_from_priority_lsbs`.

    /// 4-domain cover with a distinct per-(domain,i) bit pattern, so a
    /// mis-selected position reads a different bit and the test fails.
    fn build_cover_with_bits(k: usize) -> DomainCover {
        let mut c = DomainCover::default();
        for i in 0..k {
            c.coeff_sign_bypass.push((i & 1) as u8, pos_key(EmbedDomain::CoeffSignBypass, i));
            c.coeff_suffix_lsb.push(((i ^ 7) & 1) as u8, pos_key(EmbedDomain::CoeffSuffixLsb, i));
            c.mvd_sign_bypass.push(((i ^ 13) & 1) as u8, pos_key(EmbedDomain::MvdSignBypass, i));
            c.mvd_suffix_lsb.push(((i ^ 21) & 1) as u8, pos_key(EmbedDomain::MvdSuffixLsb, i));
        }
        c
    }

    /// The reference `all_lsbs` prefix: bits at `priority_slots(cover, seed,
    /// None, safe_msl)[..n]` (safe_csl=None, matching the verify path).
    fn ref_bits(cover: &DomainCover, seed: &[u8; 32], safe_msl: Option<&[bool]>, n: usize) -> Vec<u8> {
        priority_slots(cover, seed, None, safe_msl)
            .iter()
            .take(n)
            .map(|s| match s.domain {
                EmbedDomain::CoeffSignBypass => cover.coeff_sign_bypass.bits[s.intra_index],
                EmbedDomain::CoeffSuffixLsb => cover.coeff_suffix_lsb.bits[s.intra_index],
                EmbedDomain::MvdSignBypass => cover.mvd_sign_bypass.bits[s.intra_index],
                EmbedDomain::MvdSuffixLsb => cover.mvd_suffix_lsb.bits[s.intra_index],
            })
            .collect()
    }

    #[test]
    fn shadow_bit_topn_matches_priority_slots_bits_no_mask() {
        let cover = build_cover_with_bits(64);
        let seed = [0x5Au8; 32];
        let n = 40usize;
        let want = ref_bits(&cover, &seed, None, n);

        let mut sel = ShadowBitTopN::new(&seed, n);
        // Push order UNRELATED to priority order (intra descending) to prove
        // order-independence; MvdSuffixLsb excluded (no mask), matching ref.
        for i in (0..64).rev() {
            sel.push(EmbedDomain::MvdSignBypass, i, cover.mvd_sign_bypass.positions[i], cover.mvd_sign_bypass.bits[i]);
            sel.push(EmbedDomain::CoeffSignBypass, i, cover.coeff_sign_bypass.positions[i], cover.coeff_sign_bypass.bits[i]);
            sel.push(EmbedDomain::CoeffSuffixLsb, i, cover.coeff_suffix_lsb.positions[i], cover.coeff_suffix_lsb.bits[i]);
        }
        assert_eq!(sel.into_sorted_bits(), want);
    }

    #[test]
    fn shadow_bit_topn_matches_priority_slots_bits_with_msl_mask() {
        // The d.6 cascade-safe path: MvdSuffixLsb included only where the mask
        // says safe; safe_csl=None (CSL all-eligible), as the verify calls it.
        let cover = build_cover_with_bits(50);
        let seed = [0xC3u8; 32];
        let safe_msl: Vec<bool> = (0..50).map(|i| i % 2 == 1).collect();
        let full = priority_slots(&cover, &seed, None, Some(&safe_msl));
        let n = full.len() / 2;
        let want = ref_bits(&cover, &seed, Some(&safe_msl), n);

        let mut sel = ShadowBitTopN::new(&seed, n);
        for i in 0..50 {
            sel.push(EmbedDomain::CoeffSignBypass, i, cover.coeff_sign_bypass.positions[i], cover.coeff_sign_bypass.bits[i]);
            sel.push(EmbedDomain::CoeffSuffixLsb, i, cover.coeff_suffix_lsb.positions[i], cover.coeff_suffix_lsb.bits[i]);
            sel.push(EmbedDomain::MvdSignBypass, i, cover.mvd_sign_bypass.positions[i], cover.mvd_sign_bypass.bits[i]);
            if safe_msl[i] {
                sel.push(EmbedDomain::MvdSuffixLsb, i, cover.mvd_suffix_lsb.positions[i], cover.mvd_suffix_lsb.bits[i]);
            }
        }
        assert_eq!(sel.into_sorted_bits(), want);
    }
}
