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
use crate::stego::crypto::{self, NONCE_LEN, SALT_LEN};
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
pub(super) fn priority_slots(
    cover: &DomainCover,
    perm_seed: &[u8; 32],
    safe_csl: Option<&[bool]>,
    safe_msl: Option<&[bool]>,
) -> Vec<ShadowSlot> {
    let mut rng = ChaCha20Rng::from_seed(*perm_seed);
    let msl_count = safe_msl
        .map(|m| m.iter().filter(|&&b| b).count())
        .unwrap_or(0);
    let csl_count = match safe_csl {
        Some(m) => m.iter().filter(|&&b| b).count(),
        None => cover.coeff_suffix_lsb.len(),
    };
    let mut slots = Vec::with_capacity(
        cover.coeff_sign_bypass.len() + csl_count + cover.mvd_sign_bypass.len() + msl_count,
    );

    // CoeffSignBypass — always injectable (sign-only, no cascade).
    for (intra_index, key) in cover.coeff_sign_bypass.positions.iter().enumerate() {
        rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
        slots.push(ShadowSlot {
            domain: EmbedDomain::CoeffSignBypass,
            intra_index,
            priority: rng.next_u32(),
        });
    }

    // CoeffSuffixLsb — optional |coeff|≥17 filter (cascade-aware
    // when `safe_csl` supplied; otherwise include all).
    for (intra_index, key) in cover.coeff_suffix_lsb.positions.iter().enumerate() {
        if let Some(mask) = safe_csl
            && (intra_index >= mask.len() || !mask[intra_index])
        {
            continue;
        }
        rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
        slots.push(ShadowSlot {
            domain: EmbedDomain::CoeffSuffixLsb,
            intra_index,
            priority: rng.next_u32(),
        });
    }

    // MvdSignBypass — always injectable (sign-only bitstream-mod
    // override; doesn't mutate slot.value).
    for (intra_index, key) in cover.mvd_sign_bypass.positions.iter().enumerate() {
        rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
        slots.push(ShadowSlot {
            domain: EmbedDomain::MvdSignBypass,
            intra_index,
            priority: rng.next_u32(),
        });
    }

    // MvdSuffixLsb — ONLY when safe_msl supplied. Default-without-
    // mask: skip the domain entirely.
    if let Some(mask) = safe_msl {
        for (intra_index, key) in cover.mvd_suffix_lsb.positions.iter().enumerate() {
            if intra_index >= mask.len() || !mask[intra_index] {
                continue;
            }
            rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
            slots.push(ShadowSlot {
                domain: EmbedDomain::MvdSuffixLsb,
                intra_index,
                priority: rng.next_u32(),
            });
        }
    }

    slots.sort_by_key(|s| s.priority);
    slots
}

/// 4-domain shadow preparation with optional per-domain
/// cascade-safety masks. None = backwards-compat (the default — 3
/// always-injectable domains). Some(mask) = include
/// the corresponding suffix domain at safe positions only.
///
/// The masks must be aligned with `cover.coeff_suffix_lsb.positions`
/// and `cover.mvd_suffix_lsb.positions` respectively. Mask shorter
/// than the position vector → trailing positions treated as unsafe.
pub fn prepare_shadow(
    cover: &DomainCover,
    shadow_pass: &str,
    message: &str,
    files: &[FileEntry],
    parity_len: usize,
    safe_csl: Option<&[bool]>,
    safe_msl: Option<&[bool]>,
) -> Result<ShadowState, StegoError> {
    let payload_bytes = payload::encode_payload(message, files)?;
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, shadow_pass)?;
    let frame_bytes = build_shadow_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
    let frame_data_len = frame_bytes.len();

    let rs_bytes = ecc::rs_encode_blocks_with_parity(&frame_bytes, parity_len);
    let rs_bits = frame::bytes_to_bits(&rs_bytes);
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

    // Bit-pack the full LSB stream ONCE (not per parity tier
    // nor per fdl candidate); try_single_fdl + peek slice into this buffer.
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

#[allow(dead_code)]
fn _unused_imports_guard() -> usize {
    NONCE_LEN + SALT_LEN
}

#[cfg(test)]
mod tests {
    // All Phase 6E-C1b "residual-only" + non-safe `_all4`/`_over_emit`
    // wrapper tests were retired alongside the dead functions they
    // exercised. The live `_safe` / `_all4`-mutator shadow paths are
    // covered by integration tests in the H.264 stego test suite.
}
