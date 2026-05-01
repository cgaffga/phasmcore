// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 6E-C1b — H.264 video shadow messages (1 shadow, fixed parity).
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
//! ## Position selection (current — §6E-C2 polish + §6E-A5(c.x))
//!
//! Shadow positions are selected across the **3 injectable bypass-bin
//! domains** by hash priority:
//!
//! - **CoeffSignBypass** — sign-bin overrides applied at residual
//!   block emission (`apply_coeff_sign_overrides`).
//! - **CoeffSuffixLsb** — magnitude-LSB ±1 flips at eligible
//!   coeffs (|coeff|≥16); cascade-absorbed for the rare boundary
//!   case where a flip drops a coefficient out of the suffix-LSB
//!   set.
//! - **MvdSignBypass** — sign-bin override at MVD bypass-emit
//!   (post-§6F.2(k).1, decoupled from `slot.value` so MC + median
//!   predictor see the encoder's natural MV — no cascade).
//!
//! **MvdSuffixLsb is NOT injectable** post-§6F.2(k).2 (magnitude-
//! LSB flip changes |MVD| → cascades through the median MV
//! predictor). Pass 1 logs MvdSuffixLsb positions in the cover but
//! Pass 3 never overrides them in the bitstream. Stamping shadow
//! bits at MvdSuffixLsb positions would put non-injectable slots
//! in the shadow's RS frame — the decoder reads the natural value,
//! not the shadow bit, so ~50% of those slots become noise → RS
//! exhausts every parity tier (the #107 root cause at 1080p before
//! `priority_slots_all4` was restricted to the 3 injectable
//! domains).
//!
//! Selection is by hash priority alone —
//! `ChaCha20(shadow_perm_seed, position_key)`. No locally-adaptive
//! bias (N=1 has no inter-shadow load to balance; bias is §6E-C2).
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
//! 3. **Inter-shadow collisions** absorb into RS parity (§6E-C2).
//!
//! ## Decoder (this commit)
//!
//! Walks the entire Annex-B stream, brute-forces the 6 parity tiers
//! over hash-priority-selected positions across all 4 domains.
//! Each tier: take top-N positions globally (no biasing), extract
//! LSBs, RS-decode + first-block peek for `fdl`, AES-GCM-SIV
//! validate. First success wins. The streaming early-exit during
//! walk variant is a §6E-C1b polish item (deferred — `walk_annex_b_for_cover_with_options`
//! already accumulates whole-stream cover transparently).

use super::{DomainCover, EmbedDomain, PositionKey};
use crate::stego::armor::ecc;
use crate::stego::crypto::{self, NONCE_LEN, SALT_LEN};
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload::{self, FileEntry, PayloadData};
// §6E-C2 polish — video shadow uses the WIDE frame format (u32
// plaintext_len, 48-byte overhead) instead of the standard image
// format (u16, 46-byte overhead). Video covers can support multi-MB
// shadows (e.g., file attachments); the u16 cap was too tight.
// Image-side image stego stays on the standard format to preserve
// compatibility with released apps.
use crate::stego::shadow_layer::{
    build_shadow_frame_wide as build_shadow_frame,
    compute_max_shadow_fdl,
    parse_shadow_frame_wide as parse_shadow_frame,
    MAX_SHADOW_FRAME_BYTES_WIDE as MAX_SHADOW_FRAME_BYTES,
    SHADOW_FRAME_OVERHEAD_WIDE as SHADOW_FRAME_OVERHEAD,
    SHADOW_PARITY_TIERS,
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

/// Build the CoeffSignBypass-domain hash-priority sort key list.
/// Returns one `ShadowSlot` per sign-bypass cover position with
/// priority computed from `ChaCha20(perm_seed).next_u32()` keyed by
/// the position's `PositionKey.raw()`. Suffix-LSB and MVD positions
/// are skipped — see module-level docs for the §6E-C1b scope.
fn priority_slots(cover: &DomainCover, perm_seed: &[u8; 32]) -> Vec<ShadowSlot> {
    let mut rng = ChaCha20Rng::from_seed(*perm_seed);
    let mut slots = Vec::with_capacity(cover.coeff_sign_bypass.len());

    for (intra_index, key) in cover.coeff_sign_bypass.positions.iter().enumerate() {
        rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
        let priority = rng.next_u32();
        slots.push(ShadowSlot {
            domain: EmbedDomain::CoeffSignBypass,
            intra_index,
            priority,
        });
    }

    slots.sort_by_key(|s| s.priority);
    slots
}

/// Prepare a shadow layer for embedding.
///
/// Builds the payload, encrypts, frames, RS-encodes with `parity_len`,
/// then selects the top-`n_total` positions across all 4 cover
/// domains by hash priority.
pub fn prepare_shadow(
    cover: &DomainCover,
    shadow_pass: &str,
    message: &str,
    files: &[FileEntry],
    parity_len: usize,
) -> Result<ShadowState, StegoError> {
    let payload_bytes = payload::encode_payload(message, files)?;
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, shadow_pass)?;
    let frame_bytes = build_shadow_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
    let frame_data_len = frame_bytes.len();

    let rs_bytes = ecc::rs_encode_blocks_with_parity(&frame_bytes, parity_len);
    let rs_bits = frame::bytes_to_bits(&rs_bytes);
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

/// Inject shadow LSBs into the per-domain residual cover bit
/// arrays. Run BEFORE primary STC plans so that primary's Viterbi
/// sees shadow bits as if they were natural cover bits — combined
/// with [`overlay_infinity_costs_residual`] this guarantees primary
/// STC keeps the shadow bits at shadow positions, preserving both
/// primary's syndrome and shadow's RS-encoded payload.
pub fn embed_shadow_lsb_residual(
    coeff_sign_bypass_bits: &mut [u8],
    coeff_suffix_lsb_bits: &mut [u8],
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
            EmbedDomain::MvdSignBypass | EmbedDomain::MvdSuffixLsb => {
                unreachable!("priority_slots restricted to residual domains")
            }
        }
    }
}

/// For each shadow position in the **residual** domains, set the
/// matching slot in the per-domain residual cost vector to
/// `f32::INFINITY` so primary STC's Viterbi avoids flipping it.
/// MVD-domain shadow positions are NOT given ∞-cost in this
/// commit's flow — Pass 2A (primary MVD STC) runs before shadow
/// position selection because shadow uses Pass 1B's final residual
/// cover. Sparse MVD-domain primary flips overlapping shadow
/// positions are absorbed by RS parity (parity ≥ 4 covers many
/// bytes of error per 255-byte block at single-shadow scale).
pub fn overlay_infinity_costs_residual(
    coeff_sign_bypass_cost: &mut [f32],
    coeff_suffix_lsb_cost: &mut [f32],
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
            EmbedDomain::MvdSignBypass | EmbedDomain::MvdSuffixLsb => {
                // No ∞-cost overlay on MVD costs — RS parity absorbs
                // the rare primary/shadow MVD position overlap (see
                // doc comment above).
            }
        }
    }
}

/// Defensive override of the residual `DomainPlan` bits at shadow
/// positions with the shadow's RS-encoded LSBs. With shadow bits
/// already injected into the cover (via
/// [`embed_shadow_lsb_residual`]) AND ∞-cost on shadow positions
/// (via [`overlay_infinity_costs_residual`]), primary STC will
/// already keep the shadow bits at shadow positions. This function
/// is a defensive stamp guarding against any future plan-layer
/// drift between cover-bit injection and STC plan output.
pub fn apply_shadow_to_plan_residual(
    coeff_sign_bypass: &mut [u8],
    coeff_suffix_lsb: &mut [u8],
    state: &ShadowState,
) {
    for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
        let bit = state.bits[i];
        match slot.domain {
            EmbedDomain::CoeffSignBypass => coeff_sign_bypass[slot.intra_index] = bit,
            EmbedDomain::CoeffSuffixLsb => coeff_suffix_lsb[slot.intra_index] = bit,
            EmbedDomain::MvdSignBypass | EmbedDomain::MvdSuffixLsb => {
                unreachable!("priority_slots restricted to residual domains")
            }
        }
    }
}

/// Try one (lsbs, fdl, parity, passphrase) candidate. Returns
/// `Some(Ok(payload))` on success, `None` on any failure (RS,
/// frame parse, or AES-GCM-SIV authentication).
fn try_single_fdl(
    lsbs: &[u8],
    fdl: usize,
    parity_len: usize,
    passphrase: &str,
) -> Option<Result<PayloadData, StegoError>> {
    let rs_encoded_len = ecc::rs_encoded_len_with_parity(fdl, parity_len);
    let rs_bits_needed = rs_encoded_len * 8;
    if rs_bits_needed > lsbs.len() {
        return None;
    }
    let rs_bytes = frame::bits_to_bytes(&lsbs[..rs_bits_needed]);
    let decoded = match ecc::rs_decode_blocks_with_parity(&rs_bytes, fdl, parity_len) {
        Ok((data, _)) => data,
        Err(_) => return None,
    };
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
fn peek_fdl_from_first_block(
    lsbs: &[u8],
    parity_len: usize,
    max_fdl: usize,
) -> Option<usize> {
    let k = 255usize.saturating_sub(parity_len);
    if k < 2 || lsbs.len() < 255 * 8 {
        return None;
    }
    let first_block_bytes = frame::bits_to_bytes(&lsbs[..255 * 8]);
    let (data, _) =
        ecc::rs_decode_blocks_with_parity(&first_block_bytes, k, parity_len).ok()?;
    if data.len() < 2 {
        return None;
    }
    let plaintext_len = u16::from_be_bytes([data[0], data[1]]) as usize;
    let fdl = SHADOW_FRAME_OVERHEAD + plaintext_len;
    if fdl >= k && fdl <= max_fdl {
        Some(fdl)
    } else {
        None
    }
}

// ─── 4-domain helpers (Phase 6E-C1b-v2 cascade-equipped shadow) ──
//
// The functions below mirror the residual-only helpers above but
// span all 4 bypass-bin domains (CoeffSignBypass + CoeffSuffixLsb +
// MvdSignBypass + MvdSuffixLsb). §6E-C1b-v2 uses these; §6E-C1b
// (experimental sign-only) keeps using the residual-only variants.

/// §6E-A5(d).3 — hash-priority sort across the bypass-bin domains
/// with optional per-domain cascade-safety masks.
///
/// **Without masks (`safe_csl = safe_msl = None`)**: includes the 3
/// always-injectable domains — CoeffSignBypass + CoeffSuffixLsb +
/// MvdSignBypass — and EXCLUDES MvdSuffixLsb entirely (the post-#107
/// default; `priority_slots_all4` wrapper below).
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
/// flips by §6F.2(j) construction.
pub(super) fn priority_slots_all4_safe(
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

    // MvdSignBypass — always injectable (post-§6F.2(k).1 sign-only
    // bitstream-mod override; doesn't mutate slot.value).
    for (intra_index, key) in cover.mvd_sign_bypass.positions.iter().enumerate() {
        rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
        slots.push(ShadowSlot {
            domain: EmbedDomain::MvdSignBypass,
            intra_index,
            priority: rng.next_u32(),
        });
    }

    // MvdSuffixLsb — ONLY when safe_msl supplied. Default-without-
    // mask: skip the domain entirely (post-#107 behavior).
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

/// Backwards-compat wrapper around `priority_slots_all4_safe` with
/// no safety masks. Reproduces the post-#107 3-domain behavior
/// (MvdSuffixLsb fully excluded). Used by callers that haven't
/// computed cascade-safety masks (today's production default).
fn priority_slots_all4(
    cover: &DomainCover,
    perm_seed: &[u8; 32],
) -> Vec<ShadowSlot> {
    priority_slots_all4_safe(cover, perm_seed, None, None)
}

/// 4-domain shadow preparation. Same RS-encode + AES-GCM-SIV +
/// frame-format + position-selection structure as `prepare_shadow`,
/// but the position selection spans all 4 bypass-bin domains via
/// `priority_slots_all4`.
pub fn prepare_shadow_all4(
    cover: &DomainCover,
    shadow_pass: &str,
    message: &str,
    files: &[FileEntry],
    parity_len: usize,
) -> Result<ShadowState, StegoError> {
    prepare_shadow_all4_safe(cover, shadow_pass, message, files, parity_len, None, None)
}

/// §6E-A5(d).3 — `prepare_shadow_all4` variant accepting optional
/// per-domain cascade-safety masks. None = backwards-compat (post-
/// #107 default — 3 always-injectable domains). Some(mask) = include
/// the corresponding suffix domain at safe positions only.
///
/// The masks must be aligned with `cover.coeff_suffix_lsb.positions`
/// and `cover.mvd_suffix_lsb.positions` respectively. Mask shorter
/// than the position vector → trailing positions treated as unsafe.
pub fn prepare_shadow_all4_safe(
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
    let slots = priority_slots_all4_safe(cover, &perm_seed, safe_csl, safe_msl);

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
/// combined with [`overlay_infinity_costs_all4`] the primary plan
/// keeps shadow bits at shadow positions.
pub fn embed_shadow_lsb_all4(
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
/// shadow bits injected via [`embed_shadow_lsb_all4`].
pub fn overlay_infinity_costs_all4(
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
pub fn apply_shadow_to_plan_all4(
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

// ─── §6E-C2 polish — single-cover over primary-emit cover ────────
//
// The polish architecture (locked 2026-04-28) selects shadow
// positions over the PRIMARY-EMIT COVER — the cover obtained by
// running a primary-only Pass 3 emit and walking the bytes via
// the §6E-C0 streaming walker (see `provisional_emit::pass3_emit_provisional`).
//
// This matches the cover the DECODER sees: decoder priority-sorts
// shadow positions over the FINAL EMIT cover, which differs from
// the primary-emit cover only at shadow-override positions (where
// shadow LSBs flip primary's bit values, but don't change set
// membership beyond rare boundary cases that cascade absorbs).
//
// The functions below are thin wrappers around `priority_slots_all4`
// and `prepare_shadow_all4` — the underlying logic is unchanged.
// New names document the architectural intent. Wiring into the
// encoder cascade loop happened in commit 4 of the polish sequence
// (dual-iteration cascade with PositionKey translation).

/// §6E-C2 polish — priority sort over a single primary-emit cover.
///
/// Takes the cover produced by walking the provisional Pass-3 emit
/// (`provisional_emit::pass3_emit_provisional`) and returns shadow
/// position slots sorted by `ChaCha20(perm_seed)` priority across
/// all 4 bypass-bin domains.
///
/// The sort is identical to `priority_slots_all4`; this is a
/// public re-export with a name that documents the §6E-C2 polish
/// architecture. Cover argument MUST be a primary-emit cover (the
/// cover the decoder will see, modulo shadow overrides) — passing
/// `cover_p1` or `cover_p1b_residual` here defeats the polish fix.
pub fn priority_slots_4domain_over_cover(
    primary_emit_cover: &DomainCover,
    perm_seed: &[u8; 32],
) -> Vec<ShadowSlot> {
    priority_slots_all4(primary_emit_cover, perm_seed)
}

/// §6E-C2 polish — prepare shadow state over a single primary-emit
/// cover.
///
/// Use after running `provisional_emit::pass3_emit_provisional` to
/// obtain `primary_emit_cover`. Samples all 4 bypass-bin domains
/// from the cover the decoder will see (modulo small drift from
/// final emit). The encoder applies shadow overrides via
/// PositionKey lookup against `cover_p1.mvd_*` / `cover_p1b_residual.coeff_*`
/// for the ∞-cost mask + injection step.
///
/// `ShadowState.positions[i].intra_index` indexes into
/// `primary_emit_cover.{domain}.{positions, bits, costs}`. The
/// encoder must apply shadow overrides via PositionKey lookup
/// against the FINAL emit cover (not via direct intra_index access
/// to `cover_p1` arrays) — see commit 4 of the polish sequence
/// (override-on-top final emit).
pub fn prepare_shadow_over_emit_cover(
    primary_emit_cover: &DomainCover,
    shadow_pass: &str,
    message: &str,
    files: &[FileEntry],
    parity_len: usize,
) -> Result<ShadowState, StegoError> {
    prepare_shadow_all4(
        primary_emit_cover,
        shadow_pass,
        message,
        files,
        parity_len,
    )
}

/// §6E-A5(d).3 — cascade-safety-aware variant of
/// [`prepare_shadow_over_emit_cover`]. Threads `safe_csl` /
/// `safe_msl` through to `prepare_shadow_all4_safe` so the encoder
/// can include MvdSuffixLsb (and the brittle CoeffSuffixLsb
/// |coeff|=16 boundary case) at cascade-safe positions only.
pub fn prepare_shadow_over_emit_cover_safe(
    primary_emit_cover: &DomainCover,
    shadow_pass: &str,
    message: &str,
    files: &[FileEntry],
    parity_len: usize,
    safe_csl: Option<&[bool]>,
    safe_msl: Option<&[bool]>,
) -> Result<ShadowState, StegoError> {
    prepare_shadow_all4_safe(
        primary_emit_cover,
        shadow_pass,
        message,
        files,
        parity_len,
        safe_csl,
        safe_msl,
    )
}

/// 4-domain shadow extract — same brute-force-parity-tier algorithm
/// as `shadow_extract`, but priority sort spans all 4 bypass-bin
/// domains. Used by §6E-C1b-v2 cascade verification (encoder
/// simulates decoder on emitted bytes).
pub fn shadow_extract_all4(
    cover: &DomainCover,
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    shadow_extract_all4_safe(cover, passphrase, None, None)
}

/// §6E-A5(d).3 — cascade-safety-aware variant of
/// [`shadow_extract_all4`]. Decoder uses this when it has computed
/// `safe_csl` / `safe_msl` from the walked cover meta. Encoder +
/// decoder must use IDENTICAL mask inputs to land on the same
/// priority order.
pub fn shadow_extract_all4_safe(
    cover: &DomainCover,
    passphrase: &str,
    safe_csl: Option<&[bool]>,
    safe_msl: Option<&[bool]>,
) -> Result<PayloadData, StegoError> {
    if cover.total_len() == 0 {
        return Err(StegoError::FrameCorrupted);
    }

    let perm_seed = crypto::derive_shadow_structural_key(passphrase)?;
    let slots = priority_slots_all4_safe(cover, &perm_seed, safe_csl, safe_msl);

    let all_lsbs: Vec<u8> = slots
        .iter()
        .map(|slot| match slot.domain {
            EmbedDomain::CoeffSignBypass => cover.coeff_sign_bypass.bits[slot.intra_index],
            EmbedDomain::CoeffSuffixLsb => cover.coeff_suffix_lsb.bits[slot.intra_index],
            EmbedDomain::MvdSignBypass => cover.mvd_sign_bypass.bits[slot.intra_index],
            EmbedDomain::MvdSuffixLsb => cover.mvd_suffix_lsb.bits[slot.intra_index],
        })
        .collect();

    for &parity_len in &SHADOW_PARITY_TIERS {
        let k = 255usize.saturating_sub(parity_len);
        if k < 2 {
            continue;
        }
        let max_rs_bytes = all_lsbs.len() / 8;
        let max_fdl = compute_max_shadow_fdl(max_rs_bytes, parity_len)
            .min(MAX_SHADOW_FRAME_BYTES);
        if SHADOW_FRAME_OVERHEAD > max_fdl {
            continue;
        }

        if let Some(fdl) = peek_fdl_from_first_block(&all_lsbs, parity_len, max_fdl)
            && let Some(result) = try_single_fdl(&all_lsbs, fdl, parity_len, passphrase)
        {
            return result;
        }

        let small_max = (k - 1).min(max_fdl);
        if SHADOW_FRAME_OVERHEAD > small_max {
            continue;
        }
        for fdl in SHADOW_FRAME_OVERHEAD..=small_max {
            if let Some(result) = try_single_fdl(&all_lsbs, fdl, parity_len, passphrase) {
                return result;
            }
        }
    }

    Err(StegoError::FrameCorrupted)
}

// ─── Residual-only helpers (§6E-C1b experimental sign-only) ──────

/// Brute-force shadow extract from a walked 4-domain `DomainCover`.
///
/// For each parity tier in `[4, 8, 16, 32, 64, 128]`:
/// 1. Compute hash-priority slot ordering for `passphrase`.
/// 2. Take top-X slots where X is bounded by the cover's total
///    length and `MAX_SHADOW_FRAME_BYTES`.
/// 3. Extract LSBs at those slots from `cover.bits`.
/// 4. First-block-peek to derive `fdl`, then `try_single_fdl`. If
///    that fails, scan the small-`fdl` partial-block range.
/// 5. AES-GCM-SIV authentication validates correctness; first
///    successful tier returns the payload.
///
/// Returns `Err(StegoError::FrameCorrupted)` if no tier validates.
pub fn shadow_extract(
    cover: &DomainCover,
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    if cover.total_len() == 0 {
        return Err(StegoError::FrameCorrupted);
    }

    let perm_seed = crypto::derive_shadow_structural_key(passphrase)?;
    let slots = priority_slots(cover, &perm_seed);

    // Pre-extract LSBs in priority order — reused across parity tiers.
    // Residual-only — see module-level docs for §6E-C1b scope.
    let all_lsbs: Vec<u8> = slots
        .iter()
        .map(|slot| match slot.domain {
            EmbedDomain::CoeffSignBypass => cover.coeff_sign_bypass.bits[slot.intra_index],
            EmbedDomain::CoeffSuffixLsb => cover.coeff_suffix_lsb.bits[slot.intra_index],
            EmbedDomain::MvdSignBypass | EmbedDomain::MvdSuffixLsb => {
                unreachable!("priority_slots restricted to residual domains")
            }
        })
        .collect();

    for &parity_len in &SHADOW_PARITY_TIERS {
        let k = 255usize.saturating_sub(parity_len);
        if k < 2 {
            continue;
        }
        let max_rs_bytes = all_lsbs.len() / 8;
        let max_fdl = compute_max_shadow_fdl(max_rs_bytes, parity_len)
            .min(MAX_SHADOW_FRAME_BYTES);
        if SHADOW_FRAME_OVERHEAD > max_fdl {
            continue;
        }

        // First-block peek (works for fdl >= k — most messages).
        if let Some(fdl) = peek_fdl_from_first_block(&all_lsbs, parity_len, max_fdl)
            && let Some(result) = try_single_fdl(&all_lsbs, fdl, parity_len, passphrase)
        {
            return result;
        }

        // Small-fdl fallback: tiny payloads where fdl < k.
        let small_max = (k - 1).min(max_fdl);
        if SHADOW_FRAME_OVERHEAD > small_max {
            continue;
        }
        for fdl in SHADOW_FRAME_OVERHEAD..=small_max {
            if let Some(result) = try_single_fdl(&all_lsbs, fdl, parity_len, passphrase) {
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
    use super::*;
    use super::super::DomainBits;

    /// Build a synthetic 4-domain cover with random-ish positions
    /// + bits across all four bypass-bin domains, suitable for
    /// testing the shadow prepare → extract loop in isolation
    /// (no encoder involved).
    fn synth_cover(n_per_domain: usize) -> DomainCover {
        let mut cover = DomainCover::default();
        let mut s: u32 = 0xDEAD_BEEF;
        let mut next_key = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            s
        };
        let mut push_bits = |bits: &mut DomainBits, domain: EmbedDomain| {
            for _ in 0..n_per_domain {
                let raw = next_key();
                let bit = (raw & 1) as u8;
                let path = super::super::SyntaxPath::Mvd {
                    list: 0, partition: 0,
                    axis: super::super::Axis::X,
                    kind: super::super::BinKind::Sign,
                };
                let key = PositionKey::new(
                    (raw >> 16) & 0xFF,
                    (raw >> 8) & 0xFFFF,
                    domain,
                    path,
                );
                bits.bits.push(bit);
                bits.positions.push(key);
            }
        };
        push_bits(&mut cover.coeff_sign_bypass, EmbedDomain::CoeffSignBypass);
        push_bits(&mut cover.coeff_suffix_lsb, EmbedDomain::CoeffSuffixLsb);
        push_bits(&mut cover.mvd_sign_bypass, EmbedDomain::MvdSignBypass);
        push_bits(&mut cover.mvd_suffix_lsb, EmbedDomain::MvdSuffixLsb);
        cover
    }

    /// 4-domain helpers: prepare a shadow + inject bits + extract
    /// from the same cover → round-trip cleanly.
    #[test]
    fn shadow_all4_prepare_inject_extract_roundtrip() {
        let mut cover = synth_cover(2000); // 8000 positions total

        let state = prepare_shadow_all4(
            &cover, "test-pass", "hello shadow", &[], 4,
        ).expect("prepare 4-domain shadow");

        // Inject shadow bits into all 4 domain bit arrays.
        let (csb, csl, msb, msl) = (
            std::mem::take(&mut cover.coeff_sign_bypass.bits),
            std::mem::take(&mut cover.coeff_suffix_lsb.bits),
            std::mem::take(&mut cover.mvd_sign_bypass.bits),
            std::mem::take(&mut cover.mvd_suffix_lsb.bits),
        );
        let mut csb = csb;
        let mut csl = csl;
        let mut msb = msb;
        let mut msl = msl;
        embed_shadow_lsb_all4(&mut csb, &mut csl, &mut msb, &mut msl, &state);
        cover.coeff_sign_bypass.bits = csb;
        cover.coeff_suffix_lsb.bits = csl;
        cover.mvd_sign_bypass.bits = msb;
        cover.mvd_suffix_lsb.bits = msl;

        // Extract — should recover "hello shadow".
        let recovered = shadow_extract_all4(&cover, "test-pass")
            .expect("extract 4-domain shadow");
        assert_eq!(recovered.text, "hello shadow");
    }

    /// 4-domain shadow positions span all 4 domains (probabilistic
    /// — at parity 4 with ~80 bytes shadow ⇒ 640 bits worth of
    /// positions, spread across ~8000-position cover, all 4 domains
    /// will appear in top-N with overwhelming likelihood).
    #[test]
    fn shadow_all4_positions_span_all_domains() {
        let cover = synth_cover(2000);
        let state = prepare_shadow_all4(
            &cover, "test-pass", "x", &[], 4,
        ).expect("prepare");
        let mut domain_count = std::collections::HashMap::new();
        for slot in state.positions.iter().take(state.n_total) {
            *domain_count.entry(slot.domain as u8).or_insert(0usize) += 1;
        }
        assert!(domain_count.len() >= 3,
            "expected positions in 3+ domains, got {:?}", domain_count);
    }

    // ─── §6E-C2 polish single-cover helpers ──────────────────────

    /// `priority_slots_4domain_over_cover` is a deterministic
    /// re-export of `priority_slots_all4` — same inputs → same
    /// output ordering. Decoder matches encoder when both call
    /// either name with the same cover + perm seed.
    #[test]
    fn priority_slots_4domain_over_cover_matches_priority_slots_all4() {
        let cover = synth_cover(500);
        let seed = [42u8; 32];

        let a = priority_slots_4domain_over_cover(&cover, &seed);
        let b = priority_slots_all4(&cover, &seed);

        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.domain as u8, y.domain as u8);
            assert_eq!(x.intra_index, y.intra_index);
            assert_eq!(x.priority, y.priority);
        }
    }

    /// `prepare_shadow_over_emit_cover` matches `prepare_shadow_all4`
    /// — same crypto path, same priority sort, same cover →
    /// identical `ShadowState`. The only behavioral difference is
    /// architectural intent (cover argument is the primary-emit
    /// cover, not `cover_p1` / `cover_p1b_residual`).
    #[test]
    fn prepare_shadow_over_emit_cover_matches_prepare_shadow_all4() {
        let cover = synth_cover(2000);
        let pass = "polish-test-pass";
        let msg = "polish hello";
        let parity = 4;

        let polish = prepare_shadow_over_emit_cover(
            &cover, pass, msg, &[], parity,
        ).expect("polish prepare");
        let legacy = prepare_shadow_all4(
            &cover, pass, msg, &[], parity,
        ).expect("legacy prepare");

        assert_eq!(polish.n_total, legacy.n_total);
        assert_eq!(polish.parity_len, legacy.parity_len);
        assert_eq!(polish.frame_data_len, legacy.frame_data_len);
        // RS bits are deterministic from (frame, parity_len), so
        // ciphertext-derived `bits` may differ between calls (random
        // nonce in encrypt). Length must match.
        assert_eq!(polish.bits.len(), legacy.bits.len());
        // Position selection is deterministic from (cover, perm_seed).
        // perm_seed comes from passphrase, so both calls have the
        // same perm_seed. Positions must match exactly.
        assert_eq!(polish.positions.len(), legacy.positions.len());
        for (a, b) in polish.positions.iter().zip(legacy.positions.iter()) {
            assert_eq!(a.domain as u8, b.domain as u8);
            assert_eq!(a.intra_index, b.intra_index);
            assert_eq!(a.priority, b.priority);
        }
    }

    /// `prepare_shadow_over_emit_cover` + `embed_shadow_lsb_all4` +
    /// `shadow_extract_all4` form a self-consistent round-trip when
    /// encoder and decoder see the SAME cover (synthetic case —
    /// real-world the encoder has primary-emit cover and decoder has
    /// final-emit cover, which differ only at shadow override
    /// positions).
    #[test]
    fn shadow_over_emit_cover_roundtrip_self_consistent() {
        let mut cover = synth_cover(2000);

        let state = prepare_shadow_over_emit_cover(
            &cover, "polish-roundtrip", "single-cover msg", &[], 4,
        ).expect("polish prepare");

        let (csb, csl, msb, msl) = (
            std::mem::take(&mut cover.coeff_sign_bypass.bits),
            std::mem::take(&mut cover.coeff_suffix_lsb.bits),
            std::mem::take(&mut cover.mvd_sign_bypass.bits),
            std::mem::take(&mut cover.mvd_suffix_lsb.bits),
        );
        let mut csb = csb;
        let mut csl = csl;
        let mut msb = msb;
        let mut msl = msl;
        embed_shadow_lsb_all4(&mut csb, &mut csl, &mut msb, &mut msl, &state);
        cover.coeff_sign_bypass.bits = csb;
        cover.coeff_suffix_lsb.bits = csl;
        cover.mvd_sign_bypass.bits = msb;
        cover.mvd_suffix_lsb.bits = msl;

        let recovered = shadow_extract_all4(&cover, "polish-roundtrip")
            .expect("extract polish-prepared shadow");
        assert_eq!(recovered.text, "single-cover msg");
    }
}
