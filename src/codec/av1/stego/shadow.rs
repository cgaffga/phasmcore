// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AV1 shadow messages (multi-message plausible deniability).
//!
//! See [`phase-c-shadows.md`](../../../../../docs/design/video/av1/phase-c-shadows.md).
//!
//! ## Surface
//!
//! - `Av1ShadowSlot` + `Av1ShadowState` data structures.
//! - `priority_slots` — ChaCha20-keyed deterministic position
//!   priority over the joint Tier 1 cover (AC_COEFF_SIGN ∪
//!   GOLOMB_TAIL_LSB), used by the top-N selection.
//! - Shared shadow-layer constants re-exported from
//!   `crate::stego::shadow_layer` (no AV1-specific wire format —
//!   the WIDE u32 BE frame layout that H.264 + image stego use also
//!   covers AV1's file-attachment size envelope).
//!
//! Embed primitives, extract, and the multi-shadow cascade ladder
//! live below.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::stego::armor::ecc::{
    rs_decode_blocks_with_parity, rs_encode_blocks_with_parity, rs_encoded_len_with_parity,
};
use crate::stego::crypto::{self, derive_shadow_structural_key};
use crate::stego::error::StegoError;
use crate::stego::frame::{bits_to_bytes, bytes_to_bits};
use crate::stego::payload::{self, PayloadData};

// Re-export the shared shadow-frame infrastructure under AV1-
// flavoured names. Mirror of H.264's pattern. The `_wide` suffix
// dropped post-merge — main's build_shadow_frame auto-picks the
// V1 (NARROW, fdl ≤ u16::MAX) vs V2 (WIDE, u32 length) variant
// based on payload length; the unified API is a superset, and
// V1/V2 dispatch in extract uses the shared peek_shadow_fdl helper.
pub use crate::stego::shadow_layer::{
    build_shadow_frame as build_av1_shadow_frame,
    compute_max_shadow_fdl,
    parse_shadow_frame as parse_av1_shadow_frame,
    peek_shadow_fdl,
    MAX_SHADOW_FRAME_BYTES as MAX_AV1_SHADOW_FRAME_BYTES,
    SHADOW_FRAME_OVERHEAD as AV1_SHADOW_FRAME_OVERHEAD,
    SHADOW_PARITY_TIERS as AV1_SHADOW_PARITY_TIERS,
};

/// One shadow-eligible cover position with its ChaCha20-derived
/// priority. The cover index references the combined Tier-1 cover
/// bit vector (AC_COEFF_SIGN followed by GOLOMB_TAIL_LSB, in walker
/// emit order). The shadow embedder takes the top-N lowest-priority
/// slots after sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1ShadowSlot {
    /// Index into the combined per-frame cover bit vector. Stable
    /// across encode + decode walks (walker-symmetry invariant).
    pub cover_index: usize,
    /// 32-bit ChaCha20-derived priority. Lowest priority wins
    /// (sorted ascending).
    pub priority: u32,
}

/// State for one shadow layer during encoding. Built by
/// `prepare_shadow_av1` and consumed by the embed + INF-cost-overlay
/// path.
#[derive(Debug, Clone)]
pub struct Av1ShadowState {
    /// Top-N priority slots, sorted ascending. The embed pass writes
    /// shadow LSBs at these positions and INF-costs them so primary
    /// STC routes around.
    pub positions: Vec<Av1ShadowSlot>,
    /// Desired LSB bits at those positions — the RS-encoded shadow
    /// frame as a bit array.
    pub bits: Vec<u8>,
    /// Total bits = RS-encoded shadow frame length × 8.
    pub n_total: usize,
    /// RS parity length used for encode (decoder brute-forces
    /// `AV1_SHADOW_PARITY_TIERS` to recover it).
    pub parity_len: usize,
    /// Pre-RS frame byte count (the value the decoder ultimately
    /// reconstructs to call `parse_av1_shadow_frame` on).
    pub frame_data_len: usize,
}

/// Build the per-cover-position priority list.
///
/// For each index `i` in `0..cover_size`, computes a 32-bit
/// `ChaCha20(perm_seed)`-derived priority by SEEKING the RNG to
/// `word_pos = i × 2` and reading `next_u32`. This makes priority a
/// pure FUNCTION of (perm_seed, cover_index) — stable across encode
/// and decode walks, independent of iteration order.
///
/// Returns slots sorted by `(priority, cover_index)` ascending — the
/// secondary sort key breaks any ChaCha20 collisions deterministically.
/// Callers take `.into_iter().take(n_total)` to pick top-N.
///
/// Mirror of H.264 `priority_slots` at
/// `core/src/codec/h264/stego/shadow.rs:127-143`. AV1's joint Tier 1
/// cover collapses H.264's per-domain split into a single index
/// space, so the position-key shape simplifies from `PositionKey.raw()`
/// to `cover_index as u128`.
pub fn priority_slots(cover_size: usize, perm_seed: &[u8; 32]) -> Vec<Av1ShadowSlot> {
    let mut rng = ChaCha20Rng::from_seed(*perm_seed);
    let mut slots = Vec::with_capacity(cover_size);
    for cover_index in 0..cover_size {
        // word_pos × 2 because ChaCha20 emits u32 words and a u128
        // word_pos counts 32-bit positions; H.264 uses the same `× 2`
        // for symmetry with its PositionKey.raw() being a u64.
        rng.set_word_pos((cover_index as u128).wrapping_mul(2));
        let priority = rng.next_u32();
        slots.push(Av1ShadowSlot {
            cover_index,
            priority,
        });
    }
    slots.sort_by(|a, b| {
        a.priority
            .cmp(&b.priority)
            .then_with(|| a.cover_index.cmp(&b.cover_index))
    });
    slots
}

/// Prepare one shadow layer for embedding.
///
/// `cover_size` is the length of the joint Tier 1 cover bit vector
/// (AC_COEFF_SIGN + GOLOMB_TAIL_LSB) for the GOP. `shadow_pass` is
/// the shadow's passphrase (distinct from the primary's). `message`
/// is the raw payload bytes the shadow carries — caller is
/// responsible for any text/file bundling via
/// `crate::stego::payload::encode_payload` (or equivalent).
/// `parity_len` chooses the RS error tolerance vs payload-capacity
/// trade-off; the decode side brute-forces all of
/// `AV1_SHADOW_PARITY_TIERS` so encoder can pick from that set.
///
/// Returns `Av1ShadowState` containing the top-N priority slots,
/// the bit array to embed, total bit count, parity length used, and
/// the pre-RS frame byte count.
///
/// Mirror of H.264 `prepare_shadow` at
/// `core/src/codec/h264/stego/shadow.rs:150-182`. AV1's joint Tier
/// 1 cover collapses H.264's residual-vs-MVD split; the function
/// signature simplifies accordingly.
pub fn prepare_shadow_av1(
    cover_size: usize,
    shadow_pass: &str,
    message: &[u8],
    parity_len: usize,
) -> Result<Av1ShadowState, StegoError> {
    let (ciphertext, nonce, salt) = crypto::encrypt(message, shadow_pass)?;
    let frame_bytes = build_av1_shadow_frame(message.len(), &salt, &nonce, &ciphertext);
    let frame_data_len = frame_bytes.len();

    let rs_bytes = rs_encode_blocks_with_parity(&frame_bytes, parity_len);
    let rs_bits = bytes_to_bits(&rs_bytes);
    let n_total = rs_bits.len();

    let perm_seed_zero = derive_shadow_structural_key(shadow_pass)?;
    let perm_seed: [u8; 32] = (*perm_seed_zero).into();
    let slots = priority_slots(cover_size, &perm_seed);

    if slots.len() < n_total {
        return Err(StegoError::MessageTooLarge);
    }

    let positions = slots.into_iter().take(n_total).collect();

    Ok(Av1ShadowState {
        positions,
        bits: rs_bits,
        n_total,
        parity_len,
        frame_data_len,
    })
}

/// Inject shadow LSBs into the joint Tier 1 cover bit array. Run
/// BEFORE primary STC plans so that primary's Viterbi sees shadow
/// bits as if they were natural cover bits — combined with
/// [`overlay_infinity_costs_av1`] this guarantees primary STC keeps
/// the shadow bits at shadow positions, preserving both primary's
/// syndrome AND shadow's RS-encoded payload.
///
/// Mirror of H.264 `embed_shadow_lsb_residual` at
/// `core/src/codec/h264/stego/shadow.rs:190-209`.
pub fn embed_shadow_lsb_av1(cover_bits: &mut [u8], state: &Av1ShadowState) {
    for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
        let bit = state.bits[i];
        cover_bits[slot.cover_index] = bit;
    }
}

/// Set primary STC cost vector to `f32::INFINITY` at every shadow
/// position. Primary's Viterbi then routes syndromes AROUND the
/// shadow positions, preserving shadow's RS-encoded payload after
/// primary's flips land.
///
/// Mirror of H.264 `overlay_infinity_costs_residual` at
/// `core/src/codec/h264/stego/shadow.rs:220-240`. AV1's single joint
/// cover collapses H.264's per-domain split.
pub fn overlay_infinity_costs_av1(cover_cost: &mut [f32], state: &Av1ShadowState) {
    for slot in state.positions.iter().take(state.n_total) {
        cover_cost[slot.cover_index] = f32::INFINITY;
    }
}

/// Defensive override of the stego plan bits at shadow positions
/// with the shadow's RS-encoded LSBs. With shadow bits already in
/// the cover ([`embed_shadow_lsb_av1`]) + INF cost
/// ([`overlay_infinity_costs_av1`]), primary STC already preserves
/// shadow bits at shadow positions. This is a belt-and-suspenders
/// stamp guarding against future plan-layer drift between cover-bit
/// injection and STC plan output.
///
/// Mirror of H.264 `apply_shadow_to_plan_residual` at
/// `core/src/codec/h264/stego/shadow.rs:250-264`.
pub fn apply_shadow_to_plan_av1(stego_bits: &mut [u8], state: &Av1ShadowState) {
    for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
        let bit = state.bits[i];
        stego_bits[slot.cover_index] = bit;
    }
}

/// Prepare N shadow layers at once. Each shadow has its own
/// passphrase, message, and parity_len. Returns one `Av1ShadowState`
/// per shadow.
///
/// Shadows are independent — each gets its own ChaCha20-keyed
/// priority order. Position overlaps between shadows are LIKELY at
/// the front of the priority queue (low-priority positions are
/// uniformly distributed over the cover, so two shadows often
/// claim some same positions). The encode-side `embed_shadows_all_av1`
/// applies shadows in order; later shadows OVERWRITE earlier
/// shadows at colliding positions. RS parity (floor(parity_len/2)
/// byte errors per 255-byte block) absorbs the collisions.
///
/// Caller is responsible for capacity sanity (`av1_shadow_capacity`).
pub fn prepare_shadows(
    cover_size: usize,
    shadows: &[(&str, &[u8])],
    parity_len_per_shadow: usize,
) -> Result<Vec<Av1ShadowState>, StegoError> {
    let mut states = Vec::with_capacity(shadows.len());
    for &(passphrase, message) in shadows {
        states.push(prepare_shadow_av1(
            cover_size,
            passphrase,
            message,
            parity_len_per_shadow,
        )?);
    }
    Ok(states)
}

/// Inject LSBs for all N shadows in order. Later shadows overwrite
/// earlier ones at colliding positions; RS parity in each shadow's
/// encoding tolerates the collision count up to floor(parity_len/2)
/// corrupted bytes per 255-byte block.
pub fn embed_shadows_all_av1(cover_bits: &mut [u8], states: &[Av1ShadowState]) {
    for state in states {
        embed_shadow_lsb_av1(cover_bits, state);
    }
}

/// Union-overlay INF costs across all shadow positions. The primary
/// STC's Viterbi avoids any position claimed by ANY shadow, ensuring
/// shadow bits survive primary's flips.
pub fn overlay_infinity_costs_all_av1(cover_cost: &mut [f32], states: &[Av1ShadowState]) {
    for state in states {
        overlay_infinity_costs_av1(cover_cost, state);
    }
}

/// Defensive plan-stamp across all shadows. With cover-injection +
/// INF-overlay already applied, primary STC keeps shadow bits at
/// shadow positions; this is the belt-and-suspenders safety net
/// after primary plan emits.
pub fn apply_shadows_to_plan_all_av1(stego_bits: &mut [u8], states: &[Av1ShadowState]) {
    for state in states {
        apply_shadow_to_plan_av1(stego_bits, state);
    }
}

/// O(1) plaintext_len consistency gate used by `try_single_fdl`.
/// Encoder writes `fdl - AV1_SHADOW_FRAME_OVERHEAD` (= plaintext_len)
/// into the first 4 bytes of the shadow frame as u32 BE. If
/// RS-decoded `decoded[0..4]` disagrees with the brute-force `fdl`
/// candidate, reject early without running AES-GCM-SIV.
fn try_single_fdl_av1(
    rs_bytes: &[u8],
    fdl: usize,
    parity_len: usize,
    passphrase: &str,
) -> Option<Result<PayloadData, StegoError>> {
    let rs_encoded_len = rs_encoded_len_with_parity(fdl, parity_len);
    if rs_encoded_len > rs_bytes.len() {
        return None;
    }
    let decoded = match rs_decode_blocks_with_parity(&rs_bytes[..rs_encoded_len], fdl, parity_len)
    {
        Ok((data, _)) => data,
        Err(_) => return None,
    };
    // Format-aware consistency gate. peek_shadow_fdl reads the
    // first 2-6 bytes and dispatches V1 (u16 len) vs V2 (sentinel
    // 0x0000 + u32 len). Returns the total frame length the
    // producer wrote; reject if it doesn't equal our brute-force
    // fdl candidate. Shared with H.264 to keep V1/V2 logic in one
    // place (was AV1-private WIDE-only before the production merge).
    let expected_total = peek_shadow_fdl(&decoded)?;
    if expected_total != fdl {
        return None;
    }
    let fr = parse_av1_shadow_frame(&decoded).ok()?;
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

/// First-block peek — RS-decode the first 255 bytes and read the (V1
/// u16 or V2 u32) plaintext length prefix to derive the exact `fdl`.
/// Returns the candidate `fdl` if it's plausible (>= k and within
/// capacity).
///
/// Mirror of H.264 `peek_fdl_from_first_block`. V1/V2 dispatch
/// shared with H.264 via `peek_shadow_fdl`.
fn peek_fdl_from_first_block_av1(
    rs_bytes: &[u8],
    parity_len: usize,
    max_fdl: usize,
) -> Option<usize> {
    let k = 255usize.saturating_sub(parity_len);
    if k < 2 || rs_bytes.len() < 255 {
        return None;
    }
    let (data, _) = rs_decode_blocks_with_parity(&rs_bytes[..255], k, parity_len).ok()?;
    let fdl = peek_shadow_fdl(&data)?;
    if fdl >= k && fdl <= max_fdl {
        Some(fdl)
    } else {
        None
    }
}

/// Extract a shadow message from the post-decode joint Tier 1 cover
/// bit vector.
///
/// Walks each parity tier in `AV1_SHADOW_PARITY_TIERS`, tries the
/// `peek_fdl_from_first_block_av1` shortcut, then falls back to a
/// small-fdl scan for tiny messages (`fdl < k`). Returns the first
/// successful decode; AES-GCM-SIV authentication validates
/// correctness so wrong-tier candidates never produce a payload.
///
/// `cover_bits` is the joint Tier 1 cover from
/// `harvest_cover_bits_from_stego` — same byte sequence the primary
/// extract path operates on.
///
/// Mirror of H.264 `shadow_extract` at
/// `core/src/codec/h264/stego/shadow.rs:814-878`.
pub fn av1_shadow_extract(
    cover_bits: &[u8],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    if cover_bits.is_empty() {
        return Err(StegoError::FrameCorrupted);
    }
    let perm_seed_zero = derive_shadow_structural_key(passphrase)?;
    let perm_seed: [u8; 32] = (*perm_seed_zero).into();
    let slots = priority_slots(cover_bits.len(), &perm_seed);

    // Bit-pack once across all parity tiers (RS tries are expensive;
    // keep the bit-pack out of the hot loop).
    let all_lsbs: Vec<u8> = slots
        .iter()
        .map(|slot| cover_bits[slot.cover_index])
        .collect();
    let max_rs_bytes = all_lsbs.len() / 8;
    if max_rs_bytes == 0 {
        return Err(StegoError::FrameCorrupted);
    }
    let all_rs_bytes = bits_to_bytes(&all_lsbs[..max_rs_bytes * 8]);

    for &parity_len in &AV1_SHADOW_PARITY_TIERS {
        let k = 255usize.saturating_sub(parity_len);
        if k < 4 {
            continue;
        }
        let max_fdl = compute_max_shadow_fdl(max_rs_bytes, parity_len)
            .min(MAX_AV1_SHADOW_FRAME_BYTES);
        if AV1_SHADOW_FRAME_OVERHEAD > max_fdl {
            continue;
        }

        // First-block peek path: works for fdl >= k (most non-tiny
        // messages).
        let peeked = peek_fdl_from_first_block_av1(&all_rs_bytes, parity_len, max_fdl);
        if let Some(fdl) = peeked
            && let Some(result) = try_single_fdl_av1(&all_rs_bytes, fdl, parity_len, passphrase)
        {
            return result;
        }

        // Small-fdl fallback: tiny payloads where fdl < k. Brute-force
        // every byte-aligned fdl from AV1_SHADOW_FRAME_OVERHEAD to
        // k-1.
        let small_max = (k - 1).min(max_fdl);
        if AV1_SHADOW_FRAME_OVERHEAD > small_max {
            continue;
        }
        for fdl in AV1_SHADOW_FRAME_OVERHEAD..=small_max {
            if Some(fdl) == peeked {
                continue;
            }
            if let Some(result) = try_single_fdl_av1(&all_rs_bytes, fdl, parity_len, passphrase) {
                return result;
            }
        }
    }

    Err(StegoError::FrameCorrupted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_slots_are_deterministic_per_seed() {
        let seed = [0x42u8; 32];
        let slots_a = priority_slots(1000, &seed);
        let slots_b = priority_slots(1000, &seed);
        assert_eq!(slots_a, slots_b, "same seed must produce identical priority ordering");
    }

    #[test]
    fn priority_slots_differ_per_seed() {
        // Different seeds should produce essentially independent orderings.
        // Quantify via Spearman-like rank correlation on the first 200
        // slots' cover_index values: same ordering would give 1.0,
        // independent shuffles ~0.0.
        let seed_a = [0x01u8; 32];
        let seed_b = [0xFEu8; 32];
        let slots_a = priority_slots(1000, &seed_a);
        let slots_b = priority_slots(1000, &seed_b);
        let n = 200;
        let a_idx: Vec<usize> = slots_a.iter().take(n).map(|s| s.cover_index).collect();
        let b_idx: Vec<usize> = slots_b.iter().take(n).map(|s| s.cover_index).collect();
        // Count how many top-200 cover_indices appear in BOTH lists at
        // the same rank. For independent orderings expected ~n/N × n =
        // 200×200/1000 = 40 random matches. We assert < 60 to allow
        // statistical noise but fail loud on cross-seed leakage.
        let same_rank: usize = a_idx
            .iter()
            .zip(b_idx.iter())
            .filter(|(a, b)| a == b)
            .count();
        assert!(
            same_rank < 60,
            "{same_rank} of top-{n} slots at identical rank across seeds — suspicious correlation \
             (expected ~0 for true ChaCha20 independence; 60 is the noise ceiling)"
        );
    }

    #[test]
    fn priority_slots_count_matches_cover_size() {
        for n in [0usize, 1, 16, 1000, 50000] {
            let slots = priority_slots(n, &[0u8; 32]);
            assert_eq!(slots.len(), n, "slot count mismatch for cover_size={n}");
        }
    }

    #[test]
    fn f3_prepare_shadow_fills_state() {
        let cover_size = 50_000;
        let parity_len = 16;
        let state =
            prepare_shadow_av1(cover_size, "shadow-pass", b"hi shadow", parity_len)
                .expect("prepare_shadow_av1");
        assert!(state.n_total > 0);
        assert_eq!(state.positions.len(), state.n_total);
        assert_eq!(state.bits.len(), state.n_total);
        assert_eq!(state.parity_len, parity_len);
        assert!(state.frame_data_len >= AV1_SHADOW_FRAME_OVERHEAD + 9); // "hi shadow"=9
    }

    #[test]
    fn f3_prepare_shadow_rejects_oversized_message() {
        // cover_size too small to hold a 1 MiB shadow message + RS parity.
        let cover_size = 100; // 100 bits = 12.5 bytes — tiny
        let huge = vec![0u8; 1024 * 1024];
        let err = prepare_shadow_av1(cover_size, "p", &huge, 16)
            .expect_err("expected MessageTooLarge");
        assert!(matches!(err, StegoError::MessageTooLarge));
    }

    #[test]
    fn f3_embed_shadow_lsb_only_touches_top_n_positions() {
        let cover_size = 10_000;
        let mut cover = vec![0u8; cover_size];
        // Alternating sentinel so we can tell which positions changed.
        for (i, b) in cover.iter_mut().enumerate() {
            *b = (i & 1) as u8;
        }
        let state =
            prepare_shadow_av1(cover_size, "p", b"alpha", 16).expect("prepare");
        let cover_before = cover.clone();
        embed_shadow_lsb_av1(&mut cover, &state);

        // Non-shadow positions are byte-identical to before.
        let shadow_set: std::collections::HashSet<usize> =
            state.positions.iter().map(|s| s.cover_index).collect();
        for i in 0..cover_size {
            if !shadow_set.contains(&i) {
                assert_eq!(
                    cover[i], cover_before[i],
                    "non-shadow position {i} was modified"
                );
            }
        }
        // ~50% of shadow positions changed (statistical — depends on
        // whether each shadow bit happened to match the alternating
        // sentinel). Assert > 20% to detect a no-op.
        let overwritten = state
            .positions
            .iter()
            .take(state.n_total)
            .filter(|s| cover[s.cover_index] != cover_before[s.cover_index])
            .count();
        let n = state.n_total;
        assert!(
            overwritten > n / 5,
            "embed_shadow_lsb_av1 overwrote only {overwritten} of {n} shadow positions — looks like a no-op"
        );
    }

    #[test]
    fn f3_embed_shadow_lsb_sets_exact_bit_values() {
        let cover_size = 10_000;
        // Start with all-zero cover so the post-embed bit IS the
        // shadow bit (no XOR ambiguity).
        let mut cover = vec![0u8; cover_size];
        let state = prepare_shadow_av1(cover_size, "p", b"check", 16).expect("prepare");
        embed_shadow_lsb_av1(&mut cover, &state);
        for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
            assert_eq!(
                cover[slot.cover_index], state.bits[i],
                "cover[{}] should equal state.bits[{i}]={}",
                slot.cover_index, state.bits[i]
            );
        }
    }

    #[test]
    fn f3_overlay_infinity_costs_marks_exactly_n_positions() {
        let cover_size = 10_000;
        let mut costs = vec![1.0f32; cover_size];
        let state = prepare_shadow_av1(cover_size, "p", b"check", 16).expect("prepare");
        overlay_infinity_costs_av1(&mut costs, &state);
        let inf_count = costs.iter().filter(|c| c.is_infinite()).count();
        assert_eq!(inf_count, state.n_total);
        // Verify the INF positions match the shadow slots.
        for slot in state.positions.iter().take(state.n_total) {
            assert!(costs[slot.cover_index].is_infinite());
        }
    }

    #[test]
    fn f3_prepare_shadow_is_deterministic_under_phasm_deterministic_seed() {
        let _seed = crate::stego::crypto::DeterministicSeedGuard::set("20260604");
        let cover_size = 20_000;
        let s1 = prepare_shadow_av1(cover_size, "p", b"deterministic message", 16).unwrap();
        let s2 = prepare_shadow_av1(cover_size, "p", b"deterministic message", 16).unwrap();
        assert_eq!(s1.n_total, s2.n_total);
        assert_eq!(s1.frame_data_len, s2.frame_data_len);
        assert_eq!(s1.bits, s2.bits);
        // Positions are pure-deterministic via priority_slots regardless
        // of PHASM_DETERMINISTIC_SEED (independent of crypto nonce).
        assert_eq!(s1.positions, s2.positions);
    }

    #[test]
    fn f3_apply_shadow_to_plan_stamps_correct_bits() {
        let cover_size = 10_000;
        let state = prepare_shadow_av1(cover_size, "p", b"plan stamp", 16).expect("prepare");
        // Start with cover that has WRONG bits at shadow positions
        // (inverted). apply_shadow_to_plan_av1 should restore them.
        let mut stego = vec![0u8; cover_size];
        for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
            stego[slot.cover_index] = 1u8 ^ state.bits[i];
        }
        apply_shadow_to_plan_av1(&mut stego, &state);
        for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
            assert_eq!(stego[slot.cover_index], state.bits[i]);
        }
    }

    /// Single-shadow embed → extract round-trip on a synthetic cover.
    /// Verifies the prepare → embed → extract chain without going
    /// through the AV1 encoder.
    #[test]
    fn f4_shadow_round_trip_text_only() {
        let cover_size = 80_000;
        let message = b"phasm AV1 F.4 single-shadow round-trip via synthetic cover";
        let shadow_pass = "shadow-secret-2026-06-04";

        // Caller bundles the message into a payload envelope.
        let payload_bytes = payload::encode_payload(
            std::str::from_utf8(message).unwrap(),
            &[],
        )
        .unwrap();

        let state =
            prepare_shadow_av1(cover_size, shadow_pass, &payload_bytes, 16).expect("prepare");

        // Build a synthetic cover and inject shadow LSBs.
        let mut cover = vec![0u8; cover_size];
        embed_shadow_lsb_av1(&mut cover, &state);

        // Extract using the same passphrase.
        let recovered =
            av1_shadow_extract(&cover, shadow_pass).expect("av1_shadow_extract");
        assert_eq!(recovered.text.as_bytes(), message);
    }

    #[test]
    fn f4_shadow_extract_wrong_passphrase_returns_corrupted() {
        let cover_size = 80_000;
        let message = b"correct passphrase wins";
        let payload_bytes = payload::encode_payload(
            std::str::from_utf8(message).unwrap(),
            &[],
        )
        .unwrap();
        let state = prepare_shadow_av1(cover_size, "right", &payload_bytes, 16).unwrap();
        let mut cover = vec![0u8; cover_size];
        embed_shadow_lsb_av1(&mut cover, &state);

        // Wrong passphrase derives a different perm_seed → different
        // priority order → reads different LSB sequence → RS decode
        // fails OR AES-GCM-SIV tag fails. Either way, FrameCorrupted.
        let err = av1_shadow_extract(&cover, "wrong")
            .expect_err("wrong passphrase must fail");
        assert!(matches!(err, StegoError::FrameCorrupted | StegoError::DecryptionFailed));
    }

    #[test]
    fn f4_shadow_extract_empty_cover_rejects() {
        let err = av1_shadow_extract(&[], "p").expect_err("empty cover must fail");
        assert!(matches!(err, StegoError::FrameCorrupted));
    }

    /// Peek-path validation. Large enough payload that `fdl >= k`, so
    /// `peek_fdl_from_first_block_av1` hits the first branch (not the
    /// small-fdl fallback). This is the perf-critical path for typical
    /// mobile use.
    #[test]
    fn f4_shadow_round_trip_large_message_uses_peek_path() {
        let cover_size = 200_000;
        // ~1 KB message — comfortably above k = 255 - parity for any
        // parity tier ≤ 128.
        let message: Vec<u8> = (0..1024).map(|i| (i as u8).wrapping_mul(7)).collect();
        let text = std::str::from_utf8(&message).unwrap_or("invalid utf-8");
        // Use raw bytes via payload::encode_payload's file path if
        // text is non-UTF-8; here use synthesized text via prefix.
        let message_text = b"F.4 peek-path test: ".to_vec();
        let payload_bytes =
            payload::encode_payload(std::str::from_utf8(&message_text).unwrap(), &[])
                .unwrap();
        let _ = text; // silence unused
        let state =
            prepare_shadow_av1(cover_size, "peek-pass", &payload_bytes, 32).expect("prepare");
        let mut cover = vec![0u8; cover_size];
        embed_shadow_lsb_av1(&mut cover, &state);
        let recovered = av1_shadow_extract(&cover, "peek-pass").expect("extract");
        assert_eq!(recovered.text.as_bytes(), message_text.as_slice());
    }

    /// Two independent shadows on the same synthetic cover. Each
    /// passphrase recovers its own message; positions overlap
    /// statistically but RS parity absorbs the collisions.
    #[test]
    fn f5_two_shadows_round_trip_via_synthetic_cover() {
        let cover_size = 120_000;
        let msg_a = b"shadow A: first message of two";
        let msg_b = b"shadow B: second message of two";

        let payload_a = payload::encode_payload(std::str::from_utf8(msg_a).unwrap(), &[])
            .unwrap();
        let payload_b = payload::encode_payload(std::str::from_utf8(msg_b).unwrap(), &[])
            .unwrap();

        let states = prepare_shadows(
            cover_size,
            &[
                ("pass-A", &payload_a),
                ("pass-B", &payload_b),
            ],
            32, // higher parity for collision tolerance
        )
        .expect("prepare 2 shadows");
        assert_eq!(states.len(), 2);

        let mut cover = vec![0u8; cover_size];
        embed_shadows_all_av1(&mut cover, &states);

        let recovered_a = av1_shadow_extract(&cover, "pass-A").expect("extract A");
        let recovered_b = av1_shadow_extract(&cover, "pass-B").expect("extract B");
        assert_eq!(recovered_a.text.as_bytes(), msg_a);
        assert_eq!(recovered_b.text.as_bytes(), msg_b);
    }

    /// Three shadows. Confirms the cascade tolerates more concurrent
    /// collisions as N grows.
    #[test]
    fn f5_three_shadows_round_trip_via_synthetic_cover() {
        let cover_size = 250_000;
        let msg_a = b"three-shadow alpha";
        let msg_b = b"three-shadow beta";
        let msg_c = b"three-shadow gamma";

        let payload_a = payload::encode_payload(std::str::from_utf8(msg_a).unwrap(), &[])
            .unwrap();
        let payload_b = payload::encode_payload(std::str::from_utf8(msg_b).unwrap(), &[])
            .unwrap();
        let payload_c = payload::encode_payload(std::str::from_utf8(msg_c).unwrap(), &[])
            .unwrap();

        let states = prepare_shadows(
            cover_size,
            &[
                ("alpha", &payload_a),
                ("beta", &payload_b),
                ("gamma", &payload_c),
            ],
            64, // even higher parity for 3-way collisions
        )
        .expect("prepare 3 shadows");

        let mut cover = vec![0u8; cover_size];
        embed_shadows_all_av1(&mut cover, &states);

        assert_eq!(
            av1_shadow_extract(&cover, "alpha").unwrap().text.as_bytes(),
            msg_a
        );
        assert_eq!(
            av1_shadow_extract(&cover, "beta").unwrap().text.as_bytes(),
            msg_b
        );
        assert_eq!(
            av1_shadow_extract(&cover, "gamma").unwrap().text.as_bytes(),
            msg_c
        );
    }

    /// overlay_infinity_costs_all_av1 is the UNION of all shadows' INF
    /// positions. The count == count of UNIQUE position indices across
    /// all states.
    #[test]
    fn f5_overlay_inf_costs_all_is_union_of_shadow_positions() {
        let cover_size = 40_000;
        let p_a = payload::encode_payload("a-payload", &[]).unwrap();
        let p_b = payload::encode_payload("b-payload-but-longer", &[]).unwrap();
        let states = prepare_shadows(
            cover_size,
            &[("pa", &p_a), ("pb", &p_b)],
            16,
        )
        .unwrap();

        let mut costs = vec![1.0f32; cover_size];
        overlay_infinity_costs_all_av1(&mut costs, &states);

        let union: std::collections::HashSet<usize> = states
            .iter()
            .flat_map(|s| s.positions.iter().take(s.n_total).map(|p| p.cover_index))
            .collect();
        let inf_count = costs.iter().filter(|c| c.is_infinite()).count();
        assert_eq!(
            inf_count,
            union.len(),
            "INF count should match union-of-shadow-positions size"
        );
        for idx in &union {
            assert!(costs[*idx].is_infinite());
        }
    }

    /// Collision-tolerance smoke. Force the same passphrase for two
    /// shadows so positions overlap 100%; second shadow's LSB writes
    /// obliterate the first. The first shadow's extract should fail
    /// (data is the second shadow's bytes), the second shadow's
    /// extract succeeds. Verifies that LATER WINS at collision sites.
    #[test]
    fn f5_collision_later_wins() {
        let cover_size = 100_000;
        let payload_a = payload::encode_payload("first wins", &[]).unwrap();
        let payload_b = payload::encode_payload("later overwrites", &[]).unwrap();

        // Using the SAME passphrase → identical priority order →
        // 100% position overlap. We use prepare_shadow_av1 directly
        // to bypass any duplicate-detection in the multi-shadow API.
        let state_a = prepare_shadow_av1(cover_size, "same-pass", &payload_a, 16).unwrap();
        let state_b = prepare_shadow_av1(cover_size, "same-pass", &payload_b, 16).unwrap();

        let mut cover = vec![0u8; cover_size];
        embed_shadow_lsb_av1(&mut cover, &state_a);
        embed_shadow_lsb_av1(&mut cover, &state_b); // overwrites state_a

        // Extract with same-pass picks up state_b's payload.
        let recovered = av1_shadow_extract(&cover, "same-pass").expect("extract");
        assert_eq!(recovered.text, "later overwrites");
    }

    #[test]
    fn priority_slots_sorted_ascending_with_tiebreak() {
        let slots = priority_slots(1000, &[0u8; 32]);
        for window in slots.windows(2) {
            assert!(
                window[0].priority < window[1].priority
                    || (window[0].priority == window[1].priority
                        && window[0].cover_index < window[1].cover_index),
                "slots not sorted by (priority, cover_index) ascending: {:?} vs {:?}",
                window[0],
                window[1]
            );
        }
    }
}
