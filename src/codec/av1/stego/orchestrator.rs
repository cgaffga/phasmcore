// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Production av1_stego_encode / av1_stego_decode flow.
//!
//! Ties together:
//!   - phasm-rav1e [`encode_frame_with_phasm_tee`] (Pass 1: bytes + recorder)
//!   - phasm-core [`OverrideMap`] + [`replay_with_overrides`] (Pass 2)
//!   - phasm-core [`stc_embed`] + [`stc_extract`] (cover-bit → message)
//!   - phasm-core [`crypto`] (Argon2id + AES-GCM-SIV)
//!   - phasm-core [`frame`] (length-prefixed CRC-protected payload framing)
//!   - phasm-dav1d [`decode_with_recording`] (decoder side)
//!
//! Flow (see `docs/design/video/av1/streaming-session.md`):
//!
//! ## Encode
//! 1. Encrypt message via crypto::encrypt → (ciphertext, nonce, salt)
//! 2. Build payload frame via frame::build_frame (len + salt + nonce + CT + CRC)
//! 3. Encode via `encode_frame_with_phasm_tee` → OBU bytes + per-tile
//!    recorder + tile_group_offset
//! 4. Filter recorder bit_positions to AC_COEFF_SIGN positions
//! 5. Compute STC plan: embed payload bits into the AC cover bits → flips
//! 6. Build [`OverrideMap`] from the diff (cover_bit != stego_bit → flip)
//! 7. [`replay_with_overrides`] on the recorder storage → stego tile bytes
//!    (length-invariant 50/50 flips)
//! 8. Byte-splice stego tile bytes into natural OBU output at the
//!    tile_group_offset → final stego AV1 bytes
//!
//! ## Decode
//! 1. [`decode_with_recording`] on the stego AV1 bytes → all decoded
//!    50/50 positions + tags
//! 2. Filter to AC_COEFF_SIGN → stego AC bits
//! 3. Brute-force `w` candidates: for each, STC extract → parse_frame →
//!    decrypt. First valid hit wins (CRC + auth tag gate).
//! 4. Return plaintext.
//!
//! Initial AV1 scope: single-frame, single-tile, AC_COEFF_SIGN channel
//! only (see `docs/design/video/av1/channel-design.md`). Strict cursor
//! parity is the load-bearing equivalence — without it, the decoder
//! would extract at different positions than the encoder hid at.

#[cfg(feature = "av1-encoder")]
use phasm_rav1e::ec::WriterEncoder;
#[cfg(feature = "av1-encoder")]
use phasm_rav1e::phasm_stego::{
    AcSignMeta, PhasmFrameRecording, PHASM_TAG_AC_COEFF_SIGN,
    PHASM_TAG_GOLOMB_TAIL_LSB,
};
#[cfg(feature = "av1-encoder")]
use crate::stego::cost::av1_uniward::{
    Av1FramePosition, FramePlanes,
};

#[cfg(feature = "av1-decoder")]
use crate::codec::av1::stego::decoder::{decode_with_recording, Av1DecodeError};
#[cfg(feature = "av1-encoder")]
use crate::codec::av1::stego::writer::{replay_with_overrides, OverrideMap};

use crate::stego::error::StegoError;
#[cfg(feature = "av1-encoder")]
use crate::stego::stc::embed::stc_embed;
#[cfg(feature = "av1-decoder")]
use crate::stego::stc::extract::stc_extract;
use crate::stego::stc::hhat;
use crate::stego::{crypto, frame};

/// STC matrix height — fixed at 7 (matches phasm-core's standard).
const STC_H: usize = 7;

/// Errors from the AV1 stego orchestrator.
#[derive(Debug)]
pub enum Av1StegoError {
    /// Crypto / framing / payload error from phasm-core's stego stack.
    Stego(StegoError),
    /// dav1d-side decode error.
    #[cfg(feature = "av1-decoder")]
    Decode(Av1DecodeError),
    /// Encoder-side recording has no tiles (unexpected for any valid encode).
    EmptyRecording,
    /// Not enough AC_COEFF_SIGN cover positions for the message size.
    MessageTooLarge { needed_bits: usize, available_bits: usize },
    /// STC embedding returned None (cover too small / w bound violated).
    StcInfeasible,
    /// `replay_with_overrides` produced a tile_group of a different size
    /// than the natural encode. Indicates a Tier 1 invariant violation
    /// (50/50 flips should preserve length).
    TileGroupSizeMismatch { expected: usize, actual: usize },
    /// Decoder side: no AC_COEFF_SIGN positions found in the bitstream.
    #[cfg(feature = "av1-decoder")]
    NoCoverPositions,
    /// All brute-force `w` candidates failed to extract a valid frame.
    #[cfg(feature = "av1-decoder")]
    ExtractionFailed,
    /// Streaming-session validation failed (bad params, chunk_idx
    /// overflow, total_chunks drift across GOPs, etc.). Carries a
    /// human-readable message.
    InvalidPacket(String),
}

impl From<StegoError> for Av1StegoError {
    fn from(e: StegoError) -> Self {
        Av1StegoError::Stego(e)
    }
}

#[cfg(feature = "av1-decoder")]
impl From<Av1DecodeError> for Av1StegoError {
    fn from(e: Av1DecodeError) -> Self {
        Av1StegoError::Decode(e)
    }
}

/// Hide an encrypted message inside an AV1 frame's natural encode.
///
/// Takes the natural Pass 1 output (`natural_packet` + `recording`
/// from [`encode_frame_with_phasm_tee`]), the plaintext message, and
/// the passphrase. Returns final stego AV1 bytes that decode to the
/// same picture but carry the encrypted payload at the AC_COEFF_SIGN
/// cover positions per the STC plan.
///
/// # Initial AV1 scope
/// Single-tile, single-frame, AC_COEFF_SIGN-only.
#[cfg(feature = "av1-encoder")]
pub fn av1_stego_embed(
    natural_packet: Vec<u8>,
    recording: PhasmFrameRecording,
    message: &[u8],
    passphrase: &str,
) -> Result<Vec<u8>, Av1StegoError> {
    // 1. Derive keys from passphrase.
    let structural_key = crypto::derive_structural_key(passphrase)?;
    let hhat_seed: [u8; 32] = structural_key[32..]
        .try_into()
        .expect("derive_structural_key returns 64 bytes");

    // 2. Encrypt + frame the message into payload bits.
    let (ciphertext, nonce, salt) = crypto::encrypt(message, passphrase)?;
    let payload_bytes = frame::build_frame(message.len(), &salt, &nonce, &ciphertext);
    let payload_bits = frame::bytes_to_bits(&payload_bytes);

    // 3. Delegate to the bits-level inner. The streaming-session-v6
    // path (`session.rs`) calls this directly with pre-framed
    // chunk_frame bits to skip the per-GOP re-encrypt.
    av1_stego_embed_payload_bits(natural_packet, recording, &payload_bits, &hhat_seed)
}

/// Streaming entry point: embed pre-framed `payload_bits` into the
/// cover without performing AES encryption or `frame::build_frame`
/// wrapping. Caller is responsible for both (typically at
/// session_create).
///
/// Used by [`session::Av1StreamingEncodeSession`] to embed chunk_frame
/// bytes (the pre-encrypted + chunk-split payload) per GOP. The legacy
/// `av1_stego_embed` is now a thin wrapper around this — passing the
/// single-frame frame::build_frame output as payload bits and using the
/// passphrase-derived hhat_seed.
#[cfg(feature = "av1-encoder")]
pub(crate) fn av1_stego_embed_payload_bits(
    natural_packet: Vec<u8>,
    recording: PhasmFrameRecording,
    payload_bits: &[u8],
    hhat_seed: &[u8; 32],
) -> Result<Vec<u8>, Av1StegoError> {
    av1_stego_embed_payload_bits_with_shadows(
        natural_packet,
        recording,
        payload_bits,
        hhat_seed,
        &[],
    )
}

/// Extract a shadow message from a stego AV1 packet.
///
/// Walks the bitstream via dav1d, harvests the joint Tier-1 cover
/// (AC_COEFF_SIGN + GOLOMB_TAIL_LSB), then delegates to
/// `av1_shadow_extract` which iterates the parity-tier brute-force.
///
/// Note: this operates on the FULL cover, NOT n_used. The embed
/// path restricts shadows to [0, n_used), but the decoder doesn't
/// know n_used. priority_slots with cover.len() == full n
/// produces a SUPERSET of the embed-time top-N slots, sorted by the
/// same priority function. The encode-time top-N positions (which
/// are within [0, n_used)) appear at the SAME rank-positions in
/// the decoder's sort. The decoder's bit-pack at those positions
/// recovers the shadow bits.
///
/// Wrong-passphrase / corruption returns `StegoError::FrameCorrupted`.
#[cfg(feature = "av1-decoder")]
pub fn av1_stego_extract_shadow(
    stego_av1_bytes: &[u8],
    shadow_passphrase: &str,
) -> Result<crate::stego::payload::PayloadData, crate::stego::error::StegoError> {
    let cover_bits = harvest_cover_bits_from_stego(stego_av1_bytes).map_err(|e| match e {
        Av1StegoError::Stego(s) => s,
        Av1StegoError::Decode(_)
        | Av1StegoError::EmptyRecording
        | Av1StegoError::NoCoverPositions
        | Av1StegoError::ExtractionFailed
        | Av1StegoError::TileGroupSizeMismatch { .. }
        | Av1StegoError::StcInfeasible
        | Av1StegoError::MessageTooLarge { .. }
        | Av1StegoError::InvalidPacket(_) => crate::stego::error::StegoError::FrameCorrupted,
    })?;
    super::shadow::av1_shadow_extract(&cover_bits, shadow_passphrase)
}

/// Legacy single-frame embed + N shadows. Wraps the primary encrypt
/// + frame::build_frame for the main message, then delegates to
/// `av1_stego_embed_payload_bits_with_shadows`. Mirror of
/// `av1_stego_embed` plus a `shadows` parameter.
///
/// Each shadow is `(passphrase, payload_bytes)` — the payload bytes
/// should already be `payload::encode_payload(text, files)` output
/// when files are attached.
#[cfg(feature = "av1-encoder")]
pub fn av1_stego_embed_with_shadows(
    natural_packet: Vec<u8>,
    recording: PhasmFrameRecording,
    message: &[u8],
    passphrase: &str,
    shadows: &[(&str, &[u8])],
) -> Result<Vec<u8>, Av1StegoError> {
    let structural_key = crypto::derive_structural_key(passphrase)?;
    let hhat_seed: [u8; 32] = structural_key[32..]
        .try_into()
        .expect("derive_structural_key returns 64 bytes");
    let (ciphertext, nonce, salt) = crypto::encrypt(message, passphrase)?;
    let payload_bytes = frame::build_frame(message.len(), &salt, &nonce, &ciphertext);
    let payload_bits = frame::bytes_to_bits(&payload_bytes);
    av1_stego_embed_payload_bits_with_shadows(
        natural_packet,
        recording,
        &payload_bits,
        &hhat_seed,
        shadows,
    )
}

/// Embed payload bits + N shadows in one pass.
///
/// Shadow protocol (see `docs/design/video/av1/phase-c-shadows.md`):
///
/// 1. Snapshot the original Tier-1 cover BEFORE any shadow injection.
/// 2. Inject shadow LSBs at ChaCha20-priority-sorted top-N positions
///    over the (truncated to n_used) cover. Primary STC then sees
///    shadow bits as natural cover bits.
/// 3. Overlay `f32::INFINITY` cost at shadow positions so primary's
///    Viterbi routes around them.
/// 4. Run primary STC normally.
/// 5. Defensive plan-stamp: apply_shadows_to_plan over stego_bits in
///    case STC's path didn't perfectly avoid shadow positions (it
///    should, but belt-and-suspenders).
/// 6. Build the OverrideMap from the ORIGINAL (pre-shadow) cover vs
///    the post-STC + post-shadow-stamp stego_bits. At shadow
///    positions where the original cover bit happened to equal the
///    shadow bit, no override is recorded; otherwise an override
///    flips the natural OBU bit to the shadow bit.
///
/// Each shadow is `(passphrase: &str, payload_bytes: &[u8])`. Shadow
/// payload bytes have already been bundled (text + files) by the
/// caller via `crate::stego::payload::encode_payload` if applicable;
/// `prepare_shadow_av1` encrypts + frames + RS-encodes internally.
///
/// `shadow_parity_len` applies uniformly to all shadows. Per-shadow
/// parity selection is a follow-on.
#[cfg(feature = "av1-encoder")]
pub fn av1_stego_embed_payload_bits_with_shadows(
    natural_packet: Vec<u8>,
    recording: PhasmFrameRecording,
    payload_bits: &[u8],
    hhat_seed: &[u8; 32],
    shadows: &[(&str, &[u8])],
) -> Result<Vec<u8>, Av1StegoError> {
    av1_stego_embed_payload_bits_with_shadows_parity(
        natural_packet,
        recording,
        payload_bits,
        hhat_seed,
        shadows,
        16, // default parity_len — middle of AV1_SHADOW_PARITY_TIERS
    )
}

/// Same as `av1_stego_embed_payload_bits_with_shadows` with explicit
/// shadow parity_len. Useful for tests + capacity tuning.
#[cfg(feature = "av1-encoder")]
pub fn av1_stego_embed_payload_bits_with_shadows_parity(
    natural_packet: Vec<u8>,
    recording: PhasmFrameRecording,
    payload_bits: &[u8],
    hhat_seed: &[u8; 32],
    shadows: &[(&str, &[u8])],
    shadow_parity_len: usize,
) -> Result<Vec<u8>, Av1StegoError> {
    if recording.tiles.is_empty() {
        return Err(Av1StegoError::EmptyRecording);
    }

    let payload = payload_bits;

    // 4. Filter recorder's bit_positions to the Tier-1 channel set.
    //
    // Both GOLOMB_TAIL_LSB and AC_COEFF_SIGN are enrolled. The
    // phasm-rav1e + phasm-dav1d fork patches tag ONLY the literal
    // LSB of each golomb code (leading zeros + non-LSB literals get
    // OTHER), so every enrolled position is safe to flip ±1 without
    // changing the bitstream length or cascading through culLevel.
    let tile = &recording.tiles[0];
    let mut ac_global_cursors: Vec<u64> = Vec::new();
    let mut ac_cover_bits: Vec<u8> = Vec::new();
    let mut ac_metas: Vec<AcSignMeta> = Vec::new();
    for (cursor, ((&(_, value), &tag), &meta)) in tile
        .bit_positions
        .iter()
        .zip(tile.bit_tags.iter())
        .zip(tile.bit_meta.iter())
        .enumerate()
    {
        if tag == PHASM_TAG_AC_COEFF_SIGN || tag == PHASM_TAG_GOLOMB_TAIL_LSB {
            ac_global_cursors.push(cursor as u64);
            ac_cover_bits.push(value as u8);
            ac_metas.push(meta);
        }
    }

    let n = ac_cover_bits.len();
    // payload is now a bit array, so .len() IS the bit count.
    let m_bits = payload.len();
    if m_bits > n {
        return Err(Av1StegoError::MessageTooLarge {
            needed_bits: m_bits,
            available_bits: n,
        });
    }

    // 5. Compute STC params per phasm-core convention (Ghost
    // `compute_stc_params`): w = floor(n / m_bits), n_used = m_bits
    // * w. Each message bit gets `w` cover bits of slack. Truncate
    // cover to n_used so STC's Viterbi is balanced.
    let w = (n / m_bits.max(1)).max(1);
    let n_used = m_bits * w;

    // Prepare shadows over the FULL n cover (NOT n_used).
    //
    // Why FULL n: priority_slots's sort is over indices 0..N. If
    // encoder uses N_enc=n_used and decoder uses N_dec=n (the
    // post-harvest cover length), the two top-N sets diverge —
    // decoder's top-N may include indices in [n_used, n) whose
    // priorities sort BELOW some of encoder's top-N. With N_enc=n
    // both sides agree on the priority ordering.
    //
    // Consequence: SOME shadow positions may land in [n_used, n).
    // Primary STC operates only on [0, n_used) so these positions
    // are outside its scope. We handle them with explicit override
    // entries below (see "out-of-range shadow override" step).
    let shadow_states = if shadows.is_empty() {
        Vec::new()
    } else {
        super::shadow::prepare_shadows(n, shadows, shadow_parity_len)
            .map_err(Av1StegoError::Stego)?
    };

    // Snapshot the ORIGINAL full cover BEFORE injecting shadow LSBs.
    // The override map (below) compares against this; without it,
    // shadow positions whose natural bit happened to equal the shadow
    // bit would produce no override and the natural OBU bit would
    // survive instead of the shadow bit.
    let original_full_cover: Vec<u8> = ac_cover_bits.clone();

    // Inject shadow LSBs into the in-range portion of the cover
    // (`[0, n_used)`). Out-of-range positions are handled by the
    // explicit override step below.
    let mut shadow_position_set: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    for state in &shadow_states {
        for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
            if slot.cover_index < n_used {
                ac_cover_bits[slot.cover_index] = state.bits[i];
            }
            shadow_position_set.insert(slot.cover_index);
        }
    }

    let cover_used = &ac_cover_bits[..n_used];
    let cursors_used = &ac_global_cursors[..n_used];
    let metas_used = &ac_metas[..n_used];

    // 6. Generate HHAT matrix from passphrase-derived seed.
    let hhat_matrix = hhat::generate_hhat(STC_H, w, &hhat_seed);

    // 7. STC embed with J-UNIWARD costs. Each AC sign's cost reflects
    // how much its sign flip would perturb the Daubechies-8 wavelet
    // decomposition of the post-LR reconstructed pixel plane the flip
    // lives in. Positions with high-magnitude AC coefficients in
    // textured regions get LOW cost (safe to flip — the perturbation
    // hides in existing wavelet energy); positions in smooth regions
    // get HIGH cost.
    let frame_planes = pack_visible_planes(&recording.reconstructed_planes);
    let av1_positions: Vec<Av1FramePosition> = metas_used
        .iter()
        .map(|m| Av1FramePosition {
            plane: m.plane,
            plane_px_x: m.plane_px_x,
            plane_px_y: m.plane_px_y,
            tx_width_log2: m.tx_width_log2,
            tx_height_log2: m.tx_height_log2,
            tx_type: m.tx_type,
            scan_pos: m.scan_pos,
            // Carry encoder-side coefficient magnitude through for
            // cascade-safety v2's pre-filter + upper-bound + cache key.
            coeff_magnitude: m.coeff_magnitude,
        })
        .collect();
    // Pass the frame-level loop-filter state captured by the
    // fork-patch through to cost compute. This activates the
    // three-tier dispatch (|coeff|-based safe/reject bands + forward
    // model for the middle), replacing the earlier magnitude-proxy
    // path that wasn't discriminating cascade-amplified positions.
    let mut costs = crate::stego::cost::av1_uniward::compute_av1_uniward_costs_with_state(
        &frame_planes,
        &av1_positions,
        recording.frame_qindex,
        Some(recording.loop_filter_state),
    );

    // Overlay INF cost at IN-RANGE shadow positions so primary STC's
    // Viterbi routes around them. With shadow bits already injected
    // into cover[..n_used] + INF cost here, primary's plan preserves
    // shadow bits at shadow positions in [0, n_used).
    for &idx in &shadow_position_set {
        if idx < n_used {
            costs[idx] = f32::INFINITY;
        }
    }

    let mut embed_result = stc_embed(cover_used, &costs, &payload, &hhat_matrix, STC_H, w)
        .ok_or(Av1StegoError::StcInfeasible)?;

    // Defensive plan-stamp — re-apply shadow bits to
    // stego_bits[..n_used] at IN-RANGE shadow positions in case STC's
    // path didn't perfectly respect INF (it should, but
    // belt-and-suspenders).
    for state in &shadow_states {
        for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
            if slot.cover_index < n_used {
                embed_result.stego_bits[slot.cover_index] = state.bits[i];
            }
        }
    }

    // 8. Build OverrideMap from ORIGINAL_full_cover → stego
    // differences. Using the snapshot of the cover BEFORE shadow
    // injection ensures override entries fire at every shadow
    // position where the original natural bit differs from the
    // shadow's required bit. Without this, shadow positions where
    // the natural bit happened to equal the shadow bit would produce
    // no override and the natural OBU bit (not the shadow) would
    // survive.
    let mut plan = OverrideMap::new();
    for (i, &stego_bit) in embed_result.stego_bits.iter().enumerate() {
        // i is in [0, n_used)
        if stego_bit != original_full_cover[i] {
            plan.set(cursors_used[i], stego_bit as u16);
        }
    }

    // Out-of-range shadow override.
    // Shadow positions with cover_index in [n_used, n) are outside
    // primary STC's window. The replay path will pass them through
    // unchanged unless we add explicit override entries here.
    for state in &shadow_states {
        for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
            if slot.cover_index >= n_used {
                let want = state.bits[i];
                if want != original_full_cover[slot.cover_index] {
                    plan.set(ac_global_cursors[slot.cover_index], want as u16);
                }
            }
        }
    }
    let _ = cover_used; // silence unused-var hint; primary STC consumed it

    // 9. Replay through a fresh WriterEncoder with the override plan.
    // Produces the stego tile_group bytes.
    let mut sink = WriterEncoder::new();
    replay_with_overrides(&tile.storage, &tile.bit_positions, &plan, &mut sink);
    let stego_tile_bytes = sink.done();

    // 10. Splice stego tile bytes into the natural OBU output.
    // Fast path: same length → in-place byte-overwrite.
    // Slow path: length differs by ±1 byte (rare range-coder
    // trailing-carry edge case) → rebuild the frame_obu with a
    // recomputed ULEB128 size field.
    let final_packet = if stego_tile_bytes.len() == recording.tile_group_len {
        let mut packet = natural_packet;
        let dst = &mut packet[recording.tile_group_offset
            ..recording.tile_group_offset + recording.tile_group_len];
        dst.copy_from_slice(&stego_tile_bytes);
        packet
    } else {
        rebuild_obu_with_stego_tile_group(&natural_packet, &recording, &stego_tile_bytes)
    };
    Ok(final_packet)
}

/// Multi-frame-per-GOP stego encode primitive.
///
/// Takes the result of [`phasm_rav1e::phasm_stego::encode_gop_with_phasm_tee`]
/// (N per-frame `(natural_packet, recording)` pairs from a single GOP)
/// plus the GOP's payload bits, and produces the concatenated stego
/// OBU bytes for the whole GOP.
///
/// The combined cover for the GOP is the per-frame Tier-1 covers
/// (AC_COEFF_SIGN + GOLOMB_TAIL_LSB) concatenated in submission order.
/// A single STC plan runs over that combined cover; the resulting
/// stego_bits are split back into per-frame `OverrideMap`s and applied
/// per-frame via [`replay_with_overrides`].
///
/// Mirrors `av1_stego_embed_payload_bits_with_shadows_parity` for the
/// single-frame case (v=1 GOP). Differences:
///
/// - Per-frame cost vectors are computed independently (each frame
///   has its own `frame_qindex` + `loop_filter_state` +
///   `reconstructed_planes`) then concatenated.
/// - The override map is split per-frame so each frame's replay sees
///   only its own overrides; per-frame `tile_group_offset` /
///   `tile_group_len` / `frame_obu_start` are used for the per-frame
///   splice.
/// - Shadows are computed over the COMBINED cover (per-GOP scope), not
///   per-frame. The shadow priority sort over indices 0..n_combined
///   places shadow positions at consistent ranks regardless of which
///   frame each lives in. The override map split routes each shadow
///   override entry to the correct frame.
///
/// `shadow_parity_len` applies uniformly to all shadows. Per-shadow
/// parity selection lives at the wrapper layer.
#[cfg(feature = "av1-encoder")]
pub(crate) fn av1_stego_encode_one_gop(
    per_frame: Vec<(Vec<u8>, PhasmFrameRecording)>,
    payload_bits: &[u8],
    hhat_seed: &[u8; 32],
) -> Result<Vec<u8>, Av1StegoError> {
    av1_stego_encode_one_gop_with_shadows_parity(
        per_frame,
        payload_bits,
        hhat_seed,
        &[],
        0,
    )
}

/// Multi-frame GOP primitive + shadow embedding.
///
/// See [`av1_stego_encode_one_gop`] for the no-shadow flavor.
#[cfg(feature = "av1-encoder")]
pub(crate) fn av1_stego_encode_one_gop_with_shadows_parity(
    per_frame: Vec<(Vec<u8>, PhasmFrameRecording)>,
    payload_bits: &[u8],
    hhat_seed: &[u8; 32],
    shadows: &[(&str, &[u8])],
    shadow_parity_len: usize,
) -> Result<Vec<u8>, Av1StegoError> {
    if per_frame.is_empty() {
        return Err(Av1StegoError::EmptyRecording);
    }
    let n_frames = per_frame.len();

    // Per-frame harvest: Tier-1 cover bits + meta + global cursors +
    // per-position J-UNIWARD cost.
    struct FrameHarvest {
        global_cursors: Vec<u64>,
        cover_bits: Vec<u8>,
        costs: Vec<f32>,
    }
    let mut frames_harvest: Vec<FrameHarvest> = Vec::with_capacity(n_frames);

    for (_, recording) in per_frame.iter() {
        if recording.tiles.is_empty() {
            return Err(Av1StegoError::EmptyRecording);
        }
        let tile = &recording.tiles[0];

        let mut global_cursors: Vec<u64> = Vec::new();
        let mut cover_bits: Vec<u8> = Vec::new();
        let mut metas: Vec<AcSignMeta> = Vec::new();

        for (cursor, ((&(_, value), &tag), &meta)) in tile
            .bit_positions
            .iter()
            .zip(tile.bit_tags.iter())
            .zip(tile.bit_meta.iter())
            .enumerate()
        {
            if tag == PHASM_TAG_AC_COEFF_SIGN || tag == PHASM_TAG_GOLOMB_TAIL_LSB {
                global_cursors.push(cursor as u64);
                cover_bits.push(value as u8);
                metas.push(meta);
            }
        }

        // Per-frame J-UNIWARD cost vector. Each position uses THIS
        // frame's reconstructed planes + frame_qindex +
        // loop_filter_state — frames have independent quality
        // characteristics.
        let frame_planes = pack_visible_planes(&recording.reconstructed_planes);
        let av1_positions: Vec<Av1FramePosition> = metas
            .iter()
            .map(|m| Av1FramePosition {
                plane: m.plane,
                plane_px_x: m.plane_px_x,
                plane_px_y: m.plane_px_y,
                tx_width_log2: m.tx_width_log2,
                tx_height_log2: m.tx_height_log2,
                tx_type: m.tx_type,
                scan_pos: m.scan_pos,
                coeff_magnitude: m.coeff_magnitude,
            })
            .collect();
        let costs = crate::stego::cost::av1_uniward::compute_av1_uniward_costs_with_state(
            &frame_planes,
            &av1_positions,
            recording.frame_qindex,
            Some(recording.loop_filter_state),
        );

        frames_harvest.push(FrameHarvest {
            global_cursors,
            cover_bits,
            costs,
        });
    }

    // Build combined cover + costs + per-bit (frame_idx, frame_cursor)
    // back-reference index. The index is what lets us route a
    // combined-cover position back to the right frame's override map.
    let mut combined_cover: Vec<u8> = Vec::new();
    let mut combined_costs: Vec<f32> = Vec::new();
    let mut combined_index: Vec<(usize, u64)> = Vec::new();
    for (frame_idx, fh) in frames_harvest.iter().enumerate() {
        combined_cover.extend_from_slice(&fh.cover_bits);
        combined_costs.extend_from_slice(&fh.costs);
        for &c in &fh.global_cursors {
            combined_index.push((frame_idx, c));
        }
    }
    let n = combined_cover.len();

    let m_bits = payload_bits.len();
    if m_bits == 0 {
        return Err(Av1StegoError::InvalidPacket(
            "av1_stego_encode_one_gop: empty payload_bits".into(),
        ));
    }
    if m_bits > n {
        return Err(Av1StegoError::MessageTooLarge {
            needed_bits: m_bits,
            available_bits: n,
        });
    }

    let w = (n / m_bits).max(1);
    let n_used = m_bits * w;

    // Shadows over the FULL combined cover. Same protocol as
    // single-frame: priority-sort over indices 0..n, top-N positions
    // carry shadow LSBs.
    let shadow_states = if shadows.is_empty() {
        Vec::new()
    } else {
        super::shadow::prepare_shadows(n, shadows, shadow_parity_len)
            .map_err(Av1StegoError::Stego)?
    };

    // Snapshot ORIGINAL combined cover before shadow injection. Used
    // for OverrideMap diff so shadow positions whose natural bit
    // matches the shadow bit still produce no spurious override
    // entries.
    let original_full_cover: Vec<u8> = combined_cover.clone();

    let mut shadow_position_set: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    for state in &shadow_states {
        for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
            if slot.cover_index < n_used {
                combined_cover[slot.cover_index] = state.bits[i];
            }
            shadow_position_set.insert(slot.cover_index);
        }
    }

    let cover_used = &combined_cover[..n_used];

    let hhat_matrix = hhat::generate_hhat(STC_H, w, hhat_seed);

    // Overlay INF cost at IN-RANGE shadow positions so primary STC's
    // Viterbi routes around them.
    let mut costs_for_stc: Vec<f32> = combined_costs[..n_used].to_vec();
    for &idx in &shadow_position_set {
        if idx < n_used {
            costs_for_stc[idx] = f32::INFINITY;
        }
    }

    let mut embed_result = stc_embed(cover_used, &costs_for_stc, payload_bits, &hhat_matrix, STC_H, w)
        .ok_or(Av1StegoError::StcInfeasible)?;

    // Defensive plan-stamp: re-apply shadow bits to stego_bits in
    // case STC's path didn't perfectly respect INF cost.
    for state in &shadow_states {
        for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
            if slot.cover_index < n_used {
                embed_result.stego_bits[slot.cover_index] = state.bits[i];
            }
        }
    }

    // Split STC stego_bits → per-frame OverrideMaps.
    let mut per_frame_plans: Vec<OverrideMap> =
        (0..n_frames).map(|_| OverrideMap::new()).collect();
    for i in 0..n_used {
        if embed_result.stego_bits[i] != original_full_cover[i] {
            let (frame_idx, cursor) = combined_index[i];
            per_frame_plans[frame_idx].set(cursor, embed_result.stego_bits[i] as u16);
        }
    }

    // Out-of-range shadow override: shadow positions with cover_index
    // in [n_used, n) are outside primary STC's window. Add explicit
    // override entries; replay path passes them through unchanged
    // otherwise.
    for state in &shadow_states {
        for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
            if slot.cover_index >= n_used {
                let want = state.bits[i];
                if want != original_full_cover[slot.cover_index] {
                    let (frame_idx, cursor) = combined_index[slot.cover_index];
                    per_frame_plans[frame_idx].set(cursor, want as u16);
                }
            }
        }
    }

    // Per-frame replay + splice. Each frame's overrides land in that
    // frame's WriterRecorder storage; the resulting stego tile_group
    // bytes splice into the frame's natural OBU output via the same
    // path the single-frame primitive uses.
    let mut output = Vec::new();
    for (frame_idx, (natural_packet, recording)) in per_frame.into_iter().enumerate() {
        let tile = &recording.tiles[0];
        let mut sink = WriterEncoder::new();
        replay_with_overrides(
            &tile.storage,
            &tile.bit_positions,
            &per_frame_plans[frame_idx],
            &mut sink,
        );
        let stego_tile_bytes = sink.done();

        let final_packet = if stego_tile_bytes.len() == recording.tile_group_len {
            let mut packet = natural_packet;
            let dst = &mut packet[recording.tile_group_offset
                ..recording.tile_group_offset + recording.tile_group_len];
            dst.copy_from_slice(&stego_tile_bytes);
            packet
        } else {
            rebuild_obu_with_stego_tile_group(&natural_packet, &recording, &stego_tile_bytes)
        };

        output.extend_from_slice(&final_packet);
    }

    Ok(output)
}

/// Rebuild the frame_obu portion of the natural packet with the new
/// stego tile_group bytes, recomputing the ULEB128 size field so the
/// AV1 container stays consistent.
///
/// Handles the TileGroupSizeMismatch case: when stego flips happen
/// near the end of the entropy-coded stream, the range coder's
/// trailing-carry flush can produce a tile_group that's ±1 byte
/// different in length from the natural one. Previously this returned
/// `Av1StegoError::TileGroupSizeMismatch`; now the orchestrator
/// rebuilds the OBU header's size field and stitches the new
/// tile_group in cleanly.
///
/// Layout reminder (per `PhasmFrameRecording`):
/// ```text
/// natural_packet:
///   [0 .. frame_obu_start]              key-frame OBUs + metadata
///   [frame_obu_start]                   1-byte OBU_FRAME header
///   [frame_obu_start+1 .. fh_start]     ULEB128 of (fh_len + tg_len)
///   [fh_start .. tile_group_offset]     frame_header_obu payload
///   [tile_group_offset .. ...]          tile_group bytes
/// ```
#[cfg(feature = "av1-encoder")]
pub(crate) fn rebuild_obu_with_stego_tile_group(
    natural_packet: &[u8],
    recording: &PhasmFrameRecording,
    stego_tile_bytes: &[u8],
) -> Vec<u8> {
    let frame_header_start = recording.tile_group_offset - recording.frame_header_len;
    let frame_header_bytes = &natural_packet[frame_header_start..recording.tile_group_offset];
    let obu_header_byte = natural_packet[recording.frame_obu_start];

    // Compute new ULEB128 size for (frame_header + stego_tile_group).
    let new_payload_size = (recording.frame_header_len + stego_tile_bytes.len()) as u64;
    let new_uleb128 = encode_uleb128(new_payload_size);

    let mut packet = Vec::with_capacity(
        recording.frame_obu_start
            + 1
            + new_uleb128.len()
            + recording.frame_header_len
            + stego_tile_bytes.len(),
    );
    packet.extend_from_slice(&natural_packet[..recording.frame_obu_start]);
    packet.push(obu_header_byte);
    packet.extend_from_slice(&new_uleb128);
    packet.extend_from_slice(frame_header_bytes);
    packet.extend_from_slice(stego_tile_bytes);
    packet
}

/// Extract the visible-region pixel data from each plane of a
/// reconstructed Frame and pack into [`FramePlanes`] (contiguous, no
/// stride padding). The Frame buffer stores each plane with
/// filter-tap padding at `xorigin`/`yorigin` offsets; our J-UNIWARD
/// wavelet decomposition consumes packed YUV.
///
/// Memory cost: one Vec<u8> per plane = ~3 MB for 1080p. Could be
/// avoided via a wavelet decomp that consumes strided input directly
/// (small refactor of `compute_three_subbands`).
#[cfg(feature = "av1-encoder")]
pub(crate) fn pack_visible_planes(
    rec: &std::sync::Arc<phasm_rav1e::Frame<u8>>,
) -> FramePlanes {
    let pack = |plane_idx: usize| -> (Vec<u8>, usize, usize) {
        let p = &rec.planes[plane_idx];
        let w = p.cfg.width;
        let h = p.cfg.height;
        let stride = p.cfg.stride;
        let start = p.cfg.yorigin * stride + p.cfg.xorigin;
        let mut out = Vec::with_capacity(w * h);
        for row in 0..h {
            let row_start = start + row * stride;
            out.extend_from_slice(&p.data[row_start..row_start + w]);
        }
        (out, w, h)
    };
    let (y, luma_w, luma_h) = pack(0);
    let (cb, chroma_w, chroma_h) = pack(1);
    let (cr, _, _) = pack(2);
    FramePlanes {
        y,
        cb,
        cr,
        luma_width: luma_w,
        luma_height: luma_h,
        chroma_width: chroma_w,
        chroma_height: chroma_h,
    }
}

/// Encode a u64 as an AV1 ULEB128 byte sequence (LSB-first 7-bit
/// groups, continuation bit set on all but the last byte).
#[cfg(feature = "av1-encoder")]
fn encode_uleb128(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
            out.push(byte);
        } else {
            out.push(byte);
            return out;
        }
    }
}

/// Decode an AV1 ULEB128 from `bytes[at..]`. Returns
/// `(value, bytes_consumed)`. AV1 spec caps ULEB128 at 8 bytes
/// (56 bits of payload + 8 continuation bits).
pub(crate) fn decode_uleb128(bytes: &[u8], at: usize) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    let mut i = at;
    let mut consumed = 0usize;
    while i < bytes.len() && consumed < 8 {
        let b = bytes[i];
        let lo7 = (b & 0x7f) as u64;
        if shift >= 64 {
            return None;
        }
        value |= lo7 << shift;
        consumed += 1;
        i += 1;
        if (b & 0x80) == 0 {
            return Some((value, consumed));
        }
        shift += 7;
    }
    None
}

/// AV1 OBU type codes. Only the types we need for the walker —
/// others get classified as `Other`.
pub(crate) const OBU_SEQUENCE_HEADER: u8 = 1;

/// OBU walker: split accumulated AV1 bytes at every
/// `sequence_header_obu` boundary into per-GOP slabs. Bytes BEFORE
/// the first sequence_header are discarded; bytes between consecutive
/// sequence_headers form one slab each.
///
/// AV1 OBU header layout:
/// ```text
///   bit 7    : obu_forbidden_bit (must be 0)
///   bits 3-6 : obu_type           (4 bits)
///   bit 2    : obu_extension_flag (1 byte of extension header if set)
///   bit 1    : obu_has_size_field (ULEB128 size follows the headers)
///   bit 0    : obu_reserved (0)
/// ```
///
/// rav1e emits `obu_has_size_field=1` on every OBU, so we always
/// expect a ULEB128 payload size after the header bytes.
///
/// The walker is lenient on partial / truncated input: if it can't
/// parse a complete OBU, it stops there (returns the slabs gathered
/// so far). The caller's `frame::parse_frame` CRC catches any
/// downstream corruption from a truncated final slab.
#[cfg(feature = "av1-decoder")]
pub(crate) fn split_av1_into_gops(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut slabs: Vec<Vec<u8>> = Vec::new();
    let mut current: Option<Vec<u8>> = None;
    let mut cursor = 0usize;

    while cursor < bytes.len() {
        // Parse one OBU header.
        let header_byte = bytes[cursor];
        if header_byte & 0x80 != 0 {
            // obu_forbidden_bit set — bitstream malformed, stop walking.
            break;
        }
        let obu_type = (header_byte >> 3) & 0x0f;
        let has_extension = (header_byte >> 2) & 1 != 0;
        let has_size = (header_byte >> 1) & 1 != 0;
        let header_len = 1 + (has_extension as usize);
        if cursor + header_len > bytes.len() {
            break;
        }

        // Payload size: either ULEB128 (when has_size=1) or "to end
        // of buffer" (when 0). rav1e always emits has_size=1; we treat
        // has_size=0 as "consume remainder", which only fires at end-
        // of-stream when something upstream truncated.
        let (payload_len, size_field_len) = if has_size {
            let after_hdr = cursor + header_len;
            match decode_uleb128(bytes, after_hdr) {
                Some(v) => v,
                None => break,
            }
        } else {
            ((bytes.len() - cursor - header_len) as u64, 0)
        };
        let total_obu_len = header_len + size_field_len + payload_len as usize;
        if cursor + total_obu_len > bytes.len() {
            break;
        }

        // OBU_SEQUENCE_HEADER starts a new GOP slab. Anything before
        // the first one is discarded.
        if obu_type == OBU_SEQUENCE_HEADER {
            if let Some(slab) = current.take() {
                slabs.push(slab);
            }
            current = Some(Vec::new());
        }
        if let Some(ref mut buf) = current {
            buf.extend_from_slice(&bytes[cursor..cursor + total_obu_len]);
        }
        cursor += total_obu_len;
    }

    if let Some(slab) = current {
        slabs.push(slab);
    }
    slabs
}

/// Extract a hidden encrypted message from stego AV1 bytes.
///
/// Inverse of [`av1_stego_embed`]. Decodes the AV1 stream with the
/// recording hook, filters to AC_COEFF_SIGN positions, runs STC extract
/// with brute-force `w` candidates, and decrypts the first valid frame.
#[cfg(feature = "av1-decoder")]
pub fn av1_stego_extract(
    stego_av1_bytes: &[u8],
    passphrase: &str,
) -> Result<Vec<u8>, Av1StegoError> {
    let structural_key = crypto::derive_structural_key(passphrase)?;
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();
    let cover_bits = harvest_cover_bits_from_stego(stego_av1_bytes)?;
    let mut seen_decrypt_fail = false;
    let extracted = extract_first_valid_w(
        &cover_bits,
        &hhat_seed,
        |frame_bytes| match try_parse_and_decrypt(frame_bytes, passphrase) {
            Ok(plaintext) => Some(plaintext),
            Err(StegoError::DecryptionFailed) => {
                seen_decrypt_fail = true;
                None
            }
            Err(_) => None,
        },
    );
    match extracted {
        Some(plaintext) => Ok(plaintext),
        None if seen_decrypt_fail => Err(Av1StegoError::Stego(StegoError::DecryptionFailed)),
        None => Err(Av1StegoError::ExtractionFailed),
    }
}

/// Decode stego AV1 bytes via dav1d's recording hook and harvest the
/// Tier-1 cover bits (AC_COEFF_SIGN ∪ GOLOMB_TAIL_LSB) in walker order.
#[cfg(feature = "av1-decoder")]
pub fn harvest_cover_bits_from_stego(
    stego_av1_bytes: &[u8],
) -> Result<Vec<u8>, Av1StegoError> {
    use core_dav1d_sys::{
        DAV1D_PHASM_TAG_AC_COEFF_SIGN, DAV1D_PHASM_TAG_GOLOMB_TAIL_LSB,
    };
    let decoded = decode_with_recording(stego_av1_bytes)?;
    let bits: Vec<u8> = decoded
        .iter()
        .filter(|p| {
            p.tag == DAV1D_PHASM_TAG_AC_COEFF_SIGN
                || p.tag == DAV1D_PHASM_TAG_GOLOMB_TAIL_LSB
        })
        .map(|p| p.decoded_value)
        .collect();
    if bits.is_empty() {
        return Err(Av1StegoError::NoCoverPositions);
    }
    Ok(bits)
}

/// Streaming decode (v3 wire format): extract the **first-chunk** v3
/// frame from `cover_bits`. Returns `(total_bytes, payload)` on
/// success.
///
/// v3.1 semantics (#888, AV1 step 6):
/// - 80-bit prefix `(u32 total_bytes + u32 m_total + u16 payload_len)`.
/// - Exact `stored_m_total == iter_m_total` reject (1/2³² survival rate)
///   replaces v3's canonicality + LEN_SENTINEL-bypass branching and
///   collapses surviving candidates to ~1 full extract per slab.
/// - Total_bytes sanity: ∈ (0, max_total_bytes].
/// - Full extract + `parse_first_chunk_frame_v3_1` on prefix match.
///
/// Iterates `m_total += 8` (not `w in 1..=max_w` as v3 did), so the
/// loop visits the encoder's chosen `m_total` value directly — required
/// for the exact reject to land. Mirrors the H.264 v3.1 implementation
/// in `core/src/codec/h264/streaming_session.rs::extract_first_chunk_frame_match`.
///
/// See `docs/design/video/h264/chunk-frame-v3.1-decode.md` for the wire
/// format and survivor-cost analysis.
#[cfg(feature = "av1-decoder")]
pub(crate) fn extract_first_chunk_frame_match(
    cover_bits: &[u8],
    hhat_seed: &[u8; 32],
    max_total_bytes: u32,
) -> Option<(u32, Vec<u8>)> {
    use crate::stego::chunk_frame::{
        parse_first_chunk_frame_v3_1, CHUNK_FRAME_FIRST_HEADER_LEN_V3_1,
    };
    use crate::stego::stc::extract::stc_extract_prefix;

    let n_cover = cover_bits.len();
    let min_m = CHUNK_FRAME_FIRST_HEADER_LEN_V3_1 * 8; // 80 bits
    if n_cover < min_m {
        return None;
    }
    let mut m_total = min_m;
    while m_total <= n_cover {
        let w = n_cover / m_total;
        if w == 0 {
            break;
        }
        let used = m_total * w;
        let hhat_matrix = hhat::generate_hhat(STC_H, w, hhat_seed);

        // 80-bit prefix: u32 total_bytes + u32 m_total + u16 payload_len.
        let prefix_bits = stc_extract_prefix(&cover_bits[..used], &hhat_matrix, w, 80);
        if prefix_bits.len() < 80 {
            m_total += 8;
            continue;
        }
        let pb = frame::bits_to_bytes(&prefix_bits);
        if pb.len() < 10 {
            m_total += 8;
            continue;
        }
        // Exact m_total checksum: only the encoder's candidate satisfies it
        // (1/2³² random-match rate; supersedes v3's canonicality + sentinel
        // bypass).
        let stored_m_total = u32::from_be_bytes([pb[4], pb[5], pb[6], pb[7]]) as usize;
        if stored_m_total != m_total {
            m_total += 8;
            continue;
        }
        let cand_total = u32::from_be_bytes([pb[0], pb[1], pb[2], pb[3]]);
        if cand_total == 0 || cand_total > max_total_bytes {
            m_total += 8;
            continue;
        }
        // Prefix passed → full extract + canonical parse.
        let extracted = stc_extract(&cover_bits[..used], &hhat_matrix, w);
        let bits = &extracted[..m_total.min(extracted.len())];
        let bytes = frame::bits_to_bytes(bits);
        if let Some((total_bytes, _m, payload)) = parse_first_chunk_frame_v3_1(&bytes) {
            return Some((total_bytes, payload.to_vec()));
        }
        m_total += 8;
    }
    None
}

/// Streaming decode (v3.1 wire format): extract a **subsequent-chunk**
/// v3.1 frame from `cover_bits`. Returns `payload` on success.
///
/// - 48-bit prefix `(u32 m_total + u16 payload_len)`. Exact m_total
///   reject (1/2³² survival) — same mechanism as first-chunk variant.
/// - `payload.len() ≤ max_remaining_bytes` upper-bound sanity filter.
/// - Full extract + `parse_chunk_frame_v3_1` on prefix match.
///
/// `max_remaining_bytes` is `total_bytes - accumulated_bytes` from the
/// outer decode loop. Sanity ceiling so the decoder never reads past
/// the encoder's intent.
///
/// Iterates `m_total += 8` (mirrors H.264 v3.1 implementation).
/// See `docs/design/video/h264/chunk-frame-v3.1-decode.md` (#888).
#[cfg(feature = "av1-decoder")]
pub(crate) fn extract_chunk_frame_match(
    cover_bits: &[u8],
    hhat_seed: &[u8; 32],
    max_remaining_bytes: usize,
) -> Option<Vec<u8>> {
    use crate::stego::chunk_frame::{
        parse_chunk_frame_v3_1, CHUNK_FRAME_NEXT_HEADER_LEN_V3_1,
    };
    use crate::stego::stc::extract::stc_extract_prefix;

    let n_cover = cover_bits.len();
    let min_m = CHUNK_FRAME_NEXT_HEADER_LEN_V3_1 * 8; // 48 bits
    if n_cover < min_m {
        return None;
    }
    let mut m_total = min_m;
    while m_total <= n_cover {
        let w = n_cover / m_total;
        if w == 0 {
            break;
        }
        let used = m_total * w;
        let hhat_matrix = hhat::generate_hhat(STC_H, w, hhat_seed);

        // 48-bit prefix: u32 m_total + u16 payload_len. Exact m_total reject.
        let prefix_bits = stc_extract_prefix(&cover_bits[..used], &hhat_matrix, w, 48);
        if prefix_bits.len() < 48 {
            m_total += 8;
            continue;
        }
        let pb = frame::bits_to_bytes(&prefix_bits);
        if pb.len() < 6 {
            m_total += 8;
            continue;
        }
        let stored_m_total = u32::from_be_bytes([pb[0], pb[1], pb[2], pb[3]]) as usize;
        if stored_m_total != m_total {
            m_total += 8;
            continue;
        }
        let extracted = stc_extract(&cover_bits[..used], &hhat_matrix, w);
        let bits = &extracted[..m_total.min(extracted.len())];
        let bytes = frame::bits_to_bytes(bits);
        if let Some((_m, payload)) = parse_chunk_frame_v3_1(&bytes) {
            if payload.len() <= max_remaining_bytes {
                return Some(payload.to_vec());
            }
        }
        m_total += 8;
    }
    None
}

/// Brute-force the STC `w` parameter against `cover_bits`, calling
/// `validator(frame_bytes)` for each candidate. Returns the first
/// `Some(T)` the validator yields, or `None` after exhausting the
/// search range.
///
/// Mirrors phasm-core's Ghost decode pattern (`pipeline.rs:1108`).
/// The validator is what makes this generic — for legacy decode it
/// runs `try_parse_and_decrypt`; for streaming decode it runs
/// `parse_chunk_frame_v3_1` + expected_chunk_idx match. Each STC extract
/// is cheap; the validator gates wrong-w results from passing
/// through.
#[cfg(feature = "av1-decoder")]
pub(crate) fn extract_first_valid_w<T, F>(
    cover_bits: &[u8],
    hhat_seed: &[u8; 32],
    mut validator: F,
) -> Option<T>
where
    F: FnMut(&[u8]) -> Option<T>,
{
    let n = cover_bits.len();
    // Encoder picks `w = floor(n / m_bits)`. The smallest m_bits any
    // primary message can produce is `FRAME_OVERHEAD * 8 = 400 bits`
    // (an empty plaintext still incurs 2-byte length + 16-byte salt +
    // 12-byte nonce + 16-byte AES-GCM tag + 4-byte CRC = 50 bytes =
    // 400 bits). So `w_max = n / 400` covers every valid encoder
    // choice. At 256×144 this is ~30; at 1080p it climbs to ~2500.
    //
    // Previously this was capped at `min(n, 1000)`, which broke
    // 1080p decode because encoders at ~1M-bit covers with short
    // messages pick w in [1500, 2500] (well above the old 1000 cap).
    // All AV1 corpus tests passed because they were small-resolution
    // (256×144 / 144×256), so the cap was never exercised on
    // production-sized covers.
    let max_w = n / (crate::stego::frame::FRAME_OVERHEAD * 8);
    let max_w = max_w.max(1);
    for w in 1..=max_w {
        if w > n {
            break;
        }
        let m_max = n / w;
        if m_max == 0 {
            continue;
        }
        let n_used = m_max * w;
        let stego_bits = &cover_bits[..n_used];
        let hhat_matrix = hhat::generate_hhat(STC_H, w, hhat_seed);
        let extracted_bits = stc_extract(stego_bits, &hhat_matrix, w);
        let frame_bytes = frame::bits_to_bytes(&extracted_bits[..m_max]);
        if let Some(out) = validator(&frame_bytes) {
            return Some(out);
        }
    }
    None
}

/// Parse a payload frame + decrypt with the passphrase. Returns
/// plaintext on success. Mirror of Ghost's `try_parse_and_decrypt`
/// at `core/src/stego/ghost/pipeline.rs:1204`.
#[cfg(feature = "av1-decoder")]
fn try_parse_and_decrypt(
    frame_bytes: &[u8],
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    let parsed = frame::parse_frame(frame_bytes)?;
    let plaintext = crypto::decrypt(
        &parsed.ciphertext,
        passphrase,
        &parsed.salt,
        &parsed.nonce,
    )?;
    let len = parsed.plaintext_len as usize;
    if len > plaintext.len() {
        return Err(StegoError::FrameCorrupted);
    }
    Ok(plaintext[..len].to_vec())
}

#[cfg(all(test, feature = "av1-encoder", feature = "av1-decoder"))]
mod tests {
    use super::*;
    use phasm_rav1e::color::ChromaSampling;
    use phasm_rav1e::phasm_stego::{
        encode_frame_with_phasm_tee, make_frame, make_inter_config,
        FrameInvariants, FrameState,
    };
    use phasm_rav1e::prelude::Sequence;
    use phasm_rav1e::EncoderConfig;
    use std::sync::Arc;

    /// Build the same 128×128 gradient frame state used in the
    /// strict cursor parity test. Returns (natural_packet, recording)
    /// from encode_frame_with_phasm_tee.
    fn build_natural_encode() -> (Vec<u8>, PhasmFrameRecording) {
        let config = Arc::new(EncoderConfig {
            width: 128,
            height: 128,
            bit_depth: 8,
            chroma_sampling: ChromaSampling::Cs420,
            quantizer: 30,
            ..Default::default()
        });
        let mut sequence = Sequence::new(&config);
        sequence.enable_large_lru = false;
        let mut fi = FrameInvariants::<u8>::new_key_frame(
            config.clone(),
            Arc::new(sequence),
            0,
            Box::new([]),
        );
        // Disable segmentation to avoid kmeans underflow on
        // synthetic gradient (see encode_frame_with_phasm_tee
        // smoke test in phasm-rav1e).
        fi.enable_segmentation = false;
        let mut frame = make_frame::<u8>(128, 128, ChromaSampling::Cs420);
        // CRITICAL: use `Plane::copy_from_raw_u8` to write into the
        // visible region. Manual chunks_mut iterates raw plane.data
        // bytes including filter-tap padding rows, so synthetic
        // content lands in padding while visible stays default-init
        // neutral gray. See feedback_visual_fidelity_is_correctness.
        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let (w, h) = (plane.cfg.width, plane.cfg.height);
            let mut buf = vec![0u8; w * h];
            for row_idx in 0..h {
                for col_idx in 0..w {
                    buf[row_idx * w + col_idx] = ((row_idx.wrapping_mul(7)
                        + col_idx.wrapping_mul(3)
                        + plane_idx * 13)
                        & 0xff) as u8;
                }
            }
            plane.copy_from_raw_u8(&buf, w, 1);
        }
        let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame));
        let inter_cfg = make_inter_config(&config);
        encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg)
    }

    /// End-to-end round-trip test — the initial AV1 ship gate.
    ///
    /// Encode + decode a short encrypted message end-to-end:
    ///   1. Build natural encode (Pass 1 via WriterTee)
    ///   2. av1_stego_embed("hi from av1 stego", "secret pass") → stego AV1
    ///   3. av1_stego_extract(stego AV1, "secret pass") → "hi from av1 stego"
    ///
    /// If this passes, the entire writer-trait architecture is
    /// functional end-to-end — a real message survives the full
    /// encrypt → STC plan → replay → byte-splice → decode →
    /// STC extract → decrypt round-trip.
    #[test]
    fn w3105_round_trip_real_message() {
        let plaintext = b"hi from av1 stego";
        let passphrase = "v0.3-secret-pass";

        // Pass 1: natural encode.
        let (natural_packet, recording) = build_natural_encode();
        assert!(!natural_packet.is_empty());
        assert_eq!(recording.tiles.len(), 1);

        // Embed.
        let stego_packet = av1_stego_embed(
            natural_packet.clone(),
            recording.clone(),
            plaintext,
            passphrase,
        )
        .expect("av1_stego_embed should succeed");
        assert_eq!(
            stego_packet.len(),
            natural_packet.len(),
            "stego packet must have same length as natural"
        );

        // Extract.
        let extracted = av1_stego_extract(&stego_packet, passphrase)
            .expect("av1_stego_extract should succeed");
        assert_eq!(
            extracted.as_slice(),
            plaintext,
            "extracted plaintext must match original"
        );
    }

    /// Wrong passphrase should fail decryption (not return garbage).
    #[test]
    fn w3105_wrong_passphrase_fails() {
        let plaintext = b"secret data";
        let passphrase = "correct-pass";
        let wrong_passphrase = "wrong-pass";

        let (natural_packet, recording) = build_natural_encode();
        let stego_packet = av1_stego_embed(
            natural_packet,
            recording,
            plaintext,
            passphrase,
        )
        .expect("embed");

        let result = av1_stego_extract(&stego_packet, wrong_passphrase);
        assert!(
            result.is_err(),
            "extract with wrong passphrase should error; got Ok({:?})",
            result.as_ref().ok()
        );
    }

    /// Stego packet remains a valid AV1 bitstream (dav1d decodes
    /// without error).
    #[test]
    fn w3105_stego_bytes_decode_as_valid_av1() {
        let plaintext = b"valid av1 check";
        let passphrase = "test-pass";

        let (natural_packet, recording) = build_natural_encode();
        let stego_packet = av1_stego_embed(
            natural_packet,
            recording,
            plaintext,
            passphrase,
        )
        .expect("embed");

        // dav1d decode should succeed (no syntax errors from the
        // stego modifications).
        let decode_result = decode_with_recording(&stego_packet);
        assert!(
            decode_result.is_ok(),
            "stego bytes must decode as valid AV1; got {:?}",
            decode_result.as_ref().err()
        );
    }
}
