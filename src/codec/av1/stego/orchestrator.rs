// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! W3.10.5 — production av1_stego_encode / av1_stego_decode flow.
//!
//! Ties together:
//!   - phasm-rav1e [`encode_frame_with_phasm_tee`] (Pass 1: bytes + recorder)
//!   - phasm-core [`OverrideMap`] + [`replay_with_overrides`] (Pass 2)
//!   - phasm-core [`stc_embed`] + [`stc_extract`] (cover-bit → message)
//!   - phasm-core [`crypto`] (Argon2id + AES-GCM-SIV)
//!   - phasm-core [`frame`] (length-prefixed CRC-protected payload framing)
//!   - phasm-dav1d [`decode_with_recording`] (decoder side)
//!
//! Flow per `streaming-session.md` § 1 Option E:
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
//!    (length-invariant per W3.8.6)
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
//! v0.3-AV1 scope: single-frame, single-tile, AC_COEFF_SIGN channel
//! only (per channel-design.md § 6). Strict cursor parity (W3.10.4-fix)
//! is the load-bearing equivalence — without it, decoder would extract
//! at different positions than encoder hid at.

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

#[cfg(feature = "av1-backend")]
use crate::codec::av1::stego::decoder::{decode_with_recording, Av1DecodeError};
#[cfg(feature = "av1-encoder")]
use crate::codec::av1::stego::writer::{replay_with_overrides, OverrideMap};

use crate::stego::error::StegoError;
#[cfg(feature = "av1-encoder")]
use crate::stego::stc::embed::stc_embed;
#[cfg(feature = "av1-backend")]
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
    #[cfg(feature = "av1-backend")]
    Decode(Av1DecodeError),
    /// Encoder-side recording has no tiles (unexpected for any valid encode).
    EmptyRecording,
    /// Not enough AC_COEFF_SIGN cover positions for the message size.
    MessageTooLarge { needed_bits: usize, available_bits: usize },
    /// STC embedding returned None (cover too small / w bound violated).
    StcInfeasible,
    /// `replay_with_overrides` produced a tile_group of a different size
    /// than the natural encode. Indicates a Tier 1 invariant violation
    /// (50/50 flips should preserve length per W3.8.6).
    TileGroupSizeMismatch { expected: usize, actual: usize },
    /// Decoder side: no AC_COEFF_SIGN positions found in the bitstream.
    #[cfg(feature = "av1-backend")]
    NoCoverPositions,
    /// All brute-force `w` candidates failed to extract a valid frame.
    #[cfg(feature = "av1-backend")]
    ExtractionFailed,
}

impl From<StegoError> for Av1StegoError {
    fn from(e: StegoError) -> Self {
        Av1StegoError::Stego(e)
    }
}

#[cfg(feature = "av1-backend")]
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
/// # v0.3 scope
/// Single-tile, single-frame, AC_COEFF_SIGN-only.
#[cfg(feature = "av1-encoder")]
pub fn av1_stego_embed(
    natural_packet: Vec<u8>,
    recording: PhasmFrameRecording,
    message: &[u8],
    passphrase: &str,
) -> Result<Vec<u8>, Av1StegoError> {
    if recording.tiles.is_empty() {
        return Err(Av1StegoError::EmptyRecording);
    }

    // 1. Derive keys from passphrase. structural_key[0..32] is the
    // permutation seed (unused here for v0.3 — single-tile, no
    // shadow), structural_key[32..64] is the HHAT seed.
    let structural_key = crypto::derive_structural_key(passphrase)?;
    let hhat_seed: [u8; 32] = structural_key[32..]
        .try_into()
        .expect("derive_structural_key returns 64 bytes");

    // 2. Encrypt message → (ciphertext, nonce, salt).
    let (ciphertext, nonce, salt) = crypto::encrypt(message, passphrase)?;

    // 3. Build payload frame (len + salt + nonce + ciphertext + CRC).
    // Then unpack to bits (one byte per bit, 0 or 1) — STC's
    // message argument is a bit array per Ghost convention
    // (pipeline.rs:1066 `frame_bits = frame::bytes_to_bits`).
    let payload_bytes = frame::build_frame(message.len(), &salt, &nonce, &ciphertext);
    let payload = frame::bytes_to_bits(&payload_bytes);

    // 4. Filter recorder's bit_positions to the Tier-1 channel set.
    //
    // Phase B.2.3.proper (2026-05-21) enrolls GOLOMB_TAIL_LSB
    // alongside AC_COEFF_SIGN. The phasm-rav1e c63b7c94 + phasm-dav1d
    // 3fb3cc3c fork patches tag ONLY the literal LSB of each golomb
    // code (leading zeros + non-LSB literals get OTHER), so every
    // enrolled position is safe to flip ±1 without changing the
    // bitstream length or cascading through culLevel.
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
    // pipeline.rs:74 `compute_stc_params`): w = floor(n / m_bits),
    // n_used = m_bits * w. Each message bit gets `w` cover bits of
    // slack. Truncate cover to n_used so STC's Viterbi is balanced.
    let w = (n / m_bits.max(1)).max(1);
    let n_used = m_bits * w;
    let cover_used = &ac_cover_bits[..n_used];
    let cursors_used = &ac_global_cursors[..n_used];
    let metas_used = &ac_metas[..n_used];

    // 6. Generate HHAT matrix from passphrase-derived seed.
    let hhat_matrix = hhat::generate_hhat(STC_H, w, &hhat_seed);

    // 7. STC embed with J-UNIWARD costs (Phase B.1.2). Replaces the
    // v0.3 uniform `vec![1.0; n_used]` placeholder. Each AC sign's
    // cost reflects how much its sign flip would perturb the
    // Daubechies-8 wavelet decomposition of the post-LR
    // reconstructed pixel plane the flip lives in. Positions with
    // high-magnitude AC coefficients in textured regions get LOW
    // cost (safe to flip — the perturbation hides in existing
    // wavelet energy); positions in smooth regions get HIGH cost.
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
            // B.1.5.0.5: carry encoder-side coefficient magnitude
            // through for cascade-safety v2's EE-D pre-filter +
            // EE-C upper-bound + L3 cache key.
            coeff_magnitude: m.coeff_magnitude,
        })
        .collect();
    // B.1.5.5: pass the frame-level loop-filter state captured by
    // B.1.5.1's fork-patch through to cost compute. This activates
    // the three-tier dispatch (EE-D |coeff|-based safe/reject bands +
    // L3 forward model for the middle), replacing the v0.5 magnitude-
    // proxy path that wasn't discriminating cascade-amplified
    // positions.
    let costs = crate::stego::cost::av1_uniward::compute_av1_uniward_costs_with_state(
        &frame_planes,
        &av1_positions,
        recording.frame_qindex,
        Some(recording.loop_filter_state),
    );
    let embed_result = stc_embed(cover_used, &costs, &payload, &hhat_matrix, STC_H, w)
        .ok_or(Av1StegoError::StcInfeasible)?;

    // 8. Build OverrideMap from cover→stego differences. Only
    // iterates over n_used positions (the rest don't get planned).
    let mut plan = OverrideMap::new();
    for (i, &stego_bit) in embed_result.stego_bits.iter().enumerate() {
        if stego_bit != cover_used[i] {
            plan.set(cursors_used[i], stego_bit as u16);
        }
    }

    // 9. Replay through a fresh WriterEncoder with the override plan.
    // Produces the stego tile_group bytes.
    let mut sink = WriterEncoder::new();
    replay_with_overrides(&tile.storage, &tile.bit_positions, &plan, &mut sink);
    let stego_tile_bytes = sink.done();

    // 10. Splice stego tile bytes into the natural OBU output.
    // Fast path: same length → in-place byte-overwrite.
    // Slow path (v0.4 fix): length differs by ±1 byte (rare range-
    // coder trailing-carry edge case) → rebuild the frame_obu with
    // a recomputed ULEB128 size field.
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

/// Rebuild the frame_obu portion of the natural packet with the new
/// stego tile_group bytes, recomputing the ULEB128 size field so the
/// AV1 container stays consistent.
///
/// v0.4 fix for the TileGroupSizeMismatch case: when stego flips
/// happen near the end of the entropy-coded stream, the range
/// coder's trailing-carry flush can produce a tile_group that's
/// ±1 byte different in length from the natural one. Previously
/// this returned `Av1StegoError::TileGroupSizeMismatch`; now the
/// orchestrator rebuilds the OBU header's size field and stitches
/// the new tile_group in cleanly.
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
fn rebuild_obu_with_stego_tile_group(
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

/// Phase B.1.2: extract the visible-region pixel data from each
/// plane of a reconstructed Frame and pack into [`FramePlanes`]
/// (contiguous, no stride padding). The Frame buffer stores each
/// plane with filter-tap padding at `xorigin`/`yorigin` offsets;
/// our J-UNIWARD wavelet decomposition consumes packed YUV.
///
/// Memory cost: one Vec<u8> per plane = ~3 MB for 1080p. Avoided
/// in v0.6+ via a wavelet decomp that consumes strided input
/// directly (small refactor of `compute_three_subbands`).
#[cfg(feature = "av1-encoder")]
fn pack_visible_planes(
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

/// Extract a hidden encrypted message from stego AV1 bytes.
///
/// Inverse of [`av1_stego_embed`]. Decodes the AV1 stream with the
/// recording hook, filters to AC_COEFF_SIGN positions, runs STC extract
/// with brute-force `w` candidates, and decrypts the first valid frame.
#[cfg(feature = "av1-backend")]
pub fn av1_stego_extract(
    stego_av1_bytes: &[u8],
    passphrase: &str,
) -> Result<Vec<u8>, Av1StegoError> {
    // 1. Derive HHAT seed.
    let structural_key = crypto::derive_structural_key(passphrase)?;
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    // 2. Decode with recording.
    let decoded = decode_with_recording(stego_av1_bytes)?;

    // 3. Filter to the Tier-1 channel set. Mirror of the embed-side
    // joint enrollment (AC_COEFF_SIGN + GOLOMB_TAIL_LSB).
    use core_dav1d_sys::{
        DAV1D_PHASM_TAG_AC_COEFF_SIGN, DAV1D_PHASM_TAG_GOLOMB_TAIL_LSB,
    };
    let stego_ac_bits: Vec<u8> = decoded
        .iter()
        .filter(|p| {
            p.tag == DAV1D_PHASM_TAG_AC_COEFF_SIGN
                || p.tag == DAV1D_PHASM_TAG_GOLOMB_TAIL_LSB
        })
        .map(|p| p.decoded_value)
        .collect();

    let n = stego_ac_bits.len();
    if n == 0 {
        return Err(Av1StegoError::NoCoverPositions);
    }

    // 4. Brute-force w candidates. Mirrors phasm-core's Ghost decode
    // pattern (pipeline.rs:1108). The encoder picks w =
    // floor(n / m_bits); decoder doesn't know m, so tries every
    // value in the plausible range. Each STC extract is cheap; CRC
    // gates a wrong w from passing through.
    //
    // Cover capacity at common dims (256×144 q30 .. 1080p q30) yields
    // w in [10..400] for typical short messages. Cap at 1000 to
    // accommodate larger covers / smaller messages.
    let max_w = n.min(1000);
    let w_candidates: Vec<usize> = (1..=max_w).collect();
    let mut seen_decrypt_fail = false;
    for &w in &w_candidates {
        if w == 0 || w > n {
            continue;
        }
        let m_max = n / w;
        if m_max == 0 {
            continue;
        }
        let n_used = m_max * w;
        let stego_bits = &stego_ac_bits[..n_used];
        let hhat_matrix = hhat::generate_hhat(STC_H, w, &hhat_seed);
        let extracted_bits = stc_extract(stego_bits, &hhat_matrix, w);

        let frame_bytes = frame::bits_to_bytes(&extracted_bits[..m_max]);
        match try_parse_and_decrypt(&frame_bytes, passphrase) {
            Ok(plaintext) => return Ok(plaintext),
            Err(StegoError::DecryptionFailed) => seen_decrypt_fail = true,
            Err(_) => {}
        }
    }
    if seen_decrypt_fail {
        Err(Av1StegoError::Stego(StegoError::DecryptionFailed))
    } else {
        Err(Av1StegoError::ExtractionFailed)
    }
}

/// Parse a payload frame + decrypt with the passphrase. Returns
/// plaintext on success. Mirror of Ghost's `try_parse_and_decrypt`
/// at `core/src/stego/ghost/pipeline.rs:1204`.
#[cfg(feature = "av1-backend")]
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

#[cfg(all(test, feature = "av1-encoder", feature = "av1-backend"))]
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

    /// Build the same 128×128 gradient frame state used in W3.10.4
    /// strict parity test. Returns (natural_packet, recording) from
    /// encode_frame_with_phasm_tee.
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

    /// W3.10.5 round-trip test — the v0.3-AV1 ship gate.
    ///
    /// Encode + decode a short encrypted message end-to-end:
    ///   1. Build natural encode (Pass 1 via WriterTee)
    ///   2. av1_stego_embed("hi from av1 stego", "secret pass") → stego AV1
    ///   3. av1_stego_extract(stego AV1, "secret pass") → "hi from av1 stego"
    ///
    /// If this passes, the entire W3 architecture is functional
    /// end-to-end. First time a real message survives the full
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
