// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Shadow messages: Y-channel direct LSB embedding + RS ECC (headerless brute-force).
//!
//! Shadow messages provide plausible deniability for Ghost mode steganography.
//! Multiple messages can be hidden in a single image, each with a different
//! passphrase. They are embedded as absolute-value LSBs into Y-channel nzAC
//! positions (the same domain as primary STC), using nsf5 modification.
//!
//! The system auto-sorts messages by size: the largest becomes the primary
//! (embedded via STC for full stealth), smaller messages become shadow channels
//! (embedded via direct LSB with Reed-Solomon error correction).
//!
//! ## Headerless design
//!
//! No magic byte, no frame_data_len in the bitstream. The decoder brute-forces
//! all (parity, fdl) combinations. AES-256-GCM-SIV authentication is the only
//! validator — successful decryption proves correct parameters.
//!
//! ## Short STC + Dynamic w
//!
//! Since shadows and primary STC share the same Y-channel LSBs, the primary
//! STC uses "short" message mode: only the actual `m` message bits are passed
//! (not zero-padded to `m_max`). With dynamic `w = min(floor(n/m), 10)`, small
//! messages get high w, meaning 2,500x fewer modifications. When w >= 2, shadow
//! positions get `f32::INFINITY` cost in STC, so Viterbi routes around them,
//! achieving BER ~ 0% on shadows.
//!
//! ## Cost-pool position selection
//!
//! Shadow positions are selected in two tiers for stealth:
//! 1. **Tier 1 (cost pool):** Filter all Y nzAC positions to the cheapest
//!    fraction (5%, 10%, 20%, 50%, or 100%) by UNIWARD cost. Cheap positions
//!    are in textured regions — modifications there are least detectable.
//! 2. **Tier 2 (hash permutation):** Within the cost pool, select positions
//!    by keyed hash `ChaCha20(seed, flat_idx)` priority.
//!
//! Encoder uses cover-image costs; decoder uses stego-image costs. The cost
//! pools differ slightly at the boundary (~2-5% positions), but RS error
//! correction handles the resulting BER. Primary Ghost proves this works:
//! encoder and decoder already use cover vs stego costs for STC positions.
//! The decoder brute-forces the fraction alongside parity and fdl.
//!
//! ## Frame format (inside RS-encoded data, no header)
//!
//! ```text
//! RS-encoded frame (all positions):
//!     Inner: [plaintext_len: 2B] [salt: 16B] [nonce: 12B] [ciphertext: N+16B]
//! ```
//!
//! ## nzAC invariance
//!
//! The nsf5 anti-shrinkage rule (`|coeff|==1 -> away from zero`) ensures no
//! coefficient ever becomes zero. Since both shadow embedding and primary STC
//! use nsf5, the Y nzAC set is identical at encoder and decoder, guaranteeing
//! position agreement.

use crate::jpeg::JpegImage;
use crate::stego::armor::ecc;
use crate::stego::crypto::{self, NONCE_LEN, SALT_LEN};
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload::{self, FileEntry, PayloadData};
use crate::stego::permute::CoeffPos;
use crate::stego::pipeline::{flat_get, flat_set};
use crate::stego::side_info::nsf5_modify_coefficient;

/// Frame overhead inside the RS-encoded payload:
/// plaintext_len(2) + salt(16) + nonce(12) + tag(16) = 46 bytes.
const SHADOW_FRAME_OVERHEAD: usize = 2 + SALT_LEN + NONCE_LEN + 16;

/// RS parity tiers. Brute-forced at decode.
const SHADOW_PARITY_TIERS: [usize; 6] = [4, 8, 16, 32, 64, 128];

/// Cost pool fractions (denominators). Encoder picks smallest that fits.
/// 20 = cheapest 5%, 10 = 10%, 5 = 20%, 2 = 50%, 1 = 100%.
/// Smaller fractions give better stealth (positions in most textured regions).
/// Decoder brute-forces all fractions.
const COST_FRACTIONS: [usize; 5] = [20, 10, 5, 2, 1];

/// Maximum RS-encoded frame bytes to prevent unreasonable allocations.
const MAX_SHADOW_FRAME_BYTES: usize = 256 * 1024;

/// Maximum fdl scan range during brute-force decode (bytes above FRAME_OVERHEAD).
/// Covers shadow messages up to ~2KB plaintext. Keeps decode time bounded.
const MAX_FDL_SCAN: usize = 2048;

/// State for a single shadow layer during encoding.
#[derive(Clone)]
pub struct ShadowState {
    /// Cost-filtered + hash-permuted embedding positions (Y channel nzAC).
    pub positions: Vec<CoeffPos>,
    /// The desired LSB bits (RS-encoded frame, no header).
    pub bits: Vec<u8>,
    /// Total bits = RS-encoded data.
    pub n_total: usize,
    /// Current RS parity length.
    pub parity_len: usize,
    /// Unencoded frame byte count (before RS).
    pub frame_data_len: usize,
    /// Raw frame bytes (before RS), needed for parity bump rebuild.
    pub frame_bytes: Vec<u8>,
    /// Cached shadow structural key (ChaCha20 seed for position permutation).
    pub perm_seed: [u8; 32],
    /// Cost pool fraction denominator (20=5%, 10=10%, 5=20%, 2=50%, 1=100%).
    pub cost_fraction: usize,
}

/// Prepare a shadow layer for embedding.
///
/// Builds the payload, encrypts, frames, RS-encodes, then selects positions
/// using cost-pool filtering (Tier 1) + hash permutation (Tier 2).
/// Tries fractions from smallest (best stealth) to largest (most capacity).
///
/// `all_y_positions_sorted` must be sorted by cost (cheapest first).
pub fn prepare_shadow(
    all_y_positions_sorted: &[CoeffPos],
    shadow_pass: &str,
    message: &str,
    files: &[FileEntry],
    parity_len: usize,
) -> Result<ShadowState, StegoError> {
    // 1. Build payload (text + files + compression).
    let payload_bytes = payload::encode_payload(message, files)?;

    // 2. Encrypt payload.
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, shadow_pass)?;

    // 3. Build inner frame (no header, no CRC — RS + GCM provide integrity).
    let frame_bytes = build_shadow_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
    let frame_data_len = frame_bytes.len();

    // 4. RS encode.
    let rs_bytes = ecc::rs_encode_blocks_with_parity(&frame_bytes, parity_len);
    let rs_bits = frame::bytes_to_bits(&rs_bytes);
    let n_total = rs_bits.len();

    // 5. Select positions: try fractions from smallest (best stealth) to largest.
    let perm_seed = crypto::derive_shadow_structural_key(shadow_pass)?;
    for &fraction in &COST_FRACTIONS {
        let positions = select_shadow_positions(all_y_positions_sorted, fraction, n_total, &perm_seed);
        if positions.len() >= n_total {
            return Ok(ShadowState {
                positions,
                bits: rs_bits,
                n_total,
                parity_len,
                frame_data_len,
                frame_bytes,
                perm_seed: *perm_seed,
                cost_fraction: fraction,
            });
        }
    }

    Err(StegoError::MessageTooLarge)
}

/// Rebuild a shadow state with new parity and/or fraction.
///
/// Called during encoder escalation when decoder-side verification fails.
/// Re-RS-encodes the same frame data with the new parity, selects positions
/// from the specified cost fraction.
///
/// `all_y_positions_sorted` must be sorted by cost (cheapest first).
pub fn rebuild_shadow(
    state: &mut ShadowState,
    all_y_positions_sorted: &[CoeffPos],
    new_parity: usize,
    new_fraction: usize,
) -> Result<(), StegoError> {
    let rs_bytes = ecc::rs_encode_blocks_with_parity(&state.frame_bytes, new_parity);
    let rs_bits = frame::bytes_to_bits(&rs_bytes);
    let n_total = rs_bits.len();

    let positions = select_shadow_positions(
        all_y_positions_sorted, new_fraction, n_total, &state.perm_seed,
    );
    if positions.len() < n_total {
        return Err(StegoError::MessageTooLarge);
    }

    state.positions = positions;
    state.bits = rs_bits;
    state.n_total = n_total;
    state.parity_len = new_parity;
    state.cost_fraction = new_fraction;

    Ok(())
}

/// Embed shadow intended bits as absolute-value LSBs into the Y DctGrid.
///
/// Uses nsf5 modification: toward zero for |coeff|>1, away from zero for
/// |coeff|==1 (anti-shrinkage preserves the nzAC set).
pub fn embed_shadow_lsb(img: &mut JpegImage, state: &ShadowState) {
    for (i, pos) in state.positions.iter().enumerate() {
        if i >= state.n_total {
            break;
        }
        let fi = pos.flat_idx as usize;
        let coeff = flat_get(img.dct_grid(0), fi);
        let current_lsb = (coeff.unsigned_abs() & 1) as u8;
        if current_lsb != state.bits[i] {
            let modified = nsf5_modify_coefficient(coeff);
            flat_set(img.dct_grid_mut(0), fi, modified);
        }
    }
}

/// Verify a shadow layer can be correctly decoded from the current image state.
///
/// Extracts LSBs, RS-decodes, parses frame, and decrypts.
pub fn verify_shadow(
    img: &JpegImage,
    state: &ShadowState,
    passphrase: &str,
) -> Result<(), StegoError> {
    let grid = img.dct_grid(0);

    // Extract all LSBs for this shadow's positions.
    let lsbs: Vec<u8> = state.positions[..state.n_total].iter().map(|pos| {
        let coeff = flat_get(grid, pos.flat_idx as usize);
        (coeff.unsigned_abs() & 1) as u8
    }).collect();
    let rs_bytes = frame::bits_to_bytes(&lsbs);

    // RS decode with known parameters.
    let (decoded, _stats) = ecc::rs_decode_blocks_with_parity(
        &rs_bytes, state.frame_data_len, state.parity_len,
    ).map_err(|_| StegoError::FrameCorrupted)?;

    // Parse frame and decrypt to verify integrity.
    let fr = parse_shadow_frame(&decoded)?;
    crypto::decrypt(
        &fr.ciphertext,
        passphrase,
        &fr.salt,
        &fr.nonce,
    )?;

    Ok(())
}

/// Verify a shadow can be decoded from the decoder's perspective.
///
/// Uses stego-image UNIWARD costs (not encoder's cover costs) to select
/// positions, then checks RS decode + AES-GCM. This catches cost-pool
/// boundary disagreements that the encoder-side verify misses.
///
/// `stego_y_positions_sorted` must be sorted by cost from the STEGO image.
pub fn verify_shadow_decoder_side(
    img: &JpegImage,
    state: &ShadowState,
    passphrase: &str,
    stego_y_positions_sorted: &[CoeffPos],
) -> Result<(), StegoError> {
    let positions = select_shadow_positions(
        stego_y_positions_sorted, state.cost_fraction, state.n_total, &state.perm_seed,
    );
    if positions.len() < state.n_total {
        return Err(StegoError::FrameCorrupted);
    }

    let grid = img.dct_grid(0);
    let lsbs: Vec<u8> = positions[..state.n_total].iter().map(|pos| {
        let coeff = flat_get(grid, pos.flat_idx as usize);
        (coeff.unsigned_abs() & 1) as u8
    }).collect();
    let rs_bytes = frame::bits_to_bytes(&lsbs);

    let (decoded, _stats) = ecc::rs_decode_blocks_with_parity(
        &rs_bytes, state.frame_data_len, state.parity_len,
    ).map_err(|_| StegoError::FrameCorrupted)?;

    let fr = parse_shadow_frame(&decoded)?;
    crypto::decrypt(
        &fr.ciphertext,
        passphrase,
        &fr.salt,
        &fr.nonce,
    )?;

    Ok(())
}

/// Full shadow decode pipeline (headerless brute-force).
///
/// Brute-forces (fraction, parity, fdl) combinations. For each cost fraction,
/// selects positions from the cheapest pool, then tries all parity/fdl combos.
/// AES-256-GCM-SIV authentication validates correct parameters.
///
/// `all_y_positions_sorted` must be sorted by cost (cheapest first).
pub fn shadow_extract(
    img: &JpegImage,
    all_y_positions_sorted: &[CoeffPos],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    // 1. Derive shadow structural key for position permutation.
    let perm_seed = crypto::derive_shadow_structural_key(passphrase)?;

    let grid = img.dct_grid(0);
    let mut last_err = StegoError::FrameCorrupted;

    if all_y_positions_sorted.is_empty() {
        return Err(StegoError::FrameCorrupted);
    }

    // 2. Brute-force: fraction × parity × fdl.
    // For each fraction, select pool positions and extract LSBs once.
    for &fraction in &COST_FRACTIONS {
        let pool_size = all_y_positions_sorted.len() / fraction;
        if pool_size == 0 {
            continue;
        }

        // Hash-select all pool positions (sorted by hash priority).
        let positions = select_shadow_positions(
            all_y_positions_sorted, fraction, pool_size, &perm_seed,
        );
        if positions.is_empty() {
            continue;
        }

        // Extract LSBs once for this fraction (reused across parity/fdl).
        let all_lsbs: Vec<u8> = positions.iter().map(|pos| {
            let coeff = flat_get(grid, pos.flat_idx as usize);
            (coeff.unsigned_abs() & 1) as u8
        }).collect();

        // Inner brute-force: parity × fdl.
        if let Some(result) = try_extract_with_lsbs(
            &all_lsbs, passphrase, &mut last_err,
        ) {
            return result;
        }
    }

    Err(last_err)
}

/// Try all (parity, fdl) combinations on a given LSB vector.
/// Returns `Some(Ok(...))` on success, `Some(Err(...))` on fatal error,
/// or `None` to continue to next fraction.
fn try_extract_with_lsbs(
    all_lsbs: &[u8],
    passphrase: &str,
    last_err: &mut StegoError,
) -> Option<Result<PayloadData, StegoError>> {
    for &parity_len in &SHADOW_PARITY_TIERS {
        let k = 255usize.saturating_sub(parity_len);
        if k == 0 {
            continue;
        }

        let max_rs_bytes = all_lsbs.len() / 8;
        let max_fdl = compute_max_fdl(max_rs_bytes, parity_len)
            .min(MAX_SHADOW_FRAME_BYTES);

        if SHADOW_FRAME_OVERHEAD > max_fdl {
            continue;
        }

        let scan_max = max_fdl.min(SHADOW_FRAME_OVERHEAD + MAX_FDL_SCAN);
        for fdl in SHADOW_FRAME_OVERHEAD..=scan_max {
            let rs_encoded_len = ecc::rs_encoded_len_with_parity(fdl, parity_len);
            let rs_bits_needed = rs_encoded_len * 8;

            if rs_bits_needed > all_lsbs.len() {
                break;
            }

            let rs_bytes = frame::bits_to_bytes(&all_lsbs[..rs_bits_needed]);

            let decoded = match ecc::rs_decode_blocks_with_parity(&rs_bytes, fdl, parity_len) {
                Ok((data, _stats)) => data,
                Err(_) => continue,
            };

            let fr = match parse_shadow_frame(&decoded) {
                Ok(f) => f,
                Err(e) => {
                    *last_err = e;
                    continue;
                }
            };

            let plaintext = match crypto::decrypt(
                &fr.ciphertext,
                passphrase,
                &fr.salt,
                &fr.nonce,
            ) {
                Ok(p) => p,
                Err(StegoError::DecryptionFailed) => {
                    *last_err = StegoError::DecryptionFailed;
                    continue;
                }
                Err(e) => {
                    *last_err = e;
                    continue;
                }
            };

            let len = fr.plaintext_len as usize;
            if len > plaintext.len() {
                *last_err = StegoError::FrameCorrupted;
                continue;
            }

            return Some(payload::decode_payload(&plaintext[..len]));
        }
    }

    None
}

/// Compute shadow capacity in plaintext bytes from Y nzAC count.
///
/// Uses the full nzAC pool and smallest parity tier for maximum capacity.
pub fn shadow_capacity(y_nzac: usize) -> usize {
    if y_nzac == 0 {
        return 0;
    }

    let parity_len = SHADOW_PARITY_TIERS[0]; // smallest parity for max capacity
    let available_rs_bytes = y_nzac / 8;

    let k = 255 - parity_len;
    if k == 0 || available_rs_bytes == 0 {
        return 0;
    }

    let full_blocks = available_rs_bytes / 255;
    let remainder_bytes = available_rs_bytes % 255;

    let mut max_frame_bytes = full_blocks * k;
    if remainder_bytes > parity_len {
        max_frame_bytes += remainder_bytes - parity_len;
    }

    max_frame_bytes.saturating_sub(SHADOW_FRAME_OVERHEAD)
}

// --- Internal helpers ---

/// Build the shadow inner frame (before RS encoding).
///
/// Layout: [plaintext_len: 2B] [salt: 16B] [nonce: 12B] [ciphertext: N+16B]
fn build_shadow_frame(
    plaintext_len: usize,
    salt: &[u8; SALT_LEN],
    nonce: &[u8; NONCE_LEN],
    ciphertext: &[u8],
) -> Vec<u8> {
    assert!(plaintext_len <= u16::MAX as usize, "shadow frame plaintext exceeds u16::MAX");
    let mut fr = Vec::with_capacity(SHADOW_FRAME_OVERHEAD + plaintext_len);
    fr.extend_from_slice(&(plaintext_len as u16).to_be_bytes());
    fr.extend_from_slice(salt);
    fr.extend_from_slice(nonce);
    fr.extend_from_slice(ciphertext);
    fr
}

/// Parsed shadow frame.
struct ParsedShadowFrame {
    plaintext_len: u16,
    salt: [u8; SALT_LEN],
    nonce: [u8; NONCE_LEN],
    ciphertext: Vec<u8>,
}

/// Parse a shadow inner frame (after RS decoding).
fn parse_shadow_frame(data: &[u8]) -> Result<ParsedShadowFrame, StegoError> {
    if data.len() < SHADOW_FRAME_OVERHEAD {
        return Err(StegoError::FrameCorrupted);
    }

    let plaintext_len = u16::from_be_bytes([data[0], data[1]]);
    let expected_len = SHADOW_FRAME_OVERHEAD + plaintext_len as usize;
    if data.len() < expected_len {
        return Err(StegoError::FrameCorrupted);
    }

    let mut salt = [0u8; SALT_LEN];
    salt.copy_from_slice(&data[2..2 + SALT_LEN]);

    let mut nonce = [0u8; NONCE_LEN];
    nonce.copy_from_slice(&data[2 + SALT_LEN..2 + SALT_LEN + NONCE_LEN]);

    let ciphertext = data[2 + SALT_LEN + NONCE_LEN..expected_len].to_vec();

    Ok(ParsedShadowFrame {
        plaintext_len,
        salt,
        nonce,
        ciphertext,
    })
}

/// Two-tier position selection for shadow channels.
///
/// **Tier 1 (cost pool):** Takes the cheapest `1/fraction` of positions from
/// `cost_sorted_positions` (which must be sorted by cost, cheapest first).
/// This ensures shadow modifications land in textured/cheap regions.
///
/// **Tier 2 (hash permutation):** Within the cost pool, assigns each position
/// a deterministic priority from `ChaCha20(seed, flat_idx)`. Sorts by
/// priority, takes first `n_total`.
///
/// Encoder and decoder may disagree on a few positions near the cost-pool
/// boundary (cover vs stego costs differ slightly). RS handles the BER.
fn select_shadow_positions(
    cost_sorted_positions: &[CoeffPos],
    fraction: usize,
    n_total: usize,
    seed: &[u8; 32],
) -> Vec<CoeffPos> {
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    // Tier 1: cost pool — cheapest 1/fraction of all positions.
    let pool_size = cost_sorted_positions.len() / fraction;
    if pool_size == 0 {
        return Vec::new();
    }
    let pool = &cost_sorted_positions[..pool_size];

    // Tier 2: hash permutation within the cost pool.
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let mut candidates: Vec<(u64, CoeffPos)> = pool.iter().map(|p| {
        rng.set_word_pos(p.flat_idx as u128 * 2);
        let priority = rng.next_u64();
        (priority, p.clone())
    }).collect();
    candidates.sort_by_key(|(priority, _)| *priority);

    candidates.into_iter().map(|(_, p)| p).take(n_total).collect()
}

/// Compute the maximum frame_data_len that fits in `max_rs_bytes` with given parity.
fn compute_max_fdl(max_rs_bytes: usize, parity_len: usize) -> usize {
    let k = 255usize.saturating_sub(parity_len);
    if k == 0 || max_rs_bytes == 0 {
        return 0;
    }
    let full_blocks = max_rs_bytes / 255;
    let remainder = max_rs_bytes % 255;
    let mut max_data = full_blocks * k;
    if remainder > parity_len {
        max_data += remainder - parity_len;
    }
    max_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_frame_roundtrip() {
        let salt = [1u8; SALT_LEN];
        let nonce = [2u8; NONCE_LEN];
        let ciphertext = vec![0xAA; 20]; // 4 bytes plaintext + 16 tag
        let fr = build_shadow_frame(4, &salt, &nonce, &ciphertext);
        let parsed = parse_shadow_frame(&fr).unwrap();
        assert_eq!(parsed.plaintext_len, 4);
        assert_eq!(parsed.salt, salt);
        assert_eq!(parsed.nonce, nonce);
        assert_eq!(parsed.ciphertext, ciphertext);
    }

    #[test]
    fn shadow_capacity_basic() {
        // 100% pool = 100K positions. Capacity should be substantial.
        let cap = shadow_capacity(100_000);
        assert!(cap > 10_000, "capacity {cap} should be > 10KB for 100K positions");
        // Large image: 3M positions -> ~370KB capacity.
        let large_cap = shadow_capacity(3_000_000);
        assert!(large_cap > 300_000, "capacity {large_cap} should be > 300KB for 3M positions");
    }

    #[test]
    fn shadow_capacity_small() {
        assert_eq!(shadow_capacity(0), 0);
        // Very small pool -> 0 capacity.
        assert_eq!(shadow_capacity(7), 0);
    }

    #[test]
    fn shadow_capacity_larger_than_chroma_repetition() {
        // Use a realistically large position count (1M ~ 5MP image).
        let positions = 1_000_000usize;
        // Old chroma R=7 repetition: ~positions/7/8 bytes minus overhead.
        let old_chroma_cap = (positions / 7 / 8).saturating_sub(50);
        let cap = shadow_capacity(positions);
        assert!(
            cap > old_chroma_cap,
            "shadow ({cap}) should be larger than old chroma R=7 ({old_chroma_cap})"
        );
    }
}
