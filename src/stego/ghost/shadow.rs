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

use crate::codec::jpeg::JpegImage;
use crate::stego::armor::ecc;
use crate::stego::crypto;
#[cfg(test)]
use crate::stego::crypto::{NONCE_LEN, SALT_LEN};
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload::{self, FileEntry, PayloadData};
use crate::stego::permute::CoeffPos;
use super::pipeline::{flat_get, flat_set};
use crate::stego::shadow_layer::{
    build_shadow_frame, parse_shadow_frame, peek_shadow_fdl, SHADOW_FRAME_OVERHEAD_V1,
};
use crate::stego::side_info::nsf5_modify_coefficient;

/// Frame overhead lower bound (= v1 overhead, 46 bytes). The
/// brute-force decoder scans `[SHADOW_FRAME_OVERHEAD..=k-1]` which
/// always lives in v1 territory — v2 frames are inherently large
/// (≥65586 bytes) and reach the decoder via the peek path, not
/// brute-force. Image-side encoder picks v1 for plaintexts
/// ≤ u16::MAX (matches `shadow_layer::build_shadow_frame`).
const SHADOW_FRAME_OVERHEAD: usize = SHADOW_FRAME_OVERHEAD_V1;

/// RS parity tiers. Brute-forced at decode.
const SHADOW_PARITY_TIERS: [usize; 6] = [4, 8, 16, 32, 64, 128];

/// Cost pool fractions (denominators). Encoder picks smallest that fits.
/// 20 = cheapest 5%, 10 = 10%, 5 = 20%, 2 = 50%, 1 = 100%.
/// Smaller fractions give better stealth (positions in most textured regions).
/// Decoder brute-forces all fractions.
const COST_FRACTIONS: [usize; 5] = [20, 10, 5, 2, 1];

/// Maximum RS-encoded frame bytes to prevent unreasonable allocations.
const MAX_SHADOW_FRAME_BYTES: usize = 256 * 1024;

// T1.10 — thread-local scratch buffer for the brute-force loop's
// bits_to_bytes step. Phase 2b can call try_single_fdl ~6000× per
// smart_decode no-match path; reusing a per-thread Vec eliminates
// the per-iteration allocation churn. Each rayon worker gets its
// own scratch via thread_local.
thread_local! {
    static RS_BYTES_SCRATCH: std::cell::RefCell<Vec<u8>> =
        std::cell::RefCell::new(Vec::with_capacity(256));
}

/// Try a single (lsbs, fdl, parity, passphrase) combination.
/// Returns `Some(Ok(payload))` on success, `None` on failure.
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

    RS_BYTES_SCRATCH.with(|scratch| {
        let mut rs_bytes = scratch.borrow_mut();
        frame::bits_to_bytes_into(&lsbs[..rs_bits_needed], &mut rs_bytes);
        try_single_fdl_with_rs_bytes(&rs_bytes, fdl, parity_len, passphrase)
    })
}

/// Inner body of [`try_single_fdl`] given pre-packed RS bytes (T1.10).
fn try_single_fdl_with_rs_bytes(
    rs_bytes: &[u8],
    fdl: usize,
    parity_len: usize,
    passphrase: &str,
) -> Option<Result<PayloadData, StegoError>> {
    let decoded = match ecc::rs_decode_blocks_with_parity(rs_bytes, fdl, parity_len) {
        Ok((data, _stats)) => data,
        Err(_) => return None,
    };

    // O(1) format-aware consistency gate (port of the #532 video
    // shadow fix to the image side). peek_shadow_fdl reads the
    // first 2-6 bytes, dispatches v1 vs v2, returns the producer's
    // total frame length. If that doesn't equal the brute-force fdl
    // candidate, reject without parse + AES. Skips ~99% of bad
    // candidates' Argon2 + AES-GCM-SIV work.
    let expected_total = peek_shadow_fdl(&decoded)?;
    if expected_total != fdl {
        return None;
    }

    let fr = match parse_shadow_frame(&decoded) {
        Ok(f) => f,
        Err(_) => return None,
    };

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

/// Peek at the first RS block to read `plaintext_len` and derive the exact fdl.
/// Returns the candidate fdl if the first block decodes and the resulting fdl
/// is plausible (>= k, within pool capacity).
///
/// 2026-05-21 unification — delegates v1/v2 dispatch to the shared
/// [`peek_shadow_fdl`] helper in `shadow_layer`. Same dispatch logic
/// for image + video shadow.
fn peek_fdl_from_first_block(
    lsbs: &[u8],
    parity_len: usize,
    max_fdl: usize,
) -> Option<usize> {
    let k = 255usize.saturating_sub(parity_len);
    if k < 2 || lsbs.len() < 255 * 8 {
        return None;
    }

    // Decode the first 255 RS bytes as one full block (k data bytes).
    let first_block_bytes = frame::bits_to_bytes(&lsbs[..255 * 8]);
    let (data, _) = ecc::rs_decode_blocks_with_parity(&first_block_bytes, k, parity_len).ok()?;

    let fdl = peek_shadow_fdl(&data)?;

    // Only valid if fdl >= k (multi-block, so first block IS a full 255-byte block)
    // and within pool capacity.
    if fdl >= k && fdl <= max_fdl {
        Some(fdl)
    } else {
        None
    }
}

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

    if all_y_positions_sorted.is_empty() {
        return Err(StegoError::FrameCorrupted);
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        // Phase 1: Pre-extract LSBs for all fractions in parallel.
        let fraction_lsbs: Vec<(usize, Vec<u8>)> = COST_FRACTIONS.par_iter().filter_map(|&fraction| {
            let pool_size = all_y_positions_sorted.len() / fraction;
            if pool_size == 0 {
                return None;
            }
            let positions = select_shadow_positions(
                all_y_positions_sorted, fraction, pool_size, &perm_seed,
            );
            if positions.is_empty() {
                return None;
            }
            let lsbs: Vec<u8> = positions.iter().map(|pos| {
                let coeff = flat_get(grid, pos.flat_idx as usize);
                (coeff.unsigned_abs() & 1) as u8
            }).collect();
            Some((fraction, lsbs))
        }).collect();

        // Phase 2a: First-block peek — decode first RS block to read plaintext_len
        // and derive the exact fdl. Handles messages where fdl >= k (most cases).
        // This is O(30) RS block decodes — very fast.
        // T1.11 — parallelized across (fraction, parity) combos with
        // find_map_first short-circuit on first hit. Was sequential.
        let combos_2a: Vec<(usize, usize)> = (0..fraction_lsbs.len())
            .flat_map(|fi| SHADOW_PARITY_TIERS.iter().map(move |&p| (fi, p)))
            .collect();
        if let Some(result) = combos_2a.par_iter().find_map_first(|&(fi, parity_len)| {
            let lsbs = &fraction_lsbs[fi].1;
            let k = 255usize.saturating_sub(parity_len);
            if k == 0 { return None; }
            let max_rs_bytes = lsbs.len() / 8;
            let max_fdl = compute_max_fdl(max_rs_bytes, parity_len)
                .min(MAX_SHADOW_FRAME_BYTES);
            if SHADOW_FRAME_OVERHEAD > max_fdl { return None; }
            let fdl = peek_fdl_from_first_block(lsbs, parity_len, max_fdl)?;
            try_single_fdl(lsbs, fdl, parity_len, passphrase)
        }) {
            return result;
        }

        // Phase 2b: Small-fdl fallback — for tiny messages where fdl < k (single
        // partial RS block). Scan fdl from SHADOW_FRAME_OVERHEAD to min(k-1, max_fdl).
        // Typically ~200 values per (fraction, parity) → ~6K combos total.
        let mut combos: Vec<(usize, usize, usize)> = Vec::new();
        for (fi, (_, lsbs)) in fraction_lsbs.iter().enumerate() {
            for &parity_len in &SHADOW_PARITY_TIERS {
                let k = 255usize.saturating_sub(parity_len);
                if k < 2 { continue; }
                let max_rs_bytes = lsbs.len() / 8;
                let max_fdl = compute_max_fdl(max_rs_bytes, parity_len)
                    .min(MAX_SHADOW_FRAME_BYTES);
                // Only scan fdl < k (partial block range); peek handled fdl >= k.
                let small_max = (k - 1).min(max_fdl);
                if SHADOW_FRAME_OVERHEAD > small_max { continue; }
                for fdl in SHADOW_FRAME_OVERHEAD..=small_max {
                    combos.push((fi, parity_len, fdl));
                }
            }
        }

        let result = combos.par_iter().find_map_first(|&(fi, parity_len, fdl)| {
            let lsbs = &fraction_lsbs[fi].1;
            try_single_fdl(lsbs, fdl, parity_len, passphrase)
        });

        match result {
            Some(ok_or_err) => ok_or_err,
            None => Err(StegoError::FrameCorrupted),
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut last_err = StegoError::FrameCorrupted;

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
}

/// Try all (parity, fdl) combinations on a given LSB vector using
/// first-block peek + small-fdl fallback.
/// Returns `Some(Ok(...))` on success or `None` to continue to next fraction.
#[cfg(not(feature = "parallel"))]
fn try_extract_with_lsbs(
    all_lsbs: &[u8],
    passphrase: &str,
    _last_err: &mut StegoError,
) -> Option<Result<PayloadData, StegoError>> {
    for &parity_len in &SHADOW_PARITY_TIERS {
        let k = 255usize.saturating_sub(parity_len);
        if k < 2 { continue; }

        let max_rs_bytes = all_lsbs.len() / 8;
        let max_fdl = compute_max_fdl(max_rs_bytes, parity_len)
            .min(MAX_SHADOW_FRAME_BYTES);
        if SHADOW_FRAME_OVERHEAD > max_fdl { continue; }

        // First-block peek: derive exact fdl from first RS block.
        if let Some(fdl) = peek_fdl_from_first_block(all_lsbs, parity_len, max_fdl)
            && let Some(result) = try_single_fdl(all_lsbs, fdl, parity_len, passphrase) {
                return Some(result);
            }

        // Small-fdl fallback: scan partial-block range (fdl < k).
        let small_max = (k - 1).min(max_fdl);
        if SHADOW_FRAME_OVERHEAD > small_max { continue; }
        for fdl in SHADOW_FRAME_OVERHEAD..=small_max {
            if let Some(result) = try_single_fdl(all_lsbs, fdl, parity_len, passphrase) {
                return Some(result);
            }
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
//
// `build_shadow_frame`, `parse_shadow_frame`, and `ParsedShadowFrame`
// now live in `crate::stego::shadow_layer` (unified v1/v2 dispatch
// for image + video). Imported at the top of this file.

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
    // T2.12 — LSD radix sort on the u64 priorities is ~2-3x faster
    // than comparison sort for large pools (the 100% fraction on a
    // 12 MP photo sorts ~766 K elements; std sort_unstable_by_key
    // there runs ~80-100 ms, radsort runs ~30-40 ms). On unique
    // u64 keys (collision probability ~2^-26 at this scale) any
    // sort algorithm produces the same logical order, so the swap
    // is byte-identical to the previous T1.9 unstable-sort output
    // in practice. Path A wire-safe.
    radsort::sort_by_key(&mut candidates, |(priority, _)| *priority);

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
