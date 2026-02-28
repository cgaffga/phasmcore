// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Ghost mode encode/decode pipeline.
//!
//! Ghost mode embeds encrypted messages into JPEG DCT coefficients using:
//! 1. J-UNIWARD cost function to identify low-detectability embedding positions
//! 2. Fisher-Yates permutation (keyed by passphrase) to scatter the payload
//! 3. Syndrome-Trellis Coding (STC) to minimize total embedding distortion
//! 4. nsF5-style LSB modification (toward zero for |coeff| > 1, away from
//!    zero for |coeff| == 1 to prevent shrinkage)

use crate::jpeg::JpegImage;
use crate::stego::cost::uniward::{compute_uniward, compute_uniward_for_decode};
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::frame::{self, MAX_FRAME_BITS};
use crate::stego::payload::{self, FileEntry, PayloadData};
use crate::stego::permute;
use crate::stego::progress;
use crate::stego::stc::{embed, extract, hhat};

/// STC constraint length for Ghost Phase 1.
const STC_H: usize = 7;

/// Total number of progress steps reported by [`ghost_decode`].
///
/// Comprises [`UNIWARD_DECODE_STEPS`](crate::stego::cost::uniward::UNIWARD_DECODE_STEPS)
/// (3 steps from the cost computation) plus 1 step for STC extraction and decryption.
pub const GHOST_DECODE_STEPS: u32 = crate::stego::cost::uniward::UNIWARD_DECODE_STEPS + 1;

/// Compute the STC width `w`, usable cover length, and effective `m_max` from
/// the total number of AC positions. Both encoder and decoder must agree on
/// these values — they can because both see the same image and derive the
/// same `n`.
///
/// `m_max` is capped at `MAX_FRAME_BITS` (the u16 protocol limit) but also
/// capped at `n` so that small images still work (with `w = 1`).
///
/// Uses `w = floor(n / m_max)` so that `n_used = m_max * w <= n` and the
/// extraction produces exactly `m_max` bits.
///
/// # Returns
/// `(w, n_used, m_max)` where `w` is the STC submatrix width, `n_used` is the
/// number of cover positions to use (always <= `n`), and `m_max` is the
/// effective extraction size in bits.
///
/// # Errors
/// Returns [`StegoError::ImageTooSmall`] if `n` is 0.
fn compute_stc_params(n: usize) -> Result<(usize, usize, usize), StegoError> {
    let m_max = MAX_FRAME_BITS.min(n);
    if m_max == 0 {
        return Err(StegoError::ImageTooSmall);
    }
    let w = n / m_max; // floor division; always >= 1 since m_max <= n
    let n_used = m_max * w;
    Ok((w, n_used, m_max))
}

/// Encode a text message into a cover JPEG using Ghost mode.
///
/// # Arguments
/// - `image_bytes`: Raw bytes of the cover JPEG image.
/// - `message`: The plaintext message to embed (must fit within capacity).
/// - `passphrase`: Used for both structural key derivation and encryption.
///
/// # Returns
/// The stego JPEG image as bytes, or an error if the image is too small
/// or the message exceeds the embedding capacity.
///
/// # Errors
/// - [`StegoError::InvalidJpeg`] if `image_bytes` is not a valid baseline JPEG.
/// - [`StegoError::NoLuminanceChannel`] if the image has no Y component.
/// - [`StegoError::ImageTooSmall`] if the cover has too few usable coefficients.
/// - [`StegoError::MessageTooLarge`] if the message exceeds STC capacity.
pub fn ghost_encode(
    image_bytes: &[u8],
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    ghost_encode_impl(image_bytes, message, &[], passphrase)
}

/// Encode a text message with file attachments into a cover JPEG using Ghost mode.
///
/// Files are embedded alongside the text message in the payload. The entire
/// payload (text + files) is compressed with Brotli before encryption.
pub fn ghost_encode_with_files(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    ghost_encode_impl(image_bytes, message, files, passphrase)
}

fn ghost_encode_impl(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    // Build the payload (text + files + compression).
    let payload_bytes = payload::encode_payload(message, files)?;

    // Guard against payload exceeding the u16 length field in the frame format.
    if payload_bytes.len() > u16::MAX as usize {
        return Err(StegoError::MessageTooLarge);
    }

    let mut img = JpegImage::from_bytes(image_bytes)?;

    // Validate dimensions before any heavy processing.
    let fi = img.frame_info();
    super::validate_encode_dimensions(fi.width as u32, fi.height as u32)?;

    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    // 1. Compute J-UNIWARD costs for Y channel.
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    let cost_map = compute_uniward(img.dct_grid(0), qt);

    // 2. Derive structural key (Tier 1).
    let structural_key = crypto::derive_structural_key(passphrase);
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    // 3. Select and permute all AC positions, then truncate to n_used.
    let positions = permute::select_and_permute(&cost_map, &perm_seed);
    let n = positions.len();
    let (w, n_used, m_max) = compute_stc_params(n)?;
    let positions = &positions[..n_used];

    // 4. Encrypt payload (Tier 2 key with random salt).
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase);

    // 5. Build payload frame and pad to m_max bits.
    let frame_bytes = frame::build_frame(payload_bytes.len() as u16, &salt, &nonce, &ciphertext);
    let frame_bits = frame::bytes_to_bits(&frame_bytes);
    let m = frame_bits.len();

    if m > m_max {
        return Err(StegoError::MessageTooLarge);
    }

    let mut padded_bits = vec![0u8; m_max];
    padded_bits[..m].copy_from_slice(&frame_bits);

    // 6. Extract cover LSBs and costs in permuted order.
    let grid = img.dct_grid(0);
    let cover_bits: Vec<u8> = positions.iter().map(|p| {
        let coeff = flat_get(grid, p.flat_idx);
        (coeff.unsigned_abs() & 1) as u8
    }).collect();
    let costs: Vec<f64> = positions.iter().map(|p| p.cost).collect();

    // 7. Generate H-hat and embed.
    let hhat_matrix = hhat::generate_hhat(STC_H, w, &hhat_seed);
    let result = embed::stc_embed(&cover_bits, &costs, &padded_bits, &hhat_matrix, STC_H, w)
        .ok_or(StegoError::MessageTooLarge)?;

    // 8. Apply LSB changes to DctGrid.
    // For |coeff| > 1: move toward zero (nsF5 convention).
    // For |coeff| == 1: move AWAY from zero (±1 → ±2) to prevent shrinkage,
    // which would break encoder-decoder position alignment.
    let grid_mut = img.dct_grid_mut(0);
    for (idx, pos) in positions.iter().enumerate() {
        let old_bit = cover_bits[idx];
        let new_bit = result.stego_bits[idx];
        if old_bit != new_bit {
            let coeff = flat_get(grid_mut, pos.flat_idx);
            let modified = if coeff == 1 {
                2 // away from zero to prevent shrinkage
            } else if coeff == -1 {
                -2 // away from zero to prevent shrinkage
            } else if coeff > 1 {
                coeff - 1 // toward zero
            } else if coeff < -1 {
                coeff + 1 // toward zero
            } else {
                coeff // zero: should never happen (filtered out)
            };
            flat_set(grid_mut, pos.flat_idx, modified);
        }
    }

    // 9. Write modified JPEG.
    // Prefer original Huffman tables to avoid coefficient drift from rebuild.
    // Fall back to rebuilt tables only if the original tables can't encode
    // a new symbol introduced by the ±1/±2 coefficient changes.
    let stego_bytes = match img.to_bytes() {
        Ok(bytes) => bytes,
        Err(_) => {
            img.rebuild_huffman_tables();
            img.to_bytes().map_err(StegoError::InvalidJpeg)?
        }
    };
    Ok(stego_bytes)
}

/// Decode a payload from a stego JPEG using Ghost mode.
///
/// Returns the decoded text and any embedded files.
///
/// # Errors
/// - [`StegoError::DecryptionFailed`] if the passphrase is wrong.
/// - [`StegoError::FrameCorrupted`] if the CRC check fails.
/// - [`StegoError::InvalidUtf8`] if the decrypted payload is not valid UTF-8.
pub fn ghost_decode(
    stego_bytes: &[u8],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    let img = JpegImage::from_bytes(stego_bytes)?;

    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    // 1. Compute J-UNIWARD costs (for consistent position selection).
    //    Reports UNIWARD_DECODE_STEPS (3) progress steps internally:
    //      - pixel decompression
    //      - wavelet subbands
    //      - per-block cost computation
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    let cost_map = compute_uniward_for_decode(img.dct_grid(0), qt)?;

    progress::check_cancelled()?;

    // 2. Derive structural key.
    let structural_key = crypto::derive_structural_key(passphrase);
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    // 3. Select and permute AC positions (portable u32 shuffle).
    let positions = permute::select_and_permute(&cost_map, &perm_seed);
    let n = positions.len();
    let (w, n_used, m_max) = compute_stc_params(n)?;
    let positions = &positions[..n_used];

    // 4. Extract stego LSBs.
    let grid = img.dct_grid(0);
    let stego_bits: Vec<u8> = positions.iter().map(|p| {
        let coeff = flat_get(grid, p.flat_idx);
        (coeff.unsigned_abs() & 1) as u8
    }).collect();

    // 5. Generate H-hat and extract message bits.
    let hhat_matrix = hhat::generate_hhat(STC_H, w, &hhat_seed);
    let extracted_bits = extract::stc_extract(&stego_bits, &hhat_matrix, w);

    // 6. Convert bits to bytes and parse frame.
    let frame_bytes = frame::bits_to_bytes(&extracted_bits[..m_max]);
    let parsed = frame::parse_frame(&frame_bytes)?;

    // 7. Decrypt.
    let plaintext = crypto::decrypt(
        &parsed.ciphertext,
        passphrase,
        &parsed.salt,
        &parsed.nonce,
    )?;

    // 8. Truncate to declared length and decode payload (decompress + parse).
    let len = parsed.plaintext_len as usize;
    if len > plaintext.len() {
        return Err(StegoError::FrameCorrupted);
    }

    let result = payload::decode_payload(&plaintext[..len]);

    // Step 4: STC extraction + decryption complete.
    progress::advance();

    result
}

// --- DctGrid flat access helpers ---

use crate::jpeg::dct::DctGrid;

/// Read a coefficient from a `DctGrid` using a flat index.
///
/// The flat index encodes `block_index * 64 + row * 8 + col`, where
/// `block_index = block_row * blocks_wide + block_col`.
fn flat_get(grid: &DctGrid, flat_idx: usize) -> i16 {
    let bw = grid.blocks_wide();
    let block_idx = flat_idx / 64;
    let pos = flat_idx % 64;
    let br = block_idx / bw;
    let bc = block_idx % bw;
    let i = pos / 8;
    let j = pos % 8;
    grid.get(br, bc, i, j)
}

/// Write a coefficient into a `DctGrid` using a flat index.
fn flat_set(grid: &mut DctGrid, flat_idx: usize, val: i16) {
    let bw = grid.blocks_wide();
    let block_idx = flat_idx / 64;
    let pos = flat_idx % 64;
    let br = block_idx / bw;
    let bc = block_idx % bw;
    let i = pos / 8;
    let j = pos % 8;
    grid.set(br, bc, i, j, val);
}
