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
use crate::stego::cost::uniward::compute_positions_streaming;
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::frame::{self, MAX_FRAME_BITS};
use crate::stego::payload::{self, FileEntry, PayloadData};
use crate::stego::permute;
use crate::stego::progress;
use crate::stego::side_info::{self, SideInfo};
use crate::stego::stc::{embed, extract, hhat};

/// STC constraint length for Ghost Phase 1.
const STC_H: usize = 7;

/// Total number of progress steps reported by [`ghost_decode`].
///
/// Comprises [`UNIWARD_PROGRESS_STEPS`](crate::stego::cost::uniward::UNIWARD_PROGRESS_STEPS)
/// (3 steps from the cost computation) plus 1 step for STC extraction and decryption.
pub const GHOST_DECODE_STEPS: u32 = crate::stego::cost::uniward::UNIWARD_PROGRESS_STEPS + 1;

/// Total number of progress steps reported by Ghost encode.
///
/// 3 (UNIWARD sub-steps) + 8 (STC Viterbi sub-steps) + 1 (LSB mod + JPEG write).
pub const GHOST_ENCODE_STEPS: u32 = crate::stego::cost::uniward::UNIWARD_PROGRESS_STEPS + 8 + 1;

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
/// Returns [`StegoError::ImageTooLarge`] if the STC Viterbi back_ptr would
/// exceed the memory budget (prevents OOM on very large images).
fn compute_stc_params(n: usize) -> Result<(usize, usize, usize), StegoError> {
    let m_max = MAX_FRAME_BITS.min(n);
    if m_max == 0 {
        return Err(StegoError::ImageTooSmall);
    }
    let w = n / m_max; // floor division; always >= 1 since m_max <= n
    let n_used = m_max * w;

    // Sanity limit: reject unrealistically large images (> 500M AC positions
    // ≈ 8 gigapixels). The STC Viterbi uses a segmented checkpoint approach
    // for large images, so memory is O(√n) — typically a few MB.
    const STC_POSITION_LIMIT: usize = 500_000_000;
    if n_used > STC_POSITION_LIMIT {
        return Err(StegoError::ImageTooLarge);
    }

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
    ghost_encode_impl(image_bytes, message, &[], passphrase, None)
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
    ghost_encode_impl(image_bytes, message, files, passphrase, None)
}

/// Encode using Ghost mode with side information (SI-UNIWARD / "Deep Cover").
///
/// When the original uncompressed pixels are available (non-JPEG input like
/// PNG, HEIC, or RAW), the quantization rounding errors enable more efficient
/// embedding — roughly 1.5-2× capacity at the same detection risk.
///
/// The `raw_pixels_rgb` must be the ORIGINAL pixels that were JPEG-compressed
/// to produce `image_bytes`. They must have the same dimensions.
///
/// # Arguments
/// - `image_bytes`: Cover JPEG (as compressed by the platform from the raw pixels).
/// - `raw_pixels_rgb`: Original RGB pixels, row-major, 3 bytes per pixel.
/// - `pixel_width`, `pixel_height`: Dimensions of the raw pixel buffer.
/// - `message`: Plaintext message to embed.
/// - `passphrase`: Used for structural key derivation and encryption.
pub fn ghost_encode_si(
    image_bytes: &[u8],
    raw_pixels_rgb: &[u8],
    pixel_width: u32,
    pixel_height: u32,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    ghost_encode_si_with_files(
        image_bytes, raw_pixels_rgb, pixel_width, pixel_height,
        message, &[], passphrase,
    )
}

/// Encode with side information and file attachments.
pub fn ghost_encode_si_with_files(
    image_bytes: &[u8],
    raw_pixels_rgb: &[u8],
    pixel_width: u32,
    pixel_height: u32,
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    // Parse image first to get the grid and QT for side info computation
    let img = JpegImage::from_bytes(image_bytes)?;
    let fi = img.frame_info();
    super::validate_encode_dimensions(fi.width as u32, fi.height as u32)?;

    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    let qt_id = fi.components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;

    let si = SideInfo::compute(
        raw_pixels_rgb,
        pixel_width,
        pixel_height,
        img.dct_grid(0),
        &qt.values,
    );

    // Drop img so ghost_encode_impl can re-parse (it needs mut access)
    drop(img);

    ghost_encode_impl(image_bytes, message, files, passphrase, Some(si))
}

fn ghost_encode_impl(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    si: Option<SideInfo>,
) -> Result<Vec<u8>, StegoError> {
    // Initialize encode progress (10 steps: 1 UNIWARD + 8 STC + 1 write).
    progress::init(GHOST_ENCODE_STEPS);

    // Build the payload (text + files + compression).
    let payload_bytes = payload::encode_payload(message, files)?;

    let mut img = JpegImage::from_bytes(image_bytes)?;

    // Validate dimensions before any heavy processing.
    let fi = img.frame_info();
    super::validate_encode_dimensions(fi.width as u32, fi.height as u32)?;

    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    // 1. Compute J-UNIWARD costs strip-by-strip and collect positions directly.
    //    Reports 3 progress sub-steps. If SI info is available, modulates costs
    //    inline during position collection (no separate pass needed).
    //    Memory-optimized: never materializes full CostMap or pixel/wavelet arrays.
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    let si_ref = si.as_ref().map(|s| (s, img.dct_grid(0)));
    let mut positions = compute_positions_streaming(img.dct_grid(0), qt, si_ref)?;

    // 2. Derive structural key (Tier 1).
    let structural_key = crypto::derive_structural_key(passphrase)?;
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    // 3. Permute positions, then truncate to n_used.
    permute::permute_positions(&mut positions, &perm_seed);
    let n = positions.len();
    let (w, n_used, m_max) = compute_stc_params(n)?;
    positions.truncate(n_used);

    // 4. Encrypt payload (Tier 2 key with random salt).
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;

    // 5. Build payload frame and pad to m_max bits.
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
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
        let coeff = flat_get(grid, p.flat_idx as usize);
        (coeff.unsigned_abs() & 1) as u8
    }).collect();
    let costs: Vec<f32> = positions.iter().map(|p| p.cost).collect();

    // 7. Generate H-hat and embed (reports 8 progress sub-steps internally).
    let hhat_matrix = hhat::generate_hhat(STC_H, w, &hhat_seed);
    let result = embed::stc_embed(&cover_bits, &costs, &padded_bits, &hhat_matrix, STC_H, w);
    progress::check_cancelled()?;
    let result = result.ok_or(StegoError::MessageTooLarge)?;

    // 8. Apply LSB changes to DctGrid.
    // Direction depends on whether side information is available:
    // - SI-UNIWARD: toward pre-quantization value (rounding error direction).
    // - Standard (nsF5): toward zero for |coeff| > 1.
    // In both modes: |coeff| == 1 → away from zero (anti-shrinkage).
    let grid_mut = img.dct_grid_mut(0);
    for (idx, pos) in positions.iter().enumerate() {
        let old_bit = cover_bits[idx];
        let new_bit = result.stego_bits[idx];
        if old_bit != new_bit {
            let fi = pos.flat_idx as usize;
            let coeff = flat_get(grid_mut, fi);
            let modified = if let Some(ref side_info) = si {
                side_info::si_modify_coefficient(coeff, side_info.error_at(fi))
            } else {
                side_info::nsf5_modify_coefficient(coeff)
            };
            flat_set(grid_mut, fi, modified);
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

    // Step 10 complete: LSB modification + JPEG write.
    progress::advance();

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

    // 1. Compute J-UNIWARD costs strip-by-strip and collect positions directly.
    //    Reports UNIWARD_PROGRESS_STEPS (3) progress steps.
    //    Memory-optimized: never materializes full CostMap.
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    let mut positions = compute_positions_streaming(img.dct_grid(0), qt, None)?;

    progress::check_cancelled()?;

    // 2. Derive structural key.
    let structural_key = crypto::derive_structural_key(passphrase)?;
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    // 3. Permute positions, then truncate to n_used.
    permute::permute_positions(&mut positions, &perm_seed);
    let n = positions.len();
    let (w, n_used, m_max) = compute_stc_params(n)?;
    positions.truncate(n_used);

    // 4. Extract stego LSBs.
    let grid = img.dct_grid(0);
    let stego_bits: Vec<u8> = positions.iter().map(|p| {
        let coeff = flat_get(grid, p.flat_idx as usize);
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
