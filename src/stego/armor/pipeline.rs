//! Armor mode encode/decode pipeline.
//!
//! Armor embeds messages using STDM (Spread Transform Dither Modulation)
//! into recompression-stable DCT coefficients, protected by Reed-Solomon
//! error correction.

use crate::jpeg::JpegImage;
use crate::jpeg::dct::DctGrid;
use crate::stego::armor::ecc;
use crate::stego::armor::embedding::{self, stdm_embed, stdm_extract};
use crate::stego::armor::selection::compute_stability_map;
use crate::stego::armor::spreading::{generate_spreading_vectors, SPREAD_LEN};
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::permute;

/// Encode a text message into a cover JPEG using Armor mode.
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
/// - [`StegoError::ImageTooSmall`] if there are too few stable positions.
/// - [`StegoError::MessageTooLarge`] if the RS-encoded frame exceeds capacity.
pub fn armor_encode(
    image_bytes: &[u8],
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    let mut img = JpegImage::from_bytes(image_bytes)?;

    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    // 1. Compute stability map for Y channel.
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img
        .quant_table(qt_id)
        .ok_or(StegoError::NoLuminanceChannel)?;
    let cost_map = compute_stability_map(img.dct_grid(0), qt);

    // 2. Derive structural key with Armor salt.
    let structural_key = crypto::derive_armor_structural_key(passphrase);
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let spread_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    // 3. Select and permute stable positions.
    let positions = permute::select_and_permute(&cost_map, &perm_seed);
    let num_units = positions.len() / SPREAD_LEN;
    if num_units == 0 {
        return Err(StegoError::ImageTooSmall);
    }
    let n_used = num_units * SPREAD_LEN;
    let positions = &positions[..n_used];

    // 4. Encrypt message (Tier 2 key with random salt).
    let (ciphertext, nonce, salt) = crypto::encrypt(message.as_bytes(), passphrase);

    // 5. Build payload frame.
    let frame_bytes = frame::build_frame(
        message.len() as u16,
        &salt,
        &nonce,
        &ciphertext,
    );

    // 6. RS-encode the frame.
    let rs_encoded = ecc::rs_encode_blocks(&frame_bytes);
    let rs_bits = frame::bytes_to_bits(&rs_encoded);

    if rs_bits.len() > num_units {
        return Err(StegoError::MessageTooLarge);
    }

    // 7. Compute delta from quantization table.
    let delta = embedding::compute_delta(&qt.values);

    // 8. Generate spreading vectors.
    let vectors = generate_spreading_vectors(&spread_seed, rs_bits.len());

    // 9. STDM embed each bit into coefficient groups.
    let grid_mut = img.dct_grid_mut(0);
    for (bit_idx, &bit) in rs_bits.iter().enumerate() {
        let group_start = bit_idx * SPREAD_LEN;
        let group = &positions[group_start..group_start + SPREAD_LEN];

        // Read current coefficient values as f64
        let mut coeffs = [0.0f64; SPREAD_LEN];
        for (k, pos) in group.iter().enumerate() {
            coeffs[k] = flat_get(grid_mut, pos.flat_idx) as f64;
        }

        // Embed
        stdm_embed(&mut coeffs, &vectors[bit_idx], bit, delta);

        // Write back rounded values
        for (k, pos) in group.iter().enumerate() {
            let new_val = coeffs[k].round() as i16;
            flat_set(grid_mut, pos.flat_idx, new_val);
        }
    }

    // 10. Write modified JPEG.
    let stego_bytes = match img.to_bytes() {
        Ok(bytes) => bytes,
        Err(_) => {
            img.rebuild_huffman_tables();
            img.to_bytes().map_err(StegoError::InvalidJpeg)?
        }
    };

    Ok(stego_bytes)
}

/// Quality information from a successful decode.
#[derive(Debug, Clone)]
pub struct DecodeQuality {
    /// Mode that was used: `frame::MODE_GHOST` or `frame::MODE_ARMOR`.
    pub mode: u8,
    /// Number of RS symbol errors corrected (0 for Ghost).
    pub rs_errors_corrected: u32,
    /// Maximum correctable RS errors across all blocks (0 for Ghost).
    pub rs_error_capacity: u32,
    /// Integrity percentage: 100 = pristine, 0 = barely recovered.
    pub integrity_percent: u8,
}

impl DecodeQuality {
    /// Create quality info for a Ghost decode (binary: always 100% if successful).
    pub fn ghost() -> Self {
        Self {
            mode: super::super::frame::MODE_GHOST,
            rs_errors_corrected: 0,
            rs_error_capacity: 0,
            integrity_percent: 100,
        }
    }

    /// Create quality info from Armor RS decode stats.
    pub fn from_rs_stats(stats: &ecc::RsDecodeStats) -> Self {
        let integrity = if stats.error_capacity == 0 {
            100
        } else {
            let ratio = stats.total_errors as f64 / stats.error_capacity as f64;
            ((1.0 - ratio) * 100.0).round().max(0.0).min(100.0) as u8
        };
        Self {
            mode: super::super::frame::MODE_ARMOR,
            rs_errors_corrected: stats.total_errors as u32,
            rs_error_capacity: stats.error_capacity as u32,
            integrity_percent: integrity,
        }
    }
}

/// Decode a text message from a stego JPEG using Armor mode.
///
/// # Arguments
/// - `stego_bytes`: Raw bytes of the stego JPEG image.
/// - `passphrase`: The passphrase used during encoding.
///
/// # Returns
/// A tuple of (decoded plaintext message, decode quality info).
///
/// # Errors
/// - [`StegoError::DecryptionFailed`] if the passphrase is wrong.
/// - [`StegoError::FrameCorrupted`] if RS decoding or CRC check fails.
/// - [`StegoError::InvalidUtf8`] if the decrypted payload is not valid UTF-8.
pub fn armor_decode(stego_bytes: &[u8], passphrase: &str) -> Result<(String, DecodeQuality), StegoError> {
    let img = JpegImage::from_bytes(stego_bytes)?;

    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    // 1. Compute stability map.
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img
        .quant_table(qt_id)
        .ok_or(StegoError::NoLuminanceChannel)?;
    let cost_map = compute_stability_map(img.dct_grid(0), qt);

    // 2. Derive structural key.
    let structural_key = crypto::derive_armor_structural_key(passphrase);
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let spread_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    // 3. Select and permute stable positions.
    let positions = permute::select_and_permute(&cost_map, &perm_seed);
    let num_units = positions.len() / SPREAD_LEN;
    if num_units == 0 {
        return Err(StegoError::ImageTooSmall);
    }
    let n_used = num_units * SPREAD_LEN;
    let positions = &positions[..n_used];

    // 4. Compute delta.
    let delta = embedding::compute_delta(&qt.values);

    // 5. Generate spreading vectors for all possible units.
    let vectors = generate_spreading_vectors(&spread_seed, num_units);

    // 6. Extract all bits via STDM.
    let grid = img.dct_grid(0);

    let mut extracted_bits = Vec::with_capacity(num_units);
    for unit_idx in 0..num_units {
        let group_start = unit_idx * SPREAD_LEN;
        let group = &positions[group_start..group_start + SPREAD_LEN];

        let mut coeffs = [0.0f64; SPREAD_LEN];
        for (k, pos) in group.iter().enumerate() {
            coeffs[k] = flat_get(grid, pos.flat_idx) as f64;
        }

        let bit = stdm_extract(&coeffs, &vectors[unit_idx], delta);
        extracted_bits.push(bit);
    }

    // 7. Convert bits to bytes.
    let extracted_bytes = frame::bits_to_bytes(&extracted_bits);

    // 8. We need to figure out how long the RS-encoded data is.
    //    First, try to read the frame header to get the plaintext length,
    //    then compute expected RS block sizes.
    //    Strategy: try decoding with decreasing assumed data lengths.
    let (decoded_frame, rs_stats) = try_rs_decode_frame(&extracted_bytes)?;

    // 9. Parse frame.
    let parsed = frame::parse_frame(&decoded_frame)?;

    // 10. Decrypt.
    let plaintext = crypto::decrypt(
        &parsed.ciphertext,
        passphrase,
        &parsed.salt,
        &parsed.nonce,
    )?;

    // 11. Validate.
    let len = parsed.plaintext_len as usize;
    if len > plaintext.len() {
        return Err(StegoError::FrameCorrupted);
    }

    let text =
        std::str::from_utf8(&plaintext[..len]).map_err(|_| StegoError::InvalidUtf8)?;

    let quality = DecodeQuality::from_rs_stats(&rs_stats);
    Ok((text.to_string(), quality))
}

/// Try to RS-decode a frame from extracted bytes.
///
/// The decoder faces a chicken-and-egg problem: the RS block boundaries
/// depend on the frame length, which is inside the RS-encoded data. This
/// function resolves it by trying progressively larger data sizes for the
/// first RS block until one successfully decodes and reveals a valid
/// frame header (plaintext length). The header then determines the full
/// frame size and any additional RS blocks.
///
/// This is robust because:
/// 1. RS decoding fails fast for incorrect block sizes (syndrome check)
/// 2. The CRC provides additional validation
/// 3. AES-GCM-SIV authentication catches any remaining corruption
fn try_rs_decode_frame(extracted_bytes: &[u8]) -> Result<(Vec<u8>, ecc::RsDecodeStats), StegoError> {
    // The first RS block contains the frame header.
    // For a shortened RS block of data_len d, the encoded block is d + 64 bytes.
    // We need to figure out d. The frame header is: len(2) + ...
    // Minimum frame is FRAME_OVERHEAD bytes.
    //
    // Strategy: try to decode the first RS block at various sizes to find
    // a valid frame header, then use that to determine the full frame size.

    // Start with a single-block decode, trying sizes from small to large.
    let parity = ecc::parity_len();

    // Try progressively larger data sizes for the first block
    // Start with FRAME_OVERHEAD (minimum possible frame)
    let min_data = super::super::frame::FRAME_OVERHEAD;
    let max_first_block_data = 191usize; // K_DEFAULT

    for data_len in min_data..=max_first_block_data.min(extracted_bytes.len().saturating_sub(parity))
    {
        let block_len = data_len + parity;
        if block_len > extracted_bytes.len() {
            break;
        }

        if let Ok((first_block_data, first_errors)) = ecc::rs_decode(&extracted_bytes[..block_len], data_len) {
            // Check if this looks like a valid frame header
            if first_block_data.len() >= 2 {
                let pt_len =
                    u16::from_be_bytes([first_block_data[0], first_block_data[1]]) as usize;
                let ct_len = pt_len + 16; // AES-GCM tag
                let total_frame_len = 2 + 16 + 12 + ct_len + 4; // frame format

                // Sanity check: reject implausible plaintext lengths
                if total_frame_len > frame::MAX_FRAME_BYTES + 1024 {
                    continue;
                }

                if total_frame_len == data_len {
                    // Single block decode succeeded with exact size
                    let stats = ecc::RsDecodeStats {
                        total_errors: first_errors,
                        error_capacity: ecc::T_MAX,
                        max_block_errors: first_errors,
                        num_blocks: 1,
                    };
                    return Ok((first_block_data, stats));
                }

                if total_frame_len <= data_len {
                    // Frame fits in a single block
                    let stats = ecc::RsDecodeStats {
                        total_errors: first_errors,
                        error_capacity: ecc::T_MAX,
                        max_block_errors: first_errors,
                        num_blocks: 1,
                    };
                    return Ok((first_block_data[..total_frame_len].to_vec(), stats));
                }

                // Need multiple blocks
                if total_frame_len > data_len {
                    // Try full multi-block decode
                    let rs_encoded_len = ecc::rs_encoded_len(total_frame_len);
                    if rs_encoded_len <= extracted_bytes.len() {
                        if let Ok((decoded, stats)) = ecc::rs_decode_blocks(
                            &extracted_bytes[..rs_encoded_len],
                            total_frame_len,
                        ) {
                            return Ok((decoded, stats));
                        }
                    }
                }
            }
        }
    }

    Err(StegoError::FrameCorrupted)
}

// --- DctGrid flat access helpers ---

/// Read a coefficient from a `DctGrid` using a flat index.
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
