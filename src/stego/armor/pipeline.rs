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
    let frame_bytes = frame::build_frame_with_mode(
        frame::MODE_ARMOR,
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

    // 9. STDM embed each bit.
    let grid = img.dct_grid(0);
    let bw = grid.blocks_wide();

    // Collect coefficient groups and embed
    let grid_mut = img.dct_grid_mut(0);
    for (bit_idx, &bit) in rs_bits.iter().enumerate() {
        let group_start = bit_idx * SPREAD_LEN;
        let group = &positions[group_start..group_start + SPREAD_LEN];

        // Read current coefficient values as f64
        let mut coeffs = [0.0f64; SPREAD_LEN];
        for (k, pos) in group.iter().enumerate() {
            coeffs[k] = flat_get_from_grid(grid_mut, pos.flat_idx, bw) as f64;
        }

        // Embed
        stdm_embed(&mut coeffs, &vectors[bit_idx], bit, delta);

        // Write back rounded values
        for (k, pos) in group.iter().enumerate() {
            let new_val = coeffs[k].round() as i16;
            flat_set_in_grid(grid_mut, pos.flat_idx, bw, new_val);
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

/// Decode a text message from a stego JPEG using Armor mode.
pub fn armor_decode(stego_bytes: &[u8], passphrase: &str) -> Result<String, StegoError> {
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
    let bw = grid.blocks_wide();

    let mut extracted_bits = Vec::with_capacity(num_units);
    for unit_idx in 0..num_units {
        let group_start = unit_idx * SPREAD_LEN;
        let group = &positions[group_start..group_start + SPREAD_LEN];

        let mut coeffs = [0.0f64; SPREAD_LEN];
        for (k, pos) in group.iter().enumerate() {
            coeffs[k] = flat_get(grid, pos.flat_idx, bw) as f64;
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
    let decoded_frame = try_rs_decode_frame(&extracted_bytes)?;

    // 9. Parse frame.
    let parsed = frame::parse_frame_any_mode(&decoded_frame)?;
    if parsed.mode != frame::MODE_ARMOR {
        return Err(StegoError::UnknownFrameMode(parsed.mode));
    }

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

    Ok(text.to_string())
}

/// Try to RS-decode a frame from extracted bytes.
///
/// We don't know the exact original frame length, but we can estimate it
/// from the frame header (mode + plaintext_len in the first 3 bytes).
/// We try to read those bytes, compute the expected frame length,
/// then RS-decode that many blocks.
fn try_rs_decode_frame(extracted_bytes: &[u8]) -> Result<Vec<u8>, StegoError> {
    // The first RS block contains the frame header.
    // For a shortened RS block of data_len d, the encoded block is d + 64 bytes.
    // We need to figure out d. The frame header is: mode(1) + len(2) + ...
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

        if let Ok(first_block_data) = ecc::rs_decode(&extracted_bytes[..block_len], data_len) {
            // Check if this looks like a valid frame header
            if first_block_data.len() >= 3 {
                let mode = first_block_data[0];
                if mode != frame::MODE_ARMOR {
                    continue;
                }
                let pt_len =
                    u16::from_be_bytes([first_block_data[1], first_block_data[2]]) as usize;
                let ct_len = pt_len + 16; // AES-GCM tag
                let total_frame_len = 1 + 2 + 16 + 12 + ct_len + 4; // frame format

                if total_frame_len == data_len {
                    // Single block decode succeeded with exact size
                    return Ok(first_block_data);
                }

                if total_frame_len <= data_len {
                    // Frame fits in a single block
                    return Ok(first_block_data[..total_frame_len].to_vec());
                }

                // Need multiple blocks
                if total_frame_len > data_len {
                    // Try full multi-block decode
                    let rs_encoded_len = ecc::rs_encoded_len(total_frame_len);
                    if rs_encoded_len <= extracted_bytes.len() {
                        if let Ok(decoded) = ecc::rs_decode_blocks(
                            &extracted_bytes[..rs_encoded_len],
                            total_frame_len,
                        ) {
                            return Ok(decoded);
                        }
                    }
                }
            }
        }
    }

    Err(StegoError::FrameCorrupted)
}

// --- DctGrid flat access helpers ---

fn flat_get(grid: &DctGrid, flat_idx: usize, bw: usize) -> i16 {
    let block_idx = flat_idx / 64;
    let pos = flat_idx % 64;
    let br = block_idx / bw;
    let bc = block_idx % bw;
    let i = pos / 8;
    let j = pos % 8;
    grid.get(br, bc, i, j)
}

fn flat_get_from_grid(grid: &mut DctGrid, flat_idx: usize, bw: usize) -> i16 {
    let block_idx = flat_idx / 64;
    let pos = flat_idx % 64;
    let br = block_idx / bw;
    let bc = block_idx % bw;
    let i = pos / 8;
    let j = pos % 8;
    grid.get(br, bc, i, j)
}

fn flat_set_in_grid(grid: &mut DctGrid, flat_idx: usize, bw: usize, val: i16) {
    let block_idx = flat_idx / 64;
    let pos = flat_idx % 64;
    let br = block_idx / bw;
    let bc = block_idx % bw;
    let i = pos / 8;
    let j = pos % 8;
    grid.set(br, bc, i, j, val);
}
