//! Armor mode encode/decode pipeline.
//!
//! Armor embeds messages using STDM (Spread Transform Dither Modulation)
//! into recompression-stable DCT coefficients, protected by Reed-Solomon
//! error correction.
//!
//! **Phase 2 (adaptive robustness):** When the message is small relative to
//! image capacity, the encoder automatically uses stronger RS parity, repeats
//! the RS-encoded bitstream across spare capacity, and increases the STDM delta.
//! On decode, soft majority voting combines redundant copies for dramatically
//! improved recompression survival.

use crate::jpeg::JpegImage;
use crate::jpeg::dct::DctGrid;
use crate::stego::armor::ecc;
use crate::stego::armor::embedding::{self, stdm_embed, stdm_extract, stdm_extract_soft};
use crate::stego::armor::repetition;
use crate::stego::armor::selection::compute_stability_map;
use crate::stego::armor::spreading::{generate_spreading_vectors, SPREAD_LEN};
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::permute;

/// Number of embedding units reserved for the r-header.
/// 7 copies × 8 bits = 56 units encode the repetition factor (1 byte).
const R_HEADER_UNITS: usize = 56;
const R_HEADER_COPIES: usize = 7;

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

    // 6. Decide Phase 1 vs Phase 2 encoding.
    //
    // Phase 2 (adaptive RS + repetition + r-header) only activates when:
    //   - There's room for the r-header (num_units > R_HEADER_UNITS)
    //   - Repetition r >= 3 (otherwise the overhead isn't worth it)
    //
    // If Phase 2 isn't beneficial, fall back to Phase 1 (fixed RS parity=64).
    let baseline_delta = embedding::compute_delta(&qt.values);

    // Probe Phase 2 viability
    let use_phase2 = if num_units > R_HEADER_UNITS {
        let payload_units = num_units - R_HEADER_UNITS;
        let parity = ecc::choose_parity_tier(frame_bytes.len(), payload_units);
        let rs_encoded = ecc::rs_encode_blocks_with_parity(&frame_bytes, parity);
        let rs_bits_len = rs_encoded.len() * 8;
        let r = repetition::compute_r(rs_bits_len, payload_units);
        r >= 3 && rs_bits_len <= payload_units
    } else {
        false
    };

    let (all_bits, embed_delta_fn): (Vec<u8>, Box<dyn Fn(usize) -> f64>) = if use_phase2 {
        // --- Phase 2 encode ---
        let payload_units = num_units - R_HEADER_UNITS;
        let parity = ecc::choose_parity_tier(frame_bytes.len(), payload_units);
        let rs_encoded = ecc::rs_encode_blocks_with_parity(&frame_bytes, parity);
        let rs_bits = frame::bytes_to_bits(&rs_encoded);

        let r = repetition::compute_r(rs_bits.len(), payload_units);

        // Pad RS bits to payload_units / r so encoder and decoder agree on copy length.
        // The decoder computes rs_bit_count = payload_units / r without knowing rs_bits.len().
        let rs_bit_count_aligned = payload_units / r;
        let mut rs_bits_padded = rs_bits;
        rs_bits_padded.resize(rs_bit_count_aligned, 0);
        let (payload_bits, _) = repetition::repetition_encode(&rs_bits_padded, payload_units);

        let adaptive_delta = embedding::compute_delta_adaptive(&qt.values, r);

        // Build r-header: 7 copies of r as 8 bits
        let r_byte = r as u8;
        let r_bits_vec: Vec<u8> = (0..8).rev().map(|bp| (r_byte >> bp) & 1).collect();
        let mut combined = Vec::with_capacity(num_units);
        for _ in 0..R_HEADER_COPIES {
            combined.extend_from_slice(&r_bits_vec);
        }
        combined.extend_from_slice(&payload_bits[..payload_units.min(payload_bits.len())]);

        (combined, Box::new(move |bit_idx| {
            if bit_idx < R_HEADER_UNITS { baseline_delta } else { adaptive_delta }
        }))
    } else {
        // --- Phase 1 encode (unchanged from original) ---
        let rs_encoded = ecc::rs_encode_blocks(&frame_bytes);
        let rs_bits = frame::bytes_to_bits(&rs_encoded);

        if rs_bits.len() > num_units {
            return Err(StegoError::MessageTooLarge);
        }

        // Pad to num_units (unused bits are zero)
        let mut bits = rs_bits;
        bits.resize(num_units, 0);

        (bits, Box::new(move |_| baseline_delta))
    };

    let embed_count = all_bits.len().min(num_units);

    // 7. Generate spreading vectors.
    let vectors = generate_spreading_vectors(&spread_seed, embed_count);

    // 8. STDM embed each bit into coefficient groups.
    let grid_mut = img.dct_grid_mut(0);
    for bit_idx in 0..embed_count {
        let group_start = bit_idx * SPREAD_LEN;
        let group = &positions[group_start..group_start + SPREAD_LEN];

        let mut coeffs = [0.0f64; SPREAD_LEN];
        for (k, pos) in group.iter().enumerate() {
            coeffs[k] = flat_get(grid_mut, pos.flat_idx) as f64;
        }

        let delta = embed_delta_fn(bit_idx);
        stdm_embed(&mut coeffs, &vectors[bit_idx], all_bits[bit_idx], delta);

        for (k, pos) in group.iter().enumerate() {
            let new_val = coeffs[k].round() as i16;
            flat_set(grid_mut, pos.flat_idx, new_val);
        }
    }

    // 15. Write modified JPEG.
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
    /// Repetition factor used during decode (1 = Phase 1 / no repetition).
    pub repetition_factor: u8,
    /// RS parity symbols used per block.
    pub parity_len: u16,
}

impl DecodeQuality {
    /// Create quality info for a Ghost decode (binary: always 100% if successful).
    pub fn ghost() -> Self {
        Self {
            mode: super::super::frame::MODE_GHOST,
            rs_errors_corrected: 0,
            rs_error_capacity: 0,
            integrity_percent: 100,
            repetition_factor: 0,
            parity_len: 0,
        }
    }

    /// Create quality info from Armor RS decode stats.
    pub fn from_rs_stats(stats: &ecc::RsDecodeStats) -> Self {
        Self::from_rs_stats_with_phase2(stats, 1, ecc::parity_len() as u16)
    }

    /// Create quality info from Armor RS decode stats with Phase 2 metadata.
    pub fn from_rs_stats_with_phase2(
        stats: &ecc::RsDecodeStats,
        repetition_factor: u8,
        parity_len: u16,
    ) -> Self {
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
            repetition_factor,
            parity_len,
        }
    }
}

/// Decode a text message from a stego JPEG using Armor mode.
///
/// Supports both Phase 1 (fixed RS, no repetition) and Phase 2 (adaptive RS,
/// repetition with soft majority voting) images. Phase detection is automatic
/// via the r-header: Phase 1 images have no header (r reads as 0 or 1),
/// falling back to the original decode path.
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

    // 4. Compute baseline delta.
    let baseline_delta = embedding::compute_delta(&qt.values);

    // 5. Generate spreading vectors for all units.
    let vectors = generate_spreading_vectors(&spread_seed, num_units);

    // 6. Extract r-header using baseline delta (first 56 units).
    let grid = img.dct_grid(0);
    let r = if num_units > R_HEADER_UNITS {
        extract_r_header(grid, positions, &vectors, baseline_delta)
    } else {
        0
    };

    // 7. Decide decode path based on r.
    if r <= 1 {
        // Phase 1 path: extract all bits with baseline delta, progressive RS decode
        return decode_phase1(grid, positions, &vectors, baseline_delta, num_units, passphrase);
    }

    // Phase 2 path: adaptive RS + repetition + soft voting
    decode_phase2(grid, positions, &vectors, &qt.values, baseline_delta, num_units, r, passphrase)
}

/// Extract the r-header from the first 56 embedding units.
///
/// The header is 7 copies of a single byte (r). Uses hard majority voting
/// with baseline delta.
fn extract_r_header(
    grid: &DctGrid,
    positions: &[crate::stego::permute::CoeffPos],
    vectors: &[[f64; SPREAD_LEN]],
    baseline_delta: f64,
) -> u8 {
    // Extract 56 soft LLR values
    let mut header_llrs = [0.0f64; R_HEADER_UNITS];
    for unit_idx in 0..R_HEADER_UNITS {
        let group_start = unit_idx * SPREAD_LEN;
        let group = &positions[group_start..group_start + SPREAD_LEN];

        let mut coeffs = [0.0f64; SPREAD_LEN];
        for (k, pos) in group.iter().enumerate() {
            coeffs[k] = flat_get(grid, pos.flat_idx) as f64;
        }

        header_llrs[unit_idx] = stdm_extract_soft(&coeffs, &vectors[unit_idx], baseline_delta);
    }

    // Soft majority vote across 7 copies of 8 bits
    let mut r_byte = 0u8;
    for bit_pos in 0..8 {
        let mut total_llr = 0.0;
        for copy in 0..R_HEADER_COPIES {
            total_llr += header_llrs[copy * 8 + bit_pos];
        }
        if total_llr < 0.0 {
            r_byte |= 1 << (7 - bit_pos);
        }
    }

    r_byte
}

/// Phase 1 decode path (backward compatible with Phase 1 images).
fn decode_phase1(
    grid: &DctGrid,
    positions: &[crate::stego::permute::CoeffPos],
    vectors: &[[f64; SPREAD_LEN]],
    delta: f64,
    num_units: usize,
    passphrase: &str,
) -> Result<(String, DecodeQuality), StegoError> {
    // Extract all bits via STDM with baseline delta
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

    let extracted_bytes = frame::bits_to_bytes(&extracted_bits);
    let (decoded_frame, rs_stats) = try_rs_decode_frame(&extracted_bytes)?;

    let parsed = frame::parse_frame(&decoded_frame)?;
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

    let text = std::str::from_utf8(&plaintext[..len]).map_err(|_| StegoError::InvalidUtf8)?;
    let quality = DecodeQuality::from_rs_stats(&rs_stats);
    Ok((text.to_string(), quality))
}

/// Phase 2 decode path: adaptive RS + repetition with soft majority voting.
fn decode_phase2(
    grid: &DctGrid,
    positions: &[crate::stego::permute::CoeffPos],
    vectors: &[[f64; SPREAD_LEN]],
    qt_values: &[u16; 64],
    _baseline_delta: f64,
    num_units: usize,
    r: u8,
    passphrase: &str,
) -> Result<(String, DecodeQuality), StegoError> {
    let payload_units = num_units - R_HEADER_UNITS;

    // Compute adaptive delta from r (same formula as encoder)
    let adaptive_delta = embedding::compute_delta_adaptive(qt_values, r as usize);

    // Extract all payload units with soft LLR values
    let mut payload_llrs = Vec::with_capacity(payload_units);
    for unit_idx in R_HEADER_UNITS..num_units {
        let group_start = unit_idx * SPREAD_LEN;
        let group = &positions[group_start..group_start + SPREAD_LEN];

        let mut coeffs = [0.0f64; SPREAD_LEN];
        for (k, pos) in group.iter().enumerate() {
            coeffs[k] = flat_get(grid, pos.flat_idx) as f64;
        }

        payload_llrs.push(stdm_extract_soft(&coeffs, &vectors[unit_idx], adaptive_delta));
    }

    // Compute rs_bit_count from r and payload_units
    let rs_bit_count = payload_units / (r as usize);
    if rs_bit_count == 0 {
        return Err(StegoError::FrameCorrupted);
    }

    // Soft majority vote across r copies
    let voted_bits = repetition::repetition_decode_soft(
        &payload_llrs[..rs_bit_count * (r as usize)],
        rs_bit_count,
    );

    // Convert voted bits to bytes
    let voted_bytes = frame::bits_to_bytes(&voted_bits);

    // Try RS decode at each parity tier until one succeeds
    for &parity in &ecc::PARITY_TIERS {
        if let Some(result) = try_rs_decode_frame_with_parity(&voted_bytes, parity) {
            let (decoded_frame, rs_stats) = result;

            // Parse frame
            if let Ok(parsed) = frame::parse_frame(&decoded_frame) {
                // Decrypt
                if let Ok(plaintext) = crypto::decrypt(
                    &parsed.ciphertext,
                    passphrase,
                    &parsed.salt,
                    &parsed.nonce,
                ) {
                    let len = parsed.plaintext_len as usize;
                    if len <= plaintext.len() {
                        if let Ok(text) = std::str::from_utf8(&plaintext[..len]) {
                            let quality = DecodeQuality::from_rs_stats_with_phase2(
                                &rs_stats,
                                r,
                                parity as u16,
                            );
                            return Ok((text.to_string(), quality));
                        }
                    }
                }
            }
        }
    }

    // If no parity tier worked, fall back to Phase 1 decode as last resort
    // (in case r-header was misread)
    let _baseline_delta2 = embedding::compute_delta(qt_values);
    decode_phase1(grid, positions, vectors, _baseline_delta2, num_units, passphrase)
}

/// Try to RS-decode a frame from extracted bytes using a specific parity length.
///
/// For high-parity tiers (e.g. 240 where k_max=15), the entire frame doesn't
/// fit in one block. We decode the first block to read plaintext_len (2 bytes),
/// compute the total frame length, then do a full multi-block decode.
fn try_rs_decode_frame_with_parity(
    extracted_bytes: &[u8],
    parity: usize,
) -> Option<(Vec<u8>, ecc::RsDecodeStats)> {
    let k_max = 255 - parity;
    // Need at least 2 bytes in the first block for plaintext_len
    let min_data = 2usize.min(k_max);

    for data_len in min_data..=k_max.min(extracted_bytes.len().saturating_sub(parity)) {
        let block_len = data_len + parity;
        if block_len > extracted_bytes.len() {
            break;
        }

        if let Ok((first_block_data, first_errors)) =
            ecc::rs_decode_with_parity(&extracted_bytes[..block_len], data_len, parity)
        {
            if first_block_data.len() >= 2 {
                let pt_len =
                    u16::from_be_bytes([first_block_data[0], first_block_data[1]]) as usize;
                let ct_len = pt_len + 16;
                let total_frame_len = 2 + 16 + 12 + ct_len + 4;

                if total_frame_len > frame::MAX_FRAME_BYTES + 1024 {
                    continue;
                }

                if total_frame_len == data_len {
                    let t_max = parity / 2;
                    let stats = ecc::RsDecodeStats {
                        total_errors: first_errors,
                        error_capacity: t_max,
                        max_block_errors: first_errors,
                        num_blocks: 1,
                    };
                    return Some((first_block_data, stats));
                }

                if total_frame_len < data_len {
                    let t_max = parity / 2;
                    let stats = ecc::RsDecodeStats {
                        total_errors: first_errors,
                        error_capacity: t_max,
                        max_block_errors: first_errors,
                        num_blocks: 1,
                    };
                    return Some((first_block_data[..total_frame_len].to_vec(), stats));
                }

                if total_frame_len > data_len {
                    let rs_encoded_len =
                        ecc::rs_encoded_len_with_parity(total_frame_len, parity);
                    if rs_encoded_len <= extracted_bytes.len() {
                        if let Ok((decoded, stats)) = ecc::rs_decode_blocks_with_parity(
                            &extracted_bytes[..rs_encoded_len],
                            total_frame_len,
                            parity,
                        ) {
                            return Some((decoded, stats));
                        }
                    }
                }
            }
        }
    }

    None
}

/// Try to RS-decode a frame from extracted bytes (Phase 1 path, fixed parity=64).
fn try_rs_decode_frame(extracted_bytes: &[u8]) -> Result<(Vec<u8>, ecc::RsDecodeStats), StegoError> {
    let parity = ecc::parity_len();

    let min_data = super::super::frame::FRAME_OVERHEAD;
    let max_first_block_data = 191usize; // K_DEFAULT

    for data_len in min_data..=max_first_block_data.min(extracted_bytes.len().saturating_sub(parity))
    {
        let block_len = data_len + parity;
        if block_len > extracted_bytes.len() {
            break;
        }

        if let Ok((first_block_data, first_errors)) = ecc::rs_decode(&extracted_bytes[..block_len], data_len) {
            if first_block_data.len() >= 2 {
                let pt_len =
                    u16::from_be_bytes([first_block_data[0], first_block_data[1]]) as usize;
                let ct_len = pt_len + 16;
                let total_frame_len = 2 + 16 + 12 + ct_len + 4;

                if total_frame_len > frame::MAX_FRAME_BYTES + 1024 {
                    continue;
                }

                if total_frame_len == data_len {
                    let stats = ecc::RsDecodeStats {
                        total_errors: first_errors,
                        error_capacity: ecc::T_MAX,
                        max_block_errors: first_errors,
                        num_blocks: 1,
                    };
                    return Ok((first_block_data, stats));
                }

                if total_frame_len <= data_len {
                    let stats = ecc::RsDecodeStats {
                        total_errors: first_errors,
                        error_capacity: ecc::T_MAX,
                        max_block_errors: first_errors,
                        num_blocks: 1,
                    };
                    return Ok((first_block_data[..total_frame_len].to_vec(), stats));
                }

                if total_frame_len > data_len {
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
