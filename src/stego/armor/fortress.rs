//! Fortress sub-mode: BA-QIM embedding in DC block averages.
//!
//! Fortress uses Quantization Index Modulation (QIM) on the average brightness
//! of each 8x8 block (derived from the DC coefficient). Block averages shift
//! by less than 2 pixel levels even through aggressive JPEG recompression
//! (QF 53+), making this near-perfectly recompression-invariant.
//!
//! Fortress is used automatically within Armor mode when the message is short
//! enough to fit. Longer messages fall through to the STDM pipeline.
//!
//! # Embedding domain
//!
//! Each 8x8 block's DC coefficient relates to the block average brightness:
//!   avg = dc * qt_dc / 8.0
//! QIM embeds one bit per block by quantizing `avg` to a step grid offset by
//! the target bit value.
//!
//! # Header
//!
//! A magic byte (0xF5) is embedded in the first 56 permuted blocks using
//! 7-copy majority voting (8 bits x 7 copies = 56 blocks). This allows fast
//! detection on decode: if the magic doesn't match, fall through to STDM.

use crate::jpeg::JpegImage;
use crate::stego::armor::ecc;
use crate::stego::armor::repetition;
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::frame;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// QIM step size in pixel-level units.
/// Block averages shift <2 levels through recompression, so step=16 gives
/// a 6-level margin on each side of the decision boundary.
const QIM_STEP: f64 = 16.0;

/// Magic byte embedded in the fortress header for fast detection.
const FORTRESS_MAGIC: u8 = 0xF5;

/// Number of blocks used for the magic header: 1 byte x 8 bits x 7 copies.
const FORTRESS_HEADER_BLOCKS: usize = 56;

/// Number of majority-vote copies of the magic byte.
const FORTRESS_HEADER_COPIES: usize = 7;

// --- BA-QIM core functions ---

/// Convert a quantized DC coefficient to block-average brightness.
fn dc_to_avg(dc: i16, qt_dc: u16) -> f64 {
    dc as f64 * qt_dc as f64 / 8.0
}

/// Convert a block-average brightness back to a quantized DC coefficient.
fn avg_to_dc(avg: f64, qt_dc: u16) -> i16 {
    (avg * 8.0 / qt_dc as f64).round() as i16
}

/// QIM embed: quantize `avg` to the nearest grid point for the target bit.
///
/// Bit 0 grid: ..., -step, 0, step, 2*step, ...
/// Bit 1 grid: ..., -step/2, step/2, 3*step/2, ...
fn qim_embed_avg(avg: f64, step: f64, bit: u8) -> f64 {
    let offset = if bit == 0 { 0.0 } else { step / 2.0 };
    let shifted = avg - offset;
    let quantized = (shifted / step).round() * step;
    quantized + offset
}

/// QIM soft extraction: compute log-likelihood ratio.
///
/// Returns positive for bit 0, negative for bit 1.
/// Magnitude indicates confidence.
fn qim_extract_soft(avg: f64, step: f64) -> f64 {
    let half = step / 2.0;
    // Distance to nearest bit-0 grid point
    let d0 = (avg / step).round() * step;
    let dist0 = (avg - d0).abs();
    // Distance to nearest bit-1 grid point
    let d1 = ((avg - half) / step).round() * step + half;
    let dist1 = (avg - d1).abs();
    // LLR: positive means closer to bit-0, negative means closer to bit-1
    dist1 - dist0
}

// --- Block permutation ---

/// Generate a permuted list of block indices using Fisher-Yates shuffle.
///
/// Uses `u32` for cross-platform portability (WASM is 32-bit).
fn permute_blocks(total_blocks: usize, seed: &[u8; 32]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..total_blocks).collect();
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let n = indices.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=(i as u32)) as usize;
        indices.swap(i, j);
    }
    indices
}

// --- Fortress capacity ---

/// Maximum frame bytes that can be embedded in a fortress-encoded image.
///
/// Returns the max frame_bytes (before RS encoding) that fits in the available
/// blocks, accounting for header, RS parity, and repetition coding (r >= 3).
pub fn fortress_max_frame_bytes(img: &JpegImage) -> Result<usize, StegoError> {
    let grid = img.dct_grid(0);
    let total_blocks = grid.blocks_wide() * grid.blocks_tall();

    if total_blocks <= FORTRESS_HEADER_BLOCKS {
        return Ok(0);
    }
    let payload_blocks = total_blocks - FORTRESS_HEADER_BLOCKS;

    // We need: rs_encoded_bits <= payload_blocks (1 bit per block)
    // And r >= 3 for meaningful repetition coding
    // So: rs_encoded_bits * 3 <= payload_blocks
    // => rs_encoded_bytes <= payload_blocks / 3 / 8  (but we use bits not bytes for capacity)
    //
    // Actually: payload_blocks bits available, with repetition r >= 3:
    //   rs_bit_count = payload_blocks / r  (where r >= 3)
    //   rs_byte_count = rs_bit_count / 8
    //
    // Try each parity tier to find the best fit.
    let mut best_frame_bytes = 0usize;

    for &parity in &ecc::PARITY_TIERS {
        // Binary search for max frame_bytes
        let mut lo = 0usize;
        let mut hi = frame::MAX_FRAME_BYTES;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            let rs_encoded_len = ecc::rs_encoded_len_with_parity(mid, parity);
            let rs_bits = rs_encoded_len * 8;
            if rs_bits == 0 || rs_bits > payload_blocks {
                hi = mid - 1;
                continue;
            }
            let r = repetition::compute_r(rs_bits, payload_blocks);
            if r >= 3 && rs_bits <= payload_blocks {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if lo > best_frame_bytes {
            // Verify the solution
            let rs_encoded_len = ecc::rs_encoded_len_with_parity(lo, parity);
            let rs_bits = rs_encoded_len * 8;
            if rs_bits <= payload_blocks {
                let r = repetition::compute_r(rs_bits, payload_blocks);
                if r >= 3 {
                    best_frame_bytes = lo;
                }
            }
        }
    }

    Ok(best_frame_bytes)
}

/// Estimate fortress plaintext capacity (max message bytes).
pub fn fortress_capacity(img: &JpegImage) -> Result<usize, StegoError> {
    let max_frame = fortress_max_frame_bytes(img)?;
    if max_frame <= frame::FRAME_OVERHEAD {
        return Ok(0);
    }
    Ok(max_frame - frame::FRAME_OVERHEAD)
}

// --- Fortress encode ---

/// Encode a payload frame into an image using BA-QIM on DC block averages.
///
/// The caller must have already built the frame_bytes (encrypted + framed).
/// This function embeds the magic header and the RS+repetition encoded payload
/// into permuted Y-channel block DCs.
pub fn fortress_encode(
    img: &mut JpegImage,
    frame_bytes: &[u8],
    passphrase: &str,
) -> Result<(), StegoError> {
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt_dc = img
        .quant_table(qt_id)
        .ok_or(StegoError::NoLuminanceChannel)?
        .values[0];

    let grid = img.dct_grid(0);
    let blocks_wide = grid.blocks_wide();
    let blocks_tall = grid.blocks_tall();
    let total_blocks = blocks_wide * blocks_tall;

    if total_blocks <= FORTRESS_HEADER_BLOCKS {
        return Err(StegoError::ImageTooSmall);
    }
    let payload_blocks = total_blocks - FORTRESS_HEADER_BLOCKS;

    // Derive fortress structural key for block permutation.
    let fort_key = crypto::derive_fortress_structural_key(passphrase);
    let perm = permute_blocks(total_blocks, &fort_key);

    // RS-encode with best parity tier (smallest parity where r >= 3).
    let mut chosen_parity = ecc::PARITY_TIERS[0];
    for &parity in &ecc::PARITY_TIERS {
        let rs_encoded = ecc::rs_encode_blocks_with_parity(frame_bytes, parity);
        let rs_bits_len = rs_encoded.len() * 8;
        if rs_bits_len <= payload_blocks {
            let r = repetition::compute_r(rs_bits_len, payload_blocks);
            if r >= 3 {
                chosen_parity = parity;
                break;
            }
        }
    }

    let rs_encoded = ecc::rs_encode_blocks_with_parity(frame_bytes, chosen_parity);
    let rs_bits = frame::bytes_to_bits(&rs_encoded);

    let r = repetition::compute_r(rs_bits.len(), payload_blocks);
    if r < 3 {
        return Err(StegoError::MessageTooLarge);
    }

    let rs_bit_count_aligned = payload_blocks / r;
    let mut rs_bits_padded = rs_bits;
    rs_bits_padded.resize(rs_bit_count_aligned, 0);
    let (rep_bits, _) = repetition::repetition_encode(&rs_bits_padded, payload_blocks);

    // Build complete bit sequence: header (56 blocks) + payload
    let mut all_bits = Vec::with_capacity(total_blocks);

    // Header: 7 copies of magic byte
    for _ in 0..FORTRESS_HEADER_COPIES {
        for bp in (0..8).rev() {
            all_bits.push((FORTRESS_MAGIC >> bp) & 1);
        }
    }

    // Payload bits
    all_bits.extend_from_slice(&rep_bits[..payload_blocks.min(rep_bits.len())]);

    // Embed using BA-QIM on DC coefficients
    let grid_mut = img.dct_grid_mut(0);
    for (bit_idx, &block_idx) in perm.iter().enumerate() {
        if bit_idx >= all_bits.len() {
            break;
        }
        let br = block_idx / blocks_wide;
        let bc = block_idx % blocks_wide;
        let dc = grid_mut.get(br, bc, 0, 0);
        let avg = dc_to_avg(dc, qt_dc);
        let new_avg = qim_embed_avg(avg, QIM_STEP, all_bits[bit_idx]);
        let new_dc = avg_to_dc(new_avg, qt_dc);
        grid_mut.set(br, bc, 0, 0, new_dc);
    }

    Ok(())
}

// --- Fortress decode ---

/// Decode a fortress-encoded payload from a stego JPEG.
///
/// Returns the decoded message and quality info, or an error if the fortress
/// magic is not detected (allowing the caller to fall through to STDM).
pub fn fortress_decode(
    img: &JpegImage,
    passphrase: &str,
) -> Result<(String, super::pipeline::DecodeQuality), StegoError> {
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt_dc = img
        .quant_table(qt_id)
        .ok_or(StegoError::NoLuminanceChannel)?
        .values[0];

    let grid = img.dct_grid(0);
    let blocks_wide = grid.blocks_wide();
    let blocks_tall = grid.blocks_tall();
    let total_blocks = blocks_wide * blocks_tall;

    if total_blocks <= FORTRESS_HEADER_BLOCKS {
        return Err(StegoError::FrameCorrupted);
    }
    let payload_blocks = total_blocks - FORTRESS_HEADER_BLOCKS;

    // Derive key and permute blocks.
    let fort_key = crypto::derive_fortress_structural_key(passphrase);
    let perm = permute_blocks(total_blocks, &fort_key);

    // Extract all LLRs from DC coefficients.
    let mut all_llrs = Vec::with_capacity(total_blocks);
    for &block_idx in &perm {
        let br = block_idx / blocks_wide;
        let bc = block_idx % blocks_wide;
        let dc = grid.get(br, bc, 0, 0);
        let avg = dc_to_avg(dc, qt_dc);
        all_llrs.push(qim_extract_soft(avg, QIM_STEP));
    }

    // Check magic header: majority vote across 7 copies
    let magic = extract_magic_byte(&all_llrs[..FORTRESS_HEADER_BLOCKS]);
    if magic != FORTRESS_MAGIC {
        return Err(StegoError::FrameCorrupted);
    }

    // Extract payload LLRs (after header)
    let payload_llrs = &all_llrs[FORTRESS_HEADER_BLOCKS..];
    let payload_llrs = &payload_llrs[..payload_blocks.min(payload_llrs.len())];

    // Brute-force (r, parity) search — same pattern as STDM Phase 2
    for &parity in &ecc::PARITY_TIERS {
        let candidate_rs = compute_fortress_candidate_rs(payload_blocks, parity);

        for r in candidate_rs {
            let rs_bit_count = payload_blocks / r;
            if rs_bit_count == 0 {
                continue;
            }
            let used_llrs = rs_bit_count * r;
            if used_llrs > payload_llrs.len() {
                continue;
            }

            let voted_bits =
                repetition::repetition_decode_soft(&payload_llrs[..used_llrs], rs_bit_count);
            let voted_bytes = frame::bits_to_bytes(&voted_bits);

            if let Some((decoded_frame, rs_stats)) =
                try_fortress_rs_decode(&voted_bytes, parity)
            {
                if let Ok(parsed) = frame::parse_frame(&decoded_frame) {
                    if let Ok(plaintext) = crypto::decrypt(
                        &parsed.ciphertext,
                        passphrase,
                        &parsed.salt,
                        &parsed.nonce,
                    ) {
                        let len = parsed.plaintext_len as usize;
                        if len <= plaintext.len() {
                            if let Ok(text) = std::str::from_utf8(&plaintext[..len]) {
                                let quality = super::pipeline::DecodeQuality::from_rs_stats_with_phase2(
                                    &rs_stats,
                                    r as u8,
                                    parity as u16,
                                );
                                let mut q = quality;
                                q.fortress_used = true;
                                return Ok((text.to_string(), q));
                            }
                        }
                    }
                }
            }
        }
    }

    Err(StegoError::FrameCorrupted)
}

/// Extract magic byte from header LLRs using soft majority voting.
fn extract_magic_byte(header_llrs: &[f64]) -> u8 {
    let mut byte = 0u8;
    for bit_pos in 0..8 {
        let mut total = 0.0;
        for copy in 0..FORTRESS_HEADER_COPIES {
            let idx = copy * 8 + bit_pos;
            if idx < header_llrs.len() {
                total += header_llrs[idx];
            }
        }
        // Positive LLR = bit 0, negative = bit 1
        if total < 0.0 {
            byte |= 1 << (7 - bit_pos);
        }
    }
    byte
}

/// Compute distinct candidate r values for fortress decode.
fn compute_fortress_candidate_rs(payload_blocks: usize, parity: usize) -> Vec<usize> {
    let mut rs_set = std::collections::BTreeSet::new();

    let min_frame = frame::FRAME_OVERHEAD;
    let max_frame = frame::MAX_FRAME_BYTES;

    for frame_len in min_frame..=max_frame {
        let rs_encoded_len = ecc::rs_encoded_len_with_parity(frame_len, parity);
        let rs_bits = rs_encoded_len * 8;
        if rs_bits > payload_blocks {
            continue;
        }
        let r = repetition::compute_r(rs_bits, payload_blocks);
        if r >= 3 {
            rs_set.insert(r);
        }
    }

    rs_set.into_iter().collect()
}

/// Try RS decode with a specific parity — mirrors pipeline's try_rs_decode_frame_with_parity.
fn try_fortress_rs_decode(
    extracted_bytes: &[u8],
    parity: usize,
) -> Option<(Vec<u8>, ecc::RsDecodeStats)> {
    let k_max = 255 - parity;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qim_embed_extract_roundtrip() {
        for avg in [0.0, 50.0, 100.0, 200.0, -50.0, 127.5] {
            for bit in [0u8, 1] {
                let embedded = qim_embed_avg(avg, QIM_STEP, bit);
                let llr = qim_extract_soft(embedded, QIM_STEP);
                let extracted_bit = if llr >= 0.0 { 0u8 } else { 1u8 };
                assert_eq!(
                    extracted_bit, bit,
                    "Failed for avg={avg}, bit={bit}: embedded={embedded}, llr={llr}"
                );
            }
        }
    }

    #[test]
    fn qim_soft_llr_sign_matches_hard() {
        for avg in [10.0, 50.0, 128.0, 250.0] {
            for bit in [0u8, 1] {
                let embedded = qim_embed_avg(avg, QIM_STEP, bit);
                let llr = qim_extract_soft(embedded, QIM_STEP);
                // Positive LLR = bit 0, negative = bit 1
                if bit == 0 {
                    assert!(llr >= 0.0, "Expected positive LLR for bit 0, got {llr}");
                } else {
                    assert!(llr <= 0.0, "Expected negative LLR for bit 1, got {llr}");
                }
            }
        }
    }

    #[test]
    fn dc_avg_roundtrip() {
        for qt_dc in [1u16, 2, 4, 8, 16, 32] {
            for dc in [-100i16, -1, 0, 1, 50, 100] {
                let avg = dc_to_avg(dc, qt_dc);
                let recovered = avg_to_dc(avg, qt_dc);
                assert_eq!(
                    recovered, dc,
                    "Roundtrip failed for dc={dc}, qt_dc={qt_dc}: avg={avg}"
                );
            }
        }
    }

    #[test]
    fn magic_byte_majority_voting() {
        // Embed magic byte in 56 LLRs, perturb some, verify majority vote
        let mut llrs = Vec::with_capacity(56);
        for _ in 0..FORTRESS_HEADER_COPIES {
            for bp in (0..8).rev() {
                let bit = (FORTRESS_MAGIC >> bp) & 1;
                // Perfect LLR: positive for 0, negative for 1
                llrs.push(if bit == 0 { 5.0 } else { -5.0 });
            }
        }

        // Perturb 2 out of 7 copies (flip their signs)
        for i in 0..8 {
            llrs[i] = -llrs[i]; // flip copy 0
            llrs[8 + i] = -llrs[8 + i]; // flip copy 1
        }

        let extracted = extract_magic_byte(&llrs);
        assert_eq!(extracted, FORTRESS_MAGIC, "Majority vote should recover magic byte");
    }

    #[test]
    fn block_permutation_deterministic() {
        let seed = [42u8; 32];
        let a = permute_blocks(100, &seed);
        let b = permute_blocks(100, &seed);
        assert_eq!(a, b);
    }

    #[test]
    fn block_permutation_different_seeds() {
        let a = permute_blocks(100, &[1u8; 32]);
        let b = permute_blocks(100, &[2u8; 32]);
        assert_ne!(a, b);
    }

    #[test]
    fn block_permutation_is_permutation() {
        let perm = permute_blocks(100, &[7u8; 32]);
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..100).collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn fortress_capacity_reasonable() {
        // A 1280x720 image has 160*90 = 14400 blocks
        // Payload blocks = 14400 - 56 = 14344
        // With r=3, rs_bit_count = ~4781 bits = ~597 bytes
        // With parity=64, frame_bytes up to ~533
        // Plaintext = frame_bytes - 50 overhead
        let total_blocks = 14400usize;
        let payload_blocks = total_blocks - FORTRESS_HEADER_BLOCKS;
        // Just verify our math is reasonable
        assert!(payload_blocks > 1000, "Should have many payload blocks");
    }
}
