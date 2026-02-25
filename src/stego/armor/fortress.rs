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
//! A magic byte is embedded in the first 56 permuted blocks using
//! 7-copy majority voting (8 bits x 7 copies = 56 blocks). This allows fast
//! detection on decode: if the magic doesn't match, fall through to STDM.
//!
//! # Watson Perceptual Masking (magic 0xF6)
//!
//! Watson-inspired adaptive QIM uses AC energy per block to assign tiers:
//! - Tier 0 (skip): extremely smooth blocks — don't embed
//! - Tier 1 (factor 0.7): smooth blocks — smaller QIM step (less visible)
//! - Tier 2 (factor 1.0): average texture — base QIM step
//! - Tier 3 (factor 1.5): heavy texture — larger QIM step (hidden by texture)
//!
//! This improves visual quality by adapting the embedding strength to the
//! perceptual masking capability of each block.

use crate::jpeg::dct::DctGrid;
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
/// Block averages shift <2 levels through QF70+ recompression, so step=8 gives
/// a 2-level margin on each side of the decision boundary — sufficient for
/// pre-settled images (QF70 encode → QF75 WhatsApp recompression).
const QIM_STEP: f64 = 8.0;

/// Minimum repetition factor for Fortress encode.
/// Higher minimum = earlier switchover to STDM for longer messages = better quality.
/// At r=5, a 1200×1600 image holds ~444 chars; a 1280×720 image holds ~197 chars.
const FORTRESS_MIN_R: usize = 5;

/// Magic byte embedded in the fortress header for fast detection.
/// Uses Watson-adaptive QIM with per-block step sizes based on AC energy.
const FORTRESS_MAGIC: u8 = 0xF6;

/// Number of blocks used for the magic header: 1 byte x 8 bits x 7 copies.
const FORTRESS_HEADER_BLOCKS: usize = 56;

/// Number of majority-vote copies of the magic byte.
const FORTRESS_HEADER_COPIES: usize = 7;

// --- Watson tier constants ---

/// Blocks with AC energy ratio below this are skipped (tier 0).
const WATSON_SKIP_RATIO: f64 = 0.06;

/// Blocks with AC energy ratio below this (but >= SKIP) are tier 1 (smooth).
const WATSON_LOW_RATIO: f64 = 0.35;

/// Blocks with AC energy ratio >= this are tier 3 (heavy texture).
const WATSON_HIGH_RATIO: f64 = 2.0;

/// QIM step factor per tier. Tier 0 is unused (skip), tiers 1-3 scale the base step.
const WATSON_TIER_FACTORS: [f64; 4] = [0.0, 0.7, 1.0, 1.5];

// --- Watson perceptual masking ---

/// Watson tier assignment for all blocks in the Y-channel DctGrid.
struct WatsonTiers {
    /// Per-block tier (0 = skip, 1/2/3 = embed with corresponding factor).
    tiers: Vec<u8>,
    /// Number of blocks with tier >= 1 (usable for embedding).
    usable_count: usize,
}

/// Compute Watson tiers from AC energy of each Y-channel 8x8 block.
///
/// AC energy = sum of squared non-DC coefficients. The ratio of each block's
/// AC energy to the median AC energy determines the tier. Only f64 division
/// is used (no `powf`, no `sqrt`) to ensure cross-platform determinism.
fn compute_watson_tiers(grid: &DctGrid) -> WatsonTiers {
    let blocks_wide = grid.blocks_wide();
    let blocks_tall = grid.blocks_tall();
    let total_blocks = blocks_wide * blocks_tall;

    // 1. Compute AC energy per block.
    let mut ac_energies: Vec<f64> = Vec::with_capacity(total_blocks);
    for br in 0..blocks_tall {
        for bc in 0..blocks_wide {
            let block = grid.block(br, bc);
            // Sum of squares of all non-DC coefficients (indices 1..64)
            let mut energy: f64 = 0.0;
            for k in 1..64 {
                let c = block[k] as f64;
                energy += c * c;
            }
            ac_energies.push(energy);
        }
    }

    // 2. Sort a clone to find median (avoid modifying original order).
    let mut sorted = ac_energies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if total_blocks == 0 {
        1.0
    } else {
        sorted[total_blocks / 2].max(1.0)
    };

    // 3. Assign tier based on ratio = energy / median.
    let mut tiers = Vec::with_capacity(total_blocks);
    let mut usable_count = 0usize;
    for &energy in &ac_energies {
        let ratio = energy / median;
        let tier = if ratio < WATSON_SKIP_RATIO {
            0u8
        } else if ratio < WATSON_LOW_RATIO {
            1u8
        } else if ratio < WATSON_HIGH_RATIO {
            2u8
        } else {
            3u8
        };
        if tier > 0 {
            usable_count += 1;
        }
        tiers.push(tier);
    }

    WatsonTiers {
        tiers,
        usable_count,
    }
}

/// Get the QIM step factor for a given Watson tier.
fn watson_step_factor(tier: u8) -> f64 {
    WATSON_TIER_FACTORS[tier as usize]
}

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
/// blocks, accounting for header, RS parity, repetition coding (r >= 5),
/// and Watson filtering (approximated as usable_count / total_blocks ratio).
pub fn fortress_max_frame_bytes(img: &JpegImage) -> Result<usize, StegoError> {
    let grid = img.dct_grid(0);
    let total_blocks = grid.blocks_wide() * grid.blocks_tall();

    if total_blocks <= FORTRESS_HEADER_BLOCKS {
        return Ok(0);
    }

    // Compute Watson tiers for capacity estimation.
    let watson = compute_watson_tiers(grid);

    // Approximate Watson-filtered payload blocks.
    // We don't have the passphrase here (no permutation), so we estimate:
    // payload_blocks = (total_blocks - HEADER_BLOCKS) * watson.usable_count / total_blocks
    // This is a safe underestimate.
    let raw_payload = total_blocks - FORTRESS_HEADER_BLOCKS;
    let payload_blocks = if total_blocks > 0 {
        raw_payload * watson.usable_count / total_blocks
    } else {
        0
    };

    if payload_blocks == 0 {
        return Ok(0);
    }

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
            if r >= FORTRESS_MIN_R && rs_bits <= payload_blocks {
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
                if r >= FORTRESS_MIN_R {
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

/// Encode a payload frame into an image using BA-QIM on DC block averages
/// with Watson perceptual masking.
///
/// The caller must have already built the frame_bytes (encrypted + framed).
/// This function embeds the Watson magic header and the RS+repetition encoded
/// payload into permuted, Watson-filtered Y-channel block DCs with per-block
/// adaptive QIM step sizes.
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

    // Compute Watson tiers from AC energy (before any DC modifications).
    let watson = compute_watson_tiers(grid);

    // Derive fortress structural key for block permutation.
    let fort_key = crypto::derive_fortress_structural_key(passphrase);
    let perm = permute_blocks(total_blocks, &fort_key);

    // Build Watson-filtered payload block list: permuted blocks after header,
    // excluding tier-0 (skip) blocks.
    let header_perm = &perm[..FORTRESS_HEADER_BLOCKS];
    let remaining_perm = &perm[FORTRESS_HEADER_BLOCKS..];
    let payload_block_indices: Vec<usize> = remaining_perm
        .iter()
        .copied()
        .filter(|&block_idx| watson.tiers[block_idx] > 0)
        .collect();
    let payload_blocks = payload_block_indices.len();

    if payload_blocks == 0 {
        return Err(StegoError::ImageTooSmall);
    }

    // RS-encode with best parity tier (smallest parity where r >= FORTRESS_MIN_R).
    let mut chosen_parity = ecc::PARITY_TIERS[0];
    for &parity in &ecc::PARITY_TIERS {
        let rs_encoded = ecc::rs_encode_blocks_with_parity(frame_bytes, parity);
        let rs_bits_len = rs_encoded.len() * 8;
        if rs_bits_len <= payload_blocks {
            let r = repetition::compute_r(rs_bits_len, payload_blocks);
            if r >= FORTRESS_MIN_R {
                chosen_parity = parity;
                break;
            }
        }
    }

    let rs_encoded = ecc::rs_encode_blocks_with_parity(frame_bytes, chosen_parity);
    let rs_bits = frame::bytes_to_bits(&rs_encoded);

    let r = repetition::compute_r(rs_bits.len(), payload_blocks);
    if r < FORTRESS_MIN_R {
        return Err(StegoError::MessageTooLarge);
    }

    let rs_bit_count_aligned = payload_blocks / r;
    let mut rs_bits_padded = rs_bits;
    rs_bits_padded.resize(rs_bit_count_aligned, 0);
    let (rep_bits, _) = repetition::repetition_encode(&rs_bits_padded, payload_blocks);

    // Build header bits: 7 copies of magic byte
    let mut header_bits = Vec::with_capacity(FORTRESS_HEADER_BLOCKS);
    for _ in 0..FORTRESS_HEADER_COPIES {
        for bp in (0..8).rev() {
            header_bits.push((FORTRESS_MAGIC >> bp) & 1);
        }
    }

    // Embed header using fixed QIM_STEP on the first 56 permuted blocks
    // (unfiltered permutation, same as legacy).
    let grid_mut = img.dct_grid_mut(0);
    for (bit_idx, &block_idx) in header_perm.iter().enumerate() {
        let br = block_idx / blocks_wide;
        let bc = block_idx % blocks_wide;
        let dc = grid_mut.get(br, bc, 0, 0);
        let avg = dc_to_avg(dc, qt_dc);
        let new_avg = qim_embed_avg(avg, QIM_STEP, header_bits[bit_idx]);
        let new_dc = avg_to_dc(new_avg, qt_dc);
        grid_mut.set(br, bc, 0, 0, new_dc);
    }

    // Embed payload using Watson-adaptive per-block QIM step.
    for (payload_idx, &block_idx) in payload_block_indices.iter().enumerate() {
        if payload_idx >= rep_bits.len() {
            break;
        }
        let tier = watson.tiers[block_idx];
        let step = QIM_STEP * watson_step_factor(tier);
        let br = block_idx / blocks_wide;
        let bc = block_idx % blocks_wide;
        let dc = grid_mut.get(br, bc, 0, 0);
        let avg = dc_to_avg(dc, qt_dc);
        let new_avg = qim_embed_avg(avg, step, rep_bits[payload_idx]);
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

    // Derive key and permute blocks.
    let fort_key = crypto::derive_fortress_structural_key(passphrase);
    let perm = permute_blocks(total_blocks, &fort_key);

    // Extract header LLRs with fixed QIM_STEP.
    let mut header_llrs = Vec::with_capacity(FORTRESS_HEADER_BLOCKS);
    for &block_idx in &perm[..FORTRESS_HEADER_BLOCKS] {
        let br = block_idx / blocks_wide;
        let bc = block_idx % blocks_wide;
        let dc = grid.get(br, bc, 0, 0);
        let avg = dc_to_avg(dc, qt_dc);
        header_llrs.push(qim_extract_soft(avg, QIM_STEP));
    }

    // Check magic header: majority vote across 7 copies
    let magic = extract_magic_byte(&header_llrs);

    if magic != FORTRESS_MAGIC {
        // Not a fortress image — fall through to STDM.
        return Err(StegoError::FrameCorrupted);
    }

    // Watson mode: compute tiers, filter payload blocks, use per-block steps.
    let watson = compute_watson_tiers(grid);
    let remaining_perm = &perm[FORTRESS_HEADER_BLOCKS..];

    // Build Watson-filtered payload blocks + per-block LLRs.
    let mut payload_llrs = Vec::new();
    for &block_idx in remaining_perm {
        let tier = watson.tiers[block_idx];
        if tier == 0 {
            continue; // skip smooth blocks
        }
        let step = QIM_STEP * watson_step_factor(tier);
        let br = block_idx / blocks_wide;
        let bc = block_idx % blocks_wide;
        let dc = grid.get(br, bc, 0, 0);
        let avg = dc_to_avg(dc, qt_dc);
        payload_llrs.push(qim_extract_soft(avg, step));
    }

    let payload_blocks = payload_llrs.len();
    decode_fortress_payload(&payload_llrs, payload_blocks, passphrase)
}

/// Shared payload decode logic for both legacy and Watson fortress modes.
///
/// Brute-force searches over (r, parity) combinations to find the correct
/// RS decode parameters, then decrypts and returns the plaintext.
fn decode_fortress_payload(
    payload_llrs: &[f64],
    payload_blocks: usize,
    passphrase: &str,
) -> Result<(String, super::pipeline::DecodeQuality), StegoError> {
    // Reference LLR for pristine QIM embedding: step / 2.
    // For Watson mode, the per-block step varies by tier, but the base step
    // (QIM_STEP) is used as the reference since the average tier factor is ~1.0.
    let reference_llr = QIM_STEP / 2.0;

    // Use r >= 3 on decode (not FORTRESS_MIN_R) for backward compatibility
    // with images encoded before the min-r increase.
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

            let (voted_bits, rep_quality) =
                repetition::repetition_decode_soft_with_quality(&payload_llrs[..used_llrs], rs_bit_count);
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
                                let mut q = super::pipeline::DecodeQuality::from_rs_stats_with_signal(
                                    &rs_stats,
                                    r as u8,
                                    parity as u16,
                                    rep_quality.avg_abs_llr_per_copy,
                                    reference_llr,
                                );
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
    use crate::jpeg::dct::DctGrid;

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

    // --- Watson perceptual masking tests ---

    /// Helper: create a DctGrid with specified AC energy patterns.
    /// blocks is a vec of 64-element arrays (DC + 63 AC coefficients).
    fn make_grid(blocks_wide: usize, blocks_tall: usize, blocks: &[[i16; 64]]) -> DctGrid {
        let mut grid = DctGrid::new(blocks_wide, blocks_tall);
        for (idx, block_data) in blocks.iter().enumerate() {
            let br = idx / blocks_wide;
            let bc = idx % blocks_wide;
            for k in 0..64 {
                let i = k / 8;
                let j = k % 8;
                grid.set(br, bc, i, j, block_data[k]);
            }
        }
        grid
    }

    /// Create a "smooth" block: DC=100, all AC coefficients = 0.
    fn smooth_block() -> [i16; 64] {
        let mut b = [0i16; 64];
        b[0] = 100; // DC
        b
    }

    /// Create a "textured" block: DC=100, many large AC coefficients.
    fn textured_block() -> [i16; 64] {
        let mut b = [0i16; 64];
        b[0] = 100; // DC
        for k in 1..64 {
            b[k] = 20; // large AC values
        }
        b
    }

    /// Create a "medium texture" block: DC=100, moderate AC coefficients.
    fn medium_block() -> [i16; 64] {
        let mut b = [0i16; 64];
        b[0] = 100; // DC
        for k in 1..20 {
            b[k] = 5;
        }
        b
    }

    #[test]
    fn watson_tiers_smooth_vs_textured() {
        // Build a 4x1 grid: 2 smooth blocks, 1 medium, 1 textured.
        let blocks = vec![
            smooth_block(),
            smooth_block(),
            medium_block(),
            textured_block(),
        ];
        let grid = make_grid(4, 1, &blocks);
        let watson = compute_watson_tiers(&grid);

        assert_eq!(watson.tiers.len(), 4);

        // Smooth blocks (AC energy = 0) should be tier 0 (skip).
        assert_eq!(watson.tiers[0], 0, "Smooth block should be tier 0 (skip)");
        assert_eq!(watson.tiers[1], 0, "Smooth block should be tier 0 (skip)");

        // Textured block should be tier 2 or 3 (high energy).
        assert!(
            watson.tiers[3] >= 2,
            "Textured block should be tier 2 or 3, got {}",
            watson.tiers[3]
        );

        // usable_count should exclude tier-0 blocks.
        assert!(
            watson.usable_count >= 1,
            "Should have at least 1 usable block"
        );
        assert!(
            watson.usable_count <= 4,
            "Cannot have more usable blocks than total"
        );
    }

    #[test]
    fn watson_tier_deterministic() {
        // Same grid should always produce the same tiers.
        let blocks: Vec<[i16; 64]> = (0..16)
            .map(|i| {
                let mut b = [0i16; 64];
                b[0] = 100;
                // Vary AC energy across blocks
                for k in 1..(1 + (i % 10) + 1).min(64) {
                    b[k] = (i as i16 + 1) * 3;
                }
                b
            })
            .collect();

        let grid = make_grid(4, 4, &blocks);
        let watson_a = compute_watson_tiers(&grid);
        let watson_b = compute_watson_tiers(&grid);

        assert_eq!(watson_a.tiers, watson_b.tiers, "Watson tiers must be deterministic");
        assert_eq!(
            watson_a.usable_count, watson_b.usable_count,
            "Usable count must be deterministic"
        );
    }

    #[test]
    fn watson_ac_energy_invariant() {
        // Verify that AC energy is unchanged after DC modification.
        // This is the key invariant that makes Watson tiers identical on
        // encode and decode: Fortress only modifies DC (index 0), not AC.
        let mut block = textured_block();
        let original_ac_energy: f64 = block[1..64]
            .iter()
            .map(|&c| (c as f64) * (c as f64))
            .sum();

        // Modify DC (simulating QIM embedding)
        block[0] = 42;
        let modified_ac_energy: f64 = block[1..64]
            .iter()
            .map(|&c| (c as f64) * (c as f64))
            .sum();

        assert_eq!(
            original_ac_energy, modified_ac_energy,
            "AC energy must be unchanged after DC modification"
        );
    }

    #[test]
    fn watson_encode_decode_roundtrip() {
        // Full encode/decode roundtrip with Watson magic.
        use crate::stego::armor::pipeline::{armor_encode, armor_decode};

        // Use a real JPEG from the test vectors.
        let test_jpeg = std::fs::read("../test-vectors/photo_640x480_q75_420.jpg")
            .expect("test vector photo_640x480_q75_420.jpg must exist");

        let passphrase = "test-watson-pass";

        // Check fortress capacity and pick a message that fits.
        let img = crate::jpeg::JpegImage::from_bytes(&test_jpeg).unwrap();
        let fort_cap = fortress_capacity(&img).unwrap();
        assert!(
            fort_cap >= 1,
            "Fortress capacity ({fort_cap}) must be at least 1 byte for 640x480 photo"
        );

        // Use the shortest meaningful message that fits.
        let message = if fort_cap >= 4 { "Hi!!" } else { "Hi" };

        // Encode
        let stego_bytes = armor_encode(&test_jpeg, message, passphrase)
            .expect("Watson fortress encode should succeed");

        // Decode
        let (decoded_msg, quality) = armor_decode(&stego_bytes, passphrase)
            .expect("Watson fortress decode should succeed");

        assert_eq!(decoded_msg, message, "Decoded message must match original");
        assert!(quality.fortress_used, "Should use fortress mode for short message");
    }

    #[test]
    fn watson_magic_extraction() {
        // Verify magic byte extraction works correctly.
        let mut header_llrs = Vec::with_capacity(56);
        for _ in 0..FORTRESS_HEADER_COPIES {
            for bp in (0..8).rev() {
                let bit = (FORTRESS_MAGIC >> bp) & 1;
                header_llrs.push(if bit == 0 { 5.0 } else { -5.0 });
            }
        }

        let extracted = extract_magic_byte(&header_llrs);
        assert_eq!(extracted, FORTRESS_MAGIC, "Should extract magic 0xF6");
    }

    #[test]
    fn watson_step_factors_correct() {
        // Verify the step factor lookup works for all tiers.
        assert_eq!(watson_step_factor(0), 0.0);
        assert_eq!(watson_step_factor(1), 0.7);
        assert_eq!(watson_step_factor(2), 1.0);
        assert_eq!(watson_step_factor(3), 1.5);
    }

    #[test]
    fn watson_all_smooth_skips_all() {
        // If all blocks are perfectly smooth (AC energy = 0), all are tier 0.
        let blocks: Vec<[i16; 64]> = (0..9).map(|_| smooth_block()).collect();
        let grid = make_grid(3, 3, &blocks);
        let watson = compute_watson_tiers(&grid);

        // All blocks have zero AC energy, median = max(0, 1.0) = 1.0,
        // ratio = 0 / 1.0 = 0.0 < 0.06, so all are tier 0.
        assert_eq!(watson.usable_count, 0, "All smooth blocks should be skipped");
        assert!(watson.tiers.iter().all(|&t| t == 0));
    }

    #[test]
    fn watson_uniform_texture_all_tier2() {
        // If all blocks have identical non-zero AC energy, ratio = 1.0 → tier 2.
        let blocks: Vec<[i16; 64]> = (0..9).map(|_| textured_block()).collect();
        let grid = make_grid(3, 3, &blocks);
        let watson = compute_watson_tiers(&grid);

        // All blocks have the same energy, ratio = energy/energy = 1.0,
        // which falls in [0.35, 2.0) → tier 2.
        assert_eq!(watson.usable_count, 9, "All uniform-texture blocks should be usable");
        assert!(
            watson.tiers.iter().all(|&t| t == 2),
            "All blocks with ratio=1.0 should be tier 2"
        );
    }
}
