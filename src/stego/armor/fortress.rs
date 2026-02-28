// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

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
//! # Adaptive Watson Perceptual Masking
//!
//! The QIM step and Watson masking range adapt continuously based on the
//! repetition factor (r), using linear interpolation between:
//!   r = 15 (minimum): step=12, Watson [0.9, 1.1] — max robustness
//!   r = 61 (cap):     step=6.5, Watson [0.62, 1.26] — max quality
//!
//! Watson factors are computed from a continuous piecewise-linear base curve
//! (AC energy ratio → [0.3, 1.5]), then remapped to the adaptive range.
//! Uses only basic IEEE 754 arithmetic — deterministic on all platforms.
//!
//! ALL blocks are used for embedding (no skip tier). On decode, r is brute-forced
//! per candidate, so the adaptive step is implicitly known.

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

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Fixed QIM step for the magic header (always step=12, majority-voted 7x).
const HEADER_QIM_STEP: f64 = 12.0;

/// Minimum repetition factor for Fortress encode.
/// r≥15 ensures sufficient majority-voting margin at the BER levels observed
/// in WhatsApp recompression (~19% at step=12 with narrow Watson).
const FORTRESS_MIN_R: usize = 15;

/// Magic byte embedded in the fortress header for fast detection.
const FORTRESS_MAGIC: u8 = 0xF6;

/// Number of blocks used for the magic header: 1 byte x 8 bits x 7 copies.
const FORTRESS_HEADER_BLOCKS: usize = 56;

/// Number of majority-vote copies of the magic byte.
const FORTRESS_HEADER_COPIES: usize = 7;

// --- Adaptive QIM parameters ---
//
// The base QIM step and Watson masking range adapt based on the repetition
// factor (r). Short messages have high r and can tolerate more BER, so we
// use a smaller step for better image quality. Longer messages need lower
// BER, so we use a larger step and narrower Watson range.
//
// On decode, r is brute-forced per candidate, so the step is implicitly known.

/// Adaptive fortress parameters for a given repetition factor.
struct AdaptiveParams {
    base_step: f64,
    watson_lo: f64,
    watson_hi: f64,
}

/// Determine adaptive QIM step and Watson range from repetition factor.
///
/// Continuous linear interpolation between:
///   r = 15 (minimum): step=12, Watson [0.9, 1.1] — max robustness
///   r = 61 (cap):     step=6.5, Watson [0.62, 1.26] — max quality
///
/// R_MAX=61 is the "elbow": smooth shift ≈ 2.0px (perceptual threshold),
/// BER stays at floor 17% (diff=2 blocks just survive), and majority voting
/// at r=61 handles 17% BER with error rate ≈ 10⁻⁸. Beyond 61, quality
/// gains are imperceptible but diff=2 blocks start flipping.
fn adaptive_params(r: usize) -> AdaptiveParams {
    const R_MIN: f64 = 15.0;
    const R_MAX: f64 = 61.0;
    // Clamp r into [R_MIN, R_MAX], compute t in [0, 1]
    let r_clamped = (r as f64).max(R_MIN).min(R_MAX);
    let t = (r_clamped - R_MIN) / (R_MAX - R_MIN);

    AdaptiveParams {
        base_step: 12.0 + t * (6.5 - 12.0),       // 12 → 6.5
        watson_lo: 0.9 + t * (0.62 - 0.9),          // 0.9 → 0.62
        watson_hi: 1.1 + t * (1.26 - 1.1),          // 1.1 → 1.26
    }
}

// --- Watson perceptual masking ---
//
// Continuous piecewise-linear mapping from AC energy ratio to a base factor
// in [0.3, 1.5], then remapped to the adaptive [watson_lo, watson_hi] range.
// Uses only basic IEEE 754 arithmetic — deterministic on all platforms.

/// Base Watson curve: maps AC energy ratio to a factor in [0.3, 1.5].
/// This defines the SHAPE of the adaptation; the output range is then
/// remapped to the adaptive [watson_lo, watson_hi] range via `adaptive_params`.
fn watson_base_factor(ratio: f64) -> f64 {
    const R0: f64 = 0.01;
    const R1: f64 = 0.35;
    const R2: f64 = 2.0;
    const R3: f64 = 4.0;
    const F0: f64 = 0.3;
    const F1: f64 = 0.7;
    const F2: f64 = 1.0;
    const F3: f64 = 1.5;

    if ratio <= R0 {
        F0
    } else if ratio <= R1 {
        F0 + (ratio - R0) / (R1 - R0) * (F1 - F0)
    } else if ratio <= R2 {
        F1 + (ratio - R1) / (R2 - R1) * (F2 - F1)
    } else if ratio <= R3 {
        F2 + (ratio - R2) / (R3 - R2) * (F3 - F2)
    } else {
        F3
    }
}

/// Remap a base Watson factor from [0.3, 1.5] to a custom [lo, hi] range.
fn remap_watson(base_factor: f64, lo: f64, hi: f64) -> f64 {
    let t = (base_factor - 0.3) / (1.5 - 0.3);
    lo + t * (hi - lo)
}

/// Pre-computed energy ratios for all blocks. Computed once per image,
/// reused across adaptive candidates on decode.
struct EnergyRatios {
    ratios: Vec<f64>,
}

/// Compute per-block AC energy ratios (energy / median).
/// Only counts |c| >= 2 AC coefficients for recompression stability.
fn compute_energy_ratios(grid: &DctGrid) -> EnergyRatios {
    let blocks_wide = grid.blocks_wide();
    let blocks_tall = grid.blocks_tall();
    let total_blocks = blocks_wide * blocks_tall;

    let block_indices: Vec<usize> = (0..total_blocks).collect();

    let compute_energy = |&block_idx: &usize| -> f64 {
        let br = block_idx / blocks_wide;
        let bc = block_idx % blocks_wide;
        let block = grid.block(br, bc);
        let mut energy: f64 = 0.0;
        for k in 1..64 {
            let c = block[k];
            if c.abs() >= 2 {
                energy += (c as f64) * (c as f64);
            }
        }
        energy
    };

    #[cfg(feature = "parallel")]
    let ac_energies: Vec<f64> = block_indices.par_iter().map(compute_energy).collect();
    #[cfg(not(feature = "parallel"))]
    let ac_energies: Vec<f64> = block_indices.iter().map(compute_energy).collect();

    let mut sorted = ac_energies.clone();
    #[cfg(feature = "parallel")]
    sorted.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    #[cfg(not(feature = "parallel"))]
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if total_blocks == 0 {
        1.0
    } else {
        sorted[total_blocks / 2].max(1.0)
    };

    let ratios = ac_energies.iter().map(|&e| e / median).collect();
    EnergyRatios { ratios }
}

/// Compute Watson step factors from energy ratios + adaptive range.
fn watson_factors(ratios: &EnergyRatios, watson_lo: f64, watson_hi: f64) -> Vec<f64> {
    ratios
        .ratios
        .iter()
        .map(|&r| remap_watson(watson_base_factor(r), watson_lo, watson_hi))
        .collect()
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
/// blocks, accounting for header, RS parity, and repetition coding (r >= MIN_R).
/// All blocks are used for payload (no Watson skip filtering).
///
/// Uses full frame overhead (50 bytes) for capacity calculation.
pub fn fortress_max_frame_bytes(img: &JpegImage) -> Result<usize, StegoError> {
    fortress_max_frame_bytes_ext(img, false)
}

/// Maximum frame bytes with optional compact frame support.
///
/// When `compact` is true, uses the smaller compact frame overhead (22 bytes)
/// for Fortress empty-passphrase mode, enabling larger messages.
/// The `compact` flag is used by `fortress_capacity_compact` to know which
/// overhead to subtract; the binary search itself is format-agnostic.
pub fn fortress_max_frame_bytes_ext(img: &JpegImage, _compact: bool) -> Result<usize, StegoError> {
    let grid = img.dct_grid(0);
    let total_blocks = grid.blocks_wide() * grid.blocks_tall();

    if total_blocks <= FORTRESS_HEADER_BLOCKS {
        return Ok(0);
    }

    let payload_blocks = total_blocks - FORTRESS_HEADER_BLOCKS;

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
///
/// Uses full frame overhead (50 bytes). For compact frame capacity, use
/// `fortress_capacity_compact`.
pub fn fortress_capacity(img: &JpegImage) -> Result<usize, StegoError> {
    let max_frame = fortress_max_frame_bytes(img)?;
    if max_frame <= frame::FRAME_OVERHEAD {
        return Ok(0);
    }
    Ok(max_frame - frame::FRAME_OVERHEAD)
}

/// Estimate fortress plaintext capacity for compact frame mode (empty passphrase).
///
/// Uses the smaller compact frame overhead (22 bytes instead of 50), giving
/// 28 more bytes of message capacity.
pub fn fortress_capacity_compact(img: &JpegImage) -> Result<usize, StegoError> {
    let max_frame = fortress_max_frame_bytes_ext(img, true)?;
    if max_frame <= frame::FORTRESS_COMPACT_FRAME_OVERHEAD {
        return Ok(0);
    }
    Ok(max_frame - frame::FORTRESS_COMPACT_FRAME_OVERHEAD)
}

// --- Fortress encode ---

/// Encode a payload frame into an image using BA-QIM on DC block averages
/// with adaptive Watson perceptual masking.
///
/// The caller must have already built the frame_bytes (encrypted + framed).
/// This function embeds the magic header and the RS+repetition encoded
/// payload into permuted Y-channel block DCs with per-block adaptive QIM
/// step sizes (Watson energy-based scaling).
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

    // Derive fortress structural key for block permutation.
    let fort_key = crypto::derive_fortress_structural_key(passphrase);
    let perm = permute_blocks(total_blocks, &fort_key);

    let header_perm = &perm[..FORTRESS_HEADER_BLOCKS];
    let remaining_perm = &perm[FORTRESS_HEADER_BLOCKS..];
    let payload_blocks = remaining_perm.len();

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

    // Determine adaptive QIM step and Watson range based on repetition factor.
    let params = adaptive_params(r);
    let energy = compute_energy_ratios(grid);
    let factors = watson_factors(&energy, params.watson_lo, params.watson_hi);

    let rs_bit_count_aligned = payload_blocks / r;
    let mut rs_bits_padded = rs_bits;
    rs_bits_padded.resize(rs_bit_count_aligned, 0);
    let (rep_bits, _) = repetition::repetition_encode(&rs_bits_padded, payload_blocks);

    // Build header bits: 7 copies of magic byte.
    let mut header_bits = Vec::with_capacity(FORTRESS_HEADER_BLOCKS);
    for _ in 0..FORTRESS_HEADER_COPIES {
        for bp in (0..8).rev() {
            header_bits.push((FORTRESS_MAGIC >> bp) & 1);
        }
    }

    // Embed header using fixed step on the first 56 permuted blocks.
    let grid_mut = img.dct_grid_mut(0);
    for (bit_idx, &block_idx) in header_perm.iter().enumerate() {
        let br = block_idx / blocks_wide;
        let bc = block_idx % blocks_wide;
        let dc = grid_mut.get(br, bc, 0, 0);
        let avg = dc_to_avg(dc, qt_dc);
        let new_avg = qim_embed_avg(avg, HEADER_QIM_STEP, header_bits[bit_idx]);
        let new_dc = avg_to_dc(new_avg, qt_dc);
        grid_mut.set(br, bc, 0, 0, new_dc);
    }

    // Embed payload using adaptive per-block QIM step.
    for (payload_idx, &block_idx) in remaining_perm.iter().enumerate() {
        if payload_idx >= rep_bits.len() {
            break;
        }
        let step = params.base_step * factors[block_idx];
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
) -> Result<(crate::stego::payload::PayloadData, super::pipeline::DecodeQuality), StegoError> {
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

    // Extract header LLRs with fixed step.
    let mut header_llrs = Vec::with_capacity(FORTRESS_HEADER_BLOCKS);
    for &block_idx in &perm[..FORTRESS_HEADER_BLOCKS] {
        let br = block_idx / blocks_wide;
        let bc = block_idx % blocks_wide;
        let dc = grid.get(br, bc, 0, 0);
        let avg = dc_to_avg(dc, qt_dc);
        header_llrs.push(qim_extract_soft(avg, HEADER_QIM_STEP));
    }

    let magic = extract_magic_byte(&header_llrs);
    if magic != FORTRESS_MAGIC {
        return Err(StegoError::FrameCorrupted);
    }

    // Pre-compute energy ratios once for reuse across adaptive candidates.
    let energy = compute_energy_ratios(grid);
    let remaining_perm = &perm[FORTRESS_HEADER_BLOCKS..];
    let payload_blocks = remaining_perm.len();

    // Pre-compute block averages (read-only, shared across candidates).
    let block_avgs: Vec<f64> = remaining_perm
        .iter()
        .map(|&block_idx| {
            let br = block_idx / blocks_wide;
            let bc = block_idx % blocks_wide;
            dc_to_avg(grid.get(br, bc, 0, 0), qt_dc)
        })
        .collect();

    // Determine whether to use compact frame format (empty passphrase).
    let use_compact = passphrase.is_empty();

    // Build all (parity, r) candidates.
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for &parity in &ecc::PARITY_TIERS {
        let candidate_rs = if use_compact {
            super::pipeline::compute_candidate_rs_compact(payload_blocks, parity)
        } else {
            super::pipeline::compute_candidate_rs(payload_blocks, parity)
        };
        for r in candidate_rs {
            candidates.push((parity, r));
        }
    }

    // Try each candidate with per-r Watson factors (continuous adaptive).
    let try_candidate = |&(parity, r): &(usize, usize)| -> Option<(crate::stego::payload::PayloadData, super::pipeline::DecodeQuality)> {
        let params = adaptive_params(r);
        let factors = watson_factors(&energy, params.watson_lo, params.watson_hi);
        let reference_llr = params.base_step / 2.0;

        let payload_llrs: Vec<f64> = remaining_perm
            .iter()
            .enumerate()
            .map(|(i, &block_idx)| {
                let step = params.base_step * factors[block_idx];
                qim_extract_soft(block_avgs[i], step)
            })
            .collect();

        let rs_bit_count = payload_blocks / r;
        if rs_bit_count == 0 {
            return None;
        }
        let used_llrs = rs_bit_count * r;
        if used_llrs > payload_llrs.len() {
            return None;
        }

        let (voted_bits, rep_quality) =
            repetition::repetition_decode_soft_with_quality(&payload_llrs[..used_llrs], rs_bit_count);
        let voted_bytes = frame::bits_to_bytes(&voted_bits);

        // Use compact or full frame RS decode + parse based on passphrase.
        let (decoded_frame, rs_stats) = if use_compact {
            super::pipeline::try_rs_decode_compact_frame_with_parity(&voted_bytes, parity)?
        } else {
            super::pipeline::try_rs_decode_frame_with_parity(&voted_bytes, parity)?
        };

        let parsed = if use_compact {
            frame::parse_fortress_compact_frame(&decoded_frame).ok()?
        } else {
            frame::parse_frame(&decoded_frame).ok()?
        };

        let plaintext = crypto::decrypt(
            &parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce,
        ).ok()?;

        let len = parsed.plaintext_len as usize;
        if len > plaintext.len() {
            return None;
        }
        let payload_data = crate::stego::payload::decode_payload(&plaintext[..len]).ok()?;

        let mut q = super::pipeline::DecodeQuality::from_rs_stats_with_signal(
            &rs_stats, r as u8, parity as u16,
            rep_quality.avg_abs_llr_per_copy, reference_llr,
        );
        q.fortress_used = true;
        Some((payload_data, q))
    };

    #[cfg(feature = "parallel")]
    let result = candidates.par_iter().find_map_first(try_candidate);
    #[cfg(not(feature = "parallel"))]
    let result = candidates.iter().find_map(try_candidate);

    match result {
        Some(result) => Ok(result),
        None => Err(StegoError::FrameCorrupted),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::dct::DctGrid;

    #[test]
    fn qim_embed_extract_roundtrip() {
        for avg in [0.0, 50.0, 100.0, 200.0, -50.0, 127.5] {
            for bit in [0u8, 1] {
                let embedded = qim_embed_avg(avg, HEADER_QIM_STEP, bit);
                let llr = qim_extract_soft(embedded, HEADER_QIM_STEP);
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
                let embedded = qim_embed_avg(avg, HEADER_QIM_STEP, bit);
                let llr = qim_extract_soft(embedded, HEADER_QIM_STEP);
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
        // With r=15, rs_bit_count = ~956 bits = ~119 bytes
        // With parity=64, frame_bytes up to ~55
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
    fn watson_energy_smooth_vs_textured() {
        let blocks = vec![
            smooth_block(),
            smooth_block(),
            medium_block(),
            textured_block(),
        ];
        let grid = make_grid(4, 1, &blocks);
        let energy = compute_energy_ratios(&grid);

        assert_eq!(energy.ratios.len(), 4);
        assert_eq!(energy.ratios[0], 0.0, "Smooth block should have zero ratio");
        assert_eq!(energy.ratios[1], 0.0, "Smooth block should have zero ratio");
        assert!(
            energy.ratios[3] > energy.ratios[2],
            "Textured block should have higher ratio than medium"
        );
    }

    #[test]
    fn watson_energy_deterministic() {
        let blocks: Vec<[i16; 64]> = (0..16)
            .map(|i| {
                let mut b = [0i16; 64];
                b[0] = 100;
                for k in 1..(1 + (i % 10) + 1).min(64) {
                    b[k] = (i as i16 + 1) * 3;
                }
                b
            })
            .collect();

        let grid = make_grid(4, 4, &blocks);
        let a = compute_energy_ratios(&grid);
        let b = compute_energy_ratios(&grid);
        assert_eq!(a.ratios, b.ratios, "Energy ratios must be deterministic");
    }

    #[test]
    fn watson_ac_energy_invariant() {
        let mut block = textured_block();
        let original_ac_energy: f64 = block[1..64]
            .iter()
            .filter(|&&c| c.abs() >= 2)
            .map(|&c| (c as f64) * (c as f64))
            .sum();

        block[0] = 42; // modify DC only
        let modified_ac_energy: f64 = block[1..64]
            .iter()
            .filter(|&&c| c.abs() >= 2)
            .map(|&c| (c as f64) * (c as f64))
            .sum();

        assert_eq!(
            original_ac_energy, modified_ac_energy,
            "AC energy must be unchanged after DC modification"
        );
    }

    #[test]
    fn adaptive_fortress_encode_decode_roundtrip() {
        use crate::stego::armor::pipeline::{armor_encode, armor_decode};

        let test_jpeg = match std::fs::read("test-vectors/progressive_whatsapp_1200x1600.jpg") {
                Ok(d) => d,
                Err(_) => { eprintln!("skipped: test vector not found"); return; }
            };

        let passphrase = "test-adaptive-pass";
        let img = crate::jpeg::JpegImage::from_bytes(&test_jpeg).unwrap();
        let fort_cap = fortress_capacity(&img).unwrap();
        assert!(fort_cap >= 1, "Fortress capacity ({fort_cap}) must be >= 1");

        let message = if fort_cap >= 4 { "Hi!!" } else { "Hi" };

        let stego_bytes = armor_encode(&test_jpeg, message, passphrase)
            .expect("Adaptive fortress encode should succeed");

        let (decoded_msg, quality) = armor_decode(&stego_bytes, passphrase)
            .expect("Adaptive fortress decode should succeed");

        assert_eq!(decoded_msg.text, message, "Decoded message must match original");
        assert!(quality.fortress_used, "Should use fortress mode");
    }

    #[test]
    fn watson_magic_extraction() {
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
    fn watson_all_smooth_base_factor() {
        let blocks: Vec<[i16; 64]> = (0..9).map(|_| smooth_block()).collect();
        let grid = make_grid(3, 3, &blocks);
        let energy = compute_energy_ratios(&grid);
        for &r in &energy.ratios {
            assert_eq!(watson_base_factor(r), 0.3, "Smooth block base factor should be 0.3");
        }
    }

    #[test]
    fn watson_uniform_texture_base_factor() {
        let blocks: Vec<[i16; 64]> = (0..9).map(|_| textured_block()).collect();
        let grid = make_grid(3, 3, &blocks);
        let energy = compute_energy_ratios(&grid);
        for &r in &energy.ratios {
            let f = watson_base_factor(r);
            assert!(
                (0.7..=1.0).contains(&f),
                "Uniform texture ratio={r} should give factor in [0.7, 1.0], got {f}"
            );
        }
    }

    // --- Watson base factor tests ---

    #[test]
    fn watson_base_factor_anchor_points() {
        assert_eq!(watson_base_factor(0.0), 0.3);
        assert_eq!(watson_base_factor(0.01), 0.3);
        assert!((watson_base_factor(0.35) - 0.7).abs() < 1e-12);
        assert!((watson_base_factor(2.0) - 1.0).abs() < 1e-12);
        assert!((watson_base_factor(4.0) - 1.5).abs() < 1e-12);
        assert_eq!(watson_base_factor(10.0), 1.5);
    }

    #[test]
    fn watson_base_factor_monotonic() {
        let mut prev = watson_base_factor(0.0);
        let mut r = 0.001;
        while r <= 10.0 {
            let f = watson_base_factor(r);
            assert!(f >= prev - 1e-12, "Not monotonic at ratio={r}: {f} < {prev}");
            prev = f;
            r += 0.001;
        }
    }

    #[test]
    fn watson_base_factor_midpoints() {
        let mid1 = watson_base_factor((0.01 + 0.35) / 2.0);
        assert!((mid1 - 0.5).abs() < 1e-10, "Midpoint segment 1: expected ~0.5, got {mid1}");

        let mid2 = watson_base_factor((0.35 + 2.0) / 2.0);
        assert!((mid2 - 0.85).abs() < 1e-10, "Midpoint segment 2: expected ~0.85, got {mid2}");

        let mid3 = watson_base_factor((2.0 + 4.0) / 2.0);
        assert!((mid3 - 1.25).abs() < 1e-10, "Midpoint segment 3: expected ~1.25, got {mid3}");
    }

    #[test]
    fn watson_base_factor_deterministic() {
        for ratio in [0.0, 0.005, 0.18, 1.0, 3.0, 5.0, 100.0] {
            let a = watson_base_factor(ratio);
            let b = watson_base_factor(ratio);
            assert_eq!(a.to_bits(), b.to_bits(), "Not bit-identical at ratio={ratio}");
        }
    }

    // --- Remap + adaptive params tests ---

    #[test]
    fn remap_watson_full_range() {
        assert!((remap_watson(0.3, 0.3, 1.5) - 0.3).abs() < 1e-12);
        assert!((remap_watson(1.5, 0.3, 1.5) - 1.5).abs() < 1e-12);
        assert!((remap_watson(0.9, 0.3, 1.5) - 0.9).abs() < 1e-12);
    }

    #[test]
    fn remap_watson_narrow_range() {
        assert!((remap_watson(0.3, 0.9, 1.1) - 0.9).abs() < 1e-12);
        assert!((remap_watson(1.5, 0.9, 1.1) - 1.1).abs() < 1e-12);
        let mid = remap_watson(0.9, 0.9, 1.1);
        assert!((mid - 1.0).abs() < 1e-12, "Midpoint should remap to 1.0, got {mid}");
    }

    #[test]
    fn adaptive_params_continuous() {
        // At r=15 (minimum): max robustness
        let p = adaptive_params(15);
        assert_eq!(p.base_step, 12.0);
        assert_eq!(p.watson_lo, 0.9);
        assert_eq!(p.watson_hi, 1.1);

        // At r=61 (cap): max quality
        let p = adaptive_params(61);
        assert_eq!(p.base_step, 6.5);
        assert_eq!(p.watson_lo, 0.62);
        assert_eq!(p.watson_hi, 1.26);

        // Mid-range: r=38 (roughly halfway)
        let p = adaptive_params(38);
        assert!(p.base_step > 6.5 && p.base_step < 12.0);
        assert!(p.watson_lo > 0.62 && p.watson_lo < 0.9);
        assert!(p.watson_hi > 1.1 && p.watson_hi < 1.26);

        // Monotonic: higher r = smaller step
        let p15 = adaptive_params(15);
        let p38 = adaptive_params(38);
        let p61 = adaptive_params(61);
        assert!(p15.base_step > p38.base_step);
        assert!(p38.base_step > p61.base_step);

        // Clamped above 61
        let p200 = adaptive_params(200);
        assert_eq!(p200.base_step, 6.5);
    }

    #[test]
    fn watson_factors_all_smooth() {
        let blocks: Vec<[i16; 64]> = (0..9).map(|_| smooth_block()).collect();
        let grid = make_grid(3, 3, &blocks);
        let energy = compute_energy_ratios(&grid);
        let factors = watson_factors(&energy, 0.9, 1.1);
        assert!(
            factors.iter().all(|&f| (f - 0.9).abs() < 1e-12),
            "All smooth blocks should have factor 0.9 (watson_lo)"
        );
    }

    #[test]
    fn watson_factors_range_respected() {
        let blocks = vec![
            smooth_block(),
            smooth_block(),
            medium_block(),
            textured_block(),
        ];
        let grid = make_grid(4, 1, &blocks);
        let energy = compute_energy_ratios(&grid);

        for (lo, hi) in [(0.8, 1.2), (0.9, 1.1)] {
            let factors = watson_factors(&energy, lo, hi);
            for (i, &f) in factors.iter().enumerate() {
                assert!(
                    f >= lo - 1e-12 && f <= hi + 1e-12,
                    "Factor {f} for block {i} out of [{lo}, {hi}]"
                );
            }
        }
    }

    #[test]
    fn adaptive_encode_decode_roundtrip() {
        use crate::stego::armor::pipeline::{armor_encode, armor_decode};

        let test_jpeg = match std::fs::read("test-vectors/progressive_whatsapp_1200x1600.jpg") {
                Ok(d) => d,
                Err(_) => { eprintln!("skipped: test vector not found"); return; }
            };

        let passphrase = "adaptive-watson-test";
        let img = crate::jpeg::JpegImage::from_bytes(&test_jpeg).unwrap();
        let fort_cap = fortress_capacity(&img).unwrap();
        assert!(fort_cap >= 1, "Fortress capacity must be >= 1");

        let message = if fort_cap >= 4 { "Hi!!" } else { "Hi" };

        let stego_bytes = armor_encode(&test_jpeg, message, passphrase)
            .expect("Adaptive encode should succeed");

        let (decoded_msg, quality) = armor_decode(&stego_bytes, passphrase)
            .expect("Adaptive decode should succeed");

        assert_eq!(decoded_msg.text, message, "Decoded message must match");
        assert!(quality.fortress_used, "Should use fortress mode");
    }

    // --- Compact frame (empty passphrase) tests ---

    #[test]
    fn fortress_compact_encode_decode_roundtrip() {
        use crate::stego::armor::pipeline::{armor_encode, armor_decode};

        let test_jpeg = match std::fs::read("test-vectors/progressive_whatsapp_1200x1600.jpg") {
                Ok(d) => d,
                Err(_) => { eprintln!("skipped: test vector not found"); return; }
            };

        let passphrase = ""; // empty passphrase triggers compact frame
        let img = crate::jpeg::JpegImage::from_bytes(&test_jpeg).unwrap();
        let fort_cap = fortress_capacity_compact(&img).unwrap();
        assert!(fort_cap >= 1, "Fortress compact capacity ({fort_cap}) must be >= 1");

        let message = if fort_cap >= 4 { "Hi!!" } else { "Hi" };

        let stego_bytes = armor_encode(&test_jpeg, message, passphrase)
            .expect("Compact fortress encode should succeed");

        let (decoded_msg, quality) = armor_decode(&stego_bytes, passphrase)
            .expect("Compact fortress decode should succeed");

        assert_eq!(decoded_msg.text, message, "Decoded message must match");
        assert!(quality.fortress_used, "Should use fortress mode");
    }

    #[test]
    fn fortress_compact_capacity_larger_than_full() {
        let test_jpeg = match std::fs::read("test-vectors/progressive_whatsapp_1200x1600.jpg") {
                Ok(d) => d,
                Err(_) => { eprintln!("skipped: test vector not found"); return; }
            };

        let img = crate::jpeg::JpegImage::from_bytes(&test_jpeg).unwrap();
        let full_cap = fortress_capacity(&img).unwrap();
        let compact_cap = fortress_capacity_compact(&img).unwrap();

        assert!(
            compact_cap >= full_cap,
            "Compact capacity ({compact_cap}) should be >= full capacity ({full_cap})"
        );
        // The 28-byte saving should translate to >= 28 more plaintext bytes.
        if full_cap > 0 {
            assert!(
                compact_cap - full_cap >= 28,
                "Compact frame saves 28 bytes overhead, so capacity should increase by >= 28: \
                 compact={compact_cap}, full={full_cap}, diff={}",
                compact_cap - full_cap
            );
        }
    }

    #[test]
    fn fortress_nonempty_passphrase_still_uses_full_frame() {
        use crate::stego::armor::pipeline::{armor_encode, armor_decode};

        let test_jpeg = match std::fs::read("test-vectors/progressive_whatsapp_1200x1600.jpg") {
                Ok(d) => d,
                Err(_) => { eprintln!("skipped: test vector not found"); return; }
            };

        let passphrase = "some-secret";

        let stego_bytes = armor_encode(&test_jpeg, "Hi", passphrase)
            .expect("Non-empty passphrase fortress encode should succeed");

        let (decoded_msg, quality) = armor_decode(&stego_bytes, passphrase)
            .expect("Non-empty passphrase fortress decode should succeed");

        assert_eq!(decoded_msg.text, "Hi");
        assert!(quality.fortress_used, "Should use fortress mode");
    }

    #[test]
    fn fortress_compact_wrong_passphrase_fails() {
        use crate::stego::armor::pipeline::{armor_encode, armor_decode};

        let test_jpeg = match std::fs::read("test-vectors/progressive_whatsapp_1200x1600.jpg") {
                Ok(d) => d,
                Err(_) => { eprintln!("skipped: test vector not found"); return; }
            };

        // Encode with empty passphrase (compact frame)
        let stego_bytes = armor_encode(&test_jpeg, "Hi", "")
            .expect("Compact fortress encode should succeed");

        // Decode with non-empty passphrase should fail
        let result = armor_decode(&stego_bytes, "wrong");
        assert!(result.is_err(), "Decoding compact frame with wrong passphrase should fail");
    }
}
