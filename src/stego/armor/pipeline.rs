// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Armor encode/decode pipeline.
//!
//! Armor embeds messages using STDM (Spread Transform Dither Modulation)
//! into recompression-stable DCT coefficients, protected by Reed-Solomon
//! error correction.
//!
//! **Robustness features:**
//! - Frequency-restricted embedding (zigzag 1..=15) for stability
//! - Pre-clamp for pixel-domain settling
//! - 1-byte mean-QT header with 7x majority voting (56 units)
//! - Sequential repetition copies for soft majority voting
//! - +/-30% decode-side delta sweep (~21 candidates)
//! - Brute-force (r, parity) search on decode -- no fragile r-header
//! - DFT ring payload: resize-robust second layer in frequency domain

use crate::jpeg::JpegImage;
use crate::jpeg::dct::DctGrid;
use crate::jpeg::pixels;
use crate::stego::armor::ecc;
use crate::stego::armor::embedding::{self, stdm_embed, stdm_extract_soft};
use crate::stego::armor::fft2d;
use crate::stego::armor::fortress;
use crate::stego::armor::repetition;
use crate::stego::armor::resample;
use crate::stego::armor::selection::compute_stability_map;
use crate::stego::armor::spreading::{generate_spreading_vectors, SPREAD_LEN};
use crate::stego::armor::template;
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload::{self, PayloadData};
use crate::stego::permute;
use crate::stego::pipeline::GHOST_DECODE_STEPS;
use crate::stego::progress;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Number of embedding units for the header.
/// 1 byte x 8 bits x 7 copies = 56 units.
const HEADER_UNITS: usize = embedding::HEADER_UNITS; // 56
const HEADER_COPIES: usize = embedding::HEADER_COPIES; // 7

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
    // Build the payload (text + compression, no files for Armor).
    let payload_bytes = payload::encode_payload(message, &[])?;

    // Guard against payload exceeding the u16 length field in the frame format.
    if payload_bytes.len() > u16::MAX as usize {
        return Err(StegoError::MessageTooLarge);
    }

    let mut img = JpegImage::from_bytes(image_bytes)?;

    // Validate dimensions before any heavy processing.
    let fi = img.frame_info();
    crate::stego::validate_encode_dimensions(fi.width as u32, fi.height as u32)?;

    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    // Try Fortress (BA-QIM on DC) first if the message fits.
    // For empty passphrase, use compact frame (saves 28 bytes of overhead).
    let use_compact = passphrase.is_empty();
    if let Ok(max_fort) = fortress::fortress_max_frame_bytes_ext(&img, use_compact) {
        let fortress_frame = if use_compact {
            let ct = crypto::encrypt_with(
                &payload_bytes,
                passphrase,
                &crypto::FORTRESS_EMPTY_SALT,
                &crypto::FORTRESS_EMPTY_NONCE,
            );
            frame::build_fortress_compact_frame(payload_bytes.len() as u16, &ct)
        } else {
            let (ct, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase);
            frame::build_frame(payload_bytes.len() as u16, &salt, &nonce, &ct)
        };

        if fortress_frame.len() <= max_fort {
            // Pre-settle Y channel to QF75 QT tables before Fortress embedding.
            // This ensures block averages are on the same quantization grid that
            // WhatsApp (and most social media) use for recompression, making the
            // QIM embedding nearly idempotent under Q75 recompression.
            // Platform-side quality settings vary wildly (iOS 0.65 ≈ libjpeg Q90,
            // Android Q70, Web 0.65 ≈ varies), so we normalize here in the core.
            pre_settle_for_fortress(&mut img);
            fortress::fortress_encode(&mut img, &fortress_frame, passphrase)?;
            let stego_bytes = match img.to_bytes() {
                Ok(bytes) => bytes,
                Err(_) => {
                    img.rebuild_huffman_tables();
                    img.to_bytes().map_err(StegoError::InvalidJpeg)?
                }
            };
            return Ok(stego_bytes);
        }
    }

    // Full frame for STDM path (always uses random salt + nonce).
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase);
    let frame_bytes = frame::build_frame(payload_bytes.len() as u16, &salt, &nonce, &ciphertext);

    // Phase 3: Embed DFT template + ring payload BEFORE STDM.
    embed_dft_template(&mut img, passphrase, message)?;

    // Pre-clamp pass: IDCT -> clamp [0,255] -> DCT on all Y-channel blocks.
    // Settles coefficients so they already produce valid pixel values,
    // reducing systematic distortion from pixel clamping during recompression.
    pre_clamp_y_channel(&mut img);

    // 1. Compute stability map for Y channel (frequency-restricted).
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

    // 4-5. frame_bytes already built above (shared with Fortress path).

    // 6. Compute mean QT from actual quantization table.
    let mean_qt = embedding::compute_mean_qt(&qt.values);
    let header_byte = embedding::encode_mean_qt(mean_qt);
    let bootstrap_delta = embedding::BOOTSTRAP_DELTA;
    let reference_delta = embedding::compute_delta_from_mean_qt(mean_qt, 1);

    // 7. Build header: 7 copies of 1 header byte (1 x 8 bits x 7 copies = 56 units)
    let mut all_bits = Vec::with_capacity(num_units);
    for _ in 0..HEADER_COPIES {
        for bp in (0..8).rev() {
            all_bits.push((header_byte >> bp) & 1);
        }
    }

    // 8. Decide Phase 1 vs Phase 2 encoding.
    let payload_units = if num_units > HEADER_UNITS {
        num_units - HEADER_UNITS
    } else {
        return Err(StegoError::ImageTooSmall);
    };

    // Find best Phase 2 parity tier (r>=3) in a single pass, caching the RS result.
    let phase2_result: Option<(usize, Vec<u8>)> = {
        let mut found = None;
        for &parity in &ecc::PARITY_TIERS {
            let rs_encoded = ecc::rs_encode_blocks_with_parity(&frame_bytes, parity);
            let rs_bits_len = rs_encoded.len() * 8;
            if rs_bits_len <= payload_units {
                let r = repetition::compute_r(rs_bits_len, payload_units);
                if r >= 3 {
                    found = Some((parity, rs_encoded));
                    break;
                }
            }
        }
        found
    };

    let embed_delta_fn: Box<dyn Fn(usize) -> f64> = if let Some((_chosen_parity, rs_encoded)) = phase2_result {
        // --- Phase 2 encode: use cached RS result ---
        let rs_bits = frame::bytes_to_bits(&rs_encoded);

        let r = repetition::compute_r(rs_bits.len(), payload_units);
        let rs_bit_count_aligned = payload_units / r;
        let mut rs_bits_padded = rs_bits;
        rs_bits_padded.resize(rs_bit_count_aligned, 0);
        let (rep_bits, _) = repetition::repetition_encode(&rs_bits_padded, payload_units);

        let adaptive_delta = embedding::compute_delta_from_mean_qt(mean_qt, r);

        all_bits.extend_from_slice(&rep_bits[..payload_units.min(rep_bits.len())]);

        Box::new(move |bit_idx| {
            if bit_idx < HEADER_UNITS { bootstrap_delta } else { adaptive_delta }
        })
    } else {
        // --- Phase 1 encode: fixed RS parity=64, no repetition ---
        let rs_encoded = ecc::rs_encode_blocks(&frame_bytes);
        let rs_bits = frame::bytes_to_bits(&rs_encoded);

        if rs_bits.len() > payload_units {
            return Err(StegoError::MessageTooLarge);
        }

        let mut payload_bits = rs_bits;
        payload_bits.resize(payload_units, 0);
        all_bits.extend_from_slice(&payload_bits);

        Box::new(move |bit_idx| {
            if bit_idx < HEADER_UNITS { bootstrap_delta } else { reference_delta }
        })
    };

    let embed_count = all_bits.len().min(num_units);

    // 9. Generate spreading vectors.
    let vectors = generate_spreading_vectors(&spread_seed, embed_count);

    // 10. STDM embed each bit into coefficient groups.
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

    // 11. Write modified JPEG.
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
    /// True if geometric recovery (Phase 3) was used to decode.
    pub geometry_corrected: bool,
    /// Number of template peaks detected (out of 32).
    pub template_peaks_detected: u8,
    /// Estimated rotation angle in degrees (0 if no geometry correction).
    pub estimated_rotation_deg: f32,
    /// Estimated scale factor (1.0 if no geometry correction).
    pub estimated_scale: f32,
    /// True if DFT ring was used (message may be truncated).
    pub dft_ring_used: bool,
    /// DFT ring capacity in bytes (0 if not applicable).
    pub dft_ring_capacity: u16,
    /// True if Fortress sub-mode (BA-QIM) was used for encoding.
    pub fortress_used: bool,
    /// Signal strength from LLR analysis (0.0 = no signal, higher = stronger).
    /// Used to compute meaningful integrity when RS errors are 0.
    pub signal_strength: f64,
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
            geometry_corrected: false,
            template_peaks_detected: 0,
            estimated_rotation_deg: 0.0,
            estimated_scale: 1.0,
            dft_ring_used: false,
            dft_ring_capacity: 0,
            fortress_used: false,
            signal_strength: 0.0,
        }
    }

    /// Create quality info from Armor RS decode stats with Phase 2 metadata.
    ///
    /// **Legacy constructor** — kept for backward compatibility. Uses RS-only
    /// integrity (always 100% when RS errors are 0). Prefer
    /// `from_rs_stats_with_signal` for LLR-based integrity scoring.
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
            geometry_corrected: false,
            template_peaks_detected: 0,
            estimated_rotation_deg: 0.0,
            estimated_scale: 1.0,
            dft_ring_used: false,
            dft_ring_capacity: 0,
            fortress_used: false,
            signal_strength: 0.0,
        }
    }

    /// Create quality info from Armor RS decode stats with LLR signal quality.
    ///
    /// Combines LLR-based signal strength (70% weight) with RS error margin
    /// (30% weight) to produce a meaningful integrity percentage even when
    /// RS errors are 0 (because repetition coding absorbed all damage).
    ///
    /// - `signal_strength`: average |LLR| per copy per bit from extraction.
    /// - `reference_llr`: expected |LLR| for a pristine embedding (delta/2 for
    ///   STDM, step/2 for QIM).
    pub fn from_rs_stats_with_signal(
        stats: &ecc::RsDecodeStats,
        repetition_factor: u8,
        parity_len: u16,
        signal_strength: f64,
        reference_llr: f64,
    ) -> Self {
        let integrity = compute_integrity(signal_strength, stats, reference_llr);
        Self {
            mode: super::super::frame::MODE_ARMOR,
            rs_errors_corrected: stats.total_errors as u32,
            rs_error_capacity: stats.error_capacity as u32,
            integrity_percent: integrity,
            repetition_factor,
            parity_len,
            geometry_corrected: false,
            template_peaks_detected: 0,
            estimated_rotation_deg: 0.0,
            estimated_scale: 1.0,
            dft_ring_used: false,
            dft_ring_capacity: 0,
            fortress_used: false,
            signal_strength,
        }
    }
}

/// Compute integrity from both LLR signal strength and RS error stats.
///
/// - `signal_strength`: average |LLR| per copy per bit from extraction.
/// - `rs_stats`: Reed-Solomon error correction statistics.
/// - `reference_llr`: expected |LLR| for a pristine embedding.
///
/// Weighting: 70% signal quality (LLR), 30% RS error margin.
///
/// For pristine images: signal_strength ≈ reference → integrity ~95-100%.
/// For recompressed images: signal_strength drops → integrity ~60-80%.
/// For severely degraded images: signal_strength near 0 → integrity ~30-50%.
fn compute_integrity(signal_strength: f64, rs_stats: &ecc::RsDecodeStats, reference_llr: f64) -> u8 {
    let llr_score = if reference_llr > 0.0 {
        (signal_strength / reference_llr).min(1.0).max(0.0)
    } else {
        1.0 // No reference available, assume good
    };
    let rs_score = if rs_stats.error_capacity == 0 {
        1.0
    } else {
        let ratio = rs_stats.total_errors as f64 / rs_stats.error_capacity as f64;
        (1.0 - ratio).max(0.0)
    };
    // Weight: 70% signal quality, 30% RS margin
    let combined = 0.7 * llr_score + 0.3 * rs_score;
    (combined * 100.0).round().max(0.0).min(100.0) as u8
}

/// Compute average |LLR| from a slice of raw LLR values (Phase 1 path).
fn compute_avg_abs_llr(llrs: &[f64]) -> f64 {
    if llrs.is_empty() {
        return 0.0;
    }
    let sum: f64 = llrs.iter().map(|llr| llr.abs()).sum();
    sum / llrs.len() as f64
}

/// Decode a text message from a stego JPEG using Armor mode.
///
/// Tries standard decode with delta sweep first, then falls back to
/// geometric recovery (Phase 3) for rotated/scaled images.
///
/// # Arguments
/// - `stego_bytes`: Raw bytes of the stego JPEG image.
/// - `passphrase`: The passphrase used during encoding.
///
/// # Returns
/// A tuple of (decoded plaintext message, decode quality info).
pub fn armor_decode(stego_bytes: &[u8], passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    // Try Fortress first (fast magic-byte check on DC coefficients).
    let img = JpegImage::from_bytes(stego_bytes)?;
    if img.num_components() > 0 {
        if let Ok(result) = fortress::fortress_decode(&img, passphrase) {
            return Ok(result);
        }
    }

    // Try standard STDM decode with delta sweep (reuse already-parsed image).
    // try_armor_decode sets progress total and tracks fortress + phase 1/2 steps.
    match try_armor_decode(&img, passphrase) {
        Ok(result) => return Ok(result),
        Err(new_err) => {
            // Try geometric recovery (Phase 3).
            progress::advance(); // phase 3
            match try_geometric_recovery(stego_bytes, passphrase) {
                Ok(result) => return Ok(result),
                Err(_) => return Err(new_err),
            }
        }
    }
}

/// Armor decode with delta sweep: parse image once, try multiple mean_qt candidates.
pub(crate) fn try_armor_decode(img: &JpegImage, passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    // 1. Compute frequency-restricted stability map.
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

    // 4. Generate spreading vectors for all units.
    let vectors = generate_spreading_vectors(&spread_seed, num_units);

    let grid = img.dct_grid(0);

    // 5. Extract 1-byte header at bootstrap delta.
    if num_units <= HEADER_UNITS {
        return Err(StegoError::ImageTooSmall);
    }
    let header_byte = extract_header_byte(grid, positions, &vectors, embedding::BOOTSTRAP_DELTA, 0);
    let header_mean_qt = embedding::decode_mean_qt(header_byte);

    // 6. Compute current image's mean QT for comparison.
    let current_mean_qt = embedding::compute_mean_qt(&qt.values);

    let payload_units = num_units - HEADER_UNITS;

    // 7. Build candidate mean_qt values for delta sweep.
    // Wider sweep +/-30% in 3% steps (~21 candidates) from both header and current.
    let mut raw_candidates = Vec::with_capacity(24);
    raw_candidates.push(header_mean_qt);
    raw_candidates.push(current_mean_qt);
    for step in 1..=10 {
        let factor = step as f64 * 0.03;
        raw_candidates.push(header_mean_qt * (1.0 - factor));
        raw_candidates.push(header_mean_qt * (1.0 + factor));
    }

    // Deduplicate (within 0.1 tolerance)
    let mut candidates: Vec<f64> = Vec::with_capacity(raw_candidates.len());
    for &c in &raw_candidates {
        if c > 0.1 && !candidates.iter().any(|&existing| (existing - c).abs() < 0.1) {
            candidates.push(c);
        }
    }

    // Set progress total now that we know the candidate count.
    // fortress(1) + phase1(nc) + phase2(nc) + phase3(1) + ghost(4) per run.
    // Doubled fortress+phase1+phase2 for potential second run via geometric recovery.
    // Minimum 50 so the progress bar looks meaningful even when nc is small
    // (e.g. Ghost-encoded images where the Armor header is garbage → nc=1).
    let nc = candidates.len() as u32;
    // Only set total on first call; geometric recovery calls us again.
    if progress::get().1 == 0 {
        let total = (2 * (1 + nc + nc) + 1 + GHOST_DECODE_STEPS).max(50);
        // Use set_total (not init) to avoid resetting STEP — in parallel mode,
        // other threads may have already advanced the counter.
        progress::set_total(total);
    }
    progress::advance(); // fortress check already done by caller

    // 8. Pass 1: Try Phase 1 for ALL candidates first (fast per candidate).
    for &mean_qt in &candidates {
        progress::check_cancelled()?;
        let reference_delta = embedding::compute_delta_from_mean_qt(mean_qt, 1);

        if let Ok(result) = decode_phase1_with_offset(
            grid, positions, &vectors, reference_delta, num_units, HEADER_UNITS,
            passphrase,
        ) {
            return Ok(result);
        }
        progress::advance();
    }

    // 9. Pass 2: Try Phase 2 for ALL candidates (expensive, only if all Phase 1 failed).
    let mut cached_llrs: Vec<(f64, Vec<f64>)> = Vec::new();
    for &mean_qt in &candidates {
        progress::check_cancelled()?;
        if let Ok(result) = decode_phase2_search(
            grid, positions, &vectors, mean_qt,
            num_units, payload_units, passphrase,
            &mut cached_llrs,
        ) {
            return Ok(result);
        }
        progress::advance();
    }

    Err(StegoError::FrameCorrupted)
}

/// Armor decode without Fortress: STDM delta sweep + Phase 3 geometric recovery.
///
/// Used by the parallel smart_decode path to run STDM and Phase 3 concurrently
/// with Fortress (which runs on a separate thread).
pub(crate) fn armor_decode_no_fortress(img: &JpegImage, stego_bytes: &[u8], passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    match try_armor_decode(img, passphrase) {
        Ok(result) => Ok(result),
        Err(_stdm_err) => {
            progress::check_cancelled()?;
            match try_geometric_recovery(stego_bytes, passphrase) {
                Ok(result) => Ok(result),
                Err(_) => Err(_stdm_err),
            }
        }
    }
}

/// Phase 3 geometric recovery: detect DFT template, estimate transform, resample, retry.
/// Also tries DFT ring payload extraction as fallback.
pub(crate) fn try_geometric_recovery(stego_bytes: &[u8], passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    use crate::stego::armor::dft_payload;

    let img = JpegImage::from_bytes(stego_bytes)?;

    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    // Convert to pixel domain and compute 2D FFT.
    let (luma_pixels, w, h) = pixels::jpeg_to_luma_f64(&img);
    let spectrum = fft2d::fft2d(&luma_pixels, w, h);

    // Generate expected template peaks and search for them.
    let peaks = template::generate_template_peaks(passphrase, w, h);
    let detected = template::detect_template(&spectrum, &peaks);

    // Estimate the geometric transform from detected peaks.
    let transform = template::estimate_transform(&detected)
        .ok_or(StegoError::FrameCorrupted)?;

    // Skip if transform is essentially identity (fast path would have worked).
    if transform.rotation_rad.abs() < 0.001 && (transform.scale - 1.0).abs() < 0.001 {
        // Try DFT ring extraction directly (no geometry correction needed)
        if let Some(ring_bytes) = dft_payload::extract_ring_payload(&spectrum, passphrase) {
            if let Ok(text) = std::str::from_utf8(&ring_bytes) {
                let ring_cap = dft_payload::ring_capacity(w, h);
                return Ok((PayloadData { text: text.to_string(), files: vec![] }, DecodeQuality {
                    mode: super::super::frame::MODE_ARMOR,
                    rs_errors_corrected: 0,
                    rs_error_capacity: 0,
                    integrity_percent: 50, // truncated message
                    repetition_factor: 0,
                    parity_len: 0,
                    geometry_corrected: false,
                    template_peaks_detected: detected.len() as u8,
                    estimated_rotation_deg: 0.0,
                    estimated_scale: 1.0,
                    dft_ring_used: true,
                    dft_ring_capacity: ring_cap as u16,
                    fortress_used: false,
                    signal_strength: 0.0,
                }));
            }
        }
        return Err(StegoError::FrameCorrupted);
    }

    // P0: Drop spectrum — only needed for template detection and ring extraction above.
    drop(spectrum);

    // Resample the pixel image to undo the geometric transform.
    let corrected_pixels = resample::resample_bilinear(
        &luma_pixels, w, h, &transform, w, h,
    );

    // P0: Drop luma_pixels — only needed for FFT and resample.
    drop(luma_pixels);

    // P0: Move img instead of clone — img is not used after correction.
    let mut corrected_img = img;
    pixels::luma_f64_to_jpeg(&corrected_pixels, w, h, &mut corrected_img);

    // P0: Drop corrected_pixels — written into corrected_img.
    drop(corrected_pixels);

    // Retry standard decode on the corrected image (no re-encode/re-parse needed).
    match try_armor_decode(&corrected_img, passphrase) {
        Ok((text, mut quality)) => {
            quality.geometry_corrected = true;
            quality.template_peaks_detected = detected.len() as u8;
            quality.estimated_rotation_deg = transform.rotation_rad.to_degrees() as f32;
            quality.estimated_scale = transform.scale as f32;
            return Ok((text, quality));
        }
        Err(_) => {
            // STDM decode failed after geometry correction -- try DFT ring
            // Recompute FFT from corrected image for ring extraction
            {
                let (cp, cw, ch) = pixels::jpeg_to_luma_f64(&corrected_img);
                let corrected_spectrum = fft2d::fft2d(&cp, cw, ch);
                if let Some(ring_bytes) = dft_payload::extract_ring_payload(&corrected_spectrum, passphrase) {
                    if let Ok(text) = std::str::from_utf8(&ring_bytes) {
                        let ring_cap = dft_payload::ring_capacity(cw, ch);
                        return Ok((PayloadData { text: text.to_string(), files: vec![] }, DecodeQuality {
                            mode: super::super::frame::MODE_ARMOR,
                            rs_errors_corrected: 0,
                            rs_error_capacity: 0,
                            integrity_percent: 50,
                            repetition_factor: 0,
                            parity_len: 0,
                            geometry_corrected: true,
                            template_peaks_detected: detected.len() as u8,
                            estimated_rotation_deg: transform.rotation_rad.to_degrees() as f32,
                            estimated_scale: transform.scale as f32,
                            dft_ring_used: true,
                            dft_ring_capacity: ring_cap as u16,
                            fortress_used: false,
                            signal_strength: 0.0,
                        }));
                    }
                }
            }
        }
    }

    Err(StegoError::FrameCorrupted)
}

/// Extract the 1-byte mean-QT header from embedding units using soft majority voting.
///
/// Reads 7 copies x 1 byte x 8 bits = 56 units at the given delta and offset.
fn extract_header_byte(
    grid: &DctGrid,
    positions: &[crate::stego::permute::CoeffPos],
    vectors: &[[f64; SPREAD_LEN]],
    delta: f64,
    offset: usize,
) -> u8 {
    let mut header_llrs = [0.0f64; 56]; // 7 copies x 8 bits
    for i in 0..56 {
        let unit_idx = offset + i;
        let group_start = unit_idx * SPREAD_LEN;
        let group = &positions[group_start..group_start + SPREAD_LEN];

        let mut coeffs = [0.0f64; SPREAD_LEN];
        for (k, pos) in group.iter().enumerate() {
            coeffs[k] = flat_get(grid, pos.flat_idx) as f64;
        }

        header_llrs[i] = stdm_extract_soft(&coeffs, &vectors[unit_idx], delta);
    }

    // Majority vote across 7 copies for each of 8 bits
    let mut byte = 0u8;
    for bit_pos in 0..8 {
        let mut total = 0.0;
        for copy in 0..7 {
            total += header_llrs[copy * 8 + bit_pos];
        }
        if total < 0.0 {
            byte |= 1 << (7 - bit_pos);
        }
    }
    byte
}

/// Phase 1 decode: extract all bits with given delta, then RS decode.
fn decode_phase1_with_offset(
    grid: &DctGrid,
    positions: &[crate::stego::permute::CoeffPos],
    vectors: &[[f64; SPREAD_LEN]],
    delta: f64,
    num_units: usize,
    payload_offset: usize,
    passphrase: &str,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let payload_units = num_units - payload_offset;

    // Extract all LLRs from payload region
    let mut all_llrs = Vec::with_capacity(payload_units);
    for unit_idx in payload_offset..num_units {
        let group_start = unit_idx * SPREAD_LEN;
        let group = &positions[group_start..group_start + SPREAD_LEN];

        let mut coeffs = [0.0f64; SPREAD_LEN];
        for (k, pos) in group.iter().enumerate() {
            coeffs[k] = flat_get(grid, pos.flat_idx) as f64;
        }

        all_llrs.push(stdm_extract_soft(&coeffs, &vectors[unit_idx], delta));
    }

    // Compute signal strength from raw LLRs (before hard decision)
    let signal_strength = compute_avg_abs_llr(&all_llrs);
    // Reference LLR for pristine STDM embedding: delta / 2
    let reference_llr = delta / 2.0;

    // Convert LLRs to hard bits
    let extracted_bits: Vec<u8> = all_llrs.iter()
        .map(|&llr| if llr >= 0.0 { 0 } else { 1 })
        .collect();

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

    let payload_data = payload::decode_payload(&plaintext[..len])?;
    let quality = DecodeQuality::from_rs_stats_with_signal(
        &rs_stats, 1, ecc::parity_len() as u16, signal_strength, reference_llr,
    );
    Ok((payload_data, quality))
}

/// Phase 2 brute-force search: try all plausible (r, parity) combinations.
///
/// For each parity tier, sweeps possible frame lengths to find distinct r values,
/// then attempts decode with each (r, parity, delta) combination.
fn decode_phase2_search(
    grid: &DctGrid,
    positions: &[crate::stego::permute::CoeffPos],
    vectors: &[[f64; SPREAD_LEN]],
    mean_qt: f64,
    num_units: usize,
    payload_units: usize,
    passphrase: &str,
    cached_llrs: &mut Vec<(f64, Vec<f64>)>,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    // Build all (parity, r, delta) candidates and pre-extract unique delta LLR sets
    // so the search can run in parallel without &mut cache.
    let mut candidates: Vec<(usize, usize, f64)> = Vec::new();
    for &parity in &ecc::PARITY_TIERS {
        let candidate_rs = compute_candidate_rs(payload_units, parity);
        for r in candidate_rs {
            let delta = embedding::compute_delta_from_mean_qt(mean_qt, r);
            candidates.push((parity, r, delta));
        }
    }

    // Pre-extract LLRs for all unique deltas (populates cache sequentially)
    for &(_, _, delta) in &candidates {
        get_or_extract_llrs(
            cached_llrs, delta,
            grid, positions, vectors, num_units, HEADER_UNITS,
        );
    }

    // Snapshot the cache for read-only parallel access
    let llr_cache: &[(f64, Vec<f64>)] = cached_llrs;

    let find_llrs = |delta: f64| -> &[f64] {
        for (cached_delta, cached_llrs) in llr_cache.iter() {
            if (cached_delta - delta).abs() < 0.001 {
                return cached_llrs;
            }
        }
        &[]
    };

    let try_candidate = |&(parity, r, adaptive_delta): &(usize, usize, f64)| -> Option<(PayloadData, DecodeQuality)> {
        let raw_llrs = find_llrs(adaptive_delta);

        let rs_bit_count = payload_units / r;
        if rs_bit_count == 0 {
            return None;
        }
        let used_llrs = rs_bit_count * r;
        if used_llrs > raw_llrs.len() {
            return None;
        }

        let (voted_bits, rep_quality) = repetition::repetition_decode_soft_with_quality(
            &raw_llrs[..used_llrs], rs_bit_count,
        );
        let voted_bytes = frame::bits_to_bytes(&voted_bits);

        let (decoded_frame, rs_stats) = try_rs_decode_frame_with_parity(&voted_bytes, parity)?;
        let parsed = frame::parse_frame(&decoded_frame).ok()?;
        let plaintext = crypto::decrypt(
            &parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce,
        ).ok()?;
        let len = parsed.plaintext_len as usize;
        if len > plaintext.len() {
            return None;
        }
        let payload_data = payload::decode_payload(&plaintext[..len]).ok()?;
        let reference_llr = adaptive_delta / 2.0;
        let quality = DecodeQuality::from_rs_stats_with_signal(
            &rs_stats, r as u8, parity as u16,
            rep_quality.avg_abs_llr_per_copy, reference_llr,
        );
        Some((payload_data, quality))
    };

    #[cfg(feature = "parallel")]
    let result = candidates.par_iter().find_map_first(try_candidate);
    #[cfg(not(feature = "parallel"))]
    let result = candidates.iter().find_map(try_candidate);

    result.ok_or(StegoError::FrameCorrupted)
}

/// Compute distinct candidate r values for a given parity tier and payload capacity.
pub(super) fn compute_candidate_rs(payload_units: usize, parity: usize) -> Vec<usize> {
    let mut rs_set = std::collections::BTreeSet::new();

    // Sweep possible frame lengths to find all distinct r values.
    // rs_encoded_len is monotonically increasing with frame_len, so once
    // rs_bits exceeds payload_units we can stop.
    let min_frame = frame::FRAME_OVERHEAD;
    let max_frame = frame::MAX_FRAME_BYTES;

    for frame_len in min_frame..=max_frame {
        let rs_encoded_len = ecc::rs_encoded_len_with_parity(frame_len, parity);
        let rs_bits = rs_encoded_len * 8;
        if rs_bits > payload_units {
            break;
        }
        let r = repetition::compute_r(rs_bits, payload_units);
        if r >= 3 {
            rs_set.insert(r);
        }
    }

    rs_set.into_iter().collect()
}

/// Compute candidate repetition factors for fortress compact frames.
///
/// Same as `compute_candidate_rs` but uses the compact frame overhead
/// (22 bytes instead of 50) for minimum frame length.
pub(super) fn compute_candidate_rs_compact(payload_units: usize, parity: usize) -> Vec<usize> {
    let mut rs_set = std::collections::BTreeSet::new();

    let min_frame = frame::FORTRESS_COMPACT_FRAME_OVERHEAD;
    let max_frame = frame::MAX_FRAME_BYTES;

    for frame_len in min_frame..=max_frame {
        let rs_encoded_len = ecc::rs_encoded_len_with_parity(frame_len, parity);
        let rs_bits = rs_encoded_len * 8;
        if rs_bits > payload_units {
            break;
        }
        let r = repetition::compute_r(rs_bits, payload_units);
        if r >= 3 {
            rs_set.insert(r);
        }
    }

    rs_set.into_iter().collect()
}

/// Maximum LLR cache entries. Each entry can be ~31-65 MB for large images.
/// P2b: Limit to 5 entries to cap memory at ~155-325 MB instead of unbounded.
const LLR_CACHE_MAX: usize = 5;

/// Get cached LLRs for a delta value, or extract them from the grid.
///
/// P2b: Uses LRU eviction when the cache exceeds `LLR_CACHE_MAX` entries.
fn get_or_extract_llrs(
    cache: &mut Vec<(f64, Vec<f64>)>,
    delta: f64,
    grid: &DctGrid,
    positions: &[crate::stego::permute::CoeffPos],
    vectors: &[[f64; SPREAD_LEN]],
    num_units: usize,
    payload_offset: usize,
) -> Vec<f64> {
    // Check cache (use approximate comparison for f64)
    for i in 0..cache.len() {
        if (cache[i].0 - delta).abs() < 0.001 {
            // P2b: Move to end (most recently used) for LRU ordering
            if i < cache.len() - 1 {
                let entry = cache.remove(i);
                cache.push(entry);
            }
            return cache.last().unwrap().1.clone();
        }
    }

    // Extract fresh LLRs
    let unit_indices: Vec<usize> = (payload_offset..num_units).collect();

    let extract_one = |&unit_idx: &usize| -> f64 {
        let group_start = unit_idx * SPREAD_LEN;
        let group = &positions[group_start..group_start + SPREAD_LEN];

        let mut coeffs = [0.0f64; SPREAD_LEN];
        for (k, pos) in group.iter().enumerate() {
            coeffs[k] = flat_get(grid, pos.flat_idx) as f64;
        }

        stdm_extract_soft(&coeffs, &vectors[unit_idx], delta)
    };

    #[cfg(feature = "parallel")]
    let llrs: Vec<f64> = unit_indices.par_iter().map(extract_one).collect();
    #[cfg(not(feature = "parallel"))]
    let llrs: Vec<f64> = unit_indices.iter().map(extract_one).collect();

    // P2b: Evict oldest entry if cache is full
    if cache.len() >= LLR_CACHE_MAX {
        cache.remove(0); // Remove LRU (oldest) entry
    }

    cache.push((delta, llrs.clone()));
    llrs
}

/// Pre-clamp the Y channel: IDCT -> clamp [0, 255] -> DCT for all blocks.
///
/// This "settles" the cover image's coefficients so they produce valid pixel
/// values. Without this, recompression through a pixel-domain pipeline
/// (IDCT -> clamp -> DCT) introduces systematic distortion from clamping.
fn pre_clamp_y_channel(img: &mut JpegImage) {
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt_values = img.quant_table(qt_id).expect("Y quant table must exist").values;
    let grid = img.dct_grid_mut(0);

    let process_block = |chunk: &mut [i16]| {
        let quantized: [i16; 64] = chunk.try_into().unwrap();
        let mut px = pixels::idct_block(&quantized, &qt_values);
        for p in px.iter_mut() {
            *p = p.clamp(0.0, 255.0);
        }
        let settled = pixels::dct_block(&px, &qt_values);
        chunk.copy_from_slice(&settled);
    };

    #[cfg(feature = "parallel")]
    grid.coeffs_mut().par_chunks_mut(64).for_each(process_block);
    #[cfg(not(feature = "parallel"))]
    grid.coeffs_mut().chunks_mut(64).for_each(process_block);
}

/// Try to RS-decode a frame from extracted bytes using a specific parity length.
pub(super) fn try_rs_decode_frame_with_parity(
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

                if total_frame_len > frame::MAX_FRAME_BYTES {
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

/// Try to RS-decode a compact fortress frame from extracted bytes.
///
/// Same logic as `try_rs_decode_frame_with_parity` but uses the compact frame
/// overhead (no salt/nonce) when computing the expected total frame length.
pub(super) fn try_rs_decode_compact_frame_with_parity(
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
                // Compact frame: no salt (16) or nonce (12)
                let total_frame_len = 2 + ct_len + 4;

                if total_frame_len > frame::MAX_FRAME_BYTES {
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

                if total_frame_len > frame::MAX_FRAME_BYTES {
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

/// Dual-tier capacity information for Armor mode.
///
/// Reports both Fortress (BA-QIM) and STDM capacities so the UI can show
/// which sub-mode will be used for the current message length.
#[derive(Debug, Clone)]
pub struct ArmorCapacityInfo {
    /// Maximum plaintext bytes embeddable via Fortress (BA-QIM). 0 if image too small.
    pub fortress_capacity: usize,
    /// Maximum plaintext bytes embeddable via STDM (standard Armor).
    pub stdm_capacity: usize,
}

/// Compute dual-tier capacity information for an Armor-mode image.
///
/// Returns both Fortress and STDM capacities. The existing `armor_capacity()`
/// function continues to return `stdm_capacity` for backward compatibility.
pub fn armor_capacity_info(jpeg_bytes: &[u8]) -> Result<ArmorCapacityInfo, StegoError> {
    let img = JpegImage::from_bytes(jpeg_bytes)?;

    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    let fortress_cap = fortress::fortress_capacity(&img).unwrap_or(0);
    let stdm_cap = super::capacity::estimate_armor_capacity(&img).unwrap_or(0);

    Ok(ArmorCapacityInfo {
        fortress_capacity: fortress_cap,
        stdm_capacity: stdm_cap,
    })
}

/// Embed a DFT template and ring payload into the luminance channel.
///
/// Embeds both template peaks (for geometry resilience) and the DFT ring
/// payload (resize-robust second layer) in the same FFT pass.
fn embed_dft_template(img: &mut JpegImage, passphrase: &str, message: &str) -> Result<(), StegoError> {
    let (luma_pixels, w, h) = pixels::jpeg_to_luma_f64(img);

    // P0: Drop luma_pixels after FFT — only needed to produce the spectrum.
    let mut spectrum = fft2d::fft2d(&luma_pixels, w, h);
    drop(luma_pixels);

    // Template peaks for geometry estimation
    let peaks = template::generate_template_peaks(passphrase, w, h);
    template::embed_template(&mut spectrum, &peaks);

    // Ring payload -- truncate message to ring capacity
    use crate::stego::armor::dft_payload;
    let ring_cap = dft_payload::ring_capacity(w, h);
    if ring_cap > 0 && !message.is_empty() {
        // Truncate at a valid UTF-8 character boundary to avoid splitting
        // multi-byte characters (e.g., emoji, CJK, accented chars).
        let max_byte = message.len().min(ring_cap);
        let truncated_len = message[..max_byte]
            .char_indices()
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        let truncated = &message.as_bytes()[..truncated_len];
        dft_payload::embed_ring_payload(&mut spectrum, truncated, passphrase);
    }

    // P0: Drop spectrum after IFFT — only needed to produce the modified pixels.
    let modified = fft2d::ifft2d(&spectrum);
    drop(spectrum);

    pixels::luma_f64_to_jpeg(&modified, w, h, img);
    Ok(())
}

// --- Fortress QF75 pre-settlement ---

/// Standard JPEG luminance quantization table (Table K.1 from the JPEG spec).
/// These are the base values before quality-factor scaling.
/// NOTE: Duplicates STD_LUMA_QT in selection.rs — kept here to avoid coupling
/// the Fortress pre-settlement path to the stability-map module.
const JPEG_BASE_LUMINANCE_QT: [u16; 64] = [
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99,
];

/// Standard JPEG chrominance quantization table (Table K.2 from the JPEG spec).
/// NOTE: Duplicates STD_CHROMA_QT in selection.rs — kept here for the same reason.
const JPEG_BASE_CHROMINANCE_QT: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];

/// Compute a JPEG quantization table for a given quality factor (1-100).
///
/// Uses the standard libjpeg scaling formula:
/// - QF >= 50: scale = 200 - 2 * QF
/// - QF <  50: scale = 5000 / QF
///
/// NOTE: Duplicates scale_quant_table() in selection.rs — kept here to avoid
/// coupling the Fortress pre-settlement path to the stability-map module.
fn compute_jpeg_qt(base: &[u16; 64], qf: u32) -> [u16; 64] {
    let scale = if qf >= 50 { 200 - 2 * qf } else { 5000 / qf };
    let mut qt = [0u16; 64];
    for i in 0..64 {
        let val = (base[i] as u32 * scale + 50) / 100;
        qt[i] = val.clamp(1, 255) as u16;
    }
    qt
}

/// Pre-settle all image components to QF75 quantization tables.
///
/// For each component, performs IDCT → pixel clamp → DCT with QF75 QT tables,
/// then replaces the component's quantization table. This makes the coefficients
/// (especially Y-channel DCs used by Fortress) nearly idempotent under Q75
/// recompression, which is what WhatsApp and most social media platforms use.
///
/// Without this, platform-side quality settings vary wildly (iOS 0.65 ≈ Q90,
/// Android Q70, Web 0.65 ≈ varies), leaving coefficients on a fine grid that
/// shifts dramatically when recompressed to Q75.
fn pre_settle_for_fortress(img: &mut JpegImage) {
    use crate::jpeg::dct::QuantTable;

    let num_components = img.num_components();
    let target_qf = 75u32;

    // Pre-compute target QT for each QT slot (luminance vs chrominance base).
    // We must re-quantize ALL components, even those sharing a QT ID,
    // because each component has its own DctGrid with separate coefficients.
    let mut new_qts: Vec<(usize, [u16; 64], [u16; 64])> = Vec::new(); // (qt_id, old, new)

    for comp_idx in 0..num_components {
        let qt_id = img.frame_info().components[comp_idx].quant_table_id as usize;
        let old_qt = img.quant_table(qt_id).expect("quant table must exist").values;
        let base = if comp_idx == 0 {
            &JPEG_BASE_LUMINANCE_QT
        } else {
            &JPEG_BASE_CHROMINANCE_QT
        };
        let new_qt = compute_jpeg_qt(base, target_qf);

        // Re-quantize all blocks in this component — parallel when available.
        let grid = img.dct_grid_mut(comp_idx);

        let process_block = |chunk: &mut [i16]| {
            let quantized: [i16; 64] = chunk.try_into().unwrap();
            let mut px = pixels::idct_block(&quantized, &old_qt);
            for p in px.iter_mut() {
                *p = p.clamp(0.0, 255.0);
            }
            let settled = pixels::dct_block(&px, &new_qt);
            chunk.copy_from_slice(&settled);
        };

        #[cfg(feature = "parallel")]
        grid.coeffs_mut().par_chunks_mut(64).for_each(process_block);
        #[cfg(not(feature = "parallel"))]
        grid.coeffs_mut().chunks_mut(64).for_each(process_block);

        new_qts.push((qt_id, old_qt, new_qt));
    }

    // Replace quantization tables (deduplicate by QT ID)
    let mut replaced = [false; 4];
    for (qt_id, _old, new_qt) in &new_qts {
        if !replaced[*qt_id] {
            img.set_quant_table(*qt_id, QuantTable::new(*new_qt));
            replaced[*qt_id] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_integrity_pristine() {
        // Pristine: signal_strength == reference, 0 RS errors
        let stats = ecc::RsDecodeStats {
            total_errors: 0,
            error_capacity: 32,
            max_block_errors: 0,
            num_blocks: 1,
        };
        let integrity = compute_integrity(15.0, &stats, 15.0);
        // 0.7 * 1.0 + 0.3 * 1.0 = 1.0 → 100
        assert_eq!(integrity, 100);
    }

    #[test]
    fn compute_integrity_half_signal() {
        // Signal is half the reference, no RS errors
        let stats = ecc::RsDecodeStats {
            total_errors: 0,
            error_capacity: 32,
            max_block_errors: 0,
            num_blocks: 1,
        };
        let integrity = compute_integrity(7.5, &stats, 15.0);
        // 0.7 * 0.5 + 0.3 * 1.0 = 0.65 → 65
        assert_eq!(integrity, 65);
    }

    #[test]
    fn compute_integrity_zero_signal() {
        // No signal, no RS errors
        let stats = ecc::RsDecodeStats {
            total_errors: 0,
            error_capacity: 32,
            max_block_errors: 0,
            num_blocks: 1,
        };
        let integrity = compute_integrity(0.0, &stats, 15.0);
        // 0.7 * 0.0 + 0.3 * 1.0 = 0.3 → 30
        assert_eq!(integrity, 30);
    }

    #[test]
    fn compute_integrity_with_rs_errors() {
        // Full signal, half RS capacity used
        let stats = ecc::RsDecodeStats {
            total_errors: 16,
            error_capacity: 32,
            max_block_errors: 16,
            num_blocks: 1,
        };
        let integrity = compute_integrity(15.0, &stats, 15.0);
        // 0.7 * 1.0 + 0.3 * 0.5 = 0.85 → 85
        assert_eq!(integrity, 85);
    }

    #[test]
    fn compute_integrity_both_degraded() {
        // Half signal, half RS capacity used
        let stats = ecc::RsDecodeStats {
            total_errors: 16,
            error_capacity: 32,
            max_block_errors: 16,
            num_blocks: 1,
        };
        let integrity = compute_integrity(7.5, &stats, 15.0);
        // 0.7 * 0.5 + 0.3 * 0.5 = 0.5 → 50
        assert_eq!(integrity, 50);
    }

    #[test]
    fn compute_integrity_signal_exceeds_reference() {
        // Signal > reference (clamped to 1.0)
        let stats = ecc::RsDecodeStats {
            total_errors: 0,
            error_capacity: 32,
            max_block_errors: 0,
            num_blocks: 1,
        };
        let integrity = compute_integrity(20.0, &stats, 15.0);
        // 0.7 * 1.0 + 0.3 * 1.0 = 1.0 → 100 (clamped)
        assert_eq!(integrity, 100);
    }

    #[test]
    fn compute_integrity_zero_reference() {
        // Edge case: reference_llr = 0 → llr_score defaults to 1.0
        let stats = ecc::RsDecodeStats {
            total_errors: 0,
            error_capacity: 0,
            max_block_errors: 0,
            num_blocks: 0,
        };
        let integrity = compute_integrity(5.0, &stats, 0.0);
        // llr_score = 1.0 (fallback), rs_score = 1.0 (error_capacity == 0)
        assert_eq!(integrity, 100);
    }

    #[test]
    fn compute_avg_abs_llr_basic() {
        let llrs = vec![5.0, -3.0, 4.0, -2.0];
        let avg = compute_avg_abs_llr(&llrs);
        // (5 + 3 + 4 + 2) / 4 = 3.5
        assert!((avg - 3.5).abs() < 1e-10);
    }

    #[test]
    fn compute_avg_abs_llr_empty() {
        assert_eq!(compute_avg_abs_llr(&[]), 0.0);
    }

    #[test]
    fn decode_quality_ghost_unchanged() {
        let q = DecodeQuality::ghost();
        assert_eq!(q.integrity_percent, 100, "Ghost always 100%");
        assert_eq!(q.signal_strength, 0.0);
    }

    #[test]
    fn decode_quality_from_rs_stats_with_signal_pristine() {
        let stats = ecc::RsDecodeStats {
            total_errors: 0,
            error_capacity: 32,
            max_block_errors: 0,
            num_blocks: 1,
        };
        let q = DecodeQuality::from_rs_stats_with_signal(&stats, 5, 64, 15.0, 15.0);
        assert_eq!(q.integrity_percent, 100);
        assert!((q.signal_strength - 15.0).abs() < 1e-10);
        assert_eq!(q.repetition_factor, 5);
        assert_eq!(q.parity_len, 64);
    }

    #[test]
    fn decode_quality_legacy_constructor_still_works() {
        // from_rs_stats_with_phase2 should still give 100% for 0 errors
        let stats = ecc::RsDecodeStats {
            total_errors: 0,
            error_capacity: 32,
            max_block_errors: 0,
            num_blocks: 1,
        };
        let q = DecodeQuality::from_rs_stats_with_phase2(&stats, 5, 64);
        assert_eq!(q.integrity_percent, 100);
        assert_eq!(q.signal_strength, 0.0);
    }
}
