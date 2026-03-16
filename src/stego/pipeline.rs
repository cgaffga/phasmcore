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
use crate::stego::shadow;
use crate::stego::stc::{embed, extract, hhat};

/// STC constraint length for Ghost Phase 1.
const STC_H: usize = 7;

/// Progress steps allocated to JPEG parsing.  Reported immediately after
/// `JpegImage::from_bytes` returns so the bar moves off 0% right away.
const PARSE_STEPS: u32 = 5;

/// Total number of progress steps reported by [`ghost_decode`].
///
/// 5 (parse) + 100 (UNIWARD) + 2 (STC extraction + decryption).
pub const GHOST_DECODE_STEPS: u32 = PARSE_STEPS + crate::stego::cost::uniward::UNIWARD_PROGRESS_STEPS + 2;

/// Total number of progress steps reported by Ghost encode.
///
/// 5 (parse) + 100 (UNIWARD sub-steps) + 50 (STC Viterbi sub-steps) + 2 (permute + LSB mod/JPEG write).
pub const GHOST_ENCODE_STEPS: u32 =
    PARSE_STEPS
    + crate::stego::cost::uniward::UNIWARD_PROGRESS_STEPS
    + crate::stego::stc::embed::STC_PROGRESS_STEPS
    + 2;

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
    let w = n / m_max; // floor division
    let n_used = m_max * w;

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

/// Shadow layer descriptor for encoding.
pub struct ShadowLayer {
    /// Text message for this shadow layer.
    pub message: String,
    /// Passphrase for this shadow layer (must be unique across all layers).
    pub passphrase: String,
    /// Optional file attachments for this shadow layer.
    pub files: Vec<FileEntry>,
}

/// Base progress steps for Ghost encode with shadows (before DPC iterations).
///
/// 5 (parse) + 100 (UNIWARD) + 1 (shadow prepare).
/// DPC adds: K shadows × up to 3 iterations × (1 embed + STC_PROGRESS_STEPS + 1 check) + 2 (verify + write).
/// Total is set dynamically once shadow count is known.
pub const GHOST_ENCODE_WITH_SHADOWS_STEPS: u32 =
    PARSE_STEPS
    + crate::stego::cost::uniward::UNIWARD_PROGRESS_STEPS
    + 1
    + crate::stego::stc::embed::STC_PROGRESS_STEPS
    + 2;

/// Encode with Ghost mode plus shadow messages for plausible deniability.
///
/// Shadow layers are embedded first using repetition coding. The primary
/// message is then embedded via STC, treating shadow modifications as
/// part of the cover. Each passphrase must be unique.
pub fn ghost_encode_with_shadows(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    shadows: &[ShadowLayer],
    si: Option<SideInfo>,
) -> Result<Vec<u8>, StegoError> {
    ghost_encode_with_shadows_impl(image_bytes, message, files, passphrase, shadows, si)
}

/// Encode with Ghost SI-UNIWARD plus shadow messages.
pub fn ghost_encode_si_with_shadows(
    image_bytes: &[u8],
    raw_pixels_rgb: &[u8],
    pixel_width: u32,
    pixel_height: u32,
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    shadows: &[ShadowLayer],
) -> Result<Vec<u8>, StegoError> {
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

    drop(img);

    ghost_encode_with_shadows_impl(image_bytes, message, files, passphrase, shadows, Some(si))
}

fn ghost_encode_with_shadows_impl(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    shadows: &[ShadowLayer],
    si: Option<SideInfo>,
) -> Result<Vec<u8>, StegoError> {
    // Initialize progress.
    progress::init(GHOST_ENCODE_WITH_SHADOWS_STEPS);

    // Validate passphrases are unique (primary + all shadows).
    {
        let mut all_passes: Vec<&str> = vec![passphrase];
        for s in shadows {
            all_passes.push(&s.passphrase);
        }
        for i in 0..all_passes.len() {
            for j in (i + 1)..all_passes.len() {
                if all_passes[i] == all_passes[j] {
                    return Err(StegoError::DuplicatePassphrase);
                }
            }
        }
    }

    // Auto-sort: the largest payload becomes primary (gets full STC stealth).
    // Smaller payloads become shadows (direct LSB + RS, negligible detectability).
    let primary_payload_size = payload::compressed_payload_size(message, files);
    let mut swap_idx: Option<usize> = None;
    for (i, s) in shadows.iter().enumerate() {
        let shadow_size = payload::compressed_payload_size(&s.message, &s.files);
        if shadow_size > primary_payload_size {
            if let Some(prev) = swap_idx {
                let prev_size = payload::compressed_payload_size(&shadows[prev].message, &shadows[prev].files);
                if shadow_size > prev_size {
                    swap_idx = Some(i);
                }
            } else {
                swap_idx = Some(i);
            }
        }
    }

    // If a shadow is larger, swap it with primary for optimal stealth.
    // We need `primary_as_shadow` to live long enough, so declare it before the branch.
    let primary_as_shadow;
    let (eff_message, eff_files, eff_passphrase, eff_shadows);
    if let Some(idx) = swap_idx {
        eff_message = shadows[idx].message.as_str();
        eff_files = &shadows[idx].files[..];
        eff_passphrase = shadows[idx].passphrase.as_str();
        // Build new shadows list: original primary + all shadows except the swapped one.
        primary_as_shadow = ShadowLayer {
            message: message.to_string(),
            passphrase: passphrase.to_string(),
            files: files.to_vec(),
        };
        let mut new_shadows: Vec<&ShadowLayer> = Vec::with_capacity(shadows.len());
        new_shadows.push(&primary_as_shadow);
        for (i, s) in shadows.iter().enumerate() {
            if i != idx {
                new_shadows.push(s);
            }
        }
        eff_shadows = new_shadows;
    } else {
        eff_message = message;
        eff_files = files;
        eff_passphrase = passphrase;
        eff_shadows = shadows.iter().collect();
    };

    // Build the primary payload.
    let payload_bytes = payload::encode_payload(eff_message, eff_files)?;

    let mut img = JpegImage::from_bytes(image_bytes)?;
    progress::advance_by(PARSE_STEPS);
    let fi = img.frame_info();
    super::validate_encode_dimensions(fi.width as u32, fi.height as u32)?;

    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    // 1. Compute J-UNIWARD positions.
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    let si_ref = si.as_ref().map(|s| (s, img.dct_grid(0)));
    let mut positions = compute_positions_streaming(img.dct_grid(0), qt, si_ref)?;

    // Save a clone of all Y positions (pre-permutation) for shadow,
    // sorted by cost (cheapest first) for cost-pool selection.
    let mut shadow_positions = positions.clone();
    shadow_positions.sort_by(|a, b| a.cost.total_cmp(&b.cost));

    // 2. Prepare shadow layers (Y-channel direct LSB + RS).
    let mut shadow_states: Vec<shadow::ShadowState> = Vec::new();
    if !eff_shadows.is_empty() {
        let initial_parity = 4;
        for s in eff_shadows.iter() {
            let state = shadow::prepare_shadow(
                &shadow_positions,
                &s.passphrase,
                &s.message,
                &s.files,
                initial_parity,
            )?;
            shadow_states.push(state);
        }
    }
    progress::advance(); // shadow preparation step

    // Save original Y grid for restoration before embedding.
    let original_y = img.dct_grid(0).clone();

    // 3. Primary STC setup.
    let structural_key = crypto::derive_structural_key(eff_passphrase)?;
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    permute::permute_positions(&mut positions, &perm_seed);
    let n = positions.len();

    // Step: permutation complete.
    progress::advance();

    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, eff_passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
    let frame_bits = frame::bytes_to_bits(&frame_bytes);
    let m = frame_bits.len();

    // Dynamic w: pick highest that fits the actual primary message.
    let w = (n / m).min(10).max(1);
    let m_max = n / w;
    let n_used = m_max * w;

    if m > m_max {
        return Err(StegoError::MessageTooLarge);
    }

    const STC_POSITION_LIMIT: usize = 500_000_000;
    if n_used > STC_POSITION_LIMIT {
        return Err(StegoError::ImageTooLarge);
    }

    positions.truncate(n_used);
    let hhat_matrix = hhat::generate_hhat(STC_H, w, &hhat_seed);

    // 4. Build ∞-cost mask for shadow positions when w >= 2.
    // When w >= 2, Viterbi has enough slack to route around shadow positions,
    // achieving BER ~ 0% on shadows without any RS correction needed.
    let shadow_inf_costs = build_inf_cost_set(w, &shadow_states);

    // 5. Embed shadows + primary STC (always short STC: frame_bits, not padded).
    if shadow_states.is_empty() {
        run_stc_pass(&mut img, &original_y, &positions, &[],
                     &frame_bits, &hhat_matrix, w, &si, &None)?;
    } else {
        run_stc_pass(&mut img, &original_y, &positions, &shadow_states,
                     &frame_bits, &hhat_matrix, w, &si, &shadow_inf_costs)?;

        // Verify shadows from the DECODER's perspective: compute UNIWARD costs
        // on the stego image (as the decoder will see it), select positions
        // using stego costs + same fraction/hash, and check RS decode + decrypt.
        // This catches cost-pool boundary disagreements between cover and stego.
        let qt_verify = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
        let mut stego_y_positions = compute_positions_streaming(img.dct_grid(0), qt_verify, None)?;
        stego_y_positions.sort_by(|a, b| a.cost.total_cmp(&b.cost));

        if !verify_all_shadows_decoder_side(&img, &shadow_states, &eff_shadows, &stego_y_positions) {
            // Escalate: try progressively larger fractions (more positions,
            // lower BER) and higher parities (more error correction).
            // Ordered by stealth preference: aggressive combos first.
            const CASCADE: &[(usize, usize)] = &[
                // (fraction_denom, parity): smaller denom = bigger pool = less BER
                (10, 4), (5, 4), (2, 4), (1, 4),
                (10, 16), (5, 16), (2, 16), (1, 16),
                (5, 32), (2, 32), (1, 32),
                (2, 64), (1, 64),
                (1, 128),
            ];
            let mut verified = false;
            for &(frac, par) in CASCADE {
                for state in shadow_states.iter_mut() {
                    shadow::rebuild_shadow(state, &shadow_positions, par, frac)?;
                }
                let new_inf_costs = build_inf_cost_set(w, &shadow_states);
                run_stc_pass(&mut img, &original_y, &positions, &shadow_states,
                             &frame_bits, &hhat_matrix, w, &si, &new_inf_costs)?;

                // Recompute stego costs after re-encoding.
                let qt_re = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
                stego_y_positions = compute_positions_streaming(img.dct_grid(0), qt_re, None)?;
                stego_y_positions.sort_by(|a, b| a.cost.total_cmp(&b.cost));

                if verify_all_shadows_decoder_side(&img, &shadow_states, &eff_shadows, &stego_y_positions) {
                    verified = true;
                    break;
                }
            }
            if !verified {
                return Err(StegoError::MessageTooLarge);
            }
        }
    }

    // Write JPEG.
    let stego_bytes = match img.to_bytes() {
        Ok(bytes) => bytes,
        Err(_) => {
            img.rebuild_huffman_tables();
            img.to_bytes().map_err(StegoError::InvalidJpeg)?
        }
    };

    progress::advance();

    Ok(stego_bytes)
}

/// Run a single STC pass: restore Y grid, embed shadows, then run STC with optional ∞-cost.
fn run_stc_pass(
    img: &mut JpegImage,
    original_y: &crate::jpeg::dct::DctGrid,
    positions: &[permute::CoeffPos],
    shadow_states: &[shadow::ShadowState],
    message_bits: &[u8],
    hhat_matrix: &[Vec<u32>],
    w: usize,
    si: &Option<SideInfo>,
    shadow_inf_costs: &Option<std::collections::HashSet<u32>>,
) -> Result<(), StegoError> {
    *img.dct_grid_mut(0) = original_y.clone();

    for state in shadow_states {
        shadow::embed_shadow_lsb(img, state);
    }

    let grid = img.dct_grid(0);
    let cover_bits: Vec<u8> = positions.iter().map(|p| {
        let coeff = flat_get(grid, p.flat_idx as usize);
        (coeff.unsigned_abs() & 1) as u8
    }).collect();

    // Build cost vector with ∞ for shadow positions when w >= 2.
    let costs: Vec<f32> = if let Some(inf_set) = shadow_inf_costs {
        positions.iter().map(|p| {
            if inf_set.contains(&p.flat_idx) {
                f32::INFINITY
            } else {
                p.cost
            }
        }).collect()
    } else {
        positions.iter().map(|p| p.cost).collect()
    };

    let result = embed::stc_embed(&cover_bits, &costs, message_bits, hhat_matrix, STC_H, w);
    progress::check_cancelled()?;
    let result = result.ok_or(StegoError::MessageTooLarge)?;

    apply_stc_changes(img, positions, &cover_bits, &result.stego_bits, si);

    Ok(())
}

/// Apply STC LSB changes to the DctGrid.
fn apply_stc_changes(
    img: &mut JpegImage,
    positions: &[permute::CoeffPos],
    cover_bits: &[u8],
    stego_bits: &[u8],
    si: &Option<SideInfo>,
) {
    let grid_mut = img.dct_grid_mut(0);
    for (idx, pos) in positions.iter().enumerate() {
        if cover_bits[idx] != stego_bits[idx] {
            let fi = pos.flat_idx as usize;
            let coeff = flat_get(grid_mut, fi);
            let modified = if let Some(side_info) = si {
                side_info::si_modify_coefficient(coeff, side_info.error_at(fi))
            } else {
                side_info::nsf5_modify_coefficient(coeff)
            };
            flat_set(grid_mut, fi, modified);
        }
    }
}

/// Check if all shadow layers can be decoded from the decoder's perspective.
///
/// Uses stego-image UNIWARD costs to select positions (simulating what the
/// decoder will do), then checks RS decode + AES-GCM decrypt.
fn verify_all_shadows_decoder_side(
    img: &JpegImage,
    shadow_states: &[shadow::ShadowState],
    shadows: &[&ShadowLayer],
    stego_y_positions_sorted: &[permute::CoeffPos],
) -> bool {
    for (i, state) in shadow_states.iter().enumerate() {
        if shadow::verify_shadow_decoder_side(
            img, state, &shadows[i].passphrase, stego_y_positions_sorted,
        ).is_err() {
            return false;
        }
    }
    true
}

/// Build the ∞-cost HashSet for shadow positions (used when w >= 2).
fn build_inf_cost_set(w: usize, shadow_states: &[shadow::ShadowState]) -> Option<std::collections::HashSet<u32>> {
    if w >= 2 && !shadow_states.is_empty() {
        let mut set = std::collections::HashSet::new();
        for state in shadow_states {
            for pos in &state.positions {
                set.insert(pos.flat_idx);
            }
        }
        Some(set)
    } else {
        None
    }
}

fn ghost_encode_impl(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    si: Option<SideInfo>,
) -> Result<Vec<u8>, StegoError> {
    // Initialize encode progress (100 UNIWARD + 50 STC + 2 misc).
    progress::init(GHOST_ENCODE_STEPS);

    // Build the payload (text + files + compression).
    let payload_bytes = payload::encode_payload(message, files)?;

    let mut img = JpegImage::from_bytes(image_bytes)?;
    progress::advance_by(PARSE_STEPS);

    // Validate dimensions before any heavy processing.
    let fi = img.frame_info();
    super::validate_encode_dimensions(fi.width as u32, fi.height as u32)?;

    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    // 1. Compute J-UNIWARD costs strip-by-strip and collect positions directly.
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    let si_ref = si.as_ref().map(|s| (s, img.dct_grid(0)));
    let mut positions = compute_positions_streaming(img.dct_grid(0), qt, si_ref)?;

    // 2. Derive structural key (Tier 1).
    let structural_key = crypto::derive_structural_key(passphrase)?;
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    // 3. Permute positions.
    permute::permute_positions(&mut positions, &perm_seed);
    let n = positions.len();

    // Step: permutation + key derivation complete.
    progress::advance();

    // 4. Encrypt payload (Tier 2 key with random salt).
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;

    // 5. Build payload frame.
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
    let frame_bits = frame::bytes_to_bits(&frame_bytes);
    let m = frame_bits.len();

    // 6. Dynamic w: pick highest that fits the actual message (short STC).
    let w = (n / m).min(10).max(1);
    let m_max = n / w;
    let n_used = m_max * w;

    if m > m_max {
        return Err(StegoError::MessageTooLarge);
    }

    // Memory budget check (same as compute_stc_params).
    const STC_POSITION_LIMIT: usize = 500_000_000;
    if n_used > STC_POSITION_LIMIT {
        return Err(StegoError::ImageTooLarge);
    }

    positions.truncate(n_used);

    // 7. Extract cover LSBs and costs in permuted order.
    let grid = img.dct_grid(0);
    let cover_bits: Vec<u8> = positions.iter().map(|p| {
        let coeff = flat_get(grid, p.flat_idx as usize);
        (coeff.unsigned_abs() & 1) as u8
    }).collect();
    let costs: Vec<f32> = positions.iter().map(|p| p.cost).collect();

    // 8. Generate H-hat and embed with short STC (actual m bits, not padded).
    let hhat_matrix = hhat::generate_hhat(STC_H, w, &hhat_seed);
    let result = embed::stc_embed(&cover_bits, &costs, &frame_bits, &hhat_matrix, STC_H, w);
    progress::check_cancelled()?;
    let result = result.ok_or(StegoError::MessageTooLarge)?;

    // 9. Apply LSB changes to DctGrid.
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

    // 10. Write modified JPEG.
    let stego_bytes = match img.to_bytes() {
        Ok(bytes) => bytes,
        Err(_) => {
            img.rebuild_huffman_tables();
            img.to_bytes().map_err(StegoError::InvalidJpeg)?
        }
    };

    // Final step: LSB modification + JPEG write complete.
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
    progress::advance_by(PARSE_STEPS);

    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    // 1. Compute J-UNIWARD costs strip-by-strip and collect positions directly.
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    let mut positions = compute_positions_streaming(img.dct_grid(0), qt, None)?;

    progress::check_cancelled()?;

    // 2. Derive structural key.
    let structural_key = crypto::derive_structural_key(passphrase)?;
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    // 3. Permute positions.
    permute::permute_positions(&mut positions, &perm_seed);
    let n = positions.len();

    // Step: permutation complete.
    progress::advance();

    // 4. Extract all stego LSBs (full n, reused across w candidates).
    let grid = img.dct_grid(0);
    let all_stego_bits: Vec<u8> = positions.iter().map(|p| {
        let coeff = flat_get(grid, p.flat_idx as usize);
        (coeff.unsigned_abs() & 1) as u8
    }).collect();

    // 5. Brute-force w: try the natural w first (backward compat), then dynamic candidates.
    let w_natural = compute_stc_params(n).map(|(w, _, _)| w).unwrap_or(1);
    let w_candidates: &[usize] = &[w_natural, 1, 2, 3, 5, 7, 10];
    let mut saw_decrypt_fail = false;
    let mut tried_w = 0u16; // bitmask to skip duplicates

    for &w in w_candidates {
        if w == 0 || n / w == 0 {
            continue;
        }
        // Dedup: skip if we've already tried this w value.
        if w <= 15 && (tried_w & (1 << w)) != 0 {
            continue;
        }
        if w <= 15 {
            tried_w |= 1 << w;
        }

        let m_max = n / w;
        let n_used = m_max * w;
        if n_used > all_stego_bits.len() {
            continue;
        }

        let stego_bits = &all_stego_bits[..n_used];
        let hhat_matrix = hhat::generate_hhat(STC_H, w, &hhat_seed);
        let extracted_bits = extract::stc_extract(stego_bits, &hhat_matrix, w);

        let frame_bytes = frame::bits_to_bytes(&extracted_bits[..m_max]);
        match try_parse_and_decrypt(&frame_bytes, passphrase) {
            Ok(payload) => {
                progress::advance();
                return Ok(payload);
            }
            Err(StegoError::DecryptionFailed) => {
                saw_decrypt_fail = true;
            }
            Err(_) => {}
        }
    }

    // Step: STC extraction + decryption complete (all attempts failed).
    progress::advance();

    if saw_decrypt_fail {
        Err(StegoError::DecryptionFailed)
    } else {
        Err(StegoError::FrameCorrupted)
    }
}

/// Helper: parse frame, decrypt, and decode payload. Used by brute-force w loop.
fn try_parse_and_decrypt(
    frame_bytes: &[u8],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    let parsed = frame::parse_frame(frame_bytes)?;
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
    payload::decode_payload(&plaintext[..len])
}

/// Decode a shadow message from a stego JPEG.
///
/// Shadow uses cost-pool position selection (UNIWARD costs for stealth)
/// plus hash permutation and RS error correction.
pub fn ghost_shadow_decode(
    stego_bytes: &[u8],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    let img = JpegImage::from_bytes(stego_bytes)?;
    ghost_shadow_decode_from_image(&img, passphrase)
}

/// Decode a shadow message from an already-parsed JPEG image.
///
/// Computes Y-channel UNIWARD costs, sorts by cost (cheapest first), and
/// passes to the shadow extractor which brute-forces (fraction, parity, fdl).
pub fn ghost_shadow_decode_from_image(
    img: &JpegImage,
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }

    // Compute Y-channel UNIWARD costs for cost-pool selection.
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    let mut positions = compute_positions_streaming(img.dct_grid(0), qt, None)?;

    // Sort by cost (cheapest first) for cost-pool tier selection.
    positions.sort_by(|a, b| a.cost.total_cmp(&b.cost));

    shadow::shadow_extract(img, &positions, passphrase)
}

// --- DctGrid flat access helpers ---

use crate::jpeg::dct::DctGrid;

/// Read a coefficient from a `DctGrid` using a flat index.
///
/// The flat index encodes `block_index * 64 + row * 8 + col`, where
/// `block_index = block_row * blocks_wide + block_col`.
pub(super) fn flat_get(grid: &DctGrid, flat_idx: usize) -> i16 {
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
pub(super) fn flat_set(grid: &mut DctGrid, flat_idx: usize, val: i16) {
    let bw = grid.blocks_wide();
    let block_idx = flat_idx / 64;
    let pos = flat_idx % 64;
    let br = block_idx / bw;
    let bc = block_idx % bw;
    let i = pos / 8;
    let j = pos % 8;
    grid.set(br, bc, i, j, val);
}
