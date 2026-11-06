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
//! - Fortress T: tiled layout + key-free grid sync, so short messages survive
//!   crops, rotation and rescaling (`fortress_tile`, `gridsync`)

use crate::codec::jpeg::JpegImage;
use crate::codec::jpeg::dct::{DctGrid, QuantTable};
use crate::codec::jpeg::pixels;
// Armor's pre-clamp + QF-settle pixel-domain round-trips run
// through the integer LL&M path (NEON / AVX2 / WASM SIMD / scalar,
// all bit-exact). See `docs/design/image/t3.1-integer-aan.md`.
use crate::codec::jpeg::pixels_aan::{
    aan_dct_block as dct_block_active, aan_idct_block as idct_block_active,
};
use crate::stego::armor::ecc;
use crate::stego::armor::embedding::{self, stdm_embed, stdm_extract_soft};
use crate::stego::armor::fortress;
use crate::stego::armor::fortress_tile;
use crate::stego::armor::fortress_tl;
use crate::stego::armor::repetition;
use crate::stego::armor::selection;
use crate::stego::armor::spreading::{generate_spreading_vectors, spreading_vectors_range, SPREAD_LEN};
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload::{self, PayloadData};
use crate::stego::progress;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::stego::quality::{self, EncodeQuality, ArmorMetrics};

/// Number of embedding units for the header.
/// 1 byte x 8 bits x 7 copies = 56 units.
const HEADER_UNITS: usize = embedding::HEADER_UNITS; // 56
const HEADER_COPIES: usize = embedding::HEADER_COPIES; // 7

/// Number of sub-steps reported during the STDM embed loop. Subdividing
/// the per-bit loop (~5000 iterations at 12 MP) eliminates a multi-
/// hundred-ms "step jump" — the bar moves smoothly across the embed
/// phase instead of sitting still then leaping forward.
const STDM_EMBED_PROGRESS_STEPS: u32 = 5;

/// Total progress steps for Armor encode via STDM path.
///
/// Post-T3 recalibration (was 6):
/// - 1 (pre-clamp)
/// - 1 (stability map)
/// - 1 (RS+spreading)
/// - STDM_EMBED_PROGRESS_STEPS (STDM embed loop, subdivided)
/// - JPEG_WRITE_STEPS (JPEG entropy-coded scan write, subdivided)
///
/// Per-step wall-clock at 12 MP: ~30-150 ms. Smooth bar against
/// wall-clock-100% target.
pub const ARMOR_ENCODE_STEPS: u32 =
    1
    + 1
    + 1
    + STDM_EMBED_PROGRESS_STEPS
    + crate::codec::jpeg::scan::JPEG_WRITE_STEPS;

/// Total progress steps for Armor encode via Fortress path.
///
/// Post-T3 recalibration (was 3):
/// - 3 (pre-settle per channel: Y, Cb, Cr — biggest wall-clock share)
/// - 1 (BA-QIM fortress encode)
/// - JPEG_WRITE_STEPS (JPEG write, subdivided)
const ARMOR_ENCODE_FORTRESS_STEPS: u32 =
    3
    + 1
    + crate::codec::jpeg::scan::JPEG_WRITE_STEPS;

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
    armor_encode_impl(image_bytes, message, passphrase, &ArmorEncodeOpts::default(), &mut ArmorEncodeInfo::default())
        .map(|(bytes, _)| bytes)
}

/// Encode with Armor mode and return the encode quality score.
pub fn armor_encode_with_quality(
    image_bytes: &[u8],
    message: &str,
    passphrase: &str,
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    armor_encode_impl(image_bytes, message, passphrase, &ArmorEncodeOpts::default(), &mut ArmorEncodeInfo::default())
}

/// Armor encode with the user's "Standard" switch: `force_standard` skips
/// Fortress even when the message fits it (fix list 2.8 / C7). The returned
/// quality's `fortress` is the sub-mode actually used.
pub fn armor_encode_with_options(
    image_bytes: &[u8],
    message: &str,
    passphrase: &str,
    force_standard: bool,
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    let opts = ArmorEncodeOpts { force_stdm: force_standard, ..ArmorEncodeOpts::default() };
    armor_encode_impl(image_bytes, message, passphrase, &opts, &mut ArmorEncodeInfo::default())
}

/// STDM wire variant written by the encoder (prototype switch, 2026-10-31).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StdmVariant {
    /// Shipped format: STDM on quantized indices, delta = mult x mean_qt
    /// (a DCT-unit step applied to indices), header at BOOTSTRAP_DELTA.
    Legacy,
    /// Legacy wire (old decoders read it) with the step corrected to the index
    /// domain: the header carries `h` instead of the real mean QT, so every
    /// decoder computes delta = mult x h index units.
    LegacyIdx { h: f64 },
    /// Armor v2: STDM on dequantized coefficients (index x q), delta in DCT
    /// units, positions on the real block grid, header = magic + mean QT.
    V2,
}

/// Encoder knobs for the Armor v2 prototype.
#[derive(Debug, Clone)]
pub struct ArmorEncodeOpts {
    pub stdm: StdmVariant,
    /// delta multipliers for r = 1, r = 2, r 3-4, r 5-6, r >= 7.
    pub mults: [f64; 5],
    /// v2 header step in DCT units.
    pub v2_header_delta: f64,
    /// v2: delta = mult x max(header mean QT, v2_floor) DCT units.
    pub v2_floor: f64,
    pub rounding: embedding::StdmRounding,
    /// Skip Fortress even when the message fits (measurement only).
    pub force_stdm: bool,
    /// Luma headroom in gray levels: the pre-clamp / Fortress pre-settle maps
    /// luma through a soft knee into [h, 255 - h] so embedding changes in
    /// near-white / near-black areas are not clipped by the receiver's
    /// decoder. 0 = off (shipped behaviour).
    pub headroom: f64,
    /// Fortress CC (prototype wire variant) instead of Fortress RS + repetition.
    pub fortress_cc: Option<fortress::FortressCcParams>,
    /// v2: requantize the luma of covers finer than IJG QF65 to (at least) the
    /// QF65 table before embedding.
    pub settle_floor: bool,
    /// Fortress T (tiled layout + pilots, geometry-robust) instead of Fortress CC.
    pub fortress_tiled: bool,
    /// With Fortress T: Fortress TL (the AC-lane payload) for messages longer
    /// than Fortress T holds.
    pub fortress_tl: bool,
    /// Fortress T checkerboard sync amplitude (gray levels, busy blocks).
    pub sync_amp: f64,
}

impl Default for ArmorEncodeOpts {
    /// The production encoder: Armor v2 STDM with automatic luma headroom.
    fn default() -> Self {
        Self {
            stdm: StdmVariant::V2,
            mults: embedding::V2_MULTS,
            v2_header_delta: embedding::V2_HEADER_DELTA,
            v2_floor: embedding::V2_MEAN_QT_FLOOR,
            rounding: embedding::StdmRounding::Search { tol: embedding::V2_ROUNDING_TOL },
            force_stdm: false,
            headroom: -1.0,
            fortress_cc: Some(fortress::FortressCcParams::default()),
            settle_floor: true,
            fortress_tiled: true,
            fortress_tl: true,
            sync_amp: fortress_tile::SYNC_AMP,
        }
    }
}

impl ArmorEncodeOpts {
    /// The encoder of core <= 0.5.x (legacy STDM, no headroom), byte-compatible
    /// with the released decoders.
    pub fn legacy() -> Self {
        Self {
            stdm: StdmVariant::Legacy,
            mults: embedding::LEGACY_MULTS,
            rounding: embedding::StdmRounding::Naive,
            headroom: 0.0,
            fortress_cc: None,
            settle_floor: false,
            fortress_tiled: false,
            fortress_tl: false,
            ..Self::default()
        }
    }
}

/// What an encode did (measurement only).
#[derive(Debug, Clone, Default)]
pub struct ArmorEncodeInfo {
    pub fortress: bool,
    /// Fortress TL (the AC-lane payload) rather than Fortress T.
    pub fortress_tl: bool,
    pub r: usize,
    pub parity: usize,
    pub delta: f64,
    pub num_units: usize,
    pub payload_units: usize,
    /// mean |projection error| / delta after rounding (STDM only).
    pub mean_abs_err_rel: f64,
    /// max |projection error| / delta after rounding (STDM only).
    pub max_abs_err_rel: f64,
    /// Luma headroom applied (gray levels).
    pub headroom: f64,
}

/// Armor encode with prototype options; see [`ArmorEncodeOpts`].
#[doc(hidden)]
pub fn armor_encode_opts(
    image_bytes: &[u8],
    message: &str,
    passphrase: &str,
    opts: &ArmorEncodeOpts,
) -> Result<(Vec<u8>, ArmorEncodeInfo), StegoError> {
    let mut info = ArmorEncodeInfo::default();
    let (bytes, _) = armor_encode_impl(image_bytes, message, passphrase, opts, &mut info)?;
    Ok((bytes, info))
}

/// The real (not MCU-padded) luma block grid: ceil(W/8) x ceil(H/8).
pub fn real_luma_blocks(img: &JpegImage) -> (usize, usize) {
    let fi = img.frame_info();
    let h0 = fi.components[0].h_sampling as usize;
    let v0 = fi.components[0].v_sampling as usize;
    let hmax = fi.max_h_sampling as usize;
    let vmax = fi.max_v_sampling as usize;
    // Component 0 dimensions per JPEG A.1.1: ceil(X * h0 / hmax).
    let cw = (fi.width as usize * h0).div_ceil(hmax);
    let ch = (fi.height as usize * v0).div_ceil(vmax);
    (cw.div_ceil(8), ch.div_ceil(8))
}

fn armor_encode_impl(
    image_bytes: &[u8],
    message: &str,
    passphrase: &str,
    opts: &ArmorEncodeOpts,
    info: &mut ArmorEncodeInfo,
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    // Initialize encode progress (STDM path assumed; adjusted if Fortress).
    progress::init(ARMOR_ENCODE_STEPS);

    // Build the payload (text + compression, no files for Armor).
    let payload_bytes = payload::encode_payload(message, &[])?;

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
    let headroom = if opts.headroom < 0.0 { auto_headroom(&img) } else { opts.headroom };
    info.headroom = headroom;
    // Fortress T for what it holds, then Fortress TL; both capacities are
    // key-free and computed here on the cover as given, exactly as
    // `armor_capacity_info` does (the HUD shows what is written).
    let tl_plan = (opts.fortress_tiled && opts.fortress_tl && !opts.force_stdm).then(|| fortress_tl::Plan::new(&img));
    let t_max = if opts.fortress_tiled { fortress_tile::t_max_frame_bytes(&img) } else { 0 };
    if !opts.force_stdm
        && let Ok(max_fort) = match (&opts.fortress_cc, opts.fortress_tiled) {
            (_, true) => Ok(t_max.max(tl_plan.as_ref().map_or(0, |p| p.max_frame_bytes()))),
            (Some(prm), false) => Ok(fortress::fortress_cc_max_frame_bytes(&img, prm)),
            (None, false) => fortress::fortress_max_frame_bytes_ext(&img, use_compact),
        } {
        let fortress_frame = if use_compact {
            let ct = crypto::encrypt_with(
                &payload_bytes,
                passphrase,
                &crypto::FORTRESS_EMPTY_SALT,
                &crypto::FORTRESS_EMPTY_NONCE,
            )?;
            frame::build_fortress_compact_frame(payload_bytes.len(), &ct)
        } else {
            let (ct, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
            frame::build_frame(payload_bytes.len(), &salt, &nonce, &ct)
        };

        if fortress_frame.len() <= max_fort {
            // Switch to Fortress step count (shorter path).
            progress::set_total(ARMOR_ENCODE_FORTRESS_STEPS);

            // Pre-settle Y/Cb/Cr to QF75 QT tables before Fortress embedding.
            // Emits one progress step per channel (Y, Cb, Cr) — dominates
            // Fortress wall-clock so subdividing it eliminates the worst
            // pre-T3 stall.
            pre_settle_for_fortress(&mut img, headroom)?;

            let fort_result = match (&opts.fortress_cc, opts.fortress_tiled) {
                (_, true) => {
                    let tl = tl_plan.as_ref().filter(|_| fortress_frame.len() > t_max);
                    info.fortress_tl = tl.is_some();
                    fortress_tile::encode(&mut img, &fortress_frame, passphrase, opts.sync_amp, tl)?
                }
                (Some(prm), false) => fortress::fortress_cc_encode(&mut img, &fortress_frame, passphrase, prm)?,
                (None, false) => fortress::fortress_encode(&mut img, &fortress_frame, passphrase)?,
            };
            progress::advance(); // Step 4: fortress BA-QIM encode
            // Fortress moved real block DCs; padding blocks follow libjpeg (R3.27).
            img.refill_padding_blocks();
            info.fortress = true;
            info.r = fort_result.repetition_factor;
            info.parity = fort_result.parity_symbols;

            // JPEG write: emits JPEG_WRITE_STEPS sub-steps via the
            // scan-encode progress callback.
            let progress_cb = || progress::advance();
            let stego_bytes = if let Ok(bytes) = img.to_bytes_with_progress(Some(&progress_cb)) { bytes } else {
                img.rebuild_huffman_tables();
                img.to_bytes_with_progress(Some(&progress_cb)).map_err(StegoError::InvalidJpeg)?
            };

            // Fortress quality: use actual r and parity from encode, pre-settled to QF75.
            let fill_ratio = fortress_frame.len() as f64 / max_fort as f64;
            // After pre-settle, compute mean_qt from the settled QT.
            let fort_qt_id = img.frame_info().components[0].quant_table_id as usize;
            let fort_mean_qt = img.quant_table(fort_qt_id)
                .map_or(10.0, |qt| embedding::compute_mean_qt(&qt.values));
            // Fortress CC at coded repetition r measured about as robust as
            // RS + repetition at 5r (r = 3 ~ the r = 15 of the old format), so
            // the score sees that equivalent and the RS-64 tier.
            let (rep_eq, parity_eq) = if opts.fortress_cc.is_some() {
                (fort_result.repetition_factor * 5, 64)
            } else {
                (fort_result.repetition_factor, fort_result.parity_symbols)
            };
            let encode_quality = quality::armor_robustness_score(&ArmorMetrics {
                repetition_factor: rep_eq,
                parity_symbols: parity_eq,
                fortress: true,
                mean_qt: fort_mean_qt,
                fill_ratio,
                delta: 12.0,
            });
            return Ok((stego_bytes, encode_quality));
        }
    }

    // Full frame for STDM path (always uses random salt + nonce).
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);

    let v2_settle = opts.settle_floor && opts.stdm == StdmVariant::V2;
    // A cover quantized finer than IJG QF65 (Apple ImageIO "0.65" is roughly
    // IJG QF90) carries the STDM change in coefficient steps a coarse channel
    // quantizer rounds to zero in flat areas; settle the luma table to at
    // least the QF65 table first (eval/runs/2026-10-31-armor-v2, sips covers).
    if v2_settle {
        settle_luma_to_floor(&mut img)?;
    }

    // Pre-clamp pass: IDCT -> clamp [0,255] -> DCT on all Y-channel blocks.
    pre_clamp_y_channel(&mut img, headroom)?;
    progress::advance(); // pre-clamp done

    // 1. Quantization table (stability map = every zigzag 1..=15 slot).
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt_values = img
        .quant_table(qt_id)
        .ok_or(StegoError::NoLuminanceChannel)?
        .values;
    progress::advance(); // stability map done

    // 2. Derive structural key with Armor salt.
    let structural_key = crypto::derive_armor_structural_key(passphrase)?;
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let spread_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    // 3. Positions: legacy enumerates the MCU-padded grid, v2 the real
    //    ceil(W/8) x ceil(H/8) grid (independent of chroma subsampling, no
    //    dummy blocks). Same order as select_and_permute(stability map).
    let variant = opts.stdm;
    let v2 = variant == StdmVariant::V2;
    let (grid_bw, grid_bh) = (img.dct_grid(0).blocks_wide(), img.dct_grid(0).blocks_tall());
    let (pos_bw, pos_bh) = if v2 { real_luma_blocks(&img) } else { (grid_bw, grid_bh) };
    let positions = selection::armor_positions(pos_bw, pos_bh, &perm_seed);
    let num_units = positions.len() / SPREAD_LEN;
    let header_units = if v2 { embedding::V2_HEADER_UNITS } else { HEADER_UNITS };
    if num_units <= header_units {
        return Err(StegoError::ImageTooSmall);
    }
    let n_used = num_units * SPREAD_LEN;
    let positions = &positions[..n_used];
    // Map a position (flat in the pos_bw-wide grid) to the image's grid.
    let to_grid = |flat: u32| -> usize {
        let f = flat as usize;
        let (blk, slot) = (f / 64, f % 64);
        ((blk / pos_bw) * grid_bw + blk % pos_bw) * 64 + slot
    };

    // 6. Mean QT, header bits and deltas.
    let mean_qt = embedding::compute_mean_qt(&qt_values);
    let (header_byte, hdr_delta, delta_base) = match variant {
        StdmVariant::Legacy => (embedding::encode_mean_qt(mean_qt), embedding::BOOTSTRAP_DELTA, mean_qt),
        StdmVariant::LegacyIdx { h } => {
            let b = embedding::encode_mean_qt(h);
            (b, embedding::BOOTSTRAP_DELTA, embedding::decode_mean_qt(b))
        }
        StdmVariant::V2 => {
            let b = embedding::encode_mean_qt(mean_qt);
            (b, opts.v2_header_delta, embedding::decode_mean_qt(b).max(opts.v2_floor))
        }
    };
    let delta_for_r = |r: usize| embedding::mult_for_r(&opts.mults, r) * delta_base;
    let bootstrap_delta = hdr_delta;
    let reference_delta = delta_for_r(1);

    // 7. Header: legacy 7 x mean-QT byte; v2 7 x magic, then 7 x mean-QT byte.
    let mut all_bits = Vec::with_capacity(num_units);
    if v2 {
        for _ in 0..HEADER_COPIES {
            for bp in (0..8).rev() {
                all_bits.push((embedding::V2_MAGIC >> bp) & 1);
            }
        }
    }
    for _ in 0..HEADER_COPIES {
        for bp in (0..8).rev() {
            all_bits.push((header_byte >> bp) & 1);
        }
    }

    // 8. Decide Phase 1 vs Phase 2 encoding.
    let payload_units = num_units - header_units;

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

    // Track metrics for quality score.
    let (armor_r, armor_parity, armor_delta);

    if let Some((chosen_parity, rs_encoded)) = phase2_result {
        let rs_bits = frame::bytes_to_bits(&rs_encoded);
        let r = repetition::compute_r(rs_bits.len(), payload_units);
        let rs_bit_count_aligned = payload_units / r;
        let mut rs_bits_padded = rs_bits;
        rs_bits_padded.resize(rs_bit_count_aligned, 0);
        let (rep_bits, _) = repetition::repetition_encode(&rs_bits_padded, payload_units);
        armor_r = r;
        armor_parity = chosen_parity;
        armor_delta = delta_for_r(r);
        all_bits.extend_from_slice(&rep_bits[..payload_units.min(rep_bits.len())]);
    } else {
        // --- Phase 1 encode: fixed RS parity=64, no repetition ---
        let rs_encoded = ecc::rs_encode_blocks(&frame_bytes);
        let rs_bits = frame::bytes_to_bits(&rs_encoded);
        if rs_bits.len() > payload_units {
            return Err(StegoError::MessageTooLarge);
        }
        armor_r = 1;
        armor_parity = 64;
        armor_delta = reference_delta;
        let mut payload_bits = rs_bits;
        payload_bits.resize(payload_units, 0);
        all_bits.extend_from_slice(&payload_bits);
    }
    let payload_delta = armor_delta;
    let embed_delta_fn = |bit_idx: usize| -> f64 {
        if bit_idx < header_units { bootstrap_delta } else { payload_delta }
    };
    info.r = armor_r;
    info.parity = armor_parity;
    info.delta = armor_delta;
    info.num_units = num_units;
    info.payload_units = payload_units;

    let embed_count = all_bits.len().min(num_units);

    // 9. Generate spreading vectors.
    let vectors = generate_spreading_vectors(&spread_seed, embed_count);
    progress::advance(); // RS encode + spreading vectors done

    // 10. STDM embed each bit into coefficient groups, with
    // STDM_EMBED_PROGRESS_STEPS progress sub-steps across the loop.
    let grid_mut = img.dct_grid_mut(0);
    let mut next_step_threshold = if embed_count == 0 {
        usize::MAX
    } else {
        embed_count / STDM_EMBED_PROGRESS_STEPS as usize
    };
    let mut steps_emitted: u32 = 0;
    let (mut err_sum, mut err_max) = (0.0f64, 0.0f64);
    for bit_idx in 0..embed_count {
        let group_start = bit_idx * SPREAD_LEN;
        let group = &positions[group_start..group_start + SPREAD_LEN];
        let delta = embed_delta_fn(bit_idx);

        if variant == StdmVariant::Legacy {
            // Shipped path, byte-identical.
            let mut coeffs = [0.0f64; SPREAD_LEN];
            for (k, pos) in group.iter().enumerate() {
                coeffs[k] = flat_get(grid_mut, to_grid(*pos)) as f64;
            }
            stdm_embed(&mut coeffs, &vectors[bit_idx], all_bits[bit_idx], delta);
            for (k, pos) in group.iter().enumerate() {
                flat_set(grid_mut, to_grid(*pos), coeffs[k].round() as i16);
            }
        } else {
            let mut idx = [0i16; SPREAD_LEN];
            let mut q = [0.0f64; SPREAD_LEN];
            for (k, pos) in group.iter().enumerate() {
                idx[k] = flat_get(grid_mut, to_grid(*pos));
                q[k] = qt_values[(*pos as usize) % 64] as f64;
            }
            let w = if v2 { q } else { [1.0f64; SPREAD_LEN] };
            let (e, _) = embedding::stdm_embed_int(
                &mut idx, &w, &q, &vectors[bit_idx], all_bits[bit_idx], delta, opts.rounding,
            );
            if bit_idx >= header_units {
                err_sum += (e / delta).abs();
                err_max = err_max.max((e / delta).abs());
            }
            for (k, pos) in group.iter().enumerate() {
                flat_set(grid_mut, to_grid(*pos), idx[k]);
            }
        }

        if bit_idx + 1 >= next_step_threshold && steps_emitted < STDM_EMBED_PROGRESS_STEPS {
            progress::advance();
            steps_emitted += 1;
            next_step_threshold =
                ((steps_emitted + 1) as usize * embed_count) / STDM_EMBED_PROGRESS_STEPS as usize;
        }
    }
    info.mean_abs_err_rel = err_sum / (embed_count.saturating_sub(header_units)).max(1) as f64;
    info.max_abs_err_rel = err_max;
    // Ensure we've emitted exactly STDM_EMBED_PROGRESS_STEPS sub-steps
    // (covers the case where embed_count < STDM_EMBED_PROGRESS_STEPS).
    while steps_emitted < STDM_EMBED_PROGRESS_STEPS {
        progress::advance();
        steps_emitted += 1;
    }

    // 11. Write modified JPEG. T3 progress recalibration — uses the
    // JPEG_WRITE_STEPS scan-encode callback to emit ~20 sub-steps
    // across the entropy-coded scan write (~100-200 ms at 12 MP →
    // ~5-10 ms per step).
    let progress_cb = || progress::advance();
    let stego_bytes = if let Ok(bytes) = img.to_bytes_with_progress(Some(&progress_cb)) { bytes } else {
        img.rebuild_huffman_tables();
        img.to_bytes_with_progress(Some(&progress_cb)).map_err(StegoError::InvalidJpeg)?
    };

    // Compute quality score for STDM path.
    let fill_ratio = frame_bytes.len() as f64 / (payload_units / 8).max(1) as f64;
    let encode_quality = quality::armor_robustness_score(&ArmorMetrics {
        repetition_factor: armor_r,
        parity_symbols: armor_parity,
        fortress: false,
        // v2 does not depend on the cover table; its step base is max(mean QT, floor).
        mean_qt: if v2 { delta_base } else { mean_qt },
        fill_ratio,
        delta: armor_delta,
    });

    Ok((stego_bytes, encode_quality))
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
    /// Estimated rotation angle in degrees (0 if no geometry correction).
    pub estimated_rotation_deg: f32,
    /// Estimated scale factor (1.0 if no geometry correction).
    pub estimated_scale: f32,
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
            mode: crate::stego::frame::MODE_GHOST,
            rs_errors_corrected: 0,
            rs_error_capacity: 0,
            integrity_percent: 100,
            repetition_factor: 0,
            parity_len: 0,
            geometry_corrected: false,
            estimated_rotation_deg: 0.0,
            estimated_scale: 1.0,
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
            mode: crate::stego::frame::MODE_ARMOR,
            rs_errors_corrected: stats.total_errors as u32,
            rs_error_capacity: stats.error_capacity as u32,
            integrity_percent: integrity,
            repetition_factor,
            parity_len,
            geometry_corrected: false,
            estimated_rotation_deg: 0.0,
            estimated_scale: 1.0,
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
        (signal_strength / reference_llr).clamp(0.0, 1.0)
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
    (combined * 100.0).round().clamp(0.0, 100.0) as u8
}

/// Compute average |LLR| from a slice of raw LLR values (the Phase 1 path
/// computes the same running sum while streaming).
#[cfg(test)]
fn compute_avg_abs_llr(llrs: &[f64]) -> f64 {
    if llrs.is_empty() {
        return 0.0;
    }
    let sum: f64 = llrs.iter().map(|llr| llr.abs()).sum();
    sum / llrs.len() as f64
}

/// Decode a text message from a stego JPEG using Armor mode.
///
/// Tries the Fortress decoders, then Armor v2, the key-free grid sync +
/// Fortress T (cropped, rotated, scaled or mirrored images), legacy STDM with
/// its delta sweep, and a rescaled decode for messenger-downscaled images.
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
    if img.num_components() > 0
        && let Ok(result) = fortress::fortress_decode(&img, passphrase) {
            return Ok(result);
        }

    // v2, the grid sync, legacy STDM, the rescaled decode (reuse the parse).
    armor_decode_no_fortress_unless(&img, passphrase, &|| false).expect("never stopped")
}

/// Decoder knobs. `Default` = the production decoder, `shipped()` = core <= 0.5.x.
#[derive(Debug, Clone)]
pub struct ArmorDecodeOpts {
    /// Try the Armor v2 STDM format (magic-gated, after Fortress).
    pub try_v2: bool,
    /// Try the legacy STDM format.
    pub try_legacy: bool,
    /// Legacy STDM + Fortress: also try the other encoder block geometries
    /// (real ceil(W/8) x ceil(H/8) and its MCU-padded variants).
    pub geometry: bool,
    /// Treat positions in MCU-padding (dummy) blocks of the received grid,
    /// or outside it, as erasures (LLR 0).
    pub erasures: bool,
    /// Fortress: requantize the received AC coefficients to the QF75 table
    /// before the Watson energy ratios (the encoder pre-settles to QF75).
    pub fortress_requant: bool,
    /// Legacy STDM Phase 2: extract the LLRs of every candidate delta (the
    /// shipped 5-entry LRU drops all but the last 5 deltas).
    pub llr_cache_fix: bool,
    /// Legacy STDM: add the recompression-rescaled delta candidate
    /// (header mean QT^2 / current mean QT).
    pub qt_ratio_delta: bool,
    /// Header agreement needed to try a non-native geometry (0..1).
    pub geometry_gate: f64,
    /// Key-free grid sync + Fortress T after the aligned decoders failed.
    pub grid_sync: bool,
    /// Fortress T on the received grid (after Fortress / Fortress CC).
    pub fortress_tiled: bool,
    /// Try a rescaled decode (Fortress, v2) when everything else failed on an
    /// image smaller than the 1600 / 1280 px Armor cover sizes.
    pub rescale: bool,
    /// Rescale attempt: snap the guessed cover short side to a multiple of
    /// this (the frontends trim covers to `ARMOR_COVER_STEP`, see
    /// `cover_dims`); 1 = plain rounding (untrimmed covers).
    pub rescale_snap: u32,
    /// Snap only when the guess lies strictly within this many *received*
    /// pixels of a multiple (else plain rounding): exact for trimmed covers
    /// in every rounding mode, today's guess for untrimmed ones at x 4/5.
    pub rescale_snap_tol: f64,
    /// v2 delta multipliers (r = 1, 2, 3-4, 5-6, >= 7) and header step.
    pub v2_mults: [f64; 5],
    pub v2_header_delta: f64,
    pub v2_floor: f64,
    /// Fortress CC parameters (prototype; the decoder reads CC images with these).
    pub fortress_cc: fortress::FortressCcParams,
    /// Legacy STDM: sweep mean-QT candidates +-30 % around the header's value
    /// only when the header's 7 copies agree at least this much (0..1; 0 =
    /// always, the shipped decoder). A wrong passphrase reads a random header
    /// byte, and a large one made the sweep cost up to 93 LLR extractions,
    /// ~1.8 s for a 1076 px image (fix list C12). Measured on legacy stego
    /// (16 covers x QF65 / 90 x 20 / 100 / 300 B x none / q85 / q75 / q60):
    /// every header that decoded through the sweep agreed >= 0.76; images
    /// with a weaker header decoded through the current-QT candidates, which
    /// are always tried (eval/runs/2026-11-02-legacy-sweep-gate).
    pub legacy_sweep_gate: f64,
}

impl Default for ArmorDecodeOpts {
    /// The production decoder: v2 first, then legacy, with every decoder-side
    /// fix except the Fortress QF75 requantization (measured: as many
    /// losses as gains).
    fn default() -> Self {
        Self {
            try_v2: true,
            try_legacy: true,
            geometry: true,
            erasures: true,
            fortress_requant: false,
            llr_cache_fix: true,
            qt_ratio_delta: true,
            geometry_gate: 0.6,
            grid_sync: true,
            fortress_tiled: true,
            rescale: true,
            rescale_snap: super::cover_dims::ARMOR_COVER_STEP,
            rescale_snap_tol: super::cover_dims::SNAP_TOLERANCE,
            v2_mults: embedding::V2_MULTS,
            v2_header_delta: embedding::V2_HEADER_DELTA,
            v2_floor: embedding::V2_MEAN_QT_FLOOR,
            fortress_cc: fortress::FortressCcParams::default(),
            legacy_sweep_gate: 0.7,
        }
    }
}

impl ArmorDecodeOpts {
    /// The decoder of core <= 0.5.x (no v2, no fixes), for measurements.
    pub fn shipped() -> Self {
        Self {
            try_v2: false,
            geometry: false,
            erasures: false,
            llr_cache_fix: false,
            qt_ratio_delta: false,
            rescale: false,
            fortress_cc: fortress::FortressCcParams { decode: false, ..Default::default() },
            legacy_sweep_gate: 0.0,
            grid_sync: false,
            fortress_tiled: false,
            ..Self::default()
        }
    }

    /// Production plus the Fortress QF75 requantization.
    pub fn all() -> Self {
        Self { fortress_requant: true, ..Self::default() }
    }
}

/// Which path decoded (measurement only).
#[derive(Debug, Clone, Default)]
pub struct ArmorDecodeInfo {
    pub path: &'static str,
    pub enc_grid: (usize, usize),
    pub native_grid: bool,
    pub geometries_tried: usize,
}

/// Armor decode with prototype options: Fortress (geometry / erasure /
/// requant fixes per `opts`), then v2 STDM, then legacy STDM, then Phase 3.
#[doc(hidden)]
pub fn armor_decode_opts(
    stego_bytes: &[u8],
    passphrase: &str,
    opts: &ArmorDecodeOpts,
) -> Result<(PayloadData, DecodeQuality, ArmorDecodeInfo), StegoError> {
    let img = JpegImage::from_bytes(stego_bytes)?;
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let mut info = ArmorDecodeInfo::default();
    if let Ok((res, q, geom, native)) = fortress::fortress_decode_opts(&img, passphrase, opts) {
        info.path = "fortress";
        info.enc_grid = geom;
        info.native_grid = native;
        return Ok((res, q, info));
    }
    let mut err = StegoError::FrameCorrupted;
    if opts.try_v2 {
        match try_v2_decode(&img, passphrase, opts) {
            Ok((res, q)) => {
                info.path = "v2";
                info.enc_grid = real_luma_blocks(&img);
                info.native_grid = true;
                return Ok((res, q, info));
            }
            Err(StegoError::DecryptionFailed) => err = StegoError::DecryptionFailed,
            Err(_) => {}
        }
    }
    // The key-free grid sync + Fortress T before the legacy STDM search: it
    // costs one luma pass + FFT when there is no sync pattern, while the
    // legacy (r, parity, delta) search can take seconds; a cropped or rotated
    // Fortress T image no longer waits for it.
    if let Ok((res, q)) = try_geometry(&img, passphrase, opts) {
        info.path = "geometric";
        return Ok((res, q, info));
    }
    if opts.try_legacy {
        match try_legacy_stdm(&img, passphrase, opts, &mut info) {
            Ok((res, q)) => {
                info.path = "legacy";
                return Ok((res, q, info));
            }
            Err(StegoError::DecryptionFailed) => err = StegoError::DecryptionFailed,
            Err(e) => {
                if !matches!(err, StegoError::DecryptionFailed) {
                    err = e;
                }
            }
        }
    }
    if let Ok((res, q)) = try_rescaled_decode(&img, passphrase, opts) {
        info.path = "rescaled";
        return Ok((res, q, info));
    }
    Err(err)
}

/// Long sides an Armor cover is prepared at (the 1600 px contract and the
/// 1280 px channel preset); a received image of another size was probably
/// rescaled by a messenger.
const RESCALE_LONG_SIDES: [usize; 2] = [1600, 1280];

/// Decode attempt for a downscaled Armor image (decoder-only): scale the
/// received luma back to the size a 1600 / 1280 px cover would have had,
/// re-block it on the QF75 table and run the Fortress and v2 decoders on it.
/// Block averages and the low-frequency Armor coefficients survive a
/// 0.67-0.8x downscale; only the 8x8 grid is lost, and this restores it.
pub(crate) fn try_rescaled_decode(
    img: &JpegImage,
    passphrase: &str,
    opts: &ArmorDecodeOpts,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let fi = img.frame_info();
    let (w, h) = (fi.width as usize, fi.height as usize);
    let long = w.max(h);
    if !opts.rescale || long < 400 {
        return Err(StegoError::FrameCorrupted);
    }
    let (luma, pw, _ph) = pixels::jpeg_to_luma_f64(img).ok_or(StegoError::NoLuminanceChannel)?;
    let q75 = compute_jpeg_qt(&JPEG_BASE_LUMINANCE_QT, 75);
    // Upscales to a cover size above the received one; a slightly larger
    // received image (a messenger upscaling, e.g. 1067 -> 1080 wide) maps
    // back down to 1600.
    let wanted = |t: usize| t * 100 > long * 101 || (t == 1600 && long * 100 > t * 101 && long * 10 <= t * 11);
    for &target in RESCALE_LONG_SIDES.iter().filter(|&&t| wanted(t)) {
        let s = target as f64 / long as f64;
        // One guess per target: the received aspect ratio, snapped to the
        // cover grid when within one received pixel (frontends trim the short
        // side to a multiple of `ARMOR_COVER_STEP`, `cover_dims`), which
        // undoes the messenger's rounding of the short side (round, floor or
        // ceil alike). Extra +-1 px candidates would each cost a full attempt
        // on a wrong passphrase.
        let short =
            super::cover_dims::snap_cover_short(w.min(h) as f64 * s, target, opts.rescale_snap, opts.rescale_snap_tol * s);
        {
            let (tw, th) = if w >= h { (target, short) } else { (short, target) };
            let (grid, real) = rescaled_luma_grid(&luma, pw, w, h, tw, th, &q75);
            // The cover's Fortress grid was MCU-padded: try the real grid and
            // its 2-block-padded variants (as `fortress_decode_opts` does).
            let mut geoms = vec![real];
            for gw in [real.0, real.0.next_multiple_of(2)] {
                for gh in [real.1, real.1.next_multiple_of(2)] {
                    if !geoms.contains(&(gw, gh)) {
                        geoms.push((gw, gh));
                    }
                }
            }
            for (gw, gh) in geoms {
                if let Ok((p, q)) = fortress::fortress_decode_grid(
                    &grid, &q75, real, passphrase, gw, gh, true, false, &opts.fortress_cc,
                ) {
                    return Ok((p, q));
                }
            }
            if let Ok(r) = try_v2_decode_grid(&grid, &q75, real, passphrase, opts) {
                return Ok(r);
            }
            progress::check_cancelled()?;
        }
    }
    Err(StegoError::FrameCorrupted)
}

/// Separable Lanczos-3 resample of the real w x h area of a luma plane (row
/// stride `stride`) to tw x th (det_math sine: identical on every platform;
/// the kernel widens on downscales), then 8x8 DCT quantized with `qt` on the
/// real block grid (edge-replicated partial blocks).
fn rescaled_luma_grid(
    luma: &[f64],
    stride: usize,
    w: usize,
    h: usize,
    tw: usize,
    th: usize,
    qt: &[u16; 64],
) -> (DctGrid, (usize, usize)) {
    // Per output coordinate: first source index and normalized weights.
    fn taps(src: usize, dst: usize) -> Vec<(usize, Vec<f64>)> {
        use crate::det_math::det_sin;
        let scale = src as f64 / dst as f64;
        let support = 3.0 * scale.max(1.0);
        let inv = 1.0 / scale.max(1.0);
        let sinc = |x: f64| -> f64 {
            if x.abs() < 1e-9 {
                1.0
            } else {
                let px = std::f64::consts::PI * x;
                det_sin(px) / px
            }
        };
        (0..dst)
            .map(|d| {
                let c = (d as f64 + 0.5) * scale - 0.5;
                let lo = (c - support).ceil().max(0.0) as usize;
                let hi = ((c + support).floor() as usize).min(src - 1);
                let mut wts: Vec<f64> = (lo..=hi)
                    .map(|i| {
                        let x = (i as f64 - c) * inv;
                        if x.abs() >= 3.0 { 0.0 } else { sinc(x) * sinc(x / 3.0) }
                    })
                    .collect();
                let sum: f64 = wts.iter().sum();
                for v in wts.iter_mut() {
                    *v /= sum;
                }
                (lo, wts)
            })
            .collect()
    }
    let tx = taps(w, tw);
    let ty = taps(h, th);
    // Horizontal pass: h x tw.
    let mut mid = vec![0.0f64; h * tw];
    for y in 0..h {
        let row = &luma[y * stride..y * stride + w];
        for (x, (lo, wts)) in tx.iter().enumerate() {
            mid[y * tw + x] = wts.iter().enumerate().map(|(k, wv)| wv * row[lo + k]).sum();
        }
    }
    // Vertical pass: th x tw.
    let mut out = vec![0.0f64; th * tw];
    for (y, (lo, wts)) in ty.iter().enumerate() {
        for x in 0..tw {
            out[y * tw + x] = wts.iter().enumerate().map(|(k, wv)| wv * mid[(lo + k) * tw + x]).sum();
        }
    }
    let (bw, bh) = (tw.div_ceil(8), th.div_ceil(8));
    let mut grid = DctGrid::new(bw, bh);
    for br in 0..bh {
        for bc in 0..bw {
            let mut px = [0.0f64; 64];
            for i in 0..8 {
                for j in 0..8 {
                    let (x, y) = ((bc * 8 + j).min(tw - 1), (br * 8 + i).min(th - 1));
                    px[i * 8 + j] = out[y * tw + x].clamp(0.0, 255.0);
                }
            }
            let q = pixels::dct_block_f64(&px, qt);
            for k in 0..64 {
                grid.set(br, bc, k / 8, k % 8, q[k]);
            }
        }
    }
    (grid, (bw, bh))
}

/// The encoder block geometries a legacy decode tries: the received
/// MCU-padded grid first, then (with `geometry`) the real grid and its
/// 2-block-padded variants.
fn legacy_geometries(img: &JpegImage, geometry: bool) -> Vec<(usize, usize)> {
    if geometry {
        encoder_geometries(img)
    } else {
        let grid = img.dct_grid(0);
        vec![(grid.blocks_wide(), grid.blocks_tall())]
    }
}

/// The luma block grids an Armor encoder can have used for this image: the
/// received MCU-padded grid first, then the real ceil(W/8) x ceil(H/8) grid
/// (a 4:4:4 cover) and the 2-block-padded one (a 4:2:0 cover). 4:2:2 / 4:4:0
/// covers (one axis padded) are not tried: every Phasm frontend writes 4:2:0
/// or 4:4:4, and each extra candidate costs a full position shuffle on a
/// wrong passphrase.
pub(crate) fn encoder_geometries(img: &JpegImage) -> Vec<(usize, usize)> {
    let grid = img.dct_grid(0);
    let (rbw, rbh) = real_luma_blocks(img);
    let mut g = vec![(grid.blocks_wide(), grid.blocks_tall())];
    for c in [(rbw, rbh), (rbw.next_multiple_of(2), rbh.next_multiple_of(2))] {
        if !g.contains(&c) {
            g.push(c);
        }
    }
    g
}

/// Header agreement of the legacy 7-copy mean-QT header: mean over the 8 bits
/// of |sum of the copies' LLRs| / sum of |LLR|. ~1 for the right geometry,
/// ~0.4 for noise; 0 when the voted byte is 0 (see below).
fn legacy_header_agreement<C: ArmorCoeffs + ?Sized>(
    src: &C,
    bw: usize,
    bh: usize,
    passphrase: &str,
) -> Result<f64, StegoError> {
    let structural_key = crypto::derive_armor_structural_key(passphrase)?;
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let spread_seed: [u8; 32] = structural_key[32..].try_into().unwrap();
    let positions = selection::armor_positions(bw, bh, &perm_seed);
    if positions.len() / SPREAD_LEN <= HEADER_UNITS {
        return Ok(0.0);
    }
    let vecs = generate_spreading_vectors(&spread_seed, HEADER_UNITS);
    let llrs: Vec<f64> = (0..HEADER_UNITS)
        .map(|i| unit_llr(src, &positions, i, &vecs[i], embedding::BOOTSTRAP_DELTA))
        .collect();
    let mut acc = 0.0;
    let mut byte = 0u8;
    for b in 0..8 {
        let (mut sum, mut abs) = (0.0, 0.0);
        for c in 0..HEADER_COPIES {
            sum += llrs[c * 8 + b];
            abs += llrs[c * 8 + b].abs();
        }
        acc += if abs > 0.0 { sum.abs() / abs } else { 0.0 };
        if sum < 0.0 {
            byte |= 1 << (7 - b);
        }
    }
    // Small projections (no legacy header: Fortress, v2, a cover) read as an
    // all-zero byte with full agreement at the large bootstrap step; a legacy
    // header carries a mean-QT byte >= 1, so a zero byte is no evidence.
    if byte == 0 {
        return Ok(0.0);
    }
    Ok(acc / 8.0)
}

fn try_legacy_stdm(
    img: &JpegImage,
    passphrase: &str,
    opts: &ArmorDecodeOpts,
    info: &mut ArmorDecodeInfo,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    let grid = img.dct_grid(0);
    let (rbw, rbh) = real_luma_blocks(img);
    let mut err = StegoError::FrameCorrupted;
    for (i, &(bw, bh)) in legacy_geometries(img, opts.geometry).iter().enumerate() {
        let src = MappedGrid { grid, enc_bw: bw, real_bw: rbw, real_bh: rbh, erase_dummy: opts.erasures };
        if i > 0 && legacy_header_agreement(&src, bw, bh, passphrase)? < opts.geometry_gate {
            continue;
        }
        info.geometries_tried += 1;
        match try_armor_decode_coeffs(&src, bw, bh, qt, passphrase, opts) {
            Ok(r) => {
                info.enc_grid = (bw, bh);
                info.native_grid = i == 0;
                return Ok(r);
            }
            Err(StegoError::DecryptionFailed) => return Err(StegoError::DecryptionFailed),
            Err(e) => err = e,
        }
    }
    Err(err)
}

/// Armor v2 coefficient source: the received luma grid on its real block
/// grid, dequantized with the received table (index x q).
struct V2Src<'a> {
    grid: &'a DctGrid,
    real_bw: usize,
    q: [f64; 64],
}

impl ArmorCoeffs for V2Src<'_> {
    #[inline]
    fn coeff(&self, flat_idx: usize) -> i16 {
        let blk = flat_idx / 64;
        let slot = flat_idx % 64;
        self.grid.get(blk / self.real_bw, blk % self.real_bw, slot / 8, slot % 8)
    }
    #[inline]
    fn value(&self, flat_idx: usize) -> f64 {
        self.coeff(flat_idx) as f64 * self.q[flat_idx % 64]
    }
}

fn try_v2_decode(
    img: &JpegImage,
    passphrase: &str,
    opts: &ArmorDecodeOpts,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    try_v2_decode_grid(img.dct_grid(0), &qt.values, real_luma_blocks(img), passphrase, opts)
}

/// [`try_v2_decode`] on a bare luma grid and table.
fn try_v2_decode_grid(
    grid: &DctGrid,
    qt_values: &[u16; 64],
    (rbw, rbh): (usize, usize),
    passphrase: &str,
    opts: &ArmorDecodeOpts,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let src = V2Src { grid, real_bw: rbw, q: std::array::from_fn(|k| qt_values[k] as f64) };

    let structural_key = crypto::derive_armor_structural_key(passphrase)?;
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let spread_seed: [u8; 32] = structural_key[32..].try_into().unwrap();
    let mut positions = selection::armor_positions(rbw, rbh, &perm_seed);
    let num_units = positions.len() / SPREAD_LEN;
    let header_units = embedding::V2_HEADER_UNITS;
    if num_units <= header_units {
        return Err(StegoError::ImageTooSmall);
    }
    positions.truncate(num_units * SPREAD_LEN);
    let positions = &positions[..];
    // Magic gate: a non-v2 image (or a wrong passphrase) stops here, before
    // the spreading vectors of the payload are generated.
    let head = SpreadVectors::new(&spread_seed, header_units);
    let magic = extract_header_byte(&src, positions, &head, opts.v2_header_delta, 0);
    if magic != embedding::V2_MAGIC {
        return Err(StegoError::FrameCorrupted);
    }
    let mean_byte = extract_header_byte(&src, positions, &head, opts.v2_header_delta, HEADER_UNITS);
    drop(head);
    let vectors = SpreadVectors::new(&spread_seed, num_units);
    let mean_qt = embedding::decode_mean_qt(mean_byte).max(opts.v2_floor);
    let payload_units = num_units - header_units;

    // Phase 1 (r = 1).
    let d1 = embedding::mult_for_r(&opts.v2_mults, 1) * mean_qt;
    if let Ok(r) = decode_phase1_with_offset(&src, positions, &vectors, d1, num_units, header_units, passphrase) {
        return Ok(r);
    }
    // Phase 2: one LLR vector per distinct delta.
    let mut cands: Vec<(usize, usize, f64)> = Vec::new();
    for &parity in &ecc::PARITY_TIERS {
        for r in compute_candidate_rs(payload_units, parity) {
            cands.push((parity, r, embedding::mult_for_r(&opts.v2_mults, r) * mean_qt));
        }
    }
    let mut deltas: Vec<f64> = Vec::new();
    for &(_, _, d) in &cands {
        if !deltas.iter().any(|&x| (x - d).abs() < 0.001) {
            deltas.push(d);
        }
    }
    for &d in &deltas {
        let mut cache: Vec<(f64, Vec<f64>)> = Vec::new();
        get_or_extract_llrs(&mut cache, d, &src, positions, &vectors, num_units, header_units);
        let group: Vec<(usize, usize, f64)> =
            cands.iter().copied().filter(|&(_, _, x)| (x - d).abs() < 0.001).collect();
        if let Some(res) = p2_search(&group, &cache[0].1, payload_units, passphrase) {
            return res;
        }
    }
    Err(StegoError::FrameCorrupted)
}

/// Payload-unit LLRs of an Armor stego at a given delta (measurement only):
/// v2 on the real grid; legacy through `enc_grid` (default: the received
/// MCU-padded grid) with dummy/outside blocks erased (LLR 0).
#[doc(hidden)]
pub fn armor_debug_payload_llrs(
    stego_bytes: &[u8],
    passphrase: &str,
    v2: bool,
    delta: f64,
    enc_grid: Option<(usize, usize)>,
) -> Result<Vec<f64>, StegoError> {
    let img = JpegImage::from_bytes(stego_bytes)?;
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    let grid = img.dct_grid(0);
    let (rbw, rbh) = real_luma_blocks(&img);
    let structural_key = crypto::derive_armor_structural_key(passphrase)?;
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let spread_seed: [u8; 32] = structural_key[32..].try_into().unwrap();
    let (bw, bh) = if v2 { (rbw, rbh) } else { enc_grid.unwrap_or((grid.blocks_wide(), grid.blocks_tall())) };
    let mut positions = selection::armor_positions(bw, bh, &perm_seed);
    let num_units = positions.len() / SPREAD_LEN;
    positions.truncate(num_units * SPREAD_LEN);
    let header_units = if v2 { embedding::V2_HEADER_UNITS } else { HEADER_UNITS };
    let vectors = SpreadVectors::new(&spread_seed, num_units);
    let mut cache: Vec<(f64, Vec<f64>)> = Vec::new();
    if v2 {
        let src = V2Src { grid, real_bw: rbw, q: std::array::from_fn(|k| qt.values[k] as f64) };
        get_or_extract_llrs(&mut cache, delta, &src, &positions, &vectors, num_units, header_units);
    } else {
        let src = MappedGrid { grid, enc_bw: bw, real_bw: rbw, real_bh: rbh, erase_dummy: true };
        get_or_extract_llrs(&mut cache, delta, &src, &positions, &vectors, num_units, header_units);
    }
    Ok(cache.pop().map(|c| c.1).unwrap_or_default())
}

/// The Armor STDM decode over any source of Y coefficients: the parsed grid,
/// or the stable coefficients of a geometry-corrected image.
///
/// Working set: the embedding order as `u32` (15 per block), the spreading
/// vectors only while small (`SpreadVectors`), and the LLR cache. No cost map,
/// no `CoeffPos`, no per-candidate LLR vectors in Phase 1.
fn try_armor_decode_coeffs<C: ArmorCoeffs + ?Sized>(
    src: &C,
    blocks_wide: usize,
    blocks_tall: usize,
    qt: &QuantTable,
    passphrase: &str,
    fx: &ArmorDecodeOpts,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    // 1. Derive structural key.
    let structural_key = crypto::derive_armor_structural_key(passphrase)?;
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let spread_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    // 2. Stable positions (zigzag 1..=MAX_ARMOR_ZIGZAG of every block),
    //    permuted — the order `select_and_permute(compute_stability_map)` gives.
    let mut positions = selection::armor_positions(blocks_wide, blocks_tall, &perm_seed);
    let num_units = positions.len() / SPREAD_LEN;
    if num_units == 0 {
        return Err(StegoError::ImageTooSmall);
    }
    positions.truncate(num_units * SPREAD_LEN);
    let positions = &positions[..];

    // 3. Spreading vectors for all units.
    let vectors = SpreadVectors::new(&spread_seed, num_units);
    let grid = src; // the coefficient source the extraction helpers read

    // 5. Extract 1-byte header at bootstrap delta.
    if num_units <= HEADER_UNITS {
        return Err(StegoError::ImageTooSmall);
    }
    let (header_byte, header_agreement) =
        header_byte_agreement(grid, positions, &vectors, embedding::BOOTSTRAP_DELTA, 0);
    let header_mean_qt = embedding::decode_mean_qt(header_byte);

    // 6. Compute current image's mean QT for comparison.
    let current_mean_qt = embedding::compute_mean_qt(&qt.values);

    let payload_units = num_units - HEADER_UNITS;

    // 7. Build candidate mean_qt values for delta sweep.
    // Wider sweep +/-30% in 3% steps (~21 candidates) from both header and current.
    let mut raw_candidates = Vec::with_capacity(24);
    raw_candidates.push(header_mean_qt);
    raw_candidates.push(current_mean_qt);
    // A header whose copies disagree is noise (a wrong passphrase, or a header
    // a channel destroyed): no sweep around it (`legacy_sweep_gate`).
    if header_agreement >= fx.legacy_sweep_gate {
        for step in 1..=10 {
            let factor = step as f64 * 0.03;
            raw_candidates.push(header_mean_qt * (1.0 - factor));
            raw_candidates.push(header_mean_qt * (1.0 + factor));
        }
    }

    // Deduplicate (within 0.1 tolerance)
    let mut candidates: Vec<f64> = Vec::with_capacity(raw_candidates.len());
    for &c in &raw_candidates {
        if c > 0.1 && !candidates.iter().any(|&existing| (existing - c).abs() < 0.1) {
            candidates.push(c);
        }
    }
    if fx.qt_ratio_delta && current_mean_qt > 0.1 {
        // Recompression rescales the indices by ~ q_stego / q_channel, i.e. the
        // embedded lattice appears at mult x header^2 / current. Appended after
        // the shipped sweep, which stays exactly as it was (some decodes only
        // succeed at one particular sweep candidate), deduplicated only among
        // themselves.
        let scaled = header_mean_qt * header_mean_qt / current_mean_qt;
        let base = candidates.len();
        let mut add = vec![scaled];
        for step in 1..=4 {
            let factor = step as f64 * 0.03;
            add.push(scaled * (1.0 - factor));
            add.push(scaled * (1.0 + factor));
        }
        for c in add {
            if c > 0.1
                && !candidates.iter().any(|&e| (e - c).abs() < 0.001)
                && !candidates[base..].iter().any(|&e| (e - c).abs() < 0.1)
            {
                candidates.push(c);
            }
        }
    }

    // Set progress total now that we know the candidate count.
    // fortress(1) + phase1(nc) + phase2(nc) + phase3(1) per run.
    // Doubled fortress+phase1+phase2 for potential second run via geometric recovery.
    // Ghost decode resets progress separately (its own GHOST_DECODE_STEPS total).
    let nc = candidates.len() as u32;
    // Only set total on first call; geometric recovery calls us again.
    if progress::get().1 == 0 {
        let total = (2 * (1 + nc + nc) + 1).max(50);
        // Use set_total (not init) to avoid resetting STEP — in parallel mode,
        // other threads may have already advanced the counter.
        progress::set_total(total);
    }
    progress::advance(); // fortress check already done by caller

    // 8. Pass 1: Try Phase 1 for ALL candidates first (fast per candidate).
    #[cfg(feature = "parallel")]
    {
        let result = candidates.par_iter().find_map_first(|&mean_qt| {
            if progress::is_cancelled() { return Some(Err(StegoError::Cancelled)); }
            let reference_delta = embedding::compute_delta_from_mean_qt(mean_qt, 1);
            match decode_phase1_with_offset(
                grid, positions, &vectors, reference_delta, num_units, HEADER_UNITS,
                passphrase,
            ) {
                Ok(result) => Some(Ok(result)),
                Err(StegoError::DecryptionFailed) => Some(Err(StegoError::DecryptionFailed)),
                Err(_) => { progress::advance(); None }
            }
        });
        match result {
            Some(Ok(payload)) => return Ok(payload),
            Some(Err(e)) => return Err(e),
            None => {} // fall through to Phase 2
        }
    }
    #[cfg(not(feature = "parallel"))]
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
    // Pre-build all (parity, r, delta) candidates across ALL mean_qt values,
    // pre-extract LLRs for all unique deltas, then search in one parallel sweep.
    let mut all_p2_candidates: Vec<(usize, usize, f64)> = Vec::new();
    for &mean_qt in &candidates {
        for &parity in &ecc::PARITY_TIERS {
            let candidate_rs = compute_candidate_rs(payload_units, parity);
            for r in candidate_rs {
                let delta = embedding::compute_delta_from_mean_qt(mean_qt, r);
                all_p2_candidates.push((parity, r, delta));
            }
        }
    }

    if fx.llr_cache_fix {
        // One LLR vector per unique delta, in candidate order; every
        // (parity, r) of that delta is tried before the next is extracted.
        let mut deltas: Vec<f64> = Vec::new();
        for &(_, _, d) in &all_p2_candidates {
            if !deltas.iter().any(|&x| (x - d).abs() < 0.001) {
                deltas.push(d);
            }
        }
        for &d in &deltas {
            progress::check_cancelled()?;
            let mut cache: Vec<(f64, Vec<f64>)> = Vec::new();
            get_or_extract_llrs(&mut cache, d, grid, positions, &vectors, num_units, HEADER_UNITS);
            let llrs = &cache[0].1;
            let group: Vec<(usize, usize, f64)> = all_p2_candidates
                .iter()
                .copied()
                .filter(|&(_, _, x)| (x - d).abs() < 0.001)
                .collect();
            if let Some(res) = p2_search(&group, llrs, payload_units, passphrase) {
                for _ in 0..candidates.len() { progress::advance(); }
                return res;
            }
        }
        for _ in 0..candidates.len() { progress::advance(); }
        return Err(StegoError::FrameCorrupted);
    }

    // Pre-extract LLRs for all unique deltas (sequential; extraction itself is parallel internally).
    let mut cached_llrs: Vec<(f64, Vec<f64>)> = Vec::new();
    for &(_, _, delta) in &all_p2_candidates {
        get_or_extract_llrs(
            &mut cached_llrs, delta,
            grid, positions, &vectors, num_units, HEADER_UNITS,
        );
    }
    progress::check_cancelled()?;

    // Read-only snapshot for parallel access.
    let llr_cache: &[(f64, Vec<f64>)] = &cached_llrs;

    let find_llrs = |delta: f64| -> &[f64] {
        for (cached_delta, llrs) in llr_cache.iter() {
            if (cached_delta - delta).abs() < 0.001 {
                return llrs;
            }
        }
        &[]
    };

    let try_p2_candidate = |&(parity, r, adaptive_delta): &(usize, usize, f64)| -> Option<Result<(PayloadData, DecodeQuality), StegoError>> {
        if progress::is_cancelled() { return Some(Err(StegoError::Cancelled)); }
        let raw_llrs = find_llrs(adaptive_delta);

        let rs_bit_count = payload_units / r;
        if rs_bit_count == 0 { return None; }
        let used_llrs = rs_bit_count * r;
        if used_llrs > raw_llrs.len() { return None; }

        let (voted_bits, rep_quality) = repetition::repetition_decode_soft_with_quality(
            &raw_llrs[..used_llrs], rs_bit_count,
        );
        let voted_bytes = frame::bits_to_bytes(&voted_bits);

        let (decoded_frame, rs_stats) = try_rs_decode_frame_with_parity(&voted_bytes, parity)?;
        let parsed = frame::parse_frame(&decoded_frame).ok()?;
        match crypto::decrypt(&parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce) {
            Ok(plaintext) => {
                let len = parsed.plaintext_len as usize;
                if len > plaintext.len() { return None; }
                let payload_data = payload::decode_payload(&plaintext[..len]).ok()?;
                let reference_llr = adaptive_delta / 2.0;
                let quality = DecodeQuality::from_rs_stats_with_signal(
                    &rs_stats, r as u8, parity as u16,
                    rep_quality.avg_abs_llr_per_copy, reference_llr,
                );
                Some(Ok((payload_data, quality)))
            }
            Err(StegoError::DecryptionFailed) => Some(Err(StegoError::DecryptionFailed)),
            Err(_) => None,
        }
    };

    #[cfg(feature = "parallel")]
    let p2_result = all_p2_candidates.par_iter().find_map_first(try_p2_candidate);
    #[cfg(not(feature = "parallel"))]
    let p2_result = all_p2_candidates.iter().find_map(try_p2_candidate);

    // Advance progress for Phase 2 (bulk advance since search is now flat).
    for _ in 0..candidates.len() { progress::advance(); }

    match p2_result {
        Some(Ok(payload)) => return Ok(payload),
        Some(Err(e)) => return Err(e),
        None => {}
    }

    Err(StegoError::FrameCorrupted)
}

/// Phase 2 search over (parity, r, delta) candidates that share one LLR vector.
fn p2_search(
    group: &[(usize, usize, f64)],
    raw_llrs: &[f64],
    payload_units: usize,
    passphrase: &str,
) -> Option<Result<(PayloadData, DecodeQuality), StegoError>> {
    let try_one = |&(parity, r, adaptive_delta): &(usize, usize, f64)| -> Option<Result<(PayloadData, DecodeQuality), StegoError>> {
        if progress::is_cancelled() { return Some(Err(StegoError::Cancelled)); }
        let rs_bit_count = payload_units / r;
        if rs_bit_count == 0 { return None; }
        let used_llrs = rs_bit_count * r;
        if used_llrs > raw_llrs.len() { return None; }
        let (voted_bits, rep_quality) = repetition::repetition_decode_soft_with_quality(
            &raw_llrs[..used_llrs], rs_bit_count,
        );
        let voted_bytes = frame::bits_to_bytes(&voted_bits);
        let (decoded_frame, rs_stats) = try_rs_decode_frame_with_parity(&voted_bytes, parity)?;
        let parsed = frame::parse_frame(&decoded_frame).ok()?;
        match crypto::decrypt(&parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce) {
            Ok(plaintext) => {
                let len = parsed.plaintext_len as usize;
                if len > plaintext.len() { return None; }
                let payload_data = payload::decode_payload(&plaintext[..len]).ok()?;
                let quality = DecodeQuality::from_rs_stats_with_signal(
                    &rs_stats, r as u8, parity as u16,
                    rep_quality.avg_abs_llr_per_copy, adaptive_delta / 2.0,
                );
                Some(Ok((payload_data, quality)))
            }
            Err(StegoError::DecryptionFailed) => Some(Err(StegoError::DecryptionFailed)),
            Err(_) => None,
        }
    };
    #[cfg(feature = "parallel")]
    { group.par_iter().find_map_first(try_one) }
    #[cfg(not(feature = "parallel"))]
    { group.iter().find_map(try_one) }
}

/// Armor decode without Fortress: STDM delta sweep + Phase 3 geometric recovery.
///
/// Used by the parallel smart_decode path to run STDM and Phase 3 concurrently
/// with Fortress (which runs on a separate thread).
#[allow(dead_code)] // Reserved for parallel smart_decode path (deferred wiring).
pub(crate) fn armor_decode_no_fortress(img: &JpegImage, passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    armor_decode_no_fortress_unless(img, passphrase, &|| false).expect("never stopped")
}

/// [`armor_decode_no_fortress`], skipped (`None`) when `stop()` is true
/// between its attempts — for `smart_decode`, which stops it once an attempt
/// it prefers succeeded.
///
/// Order (as [`armor_decode_opts`]): Armor v2 (magic-gated), the key-free
/// grid sync + Fortress T, legacy STDM, the rescaled decode. The grid sync
/// costs one luma pass + FFT when there is no sync pattern, while the legacy
/// (r, parity, delta) search can take seconds; a cropped or rotated Fortress
/// T image no longer waits for it.
pub(crate) fn armor_decode_no_fortress_unless(
    img: &JpegImage,
    passphrase: &str,
    stop: &dyn Fn() -> bool,
) -> Option<Result<(PayloadData, DecodeQuality), StegoError>> {
    if stop() {
        return None;
    }
    if img.num_components() == 0 {
        return Some(Err(StegoError::NoLuminanceChannel));
    }
    let opts = ArmorDecodeOpts::default();
    // A v2 frame that fails to decrypt is the answer for STDM: no legacy search.
    let mut err = match try_v2_decode(img, passphrase, &opts) {
        Ok(result) => return Some(Ok(result)),
        Err(e) => e,
    };
    if stop() {
        return None;
    }
    if let Err(e) = progress::check_cancelled() {
        return Some(Err(e));
    }
    match try_geometry(img, passphrase, &opts) {
        Ok(result) => return Some(Ok(result)),
        Err(StegoError::Cancelled) => return Some(Err(StegoError::Cancelled)),
        Err(_) => {}
    }
    if !matches!(err, StegoError::DecryptionFailed) {
        if stop() {
            return None;
        }
        match try_legacy_stdm(img, passphrase, &opts, &mut ArmorDecodeInfo::default()) {
            Ok(result) => return Some(Ok(result)),
            Err(StegoError::Cancelled) => return Some(Err(StegoError::Cancelled)),
            Err(e) => err = e,
        }
    }
    if stop() {
        return None;
    }
    Some(try_rescaled_decode(img, passphrase, &opts).map_err(|_| err))
}

/// Geometric recovery after the grid-aligned decoders failed: the key-free
/// grid sync + Fortress T (`opts.grid_sync`). The DFT template and ring of
/// core <= 0.5.x are gone: the template never reached the file and the ring
/// carried nothing (eval/runs/2026-11-02-armor-geometry/RESULTS.md, R1-R8).
pub(crate) fn try_geometry(
    img: &JpegImage,
    passphrase: &str,
    opts: &ArmorDecodeOpts,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let mut err = StegoError::FrameCorrupted;
    if opts.grid_sync && img.num_components() > 0 {
        match fortress_tile::decode_geometric(img, passphrase) {
            Ok(r) => return Ok(r),
            Err(StegoError::Cancelled) => return Err(StegoError::Cancelled),
            Err(e) => err = e,
        }
    }
    Err(err)
}

/// Extract the 1-byte mean-QT header from embedding units using soft majority voting.
///
/// Reads 7 copies x 1 byte x 8 bits = 56 units at the given delta and offset.
fn extract_header_byte<C: ArmorCoeffs + ?Sized>(
    grid: &C,
    positions: &[u32],
    vectors: &SpreadVectors,
    delta: f64,
    offset: usize,
) -> u8 {
    header_byte_agreement(grid, positions, vectors, delta, offset).0
}

/// The 1-byte header (majority vote over its 7 copies) and how much the
/// copies agree: per bit |sum of LLRs| / sum of |LLRs|, averaged (0..1; as
/// `legacy_header_agreement`, 0 for an all-zero byte, which is no evidence).
fn header_byte_agreement<C: ArmorCoeffs + ?Sized>(
    grid: &C,
    positions: &[u32],
    vectors: &SpreadVectors,
    delta: f64,
    offset: usize,
) -> (u8, f64) {
    let mut header_llrs = [0.0f64; 56]; // 7 copies x 8 bits
    let vecs = vectors.range(offset, 56);
    for i in 0..56 {
        header_llrs[i] = unit_llr(grid, positions, offset + i, &vecs[i], delta);
    }

    // Majority vote across 7 copies for each of 8 bits
    let mut byte = 0u8;
    let mut acc = 0.0;
    for bit_pos in 0..8 {
        let (mut total, mut abs) = (0.0, 0.0);
        for copy in 0..7 {
            total += header_llrs[copy * 8 + bit_pos];
            abs += header_llrs[copy * 8 + bit_pos].abs();
        }
        acc += if abs > 0.0 { total.abs() / abs } else { 0.0 };
        if total < 0.0 {
            byte |= 1 << (7 - bit_pos);
        }
    }
    (byte, if byte == 0 { 0.0 } else { acc / 8.0 })
}

/// Phase 1 decode: extract all bits with given delta, then RS decode.
fn decode_phase1_with_offset<C: ArmorCoeffs + ?Sized>(
    grid: &C,
    positions: &[u32],
    vectors: &SpreadVectors,
    delta: f64,
    num_units: usize,
    payload_offset: usize,
    passphrase: &str,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let payload_units = num_units - payload_offset;

    // Extract the payload LLRs chunk by chunk, keeping only the hard bits and
    // the running |LLR| sum (the same in-order sum `compute_avg_abs_llr`
    // takes over the whole vector).
    let mut abs_sum = 0.0f64;
    let mut extracted_bits: Vec<u8> = Vec::with_capacity(payload_units);
    let mut start = payload_offset;
    while start < num_units {
        let n = UNIT_CHUNK.min(num_units - start);
        let vecs = vectors.range(start, n);
        for k in 0..n {
            let llr = unit_llr(grid, positions, start + k, &vecs[k], delta);
            abs_sum += llr.abs();
            extracted_bits.push(if llr >= 0.0 { 0 } else { 1 });
        }
        start += n;
    }

    // Signal strength from raw LLRs (before hard decision)
    let signal_strength = if payload_units == 0 { 0.0 } else { abs_sum / payload_units as f64 };
    // Reference LLR for pristine STDM embedding: delta / 2
    let reference_llr = delta / 2.0;

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

/// Ensure cached LLRs exist for a delta value, extracting from the grid if needed.
///
/// P2b: Uses LRU eviction when the cache exceeds `LLR_CACHE_MAX` entries.
/// The caller reads from the cache snapshot later; no return value needed.
fn get_or_extract_llrs<C: ArmorCoeffs + ?Sized>(
    cache: &mut Vec<(f64, Vec<f64>)>,
    delta: f64,
    grid: &C,
    positions: &[u32],
    vectors: &SpreadVectors,
    num_units: usize,
    payload_offset: usize,
) {
    // Check cache (use approximate comparison for f64)
    for i in 0..cache.len() {
        if (cache[i].0 - delta).abs() < 0.001 {
            // P2b: Move to end (most recently used) for LRU ordering
            if i < cache.len() - 1 {
                let entry = cache.remove(i);
                cache.push(entry);
            }
            return;
        }
    }

    // Extract fresh LLRs, chunk by chunk (each unit's LLR is independent).
    let mut llrs = vec![0.0f64; num_units - payload_offset];
    let fill = |(ci, out): (usize, &mut [f64])| {
        let start = payload_offset + ci * UNIT_CHUNK;
        let vecs = vectors.range(start, out.len());
        for (k, llr) in out.iter_mut().enumerate() {
            *llr = unit_llr(grid, positions, start + k, &vecs[k], delta);
        }
    };
    #[cfg(feature = "parallel")]
    llrs.par_chunks_mut(UNIT_CHUNK).enumerate().for_each(fill);
    #[cfg(not(feature = "parallel"))]
    llrs.chunks_mut(UNIT_CHUNK).enumerate().for_each(fill);

    // P2b: Evict oldest entry if cache is full
    if cache.len() >= LLR_CACHE_MAX {
        cache.remove(0); // Remove LRU (oldest) entry
    }

    cache.push((delta, llrs));
}

/// Pre-clamp the Y channel: IDCT -> clamp [0, 255] -> DCT for all blocks.
///
/// This "settles" the cover image's coefficients so they produce valid pixel
/// values. Without this, recompression through a pixel-domain pipeline
/// (IDCT -> clamp -> DCT) introduces systematic distortion from clamping.
/// Requantize luma (and any component sharing its table) to the elementwise
/// max of its table and the IJG QF65 luminance table, when that changes it.
fn settle_luma_to_floor(img: &mut JpegImage) -> Result<(), StegoError> {
    use crate::codec::jpeg::dct::QuantTable;
    let y_id = img.frame_info().components[0].quant_table_id as usize;
    let old = img.quant_table(y_id).ok_or(StegoError::NoLuminanceChannel)?.values;
    let floor = compute_jpeg_qt(&JPEG_BASE_LUMINANCE_QT, 65);
    let new: [u16; 64] = std::array::from_fn(|k| old[k].max(floor[k]));
    if new == old {
        return Ok(());
    }
    for c in 0..img.num_components() {
        if img.frame_info().components[c].quant_table_id as usize != y_id {
            continue;
        }
        let grid = img.dct_grid_mut(c);
        let process = |chunk: &mut [i16]| {
            let q: [i16; 64] = chunk.try_into().unwrap();
            let mut px = idct_block_active(&q, &old);
            for p in px.iter_mut() {
                *p = p.clamp(0.0, 255.0);
            }
            chunk.copy_from_slice(&dct_block_active(&px, &new));
        };
        #[cfg(feature = "parallel")]
        grid.coeffs_mut().par_chunks_mut(64).for_each(process);
        #[cfg(not(feature = "parallel"))]
        grid.coeffs_mut().chunks_mut(64).for_each(process);
    }
    img.set_quant_table(y_id, QuantTable::new(new));
    Ok(())
}

/// Soft-knee luma map into [h, 255 - h]: identity on [k, 255 - k] with
/// k = KNEE_FACTOR * h, linear compression of the outer k levels into k - h,
/// then clamp. h = 0 is a plain clamp to [0, 255].
#[inline]
pub(crate) fn headroom_map(y: f64, h: f64) -> f64 {
    if h <= 0.0 {
        return y.clamp(0.0, 255.0);
    }
    let k = KNEE_FACTOR * h;
    let v = if y > 255.0 - k {
        (255.0 - k) + (y - (255.0 - k)) * (k - h) / k
    } else if y < k {
        h + y * (k - h) / k
    } else {
        y
    };
    v.clamp(0.0, 255.0)
}

/// Headroom the encoder applies by itself: [`AUTO_HEADROOM`] gray levels
/// on covers whose saturated-block fraction (blocks with at least half of the
/// pixels within 6 levels of 0 or 255: a Fortress or STDM change there would
/// be clipped by the receiver) reaches 20 %, ramping linearly from 5 %. On the
/// other covers (185 of 210 in the corpus below 5 %) nothing changes.
pub(crate) fn auto_headroom(img: &JpegImage) -> f64 {
    let s = saturated_block_fraction(img);
    AUTO_HEADROOM * ((s - 0.05) / 0.15).clamp(0.0, 1.0)
}

/// Full automatic headroom, gray levels (measured: 3 recovers the high- and
/// low-key failures; see eval/runs/2026-10-31-armor-v2).
pub const AUTO_HEADROOM: f64 = 3.0;

/// Fraction of luma blocks with at least half of their pixels saturated.
pub(crate) fn saturated_block_fraction(img: &JpegImage) -> f64 {
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let Some(qt) = img.quant_table(qt_id) else { return 0.0 };
    let qt_values = qt.values;
    let grid = img.dct_grid(0);
    let n = grid.blocks_wide() * grid.blocks_tall();
    if n == 0 {
        return 0.0;
    }
    let sat = grid
        .coeffs()
        .chunks(64)
        .filter(|chunk| {
            let q: [i16; 64] = (*chunk).try_into().unwrap();
            let px = idct_block_active(&q, &qt_values);
            px.iter().filter(|&&p| !(6.5..248.5).contains(&p)).count() >= 32
        })
        .count();
    sat as f64 / n as f64
}

/// Width of the soft knee in multiples of the headroom.
const KNEE_FACTOR: f64 = 5.0;

fn pre_clamp_y_channel(img: &mut JpegImage, headroom: f64) -> Result<(), StegoError> {
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt_values = img.quant_table(qt_id)
        .ok_or(StegoError::NoLuminanceChannel)?.values;
    let grid = img.dct_grid_mut(0);

    let process_block = |chunk: &mut [i16]| {
        let quantized: [i16; 64] = chunk.try_into().unwrap();
        let mut px = idct_block_active(&quantized, &qt_values);
        for p in px.iter_mut() {
            *p = headroom_map(*p, headroom);
        }
        let settled = dct_block_active(&px, &qt_values);
        chunk.copy_from_slice(&settled);
    };

    #[cfg(feature = "parallel")]
    grid.coeffs_mut().par_chunks_mut(64).for_each(process_block);
    #[cfg(not(feature = "parallel"))]
    grid.coeffs_mut().chunks_mut(64).for_each(process_block);
    Ok(())
}

/// Verify the trailing CRC32 on a candidate decoded frame.
/// `data` is the RS-decoded bytes; `total_len` is the expected total
/// frame size (header + salt + nonce + ciphertext + 4-byte CRC).
/// Returns true iff `crc32fast::hash(data[..total_len-4])` matches
/// the u32 BE at `data[total_len-4..total_len]`.
///
/// Used as an O(~1µs) gate inside the RS-candidate brute-force loops
/// (`try_rs_decode_frame_with_parity` + `try_rs_decode_compact_
/// frame_with_parity`). Lets the loop keep iterating past a plausibility-
/// pass-CRC-fail data_len candidate (which would otherwise cause the
/// inner function to return a false-positive frame that the outer
/// caller would just CRC-reject via `frame::parse_frame`, abandoning
/// the entire (parity, r, delta) sweep). Mirrors the
/// `peek_shadow_fdl` gate pattern.
fn frame_crc_ok(data: &[u8], total_len: usize) -> bool {
    if total_len < 4 || data.len() < total_len {
        return false;
    }
    let payload = &data[..total_len - 4];
    let stored_crc = u32::from_be_bytes([
        data[total_len - 4],
        data[total_len - 3],
        data[total_len - 2],
        data[total_len - 1],
    ]);
    let computed_crc = crc32fast::hash(payload);
    stored_crc == computed_crc
}

/// Try to RS-decode a frame from extracted bytes using a specific parity length.
///
/// Optimized sweep: after the first RS block decodes successfully, the
/// `plaintext_len` field (first 2 bytes) determines the exact total frame
/// length. Plausibility checks on `plaintext_len` skip obviously wrong
/// values before attempting expensive multi-block RS decode.
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
            && first_block_data.len() >= 2 {
                let pt_len =
                    u16::from_be_bytes([first_block_data[0], first_block_data[1]]) as usize;

                // Plausibility: plaintext_len must be positive.
                if pt_len == 0 {
                    continue;
                }

                let ct_len = pt_len + 16;
                let total_frame_len = 2 + 16 + 12 + ct_len + 4;

                if total_frame_len > frame::MAX_FRAME_BYTES {
                    continue;
                }

                // Plausibility: the RS-encoded frame must fit in extracted data.
                let rs_encoded_len =
                    ecc::rs_encoded_len_with_parity(total_frame_len, parity);
                if rs_encoded_len > extracted_bytes.len() {
                    continue;
                }

                if total_frame_len == data_len {
                    // CRC gate before returning. plausibility
                    // passed but the random-bytes case can still slip
                    // through; CRC pins it to the true encoder output.
                    if !frame_crc_ok(&first_block_data, total_frame_len) {
                        continue;
                    }
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
                    if !frame_crc_ok(&first_block_data, total_frame_len) {
                        continue;
                    }
                    let t_max = parity / 2;
                    let stats = ecc::RsDecodeStats {
                        total_errors: first_errors,
                        error_capacity: t_max,
                        max_block_errors: first_errors,
                        num_blocks: 1,
                    };
                    return Some((first_block_data[..total_frame_len].to_vec(), stats));
                }

                if total_frame_len > data_len
                    && let Ok((decoded, stats)) = ecc::rs_decode_blocks_with_parity(
                        &extracted_bytes[..rs_encoded_len],
                        total_frame_len,
                        parity,
                    ) {
                        if !frame_crc_ok(&decoded, total_frame_len) {
                            continue;
                        }
                        return Some((decoded, stats));
                    }
                    // Multi-block decode failed — first-block was a false
                    // positive. Continue sweep to try other data_len values.
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
            && first_block_data.len() >= 2 {
                let pt_len =
                    u16::from_be_bytes([first_block_data[0], first_block_data[1]]) as usize;

                // Plausibility: plaintext_len must be positive.
                if pt_len == 0 {
                    continue;
                }

                let ct_len = pt_len + 16;
                // Compact frame: no salt (16) or nonce (12)
                let total_frame_len = 2 + ct_len + 4;

                if total_frame_len > frame::MAX_FRAME_BYTES {
                    continue;
                }

                // Plausibility: the RS-encoded frame must fit in extracted data.
                let rs_encoded_len =
                    ecc::rs_encoded_len_with_parity(total_frame_len, parity);
                if rs_encoded_len > extracted_bytes.len() {
                    continue;
                }

                if total_frame_len == data_len {
                    // CRC gate before returning.
                    if !frame_crc_ok(&first_block_data, total_frame_len) {
                        continue;
                    }
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
                    if !frame_crc_ok(&first_block_data, total_frame_len) {
                        continue;
                    }
                    let t_max = parity / 2;
                    let stats = ecc::RsDecodeStats {
                        total_errors: first_errors,
                        error_capacity: t_max,
                        max_block_errors: first_errors,
                        num_blocks: 1,
                    };
                    return Some((first_block_data[..total_frame_len].to_vec(), stats));
                }

                if total_frame_len > data_len
                    && let Ok((decoded, stats)) = ecc::rs_decode_blocks_with_parity(
                        &extracted_bytes[..rs_encoded_len],
                        total_frame_len,
                        parity,
                    ) {
                        if !frame_crc_ok(&decoded, total_frame_len) {
                            continue;
                        }
                        return Some((decoded, stats));
                    }
                    // Multi-block decode failed — first-block was a false
                    // positive. Continue sweep to try other data_len values.
            }
    }

    None
}

/// Try to RS-decode a frame from extracted bytes (Phase 1 path, fixed parity=64).
fn try_rs_decode_frame(extracted_bytes: &[u8]) -> Result<(Vec<u8>, ecc::RsDecodeStats), StegoError> {
    let parity = ecc::parity_len();

    let min_data = crate::stego::frame::FRAME_OVERHEAD;
    let max_first_block_data = 191usize; // K_DEFAULT

    for data_len in min_data..=max_first_block_data.min(extracted_bytes.len().saturating_sub(parity))
    {
        let block_len = data_len + parity;
        if block_len > extracted_bytes.len() {
            break;
        }

        if let Ok((first_block_data, first_errors)) = ecc::rs_decode(&extracted_bytes[..block_len], data_len)
            && first_block_data.len() >= 2 {
                let pt_len =
                    u16::from_be_bytes([first_block_data[0], first_block_data[1]]) as usize;

                // Plausibility: plaintext_len must be positive.
                if pt_len == 0 {
                    continue;
                }

                let ct_len = pt_len + 16;
                let total_frame_len = 2 + 16 + 12 + ct_len + 4;

                if total_frame_len > frame::MAX_FRAME_BYTES {
                    continue;
                }

                // Plausibility: the RS-encoded frame must fit in extracted data.
                let rs_encoded_len = ecc::rs_encoded_len(total_frame_len);
                if rs_encoded_len > extracted_bytes.len() {
                    continue;
                }

                if total_frame_len <= data_len {
                    let stats = ecc::RsDecodeStats {
                        total_errors: first_errors,
                        error_capacity: ecc::T_MAX,
                        max_block_errors: first_errors,
                        num_blocks: 1,
                    };
                    // Single block: truncate to exact frame length if needed.
                    let frame_data = if total_frame_len == data_len {
                        first_block_data
                    } else {
                        first_block_data[..total_frame_len].to_vec()
                    };
                    return Ok((frame_data, stats));
                }

                // total_frame_len > data_len: multi-block RS decode
                if let Ok((decoded, stats)) = ecc::rs_decode_blocks(
                    &extracted_bytes[..rs_encoded_len],
                    total_frame_len,
                ) {
                    return Ok((decoded, stats));
                }
                // Multi-block decode failed — first-block was a false
                // positive. Continue sweep to try other data_len values.
            }
    }

    Err(StegoError::FrameCorrupted)
}

// --- Decoder coefficient sources and spreading vectors ---

/// Where the STDM decoder reads Y coefficients from, by flat index.
pub(crate) trait ArmorCoeffs: Sync {
    fn coeff(&self, flat_idx: usize) -> i16;
    /// True when the position is known to carry no information (an erasure).
    fn erased(&self, _flat_idx: usize) -> bool {
        false
    }
    /// The value STDM projects: the index (legacy) or index x q (v2).
    #[inline]
    fn value(&self, flat_idx: usize) -> f64 {
        self.coeff(flat_idx) as f64
    }
}

/// LLR of one unit. Erased positions (whose received value is unrelated to
/// the embedding) count as 0, their expected value; when they carry more than
/// half of the spreading vector's energy the unit is an erasure (LLR 0).
/// Erasing a unit for a single lost position throws away the other seven:
/// measured, that made legacy decodes fail that succeed on the raw values.
#[inline]
fn unit_llr<C: ArmorCoeffs + ?Sized>(
    src: &C,
    positions: &[u32],
    unit_idx: usize,
    v: &[f64; SPREAD_LEN],
    delta: f64,
) -> f64 {
    let group = &positions[unit_idx * SPREAD_LEN..(unit_idx + 1) * SPREAD_LEN];
    if !group.iter().any(|&p| src.erased(p as usize)) {
        return stdm_extract_soft(&unit_coeffs(src, positions, unit_idx), v, delta);
    }
    let mut coeffs = [0.0f64; SPREAD_LEN];
    let mut lost = 0.0;
    for (k, &p) in group.iter().enumerate() {
        if src.erased(p as usize) {
            lost += v[k] * v[k];
        } else {
            coeffs[k] = src.value(p as usize);
        }
    }
    if lost > 0.5 {
        return 0.0;
    }
    stdm_extract_soft(&coeffs, v, delta)
}

/// A received luma grid read through an encoder block geometry
/// (`enc_bw` blocks per row): positions map by (row, col); blocks outside the
/// received grid read 0 and are erased; with `erase_dummy` so are the
/// MCU-padding blocks of the received grid (outside its real ceil(W/8) x
/// ceil(H/8) area), whose content a re-encode regenerates.
pub(crate) struct MappedGrid<'a> {
    pub(crate) grid: &'a DctGrid,
    pub(crate) enc_bw: usize,
    pub(crate) real_bw: usize,
    pub(crate) real_bh: usize,
    pub(crate) erase_dummy: bool,
}

impl MappedGrid<'_> {
    #[inline]
    fn locate(&self, flat_idx: usize) -> (usize, usize, usize) {
        let blk = flat_idx / 64;
        (blk / self.enc_bw, blk % self.enc_bw, flat_idx % 64)
    }
}

impl ArmorCoeffs for MappedGrid<'_> {
    #[inline]
    fn coeff(&self, flat_idx: usize) -> i16 {
        let (br, bc, slot) = self.locate(flat_idx);
        if br < self.grid.blocks_tall() && bc < self.grid.blocks_wide() {
            self.grid.get(br, bc, slot / 8, slot % 8)
        } else {
            0
        }
    }
    #[inline]
    fn erased(&self, flat_idx: usize) -> bool {
        let (br, bc, _) = self.locate(flat_idx);
        if br >= self.grid.blocks_tall() || bc >= self.grid.blocks_wide() {
            return true;
        }
        self.erase_dummy && (br >= self.real_bh || bc >= self.real_bw)
    }
}

impl ArmorCoeffs for DctGrid {
    #[inline]
    fn coeff(&self, flat_idx: usize) -> i16 {
        flat_get(self, flat_idx)
    }
}

/// The coefficients of embedding unit `unit_idx`, as f64.
#[inline]
fn unit_coeffs<C: ArmorCoeffs + ?Sized>(src: &C, positions: &[u32], unit_idx: usize) -> [f64; SPREAD_LEN] {
    let group = &positions[unit_idx * SPREAD_LEN..(unit_idx + 1) * SPREAD_LEN];
    let mut coeffs = [0.0f64; SPREAD_LEN];
    for (k, &pos) in group.iter().enumerate() {
        coeffs[k] = src.value(pos as usize);
    }
    coeffs
}

/// Units the decoder extracts per spreading-vector chunk.
const UNIT_CHUNK: usize = 4096;

/// Largest spreading-vector table the decoder keeps (64 bytes per unit, one
/// unit per 8 positions): up to ~2M units (~70 MP) the vectors are generated
/// once — below the geometry stage's peak, so the table costs no peak there;
/// above that every extraction pass regenerates its chunks
/// (`spreading_vectors_range`, bit-identical), trading ChaCha20 time (~1.5x
/// the STDM search) for the table's memory.
const SPREAD_TABLE_BYTES: usize = 128 << 20;

/// The decoder's spreading vectors: a table while small, regenerated per chunk
/// on large covers.
struct SpreadVectors {
    seed: [u8; 32],
    table: Option<Vec<[f64; SPREAD_LEN]>>,
}

impl SpreadVectors {
    fn new(seed: &[u8; 32], count: usize) -> Self {
        let table = (count * std::mem::size_of::<[f64; SPREAD_LEN]>() <= SPREAD_TABLE_BYTES)
            .then(|| generate_spreading_vectors(seed, count));
        SpreadVectors { seed: *seed, table }
    }

    fn range(&self, start: usize, count: usize) -> std::borrow::Cow<'_, [[f64; SPREAD_LEN]]> {
        match &self.table {
            Some(t) => std::borrow::Cow::Borrowed(&t[start..start + count]),
            None => std::borrow::Cow::Owned(spreading_vectors_range(&self.seed, start, count)),
        }
    }
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
    /// The same with an empty passphrase: the encoder then writes the compact
    /// frame (22 B overhead instead of 50), so 28 B more fit. Frontends compare
    /// against this one when the passphrase is empty.
    pub fortress_capacity_compact: usize,
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
    let fortress_compact = fortress::fortress_capacity_compact(&img).unwrap_or(0);
    let stdm_cap = super::capacity::estimate_armor_capacity(&img).unwrap_or(0);

    Ok(ArmorCapacityInfo {
        fortress_capacity: fortress_cap,
        fortress_capacity_compact: fortress_compact,
        stdm_capacity: stdm_cap,
    })
}

// --- Fortress QF75 pre-settlement ---

/// Standard JPEG luminance quantization table (Table K.1 from the JPEG spec).
/// These are the base values before quality-factor scaling.
/// NOTE: Duplicates STD_LUMA_QT in selection.rs — kept here to avoid coupling
/// the Fortress pre-settlement path to the stability-map module.
pub(crate) const JPEG_BASE_LUMINANCE_QT: [u16; 64] = [
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
pub(crate) fn compute_jpeg_qt(base: &[u16; 64], qf: u32) -> [u16; 64] {
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
fn pre_settle_for_fortress(img: &mut JpegImage, headroom: f64) -> Result<(), StegoError> {
    use crate::codec::jpeg::dct::QuantTable;

    let num_components = img.num_components();
    let target_qf = 75u32;

    // Target QT per QT slot: the luminance base for the slot luma uses (also
    // for chroma components sharing it: a component is decoded with its
    // slot's one table), the chrominance base for the others. We must
    // re-quantize ALL components, even those sharing a QT ID, because each
    // component has its own DctGrid with separate coefficients.
    let y_id = img.frame_info().components[0].quant_table_id as usize;
    let mut new_qts: Vec<(usize, [u16; 64], [u16; 64])> = Vec::new(); // (qt_id, old, new)

    for comp_idx in 0..num_components {
        let qt_id = img.frame_info().components[comp_idx].quant_table_id as usize;
        let old_qt = img.quant_table(qt_id)
            .ok_or(StegoError::NoLuminanceChannel)?.values;
        let base = if qt_id == y_id {
            &JPEG_BASE_LUMINANCE_QT
        } else {
            &JPEG_BASE_CHROMINANCE_QT
        };
        let new_qt = compute_jpeg_qt(base, target_qf);

        // Re-quantize all blocks in this component — parallel when available.
        let grid = img.dct_grid_mut(comp_idx);

        let process_block = |chunk: &mut [i16]| {
            let quantized: [i16; 64] = chunk.try_into().unwrap();
            let mut px = idct_block_active(&quantized, &old_qt);
            let h = if comp_idx == 0 { headroom } else { 0.0 };
            for p in px.iter_mut() {
                *p = headroom_map(*p, h);
            }
            let settled = dct_block_active(&px, &new_qt);
            chunk.copy_from_slice(&settled);
        };

        #[cfg(feature = "parallel")]
        grid.coeffs_mut().par_chunks_mut(64).for_each(process_block);
        #[cfg(not(feature = "parallel"))]
        grid.coeffs_mut().chunks_mut(64).for_each(process_block);

        new_qts.push((qt_id, old_qt, new_qt));

        // T3 progress recalibration — one step per channel. Pre-settle
        // is the biggest Fortress wall-clock consumer (IDCT+DCT on every
        // block × 3 channels); subdividing per-channel keeps the bar
        // moving during the dominant phase.
        progress::advance();
    }
    // `ARMOR_ENCODE_FORTRESS_STEPS` counts three channels (grayscale has one).
    for _ in num_components..3 {
        progress::advance();
    }

    // Replace quantization tables (deduplicate by QT ID)
    let mut replaced = [false; 4];
    for (qt_id, _old, new_qt) in &new_qts {
        if !replaced[*qt_id] {
            img.set_quant_table(*qt_id, QuantTable::new(*new_qt));
            replaced[*qt_id] = true;
        }
    }
    Ok(())
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

}

#[cfg(test)]
mod standard_switch_tests {
    use super::*;

    fn cover() -> Vec<u8> {
        let (w, h) = (640u32, 480u32);
        let mut s: u32 = 9;
        let px: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let (x, y) = (i % w, i / w);
                let g = x * 80 / w + y * 60 / h;
                [(40 + g + (s >> 26)) as u8, (60 + g / 2 + (s >> 27)) as u8, (90 + (s >> 26)) as u8]
            })
            .collect();
        crate::stego::deep_cover_jpeg(&px, w, h, 65).unwrap()
    }

    /// The Standard switch reaches the encoder, and the result reports the
    /// sub-mode actually used (fix list C7: the apps showed a prediction).
    #[test]
    fn force_standard_skips_fortress_and_quality_reports_the_mode() {
        let c = cover();
        let (auto, q_auto) = armor_encode_with_options(&c, "short", "pw", false).unwrap();
        assert!(q_auto.fortress, "a short message uses Fortress");
        let (std, q_std) = armor_encode_with_options(&c, "short", "pw", true).unwrap();
        assert!(!q_std.fortress, "forced Standard does not");
        assert_eq!(crate::stego::smart_decode(&auto, "pw").unwrap().1.fortress_used, true);
        assert_eq!(crate::stego::smart_decode(&std, "pw").unwrap().1.fortress_used, false);
    }

    /// With an empty passphrase the encoder writes the compact frame, so the
    /// compact capacity (28 B more) is the boundary, exactly.
    #[test]
    fn compact_fortress_capacity_is_the_empty_passphrase_boundary() {
        let c = cover();
        let info = armor_capacity_info(&c).unwrap();
        assert_eq!(info.fortress_capacity_compact, info.fortress_capacity + 28);
        // The longest message whose payload is exactly the compact capacity.
        let mut msg = String::new();
        let mut r: u32 = 1;
        while crate::stego::payload::compressed_payload_size(&msg, &[]) < info.fortress_capacity_compact {
            r = r.wrapping_mul(1103515245).wrapping_add(12345);
            msg.push((b'a' + (r >> 16) as u8 % 26) as char);
        }
        while crate::stego::payload::compressed_payload_size(&msg, &[]) > info.fortress_capacity_compact {
            msg.pop();
        }
        let size = crate::stego::payload::compressed_payload_size(&msg, &[]);
        assert!(size > info.fortress_capacity, "beyond the passphrase capacity");
        let (_, q) = armor_encode_with_options(&c, &msg, "", false).unwrap();
        assert!(q.fortress, "{size} B fits the compact frame");
        let (_, q) = armor_encode_with_options(&c, &msg, "pw", false).unwrap();
        assert!(!q.fortress, "{size} B does not fit the passphrase frame");
    }

    /// A message whose payload is exactly `target` bytes (searched over seeds:
    /// compressed text does not grow by whole bytes per character).
    fn message_of_payload(target: usize) -> String {
        for seed in 1u32..200 {
            let (mut msg, mut r) = (String::new(), seed);
            while crate::stego::payload::compressed_payload_size(&msg, &[]) < target {
                r = r.wrapping_mul(1103515245).wrapping_add(12345);
                msg.push((b'a' + (r >> 16) as u8 % 26) as char);
            }
            if crate::stego::payload::compressed_payload_size(&msg, &[]) == target {
                return msg;
            }
        }
        panic!("no message of payload {target}");
    }

    /// Fortress T's capacity does not depend on the passphrase (the HUD has
    /// none), and it is the exact Fortress / Standard boundary for each one.
    #[test]
    fn fortress_capacity_is_the_boundary_for_every_passphrase() {
        let c = cover();
        let info = armor_capacity_info(&c).unwrap();
        let at = message_of_payload(info.fortress_capacity);
        let over = message_of_payload(info.fortress_capacity + 1);
        for pass in ["pw", "correct horse battery", "x", "Zuerich-2026", "phasm"] {
            assert!(armor_encode_with_options(&c, &at, pass, false).unwrap().1.fortress, "{pass}: at capacity");
            assert!(!armor_encode_with_options(&c, &over, pass, false).unwrap().1.fortress, "{pass}: one byte over");
        }
    }
}
