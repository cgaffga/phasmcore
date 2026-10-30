// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Ghost mode encode/decode pipeline.
//!
//! **Encode** (Ghost embedding v3, `docs/design/image/ghost-embedding-v3.md`):
//! J-UNIWARD costs of every Y AC coefficient (SI-UNIWARD with side
//! information), a keyed permutation of the whole AC set, and a ternary
//! (double-layered) STC over the two lowest bit planes of the coefficient
//! values, in independent lanes. Shadow layers ([`super::shadow`]) take
//! keyed regions of their own hash orders first; the primary treats them as
//! wet.
//!
//! **Decode** supports exactly two generations: v3, and v1 — what core
//! ≤ 0.4.3 / apps ≤ 1.8 wrote: a binary STC (h = 7, width 1..10) on `|c| & 1`
//! of a keyed permutation of the Y non-zero AC set, and shadow layout v1
//! (legacy tier + cost pools, see `shadow.rs`). Neither decoder computes
//! costs except the v1 cost-pool fallback, which runs only after everything
//! else failed.

use crate::codec::jpeg::JpegImage;
use crate::codec::jpeg::dct::DctGrid;
use crate::stego::cost::uniward::compute_positions_streaming_legacy_v1;
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::frame::{self, MAX_FRAME_BITS};
use crate::stego::memory::{self, GhostRung, ModeId, TelemetryEvent};
use crate::stego::payload::{self, FileEntry, PayloadData};

use crate::stego::progress;
use crate::stego::quality::{self, EncodeQuality, GhostMetrics};
use crate::stego::shadow;
use crate::stego::side_info::SideInfo;
use crate::stego::stc::{extract, hhat};
use super::v3;

/// STC constraint height of the v1 decoder's binary STC.
const V1_STC_H: usize = 7;

/// Progress steps allocated to JPEG parsing.  Reported immediately after
/// `JpegImage::from_bytes` returns so the bar moves off 0% right away.
const PARSE_STEPS: u32 = 5;

/// Progress steps of the embedding trellises (all lanes together).
const LANE_STEPS: u32 = 40;

/// Progress steps of one shadow layer's trellis.
const SHADOW_STEPS: u32 = 5;

/// Total number of progress steps reported by [`ghost_decode`]:
/// parse + v3 attempt (order, search) + v1 attempt (permutation, extraction).
pub const GHOST_DECODE_STEPS: u32 = PARSE_STEPS + 2 + 2;

/// Total number of progress steps reported by a Ghost encode without
/// shadows: parse + key + cost passes and trellises (one meter) + JPEG write.
pub const GHOST_ENCODE_STEPS: u32 =
    PARSE_STEPS
    + crate::stego::cost::uniward::UNIWARD_PROGRESS_STEPS
    + 1
    + LANE_STEPS
    + crate::codec::jpeg::scan::JPEG_WRITE_STEPS;

/// Progress steps for Ghost encode with shadows (base budget, one layout
/// attempt): [`GHOST_ENCODE_STEPS`] + one shadow-prep step; the encoder adds
/// [`SHADOW_STEPS`] per shadow, and a retried layout (narrower width) bumps
/// the total via `progress::set_total()`.
pub const GHOST_ENCODE_WITH_SHADOWS_STEPS: u32 = GHOST_ENCODE_STEPS + 1;

/// The v1 decoder's STC parameters for a cover of `n` nzAC positions: the
/// "natural" width (`n / m_max` with `m_max = min(MAX_FRAME_BITS, n)`), the
/// first candidate the v1 decoder tries.
fn v1_natural_w(n: usize) -> usize {
    let m_max = MAX_FRAME_BITS.min(n);
    if m_max == 0 { 1 } else { (n / m_max).max(1) }
}

// ---------------------------------------------------------------------------
// Public encode API
// ---------------------------------------------------------------------------

/// Encode a text message into a cover JPEG using Ghost mode.
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
    ghost_encode_impl(image_bytes, message, &[], passphrase, None, None)
        .map(|(bytes, _)| bytes)
}

/// Encode with Ghost mode and return the encode quality score.
pub fn ghost_encode_with_quality(
    image_bytes: &[u8],
    message: &str,
    passphrase: &str,
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    ghost_encode_impl(image_bytes, message, &[], passphrase, None, None)
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
    ghost_encode_impl(image_bytes, message, files, passphrase, None, None)
        .map(|(bytes, _)| bytes)
}

/// Encode with Ghost mode + files and return the encode quality score.
pub fn ghost_encode_with_files_quality(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    ghost_encode_impl(image_bytes, message, files, passphrase, None, None)
}

/// Encode using Ghost mode with side information (SI-UNIWARD / "Deep Cover").
///
/// When the original uncompressed pixels are available (non-JPEG input like
/// PNG, HEIC, or RAW), the quantization rounding errors steer the direction
/// of each change toward the precover (cost `ρ·max(1 − 2|e|, 0.3)`).
///
/// The `raw_pixels_rgb` must be the ORIGINAL pixels that were JPEG-compressed
/// to produce `image_bytes`. They must have the same dimensions.
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

/// Encode with SI-UNIWARD and return the encode quality score.
pub fn ghost_encode_si_with_quality(
    image_bytes: &[u8],
    raw_pixels_rgb: &[u8],
    pixel_width: u32,
    pixel_height: u32,
    message: &str,
    passphrase: &str,
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    ghost_encode_si_with_files_quality(
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
    ghost_encode_si_with_files_quality(
        image_bytes, raw_pixels_rgb, pixel_width, pixel_height,
        message, files, passphrase,
    ).map(|(bytes, _)| bytes)
}

/// Encode with SI-UNIWARD + files and return the encode quality score.
pub fn ghost_encode_si_with_files_quality(
    image_bytes: &[u8],
    raw_pixels_rgb: &[u8],
    pixel_width: u32,
    pixel_height: u32,
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    let (img, si) = parse_with_side_info(image_bytes, raw_pixels_rgb, pixel_width, pixel_height)?;
    ghost_encode_impl(image_bytes, message, files, passphrase, Some(si), Some(img))
}

/// Parse the cover and compute the side information of its Y channel.
pub(super) fn parse_with_side_info(
    image_bytes: &[u8],
    raw_pixels_rgb: &[u8],
    pixel_width: u32,
    pixel_height: u32,
) -> Result<(JpegImage, SideInfo), StegoError> {
    let img = JpegImage::from_bytes(image_bytes)?;
    let fi = img.frame_info();
    crate::stego::validate_encode_dimensions(fi.width as u32, fi.height as u32)?;
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let qt_id = fi.components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    let si = SideInfo::compute(raw_pixels_rgb, pixel_width, pixel_height, img.dct_grid(0), &qt.values)?;
    Ok((img, si))
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

/// Encode with Ghost mode plus shadow messages for plausible deniability.
///
/// Each shadow takes a keyed region of the AC set (its own STC); the primary
/// message is embedded around them. Each passphrase must be unique.
pub fn ghost_encode_with_shadows(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    shadows: &[ShadowLayer],
    si: Option<SideInfo>,
) -> Result<Vec<u8>, StegoError> {
    shadows_encode(image_bytes, message, files, passphrase, shadows, si, None)
        .map(|(bytes, _)| bytes)
}

/// Encode with Ghost mode + shadows and return the encode quality score.
pub fn ghost_encode_with_shadows_quality(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    shadows: &[ShadowLayer],
    si: Option<SideInfo>,
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    shadows_encode(image_bytes, message, files, passphrase, shadows, si, None)
}

/// Encode with Ghost SI-UNIWARD plus shadow messages.
#[allow(clippy::too_many_arguments)]
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
    ghost_encode_si_with_shadows_quality(
        image_bytes, raw_pixels_rgb, pixel_width, pixel_height,
        message, files, passphrase, shadows,
    ).map(|(bytes, _)| bytes)
}

/// Encode with SI-UNIWARD + shadows and return the encode quality score.
#[allow(clippy::too_many_arguments)]
pub fn ghost_encode_si_with_shadows_quality(
    image_bytes: &[u8],
    raw_pixels_rgb: &[u8],
    pixel_width: u32,
    pixel_height: u32,
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    shadows: &[ShadowLayer],
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    let (img, si) = parse_with_side_info(image_bytes, raw_pixels_rgb, pixel_width, pixel_height)?;
    shadows_encode(image_bytes, message, files, passphrase, shadows, Some(si), Some(img))
}

// ---------------------------------------------------------------------------
// Encode implementation
// ---------------------------------------------------------------------------

/// Run an encode body with the memory telemetry around it.
fn with_telemetry<T>(
    mode: ModeId,
    width: u32,
    height: u32,
    rung: GhostRung,
    body: impl FnOnce() -> Result<T, StegoError>,
) -> Result<T, StegoError> {
    let workers = memory::encode_workers();
    let predicted_mb = (memory::predict_peak_memory(mode, width, height, rung, workers) / (1024 * 1024)) as u32;
    let budget_mb = memory::get_memory_budget().map(|b| (b / (1024 * 1024)) as u32);
    memory::emit(TelemetryEvent::LadderRungSelected {
        mode, width, height, predicted_peak_mb: predicted_mb, budget_mb, rung: rung.as_u8(),
    });
    memory::emit(TelemetryEvent::EncodeStarted {
        mode, width, height, predicted_peak_mb: predicted_mb, budget_mb, ladder_rung: rung.as_u8(),
    });
    #[cfg(not(target_arch = "wasm32"))]
    let start = std::time::Instant::now();
    let outcome = body();
    #[cfg(not(target_arch = "wasm32"))]
    let wall_ms = start.elapsed().as_millis().min(u32::MAX as u128) as u32;
    #[cfg(target_arch = "wasm32")]
    let wall_ms = 0u32;
    match &outcome {
        Ok(_) => memory::emit(TelemetryEvent::EncodeCompleted {
            mode, width, height, ladder_rung: rung.as_u8(), wall_clock_ms: wall_ms,
        }),
        Err(StegoError::Cancelled) => memory::emit(TelemetryEvent::EncodeCancelled {
            mode, width, height, ladder_rung: rung.as_u8(), wall_clock_ms_when_cancelled: wall_ms,
        }),
        Err(e) => memory::emit(TelemetryEvent::EncodeFailed { mode, width, height, reason: stego_error_kind(e) }),
    }
    outcome
}

fn stego_error_kind(e: &StegoError) -> &'static str {
    match e {
        StegoError::InvalidJpeg(_) => "InvalidJpeg",
        StegoError::ImageTooSmall => "ImageTooSmall",
        StegoError::ImageTooLarge => "ImageTooLarge",
        StegoError::MessageTooLarge => "MessageTooLarge",
        StegoError::FrameCorrupted => "FrameCorrupted",
        StegoError::DecryptionFailed => "DecryptionFailed",
        StegoError::InvalidUtf8 => "InvalidUtf8",
        StegoError::NoLuminanceChannel => "NoLuminanceChannel",
        StegoError::Cancelled => "Cancelled",
        StegoError::KeyDerivationFailed => "KeyDerivationFailed",
        StegoError::DuplicatePassphrase => "DuplicatePassphrase",
        StegoError::ShadowEmbedFailed => "ShadowEmbedFailed",
        StegoError::InvalidVideo(_) => "InvalidVideo",
    }
}

/// Encoder options for this cover and frame under the memory budget.
fn encode_opts(mode: ModeId, img: &JpegImage, m_bits: usize) -> (v3::EncodeOpts, GhostRung) {
    let fi = img.frame_info();
    let n_ac = v3::ac_count(img.dct_grid(0));
    let h = v3::V3_STC_H;
    let plan = memory::ghost_lane_plan(mode, fi.width as u32, fi.height as u32, n_ac, m_bits, h);
    let opts = v3::EncodeOpts {
        h,
        parallel_lanes: plan.parallel_lanes,
        lane_inline_bytes: plan.lane_inline_bytes,
        group_positions: plan.group_positions,
        progress_steps: crate::stego::cost::uniward::UNIWARD_PROGRESS_STEPS + LANE_STEPS,
        bound: std::env::var_os("PHASM_GHOST_V3_STATS").is_some(),
    };
    (opts, plan.rung)
}

/// Encrypt a payload and build its Ghost frame bits.
fn frame_bits_for(payload_bytes: &[u8], passphrase: &str) -> Result<Vec<u8>, StegoError> {
    let (ciphertext, nonce, salt) = crypto::encrypt(payload_bytes, passphrase)?;
    Ok(frame::bytes_to_bits(&frame::build_frame(payload_bytes.len(), &salt, &nonce, &ciphertext)))
}

/// Write the stego JPEG (original Huffman tables where they fit, else
/// rebuilt), reporting `JPEG_WRITE_STEPS`.
fn write_jpeg(img: &mut JpegImage) -> Result<Vec<u8>, StegoError> {
    let progress_cb = || progress::advance();
    if let Ok(bytes) = img.to_bytes_with_progress(Some(&progress_cb)) {
        return Ok(bytes);
    }
    img.rebuild_huffman_tables();
    img.to_bytes_with_progress(Some(&progress_cb)).map_err(StegoError::InvalidJpeg)
}

fn print_stats(stats: &v3::V3Stats) {
    if std::env::var_os("PHASM_GHOST_V3_STATS").is_some() {
        eprintln!(
            "v3stats n={} m={} lanes={} m2={} w1={} w2={} changes={} distortion={:.1} bound={:.1}",
            stats.n, stats.m, stats.lanes, stats.m2, stats.w1, stats.w2, stats.changes, stats.distortion,
            stats.bound_distortion
        );
    }
}

/// A primary-only Ghost encode up to the embedding: the parsed cover, the
/// frame bits and the encoder options.
pub(super) struct PreparedEncode {
    pub(super) img: JpegImage,
    pub(super) frame_bits: Vec<u8>,
    pub(super) opts: v3::EncodeOpts,
    pub(super) rung: GhostRung,
    pub(super) mode: ModeId,
    pub(super) width: u32,
    pub(super) height: u32,
}

/// Payload, parse (unless `pre_parsed`), validation, frame bits, options.
pub(super) fn prepare_encode(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    is_si: bool,
    pre_parsed: Option<JpegImage>,
) -> Result<PreparedEncode, StegoError> {
    let payload_bytes = payload::encode_payload(message, files)?;
    let img = match pre_parsed {
        Some(img) => img,
        None => JpegImage::from_bytes(image_bytes)?,
    };
    progress::advance_by(PARSE_STEPS);
    let fi = img.frame_info();
    let (width, height) = (fi.width as u32, fi.height as u32);
    crate::stego::validate_encode_dimensions(width, height)?;
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let mode = if is_si { ModeId::GhostSi } else { ModeId::GhostNoShadow };
    // The frame length is known before any heavy work: size the lanes now.
    let frame_bits = frame_bits_for(&payload_bytes, passphrase)?;
    let (opts, rung) = encode_opts(mode, &img, frame_bits.len());
    Ok(PreparedEncode { img, frame_bits, opts, rung, mode, width, height })
}

/// Score an embedded cover and write the stego JPEG.
pub(super) fn finish_encode(img: &mut JpegImage, stats: &v3::V3Stats, is_si: bool) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    print_stats(stats);
    let grid = img.dct_grid(0);
    let encode_quality = quality::ghost_stealth_score(&GhostMetrics {
        num_modifications: stats.changes,
        n_used: stats.n,
        w: stats.w1,
        total_cost: stats.distortion,
        median_cost: stats.median_cost,
        is_si,
        shadow_modifications: 0,
        total_coefficients: grid.blocks_wide() * grid.blocks_tall() * 64,
    });
    Ok((write_jpeg(img)?, encode_quality))
}

pub(super) fn ghost_encode_impl(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    si: Option<SideInfo>,
    pre_parsed: Option<JpegImage>,
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    progress::init(GHOST_ENCODE_STEPS);
    let PreparedEncode { mut img, frame_bits, opts, rung, mode, width, height } =
        prepare_encode(image_bytes, message, files, passphrase, si.is_some(), pre_parsed)?;

    with_telemetry(mode, width, height, rung, move || {
        let structural_key = crypto::derive_structural_key(passphrase)?;
        progress::advance();
        let stats = v3::encode(&mut img, &frame_bits, &structural_key, si.as_ref(), &opts)?;
        let is_si = si.is_some();
        drop(si);
        finish_encode(&mut img, &stats, is_si)
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn shadows_encode(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    shadows: &[ShadowLayer],
    si: Option<SideInfo>,
    pre_parsed: Option<JpegImage>,
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    // Validate: unique passphrases.
    {
        let mut seen = std::collections::HashSet::new();
        seen.insert(passphrase);
        for s in shadows {
            if !seen.insert(s.passphrase.as_str()) {
                return Err(StegoError::DuplicatePassphrase);
            }
        }
    }
    if shadows.is_empty() {
        return ghost_encode_impl(image_bytes, message, files, passphrase, si, pre_parsed);
    }
    progress::init(GHOST_ENCODE_WITH_SHADOWS_STEPS + SHADOW_STEPS * shadows.len() as u32);

    // Auto-sort: the largest payload becomes the primary; the original
    // primary then becomes a shadow.
    let size = |m: &str, f: &[FileEntry]| payload::compressed_payload_size(m, f);
    let mut swap_idx: Option<usize> = None;
    let mut largest = size(message, files);
    for (i, s) in shadows.iter().enumerate() {
        let sz = size(&s.message, &s.files);
        if sz > largest {
            largest = sz;
            swap_idx = Some(i);
        }
    }
    let (eff_message, eff_files, eff_pass): (&str, &[FileEntry], &str) = match swap_idx {
        Some(i) => (&shadows[i].message, &shadows[i].files, &shadows[i].passphrase),
        None => (message, files, passphrase),
    };
    let mut frames: Vec<shadow::ShadowFrame> = Vec::with_capacity(shadows.len());
    let mut shadow_passes: Vec<&str> = Vec::with_capacity(shadows.len());
    if swap_idx.is_some() {
        frames.push(shadow::ShadowFrame::new(passphrase, message, files)?);
        shadow_passes.push(passphrase);
    }
    for (i, s) in shadows.iter().enumerate() {
        if Some(i) != swap_idx {
            frames.push(shadow::ShadowFrame::new(&s.passphrase, &s.message, &s.files)?);
            shadow_passes.push(&s.passphrase);
        }
    }
    let primary_payload = payload::encode_payload(eff_message, eff_files)?;
    let primary_pass = eff_pass;

    let mut img = match pre_parsed {
        Some(img) => img,
        None => JpegImage::from_bytes(image_bytes)?,
    };
    progress::advance_by(PARSE_STEPS);
    let fi = img.frame_info();
    let (width, height) = (fi.width as u32, fi.height as u32);
    crate::stego::validate_encode_dimensions(width, height)?;
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let mode = if si.is_some() { ModeId::GhostSiShadow } else { ModeId::GhostShadow };
    let frame_bits = frame_bits_for(&primary_payload, primary_pass)?;
    let (opts, rung) = encode_opts(mode, &img, frame_bits.len());

    with_telemetry(mode, width, height, rung, move || {
        shadows_encode_body(img, &frame_bits, primary_pass, &frames, &shadow_passes, si, opts)
    })
}

/// The shadow layout: one common width `sw` for all shadows, from the widest
/// the cover allows (every layer at the same rate) down the v3 ladder; each
/// shadow plans its region along its hash order with earlier regions wet,
/// then the primary embeds with every region wet; the first width at which
/// every trellis is feasible and every shadow verifies wins.
///
/// The image keeps the cover during planning (every cost is the cover's);
/// planned layers live in an overlay and are written only at the end.
fn shadows_encode_body(
    mut img: JpegImage,
    frame_bits: &[u8],
    passphrase: &str,
    frames: &[shadow::ShadowFrame],
    shadow_passes: &[&str],
    si: Option<SideInfo>,
    opts: v3::EncodeOpts,
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    let structural_key = crypto::derive_structural_key(passphrase)?;
    let n = v3::ac_count(img.dct_grid(0));
    let total_coefficients = img.dct_grid(0).blocks_wide() * img.dct_grid(0).blocks_tall() * 64;

    let shadow_bits: usize = frames.iter().map(|f| f.rs_bits()).sum();
    let all_bits = frame_bits.len() + shadow_bits;
    let sw_top = n / all_bits.max(1);
    let candidates: Vec<usize> = v3::shadow_widths(n, shadow::SHADOW_MIN_STC_BITS)
        .into_iter()
        .rev()
        .filter(|&w| w <= sw_top)
        .collect();
    let Some(&widest) = candidates.first() else {
        return Err(StegoError::MessageTooLarge);
    };
    let orders: Vec<Vec<u32>> = frames
        .iter()
        .map(|f| shadow::ac_hash_order_prefix(img.dct_grid(0), &f.perm_seed, f.rs_bits() * widest))
        .collect();
    progress::advance();
    let inline_limit = opts.lane_inline_bytes.min(v3::lane_inline_cap(n, opts.h));

    let mut attempt = 0u32;
    for &sw in &candidates {
        progress::check_cancelled()?;
        if attempt > 0 {
            // A retried layout: budget another round of trellises + write.
            let (step, total) = progress::get();
            progress::set_total(total.max(step) + opts.progress_steps + SHADOW_STEPS * frames.len() as u32);
        }
        attempt += 1;
        let mut wet = v3::WetMask::new(total_coefficients);
        let mut overlay = v3::Overlay::default();
        let mut states = Vec::with_capacity(frames.len());
        let mut shadow_mods = 0;
        let mut feasible = true;
        for (f, o) in frames.iter().zip(&orders) {
            let meter = StepMeter::new(f.rs_bits() * sw, SHADOW_STEPS);
            let tick = |d: usize| meter.tick(d);
            let region = &o[..shadow::ShadowState::region_len(f, sw).min(o.len())];
            let rho = v3::gather_costs(&img, region)?;
            let grid = img.dct_grid(0);
            let cost = |i: usize| {
                let fi = region[i];
                if wet.get(fi) {
                    f32::INFINITY
                } else {
                    v3::binary_cost(grid.coeffs()[fi as usize], rho[i], v3::si_term(si.as_ref(), fi as usize))
                }
            };
            let Some(st) = shadow::ShadowState::plan(f, o, sw, &img, &overlay, cost, opts.h, inline_limit, &tick)? else {
                feasible = false;
                break;
            };
            drop(rho);
            shadow_mods += shadow::embed_shadow(&img, &mut overlay, &st, si.as_ref());
            for &fi in &st.positions {
                wet.set(fi);
            }
            states.push(st);
        }
        if !feasible {
            continue;
        }
        let planned = match v3::plan(&img, frame_bits, &structural_key, si.as_ref(), Some(&wet), Some(&overlay), &opts) {
            Ok(p) => p,
            Err(StegoError::MessageTooLarge) => continue,
            Err(e) => return Err(e),
        };
        print_stats(&planned.stats);
        // Write the layers and the primary; keep the cover values to undo.
        let mut undo: Vec<(u32, i16)> = Vec::with_capacity(overlay.len() + planned.changes.len());
        {
            let coeffs = img.dct_grid_mut(0).coeffs_mut();
            for (fi, v) in overlay.iter().chain(planned.changes.iter().copied()) {
                undo.push((fi, coeffs[fi as usize]));
                coeffs[fi as usize] = v;
            }
        }
        let verified = states
            .iter()
            .zip(shadow_passes)
            .all(|(st, pass)| shadow::verify_shadow(&img, st, pass).is_ok());
        if !verified {
            // Cannot happen by construction (regions are wet for every other
            // layer); treat as infeasible at this width rather than emit an
            // unreadable shadow.
            let coeffs = img.dct_grid_mut(0).coeffs_mut();
            for &(fi, v) in undo.iter().rev() {
                coeffs[fi as usize] = v;
            }
            continue;
        }
        let stats = planned.stats;
        let encode_quality = quality::ghost_stealth_score(&GhostMetrics {
            num_modifications: stats.changes,
            n_used: stats.n,
            w: stats.w1,
            total_cost: stats.distortion,
            median_cost: stats.median_cost,
            is_si: si.is_some(),
            shadow_modifications: shadow_mods,
            total_coefficients,
        });
        drop(si);
        drop((wet, overlay, undo));
        return Ok((write_jpeg(&mut img)?, encode_quality));
    }
    Err(StegoError::MessageTooLarge)
}

/// `steps` progress advances spread over `total` units of work.
struct StepMeter {
    total: usize,
    steps: usize,
    done: std::sync::atomic::AtomicUsize,
}

impl StepMeter {
    fn new(total: usize, steps: u32) -> Self {
        Self { total: total.max(1), steps: steps as usize, done: std::sync::atomic::AtomicUsize::new(0) }
    }

    fn tick(&self, k: usize) {
        let before = self.done.fetch_add(k, std::sync::atomic::Ordering::Relaxed);
        let s0 = before.min(self.total) * self.steps / self.total;
        let s1 = (before + k).min(self.total) * self.steps / self.total;
        for _ in s0..s1 {
            progress::advance();
        }
    }
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decode a payload from a stego JPEG using Ghost mode (v3, then v1).
///
/// # Errors
/// - [`StegoError::DecryptionFailed`] if the passphrase is wrong.
/// - [`StegoError::FrameCorrupted`] if no frame is found.
/// - [`StegoError::InvalidUtf8`] if the decrypted payload is not valid UTF-8.
pub fn ghost_decode(
    stego_bytes: &[u8],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    let img = JpegImage::from_bytes(stego_bytes)?;
    progress::advance_by(PARSE_STEPS);
    ghost_decode_from_image(&img, passphrase)
}

/// Ghost decode against an already-parsed JPEG. Used by `smart_decode`'s
/// shared-parse path to avoid re-parsing the same JPEG across the
/// Ghost / Armor / Fortress / Shadow attempt sequence.
pub fn ghost_decode_from_image(
    img: &JpegImage,
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    progress::check_cancelled()?;
    let structural_key = crypto::derive_structural_key(passphrase)?;
    // v3 first: it needs no costs, only the AC set and the key.
    let v3_decrypt_fail = match v3::decode(img, passphrase, &structural_key, try_parse_and_decrypt) {
        Some(Ok(p)) => return Ok(p),
        Some(Err(_)) => true,
        None => false,
    };
    progress::advance_by(2);
    // v1's position set is built only now (4 B per non-zero AC coefficient).
    ghost_decode_v1(img, passphrase, &structural_key, nzac_indices(img.dct_grid(0)), v3_decrypt_fail)
}

/// The v1 primary decode (core ≤ 0.4.3): binary STC on `|c| & 1` along a
/// keyed permutation of the Y nzAC set, every width 1..10.
fn ghost_decode_v1(
    img: &JpegImage,
    passphrase: &str,
    structural_key: &[u8; 64],
    mut positions: Vec<u32>,
    v3_decrypt_fail: bool,
) -> Result<PayloadData, StegoError> {
    progress::check_cancelled()?;
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();
    // The same Fisher–Yates as `permute::permute_positions` (pinned by
    // `v3::tests::shuffle_matches_coeffpos_permutation`), on bare indices.
    v3::shuffle_u32(&mut positions, &perm_seed);
    let n = positions.len();
    progress::advance();

    let all_stego_bits: Vec<u8> = {
        let grid = img.dct_grid(0);
        positions.iter().map(|&fi| (flat_get(grid, fi as usize).unsigned_abs() & 1) as u8).collect()
    };
    drop(positions);

    // Widths: the natural w first (oldest encoders), then 1..10.
    let mut ws: Vec<usize> = Vec::with_capacity(11);
    for w in std::iter::once(v1_natural_w(n)).chain(1..=10) {
        if w > 0 && n / w > 0 && !ws.contains(&w) {
            ws.push(w);
        }
    }
    let try_w = |&w: &usize| -> Option<Result<PayloadData, StegoError>> {
        let m_max = n / w;
        let stego_bits = &all_stego_bits[..m_max * w];
        let hhat_matrix = hhat::generate_hhat(V1_STC_H, w, &hhat_seed);
        // Length-prefix gate: skip wrong-w candidates after ~16 syndrome
        // XORs instead of the full m_max × w.
        if !w_length_prefix_consistent(stego_bits, &hhat_matrix, w, m_max) {
            return None;
        }
        let extracted_bits = extract::stc_extract(stego_bits, &hhat_matrix, w);
        let frame_bytes = frame::bits_to_bytes(&extracted_bits[..m_max]);
        match try_parse_and_decrypt(&frame_bytes, passphrase) {
            Ok(payload) => Some(Ok(payload)),
            Err(StegoError::DecryptionFailed) => Some(Err(StegoError::DecryptionFailed)),
            Err(_) => None,
        }
    };
    #[cfg(feature = "parallel")]
    let hit = {
        use rayon::prelude::*;
        // A decryption failure at one width must not hide a success at
        // another: prefer Ok, then DecryptionFailed.
        let results: Vec<Option<Result<PayloadData, StegoError>>> = ws.par_iter().map(try_w).collect();
        let mut fail = None;
        let mut ok = None;
        for r in results.into_iter().flatten() {
            match r {
                Ok(p) => {
                    ok = Some(Ok(p));
                    break;
                }
                Err(e) => fail = Some(Err(e)),
            }
        }
        ok.or(fail)
    };
    #[cfg(not(feature = "parallel"))]
    let hit = {
        let mut fail = None;
        let mut ok = None;
        for w in &ws {
            match try_w(w) {
                Some(Ok(p)) => {
                    ok = Some(Ok(p));
                    break;
                }
                Some(Err(e)) => fail = Some(Err(e)),
                None => {}
            }
        }
        ok.or(fail)
    };
    progress::advance();
    match hit {
        Some(r) => r,
        None if v3_decrypt_fail => Err(StegoError::DecryptionFailed),
        None => Err(StegoError::FrameCorrupted),
    }
}

/// Length-prefix early-reject for the v1 decoder's width loop.
///
/// Reads the first 16 (and if needed 48) STC syndrome bits and checks the
/// frame's length field against the candidate `m_max` (= `n / w`). For a
/// wrong `w` the partial extract is essentially random, so most wrong
/// candidates are rejected in `O(16 × w)` syndrome XORs instead of the full
/// `O(m_max × w)` of a full extract + frame parse + AES. Inequality, not
/// equality: the v1 encoder clamped `w` at 10, so short messages on large
/// covers have `m < m_max`.
fn w_length_prefix_consistent(
    stego_bits: &[u8],
    hhat_matrix: &[Vec<u32>],
    w: usize,
    m_max: usize,
) -> bool {
    let prefix = extract::stc_extract_prefix(stego_bits, hhat_matrix, w, 16);
    if prefix.len() < 16 {
        return false;
    }
    let prefix_bytes = frame::bits_to_bytes(&prefix);
    let v1_len = u16::from_be_bytes([prefix_bytes[0], prefix_bytes[1]]) as usize;
    if v1_len > 0 && (frame::FRAME_OVERHEAD + v1_len) * 8 <= m_max {
        return true;
    }
    // Extended frame: first u16 == 0 → u32 length at bits 16..48.
    if v1_len == 0 {
        let ext = extract::stc_extract_prefix(stego_bits, hhat_matrix, w, 48);
        if ext.len() >= 48 {
            let ext_bytes = frame::bits_to_bytes(&ext);
            let v2_len = u32::from_be_bytes([ext_bytes[2], ext_bytes[3], ext_bytes[4], ext_bytes[5]]) as usize;
            if v2_len > u16::MAX as usize && (frame::FRAME_OVERHEAD_EXT + v2_len) * 8 <= m_max {
                return true;
            }
        }
    }
    false
}

/// Parse a frame (v1 or v3: same format), decrypt it and decode the payload.
///
/// The size bound is a decoder-side sanity limit, not wire: any frame the
/// payload format allows (`PAYLOAD_MAX_FRAME_BYTES`). Ghost v3 needs it (a
/// 48 MP photo's capacity is ~370 KB), and so do images of the released v1
/// encoder: it never capped the frame at the 256 KB (`MAX_FRAME_BYTES`) its
/// own decoder enforced — a CLI message of ≥ 256 KB on a 12 MP photo, or
/// an app payload near the capacity of a ≥ ~40 MP photo, encoded fine and
/// then could not be decoded by core ≤ 0.4.3.
fn try_parse_and_decrypt(
    frame_bytes: &[u8],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    decrypt_parsed(frame::parse_frame_with_max(frame_bytes, frame::PAYLOAD_MAX_FRAME_BYTES)?, passphrase)
}

fn decrypt_parsed(parsed: frame::ParsedFrame, passphrase: &str) -> Result<PayloadData, StegoError> {
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
pub fn ghost_shadow_decode(
    stego_bytes: &[u8],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    let img = JpegImage::from_bytes(stego_bytes)?;
    ghost_shadow_decode_from_image(&img, passphrase)
}

/// Decode a shadow message from an already-parsed JPEG image: the v3
/// layout, then layout v1's legacy tier, then — only if both found nothing —
/// layout v1's cost pools under the v1 costs
/// ([`compute_positions_streaming_legacy_v1`]). The last step is the only
/// cost computation any decoder still does; it runs on every *failed*
/// shadow decode (a wrong passphrase is the common case).
pub fn ghost_shadow_decode_from_image(
    img: &JpegImage,
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    let current_err = match ghost_shadow_decode_current(img, passphrase) {
        Ok(p) => return Ok(p),
        Err(StegoError::Cancelled) => return Err(StegoError::Cancelled),
        Err(e) => e,
    };
    match ghost_shadow_decode_legacy_v1(img, passphrase) {
        Ok(p) => Ok(p),
        Err(StegoError::FrameCorrupted) => Err(current_err),
        Err(e) => Err(e),
    }
}

/// The cost-free shadow attempts: the v3 layout, then layout v1's legacy
/// tier. For the parallel `smart_decode`, which runs this after the primary
/// attempt and defers [`ghost_shadow_decode_legacy_v1`] until every attempt
/// failed. Memory is bounded by the frames read, not the cover.
pub(crate) fn ghost_shadow_decode_current(
    img: &JpegImage,
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let t = v3::tstart();
    let r = shadow::shadow_extract_v3(img, passphrase);
    v3::tlog("shadow v3 attempt", t);
    match r {
        Ok(p) => return Ok(p),
        Err(StegoError::Cancelled) => return Err(StegoError::Cancelled),
        Err(_) => {}
    }
    shadow::shadow_extract_legacy(img, passphrase)
}

/// Shadow layout v1's cost pools under the v1 costs (decode-only).
pub(crate) fn ghost_shadow_decode_legacy_v1(
    img: &JpegImage,
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?;
    let mut legacy = compute_positions_streaming_legacy_v1(img.dct_grid(0), qt, None)?;
    legacy.sort_by(|a, b| a.cost.total_cmp(&b.cost));
    // v1's 100 % pool is the legacy tier, already read.
    shadow::shadow_extract_cost_pools(img, &legacy, passphrase, &shadow::COST_DEPENDENT_FRACTIONS)
}

/// The Y non-zero AC flat indices in raster order: the v1 primary decoder's
/// set.
pub(crate) fn nzac_indices(grid: &DctGrid) -> Vec<u32> {
    let c = grid.coeffs();
    let mut out = Vec::with_capacity(c.len() / 4);
    for (b, blk) in c.chunks_exact(64).enumerate() {
        for (k, &v) in blk.iter().enumerate().skip(1) {
            if v != 0 {
                out.push((b * 64 + k) as u32);
            }
        }
    }
    out
}

// --- DctGrid flat access helpers ---

/// Read a coefficient from a `DctGrid` using a flat index
/// (`block_index * 64 + row * 8 + col`).
pub(super) fn flat_get(grid: &DctGrid, flat_idx: usize) -> i16 {
    grid.coeffs()[flat_idx]
}
