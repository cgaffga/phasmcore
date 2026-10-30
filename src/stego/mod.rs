// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Steganographic encoding and decoding pipelines.
//!
//! This module provides two embedding modes:
//!
//! - **Ghost** (`ghost_encode` / `ghost_decode`): Stealth mode using J-UNIWARD
//!   cost function and Syndrome-Trellis Coding (STC) to minimize statistical
//!   detectability. Best for images that will not be recompressed.
//!
//! - **Armor** (`armor_encode` / `armor_decode`): Robustness mode using STDM
//!   (Spread Transform Dither Modulation) with Reed-Solomon error correction to
//!   survive JPEG recompression. Trades capacity for survivability.
//!
//! Both modes share the same payload frame format, encryption (AES-256-GCM-SIV),
//! and key derivation (Argon2id two-tier). The `smart_decode` function
//! auto-detects which mode was used.

// --- Media-agnostic modules (shared by image and video pipelines) ---
pub mod error;
pub mod stc;
pub mod crypto;
#[doc(hidden)]
pub mod chacha_stream;
pub mod frame;
pub mod chunk_frame;
pub mod gop_cache;
pub mod calibration;
pub mod balanced_allocation;
pub mod permute;
pub mod cover_order;
pub mod payload;
pub mod progress;
pub mod shadow_layer;
pub mod memory;

// --- Steganographic algorithms ---
pub mod cost;
pub(crate) mod ghost;
pub mod armor;
#[cfg(feature = "video")]
pub mod video;

pub use error::StegoError;
pub use ghost::quality;
pub use ghost::quality::EncodeQuality;
pub use memory::{
    get_memory_budget, ghost_lane_plan, predict_peak_memory,
    set_memory_budget, set_telemetry_hook,
    GhostLanePlan, GhostRung, ModeId, TelemetryEvent, TelemetryHook,
};
pub use ghost::optimizer::{optimize_cover, OptimizerConfig, OptimizerMode};
#[doc(hidden)]
pub use ghost::optimizer::optimizer_test_hash_hex;

// Backward-compatible re-exports at original paths
pub use ghost::capacity;
pub use ghost::side_info;
pub use ghost::shadow;
pub use ghost::optimizer;

/// Maximum pixel dimension (width or height) for encode.
/// Images exceeding this are downsampled by the frontend before reaching Rust.
pub const MAX_DIMENSION: u32 = 16384;

/// Maximum total pixel count for encode (width × height).
/// 200 MP covers all current cameras including flagship 200 MP sensors.
/// Memory-optimized: strip-based UNIWARD (~170 MB/strip), compact positions
/// (8 bytes each), segmented STC Viterbi. Total ~1 GB for 200 MP.
pub const MAX_PIXELS: u32 = 200_000_000;

/// Minimum pixel dimension (width or height) for encode.
/// Images below this are rejected with an error message.
pub const MIN_ENCODE_DIMENSION: u32 = 200;

/// Target pixel dimension (longest side) for Armor/Fortress pre-resize.
/// Images larger than this are downsampled by the frontend before encoding
/// in Armor mode, so that the 8×8 block grid survives platform recompression
/// (e.g. WhatsApp resizes to ~1600px on the longest side).
pub const ARMOR_TARGET_DIMENSION: u32 = 1600;

/// Validate image dimensions for encoding.
///
/// Returns `Ok(())` if the dimensions are within acceptable bounds.
/// Called at the start of both `ghost_encode` and `armor_encode`.
///
/// # Errors
/// - [`StegoError::ImageTooSmall`] if either dimension < 200px.
/// - [`StegoError::ImageTooLarge`] if either dimension > 8192px or total pixels > 16M.
pub fn validate_encode_dimensions(width: u32, height: u32) -> Result<(), StegoError> {
    if width < MIN_ENCODE_DIMENSION || height < MIN_ENCODE_DIMENSION {
        return Err(StegoError::ImageTooSmall);
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION || width.checked_mul(height).is_none_or(|p| p > MAX_PIXELS) {
        return Err(StegoError::ImageTooLarge);
    }
    Ok(())
}
pub use ghost::session::{GhostEncodeSession, GhostTaskRunner, TASK_TARGET_OFFSET as GHOST_TASK_TARGET_OFFSET};
#[doc(hidden)]
pub use ghost::session::run_split_encode_locally;
pub use ghost::pipeline::{ghost_encode, ghost_decode, ghost_encode_with_files, ghost_encode_si, ghost_encode_si_with_files, GHOST_DECODE_STEPS, GHOST_ENCODE_STEPS};
pub use ghost::pipeline::{ghost_encode_with_quality, ghost_encode_with_files_quality, ghost_encode_si_with_quality, ghost_encode_si_with_files_quality};
pub use ghost::pipeline::{ghost_encode_with_shadows, ghost_encode_si_with_shadows, ghost_shadow_decode, ShadowLayer, GHOST_ENCODE_WITH_SHADOWS_STEPS};
pub use ghost::pipeline::{ghost_encode_with_shadows_quality, ghost_encode_si_with_shadows_quality};
pub use shadow::shadow_capacity;
pub use capacity::{estimate_shadow_capacity, estimate_shadow_capacity_si};
pub use capacity::estimate_capacity as ghost_capacity;
pub use capacity::estimate_capacity_si as ghost_capacity_si;
pub use ghost::deep_cover::{deep_cover_jpeg, ghost_encode_deep_cover, ghost_encode_deep_cover_with_shadows, ghost_capacity_deep_cover, DEEP_COVER_DEFAULT_QUALITY};
pub use capacity::estimate_capacity_with_shadows as ghost_capacity_with_shadows;
pub use capacity::{ghost_bands, ghost_bands_si, GhostBands};
pub use armor::pipeline::{armor_encode, armor_encode_with_quality, armor_decode, DecodeQuality, ArmorCapacityInfo, armor_capacity_info};
pub use armor::capacity::estimate_armor_capacity as armor_capacity;
pub use payload::{PayloadData, FileEntry, compressed_payload_size, safe_filename};

#[cfg(test)]
mod dimension_tests {
    use super::*;

    #[test]
    fn valid_dimensions() {
        assert!(validate_encode_dimensions(800, 600).is_ok());
        assert!(validate_encode_dimensions(3000, 4000).is_ok());
    }

    #[test]
    fn boundary_min() {
        assert!(validate_encode_dimensions(200, 200).is_ok());
        assert!(validate_encode_dimensions(199, 200).is_err());
        assert!(validate_encode_dimensions(200, 199).is_err());
    }

    #[test]
    fn boundary_max_dimension() {
        assert!(validate_encode_dimensions(16384, 1000).is_ok());
        assert!(validate_encode_dimensions(1000, 16384).is_ok());
        assert!(validate_encode_dimensions(16385, 1000).is_err());
        assert!(validate_encode_dimensions(1000, 16385).is_err());
    }

    #[test]
    fn too_many_pixels() {
        // 14143 * 14143 = 200_024_449 > 200M
        assert!(validate_encode_dimensions(14143, 14143).is_err());
        // 14142 * 14142 = 199_996_164 < 200M — OK
        assert!(validate_encode_dimensions(14142, 14142).is_ok());
    }

    #[test]
    fn error_variants() {
        match validate_encode_dimensions(100, 300) {
            Err(StegoError::ImageTooSmall) => {}
            other => panic!("expected ImageTooSmall, got {other:?}"),
        }
        match validate_encode_dimensions(16385, 1000) {
            Err(StegoError::ImageTooLarge) => {}
            other => panic!("expected ImageTooLarge, got {other:?}"),
        }
    }
}

/// Unified decode: auto-detects Ghost or Armor mode from the embedded frame.
///
/// Tries Ghost first, then Armor. Returns the decoded payload and quality info.
///
/// When the `parallel` feature is enabled, Ghost and Armor decodes run
/// concurrently via `rayon::join`, roughly halving decode latency on
/// multi-core devices.
pub fn smart_decode(stego_bytes: &[u8], passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let result = smart_decode_inner(stego_bytes, passphrase);
    progress::finish();
    result
}

/// Try Fortress, then Armor STDM + geometric recovery.
///
/// Shared by the serial and parallel `smart_decode_inner` so both order the
/// Armor attempts identically. Returns `Some` on a successful decode.
///
/// Sets `saw_decryption_failed` when an attempt found a well-formed payload but
/// the passphrase was wrong, so the caller reports `DecryptionFailed` rather
/// than the misleading `FrameCorrupted`.
fn try_armor_modes(
    img: &crate::codec::jpeg::JpegImage,
    passphrase: &str,
    saw_decryption_failed: &mut bool,
) -> Option<(PayloadData, DecodeQuality)> {
    try_armor_modes_unless(img, passphrase, saw_decryption_failed, &|| false).expect("never stopped")
}

/// [`try_armor_modes`], abandoned (`None`) when `stop()` is true before an
/// attempt starts (before Fortress, before STDM, before geometric
/// recovery). The parallel `smart_decode` runs it next to the v1 cost
/// pools, which it prefers, and stops it when they succeed; a running
/// attempt finishes first (its result is then ignored).
fn try_armor_modes_unless(
    img: &crate::codec::jpeg::JpegImage,
    passphrase: &str,
    saw_decryption_failed: &mut bool,
    stop: &dyn Fn() -> bool,
) -> Option<Option<(PayloadData, DecodeQuality)>> {
    use crate::stego::armor::fortress;
    use crate::stego::armor::pipeline::armor_decode_no_fortress_unless;

    if stop() {
        return None;
    }
    // Fortress first (fast).
    if img.num_components() > 0
        && let Ok(result) = fortress::fortress_decode(img, passphrase)
    {
        return Some(Some(result));
    }

    // Armor STDM + geometric recovery. Both read this parse; the geometric
    // stage streams its spectrum and corrected image with a bounded working
    // set instead of re-parsing or copying the image.
    match armor_decode_no_fortress_unless(img, passphrase, stop)? {
        Ok(result) => Some(Some(result)),
        Err(StegoError::DecryptionFailed) => {
            *saw_decryption_failed = true;
            Some(None)
        }
        Err(_) => Some(None),
    }
}

/// Serial smart_decode implementation (default path and WASM).
///
/// Tries Armor first (default mode, most common), then Ghost, then
/// Ghost shadow. Parses the JPEG ONCE at the top and threads
/// `&JpegImage` into all four mode attempts via the `_from_image`
/// variants. Otherwise each mode would re-parse the same bytes — 2-3×
/// wasted work on the failure path (Armor → Ghost → Ghost shadow),
/// each parse 100s of ms on large iPhone JPEGs.
///
/// Progress steps: 1 (fortress) + ~21 (phase1) + ~21 (phase2) +
///   1 (phase3) + GHOST_DECODE_STEPS (102). Actual total set by
/// try_armor_decode once candidate count is known.
#[cfg(not(feature = "parallel"))]
fn smart_decode_inner(stego_bytes: &[u8], passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    use crate::codec::jpeg::JpegImage;

    progress::init(0); // reset; try_armor_decode sets real total
    progress::check_cancelled()?;

    // Parse JPEG once and share across all mode attempts.
    let img = JpegImage::from_bytes(stego_bytes)?;
    let mut saw_decryption_failed = false;

    // Armor/Fortress is *designed* around a cover no larger than
    // ARMOR_TARGET_DIMENSION, so on a bigger input it is the less likely mode —
    // but it is NOT impossible. Nothing actually enforces that pre-resize, and
    // full-resolution Armor stego exists in the wild (Armor is the default
    // mode, so any phone photo encoded without resizing lands here).
    //
    // So order the attempts by likelihood rather than skipping: large inputs try
    // Ghost first and fall back to Armor/Fortress at the end. That keeps the
    // expensive Armor attempt (STDM sweep + geometric recovery: FFT, resample;
    // bounded working set, but seconds of work at 48 MP) off the common path
    // without ever making a real payload undecodable.
    //
    // Regression: an earlier version *skipped* Armor above this size, which
    // silently broke decoding of every full-resolution default-mode encode.
    let fi = img.frame_info();
    let max_dim = fi.width.max(fi.height) as u32;
    let armor_first = max_dim <= ARMOR_TARGET_DIMENSION;

    if armor_first
        && let Some(result) =
            try_armor_modes(&img, passphrase, &mut saw_decryption_failed)
    {
        return Ok(result);
    }

    // Try Ghost — extend progress total instead of resetting to avoid the
    // bar jumping back to 0%. The bar continues smoothly from the Armor
    // phase into the Ghost phase.
    let (armor_done, _) = progress::get();
    progress::set_total(armor_done + GHOST_DECODE_STEPS);

    // Ghost primary (v3, then v1), then the shadow layouts (v3, layout v1).
    // No decoder computes costs, except the v1 cost-pool fallback inside the
    // shadow attempt, which runs only after everything else in it failed.
    match ghost::pipeline::ghost_decode_from_image(&img, passphrase) {
        Ok(payload) => return Ok((payload, DecodeQuality::ghost())),
        Err(StegoError::DecryptionFailed) => { saw_decryption_failed = true; }
        Err(_) => {}
    }
    match ghost::pipeline::ghost_shadow_decode_from_image(&img, passphrase) {
        Ok(payload) => return Ok((payload, DecodeQuality::ghost())),
        Err(StegoError::DecryptionFailed) => { saw_decryption_failed = true; }
        Err(_) => {}
    }

    // Deferred Armor/Fortress for large inputs: Ghost didn't match, so give the
    // unlikely-but-possible full-resolution Armor case its turn.
    if !armor_first
        && let Some(result) =
            try_armor_modes(&img, passphrase, &mut saw_decryption_failed)
    {
        return Ok(result);
    }

    if saw_decryption_failed {
        Err(StegoError::DecryptionFailed)
    } else {
        Err(StegoError::FrameCorrupted)
    }
}

/// Parallel smart_decode: the serial path's attempts and preference order,
/// with independent attempts run side by side.
///
/// Every attempt computes exactly what it computes alone (results depend
/// only on the image and the passphrase), and the results are read in the
/// fixed preference order below, so the returned payload, mode and error
/// are those of the attempts run one after another. An attempt runs ahead
/// only next to attempts it ranks *below* or that start at the same time
/// anyway; once a preferred attempt succeeds, the lower ones are abandoned
/// at their next check (they are joined, so the call returns when they
/// reach it — the uninterruptible pieces are one Fortress / STDM /
/// geometric / Ghost attempt). The keys are shared across threads
/// ([`crypto::KeyScope`]): no stage derives a key another already derived.
///
/// - Inputs ≤ [`ARMOR_TARGET_DIMENSION`]: Fortress, Armor STDM (+ geometric
///   recovery) and Ghost (primary, then — only if it found nothing — the
///   cost-free shadow layouts) run concurrently; preference Fortress >
///   Armor STDM > Ghost primary > Ghost shadow. Fortress succeeding skips
///   geometric recovery and the Ghost attempts not yet started; STDM
///   succeeding skips the Ghost attempts not yet started. The v1 cost-pool
///   shadow fallback runs last, alone.
/// - Larger inputs (Armor deferred, see the serial path): Ghost primary
///   alone, then the shadow layouts alone (a Ghost passphrase never shares
///   the cores with speculative work), then the v1 cost pools and the
///   Armor attempts — side by side if [`memory::decode_overlap_ok`] allows
///   it (Armor stopped when the pools succeed), else one after another.
///   The Armor chain's Fortress, STDM and template keys are derived in
///   parallel when it starts.
#[cfg(feature = "parallel")]
fn smart_decode_inner(stego_bytes: &[u8], passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    use crate::codec::jpeg::JpegImage;

    // In parallel mode the attempts advance the same global counter
    // concurrently.  We init with 0 (indeterminate) — try_armor_decode will
    // set a real total once it knows the candidate count.  The cap in
    // advance() prevents step from ever exceeding total.
    progress::init(0);
    progress::check_cancelled()?;

    let img = JpegImage::from_bytes(stego_bytes)?;
    let keys = crypto::KeyScope::new(passphrase);

    // See the serial path for the full rationale: Armor/Fortress is the less
    // likely mode above ARMOR_TARGET_DIMENSION but NOT impossible, so large
    // inputs *defer* the Armor attempts rather than skipping them. Deferring
    // also keeps the Armor working set from overlapping Ghost's primary.
    let fi = img.frame_info();
    let max_dim = fi.width.max(fi.height) as u32;
    if max_dim <= ARMOR_TARGET_DIMENSION {
        decode_armor_first(&img, passphrase)
    } else {
        decode_armor_deferred(&img, passphrase, &keys)
    }
}

/// [`smart_decode_inner`] (parallel) for inputs ≤ [`ARMOR_TARGET_DIMENSION`].
#[cfg(feature = "parallel")]
fn decode_armor_first(
    img: &crate::codec::jpeg::JpegImage,
    passphrase: &str,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    use crate::stego::armor::fortress;
    use crate::stego::armor::pipeline::armor_decode_no_fortress_unless;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Preference ranks; `best` = the best rank that succeeded so far. An
    // attempt is skipped (`None`) once one it ranks below succeeded.
    const FORTRESS: usize = 0;
    const STDM: usize = 1;
    const GHOST: usize = 2;
    const SHADOW: usize = 3;
    let best = AtomicUsize::new(usize::MAX);
    let beaten = |rank: usize| best.load(Ordering::Relaxed) < rank;

    let (fortress_result, (stdm_result, (ghost_result, shadow_result))) = rayon::join(
        || {
            let r = if img.num_components() > 0 {
                fortress::fortress_decode(img, passphrase)
            } else {
                Err(StegoError::FrameCorrupted)
            };
            if r.is_ok() {
                best.fetch_min(FORTRESS, Ordering::Relaxed);
            }
            r
        },
        || rayon::join(
            || {
                let r = armor_decode_no_fortress_unless(img, passphrase, &|| beaten(STDM));
                if matches!(r, Some(Ok(_))) {
                    best.fetch_min(STDM, Ordering::Relaxed);
                }
                r
            },
            || {
                if beaten(GHOST) {
                    return (None, None);
                }
                // Primary, then — only if it found nothing — the cost-free
                // shadow layouts (v3, layout v1 legacy tier). The v1
                // cost-pool retry waits until every attempt failed (below).
                let primary = ghost::pipeline::ghost_decode_from_image(img, passphrase);
                if primary.is_ok() || matches!(primary, Err(StegoError::Cancelled)) {
                    return (Some(primary), Some(Err(StegoError::FrameCorrupted)));
                }
                if beaten(SHADOW) {
                    return (Some(primary), None);
                }
                let shadow = ghost::pipeline::ghost_shadow_decode_current(img, passphrase);
                (Some(primary), Some(shadow))
            },
        ),
    );

    // An attempt is `None` only if one it ranks below succeeded — and that
    // one returns before the `None` is read.
    let stdm_result = stdm_result.unwrap_or(Err(StegoError::FrameCorrupted));
    let ghost_result = ghost_result.unwrap_or(Err(StegoError::FrameCorrupted));
    let shadow_result = shadow_result.unwrap_or(Err(StegoError::FrameCorrupted));

    // Prefer Fortress (fastest, most robust).
    if let Ok((payload, quality)) = fortress_result {
        return Ok((payload, quality));
    }

    // Try Armor STDM + geometric recovery.
    if let Ok((payload, quality)) = stdm_result {
        return Ok((payload, quality));
    }

    // Try Ghost.
    if let Ok(payload) = ghost_result {
        return Ok((payload, DecodeQuality::ghost()));
    }

    // Try Ghost shadow.
    if let Ok(payload) = shadow_result {
        return Ok((payload, DecodeQuality::ghost()));
    }

    // All failed — determine the best error to report.
    let mut saw_decryption_failed = matches!(&fortress_result, Err(StegoError::DecryptionFailed))
        || matches!(&stdm_result, Err(StegoError::DecryptionFailed))
        || matches!(&ghost_result, Err(StegoError::DecryptionFailed))
        || matches!(&shadow_result, Err(StegoError::DecryptionFailed));

    // Ghost shadow under the legacy (phasm-core ≤ 0.4.x) costs: a shadow
    // of an older image whose cost pool only those costs select. Run only
    // now, once nothing else matched, so a successful decode never pays
    // for the recompute.
    if img.num_components() > 0 && !matches!(&shadow_result, Err(StegoError::Cancelled)) {
        match ghost::pipeline::ghost_shadow_decode_legacy_v1(img, passphrase) {
            Ok(payload) => return Ok((payload, DecodeQuality::ghost())),
            Err(StegoError::DecryptionFailed) => saw_decryption_failed = true,
            Err(StegoError::Cancelled) => return Err(StegoError::Cancelled),
            Err(_) => {}
        }
    }

    if saw_decryption_failed {
        return Err(StegoError::DecryptionFailed);
    }

    Err(stdm_result.unwrap_err())
}

/// [`smart_decode_inner`] (parallel) for inputs above
/// [`ARMOR_TARGET_DIMENSION`]: Armor deferred to the end. Preference: Ghost
/// primary > Ghost shadow (v3, legacy tier) > v1 cost pools > Fortress >
/// Armor STDM (+ geometric) — the serial order.
#[cfg(feature = "parallel")]
fn decode_armor_deferred(
    img: &crate::codec::jpeg::JpegImage,
    passphrase: &str,
    keys: &crypto::KeyScope,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    use crypto::DecodeKey;
    use std::sync::atomic::{AtomicBool, Ordering};

    // Keys the Armor chain derives one after another once it runs (Fortress,
    // then STDM, then — if STDM failed — geometric recovery's template).
    // The ring key is left to geometric recovery: a wrong passphrase rarely
    // gets that far.
    const ARMOR_KEYS: [DecodeKey; 3] = [DecodeKey::Fortress, DecodeKey::Armor, DecodeKey::Template];

    // Ghost primary, alone: a primary passphrase never shares the cores.
    let ghost_result = ghost::pipeline::ghost_decode_from_image(img, passphrase);
    if let Ok(payload) = ghost_result {
        return Ok((payload, DecodeQuality::ghost()));
    }

    // The shadow layouts (v3, layout v1's legacy tier), alone — skipped (as
    // the dummy `FrameCorrupted`) when the primary was cancelled.
    let shadow_result = if matches!(ghost_result, Err(StegoError::Cancelled)) {
        Err(StegoError::FrameCorrupted)
    } else {
        ghost::pipeline::ghost_shadow_decode_current(img, passphrase)
    };
    if let Ok(payload) = shadow_result {
        return Ok((payload, DecodeQuality::ghost()));
    }

    let mut saw_decryption_failed = matches!(&ghost_result, Err(StegoError::DecryptionFailed))
        || matches!(&shadow_result, Err(StegoError::DecryptionFailed));

    // Ghost shadow under the legacy (phasm-core ≤ 0.4.x) costs, then the
    // deferred Armor/Fortress (full-resolution Armor must stay decodable).
    // Side by side when both fit (`memory::decode_overlap_ok`): the Armor
    // chain, which ranks below the cost pools, stops at its next attempt
    // once they succeed, and its result is read only if they failed.
    if img.num_components() > 0 && !matches!(&shadow_result, Err(StegoError::Cancelled)) {
        let grid = img.dct_grid(0);
        let heap = memory::DecodeStageHeap::for_cover(grid.coeffs().len(), ghost::capacity::count_nonzero_ac(grid));
        if memory::decode_overlap_ok(heap.legacy_order.max(heap.legacy_pools), heap.armor) {
            keys.prederive(&ARMOR_KEYS);
            let pools_won = AtomicBool::new(false);
            let (legacy_result, armor) = rayon::join(
                || {
                    let r = ghost::pipeline::ghost_shadow_decode_legacy_v1(img, passphrase);
                    if r.is_ok() {
                        pools_won.store(true, Ordering::Relaxed);
                    }
                    r
                },
                || {
                    let mut saw = false;
                    try_armor_modes_unless(img, passphrase, &mut saw, &|| pools_won.load(Ordering::Relaxed))
                        .map(|hit| (hit, saw))
                },
            );
            match legacy_result {
                Ok(payload) => return Ok((payload, DecodeQuality::ghost())),
                Err(StegoError::DecryptionFailed) => saw_decryption_failed = true,
                Err(StegoError::Cancelled) => return Err(StegoError::Cancelled),
                Err(_) => {}
            }
            // Stopped only when the pools succeeded, which returned above.
            let (hit, saw) = armor.unwrap_or((None, false));
            if let Some(result) = hit {
                return Ok(result);
            }
            saw_decryption_failed |= saw;
            return if saw_decryption_failed { Err(StegoError::DecryptionFailed) } else { Err(StegoError::FrameCorrupted) };
        }
        match ghost::pipeline::ghost_shadow_decode_legacy_v1(img, passphrase) {
            Ok(payload) => return Ok((payload, DecodeQuality::ghost())),
            Err(StegoError::DecryptionFailed) => saw_decryption_failed = true,
            Err(StegoError::Cancelled) => return Err(StegoError::Cancelled),
            Err(_) => {}
        }
    }

    keys.prederive(&ARMOR_KEYS);
    if let Some(result) = try_armor_modes(img, passphrase, &mut saw_decryption_failed) {
        return Ok(result);
    }

    if saw_decryption_failed {
        Err(StegoError::DecryptionFailed)
    } else {
        Err(StegoError::FrameCorrupted)
    }
}
