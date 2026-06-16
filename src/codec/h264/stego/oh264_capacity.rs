// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! OH264 video-stego capacity surface + N-shadow input prep.
//!
//! Pre-encode capacity prediction for the OpenH264 streaming path — the
//! production video-stego encoder. These helpers walk the *OH264*
//! baseline cover (`openh264_stego::h264_walk_cover` /
//! `h264_gop_capacity`); they never touch the pure-Rust
//! `pass1_count_4domain` (retired with the pure-Rust encoder in the
//! video-retirement Phase 3).
//!
//! Relocated here from the former `stego::encode_pixels` so the OH264
//! capacity API + N-shadow validation survive the deletion of the
//! pure-Rust encoder. Gated `h264-encoder`: video stego is
//! OpenH264-only.

use super::keys::CabacStegoMasterKeys;
use crate::stego::error::StegoError;

/// Pure, stateless per-shadow capacity for the N-aware HUD bar.
///
/// Given a cover pool of `pool_bits` injectable bits (the 3-domain
/// `shadow_pool_bits` reported by the capacity probe / one-shot
/// capacity surface) and `n_shadows` total shadow messages sharing it,
/// returns the maximum plaintext bytes available to EACH shadow. The
/// per-shadow ceiling is symmetric (all N shadows share the same
/// budget) and shrinks ~√N as messages are added, per the
/// birthday-paradox collision bound in
/// [`shadow_max_message_bytes_from_cover_bits`].
///
/// This is the single source of truth the mobile bridges call to size
/// each segment of the N-aware shadow capacity bar without re-probing:
/// the probe surfaces `shadow_pool_bits` once, then the UI calls this
/// for the current N. Mirrors
/// [`crate::codec::h264::streaming_session::CapacityProbeResult::shadow_max_message_bytes`]
/// exactly, so the streaming HUD and the closed-form bar agree.
pub fn h264_shadow_capacity_for_n(pool_bits: usize, n_shadows: usize) -> usize {
    super::shadow_capacity::shadow_max_message_bytes_from_cover_bits(pool_bits, n_shadows)
}

/// Primary + shadow capacity for an OH264 video + gop_size combination,
/// returned by [`h264_video_capacity`].
///
/// The shadow value accounts for the `[4, 8, 16, 32, 64, 128]`
/// parity-tier cascade absorption at single-shadow load.
#[derive(Debug, Clone, Copy)]
pub struct H264StegoCapacityInfo {
    /// Total injectable bits across the 3 bypass-bin domains used
    /// by stego: `csb + csl + msb`. Excludes `msl` since
    /// MvdSuffixLsb isn't injectable.
    pub cover_size_bits: usize,
    /// Maximum primary message bytes (text + attached files
    /// combined) embeddable via the 4-domain orchestrator.
    /// Equals `cover_size_bits/8 - FRAME_OVERHEAD`.
    pub primary_max_message_bytes: usize,
    /// Maximum SHADOW message bytes per shadow at single-shadow
    /// (n_shadows = 1) load. For multi-shadow scenarios use
    /// [`h264_shadow_capacity_for_n`] with the actual shadow count.
    pub shadow_max_message_bytes: usize,
    /// Per-tier primary capacity (tunable cascade-safety).
    /// Index = tier 0..4. Tier 0 = no filter = baseline (same as
    /// `primary_max_message_bytes`). Higher tiers = stricter filter =
    /// less capacity but better visual quality. See
    /// `tier_filter::CascadeTier`.
    pub per_tier_primary_max_message_bytes: [usize; 5],
}

/// Resolve the auto-selected cascade tier for an OH264 encode.
///
/// `msg_bytes` should include the framed payload size (text + crypto
/// envelope + chunk header). `headroom` defaults to
/// [`crate::CASCADE_DEFAULT_HEADROOM`].
///
/// Cost: one OH264 baseline encode + walk (same as
/// [`h264_video_capacity`]). Call once per encode.
pub fn h264_resolve_auto_tier(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    opts: super::super::openh264_stego::EncodeOpts,
    msg_bytes: usize,
    headroom: f32,
) -> Result<super::tier_filter::CascadeTier, StegoError> {
    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let cover = super::super::openh264_stego::h264_walk_cover(
        yuv, width, height, n_frames as u32, opts,
    )?;
    let qp_value = opts.qp;
    let csb_qp = vec![qp_value; cover.coeff_sign_bypass.len()];
    let csl_qp = vec![qp_value; cover.coeff_suffix_lsb.len()];
    Ok(super::tier_filter::auto_select_tier(
        &cover, &csb_qp, &csl_qp, msg_bytes, headroom,
    ))
}

/// OH264-accurate 4-domain capacity reporting.
///
/// The OH264 streaming session (CLI, mobile bridges) is the
/// production encode path, so this is the
/// production capacity function. It uses the OH264 baseline walker
/// (`openh264_stego::h264_walk_cover`) so the reported number
/// matches the mode decisions the encoder actually makes.
///
/// `opts.intra_period` should match the encoder's gop_size for an
/// accurate per-GOP estimate; `opts.qp` should match the encoder's QP.
pub fn h264_video_capacity(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    opts: super::super::openh264_stego::EncodeOpts,
    // `false` (live mobile path) computes only tier 0 per GOP, 5×
    // faster; `per_tier_primary_max_message_bytes[1..4]` are filled from
    // tier 0 (capacity-neutral on real content). `true` (CLI
    // diagnostic / calibration) computes the true per-tier breakdown.
    full_tiers: bool,
) -> Result<H264StegoCapacityInfo, StegoError> {
    use crate::stego::frame::FRAME_OVERHEAD;
    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    if yuv.len() != frame_size * n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {}", yuv.len(), frame_size * n_frames
        )));
    }

    // Accurate per-GOP capacity. The streaming encoder splits the
    // message into one chunk PER GOP and embeds each independently, so a
    // GOP holds at most its own STC budget. `h264_gop_capacity`
    // rebuilds each GOP's real cover (safe_msl + tier ∞-costs + Tier-3
    // costs) and STC-trials the largest embeddable chunk payload —
    // reproducing the encode's `MessageTooLarge` boundary by construction,
    // replacing the legacy CoeffSign × 0.40 heuristic that over-reported
    // the true encodable capacity by 8–350×.
    //
    // The carry/proportional encoder fills each GOP to its OWN cap, so the
    // message ceiling is the SUM of per-GOP caps (Σ), not the even-split
    // `n_gops × min` (which low-balled on heterogeneous content — the
    // worst GOP dragged the min down, e.g. woman_subway 0.16×). Σ matches
    // the streaming `CapacityProbeResult::primary_max_message_bytes` the
    // encode HUD shows. UPPER bound: the per-GOP STC cap over-reports ~2%
    // on J2 jitter, absorbed by the encoder's per-GOP verify + shrink-carry
    // (graceful `MessageTooLarge`, never data loss).
    let gop_size = (opts.intra_period.max(1)) as usize;
    let n_gops = n_frames.div_ceil(gop_size);
    let weights = super::cost_weights::CostWeights::default();
    let mut per_tier_sum = [0usize; 5];
    let mut cover_size_bits = 0usize;
    // Accumulate the 3-domain shadow pool straight off the OH264
    // cover walk we're already doing per GOP, so the shadow number comes
    // from the SAME encoder the shadow encode uses — no separate pure-Rust
    // `pass1_count_4domain` whole-video pass (the ~45 s/clip bottleneck the
    // probe profiling pinned).
    let mut shadow_pool_bits = 0usize;
    for g in 0..n_gops {
        let f0 = g * gop_size;
        let f1 = ((g + 1) * gop_size).min(n_frames);
        let gop_n = (f1 - f0) as u32;
        let gop_yuv = &yuv[f0 * frame_size..f1 * frame_size];
        let cap = super::super::openh264_stego::h264_gop_capacity(
            gop_yuv, width, height, gop_n, opts, &weights, full_tiers,
        )?;
        cover_size_bits += cap.coeff_sign_cover_bits;
        shadow_pool_bits += cap.injectable_cover_bits;
        for t in 0..5 {
            per_tier_sum[t] += cap.per_tier_payload[t];
        }
    }

    // Σ message capacity per tier = (Σ per-GOP payload) − crypto-frame
    // envelope. Per-GOP payload is already post-chunk-header, so the only
    // subtraction is the one-time envelope. Mirrors
    // `CapacityProbeResult::primary_max_message_bytes_at_tier`.
    let mut per_tier_primary_max_message_bytes = [0usize; 5];
    for t in 0..5 {
        per_tier_primary_max_message_bytes[t] =
            per_tier_sum[t].saturating_sub(FRAME_OVERHEAD);
    }
    let primary_max_message_bytes = per_tier_primary_max_message_bytes[0];

    // Shadow capacity from the OH264 pool we accumulated above
    // (collision-limited √-formula), NOT a separate pure-Rust whole-video
    // `pass1_count_4domain`. This is both faster (drops the dominant probe
    // cost) and more correct: the number now reflects the OH264 encoder the
    // shadow encode actually runs, instead of the divergent pure-Rust cover.
    let shadow_max_message_bytes =
        super::shadow_capacity::shadow_max_message_bytes_from_cover_bits(shadow_pool_bits, /* n_shadows */ 1);

    Ok(H264StegoCapacityInfo {
        cover_size_bits,
        primary_max_message_bytes,
        shadow_max_message_bytes,
        per_tier_primary_max_message_bytes,
    })
}

/// Result of [`validate_n_shadow_inputs`]. Carries the
/// derived constants the outer encode flow needs after validation
/// succeeds.
pub(crate) struct NShadowSetup {
    pub(crate) frame_size: usize,
    pub(crate) gop_size: usize,
    pub(crate) b_count: usize,
}

/// Validate `n-shadow` encode inputs.
///
/// Performs the input checks that must succeed before any encode
/// work starts:
/// - 16-pixel dimension alignment
/// - YUV byte length consistency with `n_frames`
/// - `gop_size` in `1..=n_frames`
/// - Unique passphrases across primary + every shadow
///
/// Returns the derived `(frame_size, gop_size, b_count)` triple so
/// the caller doesn't recompute them.
///
/// Capacity is NOT pre-checked here — the UI calls
/// [`h264_shadow_capacity_for_n`] up front, and the OH264 shadow
/// orchestrator's own cascade + `MessageTooLarge` boundary enforce the
/// hard limit at encode time. (The former belt-and-suspenders guard ran
/// the retired pure-Rust `pass1_count_4domain`; it was dropped with the
/// pure-Rust encoder.)
pub(crate) fn validate_n_shadow_inputs<'a>(
    // WV.6.b.3: the YUV is no longer materialised whole (it lives in a
    // `YuvSpill`), so this takes the byte length instead of the buffer — the
    // only thing it ever read from `yuv` was `yuv.len()` for the size check.
    yuv_len: usize,
    width: u32,
    height: u32,
    n_frames: usize,
    pattern: super::gop_pattern::GopPattern,
    primary_passphrase: &str,
    shadows: &'a [crate::stego::shadow_layer::ShadowLayer<'a>],
) -> Result<NShadowSetup, StegoError> {
    let gop_size = pattern.gop_size();
    let b_count = pattern.legacy_b_count();

    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    if yuv_len != frame_size * n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {}", yuv_len, frame_size * n_frames
        )));
    }
    if gop_size == 0 || gop_size > n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "gop_size {gop_size} must be in 1..={n_frames}"
        )));
    }

    // Reject duplicate passphrases across primary + shadows.
    let mut all_passphrases: Vec<&str> = Vec::with_capacity(shadows.len() + 1);
    all_passphrases.push(primary_passphrase);
    for s in shadows {
        all_passphrases.push(s.passphrase);
    }
    for i in 0..all_passphrases.len() {
        for j in (i + 1)..all_passphrases.len() {
            if all_passphrases[i] == all_passphrases[j] {
                return Err(StegoError::DuplicatePassphrase);
            }
        }
    }

    Ok(NShadowSetup { frame_size, gop_size, b_count })
}

/// Result of [`prep_n_shadow_primary_payload`]. Holds the
/// framed primary payload as a bit vector plus the derived stego
/// master keys.
pub(crate) struct NShadowPrimary {
    pub(crate) frame_bits: Vec<u8>,
    /// The `build_frame` output bytes (`frame_bits` is just this expanded
    /// MSB-first). WV.6.a chunks THESE into per-GOP `chunk_frame` v3
    /// payloads; `total_bytes == frame_bytes.len()`.
    pub(crate) frame_bytes: Vec<u8>,
    pub(crate) keys: CabacStegoMasterKeys,
    pub(crate) m_total: usize,
}

/// Frame + encrypt the primary payload, derive master
/// keys, expand to a bit vector ready for STC.
pub(crate) fn prep_n_shadow_primary_payload(
    primary_message: &str,
    primary_files: &[crate::stego::payload::FileEntry],
    primary_passphrase: &str,
) -> Result<NShadowPrimary, StegoError> {
    use crate::stego::{crypto, frame, payload};
    let primary_bytes = payload::encode_payload(primary_message, primary_files)?;
    let (ct, nonce, salt) = crypto::encrypt(&primary_bytes, primary_passphrase)?;
    let frame_bytes = frame::build_frame(primary_bytes.len(), &salt, &nonce, &ct);
    let frame_bits: Vec<u8> = frame_bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect();
    let keys = CabacStegoMasterKeys::derive(primary_passphrase)?;
    let m_total = frame_bits.len();
    Ok(NShadowPrimary { frame_bits, frame_bytes, keys, m_total })
}
