// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! CAP2 §17.1 — v3 tiered **sampled** capacity estimator for OH264 video.
//!
//! Revives the ratified Phase 6D.0 three-tier estimator (`_quick` /
//! `_estimate` / `_exact`) that the CAP2.1 pivot dropped, now on the OH264
//! per-GOP STC-trial primitive.
//!
//! ## The reframe (§17.0)
//!
//! **Exactness is an ENCODE-TIME invariant, not a display one.** The encode
//! enforces the true limit gracefully (`MessageTooLarge`, guaranteed by the
//! cascade audit §A.1.11 — never silent loss). The *displayed* capacity is an
//! honest **sampled estimate**: an over-estimate degrades to a graceful
//! "shorten it", an under-estimate wastes a little headroom; neither loses
//! data. So a display margin is a UX knob, not the §15-banned correctness
//! band-aid.
//!
//! ## Why sampled (§17.1)
//!
//! The full per-GOP probe is ~1.1 s/GOP after #809 — minutes for a long clip,
//! and the HUD shows capacity *after video-select, before the passphrase*, so
//! it must be fast + refining. We **stratify-sample** a capped number of GOPs
//! (default 16) so cost is **independent of clip length**, and extrapolate.
//!
//! ## Aggregation — Σ sample-mean, not `n_gops × min`
//!
//! `min` both wastes capacity (the CAP2.2 carry allocator beats it) AND is the
//! worst statistic to sample (outlier-dominated — a sample's `min` is a biased
//! over-estimate of the true `min`). The **Σ carry total** is the real ceiling
//! and the **sample mean is an unbiased estimator** with error ~1/√K. Reporting
//! Σ presumes the encoder *achieves* Σ → that's CAP2.2/#804 (proportional GOP
//! allocation); until #804 lands the encoder even-splits, so this estimate
//! over-claims slightly vs. what the encoder delivers — **graceful** per the
//! reframe, and this module stays internal until #807 wires the HUD (after
//! #804), so the window never reaches users.
//!
//! ## Margin — adaptive confidence bound
//!
//! A one-sided lower bound on the mean from the sample's *own* spread + size
//! (`Z·s/√K`), shrinking as it refines. Per-instance — NOT a corpus-fit
//! constant (it multiplies the *measured* standard error).
//!
//! ## Memory
//!
//! Provider-based: the caller materializes only the *sampled* GOPs' YUV (one
//! at a time), so peak memory is one GOP — the bridge decodes just those GOPs
//! (#807). The `_yuv` convenience wrapper slices a full in-memory clip for
//! CLI / tests.

#![cfg(feature = "openh264-backend")]

use super::super::openh264_stego::{oh264_gop_capacity_per_tier, EncodeOpts};
use crate::stego::error::StegoError;
use crate::stego::frame::FRAME_OVERHEAD;

/// Default cap on sampled GOPs. Keeps `_estimate` cost independent of clip
/// length while giving the sample mean a tight enough standard error
/// (~1/√16 = 25% of the per-GOP CV before the margin even applies).
pub const DEFAULT_MAX_SAMPLE_GOPS: usize = 16;

/// One-sided 90th-percentile normal factor for the lower confidence bound on
/// the mean. NOT a corpus constant: it scales the sample's *measured* standard
/// error, so the resulting margin is per-instance and shrinks as K grows.
const Z_90: f64 = 1.2816;

/// A refining capacity estimate, streamed to the HUD as the sample grows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapacityEstimate {
    /// Displayed primary capacity (bytes): Σ sample-mean carry total, minus the
    /// adaptive confidence margin, minus the crypto-frame envelope.
    pub bytes_estimate: usize,
    /// 3-domain injectable shadow pool **C** (bits) for the whole clip,
    /// extrapolated from the sample. Feeds the closed-form shadow segmented
    /// bar (§17.3) — `per_shadow_cap(C, N)` is pure arithmetic on this.
    pub pool_c_bits: usize,
    /// Sampled per-GOP mean primary payload (bytes, pre-overhead, pre-margin) —
    /// the raw Σ basis.
    pub gop_mean_payload: usize,
    /// GOPs probed so far.
    pub gops_scanned: usize,
    /// Total GOPs in the clip (the extrapolation target).
    pub gops_total: usize,
    /// One-sided relative margin actually applied (0.0..=1.0), shrinking as the
    /// sample grows; 0 once the whole clip is scanned (no extrapolation error).
    pub margin_frac: f32,
    /// True once no further refinement will arrive (sample exhausted, whole
    /// clip scanned, or cancelled).
    pub is_final: bool,
}

impl CapacityEstimate {
    fn empty(gops_total: usize, is_final: bool) -> Self {
        CapacityEstimate {
            bytes_estimate: 0,
            pool_c_bits: 0,
            gop_mean_payload: 0,
            gops_scanned: 0,
            gops_total,
            margin_frac: 0.0,
            is_final,
        }
    }
}

/// Stratified GOP sample indices: an even stride across `0..n_gops` (always
/// including GOP 0), capped at `max_sample`. Deterministic, no RNG — an even
/// stride covers scene structure better than random picks (§17.1) and keeps
/// the estimate reproducible.
pub(crate) fn stratified_sample(n_gops: usize, max_sample: usize) -> Vec<usize> {
    if n_gops == 0 {
        return Vec::new();
    }
    let k = n_gops.min(max_sample.max(1));
    if k >= n_gops {
        return (0..n_gops).collect();
    }
    // `i * n_gops / k` spreads k picks across [0, n_gops); i=0 → GOP 0.
    (0..k).map(|i| i * n_gops / k).collect()
}

/// Fold the sampled per-GOP payloads + injectable-bit counts into a
/// [`CapacityEstimate`]. Pure (no I/O) so the statistics are unit-testable on
/// synthetic samples.
pub(crate) fn aggregate(
    payloads: &[f64],
    inject_bits: &[f64],
    n_gops_total: usize,
    is_final: bool,
) -> CapacityEstimate {
    let k = payloads.len();
    if k == 0 || n_gops_total == 0 {
        return CapacityEstimate::empty(n_gops_total, is_final);
    }
    let mean_payload = payloads.iter().sum::<f64>() / k as f64;
    let mean_inject = inject_bits.iter().sum::<f64>() / k as f64;

    // Adaptive one-sided lower bound on the mean. Needs ≥2 samples for spread;
    // a full scan (k == n_gops) has no extrapolation error so margin is 0.
    let margin_frac = if k >= 2 && k < n_gops_total && mean_payload > 0.0 {
        let var = payloads
            .iter()
            .map(|x| (x - mean_payload) * (x - mean_payload))
            .sum::<f64>()
            / (k as f64 - 1.0); // Bessel-corrected sample variance
        let std_err = (var / k as f64).sqrt();
        ((Z_90 * std_err) / mean_payload).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let lower_mean = mean_payload * (1.0 - margin_frac);
    let sigma_total = (n_gops_total as f64) * lower_mean;
    let bytes_estimate = (sigma_total.max(0.0) as usize).saturating_sub(FRAME_OVERHEAD);
    // Pool C is reported at the sample mean (no margin): the shadow formula is
    // already conservative (worst-case parity 128), and the segmented bar wants
    // the best central estimate of the pool to partition.
    let pool_c_bits = ((n_gops_total as f64) * mean_inject).max(0.0) as usize;

    CapacityEstimate {
        bytes_estimate,
        pool_c_bits,
        gop_mean_payload: mean_payload as usize,
        gops_scanned: k,
        gops_total: n_gops_total,
        margin_frac: margin_frac as f32,
        is_final,
    }
}

/// Probe one GOP's tier-0 STC budget + injectable pool. Returns
/// `(primary_payload_bytes, injectable_cover_bits)`.
fn probe_one_gop(
    gop_yuv: &[u8],
    width: u32,
    height: u32,
    opts: EncodeOpts,
) -> Result<(f64, f64), StegoError> {
    let frame_size = (width as usize) * (height as usize) * 3 / 2;
    if frame_size == 0 || gop_yuv.len() < frame_size {
        return Err(StegoError::InvalidVideo(format!(
            "estimator: GOP yuv {} too small for {}x{}",
            gop_yuv.len(),
            width,
            height
        )));
    }
    let gop_n = (gop_yuv.len() / frame_size) as u32;
    let weights = super::cost_weights::CostWeights::default();
    // tier-0 only (full_tiers = false) — the live path; tiers are
    // capacity-neutral on real content (#814).
    let cap = oh264_gop_capacity_per_tier(gop_yuv, width, height, gop_n, opts, &weights, false)?;
    Ok((cap.per_tier_payload[0] as f64, cap.injectable_cover_bits as f64))
}

/// Quick capacity estimate from **GOP 0 only** → instant rough number for the
/// HUD ("≈ X KB, estimating…"). `gop0_yuv` is GOP 0's tight I420 bytes;
/// `n_gops_total` is the clip's GOP count (for the extrapolation). Refine with
/// [`oh264_capacity_estimate`].
pub fn oh264_capacity_quick(
    gop0_yuv: &[u8],
    width: u32,
    height: u32,
    opts: EncodeOpts,
    n_gops_total: usize,
) -> Result<CapacityEstimate, StegoError> {
    if n_gops_total == 0 {
        return Ok(CapacityEstimate::empty(0, true));
    }
    let (payload, inject) = probe_one_gop(gop0_yuv, width, height, opts)?;
    // is_final = false: GOP 0 is a single sample, refinement expected.
    Ok(aggregate(&[payload], &[inject], n_gops_total, false))
}

/// Background-refining sampled estimator (§17.1). Probes a stratified sample of
/// at most `max_sample_gops` GOPs (default [`DEFAULT_MAX_SAMPLE_GOPS`]) so cost
/// is **independent of clip length**, aggregates by Σ sample-mean + adaptive
/// margin, and streams a refining [`CapacityEstimate`] via `progress` after
/// each GOP. Returns the final estimate (`is_final = true`).
///
/// - `get_gop_yuv(idx)` yields GOP `idx`'s tight I420 bytes — the caller
///   materializes only the sampled GOPs (memory-bounded to one GOP).
/// - `cancel()` returning `true` stops early and returns the best estimate so
///   far (e.g. the user changed the video).
#[allow(clippy::too_many_arguments)]
pub fn oh264_capacity_estimate(
    n_gops_total: usize,
    width: u32,
    height: u32,
    opts: EncodeOpts,
    max_sample_gops: usize,
    mut get_gop_yuv: impl FnMut(usize) -> Result<Vec<u8>, StegoError>,
    mut progress: impl FnMut(CapacityEstimate),
    cancel: impl Fn() -> bool,
) -> Result<CapacityEstimate, StegoError> {
    let sample = stratified_sample(n_gops_total, max_sample_gops);
    if sample.is_empty() {
        return Ok(CapacityEstimate::empty(n_gops_total, true));
    }

    let mut payloads: Vec<f64> = Vec::with_capacity(sample.len());
    let mut inject_bits: Vec<f64> = Vec::with_capacity(sample.len());

    for &gop_idx in &sample {
        if cancel() {
            // Stop early; the estimate so far is the final answer (the caller
            // is discarding it anyway, but mark it terminal).
            return Ok(aggregate(&payloads, &inject_bits, n_gops_total, true));
        }
        let gop_yuv = get_gop_yuv(gop_idx)?;
        let (payload, inject) = probe_one_gop(&gop_yuv, width, height, opts)?;
        payloads.push(payload);
        inject_bits.push(inject);
        // Refining: not final until the loop completes.
        progress(aggregate(&payloads, &inject_bits, n_gops_total, false));
    }

    Ok(aggregate(&payloads, &inject_bits, n_gops_total, true))
}

/// Convenience wrapper over [`oh264_capacity_estimate`] for an in-memory clip
/// (CLI / tests). Slices the sampled GOPs out of `yuv`; GOP size is
/// `opts.intra_period`. Holds the whole clip — the bridge uses the
/// provider form to stay memory-bounded.
#[allow(clippy::too_many_arguments)]
pub fn oh264_capacity_estimate_yuv(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    opts: EncodeOpts,
    max_sample_gops: usize,
    progress: impl FnMut(CapacityEstimate),
    cancel: impl Fn() -> bool,
) -> Result<CapacityEstimate, StegoError> {
    let frame_size = (width as usize) * (height as usize) * 3 / 2;
    if frame_size == 0 || yuv.len() != frame_size * n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "estimator: yuv len {} != {}x{}x{}",
            yuv.len(),
            width,
            height,
            n_frames
        )));
    }
    let gop_size = (opts.intra_period.max(1)) as usize;
    let n_gops = n_frames.div_ceil(gop_size);
    oh264_capacity_estimate(
        n_gops,
        width,
        height,
        opts,
        max_sample_gops,
        |gop_idx| {
            let f0 = gop_idx * gop_size;
            let f1 = ((gop_idx + 1) * gop_size).min(n_frames);
            Ok(yuv[f0 * frame_size..f1 * frame_size].to_vec())
        },
        progress,
        cancel,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stratified_sample_basics() {
        assert_eq!(stratified_sample(0, 16), Vec::<usize>::new());
        // Fewer GOPs than the cap → take them all.
        assert_eq!(stratified_sample(3, 16), vec![0, 1, 2]);
        // Even stride, always starts at GOP 0.
        let s = stratified_sample(10, 4);
        assert_eq!(s, vec![0, 2, 5, 7]);
        assert_eq!(s[0], 0);
        // Capped regardless of clip length; spread across the range.
        let big = stratified_sample(1000, 16);
        assert_eq!(big.len(), 16);
        assert_eq!(big[0], 0);
        assert!(big.last().unwrap() < &1000);
        assert!(big.windows(2).all(|w| w[0] < w[1])); // strictly increasing
    }

    #[test]
    fn aggregate_mean_and_extrapolation() {
        // 4 sampled GOPs, mean payload 100, over 10 GOPs → Σ ≈ 1000 − overhead.
        let payloads = [100.0, 100.0, 100.0, 100.0];
        let inject = [800.0, 800.0, 800.0, 800.0];
        let est = aggregate(&payloads, &inject, 10, false);
        assert_eq!(est.gop_mean_payload, 100);
        assert_eq!(est.gops_scanned, 4);
        assert_eq!(est.gops_total, 10);
        // Zero spread → zero margin → Σ = 10·100 − FRAME_OVERHEAD.
        assert_eq!(est.margin_frac, 0.0);
        assert_eq!(est.bytes_estimate, 1000 - FRAME_OVERHEAD);
        assert_eq!(est.pool_c_bits, 8000);
        assert!(!est.is_final);
    }

    #[test]
    fn aggregate_margin_shrinks_with_sample_size() {
        // Same relative spread, more samples → tighter margin (∝ 1/√K).
        let small = aggregate(&[80.0, 120.0], &[0.0, 0.0], 100, false);
        let large = aggregate(
            &[80.0, 120.0, 80.0, 120.0, 80.0, 120.0, 80.0, 120.0],
            &[0.0; 8],
            100,
            false,
        );
        assert!(small.margin_frac > 0.0);
        assert!(large.margin_frac > 0.0);
        assert!(
            large.margin_frac < small.margin_frac,
            "margin {} (K=8) should be < {} (K=2)",
            large.margin_frac,
            small.margin_frac
        );
        // Margin lowers the displayed bytes below the raw mean·n_gops.
        assert!(small.bytes_estimate < 100 * 100);
    }

    #[test]
    fn aggregate_full_scan_has_no_margin() {
        // k == n_gops_total → whole clip measured, no extrapolation error.
        let est = aggregate(&[80.0, 120.0, 100.0], &[0.0; 3], 3, true);
        assert_eq!(est.margin_frac, 0.0);
        assert!(est.is_final);
        assert_eq!(est.bytes_estimate, 300 - FRAME_OVERHEAD);
    }

    #[test]
    fn aggregate_empty_is_zero() {
        let est = aggregate(&[], &[], 10, true);
        assert_eq!(est.bytes_estimate, 0);
        assert_eq!(est.pool_c_bits, 0);
        assert!(est.is_final);
    }
}
