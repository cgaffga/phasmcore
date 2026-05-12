// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Rate control + the public quality API. Phase 6A.9.
//!
//! Variance-based adaptive QP (AQ-mode 1) and mini-lookahead land in
//! later sub-phases (6B.5 and post-6A polish respectively). Phase 6A.9
//! ships:
//!
//!   - `quality_to_crf(u8) -> u8` — user-facing [0, 100] → CRF [14, 51].
//!   - `crf_to_qp(u8) -> u8` — CRF → base QP (identity for Phase 6A
//!     I-frame-only; P/B offsets arrive with Phase 6B).
//!   - `estimate_source_crf_from_sps_pps` — smart "match-source"
//!     default so the encoder doesn't force a choice on users who
//!     haven't picked one.
//!   - `quality_label(crf) -> &str` — human-readable tier for HUD/logs.
//!   - `RateController` — holds the chosen CRF + flag for auto-match.
//!
//! Algorithm note:
//!   docs/design/video/h264/encoder-algorithms/quality-model.md

use crate::codec::h264::sps::{Pps, Sps};

// ─── Quality → CRF mapping ────────────────────────────────────────

/// Anchor points from the design doc:
/// quality 100→14, 90→18, 75→23, 50→28, 25→33, 10→38, 0→51.
const QUALITY_CRF_ANCHORS: [(u8, u8); 7] = [
    (0, 51),
    (10, 38),
    (25, 33),
    (50, 28),
    (75, 23),
    (90, 18),
    (100, 14),
];

/// Map user quality (0..=100) to internal CRF (14..=51).
///
/// Higher `quality` → lower CRF → better compression ratio kept. The
/// mapping is piecewise-linear between the anchor points above;
/// rounded to the nearest integer CRF (sign-aware — our slope is
/// negative because CRF decreases with quality).
pub fn quality_to_crf(quality: u8) -> u8 {
    let q = quality.min(100);
    for w in QUALITY_CRF_ANCHORS.windows(2) {
        let (q_lo, crf_lo) = w[0];
        let (q_hi, crf_hi) = w[1];
        if q >= q_lo && q <= q_hi {
            if q == q_lo {
                return crf_lo;
            }
            if q == q_hi {
                return crf_hi;
            }
            let span_q = (q_hi - q_lo) as i32;
            let span_crf = crf_hi as i32 - crf_lo as i32;
            let frac = (q - q_lo) as i32;
            let numerator = span_crf * frac;
            // Sign-aware round-to-nearest — don't use the
            // (a + b/2) / b idiom because it rounds toward zero, not
            // toward nearest, for negative a.
            let rounded = if numerator >= 0 {
                (numerator + span_q / 2) / span_q
            } else {
                (numerator - span_q / 2) / span_q
            };
            return (crf_lo as i32 + rounded).clamp(14, 51) as u8;
        }
    }
    QUALITY_CRF_ANCHORS.last().unwrap().1
}

/// Map CRF to the encoder's base QP (per-frame-type offsets applied
/// by the rate controller later). For Phase 6A (I-frame only), this
/// is the identity.
#[inline]
pub fn crf_to_qp(crf: u8) -> u8 {
    crf.clamp(0, 51)
}

/// Human-readable quality tier for HUD / CLI output.
pub fn quality_label(crf: u8) -> &'static str {
    match crf {
        0..=16 => "visually lossless",
        17..=20 => "very high",
        21..=25 => "high",
        26..=30 => "medium",
        31..=35 => "low",
        36..=40 => "very low",
        _ => "extremely low",
    }
}

// ─── Source-quality estimator ────────────────────────────────────

/// Estimate the source's effective CRF from its SPS/PPS + sampled
/// slice_qp_delta values. Used when the user's `quality:` input is
/// `None` — the encoder then matches the source's tier rather than
/// picking an arbitrary default.
///
/// Heuristic: `base = pic_init_qp_minus26 + 26`; average in the
/// provided slice deltas; clamp to the valid CRF range.
///
/// Falls back to 23 (a conventional medium-quality default) if
/// `slice_qp_deltas` is empty.
pub fn estimate_source_crf_from_sps_pps(
    _sps: &Sps,
    pps: &Pps,
    slice_qp_deltas: &[i32],
) -> u8 {
    let base = pps.pic_init_qp_minus26 + 26;
    if slice_qp_deltas.is_empty() {
        return base.clamp(14, 51) as u8;
    }
    let sum: i32 = slice_qp_deltas.iter().sum();
    let avg = sum / slice_qp_deltas.len() as i32;
    (base + avg).clamp(14, 51) as u8
}

// ─── RateController state ────────────────────────────────────────

/// Rate controller — holds the chosen target CRF.
///
/// Later phases (6B.5 mini-lookahead, 6A.9 adaptive-QP variance pass)
/// extend this with per-MB QP deltas. For Phase 6A.9 the only state
/// is the frame-level target.
#[derive(Debug, Clone)]
pub struct RateController {
    pub target_crf: u8,
    pub auto_match_source: bool,
}

impl RateController {
    /// Construct a rate controller.
    ///
    /// - `Some(quality)` → pin to the user's choice (0..=100).
    /// - `None` → mark as auto-match; caller should call
    ///   `set_from_source_estimate` once it has parsed the input's
    ///   SPS/PPS.
    pub fn new(quality: Option<u8>) -> Self {
        match quality {
            Some(q) => Self {
                target_crf: quality_to_crf(q),
                auto_match_source: false,
            },
            None => Self {
                target_crf: 23, // fallback until set_from_source_estimate runs
                auto_match_source: true,
            },
        }
    }

    /// Adopt the estimated source CRF when `auto_match_source` is on.
    /// No-op if the user pinned an explicit quality.
    pub fn set_from_source_estimate(&mut self, estimated_crf: u8) {
        if self.auto_match_source {
            self.target_crf = estimated_crf.clamp(14, 51);
        }
    }

    /// Base QP for the given frame type. I-slice uses the target CRF
    /// directly; P-slice gets a +2 offset (standard convention across
    /// H.264 encoders) so more residual coefficients round to zero
    /// and skip-eligible MBs enter the mb_skip_run path. Without
    /// this offset the encoder wastes bits on near-zero AC
    /// coefficients that a well-tuned encoder would have quantised
    /// away — our pre-offset version was emitting ~50% more bits at
    /// comparable quality settings.
    pub fn base_qp_for_frame_type(&self, frame_type: FrameType) -> u8 {
        let base = crf_to_qp(self.target_crf);
        match frame_type {
            FrameType::I => base,
            FrameType::P => (base as i32 + 1).clamp(0, 51) as u8,
            FrameType::B => {
                // §B-quality-experiment 2026-05-09 — env override for the
                // B-frame QP offset experiment (#290 follow-on). The +2
                // default suppresses near-zero residual coefficients to
                // hit skip-eligible MB share, but on textured-motion
                // content it ALSO suppresses the texture-grain coefficients
                // that perceptually matter. PHASM_B_QP=N overrides to
                // absolute QP N (clamped 0..=51); PHASM_B_QP_OFFSET=±N
                // adjusts the +2 default by ±N.
                if let Some(qp) = super::mb_decision_b::env_var("PHASM_B_QP")
                    .and_then(|s| s.parse::<i32>().ok())
                {
                    qp.clamp(0, 51) as u8
                } else if let Some(off) = super::mb_decision_b::env_var("PHASM_B_QP_OFFSET")
                    .and_then(|s| s.parse::<i32>().ok())
                {
                    (base as i32 + off).clamp(0, 51) as u8
                } else {
                    (base as i32 + 2).clamp(0, 51) as u8
                }
            }
        }
    }

    /// Human label for the current tier (HUD / logs).
    pub fn quality_label(&self) -> &'static str {
        quality_label(self.target_crf)
    }
}

/// Frame type for rate-control queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    I,
    P,
    B,
}

// ─── Adaptive QP (AQ-mode 1) ─────────────────────────────────────
//
// Phase 6A polish #8. Per-MB variance-based QP adjustment:
// low-variance (flat) MBs get a QP bump (less bits, imperceptible
// loss); high-variance (textured) MBs get a QP cut (more bits,
// preserves detail).

/// Compute the per-pixel variance of a 16×16 luma MB. Returns a
/// u32 suitable for log-scale comparisons.
pub fn mb_variance_16x16(src: &[[u8; 16]; 16]) -> u32 {
    let mut sum: u32 = 0;
    let mut sumsq: u32 = 0;
    for row in src {
        for &v in row {
            let x = v as u32;
            sum = sum.saturating_add(x);
            sumsq = sumsq.saturating_add(x * x);
        }
    }
    // var = E[X²] − (E[X])². 256 samples.
    let n = 256u32;
    let mean = sum / n;
    let mean_sq = mean * mean;
    sumsq / n - mean_sq.min(sumsq / n)
}

/// Default AQ strength. 1.0 is the unity-strength reference point
/// from the adaptive-quantisation literature; we store as a
/// fixed-point fraction to avoid float dependencies.
pub const AQ_STRENGTH_NUM: i32 = 10; // = 1.0 × 10
pub const AQ_STRENGTH_DEN: i32 = 10;

/// Derive a per-MB QP offset from variance.
///
/// Heuristic: `offset = -strength × log2(variance / reference_var)`,
/// clamped to ±6 QP. Flat MBs (variance << reference) get a positive
/// offset (higher QP, cheaper encode); textured MBs (variance >>
/// reference) get a negative offset (lower QP, preserves detail).
///
/// `reference_var` is a scene-average variance estimate; for a
/// first-pass implementation we use a fixed value tuned to typical
/// 8-bit content (~256 ~= moderate texture).
pub fn variance_to_qp_offset(variance: u32) -> i32 {
    const REFERENCE_VAR: u32 = 256;
    const MAX_OFFSET: i32 = 6;
    let var = variance.max(1);
    // Compute log2 ratio as int: log2(var) - log2(REF).
    let log2_var = 31 - var.leading_zeros() as i32;
    let log2_ref = 31 - REFERENCE_VAR.leading_zeros() as i32;
    let log_ratio = log2_var - log2_ref;
    let offset = -(log_ratio * AQ_STRENGTH_NUM) / AQ_STRENGTH_DEN;
    offset.clamp(-MAX_OFFSET, MAX_OFFSET)
}

/// Integer log2 of a 16-bit variance bucket in Q8 fixed-point
/// (8 fractional bits). Returns 0 for variance ≤ 1.
#[inline]
pub fn log2_var_q8(variance: u32) -> i32 {
    let v = variance.max(1);
    let int = 31 - v.leading_zeros() as i32;
    // Fractional part: next 8 bits below the top bit give a rough
    // log2 interpolation via (v - 2^int) / 2^int × 256.
    let frac = if int >= 8 {
        ((v >> (int - 8)) & 0xFF) as i32
    } else {
        ((v << (8 - int)) & 0xFF) as i32
    };
    (int << 8) + frac
}

/// Frame-aggregated log2-variance centroid. Computed by summing
/// [`log2_var_q8`] across all MBs and dividing by count. Serves as
/// the "auto-variance" reference for variance-based AQ mode 2/3:
/// each MB's offset is applied relative to this frame mean so that
/// the encoder's total bit budget is approximately preserved
/// regardless of overall scene complexity.
#[inline]
pub fn frame_mean_log2_q8(per_mb_log2: &[i32]) -> i32 {
    if per_mb_log2.is_empty() {
        return 0;
    }
    let sum: i64 = per_mb_log2.iter().map(|&v| v as i64).sum();
    (sum / per_mb_log2.len() as i64) as i32
}

/// AQ mode 3 (auto-variance + dark bias) per-MB QP offset.
///
/// `offset = strength_q8 × (log2(var) − frame_mean_log2) / Q8`
/// plus a negative bias when MB average luma is very dark (banding
/// protection). Textured MBs above frame mean get a negative offset
/// → lower QP → finer quant → text/edges preserved. Flat MBs below
/// the mean get a positive offset → coarser quant → saved bits.
/// Dark regions (avg_luma < 30) additionally get `-2 × strength`
/// offset to avoid visible banding in shadows.
///
/// `strength_q10` is a Q10 fixed-point strength (1024 = 1.0 —
/// the unity-strength reference point from the variance-AQ
/// literature). Final offset is clamped to ±6 QP.
pub fn variance_to_qp_offset_mode3(
    variance: u32,
    frame_mean_log2_q8: i32,
    avg_luma: u32,
    strength_q10: i32,
) -> i32 {
    const MAX_OFFSET: i32 = 6;
    let var_log_q8 = log2_var_q8(variance);
    let delta_log_q8 = var_log_q8 - frame_mean_log2_q8;
    // offset = strength_q10 × delta_log_q8 / (256 × 1024)
    //        = strength_q10 × delta_log_q8 >> 18.
    // Positive delta_log → textured MB → negative offset (finer quant).
    let base = -((strength_q10 as i64 * delta_log_q8 as i64) >> 18) as i32;
    let dark = if avg_luma < 30 {
        -((2 * strength_q10) >> 10)
    } else {
        0
    };
    (base + dark).clamp(-MAX_OFFSET, MAX_OFFSET)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── quality_to_crf ──────────────────────────────────────────

    #[test]
    fn quality_to_crf_hits_anchors() {
        assert_eq!(quality_to_crf(0), 51);
        assert_eq!(quality_to_crf(10), 38);
        assert_eq!(quality_to_crf(25), 33);
        assert_eq!(quality_to_crf(50), 28);
        assert_eq!(quality_to_crf(75), 23);
        assert_eq!(quality_to_crf(90), 18);
        assert_eq!(quality_to_crf(100), 14);
    }

    #[test]
    fn quality_to_crf_monotonic_non_increasing() {
        let mut prev = u8::MAX;
        for q in 0..=100u8 {
            let crf = quality_to_crf(q);
            assert!(
                crf <= prev,
                "quality_to_crf not monotonic: q={q} crf={crf} > prev={prev}"
            );
            prev = crf;
        }
    }

    #[test]
    fn quality_to_crf_clamps_above_100() {
        assert_eq!(quality_to_crf(255), 14);
    }

    #[test]
    fn quality_to_crf_interpolates_between_anchors() {
        // 60 sits between 50 (crf 28) and 75 (crf 23).
        // Interp: 28 + (23-28)*(60-50)/(75-50) = 28 + (-5)*(10/25) = 26.
        assert_eq!(quality_to_crf(60), 26);
    }

    // ─── crf_to_qp ───────────────────────────────────────────────

    #[test]
    fn crf_to_qp_identity_in_range() {
        for crf in 0..=51u8 {
            assert_eq!(crf_to_qp(crf), crf);
        }
    }

    #[test]
    fn crf_to_qp_clamps_above_51() {
        assert_eq!(crf_to_qp(60), 51);
    }

    // ─── quality_label ───────────────────────────────────────────

    #[test]
    fn quality_label_tiers() {
        assert_eq!(quality_label(14), "visually lossless");
        assert_eq!(quality_label(18), "very high");
        assert_eq!(quality_label(23), "high");
        assert_eq!(quality_label(28), "medium");
        assert_eq!(quality_label(33), "low");
        assert_eq!(quality_label(38), "very low");
        assert_eq!(quality_label(50), "extremely low");
    }

    // ─── estimate_source_crf_from_sps_pps ────────────────────────

    fn dummy_sps() -> Sps {
        Sps {
            profile_idc: 66,
            constraint_set_flags: 0,
            level_idc: 30,
            sps_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            qpprime_y_zero_transform_bypass_flag: false,
            log2_max_frame_num: 4,
            pic_order_cnt_type: 2,
            log2_max_pic_order_cnt_lsb: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            max_num_ref_frames: 1,
            gaps_in_frame_num_allowed: false,
            pic_width_in_mbs: 20,
            pic_height_in_map_units: 15,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: true,
            frame_cropping_flag: false,
            crop_left: 0,
            crop_right: 0,
            crop_top: 0,
            crop_bottom: 0,
            width_in_pixels: 320,
            height_in_pixels: 240,
            pic_size_in_mbs: 300,
        }
    }

    fn dummy_pps(pic_init_qp_minus26: i32) -> Pps {
        Pps {
            pps_id: 0,
            sps_id: 0,
            entropy_coding_mode_flag: false,
            bottom_field_pic_order_in_frame_present_flag: false,
            num_slice_groups_minus1: 0,
            num_ref_idx_l0_default: 0,
            num_ref_idx_l1_default: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            deblocking_filter_control_present_flag: false,
            constrained_intra_pred_flag: false,
            redundant_pic_cnt_present_flag: false,
            transform_8x8_mode_flag: false,
            second_chroma_qp_index_offset: 0,
        }
    }

    #[test]
    fn estimate_with_no_slice_deltas_uses_base() {
        let sps = dummy_sps();
        let pps = dummy_pps(0); // pic_init_qp = 26
        let crf = estimate_source_crf_from_sps_pps(&sps, &pps, &[]);
        assert_eq!(crf, 26);
    }

    #[test]
    fn estimate_adds_average_slice_delta() {
        let sps = dummy_sps();
        let pps = dummy_pps(-3); // pic_init_qp = 23
        let crf = estimate_source_crf_from_sps_pps(&sps, &pps, &[1, 2, 3]);
        // avg delta = 2, base = 23, → 25
        assert_eq!(crf, 25);
    }

    #[test]
    fn estimate_clamps_to_valid_crf_range() {
        let sps = dummy_sps();
        let pps = dummy_pps(50); // pic_init_qp = 76 — way out of range
        let crf = estimate_source_crf_from_sps_pps(&sps, &pps, &[]);
        assert_eq!(crf, 51);

        let pps = dummy_pps(-30); // pic_init_qp = -4 — clamped up
        let crf = estimate_source_crf_from_sps_pps(&sps, &pps, &[]);
        assert_eq!(crf, 14);
    }

    // ─── RateController ──────────────────────────────────────────

    #[test]
    fn rate_controller_pinned_quality() {
        let rc = RateController::new(Some(75));
        assert_eq!(rc.target_crf, 23);
        assert!(!rc.auto_match_source);
        assert_eq!(rc.base_qp_for_frame_type(FrameType::I), 23);
        assert_eq!(rc.quality_label(), "high");
    }

    #[test]
    fn rate_controller_auto_match_updates_on_estimate() {
        let mut rc = RateController::new(None);
        assert!(rc.auto_match_source);
        rc.set_from_source_estimate(30);
        assert_eq!(rc.target_crf, 30);
        assert_eq!(rc.quality_label(), "medium");
    }

    #[test]
    fn rate_controller_pinned_ignores_source_estimate() {
        let mut rc = RateController::new(Some(90));
        assert_eq!(rc.target_crf, 18);
        rc.set_from_source_estimate(40);
        // Should still be 18 — user pinned, source estimate ignored.
        assert_eq!(rc.target_crf, 18);
    }

    // ─── Adaptive QP ─────────────────────────────────────────────

    #[test]
    fn mb_variance_flat_block_is_zero() {
        let block = [[100u8; 16]; 16];
        assert_eq!(mb_variance_16x16(&block), 0);
    }

    #[test]
    fn mb_variance_checkerboard_is_nonzero() {
        let mut block = [[0u8; 16]; 16];
        for y in 0..16 {
            for x in 0..16 {
                block[y][x] = if (x ^ y) & 1 == 0 { 50 } else { 200 };
            }
        }
        let v = mb_variance_16x16(&block);
        // Variance of {50, 200} with equal counts: mean=125, var=75²=5625.
        assert!(v > 5000 && v < 6000, "unexpected variance {v}");
    }

    #[test]
    fn qp_offset_flat_block_raises_qp() {
        // Very low variance → positive offset (bigger QP, cheaper).
        let offset = variance_to_qp_offset(4);
        assert!(offset > 0, "flat MB should raise QP, got {offset}");
    }

    #[test]
    fn qp_offset_textured_block_lowers_qp() {
        // Very high variance → negative offset (smaller QP, preserves detail).
        let offset = variance_to_qp_offset(4096);
        assert!(offset < 0, "textured MB should lower QP, got {offset}");
    }

    #[test]
    fn qp_offset_clamped_at_reasonable_range() {
        // Extreme values should be clamped, not overflow.
        assert!((-6..=6).contains(&variance_to_qp_offset(1)));
        assert!((-6..=6).contains(&variance_to_qp_offset(u32::MAX)));
    }
}
