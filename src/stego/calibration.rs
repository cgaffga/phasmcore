// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! `AllocationCalibration` constants for the balanced safe planner
//! (`balanced_allocation::plan_safe_balanced`).
//!
//! All constants are corpus-measured (not hand-tuned):
//!
//! - **Tier 1** (per-clip cover-bit distribution): `cover_bits_per_frame_floor`,
//!   `pessimism_factor`, `K_min`, `table_tolerance`, `safety_margin`.
//!   See [`calibration-spike-2026-06.md`](../../docs/design/video/calibration-spike-2026-06.md).
//!
//! - **Tier 2** (W6 self-steganalyzer AUC-vs-utilization): `target_utilization`.
//!   See [`calibration-tier2-stealth-2026-06.md`](../../docs/design/video/calibration-tier2-stealth-2026-06.md).
//!
//! - **Tier 3** (spread-vs-concentrate AUC measurement): `W_floor_abs`,
//!   `W_floor_frac`. Still design-default; refine when the Tier 3
//!   measurement lands.

/// Calibration constants for `plan_safe_balanced`. Codec-agnostic
/// (same struct serves AV1 and H.264 once the H.264 side wires the
/// planner). The corpus-locked values for AV1 1080p QP30 are in
/// [`Self::AV1_1080P_QP30`]; H.264 + other (res, QP) combos will get
/// their own constants as Tier 1 measurement gets re-run per surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AllocationCalibration {
    /// Conservative cover-bits-per-frame floor for the (resolution, QP)
    /// this calibration targets. Anchored to corpus minimum so the
    /// planner over-deducts gracefully on hard content. Pre-probe
    /// table lookup uses this; the K_min stratified probe validates
    /// the actual content against it via `table_tolerance`.
    pub cover_bits_per_frame_floor: usize,

    /// Multiplier applied to the statistical lower bound
    /// `μ − t × σ/√K` from the stratified probe. Combined with the
    /// confidence lower bound, this targets the worst-observed
    /// within-clip p1/p50 ratio (≈ 0.24 on IMG_4138).
    pub pessimism_factor: f64,

    /// Target per-GOP utilization rate. Drives `W_util` upward toward
    /// balanced spread. **Locked 2026-06-08** against the W6
    /// self-steganalyzer at 1080p QP30: W6 AUC = 0.55 crossover at
    /// u ≈ 0.018; 0.05 is the balanced choice (point-estimate AUC
    /// 0.573, within ~0.4σ of the 0.55 design gate).
    pub target_utilization: f64,

    /// Initial probe sample count. Adaptive escalation in the planner
    /// can extend up to `n_gops` at tight fit.
    pub k_min: usize,

    /// Absolute floor on stego window width. Guards against
    /// changepoint detectability — even a trivial payload spreads
    /// over at least this many GOPs.
    pub w_floor_abs: usize,

    /// Fractional floor on stego window: `W ≥ n_gops × this`.
    /// Combined with `w_floor_abs` via `max()`.
    pub w_floor_frac: f64,

    /// Pre-encode hard-check margin: `total_safe ≥ message_len × this`
    /// before the planner commits. 1.1 = 10% over the pessimistic
    /// estimate.
    pub safety_margin: f64,

    /// Tolerance for sample-vs-table agreement. If
    /// `sample_mean / table_pred` is outside `[1/this, this]` the
    /// table is mis-applied for the current content; the planner
    /// logs a warning but doesn't fail — falls back to sample-driven
    /// planning, which is more robust to weird content.
    pub table_tolerance: f64,
}

impl AllocationCalibration {
    /// Corpus-locked calibration for AV1 at 1920×1080, QP=30.
    ///
    /// Source measurements:
    /// - Tier 1 (cover-bit distribution): 11-clip corpus, 2026-06-08,
    ///   commit `c4c1254e`. min mean cb/frame = 79,188 (Artlist_SchoolFight);
    ///   worst within-clip p1/p50 = 0.24 (IMG_4138); cross-clip CV
    ///   range 0.075-0.71.
    /// - Tier 2 (AUC vs utilization via W6 self-steganalyzer): 3-clip
    ///   fixture × 10 seek points, 2026-06-08, commit `41922eee`. W6
    ///   AUC=0.55 crossover at u_eff ≈ 0.018; u=0.05 picks the balanced
    ///   choice clearing the W6 v1.0 ceiling 0.60 with margin.
    pub const AV1_1080P_QP30: Self = Self {
        cover_bits_per_frame_floor: 80_000,
        pessimism_factor: 0.5,
        target_utilization: 0.05,
        k_min: 8,
        w_floor_abs: 8,
        w_floor_frac: 0.05,
        safety_margin: 1.1,
        table_tolerance: 2.0,
    };

    /// Corpus-locked calibration for H.264 (OpenH264) at 1920×1088, QP=26
    /// (#844). Added by the H.264 session with the AV1 session's sign-off
    /// (the struct is codec-agnostic; this is purely additive).
    ///
    /// Source measurements (14-clip real-world corpus, 2026-06-09):
    /// - Tier 1 (per-GOP cap-byte distribution at the PRODUCTION gop_size,
    ///   not the AV1 spike's gop_size=1 upper bound): per-GOP cap yield
    ///   spans 190× (WomanSubway 88 B/GOP .. iPhone-5 16.7 KB/GOP grain).
    ///   `cover_bits_per_frame_floor` is the H.264 per-GOP PAYLOAD basis
    ///   (lowest clip mean 88 B/GOP × 8 / gop 30 ≈ 23 bits/frame) — NOT
    ///   comparable to AV1's 80_000 raw-cover-bit floor. See
    ///   `docs/design/video/calibration-h264-tier1-2026-06.md`.
    /// - Tier 2 (W6 AUC vs utilization): 4-clip × 8-seek, 1080p/GOP30/QP26.
    ///   AUC FLAT at ~0.51 across u ∈ {5..100}% — no 0.55 crossover (the
    ///   adaptive cost preserves the marginals W6 measures, so detectability
    ///   is rate-insensitive). `target_utilization=0.50` is the user's
    ///   posture choice (aggressive — max capacity, banking on the flat
    ///   curve; `w_floor_*` still guarantees minimum spread). See
    ///   `docs/design/video/calibration-h264-tier2-2026-06.md`.
    pub const H264_1080P_QP26: Self = Self {
        cover_bits_per_frame_floor: 23,
        pessimism_factor: 0.5,
        target_utilization: 0.50,
        k_min: 8,
        w_floor_abs: 8,
        w_floor_frac: 0.05,
        safety_margin: 1.1,
        table_tolerance: 2.0,
    };

    /// Provisional default for callers that don't yet know their
    /// (codec, resolution, QP) — uses the AV1 1080p QP30 lock. Re-run
    /// Tier 1 / Tier 2 for other surfaces before relying on this.
    pub const DEFAULT: Self = Self::AV1_1080P_QP30;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn av1_1080p_qp30_locked_values() {
        // Regression guard — these are corpus-measured, not hand-tuned.
        // Any change requires a fresh calibration spike + design doc update.
        let c = AllocationCalibration::AV1_1080P_QP30;
        assert_eq!(c.cover_bits_per_frame_floor, 80_000);
        assert!((c.pessimism_factor - 0.5).abs() < 1e-9);
        assert!((c.target_utilization - 0.05).abs() < 1e-9);
        assert_eq!(c.k_min, 8);
        assert_eq!(c.w_floor_abs, 8);
        assert!((c.w_floor_frac - 0.05).abs() < 1e-9);
        assert!((c.safety_margin - 1.1).abs() < 1e-9);
        assert!((c.table_tolerance - 2.0).abs() < 1e-9);
    }

    #[test]
    fn h264_1080p_qp26_locked_values() {
        // #844 — corpus-measured (Tier 1) + W6-swept (Tier 2), not hand-tuned.
        // Any change requires a fresh H.264 calibration spike + doc update.
        let c = AllocationCalibration::H264_1080P_QP26;
        assert_eq!(c.cover_bits_per_frame_floor, 23);
        assert!((c.pessimism_factor - 0.5).abs() < 1e-9);
        assert!((c.target_utilization - 0.50).abs() < 1e-9);
        assert_eq!(c.k_min, 8);
        assert_eq!(c.w_floor_abs, 8);
        assert!((c.w_floor_frac - 0.05).abs() < 1e-9);
        assert!((c.safety_margin - 1.1).abs() < 1e-9);
        assert!((c.table_tolerance - 2.0).abs() < 1e-9);
    }

    #[test]
    fn default_equals_av1_1080p_qp30() {
        assert_eq!(
            AllocationCalibration::DEFAULT,
            AllocationCalibration::AV1_1080P_QP30
        );
    }
}
