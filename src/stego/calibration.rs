// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! `AllocationCalibration` constants for the balanced safe planner
//! (`balanced_allocation::plan_safe_balanced`).
//!
//! All constants are corpus-measured (not hand-tuned):
//!
//! - **Tier 1** (per-clip cover-bit distribution): `payload_bits_per_frame_floor`,
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
    /// Conservative **payload bits per frame** floor for the
    /// (resolution, QP) this calibration targets, in units of
    /// STC-ratio-applied capacity bits per frame — the same basis as
    /// `(cap_bytes_per_gop × 8) / gop_size`. Anchored to corpus
    /// minimum so the planner over-deducts gracefully on hard
    /// content. Pre-probe table lookup uses this; the K_min
    /// stratified probe validates the actual content against it via
    /// `table_tolerance`.
    ///
    /// Renamed 2026-10-18 from `cover_bits_per_frame_floor`. The old
    /// name was ambiguous: AV1 stored *raw* cover bits
    /// (pre-STC-ratio) while H.264 stored *payload* bits
    /// (post-STC-ratio, see `h264_calibrate.rs::write_json`'s
    /// `min_cap_bits_per_frame` derivation). Unified on H.264's
    /// post-STC-ratio basis — AV1's constant updated from
    /// `80_000` → `12_000` (= `80_000 × STC_PAYLOAD_RATIO = 80_000
    /// × 0.15`). See `docs/design/video/av1/pessimism-recalibration-
    /// 2026-10-18.md` § 4.5 for the surfacing + reasoning.
    pub payload_bits_per_frame_floor: usize,

    /// Multiplier applied to the statistical lower bound
    /// `μ − t × σ/√K` from the stratified probe. Combined with the
    /// confidence lower bound, this targets the worst-observed
    /// within-clip p1/p50 ratio (≈ 0.24 on IMG_4138).
    pub pessimism_factor: f64,

    /// Target per-GOP utilization rate: the largest share of its own
    /// basis a stego GOP is loaded to before the planner widens the stego
    /// window (`balanced_allocation::select_window`; above
    /// `target_utilization × Σ basis` every GOP carries the clip's own rate).
    /// Must stay `≤ 1 / safety_margin` (then every selected window fits). **Locked 2026-06-08** against the W6
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
        // 2026-10-18: was `80_000` (raw cover bits per frame — the
        // original Phase-1 spike measurement). Updated to the
        // post-STC-ratio payload-bits basis so the planner's
        // `table_pred` is unit-consistent with H.264's constant.
        // Derivation: `79_188 cb/frame × STC_PAYLOAD_RATIO (= 0.15)
        // ≈ 11_878 payload bits/frame`, rounded to the corpus floor
        // `12_000` for conservative over-deduct.
        payload_bits_per_frame_floor: 12_000,
        // 2026-10-26: 0.5 → 0.95. Deliverability measured on the v2 wire
        // (decode-and-replan): 4 clips × 360p/720p × 6 passphrases, every
        // encode at 100 % and at 95 % of the full-probe primary delivered
        // (48/48 at each fill). 0.5 had clamped the red end to α ≈ 7.5 % of the
        // dry cover, under the 1/8 band, so the bar never reached red.
        // 0.95 keeps a 5 % margin over the measured edge
        // (`docs/design/video/av1/v2-gate-results.md`, "Pessimism").
        pessimism_factor: 0.95,
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
    ///   `payload_bits_per_frame_floor` is the H.264 per-GOP PAYLOAD basis
    ///   (lowest clip mean 88 B/GOP × 8 / gop 30 ≈ 23 bits/frame). As of
    ///   2026-10-18 this basis is also AV1's (= post-STC-ratio) — see
    ///   the AV1 constant above. `docs/design/video/calibration-h264-
    ///   tier1-2026-06.md` has the H.264 corpus detail.
    /// - Tier 2 (W6 AUC vs utilization): 4-clip × 8-seek, 1080p/GOP30/QP26.
    ///   AUC FLAT at ~0.51 across u ∈ {5..100}% — no 0.55 crossover (the
    ///   adaptive cost preserves the marginals W6 measures, so detectability
    ///   is rate-insensitive). See
    ///   `docs/design/video/calibration-h264-tier2-2026-06.md`.
    /// - `target_utilization` (2026-10-26, was 0.50): set by VISUAL damage,
    ///   not W6 — 0.042 = the comfort α 1/25 (`CAP_ALPHA_COMFORT`) over the
    ///   probe cap's α ≈ 0.95 (cap = 0.95 × syntax-dry bits / 8, the
    ///   `CAP2_SAFETY` derate). A message below the comfort figure goes to
    ///   the shortest prefix whose GOPs stay at α ≤ 1/25 (plain tail after);
    ///   anything larger spreads over every GOP at the clip's own α. 0.50 had
    ///   loaded the window GOPs of a > 8-GOP clip to α ≈ 0.24 (½ × the old
    ///   pessimism ½ of the mean cap) — 2× the 1/8 red end — and broke the
    ///   27 / 13 dB floor (`intra-drift-diagnosis.md` follow-up 5). Capacity
    ///   is unaffected: the accept ceiling is `Σ caps / safety_margin` either
    ///   way.
    pub const H264_1080P_QP26: Self = Self {
        payload_bits_per_frame_floor: 23,
        pessimism_factor: 0.5,
        target_utilization: 0.042,
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
        assert_eq!(c.payload_bits_per_frame_floor, 12_000);
        // 2026-10-26: 0.5 → 0.95 (v2 deliverability, see the constant).
        assert!((c.pessimism_factor - 0.95).abs() < 1e-9);
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
        assert_eq!(c.payload_bits_per_frame_floor, 23);
        assert!((c.pessimism_factor - 0.5).abs() < 1e-9);
        // 2026-10-26: 0.50 → 0.042 (visual window bound, see the constant).
        assert!((c.target_utilization - 0.042).abs() < 1e-9);
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
