// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Ship-gate threshold configuration shared across Phase C tests.
//!
//! See `docs/design/video/h264/phase-c-ship-criteria.md` for the
//! rationale behind each threshold. Single source of truth — Phase C
//! tests import `ShipGateConfig::PHASE_C_DEFAULT` instead of
//! redefining their own constants.

/// Pass/fail thresholds for the OpenH264-backend production-swap
/// decision. Hard gates (G*) block ship; soft gates (S*) are advisory.
///
/// `G1`/`G7`/`G8` are boolean correctness gates without a scalar
/// threshold and don't have fields here — they're tested via separate
/// dedicated harnesses (round-trip, cascade-safety predict-vs-actual,
/// SHA-256 hash-pin).
#[derive(Debug, Clone, Copy)]
pub struct ShipGateConfig {
    // Hard gates
    /// G2 — min per-frame Y-PSNR floor across corpus (dB).
    pub psnr_min_db: f64,
    /// G3 — worst-MB Y-PSNR floor across corpus (dB).
    pub worst_mb_psnr_min_db: f64,
    /// G4 — max per-frame count of pixels with `|src − dec| > 50`.
    pub max_bad_pixels_per_frame: u32,
    /// G5 — min per-frame SSIM across corpus.
    pub ssim_min: f64,
    /// G6 — max marginal-distribution delta vs cover-story reference
    /// (percentage points).
    pub l3_marginal_epsilon_pp: f64,

    // Soft gates
    /// S1 — max Y-PSNR delta between OpenH264-backend and pure-Rust
    /// on identical fixtures (dB).
    pub cross_encoder_psnr_delta_max_db: f64,
    /// S2 — max encode wall-clock ratio (OpenH264 / pure-Rust).
    pub encode_wall_ratio_max: f64,
    /// S3 — max decode wall-clock ratio (OpenH264 / pure-Rust walker).
    pub decode_wall_ratio_max: f64,
    /// S4 — max stego encode wall-clock ratio (stego / clean) on the
    /// same backend.
    pub stego_overhead_ratio_max: f64,
    /// S5 — min recall of cascade-safety predicate (predicted-safe /
    /// actually-safe).
    pub cascade_safe_recall_min: f64,
    /// S6 — max KL-divergence of spatial mb_type×QP joint distribution
    /// vs cover-story reference. `None` until C.4.3 first run sets
    /// baseline.
    pub spatial_fingerprint_kl_max: Option<f64>,
}

impl ShipGateConfig {
    /// Phase C default thresholds, locked 2026-05-12 in
    /// `phase-c-ship-criteria.md`. Each value traces to existing
    /// production-test constants or to a documented architecture-doc
    /// budget.
    pub const PHASE_C_DEFAULT: Self = Self {
        // G2: from PSNR_FLOOR_DB in h264_visual_psnr_regression.rs:78
        psnr_min_db: 22.0,
        // G3: from worst_mb_min >= 25.0 assertion in same file:255
        worst_mb_psnr_min_db: 25.0,
        // G4: from max_bad_pix_per_frame < 50 assertion :264
        max_bad_pixels_per_frame: 50,
        // G5: introduced in Phase C (SSIM was measured but unasserted)
        ssim_min: 0.90,
        // G6: from EPSILON_PCT in h264_stego_distribution_gate.rs:80
        l3_marginal_epsilon_pp: 5.0,
        // S1: introduced in Phase C
        cross_encoder_psnr_delta_max_db: 1.0,
        // S2/S3: from architecture doc + Phase C plan
        encode_wall_ratio_max: 1.5,
        decode_wall_ratio_max: 2.0,
        // S4: stego at most 1.5x clean encode on same backend
        stego_overhead_ratio_max: 1.5,
        // S5: 90% recall on cascade-safety predicate (capacity)
        cascade_safe_recall_min: 0.90,
        // S6: TBD — first C.4.3 measurement sets the baseline
        spatial_fingerprint_kl_max: None,
    };
}
