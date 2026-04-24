// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! P-slice MB partition decision. Phase 6B.3.2b/c/d.
//!
//! For each P-slice MB, enumerate the partition candidates we support
//! — P_L0_16x16, P_L0_L0_16x8, P_L0_L0_8x16, and (Phase 6B.3.2c/d)
//! P_8x8 with sub_mb_types — and pick the one with lowest SATD +
//! fixed overhead penalty.

use super::intra_predictor::hadamard_ac_sum_16x16;
use super::motion_estimation::{MotionEstimator, MotionVector};
use super::partition_state::{predict_mv_for_partition, EncoderMvGrid};
use super::reference_buffer::ReconFrame;

/// Task #121 Phase 1 — multi-predictor ME seeding. Default ON
/// since 2026-04-23 measurement: 90f IMG_4138 deltas were +0.20 dB
/// (Q=80), +0.48 dB (Q=40), +0.62 dB (Q=26) — monotonic with
/// residual energy, Q=26 clears the +0.5 dB go/no-go. Opt out via
/// `PHASM_ME_MULTI_PRED=0`.
fn multi_pred_enabled() -> bool {
    std::env::var("PHASM_ME_MULTI_PRED")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Build the multi-predictor candidate list for an ME call at
/// 4×4-grid position `(tl_bx, tl_by)` spanning `part_w_4x4 ×
/// part_h_4x4`. Always includes the median (`predictor`) and zero;
/// adds A (left), B (top), C (top-right) raw MVs when decoded.
/// Duplicates are common and harmless — evaluation is cheap and the
/// winner is the cheapest under SATD + λ·mv_bits.
fn build_me_candidates(
    grid: &EncoderMvGrid,
    tl_bx: usize,
    tl_by: usize,
    part_w_4x4: usize,
    predictor: MotionVector,
) -> Vec<MotionVector> {
    if !multi_pred_enabled() {
        return vec![predictor];
    }
    let x = tl_bx as isize;
    let y = tl_by as isize;
    let mut cands = Vec::with_capacity(6);
    cands.push(predictor);
    cands.push(MotionVector::ZERO);
    if let Some((mv, _)) = grid.get(x - 1, y) {
        cands.push(mv);
    }
    if let Some((mv, _)) = grid.get(x, y - 1) {
        cands.push(mv);
    }
    if let Some((mv, _)) = grid.get(x + part_w_4x4 as isize, y - 1) {
        cands.push(mv);
    }
    cands
}

/// Environment-configurable psy bias strength for P-MB partition
/// cost (Phase E.2-LITE). Adds `strength × |hadamard_ac(src) −
/// hadamard_ac(pred)| / 256` to every P-candidate's SATD+penalty
/// cost. Psychovisual-RDO style — penalise predictors that smooth
/// away high-frequency AC content humans perceive. Grafted onto
/// our SATD cost instead of a full RDO D+λR framework.
///
/// Default 0 (disabled). 2026-04-23 sweep on IMG_4138 90f:
///   Q=40 STR=64:   +0.01 dB / +0.2%  bits (rate-neutral, marginal)
///   Q=40 STR=256:  +0.05 dB / +3.5%  bits (poor R-D)
///   Q=40 STR=1024: −0.06 dB / −1.0%  bits (trades quality for bits)
///   Q=26 STR=64:   +0.03 dB / +0.1%  bits (rate-neutral, marginal)
///   Q=26 STR=256:  +0.06 dB / +2.8%  bits (poor R-D)
///   Q=26 STR=1024: −0.06 dB / −4.2%  bits (best R-D — small PSNR
///                  drop for meaningful bitrate save; needs visual
///                  A/B to confirm AC preservation isn't hurt).
/// Kept opt-in (`PHASM_INTER_PSY_STRENGTH=N`) pending visual
/// confirmation of STR=1024 at low QP.
fn inter_psy_strength() -> u32 {
    std::env::var("PHASM_INTER_PSY_STRENGTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Partition overhead in SATD units. Captures "this mb_type produces
/// more MVDs and more header bits, so the distortion win has to be
/// bigger than this before we pick it." First-pass values; tunable.
///
/// 2026-04-20: tried tightening to 24/24/96 (λ×bits calibration).
/// Regressed: 30f −2 dB, bitrate +18%. Reason: tighter penalties
/// let sub-partitions win more (P_8x8 2%→7%), which dropped avg
/// inter_cost enough that RDO P_SKIP fired LESS (skip was passing
/// the `skip_satd < inter_cost + λ·30` threshold less often). Net
/// quality regression. Keep 64/64/256 until we do full RDO (including
/// sub-part decisions going through the same λ·bits framework).
pub const PENALTY_16X8: u32 = 64;
pub const PENALTY_8X16: u32 = 64;
pub const PENALTY_8X8: u32 = 256;

/// The encoder's resolved partition choice for a single P-slice MB.
///
/// Each variant carries the absolute MVs (not MVDs) for each
/// partition in the spec's emit order.
#[derive(Debug, Clone, Copy)]
pub enum PMbChoice {
    /// One 16×16 partition, 1 MV.
    P16x16 { mv: MotionVector },
    /// Two 16×8 partitions (top, bottom). 2 MVs.
    P16x8 { mvs: [MotionVector; 2] },
    /// Two 8×16 partitions (left, right). 2 MVs.
    P8x16 { mvs: [MotionVector; 2] },
    /// Four 8×8 sub-macroblocks. Each has its own sub_mb_type that
    /// may introduce further sub-partitioning; Phase 6B.3.2c ships
    /// `SubMbChoice::P8x8` only (one MV per sub-MB).
    P8x8 { sub: [SubMbChoice; 4] },
}

impl PMbChoice {
    /// mb_type codeNum per spec Table 7-13 for P-slices.
    pub fn mb_type_codenum(self) -> u32 {
        match self {
            PMbChoice::P16x16 { .. } => 0,
            PMbChoice::P16x8 { .. } => 1,
            PMbChoice::P8x16 { .. } => 2,
            PMbChoice::P8x8 { .. } => 3,
        }
    }

    /// Spec `noSubMbPartSizeLessThan8x8Flag`. Returns true iff every
    /// partition is at least 8×8 (P_8×8 with P_L0_8×8 sub-choice counts;
    /// P_L0_8×4 / 4×8 / 4×4 do not). Controls whether
    /// `transform_size_8x8_flag` may be emitted for this MB.
    pub fn no_sub_mb_part_size_lt_8x8(&self) -> bool {
        match self {
            PMbChoice::P16x16 { .. } | PMbChoice::P16x8 { .. } | PMbChoice::P8x16 { .. } => true,
            PMbChoice::P8x8 { sub } => sub.iter().all(|s| matches!(s, SubMbChoice::P8x8 { .. })),
        }
    }
}

/// The encoder's resolved choice inside a single 8×8 sub-MB.
/// Variants list MVs in spec emit order (partition 0 first, etc.).
#[derive(Debug, Clone, Copy)]
pub enum SubMbChoice {
    /// One 8×8 partition, 1 MV.
    P8x8 { mv: MotionVector },
    /// Two 8×4 partitions (top, bottom). 2 MVs.
    P8x4 { mvs: [MotionVector; 2] },
    /// Two 4×8 partitions (left, right). 2 MVs.
    P4x8 { mvs: [MotionVector; 2] },
    /// Four 4×4 partitions (top-left, top-right, bottom-left,
    /// bottom-right). 4 MVs.
    P4x4 { mvs: [MotionVector; 4] },
}

impl SubMbChoice {
    /// sub_mb_type codeNum per spec Table 7-17.
    pub fn sub_mb_type_codenum(self) -> u32 {
        match self {
            SubMbChoice::P8x8 { .. } => 0,
            SubMbChoice::P8x4 { .. } => 1,
            SubMbChoice::P4x8 { .. } => 2,
            SubMbChoice::P4x4 { .. } => 3,
        }
    }
}

/// Overhead penalties for the sub_mb_type decision (SATD units,
/// first-pass; tunable).
pub const SUB_PENALTY_P8X4: u32 = 32;
pub const SUB_PENALTY_P4X8: u32 = 32;
pub const SUB_PENALTY_P4X4: u32 = 96;

/// Sub-MB origins in 4×4-block units within a 16×16 MB.
pub const SUB_MB_ORIGINS_4X4: [(usize, usize); 4] = [(0, 0), (2, 0), (0, 2), (2, 2)];
/// Sub-MB origins in 8-pixel luma units within a 16×16 MB.
pub const SUB_MB_ORIGINS_PX: [(u32, u32); 4] = [(0, 0), (8, 0), (0, 8), (8, 8)];

/// Run ME for every candidate partition and pick the cheapest.
///
/// `src_y` is the 16×16 source luma block (MB-aligned). `mb_x`,
/// `mb_y` are MB coordinates. `grid` supplies already-resolved
/// neighbor MVs for the median predictor.
pub fn decide_p_mb(
    src_y: &[[u8; 16]; 16],
    reference: &ReconFrame,
    me: &mut MotionEstimator,
    grid: &mut EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
) -> PMbChoice {
    decide_p_mb_with_cost(src_y, reference, me, grid, mb_x, mb_y).best
}

/// Full SATD+penalty decision over all 4 P-partition types. Records
/// every candidate + its cost so Phase C's RDO pass can top-K rerank
/// without rerunning ME.
///
/// Field order in `candidates` is fixed — index `i` matches the
/// ordering in `SATD_CANDIDATE_ORDER` for ergonomic lookups.
#[derive(Debug, Clone, Copy)]
pub struct PMbDecision {
    pub best: PMbChoice,
    pub best_cost: u32,
    pub candidates: [PMbChoice; 4],
    pub satd_costs: [u32; 4],
}

/// Ordering of the `candidates` array in [`PMbDecision`]:
/// coarsest-to-finest partition size, which is the standard
/// evaluation order for H.264 partition RDO (larger partitions are
/// cheaper to encode, so they're evaluated first to serve as early
/// reject baselines for finer partitions).
pub const SATD_CANDIDATE_ORDER: [&str; 4] = ["P16x16", "P16x8", "P8x16", "P8x8"];

/// Same as `decide_p_mb` but also returns the winning partition's
/// total cost (SATD + per-partition overhead penalty) and the full
/// SATD-cost list for all 4 candidates. The cost list enables
/// Phase C's MB-level RDO to run full D+λR on the top-K SATD
/// survivors instead of blindly picking the SATD winner.
///
/// Takes `&mut EncoderMvGrid` so the P_8x8 sub-MB path can
/// speculatively commit sub-MB winners into a scratch grid (so later
/// sub-MBs in raster order can read them as median-predictor
/// neighbors). The rectangle is snapshot-and-restored before return
/// so the caller's grid state is unchanged — the caller still commits
/// the actual winning partition's MVs exactly as before.
pub fn decide_p_mb_with_cost(
    src_y: &[[u8; 16]; 16],
    reference: &ReconFrame,
    me: &mut MotionEstimator,
    grid: &mut EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
) -> PMbDecision {
    let mb_px_x = (mb_x * 16) as u32;
    let mb_px_y = (mb_y * 16) as u32;
    let src_flat = src_y.as_flattened();

    // ── P_16x16 candidate ─────────────────────────────────────────
    let pred_16x16 = predict_mv_for_partition(grid, mb_x * 4, mb_y * 4, 4, 0);
    let cand_16x16 = build_me_candidates(grid, mb_x * 4, mb_y * 4, 4, pred_16x16);
    let r16 = me.search_block_with_candidates(
        src_flat, 16, reference, mb_px_x, mb_px_y, 16, 16, pred_16x16, &cand_16x16,
    );
    let cost_16x16 = r16.cost;

    // ── P_16x8 candidate: top half then bottom half ───────────────
    // For the bottom partition the left neighbor now includes the
    // top partition of the same MB — but since the mb_grid isn't
    // updated yet, we use the same predictor for both halves. This
    // is a simplification; the spec defines the neighbor lookup
    // per-partition, but for a first-pass decision the tiny
    // predictor drift is ignored.
    let src_top = extract_half(src_y, 0, 0, 16, 8);
    let src_bot = extract_half(src_y, 0, 8, 16, 8);
    let pred_top = predict_mv_for_partition(grid, mb_x * 4, mb_y * 4, 4, 0);
    let cand_top = build_me_candidates(grid, mb_x * 4, mb_y * 4, 4, pred_top);
    let r_top = me.search_block_with_candidates(
        &src_top, 16, reference, mb_px_x, mb_px_y, 16, 8, pred_top, &cand_top,
    );
    // For the bottom partition's predictor we ideally look up the
    // already-resolved top partition's MV. We approximate by taking
    // the top half's resolved MV as the predictor's A neighbour —
    // but since grid isn't mutated here, we use pred_top for now.
    let pred_bot = r_top.mv;
    // Bottom half's candidates read from the grid row 2 blocks down
    // (not strictly spec-correct since the partitioning model here
    // isn't 4×4-blocked, but the neighbors read are conservative).
    let cand_bot = build_me_candidates(grid, mb_x * 4, mb_y * 4 + 2, 4, pred_bot);
    let r_bot = me.search_block_with_candidates(
        &src_bot, 16, reference, mb_px_x, mb_px_y + 8, 16, 8, pred_bot, &cand_bot,
    );
    let cost_16x8 = r_top.cost.saturating_add(r_bot.cost).saturating_add(PENALTY_16X8);

    // ── P_8x16 candidate: left half then right half ───────────────
    let src_left = extract_half(src_y, 0, 0, 8, 16);
    let src_right = extract_half(src_y, 8, 0, 8, 16);
    let pred_left = predict_mv_for_partition(grid, mb_x * 4, mb_y * 4, 2, 0);
    let cand_left = build_me_candidates(grid, mb_x * 4, mb_y * 4, 2, pred_left);
    let r_left = me.search_block_with_candidates(
        &src_left, 8, reference, mb_px_x, mb_px_y, 8, 16, pred_left, &cand_left,
    );
    let pred_right = r_left.mv;
    let cand_right = build_me_candidates(grid, mb_x * 4 + 2, mb_y * 4, 2, pred_right);
    let r_right = me.search_block_with_candidates(
        &src_right, 8, reference, mb_px_x + 8, mb_px_y, 8, 16, pred_right, &cand_right,
    );
    let cost_8x16 = r_left.cost.saturating_add(r_right.cost).saturating_add(PENALTY_8X16);

    // ── P_8x8 candidate: four 8×8 sub-MBs ────────────────────────
    // Each sub-MB picks its own sub_mb_type (P_L0_8x8 / P_L0_8x4 /
    // P_L0_4x8 / P_L0_4x4) by SATD + fixed overhead penalty.
    //
    // To give each sub-MB a spec-correct median-of-neighbors predictor
    // (§ 8.4.1.3), we speculatively commit each sub-MB's winning MVs
    // into `grid` before moving to the next sub-MB in raster order.
    // The full rectangle is snapshot-and-restored around this block so
    // the caller's view of `grid` is unchanged; the caller still
    // commits the winning outer partition's MVs as before.
    let mb_mv_snap = grid.snapshot_mb(mb_x, mb_y);
    let mut sub = [SubMbChoice::P8x8 { mv: MotionVector::ZERO }; 4];
    let mut cost_8x8 = 0u32;
    for (i, &(off_x_px, off_y_px)) in SUB_MB_ORIGINS_PX.iter().enumerate() {
        let (dx_4x4, dy_4x4) = SUB_MB_ORIGINS_4X4[i];
        let sub_bx = mb_x * 4 + dx_4x4;
        let sub_by = mb_y * 4 + dy_4x4;
        let (sub_choice, sub_cost) = decide_sub_mb(
            src_y, reference, me, grid, mb_px_x, mb_px_y, off_x_px, off_y_px, sub_bx, sub_by,
        );
        commit_sub_mb_to_grid(grid, sub_bx, sub_by, &sub_choice);
        sub[i] = sub_choice;
        cost_8x8 = cost_8x8.saturating_add(sub_cost);
    }
    cost_8x8 = cost_8x8.saturating_add(PENALTY_8X8);
    grid.restore_mb(&mb_mv_snap);

    // Gather all 4 candidates in a fixed order so RDO can index by
    // partition type without re-running ME.
    let candidates = [
        PMbChoice::P16x16 { mv: r16.mv },
        PMbChoice::P16x8 { mvs: [r_top.mv, r_bot.mv] },
        PMbChoice::P8x16 { mvs: [r_left.mv, r_right.mv] },
        PMbChoice::P8x8 { sub },
    ];
    let mut satd_costs = [cost_16x16, cost_16x8, cost_8x16, cost_8x8];

    // ─── Phase E.2-LITE: inter psy bias ───
    // For each candidate, build its MC prediction and add
    //   strength × |hadamard_ac(src) − hadamard_ac(pred)| / 256
    // to the SATD+penalty cost. Biases selection toward MVs/partitions
    // whose reconstruction preserves source high-frequency energy.
    // Opt-in via PHASM_INTER_PSY_STRENGTH (default 0 = unchanged).
    let psy = inter_psy_strength();
    if psy != 0 {
        let src_ac = hadamard_ac_sum_16x16(src_y);
        for i in 0..4 {
            let pred_y = super::encoder::build_luma_prediction(
                reference, mb_x, mb_y, &candidates[i],
            );
            let pred_ac = hadamard_ac_sum_16x16(&pred_y);
            let ac_diff = (src_ac as i64 - pred_ac as i64).unsigned_abs() as u32;
            let bias = ((ac_diff as u64 * psy as u64) / 256) as u32;
            satd_costs[i] = satd_costs[i].saturating_add(bias);
        }
    }

    // Pick min by SATD+penalty (+psy if enabled).
    let mut best_idx = 0usize;
    for i in 1..4 {
        if satd_costs[i] < satd_costs[best_idx] {
            best_idx = i;
        }
    }
    PMbDecision {
        best: candidates[best_idx],
        best_cost: satd_costs[best_idx],
        candidates,
        satd_costs,
    }
}

/// Speculatively write a sub-MB's winning MVs into the 2×2 of 4×4
/// blocks that cover it. Mirrors the per-sub_mb_type geometry exactly
/// so downstream median-predictor lookups see what a spec-conformant
/// decoder would see at the same position.
fn commit_sub_mb_to_grid(
    grid: &mut EncoderMvGrid,
    sub_bx: usize,
    sub_by: usize,
    choice: &SubMbChoice,
) {
    match choice {
        SubMbChoice::P8x8 { mv } => grid.fill(sub_bx, sub_by, 2, 2, *mv, 0),
        SubMbChoice::P8x4 { mvs } => {
            grid.fill(sub_bx, sub_by, 2, 1, mvs[0], 0);
            grid.fill(sub_bx, sub_by + 1, 2, 1, mvs[1], 0);
        }
        SubMbChoice::P4x8 { mvs } => {
            grid.fill(sub_bx, sub_by, 1, 2, mvs[0], 0);
            grid.fill(sub_bx + 1, sub_by, 1, 2, mvs[1], 0);
        }
        SubMbChoice::P4x4 { mvs } => {
            grid.fill(sub_bx, sub_by, 1, 1, mvs[0], 0);
            grid.fill(sub_bx + 1, sub_by, 1, 1, mvs[1], 0);
            grid.fill(sub_bx, sub_by + 1, 1, 1, mvs[2], 0);
            grid.fill(sub_bx + 1, sub_by + 1, 1, 1, mvs[3], 0);
        }
    }
}

/// Copy a `w × h` sub-rectangle out of a 16×16 MB into a flat buffer
/// of stride `w`. Caller expects `w * h` bytes.
fn extract_half(
    src: &[[u8; 16]; 16],
    off_x: usize,
    off_y: usize,
    w: usize,
    h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for dy in 0..h {
        for dx in 0..w {
            out[dy * w + dx] = src[off_y + dy][off_x + dx];
        }
    }
    out
}

/// For a single 8×8 sub-MB at `(off_x_px, off_y_px)` within the MB,
/// enumerate all four sub_mb_types (P_L0_8x8, P_L0_8x4, P_L0_4x8,
/// P_L0_4x4) and return the cheapest choice + its total cost (with
/// overhead penalty already baked in).
#[allow(clippy::too_many_arguments)]
fn decide_sub_mb(
    src_y: &[[u8; 16]; 16],
    reference: &ReconFrame,
    me: &mut MotionEstimator,
    grid: &EncoderMvGrid,
    mb_px_x: u32,
    mb_px_y: u32,
    off_x_px: u32,
    off_y_px: u32,
    sub_bx: usize,
    sub_by: usize,
) -> (SubMbChoice, u32) {
    let sub_px_x = mb_px_x + off_x_px;
    let sub_px_y = mb_px_y + off_y_px;
    let off_x = off_x_px as usize;
    let off_y = off_y_px as usize;

    // ── P_L0_8x8: one 8×8 partition ──────────────────────────────
    //
    // Historical: the sub-MB median predictor was deferred because it
    // regressed enc-dec parity (4.14% Y pixel diff on IMG_4138 f1) by
    // pulling more MBs into the intra-in-P fallback, which had a
    // latent MV-predictor-availability bug (task #154).
    //
    // Task #154 RESOLVED 2026-04-21 (commit 0c9710f) — the MV
    // predictor now correctly distinguishes in-frame intra neighbours
    // from not-yet-decoded / off-frame positions via the
    // `EncoderMvGrid::decoded` field. 30f/90f enc-vs-dec parity is
    // 99.99 dB across the full GOP.
    //
    // `PHASM_SUBMB_MEDIAN_PRED=1` re-enables the spec median. The
    // parity gate is now clean, BUT on IMG_4138 30f the median
    // catastrophically blows up the bitstream without helping PSNR
    // (measured 2026-04-23):
    //   Q=40:  -2.89 dB avg  / size 21.9 MB vs baseline 0.84 MB  (+2500%!)
    //   Q=60:  -1.31 dB avg  / size 13.7 MB vs baseline 1.00 MB  (+1270%)
    //   Q=80:  -0.81 dB avg  / size 18.4 MB vs baseline 1.46 MB  (+1160%)
    // Root cause: hex search starts from the (content-inappropriate)
    // median predictor, finds MVs with huge MVDs, and the MVD bits
    // swamp any residual savings. Parity stays clean because MVDs are
    // bitstream-valid — just catastrophically expensive.
    // Kept as an env knob for isolated MV-predictor investigation,
    // off by default.
    //
    // 2026-04-23 re-measurement (Task #25 multi-pred ME default ON,
    // commit f82675d): 90f Q=26 delta is −0.02 dB (was −0.94 dB
    // before multi-pred), Q=80 delta is −0.01 dB. The catastrophic
    // bitstream explosion is gone — multi-pred ME now covers the
    // sub-MB median's chosen seed as one of its candidates, so hex
    // no longer starts from a bad place. The flip is no longer
    // blocked, but also no longer a clear win on PSNR alone.
    // Leaving env-gated pending bitrate measurement; A.2 ship
    // criterion per the quality plan is "net positive OR neutral
    // on R-D" not just PSNR.
    let pred_sub_8x8 = if std::env::var("PHASM_SUBMB_MEDIAN_PRED").ok().as_deref() == Some("1") {
        predict_mv_for_partition(grid, sub_bx, sub_by, 2, 0)
    } else {
        MotionVector::ZERO
    };
    let src_8x8 = extract_half(src_y, off_x, off_y, 8, 8);
    let cand_8x8 = build_me_candidates(grid, sub_bx, sub_by, 2, pred_sub_8x8);
    let r_8x8 = me.search_block_with_candidates(
        &src_8x8,
        8,
        reference,
        sub_px_x,
        sub_px_y,
        8,
        8,
        pred_sub_8x8,
        &cand_8x8,
    );
    let cost_p8x8 = r_8x8.cost;

    // ── P_L0_8x4: two 8×4 partitions, top then bottom ────────────
    let src_top = extract_half(src_y, off_x, off_y, 8, 4);
    let src_bot = extract_half(src_y, off_x, off_y + 4, 8, 4);
    let r_top = me.search_block(
        &src_top, 8, reference, sub_px_x, sub_px_y, 8, 4, r_8x8.mv,
    );
    let r_bot = me.search_block(
        &src_bot, 8, reference, sub_px_x, sub_px_y + 4, 8, 4, r_top.mv,
    );
    let cost_p8x4 = r_top
        .cost
        .saturating_add(r_bot.cost)
        .saturating_add(SUB_PENALTY_P8X4);

    // ── P_L0_4x8: two 4×8 partitions, left then right ────────────
    let src_left = extract_half(src_y, off_x, off_y, 4, 8);
    let src_right = extract_half(src_y, off_x + 4, off_y, 4, 8);
    let r_left = me.search_block(
        &src_left, 4, reference, sub_px_x, sub_px_y, 4, 8, r_8x8.mv,
    );
    let r_right = me.search_block(
        &src_right, 4, reference, sub_px_x + 4, sub_px_y, 4, 8, r_left.mv,
    );
    let cost_p4x8 = r_left
        .cost
        .saturating_add(r_right.cost)
        .saturating_add(SUB_PENALTY_P4X8);

    // ── P_L0_4x4: four 4×4 partitions, TL/TR/BL/BR ───────────────
    // Local offsets (in pixel units) within the 8×8 sub-MB:
    // (0,0), (4,0), (0,4), (4,4).
    let mut r_4x4 = [MotionVector::ZERO; 4];
    let mut cost_p4x4 = 0u32;
    let quarter_origins = [(0u32, 0u32), (4, 0), (0, 4), (4, 4)];
    for (qi, &(qx, qy)) in quarter_origins.iter().enumerate() {
        let src_q = extract_half(src_y, off_x + qx as usize, off_y + qy as usize, 4, 4);
        let start = if qi == 0 { r_8x8.mv } else { r_4x4[qi - 1] };
        let r = me.search_block(
            &src_q,
            4,
            reference,
            sub_px_x + qx,
            sub_px_y + qy,
            4,
            4,
            start,
        );
        r_4x4[qi] = r.mv;
        cost_p4x4 = cost_p4x4.saturating_add(r.cost);
    }
    cost_p4x4 = cost_p4x4.saturating_add(SUB_PENALTY_P4X4);

    // Pick min.
    let mut best_cost = cost_p8x8;
    let mut best = SubMbChoice::P8x8 { mv: r_8x8.mv };
    if cost_p8x4 < best_cost {
        best_cost = cost_p8x4;
        best = SubMbChoice::P8x4 { mvs: [r_top.mv, r_bot.mv] };
    }
    if cost_p4x8 < best_cost {
        best_cost = cost_p4x8;
        best = SubMbChoice::P4x8 { mvs: [r_left.mv, r_right.mv] };
    }
    if cost_p4x4 < best_cost {
        best_cost = cost_p4x4;
        best = SubMbChoice::P4x4 { mvs: r_4x4 };
    }
    (best, best_cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::reconstruction::ReconBuffer;

    fn build_ref(w: u32, h: u32, fill: impl Fn(u32, u32) -> u8) -> ReconFrame {
        let mut rb = ReconBuffer::new(w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                rb.y[(y * w + x) as usize] = fill(x, y);
            }
        }
        for v in rb.cb.iter_mut() {
            *v = 128;
        }
        for v in rb.cr.iter_mut() {
            *v = 128;
        }
        ReconFrame::snapshot(&rb)
    }

    /// Low-variance gradient content the partition-decision tests
    /// were tuned for: on this content, 4×4 sub-searches do no better
    /// than 16×16 so partition penalties decide. Works only when UMH
    /// is off (see `UmhOffGuard` below) — wider searches find aliased
    /// matches that change the decision.
    fn unique_content(x: u32, y: u32) -> u8 {
        ((x * 11 + y * 7) & 0xFF) as u8
    }

    /// These synthetic partition-decision tests use a 64×48 frame and
    /// rely on the ME converging tightly to a specific MV so partition
    /// costs line up with the hand-picked expected decision. UMH's
    /// wider cross+multi-hex pre-search finds equivalent-SATD matches
    /// at larger radii on small synthetic content, changing which
    /// partition ties out as cheapest. Disable UMH for the scope of
    /// the test. (This is test-harness scoping — the production path
    /// uses UMH by default and gets measured on real content.)
    struct UmhOffGuard;
    impl UmhOffGuard {
        fn new() -> Self {
            unsafe { std::env::set_var("PHASM_ME_UMH", "0"); }
            Self
        }
    }
    impl Drop for UmhOffGuard {
        fn drop(&mut self) {
            unsafe { std::env::remove_var("PHASM_ME_UMH"); }
        }
    }

    #[test]
    fn decide_prefers_16x16_on_uniform_motion() {
        let _g = UmhOffGuard::new();
        // Reference is a pattern; source is the pattern shifted by
        // +4 int pels horizontally — the same motion applies to every
        // 4×4 sub-block, so P_16x16 should be chosen.
        let reference = build_ref(64, 48, unique_content);
        let mut src = [[0u8; 16]; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                src[dy][dx] = reference.y_at(20 + dx as u32, 16 + dy as u32);
            }
        }
        let mut grid = EncoderMvGrid::new(4, 3);
        let mut me = MotionEstimator::new();
        let choice = decide_p_mb(&src, &reference, &mut me, &mut grid, 1, 1);
        assert!(
            matches!(choice, PMbChoice::P16x16 { .. }),
            "expected P16x16, got {choice:?}"
        );
    }

    #[test]
    fn decide_prefers_16x8_on_horizontal_stripe_motion() {
        let _g = UmhOffGuard::new();
        // Source: top half shifted by (+4, 0), bottom half unshifted.
        let reference = build_ref(64, 48, unique_content);
        let mut src = [[0u8; 16]; 16];
        for dy in 0..8 {
            for dx in 0..16 {
                // top half: reference at (x+4, y)
                src[dy][dx] = reference.y_at(16 + dx as u32 + 4, 16 + dy as u32);
            }
        }
        for dy in 8..16 {
            for dx in 0..16 {
                // bottom half: reference at (x, y)
                src[dy][dx] = reference.y_at(16 + dx as u32, 16 + dy as u32);
            }
        }
        let mut grid = EncoderMvGrid::new(4, 3);
        let mut me = MotionEstimator::new();
        let choice = decide_p_mb(&src, &reference, &mut me, &mut grid, 1, 1);
        assert!(
            matches!(choice, PMbChoice::P16x8 { .. }),
            "expected P16x8, got {choice:?}"
        );
    }

    #[test]
    fn sub_mb_type_codenums_match_spec_table_7_17() {
        assert_eq!(
            SubMbChoice::P8x8 { mv: MotionVector::ZERO }.sub_mb_type_codenum(),
            0
        );
        assert_eq!(
            SubMbChoice::P8x4 { mvs: [MotionVector::ZERO; 2] }.sub_mb_type_codenum(),
            1
        );
        assert_eq!(
            SubMbChoice::P4x8 { mvs: [MotionVector::ZERO; 2] }.sub_mb_type_codenum(),
            2
        );
        assert_eq!(
            SubMbChoice::P4x4 { mvs: [MotionVector::ZERO; 4] }.sub_mb_type_codenum(),
            3
        );
    }

    #[test]
    fn decide_prefers_p8x8_on_quadrant_motion() {
        let _g = UmhOffGuard::new();
        // Four quadrants moving independently — no two-partition split
        // (16×8 or 8×16) can match all four regions, so P_8x8 should
        // win once the fixed penalties clear.
        let reference = build_ref(64, 48, unique_content);
        let mut src = [[0u8; 16]; 16];
        for dy in 0..16i32 {
            for dx in 0..16i32 {
                // Quadrants of the source MB pull from different ref shifts.
                let (sx, sy) = match (dx < 8, dy < 8) {
                    (true, true) => (dx + 4, dy),  // TL: +4x
                    (false, true) => (dx, dy + 4), // TR: +4y
                    (true, false) => (dx, dy - 4), // BL: -4y
                    (false, false) => (dx - 4, dy), // BR: -4x
                };
                src[dy as usize][dx as usize] =
                    reference.y_at((16 + sx) as u32, (16 + sy) as u32);
            }
        }
        let mut grid = EncoderMvGrid::new(4, 3);
        let mut me = MotionEstimator::new();
        let choice = decide_p_mb(&src, &reference, &mut me, &mut grid, 1, 1);
        assert!(
            matches!(choice, PMbChoice::P8x8 { .. }),
            "expected P8x8, got {choice:?}"
        );
    }

    #[test]
    fn decide_prefers_8x16_on_vertical_stripe_motion() {
        let _g = UmhOffGuard::new();
        let reference = build_ref(64, 48, unique_content);
        let mut src = [[0u8; 16]; 16];
        for dy in 0..16 {
            for dx in 0..8 {
                // left half: reference at (x, y+4)
                src[dy][dx] = reference.y_at(16 + dx as u32, 16 + dy as u32 + 4);
            }
            for dx in 8..16 {
                // right half: reference at (x, y)
                src[dy][dx] = reference.y_at(16 + dx as u32, 16 + dy as u32);
            }
        }
        let mut grid = EncoderMvGrid::new(4, 3);
        let mut me = MotionEstimator::new();
        let choice = decide_p_mb(&src, &reference, &mut me, &mut grid, 1, 1);
        assert!(
            matches!(choice, PMbChoice::P8x16 { .. }),
            "expected P8x16, got {choice:?}"
        );
    }
}
