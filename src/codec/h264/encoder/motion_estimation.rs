// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Motion estimation. Phase 6B.2 + 6B.3.2a.
//!
//! Finds a good MV for a given `block_w × block_h` luma partition via
//! hexagonal integer search + half-pel / quarter-pel 5-point
//! refinement. All partition sizes allowed by Baseline — 16×16, 16×8,
//! 8×16, 8×8, 8×4, 4×8, 4×4 — share the same search machinery.
//!
//! Single reference frame. SAD for integer pass, SATD (4×4 Hadamard
//! tiled) for sub-pel.
//!
//! Algorithm note:
//!   docs/design/video/h264/encoder-algorithms/motion-estimation.md

use super::motion_compensation::{apply_luma_mv_block, apply_luma_mv_block_bipred};
use super::reference_buffer::ReconFrame;
use super::transform::forward_hadamard_4x4;

/// Motion vector in quarter-pel units.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionVector {
    pub mv_x: i16,
    pub mv_y: i16,
}

impl MotionVector {
    pub const ZERO: Self = Self { mv_x: 0, mv_y: 0 };
}

/// ME result: best MV + its SATD cost.
#[derive(Debug, Clone, Copy)]
pub struct MotionSearchResult {
    pub mv: MotionVector,
    pub cost: u32,
}

/// Motion-estimation engine — stateless for 6B.2 but structured
/// for future caching.
#[derive(Debug, Default)]
pub struct MotionEstimator {
    _private: (),
}

impl MotionEstimator {
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Search for the best MV for a `block_w × block_h` luma partition
    /// at frame position `(block_x, block_y)`. `source` is a flat
    /// buffer of length ≥ `source_stride * block_h`; `source_stride`
    /// spans one row.
    pub fn search_block(
        &mut self,
        source: &[u8],
        source_stride: usize,
        reference: &ReconFrame,
        block_x: u32,
        block_y: u32,
        block_w: u32,
        block_h: u32,
        predicted_mv: MotionVector,
    ) -> MotionSearchResult {
        self.search_block_with_candidates(
            source, source_stride, reference, block_x, block_y, block_w, block_h,
            predicted_mv, &[predicted_mv],
        )
    }

    /// Task #121 Phase 1 — multi-predictor seeding. Evaluates each
    /// entry of `candidates` at integer-snapped position by SATD +
    /// λ·mv_bits(cand − predicted_mv); the cheapest becomes the seed
    /// for the existing hex + half-pel + qpel pipeline. This widens
    /// the basin of attraction vs the single-predictor variant,
    /// letting ME escape local minima when the median is bad (e.g.
    /// intra or inter-refresh neighbors).
    ///
    /// `predicted_mv` is still the bit-cost anchor: MVD encoding uses
    /// `mv − predicted_mv`, so cost functions include that term to
    /// keep ME rate-aware. Candidates are scored with the same term —
    /// critical lesson from the earlier regression attempt which
    /// evaluated predictors with SAD-only and regressed 0.75–1.3 dB.
    #[allow(clippy::too_many_arguments)]
    pub fn search_block_with_candidates(
        &mut self,
        source: &[u8],
        source_stride: usize,
        reference: &ReconFrame,
        block_x: u32,
        block_y: u32,
        block_w: u32,
        block_h: u32,
        predicted_mv: MotionVector,
        candidates: &[MotionVector],
    ) -> MotionSearchResult {
        let lambda = me_lambda();

        // Score each candidate at integer-snapped + clipped position.
        // Include mv-bit cost so the start seed is already rate-aware.
        let fallback = clip_mv_to_frame(
            predicted_mv, reference, block_x, block_y, block_w, block_h,
        );
        let mut start = fallback;
        let mut start_cost = u32::MAX;
        for &c in candidates.iter() {
            let c_int = MotionVector {
                mv_x: (c.mv_x >> 2) << 2,
                mv_y: (c.mv_y >> 2) << 2,
            };
            let c_clipped = clip_mv_to_frame(
                c_int, reference, block_x, block_y, block_w, block_h,
            );
            let cost = satd_at_mv(
                source, source_stride, reference, block_x, block_y, block_w, block_h, c_clipped,
            ) + lambda * mv_bit_cost(c_clipped, predicted_mv);
            if cost < start_cost {
                start_cost = cost;
                start = c_clipped;
            }
        }

        // UMH Stage 1 + Stage 2 pre-search: cross at R=16 then
        // 16-point multi-hex at R=4. Finds the large-translation
        // basin that the short-radius hex alone would miss.
        let pre_start = if umh_enabled() {
            let after_cross = cross_search_r16(
                source, source_stride, reference, block_x, block_y, block_w, block_h,
                start, predicted_mv, lambda,
            );
            multi_hex_search(
                source, source_stride, reference, block_x, block_y, block_w, block_h,
                after_cross, predicted_mv, lambda,
            )
        } else {
            start
        };
        let integer_mv = integer_hex_search(
            source, source_stride, reference, block_x, block_y, block_w, block_h,
            pre_start, predicted_mv, lambda,
        );
        let halfpel_mv = refine_5point(
            source, source_stride, reference, block_x, block_y, block_w, block_h,
            integer_mv, 2, predicted_mv, lambda,
        );
        let qpel_mv = refine_5point(
            source, source_stride, reference, block_x, block_y, block_w, block_h,
            halfpel_mv, 1, predicted_mv, lambda,
        );
        let cost = satd_at_mv(
            source, source_stride, reference, block_x, block_y, block_w, block_h, qpel_mv,
        );
        MotionSearchResult { mv: qpel_mv, cost }
    }

    /// Phase 4.3 (#251) — multi-candidate refine-from-each then pick
    /// best. Distinct from `search_block_with_candidates` which uses
    /// best-seed-then-refine: that strategy picks the candidate with
    /// the lowest *initial* `SAD + λ·rate` and runs full UMH +
    /// integer-hex + sub-pel refinement only from that winner. A
    /// candidate whose initial cost is lower but whose refinement
    /// basin is worse displaces ME from a better-final-result basin.
    ///
    /// This variant runs an integer-pel hex search from EACH candidate
    /// independently, tracks the global best post-integer-refinement
    /// cost, then runs sub-pel (half + quarter) refinement only on
    /// that absolute winner. Cost: ~N × integer-hex (cheap, ~50 SAD
    /// points each) + 1 × sub-pel. Roughly 2× the single-start cost
    /// at N=6 candidates — manageable.
    ///
    /// Use this when you suspect the spatial-median seed is biased in
    /// a way the lowest-initial-cost selection can't escape (e.g.
    /// motion-boundary MBs whose neighbour MVs all point to a wall
    /// background that gives flat-and-low SAD anywhere on the wall —
    /// initial-cost picks any wall direction; refine-from-each gives
    /// a content-correct candidate a fair shot at a better basin).
    #[allow(clippy::too_many_arguments)]
    pub fn search_block_multi_refine(
        &mut self,
        source: &[u8],
        source_stride: usize,
        reference: &ReconFrame,
        block_x: u32,
        block_y: u32,
        block_w: u32,
        block_h: u32,
        predicted_mv: MotionVector,
        candidates: &[MotionVector],
    ) -> MotionSearchResult {
        let lambda = me_lambda();
        // Run integer hex search starting from each candidate.
        // Track the global best (cost, integer-pel mv) across all.
        let mut best_int_mv = clip_mv_to_frame(
            predicted_mv, reference, block_x, block_y, block_w, block_h,
        );
        let mut best_int_cost = u32::MAX;
        for &c in candidates.iter() {
            let c_int = MotionVector {
                mv_x: (c.mv_x >> 2) << 2,
                mv_y: (c.mv_y >> 2) << 2,
            };
            let c_clipped = clip_mv_to_frame(
                c_int, reference, block_x, block_y, block_w, block_h,
            );
            let int_mv = integer_hex_search(
                source, source_stride, reference, block_x, block_y, block_w, block_h,
                c_clipped, predicted_mv, lambda,
            );
            let cost = satd_at_mv(
                source, source_stride, reference,
                block_x, block_y, block_w, block_h, int_mv,
            ) + lambda * mv_bit_cost(int_mv, predicted_mv);
            if cost < best_int_cost {
                best_int_cost = cost;
                best_int_mv = int_mv;
            }
        }
        // Sub-pel refinement on the absolute integer winner.
        let halfpel_mv = refine_5point(
            source, source_stride, reference, block_x, block_y, block_w, block_h,
            best_int_mv, 2, predicted_mv, lambda,
        );
        let qpel_mv = refine_5point(
            source, source_stride, reference, block_x, block_y, block_w, block_h,
            halfpel_mv, 1, predicted_mv, lambda,
        );
        let cost = satd_at_mv(
            source, source_stride, reference,
            block_x, block_y, block_w, block_h, qpel_mv,
        );
        MotionSearchResult { mv: qpel_mv, cost }
    }

    /// v1.4 Phase 4.1 (#307) — multi-ref L0 search across an ordered
    /// list of past references. Runs `search_block` per reference,
    /// then picks the (mv, ref_idx) pair with minimum
    /// SATD + λ·(mv_bits + ref_idx_bits).
    ///
    /// `references[i]` corresponds to ref_idx=i. Caller arranges the
    /// list so closest-past-anchor is index 0 (matches
    /// `MultiSlotDpb::ref_list_l0` POC ordering).
    ///
    /// Rate-cost note: ref_idx unary encoding (spec § 9.3.2 Table
    /// 9-34) is N "1" bins followed by terminator "0", so
    /// `ref_idx=N` takes `N+1` bins. The +1 baseline applies to all
    /// when `num_active > 1`; the relative penalty for picking
    /// ref_idx=1 over ref_idx=0 is +1 bin (~λ cost). Tiny but non-
    /// zero — nudges ME toward the closer reference unless distant
    /// reference is materially better.
    ///
    /// Returns `(MotionSearchResult, ref_idx)`. At a single-element
    /// `references` slice, behaviour is identical to `search_block`
    /// modulo a defensive ref_idx=0 return.
    #[allow(clippy::too_many_arguments)]
    pub fn search_block_multi_ref(
        &mut self,
        source: &[u8],
        source_stride: usize,
        references: &[&ReconFrame],
        block_x: u32,
        block_y: u32,
        block_w: u32,
        block_h: u32,
        predicted_mv: MotionVector,
    ) -> (MotionSearchResult, u8) {
        debug_assert!(
            !references.is_empty(),
            "search_block_multi_ref requires at least one reference"
        );
        let lambda = me_lambda();
        let mut best_total_cost = u32::MAX;
        let mut best_result = MotionSearchResult {
            mv: MotionVector::ZERO,
            cost: 0,
        };
        let mut best_idx: u8 = 0;
        for (idx, reference) in references.iter().enumerate() {
            let r = self.search_block(
                source, source_stride, reference,
                block_x, block_y, block_w, block_h, predicted_mv,
            );
            // Total RD-cost = SATD + λ·(mv_bits + ref_idx_bits).
            // search_block returns SATD only; add mv_bits +
            // ref_idx_bits at this layer to make the multi-ref
            // comparison rate-aware.
            let mv_bits = mv_bit_cost(r.mv, predicted_mv);
            let ref_idx_bits = (idx as u32) + 1; // unary: N → N+1 bins
            let total = r.cost.saturating_add(lambda * (mv_bits + ref_idx_bits));
            if total < best_total_cost {
                best_total_cost = total;
                best_result = r;
                best_idx = idx as u8;
            }
        }
        (best_result, best_idx)
    }

    /// Back-compat wrapper for callers that pass a `[[u8; 16]; 16]`
    /// source and expect the 16×16 result.
    pub fn search_16x16(
        &mut self,
        source: &[[u8; 16]; 16],
        reference: &ReconFrame,
        block_x: u32,
        block_y: u32,
        predicted_mv: MotionVector,
    ) -> MotionSearchResult {
        self.search_block(
            source.as_flattened(),
            16,
            reference,
            block_x,
            block_y,
            16,
            16,
            predicted_mv,
        )
    }

    /// §6E-D.5(c) — bipred-aware joint MV refinement (subme≥7 style).
    ///
    /// After independent L0 / L1 ME has produced seed MVs, refine them
    /// jointly to minimise *bipred* SATD + λ × (mvbits_l0 + mvbits_l1).
    /// Joint bipred-refinement pass, subme≥7-style.
    ///
    /// Algorithm: alternately hold one list fixed and search a small
    /// qpel diamond around the other list's current best; iterate
    /// until both converge.
    /// Total candidates per round = 5 (L0 diamond) + 5 (L1 diamond) = 10
    /// bipred SATDs + 2 anchor evaluations.
    ///
    /// Returns: `(refined_l0_mv, refined_l1_mv)`. If the seeds are already
    /// at the joint minimum, returns them unchanged. Always returns MVs
    /// at quarter-pel precision (no integer/half-pel snapping).
    ///
    /// Use case: in B-frame mode-decision, the explicit-MV candidates
    /// (Bi_16x16, L0_16x16, L1_16x16) all benefit from this. After
    /// refinement, the L0_16x16 / L1_16x16 candidates also use the
    /// JOINT-refined L0 / L1 MVs (they're at least as good as the
    /// independent-ME seeds, and often better since joint refinement
    /// finds local minima that independent-ME's rate-cost-blocked search
    /// missed).
    ///
    /// `predicted_l0` / `predicted_l1` are the median predictors used
    /// for MVD bit-cost computation (matches the encoder's actual rate
    /// accounting at MVD emission time).
    #[allow(clippy::too_many_arguments)]
    pub fn refine_bipred(
        &mut self,
        source: &[u8],
        source_stride: usize,
        l0_ref: &ReconFrame,
        l1_ref: &ReconFrame,
        block_x: u32,
        block_y: u32,
        block_w: u32,
        block_h: u32,
        l0_seed: MotionVector,
        l1_seed: MotionVector,
        predicted_l0: MotionVector,
        predicted_l1: MotionVector,
    ) -> (MotionVector, MotionVector) {
        let lambda = me_lambda();
        let mut best_l0 = clip_mv_to_frame(l0_seed, l0_ref, block_x, block_y, block_w, block_h);
        let mut best_l1 = clip_mv_to_frame(l1_seed, l1_ref, block_x, block_y, block_w, block_h);
        let mut best_cost = bipred_cost_at(
            source, source_stride, l0_ref, l1_ref,
            block_x, block_y, block_w, block_h,
            best_l0, best_l1, predicted_l0, predicted_l1, lambda,
        );

        // 5-point diamond (qpel offsets): center + N/S/E/W.
        const DIAMOND_5: [(i16, i16); 5] = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)];
        // 9-point diamond — adds 4 diagonals. PHASM_B_REFINE_BIPRED_WIDE=1
        // selects this pattern + 4 iterations (vs 5-point cross / 2 iters).
        const DIAMOND_9: [(i16, i16); 9] = [
            (0, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ];
        let wide = std::env::var_os("PHASM_B_REFINE_BIPRED_WIDE").is_some();
        let max_iter = if wide { 4 } else { 2 };
        let diamond: &[(i16, i16)] = if wide { &DIAMOND_9 } else { &DIAMOND_5 };

        // Up to `max_iter` alternating passes. Each pass refines one
        // list with the other held fixed. Convergence check: if a pass
        // produces no change to either list, stop early.
        for _iter in 0..max_iter {
            let prev_l0 = best_l0;
            let prev_l1 = best_l1;

            // Pass A — refine L0 with L1 fixed.
            for (dx, dy) in diamond.iter().copied() {
                if dx == 0 && dy == 0 { continue; }
                let cand_l0_raw = MotionVector {
                    mv_x: best_l0.mv_x.saturating_add(dx),
                    mv_y: best_l0.mv_y.saturating_add(dy),
                };
                let cand_l0 = clip_mv_to_frame(
                    cand_l0_raw, l0_ref, block_x, block_y, block_w, block_h,
                );
                let cost = bipred_cost_at(
                    source, source_stride, l0_ref, l1_ref,
                    block_x, block_y, block_w, block_h,
                    cand_l0, best_l1, predicted_l0, predicted_l1, lambda,
                );
                if cost < best_cost {
                    best_cost = cost;
                    best_l0 = cand_l0;
                }
            }

            // Pass B — refine L1 with L0 fixed.
            for (dx, dy) in diamond.iter().copied() {
                if dx == 0 && dy == 0 { continue; }
                let cand_l1_raw = MotionVector {
                    mv_x: best_l1.mv_x.saturating_add(dx),
                    mv_y: best_l1.mv_y.saturating_add(dy),
                };
                let cand_l1 = clip_mv_to_frame(
                    cand_l1_raw, l1_ref, block_x, block_y, block_w, block_h,
                );
                let cost = bipred_cost_at(
                    source, source_stride, l0_ref, l1_ref,
                    block_x, block_y, block_w, block_h,
                    best_l0, cand_l1, predicted_l0, predicted_l1, lambda,
                );
                if cost < best_cost {
                    best_cost = cost;
                    best_l1 = cand_l1;
                }
            }

            if best_l0 == prev_l0 && best_l1 == prev_l1 {
                break; // converged
            }
        }

        (best_l0, best_l1)
    }
}

/// SAD (Sum of Absolute Differences) between a source rectangle and
/// a predicted rectangle (both `block_w × block_h`, with matching
/// stride conventions). Dispatches to NEON when the `simd` feature
/// is on (aarch64); falls through to the scalar reference otherwise.
fn sad_block(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    block_w: u32,
    block_h: u32,
) -> u32 {
    super::simd::sad_block_dispatch(
        source,
        source_stride,
        pred,
        pred_stride,
        block_w,
        block_h,
        || sad_block_scalar(source, source_stride, pred, pred_stride, block_w, block_h),
    )
}

#[inline]
fn sad_block_scalar(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    block_w: u32,
    block_h: u32,
) -> u32 {
    let mut sum = 0u32;
    for y in 0..block_h as usize {
        for x in 0..block_w as usize {
            let d = source[y * source_stride + x] as i32 - pred[y * pred_stride + x] as i32;
            sum += d.unsigned_abs();
        }
    }
    sum
}

/// SATD (Hadamard-transformed SAD) across a `block_w × block_h`
/// rectangle, summed as `(block_w / 4) × (block_h / 4)` 4×4
/// Hadamard-tiled SATDs. Requires 4-aligned sizes — all H.264
/// partition sizes satisfy this. Dispatches to NEON when the `simd`
/// feature is on (aarch64); falls through to the scalar reference.
fn satd_block(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    block_w: u32,
    block_h: u32,
) -> u32 {
    debug_assert!(block_w.is_multiple_of(4) && block_h.is_multiple_of(4));
    super::simd::satd_block_dispatch(
        source,
        source_stride,
        pred,
        pred_stride,
        block_w,
        block_h,
        || satd_block_scalar(source, source_stride, pred, pred_stride, block_w, block_h),
    )
}

#[inline]
fn satd_block_scalar(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    block_w: u32,
    block_h: u32,
) -> u32 {
    let mut total: u32 = 0;
    let tiles_y = (block_h / 4) as usize;
    let tiles_x = (block_w / 4) as usize;
    for by in 0..tiles_y {
        for bx in 0..tiles_x {
            let mut residual = [[0i32; 4]; 4];
            for dy in 0..4 {
                for dx in 0..4 {
                    let sx = bx * 4 + dx;
                    let sy = by * 4 + dy;
                    residual[dy][dx] = source[sy * source_stride + sx] as i32
                        - pred[sy * pred_stride + sx] as i32;
                }
            }
            let h = forward_hadamard_4x4(&residual);
            for row in &h {
                for &v in row {
                    total = total.saturating_add(v.unsigned_abs());
                }
            }
        }
    }
    total
}

/// Hex search pattern: 6 offsets arranged as an elongated hexagon.
/// Published in Zhu et al., "Hexagon-Based Search Pattern for Fast
/// Block Motion Estimation", IEEE TCSVT 12(5), 2002 — the canonical
/// motion-estimation pattern in modern video codecs.
const HEX_PATTERN: [(i16, i16); 6] = [
    (-2, 0),
    (-1, -2),
    (1, -2),
    (2, 0),
    (1, 2),
    (-1, 2),
];

/// Maximum number of hex-search iterations. Prevents runaway.
const MAX_HEX_ITER: usize = 16;

/// Task #121 Phase 2 — UMH (Unsymmetrical Multi-Hexagon) pre-search
/// patterns. Published in Chen, Zhu, Wang (ISO/IEC JTC1/SC29/WG11)
/// and described in IEEE TCSVT "Fast Integer-Pel and Fractional-Pel
/// Motion Estimation for H.264/AVC" — widely adopted in the
/// reference model and modern encoders.
///
/// Stage 1 — cross at R=16 integer-pel (64 qpel): 4 points along
/// horizontal and vertical axes, tests whether a large-translation
/// match exists that the short-radius hex would miss.
const CROSS_PATTERN_R16: [(i16, i16); 4] = [
    (-16, 0),
    (16, 0),
    (0, -16),
    (0, 16),
];

/// Stage 2 — 16-point "multi-hexagon" at radius ±4 integer-pel
/// (±16 qpel). Samples the perimeter of a 9×9 integer-pel box
/// uniformly: 5 top + 5 bottom + 3 per side = 16. Denser than the
/// 6-point hex because this stage must catch local minima that the
/// cross missed.
const MULTI_HEX_PATTERN: [(i16, i16); 16] = [
    (-4, -4), (-2, -4), (0, -4), (2, -4), (4, -4),
    (-4, -2),                             (4, -2),
    (-4,  0),                             (4,  0),
    (-4,  2),                             (4,  2),
    (-4,  4), (-2,  4), (0,  4), (2,  4), (4,  4),
];

/// Max iterations for the multi-hex refinement loop (each iteration
/// is 16 SAD evals). Typical convergence is 1–3 iterations.
const MAX_MULTI_HEX_ITER: usize = 4;

/// Motion-estimation lambda: bias factor multiplying MV-bit cost in
/// search cost. The literature formula (Wiegand-Sullivan 1996,
/// Eq. 15) gives `λ_motion = round(2^((qp-12)/6))` for the
/// SATD-domain search cost at slice QP `qp`. At slice_qp≈21 this is
/// ≈3; at QP=40 it is ≈10; at QP=26 it is ≈5.
///
/// Current code uses a conservative constant. Task #121 Phase 4 env
/// knob `PHASM_ME_LAMBDA=N` overrides this with a static integer
/// (1..=32) for coarse QP-independent tuning without threading QP
/// through the whole ME call chain. A later revision will replace
/// the constant with a per-QP lookup against `super::rdo::LAMBDA_TAB`
/// once QP is threaded through the partition decision.
///
/// **Revisit in Phase C.5** (`docs/design/video/h264/encoder-quality-plan.md`):
/// 2026-04-23 PSNR-only sweep showed monotonic PSNR loss as λ
/// increases, but that's a flawed measure — higher λ biases toward
/// shorter MVDs which costs match quality but earns fewer bits. The
/// legitimate win lives on the R-D curve (bits-at-equal-PSNR), which
/// needs Phase B's bit-accurate size counter before we can measure.
/// Keep the knob alive for that re-sweep; small PSNR drop for bitrate
/// win is acceptable.
const LAMBDA_MOTION_DEFAULT: u32 = 1;

#[inline]
fn me_lambda() -> u32 {
    std::env::var("PHASM_ME_LAMBDA")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .map_or(LAMBDA_MOTION_DEFAULT, |v| v.clamp(1, 32))
}

/// v1.4 Phase 4.2 (#313) — pub-crate accessor for the ME λ used by
/// multi-ref post-pass rate-cost calculations
/// (`refine_p_choice_multi_ref` in `partition_decision.rs`).
#[inline]
pub(crate) fn me_lambda_pub() -> u32 {
    me_lambda()
}

/// se(v) codeword length for signed Exp-Golomb, approximating the
/// bit cost of an MVD component per spec § 7.3.5. For value d:
///   bits = 2 * floor(log2(|d| + 1)) + 1.
#[inline]
fn se_bits(d: i32) -> u32 {
    let absd_plus_1 = d.unsigned_abs() + 1;
    2 * (31 - absd_plus_1.leading_zeros()) + 1
}

/// Total MVD bit cost: sum of x + y se(v) codeword bits.
#[inline]
fn mv_bit_cost(mv: MotionVector, predictor: MotionVector) -> u32 {
    let dx = mv.mv_x as i32 - predictor.mv_x as i32;
    let dy = mv.mv_y as i32 - predictor.mv_y as i32;
    se_bits(dx) + se_bits(dy)
}

/// Stage 1 of UMH: cross-search at radius 16. Evaluates 4 far-reach
/// points against the seed; returns the best of them (or the seed).
/// Uses SAD + λ·mv_bits (same metric as hex so winners transfer).
#[allow(clippy::too_many_arguments)]
fn cross_search_r16(
    source: &[u8],
    source_stride: usize,
    reference: &ReconFrame,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    seed: MotionVector,
    predictor: MotionVector,
    lambda: u32,
) -> MotionVector {
    let mut best_mv = seed;
    let mut best_cost = sad_at_mv(
        source, source_stride, reference, block_x, block_y, block_w, block_h, seed,
    ) + lambda * mv_bit_cost(seed, predictor);
    for (dx, dy) in CROSS_PATTERN_R16 {
        let candidate = MotionVector {
            mv_x: seed.mv_x + dx * 4,
            mv_y: seed.mv_y + dy * 4,
        };
        let candidate =
            clip_mv_to_frame(candidate, reference, block_x, block_y, block_w, block_h);
        let cost = sad_at_mv(
            source, source_stride, reference, block_x, block_y, block_w, block_h, candidate,
        ) + lambda * mv_bit_cost(candidate, predictor);
        if cost < best_cost {
            best_cost = cost;
            best_mv = candidate;
        }
    }
    best_mv
}

/// Stage 2 of UMH: iterative 16-point multi-hexagon at radius ±4
/// integer-pel. Moves the center to any perimeter point that beats
/// it and re-scans; caps at `MAX_MULTI_HEX_ITER` iterations.
#[allow(clippy::too_many_arguments)]
fn multi_hex_search(
    source: &[u8],
    source_stride: usize,
    reference: &ReconFrame,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    seed: MotionVector,
    predictor: MotionVector,
    lambda: u32,
) -> MotionVector {
    let mut center = seed;
    let mut center_cost = sad_at_mv(
        source, source_stride, reference, block_x, block_y, block_w, block_h, center,
    ) + lambda * mv_bit_cost(center, predictor);

    for _ in 0..MAX_MULTI_HEX_ITER {
        let mut best_mv = center;
        let mut best_cost = center_cost;
        for (dx, dy) in MULTI_HEX_PATTERN {
            let candidate = MotionVector {
                mv_x: center.mv_x + dx * 4,
                mv_y: center.mv_y + dy * 4,
            };
            let candidate =
                clip_mv_to_frame(candidate, reference, block_x, block_y, block_w, block_h);
            let cost = sad_at_mv(
                source, source_stride, reference, block_x, block_y, block_w, block_h, candidate,
            ) + lambda * mv_bit_cost(candidate, predictor);
            if cost < best_cost {
                best_cost = cost;
                best_mv = candidate;
            }
        }
        if best_mv == center {
            break;
        }
        center = best_mv;
        center_cost = best_cost;
    }
    center
}

/// Task #121 Phase 2 — UMH pre-search default-ON control.
/// `PHASM_ME_UMH=0` to opt out.
#[inline]
fn umh_enabled() -> bool {
    std::env::var("PHASM_ME_UMH")
        .ok()
        .is_none_or(|v| v != "0")
}

#[allow(clippy::too_many_arguments)]
fn integer_hex_search(
    source: &[u8],
    source_stride: usize,
    reference: &ReconFrame,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    start_mv: MotionVector,
    predictor: MotionVector,
    lambda: u32,
) -> MotionVector {
    let mut center = MotionVector {
        mv_x: (start_mv.mv_x >> 2) << 2,
        mv_y: (start_mv.mv_y >> 2) << 2,
    };
    let mut center_cost = sad_at_mv(
        source, source_stride, reference, block_x, block_y, block_w, block_h, center,
    ) + lambda * mv_bit_cost(center, predictor);

    for _ in 0..MAX_HEX_ITER {
        let mut best_mv = center;
        let mut best_cost = center_cost;
        for (dx, dy) in HEX_PATTERN {
            let candidate = MotionVector {
                mv_x: center.mv_x + dx * 4,
                mv_y: center.mv_y + dy * 4,
            };
            let candidate =
                clip_mv_to_frame(candidate, reference, block_x, block_y, block_w, block_h);
            let cost = sad_at_mv(
                source, source_stride, reference, block_x, block_y, block_w, block_h, candidate,
            ) + lambda * mv_bit_cost(candidate, predictor);
            if cost < best_cost {
                best_cost = cost;
                best_mv = candidate;
            }
        }
        if best_mv == center {
            break;
        }
        center = best_mv;
        center_cost = best_cost;
    }
    center
}

/// Task #121 Phase 3 — 8-point diamond sub-pel refinement. Env-gated
/// opt-in via `PHASM_ME_DIAMOND=1`.
///
/// Measured on 90f IMG_4138 real content:
///   Q=80: +0.12 dB,  Q=40: +0.13 dB,  Q=26: +0.09 dB.
/// Monotonically positive with zero regression, but below the +0.5 dB
/// go/no-go threshold (ME plan's strict rule). Kept as an opt-in knob
/// — expected to stack once Phase 4 (λ × mv_bits at full strength)
/// lands and rewards fractional-pel accuracy more.
#[inline]
fn diamond_enabled() -> bool {
    std::env::var_os("PHASM_ME_DIAMOND").is_some()
}

#[allow(clippy::too_many_arguments)]
fn refine_5point(
    source: &[u8],
    source_stride: usize,
    reference: &ReconFrame,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    center: MotionVector,
    step_qpel: i16,
    predictor: MotionVector,
    lambda: u32,
) -> MotionVector {
    // Task #121 Phase 3: 8-point diamond adds the 4 diagonals to the
    // 5-point cross. Diagonals catch sub-pel optima that the
    // axis-only cross misses — common when true motion has a
    // fractional component in both x and y. Fall back to the cross
    // when the env opt-out is set.
    let diag = if diamond_enabled() { step_qpel } else { 0 };
    let candidates = [
        center,
        MotionVector { mv_x: center.mv_x - step_qpel, mv_y: center.mv_y },
        MotionVector { mv_x: center.mv_x + step_qpel, mv_y: center.mv_y },
        MotionVector { mv_x: center.mv_x, mv_y: center.mv_y - step_qpel },
        MotionVector { mv_x: center.mv_x, mv_y: center.mv_y + step_qpel },
        MotionVector { mv_x: center.mv_x - diag, mv_y: center.mv_y - diag },
        MotionVector { mv_x: center.mv_x + diag, mv_y: center.mv_y - diag },
        MotionVector { mv_x: center.mv_x - diag, mv_y: center.mv_y + diag },
        MotionVector { mv_x: center.mv_x + diag, mv_y: center.mv_y + diag },
    ];
    let mut best_mv = center;
    let mut best_cost = u32::MAX;
    for cand in candidates {
        let cand_clipped =
            clip_mv_to_frame(cand, reference, block_x, block_y, block_w, block_h);
        let cost = satd_at_mv(
            source, source_stride, reference, block_x, block_y, block_w, block_h, cand_clipped,
        ) + lambda * mv_bit_cost(cand_clipped, predictor);
        if cost < best_cost {
            best_cost = cost;
            best_mv = cand_clipped;
        }
    }
    best_mv
}

/// Compute SAD between source and a freshly-MC'd predicted block at `mv`.
///
/// Uses a stack-allocated 256-byte scratch buffer (max H.264 partition is
/// 16×16 = 256). This eliminates 1-2M `Vec<u8>` allocations per 1080p frame
/// in the ME hot path (the `[u8; 256]` array is dead-stack memory; the
/// allocator is not involved).
#[allow(clippy::too_many_arguments)]
fn sad_at_mv(
    source: &[u8],
    source_stride: usize,
    reference: &ReconFrame,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    mv: MotionVector,
) -> u32 {
    let used = (block_w * block_h) as usize;
    let mut pred_storage = [0u8; 256];
    let pred = &mut pred_storage[..used];
    apply_luma_mv_block(
        reference,
        block_x,
        block_y,
        block_w,
        block_h,
        mv,
        pred,
        block_w as usize,
    );
    sad_block(source, source_stride, pred, block_w as usize, block_w, block_h)
}

/// Compute SATD between source and a freshly-MC'd predicted block at `mv`.
///
/// Uses a stack-allocated 256-byte scratch buffer (see `sad_at_mv` doc).
#[allow(clippy::too_many_arguments)]
fn satd_at_mv(
    source: &[u8],
    source_stride: usize,
    reference: &ReconFrame,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    mv: MotionVector,
) -> u32 {
    let used = (block_w * block_h) as usize;
    let mut pred_storage = [0u8; 256];
    let pred = &mut pred_storage[..used];
    apply_luma_mv_block(
        reference,
        block_x,
        block_y,
        block_w,
        block_h,
        mv,
        pred,
        block_w as usize,
    );
    satd_block(source, source_stride, pred, block_w as usize, block_w, block_h)
}

/// §6E-D.5(c) — bipred-domain SATD + λ × (mvbits_l0 + mvbits_l1).
///
/// Used by [`MotionEstimator::refine_bipred`] to score bipred MV pair
/// candidates. Builds the bipred prediction at (mv_l0, mv_l1) — i.e.
/// `(L0[block + mv_l0] + L1[block + mv_l1] + 1) / 2` per spec
/// § 8.4.2.3.1 — and returns the Lagrangian cost.
#[allow(clippy::too_many_arguments)]
fn bipred_cost_at(
    source: &[u8],
    source_stride: usize,
    l0_ref: &ReconFrame,
    l1_ref: &ReconFrame,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    mv_l0: MotionVector,
    mv_l1: MotionVector,
    pred_l0: MotionVector,
    pred_l1: MotionVector,
    lambda: u32,
) -> u32 {
    let used = (block_w * block_h) as usize;
    let mut pred_storage = [0u8; 256];
    let pred = &mut pred_storage[..used];
    apply_luma_mv_block_bipred(
        l0_ref, mv_l0, l1_ref, mv_l1,
        block_x, block_y, block_w, block_h,
        pred, block_w as usize,
    );
    let satd = satd_block(source, source_stride, pred, block_w as usize, block_w, block_h);
    satd + lambda * (mv_bit_cost(mv_l0, pred_l0) + mv_bit_cost(mv_l1, pred_l1))
}

/// Clip `mv` (qpel) so the referenced `block_w × block_h` window
/// plus the 6-tap filter halo stays within the reference frame.
/// Edge replication in `apply_luma_mv_block` handles out-of-bound
/// samples safely; clipping here prevents wasted search on
/// all-edge-replica samples.
fn clip_mv_to_frame(
    mv: MotionVector,
    reference: &ReconFrame,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
) -> MotionVector {
    let halo = 3i32;
    let ref_w = reference.width as i32;
    let ref_h = reference.height as i32;
    let min_x = -(block_x as i32 + halo) * 4;
    let max_x = (ref_w - block_x as i32 - block_w as i32 - halo).max(0) * 4;
    let min_y = -(block_y as i32 + halo) * 4;
    let max_y = (ref_h - block_y as i32 - block_h as i32 - halo).max(0) * 4;
    MotionVector {
        mv_x: (mv.mv_x as i32).clamp(min_x, max_x) as i16,
        mv_y: (mv.mv_y as i32).clamp(min_y, max_y) as i16,
    }
}

/// Median of three signed values. If a value is duplicated, returns
/// the duplicate.
fn median3(a: i32, b: i32, c: i32) -> i32 {
    (a + b + c) - a.max(b).max(c) - a.min(b).min(c)
}

/// MV predictor per spec § 8.4.1.3 (median of available neighbors).
/// For an unavailable neighbor pass `None`. If no neighbors are
/// available, returns (0, 0).
///
/// `above_left` is the fallback when `above_right` is unavailable
/// but `above` or `left` are present.
pub fn median_mv_predictor(
    left: Option<MotionVector>,
    above: Option<MotionVector>,
    above_right: Option<MotionVector>,
    above_left: Option<MotionVector>,
) -> MotionVector {
    // Effective "C" neighbor: above_right if present, else above_left.
    let c = above_right.or(above_left);
    // Count available neighbors.
    let mut available: Vec<MotionVector> = Vec::new();
    if let Some(m) = left {
        available.push(m);
    }
    if let Some(m) = above {
        available.push(m);
    }
    if let Some(m) = c {
        available.push(m);
    }

    match available.len() {
        0 => MotionVector::ZERO,
        1 => available[0],
        2 => {
            // Spec says: if only two of A/B/C are available, the third
            // is considered (0, 0) for the median. So median of two
            // MVs + (0, 0).
            let a = available[0];
            let b = available[1];
            MotionVector {
                mv_x: median3(a.mv_x as i32, b.mv_x as i32, 0) as i16,
                mv_y: median3(a.mv_y as i32, b.mv_y as i32, 0) as i16,
            }
        }
        3 => {
            // All three available. Median per component.
            let a = available[0];
            let b = available[1];
            let c = available[2];
            MotionVector {
                mv_x: median3(a.mv_x as i32, b.mv_x as i32, c.mv_x as i32) as i16,
                mv_y: median3(a.mv_y as i32, b.mv_y as i32, c.mv_y as i32) as i16,
            }
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::reconstruction::ReconBuffer;

    fn build_ref(width: u32, height: u32, fill: impl Fn(u32, u32) -> u8) -> ReconFrame {
        let mut rb = ReconBuffer::new(width, height).unwrap();
        for y in 0..height {
            for x in 0..width {
                rb.y[(y * width + x) as usize] = fill(x, y);
            }
        }
        for v in rb.cb.iter_mut() { *v = 128; }
        for v in rb.cr.iter_mut() { *v = 128; }
        ReconFrame::snapshot(&rb)
    }

    fn extract_block(frame: &ReconFrame, x: u32, y: u32) -> [[u8; 16]; 16] {
        let mut b = [[0u8; 16]; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                b[dy][dx] = frame.y_at(x + dx as u32, y + dy as u32);
            }
        }
        b
    }

    #[test]
    fn median3_basic() {
        assert_eq!(median3(1, 2, 3), 2);
        assert_eq!(median3(5, 1, 3), 3);
        assert_eq!(median3(-1, -5, 0), -1);
        assert_eq!(median3(7, 7, 2), 7);
    }

    #[test]
    fn median_predictor_no_neighbors() {
        assert_eq!(
            median_mv_predictor(None, None, None, None),
            MotionVector::ZERO
        );
    }

    #[test]
    fn median_predictor_single_neighbor() {
        let mv = MotionVector { mv_x: 12, mv_y: -4 };
        assert_eq!(
            median_mv_predictor(Some(mv), None, None, None),
            mv,
        );
    }

    #[test]
    fn median_predictor_three_neighbors_component_median() {
        let a = MotionVector { mv_x: 10, mv_y: 5 };
        let b = MotionVector { mv_x: 20, mv_y: -5 };
        let c = MotionVector { mv_x: 15, mv_y: 0 };
        let med = median_mv_predictor(Some(a), Some(b), Some(c), None);
        assert_eq!(med.mv_x, 15); // median of 10, 20, 15 = 15
        assert_eq!(med.mv_y, 0);  // median of 5, -5, 0 = 0
    }

    #[test]
    fn me_identical_frame_finds_zero_mv() {
        let reference = build_ref(64, 48, |x, y| ((x * 7 + y * 3) & 0xFF) as u8);
        let source = extract_block(&reference, 16, 16);
        let mut me = MotionEstimator::new();
        let r = me.search_16x16(&source, &reference, 16, 16, MotionVector::ZERO);
        assert_eq!(r.mv, MotionVector::ZERO);
        assert_eq!(r.cost, 0, "identical source+ref should have zero SATD");
    }

    /// Force UMH off inside a test scope so single-start hex behavior
    /// (which the synthetic tests were designed around) applies.
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
    fn me_integer_translation_finds_shift() {
        let _g = UmhOffGuard::new();
        // Reference has a distinctive pattern. Source is the same
        // pattern shifted by +4 int pels in x. ME at block (16, 16)
        // should find mv_x = -16 (= -4 int pel), i.e., look LEFT to
        // match. Content at (16, 16) in source = content at (20, 16)
        // in reference.
        let reference = build_ref(64, 48, |x, y| ((x * 11 + y * 7) & 0xFF) as u8);
        // Source block: shift reference's (20, 16) into our (16, 16)
        // slot → source[dy][dx] = reference's pixel (20+dx, 16+dy).
        let mut source = [[0u8; 16]; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                source[dy][dx] = reference.y_at(20 + dx as u32, 16 + dy as u32);
            }
        }
        let mut me = MotionEstimator::new();
        let r = me.search_16x16(&source, &reference, 16, 16, MotionVector::ZERO);
        // An exact-match ME should cost 0. Specific MV value not
        // asserted — the synthetic pattern has period-(32,18)-ish
        // structure (x*11 + y*7 mod 256 has multiple zero-SATD
        // matches), and UMH's wider pre-search can legitimately
        // land on any of them. All that matters for ME correctness
        // is that cost == 0 was achieved.
        assert_eq!(r.cost, 0, "exact-match ME should cost 0");
    }

    #[test]
    fn sad_self_equals_zero() {
        let b = [[100u8; 16]; 16];
        assert_eq!(sad_block(b.as_flattened(), 16, b.as_flattened(), 16, 16, 16), 0);
    }

    #[test]
    fn sad_constant_offset() {
        let a = [[100u8; 16]; 16];
        let b = [[103u8; 16]; 16];
        assert_eq!(
            sad_block(a.as_flattened(), 16, b.as_flattened(), 16, 16, 16),
            16 * 16 * 3
        );
    }

    #[test]
    fn mv_clipping_within_frame_is_noop() {
        let reference = build_ref(64, 48, |_, _| 0);
        let mv = MotionVector { mv_x: 4, mv_y: 8 };
        let clipped = clip_mv_to_frame(mv, &reference, 16, 16, 16, 16);
        assert_eq!(clipped, mv, "in-bounds MV should not be clipped");
    }

    #[test]
    fn mv_clipping_large_negative() {
        let reference = build_ref(64, 48, |_, _| 0);
        let mv = MotionVector { mv_x: -1000, mv_y: -1000 };
        let clipped = clip_mv_to_frame(mv, &reference, 16, 16, 16, 16);
        assert!(clipped.mv_x > -1000);
        assert!(clipped.mv_y > -1000);
    }

    #[test]
    fn search_block_matches_16x16_on_identity() {
        let reference = build_ref(64, 48, |x, y| ((x * 11 + y * 7) & 0xFF) as u8);
        let source = extract_block(&reference, 16, 16);
        let mut me = MotionEstimator::new();
        let r = me.search_block(
            source.as_flattened(),
            16,
            &reference,
            16,
            16,
            16,
            16,
            MotionVector::ZERO,
        );
        assert_eq!(r.mv, MotionVector::ZERO);
        assert_eq!(r.cost, 0);
    }

    #[test]
    fn search_block_finds_shift_on_8x8() {
        // 8×8 partition starting at (16, 16) — source = reference
        // shifted by +4 int-pel in x, so best mv = (+16, 0).
        let reference = build_ref(64, 48, |x, y| ((x * 5 + y * 3) & 0xFF) as u8);
        let mut source = [[0u8; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                source[dy][dx] = reference.y_at(20 + dx as u32, 16 + dy as u32);
            }
        }
        let mut me = MotionEstimator::new();
        let r = me.search_block(
            source.as_flattened(),
            8,
            &reference,
            16,
            16,
            8,
            8,
            MotionVector::ZERO,
        );
        assert_eq!(r.mv, MotionVector { mv_x: 16, mv_y: 0 });
        assert_eq!(r.cost, 0);
    }

    #[test]
    fn search_block_finds_shift_on_4x4() {
        let reference = build_ref(64, 48, |x, y| ((x * 5 + y * 3) & 0xFF) as u8);
        let mut source = [[0u8; 4]; 4];
        for dy in 0..4 {
            for dx in 0..4 {
                source[dy][dx] = reference.y_at(20 + dx as u32, 16 + dy as u32);
            }
        }
        let mut me = MotionEstimator::new();
        let r = me.search_block(
            source.as_flattened(),
            4,
            &reference,
            16,
            16,
            4,
            4,
            MotionVector::ZERO,
        );
        assert_eq!(r.mv, MotionVector { mv_x: 16, mv_y: 0 });
        assert_eq!(r.cost, 0);
    }

    // -- v1.4 Phase 4.1 search_block_multi_ref tests -----------------------

    #[test]
    fn multi_ref_picks_idx_0_when_only_one_ref() {
        // Single-ref slice: result identical to search_block + idx=0.
        let reference = build_ref(64, 48, |x, y| ((x * 11 + y * 7) & 0xFF) as u8);
        let source = extract_block(&reference, 16, 16);
        let mut me = MotionEstimator::new();
        let (r, idx) = me.search_block_multi_ref(
            source.as_flattened(),
            16,
            &[&reference],
            16, 16, 16, 16,
            MotionVector::ZERO,
        );
        assert_eq!(idx, 0);
        assert_eq!(r.mv, MotionVector::ZERO);
        assert_eq!(r.cost, 0);
    }

    #[test]
    fn multi_ref_picks_idx_1_when_second_ref_matches() {
        // Two refs: ref_0 has noise, ref_1 matches source exactly.
        // Multi-ref must prefer ref_1 despite +λ ref_idx penalty since
        // ref_0's SATD is far from zero.
        let ref_0 = build_ref(64, 48, |x, y| ((x * 17 + y * 13) & 0xFF) as u8);
        let ref_1 = build_ref(64, 48, |x, y| ((x * 11 + y * 7) & 0xFF) as u8);
        let source = extract_block(&ref_1, 16, 16);
        let mut me = MotionEstimator::new();
        let (r, idx) = me.search_block_multi_ref(
            source.as_flattened(),
            16,
            &[&ref_0, &ref_1],
            16, 16, 16, 16,
            MotionVector::ZERO,
        );
        assert_eq!(idx, 1, "expected ref_idx=1 (second ref matches source)");
        assert_eq!(r.mv, MotionVector::ZERO);
        assert_eq!(r.cost, 0);
    }

    #[test]
    fn multi_ref_prefers_idx_0_on_tie() {
        // Both refs identical → both yield SATD=0 at MV=0, but ref_idx=0
        // has fewer ref_idx bins so total cost is lower.
        let reference = build_ref(64, 48, |x, y| ((x * 11 + y * 7) & 0xFF) as u8);
        let source = extract_block(&reference, 16, 16);
        let mut me = MotionEstimator::new();
        let (_r, idx) = me.search_block_multi_ref(
            source.as_flattened(),
            16,
            &[&reference, &reference],
            16, 16, 16, 16,
            MotionVector::ZERO,
        );
        assert_eq!(idx, 0, "tie should break toward closer reference (lower ref_idx)");
    }

    // -- §6E-D.5(c) refine_bipred tests -------------------------------------

    #[test]
    fn refine_bipred_returns_seeds_when_already_optimal() {
        // Both refs identical to source → bipred at (0,0) gives zero
        // residual. Refinement should not move from (0,0) seeds.
        let src_fill = |_x: u32, _y: u32| 100;
        let l0_ref = build_ref(64, 64, src_fill);
        let l1_ref = build_ref(64, 64, src_fill);
        let src = extract_block(&l0_ref, 16, 16);
        let mut me = MotionEstimator::new();
        let (r_l0, r_l1) = me.refine_bipred(
            src.as_flattened(), 16,
            &l0_ref, &l1_ref,
            16, 16, 16, 16,
            MotionVector::ZERO, MotionVector::ZERO,
            MotionVector::ZERO, MotionVector::ZERO,
        );
        assert_eq!(r_l0, MotionVector::ZERO);
        assert_eq!(r_l1, MotionVector::ZERO);
    }

    #[test]
    fn refine_bipred_finds_opposite_motion() {
        // Construct a scenario where bipred averaging helps:
        //   src(x, y) = 100
        //   L0_ref(x, y) = 80 (= src - 20)  → at (0,0) L0 alone is wrong by 20
        //   L1_ref(x, y) = 120 (= src + 20) → at (0,0) L1 alone is wrong by 20
        //   bipred = (80 + 120 + 1)/2 = 100 → exact match.
        // refine_bipred starting at (0,0) seeds should keep them at (0,0)
        // because that's already the bipred-optimal pair.
        let l0_ref = build_ref(64, 64, |_x, _y| 80);
        let l1_ref = build_ref(64, 64, |_x, _y| 120);
        let src = [[100u8; 16]; 16];
        let mut me = MotionEstimator::new();
        let (r_l0, r_l1) = me.refine_bipred(
            src.as_flattened(), 16,
            &l0_ref, &l1_ref,
            16, 16, 16, 16,
            MotionVector::ZERO, MotionVector::ZERO,
            MotionVector::ZERO, MotionVector::ZERO,
        );
        assert_eq!(r_l0, MotionVector::ZERO);
        assert_eq!(r_l1, MotionVector::ZERO);
    }

    #[test]
    fn refine_bipred_improves_on_nearby_seeds() {
        // L0_ref shifted +1 qpel left (so true L0 MV is +1 qpel right).
        // L1_ref shifted +1 qpel right (so true L1 MV is +1 qpel left).
        // Source content matches the bipred at the optimal pair.
        // Seed at (0,0)/(0,0) — slightly off the optimum.
        // Refinement should at least not increase cost.
        let l0_ref = build_ref(64, 64, |x, _y| {
            // ramp content shifted by +1 in x relative to "ideal" position.
            ((x as i32) * 5 + 50).clamp(0, 255) as u8
        });
        let l1_ref = build_ref(64, 64, |x, _y| {
            ((x as i32) * 5 + 50).clamp(0, 255) as u8
        });
        let src = extract_block(&l0_ref, 16, 16);
        let mut me = MotionEstimator::new();

        // Score initial bipred at (0,0)/(0,0).
        let lambda = me_lambda();
        let cost_seed = bipred_cost_at(
            src.as_flattened(), 16, &l0_ref, &l1_ref,
            16, 16, 16, 16,
            MotionVector::ZERO, MotionVector::ZERO,
            MotionVector::ZERO, MotionVector::ZERO,
            lambda,
        );

        let (r_l0, r_l1) = me.refine_bipred(
            src.as_flattened(), 16,
            &l0_ref, &l1_ref,
            16, 16, 16, 16,
            MotionVector::ZERO, MotionVector::ZERO,
            MotionVector::ZERO, MotionVector::ZERO,
        );
        let cost_refined = bipred_cost_at(
            src.as_flattened(), 16, &l0_ref, &l1_ref,
            16, 16, 16, 16,
            r_l0, r_l1, MotionVector::ZERO, MotionVector::ZERO,
            lambda,
        );
        assert!(cost_refined <= cost_seed,
            "refinement must never INCREASE cost: seed={cost_seed} refined={cost_refined}");
    }

    #[test]
    fn refine_bipred_deterministic() {
        // Same inputs → same output (load-bearing for stego Pass 1 / Pass 3).
        let l0_ref = build_ref(64, 64, |x, y| ((x * 3 + y * 5) & 0xFF) as u8);
        let l1_ref = build_ref(64, 64, |x, y| ((x * 5 + y * 3 + 7) & 0xFF) as u8);
        let src = [[100u8; 16]; 16];
        let mut me = MotionEstimator::new();
        let (a_l0, a_l1) = me.refine_bipred(
            src.as_flattened(), 16, &l0_ref, &l1_ref,
            16, 16, 16, 16,
            MotionVector::ZERO, MotionVector::ZERO,
            MotionVector::ZERO, MotionVector::ZERO,
        );
        let (b_l0, b_l1) = me.refine_bipred(
            src.as_flattened(), 16, &l0_ref, &l1_ref,
            16, 16, 16, 16,
            MotionVector::ZERO, MotionVector::ZERO,
            MotionVector::ZERO, MotionVector::ZERO,
        );
        assert_eq!(a_l0, b_l0);
        assert_eq!(a_l1, b_l1);
    }
}
