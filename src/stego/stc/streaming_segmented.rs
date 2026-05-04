// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 6E-C / Task #24 — Streaming-segmented Viterbi STC.
//!
//! Generalizes `stc_embed_segmented` from "in-memory cover slice"
//! to "on-demand cover fetch via callback". Memory bound is
//! `O(num_segments × num_states × 8 B + segment_size × 16 B)`
//! instead of `O(n)` cover materialization.
//!
//! ## When to use
//!
//! - **Long-clip video stego** (15+ minutes 1080p) where the
//!   per-domain cover bit count exceeds practical RAM
//!   (~16 billion positions × 9 bytes = ~145 GB).
//! - **Multi-shadow encoder integration** (deferred to a
//!   follow-on task): collapsing the §30D-C 3-pass MVD/residual
//!   split into a single primary STC pass requires the encoder
//!   driver to access cover via a per-GOP fetch callback rather
//!   than a pre-materialized slice. Streaming-segmented Viterbi
//!   is the natural primitive on the STC side of that integration.
//!
//! ## Status (this commit — scaffold only)
//!
//! Documented, type-stub level. The full implementation requires:
//! 1. Refactoring `stc_embed_segmented` to accept a `CoverFetch`
//!    callback in place of the slice arguments.
//! 2. An encoder-side adapter that runs Pass 1 per GOP via §6E-C0
//!    streaming walker and feeds the result into the callback.
//! 3. An encoder-side adapter that re-runs Pass 1 per GOP for
//!    Phase B's recompute step.
//!
//! The actual work is broken out as separate tasks:
//! - **Task #24.1**: refactor `stc_embed_segmented` to callback API.
//! - **Task #24.2**: encoder per-GOP replay adapter.
//! - **Task #24.3**: integrate into the §30D-C orchestrator
//!   (or into a new unified primary STC encoder if the §30D-C
//!   split is collapsed at that point).
//!
//! ## Architectural notes
//!
//! The §30D-C 3-pass split (Pass 2A MVD STC → Pass 1B → Pass 2B
//! residual STC) is **orthogonal** to the streaming-Viterbi memory
//! bound. Streaming-Viterbi makes the in-memory STC plan use
//! O(√n) rather than O(n) memory; the 3-pass orchestration
//! structure is unchanged.
//!
//! The §30D-C 3-pass collapse — which would benefit multi-shadow
//! N>1 by giving a single unified position list — is a **separate**
//! architectural change. It requires either:
//!
//! - **Decision cache** (Phase 6D Option A): cache full per-MB
//!   encoder state so the planner can simulate post-MVD-plan
//!   residuals without running Pass 1B. Heavy refactor; ~30 GB
//!   cache for 15-min 1080p.
//! - **Iteration to fixed point**: Pass 1 → tentative plan → Pass 3
//!   dryrun → re-plan → ... until convergence. 4-7× single-encode
//!   wall-clock per iteration.
//!
//! Both rejected at session-design time on cost/complexity grounds.
//! Multi-shadow N>1 stays bounded by these architectural limits
//! until a future session prioritizes the unification refactor.

use crate::stego::stc::embed::{EmbedResult, STC_PROGRESS_STEPS};
use crate::stego::stc::extract::stc_extract;
use crate::stego::stc::hhat;
use crate::stego::progress;

/// Cover-fetch callback signature for streaming-segmented STC.
///
/// `get_segment(seg_idx) -> (bits, costs)` returns the cover bits
/// and costs for segment `seg_idx`. Each segment is `K × w` cover
/// positions where `K` is the segment size in message blocks.
///
/// The callback may be invoked multiple times for the same
/// `seg_idx` (Phase A and Phase B both visit each segment).
/// Implementations must return identical data on repeated calls
/// for correctness.
pub trait CoverFetch {
    /// Total cover position count `n`.
    fn total_positions(&self) -> usize;

    /// Number of segments. Equals `m.div_ceil(K)` where `K` is
    /// the segment size.
    fn num_segments(&self) -> usize;

    /// Segment size in message blocks (constant across segments
    /// except possibly the last). The last segment may have fewer
    /// blocks if `m` doesn't divide evenly by `K`.
    fn segment_size_in_blocks(&self) -> usize;

    /// Fetch one segment's cover bits and costs.
    ///
    /// Returned vectors have length `(K × w)` (full segment) or
    /// shorter if `seg_idx == num_segments() - 1` and `m` doesn't
    /// divide evenly by `K`.
    ///
    /// Implementations should free transient state (the encoder
    /// per-GOP working set) after returning; the caller releases
    /// the returned vectors after a single segment's traceback.
    fn fetch_segment(&mut self, seg_idx: usize) -> (Vec<u8>, Vec<f32>);
}

/// §6E-C / Task #24.2 — in-memory `CoverFetch` adapter.
///
/// Wraps a pre-materialized `(cover_bits, costs)` pair and slices
/// on-demand per segment. Suitable for orchestrator integrations
/// (Task #24.3) where the encoder's Pass-1 cover already lives in
/// memory and the streaming-Viterbi memory savings come from the
/// STC-internal O(√n) checkpoint + back-pointer working set, not
/// the cover side.
///
/// **Memory bound**: O(n) cover (caller-provided) + O(K × w) per
/// segment fetch + O(num_segments × num_states × 8 B) checkpoints
/// inside `stc_embed_streaming_segmented`. For long-clip video
/// stego where the encoder's full per-GOP cover materialization
/// is itself the OOM source, a follow-on per-GOP-replay adapter
/// (left as v1.1+ work) replaces the cover materialization with
/// repeated Pass 1 invocations bounded by the segment's GOP range.
/// That follow-on requires encoder restartability at arbitrary
/// GOP boundaries — out of scope for v1.0 (mobile clips are
/// short enough that in-memory cover fits).
pub struct InMemoryCoverFetch<'a> {
    cover_bits: &'a [u8],
    costs: &'a [f32],
    /// Segment size in message blocks. Caller chooses;
    /// `((m as f64).sqrt().ceil() as usize).max(1)` matches the
    /// inline `stc_embed_segmented` checkpoint cadence and
    /// preserves bit-exact equivalence.
    segment_size_in_blocks: usize,
    /// STC stride `w`.
    w: usize,
    /// Message length `m`.
    m: usize,
}

impl<'a> InMemoryCoverFetch<'a> {
    /// Construct a new in-memory cover fetcher. `cover_bits.len()`
    /// must equal `costs.len()` and must equal `m * w`. Returns
    /// `None` on length-mismatch.
    pub fn new(
        cover_bits: &'a [u8],
        costs: &'a [f32],
        m: usize,
        w: usize,
        segment_size_in_blocks: usize,
    ) -> Option<Self> {
        if cover_bits.len() != costs.len() {
            return None;
        }
        // Mirror the inline `stc_embed` contract: caller may
        // provide a cover slice whose length is >= m*w; the
        // streaming-Viterbi only walks the first m*w positions
        // (the trailing positions are unused at this `w`).
        if cover_bits.len() < m * w {
            return None;
        }
        if segment_size_in_blocks == 0 {
            return None;
        }
        Some(Self {
            cover_bits,
            costs,
            segment_size_in_blocks,
            w,
            m,
        })
    }
}

impl<'a> CoverFetch for InMemoryCoverFetch<'a> {
    fn total_positions(&self) -> usize {
        // Only the first m * w positions are exercised by streaming-
        // Viterbi (matches inline `stc_embed_segmented`'s
        // `for j in 0..n` where n = m*w). The trailing slack in
        // `cover_bits` is intentionally ignored.
        self.m * self.w
    }
    fn num_segments(&self) -> usize {
        self.m.div_ceil(self.segment_size_in_blocks)
    }
    fn segment_size_in_blocks(&self) -> usize {
        self.segment_size_in_blocks
    }
    fn fetch_segment(&mut self, seg_idx: usize) -> (Vec<u8>, Vec<f32>) {
        let block_start = seg_idx * self.segment_size_in_blocks;
        let block_end =
            ((seg_idx + 1) * self.segment_size_in_blocks).min(self.m);
        let j_start = block_start * self.w;
        let j_end = block_end * self.w;
        (
            self.cover_bits[j_start..j_end].to_vec(),
            self.costs[j_start..j_end].to_vec(),
        )
    }
}

/// One Phase B segment's emission, returned by
/// `StreamingViterbiPhaseB::step()`.
///
/// Phase 6 uses these per-segment emissions to feed the
/// `PerGopPlanBuilder` directly, avoiding the O(n) plan
/// materialization that the wrapper `stc_embed_streaming_segmented`
/// has to assemble for the legacy result type.
#[derive(Debug, Clone)]
pub struct PhaseBSegmentEmission {
    /// Segment index in the cover's segment space. Step calls walk
    /// `seg_idx` from `num_segments() - 1` down to `0`.
    pub seg_idx: usize,
    /// Position range within the full cover:
    /// `[j_start, j_start + stego_bits.len())`.
    pub j_start: usize,
    /// Stego bits for this segment, length = `seg_blocks × w`
    /// (or 0 for empty trailing segments).
    pub stego_bits: Vec<u8>,
    /// Number of modifications (`cover_bit ^ stego_bit` count) in
    /// this segment. Tracked inline during traceback so the
    /// streaming pipeline doesn't need a third per-segment re-fetch.
    pub num_modifications: usize,
}

/// Step-driven streaming-segmented Viterbi STC.
///
/// Lifecycle:
/// 1. `new(cover, message, hhat, h, w)` — runs Phase A (forward
///    Viterbi) and stores per-segment checkpoints. Returns a
///    driver primed for Phase B traceback.
/// 2. `step()` — processes ONE segment in reverse order
///    (`num_segments-1`, `num_segments-2`, ..., `0`). Each call
///    returns `Some(PhaseBSegmentEmission)`; `Ok(None)` signals
///    all segments have been emitted.
/// 3. `total_cost()` / `final_state()` — accessors for
///    post-traceback validation.
///
/// The legacy `stc_embed_streaming_segmented` function is now a
/// thin wrapper that loops `step()` and concatenates the emissions
/// into a single `EmbedResult`. The Phase 6 interleaved
/// orchestrator drives 4× `StreamingViterbiPhaseB` in round-robin
/// lockstep, feeding emissions directly into `PerGopPlanBuilder`
/// without ever materializing the full per-domain plan.
pub struct StreamingViterbiPhaseB<'a> {
    cover: &'a mut dyn CoverFetch,
    message: &'a [u8],
    columns: Vec<usize>,
    w: usize,
    m: usize,
    n: usize,
    k: usize,
    num_states: usize,
    num_segments: usize,
    /// Phase A checkpoints — `checkpoints[seg]` is the cost array at
    /// the start of segment `seg`. Phase B copies into `prev_cost`
    /// before recomputing the segment's back-pointers.
    checkpoints: Vec<Vec<f64>>,
    /// Best total cost from Phase A.
    total_cost: f64,
    /// Running entry state across reverse-order Phase B traceback.
    /// Initialized to Phase A's argmin state; ends at 0 after the
    /// last segment for valid embeddings.
    entry_state: usize,
    /// Reusable working buffers, owned by the driver to avoid
    /// per-step allocation churn.
    prev_cost: Vec<f64>,
    curr_cost: Vec<f64>,
    shifted_cost: Vec<f64>,
    /// Cursor: which segment to emit next (counts down). `None`
    /// means all segments have been emitted.
    next_seg: Option<usize>,
    progress_interval_b: usize,
    progress_counter_b: usize,
}

impl<'a> StreamingViterbiPhaseB<'a> {
    /// Run Phase A (forward Viterbi + checkpoints). Returns a
    /// driver ready for `step()` calls.
    pub fn new(
        cover: &'a mut dyn CoverFetch,
        message: &'a [u8],
        hhat_matrix: &[Vec<u32>],
        h: usize,
        w: usize,
    ) -> Result<Self, &'static str> {
        let n = cover.total_positions();
        let m = message.len();
        let num_states = 1usize << h;
        let inf = f64::INFINITY;

        let k = cover.segment_size_in_blocks();
        if k == 0 {
            return Err("segment_size_in_blocks must be > 0");
        }
        let num_segments = cover.num_segments();
        if m > 0 && num_segments != m.div_ceil(k) {
            return Err("num_segments inconsistent with m and segment_size_in_blocks");
        }

        let columns: Vec<usize> = (0..w)
            .map(|c| hhat::column_packed(hhat_matrix, c) as usize)
            .collect();

        let phase_a_steps = STC_PROGRESS_STEPS / 2;
        let progress_interval_a = (n / phase_a_steps as usize).max(1);
        let phase_b_steps = STC_PROGRESS_STEPS - phase_a_steps;
        let progress_interval_b = (n / phase_b_steps as usize).max(1);

        let mut prev_cost = vec![inf; num_states];
        prev_cost[0] = 0.0;
        let mut curr_cost = vec![0.0f64; num_states];
        let mut shifted_cost = vec![inf; num_states];

        let mut checkpoints: Vec<Vec<f64>> = Vec::with_capacity(num_segments.max(1));
        checkpoints.push(prev_cost.clone());

        let mut progress_counter_a = 0usize;
        let mut msg_idx = 0;
        let mut j_global = 0usize;

        for seg in 0..num_segments {
            let block_start = seg * k;
            let block_end = ((seg + 1) * k).min(m);
            let seg_blocks = block_end - block_start;
            let seg_len = seg_blocks * w;
            if seg_len == 0 {
                continue;
            }

            let (seg_bits, seg_costs) = cover.fetch_segment(seg);
            if seg_bits.len() != seg_len || seg_costs.len() != seg_len {
                return Err("fetch_segment returned inconsistent length");
            }

            for local_j in 0..seg_len {
                let j = j_global + local_j;
                let col_idx = j % w;
                let col = columns[col_idx];
                let flip_cost = seg_costs[local_j] as f64;
                let cover_bit = seg_bits[local_j] & 1;

                let (cost_s0, cost_s1) = if cover_bit == 0 {
                    (0.0, flip_cost)
                } else {
                    (flip_cost, 0.0)
                };

                for s in 0..num_states {
                    let cost_0 = prev_cost[s] + cost_s0;
                    let cost_1 = prev_cost[s ^ col] + cost_s1;
                    curr_cost[s] = if cost_1 < cost_0 { cost_1 } else { cost_0 };
                }

                if col_idx == w - 1 && msg_idx < m {
                    let required_bit = message[msg_idx] as usize;
                    shifted_cost.fill(inf);
                    for s in 0..num_states {
                        if curr_cost[s] == inf {
                            continue;
                        }
                        if (s & 1) != required_bit {
                            continue;
                        }
                        let s_shifted = s >> 1;
                        if curr_cost[s] < shifted_cost[s_shifted] {
                            shifted_cost[s_shifted] = curr_cost[s];
                        }
                    }
                    std::mem::swap(&mut prev_cost, &mut shifted_cost);
                    msg_idx += 1;

                    if msg_idx % k == 0 && msg_idx < m {
                        checkpoints.push(prev_cost.clone());
                    }
                } else {
                    std::mem::swap(&mut prev_cost, &mut curr_cost);
                }

                progress_counter_a += 1;
                if progress_counter_a.is_multiple_of(progress_interval_a) {
                    if progress::is_cancelled() {
                        return Err("cancelled");
                    }
                    progress::advance();
                }
            }

            j_global += seg_len;
        }

        let (best_state, best_cost) = {
            let mut best = 0usize;
            let mut best_cost = inf;
            for (s, &c) in prev_cost.iter().enumerate() {
                if c < best_cost {
                    best_cost = c;
                    best = s;
                }
            }
            (best, best_cost)
        };
        if best_cost == inf {
            return Err("no valid embedding (all paths Inf)");
        }

        let next_seg = if num_segments == 0 {
            None
        } else {
            Some(num_segments - 1)
        };

        Ok(Self {
            cover,
            message,
            columns,
            w,
            m,
            n,
            k,
            num_states,
            num_segments,
            checkpoints,
            total_cost: best_cost,
            entry_state: best_state,
            prev_cost,
            curr_cost,
            shifted_cost,
            next_seg,
            progress_interval_b,
            progress_counter_b: 0,
        })
    }

    /// Total cost from Phase A. Constant across the driver's
    /// lifetime once `new()` returns.
    pub fn total_cost(&self) -> f64 {
        self.total_cost
    }

    /// Cover-position count `n`.
    pub fn total_positions(&self) -> usize {
        self.n
    }

    /// Segment count.
    pub fn num_segments(&self) -> usize {
        self.num_segments
    }

    /// Segment size in message blocks (constant `K` across all
    /// segments except possibly the last).
    pub fn segment_size_in_blocks(&self) -> usize {
        self.k
    }

    /// Final running state after the last `step()` call. Should
    /// equal 0 for valid embeddings.
    pub fn final_state(&self) -> usize {
        self.entry_state
    }

    /// Process one Phase B segment in reverse order. Returns
    /// `Ok(Some(emission))` for each segment, or `Ok(None)` once
    /// every segment has been emitted.
    pub fn step(&mut self) -> Result<Option<PhaseBSegmentEmission>, &'static str> {
        let Some(seg) = self.next_seg else {
            return Ok(None);
        };

        let block_start = seg * self.k;
        let block_end = ((seg + 1) * self.k).min(self.m);
        let seg_blocks = block_end - block_start;
        let seg_len = seg_blocks * self.w;
        let j_start = block_start * self.w;

        if seg_len == 0 {
            self.next_seg = seg.checked_sub(1);
            return Ok(Some(PhaseBSegmentEmission {
                seg_idx: seg,
                j_start,
                stego_bits: Vec::new(),
                num_modifications: 0,
            }));
        }

        let (seg_bits, seg_costs) = self.cover.fetch_segment(seg);
        if seg_bits.len() != seg_len || seg_costs.len() != seg_len {
            return Err("fetch_segment returned inconsistent length (Phase B)");
        }

        self.prev_cost.copy_from_slice(&self.checkpoints[seg]);

        let inf = f64::INFINITY;
        let mut seg_back_ptr: Vec<u128> = Vec::with_capacity(seg_len);
        let mut seg_msg_idx = block_start;

        for local_j in 0..seg_len {
            let j = j_start + local_j;
            let col_idx = j % self.w;
            let col = self.columns[col_idx];
            let flip_cost = seg_costs[local_j] as f64;
            let cover_bit = seg_bits[local_j] & 1;

            let (cost_s0, cost_s1) = if cover_bit == 0 {
                (0.0, flip_cost)
            } else {
                (flip_cost, 0.0)
            };

            let mut packed_bp = 0u128;

            for s in 0..self.num_states {
                let cost_0 = self.prev_cost[s] + cost_s0;
                let cost_1 = self.prev_cost[s ^ col] + cost_s1;
                if cost_1 < cost_0 {
                    self.curr_cost[s] = cost_1;
                    packed_bp |= 1u128 << s;
                } else {
                    self.curr_cost[s] = cost_0;
                }
            }

            seg_back_ptr.push(packed_bp);

            if col_idx == self.w - 1 && seg_msg_idx < self.m {
                let required_bit = self.message[seg_msg_idx] as usize;
                self.shifted_cost.fill(inf);
                for s in 0..self.num_states {
                    if self.curr_cost[s] == inf {
                        continue;
                    }
                    if (s & 1) != required_bit {
                        continue;
                    }
                    let s_shifted = s >> 1;
                    if self.curr_cost[s] < self.shifted_cost[s_shifted] {
                        self.shifted_cost[s_shifted] = self.curr_cost[s];
                    }
                }
                std::mem::swap(&mut self.prev_cost, &mut self.shifted_cost);
                seg_msg_idx += 1;
            } else {
                std::mem::swap(&mut self.prev_cost, &mut self.curr_cost);
            }

            self.progress_counter_b += 1;
            if self.progress_counter_b.is_multiple_of(self.progress_interval_b) {
                if progress::is_cancelled() {
                    return Err("cancelled");
                }
                progress::advance();
            }
        }

        // Traceback within this segment. Mod count folded inline so
        // Phase 6 doesn't need a third per-segment cover re-fetch.
        let mut stego_bits = vec![0u8; seg_len];
        let mut num_modifications = 0usize;
        let mut s = self.entry_state;
        for local_j in (0..seg_len).rev() {
            let j = j_start + local_j;
            let col_idx = j % self.w;

            if col_idx == self.w - 1 && (j / self.w) < self.m {
                let msg_bit = self.message[j / self.w] as usize;
                s = ((s << 1) | msg_bit) & (self.num_states - 1);
            }

            let bit = ((seg_back_ptr[local_j] >> s) & 1) as u8;
            stego_bits[local_j] = bit;

            if bit != (seg_bits[local_j] & 1) {
                num_modifications += 1;
            }

            if bit == 1 {
                s ^= self.columns[col_idx];
            }
        }

        self.entry_state = s;
        self.next_seg = seg.checked_sub(1);

        Ok(Some(PhaseBSegmentEmission {
            seg_idx: seg,
            j_start,
            stego_bits,
            num_modifications,
        }))
    }
}

/// Streaming-segmented STC embed. Bit-exact equivalent to
/// `stc_embed_segmented` (verified by
/// `streaming_matches_inline_segmented_large` test below) but
/// fetches cover on-demand via the `CoverFetch` callback rather
/// than reading a pre-materialized slice.
///
/// This is now a thin wrapper around `StreamingViterbiPhaseB`:
/// `new()` runs Phase A, then `step()` is looped to drive Phase B
/// segment-by-segment in reverse order. The wrapper assembles the
/// per-segment emissions into a single `EmbedResult` for callers
/// that want the legacy interface.
///
/// For long-form video stego where the assembled `Vec<u8>` plan is
/// itself O(n) memory, use `StreamingViterbiPhaseB` directly and
/// route per-segment emissions to a `PerGopPlanBuilder` (Phase 6.2)
/// instead — that's the path that keeps mobile peak RSS under
/// ~400 MB on 15-min 1080p clips.
pub fn stc_embed_streaming_segmented(
    cover: &mut dyn CoverFetch,
    message: &[u8],
    hhat_matrix: &[Vec<u32>],
    h: usize,
    w: usize,
) -> Result<EmbedResult, &'static str> {
    let mut driver =
        StreamingViterbiPhaseB::new(cover, message, hhat_matrix, h, w)?;
    let total_cost = driver.total_cost();
    let n = driver.total_positions();
    let m = message.len();

    let mut stego_bits = vec![0u8; n];
    let mut num_modifications = 0usize;

    while let Some(em) = driver.step()? {
        let end = em.j_start + em.stego_bits.len();
        stego_bits[em.j_start..end].copy_from_slice(&em.stego_bits);
        num_modifications += em.num_modifications;
    }

    debug_assert_eq!(
        driver.final_state(),
        0,
        "traceback did not return to initial state 0",
    );
    debug_assert_eq!(stc_extract(&stego_bits, hhat_matrix, w)[..m], message[..m],);

    Ok(EmbedResult {
        stego_bits,
        total_cost,
        num_modifications,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stego::stc::embed::stc_embed;
    use crate::stego::stc::extract::stc_extract;
    use crate::stego::stc::hhat::generate_hhat;

    // The InMemoryCoverFetch adapter (Task #24.2) is exercised by
    // streaming_matches_inline_segmented_large below.

    /// Equivalence gate: streaming-segmented STC must produce
    /// byte-exact same `(stego_bits, total_cost, num_modifications)`
    /// as `stc_embed` (which dispatches to `stc_embed_segmented`
    /// for large enough n). Mirrors
    /// `inline_segmented_equivalence_large` in `embed.rs`.
    #[test]
    fn streaming_matches_inline_segmented_large() {
        let h = 4;
        let w = 1;
        let m = 200;
        let n = m * w;
        let mut seed = [0u8; 32];
        seed[..19].copy_from_slice(b"streaming-test-seed");
        let hhat = generate_hhat(h, w, &seed);

        // Deterministic cover bits + costs from a small LCG so the
        // test is reproducible and same inputs feed both paths.
        let mut s: u32 = 0xDEAD_BEEF;
        let mut cover_bits = vec![0u8; n];
        let mut costs = vec![0.0f32; n];
        for j in 0..n {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            cover_bits[j] = ((s >> 16) & 1) as u8;
            costs[j] = ((s >> 17) & 0xFFF) as f32 / 4096.0 + 0.01;
        }

        let message: Vec<u8> = (0..m).map(|i| (i & 1) as u8).collect();

        let inline = stc_embed(&cover_bits, &costs, &message, &hhat, h, w)
            .expect("inline embed succeeds");

        // Streaming-segmented uses K = ceil(sqrt(m)).
        let k = ((m as f64).sqrt().ceil() as usize).max(1);
        let mut cover = InMemoryCoverFetch::new(&cover_bits, &costs, m, w, k)
            .expect("InMemoryCoverFetch construction");
        let streaming =
            stc_embed_streaming_segmented(&mut cover, &message, &hhat, h, w)
                .expect("streaming embed succeeds");

        assert_eq!(
            inline.stego_bits, streaming.stego_bits,
            "stego bits diverge between inline and streaming",
        );
        assert!(
            (inline.total_cost - streaming.total_cost).abs() < 1e-6,
            "total_cost diverges: inline={} streaming={}",
            inline.total_cost,
            streaming.total_cost,
        );
        assert_eq!(
            inline.num_modifications, streaming.num_modifications,
            "num_modifications diverges",
        );

        // Sanity: the embedded message extracts cleanly.
        let extracted = stc_extract(&streaming.stego_bits, &hhat, w);
        assert_eq!(&extracted[..m], &message[..]);
    }

    /// Empty-message edge case: `m == 0` produces an empty
    /// stego_bits with zero modifications and zero cost.
    #[test]
    fn streaming_empty_message_returns_empty() {
        struct EmptyCover;
        impl CoverFetch for EmptyCover {
            fn total_positions(&self) -> usize {
                0
            }
            fn num_segments(&self) -> usize {
                0
            }
            fn segment_size_in_blocks(&self) -> usize {
                1
            }
            fn fetch_segment(
                &mut self,
                _seg_idx: usize,
            ) -> (Vec<u8>, Vec<f32>) {
                (Vec::new(), Vec::new())
            }
        }
        let mut cover = EmptyCover;
        let hhat: Vec<Vec<u32>> = vec![vec![0u32]; 4];
        let result =
            stc_embed_streaming_segmented(&mut cover, &[], &hhat, 4, 1)
                .expect("empty embed");
        assert_eq!(result.stego_bits.len(), 0);
        assert_eq!(result.num_modifications, 0);
    }

    /// `InMemoryCoverFetch::new` rejects length mismatches and
    /// zero segment_size_in_blocks. Defensive constructor guard.
    #[test]
    fn in_memory_cover_fetch_validates_inputs() {
        let bits = vec![0u8; 10];
        let costs_short = vec![0.0f32; 9];
        let costs_full = vec![0.0f32; 10];
        // Length mismatch between bits and costs.
        assert!(
            InMemoryCoverFetch::new(&bits, &costs_short, 10, 1, 4).is_none(),
            "expected None on bits/costs length mismatch",
        );
        // m * w exceeds bits.len() (Phase-5-relaxed contract: m*w
        // must fit within bits.len(), can be smaller).
        assert!(
            InMemoryCoverFetch::new(&bits, &costs_full, 11, 1, 4).is_none(),
            "expected None on m*w > bits.len()",
        );
        // m * w smaller than bits.len() is now allowed (matches
        // inline stc_embed contract).
        assert!(
            InMemoryCoverFetch::new(&bits, &costs_full, 5, 1, 4).is_some(),
            "expected Some on m*w < bits.len() (slack-allowed)",
        );
        // Zero segment_size_in_blocks.
        assert!(
            InMemoryCoverFetch::new(&bits, &costs_full, 10, 1, 0).is_none(),
            "expected None on zero segment_size_in_blocks",
        );
        // Valid construction.
        let cover = InMemoryCoverFetch::new(&bits, &costs_full, 10, 1, 4);
        assert!(cover.is_some());
        let cover = cover.unwrap();
        assert_eq!(cover.total_positions(), 10);
        assert_eq!(cover.num_segments(), 3); // ceil(10/4) = 3
        assert_eq!(cover.segment_size_in_blocks(), 4);
    }

    /// Length-mismatch from the callback returns Err rather than
    /// panicking. Defensive guard against buggy CoverFetch impls.
    #[test]
    fn streaming_rejects_inconsistent_segment_lengths() {
        struct BadCover;
        impl CoverFetch for BadCover {
            fn total_positions(&self) -> usize {
                10
            }
            fn num_segments(&self) -> usize {
                1
            }
            fn segment_size_in_blocks(&self) -> usize {
                10
            }
            fn fetch_segment(
                &mut self,
                _seg_idx: usize,
            ) -> (Vec<u8>, Vec<f32>) {
                // Returns the wrong size (5 instead of 10).
                (vec![0u8; 5], vec![0.0f32; 5])
            }
        }
        let mut cover = BadCover;
        let hhat: Vec<Vec<u32>> = vec![vec![0u32]; 4];
        let message = vec![0u8; 10];
        let result =
            stc_embed_streaming_segmented(&mut cover, &message, &hhat, 4, 1);
        assert!(result.is_err(), "expected Err on inconsistent length");
    }
}
