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

use crate::stego::stc::embed::EmbedResult;

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

/// Streaming-segmented STC embed. Same algorithmic semantics as
/// `stc_embed_segmented` (bit-exact equivalent to inline Viterbi
/// per the existing `inline_segmented_equivalence_large` test) but
/// fetches cover on-demand via the `CoverFetch` callback rather
/// than reading a pre-materialized slice.
///
/// **NOT YET IMPLEMENTED** — this is the API stub for task #24.1.
/// The implementation is a refactor of `stc_embed_segmented` to
/// replace `cover_bits[j]` / `costs[j]` indexed access with
/// segment-level fetches via `cover.fetch_segment(seg_idx)`. The
/// algorithmic structure (Phase A forward + checkpoints, Phase B
/// per-segment recompute + traceback) is unchanged.
///
/// Returns `Err("not implemented")` for now. Tracks task #24.1.
#[allow(unused_variables)]
pub fn stc_embed_streaming_segmented(
    cover: &mut dyn CoverFetch,
    message: &[u8],
    hhat_matrix: &[Vec<u32>],
    h: usize,
    w: usize,
) -> Result<EmbedResult, &'static str> {
    Err("stc_embed_streaming_segmented not yet implemented — see task #24.1")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// API surface stub test — confirms the public types compile
    /// and the function returns the documented "not implemented"
    /// error. Replace with a real round-trip test when task #24.1
    /// ships the implementation.
    #[test]
    fn streaming_segmented_returns_not_implemented_stub() {
        struct EmptyCover;
        impl CoverFetch for EmptyCover {
            fn total_positions(&self) -> usize { 0 }
            fn num_segments(&self) -> usize { 0 }
            fn segment_size_in_blocks(&self) -> usize { 0 }
            fn fetch_segment(&mut self, _seg_idx: usize) -> (Vec<u8>, Vec<f32>) {
                (Vec::new(), Vec::new())
            }
        }
        let mut cover = EmptyCover;
        let hhat: Vec<Vec<u32>> = Vec::new();
        let result = stc_embed_streaming_segmented(&mut cover, &[], &hhat, 4, 1);
        assert!(result.is_err());
    }
}
