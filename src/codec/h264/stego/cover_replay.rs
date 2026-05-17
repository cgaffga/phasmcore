// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! §long-form-stego Phase 4 — H264GopReplayCover.
//!
//! Concrete `CoverFetch` adapter that bounds cover-side memory to
//! O(K·w + num_gops) by re-running Pass 1 per `fetch_segment` over
//! only the GOPs that contain the segment's cover positions.
//!
//! Construction does ONE counting-only Pass 1 (Phase 2) to harvest
//! per-GOP per-domain counts; cumulative counts let `fetch_segment`
//! map a segment's `[j_start..j_end)` cover-position range to a
//! GOP range `[gop_start..gop_end)` in O(log num_gops) via binary
//! search. Each fetch then runs a per-GOP-range Pass 1 (Phase 3),
//! slices to `[j_start..j_end)`, and drops the GOP-range cover —
//! peak transient memory is one GOP-range cover (~few MB at
//! 1080p).
//!
//! ## When to use
//!
//! Long-clip video stego where the per-domain cover would
//! materialize to gigabytes of bits + costs in the inline
//! `InMemoryCoverFetch` path. For a 15-min 1080p clip with ~16B
//! per-domain positions, the in-memory adapter holds ~210 GB; this
//! adapter holds ~50 MB working set across encode + Pass 1 replay
//! + STC checkpoints.
//!
//! ## Trade-off
//!
//! Each `fetch_segment` re-runs Pass 1 over the segment's GOP
//! range. Phase A (forward Viterbi checkpoints) + Phase B
//! (per-segment recompute + traceback) of streaming-Viterbi each
//! call `fetch_segment` once per segment, so total encoder work is
//! `2 × num_segments × avg_gops_per_segment` Pass 1 invocations.
//! For balanced segment-to-GOP ratios this is ~2-3× the
//! single-encode cost. Acceptable for long-form stego where the
//! alternative is OOM.

use crate::stego::stc::streaming_segmented::CoverFetch;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Cross-driver cache of captured `GopCover`s keyed by `(gop_start,
/// gop_end)`.
///
/// **Why.** The 4-domain streaming orchestrator constructs one
/// `H264GopReplayCover` per active `EmbedDomain` (typically 3 at
/// default weights: CS + CSL + MVDs). Each cover's `fetch_segment`
/// calls `pass1_capture_4domain_for_gop_range`, which performs a
/// full per-GOP-range encode and returns ALL 4 domains' cover bits.
/// Each cover then slices off only its own domain. Without sharing,
/// every GOP range is re-encoded N-active-domains times — pure
/// redundancy.
///
/// **What.** Phase A (forward Viterbi + checkpoints) and Phase B
/// (backward traceback) each iterate every segment once, so without
/// caching the per-GOP encode count is `3 active drivers × 2 phases
/// × num_segments`. With this cache shared across all drivers the
/// count drops to `num_distinct_ranges` (typically `num_gops`).
/// Empirically ~3× speedup on the capture phase and ~2× overall on
/// the streaming-v2 path.
///
/// **Memory bound.** `HashMap` over `(usize, usize)` keys, value is
/// an `Arc<GopCover>` so cache hits are O(1) refcount bumps rather
/// than ~MB-scale `Vec` clones (at 1080p × 10f the full 4-domain
/// `GopCover` is ~2-3 MB). For typical short clips (n_gops ≤ 6) the
/// cache holds a few `Arc`s plus the inner cover bytes — bounded by
/// `num_distinct_ranges × sizeof(GopCover)`. Long-form (#79 / #472)
/// streaming sessions construct fresh covers per chunk and discard
/// the cache between chunks; memory bound is preserved.
pub type SharedGopRangeCache = Arc<Mutex<HashMap<(usize, usize), Arc<super::orchestrate::GopCover>>>>;

/// Factory for an empty shared cache. Orchestrator calls this once
/// and passes `cache.clone()` to each `H264GopReplayCover`'s
/// `set_cache`. Returns an `Arc` so multiple covers can share the
/// same underlying `HashMap` cheaply.
pub fn new_shared_gop_range_cache() -> SharedGopRangeCache {
    Arc::new(Mutex::new(HashMap::new()))
}

/// One embedding domain's view of the per-GOP cover replay.
/// Constructing four of these (one per `EmbedDomain`) provides the
/// full §30D-C orchestrator's cover-fetch surface.
pub struct H264GopReplayCover<'a> {
    yuv: &'a [u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    b_count: usize,
    quality: Option<u8>,
    /// Which embedding domain this adapter serves cover for.
    domain: super::hook::EmbedDomain,
    /// Cumulative per-GOP position counts for `domain`. Length =
    /// num_gops + 1, with `cum[0] = 0` and `cum[num_gops] =
    /// total_positions`. `cum[g]` is the prefix sum of GOPs
    /// `[0..g)`.
    cum_positions: Vec<usize>,
    /// Segment size in message blocks (for `CoverFetch` reporting).
    segment_size_in_blocks: usize,
    /// STC stride `w`.
    w: usize,
    /// Message block count `m`.
    m: usize,
    /// Task #383 — `EncoderEnvSnapshot` captured ONCE at cover
    /// construction, installed as the thread-local "current
    /// snapshot" for this cover's lifetime. Both the count Pass 1
    /// (in `new` / `from_counts`) and every capture Pass 1 (in
    /// `fetch_segment`) build encoders via
    /// `Encoder::new` → `EncoderEnvSnapshot::capture_or_inherit`,
    /// which inherits this installed snapshot. Result: count and
    /// capture see byte-identical env state, even if env-mutating
    /// tests on other threads change `PHASM_*` env vars between
    /// `new` and `fetch_segment`. Production: no installed snapshot
    /// → `capture` falls back to live env, behavior unchanged.
    ///
    /// `EnvSnapshotGuard` handles reentrance: multiple covers in
    /// the same orchestrator share the outermost cover's install
    /// (their inner installs are no-ops).
    _env_guard: super::super::encoder::mb_decision_b::EnvSnapshotGuard,
    /// Optional shared cache of captured `GopCover`s keyed by
    /// `(gop_start, gop_end)`. When `Some`, `fetch_segment`
    /// consults the cache before calling
    /// `pass1_capture_4domain_for_gop_range`; on miss it fills the
    /// cache so siblings (the other active-domain covers in the
    /// 4-domain orchestrator) can reuse it. When `None`, behaves
    /// exactly as the pre-#512 path — every fetch re-encodes.
    /// Default is `None`; orchestrator opts in via `set_cache`.
    cache: Option<SharedGopRangeCache>,
}

impl<'a> H264GopReplayCover<'a> {
    /// Construct a `H264GopReplayCover` for one embedding domain.
    ///
    /// Runs ONE counting-only Pass 1 (Phase 2) to harvest per-GOP
    /// counts; subsequent `fetch_segment` calls do partial-range
    /// Pass 1's via Phase 3.
    ///
    /// `m` and `w` and `segment_size_in_blocks` come from the
    /// orchestrator's STC plan, not from the cover. Callers should
    /// ensure `m × w == cum_positions[num_gops]` (the total
    /// position count for this domain) — `new` returns Err on
    /// mismatch.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        yuv: &'a [u8],
        width: u32,
        height: u32,
        n_frames: usize,
        gop_size: usize,
        b_count: usize,
        quality: Option<u8>,
        domain: super::hook::EmbedDomain,
        m: usize,
        w: usize,
        segment_size_in_blocks: usize,
    ) -> Result<Self, crate::stego::error::StegoError> {
        if w == 0 || segment_size_in_blocks == 0 {
            return Err(crate::stego::error::StegoError::InvalidVideo(
                "w and segment_size_in_blocks must be > 0".into(),
            ));
        }

        // Task #383 — capture + install env snapshot BEFORE the
        // counting Pass 1 so all encoder constructions within this
        // cover's lifetime (count + every fetch_segment capture)
        // inherit the same snapshot via
        // `EncoderEnvSnapshot::capture_or_inherit`.
        let env_snapshot = super::super::encoder::mb_decision_b::EncoderEnvSnapshot::capture();
        let env_guard = env_snapshot.install();

        // Phase 2 — counting-only Pass 1 for per-GOP per-domain
        // counts.
        let per_gop_counts = super::encode_pixels::pass1_count_per_gop_4domain(
            yuv, width, height, n_frames, gop_size, b_count, quality,
        )?;
        let domain_idx = domain as usize;
        let mut cum_positions = Vec::with_capacity(per_gop_counts.len() + 1);
        cum_positions.push(0);
        for row in per_gop_counts.iter() {
            let last = *cum_positions.last().unwrap();
            cum_positions.push(last + row[domain_idx]);
        }
        let total = *cum_positions.last().unwrap();

        if m * w > total {
            return Err(crate::stego::error::StegoError::InvalidVideo(format!(
                "m * w = {} exceeds domain {:?} cover total {}",
                m * w,
                domain,
                total,
            )));
        }

        Ok(Self {
            yuv,
            width,
            height,
            n_frames,
            gop_size,
            b_count,
            quality,
            domain,
            cum_positions,
            segment_size_in_blocks,
            w,
            m,
            _env_guard: env_guard,
            cache: None,
        })
    }

    /// Construct from pre-computed per-GOP counts. Used by the
    /// streaming orchestrator (Phase 5) to avoid running the
    /// counting Pass 1 once per domain — the orchestrator runs it
    /// once and feeds the same `per_gop_counts` to all four
    /// adapters.
    #[allow(clippy::too_many_arguments)]
    pub fn from_counts(
        yuv: &'a [u8],
        width: u32,
        height: u32,
        n_frames: usize,
        gop_size: usize,
        b_count: usize,
        quality: Option<u8>,
        domain: super::hook::EmbedDomain,
        per_gop_counts: &[[usize; 4]],
        m: usize,
        w: usize,
        segment_size_in_blocks: usize,
    ) -> Result<Self, crate::stego::error::StegoError> {
        if w == 0 || segment_size_in_blocks == 0 {
            return Err(crate::stego::error::StegoError::InvalidVideo(
                "w and segment_size_in_blocks must be > 0".into(),
            ));
        }

        // Task #383 — same env-snapshot install semantics as `new`.
        // The first cover in an orchestrator installs; subsequent
        // covers' guards are no-ops (outer install wins). This makes
        // all encoders built within the orchestrator's lifetime
        // inherit one consistent snapshot.
        let env_snapshot = super::super::encoder::mb_decision_b::EncoderEnvSnapshot::capture();
        let env_guard = env_snapshot.install();

        let domain_idx = domain as usize;
        let mut cum_positions = Vec::with_capacity(per_gop_counts.len() + 1);
        cum_positions.push(0);
        for row in per_gop_counts.iter() {
            let last = *cum_positions.last().unwrap();
            cum_positions.push(last + row[domain_idx]);
        }
        let total = *cum_positions.last().unwrap();
        if m * w > total {
            return Err(crate::stego::error::StegoError::InvalidVideo(format!(
                "m * w = {} exceeds domain {:?} cover total {}",
                m * w,
                domain,
                total,
            )));
        }
        Ok(Self {
            yuv,
            width,
            height,
            n_frames,
            gop_size,
            b_count,
            quality,
            domain,
            cum_positions,
            segment_size_in_blocks,
            w,
            m,
            _env_guard: env_guard,
            cache: None,
        })
    }

    /// #512 — attach a shared `(gop_start, gop_end)` → `GopCover`
    /// cache. The 4-domain orchestrator constructs one cache via
    /// `new_shared_gop_range_cache()` and calls `set_cache` on every
    /// `H264GopReplayCover` (one per active `EmbedDomain`). On
    /// `fetch_segment` cache miss, the capture result is inserted so
    /// sibling covers + Phase B's backward re-fetch find it.
    pub fn set_cache(&mut self, cache: SharedGopRangeCache) {
        self.cache = Some(cache);
    }

    /// Map a cover-position range `[j_start..j_end)` to a GOP range
    /// `[gop_start..gop_end)` such that `cum[gop_start] <= j_start`
    /// and `cum[gop_end] >= j_end`. Returns the GOP range plus the
    /// in-GOP-range slice offsets `[off_start..off_end)` such that
    /// `cum[gop_start] + off_start == j_start` and
    /// `cum[gop_start] + off_end == j_end`.
    fn map_range(&self, j_start: usize, j_end: usize) -> (usize, usize, usize, usize) {
        // Find largest gop_start with cum[gop_start] <= j_start.
        let gop_start = self
            .cum_positions
            .partition_point(|&c| c <= j_start)
            .saturating_sub(1);
        // Find smallest gop_end with cum[gop_end] >= j_end.
        let gop_end = self.cum_positions.partition_point(|&c| c < j_end);
        let off_start = j_start - self.cum_positions[gop_start];
        let off_end = j_end - self.cum_positions[gop_start];
        (gop_start, gop_end, off_start, off_end)
    }

    /// Pull this adapter's domain slice from a `GopCover` into a
    /// `(bits, costs)` pair. Called by `fetch_segment` after the
    /// per-GOP-range Pass 1 returns.
    fn slice_domain(
        &self,
        cov: &super::orchestrate::GopCover,
        off_start: usize,
        off_end: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        use super::hook::EmbedDomain;
        match self.domain {
            EmbedDomain::CoeffSignBypass => (
                cov.cover.coeff_sign_bypass.bits[off_start..off_end].to_vec(),
                cov.costs.coeff_sign_bypass[off_start..off_end].to_vec(),
            ),
            EmbedDomain::CoeffSuffixLsb => (
                cov.cover.coeff_suffix_lsb.bits[off_start..off_end].to_vec(),
                cov.costs.coeff_suffix_lsb[off_start..off_end].to_vec(),
            ),
            EmbedDomain::MvdSignBypass => (
                cov.cover.mvd_sign_bypass.bits[off_start..off_end].to_vec(),
                cov.costs.mvd_sign_bypass[off_start..off_end].to_vec(),
            ),
            EmbedDomain::MvdSuffixLsb => (
                cov.cover.mvd_suffix_lsb.bits[off_start..off_end].to_vec(),
                cov.costs.mvd_suffix_lsb[off_start..off_end].to_vec(),
            ),
        }
    }
}

impl<'a> CoverFetch for H264GopReplayCover<'a> {
    fn total_positions(&self) -> usize {
        // Same contract as InMemoryCoverFetch: report m*w (the
        // exercised cover range) rather than the raw per-GOP
        // total. Streaming-Viterbi uses this for progress reporting
        // only; iteration is m-driven via num_segments + segment
        // sizes, so any unused trailing positions are spec-fine.
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
        if j_end <= j_start {
            return (Vec::new(), Vec::new());
        }
        let (gop_start, gop_end, off_start, off_end) =
            self.map_range(j_start, j_end);

        // #512 cross-driver cache. Cache value type is
        // `Arc<GopCover>` so hits are refcount bumps rather than
        // multi-MB `Vec` clones (a 1080p × 10f GopCover holds ~2-3
        // MB of bits + costs across 4 domains). On miss we encode
        // once and Arc-wrap; sibling domain covers + Phase B's
        // backward re-fetch all share the same allocation.
        let key = (gop_start, gop_end);
        let cover_arc: Arc<super::orchestrate::GopCover> = if let Some(cache) = &self.cache {
            {
                let guard = cache.lock().expect("cover-range cache poisoned");
                if let Some(c) = guard.get(&key) {
                    Arc::clone(c)
                } else {
                    drop(guard);
                    let fresh = super::encode_pixels::pass1_capture_4domain_for_gop_range(
                        self.yuv,
                        self.width,
                        self.height,
                        self.n_frames,
                        self.gop_size,
                        self.b_count,
                        self.quality,
                        gop_start,
                        gop_end,
                    )
                    .expect("pass1_capture_4domain_for_gop_range");
                    let arc = Arc::new(fresh);
                    // Race-window is benign: a sibling driver may
                    // have inserted between the drop above and this
                    // re-lock — we overwrite with byte-identical
                    // content since the encode is deterministic
                    // (Task #383 env guard pins the snapshot).
                    let mut guard = cache.lock().expect("cover-range cache poisoned");
                    guard.insert(key, Arc::clone(&arc));
                    arc
                }
            }
        } else {
            Arc::new(
                super::encode_pixels::pass1_capture_4domain_for_gop_range(
                    self.yuv,
                    self.width,
                    self.height,
                    self.n_frames,
                    self.gop_size,
                    self.b_count,
                    self.quality,
                    gop_start,
                    gop_end,
                )
                .expect("pass1_capture_4domain_for_gop_range"),
            )
        };
        self.slice_domain(&cover_arc, off_start, off_end)
    }
}
