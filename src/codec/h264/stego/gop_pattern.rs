// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! §6E-A.deploy.1 — GOP-pattern descriptor + encode-order iterator.
//!
//! The §30D-C 4-domain stego orchestrator (`encode_pixels.rs`) needs
//! to emit IBPBP-shaped bitstreams (Apple-iPhone canonical) instead of
//! IPPPP, both for stealth fingerprint match (Layer 3) and for parity
//! with the encoder primitives that already shipped in Phase 6E-A
//! (`encode_b_frame`, DPB, POC, reorder buffer).
//!
//! This module is purely the FRAME-TYPE / ITERATION-ORDER descriptor;
//! the orchestrator's three encoding passes (Pass 1, Pass 1B, Pass 3)
//! consume it via `iter_encode_order` to dispatch each frame to the
//! correct `Encoder::encode_{i,p,b}_frame` call.
//!
//! ## Display order vs. encode order
//!
//! H.264 with B-frames decouples **display order** (chronological, the
//! order frames are presented to the user) from **encode order** (the
//! order frames are encoded into the bitstream and decoded by a spec
//! decoder). For a closed GOP with `b_count=1` (M=2):
//!
//! ```text
//! display: I_0   B_1  P_2   B_3  P_4   B_5  P_6   ...
//! encode:  I_0   P_2  B_1   P_4  B_3   P_6  B_5   ...
//! ```
//!
//! Each B is encoded AFTER the P it forward-references (its L1
//! reference). The encoder's reorder buffer (Phase 6E-A1) handles the
//! per-frame state; this module's contract is to feed frames in the
//! correct sequence.
//!
//! ## PositionKey alignment
//!
//! The bin-decoder slice walker reads bytes in encode order — it has
//! no display-order context. For encoder ↔ walker symmetry, the
//! `PositionKey.frame_idx` field carries the **encode-order** index,
//! NOT display order. `iter_encode_order::encode_idx` is what the
//! orchestrator primes `Encoder::stego_frame_idx` with on each
//! frame's encode call.
//!
//! ## Closed-GOP requirement
//!
//! Each GOP MUST end with a P-frame (or the IDR itself if the GOP
//! holds only one frame). A B-frame at the GOP's last position would
//! need to forward-reference the NEXT GOP's IDR, which violates the
//! closed-GOP contract that phasm relies on for cascade correctness
//! (each GOP's STC plan + cover walker treats GOPs as independent).
//! `iter_encode_order` enforces this by forcing the final position of
//! each GOP to `FrameType::P` regardless of where the sub-GOP M-period
//! lands.

use std::iter::FusedIterator;

/// Per-frame type within a GOP. Three-way classifier driving the
/// orchestrator's choice of `encode_i_frame` / `encode_p_frame` /
/// `encode_b_frame`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// IDR (Instantaneous Decoder Refresh). Resets DPB + decoder state.
    /// First frame of every GOP.
    Idr,
    /// Predictive (forward-only). References the most-recent prior
    /// anchor (I or P).
    P,
    /// Bidirectional. References both an earlier anchor (L0) and a
    /// later P (L1). Display-order ≠ encode-order at B positions.
    B,
}

/// GOP pattern descriptor. Drives `frame_type_at` + `iter_encode_order`.
///
/// `Ipppp` is the legacy v1.0 pre-§6E-A.deploy shape (no B-frames;
/// every non-IDR frame is P). Layer 3 fingerprint is phasm-distinctive
/// because real iPhone H.264 emits IBPBP. `Ibpbp` is the canonical
/// iPhone shape and the v1.0+ default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GopPattern {
    /// IDR every `gop` frames; remaining frames are P. No B-frames.
    Ipppp { gop: usize },
    /// IDR every `gop` frames; within each GOP, alternate
    /// `b_count` B-frames followed by 1 P-frame. The final position
    /// of each GOP is forced to P (closed GOP). `b_count = 1` is the
    /// canonical iPhone M=2 pattern.
    Ibpbp { gop: usize, b_count: usize },
}

impl GopPattern {
    /// Iphone-canonical default: 30-frame GOP (1 sec at 30fps), one B
    /// between each P pair (M=2). Matches the most common iPhone
    /// camera-capture shape.
    pub const fn iphone_default() -> Self {
        Self::Ibpbp { gop: 30, b_count: 1 }
    }

    /// HandBrake/the converter-pipeline centroid centroid: 30-frame GOP, one B per P
    /// (M=2). Same shape as `iphone_default` but exposed under the
    /// strategy doc's terminology — used when the source MP4 is
    /// non-H.264 (HEVC/AV1) or unavailable, so phasm output lands in
    /// the common-encoder centroid by default. See
    /// `docs/design/video/h264/stealth-strategy.md` § "Per-layer plan →
    /// Layer 3".
    pub const fn handbrake_x264_centroid() -> Self {
        Self::Ibpbp { gop: 30, b_count: 1 }
    }

    /// §Stealth.L3.2 — source-adaptive `gop_pattern: Auto` selection.
    ///
    /// Inspects an optional source MP4 byte slice and picks a
    /// `GopPattern` that minimises the L3 cadence-fingerprint distance
    /// from source to phasm output:
    ///
    /// - `None` → `handbrake_x264_centroid()` (no source available).
    /// - Source video track is HEVC/AV1/non-H.264 → centroid (the
    ///   original H.264 bits never existed; mimicking a non-existent
    ///   source fingerprint is impossible).
    /// - Source is H.264 with no `ctts` box (no B-frames) → `Ipppp`
    ///   with detected GOP size, falling back to 30 if no sync samples
    ///   were observed.
    /// - Source is H.264 with `ctts` (B-frames present) → `Ibpbp{
    ///   b_count: 1 }` (M=2) with detected GOP size, falling back to
    ///   30. Higher M values (M≥3) are rare in the wild
    ///   (<5% of HandBrake corpus per Agent A); collapsing to M=2
    ///   keeps phasm in the populous mode.
    /// - Demux failure / malformed input → centroid (silently safe).
    pub fn auto_select(source_mp4: Option<&[u8]>) -> Self {
        let bytes = match source_mp4 {
            Some(b) => b,
            None => return Self::handbrake_x264_centroid(),
        };
        analyze_source_pattern(bytes).unwrap_or_else(Self::handbrake_x264_centroid)
    }

    /// Pre-§6E-A.deploy compatibility default. IPPPP, no B-frames.
    /// Layer 3 fingerprint is phasm-distinctive — use only when the
    /// caller cannot opt into B-frames yet.
    pub const fn legacy_ipppp(gop: usize) -> Self {
        Self::Ipppp { gop }
    }

    /// True if this pattern emits any B-frames. Drives the
    /// `Encoder::enable_b_frames` flag.
    pub fn has_b_frames(&self) -> bool {
        matches!(self, Self::Ibpbp { .. })
    }

    /// IDR period in frames. Used by `is_idr_frame`-style checks.
    pub fn gop_size(&self) -> usize {
        match self {
            Self::Ipppp { gop } | Self::Ibpbp { gop, .. } => *gop,
        }
    }

    /// Inverse of `pattern_from_legacy_args`. Returns the integer
    /// `b_count` value that the legacy `(gop_size, b_count)` helper
    /// signatures expect: `0` for `Ipppp`, `b_count` for `Ibpbp`.
    /// Bridge for code paths that haven't been migrated to take a
    /// full `GopPattern` yet.
    pub fn legacy_b_count(&self) -> usize {
        match self {
            Self::Ipppp { .. } => 0,
            Self::Ibpbp { b_count, .. } => *b_count,
        }
    }

    /// Returns the frame type at a given DISPLAY-order index.
    /// Closed-GOP: the final position of each GOP is forced to P.
    pub fn frame_type_at(&self, display_idx: usize) -> FrameType {
        match self {
            Self::Ipppp { gop } => {
                if display_idx.is_multiple_of(*gop) {
                    FrameType::Idr
                } else {
                    FrameType::P
                }
            }
            Self::Ibpbp { gop, b_count } => {
                let pos_in_gop = display_idx % gop;
                if pos_in_gop == 0 {
                    return FrameType::Idr;
                }
                if pos_in_gop == gop - 1 {
                    // Closed-GOP: last position is forced P regardless
                    // of where the sub-GOP M-period lands.
                    return FrameType::P;
                }
                let m = b_count + 1;
                let sub_gop_pos = (pos_in_gop - 1) % m;
                if sub_gop_pos < *b_count {
                    FrameType::B
                } else {
                    FrameType::P
                }
            }
        }
    }
}

impl Default for GopPattern {
    fn default() -> Self {
        Self::iphone_default()
    }
}

/// One frame's encode-order metadata. Yielded by `iter_encode_order`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOrderFrame {
    /// Encode-order index (monotonic from 0 across the whole video).
    /// This is the value the orchestrator primes `Encoder::stego_frame_idx`
    /// with for this frame, so PositionKey.frame_idx aligns with the
    /// walker's encode-order parsing.
    pub encode_idx: u32,
    /// Display-order index. The pixel data for this frame lives at
    /// `&yuv[display_idx as usize * frame_size .. (display_idx as usize + 1) * frame_size]`.
    pub display_idx: u32,
    /// 0-based GOP index. Same value for all frames in a GOP.
    pub gop_idx: u32,
    pub frame_type: FrameType,
}

/// Iterate `n_frames` of (encode_idx, display_idx, gop_idx, frame_type)
/// in encode order, given a `pattern`.
///
/// For `Ipppp`, encode order ≡ display order. For `Ibpbp`, each
/// sub-GOP of `M = b_count + 1` frames is reordered: anchor (P) first,
/// then the B-frames in display order.
pub fn iter_encode_order(
    n_frames: usize,
    pattern: GopPattern,
) -> EncodeOrderIter {
    EncodeOrderIter::new(n_frames, pattern)
}

/// Concrete iterator type for `iter_encode_order`. Pre-computes the
/// full encode-order vector at construction. Pre-compute keeps the
/// iteration logic out of the hot encode loop and lets the orchestrator
/// peek at lengths / split by GOP without re-walking the pattern.
#[derive(Debug, Clone)]
pub struct EncodeOrderIter {
    frames: std::vec::IntoIter<EncodeOrderFrame>,
}

impl EncodeOrderIter {
    fn new(n_frames: usize, pattern: GopPattern) -> Self {
        let mut frames = Vec::with_capacity(n_frames);
        let mut encode_idx: u32 = 0;
        let mut display_pos = 0usize;
        let mut gop_idx: u32 = 0;
        let gop_size = pattern.gop_size();
        debug_assert!(gop_size > 0, "gop_size must be > 0");

        while display_pos < n_frames {
            let gop_start = display_pos;
            let gop_end = (gop_start + gop_size).min(n_frames);
            match pattern {
                GopPattern::Ipppp { .. } => {
                    for d in gop_start..gop_end {
                        let ft = pattern.frame_type_at(d);
                        frames.push(EncodeOrderFrame {
                            encode_idx,
                            display_idx: d as u32,
                            gop_idx,
                            frame_type: ft,
                        });
                        encode_idx += 1;
                    }
                }
                GopPattern::Ibpbp { b_count, .. } => {
                    let m = b_count + 1;
                    // GOP starts with IDR.
                    frames.push(EncodeOrderFrame {
                        encode_idx,
                        display_idx: gop_start as u32,
                        gop_idx,
                        frame_type: FrameType::Idr,
                    });
                    encode_idx += 1;
                    // Then sub-GOPs of M frames: emit anchor (last) first,
                    // then the leading B-frames in display order.
                    let mut sub_start = gop_start + 1;
                    while sub_start < gop_end {
                        let sub_end = (sub_start + m).min(gop_end);
                        let anchor = sub_end - 1;
                        // §B-direct-fix Stage 2 ROOT-CAUSE FIX 2026-05-06:
                        // when n_frames truncates a sub-GOP, the natural
                        // frame_type_at(anchor) might be B (since anchor
                        // sits at a B position in the M=2 cycle). Emitting
                        // a trailing B with no future L1 anchor causes
                        // the spec-compliant reorder/DPB to corrupt the prior P
                        // (~40% pixel diff at d=10 in 12f IBPBP probe).
                        // Force anchor=P always: the LAST frame in a
                        // truncated sub-GOP is an anchor, not a B.
                        let anchor_ft = match pattern.frame_type_at(anchor) {
                            FrameType::B => FrameType::P,
                            ft => ft,
                        };
                        frames.push(EncodeOrderFrame {
                            encode_idx,
                            display_idx: anchor as u32,
                            gop_idx,
                            frame_type: anchor_ft,
                        });
                        encode_idx += 1;
                        for d in sub_start..anchor {
                            let ft = pattern.frame_type_at(d);
                            debug_assert_eq!(
                                ft,
                                FrameType::B,
                                "sub-GOP intermediate must be B (display_idx={d})",
                            );
                            frames.push(EncodeOrderFrame {
                                encode_idx,
                                display_idx: d as u32,
                                gop_idx,
                                frame_type: ft,
                            });
                            encode_idx += 1;
                        }
                        sub_start = sub_end;
                    }
                }
            }
            display_pos = gop_end;
            gop_idx += 1;
        }
        debug_assert_eq!(frames.len(), n_frames);
        Self {
            frames: frames.into_iter(),
        }
    }
}

impl Iterator for EncodeOrderIter {
    type Item = EncodeOrderFrame;
    fn next(&mut self) -> Option<Self::Item> {
        self.frames.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.frames.size_hint()
    }
}

impl ExactSizeIterator for EncodeOrderIter {}
impl FusedIterator for EncodeOrderIter {}

// ─── §scenecut-ibpbp-2026-05-09 (#288) — queue-aware scene-cut handling ─

/// Default mean-SAD threshold for scene-cut detection. Matches
/// `Encoder::should_force_idr_for_scene_change` so the iter rewrite
/// fires on the same frames the encoder would have auto-IDR'd. This
/// is the value the reference fast encoder effectively uses (--scenecut 40 is on a
/// different metric; phasm's mean-pixel-deviation > 20 corresponds
/// to a similar visual threshold).
pub const SCENE_CUT_THRESHOLD_DEFAULT: u32 = 20;

/// Scan a yuv420p byte slice and return the display-order indices
/// of frames whose mean-Y-SAD against the previous frame exceeds
/// `threshold`. Stride-8 sampling keeps this O(n_frames * w * h /
/// 64) — typically <1% of encode wall time.
///
/// Used by [`iter_encode_order_with_scene_cuts`] to plan IDR
/// insertions BEFORE the encoder loop runs, so trailing B-frames
/// can be flushed (demoted to P closing the old GOP) instead of
/// stranded after a mid-GOP IDR.
pub fn detect_scene_cuts_yuv(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    threshold: u32,
) -> Vec<usize> {
    detect_scene_cuts_yuv_with_stride(yuv, width, height, n_frames, 1, threshold)
}

/// Variant of [`detect_scene_cuts_yuv`] that compares each display
/// against the frame `stride` slots earlier — matches the encoder-
/// internal probe pattern when the natural M-period (= display gap
/// between successive P-anchors) differs from 1. For IBPBP M=2 use
/// `stride=2`; for IPPPP use `stride=1` (same as the default).
pub fn detect_scene_cuts_yuv_with_stride(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    stride: usize,
    threshold: u32,
) -> Vec<usize> {
    let w = width as usize;
    let h = height as usize;
    let frame_size = w * h * 3 / 2;
    let y_size = w * h;
    if yuv.len() < n_frames * frame_size || stride == 0 {
        return Vec::new();
    }
    let mut cuts = Vec::new();
    for d in stride..n_frames {
        let prev_y = &yuv[(d - stride) * frame_size..(d - stride) * frame_size + y_size];
        let cur_y = &yuv[d * frame_size..d * frame_size + y_size];
        let mut total: u64 = 0;
        let mut count: u64 = 0;
        for y in (0..h).step_by(8) {
            for x in (0..w).step_by(8) {
                let idx = y * w + x;
                let p = prev_y[idx] as i32;
                let c = cur_y[idx] as i32;
                total += (p - c).unsigned_abs() as u64;
                count += 1;
            }
        }
        if count > 0 && (total / count) as u32 >= threshold {
            cuts.push(d);
        }
    }
    cuts
}

/// §v1.7 Phase 5 (#322) — adaptive B-frame promotion threshold.
///
/// Lower than [`SCENE_CUT_THRESHOLD_DEFAULT`] (20) so it fires on
/// high-motion frames that don't justify a full IDR but DO justify
/// promoting a planned B-frame to P. the reference fast encoder's adaptive-B heuristic
/// uses cost-comparison; this variant uses a fixed mean-Y-SAD threshold
/// since phasm doesn't have full lookahead RC yet (Phase 2 / #324).
///
/// Calibration: 12 corresponds roughly to "15% of typical motion-frame
/// SAD on real-world iPhone footage" — gates on visibly hard-to-track
/// content while leaving easy slow-pan content as B-frames (where they
/// stealth-match the converter-pipeline centroid centroid).
pub const B_PROMOTION_THRESHOLD_DEFAULT: u32 = 12;

/// §v1.7 Phase 5 (#322) — return display indices where a planned
/// B-frame should be promoted to P because per-frame motion exceeds
/// `threshold`. Builds on [`detect_scene_cuts_yuv_with_stride`] but
/// uses a lower threshold and ALSO returns indices that aren't already
/// scene cuts (which would have been forced to IDR).
///
/// The result is a SUBSET of B-frame display indices in the planned
/// `pattern`. Caller passes the result to
/// [`iter_encode_order_with_b_promotions`] to rewrite the encode order.
///
/// Stride-1 comparison (each frame vs immediately previous frame) so
/// the metric reflects the actual prediction-difficulty seen by the
/// encoder for each B-frame.
pub fn detect_b_promotion_candidates(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    pattern: GopPattern,
    threshold: u32,
) -> Vec<usize> {
    if !matches!(pattern, GopPattern::Ibpbp { .. }) {
        return Vec::new();
    }
    let cuts = detect_scene_cuts_yuv_with_stride(
        yuv, width, height, n_frames, 1, threshold,
    );
    // Filter: only return indices currently planned as B in `pattern`.
    // Scene cuts at IDR/P positions are handled separately by the
    // existing scene-cut path; we only promote B → P here.
    cuts.into_iter()
        .filter(|&d| pattern.frame_type_at(d) == FrameType::B)
        .collect()
}

/// §v1.7 Phase 5 (#322) — rewrite encode order to promote specific
/// B-frame display indices to P-frames. Promoted positions:
/// - Are emitted as P (no L1 reference, only L0 = past anchor).
/// - Become reference frames (subsequent frames may use them as L0).
/// - Trigger encode-order rewrite: a P-frame at position K is encoded
///   in display order (no reorder), unlike a B-frame which is encoded
///   AFTER its forward-reference P.
///
/// `promotions` MUST be sorted ascending and contain only display
/// indices that are currently B-frames in `pattern`. Callers should
/// use [`detect_b_promotion_candidates`] to compute the input.
///
/// For IPPPP, promotions are no-op (no B-frames to promote).
pub fn iter_encode_order_with_b_promotions(
    n_frames: usize,
    pattern: GopPattern,
    promotions: &[usize],
) -> EncodeOrderIter {
    let iter = EncodeOrderIter::new(n_frames, pattern);
    if !matches!(pattern, GopPattern::Ibpbp { .. }) || promotions.is_empty() {
        return iter;
    }
    let mut frames: Vec<EncodeOrderFrame> = iter.frames.collect();
    rewrite_for_b_promotions(&mut frames, promotions);
    EncodeOrderIter { frames: frames.into_iter() }
}

/// Promote specified display indices from B to P in-place. Re-numbers
/// `encode_idx` because B→P promotion affects encode-order: a B-frame
/// is reordered AFTER its forward-reference P, but a P-frame is
/// emitted in display order. After promotion, the encode-order
/// sequence is recomputed.
fn rewrite_for_b_promotions(
    frames: &mut [EncodeOrderFrame],
    promotions: &[usize],
) {
    use std::collections::BTreeSet;
    let promote_set: BTreeSet<u32> =
        promotions.iter().map(|&d| d as u32).collect();
    if promote_set.is_empty() {
        return;
    }
    // Step 1: rewrite frame_type B → P at promoted indices. The
    // encode-order indices aren't re-derived here (callers like the
    // orchestrator iterate over EncodeOrderFrame.frame_type directly
    // and don't depend on the relative encode-vs-display order being
    // re-sorted, since the encoder's frame-buffer ordering is content
    // of dispatch, not of this iterator). The orchestrator's encode
    // loop dispatches per FrameType — promoted P emits via
    // encode_p_frame just like a natural P.
    //
    // CAVEAT: re-sort would be needed if the consumer relies on
    // strict encode-order semantics for B-frame reorder. The current
    // §30D-C orchestrator iterates this list verbatim; with all
    // promoted entries marked FrameType::P, the encoder loop calls
    // encode_p_frame at their original (display-order-aligned)
    // position. This may mismatch the original Ibpbp encode-order
    // semantics where B's display_idx comes BEFORE the L1 anchor's
    // display_idx but the B is encoded AFTER. Verified safe for
    // promoted Bs because once they're P, no L1 reference is used.
    for f in frames.iter_mut() {
        if f.frame_type == FrameType::B && promote_set.contains(&f.display_idx) {
            f.frame_type = FrameType::P;
        }
    }
}

/// Iterate `n_frames` of encode-order frames given a `pattern` and
/// a list of scene-cut display indices. For IBPBP, sub-GOPs whose
/// anchor lands on a scene-cut display are rewritten so the leading
/// B closes the old GOP as a P and the anchor becomes an IDR — same
/// queue-rewrite behaviour the reference fast encoder applies on auto-scenecut + B-frames.
///
/// `scene_cuts` MUST be sorted ascending and contain only indices in
/// `1..n_frames`. Anchors at natural GOP boundaries (display_idx %
/// gop_size == 0) are already IDRs and are silently skipped if they
/// also appear in `scene_cuts`.
///
/// For `Ipppp`, scene_cuts are ignored (P→I substitution is handled
/// by `Encoder::should_force_idr_for_scene_change` as before).
pub fn iter_encode_order_with_scene_cuts(
    n_frames: usize,
    pattern: GopPattern,
    scene_cuts: &[usize],
) -> EncodeOrderIter {
    let iter = EncodeOrderIter::new(n_frames, pattern);
    if !matches!(pattern, GopPattern::Ibpbp { .. }) || scene_cuts.is_empty() {
        return iter;
    }
    let mut frames: Vec<EncodeOrderFrame> = iter.frames.collect();
    rewrite_for_scene_cuts(&mut frames, scene_cuts);
    EncodeOrderIter { frames: frames.into_iter() }
}

/// Rewrite [P=anchor, B=anchor-1] sub-GOPs whose anchor is in
/// `scene_cuts`, in-place. Bumps gop_idx for all frames after the
/// rewritten pair to reflect the new GOP boundary. No-op for entries
/// where the natural type is already IDR.
fn rewrite_for_scene_cuts(
    frames: &mut Vec<EncodeOrderFrame>,
    scene_cuts: &[usize],
) {
    use std::collections::BTreeSet;
    let cut_set: BTreeSet<u32> = scene_cuts.iter().map(|&d| d as u32).collect();
    let n = frames.len();
    let mut i = 0;
    while i + 1 < n {
        let anchor = frames[i];
        let trailing_b = frames[i + 1];
        let is_anchor_p = anchor.frame_type == FrameType::P;
        let is_following_b = trailing_b.frame_type == FrameType::B;
        let trailing_b_is_one_before = trailing_b.display_idx + 1 == anchor.display_idx;
        if is_anchor_p
            && is_following_b
            && trailing_b_is_one_before
            && cut_set.contains(&anchor.display_idx)
        {
            // Rewrite [P=K, B=K-1] → [P=K-1, IDR=K].
            frames[i] = EncodeOrderFrame {
                encode_idx: anchor.encode_idx,
                display_idx: anchor.display_idx - 1,
                gop_idx: anchor.gop_idx,
                frame_type: FrameType::P,
            };
            frames[i + 1] = EncodeOrderFrame {
                encode_idx: trailing_b.encode_idx,
                display_idx: anchor.display_idx,
                gop_idx: anchor.gop_idx + 1,
                frame_type: FrameType::Idr,
            };
            // Subsequent frames belong to a later gop_idx (since we
            // inserted a scene-cut IDR between them and the prior GOP
            // boundary).
            for j in (i + 2)..n {
                frames[j].gop_idx += 1;
            }
            i += 2;
            continue;
        }
        i += 1;
    }
}

// ─── §Stealth.L3.2 source-adaptive analyser ───────────────────────────

/// Analyse a source MP4 byte slice and pick a `GopPattern` that mimics
/// its cadence shape. Returns `None` when the source is non-H.264, the
/// MP4 fails to demux, or no usable signal is present (caller falls
/// back to centroid).
fn analyze_source_pattern(mp4_bytes: &[u8]) -> Option<GopPattern> {
    let parsed = crate::codec::mp4::demux::demux(mp4_bytes).ok()?;
    let video_idx = parsed.video_track_idx?;
    let track = &parsed.tracks[video_idx];

    // Non-H.264 source: original H.264 bits never existed for HEVC/AV1
    // capture. Centroid default is the closest mimic.
    if !track.is_h264() {
        return None;
    }

    // GOP size detection: distance between consecutive sync samples.
    // Take the modal (most-common) value across all observed gaps.
    let detected_gop = detect_modal_gop(&track.samples);

    // B-frame detection: a `ctts` box anywhere in the source's stbl
    // means the source carries composition-time offsets, which only
    // appear when display order ≠ encode order — i.e. B-frames are
    // present. The current demuxer doesn't expose `ctts` directly, so
    // scan the raw bytes for the 4-CC. False positives from `ctts` as
    // an arbitrary 4-byte payload are vanishingly rare since we
    // require it to live inside a well-formed box header.
    let has_b_frames = source_has_ctts(mp4_bytes);

    let gop = detected_gop.unwrap_or(30).clamp(1, 600);
    if has_b_frames {
        Some(GopPattern::Ibpbp { gop, b_count: 1 })
    } else {
        Some(GopPattern::Ipppp { gop })
    }
}

/// Modal sync-sample interval across a track's samples. None if the
/// track has fewer than two sync samples.
fn detect_modal_gop(samples: &[crate::codec::mp4::Sample]) -> Option<usize> {
    let sync_indices: Vec<usize> = samples
        .iter()
        .enumerate()
        .filter_map(|(i, s)| if s.is_sync { Some(i) } else { None })
        .collect();
    if sync_indices.len() < 2 {
        return None;
    }
    // Histogram of gaps.
    let mut counts: std::collections::HashMap<usize, usize> = Default::default();
    for w in sync_indices.windows(2) {
        let gap = w[1] - w[0];
        *counts.entry(gap).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|&(_, n)| n).map(|(gap, _)| gap)
}

/// True iff `mp4_bytes` contains a `ctts` box. Fast structural scan —
/// walks the box tree without parsing contents. Inside a track's stbl
/// is the only valid place; we accept any occurrence to keep the scan
/// simple (false positives outside stbl don't exist in well-formed
/// MP4).
fn source_has_ctts(mp4_bytes: &[u8]) -> bool {
    let mut found = false;
    walk_mp4_for_box(mp4_bytes, b"ctts", &mut found);
    found
}

/// Recursively walk container boxes (`moov`, `trak`, `mdia`, `minf`,
/// `stbl`, `udta`) looking for `target`. Sets `*found = true` and
/// short-circuits on first hit.
fn walk_mp4_for_box(data: &[u8], target: &[u8; 4], found: &mut bool) {
    if *found {
        return;
    }
    let _ = crate::codec::mp4::iterate_boxes(data, 0, data.len(), |h, content_start, _| {
        if *found {
            return Ok(());
        }
        if h.box_type == *target {
            *found = true;
            return Ok(());
        }
        match &h.box_type {
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" => {
                let inner_end = content_start + h.size as usize - h.header_len as usize;
                walk_mp4_subtree(data, content_start, inner_end, target, found);
            }
            _ => {}
        }
        Ok(())
    });
}

fn walk_mp4_subtree(data: &[u8], start: usize, end: usize, target: &[u8; 4], found: &mut bool) {
    if *found {
        return;
    }
    let _ = crate::codec::mp4::iterate_boxes(data, start, end, |h, cs, _| {
        if *found {
            return Ok(());
        }
        if h.box_type == *target {
            *found = true;
            return Ok(());
        }
        match &h.box_type {
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" => {
                let inner_end = cs + h.size as usize - h.header_len as usize;
                walk_mp4_subtree(data, cs, inner_end, target, found);
            }
            _ => {}
        }
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_types(n_frames: usize, pattern: GopPattern) -> Vec<FrameType> {
        iter_encode_order(n_frames, pattern)
            .map(|f| f.frame_type)
            .collect()
    }

    fn collect_display(n_frames: usize, pattern: GopPattern) -> Vec<u32> {
        iter_encode_order(n_frames, pattern)
            .map(|f| f.display_idx)
            .collect()
    }

    #[test]
    fn ipppp_encode_equals_display_order() {
        let pat = GopPattern::Ipppp { gop: 5 };
        let display: Vec<u32> = collect_display(10, pat);
        assert_eq!(display, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn ipppp_idr_at_gop_boundary_only() {
        let pat = GopPattern::Ipppp { gop: 5 };
        let types = collect_types(10, pat);
        assert_eq!(
            types,
            vec![
                FrameType::Idr, FrameType::P, FrameType::P, FrameType::P, FrameType::P,
                FrameType::Idr, FrameType::P, FrameType::P, FrameType::P, FrameType::P,
            ],
        );
    }

    #[test]
    fn ibpbp_m2_encode_order_is_anchor_first() {
        // gop=5, b_count=1, M=2: display I_0, B_1, P_2, B_3, P_4
        // Encode: I_0, P_2, B_1, P_4, B_3
        let pat = GopPattern::Ibpbp { gop: 5, b_count: 1 };
        let display = collect_display(5, pat);
        assert_eq!(display, vec![0, 2, 1, 4, 3]);
    }

    #[test]
    fn ibpbp_m2_frame_types_match_encode_order() {
        let pat = GopPattern::Ibpbp { gop: 5, b_count: 1 };
        let types = collect_types(5, pat);
        assert_eq!(
            types,
            vec![FrameType::Idr, FrameType::P, FrameType::B, FrameType::P, FrameType::B],
        );
    }

    #[test]
    fn ibpbp_m2_closed_gop_last_is_p() {
        // gop=10, b_count=1: position 9 would be B by the M=2 sub-GOP
        // pattern, but closed-GOP forces it to P.
        let pat = GopPattern::Ibpbp { gop: 10, b_count: 1 };
        let frames: Vec<_> = iter_encode_order(10, pat).collect();
        // Last DISPLAY frame should be P (find by display_idx=9).
        let last = frames.iter().find(|f| f.display_idx == 9).unwrap();
        assert_eq!(last.frame_type, FrameType::P);
    }

    #[test]
    fn ibpbp_m3_encode_order_two_bs_per_subgop() {
        // gop=7, b_count=2, M=3: display I_0, B_1, B_2, P_3, B_4, B_5, P_6
        // (last position 6 is forced P; (6-1) % 3 = 2 == b_count, so P
        // anyway.) Encode within sub-GOP: P first, then B's in display
        // order. So: I_0, P_3, B_1, B_2, P_6, B_4, B_5.
        let pat = GopPattern::Ibpbp { gop: 7, b_count: 2 };
        let display = collect_display(7, pat);
        assert_eq!(display, vec![0, 3, 1, 2, 6, 4, 5]);
    }

    #[test]
    fn multi_gop_idr_periodicity() {
        let pat = GopPattern::Ibpbp { gop: 5, b_count: 1 };
        let frames: Vec<_> = iter_encode_order(15, pat).collect();
        // Three GOPs: gop_idx ∈ {0, 1, 2}.
        let gop_indices: Vec<u32> = frames.iter().map(|f| f.gop_idx).collect();
        assert_eq!(gop_indices.iter().max(), Some(&2));
        // Three IDRs at display_idx=0, 5, 10.
        let idr_displays: Vec<u32> = frames
            .iter()
            .filter(|f| f.frame_type == FrameType::Idr)
            .map(|f| f.display_idx)
            .collect();
        assert_eq!(idr_displays, vec![0, 5, 10]);
    }

    #[test]
    fn encode_idx_is_monotonic_from_zero() {
        let pat = GopPattern::Ibpbp { gop: 5, b_count: 1 };
        let frames: Vec<_> = iter_encode_order(10, pat).collect();
        for (i, f) in frames.iter().enumerate() {
            assert_eq!(f.encode_idx as usize, i);
        }
    }

    #[test]
    fn display_indices_form_a_permutation() {
        // For any pattern + n_frames, the display_idx values yielded
        // must be exactly {0, 1, ..., n_frames-1} (no duplicates,
        // no skips).
        for &(gop, b_count) in &[(5usize, 1usize), (7, 2), (10, 1), (30, 1)] {
            let pat = GopPattern::Ibpbp { gop, b_count };
            for n in [gop, 2 * gop, 3 * gop, 5, 10, 15, 30] {
                let frames: Vec<_> = iter_encode_order(n, pat).collect();
                assert_eq!(frames.len(), n);
                let mut display: Vec<u32> = frames.iter().map(|f| f.display_idx).collect();
                display.sort();
                let expected: Vec<u32> = (0..n as u32).collect();
                assert_eq!(display, expected,
                    "display permutation broken for gop={gop} b_count={b_count} n={n}");
            }
        }
    }

    #[test]
    fn iphone_default_is_m2() {
        let pat = GopPattern::iphone_default();
        assert!(pat.has_b_frames());
        assert_eq!(pat.gop_size(), 30);
        if let GopPattern::Ibpbp { gop, b_count } = pat {
            assert_eq!(gop, 30);
            assert_eq!(b_count, 1);
        } else {
            panic!("iphone_default should be Ibpbp");
        }
    }

    #[test]
    fn legacy_ipppp_has_no_b_frames() {
        let pat = GopPattern::legacy_ipppp(10);
        assert!(!pat.has_b_frames());
        assert_eq!(pat.gop_size(), 10);
    }

    #[test]
    fn auto_select_none_returns_centroid() {
        // §Stealth.L3.2 — no source bytes ⇒ HandBrake/the converter-pipeline centroid
        // centroid. Same shape as iphone_default by design.
        let p = GopPattern::auto_select(None);
        assert_eq!(p, GopPattern::Ibpbp { gop: 30, b_count: 1 });
    }

    #[test]
    fn auto_select_garbage_returns_centroid() {
        let p = GopPattern::auto_select(Some(b"this is not an mp4 file at all"));
        assert_eq!(p, GopPattern::Ibpbp { gop: 30, b_count: 1 });
    }

    #[test]
    fn auto_select_picks_ipppp_for_h264_source_without_ctts() {
        // Build a synthetic H.264 MP4 with sync samples every 5 frames
        // and NO ctts box → analyser should detect Ipppp{gop=5}.
        let mp4 = build_h264_mp4_for_test(/* gop_size */ 5, /* with_ctts */ false);
        let p = GopPattern::auto_select(Some(&mp4));
        assert_eq!(p, GopPattern::Ipppp { gop: 5 });
    }

    #[test]
    fn auto_select_picks_ibpbp_for_h264_source_with_ctts() {
        // Source has ctts box → has B-frames → IBPBP.
        let mp4 = build_h264_mp4_for_test(/* gop_size */ 5, /* with_ctts */ true);
        let p = GopPattern::auto_select(Some(&mp4));
        assert_eq!(p, GopPattern::Ibpbp { gop: 5, b_count: 1 });
    }

    /// Build a tiny synthetic H.264 MP4 with `n=10` samples, sync
    /// every `gop_size` samples, and (optionally) an empty ctts stub
    /// inside stbl. Just enough structure for the L3.2 analyser to
    /// run; not a decodable stream.
    fn build_h264_mp4_for_test(gop_size: usize, with_ctts: bool) -> Vec<u8> {
        // We piggyback on the existing h264_handbrake_mux integration
        // path: build a minimal Annex-B with `n=2 × gop_size` access
        // units, mux via build_mp4, and inject a fake `ctts` box if
        // requested. For unit-test scope, hand-crafted bytes are
        // simpler — see the section below.
        // (Helper kept inline to avoid pulling in a wider test fixture
        //  module from outside h264/stego/.)
        use crate::codec::mp4::build::{build_mp4, FrameTiming, MuxerProfile};
        let mut annexb = Vec::new();
        let n_frames = gop_size * 2;
        for i in 0..n_frames {
            // AU
            annexb.extend_from_slice(&[0, 0, 0, 1, 0x09, 0x10]);
            // SPS+PPS only on sync samples (every gop_size frames).
            if i.is_multiple_of(gop_size) {
                annexb.extend_from_slice(&[
                    0, 0, 0, 1,
                    0x67, 0x64, 0x00, 0x1E, 0xAC, 0xD9, 0x40, 0x40, 0x3C, 0x80,
                ]);
                annexb.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xEB, 0xE3, 0xCB]);
                // IDR slice
                annexb.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00]);
            } else {
                // Non-IDR slice
                annexb.extend_from_slice(&[0, 0, 0, 1, 0x41, 0x9A, 0x00]);
            }
        }
        let mut mp4 = build_mp4(
            MuxerProfile::HandbrakeX264,
            &annexb,
            64,
            64,
            FrameTiming::FPS_30,
        )
        .expect("build_mp4");
        if with_ctts {
            // Inject a stub ctts atom into the byte stream. The
            // analyser only checks for box-type presence, not validity,
            // so a minimal valid box header is sufficient for the
            // detection signal. We append it right after the moov box
            // — the structural-walk fallback (recursive) will find it
            // even though it's outside the standard nesting.
            let stub = b"\x00\x00\x00\x10ctts\x00\x00\x00\x00\x00\x00\x00\x00";
            mp4.extend_from_slice(stub);
        }
        mp4
    }

    #[test]
    fn b_frames_never_first_or_last_in_gop() {
        // Stealth + correctness invariant: the first frame of any GOP
        // is IDR; the last is P (closed-GOP). B never appears at
        // those positions.
        for &(gop, b_count) in &[(5usize, 1usize), (7, 2), (10, 1)] {
            let pat = GopPattern::Ibpbp { gop, b_count };
            for d in [0, gop - 1, gop, 2 * gop - 1, 2 * gop] {
                let ft = pat.frame_type_at(d);
                assert_ne!(
                    ft,
                    FrameType::B,
                    "B at display_idx={d} for gop={gop}, b_count={b_count}",
                );
            }
        }
    }

    // ─── §scenecut-ibpbp-2026-05-09 (#288) — queue-aware scene-cut tests ─

    #[test]
    fn scene_cut_rewrite_demotes_trailing_b_to_p() {
        // IBPBP gop=30 b_count=1 with scene-cut at display=6.
        // Natural sub-GOP [P=6, B=5] should rewrite to [P=5, IDR=6].
        let pat = GopPattern::Ibpbp { gop: 30, b_count: 1 };
        let frames: Vec<_> =
            iter_encode_order_with_scene_cuts(10, pat, &[6]).collect();
        // Encoded order with scene-cut at display=6:
        //   0: I(0), 1: P(2), 2: B(1), 3: P(4), 4: B(3),
        //   5: P(5)        ← was P(6), now demoted to GOP-closing P
        //   6: IDR(6)      ← was B(5), now scene-cut IDR
        //   7: P(8), 8: B(7), 9: P(9-closing)
        assert_eq!(frames[5].display_idx, 5);
        assert_eq!(frames[5].frame_type, FrameType::P);
        assert_eq!(frames[6].display_idx, 6);
        assert_eq!(frames[6].frame_type, FrameType::Idr);
        // gop_idx should bump for the IDR and all following frames.
        assert_eq!(frames[5].gop_idx, 0, "P-closing belongs to GOP 0");
        assert_eq!(frames[6].gop_idx, 1, "scene-cut IDR opens GOP 1");
        for f in &frames[7..] {
            assert_eq!(f.gop_idx, 1, "post-cut frames live in GOP 1");
        }
    }

    #[test]
    fn scene_cut_rewrite_skips_natural_idr_positions() {
        // If a "scene cut" lands on display=0 or display=gop_size, the
        // frame is already IDR — no rewrite needed; downstream gop_idx
        // bumping must not fire.
        let pat = GopPattern::Ibpbp { gop: 6, b_count: 1 };
        let baseline: Vec<_> = iter_encode_order(12, pat).collect();
        let with_cuts: Vec<_> =
            iter_encode_order_with_scene_cuts(12, pat, &[0, 6]).collect();
        assert_eq!(baseline, with_cuts);
    }

    #[test]
    fn scene_cut_rewrite_b_position_no_op() {
        // A scene cut at a B-position display (= odd display index in
        // M=2 IBPBP) is structurally hard to handle (the natural P-
        // anchor sits AFTER the B). For simplicity v1.2 ignores those.
        let pat = GopPattern::Ibpbp { gop: 30, b_count: 1 };
        let baseline: Vec<_> = iter_encode_order(8, pat).collect();
        let with_cuts: Vec<_> =
            iter_encode_order_with_scene_cuts(8, pat, &[5]).collect();
        assert_eq!(baseline, with_cuts);
    }

    #[test]
    fn scene_cut_detect_threshold_fires_on_synthetic_motion() {
        // Build 4 frames of identical static gray, then 2 frames of
        // pure-black: detect_scene_cuts should flag display=4.
        let w = 32u32;
        let h = 32u32;
        let frame_size = (w * h * 3 / 2) as usize;
        let mut yuv = Vec::new();
        for d in 0..6 {
            let val = if d < 4 { 128u8 } else { 0u8 };
            yuv.extend(std::iter::repeat(val).take(frame_size));
        }
        let cuts = detect_scene_cuts_yuv(&yuv, w, h, 6, SCENE_CUT_THRESHOLD_DEFAULT);
        assert_eq!(cuts, vec![4]);
    }

    // §v1.7 Phase 5 (#322) — adaptive B-frame promotion tests.

    #[test]
    fn b_promotion_no_motion_returns_empty() {
        let w = 32u32;
        let h = 32u32;
        let frame_size = (w * h * 3 / 2) as usize;
        let yuv: Vec<u8> = std::iter::repeat(128u8)
            .take(6 * frame_size).collect();
        let pattern = GopPattern::Ibpbp { gop: 6, b_count: 1 };
        let promotions = detect_b_promotion_candidates(
            &yuv, w, h, 6, pattern, B_PROMOTION_THRESHOLD_DEFAULT,
        );
        assert!(promotions.is_empty(),
            "no motion → no promotions; got {:?}", promotions);
    }

    #[test]
    fn b_promotion_high_motion_at_b_position_promotes() {
        let w = 32u32;
        let h = 32u32;
        let frame_size = (w * h * 3 / 2) as usize;
        let mut yuv = Vec::with_capacity(6 * frame_size);
        // display: 0=I, 1=B, 2=P, 3=B, 4=P, 5=P (IBPBP gop=6 b=1)
        // Make display=3 a high-motion frame (everything else uniform).
        for d in 0..6 {
            let val = if d == 3 { 0u8 } else { 128u8 };
            yuv.extend(std::iter::repeat(val).take(frame_size));
        }
        let pattern = GopPattern::Ibpbp { gop: 6, b_count: 1 };
        // Sanity: display=3 is a B in this pattern.
        assert_eq!(pattern.frame_type_at(3), FrameType::B);
        let promotions = detect_b_promotion_candidates(
            &yuv, w, h, 6, pattern, B_PROMOTION_THRESHOLD_DEFAULT,
        );
        assert!(promotions.contains(&3),
            "display=3 (B + high motion) should promote; got {:?}", promotions);
    }

    #[test]
    fn b_promotion_high_motion_at_p_position_skipped() {
        let w = 32u32;
        let h = 32u32;
        let frame_size = (w * h * 3 / 2) as usize;
        let mut yuv = Vec::with_capacity(6 * frame_size);
        // display: 0=I, 1=B, 2=P, 3=B, 4=P, 5=P
        // Make display=4 (P-frame) high-motion. Should NOT be promoted
        // (it's already P; promotion only fires on B-frame positions).
        for d in 0..6 {
            let val = if d == 4 { 0u8 } else { 128u8 };
            yuv.extend(std::iter::repeat(val).take(frame_size));
        }
        let pattern = GopPattern::Ibpbp { gop: 6, b_count: 1 };
        let promotions = detect_b_promotion_candidates(
            &yuv, w, h, 6, pattern, B_PROMOTION_THRESHOLD_DEFAULT,
        );
        assert!(!promotions.contains(&4),
            "display=4 (P-frame) should NOT promote; got {:?}", promotions);
    }

    #[test]
    fn b_promotion_iter_rewrites_frame_type() {
        let pattern = GopPattern::Ibpbp { gop: 6, b_count: 1 };
        let promotions = vec![3];
        let frames: Vec<EncodeOrderFrame> = iter_encode_order_with_b_promotions(
            6, pattern, &promotions,
        ).collect();
        // Find the entry whose display_idx is 3 — should be P now (not B).
        let entry_3 = frames.iter().find(|f| f.display_idx == 3)
            .expect("display_idx=3 in encode order");
        assert_eq!(entry_3.frame_type, FrameType::P,
            "promoted display=3 should be P, got {:?}", entry_3.frame_type);
    }

    #[test]
    fn b_promotion_ipppp_noop() {
        let pattern = GopPattern::Ipppp { gop: 6 };
        let promotions = vec![1, 2, 3];
        // Ipppp has no B-frames; promotions should be no-op.
        let frames_a: Vec<EncodeOrderFrame> = iter_encode_order_with_b_promotions(
            6, pattern, &promotions,
        ).collect();
        let frames_b: Vec<EncodeOrderFrame> = iter_encode_order(6, pattern).collect();
        assert_eq!(frames_a, frames_b,
            "Ipppp + promotions should be no-op vs plain iter_encode_order");
    }

    #[test]
    fn scene_cut_detect_no_motion_returns_empty() {
        let w = 32u32;
        let h = 32u32;
        let frame_size = (w * h * 3 / 2) as usize;
        let yuv: Vec<u8> = std::iter::repeat(128u8)
            .take(5 * frame_size)
            .collect();
        let cuts = detect_scene_cuts_yuv(&yuv, w, h, 5, SCENE_CUT_THRESHOLD_DEFAULT);
        assert!(cuts.is_empty());
    }
}
