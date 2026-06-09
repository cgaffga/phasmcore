// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! GOP-pattern descriptor + encode-order iterator.
//!
//! H.264 video stego emits IBPBP-shaped bitstreams (Apple-iPhone
//! canonical) instead of IPPPP for stealth fingerprint match (Layer 3).
//! The bitstream itself is produced by the OpenH264 fork (the sole
//! encoder after the 2026-06 video-retirement deleted the pure-Rust
//! encoder + `encode_pixels.rs`); this module is purely the
//! FRAME-TYPE / ITERATION-ORDER descriptor.
//!
//! `iter_encode_order` is consumed by the MP4 muxer
//! (`mp4::build::build_mp4_with_pattern`), which derives each sample's
//! per-frame `ctts` composition-time offset (`display_idx - encode_idx`)
//! from the encode-order metadata so B-frame display timing matches
//! HandBrake/x264 output, and by `oh264_capacity` for per-GOP capacity
//! sizing.
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
//! reference). The OpenH264 encoder's own reorder buffer (drained via
//! `FlushFrame`) handles the per-frame state; this module's contract is
//! to feed frames in the correct sequence.
//!
//! ## PositionKey alignment
//!
//! The bin-decoder slice walker reads bytes in encode order — it has
//! no display-order context. For encode ↔ walker symmetry, the
//! `PositionKey.frame_idx` field carries the **encode-order** index,
//! NOT display order. `iter_encode_order::encode_idx` is the
//! encode-order index the OH264 stego path stamps into each emitted
//! `PositionKey.frame_idx`, so it aligns with the walker's
//! encode-order parsing.
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

/// Per-frame type within a GOP. Three-way classifier; the MP4 muxer
/// reads it (via `iter_encode_order`) to decide composition-time
/// (`ctts`) emission — `any B-frame ⇒ emit ctts` — and per-GOP
/// capacity sizing keys off it too.
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
/// `Ipppp` is the legacy pre-B-frame shape (no B-frames;
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

    /// Source-adaptive `gop_pattern: Auto` selection.
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
    ///   (<5% of the HandBrake corpus); collapsing to M=2
    ///   keeps phasm in the populous mode.
    /// - Demux failure / malformed input → centroid (silently safe).
    pub fn auto_select(source_mp4: Option<&[u8]>) -> Self {
        let bytes = match source_mp4 {
            Some(b) => b,
            None => return Self::handbrake_x264_centroid(),
        };
        analyze_source_pattern(bytes).unwrap_or_else(Self::handbrake_x264_centroid)
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
    /// This is the value the OH264 stego path stamps into this frame's
    /// `PositionKey.frame_idx`, so it aligns with the walker's
    /// encode-order parsing.
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
                        // When n_frames truncates a sub-GOP, the natural
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

// ─── Source-adaptive analyser ─────────────────────────────────────────

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
        assert_eq!(pat.gop_size(), 30);
        if let GopPattern::Ibpbp { gop, b_count } = pat {
            assert_eq!(gop, 30);
            assert_eq!(b_count, 1);
        } else {
            panic!("iphone_default should be Ibpbp");
        }
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

}
