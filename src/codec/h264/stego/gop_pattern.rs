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
                        let anchor_ft = pattern.frame_type_at(anchor);
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
