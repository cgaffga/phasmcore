// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Display-order ↔ encode-order reorder buffer for B-frame support.
//!
//! H.264 separates the order frames are PRESENTED to the user
//! (display order) from the order they're WRITTEN to the bitstream
//! (encode order). With B-frames, a B-frame at display index N is
//! encoded AFTER the next P-frame at display index N+1, because
//! the B needs that P as its L1 reference. The decoder reorders
//! at presentation time using POC fields.
//!
//! ## Scope (Phase 6E-A1) — M=2 IBPBP
//!
//! The simplest non-trivial B-frame configuration: every other
//! frame is a B (display order `I, B, P, B, P, B, P, …`).
//! Encode order: `I, P, B, P, B, P, B, …` (each P is encoded
//! before the B that displays just before it).
//!
//! This module ships the reorder STATE MACHINE — given a stream
//! of display-order frame YUV inputs, it emits encode-order
//! `(yuv, role)` pairs where `role` carries the B-frame's
//! reference indices and POC info. NO encoder integration yet
//! (§6E-A1 is infrastructure only); the encode driver will plumb
//! this in later sub-phases.
//!
//! ## API shape
//!
//! ```text
//! ReorderBuffer::new(gop_length, m_factor)
//! buf.push(display_yuv)              // returns Vec<EncodeFrame>
//! buf.flush()                        // returns final B-frames at EOS
//! ```
//!
//! At each `push(yuv)`:
//! - If display index is divisible by `m_factor` (M=2 → even
//!   indices) OR is the IDR: emit immediately as `I` or `P`.
//!   Hold any pending B in the buffer until the NEXT P (the L1
//!   ref) is encoded.
//! - Else: buffer the YUV as a pending B-frame.
//!
//! Output order for `I, B, P, B, P, B, P, …` (display) becomes
//! `I, P, B, P, B, P, B, …` (encode) once the buffer flushes
//! after each P.

/// Frame role in the output sequence — what kind of slice this
/// frame should be encoded as, plus the display-order metadata
/// the encoder needs for POC + reference-list construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameRole {
    /// IDR (instantaneous decoder refresh). First frame, or every
    /// `gop_length`-th frame. Resets DPB + POC.
    Idr {
        display_index: u32,
    },
    /// Non-IDR P-frame. References the previous P (or I) in the
    /// DPB.
    P {
        display_index: u32,
    },
    /// Non-IDR B-frame. References the previous P (L0) and the
    /// next P (L1). Both must be in the DPB before this B is
    /// encoded.
    B {
        display_index: u32,
        l0_display_index: u32,
        l1_display_index: u32,
    },
}

impl FrameRole {
    pub fn display_index(&self) -> u32 {
        match self {
            FrameRole::Idr { display_index }
            | FrameRole::P { display_index }
            | FrameRole::B { display_index, .. } => *display_index,
        }
    }

    pub fn is_b(&self) -> bool {
        matches!(self, FrameRole::B { .. })
    }

    pub fn is_idr(&self) -> bool {
        matches!(self, FrameRole::Idr { .. })
    }
}

/// One frame about to be handed to the encoder, in encode order.
/// `yuv` is the I420 buffer the user provided at the
/// corresponding display index. `role` tells the encoder which
/// slice type to use and which DPB slots to reference.
#[derive(Debug)]
pub struct EncodeFrame {
    pub yuv: Vec<u8>,
    pub role: FrameRole,
}

/// Display-order ↔ encode-order reorder buffer.
///
/// Configuration:
/// - `gop_length`: frames per GOP (default 30 = 1 second at 30 fps).
///   The frame at display index 0, gop_length, 2*gop_length, … is
///   IDR.
/// - `m_factor`: GOP M-parameter. M=1 means no B-frames (legacy
///   behavior). M=2 means alternating B/P (`IBPBP…`). M=3 would
///   mean `IBBP…`; deferred to a follow-up tweak post §6E-A6.
///
/// State machine: holds at most ONE pending B-frame for M=2; for
/// M>2 would generalize to a queue of length `m_factor − 1`.
#[derive(Debug)]
pub struct ReorderBuffer {
    pub gop_length: u32,
    pub m_factor: u32,
    /// Display-order index of the next frame the user will push.
    next_display_index: u32,
    /// Pending B-frames awaiting their L1 reference (the next P).
    /// For M=2 this is at most one entry.
    pending_b: Vec<(u32, Vec<u8>)>,
    /// Most-recently-emitted P-frame's display index. Used to wire
    /// pending B-frames to their L0 reference. `None` before the
    /// first P is emitted in the current GOP.
    last_p_display: Option<u32>,
}

impl ReorderBuffer {
    pub fn new(gop_length: u32, m_factor: u32) -> Self {
        assert!(m_factor >= 1, "m_factor must be >= 1");
        assert!(gop_length >= 1, "gop_length must be >= 1");
        Self {
            gop_length,
            m_factor,
            next_display_index: 0,
            pending_b: Vec::with_capacity((m_factor - 1) as usize),
            last_p_display: None,
        }
    }

    /// Reset for a new clip / sequence boundary. Drops any
    /// pending B-frames (caller's responsibility to flush first).
    pub fn reset(&mut self) {
        self.next_display_index = 0;
        self.pending_b.clear();
        self.last_p_display = None;
    }

    /// Push the next display-order frame. Returns 0 or more
    /// `EncodeFrame`s to hand to the encoder, in encode order.
    ///
    /// Empty return = the frame was buffered as a pending B; the
    /// next push (the next P) will release both this and the
    /// buffered B in encode order.
    pub fn push(&mut self, yuv: Vec<u8>) -> Vec<EncodeFrame> {
        let display_index = self.next_display_index;
        self.next_display_index += 1;

        // Is this an IDR position?
        let is_idr = display_index.is_multiple_of(self.gop_length);
        // Is this an "anchor" position (IDR or P)? At M=2, even
        // display indices within the GOP are anchors; at M=1 every
        // index is an anchor (legacy P-only behavior).
        let is_anchor =
            is_idr || (display_index - self.gop_position_zero(display_index)).is_multiple_of(self.m_factor);

        if is_idr {
            // IDR closes any pending B awkwardly — caller should
            // have flushed before the GOP boundary. Drop any
            // pending B-frames as they have no L1 ref (the IDR
            // resets the DPB).
            self.pending_b.clear();
            // IDR itself is the first L0 anchor for the new GOP.
            // The next P-frame in this GOP will have L0 = IDR;
            // any B between IDR and the next P will have
            // L0 = IDR, L1 = next P.
            self.last_p_display = Some(display_index);
            return vec![EncodeFrame {
                yuv,
                role: FrameRole::Idr { display_index },
            }];
        }

        if is_anchor {
            // P-frame. Emit it FIRST, then any pending B-frames
            // (now they have both refs: prev P as L0, this P as L1).
            let mut out = Vec::with_capacity(1 + self.pending_b.len());
            out.push(EncodeFrame {
                yuv,
                role: FrameRole::P { display_index },
            });
            let l0 = self
                .last_p_display
                .expect("anchor frame must have a previous P or IDR");
            for (b_display, b_yuv) in self.pending_b.drain(..) {
                out.push(EncodeFrame {
                    yuv: b_yuv,
                    role: FrameRole::B {
                        display_index: b_display,
                        l0_display_index: l0,
                        l1_display_index: display_index,
                    },
                });
            }
            self.last_p_display = Some(display_index);
            return out;
        }

        // Non-anchor → B-frame. Buffer until next anchor.
        self.pending_b.push((display_index, yuv));
        Vec::new()
    }

    /// Emit any remaining pending B-frames at end-of-stream.
    /// In typical operation the input clip ends on a P-frame
    /// boundary; if it doesn't, the final B-frames have no L1
    /// reference and must be re-purposed as P-frames or dropped.
    /// This implementation drops them (the caller can detect by
    /// checking `pending_count_at_eos()` before flushing if it
    /// wants different behavior).
    pub fn flush(&mut self) -> Vec<EncodeFrame> {
        // For M=2 IBPBP, end-on-B is uncommon; drop without
        // recovery for simplicity. Caller can pre-check if needed.
        self.pending_b.clear();
        self.last_p_display = None;
        Vec::new()
    }

    /// Number of pending B-frames currently buffered. > 0 only
    /// between push of a B and the next push of a P.
    pub fn pending_count(&self) -> usize {
        self.pending_b.len()
    }

    /// Round `display_index` down to its GOP-start (the most
    /// recent IDR position). Used to compute "anchor offset within
    /// GOP" which determines P vs B placement under M-factor.
    fn gop_position_zero(&self, display_index: u32) -> u32 {
        (display_index / self.gop_length) * self.gop_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_yuv(seed: u8) -> Vec<u8> {
        // 32×32 I420 = 32×32 + 16×16 + 16×16 = 1024 + 256 + 256 = 1536.
        vec![seed; 1536]
    }

    #[test]
    fn m1_emits_p_immediately() {
        // Legacy: M=1 = no B-frames. Every push emits one frame.
        let mut buf = ReorderBuffer::new(/* gop */ 30, /* M */ 1);
        let out = buf.push(dummy_yuv(0));
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0].role, FrameRole::Idr { display_index: 0 }));
        let out = buf.push(dummy_yuv(1));
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0].role, FrameRole::P { display_index: 1 }));
    }

    #[test]
    fn m2_alternates_p_and_b_in_encode_order() {
        // Display order: I₀ B₁ P₂ B₃ P₄ B₅ P₆ …
        // Encode order:  I₀ P₂ B₁ P₄ B₃ P₆ B₅ …
        let mut buf = ReorderBuffer::new(/* gop */ 30, /* M */ 2);

        // IDR at display 0 → emits immediately.
        let out = buf.push(dummy_yuv(0));
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0].role, FrameRole::Idr { display_index: 0 }));

        // Display 1 (B) → buffered.
        let out = buf.push(dummy_yuv(1));
        assert_eq!(out.len(), 0);
        assert_eq!(buf.pending_count(), 1);

        // Display 2 (P) → emits P₂ then released B₁.
        let out = buf.push(dummy_yuv(2));
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0].role, FrameRole::P { display_index: 2 }));
        assert!(matches!(
            out[1].role,
            FrameRole::B {
                display_index: 1,
                l0_display_index: 0,
                l1_display_index: 2
            }
        ));
        assert_eq!(buf.pending_count(), 0);

        // Display 3 (B) → buffered.
        let out = buf.push(dummy_yuv(3));
        assert_eq!(out.len(), 0);

        // Display 4 (P) → emits P₄ then B₃.
        let out = buf.push(dummy_yuv(4));
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0].role, FrameRole::P { display_index: 4 }));
        assert!(matches!(
            out[1].role,
            FrameRole::B {
                display_index: 3,
                l0_display_index: 2,
                l1_display_index: 4
            }
        ));
    }

    #[test]
    fn idr_at_gop_boundary_resets() {
        // gop_length = 4, M = 2. Display: I₀ B₁ P₂ B₃ I₄ B₅ P₆ B₇ …
        let mut buf = ReorderBuffer::new(/* gop */ 4, /* M */ 2);

        let _ = buf.push(dummy_yuv(0)); // IDR
        let _ = buf.push(dummy_yuv(1)); // B (buffered)
        let _ = buf.push(dummy_yuv(2)); // P → emits P₂, B₁
        // Push display 3 (B). Buffered.
        let out = buf.push(dummy_yuv(3));
        assert_eq!(out.len(), 0);
        assert_eq!(buf.pending_count(), 1);

        // Display 4 = next GOP IDR. The pending B at display 3
        // gets dropped (no valid L1 ref across the GOP boundary;
        // caller responsibility to set gop_length so this doesn't
        // happen, or accept the loss).
        let out = buf.push(dummy_yuv(4));
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0].role, FrameRole::Idr { display_index: 4 }));
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn flush_drops_pending() {
        let mut buf = ReorderBuffer::new(30, 2);
        let _ = buf.push(dummy_yuv(0));
        let _ = buf.push(dummy_yuv(1));
        assert_eq!(buf.pending_count(), 1);
        let out = buf.flush();
        assert_eq!(out.len(), 0);
        assert_eq!(buf.pending_count(), 0);
    }

    #[test]
    fn frame_role_helpers() {
        assert!(FrameRole::Idr { display_index: 0 }.is_idr());
        assert!(!FrameRole::Idr { display_index: 0 }.is_b());
        assert!(FrameRole::B {
            display_index: 1,
            l0_display_index: 0,
            l1_display_index: 2,
        }.is_b());
        assert!(!FrameRole::P { display_index: 2 }.is_b());
        assert_eq!(
            FrameRole::B {
                display_index: 5,
                l0_display_index: 4,
                l1_display_index: 6,
            }
            .display_index(),
            5
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut buf = ReorderBuffer::new(30, 2);
        let _ = buf.push(dummy_yuv(0));
        let _ = buf.push(dummy_yuv(1));
        buf.reset();
        assert_eq!(buf.pending_count(), 0);
        // Next push behaves as fresh start.
        let out = buf.push(dummy_yuv(0));
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0].role, FrameRole::Idr { display_index: 0 }));
    }
}
