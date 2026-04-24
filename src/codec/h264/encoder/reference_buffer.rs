// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Reference frame management for Phase 6B P-frame encoding.
//!
//! Single-slot DPB — holds the most-recently-encoded frame's
//! reconstructed pixels. The next P-frame's motion compensation
//! reads from this. IDR frames clear the slot. See
//!   docs/design/h264-encoder-algorithms/reference-management.md
//!
//! Intentionally minimal. Multi-ref DPBs, reordering, long-term
//! refs, and temporal scalability are all out-of-scope for phasm's
//! short-clip workload (logged in deferred-items.md).

use super::reconstruction::ReconBuffer;

/// Immutable snapshot of reconstructed YUV420p planes.
#[derive(Debug, Clone)]
pub struct ReconFrame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
}

impl ReconFrame {
    /// Snapshot a `ReconBuffer`'s current state.
    pub fn snapshot(recon: &ReconBuffer) -> Self {
        Self {
            width: recon.width,
            height: recon.height,
            y: recon.y.clone(),
            cb: recon.cb.clone(),
            cr: recon.cr.clone(),
        }
    }

    /// Read a single luma pixel. Panics if out of bounds.
    pub fn y_at(&self, x: u32, y: u32) -> u8 {
        self.y[(y * self.width + x) as usize]
    }

    /// Read a single chroma pixel. Panics if out of bounds.
    pub fn chroma_at(&self, component: u8, x: u32, y: u32) -> u8 {
        let stride = self.width / 2;
        let plane = if component == 0 { &self.cb } else { &self.cr };
        plane[(y * stride + x) as usize]
    }
}

/// Single-slot DPB. Holds the previous frame for P-frame prediction.
#[derive(Debug, Default)]
pub struct ReferenceBuffer {
    pub last_ref: Option<ReconFrame>,
    /// `frame_num` of the last-emitted reference, wraps at
    /// `1 << log2_max_frame_num`. `None` when `last_ref.is_none()`.
    pub last_ref_frame_num: Option<u8>,
}

impl ReferenceBuffer {
    pub fn new() -> Self {
        Self {
            last_ref: None,
            last_ref_frame_num: None,
        }
    }

    /// Clear the DPB — called on every IDR.
    pub fn reset(&mut self) {
        self.last_ref = None;
        self.last_ref_frame_num = None;
    }

    /// Replace the current reference with a snapshot of the
    /// just-encoded frame.
    pub fn promote(&mut self, recon: &ReconBuffer, frame_num: u8) {
        self.last_ref = Some(ReconFrame::snapshot(recon));
        self.last_ref_frame_num = Some(frame_num);
    }

    /// True when a P-frame can be encoded (reference is available).
    pub fn has_reference(&self) -> bool {
        self.last_ref.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_recon(width: u32, height: u32, y_fill: u8, c_fill: u8) -> ReconBuffer {
        let mut b = ReconBuffer::new(width, height).unwrap();
        for v in b.y.iter_mut() {
            *v = y_fill;
        }
        for v in b.cb.iter_mut() {
            *v = c_fill;
        }
        for v in b.cr.iter_mut() {
            *v = c_fill;
        }
        b
    }

    #[test]
    fn reference_buffer_new_is_empty() {
        let rb = ReferenceBuffer::new();
        assert!(!rb.has_reference());
        assert!(rb.last_ref.is_none());
        assert!(rb.last_ref_frame_num.is_none());
    }

    #[test]
    fn promote_captures_pixels() {
        let mut rb = ReferenceBuffer::new();
        let recon = make_recon(32, 32, 100, 128);
        rb.promote(&recon, 0);
        assert!(rb.has_reference());
        assert_eq!(rb.last_ref_frame_num, Some(0));
        let frame = rb.last_ref.as_ref().unwrap();
        assert_eq!(frame.width, 32);
        assert_eq!(frame.height, 32);
        assert_eq!(frame.y_at(5, 5), 100);
        assert_eq!(frame.chroma_at(0, 5, 5), 128);
        assert_eq!(frame.chroma_at(1, 5, 5), 128);
    }

    #[test]
    fn reset_clears_dpb() {
        let mut rb = ReferenceBuffer::new();
        let recon = make_recon(32, 32, 100, 128);
        rb.promote(&recon, 5);
        rb.reset();
        assert!(!rb.has_reference());
        assert_eq!(rb.last_ref_frame_num, None);
    }

    #[test]
    fn promote_replaces_previous() {
        let mut rb = ReferenceBuffer::new();
        rb.promote(&make_recon(32, 32, 50, 128), 0);
        rb.promote(&make_recon(32, 32, 200, 100), 1);
        let frame = rb.last_ref.as_ref().unwrap();
        assert_eq!(frame.y_at(0, 0), 200);
        assert_eq!(frame.chroma_at(0, 0, 0), 100);
        assert_eq!(rb.last_ref_frame_num, Some(1));
    }

    #[test]
    fn snapshot_is_independent_of_source() {
        let mut recon = make_recon(32, 32, 100, 128);
        let mut rb = ReferenceBuffer::new();
        rb.promote(&recon, 0);
        // Mutate the source after snapshot.
        for v in recon.y.iter_mut() {
            *v = 200;
        }
        // Snapshot should be unaffected.
        let frame = rb.last_ref.as_ref().unwrap();
        assert_eq!(frame.y_at(0, 0), 100);
    }
}
