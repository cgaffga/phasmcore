// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Picture Order Count (POC) tracker for B-frame support.
//!
//! H.264 separates encode order from display order. With B-frames,
//! a B-frame at display index N is encoded AFTER the next P-frame
//! at display index N+1 (the B needs that P as its L1 reference).
//! The bitstream carries POC fields (`pic_order_cnt_lsb` etc.) so
//! the decoder can reorder frames for display.
//!
//! ## Scope (Phase 6E-A1)
//!
//! - `pic_order_cnt_type = 0` (LSB-based, simplest scheme).
//! - `log2_max_pic_order_cnt_lsb_minus4 = 4` → POC LSB range
//!   `[0, 255]`.
//! - **POC = 2 × display_index** for I/P/B at top-field semantics
//!   in non-interlaced content. Each display step advances POC by
//!   2 (the spec reserves odd values for bottom fields, never
//!   used in our progressive-only encoder).
//! - Wrapping: when full POC reaches `2 << log2_max_lsb` it wraps;
//!   `pic_order_cnt_lsb` is `full_poc mod (1 << (log2_max_lsb + 4))`.
//!
//! Spec refs: § 8.2.1 (decoding of POC), § 7.4.3 (slice header
//! POC fields).

/// `log2_max_pic_order_cnt_lsb_minus4` SPS field. With value 4,
/// `MaxPicOrderCntLsb = 1 << (4 + 4) = 256`. Adequate for short
/// clips — at 30 fps a wrap takes ~8.5 seconds. For longer clips
/// the encoder must monotonically wrap; the decoder follows via
/// the spec's pic_order_cnt LSB→MSB extension algorithm
/// (§ 8.2.1.1) which handles wrap correctly.
pub const POC_LOG2_MAX_LSB_MINUS4: u8 = 4;

/// `MaxPicOrderCntLsb` derived constant per spec.
pub const POC_MAX_LSB: u32 = 1u32 << (POC_LOG2_MAX_LSB_MINUS4 as u32 + 4);

/// POC tracker for the Phase 6E-A scope (`pic_order_cnt_type = 0`,
/// progressive-only, M=2 IBPBP).
///
/// Owned by `Encoder`. Each call to `display_index_to_poc` returns
/// the LSB to emit in the slice header for that display index;
/// internally tracks the full POC for wrap calculations and so
/// downstream can compute LSB at arbitrary display indices.
#[derive(Debug, Clone, Copy)]
pub struct PocTracker {
    /// IDR display index of the current GOP. Reset on every IDR.
    /// All POCs in this GOP are computed relative to this anchor.
    idr_display_index: u32,
}

impl PocTracker {
    /// Build a fresh tracker. The first IDR encoded sets the
    /// anchor via `reset_at_idr(display_index)`.
    pub fn new() -> Self {
        Self {
            idr_display_index: 0,
        }
    }

    /// Reset POC anchor at an IDR. The IDR's POC is always 0.
    /// Subsequent frames in this GOP get POC = 2 × (display_index −
    /// idr_display_index), wrapped mod `POC_MAX_LSB` for the LSB
    /// that lands in the slice header.
    pub fn reset_at_idr(&mut self, display_index: u32) {
        self.idr_display_index = display_index;
    }

    /// Compute the `pic_order_cnt_lsb` to emit for a frame at
    /// `display_index`. Caller must have called `reset_at_idr`
    /// for the current GOP first.
    pub fn poc_lsb_for(&self, display_index: u32) -> u32 {
        let full_poc =
            2u32.wrapping_mul(display_index.wrapping_sub(self.idr_display_index));
        full_poc & (POC_MAX_LSB - 1)
    }

    /// Compute the FULL (un-wrapped) POC for a frame at
    /// `display_index`. Used for DPB ordering / reference-list
    /// construction; not emitted directly.
    pub fn full_poc_for(&self, display_index: u32) -> u32 {
        2u32.wrapping_mul(display_index.wrapping_sub(self.idr_display_index))
    }
}

impl Default for PocTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idr_poc_is_zero() {
        let mut t = PocTracker::new();
        t.reset_at_idr(0);
        assert_eq!(t.poc_lsb_for(0), 0);
        assert_eq!(t.full_poc_for(0), 0);
    }

    #[test]
    fn step_of_two_per_display_frame() {
        let mut t = PocTracker::new();
        t.reset_at_idr(0);
        assert_eq!(t.poc_lsb_for(1), 2);
        assert_eq!(t.poc_lsb_for(2), 4);
        assert_eq!(t.poc_lsb_for(3), 6);
        assert_eq!(t.poc_lsb_for(10), 20);
    }

    #[test]
    fn idr_anchor_offsets_correctly() {
        let mut t = PocTracker::new();
        // First GOP: IDR at display index 0.
        t.reset_at_idr(0);
        assert_eq!(t.poc_lsb_for(5), 10);
        // Second GOP: IDR at display index 30 (1-second GOP at 30 fps).
        t.reset_at_idr(30);
        assert_eq!(t.poc_lsb_for(30), 0); // IDR itself
        assert_eq!(t.poc_lsb_for(31), 2);
        assert_eq!(t.poc_lsb_for(35), 10);
    }

    #[test]
    fn lsb_wraps_at_max() {
        let mut t = PocTracker::new();
        t.reset_at_idr(0);
        // POC LSB = 2 × (display_index − idr) mod 256.
        // At display 128: full POC = 256, LSB wraps to 0.
        assert_eq!(t.poc_lsb_for(128), 0);
        // At display 129: full POC = 258, LSB = 2.
        assert_eq!(t.poc_lsb_for(129), 2);
        // Full POC continues counting un-wrapped.
        assert_eq!(t.full_poc_for(128), 256);
        assert_eq!(t.full_poc_for(200), 400);
    }

    #[test]
    fn poc_max_lsb_constant_matches_spec() {
        // log2_max_pic_order_cnt_lsb_minus4 = 4 → MaxPicOrderCntLsb
        // = 1 << (4 + 4) = 256.
        assert_eq!(POC_MAX_LSB, 256);
    }
}
