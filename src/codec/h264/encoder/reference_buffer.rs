// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Reference frame management for Phase 6B P-frame encoding +
//! Phase 6E-A B-frame extension.
//!
//! **Single-slot DPB (Phase 6B, default)**: holds the most-recently-
//! encoded frame's reconstructed pixels. The next P-frame's motion
//! compensation reads from this. IDR frames clear the slot.
//!
//! **Multi-slot DPB (Phase 6E-A1, opt-in)**: holds up to 3 slots
//! to support B-frame encoding under M=2 IBPBP. Slot 0 = previous
//! P (L0 ref for the current B and the next P). Slot 1 = next P
//! encoded BEFORE the displayed-between B (L1 ref for that B).
//! Slot 2 = IDR carry-over slack at GOP boundaries.
//!
//! See `docs/design/h264-encoder-algorithms/reference-management.md`
//! and `docs/design/h264-encoder-algorithms/b-frames.md`.
//!
//! ## Backwards compatibility
//!
//! The legacy `ReferenceBuffer { last_ref, last_ref_frame_num }`
//! API stays — Phase 6B P-frame paths are bit-identical. The
//! multi-slot extension lives in a sibling type `MultiSlotDpb`
//! exposed under the same module. Phase 6E-A2+ wires the encoder
//! to use `MultiSlotDpb` for B-slice encoding paths only; P-only
//! encodes continue using the legacy `ReferenceBuffer`.

use super::reconstruction::ReconBuffer;

/// §B-direct-fix — per-MB motion data captured from a P/B-frame's
/// final mode decisions, snapshotted into [`ReconFrame`] so that the
/// NEXT B-frame's spatial-direct prediction can read this frame's
/// "colMb" MV per spec § 8.4.1.2.2.
///
/// Indexed by MB position in raster order. For B-frame `B_n` whose
/// L1 reference is the P-frame `P_{n+1}`, the spatial-direct
/// algorithm reads `colMb` = the MB at `(mb_x, mb_y)` in `P_{n+1}`'s
/// motion grid — i.e., the L1 reference's `motion_grid[mb_y][mb_x]`.
///
/// Only L0 motion is stored for now: P-frames are pure-L0 (no L1
/// ref); B-frame storage is a follow-on if/when phasm uses
/// B-as-reference (currently `nal_ref_idc=0` so B-frames don't
/// land in the DPB). Spec § 8.4.1.2.2's `colZeroFlag` derivation
/// reads colMb's L0 (and L1, when present); for our IBPBP M=2 the
/// L1 reference is always a P-frame so the L1-side is moot here.
#[derive(Debug, Clone)]
pub struct ColocatedMvCell {
    /// L0 reference index used by this MB. -1 = list unused
    /// (e.g. intra MB or B-MB that uses only L1).
    pub ref_idx_l0: i8,
    /// L0 motion vector in quarter-pel units. (0, 0) when the MB
    /// is intra or otherwise has no L0 motion.
    pub mv_l0_x: i16,
    pub mv_l0_y: i16,
}

impl ColocatedMvCell {
    pub const INTRA: Self = Self {
        ref_idx_l0: -1,
        mv_l0_x: 0,
        mv_l0_y: 0,
    };
}

/// §B-direct-fix — per-frame collocated MV grid.
///
/// `cells[mb_y * mb_w + mb_x]` is the MB at MB-position `(mb_x, mb_y)`
/// in the frame this grid was snapshotted from. `mb_w` and `mb_h`
/// are macroblock counts (= `ceil(width / 16)` × `ceil(height / 16)`).
#[derive(Debug, Clone)]
pub struct ColocatedMvGrid {
    pub mb_w: u32,
    pub mb_h: u32,
    pub cells: Vec<ColocatedMvCell>,
}

impl ColocatedMvGrid {
    pub fn new(mb_w: u32, mb_h: u32) -> Self {
        Self {
            mb_w,
            mb_h,
            cells: vec![ColocatedMvCell::INTRA; (mb_w * mb_h) as usize],
        }
    }

    pub fn get(&self, mb_x: u32, mb_y: u32) -> &ColocatedMvCell {
        &self.cells[(mb_y * self.mb_w + mb_x) as usize]
    }

    pub fn set(&mut self, mb_x: u32, mb_y: u32, cell: ColocatedMvCell) {
        self.cells[(mb_y * self.mb_w + mb_x) as usize] = cell;
    }
}

/// Immutable snapshot of reconstructed YUV420p planes.
///
/// §B-direct-fix adds an optional `motion_grid` carrying the
/// per-MB motion state of the frame this was snapshotted from.
/// `Some(_)` for P-frame snapshots (and B-frame snapshots, but
/// we don't currently store B-frames as references); `None` for
/// I-frame snapshots and any path where motion data wasn't
/// captured.
#[derive(Debug, Clone)]
pub struct ReconFrame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    /// §B-direct-fix — per-MB collocated MV state. Read by B-frame
    /// spatial-direct prediction's `colZeroFlag` check (spec
    /// § 8.4.1.2.2).
    pub motion_grid: Option<ColocatedMvGrid>,
}

impl ReconFrame {
    /// Snapshot a `ReconBuffer`'s current state. Motion data is
    /// captured separately via [`Self::with_motion_grid`].
    pub fn snapshot(recon: &ReconBuffer) -> Self {
        Self {
            width: recon.width,
            height: recon.height,
            y: recon.y.clone(),
            cb: recon.cb.clone(),
            cr: recon.cr.clone(),
            motion_grid: None,
        }
    }

    /// §B-direct-fix — attach a motion grid to a snapshot.
    pub fn with_motion_grid(mut self, grid: ColocatedMvGrid) -> Self {
        self.motion_grid = Some(grid);
        self
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

/// Two-slot reference buffer. Holds the most-recently-encoded anchor
/// (`last_ref`) for P-frame L0 prediction, plus the previous anchor
/// (`past_anchor`) for B-frame L0 prediction under M=2 IBPBP encode
/// order.
///
/// **Encode-order timeline (M=2 IBPBP)**:
///
/// | step | encode | last_ref | past_anchor | next consumer |
/// |---|---|---|---|---|
/// | 0 | I    | I  | None | P0 reads last_ref=I |
/// | 1 | P0   | P0 | I    | B0 reads (past_anchor=I, last_ref=P0) |
/// | 2 | B0   | P0 | I    | (B doesn't promote — non-reference) |
/// | 3 | P1   | P1 | P0   | B1 reads (past_anchor=P0, last_ref=P1) |
/// | 4 | B1   | P1 | P0   | (no promote) |
/// | 5 | P2   | P2 | P1   | … |
///
/// Anchors encode in chronological / POC order (every P → P transition
/// has the older anchor's POC < newer's), so `past_anchor` is always
/// the L0 reference for the next B (smaller POC than B's POC), and
/// `last_ref` is always the L1 reference (larger POC).
///
/// **Backwards compat**: `last_ref` field + accessor preserved
/// bit-identical for P-frame paths; `past_anchor` is a new field
/// transparent to P-only encodes (always `None` after IDR until the
/// second post-IDR anchor is promoted, which is exactly when we'd
/// start encoding B-frames anyway).
#[derive(Debug, Default)]
pub struct ReferenceBuffer {
    pub last_ref: Option<ReconFrame>,
    /// `frame_num` of the last-emitted reference, wraps at
    /// `1 << log2_max_frame_num`. `None` when `last_ref.is_none()`.
    pub last_ref_frame_num: Option<u8>,
    /// §6E-A6.1q.a (#150) — previous anchor's reconstructed pixels.
    /// Set by `promote` to whatever `last_ref` held before being
    /// overwritten. Reads as the L0 reference for B-frame ME under
    /// M=2 IBPBP encode order.
    pub past_anchor: Option<ReconFrame>,
    /// `frame_num` of `past_anchor`. Mirrors `last_ref_frame_num`.
    pub past_anchor_frame_num: Option<u8>,
}

impl ReferenceBuffer {
    pub fn new() -> Self {
        Self {
            last_ref: None,
            last_ref_frame_num: None,
            past_anchor: None,
            past_anchor_frame_num: None,
        }
    }

    /// Clear both slots — called on every IDR.
    pub fn reset(&mut self) {
        self.last_ref = None;
        self.last_ref_frame_num = None;
        self.past_anchor = None;
        self.past_anchor_frame_num = None;
    }

    /// Promote the just-reconstructed frame into the buffer. Shifts
    /// the prior `last_ref` into `past_anchor` first, then overwrites
    /// `last_ref` with the new snapshot.
    ///
    /// B-frames must NOT call this — they're non-reference (nal_ref_idc=0)
    /// and don't enter the DPB. Callers gate via frame type.
    pub fn promote(&mut self, recon: &ReconBuffer, frame_num: u8) {
        // Shift current → past first. `take()` clears the source so
        // there's no double-borrow when assigning the new last_ref.
        self.past_anchor = self.last_ref.take();
        self.past_anchor_frame_num = self.last_ref_frame_num.take();
        self.last_ref = Some(ReconFrame::snapshot(recon));
        self.last_ref_frame_num = Some(frame_num);
    }

    /// §B-direct-fix — `promote` variant that ALSO attaches a
    /// per-MB collocated MV grid to the snapshot, so the next
    /// B-frame's spatial-direct prediction can read this frame's
    /// `colMb` MV per spec § 8.4.1.2.2.
    ///
    /// Caller (encoder side) is responsible for converting its
    /// `EncoderMvGrid` into a [`ColocatedMvGrid`] via
    /// `EncoderMvGrid::to_colocated_grid` before calling.
    pub fn promote_with_motion(
        &mut self,
        recon: &ReconBuffer,
        frame_num: u8,
        motion_grid: ColocatedMvGrid,
    ) {
        self.past_anchor = self.last_ref.take();
        self.past_anchor_frame_num = self.last_ref_frame_num.take();
        self.last_ref = Some(
            ReconFrame::snapshot(recon).with_motion_grid(motion_grid),
        );
        self.last_ref_frame_num = Some(frame_num);
    }

    /// True when a P-frame can be encoded (a reference is available).
    pub fn has_reference(&self) -> bool {
        self.last_ref.is_some()
    }

    /// True when a B-frame can be encoded (both L0 + L1 references
    /// available — the encoder needs the past anchor as L0 and the
    /// most-recent anchor as L1 under M=2 IBPBP encode order).
    pub fn has_b_references(&self) -> bool {
        self.past_anchor.is_some() && self.last_ref.is_some()
    }

    /// Borrow the (L0, L1) reference pair for B-frame ME. Returns
    /// `None` if either slot is empty (caller must check
    /// `has_b_references` or fall back to all-Skip / all-Direct).
    ///
    /// Convention: `(past_anchor, last_ref)` = `(L0, L1)`. Anchors
    /// promote in chronological order under IBPBP encode order so
    /// `past_anchor`'s POC < `last_ref`'s POC, matching the spec's
    /// L0/L1 list construction (L0 has smaller POC).
    pub fn b_references(&self) -> Option<(&ReconFrame, &ReconFrame)> {
        match (self.past_anchor.as_ref(), self.last_ref.as_ref()) {
            (Some(l0), Some(l1)) => Some((l0, l1)),
            _ => None,
        }
    }
}

// ─── Phase 6E-A1: multi-slot DPB for B-frames ────────────────────

/// Role hint attached to a DPB slot. Distinguishes the temporally-
/// earlier vs later anchor when filling reference lists for a
/// B-slice. The decoder side uses POC ordering; this hint mirrors
/// that on the encoder side without the encoder having to compute
/// full POC for every reference look-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotRole {
    /// Temporally-earlier reference relative to the slice being
    /// encoded. Becomes List 0 entry 0 for B-slices, and the sole
    /// reference for P-slices.
    Past,
    /// Temporally-later reference relative to the slice being
    /// encoded. Becomes List 1 entry 0 for B-slices. Unused for
    /// P-slices.
    Future,
}

/// One occupied slot in the multi-slot DPB. Bundles the
/// reconstructed frame, its `frame_num`, the FULL POC (so we can
/// order references for List 0 / List 1 construction), and the
/// role hint.
#[derive(Debug, Clone)]
pub struct DpbSlot {
    pub recon: ReconFrame,
    pub frame_num: u8,
    /// Full (un-wrapped) POC for this reference, mod 2^32. Used
    /// to compute List 0 / List 1 ordering relative to the
    /// current slice's POC.
    pub full_poc: u32,
    pub role: SlotRole,
}

/// Multi-slot DPB. Capacity is fixed at construction; sliding-
/// window MMCO retires the oldest slot when a new one is promoted
/// past capacity. For Phase 6E-A1 M=2 IBPBP, capacity = 3 is
/// adequate (Past P + Future P encoded ahead of the B between
/// them + IDR carry slack).
#[derive(Debug)]
pub struct MultiSlotDpb {
    slots: Vec<Option<DpbSlot>>,
}

impl MultiSlotDpb {
    /// Build a DPB with `capacity` slots, all initially empty.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "MultiSlotDpb capacity must be > 0");
        Self {
            slots: vec![None; capacity],
        }
    }

    /// Number of slots (occupied or empty).
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Number of occupied slots.
    pub fn occupied(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Clear all slots — called on every IDR.
    pub fn reset(&mut self) {
        for s in self.slots.iter_mut() {
            *s = None;
        }
    }

    /// Promote a freshly-reconstructed frame into the DPB.
    /// Sliding-window MMCO: if all slots are occupied, the slot
    /// with the OLDEST `full_poc` value is evicted to make room.
    pub fn promote(
        &mut self,
        recon: &ReconBuffer,
        frame_num: u8,
        full_poc: u32,
        role: SlotRole,
    ) {
        let new_slot = DpbSlot {
            recon: ReconFrame::snapshot(recon),
            frame_num,
            full_poc,
            role,
        };

        // First-fit: place into any empty slot.
        if let Some(empty) = self.slots.iter_mut().find(|s| s.is_none()) {
            *empty = Some(new_slot);
            return;
        }

        // All full → evict oldest (smallest full_poc).
        let oldest_idx = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|sl| (i, sl.full_poc)))
            .min_by_key(|(_, poc)| *poc)
            .map(|(i, _)| i)
            .expect("DPB is non-empty (capacity > 0)");
        self.slots[oldest_idx] = Some(new_slot);
    }

    /// Borrow the slot at `idx`. Returns `None` if the slot is
    /// empty or `idx` exceeds capacity.
    pub fn get(&self, idx: usize) -> Option<&DpbSlot> {
        self.slots.get(idx).and_then(|s| s.as_ref())
    }

    /// Find the most-recent Past slot (highest full_poc among
    /// `Past` role). For P-slices this is the L0 reference; for
    /// B-slices this is the L0 reference too.
    pub fn most_recent_past(&self) -> Option<&DpbSlot> {
        self.slots
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|s| s.role == SlotRole::Past)
            .max_by_key(|s| s.full_poc)
    }

    /// Find the soonest Future slot (smallest full_poc among
    /// `Future` role > current_poc). For B-slices this is the L1
    /// reference. Returns `None` if no Future slot is available.
    pub fn soonest_future(&self, current_poc: u32) -> Option<&DpbSlot> {
        self.slots
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|s| s.role == SlotRole::Future && s.full_poc > current_poc)
            .min_by_key(|s| s.full_poc)
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
    fn promote_shifts_last_ref_to_past_anchor() {
        // §6E-A6.1q.a (#150) — second promote shifts the prior
        // last_ref into past_anchor before overwriting.
        let mut rb = ReferenceBuffer::new();
        rb.promote(&make_recon(32, 32, 50, 128), 0);
        rb.promote(&make_recon(32, 32, 200, 100), 1);
        // last_ref carries the second promote.
        let frame = rb.last_ref.as_ref().unwrap();
        assert_eq!(frame.y_at(0, 0), 200);
        assert_eq!(frame.chroma_at(0, 0, 0), 100);
        assert_eq!(rb.last_ref_frame_num, Some(1));
        // past_anchor carries the first promote (shifted).
        let past = rb.past_anchor.as_ref().unwrap();
        assert_eq!(past.y_at(0, 0), 50);
        assert_eq!(past.chroma_at(0, 0, 0), 128);
        assert_eq!(rb.past_anchor_frame_num, Some(0));
    }

    #[test]
    fn ibpbp_anchor_sequence_makes_b_references_available() {
        // §6E-A6.1q.a (#150) — IBPBP M=2 encode-order timeline:
        //   I → P0 → B0 → P1 → B1 → ...
        // After encoding I + P0 (in encode order), B0's L0 = I and
        // L1 = P0 must both be reachable via b_references().
        let mut rb = ReferenceBuffer::new();
        // After IDR.
        rb.reset();
        assert!(!rb.has_b_references(),
            "B-references unavailable after IDR (only one anchor possible)");
        // Encode I (first anchor).
        rb.promote(&make_recon(32, 32, 10, 128), 0);
        assert!(!rb.has_b_references(),
            "B-references unavailable after only one anchor");
        // Encode P0 (second anchor under IBPBP encode order).
        rb.promote(&make_recon(32, 32, 90, 128), 1);
        assert!(rb.has_b_references(),
            "B-references available after I+P pair");
        let (l0, l1) = rb.b_references().unwrap();
        assert_eq!(l0.y_at(0, 0), 10, "L0 = past_anchor = I");
        assert_eq!(l1.y_at(0, 0), 90, "L1 = last_ref = P0");
        // Encode B0 — does NOT promote (non-reference).
        // No-op for the buffer.
        // Encode P1 (third anchor, fills DPB the same way).
        rb.promote(&make_recon(32, 32, 170, 128), 2);
        let (l0, l1) = rb.b_references().unwrap();
        assert_eq!(l0.y_at(0, 0), 90, "L0 = past_anchor = P0");
        assert_eq!(l1.y_at(0, 0), 170, "L1 = last_ref = P1");
        // IDR clears both slots.
        rb.reset();
        assert!(!rb.has_b_references());
        assert!(!rb.has_reference());
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

    // ─── Phase 6E-A1 multi-slot DPB tests ────────────────────────

    #[test]
    fn multi_slot_dpb_starts_empty() {
        let dpb = MultiSlotDpb::with_capacity(3);
        assert_eq!(dpb.capacity(), 3);
        assert_eq!(dpb.occupied(), 0);
        assert!(dpb.get(0).is_none());
        assert!(dpb.most_recent_past().is_none());
        assert!(dpb.soonest_future(0).is_none());
    }

    #[test]
    fn multi_slot_dpb_promote_fills_empty_slots() {
        let mut dpb = MultiSlotDpb::with_capacity(3);
        let recon0 = make_recon(32, 32, 50, 128);
        dpb.promote(&recon0, 0, /* poc */ 0, SlotRole::Past);
        assert_eq!(dpb.occupied(), 1);

        let recon1 = make_recon(32, 32, 100, 128);
        dpb.promote(&recon1, 1, /* poc */ 4, SlotRole::Future);
        assert_eq!(dpb.occupied(), 2);

        let past = dpb.most_recent_past().unwrap();
        assert_eq!(past.full_poc, 0);
        let future = dpb.soonest_future(0).unwrap();
        assert_eq!(future.full_poc, 4);
    }

    #[test]
    fn multi_slot_dpb_evicts_oldest_when_full() {
        let mut dpb = MultiSlotDpb::with_capacity(2);
        let r = make_recon(32, 32, 0, 128);

        dpb.promote(&r, 0, 0, SlotRole::Past);
        dpb.promote(&r, 1, 4, SlotRole::Past);
        assert_eq!(dpb.occupied(), 2);

        // Slots full; promote a 3rd → evicts poc=0.
        dpb.promote(&r, 2, 8, SlotRole::Past);
        assert_eq!(dpb.occupied(), 2);
        assert_eq!(dpb.most_recent_past().unwrap().full_poc, 8);

        // poc=0 should be gone.
        let pocs: Vec<u32> = (0..dpb.capacity())
            .filter_map(|i| dpb.get(i).map(|s| s.full_poc))
            .collect();
        assert!(!pocs.contains(&0));
        assert!(pocs.contains(&4));
        assert!(pocs.contains(&8));
    }

    #[test]
    fn multi_slot_dpb_reset_clears() {
        let mut dpb = MultiSlotDpb::with_capacity(3);
        let r = make_recon(32, 32, 100, 128);
        dpb.promote(&r, 0, 0, SlotRole::Past);
        dpb.promote(&r, 1, 4, SlotRole::Future);
        assert_eq!(dpb.occupied(), 2);

        dpb.reset();
        assert_eq!(dpb.occupied(), 0);
        assert!(dpb.most_recent_past().is_none());
    }

    #[test]
    fn multi_slot_dpb_filters_role_correctly() {
        let mut dpb = MultiSlotDpb::with_capacity(3);
        let r = make_recon(32, 32, 0, 128);

        dpb.promote(&r, 0, 0, SlotRole::Past);
        dpb.promote(&r, 1, 4, SlotRole::Future);
        dpb.promote(&r, 2, 2, SlotRole::Past);

        // most_recent_past picks highest-poc Past slot only
        // (ignores Future).
        assert_eq!(dpb.most_recent_past().unwrap().full_poc, 2);

        // soonest_future at current_poc=2 picks the Future slot
        // with smallest poc > 2.
        assert_eq!(dpb.soonest_future(2).unwrap().full_poc, 4);

        // soonest_future at current_poc=4 returns None (no future
        // slot beyond 4).
        assert!(dpb.soonest_future(4).is_none());
    }

    #[test]
    fn multi_slot_dpb_capacity_one_acts_like_single_slot() {
        // Sanity: capacity=1 behaves like the legacy ReferenceBuffer.
        let mut dpb = MultiSlotDpb::with_capacity(1);
        let r0 = make_recon(32, 32, 50, 128);
        let r1 = make_recon(32, 32, 200, 100);
        dpb.promote(&r0, 0, 0, SlotRole::Past);
        assert_eq!(dpb.occupied(), 1);
        assert_eq!(dpb.get(0).unwrap().recon.y_at(0, 0), 50);

        // Second promote evicts the first.
        dpb.promote(&r1, 1, 2, SlotRole::Past);
        assert_eq!(dpb.occupied(), 1);
        assert_eq!(dpb.get(0).unwrap().recon.y_at(0, 0), 200);
    }
}
