// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Decode-side position recorder for paired stego decode (Phase
// 6D.8 chunk 6A).
//
// Symmetric to the encoder-side `PositionLoggerHook`: as the slice
// walker (chunk 6B+) decodes each macroblock, it pipes the already-
// decoded coefficients + MVD slots into this recorder, which
// enumerates per-domain positions + bit values in the same order
// the encoder emitted them. The accumulated `DomainCover` then
// feeds STC extract (per-domain) → frame parse → decrypt → message
// recovery.
//
// Costs are intentionally NOT tracked here. Pass 2 (STC plan) ran at
// encode time; the decoder only needs the cover bits + positions to
// reverse the embed. Skipping cost vectors keeps the decoder light.
//
// **Parity with `PositionLoggerHook`**: by-construction. Both call
// the same `enumerate_*_positions` + `extract_*_bits` primitives
// from `crate::codec::h264::stego` on identical (scan_coeffs,
// MvdSlot) inputs in identical (frame_idx, mb_addr, path_kind)
// contexts. By the encode-time stego invariant (post-quantize hook
// fires before entropy emit; decoder dequantize produces the same
// scan_coeffs the encoder entropy-coded), inputs match.

use crate::codec::h264::stego::{
    record_residual_block_into_cover,
    enumerate_coeff_sign_positions, enumerate_coeff_suffix_lsb_positions,
    enumerate_mvd_sign_positions, enumerate_mvd_suffix_lsb_positions,
    extract_coeff_sign_bits, extract_coeff_suffix_lsb_bits,
    extract_mvd_sign_bits, extract_mvd_suffix_lsb_bits,
    Axis, BinKind, DomainCover, MvdSlot, ResidualPathKind,
};
use crate::codec::h264::stego::encoder_hook::MvdPositionMeta;
use crate::codec::h264::stego::hook::{
    EmbedDomain, GopCapacity, PositionKey, PositionLogger,
};

/// Phase C.3.6.1 (task #428) — RBSP bit offset + NAL index of a
/// single bypass-coded bin. Captured by the walker before the bin is
/// read, so a downstream caller can re-locate the exact byte+bit in
/// the original (post-emulation-prevention-strip) RBSP. The Option C
/// bitstream-mod stego splicer uses this to flip selected cover bits
/// directly in the encoded Annex-B stream without re-encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PositionOffset {
    /// RBSP bit position of the bypass-coded bin, measured from the
    /// start of the NAL's RBSP (i.e. after the leading NAL header
    /// byte is consumed and emulation-prevention triplets `0x000003`
    /// have been stripped to `0x0000`).
    pub rbsp_bit: u64,
    /// Zero-based index into the input NALU sequence identifying
    /// which slice NAL contains this bypass bin.
    pub nal_idx: u32,
}

/// Phase C.3.6.1 — four parallel vectors of `PositionOffset`, one per
/// stego domain, populated 1:1 with each domain's cover-bit vector in
/// `DomainCover` when `WalkOptions::record_offsets` is set.
///
/// **Index alignment**: `offsets.coeff_sign_bypass[i]` is the byte+bit
/// offset of `cover.coeff_sign_bypass.bits[i]`. The walker guarantees
/// this by capturing offsets at the same syntax sites that the
/// encoder's `PositionLoggerHook` + decoder's `enumerate_*_positions`
/// derive cover bits from, in the same scan order.
#[derive(Default, Debug, Clone)]
pub struct DomainOffsets {
    pub coeff_sign_bypass: Vec<PositionOffset>,
    pub coeff_suffix_lsb: Vec<PositionOffset>,
    pub mvd_sign_bypass: Vec<PositionOffset>,
    pub mvd_suffix_lsb: Vec<PositionOffset>,
}

impl DomainOffsets {
    pub fn total_len(&self) -> usize {
        self.coeff_sign_bypass.len()
            + self.coeff_suffix_lsb.len()
            + self.mvd_sign_bypass.len()
            + self.mvd_suffix_lsb.len()
    }
}

/// Phase C.3.6.1 — `PositionLogger` impl that captures
/// `(rbsp_bit_offset, nal_idx)` per domain into a `DomainOffsets`.
/// Plugged into the 5 inner `PositionCtx` construction sites in the
/// walker when `WalkOptions::record_offsets` is set; otherwise those
/// sites use `NullLogger` and pay no per-bin cost.
///
/// Capacity tracking is not relevant here — the recorder's per-domain
/// `cover` vectors already track length. `capacity()` returns the
/// running per-domain offset count for diagnostic parity.
#[derive(Default, Debug)]
pub struct OffsetCapturingLogger {
    pub offsets: DomainOffsets,
}

impl OffsetCapturingLogger {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PositionLogger for OffsetCapturingLogger {
    #[inline]
    fn register(&mut self, _key: PositionKey) -> bool {
        // The walker-side recorder derives positions from enumerate_*
        // post-decode, so the trait's register() is a no-op here — the
        // log of cover bits is on `PositionRecorder`, not on us.
        true
    }

    #[inline]
    fn capacity(&self) -> GopCapacity {
        GopCapacity {
            coeff_sign_bypass: self.offsets.coeff_sign_bypass.len(),
            coeff_suffix_lsb: self.offsets.coeff_suffix_lsb.len(),
            mvd_sign_bypass: self.offsets.mvd_sign_bypass.len(),
            mvd_suffix_lsb: self.offsets.mvd_suffix_lsb.len(),
        }
    }

    #[inline]
    fn register_with_offset(
        &mut self,
        key: PositionKey,
        rbsp_bit_offset: u64,
        nal_idx: u32,
    ) -> bool {
        let off = PositionOffset { rbsp_bit: rbsp_bit_offset, nal_idx };
        match key.domain() {
            EmbedDomain::CoeffSignBypass => self.offsets.coeff_sign_bypass.push(off),
            EmbedDomain::CoeffSuffixLsb => self.offsets.coeff_suffix_lsb.push(off),
            EmbedDomain::MvdSignBypass => self.offsets.mvd_sign_bypass.push(off),
            EmbedDomain::MvdSuffixLsb => self.offsets.mvd_suffix_lsb.push(off),
        }
        true
    }
}

/// Decode-side recorder. Mirrors `stego::PositionLoggerHook`
/// but takes immutable inputs (decoded coefficients aren't mutated)
/// and skips cost tracking (decoder doesn't replan).
///
/// **Usage**: the slice walker calls `on_residual_block` after each
/// CABAC `decode_residual_block_*` returns scan_coeffs, and
/// `on_mvd_slot` after each `decode_mvd_with_bin0_inc` returns an
/// MVD value. After walking the entire GOP, `into_cover()` returns
/// the accumulated `DomainCover` ready for STC extract.
#[derive(Debug, Default)]
pub struct PositionRecorder {
    cover: DomainCover,
    /// Phase 6F.2(j) — per-MVD-position metadata aligned by index
    /// with `cover.mvd_sign_bypass.positions`. Mirrors the encoder's
    /// `PositionLoggerHook::mvd_meta`. Drained via `take_mvd_meta()`
    /// for the cascade-safety analysis at decode time.
    mvd_meta: Vec<MvdPositionMeta>,
    /// Phase C.3.6.1 (task #428) — opt-in capture of RBSP bit
    /// offsets, one entry per cover bit per domain. `None` keeps the
    /// walker on the zero-cost NullLogger fast path; `Some(_)`
    /// activates per-bin offset push at the 5 inner production sites
    /// in slice.rs. Toggled by `WalkOptions::record_offsets` in the
    /// top-level walker entry points.
    pub offset_logger: Option<OffsetCapturingLogger>,
}

impl PositionRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Phase C.3.6.1 — construct a recorder that captures RBSP bit
    /// offsets in addition to cover bits. Equivalent to
    /// `let mut r = PositionRecorder::new(); r.enable_offset_capture();`.
    pub fn with_offsets() -> Self {
        Self {
            offset_logger: Some(OffsetCapturingLogger::new()),
            ..Self::default()
        }
    }

    /// Phase C.3.6.1 — opt-in to offset capture mid-stream. Idempotent
    /// when capture is already enabled.
    pub fn enable_offset_capture(&mut self) {
        if self.offset_logger.is_none() {
            self.offset_logger = Some(OffsetCapturingLogger::new());
        }
    }

    /// #516.1 perf — pre-size cover Vecs from MB count.
    ///
    /// Called by the walker after SPS parse, once `mb_count = mb_w *
    /// mb_h` is known for the slice. Per-domain heuristics based on
    /// the 2026-05-17 1080p × 30f IPPPP IMG_4138 baseline (`memory/
    /// h264_perf_baseline_2026_05_17.md`):
    ///
    /// - CoeffSign:       ~2.13 positions per MB on real-world IPPPP
    ///                    @ QP=26 → pre-size to `2 * mb_count` for
    ///                    ~0-1 reallocs even on dense content.
    /// - CoeffSuffixLsb:  sparse (~0.4/MB) → pre-size to mb_count / 2.
    /// - MvdSignBypass:   only fires with `record_mvd=true`; rate
    ///                    depends on slice type. Pre-size to mb_count
    ///                    / 4 — overshoots on intra slices, undershoots
    ///                    moderately on high-motion P/B (1 realloc).
    /// - MvdSuffixLsb:    very sparse (~0.02/MB on real-world motion)
    ///                    → pre-size to mb_count / 8.
    ///
    /// Idempotent: calling multiple times only widens capacity, never
    /// shrinks. Safe to call before each GOP if the walker wants to
    /// reset per-GOP caps after `take_cover()`.
    ///
    /// Total upfront cap at 1080p (241k MBs): ~482k + 121k + 60k + 30k
    /// = 693k entries × ~25 bytes avg = ~17 MB. At 60-frame GOP
    /// long-form (482k MBs): ~34 MB worst case. Stays well under the
    /// per-GOP memory bound from `streaming-Viterbi` (#472).
    pub fn reserve_for_mb_count(&mut self, mb_count: usize) {
        self.cover.coeff_sign_bypass.reserve(mb_count * 2);
        self.cover.coeff_suffix_lsb.reserve(mb_count / 2);
        self.cover.mvd_sign_bypass.reserve(mb_count / 4);
        self.cover.mvd_suffix_lsb.reserve(mb_count / 8);
    }

    /// Consume the recorder and return the accumulated cover.
    pub fn into_cover(self) -> DomainCover {
        self.cover
    }

    /// Swap out the accumulated cover, leaving the recorder empty
    /// and ready to record into a fresh `DomainCover`. Used by the
    /// streaming walker to emit one cover per GOP without
    /// re-allocating the recorder.
    pub fn take_cover(&mut self) -> DomainCover {
        std::mem::take(&mut self.cover)
    }

    /// Phase 6F.2(j) — drain the per-MVD-position metadata captured
    /// alongside the cover. Aligned by index with
    /// `take_cover().mvd_sign_bypass.positions`.
    pub fn take_mvd_meta(&mut self) -> Vec<MvdPositionMeta> {
        std::mem::take(&mut self.mvd_meta)
    }

    /// Phase C.3.6.1 — drain captured offsets. Returns `None` if
    /// offset capture was never enabled on this recorder; returns
    /// `Some(empty)` if enabled but no bypass bins were emitted.
    pub fn take_offsets(&mut self) -> Option<DomainOffsets> {
        self.offset_logger.as_mut().map(|ol| std::mem::take(&mut ol.offsets))
    }

    /// Record a residual block's bypass-bin positions + bits across
    /// the CoeffSignBypass + CoeffSuffixLsb domains.
    ///
    /// `scan_coeffs[start_idx..=end_idx]` is the dequantized scan
    /// (zigzag/field) order coefficient list as decoded by
    /// `decode_residual_block_cabac` or `_8x8`. Inputs MUST equal
    /// the values the encoder entropy-coded for parity to hold.
    pub fn on_residual_block(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        scan_coeffs: &[i32],
        start_idx: usize,
        end_idx: usize,
        path_kind: ResidualPathKind,
    ) {
        // #516.1.b perf — fused single-pass walker that pushes
        // CoeffSign + CoeffSuffixLsb (bit, position) directly into
        // the cover Vecs. The legacy implementation built 4
        // intermediate Vecs per call (sig indices, positions, sig
        // indices again, bits) — at ~16 residual blocks per MB ×
        // 241k MBs (1080p × 30f) = ~3.86M small alloc/free pairs
        // per cover walk. The fused walker eliminates them; output
        // is byte-identical (verified by the lib parity tests at
        // bottom of this file + 98 walker tests).
        record_residual_block_into_cover(
            &mut self.cover,
            scan_coeffs,
            start_idx,
            end_idx,
            frame_idx,
            mb_addr,
            |ci, kind| path_kind.path(ci, kind),
        );
    }

    /// Record an MVD slot's bypass-bin positions + bits across the
    /// MvdSignBypass + MvdSuffixLsb domains.
    pub fn on_mvd_slot(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        slot: &MvdSlot,
    ) {
        let single = [*slot];

        // MvdSignBypass.
        let positions = enumerate_mvd_sign_positions(&single, frame_idx, mb_addr);
        let bits = extract_mvd_sign_bits(&single);
        let pre_sign_len = self.cover.mvd_sign_bypass.len();
        for (p, b) in positions.iter().zip(bits.iter()) {
            self.cover.mvd_sign_bypass.push(*b, *p);
        }
        // Phase 6F.2(j) — capture per-position metadata aligned with
        // mvd_sign_bypass. `enumerate_mvd_sign_positions` filters
        // zero-valued slots, matching the encoder's PositionLoggerHook.
        let pushed = self.cover.mvd_sign_bypass.len() - pre_sign_len;
        if pushed > 0 {
            self.mvd_meta.push(MvdPositionMeta {
                magnitude: slot.value.unsigned_abs(),
                mb_addr,
                frame_idx,
                partition: slot.partition,
                axis: match slot.axis { Axis::X => 0, Axis::Y => 1 },
            });
        }

        // MvdSuffixLsb.
        let positions = enumerate_mvd_suffix_lsb_positions(&single, frame_idx, mb_addr);
        let bits = extract_mvd_suffix_lsb_bits(&single);
        for (p, b) in positions.iter().zip(bits.iter()) {
            self.cover.mvd_suffix_lsb.push(*b, *p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::stego::Axis;
    use crate::codec::h264::stego::encoder_hook::{PositionLoggerHook, StegoMbHook};

    #[test]
    fn recorder_residual_block_collects_per_domain() {
        let mut rec = PositionRecorder::new();
        let scan: Vec<i32> = {
            let mut v = vec![0i32; 16];
            v[0] = 5; v[3] = -7; v[6] = 20;
            v
        };
        rec.on_residual_block(
            0, 0, &scan, 0, 15,
            ResidualPathKind::Luma4x4 { block_idx: 0 },
        );
        let cover = rec.into_cover();
        assert_eq!(cover.coeff_sign_bypass.len(), 3);
        assert_eq!(cover.coeff_suffix_lsb.len(), 1);
    }

    #[test]
    fn recorder_mvd_slot_collects_per_domain() {
        let mut rec = PositionRecorder::new();
        let slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 15 };
        rec.on_mvd_slot(0, 0, &slot);
        let cover = rec.into_cover();
        assert_eq!(cover.mvd_sign_bypass.len(), 1);
        assert_eq!(cover.mvd_suffix_lsb.len(), 1);
    }

    #[test]
    fn recorder_zero_inputs_emit_no_positions() {
        let mut rec = PositionRecorder::new();
        let scan = vec![0i32; 16];
        rec.on_residual_block(
            0, 0, &scan, 0, 15,
            ResidualPathKind::Luma4x4 { block_idx: 0 },
        );
        let slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 0 };
        rec.on_mvd_slot(0, 0, &slot);
        let cover = rec.into_cover();
        assert_eq!(cover.coeff_sign_bypass.len(), 0);
        assert_eq!(cover.coeff_suffix_lsb.len(), 0);
        assert_eq!(cover.mvd_sign_bypass.len(), 0);
        assert_eq!(cover.mvd_suffix_lsb.len(), 0);
    }

    /// Parity gate vs `PositionLoggerHook` (encode side). Same
    /// inputs ⇒ same per-domain positions + bits. This is the
    /// load-bearing invariant for chunk 6: STC extract on the
    /// decode-side cover must operate on cover bits identical to
    /// what STC embed saw at encode time.
    #[test]
    fn recorder_matches_encoder_position_logger_hook() {
        let mut scan: Vec<i32> = (0..16).map(|i| ((i * 13) as i32) - 50).collect();
        let logger_scan_input = scan.clone();
        let path = ResidualPathKind::Luma4x4 { block_idx: 3 };

        // Encode-side: PositionLoggerHook reads (and does not
        // mutate) scan_coeffs.
        let mut logger = PositionLoggerHook::new();
        logger.on_residual_block(
            7, 42, &mut scan, 0, 15, path,
        );
        let logger_cover = logger.take_cover();

        // PositionLoggerHook is non-mutating — confirm and use the
        // pristine input for the recorder.
        assert_eq!(scan, logger_scan_input,
            "PositionLoggerHook must not mutate scan");

        let mut rec = PositionRecorder::new();
        rec.on_residual_block(
            7, 42, &scan, 0, 15, path,
        );
        let rec_cover = rec.into_cover();

        // Same positions, same bits in the same order.
        assert_eq!(
            rec_cover.coeff_sign_bypass.positions,
            logger_cover.cover.coeff_sign_bypass.positions,
            "coeff_sign_bypass positions must match"
        );
        assert_eq!(
            rec_cover.coeff_sign_bypass.bits,
            logger_cover.cover.coeff_sign_bypass.bits,
            "coeff_sign_bypass bits must match"
        );
        assert_eq!(
            rec_cover.coeff_suffix_lsb.positions,
            logger_cover.cover.coeff_suffix_lsb.positions,
            "coeff_suffix_lsb positions must match"
        );
        assert_eq!(
            rec_cover.coeff_suffix_lsb.bits,
            logger_cover.cover.coeff_suffix_lsb.bits,
            "coeff_suffix_lsb bits must match"
        );
    }

    #[test]
    fn recorder_mvd_matches_encoder_position_logger_hook() {
        let mut slot = MvdSlot { list: 0, partition: 1, axis: Axis::Y, value: -23 };
        let mut logger = PositionLoggerHook::new();
        logger.on_mvd_slot(11, 99, &mut slot);
        let logger_cover = logger.take_cover();

        let mut rec = PositionRecorder::new();
        rec.on_mvd_slot(11, 99, &slot);
        let rec_cover = rec.into_cover();

        assert_eq!(
            rec_cover.mvd_sign_bypass.positions,
            logger_cover.cover.mvd_sign_bypass.positions,
        );
        assert_eq!(
            rec_cover.mvd_sign_bypass.bits,
            logger_cover.cover.mvd_sign_bypass.bits,
        );
        assert_eq!(
            rec_cover.mvd_suffix_lsb.positions,
            logger_cover.cover.mvd_suffix_lsb.positions,
        );
        assert_eq!(
            rec_cover.mvd_suffix_lsb.bits,
            logger_cover.cover.mvd_suffix_lsb.bits,
        );
    }
}
