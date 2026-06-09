// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Decode-side position recorder for paired stego decode.
//
// Symmetric to the encode-side cover capture: as the slice walker
// decodes each macroblock, it pipes the already-decoded
// coefficients + MVD slots into this recorder, which enumerates
// per-domain positions + bit values in the same order the OH264
// encoder's wire_only emit produced them. The accumulated
// `DomainCover` then feeds STC extract (per-domain) → frame parse →
// decrypt → message recovery.
//
// Costs are intentionally NOT tracked here. Pass 2 (STC plan) ran at
// encode time; the decoder only needs the cover bits + positions to
// reverse the embed. Skipping cost vectors keeps the decoder light.
//
// **Parity with the encode-side cover capture**: by-construction.
// Both sides call the same `enumerate_*_positions` + `extract_*_bits`
// primitives from `crate::codec::h264::stego` (the shared
// `stego::inject` helpers) on identical (scan_coeffs, MvdSlot) inputs
// in identical (frame_idx, mb_addr, path_kind) contexts. By the
// encode-time stego invariant (the OH264 wire_only override is keyed
// off the post-quantize coefficients before entropy emit; decoder
// dequantize produces the same scan_coeffs the encoder entropy-
// coded), inputs match.

use crate::codec::h264::stego::{
    record_residual_block_into_cover,
    enumerate_mvd_sign_positions, enumerate_mvd_suffix_lsb_positions,
    extract_mvd_sign_bits, extract_mvd_suffix_lsb_bits,
    Axis, BinKind, DomainCover, MvdSlot, ResidualPathKind,
};
use crate::codec::h264::stego::hook::MvdPositionMeta;

/// Decode-side recorder. Mirrors the encode-side cover capture
/// (the shared `stego::inject` enumerate/extract primitives) but
/// takes immutable inputs (decoded coefficients aren't mutated)
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
    /// Per-MVD-position metadata aligned by index with
    /// `cover.mvd_sign_bypass.positions`. Mirrors the per-MVD
    /// metadata the encode-side cover capture records. Drained via
    /// `take_mvd_meta()` for the cascade-safety analysis at decode
    /// time.
    mvd_meta: Vec<MvdPositionMeta>,
}

impl PositionRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-size cover Vecs from MB count to cut reallocations.
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
    /// per-GOP streaming memory bound.
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

    /// Drain the per-MVD-position metadata captured alongside the
    /// cover. Aligned by index with
    /// `take_cover().mvd_sign_bypass.positions`.
    pub fn take_mvd_meta(&mut self) -> Vec<MvdPositionMeta> {
        std::mem::take(&mut self.mvd_meta)
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
        // Fused single-pass walker that pushes CoeffSign +
        // CoeffSuffixLsb (bit, position) directly into the cover
        // Vecs. The legacy implementation built 4
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
        // Capture per-position metadata aligned with
        // mvd_sign_bypass. `enumerate_mvd_sign_positions` filters
        // zero-valued slots, matching the shared encode-side
        // enumerate primitive.
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

}
