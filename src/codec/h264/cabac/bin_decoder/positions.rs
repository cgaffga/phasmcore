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
    enumerate_coeff_sign_positions, enumerate_coeff_suffix_lsb_positions,
    enumerate_mvd_sign_positions, enumerate_mvd_suffix_lsb_positions,
    extract_coeff_sign_bits, extract_coeff_suffix_lsb_bits,
    extract_mvd_sign_bits, extract_mvd_suffix_lsb_bits,
    Axis, BinKind, DomainCover, MvdSlot, ResidualPathKind,
};
use crate::codec::h264::stego::encoder_hook::MvdPositionMeta;

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
}

impl PositionRecorder {
    pub fn new() -> Self {
        Self::default()
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
        // CoeffSignBypass.
        let positions = enumerate_coeff_sign_positions(
            scan_coeffs, start_idx, end_idx, frame_idx, mb_addr,
            |ci| path_kind.path(ci, BinKind::Sign),
        );
        let bits = extract_coeff_sign_bits(scan_coeffs, start_idx, end_idx);
        for (p, b) in positions.iter().zip(bits.iter()) {
            self.cover.coeff_sign_bypass.push(*b, *p);
        }

        // CoeffSuffixLsb.
        let positions = enumerate_coeff_suffix_lsb_positions(
            scan_coeffs, start_idx, end_idx, frame_idx, mb_addr,
            |ci| path_kind.path(ci, BinKind::SuffixLsb),
        );
        let bits = extract_coeff_suffix_lsb_bits(scan_coeffs, start_idx, end_idx);
        for (p, b) in positions.iter().zip(bits.iter()) {
            self.cover.coeff_suffix_lsb.push(*b, *p);
        }
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
