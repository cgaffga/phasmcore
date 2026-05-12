// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! §6E-A6.1q.b (#151 part 3) — B-frame inter prediction helpers.
//!
//! Builds the 16x16 luma + 8x8 chroma prediction for a B-MB given
//! the chosen non-direct mode (L0_16x16 / L1_16x16 / Bi_16x16) and
//! the L0 + L1 reference frames. Single-list paths reuse the P-side
//! `apply_luma_mv_block` / `apply_chroma_mv_block` helpers; bipred
//! paths call the `_bipred` variants which compute spec § 8.4.2.3.1
//! `(L0 + L1 + 1) >> 1` averaging.
//!
//! Used by the encoder loop (post-§6E-A6.1q.a DPB extension + #151
//! parts 1+2 ME wiring) when the chosen mode is non-direct + the
//! caller wants to compute residual + emit non-zero CBP. CBP=0 modes
//! (Skip / Direct / forced zero-MV uniform-content L0/L1/Bi) bypass
//! this and use the existing `emit_b_l0_16x16` / `_l1_16x16` /
//! `_bi_16x16` syntax-only helpers in `encoder.rs`.
//!
//! ## Wire-up status (2026-05-02)
//!
//! Prediction helpers (this module) shipped. Next commit pairs:
//!
//! 1. Encoder loop: per non-direct-bucket B-MB, compute pred via
//!    `build_b_luma_prediction` + `build_b_chroma_prediction`,
//!    forward DCT + quant per 4x4 block (luma) + 2x2 DC chroma,
//!    pack CBP from non-zero levels.
//! 2. Refactor `emit_b_l0_16x16` / `_l1_16x16` / `_bi_16x16` to
//!    drop the hardcoded `cbp_value=0` and emit residual blocks
//!    via P-side `encode_residual_block_cabac_with_cbf_inc` after
//!    the MVDs.
//! 3. Reconstruct (inverse quant + inverse DCT + add to pred) into
//!    `self.recon` so downstream MBs + deblock filter see consistent
//!    state.
//! 4. Update neighbour state: `total_coeff_grid`,
//!    `chroma_cb_tc_grid`, `chroma_cr_tc_grid`, `qp_grid`,
//!    `intra_grid` (false), `transform_8x8_grid` (false).
//! 5. Walker un-rejection (#152): lift `cbp_byte != 0 → Unsupported`
//!    in `walk_b_l0_16x16` / `walk_b_l1_16x16` / `walk_b_bi_16x16`.
//!    Wire P-side `decode_residual_block_cabac` into B-walker tails.
//!
//! Estimated 300-500 LOC for the full wiring; partitioned (mb_types
//! 4..21) + B_8x8 (mb_type 22) stay at CBP=0 in this scope (their
//! residual extension is a follow-on, not a v1.0 shipping requirement
//! per the §6E-A6.5 distribution-match analysis).

use super::motion_compensation::{
    apply_chroma_mv_block, apply_chroma_mv_block_bipred, apply_luma_mv_block,
    apply_luma_mv_block_bipred,
};
use super::motion_estimation::MotionVector;
use super::reference_buffer::ReconFrame;

/// §6E-A6.1q.b — chosen non-direct B-MB mode for the 16x16 family.
/// Mirrors the `BMbDecision` variants for L0_16x16 / L1_16x16 /
/// Bi_16x16 but strips the dispatch surface — this enum is the input
/// to the prediction + residual helpers and carries only what they
/// need (chosen MVs).
// Variant names mirror H.264 spec Table 7-14 (`B_L0_16x16`,
// `B_L1_16x16`, `B_Bi_16x16`); the spec uses CamelCase prefixes +
// underscored size suffixes which clippy reads as non-camel-case.
// Ship-locked to spec naming so docs/tests match the syntax tree.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
pub enum BInterMode {
    /// `mb_type = 1` — single L0 reference.
    L0_16x16 { mv: MotionVector },
    /// `mb_type = 2` — single L1 reference.
    L1_16x16 { mv: MotionVector },
    /// `mb_type = 3` — bipred over both lists. Spec § 8.4.2.3.1
    /// averaging at MC time.
    Bi_16x16 { mv_l0: MotionVector, mv_l1: MotionVector },
}

/// §6E-A6.1q.b — build the 16x16 luma prediction for a B-MB at MB
/// position `(mb_x, mb_y)`. Mirrors `encoder::build_luma_prediction`'s
/// shape but dispatches on [`BInterMode`] instead of `PMbChoice`.
pub fn build_b_luma_prediction(
    mode: BInterMode,
    l0_ref: &ReconFrame,
    l1_ref: &ReconFrame,
    mb_x: usize,
    mb_y: usize,
) -> [[u8; 16]; 16] {
    let mut out = [[0u8; 16]; 16];
    let mb_px_x = (mb_x * 16) as u32;
    let mb_px_y = (mb_y * 16) as u32;
    let flat = out.as_flattened_mut();
    match mode {
        BInterMode::L0_16x16 { mv } => {
            apply_luma_mv_block(l0_ref, mb_px_x, mb_px_y, 16, 16, mv, flat, 16);
        }
        BInterMode::L1_16x16 { mv } => {
            apply_luma_mv_block(l1_ref, mb_px_x, mb_px_y, 16, 16, mv, flat, 16);
        }
        BInterMode::Bi_16x16 { mv_l0, mv_l1 } => {
            apply_luma_mv_block_bipred(
                l0_ref, mv_l0, l1_ref, mv_l1,
                mb_px_x, mb_px_y, 16, 16, flat, 16,
            );
        }
    }
    out
}

/// §6E-A6.1q.b — build the 8x8 chroma prediction for a B-MB at MB
/// position `(mb_x, mb_y)` for the given component (0 = Cb, 1 = Cr).
/// Mirrors `encoder::build_chroma_prediction`'s shape but dispatches
/// on [`BInterMode`].
pub fn build_b_chroma_prediction(
    mode: BInterMode,
    l0_ref: &ReconFrame,
    l1_ref: &ReconFrame,
    component: u8,
    mb_x: usize,
    mb_y: usize,
) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    let mb_chroma_px_x = (mb_x * 8) as u32;
    let mb_chroma_px_y = (mb_y * 8) as u32;
    let flat = out.as_flattened_mut();
    match mode {
        BInterMode::L0_16x16 { mv } => {
            apply_chroma_mv_block(
                l0_ref, component, mb_chroma_px_x, mb_chroma_px_y,
                8, 8, mv, flat, 8,
            );
        }
        BInterMode::L1_16x16 { mv } => {
            apply_chroma_mv_block(
                l1_ref, component, mb_chroma_px_x, mb_chroma_px_y,
                8, 8, mv, flat, 8,
            );
        }
        BInterMode::Bi_16x16 { mv_l0, mv_l1 } => {
            apply_chroma_mv_block_bipred(
                l0_ref, mv_l0, l1_ref, mv_l1, component,
                mb_chroma_px_x, mb_chroma_px_y, 8, 8, flat, 8,
            );
        }
    }
    out
}

/// Phase 2.12.d (#280, 2026-05-08) — per-8×8 luma prediction builder
/// for B_Skip / B_Direct_16x16 with per-sub-block MVs.
///
/// Spec § 8.4.1.2.2 step 6 applies the colZeroFlag override PER 8×8
/// sub-block. Some sub-blocks get MV=(0,0), others keep the median
/// predictor. The encoder must apply MC per-sub-block to match the
/// decoder's reconstruction.
///
/// `mv_l0_per_8x8`/`mv_l1_per_8x8` indexed in raster order
/// (TL=0, TR=1, BL=2, BR=3). Each 8×8 luma sub-block covers 8×8
/// pixels at offsets (0,0), (8,0), (0,8), (8,8).
///
/// `uses_l0`/`uses_l1` indicate which lists are active for this MB.
/// At least one MUST be active (caller responsibility — boundary
/// case where both are inactive should never reach this function).
pub fn build_b_luma_prediction_per_8x8(
    mv_l0_per_8x8: [MotionVector; 4],
    mv_l1_per_8x8: [MotionVector; 4],
    uses_l0: bool,
    uses_l1: bool,
    l0_ref: &ReconFrame,
    l1_ref: &ReconFrame,
    mb_x: usize,
    mb_y: usize,
) -> [[u8; 16]; 16] {
    let mut out = [[0u8; 16]; 16];
    let mb_px_x = (mb_x * 16) as u32;
    let mb_px_y = (mb_y * 16) as u32;
    for i8 in 0..4 {
        let off_x = ((i8 & 1) * 8) as u32;
        let off_y = ((i8 >> 1) * 8) as u32;
        let sub_px_x = mb_px_x + off_x;
        let sub_px_y = mb_px_y + off_y;
        // Slice into the 16×16 output buffer at the sub-block's
        // top-left, with stride 16 so the 8×8 fill writes to the
        // correct rows.
        let flat = out.as_flattened_mut();
        let start = (off_y as usize) * 16 + (off_x as usize);
        let sub_slice = &mut flat[start..];
        match (uses_l0, uses_l1) {
            (true, false) => {
                apply_luma_mv_block(
                    l0_ref, sub_px_x, sub_px_y, 8, 8,
                    mv_l0_per_8x8[i8], sub_slice, 16,
                );
            }
            (false, true) => {
                apply_luma_mv_block(
                    l1_ref, sub_px_x, sub_px_y, 8, 8,
                    mv_l1_per_8x8[i8], sub_slice, 16,
                );
            }
            (true, true) => {
                apply_luma_mv_block_bipred(
                    l0_ref, mv_l0_per_8x8[i8],
                    l1_ref, mv_l1_per_8x8[i8],
                    sub_px_x, sub_px_y, 8, 8, sub_slice, 16,
                );
            }
            (false, false) => {
                // Should never happen — boundary case forces both refs
                // to 0 in the spatial-direct derivation.
                debug_assert!(false, "per_8x8 prediction with both lists inactive");
            }
        }
    }
    out
}

/// Phase 2.12.d (#280) — per-8×8 chroma prediction builder for
/// B_Skip / B_Direct_16x16. Each luma 8×8 sub-block corresponds to
/// a chroma 4×4 sub-block (4:2:0 subsampling). Sub-block raster:
/// TL chroma at (0,0), TR at (4,0), BL at (0,4), BR at (4,4) in
/// the 8×8 chroma output buffer.
pub fn build_b_chroma_prediction_per_8x8(
    mv_l0_per_8x8: [MotionVector; 4],
    mv_l1_per_8x8: [MotionVector; 4],
    uses_l0: bool,
    uses_l1: bool,
    l0_ref: &ReconFrame,
    l1_ref: &ReconFrame,
    component: u8,
    mb_x: usize,
    mb_y: usize,
) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    let mb_chroma_px_x = (mb_x * 8) as u32;
    let mb_chroma_px_y = (mb_y * 8) as u32;
    for i8 in 0..4 {
        // Each luma 8×8 sub-block = chroma 4×4 sub-block in 4:2:0.
        let off_x_c = ((i8 & 1) * 4) as u32;
        let off_y_c = ((i8 >> 1) * 4) as u32;
        let sub_cpx_x = mb_chroma_px_x + off_x_c;
        let sub_cpx_y = mb_chroma_px_y + off_y_c;
        let flat = out.as_flattened_mut();
        let start = (off_y_c as usize) * 8 + (off_x_c as usize);
        let sub_slice = &mut flat[start..];
        match (uses_l0, uses_l1) {
            (true, false) => {
                apply_chroma_mv_block(
                    l0_ref, component, sub_cpx_x, sub_cpx_y, 4, 4,
                    mv_l0_per_8x8[i8], sub_slice, 8,
                );
            }
            (false, true) => {
                apply_chroma_mv_block(
                    l1_ref, component, sub_cpx_x, sub_cpx_y, 4, 4,
                    mv_l1_per_8x8[i8], sub_slice, 8,
                );
            }
            (true, true) => {
                apply_chroma_mv_block_bipred(
                    l0_ref, mv_l0_per_8x8[i8],
                    l1_ref, mv_l1_per_8x8[i8],
                    component, sub_cpx_x, sub_cpx_y, 4, 4, sub_slice, 8,
                );
            }
            (false, false) => {
                debug_assert!(false, "per_8x8 chroma prediction with both lists inactive");
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::encoder::reconstruction::ReconBuffer;

    fn make_recon(width: u32, height: u32, y_fill: u8, c_fill: u8) -> ReconFrame {
        let mut buf = ReconBuffer::new(width, height).unwrap();
        for v in buf.y.iter_mut() { *v = y_fill; }
        for v in buf.cb.iter_mut() { *v = c_fill; }
        for v in buf.cr.iter_mut() { *v = c_fill; }
        ReconFrame::snapshot(&buf)
    }

    #[test]
    fn b_l0_prediction_zero_mv_matches_l0_ref() {
        // L0=100, L1=200, mode=L0 with mv=(0,0): pred should be all 100.
        let l0 = make_recon(32, 32, 100, 128);
        let l1 = make_recon(32, 32, 200, 128);
        let mode = BInterMode::L0_16x16 { mv: MotionVector::ZERO };
        let pred = build_b_luma_prediction(mode, &l0, &l1, 0, 0);
        for row in &pred {
            for &px in row {
                assert_eq!(px, 100, "L0 mode with zero MV should be all-100");
            }
        }
    }

    #[test]
    fn b_l1_prediction_zero_mv_matches_l1_ref() {
        let l0 = make_recon(32, 32, 100, 128);
        let l1 = make_recon(32, 32, 200, 128);
        let mode = BInterMode::L1_16x16 { mv: MotionVector::ZERO };
        let pred = build_b_luma_prediction(mode, &l0, &l1, 0, 0);
        for row in &pred {
            for &px in row {
                assert_eq!(px, 200, "L1 mode with zero MV should be all-200");
            }
        }
    }

    #[test]
    fn b_bi_prediction_zero_mv_averages_l0_l1() {
        // L0=100, L1=200 → bipred = (100+200+1)>>1 = 150.
        let l0 = make_recon(32, 32, 100, 128);
        let l1 = make_recon(32, 32, 200, 128);
        let mode = BInterMode::Bi_16x16 {
            mv_l0: MotionVector::ZERO,
            mv_l1: MotionVector::ZERO,
        };
        let pred = build_b_luma_prediction(mode, &l0, &l1, 0, 0);
        for row in &pred {
            for &px in row {
                assert_eq!(px, 150, "Bi mode should average L0+L1: (100+200+1)>>1 = 150");
            }
        }
    }

    #[test]
    fn b_chroma_bi_prediction_averages_components() {
        // Cb: L0=80, L1=120 → bipred = (80+120+1)>>1 = 100.
        let mut l0 = ReconBuffer::new(32, 32).unwrap();
        let mut l1 = ReconBuffer::new(32, 32).unwrap();
        for v in l0.cb.iter_mut() { *v = 80; }
        for v in l0.cr.iter_mut() { *v = 80; }
        for v in l1.cb.iter_mut() { *v = 120; }
        for v in l1.cr.iter_mut() { *v = 120; }
        let l0 = ReconFrame::snapshot(&l0);
        let l1 = ReconFrame::snapshot(&l1);
        let mode = BInterMode::Bi_16x16 {
            mv_l0: MotionVector::ZERO,
            mv_l1: MotionVector::ZERO,
        };
        let pred_cb = build_b_chroma_prediction(mode, &l0, &l1, /* cb */ 0, 0, 0);
        let pred_cr = build_b_chroma_prediction(mode, &l0, &l1, /* cr */ 1, 0, 0);
        for row in &pred_cb {
            for &px in row {
                assert_eq!(px, 100, "Cb bipred should be (80+120+1)>>1 = 100");
            }
        }
        for row in &pred_cr {
            for &px in row {
                assert_eq!(px, 100, "Cr bipred should match");
            }
        }
    }
}
