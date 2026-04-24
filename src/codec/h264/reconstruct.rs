// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 I-frame Y-plane reconstruction for Phase 1b UNIWARD cost.
//!
//! Given a list of [`Macroblock`]s parsed from an I-slice with their
//! [`ReconstructionData`] captured, this module walks the macroblocks in
//! raster order and reconstructs the full Y-plane pixel buffer by applying
//! intra prediction + inverse quantisation + inverse 4×4 integer transform
//! on each macroblock.
//!
//! Output is a flat `Vec<u8>` of size `width × height`, ready to feed the
//! J-UNIWARD wavelet decomposition.

use super::intra_pred::{
    predict_16x16, predict_4x4, predict_chroma_8x8, Intra16x16Mode, Intra4x4Mode,
    IntraChroma8x8Mode, Neighbors16x16, Neighbors4x4, NeighborsChroma8x8,
};
use super::macroblock::{Macroblock, MbType, ReconstructionData, BLOCK_INDEX_TO_POS};
use super::transform::{
    inverse_16x16_dc_hadamard, inverse_chroma_dc_2x2_hadamard, reconstruct_residual_4x4,
    reconstruct_residual_4x4_with_dc, unzigzag_4x4,
};

/// Default fill value for MBs we can't reconstruct (P-frame MBs hit during an
/// I-frame reconstruct by mistake, I_PCM where we dropped samples, etc.).
const DEFAULT_SAMPLE: u8 = 128;

/// Reconstructed I-frame planes (Y only by default; `cb` and `cr` populated
/// by [`reconstruct_i_frame_planes`]).
pub struct IFramePlanes {
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
}

/// Reconstruct Y, Cb and Cr planes of an I-frame from parsed macroblocks.
///
/// Y plane: `width_in_mbs×16 × height_in_mbs×16` samples in raster order.
/// Cb/Cr planes: `width_in_mbs×8 × height_in_mbs×8` samples (4:2:0 subsampling).
pub fn reconstruct_i_frame_planes(
    mbs: &[Macroblock],
    width_in_mbs: usize,
    height_in_mbs: usize,
) -> IFramePlanes {
    let y = reconstruct_i_frame_y_plane(mbs, width_in_mbs, height_in_mbs);
    let cb = reconstruct_i_frame_chroma_plane(mbs, width_in_mbs, height_in_mbs, false);
    let cr = reconstruct_i_frame_chroma_plane(mbs, width_in_mbs, height_in_mbs, true);
    IFramePlanes { y, cb, cr }
}

/// Reconstruct the Y plane of an I-frame from parsed macroblocks.
///
/// `mbs` must contain `width_in_mbs × height_in_mbs` macroblocks in raster
/// order. Macroblocks without `recon` data (P/B MBs, I_PCM) are filled with
/// mid-grey — callers targeting I-slices should ensure recon capture was
/// enabled on the parse.
///
/// Returns a `width × height` u8 buffer.
pub fn reconstruct_i_frame_y_plane(
    mbs: &[Macroblock],
    width_in_mbs: usize,
    height_in_mbs: usize,
) -> Vec<u8> {
    let width = width_in_mbs * 16;
    let height = height_in_mbs * 16;
    let mut y_plane = vec![DEFAULT_SAMPLE; width * height];

    debug_assert_eq!(mbs.len(), width_in_mbs * height_in_mbs);

    for (mb_idx, mb) in mbs.iter().enumerate() {
        let mb_x = mb_idx % width_in_mbs;
        let mb_y = mb_idx / width_in_mbs;
        let origin_x = mb_x * 16;
        let origin_y = mb_y * 16;

        let Some(recon) = mb.recon.as_ref() else {
            // No reconstruction data: fill MB with 128. Only happens for
            // P-frame MBs that snuck in or I_PCM MBs (which we could fill
            // from raw samples if we chose to capture them).
            fill_mb_default(&mut y_plane, width, origin_x, origin_y);
            continue;
        };

        match mb.mb_type {
            MbType::I4x4 => {
                reconstruct_i4x4_mb(&mut y_plane, width, height, origin_x, origin_y, recon);
            }
            MbType::I16x16(_, _, _) => {
                reconstruct_i16x16_mb(
                    &mut y_plane,
                    width,
                    height,
                    origin_x,
                    origin_y,
                    recon,
                );
            }
            _ => {
                fill_mb_default(&mut y_plane, width, origin_x, origin_y);
            }
        }
    }

    y_plane
}

fn fill_mb_default(y_plane: &mut [u8], stride: usize, origin_x: usize, origin_y: usize) {
    for y in 0..16 {
        let row_start = (origin_y + y) * stride + origin_x;
        y_plane[row_start..row_start + 16].fill(DEFAULT_SAMPLE);
    }
}

/// Reconstruct an I_4x4 macroblock, 16 4×4 blocks in CAVLC decoding order.
///
/// The order visits each 8×8 sub-macroblock (top-left, top-right, bottom-left,
/// bottom-right) and within each sub-MB visits its 4 children in raster order.
/// By the time we reach any block, all samples above and to the left of it
/// have been reconstructed, so intra prediction can read its neighbours
/// straight out of `y_plane`.
fn reconstruct_i4x4_mb(
    y_plane: &mut [u8],
    stride: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    recon: &ReconstructionData,
) {
    for blk_idx in 0..16 {
        let (bx, by) = BLOCK_INDEX_TO_POS[blk_idx];
        let block_origin_x = origin_x + bx as usize * 4;
        let block_origin_y = origin_y + by as usize * 4;

        let mode = Intra4x4Mode::from_u8(recon.intra4x4_modes[blk_idx]).unwrap_or(Intra4x4Mode::Dc);
        let neighbours = collect_4x4_neighbours(y_plane, stride, height, block_origin_x, block_origin_y);
        let pred = predict_4x4(mode, &neighbours);

        let residual = reconstruct_residual_4x4(&recon.luma_blocks[blk_idx], recon.qp_y);

        write_4x4(y_plane, stride, block_origin_x, block_origin_y, &pred, &residual);
    }
}

/// Reconstruct an I_16x16 macroblock: one shared prediction, 16 AC residual
/// blocks, DC coefficients that feed every block's (0,0) position via the
/// 4×4 Hadamard path.
fn reconstruct_i16x16_mb(
    y_plane: &mut [u8],
    stride: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    recon: &ReconstructionData,
) {
    let mode_idx = recon.intra16x16_mode.unwrap_or(2);
    let mode = Intra16x16Mode::from_u8(mode_idx).unwrap_or(Intra16x16Mode::Dc);
    let neighbours = collect_16x16_neighbours(y_plane, stride, origin_x, origin_y);
    let pred = predict_16x16(mode, &neighbours);

    // Dequant + Hadamard the DC block, then dequant + IDCT each AC block
    // with the DC value injected at its (0,0) position.
    let dc_raster = match recon.luma_dc_block.as_ref() {
        Some(scan) => unzigzag_4x4(scan),
        None => [[0i32; 4]; 4],
    };
    let dc_values = inverse_16x16_dc_hadamard(&dc_raster, recon.qp_y);

    for blk_idx in 0..16 {
        let (bx, by) = BLOCK_INDEX_TO_POS[blk_idx];
        let block_origin_x = origin_x + bx as usize * 4;
        let block_origin_y = origin_y + by as usize * 4;
        let dc_val = dc_values[by as usize][bx as usize];

        let residual = reconstruct_residual_4x4_with_dc(
            &recon.luma_blocks[blk_idx],
            dc_val,
            recon.qp_y,
        );

        // Slice the prediction down to the 4×4 sub-region and write.
        let mut pred_4x4 = [[0u8; 4]; 4];
        for y in 0..4 {
            for x in 0..4 {
                pred_4x4[y][x] = pred[by as usize * 4 + y][bx as usize * 4 + x];
            }
        }
        write_4x4(y_plane, stride, block_origin_x, block_origin_y, &pred_4x4, &residual);
    }

    let _ = height; // reserved for availability handling when we need bounds checks
}

/// Collect 4×4 intra prediction neighbours from the current Y plane.
///
/// Availability mirrors H.264 Section 8.3.1.1: a neighbour is "available"
/// only when its sample block has been decoded in the current frame. For
/// intra-only I-frames that is equivalent to "inside the frame AND in
/// raster order before the current block".
fn collect_4x4_neighbours(
    y_plane: &[u8],
    stride: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
) -> Neighbors4x4 {
    let top_available = origin_y > 0;
    let left_available = origin_x > 0;
    let top_left_available = top_available && left_available;
    // Top-right neighbour availability: we need 4 more samples to the right
    // of the block's top edge. Standard H.264 rule — those samples must be
    // from a block already decoded. For our simple raster-order case it's
    // equivalent to "top row exists AND block isn't against the right edge
    // of a 16×16 MB at the top border of the frame".
    //
    // For cost-function purposes this is slightly conservative — we treat
    // "not yet decoded" as unavailable and rely on the spec's mode-specific
    // fallback (replicate top[3]).
    let top_right_available = top_available && origin_x + 8 <= stride;

    let mut top = [0u8; 8];
    if top_available {
        let row = origin_y - 1;
        for i in 0..4 {
            top[i] = y_plane[row * stride + origin_x + i];
        }
        if top_right_available {
            for i in 0..4 {
                let col = origin_x + 4 + i;
                if col < stride {
                    top[4 + i] = y_plane[row * stride + col];
                } else {
                    top[4 + i] = top[3];
                }
            }
        }
    }

    let mut left = [0u8; 4];
    if left_available {
        for i in 0..4 {
            left[i] = y_plane[(origin_y + i) * stride + origin_x - 1];
        }
    }

    let top_left = if top_left_available {
        y_plane[(origin_y - 1) * stride + origin_x - 1]
    } else {
        128
    };

    let _ = height;

    Neighbors4x4 {
        top,
        left,
        top_left,
        top_available,
        top_right_available,
        left_available,
        top_left_available,
    }
}

fn collect_16x16_neighbours(
    y_plane: &[u8],
    stride: usize,
    origin_x: usize,
    origin_y: usize,
) -> Neighbors16x16 {
    let top_available = origin_y > 0;
    let left_available = origin_x > 0;
    let top_left_available = top_available && left_available;

    let mut top = [0u8; 16];
    if top_available {
        let row = origin_y - 1;
        for i in 0..16 {
            top[i] = y_plane[row * stride + origin_x + i];
        }
    }

    let mut left = [0u8; 16];
    if left_available {
        for i in 0..16 {
            left[i] = y_plane[(origin_y + i) * stride + origin_x - 1];
        }
    }

    let top_left = if top_left_available {
        y_plane[(origin_y - 1) * stride + origin_x - 1]
    } else {
        128
    };

    Neighbors16x16 {
        top,
        left,
        top_left,
        top_available,
        left_available,
        top_left_available,
    }
}

/// Write `pred + residual` clipped to `0..=255` into the Y plane at
/// `(origin_x, origin_y)`.
fn write_4x4(
    y_plane: &mut [u8],
    stride: usize,
    origin_x: usize,
    origin_y: usize,
    pred: &[[u8; 4]; 4],
    residual: &[[i32; 4]; 4],
) {
    for y in 0..4 {
        let row = (origin_y + y) * stride + origin_x;
        for x in 0..4 {
            let v = pred[y][x] as i32 + residual[y][x];
            y_plane[row + x] = v.clamp(0, 255) as u8;
        }
    }
}

/// Reconstruct one chroma plane (Cb when `is_cr = false`, Cr otherwise) of
/// an I-frame, at 4:2:0 subsampling (half resolution of Y on each axis).
pub fn reconstruct_i_frame_chroma_plane(
    mbs: &[Macroblock],
    width_in_mbs: usize,
    height_in_mbs: usize,
    is_cr: bool,
) -> Vec<u8> {
    let width = width_in_mbs * 8;
    let height = height_in_mbs * 8;
    let mut plane = vec![DEFAULT_SAMPLE; width * height];
    let component = if is_cr { 1 } else { 0 };

    debug_assert_eq!(mbs.len(), width_in_mbs * height_in_mbs);

    for (mb_idx, mb) in mbs.iter().enumerate() {
        let mb_x = mb_idx % width_in_mbs;
        let mb_y = mb_idx / width_in_mbs;
        let origin_x = mb_x * 8;
        let origin_y = mb_y * 8;

        let Some(recon) = mb.recon.as_ref() else {
            fill_chroma_default(&mut plane, width, origin_x, origin_y);
            continue;
        };

        let mode = IntraChroma8x8Mode::from_u8(recon.intra_chroma_pred_mode)
            .unwrap_or(IntraChroma8x8Mode::Dc);
        let neighbours = collect_chroma_neighbours(&plane, width, origin_x, origin_y);
        let pred = predict_chroma_8x8(mode, &neighbours);

        // Residual path: if cbp_chroma == 0 (no DC, no AC), plane[...] = prediction.
        // Otherwise decode DC via the 2×2 Hadamard, then each 4×4 AC block with its
        // DC value injected.
        let qp_c = if is_cr { recon.qp_cr } else { recon.qp_cb };
        let dc_block = recon.chroma_dc_blocks[component];
        let ac_blocks = &recon.chroma_ac_blocks[component];

        // Dequant + Hadamard the 4-entry DC, layout as 2×2.
        let dc_values: [[i32; 2]; 2] = match dc_block {
            Some(scan4) => {
                let mut c = [[0i32; 2]; 2];
                c[0][0] = scan4[0];
                c[0][1] = scan4[1];
                c[1][0] = scan4[2];
                c[1][1] = scan4[3];
                inverse_chroma_dc_2x2_hadamard(&c, qp_c)
            }
            None => [[0; 2]; 2],
        };

        for blk_idx in 0..4usize {
            let bx = blk_idx % 2;
            let by = blk_idx / 2;
            let block_origin_x = origin_x + bx * 4;
            let block_origin_y = origin_y + by * 4;

            let residual =
                reconstruct_residual_4x4_with_dc(&ac_blocks[blk_idx], dc_values[by][bx], qp_c);

            // Slice the 8×8 prediction down to the 4×4 sub-region.
            let mut pred_4x4 = [[0u8; 4]; 4];
            for y in 0..4 {
                for x in 0..4 {
                    pred_4x4[y][x] = pred[by * 4 + y][bx * 4 + x];
                }
            }
            write_4x4(&mut plane, width, block_origin_x, block_origin_y, &pred_4x4, &residual);
        }
    }

    plane
}

fn fill_chroma_default(plane: &mut [u8], stride: usize, origin_x: usize, origin_y: usize) {
    for y in 0..8 {
        let row_start = (origin_y + y) * stride + origin_x;
        plane[row_start..row_start + 8].fill(DEFAULT_SAMPLE);
    }
}

fn collect_chroma_neighbours(
    plane: &[u8],
    stride: usize,
    origin_x: usize,
    origin_y: usize,
) -> NeighborsChroma8x8 {
    let top_available = origin_y > 0;
    let left_available = origin_x > 0;
    let top_left_available = top_available && left_available;

    let mut top = [0u8; 8];
    if top_available {
        let row = origin_y - 1;
        for i in 0..8 {
            top[i] = plane[row * stride + origin_x + i];
        }
    }

    let mut left = [0u8; 8];
    if left_available {
        for i in 0..8 {
            left[i] = plane[(origin_y + i) * stride + origin_x - 1];
        }
    }

    let top_left = if top_left_available {
        plane[(origin_y - 1) * stride + origin_x - 1]
    } else {
        128
    };

    NeighborsChroma8x8 {
        top,
        left,
        top_left,
        top_available,
        left_available,
        top_left_available,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_macroblock(mb_type: MbType, recon: Option<ReconstructionData>) -> Macroblock {
        Macroblock {
            mb_type,
            mb_qp_delta: 0,
            coded_block_pattern: 0,
            luma_total_coeffs: [0; 16],
            chroma_total_coeffs: [0; 8],
            positions: Vec::new(),
            recon,
            mv_field: None,
        }
    }

    #[test]
    fn reconstruct_all_none_gives_grey_frame() {
        // With no reconstruction data, the output is a flat 128 buffer.
        let mbs = vec![
            empty_macroblock(MbType::P16x16, None),
            empty_macroblock(MbType::P16x16, None),
        ];
        let y = reconstruct_i_frame_y_plane(&mbs, 2, 1);
        assert_eq!(y.len(), 32 * 16);
        assert!(y.iter().all(|&v| v == 128));
    }

    #[test]
    fn reconstruct_zero_residual_i16x16_dc_matches_neighbour_average() {
        // Single I_16x16 MB with DC prediction mode and zero residuals. Since
        // there is no neighbour data (top-left corner of the frame), the DC
        // fallback is 128 → output should be a flat 128 block.
        let mut recon = ReconstructionData::empty(26);
        recon.intra16x16_mode = Some(Intra16x16Mode::Dc as u8);
        // luma_dc_block = all zeros -> DC contributions are zero after
        // Hadamard, so residuals are zero and pixels == prediction == 128.
        recon.luma_dc_block = Some([0; 16]);

        let mbs = vec![empty_macroblock(MbType::I16x16(2, 0, 0), Some(recon))];
        let y = reconstruct_i_frame_y_plane(&mbs, 1, 1);
        assert!(
            y.iter().all(|&v| v == 128),
            "expected flat 128, got min={} max={}",
            y.iter().min().unwrap(),
            y.iter().max().unwrap()
        );
    }

    #[test]
    fn reconstruct_zero_residual_i4x4_dc_at_frame_corner_is_128() {
        // 16 I_4x4 blocks all in DC mode with zero residuals. At the top-left
        // corner of the frame no neighbours are available, so the very first
        // block falls back to DC=128. Subsequent blocks see their
        // already-reconstructed 128 neighbours and compute DC = 128, so the
        // whole MB ends up 128.
        let mut recon = ReconstructionData::empty(26);
        for m in &mut recon.intra4x4_modes {
            *m = Intra4x4Mode::Dc as u8;
        }
        // luma_blocks left at zero.
        let mbs = vec![empty_macroblock(MbType::I4x4, Some(recon))];
        let y = reconstruct_i_frame_y_plane(&mbs, 1, 1);
        assert!(y.iter().all(|&v| v == 128));
    }
}
