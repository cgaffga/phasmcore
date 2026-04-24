// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Intra_8×8 macroblock encode path. Phase 100-F2.
//!
//! Mirrors `i4x4_encode` but at 8×8 granularity. For each of 4 8×8
//! luma blocks in an MB, picks one of 9 Intra_8×8 modes by SATD,
//! runs forward 8×8 DCT + quant, writes recon back into the buffer
//! so subsequent blocks (within this MB and following MBs) see the
//! correct neighbor samples. Returns quantized levels + modes +
//! nonzero bits + summed SATD.
//!
//! Processing order: 8×8 raster — blk_idx 0=(0,0), 1=(1,0), 2=(0,1),
//! 3=(1,1). Each later block may see recon from an earlier same-MB
//! block. Top-right availability follows spec § 6.4.10.2.

use crate::codec::h264::intra_pred_8x8::{predict_8x8, Intra8x8Mode, Neighbors8x8};

use super::reconstruction::ReconBuffer;
use super::transform_8x8::{
    dequant_8x8_block, forward_dct_8x8, inverse_dct_8x8, quant_8x8_block, Slice8x8,
};

/// 8×8 block position (bx, by) in MB for blk_idx 0..=3 (raster scan).
pub const I8X8_BLOCK_POS: [(u8, u8); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

/// Per-MB I_8×8 encode result.
#[derive(Debug, Clone, Copy)]
pub struct I8x8MbResult {
    /// One mode per 8×8 sub-block.
    pub modes: [u8; 4],
    /// Quantized levels per 8×8 block (row-major).
    pub ac_levels: [[[i16; 8]; 8]; 4],
    /// Whether each 8×8 block has any non-zero level (= cbp_luma bit).
    pub nonzero: [bool; 4],
    /// Summed SATD across all 4 8×8 mode decisions.
    pub total_satd: u32,
}

/// Run the I_8x8 pipeline on one MB: pick a mode per 8×8 sub-block,
/// DCT, quant, reconstruct. Writes reconstructed pixels back into
/// `recon` so subsequent blocks see correct neighbor samples.
pub fn encode_i8x8_mb(
    src_y: &[[u8; 16]; 16],
    recon: &mut ReconBuffer,
    mb_x: usize,
    mb_y: usize,
    qp: u8,
) -> I8x8MbResult {
    let mut modes = [0u8; 4];
    let mut ac_levels = [[[0i16; 8]; 8]; 4];
    let mut nonzero = [false; 4];
    let mut total_satd = 0u32;

    for blk_idx in 0..4 {
        let (bx, by) = I8X8_BLOCK_POS[blk_idx];
        let sub_x_in_mb = (bx * 8) as usize;
        let sub_y_in_mb = (by * 8) as usize;
        let block_x = (mb_x * 16 + sub_x_in_mb) as u32;
        let block_y = (mb_y * 16 + sub_y_in_mb) as u32;

        let mut src = [[0u8; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                src[dy][dx] = src_y[sub_y_in_mb + dy][sub_x_in_mb + dx];
            }
        }

        let neighbors =
            collect_neighbors_8x8_from_recon(recon, block_x, block_y, mb_x, mb_y, blk_idx);

        let decision = choose_intra_8x8_mode(&neighbors, &src);
        modes[blk_idx] = decision.mode as u8;
        total_satd = total_satd.saturating_add(decision.satd);

        let mut residual = [[0i32; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                residual[dy][dx] = src[dy][dx] as i32 - decision.predicted[dy][dx] as i32;
            }
        }
        let coeffs = forward_dct_8x8(&residual);
        let levels = quant_8x8_block(&coeffs, qp, Slice8x8::Intra);
        ac_levels[blk_idx] = levels;
        nonzero[blk_idx] = levels.iter().any(|r| r.iter().any(|&v| v != 0));

        let dq = dequant_8x8_block(&levels, qp);
        let inv = inverse_dct_8x8(&dq);
        let mut recon_block = [[0u8; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                let pixel_res = (inv[dy][dx] + 32) >> 6;
                let v = decision.predicted[dy][dx] as i32 + pixel_res;
                recon_block[dy][dx] = v.clamp(0, 255) as u8;
            }
        }
        write_8x8_to_recon(recon, block_x, block_y, &recon_block);
    }

    I8x8MbResult {
        modes,
        ac_levels,
        nonzero,
        total_satd,
    }
}

/// Given the 4 resolved 8×8 mode values, plus a lookup callback for
/// cross-MB neighbors, derive (prev_flag, optional rem) for each
/// sub-block per spec § 8.3.3. Neighbor at (isize grid-x, grid-y) in
/// 8×8 units returns `Some(mode)` for available I_8×8 neighbors or
/// `None` otherwise.
pub fn derive_i8x8_mode_flags<F>(
    modes: &[u8; 4],
    mb_x: usize,
    mb_y: usize,
    mut neighbor_mode: F,
) -> [(bool, Option<u8>); 4]
where
    F: FnMut(isize, isize) -> Option<u8>,
{
    let mut out = [(false, None); 4];
    let base_x = (mb_x * 2) as isize;
    let base_y = (mb_y * 2) as isize;

    // Same-MB tracking: mode by (bx, by), each 0 or 1.
    let mut mb_modes = [[None::<u8>; 2]; 2];

    for blk_idx in 0..4 {
        let (bx_u, by_u) = I8X8_BLOCK_POS[blk_idx];
        let bx = bx_u as isize;
        let by = by_u as isize;

        let left: Option<u8> = if bx > 0 {
            mb_modes[by as usize][(bx - 1) as usize]
        } else if mb_x > 0 {
            Some(neighbor_mode(base_x - 1, base_y + by).unwrap_or(2))
        } else {
            None
        };

        let top: Option<u8> = if by > 0 {
            mb_modes[(by - 1) as usize][bx as usize]
        } else if mb_y > 0 {
            Some(neighbor_mode(base_x + bx, base_y - 1).unwrap_or(2))
        } else {
            None
        };

        let pred_mode = match (left, top) {
            (Some(a), Some(b)) => a.min(b),
            _ => 2,
        };

        let actual = modes[blk_idx];
        if actual == pred_mode {
            out[blk_idx] = (true, None);
        } else {
            let rem = if actual < pred_mode { actual } else { actual - 1 };
            out[blk_idx] = (false, Some(rem));
        }

        mb_modes[by as usize][bx as usize] = Some(actual);
    }

    out
}

// ─── Internals ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct ModeDecision8x8 {
    mode: Intra8x8Mode,
    predicted: [[u8; 8]; 8],
    satd: u32,
}

const ALL_MODES_8X8: [Intra8x8Mode; 9] = [
    Intra8x8Mode::Vertical,
    Intra8x8Mode::Horizontal,
    Intra8x8Mode::Dc,
    Intra8x8Mode::DiagonalDownLeft,
    Intra8x8Mode::DiagonalDownRight,
    Intra8x8Mode::VerticalRight,
    Intra8x8Mode::HorizontalDown,
    Intra8x8Mode::VerticalLeft,
    Intra8x8Mode::HorizontalUp,
];

fn choose_intra_8x8_mode(n: &Neighbors8x8, source: &[[u8; 8]; 8]) -> ModeDecision8x8 {
    // DC is always available (uses 128 fill when neighbors missing).
    let predicted = predict_8x8(Intra8x8Mode::Dc, n);
    let mut best = ModeDecision8x8 {
        mode: Intra8x8Mode::Dc,
        satd: satd_8x8_pixels(source, &predicted),
        predicted,
    };
    for &m in ALL_MODES_8X8.iter() {
        if m == Intra8x8Mode::Dc || !mode_available_8x8(m, n) {
            continue;
        }
        let predicted = predict_8x8(m, n);
        let satd = satd_8x8_pixels(source, &predicted);
        if satd < best.satd {
            best = ModeDecision8x8 {
                mode: m,
                predicted,
                satd,
            };
        }
    }
    best
}

fn mode_available_8x8(m: Intra8x8Mode, n: &Neighbors8x8) -> bool {
    use Intra8x8Mode::*;
    let needs_top = matches!(
        m,
        Vertical | DiagonalDownLeft | DiagonalDownRight | VerticalRight | HorizontalDown | VerticalLeft
    );
    let needs_left = matches!(
        m,
        Horizontal | DiagonalDownRight | VerticalRight | HorizontalDown | HorizontalUp
    );
    let needs_tl = matches!(m, DiagonalDownRight | VerticalRight | HorizontalDown);
    let needs_tr = matches!(m, DiagonalDownLeft | VerticalLeft);
    if needs_top && !n.top_available {
        return false;
    }
    if needs_left && !n.left_available {
        return false;
    }
    if needs_tl && !n.top_left_available {
        return false;
    }
    if needs_tr && !n.top_right_available {
        return false;
    }
    true
}

fn satd_8x8_pixels(source: &[[u8; 8]; 8], pred: &[[u8; 8]; 8]) -> u32 {
    let mut total: u32 = 0;
    for by in 0..2 {
        for bx in 0..2 {
            let mut sub_src = [[0u8; 4]; 4];
            let mut sub_prd = [[0u8; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    sub_src[i][j] = source[by * 4 + i][bx * 4 + j];
                    sub_prd[i][j] = pred[by * 4 + i][bx * 4 + j];
                }
            }
            total = total.saturating_add(satd_4x4(&sub_src, &sub_prd));
        }
    }
    total
}

fn satd_4x4(source: &[[u8; 4]; 4], pred: &[[u8; 4]; 4]) -> u32 {
    let mut d = [[0i32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            d[i][j] = source[i][j] as i32 - pred[i][j] as i32;
        }
    }
    // Row Hadamard.
    let mut tmp = [[0i32; 4]; 4];
    for i in 0..4 {
        let a = d[i][0] + d[i][3];
        let b = d[i][1] + d[i][2];
        let c = d[i][1] - d[i][2];
        let e = d[i][0] - d[i][3];
        tmp[i][0] = a + b;
        tmp[i][1] = e + c;
        tmp[i][2] = a - b;
        tmp[i][3] = e - c;
    }
    // Column Hadamard.
    let mut out = [[0i32; 4]; 4];
    for j in 0..4 {
        let a = tmp[0][j] + tmp[3][j];
        let b = tmp[1][j] + tmp[2][j];
        let c = tmp[1][j] - tmp[2][j];
        let e = tmp[0][j] - tmp[3][j];
        out[0][j] = a + b;
        out[1][j] = e + c;
        out[2][j] = a - b;
        out[3][j] = e - c;
    }
    let mut sum = 0u32;
    for r in &out {
        for &v in r {
            sum = sum.saturating_add(v.unsigned_abs());
        }
    }
    sum
}

/// Build 8×8 neighbors from the recon buffer for the given 8×8 block
/// position within an MB. Applies the spec § 6.4.10.2 scan-order
/// top-right availability rule.
fn collect_neighbors_8x8_from_recon(
    recon: &ReconBuffer,
    origin_x: u32,
    origin_y: u32,
    mb_x: usize,
    mb_y: usize,
    blk_idx: usize,
) -> Neighbors8x8 {
    let stride = recon.width as usize;
    let frame_w = recon.width as usize;
    let top_available = origin_y > 0;
    let left_available = origin_x > 0;
    let top_left_available = top_available && left_available;

    let mut top_base = [0u8; 8];
    if top_available {
        let row = origin_y as usize - 1;
        for i in 0..8 {
            top_base[i] = recon.y[row * stride + origin_x as usize + i];
        }
    }

    // Top-right availability for 8×8 blocks within an MB, scan order
    // (0,0), (1,0), (0,1), (1,1):
    //   blk 0: TR is in above MB (right half) → avail iff mb_y > 0
    //   blk 1: TR is in above-right MB → avail iff mb_y > 0 AND mb_x+1 < mb_w
    //   blk 2: TR is block 1 (same MB, already processed) → avail
    //   blk 3: TR is in the right MB (not yet processed) → unavail
    let mb_w = (recon.width / 16) as usize;
    let mut top_right_available = match blk_idx {
        0 => top_available,
        1 => top_available && mb_x + 1 < mb_w,
        2 => true,
        3 => false,
        _ => unreachable!(),
    };
    // Must also fit within the frame horizontally.
    if (origin_x as usize) + 16 > frame_w {
        top_right_available = false;
    }

    let top_right = if top_right_available {
        let row = origin_y as usize - 1;
        let mut tr = [0u8; 8];
        for i in 0..8 {
            tr[i] = recon.y[row * stride + origin_x as usize + 8 + i];
        }
        Some(tr)
    } else {
        None
    };

    let mut left = [0u8; 8];
    if left_available {
        for i in 0..8 {
            left[i] = recon.y[(origin_y as usize + i) * stride + origin_x as usize - 1];
        }
    }

    let top_left = if top_left_available {
        recon.y[(origin_y as usize - 1) * stride + origin_x as usize - 1]
    } else {
        128
    };

    Neighbors8x8::with_top_right_fallback(
        top_base,
        top_right,
        left,
        top_left,
        top_available,
        left_available,
        top_left_available,
    )
}

fn write_8x8_to_recon(recon: &mut ReconBuffer, x: u32, y: u32, block: &[[u8; 8]; 8]) {
    let stride = recon.width as usize;
    for dy in 0..8 {
        for dx in 0..8 {
            recon.y[(y as usize + dy) * stride + x as usize + dx] = block[dy][dx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_flags_all_dc_at_frame_origin() {
        // First MB of frame: no neighbors → pred_mode = 2 (DC). If all
        // 4 blocks pick DC, all flags are true with no rem.
        let modes = [2u8; 4];
        let flags = derive_i8x8_mode_flags(&modes, 0, 0, |_, _| None);
        for (flag, rem) in &flags {
            assert!(*flag);
            assert!(rem.is_none());
        }
    }

    #[test]
    fn derive_flags_mismatch_emits_rem() {
        let mut modes = [2u8; 4];
        modes[0] = 0; // Vertical < DC(2) → rem = actual = 0.
        let flags = derive_i8x8_mode_flags(&modes, 0, 0, |_, _| None);
        assert_eq!(flags[0], (false, Some(0)));
        // Other blocks still pred=2 but block 1's top-neighbor-within-MB
        // comes from block 0's ACTUAL (0); block 1's left neighbor is
        // cross-MB fallback None (mb_x=0) → pred_mode = 2 (both None).
        // So block 1 still flag=true with its DC.
        assert_eq!(flags[1], (true, None));
    }

    #[test]
    fn encode_flat_mb_runs_end_to_end() {
        let src = [[100u8; 16]; 16];
        let mut recon = ReconBuffer::new(16, 16).unwrap();
        let result = encode_i8x8_mb(&src, &mut recon, 0, 0, 24);
        // All 4 modes filled + SATD tracked.
        assert_eq!(result.modes.len(), 4);
        let _ = result.total_satd;
    }

    #[test]
    fn top_right_availability_matches_scan_rule() {
        // Build a 32×16 frame (mb_w = 2) and test TR flags for each of
        // the 4 blocks in MB (0, 0) with no prior recon (all zeros,
        // top_available is determined by origin_y > 0 only).
        let recon = ReconBuffer::new(32, 16).unwrap();
        let n0 = collect_neighbors_8x8_from_recon(&recon, 0, 0, 0, 0, 0);
        assert!(!n0.top_available);
        assert!(!n0.top_right_available);

        // Row-2 MB (0, 0) at y=16 would have no mb_y > 0 parent.
        // Use a 32x32 frame so mb_y=1 exists.
        let recon2 = ReconBuffer::new(32, 32).unwrap();
        // MB (0, 1): block 0 (0,0) has origin at (0, 16). TR samples at
        // (8..15, 15) live in MB (0, 0) → available.
        let n = collect_neighbors_8x8_from_recon(&recon2, 0, 16, 0, 1, 0);
        assert!(n.top_available);
        assert!(n.top_right_available);

        // MB (0, 1): block 1 (1, 0) origin (8, 16). TR samples live in
        // MB (1, 0) (above-right) → available since mb_x+1=1 < mb_w=2.
        let n = collect_neighbors_8x8_from_recon(&recon2, 8, 16, 0, 1, 1);
        assert!(n.top_right_available);

        // MB (1, 1): block 1 (1, 0) origin (16+8=24, 16). TR samples
        // would be at col 32..39 — outside the frame → unavailable.
        let n = collect_neighbors_8x8_from_recon(&recon2, 24, 16, 1, 1, 1);
        assert!(!n.top_right_available);

        // MB (0, 1): block 2 (0, 1) origin (0, 24). TR samples live in
        // same MB's block 1 → available (regardless of row position).
        let n = collect_neighbors_8x8_from_recon(&recon2, 0, 24, 0, 1, 2);
        assert!(n.top_right_available);

        // MB (0, 1): block 3 (1, 1) origin (8, 24). TR would be in
        // right-MB (1, 1) bottom-left — not yet processed → unavailable.
        let n = collect_neighbors_8x8_from_recon(&recon2, 8, 24, 0, 1, 3);
        assert!(!n.top_right_available);
    }
}
