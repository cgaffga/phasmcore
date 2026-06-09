// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 4×4 scan-order tables.
//!
//! The CAVLC VLC lookup tables (coeff_token / total_zeros / run_before)
//! were retired with the CAVLC subsystem. The surviving CABAC walker +
//! OH264 cost path only need the 4×4 zigzag scan orders and the block
//! raster→position map, which live here.

// ---------------------------------------------------------------------------
// 4x4 Zigzag Scan Order (H.264 Table 8-13)
// ---------------------------------------------------------------------------

/// 4x4 zigzag scan: maps scan index (0-15) to raster index (row*4+col).
pub const ZIGZAG_4X4: [u8; 16] = [
    0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15,
];

/// Inverse zigzag: maps raster index to scan index.
pub const ZIGZAG_4X4_INV: [u8; 16] = [
    0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15,
];

/// 4x4 block raster-scan index within a macroblock → (col, row) within MB.
/// H.264 scans 4x4 luma blocks in raster order within each 8x8 block,
/// and the four 8x8 blocks in raster order.
/// Block index 0-3: top-left 8x8, Block index 4-7: top-right 8x8, etc.
///
/// (Re-homed from `macroblock.rs` in the CAVLC retirement — it is the
/// one item the surviving walker + OH264 cost path still needed out of the
/// otherwise-deleted CAVLC macroblock module.)
pub const BLOCK_INDEX_TO_POS: [(u8, u8); 16] = [
    (0, 0), (1, 0), (0, 1), (1, 1), // 8x8 block 0 (top-left)
    (2, 0), (3, 0), (2, 1), (3, 1), // 8x8 block 1 (top-right)
    (0, 2), (1, 2), (0, 3), (1, 3), // 8x8 block 2 (bottom-left)
    (2, 2), (3, 2), (2, 3), (3, 3), // 8x8 block 3 (bottom-right)
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_4x4_inverse() {
        for i in 0..16u8 {
            assert_eq!(ZIGZAG_4X4_INV[ZIGZAG_4X4[i as usize] as usize], i);
        }
    }
}
