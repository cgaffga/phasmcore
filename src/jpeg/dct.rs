// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! DCT coefficient storage and quantization tables.
//!
//! Provides [`DctGrid`] for storing quantized DCT coefficients in block-raster
//! order, and [`QuantTable`] for the 64-entry quantization matrices.

/// Quantization table: 64 values in natural (row-major) order.
#[derive(Debug, Clone)]
pub struct QuantTable {
    /// Quantization values, indexed by row * 8 + col.
    pub values: [u16; 64],
}

impl QuantTable {
    pub fn new(values: [u16; 64]) -> Self {
        Self { values }
    }
}

/// Grid of quantized DCT coefficients for one image component.
///
/// Coefficients are stored in block-raster order. Within each block,
/// the 64 coefficients are in natural (row-major) order, i.e. index = row * 8 + col.
#[derive(Debug, Clone)]
pub struct DctGrid {
    /// Number of 8×8 blocks horizontally.
    blocks_wide: usize,
    /// Number of 8×8 blocks vertically.
    blocks_tall: usize,
    /// Flat storage: blocks_tall * blocks_wide * 64 coefficients.
    coeffs: Vec<i16>,
}

impl DctGrid {
    /// Create a new grid initialized to zero.
    pub fn new(blocks_wide: usize, blocks_tall: usize) -> Self {
        Self {
            blocks_wide,
            blocks_tall,
            coeffs: vec![0i16; blocks_wide * blocks_tall * 64],
        }
    }

    pub fn blocks_wide(&self) -> usize {
        self.blocks_wide
    }

    pub fn blocks_tall(&self) -> usize {
        self.blocks_tall
    }

    /// Get a coefficient value.
    /// - `br`, `bc`: block row and column (0-based)
    /// - `i`, `j`: frequency row and column within the block (0–7)
    pub fn get(&self, br: usize, bc: usize, i: usize, j: usize) -> i16 {
        self.coeffs[self.index(br, bc, i, j)]
    }

    /// Set a coefficient value.
    pub fn set(&mut self, br: usize, bc: usize, i: usize, j: usize, val: i16) {
        let idx = self.index(br, bc, i, j);
        self.coeffs[idx] = val;
    }

    /// Get a mutable reference to the 64-coefficient block at (br, bc).
    pub fn block_mut(&mut self, br: usize, bc: usize) -> &mut [i16] {
        let start = (br * self.blocks_wide + bc) * 64;
        &mut self.coeffs[start..start + 64]
    }

    /// Get a reference to the 64-coefficient block at (br, bc).
    pub fn block(&self, br: usize, bc: usize) -> &[i16] {
        let start = (br * self.blocks_wide + bc) * 64;
        &self.coeffs[start..start + 64]
    }

    /// Total number of blocks.
    pub fn total_blocks(&self) -> usize {
        self.blocks_wide * self.blocks_tall
    }

    /// Raw mutable access to all coefficients.
    ///
    /// Layout: `blocks_tall * blocks_wide * 64` contiguous i16 values in
    /// block-raster order. Each 64-element chunk is one 8×8 block.
    /// Used by parallel processing (Rayon `par_chunks_mut`).
    pub fn coeffs_mut(&mut self) -> &mut [i16] {
        &mut self.coeffs
    }

    /// Raw read-only access to all coefficients.
    pub fn coeffs(&self) -> &[i16] {
        &self.coeffs
    }

    fn index(&self, br: usize, bc: usize, i: usize, j: usize) -> usize {
        debug_assert!(br < self.blocks_tall, "block row {br} >= {}", self.blocks_tall);
        debug_assert!(bc < self.blocks_wide, "block col {bc} >= {}", self.blocks_wide);
        debug_assert!(i < 8 && j < 8);
        (br * self.blocks_wide + bc) * 64 + i * 8 + j
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_get_set() {
        let mut grid = DctGrid::new(2, 3);
        assert_eq!(grid.blocks_wide(), 2);
        assert_eq!(grid.blocks_tall(), 3);
        assert_eq!(grid.total_blocks(), 6);

        // All initialized to zero
        assert_eq!(grid.get(0, 0, 0, 0), 0);
        assert_eq!(grid.get(2, 1, 7, 7), 0);

        grid.set(1, 0, 3, 4, 42);
        assert_eq!(grid.get(1, 0, 3, 4), 42);

        // Other positions unchanged
        assert_eq!(grid.get(1, 0, 3, 3), 0);
        assert_eq!(grid.get(0, 0, 3, 4), 0);
    }

    #[test]
    fn block_slice_access() {
        let mut grid = DctGrid::new(1, 1);
        grid.set(0, 0, 0, 0, 100); // DC
        grid.set(0, 0, 7, 7, -50);

        let blk = grid.block(0, 0);
        assert_eq!(blk[0], 100);
        assert_eq!(blk[63], -50);
        assert_eq!(blk.len(), 64);
    }

    #[test]
    fn block_mut_access() {
        let mut grid = DctGrid::new(2, 2);
        let blk = grid.block_mut(1, 1);
        for (i, v) in blk.iter_mut().enumerate() {
            *v = i as i16;
        }
        assert_eq!(grid.get(1, 1, 0, 0), 0);
        assert_eq!(grid.get(1, 1, 0, 1), 1);
        assert_eq!(grid.get(1, 1, 7, 7), 63);
        // Other block untouched
        assert_eq!(grid.get(0, 0, 0, 0), 0);
    }

    #[test]
    fn quant_table() {
        let mut vals = [0u16; 64];
        vals[0] = 16;
        vals[63] = 99;
        let qt = QuantTable::new(vals);
        assert_eq!(qt.values[0], 16);
        assert_eq!(qt.values[63], 99);
    }
}
