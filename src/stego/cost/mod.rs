// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Embedding cost functions for steganographic coding.
//!
//! Cost functions assign a distortion value to each DCT coefficient position,
//! guiding the STC encoder to modify only the least detectable positions.
//! Lower cost = safer to modify. Infinite cost (`WET_COST`) = must never modify.
//!
//! Implements:
//! - **J-UNIWARD** (JPEG Universal Wavelet Relative Distortion): state-of-the-art
//!   cost function using directional wavelet decomposition for superior
//!   steganalysis resistance.
//! - **UERD** (test-only): simple block-energy cost, replaced by J-UNIWARD in production.

#[cfg(test)]
pub mod uerd;
pub mod uniward;

/// Cost assigned to coefficients that must never be modified (f32 version).
/// Using infinity ensures the Viterbi STC never selects these positions for flipping.
pub const WET_COST: f32 = f32::INFINITY;

/// WET_COST as f64 for use in the STC pipeline (which operates in f64).
pub const WET_COST_F64: f64 = f64::INFINITY;

/// Per-coefficient embedding costs for one image component.
///
/// Stored in the same block-raster order as `DctGrid`: block index * 64 + row * 8 + col.
/// DC positions (index % 64 == 0) always have `WET_COST`.
///
/// Uses f32 storage to halve memory usage (~46 MB instead of ~93 MB for a
/// 4032x3024 image). f32 has more than enough precision for cost ranking.
pub struct CostMap {
    /// Number of 8×8 blocks horizontally.
    blocks_wide: usize,
    /// Number of 8×8 blocks vertically.
    blocks_tall: usize,
    /// Flat storage: one f32 cost per coefficient position.
    costs: Vec<f32>,
}

impl CostMap {
    pub fn new(blocks_wide: usize, blocks_tall: usize) -> Self {
        let n = blocks_wide * blocks_tall * 64;
        Self {
            blocks_wide,
            blocks_tall,
            costs: vec![WET_COST; n],
        }
    }

    pub fn blocks_wide(&self) -> usize {
        self.blocks_wide
    }

    pub fn blocks_tall(&self) -> usize {
        self.blocks_tall
    }

    pub fn total_blocks(&self) -> usize {
        self.blocks_wide * self.blocks_tall
    }

    /// Get the cost at block (br, bc), frequency position (i, j).
    pub fn get(&self, br: usize, bc: usize, i: usize, j: usize) -> f32 {
        self.costs[self.index(br, bc, i, j)]
    }

    /// Set the cost at block (br, bc), frequency position (i, j).
    pub fn set(&mut self, br: usize, bc: usize, i: usize, j: usize, val: f32) {
        let idx = self.index(br, bc, i, j);
        self.costs[idx] = val;
    }

    /// Get a raw pointer into the cost storage for direct writes.
    ///
    /// # Safety
    /// Caller must ensure no aliasing writes to the same index.
    /// Safe when blocks write to non-overlapping regions (each block
    /// occupies indices `[block_idx * 64 .. block_idx * 64 + 64)`).
    pub(crate) fn costs_ptr(&mut self) -> *mut f32 {
        self.costs.as_mut_ptr()
    }

    fn index(&self, br: usize, bc: usize, i: usize, j: usize) -> usize {
        (br * self.blocks_wide + bc) * 64 + i * 8 + j
    }
}
