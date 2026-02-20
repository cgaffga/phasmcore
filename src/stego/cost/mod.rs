//! Embedding cost functions for steganographic coding.
//!
//! Cost functions assign a distortion value to each DCT coefficient position,
//! guiding the STC encoder to modify only the least detectable positions.
//! Lower cost = safer to modify. Infinite cost (`WET_COST`) = must never modify.
//!
//! Currently implements UERD (Uniform Embedding Revisited Distortion). Future
//! phases will add J-UNIWARD and SI-UNIWARD.

pub mod uerd;

/// Cost assigned to coefficients that must never be modified.
/// Using infinity ensures the Viterbi STC never selects these positions for flipping.
pub const WET_COST: f64 = f64::INFINITY;

/// Per-coefficient embedding costs for one image component.
///
/// Stored in the same block-raster order as `DctGrid`: block index * 64 + row * 8 + col.
/// DC positions (index % 64 == 0) always have `WET_COST`.
pub struct CostMap {
    /// Number of 8×8 blocks horizontally.
    blocks_wide: usize,
    /// Number of 8×8 blocks vertically.
    blocks_tall: usize,
    /// Flat storage: one f64 cost per coefficient position.
    costs: Vec<f64>,
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
    pub fn get(&self, br: usize, bc: usize, i: usize, j: usize) -> f64 {
        self.costs[self.index(br, bc, i, j)]
    }

    /// Set the cost at block (br, bc), frequency position (i, j).
    pub fn set(&mut self, br: usize, bc: usize, i: usize, j: usize, val: f64) {
        let idx = self.index(br, bc, i, j);
        self.costs[idx] = val;
    }

    /// Get cost by flat coefficient index (block_idx * 64 + pos_in_block).
    pub fn get_flat(&self, idx: usize) -> f64 {
        self.costs[idx]
    }

    fn index(&self, br: usize, bc: usize, i: usize, j: usize) -> usize {
        (br * self.blocks_wide + bc) * 64 + i * 8 + j
    }
}
