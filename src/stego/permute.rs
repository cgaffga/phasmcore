use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use crate::stego::cost::CostMap;

/// A coefficient position: (flat_index_in_dct_grid, cost).
#[derive(Clone)]
pub struct CoeffPos {
    /// Flat index into the DctGrid: block_index * 64 + row * 8 + col.
    pub flat_idx: usize,
    /// Embedding cost at this position.
    pub cost: f64,
}

/// Select all AC coefficient positions and apply a Fisher-Yates permutation.
///
/// Returns positions in a deterministic pseudo-random order derived from `seed`.
/// All 63 AC positions per block are included (even zeros, which have WET_COST).
/// DC positions (flat_idx % 64 == 0) are excluded.
pub fn select_and_permute(cost_map: &CostMap, seed: &[u8; 32]) -> Vec<CoeffPos> {
    let total_blocks = cost_map.total_blocks();
    let bt = cost_map.blocks_tall();
    let bw = cost_map.blocks_wide();

    // Collect nonzero AC positions in deterministic raster order.
    // Positions with infinite cost (DC and zero-valued coefficients) are excluded
    // so the STC only operates on modifiable positions.
    let mut positions: Vec<CoeffPos> = Vec::with_capacity(total_blocks * 63);
    for br in 0..bt {
        for bc in 0..bw {
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 {
                        continue; // skip DC
                    }
                    let cost = cost_map.get(br, bc, i, j);
                    if !cost.is_finite() {
                        continue; // skip zero-valued coefficients
                    }
                    let flat_idx = (br * bw + bc) * 64 + i * 8 + j;
                    positions.push(CoeffPos { flat_idx, cost });
                }
            }
        }
    }

    // Fisher-Yates shuffle with deterministic PRNG.
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let n = positions.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        positions.swap(i, j);
    }

    positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stego::cost::CostMap;

    /// Create a cost map with all AC positions having finite cost.
    fn all_finite_map(bw: usize, bt: usize) -> CostMap {
        let mut map = CostMap::new(bw, bt);
        for br in 0..bt {
            for bc in 0..bw {
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 {
                            continue;
                        }
                        map.set(br, bc, i, j, 1.0);
                    }
                }
            }
        }
        map
    }

    #[test]
    fn deterministic() {
        let map = all_finite_map(2, 2);
        let seed = [42u8; 32];
        let a = select_and_permute(&map, &seed);
        let b = select_and_permute(&map, &seed);
        let a_idx: Vec<_> = a.iter().map(|p| p.flat_idx).collect();
        let b_idx: Vec<_> = b.iter().map(|p| p.flat_idx).collect();
        assert_eq!(a_idx, b_idx);
    }

    #[test]
    fn all_finite_entries_preserved() {
        let map = all_finite_map(2, 2);
        let positions = select_and_permute(&map, &[0u8; 32]);
        // 4 blocks × 63 AC positions = 252
        assert_eq!(positions.len(), 252);
        let mut indices: Vec<_> = positions.iter().map(|p| p.flat_idx).collect();
        indices.sort();
        indices.dedup();
        assert_eq!(indices.len(), 252);
    }

    #[test]
    fn wet_positions_excluded() {
        // Default CostMap has all WET (infinity) costs.
        let map = CostMap::new(2, 2);
        let positions = select_and_permute(&map, &[0u8; 32]);
        assert_eq!(positions.len(), 0, "WET positions should be excluded");
    }

    #[test]
    fn different_seeds_differ() {
        let map = all_finite_map(4, 4);
        let a = select_and_permute(&map, &[1u8; 32]);
        let b = select_and_permute(&map, &[2u8; 32]);
        let a_idx: Vec<_> = a.iter().map(|p| p.flat_idx).collect();
        let b_idx: Vec<_> = b.iter().map(|p| p.flat_idx).collect();
        assert_ne!(a_idx, b_idx);
    }

    #[test]
    fn no_dc_positions() {
        let map = all_finite_map(3, 3);
        let positions = select_and_permute(&map, &[99u8; 32]);
        for p in &positions {
            assert_ne!(
                p.flat_idx % 64,
                0,
                "DC position included: flat_idx={}",
                p.flat_idx
            );
        }
    }
}
