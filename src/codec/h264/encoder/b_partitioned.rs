// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! §6E-A6.2 — B-slice 16x8 / 8x16 partitioned mb_type metadata.
//!
//! Spec Table 7-14 maps the 18 partitioned B mb_types (values 4..21)
//! to a `(shape, list_usage_part0, list_usage_part1)` triple.
//! Encoder + walker share this lookup so dispatch stays consistent
//! across the two sides.
//!
//! Layout is regular: nine `(part0, part1)` list-usage combinations
//! crossed with two partition shapes (16x8 / 8x16):
//!
//! | combo | (part0, part1) | 16x8 mb_type | 8x16 mb_type |
//! |---|---|---:|---:|
//! | 0 | (L0, L0)       | 4  | 5  |
//! | 1 | (L1, L1)       | 6  | 7  |
//! | 2 | (L0, L1)       | 8  | 9  |
//! | 3 | (L1, L0)       | 10 | 11 |
//! | 4 | (L0, Bi)       | 12 | 13 |
//! | 5 | (L1, Bi)       | 14 | 15 |
//! | 6 | (Bi, L0)       | 16 | 17 |
//! | 7 | (Bi, L1)       | 18 | 19 |
//! | 8 | (Bi, Bi)       | 20 | 21 |
//!
//! Mapping derived directly from spec Table 7-14 (B macroblock
//! types) — the bin-tree → mb_type translation is normative.

use super::motion_estimation::MotionVector;

/// Per-partition list usage for a B-slice partitioned MB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BListUse {
    /// Partition uses List 0 only — emits L0 MVD.
    L0,
    /// Partition uses List 1 only — emits L1 MVD.
    L1,
    /// Partition is bipred — emits L0 MVD + L1 MVD.
    Bi,
}

/// Partition shape for B mb_types 4..21. The other partitions
/// (B_8x8 = 22, with sub-MB tree) live in `b_8x8_meta` (§6E-A6.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BPartitionShape {
    /// 16x8 — two horizontal halves: partition 0 = top, 1 = bottom.
    H,
    /// 8x16 — two vertical halves: partition 0 = left, 1 = right.
    V,
}

impl BPartitionShape {
    /// Top-left anchor offset (in 4x4 blocks) of partition `idx`
    /// relative to the MB's top-left.
    pub fn part_offset(self, idx: usize) -> (usize, usize) {
        debug_assert!(idx < 2);
        match (self, idx) {
            (BPartitionShape::H, 0) => (0, 0), // top half
            (BPartitionShape::H, 1) => (0, 2), // bottom half
            (BPartitionShape::V, 0) => (0, 0), // left half
            (BPartitionShape::V, 1) => (2, 0), // right half
            _ => unreachable!(),
        }
    }
    /// Partition dimensions in 4x4 blocks.
    pub fn part_dim_4x4(self) -> (usize, usize) {
        match self {
            BPartitionShape::H => (4, 2), // 16x8 → 4 wide, 2 tall in 4x4 cells
            BPartitionShape::V => (2, 4), // 8x16
        }
    }
}

/// Decoded metadata for a partitioned B mb_type (4..21).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BPartitionedMeta {
    pub shape: BPartitionShape,
    pub part0: BListUse,
    pub part1: BListUse,
}

/// Per-partition MV pair (L0 / L1 optional). `None` for a list
/// means the partition doesn't use that list (matching `BListUse`).
#[derive(Debug, Clone, Copy, Default)]
pub struct BPartitionMv {
    pub mv_l0: Option<MotionVector>,
    pub mv_l1: Option<MotionVector>,
}

/// §6E-A6.2 — look up partition metadata for a B mb_type.
/// Returns `None` for non-partitioned mb_types (0..=3 = 16x16
/// family, 22 = B_8x8, 23 = intra-in-B). Spec Table 7-14.
pub const fn partitioned_b_meta(mb_type: u32) -> Option<BPartitionedMeta> {
    use BListUse::*;
    use BPartitionShape::*;
    let (shape, p0, p1) = match mb_type {
        4  => (H, L0, L0),
        5  => (V, L0, L0),
        6  => (H, L1, L1),
        7  => (V, L1, L1),
        8  => (H, L0, L1),
        9  => (V, L0, L1),
        10 => (H, L1, L0),
        11 => (V, L1, L0),
        12 => (H, L0, Bi),
        13 => (V, L0, Bi),
        14 => (H, L1, Bi),
        15 => (V, L1, Bi),
        16 => (H, Bi, L0),
        17 => (V, Bi, L0),
        18 => (H, Bi, L1),
        19 => (V, Bi, L1),
        20 => (H, Bi, Bi),
        21 => (V, Bi, Bi),
        _ => return None,
    };
    Some(BPartitionedMeta { shape, part0: p0, part1: p1 })
}

/// §6E-A6.2 — encode shape + per-partition list usage back into
/// the spec mb_type. Used by `mb_decision_b` once it picks a
/// partitioned variant + by `emit_b_partitioned_*` to call
/// `encode_mb_type_b` with the right value.
pub const fn mb_type_from_partitioned(meta: BPartitionedMeta) -> u32 {
    use BListUse::*;
    use BPartitionShape::*;
    let combo = match (meta.part0, meta.part1) {
        (L0, L0) => 0,
        (L1, L1) => 1,
        (L0, L1) => 2,
        (L1, L0) => 3,
        (L0, Bi) => 4,
        (L1, Bi) => 5,
        (Bi, L0) => 6,
        (Bi, L1) => 7,
        (Bi, Bi) => 8,
    };
    let shape_offset = match meta.shape {
        H => 0,
        V => 1,
    };
    4 + 2 * combo + shape_offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_round_trip_4_to_21() {
        // Forward lookup → encode-back → forward again must agree.
        for mb_type in 4..=21u32 {
            let meta = partitioned_b_meta(mb_type)
                .unwrap_or_else(|| panic!("mb_type {mb_type} should have meta"));
            let mb_type_back = mb_type_from_partitioned(meta);
            assert_eq!(mb_type, mb_type_back,
                "round-trip mismatch at mb_type {mb_type}: meta={meta:?} → back={mb_type_back}");
        }
    }

    #[test]
    fn meta_returns_none_for_non_partitioned() {
        assert!(partitioned_b_meta(0).is_none());
        assert!(partitioned_b_meta(1).is_none());
        assert!(partitioned_b_meta(2).is_none());
        assert!(partitioned_b_meta(3).is_none());
        assert!(partitioned_b_meta(22).is_none());
        assert!(partitioned_b_meta(23).is_none());
        assert!(partitioned_b_meta(99).is_none());
    }

    #[test]
    fn mb_type_4_is_h_l0_l0_per_spec() {
        // Spec Table 7-14: B_L0_L0_16x8 (mb_type 4) is two
        // horizontal halves, both using L0.
        let m = partitioned_b_meta(4).unwrap();
        assert_eq!(m.shape, BPartitionShape::H);
        assert_eq!(m.part0, BListUse::L0);
        assert_eq!(m.part1, BListUse::L0);
    }

    #[test]
    fn mb_type_21_is_v_bi_bi_per_spec() {
        // Spec Table 7-14: B_Bi_Bi_8x16 (mb_type 21) is two
        // vertical halves, both bipred.
        let m = partitioned_b_meta(21).unwrap();
        assert_eq!(m.shape, BPartitionShape::V);
        assert_eq!(m.part0, BListUse::Bi);
        assert_eq!(m.part1, BListUse::Bi);
    }

    #[test]
    fn h_partition_offsets_match_spec() {
        // H = 16x8 horizontal split: partition 0 at (0, 0)
        // covers 4-wide × 2-tall 4x4 cells (= 16x8 pixels);
        // partition 1 at (0, 2) covers the bottom 16x8.
        assert_eq!(BPartitionShape::H.part_offset(0), (0, 0));
        assert_eq!(BPartitionShape::H.part_offset(1), (0, 2));
        assert_eq!(BPartitionShape::H.part_dim_4x4(), (4, 2));
    }

    #[test]
    fn v_partition_offsets_match_spec() {
        // V = 8x16 vertical split: partition 0 at (0, 0) covers
        // 2-wide × 4-tall 4x4 cells (= 8x16 pixels); partition 1
        // at (2, 0) covers the right 8x16.
        assert_eq!(BPartitionShape::V.part_offset(0), (0, 0));
        assert_eq!(BPartitionShape::V.part_offset(1), (2, 0));
        assert_eq!(BPartitionShape::V.part_dim_4x4(), (2, 4));
    }

    #[test]
    fn all_18_partitioned_mb_types_have_unique_meta() {
        // Sanity: no two mb_types in 4..=21 produce the same
        // metadata triple. Inverse-mapping correctness depends on
        // this.
        let mut seen: Vec<BPartitionedMeta> = Vec::new();
        for mb_type in 4..=21u32 {
            let meta = partitioned_b_meta(mb_type).unwrap();
            for prior in &seen {
                assert_ne!(*prior, meta,
                    "mb_type {mb_type} duplicates a prior meta: {meta:?}");
            }
            seen.push(meta);
        }
        assert_eq!(seen.len(), 18);
    }
}
