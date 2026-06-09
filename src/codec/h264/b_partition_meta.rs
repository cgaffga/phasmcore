// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! B-slice 16x8 / 8x16 partitioned mb_type metadata.
//!
//! Spec Table 7-14 maps the 18 partitioned B mb_types (values 4..21) to a
//! `(shape, list_usage_part0, list_usage_part1)` triple. The H.264 walker
//! (decode) consumes this lookup, so it lives in this small, dependency-free
//! module (no `MotionVector`, no encoder coupling) that the decoder can reach
//! under `h264-decoder`. (The pure-Rust encoder that also consumed it was
//! removed in the 2026-06 video-retirement; OpenH264 is now the sole encoder.)
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
//! Mapping derived directly from spec Table 7-14 (B macroblock types) — the
//! bin-tree → mb_type translation is normative.

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
/// (B_8x8 = 22, with sub-MB tree) are handled by the decoder's
/// `walk_b_8x8` / `decode_sub_mb_type_b`.
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

/// Look up partition metadata for a B mb_type.
/// Returns `None` for non-partitioned mb_types (0..=3 = 16x16
/// family, 22 = B_8x8, 23 = intra-in-B). Spec Table 7-14.
pub const fn partitioned_b_meta(mb_type: u32) -> Option<BPartitionedMeta> {
    use BListUse::*;
    use BPartitionShape::*;
    let (shape, p0, p1) = match mb_type {
        4 => (H, L0, L0),
        5 => (V, L0, L0),
        6 => (H, L1, L1),
        7 => (V, L1, L1),
        8 => (H, L0, L1),
        9 => (V, L0, L1),
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
