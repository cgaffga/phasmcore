// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Inter-mode encoding helpers. Phase 6B.3.
//!
//! Currently ships only what's needed for `P_L0_16x16` macroblocks:
//!  - `cbp_to_codenum_inter` — spec Table 9-4 column "Inter" for
//!    ChromaArrayType = 1 (4:2:0).
//!  - `encode_p16x16_mvd` — thin wrapper around `se()` for the
//!    two MVD components.
//!
//! Sub-MB partitions (P_16x8, P_8x16, P_8x8) are deferred to a
//! follow-up under the same Phase 6B.3 umbrella once the basic
//! P-frame pipeline is validated end-to-end.

/// Table 9-4 (a) Inter column — map from `codeNum` to `coded_block_
/// pattern` value for inter MBs in 4:2:0. We store it forward so
/// the decoder side can reuse; the encoder inverts at lookup time.
pub const INTER_CBP_CODENUM_TO_VALUE: [u8; 48] = [
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33,
    34, 36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

/// Given a `coded_block_pattern` value in [0, 47], return the
/// `codeNum` to emit via `ue(v)`. Returns `None` if the value isn't
/// in the table (shouldn't happen for valid CBPs).
pub fn cbp_to_codenum_inter(cbp: u8) -> Option<u32> {
    INTER_CBP_CODENUM_TO_VALUE
        .iter()
        .position(|&v| v == cbp)
        .map(|i| i as u32)
}

/// Given a `coded_block_pattern` value in [0, 47], return the
/// `codeNum` to emit for an intra MB (I_4x4 / I_8x8) via `ue(v)`.
/// Uses the intra column of spec Table 9-4 (a). Reverses the
/// parser's `macroblock::CBP_INTRA_TABLE`.
pub fn cbp_to_codenum_intra(cbp: u8) -> Option<u32> {
    crate::codec::h264::macroblock::CBP_INTRA_TABLE
        .iter()
        .position(|&v| v as u8 == cbp)
        .map(|i| i as u32)
}

/// Pack a CBP-luma (0 or 15 — bitmask of which 8×8 blocks have AC
/// coeffs) and CBP-chroma (0, 1, 2) into the 6-bit CBP value used in
/// the bitstream.
///
/// The standard packing is `(cbp_chroma << 4) | cbp_luma_8x8_mask`,
/// where `cbp_luma_8x8_mask` is a 4-bit bitmap of the 8×8 blocks
/// containing any nonzero AC coefficient (bit k ⇒ 8×8 block k).
pub fn pack_cbp(cbp_luma_8x8_mask: u8, cbp_chroma: u8) -> u8 {
    debug_assert!(cbp_luma_8x8_mask <= 0xF);
    debug_assert!(cbp_chroma <= 2);
    (cbp_chroma << 4) | cbp_luma_8x8_mask
}

/// Derive `cbp_luma_8x8_mask` from per-4×4-AC-block nonzero flags.
/// `nonzero_ac[k]` is `true` iff 4×4 block `k` (per `BLOCK_INDEX_TO_POS`
/// ordering) has any non-zero AC coefficient.
pub fn luma_8x8_cbp_mask(nonzero_ac: &[bool; 16]) -> u8 {
    let mut mask = 0u8;
    // 4 ×4 blocks 0-3 → 8×8 block 0 (top-left). 4-7 → 8×8 block 1
    // (top-right). 8-11 → 8×8 block 2 (bottom-left). 12-15 → 8×8
    // block 3 (bottom-right). Per spec § 6.4.3 / Figure 6-10.
    for (k, &nz) in nonzero_ac.iter().enumerate() {
        if nz {
            mask |= 1 << (k / 4);
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cbp_codenum_zero_maps_to_zero() {
        assert_eq!(cbp_to_codenum_inter(0), Some(0));
    }

    #[test]
    fn cbp_codenum_inter_full_table_known_values() {
        // From spec Table 9-4 (a) inter column:
        //   codeNum 1 ↔ CBP 16 (chroma DC only, no luma)
        //   codeNum 2 ↔ CBP 1
        //   codeNum 11 ↔ CBP 15
        //   codeNum 12 ↔ CBP 47
        assert_eq!(cbp_to_codenum_inter(16), Some(1));
        assert_eq!(cbp_to_codenum_inter(1), Some(2));
        assert_eq!(cbp_to_codenum_inter(15), Some(11));
        assert_eq!(cbp_to_codenum_inter(47), Some(12));
    }

    #[test]
    fn cbp_codenum_out_of_range() {
        assert_eq!(cbp_to_codenum_inter(48), None);
        assert_eq!(cbp_to_codenum_inter(255), None);
    }

    #[test]
    fn pack_cbp_basic() {
        assert_eq!(pack_cbp(0, 0), 0);
        assert_eq!(pack_cbp(15, 0), 15);
        assert_eq!(pack_cbp(0, 1), 16); // chroma DC only
        assert_eq!(pack_cbp(0, 2), 32); // chroma AC + DC
        assert_eq!(pack_cbp(15, 2), 47); // full
    }

    #[test]
    fn luma_8x8_mask_from_ac_flags() {
        // All zero → mask 0.
        assert_eq!(luma_8x8_cbp_mask(&[false; 16]), 0);
        // Only block 0 (= 4×4 blkIdx 0) nonzero → mask bit 0.
        let mut flags = [false; 16];
        flags[0] = true;
        assert_eq!(luma_8x8_cbp_mask(&flags), 0b0001);
        // Blocks 4-7 (8×8 block 1) have at least one nonzero → mask bit 1.
        let mut flags = [false; 16];
        flags[5] = true;
        assert_eq!(luma_8x8_cbp_mask(&flags), 0b0010);
        // All 16 AC blocks nonzero → full mask.
        assert_eq!(luma_8x8_cbp_mask(&[true; 16]), 0b1111);
    }
}
