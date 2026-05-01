// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Pack / unpack helpers for the I_16x16 mb_type encoding per spec
//! § 7.3.5 + Table 7-11.
//!
//! Phase 6F.1 follow-on tidy (Task #50, deferred-item #36) — the
//! forward and inverse formulas were duplicated across
//! `cabac::encoder` (forward) and `cabac::bin_decoder::slice`
//! (inverse). They now live here as a single paired pair so any
//! future spec-table change touches one place.
//!
//! The encoding packs three I_16x16 macroblock-type fields into one
//! `mb_type` value in the range 1..=24:
//!
//! - `luma_pred_mode` ∈ 0..=3 (Vertical / Horizontal / DC / Plane)
//! - `cbp_chroma` ∈ 0..=2 (Coded Block Pattern for chroma:
//!   0 = none, 1 = DC, 2 = AC + DC)
//! - `cbp_luma_flag` ∈ {0, 1} (any-luma-coefficient indicator)
//!
//! The packed `mb_type` formula (spec Table 7-11):
//!   `mb_type = 1 + luma_pred_mode + 4*cbp_chroma + 12*cbp_luma_flag`

/// I_16x16 mb_type field tuple — output of [`unpack_i_16x16_mb_type`]
/// and input of [`pack_i_16x16_mb_type`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I16x16MbType {
    /// 0..=3 (Vertical / Horizontal / DC / Plane).
    pub luma_pred_mode: u32,
    /// 0..=2 (none / DC / AC+DC).
    pub cbp_chroma: u32,
    /// 0 or 1 — any non-zero luma coefficient in the MB.
    pub cbp_luma_flag: u32,
}

/// Pack the three I_16x16 fields into a single `mb_type` value.
/// Result is in the range `1..=24` per spec Table 7-11.
#[inline]
pub fn pack_i_16x16_mb_type(fields: I16x16MbType) -> u32 {
    1 + fields.luma_pred_mode + 4 * fields.cbp_chroma + 12 * fields.cbp_luma_flag
}

/// Unpack `mb_type` (spec Table 7-11 row index) into the three
/// I_16x16 fields. Inverse of [`pack_i_16x16_mb_type`]; preserves
/// `pack(unpack(x)) == x` for valid `x ∈ 1..=24`.
#[inline]
pub fn unpack_i_16x16_mb_type(mb_type: u32) -> I16x16MbType {
    let v = mb_type - 1;
    I16x16MbType {
        luma_pred_mode: v % 4,
        cbp_chroma: (v / 4) % 3,
        cbp_luma_flag: v / 12,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip_full_range() {
        // Spec Table 7-11 covers mb_type 1..=24 for I_16x16.
        for mb_type in 1u32..=24 {
            let fields = unpack_i_16x16_mb_type(mb_type);
            assert_eq!(
                pack_i_16x16_mb_type(fields), mb_type,
                "round-trip break at mb_type={mb_type} ({fields:?})",
            );
        }
    }

    #[test]
    fn pack_known_examples() {
        // Spec sanity check: I_16x16_0_0_0 = mb_type 1 (luma=0, cbp_chroma=0, cbp_luma=0).
        assert_eq!(pack_i_16x16_mb_type(I16x16MbType {
            luma_pred_mode: 0, cbp_chroma: 0, cbp_luma_flag: 0,
        }), 1);
        // I_16x16_3_2_1 = 1 + 3 + 4*2 + 12*1 = 24 (the maximum I_16x16 mb_type).
        assert_eq!(pack_i_16x16_mb_type(I16x16MbType {
            luma_pred_mode: 3, cbp_chroma: 2, cbp_luma_flag: 1,
        }), 24);
    }
}
