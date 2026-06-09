// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 CABAC I-slice `mb_type` bin table (spec Table 9-36).
//!
//! This is the only surviving binarization artifact: the pure-Rust
//! CABAC bin-walker (`cabac::bin_decoder`) inverts this table to decode
//! `mb_type` — see `match_mb_type_i_bins` in `bin_decoder/syntax.rs`.
//! The forward CABAC binarizers that once consumed it (the Unary / TU / EGk
//! / UEGk / FL primitives + the P/B `mb_type` and `sub_mb_type` tables) were
//! deleted in the 2026-06 video-retirement — OpenH264 now owns the emit side.
//!
//! Algorithm note:
//!   `docs/design/video/_archive/h264/encoder-algorithms/binarization.md`.

// ─── Table-driven: mb_type (Table 9-36 I-slice) ─────────────────

/// Bin string for an I-slice mb_type value (0..=25). Spec Table 9-36.
/// Returns a slice of bins. Bin 1 of I_PCM (mb_type=25) is
/// terminate-coded (spec § 9.3.3.2.4 DecodeTerminate, the same path as
/// `end_of_slice_flag`), not regular context-coded — the decode engine
/// handles that bin via its terminate path, not the usual ctxIdx decode.
pub fn mb_type_i_bins(mb_type: u32) -> &'static [u8] {
    debug_assert!(mb_type <= 25, "I-slice mb_type {mb_type} out of range");
    MB_TYPE_I_BINS[mb_type as usize]
}

/// Full Table 9-36 — I-slice mb_type → bin string.
const MB_TYPE_I_BINS: &[&[u8]] = &[
    // 0: I_NxN
    &[0],
    // 1..4: I_16x16_<mode>_0_0 — 6 bins
    &[1, 0, 0, 0, 0, 0],
    &[1, 0, 0, 0, 0, 1],
    &[1, 0, 0, 0, 1, 0],
    &[1, 0, 0, 0, 1, 1],
    // 5..12: I_16x16_<mode>_1_0 / _2_0 — 7 bins
    &[1, 0, 0, 1, 0, 0, 0],
    &[1, 0, 0, 1, 0, 0, 1],
    &[1, 0, 0, 1, 0, 1, 0],
    &[1, 0, 0, 1, 0, 1, 1],
    &[1, 0, 0, 1, 1, 0, 0],
    &[1, 0, 0, 1, 1, 0, 1],
    &[1, 0, 0, 1, 1, 1, 0],
    &[1, 0, 0, 1, 1, 1, 1],
    // 13..16: I_16x16_<mode>_0_1 — 6 bins
    &[1, 0, 1, 0, 0, 0],
    &[1, 0, 1, 0, 0, 1],
    &[1, 0, 1, 0, 1, 0],
    &[1, 0, 1, 0, 1, 1],
    // 17..24: I_16x16_<mode>_1_1 / _2_1 — 7 bins
    &[1, 0, 1, 1, 0, 0, 0],
    &[1, 0, 1, 1, 0, 0, 1],
    &[1, 0, 1, 1, 0, 1, 0],
    &[1, 0, 1, 1, 0, 1, 1],
    &[1, 0, 1, 1, 1, 0, 0],
    &[1, 0, 1, 1, 1, 0, 1],
    &[1, 0, 1, 1, 1, 1, 0],
    &[1, 0, 1, 1, 1, 1, 1],
    // 25: I_PCM — 2 bins; bin 1 is terminate-coded.
    &[1, 1],
];

#[cfg(test)]
mod tests {
    use super::*;
    // ─── mb_type tables ─────────────────────────────────────────

    #[test]
    fn mb_type_i_all_values_have_bins() {
        for v in 0..=25 {
            let bins = mb_type_i_bins(v);
            assert!(!bins.is_empty(), "mb_type_i_bins({v}) empty");
        }
    }

    #[test]
    fn mb_type_i_spec_fixed_points() {
        assert_eq!(mb_type_i_bins(0), &[0][..]);
        assert_eq!(mb_type_i_bins(1), &[1, 0, 0, 0, 0, 0][..]);
        assert_eq!(mb_type_i_bins(2), &[1, 0, 0, 0, 0, 1][..]);
        assert_eq!(mb_type_i_bins(24), &[1, 0, 1, 1, 1, 1, 1][..]);
        assert_eq!(mb_type_i_bins(25), &[1, 1][..]);
    }
}
