// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 inverse quantisation and inverse 4×4 integer transform.
//!
//! Implements the scaling and transformation process for residual 4×4 blocks
//! from H.264 spec Section 8.5.12, plus the inverse 4×4 Hadamard used for the
//! Intra_16x16 DC block (Section 8.5.11) and the inverse 2×2 Hadamard used
//! for chroma DC in 4:2:0 (Section 8.5.11.2).
//!
//! Originally added for I-frame pixel reconstruction in the pure-Rust
//! UNIWARD-cost / decode path. After the 2026-06 video-retirement (the
//! pure-Rust H.264 encoder + CAVLC `reconstruct.rs` were deleted) these
//! have no production consumer: UNIWARD cost (`stego/cost/h264_uniward.rs`)
//! reimplements its own integer-transform basis inline, and the only
//! in-tree caller is a `#[cfg(test)]` reference-check in the disabled
//! `stego/dpb_correction.rs` — for which this module is the canonical
//! oracle. Retained as a spec-correct inverse-transform reference.
//!
//! The dead-code sweep re-confirmed KEEP: this pairs with
//! `dpb_correction.rs`, and deleting the pair is a 12-call-site
//! `encode_once` signature change (dropping the always-`None` `dpb_cover`
//! parameter) for provably-inert code — not worth the encoder churn. See
//! `docs/design/video/_RETIREMENT-PLAN.md` §8.

use super::tables::ZIGZAG_4X4;

/// `LevelScale4x4[qp%6][class]` — the spec's `LevelScale` table for
/// 4×4 blocks with flat default `weightScale` of 16. Equal to
/// `weightScale * normAdjust = 16 × normAdjust`.
///
/// Spec § 8.5.8 (Eq. 8-317): `LevelScale4x4(m, i, j) = weightScale ·
/// normAdjust(m, i, j)`. For Baseline / Main profiles with the
/// default scaling list (flat 16), weightScale = 16 uniformly, so
/// `LevelScale = 16 × normAdjust`. The entries below are the spec's
/// `v` matrix (§ 8.5.8 Eq. 8-319) multiplied by 16.
///
/// `class` is determined by coefficient position:
/// * 0 — positions (0,0), (0,2), (2,0), (2,2) (the 2×2 DC sub-lattice)
/// * 1 — positions (1,1), (1,3), (3,1), (3,3) (the 2×2 AC sub-lattice)
/// * 2 — all other positions
///
/// Phase 6A.11 renamed this table from `NORM_ADJUST_4X4` (raw
/// normAdjust) to `LEVEL_SCALE_4X4` (×16). The corresponding dequant
/// shifts are adjusted to match spec Eq. 8-325/-326/-330/-340/-341
/// directly.
const LEVEL_SCALE_4X4: [[i32; 3]; 6] = [
    [160, 256, 208],
    [176, 288, 224],
    [208, 320, 256],
    [224, 368, 288],
    [256, 400, 320],
    [288, 464, 368],
];

/// Classify a 4×4 coefficient position into one of the three LevelScale classes.
#[inline]
const fn norm_adjust_class(i: usize, j: usize) -> usize {
    let even_i = i & 1 == 0;
    let even_j = j & 1 == 0;
    if even_i && even_j {
        0
    } else if !even_i && !even_j {
        1
    } else {
        2
    }
}

/// Reorder 16 scan-order coefficients into a 4×4 raster grid using the H.264
/// zigzag scan (Table 8-13). Coefficients beyond the first `total_coeffs` are
/// zero.
pub fn unzigzag_4x4(scan_coeffs: &[i32; 16]) -> [[i32; 4]; 4] {
    let mut out = [[0i32; 4]; 4];
    for scan_idx in 0..16 {
        let raster = ZIGZAG_4X4[scan_idx] as usize;
        let i = raster / 4;
        let j = raster % 4;
        out[i][j] = scan_coeffs[scan_idx];
    }
    out
}

/// Scale a 4×4 block of quantised coefficients back to the transform domain
/// per H.264 Section 8.5.12.2.
///
/// `qp` is the effective luma QP for the block (already adjusted by any
/// `mb_qp_delta`). `has_dc_separate = true` when the block's DC coefficient
/// lives in a separate DC block (Intra_16x16 AC and chroma AC blocks) — in
/// that case the `(0,0)` position is left at zero because the real DC value
/// is supplied later by the Hadamard-transformed DC block.
pub fn dequant_4x4(
    raster: &[[i32; 4]; 4],
    qp: i32,
    has_dc_separate: bool,
) -> [[i32; 4]; 4] {
    debug_assert!((0..52).contains(&qp), "luma QP must be in [0,51], got {qp}");

    let q_mod = (qp.rem_euclid(6)) as usize;
    let q_bits = qp / 6;

    let mut d = [[0i32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            if has_dc_separate && i == 0 && j == 0 {
                continue;
            }
            // Spec § 8.5.12.1 Eq. 8-340 / 8-341 with LevelScale:
            //   qP ≥ 24: dij = (cij · LevelScale) << (qP/6 − 4)
            //   qP < 24: dij = (cij · LevelScale + 2^(3-qP/6)) >> (4-qP/6)
            let c = raster[i][j];
            let scale = LEVEL_SCALE_4X4[q_mod][norm_adjust_class(i, j)];
            d[i][j] = if q_bits >= 4 {
                (c * scale) << (q_bits - 4)
            } else {
                let shift = 4 - q_bits;
                let rnd = 1 << (shift - 1);
                (c * scale + rnd) >> shift
            };
        }
    }
    d
}

/// Inverse 4×4 integer transform per H.264 Section 8.5.12.1.
///
/// Input is a 4×4 block of dequantised coefficients `d`. Output is the 4×4
/// residual sample block `r`, not yet added to intra prediction.
///
/// Formula (shortcut notation):
/// 1. For each column: butterfly rows 0..3 into an intermediate f.
/// 2. For each row of f: same butterfly.
/// 3. `r[i][j] = (result + 32) >> 6`.
pub fn inverse_4x4_integer(d: &[[i32; 4]; 4]) -> [[i32; 4]; 4] {
    let mut g = [[0i32; 4]; 4];

    // Stage 1: 1-D inverse transform along columns.
    for j in 0..4 {
        let e0 = d[0][j] + d[2][j];
        let e1 = d[0][j] - d[2][j];
        let e2 = (d[1][j] >> 1) - d[3][j];
        let e3 = d[1][j] + (d[3][j] >> 1);

        g[0][j] = e0 + e3;
        g[1][j] = e1 + e2;
        g[2][j] = e1 - e2;
        g[3][j] = e0 - e3;
    }

    let mut h = [[0i32; 4]; 4];

    // Stage 2: 1-D inverse transform along rows.
    for i in 0..4 {
        let e0 = g[i][0] + g[i][2];
        let e1 = g[i][0] - g[i][2];
        let e2 = (g[i][1] >> 1) - g[i][3];
        let e3 = g[i][1] + (g[i][3] >> 1);

        h[i][0] = e0 + e3;
        h[i][1] = e1 + e2;
        h[i][2] = e1 - e2;
        h[i][3] = e0 - e3;
    }

    // Stage 3: rounding right-shift by 6.
    let mut r = [[0i32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            r[i][j] = (h[i][j] + 32) >> 6;
        }
    }
    r
}

/// Inverse 4×4 Hadamard transform and dequant for the Intra_16x16 luma
/// DC block per H.264 § 8.5.10.
///
/// Input `c` is the 16 DC coefficients arranged in a 4×4 grid. The
/// `(i, j)` position corresponds to the DC of the sub-block at
/// `luma4x4BlkIdx = k` where `i = k / 4, j = k % 4` (per spec § 6.4.3
/// block-index scan). Output `dcY[i][j]` is the reconstructed DC
/// value that gets injected as `d[0][0]` into the AC sub-block at
/// the same BlockIndex position before [`inverse_4x4_integer`].
///
/// Dequant per spec § 8.5.10 Eq. 8-325 / 8-326 (Phase 6A.11 —
/// unified to `LevelScale4x4`):
///   qP ≥ 36: dcY = (f · LevelScale4x4) << (qP/6 − 6)
///   qP < 36: dcY = (f · LevelScale4x4 + 2^(5-qP/6)) >> (6-qP/6)
pub fn inverse_16x16_dc_hadamard(c: &[[i32; 4]; 4], qp: i32) -> [[i32; 4]; 4] {
    debug_assert!((0..52).contains(&qp), "luma QP must be in [0,51], got {qp}");

    // Stage 1: 4×4 Hadamard (rows then cols). Hadamard matrix per
    // spec Eq. 8-324 is symmetric, so rows and cols use the same
    // butterfly.
    let mut f = [[0i32; 4]; 4];
    for i in 0..4 {
        let a0 = c[i][0] + c[i][2];
        let a1 = c[i][1] + c[i][3];
        let a2 = c[i][0] - c[i][2];
        let a3 = c[i][1] - c[i][3];

        f[i][0] = a0 + a1;
        f[i][1] = a2 + a3;
        f[i][2] = a2 - a3;
        f[i][3] = a0 - a1;
    }
    let mut g = [[0i32; 4]; 4];
    for j in 0..4 {
        let b0 = f[0][j] + f[2][j];
        let b1 = f[1][j] + f[3][j];
        let b2 = f[0][j] - f[2][j];
        let b3 = f[1][j] - f[3][j];

        g[0][j] = b0 + b1;
        g[1][j] = b2 + b3;
        g[2][j] = b2 - b3;
        g[3][j] = b0 - b1;
    }

    // Stage 2: scale per spec § 8.5.10 Eq. 8-325 / 8-326.
    let q_mod = qp.rem_euclid(6) as usize;
    let q_bits = qp / 6;
    let scale = LEVEL_SCALE_4X4[q_mod][0];

    let mut out = [[0i32; 4]; 4];
    if q_bits >= 6 {
        for i in 0..4 {
            for j in 0..4 {
                out[i][j] = (g[i][j] * scale) << (q_bits - 6);
            }
        }
    } else {
        let shift = 6 - q_bits;
        let rnd = 1 << (shift - 1);
        for i in 0..4 {
            for j in 0..4 {
                out[i][j] = (g[i][j] * scale + rnd) >> shift;
            }
        }
    }
    out
}

/// H.264 8-bit chroma QP table (spec Section 8.5.1 / Table 8-9).
/// Indexed by `qPI` = clip3(0, 51, qp_y + chroma_qp_offset).
const CHROMA_QP_TABLE_8BIT: [u8; 52] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39,
    39, 39,
];

/// Derive a chroma QP from a luma QP and the PPS-level `chroma_qp_offset`
/// (or `second_chroma_qp_offset` for Cr).
pub fn derive_chroma_qp(qp_y: i32, chroma_qp_offset: i32) -> i32 {
    let qpi = (qp_y + chroma_qp_offset).clamp(0, 51);
    CHROMA_QP_TABLE_8BIT[qpi as usize] as i32
}

/// Inverse 2×2 Hadamard transform and dequant for the chroma DC block in
/// 4:2:0 video, per H.264 Section 8.5.11.2.
///
/// `c` is the 4 DC coefficients of the chroma component (Cb OR Cr) in a 2×2
/// layout. Output entry `(i, j)` is the DC value to inject at `[0][0]` of
/// chroma AC block `i*2 + j` before running [`reconstruct_residual_4x4_with_dc`].
pub fn inverse_chroma_dc_2x2_hadamard(c: &[[i32; 2]; 2], qp_c: i32) -> [[i32; 2]; 2] {
    debug_assert!((0..52).contains(&qp_c), "chroma QP must be in [0,51], got {qp_c}");

    // 2×2 Hadamard: transform twice (rows then cols) with all ±1 signs.
    let f00 = c[0][0] + c[0][1];
    let f01 = c[0][0] - c[0][1];
    let f10 = c[1][0] + c[1][1];
    let f11 = c[1][0] - c[1][1];

    let g00 = f00 + f10;
    let g01 = f01 + f11;
    let g10 = f00 - f10;
    let g11 = f01 - f11;

    // Spec § 8.5.11.2 Eq. 8-330 for ChromaArrayType = 1 (4:2:0):
    //   dcC[i][j] = ((f[i][j] · LevelScale4x4(qP%6, 0, 0)) << (qP/6)) >> 5
    //
    // Regime-split equivalent for positive / negative net shift
    // (break point at q_bits = 5).
    let q_mod = qp_c.rem_euclid(6) as usize;
    let q_bits = qp_c / 6;
    let scale = LEVEL_SCALE_4X4[q_mod][0];

    let raw = [[g00, g01], [g10, g11]];
    let mut out = [[0i32; 2]; 2];
    if q_bits >= 5 {
        for i in 0..2 {
            for j in 0..2 {
                out[i][j] = (raw[i][j] * scale) << (q_bits - 5);
            }
        }
    } else {
        let shift = 5 - q_bits;
        for i in 0..2 {
            for j in 0..2 {
                out[i][j] = (raw[i][j] * scale) >> shift;
            }
        }
    }
    out
}

/// Combined dequant + inverse transform for an ordinary 4×4 residual block.
///
/// `scan_coeffs` is a 16-entry zigzag-scan-order coefficient array (the
/// format the retired CAVLC residual decode produced).
/// Returns a 4×4 residual sample block.
///
/// For Intra_16x16 AC blocks where the DC coefficient lives in a separate DC
/// block, pass `has_dc_separate = true` so the `(0,0)` position is left at
/// zero; the caller must then inject the DC value from
/// [`inverse_16x16_dc_hadamard`] before this function runs. The full
/// reconstruction helper [`reconstruct_residual_4x4_with_dc`] does this
/// wiring for you.
pub fn reconstruct_residual_4x4(scan_coeffs: &[i32; 16], qp: i32) -> [[i32; 4]; 4] {
    let raster = unzigzag_4x4(scan_coeffs);
    let dequant = dequant_4x4(&raster, qp, false);
    inverse_4x4_integer(&dequant)
}

/// As [`reconstruct_residual_4x4`] but injects a precomputed DC value at
/// position `(0,0)` after dequantisation. Used for Intra_16x16 AC blocks
/// where the DC was dequantised via the Hadamard path.
pub fn reconstruct_residual_4x4_with_dc(
    ac_scan_coeffs: &[i32; 16],
    dc_value: i32,
    qp: i32,
) -> [[i32; 4]; 4] {
    let raster = unzigzag_4x4(ac_scan_coeffs);
    let mut dequant = dequant_4x4(&raster, qp, true);
    dequant[0][0] = dc_value;
    inverse_4x4_integer(&dequant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_adjust_class_covers_all_positions() {
        // Spec Table 7-7: class 0 = corners of the 2x2 DC pattern;
        // class 1 = interior odd-odd positions; class 2 = the rest.
        assert_eq!(norm_adjust_class(0, 0), 0);
        assert_eq!(norm_adjust_class(0, 2), 0);
        assert_eq!(norm_adjust_class(2, 0), 0);
        assert_eq!(norm_adjust_class(2, 2), 0);
        assert_eq!(norm_adjust_class(1, 1), 1);
        assert_eq!(norm_adjust_class(1, 3), 1);
        assert_eq!(norm_adjust_class(3, 1), 1);
        assert_eq!(norm_adjust_class(3, 3), 1);
        // Spot-check a few class-2 positions.
        assert_eq!(norm_adjust_class(0, 1), 2);
        assert_eq!(norm_adjust_class(1, 0), 2);
        assert_eq!(norm_adjust_class(2, 1), 2);
        assert_eq!(norm_adjust_class(3, 2), 2);
    }

    #[test]
    fn zero_block_reconstructs_to_zero_residual() {
        let zero = [0i32; 16];
        for qp in [0, 12, 24, 36, 51] {
            let r = reconstruct_residual_4x4(&zero, qp);
            for row in &r {
                for &v in row {
                    assert_eq!(v, 0, "qp={qp} residual must be zero");
                }
            }
        }
    }

    #[test]
    fn dc_only_block_reconstructs_to_flat_residual() {
        // A non-zero DC coefficient should produce an (approximately) flat
        // positive residual after dequant + inverse transform.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 4; // DC in scan position 0
        for qp in [24, 30, 36, 40] {
            let r = reconstruct_residual_4x4(&coeffs, qp);
            let v00 = r[0][0];
            assert!(v00 != 0, "qp={qp} dc-only residual should be non-zero");
            // All pixels should have the same (or very close) value for a flat DC.
            for row in &r {
                for &v in row {
                    assert_eq!(v, v00, "qp={qp} flat DC residual must be uniform");
                }
            }
        }
    }

    #[test]
    fn inverse_transform_is_linear() {
        // Linearity: transform(a + b) == transform(a) + transform(b).
        let a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]];
        let b = [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]];
        let mut sum = [[0i32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                sum[i][j] = a[i][j] + b[i][j];
            }
        }

        let ta = inverse_4x4_integer(&a);
        let tb = inverse_4x4_integer(&b);
        let tsum = inverse_4x4_integer(&sum);

        for i in 0..4 {
            for j in 0..4 {
                // Linearity holds up to the final (+32)>>6 rounding: allow
                // a ±1 slack per position.
                let combined = ta[i][j] + tb[i][j];
                let diff = (tsum[i][j] - combined).abs();
                assert!(
                    diff <= 1,
                    "linearity violated at ({i},{j}): sum={} a+b={} diff={}",
                    tsum[i][j],
                    combined,
                    diff
                );
            }
        }
    }

    #[test]
    fn unzigzag_roundtrips_through_known_scan() {
        let scan: [i32; 16] = [
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        ];
        let raster = unzigzag_4x4(&scan);
        // Spec Table 8-13 scan order: 0,1,4,8,5,2,3,6,9,12,13,10,7,11,14,15
        // So raster position 0 = scan position 0 (10).
        assert_eq!(raster[0][0], 10); // raster 0, scan 0
        assert_eq!(raster[0][1], 11); // raster 1, scan 1
        assert_eq!(raster[1][0], 12); // raster 4, scan 2
        assert_eq!(raster[2][0], 13); // raster 8, scan 3
        assert_eq!(raster[1][1], 14); // raster 5, scan 4
        assert_eq!(raster[3][3], 25); // raster 15, scan 15
    }

    #[test]
    fn chroma_qp_table_matches_spec_fixed_points() {
        // H.264 spec Table 8-9 (chroma QP derivation) for 8-bit depth.
        // First 30 entries are identity, then the saturation region kicks in.
        assert_eq!(derive_chroma_qp(0, 0), 0);
        assert_eq!(derive_chroma_qp(29, 0), 29);
        assert_eq!(derive_chroma_qp(30, 0), 29);
        assert_eq!(derive_chroma_qp(31, 0), 30);
        assert_eq!(derive_chroma_qp(34, 0), 32);
        assert_eq!(derive_chroma_qp(51, 0), 39);

        // Offset clipping: qp + offset saturates at the 0..=51 range.
        assert_eq!(derive_chroma_qp(50, 5), 39); // 55 clipped to 51 -> 39
        assert_eq!(derive_chroma_qp(5, -10), 0); // -5 clipped to 0 -> 0
    }

    #[test]
    fn chroma_dc_hadamard_preserves_flat_signal() {
        // A 2×2 DC block with all entries equal feeds each Hadamard output
        // position one of {4x, 0, 0, 0} — only the "sum of sums" position
        // survives. After dequant the others stay zero.
        let c = [[3, 3], [3, 3]];
        let out = inverse_chroma_dc_2x2_hadamard(&c, 26);
        assert_ne!(out[0][0], 0, "sum-of-sums entry should be non-zero");
        assert_eq!(out[0][1], 0, "diff entries should be zero for flat DC");
        assert_eq!(out[1][0], 0);
        assert_eq!(out[1][1], 0);
    }

    #[test]
    fn chroma_dc_hadamard_zero_input_is_zero() {
        let c = [[0; 2]; 2];
        for qp in [0, 12, 26, 40, 51] {
            let out = inverse_chroma_dc_2x2_hadamard(&c, qp);
            for row in &out {
                for &v in row {
                    assert_eq!(v, 0, "qp={qp}");
                }
            }
        }
    }

    #[test]
    fn dequant_respects_qp_shift_regions() {
        // For qp = 30 (qp/6 = 5, qp%6 = 0), the scale should left-shift by 1.
        let mut raster = [[0i32; 4]; 4];
        raster[0][0] = 1;
        let d = dequant_4x4(&raster, 30, false);
        // q_bits = 5, scale at (0,0) with class 0 is LEVEL_SCALE_4X4[0][0] = 160.
        // Expected: (1 * 160) << (5 - 4) = 320.
        assert_eq!(d[0][0], 320);

        // For qp = 12 (qp/6 = 2, qp%6 = 0), shift is (4-2)=2 with rounding.
        let d = dequant_4x4(&raster, 12, false);
        // (1 * 160 + (1 << 1)) >> 2 = 162 >> 2 = 40.
        assert_eq!(d[0][0], 40);
    }
}
