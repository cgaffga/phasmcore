// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Forward integer transforms for Phase 6 H.264 encoder.
//!
//! Phase 6A.1 ships:
//!   - `forward_dct_4x4`       ITU-T H.264 § 8.5.8
//!   - `forward_hadamard_4x4`  § 8.5.10 (Intra_16x16 luma DC)
//!   - `forward_hadamard_2x2`  § 8.5.11.2 (chroma DC, 4:2:0)
//!
//! 8×8 DCT (High profile, § 8.5.13) is deferred to Phase 6C.
//!
//! All three are **pure** (no state, no I/O) and operate on
//! `[[i32; N]; N]` buffers. Outputs fit in `i16` for residuals
//! bounded by ±255, but we keep `i32` throughout so intermediate
//! stages cannot overflow under any input.
//!
//! Algorithm note:
//!   docs/design/h264-encoder-algorithms/transform.md

/// Forward 4×4 integer DCT (Phase 6A.1).
///
/// Implements the residual-domain part of H.264 § 8.5.8:
///   `Y = T · X · Tᵀ`
/// with T the H.264 4×4 basis matrix. The transform is performed as
/// two separable butterfly passes (rows then columns); no
/// multiplications are needed since all matrix entries are in {±1, ±2}.
///
/// Output is **unnormalized** — the row-norm-squared factors
/// {4, 10, 4, 10} are absorbed by the `normAdjust[qp%6][class]` table
/// in Phase 6A.2's quantizer.
///
/// Input must be the residual `source − prediction` for a 4×4 block,
/// with values in approximately ±255 (wider for pathological inputs,
/// but no real-world residual exceeds that range).
pub fn forward_dct_4x4(input: &[[i32; 4]; 4]) -> [[i32; 4]; 4] {
    // Stage 1 — 1-D transform along rows.
    let mut f = [[0i32; 4]; 4];
    for i in 0..4 {
        let a = input[i][0] + input[i][3];
        let b = input[i][1] + input[i][2];
        let c = input[i][1] - input[i][2];
        let d = input[i][0] - input[i][3];
        f[i][0] = a + b;
        f[i][1] = (d << 1) + c;
        f[i][2] = a - b;
        f[i][3] = d - (c << 1);
    }

    // Stage 2 — 1-D transform along columns of f.
    let mut y = [[0i32; 4]; 4];
    for j in 0..4 {
        let a = f[0][j] + f[3][j];
        let b = f[1][j] + f[2][j];
        let c = f[1][j] - f[2][j];
        let d = f[0][j] - f[3][j];
        y[0][j] = a + b;
        y[1][j] = (d << 1) + c;
        y[2][j] = a - b;
        y[3][j] = d - (c << 1);
    }
    y
}

/// Forward 4×4 Hadamard transform for Intra_16x16 luma DC (§ 8.5.10).
///
/// Applied to the 16 DC coefficients of the 4×4 AC blocks within a
/// single 16×16 Intra macroblock. All entries of the basis matrix are
/// ±1, so the transform is additions only. No scale factor — the
/// 2-D transform produces values 4× larger than a normalized Hadamard
/// would; Phase 6A.2's Hadamard-DC quantizer uses the `qp-2` shift
/// region to absorb this gain.
///
/// Basis matrix (symmetric, matching the inverse Hadamard in
/// `codec/h264/transform.rs::inverse_16x16_dc_hadamard`):
/// ```text
///   ⎡ 1   1   1   1 ⎤
///   ⎢ 1   1  -1  -1 ⎥
///   ⎢ 1  -1  -1   1 ⎥
///   ⎣ 1  -1   1  -1 ⎦
/// ```
/// Forward and inverse use the same butterfly — the matrix is its
/// own transpose, so `H · H = H · Hᵀ = 4 · I`.
pub fn forward_hadamard_4x4(input: &[[i32; 4]; 4]) -> [[i32; 4]; 4] {
    // Stage 1 — rows.
    let mut f = [[0i32; 4]; 4];
    for i in 0..4 {
        let a0 = input[i][0] + input[i][2];
        let a1 = input[i][1] + input[i][3];
        let a2 = input[i][0] - input[i][2];
        let a3 = input[i][1] - input[i][3];
        f[i][0] = a0 + a1;
        f[i][1] = a2 + a3;
        f[i][2] = a2 - a3;
        f[i][3] = a0 - a1;
    }

    // Stage 2 — columns.
    let mut y = [[0i32; 4]; 4];
    for j in 0..4 {
        let b0 = f[0][j] + f[2][j];
        let b1 = f[1][j] + f[3][j];
        let b2 = f[0][j] - f[2][j];
        let b3 = f[1][j] - f[3][j];
        y[0][j] = b0 + b1;
        y[1][j] = b2 + b3;
        y[2][j] = b2 - b3;
        y[3][j] = b0 - b1;
    }
    y
}

/// Forward 2×2 Hadamard for chroma DC (§ 8.5.11.2, 4:2:0 only).
///
/// Input: the 4 DC coefficients of the 4 chroma AC blocks for one
/// chroma component (Cb or Cr separately). Output: the 4 Hadamard-
/// transformed DC values that Phase 6A.4 emits in the slice header's
/// chroma-DC syntax group.
pub fn forward_hadamard_2x2(input: &[[i32; 2]; 2]) -> [[i32; 2]; 2] {
    // Row-then-column Hadamard.
    let r00 = input[0][0] + input[0][1];
    let r01 = input[0][0] - input[0][1];
    let r10 = input[1][0] + input[1][1];
    let r11 = input[1][0] - input[1][1];

    [
        [r00 + r10, r01 + r11],
        [r00 - r10, r01 - r11],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::transform::inverse_4x4_integer;

    // ─── forward_dct_4x4 ───────────────────────────────────────────

    #[test]
    fn dct_zero_input_zero_output() {
        let zero = [[0i32; 4]; 4];
        let y = forward_dct_4x4(&zero);
        for row in &y {
            assert_eq!(row, &[0, 0, 0, 0]);
        }
    }

    #[test]
    fn dct_flat_input_only_dc_nonzero() {
        // Constant residual — all AC coefficients must be zero; DC
        // equals N · N · constant where N = sum of basis-row entries
        // for row 0 (= 4). So DC coeff = 16 * constant.
        let flat = [[5i32; 4]; 4];
        let y = forward_dct_4x4(&flat);
        assert_eq!(y[0][0], 16 * 5);
        for i in 0..4 {
            for j in 0..4 {
                if (i, j) != (0, 0) {
                    assert_eq!(y[i][j], 0, "AC at ({i},{j}) must be zero for flat input");
                }
            }
        }
    }

    #[test]
    fn dct_linearity() {
        // f(a + b) == f(a) + f(b) exactly — the forward transform is
        // a pure linear map with no rounding.
        let a = [
            [10, -5, 22, -3],
            [7, 0, -11, 4],
            [-2, 18, -6, 1],
            [9, -14, 0, 12],
        ];
        let b = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ];
        let mut sum = [[0i32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                sum[i][j] = a[i][j] + b[i][j];
            }
        }

        let ya = forward_dct_4x4(&a);
        let yb = forward_dct_4x4(&b);
        let ysum = forward_dct_4x4(&sum);

        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    ysum[i][j],
                    ya[i][j] + yb[i][j],
                    "linearity at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn dct_delta_at_origin_is_outer_product_of_first_rows() {
        // A delta at (0, 0) — only x[0][0] = 1 — gives Y = T · E · Tᵀ
        // where E has a single 1 at (0, 0). Y[i][j] = T[i][0] * T[j][0],
        // and T[·][0] = [1, 2, 1, 1] ⇒ Y is the outer product of that
        // vector with itself.
        let mut x = [[0i32; 4]; 4];
        x[0][0] = 1;
        let y = forward_dct_4x4(&x);
        let t_col0 = [1, 2, 1, 1];
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    y[i][j],
                    t_col0[i] * t_col0[j],
                    "delta @ (0,0) at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn dct_inverse_approx_shape_recovered() {
        // The spec's forward T and inverse Tinv are NOT direct
        // inverses without the `normAdjust` quantization in between
        // — that's Phase 6A.2's job. Full forward+inverse round-trip
        // through quant lands in 6A.2's tests.
        //
        // Here we check a weaker property: `sign(forward · inverse of
        // constant input)` recovers `sign(input)` for most positions.
        // i.e. the transform preserves the general shape of the
        // residual without reversing any polarities on its own.
        let x = [
            [16, -16, 16, -16],
            [16, -16, 16, -16],
            [16, -16, 16, -16],
            [16, -16, 16, -16],
        ];
        let y = forward_dct_4x4(&x);
        let recovered = inverse_4x4_integer(&y);
        // For this input, every pixel has the same sign as before the
        // round-trip. (Magnitude differs because of the missing quant.)
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    recovered[i][j].signum(),
                    x[i][j].signum(),
                    "sign flip at ({i},{j}): {} → {}",
                    x[i][j],
                    recovered[i][j],
                );
            }
        }
    }

    // ─── forward_hadamard_4x4 ──────────────────────────────────────

    #[test]
    fn hadamard4_zero_input_zero_output() {
        let zero = [[0i32; 4]; 4];
        let y = forward_hadamard_4x4(&zero);
        for row in &y {
            assert_eq!(row, &[0, 0, 0, 0]);
        }
    }

    #[test]
    fn hadamard4_flat_input_only_top_left_nonzero() {
        // Flat input through a 4×4 Hadamard concentrates all energy
        // at (0, 0), scaled by 4² = 16.
        let flat = [[3i32; 4]; 4];
        let y = forward_hadamard_4x4(&flat);
        assert_eq!(y[0][0], 16 * 3);
        for i in 0..4 {
            for j in 0..4 {
                if (i, j) != (0, 0) {
                    assert_eq!(y[i][j], 0, "Hadamard AC at ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn hadamard4_self_inverse_up_to_scale16() {
        // H₄ · H₄ᵀ = 4 · I, so applying forward_hadamard_4x4 twice
        // scales each entry by 16.
        let x = [
            [1, -2, 3, -4],
            [5, 6, -7, 8],
            [-9, 10, 11, -12],
            [13, -14, -15, 16],
        ];
        let once = forward_hadamard_4x4(&x);
        let twice = forward_hadamard_4x4(&once);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    twice[i][j],
                    16 * x[i][j],
                    "H4² scale at ({i},{j})"
                );
            }
        }
    }

    // ─── forward_hadamard_2x2 ──────────────────────────────────────

    #[test]
    fn hadamard2_zero_input_zero_output() {
        let zero = [[0i32; 2]; 2];
        let y = forward_hadamard_2x2(&zero);
        assert_eq!(y, [[0, 0], [0, 0]]);
    }

    #[test]
    fn hadamard2_flat_input_only_top_left_nonzero() {
        let flat = [[7i32; 2]; 2];
        let y = forward_hadamard_2x2(&flat);
        // 2×2 Hadamard of flat = 4x the constant at (0,0), rest zero.
        assert_eq!(y, [[28, 0], [0, 0]]);
    }

    #[test]
    fn hadamard2_self_inverse_up_to_scale4() {
        // H₂ · H₂ᵀ = 2 · I, so forward twice scales by 4.
        let x = [[3, -5], [7, -2]];
        let once = forward_hadamard_2x2(&x);
        let twice = forward_hadamard_2x2(&once);
        assert_eq!(twice, [[4 * x[0][0], 4 * x[0][1]], [4 * x[1][0], 4 * x[1][1]]]);
    }

    #[test]
    fn hadamard2_known_delta() {
        // Delta at (0, 0) through H₂ · X · H₂ᵀ:
        //   Y[i][j] = H₂[i][0] · H₂[0][j] = 1 · 1 = 1 for all (i, j).
        let mut x = [[0i32; 2]; 2];
        x[0][0] = 1;
        let y = forward_hadamard_2x2(&x);
        assert_eq!(y, [[1, 1], [1, 1]]);
    }
}
