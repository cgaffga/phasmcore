// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! J-UNIWARD cost function adapted for H.264 4×4 integer transform blocks.
//!
//! This is the 4×4 analogue of [`crate::stego::cost::uniward`] (which targets
//! JPEG 8×8 DCT blocks). The algorithm is unchanged — decompress to pixels,
//! compute a Daubechies-8 wavelet decomposition, and score each candidate
//! flip by how much it perturbs the wavelet coefficients relative to the
//! cover's wavelet magnitudes — but the constants change:
//!
//! * **Block size** 8 → 4
//! * **Impact window** 23 → 19 (= 4 + 16 − 1)
//! * **Basis** JPEG float DCT → H.264 integer inverse transform
//!
//! Phase 1b routes I-frame positions through this cost. P-frame positions
//! stay on the Phase 1a CSF cost because reconstructing P-frame pixels needs
//! motion compensation (deferred to Phase 7).

use crate::codec::h264::cavlc::{EmbedDomain, EmbeddablePosition};
use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;
use crate::codec::h264::tables::ZIGZAG_4X4;

/// 2^(q_bits - 4) for q_bits ∈ [0, 8] (i.e. exponent ∈ [-4, 4]).
/// Powers of 2 are exact f64 — bit-identical across iOS / Android /
/// x86_64 / WASM. Replaces `2f64.powi(q_bits - 4)` which lowers to
/// `@llvm.powi.f64` on WASM (non-deterministic).
const POW2_QBITS_MINUS_4: [f64; 9] = [
    0.0625, // q_bits = 0 → 2^-4
    0.125,  // q_bits = 1 → 2^-3
    0.25,   // q_bits = 2 → 2^-2
    0.5,    // q_bits = 3 → 2^-1
    1.0,    // q_bits = 4 → 2^0
    2.0,    // q_bits = 5 → 2^1
    4.0,    // q_bits = 6 → 2^2
    8.0,    // q_bits = 7 → 2^3
    16.0,   // q_bits = 8 → 2^4
];

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Stabilisation constant σ from the UNIWARD paper (avoids div-by-zero and
/// controls sensitivity to cover texture). Same value as JPEG J-UNIWARD.
const SIGMA: f64 = 0.015625; // 2^-6

/// Daubechies-8 high-pass decomposition filter (16 taps). Identical to the
/// JPEG uniward module's HPDF.
const HPDF: [f64; 16] = [
    -0.0544158422,
    0.3128715909,
    -0.6756307363,
    0.5853546837,
    0.0158291053,
    -0.2840155430,
    -0.0004724846,
    0.1287474266,
    0.0173693010,
    -0.0440882539,
    -0.0139810279,
    0.0087460940,
    0.0048703530,
    -0.0003917404,
    -0.0006754494,
    -0.0001174768,
];

/// Daubechies-8 low-pass filter (derived from HPDF via the QMF relation).
fn lpdf() -> [f64; 16] {
    let mut lp = [0.0f64; 16];
    for n in 0..16 {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        lp[n] = sign * HPDF[15 - n];
    }
    lp
}

const FILT_LEN: usize = 16;

/// Size of the wavelet-domain impact window produced by a single 4×4
/// coefficient flip: 4 + 16 − 1 = 19.
const IMPACT_SIZE: usize = 4 + FILT_LEN - 1;

/// Three wavelet subbands (LH / HL / HH) of the cover Y plane. Low-pass row
/// + high-pass column (and permutations) give the three directional bands
/// used by UNIWARD.
pub struct ThreeSubbands {
    pub lh: Vec<f32>,
    pub hl: Vec<f32>,
    pub hh: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// X offset of the subband buffer relative to the input image origin.
    pub x_offset: isize,
    /// Y offset of the subband buffer relative to the input image origin.
    pub y_offset: isize,
}

/// Compute the three Daubechies-8 directional subbands of a Y plane.
///
/// Uses symmetric reflection padding at the image borders. Output buffers
/// are sized `(width + 2·pad) × (height + 2·pad)` where `pad = FILT_LEN − 1 = 15`
/// so that a flip's 19×19 impact window never runs off the edges for any
/// valid block position.
pub fn compute_three_subbands(y_plane: &[u8], width: usize, height: usize) -> ThreeSubbands {
    let pad = FILT_LEN - 1; // 15
    let padded_w = width + 2 * pad;
    let padded_h = height + 2 * pad;

    // Row-filter the Y plane: produce two buffers (low-pass and high-pass
    // along rows), each padded by `pad` on top and bottom (column direction
    // not yet filtered, so just pass through for now on the row axis).
    let mut row_low = vec![0.0f32; padded_w * height];
    let mut row_high = vec![0.0f32; padded_w * height];
    let lp = lpdf();

    for y in 0..height {
        for out_x in 0..padded_w {
            let mut sum_low = 0.0f64;
            let mut sum_high = 0.0f64;
            for k in 0..FILT_LEN {
                let _src_x = out_x as isize + k as isize - (2 * pad as isize) + pad as isize;
                // src_x = out_x - pad + k  (simplified)
                let src_x = out_x as isize - pad as isize + k as isize;
                let clamped = symmetric_reflect(src_x, width as isize);
                let v = y_plane[y * width + clamped as usize] as f64;
                sum_low += lp[k] * v;
                sum_high += HPDF[k] * v;
            }
            row_low[y * padded_w + out_x] = sum_low as f32;
            row_high[y * padded_w + out_x] = sum_high as f32;
        }
    }

    // Column-filter the row-filtered buffers to produce LH, HL, HH subbands.
    // Each subband is `padded_w × padded_h` with `pad` reflection on all four sides.
    let mut lh = vec![0.0f32; padded_w * padded_h];
    let mut hl = vec![0.0f32; padded_w * padded_h];
    let mut hh = vec![0.0f32; padded_w * padded_h];

    for out_y in 0..padded_h {
        for x in 0..padded_w {
            let mut sum_lh = 0.0f64; // row_low  → high-pass col
            let mut sum_hl = 0.0f64; // row_high → low-pass col
            let mut sum_hh = 0.0f64; // row_high → high-pass col
            for k in 0..FILT_LEN {
                let src_y = out_y as isize - pad as isize + k as isize;
                let clamped = symmetric_reflect(src_y, height as isize);
                let low_val = row_low[clamped as usize * padded_w + x] as f64;
                let high_val = row_high[clamped as usize * padded_w + x] as f64;
                sum_lh += HPDF[k] * low_val;
                sum_hl += lp[k] * high_val;
                sum_hh += HPDF[k] * high_val;
            }
            lh[out_y * padded_w + x] = sum_lh as f32;
            hl[out_y * padded_w + x] = sum_hl as f32;
            hh[out_y * padded_w + x] = sum_hh as f32;
        }
    }

    ThreeSubbands {
        lh,
        hl,
        hh,
        width: padded_w,
        height: padded_h,
        x_offset: -(pad as isize),
        y_offset: -(pad as isize),
    }
}

#[inline]
pub(crate) fn symmetric_reflect(i: isize, len: isize) -> isize {
    if len <= 0 {
        return 0;
    }
    let mut v = i;
    while v < 0 || v >= len {
        if v < 0 {
            v = -v - 1;
        }
        if v >= len {
            v = 2 * len - v - 1;
        }
    }
    v
}

/// Pre-computed pixel-domain basis for a unit flip at each 4×4 coefficient
/// position. `basis_unit[u][v]` is the pattern the inverse H.264 integer
/// transform produces from a value of 1 at position `(u, v)`, BEFORE dequant
/// scaling and BEFORE the final `(+32) >> 6` rounding.
///
/// The real pixel-domain delta for a coefficient flip is
/// `Δ_coeff × scale(qp,u,v) / 64 × basis_unit[u][v]`. We fold the scale and
/// `/64` in at cost time in [`compute_position_cost`].
pub(crate) fn precompute_unit_basis() -> [[[[f64; 4]; 4]; 4]; 4] {
    // For each (u, v), run the inverse transform on a matrix with a 1 at
    // (u, v) and zeros elsewhere, using exact integer arithmetic but keeping
    // the result as f64 (no rounding).
    let mut out = [[[[0.0f64; 4]; 4]; 4]; 4];
    for u in 0..4 {
        for v in 0..4 {
            let mut d = [[0i32; 4]; 4];
            d[u][v] = 64; // keep the scale up so integer butterfly doesn't lose precision
            // Column butterfly:
            let mut g = [[0i32; 4]; 4];
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
            // Row butterfly:
            let mut h = [[0i32; 4]; 4];
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
            // Store as f64 divided by 64 (we fed 64 in) → unit response.
            for i in 0..4 {
                for j in 0..4 {
                    out[u][v][i][j] = h[i][j] as f64 / 64.0;
                }
            }
        }
    }
    out
}

/// `normAdjust4x4[qp%6][class]` — H.264 Table 7-7 scaling weights.
/// class: 0 = even-even, 1 = odd-odd, 2 = mixed-parity.
const NORM_ADJUST_4X4: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

#[inline]
const fn norm_adjust_class(u: usize, v: usize) -> usize {
    let even_u = u & 1 == 0;
    let even_v = v & 1 == 0;
    if even_u && even_v {
        0
    } else if !even_u && !even_v {
        1
    } else {
        2
    }
}

/// Effective dequant + IDCT scale factor in pixel units, for a unit
/// coefficient change at `(u, v)` under luma QP `qp`. Matches the pixel-
/// domain delta per unit coefficient change; negative/positive sign does
/// not matter for UNIWARD (it uses absolute value).
#[inline]
fn pixel_scale(qp: i32, u: usize, v: usize) -> f64 {
    let q_mod = qp.rem_euclid(6) as usize;
    let q_bits = qp / 6;
    let s = NORM_ADJUST_4X4[q_mod][norm_adjust_class(u, v)] as f64;
    // The unit basis has already been normalised to 1/64 of the butterfly
    // output; dequant multiplies by `s * 2^(q_bits - 4)`. The final residual
    // was `(butterfly + 32) >> 6`, which we approximate by dividing by 64
    // in the basis pre-compute, so the net scale here is just the dequant.
    // Lookup is bit-exact across platforms (powers of 2 are exact f64).
    s * POW2_QBITS_MINUS_4[q_bits as usize]
}

/// Compute J-UNIWARD cost for a single candidate flip in an I-frame.
///
/// `block_px_x`, `block_px_y` — pixel coordinates of the top-left corner of
/// the 4×4 block the flip lives in (block-aligned, within the Y plane).
/// `scan_pos` — position within the 4×4 block in zigzag order (0..=15).
/// `delta_magnitude` — absolute value of the coefficient change:
///    * T1 sign flip: 2 (coefficient goes +1 ↔ −1)
///    * LevelSuffixMag flip: 1 (LSB of magnitude toggles)
pub(crate) fn compute_position_cost(
    unit_basis: &[[[[f64; 4]; 4]; 4]; 4],
    wavelets: &ThreeSubbands,
    img_w: usize,
    img_h: usize,
    block_px_x: usize,
    block_px_y: usize,
    scan_pos: u8,
    qp: i32,
    delta_magnitude: f64,
) -> f64 {
    let raster = ZIGZAG_4X4[scan_pos as usize] as usize;
    let u = raster / 4;
    let v = raster % 4;

    let scale = pixel_scale(qp, u, v) * delta_magnitude;

    // Basis block at this (u, v) position, scaled for the actual coefficient delta.
    let mut basis = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            basis[i][j] = unit_basis[u][v][i][j] * scale;
        }
    }

    // Row-filter the 4 rows of the basis block into a 4-row × IMPACT_SIZE-col buffer.
    let mut row_low = [[0.0f64; IMPACT_SIZE]; 4];
    let mut row_high = [[0.0f64; IMPACT_SIZE]; 4];
    let lp = lpdf();
    for r in 0..4 {
        for out_c in 0..IMPACT_SIZE {
            let mut sum_low = 0.0;
            let mut sum_high = 0.0;
            for k in 0..FILT_LEN {
                // Output column out_c corresponds to block-relative input
                // column src = out_c - (FILT_LEN - 1) + k = out_c - 15 + k.
                let src = out_c as isize - (FILT_LEN - 1) as isize + k as isize;
                if (0..4).contains(&src) {
                    let v = basis[r][src as usize];
                    sum_low += lp[k] * v;
                    sum_high += HPDF[k] * v;
                }
            }
            row_low[r][out_c] = sum_low;
            row_high[r][out_c] = sum_high;
        }
    }

    // Column-filter into 3 directional subbands and accumulate cost against
    // the cover wavelets.
    let pad = FILT_LEN - 1; // 15
    let mut cost = 0.0f64;
    for out_r in 0..IMPACT_SIZE {
        for out_c in 0..IMPACT_SIZE {
            let mut delta_lh = 0.0;
            let mut delta_hl = 0.0;
            let mut delta_hh = 0.0;
            for k in 0..FILT_LEN {
                let src_r = out_r as isize - (FILT_LEN - 1) as isize + k as isize;
                if (0..4).contains(&src_r) {
                    let r = src_r as usize;
                    let low_val = row_low[r][out_c];
                    let high_val = row_high[r][out_c];
                    delta_lh += HPDF[k] * low_val;
                    delta_hl += lp[k] * high_val;
                    delta_hh += HPDF[k] * high_val;
                }
            }

            let abs_x = block_px_x as isize + out_c as isize - pad as isize;
            let abs_y = block_px_y as isize + out_r as isize - pad as isize;
            if abs_x < 0 || abs_y < 0 || abs_x >= img_w as isize || abs_y >= img_h as isize {
                continue;
            }

            let wx = (abs_x - wavelets.x_offset) as usize;
            let wy = (abs_y - wavelets.y_offset) as usize;
            let idx = wy * wavelets.width + wx;

            let w_lh = wavelets.lh[idx].abs() as f64;
            let w_hl = wavelets.hl[idx].abs() as f64;
            let w_hh = wavelets.hh[idx].abs() as f64;

            cost += delta_lh.abs() / (w_lh + SIGMA);
            cost += delta_hl.abs() / (w_hl + SIGMA);
            cost += delta_hh.abs() / (w_hh + SIGMA);
        }
    }
    cost
}

/// Reconstructed I-frame planes fed into the UNIWARD cost. Produced by
/// `codec::h264::reconstruct::reconstruct_i_frame_planes`. Cb/Cr use the
/// 4:2:0 layout (W/2 × H/2).
pub struct FramePlanes<'a> {
    pub y: &'a [u8],
    pub cb: &'a [u8],
    pub cr: &'a [u8],
    pub width: usize,
    pub height: usize,
}

/// Compute J-UNIWARD costs for a slice of I-frame positions over all three
/// planes (Y, Cb, Cr) at 4:2:0 subsampling.
///
/// Position routing via [`FramePosition::within_mb_block_idx`] (0..=25):
/// * 0..=15 → luma wavelets, 4×4 block at `(mb_x*16 + bx*4, mb_y*16 + by*4)`
///   using `qps[mb_idx]` and the `BLOCK_INDEX_TO_POS` ordering.
/// * 16..=17 → chroma DC (Cb=16, Cr=17). Returns `f32::INFINITY` — the 2×2
///   Hadamard DC has only three scan positions > 0 per block, not worth
///   the code complexity and they cascade across all 4 AC blocks anyway.
/// * 18..=21 → Cb wavelets, 4×4 block at `(mb_x*8 + bx*4, mb_y*8 + by*4)`
///   where `(bx, by) = ((slot-18) % 2, (slot-18) / 2)`, using `qp_cb`.
/// * 22..=25 → Cr wavelets, same shape with `qp_cr`.
///
/// DC positions (`scan_pos == 0`) in any plane are also returned as INF.
pub fn compute_frame_uniward_costs(
    planes: &FramePlanes,
    frame_positions: &[FramePosition],
    qps: &[i32],
) -> Vec<f32> {
    let width = planes.width;
    let height = planes.height;
    let chroma_w = width / 2;
    let chroma_h = height / 2;

    // Wavelet decomp of each plane (one-off per frame).
    let y_wavelets = compute_three_subbands(planes.y, width, height);
    let cb_wavelets = compute_three_subbands(planes.cb, chroma_w, chroma_h);
    let cr_wavelets = compute_three_subbands(planes.cr, chroma_w, chroma_h);
    let unit_basis = precompute_unit_basis();
    let width_in_mbs = width / 16;

    let compute_one = |fp: &FramePosition| -> f32 {
        let pos = fp.pos;
        // Phase 3a: MVD positions have no wavelet interpretation. Defensive
        // guard — in practice the pipeline doesn't feed MVD positions here
        // because UNIWARD runs I-slice only and MVDs are P-slice only.
        if pos.domain == EmbedDomain::MvdLsb {
            return f32::INFINITY;
        }
        if pos.scan_pos == 0 {
            return f32::INFINITY;
        }
        let within_mb = fp.within_mb_block_idx;
        let mb_x = fp.mb_idx % width_in_mbs;
        let mb_y = fp.mb_idx / width_in_mbs;

        // Pick plane, block-pixel offset, and QP based on the slot.
        let (wavelets, img_w, img_h, block_px_x, block_px_y, qp) = if within_mb < 16 {
            let (bx, by) = BLOCK_INDEX_TO_POS[within_mb];
            (
                &y_wavelets,
                width,
                height,
                mb_x * 16 + bx as usize * 4,
                mb_y * 16 + by as usize * 4,
                qps.get(fp.mb_idx).copied().unwrap_or(26),
            )
        } else if within_mb <= 17 {
            // Chroma DC — kept WET.
            return f32::INFINITY;
        } else if within_mb < 22 {
            let slot = within_mb - 18;
            let bx = slot % 2;
            let by = slot / 2;
            (
                &cb_wavelets,
                chroma_w,
                chroma_h,
                mb_x * 8 + bx * 4,
                mb_y * 8 + by * 4,
                fp.qp_cb,
            )
        } else if within_mb < 26 {
            let slot = within_mb - 22;
            let bx = slot % 2;
            let by = slot / 2;
            (
                &cr_wavelets,
                chroma_w,
                chroma_h,
                mb_x * 8 + bx * 4,
                mb_y * 8 + by * 4,
                fp.qp_cr,
            )
        } else {
            return f32::INFINITY;
        };

        let delta = match pos.domain {
            EmbedDomain::T1Sign => 2.0,
            EmbedDomain::LevelSuffixMag => 1.0,
            EmbedDomain::LevelSuffixSign => 2.0 * pos.coeff_value.unsigned_abs() as f64,
            EmbedDomain::MvdLsb => unreachable!("MvdLsb handled above"),
        };
        let cost = compute_position_cost(
            &unit_basis,
            wavelets,
            img_w,
            img_h,
            block_px_x,
            block_px_y,
            pos.scan_pos,
            qp,
            delta,
        );
        if cost.is_finite() && cost > 0.0 {
            cost as f32
        } else {
            f32::INFINITY
        }
    };

    #[cfg(feature = "parallel")]
    {
        frame_positions.par_iter().map(compute_one).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        frame_positions.iter().map(compute_one).collect()
    }
}

/// Minimal description of a position needed to compute its UNIWARD cost in
/// a reconstructed frame. Build this in the pipeline layer by translating
/// the pipeline's global `block_idx` into a per-frame `mb_idx` +
/// `within_mb_block_idx` pair.
pub struct FramePosition<'a> {
    pub pos: &'a EmbeddablePosition,
    /// Macroblock index inside the frame (raster order, 0..width_in_mbs × height_in_mbs).
    pub mb_idx: usize,
    /// Block index inside the macroblock (see [`compute_frame_uniward_costs`]
    /// for the slot layout; 0..=25 under the Phase 2 scheme).
    pub within_mb_block_idx: usize,
    /// Chroma QP for Cb (derived once at parse time from qp_y +
    /// pps.chroma_qp_index_offset). Used when the slot routes to the Cb plane.
    pub qp_cb: i32,
    /// Chroma QP for Cr (derived from qp_y + pps.second_chroma_qp_index_offset).
    pub qp_cr: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wavelets_of_constant_plane_are_zero_away_from_border() {
        // A constant-valued image: all wavelet HP coefficients are 0 except
        // possibly near the padded borders where the reflection introduces
        // slight asymmetry.
        let w = 32;
        let h = 32;
        let img = vec![128u8; w * h];
        let bands = compute_three_subbands(&img, w, h);
        // Central region should be exactly zero on an infinite constant.
        for y in (FILT_LEN + 5)..(bands.height - FILT_LEN - 5) {
            for x in (FILT_LEN + 5)..(bands.width - FILT_LEN - 5) {
                let idx = y * bands.width + x;
                assert!(
                    bands.lh[idx].abs() < 1e-3,
                    "center LH should be zero, got {}",
                    bands.lh[idx]
                );
                assert!(bands.hl[idx].abs() < 1e-3);
                assert!(bands.hh[idx].abs() < 1e-3);
            }
        }
    }

    #[test]
    fn unit_basis_dc_position_is_flat() {
        // The DC unit at (0, 0) produces a constant positive pixel-domain
        // response when inverse-transformed — all 16 entries should be equal.
        let basis = precompute_unit_basis();
        let dc = &basis[0][0];
        let first = dc[0][0];
        assert!(first > 0.0, "DC basis should be positive");
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (dc[i][j] - first).abs() < 1e-6,
                    "DC unit basis should be uniform, got {} vs {}",
                    dc[i][j],
                    first
                );
            }
        }
    }

    #[test]
    fn pixel_scale_matches_transform_module_contract() {
        // At qp=30 (q_bits=5, q_mod=0), (0,0) class 0 has normAdjust 10.
        // Expected factor: 10 * 2^(5-4) = 20. Sanity-check against our
        // derivation.
        let s = pixel_scale(30, 0, 0);
        assert!((s - 20.0).abs() < 1e-9, "pixel_scale(30, 0, 0) = {s}, expected 20.0");
    }

    #[test]
    fn flat_image_gives_infinite_cost_for_mag_lsb() {
        // A perfectly flat image has zero wavelet magnitude. UNIWARD should
        // produce very large cost (anything + σ in the denominator, with a
        // non-zero numerator) — not infinite since σ stabilises the division.
        let w = 32;
        let h = 32;
        let img = vec![128u8; w * h];

        let wavelets = compute_three_subbands(&img, w, h);
        let unit_basis = precompute_unit_basis();

        let cost = compute_position_cost(
            &unit_basis,
            &wavelets,
            w,
            h,
            16, // middle of the image, well inside the padding
            16,
            5, // some non-DC AC position
            26,
            1.0,
        );
        // Large but finite.
        assert!(cost.is_finite(), "flat image cost must be finite");
        assert!(cost > 1.0, "flat image should give high cost, got {cost}");
    }
}
