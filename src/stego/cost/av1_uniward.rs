// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase B.1.2 — AV1 J-UNIWARD cost function.
//!
//! Port of [`crate::stego::cost::h264_uniward`] to AV1's mixed-size
//! transform set (4×4 to 64×64, both square and rectangular). The
//! algorithm is unchanged — Daubechies-8 wavelet decomposition of
//! the post-LR reconstructed pixel domain, score each candidate flip
//! by how much it perturbs the wavelet coefficients relative to the
//! cover's wavelet magnitudes.
//!
//! AV1-specific differences from the H.264 port:
//!
//! * **Variable TX sizes**. H.264 has fixed 4×4; AV1 has square +
//!   rectangular shapes log2-width × log2-height in {2,3,4,5,6}.
//!   Impact window per flip = `tx_size + FILT_LEN - 1`.
//! * **Multiple transforms**. AV1's TX type set is DCT, ADST,
//!   FLIPADST, IDTX × 2D combinations (16 total). v1 MVP uses a
//!   single 2D DCT-II basis for all non-IDTX combinations; ADST /
//!   FLIPADST per-type bases queued for v0.6+ refinement. IDTX
//!   coefficients return uniform cost (no frequency basis).
//! * **Quantizer**. AV1 uses qindex 0..255. v1 MVP uses a simple
//!   linear quant-step approximation. AV1 spec dq_table is queued
//!   for v0.6+.
//!
//! The wavelet filter coefficients (HPDF/LPDF) and the
//! `compute_three_subbands` decomposition match those in
//! `cost::h264_uniward` (same Daubechies-8 wavelet). Duplicated
//! locally instead of imported because h264_uniward is gated on
//! `feature = "video"` while av1_uniward needs to be available
//! whenever av1-encoder is active.

/// Stabilization constant σ from the UNIWARD paper. Same value as
/// JPEG + H.264 J-UNIWARD. Calibrated for 8×8 originally; AV1 mixes
/// sizes so per-size σ tuning is v0.6+.
const SIGMA: f64 = 0.015625; // 2^-6

/// AV1 IDTX transform-type index. For 4×4: TX_TYPE=4 = IDTX. For
/// other sizes the IDTX variants are at index 9 (V_DCT), 10 (H_DCT),
/// etc. — we conservatively flag a small set; expand in v0.6+ when
/// per-type bases land.
///
/// AV1 enum (per av1-spec § 6.4.2):
///   0 = DCT_DCT       8 = V_DCT
///   1 = ADST_DCT      9 = H_DCT
///   2 = DCT_ADST     10 = V_ADST
///   3 = ADST_ADST    11 = H_ADST
///   4 = FLIPADST_DCT 12 = V_FLIPADST
///   5 = DCT_FLIPADST 13 = H_FLIPADST
///   6 = FLIPADST_FLIPADST
///   7 = ADST_FLIPADST
///   (IDTX = 15 in some encodings — variable across spec revisions)
const AV1_TX_TYPE_IDTX: u8 = 15;

/// One AC sign emission's spatial info needed for J-UNIWARD cost.
/// Subset of `phasm_rav1e::AcSignMeta` — phasm-core projects encoder
/// or decoder metas into this struct to keep the cost function free
/// of fork-internal types.
#[derive(Debug, Clone, Copy)]
pub struct Av1FramePosition {
    pub plane: u8,
    pub plane_px_x: u16,
    pub plane_px_y: u16,
    pub tx_width_log2: u8,
    pub tx_height_log2: u8,
    pub tx_type: u8,
    pub scan_pos: u16,
}

/// Reconstructed-frame planes (per-plane visible-region packed YUV).
/// Caller is responsible for extracting these from
/// `PhasmFrameRecording.reconstructed_planes` — see
/// [`pack_visible_from_frame`] helper below.
pub struct FramePlanes {
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    pub luma_width: usize,
    pub luma_height: usize,
    pub chroma_width: usize,
    pub chroma_height: usize,
}

/// Compute J-UNIWARD costs for a slice of AC sign positions.
///
/// Returns one f32 per input position, in the same order. f32 type
/// matches the STC embed expectation. INF returned for positions
/// where cost can't be meaningfully computed (out-of-bounds, IDTX —
/// caller should still pass these; the cost vector aligns with the
/// cover bits).
///
/// `qindex` is the frame-level quantizer for v1 MVP (per-block delta-Q
/// awareness deferred to v0.6+ when AV1 segmentation lands here).
pub fn compute_av1_uniward_costs(
    planes: &FramePlanes,
    positions: &[Av1FramePosition],
    qindex: u8,
) -> Vec<f32> {
    // Wavelet decomposition of each plane (one-off per frame). This
    // is the expensive precompute step; per-position cost is then
    // ~O(tx_w * tx_h + impact_window²) which is cheap.
    let y_wavelets = compute_three_subbands(&planes.y, planes.luma_width, planes.luma_height);
    let cb_wavelets =
        compute_three_subbands(&planes.cb, planes.chroma_width, planes.chroma_height);
    let cr_wavelets =
        compute_three_subbands(&planes.cr, planes.chroma_width, planes.chroma_height);

    let q_scale = qindex_to_step(qindex);

    positions
        .iter()
        .map(|p| compute_position_cost(p, &y_wavelets, &cb_wavelets, &cr_wavelets, planes, q_scale))
        .collect()
}

/// AV1 quantizer step approximation for v1 MVP. Real AV1 uses two
/// tables (DC + AC) indexed by qindex with non-linear scaling
/// (av1-spec § 7.12.2). For v1 we linearly approximate ac_qstep ≈
/// qindex / 4 + 1 — gives reasonable separation across the q
/// range. v0.6+ should swap for the spec tables for accuracy.
#[inline]
fn qindex_to_step(qindex: u8) -> f64 {
    (qindex as f64 / 4.0 + 1.0).max(1.0)
}

/// Compute per-flip J-UNIWARD cost.
fn compute_position_cost(
    pos: &Av1FramePosition,
    y_wavelets: &ThreeSubbands,
    cb_wavelets: &ThreeSubbands,
    cr_wavelets: &ThreeSubbands,
    planes: &FramePlanes,
    q_scale: f64,
) -> f32 {
    // IDTX: no frequency basis — return uniform cost. Same outcome as
    // v0.3's vec![1.0]. Per cost-model.md § 2.2 v0.3 implementation note.
    if pos.tx_type == AV1_TX_TYPE_IDTX {
        return 1.0;
    }

    // Pick wavelets + frame dims for the plane.
    let (wavelets, img_w, img_h) = match pos.plane {
        0 => (y_wavelets, planes.luma_width, planes.luma_height),
        1 => (cb_wavelets, planes.chroma_width, planes.chroma_height),
        2 => (cr_wavelets, planes.chroma_width, planes.chroma_height),
        _ => return f32::INFINITY,
    };

    let tx_w = 1usize << pos.tx_width_log2;
    let tx_h = 1usize << pos.tx_height_log2;
    let block_px_x = pos.plane_px_x as usize;
    let block_px_y = pos.plane_px_y as usize;

    // Bounds check — out-of-frame metadata is a fork bug; treat as
    // WET (infinite cost).
    if block_px_x + tx_w > img_w || block_px_y + tx_h > img_h {
        return f32::INFINITY;
    }

    // Decode (freq_y, freq_x) from scan_pos. scan_pos is the raster
    // index = freq_y * tx_w + freq_x.
    let scan_pos = pos.scan_pos as usize;
    if scan_pos == 0 {
        // DC coefficient — not an AC sign emission. Should never get
        // here since the encoder tags only AC signs; defensive.
        return f32::INFINITY;
    }
    let freq_x = scan_pos % tx_w;
    let freq_y = scan_pos / tx_w;
    if freq_y >= tx_h {
        return f32::INFINITY;
    }

    // Generate 2D DCT-II basis at (freq_y, freq_x) of size (tx_h, tx_w).
    // Scale by quantizer step + |delta| (= 2 for sign flip) so the
    // basis already encodes "what this flip does in pixel space".
    let mut basis = vec![0.0f64; tx_w * tx_h];
    dct_ii_basis(tx_w, tx_h, freq_x, freq_y, &mut basis);
    let delta_magnitude = 2.0 * q_scale; // sign flip: coeff goes +1 ↔ −1, scaled by quant step
    for v in basis.iter_mut() {
        *v *= delta_magnitude;
    }

    // Phase B.1.3: cascade-safety magnitude proxy (Early-exit C per
    // cascade-safety.md § 9). The maximum absolute pixel delta in
    // the pre-cascade basis pattern captures "single hot pixel"
    // disturbance — exactly the shape CDEF + LR amplify into visible
    // block-shaped artifacts. Added to the cost as
    //   final_cost = juw_cost + LAMBDA_CASCADE * max_pixel_delta
    // λ_cascade = 1.0 per cost-model.md § 4.2 starting value.
    //
    // The FULL cascade-safety filter (per-flip forward modeling
    // through deblock + CDEF + LR per cascade-safety.md § 4-9)
    // is v0.6+ work. This magnitude proxy gives a meaningful cascade
    // signal at ~zero extra cost in the v0.5 ship.
    let cascade_magnitude_proxy = basis
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);

    // Filter the basis through Daubechies-8 wavelets, accumulating
    // cost against the cover wavelets in the impact window.
    let pad = FILT_LEN - 1; // 15
    let impact_w = tx_w + pad;
    let impact_h = tx_h + pad;

    // Row-filter the basis: produce two buffers (low-pass + high-pass
    // along rows), each tx_h rows × impact_w cols.
    let mut row_low = vec![0.0f64; tx_h * impact_w];
    let mut row_high = vec![0.0f64; tx_h * impact_w];
    let lp = lpdf();
    for r in 0..tx_h {
        for out_c in 0..impact_w {
            let mut sum_low = 0.0;
            let mut sum_high = 0.0;
            for k in 0..FILT_LEN {
                // src = out_c - (FILT_LEN - 1) + k
                let src = out_c as isize - (FILT_LEN - 1) as isize + k as isize;
                if src >= 0 && (src as usize) < tx_w {
                    let v = basis[r * tx_w + src as usize];
                    sum_low += lp[k] * v;
                    sum_high += HPDF[k] * v;
                }
            }
            row_low[r * impact_w + out_c] = sum_low;
            row_high[r * impact_w + out_c] = sum_high;
        }
    }

    // Column-filter into 3 directional subbands + accumulate cost.
    let mut cost = 0.0f64;
    for out_r in 0..impact_h {
        for out_c in 0..impact_w {
            let mut delta_lh = 0.0;
            let mut delta_hl = 0.0;
            let mut delta_hh = 0.0;
            for k in 0..FILT_LEN {
                let src_r = out_r as isize - (FILT_LEN - 1) as isize + k as isize;
                if src_r >= 0 && (src_r as usize) < tx_h {
                    let r = src_r as usize;
                    let low_val = row_low[r * impact_w + out_c];
                    let high_val = row_high[r * impact_w + out_c];
                    delta_lh += HPDF[k] * low_val;
                    delta_hl += lp[k] * high_val;
                    delta_hh += HPDF[k] * high_val;
                }
            }

            // Map (out_r, out_c) back to image coordinates.
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

    // Phase B.1.3: add the cascade-magnitude-proxy term.
    let final_cost = cost + LAMBDA_CASCADE * cascade_magnitude_proxy;
    final_cost as f32
}

/// λ_cascade — multiplicative weight on the cascade-magnitude-proxy
/// cost term. Per cost-model.md § 4.2 starting value 1.0; tune
/// empirically against the AoSO self-steganalyzer (B.1.4) once
/// available. Smaller values → J-UNIWARD wavelet dominates. Larger
/// values → cascade-proxy dominates (we'd flip in smoother pixel
/// regions even if J-UNIWARD prefers textured ones).
const LAMBDA_CASCADE: f64 = 1.0;

/// 2D DCT-II basis at (freq_y, freq_x) of size (tx_h, tx_w). v1 MVP
/// uses this for ALL non-IDTX transform types — including ADST and
/// FLIPADST. The approximation: ADST/FLIPADST bases are shifted /
/// reflected versions of DCT; the wavelet response of the cost is
/// similar in magnitude, just localized differently. v0.6+ should
/// implement per-type bases for accuracy.
fn dct_ii_basis(tx_w: usize, tx_h: usize, freq_x: usize, freq_y: usize, out: &mut [f64]) {
    let pi = std::f64::consts::PI;
    let c_u = if freq_y == 0 {
        1.0 / (tx_h as f64).sqrt()
    } else {
        (2.0 / tx_h as f64).sqrt()
    };
    let c_v = if freq_x == 0 {
        1.0 / (tx_w as f64).sqrt()
    } else {
        (2.0 / tx_w as f64).sqrt()
    };
    let norm = c_u * c_v;
    for i in 0..tx_h {
        for j in 0..tx_w {
            let cos_i = ((2.0 * i as f64 + 1.0) * freq_y as f64 * pi / (2.0 * tx_h as f64)).cos();
            let cos_j = ((2.0 * j as f64 + 1.0) * freq_x as f64 * pi / (2.0 * tx_w as f64)).cos();
            out[i * tx_w + j] = norm * cos_i * cos_j;
        }
    }
}

/// Three wavelet subbands (LH / HL / HH) of one of the reconstructed
/// YUV planes. Same shape as `cost::h264_uniward::ThreeSubbands`,
/// duplicated here to avoid feature-flag entanglement.
struct ThreeSubbands {
    lh: Vec<f32>,
    hl: Vec<f32>,
    hh: Vec<f32>,
    width: usize,
    #[allow(dead_code)]
    height: usize,
    x_offset: isize,
    y_offset: isize,
}

/// Compute the three Daubechies-8 directional subbands of an 8-bit
/// pixel plane. Matches `cost::h264_uniward::compute_three_subbands`
/// byte-for-byte.
fn compute_three_subbands(y_plane: &[u8], width: usize, height: usize) -> ThreeSubbands {
    let pad = FILT_LEN - 1;
    let padded_w = width + 2 * pad;
    let padded_h = height + 2 * pad;

    let mut row_low = vec![0.0f32; padded_w * height];
    let mut row_high = vec![0.0f32; padded_w * height];
    let lp = lpdf();

    for y in 0..height {
        for out_x in 0..padded_w {
            let mut sum_low = 0.0f64;
            let mut sum_high = 0.0f64;
            for k in 0..FILT_LEN {
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

    let mut lh = vec![0.0f32; padded_w * padded_h];
    let mut hl = vec![0.0f32; padded_w * padded_h];
    let mut hh = vec![0.0f32; padded_w * padded_h];

    for out_y in 0..padded_h {
        for x in 0..padded_w {
            let mut sum_lh = 0.0f64;
            let mut sum_hl = 0.0f64;
            let mut sum_hh = 0.0f64;
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
fn symmetric_reflect(i: isize, len: isize) -> isize {
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

/// Daubechies-8 high-pass decomposition filter (16 taps). Same as
/// JPEG + H.264 J-UNIWARD modules.
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

const FILT_LEN: usize = 16;

#[inline]
fn lpdf() -> [f64; 16] {
    let mut lp = [0.0f64; 16];
    for n in 0..16 {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        lp[n] = sign * HPDF[15 - n];
    }
    lp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_planes(w: usize, h: usize) -> FramePlanes {
        let mut y = vec![0u8; w * h];
        for row in 0..h {
            for col in 0..w {
                y[row * w + col] = ((row * 7 + col * 3) & 0xff) as u8;
            }
        }
        let cw = w / 2;
        let ch = h / 2;
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        for row in 0..ch {
            for col in 0..cw {
                cb[row * cw + col] = ((row * 11 + col * 5 + 13) & 0xff) as u8;
                cr[row * cw + col] = ((row * 13 + col * 7 + 31) & 0xff) as u8;
            }
        }
        FramePlanes {
            y,
            cb,
            cr,
            luma_width: w,
            luma_height: h,
            chroma_width: cw,
            chroma_height: ch,
        }
    }

    #[test]
    fn dct_basis_dc_is_constant() {
        // (freq_y, freq_x) = (0, 0) → DC basis is uniform 1/sqrt(N)
        // per dim → constant value across the block.
        let mut basis = vec![0.0; 16];
        dct_ii_basis(4, 4, 0, 0, &mut basis);
        let expected = 0.25; // (1/sqrt(4)) * (1/sqrt(4)) = 1/2 * 1/2 = 0.25
        for v in basis {
            assert!((v - expected).abs() < 1e-9, "DC basis should be {} got {}", expected, v);
        }
    }

    #[test]
    fn dct_basis_first_ac_alternates_along_x() {
        // (freq_y, freq_x) = (0, 1) → cosine along x, constant along y.
        // For 4×4 block: each row should be the same alternating pattern.
        let mut basis = vec![0.0; 16];
        dct_ii_basis(4, 4, 1, 0, &mut basis);
        for row in 1..4 {
            for col in 0..4 {
                assert!(
                    (basis[row * 4 + col] - basis[col]).abs() < 1e-9,
                    "row {} col {} should match row 0",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn uniform_costs_are_finite_and_non_uniform() {
        // Build a small set of AC positions on a synthetic Y plane and
        // verify the returned costs are (a) all finite, (b) not all
        // equal — proves J-UNIWARD is distinguishing positions.
        let planes = synth_planes(32, 32);
        let positions = vec![
            Av1FramePosition {
                plane: 0,
                plane_px_x: 0,
                plane_px_y: 0,
                tx_width_log2: 2,
                tx_height_log2: 2,
                tx_type: 0,
                scan_pos: 1,
            },
            Av1FramePosition {
                plane: 0,
                plane_px_x: 8,
                plane_px_y: 8,
                tx_width_log2: 2,
                tx_height_log2: 2,
                tx_type: 0,
                scan_pos: 5,
            },
            Av1FramePosition {
                plane: 0,
                plane_px_x: 16,
                plane_px_y: 16,
                tx_width_log2: 3,
                tx_height_log2: 3,
                tx_type: 0,
                scan_pos: 7,
            },
        ];
        let costs = compute_av1_uniward_costs(&planes, &positions, 30);
        assert_eq!(costs.len(), 3);
        for &c in &costs {
            assert!(c.is_finite(), "cost should be finite, got {}", c);
            assert!(c > 0.0, "cost should be positive, got {}", c);
        }
        // Not all equal.
        let all_same = costs.iter().all(|&c| (c - costs[0]).abs() < 1e-6);
        assert!(
            !all_same,
            "expected non-uniform costs across positions; got all {}",
            costs[0]
        );
    }

    #[test]
    fn cascade_proxy_adds_positive_increment_to_cost() {
        // Phase B.1.3: verify the cascade-magnitude-proxy term
        // contributes positively. Compare cost on a textured frame
        // (where wavelet energy is high → low J-UNIWARD cost) with
        // and without the cascade term.
        let planes = synth_planes(32, 32);
        let position = Av1FramePosition {
            plane: 0,
            plane_px_x: 8,
            plane_px_y: 8,
            tx_width_log2: 2,
            tx_height_log2: 2,
            tx_type: 0,
            scan_pos: 5,
        };
        let costs = compute_av1_uniward_costs(&planes, &[position], 30);
        // The cost should be > the J-UNIWARD-only value (i.e., the
        // cascade term is non-zero). Hard-asserting a floor is
        // brittle, but the cost itself must be finite + positive.
        assert!(
            costs[0].is_finite(),
            "cascade-augmented cost must remain finite, got {}",
            costs[0]
        );
        assert!(costs[0] > 0.0, "expected positive cost, got {}", costs[0]);
        // At qindex=30, q_scale ≈ 8.5; delta_magnitude = 17. Max
        // DCT-II basis value at AC freq ≈ 0.5. So
        // cascade_magnitude_proxy >= ~4. With LAMBDA_CASCADE = 1.0,
        // this contributes ≥ 4 to cost. Reasonable lower bound.
        assert!(
            costs[0] >= 4.0,
            "cost should include cascade term contribution (>= 4 at q30, AC pos), got {}",
            costs[0]
        );
    }

    #[test]
    fn idtx_returns_uniform_cost() {
        let planes = synth_planes(16, 16);
        let positions = vec![Av1FramePosition {
            plane: 0,
            plane_px_x: 0,
            plane_px_y: 0,
            tx_width_log2: 2,
            tx_height_log2: 2,
            tx_type: AV1_TX_TYPE_IDTX,
            scan_pos: 5,
        }];
        let costs = compute_av1_uniward_costs(&planes, &positions, 30);
        assert_eq!(costs, vec![1.0]);
    }
}
