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

    // #809 — each output row is independent: disjoint write into
    // row_low/row_high, and the inner k-sum stays serial. Parallelize over
    // rows under `parallel`. Result is BIT-IDENTICAL to serial (no reduction
    // is reordered) and adds NO allocation, so the module's documented
    // ~32 MB-per-frame peak is preserved.
    //
    // The row pass mirrors the #848 column-pass treatment: k-outer / x-inner
    // over the shared `saxpy_inner` primitive (NEON / AVX / WASM-SIMD /
    // scalar). The reflection here is on the SOURCE index rather than the
    // output index, so unlike the column pass it does not vectorise over the
    // whole row — but for `out_x in [pad, width)` no tap reflects for ANY `k`
    // (`0 <= out_x - pad + k < width` holds for every `k in 0..=pad`), so that
    // interior reads one fully-contiguous source span per tap. At 1080p that
    // is 1905 of 1950 output columns (98%); the ~45 border columns keep the
    // original scalar arithmetic verbatim.
    //
    // **Bit-identical by construction**, the same argument as #848: each
    // `out_x` keeps its own f64 accumulator, taps are still summed in
    // `k = 0..FILT_LEN` order, and nothing is FMA-fused. The one change in
    // form is `coef * v` becoming `v * coef` inside `saxpy_inner`, which is
    // exact (IEEE 754 multiplication is commutative). Widening the source row
    // to f32 up front is also exact — a u8 is representable in f32, and
    // `u8 as f32 as f64 == u8 as f64`.
    //
    // Scratch per thread: one f32 source row + two f64 accumulator rows
    // (~39 KB at 1080p), the same order as the column pass's ~47 KB.
    let interior = pad < width; // false only for a plane narrower than the filter
    let fill_row = |scratch_wide: &mut Vec<f32>,
                    scratch_acc: &mut Vec<f64>,
                    y: usize,
                    low_row: &mut [f32],
                    high_row: &mut [f32]| {
        let src_row = &y_plane[y * width..y * width + width];

        if interior {
            scratch_wide.clear();
            scratch_wide.extend(src_row.iter().map(|&v| v as f32));
            let (acc_low, acc_high) = scratch_acc.split_at_mut(padded_w);
            let span = width - pad; // interior output columns [pad, width)
            acc_low[..span].fill(0.0);
            acc_high[..span].fill(0.0);
            for k in 0..FILT_LEN {
                // out_x = pad + i  ⇒  src_x = out_x - pad + k = i + k, all
                // in-bounds across the whole span, hence one contiguous read.
                let src = &scratch_wide[k..k + span];
                super::uniward_simd::saxpy_inner(&mut acc_low[..span], src, lp[k]);
                super::uniward_simd::saxpy_inner(&mut acc_high[..span], src, HPDF[k]);
            }
            for i in 0..span {
                low_row[pad + i] = acc_low[i] as f32;
                high_row[pad + i] = acc_high[i] as f32;
            }
        }

        // Border columns (and the whole row when the plane is narrower than
        // the filter): the original scalar path, reflection included.
        let scalar_cols = (0..padded_w)
            .filter(|&out_x| !interior || out_x < pad || out_x >= width);
        for out_x in scalar_cols {
            let mut sum_low = 0.0f64;
            let mut sum_high = 0.0f64;
            for k in 0..FILT_LEN {
                // src_x = out_x - pad + k
                let src_x = out_x as isize - pad as isize + k as isize;
                let clamped = symmetric_reflect(src_x, width as isize);
                let v = src_row[clamped as usize] as f64;
                sum_low += lp[k] * v;
                sum_high += HPDF[k] * v;
            }
            low_row[out_x] = sum_low as f32;
            high_row[out_x] = sum_high as f32;
        }
    };
    #[cfg(feature = "parallel")]
    row_low
        .par_chunks_mut(padded_w)
        .zip(row_high.par_chunks_mut(padded_w))
        .enumerate()
        .for_each_init(
            || (Vec::<f32>::with_capacity(width), vec![0.0f64; padded_w * 2]),
            |(wide, acc), (y, (low_row, high_row))| {
                fill_row(wide, acc, y, low_row, high_row)
            },
        );
    #[cfg(not(feature = "parallel"))]
    {
        let mut wide = Vec::<f32>::with_capacity(width);
        let mut acc = vec![0.0f64; padded_w * 2];
        row_low
            .chunks_mut(padded_w)
            .zip(row_high.chunks_mut(padded_w))
            .enumerate()
            .for_each(|(y, (low_row, high_row))| {
                fill_row(&mut wide, &mut acc, y, low_row, high_row)
            });
    }

    // Column-filter the row-filtered buffers to produce LH, HL, HH subbands.
    // Each subband is `padded_w × padded_h` with `pad` reflection on all four sides.
    let mut lh = vec![0.0f32; padded_w * padded_h];
    let mut hl = vec![0.0f32; padded_w * padded_h];
    let mut hh = vec![0.0f32; padded_w * padded_h];

    // #848 — k-outer / x-inner SAXPY over the shared `saxpy_inner` primitive
    // (NEON / AVX / WASM-SIMD / scalar). The column reflection is on `out_y`
    // only, so each tap reads one fully-contiguous source row and the whole
    // x-range vectorises with no per-element border. Byte-identical to the
    // prior scalar sum by construction: each x keeps its own f64 accumulator,
    // summed in the SAME tap order (k=0..15), FMA-free (`a·b == b·a` in IEEE
    // 754, so coef·val == val·coef). Each output row is independent → still
    // row-parallel; the only added memory is a per-thread 3×padded_w f64
    // scratch (~47 KB/thread), negligible against the lh/hl/hh outputs.
    let fill_col_row = |scratch: &mut [f64], out_y: usize,
                        lh_row: &mut [f32], hl_row: &mut [f32], hh_row: &mut [f32]| {
        let (s_lh, rest) = scratch.split_at_mut(padded_w);
        let (s_hl, s_hh) = rest.split_at_mut(padded_w);
        s_lh.iter_mut().for_each(|v| *v = 0.0); // row_low  → high-pass col
        s_hl.iter_mut().for_each(|v| *v = 0.0); // row_high → low-pass col
        s_hh.iter_mut().for_each(|v| *v = 0.0); // row_high → high-pass col
        for k in 0..FILT_LEN {
            let src_y = out_y as isize - pad as isize + k as isize;
            let clamped = symmetric_reflect(src_y, height as isize) as usize;
            let rl = &row_low[clamped * padded_w..clamped * padded_w + padded_w];
            let rh = &row_high[clamped * padded_w..clamped * padded_w + padded_w];
            super::uniward_simd::saxpy_inner(s_lh, rl, HPDF[k]);
            super::uniward_simd::saxpy_inner(s_hl, rh, lp[k]);
            super::uniward_simd::saxpy_inner(s_hh, rh, HPDF[k]);
        }
        for x in 0..padded_w {
            lh_row[x] = s_lh[x] as f32;
            hl_row[x] = s_hl[x] as f32;
            hh_row[x] = s_hh[x] as f32;
        }
    };
    #[cfg(feature = "parallel")]
    lh.par_chunks_mut(padded_w)
        .zip(hl.par_chunks_mut(padded_w))
        .zip(hh.par_chunks_mut(padded_w))
        .enumerate()
        .for_each_init(
            || vec![0.0f64; padded_w * 3],
            |scratch, (out_y, ((lh_row, hl_row), hh_row))| {
                fill_col_row(scratch.as_mut_slice(), out_y, lh_row, hl_row, hh_row)
            },
        );
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = vec![0.0f64; padded_w * 3];
        lh.chunks_mut(padded_w)
            .zip(hl.chunks_mut(padded_w))
            .zip(hh.chunks_mut(padded_w))
            .enumerate()
            .for_each(|(out_y, ((lh_row, hl_row), hh_row))| {
                fill_col_row(scratch.as_mut_slice(), out_y, lh_row, hl_row, hh_row)
            });
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

/// The subband outputs a cost pass will actually READ, as merged half-open
/// x-spans per output row, in **absolute plane coordinates** (in-bounds pixels
/// only — every reader skips out-of-plane taps, so the reflection padding is
/// never read).
///
/// Built by [`SubbandDemandBuilder`] from the two read footprints the H.264
/// cost code has:
///
/// * a 4×4 coefficient flip at block `(bx, by)` reads the 19×19 window
///   `[bx-15, bx+3] × [by-15, by+3]` ([`accumulate_cost_from_field`]);
/// * an MVD position reads its macroblock's 16×16 area
///   (`content_costs::mb_y_texture`).
///
/// [`compute_three_subbands_for`] evaluates the decomposition only there; see
/// it for why the result is bit-identical where it is defined.
#[derive(Debug, Clone)]
pub(crate) struct SubbandDemand {
    width: usize,
    height: usize,
    /// `rows[y]` = sorted, disjoint, non-touching `[x0, x1)` spans.
    rows: Vec<Vec<(usize, usize)>>,
}

impl SubbandDemand {
    /// Number of demanded output pixels (per subband).
    pub(crate) fn demanded_px(&self) -> usize {
        self.rows.iter().flatten().map(|&(a, b)| b - a).sum()
    }

    /// `true` when no output is read at all.
    pub(crate) fn is_empty(&self) -> bool {
        self.rows.iter().all(|r| r.is_empty())
    }

    /// Whether output pixel `(x, y)` (absolute plane coordinates) is demanded.
    #[cfg(test)]
    pub(crate) fn contains(&self, x: usize, y: usize) -> bool {
        y < self.height && self.rows[y].iter().any(|&(a, b)| a <= x && x < b)
    }
}

/// Collects read footprints for one plane; [`Self::build`] merges them.
#[derive(Debug, Clone)]
pub(crate) struct SubbandDemandBuilder {
    width: usize,
    height: usize,
    /// `(block_px_y, block_px_x)` — top-left of 4×4 coefficient blocks.
    blocks: Vec<(usize, usize)>,
    /// `(px_y, px_x)` — top-left of 16×16 macroblock areas.
    mb_areas: Vec<(usize, usize)>,
}

impl SubbandDemandBuilder {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        Self { width, height, blocks: Vec::new(), mb_areas: Vec::new() }
    }

    /// A coefficient flip in the 4×4 block whose top-left pixel is
    /// `(block_px_x, block_px_y)` — demands its 19×19 impact window.
    pub(crate) fn add_coeff_block(&mut self, block_px_x: usize, block_px_y: usize) {
        self.blocks.push((block_px_y, block_px_x));
    }

    /// An MVD position in the macroblock whose top-left pixel is
    /// `(px_x, px_y)` — demands that 16×16 area.
    pub(crate) fn add_mb_area(&mut self, px_x: usize, px_y: usize) {
        self.mb_areas.push((px_y, px_x));
    }

    pub(crate) fn build(mut self) -> SubbandDemand {
        let (w, h) = (self.width, self.height);
        let mut rows: Vec<Vec<(usize, usize)>> = vec![Vec::new(); h];
        // A primitive at (py, px) covers x ∈ [px+lo, px+hi) and
        // y ∈ [py+lo, py+hi), both clipped to the plane. Primitives sharing a
        // row band are coalesced into spans BEFORE fanning out to the band's
        // output rows, so the work scales with distinct bands, not with
        // positions × window height.
        let mut fan_out = |items: &mut Vec<(usize, usize)>, lo: isize, hi: isize| {
            items.sort_unstable();
            items.dedup();
            let mut i = 0;
            while i < items.len() {
                let py = items[i].0;
                let mut spans: Vec<(usize, usize)> = Vec::new();
                while i < items.len() && items[i].0 == py {
                    let px = items[i].1 as isize;
                    let x0 = (px + lo).clamp(0, w as isize) as usize;
                    let x1 = (px + hi).clamp(0, w as isize) as usize;
                    if x0 < x1 {
                        match spans.last_mut() {
                            Some(last) if x0 <= last.1 => last.1 = last.1.max(x1),
                            _ => spans.push((x0, x1)),
                        }
                    }
                    i += 1;
                }
                let y0 = (py as isize + lo).clamp(0, h as isize) as usize;
                let y1 = (py as isize + hi).clamp(0, h as isize) as usize;
                if spans.is_empty() || y0 >= y1 {
                    continue;
                }
                for row in &mut rows[y0..y1] {
                    row.extend_from_slice(&spans);
                }
            }
        };
        // 19×19 impact window: offsets [-15, +4) on both axes.
        fan_out(&mut self.blocks, -((FILT_LEN - 1) as isize), 4);
        // 16×16 macroblock area: offsets [0, 16).
        fan_out(&mut self.mb_areas, 0, 16);
        for row in &mut rows {
            merge_spans(row);
        }
        SubbandDemand { width: w, height: h, rows }
    }
}

/// Sort and merge overlapping or touching half-open spans in place.
fn merge_spans(spans: &mut Vec<(usize, usize)>) {
    if spans.len() <= 1 {
        return;
    }
    spans.sort_unstable();
    let mut out = 0usize;
    for i in 1..spans.len() {
        let cur = spans[i];
        if cur.0 <= spans[out].1 {
            spans[out].1 = spans[out].1.max(cur.1);
        } else {
            out += 1;
            spans[out] = cur;
        }
    }
    spans.truncate(out + 1);
}

/// [`compute_three_subbands`] evaluated **only at the outputs in `demand`**.
///
/// ## Why this is lossless
///
/// Every output sample of the decomposition is an independent function of the
/// input plane: `band[oy][ox] = Σ_k c₂[k] · row[reflect(oy-15+k)][ox]` with
/// `row[r][ox] = Σ_k c₁[k] · plane[r][reflect(ox-15+k)]`, each sum taken in
/// `k = 0..16` order in its own f64 accumulator and rounded to f32 once. No
/// sample depends on any other output sample. So computing a sample here with
/// exactly the arithmetic [`compute_three_subbands`] uses — the same
/// `saxpy_inner` element-wise mul-then-add (SIMD lanes are independent and
/// FMA-free, so a shorter span changes which elements are processed, never how
/// one is computed) and the same scalar reflected border loop — reproduces it
/// **bit for bit**. The row-pass samples a demanded output needs are exactly
/// `row[reflect(oy-15+k)][ox]` for `k ∈ 0..16`, and those are the ones
/// computed.
///
/// Outputs outside `demand` are left at `0.0` and must never be read: the
/// caller builds `demand` from the same positions it then prices (see
/// `content_costs::fill_frame_costs`). The buffers keep the full padded shape,
/// so every index the cost code computes is unchanged.
///
/// Pinned by `subbands_for_demand_match_full_decomposition_bit_exactly`.
pub(crate) fn compute_three_subbands_for(
    plane: &[u8],
    width: usize,
    height: usize,
    demand: &SubbandDemand,
) -> ThreeSubbands {
    debug_assert_eq!((demand.width, demand.height), (width, height));
    let pad = FILT_LEN - 1; // 15
    let padded_w = width + 2 * pad;
    let padded_h = height + 2 * pad;
    let lp = lpdf();

    let mut lh = vec![0.0f32; padded_w * padded_h];
    let mut hl = vec![0.0f32; padded_w * padded_h];
    let mut hh = vec![0.0f32; padded_w * padded_h];
    let finish = |lh, hl, hh| ThreeSubbands {
        lh,
        hl,
        hh,
        width: padded_w,
        height: padded_h,
        x_offset: -(pad as isize),
        y_offset: -(pad as isize),
    };
    if demand.is_empty() || width == 0 || height == 0 {
        return finish(lh, hl, hh);
    }

    // Row-pass demand: output row y reads row-pass rows reflect(y + k).
    let mut row_need: Vec<Vec<(usize, usize)>> = vec![Vec::new(); height];
    for (y, spans) in demand.rows.iter().enumerate() {
        if spans.is_empty() {
            continue;
        }
        for k in 0..FILT_LEN {
            let r = symmetric_reflect(y as isize + k as isize, height as isize) as usize;
            row_need[r].extend_from_slice(spans);
        }
    }
    for spans in &mut row_need {
        merge_spans(spans);
    }

    // ── Row pass over the demanded (row, x-span) pieces. ──
    //
    // Mirrors `compute_three_subbands`'s `fill_row`: interior output columns
    // `out_x ∈ [pad, width)` — absolute `x ∈ [0, width - pad)` — read one
    // contiguous source span per tap through `saxpy_inner`; every other column
    // takes the scalar reflected loop. Same accumulator, same tap order, per
    // element.
    let mut row_low = vec![0.0f32; padded_w * height];
    let mut row_high = vec![0.0f32; padded_w * height];
    let interior = pad < width; // false only for a plane narrower than the filter
    let interior_end = if interior { width - pad } else { 0 };
    let fill_row = |scratch_wide: &mut Vec<f32>,
                    scratch_acc: &mut Vec<f64>,
                    y: usize,
                    low_row: &mut [f32],
                    high_row: &mut [f32]| {
        let spans = &row_need[y];
        if spans.is_empty() {
            return;
        }
        let src_row = &plane[y * width..y * width + width];
        if interior {
            scratch_wide.clear();
            scratch_wide.extend(src_row.iter().map(|&v| v as f32));
        }
        for &(a0, a1) in spans {
            // Interior piece: absolute x in [a0, i1).
            let i1 = a1.min(interior_end).max(a0);
            if a0 < i1 {
                let len = i1 - a0;
                let (acc_low, acc_high) = scratch_acc.split_at_mut(padded_w);
                acc_low[..len].fill(0.0);
                acc_high[..len].fill(0.0);
                for k in 0..FILT_LEN {
                    let src = &scratch_wide[a0 + k..a0 + k + len];
                    super::uniward_simd::saxpy_inner(&mut acc_low[..len], src, lp[k]);
                    super::uniward_simd::saxpy_inner(&mut acc_high[..len], src, HPDF[k]);
                }
                for i in 0..len {
                    low_row[pad + a0 + i] = acc_low[i] as f32;
                    high_row[pad + a0 + i] = acc_high[i] as f32;
                }
            }
            // Border piece (reflection): the original scalar arithmetic.
            for ax in i1..a1 {
                let out_x = ax + pad;
                let mut sum_low = 0.0f64;
                let mut sum_high = 0.0f64;
                for k in 0..FILT_LEN {
                    let src_x = out_x as isize - pad as isize + k as isize;
                    let clamped = symmetric_reflect(src_x, width as isize);
                    let v = src_row[clamped as usize] as f64;
                    sum_low += lp[k] * v;
                    sum_high += HPDF[k] * v;
                }
                low_row[out_x] = sum_low as f32;
                high_row[out_x] = sum_high as f32;
            }
        }
    };
    #[cfg(feature = "parallel")]
    row_low
        .par_chunks_mut(padded_w)
        .zip(row_high.par_chunks_mut(padded_w))
        .enumerate()
        .for_each_init(
            || (Vec::<f32>::with_capacity(width), vec![0.0f64; padded_w * 2]),
            |(wide, acc), (y, (low_row, high_row))| fill_row(wide, acc, y, low_row, high_row),
        );
    #[cfg(not(feature = "parallel"))]
    {
        let mut wide = Vec::<f32>::with_capacity(width);
        let mut acc = vec![0.0f64; padded_w * 2];
        row_low
            .chunks_mut(padded_w)
            .zip(row_high.chunks_mut(padded_w))
            .enumerate()
            .for_each(|(y, (low_row, high_row))| {
                fill_row(&mut wide, &mut acc, y, low_row, high_row)
            });
    }

    // ── Column pass over the demanded (row, x-span) pieces. ──
    //
    // Mirrors `fill_col_row`: per tap, one `saxpy_inner` over the span of the
    // reflected source row, f64 accumulators in tap order, one f32 rounding.
    let fill_col_row = |scratch: &mut [f64],
                        out_y: usize,
                        lh_row: &mut [f32],
                        hl_row: &mut [f32],
                        hh_row: &mut [f32]| {
        // Only in-plane output rows are ever demanded.
        if out_y < pad || out_y >= pad + height {
            return;
        }
        let spans = &demand.rows[out_y - pad];
        if spans.is_empty() {
            return;
        }
        let (s_lh, rest) = scratch.split_at_mut(padded_w);
        let (s_hl, s_hh) = rest.split_at_mut(padded_w);
        for &(a0, a1) in spans {
            let (x0, x1) = (a0 + pad, a1 + pad);
            let len = x1 - x0;
            s_lh[..len].fill(0.0);
            s_hl[..len].fill(0.0);
            s_hh[..len].fill(0.0);
            for k in 0..FILT_LEN {
                let src_y = out_y as isize - pad as isize + k as isize;
                let c = symmetric_reflect(src_y, height as isize) as usize;
                let rl = &row_low[c * padded_w + x0..c * padded_w + x1];
                let rh = &row_high[c * padded_w + x0..c * padded_w + x1];
                super::uniward_simd::saxpy_inner(&mut s_lh[..len], rl, HPDF[k]);
                super::uniward_simd::saxpy_inner(&mut s_hl[..len], rh, lp[k]);
                super::uniward_simd::saxpy_inner(&mut s_hh[..len], rh, HPDF[k]);
            }
            for i in 0..len {
                lh_row[x0 + i] = s_lh[i] as f32;
                hl_row[x0 + i] = s_hl[i] as f32;
                hh_row[x0 + i] = s_hh[i] as f32;
            }
        }
    };
    #[cfg(feature = "parallel")]
    lh.par_chunks_mut(padded_w)
        .zip(hl.par_chunks_mut(padded_w))
        .zip(hh.par_chunks_mut(padded_w))
        .enumerate()
        .for_each_init(
            || vec![0.0f64; padded_w * 3],
            |scratch, (out_y, ((lh_row, hl_row), hh_row))| {
                fill_col_row(scratch.as_mut_slice(), out_y, lh_row, hl_row, hh_row)
            },
        );
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = vec![0.0f64; padded_w * 3];
        lh.chunks_mut(padded_w)
            .zip(hl.chunks_mut(padded_w))
            .zip(hh.chunks_mut(padded_w))
            .enumerate()
            .for_each(|(out_y, ((lh_row, hl_row), hh_row))| {
                fill_col_row(scratch.as_mut_slice(), out_y, lh_row, hl_row, hh_row)
            });
    }

    finish(lh, hl, hh)
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

/// Pixel-domain wavelet impact of a unit coefficient flip — the J-UNIWARD
/// "embedding change" decomposed into the 3 directional sub-bands over the
/// 19×19 window it touches. Depends ONLY on `(scan_pos, qp, delta_magnitude)`,
/// NOT on the flip's position in the frame, so the same field serves every
/// position that shares them — see the memoised field table in
/// `content_costs`. 3 × 19 × 19 × f64 ≈ 8.5 KB.
pub(crate) struct DeltaField {
    lh: [[f64; IMPACT_SIZE]; IMPACT_SIZE],
    hl: [[f64; IMPACT_SIZE]; IMPACT_SIZE],
    hh: [[f64; IMPACT_SIZE]; IMPACT_SIZE],
}

/// Build the position-independent [`DeltaField`] for a unit flip at zigzag
/// `scan_pos` (0..=15) under luma `qp`, scaled for `delta_magnitude`. This is
/// the expensive half of the J-UNIWARD cost — the separable 19×19 row+column
/// wavelet filter of the scaled 4×4 basis impulse — and is recomputed for
/// every position by [`compute_position_cost`]; memoise it across positions
/// that share `(scan_pos, qp, delta_magnitude)` to avoid that.
pub(crate) fn compute_delta_field(
    unit_basis: &[[[[f64; 4]; 4]; 4]; 4],
    scan_pos: u8,
    qp: i32,
    delta_magnitude: f64,
) -> DeltaField {
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
                    let bv = basis[r][src as usize];
                    sum_low += lp[k] * bv;
                    sum_high += HPDF[k] * bv;
                }
            }
            row_low[r][out_c] = sum_low;
            row_high[r][out_c] = sum_high;
        }
    }

    // Column-filter into the 3 directional sub-bands.
    let mut field = DeltaField {
        lh: [[0.0; IMPACT_SIZE]; IMPACT_SIZE],
        hl: [[0.0; IMPACT_SIZE]; IMPACT_SIZE],
        hh: [[0.0; IMPACT_SIZE]; IMPACT_SIZE],
    };
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
            field.lh[out_r][out_c] = delta_lh;
            field.hl[out_r][out_c] = delta_hl;
            field.hh[out_r][out_c] = delta_hh;
        }
    }
    field
}

/// Accumulate the J-UNIWARD cost of a flip whose pixel-domain impact is
/// `field`, for a 4×4 block at top-left pixel `(block_px_x, block_px_y)`,
/// against the cover `wavelets`. The position-dependent half:
/// `Σ |Δ_band| / (|cover_band| + σ)` over the in-bounds 19×19 window.
pub(crate) fn accumulate_cost_from_field(
    field: &DeltaField,
    wavelets: &ThreeSubbands,
    img_w: usize,
    img_h: usize,
    block_px_x: usize,
    block_px_y: usize,
) -> f64 {
    let pad = FILT_LEN - 1; // 15
    let mut cost = 0.0f64;
    for out_r in 0..IMPACT_SIZE {
        for out_c in 0..IMPACT_SIZE {
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

            cost += field.lh[out_r][out_c].abs() / (w_lh + SIGMA);
            cost += field.hl[out_r][out_c].abs() / (w_hl + SIGMA);
            cost += field.hh[out_r][out_c].abs() / (w_hh + SIGMA);
        }
    }
    cost
}

/// Compute J-UNIWARD cost for a single candidate flip in an I-frame.
///
/// `block_px_x`, `block_px_y` — pixel coordinates of the top-left corner of
/// the 4×4 block the flip lives in (block-aligned, within the Y plane).
/// `scan_pos` — position within the 4×4 block in zigzag order (0..=15).
/// `delta_magnitude` — absolute value of the coefficient change:
///    * T1 sign flip: 2 (coefficient goes +1 ↔ −1)
///    * LevelSuffixMag flip: 1 (LSB of magnitude toggles)
///
/// Equivalent to `compute_delta_field` then `accumulate_cost_from_field`;
/// the hot path in `content_costs` memoises the former and calls the latter
/// directly, so this reference form is kept for tests + one-shot callers.
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
    let field = compute_delta_field(unit_basis, scan_pos, qp, delta_magnitude);
    accumulate_cost_from_field(&field, wavelets, img_w, img_h, block_px_x, block_px_y)
}

// NOTE (Phase-4 CAVLC retirement): `compute_frame_uniward_costs` +
// `FramePlanes` + `FramePosition` lived here to cost the legacy CAVLC
// pipeline's per-position pool (they routed via `EmbeddablePosition` /
// `EmbedDomain` from `codec::h264::cavlc` + `macroblock::BLOCK_INDEX_TO_POS`).
// That CAVLC stego subsystem was retired; the surviving OH264 Tier-3 cost
// path (`codec::h264::stego::content_costs`) builds its own per-position
// routing and consumes only the primitives kept above
// (`compute_three_subbands` / `precompute_unit_basis` / `compute_position_cost`
// / `ThreeSubbands`). See `docs/design/video/_RETIREMENT-PLAN.md` § "Phase 4".

#[cfg(test)]
mod tests {
    use super::*;

    /// The row pass exactly as it was before the k-outer/SIMD restructure:
    /// x-outer, k-inner, `symmetric_reflect` on every tap. Kept here as the
    /// reference the fast path is held against — a whole-pipeline SHA pin
    /// covers one fixture at one resolution, which is not enough to trust a
    /// kernel rewrite.
    fn reference_row_pass(
        y_plane: &[u8],
        width: usize,
        height: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let pad = FILT_LEN - 1;
        let padded_w = width + 2 * pad;
        let lp = lpdf();
        let mut row_low = vec![0.0f32; padded_w * height];
        let mut row_high = vec![0.0f32; padded_w * height];
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
        (row_low, row_high)
    }

    /// Re-derive the row buffers from the shipped `compute_three_subbands` by
    /// inverting the column pass — not possible, so instead assert the
    /// property that matters end to end: the three subbands a plane produces
    /// must be **bit-identical** to those produced with the reference row
    /// pass feeding the same column pass. We get that by running the column
    /// pass here over the reference row buffers.
    fn reference_three_subbands(
        y_plane: &[u8],
        width: usize,
        height: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let pad = FILT_LEN - 1;
        let padded_w = width + 2 * pad;
        let padded_h = height + 2 * pad;
        let lp = lpdf();
        let (row_low, row_high) = reference_row_pass(y_plane, width, height);

        let mut lh = vec![0.0f32; padded_w * padded_h];
        let mut hl = vec![0.0f32; padded_w * padded_h];
        let mut hh = vec![0.0f32; padded_w * padded_h];
        for out_y in 0..padded_h {
            for x in 0..padded_w {
                let mut s_lh = 0.0f64;
                let mut s_hl = 0.0f64;
                let mut s_hh = 0.0f64;
                for k in 0..FILT_LEN {
                    let src_y = out_y as isize - pad as isize + k as isize;
                    let c = symmetric_reflect(src_y, height as isize) as usize;
                    let rl = row_low[c * padded_w + x] as f64;
                    let rh = row_high[c * padded_w + x] as f64;
                    s_lh += HPDF[k] * rl;
                    s_hl += lp[k] * rh;
                    s_hh += HPDF[k] * rh;
                }
                lh[out_y * padded_w + x] = s_lh as f32;
                hl[out_y * padded_w + x] = s_hl as f32;
                hh[out_y * padded_w + x] = s_hh as f32;
            }
        }
        (lh, hl, hh)
    }

    /// Deterministic pseudo-texture — flat regions, edges and noise in one
    /// plane, so the comparison covers both the vectorised interior and the
    /// scalar border columns with non-trivial values.
    fn synth_plane(width: usize, height: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(width * height);
        let mut s: u32 = 0x1234_5678;
        for y in 0..height {
            for x in 0..width {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (s >> 24) as u8;
                let px = match x * 4 / width.max(1) {
                    0 => 16u8,                                  // flat dark
                    1 => 235,                                   // flat bright
                    2 => ((x * 7 + y * 3) % 256) as u8,         // ramp / edges
                    _ => noise,                                 // noise
                };
                v.push(px);
            }
        }
        v
    }

    /// The k-outer/SIMD row pass must be BIT-identical to the scalar original,
    /// not merely close: these costs select STC flip positions, so a 1-ULP
    /// difference is a different stego file and a broken cross-platform
    /// determinism claim.
    ///
    /// Shapes chosen to hit every branch: widths above, equal to and below the
    /// 15-column filter pad (the `interior` guard), an odd width, and a width
    /// where the interior span is exactly one column.
    #[test]
    fn row_pass_simd_is_bit_identical_to_scalar_reference() {
        for &(w, h) in &[
            (64usize, 8usize),   // ordinary small plane
            (240, 6),            // chroma-ish width
            (97, 5),             // odd width, unaligned interior span
            (16, 4),             // width = pad + 1 → interior span of 1
            (15, 4),             // width = pad     → interior disabled
            (8, 4),              // width < pad     → all-scalar path
            (1, 3),              // degenerate single column
            (1920, 3),           // production luma width (span 1905, odd)
            (960, 3),            // production chroma width (span 945, odd)
        ] {
            let plane = synth_plane(w, h);
            let got = compute_three_subbands(&plane, w, h);
            let (lh, hl, hh) = reference_three_subbands(&plane, w, h);

            assert_eq!(got.lh.len(), lh.len(), "{w}x{h}: LH length");
            for (band, (a, b)) in [("LH", (&got.lh, &lh)), ("HL", (&got.hl, &hl)), ("HH", (&got.hh, &hh))]
            {
                let mismatches = a
                    .iter()
                    .zip(b.iter())
                    .enumerate()
                    .filter(|(_, (x, y))| x.to_bits() != y.to_bits())
                    .take(4)
                    .map(|(i, (x, y))| format!("[{i}] {x:e} vs {y:e}"))
                    .collect::<Vec<_>>();
                assert!(
                    mismatches.is_empty(),
                    "{w}x{h} {band}: SIMD row pass diverges from scalar reference at {}",
                    mismatches.join(", ")
                );
            }
        }
    }

    /// Brute-force reference for [`SubbandDemandBuilder`]: a per-pixel bitmap
    /// of exactly the samples `accumulate_cost_from_field` (19×19 window) and
    /// `mb_y_texture` (16×16 MB area) read, in-bounds only.
    fn reference_demand_bitmap(
        w: usize,
        h: usize,
        blocks: &[(usize, usize)],
        mbs: &[(usize, usize)],
    ) -> Vec<bool> {
        let mut m = vec![false; w * h];
        let mut mark = |x0: isize, y0: isize, x1: isize, y1: isize| {
            for y in y0.max(0)..y1.min(h as isize) {
                for x in x0.max(0)..x1.min(w as isize) {
                    m[y as usize * w + x as usize] = true;
                }
            }
        };
        for &(bx, by) in blocks {
            let (bx, by) = (bx as isize, by as isize);
            mark(bx - 15, by - 15, bx + 4, by + 4);
        }
        for &(px, py) in mbs {
            let (px, py) = (px as isize, py as isize);
            mark(px, py, px + 16, py + 16);
        }
        m
    }

    /// Small deterministic LCG for the demand patterns below.
    fn lcg(s: &mut u64) -> usize {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 33) as usize
    }

    /// The band-limited decomposition must be BIT-identical to the full one at
    /// every sample a cost pass reads, and the demand set must be exactly the
    /// union of the read footprints — no more (wasted work) and, critically,
    /// no less (a read of an uncomputed 0.0 would silently change a cost).
    ///
    /// Patterns cover: sparse blocks, dense blocks, blocks hugging all four
    /// borders (reflection on both passes), MB areas, a mix, and the empty
    /// set; shapes cover the narrow-plane all-scalar path and the production
    /// luma/chroma widths.
    #[test]
    fn subbands_for_demand_match_full_decomposition_bit_exactly() {
        for &(w, h) in &[
            (64usize, 48usize),
            (96, 40),
            (100, 36), // width not a multiple of 16
            (16, 20),  // width = pad + 1
            (12, 12),  // width < pad → all-scalar row pass
            (960, 24), // production chroma width
            (1920, 20), // production luma width
        ] {
            let plane = synth_plane(w, h);
            let full = compute_three_subbands(&plane, w, h);
            let pw = full.width;
            let mut seed = (w * 131 + h) as u64;
            for pattern in 0..6 {
                let mut blocks = Vec::new();
                let mut mbs = Vec::new();
                let (nbx, nby) = ((w / 4).max(1), (h / 4).max(1));
                match pattern {
                    0 => {} // empty
                    1 => {
                        for _ in 0..3 {
                            blocks.push((lcg(&mut seed) % nbx * 4, lcg(&mut seed) % nby * 4));
                        }
                    }
                    2 => {
                        for _ in 0..(nbx * nby / 2).max(1) {
                            blocks.push((lcg(&mut seed) % nbx * 4, lcg(&mut seed) % nby * 4));
                        }
                    }
                    3 => {
                        // Corners and edges: exercises reflection on both axes.
                        for &(bx, by) in &[(0, 0), ((nbx - 1) * 4, 0), (0, (nby - 1) * 4),
                                           ((nbx - 1) * 4, (nby - 1) * 4)] {
                            blocks.push((bx, by));
                        }
                    }
                    4 => {
                        for _ in 0..4 {
                            let mx = lcg(&mut seed) % (w / 16).max(1);
                            let my = lcg(&mut seed) % (h / 16).max(1);
                            mbs.push((mx * 16, my * 16));
                        }
                    }
                    _ => {
                        for _ in 0..6 {
                            blocks.push((lcg(&mut seed) % nbx * 4, lcg(&mut seed) % nby * 4));
                        }
                        mbs.push((0, 0));
                    }
                }
                let mut b = SubbandDemandBuilder::new(w, h);
                for &(x, y) in &blocks {
                    b.add_coeff_block(x, y);
                }
                for &(x, y) in &mbs {
                    b.add_mb_area(x, y);
                }
                let demand = b.build();
                let reference = reference_demand_bitmap(w, h, &blocks, &mbs);
                let got = compute_three_subbands_for(&plane, w, h, &demand);
                assert_eq!((got.width, got.height), (full.width, full.height));
                assert_eq!((got.x_offset, got.y_offset), (full.x_offset, full.y_offset));
                assert_eq!(got.lh.len(), full.lh.len());
                for y in 0..h {
                    for x in 0..w {
                        let need = reference[y * w + x];
                        assert_eq!(
                            demand.contains(x, y), need,
                            "{w}x{h} pattern {pattern}: demand at ({x},{y})"
                        );
                        let i = (y + 15) * pw + (x + 15);
                        for (band, a, f) in [("LH", &got.lh, &full.lh), ("HL", &got.hl, &full.hl),
                                             ("HH", &got.hh, &full.hh)] {
                            if need {
                                assert_eq!(
                                    a[i].to_bits(), f[i].to_bits(),
                                    "{w}x{h} pattern {pattern} {band} ({x},{y}): {} vs {}",
                                    a[i], f[i]
                                );
                            } else {
                                assert_eq!(a[i].to_bits(), 0, "{band} ({x},{y}) not demanded");
                            }
                        }
                    }
                }
                assert_eq!(demand.demanded_px(), reference.iter().filter(|&&b| b).count());
            }
        }
    }

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
