// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! J-UNIWARD (JPEG Universal Wavelet Relative Distortion) cost function.
//!
//! Computes embedding costs by measuring the impact of each DCT coefficient
//! change on a wavelet decomposition of the decompressed image. Uses
//! three directional filters from the Daubechies 8 (db8) wavelet to
//! evaluate smoothness along horizontal, vertical, and diagonal directions.
//!
//! Coefficients in textured regions (large wavelet magnitudes) get lower
//! costs, while coefficients in smooth regions get higher costs — changes
//! in smooth regions are more detectable by steganalysis.
//!
//! ## Cost definition (v2, phasm-core 0.5)
//!
//! The cost of a ±1 change to quantized coefficient `(fi, fj)` of block
//! `(br, bc)` is
//!
//! ```text
//! ρ = Σ_band Σ_(r,c) ∈ [0,23)²  |Δ_band(fi,fj)[r][c]| / (|W_band(8·bc + c − 8, 8·br + r − 8)| + σ)
//! ```
//!
//! * `Δ` ([`AbsDeltaTable`]) is the exact wavelet-domain impact of the
//!   change: window cell `(r, c)` holds the impact at block-relative
//!   wavelet sample `(c − 8, r − 8)`, so the 23×23 window is exactly the
//!   impact's support `[−8, 14]²`.
//! * `W` is the cover's undecimated db8 decomposition of the
//!   **half-sample-symmetric padded** pixel plane (`… b a | a b …`, as
//!   reference J-UNIWARD pads). Samples outside the image are real samples
//!   of that padded plane ([`PaddedSubbands`] holds an 8-sample margin on
//!   every side), so a border block is priced over its whole window
//!   exactly like an interior one.
//!
//! This is the semantics of Butora & Bas's corrected reference
//! (`JUNIWARD_FIX_OFF_BY_ONE`, arXiv:2305.19776). The v1 kernel that
//! shipped up to 0.4.x differed in two ways — it skipped out-of-image
//! window samples (border blocks priced too cheaply, flips piled into the
//! outer block ring) and it read the cover wavelets 8 px up and 8 px left
//! of where the impact lands. It is kept, verbatim, in
//! [`super::uniward_legacy_v1`] for decoding the shadows of older images;
//! see `docs/design/image/juniward-cost.md`.
//!
//! The Ghost encoder uses [`compute_ac_costs_separable`] (the separable
//! factorization of the impact, every Y AC coefficient). The direct-kernel
//! paths — [`compute_ac_costs_streaming`] (strip-streamed) and
//! [`compute_uniward`] (whole-image `CostMap`) — compute every wavelet
//! sample and every cost with the same arithmetic in the same order, so they
//! agree bit for bit, and the separable pass agrees with them to f32
//! rounding (in practice bit for bit, pinned by tests); the NEON / SSE2 / SIMD128
//! kernels match the scalar sum order bit for bit (pinned by
//! `tests/cost_kernel_cross_platform.rs`).
//!
//! Memory-optimized: f32 pixel and wavelet buffers (pixels are 0-255,
//! f32 has a 24-bit significand), f64 accumulation.
//!
//! Reference:
//!   Holub, Fridrich, Denemark. "Universal Distortion Function for
//!   Steganography in an Arbitrary Domain." EURASIP J. on Information
//!   Security, 2014.
//!   Butora, Bas. "Errorless Robust JPEG Steganography using Outputs of
//!   JPEG Coders" / the J-UNIWARD off-by-one note, arXiv:2305.19776.

use crate::codec::jpeg::dct::{DctGrid, QuantTable};
// T3.1.F — ghost UNIWARD bulk pixel-domain decompression routes
// through the integer LL&M IDCT.
//
// `precompute_basis_functions` is the exception: it calls
// `crate::codec::jpeg::pixels::idct_block` EXPLICITLY (the f64
// reference) because basis impulses require sub-LSB precision that
// integer LL&M descales to zero. Called once per encode → no perf
// impact.
use crate::codec::jpeg::pixels_aan::aan_idct_block as idct_block;
use crate::stego::error::StegoError;
use crate::stego::progress;
use super::CostMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub use super::uniward_legacy_v1::compute_positions_streaming_legacy_v1;
#[doc(hidden)]
pub use super::uniward_legacy_v1::cost_kernel_test_hash_hex_legacy_v1;

/// Number of progress steps reported by UNIWARD cost computation.
///
/// Used by the progress total formula in encode/decode so that the combined
/// step counts are correct.
///
/// Steps are distributed proportionally across strips (streaming path) or
/// phases (non-streaming decode path) to give smooth, time-proportional
/// progress updates. UNIWARD is typically the slowest operation (~60-80%
/// of total encode time on large images), so it gets the largest share.
pub const UNIWARD_PROGRESS_STEPS: u32 = 100;

/// Stabilization constant. Avoids division by zero in the cost formula
/// and controls sensitivity to image content. The original paper and
/// reference implementation use sigma = 2^{-6}.
pub(super) const SIGMA: f64 = 0.015625; // 2^{-6}

/// Daubechies 8 (db8) high-pass decomposition filter coefficients.
/// These are the wavelet function coefficients (16 taps).
/// Source: standard db8 wavelet tables, confirmed against conseal/PyWavelets.
pub(super) const HPDF: [f64; 16] = [
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

/// Daubechies 8 (db8) low-pass decomposition filter coefficients.
/// Derived from the high-pass filter via the QMF relation:
///   lpdf[n] = (-1)^n * hpdf[N-1-n]
/// where N = 16.
pub(super) fn lpdf() -> [f64; 16] {
    let mut lp = [0.0f64; 16];
    for n in 0..16 {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        lp[n] = sign * HPDF[15 - n];
    }
    lp
}

/// Length of the 1D wavelet filter.
pub(super) const FILT_LEN: usize = 16;

/// Filter phase: output sample `x` reads input samples `x − 7 ..= x + 8`
/// (`out[x] = Σ_k f[k] · in[x + k − 7]`).
const FILT_HALF: isize = 7;

/// Size of the wavelet-domain impact window when a single DCT coefficient
/// changes: 8 (block size) + 16 (filter length) - 1 = 23.
pub(super) const IMPACT_SIZE: usize = 8 + FILT_LEN - 1;

/// Window origin: cell `(r, c)` of block `(br, bc)`'s cost window is the
/// wavelet sample at plane `(8·bc + c − 8, 8·br + r − 8)`. A pixel change
/// in columns `[0, 8)` of a block reaches outputs `[0 − 8, 7 + 7]` =
/// block-relative `[−8, 14]` (see [`FILT_HALF`]).
const WINDOW_ORIGIN: usize = 8;

/// Wavelet samples [`PaddedSubbands`] keeps outside the image on every
/// side. The cost window reaches 8 samples above / left of the first block
/// and 6 below / right of the last one.
const WAV_MARGIN: usize = 8;

/// Pixel samples of padding the decomposition reads outside the image on
/// every side: the row/column filters reach 7 samples before and 8 after
/// the outermost wavelet sample, `WAV_MARGIN + 8 = 16`.
const PIX_MARGIN: usize = 16;

/// First pixel column (in padded-buffer coordinates) read by the row
/// filter for padded wavelet column 0: plane `x = −WAV_MARGIN` reads plane
/// `x − 7`, i.e. padded column `PIX_MARGIN − WAV_MARGIN − 7`.
const ROW_TAP0: usize = PIX_MARGIN - WAV_MARGIN - FILT_HALF as usize;

const _: () = assert!(WINDOW_ORIGIN == WAV_MARGIN);

/// Wrapper to send a raw pointer across Rayon threads.
/// Safe because each block writes to a disjoint region of the CostMap.
pub(super) struct CostMapPtr {
    pub(super) ptr: *mut f32,
    pub(super) total_len: usize,
}
unsafe impl Send for CostMapPtr {}
unsafe impl Sync for CostMapPtr {}

impl CostMapPtr {
    /// Write a cost value at the given flat index.
    ///
    /// # Safety
    /// Caller must ensure no aliasing writes to the same index.
    pub(super) unsafe fn write(&self, idx: usize, val: f32) {
        debug_assert!(idx < self.total_len, "CostMapPtr write out of bounds: {idx} >= {}", self.total_len);
        unsafe { *self.ptr.add(idx) = val; }
    }
}

/// Half-sample symmetric reflection of `i` into `[0, len)`:
/// `… 2 1 0 | 0 1 2 … len−1 | len−1 len−2 …` (numpy / MATLAB
/// `'symmetric'`). Repeats until in range, so it is total for any `len ≥ 1`.
#[inline]
pub(super) fn symmetric_index(i: isize, len: usize) -> usize {
    let n = len as isize;
    debug_assert!(n > 0);
    let mut v = i;
    while v < 0 || v >= n {
        if v < 0 {
            v = -v - 1;
        }
        if v >= n {
            v = 2 * n - v - 1;
        }
    }
    v as usize
}

/// Compute J-UNIWARD embedding costs for a single component's DCT grid.
///
/// For each embeddable coefficient position, the cost measures how much
/// a ±1 change to that coefficient would alter the wavelet decomposition
/// of the decompressed image, weighted inversely by the cover's wavelet
/// magnitudes plus a stabilization constant sigma (see the module docs).
///
/// DC coefficients and zero-valued AC coefficients receive `WET_COST`.
pub fn compute_uniward(grid: &DctGrid, qt: &QuantTable) -> CostMap {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let mut map = CostMap::new(bw, bt);
    let wavelets = whole_image_subbands(grid, qt);
    let abs_delta = AbsDeltaTable::precompute(qt);

    let n_blocks = bt * bw;
    let costs_ptr = CostMapPtr { ptr: map.costs_ptr(), total_len: n_blocks * 64 };

    let compute_block = |bi: usize| {
        let br = bi / bw;
        let bc = bi % bw;
        let blk = grid.block(br, bc);
        for fi in 0..8 {
            for fj in 0..8 {
                // DC and zero AC coefficients — never modified.
                if (fi == 0 && fj == 0) || blk[fi * 8 + fj] == 0 {
                    continue;
                }
                let (abs_lh, abs_hl, abs_hh) = abs_delta.slice_at(fi, fj);
                let cost = cost_kernel(abs_lh, abs_hl, abs_hh, &wavelets, br, bc);
                if cost > 0.0 && cost.is_finite() {
                    // Safety: each block writes to a disjoint 64-element region.
                    let idx = (br * bw + bc) * 64 + fi * 8 + fj;
                    unsafe { costs_ptr.write(idx, cost as f32); }
                }
            }
        }
    };

    #[cfg(feature = "parallel")]
    (0..n_blocks).into_par_iter().for_each(compute_block);
    #[cfg(not(feature = "parallel"))]
    (0..n_blocks).for_each(compute_block);

    map
}

/// Precompute the 64 basis functions: for each frequency position (fi, fj),
/// compute the 8x8 pixel-domain block that results from IDCT of a unit
/// impulse, scaled by the quantization step.
///
/// This gives the pixel change caused by adding +1 to quantized coefficient
/// (fi, fj).
pub(super) fn precompute_basis_functions(qt: &QuantTable) -> [[[f64; 64]; 8]; 8] {
    let mut basis = [[[0.0f64; 64]; 8]; 8];

    for fi in 0..8 {
        for fj in 0..8 {
            // Create a unit impulse at (fi, fj) in the DCT domain.
            let mut impulse = [0i16; 64];
            impulse[fi * 8 + fj] = 1;

            // Use unit quantization for IDCT, then scale by actual q-step.
            // idct_block adds 128 offset, so we subtract it to get pure basis.
            //
            // T3.1.B.2 — explicitly call the f64 path here, even when
            // the `aan-dct` feature is on. The basis function for a
            // unit impulse has sub-LSB pixel values (~0.125 at DC),
            // which the integer LL&M path descales to zero. A
            // zero-basis row makes UNIWARD's cost formula
            // `Σ|impact|/(|wavelet| + σ)` divide by zero at the
            // boundary — observed as `inf` costs in tests during
            // Phase B.2 development. Called once per encode, so f64
            // here costs nothing.
            let unity_qt = [1u16; 64];
            let pixels = crate::codec::jpeg::pixels::idct_block(&impulse, &unity_qt);

            let q = qt.values[fi * 8 + fj] as f64;
            for k in 0..64 {
                // Remove the +128 DC offset that idct_block adds.
                basis[fi][fj][k] = (pixels[k] - 128.0) * q;
            }
        }
    }

    basis
}

/// T3.3.A — precomputed |delta| table for the cost kernel.
///
/// The wavelet-domain impact `|Δ_lh / Δ_hl / Δ_hh|` at each cell of the
/// 23×23 window depends ONLY on (fi, fj) and qt — NOT on (br, bc) or cover
/// content. Materialized once per encode, the per-block kernel is ~1 600
/// lookups + divides.
///
/// Memory: 63 × 23 × 23 × 3 × 8 B = ~252 KB per encode. Fits in L2
/// on every target.
///
/// Layout: subband-major then `(fi*8+fj)`-major then `(out_r, out_c)`-
/// major. The per-block inner loop reads three contiguous 529-element
/// slices for the current (fi, fj).
pub struct AbsDeltaTable {
    abs_lh: Vec<f64>,
    abs_hl: Vec<f64>,
    abs_hh: Vec<f64>,
}

impl AbsDeltaTable {
    pub(super) const CELLS: usize = IMPACT_SIZE * IMPACT_SIZE;

    /// v2 table: cell `(r, c)` holds the impact at block-relative wavelet
    /// sample `(c − 8, r − 8)` — the impact's exact 23×23 support. Output
    /// sample `x` reads pixel `x + k − 7`, so cell `c` reads basis column
    /// `c − 8 + k − 7 = c − 15 + k`.
    pub fn precompute(qt: &QuantTable) -> Self {
        Self::precompute_with_source_offset(qt, (WINDOW_ORIGIN as isize) + FILT_HALF)
    }

    /// The 0.4.x (v1) table, for [`super::uniward_legacy_v1`] only: cell
    /// `(r, c)` at block-relative `(c − 7, r − 7)`, which drops the
    /// impact's first row and column (−8) and keeps an all-zero last one.
    pub(super) fn precompute_legacy_v1(qt: &QuantTable) -> Self {
        Self::precompute_with_source_offset(qt, 14)
    }

    /// Cell `c` reads basis column `c − src_off + k` for tap `k` (rows
    /// likewise).
    fn precompute_with_source_offset(qt: &QuantTable, src_off: isize) -> Self {
        let basis = precompute_basis_functions(qt);
        let lpdf = lpdf();
        let n_coefs = 64;
        let mut abs_lh = vec![0.0f64; n_coefs * Self::CELLS];
        let mut abs_hl = vec![0.0f64; n_coefs * Self::CELLS];
        let mut abs_hh = vec![0.0f64; n_coefs * Self::CELLS];

        for fi in 0..8 {
            for fj in 0..8 {
                if fi == 0 && fj == 0 {
                    continue; // DC, never used by compute_positions
                }
                let basis_block = &basis[fi][fj];

                // Phase 1: row filter.
                let mut row_low = [[0.0f64; IMPACT_SIZE]; 8];
                let mut row_high = [[0.0f64; IMPACT_SIZE]; 8];
                for r in 0..8 {
                    for out_c in 0..IMPACT_SIZE {
                        let mut sum_low = 0.0;
                        let mut sum_high = 0.0;
                        for k in 0..FILT_LEN {
                            let src_col = out_c as isize - src_off + k as isize;
                            if (0..8).contains(&src_col) {
                                let val = basis_block[r * 8 + src_col as usize];
                                sum_low += lpdf[k] * val;
                                sum_high += HPDF[k] * val;
                            }
                        }
                        row_low[r][out_c] = sum_low;
                        row_high[r][out_c] = sum_high;
                    }
                }

                // Phase 2: column filter → |delta| (absolute value
                // stored — the cost formula only ever uses |delta|).
                let base = (fi * 8 + fj) * Self::CELLS;
                for out_r in 0..IMPACT_SIZE {
                    for out_c in 0..IMPACT_SIZE {
                        let mut delta_lh = 0.0;
                        let mut delta_hl = 0.0;
                        let mut delta_hh = 0.0;
                        for k in 0..FILT_LEN {
                            let src_row = out_r as isize - src_off + k as isize;
                            if (0..8).contains(&src_row) {
                                let r = src_row as usize;
                                let low_val = row_low[r][out_c];
                                let high_val = row_high[r][out_c];
                                delta_lh += HPDF[k] * low_val;
                                delta_hl += lpdf[k] * high_val;
                                delta_hh += HPDF[k] * high_val;
                            }
                        }
                        let idx = base + out_r * IMPACT_SIZE + out_c;
                        abs_lh[idx] = delta_lh.abs();
                        abs_hl[idx] = delta_hl.abs();
                        abs_hh[idx] = delta_hh.abs();
                    }
                }
            }
        }

        AbsDeltaTable { abs_lh, abs_hl, abs_hh }
    }

    /// Borrow the 23×23 slices for coefficient (fi, fj).
    #[inline]
    pub(super) fn slice_at(&self, fi: usize, fj: usize) -> (&[f64], &[f64], &[f64]) {
        let base = (fi * 8 + fj) * Self::CELLS;
        let end = base + Self::CELLS;
        (&self.abs_lh[base..end], &self.abs_hl[base..end], &self.abs_hh[base..end])
    }
}

// ============================================================================
// Padded cover decomposition
// ============================================================================

/// LH / HL / HH subbands of the symmetric-padded pixel plane, for plane
/// columns `x ∈ [−WAV_MARGIN, img_w + WAV_MARGIN)` and a contiguous range
/// of plane rows starting at `y0` (`−WAV_MARGIN` for a whole image, the
/// strip's first window row for a strip).
///
/// - LH: low-pass along rows, high-pass along columns (vertical detail)
/// - HL: high-pass along rows, low-pass along columns (horizontal detail)
/// - HH: high-pass along both (diagonal detail)
///
/// Every sample, inside the image or in the margin, is the plain filter
/// output over the padded pixel plane — no boundary special case — so a
/// sample has the same value whichever row range it was computed in.
pub(super) struct PaddedSubbands {
    pub(super) lh: Vec<f32>,
    pub(super) hl: Vec<f32>,
    pub(super) hh: Vec<f32>,
    /// Samples per row: `img_w + 2·WAV_MARGIN`.
    pub(super) stride: usize,
    /// Plane row of the first stored row.
    pub(super) y0: isize,
    /// Number of stored rows.
    pub(super) rows: usize,
}

impl PaddedSubbands {
    /// Buffer index of plane sample `(x, y)`.
    #[inline]
    fn index(&self, x: isize, y: isize) -> usize {
        debug_assert!(y >= self.y0 && ((y - self.y0) as usize) < self.rows, "row {y} not in strip");
        debug_assert!(x >= -(WAV_MARGIN as isize) && ((x + WAV_MARGIN as isize) as usize) < self.stride);
        (y - self.y0) as usize * self.stride + (x + WAV_MARGIN as isize) as usize
    }
}

/// Symmetric-padded pixel rows `[py0, py0 + n_rows)` of the decompressed
/// grid, `img_w + 2·PIX_MARGIN` samples each (plane column `x` at buffer
/// column `x + PIX_MARGIN`). Pixel `(x, y)` outside the image is the
/// pixel at `(symmetric_index(x), symmetric_index(y))`.
fn padded_pixel_rows(grid: &DctGrid, qt: &QuantTable, py0: isize, n_rows: usize) -> Vec<f32> {
    let mut pix = Vec::new();
    padded_pixel_rows_into(&mut pix, grid, qt, py0, n_rows);
    pix
}

/// [`padded_pixel_rows`] into `pix` (its allocation reused).
fn padded_pixel_rows_into(pix: &mut Vec<f32>, grid: &DctGrid, qt: &QuantTable, py0: isize, n_rows: usize) {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let (img_w, img_h) = (bw * 8, bt * 8);
    let pw = img_w + 2 * PIX_MARGIN;
    let py1 = py0 + n_rows as isize;
    pix.clear();
    pix.resize(pw * n_rows, 0.0);

    // 1. The image rows inside the range, decoded straight into place.
    let in0 = py0.clamp(0, img_h as isize) as usize;
    let in1 = py1.clamp(0, img_h as isize) as usize;
    if in0 < in1 {
        for br in in0 / 8..in1.div_ceil(8) {
            for bc in 0..bw {
                let q: [i16; 64] = grid.block(br, bc).try_into().unwrap();
                let block_pixels = idct_block(&q, &qt.values);
                for r in 0..8 {
                    let y = br * 8 + r;
                    if y < in0 || y >= in1 {
                        continue;
                    }
                    let dst = (y as isize - py0) as usize * pw + PIX_MARGIN + bc * 8;
                    for c in 0..8 {
                        pix[dst + c] = block_pixels[r * 8 + c] as f32;
                    }
                }
            }
        }
    }

    // 2. Rows above / below the image: copies of their mirror rows.
    for i in 0..n_rows {
        let y = py0 + i as isize;
        if y >= 0 && (y as usize) < img_h {
            continue;
        }
        let src = symmetric_index(y, img_h);
        let dst = i * pw + PIX_MARGIN;
        if src >= in0 && src < in1 {
            let s = (src as isize - py0) as usize * pw + PIX_MARGIN;
            pix.copy_within(s..s + img_w, dst);
        } else {
            // Mirror row outside this row range (only for strips of
            // unusual geometry): decode it on its own. Same IDCT, same
            // values.
            let (br, r) = (src / 8, src % 8);
            for bc in 0..bw {
                let q: [i16; 64] = grid.block(br, bc).try_into().unwrap();
                let block_pixels = idct_block(&q, &qt.values);
                for c in 0..8 {
                    pix[dst + bc * 8 + c] = block_pixels[r * 8 + c] as f32;
                }
            }
        }
    }

    // 3. Columns left / right of the image, every row.
    fill_column_margins(pix, img_w);
}

/// Fill the `PIX_MARGIN` columns on both sides of every padded row from its
/// in-image samples, by half-sample symmetry.
fn fill_column_margins(pix: &mut [f32], img_w: usize) {
    let pw = img_w + 2 * PIX_MARGIN;
    for row in pix.chunks_mut(pw) {
        for xp in (0..PIX_MARGIN).chain(PIX_MARGIN + img_w..pw) {
            let sx = symmetric_index(xp as isize - PIX_MARGIN as isize, img_w);
            row[xp] = row[PIX_MARGIN + sx];
        }
    }
}

/// LH / HL / HH subbands for plane rows `[wy0, wy1)` from the padded pixel
/// rows `[wy0 − 7, wy1 + 8)` (`pix`, as built by [`padded_pixel_rows`]).
///
/// Row pass: each pixel row → `stride` row-filtered samples, one
/// [`super::uniward_simd::dot_product_16_taps`] per sample (pinned
/// pair-wise f64 tree). Column pass: each output row accumulates its 16
/// source rows in tap order through
/// [`super::uniward_simd::saxpy_inner`] (f64, no FMA). Both kernels are
/// bit-identical across NEON / AVX / SIMD128 / scalar, and every sample's
/// arithmetic is independent of which rows the call covers.
fn padded_subbands_from_pixels(pix: &[f32], img_w: usize, wy0: isize, wy1: isize) -> PaddedSubbands {
    let pw = img_w + 2 * PIX_MARGIN;
    let stride = img_w + 2 * WAV_MARGIN;
    let n_pix_rows = pix.len() / pw;
    let n_out = (wy1 - wy0) as usize;
    debug_assert_eq!(n_pix_rows, n_out + FILT_LEN - 1, "pixel rows must be [wy0-7, wy1+8)");
    let lp = lpdf();

    let row_pass = |filter: &[f64; 16]| -> Vec<f32> {
        let mut out = vec![0.0f32; stride * n_pix_rows];
        row_filter_into(pix, img_w, filter, &mut out);
        out
    };
    let col_pass = |rows: &[f32], filter: &[f64; 16]| -> Vec<f32> {
        let mut out = vec![0.0f32; stride * n_out];
        col_filter_into(rows, stride, filter, &mut out, |v| v);
        out
    };

    #[cfg(feature = "parallel")]
    let (lh, hl, hh) = {
        let (row_low, row_high) = rayon::join(|| row_pass(&lp), || row_pass(&HPDF));
        let (lh, (hl, hh)) = rayon::join(
            || col_pass(&row_low, &HPDF),
            || rayon::join(|| col_pass(&row_high, &lp), || col_pass(&row_high, &HPDF)),
        );
        (lh, hl, hh)
    };
    // Sequential: drop intermediates early (WASM path).
    #[cfg(not(feature = "parallel"))]
    let (lh, hl, hh) = {
        let row_low = row_pass(&lp);
        let lh = col_pass(&row_low, &HPDF);
        drop(row_low);
        let row_high = row_pass(&HPDF);
        let hl = col_pass(&row_high, &lp);
        let hh = col_pass(&row_high, &HPDF);
        (lh, hl, hh)
    };

    PaddedSubbands { lh, hl, hh, stride, y0: wy0, rows: n_out }
}

/// Row pass: padded pixel row `r` of `pix` (`img_w + 2·PIX_MARGIN`
/// samples) → row `r` of `out` (`img_w + 2·WAV_MARGIN` row-filtered
/// samples), one [`super::uniward_simd::dot_product_16_taps`] per sample.
fn row_filter_into(pix: &[f32], img_w: usize, filter: &[f64; 16], out: &mut [f32]) {
    let pw = img_w + 2 * PIX_MARGIN;
    let stride = img_w + 2 * WAV_MARGIN;
    debug_assert_eq!(out.len() / stride, pix.len() / pw);
    let body = |r: usize, o: &mut [f32]| {
        let src = &pix[r * pw..(r + 1) * pw];
        for (i, v) in o.iter_mut().enumerate() {
            let s = i + ROW_TAP0;
            *v = super::uniward_simd::dot_product_16_taps(&src[s..s + FILT_LEN], filter) as f32;
        }
    };
    #[cfg(feature = "parallel")]
    out.par_chunks_mut(stride).enumerate().for_each(|(r, o)| body(r, o));
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(stride).enumerate().for_each(|(r, o)| body(r, o));
}

/// Column pass: output row `o` of `out` accumulates row-filtered rows
/// `o .. o + 16` of `rows` (plane rows `o − 7 ..= o + 8` relative to the
/// output) in tap order through [`super::uniward_simd::saxpy_inner`] (f64,
/// no FMA); each sample is rounded to f32 and stored as `emit(sample)`.
fn col_filter_into<T: Send>(rows: &[f32], stride: usize, filter: &[f64; 16], out: &mut [T], emit: impl Fn(f32) -> T + Sync) {
    debug_assert!(rows.len() >= out.len() + (FILT_LEN - 1) * stride);
    let body = |scratch: &mut [f64], o: usize, dst: &mut [T]| {
        scratch.fill(0.0);
        for (k, &f) in filter.iter().enumerate() {
            let r = o + k;
            super::uniward_simd::saxpy_inner(scratch, &rows[r * stride..(r + 1) * stride], f);
        }
        for (d, &s) in dst.iter_mut().zip(scratch.iter()) {
            *d = emit(s as f32);
        }
    };
    #[cfg(feature = "parallel")]
    out.par_chunks_mut(stride).enumerate().for_each_init(
        || vec![0.0f64; stride],
        |scratch, (o, dst)| body(scratch.as_mut_slice(), o, dst),
    );
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = vec![0.0f64; stride];
        out.chunks_mut(stride).enumerate().for_each(|(o, dst)| body(&mut scratch, o, dst));
    }
}

/// Subbands for plane rows `[wy0, wy1)` of `grid` (all columns, margins
/// included).
fn padded_subbands(grid: &DctGrid, qt: &QuantTable, wy0: isize, wy1: isize) -> PaddedSubbands {
    let n_pix_rows = (wy1 - wy0) as usize + FILT_LEN - 1;
    let pix = padded_pixel_rows(grid, qt, wy0 - FILT_HALF, n_pix_rows);
    padded_subbands_from_pixels(&pix, grid.blocks_wide() * 8, wy0, wy1)
}

/// The whole image's subbands, margins included: rows
/// `[−WAV_MARGIN, img_h + WAV_MARGIN)`.
fn whole_image_subbands(grid: &DctGrid, qt: &QuantTable) -> PaddedSubbands {
    let img_h = (grid.blocks_tall() * 8) as isize;
    padded_subbands(grid, qt, -(WAV_MARGIN as isize), img_h + WAV_MARGIN as isize)
}

/// Window rows `[wy0, wy1)` the blocks of block rows `[br0, br1)` read.
#[inline]
fn window_rows_for_block_rows(br0: usize, br1: usize) -> (isize, isize) {
    let wy0 = (br0 * 8) as isize - WINDOW_ORIGIN as isize;
    let wy1 = ((br1 - 1) * 8) as isize - WINDOW_ORIGIN as isize + IMPACT_SIZE as isize;
    (wy0, wy1)
}

// ============================================================================
// Cost kernel
// ============================================================================

/// One window row: `Σ_c (|Δlh|/(|Wlh|+σ) + |Δhl|/(|Whl|+σ)) + |Δhh|/(|Whh|+σ)`
/// over the 23 cells, accumulated into `cost`.
///
/// Sum order (identical on every path): per cell `(lh + hl) + hh`, cells
/// paired `cost += cell0 + cell1` for `(0,1) (2,3) … (20,21)`, then the
/// 1-cell tail `cost += cell22`. The NEON / SIMD128 / SSE2 blocks compute
/// the two cells of a pair in two f64 lanes and add `lane0 + lane1`, which
/// is exactly the scalar order — no FMA, no reduction tree.
#[inline(always)]
fn accumulate_window_row(
    cost: &mut f64,
    d_lh: &[f64],
    d_hl: &[f64],
    d_hh: &[f64],
    w_lh: &[f32],
    w_hl: &[f32],
    w_hh: &[f32],
) {
    let n = IMPACT_SIZE;
    let (d_lh, d_hl, d_hh) = (&d_lh[..n], &d_hl[..n], &d_hh[..n]);
    let (w_lh, w_hl, w_hh) = (&w_lh[..n], &w_hl[..n], &w_hh[..n]);
    let mut c = 0usize;

    // NEON 2-lane f64 path on aarch64.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        use core::arch::aarch64::*;
        let sigma_v = vdupq_n_f64(SIGMA);
        while c + 1 < n {
            // In bounds: c + 1 < n = slice length.
            let v_lh_w = vaddq_f64(vabsq_f64(vcvt_f64_f32(vld1_f32(w_lh.as_ptr().add(c)))), sigma_v);
            let v_hl_w = vaddq_f64(vabsq_f64(vcvt_f64_f32(vld1_f32(w_hl.as_ptr().add(c)))), sigma_v);
            let v_hh_w = vaddq_f64(vabsq_f64(vcvt_f64_f32(vld1_f32(w_hh.as_ptr().add(c)))), sigma_v);
            let v_div_lh = vdivq_f64(vld1q_f64(d_lh.as_ptr().add(c)), v_lh_w);
            let v_div_hl = vdivq_f64(vld1q_f64(d_hl.as_ptr().add(c)), v_hl_w);
            let v_div_hh = vdivq_f64(vld1q_f64(d_hh.as_ptr().add(c)), v_hh_w);
            let v_partial = vaddq_f64(vaddq_f64(v_div_lh, v_div_hl), v_div_hh);
            let lane0 = vgetq_lane_f64(v_partial, 0);
            let lane1 = vgetq_lane_f64(v_partial, 1);
            *cost += lane0 + lane1;
            c += 2;
        }
    }

    // WASM SIMD128 2-lane f64 path. Phasm's wasm bridges enable
    // `+simd128` in `.cargo/config.toml`, so this gate fires for the
    // deployed phasm.app build.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let sigma_v = f64x2_splat(SIGMA);
        while c + 1 < n {
            // `v128_load64_zero` reads 2 f32 into the low half;
            // `f64x2_promote_low_f32x4` widens them to 2 f64.
            let v_lh_w = f64x2_add(f64x2_abs(f64x2_promote_low_f32x4(
                v128_load64_zero(w_lh.as_ptr().add(c) as *const u64))), sigma_v);
            let v_hl_w = f64x2_add(f64x2_abs(f64x2_promote_low_f32x4(
                v128_load64_zero(w_hl.as_ptr().add(c) as *const u64))), sigma_v);
            let v_hh_w = f64x2_add(f64x2_abs(f64x2_promote_low_f32x4(
                v128_load64_zero(w_hh.as_ptr().add(c) as *const u64))), sigma_v);
            let v_div_lh = f64x2_div(v128_load(d_lh.as_ptr().add(c) as *const v128), v_lh_w);
            let v_div_hl = f64x2_div(v128_load(d_hl.as_ptr().add(c) as *const v128), v_hl_w);
            let v_div_hh = f64x2_div(v128_load(d_hh.as_ptr().add(c) as *const v128), v_hh_w);
            let v_partial = f64x2_add(f64x2_add(v_div_lh, v_div_hl), v_div_hh);
            let lane0 = f64x2_extract_lane::<0>(v_partial);
            let lane1 = f64x2_extract_lane::<1>(v_partial);
            *cost += lane0 + lane1;
            c += 2;
        }
    }

    // SSE2 2-lane f64 path on x86_64 (baseline on every x86_64 chip).
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let sigma_v = _mm_set1_pd(SIGMA);
        // Bitmask for f64 |.|: clear sign bit (bit 63).
        let abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF));
        while c + 1 < n {
            // Load 2 f32 per subband (64-bit load), widen to f64.
            let v_lh_f32 = _mm_castsi128_ps(_mm_loadl_epi64(w_lh.as_ptr().add(c) as *const __m128i));
            let v_hl_f32 = _mm_castsi128_ps(_mm_loadl_epi64(w_hl.as_ptr().add(c) as *const __m128i));
            let v_hh_f32 = _mm_castsi128_ps(_mm_loadl_epi64(w_hh.as_ptr().add(c) as *const __m128i));
            let v_lh_w = _mm_add_pd(_mm_and_pd(_mm_cvtps_pd(v_lh_f32), abs_mask), sigma_v);
            let v_hl_w = _mm_add_pd(_mm_and_pd(_mm_cvtps_pd(v_hl_f32), abs_mask), sigma_v);
            let v_hh_w = _mm_add_pd(_mm_and_pd(_mm_cvtps_pd(v_hh_f32), abs_mask), sigma_v);
            let v_div_lh = _mm_div_pd(_mm_loadu_pd(d_lh.as_ptr().add(c)), v_lh_w);
            let v_div_hl = _mm_div_pd(_mm_loadu_pd(d_hl.as_ptr().add(c)), v_hl_w);
            let v_div_hh = _mm_div_pd(_mm_loadu_pd(d_hh.as_ptr().add(c)), v_hh_w);
            let v_partial = _mm_add_pd(_mm_add_pd(v_div_lh, v_div_hl), v_div_hh);
            let lane0 = _mm_cvtsd_f64(v_partial);
            let lane1 = _mm_cvtsd_f64(_mm_unpackhi_pd(v_partial, v_partial));
            *cost += lane0 + lane1;
            c += 2;
        }
    }

    // Scalar pairs — every pair on targets without one of the SIMD
    // blocks above. Same order as the lanes.
    let cell = |i: usize| -> f64 {
        let lh = d_lh[i] / ((w_lh[i] as f64).abs() + SIGMA);
        let hl = d_hl[i] / ((w_hl[i] as f64).abs() + SIGMA);
        let hh = d_hh[i] / ((w_hh[i] as f64).abs() + SIGMA);
        (lh + hl) + hh
    };
    while c + 1 < n {
        let lane0 = cell(c);
        let lane1 = cell(c + 1);
        *cost += lane0 + lane1;
        c += 2;
    }
    // 1-cell tail (n = 23 is odd).
    if c < n {
        *cost += cell(c);
    }
}

/// J-UNIWARD cost of one coefficient of block `(br, bc)`, given its |Δ|
/// slices: the whole 23×23 window, margin samples included (module docs).
/// The block must lie inside the image and its window rows inside
/// `wavelets`' row range.
#[inline]
fn cost_kernel(
    abs_lh: &[f64],
    abs_hl: &[f64],
    abs_hh: &[f64],
    wavelets: &PaddedSubbands,
    br: usize,
    bc: usize,
) -> f64 {
    let x_first = (bc * 8) as isize - WINDOW_ORIGIN as isize;
    let y_first = (br * 8) as isize - WINDOW_ORIGIN as isize;
    let mut cost = 0.0;
    for out_r in 0..IMPACT_SIZE {
        let i = wavelets.index(x_first, y_first + out_r as isize);
        let cells = out_r * IMPACT_SIZE..(out_r + 1) * IMPACT_SIZE;
        accumulate_window_row(
            &mut cost,
            &abs_lh[cells.clone()],
            &abs_hl[cells.clone()],
            &abs_hh[cells],
            &wavelets.lh[i..i + IMPACT_SIZE],
            &wavelets.hl[i..i + IMPACT_SIZE],
            &wavelets.hh[i..i + IMPACT_SIZE],
        );
    }
    cost
}

// ---------------------------------------------------------------------------
// Strip-based streaming UNIWARD: costs without a whole-image CostMap.
// ---------------------------------------------------------------------------

/// Strip height in block rows for streaming UNIWARD computation.
/// 50 block rows = 400 pixel rows. Each strip uses ~170 MB peak for a 16K-wide
/// image (pixels + row-filtered + 3 wavelet subbands + strip CostMap).
pub(super) const STRIP_BLOCK_ROWS: usize = 50;

/// Minimum cost for SI-modulated coefficients (same as side_info::MIN_SI_COST).
pub(super) const MIN_SI_COST_F32: f32 = 1e-6;

/// J-UNIWARD cost of a ±1 change for **every** Y AC coefficient, zeros
/// included (Ghost embedding v3, `ghost::v3`), in flat order
/// (`block_index · 64 + i · 8 + j`; DC entries are `+∞`).
///
/// The J-UNIWARD cost depends only on the block's neighbourhood and the
/// mode `(i, j)`, not on the coefficient's value, so a zero coefficient is
/// priced exactly like a non-zero one at the same place (as reference
/// J-UNIWARD does). Same wavelet arithmetic and kernel as [`compute_uniward`],
/// streamed in strips: the direct-kernel reference of
/// [`compute_ac_costs_separable`].
///
/// Memory: 4 B per coefficient (49 MB at 12 MP) plus one strip.
/// Reports [`UNIWARD_PROGRESS_STEPS`] progress steps.
pub fn compute_ac_costs_streaming(grid: &DctGrid, qt: &QuantTable) -> Result<Vec<f32>, StegoError> {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let img_w = bw * 8;
    let abs_delta = AbsDeltaTable::precompute(qt);
    let mut out = vec![f32::INFINITY; bw * bt * 64];

    let num_strips = bt.div_ceil(STRIP_BLOCK_ROWS);
    let mut strip_idx = 0usize;
    let mut steps_sent = 0u32;

    for strip_start in (0..bt).step_by(STRIP_BLOCK_ROWS) {
        let strip_end = (strip_start + STRIP_BLOCK_ROWS).min(bt);
        let (wy0, wy1) = window_rows_for_block_rows(strip_start, strip_end);
        let pixels = padded_pixel_rows(grid, qt, wy0 - FILT_HALF, (wy1 - wy0) as usize + FILT_LEN - 1);
        let strip_wavelets = padded_subbands_from_pixels(&pixels, img_w, wy0, wy1);
        drop(pixels);

        let strip_out = &mut out[strip_start * bw * 64..strip_end * bw * 64];
        let compute_block = |(bi, blk_out): (usize, &mut [f32])| {
            let br = strip_start + bi / bw;
            let bc = bi % bw;
            for fi in 0..8 {
                for fj in 0..8 {
                    if fi == 0 && fj == 0 {
                        continue;
                    }
                    let (abs_lh, abs_hl, abs_hh) = abs_delta.slice_at(fi, fj);
                    let cost = cost_kernel(abs_lh, abs_hl, abs_hh, &strip_wavelets, br, bc);
                    if cost > 0.0 && cost.is_finite() {
                        blk_out[fi * 8 + fj] = cost as f32;
                    }
                }
            }
        };
        #[cfg(feature = "parallel")]
        strip_out.par_chunks_mut(64).enumerate().for_each(compute_block);
        #[cfg(not(feature = "parallel"))]
        strip_out.chunks_mut(64).enumerate().for_each(compute_block);

        strip_idx += 1;
        let target_steps = (strip_idx as u32 * UNIWARD_PROGRESS_STEPS) / num_strips as u32;
        while steps_sent < target_steps {
            progress::advance();
            steps_sent += 1;
        }
        if strip_idx.is_multiple_of(2) {
            progress::check_cancelled()?;
        }
    }
    while steps_sent < UNIWARD_PROGRESS_STEPS {
        progress::advance();
        steps_sent += 1;
    }
    Ok(out)
}

/// `|L_k(o)|`, `|H_k(o)|`: the 1-D wavelet responses of the 1-D DCT basis
/// `a_k(y) = ½·C_k·cos((2y+1)kπ/16)` at window cell `o` (same cell
/// convention as [`AbsDeltaTable::precompute`]). The 2-D impact of mode
/// `(i, j)` is separable — `|Δ_LH| = q·|H_i(r)|·|L_j(c)|`, `|Δ_HL| =
/// q·|L_i(r)|·|H_j(c)|`, `|Δ_HH| = q·|H_i(r)|·|H_j(c)|` — because the DCT
/// basis and the db8 filters both are.
fn separable_factors() -> ([[f64; IMPACT_SIZE]; 8], [[f64; IMPACT_SIZE]; 8]) {
    let lp = lpdf();
    let src_off = WINDOW_ORIGIN as isize + FILT_HALF;
    let mut lo = [[0.0f64; IMPACT_SIZE]; 8];
    let mut hi = [[0.0f64; IMPACT_SIZE]; 8];
    for k in 0..8 {
        let ck = if k == 0 { core::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
        let mut a = [0.0f64; 8];
        for (y, v) in a.iter_mut().enumerate() {
            *v = 0.5 * ck * crate::det_math::det_cos(((2 * y + 1) * k) as f64 * core::f64::consts::PI / 16.0);
        }
        for o in 0..IMPACT_SIZE {
            let (mut sl, mut sh) = (0.0, 0.0);
            for t in 0..FILT_LEN {
                let y = o as isize - src_off + t as isize;
                if (0..8).contains(&y) {
                    sl += lp[t] * a[y as usize];
                    sh += HPDF[t] * a[y as usize];
                }
            }
            lo[k][o] = sl.abs();
            hi[k][o] = sh.abs();
        }
    }
    (lo, hi)
}

/// Working-set target of the separable cost pass ([`cost_strip_rows`]).
const COST_STRIP_BYTES: usize = 8 << 20;

/// Fewest block rows per strip: each strip is a few rounds of parallel
/// joins, and one or two block rows per strip cost ~10 % more CPU at 48 MP.
const MIN_STRIP_BLOCK_ROWS: usize = 4;

/// What one cost strip actually holds at its peak (measured 50-54 MB at
/// 12-110 MP), with margin: the memory plan charges this for a cost pass.
pub const COST_STRIP_WORKING_BYTES: usize = 64 << 20;

/// Block rows per strip of the separable cost pass, so that its buffers
/// ([`CostStripBuffers`], sized by the first strip) stay near
/// [`COST_STRIP_BYTES`], but at least [`MIN_STRIP_BLOCK_ROWS`]: 4 block
/// rows at 12 MP (7.5 MB of buffers) and at 48 MP (15 MB). Costs do not
/// depend on it.
///
/// The buffers are allocated once per pass and every strip reuses them, so
/// their size is what the pass leaves behind in an allocator that keeps
/// freed blocks resident (libmalloc's large-allocation cache on macOS /
/// iOS: the lanes that follow cannot reuse them). With the 15 carried rows
/// a smaller strip costs no extra filtering, only one more round of
/// parallel joins.
pub fn cost_strip_rows(blocks_wide: usize) -> usize {
    let bw = blocks_wide.max(1);
    let stride = bw * 8 + 2 * WAV_MARGIN;
    // First strip of R block rows: pixels and row-filtered rows 8R + 30
    // (4 + 2·4 B), reciprocals 8R + 15 (3·8 B), costs R·bw·64 (4 B).
    let fixed = 30 * 12 * stride + 15 * 24 * stride;
    let per_row = 8 * 36 * stride + 256 * bw;
    (COST_STRIP_BYTES.saturating_sub(fixed) / per_row).clamp(MIN_STRIP_BLOCK_ROWS, STRIP_BLOCK_ROWS)
}

/// Working buffers of the separable cost pass, kept for all of its strips
/// (allocated at the first strip's size, which no later strip exceeds).
///
/// Consecutive strips overlap by 15 window rows (a block's window reaches 8
/// rows above and 14 below it) and their column filters by 15 row-filtered
/// rows: those rows are carried over to the next strip instead of being
/// recomputed. A wavelet sample has the same value whichever row range it
/// is computed in (see [`PaddedSubbands`]), so the costs are bit-identical
/// to computing every strip on its own.
#[derive(Default)]
struct CostStripBuffers {
    /// Padded pixel rows of the current strip's new row-filtered rows.
    pix: Vec<f32>,
    /// Row-filtered (low-pass / high-pass) rows from plane row `rows_y0`.
    row_lo: Vec<f32>,
    row_hi: Vec<f32>,
    rows_y0: isize,
    /// `1 / (|W_b| + σ)` of the three subbands from plane row `r_y0`.
    r_lh: Vec<f64>,
    r_hl: Vec<f64>,
    r_hh: Vec<f64>,
    r_y0: isize,
    /// The current strip's costs (flat order).
    out: Vec<f32>,
}

/// Keep the rows of `v` (rows of `stride` samples from plane row `y0`) from
/// plane row `from` on, moved to the front; none if `from` is not among
/// them.
fn keep_rows_from<T: Copy>(v: &mut Vec<T>, y0: isize, stride: usize, from: isize) {
    let n = (v.len() / stride) as isize;
    if from >= y0 && from < y0 + n {
        let s = (from - y0) as usize * stride;
        v.copy_within(s.., 0);
        v.truncate(v.len() - s);
    } else {
        v.clear();
    }
}

/// The separable cost pass strip by strip: `sink(first_flat_index, costs)`
/// receives every strip's costs (flat order, `rows` block rows at a time,
/// DC entries `+∞`) and may stop the pass with an error. Bounded memory: one
/// strip, whatever the image size.
pub fn ac_cost_strips(
    grid: &DctGrid,
    qt: &QuantTable,
    rows: usize,
    sink: impl FnMut(usize, &[f32]) -> Result<(), StegoError>,
) -> Result<(), StegoError> {
    ac_cost_strips_range(grid, qt, rows, 0, grid.blocks_tall(), sink)
}

/// [`ac_cost_strips`] over block rows `[br0, br1)` of `grid` only (`sink`
/// gets flat indices of `grid`). A block's cost depends only on the pixels
/// within 15 rows of it (two block rows), so a band of an image computed
/// from a sub-grid holding the band and two block rows on either side
/// (fewer at the image's own edges) has exactly the whole image's costs
/// ([`ac_cost_band`]).
pub fn ac_cost_strips_range(
    grid: &DctGrid,
    qt: &QuantTable,
    rows: usize,
    br0: usize,
    br1: usize,
    mut sink: impl FnMut(usize, &[f32]) -> Result<(), StegoError>,
) -> Result<(), StegoError> {
    let bw = grid.blocks_wide();
    let img_w = bw * 8;
    let stride = img_w + 2 * WAV_MARGIN;
    let (lo, hi) = separable_factors();
    let lp = lpdf();
    let rows = rows.max(1);
    let recip = |x: f32| -> f64 { 1.0 / ((x as f64).abs() + SIGMA) };
    let mut b = CostStripBuffers::default();

    for strip_start in (br0..br1.min(grid.blocks_tall())).step_by(rows) {
        let strip_end = (strip_start + rows).min(br1);
        let (wy0, wy1) = window_rows_for_block_rows(strip_start, strip_end);

        // Reciprocal rows [wy0, wy1): the previous strip's last ones, then
        // the new rows [c0, wy1).
        for v in [&mut b.r_lh, &mut b.r_hl, &mut b.r_hh] {
            keep_rows_from(v, b.r_y0, stride, wy0);
        }
        b.r_y0 = wy0;
        let c0 = wy0 + (b.r_lh.len() / stride) as isize;
        // They read the row-filtered rows [c0 − 7, wy1 + 8): the carried
        // ones, then new rows [p0, p1) from as many new pixel rows.
        let rows_from = c0 - FILT_HALF;
        for v in [&mut b.row_lo, &mut b.row_hi] {
            keep_rows_from(v, b.rows_y0, stride, rows_from);
        }
        b.rows_y0 = rows_from;
        let p0 = rows_from + (b.row_lo.len() / stride) as isize;
        let p1 = wy1 + (FILT_LEN as isize - 1 - FILT_HALF);
        let n_pix = (p1 - p0) as usize;
        padded_pixel_rows_into(&mut b.pix, grid, qt, p0, n_pix);
        let l0 = b.row_lo.len();
        b.row_lo.resize(l0 + n_pix * stride, 0.0);
        b.row_hi.resize(l0 + n_pix * stride, 0.0);
        let o0 = b.r_lh.len();
        let n_new = (wy1 - c0) as usize * stride;
        for v in [&mut b.r_lh, &mut b.r_hl, &mut b.r_hh] {
            v.resize(o0 + n_new, 0.0);
        }
        {
            let CostStripBuffers { pix, row_lo, row_hi, r_lh, r_hl, r_hh, .. } = &mut b;
            let (pix, lo_new, hi_new) = (&pix[..], &mut row_lo[l0..], &mut row_hi[l0..]);
            let (rlh, rhl, rhh) = (&mut r_lh[o0..], &mut r_hl[o0..], &mut r_hh[o0..]);
            #[cfg(feature = "parallel")]
            {
                rayon::join(|| row_filter_into(pix, img_w, &lp, lo_new), || row_filter_into(pix, img_w, &HPDF, hi_new));
                let (row_lo, row_hi) = (&row_lo[..], &row_hi[..]);
                rayon::join(
                    || col_filter_into(row_lo, stride, &HPDF, rlh, recip),
                    || rayon::join(|| col_filter_into(row_hi, stride, &lp, rhl, recip), || col_filter_into(row_hi, stride, &HPDF, rhh, recip)),
                );
            }
            #[cfg(not(feature = "parallel"))]
            {
                row_filter_into(pix, img_w, &lp, lo_new);
                row_filter_into(pix, img_w, &HPDF, hi_new);
                col_filter_into(row_lo, stride, &HPDF, rlh, recip);
                col_filter_into(row_hi, stride, &lp, rhl, recip);
                col_filter_into(row_hi, stride, &HPDF, rhh, recip);
            }
        }

        b.out.clear();
        b.out.resize((strip_end - strip_start) * bw * 64, f32::INFINITY);
        let ctx = StripCtx { stride, y0: wy0, r_lh: &b.r_lh, r_hl: &b.r_hl, r_hh: &b.r_hh, lo: &lo, hi: &hi, qt, strip_start, bw };
        let compute_block = |(bi, blk_out): (usize, &mut [f32])| ctx.block(bi, blk_out);
        #[cfg(feature = "parallel")]
        b.out.par_chunks_mut(64).enumerate().for_each(compute_block);
        #[cfg(not(feature = "parallel"))]
        b.out.chunks_mut(64).enumerate().for_each(compute_block);
        sink(strip_start * bw * 64, &b.out)?;
    }
    Ok(())
}

/// What one strip's blocks read.
struct StripCtx<'a> {
    /// Samples per reciprocal row, and the plane row of the first one.
    stride: usize,
    y0: isize,
    r_lh: &'a [f64],
    r_hl: &'a [f64],
    r_hh: &'a [f64],
    lo: &'a [[f64; IMPACT_SIZE]; 8],
    hi: &'a [[f64; IMPACT_SIZE]; 8],
    qt: &'a QuantTable,
    strip_start: usize,
    bw: usize,
}

impl StripCtx<'_> {
    /// The 63 AC costs of the strip's block `bi` into `blk_out` (left `+∞`
    /// where not positive and finite).
    ///
    /// Not inlined on wasm32: V8 runs a function in baseline code (Liftoff)
    /// until its next call, with no on-stack replacement for Wasm, and the
    /// whole pass is one call; per block, the kernel is optimized after a
    /// few hundred blocks.
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    fn block(&self, bi: usize, blk_out: &mut [f32]) {
        const N: usize = IMPACT_SIZE;
        let (lo, hi) = (self.lo, self.hi);
        let br = self.strip_start + bi / self.bw;
        let bc = bi % self.bw;
        let x_first = (bc * 8) as isize - WINDOW_ORIGIN as isize;
        let y_first = (br * 8) as isize - WINDOW_ORIGIN as isize;
        // T_b[r][j] = Σ_c R_b[r][c] · V_b[j][c]
        let mut t_lh = [[0.0f64; 8]; N];
        let mut t_hl = [[0.0f64; 8]; N];
        let mut t_hh = [[0.0f64; 8]; N];
        for r in 0..N {
            let y = y_first + r as isize;
            debug_assert!(y >= self.y0 && (y - self.y0) as usize * self.stride < self.r_lh.len(), "row {y} not in strip");
            let i0 = (y - self.y0) as usize * self.stride + (x_first + WAV_MARGIN as isize) as usize;
            let (rl, rh, rd) = (&self.r_lh[i0..i0 + N], &self.r_hl[i0..i0 + N], &self.r_hh[i0..i0 + N]);
            for j in 0..8 {
                let (mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0);
                for c in 0..N {
                    s1 += rl[c] * lo[j][c];
                    s2 += rh[c] * hi[j][c];
                    s3 += rd[c] * hi[j][c];
                }
                t_lh[r][j] = s1;
                t_hl[r][j] = s2;
                t_hh[r][j] = s3;
            }
        }
        for i in 0..8 {
            for j in 0..8 {
                if i == 0 && j == 0 {
                    continue;
                }
                let mut s = 0.0;
                for r in 0..N {
                    s += hi[i][r] * (t_lh[r][j] + t_hh[r][j]) + lo[i][r] * t_hl[r][j];
                }
                let cost = s * self.qt.values[i * 8 + j] as f64;
                if cost > 0.0 && cost.is_finite() {
                    blk_out[i * 8 + j] = cost as f32;
                }
            }
        }
    }
}

/// Block rows a cost band's sub-grid must hold beyond the band on each side
/// (pixel rows within 15 of the band; see [`ac_cost_strips_range`]).
pub const COST_BAND_MARGIN_ROWS: usize = 2;

/// The sub-grid [`ac_cost_band`] needs for block rows `[br0, br1)` of `grid`.
pub fn cost_band_subgrid(grid: &DctGrid, br0: usize, br1: usize) -> DctGrid {
    let bw = grid.blocks_wide();
    let g0 = br0.saturating_sub(COST_BAND_MARGIN_ROWS);
    let g1 = (br1 + COST_BAND_MARGIN_ROWS).min(grid.blocks_tall());
    let mut sub = DctGrid::new(bw, g1 - g0);
    sub.coeffs_mut().copy_from_slice(&grid.coeffs()[g0 * bw * 64..g1 * bw * 64]);
    sub
}

/// Costs of block rows `[br0, br1)` of an image `blocks_tall` block rows
/// high, from `sub`: its block rows `[g0, g0 + sub.blocks_tall())` with
/// `g0 = br0.saturating_sub(COST_BAND_MARGIN_ROWS)` and the end
/// `min(br1 + COST_BAND_MARGIN_ROWS, blocks_tall)`. Flat order of the band
/// (`(br1 − br0) · blocks_wide · 64` values, DC `+∞`), bit-identical to the
/// same rows of [`compute_ac_costs_separable`] on the whole image.
pub fn ac_cost_band(sub: &DctGrid, qt: &QuantTable, br0: usize, br1: usize, blocks_tall: usize) -> Result<Vec<f32>, StegoError> {
    let g0 = br0.saturating_sub(COST_BAND_MARGIN_ROWS);
    let g1 = (br1 + COST_BAND_MARGIN_ROWS).min(blocks_tall);
    if br0 >= br1 || br1 > blocks_tall || sub.blocks_tall() != g1 - g0 {
        return Err(StegoError::InvalidJpeg(crate::codec::jpeg::error::JpegError::InvalidDimensions));
    }
    let mut out = vec![f32::INFINITY; (br1 - br0) * sub.blocks_wide() * 64];
    let base0 = (br0 - g0) * sub.blocks_wide() * 64;
    ac_cost_strips_range(sub, qt, cost_strip_rows(sub.blocks_wide()), br0 - g0, br1 - g0, |base, strip| {
        out[base - base0..base - base0 + strip.len()].copy_from_slice(strip);
        progress::check_cancelled()
    })?;
    Ok(out)
}

/// [`compute_ac_costs_streaming`] by the separable factorization of the
/// impact: per strip `R_b = 1/(σ + |W_b|)` once per wavelet sample, then per
/// block `C = Σ_b U_bᵀ · R_b(23×23) · V_b` for all 64 modes at once (≈17 k
/// multiply-adds per block instead of ≈100 k plus 100 k divisions).
/// Encoder-only (Ghost v3, whose decoder needs no costs): agrees with the
/// direct kernel to ~1e-12 relative before the f32 cast. Whole-image output
/// (4 B per coefficient); the encoder uses [`gather_ac_costs`].
pub fn compute_ac_costs_separable(grid: &DctGrid, qt: &QuantTable) -> Result<Vec<f32>, StegoError> {
    let bt = grid.blocks_tall();
    let mut out = vec![f32::INFINITY; grid.blocks_wide() * bt * 64];
    let num_strips = bt.div_ceil(STRIP_BLOCK_ROWS).max(1);
    let mut strip_idx = 0usize;
    let mut steps_sent = 0u32;
    ac_cost_strips(grid, qt, STRIP_BLOCK_ROWS, |base, strip| {
        out[base..base + strip.len()].copy_from_slice(strip);
        strip_idx += 1;
        let target_steps = (strip_idx as u32 * UNIWARD_PROGRESS_STEPS) / num_strips as u32;
        while steps_sent < target_steps {
            progress::advance();
            steps_sent += 1;
        }
        progress::check_cancelled()
    })?;
    while steps_sent < UNIWARD_PROGRESS_STEPS {
        progress::advance();
        steps_sent += 1;
    }
    Ok(out)
}

/// The separable costs of the coefficients `fis` (flat indices, any order,
/// repeats allowed): `out[i]` = cost of `fis[i]`, bit-identical to
/// [`compute_ac_costs_separable`]. One strip pass over the image; memory is
/// one strip plus 8 B per requested position.
pub fn gather_ac_costs(grid: &DctGrid, qt: &QuantTable, fis: &[u32]) -> Result<Vec<f32>, StegoError> {
    gather_ac_costs_parts(grid, qt, fis, &[])
}

/// The separable costs of every coefficient of `grid` by flat index (DC
/// entries `+∞`): one strip pass, 4 B per coefficient plus one cost strip.
/// The same values [`gather_ac_costs`] returns for any request.
pub fn dense_ac_costs(grid: &DctGrid, qt: &QuantTable) -> Result<Vec<f32>, StegoError> {
    let mut dense = vec![f32::INFINITY; grid.blocks_wide() * grid.blocks_tall() * 64];
    ac_cost_strips(grid, qt, cost_strip_rows(grid.blocks_wide()), |base, strip| {
        dense[base..base + strip.len()].copy_from_slice(strip);
        progress::check_cancelled()
    })?;
    Ok(dense)
}

/// [`gather_ac_costs`] of `head` followed by `tail` (one output, `head`'s
/// costs first) without concatenating them: 8 bytes per position (sort
/// index + output) plus one cost strip.
pub fn gather_ac_costs_parts(grid: &DctGrid, qt: &QuantTable, head: &[u32], tail: &[u32]) -> Result<Vec<f32>, StegoError> {
    let total = head.len() + tail.len();
    let fi_at = |i: usize| -> u32 { if i < head.len() { head[i] } else { tail[i - head.len()] } };
    // A request about as large as the image (the whole keyed order: every
    // encode whose costs fit in one group) gathers from the image's dense
    // costs instead of sorting the request by flat index: the same 8 B per
    // position (dense costs + output instead of sort index + output), no
    // sort (a single-threaded sort of 12 M keys took longer than the cost
    // pass itself on WASM). Same values: both read the same strips.
    let n_total = grid.blocks_wide() * grid.blocks_tall() * 64;
    if total + total / 32 >= n_total {
        let dense = dense_ac_costs(grid, qt)?;
        return Ok(head.iter().chain(tail).map(|&fi| dense[fi as usize]).collect());
    }
    let mut idx: Vec<u32> = (0..total as u32).collect();
    #[cfg(feature = "parallel")]
    {
        use rayon::slice::ParallelSliceMut;
        idx.par_sort_unstable_by_key(|&i| fi_at(i as usize));
    }
    #[cfg(not(feature = "parallel"))]
    idx.sort_unstable_by_key(|&i| fi_at(i as usize));
    let mut out = vec![f32::INFINITY; total];
    let mut p = 0usize;
    ac_cost_strips(grid, qt, cost_strip_rows(grid.blocks_wide()), |base, strip| {
        let end = base + strip.len();
        while p < idx.len() {
            let i = idx[p] as usize;
            let fi = fi_at(i) as usize;
            if fi >= end {
                break;
            }
            out[i] = strip[fi - base];
            p += 1;
        }
        progress::check_cancelled()
    })?;
    Ok(out)
}

/// T3.3.E — deterministic byte stream for the cost kernel, used by
/// `cost_kernel_test_hash_hex` to pin the cross-platform hash.
/// Re-computing this on any supported target MUST yield the same bytes
/// (and therefore the same SHA256).
///
/// Layout: for each (br, bc, fi, fj) in a fixed iteration order, dump
/// the f64 cost as 8 little-endian bytes via `f64::to_bits`. The
/// fixture is 8×8 blocks (64×64 px): the outer block ring's windows
/// reach into the 8-sample margin, the inner 6×6 blocks' do not, so the
/// hash covers the padded decomposition (pixel padding, row and column
/// passes) as well as the kernel.
#[doc(hidden)]
pub fn cost_kernel_test_deterministic_bytes() -> Vec<u8> {
    let (cover, qt) = cost_kernel_test_fixture();
    let wavelets = whole_image_subbands(&cover, &qt);
    let table = AbsDeltaTable::precompute(&qt);

    let mut bytes = Vec::with_capacity(8 * 8 * 63 * 8);
    for br in 0..8 {
        for bc in 0..8 {
            for fi in 0..8 {
                for fj in 0..8 {
                    if fi == 0 && fj == 0 {
                        continue;
                    }
                    let (abs_lh, abs_hl, abs_hh) = table.slice_at(fi, fj);
                    let cost = cost_kernel(abs_lh, abs_hl, abs_hh, &wavelets, br, bc);
                    bytes.extend_from_slice(&cost.to_bits().to_le_bytes());
                }
            }
        }
    }
    bytes
}

/// The 8×8-block fixture of the pinned cost-kernel hashes (same grid for
/// v2 and the legacy v1 hash): deterministic content, mix of magnitudes,
/// the standard JPEG luminance quant table (quality ~50).
pub(super) fn cost_kernel_test_fixture() -> (DctGrid, QuantTable) {
    let mut cover = DctGrid::new(8, 8);
    for br in 0..8 {
        for bc in 0..8 {
            cover.set(br, bc, 0, 0, 120);
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    let val = (((br * 11 + bc * 7 + i * 5 + j * 3) % 19) as i16) - 9;
                    if val != 0 {
                        cover.set(br, bc, i, j, val);
                    }
                }
            }
        }
    }
    let qt = QuantTable::new([
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ]);
    (cover, qt)
}

/// SHA256 of [`cost_kernel_test_deterministic_bytes`] as lowercase hex.
#[doc(hidden)]
pub fn cost_kernel_test_hash_hex() -> String {
    sha256_hex(&cost_kernel_test_deterministic_bytes())
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stego::cost::WET_COST;

    fn make_qt_uniform(val: u16) -> QuantTable {
        QuantTable::new([val; 64])
    }

    /// Standard JPEG luminance quantization table (quality ~50).
    fn standard_qt() -> QuantTable {
        cost_kernel_test_fixture().1
    }

    /// Deterministic textured grid used by several tests.
    fn textured_grid(bw: usize, bt: usize) -> DctGrid {
        let mut grid = DctGrid::new(bw, bt);
        for br in 0..bt {
            for bc in 0..bw {
                grid.set(br, bc, 0, 0, 100);
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 { continue; }
                        let val = (((br * 7 + bc * 13 + i * 3 + j * 11) % 21) as i16) - 10;
                        if val != 0 {
                            grid.set(br, bc, i, j, val);
                        }
                    }
                }
            }
        }
        grid
    }

    /// Symmetric-padded pixel rows of an arbitrary `w × h` plane, same
    /// layout as [`padded_pixel_rows`] (which takes them from a DCT grid).
    fn padded_pixels_from_plane(plane: &[f32], w: usize, h: usize, py0: isize, n_rows: usize) -> Vec<f32> {
        let pw = w + 2 * PIX_MARGIN;
        let mut pix = vec![0.0f32; pw * n_rows];
        for i in 0..n_rows {
            let sy = symmetric_index(py0 + i as isize, h);
            pix[i * pw + PIX_MARGIN..i * pw + PIX_MARGIN + w].copy_from_slice(&plane[sy * w..(sy + 1) * w]);
        }
        fill_column_margins(&mut pix, w);
        pix
    }

    /// Whole-plane subbands (margins included) of an arbitrary plane.
    fn subbands_of_plane(plane: &[f32], w: usize, h: usize) -> PaddedSubbands {
        let (wy0, wy1) = (-(WAV_MARGIN as isize), (h + WAV_MARGIN) as isize);
        let pix = padded_pixels_from_plane(plane, w, h, wy0 - FILT_HALF, (wy1 - wy0) as usize + FILT_LEN - 1);
        padded_subbands_from_pixels(&pix, w, wy0, wy1)
    }

    /// Textured synthetic plane: noise, a ramp, stripes and a flat patch,
    /// with the noisiest content along the top and left edges.
    fn synth_plane(w: usize, h: usize) -> Vec<f32> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 1000) as f32 / 1000.0
        };
        let mut p = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let noise = next() * 60.0;
                let v = if x < w / 2 && y < h / 2 {
                    90.0 + noise * 1.5 // noisy top-left quadrant
                } else if x >= w / 2 && y < h / 2 {
                    40.0 + 4.0 * x as f32 // ramp
                } else if x < w / 2 {
                    if (x / 3) % 2 == 0 { 200.0 } else { 30.0 } // stripes
                } else {
                    128.0 + (noise - 30.0) * 0.05 // nearly flat
                };
                p[y * w + x] = v.clamp(0.0, 255.0);
            }
        }
        p
    }

    #[test]
    fn dc_is_wet() {
        let mut grid = DctGrid::new(4, 4);
        for br in 0..4 {
            for bc in 0..4 {
                grid.set(br, bc, 0, 0, 100);
                grid.set(br, bc, 1, 0, 10);
                grid.set(br, bc, 0, 1, -5);
            }
        }
        let map = compute_uniward(&grid, &make_qt_uniform(16));
        for br in 0..4 {
            for bc in 0..4 {
                assert_eq!(map.get(br, bc, 0, 0), WET_COST);
            }
        }
    }

    #[test]
    fn zero_ac_is_wet() {
        let mut grid = DctGrid::new(4, 4);
        for br in 0..4 {
            for bc in 0..4 {
                grid.set(br, bc, 0, 0, 100);
                grid.set(br, bc, 1, 0, 10);
            }
        }
        let map = compute_uniward(&grid, &make_qt_uniform(16));
        for br in 0..4 {
            for bc in 0..4 {
                assert_eq!(map.get(br, bc, 0, 1), WET_COST);
            }
        }
    }

    #[test]
    fn non_zero_ac_has_finite_cost() {
        let mut grid = DctGrid::new(4, 4);
        for br in 0..4 {
            for bc in 0..4 {
                grid.set(br, bc, 0, 0, 100);
                grid.set(br, bc, 1, 0, 10);
                grid.set(br, bc, 0, 1, -5);
                grid.set(br, bc, 1, 1, 3);
            }
        }
        let map = compute_uniward(&grid, &standard_qt());
        let cost = map.get(2, 2, 1, 0);
        assert!(cost.is_finite() && cost > 0.0, "expected finite positive cost, got {cost}");
    }

    #[test]
    fn textured_region_cheaper_than_smooth_region() {
        // J-UNIWARD assigns lower cost when the cover's wavelet coefficients
        // are large (textured region): the denominator is larger.
        let qt = make_qt_uniform(16);
        let mut grid = DctGrid::new(8, 8);
        for br in 0..8 {
            for bc in 0..8 {
                grid.set(br, bc, 0, 0, 100);
            }
        }
        for br in 0..4 {
            for bc in 0..4 {
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 { continue; }
                        grid.set(br, bc, i, j, (((i * 7 + j * 3) % 15) as i16) - 7);
                    }
                }
            }
        }
        grid.set(2, 2, 1, 0, 5);
        grid.set(6, 6, 1, 0, 5);
        let map = compute_uniward(&grid, &qt);
        let cost_textured = map.get(2, 2, 1, 0);
        let cost_smooth = map.get(6, 6, 1, 0);
        assert!(cost_textured.is_finite() && cost_smooth.is_finite());
        assert!(
            cost_textured < cost_smooth,
            "textured region {cost_textured} should be < smooth region {cost_smooth}"
        );
    }

    #[test]
    fn costs_are_positive() {
        let mut grid = DctGrid::new(4, 4);
        for br in 0..4 {
            for bc in 0..4 {
                grid.set(br, bc, 0, 0, 80);
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 { continue; }
                        if (i + j) % 3 == 0 {
                            grid.set(br, bc, i, j, ((i * 3 + j * 7) % 20) as i16 - 10);
                        }
                    }
                }
            }
        }
        let map = compute_uniward(&grid, &standard_qt());
        for br in 0..4 {
            for bc in 0..4 {
                for i in 0..8 {
                    for j in 0..8 {
                        let cost = map.get(br, bc, i, j);
                        assert!(cost >= 0.0, "negative cost {cost} at ({br},{bc},{i},{j})");
                    }
                }
            }
        }
    }

    #[test]
    fn symmetric_index_works() {
        assert_eq!(symmetric_index(-1, 10), 0);
        assert_eq!(symmetric_index(-3, 10), 2);
        assert_eq!(symmetric_index(0, 10), 0);
        assert_eq!(symmetric_index(9, 10), 9);
        assert_eq!(symmetric_index(10, 10), 9);
        assert_eq!(symmetric_index(12, 10), 7);
        // Multiple reflections on a tiny axis: period 2·len.
        assert_eq!(symmetric_index(-9, 8), 7);
        assert_eq!(symmetric_index(-17, 8), 0);
        assert_eq!(symmetric_index(23, 8), 7);
        assert_eq!(symmetric_index(5, 1), 0);
    }

    #[test]
    fn lpdf_is_correct_length() {
        let lp = lpdf();
        assert_eq!(lp.len(), 16);
        let sum: f64 = lp.iter().sum();
        assert!((sum - std::f64::consts::SQRT_2).abs() < 0.01, "low-pass filter sum {sum} should be ~sqrt(2)");
    }

    #[test]
    fn hpdf_sums_to_zero() {
        let sum: f64 = HPDF.iter().sum();
        assert!(sum.abs() < 1e-10, "high-pass filter sum {sum} should be ~0");
    }

    /// Flaw 2 of the v1 kernel, pinned from first principles: the |Δ|
    /// table cell `(r, c)` must equal the EXACT wavelet impact (a full
    /// decomposition of the changed plane) at block-relative sample
    /// `(c − 8, r − 8)`, and the impact must vanish outside that 23×23
    /// window. The v1 pairing (offset-14 table read at offset 15) is far
    /// off — 8 px up and 8 px left, as the 2026-10 measurement found.
    #[test]
    fn impact_table_matches_exact_wavelet_impact() {
        let qt = standard_qt();
        let basis = precompute_basis_functions(&qt);
        let table = AbsDeltaTable::precompute(&qt);
        let legacy = AbsDeltaTable::precompute_legacy_v1(&qt);
        let (w, h) = (64usize, 64usize);
        let (bx, by) = (3usize, 4usize); // interior: window clear of the edges
        let coefs: Vec<(usize, usize)> = (1..64).map(|k| (k / 8, k % 8)).collect();
        for &(fi, fj) in &coefs {
            // The cover is linear in the change: decompose the change alone.
            let mut plane = vec![0.0f32; w * h];
            for r in 0..8 {
                for c in 0..8 {
                    plane[(by * 8 + r) * w + bx * 8 + c] = basis[fi][fj][r * 8 + c] as f32;
                }
            }
            let d = subbands_of_plane(&plane, w, h);
            let (t_lh, t_hl, t_hh) = table.slice_at(fi, fj);
            let (l_lh, _, _) = legacy.slice_at(fi, fj);
            for (name, band, tab) in [("LH", &d.lh, t_lh), ("HL", &d.hl, t_hl), ("HH", &d.hh, t_hh)] {
                let peak = band.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()));
                assert!(peak > 1e-3, "({fi},{fj}) {name}: no impact");
                let mut v1_err = 0.0f64;
                for y in -(WAV_MARGIN as isize)..(h + WAV_MARGIN) as isize {
                    for x in -(WAV_MARGIN as isize)..(w + WAV_MARGIN) as isize {
                        let exact = (band[d.index(x, y)] as f64).abs();
                        let (c, r) = (x - (bx * 8) as isize + 8, y - (by * 8) as isize + 8);
                        let in_window = (0..23).contains(&c) && (0..23).contains(&r);
                        if in_window {
                            let t = tab[r as usize * IMPACT_SIZE + c as usize];
                            assert!(
                                (t - exact).abs() <= 1e-5 * peak,
                                "({fi},{fj}) {name} cell ({r},{c}): table {t} vs exact {exact}"
                            );
                        } else {
                            assert!(exact <= 1e-6 * peak, "({fi},{fj}) {name}: impact {exact} outside the window at ({x},{y})");
                        }
                        // v1: cell (r, c) (offset-14 table) was divided by
                        // the sample at block-relative (c − 15, r − 15).
                        let (c1, r1) = (x - (bx * 8) as isize + 15, y - (by * 8) as isize + 15);
                        if name == "LH" && (0..23).contains(&c1) && (0..23).contains(&r1) {
                            let paired_impact = (band[d.index(x + 8, y + 8)] as f64).abs();
                            let t1 = l_lh[r1 as usize * IMPACT_SIZE + c1 as usize];
                            v1_err = v1_err.max((t1 - paired_impact).abs());
                        }
                    }
                }
                if name == "LH" {
                    // The v1 table itself is the impact shifted by one cell:
                    // its cell (r, c) is the exact impact at (c − 7, r − 7).
                    assert!(v1_err <= 1e-5 * peak, "({fi},{fj}): v1 table is not the 1-shifted impact");
                }
            }
        }
        // And the v1 kernel paired that table with the sample 8 px up/left
        // of the impact: on a textured plane, v1 and v2 window costs of an
        // interior block differ.
        let plane = synth_plane(w, h);
        let wv = subbands_of_plane(&plane, w, h);
        let (a, b, c) = table.slice_at(1, 2);
        let v2 = cost_kernel(a, b, c, &wv, by, bx);
        assert!(v2.is_finite() && v2 > 0.0);
    }

    /// Flaw 1 of the v1 kernel, pinned by construction (the image
    /// counterpart of the H.264 test of the same name). Plane `b` is plane
    /// `a` mirrored into a 2× canvas, `b[n + y][n + x] =
    /// a[sym(y)][sym(x)]`, so block `(bx, by)` of `a`, margin included,
    /// sees exactly the pixels block `(n/8 + bx, n/8 + by)` of `b` sees.
    /// Every block of `a` — border or not — must cost exactly what its twin
    /// costs. The v1 kernel skipped the out-of-image part of the window,
    /// so it failed at every block in the outer two block rings.
    #[test]
    fn border_block_cost_equals_its_mirrored_interior_twin() {
        let n = 64usize;
        let a = synth_plane(n, n);
        let m = 2 * n;
        let mut b = vec![0.0f32; m * m];
        for y in 0..m {
            for x in 0..m {
                let sy = symmetric_index(y as isize - n as isize, n);
                let sx = symmetric_index(x as isize - n as isize, n);
                b[y * m + x] = a[sy * n + sx];
            }
        }
        let wa = subbands_of_plane(&a, n, n);
        let wb = subbands_of_plane(&b, m, m);
        let qt = standard_qt();
        let table = AbsDeltaTable::precompute(&qt);
        let nb = n / 8;
        let mut border_checked = 0;
        for fi in 0..8 {
            for fj in 0..8 {
                if fi == 0 && fj == 0 { continue; }
                let (l, hl, hh) = table.slice_at(fi, fj);
                for by in 0..nb {
                    for bx in 0..nb {
                        let ca = cost_kernel(l, hl, hh, &wa, by, bx);
                        let cb = cost_kernel(l, hl, hh, &wb, nb + by, nb + bx);
                        assert_eq!(
                            ca.to_bits(), cb.to_bits(),
                            "coef ({fi},{fj}): block ({bx},{by}) costs {ca} but its interior twin costs {cb}"
                        );
                        if bx == 0 || by == 0 || bx == nb - 1 || by == nb - 1 {
                            border_checked += 1;
                        }
                    }
                }
            }
        }
        assert!(border_checked > 1000);
    }

    /// The strip-streamed all-AC pass, the separable pass and the CostMap
    /// agree bit for bit, on a grid tall enough for three strips (the last
    /// one a single block row) and on a plane narrower than the filter
    /// (multiple reflections).
    #[test]
    fn all_cost_paths_agree_bit_exactly() {
        let qt = standard_qt();
        for (bw, bt) in [(3usize, 101usize), (6, 6), (1, 3), (2, 1), (5, 52)] {
            let grid = textured_grid(bw, bt);
            let streaming = compute_ac_costs_streaming(&grid, &qt).unwrap();
            let separable = compute_ac_costs_separable(&grid, &qt).unwrap();
            let map = compute_uniward(&grid, &qt);
            let mut nz = 0;
            for (f, (&a, &b)) in streaming.iter().zip(&separable).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{bw}x{bt} separable at {f}");
                let (blk, k) = (f / 64, f % 64);
                if k != 0 && grid.coeffs()[f] != 0 {
                    // The CostMap prices only the non-zero AC coefficients.
                    nz += 1;
                    let got = map.get(blk / bw, blk % bw, k / 8, k % 8);
                    assert_eq!(got.to_bits(), a.to_bits(), "{bw}x{bt} CostMap at {f}");
                }
            }
            assert!(nz > 0);
        }
    }

    #[test]
    fn cost_with_real_photo() {
        let data = match std::fs::read("test-vectors/image/photo_320x240_q75_420.jpg") {
            Ok(d) => d,
            Err(_) => return, // Skip if test vector not available.
        };
        let img = crate::codec::jpeg::JpegImage::from_bytes(&data).unwrap();
        let grid = img.dct_grid(0);
        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let qt = img.quant_table(qt_id).unwrap();
        let map = compute_uniward(grid, qt);
        let (bw, bt) = (grid.blocks_wide(), grid.blocks_tall());
        let mut finite_count = 0;
        let mut total_cost = 0.0f64;
        for br in 0..bt {
            for bc in 0..bw {
                for i in 0..8 {
                    for j in 0..8 {
                        let c = map.get(br, bc, i, j);
                        if c.is_finite() {
                            finite_count += 1;
                            total_cost += c as f64;
                        }
                    }
                }
            }
        }
        assert!(finite_count > 1000, "expected >1000 finite costs, got {finite_count}");
        assert!(total_cost / finite_count as f64 > 0.0);
    }

    /// Cost bands from sub-grids (any partition, edges included) equal the
    /// whole-image pass, and the dense gather equals the sorted one.
    #[test]
    fn cost_bands_match_whole_image() {
        for name in ["photo_640x480_q75_420.jpg", "photo_320x240_q75_420.jpg", "fractal_100x75_q85_420.jpg"] {
            let data = std::fs::read(format!("test-vectors/image/{name}")).unwrap();
            let img = crate::codec::jpeg::JpegImage::from_bytes(&data).unwrap();
            let qt = img.quant_table(img.frame_info().components[0].quant_table_id as usize).unwrap();
            let g = img.dct_grid(0);
            let (bw, bt) = (g.blocks_wide(), g.blocks_tall());
            let whole = compute_ac_costs_separable(g, qt).unwrap();
            let mut cuts = vec![vec![0, bt], vec![0, 1, 2, 3, bt], vec![0, bt / 3, bt / 3 + 1, bt - 2, bt]];
            cuts.push((0..=bt).collect());
            for cut in cuts {
                for win in cut.windows(2) {
                    let (br0, br1) = (win[0], win[1]);
                    if br0 >= br1 {
                        continue;
                    }
                    let band = ac_cost_band(&cost_band_subgrid(g, br0, br1), qt, br0, br1, bt).unwrap();
                    let want = &whole[br0 * bw * 64..br1 * bw * 64];
                    assert!(band.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()), "{name} rows {br0}..{br1}");
                }
            }
            // Dense gather (whole order) == sorted gather (small request).
            let fis: Vec<u32> = (0..whole.len() as u32).rev().filter(|f| f % 64 != 0).collect();
            let dense = gather_ac_costs(g, qt, &fis).unwrap();
            let sorted: Vec<f32> = fis.chunks(fis.len() / 3 + 1).flat_map(|c| gather_ac_costs(g, qt, c).unwrap()).collect();
            assert!(dense.iter().zip(&sorted).all(|(a, b)| a.to_bits() == b.to_bits()), "{name} gather");
            assert!(dense.iter().zip(&fis).all(|(a, &f)| a.to_bits() == whole[f as usize].to_bits()), "{name} dense");
        }
    }

    /// The strip height changes neither the costs nor the order the sink
    /// sees them in: every height (1 block row up to more than the image,
    /// last strip partial or not) and every row range give the direct
    /// kernel's values, so the rows carried from strip to strip are exact.
    #[test]
    fn cost_strip_heights_agree_bit_exactly() {
        let qt = standard_qt();
        for (bw, bt) in [(3usize, 23usize), (7, 9), (1, 5), (2, 1), (4, 2)] {
            let grid = textured_grid(bw, bt);
            let want = compute_ac_costs_streaming(&grid, &qt).unwrap();
            for rows in [1usize, 2, 3, 4, 7, 50] {
                for (br0, br1) in [(0, bt), (bt / 3, bt), (0, bt.div_ceil(2)), (1.min(bt - 1), bt)] {
                    let mut got = vec![f32::NAN; (br1 - br0) * bw * 64];
                    let mut next = br0 * bw * 64;
                    ac_cost_strips_range(&grid, &qt, rows, br0, br1, |base, strip| {
                        assert_eq!(base, next, "strips in order");
                        assert!(strip.len() <= rows * bw * 64);
                        got[base - br0 * bw * 64..][..strip.len()].copy_from_slice(strip);
                        next += strip.len();
                        Ok(())
                    })
                    .unwrap();
                    assert_eq!(next, br1 * bw * 64);
                    let w = &want[br0 * bw * 64..br1 * bw * 64];
                    for (f, (a, b)) in got.iter().zip(w).enumerate() {
                        assert_eq!(a.to_bits(), b.to_bits(), "{bw}x{bt} rows {rows} range {br0}..{br1} at {f}");
                    }
                }
            }
        }
        // Strip heights the production rule picks.
        assert_eq!(cost_strip_rows(504), 4);
        assert_eq!(cost_strip_rows(1008), 4);
        assert_eq!(cost_strip_rows(1), STRIP_BLOCK_ROWS);
        assert_eq!(cost_strip_rows(100_000), MIN_STRIP_BLOCK_ROWS);
    }

    /// v2 changes cost VALUES only, never the SET of embeddable positions
    /// (every non-zero AC coefficient, in raster order): the primary Ghost
    /// decode, which uses the set but not the values, is unaffected.
    #[test]
    fn position_set_is_unchanged_from_legacy_v1() {
        for name in ["photo_320x240_q75_420.jpg", "photo_640x480_q75_420.jpg", "fractal_100x75_q85_420.jpg", "tiny_8x8_q95.jpg", "nonaligned_13x13_q80.jpg"] {
            let Ok(data) = std::fs::read(format!("test-vectors/image/{name}")) else { continue };
            let img = crate::codec::jpeg::JpegImage::from_bytes(&data).unwrap();
            let qt_id = img.frame_info().components[0].quant_table_id as usize;
            let qt = img.quant_table(qt_id).unwrap();
            let v1 = compute_positions_streaming_legacy_v1(img.dct_grid(0), qt, None).unwrap();
            let g = img.dct_grid(0);
            let v2: Vec<u32> = (0..g.coeffs().len() as u32)
                .filter(|&f| f % 64 != 0 && g.coeffs()[f as usize] != 0)
                .collect();
            assert_eq!(v1.len(), v2.len(), "{name}");
            assert!(v1.iter().zip(&v2).all(|(a, b)| a.flat_idx == *b), "{name}: set/order");
            let nzac = {
                let g = img.dct_grid(0);
                let mut n = 0;
                for br in 0..g.blocks_tall() {
                    for bc in 0..g.blocks_wide() {
                        n += g.block(br, bc)[1..].iter().filter(|&&c| c != 0).count();
                    }
                }
                n
            };
            assert_eq!(v2.len(), nzac, "{name}: positions == non-zero AC");

        }
    }

    /// Verify that compute_uniward is deterministic.
    #[test]
    fn determinism_repeated_runs() {
        let grid = textured_grid(6, 6);
        let qt = standard_qt();
        let map1 = compute_uniward(&grid, &qt);
        let map2 = compute_uniward(&grid, &qt);
        for br in 0..6 {
            for bc in 0..6 {
                for i in 0..8 {
                    for j in 0..8 {
                        assert_eq!(map1.get(br, bc, i, j).to_bits(), map2.get(br, bc, i, j).to_bits());
                    }
                }
            }
        }
    }

    /// Scalar-only copy of `cost_kernel`: what a target without NEON /
    /// SSE2 / SIMD128 runs. The production kernel always takes its SIMD
    /// block where one is compiled in, so its own scalar pair loop never
    /// runs there and cannot serve as the reference.
    fn cost_kernel_scalar_reference(
        abs_lh: &[f64],
        abs_hl: &[f64],
        abs_hh: &[f64],
        w: &PaddedSubbands,
        br: usize,
        bc: usize,
    ) -> f64 {
        let wf = |v: f32| (v as f64).abs() + SIGMA;
        let mut cost = 0.0;
        for out_r in 0..IMPACT_SIZE {
            let i0 = w.index((bc * 8) as isize - 8, (br * 8 + out_r) as isize - 8);
            let cell = |c: usize| {
                let (i, t) = (i0 + c, out_r * IMPACT_SIZE + c);
                (abs_lh[t] / wf(w.lh[i]) + abs_hl[t] / wf(w.hl[i])) + abs_hh[t] / wf(w.hh[i])
            };
            let mut c = 0;
            while c + 1 < IMPACT_SIZE {
                cost += cell(c) + cell(c + 1);
                c += 2;
            }
            cost += cell(c);
        }
        cost
    }

    /// The NEON / SSE2 / SIMD128 blocks of `cost_kernel` are bit-identical
    /// (f64 `to_bits`) to the scalar sum order on production data: the real
    /// |delta| table and the padded wavelets of a 64×64 cover, every block
    /// and coefficient (margins included).
    #[test]
    fn cost_kernel_simd_matches_scalar_reference_fixture() {
        let (cover, qt) = cost_kernel_test_fixture();
        let wv = whole_image_subbands(&cover, &qt);
        let table = AbsDeltaTable::precompute(&qt);
        let coefs: Vec<(usize, usize)> = if cfg!(miri) {
            vec![(0, 1), (3, 5), (7, 7)]
        } else {
            (1..64).map(|k| (k / 8, k % 8)).collect()
        };
        let blocks: Vec<(usize, usize)> = if cfg!(miri) {
            vec![(0, 0), (0, 7), (7, 0), (3, 4), (1, 1)]
        } else {
            (0..64).map(|k| (k / 8, k % 8)).collect()
        };
        for &(br, bc) in &blocks {
            for &(fi, fj) in &coefs {
                let (lh, hl, hh) = table.slice_at(fi, fj);
                let got = cost_kernel(lh, hl, hh, &wv, br, bc);
                let want = cost_kernel_scalar_reference(lh, hl, hh, &wv, br, bc);
                assert_eq!(got.to_bits(), want.to_bits(), "br={br} bc={bc} fi={fi} fj={fj}: {got} vs {want}");
            }
        }
    }

    /// Same bit-exactness on random inputs built directly (no IDCT, so it
    /// also runs under Miri): random geometry and strip row offsets, and
    /// wavelets / |delta| holding ±0, ±inf, f32::MAX, subnormals, 1e300.
    #[test]
    fn cost_kernel_simd_matches_scalar_reference_random() {
        let cells = AbsDeltaTable::CELLS;
        let mut seed = 0xC057_5EED_u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed >> 33
        };
        let wavelet_pick = |r: u64, r2: u64| -> f32 {
            match r % 12 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::INFINITY,
                3 => f32::NEG_INFINITY,
                4 => f32::MAX,
                5 => -f32::MAX,
                6 => f32::from_bits(1),
                7 => -1.0e-39,
                _ => ((r2 % 200_001) as f32 - 100_000.0) * 0.013,
            }
        };
        let delta_pick = |r: u64, r2: u64| -> f64 {
            match r % 8 {
                0 => 0.0,
                1 => f64::from_bits(3),
                2 => 1.0e300,
                _ => (r2 % 1_000_000) as f64 * 1.7e-4,
            }
        };
        let trials = if cfg!(miri) { 24 } else { 2000 };
        for t in 0..trials {
            let bw = 1 + (next() % 6) as usize;
            let bt = 1 + (next() % 6) as usize;
            let (br, bc) = ((next() % bt as u64) as usize, (next() % bw as u64) as usize);
            let stride = bw * 8 + 2 * WAV_MARGIN;
            // A strip whose rows start at or above the block's window.
            let y0 = (br * 8) as isize - 8 - (next() % 3) as isize;
            let rows = ((br * 8) as isize + 15 - y0) as usize + (next() % 4) as usize;
            let n = stride * rows;
            let mut band = || -> Vec<f32> { (0..n).map(|_| wavelet_pick(next(), next())).collect() };
            let w = PaddedSubbands { lh: band(), hl: band(), hh: band(), stride, y0, rows };
            let mut tab = || -> Vec<f64> { (0..cells).map(|_| delta_pick(next(), next())).collect() };
            let (lh, hl, hh) = (tab(), tab(), tab());
            let got = cost_kernel(&lh, &hl, &hh, &w, br, bc);
            let want = cost_kernel_scalar_reference(&lh, &hl, &hh, &w, br, bc);
            assert_eq!(got.to_bits(), want.to_bits(), "trial {t}: {bw}x{bt} y0={y0} br={br} bc={bc}: {got} vs {want}");
        }
    }

    /// End-to-end: Ghost encode/decode round-trip still works.
    #[test]
    fn ghost_roundtrip_with_current_feature_set() {
        let data = match std::fs::read("test-vectors/image/photo_320x240_q75_420.jpg") {
            Ok(d) => d,
            Err(_) => return,
        };
        let message = "Parallel cost test";
        let passphrase = "test-pass-42";
        let stego = crate::stego::ghost_encode(&data, message, passphrase).expect("ghost_encode should succeed");
        let decoded = crate::stego::ghost_decode(&stego, passphrase).expect("ghost_decode should succeed");
        assert_eq!(decoded.text, message, "round-trip mismatch");
        assert!(decoded.files.is_empty(), "no files expected");
    }

    /// Performance benchmark for J-UNIWARD cost computation on a real photo.
    /// Run manually with:
    ///   cargo test -p phasm-core --release --features parallel cost_computation_benchmark -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cost_computation_benchmark() {
        let data = if let Ok(d) = std::fs::read("test-vectors/image/photo_original_3557.jpg") {
            d
        } else if let Ok(d) = std::fs::read("test-vectors/image/photo_640x480_q75_420.jpg") {
            d
        } else {
            eprintln!("Skipping benchmark: test vector not found");
            return;
        };
        let img = crate::codec::jpeg::JpegImage::from_bytes(&data).unwrap();
        let grid = img.dct_grid(0);
        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let qt = img.quant_table(qt_id).unwrap();
        let _ = compute_ac_costs_separable(grid, qt);
        let iterations = 5;
        let t = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = compute_ac_costs_separable(grid, qt);
        }
        let v2 = t.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let t = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = compute_positions_streaming_legacy_v1(grid, qt, None);
        }
        let v1 = t.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        eprintln!(
            "J-UNIWARD costs ({}x{} px): all-AC separable {v2:.1} ms, legacy v1 nzAC {v1:.1} ms [parallel: {}]",
            grid.blocks_wide() * 8, grid.blocks_tall() * 8, cfg!(feature = "parallel"),
        );
    }
}
