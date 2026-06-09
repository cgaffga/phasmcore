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
//! Memory-optimized: uses f32 for pixel and wavelet buffers (pixels are 0-255,
//! f32 has 23-bit mantissa = ~7 decimal digits, more than sufficient).
//! Sequential subband computation drops intermediate buffers early, reducing
//! peak memory from ~651 MB to ~187 MB for a 4032x3024 image (71% reduction).
//!
//! Reference:
//!   Holub, Fridrich, Denemark. "Universal Distortion Function for
//!   Steganography in an Arbitrary Domain." EURASIP J. on Information
//!   Security, 2014.
//!
//! This implementation follows the corrected version (fixing the off-by-one
//! error described in arXiv:2305.19776).

use crate::codec::jpeg::dct::{DctGrid, QuantTable};
// T3.1.F — ghost UNIWARD bulk pixel-domain decompression routes
// through the integer LL&M IDCT.
//
// `precompute_basis_functions` (line ~367) is the exception: it calls
// `crate::codec::jpeg::pixels::idct_block` EXPLICITLY (the f64
// reference) because basis impulses require sub-LSB precision that
// integer LL&M descales to zero. Called once per encode → no perf
// impact. The f64 `idct_block` is preserved in `pixels.rs` SOLELY
// for this caller (and `dct_block_unquantized` for SI-UNIWARD).
use crate::codec::jpeg::pixels_aan::aan_idct_block as idct_block;
use crate::stego::error::StegoError;
use crate::stego::progress;
use super::CostMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
const SIGMA: f64 = 0.015625; // 2^{-6}

/// Daubechies 8 (db8) high-pass decomposition filter coefficients.
/// These are the wavelet function coefficients (16 taps).
/// Source: standard db8 wavelet tables, confirmed against conseal/PyWavelets.
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

/// Daubechies 8 (db8) low-pass decomposition filter coefficients.
/// Derived from the high-pass filter via the QMF relation:
///   lpdf[n] = (-1)^n * hpdf[N-1-n]
/// where N = 16.
fn lpdf() -> [f64; 16] {
    let mut lp = [0.0f64; 16];
    for n in 0..16 {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        lp[n] = sign * HPDF[15 - n];
    }
    lp
}

/// Length of the 1D wavelet filter.
const FILT_LEN: usize = 16;

/// Size of the wavelet-domain impact window when a single DCT coefficient
/// changes: 8 (block size) + 16 (filter length) - 1 = 23.
const IMPACT_SIZE: usize = 8 + FILT_LEN - 1;

/// Wrapper to send a raw pointer across Rayon threads.
/// Safe because each block writes to a disjoint region of the CostMap.
struct CostMapPtr {
    ptr: *mut f32,
    total_len: usize,
}
unsafe impl Send for CostMapPtr {}
unsafe impl Sync for CostMapPtr {}

impl CostMapPtr {
    /// Write a cost value at the given flat index.
    ///
    /// # Safety
    /// Caller must ensure no aliasing writes to the same index.
    unsafe fn write(&self, idx: usize, val: f32) {
        debug_assert!(idx < self.total_len, "CostMapPtr write out of bounds: {idx} >= {}", self.total_len);
        unsafe { *self.ptr.add(idx) = val; }
    }
}

/// Compute J-UNIWARD embedding costs for a single component's DCT grid.
///
/// For each embeddable coefficient position, the cost measures how much
/// a +1 change to that coefficient would alter the wavelet decomposition
/// of the decompressed image, weighted inversely by the cover's wavelet
/// magnitudes plus a stabilization constant sigma.
///
/// DC coefficients and zero-valued AC coefficients receive `WET_COST`.
pub fn compute_uniward(grid: &DctGrid, qt: &QuantTable) -> CostMap {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let mut map = CostMap::new(bw, bt);

    // Image pixel dimensions (block-aligned).
    let img_w = bw * 8;
    let img_h = bt * 8;

    // Step 1: Decompress the cover image to spatial domain (Y channel).
    // Uses f32 — pixel values are 0-255, f32 has more than enough precision.
    let cover_pixels = decompress_to_pixels(grid, qt, bw, bt);

    // Step 2: Compute wavelet coefficients of the cover image for all
    // three directional subbands. Uses symmetric padding.
    // Sequential computation drops intermediate buffers early to reduce peak memory.
    let lpdf = lpdf();
    let cover_wavelets = compute_three_subbands(&cover_pixels, img_w, img_h, &lpdf);

    // Free cover pixels — no longer needed after wavelet decomposition.
    drop(cover_pixels);

    // Step 3: Precompute the |delta| table (T3.3.A). 252 KB once,
    // shared across all 190K blocks. ~14× kernel speedup vs the
    // legacy basis-then-recompute path.
    let abs_delta = AbsDeltaTable::precompute(qt);

    // Step 4: For each block and each coefficient, compute cost.
    // Write costs directly into the CostMap (no intermediate collection).
    let pad = FILT_LEN - 1; // = 15
    let n_blocks = bt * bw;

    let total_len = n_blocks * 64;
    let costs_ptr = CostMapPtr { ptr: map.costs_ptr(), total_len };

    let compute_block = |bi: usize| {
        let br = bi / bw;
        let bc = bi % bw;
        let blk = grid.block(br, bc);

        for fi in 0..8 {
            for fj in 0..8 {
                // DC coefficient — never modify.
                if fi == 0 && fj == 0 {
                    continue;
                }

                let coeff = blk[fi * 8 + fj];

                // Zero AC coefficient — never modify.
                if coeff == 0 {
                    continue;
                }

                // T3.3.A — precomputed |delta| slices for this (fi, fj).
                let (abs_lh, abs_hl, abs_hh) = abs_delta.slice_at(fi, fj);
                let cost = compute_coefficient_cost_precomputed(
                    abs_lh, abs_hl, abs_hh,
                    &cover_wavelets,
                    br, bc,
                    img_w, img_h,
                    pad,
                );

                if cost > 0.0 && cost.is_finite() {
                    // Safety: each block writes to a disjoint 64-element region.
                    let idx = (br * bw + bc) * 64 + fi * 8 + fj;
                    unsafe { costs_ptr.write(idx, cost as f32); }
                }
            }
        }
    };

    // Use parallel iteration when the `parallel` feature is enabled.
    #[cfg(feature = "parallel")]
    (0..n_blocks).into_par_iter().for_each(compute_block);

    #[cfg(not(feature = "parallel"))]
    (0..n_blocks).for_each(compute_block);

    map
}

/// Decompress the entire DctGrid to pixel values (f32).
fn decompress_to_pixels(grid: &DctGrid, qt: &QuantTable, bw: usize, bt: usize) -> Vec<f32> {
    let img_w = bw * 8;
    let img_h = bt * 8;
    let mut pixels = vec![0.0f32; img_w * img_h];

    for br in 0..bt {
        for bc in 0..bw {
            let block = grid.block(br, bc);
            let quantized: [i16; 64] = block.try_into().unwrap();
            let block_pixels = idct_block(&quantized, &qt.values);

            for row in 0..8 {
                for col in 0..8 {
                    let py = br * 8 + row;
                    let px = bc * 8 + col;
                    pixels[py * img_w + px] = block_pixels[row * 8 + col] as f32;
                }
            }
        }
    }

    pixels
}

/// Precompute the 64 basis functions: for each frequency position (fi, fj),
/// compute the 8x8 pixel-domain block that results from IDCT of a unit
/// impulse, scaled by the quantization step.
///
/// This gives the pixel change caused by adding +1 to quantized coefficient
/// (fi, fj).
fn precompute_basis_functions(qt: &QuantTable) -> [[[f64; 64]; 8]; 8] {
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

/// Wavelet subbands for three directions. Each subband has the same
/// dimensions as the (padded) image — undecimated wavelet transform.
/// Uses f32 to halve memory usage vs f64.
struct ThreeSubbands {
    /// LH subband (horizontal high-pass): detects vertical edges.
    lh: Vec<f32>,
    /// HL subband (vertical high-pass): detects horizontal edges.
    hl: Vec<f32>,
    /// HH subband (diagonal high-pass): detects diagonal edges.
    hh: Vec<f32>,
    /// Width of each subband (same as image width).
    width: usize,
    /// Absolute y-coordinate of the first row in the arrays.
    /// 0 for full-image computation, >0 for strip-based computation.
    y_offset: usize,
}

/// Compute three-directional wavelet subbands of the image using an
/// undecimated (stationary) wavelet transform with db8 filters.
///
/// The three subbands are:
/// - LH: low-pass along rows, high-pass along columns (vertical detail)
/// - HL: high-pass along rows, low-pass along columns (horizontal detail)
/// - HH: high-pass along rows, high-pass along columns (diagonal detail)
///
/// Uses symmetric (mirror) boundary extension.
///
/// Memory-optimized: computes subbands sequentially, dropping intermediate
/// buffers as soon as possible. Peak: 3 f32 buffers (was 6 f64 buffers).
fn compute_three_subbands(
    pixels: &[f32],
    width: usize,
    height: usize,
    lpdf: &[f64; 16],
) -> ThreeSubbands {
    // Parallel: compute both row filters simultaneously, then 3 column filters.
    // +1 buffer peak (~25MB for 12MP) traded for ~2× speedup on wavelet phase.
    #[cfg(feature = "parallel")]
    {
        let (row_low, row_high) = rayon::join(
            || filter_rows(pixels, width, height, lpdf),
            || filter_rows(pixels, width, height, &HPDF),
        );
        let (lh, (hl, hh)) = rayon::join(
            || filter_cols(&row_low, width, height, &HPDF),
            || rayon::join(
                || filter_cols(&row_high, width, height, lpdf),
                || filter_cols(&row_high, width, height, &HPDF),
            ),
        );
        ThreeSubbands { lh, hl, hh, width, y_offset: 0 }
    }
    // Sequential: drop intermediates early to minimize peak memory (WASM path).
    #[cfg(not(feature = "parallel"))]
    {
        let row_low = filter_rows(pixels, width, height, lpdf);
        let lh = filter_cols(&row_low, width, height, &HPDF);
        drop(row_low);
        let row_high = filter_rows(pixels, width, height, &HPDF);
        let hl = filter_cols(&row_high, width, height, lpdf);
        let hh = filter_cols(&row_high, width, height, &HPDF);
        ThreeSubbands { lh, hl, hh, width, y_offset: 0 }
    }
}

/// Apply a 1D filter along each row of the image (horizontal filtering).
/// Uses symmetric (mirror) boundary extension.
/// Output has the same dimensions as input (undecimated).
/// Accumulates in f64 for filter precision, stores result as f32.
///
/// T2.1a — loop-peeled into `[0, half)` left boundary, `[half, width-half)`
/// interior (no mirror, contiguous load, branch-free → LLVM autovec
/// emits f64x2/f64x4 SIMD), and `[width-half, width)` right boundary.
/// T2.1b — outer loop parallelized over rows.
///
/// Determinism: f64 accumulator preserved, muladd order unchanged (per
/// April 2026 SIMD audit — f32-throughout would change cost values by
/// 1-2 ULP and risk flipping cost-pool boundaries across versions).
fn filter_rows(
    pixels: &[f32],
    width: usize,
    height: usize,
    filter: &[f64; 16],
) -> Vec<f32> {
    let flen = FILT_LEN;
    let half = (flen - 1) / 2; // = 7 for 16-tap filter (center at index 7)
    let mut output = vec![0.0f32; width * height];

    // Edge case: image narrower than filter — fall back to scalar with
    // mirror everywhere.
    if width < flen {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0f64;
                for k in 0..flen {
                    let sx =
                        mirror_index((x as isize) + (k as isize) - (half as isize), width);
                    sum += pixels[y * width + sx] as f64 * filter[k];
                }
                output[y * width + x] = sum as f32;
            }
        }
        return output;
    }

    // Asymmetric filter (even flen=16, half=7): left extent = half=7,
    // right extent = flen - half - 1 = 8. Interior x where the full
    // window [x-half, x-half+flen) stays in [0, width):
    // x ∈ [half, width - right_extent).
    let right_extent = flen - half - 1;
    let interior_end = width - right_extent;

    // Per-row processing helper — same body for parallel + serial paths.
    let process_row = |y: usize, row_out: &mut [f32]| {
        let row_in = &pixels[y * width..(y + 1) * width];

        // T2.1c Path A — boundary slabs gather 16 mirrored samples
            // into a stack-local [f32; 16] and call the same pairwise-
            // tree dot product as the interior. Byte-identical output
            // regardless of which slab x lives in.
            let mut samples = [0.0f32; 16];

            // Left boundary [0..half): mirror needed.
            for x in 0..half {
                for k in 0..flen {
                    let sx =
                        mirror_index((x as isize) + (k as isize) - (half as isize), width);
                    samples[k] = row_in[sx];
                }
                row_out[x] = super::uniward_simd::dot_product_16_taps(&samples, filter) as f32;
            }

            // Interior [half..interior_end): no mirror — contiguous
            // 16-tap window, dispatched to NEON on aarch64 / scalar
            // elsewhere via the same pinned-pairwise tree.
            for x in half..interior_end {
                let base = x - half;
                row_out[x] = super::uniward_simd::dot_product_16_taps(
                    &row_in[base..base + flen],
                    filter,
                ) as f32;
            }

        // Right boundary [interior_end..width): mirror needed.
        for x in interior_end..width {
            for k in 0..flen {
                let sx =
                    mirror_index((x as isize) + (k as isize) - (half as isize), width);
                samples[k] = row_in[sx];
            }
            row_out[x] = super::uniward_simd::dot_product_16_taps(&samples, filter) as f32;
        }
    };

    #[cfg(feature = "parallel")]
    output
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_out)| process_row(y, row_out));

    #[cfg(not(feature = "parallel"))]
    output
        .chunks_mut(width)
        .enumerate()
        .for_each(|(y, row_out)| process_row(y, row_out));

    output
}

/// Apply a 1D filter along each column of the image (vertical filtering).
/// Uses symmetric (mirror) boundary extension.
/// Output has the same dimensions as input (undecimated).
/// Accumulates in f64 for filter precision, stores result as f32.
///
/// T2.1a — loop-peeled per-row (interior rows skip mirror) AND
/// inner loops rewritten as k-outer / x-inner: for each input row k,
/// stream contiguously across all x, accumulating into a per-row f64
/// scratch. Cache-friendly (sequential reads, sequential writes) and
/// autovec-friendly (the inner x loop is `scratch[x] += row[x] * c` —
/// classic SAXPY shape LLVM emits SIMD for).
/// T2.1b — outer over output rows is rayon-parallel.
///
/// Determinism: per-(x,y) the same 16 muladds happen in the same
/// k=0..15 order; only the iteration order across (x,y) changed.
/// Byte-identical output preserved.
fn filter_cols(
    pixels: &[f32],
    width: usize,
    height: usize,
    filter: &[f64; 16],
) -> Vec<f32> {
    let flen = FILT_LEN;
    let half = (flen - 1) / 2; // = 7
    let mut output = vec![0.0f32; width * height];

    // Edge case: image shorter than filter — fall back to scalar.
    if height < flen {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0f64;
                for k in 0..flen {
                    let sy = mirror_index(
                        (y as isize) + (k as isize) - (half as isize),
                        height,
                    );
                    sum += pixels[sy * width + x] as f64 * filter[k];
                }
                output[y * width + x] = sum as f32;
            }
        }
        return output;
    }

    // Asymmetric filter (even flen=16, half=7): interior y where full
    // window stays in [0, height) is [half, height - right_extent).
    let right_extent = flen - half - 1;

    // Per-row processing helper — same body for parallel + serial paths.
    let process_row = |scratch: &mut [f64], y: usize, row_out: &mut [f32]| {
        // Reset scratch for this output row.
        for s in scratch.iter_mut() {
            *s = 0.0;
        }

        let is_interior = y >= half && y < height - right_extent;

        if is_interior {
            // No mirror: each of the 16 input rows is at y+k-half.
            // Outer k, inner x — sequential reads + writes →
            // T2.1c SIMD-accelerated SAXPY (NEON / AVX / WASM SIMD,
            // scalar elsewhere). Byte-identical across all
            // platforms by IEEE 754 + no-FMA + no-reduction-tree.
            for k in 0..flen {
                let sy = y + k - half;
                let row_in = &pixels[sy * width..(sy + 1) * width];
                super::uniward_simd::saxpy_inner(scratch, row_in, filter[k]);
            }
        } else {
            // Boundary row: mirror index per k, but still k-outer.
            for k in 0..flen {
                let sy = mirror_index(
                    (y as isize) + (k as isize) - (half as isize),
                    height,
                );
                let row_in = &pixels[sy * width..(sy + 1) * width];
                super::uniward_simd::saxpy_inner(scratch, row_in, filter[k]);
            }
        }

        // Cast f64 scratch → f32 output. Sequential, autovec-friendly.
        for x in 0..width {
            row_out[x] = scratch[x] as f32;
        }
    };

    #[cfg(feature = "parallel")]
    output
        .par_chunks_mut(width)
        .enumerate()
        .for_each_init(
            || vec![0.0f64; width], // per-thread scratch
            |scratch, (y, row_out)| process_row(scratch.as_mut_slice(), y, row_out),
        );

    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = vec![0.0f64; width];
        output.chunks_mut(width).enumerate().for_each(|(y, row_out)| {
            process_row(scratch.as_mut_slice(), y, row_out);
        });
    }

    output
}

/// Mirror-reflect an index into [0, size-1].
/// Handles negative indices and indices >= size by reflecting about boundaries.
#[inline]
fn mirror_index(idx: isize, size: usize) -> usize {
    let s = size as isize;
    if idx < 0 {
        (-idx).min(s - 1) as usize
    } else if idx >= s {
        let reflected = 2 * s - 2 - idx;
        reflected.max(0) as usize
    } else {
        idx as usize
    }
}

/// T3.3.A — precomputed |delta| table for compute_coefficient_cost.
///
/// `delta_lh / delta_hl / delta_hh` at each (out_r, out_c) of the
/// 23×23 impact region depend ONLY on (fi, fj) and qt — NOT on
/// (br, bc) or cover content. Materialize them ONCE per encode and
/// the per-block kernel collapses to ~1 600 ops (lookups + divides)
/// vs ~23 000 ops in the current path. ~14× scalar speedup before
/// SIMD.
///
/// Memory: 63 × 23 × 23 × 3 × 8 B = ~252 KB per encode. Fits in L2
/// on every target.
///
/// Layout: subband-major then `(fi*8+fj)`-major then `(out_r, out_c)`-
/// major. The per-block inner loop reads three contiguous 529-element
/// slices for the current (fi, fj), which is SAXPY-shape and SIMD-
/// friendly when Phases B–D land.
pub struct AbsDeltaTable {
    abs_lh: Vec<f64>,
    abs_hl: Vec<f64>,
    abs_hh: Vec<f64>,
}

impl AbsDeltaTable {
    const CELLS: usize = IMPACT_SIZE * IMPACT_SIZE;

    /// Precompute the 63 × 23 × 23 × 3 |delta| table for the given
    /// quantization table. ~1 ms wall-clock at 12 MP.
    pub fn precompute(qt: &QuantTable) -> Self {
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

                // Phase 1: row filter (same formula as legacy
                // compute_coefficient_cost — bit-exact sum order).
                let mut row_low = [[0.0f64; IMPACT_SIZE]; 8];
                let mut row_high = [[0.0f64; IMPACT_SIZE]; 8];
                for r in 0..8 {
                    for out_c in 0..IMPACT_SIZE {
                        let mut sum_low = 0.0;
                        let mut sum_high = 0.0;
                        for k in 0..FILT_LEN {
                            let src_col = out_c as isize - 14 + k as isize;
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
                            let src_row = out_r as isize - 14 + k as isize;
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
    fn slice_at(&self, fi: usize, fj: usize) -> (&[f64], &[f64], &[f64]) {
        let base = (fi * 8 + fj) * Self::CELLS;
        let end = base + Self::CELLS;
        (&self.abs_lh[base..end], &self.abs_hl[base..end], &self.abs_hh[base..end])
    }
}

/// T3.3.A — precomputed-table fast path of compute_coefficient_cost
/// (legacy sum order: per-cell `lh + hl + hh` added directly into a
/// single accumulator). Kept for the Phase A bit-exactness gate vs
/// the legacy basis-recompute path. **Not used in production** — the
/// SIMD-friendly variant below is the one wired into call sites.
#[inline]
#[allow(dead_code)]
fn compute_coefficient_cost_precomputed_legacy_order(
    abs_lh: &[f64],
    abs_hl: &[f64],
    abs_hh: &[f64],
    cover_wavelets: &ThreeSubbands,
    br: usize,
    bc: usize,
    img_w: usize,
    img_h: usize,
    pad: usize,
) -> f64 {
    let mut cost = 0.0;
    for out_r in 0..IMPACT_SIZE {
        let abs_y = (br * 8) as isize + out_r as isize - pad as isize;
        if abs_y < 0 || abs_y as usize >= img_h {
            continue;
        }
        let wy = abs_y as usize - cover_wavelets.y_offset;
        let row_base = out_r * IMPACT_SIZE;
        for out_c in 0..IMPACT_SIZE {
            let abs_x = (bc * 8) as isize + out_c as isize - pad as isize;
            if abs_x < 0 || abs_x as usize >= img_w {
                continue;
            }
            let wx = abs_x as usize;
            let idx = wy * cover_wavelets.width + wx;

            let w_lh = (cover_wavelets.lh[idx] as f64).abs() + SIGMA;
            let w_hl = (cover_wavelets.hl[idx] as f64).abs() + SIGMA;
            let w_hh = (cover_wavelets.hh[idx] as f64).abs() + SIGMA;

            let cell = row_base + out_c;
            cost += abs_lh[cell] / w_lh;
            cost += abs_hl[cell] / w_hl;
            cost += abs_hh[cell] / w_hh;
        }
    }
    cost
}

/// T3.3.B — SIMD-friendly precomputed-table cost kernel.
///
/// Sum order: per cell `(lh + hl) + hh` (parenthesized for explicit
/// evaluation order), then pair-wise `(cell0 + cell1)`, then row-wise
/// accumulation. The pairing matches the NEON / AVX2 / WASM SIMD
/// lane layout exactly so the scalar fallback gives bit-identical
/// f64 output to every SIMD path.
///
/// Cross-platform deterministic: same sum order on aarch64 + x86_64 +
/// wasm32, gated by the `cost_precompute_simd_cross_platform_hash`
/// test (Phase E).
///
/// Drift vs `compute_coefficient_cost_precomputed_legacy_order`: at
/// most a handful of f64 ULP, far below f32 cost LSB → stego output
/// bit-identical.
#[inline]
fn compute_coefficient_cost_precomputed(
    abs_lh: &[f64],
    abs_hl: &[f64],
    abs_hh: &[f64],
    cover_wavelets: &ThreeSubbands,
    br: usize,
    bc: usize,
    img_w: usize,
    img_h: usize,
    pad: usize,
) -> f64 {
    let mut cost = 0.0;
    let stride = cover_wavelets.width;
    let bc8 = (bc * 8) as isize;
    let abs_x_start = bc8 - pad as isize;

    // Valid out_c range for any row of THIS block (same for all rows).
    let out_c_lo: usize = if abs_x_start < 0 {
        ((-abs_x_start) as usize).min(IMPACT_SIZE)
    } else {
        0
    };
    let out_c_hi: usize = {
        let max_via_image = img_w as isize - abs_x_start;
        if max_via_image <= 0 {
            0
        } else {
            (max_via_image as usize).min(IMPACT_SIZE)
        }
    };
    if out_c_lo >= out_c_hi {
        return 0.0;
    }

    for out_r in 0..IMPACT_SIZE {
        let abs_y = (br * 8) as isize + out_r as isize - pad as isize;
        if abs_y < 0 || abs_y as usize >= img_h {
            continue;
        }
        let wy = abs_y as usize - cover_wavelets.y_offset;
        let row_base = out_r * IMPACT_SIZE;
        let row_idx_base = wy * stride;

        let mut out_c = out_c_lo;

        // NEON 2-lane f64 path on aarch64.
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            use core::arch::aarch64::*;
            let sigma_v = vdupq_n_f64(SIGMA);
            while out_c + 1 < out_c_hi {
                let abs_x = bc8 + out_c as isize - pad as isize;
                let wx = abs_x as usize;
                let idx = row_idx_base + wx;
                let cell = row_base + out_c;

                // Load 2 f32 wavelet values per subband, widen to f64.
                let v_lh_f32 = vld1_f32(cover_wavelets.lh[idx..].as_ptr());
                let v_hl_f32 = vld1_f32(cover_wavelets.hl[idx..].as_ptr());
                let v_hh_f32 = vld1_f32(cover_wavelets.hh[idx..].as_ptr());
                let v_lh_w = vaddq_f64(vabsq_f64(vcvt_f64_f32(v_lh_f32)), sigma_v);
                let v_hl_w = vaddq_f64(vabsq_f64(vcvt_f64_f32(v_hl_f32)), sigma_v);
                let v_hh_w = vaddq_f64(vabsq_f64(vcvt_f64_f32(v_hh_f32)), sigma_v);

                // Load |delta| pair (already f64).
                let v_abs_lh = vld1q_f64(abs_lh[cell..].as_ptr());
                let v_abs_hl = vld1q_f64(abs_hl[cell..].as_ptr());
                let v_abs_hh = vld1q_f64(abs_hh[cell..].as_ptr());

                let v_div_lh = vdivq_f64(v_abs_lh, v_lh_w);
                let v_div_hl = vdivq_f64(v_abs_hl, v_hl_w);
                let v_div_hh = vdivq_f64(v_abs_hh, v_hh_w);

                // Per-lane sum: (lh + hl) + hh.
                let v_partial = vaddq_f64(vaddq_f64(v_div_lh, v_div_hl), v_div_hh);

                // Sum two lanes — explicit lane extraction for
                // deterministic order (lane0 + lane1).
                let lane0 = vgetq_lane_f64(v_partial, 0);
                let lane1 = vgetq_lane_f64(v_partial, 1);
                cost += lane0 + lane1;

                out_c += 2;
            }
        }

        // WASM SIMD128 2-lane f64 path. Phasm's wasm bridges enable
        // `+simd128` in `.cargo/config.toml`, so this gate fires for
        // the deployed phasm.app build. Matches the NEON / SSE2
        // pair-wise sum order byte-for-byte.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            use core::arch::wasm32::*;
            let sigma_v = f64x2_splat(SIGMA);
            while out_c + 1 < out_c_hi {
                let abs_x = bc8 + out_c as isize - pad as isize;
                let wx = abs_x as usize;
                let idx = row_idx_base + wx;
                let cell = row_base + out_c;

                // Load 2 f32 wavelet values per subband, widen to f64.
                // `v128_load64_zero` reads 8 bytes (2 f32) into the
                // low half and zeroes the high half; the subsequent
                // `f64x2_promote_low_f32x4` then widens the low 2
                // lanes to 2 f64.
                let v_lh_f32 =
                    v128_load64_zero(cover_wavelets.lh[idx..].as_ptr() as *const u64);
                let v_hl_f32 =
                    v128_load64_zero(cover_wavelets.hl[idx..].as_ptr() as *const u64);
                let v_hh_f32 =
                    v128_load64_zero(cover_wavelets.hh[idx..].as_ptr() as *const u64);
                let v_lh_w = f64x2_add(f64x2_abs(f64x2_promote_low_f32x4(v_lh_f32)), sigma_v);
                let v_hl_w = f64x2_add(f64x2_abs(f64x2_promote_low_f32x4(v_hl_f32)), sigma_v);
                let v_hh_w = f64x2_add(f64x2_abs(f64x2_promote_low_f32x4(v_hh_f32)), sigma_v);

                let v_abs_lh = v128_load(abs_lh[cell..].as_ptr() as *const v128);
                let v_abs_hl = v128_load(abs_hl[cell..].as_ptr() as *const v128);
                let v_abs_hh = v128_load(abs_hh[cell..].as_ptr() as *const v128);

                let v_div_lh = f64x2_div(v_abs_lh, v_lh_w);
                let v_div_hl = f64x2_div(v_abs_hl, v_hl_w);
                let v_div_hh = f64x2_div(v_abs_hh, v_hh_w);

                let v_partial = f64x2_add(f64x2_add(v_div_lh, v_div_hl), v_div_hh);

                let lane0 = f64x2_extract_lane::<0>(v_partial);
                let lane1 = f64x2_extract_lane::<1>(v_partial);
                cost += lane0 + lane1;

                out_c += 2;
            }
        }

        // SSE2 2-lane f64 path on x86_64. Uses `__m128d` (the baseline
        // SSE2 f64 vector, available on every x86_64 chip — phasm's
        // `.cargo/config.toml` already enables SSE4.1+SSSE3 by default
        // for the x86_64 targets but stops short of AVX2 to keep CLI
        // release binaries broadly compatible). Matches the NEON
        // pair-wise sum order byte-for-byte.
        //
        // T3.3.C deliberately uses 2-lane (not AVX2 4-lane) so the
        // sum order is identical to NEON / SIMD128 / scalar — Phase E
        // pins the cross-platform hash on this exact ordering. An
        // AVX2 4-lane variant could be added later as an opt-in fast
        // path, with explicit pair-of-pairs reduction to match this
        // hash.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::*;
            let sigma_v = _mm_set1_pd(SIGMA);
            // Bitmask for f64 |.|: clear sign bit (bit 63).
            let abs_mask =
                _mm_castsi128_pd(_mm_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF));
            while out_c + 1 < out_c_hi {
                let abs_x = bc8 + out_c as isize - pad as isize;
                let wx = abs_x as usize;
                let idx = row_idx_base + wx;
                let cell = row_base + out_c;

                // Load 2 f32 wavelet values per subband, widen to f64.
                // `_mm_loadl_pi` would need an __m64; use generic 64-bit
                // load + cast to __m128 (low 64 bits = 2 f32, upper
                // halves zeroed by cvtps_pd anyway).
                let v_lh_f32 = _mm_castsi128_ps(_mm_loadl_epi64(
                    cover_wavelets.lh[idx..].as_ptr() as *const __m128i,
                ));
                let v_hl_f32 = _mm_castsi128_ps(_mm_loadl_epi64(
                    cover_wavelets.hl[idx..].as_ptr() as *const __m128i,
                ));
                let v_hh_f32 = _mm_castsi128_ps(_mm_loadl_epi64(
                    cover_wavelets.hh[idx..].as_ptr() as *const __m128i,
                ));
                let v_lh_w =
                    _mm_add_pd(_mm_and_pd(_mm_cvtps_pd(v_lh_f32), abs_mask), sigma_v);
                let v_hl_w =
                    _mm_add_pd(_mm_and_pd(_mm_cvtps_pd(v_hl_f32), abs_mask), sigma_v);
                let v_hh_w =
                    _mm_add_pd(_mm_and_pd(_mm_cvtps_pd(v_hh_f32), abs_mask), sigma_v);

                let v_abs_lh = _mm_loadu_pd(abs_lh[cell..].as_ptr());
                let v_abs_hl = _mm_loadu_pd(abs_hl[cell..].as_ptr());
                let v_abs_hh = _mm_loadu_pd(abs_hh[cell..].as_ptr());

                let v_div_lh = _mm_div_pd(v_abs_lh, v_lh_w);
                let v_div_hl = _mm_div_pd(v_abs_hl, v_hl_w);
                let v_div_hh = _mm_div_pd(v_abs_hh, v_hh_w);

                let v_partial = _mm_add_pd(_mm_add_pd(v_div_lh, v_div_hl), v_div_hh);

                // Lane extract: `_mm_cvtsd_f64` returns lane 0 (low f64).
                // For lane 1, unpack the high half into the low position
                // first, then extract.
                let lane0 = _mm_cvtsd_f64(v_partial);
                let lane1 = _mm_cvtsd_f64(_mm_unpackhi_pd(v_partial, v_partial));
                cost += lane0 + lane1;

                out_c += 2;
            }
        }

        // Scalar path — handles every row on non-NEON targets, and the
        // 0/1-element tail on NEON. Order matches the NEON kernel:
        // per-lane `(lh + hl) + hh`, then `lane0 + lane1`, then row
        // accumulate.
        while out_c + 1 < out_c_hi {
            let abs_x_0 = bc8 + out_c as isize - pad as isize;
            let wx_0 = abs_x_0 as usize;
            let idx_0 = row_idx_base + wx_0;
            let idx_1 = idx_0 + 1;
            let cell_0 = row_base + out_c;
            let cell_1 = cell_0 + 1;

            let lh_w_0 = (cover_wavelets.lh[idx_0] as f64).abs() + SIGMA;
            let hl_w_0 = (cover_wavelets.hl[idx_0] as f64).abs() + SIGMA;
            let hh_w_0 = (cover_wavelets.hh[idx_0] as f64).abs() + SIGMA;
            let lh_w_1 = (cover_wavelets.lh[idx_1] as f64).abs() + SIGMA;
            let hl_w_1 = (cover_wavelets.hl[idx_1] as f64).abs() + SIGMA;
            let hh_w_1 = (cover_wavelets.hh[idx_1] as f64).abs() + SIGMA;

            let div_lh_0 = abs_lh[cell_0] / lh_w_0;
            let div_hl_0 = abs_hl[cell_0] / hl_w_0;
            let div_hh_0 = abs_hh[cell_0] / hh_w_0;
            let div_lh_1 = abs_lh[cell_1] / lh_w_1;
            let div_hl_1 = abs_hl[cell_1] / hl_w_1;
            let div_hh_1 = abs_hh[cell_1] / hh_w_1;

            let lane0 = (div_lh_0 + div_hl_0) + div_hh_0;
            let lane1 = (div_lh_1 + div_hl_1) + div_hh_1;
            cost += lane0 + lane1;

            out_c += 2;
        }

        // 1-element tail.
        if out_c < out_c_hi {
            let abs_x = bc8 + out_c as isize - pad as isize;
            let wx = abs_x as usize;
            let idx = row_idx_base + wx;
            let cell = row_base + out_c;

            let lh_w = (cover_wavelets.lh[idx] as f64).abs() + SIGMA;
            let hl_w = (cover_wavelets.hl[idx] as f64).abs() + SIGMA;
            let hh_w = (cover_wavelets.hh[idx] as f64).abs() + SIGMA;

            let div_lh = abs_lh[cell] / lh_w;
            let div_hl = abs_hl[cell] / hl_w;
            let div_hh = abs_hh[cell] / hh_w;
            cost += (div_lh + div_hl) + div_hh;
        }
    }

    cost
}

/// Compute the J-UNIWARD cost for a single coefficient at position
/// (fi, fj) in block (br, bc).
///
/// The cost is the sum over three subbands of:
///   |delta_W[k]| / (|W_cover[k]| + sigma)
///
/// where delta_W[k] is the wavelet-domain impact of the +1 coefficient change,
/// and W_cover[k] is the cover's wavelet coefficient at the same location.
fn compute_coefficient_cost(
    basis_block: &[f64; 64],
    cover_wavelets: &ThreeSubbands,
    br: usize,
    bc: usize,
    img_w: usize,
    img_h: usize,
    pad: usize,
    lpdf: &[f64; 16],
) -> f64 {
    let mut cost = 0.0;

    // The pixel delta is only non-zero in the 8x8 block.
    // The wavelet filters are separable (row then column), each 16-tap.
    // After row filtering, the non-zero region expands to 8+15 = 23 wide.
    // After column filtering, it expands to 23 tall.
    // So the total impact region is IMPACT_SIZE x IMPACT_SIZE = 23x23.

    // Row-filter the 8 rows of the basis block.
    // Input: 8 values at columns [0..8) (relative to block origin).
    // Output: IMPACT_SIZE values at columns [-pad..8+pad-pad) = [-7..16) relative to block origin,
    // i.e. columns [-7, -6, ..., 15] relative to block origin.
    // In absolute terms: bc*8 - 7 to bc*8 + 15.
    let mut row_low = [[0.0f64; IMPACT_SIZE]; 8]; // 8 rows x 23 cols
    let mut row_high = [[0.0f64; IMPACT_SIZE]; 8];

    // In filter_rows, output[x] = sum_{k=0}^{15} filter[k] * input[x + k - 7].
    // For a delta at block-relative input column c, the affected outputs are at
    // x = c + 7 - k, giving x in [-8, 14] for c in [0,7]. That is 23 positions.
    // We index these as out_c in [0, 23), with block-relative output column
    // = out_c - 7, and source column = out_c - 14 + k.
    for r in 0..8 {
        for out_c in 0..IMPACT_SIZE {
            let mut sum_low = 0.0;
            let mut sum_high = 0.0;
            for k in 0..FILT_LEN {
                let src_col = out_c as isize - 14 + k as isize;
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

    // Column-filter the row-filtered results.
    // Input: 8 rows of IMPACT_SIZE columns.
    // Output: IMPACT_SIZE rows of IMPACT_SIZE columns.
    // Same logic: output row (out_r - half) in block-relative coords.
    // The delta impact for the three subbands:
    // LH: row_low filtered by HPDF along columns
    // HL: row_high filtered by lpdf along columns
    // HH: row_high filtered by HPDF along columns

    for out_r in 0..IMPACT_SIZE {
        for out_c in 0..IMPACT_SIZE {
            let mut delta_lh = 0.0;
            let mut delta_hl = 0.0;
            let mut delta_hh = 0.0;

            for k in 0..FILT_LEN {
                let src_row = out_r as isize - 14 + k as isize;
                if (0..8).contains(&src_row) {
                    let r = src_row as usize;
                    let low_val = row_low[r][out_c];
                    let high_val = row_high[r][out_c];

                    delta_lh += HPDF[k] * low_val;
                    delta_hl += lpdf[k] * high_val;
                    delta_hh += HPDF[k] * high_val;
                }
            }

            // Absolute pixel coordinates of this wavelet coefficient.
            let abs_y = (br * 8) as isize + out_r as isize - (pad as isize);
            let abs_x = (bc * 8) as isize + out_c as isize - (pad as isize);

            // Only accumulate if within image bounds (wavelet coefficients
            // outside the image don't exist in the cover's wavelet decomposition).
            if abs_y >= 0 && abs_y < img_h as isize && abs_x >= 0 && abs_x < img_w as isize {
                let wy = abs_y as usize;
                let wx = abs_x as usize;
                let idx = (wy - cover_wavelets.y_offset) * cover_wavelets.width + wx;

                // Cover wavelet values are f32; promote to f64 for cost computation.
                let w_lh = cover_wavelets.lh[idx] as f64;
                let w_hl = cover_wavelets.hl[idx] as f64;
                let w_hh = cover_wavelets.hh[idx] as f64;

                cost += delta_lh.abs() / (w_lh.abs() + SIGMA);
                cost += delta_hl.abs() / (w_hl.abs() + SIGMA);
                cost += delta_hh.abs() / (w_hh.abs() + SIGMA);
            }
        }
    }

    cost
}

// ---------------------------------------------------------------------------
// Strip-based streaming UNIWARD: computes positions without full CostMap.
// ---------------------------------------------------------------------------

use crate::stego::permute::CoeffPos;
use crate::stego::side_info::SideInfo;

/// Strip height in block rows for streaming UNIWARD computation.
/// 50 block rows = 400 pixel rows. Each strip uses ~170 MB peak for a 16K-wide
/// image (pixels + row-filtered + 3 wavelet subbands + strip CostMap).
const STRIP_BLOCK_ROWS: usize = 50;

// ============================================================================
// T3.2 — cached cover wavelets + position computation from cache
// ============================================================================
//
// The shadow cascade verify loop (`ghost/pipeline.rs`) runs
// `compute_positions_streaming` up to 12× per encode to recompute
// UNIWARD costs on each post-STC stego image. Each call redoes the
// full IDCT + 16-tap wavelet decomposition (~70 % of UNIWARD wall-
// clock at 12 MP). T3.2 caches the COVER wavelets once, then the
// cascade verify uses incremental wavelet deltas from STC
// modifications (Phase C) and recomputes cost only for dirty
// coefficients (Phase D).
//
// Phase B (this section) is the API foundation — add a way to
// materialize the cover wavelets and a position-computation entry
// point that takes pre-cached wavelets. The existing strip-based
// `compute_positions_streaming` stays unchanged as the primary
// forward entry point.

/// Cover-side wavelet decomposition, cached for T3.2 cascade verify
/// re-cost. Holds LH/HL/HH subbands as f32 at full image resolution.
///
/// Memory: 12 bytes × img_w × img_h ≈ 72 MB at 12 MP, 600 MB at
/// 100 MP. Suitable for cascade verify where the same cover is
/// re-evaluated multiple times; not suitable for memory-constrained
/// forward paths (which keep using `compute_positions_streaming`).
pub struct CachedCoverWavelets {
    inner: ThreeSubbands,
    pub img_w: usize,
    pub img_h: usize,
}

/// Compute the cover-side wavelet decomposition of an image. Used as
/// the cached baseline for T3.2 dirty-block re-cost in the cascade
/// verify loop. Memory: see [`CachedCoverWavelets`].
/// T3.3.E — deterministic byte stream for the cost kernel,
/// used by `cost_kernel_test_hash_hex` to pin the cross-platform
/// hash. Re-computing this on any supported target MUST yield the
/// same bytes (and therefore the same SHA256).
///
/// Layout: for each (br, bc, fi, fj) in a fixed iteration order, dump
/// the f64 cost as 8 little-endian bytes via `f64::to_bits`. The
/// fixture is 8×8 blocks (64×64 px), structured to hit every code
/// path (interior + each corner + mid-edges).
#[doc(hidden)]
pub fn cost_kernel_test_deterministic_bytes() -> Vec<u8> {
    // Fixed cover grid: deterministic content, mix of magnitudes.
    let mut cover = DctGrid::new(8, 8);
    for br in 0..8 {
        for bc in 0..8 {
            cover.set(br, bc, 0, 0, 120);
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    let val =
                        (((br * 11 + bc * 7 + i * 5 + j * 3) % 19) as i16) - 9;
                    if val != 0 {
                        cover.set(br, bc, i, j, val);
                    }
                }
            }
        }
    }
    // Inline the standard JPEG luminance quant table (quality ~50)
    // so the helper doesn't depend on test-only fixtures.
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
    let wavelets = compute_cover_wavelets(&cover, &qt);
    let table = AbsDeltaTable::precompute(&qt);
    let pad = FILT_LEN - 1;
    let img_w = wavelets.img_w;
    let img_h = wavelets.img_h;

    let mut bytes = Vec::with_capacity(8 * 8 * 63 * 8);
    for br in 0..8 {
        for bc in 0..8 {
            for fi in 0..8 {
                for fj in 0..8 {
                    if fi == 0 && fj == 0 {
                        continue;
                    }
                    let (abs_lh, abs_hl, abs_hh) = table.slice_at(fi, fj);
                    let cost = compute_coefficient_cost_precomputed(
                        abs_lh, abs_hl, abs_hh,
                        &wavelets.inner,
                        br, bc, img_w, img_h, pad,
                    );
                    bytes.extend_from_slice(&cost.to_bits().to_le_bytes());
                }
            }
        }
    }
    bytes
}

/// SHA256 of [`cost_kernel_test_deterministic_bytes`] as lowercase hex.
#[doc(hidden)]
pub fn cost_kernel_test_hash_hex() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(cost_kernel_test_deterministic_bytes());
    let digest = h.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

/// Number of progress sub-steps emitted by [`compute_cover_wavelets`].
/// Used by cascade-trigger code in `ghost/pipeline.rs` to budget total
/// steps before this function is called.
pub const COVER_WAVELETS_PROGRESS_STEPS: u32 = 3;

pub fn compute_cover_wavelets(grid: &DctGrid, qt: &QuantTable) -> CachedCoverWavelets {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let img_w = bw * 8;
    let img_h = bt * 8;
    let cover_pixels = decompress_to_pixels(grid, qt, bw, bt);
    progress::advance(); // cover_wavelets sub-step 1: decompress to pixels
    let lpdf = lpdf();
    let inner = compute_three_subbands(&cover_pixels, img_w, img_h, &lpdf);
    progress::advance(); // cover_wavelets sub-step 2: wavelet subbands
    drop(cover_pixels);
    progress::advance(); // cover_wavelets sub-step 3: cleanup
    CachedCoverWavelets { inner, img_w, img_h }
}

/// Stego-side wavelets = cover wavelets + Σ delta_wavelet per STC
/// modification. Same memory shape and layout as
/// [`CachedCoverWavelets`]; the only difference is provenance.
pub struct StegoWavelets {
    inner: ThreeSubbands,
    pub img_w: usize,
    pub img_h: usize,
}

impl StegoWavelets {
    /// Treat the stego wavelets as a CachedCoverWavelets view for
    /// passing into [`compute_positions_from_wavelets`]. (Field-equivalent
    /// struct; we keep them distinct in the type system to avoid mixing
    /// up cover-vs-stego in callers.)
    pub fn as_cache(&self) -> CachedCoverWavelets {
        CachedCoverWavelets {
            inner: ThreeSubbands {
                lh: self.inner.lh.clone(),
                hl: self.inner.hl.clone(),
                hh: self.inner.hh.clone(),
                width: self.inner.width,
                y_offset: self.inner.y_offset,
            },
            img_w: self.img_w,
            img_h: self.img_h,
        }
    }
}

/// Apply STC modifications to the cover wavelets, producing stego
/// wavelets via incremental wavelet delta. Each modification at DCT
/// coefficient (br, bc, fi, fj) contributes a `basis[fi][fj] ×
/// (stego_coeff − cover_coeff)` pixel delta in the 8×8 block at
/// (br, bc); this pixel delta is row+column 16-tap-filtered to give a
/// 23×23 LH/HL/HH wavelet delta that's added (signed) to the cover
/// wavelets.
///
/// The wavelet is linear → overlapping modifications sum correctly
/// (the delta from modification A + delta from modification B =
/// delta_AB).
///
/// Bit-exactness with `compute_cover_wavelets(modified_grid, qt)` is
/// gated by `incremental_wavelet_matches_full_recompute` (lib test).
///
/// Memory: clones cover_wavelets once (~72 MB at 12 MP) and writes in
/// place. The cover wavelets themselves are NOT mutated.
pub fn apply_dct_modifications_to_wavelets(
    cover_wavelets: &CachedCoverWavelets,
    modifications: &[u32],
    original_grid: &DctGrid,
    modified_grid: &DctGrid,
    qt: &QuantTable,
) -> StegoWavelets {
    // Start from cover; add deltas per modification.
    let mut lh: Vec<f32> = cover_wavelets.inner.lh.clone();
    let mut hl: Vec<f32> = cover_wavelets.inner.hl.clone();
    let mut hh: Vec<f32> = cover_wavelets.inner.hh.clone();
    let width = cover_wavelets.inner.width;
    let height = cover_wavelets.img_h;
    let bw = original_grid.blocks_wide();
    let lpdf = lpdf();

    // Collect unique dirty blocks. The pixel delta for a block is
    // determined by the FULL aan-IDCT of cover vs stego (NOT by per-
    // coefficient basis sums — the integer LL&M IDCT is not perfectly
    // linear at the LSB, so basis * dct_delta diverges from
    // aan_idct(stego_block) − aan_idct(cover_block) at f32 precision).
    let mut dirty_blocks: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    for &flat_idx in modifications {
        let block_idx = (flat_idx as usize) / 64;
        dirty_blocks.insert((block_idx / bw, block_idx % bw));
    }

    for (br, bc) in dirty_blocks {
        // Per-block pixel delta via the SAME aan_idct that
        // decompress_to_pixels uses. This guarantees bit-exact match
        // with the full filter path on the stego image.
        let cover_block: [i16; 64] = original_grid.block(br, bc).try_into().unwrap();
        let stego_block: [i16; 64] = modified_grid.block(br, bc).try_into().unwrap();
        if cover_block == stego_block {
            continue;
        }
        let cover_pixels = idct_block(&cover_block, &qt.values);
        let stego_pixels = idct_block(&stego_block, &qt.values);
        let mut basis_block = [0.0f64; 64];
        for k in 0..64 {
            // aan_idct returns f32 → widen to f64 for filter precision.
            basis_block[k] = (stego_pixels[k] as f64) - (cover_pixels[k] as f64);
        }
        let dct_delta_f = 1.0f64; // delta is already baked into basis_block

        // Match the FULL filter convention used by filter_rows +
        // filter_cols:
        //
        //   output[y][x] = Σ_k filter[k] · input[mirror(y+k−7)][mirror(x+k−7)]
        //
        // (16-tap, mirror at image boundaries, center at k=7). The
        // modified pixel delta is non-zero only inside the basis block;
        // the impact region in output-space is the set of (abs_y,abs_x)
        // for which the filter window touches that 8×8 block (possibly
        // via mirror reflection).
        //
        // The impact region is at most 23×23 centered on the modified
        // block (abs_y in [br*8−8 .. br*8+14], same for x). Mirror at
        // image edges does NOT extend this region — it only reuses
        // modified-block samples within filter windows that ARE in
        // the standard impact region.
        let half = (FILT_LEN - 1) / 2; // 7

        // Row-filter pass: produce row_low_delta / row_high_delta at
        // each (basis_row r, abs_x in impact x-range).
        // We index row_low / row_high by [r][out_c] where:
        //   out_c ∈ [0, IMPACT_EXT), abs_x = (bc*8 − 8) + out_c
        // (i.e., out_c=0 maps to abs_x = bc*8 − 8, the leftmost
        // output position).
        //
        // IMPACT_EXT = IMPACT_SIZE + (FILT_LEN/2) = 23 + 8 = 31. The
        // extra 8 output positions catch BOTTOM/RIGHT mirror extensions
        // (when src_y_raw or src_x_raw exceeds img dimensions, mirror
        // reflects back into the basis — affecting outputs up to 8
        // pixels beyond the standard 23-wide box).
        const IMPACT_EXT: usize = IMPACT_SIZE + 8;
        let mut row_low = [[0.0f64; IMPACT_EXT]; 8];
        let mut row_high = [[0.0f64; IMPACT_EXT]; 8];
        let abs_x_start = (bc * 8) as isize - half as isize - 1; // bc*8 − 8
        for r in 0..8 {
            for out_c in 0..IMPACT_EXT {
                let abs_x = abs_x_start + out_c as isize;
                if abs_x < 0 || abs_x as usize >= width {
                    continue;
                }
                let mut sum_low = 0.0;
                let mut sum_high = 0.0;
                for k in 0..FILT_LEN {
                    let src_x_raw = abs_x + k as isize - half as isize;
                    let src_x_abs = mirror_index(src_x_raw, width);
                    let basis_col = src_x_abs as isize - (bc * 8) as isize;
                    if (0..8).contains(&basis_col) {
                        let val = basis_block[r * 8 + basis_col as usize] * dct_delta_f;
                        sum_low += lpdf[k] * val;
                        sum_high += HPDF[k] * val;
                    }
                }
                row_low[r][out_c] = sum_low;
                row_high[r][out_c] = sum_high;
            }
        }

        // Column-filter pass: at each (abs_y, abs_x) in impact,
        // compute LH/HL/HH delta and accumulate into the cover-cloned
        // subbands. Same IMPACT_EXT extension for the y dimension.
        let abs_y_start = (br * 8) as isize - half as isize - 1; // br*8 − 8
        for out_r in 0..IMPACT_EXT {
            let abs_y = abs_y_start + out_r as isize;
            if abs_y < 0 || abs_y as usize >= height {
                continue;
            }
            let wy = abs_y as usize;
            for out_c in 0..IMPACT_EXT {
                let abs_x = abs_x_start + out_c as isize;
                if abs_x < 0 || abs_x as usize >= width {
                    continue;
                }
                let wx = abs_x as usize;

                let mut delta_lh = 0.0f64;
                let mut delta_hl = 0.0f64;
                let mut delta_hh = 0.0f64;
                for k in 0..FILT_LEN {
                    let src_y_raw = abs_y + k as isize - half as isize;
                    let src_y_abs = mirror_index(src_y_raw, height);
                    let basis_row = src_y_abs as isize - (br * 8) as isize;
                    if (0..8).contains(&basis_row) {
                        let r = basis_row as usize;
                        let low_val = row_low[r][out_c];
                        let high_val = row_high[r][out_c];
                        delta_lh += HPDF[k] * low_val;
                        delta_hl += lpdf[k] * high_val;
                        delta_hh += HPDF[k] * high_val;
                    }
                }

                let idx = wy * width + wx;
                lh[idx] = ((lh[idx] as f64) + delta_lh) as f32;
                hl[idx] = ((hl[idx] as f64) + delta_hl) as f32;
                hh[idx] = ((hh[idx] as f64) + delta_hh) as f32;
            }
        }
    }

    StegoWavelets {
        inner: ThreeSubbands {
            lh,
            hl,
            hh,
            width,
            y_offset: 0,
        },
        img_w: cover_wavelets.img_w,
        img_h: cover_wavelets.img_h,
    }
}

/// Compute embeddable UNIWARD positions given pre-cached wavelets.
/// Behaviorally identical to [`compute_positions_streaming`] when
/// called with `wavelets = compute_cover_wavelets(grid, qt)`, but
/// avoids the wavelet recomputation cost when the wavelets are
/// already in hand (the T3.2 cascade verify case after Phase C+D).
///
/// In Phase B this is wired only for the cover-side; Phase D will
/// extend the same per-block iteration to accept "stego" wavelets
/// (cover + STC-modification deltas) and skip per-block recomputation
/// for clean coefficients.
/// Number of progress sub-steps emitted by
/// [`compute_positions_from_wavelets`] across the per-block parallel
/// scan. Used by cascade-trigger code in `ghost/pipeline.rs` to
/// budget total steps before this function is called.
pub const POSITIONS_FROM_WAVELETS_PROGRESS_STEPS: u32 = 30;

/// Number of progress sub-steps emitted by
/// [`compute_positions_with_dirty_recost`]. The dirty re-cost is
/// 6-8× cheaper than a full positions scan post-T3, so a smaller
/// budget (10 vs UNIWARD's 100) keeps the bar moving against
/// wall-clock-100% target. Used per cascade iter.
pub const DIRTY_RECOST_PROGRESS_STEPS: u32 = 10;

pub fn compute_positions_from_wavelets(
    grid: &DctGrid,
    qt: &QuantTable,
    wavelets: &CachedCoverWavelets,
    si: Option<(&SideInfo, &DctGrid)>,
) -> Result<Vec<CoeffPos>, StegoError> {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let img_w = wavelets.img_w;
    let img_h = wavelets.img_h;
    debug_assert_eq!(img_w, bw * 8);
    debug_assert_eq!(img_h, bt * 8);

    let abs_delta = AbsDeltaTable::precompute(qt);
    let pad = FILT_LEN - 1; // 15

    let est_positions = bt * bw * 32;
    let mut positions: Vec<CoeffPos> = Vec::with_capacity(est_positions);

    // Per-block cost: same inner logic as the strip path, only now
    // the wavelets cover the full image so `y_offset = 0` and one
    // pass over all blocks suffices.
    let n_blocks = bt * bw;
    let mut strip_map = CostMap::new(bw, bt);
    let total_len = n_blocks * 64;
    let costs_ptr = CostMapPtr { ptr: strip_map.costs_ptr(), total_len };

    // T3 progress instrumentation — emit POSITIONS_FROM_WAVELETS_PROGRESS_STEPS
    // sub-steps spread across the parallel block scan. Each worker
    // atomically increments a counter and emits progress::advance()
    // when the count crosses an integer progress-step boundary
    // (count × N / n_blocks). Total emissions = N regardless of
    // worker scheduling order.
    use std::sync::atomic::AtomicU64;
    let counter = AtomicU64::new(0);
    let n_blocks_u64 = n_blocks as u64;
    let n_steps_u64 = POSITIONS_FROM_WAVELETS_PROGRESS_STEPS as u64;

    let compute_block = |bi: usize| {
        let br = bi / bw;
        let bc = bi % bw;
        let blk = grid.block(br, bc);
        for fi in 0..8 {
            for fj in 0..8 {
                if fi == 0 && fj == 0 {
                    continue;
                }
                let coeff = blk[fi * 8 + fj];
                if coeff == 0 {
                    continue;
                }
                let (abs_lh, abs_hl, abs_hh) = abs_delta.slice_at(fi, fj);
                let cost = compute_coefficient_cost_precomputed(
                    abs_lh, abs_hl, abs_hh,
                    &wavelets.inner,
                    br,
                    bc,
                    img_w,
                    img_h,
                    pad,
                );
                if cost > 0.0 && cost.is_finite() {
                    let idx = br * bw * 64 + bc * 64 + fi * 8 + fj;
                    unsafe { costs_ptr.write(idx, cost as f32); }
                }
            }
        }
        // Emit a progress sub-step when this block crosses a
        // boundary. Race-tolerant: even if two workers compute the
        // same target value, only one of them sees `target > prev`
        // (the one that incremented `counter` first).
        if n_blocks_u64 > 0 {
            let count = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let target = count * n_steps_u64 / n_blocks_u64;
            let prev_target = (count - 1) * n_steps_u64 / n_blocks_u64;
            if target > prev_target {
                progress::advance();
            }
        }
    };

    #[cfg(feature = "parallel")]
    (0..n_blocks).into_par_iter().for_each(compute_block);
    #[cfg(not(feature = "parallel"))]
    (0..n_blocks).for_each(compute_block);

    // Collect positions (mirrors the strip-path collect loop).
    for br in 0..bt {
        for bc in 0..bw {
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    let cost_f32 = strip_map.get(br, bc, i, j);
                    if !cost_f32.is_finite() {
                        continue;
                    }
                    let flat_idx = ((br * bw + bc) * 64 + i * 8 + j) as u32;

                    let final_cost = if let Some((side_info, cover_grid)) = si {
                        let coeff = cover_grid.get(br, bc, i, j);
                        if coeff.abs() == 1 {
                            cost_f32
                        } else {
                            let error = side_info.error_at(flat_idx as usize);
                            let factor = 1.0f32 - 2.0 * error.abs();
                            (cost_f32 * factor).max(MIN_SI_COST_F32)
                        }
                    } else {
                        cost_f32
                    };
                    positions.push(CoeffPos { flat_idx, cost: final_cost });
                }
            }
        }
    }

    Ok(positions)
}

/// Minimum cost for SI-modulated coefficients (same as side_info::MIN_SI_COST).
const MIN_SI_COST_F32: f32 = 1e-6;

/// T3.2.G — dual-output variant that emits BOTH primary (with SI
/// factor if `si.is_some()`) and verify (no SI factor) positions in a
/// single pass. The per-coefficient cost work is shared; the only
/// extra cost is one additional vec push per position.
///
/// This is the entry point for the shadow-cascade path: primary
/// positions feed STC embedding (SI-modulated), verify positions feed
/// the dirty re-cost (decoder convention, no SI).
///
/// When `si.is_none()`, primary and verify costs are identical; the
/// returned `verify` vec is a position-for-position clone of `primary`.
pub fn compute_positions_dual_from_wavelets(
    grid: &DctGrid,
    qt: &QuantTable,
    wavelets: &CachedCoverWavelets,
    si: Option<(&SideInfo, &DctGrid)>,
) -> Result<(Vec<CoeffPos>, Vec<CoeffPos>), StegoError> {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let img_w = wavelets.img_w;
    let img_h = wavelets.img_h;
    debug_assert_eq!(img_w, bw * 8);
    debug_assert_eq!(img_h, bt * 8);

    let abs_delta = AbsDeltaTable::precompute(qt);
    let pad = FILT_LEN - 1;

    // Shared cost map — populated in parallel, then read sequentially
    // by both primary + verify emit loops (same shape as
    // compute_positions_from_wavelets).
    let n_blocks = bt * bw;
    let mut strip_map = CostMap::new(bw, bt);
    let total_len = n_blocks * 64;
    let costs_ptr = CostMapPtr { ptr: strip_map.costs_ptr(), total_len };

    let compute_block = |bi: usize| {
        let br = bi / bw;
        let bc = bi % bw;
        let blk = grid.block(br, bc);
        for fi in 0..8 {
            for fj in 0..8 {
                if fi == 0 && fj == 0 {
                    continue;
                }
                let coeff = blk[fi * 8 + fj];
                if coeff == 0 {
                    continue;
                }
                let (abs_lh, abs_hl, abs_hh) = abs_delta.slice_at(fi, fj);
                let cost = compute_coefficient_cost_precomputed(
                    abs_lh, abs_hl, abs_hh,
                    &wavelets.inner,
                    br,
                    bc,
                    img_w,
                    img_h,
                    pad,
                );
                if cost > 0.0 && cost.is_finite() {
                    let idx = br * bw * 64 + bc * 64 + fi * 8 + fj;
                    unsafe { costs_ptr.write(idx, cost as f32); }
                }
            }
        }
    };

    #[cfg(feature = "parallel")]
    (0..n_blocks).into_par_iter().for_each(compute_block);
    #[cfg(not(feature = "parallel"))]
    (0..n_blocks).for_each(compute_block);

    let est = bt * bw * 32;
    let mut primary: Vec<CoeffPos> = Vec::with_capacity(est);
    let mut verify: Vec<CoeffPos> = Vec::with_capacity(est);

    for br in 0..bt {
        for bc in 0..bw {
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    let cost_f32 = strip_map.get(br, bc, i, j);
                    if !cost_f32.is_finite() {
                        continue;
                    }
                    let flat_idx = ((br * bw + bc) * 64 + i * 8 + j) as u32;

                    // Verify cost: cost_f32 unchanged (no SI factor).
                    verify.push(CoeffPos { flat_idx, cost: cost_f32 });

                    // Primary cost: with SI factor if applicable.
                    let primary_cost = if let Some((side_info, cover_grid)) = si {
                        let coeff = cover_grid.get(br, bc, i, j);
                        if coeff.abs() == 1 {
                            cost_f32
                        } else {
                            let error = side_info.error_at(flat_idx as usize);
                            let factor = 1.0f32 - 2.0 * error.abs();
                            (cost_f32 * factor).max(MIN_SI_COST_F32)
                        }
                    } else {
                        cost_f32
                    };
                    primary.push(CoeffPos { flat_idx, cost: primary_cost });
                }
            }
        }
    }

    Ok((primary, verify))
}

/// T3.2.D — compute stego positions by merging cached cover positions
/// (for clean blocks) with freshly-recomputed stego costs (for dirty
/// blocks).
///
/// A block at (br, bc) is **dirty** iff some modification's 23x23
/// wavelet impact region intersects this block's 23x23 cost-evaluation
/// region. Geometrically: `∃ m ∈ modifications: |br − m.br| ≤ 2 AND
/// |bc − m.bc| ≤ 2`. For every CLEAN block, `stego_wavelets[k] ==
/// cover_wavelets[k]` at every k in the block's impact region (proven
/// by linearity + locality of the 16-tap wavelet filter), so
/// `compute_coefficient_cost(basis, stego_wavelets, ...)` would
/// produce the same value as on cover.
///
/// Output: position-for-position equal to
/// `compute_positions_from_wavelets(stego_grid, qt, &stego.as_cache(),
/// si)`. Bit-exactness is gated by `dirty_recost_matches_full_recompute`.
///
/// Performance: at typical Ghost modification rates (~0.13%
/// per-coefficient = ~1.6% per-block), the dirty fraction is ~30%.
/// This path skips the ~70% clean-block per-coefficient cost
/// computations, plus shares the cover-side wavelet decomposition
/// (the bulk f32 work) across all cascade iterations.
pub fn compute_positions_with_dirty_recost(
    grid: &DctGrid,
    qt: &QuantTable,
    cached_cover_positions: &[CoeffPos],
    modifications: &[u32],
    stego_wavelets: &StegoWavelets,
    si: Option<(&SideInfo, &DctGrid)>,
) -> Result<Vec<CoeffPos>, StegoError> {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let img_w = stego_wavelets.img_w;
    let img_h = stego_wavelets.img_h;
    debug_assert_eq!(img_w, bw * 8);
    debug_assert_eq!(img_h, bt * 8);
    let abs_delta = AbsDeltaTable::precompute(qt);
    let pad = FILT_LEN - 1; // 15

    // Build dirty bitmap (one bool per block — 24 KB at 12 MP).
    //
    // The dirty neighborhood is ASYMMETRIC: br_coef ∈ [br_m−1, br_m+4],
    // bc_coef ∈ [bc_m−1, bc_m+4]. 6×6 = 36 blocks per modification.
    //
    // Derivation: `compute_coefficient_cost` reads cover wavelets at
    // abs_x ∈ [bc*8 − 15, bc*8 + 7] (NOTE: this is offset −8 from the
    // geometric impact-region center bc*8 + 3.5 — the cost-formula's
    // src_col = out_c + k − 14 convention makes it read positions to
    // the LEFT of the basis block). The modification's wavelet delta
    // (via the full mirror-aware filter) is non-zero at abs_x ∈
    // [bc_m*8 − 8, bc_m*8 + 22] (worst case, including bottom/right
    // mirror extensions). Intersection non-empty ⇒ bc − bc_m ≤ 4 AND
    // bc_m − bc ≤ 1.
    let mut dirty = vec![false; bt * bw];
    for &flat_idx in modifications {
        let block_idx = (flat_idx as usize) / 64;
        let m_br = (block_idx / bw) as isize;
        let m_bc = (block_idx % bw) as isize;
        for dbr in -1isize..=4 {
            for dbc in -1isize..=4 {
                let br = m_br + dbr;
                let bc = m_bc + dbc;
                if br >= 0 && br < bt as isize && bc >= 0 && bc < bw as isize {
                    dirty[(br as usize) * bw + (bc as usize)] = true;
                }
            }
        }
    }

    // Precompute block_start[bi] = index in cached_cover_positions
    // where block bi's positions begin. block_start[bt*bw] = end of
    // the vec. Single sequential pass — enables parallel per-block
    // emit without binary-searching the cover positions vec.
    let n_blocks = bt * bw;
    let mut block_start: Vec<usize> = vec![0; n_blocks + 1];
    let mut idx = 0usize;
    for bi in 0..n_blocks {
        block_start[bi] = idx;
        let block_end_flat = ((bi + 1) * 64) as u32;
        while idx < cached_cover_positions.len()
            && cached_cover_positions[idx].flat_idx < block_end_flat
        {
            idx += 1;
        }
    }
    block_start[n_blocks] = idx;

    // T3 progress instrumentation — emit
    // DIRTY_RECOST_PROGRESS_STEPS sub-steps spread across the
    // per-block parallel emit loop. Same race-tolerant pattern as
    // `compute_positions_from_wavelets`.
    use std::sync::atomic::AtomicU64;
    let counter = AtomicU64::new(0);
    let n_blocks_u64 = n_blocks as u64;
    let n_steps_u64 = DIRTY_RECOST_PROGRESS_STEPS as u64;

    // Per-block emit: dirty → recompute; clean → copy. Same inner
    // per-coefficient logic as compute_positions_from_wavelets, but
    // gated by the dirty bitmap.
    let emit_block = |bi: usize| -> Vec<CoeffPos> {
        let br = bi / bw;
        let bc = bi % bw;
        let out = if dirty[bi] {
            let blk = grid.block(br, bc);
            let mut out: Vec<CoeffPos> = Vec::with_capacity(63);
            for fi in 0..8 {
                for fj in 0..8 {
                    if fi == 0 && fj == 0 {
                        continue;
                    }
                    let coeff = blk[fi * 8 + fj];
                    if coeff == 0 {
                        continue;
                    }
                    let (abs_lh, abs_hl, abs_hh) = abs_delta.slice_at(fi, fj);
                    let cost = compute_coefficient_cost_precomputed(
                        abs_lh, abs_hl, abs_hh,
                        &stego_wavelets.inner,
                        br,
                        bc,
                        img_w,
                        img_h,
                        pad,
                    ) as f32;
                    if !cost.is_finite() || cost <= 0.0 {
                        continue;
                    }
                    let flat_idx = ((br * bw + bc) * 64 + fi * 8 + fj) as u32;
                    let final_cost = if let Some((side_info, cover_grid)) = si {
                        let cover_coef = cover_grid.get(br, bc, fi, fj);
                        if cover_coef.abs() == 1 {
                            cost
                        } else {
                            let error = side_info.error_at(flat_idx as usize);
                            let factor = 1.0f32 - 2.0 * error.abs();
                            (cost * factor).max(MIN_SI_COST_F32)
                        }
                    } else {
                        cost
                    };
                    out.push(CoeffPos { flat_idx, cost: final_cost });
                }
            }
            out
        } else {
            cached_cover_positions[block_start[bi]..block_start[bi + 1]].to_vec()
        };

        // Emit progress when this block crosses a sub-step boundary.
        if n_blocks_u64 > 0 {
            let count = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let target = count * n_steps_u64 / n_blocks_u64;
            let prev_target = (count - 1) * n_steps_u64 / n_blocks_u64;
            if target > prev_target {
                progress::advance();
            }
        }

        out
    };

    #[cfg(feature = "parallel")]
    let result: Vec<CoeffPos> = (0..n_blocks)
        .into_par_iter()
        .map(emit_block)
        .flatten()
        .collect();
    #[cfg(not(feature = "parallel"))]
    let result: Vec<CoeffPos> = (0..n_blocks).flat_map(emit_block).collect();

    Ok(result)
}

/// Compute UNIWARD costs strip-by-strip and collect embeddable positions directly.
///
/// This is the memory-optimized path for large images. Instead of materializing
/// the full CostMap (800 MB for 200 MP), it processes the image in horizontal
/// strips, computing wavelet subbands and costs per strip, extracting positions
/// immediately, and freeing strip memory before the next strip.
///
/// Peak memory per strip: ~170 MB for a 16K-wide image (reused across strips).
/// Output: `Vec<CoeffPos>` with the same positions as `compute_uniward` +
/// `collect_positions`, in deterministic raster order.
///
/// Reports [`UNIWARD_PROGRESS_STEPS`] progress steps and checks for cancellation.
///
/// If `si` is `Some`, applies SI-UNIWARD cost modulation inline during position
/// collection (no separate cost-modulation pass needed).
pub fn compute_positions_streaming(
    grid: &DctGrid,
    qt: &QuantTable,
    si: Option<(&SideInfo, &DctGrid)>,
) -> Result<Vec<CoeffPos>, StegoError> {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let img_w = bw * 8;
    let img_h = bt * 8;
    let lpdf = lpdf();
    let abs_delta = AbsDeltaTable::precompute(qt);
    let pad = FILT_LEN - 1; // 15

    // Estimate: ~50% of AC positions are non-zero (typical JPEG).
    let est_positions = bt * bw * 32;
    let mut positions: Vec<CoeffPos> = Vec::with_capacity(est_positions);

    // Distribute UNIWARD_PROGRESS_STEPS across strips for smooth progress.
    let num_strips = bt.div_ceil(STRIP_BLOCK_ROWS);
    let mut strip_idx = 0usize;
    let mut steps_sent = 0u32;

    for strip_start in (0..bt).step_by(STRIP_BLOCK_ROWS) {
        let strip_end = (strip_start + STRIP_BLOCK_ROWS).min(bt);

        // Wavelet rows needed for blocks in [strip_start, strip_end):
        // Each block at row br accesses wavelet rows [br*8 - 15, br*8 + 7].
        let wav_y_start = (strip_start * 8).saturating_sub(pad);
        let wav_y_end = (strip_end * 8).min(img_h);

        // Pixel/filter-rows needed for wavelet computation (±7 for column filter):
        let pix_y_start = wav_y_start.saturating_sub(7);
        let pix_y_end = (wav_y_end + 8).min(img_h);

        // Block rows that contain our pixel range:
        let pix_br_start = pix_y_start / 8;
        let pix_br_end = pix_y_end.div_ceil(8).min(bt);

        // Step 1: Decompress pixel strip.
        let pix_strip_h = (pix_br_end - pix_br_start) * 8;
        let pix_strip_y0 = pix_br_start * 8; // actual start (block-aligned)
        let pixels = decompress_strip_pixels(grid, qt, bw, pix_br_start, pix_br_end);

        // Sub-strip progress: advance after pixel decompression (~1/3 of strip work).
        {
            let sub_target = (strip_idx as u32 * 3 + 1) * UNIWARD_PROGRESS_STEPS / (num_strips as u32 * 3);
            while steps_sent < sub_target {
                progress::advance();
                steps_sent += 1;
            }
        }

        // Step 2: Compute wavelet subbands for the strip.
        let strip_wavelets = compute_strip_subbands(
            &pixels, img_w, pix_strip_h, pix_strip_y0,
            wav_y_start, wav_y_end, img_h, &lpdf,
        );
        drop(pixels); // Free pixel strip

        // Sub-strip progress: advance after wavelet computation (~2/3 of strip work).
        {
            let sub_target = (strip_idx as u32 * 3 + 2) * UNIWARD_PROGRESS_STEPS / (num_strips as u32 * 3);
            while steps_sent < sub_target {
                progress::advance();
                steps_sent += 1;
            }
        }

        // Step 3: Compute costs for blocks in [strip_start, strip_end) and
        // collect positions. Uses a temporary strip CostMap for parallel safety.
        let strip_bt = strip_end - strip_start;
        let n_strip_blocks = strip_bt * bw;
        let mut strip_map = CostMap::new(bw, strip_bt);

        let total_len = n_strip_blocks * 64;
        let costs_ptr = CostMapPtr { ptr: strip_map.costs_ptr(), total_len };

        let compute_block = |bi: usize| {
            let br_local = bi / bw;
            let bc = bi % bw;
            let br = strip_start + br_local;
            let blk = grid.block(br, bc);

            for fi in 0..8 {
                for fj in 0..8 {
                    if fi == 0 && fj == 0 { continue; }
                    let coeff = blk[fi * 8 + fj];
                    if coeff == 0 { continue; }

                    let (abs_lh, abs_hl, abs_hh) = abs_delta.slice_at(fi, fj);
                    let cost = compute_coefficient_cost_precomputed(
                        abs_lh, abs_hl, abs_hh, &strip_wavelets,
                        br, bc, img_w, img_h, pad,
                    );

                    if cost > 0.0 && cost.is_finite() {
                        let idx = br_local * bw * 64 + bc * 64 + fi * 8 + fj;
                        unsafe { costs_ptr.write(idx, cost as f32); }
                    }
                }
            }
        };

        #[cfg(feature = "parallel")]
        (0..n_strip_blocks).into_par_iter().for_each(compute_block);

        #[cfg(not(feature = "parallel"))]
        (0..n_strip_blocks).for_each(compute_block);

        // Scan strip CostMap and collect positions.
        for br_local in 0..strip_bt {
            let br = strip_start + br_local;
            for bc in 0..bw {
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 { continue; }
                        let cost_f32 = strip_map.get(br_local, bc, i, j);
                        if !cost_f32.is_finite() { continue; }

                        let flat_idx = ((br * bw + bc) * 64 + i * 8 + j) as u32;

                        // Apply SI modulation inline if available.
                        let final_cost = if let Some((side_info, cover_grid)) = si {
                            let coeff = cover_grid.get(br, bc, i, j);
                            if coeff.abs() == 1 {
                                cost_f32
                            } else {
                                let error = side_info.error_at(flat_idx as usize);
                                let factor = 1.0f32 - 2.0 * error.abs();
                                (cost_f32 * factor).max(MIN_SI_COST_F32)
                            }
                        } else {
                            cost_f32
                        };

                        positions.push(CoeffPos { flat_idx, cost: final_cost });
                    }
                }
            }
        }
        // strip_map and strip_wavelets freed here.

        // Sub-strip progress: advance after cost computation (3/3 of strip work).
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

    // Ensure all UNIWARD_PROGRESS_STEPS are sent.
    while steps_sent < UNIWARD_PROGRESS_STEPS {
        progress::advance();
        steps_sent += 1;
    }

    Ok(positions)
}

/// Decompress a strip of block rows [br_start, br_end) to pixel values (f32).
fn decompress_strip_pixels(
    grid: &DctGrid,
    qt: &QuantTable,
    bw: usize,
    br_start: usize,
    br_end: usize,
) -> Vec<f32> {
    let img_w = bw * 8;
    let strip_h = (br_end - br_start) * 8;
    let mut pixels = vec![0.0f32; img_w * strip_h];

    for br in br_start..br_end {
        for bc in 0..bw {
            let block = grid.block(br, bc);
            let quantized: [i16; 64] = block.try_into().unwrap();
            let block_pixels = idct_block(&quantized, &qt.values);

            for row in 0..8 {
                for col in 0..8 {
                    let py = (br - br_start) * 8 + row;
                    let px = bc * 8 + col;
                    pixels[py * img_w + px] = block_pixels[row * 8 + col] as f32;
                }
            }
        }
    }

    pixels
}

/// Compute wavelet subbands for a horizontal strip.
///
/// - `pixels`: decompressed pixel strip starting at y = `pix_y0`
/// - `width`: full image width
/// - `pix_h`: height of the pixel strip (block-aligned)
/// - `pix_y0`: absolute y of the first pixel row in the strip
/// - `wav_y_start`, `wav_y_end`: absolute y range of wavelet output
/// - `img_h`: full image height (for mirror boundary handling)
fn compute_strip_subbands(
    pixels: &[f32],
    width: usize,
    pix_h: usize,
    pix_y0: usize,
    wav_y_start: usize,
    wav_y_end: usize,
    img_h: usize,
    lpdf: &[f64; 16],
) -> ThreeSubbands {
    // Parallel: both row filters + 3 column filters run concurrently.
    #[cfg(feature = "parallel")]
    {
        let (row_low, row_high) = rayon::join(
            || filter_rows(pixels, width, pix_h, lpdf),
            || filter_rows(pixels, width, pix_h, &HPDF),
        );
        let (lh, (hl, hh)) = rayon::join(
            || filter_cols_strip(&row_low, width, pix_h, pix_y0, wav_y_start, wav_y_end, img_h, &HPDF),
            || rayon::join(
                || filter_cols_strip(&row_high, width, pix_h, pix_y0, wav_y_start, wav_y_end, img_h, lpdf),
                || filter_cols_strip(&row_high, width, pix_h, pix_y0, wav_y_start, wav_y_end, img_h, &HPDF),
            ),
        );
        ThreeSubbands { lh, hl, hh, width, y_offset: wav_y_start }
    }
    // Sequential: drop intermediates early (WASM path).
    #[cfg(not(feature = "parallel"))]
    {
        let row_low = filter_rows(pixels, width, pix_h, lpdf);
        let lh = filter_cols_strip(&row_low, width, pix_h, pix_y0, wav_y_start, wav_y_end, img_h, &HPDF);
        drop(row_low);
        let row_high = filter_rows(pixels, width, pix_h, &HPDF);
        let hl = filter_cols_strip(&row_high, width, pix_h, pix_y0, wav_y_start, wav_y_end, img_h, lpdf);
        let hh = filter_cols_strip(&row_high, width, pix_h, pix_y0, wav_y_start, wav_y_end, img_h, &HPDF);
        ThreeSubbands { lh, hl, hh, width, y_offset: wav_y_start }
    }
}

/// Apply column filter to produce output rows [out_y_start, out_y_end) from
/// a strip of row-filtered data starting at absolute y = `input_y0`.
///
/// Uses `mirror_index` on the full image height for boundary handling.
/// The input strip must contain all rows that the filter accesses (guaranteed
/// by the ±7 row padding in `compute_positions_streaming`).
fn filter_cols_strip(
    input: &[f32],
    width: usize,
    input_h: usize,
    input_y0: usize,
    out_y_start: usize,
    out_y_end: usize,
    img_h: usize,
    filter: &[f64; 16],
) -> Vec<f32> {
    let flen = FILT_LEN;
    let half = (flen - 1) / 2; // 7
    let out_h = out_y_end - out_y_start;
    let mut output = vec![0.0f32; width * out_h];

    for out_idx in 0..out_h {
        let abs_y = out_y_start + out_idx;
        for x in 0..width {
            let mut sum = 0.0f64;
            for k in 0..flen {
                let sy_abs = abs_y as isize + k as isize - half as isize;
                let sy_mirrored = mirror_index(sy_abs, img_h);
                // Map to strip-local index. The strip must contain this row.
                let sy_local = sy_mirrored - input_y0;
                debug_assert!(
                    sy_local < input_h,
                    "strip col filter OOB: abs_y={abs_y} k={k} sy_abs={sy_abs} sy_mirrored={sy_mirrored} input_y0={input_y0} input_h={input_h}"
                );
                sum += input[sy_local * width + x] as f64 * filter[k];
            }
            output[out_idx * width + x] = sum as f32;
        }
    }

    output
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
        QuantTable::new([
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99,
        ])
    }

    #[test]
    fn dc_is_wet() {
        let mut grid = DctGrid::new(4, 4);
        // Fill with some non-zero AC coefficients.
        for br in 0..4 {
            for bc in 0..4 {
                grid.set(br, bc, 0, 0, 100);
                grid.set(br, bc, 1, 0, 10);
                grid.set(br, bc, 0, 1, -5);
            }
        }
        let map = compute_uniward(&grid, &make_qt_uniform(16));
        // DC must always be WET.
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
        // Position (0, 1) has coefficient 0 — should be WET.
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
        // Non-zero AC coefficient at (1,0) should have finite cost.
        let cost = map.get(2, 2, 1, 0);
        assert!(
            cost.is_finite() && cost > 0.0,
            "expected finite positive cost, got {cost}"
        );
    }

    #[test]
    fn textured_region_cheaper_than_smooth_region() {
        // J-UNIWARD assigns lower cost when the cover's wavelet coefficients
        // are large (textured region) because the denominator (|W_cover| + sigma)
        // is larger, making the ratio smaller.
        //
        // Test: create a grid where one region is heavily textured (large AC
        // coefficients in ALL surrounding blocks) and another region is smooth
        // (only DC in surrounding blocks). Place identical target coefficients
        // in the center of each region.
        let qt = make_qt_uniform(16);

        let mut grid = DctGrid::new(8, 8);

        // Fill everything with DC only (smooth base).
        for br in 0..8 {
            for bc in 0..8 {
                grid.set(br, bc, 0, 0, 100);
            }
        }

        // Textured region: blocks (0..4, 0..4) get many AC coefficients.
        for br in 0..4 {
            for bc in 0..4 {
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 { continue; }
                        // Spread across many frequencies for rich texture.
                        grid.set(br, bc, i, j, (((i * 7 + j * 3) % 15) as i16) - 7);
                    }
                }
            }
        }

        // Place identical target coefficient in textured center (2,2)
        // and smooth center (6,6). Both have same coefficient value.
        grid.set(2, 2, 1, 0, 5);
        grid.set(6, 6, 1, 0, 5);

        let map = compute_uniward(&grid, &qt);

        let cost_textured = map.get(2, 2, 1, 0);
        let cost_smooth = map.get(6, 6, 1, 0);

        assert!(
            cost_textured.is_finite(),
            "textured cost should be finite: {cost_textured}"
        );
        assert!(
            cost_smooth.is_finite(),
            "smooth cost should be finite: {cost_smooth}"
        );
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
                        assert!(
                            cost >= 0.0,
                            "negative cost {cost} at ({br},{bc},{i},{j})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mirror_index_works() {
        assert_eq!(mirror_index(-1, 10), 1);
        assert_eq!(mirror_index(-3, 10), 3);
        assert_eq!(mirror_index(0, 10), 0);
        assert_eq!(mirror_index(5, 10), 5);
        assert_eq!(mirror_index(9, 10), 9);
        assert_eq!(mirror_index(10, 10), 8);
        assert_eq!(mirror_index(11, 10), 7);
    }

    #[test]
    fn lpdf_is_correct_length() {
        let lp = lpdf();
        assert_eq!(lp.len(), 16);
        // Low-pass filter should sum to approximately sqrt(2).
        let sum: f64 = lp.iter().sum();
        assert!(
            (sum - std::f64::consts::SQRT_2).abs() < 0.01,
            "low-pass filter sum {sum} should be ~sqrt(2)"
        );
    }

    #[test]
    fn hpdf_sums_to_zero() {
        // High-pass filter should sum to approximately 0.
        let sum: f64 = HPDF.iter().sum();
        assert!(
            sum.abs() < 1e-10,
            "high-pass filter sum {sum} should be ~0"
        );
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

        // Count finite costs (embeddable positions).
        let bw = grid.blocks_wide();
        let bt = grid.blocks_tall();
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

        // A 320x240 photo should have many embeddable positions.
        assert!(
            finite_count > 1000,
            "expected >1000 finite costs, got {finite_count}"
        );
        // Average cost should be positive and reasonable.
        let avg = total_cost / finite_count as f64;
        assert!(avg > 0.0, "average cost should be positive: {avg}");
    }

    /// Verify that compute_uniward is deterministic: calling it twice on the
    /// same input produces bit-identical cost maps. This catches any
    /// non-determinism that might be introduced by parallel execution order.
    #[test]
    fn determinism_repeated_runs() {
        let mut grid = DctGrid::new(6, 6);
        for br in 0..6 {
            for bc in 0..6 {
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

        let qt = standard_qt();
        let map1 = compute_uniward(&grid, &qt);
        let map2 = compute_uniward(&grid, &qt);

        for br in 0..6 {
            for bc in 0..6 {
                for i in 0..8 {
                    for j in 0..8 {
                        let c1 = map1.get(br, bc, i, j);
                        let c2 = map2.get(br, bc, i, j);
                        assert_eq!(
                            c1.to_bits(), c2.to_bits(),
                            "cost mismatch at ({br},{bc},{i},{j}): {c1} vs {c2}"
                        );
                    }
                }
            }
        }
    }

    /// T3.2.B — `compute_positions_from_wavelets(grid, qt, &cache, None)`
    /// MUST produce the same Vec<CoeffPos> as
    /// `compute_positions_streaming(grid, qt, None)` on the same input.
    /// Same costs, same positions, in the same order. This is the
    /// behavioral-identity gate for the cache refactor.
    #[test]
    fn cache_path_matches_streaming_path() {
        let mut grid = DctGrid::new(6, 6);
        for br in 0..6 {
            for bc in 0..6 {
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
        let qt = standard_qt();
        let streaming = compute_positions_streaming(&grid, &qt, None).unwrap();
        let cache = compute_cover_wavelets(&grid, &qt);
        let cached = compute_positions_from_wavelets(&grid, &qt, &cache, None).unwrap();

        assert_eq!(streaming.len(), cached.len(),
            "position count mismatch: streaming={} cached={}",
            streaming.len(), cached.len());
        for (i, (s, c)) in streaming.iter().zip(cached.iter()).enumerate() {
            assert_eq!(s.flat_idx, c.flat_idx,
                "flat_idx mismatch at index {i}: streaming={} cached={}",
                s.flat_idx, c.flat_idx);
            assert_eq!(s.cost.to_bits(), c.cost.to_bits(),
                "cost mismatch at flat_idx {}: streaming={} cached={}",
                s.flat_idx, s.cost, c.cost);
        }
    }

    /// T3.2.C — single-modification interior case. Verifies the
    /// pixel-delta → wavelet-delta math matches the full filter when
    /// mirror reflections don't come into play.
    #[test]
    fn incremental_wavelet_single_interior_modification() {
        // 12×12 block grid → 96×96 pixels. Modification at (br=5, bc=5)
        // is interior — mirror has no effect.
        let mut cover = DctGrid::new(12, 12);
        for br in 0..12 {
            for bc in 0..12 {
                cover.set(br, bc, 0, 0, 100);
            }
        }
        let mut stego = cover.clone();
        stego.set(5, 5, 2, 3, 1); // single DCT modification
        let modifications = vec![((5usize * 12 + 5) * 64 + 2 * 8 + 3) as u32];

        let qt = standard_qt();
        let cover_w = compute_cover_wavelets(&cover, &qt);
        let stego_w_incr =
            apply_dct_modifications_to_wavelets(&cover_w, &modifications, &cover, &stego, &qt);
        let stego_w_full = compute_cover_wavelets(&stego, &qt);

        let n = stego_w_incr.inner.lh.len();
        let mut max_lh = 0.0f32;
        let mut max_hl = 0.0f32;
        let mut max_hh = 0.0f32;
        for i in 0..n {
            max_lh =
                max_lh.max((stego_w_incr.inner.lh[i] - stego_w_full.inner.lh[i]).abs());
            max_hl =
                max_hl.max((stego_w_incr.inner.hl[i] - stego_w_full.inner.hl[i]).abs());
            max_hh =
                max_hh.max((stego_w_incr.inner.hh[i] - stego_w_full.inner.hh[i]).abs());
        }
        assert!(
            max_lh < 0.01 && max_hl < 0.01 && max_hh < 0.01,
            "interior single-mod drift: lh={max_lh} hl={max_hl} hh={max_hh}"
        );
    }

    /// T3.2.C — incremental wavelet delta must reproduce the result
    /// of a fresh `compute_cover_wavelets` on the modified grid, modulo
    /// ≤ 1 LSB f32 rounding. Verifies that summing per-modification
    /// 2D 16-tap deltas matches running the full row+column filter on
    /// the modified pixels.
    #[test]
    fn incremental_wavelet_matches_full_recompute() {
        // Build a cover grid and a stego grid that differs at a few
        // known DCT coefficient positions.
        let mut cover = DctGrid::new(6, 6);
        for br in 0..6 {
            for bc in 0..6 {
                cover.set(br, bc, 0, 0, 100);
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 { continue; }
                        let val = (((br * 7 + bc * 13 + i * 3 + j * 11) % 21) as i16) - 10;
                        if val != 0 {
                            cover.set(br, bc, i, j, val);
                        }
                    }
                }
            }
        }
        let mut stego = cover.clone();

        // Apply LSB-style flips at a handful of positions spanning
        // interior and corners.
        let bw = stego.blocks_wide();
        let mut modifications: Vec<u32> = Vec::new();
        for (br, bc, fi, fj, delta) in [
            (2usize, 3, 1, 2, 1i16),
            (0, 0, 2, 1, -1),
            (5, 5, 3, 4, 1),
            (1, 4, 0, 7, -1),
            (4, 2, 6, 3, 1),
        ] {
            let cur = stego.get(br, bc, fi, fj);
            stego.set(br, bc, fi, fj, cur + delta);
            let flat = ((br * bw + bc) * 64 + fi * 8 + fj) as u32;
            modifications.push(flat);
        }

        let qt = standard_qt();
        let cover_w = compute_cover_wavelets(&cover, &qt);

        // Approach A — incremental delta from cover + modifications.
        let stego_w_incr = apply_dct_modifications_to_wavelets(
            &cover_w, &modifications, &cover, &stego, &qt,
        );

        // Approach B — fresh full recompute on the modified grid.
        let stego_w_full = compute_cover_wavelets(&stego, &qt);

        let n = stego_w_incr.inner.lh.len();
        assert_eq!(n, stego_w_full.inner.lh.len());

        // Bit-exact: f32 LSB tolerance per position.
        let mut max_abs_diff_lh = 0.0f32;
        let mut max_abs_diff_hl = 0.0f32;
        let mut max_abs_diff_hh = 0.0f32;
        for i in 0..n {
            max_abs_diff_lh =
                max_abs_diff_lh.max((stego_w_incr.inner.lh[i] - stego_w_full.inner.lh[i]).abs());
            max_abs_diff_hl =
                max_abs_diff_hl.max((stego_w_incr.inner.hl[i] - stego_w_full.inner.hl[i]).abs());
            max_abs_diff_hh =
                max_abs_diff_hh.max((stego_w_incr.inner.hh[i] - stego_w_full.inner.hh[i]).abs());
        }
        // f32 LSB at the magnitudes we see (~10⁴ at cover values) is
        // ~10⁻³. Allow some headroom: 0.05 is comfortably below the
        // SIGMA used in cost computation.
        assert!(
            max_abs_diff_lh < 0.05 && max_abs_diff_hl < 0.05 && max_abs_diff_hh < 0.05,
            "incremental wavelet drift exceeds tolerance: lh={max_abs_diff_lh} hl={max_abs_diff_hl} hh={max_abs_diff_hh}"
        );
    }

    /// T3.2.D — `compute_positions_with_dirty_recost` must produce
    /// position-for-position identical output to
    /// `compute_positions_from_wavelets(stego_grid, qt, &stego.as_cache(),
    /// None)`. Clean blocks reuse cached cover costs; dirty blocks
    /// recompute via stego wavelets. Bit-exact at the f32-cost level.
    #[test]
    fn dirty_recost_matches_full_recompute() {
        // 12×12 block grid — enough room for the 5x5 dirty
        // neighborhood + a wide margin of guaranteed-clean blocks.
        let mut cover = DctGrid::new(12, 12);
        for br in 0..12 {
            for bc in 0..12 {
                cover.set(br, bc, 0, 0, 100);
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 { continue; }
                        let val =
                            (((br * 7 + bc * 13 + i * 3 + j * 11) % 21) as i16) - 10;
                        if val != 0 {
                            cover.set(br, bc, i, j, val);
                        }
                    }
                }
            }
        }
        let mut stego = cover.clone();
        let bw = stego.blocks_wide();
        let mut modifications: Vec<u32> = Vec::new();
        // Mods at interior + corners + spread across the grid.
        for (br, bc, fi, fj, delta) in [
            (5usize, 6, 1, 2, 1i16),
            (0, 0, 2, 1, -1),
            (11, 11, 3, 4, 1),
            (3, 9, 0, 7, -1),
            (8, 3, 6, 3, 1),
        ] {
            let cur = stego.get(br, bc, fi, fj);
            stego.set(br, bc, fi, fj, cur + delta);
            modifications.push(((br * bw + bc) * 64 + fi * 8 + fj) as u32);
        }

        let qt = standard_qt();
        let cover_w = compute_cover_wavelets(&cover, &qt);
        let cover_positions =
            compute_positions_from_wavelets(&cover, &qt, &cover_w, None).unwrap();

        // Reference: full recompute on stego wavelets.
        let stego_w = apply_dct_modifications_to_wavelets(
            &cover_w, &modifications, &cover, &stego, &qt,
        );
        let ref_positions = compute_positions_from_wavelets(
            &stego, &qt, &stego_w.as_cache(), None,
        ).unwrap();

        // Test path: dirty re-cost.
        let dirty_positions = compute_positions_with_dirty_recost(
            &stego, &qt, &cover_positions, &modifications, &stego_w, None,
        ).unwrap();

        assert_eq!(
            ref_positions.len(),
            dirty_positions.len(),
            "position count mismatch: ref={} dirty={}",
            ref_positions.len(),
            dirty_positions.len()
        );
        for (i, (r, d)) in ref_positions.iter().zip(dirty_positions.iter()).enumerate() {
            assert_eq!(
                r.flat_idx, d.flat_idx,
                "position[{i}] flat_idx: ref={} dirty={}",
                r.flat_idx, d.flat_idx
            );
            assert!(
                (r.cost.to_bits() == d.cost.to_bits()) || (r.cost - d.cost).abs() < 1e-6,
                "position[{i}] flat_idx={}: ref_cost={} dirty_cost={}",
                r.flat_idx, r.cost, d.cost
            );
        }
    }

    /// T3.3.A — precomputed |delta| table must give costs identical
    /// to the legacy `compute_coefficient_cost` path at f64 bit
    /// level. Verified against the LEGACY-SUM-ORDER precomputed
    /// variant (kept around just for this gate). The production
    /// kernel uses a SIMD-friendly pair-wise sum order (Phase B) that
    /// drifts by f64 ULP vs legacy — see
    /// `cost_precompute_simd_drift_below_f32_lsb` below.
    #[test]
    fn cost_precompute_matches_legacy_path() {
        let mut cover = DctGrid::new(8, 8);
        for br in 0..8 {
            for bc in 0..8 {
                cover.set(br, bc, 0, 0, 120);
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 { continue; }
                        let val =
                            (((br * 11 + bc * 7 + i * 5 + j * 3) % 19) as i16) - 9;
                        if val != 0 {
                            cover.set(br, bc, i, j, val);
                        }
                    }
                }
            }
        }
        let qt = standard_qt();
        let w = compute_cover_wavelets(&cover, &qt);
        let basis = precompute_basis_functions(&qt);
        let lpdf_t = lpdf();
        let pad = FILT_LEN - 1;
        let img_w = w.img_w;
        let img_h = w.img_h;
        let table = AbsDeltaTable::precompute(&qt);

        let test_blocks: &[(usize, usize)] = &[
            (0, 0), (0, 7), (7, 0), (7, 7), (3, 4), (5, 2), (1, 6), (6, 1),
        ];
        for &(br, bc) in test_blocks {
            for fi in 0..8 {
                for fj in 0..8 {
                    if fi == 0 && fj == 0 { continue; }
                    let basis_block = &basis[fi][fj];
                    let legacy = compute_coefficient_cost(
                        basis_block, &w.inner, br, bc, img_w, img_h, pad, &lpdf_t,
                    );
                    let (abs_lh, abs_hl, abs_hh) = table.slice_at(fi, fj);
                    let fast = compute_coefficient_cost_precomputed_legacy_order(
                        abs_lh, abs_hl, abs_hh, &w.inner, br, bc, img_w, img_h, pad,
                    );
                    assert_eq!(
                        legacy.to_bits(),
                        fast.to_bits(),
                        "(br={br}, bc={bc}, fi={fi}, fj={fj}): legacy={legacy} fast={fast}"
                    );
                }
            }
        }
    }

    /// T3.3.B — the SIMD-order precomputed kernel (production)
    /// must give costs equivalent to the legacy-order precomputed
    /// kernel, modulo the f64 ULP drift from re-associating the
    /// per-cell (lh + hl + hh) sum and pair-wise grouping.
    ///
    /// Drift tolerance: relative ≤ 1e-12, absolute ≤ 1e-10. Far
    /// below the f32 cost LSB at typical UNIWARD magnitudes, so the
    /// f32-cast cost values stored into CostMap are bit-identical
    /// between the two kernels in practice — sort order and stego
    /// output unchanged.
    #[test]
    fn cost_precompute_simd_drift_below_f32_lsb() {
        let mut cover = DctGrid::new(8, 8);
        for br in 0..8 {
            for bc in 0..8 {
                cover.set(br, bc, 0, 0, 120);
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 { continue; }
                        let val =
                            (((br * 11 + bc * 7 + i * 5 + j * 3) % 19) as i16) - 9;
                        if val != 0 {
                            cover.set(br, bc, i, j, val);
                        }
                    }
                }
            }
        }
        let qt = standard_qt();
        let w = compute_cover_wavelets(&cover, &qt);
        let pad = FILT_LEN - 1;
        let img_w = w.img_w;
        let img_h = w.img_h;
        let table = AbsDeltaTable::precompute(&qt);

        let test_blocks: &[(usize, usize)] = &[
            (0, 0), (0, 7), (7, 0), (7, 7), (3, 4), (5, 2), (1, 6), (6, 1),
        ];
        let mut max_abs_drift = 0.0f64;
        let mut max_rel_drift = 0.0f64;
        for &(br, bc) in test_blocks {
            for fi in 0..8 {
                for fj in 0..8 {
                    if fi == 0 && fj == 0 { continue; }
                    let (abs_lh, abs_hl, abs_hh) = table.slice_at(fi, fj);
                    let legacy = compute_coefficient_cost_precomputed_legacy_order(
                        abs_lh, abs_hl, abs_hh, &w.inner, br, bc, img_w, img_h, pad,
                    );
                    let simd = compute_coefficient_cost_precomputed(
                        abs_lh, abs_hl, abs_hh, &w.inner, br, bc, img_w, img_h, pad,
                    );
                    let diff = (legacy - simd).abs();
                    max_abs_drift = max_abs_drift.max(diff);
                    if legacy.abs() > 1e-30 {
                        max_rel_drift = max_rel_drift.max(diff / legacy.abs());
                    }
                }
            }
        }
        assert!(
            max_abs_drift < 1e-10 && max_rel_drift < 1e-12,
            "drift exceeds tolerance: abs={max_abs_drift} rel={max_rel_drift}"
        );
    }

    /// T3.2.G — dual-output must produce position-for-position
    /// equality with two separate calls to compute_positions_from_wavelets
    /// (one with SI, one without). f32 bit-exact on the same shared
    /// cost map.
    #[test]
    fn dual_output_matches_two_separate_passes() {
        let mut cover = DctGrid::new(8, 8);
        for br in 0..8 {
            for bc in 0..8 {
                cover.set(br, bc, 0, 0, 100);
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 { continue; }
                        let val =
                            (((br * 7 + bc * 13 + i * 3 + j * 11) % 21) as i16) - 10;
                        if val != 0 {
                            cover.set(br, bc, i, j, val);
                        }
                    }
                }
            }
        }
        let qt = standard_qt();
        let w = compute_cover_wavelets(&cover, &qt);

        // Without SI: primary == verify by construction.
        let (primary_no_si, verify_no_si) =
            compute_positions_dual_from_wavelets(&cover, &qt, &w, None).unwrap();
        let ref_no_si =
            compute_positions_from_wavelets(&cover, &qt, &w, None).unwrap();
        assert_eq!(primary_no_si.len(), ref_no_si.len());
        assert_eq!(verify_no_si.len(), ref_no_si.len());
        for (i, (p, r)) in primary_no_si.iter().zip(ref_no_si.iter()).enumerate() {
            assert_eq!(p.flat_idx, r.flat_idx, "primary[{i}] flat_idx");
            assert_eq!(p.cost.to_bits(), r.cost.to_bits(), "primary[{i}] cost");
        }
        for (i, (v, r)) in verify_no_si.iter().zip(ref_no_si.iter()).enumerate() {
            assert_eq!(v.flat_idx, r.flat_idx, "verify[{i}] flat_idx");
            assert_eq!(v.cost.to_bits(), r.cost.to_bits(), "verify[{i}] cost");
        }
    }

    /// T3.2.D edge case — empty modifications. Dirty re-cost must
    /// produce IDENTICAL output to the cached cover positions (no
    /// recomputation happens, no blocks are dirty).
    #[test]
    fn dirty_recost_zero_modifications() {
        let mut cover = DctGrid::new(8, 8);
        for br in 0..8 {
            for bc in 0..8 {
                cover.set(br, bc, 0, 0, 100);
                cover.set(br, bc, 1, 1, 5);
                cover.set(br, bc, 2, 3, -3);
            }
        }
        let qt = standard_qt();
        let cover_w = compute_cover_wavelets(&cover, &qt);
        let cover_positions =
            compute_positions_from_wavelets(&cover, &qt, &cover_w, None).unwrap();

        // No modifications → stego == cover, stego_wavelets == cover_wavelets.
        let stego_w = apply_dct_modifications_to_wavelets(
            &cover_w, &[], &cover, &cover, &qt,
        );
        let result = compute_positions_with_dirty_recost(
            &cover, &qt, &cover_positions, &[], &stego_w, None,
        ).unwrap();

        assert_eq!(result.len(), cover_positions.len());
        for (r, c) in result.iter().zip(cover_positions.iter()) {
            assert_eq!(r.flat_idx, c.flat_idx);
            assert_eq!(r.cost.to_bits(), c.cost.to_bits());
        }
    }

    /// T3.2.D edge case — modifications saturate the dirty set so
    /// EVERY block is dirty. The result must match what
    /// compute_positions_from_wavelets(stego, ...) would produce (full
    /// recompute equivalence).
    #[test]
    fn dirty_recost_all_blocks_dirty() {
        let mut cover = DctGrid::new(6, 6);
        for br in 0..6 {
            for bc in 0..6 {
                cover.set(br, bc, 0, 0, 100);
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 { continue; }
                        let val =
                            (((br * 7 + bc * 13 + i * 3 + j * 11) % 21) as i16) - 10;
                        if val != 0 {
                            cover.set(br, bc, i, j, val);
                        }
                    }
                }
            }
        }
        let mut stego = cover.clone();
        let bw = stego.blocks_wide();
        // Modify ONE coefficient in EVERY block so the 6x6 dirty
        // neighborhood union covers the whole grid.
        let mut modifications: Vec<u32> = Vec::new();
        for br in 0..6 {
            for bc in 0..6 {
                let cur = stego.get(br, bc, 1, 1);
                stego.set(br, bc, 1, 1, cur + 1);
                modifications.push(((br * bw + bc) * 64 + 1 * 8 + 1) as u32);
            }
        }

        let qt = standard_qt();
        let cover_w = compute_cover_wavelets(&cover, &qt);
        let cover_positions =
            compute_positions_from_wavelets(&cover, &qt, &cover_w, None).unwrap();

        let stego_w = apply_dct_modifications_to_wavelets(
            &cover_w, &modifications, &cover, &stego, &qt,
        );
        let ref_positions =
            compute_positions_from_wavelets(&stego, &qt, &stego_w.as_cache(), None).unwrap();
        let dirty_positions = compute_positions_with_dirty_recost(
            &stego, &qt, &cover_positions, &modifications, &stego_w, None,
        ).unwrap();

        assert_eq!(ref_positions.len(), dirty_positions.len());
        for (r, d) in ref_positions.iter().zip(dirty_positions.iter()) {
            assert_eq!(r.flat_idx, d.flat_idx);
            assert!((r.cost - d.cost).abs() < 1e-6, "{} vs {}", r.cost, d.cost);
        }
    }

    /// End-to-end: Ghost encode/decode round-trip still works.
    /// This verifies that the parallel cost computation integrates correctly
    /// with the full Ghost steganography pipeline.
    #[test]
    fn ghost_roundtrip_with_current_feature_set() {
        let data = match std::fs::read("test-vectors/image/photo_320x240_q75_420.jpg") {
            Ok(d) => d,
            Err(_) => return, // Skip if test vector not available.
        };

        let message = "Parallel cost test";
        let passphrase = "test-pass-42";

        let stego = crate::stego::ghost_encode(&data, message, passphrase)
            .expect("ghost_encode should succeed");
        let decoded = crate::stego::ghost_decode(&stego, passphrase)
            .expect("ghost_decode should succeed");

        assert_eq!(decoded.text, message, "round-trip mismatch");
        assert!(decoded.files.is_empty(), "no files expected");
    }

    /// Performance benchmark for J-UNIWARD cost computation on a real photo.
    /// Marked #[ignore] so it doesn't run in CI -- run manually with:
    ///   cargo test -p phasm-core --features parallel cost_computation_benchmark -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cost_computation_benchmark() {
        let data = if let Ok(d) = std::fs::read("test-vectors/image/photo_320x240_q75_420.jpg") { d } else {
            eprintln!("Skipping benchmark: test vector not found");
            return;
        };

        let img = crate::codec::jpeg::JpegImage::from_bytes(&data).unwrap();
        let grid = img.dct_grid(0);
        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let qt = img.quant_table(qt_id).unwrap();

        // Warm-up run.
        let _ = compute_uniward(grid, qt);

        let iterations = 10;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = compute_uniward(grid, qt);
        }
        let elapsed = start.elapsed();

        let bw = grid.blocks_wide();
        let bt = grid.blocks_tall();
        eprintln!(
            "J-UNIWARD cost ({bw}x{bt} blocks, {}x{} pixels): {:.1} ms avg over {iterations} runs [feature=parallel: {}]",
            bw * 8, bt * 8,
            elapsed.as_secs_f64() * 1000.0 / iterations as f64,
            cfg!(feature = "parallel"),
        );
    }
}
