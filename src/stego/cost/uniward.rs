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

use crate::jpeg::dct::{DctGrid, QuantTable};
use crate::jpeg::pixels::idct_block;
use crate::stego::error::StegoError;
use crate::stego::progress;
use super::CostMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Number of progress steps reported by [`compute_uniward_with_progress`].
///
/// Used by the progress total formula in encode/decode so that the combined
/// step counts are correct.
///
/// Steps:
/// 1. Pixel decompression complete
/// 2. Wavelet subbands complete
/// 3. Per-block cost computation complete
pub const UNIWARD_PROGRESS_STEPS: u32 = 3;

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

    // Step 3: Precompute the 64 DCT basis functions (the IDCT of a unit
    // impulse at each frequency position, already scaled by q-step).
    // basis[fi][fj] is an 8x8 block of pixel-domain values for a +1
    // change to quantized coefficient (fi, fj).
    let basis = precompute_basis_functions(qt);

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

                // The basis function for this frequency position gives
                // the pixel-domain change when this coefficient changes by +1.
                let basis_block = &basis[fi][fj];

                // Compute the cost as the sum over three subbands of
                // the weighted wavelet-domain distortion.
                let cost = compute_coefficient_cost(
                    basis_block,
                    &cover_wavelets,
                    br, bc,
                    img_w, img_h,
                    pad,
                    &lpdf,
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
    (0..n_blocks).into_par_iter().for_each(|bi| compute_block(bi));

    #[cfg(not(feature = "parallel"))]
    (0..n_blocks).for_each(|bi| compute_block(bi));

    map
}

/// Compute J-UNIWARD costs with progress tracking.
///
/// Identical to [`compute_uniward`] but reports [`UNIWARD_PROGRESS_STEPS`]
/// progress steps and checks for cancellation between phases. Used by
/// both Ghost encode and decode paths; capacity uses the plain version.
///
/// # Progress steps
/// 1. Pixel decompression complete
/// 2. Wavelet subbands complete
/// 3. Per-block cost computation complete
pub fn compute_uniward_with_progress(grid: &DctGrid, qt: &QuantTable) -> Result<CostMap, StegoError> {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let mut map = CostMap::new(bw, bt);

    let img_w = bw * 8;
    let img_h = bt * 8;

    // Phase 1: Decompress to pixels.
    let cover_pixels = decompress_to_pixels(grid, qt, bw, bt);
    progress::advance();
    progress::check_cancelled()?;

    // Phase 2: Wavelet subbands.
    let lpdf = lpdf();
    let cover_wavelets = compute_three_subbands(&cover_pixels, img_w, img_h, &lpdf);
    drop(cover_pixels);
    progress::advance();
    progress::check_cancelled()?;

    // Phase 3: Basis functions + per-block cost computation.
    let basis = precompute_basis_functions(qt);
    let pad = FILT_LEN - 1;
    let n_blocks = bt * bw;

    let total_len = n_blocks * 64;
    let costs_ptr = CostMapPtr { ptr: map.costs_ptr(), total_len };

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

                let basis_block = &basis[fi][fj];
                let cost = compute_coefficient_cost(
                    basis_block,
                    &cover_wavelets,
                    br, bc,
                    img_w, img_h,
                    pad,
                    &lpdf,
                );

                if cost > 0.0 && cost.is_finite() {
                    let idx = (br * bw + bc) * 64 + fi * 8 + fj;
                    unsafe { costs_ptr.write(idx, cost as f32); }
                }
            }
        }
    };

    #[cfg(feature = "parallel")]
    (0..n_blocks).into_par_iter().for_each(|bi| compute_block(bi));

    #[cfg(not(feature = "parallel"))]
    (0..n_blocks).for_each(|bi| compute_block(bi));

    progress::advance();

    Ok(map)
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
            let unity_qt = [1u16; 64];
            let pixels = idct_block(&impulse, &unity_qt);

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
    // LH subband: low-pass rows → high-pass cols
    let row_low = filter_rows(pixels, width, height, lpdf);
    let lh = filter_cols(&row_low, width, height, &HPDF);
    drop(row_low); // Free before allocating row_high

    // HL and HH subbands: high-pass rows → low/high-pass cols
    let row_high = filter_rows(pixels, width, height, &HPDF);
    let hl = filter_cols(&row_high, width, height, lpdf);
    let hh = filter_cols(&row_high, width, height, &HPDF);
    // row_high dropped at return

    ThreeSubbands { lh, hl, hh, width, y_offset: 0 }
}

/// Apply a 1D filter along each row of the image (horizontal filtering).
/// Uses symmetric (mirror) boundary extension.
/// Output has the same dimensions as input (undecimated).
/// Accumulates in f64 for filter precision, stores result as f32.
fn filter_rows(
    pixels: &[f32],
    width: usize,
    height: usize,
    filter: &[f64; 16],
) -> Vec<f32> {
    let flen = FILT_LEN;
    let half = (flen - 1) / 2; // = 7 for 16-tap filter (center at index 7)
    let mut output = vec![0.0f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f64;
            for k in 0..flen {
                // Sample position with filter centered at x.
                let sx = (x as isize) + (k as isize) - (half as isize);
                let sx = mirror_index(sx, width);
                sum += pixels[y * width + sx] as f64 * filter[k];
            }
            output[y * width + x] = sum as f32;
        }
    }

    output
}

/// Apply a 1D filter along each column of the image (vertical filtering).
/// Uses symmetric (mirror) boundary extension.
/// Output has the same dimensions as input (undecimated).
/// Accumulates in f64 for filter precision, stores result as f32.
fn filter_cols(
    pixels: &[f32],
    width: usize,
    height: usize,
    filter: &[f64; 16],
) -> Vec<f32> {
    let flen = FILT_LEN;
    let half = (flen - 1) / 2; // = 7
    let mut output = vec![0.0f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f64;
            for k in 0..flen {
                let sy = (y as isize) + (k as isize) - (half as isize);
                let sy = mirror_index(sy, height);
                sum += pixels[sy * width + x] as f64 * filter[k];
            }
            output[y * width + x] = sum as f32;
        }
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
                if src_col >= 0 && src_col < 8 {
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
                if src_row >= 0 && src_row < 8 {
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

/// Minimum cost for SI-modulated coefficients (same as side_info::MIN_SI_COST).
const MIN_SI_COST_F32: f32 = 1e-6;

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
/// collection (no separate `modulate_costs_si` pass needed).
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
    let basis = precompute_basis_functions(qt);
    let pad = FILT_LEN - 1; // 15

    // Estimate: ~50% of AC positions are non-zero (typical JPEG).
    let est_positions = bt * bw * 32;
    let mut positions: Vec<CoeffPos> = Vec::with_capacity(est_positions);

    // Distribute UNIWARD_PROGRESS_STEPS (3) across strips for smooth progress.
    let num_strips = (bt + STRIP_BLOCK_ROWS - 1) / STRIP_BLOCK_ROWS;
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
        let pix_br_end = ((pix_y_end + 7) / 8).min(bt);

        // Step 1: Decompress pixel strip.
        let pix_strip_h = (pix_br_end - pix_br_start) * 8;
        let pix_strip_y0 = pix_br_start * 8; // actual start (block-aligned)
        let pixels = decompress_strip_pixels(grid, qt, bw, pix_br_start, pix_br_end);

        // Step 2: Compute wavelet subbands for the strip.
        let strip_wavelets = compute_strip_subbands(
            &pixels, img_w, pix_strip_h, pix_strip_y0,
            wav_y_start, wav_y_end, img_h, &lpdf,
        );
        drop(pixels); // Free pixel strip

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

                    let basis_block = &basis[fi][fj];
                    let cost = compute_coefficient_cost(
                        basis_block, &strip_wavelets,
                        br, bc, img_w, img_h, pad, &lpdf,
                    );

                    if cost > 0.0 && cost.is_finite() {
                        let idx = br_local * bw * 64 + bc * 64 + fi * 8 + fj;
                        unsafe { costs_ptr.write(idx, cost as f32); }
                    }
                }
            }
        };

        #[cfg(feature = "parallel")]
        (0..n_strip_blocks).into_par_iter().for_each(|bi| compute_block(bi));

        #[cfg(not(feature = "parallel"))]
        (0..n_strip_blocks).for_each(|bi| compute_block(bi));

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
                                let factor = (1.0 - 2.0 * error.abs()) as f32;
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

        // Advance progress proportionally across strips.
        strip_idx += 1;
        let target_steps = (strip_idx as u32 * UNIWARD_PROGRESS_STEPS) / num_strips as u32;
        while steps_sent < target_steps {
            progress::advance();
            steps_sent += 1;
        }
        if strip_idx % 2 == 0 {
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
    // Row filtering is independent per row — works on the strip directly.
    let row_low = filter_rows(pixels, width, pix_h, lpdf);
    let lh = filter_cols_strip(&row_low, width, pix_h, pix_y0, wav_y_start, wav_y_end, img_h, &HPDF);
    drop(row_low);

    let row_high = filter_rows(pixels, width, pix_h, &HPDF);
    let hl = filter_cols_strip(&row_high, width, pix_h, pix_y0, wav_y_start, wav_y_end, img_h, lpdf);
    let hh = filter_cols_strip(&row_high, width, pix_h, pix_y0, wav_y_start, wav_y_end, img_h, &HPDF);

    ThreeSubbands {
        lh, hl, hh, width,
        y_offset: wav_y_start,
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
        let data = match std::fs::read("test-vectors/photo_320x240_q75_420.jpg") {
            Ok(d) => d,
            Err(_) => return, // Skip if test vector not available.
        };
        let img = crate::jpeg::JpegImage::from_bytes(&data).unwrap();
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

    /// End-to-end: Ghost encode/decode round-trip still works.
    /// This verifies that the parallel cost computation integrates correctly
    /// with the full Ghost steganography pipeline.
    #[test]
    fn ghost_roundtrip_with_current_feature_set() {
        let data = match std::fs::read("test-vectors/photo_320x240_q75_420.jpg") {
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
        let data = match std::fs::read("test-vectors/photo_320x240_q75_420.jpg") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping benchmark: test vector not found");
                return;
            }
        };

        let img = crate::jpeg::JpegImage::from_bytes(&data).unwrap();
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
