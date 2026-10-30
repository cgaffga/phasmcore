// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! **Legacy v1 J-UNIWARD cost function — decode-side compatibility only.**
//!
//! This is, verbatim, the image Ghost cost function that shipped up to
//! phasm-core 0.4.x (app releases up to 1.8). It is kept for exactly one
//! caller: the Ghost **shadow** decoder's fallback
//! (`ghost::pipeline::ghost_shadow_decode_from_image_with_positions`),
//! which recomputes the cost pool of an image made by an older release
//! when the current (v2) costs find no shadow. The primary Ghost message
//! never depends on cost values (only on the set of non-zero AC
//! coefficients, identical in v1 and v2), so nothing else needs it.
//!
//! **Never use it to encode.** It has two flaws that v2
//! ([`super::uniward`]) fixes, both documented in
//! `docs/design/image/juniward-cost.md`:
//!
//! 1. *Border truncation* — [`cost_kernel_v1`] skips the window samples
//!    outside the image instead of reading reflection-padded wavelets, so
//!    blocks in the outer top/left ring are priced too cheaply and attract
//!    several times their share of the flips.
//! 2. *Window misalignment* — the |Δ| table
//!    (`AbsDeltaTable::precompute_legacy_v1`, source offset 14) puts window
//!    cell `c` at block-relative wavelet `c − 7`, but the kernel reads the
//!    cover wavelets at `c − 15`: 8 px up and 8 px left of the impact. (The
//!    table also drops the impact's first row/column, at −8.)
//!
//! Bit-exactness with the shipped release is the whole point of this
//! module: [`cost_kernel_test_hash_hex_legacy_v1`] is pinned to the hash
//! the 0.4.x kernel produced (`tests/cost_kernel_cross_platform.rs`), and
//! `tests/ghost_cost_v1_compat.rs` decodes the shadows of images encoded by
//! the 0.4.x encoder. Do not "clean up" anything here.

use crate::codec::jpeg::dct::{DctGrid, QuantTable};
use crate::codec::jpeg::pixels_aan::aan_idct_block as idct_block;
use crate::stego::cost::CostMap;
use crate::stego::error::StegoError;
use crate::stego::permute::CoeffPos;
use crate::stego::progress;
use crate::stego::side_info::SideInfo;

use super::uniward::{
    lpdf, AbsDeltaTable, CostMapPtr, FILT_LEN, HPDF, IMPACT_SIZE, MIN_SI_COST_F32, SIGMA,
    STRIP_BLOCK_ROWS, UNIWARD_PROGRESS_STEPS,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Decompress the entire DctGrid to pixel values (f32).
pub(super) fn decompress_to_pixels(grid: &DctGrid, qt: &QuantTable, bw: usize, bt: usize) -> Vec<f32> {
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

/// Wavelet subbands for three directions. Each subband has the same
/// dimensions as the (padded) image — undecimated wavelet transform.
/// Uses f32 to halve memory usage vs f64.
pub(super) struct ThreeSubbands {
    /// LH subband (horizontal high-pass): detects vertical edges.
    pub(super) lh: Vec<f32>,
    /// HL subband (vertical high-pass): detects horizontal edges.
    pub(super) hl: Vec<f32>,
    /// HH subband (diagonal high-pass): detects diagonal edges.
    pub(super) hh: Vec<f32>,
    /// Width of each subband (same as image width).
    pub(super) width: usize,
    /// Absolute y-coordinate of the first row in the arrays.
    /// 0 for full-image computation, >0 for strip-based computation.
    pub(super) y_offset: usize,
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
pub(super) fn compute_three_subbands(
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
pub(super) fn mirror_index(idx: isize, size: usize) -> usize {
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

/// T3.3.A — precomputed-table fast path of compute_coefficient_cost
/// (legacy sum order: per-cell `lh + hl + hh` added directly into a
/// single accumulator). Kept for the Phase A bit-exactness gate vs
/// the legacy basis-recompute path. **Not used in production** — the
/// SIMD-friendly variant below is the one wired into call sites.
#[inline]
#[allow(dead_code)]
pub(super) fn compute_coefficient_cost_precomputed_legacy_order(
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
pub(super) fn cost_kernel_v1(
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
#[allow(dead_code)]
pub(super) fn compute_coefficient_cost(
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
pub fn cost_kernel_test_deterministic_bytes_legacy_v1() -> Vec<u8> {
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
    let (img_w, img_h) = (cover.blocks_wide() * 8, cover.blocks_tall() * 8);
    let pixels = decompress_to_pixels(&cover, &qt, cover.blocks_wide(), cover.blocks_tall());
    let wavelets = compute_three_subbands(&pixels, img_w, img_h, &lpdf());
    let table = AbsDeltaTable::precompute_legacy_v1(&qt);
    let pad = FILT_LEN - 1;

    let mut bytes = Vec::with_capacity(8 * 8 * 63 * 8);
    for br in 0..8 {
        for bc in 0..8 {
            for fi in 0..8 {
                for fj in 0..8 {
                    if fi == 0 && fj == 0 {
                        continue;
                    }
                    let (abs_lh, abs_hl, abs_hh) = table.slice_at(fi, fj);
                    let cost = cost_kernel_v1(
                        abs_lh, abs_hl, abs_hh,
                        &wavelets,
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
pub fn cost_kernel_test_hash_hex_legacy_v1() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(cost_kernel_test_deterministic_bytes_legacy_v1());
    let digest = h.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
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
pub fn compute_positions_streaming_legacy_v1(
    grid: &DctGrid,
    qt: &QuantTable,
    si: Option<(&SideInfo, &DctGrid)>,
) -> Result<Vec<CoeffPos>, StegoError> {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    let img_w = bw * 8;
    let img_h = bt * 8;
    let lpdf = lpdf();
    let abs_delta = AbsDeltaTable::precompute_legacy_v1(qt);
    let pad = FILT_LEN - 1; // 15

    // Positions are the non-zero AC coefficients (the rest are wet): reserve
    // exactly that (a fixed 32 per block over-reserved ~2x on real photos,
    // which is the decoder's peak on large covers).
    let est_positions = grid.coeffs().chunks_exact(64)
        .map(|b| b[1..].iter().filter(|&&c| c != 0).count())
        .sum::<usize>();
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
                    let cost = cost_kernel_v1(
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
    use crate::stego::cost::uniward::precompute_basis_functions;

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

    /// The 0.4.x whole-image decomposition (`compute_cover_wavelets` as it
    /// was): in-image samples only, `mirror_index` boundary.
    struct V1Wavelets {
        inner: ThreeSubbands,
        img_w: usize,
        img_h: usize,
    }

    fn v1_wavelets(grid: &DctGrid, qt: &QuantTable) -> V1Wavelets {
        let (bw, bt) = (grid.blocks_wide(), grid.blocks_tall());
        let pixels = decompress_to_pixels(grid, qt, bw, bt);
        V1Wavelets { inner: compute_three_subbands(&pixels, bw * 8, bt * 8, &lpdf()), img_w: bw * 8, img_h: bt * 8 }
    }

    /// The legacy streaming path (what the 0.4.x decoder ran) equals the
    /// 0.4.x whole-image kernel, position for position.
    #[test]
    fn legacy_streaming_matches_legacy_whole_image_kernel() {
        let mut grid = DctGrid::new(5, 57);
        for br in 0..57 {
            for bc in 0..5 {
                grid.set(br, bc, 0, 0, 100);
                for k in 1..64 {
                    let v = (((br * 7 + bc * 13 + k * 5) % 21) as i16) - 10;
                    if v != 0 {
                        grid.set(br, bc, k / 8, k % 8, v);
                    }
                }
            }
        }
        let qt = standard_qt();
        let streaming = compute_positions_streaming_legacy_v1(&grid, &qt, None).unwrap();
        let w = v1_wavelets(&grid, &qt);
        let table = AbsDeltaTable::precompute_legacy_v1(&qt);
        let mut i = 0;
        for br in 0..57 {
            for bc in 0..5 {
                for k in 1..64 {
                    if grid.get(br, bc, k / 8, k % 8) == 0 {
                        continue;
                    }
                    let (a, b, c) = table.slice_at(k / 8, k % 8);
                    let cost = cost_kernel_v1(a, b, c, &w.inner, br, bc, w.img_w, w.img_h, FILT_LEN - 1) as f32;
                    assert_eq!(streaming[i].flat_idx as usize, (br * 5 + bc) * 64 + k);
                    assert_eq!(streaming[i].cost.to_bits(), cost.to_bits());
                    i += 1;
                }
            }
        }
        assert_eq!(i, streaming.len());
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
        let w = v1_wavelets(&cover, &qt);
        let basis = precompute_basis_functions(&qt);
        let lpdf_t = lpdf();
        let pad = FILT_LEN - 1;
        let img_w = w.img_w;
        let img_h = w.img_h;
        let table = AbsDeltaTable::precompute_legacy_v1(&qt);

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
        let w = v1_wavelets(&cover, &qt);
        let pad = FILT_LEN - 1;
        let img_w = w.img_w;
        let img_h = w.img_h;
        let table = AbsDeltaTable::precompute_legacy_v1(&qt);

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
                    let simd = cost_kernel_v1(
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

    /// Scalar-only copy of `compute_coefficient_cost_precomputed`: the
    /// pair loop + 1-element tail with no SIMD block in front, i.e. what
    /// a target without NEON / SSE2 / SIMD128 runs. The production
    /// kernel always takes its SIMD block where one is compiled in, so
    /// its own scalar pair loop never runs there and cannot serve as the
    /// reference.
    fn cost_precomputed_scalar_reference(
        abs_lh: &[f64],
        abs_hl: &[f64],
        abs_hh: &[f64],
        cw: &ThreeSubbands,
        br: usize,
        bc: usize,
        img_w: usize,
        img_h: usize,
        pad: usize,
    ) -> f64 {
        let mut cost = 0.0;
        let bc8 = (bc * 8) as isize;
        let abs_x_start = bc8 - pad as isize;
        let out_c_lo = if abs_x_start < 0 {
            ((-abs_x_start) as usize).min(IMPACT_SIZE)
        } else {
            0
        };
        let max_via_image = img_w as isize - abs_x_start;
        let out_c_hi = if max_via_image <= 0 { 0 } else { (max_via_image as usize).min(IMPACT_SIZE) };
        if out_c_lo >= out_c_hi {
            return 0.0;
        }
        let w = |v: f32| (v as f64).abs() + SIGMA;
        for out_r in 0..IMPACT_SIZE {
            let abs_y = (br * 8) as isize + out_r as isize - pad as isize;
            if abs_y < 0 || abs_y as usize >= img_h {
                continue;
            }
            let row_idx_base = (abs_y as usize - cw.y_offset) * cw.width;
            let row_base = out_r * IMPACT_SIZE;
            let mut out_c = out_c_lo;
            while out_c + 1 < out_c_hi {
                let i0 = row_idx_base + (bc8 + out_c as isize - pad as isize) as usize;
                let (i1, c0, c1) = (i0 + 1, row_base + out_c, row_base + out_c + 1);
                let lane0 = (abs_lh[c0] / w(cw.lh[i0]) + abs_hl[c0] / w(cw.hl[i0]))
                    + abs_hh[c0] / w(cw.hh[i0]);
                let lane1 = (abs_lh[c1] / w(cw.lh[i1]) + abs_hl[c1] / w(cw.hl[i1]))
                    + abs_hh[c1] / w(cw.hh[i1]);
                cost += lane0 + lane1;
                out_c += 2;
            }
            if out_c < out_c_hi {
                let i = row_idx_base + (bc8 + out_c as isize - pad as isize) as usize;
                let c = row_base + out_c;
                cost += (abs_lh[c] / w(cw.lh[i]) + abs_hl[c] / w(cw.hl[i]))
                    + abs_hh[c] / w(cw.hh[i]);
            }
        }
        cost
    }

    /// The NEON / SSE2 / SIMD128 blocks of `compute_coefficient_cost_precomputed`
    /// must be bit-identical (f64 `to_bits`) to the scalar sum order, which is
    /// what a SIMD-less target runs. This test covers production data: the real
    /// |delta| table and the wavelets of a 64×64 cover, every block and
    /// coefficient. The pinned-hash test checks the same geometry but against
    /// a hash, not the scalar order. See
    /// `cost_precompute_simd_matches_scalar_reference_random` for the edge
    /// cases.
    #[test]
    fn cost_precompute_simd_matches_scalar_reference_fixture() {
        let pad = FILT_LEN - 1;
        let mut cover = DctGrid::new(8, 8);
        for br in 0..8 {
            for bc in 0..8 {
                cover.set(br, bc, 0, 0, 120);
                for i in 0..8 {
                    for j in 0..8 {
                        if i == 0 && j == 0 { continue; }
                        let val = (((br * 13 + bc * 5 + i * 7 + j * 3) % 23) as i16) - 11;
                        if val != 0 {
                            cover.set(br, bc, i, j, val);
                        }
                    }
                }
            }
        }
        let qt = standard_qt();
        let wv = v1_wavelets(&cover, &qt);
        let table = AbsDeltaTable::precompute_legacy_v1(&qt);
        let coefs: Vec<(usize, usize)> = if cfg!(miri) {
            vec![(0, 1), (3, 5), (7, 7)]
        } else {
            (0..64).filter(|&k| k != 0).map(|k| (k / 8, k % 8)).collect()
        };
        let blocks: Vec<(usize, usize)> = if cfg!(miri) {
            vec![(0, 0), (0, 7), (7, 0), (3, 4), (1, 1)]
        } else {
            (0..64).map(|k| (k / 8, k % 8)).collect()
        };
        for &(br, bc) in &blocks {
            for &(fi, fj) in &coefs {
                let (lh, hl, hh) = table.slice_at(fi, fj);
                let got = cost_kernel_v1(
                    lh, hl, hh, &wv.inner, br, bc, wv.img_w, wv.img_h, pad,
                );
                let want = cost_precomputed_scalar_reference(
                    lh, hl, hh, &wv.inner, br, bc, wv.img_w, wv.img_h, pad,
                );
                assert_eq!(got.to_bits(), want.to_bits(), "fixture br={br} bc={bc} fi={fi} fj={fj}: {got} vs {want}");
            }
        }
    }

    /// Same bit-exactness as `cost_precompute_simd_matches_scalar_reference_fixture`,
    /// on random geometry: odd image widths, a subband stride wider than the
    /// image, strip `y_offset`, and blocks clipped on every side or entirely
    /// outside the image. That gives every `out_c` span length 0..=23, so the
    /// pair loop, the 1-element tail, and the all-tail rows are all exercised.
    /// Wavelets include ±0, ±inf, f32::MAX, subnormals and negative values.
    /// The |delta| tables hold zeros, subnormals and 1e300, which overflows
    /// the sum to inf. The kernel's inputs are built directly, with no IDCT,
    /// so this test also runs under Miri on aarch64. Miri cannot run the NEON
    /// IDCT the fixture test needs.
    #[test]
    fn cost_precompute_simd_matches_scalar_reference_random() {
        let pad = FILT_LEN - 1;
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
                6 => f32::from_bits(1), // smallest subnormal
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
        let trials = if cfg!(miri) { 24 } else { 3000 };
        for t in 0..trials {
            let img_w = 1 + (next() % 40) as usize;
            let width = img_w + (next() % 4) as usize;
            // Every 4th trial is a strip: rows start at y_offset > 0 and the
            // block row is chosen so all of its 23 impact rows are present.
            let (y_offset, rows, img_h, br) = if t % 4 == 3 {
                let y_offset = 1 + (next() % 9) as usize;
                let br = (y_offset + pad).div_ceil(8) + (next() % 2) as usize;
                let rows = br * 8 + 8 - y_offset + (next() % 3) as usize;
                (y_offset, rows, y_offset + rows, br)
            } else {
                let img_h = 1 + (next() % 40) as usize;
                (0, img_h, img_h, (next() % (img_h as u64 / 8 + 3)) as usize)
            };
            let bc = (next() % (img_w as u64 / 8 + 4)) as usize;
            let n = width * rows;
            let mut band = || -> Vec<f32> { (0..n).map(|_| wavelet_pick(next(), next())).collect() };
            let cw = ThreeSubbands { lh: band(), hl: band(), hh: band(), width, y_offset };
            let mut tab = || -> Vec<f64> { (0..cells).map(|_| delta_pick(next(), next())).collect() };
            let (lh, hl, hh) = (tab(), tab(), tab());
            let got = cost_kernel_v1(&lh, &hl, &hh, &cw, br, bc, img_w, img_h, pad);
            let want = cost_precomputed_scalar_reference(&lh, &hl, &hh, &cw, br, bc, img_w, img_h, pad);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "trial {t}: img {img_w}x{img_h} width={width} y_offset={y_offset} br={br} bc={bc}: {got} vs {want}"
            );
        }
    }

}
