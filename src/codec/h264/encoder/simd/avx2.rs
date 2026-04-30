// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AVX2 (x86_64) SIMD kernels for the H.264 encoder hot paths.
//!
//! Mirrors the aarch64 NEON kernels in `super::neon` exactly — same
//! algorithm, same input/output semantics, same bound proofs. Output
//! is bit-exact to NEON / scalar per the determinism contract
//! documented in the parent module.
//!
//! All kernels are 128-bit XMM-width (matching the 128-bit NEON port
//! one-to-one). They're called via `is_x86_feature_detected!("avx2")`
//! gating in the dispatcher, so AVX2 instruction set is guaranteed
//! at function entry. AVX2 is Haswell-and-later (2013+).
//!
//! All kernels are `unsafe fn`. Safety contracts mirror the NEON
//! versions:
//! - Caller guarantees the read ranges fit within the source / pred
//!   slices.
//! - All loads are unaligned (`_mm_loadu_si128` / `_mm_loadl_epi64`).

use core::arch::x86_64::*;

// ============================================================================
// SAD — Sum of Absolute Differences (via PSADBW)
// ============================================================================

/// SAD over a 16-wide × `h`-tall block.
///
/// AVX2 has the perfect instruction for this: `PSADBW` (SSE2-onwards
/// `_mm_sad_epu8`) takes two 16-byte vectors and returns two u64
/// lanes, each containing the sum of `|a - b|` over the corresponding
/// 8-byte half. Per-row work: one load × 2, one PSADBW, one add. Very
/// efficient.
///
/// # Safety
/// Same as NEON `sad_w16`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn sad_w16(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> u32 {
    let src_ptr = source.as_ptr();
    let prd_ptr = pred.as_ptr();
    let mut acc = unsafe { _mm_setzero_si128() };
    let mut y = 0;
    while y < h {
        let s = unsafe {
            _mm_loadu_si128(src_ptr.add(y * source_stride) as *const __m128i)
        };
        let p = unsafe {
            _mm_loadu_si128(prd_ptr.add(y * pred_stride) as *const __m128i)
        };
        let d = unsafe { _mm_sad_epu8(s, p) };
        acc = unsafe { _mm_add_epi64(acc, d) };
        y += 1;
    }
    let lo = unsafe { _mm_extract_epi64::<0>(acc) as u32 };
    let hi = unsafe { _mm_extract_epi64::<1>(acc) as u32 };
    lo + hi
}

/// SAD over an 8-wide × `h`-tall block. Loads 8 bytes via
/// `_mm_loadl_epi64`, the upper half of the XMM register stays
/// zero, so PSADBW returns the per-row SAD in lane 0.
///
/// # Safety
/// Same as NEON `sad_w8`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn sad_w8(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> u32 {
    let src_ptr = source.as_ptr();
    let prd_ptr = pred.as_ptr();
    let mut acc = unsafe { _mm_setzero_si128() };
    let mut y = 0;
    while y < h {
        let s = unsafe {
            _mm_loadl_epi64(src_ptr.add(y * source_stride) as *const __m128i)
        };
        let p = unsafe {
            _mm_loadl_epi64(prd_ptr.add(y * pred_stride) as *const __m128i)
        };
        let d = unsafe { _mm_sad_epu8(s, p) };
        acc = unsafe { _mm_add_epi64(acc, d) };
        y += 1;
    }
    unsafe { _mm_extract_epi64::<0>(acc) as u32 }
}

// ============================================================================
// SATD — Sum of Absolute Hadamard-Transformed Differences (4×4 tiled)
// ============================================================================

/// 4×4 Hadamard transform on i32 inputs, vectorised. Mirrors the NEON
/// `hadamard_4x4` two-stage butterfly + transpose.
///
/// # Bound proof
/// Same as NEON: |coef| ≤ 4080 after 2 stages, fits i16; we keep i32
/// for headroom and to match scalar signature.
///
/// # Safety
/// Operates on `[i32; 4]` rows from `&[[i32; 4]; 4]`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hadamard_4x4(input: &[[i32; 4]; 4]) -> [__m128i; 4] {
    let r0 = unsafe { _mm_loadu_si128(input[0].as_ptr() as *const __m128i) };
    let r1 = unsafe { _mm_loadu_si128(input[1].as_ptr() as *const __m128i) };
    let r2 = unsafe { _mm_loadu_si128(input[2].as_ptr() as *const __m128i) };
    let r3 = unsafe { _mm_loadu_si128(input[3].as_ptr() as *const __m128i) };

    // Stage 1 — column butterflies (mix rows of X).
    let p02 = _mm_add_epi32(r0, r2);
    let m02 = _mm_sub_epi32(r0, r2);
    let p13 = _mm_add_epi32(r1, r3);
    let m13 = _mm_sub_epi32(r1, r3);
    let a0 = _mm_add_epi32(p02, p13);
    let a1 = _mm_add_epi32(m02, m13);
    let a2 = _mm_sub_epi32(m02, m13);
    let a3 = _mm_sub_epi32(p02, p13);

    // Transpose 4×4 i32 lanes via unpack:
    //   t01_lo = [a0_0, a1_0, a0_1, a1_1]
    //   t01_hi = [a0_2, a1_2, a0_3, a1_3]
    //   t23_lo = [a2_0, a3_0, a2_1, a3_1]
    //   t23_hi = [a2_2, a3_2, a2_3, a3_3]
    //   b0 = unpacklo64(t01_lo, t23_lo) = [a0_0, a1_0, a2_0, a3_0]
    //   b1 = unpackhi64(t01_lo, t23_lo) = [a0_1, a1_1, a2_1, a3_1]
    //   b2 = unpacklo64(t01_hi, t23_hi) = [a0_2, a1_2, a2_2, a3_2]
    //   b3 = unpackhi64(t01_hi, t23_hi) = [a0_3, a1_3, a2_3, a3_3]
    let t01_lo = _mm_unpacklo_epi32(a0, a1);
    let t01_hi = _mm_unpackhi_epi32(a0, a1);
    let t23_lo = _mm_unpacklo_epi32(a2, a3);
    let t23_hi = _mm_unpackhi_epi32(a2, a3);
    let b0 = _mm_unpacklo_epi64(t01_lo, t23_lo);
    let b1 = _mm_unpackhi_epi64(t01_lo, t23_lo);
    let b2 = _mm_unpacklo_epi64(t01_hi, t23_hi);
    let b3 = _mm_unpackhi_epi64(t01_hi, t23_hi);

    // Stage 2 — column butterflies on transposed.
    let p02 = _mm_add_epi32(b0, b2);
    let m02 = _mm_sub_epi32(b0, b2);
    let p13 = _mm_add_epi32(b1, b3);
    let m13 = _mm_sub_epi32(b1, b3);
    let c0 = _mm_add_epi32(p02, p13);
    let c1 = _mm_add_epi32(m02, m13);
    let c2 = _mm_sub_epi32(m02, m13);
    let c3 = _mm_sub_epi32(p02, p13);

    // Transpose back.
    let u01_lo = _mm_unpacklo_epi32(c0, c1);
    let u01_hi = _mm_unpackhi_epi32(c0, c1);
    let u23_lo = _mm_unpacklo_epi32(c2, c3);
    let u23_hi = _mm_unpackhi_epi32(c2, c3);
    let d0 = _mm_unpacklo_epi64(u01_lo, u23_lo);
    let d1 = _mm_unpackhi_epi64(u01_lo, u23_lo);
    let d2 = _mm_unpacklo_epi64(u01_hi, u23_hi);
    let d3 = _mm_unpackhi_epi64(u01_hi, u23_hi);
    [d0, d1, d2, d3]
}

/// SATD over `block_w × block_h` (both multiples of 4), tiled into
/// 4×4 Hadamard sums. Mirror of NEON `satd_block_4x4_tiled`.
///
/// # Safety
/// Same as NEON `satd_block_4x4_tiled`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn satd_block_4x4_tiled(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    block_w: usize,
    block_h: usize,
) -> u32 {
    debug_assert!(block_w.is_multiple_of(4));
    debug_assert!(block_h.is_multiple_of(4));
    let tiles_y = block_h / 4;
    let tiles_x = block_w / 4;

    let mut total: u32 = 0;
    for by in 0..tiles_y {
        for bx in 0..tiles_x {
            // Scalar residual load (same as NEON path; the win is in
            // the Hadamard + abs-sum, not the byte-load).
            let mut residual = [[0i32; 4]; 4];
            for dy in 0..4 {
                let sy = by * 4 + dy;
                let sx = bx * 4;
                for dx in 0..4 {
                    let s = source[sy * source_stride + sx + dx] as i32;
                    let p = pred[sy * pred_stride + sx + dx] as i32;
                    residual[dy][dx] = s - p;
                }
            }
            let h = unsafe { hadamard_4x4(&residual) };
            // Sum |coef| via _mm_abs_epi32 (SSSE3) + horizontal add.
            let s0 = _mm_abs_epi32(h[0]);
            let s1 = _mm_abs_epi32(h[1]);
            let s2 = _mm_abs_epi32(h[2]);
            let s3 = _mm_abs_epi32(h[3]);
            let row01 = _mm_add_epi32(s0, s1);
            let row23 = _mm_add_epi32(s2, s3);
            let row_sums = _mm_add_epi32(row01, row23);
            // Horizontal sum 4 i32 lanes.
            let pair = _mm_add_epi32(
                row_sums,
                _mm_shuffle_epi32(row_sums, 0b00_01_10_11),
            );
            let single = _mm_add_epi32(
                pair,
                _mm_shuffle_epi32(pair, 0b00_00_00_01),
            );
            let tile_sum = _mm_cvtsi128_si32(single) as u32;
            total = total.saturating_add(tile_sum);
        }
    }
    total
}

// ============================================================================
// Motion compensation — luma (Phase I.2c, AVX2 ports)
// ============================================================================
//
// Mirrors the NEON port one-to-one. Each kernel uses 128-bit XMM
// width (matches NEON 128-bit). Helpers:
//
//   load_8u_to_s16(src) -> __m128i with 8 i16 lanes (zero-extend
//                          from u8, safe because samples are 0..255)
//   apply_h_tap(s0..s5)  -> b1 i16 vector
//   b_from_b1(b1)        -> 8 packed u8 = clip1y((b1 + 16) >> 5)

/// Zero-extend 8 u8 samples to 8 i16 lanes (low half of XMM),
/// upper 4 lanes are zero (we use a 0-wide pack later).
///
/// Implementation: load 64-bit (8 bytes), then `_mm_unpacklo_epi8`
/// against zero spreads each byte into the low half of an i16 lane.
/// Since input bytes are 0..255 (always positive), this matches the
/// NEON `vmovl_u8 → vreinterpretq_s16_u16` semantics.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_8u_to_s16(src: *const u8) -> __m128i {
    let v8 = unsafe { _mm_loadl_epi64(src as *const __m128i) };
    unsafe { _mm_unpacklo_epi8(v8, _mm_setzero_si128()) }
}

/// 6-tap horizontal/vertical filter in i16: out = s0 - 5*s1 + 20*s2 + 20*s3 - 5*s4 + s5.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn apply_6tap_s16(
    s0: __m128i, s1: __m128i, s2: __m128i,
    s3: __m128i, s4: __m128i, s5: __m128i,
) -> __m128i {
    let s2p3 = _mm_add_epi16(s2, s3);
    let s1p4 = _mm_add_epi16(s1, s4);
    let s0p5 = _mm_add_epi16(s0, s5);
    let term20 = _mm_mullo_epi16(s2p3, _mm_set1_epi16(20));
    let term5 = _mm_mullo_epi16(s1p4, _mm_set1_epi16(5));
    _mm_add_epi16(_mm_sub_epi16(term20, term5), s0p5)
}

/// clip1y((b1 + 16) >> 5) for 8 i16 lanes → 8 u8 (low half packed).
/// Returns 8 bytes packed into the low 64 bits of the result; caller
/// uses `_mm_storel_epi64` to write.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn b_from_b1(b1: __m128i) -> __m128i {
    let off = _mm_add_epi16(b1, _mm_set1_epi16(16));
    let shr = _mm_srai_epi16(off, 5);
    // Saturate i16 → u8 via _mm_packus_epi16 (low 8 lanes packed
    // into low 8 bytes of the result; upper 8 bytes are from the
    // second arg, we use zero so they stay zero). _mm_packus_epi16
    // saturates negatives to 0 and >255 to 255 = clip1y.
    _mm_packus_epi16(shr, _mm_setzero_si128())
}

/// Apply integer-MV (x_frac=0, y_frac=0) luma MC. Mirror of NEON
/// `mc_luma_integer_mv`.
///
/// # Safety
/// Same as NEON kernel. `y_plane[y * plane_w + 0..plane_w]` valid for
/// `y in 0..plane_h`. `out[y * out_stride + 0..block_w]` valid for
/// `y in 0..block_h`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn mc_luma_integer_mv(
    y_plane: &[u8],
    plane_w: u32,
    plane_h: u32,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    mv_x_int: i32,
    mv_y_int: i32,
    out: &mut [u8],
    out_stride: usize,
) -> bool {
    let src_x_start = block_x as i32 + mv_x_int;
    let src_y_start = block_y as i32 + mv_y_int;
    if src_x_start < 0
        || src_y_start < 0
        || src_x_start as u32 + block_w > plane_w
        || src_y_start as u32 + block_h > plane_h
    {
        return false;
    }
    let src_x = src_x_start as usize;
    let src_y = src_y_start as usize;
    let plane_w_us = plane_w as usize;
    let w = block_w as usize;

    let src_ptr = y_plane.as_ptr();
    let dst_ptr = out.as_mut_ptr();
    for dy in 0..block_h as usize {
        let src_off = (src_y + dy) * plane_w_us + src_x;
        let dst_off = dy * out_stride;
        let mut x = 0;
        while x + 16 <= w {
            let v = unsafe {
                _mm_loadu_si128(src_ptr.add(src_off + x) as *const __m128i)
            };
            unsafe {
                _mm_storeu_si128(dst_ptr.add(dst_off + x) as *mut __m128i, v);
            }
            x += 16;
        }
        while x + 8 <= w {
            let v = unsafe {
                _mm_loadl_epi64(src_ptr.add(src_off + x) as *const __m128i)
            };
            unsafe {
                _mm_storel_epi64(dst_ptr.add(dst_off + x) as *mut __m128i, v);
            }
            x += 8;
        }
    }
    true
}

/// Apply pure-horizontal half-pel luma MC. Mirror of NEON
/// `mc_luma_h_only`. `x_frac` ∈ {1, 2, 3}, `y_frac == 0`.
///
/// # Safety
/// Same as NEON kernel.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn mc_luma_h_only(
    y_plane: &[u8],
    plane_w: u32,
    plane_h: u32,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    mv_x_int: i32,
    mv_y_int: i32,
    x_frac: u8,
    out: &mut [u8],
    out_stride: usize,
) -> bool {
    let src_x_start = block_x as i32 + mv_x_int;
    let src_y_start = block_y as i32 + mv_y_int;
    if src_x_start - 2 < 0
        || src_y_start < 0
        || (src_x_start + block_w as i32 + 3) as u32 > plane_w
        || src_y_start as u32 + block_h > plane_h
    {
        return false;
    }
    let src_x = src_x_start as usize;
    let src_y = src_y_start as usize;
    let plane_w_us = plane_w as usize;
    let w = block_w as usize;

    let src_ptr = y_plane.as_ptr();
    let dst_ptr = out.as_mut_ptr();
    for dy in 0..block_h as usize {
        let row_base = (src_y + dy) * plane_w_us + src_x;
        let mut x = 0;
        while x + 8 <= w {
            let base = unsafe { src_ptr.add(row_base + x) };
            let s0 = unsafe { load_8u_to_s16(base.wrapping_sub(2)) };
            let s1 = unsafe { load_8u_to_s16(base.wrapping_sub(1)) };
            let s2 = unsafe { load_8u_to_s16(base) };
            let s3 = unsafe { load_8u_to_s16(base.add(1)) };
            let s4 = unsafe { load_8u_to_s16(base.add(2)) };
            let s5 = unsafe { load_8u_to_s16(base.add(3)) };
            let b1 = unsafe { apply_6tap_s16(s0, s1, s2, s3, s4, s5) };
            let b_packed = unsafe { b_from_b1(b1) };
            // Integer samples for x_frac=1 (G = s2's bytes) / x_frac=3
            // (H = s3's bytes). We need them as u8, not i16, so reload
            // the source row at the correct offset directly.
            let out_xmm = match x_frac {
                2 => b_packed,
                1 => {
                    let g = unsafe {
                        _mm_loadl_epi64(base as *const __m128i)
                    };
                    // (g + b + 1) >> 1 = _mm_avg_epu8
                    unsafe { _mm_avg_epu8(g, b_packed) }
                }
                3 => {
                    let h_int = unsafe {
                        _mm_loadl_epi64(base.add(1) as *const __m128i)
                    };
                    unsafe { _mm_avg_epu8(h_int, b_packed) }
                }
                _ => unreachable!(),
            };
            unsafe {
                _mm_storel_epi64(
                    dst_ptr.add(dy * out_stride + x) as *mut __m128i,
                    out_xmm,
                );
            }
            x += 8;
        }
    }
    true
}

/// Apply pure-vertical half-pel luma MC. Mirror of NEON
/// `mc_luma_v_only`. `x_frac == 0`, `y_frac` ∈ {1, 2, 3}.
///
/// # Safety
/// Same as NEON kernel.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn mc_luma_v_only(
    y_plane: &[u8],
    plane_w: u32,
    plane_h: u32,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    mv_x_int: i32,
    mv_y_int: i32,
    y_frac: u8,
    out: &mut [u8],
    out_stride: usize,
) -> bool {
    let src_x_start = block_x as i32 + mv_x_int;
    let src_y_start = block_y as i32 + mv_y_int;
    if src_x_start < 0
        || src_y_start - 2 < 0
        || src_x_start as u32 + block_w > plane_w
        || (src_y_start + block_h as i32 + 3) as u32 > plane_h
    {
        return false;
    }
    let src_x = src_x_start as usize;
    let src_y = src_y_start as usize;
    let plane_w_us = plane_w as usize;
    let w = block_w as usize;

    let src_ptr = y_plane.as_ptr();
    let dst_ptr = out.as_mut_ptr();
    for dy in 0..block_h as usize {
        let mut x = 0;
        while x + 8 <= w {
            let row_off = |delta: i32| -> *const u8 {
                let off = ((src_y as i32 + dy as i32 + delta) as usize)
                    * plane_w_us
                    + src_x + x;
                unsafe { src_ptr.add(off) }
            };
            let s0 = unsafe { load_8u_to_s16(row_off(-2)) };
            let s1 = unsafe { load_8u_to_s16(row_off(-1)) };
            let s2 = unsafe { load_8u_to_s16(row_off(0)) };
            let s3 = unsafe { load_8u_to_s16(row_off(1)) };
            let s4 = unsafe { load_8u_to_s16(row_off(2)) };
            let s5 = unsafe { load_8u_to_s16(row_off(3)) };
            let h1 = unsafe { apply_6tap_s16(s0, s1, s2, s3, s4, s5) };
            let h_packed = unsafe { b_from_b1(h1) };
            let out_xmm = match y_frac {
                2 => h_packed,
                1 => {
                    let g = unsafe {
                        _mm_loadl_epi64(row_off(0) as *const __m128i)
                    };
                    unsafe { _mm_avg_epu8(g, h_packed) }
                }
                3 => {
                    let m_int = unsafe {
                        _mm_loadl_epi64(row_off(1) as *const __m128i)
                    };
                    unsafe { _mm_avg_epu8(m_int, h_packed) }
                }
                _ => unreachable!(),
            };
            unsafe {
                _mm_storel_epi64(
                    dst_ptr.add(dy * out_stride + x) as *mut __m128i,
                    out_xmm,
                );
            }
            x += 8;
        }
    }
    true
}

/// Apply composite (x_frac > 0 AND y_frac > 0) luma MC. Mirror of
/// NEON `mc_luma_composite`. Covers the 9 remaining MC fraction
/// classes including the centre `j`.
///
/// j (centre) needs i32 widening for the second-pass filter
/// (sum-abs 52 × 13260 = 689,520 exceeds i16 range). We sign-extend
/// i16 lanes to i32 via `_mm_unpacklo/hi_epi16` against the
/// arithmetic-shifted sign bits, then narrow back via
/// `_mm_packs_epi32` (won't saturate; |j| ≤ 690k>>10 = 673 fits i16)
/// + `_mm_packus_epi16` (this one IS the spec clip1y).
///
/// # Safety
/// Same as NEON kernel.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn mc_luma_composite(
    y_plane: &[u8],
    plane_w: u32,
    plane_h: u32,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    mv_x_int: i32,
    mv_y_int: i32,
    x_frac: u8,
    y_frac: u8,
    out: &mut [u8],
    out_stride: usize,
) -> bool {
    debug_assert!(x_frac > 0 && y_frac > 0 && x_frac <= 3 && y_frac <= 3);
    let src_x_start = block_x as i32 + mv_x_int;
    let src_y_start = block_y as i32 + mv_y_int;
    if src_x_start - 2 < 0
        || src_y_start - 2 < 0
        || (src_x_start + block_w as i32 + 3 + 1) as u32 > plane_w
        || (src_y_start + block_h as i32 + 3) as u32 > plane_h
    {
        return false;
    }
    let src_x = src_x_start as usize;
    let src_y = src_y_start as usize;
    let plane_w_us = plane_w as usize;
    let w = block_w as usize;

    let src_ptr = y_plane.as_ptr();
    let dst_ptr = out.as_mut_ptr();

    for dy in 0..block_h as usize {
        let mut x = 0;
        while x + 8 <= w {
            let row_off = |delta: i32| -> usize {
                ((src_y as i32 + dy as i32 + delta) as usize) * plane_w_us
                    + src_x + x
            };
            // Horizontal 6-tap at row `off`, returning b1 (i16x8).
            let h6 = |off: usize| -> __m128i {
                let base = unsafe { src_ptr.add(off) };
                let s0 = unsafe { load_8u_to_s16(base.wrapping_sub(2)) };
                let s1 = unsafe { load_8u_to_s16(base.wrapping_sub(1)) };
                let s2 = unsafe { load_8u_to_s16(base) };
                let s3 = unsafe { load_8u_to_s16(base.add(1)) };
                let s4 = unsafe { load_8u_to_s16(base.add(2)) };
                let s5 = unsafe { load_8u_to_s16(base.add(3)) };
                unsafe { apply_6tap_s16(s0, s1, s2, s3, s4, s5) }
            };
            let b1_m2 = h6(row_off(-2));
            let b1_m1 = h6(row_off(-1));
            let b1_0 = h6(row_off(0));
            let b1_1 = h6(row_off(1));
            let b1_2 = h6(row_off(2));
            let b1_3 = h6(row_off(3));

            // Vertical 6-tap at 8 cols starting `col_extra` past src_x+x.
            let v6 = |col_extra: usize| -> __m128i {
                let load = |delta: i32| -> __m128i {
                    let off = ((src_y as i32 + dy as i32 + delta) as usize)
                        * plane_w_us
                        + src_x + x + col_extra;
                    unsafe { load_8u_to_s16(src_ptr.add(off)) }
                };
                let s0 = load(-2);
                let s1 = load(-1);
                let s2 = load(0);
                let s3 = load(1);
                let s4 = load(2);
                let s5 = load(3);
                unsafe { apply_6tap_s16(s0, s1, s2, s3, s4, s5) }
            };

            // j from 6 b1 vectors. Widen i16 → i32 (low + high halves)
            // because the sum-abs-52 weighted sum exceeds i16 range.
            let j_compute = || -> __m128i {
                // Sign-extend i16 → i32: unpacklo with arithmetic
                // shift right (>> 15) producing the sign-extension
                // word. _mm_unpacklo_epi16(v, _mm_srai_epi16(v, 15))
                // = sign-extend low 4 i16 lanes to 4 i32 lanes.
                let sign_ext_lo = |v: __m128i| -> __m128i {
                    unsafe { _mm_unpacklo_epi16(v, _mm_srai_epi16(v, 15)) }
                };
                let sign_ext_hi = |v: __m128i| -> __m128i {
                    unsafe { _mm_unpackhi_epi16(v, _mm_srai_epi16(v, 15)) }
                };
                let one_half = |a_m2: __m128i, a_m1: __m128i, a_0: __m128i,
                                a_1: __m128i, a_2: __m128i, a_3: __m128i|
                 -> __m128i {
                    let sp01 = _mm_add_epi32(a_0, a_1);
                    let sp12 = _mm_add_epi32(a_m1, a_2);
                    let sp23 = _mm_add_epi32(a_m2, a_3);
                    let term20 = _mm_mullo_epi32(sp01, _mm_set1_epi32(20));
                    let term5 = _mm_mullo_epi32(sp12, _mm_set1_epi32(5));
                    _mm_add_epi32(_mm_sub_epi32(term20, term5), sp23)
                };
                let j1_lo = one_half(
                    sign_ext_lo(b1_m2), sign_ext_lo(b1_m1), sign_ext_lo(b1_0),
                    sign_ext_lo(b1_1), sign_ext_lo(b1_2), sign_ext_lo(b1_3),
                );
                let j1_hi = one_half(
                    sign_ext_hi(b1_m2), sign_ext_hi(b1_m1), sign_ext_hi(b1_0),
                    sign_ext_hi(b1_1), sign_ext_hi(b1_2), sign_ext_hi(b1_3),
                );
                // j = clip1y((j1 + 512) >> 10)
                let off = _mm_set1_epi32(512);
                let j_lo = _mm_srai_epi32(_mm_add_epi32(j1_lo, off), 10);
                let j_hi = _mm_srai_epi32(_mm_add_epi32(j1_hi, off), 10);
                // i32 → i16 (won't saturate; |j| ≤ 673 fits i16).
                let s16 = _mm_packs_epi32(j_lo, j_hi);
                // i16 → u8 saturating = clip1y. Pack with zero; result
                // 8 bytes in low half.
                _mm_packus_epi16(s16, _mm_setzero_si128())
            };

            let result = match (x_frac, y_frac) {
                (1, 1) => {
                    // e = (b(0) + h(0) + 1) >> 1
                    let b0 = unsafe { b_from_b1(b1_0) };
                    let h0 = unsafe { b_from_b1(v6(0)) };
                    unsafe { _mm_avg_epu8(b0, h0) }
                }
                (2, 1) => {
                    // f = (b(0) + j + 1) >> 1
                    let b0 = unsafe { b_from_b1(b1_0) };
                    let j = j_compute();
                    unsafe { _mm_avg_epu8(b0, j) }
                }
                (3, 1) => {
                    // g = (b(0) + h(1) + 1) >> 1
                    let b0 = unsafe { b_from_b1(b1_0) };
                    let h1 = unsafe { b_from_b1(v6(1)) };
                    unsafe { _mm_avg_epu8(b0, h1) }
                }
                (1, 2) => {
                    // i = (h(0) + j + 1) >> 1
                    let h0 = unsafe { b_from_b1(v6(0)) };
                    let j = j_compute();
                    unsafe { _mm_avg_epu8(h0, j) }
                }
                (2, 2) => j_compute(),
                (3, 2) => {
                    // k = (j + h(1) + 1) >> 1
                    let h1 = unsafe { b_from_b1(v6(1)) };
                    let j = j_compute();
                    unsafe { _mm_avg_epu8(j, h1) }
                }
                (1, 3) => {
                    // p = (h(0) + b(1) + 1) >> 1
                    let h0 = unsafe { b_from_b1(v6(0)) };
                    let b_one = unsafe { b_from_b1(b1_1) };
                    unsafe { _mm_avg_epu8(h0, b_one) }
                }
                (2, 3) => {
                    // q = (j + b(1) + 1) >> 1
                    let j = j_compute();
                    let b_one = unsafe { b_from_b1(b1_1) };
                    unsafe { _mm_avg_epu8(j, b_one) }
                }
                (3, 3) => {
                    // r = (h(1) + b(1) + 1) >> 1
                    let h1 = unsafe { b_from_b1(v6(1)) };
                    let b_one = unsafe { b_from_b1(b1_1) };
                    unsafe { _mm_avg_epu8(h1, b_one) }
                }
                _ => unreachable!(),
            };
            unsafe {
                _mm_storel_epi64(
                    dst_ptr.add(dy * out_stride + x) as *mut __m128i,
                    result,
                );
            }
            x += 8;
        }
    }
    true
}
