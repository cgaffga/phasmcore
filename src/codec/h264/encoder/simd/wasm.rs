// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! WASM SIMD128 (wasm32) kernels for the H.264 encoder hot paths.
//!
//! Mirrors the aarch64 NEON / x86_64 AVX2 kernels in `super::neon` /
//! `super::avx2` exactly — same algorithm, same input/output
//! semantics, same bound proofs. Output is bit-exact across all four
//! ISAs (NEON / AVX2 / SIMD128 / scalar) per the determinism contract.
//!
//! All kernels are 128-bit `v128`-width (matching NEON/AVX2-128
//! one-to-one). They're enabled via `#[target_feature(enable =
//! "simd128")]` per function — caller invokes inside `unsafe { }`.
//! The build must target `wasm32` with the `simd128` target feature
//! available; if not, the entire module is `#[cfg]`-gated out and
//! callers fall through to scalar.
//!
//! ## Why WASM SIMD differs from x86_64 / aarch64 detection
//!
//! There's no runtime "host has SIMD?" check in WASM — SIMD support
//! is decided at WASM module load time. A module compiled with
//! `simd128` instructions either loads on the host (modern browsers
//! / runtimes) or fails to load. We don't ship two builds; the
//! caller's build system decides whether to enable the `simd`
//! Cargo feature for the WASM target. (The `PHASM_H264_DISABLE_SIMD`
//! env var still works as a runtime A/B knob inside WASI; in the
//! browser it always reads as unset → SIMD enabled.)

use core::arch::wasm32::*;

// ============================================================================
// SAD — Sum of Absolute Differences
// ============================================================================
//
// WASM SIMD128 has no PSADBW-equivalent. Standard pattern:
//   abs(a - b) = max(a, b) - min(a, b)
//   or equivalently: u8x16_sub_sat(a, b) | u8x16_sub_sat(b, a)
// then pairwise add u8 → u16 and accumulate.

/// SAD over a 16-wide × `h`-tall block.
///
/// # Bound proof
/// Same as NEON / AVX2: per-row 16-byte sum ≤ 16 × 255 = 4080,
/// pairwise-added u16 lanes ≤ 510 each. Across h ≤ 16 rows the
/// per-lane sum ≤ 8160, fits u16 with margin. Final pairwise add
/// to u32 then 4-lane horizontal sum gives total fit in u32.
///
/// # Safety
/// `source[y * source_stride + 0..16]` and `pred[y * pred_stride + 0..16]`
/// must be valid for `y in 0..h`.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn sad_w16(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> u32 {
    let src_ptr = source.as_ptr();
    let prd_ptr = pred.as_ptr();
    let mut acc = u16x8_splat(0);
    let mut y = 0;
    while y < h {
        let s = unsafe { v128_load(src_ptr.add(y * source_stride) as *const v128) };
        let p = unsafe { v128_load(prd_ptr.add(y * pred_stride) as *const v128) };
        let d1 = u8x16_sub_sat(s, p);
        let d2 = u8x16_sub_sat(p, s);
        let abs = v128_or(d1, d2);
        // Pairwise add u8 lanes into u16 lanes.
        let pair = u16x8_extadd_pairwise_u8x16(abs);
        acc = u16x8_add(acc, pair);
        y += 1;
    }
    // Horizontal sum: 8x u16 → 4x u32 → scalar.
    let acc32 = u32x4_extadd_pairwise_u16x8(acc);
    let pair = u32x4_add(acc32, u32x4_shuffle::<2, 3, 0, 1>(acc32, acc32));
    let single = u32x4_add(pair, u32x4_shuffle::<1, 0, 3, 2>(pair, pair));
    u32x4_extract_lane::<0>(single)
}

/// SAD over an 8-wide × `h`-tall block.
///
/// # Safety
/// Same as `sad_w16` but for 8-byte rows.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn sad_w8(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> u32 {
    let src_ptr = source.as_ptr();
    let prd_ptr = pred.as_ptr();
    let mut acc = u16x8_splat(0);
    let mut y = 0;
    while y < h {
        let s = unsafe {
            v128_load64_zero(src_ptr.add(y * source_stride) as *const u64)
        };
        let p = unsafe {
            v128_load64_zero(prd_ptr.add(y * pred_stride) as *const u64)
        };
        let d1 = u8x16_sub_sat(s, p);
        let d2 = u8x16_sub_sat(p, s);
        let abs = v128_or(d1, d2);
        let pair = u16x8_extadd_pairwise_u8x16(abs);
        acc = u16x8_add(acc, pair);
        y += 1;
    }
    // Only low 4 u16 lanes contain SAD bytes (upper 8 bytes were
    // zero-extended → zero diffs → zero pair sums). Horizontal sum
    // still works on the full vector (zeros add nothing).
    let acc32 = u32x4_extadd_pairwise_u16x8(acc);
    let pair = u32x4_add(acc32, u32x4_shuffle::<2, 3, 0, 1>(acc32, acc32));
    let single = u32x4_add(pair, u32x4_shuffle::<1, 0, 3, 2>(pair, pair));
    u32x4_extract_lane::<0>(single)
}

// ============================================================================
// SATD — Sum of Absolute Hadamard-Transformed Differences (4×4 tiled)
// ============================================================================

/// 4×4 Hadamard transform on i32 inputs. Mirror of NEON / AVX2
/// `hadamard_4x4`.
///
/// # Bound proof
/// Same: |coef| ≤ 4080 after 2 stages, fits i16; we keep i32.
///
/// # Safety
/// Operates on `[i32; 4]` rows from `&[[i32; 4]; 4]`.
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn hadamard_4x4(input: &[[i32; 4]; 4]) -> [v128; 4] {
    let r0 = unsafe { v128_load(input[0].as_ptr() as *const v128) };
    let r1 = unsafe { v128_load(input[1].as_ptr() as *const v128) };
    let r2 = unsafe { v128_load(input[2].as_ptr() as *const v128) };
    let r3 = unsafe { v128_load(input[3].as_ptr() as *const v128) };

    // Stage 1 — column butterflies.
    let p02 = i32x4_add(r0, r2);
    let m02 = i32x4_sub(r0, r2);
    let p13 = i32x4_add(r1, r3);
    let m13 = i32x4_sub(r1, r3);
    let a0 = i32x4_add(p02, p13);
    let a1 = i32x4_add(m02, m13);
    let a2 = i32x4_sub(m02, m13);
    let a3 = i32x4_sub(p02, p13);

    // Transpose 4×4 i32 lanes via shuffle:
    //   t01_lo = (a0[0], a1[0], a0[1], a1[1])
    //   t01_hi = (a0[2], a1[2], a0[3], a1[3])
    //   t23_lo = (a2[0], a3[0], a2[1], a3[1])
    //   t23_hi = (a2[2], a3[2], a2[3], a3[3])
    let t01_lo = i32x4_shuffle::<0, 4, 1, 5>(a0, a1);
    let t01_hi = i32x4_shuffle::<2, 6, 3, 7>(a0, a1);
    let t23_lo = i32x4_shuffle::<0, 4, 1, 5>(a2, a3);
    let t23_hi = i32x4_shuffle::<2, 6, 3, 7>(a2, a3);
    //   b0 = (t01_lo[0], t01_lo[1], t23_lo[0], t23_lo[1])
    //   b1 = (t01_lo[2], t01_lo[3], t23_lo[2], t23_lo[3])
    //   b2 = (t01_hi[0], t01_hi[1], t23_hi[0], t23_hi[1])
    //   b3 = (t01_hi[2], t01_hi[3], t23_hi[2], t23_hi[3])
    let b0 = i32x4_shuffle::<0, 1, 4, 5>(t01_lo, t23_lo);
    let b1 = i32x4_shuffle::<2, 3, 6, 7>(t01_lo, t23_lo);
    let b2 = i32x4_shuffle::<0, 1, 4, 5>(t01_hi, t23_hi);
    let b3 = i32x4_shuffle::<2, 3, 6, 7>(t01_hi, t23_hi);

    // Stage 2 — column butterflies on transposed.
    let p02 = i32x4_add(b0, b2);
    let m02 = i32x4_sub(b0, b2);
    let p13 = i32x4_add(b1, b3);
    let m13 = i32x4_sub(b1, b3);
    let c0 = i32x4_add(p02, p13);
    let c1 = i32x4_add(m02, m13);
    let c2 = i32x4_sub(m02, m13);
    let c3 = i32x4_sub(p02, p13);

    // Transpose back.
    let u01_lo = i32x4_shuffle::<0, 4, 1, 5>(c0, c1);
    let u01_hi = i32x4_shuffle::<2, 6, 3, 7>(c0, c1);
    let u23_lo = i32x4_shuffle::<0, 4, 1, 5>(c2, c3);
    let u23_hi = i32x4_shuffle::<2, 6, 3, 7>(c2, c3);
    let d0 = i32x4_shuffle::<0, 1, 4, 5>(u01_lo, u23_lo);
    let d1 = i32x4_shuffle::<2, 3, 6, 7>(u01_lo, u23_lo);
    let d2 = i32x4_shuffle::<0, 1, 4, 5>(u01_hi, u23_hi);
    let d3 = i32x4_shuffle::<2, 3, 6, 7>(u01_hi, u23_hi);
    [d0, d1, d2, d3]
}

/// SATD over `block_w × block_h` (both multiples of 4), tiled into
/// 4×4 Hadamard sums. Mirror of NEON / AVX2 `satd_block_4x4_tiled`.
///
/// # Safety
/// Same as NEON / AVX2 versions.
#[target_feature(enable = "simd128")]
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
            // Scalar residual load (matches NEON / AVX2 patterns).
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
            let s0 = i32x4_abs(h[0]);
            let s1 = i32x4_abs(h[1]);
            let s2 = i32x4_abs(h[2]);
            let s3 = i32x4_abs(h[3]);
            let row01 = i32x4_add(s0, s1);
            let row23 = i32x4_add(s2, s3);
            let row_sums = i32x4_add(row01, row23);
            // Horizontal sum 4 i32 lanes.
            let pair = i32x4_add(row_sums, i32x4_shuffle::<2, 3, 0, 1>(row_sums, row_sums));
            let single = i32x4_add(pair, i32x4_shuffle::<1, 0, 3, 2>(pair, pair));
            let tile_sum = i32x4_extract_lane::<0>(single) as u32;
            total = total.saturating_add(tile_sum);
        }
    }
    total
}

// ============================================================================
// Motion compensation — luma (Phase I.2d, WASM SIMD128 ports)
// ============================================================================
//
// Mirrors the NEON / AVX2 ports one-to-one. 128-bit v128 width.

/// Zero-extend 8 u8 samples to 8 i16 lanes (low half of v128).
/// Samples are 0..255 → fits i16 exactly (zero-extend = same value).
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn load_8u_to_s16(src: *const u8) -> v128 {
    let v8 = unsafe { v128_load64_zero(src as *const u64) };
    // u16x8_extend_low_u8x16 zero-extends low 8 u8 lanes → 8 u16 lanes.
    // Reinterpret as i16 (positive values, identical bit pattern).
    u16x8_extend_low_u8x16(v8)
}

/// 6-tap horizontal/vertical filter in i16:
///   out = s0 - 5*s1 + 20*s2 + 20*s3 - 5*s4 + s5
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn apply_6tap_s16(
    s0: v128, s1: v128, s2: v128,
    s3: v128, s4: v128, s5: v128,
) -> v128 {
    let s2p3 = i16x8_add(s2, s3);
    let s1p4 = i16x8_add(s1, s4);
    let s0p5 = i16x8_add(s0, s5);
    let term20 = i16x8_mul(s2p3, i16x8_splat(20));
    let term5 = i16x8_mul(s1p4, i16x8_splat(5));
    i16x8_add(i16x8_sub(term20, term5), s0p5)
}

/// `clip1y((b1 + 16) >> 5)` for 8 i16 lanes → 8 u8 (low half packed).
/// Returns v128 with the 8 result bytes in lanes 0..7; lanes 8..15
/// are zero. Caller writes via `v128_store64_lane::<0>`.
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn b_from_b1(b1: v128) -> v128 {
    let off = i16x8_add(b1, i16x8_splat(16));
    let shr = i16x8_shr(off, 5);
    // u8x16_narrow_i16x8 saturates i16 → u8 (negatives → 0, >255 → 255)
    // = clip1y. Pack low 8 lanes from `shr`, upper 8 lanes from zero.
    u8x16_narrow_i16x8(shr, i16x8_splat(0))
}

/// Apply integer-MV (x_frac=0, y_frac=0) luma MC. Mirror of NEON/AVX2.
///
/// # Safety
/// Same as NEON `mc_luma_integer_mv`.
#[target_feature(enable = "simd128")]
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
            let v = unsafe { v128_load(src_ptr.add(src_off + x) as *const v128) };
            unsafe { v128_store(dst_ptr.add(dst_off + x) as *mut v128, v); }
            x += 16;
        }
        while x + 8 <= w {
            let v = unsafe { v128_load64_zero(src_ptr.add(src_off + x) as *const u64) };
            unsafe { v128_store64_lane::<0>(v, dst_ptr.add(dst_off + x) as *mut u64); }
            x += 8;
        }
    }
    true
}

/// Apply pure-horizontal half-pel luma MC. Mirror of NEON/AVX2.
///
/// # Safety
/// Same as NEON `mc_luma_h_only`.
#[target_feature(enable = "simd128")]
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
            let out_v = match x_frac {
                2 => b_packed,
                1 => {
                    let g = unsafe { v128_load64_zero(base as *const u64) };
                    u8x16_avgr(g, b_packed)
                }
                3 => {
                    let h_int = unsafe { v128_load64_zero(base.add(1) as *const u64) };
                    u8x16_avgr(h_int, b_packed)
                }
                _ => unreachable!(),
            };
            unsafe { v128_store64_lane::<0>(out_v, dst_ptr.add(dy * out_stride + x) as *mut u64); }
            x += 8;
        }
    }
    true
}

/// Apply pure-vertical half-pel luma MC. Mirror of NEON/AVX2.
///
/// # Safety
/// Same as NEON `mc_luma_v_only`.
#[target_feature(enable = "simd128")]
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
            let out_v = match y_frac {
                2 => h_packed,
                1 => {
                    let g = unsafe { v128_load64_zero(row_off(0) as *const u64) };
                    u8x16_avgr(g, h_packed)
                }
                3 => {
                    let m_int = unsafe { v128_load64_zero(row_off(1) as *const u64) };
                    u8x16_avgr(m_int, h_packed)
                }
                _ => unreachable!(),
            };
            unsafe { v128_store64_lane::<0>(out_v, dst_ptr.add(dy * out_stride + x) as *mut u64); }
            x += 8;
        }
    }
    true
}

/// Apply composite (x_frac > 0 AND y_frac > 0) luma MC. Mirror of
/// NEON/AVX2 `mc_luma_composite`. Covers the 9 remaining MC fraction
/// classes including the centre `j`.
///
/// j needs i32 widening for the second-pass filter. WASM SIMD128 has
/// direct sign-extending intrinsics (`i32x4_extend_low_i16x8` /
/// `i32x4_extend_high_i16x8`), cleaner than the AVX2 arithmetic-shift
/// trick.
///
/// # Safety
/// Same as NEON `mc_luma_composite`.
#[target_feature(enable = "simd128")]
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
            let h6 = |off: usize| -> v128 {
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

            let v6 = |col_extra: usize| -> v128 {
                let load = |delta: i32| -> v128 {
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
            let j_compute = || -> v128 {
                let one_half = |a_m2: v128, a_m1: v128, a_0: v128,
                                a_1: v128, a_2: v128, a_3: v128| -> v128 {
                    let sp01 = i32x4_add(a_0, a_1);
                    let sp12 = i32x4_add(a_m1, a_2);
                    let sp23 = i32x4_add(a_m2, a_3);
                    let term20 = i32x4_mul(sp01, i32x4_splat(20));
                    let term5 = i32x4_mul(sp12, i32x4_splat(5));
                    i32x4_add(i32x4_sub(term20, term5), sp23)
                };
                let lo = i32x4_extend_low_i16x8;
                let hi = i32x4_extend_high_i16x8;
                let j1_lo = one_half(
                    lo(b1_m2), lo(b1_m1), lo(b1_0),
                    lo(b1_1), lo(b1_2), lo(b1_3),
                );
                let j1_hi = one_half(
                    hi(b1_m2), hi(b1_m1), hi(b1_0),
                    hi(b1_1), hi(b1_2), hi(b1_3),
                );
                // j = clip1y((j1 + 512) >> 10)
                let off = i32x4_splat(512);
                let j_lo = i32x4_shr(i32x4_add(j1_lo, off), 10);
                let j_hi = i32x4_shr(i32x4_add(j1_hi, off), 10);
                // i32 → i16 (won't saturate; |j| ≤ 673 fits i16).
                let s16 = i16x8_narrow_i32x4(j_lo, j_hi);
                // i16 → u8 saturating = clip1y. Pack with zero; result
                // 8 bytes in low half.
                u8x16_narrow_i16x8(s16, i16x8_splat(0))
            };

            let result = match (x_frac, y_frac) {
                (1, 1) => {
                    let b0 = unsafe { b_from_b1(b1_0) };
                    let h0 = unsafe { b_from_b1(v6(0)) };
                    u8x16_avgr(b0, h0)
                }
                (2, 1) => {
                    let b0 = unsafe { b_from_b1(b1_0) };
                    let j = j_compute();
                    u8x16_avgr(b0, j)
                }
                (3, 1) => {
                    let b0 = unsafe { b_from_b1(b1_0) };
                    let h1 = unsafe { b_from_b1(v6(1)) };
                    u8x16_avgr(b0, h1)
                }
                (1, 2) => {
                    let h0 = unsafe { b_from_b1(v6(0)) };
                    let j = j_compute();
                    u8x16_avgr(h0, j)
                }
                (2, 2) => j_compute(),
                (3, 2) => {
                    let h1 = unsafe { b_from_b1(v6(1)) };
                    let j = j_compute();
                    u8x16_avgr(j, h1)
                }
                (1, 3) => {
                    let h0 = unsafe { b_from_b1(v6(0)) };
                    let b_one = unsafe { b_from_b1(b1_1) };
                    u8x16_avgr(h0, b_one)
                }
                (2, 3) => {
                    let j = j_compute();
                    let b_one = unsafe { b_from_b1(b1_1) };
                    u8x16_avgr(j, b_one)
                }
                (3, 3) => {
                    let h1 = unsafe { b_from_b1(v6(1)) };
                    let b_one = unsafe { b_from_b1(b1_1) };
                    u8x16_avgr(h1, b_one)
                }
                _ => unreachable!(),
            };
            unsafe { v128_store64_lane::<0>(result, dst_ptr.add(dy * out_stride + x) as *mut u64); }
            x += 8;
        }
    }
    true
}
