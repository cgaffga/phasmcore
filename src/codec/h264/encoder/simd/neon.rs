// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! NEON (aarch64) SIMD kernels for the H.264 encoder hot paths.
//!
//! Integer-only and FMA-free per the determinism contract documented
//! in the parent module. NEON on aarch64 is mandatory in the
//! architecture spec — every shipping iPhone, M-series Mac, and
//! arm64-v8a Android device has it — so we don't gate behind runtime
//! feature detection.
//!
//! All kernels are `unsafe fn`. Safety contracts:
//! - Caller guarantees the read ranges fit within the source / pred
//!   slices (mirrors the scalar implementation's read pattern). The
//!   dispatcher in `simd::mod` enforces this by only calling kernels
//!   that match the scalar-sized callers.
//! - All loads are unaligned (`vld1q_*`) — no alignment requirement
//!   on the input slices.

use core::arch::aarch64::*;

// ============================================================================
// SAD — Sum of Absolute Differences
// ============================================================================

/// SAD over a 16-wide × `h`-tall block.
///
/// # Bound proof
/// Each `vabdq_u8` lane holds an absolute difference ≤ 255. Pairwise add
/// (`vpaddlq_u8`) widens to u16, so each u16 lane holds ≤ 510. We
/// accumulate up to 16 rows (max H.264 partition height): per-lane sum
/// ≤ 16 × 510 = 8160, well within u16 range. Final horizontal sum
/// (`vaddlvq_u16`) widens to u32: total ≤ 8 × 8160 = 65280, fits u32.
///
/// # Safety
/// `source[y * source_stride + 0..16]` and `pred[y * pred_stride + 0..16]`
/// must be valid for `y in 0..h`. Caller (the dispatcher) ensures this.
#[target_feature(enable = "neon")]
pub(super) unsafe fn sad_w16(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> u32 {
    let src_ptr = source.as_ptr();
    let prd_ptr = pred.as_ptr();
    let mut acc = vdupq_n_u16(0);
    let mut y = 0;
    while y < h {
        let s = unsafe { vld1q_u8(src_ptr.add(y * source_stride)) };
        let p = unsafe { vld1q_u8(prd_ptr.add(y * pred_stride)) };
        let d = vabdq_u8(s, p);
        acc = vaddq_u16(acc, vpaddlq_u8(d));
        y += 1;
    }
    vaddlvq_u16(acc)
}

/// SAD over an 8-wide × `h`-tall block. See `sad_w16` for the general
/// pattern; uses the half-register (uint8x8_t) variants.
///
/// # Bound proof
/// Per-lane after pairwise add ≤ 510; over h ≤ 16 rows ≤ 8160 (u16).
/// Final horizontal sum across 4 lanes ≤ 4 × 8160 = 32640 (u32).
///
/// # Safety
/// `source[y * source_stride + 0..8]` and `pred[y * pred_stride + 0..8]`
/// must be valid for `y in 0..h`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn sad_w8(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    h: usize,
) -> u32 {
    let src_ptr = source.as_ptr();
    let prd_ptr = pred.as_ptr();
    let mut acc = vdup_n_u16(0);
    let mut y = 0;
    while y < h {
        let s = unsafe { vld1_u8(src_ptr.add(y * source_stride)) };
        let p = unsafe { vld1_u8(prd_ptr.add(y * pred_stride)) };
        let d = vabd_u8(s, p);
        acc = vadd_u16(acc, vpaddl_u8(d));
        y += 1;
    }
    vaddlv_u16(acc)
}

// ============================================================================
// SATD — Sum of Absolute Hadamard-Transformed Differences (4×4 tiled)
// ============================================================================

/// 4×4 Hadamard transform on i32 inputs, vectorised. Used by SATD.
///
/// Computes Y = H · X · H where H is the 4×4 Walsh-Hadamard matrix.
/// Strategy: column-stage on rows of X (mixes rows = adds across
/// rows), transpose, column-stage again (which is now mixing the
/// transposed rows = original columns), transpose back.
///
/// # Bound proof
/// Inputs are residuals (src - pred) with absolute value ≤ 255. After
/// each butterfly stage the absolute value can grow by a factor of 4
/// (sum of 4 inputs in worst case). After 2 stages: ≤ 16 × 255 = 4080.
/// Fits i16; we use i32 for headroom and to match the existing scalar
/// signature.
///
/// # Safety
/// Operates on `[i32; 4]` rows from a `&[[i32; 4]; 4]`. No external
/// pointer arithmetic.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hadamard_4x4(input: &[[i32; 4]; 4]) -> [int32x4_t; 4] {
    let r0 = unsafe { vld1q_s32(input[0].as_ptr()) };
    let r1 = unsafe { vld1q_s32(input[1].as_ptr()) };
    let r2 = unsafe { vld1q_s32(input[2].as_ptr()) };
    let r3 = unsafe { vld1q_s32(input[3].as_ptr()) };

    // Stage 1 — column butterflies (mix rows of X).
    let p02 = vaddq_s32(r0, r2);
    let m02 = vsubq_s32(r0, r2);
    let p13 = vaddq_s32(r1, r3);
    let m13 = vsubq_s32(r1, r3);
    let a0 = vaddq_s32(p02, p13);
    let a1 = vaddq_s32(m02, m13);
    let a2 = vsubq_s32(m02, m13);
    let a3 = vsubq_s32(p02, p13);

    // Transpose 4x4 (i32 lanes). aarch64 idiom: vtrn1q_s32 / vtrn2q_s32
    // for 32-bit pairs, then vtrn1q_s64 / vtrn2q_s64 (via reinterpret)
    // for 64-bit halves.
    let [b0, b1, b2, b3] = unsafe { transpose_4x4_s32(a0, a1, a2, a3) };

    // Stage 2 — column butterflies on transposed (= row butterflies on original).
    let p02 = vaddq_s32(b0, b2);
    let m02 = vsubq_s32(b0, b2);
    let p13 = vaddq_s32(b1, b3);
    let m13 = vsubq_s32(b1, b3);
    let c0 = vaddq_s32(p02, p13);
    let c1 = vaddq_s32(m02, m13);
    let c2 = vsubq_s32(m02, m13);
    let c3 = vsubq_s32(p02, p13);

    // Transpose back to row-major.
    unsafe { transpose_4x4_s32(c0, c1, c2, c3) }
}

/// 4×4 i32 transpose helper. Returns the transposed rows.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn transpose_4x4_s32(
    a0: int32x4_t,
    a1: int32x4_t,
    a2: int32x4_t,
    a3: int32x4_t,
) -> [int32x4_t; 4] {
    // vtrn1q_s32 / vtrn2q_s32 interleave i32 lanes pairwise:
    //   t01 = vtrn1q(a0, a1) = [a0[0], a1[0], a0[2], a1[2]]
    //   t11 = vtrn2q(a0, a1) = [a0[1], a1[1], a0[3], a1[3]]
    let t01 = vtrn1q_s32(a0, a1);
    let t11 = vtrn2q_s32(a0, a1);
    let t02 = vtrn1q_s32(a2, a3);
    let t12 = vtrn2q_s32(a2, a3);

    // Now interleave 64-bit halves to finish the 4×4 transpose.
    let t01_64 = vreinterpretq_s64_s32(t01);
    let t11_64 = vreinterpretq_s64_s32(t11);
    let t02_64 = vreinterpretq_s64_s32(t02);
    let t12_64 = vreinterpretq_s64_s32(t12);

    let b0 = vreinterpretq_s32_s64(vtrn1q_s64(t01_64, t02_64));
    let b1 = vreinterpretq_s32_s64(vtrn1q_s64(t11_64, t12_64));
    let b2 = vreinterpretq_s32_s64(vtrn2q_s64(t01_64, t02_64));
    let b3 = vreinterpretq_s32_s64(vtrn2q_s64(t11_64, t12_64));
    [b0, b1, b2, b3]
}

/// SATD over `block_w × block_h` (both multiples of 4), tiled into
/// 4×4 Hadamard sums. Vectorised residual-load + Hadamard +
/// abs-sum.
///
/// # Safety
/// `source[y * source_stride + 0..block_w]` and
/// `pred[y * pred_stride + 0..block_w]` must be valid for `y in 0..block_h`.
#[target_feature(enable = "neon")]
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
            // Load 4×4 residual into [[i32; 4]; 4]. Use `vsubl_u8` to
            // get a 16-bit signed difference, then widen to i32.
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
            // Hadamard 4×4 (vectorised).
            let h = unsafe { hadamard_4x4(&residual) };
            // Sum |h[i][j]| via |s32| + horizontal add. Each row in
            // i32; abs + horizontal pairwise sum + horizontal sum.
            let s0 = vabsq_s32(h[0]);
            let s1 = vabsq_s32(h[1]);
            let s2 = vabsq_s32(h[2]);
            let s3 = vabsq_s32(h[3]);
            // Reinterpret abs values as u32 (always non-negative)
            // and sum.
            let u0 = vreinterpretq_u32_s32(s0);
            let u1 = vreinterpretq_u32_s32(s1);
            let u2 = vreinterpretq_u32_s32(s2);
            let u3 = vreinterpretq_u32_s32(s3);
            let row_sums = vaddq_u32(vaddq_u32(u0, u1), vaddq_u32(u2, u3));
            // Horizontal sum across the 4 lanes.
            let tile_sum = vaddvq_u32(row_sums);
            total = total.saturating_add(tile_sum);
        }
    }
    total
}

// ============================================================================
// Motion compensation — luma (Phase I.2b)
// ============================================================================
//
// Three NEON kernels covering 7/16 (x_frac, y_frac) cases:
//   - integer MV (1 case): edge-replication memcpy
//   - pure horizontal (3 cases, y_frac=0, x_frac in 1,2,3): row 6-tap +
//     optional bilinear blend with integer sample
//   - pure vertical (3 cases, x_frac=0, y_frac in 1,2,3): column 6-tap +
//     optional bilinear blend
//
// Composite cases (x_frac & y_frac both non-zero) fall back to scalar.
// All three NEON kernels require `block_w.is_multiple_of(8)` and a block
// fully inside the reference frame (no edge replication needed). Edge or
// odd-width blocks fall back to scalar.

/// Apply integer-MV (x_frac=0, y_frac=0) luma MC via NEON 16-byte
/// or 8-byte memcpy, no filtering. Returns false if the block straddles
/// a frame edge (caller falls back to scalar with edge replication).
///
/// # Safety
/// `y_plane[y * plane_w + 0..plane_w]` valid for `y in 0..plane_h`.
/// `out[y * out_stride + 0..block_w]` valid for `y in 0..block_h`.
#[target_feature(enable = "neon")]
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
            let v = unsafe { vld1q_u8(src_ptr.add(src_off + x)) };
            unsafe { vst1q_u8(dst_ptr.add(dst_off + x), v); }
            x += 16;
        }
        while x + 8 <= w {
            let v = unsafe { vld1_u8(src_ptr.add(src_off + x)) };
            unsafe { vst1_u8(dst_ptr.add(dst_off + x), v); }
            x += 8;
        }
    }
    true
}

/// Apply pure-horizontal half-pel luma MC. `x_frac` ∈ {1, 2, 3},
/// `y_frac == 0`. Returns false if edge replication needed.
///
/// Algorithm:
/// - For each output row, load `block_w + 5` source samples (offset -2..+block_w+3)
/// - Horizontal 6-tap filter on each 8-pixel chunk produces b1 (i16)
/// - Half-pel sample b = clip1y((b1 + 16) >> 5)
/// - For x_frac == 2: output = b
/// - For x_frac == 1: output = (G + b + 1) >> 1 where G is integer sample at offset 0
/// - For x_frac == 3: output = (H + b + 1) >> 1 where H is integer sample at offset 1
///
/// # Safety
/// Same as `mc_luma_integer_mv`. The kernel reads `block_w + 5` samples
/// per row starting at `src_x - 2`; bounds-checks ensure this is in-frame
/// before calling NEON loads.
#[target_feature(enable = "neon")]
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
    // Horizontal filter needs samples [src_x - 2, src_x + block_w + 3].
    // For x_frac == 3 we also need an additional integer sample at +1
    // for the bilinear blend (already covered by + 3 right margin).
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
    let const16 = vdupq_n_s16(16);
    for dy in 0..block_h as usize {
        let row_base = (src_y + dy) * plane_w_us + src_x;
        let mut x = 0;
        while x + 8 <= w {
            // Load 8 samples at each of 6 horizontal offsets (-2..+3).
            // SAFETY: bounds-checked above.
            let s0 = unsafe { vld1_u8(src_ptr.add(row_base + x).wrapping_sub(2)) };
            let s1 = unsafe { vld1_u8(src_ptr.add(row_base + x).wrapping_sub(1)) };
            let s2 = unsafe { vld1_u8(src_ptr.add(row_base + x)) };
            let s3 = unsafe { vld1_u8(src_ptr.add(row_base + x + 1)) };
            let s4 = unsafe { vld1_u8(src_ptr.add(row_base + x + 2)) };
            let s5 = unsafe { vld1_u8(src_ptr.add(row_base + x + 3)) };
            // Promote to i16, apply 6-tap: b1 = s0 - 5*s1 + 20*s2 + 20*s3 - 5*s4 + s5
            let i0 = vreinterpretq_s16_u16(vmovl_u8(s0));
            let i1 = vreinterpretq_s16_u16(vmovl_u8(s1));
            let i2 = vreinterpretq_s16_u16(vmovl_u8(s2));
            let i3 = vreinterpretq_s16_u16(vmovl_u8(s3));
            let i4 = vreinterpretq_s16_u16(vmovl_u8(s4));
            let i5 = vreinterpretq_s16_u16(vmovl_u8(s5));
            let s2p3 = vaddq_s16(i2, i3);
            let s1p4 = vaddq_s16(i1, i4);
            let s0p5 = vaddq_s16(i0, i5);
            // 20*(s2+s3) - 5*(s1+s4) + (s0+s5):
            //   = (s2+s3) << 4 + (s2+s3) << 2  [= 20×]
            //     - ((s1+s4) << 2 + (s1+s4))   [= 5×]
            //     + (s0+s5)
            // Use multiply instructions for clarity (NEON has vmulq_n_s16):
            let term20 = vmulq_n_s16(s2p3, 20);
            let term5  = vmulq_n_s16(s1p4, 5);
            let b1 = vaddq_s16(vsubq_s16(term20, term5), s0p5);
            // b = clip1y((b1 + 16) >> 5)
            let b1_off = vaddq_s16(b1, const16);
            let b_i16 = vshrq_n_s16(b1_off, 5);
            // Saturate to u8 (clip to 0..255).
            let b_u8 = vqmovun_s16(b_i16);
            // x_frac dispatch.
            let out_u8 = match x_frac {
                2 => b_u8,
                1 => {
                    // (G + b + 1) >> 1; G = integer sample at offset 0 = s2.
                    vrhadd_u8(s2, b_u8)
                }
                3 => {
                    // (H + b + 1) >> 1; H = integer sample at offset +1 = s3.
                    vrhadd_u8(s3, b_u8)
                }
                _ => unreachable!(),
            };
            unsafe { vst1_u8(dst_ptr.add(dy * out_stride + x), out_u8); }
            x += 8;
        }
    }
    true
}

/// Apply pure-vertical half-pel luma MC. `y_frac` ∈ {1, 2, 3},
/// `x_frac == 0`. Returns false if edge replication needed.
///
/// Algorithm: same as `mc_luma_h_only` but along the column axis.
/// Loads 6 source rows (offset -2..+3) per output row, applies the
/// 6-tap kernel vertically.
///
/// # Safety
/// Same as `mc_luma_h_only`.
#[target_feature(enable = "neon")]
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
    let const16 = vdupq_n_s16(16);
    for dy in 0..block_h as usize {
        let mut x = 0;
        while x + 8 <= w {
            // Load 8 samples from each of 6 vertical neighbors.
            // SAFETY: bounds-checked above.
            let row = |delta: i32| -> uint8x8_t {
                let off = ((src_y as i32 + dy as i32 + delta) as usize) * plane_w_us + src_x + x;
                unsafe { vld1_u8(src_ptr.add(off)) }
            };
            let s0 = row(-2);
            let s1 = row(-1);
            let s2 = row(0);
            let s3 = row(1);
            let s4 = row(2);
            let s5 = row(3);
            let i0 = vreinterpretq_s16_u16(vmovl_u8(s0));
            let i1 = vreinterpretq_s16_u16(vmovl_u8(s1));
            let i2 = vreinterpretq_s16_u16(vmovl_u8(s2));
            let i3 = vreinterpretq_s16_u16(vmovl_u8(s3));
            let i4 = vreinterpretq_s16_u16(vmovl_u8(s4));
            let i5 = vreinterpretq_s16_u16(vmovl_u8(s5));
            let s2p3 = vaddq_s16(i2, i3);
            let s1p4 = vaddq_s16(i1, i4);
            let s0p5 = vaddq_s16(i0, i5);
            let term20 = vmulq_n_s16(s2p3, 20);
            let term5  = vmulq_n_s16(s1p4, 5);
            let h1 = vaddq_s16(vsubq_s16(term20, term5), s0p5);
            let h1_off = vaddq_s16(h1, const16);
            let h_i16 = vshrq_n_s16(h1_off, 5);
            let h_u8 = vqmovun_s16(h_i16);
            let out_u8 = match y_frac {
                2 => h_u8,
                1 => {
                    // (G + h + 1) >> 1; G = integer sample at delta 0 = s2.
                    vrhadd_u8(s2, h_u8)
                }
                3 => {
                    // (M + h + 1) >> 1; M = integer sample at delta +1 = s3.
                    vrhadd_u8(s3, h_u8)
                }
                _ => unreachable!(),
            };
            unsafe { vst1_u8(dst_ptr.add(dy * out_stride + x), out_u8); }
            x += 8;
        }
    }
    true
}

/// Apply composite (x_frac > 0 AND y_frac > 0) luma MC. Covers all
/// 9 remaining MC fraction classes that `mc_luma_h_only` /
/// `mc_luma_v_only` don't:
///   (1,1) e, (2,1) f, (3,1) g, (1,2) i, (2,2) j, (3,2) k,
///   (1,3) p, (2,3) q, (3,3) r
///
/// Strategy: per 8-pixel output chunk, compute the shared intermediates
/// once — b1 at 6 vertical offsets [dy-2..dy+3] (which directly yields
/// b(0), b(1), and j) — then h(0) and optionally h(1) via vertical
/// 6-tap, then combine per (x_frac, y_frac).
///
/// j needs i32 arithmetic for the second-pass filter: |b1| ≤ 13260
/// (sum-abs 52 × 255), so |j1| = |sum-abs 52 × b1| ≤ 689,520 — exceeds
/// i16 range. We widen to i32 lanes (low + high i32x4_t pairs) before
/// the second filter, then narrow back to u8.
///
/// Returns false if edge replication needed (composite needs source
/// rows [src_y-2 .. src_y+block_h+3] and columns
/// [src_x-2 .. src_x+block_w+4]; +4 because h(1) at the rightmost
/// chunk reads source col x+8+3 = x+11).
///
/// # Safety
/// `y_plane[y * plane_w + 0..plane_w]` valid for `y in 0..plane_h`.
/// Bounds-checked at entry.
#[target_feature(enable = "neon")]
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

    let const16 = vdupq_n_s16(16);
    let const512 = vdupq_n_s32(512);

    let src_ptr = y_plane.as_ptr();
    let dst_ptr = out.as_mut_ptr();

    for dy in 0..block_h as usize {
        let mut x = 0;
        while x + 8 <= w {
            // Source row offsets for the 6 vertical neighbors.
            let row_off = |delta: i32| -> usize {
                ((src_y as i32 + dy as i32 + delta) as usize) * plane_w_us
                    + src_x
                    + x
            };
            // Horizontal 6-tap at one row, returning b1 (int16x8_t).
            // SAFETY: bounds-checked at entry; off-2..off+10 in-frame.
            let h6 = |off: usize| -> int16x8_t {
                let s0 = unsafe { vld1_u8(src_ptr.add(off).wrapping_sub(2)) };
                let s1 = unsafe { vld1_u8(src_ptr.add(off).wrapping_sub(1)) };
                let s2 = unsafe { vld1_u8(src_ptr.add(off)) };
                let s3 = unsafe { vld1_u8(src_ptr.add(off + 1)) };
                let s4 = unsafe { vld1_u8(src_ptr.add(off + 2)) };
                let s5 = unsafe { vld1_u8(src_ptr.add(off + 3)) };
                let i0 = vreinterpretq_s16_u16(vmovl_u8(s0));
                let i1 = vreinterpretq_s16_u16(vmovl_u8(s1));
                let i2 = vreinterpretq_s16_u16(vmovl_u8(s2));
                let i3 = vreinterpretq_s16_u16(vmovl_u8(s3));
                let i4 = vreinterpretq_s16_u16(vmovl_u8(s4));
                let i5 = vreinterpretq_s16_u16(vmovl_u8(s5));
                let s2p3 = vaddq_s16(i2, i3);
                let s1p4 = vaddq_s16(i1, i4);
                let s0p5 = vaddq_s16(i0, i5);
                vaddq_s16(
                    vsubq_s16(vmulq_n_s16(s2p3, 20), vmulq_n_s16(s1p4, 5)),
                    s0p5,
                )
            };
            let b1_m2 = h6(row_off(-2));
            let b1_m1 = h6(row_off(-1));
            let b1_0 = h6(row_off(0));
            let b1_1 = h6(row_off(1));
            let b1_2 = h6(row_off(2));
            let b1_3 = h6(row_off(3));

            // clip1y((b1 + 16) >> 5)
            let b_from = |b1: int16x8_t| -> uint8x8_t {
                vqmovun_s16(vshrq_n_s16(vaddq_s16(b1, const16), 5))
            };

            // Vertical 6-tap at 8 cols starting `col_extra` past src_x+x.
            // Returns h1 (int16x8_t). SAFETY: bounds-checked at entry.
            let v6 = |col_extra: usize| -> int16x8_t {
                let load = |delta: i32| -> uint8x8_t {
                    let off = ((src_y as i32 + dy as i32 + delta) as usize)
                        * plane_w_us
                        + src_x + x + col_extra;
                    unsafe { vld1_u8(src_ptr.add(off)) }
                };
                let s0 = load(-2);
                let s1 = load(-1);
                let s2 = load(0);
                let s3 = load(1);
                let s4 = load(2);
                let s5 = load(3);
                let i0 = vreinterpretq_s16_u16(vmovl_u8(s0));
                let i1 = vreinterpretq_s16_u16(vmovl_u8(s1));
                let i2 = vreinterpretq_s16_u16(vmovl_u8(s2));
                let i3 = vreinterpretq_s16_u16(vmovl_u8(s3));
                let i4 = vreinterpretq_s16_u16(vmovl_u8(s4));
                let i5 = vreinterpretq_s16_u16(vmovl_u8(s5));
                let s2p3 = vaddq_s16(i2, i3);
                let s1p4 = vaddq_s16(i1, i4);
                let s0p5 = vaddq_s16(i0, i5);
                vaddq_s16(
                    vsubq_s16(vmulq_n_s16(s2p3, 20), vmulq_n_s16(s1p4, 5)),
                    s0p5,
                )
            };
            let h_from = |h1: int16x8_t| -> uint8x8_t {
                vqmovun_s16(vshrq_n_s16(vaddq_s16(h1, const16), 5))
            };

            // j from 6 b1 vectors. Widen to i32 (low/high halves)
            // because the sum-abs-52 weighted sum exceeds i16 range.
            let j_compute = || -> uint8x8_t {
                // Widen i16 → i32 (low + high).
                let lo = |v: int16x8_t| vmovl_s16(vget_low_s16(v));
                let hi = |v: int16x8_t| vmovl_s16(vget_high_s16(v));
                let one_half = |a_m2: int32x4_t,
                                a_m1: int32x4_t,
                                a_0: int32x4_t,
                                a_1: int32x4_t,
                                a_2: int32x4_t,
                                a_3: int32x4_t|
                 -> int32x4_t {
                    // j1 = a-2 - 5*a-1 + 20*a0 + 20*a1 - 5*a2 + a3
                    let sp01 = vaddq_s32(a_0, a_1);
                    let sp12 = vaddq_s32(a_m1, a_2);
                    let sp23 = vaddq_s32(a_m2, a_3);
                    vaddq_s32(
                        vsubq_s32(
                            vmulq_n_s32(sp01, 20),
                            vmulq_n_s32(sp12, 5),
                        ),
                        sp23,
                    )
                };
                let j1_lo = one_half(
                    lo(b1_m2), lo(b1_m1), lo(b1_0), lo(b1_1), lo(b1_2), lo(b1_3),
                );
                let j1_hi = one_half(
                    hi(b1_m2), hi(b1_m1), hi(b1_0), hi(b1_1), hi(b1_2), hi(b1_3),
                );
                // j = clip1y((j1 + 512) >> 10)
                let j_lo = vshrq_n_s32(vaddq_s32(j1_lo, const512), 10);
                let j_hi = vshrq_n_s32(vaddq_s32(j1_hi, const512), 10);
                // i32 → i16 (won't saturate; |j| ≤ 690k>>10 = 673 fits i16).
                let s16 = vcombine_s16(vqmovn_s32(j_lo), vqmovn_s32(j_hi));
                // i16 → u8 saturation = clip1y.
                vqmovun_s16(s16)
            };

            let result = match (x_frac, y_frac) {
                (1, 1) => vrhadd_u8(b_from(b1_0), h_from(v6(0))), // e
                (2, 1) => vrhadd_u8(b_from(b1_0), j_compute()),   // f
                (3, 1) => vrhadd_u8(b_from(b1_0), h_from(v6(1))), // g
                (1, 2) => vrhadd_u8(h_from(v6(0)), j_compute()),  // i
                (2, 2) => j_compute(),                            // j
                (3, 2) => vrhadd_u8(j_compute(), h_from(v6(1))),  // k
                (1, 3) => vrhadd_u8(h_from(v6(0)), b_from(b1_1)), // p
                (2, 3) => vrhadd_u8(j_compute(), b_from(b1_1)),   // q
                (3, 3) => vrhadd_u8(h_from(v6(1)), b_from(b1_1)), // r
                _ => unreachable!(),
            };
            unsafe { vst1_u8(dst_ptr.add(dy * out_stride + x), result); }
            x += 8;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_sad(s: &[u8], ss: usize, p: &[u8], ps: usize, w: u32, h: u32) -> u32 {
        let mut sum = 0u32;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let d = s[y * ss + x] as i32 - p[y * ps + x] as i32;
                sum += d.unsigned_abs();
            }
        }
        sum
    }

    fn scalar_satd(s: &[u8], ss: usize, p: &[u8], ps: usize, w: u32, h: u32) -> u32 {
        let mut total: u32 = 0;
        let tiles_y = (h / 4) as usize;
        let tiles_x = (w / 4) as usize;
        for by in 0..tiles_y {
            for bx in 0..tiles_x {
                let mut residual = [[0i32; 4]; 4];
                for dy in 0..4 {
                    for dx in 0..4 {
                        let sx = bx * 4 + dx;
                        let sy = by * 4 + dy;
                        residual[dy][dx] =
                            s[sy * ss + sx] as i32 - p[sy * ps + sx] as i32;
                    }
                }
                let mut f = [[0i32; 4]; 4];
                for i in 0..4 {
                    let a0 = residual[i][0] + residual[i][2];
                    let a1 = residual[i][1] + residual[i][3];
                    let a2 = residual[i][0] - residual[i][2];
                    let a3 = residual[i][1] - residual[i][3];
                    f[i][0] = a0 + a1;
                    f[i][1] = a2 + a3;
                    f[i][2] = a2 - a3;
                    f[i][3] = a0 - a1;
                }
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
                for row in &y {
                    for &v in row {
                        total = total.saturating_add(v.unsigned_abs());
                    }
                }
            }
        }
        total
    }

    fn make_buf(seed: u32) -> [u8; 256] {
        let mut buf = [0u8; 256];
        let mut s = seed;
        for b in buf.iter_mut() {
            // Tiny LCG so the buffer has structure rather than gradients.
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (s >> 16) as u8;
        }
        buf
    }

    #[test]
    fn neon_sad_w16_matches_scalar() {
        let src = make_buf(0xdead_beef);
        let prd = make_buf(0xcafe_f00d);
        for h in [4, 8, 12, 16] {
            let want = scalar_sad(&src, 16, &prd, 16, 16, h);
            let got = unsafe { sad_w16(&src, 16, &prd, 16, h as usize) };
            assert_eq!(got, want, "sad_w16 h={h}");
        }
    }

    #[test]
    fn neon_sad_w8_matches_scalar() {
        let src = make_buf(0x1234_5678);
        let prd = make_buf(0x9abc_def0);
        for h in [4, 8, 12, 16] {
            let want = scalar_sad(&src, 16, &prd, 16, 8, h);
            let got = unsafe { sad_w8(&src, 16, &prd, 16, h as usize) };
            assert_eq!(got, want, "sad_w8 h={h}");
        }
    }

    #[test]
    fn neon_satd_4x4_matches_scalar() {
        let src = make_buf(0xa5a5_a5a5);
        let prd = make_buf(0x5a5a_5a5a);
        for (w, h) in [(4u32, 4u32), (8, 4), (4, 8), (8, 8), (16, 8), (8, 16), (16, 16)] {
            let want = scalar_satd(&src, 16, &prd, 16, w, h);
            let got = unsafe {
                satd_block_4x4_tiled(&src, 16, &prd, 16, w as usize, h as usize)
            };
            assert_eq!(got, want, "satd {w}x{h}");
        }
    }

    #[test]
    fn neon_sad_zero_diff() {
        // Identical source and pred → SAD must be 0.
        let buf = make_buf(0x42);
        let h = 16;
        unsafe {
            assert_eq!(sad_w16(&buf, 16, &buf, 16, h as usize), 0);
            assert_eq!(sad_w8(&buf, 16, &buf, 16, h as usize), 0);
        }
    }

    #[test]
    fn neon_satd_zero_diff() {
        let buf = make_buf(0x99);
        unsafe {
            assert_eq!(satd_block_4x4_tiled(&buf, 16, &buf, 16, 16, 16), 0);
        }
    }
}
