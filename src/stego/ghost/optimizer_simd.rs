// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! SIMD-accelerated kernels for the Ghost cover optimizer.
//!
//! ## Phase C2 contract (2026-05-22)
//!
//! Per-platform deterministic; cross-platform bit-exactness no longer
//! required after #678 confirmed no code derives keys/salts from
//! optimizer output bytes (wire-format cross-platform decode is
//! independent). This unlocks:
//!
//! - **f32x4 lanes**: 2× the throughput of the previous f64x2 path.
//! - **NEON FMA** (`vfmaq_f32`): essentially free on Apple Silicon;
//!   `acc + a*b` in one rounding step. Each platform may round
//!   differently; that's fine under the new contract.
//! - **WASM native f32x4**: WASM SIMD128's f64x2 was synthetic /
//!   slow; f32x4 is the native lane width and 2-3× faster.
//!
//! ## Pre-conditioning
//!
//! The Gaussian blur kernels use pre-padded planar f32 buffers (one
//! per channel, length `w + 2*radius`, clamp-to-edge extension).
//! This eliminates the per-iteration `saturating_sub + min` branch
//! the scalar code used and lets the SIMD loop be a flat sequence of
//! contiguous loads.
//!
//! ## Per-arch status (2026-05-22)
//!
//! - aarch64 NEON: f32x4 with FMA (`vfmaq_f32`)
//! - x86_64 SSE2 baseline: f32x4 (`_mm_mul_ps` / `_mm_add_ps`). FMA3
//!   intentionally skipped; gain is small (+5-10%) on a non-critical
//!   dev/CLI path.
//! - WASM SIMD128: f32x4 (`f32x4_mul` / `f32x4_add`). Relaxed-simd
//!   FMA not yet stable enough to depend on.
//! - Other archs: scalar fallback in f32.

/// SIMD-aware Gaussian blur horizontal pass for a single row.
///
/// `padded_row` must hold one channel's `w + 2*radius` f32 samples
/// in contiguous memory (clamp-to-edge extension already applied).
/// `kernel` is the normalized 1D Gaussian kernel of length
/// `kernel_size = 2*radius + 1`. Writes the blurred `w` output
/// values (rounded + clamped) into `dst` at the indicated channel
/// offset + stride.
///
/// `dst_offset_bytes` = `c` (channel index), `dst_stride_bytes` = 3
/// (RGB interleaved).
#[inline]
pub(super) fn blur_row_h(
    padded_row: &[f32],
    kernel: &[f32],
    radius: usize,
    w: usize,
    dst: &mut [u8],
    dst_offset_bytes: usize,
    dst_stride_bytes: usize,
) {
    debug_assert_eq!(padded_row.len(), w + 2 * radius);
    debug_assert_eq!(kernel.len(), 2 * radius + 1);

    #[cfg(target_arch = "aarch64")]
    // Safety: NEON is mandatory on aarch64.
    unsafe {
        blur_row_h_neon(padded_row, kernel, radius, w, dst, dst_offset_bytes, dst_stride_bytes);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    unsafe {
        blur_row_h_sse(padded_row, kernel, radius, w, dst, dst_offset_bytes, dst_stride_bytes);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        blur_row_h_wasm(padded_row, kernel, radius, w, dst, dst_offset_bytes, dst_stride_bytes);
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse2"),
        all(target_arch = "wasm32", target_feature = "simd128"),
    )))]
    blur_row_h_scalar(padded_row, kernel, radius, w, dst, dst_offset_bytes, dst_stride_bytes);
}

#[inline]
pub(super) fn blur_row_h_scalar(
    padded_row: &[f32],
    kernel: &[f32],
    _radius: usize,
    w: usize,
    dst: &mut [u8],
    dst_offset_bytes: usize,
    dst_stride_bytes: usize,
) {
    let kernel_size = kernel.len();
    for x in 0..w {
        let mut acc = 0.0f32;
        for k in 0..kernel_size {
            acc += padded_row[x + k] * kernel[k];
        }
        dst[x * dst_stride_bytes + dst_offset_bytes] =
            acc.round().clamp(0.0, 255.0) as u8;
    }
}

// ─── aarch64 NEON f32x4 with FMA ────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn blur_row_h_neon(
    padded_row: &[f32],
    kernel: &[f32],
    _radius: usize,
    w: usize,
    dst: &mut [u8],
    dst_offset_bytes: usize,
    dst_stride_bytes: usize,
) {
    use core::arch::aarch64::*;
    let kernel_size = kernel.len();
    let pr = padded_row.as_ptr();

    let mut x = 0;
    // f32x4 lane-parallel: 4 output pixels per iteration with FMA.
    // `vfmaq_f32(acc, a, b)` = acc + a*b in one rounded op. Cheaper
    // than separate mul+add on aarch64 and produces (slightly)
    // different rounding than the scalar path — fine under the
    // C2 per-platform-deterministic contract.
    while x + 4 <= w {
        let mut acc = vdupq_n_f32(0.0);
        for k in 0..kernel_size {
            let kv = vdupq_n_f32(kernel[k]);
            let inv = vld1q_f32(pr.add(x + k));
            acc = vfmaq_f32(acc, inv, kv);
        }
        let l0 = vgetq_lane_f32::<0>(acc);
        let l1 = vgetq_lane_f32::<1>(acc);
        let l2 = vgetq_lane_f32::<2>(acc);
        let l3 = vgetq_lane_f32::<3>(acc);
        dst[x * dst_stride_bytes + dst_offset_bytes] = l0.round().clamp(0.0, 255.0) as u8;
        dst[(x + 1) * dst_stride_bytes + dst_offset_bytes] = l1.round().clamp(0.0, 255.0) as u8;
        dst[(x + 2) * dst_stride_bytes + dst_offset_bytes] = l2.round().clamp(0.0, 255.0) as u8;
        dst[(x + 3) * dst_stride_bytes + dst_offset_bytes] = l3.round().clamp(0.0, 255.0) as u8;
        x += 4;
    }
    // Scalar tail (at most 3 pixels). Uses FMA-equivalent
    // `f32::mul_add` so the boundary pixels round the same way the
    // SIMD lanes do.
    while x < w {
        let mut acc = 0.0f32;
        for k in 0..kernel_size {
            acc = padded_row[x + k].mul_add(kernel[k], acc);
        }
        dst[x * dst_stride_bytes + dst_offset_bytes] =
            acc.round().clamp(0.0, 255.0) as u8;
        x += 1;
    }
}

// ─── x86_64 SSE2 f32x4 (no FMA3 per C2 scope) ───────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[target_feature(enable = "sse2")]
unsafe fn blur_row_h_sse(
    padded_row: &[f32],
    kernel: &[f32],
    _radius: usize,
    w: usize,
    dst: &mut [u8],
    dst_offset_bytes: usize,
    dst_stride_bytes: usize,
) {
    use core::arch::x86_64::*;
    let kernel_size = kernel.len();
    let pr = padded_row.as_ptr();

    let mut x = 0;
    while x + 4 <= w {
        let mut acc = _mm_setzero_ps();
        for k in 0..kernel_size {
            let kv = _mm_set1_ps(kernel[k]);
            let inv = _mm_loadu_ps(pr.add(x + k));
            let prod = _mm_mul_ps(inv, kv);
            acc = _mm_add_ps(acc, prod);
        }
        let mut lanes = [0.0f32; 4];
        _mm_storeu_ps(lanes.as_mut_ptr(), acc);
        for i in 0..4 {
            dst[(x + i) * dst_stride_bytes + dst_offset_bytes] =
                lanes[i].round().clamp(0.0, 255.0) as u8;
        }
        x += 4;
    }
    while x < w {
        let mut acc = 0.0f32;
        for k in 0..kernel_size {
            acc += padded_row[x + k] * kernel[k];
        }
        dst[x * dst_stride_bytes + dst_offset_bytes] =
            acc.round().clamp(0.0, 255.0) as u8;
        x += 1;
    }
}

// ─── WASM SIMD128 f32x4 (native, was synthetic f64x2 pre-C2) ────

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn blur_row_h_wasm(
    padded_row: &[f32],
    kernel: &[f32],
    _radius: usize,
    w: usize,
    dst: &mut [u8],
    dst_offset_bytes: usize,
    dst_stride_bytes: usize,
) {
    use core::arch::wasm32::*;
    let kernel_size = kernel.len();
    let pr = padded_row.as_ptr();

    let mut x = 0;
    while x + 4 <= w {
        let mut acc = f32x4_splat(0.0);
        for k in 0..kernel_size {
            let kv = f32x4_splat(kernel[k]);
            let inv = v128_load(pr.add(x + k) as *const v128);
            let prod = f32x4_mul(inv, kv);
            acc = f32x4_add(acc, prod);
        }
        let l0 = f32x4_extract_lane::<0>(acc);
        let l1 = f32x4_extract_lane::<1>(acc);
        let l2 = f32x4_extract_lane::<2>(acc);
        let l3 = f32x4_extract_lane::<3>(acc);
        dst[x * dst_stride_bytes + dst_offset_bytes] = l0.round().clamp(0.0, 255.0) as u8;
        dst[(x + 1) * dst_stride_bytes + dst_offset_bytes] = l1.round().clamp(0.0, 255.0) as u8;
        dst[(x + 2) * dst_stride_bytes + dst_offset_bytes] = l2.round().clamp(0.0, 255.0) as u8;
        dst[(x + 3) * dst_stride_bytes + dst_offset_bytes] = l3.round().clamp(0.0, 255.0) as u8;
        x += 4;
    }
    while x < w {
        let mut acc = 0.0f32;
        for k in 0..kernel_size {
            acc += padded_row[x + k] * kernel[k];
        }
        dst[x * dst_stride_bytes + dst_offset_bytes] =
            acc.round().clamp(0.0, 255.0) as u8;
        x += 1;
    }
}

// ─── T2.11e — Stage 2 3×3 mean SIMD ─────────────────────────────

/// SIMD-aware sum of a 3×3 window in a padded planar f64 buffer
/// centered at (y, x) original coords (= (y, x) in padded coords
/// when the buffer was padded by 1 cell on each side).
///
/// Returns `(sum_x, sum_x1)` where lane 0 sums the 3×3 around
/// original pixel x, lane 1 sums around pixel x+1. Each lane uses
/// the same f64 add sequence as the scalar 3×3 sum — bit-identical
/// per lane.
///
/// `pw` is the padded plane width (= w + 2 for radius=1 padding).
/// Caller guarantees x+1 < w and y+1 < h (i.e. (y+dy, x+dx) for
/// dy/dx in 0..3 are in bounds in the padded buffer).
#[inline]
pub(super) fn sum_3x3_pair(plane: &[f64], pw: usize, x: usize, y: usize) -> (f64, f64) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        sum_3x3_pair_neon(plane, pw, x, y)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
    unsafe {
        sum_3x3_pair_sse(plane, pw, x, y)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        sum_3x3_pair_wasm(plane, pw, x, y)
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse4.1"),
        all(target_arch = "wasm32", target_feature = "simd128"),
    )))]
    sum_3x3_pair_scalar(plane, pw, x, y)
}

#[inline]
pub(super) fn sum_3x3_pair_scalar(plane: &[f64], pw: usize, x: usize, y: usize) -> (f64, f64) {
    let mut sum_x = 0.0f64;
    let mut sum_x1 = 0.0f64;
    for dy in 0..3 {
        let row_off = (y + dy) * pw;
        for dx in 0..3 {
            sum_x += plane[row_off + x + dx];
            sum_x1 += plane[row_off + x + 1 + dx];
        }
    }
    (sum_x, sum_x1)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_3x3_pair_neon(plane: &[f64], pw: usize, x: usize, y: usize) -> (f64, f64) {
    use core::arch::aarch64::*;
    let mut acc = vdupq_n_f64(0.0);
    let p = plane.as_ptr();
    for dy in 0..3 {
        let row_off = (y + dy) * pw;
        for dx in 0..3 {
            // Lane 0: plane[row_off + x + dx]
            // Lane 1: plane[row_off + x + dx + 1] = plane[row_off + (x+1) + dx]
            let v = vld1q_f64(p.add(row_off + x + dx));
            acc = vaddq_f64(acc, v);
        }
    }
    (vgetq_lane_f64::<0>(acc), vgetq_lane_f64::<1>(acc))
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
#[target_feature(enable = "sse2")]
unsafe fn sum_3x3_pair_sse(plane: &[f64], pw: usize, x: usize, y: usize) -> (f64, f64) {
    use core::arch::x86_64::*;
    let mut acc = _mm_setzero_pd();
    let p = plane.as_ptr();
    for dy in 0..3 {
        let row_off = (y + dy) * pw;
        for dx in 0..3 {
            let v = _mm_loadu_pd(p.add(row_off + x + dx));
            acc = _mm_add_pd(acc, v);
        }
    }
    let mut lanes = [0.0f64; 2];
    _mm_storeu_pd(lanes.as_mut_ptr(), acc);
    (lanes[0], lanes[1])
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn sum_3x3_pair_wasm(plane: &[f64], pw: usize, x: usize, y: usize) -> (f64, f64) {
    use core::arch::wasm32::*;
    let mut acc = f64x2_splat(0.0);
    let p = plane.as_ptr();
    for dy in 0..3 {
        let row_off = (y + dy) * pw;
        for dx in 0..3 {
            let v = v128_load(p.add(row_off + x + dx) as *const v128);
            acc = f64x2_add(acc, v);
        }
    }
    (f64x2_extract_lane::<0>(acc), f64x2_extract_lane::<1>(acc))
}

/// Helper: de-interleave + pad an RGB u8 buffer into 3 planar f64
/// buffers, clamp-to-edge replicated by 1 cell on each side. Each
/// output plane has dimensions `(w+2) × (h+2)` (laid out row-major).
///
/// Used by Stage 2 to amortize the boundary-clamp work outside the
/// per-pixel SIMD loop.
pub(super) fn deinterleave_pad1_to_planar_f64(
    pixels: &[u8],
    w: usize,
    h: usize,
) -> [Vec<f64>; 3] {
    let pw = w + 2;
    let ph = h + 2;
    let mut planes: [Vec<f64>; 3] = [
        vec![0.0; pw * ph],
        vec![0.0; pw * ph],
        vec![0.0; pw * ph],
    ];
    for py in 0..ph {
        let sy = py.saturating_sub(1).min(h - 1);
        for px in 0..pw {
            let sx = px.saturating_sub(1).min(w - 1);
            let src_idx = (sy * w + sx) * 3;
            for c in 0..3 {
                planes[c][py * pw + px] = pixels[src_idx + c] as f64;
            }
        }
    }
    planes
}

// ─── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run_test(seed: u32, w: usize, radius: usize) {
        let kernel_size = 2 * radius + 1;
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s as i32) as f32) / (i32::MAX as f32)
        };
        let padded: Vec<f32> = (0..w + 2 * radius).map(|_| next() * 255.0).collect();
        let kernel: Vec<f32> = (0..kernel_size).map(|_| next().abs()).collect();
        let stride = 3;

        let mut dst_scalar = vec![0u8; w * stride];
        blur_row_h_scalar(&padded, &kernel, radius, w, &mut dst_scalar, 0, stride);

        let mut dst_dispatch = vec![0u8; w * stride];
        blur_row_h(&padded, &kernel, radius, w, &mut dst_dispatch, 0, stride);

        // C2 contract: per-platform deterministic, not bit-exact
        // dispatch-vs-scalar (NEON FMA rounds differently than the
        // separate mul+add the scalar path uses). Allow up to ±1 u8
        // difference at any pixel — well under the optimizer's
        // existing 1-2 ULP boundary tolerance.
        for x in 0..w {
            let a = dst_scalar[x * stride] as i32;
            let b = dst_dispatch[x * stride] as i32;
            assert!(
                (a - b).abs() <= 1,
                "seed={seed} w={w} x={x}: scalar={a} dispatch={b} (diff > 1)"
            );
        }
    }

    #[test]
    fn blur_row_h_matches_scalar_random() {
        for seed in 0..200u32 {
            run_test(seed.wrapping_mul(2654435761), 32, 3);
            run_test(seed.wrapping_mul(2654435761), 33, 3);
            run_test(seed.wrapping_mul(2654435761), 64, 3);
            run_test(seed.wrapping_mul(2654435761), 4032, 3);
        }
    }

    fn sum_3x3_test(seed: u32, w: usize, h: usize) {
        let pw = w + 2;
        let ph = h + 2;
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s as i32) as f64) / (i32::MAX as f64) * 255.0
        };
        let plane: Vec<f64> = (0..pw * ph).map(|_| next()).collect();

        for y in 0..h {
            for x in 0..(w - 1) {
                let scalar = sum_3x3_pair_scalar(&plane, pw, x, y);
                let dispatched = sum_3x3_pair(&plane, pw, x, y);
                assert_eq!(scalar.0.to_bits(), dispatched.0.to_bits(),
                    "seed={seed} y={y} x={x} lane0 scalar={} dispatch={}",
                    scalar.0, dispatched.0);
                assert_eq!(scalar.1.to_bits(), dispatched.1.to_bits(),
                    "seed={seed} y={y} x={x} lane1 scalar={} dispatch={}",
                    scalar.1, dispatched.1);
            }
        }
    }

    #[test]
    fn sum_3x3_matches_scalar() {
        for seed in 0..50u32 {
            sum_3x3_test(seed.wrapping_mul(2654435761), 16, 16);
            sum_3x3_test(seed.wrapping_mul(2654435761), 17, 16);
            sum_3x3_test(seed.wrapping_mul(2654435761), 32, 32);
        }
    }
}
