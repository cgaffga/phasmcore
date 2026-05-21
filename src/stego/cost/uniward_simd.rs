// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! SIMD-accelerated kernels for J-UNIWARD filter operations (T2.1c+).
//!
//! Wraps arch-specific intrinsics with safe Rust fns. Falls back to
//! scalar on architectures without SIMD support.
//!
//! ## Scope
//!
//! - `saxpy_inner` — `scratch[x] += row_in[x] as f64 * coef` over a
//!   contiguous slice. Used in `filter_cols`'s k-outer / x-inner
//!   inner loop (T2.1b refactor). Per-x f64 mul + add identical to
//!   scalar; the only difference SIMD makes is computing 2 elements
//!   per iteration (NEON f64x2). **Byte-identical to scalar by
//!   construction** — no reduction-tree-order issue because each x
//!   has its own independent accumulator.
//!
//! - `dot_product_16_taps` — 16-tap f32×f64 → f64 dot product with
//!   **pinned-pairwise reduction tree** (T2.1c Path A). Used in
//!   `filter_rows`'s interior fast path AND boundary slabs.
//!   Pairwise tree:
//!   ```text
//!   sum = ((p0+p1 + p2+p3) + (p4+p5 + p6+p7))
//!       + ((p8+p9 + p10+p11) + (p12+p13 + p14+p15))
//!   ```
//!   identical on scalar / NEON / AVX2 / WASM because the tree
//!   shape doesn't depend on hardware. NEON uses `vpaddq_f64`
//!   (FADDP pairwise-add-within-vector) to traverse the same tree
//!   in 4 levels. **Cross-platform bit-wise identical by
//!   construction.**
//!
//! ### Path A wire-format cutover (2026-05-21)
//!
//! Original scalar `filter_rows` used a SEQUENTIAL accumulator
//! (`sum += p[0]; sum += p[1]; ...`). Path A switches scalar to
//! pairwise, which shifts UNIWARD cost values by 1-2 ULP vs v0.x.
//! Validated against the phasm.link never-expire corpus (10 stego
//! images, no-passphrase Fortress mode) — all 10 plaintexts decode
//! byte-identically pre- and post-Path-A → no boundary positions
//! flipped → wire-format-safe in practice for the deployed corpus.
//!
//! ## Per-arch status (2026-05-21)
//!
//! - **aarch64 NEON** ✅ shipped (T2.1c) — f64x2 SAXPY + 16-tap dot.
//! - **x86_64 AVX** ✅ shipped (T2.1d) — f64x4 SAXPY + 16-tap dot
//!   (mul in SIMD, reduce in scalar). AVX-1 not AVX2 (Sandybridge
//!   2011+ baseline). Runtime feature detection cached in OnceLock.
//!   Empirically validated via Rosetta 2 byte-equivalence tests.
//! - **WASM SIMD128** ✅ shipped (T2.1e) — f64x2 SAXPY + 16-tap dot
//!   (mul in SIMD via `f64x2_mul`, reduce in scalar). Active when
//!   compiled with `target-feature=+simd128`; falls back to scalar
//!   otherwise. Same SIMD-mul + scalar-reduce pattern as NEON/AVX.
//! - **Other archs** — scalar fallback (current behavior).

/// SAXPY-style inner loop: `scratch[x] += row_in[x] as f64 * coef`
/// for `x in 0..scratch.len()`. Slices must have equal length.
///
/// Dispatches to NEON on aarch64, AVX on x86_64 (if available),
/// scalar elsewhere. Byte-identical output across all dispatch
/// paths because each x has an independent f64 accumulator (no
/// reduction-tree-order issue).
#[inline]
pub fn saxpy_inner(scratch: &mut [f64], row_in: &[f32], coef: f64) {
    debug_assert_eq!(scratch.len(), row_in.len());

    #[cfg(target_arch = "aarch64")]
    {
        // Safety: aarch64 always has NEON per ARMv8 spec.
        unsafe { saxpy_inner_neon(scratch, row_in, coef) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_avx_available() {
            // Safety: runtime feature check above.
            unsafe { saxpy_inner_avx(scratch, row_in, coef) }
        } else {
            saxpy_inner_scalar(scratch, row_in, coef)
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        // Safety: simd128 enabled at compile time per cfg gate.
        unsafe { saxpy_inner_wasm(scratch, row_in, coef) }
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128"),
    )))]
    saxpy_inner_scalar(scratch, row_in, coef);
}

/// Cached runtime AVX feature detection for x86_64. Result memoized
/// in a OnceLock — one atomic load per kernel call after warmup.
#[cfg(target_arch = "x86_64")]
fn is_avx_available() -> bool {
    use std::sync::OnceLock;
    static AVX: OnceLock<bool> = OnceLock::new();
    *AVX.get_or_init(|| std::is_x86_feature_detected!("avx"))
}

/// Scalar reference implementation. Always available; used directly
/// on non-aarch64 targets and as the byte-equivalence reference in
/// tests.
#[inline]
fn saxpy_inner_scalar(scratch: &mut [f64], row_in: &[f32], coef: f64) {
    for x in 0..scratch.len() {
        scratch[x] += row_in[x] as f64 * coef;
    }
}

/// aarch64 NEON f64x2 SAXPY. Processes 2 elements per iteration.
/// Same per-x f64 mul + add as scalar — no FMA, no reduction tree.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn saxpy_inner_neon(scratch: &mut [f64], row_in: &[f32], coef: f64) {
    use core::arch::aarch64::*;

    let n = scratch.len();
    let coef_v = vdupq_n_f64(coef);

    let scratch_ptr = scratch.as_mut_ptr();
    let row_ptr = row_in.as_ptr();

    let mut x = 0;
    // Vector loop: 2 elements per iteration.
    while x + 2 <= n {
        // Load 2 f32s, convert to f64x2 (widening).
        let r_f32x2 = vld1_f32(row_ptr.add(x));
        let r_f64x2 = vcvt_f64_f32(r_f32x2);

        // Multiply by broadcast coef.
        let prod = vmulq_f64(r_f64x2, coef_v);

        // Load scratch[x..x+2], add, store.
        let s = vld1q_f64(scratch_ptr.add(x));
        let new_v = vaddq_f64(s, prod);
        vst1q_f64(scratch_ptr.add(x), new_v);

        x += 2;
    }

    // Tail: handle the 0 or 1 remaining elements scalar.
    while x < n {
        *scratch_ptr.add(x) += *row_ptr.add(x) as f64 * coef;
        x += 1;
    }
}

/// x86_64 AVX f64x4 SAXPY. Processes 4 elements per iteration.
/// Same per-x f64 mul + add as scalar — no FMA, no reduction tree.
/// Runtime gated on `is_avx_available`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn saxpy_inner_avx(scratch: &mut [f64], row_in: &[f32], coef: f64) {
    use core::arch::x86_64::*;

    let n = scratch.len();
    let coef_v = _mm256_set1_pd(coef);

    let scratch_ptr = scratch.as_mut_ptr();
    let row_ptr = row_in.as_ptr();

    let mut x = 0;
    // Vector loop: 4 elements per iteration (f64x4).
    while x + 4 <= n {
        // Load 4 f32s (__m128 = f32x4), widen to f64x4.
        let r_f32x4 = _mm_loadu_ps(row_ptr.add(x));
        let r_f64x4 = _mm256_cvtps_pd(r_f32x4);

        // Multiply by broadcast coef (separate mul — no FMA).
        let prod = _mm256_mul_pd(r_f64x4, coef_v);

        // Load scratch[x..x+4], add (separate, no FMA), store.
        let s = _mm256_loadu_pd(scratch_ptr.add(x));
        let new_v = _mm256_add_pd(s, prod);
        _mm256_storeu_pd(scratch_ptr.add(x), new_v);

        x += 4;
    }

    // Tail: handle 0-3 remaining elements scalar.
    while x < n {
        *scratch_ptr.add(x) += *row_ptr.add(x) as f64 * coef;
        x += 1;
    }
}

// ─── T2.1c Path A — 16-tap dot product with pinned-pairwise tree ───

/// 16-tap dot product: `sum_{k=0..16} (row_in[k] as f64) * filter[k]`
/// using a **fixed pairwise reduction tree** so the result is bit-
/// wise identical across all platforms (NEON / AVX2 / WASM / scalar
/// fallback).
///
/// `row_in` must have at least 16 elements; only the first 16 are
/// read. Tap convention: `row_in[k]` pairs with `filter[k]` for
/// k = 0..16.
#[inline]
pub fn dot_product_16_taps(row_in: &[f32], filter: &[f64; 16]) -> f64 {
    debug_assert!(row_in.len() >= 16);

    #[cfg(target_arch = "aarch64")]
    {
        // Safety: aarch64 always has NEON per ARMv8 spec.
        unsafe { dot_product_16_taps_neon(row_in, filter) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_avx_available() {
            // Safety: runtime feature check above.
            unsafe { dot_product_16_taps_avx(row_in, filter) }
        } else {
            dot_product_16_taps_scalar(row_in, filter)
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        // Safety: simd128 enabled at compile time per cfg gate.
        unsafe { dot_product_16_taps_wasm(row_in, filter) }
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128"),
    )))]
    dot_product_16_taps_scalar(row_in, filter)
}

/// x86_64 AVX implementation. SIMD f64x4 muls + scalar pairwise
/// reduce. The pairwise tree is identical scalar code to
/// `dot_product_16_taps_scalar` — by construction the f64 output
/// is bit-identical across scalar / NEON / AVX paths. The only
/// SIMD operations are the 16 multiplications (4 × `_mm256_mul_pd`
/// = 4 × 4-wide); the 15 additions of the pairwise tree happen
/// in scalar with the same source code.
///
/// Uses AVX (1) only — `_mm256_mul_pd` was introduced with
/// Sandybridge (2011); no AVX2 / FMA required.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dot_product_16_taps_avx(row_in: &[f32], filter: &[f64; 16]) -> f64 {
    use core::arch::x86_64::*;

    debug_assert!(row_in.len() >= 16);

    let row_ptr = row_in.as_ptr();
    let filter_ptr = filter.as_ptr();

    // Load 16 f32s as 4 f32x4 (__m128).
    let r0 = _mm_loadu_ps(row_ptr);
    let r1 = _mm_loadu_ps(row_ptr.add(4));
    let r2 = _mm_loadu_ps(row_ptr.add(8));
    let r3 = _mm_loadu_ps(row_ptr.add(12));

    // Widen f32x4 → f64x4. Lane 0..3 of v0 = row[0..3], etc.
    let v0 = _mm256_cvtps_pd(r0); // {row[0], row[1], row[2], row[3]}
    let v1 = _mm256_cvtps_pd(r1); // {row[4]..row[7]}
    let v2 = _mm256_cvtps_pd(r2); // {row[8]..row[11]}
    let v3 = _mm256_cvtps_pd(r3); // {row[12]..row[15]}

    // Filter as 4 f64x4.
    let f0 = _mm256_loadu_pd(filter_ptr);
    let f1 = _mm256_loadu_pd(filter_ptr.add(4));
    let f2 = _mm256_loadu_pd(filter_ptr.add(8));
    let f3 = _mm256_loadu_pd(filter_ptr.add(12));

    // Multiply (separate mul — no FMA, matches scalar semantics).
    let p0_p3 = _mm256_mul_pd(v0, f0);
    let p4_p7 = _mm256_mul_pd(v1, f1);
    let p8_p11 = _mm256_mul_pd(v2, f2);
    let p12_p15 = _mm256_mul_pd(v3, f3);

    // Extract products to scalar [f64; 16].
    let mut products = [0.0f64; 16];
    _mm256_storeu_pd(products.as_mut_ptr(), p0_p3);
    _mm256_storeu_pd(products.as_mut_ptr().add(4), p4_p7);
    _mm256_storeu_pd(products.as_mut_ptr().add(8), p8_p11);
    _mm256_storeu_pd(products.as_mut_ptr().add(12), p12_p15);

    // Scalar pairwise reduce — IDENTICAL source to the scalar
    // fallback fn. Guarantees bit-equivalence at the reduce stage.
    // (SIMD multiplies above produce f64 results uniquely
    // specified by IEEE 754 for given inputs, matching scalar mul.)
    let s0 = products[0] + products[1];
    let s1 = products[2] + products[3];
    let s2 = products[4] + products[5];
    let s3 = products[6] + products[7];
    let s4 = products[8] + products[9];
    let s5 = products[10] + products[11];
    let s6 = products[12] + products[13];
    let s7 = products[14] + products[15];

    let t0 = s0 + s1;
    let t1 = s2 + s3;
    let t2 = s4 + s5;
    let t3 = s6 + s7;

    let u0 = t0 + t1;
    let u1 = t2 + t3;

    u0 + u1
}

/// Scalar reference implementation. Always available; used directly
/// on non-aarch64 targets and as the byte-equivalence reference in
/// tests. **Pairwise tree** — same shape as the NEON path.
#[inline]
fn dot_product_16_taps_scalar(row_in: &[f32], filter: &[f64; 16]) -> f64 {
    // 16 products
    let p0 = row_in[0] as f64 * filter[0];
    let p1 = row_in[1] as f64 * filter[1];
    let p2 = row_in[2] as f64 * filter[2];
    let p3 = row_in[3] as f64 * filter[3];
    let p4 = row_in[4] as f64 * filter[4];
    let p5 = row_in[5] as f64 * filter[5];
    let p6 = row_in[6] as f64 * filter[6];
    let p7 = row_in[7] as f64 * filter[7];
    let p8 = row_in[8] as f64 * filter[8];
    let p9 = row_in[9] as f64 * filter[9];
    let p10 = row_in[10] as f64 * filter[10];
    let p11 = row_in[11] as f64 * filter[11];
    let p12 = row_in[12] as f64 * filter[12];
    let p13 = row_in[13] as f64 * filter[13];
    let p14 = row_in[14] as f64 * filter[14];
    let p15 = row_in[15] as f64 * filter[15];

    // L1: 16 → 8 pairs
    let s0 = p0 + p1;
    let s1 = p2 + p3;
    let s2 = p4 + p5;
    let s3 = p6 + p7;
    let s4 = p8 + p9;
    let s5 = p10 + p11;
    let s6 = p12 + p13;
    let s7 = p14 + p15;

    // L2: 8 → 4
    let t0 = s0 + s1;
    let t1 = s2 + s3;
    let t2 = s4 + s5;
    let t3 = s6 + s7;

    // L3: 4 → 2
    let u0 = t0 + t1;
    let u1 = t2 + t3;

    // L4: 2 → 1
    u0 + u1
}

/// aarch64 NEON implementation. Same tree as scalar: 4 levels of
/// pairwise add via `vpaddq_f64` (FADDP), then `vaddvq_f64` for the
/// final 2 → 1 horizontal reduce.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_16_taps_neon(row_in: &[f32], filter: &[f64; 16]) -> f64 {
    use core::arch::aarch64::*;

    debug_assert!(row_in.len() >= 16);

    let row_ptr = row_in.as_ptr();
    let filter_ptr = filter.as_ptr();

    // Load 16 f32s as 4 f32x4.
    let r0q = vld1q_f32(row_ptr);
    let r1q = vld1q_f32(row_ptr.add(4));
    let r2q = vld1q_f32(row_ptr.add(8));
    let r3q = vld1q_f32(row_ptr.add(12));

    // Widen f32x4 → 2 × f64x2 (low half + high half).
    // Each f64x2 holds two consecutive widened f32 values.
    let v0 = vcvt_f64_f32(vget_low_f32(r0q)); // {row[0], row[1]}
    let v1 = vcvt_high_f64_f32(r0q); // {row[2], row[3]}
    let v2 = vcvt_f64_f32(vget_low_f32(r1q)); // {row[4], row[5]}
    let v3 = vcvt_high_f64_f32(r1q); // {row[6], row[7]}
    let v4 = vcvt_f64_f32(vget_low_f32(r2q)); // {row[8], row[9]}
    let v5 = vcvt_high_f64_f32(r2q); // {row[10], row[11]}
    let v6 = vcvt_f64_f32(vget_low_f32(r3q)); // {row[12], row[13]}
    let v7 = vcvt_high_f64_f32(r3q); // {row[14], row[15]}

    // Filter as 8 f64x2.
    let f0 = vld1q_f64(filter_ptr);
    let f1 = vld1q_f64(filter_ptr.add(2));
    let f2 = vld1q_f64(filter_ptr.add(4));
    let f3 = vld1q_f64(filter_ptr.add(6));
    let f4 = vld1q_f64(filter_ptr.add(8));
    let f5 = vld1q_f64(filter_ptr.add(10));
    let f6 = vld1q_f64(filter_ptr.add(12));
    let f7 = vld1q_f64(filter_ptr.add(14));

    // Multiply (separate vmulq — no FMA, matches scalar two-rounding semantics).
    let p0 = vmulq_f64(v0, f0); // {p0, p1}
    let p1 = vmulq_f64(v1, f1); // {p2, p3}
    let p2 = vmulq_f64(v2, f2); // {p4, p5}
    let p3 = vmulq_f64(v3, f3); // {p6, p7}
    let p4 = vmulq_f64(v4, f4); // {p8, p9}
    let p5 = vmulq_f64(v5, f5); // {p10, p11}
    let p6 = vmulq_f64(v6, f6); // {p12, p13}
    let p7 = vmulq_f64(v7, f7); // {p14, p15}

    // L1: 8 vectors → 4. vpaddq_f64(a, b) = {a[0]+a[1], b[0]+b[1]}.
    let s01 = vpaddq_f64(p0, p1); // {s0, s1} = {p0+p1, p2+p3}
    let s23 = vpaddq_f64(p2, p3); // {s2, s3} = {p4+p5, p6+p7}
    let s45 = vpaddq_f64(p4, p5); // {s4, s5} = {p8+p9, p10+p11}
    let s67 = vpaddq_f64(p6, p7); // {s6, s7} = {p12+p13, p14+p15}

    // L2: 4 → 2.
    let t01 = vpaddq_f64(s01, s23); // {t0, t1}
    let t23 = vpaddq_f64(s45, s67); // {t2, t3}

    // L3: 2 → 1.
    let u01 = vpaddq_f64(t01, t23); // {u0, u1}

    // L4: horizontal final.
    vaddvq_f64(u01)
}

// ─── T2.1e WASM SIMD128 implementations ─────────────────────────

/// WASM SIMD128 f64x2 SAXPY. Processes 4 elements per iteration
/// (two f64x2 vectors per loop body). Same per-x f64 mul + add as
/// scalar — no FMA, no reduction tree.
///
/// Requires `target-feature=+simd128` at compile time.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn saxpy_inner_wasm(scratch: &mut [f64], row_in: &[f32], coef: f64) {
    use core::arch::wasm32::*;

    let n = scratch.len();
    let coef_v = f64x2_splat(coef);

    let scratch_ptr = scratch.as_mut_ptr();
    let row_ptr = row_in.as_ptr();

    let mut x = 0;
    // Vector loop: 4 elements per iteration (2 × f64x2).
    while x + 4 <= n {
        // Load 4 f32s as f32x4 v128.
        let r_q = v128_load(row_ptr.add(x) as *const v128);

        // Widen low half (lanes 0, 1) to f64x2.
        let r_lo = f64x2_promote_low_f32x4(r_q);

        // Shuffle high half (lanes 2, 3) to low positions, then widen.
        let r_hi_q = i32x4_shuffle::<2, 3, 2, 3>(r_q, r_q);
        let r_hi = f64x2_promote_low_f32x4(r_hi_q);

        // Multiply both halves by coef (separate mul — no FMA).
        let prod_lo = f64x2_mul(r_lo, coef_v);
        let prod_hi = f64x2_mul(r_hi, coef_v);

        // Load scratch[x..x+2] and scratch[x+2..x+4], add, store.
        let s_lo = v128_load(scratch_ptr.add(x) as *const v128);
        let s_hi = v128_load(scratch_ptr.add(x + 2) as *const v128);

        let new_lo = f64x2_add(s_lo, prod_lo);
        let new_hi = f64x2_add(s_hi, prod_hi);

        v128_store(scratch_ptr.add(x) as *mut v128, new_lo);
        v128_store(scratch_ptr.add(x + 2) as *mut v128, new_hi);

        x += 4;
    }

    // Tail: 0-3 elements scalar.
    while x < n {
        *scratch_ptr.add(x) += *row_ptr.add(x) as f64 * coef;
        x += 1;
    }
}

/// WASM SIMD128 16-tap dot product. SIMD f64x2 muls (8 vectors,
/// matching NEON's layout) + scalar pairwise reduce identical to
/// the scalar fallback. Bit-identical to all other arch paths.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot_product_16_taps_wasm(row_in: &[f32], filter: &[f64; 16]) -> f64 {
    use core::arch::wasm32::*;

    debug_assert!(row_in.len() >= 16);

    let row_ptr = row_in.as_ptr();
    let filter_ptr = filter.as_ptr();

    // Load 16 f32s as 4 v128 (f32x4 each).
    let r0q = v128_load(row_ptr as *const v128);
    let r1q = v128_load(row_ptr.add(4) as *const v128);
    let r2q = v128_load(row_ptr.add(8) as *const v128);
    let r3q = v128_load(row_ptr.add(12) as *const v128);

    // Promote each f32x4 to 2 f64x2 (low half direct + high half
    // via shuffle-to-low). 8 f64x2 vectors hold all 16 widened f64s.
    let v0 = f64x2_promote_low_f32x4(r0q); // {row[0], row[1]}
    let v1 = f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(r0q, r0q)); // {row[2], row[3]}
    let v2 = f64x2_promote_low_f32x4(r1q); // {row[4], row[5]}
    let v3 = f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(r1q, r1q)); // {row[6], row[7]}
    let v4 = f64x2_promote_low_f32x4(r2q); // {row[8], row[9]}
    let v5 = f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(r2q, r2q)); // {row[10], row[11]}
    let v6 = f64x2_promote_low_f32x4(r3q); // {row[12], row[13]}
    let v7 = f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(r3q, r3q)); // {row[14], row[15]}

    // Filter as 8 f64x2.
    let f0 = v128_load(filter_ptr as *const v128);
    let f1 = v128_load(filter_ptr.add(2) as *const v128);
    let f2 = v128_load(filter_ptr.add(4) as *const v128);
    let f3 = v128_load(filter_ptr.add(6) as *const v128);
    let f4 = v128_load(filter_ptr.add(8) as *const v128);
    let f5 = v128_load(filter_ptr.add(10) as *const v128);
    let f6 = v128_load(filter_ptr.add(12) as *const v128);
    let f7 = v128_load(filter_ptr.add(14) as *const v128);

    // Multiply (separate, no FMA).
    let p0 = f64x2_mul(v0, f0);
    let p1 = f64x2_mul(v1, f1);
    let p2 = f64x2_mul(v2, f2);
    let p3 = f64x2_mul(v3, f3);
    let p4 = f64x2_mul(v4, f4);
    let p5 = f64x2_mul(v5, f5);
    let p6 = f64x2_mul(v6, f6);
    let p7 = f64x2_mul(v7, f7);

    // Extract products to scalar [f64; 16].
    let mut products = [0.0f64; 16];
    v128_store(products.as_mut_ptr() as *mut v128, p0);
    v128_store(products.as_mut_ptr().add(2) as *mut v128, p1);
    v128_store(products.as_mut_ptr().add(4) as *mut v128, p2);
    v128_store(products.as_mut_ptr().add(6) as *mut v128, p3);
    v128_store(products.as_mut_ptr().add(8) as *mut v128, p4);
    v128_store(products.as_mut_ptr().add(10) as *mut v128, p5);
    v128_store(products.as_mut_ptr().add(12) as *mut v128, p6);
    v128_store(products.as_mut_ptr().add(14) as *mut v128, p7);

    // Scalar pairwise reduce — IDENTICAL source to scalar fallback.
    let s0 = products[0] + products[1];
    let s1 = products[2] + products[3];
    let s2 = products[4] + products[5];
    let s3 = products[6] + products[7];
    let s4 = products[8] + products[9];
    let s5 = products[10] + products[11];
    let s6 = products[12] + products[13];
    let s7 = products[14] + products[15];

    let t0 = s0 + s1;
    let t1 = s2 + s3;
    let t2 = s4 + s5;
    let t3 = s6 + s7;

    let u0 = t0 + t1;
    let u1 = t2 + t3;

    u0 + u1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saxpy_dispatch_matches_scalar_for_even_length() {
        // Deterministic input; compare dispatched (NEON on aarch64,
        // scalar elsewhere) against the scalar reference at every
        // index. Byte-equivalence required.
        let n = 256;
        let row_in: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 0.128).collect();
        let coef = 0.0314159_f64;
        let mut scratch_ref = vec![1.5f64; n];
        let mut scratch_dispatched = vec![1.5f64; n];

        saxpy_inner_scalar(&mut scratch_ref, &row_in, coef);
        saxpy_inner(&mut scratch_dispatched, &row_in, coef);

        for i in 0..n {
            assert_eq!(
                scratch_ref[i].to_bits(),
                scratch_dispatched[i].to_bits(),
                "byte-equivalence violated at index {i}",
            );
        }
    }

    #[test]
    fn saxpy_dispatch_matches_scalar_for_odd_length() {
        // Odd length exercises the tail path on NEON.
        let n = 257;
        let row_in: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002).collect();
        let coef = -0.7;
        let mut scratch_ref = vec![-2.25f64; n];
        let mut scratch_dispatched = vec![-2.25f64; n];

        saxpy_inner_scalar(&mut scratch_ref, &row_in, coef);
        saxpy_inner(&mut scratch_dispatched, &row_in, coef);

        for i in 0..n {
            assert_eq!(
                scratch_ref[i].to_bits(),
                scratch_dispatched[i].to_bits(),
                "byte-equivalence violated at index {i} (tail)",
            );
        }
    }

    #[test]
    fn saxpy_empty_is_noop() {
        let mut scratch: Vec<f64> = vec![];
        let row_in: Vec<f32> = vec![];
        saxpy_inner(&mut scratch, &row_in, 0.5);
        assert!(scratch.is_empty());
    }

    #[test]
    fn saxpy_zero_coef_unchanged() {
        let n = 16;
        let row_in: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut scratch = vec![3.14f64; n];
        let before = scratch.clone();
        saxpy_inner(&mut scratch, &row_in, 0.0);
        // scratch[x] += row_in[x] * 0.0 = scratch[x] + 0.0 = scratch[x]
        for i in 0..n {
            assert_eq!(scratch[i].to_bits(), before[i].to_bits(),
                "zero coef should leave scratch unchanged at {i}");
        }
    }

    // ── dot_product_16_taps tests ──

    #[test]
    fn dot_product_16_known_value() {
        // Identity-ish check: filter = [1, 0, 0, ..., 0], row = [42, 0, 0, ...]
        // → expected = 42.0.
        let mut filter = [0.0f64; 16];
        filter[0] = 1.0;
        let mut row = vec![0.0f32; 16];
        row[0] = 42.0;
        let r = dot_product_16_taps(&row, &filter);
        assert_eq!(r.to_bits(), 42.0f64.to_bits());
    }

    #[test]
    fn dot_product_16_dispatch_matches_scalar() {
        // Pseudo-random inputs; compare dispatched (NEON on aarch64,
        // scalar elsewhere) against the scalar reference.
        let mut filter = [0.0f64; 16];
        for k in 0..16 {
            filter[k] = ((k as f64 * 0.31415).sin() - 0.5) * 1.7;
        }
        // 20 different starting rows
        for start in 0..20 {
            let row: Vec<f32> = (start..start + 16)
                .map(|i| ((i as f32 * 0.013).cos() + 0.2) * 11.7)
                .collect();
            let r_scalar = dot_product_16_taps_scalar(&row, &filter);
            let r_dispatched = dot_product_16_taps(&row, &filter);
            assert_eq!(
                r_scalar.to_bits(),
                r_dispatched.to_bits(),
                "byte-equivalence violated at start={start}",
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dot_product_16_neon_matches_scalar() {
        // Direct scalar vs NEON byte-exact check on aarch64.
        let mut filter = [0.0f64; 16];
        for k in 0..16 {
            filter[k] = ((k as f64 * 0.7) - 5.5).sin();
        }
        for start in 0..50 {
            let row: Vec<f32> = (start..start + 16)
                .map(|i| (i as f32) * 0.13 + 0.7)
                .collect();
            let r_scalar = dot_product_16_taps_scalar(&row, &filter);
            let r_neon = unsafe { dot_product_16_taps_neon(&row, &filter) };
            assert_eq!(
                r_scalar.to_bits(),
                r_neon.to_bits(),
                "scalar vs NEON byte-mismatch at start={start}",
            );
        }
    }

    // ── x86_64 AVX byte-equivalence tests ──

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn saxpy_avx_matches_scalar_direct() {
        if !is_avx_available() {
            eprintln!("skipping: CPU lacks AVX (Sandybridge+ required)");
            return;
        }
        let n = 1024;
        let row_in: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.0017).sin() * 100.0)
            .collect();
        let coef = 0.123456789_f64;
        let mut scratch_scalar = vec![42.0f64; n];
        let mut scratch_avx = vec![42.0f64; n];

        saxpy_inner_scalar(&mut scratch_scalar, &row_in, coef);
        unsafe {
            saxpy_inner_avx(&mut scratch_avx, &row_in, coef);
        }

        for i in 0..n {
            assert_eq!(
                scratch_scalar[i].to_bits(),
                scratch_avx[i].to_bits(),
                "scalar vs AVX byte-mismatch at index {i}",
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn saxpy_avx_tail_correctness() {
        // Lengths NOT divisible by 4 exercise the AVX tail path.
        if !is_avx_available() {
            return;
        }
        for n in [1, 2, 3, 5, 7, 13, 17, 100, 101, 102, 103] {
            let row_in: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
            let coef = 0.7_f64;
            let mut scratch_scalar = vec![1.0f64; n];
            let mut scratch_avx = vec![1.0f64; n];

            saxpy_inner_scalar(&mut scratch_scalar, &row_in, coef);
            unsafe {
                saxpy_inner_avx(&mut scratch_avx, &row_in, coef);
            }

            for i in 0..n {
                assert_eq!(
                    scratch_scalar[i].to_bits(),
                    scratch_avx[i].to_bits(),
                    "tail mismatch at n={n} index={i}",
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dot_product_16_avx_matches_scalar() {
        if !is_avx_available() {
            return;
        }
        let mut filter = [0.0f64; 16];
        for k in 0..16 {
            filter[k] = ((k as f64 * 0.7) - 5.5).sin();
        }
        for start in 0..50 {
            let row: Vec<f32> = (start..start + 16)
                .map(|i| (i as f32) * 0.13 + 0.7)
                .collect();
            let r_scalar = dot_product_16_taps_scalar(&row, &filter);
            let r_avx = unsafe { dot_product_16_taps_avx(&row, &filter) };
            assert_eq!(
                r_scalar.to_bits(),
                r_avx.to_bits(),
                "scalar vs AVX byte-mismatch at start={start}",
            );
        }
    }

    /// Explicit aarch64 NEON test: directly compare scalar reference
    /// against NEON impl (both available on this target).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn saxpy_neon_matches_scalar_direct() {
        let n = 1024;
        let row_in: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.0017).sin() * 100.0)
            .collect();
        let coef = 0.123456789_f64;
        let mut scratch_scalar = vec![42.0f64; n];
        let mut scratch_neon = vec![42.0f64; n];

        saxpy_inner_scalar(&mut scratch_scalar, &row_in, coef);
        unsafe {
            saxpy_inner_neon(&mut scratch_neon, &row_in, coef);
        }

        for i in 0..n {
            assert_eq!(
                scratch_scalar[i].to_bits(),
                scratch_neon[i].to_bits(),
                "scalar vs NEON byte-mismatch at index {i}",
            );
        }
    }
}
