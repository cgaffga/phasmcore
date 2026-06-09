// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! SIMD-accelerated radix-2 FFT butterfly chunks.
//!
//! Path A pattern: each SIMD lane computes one butterfly's complex
//! mul + add/sub. No reduction across lanes, no FMA — every operation
//! is a single IEEE 754 f32 op whose result is uniquely specified by
//! its inputs. Output is bit-identical across scalar / NEON / SSE /
//! WASM SIMD128 paths by construction.
//!
//! ## Butterfly math
//!
//! For one butterfly at indices (k, k+half) with twiddle w:
//! ```text
//!   v.re = b.re * w.re - b.im * w.im
//!   v.im = b.re * w.im + b.im * w.re
//!   u' = u + v
//!   b' = u - v
//! ```
//! where u = data[start+k], b = data[start+k+half], in interleaved
//! (re, im) memory layout.
//!
//! ## Chunk-of-4 layout
//!
//! We process 4 butterflies per SIMD chunk (k, k+1, k+2, k+3). Lane
//! semantics for each SIMD register:
//!   lane 0 → butterfly k     (data[start+k])
//!   lane 1 → butterfly k+1   (data[start+k+1])
//!   lane 2 → butterfly k+2   (data[start+k+2])
//!   lane 3 → butterfly k+3   (data[start+k+3])
//!
//! Each lane carries independent (re or im) of one butterfly. Math
//! is lane-pure: no shuffle/reduce after the muladd chain.
//!
//! ## Memory access pattern
//!
//! Input/output is interleaved Complex32 = (re, im) pairs. Loading
//! 4 Complex32 = 8 f32 = 2 × 128-bit vectors. We deinterleave to
//! separate re_vec / im_vec, run lane-pure SoA math, then re-
//! interleave to write back.
//!
//! Deinterleave intrinsics:
//!   - NEON: vuzp1q_f32 (even) / vuzp2q_f32 (odd)
//!   - SSE:  _mm_shuffle_ps with mask 0x88 (even) / 0xDD (odd)
//!   - WASM: i32x4_shuffle::<0, 2, 4, 6> / <1, 3, 5, 7>
//!
//! Interleave intrinsics:
//!   - NEON: vzip1q_f32 (low) / vzip2q_f32 (high)
//!   - SSE:  _mm_unpacklo_ps / _mm_unpackhi_ps
//!   - WASM: i32x4_shuffle::<0, 4, 1, 5> / <2, 6, 3, 7>
//!
//! ## Per-arch status (2026-05-21)
//!
//! - **aarch64 NEON** ✅ shipped — f32x4
//! - **x86_64 SSE** ✅ shipped — f32x4 via SSE (no runtime check;
//!   SSE4.1 is the x86_64 compile baseline per `.cargo/config.toml`)
//! - **WASM SIMD128** ✅ shipped — f32x4 (requires
//!   `target-feature=+simd128` at compile time)
//! - **Other archs** — scalar fallback
//!
//! Empirical validation:
//!   - aarch64: direct scalar-vs-NEON byte-equivalence tests
//!   - x86_64: same tests run under Rosetta 2 (macOS 26.5)
//!   - WASM: V8 (Node.js v25) smoke test on phasm.link corpus

use num_complex::Complex32;

/// Process 4 consecutive butterflies at indices `(start+k), (start+k+1),
/// (start+k+2), (start+k+3)` paired with `(start+k+half), ...,
/// (start+k+half+3)`. `twiddles` holds the 4 twiddle factors for k..k+4.
///
/// Path A invariant: bit-identical output across all dispatch paths.
#[inline]
pub fn butterfly_chunk_4(
    data: &mut [Complex32],
    start: usize,
    k: usize,
    half: usize,
    twiddles: &[Complex32],
) {
    debug_assert!(twiddles.len() >= 4);
    debug_assert!(start + k + 3 < data.len());
    debug_assert!(start + k + half + 3 < data.len());

    #[cfg(target_arch = "aarch64")]
    // Safety: aarch64 always has NEON per ARMv8 spec.
    unsafe {
        butterfly_chunk_4_neon(data, start, k, half, twiddles)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
    // Safety: SSE4.1 is the x86_64 compile baseline in this project
    // (see `.cargo/config.toml`). No runtime check needed.
    unsafe {
        butterfly_chunk_4_sse(data, start, k, half, twiddles)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // Safety: simd128 enabled at compile time per cfg gate.
    unsafe {
        butterfly_chunk_4_wasm(data, start, k, half, twiddles)
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse4.1"),
        all(target_arch = "wasm32", target_feature = "simd128"),
    )))]
    butterfly_chunk_4_scalar(data, start, k, half, twiddles);
}

/// Scalar reference implementation. Always available; used as the
/// byte-equivalence baseline in tests AND as the runtime fallback
/// on architectures without a SIMD path.
#[inline]
pub(super) fn butterfly_chunk_4_scalar(
    data: &mut [Complex32],
    start: usize,
    k: usize,
    half: usize,
    twiddles: &[Complex32],
) {
    for i in 0..4 {
        let w = twiddles[i];
        let u = data[start + k + i];
        let v = data[start + k + half + i] * w;
        data[start + k + i] = u + v;
        data[start + k + half + i] = u - v;
    }
}

// ─── aarch64 NEON ───────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn butterfly_chunk_4_neon(
    data: &mut [Complex32],
    start: usize,
    k: usize,
    half: usize,
    twiddles: &[Complex32],
) {
    use core::arch::aarch64::*;

    // Pointer to start + k (u values, 4 Complex32 = 8 f32 = 2 × f32x4).
    let u_ptr = data.as_mut_ptr().add(start + k) as *const f32;
    // Pointer to start + k + half (b values).
    let b_ptr = data.as_mut_ptr().add(start + k + half) as *const f32;
    // Pointer to twiddles[0..4].
    let w_ptr = twiddles.as_ptr() as *const f32;

    // Load 2 × f32x4 each for u, b, w (interleaved re, im).
    let u_lo = vld1q_f32(u_ptr); // [u0.re, u0.im, u1.re, u1.im]
    let u_hi = vld1q_f32(u_ptr.add(4)); // [u2.re, u2.im, u3.re, u3.im]
    let b_lo = vld1q_f32(b_ptr); // [b0.re, b0.im, b1.re, b1.im]
    let b_hi = vld1q_f32(b_ptr.add(4));
    let w_lo = vld1q_f32(w_ptr); // [w0.re, w0.im, w1.re, w1.im]
    let w_hi = vld1q_f32(w_ptr.add(4));

    // Deinterleave: re lanes (even) and im lanes (odd).
    let u_re = vuzp1q_f32(u_lo, u_hi); // [u0.re, u1.re, u2.re, u3.re]
    let u_im = vuzp2q_f32(u_lo, u_hi); // [u0.im, u1.im, u2.im, u3.im]
    let b_re = vuzp1q_f32(b_lo, b_hi);
    let b_im = vuzp2q_f32(b_lo, b_hi);
    let w_re = vuzp1q_f32(w_lo, w_hi);
    let w_im = vuzp2q_f32(w_lo, w_hi);

    // Complex multiply v = b * w (no FMA — separate mul + add/sub).
    // v.re = b.re * w.re - b.im * w.im
    // v.im = b.re * w.im + b.im * w.re
    let bw_re_re = vmulq_f32(b_re, w_re);
    let bw_im_im = vmulq_f32(b_im, w_im);
    let bw_re_im = vmulq_f32(b_re, w_im);
    let bw_im_re = vmulq_f32(b_im, w_re);

    let v_re = vsubq_f32(bw_re_re, bw_im_im);
    let v_im = vaddq_f32(bw_re_im, bw_im_re);

    // Butterfly: new_u = u + v, new_b = u - v.
    let new_u_re = vaddq_f32(u_re, v_re);
    let new_u_im = vaddq_f32(u_im, v_im);
    let new_b_re = vsubq_f32(u_re, v_re);
    let new_b_im = vsubq_f32(u_im, v_im);

    // Re-interleave for store: zip pairs (re, im) lane-wise.
    let new_u_lo = vzip1q_f32(new_u_re, new_u_im); // [re0, im0, re1, im1]
    let new_u_hi = vzip2q_f32(new_u_re, new_u_im); // [re2, im2, re3, im3]
    let new_b_lo = vzip1q_f32(new_b_re, new_b_im);
    let new_b_hi = vzip2q_f32(new_b_re, new_b_im);

    let u_out = data.as_mut_ptr().add(start + k) as *mut f32;
    let b_out = data.as_mut_ptr().add(start + k + half) as *mut f32;
    vst1q_f32(u_out, new_u_lo);
    vst1q_f32(u_out.add(4), new_u_hi);
    vst1q_f32(b_out, new_b_lo);
    vst1q_f32(b_out.add(4), new_b_hi);
}

// ─── x86_64 SSE4.1 ──────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
#[target_feature(enable = "sse4.1")]
unsafe fn butterfly_chunk_4_sse(
    data: &mut [Complex32],
    start: usize,
    k: usize,
    half: usize,
    twiddles: &[Complex32],
) {
    use core::arch::x86_64::*;

    let u_ptr = data.as_mut_ptr().add(start + k) as *const f32;
    let b_ptr = data.as_mut_ptr().add(start + k + half) as *const f32;
    let w_ptr = twiddles.as_ptr() as *const f32;

    let u_lo = _mm_loadu_ps(u_ptr);
    let u_hi = _mm_loadu_ps(u_ptr.add(4));
    let b_lo = _mm_loadu_ps(b_ptr);
    let b_hi = _mm_loadu_ps(b_ptr.add(4));
    let w_lo = _mm_loadu_ps(w_ptr);
    let w_hi = _mm_loadu_ps(w_ptr.add(4));

    // Deinterleave via _mm_shuffle_ps:
    //   even: lanes (a0, a2, b0, b2) → mask 0b10001000 = 0x88
    //   odd:  lanes (a1, a3, b1, b3) → mask 0b11011101 = 0xDD
    let u_re = _mm_shuffle_ps::<0x88>(u_lo, u_hi);
    let u_im = _mm_shuffle_ps::<0xDD>(u_lo, u_hi);
    let b_re = _mm_shuffle_ps::<0x88>(b_lo, b_hi);
    let b_im = _mm_shuffle_ps::<0xDD>(b_lo, b_hi);
    let w_re = _mm_shuffle_ps::<0x88>(w_lo, w_hi);
    let w_im = _mm_shuffle_ps::<0xDD>(w_lo, w_hi);

    // Complex multiply v = b * w (no FMA).
    let bw_re_re = _mm_mul_ps(b_re, w_re);
    let bw_im_im = _mm_mul_ps(b_im, w_im);
    let bw_re_im = _mm_mul_ps(b_re, w_im);
    let bw_im_re = _mm_mul_ps(b_im, w_re);

    let v_re = _mm_sub_ps(bw_re_re, bw_im_im);
    let v_im = _mm_add_ps(bw_re_im, bw_im_re);

    // Butterfly.
    let new_u_re = _mm_add_ps(u_re, v_re);
    let new_u_im = _mm_add_ps(u_im, v_im);
    let new_b_re = _mm_sub_ps(u_re, v_re);
    let new_b_im = _mm_sub_ps(u_im, v_im);

    // Re-interleave via unpack:
    //   _mm_unpacklo_ps(re, im) = [re0, im0, re1, im1]
    //   _mm_unpackhi_ps(re, im) = [re2, im2, re3, im3]
    let new_u_lo = _mm_unpacklo_ps(new_u_re, new_u_im);
    let new_u_hi = _mm_unpackhi_ps(new_u_re, new_u_im);
    let new_b_lo = _mm_unpacklo_ps(new_b_re, new_b_im);
    let new_b_hi = _mm_unpackhi_ps(new_b_re, new_b_im);

    let u_out = data.as_mut_ptr().add(start + k) as *mut f32;
    let b_out = data.as_mut_ptr().add(start + k + half) as *mut f32;
    _mm_storeu_ps(u_out, new_u_lo);
    _mm_storeu_ps(u_out.add(4), new_u_hi);
    _mm_storeu_ps(b_out, new_b_lo);
    _mm_storeu_ps(b_out.add(4), new_b_hi);
}

// ─── WASM SIMD128 ───────────────────────────────────────────────

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn butterfly_chunk_4_wasm(
    data: &mut [Complex32],
    start: usize,
    k: usize,
    half: usize,
    twiddles: &[Complex32],
) {
    use core::arch::wasm32::*;

    let u_ptr = data.as_mut_ptr().add(start + k) as *mut v128;
    let b_ptr = data.as_mut_ptr().add(start + k + half) as *mut v128;
    let w_ptr = twiddles.as_ptr() as *const v128;

    let u_lo = v128_load(u_ptr);
    let u_hi = v128_load(u_ptr.add(1));
    let b_lo = v128_load(b_ptr);
    let b_hi = v128_load(b_ptr.add(1));
    let w_lo = v128_load(w_ptr);
    let w_hi = v128_load(w_ptr.add(1));

    // Deinterleave via i32x4_shuffle (type-erased 32-bit lane shuffle).
    // even lanes (re): pick lanes 0, 2, 4, 6 from (a, b)
    // odd lanes (im):  pick lanes 1, 3, 5, 7
    let u_re = i32x4_shuffle::<0, 2, 4, 6>(u_lo, u_hi);
    let u_im = i32x4_shuffle::<1, 3, 5, 7>(u_lo, u_hi);
    let b_re = i32x4_shuffle::<0, 2, 4, 6>(b_lo, b_hi);
    let b_im = i32x4_shuffle::<1, 3, 5, 7>(b_lo, b_hi);
    let w_re = i32x4_shuffle::<0, 2, 4, 6>(w_lo, w_hi);
    let w_im = i32x4_shuffle::<1, 3, 5, 7>(w_lo, w_hi);

    // Complex multiply v = b * w (no FMA).
    let bw_re_re = f32x4_mul(b_re, w_re);
    let bw_im_im = f32x4_mul(b_im, w_im);
    let bw_re_im = f32x4_mul(b_re, w_im);
    let bw_im_re = f32x4_mul(b_im, w_re);

    let v_re = f32x4_sub(bw_re_re, bw_im_im);
    let v_im = f32x4_add(bw_re_im, bw_im_re);

    // Butterfly.
    let new_u_re = f32x4_add(u_re, v_re);
    let new_u_im = f32x4_add(u_im, v_im);
    let new_b_re = f32x4_sub(u_re, v_re);
    let new_b_im = f32x4_sub(u_im, v_im);

    // Re-interleave: zip (re, im) lane-wise.
    //   low half:  [re0, im0, re1, im1] = shuffle::<0, 4, 1, 5>(re, im)
    //   high half: [re2, im2, re3, im3] = shuffle::<2, 6, 3, 7>(re, im)
    let new_u_lo = i32x4_shuffle::<0, 4, 1, 5>(new_u_re, new_u_im);
    let new_u_hi = i32x4_shuffle::<2, 6, 3, 7>(new_u_re, new_u_im);
    let new_b_lo = i32x4_shuffle::<0, 4, 1, 5>(new_b_re, new_b_im);
    let new_b_hi = i32x4_shuffle::<2, 6, 3, 7>(new_b_re, new_b_im);

    v128_store(u_ptr, new_u_lo);
    v128_store(u_ptr.add(1), new_u_hi);
    v128_store(b_ptr, new_b_lo);
    v128_store(b_ptr.add(1), new_b_hi);
}

// ─── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a deterministic test input: 8 Complex32 values (4 u + 4 b)
    /// + 4 Complex32 twiddles. Seeded so each test runs identically.
    fn build_test_inputs(seed: u32) -> (Vec<Complex32>, Vec<Complex32>) {
        let mut s = seed;
        let mut next_f32 = || {
            // Simple LCG → f32 in roughly [-1, 1].
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s as i32) as f32) / (i32::MAX as f32)
        };
        let mut data = Vec::with_capacity(16);
        // 8 Complex32 (4 u + 4 b), placed at [start+k..start+k+4] and
        // [start+k+half..start+k+half+4]. For tests, start=0, k=0, half=4.
        for _ in 0..8 {
            data.push(Complex32::new(next_f32() * 100.0, next_f32() * 100.0));
        }
        // Pad to 16 so half=4 layout is safe (need start+k+half+3 < 16).
        while data.len() < 16 {
            data.push(Complex32::new(0.0, 0.0));
        }
        let mut twiddles = Vec::with_capacity(4);
        for _ in 0..4 {
            twiddles.push(Complex32::new(next_f32(), next_f32()));
        }
        (data, twiddles)
    }

    /// Run the dispatched butterfly_chunk_4 and the scalar reference
    /// on identical inputs; assert per-component bit-equivalence on
    /// the 8 written outputs (4 u positions + 4 b positions).
    fn assert_dispatch_matches_scalar(seed: u32) {
        let (data_ref, twiddles) = build_test_inputs(seed);
        let mut data_scalar = data_ref.clone();
        let mut data_dispatched = data_ref.clone();

        butterfly_chunk_4_scalar(&mut data_scalar, 0, 0, 4, &twiddles);
        butterfly_chunk_4(&mut data_dispatched, 0, 0, 4, &twiddles);

        for i in 0..8 {
            assert_eq!(
                data_scalar[i].re.to_bits(),
                data_dispatched[i].re.to_bits(),
                "seed={seed} idx={i} re scalar != dispatched",
            );
            assert_eq!(
                data_scalar[i].im.to_bits(),
                data_dispatched[i].im.to_bits(),
                "seed={seed} idx={i} im scalar != dispatched",
            );
        }
    }

    #[test]
    fn butterfly_chunk_4_dispatch_matches_scalar() {
        // Sweep many random seeds for coverage.
        for seed in 0..200u32 {
            assert_dispatch_matches_scalar(seed.wrapping_mul(2654435761));
        }
    }

    #[test]
    fn butterfly_chunk_4_zero_twiddle() {
        // Edge case: twiddle = 0 → v = 0 → new_u = u, new_b = u.
        let (data_ref, _) = build_test_inputs(42);
        let zero_w = vec![Complex32::new(0.0, 0.0); 4];
        let mut data_scalar = data_ref.clone();
        let mut data_dispatched = data_ref.clone();

        butterfly_chunk_4_scalar(&mut data_scalar, 0, 0, 4, &zero_w);
        butterfly_chunk_4(&mut data_dispatched, 0, 0, 4, &zero_w);

        for i in 0..8 {
            assert_eq!(data_scalar[i].re.to_bits(), data_dispatched[i].re.to_bits());
            assert_eq!(data_scalar[i].im.to_bits(), data_dispatched[i].im.to_bits());
        }
    }

    #[test]
    fn butterfly_chunk_4_unit_twiddle() {
        // Twiddle = (1, 0): identity multiply → v = b → new_u = u+b,
        // new_b = u-b. Same as a half=4 butterfly with no twist.
        let (data_ref, _) = build_test_inputs(99);
        let unit_w = vec![Complex32::new(1.0, 0.0); 4];
        let mut data_scalar = data_ref.clone();
        let mut data_dispatched = data_ref.clone();

        butterfly_chunk_4_scalar(&mut data_scalar, 0, 0, 4, &unit_w);
        butterfly_chunk_4(&mut data_dispatched, 0, 0, 4, &unit_w);

        for i in 0..8 {
            assert_eq!(data_scalar[i].re.to_bits(), data_dispatched[i].re.to_bits());
            assert_eq!(data_scalar[i].im.to_bits(), data_dispatched[i].im.to_bits());
        }
    }

    /// Direct aarch64 NEON vs scalar comparison (both available here).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn butterfly_chunk_4_neon_matches_scalar_direct() {
        for seed in 0..500u32 {
            let s = seed.wrapping_mul(2654435761);
            let (data_ref, twiddles) = build_test_inputs(s);
            let mut data_scalar = data_ref.clone();
            let mut data_neon = data_ref.clone();

            butterfly_chunk_4_scalar(&mut data_scalar, 0, 0, 4, &twiddles);
            unsafe { butterfly_chunk_4_neon(&mut data_neon, 0, 0, 4, &twiddles); }

            for i in 0..8 {
                assert_eq!(
                    data_scalar[i].re.to_bits(),
                    data_neon[i].re.to_bits(),
                    "seed={s} idx={i} re: scalar={} neon={}",
                    data_scalar[i].re,
                    data_neon[i].re,
                );
                assert_eq!(
                    data_scalar[i].im.to_bits(),
                    data_neon[i].im.to_bits(),
                    "seed={s} idx={i} im",
                );
            }
        }
    }

    /// Direct x86_64 SSE vs scalar (under Rosetta 2 on Apple Silicon).
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
    #[test]
    fn butterfly_chunk_4_sse_matches_scalar_direct() {
        for seed in 0..500u32 {
            let s = seed.wrapping_mul(2654435761);
            let (data_ref, twiddles) = build_test_inputs(s);
            let mut data_scalar = data_ref.clone();
            let mut data_sse = data_ref.clone();

            butterfly_chunk_4_scalar(&mut data_scalar, 0, 0, 4, &twiddles);
            unsafe { butterfly_chunk_4_sse(&mut data_sse, 0, 0, 4, &twiddles); }

            for i in 0..8 {
                assert_eq!(data_scalar[i].re.to_bits(), data_sse[i].re.to_bits());
                assert_eq!(data_scalar[i].im.to_bits(), data_sse[i].im.to_bits());
            }
        }
    }

    /// Larger half value — exercises that the start/k/half indexing
    /// is correct when b is far from u in memory.
    #[test]
    fn butterfly_chunk_4_large_half() {
        // Build 1024 random complex values; choose start=128, k=8, half=256.
        let mut s: u32 = 0xC0FFEE;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s as i32) as f32) / (i32::MAX as f32)
        };
        let n = 1024;
        let data_ref: Vec<Complex32> =
            (0..n).map(|_| Complex32::new(next(), next())).collect();
        let twiddles: Vec<Complex32> =
            (0..4).map(|_| Complex32::new(next(), next())).collect();

        let mut data_scalar = data_ref.clone();
        let mut data_dispatched = data_ref.clone();

        let start = 128;
        let k = 8;
        let half = 256;

        butterfly_chunk_4_scalar(&mut data_scalar, start, k, half, &twiddles);
        butterfly_chunk_4(&mut data_dispatched, start, k, half, &twiddles);

        // Check the 8 written positions.
        for i in 0..4 {
            assert_eq!(
                data_scalar[start + k + i].re.to_bits(),
                data_dispatched[start + k + i].re.to_bits(),
            );
            assert_eq!(
                data_scalar[start + k + i].im.to_bits(),
                data_dispatched[start + k + i].im.to_bits(),
            );
            assert_eq!(
                data_scalar[start + k + half + i].re.to_bits(),
                data_dispatched[start + k + half + i].re.to_bits(),
            );
            assert_eq!(
                data_scalar[start + k + half + i].im.to_bits(),
                data_dispatched[start + k + half + i].im.to_bits(),
            );
        }
        // Untouched positions should be untouched.
        for i in 0..n {
            if (i >= start + k && i < start + k + 4)
                || (i >= start + k + half && i < start + k + half + 4)
            {
                continue;
            }
            assert_eq!(data_scalar[i].re.to_bits(), data_dispatched[i].re.to_bits(), "untouched idx={i}");
            assert_eq!(data_scalar[i].im.to_bits(), data_dispatched[i].im.to_bits());
        }
    }
}
