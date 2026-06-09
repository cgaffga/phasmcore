// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! SIMD-accelerated SPREAD_LEN=8 dot product for STDM.
//!
//! `stdm_extract_soft` and `stdm_embed` previously computed the
//! projection `p = sum(coeffs[i] * v[i] for i in 0..8)` via
//! `iter().zip(...).map(|(c, vi)| c * vi).sum()` — left-to-right
//! associative accumulation.
//!
//! This module reorders that to a pinned 3-level pairwise tree so SIMD lanes
//! can be reduced via the architecture's natural horizontal-add
//! intrinsic (`vpaddq_f64` / `_mm_hadd_pd`) while remaining
//! bit-identical across all SIMD paths. Math equivalent of:
//!
//! ```text
//!   L1: t0=a0+a1, t1=a2+a3, t2=a4+a5, t3=a6+a7
//!   L2: u0=t0+t1, u1=t2+t3
//!   L3: u0+u1
//! ```
//!
//! Output differs from the original `.iter().sum()` order in the
//! last bits — but cross-platform consistency is preserved by Path A:
//! every dispatch path (scalar, NEON, SSE3, WASM SIMD128) executes
//! the same pairwise tree on identical IEEE 754 f64 ops, no FMA,
//! no further reordering.
//!
//! ## Wire-format safety
//!
//! The LLR produced by `stdm_extract_soft` is internal to Armor
//! decode (not stored in any stego). The decode soft-vote tolerates
//! small magnitude perturbations as long as the LLR sign stays the
//! same for each bit. Empirically verified on a 12-image Armor
//! baseline corpus (stegos from the old left-to-right order decode
//! byte-identical plaintext under the pairwise-tree order).
//!
//! `stdm_embed` also uses the same projection. Encoder output bytes
//! depend on the projection's f64 value — so encode output WILL
//! differ from the old left-to-right path in the last bits. Decode of NEW
//! stego is internally consistent (same tree both sides). Decode of
//! OLD stego succeeds because LLR signs match (ECC absorbs any
//! magnitude perturbations).
//!
//! ## Per-arch status (2026-05-21)
//!
//! - aarch64 NEON — 4× f64x2 muls + 3× vpaddq_f64 + 1× vaddvq_f64
//! - x86_64 SSE3 — 4× _mm_mul_pd + 3× _mm_hadd_pd + 1× extract
//!   (SSE3's `_mm_hadd_pd` is in the project compile baseline via
//!   `+ssse3` in `.cargo/config.toml`)
//! - WASM SIMD128 — 4× f64x2_mul + 8× lane extract + scalar tree
//!   (WASM SIMD128 has no native horizontal f64 add)
//! - Other archs — scalar pairwise tree

use super::spreading::SPREAD_LEN;

/// SIMD-aware projection: returns `sum(coeffs[i] * v[i])` via a
/// pinned 3-level pairwise tree. Bit-identical across all dispatch
/// paths.
#[inline]
pub fn spread_dot_product(coeffs: &[f64; SPREAD_LEN], v: &[f64; SPREAD_LEN]) -> f64 {
    #[cfg(target_arch = "aarch64")]
    // Safety: aarch64 always has NEON per ARMv8 spec.
    unsafe {
        spread_dot_product_neon(coeffs, v)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
    // Safety: SSE3/SSSE3 is the x86_64 compile baseline in this
    // project per `.cargo/config.toml`. `_mm_hadd_pd` is SSE3.
    unsafe {
        spread_dot_product_sse(coeffs, v)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // Safety: simd128 enabled at compile time per cfg gate.
    unsafe {
        spread_dot_product_wasm(coeffs, v)
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"),
        all(target_arch = "wasm32", target_feature = "simd128"),
    )))]
    spread_dot_product_scalar(coeffs, v)
}

/// Scalar reference. Always available; used as truth baseline in
/// tests and as the fallback path on archs without a SIMD impl.
///
/// The pinned 3-level pairwise tree is the SHARED specification —
/// every SIMD path emits exactly this sequence of IEEE 754 muls and
/// pairwise adds (no FMA, no cross-pair reduction).
#[inline]
pub(super) fn spread_dot_product_scalar(
    coeffs: &[f64; SPREAD_LEN],
    v: &[f64; SPREAD_LEN],
) -> f64 {
    let a0 = coeffs[0] * v[0];
    let a1 = coeffs[1] * v[1];
    let a2 = coeffs[2] * v[2];
    let a3 = coeffs[3] * v[3];
    let a4 = coeffs[4] * v[4];
    let a5 = coeffs[5] * v[5];
    let a6 = coeffs[6] * v[6];
    let a7 = coeffs[7] * v[7];
    // L1
    let t0 = a0 + a1;
    let t1 = a2 + a3;
    let t2 = a4 + a5;
    let t3 = a6 + a7;
    // L2
    let u0 = t0 + t1;
    let u1 = t2 + t3;
    // L3
    u0 + u1
}

// ─── aarch64 NEON ───────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn spread_dot_product_neon(
    coeffs: &[f64; SPREAD_LEN],
    v: &[f64; SPREAD_LEN],
) -> f64 {
    use core::arch::aarch64::*;

    let cp = coeffs.as_ptr();
    let vp = v.as_ptr();

    // 4 × f64x2 muls — each produces 2 products.
    let vp0 = vmulq_f64(vld1q_f64(cp), vld1q_f64(vp));
    let vp1 = vmulq_f64(vld1q_f64(cp.add(2)), vld1q_f64(vp.add(2)));
    let vp2 = vmulq_f64(vld1q_f64(cp.add(4)), vld1q_f64(vp.add(4)));
    let vp3 = vmulq_f64(vld1q_f64(cp.add(6)), vld1q_f64(vp.add(6)));

    // L1 via pairwise add within each pair of vectors.
    //   vpaddq_f64(a, b) = (a[0]+a[1], b[0]+b[1])
    let l1_01 = vpaddq_f64(vp0, vp1); // (a0+a1, a2+a3)
    let l1_23 = vpaddq_f64(vp2, vp3); // (a4+a5, a6+a7)

    // L2 via pairwise add.
    let l2 = vpaddq_f64(l1_01, l1_23); // ((a0+a1)+(a2+a3), (a4+a5)+(a6+a7))

    // L3 via vaddvq_f64 — horizontal add of the 2 lanes.
    vaddvq_f64(l2)
}

// ─── x86_64 SSE3 ────────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
#[target_feature(enable = "sse3")]
unsafe fn spread_dot_product_sse(
    coeffs: &[f64; SPREAD_LEN],
    v: &[f64; SPREAD_LEN],
) -> f64 {
    use core::arch::x86_64::*;

    let cp = coeffs.as_ptr();
    let vp = v.as_ptr();

    // 4 × _mm_mul_pd — each produces 2 products in a __m128d.
    let vp0 = _mm_mul_pd(_mm_loadu_pd(cp), _mm_loadu_pd(vp));
    let vp1 = _mm_mul_pd(_mm_loadu_pd(cp.add(2)), _mm_loadu_pd(vp.add(2)));
    let vp2 = _mm_mul_pd(_mm_loadu_pd(cp.add(4)), _mm_loadu_pd(vp.add(4)));
    let vp3 = _mm_mul_pd(_mm_loadu_pd(cp.add(6)), _mm_loadu_pd(vp.add(6)));

    // L1 via _mm_hadd_pd — horizontal pairwise add.
    //   _mm_hadd_pd(a, b) = (a[0]+a[1], b[0]+b[1])
    let l1_01 = _mm_hadd_pd(vp0, vp1); // (a0+a1, a2+a3)
    let l1_23 = _mm_hadd_pd(vp2, vp3); // (a4+a5, a6+a7)

    // L2.
    let l2 = _mm_hadd_pd(l1_01, l1_23); // ((a0+a1)+(a2+a3), (a4+a5)+(a6+a7))

    // L3 — sum the 2 lanes. Extract both and scalar add for
    // bit-identical pairing with NEON's vaddvq_f64.
    let mut out = [0.0f64; 2];
    _mm_storeu_pd(out.as_mut_ptr(), l2);
    out[0] + out[1]
}

// ─── WASM SIMD128 ───────────────────────────────────────────────

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn spread_dot_product_wasm(
    coeffs: &[f64; SPREAD_LEN],
    v: &[f64; SPREAD_LEN],
) -> f64 {
    use core::arch::wasm32::*;

    let cp = coeffs.as_ptr() as *const v128;
    let vp = v.as_ptr() as *const v128;

    // 4 × f64x2_mul — each produces 2 products.
    let vp0 = f64x2_mul(v128_load(cp), v128_load(vp));
    let vp1 = f64x2_mul(v128_load(cp.add(1)), v128_load(vp.add(1)));
    let vp2 = f64x2_mul(v128_load(cp.add(2)), v128_load(vp.add(2)));
    let vp3 = f64x2_mul(v128_load(cp.add(3)), v128_load(vp.add(3)));

    // WASM SIMD128 has no native horizontal pairwise add for f64.
    // Extract 8 lanes and apply the scalar pinned-pairwise tree —
    // by-construction bit-equivalent to NEON/SSE outputs because
    // each lane carries one f64 product computed via the same
    // IEEE 754 mul.
    let a0 = f64x2_extract_lane::<0>(vp0);
    let a1 = f64x2_extract_lane::<1>(vp0);
    let a2 = f64x2_extract_lane::<0>(vp1);
    let a3 = f64x2_extract_lane::<1>(vp1);
    let a4 = f64x2_extract_lane::<0>(vp2);
    let a5 = f64x2_extract_lane::<1>(vp2);
    let a6 = f64x2_extract_lane::<0>(vp3);
    let a7 = f64x2_extract_lane::<1>(vp3);
    let t0 = a0 + a1;
    let t1 = a2 + a3;
    let t2 = a4 + a5;
    let t3 = a6 + a7;
    let u0 = t0 + t1;
    let u1 = t2 + t3;
    u0 + u1
}

// ─── Test/diagnostic helpers ────────────────────────────────────

/// Test/diagnostic: returns the hex SHA256 of a deterministic batch
/// of `spread_dot_product` calls. Used to verify cross-platform
/// bit-equivalence of the LLR projection across scalar / NEON / SSE
/// / WASM SIMD128 paths.
#[doc(hidden)]
pub fn spread_dot_test_hash_hex() -> String {
    use sha2::{Digest, Sha256};
    let mut s: u32 = 0xDEADBEEF;
    let mut next_f64 = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((s as i32) as f64) / (i32::MAX as f64)
    };
    let mut hasher = Sha256::new();
    // 10 000 random (coeffs, v) inputs.
    for _ in 0..10_000 {
        let mut coeffs = [0.0f64; SPREAD_LEN];
        let mut vv = [0.0f64; SPREAD_LEN];
        for i in 0..SPREAD_LEN {
            coeffs[i] = next_f64() * 100.0;
            vv[i] = next_f64();
        }
        let p = spread_dot_product(&coeffs, &vv);
        hasher.update(&p.to_le_bytes());
    }
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

// ─── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_inputs(seed: u32) -> ([f64; SPREAD_LEN], [f64; SPREAD_LEN]) {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s as i32) as f64) / (i32::MAX as f64)
        };
        let mut c = [0.0f64; SPREAD_LEN];
        let mut v = [0.0f64; SPREAD_LEN];
        for i in 0..SPREAD_LEN {
            c[i] = next() * 100.0;
            v[i] = next();
        }
        (c, v)
    }

    #[test]
    fn dispatch_matches_scalar_random() {
        for seed in 0..1000u32 {
            let s = seed.wrapping_mul(2654435761);
            let (c, v) = lcg_inputs(s);
            let scalar = spread_dot_product_scalar(&c, &v);
            let dispatched = spread_dot_product(&c, &v);
            assert_eq!(
                scalar.to_bits(),
                dispatched.to_bits(),
                "seed={s} scalar={scalar} dispatched={dispatched}"
            );
        }
    }

    #[test]
    fn dispatch_zero_coeffs() {
        let c = [0.0f64; SPREAD_LEN];
        let v = [1.0; SPREAD_LEN];
        assert_eq!(spread_dot_product(&c, &v), 0.0);
    }

    #[test]
    fn dispatch_unit_v_matches_sum_of_coeffs_pinned_tree() {
        // v = (1, 1, 1, 1, 1, 1, 1, 1) → dot product = sum(coeffs).
        // Both scalar reference and dispatched produce the same
        // pinned-tree sum.
        let c = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = [1.0; SPREAD_LEN];
        // Pinned tree result: ((1+2)+(3+4)) + ((5+6)+(7+8)) = 10 + 26 = 36.
        assert_eq!(spread_dot_product_scalar(&c, &v), 36.0);
        assert_eq!(spread_dot_product(&c, &v), 36.0);
    }

    /// Direct aarch64 NEON vs scalar (always available here).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_direct() {
        for seed in 0..2000u32 {
            let s = seed.wrapping_mul(2654435761);
            let (c, v) = lcg_inputs(s);
            let scalar = spread_dot_product_scalar(&c, &v);
            let neon = unsafe { spread_dot_product_neon(&c, &v) };
            assert_eq!(
                scalar.to_bits(),
                neon.to_bits(),
                "seed={s}: scalar={scalar} neon={neon}"
            );
        }
    }

    /// Direct x86_64 SSE vs scalar (under Rosetta 2).
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
    #[test]
    fn sse_matches_scalar_direct() {
        for seed in 0..2000u32 {
            let s = seed.wrapping_mul(2654435761);
            let (c, v) = lcg_inputs(s);
            let scalar = spread_dot_product_scalar(&c, &v);
            let sse = unsafe { spread_dot_product_sse(&c, &v) };
            assert_eq!(
                scalar.to_bits(),
                sse.to_bits(),
                "seed={s}: scalar={scalar} sse={sse}"
            );
        }
    }
}
