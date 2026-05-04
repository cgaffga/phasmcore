// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Hand-vectorised SIMD kernels for the H.264 encoder hot paths
//! (Phase I.2). Per the perf baseline the dominant scalar hotspots are
//! motion compensation 64.9% inclusive, SATD 58.6%, and SAD 7.0% — all
//! per-pixel byte arithmetic with no inter-pixel data dependencies, so
//! they map cleanly to byte-/halfword-wide SIMD lanes.
//!
//! ## Architecture & feature gating
//!
//! - **`simd` Cargo feature** (`core/Cargo.toml`) — top-level on/off.
//!   When OFF, the entire SIMD layer is `#[cfg]`-gated out and callers
//!   transparently fall through to the scalar implementations.
//! - **Per-ISA `target_arch` selection** — picks NEON (aarch64), AVX2
//!   (x86_64), or SIMD128 (wasm32) at compile time. The scalar
//!   fallback is always present as the reference implementation.
//! - **Runtime CPU feature detection** (x86_64 only) — `is_x86_feature_detected!("avx2")`
//!   guards AVX2 dispatch since AVX2 is not universal on x86_64
//!   (Haswell-and-later only). aarch64 NEON is mandatory in the
//!   architecture spec; no runtime check.
//! - **`PHASM_H264_DISABLE_SIMD=1` env var** — forces scalar path even
//!   when SIMD is compiled in. Useful for bit-exactness verification
//!   (encode same input both ways, hash-compare must match) and for
//!   scalar-vs-SIMD profiling.
//!
//! ## Determinism
//!
//! All SIMD kernels in this module are **integer-only and FMA-free**.
//! Per the cross-platform determinism audit
//! (`docs/design/h264-encoder-determinism-audit.md`), integer SIMD ops
//! are bit-exact across vendors and ISAs; floating-point and
//! fused-multiply-add are NOT. This module's contract:
//!
//! 1. Same inputs → same output bytes on iOS arm64 / Android arm64 /
//!    x86_64 / WASM (whether using NEON / AVX2 / SIMD128 / scalar).
//! 2. Same outputs as the existing scalar path. The dispatcher's
//!    `PHASM_H264_DISABLE_SIMD=1` mode exists specifically to verify
//!    this gate via hash-compare.
//!
//! Phase I.2 ships in stages (this commit = stage I.2a):
//! - **I.2a (this commit)** — NEON SAD + NEON SATD + NEON Hadamard 4×4.
//!   Smaller hot paths, simpler kernels, prove the dispatch
//!   infrastructure.
//! - **I.2b** (#73 follow-up) — NEON MC: `apply_luma_mv_block` +
//!   `sample_luma_frac`. Biggest hotspot but largest structural
//!   rewrite (16-way fractional-pel match table + 6-tap separable
//!   filter).
//! - **I.2c** (#74) — AVX2 ports of all kernels.
//! - **I.2d** (#75) — WASM SIMD128 ports.

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub(crate) mod neon;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(crate) mod avx2;

#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod wasm;

/// True if SIMD should be used for this call. Honors both compile-time
/// (`feature = "simd"` + `target_arch` + relevant `target_feature`)
/// and runtime gates:
/// - aarch64: NEON is mandatory in the spec, no runtime check.
/// - x86_64:  AVX2 is Haswell+ (2013+), runtime-detected via
///   `is_x86_feature_detected!("avx2")`.
/// - wasm32:  SIMD128 is decided at WASM module load time. If the
///   crate was built with `target_feature = "simd128"` enabled, the
///   module exists and SIMD is used; otherwise the module is
///   `#[cfg]`-out and callers fall through to scalar.
/// - `PHASM_H264_DISABLE_SIMD=1` env var forces scalar regardless
///   (no-op in browser WASM since `std::env::var` returns `Err`
///   there → defaults to enabled).
#[allow(dead_code)] // Diagnostic — feature-flag introspection; reserved for tooling.
#[inline]
pub(crate) fn simd_enabled() -> bool {
    #[cfg(not(all(
        feature = "simd",
        any(
            target_arch = "aarch64",
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128"),
        ),
    )))]
    {
        false
    }
    #[cfg(all(
        feature = "simd",
        any(
            target_arch = "aarch64",
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128"),
        ),
    ))]
    {
        use std::sync::OnceLock;
        static DISABLED: OnceLock<bool> = OnceLock::new();
        let disabled = *DISABLED.get_or_init(|| {
            // Env-var off-switch.
            let env_off = std::env::var("PHASM_H264_DISABLE_SIMD")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if env_off {
                return true;
            }
            // x86_64: AVX2 must be present at runtime.
            #[cfg(target_arch = "x86_64")]
            {
                if !std::is_x86_feature_detected!("avx2") {
                    return true;
                }
            }
            false
        });
        !disabled
    }
}

/// SIMD-dispatched `sad_block` matching the scalar signature. Falls
/// through to `scalar` for sizes / archs without a kernel.
#[inline]
pub(crate) fn sad_block_dispatch(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    block_w: u32,
    block_h: u32,
    scalar: impl FnOnce() -> u32,
) -> u32 {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if simd_enabled() {
        // SAFETY: kernel reads source[y * source_stride + 0..block_w] and
        // pred[y * pred_stride + 0..block_w] for y in 0..block_h. Caller
        // guarantees these ranges are in bounds (mirrors the scalar
        // implementation's read pattern).
        unsafe {
            if block_w == 16 {
                return neon::sad_w16(source, source_stride, pred, pred_stride, block_h as usize);
            }
            if block_w == 8 {
                return neon::sad_w8(source, source_stride, pred, pred_stride, block_h as usize);
            }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if simd_enabled() {
        // SAFETY: same contract as the NEON path. AVX2 availability
        // is gated by simd_enabled() (which checks
        // is_x86_feature_detected!("avx2")), so the target_feature
        // requirement is satisfied at the call site.
        unsafe {
            if block_w == 16 {
                return avx2::sad_w16(source, source_stride, pred, pred_stride, block_h as usize);
            }
            if block_w == 8 {
                return avx2::sad_w8(source, source_stride, pred, pred_stride, block_h as usize);
            }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    if simd_enabled() {
        // SAFETY: same contract as the NEON path.
        unsafe {
            if block_w == 16 {
                return wasm::sad_w16(source, source_stride, pred, pred_stride, block_h as usize);
            }
            if block_w == 8 {
                return wasm::sad_w8(source, source_stride, pred, pred_stride, block_h as usize);
            }
        }
    }
    let _ = (source, source_stride, pred, pred_stride, block_w, block_h);
    scalar()
}

/// SIMD-dispatched `satd_block` matching the scalar signature. Falls
/// through to `scalar` for sizes / archs without a kernel.
#[inline]
pub(crate) fn satd_block_dispatch(
    source: &[u8],
    source_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    block_w: u32,
    block_h: u32,
    scalar: impl FnOnce() -> u32,
) -> u32 {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if simd_enabled() {
        // SAFETY: same as sad_block_dispatch.
        unsafe {
            // SATD is computed as a sum over 4×4 tiles; works for any
            // multiple-of-4 width/height. Tile size matches the spec
            // 4×4 Hadamard in motion search.
            if block_w.is_multiple_of(4) && block_h.is_multiple_of(4) {
                return neon::satd_block_4x4_tiled(
                    source,
                    source_stride,
                    pred,
                    pred_stride,
                    block_w as usize,
                    block_h as usize,
                );
            }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if simd_enabled() {
        // SAFETY: same as NEON.
        unsafe {
            if block_w.is_multiple_of(4) && block_h.is_multiple_of(4) {
                return avx2::satd_block_4x4_tiled(
                    source,
                    source_stride,
                    pred,
                    pred_stride,
                    block_w as usize,
                    block_h as usize,
                );
            }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    if simd_enabled() {
        // SAFETY: same as NEON.
        unsafe {
            if block_w.is_multiple_of(4) && block_h.is_multiple_of(4) {
                return wasm::satd_block_4x4_tiled(
                    source,
                    source_stride,
                    pred,
                    pred_stride,
                    block_w as usize,
                    block_h as usize,
                );
            }
        }
    }
    let _ = (source, source_stride, pred, pred_stride, block_w, block_h);
    scalar()
}

/// SIMD-dispatched luma motion compensation for an `block_w × block_h`
/// rectangle. Returns true if the SIMD path handled the request; false
/// if the caller must fall back to scalar (e.g. unsupported size or
/// the block straddles a frame edge).
///
/// `mv_x` / `mv_y` are quarter-pel motion vectors. Integer part =
/// `mv >> 2`, fractional phase = `mv & 3`. Matches the scalar
/// `motion_compensation::apply_luma_mv_block` semantics exactly.
#[inline]
pub(crate) fn apply_luma_mv_block_dispatch(
    y_plane: &[u8],
    plane_w: u32,
    plane_h: u32,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    mv_x: i16,
    mv_y: i16,
    out: &mut [u8],
    out_stride: usize,
) -> bool {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if simd_enabled() {
        let mv_x_int = mv_x as i32 >> 2;
        let mv_y_int = mv_y as i32 >> 2;
        let x_frac = (mv_x & 3) as u8;
        let y_frac = (mv_y & 3) as u8;
        // Fast path: integer MV, in-bounds, multiple-of-8 width.
        // SAFETY: kernel reads y_plane[(src_y + dy) * plane_w + src_x .. + block_w]
        // for dy in 0..block_h, after the in-bounds bounds check inside
        // the kernel returns true. Caller guarantees `out` covers the
        // `block_w × block_h` rectangle at `out_stride`.
        if x_frac == 0 && y_frac == 0 && block_w.is_multiple_of(8) {
            unsafe {
                return neon::mc_luma_integer_mv(
                    y_plane, plane_w, plane_h,
                    block_x, block_y, block_w, block_h,
                    mv_x_int, mv_y_int,
                    out, out_stride,
                );
            }
        }
        // Pure horizontal half-pel (x_frac == 2, y_frac == 0): single
        // 6-tap filter on each row, in-bounds fast path only.
        if y_frac == 0 && (x_frac == 1 || x_frac == 2 || x_frac == 3)
            && block_w.is_multiple_of(8)
        {
            unsafe {
                return neon::mc_luma_h_only(
                    y_plane, plane_w, plane_h,
                    block_x, block_y, block_w, block_h,
                    mv_x_int, mv_y_int, x_frac,
                    out, out_stride,
                );
            }
        }
        // Pure vertical half-pel (x_frac == 0, y_frac in {1,2,3}):
        // single 6-tap filter on each column, in-bounds fast path only.
        if x_frac == 0 && (y_frac == 1 || y_frac == 2 || y_frac == 3)
            && block_w.is_multiple_of(8)
        {
            unsafe {
                return neon::mc_luma_v_only(
                    y_plane, plane_w, plane_h,
                    block_x, block_y, block_w, block_h,
                    mv_x_int, mv_y_int, y_frac,
                    out, out_stride,
                );
            }
        }
        // Composite cases (x_frac > 0 AND y_frac > 0): handles all 9
        // remaining MC fraction classes including the centre `j`.
        if x_frac > 0 && y_frac > 0 && block_w.is_multiple_of(8) {
            unsafe {
                return neon::mc_luma_composite(
                    y_plane, plane_w, plane_h,
                    block_x, block_y, block_w, block_h,
                    mv_x_int, mv_y_int, x_frac, y_frac,
                    out, out_stride,
                );
            }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if simd_enabled() {
        let mv_x_int = mv_x as i32 >> 2;
        let mv_y_int = mv_y as i32 >> 2;
        let x_frac = (mv_x & 3) as u8;
        let y_frac = (mv_y & 3) as u8;
        // Mirror the NEON dispatch table 1:1.
        if x_frac == 0 && y_frac == 0 && block_w.is_multiple_of(8) {
            unsafe {
                return avx2::mc_luma_integer_mv(
                    y_plane, plane_w, plane_h,
                    block_x, block_y, block_w, block_h,
                    mv_x_int, mv_y_int,
                    out, out_stride,
                );
            }
        }
        if y_frac == 0 && (x_frac == 1 || x_frac == 2 || x_frac == 3)
            && block_w.is_multiple_of(8)
        {
            unsafe {
                return avx2::mc_luma_h_only(
                    y_plane, plane_w, plane_h,
                    block_x, block_y, block_w, block_h,
                    mv_x_int, mv_y_int, x_frac,
                    out, out_stride,
                );
            }
        }
        if x_frac == 0 && (y_frac == 1 || y_frac == 2 || y_frac == 3)
            && block_w.is_multiple_of(8)
        {
            unsafe {
                return avx2::mc_luma_v_only(
                    y_plane, plane_w, plane_h,
                    block_x, block_y, block_w, block_h,
                    mv_x_int, mv_y_int, y_frac,
                    out, out_stride,
                );
            }
        }
        if x_frac > 0 && y_frac > 0 && block_w.is_multiple_of(8) {
            unsafe {
                return avx2::mc_luma_composite(
                    y_plane, plane_w, plane_h,
                    block_x, block_y, block_w, block_h,
                    mv_x_int, mv_y_int, x_frac, y_frac,
                    out, out_stride,
                );
            }
        }
    }
    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    if simd_enabled() {
        let mv_x_int = mv_x as i32 >> 2;
        let mv_y_int = mv_y as i32 >> 2;
        let x_frac = (mv_x & 3) as u8;
        let y_frac = (mv_y & 3) as u8;
        if x_frac == 0 && y_frac == 0 && block_w.is_multiple_of(8) {
            unsafe {
                return wasm::mc_luma_integer_mv(
                    y_plane, plane_w, plane_h,
                    block_x, block_y, block_w, block_h,
                    mv_x_int, mv_y_int,
                    out, out_stride,
                );
            }
        }
        if y_frac == 0 && (x_frac == 1 || x_frac == 2 || x_frac == 3)
            && block_w.is_multiple_of(8)
        {
            unsafe {
                return wasm::mc_luma_h_only(
                    y_plane, plane_w, plane_h,
                    block_x, block_y, block_w, block_h,
                    mv_x_int, mv_y_int, x_frac,
                    out, out_stride,
                );
            }
        }
        if x_frac == 0 && (y_frac == 1 || y_frac == 2 || y_frac == 3)
            && block_w.is_multiple_of(8)
        {
            unsafe {
                return wasm::mc_luma_v_only(
                    y_plane, plane_w, plane_h,
                    block_x, block_y, block_w, block_h,
                    mv_x_int, mv_y_int, y_frac,
                    out, out_stride,
                );
            }
        }
        if x_frac > 0 && y_frac > 0 && block_w.is_multiple_of(8) {
            unsafe {
                return wasm::mc_luma_composite(
                    y_plane, plane_w, plane_h,
                    block_x, block_y, block_w, block_h,
                    mv_x_int, mv_y_int, x_frac, y_frac,
                    out, out_stride,
                );
            }
        }
    }
    let _ = (
        y_plane, plane_w, plane_h,
        block_x, block_y, block_w, block_h,
        mv_x, mv_y, out, out_stride,
    );
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: allocate a 16×16 raster where each pixel is `(x + y) % 256`.
    fn rasters() -> ([u8; 256], [u8; 256]) {
        let mut src = [0u8; 256];
        let mut prd = [0u8; 256];
        for y in 0..16 {
            for x in 0..16 {
                src[y * 16 + x] = ((x + y) % 256) as u8;
                prd[y * 16 + x] = ((x + 2 * y + 7) % 256) as u8;
            }
        }
        (src, prd)
    }

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

    #[test]
    fn sad_dispatch_matches_scalar_16x16() {
        let (src, prd) = rasters();
        let want = scalar_sad(&src, 16, &prd, 16, 16, 16);
        let got = sad_block_dispatch(&src, 16, &prd, 16, 16, 16, || {
            scalar_sad(&src, 16, &prd, 16, 16, 16)
        });
        assert_eq!(got, want);
    }

    #[test]
    fn sad_dispatch_matches_scalar_8x8() {
        let (src, prd) = rasters();
        let want = scalar_sad(&src, 16, &prd, 16, 8, 8);
        let got = sad_block_dispatch(&src, 16, &prd, 16, 8, 8, || {
            scalar_sad(&src, 16, &prd, 16, 8, 8)
        });
        assert_eq!(got, want);
    }

    #[test]
    fn sad_dispatch_falls_through_to_scalar_for_4x4() {
        let (src, prd) = rasters();
        let want = scalar_sad(&src, 16, &prd, 16, 4, 4);
        let got = sad_block_dispatch(&src, 16, &prd, 16, 4, 4, || {
            scalar_sad(&src, 16, &prd, 16, 4, 4)
        });
        assert_eq!(got, want);
    }

    #[test]
    fn sad_dispatch_8x16_mixed_dimensions() {
        let (src, prd) = rasters();
        let want = scalar_sad(&src, 16, &prd, 16, 8, 16);
        let got = sad_block_dispatch(&src, 16, &prd, 16, 8, 16, || {
            scalar_sad(&src, 16, &prd, 16, 8, 16)
        });
        assert_eq!(got, want);
    }

    #[test]
    fn sad_dispatch_16x8_mixed_dimensions() {
        let (src, prd) = rasters();
        let want = scalar_sad(&src, 16, &prd, 16, 16, 8);
        let got = sad_block_dispatch(&src, 16, &prd, 16, 16, 8, || {
            scalar_sad(&src, 16, &prd, 16, 16, 8)
        });
        assert_eq!(got, want);
    }

    // ---- mc_luma_block_dispatch ----
    //
    // Cross-checks: SIMD path output must match the existing scalar
    // `apply_luma_mv_block` byte-for-byte.

    fn make_y_plane(w: u32, h: u32, seed: u32) -> Vec<u8> {
        let mut s = seed;
        let mut buf = Vec::with_capacity((w * h) as usize);
        for _ in 0..(w * h) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            buf.push((s >> 16) as u8);
        }
        buf
    }

    /// Run scalar `apply_luma_mv_block` against a synthesised plane and
    /// return its output. Rebuilt locally to avoid a cyclic dep on the
    /// motion_compensation module (which now calls the dispatcher).
    fn scalar_mc(
        plane: &[u8], w: u32, h: u32,
        block_x: u32, block_y: u32, bw: u32, bh: u32,
        mv_x: i16, mv_y: i16,
    ) -> Vec<u8> {
        // Mirror the `motion_compensation::sample_luma_frac` filter
        // exactly across all 16 (x_frac, y_frac) cases.
        let clip = |v: i32| -> u8 { v.clamp(0, 255) as u8 };
        let sample = |x: i32, y: i32| -> i32 {
            let xc = x.clamp(0, w as i32 - 1) as u32;
            let yc = y.clamp(0, h as i32 - 1) as u32;
            plane[(yc * w + xc) as usize] as i32
        };
        let filter6 = |s: [i32; 6]| -> i32 {
            s[0] - 5*s[1] + 20*s[2] + 20*s[3] - 5*s[4] + s[5]
        };
        let mut out = vec![0u8; (bw * bh) as usize];
        for yl in 0..bh {
            for xl in 0..bw {
                let xi = block_x as i32 + (mv_x as i32 >> 2) + xl as i32;
                let yi = block_y as i32 + (mv_y as i32 >> 2) + yl as i32;
                let xf = (mv_x & 3) as u8;
                let yf = (mv_y & 3) as u8;
                let g = sample(xi, yi);
                let h_int = sample(xi + 1, yi);
                let m_int = sample(xi, yi + 1);
                let b1_at = |dy: i32| -> i32 {
                    filter6([
                        sample(xi - 2, yi + dy),
                        sample(xi - 1, yi + dy),
                        sample(xi, yi + dy),
                        sample(xi + 1, yi + dy),
                        sample(xi + 2, yi + dy),
                        sample(xi + 3, yi + dy),
                    ])
                };
                let b_at = |dy: i32| -> i32 { clip((b1_at(dy) + 16) >> 5) as i32 };
                let h_at = |dx: i32| -> i32 {
                    let h1 = filter6([
                        sample(xi + dx, yi - 2),
                        sample(xi + dx, yi - 1),
                        sample(xi + dx, yi),
                        sample(xi + dx, yi + 1),
                        sample(xi + dx, yi + 2),
                        sample(xi + dx, yi + 3),
                    ]);
                    clip((h1 + 16) >> 5) as i32
                };
                let j = || {
                    let j1 = filter6([
                        b1_at(-2), b1_at(-1), b1_at(0),
                        b1_at(1), b1_at(2), b1_at(3),
                    ]);
                    clip((j1 + 512) >> 10) as i32
                };
                let val = match (xf, yf) {
                    (0, 0) => g,
                    (1, 0) => (g + b_at(0) + 1) >> 1,
                    (2, 0) => b_at(0),
                    (3, 0) => (h_int + b_at(0) + 1) >> 1,
                    (0, 1) => (g + h_at(0) + 1) >> 1,
                    (0, 2) => h_at(0),
                    (0, 3) => (m_int + h_at(0) + 1) >> 1,
                    (1, 1) => (b_at(0) + h_at(0) + 1) >> 1,
                    (2, 1) => (b_at(0) + j() + 1) >> 1,
                    (3, 1) => (b_at(0) + h_at(1) + 1) >> 1,
                    (1, 2) => (h_at(0) + j() + 1) >> 1,
                    (2, 2) => j(),
                    (3, 2) => (j() + h_at(1) + 1) >> 1,
                    (1, 3) => (h_at(0) + b_at(1) + 1) >> 1,
                    (2, 3) => (j() + b_at(1) + 1) >> 1,
                    (3, 3) => (h_at(1) + b_at(1) + 1) >> 1,
                    _ => unreachable!(),
                };
                out[(yl * bw + xl) as usize] = val.clamp(0, 255) as u8;
            }
        }
        out
    }

    fn run_dispatch(
        plane: &[u8], w: u32, h: u32,
        bx: u32, by: u32, bw: u32, bh: u32,
        mv_x: i16, mv_y: i16,
    ) -> Option<Vec<u8>> {
        let mut out = vec![0u8; (bw * bh) as usize];
        let handled = apply_luma_mv_block_dispatch(
            plane, w, h, bx, by, bw, bh, mv_x, mv_y,
            &mut out, bw as usize,
        );
        if handled { Some(out) } else { None }
    }

    /// Assert SIMD output matches scalar IF the dispatcher used SIMD.
    /// If SIMD was unavailable at runtime (e.g. x86_64 without AVX2,
    /// running under Rosetta 2, or `PHASM_H264_DISABLE_SIMD=1`), the
    /// dispatcher correctly returns false → there's no SIMD output to
    /// compare. The fallback-to-scalar path is then exercised by the
    /// caller in production.
    #[track_caller]
    fn assert_simd_matches_scalar(
        plane: &[u8], w: u32, h: u32,
        bx: u32, by: u32, bw: u32, bh: u32,
        mv_x: i16, mv_y: i16, label: &str,
    ) {
        let want = scalar_mc(plane, w, h, bx, by, bw, bh, mv_x, mv_y);
        if let Some(got) = run_dispatch(plane, w, h, bx, by, bw, bh, mv_x, mv_y) {
            assert_eq!(got, want, "{label}");
        }
    }

    #[test]
    fn mc_dispatch_integer_mv_16x16_matches_scalar() {
        let (w, h) = (64, 48);
        let plane = make_y_plane(w, h, 0xc0ffee);
        for &(mvx, mvy) in &[(0i16, 0i16), (8, 0), (-4, 12), (16, -8)] {
            assert_simd_matches_scalar(
                &plane, w, h, 16, 16, 16, 16, mvx, mvy,
                &format!("mv=({mvx},{mvy})"),
            );
        }
    }

    #[test]
    fn mc_dispatch_integer_mv_8x8_matches_scalar() {
        let (w, h) = (64, 48);
        let plane = make_y_plane(w, h, 0xdeadbeef);
        assert_simd_matches_scalar(&plane, w, h, 8, 8, 8, 8, 0, 0, "8x8 zero MV");
    }

    #[test]
    fn mc_dispatch_integer_mv_falls_back_at_edge() {
        let (w, h) = (64, 48);
        let plane = make_y_plane(w, h, 0xfeedface);
        // MV pulls block off the left edge → kernel should return false.
        assert!(run_dispatch(&plane, w, h, 0, 16, 16, 16, -80, 0).is_none());
    }

    #[test]
    fn mc_dispatch_h_only_halfpel_matches_scalar() {
        let (w, h) = (64, 48);
        let plane = make_y_plane(w, h, 0xa5a5a5a5);
        for &(mvx, mvy) in &[(1i16, 0i16), (2, 0), (3, 0), (5, 0), (6, 0), (-2, 0)] {
            assert_simd_matches_scalar(
                &plane, w, h, 16, 16, 16, 16, mvx, mvy,
                &format!("h-only mv=({mvx},{mvy})"),
            );
        }
    }

    #[test]
    fn mc_dispatch_v_only_halfpel_matches_scalar() {
        let (w, h) = (64, 48);
        let plane = make_y_plane(w, h, 0x12345678);
        for &(mvx, mvy) in &[(0i16, 1i16), (0, 2), (0, 3), (0, 5), (0, 6), (0, -2)] {
            assert_simd_matches_scalar(
                &plane, w, h, 16, 16, 16, 16, mvx, mvy,
                &format!("v-only mv=({mvx},{mvy})"),
            );
        }
    }

    #[test]
    fn mc_dispatch_composite_all_9_cases_match_scalar() {
        let (w, h) = (96, 64);
        let plane = make_y_plane(w, h, 0xc0deface);
        for &(xf, yf) in &[
            (1u8, 1u8), (2, 1), (3, 1),
            (1, 2),     (2, 2), (3, 2),
            (1, 3),     (2, 3), (3, 3),
        ] {
            for &(bw, bh) in &[(16u32, 16u32), (8, 8), (16, 8), (8, 16)] {
                let mvx = xf as i16;
                let mvy = yf as i16;
                assert_simd_matches_scalar(
                    &plane, w, h, 16, 16, bw, bh, mvx, mvy,
                    &format!("composite ({xf},{yf}) {bw}x{bh}"),
                );
            }
        }
    }

    #[test]
    fn mc_dispatch_composite_falls_back_at_edge() {
        let (w, h) = (64, 48);
        let plane = make_y_plane(w, h, 0x42);
        // Composite at frame edge: needs source col +1 past block_w + 3,
        // so block_x at the rightmost column triggers the fallback.
        assert!(run_dispatch(&plane, w, h, w - 16, 16, 16, 16, 2, 2).is_none());
    }

    #[test]
    fn mc_dispatch_composite_falls_back_for_4_wide() {
        let (w, h) = (64, 48);
        let plane = make_y_plane(w, h, 0x42);
        // 4-wide blocks fall through (not multiple-of-8).
        assert!(run_dispatch(&plane, w, h, 16, 16, 4, 4, 2, 2).is_none());
    }
}
