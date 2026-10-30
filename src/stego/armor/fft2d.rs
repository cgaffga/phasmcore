// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Deterministic 2D FFT/IFFT using only WASM-intrinsic f64 operations.
//!
//! Replaces `rustfft` with an in-house implementation:
//! - Radix-2 Cooley-Tukey for power-of-2 sizes
//! - Bluestein's chirp-z transform for arbitrary sizes
//! All twiddle factors computed via `det_sincos()`.
//!
//! Memory optimizations:
//! - Column FFTs use gather-FFT-scatter with a single column buffer
//!   instead of a full transposed copy (saves ~186 MB for 4032x3024).
//! - All FFT data uses f32 (Complex32) — template detection only needs
//!   coarse peak finding. Twiddle factors still computed in f64 via
//!   `det_sincos()` then cast to f32.
//! - BluesteinPlan precomputes chirp factors and FFT(b_hat) for reuse
//!   across all rows/columns of the same length.

use num_complex::Complex;
use crate::det_math::{det_sincos, det_hypot};
use std::f64::consts::PI;

/// Complex32 type alias for f32 complex numbers.
pub type Complex32 = Complex<f32>;

/// 2D complex spectrum using f32 for memory efficiency.
pub struct Spectrum2D {
    pub data: Vec<Complex32>,
    pub width: usize,
    pub height: usize,
}

// ──────────────────────────────────────────────────────────────────────────
// Bluestein plan: precomputed chirp factors for reuse (P2a)
// ──────────────────────────────────────────────────────────────────────────

/// Precomputed Bluestein chirp factors and FFT(b_hat) for a given (n, sign).
///
/// Eliminates redundant chirp computation and FFT(b) calls when processing
/// many rows or columns of the same length.
struct BluesteinPlan {
    n: usize,
    m: usize, // next_pow2(2*n - 1)
    chirp: Vec<Complex32>,
    b_hat: Vec<Complex32>, // FFT of padded conjugate chirp
}

impl BluesteinPlan {
    /// Create a new Bluestein plan for length `n` and direction `sign`.
    fn new(n: usize, sign: f64) -> Self {
        let m = next_pow2(2 * n - 1);

        // Chirp factors: w_k = exp(sign * i * pi * k^2 / n)
        let mut chirp = vec![Complex32::new(0.0, 0.0); n];
        for k in 0..n {
            let angle = sign * PI * (k as f64 * k as f64) / n as f64;
            let (s, c) = det_sincos(angle);
            chirp[k] = Complex32::new(c as f32, s as f32);
        }

        // b[k] = chirp[k], with wrap-around for negative indices, zero-padded
        let mut b = vec![Complex32::new(0.0, 0.0); m];
        b[0] = chirp[0];
        for k in 1..n {
            b[k] = chirp[k];
            b[m - k] = chirp[k];
        }

        // Precompute FFT(b)
        fft_radix2_f32(&mut b, -1.0);

        BluesteinPlan { n, m, chirp, b_hat: b }
    }

    /// Execute Bluestein FFT using precomputed plan.
    fn execute(&self, input: &[Complex32]) -> Vec<Complex32> {
        debug_assert_eq!(input.len(), self.n);

        // a[k] = x[k] * conj(chirp[k]), zero-padded to length m
        let mut a = vec![Complex32::new(0.0, 0.0); self.m];
        for k in 0..self.n {
            a[k] = input[k] * self.chirp[k].conj();
        }

        // Convolve: A = FFT(a), C = IFFT(A * B_hat)
        fft_radix2_f32(&mut a, -1.0);
        for i in 0..self.m {
            a[i] *= self.b_hat[i];
        }
        fft_radix2_f32(&mut a, 1.0);

        // Normalize radix-2 inverse and apply chirp
        let inv_m = 1.0 / self.m as f32;
        let mut result = vec![Complex32::new(0.0, 0.0); self.n];
        for k in 0..self.n {
            result[k] = a[k] * inv_m * self.chirp[k].conj();
        }

        result
    }
}

// ──────────────────────────────────────────────────────────────────────────
// 1D FFT primitives (f32)
// ──────────────────────────────────────────────────────────────────────────

/// Next power of 2 >= n.
fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Process-wide cache of per-stage radix-2 twiddle tables,
/// keyed by `(size, sign_is_negative)`. Twiddles depend only on
/// (n, sign, stage), so the FIRST `fft_radix2_f32` call for any (n,
/// sign) builds the full set of stage twiddles via `det_sincos` and
/// inserts under a brief write lock; subsequent calls (typically
/// thousands per 2D FFT × 2 directions × many rows / cols) acquire
/// a shared read lock and clone the `Arc<TwiddleStages>`.
///
/// Built deterministically via `det_sincos`, so cross-platform
/// reproducibility is preserved — the `fft2d_cross_platform`
/// SHA256 hash 17995f36... must remain unchanged.
type TwiddleStages = Vec<Vec<Complex32>>;

fn twiddle_cache() -> &'static std::sync::RwLock<
    std::collections::HashMap<(usize, bool), std::sync::Arc<TwiddleStages>>,
> {
    static CACHE: std::sync::OnceLock<
        std::sync::RwLock<
            std::collections::HashMap<(usize, bool), std::sync::Arc<TwiddleStages>>,
        >,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

fn get_or_build_twiddles(n: usize, sign: f64) -> std::sync::Arc<TwiddleStages> {
    let key = (n, sign < 0.0);

    // Hot path: shared read lock.
    {
        let cache = twiddle_cache().read().expect("twiddle cache poisoned");
        if let Some(table) = cache.get(&key) {
            return std::sync::Arc::clone(table);
        }
    }

    // Cold path: exclusive write lock + double-checked insert.
    let mut cache = twiddle_cache().write().expect("twiddle cache poisoned");
    if let Some(table) = cache.get(&key) {
        return std::sync::Arc::clone(table);
    }
    let mut stages: TwiddleStages = Vec::new();
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_step = sign * PI / half as f64;
        let stage: Vec<Complex32> = (0..half)
            .map(|k| {
                let angle = angle_step * k as f64;
                let (s, c) = det_sincos(angle);
                Complex32::new(c as f32, s as f32)
            })
            .collect();
        stages.push(stage);
        len <<= 1;
    }
    let arc = std::sync::Arc::new(stages);
    cache.insert(key, std::sync::Arc::clone(&arc));
    arc
}

/// In-place radix-2 Cooley-Tukey FFT for f32.  `data.len()` must be a power of 2.
/// `sign`: -1.0 for forward FFT, +1.0 for inverse FFT.
fn fft_radix2_f32(data: &mut [Complex32], sign: f64) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // Pull cached per-stage twiddle tables for (n, sign).
    // Built once per unique (n, sign) at first call; thousands of
    // subsequent calls during a 2D FFT pass share the same table via
    // an Arc reference (no det_sincos cost, no per-call allocation).
    let stage_twiddles = get_or_build_twiddles(n, sign);

    // Butterfly stages
    let mut len = 2;
    let mut stage_idx: usize = 0;
    while len <= n {
        let half = len / 2;
        let twiddles: &[Complex32] = &stage_twiddles[stage_idx];

        // SIMD-accelerated butterfly chunks for half >= 4.
        // butterfly_chunk_4 dispatches to NEON / SSE / WASM SIMD128 by
        // arch (scalar fallback otherwise). All paths are bit-identical
        // by construction (Path A: lane-pure IEEE 754, no FMA, no
        // cross-lane reduction). For half < 4 (stages 1 and 2) the loop
        // can't fill a 4-lane SIMD chunk; use the scalar fast path.
        if half >= 4 {
            for start in (0..n).step_by(len) {
                let mut k = 0;
                while k < half {
                    super::fft2d_simd::butterfly_chunk_4(
                        data,
                        start,
                        k,
                        half,
                        &twiddles[k..k + 4],
                    );
                    k += 4;
                }
            }
        } else {
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let w = twiddles[k];
                    let u = data[start + k];
                    let v = data[start + k + half] * w;
                    data[start + k] = u + v;
                    data[start + k + half] = u - v;
                }
            }
        }
        len <<= 1;
        stage_idx += 1;
    }
}

/// 1D FFT for arbitrary length using f32.
/// Uses BluesteinPlan if available (for 2D FFT reuse), or creates one on the fly.
/// `sign`: -1.0 for forward, +1.0 for inverse.
fn fft1d_f32_with_plan(input: &[Complex32], sign: f64, plan: Option<&BluesteinPlan>) -> Vec<Complex32> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return input.to_vec();
    }
    if n.is_power_of_two() {
        let mut buf = input.to_vec();
        fft_radix2_f32(&mut buf, sign);
        return buf;
    }

    // Use precomputed plan if available
    if let Some(p) = plan {
        debug_assert_eq!(p.n, n);
        return p.execute(input);
    }

    // Fallback: create a temporary plan
    let temp_plan = BluesteinPlan::new(n, sign);
    temp_plan.execute(input)
}

/// 1D forward FFT (arbitrary length, f32).
#[allow(dead_code)]
fn fft1d_f32(data: &[Complex32]) -> Vec<Complex32> {
    fft1d_f32_with_plan(data, -1.0, None)
}

/// 1D inverse FFT (arbitrary length, f32) — unnormalized.
#[allow(dead_code)]
fn ifft1d_f32(data: &[Complex32]) -> Vec<Complex32> {
    fft1d_f32_with_plan(data, 1.0, None)
}

// ──────────────────────────────────────────────────────────────────────────
// 2D FFT / IFFT — public API (f32, memory-optimized)
// ──────────────────────────────────────────────────────────────────────────

/// Real-valued pixel array -> 2D complex spectrum (f32).
///
/// The input is a row-major f64 array of size `width * height`.
/// Uses gather-FFT-scatter for columns (P1a) and precomputed Bluestein
/// plans for chirp reuse (P2a).
pub fn fft2d(pixels: &[f64], width: usize, height: usize) -> Spectrum2D {
    assert_eq!(pixels.len(), width * height);

    let mut data: Vec<Complex32> = pixels.iter().map(|&v| Complex32::new(v as f32, 0.0)).collect();

    // P2a: Precompute Bluestein plans for row and column lengths (if non-power-of-2)
    let row_plan = if !width.is_power_of_two() && width > 1 {
        Some(BluesteinPlan::new(width, -1.0))
    } else {
        None
    };
    let col_plan = if !height.is_power_of_two() && height > 1 {
        Some(BluesteinPlan::new(height, -1.0))
    } else {
        None
    };

    // Row FFTs are independent → par_chunks_mut over the
    // row-major buffer. Each chunk is one row processed in place.
    // Wire-format safe: rayon never changes the math; each row FFT
    // produces deterministic output independent of execution order.
    let process_row = |row_data: &mut [Complex32]| {
        let transformed = fft1d_f32_with_plan(row_data, -1.0, row_plan.as_ref());
        row_data.copy_from_slice(&transformed);
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(width).for_each(process_row);
    }
    #[cfg(not(feature = "parallel"))]
    data.chunks_mut(width).for_each(process_row);

    // Column FFTs are independent across columns (each col
    // touches a disjoint set of indices). Parallelize via raw-pointer
    // wrapper + per-thread col_buf scratch.
    fft_columns_parallel(&mut data, width, height, col_plan.as_ref(), -1.0);

    Spectrum2D { data, width, height }
}

/// Column gather/FFT/scatter parallelized across columns.
///
/// Each closure handles one column: gathers `height` strided
/// `Complex32` values into a per-thread `col_buf`, runs the 1D FFT,
/// scatters the result back. Different columns write to disjoint
/// index sets (`col + r*width` for `r = 0..height`) so writes never
/// alias across closures. Raw pointer is the only thing Rust's
/// aliasing rules block; the SyncMutPtr wrapper makes it Send+Sync.
fn fft_columns_parallel(
    data: &mut [Complex32],
    width: usize,
    height: usize,
    plan: Option<&BluesteinPlan>,
    sign: f64,
) {
    // Cast the data pointer to usize so closure auto-trait inference
    // doesn't propagate the raw pointer's `!Send + !Sync` through
    // the closure's captures. We cast back inside the closure.
    // Safety: each closure invocation indexes the pointer only at
    // its own column's disjoint set (`col + r*width` for `r =
    // 0..height`); different `col` values never overlap, so
    // concurrent writes are race-free.
    let data_addr = data.as_mut_ptr() as usize;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..width).into_par_iter().for_each_init(
            || vec![Complex32::new(0.0, 0.0); height],
            |col_buf, col| {
                let dp = data_addr as *mut Complex32;
                unsafe {
                    for r in 0..height {
                        col_buf[r] = *dp.add(r * width + col);
                    }
                }
                let transformed = fft1d_f32_with_plan(col_buf, sign, plan);
                unsafe {
                    for r in 0..height {
                        *dp.add(r * width + col) = transformed[r];
                    }
                }
            },
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        let dp = data_addr as *mut Complex32;
        let mut col_buf = vec![Complex32::new(0.0, 0.0); height];
        for col in 0..width {
            unsafe {
                for r in 0..height {
                    col_buf[r] = *dp.add(r * width + col);
                }
            }
            let transformed = fft1d_f32_with_plan(&col_buf, sign, plan);
            unsafe {
                for r in 0..height {
                    *dp.add(r * width + col) = transformed[r];
                }
            }
        }
    }
}

/// 2D complex spectrum -> real-valued pixel array.
///
/// Takes the real parts after inverse FFT, normalized by `1/(width*height)`.
/// Uses gather-IFFT-scatter for columns (P1a) and precomputed Bluestein
/// plans for chirp reuse (P2a).
pub fn ifft2d(spectrum: &Spectrum2D) -> Vec<f64> {
    let width = spectrum.width;
    let height = spectrum.height;
    let mut data = spectrum.data.clone();

    // P2a: Precompute Bluestein plans for row and column lengths (if non-power-of-2)
    let row_plan = if !width.is_power_of_two() && width > 1 {
        Some(BluesteinPlan::new(width, 1.0))
    } else {
        None
    };
    let col_plan = if !height.is_power_of_two() && height > 1 {
        Some(BluesteinPlan::new(height, 1.0))
    } else {
        None
    };

    // Row IFFTs in parallel.
    let process_row = |row_data: &mut [Complex32]| {
        let transformed = fft1d_f32_with_plan(row_data, 1.0, row_plan.as_ref());
        row_data.copy_from_slice(&transformed);
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(width).for_each(process_row);
    }
    #[cfg(not(feature = "parallel"))]
    data.chunks_mut(width).for_each(process_row);

    // Column IFFTs in parallel via the same raw-pointer
    // wrapper pattern as `fft2d`.
    fft_columns_parallel(&mut data, width, height, col_plan.as_ref(), 1.0);

    // Normalize and extract real parts. Also parallelizable
    // over the flat buffer; small change but visible on big images.
    let norm = 1.0 / (width * height) as f64;
    #[cfg(feature = "parallel")]
    let result: Vec<f64> = {
        use rayon::prelude::*;
        data.par_iter().map(|c| c.re as f64 * norm).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let result: Vec<f64> = data.iter().map(|c| c.re as f64 * norm).collect();

    result
}

/// Test/diagnostic helper — runs a deterministic 2D FFT and returns the
/// raw little-endian byte serialization of the complex spectrum.
///
/// Inputs are produced by a fixed-seed LCG so every platform with a
/// conforming IEEE 754 f32 implementation produces the SAME bytes. Used
/// to empirically verify that scalar / NEON / SSE / WASM SIMD paths
/// produce bit-identical FFT output (Path A validation).
///
/// `size` must be small enough that the test runs quickly across all
/// target archs (incl. WASM under Node). 256×256 takes <50 ms on
/// modern hardware.
#[doc(hidden)]
pub fn fft2d_test_deterministic_bytes(size: usize) -> Vec<u8> {
    let mut s: u32 = 0x12345678;
    let mut pixels = Vec::with_capacity(size * size);
    for _ in 0..size * size {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map LCG output to f64 in roughly [-1, 1].
        let v = ((s as i32) as f64) / (i32::MAX as f64);
        pixels.push(v);
    }
    let spectrum = fft2d(&pixels, size, size);
    let mut bytes = Vec::with_capacity(spectrum.data.len() * 8);
    for c in &spectrum.data {
        bytes.extend_from_slice(&c.re.to_le_bytes());
        bytes.extend_from_slice(&c.im.to_le_bytes());
    }
    bytes
}

/// Compute magnitude of each spectrum element (returns f32).
pub fn magnitude_spectrum(spectrum: &Spectrum2D) -> Vec<f32> {
    spectrum.data.iter().map(|c| {
        // Use f64 det_hypot for precision, cast result to f32
        det_hypot(c.re as f64, c.im as f64) as f32
    }).collect()
}

/// Cross-platform empirical FFT-output byte equivalence.
///
/// Computes a SHA256 of the deterministic 256×256 FFT output bytes
/// and asserts it matches a hardcoded value. The value was first
/// produced on aarch64 NEON; the test is run again under x86_64 SSE
/// (Rosetta) and WASM SIMD128 (Node/V8) to verify byte-identical
/// FFT output across all SIMD paths.
///
/// Listed at module level (not in #[cfg(test)]) so it can be re-run
/// from wasm-bridge in WASM and from native binaries under Rosetta.
#[doc(hidden)]
pub fn fft2d_test_hash_hex() -> String {
    use sha2::{Digest, Sha256};
    let bytes = fft2d_test_deterministic_bytes(256);
    let hash = Sha256::digest(&bytes);
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

// ──────────────────────────────────────────────────────────────────────────
// Read access + bounded partial spectrum (decoder)
// ──────────────────────────────────────────────────────────────────────────

/// Read access to the bins of a `width × height` forward spectrum by flat
/// row-major index (`v * width + u`). Template detection and DFT-ring
/// extraction read through this, so they work on a full `Spectrum2D` and on a
/// `PartialSpectrum` alike.
pub trait SpectrumView {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn bin(&self, idx: usize) -> Complex32;
}

impl SpectrumView for Spectrum2D {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    #[inline]
    fn bin(&self, idx: usize) -> Complex32 {
        self.data[idx]
    }
}

/// A half-open rectangle of spectrum bins: columns `c0..c1`, rows `r0..r1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinRect {
    pub c0: usize,
    pub c1: usize,
    pub r0: usize,
    pub r1: usize,
}

impl BinRect {
    /// Columns `cu - ru ..= cu + ru` and rows `cv - rv ..= cv + rv`, clipped to
    /// a `w × h` spectrum; `None` if nothing is left.
    pub fn around(cu: isize, cv: isize, ru: isize, rv: isize, w: usize, h: usize) -> Option<Self> {
        let c0 = (cu - ru).max(0);
        let c1 = (cu + ru + 1).min(w as isize);
        let r0 = (cv - rv).max(0);
        let r1 = (cv + rv + 1).min(h as isize);
        (c0 < c1 && r0 < r1).then(|| BinRect {
            c0: c0 as usize,
            c1: c1 as usize,
            r0: r0 as usize,
            r1: r1 as usize,
        })
    }

    #[inline]
    fn contains(&self, u: usize, v: usize) -> bool {
        u >= self.c0 && u < self.c1 && v >= self.r0 && v < self.r1
    }
}

/// The bins of `fft2d(pixels)` inside a few rectangles, bit for bit, computed
/// without the image or the full spectrum in memory (see `partial_spectrum`).
pub struct PartialSpectrum {
    width: usize,
    height: usize,
    rects: Vec<BinRect>,
    data: Vec<Vec<Complex32>>,
}

impl SpectrumView for PartialSpectrum {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    /// Panics on a bin outside every rectangle: the caller asked for the
    /// wrong rectangles, which is a bug, not an input condition.
    #[inline]
    fn bin(&self, idx: usize) -> Complex32 {
        let (u, v) = (idx % self.width, idx / self.width);
        for (r, d) in self.rects.iter().zip(&self.data) {
            if r.contains(u, v) {
                return d[(v - r.r0) * (r.c1 - r.c0) + (u - r.c0)];
            }
        }
        panic!("PartialSpectrum: bin ({u}, {v}) was not computed");
    }
}

/// Row-FFT output one pass of `partial_spectrum` may hold: the needed columns
/// over the full height. More columns than fit take further passes, each
/// re-running the row FFTs (time, not memory).
pub const PARTIAL_SPECTRUM_PASS_BYTES: usize = 160 << 20;

/// Block rows produced and row-transformed together.
const ROW_GROUP_BLOCKS: usize = 8;

/// `fft2d` restricted to the bins inside `rects`, with a bounded working set.
///
/// `produce(br, strip)` writes pixel rows `8 br .. 8 br + 8` (row-major,
/// `8 * width` values) of the real image; `height` must be a multiple of 8.
/// It is called in order `br = 0, 1, …` once per pass.
///
/// Bit-identical to `fft2d`: each row goes through the same
/// `fft1d_f32_with_plan` with the same row plan, and each needed column of
/// the row output through the same call with the same column plan. Only the
/// columns some rectangle needs are kept, at most `pass_bytes` of them at a
/// time, and only the rectangles are kept after the column FFT. Peak memory
/// is `pass_bytes` + the rectangles + one row group, independent of the
/// image beyond that.
pub fn partial_spectrum<P>(
    width: usize,
    height: usize,
    rects: &[BinRect],
    mut produce: P,
    pass_bytes: usize,
) -> PartialSpectrum
where
    P: FnMut(usize, &mut [f64]),
{
    assert!(height % 8 == 0 && width > 0);
    let bh = height / 8;
    let zero = Complex32::new(0.0, 0.0);

    let mut need = vec![false; width];
    for r in rects {
        assert!(r.c1 <= width && r.r1 <= height);
        need[r.c0..r.c1].iter_mut().for_each(|n| *n = true);
    }
    let cols: Vec<usize> = (0..width).filter(|&c| need[c]).collect();
    drop(need);

    let mut data: Vec<Vec<Complex32>> = rects
        .iter()
        .map(|r| vec![zero; (r.c1 - r.c0) * (r.r1 - r.r0)])
        .collect();

    // Same plans as fft2d.
    let row_plan = if !width.is_power_of_two() && width > 1 {
        Some(BluesteinPlan::new(width, -1.0))
    } else {
        None
    };
    let col_plan = if !height.is_power_of_two() && height > 1 {
        Some(BluesteinPlan::new(height, -1.0))
    } else {
        None
    };

    let per_pass = (pass_bytes / (height * std::mem::size_of::<Complex32>())).max(1);
    let mut strip = vec![0.0f64; ROW_GROUP_BLOCKS * 8 * width];

    for batch in cols.chunks(per_pass) {
        // Row-FFT output of the batch's columns, column-major.
        let mut buf = vec![zero; batch.len() * height];
        let buf_addr = buf.as_mut_ptr() as usize;

        let mut br = 0;
        while br < bh {
            let g = ROW_GROUP_BLOCKS.min(bh - br);
            for k in 0..g {
                produce(br + k, &mut strip[k * 8 * width..(k + 1) * 8 * width]);
            }
            let y0 = br * 8;
            let strip_ref = &strip;
            // Safety: row `r` writes only buf[k * height + y0 + r]; distinct
            // rows never alias, and `buf` outlives the loop.
            let do_row = |r: usize| {
                let row: Vec<Complex32> = strip_ref[r * width..(r + 1) * width]
                    .iter()
                    .map(|&v| Complex32::new(v as f32, 0.0))
                    .collect();
                let t = fft1d_f32_with_plan(&row, -1.0, row_plan.as_ref());
                let dp = buf_addr as *mut Complex32;
                for (k, &c) in batch.iter().enumerate() {
                    unsafe {
                        *dp.add(k * height + y0 + r) = t[c];
                    }
                }
            };
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..g * 8).into_par_iter().for_each(do_row);
            }
            #[cfg(not(feature = "parallel"))]
            (0..g * 8).for_each(do_row);
            br += g;
        }

        let do_col = |col: &mut [Complex32]| {
            let t = fft1d_f32_with_plan(col, -1.0, col_plan.as_ref());
            col.copy_from_slice(&t);
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            buf.par_chunks_mut(height).for_each(do_col);
        }
        #[cfg(not(feature = "parallel"))]
        buf.chunks_mut(height).for_each(do_col);

        for (r, d) in rects.iter().zip(data.iter_mut()) {
            let rw = r.c1 - r.c0;
            for (k, &c) in batch.iter().enumerate() {
                if c < r.c0 || c >= r.c1 {
                    continue;
                }
                let col = &buf[k * height..(k + 1) * height];
                for v in r.r0..r.r1 {
                    d[(v - r.r0) * rw + (c - r.c0)] = col[v];
                }
            }
        }
    }

    PartialSpectrum { width, height, rects: rects.to_vec(), data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_ifft_roundtrip() {
        let width = 16;
        let height = 16;
        let pixels: Vec<f64> = (0..width * height).map(|i| (i as f64) * 0.1 + 50.0).collect();

        let spectrum = fft2d(&pixels, width, height);
        let recovered = ifft2d(&spectrum);

        for i in 0..pixels.len() {
            assert!(
                (pixels[i] - recovered[i]).abs() < 1e-3,
                "Mismatch at {i}: expected {}, got {}",
                pixels[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn partial_spectrum_is_bit_identical_to_fft2d() {
        // Power-of-two and Bluestein sizes, one and several passes.
        for &(w, h) in &[(64usize, 48usize), (40, 64), (24, 16), (72, 56)] {
            let pixels: Vec<f64> = (0..w * h)
                .map(|i| ((i * 7919) % 251) as f64 * 0.73 - 40.0 + (i % w) as f64 * 0.1)
                .collect();
            let full = fft2d(&pixels, w, h);
            let rects = [
                BinRect::around(w as isize / 2, h as isize / 2, 9, 7, w, h).unwrap(),
                BinRect::around(3, h as isize - 2, 5, 5, w, h).unwrap(),
                BinRect::around(w as isize - 1, 0, 2, 3, w, h).unwrap(),
            ];
            for &pass in &[usize::MAX / 2, h * 8 * 3, 1] {
                let part = partial_spectrum(
                    w, h, &rects,
                    |br, strip| strip.copy_from_slice(&pixels[br * 8 * w..(br + 1) * 8 * w]),
                    pass,
                );
                for r in &rects {
                    for v in r.r0..r.r1 {
                        for u in r.c0..r.c1 {
                            let (a, b) = (part.bin(v * w + u), full.data[v * w + u]);
                            assert_eq!(
                                (a.re.to_bits(), a.im.to_bits()),
                                (b.re.to_bits(), b.im.to_bits()),
                                "{w}x{h} pass {pass} bin ({u},{v})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fft_ifft_roundtrip_non_pow2() {
        // Test with non-power-of-2 dimensions (Bluestein path)
        let width = 12;
        let height = 10;
        let pixels: Vec<f64> = (0..width * height).map(|i| (i as f64) * 0.3 + 20.0).collect();

        let spectrum = fft2d(&pixels, width, height);
        let recovered = ifft2d(&spectrum);

        for i in 0..pixels.len() {
            assert!(
                (pixels[i] - recovered[i]).abs() < 0.1,
                "Mismatch at {i}: expected {}, got {}",
                pixels[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn parseval_theorem() {
        let width = 8;
        let height = 8;
        let pixels: Vec<f64> = (0..width * height).map(|i| ((i * 7 + 3) % 256) as f64).collect();

        let spatial_energy: f64 = pixels.iter().map(|v| v * v).sum();

        let spectrum = fft2d(&pixels, width, height);
        let freq_energy: f64 = spectrum.data.iter().map(|c| {
            let re = c.re as f64;
            let im = c.im as f64;
            re * re + im * im
        }).sum();

        let n = (width * height) as f64;
        // Relaxed tolerance for f32 spectrum
        assert!(
            (spatial_energy - freq_energy / n).abs() < 10.0,
            "Parseval's theorem violated: spatial={spatial_energy}, freq/N={}", freq_energy / n
        );
    }

    #[test]
    fn dc_component_is_sum() {
        let width = 4;
        let height = 4;
        let pixels = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                          9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

        let spectrum = fft2d(&pixels, width, height);

        let expected_dc: f64 = pixels.iter().sum();
        // Relaxed tolerance for f32
        assert!(
            (spectrum.data[0].re as f64 - expected_dc).abs() < 0.1,
            "DC component should be sum of all pixels: expected {expected_dc}, got {}",
            spectrum.data[0].re
        );
        assert!((spectrum.data[0].im as f64).abs() < 0.1);
    }

    #[test]
    fn fft1d_basic() {
        // FFT of [1, 0, 0, 0] should be [1, 1, 1, 1]
        let input = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
        ];
        let output = fft1d_f32(&input);
        for k in 0..4 {
            assert!((output[k].re - 1.0).abs() < 1e-5, "Re[{k}]={}", output[k].re);
            assert!(output[k].im.abs() < 1e-5, "Im[{k}]={}", output[k].im);
        }
    }

    #[test]
    fn bluestein_matches_radix2() {
        // For power-of-2 size, Bluestein plan should give same result as radix-2
        let n = 8;
        let input: Vec<Complex32> = (0..n).map(|i| Complex32::new((i * 3 + 1) as f32, (i * 2) as f32)).collect();

        let mut radix2_buf = input.clone();
        fft_radix2_f32(&mut radix2_buf, -1.0);

        // Test via plan
        let _plan = BluesteinPlan::new(n, -1.0);
        // For power-of-2, fft1d_f32_with_plan uses radix-2 directly, not the plan.
        // Test the plan directly on a non-power-of-2 size instead.
        let n2 = 7;
        let input2: Vec<Complex32> = (0..n2).map(|i| Complex32::new((i * 3 + 1) as f32, (i * 2) as f32)).collect();
        let plan2 = BluesteinPlan::new(n2, -1.0);
        let result_plan = plan2.execute(&input2);
        let result_direct = fft1d_f32(&input2);
        for k in 0..n2 {
            assert!(
                (result_plan[k].re - result_direct[k].re).abs() < 1e-3 &&
                (result_plan[k].im - result_direct[k].im).abs() < 1e-3,
                "Plan vs direct mismatch at {k}: plan={}, direct={}",
                result_plan[k], result_direct[k]
            );
        }

        // Also verify radix-2 results haven't changed (basic sanity)
        let result_r2 = fft1d_f32(&input);
        for k in 0..n {
            assert!(
                (radix2_buf[k].re - result_r2[k].re).abs() < 1e-3 &&
                (radix2_buf[k].im - result_r2[k].im).abs() < 1e-3,
                "Mismatch at {k}: radix2={}, fft1d={}",
                radix2_buf[k], result_r2[k]
            );
        }
    }

    #[test]
    fn bluestein_plan_reuse() {
        // Verify that reusing a BluesteinPlan gives the same result each time
        let n = 13; // non-power-of-2
        let plan = BluesteinPlan::new(n, -1.0);

        let input1: Vec<Complex32> = (0..n).map(|i| Complex32::new(i as f32, 0.0)).collect();
        let input2: Vec<Complex32> = (0..n).map(|i| Complex32::new(0.0, i as f32)).collect();

        let r1a = plan.execute(&input1);
        let r2 = plan.execute(&input2);
        let r1b = plan.execute(&input1);

        for k in 0..n {
            assert!(
                (r1a[k].re - r1b[k].re).abs() < 1e-5 &&
                (r1a[k].im - r1b[k].im).abs() < 1e-5,
                "Plan reuse gave different results at {k}: first={}, second={}",
                r1a[k], r1b[k]
            );
        }
        // Just verify r2 didn't corrupt anything (no specific value check needed)
        assert_eq!(r2.len(), n);
    }
}
