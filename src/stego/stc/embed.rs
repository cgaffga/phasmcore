// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Viterbi-based STC embedding.
//!
//! Implements the forward (Viterbi) and backward (traceback) passes of the
//! STC embedding algorithm. The encoder finds the minimum-cost stego bit
//! sequence whose syndrome (under the H-hat matrix) matches the message.
//!
//! Two internal paths:
//! - **Inline** (n ≤ 1M): stores all back pointers in one pass — fastest.
//! - **Segmented** (n > 1M): checkpoint/recompute approach — O(√n) memory,
//!   2× compute. Enables 48 MP+ images on memory-constrained devices.

use super::hhat;
use super::extract::stc_extract;
use crate::stego::progress;

/// T3.6 — state-axis-vectorizable Viterbi step kernel.
///
/// Computes one j-step of the Viterbi forward recursion across all
/// `num_states` states. Replaces the legacy
///   `for s in 0..num_states { ... if cost_1 < cost_0 ... packed_bp |= ... }`
/// pattern with three vectorizable passes:
///
///   1. **gather** — `prev_cost_perm[s] = prev_cost[s ^ col]`. Cheap
///      128 loads + 128 stores per j; eliminates the XOR-indexed
///      gather from the hot loop.
///   2. **min**    — `curr_cost[s] = (prev_cost[s] + cost_s0).min(
///                                   prev_cost_perm[s] + cost_s1)`
///      and write the comparison mask `(c1 < c0) as u8` into
///      `bp_byte[s]`. Branchless, autovec-friendly SAXPY shape.
///   3. **pack**   — pack 128 mask bytes into a `u128` `packed_bp`.
///
/// Bit-exact with the legacy in-line kernel: same `add`s, same
/// `<` test (predictable tie-break on equality → cost_0 wins, matches
/// the original `if cost_1 < cost_0 { take cost_1 }`).
///
/// `bp_byte` and `prev_cost_perm` are caller-owned scratch reused
/// across all j iterations (zero per-step alloc).
///
/// Production callers dispatch to platform-specific intrinsic
/// variants in T3.6.B–D via the [`viterbi_step`] entry below.
#[inline]
fn viterbi_step_scalar(
    prev_cost: &[f64],
    curr_cost: &mut [f64],
    prev_cost_perm: &mut [f64],
    bp_byte: &mut [u8],
    col: usize,
    cost_s0: f64,
    cost_s1: f64,
) -> u128 {
    let num_states = prev_cost.len();
    // Pass 1: gather. `prev_cost[s ^ col]` → contiguous-indexed.
    for s in 0..num_states {
        prev_cost_perm[s] = prev_cost[s ^ col];
    }
    // Pass 2: SAXPY + branchless select. Use the `<` comparison
    // explicitly (matches the legacy `if cost_1 < cost_0 { ... }`
    // tie-break: equal values pick cost_0). Avoiding `f64::min`
    // sidesteps its impl-defined NaN/equal handling, which would
    // differ from the SIMD `vminq_f64` / `_mm_min_pd` / `f64x2_min`
    // intrinsics in B/C/D.
    for s in 0..num_states {
        let c0 = prev_cost[s] + cost_s0;
        let c1 = prev_cost_perm[s] + cost_s1;
        let take_c1 = c1 < c0;
        curr_cost[s] = if take_c1 { c1 } else { c0 };
        bp_byte[s] = take_c1 as u8;
    }
    // Pass 3: pack 128 mask bytes into a single u128.
    let mut packed_bp = 0u128;
    for s in 0..num_states {
        packed_bp |= (bp_byte[s] as u128) << s;
    }
    packed_bp
}

/// Same kernel as [`viterbi_step_scalar`] but skips the bit-pack
/// pass — used by the segmented variant's Phase A forward scan
/// where back pointers are reconstructed in Phase B.
#[inline]
fn viterbi_step_scalar_no_bp(
    prev_cost: &[f64],
    curr_cost: &mut [f64],
    prev_cost_perm: &mut [f64],
    col: usize,
    cost_s0: f64,
    cost_s1: f64,
) {
    let num_states = prev_cost.len();
    for s in 0..num_states {
        prev_cost_perm[s] = prev_cost[s ^ col];
    }
    for s in 0..num_states {
        let c0 = prev_cost[s] + cost_s0;
        let c1 = prev_cost_perm[s] + cost_s1;
        let take_c1 = c1 < c0;
        curr_cost[s] = if take_c1 { c1 } else { c0 };
    }
}

/// T3.6.B — NEON 2-lane f64 intrinsic kernel (aarch64).
///
/// Same algorithmic shape as `viterbi_step_scalar`. The pass-1
/// XOR gather stays scalar (128 random-ish reads from a 1 KB
/// `prev_cost` array — all in L1, no SIMD shuffle needed). Pass 2
/// (the SAXPY-then-select hot path) is 2-lane SIMD via `vaddq_f64`
/// / `vcltq_f64` / `vbslq_f64`. Pass 3 bit-pack extracts the
/// comparison mask per pair and ORs into the `u128` accumulator.
///
/// Byte-identical to scalar: same per-lane add, same `<` compare,
/// same select (mask ? c1 : c0). Wire-safety preserved.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn viterbi_step_neon(
    prev_cost: &[f64],
    curr_cost: &mut [f64],
    prev_cost_perm: &mut [f64],
    col: usize,
    cost_s0: f64,
    cost_s1: f64,
) -> u128 {
    use core::arch::aarch64::*;
    let num_states = prev_cost.len();
    // Pass 1: scalar gather.
    for s in 0..num_states {
        prev_cost_perm[s] = prev_cost[s ^ col];
    }
    // Pass 2+3: SIMD SAXPY + select + bit-pack.
    let v_cost_s0 = vdupq_n_f64(cost_s0);
    let v_cost_s1 = vdupq_n_f64(cost_s1);
    let mut packed_bp: u128 = 0;
    let chunks = num_states / 2;
    let prev_ptr = prev_cost.as_ptr();
    let perm_ptr = prev_cost_perm.as_ptr();
    let curr_ptr = curr_cost.as_mut_ptr();
    for chunk in 0..chunks {
        let s = chunk * 2;
        let v_prev = vld1q_f64(prev_ptr.add(s));
        let v_perm = vld1q_f64(perm_ptr.add(s));
        let v_c0 = vaddq_f64(v_prev, v_cost_s0);
        let v_c1 = vaddq_f64(v_perm, v_cost_s1);
        // c1 < c0 per lane → all-1 mask on true, all-0 on false.
        let cmp_mask = vcltq_f64(v_c1, v_c0);
        // vbslq_f64(mask, a, b): mask-bit-1 → a, else b. So select c1
        // when c1 < c0 (mask=all-1), c0 otherwise.
        let v_curr = vbslq_f64(cmp_mask, v_c1, v_c0);
        vst1q_f64(curr_ptr.add(s), v_curr);
        // Bit-pack: vcltq_f64 lanes are 0 or u64::MAX. Extract per
        // lane and OR into packed_bp at bit `s` / `s+1`.
        let lane0 = vgetq_lane_u64(cmp_mask, 0);
        let lane1 = vgetq_lane_u64(cmp_mask, 1);
        packed_bp |= ((lane0 != 0) as u128) << s;
        packed_bp |= ((lane1 != 0) as u128) << (s + 1);
    }
    packed_bp
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn viterbi_step_neon_no_bp(
    prev_cost: &[f64],
    curr_cost: &mut [f64],
    prev_cost_perm: &mut [f64],
    col: usize,
    cost_s0: f64,
    cost_s1: f64,
) {
    use core::arch::aarch64::*;
    let num_states = prev_cost.len();
    for s in 0..num_states {
        prev_cost_perm[s] = prev_cost[s ^ col];
    }
    let v_cost_s0 = vdupq_n_f64(cost_s0);
    let v_cost_s1 = vdupq_n_f64(cost_s1);
    let chunks = num_states / 2;
    let prev_ptr = prev_cost.as_ptr();
    let perm_ptr = prev_cost_perm.as_ptr();
    let curr_ptr = curr_cost.as_mut_ptr();
    for chunk in 0..chunks {
        let s = chunk * 2;
        let v_prev = vld1q_f64(prev_ptr.add(s));
        let v_perm = vld1q_f64(perm_ptr.add(s));
        let v_c0 = vaddq_f64(v_prev, v_cost_s0);
        let v_c1 = vaddq_f64(v_perm, v_cost_s1);
        let cmp_mask = vcltq_f64(v_c1, v_c0);
        let v_curr = vbslq_f64(cmp_mask, v_c1, v_c0);
        vst1q_f64(curr_ptr.add(s), v_curr);
    }
}

/// T3.6.D — WASM SIMD128 2-lane f64 intrinsic kernel (wasm32).
///
/// Phasm's wasm bridges enable `+simd128` in `.cargo/config.toml`,
/// so this gate fires for the deployed phasm.app build. Matches the
/// NEON / SSE2 pair-wise sum order byte-for-byte.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn viterbi_step_simd128(
    prev_cost: &[f64],
    curr_cost: &mut [f64],
    prev_cost_perm: &mut [f64],
    col: usize,
    cost_s0: f64,
    cost_s1: f64,
) -> u128 {
    use core::arch::wasm32::*;
    let num_states = prev_cost.len();
    for s in 0..num_states {
        prev_cost_perm[s] = prev_cost[s ^ col];
    }
    let v_cost_s0 = f64x2_splat(cost_s0);
    let v_cost_s1 = f64x2_splat(cost_s1);
    let mut packed_bp: u128 = 0;
    let chunks = num_states / 2;
    let prev_ptr = prev_cost.as_ptr();
    let perm_ptr = prev_cost_perm.as_ptr();
    let curr_ptr = curr_cost.as_mut_ptr();
    for chunk in 0..chunks {
        let s = chunk * 2;
        let v_prev = v128_load(prev_ptr.add(s) as *const v128);
        let v_perm = v128_load(perm_ptr.add(s) as *const v128);
        let v_c0 = f64x2_add(v_prev, v_cost_s0);
        let v_c1 = f64x2_add(v_perm, v_cost_s1);
        let cmp_mask = f64x2_lt(v_c1, v_c0);
        // v128_bitselect(a, b, mask): mask-bit-1 → a, else b.
        let v_curr = v128_bitselect(v_c1, v_c0, cmp_mask);
        v128_store(curr_ptr.add(s) as *mut v128, v_curr);
        // i64x2_bitmask extracts the high bit (sign) of each lane.
        // For an all-1 (true) lane, high bit = 1 → match
        // _mm_movemask_pd / vbslq mask-extraction semantics.
        let mask_bits = i64x2_bitmask(cmp_mask) as u128;
        packed_bp |= mask_bits << s;
    }
    packed_bp
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn viterbi_step_simd128_no_bp(
    prev_cost: &[f64],
    curr_cost: &mut [f64],
    prev_cost_perm: &mut [f64],
    col: usize,
    cost_s0: f64,
    cost_s1: f64,
) {
    use core::arch::wasm32::*;
    let num_states = prev_cost.len();
    for s in 0..num_states {
        prev_cost_perm[s] = prev_cost[s ^ col];
    }
    let v_cost_s0 = f64x2_splat(cost_s0);
    let v_cost_s1 = f64x2_splat(cost_s1);
    let chunks = num_states / 2;
    let prev_ptr = prev_cost.as_ptr();
    let perm_ptr = prev_cost_perm.as_ptr();
    let curr_ptr = curr_cost.as_mut_ptr();
    for chunk in 0..chunks {
        let s = chunk * 2;
        let v_prev = v128_load(prev_ptr.add(s) as *const v128);
        let v_perm = v128_load(perm_ptr.add(s) as *const v128);
        let v_c0 = f64x2_add(v_prev, v_cost_s0);
        let v_c1 = f64x2_add(v_perm, v_cost_s1);
        let cmp_mask = f64x2_lt(v_c1, v_c0);
        let v_curr = v128_bitselect(v_c1, v_c0, cmp_mask);
        v128_store(curr_ptr.add(s) as *mut v128, v_curr);
    }
}

/// T3.6.C — SSE2 / SSE4.1 2-lane f64 intrinsic kernel (x86_64).
///
/// Same pair-wise lane layout as NEON. `_mm_cmplt_pd` returns
/// all-1 mask lanes (negative f64 bit-pattern) for true; bit-pack
/// uses `_mm_movemask_pd` which extracts the 2 sign bits directly
/// — cleanest mask extraction across the three SIMD ISAs.
///
/// Phasm's `.cargo/config.toml` enables `+sse4.1` baseline for
/// x86_64, so `_mm_blendv_pd` is available for the select step.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn viterbi_step_sse(
    prev_cost: &[f64],
    curr_cost: &mut [f64],
    prev_cost_perm: &mut [f64],
    col: usize,
    cost_s0: f64,
    cost_s1: f64,
) -> u128 {
    use core::arch::x86_64::*;
    let num_states = prev_cost.len();
    for s in 0..num_states {
        prev_cost_perm[s] = prev_cost[s ^ col];
    }
    let v_cost_s0 = _mm_set1_pd(cost_s0);
    let v_cost_s1 = _mm_set1_pd(cost_s1);
    let mut packed_bp: u128 = 0;
    let chunks = num_states / 2;
    let prev_ptr = prev_cost.as_ptr();
    let perm_ptr = prev_cost_perm.as_ptr();
    let curr_ptr = curr_cost.as_mut_ptr();
    for chunk in 0..chunks {
        let s = chunk * 2;
        let v_prev = _mm_loadu_pd(prev_ptr.add(s));
        let v_perm = _mm_loadu_pd(perm_ptr.add(s));
        let v_c0 = _mm_add_pd(v_prev, v_cost_s0);
        let v_c1 = _mm_add_pd(v_perm, v_cost_s1);
        let cmp_mask = _mm_cmplt_pd(v_c1, v_c0); // c1 < c0 → all-1 lanes
        // _mm_blendv_pd(a, b, mask): mask-bit-1 → b, else a.
        let v_curr = _mm_blendv_pd(v_c0, v_c1, cmp_mask);
        _mm_storeu_pd(curr_ptr.add(s), v_curr);
        // movmskpd returns 2-bit integer (0b{lane1}{lane0}).
        let mask_bits = _mm_movemask_pd(cmp_mask) as u128;
        packed_bp |= mask_bits << s;
    }
    packed_bp
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn viterbi_step_sse_no_bp(
    prev_cost: &[f64],
    curr_cost: &mut [f64],
    prev_cost_perm: &mut [f64],
    col: usize,
    cost_s0: f64,
    cost_s1: f64,
) {
    use core::arch::x86_64::*;
    let num_states = prev_cost.len();
    for s in 0..num_states {
        prev_cost_perm[s] = prev_cost[s ^ col];
    }
    let v_cost_s0 = _mm_set1_pd(cost_s0);
    let v_cost_s1 = _mm_set1_pd(cost_s1);
    let chunks = num_states / 2;
    let prev_ptr = prev_cost.as_ptr();
    let perm_ptr = prev_cost_perm.as_ptr();
    let curr_ptr = curr_cost.as_mut_ptr();
    for chunk in 0..chunks {
        let s = chunk * 2;
        let v_prev = _mm_loadu_pd(prev_ptr.add(s));
        let v_perm = _mm_loadu_pd(perm_ptr.add(s));
        let v_c0 = _mm_add_pd(v_prev, v_cost_s0);
        let v_c1 = _mm_add_pd(v_perm, v_cost_s1);
        let cmp_mask = _mm_cmplt_pd(v_c1, v_c0);
        let v_curr = _mm_blendv_pd(v_c0, v_c1, cmp_mask);
        _mm_storeu_pd(curr_ptr.add(s), v_curr);
    }
}

/// Platform-dispatching entry point. Selects the best available
/// kernel at compile time via `target_feature` cfg gates.
#[inline]
fn viterbi_step(
    prev_cost: &[f64],
    curr_cost: &mut [f64],
    prev_cost_perm: &mut [f64],
    bp_byte: &mut [u8],
    col: usize,
    cost_s0: f64,
    cost_s1: f64,
) -> u128 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return viterbi_step_neon(prev_cost, curr_cost, prev_cost_perm, col, cost_s0, cost_s1);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return viterbi_step_sse(prev_cost, curr_cost, prev_cost_perm, col, cost_s0, cost_s1);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        return viterbi_step_simd128(prev_cost, curr_cost, prev_cost_perm, col, cost_s0, cost_s1);
    }
    #[allow(unreachable_code)]
    viterbi_step_scalar(prev_cost, curr_cost, prev_cost_perm, bp_byte, col, cost_s0, cost_s1)
}

#[inline]
fn viterbi_step_no_bp(
    prev_cost: &[f64],
    curr_cost: &mut [f64],
    prev_cost_perm: &mut [f64],
    col: usize,
    cost_s0: f64,
    cost_s1: f64,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return viterbi_step_neon_no_bp(prev_cost, curr_cost, prev_cost_perm, col, cost_s0, cost_s1);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return viterbi_step_sse_no_bp(prev_cost, curr_cost, prev_cost_perm, col, cost_s0, cost_s1);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        return viterbi_step_simd128_no_bp(prev_cost, curr_cost, prev_cost_perm, col, cost_s0, cost_s1);
    }
    #[allow(unreachable_code)]
    viterbi_step_scalar_no_bp(prev_cost, curr_cost, prev_cost_perm, col, cost_s0, cost_s1)
}

/// Perf-bench helpers (T3.6). doc-hidden, pub so the
/// `perf_t36_viterbi` integration test can compare paths.
#[doc(hidden)]
pub fn perf_legacy_viterbi_step(
    prev_cost: &[f64],
    curr_cost: &mut [f64],
    col: usize,
    cost_s0: f64,
    cost_s1: f64,
) -> u128 {
    // Replica of the pre-T3.6 inline kernel for before/after timing.
    let num_states = prev_cost.len();
    let mut packed_bp = 0u128;
    for s in 0..num_states {
        let cost_0 = prev_cost[s] + cost_s0;
        let cost_1 = prev_cost[s ^ col] + cost_s1;
        if cost_1 < cost_0 {
            curr_cost[s] = cost_1;
            packed_bp |= 1u128 << s;
        } else {
            curr_cost[s] = cost_0;
        }
    }
    packed_bp
}

#[doc(hidden)]
pub fn perf_fast_viterbi_step(
    prev_cost: &[f64],
    curr_cost: &mut [f64],
    prev_cost_perm: &mut [f64],
    bp_byte: &mut [u8],
    col: usize,
    cost_s0: f64,
    cost_s1: f64,
) -> u128 {
    viterbi_step(prev_cost, curr_cost, prev_cost_perm, bp_byte, col, cost_s0, cost_s1)
}

/// T3.6.E — deterministic byte stream produced by the Viterbi
/// kernel on a fixed fixture, used by
/// `viterbi_test_hash_hex` to pin the cross-platform hash. Re-
/// computing this on any supported target MUST yield the same bytes
/// (and therefore the same SHA256). Covers scalar + NEON + SSE2 +
/// SIMD128 paths.
///
/// Fixture: deterministic LCG-generated cover bits + costs + message
/// + H-hat matrix, n=512, m=64, h=7, w=8. Output dumps every
/// `prev_cost` snapshot (after each j step) + the final packed back-
/// pointer bytes, ~64 KB total per fixture.
#[doc(hidden)]
pub fn viterbi_test_deterministic_bytes() -> Vec<u8> {
    let h: usize = 7;
    let w: usize = 8;
    let m: usize = 64;
    let n: usize = m * w; // 512
    let num_states = 1usize << h;

    // Deterministic inputs via LCG.
    let mut s: u32 = 0xC0FFEE_42;
    let mut step = || -> u8 {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 24) as u8
    };
    let cover_bits: Vec<u8> = (0..n).map(|_| step() & 1).collect();
    let costs: Vec<f32> = (0..n)
        .map(|_| {
            let raw = step() as f32;
            // Spread costs in [0.5, 4.5], some WET-ish but never inf.
            0.5 + raw / 64.0
        })
        .collect();
    let message: Vec<u8> = (0..m).map(|_| step() & 1).collect();
    // Random h-hat matrix: h rows × w cols (column_packed reads
    // `hhat[r][col]` so outer is h, inner is w).
    let mut hhat_matrix: Vec<Vec<u32>> = vec![vec![0u32; w]; h];
    for row in &mut hhat_matrix {
        for v in row.iter_mut() {
            *v = (step() & 1) as u32;
        }
    }

    let columns: Vec<usize> = (0..w).map(|c| hhat::column_packed(&hhat_matrix, c) as usize).collect();
    let inf = f64::INFINITY;
    let mut prev_cost = vec![inf; num_states];
    prev_cost[0] = 0.0;
    let mut curr_cost = vec![0.0f64; num_states];
    let mut shifted_cost = vec![inf; num_states];
    let mut prev_cost_perm = vec![0.0f64; num_states];
    let mut bp_byte = vec![0u8; num_states];

    let mut bytes: Vec<u8> = Vec::with_capacity(n * (num_states * 8 + 16));
    let mut msg_idx = 0usize;

    for j in 0..n {
        let col_idx = j % w;
        let col = columns[col_idx];
        let flip_cost = costs[j] as f64;
        let cover_bit = cover_bits[j] & 1;
        let (cost_s0, cost_s1) = if cover_bit == 0 {
            (0.0, flip_cost)
        } else {
            (flip_cost, 0.0)
        };
        let packed_bp = viterbi_step(
            &prev_cost,
            &mut curr_cost,
            &mut prev_cost_perm,
            &mut bp_byte,
            col,
            cost_s0,
            cost_s1,
        );
        // Dump packed_bp (16 bytes).
        bytes.extend_from_slice(&packed_bp.to_le_bytes());

        if col_idx == w - 1 && msg_idx < m {
            let required_bit = message[msg_idx] as usize;
            shifted_cost.fill(inf);
            for s in 0..num_states {
                if curr_cost[s] == inf { continue; }
                if (s & 1) != required_bit { continue; }
                let s_shifted = s >> 1;
                if curr_cost[s] < shifted_cost[s_shifted] {
                    shifted_cost[s_shifted] = curr_cost[s];
                }
            }
            std::mem::swap(&mut prev_cost, &mut shifted_cost);
            msg_idx += 1;
        } else {
            std::mem::swap(&mut prev_cost, &mut curr_cost);
        }

        // Dump current prev_cost (num_states × 8 bytes via f64::to_bits).
        for &c in &prev_cost {
            bytes.extend_from_slice(&c.to_bits().to_le_bytes());
        }
    }

    bytes
}

/// SHA256 of [`viterbi_test_deterministic_bytes`] as lowercase hex.
#[doc(hidden)]
pub fn viterbi_test_hash_hex() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(viterbi_test_deterministic_bytes());
    let digest = h.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

/// Result of STC embedding: the stego bit sequence and its total distortion cost.
pub struct EmbedResult {
    pub stego_bits: Vec<u8>,
    pub total_cost: f64,
    /// Number of positions where cover bit != stego bit.
    pub num_modifications: usize,
}

/// Number of progress steps reported during STC Viterbi embedding.
///
/// Distributed across the forward pass(es). Post-T3.6 STC is ~5-10%
/// of total encode wall-clock on 12 MP (was 20-30%); was over-weighted
/// at 50 steps. Dropped to 15 to match the new wall-clock share — the
/// bar no longer crawls during the STC phase.
pub const STC_PROGRESS_STEPS: u32 = 15;

/// Back-pointer memory threshold (in cover positions). Above this, the
/// segmented path is used. 1M positions × 16 bytes/u128 = 16 MB.
const SEGMENTED_THRESHOLD: usize = 1_000_000;

/// Embed a message into cover bits using the Viterbi-based STC algorithm.
///
/// Automatically selects the inline path (single-pass, O(n) memory) for
/// small inputs, or the segmented path (checkpoint/recompute, O(√n) memory)
/// for large inputs. Both paths produce identical output.
///
/// - `cover_bits`: LSBs of the cover coefficients (length n)
/// - `costs`: cost of flipping each cover bit (length n). Use f32::INFINITY for WET.
///   Promoted to f64 internally for accumulation precision.
/// - `message`: message bits to embed (length m)
/// - `hhat_matrix`: the H-hat submatrix (h rows × w columns)
/// - `h`: constraint length (must be ≤ 7 so 2^h fits in u128)
/// - `w`: submatrix width (should be ceil(n/m))
///
/// Returns the stego bit sequence that encodes the message with minimum
/// distortion, or `None` if embedding is infeasible.
///
/// Reports [`STC_PROGRESS_STEPS`] progress sub-steps via [`progress::advance`]
/// during the Viterbi forward pass(es).
pub fn stc_embed(
    cover_bits: &[u8],
    costs: &[f32],
    message: &[u8],
    hhat_matrix: &[Vec<u32>],
    h: usize,
    w: usize,
) -> Option<EmbedResult> {
    // h ≤ 7 required: 2^7 = 128 states fit exactly in u128.
    if w == 0 || h > 7 {
        return None;
    }

    let n = cover_bits.len();
    let m = message.len();

    if m == 0 {
        return Some(EmbedResult {
            stego_bits: cover_bits.to_vec(),
            total_cost: 0.0,
            num_modifications: 0,
        });
    }

    if n > SEGMENTED_THRESHOLD {
        // §6E-C / Task #24.3 — route the segmented path through
        // streaming-Viterbi via the InMemoryCoverFetch adapter.
        // Bit-exact equivalent to the legacy `stc_embed_segmented`
        // (verified by `streaming_matches_inline_segmented_large` in
        // streaming_segmented.rs), so callers see no observable
        // change. The win is the STC-internal O(√n) memory bound
        // (checkpoint + back-pointer working set vs O(n) back-ptrs
        // in the inline path) — relevant for long-clip video stego.
        // A future per-GOP-replay CoverFetch adapter (v1.1+) will
        // bound cover-side memory to O(√n) too.
        use crate::stego::stc::streaming_segmented::{
            stc_embed_streaming_segmented, InMemoryCoverFetch,
        };
        let k = ((m as f64).sqrt().ceil() as usize).max(1);
        let mut cover = InMemoryCoverFetch::new(cover_bits, costs, m, w, k)?;
        stc_embed_streaming_segmented(&mut cover, message, hhat_matrix, h, w).ok()
    } else {
        stc_embed_inline(cover_bits, costs, message, hhat_matrix, h, w)
    }
}

/// `stc_embed(cover_bits, costs, message, hhat_matrix, h, w).is_some()`,
/// computed without building the plan.
///
/// ## Why this is exact
///
/// `stc_embed` reports infeasible on exactly one data condition — the
/// trellis's best terminal cost is `+∞` (both the inline and the segmented
/// path: `best_cost == inf`). In the Viterbi recursion a state's cost is
/// finite iff some path of all-finite-cost choices reaches it:
///
/// * `curr[s] = min(prev[s] + c₀, prev[s ^ col] + c₁)` is finite iff either
///   operand is, and a sum of finite non-negative-or-negative f64 terms
///   cannot overflow here (each term is an f32 widened to f64, ≤ 3.4e38, over
///   at most 2³² positions — nowhere near f64::MAX);
/// * the message-bit filter keeps finite states `s` with `s & 1 == bit` as
///   state `s >> 1`.
///
/// So the set of finite states evolves as a 2ʰ-bit set `R` — with `x` the
/// cover bit and `c` the flip cost at a position: `c` finite ⇒
/// `R ∪ (R ⊕ col)`; `c = +∞`, `x = 0` ⇒ `R`; `c = +∞`, `x = 1` ⇒ `R ⊕ col`;
/// message boundary ⇒ keep the bits with the required parity, halve the
/// index. Feasible ⇔ `R ≠ ∅` at the end. That is the same predicate the f64
/// trellis evaluates, as integer set operations: ~10× less work per
/// position and no `n × 16 B` back-pointer table (16 MB at 1 M positions).
///
/// Cost magnitudes never enter it, which is the whole point for callers that
/// only ask "does this payload fit?" (the H.264 capacity probe's bisection,
/// `openh264_stego::cap2_stc_trial_max_payload`, ~25 full trellises per GOP).
///
/// Inputs outside that argument — a `NaN` or `-∞` cost — fall back to
/// `stc_embed` itself, so the equality holds for every input.
///
/// **Cancellation and progress** follow the inline path's schedule: every
/// `max(n / STC_PROGRESS_STEPS, 1)` positions it checks
/// [`progress::is_cancelled`] (cancelled ⇒ `false`, as `stc_embed` returns
/// `None`) and calls [`progress::advance`]. The segmented path (n > 1 M)
/// checks at a different cadence, so under a concurrent cancel the two can
/// stop at different positions — both still answer "infeasible".
///
/// Pinned against `stc_embed(...).is_some()` by
/// `stc_is_feasible_matches_stc_embed_exhaustively`.
pub fn stc_is_feasible(
    cover_bits: &[u8],
    costs: &[f32],
    message: &[u8],
    hhat_matrix: &[Vec<u32>],
    h: usize,
    w: usize,
) -> bool {
    if w == 0 || h > 7 {
        return false;
    }
    let n = cover_bits.len();
    let m = message.len();
    if m == 0 {
        return true;
    }
    if costs.iter().take(n).any(|c| c.is_nan() || *c == f32::NEG_INFINITY) {
        return stc_embed(cover_bits, costs, message, hhat_matrix, h, w).is_some();
    }
    // The segmented path walks only the first m·w positions and rejects a
    // shorter cover outright (`InMemoryCoverFetch::new` → None).
    let n_walk = if n > SEGMENTED_THRESHOLD {
        if n < m * w {
            return false;
        }
        m * w
    } else {
        n
    };

    let columns: Vec<u128> = (0..w)
        .map(|c| hhat::column_packed(hhat_matrix, c) as u128)
        .collect();
    let progress_interval = (n / STC_PROGRESS_STEPS as usize).max(1);

    // Bit s of `reach` ⇔ trellis state s has finite cost.
    let mut reach: u128 = 1; // state 0, cost 0.0
    let mut msg_idx = 0usize;
    for j in 0..n_walk {
        let col_idx = j % w;
        let col = columns[col_idx];
        let finite = costs[j] != f32::INFINITY;
        reach = if finite {
            reach | xor_permute_states(reach, col)
        } else if cover_bits[j] & 1 == 0 {
            reach
        } else {
            xor_permute_states(reach, col)
        };
        if col_idx == w - 1 && msg_idx < m {
            // `stc_embed` compares `s & 1` against the raw message byte, so a
            // byte other than 0/1 matches no state at all.
            let bit = message[msg_idx];
            reach = if bit > 1 { 0 } else { keep_parity_and_halve(reach, bit as u32) };
            msg_idx += 1;
            if reach == 0 {
                // No state survives, and an empty set stays empty (permuting or
                // unioning ∅ gives ∅), so `stc_embed` would walk on and fail at
                // the end — same answer, minus the remaining progress ticks.
                return false;
            }
        }
        if (j + 1) % progress_interval == 0 {
            if progress::is_cancelled() {
                return false;
            }
            progress::advance();
        }
    }
    reach != 0
}

/// `{ s ⊕ col : s ∈ R }` for a 128-bit state set: one block swap per set bit
/// of `col` (bit k swaps every pair of 2ᵏ-sized blocks), so bit `s` lands on
/// bit `s ^ col`.
#[inline]
fn xor_permute_states(mut r: u128, col: u128) -> u128 {
    const LOW: [u128; 7] = [
        0x5555_5555_5555_5555_5555_5555_5555_5555,
        0x3333_3333_3333_3333_3333_3333_3333_3333,
        0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F,
        0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF,
        0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF,
        0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF,
        0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF,
    ];
    for (k, &low) in LOW.iter().enumerate() {
        if (col >> k) & 1 == 1 {
            let sh = 1u32 << k;
            r = ((r & low) << sh) | ((r >> sh) & low);
        }
    }
    r
}

/// The message-bit filter: keep states `s` with `s & 1 == bit`, renumbered
/// `s >> 1` (bit `2t + bit` of `r` → bit `t`).
#[inline]
fn keep_parity_and_halve(r: u128, bit: u32) -> u128 {
    let mut x = (r >> bit) & 0x5555_5555_5555_5555_5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x = (x | (x >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
    x
}

// ---------------------------------------------------------------------------
// Inline path: single forward pass, stores all back pointers.
// Best for small/medium images where O(n) memory is acceptable.
// ---------------------------------------------------------------------------

fn stc_embed_inline(
    cover_bits: &[u8],
    costs: &[f32],
    message: &[u8],
    hhat_matrix: &[Vec<u32>],
    h: usize,
    w: usize,
) -> Option<EmbedResult> {
    let n = cover_bits.len();
    let m = message.len();
    let num_states = 1usize << h;
    let inf = f64::INFINITY;

    // Pre-compute H-hat columns (avoids repeated column_packed calls).
    let columns: Vec<usize> = (0..w)
        .map(|c| hhat::column_packed(hhat_matrix, c) as usize)
        .collect();

    // Progress: advance every n/STC_PROGRESS_STEPS elements.
    let progress_interval = (n / STC_PROGRESS_STEPS as usize).max(1);

    // Forward Viterbi pass with 1-bit packed back pointers.
    // back_ptr[j] is a u128: bit s = 1 means stego_bit=1 was chosen for
    // target state s (predecessor = s ^ col). 16 bytes per step.
    //
    // Pre-allocated cost buffers avoid per-iteration heap allocations.
    // The target-state iteration writes every entry, so no fill needed
    // for curr_cost. Only shifted_cost needs fill (sparse writes).
    let mut prev_cost = vec![inf; num_states];
    prev_cost[0] = 0.0;
    let mut curr_cost = vec![0.0f64; num_states];
    let mut shifted_cost = vec![inf; num_states];
    // T3.6 — caller-owned scratch for the viterbi_step kernel.
    // Reused across all j iterations, zero per-step allocation.
    let mut prev_cost_perm = vec![0.0f64; num_states];
    let mut bp_byte = vec![0u8; num_states];

    let mut back_ptr: Vec<u128> = Vec::with_capacity(n);
    let mut msg_idx = 0;

    for j in 0..n {
        let col_idx = j % w;
        let col = columns[col_idx];
        let flip_cost = costs[j] as f64; // promote f32→f64 for accumulation
        let cover_bit = cover_bits[j] & 1;

        let (cost_s0, cost_s1) = if cover_bit == 0 {
            (0.0, flip_cost)
        } else {
            (flip_cost, 0.0)
        };

        let packed_bp = viterbi_step(
            &prev_cost,
            &mut curr_cost,
            &mut prev_cost_perm,
            &mut bp_byte,
            col,
            cost_s0,
            cost_s1,
        );

        back_ptr.push(packed_bp);

        if col_idx == w - 1 && msg_idx < m {
            let required_bit = message[msg_idx] as usize;
            shifted_cost.fill(inf);

            for s in 0..num_states {
                if curr_cost[s] == inf { continue; }
                if (s & 1) != required_bit { continue; }
                let s_shifted = s >> 1;
                if curr_cost[s] < shifted_cost[s_shifted] {
                    shifted_cost[s_shifted] = curr_cost[s];
                }
            }

            std::mem::swap(&mut prev_cost, &mut shifted_cost);
            msg_idx += 1;
        } else {
            std::mem::swap(&mut prev_cost, &mut curr_cost);
        }

        if (j + 1) % progress_interval == 0 {
            if progress::is_cancelled() { return None; }
            progress::advance();
        }
    }

    // Find terminal state with minimum cost.
    let (best_state, best_cost) = find_best_state(&prev_cost);
    if best_cost == inf { return None; }

    // Backward traceback.
    let mut stego_bits = vec![0u8; n];
    let mut s = best_state;

    for j in (0..n).rev() {
        let col_idx = j % w;

        if col_idx == w - 1 && (j / w) < m {
            let msg_bit = message[j / w] as usize;
            s = ((s << 1) | msg_bit) & (num_states - 1);
        }

        let bit = ((back_ptr[j] >> s) & 1) as u8;
        stego_bits[j] = bit;

        if bit == 1 {
            s ^= columns[col_idx];
        }
    }

    debug_assert_eq!(s, 0, "traceback did not return to initial state 0");
    debug_assert_eq!(
        stc_extract(&stego_bits, hhat_matrix, w)[..m],
        message[..m],
    );

    let num_modifications = stego_bits.iter().zip(cover_bits.iter())
        .filter(|(s, c)| s != c).count();

    Some(EmbedResult { stego_bits, total_cost: best_cost, num_modifications })
}

// ---------------------------------------------------------------------------
// Segmented path: checkpoint/recompute for O(√n) memory.
// Two forward passes: one to save checkpoints, one to recompute segments.
// ---------------------------------------------------------------------------

#[allow(dead_code)] // Superseded by stc_embed_segmented streaming variant; kept for ref.
fn stc_embed_segmented(
    cover_bits: &[u8],
    costs: &[f32],
    message: &[u8],
    hhat_matrix: &[Vec<u32>],
    h: usize,
    w: usize,
) -> Option<EmbedResult> {
    let n = cover_bits.len();
    let m = message.len();
    let num_states = 1usize << h;
    let inf = f64::INFINITY;

    // Pre-compute H-hat columns (avoids repeated column_packed calls
    // across Phase A, Phase B forward, and Phase B traceback).
    let columns: Vec<usize> = (0..w)
        .map(|c| hhat::column_packed(hhat_matrix, c) as usize)
        .collect();

    // Checkpoint interval: K message blocks per segment.
    // sqrt(m) balances checkpoint memory (K × 1 KB) with segment back_ptr
    // memory (K × w × 16 bytes).
    let k = ((m as f64).sqrt().ceil() as usize).max(1);
    let num_segments = m.div_ceil(k);

    // --- Phase A: forward scan, save checkpoints, no back_ptr ---
    // Reports half the STC progress sub-steps.
    let phase_a_steps = STC_PROGRESS_STEPS / 2;
    let progress_interval_a = (n / phase_a_steps as usize).max(1);

    // Pre-allocated cost buffers reused across all iterations.
    let mut prev_cost = vec![inf; num_states];
    prev_cost[0] = 0.0;
    let mut curr_cost = vec![0.0f64; num_states];
    let mut shifted_cost = vec![inf; num_states];
    // T3.6 — scratch for the restructured Viterbi kernel.
    let mut prev_cost_perm = vec![0.0f64; num_states];

    // checkpoint[s] = cost array at the START of segment s (post-shift from
    // the preceding block, or the initial state for s=0).
    let mut checkpoints: Vec<Vec<f64>> = Vec::with_capacity(num_segments);
    checkpoints.push(prev_cost.clone());

    let mut msg_idx = 0;

    for j in 0..n {
        let col_idx = j % w;
        let col = columns[col_idx];
        let flip_cost = costs[j] as f64; // promote f32→f64
        let cover_bit = cover_bits[j] & 1;

        let (cost_s0, cost_s1) = if cover_bit == 0 {
            (0.0, flip_cost)
        } else {
            (flip_cost, 0.0)
        };

        viterbi_step_no_bp(
            &prev_cost,
            &mut curr_cost,
            &mut prev_cost_perm,
            col,
            cost_s0,
            cost_s1,
        );

        if col_idx == w - 1 && msg_idx < m {
            let required_bit = message[msg_idx] as usize;
            shifted_cost.fill(inf);
            for s in 0..num_states {
                if curr_cost[s] == inf { continue; }
                if (s & 1) != required_bit { continue; }
                let s_shifted = s >> 1;
                if curr_cost[s] < shifted_cost[s_shifted] {
                    shifted_cost[s_shifted] = curr_cost[s];
                }
            }
            std::mem::swap(&mut prev_cost, &mut shifted_cost);
            msg_idx += 1;

            // Save checkpoint at segment boundaries.
            if msg_idx % k == 0 && msg_idx < m {
                checkpoints.push(prev_cost.clone());
            }
        } else {
            std::mem::swap(&mut prev_cost, &mut curr_cost);
        }

        if (j + 1) % progress_interval_a == 0 {
            if progress::is_cancelled() { return None; }
            progress::advance();
        }
    }

    // Find terminal state with minimum cost.
    let (best_state, best_cost) = find_best_state(&prev_cost);
    if best_cost == inf { return None; }

    // --- Phase B: segment-by-segment recomputation + traceback ---
    // Reports the remaining STC progress sub-steps.
    let phase_b_steps = STC_PROGRESS_STEPS - phase_a_steps;
    let progress_interval_b = (n / phase_b_steps as usize).max(1);
    let mut progress_counter = 0usize;

    let mut stego_bits = vec![0u8; n];
    let mut entry_state = best_state;
    // T3.6 — Phase B scratch (back-pointer byte array).
    let mut bp_byte = vec![0u8; num_states];

    for seg in (0..num_segments).rev() {
        let block_start = seg * k;
        let block_end = ((seg + 1) * k).min(m);
        let j_start = block_start * w;
        let j_end = block_end * w;
        let seg_len = j_end - j_start;

        // Reset prev_cost from checkpoint (reuses the same buffer).
        prev_cost.copy_from_slice(&checkpoints[seg]);

        // Recompute forward Viterbi for this segment, storing back_ptr.
        let mut seg_back_ptr: Vec<u128> = Vec::with_capacity(seg_len);
        let mut seg_msg_idx = block_start;

        for j in j_start..j_end {
            let col_idx = j % w;
            let col = columns[col_idx];
            let flip_cost = costs[j] as f64; // promote f32→f64
            let cover_bit = cover_bits[j] & 1;

            let (cost_s0, cost_s1) = if cover_bit == 0 {
                (0.0, flip_cost)
            } else {
                (flip_cost, 0.0)
            };

            let packed_bp = viterbi_step(
                &prev_cost,
                &mut curr_cost,
                &mut prev_cost_perm,
                &mut bp_byte,
                col,
                cost_s0,
                cost_s1,
            );

            seg_back_ptr.push(packed_bp);

            if col_idx == w - 1 && seg_msg_idx < m {
                let required_bit = message[seg_msg_idx] as usize;
                shifted_cost.fill(inf);
                for s in 0..num_states {
                    if curr_cost[s] == inf { continue; }
                    if (s & 1) != required_bit { continue; }
                    let s_shifted = s >> 1;
                    if curr_cost[s] < shifted_cost[s_shifted] {
                        shifted_cost[s_shifted] = curr_cost[s];
                    }
                }
                std::mem::swap(&mut prev_cost, &mut shifted_cost);
                seg_msg_idx += 1;
            } else {
                std::mem::swap(&mut prev_cost, &mut curr_cost);
            }

            progress_counter += 1;
            if progress_counter.is_multiple_of(progress_interval_b) {
                if progress::is_cancelled() { return None; }
                progress::advance();
            }
        }

        // Traceback within this segment.
        let mut s = entry_state;
        for local_j in (0..seg_len).rev() {
            let j = j_start + local_j;
            let col_idx = j % w;

            if col_idx == w - 1 && (j / w) < m {
                let msg_bit = message[j / w] as usize;
                s = ((s << 1) | msg_bit) & (num_states - 1);
            }

            let bit = ((seg_back_ptr[local_j] >> s) & 1) as u8;
            stego_bits[j] = bit;

            if bit == 1 {
                s ^= columns[col_idx];
            }
        }

        // State at the start of this segment = entry state for previous segment.
        entry_state = s;
        // seg_back_ptr is dropped here, freeing the segment's memory.
    }

    debug_assert_eq!(entry_state, 0, "traceback did not return to initial state 0");
    debug_assert_eq!(
        stc_extract(&stego_bits, hhat_matrix, w)[..m],
        message[..m],
    );

    let num_modifications = stego_bits.iter().zip(cover_bits.iter())
        .filter(|(s, c)| s != c).count();

    Some(EmbedResult { stego_bits, total_cost: best_cost, num_modifications })
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Find the state with minimum cost. Returns (state, cost).
fn find_best_state(costs: &[f64]) -> (usize, f64) {
    let mut best = 0;
    let mut best_cost = f64::INFINITY;
    for (s, &c) in costs.iter().enumerate() {
        if c < best_cost {
            best_cost = c;
            best = s;
        }
    }
    (best, best_cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::hhat::generate_hhat;
    use super::super::extract::stc_extract;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 33
    }

    /// One random trial instance. `wet` is the per-mille ∞-cost probability;
    /// the high settings are what make infeasible instances common.
    fn random_instance(
        s: &mut u64,
        h: usize,
        n: usize,
        m: usize,
        wet: u64,
    ) -> (Vec<u8>, Vec<f32>, Vec<u8>, Vec<Vec<u32>>, usize) {
        let w = (n / m).max(1);
        let mut seed = [0u8; 32];
        for b in seed.iter_mut() {
            *b = lcg(s) as u8;
        }
        let hhat = generate_hhat(h, w, &seed);
        let cover: Vec<u8> = (0..n).map(|_| (lcg(s) & 1) as u8).collect();
        let costs: Vec<f32> = (0..n)
            .map(|_| {
                if lcg(s) % 1000 < wet {
                    f32::INFINITY
                } else {
                    // Mix of magnitudes incl. zero and huge — magnitudes must
                    // not matter to the answer.
                    match lcg(s) % 4 {
                        0 => 0.0,
                        1 => 1.0,
                        2 => (lcg(s) % 10_000) as f32 * 0.37,
                        _ => 3.0e38,
                    }
                }
            })
            .collect();
        let msg: Vec<u8> = (0..m).map(|_| (lcg(s) & 1) as u8).collect();
        (cover, costs, msg, hhat, w)
    }

    /// `stc_is_feasible` must equal `stc_embed(..).is_some()` on EVERY input:
    /// it replaces the full trellis in the H.264 capacity probe, whose answers
    /// feed the per-GOP primary plan and therefore the stego bytes. Covers all
    /// constraint lengths, exact and ragged `n` vs `m·w` (incl. a cover shorter
    /// than `m·w`), wet fractions from 0 to 100 % (both outcomes must occur
    /// often), a non-0/1 message byte, and the NaN fallback.
    #[test]
    fn stc_is_feasible_matches_stc_embed_exhaustively() {
        let mut s = 0xFEA5_1B1E_u64;
        let (mut yes, mut no) = (0usize, 0usize);
        for h in 1..=7usize {
            for &(n, m) in &[(64usize, 8usize), (200, 20), (203, 20), (512, 64), (1000, 37),
                             (999, 333), (30, 40)] {
                for &wet in &[0u64, 300, 600, 800, 900, 950, 980, 1000] {
                    for _ in 0..3 {
                        let (cover, costs, msg, hhat, w) = random_instance(&mut s, h, n, m, wet);
                        let want = stc_embed(&cover, &costs, &msg, &hhat, h, w).is_some();
                        let got = stc_is_feasible(&cover, &costs, &msg, &hhat, h, w);
                        assert_eq!(got, want, "h={h} n={n} m={m} w={w} wet={wet}");
                        if want { yes += 1 } else { no += 1 }
                    }
                }
            }
        }
        assert!(yes > 200 && no > 200, "both outcomes must be well exercised: {yes} / {no}");

        // Degenerate parameters.
        let (cover, costs, msg, hhat, _) = random_instance(&mut s, 7, 100, 10, 0);
        for (hh, ww) in [(7usize, 0usize), (8, 10)] {
            assert_eq!(
                stc_is_feasible(&cover, &costs, &msg, &hhat, hh, ww),
                stc_embed(&cover, &costs, &msg, &hhat, hh, ww).is_some()
            );
        }
        assert_eq!(stc_is_feasible(&cover, &costs, &[], &hhat, 7, 10), true);
        // Non-0/1 message byte: matches no state in either.
        let mut bad = msg.clone();
        bad[3] = 2;
        assert_eq!(
            stc_is_feasible(&cover, &costs, &bad, &hhat, 7, 10),
            stc_embed(&cover, &costs, &bad, &hhat, 7, 10).is_some()
        );
        // NaN cost → falls back to the full trellis.
        let mut nan = costs.clone();
        nan[5] = f32::NAN;
        assert_eq!(
            stc_is_feasible(&cover, &nan, &msg, &hhat, 7, 10),
            stc_embed(&cover, &nan, &msg, &hhat, 7, 10).is_some()
        );
    }

    /// Same equality on the segmented path (n > SEGMENTED_THRESHOLD), which
    /// walks only the first m·w positions — one feasible and one infeasible
    /// instance (the wet fraction straddles the capacity boundary).
    #[test]
    fn stc_is_feasible_matches_stc_embed_segmented() {
        let mut s = 0x5E6_7E57_u64;
        let n = SEGMENTED_THRESHOLD + 4_321;
        let m = 1_200;
        let mut seen = [false; 2];
        for &wet in &[700u64, 1000] {
            let (cover, costs, msg, hhat, w) = random_instance(&mut s, 7, n, m, wet);
            assert!(n > SEGMENTED_THRESHOLD && m * w <= n);
            let want = stc_embed(&cover, &costs, &msg, &hhat, 7, w).is_some();
            assert_eq!(stc_is_feasible(&cover, &costs, &msg, &hhat, 7, w), want, "wet={wet}");
            seen[want as usize] = true;
        }
        assert!(seen[0] && seen[1], "need one feasible and one infeasible instance: {seen:?}");
    }

    #[test]
    fn embed_extract_roundtrip_tiny() {
        let h = 3;
        let n: usize = 20;
        let m: usize = 4;
        let w = n.div_ceil(m); // ceil(20/4) = 5
        let seed = [42u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let costs: Vec<f32> = vec![1.0; n];
        let message = vec![1u8, 0, 1, 1];

        let result = stc_embed(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        assert_eq!(result.stego_bits.len(), n);

        let extracted = stc_extract(&result.stego_bits, &hhat, w);
        assert_eq!(&extracted[..m], &message[..]);
    }

    #[test]
    fn embed_extract_roundtrip_h7() {
        let h = 7;
        let n: usize = 500;
        let m: usize = 50;
        let w = n.div_ceil(m);
        let seed = [13u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| ((i * 7 + 3) % 2) as u8).collect();
        let costs: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let message: Vec<u8> = (0..m).map(|i| (i % 2) as u8).collect();

        let result = stc_embed(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        let extracted = stc_extract(&result.stego_bits, &hhat, w);
        assert_eq!(&extracted[..m], &message[..]);
    }

    #[test]
    fn wet_coefficients_not_modified() {
        let h = 3;
        let n: usize = 20;
        let m: usize = 4;
        let w = n.div_ceil(m);
        let seed = [55u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = vec![0; n];
        let mut costs: Vec<f32> = vec![1.0; n];
        // Make positions 0, 5, 10, 15 WET
        for i in (0..n).step_by(5) {
            costs[i] = 1e13;
        }
        let message = vec![0u8, 1, 0, 1];

        let result = stc_embed(&cover_bits, &costs, &message, &hhat, h, w).unwrap();

        // WET positions must not change
        for i in (0..n).step_by(5) {
            assert_eq!(
                result.stego_bits[i], cover_bits[i],
                "WET position {i} was modified"
            );
        }

        // Message still recoverable
        let extracted = stc_extract(&result.stego_bits, &hhat, w);
        assert_eq!(&extracted[..m], &message[..]);
    }

    #[test]
    fn empty_message() {
        let h = 3;
        let n = 10;
        let w = 5;
        let seed = [0u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = vec![1; n];
        let costs: Vec<f32> = vec![1.0; n];
        let message: Vec<u8> = vec![];

        let result = stc_embed(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        assert_eq!(result.stego_bits, cover_bits);
        assert_eq!(result.total_cost, 0.0);
    }

    /// Large synthetic test to verify 1-bit packed back pointers at scale.
    #[test]
    fn embed_extract_roundtrip_large() {
        let h = 7;
        let m = 10_000;
        let w = 10;
        let n = m * w; // 100K cover elements
        let seed = [77u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| ((i * 31 + 17) % 2) as u8).collect();
        let costs: Vec<f32> = (0..n).map(|i| {
            let base = 0.5 + (i % 100) as f32 * 0.02;
            if i % 500 == 0 { f32::INFINITY } else { base }
        }).collect();
        let message: Vec<u8> = (0..m).map(|i| ((i * 13 + 7) % 2) as u8).collect();

        let result = stc_embed(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        assert_eq!(result.stego_bits.len(), n);

        let extracted = stc_extract(&result.stego_bits, &hhat, w);
        assert_eq!(&extracted[..m], &message[..]);

        for i in (0..n).step_by(500) {
            assert_eq!(
                result.stego_bits[i], cover_bits[i],
                "WET position {i} was modified"
            );
        }
    }

    /// Verify that inline and segmented paths produce identical output.
    #[test]
    fn inline_segmented_equivalence() {
        let h = 7;
        let m = 500;
        let w = 10;
        let n = m * w; // 5000 cover elements
        let seed = [99u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| ((i * 31 + 17) % 2) as u8).collect();
        let costs: Vec<f32> = (0..n).map(|i| {
            let base = 0.5 + (i % 100) as f32 * 0.02;
            if i % 500 == 0 { f32::INFINITY } else { base }
        }).collect();
        let message: Vec<u8> = (0..m).map(|i| ((i * 13 + 7) % 2) as u8).collect();

        let inline = stc_embed_inline(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        let segmented = stc_embed_segmented(&cover_bits, &costs, &message, &hhat, h, w).unwrap();

        assert_eq!(inline.stego_bits, segmented.stego_bits, "stego bits differ");
        assert_eq!(inline.total_cost, segmented.total_cost, "total cost differs");
    }

    /// Equivalence test with a larger input covering multiple segments.
    #[test]
    fn inline_segmented_equivalence_large() {
        let h = 7;
        let m = 10_000;
        let w = 10;
        let n = m * w; // 100K cover elements, K ≈ 100 → ~100 segments
        let seed = [88u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| ((i * 37 + 11) % 2) as u8).collect();
        let costs: Vec<f32> = (0..n).map(|i| {
            let base = 0.3 + (i % 200) as f32 * 0.01;
            if i % 1000 == 0 { f32::INFINITY } else { base }
        }).collect();
        let message: Vec<u8> = (0..m).map(|i| ((i * 19 + 3) % 2) as u8).collect();

        let inline = stc_embed_inline(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        let segmented = stc_embed_segmented(&cover_bits, &costs, &message, &hhat, h, w).unwrap();

        assert_eq!(inline.stego_bits, segmented.stego_bits, "stego bits differ");
        assert_eq!(inline.total_cost, segmented.total_cost, "total cost differs");
    }

    /// Segmented path with a single segment (m ≤ K).
    #[test]
    fn segmented_single_segment() {
        let h = 7;
        let m = 4;
        let w = 5;
        let n = m * w;
        let seed = [33u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let costs: Vec<f32> = vec![1.0; n];
        let message: Vec<u8> = vec![1, 0, 1, 1];

        let inline = stc_embed_inline(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        let segmented = stc_embed_segmented(&cover_bits, &costs, &message, &hhat, h, w).unwrap();

        assert_eq!(inline.stego_bits, segmented.stego_bits);
        assert_eq!(inline.total_cost, segmented.total_cost);
    }
}
