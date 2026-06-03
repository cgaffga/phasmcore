// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase B.1.2 — AV1 J-UNIWARD cost function.
//!
//! Port of [`crate::stego::cost::h264_uniward`] to AV1's mixed-size
//! transform set (4×4 to 64×64, both square and rectangular). The
//! algorithm is unchanged — Daubechies-8 wavelet decomposition of
//! the post-LR reconstructed pixel domain, score each candidate flip
//! by how much it perturbs the wavelet coefficients relative to the
//! cover's wavelet magnitudes.
//!
//! AV1-specific differences from the H.264 port:
//!
//! * **Variable TX sizes**. H.264 has fixed 4×4; AV1 has square +
//!   rectangular shapes log2-width × log2-height in {2,3,4,5,6}.
//!   Impact window per flip = `tx_size + FILT_LEN - 1`.
//! * **Multiple transforms**. AV1's TX type set is DCT, ADST,
//!   FLIPADST, IDTX × 2D combinations (16 total). v1 MVP uses a
//!   single 2D DCT-II basis for all non-IDTX combinations; ADST /
//!   FLIPADST per-type bases queued for v0.6+ refinement. IDTX
//!   coefficients return uniform cost (no frequency basis).
//! * **Quantizer**. AV1 uses qindex 0..255. v1 MVP uses a simple
//!   linear quant-step approximation. AV1 spec dq_table is queued
//!   for v0.6+.
//!
//! The wavelet filter coefficients (HPDF/LPDF) and the
//! `compute_three_subbands` decomposition match those in
//! `cost::h264_uniward` (same Daubechies-8 wavelet). Duplicated
//! locally instead of imported because h264_uniward is gated on
//! `feature = "video"` while av1_uniward needs to be available
//! whenever av1-encoder is active.
//!
//! ## B.3.1 perf pivot (2026-06-03)
//!
//! Per-position cost was 30-40 μs because [`dct_ii_basis`] computed
//! the basis from scratch per cover position using `f64::cos()`. Two
//! problems with the old code:
//!
//! 1. **WASM determinism violation**. `f64::cos()` compiles to
//!    `Math.cos` in WASM, non-deterministic across runtimes. Violated
//!    the project-wide `det_math` invariant (CLAUDE.md: "Never use
//!    `f64::sin/cos/atan2`"). Silently produced different basis values
//!    across browser engines / Node / Wasmtime.
//! 2. **Redundant work**. The basis at `(tx_w, tx_h, freq_x, freq_y)`
//!    is identical across every position with that tuple — typically
//!    100+ positions per tuple in a 1080p frame. Re-doing 1024
//!    `cos()` calls per position is 99% wasted work.
//!
//! Fix: [`FrameBasisCache`] memoizes the basis per tuple at frame
//! scope, computed via [`crate::det_math::det_cos`] (deterministic
//! IEEE-754 polynomial cos < 1 ULP). Cache lifetime == one
//! `compute_av1_uniward_costs` call. Lookup is O(1) HashMap; cache
//! miss path runs the deterministic cos loop once. See
//! `docs/design/video/av1/phase-b31-embed-perf.md` for the full
//! rationale + the audit that ruled out the rav1e-fork-patch
//! integer-butterfly alternative (sizes 16/32/64 not pub-exported by
//! rav1e; `det_math::det_cos` is the project's existing primitive
//! for exactly this case).

use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::det_math::det_cos;

#[cfg(feature = "av1-encoder")]
use crate::codec::av1::stego::cascade_kernel::{
    scale_l3_to_cascade, L2Key, L3CascadeCache,
};
#[cfg(feature = "av1-encoder")]
use phasm_rav1e::ec::PhasmFrameLoopFilterState;

// ======================================================================
//  B.1.5.5 — Three-tier dispatch thresholds (calibrated from
//  B.1.5.0.5 spike data)
// ======================================================================

/// Coefficient-magnitude to cascade-magnitude conversion factor.
/// Default 6.0 — empirical from B.1.5.0.5 spike. Env-overridable via
/// `PHASM_AV1_COEFF_FACTOR` for B.1.5.6 threshold sweep.
#[cfg(feature = "av1-encoder")]
fn coeff_to_cascade_factor() -> f64 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<f64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("PHASM_AV1_COEFF_FACTOR")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6.0)
    })
}

/// EE-D safe-band threshold: `|coeff| × factor < THRESHOLD_SAFE` →
/// position accepted without forward modeling. Default 12.0
/// (catches |coeff| ≤ 2 at factor=6, ~60% of cover positions).
/// Env-overridable via `PHASM_AV1_THRESHOLD_SAFE`.
#[cfg(feature = "av1-encoder")]
fn threshold_safe() -> f64 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<f64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("PHASM_AV1_THRESHOLD_SAFE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12.0)
    })
}

/// EE-D reject-band threshold: `|coeff| × factor > THRESHOLD_REJECT`
/// → position rejected (INF cost) without forward modeling. Default
/// 30.0 (catches |coeff| ≥ 5 at factor=6, ~25% of positions).
/// Env-overridable via `PHASM_AV1_THRESHOLD_REJECT`.
#[cfg(feature = "av1-encoder")]
fn threshold_reject() -> f64 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<f64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("PHASM_AV1_THRESHOLD_REJECT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30.0)
    })
}

/// EE-E wavelet smoothness threshold. When the cover region's local
/// high-frequency wavelet energy (|LH|+|HL|+|HH| at the block's
/// top-left) is below this, the position is in a smooth region where
/// cascade ripples are visible — its cost is multiplied by
/// `EE_E_SMOOTH_PENALTY`. When above, the texture masks the cascade
/// and the cost stays at its EE-D / L3 value. Applied to both safe
/// and middle bands. Set to 0.0 to disable.
///
/// **B.1.5.6 calibration**: default 40.0 (in absolute Daubechies-8
/// wavelet-coefficient magnitude units on 0..255 luma). Sweep on
/// carplane TG-2: SMOOTH=40 / PENALTY=4 → cliff 1.90 dB ✓ (vs 5.32
/// baseline w/o EE-E), AoSO AUC stays ≤ 0.70 ✓. SMOOTH=36 / PENALTY=4
/// passes TG-2 too but fails AoSO (over-funnels into textured-only).
/// SMOOTH=10..30 too tight (loses too many candidates); 75+ catches
/// nothing because the threshold sits above typical wavelet HF magnitudes.
/// Env-overridable via `PHASM_AV1_EE_E_SMOOTHNESS`.
#[cfg(feature = "av1-encoder")]
fn ee_e_smoothness_threshold() -> f64 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<f64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("PHASM_AV1_EE_E_SMOOTHNESS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(40.0)
    })
}

/// EE-E penalty multiplier applied to cost in smooth regions.
/// Default 4.0 — calibrated B.1.5.6 alongside SMOOTHNESS=40.
/// Higher penalty (8/16/32) hurts: forces STC into the few non-smooth
/// positions, concentrating damage there. Lower (2-3) doesn't shift
/// STC enough.
/// Env-overridable via `PHASM_AV1_EE_E_PENALTY`.
#[cfg(feature = "av1-encoder")]
fn ee_e_smooth_penalty() -> f64 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<f64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("PHASM_AV1_EE_E_PENALTY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4.0)
    })
}

/// SIMD determinism gate, lifted from H.264's
/// `core/src/codec/h264/encoder/simd/mod.rs:62-119` pattern. Returns
/// `true` unless the `PHASM_AV1_DISABLE_SIMD` env var is set at first
/// call (cached via `OnceLock`). Currently informational — av1_uniward
/// has no SIMD code paths yet — but the gate ships from day 1 so any
/// future AVX2 / NEON / SIMD128 inner loops can branch on it and be
/// verified against scalar fallback (CI gate: SIMD-on output ==
/// SIMD-off output byte-for-byte; see phase-b31-embed-perf.md § 3).
#[allow(dead_code)]
pub(crate) fn phasm_av1_simd_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PHASM_AV1_DISABLE_SIMD").is_err())
}

/// Stabilization constant σ from the UNIWARD paper. Same value as
/// JPEG + H.264 J-UNIWARD. Calibrated for 8×8 originally; AV1 mixes
/// sizes so per-size σ tuning is v0.6+.
const SIGMA: f64 = 0.015625; // 2^-6

/// AV1 IDTX transform-type index. For 4×4: TX_TYPE=4 = IDTX. For
/// other sizes the IDTX variants are at index 9 (V_DCT), 10 (H_DCT),
/// etc. — we conservatively flag a small set; expand in v0.6+ when
/// per-type bases land.
///
/// AV1 enum (per av1-spec § 6.4.2):
///   0 = DCT_DCT       8 = V_DCT
///   1 = ADST_DCT      9 = H_DCT
///   2 = DCT_ADST     10 = V_ADST
///   3 = ADST_ADST    11 = H_ADST
///   4 = FLIPADST_DCT 12 = V_FLIPADST
///   5 = DCT_FLIPADST 13 = H_FLIPADST
///   6 = FLIPADST_FLIPADST
///   7 = ADST_FLIPADST
///   (IDTX = 15 in some encodings — variable across spec revisions)
const AV1_TX_TYPE_IDTX: u8 = 15;

/// One AC sign emission's spatial info needed for J-UNIWARD cost.
/// Subset of `phasm_rav1e::AcSignMeta` — phasm-core projects encoder
/// or decoder metas into this struct to keep the cost function free
/// of fork-internal types.
///
/// Phase B.1.5.0.5: extended with `coeff_magnitude` from the encoder-
/// side `AcSignMeta`. Used by cascade-safety v2's EE-D coefficient-
/// magnitude pre-filter + EE-C upper-bound check + L3 cache key.
/// Walker-side projections leave the field at default zero (the
/// decoder doesn't populate it — extract has no cost-compute step).
#[derive(Debug, Clone, Copy, Default)]
pub struct Av1FramePosition {
    pub plane: u8,
    pub plane_px_x: u16,
    pub plane_px_y: u16,
    pub tx_width_log2: u8,
    pub tx_height_log2: u8,
    pub tx_type: u8,
    pub scan_pos: u16,
    /// Absolute quantized coefficient magnitude at this position.
    /// Saturated at `u16::MAX`. Encoder-side populated; walker leaves
    /// at zero. See `AcSignMeta::coeff_magnitude` in phasm-rav1e
    /// `src/ec.rs` for the fork-side definition.
    pub coeff_magnitude: u16,
}

/// Reconstructed-frame planes (per-plane visible-region packed YUV).
/// Caller is responsible for extracting these from
/// `PhasmFrameRecording.reconstructed_planes` — see
/// [`pack_visible_from_frame`] helper below.
pub struct FramePlanes {
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    pub luma_width: usize,
    pub luma_height: usize,
    pub chroma_width: usize,
    pub chroma_height: usize,
}

/// Compute J-UNIWARD costs for a slice of AC sign positions.
///
/// Returns one f32 per input position, in the same order. f32 type
/// matches the STC embed expectation. INF returned for positions
/// where cost can't be meaningfully computed (out-of-bounds, IDTX —
/// caller should still pass these; the cost vector aligns with the
/// cover bits).
///
/// `qindex` is the frame-level quantizer for v1 MVP (per-block delta-Q
/// awareness deferred to v0.6+ when AV1 segmentation lands here).
pub fn compute_av1_uniward_costs(
    planes: &FramePlanes,
    positions: &[Av1FramePosition],
    qindex: u8,
) -> Vec<f32> {
    compute_av1_uniward_costs_with_state(planes, positions, qindex, None)
}

/// B.1.5.5 entry point: cost compute with optional `PhasmFrameLoopFilterState`.
/// When `Some`, the three-tier dispatch is active (EE-D safe/reject
/// bands + L3 forward model for the |coeff| middle band). When
/// `None`, falls back to the legacy J-UNIWARD path (B.1.3 magnitude
/// proxy) — primarily for the in-module unit tests that don't have a
/// real encode behind them.
///
/// The B.1.5.6 verification gate (TG-2 + AoSO + capacity sweep) runs
/// against the `Some` path; the legacy path stays around for the
/// existing unit tests until they're updated to plumb a state.
pub fn compute_av1_uniward_costs_with_state(
    planes: &FramePlanes,
    positions: &[Av1FramePosition],
    qindex: u8,
    #[cfg(feature = "av1-encoder")] loop_filter_state: Option<PhasmFrameLoopFilterState>,
    #[cfg(not(feature = "av1-encoder"))] _loop_filter_state: Option<()>,
) -> Vec<f32> {
    // Wavelet decomposition of each plane (one-off per frame). This
    // is the expensive precompute step; per-position cost is then
    // ~O(tx_w * tx_h + impact_window²) which is cheap.
    let y_wavelets = compute_three_subbands(&planes.y, planes.luma_width, planes.luma_height);
    let cb_wavelets =
        compute_three_subbands(&planes.cb, planes.chroma_width, planes.chroma_height);
    let cr_wavelets =
        compute_three_subbands(&planes.cr, planes.chroma_width, planes.chroma_height);

    let q_scale = qindex_to_step(qindex);
    let delta_magnitude = 2.0 * q_scale; // sign flip: ±1 × quant step

    // B.3.1.3: prefill the cache serially, then run the per-position
    // cost compute in parallel against the now-read-only cache.
    // Rayon's `par_iter().collect()` preserves input order so the
    // returned `Vec<f32>` aligns position-for-position with `positions`
    // — same contract as the serial path.
    let mut basis_cache = FrameBasisCache::new(delta_magnitude);
    basis_cache.prefill(positions);
    let basis_cache = basis_cache; // freeze: subsequent reads are &-only

    // B.1.5.5: prefill L3 cascade cache for middle-band positions
    // (the ~10-15% where |coeff| is ambiguous). Safe + reject bands
    // get O(1) closed-form cost; only middle needs forward modeling.
    #[cfg(feature = "av1-encoder")]
    let l3_cache = loop_filter_state.map(|state| {
        let mut l3 = L3CascadeCache::new(state);
        prefill_l3_middle_band(&mut l3, positions);
        l3
    });
    #[cfg(not(feature = "av1-encoder"))]
    let l3_cache: Option<()> = None;

    let compute_one = |p: &Av1FramePosition| {
        compute_position_cost(
            p,
            &y_wavelets,
            &cb_wavelets,
            &cr_wavelets,
            planes,
            &basis_cache,
            #[cfg(feature = "av1-encoder")]
            l3_cache.as_ref(),
            #[cfg(not(feature = "av1-encoder"))]
            l3_cache.as_ref(),
        )
    };

    #[cfg(feature = "parallel")]
    {
        positions.par_iter().map(compute_one).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        positions.iter().map(compute_one).collect()
    }
}

/// B.1.5.5: prefill L3 entries for cover positions whose `|coeff|`
/// upper bound falls in the middle band. Skips safe + reject band
/// positions (they get O(1) closed-form cost via three-tier dispatch).
#[cfg(feature = "av1-encoder")]
fn prefill_l3_middle_band(l3: &mut L3CascadeCache, positions: &[Av1FramePosition]) {
    for pos in positions {
        if pos.tx_type == AV1_TX_TYPE_IDTX {
            continue;
        }
        let coeff_upper_bound = pos.coeff_magnitude as f64 * coeff_to_cascade_factor();
        if coeff_upper_bound < threshold_safe() || coeff_upper_bound > threshold_reject() {
            continue;
        }
        let tx_w = 1usize << pos.tx_width_log2;
        let tx_h = 1usize << pos.tx_height_log2;
        let scan_pos = pos.scan_pos as usize;
        if scan_pos == 0 {
            continue;
        }
        let freq_x = scan_pos % tx_w;
        let freq_y = scan_pos / tx_w;
        if freq_y >= tx_h {
            continue;
        }
        // Compute unit-magnitude basis on demand for L3's
        // normalized-energy storage.
        let mut basis = vec![0.0f64; tx_w * tx_h];
        dct_ii_basis(tx_w, tx_h, freq_x, freq_y, &mut basis);
        let key = L2Key::new(tx_w, tx_h, freq_x, freq_y, pos.plane);
        let _ = l3.get_or_compute(key, &basis);
    }
}

/// AV1 quantizer step approximation for v1 MVP. Real AV1 uses two
/// tables (DC + AC) indexed by qindex with non-linear scaling
/// (av1-spec § 7.12.2). For v1 we linearly approximate ac_qstep ≈
/// qindex / 4 + 1 — gives reasonable separation across the q
/// range. v0.6+ should swap for the spec tables for accuracy.
#[inline]
fn qindex_to_step(qindex: u8) -> f64 {
    (qindex as f64 / 4.0 + 1.0).max(1.0)
}

/// Compute per-flip J-UNIWARD cost.
///
/// `basis_cache` is a per-frame memo (see [`FrameBasisCache`]) — read-
/// only here because the caller pre-fills it via
/// [`FrameBasisCache::prefill`] before launching the parallel
/// position loop. Lookup is `&self`, so the cache is freely shared
/// across rayon worker threads.
fn compute_position_cost(
    pos: &Av1FramePosition,
    y_wavelets: &ThreeSubbands,
    cb_wavelets: &ThreeSubbands,
    cr_wavelets: &ThreeSubbands,
    planes: &FramePlanes,
    basis_cache: &FrameBasisCache,
    #[cfg(feature = "av1-encoder")] l3_cache: Option<&L3CascadeCache>,
    #[cfg(not(feature = "av1-encoder"))] _l3_cache: Option<&()>,
) -> f32 {
    // IDTX: no frequency basis — return uniform low cost. IDTX is
    // identity (no transform), so flipping its sign in pixel space
    // doesn't spread — cascade is small. Preferred by STC over
    // |coeff|-band positions.
    if pos.tx_type == AV1_TX_TYPE_IDTX {
        return 1.0;
    }

    // B.1.5.5 three-tier dispatch (active iff `l3_cache` is Some, i.e.,
    // the orchestrator passed a `PhasmFrameLoopFilterState`). Falls
    // back to the legacy J-UNIWARD path below when l3_cache is None
    // (in-module unit tests + paths without loop-filter state).
    #[cfg(feature = "av1-encoder")]
    if let Some(l3) = l3_cache {
        let coeff_upper_bound = pos.coeff_magnitude as f64 * coeff_to_cascade_factor();

        // EE-D reject band: above threshold → STC will never pick.
        // Checked FIRST so smoothness lookup doesn't run for these.
        if coeff_upper_bound > threshold_reject() {
            return f32::INFINITY;
        }

        // EE-E wavelet smoothness penalty — applied to BOTH safe and
        // middle bands. Smooth regions amplify cascade visibility; STC
        // should treat |coeff|=1 in a smooth area as MORE expensive
        // than |coeff|=1 in a textured area, even though both pass
        // EE-D. Without this, the safe band's flat `|coeff|×factor`
        // cost can't distinguish smooth vs textured positions →
        // detectable bias on smooth content + frame-to-frame cliff
        // variance when one frame is smoother than another.
        // Active iff PHASM_AV1_EE_E_SMOOTHNESS > 0.
        let smoothness_thresh = ee_e_smoothness_threshold();
        let smoothness_mult = if smoothness_thresh > 0.0 {
            let wavelets = match pos.plane {
                0 => y_wavelets,
                1 => cb_wavelets,
                2 => cr_wavelets,
                _ => return f32::INFINITY,
            };
            let abs_x = pos.plane_px_x as isize;
            let abs_y = pos.plane_px_y as isize;
            let wx = (abs_x - wavelets.x_offset) as usize;
            let wy = (abs_y - wavelets.y_offset) as usize;
            if wx < wavelets.width && wy * wavelets.width + wx < wavelets.lh.len() {
                let idx = wy * wavelets.width + wx;
                let w_lh = wavelets.lh[idx].abs() as f64;
                let w_hl = wavelets.hl[idx].abs() as f64;
                let w_hh = wavelets.hh[idx].abs() as f64;
                let local_hf = w_lh + w_hl + w_hh;
                if local_hf < smoothness_thresh {
                    ee_e_smooth_penalty()
                } else {
                    1.0
                }
            } else {
                1.0
            }
        } else {
            1.0
        };

        // EE-D safe band: |coeff| × factor below threshold → low cost.
        // Gradient: |coeff|=1 → cost 6, |coeff|=2 → cost 12, modulated
        // by EE-E smoothness multiplier.
        if coeff_upper_bound < threshold_safe() {
            return (coeff_upper_bound * smoothness_mult) as f32;
        }

        // Middle band: look up the predicted post-cascade magnitude
        // from L3 (prefilled by `prefill_l3_middle_band` before this
        // par_iter started).
        let tx_w_local = 1usize << pos.tx_width_log2;
        let tx_h_local = 1usize << pos.tx_height_log2;
        let scan_pos_local = pos.scan_pos as usize;
        if scan_pos_local == 0 {
            return f32::INFINITY;
        }
        let freq_x_local = scan_pos_local % tx_w_local;
        let freq_y_local = scan_pos_local / tx_w_local;
        if freq_y_local >= tx_h_local {
            return f32::INFINITY;
        }
        let key = L2Key::new(
            tx_w_local,
            tx_h_local,
            freq_x_local,
            freq_y_local,
            pos.plane,
        );
        let normalized_energy = l3.get(&key).unwrap_or(0.0);
        let cascade = scale_l3_to_cascade(normalized_energy, pos.coeff_magnitude);
        return (cascade * smoothness_mult) as f32;
    }

    // ---- Legacy J-UNIWARD path (B.1.3 magnitude proxy) follows ----
    // Active when l3_cache is None (unit tests, no-state callers).
    let _ = (y_wavelets, cb_wavelets, cr_wavelets); // legacy path uses these via match below

    // Pick wavelets + frame dims for the plane.
    let (wavelets, img_w, img_h) = match pos.plane {
        0 => (y_wavelets, planes.luma_width, planes.luma_height),
        1 => (cb_wavelets, planes.chroma_width, planes.chroma_height),
        2 => (cr_wavelets, planes.chroma_width, planes.chroma_height),
        _ => return f32::INFINITY,
    };

    let tx_w = 1usize << pos.tx_width_log2;
    let tx_h = 1usize << pos.tx_height_log2;
    let block_px_x = pos.plane_px_x as usize;
    let block_px_y = pos.plane_px_y as usize;

    // Bounds check — out-of-frame metadata is a fork bug; treat as
    // WET (infinite cost).
    if block_px_x + tx_w > img_w || block_px_y + tx_h > img_h {
        return f32::INFINITY;
    }

    // Decode (freq_y, freq_x) from scan_pos. scan_pos is the raster
    // index = freq_y * tx_w + freq_x.
    let scan_pos = pos.scan_pos as usize;
    if scan_pos == 0 {
        // DC coefficient — not an AC sign emission. Should never get
        // here since the encoder tags only AC signs; defensive.
        return f32::INFINITY;
    }
    let freq_x = scan_pos % tx_w;
    let freq_y = scan_pos / tx_w;
    if freq_y >= tx_h {
        return f32::INFINITY;
    }

    // B.3.1.2 + B.3.1.3: read-only tuple cache lookup (`get`, not
    // `get_or_compute`). Caller has pre-filled the cache via
    // `FrameBasisCache::prefill`; missing the cache here means the
    // walker found a position whose tuple wasn't enumerated during
    // prefill — a programmer error, not a correctness fallback.
    let tuple = basis_cache.get(tx_w, tx_h, freq_x, freq_y).expect(
        "tuple must be pre-filled by FrameBasisCache::prefill before \
         compute_position_cost is called",
    );
    let cascade_magnitude_proxy = tuple.basis_max_abs;
    let impact_w = tuple.impact_w as usize;
    let impact_h = tuple.impact_h as usize;
    let pad = FILT_LEN - 1; // 15

    // Per-position cost accumulation: walk the impact window, look up
    // the cached delta values, accumulate cost against the cover
    // wavelet magnitudes at the image-coordinate-mapped position.
    let mut cost = 0.0f64;
    for out_r in 0..impact_h {
        for out_c in 0..impact_w {
            // Map (out_r, out_c) back to image coordinates first so
            // we can early-skip on bounds without touching the cache
            // arrays (negligible — included for clarity).
            let abs_x = block_px_x as isize + out_c as isize - pad as isize;
            let abs_y = block_px_y as isize + out_r as isize - pad as isize;
            if abs_x < 0 || abs_y < 0 || abs_x >= img_w as isize || abs_y >= img_h as isize {
                continue;
            }
            let wx = (abs_x - wavelets.x_offset) as usize;
            let wy = (abs_y - wavelets.y_offset) as usize;
            let idx = wy * wavelets.width + wx;

            let cell = out_r * impact_w + out_c;
            let delta_lh = tuple.delta_lh[cell];
            let delta_hl = tuple.delta_hl[cell];
            let delta_hh = tuple.delta_hh[cell];

            let w_lh = wavelets.lh[idx].abs() as f64;
            let w_hl = wavelets.hl[idx].abs() as f64;
            let w_hh = wavelets.hh[idx].abs() as f64;

            cost += delta_lh.abs() / (w_lh + SIGMA);
            cost += delta_hl.abs() / (w_hl + SIGMA);
            cost += delta_hh.abs() / (w_hh + SIGMA);
        }
    }

    // Phase B.1.3: add the cascade-magnitude-proxy term.
    let final_cost = cost + LAMBDA_CASCADE * cascade_magnitude_proxy;
    final_cost as f32
}

/// λ_cascade — multiplicative weight on the cascade-magnitude-proxy
/// cost term. Per cost-model.md § 4.2 starting value 1.0; tune
/// empirically against the AoSO self-steganalyzer (B.1.4) once
/// available. Smaller values → J-UNIWARD wavelet dominates. Larger
/// values → cascade-proxy dominates (we'd flip in smoother pixel
/// regions even if J-UNIWARD prefers textured ones).
const LAMBDA_CASCADE: f64 = 1.0;

/// 2D DCT-II basis at (freq_y, freq_x) of size (tx_h, tx_w). v1 MVP
/// uses this for ALL non-IDTX transform types — including ADST and
/// FLIPADST. The approximation: ADST/FLIPADST bases are shifted /
/// reflected versions of DCT; the wavelet response of the cost is
/// similar in magnitude, just localized differently. v0.6+ should
/// implement per-type bases for accuracy.
///
/// Uses `det_math::det_cos` (NOT `f64::cos()`) to keep cross-platform
/// WASM determinism — see module-level doc comment + B.3.1 pivot.
fn dct_ii_basis(tx_w: usize, tx_h: usize, freq_x: usize, freq_y: usize, out: &mut [f64]) {
    let pi = std::f64::consts::PI;
    let c_u = if freq_y == 0 {
        1.0 / (tx_h as f64).sqrt()
    } else {
        (2.0 / tx_h as f64).sqrt()
    };
    let c_v = if freq_x == 0 {
        1.0 / (tx_w as f64).sqrt()
    } else {
        (2.0 / tx_w as f64).sqrt()
    };
    let norm = c_u * c_v;
    for i in 0..tx_h {
        for j in 0..tx_w {
            let cos_i = det_cos((2.0 * i as f64 + 1.0) * freq_y as f64 * pi / (2.0 * tx_h as f64));
            let cos_j = det_cos((2.0 * j as f64 + 1.0) * freq_x as f64 * pi / (2.0 * tx_w as f64));
            out[i * tx_w + j] = norm * cos_i * cos_j;
        }
    }
}

/// Per-tuple cached output of the full DCT-basis + wavelet-filter
/// pipeline. All fields are invariant in
/// `(tx_w, tx_h, freq_x, freq_y)` — identical for every cover position
/// sharing that tuple.
///
/// Stored fields:
/// * `basis_max_abs` — the cascade-safety magnitude proxy (max |basis|
///   over the (tx_w × tx_h) DCT-II response, post-`delta_magnitude`
///   scale). Used as the Early-exit C term in the J-UNIWARD cost.
/// * `delta_lh / delta_hl / delta_hh` — the three Daubechies-8 directional
///   subband responses of the basis pattern, length `impact_h × impact_w`
///   each. Replaces the per-position row-filter + column-filter loops
///   (~ 8 μs / position) with an O(1) lookup.
///
/// The raw basis matrix is intentionally NOT stored — only the
/// `basis_max_abs` summary survives the cache build, since downstream
/// cost compute only needs the three delta maps + the proxy.
struct TupleEntry {
    basis_max_abs: f64,
    delta_lh: Vec<f64>,
    delta_hl: Vec<f64>,
    delta_hh: Vec<f64>,
    impact_w: u16,
    impact_h: u16,
}

/// Per-frame DCT-basis + wavelet-filter cache keyed by
/// `(tx_w, tx_h, freq_x, freq_y)`. Each entry is a [`TupleEntry`]
/// holding the full per-tuple invariant data (basis_max_abs +
/// 3 Daubechies-8 subband responses).
///
/// Lifetime: one frame (one call to [`compute_av1_uniward_costs`]).
/// Per-frame because:
/// * The pipeline (basis → row-filter → column-filter) is fully
///   determined by `(tx_w, tx_h, freq_x, freq_y) + delta_magnitude`.
///   A 1080p frame at speed 10 (32×32 TX dominant) has ~1024 unique
///   tuples but 100K+ positions → ~99 % reuse rate.
/// * `delta_magnitude = 2 × q_scale` is per-frame constant (single
///   qindex per frame in v1 MVP) — pre-multiplied at cache insert.
/// * Cache lifetime == one frame avoids stale q_scale across frames.
///
/// Memory bound: ≤ `unique_tuples × 3 × impact_w × impact_h × 8 B`.
/// For 32×32 dominant: ~1024 entries × 3 × 47² × 8 B ≈ 53 MB worst
/// case; typically <20 MB. Dropped at frame end. The B.3.1.2 cache
/// trades ~50 MB peak RAM for eliminating the per-position
/// row-filter + column-filter loops (the dominant per-position cost
/// after B.3.1.1 cached the basis).
struct FrameBasisCache {
    inner: HashMap<(u16, u16, u16, u16), TupleEntry>,
    delta_magnitude: f64,
}

impl FrameBasisCache {
    fn new(delta_magnitude: f64) -> Self {
        Self {
            inner: HashMap::new(),
            delta_magnitude,
        }
    }

    /// Get the cached `TupleEntry` for the given tuple, computing
    /// the full basis + wavelet-filter pipeline on first request.
    /// `&mut self` — used by [`Self::prefill`] only.
    fn get_or_compute(
        &mut self,
        tx_w: usize,
        tx_h: usize,
        freq_x: usize,
        freq_y: usize,
    ) -> &TupleEntry {
        let key = (tx_w as u16, tx_h as u16, freq_x as u16, freq_y as u16);
        let delta = self.delta_magnitude;
        self.inner
            .entry(key)
            .or_insert_with(|| compute_tuple_entry(tx_w, tx_h, freq_x, freq_y, delta))
    }

    /// Read-only cache lookup. Returns `None` if the tuple was not
    /// pre-filled. Used by the parallel position loop after
    /// [`Self::prefill`] — `&self` makes the cache `Sync` and lets
    /// rayon workers share a single reference across threads.
    fn get(
        &self,
        tx_w: usize,
        tx_h: usize,
        freq_x: usize,
        freq_y: usize,
    ) -> Option<&TupleEntry> {
        self.inner
            .get(&(tx_w as u16, tx_h as u16, freq_x as u16, freq_y as u16))
    }

    /// Walk `positions` once and populate the cache for every valid
    /// `(tx_w, tx_h, freq_x, freq_y)` tuple. After this, the cache
    /// is frozen (`&self`-readable) for the parallel cost loop.
    ///
    /// Skips positions that would short-circuit before basis lookup
    /// in [`compute_position_cost`]:
    /// * `tx_type == IDTX` — returns uniform cost
    /// * `scan_pos == 0` — DC, encoder shouldn't tag these
    /// * `freq_y >= tx_h` — out-of-range scan
    ///
    /// Plane / bounds checks aren't applied here — they're position-
    /// specific, not tuple-specific. A bad position simply doesn't
    /// touch the cache.
    fn prefill(&mut self, positions: &[Av1FramePosition]) {
        for pos in positions {
            if pos.tx_type == AV1_TX_TYPE_IDTX {
                continue;
            }
            let tx_w = 1usize << pos.tx_width_log2;
            let tx_h = 1usize << pos.tx_height_log2;
            let scan_pos = pos.scan_pos as usize;
            if scan_pos == 0 {
                continue;
            }
            let freq_x = scan_pos % tx_w;
            let freq_y = scan_pos / tx_w;
            if freq_y >= tx_h {
                continue;
            }
            // Touch the cache — populates on first miss for the tuple.
            let _ = self.get_or_compute(tx_w, tx_h, freq_x, freq_y);
        }
    }
}

/// Build a fresh [`TupleEntry`] by running the full per-tuple
/// pipeline: basis → row-filter → column-filter → three delta maps.
/// The basis itself is dropped after row+column filtering — only the
/// summary `basis_max_abs` + the three deltas survive into the cache.
fn compute_tuple_entry(
    tx_w: usize,
    tx_h: usize,
    freq_x: usize,
    freq_y: usize,
    delta_magnitude: f64,
) -> TupleEntry {
    // 1. Basis (det_cos — bit-exact across ISA + WASM).
    let mut basis = vec![0.0f64; tx_w * tx_h];
    dct_ii_basis(tx_w, tx_h, freq_x, freq_y, &mut basis);
    for v in basis.iter_mut() {
        *v *= delta_magnitude;
    }

    // 2. Cascade-safety magnitude proxy (Early-exit C; cascade-safety.md § 9).
    let basis_max_abs = basis.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

    // 3. Row-filter the basis into low/high pass intermediate buffers
    //    (tx_h rows × impact_w columns).
    let pad = FILT_LEN - 1; // 15
    let impact_w = tx_w + pad;
    let impact_h = tx_h + pad;
    let mut row_low = vec![0.0f64; tx_h * impact_w];
    let mut row_high = vec![0.0f64; tx_h * impact_w];
    let lp = lpdf();
    for r in 0..tx_h {
        for out_c in 0..impact_w {
            let mut sum_low = 0.0;
            let mut sum_high = 0.0;
            for k in 0..FILT_LEN {
                let src = out_c as isize - (FILT_LEN - 1) as isize + k as isize;
                if src >= 0 && (src as usize) < tx_w {
                    let v = basis[r * tx_w + src as usize];
                    sum_low += lp[k] * v;
                    sum_high += HPDF[k] * v;
                }
            }
            row_low[r * impact_w + out_c] = sum_low;
            row_high[r * impact_w + out_c] = sum_high;
        }
    }

    // 4. Column-filter into three directional delta maps (impact_h × impact_w).
    let mut delta_lh = vec![0.0f64; impact_h * impact_w];
    let mut delta_hl = vec![0.0f64; impact_h * impact_w];
    let mut delta_hh = vec![0.0f64; impact_h * impact_w];
    for out_r in 0..impact_h {
        for out_c in 0..impact_w {
            let mut lh = 0.0;
            let mut hl = 0.0;
            let mut hh = 0.0;
            for k in 0..FILT_LEN {
                let src_r = out_r as isize - (FILT_LEN - 1) as isize + k as isize;
                if src_r >= 0 && (src_r as usize) < tx_h {
                    let r = src_r as usize;
                    let low_val = row_low[r * impact_w + out_c];
                    let high_val = row_high[r * impact_w + out_c];
                    lh += HPDF[k] * low_val;
                    hl += lp[k] * high_val;
                    hh += HPDF[k] * high_val;
                }
            }
            delta_lh[out_r * impact_w + out_c] = lh;
            delta_hl[out_r * impact_w + out_c] = hl;
            delta_hh[out_r * impact_w + out_c] = hh;
        }
    }

    TupleEntry {
        basis_max_abs,
        delta_lh,
        delta_hl,
        delta_hh,
        impact_w: impact_w as u16,
        impact_h: impact_h as u16,
    }
}

/// Three wavelet subbands (LH / HL / HH) of one of the reconstructed
/// YUV planes. Same shape as `cost::h264_uniward::ThreeSubbands`,
/// duplicated here to avoid feature-flag entanglement.
struct ThreeSubbands {
    lh: Vec<f32>,
    hl: Vec<f32>,
    hh: Vec<f32>,
    width: usize,
    #[allow(dead_code)]
    height: usize,
    x_offset: isize,
    y_offset: isize,
}

/// Compute the three Daubechies-8 directional subbands of an 8-bit
/// pixel plane. Matches `cost::h264_uniward::compute_three_subbands`
/// byte-for-byte.
fn compute_three_subbands(y_plane: &[u8], width: usize, height: usize) -> ThreeSubbands {
    let pad = FILT_LEN - 1;
    let padded_w = width + 2 * pad;
    let padded_h = height + 2 * pad;

    let mut row_low = vec![0.0f32; padded_w * height];
    let mut row_high = vec![0.0f32; padded_w * height];
    let lp = lpdf();

    for y in 0..height {
        for out_x in 0..padded_w {
            let mut sum_low = 0.0f64;
            let mut sum_high = 0.0f64;
            for k in 0..FILT_LEN {
                let src_x = out_x as isize - pad as isize + k as isize;
                let clamped = symmetric_reflect(src_x, width as isize);
                let v = y_plane[y * width + clamped as usize] as f64;
                sum_low += lp[k] * v;
                sum_high += HPDF[k] * v;
            }
            row_low[y * padded_w + out_x] = sum_low as f32;
            row_high[y * padded_w + out_x] = sum_high as f32;
        }
    }

    let mut lh = vec![0.0f32; padded_w * padded_h];
    let mut hl = vec![0.0f32; padded_w * padded_h];
    let mut hh = vec![0.0f32; padded_w * padded_h];

    for out_y in 0..padded_h {
        for x in 0..padded_w {
            let mut sum_lh = 0.0f64;
            let mut sum_hl = 0.0f64;
            let mut sum_hh = 0.0f64;
            for k in 0..FILT_LEN {
                let src_y = out_y as isize - pad as isize + k as isize;
                let clamped = symmetric_reflect(src_y, height as isize);
                let low_val = row_low[clamped as usize * padded_w + x] as f64;
                let high_val = row_high[clamped as usize * padded_w + x] as f64;
                sum_lh += HPDF[k] * low_val;
                sum_hl += lp[k] * high_val;
                sum_hh += HPDF[k] * high_val;
            }
            lh[out_y * padded_w + x] = sum_lh as f32;
            hl[out_y * padded_w + x] = sum_hl as f32;
            hh[out_y * padded_w + x] = sum_hh as f32;
        }
    }

    ThreeSubbands {
        lh,
        hl,
        hh,
        width: padded_w,
        height: padded_h,
        x_offset: -(pad as isize),
        y_offset: -(pad as isize),
    }
}

#[inline]
fn symmetric_reflect(i: isize, len: isize) -> isize {
    if len <= 0 {
        return 0;
    }
    let mut v = i;
    while v < 0 || v >= len {
        if v < 0 {
            v = -v - 1;
        }
        if v >= len {
            v = 2 * len - v - 1;
        }
    }
    v
}

/// Daubechies-8 high-pass decomposition filter (16 taps). Same as
/// JPEG + H.264 J-UNIWARD modules.
const HPDF: [f64; 16] = [
    -0.0544158422,
    0.3128715909,
    -0.6756307363,
    0.5853546837,
    0.0158291053,
    -0.2840155430,
    -0.0004724846,
    0.1287474266,
    0.0173693010,
    -0.0440882539,
    -0.0139810279,
    0.0087460940,
    0.0048703530,
    -0.0003917404,
    -0.0006754494,
    -0.0001174768,
];

const FILT_LEN: usize = 16;

#[inline]
fn lpdf() -> [f64; 16] {
    let mut lp = [0.0f64; 16];
    for n in 0..16 {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        lp[n] = sign * HPDF[15 - n];
    }
    lp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_planes(w: usize, h: usize) -> FramePlanes {
        let mut y = vec![0u8; w * h];
        for row in 0..h {
            for col in 0..w {
                y[row * w + col] = ((row * 7 + col * 3) & 0xff) as u8;
            }
        }
        let cw = w / 2;
        let ch = h / 2;
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        for row in 0..ch {
            for col in 0..cw {
                cb[row * cw + col] = ((row * 11 + col * 5 + 13) & 0xff) as u8;
                cr[row * cw + col] = ((row * 13 + col * 7 + 31) & 0xff) as u8;
            }
        }
        FramePlanes {
            y,
            cb,
            cr,
            luma_width: w,
            luma_height: h,
            chroma_width: cw,
            chroma_height: ch,
        }
    }

    #[test]
    fn dct_basis_dc_is_constant() {
        // (freq_y, freq_x) = (0, 0) → DC basis is uniform 1/sqrt(N)
        // per dim → constant value across the block.
        let mut basis = vec![0.0; 16];
        dct_ii_basis(4, 4, 0, 0, &mut basis);
        let expected = 0.25; // (1/sqrt(4)) * (1/sqrt(4)) = 1/2 * 1/2 = 0.25
        for v in basis {
            assert!((v - expected).abs() < 1e-9, "DC basis should be {} got {}", expected, v);
        }
    }

    #[test]
    fn dct_basis_first_ac_alternates_along_x() {
        // (freq_y, freq_x) = (0, 1) → cosine along x, constant along y.
        // For 4×4 block: each row should be the same alternating pattern.
        let mut basis = vec![0.0; 16];
        dct_ii_basis(4, 4, 1, 0, &mut basis);
        for row in 1..4 {
            for col in 0..4 {
                assert!(
                    (basis[row * 4 + col] - basis[col]).abs() < 1e-9,
                    "row {} col {} should match row 0",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn uniform_costs_are_finite_and_non_uniform() {
        // Build a small set of AC positions on a synthetic Y plane and
        // verify the returned costs are (a) all finite, (b) not all
        // equal — proves J-UNIWARD is distinguishing positions.
        let planes = synth_planes(32, 32);
        let positions = vec![
            Av1FramePosition {
                plane: 0,
                plane_px_x: 0,
                plane_px_y: 0,
                tx_width_log2: 2,
                tx_height_log2: 2,
                tx_type: 0,
                scan_pos: 1,
                coeff_magnitude: 8,
            },
            Av1FramePosition {
                plane: 0,
                plane_px_x: 8,
                plane_px_y: 8,
                tx_width_log2: 2,
                tx_height_log2: 2,
                tx_type: 0,
                scan_pos: 5,
                coeff_magnitude: 8,
            },
            Av1FramePosition {
                plane: 0,
                plane_px_x: 16,
                plane_px_y: 16,
                tx_width_log2: 3,
                tx_height_log2: 3,
                tx_type: 0,
                scan_pos: 7,
                coeff_magnitude: 8,
            },
        ];
        let costs = compute_av1_uniward_costs(&planes, &positions, 30);
        assert_eq!(costs.len(), 3);
        for &c in &costs {
            assert!(c.is_finite(), "cost should be finite, got {}", c);
            assert!(c > 0.0, "cost should be positive, got {}", c);
        }
        // Not all equal.
        let all_same = costs.iter().all(|&c| (c - costs[0]).abs() < 1e-6);
        assert!(
            !all_same,
            "expected non-uniform costs across positions; got all {}",
            costs[0]
        );
    }

    #[test]
    fn cascade_proxy_adds_positive_increment_to_cost() {
        // Phase B.1.3: verify the cascade-magnitude-proxy term
        // contributes positively. Compare cost on a textured frame
        // (where wavelet energy is high → low J-UNIWARD cost) with
        // and without the cascade term.
        let planes = synth_planes(32, 32);
        let position = Av1FramePosition {
            plane: 0,
            plane_px_x: 8,
            plane_px_y: 8,
            tx_width_log2: 2,
            tx_height_log2: 2,
            tx_type: 0,
            scan_pos: 5,
            coeff_magnitude: 8,
        };
        let costs = compute_av1_uniward_costs(&planes, &[position], 30);
        // The cost should be > the J-UNIWARD-only value (i.e., the
        // cascade term is non-zero). Hard-asserting a floor is
        // brittle, but the cost itself must be finite + positive.
        assert!(
            costs[0].is_finite(),
            "cascade-augmented cost must remain finite, got {}",
            costs[0]
        );
        assert!(costs[0] > 0.0, "expected positive cost, got {}", costs[0]);
        // At qindex=30, q_scale ≈ 8.5; delta_magnitude = 17. Max
        // DCT-II basis value at AC freq ≈ 0.5. So
        // cascade_magnitude_proxy >= ~4. With LAMBDA_CASCADE = 1.0,
        // this contributes ≥ 4 to cost. Reasonable lower bound.
        assert!(
            costs[0] >= 4.0,
            "cost should include cascade term contribution (>= 4 at q30, AC pos), got {}",
            costs[0]
        );
    }

    #[test]
    fn idtx_returns_uniform_cost() {
        let planes = synth_planes(16, 16);
        let positions = vec![Av1FramePosition {
            plane: 0,
            plane_px_x: 0,
            plane_px_y: 0,
            tx_width_log2: 2,
            tx_height_log2: 2,
            tx_type: AV1_TX_TYPE_IDTX,
            scan_pos: 5,
            coeff_magnitude: 8,
        }];
        let costs = compute_av1_uniward_costs(&planes, &positions, 30);
        assert_eq!(costs, vec![1.0]);
    }
}
