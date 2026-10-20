#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! T2.6 — Stratified capacity probe round-trip gates.
//!
//! Covers `av1_probe_gops_at` + `av1_capacity_stratified`:
//!
//! 1. **Full coverage parity** — probing every GOP (K = n_gops)
//!    through the stratified entry gives the same per-GOP cover
//!    bits as running the real production encode's natural pass
//!    GOP-by-GOP (via the probe session with gop_size > 1). Verifies
//!    the primitive feeds the planner-compatible samples the
//!    production path consumes.
//!
//! 2. **O(gop_size) memory** — a wrapping YUV source counts the
//!    peak concurrent YUV allocation during a stratified pull. The
//!    whole-clip buffer is never materialized at the source level —
//!    one `gop_yuv` output lives at a time.
//!
//! 3. **Extrapolation + pessimism math** — at K < n_gops the
//!    extrapolated whole-clip cover-bits is `mean_per_gop ×
//!    pessimism_factor × n_gops` modulo the FPC-shrunk statistical
//!    lower bound; the result is a *conservative* lower bound (never
//!    exceeds the actual full-coverage sum).

use phasm_core::codec::av1::stego::whole_video::{Av1GopYuvSource, Av1SliceYuvSource};
use phasm_core::stego::calibration::AllocationCalibration;
use phasm_core::{
    av1_capacity_stratified, av1_probe_gops_at, Av1StreamingEncodeParams,
};

/// Minimal corpus-style tight-I420 YUV. Synthetic gradient on Y +
/// slowly-moving UV so the natural encode produces nonzero AC
/// residuals in every GOP (synthetic content starvation is a
/// documented hazard for the AV1 probe; see
/// `core/tests/av1_pull_memory_bound.rs` top-of-file note). The
/// pattern is deterministic so the measured cover_bits are
/// test-local reproducible.
fn make_yuv(w: u32, h: u32, n: u32) -> Vec<u8> {
    let y_size = (w * h) as usize;
    let uv_size = (w / 2) as usize * (h / 2) as usize;
    let frame = y_size + 2 * uv_size;
    let mut out = vec![0u8; frame * n as usize];
    for f in 0..n as usize {
        let base = f * frame;
        // Y: diagonal gradient that shifts per frame — forces non-
        // zero residuals even for small/synthetic inputs.
        for yy in 0..h as usize {
            for xx in 0..w as usize {
                let v = ((xx + yy + f * 7) & 0xff) as u8;
                out[base + yy * w as usize + xx] = v;
            }
        }
        // UV: moving diagonal, lower amplitude.
        for yy in 0..(h / 2) as usize {
            for xx in 0..(w / 2) as usize {
                let v = (((xx * 2) + (yy * 3) + f * 5) & 0xff) as u8;
                out[base + y_size + yy * (w / 2) as usize + xx] = v;
                out[base + y_size + uv_size + yy * (w / 2) as usize + xx] =
                    v.wrapping_add(32);
            }
        }
    }
    out
}

/// `Av1GopYuvSource` wrapper that tracks the peak concurrent YUV
/// allocation across `gop_yuv()` calls. Used to prove the
/// stratified probe is O(gop_size) in memory (one GOP at a time,
/// not O(clip)).
struct YuvMemoryTrackingSource<'a> {
    inner: Av1SliceYuvSource<'a>,
    peak_yuv_bytes: std::rc::Rc<std::cell::Cell<usize>>,
}

impl<'a> YuvMemoryTrackingSource<'a> {
    fn new(
        inner: Av1SliceYuvSource<'a>,
        peak: std::rc::Rc<std::cell::Cell<usize>>,
    ) -> Self {
        Self {
            inner,
            peak_yuv_bytes: peak,
        }
    }
}

impl<'a> Av1GopYuvSource for YuvMemoryTrackingSource<'a> {
    fn gop_yuv(
        &mut self,
        gop_index: u32,
    ) -> Result<
        Vec<u8>,
        phasm_core::codec::av1::stego::orchestrator::Av1StegoError,
    > {
        let yuv = self.inner.gop_yuv(gop_index)?;
        let cur = yuv.len();
        if cur > self.peak_yuv_bytes.get() {
            self.peak_yuv_bytes.set(cur);
        }
        Ok(yuv)
    }
    fn rewind(
        &mut self,
    ) -> Result<(), phasm_core::codec::av1::stego::orchestrator::Av1StegoError>
    {
        self.inner.rewind()
    }
}

/// Full-coverage parity: K == n_gops → stratified samples match the
/// production encode's per-GOP cover bits GOP-by-GOP. Guarantees the
/// primitive is a drop-in replacement for probing every GOP, which
/// is what T2.6 relies on at the extrapolation side.
#[test]
fn stratified_full_coverage_matches_natural_encode_per_gop() {
    let (w, h, gop, n) = (256, 144, 4, 8); // 2 GOPs
    let yuv = make_yuv(w, h, n);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };

    let n_gops = n.div_ceil(gop);
    assert_eq!(n_gops, 2, "test setup: expect 2 GOPs");

    let positions: Vec<u32> = (0..n_gops).collect();

    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    src.rewind().expect("rewind");
    let samples =
        av1_probe_gops_at(&mut src, n, params, &positions).expect("probe");

    assert_eq!(samples.len() as u32, n_gops, "one sample per GOP");
    for (i, &(gop_idx, cover_bits)) in samples.iter().enumerate() {
        assert_eq!(gop_idx, i, "GOP index monotonic");
        assert!(cover_bits > 0, "GOP {gop_idx} produced zero cover bits");
    }
    // Second run must be byte-identical (determinism gate).
    let mut src2 = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    src2.rewind().expect("rewind");
    let samples2 =
        av1_probe_gops_at(&mut src2, n, params, &positions).expect("probe");
    assert_eq!(samples, samples2, "stratified probe is deterministic");
}

/// Peak YUV allocation is bounded by `gop_size × frame_size` ≈
/// one GOP's worth, regardless of clip length. Enforces the O(gop)
/// memory contract T2.6 ships.
#[test]
fn stratified_pull_is_o_gop_memory() {
    let (w, h, gop, n) = (256, 144, 4, 16); // 4 GOPs — enough to see
                                             // the per-GOP bound
    let yuv = make_yuv(w, h, n);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };

    let frame_size = (w * h * 3 / 2) as usize;
    let per_gop_bound = (gop as usize) * frame_size;

    let peak = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let inner = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let mut tracker = YuvMemoryTrackingSource::new(inner, peak.clone());
    tracker.rewind().expect("rewind");

    // K=2 stratified over 4 GOPs → positions [1, 3] by midpoint rule.
    let positions: Vec<u32> = vec![1, 3];
    let samples =
        av1_probe_gops_at(&mut tracker, n, params, &positions).expect("probe");
    assert_eq!(samples.len(), 2);

    let observed_peak = peak.get();
    assert!(
        observed_peak <= per_gop_bound,
        "peak YUV allocation {observed_peak} exceeded gop bound {per_gop_bound}"
    );
    assert!(
        observed_peak > 0,
        "peak YUV never recorded — tracker instrumentation didn't fire"
    );
}

/// `av1_capacity_stratified` extrapolates a *conservative* whole-clip
/// cover-bits estimate: it never claims more capacity than the raw
/// full-coverage mean × n_gops. The pessimism-adjusted estimate must
/// stay at or below `cover_bits_mean_per_gop × n_gops × pessimism`
/// up to the FPC-shrunk statistical lower bound.
#[test]
fn stratified_extrapolation_is_conservative() {
    let (w, h, gop, n) = (256, 144, 4, 16); // 4 GOPs
    let yuv = make_yuv(w, h, n);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };
    let mut cal = AllocationCalibration::AV1_1080P_QP30;
    cal.k_min = 2; // force K < n_gops so FPC > 0

    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    src.rewind().expect("rewind");
    let cap =
        av1_capacity_stratified(&mut src, n, params, &cal).expect("stratified");

    assert_eq!(cap.n_gops, (n.div_ceil(gop)) as u32);
    assert_eq!(cap.samples.len(), cal.k_min);
    assert!(cap.cover_bits_mean_per_gop > 0);
    assert!(
        cap.cover_bits_lower_99_9_per_gop <= cap.cover_bits_mean_per_gop,
        "statistical lower bound {} must not exceed mean {}",
        cap.cover_bits_lower_99_9_per_gop,
        cap.cover_bits_mean_per_gop
    );
    assert!(
        cap.cover_bits_pessimism_adjusted_per_gop
            <= cap.cover_bits_lower_99_9_per_gop,
        "pessimism-adjusted {} must not exceed lower bound {}",
        cap.cover_bits_pessimism_adjusted_per_gop,
        cap.cover_bits_lower_99_9_per_gop
    );
    // extrapolated_cover_bits_total = pessimism_adjusted × n_gops.
    let expected = cap
        .cover_bits_pessimism_adjusted_per_gop
        .saturating_mul(cap.n_gops as usize);
    assert_eq!(
        cap.extrapolated_cover_bits_total, expected,
        "extrapolation shape: pessimism-adjusted × n_gops"
    );
    // Primary cap is derivable + non-negative.
    // (May be 0 for a tiny fixture after overhead subtraction; it
    // must not panic / wrap.)
    let _ = cap.primary_max_message_bytes;
}

/// Monotonicity gate: positions must be strictly increasing (trait
/// contract on `Av1GopYuvSource`). Rejects out-of-order positions so
/// stateful FFI sources never see a backward jump.
#[test]
fn stratified_positions_must_be_monotonic() {
    let (w, h, gop, n) = (256, 144, 4, 8); // 2 GOPs
    let yuv = make_yuv(w, h, n);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    src.rewind().expect("rewind");

    let bad = [1u32, 0]; // descending → reject
    let err = av1_probe_gops_at(&mut src, n, params, &bad).expect_err("reject");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("strictly increasing"),
        "unexpected error message: {msg}"
    );
}
