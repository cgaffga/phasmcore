#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! T2.6 — Stratified capacity probe PROGRESS CALLBACK gates.
//!
//! Covers `av1_capacity_stratified_with_progress` +
//! `av1_probe_gops_at_with_progress`:
//!
//! 1. **Callback fires once per probed GOP** with the correct
//!    `(gops_done, gops_total, cover_bits_so_far)` shape.
//! 2. **Monotonic + convergent** — `gops_done` increases strictly;
//!    `gops_total` is constant; `cover_bits_so_far` is
//!    non-decreasing.
//! 3. **Final event converges** — on the last event, `gops_done ==
//!    gops_total == K`, and the running cover-bits total equals the
//!    sum over the samples the non-progress API returns on the same
//!    inputs (both methods walk the same encode path).

use phasm_core::codec::av1::stego::whole_video::Av1SliceYuvSource;
use phasm_core::codec::av1::stego::whole_video::Av1GopYuvSource;
use phasm_core::stego::calibration::AllocationCalibration;
use phasm_core::{
    av1_capacity_stratified, av1_capacity_stratified_with_progress,
    Av1StreamingEncodeParams,
};

/// Deterministic synthetic YUV — mirrors the shape used by
/// `av1_stratified_probe.rs::make_yuv` so the probe sees enough
/// non-zero residuals to produce comparable cover-bit counts.
fn make_yuv(w: u32, h: u32, n: u32) -> Vec<u8> {
    let y_size = (w * h) as usize;
    let uv_size = (w / 2) as usize * (h / 2) as usize;
    let frame = y_size + 2 * uv_size;
    let mut out = vec![0u8; frame * n as usize];
    for f in 0..n as usize {
        let base = f * frame;
        for yy in 0..h as usize {
            for xx in 0..w as usize {
                let v = ((xx + yy + f * 7) & 0xff) as u8;
                out[base + yy * w as usize + xx] = v;
            }
        }
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

#[test]
fn progress_callback_fires_once_per_gop_and_converges() {
    let (w, h, gop, n) = (256, 144, 4, 16); // 4 GOPs
    let yuv = make_yuv(w, h, n);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };
    // Keep K below n_gops so the progress sequence has multiple
    // events (FPC path exercised too; `k_min = 3` for 4 GOPs).
    let mut cal = AllocationCalibration::AV1_1080P_QP30;
    cal.k_min = 3;

    // --- With progress ---
    let events: std::rc::Rc<std::cell::RefCell<Vec<(u32, u32, u64)>>> =
        std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let events_ref = std::rc::Rc::clone(&events);

    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    src.rewind().expect("rewind");
    let mut cb = move |done, total, bits| {
        events_ref.borrow_mut().push((done, total, bits));
    };
    let result = av1_capacity_stratified_with_progress(
        &mut src, n, params, &cal, &mut cb,
    )
    .expect("progress probe");

    let events = events.borrow().clone();

    // (1) one event per probed GOP
    assert_eq!(events.len(), result.samples.len(), "one event per probed GOP");
    assert_eq!(
        events.len(),
        cal.k_min,
        "exactly K events for the stratified K"
    );

    // (2) gops_total constant
    let total = events[0].1;
    for &(_, t, _) in &events {
        assert_eq!(t, total, "gops_total must stay constant across events");
    }
    assert_eq!(total as usize, cal.k_min);

    // (3) gops_done strictly increasing
    for pair in events.windows(2) {
        assert!(
            pair[1].0 > pair[0].0,
            "gops_done must strictly increase: {:?} then {:?}",
            pair[0],
            pair[1]
        );
    }

    // (4) cover_bits_so_far non-decreasing
    for pair in events.windows(2) {
        assert!(
            pair[1].2 >= pair[0].2,
            "cover_bits_so_far must be non-decreasing: {:?} then {:?}",
            pair[0],
            pair[1]
        );
    }

    // (5) final event: gops_done == gops_total == K
    let last = events.last().expect("at least one event");
    assert_eq!(last.0, total, "final gops_done == gops_total");
    assert_eq!(last.0 as usize, cal.k_min);

    // (6) final running cover = sum over samples (progress API matches
    //     the no-progress API on the exact same inputs/positions)
    let samples_sum: u64 = result.samples.iter().map(|&(_, c)| c as u64).sum();
    assert_eq!(
        last.2, samples_sum,
        "final cover_bits_so_far must equal sum of samples returned"
    );

    // --- Convergence vs non-progress API ---
    let mut src2 = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    src2.rewind().expect("rewind");
    let result_no_progress =
        av1_capacity_stratified(&mut src2, n, params, &cal).expect("no-progress probe");
    assert_eq!(
        result.samples, result_no_progress.samples,
        "progress + no-progress APIs must agree on samples"
    );
    assert_eq!(
        result.primary_max_message_bytes, result_no_progress.primary_max_message_bytes,
        "progress + no-progress APIs must agree on primary cap"
    );
    assert_eq!(
        result.extrapolated_cover_bits_total,
        result_no_progress.extrapolated_cover_bits_total,
        "progress + no-progress APIs must agree on extrapolation"
    );
}

#[test]
fn progress_callback_optional_at_full_coverage() {
    // K == n_gops: FPC vanishes, every GOP is probed. Just confirm the
    // callback still fires exactly K times and the final event
    // reports (K, K, _).
    let (w, h, gop, n) = (256, 144, 4, 8); // 2 GOPs → K == 2 at k_min=2
    let yuv = make_yuv(w, h, n);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };
    let mut cal = AllocationCalibration::AV1_1080P_QP30;
    cal.k_min = 2;

    let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let events_ref = std::rc::Rc::clone(&events);
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    src.rewind().expect("rewind");
    let mut cb = move |done, total, bits| {
        events_ref.borrow_mut().push((done, total, bits));
    };
    let _ = av1_capacity_stratified_with_progress(&mut src, n, params, &cal, &mut cb)
        .expect("full-coverage progress probe");

    let events = events.borrow();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].0, 1);
    assert_eq!(events[0].1, 2);
    assert_eq!(events[1].0, 2);
    assert_eq!(events[1].1, 2);
    assert!(events[1].2 >= events[0].2);
}
