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
//!    `(gops_done, gops_total, estimate_bytes)` shape.
//! 2. **Monotonic** — `gops_done` increases strictly; `gops_total` is
//!    constant within a sweep (an escalated probe adds sweeps, see
//!    `weak_lower_bound_escalates_instead_of_reporting_zero`).
//! 3. **Final event converges** — on the last event, `gops_done ==
//!    gops_total == K`, and the running estimate is exactly the
//!    result's `max_message_bytes()`: the figure the HUD shows while
//!    probing is the policy maximum it shows once the probe is done,
//!    not the deliverable STC ceiling (which overshot about 2×).

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
                // Gradient plus hashed noise: enough residual for a
                // nonzero capacity at this tiny size.
                let n = ((xx * 7919) ^ (yy * 104_729) ^ (f * 31)).wrapping_mul(2_654_435_761) >> 24;
                let v = ((xx + yy + f * 7 + (n & 0x3f)) & 0xff) as u8;
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
        quantizer: 10,
        gop_size: gop,
        total_frames_hint: n,
        colour: Default::default(),
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

    // (5) final event: gops_done == gops_total == K
    let last = events.last().expect("at least one event");
    assert_eq!(last.0, total, "final gops_done == gops_total");
    assert_eq!(last.0 as usize, cal.k_min);

    // (6) final running estimate = the policy maximum the HUD shows
    //     once the probe finishes (not the deliverable STC ceiling).
    assert!(result.max_message_bytes() > 0, "cover too small to test the estimate: {result:?}");
    assert_eq!(
        last.2,
        result.max_message_bytes() as u64,
        "final estimate must equal max_message_bytes()"
    );
    assert!(
        last.2 <= result.primary_max_message_bytes as u64,
        "estimate must never exceed the deliverable primary"
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
        colour: Default::default(),
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
}

/// Frames of GOP `g` (of `gop` frames) carry hashed noise when `g` is odd
/// and are a flat gradient otherwise: a clip whose GOP cover is so uneven
/// that a few stratified samples (K = 2 of 6 picks GOPs 1 and 4) put the
/// 99.9 % lower bound at 0.
fn make_uneven_yuv(w: u32, h: u32, n: u32, gop: u32) -> Vec<u8> {
    let mut out = make_yuv(w, h, n);
    let y_size = (w * h) as usize;
    let frame = y_size + 2 * ((w / 2) * (h / 2)) as usize;
    for f in 0..n as usize {
        if (f as u32 / gop) % 2 == 1 {
            continue;
        }
        for yy in 0..h as usize {
            for xx in 0..w as usize {
                out[f * frame + yy * w as usize + xx] = (64 + (xx + yy) / 8) as u8;
            }
        }
    }
    out
}

/// A weak lower bound (the samples disagree too much) escalates the probe
/// to more GOPs instead of reporting 0. Before: IMG_4138, 20 GOPs at 720p,
/// 8 samples from 226 k to 3.83 M cover bits, lower bound 0, max 0 B for a
/// clip whose full probe measures 472 874 B.
#[test]
fn weak_lower_bound_escalates_instead_of_reporting_zero() {
    let (w, h, gop, n) = (256, 144, 4, 24); // 6 GOPs
    let yuv = make_uneven_yuv(w, h, n, gop);
    let params = Av1StreamingEncodeParams { width: w, height: h, quantizer: 10, gop_size: gop, total_frames_hint: n, colour: Default::default() };
    let mut cal = AllocationCalibration::AV1_1080P_QP30;
    cal.k_min = 2;
    let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::<(u32, u32, u64)>::new()));
    let ev = std::rc::Rc::clone(&events);
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    src.rewind().unwrap();
    let result = av1_capacity_stratified_with_progress(&mut src, n, params, &cal, &mut move |d, t, e| {
        ev.borrow_mut().push((d, t, e));
    })
    .expect("probe");
    assert!(result.samples.len() > cal.k_min, "no escalation: {:?}", result.samples);
    assert!(
        result.cover_bits_lower_99_9_per_gop * 4 >= result.cover_bits_mean_per_gop * 3,
        "bound still weak after escalation: {result:?}"
    );
    assert!(result.max_message_bytes() > 0, "{result:?}");
    let events = events.borrow();
    assert_eq!(events.len(), result.samples.len(), "one event per probed GOP");
    for pair in events.windows(2) {
        assert_eq!(pair[1].0, pair[0].0 + 1, "gops_done counts on across sweeps");
    }
    let last = *events.last().unwrap();
    assert_eq!((last.0, last.1), (result.samples.len() as u32, result.samples.len() as u32));
    assert_eq!(last.2, result.max_message_bytes() as u64, "final event = result");
}
