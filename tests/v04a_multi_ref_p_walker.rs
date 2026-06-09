// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// V0.4.A.2 — multi-ref P investigation regression gate.
//
// V0.4.A.1 (commit ad648e3b) bumped `param.iNumRefFrame` from 1 to 2,
// claiming "multi-ref P + per-MB ref_idx_l0 emission." Empirical
// measurement (this file) and fork code reading
// (`encoder_ext.cpp:2869` "always get item 0 due to reordering done",
// `au_set.cpp:130` unconditional iNumRefFrame rewrite) showed OH264 cannot
// deliver real multi-ref P: ME is hardwired single-ref, and the runtime
// cap rewrites iNumRefFrame back to 1. Net effect of A.1 was an SPS-only
// shift with no per-MB encoding change.
//
// Real-world corpus survey showed the SPS=2 / per-MB-single-ref pattern
// matches only the Lumix G9 cohort (rare), while pre-A.1 (SPS=1) matches
// the dominant phone-camera cohort (iPhone5/7, DJI). A.1 was a
// net stealth loss; reverted 2026-05-23. Full investigation in
// `memory/h264_v04a_multi_ref_p_negative.md`.
//
// This test is the regression GATE for the revert decision:
//   1. SPS must signal `max_num_ref_frames == 1` (matches iPhone7 cohort).
//   2. All P-slices must have `num_ref_idx_l0_active == 1` (no ref_idx_l0
//      bins emitted, matching universal real-world phone-camera Layer 3).
//
// If a future change touches `phasm_encoder_shim.cc::iNumRefFrame` or the
// fork's SPS/slice-header emission, these assertions will fail and force
// re-reading the negative-result memory before shipping.

#![cfg(feature = "h264-encoder")]

mod common;
use common::oh264_stream;

use phasm_core::codec::h264::bitstream::parse_nal_units_annexb;
use phasm_core::codec::h264::slice::parse_slice_header;
use phasm_core::codec::h264::sps::{parse_pps, parse_sps};
use phasm_core::codec::h264::NalType;
use std::sync::{Mutex, OnceLock};

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

/// Deterministic content-rich YUV. Same LCG pattern the cross-arch
/// determinism test uses — non-flat content so the encoder actually
/// engages mode decision (which is what we want to observe).
fn deterministic_yuv(w: u32, h: u32, n_frames: u32) -> Vec<u8> {
    let frame_size = (w * h * 3 / 2) as usize;
    let mut out = Vec::with_capacity(frame_size * n_frames as usize);
    let mut s: u32 = 0x1234_5678;
    for _ in 0..n_frames {
        for _ in 0..frame_size {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            out.push((s >> 16) as u8);
        }
    }
    out
}

#[test]
fn v04a_sps_signals_single_ref_frame_matches_iphone7_cohort() {
    let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());

    // Smallest 16-aligned content-rich YUV with a P-frame: 64x48x3.
    // Frames 0 = IDR, 1+ = P (single GOP, so all post-IDR frames are P).
    let w = 64u32;
    let h = 48u32;
    let n_frames = 3u32;
    let yuv = deterministic_yuv(w, h, n_frames);

    let annex_b = oh264_stream::encode(&yuv, w, h, n_frames, 26, "x", "p")
        .expect("OH264 stego encode must succeed on tiny deterministic YUV");

    let nals = parse_nal_units_annexb(&annex_b).expect("Annex-B parse");
    let sps_nal = nals
        .iter()
        .find(|n| n.nal_type == NalType::SPS)
        .expect("OH264 must emit SPS");
    let sps = parse_sps(&sps_nal.rbsp).expect("SPS parse");

    // V0.4.A.1 revert gate: SPS must say max_num_ref_frames=1, matching
    // iPhone5/7/DJI/most Artlist (dominant phone-camera cohort).
    // If this fails, someone bumped iNumRefFrame again — re-read
    // memory/h264_v04a_multi_ref_p_negative.md before shipping.
    assert_eq!(
        sps.max_num_ref_frames, 1,
        "V0.4.A revert: SPS.max_num_ref_frames must be 1 (iPhone7 cohort match). \
         Re-read memory/h264_v04a_multi_ref_p_negative.md before bumping."
    );
}

#[test]
fn v04a_slice_header_num_active_l0_observation() {
    let _g = session_guard().lock().unwrap_or_else(|p| p.into_inner());

    let w = 64u32;
    let h = 48u32;
    // 5 frames: IDR + 4 P. Gives the encoder several P-slices to
    // potentially use the older reference. Single GOP, so all four
    // post-IDR frames are P-slices.
    let n_frames = 5u32;
    let yuv = deterministic_yuv(w, h, n_frames);

    let annex_b = oh264_stream::encode(&yuv, w, h, n_frames, 26, "x", "p")
        .expect("OH264 stego encode");

    let nals = parse_nal_units_annexb(&annex_b).expect("Annex-B parse");
    let sps_nal = nals
        .iter()
        .find(|n| n.nal_type == NalType::SPS)
        .expect("SPS");
    let pps_nal = nals
        .iter()
        .find(|n| n.nal_type == NalType::PPS)
        .expect("PPS");
    let sps = parse_sps(&sps_nal.rbsp).expect("SPS parse");
    let pps = parse_pps(&pps_nal.rbsp).expect("PPS parse");

    // Diagnostic: what does PPS default to?
    eprintln!(
        "V0.4.A.2 PPS.num_ref_idx_l0_default = {} (active refs the encoder \
         relies on per slice unless overridden)",
        pps.num_ref_idx_l0_default
    );

    // Find every P-slice and inspect its num_ref_idx_l0_active.
    let mut p_slices_seen = 0;
    let mut p_slices_with_active_ge_2 = 0;
    for n in &nals {
        if !matches!(n.nal_type, NalType::SLICE) {
            continue;
        }
        let hdr = parse_slice_header(&n.rbsp, &sps, &pps, n.nal_type, n.nal_ref_idc)
            .expect("slice header parse");
        // SliceType::P is value 0/5; we just want the active L0 count
        // for all non-IDR inter slices.
        eprintln!(
            "V0.4.A.2 slice_type={:?} num_ref_idx_l0_active={}",
            hdr.slice_type, hdr.num_ref_idx_l0_active
        );
        p_slices_seen += 1;
        if hdr.num_ref_idx_l0_active >= 2 {
            p_slices_with_active_ge_2 += 1;
        }
    }
    eprintln!(
        "V0.4.A.2 summary: {} P-slices seen, {} with active_l0 >= 2",
        p_slices_seen, p_slices_with_active_ge_2
    );

    assert!(
        p_slices_seen > 0,
        "fixture must contain at least one P-slice for this measurement"
    );
    // V0.4.A revert gate: zero P-slices with active_l0 >= 2 means OH264
    // emits no per-MB ref_idx_l0 bins, matching universal real-world
    // phone-camera Layer 3 fingerprint. If non-zero, the encoder has
    // started emitting multi-ref bins (unexpected on OH264) — investigate
    // before shipping.
    assert_eq!(
        p_slices_with_active_ge_2, 0,
        "V0.4.A revert: expected zero P-slices with active_l0 >= 2 on OH264. \
         Re-read memory/h264_v04a_multi_ref_p_negative.md."
    );
}
