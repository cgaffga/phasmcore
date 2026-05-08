// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Task #209 root-cause diagnostic — bin-level trace probe for Option A
// (Direct-with-residual emission). Encodes a minimal I+P+B sequence at
// 16x16 (single MB) with PHASM_B_FORCE_MODE=direct +
// PHASM_DIRECT_WITH_RESIDUAL=1, captures the encoder's CABAC bin
// sequence for the B-slice, and checks both phasm's walker and ffmpeg
// for parser failures. Three diagnostic outcomes:
//
//   1. Walker OK, ffmpeg OK            → emission spec-correct.
//   2. Walker OK, ffmpeg fails          → both my encoder + walker are
//                                          non-spec the SAME way.
//   3. Walker fails (any error)         → my encoder + walker disagree
//                                          internally on the bin schedule.
//
// The encoder CABAC trace is printed in (2) / (3) so the FIRST divergent
// or surprising bin can be identified visually.

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::cabac::bin_decoder::slice::walk_annex_b_for_cover;
use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};

// Build a synthetic 16x16 yuv420p frame whose pixel values vary by
// position so the encoder must do work (vs flat content that quantizes
// to all zeros and produces CBP=0).
fn synth_frame(seed: u8, w: u32, h: u32) -> Vec<u8> {
    let y_size = (w * h) as usize;
    let c_size = (w * h / 4) as usize;
    let mut out = Vec::with_capacity(y_size + 2 * c_size);
    // Sharp-edge stripe pattern that SHIFTS by `seed` pixels per
    // frame. Produces strong low-frequency DCT coefficients that
    // survive trellis-quant + spatial-direct prediction (zero MV)
    // can't perfectly compensate the shift → guaranteed non-zero CBP
    // on Direct-with-residual at any non-trivial QP.
    for y in 0..h {
        for x in 0..w {
            let stripe = ((x.wrapping_add(seed as u32) / 8) & 1) as u8;
            out.push(if stripe != 0 { 200 } else { 60 });
            let _ = y;
        }
    }
    for _ in 0..(2 * c_size) {
        out.push(128);
    }
    out
}

#[test]
#[ignore]
fn b_direct_residual_trace_probe() {
    use phasm_core::codec::h264::stego::gop_pattern::{
        iter_encode_order, FrameType, GopPattern,
    };

    // Single-MB frame to keep the trace short + neighbours absent
    // (all ctxIdxInc=0). 16-aligned per encoder constructor's check.
    // Sweep size via env: PHASM_TRACE_PROBE_W / PHASM_TRACE_PROBE_H
    // (defaults to single-MB 16x16 = clean baseline).
    let width = std::env::var("PHASM_TRACE_PROBE_W")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(16);
    let height = std::env::var("PHASM_TRACE_PROBE_H")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(16);
    assert!(
        width.is_multiple_of(16) && height.is_multiple_of(16),
        "dimensions must be 16-aligned; got {width}x{height}",
    );

    unsafe {
        std::env::set_var("PHASM_B_FORCE_MODE", "direct");
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DIRECT_WITH_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
    }

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    let quality = std::env::var("PHASM_TRACE_PROBE_QUALITY")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(26);
    let mut enc = Encoder::new(width, height, Some(quality)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.enable_cabac_trace();

    // Encode I + P + B (the B is the only Direct-with-residual MB
    // we care to trace). When PHASM_TRACE_PROBE_USE_IPHONE7=1, swap
    // synthetic content for the iphone7 fixture (display 0/1/2).
    let use_iphone7 = std::env::var("PHASM_TRACE_PROBE_USE_IPHONE7").is_ok();
    let mut bitstream = Vec::new();
    let frames: Vec<Vec<u8>> = if use_iphone7 {
        let yuv = std::fs::read("/tmp/iphone7_1920x1072_f10.yuv")
            .expect("missing /tmp/iphone7_1920x1072_f10.yuv");
        let frame_size = (width * height * 3 / 2) as usize;
        assert!(yuv.len() >= 3 * frame_size, "iphone7 fixture too short");
        (0..3).map(|d| yuv[d * frame_size..(d + 1) * frame_size].to_vec()).collect()
    } else {
        vec![
            synth_frame(40, width, height),
            synth_frame(80, width, height),
            synth_frame(120, width, height),
        ]
    };
    for eo in iter_encode_order(3, pattern) {
        let f = &frames[eo.display_idx as usize];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode (encode_idx={}, display={}): {e}", eo.encode_idx, eo.display_idx));
        bitstream.extend_from_slice(&bytes);
    }

    let trace = enc.take_cabac_trace();
    eprintln!("\n=== Encoder CABAC trace ({} bins, fixture {width}x{height}) ===", trace.len());
    // Only dump the trace for tiny fixtures (else it's noise).
    if width <= 32 && height <= 32 {
        for (i, line) in trace.iter().enumerate() {
            eprintln!("  [{i:4}] {line}");
        }
    }

    // Try phasm walker.
    eprintln!("\n=== phasm walker on Option A bytes ===");
    let walker_result = walk_annex_b_for_cover(&bitstream);
    match &walker_result {
        Ok(out) => eprintln!(
            "  walker OK: n_mb={}, n_slices={}",
            out.n_mb, out.n_slices,
        ),
        Err(e) => eprintln!("  walker ERR: {e:?}"),
    }

    // Try ffmpeg decode + check concealment.
    let h264_path = std::env::temp_dir().join("phasm_direct_trace.h264");
    let dec_path = std::env::temp_dir().join("phasm_direct_trace.dec.yuv");
    std::fs::write(&h264_path, &bitstream).expect("write h264");
    let ffmpeg_out = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "info", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .output()
        .expect("ffmpeg run");
    let stderr = String::from_utf8_lossy(&ffmpeg_out.stderr);
    let conceal_lines: Vec<&str> = stderr
        .lines()
        .filter(|l| l.contains("concealing") || l.to_lowercase().contains("error"))
        .collect();
    eprintln!("\n=== ffmpeg decode (status={}) ===", ffmpeg_out.status);
    if conceal_lines.is_empty() {
        eprintln!("  ffmpeg OK (no concealment / errors)");
    } else {
        for l in &conceal_lines {
            eprintln!("  {l}");
        }
    }

    // Diagnostic only — let the test always pass; we read the eprintln.
    let _ = walker_result;
}
