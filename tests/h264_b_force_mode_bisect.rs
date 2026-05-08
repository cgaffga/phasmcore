// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Task #209 — force-mode bisect for the B-frame encoder.recon vs
// ffmpeg.decode divergence. Inherits PHASM_B_FORCE_MODE from the
// caller's env.
//
// Two probes:
//
// 1. recon-vs-decode (same as h264_b_recon_vs_decode but parametric
//    on PHASM_B_FORCE_MODE).
//
// 2. dpb-vs-prior-recon: BEFORE encoding each P-frame, dump
//    enc.dpb.last_ref.y and compare against the previously recorded
//    encoder.recon for the prior reference frame. Pinpoints whether
//    DPB.last_ref is byte-identical to the prior P-frame's recon at
//    the moment of next P-frame's start (= what ffmpeg sees as the
//    L0 reference).

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::stego::gop_pattern::{
    iter_encode_order, FrameType, GopPattern,
};

#[test]
#[ignore]
fn force_mode_bisect_recon_vs_decode() {
    let yuv_path = "/tmp/iphone7_1920x1072_f10.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f10.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 10usize;
    let frame_size = (width * height * 3 / 2) as usize;

    let force_mode = std::env::var("PHASM_B_FORCE_MODE").unwrap_or_default();
    let label = if force_mode.is_empty() { "<unset>".to_string() } else { force_mode.clone() };

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
    }

    let pattern = if std::env::var("PHASM_TEST_IPPPP").is_ok() {
        GopPattern::Ipppp { gop: 30 }
    } else {
        GopPattern::Ibpbp { gop: 30, b_count: 1 }
    };
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;

    let mut bitstream = Vec::new();
    let mut recon_dumps: Vec<(u32, FrameType, Vec<u8>)> = Vec::new();
    let mut dpb_probes: Vec<(u32, Vec<u8>)> = Vec::new();
    // Track the previously-promoted reference recon so we can compare
    // dpb.last_ref to it at the start of each subsequent P-frame.
    let mut prev_p_recon: Option<Vec<u8>> = None;
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];

        // Probe DPB at start of P (before encode mutates it) so we
        // see the L0 reference exactly as ffmpeg would for this slice.
        if eo.frame_type == FrameType::P {
            let dpb_y = enc.dpb.last_ref.as_ref().map(|f| f.y.clone()).unwrap_or_default();
            dpb_probes.push((eo.display_idx, dpb_y));
        }

        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame),
            FrameType::P => enc.encode_p_frame(frame),
            FrameType::B => enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("encode (encode={}, display={}): {e}", eo.encode_idx, eo.display_idx));
        bitstream.extend_from_slice(&bytes);
        recon_dumps.push((eo.display_idx, eo.frame_type, enc.recon.y.clone()));

        // Track the most-recent P/I recon so the next P-frame probe
        // can compare against it (= what we expect dpb.last_ref to be).
        if matches!(eo.frame_type, FrameType::P | FrameType::Idr) {
            prev_p_recon = Some(enc.recon.y.clone());
        }
        let _ = &prev_p_recon; // not currently used in body, retained for future debug
    }

    let h264_path = std::env::temp_dir().join(format!("phasm_force_{}.h264", label));
    let dec_path = std::env::temp_dir().join(format!("phasm_force_{}.dec.yuv", label));
    std::fs::write(&h264_path, &bitstream).expect("write h264");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .status()
        .expect("ffmpeg run");
    assert!(status.success(), "ffmpeg decode failed (force_mode={label})");

    let decoded = std::fs::read(&dec_path).expect("read decoded yuv");
    let y_size = (width * height) as usize;

    // Measure |source - decoded| per frame to characterise visible
    // artifact level under this force_mode.
    eprintln!("\n=== force_mode={label} — SOURCE vs decode (visible) ===");
    let mut total_visible_bad = 0u64;
    for d in 0..n_frames {
        let off = d * frame_size;
        let src_y = &yuv[off..off + y_size];
        let dec_y = &decoded[off..off + y_size];
        let mut bad_pix50 = 0u32; // |Δ|>50 (gate threshold)
        let mut max_abs = 0u32;
        let mut sum_sq = 0u64;
        for (s, dc) in src_y.iter().zip(dec_y.iter()) {
            let dd = (*s as i32 - *dc as i32).unsigned_abs();
            sum_sq += (dd as u64) * (dd as u64);
            if dd > 50 { bad_pix50 += 1; }
            if dd > max_abs { max_abs = dd; }
        }
        let mse = (sum_sq as f64) / (y_size as f64);
        let psnr_y = if mse > 0.0 { 10.0 * (255.0_f64 * 255.0 / mse).log10() } else { 99.0 };
        total_visible_bad += bad_pix50 as u64;
        eprintln!(
            "  display={}  bad_pix>50={}  max|Δ|={}  PSNR_Y={:.2} dB",
            d, bad_pix50, max_abs, psnr_y,
        );
    }
    eprintln!("  TOTAL |source-decoded|>50 pixels across {n_frames} frames: {total_visible_bad}");

    eprintln!("\n=== force_mode={label} — recon vs decode ===");
    let mut total_diverging_pixels = 0u64;
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    for (display_idx, ft, enc_y) in &recon_dumps {
        let off = (*display_idx as usize) * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut diff = 0u64;
        let mut max_abs = 0u32;
        let mut nz_pixels = 0u32;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            diff += d as u64;
            if d > 0 {
                nz_pixels += 1;
                if d > max_abs { max_abs = d; }
            }
        }
        let avg = (diff as f64) / (y_size as f64);
        total_diverging_pixels += nz_pixels as u64;
        // Per-MB |Δ| histogram so we can find the FIRST diverging MB.
        let mut first_div_mb: Option<(usize, usize, u32)> = None;
        let mut mb_div_count = 0u32;
        for my in 0..mb_h {
            for mx in 0..mb_w {
                let mut mb_max = 0u32;
                for py in 0..16 {
                    for px in 0..16 {
                        let pix_y = my * 16 + py;
                        let pix_x = mx * 16 + px;
                        let i = pix_y * (width as usize) + pix_x;
                        let d = (enc_y[i] as i32 - dec_y[i] as i32).unsigned_abs();
                        if d > mb_max { mb_max = d; }
                    }
                }
                if mb_max > 0 {
                    mb_div_count += 1;
                    if first_div_mb.is_none() {
                        first_div_mb = Some((mx, my, mb_max));
                    }
                }
            }
        }
        eprintln!(
            "  display={}  type={:?}  avg|Δ|={:.3}  max|Δ|={}  nonzero_pixels={} ({:.2}%)  diverging_mbs={}/{}  first_div_mb={:?}",
            display_idx, ft, avg, max_abs, nz_pixels,
            100.0 * (nz_pixels as f64) / (y_size as f64),
            mb_div_count, mb_w * mb_h, first_div_mb,
        );
    }
    eprintln!("  TOTAL diverging Y pixels across {n_frames} frames: {total_diverging_pixels}");

    // DPB integrity report: for each P-probe, find the most recent
    // recon snapshot that was a reference (P or Idr) at a smaller
    // encode_idx, and compare to the dpb_y we captured at that P's
    // start. They should be byte-identical.
    eprintln!("\n=== force_mode={label} — DPB.last_ref integrity at P-frame entry ===");
    for (display_idx, dpb_y) in &dpb_probes {
        // Find the most recent prior P/Idr recon snapshot whose
        // display_idx differs from this one. recon_dumps are in
        // encode order; we want the one immediately prior in encode
        // order that is P/Idr.
        let mut prior_ref: Option<(u32, &Vec<u8>)> = None;
        for (d, ft, ry) in &recon_dumps {
            if *d == *display_idx { break; }
            if matches!(ft, FrameType::P | FrameType::Idr) {
                prior_ref = Some((*d, ry));
            }
        }
        match prior_ref {
            Some((prior_d, prior_y)) => {
                let mut diff = 0u64;
                let mut max_abs = 0u32;
                let mut nz = 0u32;
                for (a, b) in dpb_y.iter().zip(prior_y.iter()) {
                    let d = (*a as i32 - *b as i32).unsigned_abs();
                    diff += d as u64;
                    if d > 0 { nz += 1; if d > max_abs { max_abs = d; } }
                }
                eprintln!(
                    "  P at display={}: dpb.last_ref vs encoder.recon(display={}, prior P/I)  avg|Δ|={:.3}  max|Δ|={}  nonzero={}",
                    display_idx, prior_d,
                    (diff as f64) / (y_size as f64), max_abs, nz
                );
            }
            None => {
                eprintln!("  P at display={}: no prior reference recon to compare", display_idx);
            }
        }
    }
}
