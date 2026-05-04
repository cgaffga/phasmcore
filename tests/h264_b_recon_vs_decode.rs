// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §B-RDO.debug.4 (2026-05-04) — compare phasm encoder's internal
// recon for each B-frame to ffmpeg's decode of the same bitstream.
//
// Encoder writes self.recon as it goes; for B-frames the recon is
// `pred + residual` (matching what ffmpeg's decoder produces from
// the bitstream IF math is spec-correct on both sides).
//
// If enc.recon == ffmpeg.decode for B-frames → encoder is honest;
// the visual artifact is the result of encoder PICKING choices that
// produce a poor recon (mode decision, MV quality). The fix is in
// the mode-decision / ME path, not the residual emission.
//
// If enc.recon != ffmpeg.decode → encoder/decoder semantic
// divergence. Encoder writes one thing to its recon buffer (used as
// reference for next P/B), but the bitstream represents something
// different (which ffmpeg renders). The fix is to align encoder
// math with the bitstream's true semantics.

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::stego::gop_pattern::{
    iter_encode_order, FrameType, GopPattern,
};

#[test]
#[ignore]
fn dump_b_frame_recon_vs_decode() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 6usize; // small enough for fast iteration; covers I0 P2 B1 P4 B3 P5
    let frame_size = (width * height * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;

    // Build the bitstream + capture enc.recon Y plane after every frame in encode order.
    let mut bitstream = Vec::new();
    let mut recon_dumps: Vec<(u32, FrameType, Vec<u8>)> = Vec::new();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame),
            FrameType::P => enc.encode_p_frame(frame),
            FrameType::B => enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("encode (encode_idx={}, display={}): {e}", eo.encode_idx, eo.display_idx));
        bitstream.extend_from_slice(&bytes);
        recon_dumps.push((eo.display_idx, eo.frame_type, enc.recon.y.clone()));
    }

    // Write the bitstream + decode it via ffmpeg.
    let h264_path = std::env::temp_dir().join("phasm_b_recon_probe.h264");
    let dec_path = std::env::temp_dir().join("phasm_b_recon_probe.dec.yuv");
    std::fs::write(&h264_path, &bitstream).expect("write h264");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .status()
        .expect("ffmpeg run");
    assert!(status.success(), "ffmpeg decode failed");

    let decoded = std::fs::read(&dec_path).expect("read decoded yuv");
    let y_size = (width * height) as usize;

    eprintln!("\n=== Encoder.recon vs ffmpeg.decode (Y plane) ===");
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
        eprintln!(
            "  display={}  type={:?}  avg|Δ|={:.3}  max|Δ|={}  nonzero_pixels={} ({:.2}%)",
            display_idx, ft, avg, max_abs, nz_pixels,
            100.0 * (nz_pixels as f64) / (y_size as f64)
        );
    }
}
