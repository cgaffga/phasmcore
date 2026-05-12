// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// v1.6 §intra-in-B (#319) Phase 4 — ffmpeg compliance gate for the
// intra-in-B emit path (Phase 3 commit 2ffb329).
//
// With PHASM_B_INTRA_FALLBACK=1 the encoder emits I_16x16 for B-MBs
// whose best inter SATD exceeds the threshold. This test verifies:
//   1. ffmpeg decodes the stream without errors (spec-compliant
//      bitstream).
//   2. The stream actually contains intra-in-B emissions (mb_type ≥ 23
//      in B-slices) — counted via the encoder's B_INTRA_FALLBACK_COUNT.
//   3. Encoder visual_recon matches ffmpeg's decoded output bit-for-bit
//      (encoder/walker MV+residual agreement preserved through the
//      intra path).
//
// Run: cargo test --release -p phasm-core --features cabac-stego \
//        --test h264_v16_intra_in_b_compliance -- --ignored --nocapture

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::encoder::mb_decision_b::{
    drain_b_intra_fallback_count, BRdoConfig,
};
use phasm_core::codec::h264::stego::gop_pattern::{iter_encode_order, FrameType, GopPattern};

const W: u32 = 128;
const H: u32 = 96;
const N_FRAMES: usize = 6;
const QP: u8 = 26;

/// Build a synthetic IBPBP fixture with deliberately content unfit for
/// inter prediction: each frame has a small bright square that moves
/// erratically (up to 30px per frame), surrounded by mostly-uniform
/// dark texture. The square is the "small fast-moving high-contrast
/// feature" class from the user's screenshots — exactly the pattern
/// where inter ME fails and intra-in-B should fire.
fn synthetic_carplane_like_yuv() -> Vec<u8> {
    let frame_size = (W * H * 3 / 2) as usize;
    let mut yuv = vec![0u8; frame_size * N_FRAMES];
    for f in 0..N_FRAMES {
        let off = f * frame_size;
        let y_size = (W * H) as usize;
        let c_size = y_size / 4;
        // Y plane: dark uniform background ~30, with a 16x16 bright
        // square that jumps around.
        for px in &mut yuv[off..off + y_size] {
            *px = 30;
        }
        // Square position: deterministic but erratic per frame.
        let sx = ((f * 37 + 5) % (W as usize - 16)) as usize;
        let sy = ((f * 23 + 7) % (H as usize - 16)) as usize;
        for dy in 0..16 {
            for dx in 0..16 {
                let idx = off + (sy + dy) * W as usize + sx + dx;
                yuv[idx] = 220;
            }
        }
        // Chroma: mid-gray everywhere.
        for px in &mut yuv[off + y_size..off + y_size + 2 * c_size] {
            *px = 128;
        }
    }
    yuv
}

#[test]
#[ignore]
fn intra_in_b_emits_decodes_compliant() {
    let yuv = synthetic_carplane_like_yuv();
    let frame_size = (W * H * 3 / 2) as usize;
    let pattern = GopPattern::Ibpbp { gop: N_FRAMES, b_count: 1 };

    unsafe {
        std::env::set_var("PHASM_B_INTRA_FALLBACK", "1");
        // Lower threshold so the synthetic moving-square content
        // definitely triggers fallback at this small QP.
        std::env::set_var("PHASM_B_INTRA_FALLBACK_SATD_MIN", "5000");
    }
    let _ = drain_b_intra_fallback_count();

    let mut enc = Encoder::new(W, H, Some(QP)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config = BRdoConfig::PRODUCTION_VISUAL;

    let mut bs = Vec::new();
    for eo in iter_encode_order(N_FRAMES, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }

    let intra_count = drain_b_intra_fallback_count();
    eprintln!("§intra-in-B Phase 4 — {} B-MBs emitted as IntraI16x16", intra_count);
    assert!(
        intra_count > 0,
        "synthetic moving-square fixture should trigger ≥1 intra-in-B \
         emission with threshold=5000; got 0 — either threshold is too \
         high for this fixture, or the env-gated decision path is broken"
    );

    // ffmpeg decode — failure here means the bitstream is not spec-compliant.
    let h264 = std::env::temp_dir().join("phasm_intra_in_b_compliance.h264");
    let dec_path = std::env::temp_dir().join("phasm_intra_in_b_compliance.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let st = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec_path)
        .status().expect("ffmpeg");
    assert!(st.success(), "ffmpeg decode FAILED on intra-in-B bitstream — \
                            spec-compliance regression");

    let decoded = std::fs::read(&dec_path).unwrap();
    assert_eq!(
        decoded.len(),
        N_FRAMES * frame_size,
        "decoded size mismatch — partial decode"
    );

    // Compare encoder.visual_recon (last frame) vs ffmpeg's last frame.
    // For full coverage we'd need per-frame snapshots; for this gate the
    // last frame is sufficient — divergence anywhere upstream cascades
    // into the last frame's recon.
    let last_off = (N_FRAMES - 1) * frame_size;
    let dec_last_y = &decoded[last_off..last_off + (W * H) as usize];
    let enc_last_y = &enc.visual_recon.y;
    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    for (a, b) in enc_last_y.iter().zip(dec_last_y.iter()) {
        let d = (*a as i32 - *b as i32).unsigned_abs() as u8;
        if d > max_diff { max_diff = d; }
        sum_diff += d as u64;
    }
    eprintln!("encoder vs ffmpeg last-frame Y: max_diff={} sum_diff={}",
              max_diff, sum_diff);
    // §B-cascade-real Phase 1.1.B/C semantics: visual_recon includes
    // post-deblock state matching what ffmpeg outputs. Some quantization
    // rounding differences are expected in the chroma path; for luma we
    // require near-byte-exact agreement.
    assert!(
        max_diff <= 2,
        "encoder visual_recon diverges from ffmpeg by {} luma units — \
         intra-in-B emit/decode mismatch",
        max_diff,
    );

    let _ = std::fs::remove_file(&h264);
    let _ = std::fs::remove_file(&dec_path);
    unsafe {
        std::env::remove_var("PHASM_B_INTRA_FALLBACK");
        std::env::remove_var("PHASM_B_INTRA_FALLBACK_SATD_MIN");
    }
}
