// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §B-RDO.debug.1 (2026-05-04) — discriminating probe.
//
// Encodes the iPhone7 1080p × 60f fixture via the H.264 encoder
// directly, with no stego hook installed. Same QP, same High
// profile, same GopPattern::Ibpbp{30,1}, same PHASM_B_RDO=1 +
// PHASM_B_RESIDUAL=1 flags as the v10 stego demo. Output is a
// pure encoder bitstream — no STC, no flips, no orchestrator.
//
// Purpose: discriminate whether the v10 4×4 block speckle on the
// moving subject in B-frames is encoder-intrinsic or stego-flip-
// induced. If THIS output (no stego at all) shows the same
// speckle on B-frames, the bug is in the encoder's B-frame
// residual emission path. If it's clean, the bug is somewhere
// in the stego pipeline I haven't traced.

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::stego::gop_pattern::{
    iter_encode_order, FrameType, GopPattern,
};
use phasm_core::codec::mp4::build::{
    build_mp4_with_pattern, FrameTiming, MuxerProfile,
};

#[test]
#[ignore]
fn encode_iphone7_no_stego() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 60usize;
    let frame_size = (width * height * 3 / 2) as usize;
    assert_eq!(yuv.len(), frame_size * n_frames, "yuv size mismatch");

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        // §B-RDO.debug.3 — force ALL B-MBs to Skip. Skip = inherit
        // spatial-direct MV + ZERO residual. ffmpeg renders pred only.
        // If v17 is clean → bug is in residual emission path.
        // If v17 has same speckle → bug is in spatial-direct prediction
        // (colMb fix doesn't catch all motion cases).
        // §B-RDO.debug.5 — force ALL B-MBs to Direct16x16. If v18 is
        // clean → bug is in residual-emitting modes (L0/L1/Bi/etc.).
        // If v18 has speckle → Direct itself broken (despite Skip
        // being visually equivalent on spec).
        // Default: real RDO (no force-mode). Used for v14 baseline
        // and for re-verification after the §B-Partitioned-Residual
        // fix (task #206) lands.
        std::env::remove_var("PHASM_B_FORCE_MODE");
        std::env::remove_var("PHASM_B_FORCE_MV");
        std::env::remove_var("PHASM_B_FORCE_MV_L1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
    }

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    // No stego hook installed.

    // Iterate in ENCODE order (anchor P before B for IBPBP). Reads
    // pixel data by display_idx; encoder gets frames in the same
    // sequence the orchestrator's Pass 3 uses.
    let mut bitstream = Vec::new();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame),
            FrameType::P => enc.encode_p_frame(frame),
            FrameType::B => enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("encode {:?} (encode_idx={}, display_idx={}): {e}", eo.frame_type, eo.encode_idx, eo.display_idx));
        bitstream.extend_from_slice(&bytes);
    }

    eprintln!("no-stego encoded {} bytes", bitstream.len());

    let h264_out = std::env::temp_dir().join("phasm_iphone7_no_stego.h264");
    std::fs::write(&h264_out, &bitstream).expect("write h264");
    eprintln!("wrote {}", h264_out.display());

    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264,
        &bitstream,
        width,
        height,
        FrameTiming::FPS_30,
        pattern,
        n_frames,
    )
    .expect("phasm mp4 mux");
    let mp4_out = std::env::temp_dir().join("phasm_iphone7_no_stego.mp4");
    std::fs::write(&mp4_out, &mp4).expect("write mp4");
    eprintln!("wrote {} ({} bytes)", mp4_out.display(), mp4.len());
}
