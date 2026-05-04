// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// One-shot iPhone video encode for visual quality inspection. Reads a
// pre-converted YUV from /tmp, encodes via the §30D-C orchestrator
// with current B-RDO + HF-prop multiplier + Bi hacks active. Writes
// BOTH the Annex-B .h264 AND a properly-muxed .mp4 with B-frame
// ctts + edts/elst (via phasm's HandBrake/x264 muxer) so the demo
// plays without B-frame display-order stutter.
//
// Run with:
//   cargo test --release --features cabac-stego --test h264_iphone_demo -- --ignored --nocapture

#![cfg(feature = "cabac-stego")]

use phasm_core::{
    h264_stego_encode_yuv_string_4domain_multigop_with_pattern, GopPattern,
};
use phasm_core::codec::mp4::build::{
    build_mp4_with_pattern, FrameTiming, MuxerProfile,
};
use std::time::Instant;

#[test]
#[ignore]
fn encode_iphone7_demo() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    eprintln!("loaded {} bytes from {}", yuv.len(), yuv_path);

    unsafe {
        // §B-RDO.proper (2026-05-04): demo opts in to PHASM_B_RDO=1
        // + PHASM_B_RESIDUAL=1 explicitly. The lib-wide default is
        // still OFF — flipping global default regressed two shadow
        // tests (n_shadows_roundtrip_n_equals_2 +
        // shadow_roundtrip_handles_longer_primary_via_cascade)
        // because the larger residual cover-space changes
        // shadow-cascade safe-position selection. v1.1 plumbs a
        // typed BRdoConfig through the encoder API.
        //
        // Verified clean on this fixture: 0 ffmpeg concealment
        // events, B-frame Y-PSNR 35-37 dB, P-frame Y-PSNR 39-41 dB,
        // chroma U/V 45-47 dB across all frames.
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
    }

    // §B-direct-streaks (#196) FIXED: spec § 8.4.1.2.2 step 6
    // colZeroFlag check is now wired in `derive_b_direct_spatial`,
    // P-frame DPB promote captures the colocated motion grid, and
    // the B-slice encoder threads it through to all spatial-direct
    // sites. Static-background MBs adjacent to moving subjects are
    // forced to (0, 0) MV → no more ghost streaks on smooth walls.
    // Demo back to IBPBP (production stealth-match pattern).
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    let n_frames = 60usize;

    let t0 = Instant::now();
    let stego = h264_stego_encode_yuv_string_4domain_multigop_with_pattern(
        &yuv,
        1920,
        1072,
        n_frames,
        pattern,
        "phasm-demo-secret",
        "phasm-iphone7-demo",
    )
    .expect("phasm encode");
    let dur = t0.elapsed();
    eprintln!("encoded {} bytes in {:?}", stego.len(), dur);

    let h264_out = std::env::temp_dir().join("phasm_iphone7_demo.h264");
    std::fs::write(&h264_out, &stego).expect("write h264");
    eprintln!("wrote {}", h264_out.display());

    // §Stutter-fix: mux to MP4 via phasm's HandBrake/x264 muxer.
    // Computes per-sample ctts (display-idx − encode-idx) +
    // edts/elst leading empty edit so the player presents frames
    // in display order. ffmpeg's `-c copy` mux from Annex-B does
    // NOT compute ctts from POC, which made earlier demos play in
    // encode order (= visible frame stutter on every B-frame).
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264,
        &stego,
        1920,
        1072,
        FrameTiming::FPS_30,
        pattern,
        n_frames,
    )
    .expect("phasm mp4 mux");

    let mp4_out = std::env::temp_dir().join("phasm_iphone7_demo.mp4");
    std::fs::write(&mp4_out, &mp4).expect("write mp4");
    eprintln!("wrote {} ({} bytes)", mp4_out.display(), mp4.len());
}
