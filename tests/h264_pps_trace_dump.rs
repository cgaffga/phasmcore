// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §Stealth.L4.6.5 — One-shot dump of phasm encoded Annex-B for
// `ffmpeg -bsf trace_headers` PPS+slice-header diff against the
// same-fixture x264-medium reference.
//
// `#[ignore]` because this runs the full §30D-C orchestrator on a
// 1080p × 10-frame IBPBP fixture (~30 sec). Run with:
//   cargo test --features cabac-stego --test h264_pps_trace_dump -- --ignored

#![cfg(feature = "cabac-stego")]

use phasm_core::{
    h264_stego_encode_yuv_string_4domain_multigop_with_pattern, GopPattern,
};

#[test]
#[ignore]
fn dump_phasm_pps_for_trace() {
    let yuv_path = "/tmp/img4138_1080p_f10.yuv";
    let yuv = std::fs::read(yuv_path).expect(
        "missing /tmp/img4138_1080p_f10.yuv — run \
         core/test-vectors/video/h264/real-world/source/regen.sh",
    );

    let stego = h264_stego_encode_yuv_string_4domain_multigop_with_pattern(
        &yuv,
        1920,
        1072,
        10,
        GopPattern::Ibpbp { gop: 5, b_count: 1 },
        "trace-dump",
        "phasm-pps-trace-dump",
    )
    .expect("phasm encode");

    let out = std::env::temp_dir().join("phasm_pps_trace_fixture.h264");
    std::fs::write(&out, &stego).expect("write phasm h264");
    eprintln!(
        "wrote {} bytes to {}",
        stego.len(),
        out.display()
    );
}
