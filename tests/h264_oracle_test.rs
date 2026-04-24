// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Smoke tests for the Phase 6.0c oracle harness.
//!
//! Verifies that `common::h264_oracle::decode_via_ffmpeg` can process
//! our committed corpus clips. Once Phase 6A ships real phasm-encoded
//! streams, further tests add Layer 2 reconstruction checks.

mod common;

use common::h264_oracle::{decode_via_ffmpeg, system_has_ffmpeg};
use std::fs;
use std::path::Path;

const CORPUS_DIR: &str = "test-vectors/video/h264/encoder-corpus";

#[test]
fn ffmpeg_decodes_corpus_libx264() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    for name in &[
        "testsrc_x264_crf23_fast.mp4",
        "testsrc_x264_crf18_slow.mp4",
        "mandelbrot_x264_crf23_fast.mp4",
        "smptebars_x264_crf23_fast.mp4",
    ] {
        decode_one(name);
    }
}

#[test]
fn ffmpeg_decodes_corpus_videotoolbox() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    for name in &["testsrc_vt_q60.mp4", "mandelbrot_vt_q60.mp4"] {
        decode_one(name);
    }
}

fn decode_one(name: &str) {
    let path = Path::new(CORPUS_DIR).join(name);
    let Ok(bytes) = fs::read(&path) else {
        eprintln!(
            "skipping {name} — corpus clip missing. Regenerate with \
             core/test-vectors/video/h264/encoder-corpus/generate.sh"
        );
        return;
    };
    let result = decode_via_ffmpeg(&bytes).unwrap_or_else(|e| {
        panic!("oracle decode failed for {name}: {:?}", e);
    });

    assert!(
        !result.decoded_yuv.is_empty(),
        "{name}: decoded YUV empty"
    );
    assert!(result.width > 0, "{name}: width 0");
    assert!(result.height > 0, "{name}: height 0");
    let frame_bytes = (result.width as usize) * (result.height as usize) * 3 / 2;
    assert_eq!(
        result.decoded_yuv.len() % frame_bytes,
        0,
        "{name}: decoded YUV not an integer number of {}x{} yuv420p frames",
        result.width,
        result.height,
    );
}
