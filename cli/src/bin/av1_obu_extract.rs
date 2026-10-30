// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// One-off measurement helper for the AV1 W2 speed × QP sweep.
//
// This binary writes a raw "section 5" OBU stream from a phasm AV1
// MP4 — the samples with a temporal delimiter before each (the muxer
// drops them, as AV1-ISOBMFF samples carry none); every sync sample
// already starts with its sequence_header_obu — that the standalone
// `dav1d` CLI can decode straight to YUV.
// We then run ffmpeg's psnr + ssim filters between the decoded
// YUV and the original source YUV to fill out the sweep's
// quality columns.
//
// Build: `cargo build -p phasm-cli --features av1-encoder --bin av1_obu_extract`
// Usage: `av1_obu_extract <stego.mp4>` → writes the OBU stream to stdout.

use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: av1_obu_extract <stego.mp4>");
        std::process::exit(2);
    }
    let mp4_path = &args[1];
    let bytes = match std::fs::read(mp4_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("av1_obu_extract: read {mp4_path}: {e}");
            std::process::exit(1);
        }
    };

    let mp4 = match phasm_core::codec::mp4::demux::demux(&bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("av1_obu_extract: demux failed: {e:?}");
            std::process::exit(1);
        }
    };

    let idx = match mp4.video_track_idx {
        Some(i) => i,
        None => {
            eprintln!("av1_obu_extract: no video track");
            std::process::exit(1);
        }
    };
    let track = &mp4.tracks[idx];
    if !track.is_av1() {
        eprintln!("av1_obu_extract: video track is not av01 (codec={:?})", track.codec);
        std::process::exit(1);
    }

    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    // Section 5 wants a temporal_delimiter_obu at the start of each TU
    // (0x12 0x00 = TD header + empty body).
    const TD_OBU: [u8; 2] = [0x12, 0x00];
    for sample in &track.samples {
        if let Err(e) = out.write_all(&TD_OBU) {
            eprintln!("av1_obu_extract: write TD: {e}");
            std::process::exit(1);
        }
        if let Err(e) = out.write_all(&sample.data) {
            eprintln!("av1_obu_extract: write sample: {e}");
            std::process::exit(1);
        }
    }
    if let Err(e) = out.flush() {
        eprintln!("av1_obu_extract: flush: {e}");
        std::process::exit(1);
    }
}
