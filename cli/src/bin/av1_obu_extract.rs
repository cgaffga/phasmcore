// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// One-off measurement helper for the AV1 W2 speed × QP sweep.
//
// phasm's MP4 muxer (`split_av1_into_samples`) strips
// sequence_header_obus from per-sample bytes (the canonical
// AV1-ISOBMFF § 2.2.1 layout — SH lives in av1C only). ffmpeg's
// MP4 demuxer doesn't splice av1C into samples before handing
// them to libdav1d, so ffmpeg fails to decode phasm AV1 stego
// output (iOS / AVPlayer + the in-tree streaming decoder both
// handle it fine).
//
// This binary writes a raw "section 5" OBU stream (with
// sequence_header_obu re-prepended before every sync sample)
// that the standalone `dav1d` CLI can decode straight to YUV.
// We then run ffmpeg's psnr + ssim filters between the decoded
// YUV and the original source YUV to fill out the sweep's
// quality columns.
//
// Build: `cargo build -p phasm-cli --features av1-encoder --bin av1_obu_extract`
// Usage: `av1_obu_extract <stego.mp4>` → writes spliced OBU bytes to stdout.

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

    let sh: Vec<u8> = track
        .av1c_data
        .as_ref()
        .map(|c| c.config_obus.clone())
        .unwrap_or_default();
    if sh.is_empty() {
        eprintln!("av1_obu_extract: av1C has empty config_obus; cannot splice SH");
        std::process::exit(1);
    }

    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    // Per-sample: prepend the av1C sequence_header_obu before each
    // sync (keyframe) sample so dav1d's section 5 demuxer can split
    // on real GOP boundaries — same pattern as
    // `decode_av1_streaming` (decode.rs:248-268). Section 5 also wants
    // a temporal_delimiter_obu at the start of each TU (the muxer
    // stripped these — AV1-ISOBMFF doesn't carry TDs in per-sample
    // OBUs because the sample boundary itself marks the TU); we
    // re-inject one (0x12 0x00 = TD header + empty body) before the
    // SH for sync samples and before the frame OBU for non-sync.
    const TD_OBU: [u8; 2] = [0x12, 0x00];
    for sample in &track.samples {
        if let Err(e) = out.write_all(&TD_OBU) {
            eprintln!("av1_obu_extract: write TD: {e}");
            std::process::exit(1);
        }
        if sample.is_sync {
            if let Err(e) = out.write_all(&sh) {
                eprintln!("av1_obu_extract: write SH: {e}");
                std::process::exit(1);
            }
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
