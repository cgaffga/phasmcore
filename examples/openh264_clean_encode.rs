// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
// https://github.com/cgaffga/phasmcore

//! Clean OpenH264-backend encode (no stego) — produces an Annex-B
//! H.264 bitstream for baseline fingerprint measurement.
//!
//! Phase C.4-pre.2 (task #390) consumes this to generate reference
//! mb_type / direction / partition-shape histograms for the OpenH264
//! cover-story comparison. The reverse-path stego pipeline is not
//! exercised here — the encoder is constructed with no callback
//! registered, so all 15 stego hooks are inactive.
//!
//! Usage:
//!   cargo run --example openh264_clean_encode --release \
//!       --features openh264-backend -- \
//!       <input.yuv> <output.h264> <width> <height> <n_frames> <qp> <gop_size>
//!
//! Input: I420 YUV, plane-packed, no header. Plane sizes derived from
//! width / height. The fork is configured with SM_SINGLE_SLICE so
//! one frame produces a single slice (matches Phase A.5 validation
//! harness).

#[cfg(feature = "openh264-backend")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use phasm_core::codec::h264::openh264::Encoder;
    use std::fs::File;
    use std::io::{Read, Write};
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 8 {
        eprintln!(
            "Usage: {} <input.yuv> <output.h264> <width> <height> <n_frames> <qp> <gop_size>",
            args[0]
        );
        std::process::exit(2);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let width: i32 = args[3].parse()?;
    let height: i32 = args[4].parse()?;
    let n_frames: u32 = args[5].parse()?;
    let qp: i32 = args[6].parse()?;
    let gop_size: i32 = args[7].parse()?;

    let luma_plane = (width as usize) * (height as usize);
    let chroma_plane = (width as usize / 2) * (height as usize / 2);
    let frame_bytes = luma_plane + 2 * chroma_plane;

    let mut yuv = Vec::new();
    File::open(input_path)?.read_to_end(&mut yuv)?;
    let needed = frame_bytes * n_frames as usize;
    if yuv.len() < needed {
        return Err(format!(
            "{} too short: {} bytes, need {} for {} frames @ {}x{} I420",
            input_path,
            yuv.len(),
            needed,
            n_frames,
            width,
            height
        )
        .into());
    }

    let mut enc = Encoder::new(width, height, qp, gop_size)?;
    let mut out = File::create(output_path)?;
    let mut nal_buf = vec![0u8; frame_bytes * 2];
    let mut frame_types = Vec::with_capacity(n_frames as usize);
    let mut total_bytes = 0usize;

    for i in 0..n_frames {
        let frame_off = (i as usize) * frame_bytes;
        let y = &yuv[frame_off..frame_off + luma_plane];
        let u = &yuv[frame_off + luma_plane..frame_off + luma_plane + chroma_plane];
        let v = &yuv[frame_off + luma_plane + chroma_plane..frame_off + frame_bytes];
        let ts = (i as i64) * 33;
        let (ft, nbytes) = enc.encode_frame(y, u, v, ts, &mut nal_buf)?;
        out.write_all(&nal_buf[..nbytes])?;
        frame_types.push(format!("{ft:?}"));
        total_bytes += nbytes;
    }

    eprintln!(
        "Encoded {} frames {}x{} qp={} gop={} -> {} ({} bytes)",
        n_frames,
        width,
        height,
        qp,
        gop_size,
        Path::new(output_path).display(),
        total_bytes
    );
    eprintln!("Frame types: {}", frame_types.join(","));
    Ok(())
}

#[cfg(not(feature = "openh264-backend"))]
fn main() {
    eprintln!("openh264_clean_encode requires --features openh264-backend");
    std::process::exit(1);
}
