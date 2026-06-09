// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
// https://github.com/cgaffga/phasmcore

//! Visual A/B demo: encode the same YUV input twice via OpenH264
//! backend, once clean and once with a small stego override on the
//! first N CoeffSign positions. Both Annex-B outputs are written for
//! a shell wrapper to mux into mp4.
//!
//! **NOT** cascade-safe — the stego variant won't round-trip the
//! payload. Purpose is solely "does the encoder produce a viewable
//! bitstream when stego overrides fire?" — a smoke test for the
//! OpenH264 backend + hook pipeline before investing in cascade-safety
//! work for real fixtures.
//!
//! Usage:
//!   cargo run --example openh264_visual_demo --release \
//!       --features h264-encoder -- \
//!       <input.yuv> <out_clean.h264> <out_stego.h264> \
//!       <width> <height> <n_frames> <qp> <gop_size> <payload_text>

#[cfg(feature = "h264-encoder")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use phasm_core::codec::h264::openh264::{
        Encoder, PhasmStegoDomain, Position, StegoHandlers, StegoSession,
    };
    use std::fs::File;
    use std::io::{Read, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 10 {
        eprintln!(
            "Usage: {} <input.yuv> <out_clean.h264> <out_stego.h264> \\\n  <width> <height> <n_frames> <qp> <gop_size> <payload_text>",
            args[0]
        );
        std::process::exit(2);
    }

    let input_path = &args[1];
    let out_clean_path = &args[2];
    let out_stego_path = &args[3];
    let width: i32 = args[4].parse()?;
    let height: i32 = args[5].parse()?;
    let n_frames: u32 = args[6].parse()?;
    let qp: i32 = args[7].parse()?;
    let gop_size: i32 = args[8].parse()?;
    let payload_text = &args[9];

    let luma_plane = (width as usize) * (height as usize);
    let chroma_plane = (width as usize / 2) * (height as usize / 2);
    let frame_bytes = luma_plane + 2 * chroma_plane;

    let mut yuv = Vec::new();
    File::open(input_path)?.read_to_end(&mut yuv)?;
    let needed = frame_bytes * n_frames as usize;
    if yuv.len() < needed {
        return Err(format!(
            "{} too short: {} bytes < {} needed",
            input_path,
            yuv.len(),
            needed
        )
        .into());
    }

    // Payload as a bit vector — overrides the first payload_bits.len()
    // CoeffSign positions the encoder sees.
    let mut payload_bits: Vec<i32> = Vec::new();
    for byte in payload_text.bytes() {
        for shift in (0..8).rev() {
            payload_bits.push(((byte >> shift) & 1) as i32);
        }
    }
    eprintln!(
        "Payload: {:?} = {} bits to override on first {} CoeffSign positions",
        payload_text,
        payload_bits.len(),
        payload_bits.len()
    );

    // ----- Clean encode (no callbacks) -----
    encode_to_file(
        input_path,
        out_clean_path,
        width,
        height,
        n_frames,
        qp,
        gop_size,
        None,
    )?;

    // ----- Stego encode (override first N CoeffSign bits) -----
    let counter = Arc::new(AtomicUsize::new(0));
    let payload_clone = payload_bits.clone();
    let counter_clone = counter.clone();
    let fires_total = Arc::new(Mutex::new((0usize, 0usize))); // (overridden, total)
    let fires_clone = fires_total.clone();
    let handlers = StegoHandlers {
        enc_pre_emit: Some(Box::new(move |pos: &Position, original: i32| -> Option<i32> {
            let mut f = fires_clone.lock().unwrap();
            f.1 += 1;
            // Only override CoeffSign bits — touching other domains
            // creates desync between residual reconstruction + bin
            // ordering on more domains than we need for a visual demo.
            if pos.domain != PhasmStegoDomain::CoeffSign as u8 {
                return None;
            }
            let i = counter_clone.fetch_add(1, Ordering::Relaxed);
            if i < payload_clone.len() {
                let target = payload_clone[i];
                if target != original {
                    f.0 += 1;
                }
                Some(target)
            } else {
                None
            }
        })),
        ..Default::default()
    };
    let _session = StegoSession::register(handlers)?;
    encode_to_file(
        input_path,
        out_stego_path,
        width,
        height,
        n_frames,
        qp,
        gop_size,
        Some(()),
    )?;
    let (overridden, total) = *fires_total.lock().unwrap();
    eprintln!(
        "Stego encode complete: {} bits actually flipped (of {} payload bits) over {} total hook fires",
        overridden,
        payload_bits.len(),
        total
    );

    Ok(())
}

#[cfg(feature = "h264-encoder")]
fn encode_to_file(
    input_path: &str,
    output_path: &str,
    width: i32,
    height: i32,
    n_frames: u32,
    qp: i32,
    gop_size: i32,
    _session_marker: Option<()>,
) -> Result<(), Box<dyn std::error::Error>> {
    use phasm_core::codec::h264::openh264::{set_frame_num, Encoder};
    use std::fs::File;
    use std::io::{Read, Write};

    let luma_plane = (width as usize) * (height as usize);
    let chroma_plane = (width as usize / 2) * (height as usize / 2);
    let frame_bytes = luma_plane + 2 * chroma_plane;

    let mut yuv = Vec::new();
    File::open(input_path)?.read_to_end(&mut yuv)?;

    set_frame_num(0);
    let mut enc = Encoder::new(width, height, qp, gop_size)?;
    let mut out_file = File::create(output_path)?;
    let mut nal_buf = vec![0u8; frame_bytes * 2];
    let mut total_bytes = 0usize;

    for i in 0..n_frames {
        let frame_off = (i as usize) * frame_bytes;
        let y = &yuv[frame_off..frame_off + luma_plane];
        let u = &yuv[frame_off + luma_plane..frame_off + luma_plane + chroma_plane];
        let v = &yuv[frame_off + luma_plane + chroma_plane..frame_off + frame_bytes];
        let ts = (i as i64) * 33;
        let (_ft, nbytes) = enc.encode_frame(y, u, v, ts, &mut nal_buf)?;
        out_file.write_all(&nal_buf[..nbytes])?;
        total_bytes += nbytes;
    }

    eprintln!("  -> {} ({} bytes)", output_path, total_bytes);
    Ok(())
}

#[cfg(not(feature = "h264-encoder"))]
fn main() {
    eprintln!("openh264_visual_demo requires --features h264-encoder");
    std::process::exit(1);
}
