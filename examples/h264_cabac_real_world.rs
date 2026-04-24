// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Pure CABAC / CAVLC H.264 encode of a raw yuv420p sequence.
//!
//! Phase 6C.6d real-world content test. No stego embedding — this is a
//! visual-quality sanity check for the encoder alone.
//!
//! Usage:
//! ```
//! cargo run --features video,h264-encoder --release --example h264_cabac_real_world -- \
//!     <input.yuv> <width> <height> <num_frames> <crf> <cabac|cavlc> <output.h264>
//! ```
//!
//! The input must be planar yuv420p: Y (w×h) then Cb (w/2×h/2) then Cr.
//! Width and height MUST be 16-aligned — pad upstream if needed (e.g.
//! via ffmpeg's `pad` filter).

use std::env;
use std::fs::File;
use std::io::{Read, Write};

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 8 {
        eprintln!(
            "usage: {} <input.yuv> <width> <height> <num_frames> <crf> <cabac|cavlc> <output.h264>",
            args[0]
        );
        std::process::exit(2);
    }
    let input = &args[1];
    let width: u32 = args[2].parse().expect("width");
    let height: u32 = args[3].parse().expect("height");
    let num_frames: u32 = args[4].parse().expect("num_frames");
    let crf: u8 = args[5].parse().expect("crf");
    let entropy = match args[6].as_str() {
        "cabac" => EntropyMode::Cabac,
        "cavlc" => EntropyMode::Cavlc,
        other => {
            eprintln!("entropy must be 'cabac' or 'cavlc', got '{other}'");
            std::process::exit(2);
        }
    };
    let output = &args[7];

    assert!(width % 16 == 0, "width must be 16-aligned");
    assert!(height % 16 == 0, "height must be 16-aligned");

    let frame_size = (width as usize) * (height as usize) * 3 / 2;
    let mut reader = File::open(input).expect("open input yuv");
    let mut writer = File::create(output).expect("create output h264");

    let mut enc = Encoder::new(width, height, Some(crf)).expect("encoder new");
    enc.entropy_mode = entropy;

    println!(
        "Encoding {num_frames} frames {width}×{height} yuv420p → {output} (crf={crf}, entropy={:?})",
        enc.entropy_mode
    );

    let t0 = std::time::Instant::now();
    let mut buf = vec![0u8; frame_size];
    let mut out_total = 0usize;
    for f in 0..num_frames {
        if let Err(e) = reader.read_exact(&mut buf) {
            eprintln!("read frame {f}: {e}");
            break;
        }
        // Frame 0 is IDR. After that, respect the encoder's GOP length
        // (default 30) by checking gop_position — once it rolls over to
        // 0, the next frame is another IDR.
        let nal = if f == 0 || enc.gop_position == 0 {
            enc.encode_i_frame(&buf).expect("encode_i_frame")
        } else {
            enc.encode_p_frame(&buf).expect("encode_p_frame")
        };
        out_total += nal.len();
        writer.write_all(&nal).expect("write nal");
        if f % 10 == 0 || f == num_frames - 1 {
            println!(
                "  frame {f:3} / {num_frames}  cumulative {:.2} MB",
                out_total as f64 / 1_000_000.0
            );
        }
    }
    let dt = t0.elapsed();
    println!(
        "Done. {} bytes ({:.2} MB) in {:.2}s ({:.1} fps)",
        out_total,
        out_total as f64 / 1_000_000.0,
        dt.as_secs_f64(),
        num_frames as f64 / dt.as_secs_f64()
    );
}
