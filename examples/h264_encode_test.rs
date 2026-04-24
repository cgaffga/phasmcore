// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Simple H.264 video steganography example.
//!
//! Reads an H.264 Baseline CAVLC MP4, embeds a message, writes the stego output.
//!
//! Usage:
//! ```
//! cargo run --features video --example h264_encode_test --release -- input.mp4 output.mp4 "message"
//! ```

use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let input = args.get(1).cloned().unwrap_or_else(|| "/tmp/h264_baseline_1080p.mp4".into());
    let output = args.get(2).cloned().unwrap_or_else(|| "/tmp/h264_stego_1080p.mp4".into());
    let message = args.get(3).cloned().unwrap_or_else(|| "Phasm Test".into());
    let passphrase = args.get(4).cloned().unwrap_or_default();

    let cover = fs::read(&input).expect("read cover");
    println!("Cover: {input} ({} bytes, {:.1} MB)", cover.len(), cover.len() as f64 / 1_000_000.0);

    match phasm_core::stego::video::h264_ghost_capacity(&cover) {
        Ok(cap) => println!("Capacity: ~{cap} bytes"),
        Err(e) => { println!("Capacity error: {e}"); return; }
    }

    println!("Encoding message: \"{message}\" (passphrase: \"{}\")",
        if passphrase.is_empty() { "<empty>" } else { &passphrase });

    let stego = match phasm_core::stego::video::h264_ghost_encode(&cover, &message, &passphrase) {
        Ok(s) => s,
        Err(e) => { println!("Encode error: {e}"); return; }
    };

    let diffs: usize = cover.iter().zip(stego.iter()).filter(|(a, b)| a != b).count();
    let bits: u32 = cover.iter().zip(stego.iter()).map(|(a, b)| (a ^ b).count_ones()).sum();
    println!("Stego: {} bytes (same size: {}) — {diffs} bytes changed, {bits} bits flipped",
        stego.len(), stego.len() == cover.len());

    fs::write(&output, &stego).expect("write stego");
    println!("Written to {output}");

    match phasm_core::stego::video::h264_ghost_decode(&stego, &passphrase) {
        Ok(decoded) => {
            println!("Decoded: \"{}\"", decoded.text);
            println!("Roundtrip: {}", if decoded.text == message { "PASS" } else { "FAIL" });
        }
        Err(e) => println!("Decode error: {e}"),
    }
}
