// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Timing benchmark for Armor and Ghost decode paths.
//!
//! Usage: `cargo run -p phasm-core --example test_timing -- <stego.jpg>`

use phasm_core::stego::{armor_decode, ghost_decode};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("/tmp/phasm_shared.jpg");
    let img = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Error reading {path}: {e}");
        std::process::exit(1);
    });
    eprintln!("Image: {} bytes", img.len());

    let start = Instant::now();
    eprintln!("Trying Armor decode...");
    match armor_decode(&img, "") {
        Ok((p, q)) => eprintln!("ARMOR SUCCESS: text='{}' integrity={:.1}% [{:.1}s]", p.text, q.integrity_percent, start.elapsed().as_secs_f64()),
        Err(e) => eprintln!("ARMOR FAILED: {:?} [{:.1}s]", e, start.elapsed().as_secs_f64()),
    }

    let start2 = Instant::now();
    eprintln!("Trying Ghost decode...");
    match ghost_decode(&img, "") {
        Ok(p) => eprintln!("GHOST SUCCESS: text='{}' [{:.1}s]", p.text, start2.elapsed().as_secs_f64()),
        Err(e) => eprintln!("GHOST FAILED: {:?} [{:.1}s]", e, start2.elapsed().as_secs_f64()),
    }
}
