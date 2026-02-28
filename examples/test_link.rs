// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Quick decode test for stego images.
//!
//! Usage: `cargo run -p phasm-core --example test_link -- <stego.jpg>`

use phasm_core::stego::smart_decode;
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
    eprintln!("Trying empty passphrase...");
    match smart_decode(&img, "") {
        Ok((p, _q)) => eprintln!("SUCCESS (empty): text='{}' [{:.1}s]", p.text, start.elapsed().as_secs_f64()),
        Err(e) => eprintln!("FAILED (empty): {:?} [{:.1}s]", e, start.elapsed().as_secs_f64()),
    }
}
