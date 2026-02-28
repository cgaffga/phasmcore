// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Example: encode and decode a hidden message in a JPEG image.
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: test_encode <input.jpg> <message> <passphrase>");
        eprintln!("       test_encode --decode <stego.jpg> <passphrase>");
        std::process::exit(1);
    }

    if args[1] == "--decode" {
        let stego = fs::read(&args[2]).expect("Could not read stego image");
        match phasm_core::ghost_decode(&stego, &args[3]) {
            Ok(payload) => {
                println!("Decoded message: {}", payload.text);
                for f in &payload.files {
                    println!("  File: {} ({} bytes)", f.filename, f.content.len());
                }
            }
            Err(e) => eprintln!("Decode failed: {:?}", e),
        }
    } else {
        let cover = fs::read(&args[1]).expect("Could not read cover image");
        let message = &args[2];
        let passphrase = &args[3];

        let stego = phasm_core::ghost_encode(&cover, message, passphrase)
            .expect("Encode failed");

        let out_path = args[1].replace(".jpg", "_stego.jpg").replace(".JPG", "_stego.jpg");
        fs::write(&out_path, &stego).expect("Could not write output");
        println!("Stego image written to: {}", out_path);
        println!("Original: {} bytes, Stego: {} bytes", cover.len(), stego.len());
    }
}
