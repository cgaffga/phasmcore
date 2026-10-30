// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Write a Deep Cover cover JPEG (phasm's libjpeg-turbo-compatible writer)
//! from a binary PGM (P5) or PPM (P6). Used by the Deep Cover evaluation and
//! the byte-identity check against `cjpeg -quality Q`.
//!
//! Usage: deep_cover_writer <quality> <out_dir> <in.pgm|in.ppm>...
//! Writes <out_dir>/<stem>.jpg for every input.

use std::path::Path;

fn read_pnm(path: &Path) -> (Vec<u8>, u32, u32) {
    let data = std::fs::read(path).expect("read input");
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while data[i].is_ascii_whitespace() {
            i += 1;
        }
        if data[i] == b'#' {
            while data[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let s = i;
        while !data[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(String::from_utf8_lossy(&data[s..i]).to_string());
    }
    i += 1; // single whitespace after maxval
    let (w, h): (u32, u32) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    assert_eq!(fields[3], "255", "8-bit PNM only");
    let px = &data[i..];
    let rgb = match fields[0].as_str() {
        "P5" => px[..(w * h) as usize].iter().flat_map(|&g| [g, g, g]).collect(),
        "P6" => px[..(w * h * 3) as usize].to_vec(),
        m => panic!("unsupported PNM {m}"),
    };
    (rgb, w, h)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: deep_cover_writer <quality> <out_dir> <in.pgm|ppm>...");
        std::process::exit(2);
    }
    let q: u8 = args[1].parse().expect("quality");
    let out = Path::new(&args[2]);
    std::fs::create_dir_all(out).unwrap();
    for a in &args[3..] {
        let p = Path::new(a);
        let (rgb, w, h) = read_pnm(p);
        let bytes = phasm_core::jpeg::cover_writer::encode_rgb(&rgb, w, h, q, |_, _, _, _| {}).unwrap();
        let stem = p.file_stem().unwrap().to_string_lossy();
        std::fs::write(out.join(format!("{stem}.jpg")), bytes).unwrap();
    }
}
