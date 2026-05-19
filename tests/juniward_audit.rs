// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! E1 audit support — dumps Phasm's J-UNIWARD cost map for an input JPEG
//! so an external comparison script (eval/scripts/audit_juniward_off_by_one.py)
//! can numerically compare against conseal's ORIGINAL and FIX_OFF_BY_ONE
//! implementations.
//!
//! Driven via env vars so this test is a no-op in normal `cargo test` runs:
//!   PHASM_AUDIT_INPUT   absolute path to a baseline JPEG
//!   PHASM_AUDIT_OUTPUT  absolute path where cost-map JSON will be written
//!
//! JSON shape:
//!   {
//!     "blocks_wide": <usize>,
//!     "blocks_tall": <usize>,
//!     "costs": [...]   // bt*bw*64 entries, row-major (br, bc, fi, fj)
//!   }
//! Non-finite costs (WET) emit as JSON null.

use phasm_core::stego::cost::uniward::compute_uniward;
use phasm_core::JpegImage;
use std::env;
use std::fs;
use std::io::Write;

#[test]
fn dump_cost_map() {
    let Ok(input_path) = env::var("PHASM_AUDIT_INPUT") else {
        return;
    };
    let Ok(output_path) = env::var("PHASM_AUDIT_OUTPUT") else {
        return;
    };

    let bytes = fs::read(&input_path)
        .unwrap_or_else(|e| panic!("Failed to read {input_path}: {e}"));
    let img = JpegImage::from_bytes(&bytes)
        .unwrap_or_else(|e| panic!("Failed to parse JPEG: {e:?}"));

    let grid = img.dct_grid(0);
    let qt = img.quant_table(0).expect("quant table 0 missing");
    let map = compute_uniward(grid, qt);

    let bw = map.blocks_wide();
    let bt = map.blocks_tall();

    let mut out = fs::File::create(&output_path)
        .unwrap_or_else(|e| panic!("Failed to create {output_path}: {e}"));

    writeln!(out, "{{").unwrap();
    writeln!(out, "  \"blocks_wide\": {bw},").unwrap();
    writeln!(out, "  \"blocks_tall\": {bt},").unwrap();
    write!(out, "  \"costs\": [").unwrap();
    let mut first = true;
    for br in 0..bt {
        for bc in 0..bw {
            for fi in 0..8 {
                for fj in 0..8 {
                    if !first {
                        write!(out, ",").unwrap();
                    }
                    first = false;
                    let c = map.get(br, bc, fi, fj);
                    if c.is_finite() && c > 0.0 {
                        write!(out, "{c}").unwrap();
                    } else {
                        write!(out, "null").unwrap();
                    }
                }
            }
        }
    }
    writeln!(out, "]").unwrap();
    writeln!(out, "}}").unwrap();
}
