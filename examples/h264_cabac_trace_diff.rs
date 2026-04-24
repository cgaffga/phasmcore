// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Encoder-emit trace vs spec-decoder trace diff harness.
//!
//! Encodes a YUV file through our CABAC encoder with bin-by-bin tracing
//! enabled (via `PHASM_CABAC_ENC_TRACE=1`), then decodes the output via
//! the spec-direct decoder (also with tracing). Compares the two trace
//! files and prints the first divergent line — that's where the
//! encoder and spec decoder disagree, which is the bug.
//!
//! Usage:
//!   h264_cabac_trace_diff <yuv> <w> <h> <qp>
//!
//! Expects a single-frame raw yuv420p input. Encodes I-frame with
//! CABAC. Dumps three files:
//!   /tmp/cabac_enc_trace.txt  — encoder bin trace
//!   /tmp/cabac_dec_trace.txt  — decoder bin trace
//!   /tmp/cabac_diff.txt       — first N diverging lines side-by-side

use std::env;
use std::fs;

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: {} <yuv> <w> <h> <quality>", args[0]);
        std::process::exit(2);
    }
    let yuv_path = &args[1];
    let w: u32 = args[2].parse().expect("w");
    let h: u32 = args[3].parse().expect("h");
    let q: u8 = args[4].parse().expect("quality");

    let pixels = fs::read(yuv_path).expect("read yuv");
    let expected = (w * h * 3 / 2) as usize;
    assert_eq!(pixels.len(), expected, "unexpected yuv size");

    // Encode with tracing.
    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_cabac_trace();
    let bytes = enc.encode_i_frame(&pixels).expect("encode");
    let enc_trace = enc.take_cabac_trace();
    fs::write("/tmp/cabac_enc_trace.txt", enc_trace.join("\n") + "\n").unwrap();
    fs::write("/tmp/cabac_diff_input.h264", &bytes).unwrap();
    eprintln!(
        "encoded {} bytes, {} enc trace lines",
        bytes.len(), enc_trace.len()
    );

    // Now decode via the spec decoder binary for its trace. We invoke
    // it as a subprocess so we don't duplicate the decoder code here.
    let dec_out = std::process::Command::new("target/release/examples/h264_cabac_decoder")
        .arg("/tmp/cabac_diff_input.h264")
        .output()
        .expect("run h264_cabac_decoder");
    let dec_trace: Vec<String> = String::from_utf8_lossy(&dec_out.stdout)
        .lines()
        .filter(|l| l.starts_with("DEC "))
        .map(String::from)
        .collect();
    fs::write("/tmp/cabac_dec_trace.txt", dec_trace.join("\n") + "\n").unwrap();
    eprintln!("decoder emitted {} trace lines", dec_trace.len());

    // Diff. The ENC and DEC formats differ slightly; compare just the
    // bin value + context-index to find the first divergence.
    let enc_keys: Vec<(u32, u8, String)> = enc_trace
        .iter()
        .map(|l| parse_enc_key(l))
        .collect();
    let dec_keys: Vec<(u32, u8, String)> = dec_trace
        .iter()
        .map(|l| parse_dec_key(l))
        .collect();

    let mut diff_out = String::new();
    let max = enc_keys.len().max(dec_keys.len());
    let mut divergences = 0;
    for i in 0..max {
        let e = enc_keys.get(i);
        let d = dec_keys.get(i);
        let match_ok = match (e, d) {
            (Some((ec, eb, _)), Some((dc, db, _))) => ec == dc && eb == db,
            _ => false,
        };
        if !match_ok {
            diff_out.push_str(&format!("=== divergence at line {i} ===\n"));
            if let Some((ec, eb, el)) = e {
                diff_out.push_str(&format!("  ENC[{i}]: ctx={ec} bin={eb} | {el}\n"));
            } else {
                diff_out.push_str(&format!("  ENC[{i}]: (out of lines)\n"));
            }
            if let Some((dc, db, dl)) = d {
                diff_out.push_str(&format!("  DEC[{i}]: ctx={dc} bin={db} | {dl}\n"));
            } else {
                diff_out.push_str(&format!("  DEC[{i}]: (out of lines)\n"));
            }
            divergences += 1;
            if divergences >= 10 {
                diff_out.push_str("... more divergences truncated\n");
                break;
            }
        }
    }

    if divergences == 0 {
        println!("MATCH: all {} bins agree between encoder and spec decoder", max);
    } else {
        println!("{} divergences found (first 10 in /tmp/cabac_diff.txt)", divergences);
    }
    fs::write("/tmp/cabac_diff.txt", diff_out).unwrap();
}

/// Parse an encoder trace line for (ctx_idx, bin, raw line).
fn parse_enc_key(l: &str) -> (u32, u8, String) {
    // Format: "ENC <label>: ctx=N pre_range=... bin=B post_..."
    // or "ENC <label>: BYPASS pre_low=... bin=B ..."
    // or "ENC <label>: TERMINATE pre_range=... bin=B ..."
    let ctx = if l.contains("BYPASS") {
        u32::MAX - 1
    } else if l.contains("TERMINATE") {
        u32::MAX - 2
    } else {
        l.split_whitespace()
            .find_map(|t| t.strip_prefix("ctx="))
            .and_then(|v| v.parse().ok())
            .unwrap_or(u32::MAX)
    };
    let bin = l.split_whitespace()
        .find_map(|t| t.strip_prefix("bin="))
        .and_then(|v| v.parse().ok())
        .unwrap_or(255);
    (ctx, bin, l.to_string())
}

fn parse_dec_key(l: &str) -> (u32, u8, String) {
    let ctx = if l.contains("BYPASS") {
        u32::MAX - 1
    } else if l.contains("TERMINATE") {
        u32::MAX - 2
    } else {
        l.split_whitespace()
            .find_map(|t| t.strip_prefix("ctx="))
            .and_then(|v| v.parse().ok())
            .unwrap_or(u32::MAX)
    };
    let bin = l.split_whitespace()
        .find_map(|t| t.strip_prefix("bin="))
        .and_then(|v| v.parse().ok())
        .unwrap_or(255);
    (ctx, bin, l.to_string())
}
