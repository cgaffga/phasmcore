// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! T3.7 perf comparison — pre-T3.7 8-bit fast table + linear-scan
//! slow path vs new 10-bit fast table + per-length canonical lookup.
//!
//! Run:
//!   cargo test --release --test perf_t37_huffman -- --nocapture

use phasm_core::codec::jpeg::bitio::BitReader;
use phasm_core::codec::jpeg::huffman::{
    perf_huffman_decode_n, perf_legacy_huffman_decode_n, HuffmanDecodeTable,
    HuffmanEncodeTable, LegacyHuffmanDecodeTable,
};
use std::time::Instant;

fn lum_ac_table() -> ([u8; 16], Vec<u8>) {
    let bits = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
    let vals = vec![
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13,
        0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42,
        0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a,
        0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35,
        0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a,
        0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67,
        0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84,
        0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3,
        0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
        0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4,
        0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    ];
    (bits, vals)
}

#[test]
fn t37_huffman_bench() {
    let (bits, vals) = lum_ac_table();
    let enc = HuffmanEncodeTable::build(&bits, &vals);
    let dec_legacy = LegacyHuffmanDecodeTable::build(&bits, &vals).unwrap();
    let dec_fast = HuffmanDecodeTable::build(&bits, &vals).unwrap();

    // Build a synthetic bitstream by encoding a deterministic stream
    // of symbols sampled from the table. Mix of frequencies to hit
    // both fast and slow paths.
    let mut state: u32 = 0xCAFE_BABE;
    let mut next_sym_idx = || -> usize {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 16) as usize) % vals.len()
    };

    let n_symbols: usize = 1_000_000;
    let mut encoded_bits: Vec<u8> = Vec::new();
    let mut byte_buf: u32 = 0;
    let mut bits_in_buf: u8 = 0;

    for _ in 0..n_symbols {
        let sym = vals[next_sym_idx()];
        let (code, len) = enc.encode(sym).unwrap();
        byte_buf = (byte_buf << len) | (code as u32);
        bits_in_buf += len;
        while bits_in_buf >= 8 {
            let shift = bits_in_buf - 8;
            let byte = ((byte_buf >> shift) & 0xFF) as u8;
            encoded_bits.push(byte);
            if byte == 0xFF {
                encoded_bits.push(0x00); // JPEG byte stuffing
            }
            byte_buf &= (1u32 << shift) - 1;
            bits_in_buf = shift;
        }
    }
    // Flush remaining bits with 1-padding (JPEG convention).
    if bits_in_buf > 0 {
        let byte = ((byte_buf << (8 - bits_in_buf)) | ((1u32 << (8 - bits_in_buf)) - 1)) as u8;
        encoded_bits.push(byte);
    }
    // Pad with enough 0xFF + 0x00 to avoid end-of-stream errors mid-decode.
    for _ in 0..32 {
        encoded_bits.push(0xFF);
        encoded_bits.push(0x00);
    }

    eprintln!("[T3.7 bench] {n_symbols} symbols, encoded to {} bytes", encoded_bits.len());

    // Warm both paths (and verify they agree on the first 1000).
    let mut r_legacy = BitReader::new(&encoded_bits, 0);
    let mut r_fast = BitReader::new(&encoded_bits, 0);
    for i in 0..1000 {
        let a = dec_legacy.decode(&mut r_legacy).unwrap();
        let b = dec_fast.decode(&mut r_fast).unwrap();
        assert_eq!(a, b, "decoder divergence at symbol {i}: legacy={a}, fast={b}");
    }

    // Legacy timing.
    let mut r = BitReader::new(&encoded_bits, 0);
    let t = Instant::now();
    let csum_legacy = perf_legacy_huffman_decode_n(&dec_legacy, &mut r, n_symbols).unwrap();
    let legacy_ms = t.elapsed().as_millis();
    std::hint::black_box(csum_legacy);

    // Fast timing.
    let mut r = BitReader::new(&encoded_bits, 0);
    let t = Instant::now();
    let csum_fast = perf_huffman_decode_n(&dec_fast, &mut r, n_symbols).unwrap();
    let fast_ms = t.elapsed().as_millis();
    std::hint::black_box(csum_fast);

    assert_eq!(csum_legacy, csum_fast, "decoded checksum mismatch");

    eprintln!("  legacy (8-bit fast + linear scan):  {legacy_ms} ms   ({:.3} ns/symbol)",
              (legacy_ms as f64 * 1_000_000.0) / n_symbols as f64);
    eprintln!("  fast   (10-bit fast + per-length): {fast_ms} ms   ({:.3} ns/symbol)",
              (fast_ms as f64 * 1_000_000.0) / n_symbols as f64);
    if fast_ms > 0 {
        eprintln!("  speedup: {:.2}x", legacy_ms as f64 / fast_ms as f64);
    }
}
