// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! T3.5 perf comparison — legacy `compute_syndromes` (poly_eval per
//! syndrome) vs the restructured fast path (loop-swapped + per-α^i
//! lookup tables).
//!
//! Run:
//!   cargo test --release --test perf_t35_syndromes -- --nocapture

use phasm_core::stego::armor::ecc::{
    perf_fast_compute_syndromes, perf_legacy_compute_syndromes,
};
use std::time::Instant;

fn bench(label: &str, parity_len: usize, n_received: usize, n_iters: usize) {
    // Deterministic varied received block.
    let mut state: u32 = 0xDEAD_BEEF;
    let received: Vec<u8> = (0..n_received)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        })
        .collect();

    // Warm.
    let _ = perf_legacy_compute_syndromes(&received, parity_len);
    let _ = perf_fast_compute_syndromes(&received, parity_len);

    let mut acc: u64 = 0;

    let t = Instant::now();
    for _ in 0..n_iters {
        let s = perf_legacy_compute_syndromes(&received, parity_len);
        acc = acc.wrapping_add(s.iter().map(|&b| b as u64).sum::<u64>());
    }
    let legacy_us = t.elapsed().as_micros();
    std::hint::black_box(acc);

    let mut acc: u64 = 0;
    let t = Instant::now();
    for _ in 0..n_iters {
        let s = perf_fast_compute_syndromes(&received, parity_len);
        acc = acc.wrapping_add(s.iter().map(|&b| b as u64).sum::<u64>());
    }
    let fast_us = t.elapsed().as_micros();
    std::hint::black_box(acc);

    eprintln!(
        "[{label}] parity_len={parity_len} received={n_received} iters={n_iters}",
    );
    eprintln!("  legacy: {legacy_us} µs   ({:.2} µs/call)", legacy_us as f64 / n_iters as f64);
    eprintln!("  fast:   {fast_us} µs   ({:.2} µs/call)", fast_us as f64 / n_iters as f64);
    if fast_us > 0 {
        eprintln!(
            "  speedup: {:.2}x",
            legacy_us as f64 / fast_us as f64
        );
    }
    eprintln!();
}

#[test]
fn t35_syndromes_bench() {
    // Shadow brute-force tiers (most user-visible — wrong-pass smart_decode
    // calls rs_decode_with_parity ~10K times via try_single_fdl).
    bench("shadow small  parity=4  ",   4, 255, 10_000);
    bench("shadow med    parity=32 ",  32, 255, 10_000);
    bench("shadow large  parity=128", 128, 255, 10_000);
    // Armor adaptive tiers.
    bench("armor base    parity=64 ",  64, 255, 10_000);
    bench("armor max     parity=240", 240, 255, 10_000);
}
