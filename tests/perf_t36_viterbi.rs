// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! T3.6 perf bench — legacy in-line 128-state Viterbi step vs the
//! restructured + intrinsic-dispatched fast path.
//!
//! Run:
//!   cargo test --release --test perf_t36_viterbi -- --nocapture

use phasm_core::stego::stc::embed::{perf_fast_viterbi_step, perf_legacy_viterbi_step};
use std::time::Instant;

const NUM_STATES: usize = 128;
const N_STEPS: usize = 100_000;

#[test]
fn t36_viterbi_step_bench() {
    // Deterministic fixture.
    let mut state: u32 = 0xC0FFEE_42;
    let mut step = || -> u8 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 24) as u8
    };
    let mut prev_cost = vec![0.0f64; NUM_STATES];
    for v in prev_cost.iter_mut() {
        *v = (step() as f64) / 32.0;
    }
    // Pick a varying col per step to exercise the gather pattern;
    // pre-generate the schedule so legacy + fast see identical inputs.
    let cols: Vec<usize> = (0..N_STEPS).map(|_| (step() as usize) & 0x7F).collect();
    let cost_pairs: Vec<(f64, f64)> = (0..N_STEPS)
        .map(|_| ((step() as f64) / 16.0, (step() as f64) / 16.0))
        .collect();

    let mut curr_legacy = vec![0.0f64; NUM_STATES];
    let mut curr_fast = vec![0.0f64; NUM_STATES];
    let mut prev_perm = vec![0.0f64; NUM_STATES];
    let mut bp_byte = vec![0u8; NUM_STATES];

    // Warm both paths.
    for &col in &cols[..100] {
        let _ = perf_legacy_viterbi_step(&prev_cost, &mut curr_legacy, col, 1.0, 1.0);
        let _ = perf_fast_viterbi_step(
            &prev_cost, &mut curr_fast, &mut prev_perm, &mut bp_byte, col, 1.0, 1.0,
        );
    }

    // Legacy.
    let mut acc: u128 = 0;
    let t = Instant::now();
    for i in 0..N_STEPS {
        let (s0, s1) = cost_pairs[i];
        acc = acc.wrapping_add(perf_legacy_viterbi_step(
            &prev_cost, &mut curr_legacy, cols[i], s0, s1,
        ));
    }
    let legacy_us = t.elapsed().as_micros();
    std::hint::black_box(acc);

    // Fast.
    let mut acc: u128 = 0;
    let t = Instant::now();
    for i in 0..N_STEPS {
        let (s0, s1) = cost_pairs[i];
        acc = acc.wrapping_add(perf_fast_viterbi_step(
            &prev_cost, &mut curr_fast, &mut prev_perm, &mut bp_byte, cols[i], s0, s1,
        ));
    }
    let fast_us = t.elapsed().as_micros();
    std::hint::black_box(acc);

    eprintln!("Viterbi 128-state step bench ({N_STEPS} steps):");
    eprintln!("  legacy: {legacy_us} µs   ({:.3} µs/step)", legacy_us as f64 / N_STEPS as f64);
    eprintln!("  fast:   {fast_us} µs   ({:.3} µs/step)", fast_us as f64 / N_STEPS as f64);
    if fast_us > 0 {
        eprintln!("  speedup: {:.2}x", legacy_us as f64 / fast_us as f64);
    }
}
