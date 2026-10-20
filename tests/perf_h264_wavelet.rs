// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// J-UNIWARD 3-subband wavelet micro-bench (the H.264 Tier 3 cost path).
//
// **min-of-N on purpose**: this machine is routinely running other work, and
// competing load can only ever ADD time to a run. The minimum is therefore the
// measurement that survives a noisy host, where a mean or an end-to-end wall
// comparison does not — which is exactly how the parallel-GOP projection
// (2026-06) came to look like a win it wasn't.
//
// Reference numbers, Apple silicon, release, `h264-encoder` (2026-10-18):
//   row pass scalar (pre-#848-style restructure): luma 9.23 ms / chroma 3.74 ms
//   row pass k-outer SIMD saxpy:                  luma 8.13 ms / chroma 2.67 ms
// i.e. 1.24× on the per-frame wavelet (1 luma + 2 chroma), bit-identical —
// `row_pass_simd_is_bit_identical_to_scalar_reference` is the correctness gate;
// this is only the speed observation.
#![cfg(feature = "h264-encoder")]

use std::time::Instant;

fn synth(width: usize, height: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(width * height);
    let mut s: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let px = match x * 4 / width.max(1) {
                0 => 16u8,
                1 => 235,
                2 => ((x * 7 + y * 3) % 256) as u8,
                _ => (s >> 24) as u8,
            };
            v.push(px);
        }
    }
    v
}

#[test]
fn bench_three_subbands_min_of_n() {
    const REPS: usize = 12;
    for &(w, h, label) in &[
        (1920usize, 1072usize, "luma 1920x1072"),
        (960, 536, "chroma 960x536"),
    ] {
        let plane = synth(w, h);
        let mut best = f64::MAX;
        let mut checksum = 0.0f64;
        for _ in 0..REPS {
            let t = Instant::now();
            let b = phasm_core::stego::cost::h264_uniward::compute_three_subbands(&plane, w, h);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            checksum += b.lh[b.lh.len() / 2] as f64;
            if ms < best {
                best = ms;
            }
        }
        println!("BENCH {label:<18} min-of-{REPS} {best:>8.2} ms   (checksum {checksum:e})");
    }
}
