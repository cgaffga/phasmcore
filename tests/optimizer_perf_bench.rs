// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Wall-time benchmark for `optimize_cover` — used for M9.5
//! before/after comparison.
//!
//! Not part of the regular gate; run explicitly:
//!     cargo test --release --test optimizer_perf_bench -- --nocapture
//!
//! Reports min/median/max of N runs at a representative image size.

use std::time::Instant;

use phasm_core::stego::{optimize_cover, OptimizerConfig, OptimizerMode};

fn make_pixels(w: usize, h: usize) -> Vec<u8> {
    // Realistic photo content: mix of smooth gradient + per-pixel
    // noise + sinusoidal pattern so the optimizer's adaptive stages
    // all see varied input.
    let mut pixels = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let xf = x as f64 / w as f64;
            let yf = y as f64 / h as f64;
            let hash = ((x * 37 + y * 53 + 11) % 19) as f64 - 9.0;
            let r = ((xf * 200.0) + hash + 40.0).clamp(0.0, 255.0) as u8;
            let g = ((yf * 180.0) + hash * 0.7 + 60.0).clamp(0.0, 255.0) as u8;
            let b = (((xf + yf) * 120.0) + hash * 0.5 + 30.0).clamp(0.0, 255.0) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

fn bench(name: &str, w: usize, h: usize, runs: usize) {
    let pixels = make_pixels(w, h);
    let config = OptimizerConfig {
        strength: 0.85,
        seed: [42u8; 32],
        mode: OptimizerMode::Ghost,
    };

    // Warmup
    let _ = optimize_cover(&pixels, w as u32, h as u32, &config);

    let mut samples: Vec<u128> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _out = optimize_cover(&pixels, w as u32, h as u32, &config);
        samples.push(t0.elapsed().as_micros());
    }
    samples.sort();

    let min_ms = samples[0] as f64 / 1000.0;
    let med_ms = samples[samples.len() / 2] as f64 / 1000.0;
    let max_ms = samples[samples.len() - 1] as f64 / 1000.0;
    eprintln!(
        "[{name}] {w}×{h} ({:.1} MP) — min={min_ms:.1}ms med={med_ms:.1}ms max={max_ms:.1}ms over {runs} runs",
        (w * h) as f64 / 1_000_000.0
    );
}

#[test]
#[ignore = "perf bench; opt in with --include-ignored or by name"]
fn optimizer_perf_bench() {
    eprintln!();
    eprintln!("=== M9.5 optimizer wall-time bench ===");
    bench("1MP", 1024, 1024, 7);
    bench("4MP", 2048, 2048, 5);
    bench("12MP", 4032, 3024, 3);
    // 60 MP — iPhone Pro DNG case the wand was failing on pre-M9.5.
    bench("60MP", 9520, 6336, 2);
}
