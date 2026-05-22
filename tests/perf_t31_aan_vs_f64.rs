// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! T3.1 perf comparison — f64 path vs integer LL&M + SIMD path.
//!
//! Run with:
//!   cargo test --release --test perf_t31_aan_vs_f64 -- --nocapture
//!   cargo test --release --features aan-dct --test perf_t31_aan_vs_f64 -- --nocapture
//!
//! Then diff the two outputs. The active-path label in the output
//! identifies which config ran.
//!
//! Workload: armor-encode a 4032×3024 (12 MP) iPhone JPEG 5 times.
//! Armor encode exercises:
//!   - pre_clamp_y_channel (IDCT + clamp + DCT per Y block)
//!   - Fortress pre-settle (IDCT + DCT per Y/Cb/Cr block at QF=75)
//!   - FFT-based DFT template
//!   - STDM embedding + RS ECC
//! The IDCT/DCT-bound pre-clamp + pre-settle phases dominate; LL&M
//! SIMD wins land there.

use std::time::Instant;

const FIXTURE_PATH: &str = "../test-vectors/photo_original_3557.jpg";

/// Micro-bench: just the IDCT + DCT kernels in a tight loop. Isolates
/// the SIMD win from rayon-saturated multi-threaded encode flows.
#[test]
fn t31_kernel_microbench() {
    use phasm_core::codec::jpeg::pixels::{dct_block, idct_block};
    let qt: [u16; 64] = [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57,
        69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64,
        81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ];
    let label = if cfg!(feature = "aan-dct") {
        "aan-dct (integer LL&M + SIMD)"
    } else {
        "default (f64 path)"
    };

    // ~50K blocks ≈ a 12 MP image worth of luma blocks.
    let n_blocks = 50_000usize;
    // Deterministic input (avoid optimizer eliding the loop).
    let mut s: u32 = 0xDEAD_BEEF;
    let mut pixels = [0.0f64; 64];
    for p in &mut pixels {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *p = ((s >> 24) as u8) as f64;
    }

    // Warm up.
    for _ in 0..1000 {
        let q = dct_block(&pixels, &qt);
        let _ = idct_block(&q, &qt);
    }

    // f64 path (always reachable, even under aan-dct).
    let t = Instant::now();
    let mut sink = 0i64;
    for _ in 0..n_blocks {
        let q = dct_block(&pixels, &qt);
        let r = idct_block(&q, &qt);
        sink = sink.wrapping_add(q[0] as i64).wrapping_add(r[0] as i64);
    }
    let f64_ms = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!("=== T3.1 kernel microbench — {label} ===");
    eprintln!("{n_blocks} blocks × (dct_block + idct_block) f64 reference:");
    eprintln!("  total: {f64_ms:.1} ms  per-block: {:.2} µs  (sink={sink})", f64_ms * 1000.0 / n_blocks as f64);

    // aan path (only callable when feature is on).
    #[cfg(feature = "aan-dct")]
    {
        use phasm_core::codec::jpeg::pixels_aan::{aan_dct_block, aan_idct_block};
        let t = Instant::now();
        let mut sink = 0i64;
        for _ in 0..n_blocks {
            let q = aan_dct_block(&pixels, &qt);
            let r = aan_idct_block(&q, &qt);
            sink = sink
                .wrapping_add(q[0] as i64)
                .wrapping_add(r[0].round() as i64);
        }
        let aan_ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!("{n_blocks} blocks × (aan_dct_block + aan_idct_block) integer LL&M + SIMD:");
        eprintln!("  total: {aan_ms:.1} ms  per-block: {:.2} µs  (sink={sink})", aan_ms * 1000.0 / n_blocks as f64);
        eprintln!("Speedup: f64 / aan = {:.2}× (>1.0× means LL&M is faster)", f64_ms / aan_ms);
    }
}

#[test]
fn t31_armor_encode_perf() {
    bench_encode("armor_encode", |bytes, msg, pass| {
        phasm_core::armor_encode(bytes, msg, pass).map(|_| ())
    });
}

#[test]
fn t31_ghost_encode_perf() {
    bench_encode("ghost_encode", |bytes, msg, pass| {
        phasm_core::ghost_encode(bytes, msg, pass).map(|_| ())
    });
}

fn bench_encode(
    name: &str,
    op: impl Fn(&[u8], &str, &str) -> Result<(), phasm_core::stego::error::StegoError>,
) {
    let bytes = std::fs::read(FIXTURE_PATH).unwrap_or_else(|_| {
        eprintln!("skipping: fixture not found");
        Vec::new()
    });
    if bytes.is_empty() {
        return;
    }
    let label = if cfg!(feature = "aan-dct") {
        "aan-dct (integer LL&M + SIMD)"
    } else {
        "default (f64 path)"
    };

    let _ = op(&bytes, "warmup", "warmup");

    let iters = 5usize;
    let mut times_ms = Vec::with_capacity(iters);
    for i in 0..iters {
        let t = Instant::now();
        let _ = op(&bytes, "perf test", &format!("pass-{i}"));
        times_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times_ms.iter().sum::<f64>() / iters as f64;
    eprintln!("=== T3.1 {name} perf — {label} ===");
    eprintln!(
        "iters {iters}  min {:.1} median {:.1} mean {:.1} max {:.1} ms",
        times_ms[0],
        times_ms[iters / 2],
        mean,
        times_ms[iters - 1],
    );
}
