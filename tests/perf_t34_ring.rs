// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! T3.4 perf comparison — legacy ring-scan compute_sector_magnitude
//! vs LUT-backed compute_sector_magnitude_lut.
//!
//! Run:
//!   cargo test --release --test perf_t34_ring -- --nocapture
//!
//! The legacy path scans the (2*r_max+1)^2 bin square and calls
//! det_hypot + det_atan2 PER BIN, PER SECTOR. The LUT path hoists
//! the bin->sector classification into a single ~ring-area scan and
//! reuses it across all 256 sector accesses.

use phasm_core::stego::armor::dft_payload::{
    perf_legacy_compute_sector_magnitude, perf_lut_compute_sector_magnitude,
    ring_radii, SectorLut,
};
use phasm_core::stego::armor::fft2d::{Complex32, Spectrum2D};
use std::time::Instant;

const NUM_SECTORS: usize = 256;

fn make_spectrum(w: usize, h: usize) -> Spectrum2D {
    // Deterministic content via a simple LCG so the bench output is
    // stable across runs.
    let mut state: u32 = 0xDEAD_BEEF;
    let mut data = vec![Complex32::new(0.0, 0.0); w * h];
    for val in data.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let mag = 100.0 + ((state >> 24) as u8) as f32;
        *val = Complex32::new(mag, 0.0);
    }
    Spectrum2D { data, width: w, height: h }
}

fn bench_size(label: &str, w: usize, h: usize) {
    let spectrum = make_spectrum(w, h);
    let (r_inner, r_outer) = ring_radii(w, h);
    eprintln!(
        "[{label}] spectrum: {w}x{h} ({:.2} MP), r_inner={:.1}, r_outer={:.1}",
        (w * h) as f64 / 1_048_576.0,
        r_inner,
        r_outer
    );

    // Warm both paths.
    let warm_lut = SectorLut::build(w, h);
    for s in 0..NUM_SECTORS {
        let _ = perf_legacy_compute_sector_magnitude(&spectrum, s, r_inner, r_outer);
        let _ = perf_lut_compute_sector_magnitude(&spectrum, s, &warm_lut);
    }

    // LEGACY: 256 sectors x ring scan.
    let t = Instant::now();
    let mut acc = 0.0f64;
    for s in 0..NUM_SECTORS {
        acc += perf_legacy_compute_sector_magnitude(&spectrum, s, r_inner, r_outer);
    }
    let legacy_ms = t.elapsed().as_millis();
    std::hint::black_box(acc);

    // LUT: build once, then 256 sector accesses.
    let t = Instant::now();
    let lut = SectorLut::build(w, h);
    let lut_build_ms = t.elapsed().as_millis();
    let total_bins = lut.total_bins();
    let t = Instant::now();
    let mut acc = 0.0f64;
    for s in 0..NUM_SECTORS {
        acc += perf_lut_compute_sector_magnitude(&spectrum, s, &lut);
    }
    let lut_query_ms = t.elapsed().as_millis();
    std::hint::black_box(acc);
    let lut_total_ms = lut_build_ms + lut_query_ms;

    eprintln!(
        "[{label}] LUT total bins: {total_bins} (avg {:.1}/sector)",
        total_bins as f64 / NUM_SECTORS as f64
    );
    eprintln!("[{label}] legacy: 256 sectors via ring scan: {legacy_ms} ms");
    eprintln!(
        "[{label}] LUT:    build {lut_build_ms} ms + 256 queries {lut_query_ms} ms = {lut_total_ms} ms"
    );
    if lut_total_ms > 0 {
        let speedup = legacy_ms as f64 / lut_total_ms as f64;
        eprintln!("[{label}] speedup vs legacy: {speedup:.1}x");
    }
    eprintln!();
}

#[test]
fn t34_ring_bench() {
    // Small / medium / large representative of phasm.app + mobile +
    // 4K-camera capture surfaces.
    bench_size("small  240x180", 240, 180);
    bench_size("medium 1024x768", 1024, 768);
    bench_size("large  2048x1536", 2048, 1536);
    // 4K is too big to run on CI in a reasonable timeout — skipped
    // unless explicitly enabled. Uncomment for a local 4K bench:
    // bench_size("4K     4032x3024", 4032, 3024);
}
