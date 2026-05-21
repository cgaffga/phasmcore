// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! T1.1 — Argon2id structural-key cache demo.
//!
//! Demonstrates the perf win from the thread-local key cache added
//! 2026-05-21 (perf audit T1.1). Cold-vs-warm Argon2id timing.
//!
//! Reference numbers (developer macOS, 2026-05-21):
//!   - cold derive_structural_key: ~20 ms (Apple Silicon, release)
//!   - warm cache hit:             <1 µs
//!   - speedup:                    ~25 000×
//!
//! On phones (~200 ms cold Argon2), the second smart_decode call in
//! a session lands all 4 structural keys in <5 µs total instead of
//! ~800 ms — effectively zero amortized cost.
//!
//! Asserts cold > warm by a wide margin. Doesn't pin specific
//! timings — those are machine-dependent — but ratio is enough to
//! catch a regression that disables the cache.
//!
//! Run with: `cargo test --release -p phasm-core --test
//! t11_cache_perf_demo -- --nocapture`

use phasm_core::stego::crypto::{clear_key_cache, derive_structural_key};
use std::time::Instant;

#[test]
fn t11_cache_perf_demo() {
    clear_key_cache();
    let pass = "perf-check-passphrase";

    let cold = Instant::now();
    let _ = derive_structural_key(pass).unwrap();
    let cold_ns = cold.elapsed().as_nanos();
    let cold_ms = cold_ns as f64 / 1_000_000.0;

    let warm = Instant::now();
    let _ = derive_structural_key(pass).unwrap();
    let warm_ns = warm.elapsed().as_nanos();
    let warm_us = warm_ns as f64 / 1_000.0;

    println!("\n=== T1.1 Argon2 cache demo ===");
    println!("cold derive_structural_key: {cold_ms:.1} ms");
    println!("warm (cache hit):           {warm_us:.1} µs");
    println!("speedup:                    ~{:.0}×", cold_ns as f64 / warm_ns as f64);

    // Regression gate: warm must be at least 100× faster than cold.
    // Typical observed ratio is 25k-200k×. A 100× floor catches any
    // regression that disables the cache while staying well clear of
    // CI noise.
    assert!(
        cold_ns >= 100 * warm_ns,
        "warm cache hit ({warm_ns} ns) is not >100× faster than cold ({cold_ns} ns) — cache likely disabled or broken"
    );
}
