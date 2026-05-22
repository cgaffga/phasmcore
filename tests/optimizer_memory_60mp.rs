// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! 60 MP optimizer memory regression test.
//!
//! M9.5 (2026-05-22) rewrote the magic-wand optimizer to fit the
//! iPhone Pro per-process memory budget at 60 MP DNG input. Pre-M9.5
//! peak was ~3.9 GB (SIGKILL on iPhone). Post-M9.5 peak is ~325 MB
//! based on analytical accounting (strip refactor + type narrowing +
//! per-platform SIMD).
//!
//! This test guards against future regressions that re-bloat the
//! optimizer's working memory. It runs `optimize_cover` on a real
//! 60 MP RGB buffer and asserts:
//!
//! 1. The call returns successfully (no panic, no OOM kill — though
//!    OOM kill terminates the test process before the assertion
//!    fires anyway; this gate catches the "still runs but slowly"
//!    case where a regression doubles memory without crashing).
//!
//! 2. Peak resident set size (RSS) delta during the call stays under
//!    1 GB. The M9.5 target is ~325 MB; the 1 GB threshold catches
//!    any major regression (e.g. accidentally restoring the full-
//!    image f64 variance map, which alone is 460 MB) without being
//!    so tight it flakes on allocator overhead.
//!
//! 3. Wall time stays under 30 s — rough sanity check that the
//!    optimizer is using the strip pipeline (full-image f64 would
//!    take much longer at this size on most dev machines).
//!
//! Marked `#[ignore]` because allocating a 60 MP RGB buffer (~180 MB)
//! is expensive even before running the optimizer; opt in with:
//!
//!     cargo test --release --test optimizer_memory_60mp -- --ignored --nocapture

use std::time::Instant;

use phasm_core::stego::{optimize_cover, OptimizerConfig, OptimizerMode};

/// Returns the current process's peak resident set size in bytes.
/// Cross-platform via POSIX `getrusage` (`ru_maxrss`).
///
/// Returns 0 if the query fails. macOS reports `ru_maxrss` in bytes;
/// Linux reports it in KiB. Normalised here.
#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    if ret != 0 {
        return 0;
    }
    let max_rss = ru.ru_maxrss as u64;
    #[cfg(target_os = "macos")]
    {
        max_rss
    }
    #[cfg(not(target_os = "macos"))]
    {
        max_rss * 1024
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
}

fn make_pixels_60mp() -> Vec<u8> {
    // 9520 × 6336 is the iPhone 16 Pro Max DNG dimensions — the
    // actual user-facing case the wand crashes / used to crash on.
    let w = 9520usize;
    let h = 6336usize;
    let mut pixels = vec![0u8; w * h * 3];
    // Fill with a realistic-ish mix of smooth gradient + per-pixel
    // hash noise. Smooth regions exercise stages 1/2/4 (noise + 3×3
    // sharpen + dither) where the optimizer does the most work;
    // textured regions exercise the do-no-harm guard.
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let xf = x as f64 / w as f64;
            let yf = y as f64 / h as f64;
            let hash = ((x * 37 + y * 53 + 11) % 19) as f64 - 9.0;
            pixels[idx] = ((xf * 200.0) + hash + 40.0).clamp(0.0, 255.0) as u8;
            pixels[idx + 1] = ((yf * 180.0) + hash * 0.7 + 60.0).clamp(0.0, 255.0) as u8;
            pixels[idx + 2] = (((xf + yf) * 120.0) + hash * 0.5 + 30.0).clamp(0.0, 255.0) as u8;
        }
    }
    pixels
}

#[test]
#[ignore = "expensive; opt in with --ignored"]
fn optimize_cover_60mp_memory_regression() {
    const PEAK_RSS_LIMIT_BYTES: u64 = 1024 * 1024 * 1024; // 1 GB
    const WALL_TIME_LIMIT_SECS: u64 = 30;

    eprintln!("[setup] allocating 60 MP RGB buffer (180 MB)…");
    let pixels = make_pixels_60mp();
    let w = 9520u32;
    let h = 6336u32;

    let config = OptimizerConfig {
        strength: 0.85,
        seed: [42u8; 32],
        mode: OptimizerMode::Ghost,
    };

    let rss_before = peak_rss_bytes();
    eprintln!("[setup] peak RSS before optimize: {:.1} MB", rss_before as f64 / 1_048_576.0);

    let t0 = Instant::now();
    let out = optimize_cover(&pixels, w, h, &config);
    let wall = t0.elapsed();

    let rss_after = peak_rss_bytes();
    let rss_delta = rss_after.saturating_sub(rss_before);

    eprintln!(
        "[result] wall={:.1}s peak_rss_after={:.1}MB delta={:.1}MB out_bytes={}",
        wall.as_secs_f64(),
        rss_after as f64 / 1_048_576.0,
        rss_delta as f64 / 1_048_576.0,
        out.len()
    );

    assert_eq!(
        out.len(),
        pixels.len(),
        "optimizer must return same-size buffer"
    );

    assert!(
        wall.as_secs() < WALL_TIME_LIMIT_SECS,
        "60 MP optimize took {:.1}s — likely indicates an algorithmic \
         regression (M9.5 baseline ~4s on M-series 8-core release).",
        wall.as_secs_f64()
    );

    // RSS delta gate. Only assert on Unix where peak_rss_bytes() is
    // meaningful; on other platforms log only.
    #[cfg(unix)]
    if rss_before > 0 {
        // Peak RSS can be sticky — if a prior test in the same binary
        // already pushed RSS high, the delta here might be near zero.
        // We gate on absolute peak AFTER the call: total resident
        // memory must stay below limit_bytes + the input buffer size
        // (we built one full-image 180 MB input buffer above + the
        // optimizer's working set).
        let total_budget = PEAK_RSS_LIMIT_BYTES + (60 * 1_048_576); // 1 GB optimizer + ~180 MB input
        assert!(
            rss_after < total_budget,
            "60 MP optimize peak RSS {} MB exceeds budget {} MB — \
             M9.5 strip refactor + type narrowing may have regressed.",
            rss_after / 1_048_576,
            total_budget / 1_048_576
        );
    }
}
