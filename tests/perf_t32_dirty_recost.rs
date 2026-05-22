// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! T3.2 perf comparison — per-cascade-iteration wall-clock of the
//! shadow-verify cost recomputation: BASELINE
//! `compute_positions_streaming(stego, qt, None)` vs OPTIMIZED
//! `apply_dct_modifications_to_wavelets` + `compute_positions_with_dirty_recost`.
//!
//! Run:
//!   cargo test --release --test perf_t32_dirty_recost -- --nocapture
//!
//! The `compute_cover_wavelets` upfront cost is timed separately — it
//! is amortized once per encode across N cascade iterations (typically
//! 1 in the happy path, up to ~12 if the cascade triggers).

use std::time::Instant;

use phasm_core::codec::jpeg::JpegImage;
use phasm_core::stego::cost::uniward::{
    apply_dct_modifications_to_wavelets, compute_cover_wavelets,
    compute_positions_from_wavelets, compute_positions_streaming,
    compute_positions_with_dirty_recost,
};

const FIXTURE_PATH: &str = "test-vectors/image/photo_original_3557.jpg";

/// Simulate a realistic STC modification set at the typical Ghost
/// rate (~0.13% per-coefficient). Deterministic seed for repeatability.
fn synthesize_modifications(
    cover: &phasm_core::DctGrid,
) -> (Vec<u32>, phasm_core::DctGrid) {
    let bw = cover.blocks_wide();
    let bt = cover.blocks_tall();
    // Realistic Ghost: m ≈ 800 bits × w ≈ 8 / 2 = ~3200 modifications
    // for a 100-byte payload. Independent of image size at typical
    // message lengths. Documented in design/image/t3.2-dirty-block-
    // recost.md § 1.
    let _ = bt * bw;
    let target_n_mods = 3200usize;

    // PRNG to pick positions deterministically.
    let mut state: u64 = 0xC0FFEE_DEADBEEF;
    let mut next = || -> u32 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 32) as u32
    };

    let mut stego = cover.clone();
    let mut modifications: Vec<u32> = Vec::with_capacity(target_n_mods);
    let mut tries = 0usize;
    while modifications.len() < target_n_mods && tries < target_n_mods * 4 {
        tries += 1;
        let block_idx = (next() as usize) % (bw * bt);
        let coef_idx = ((next() as usize) % 63) + 1; // skip DC
        let br = block_idx / bw;
        let bc = block_idx % bw;
        let fi = coef_idx / 8;
        let fj = coef_idx % 8;
        let cur = stego.get(br, bc, fi, fj);
        if cur == 0 { continue; }
        // nsf5-style flip: move |coeff| toward zero by 1.
        let new = if cur > 0 { cur - 1 } else { cur + 1 };
        stego.set(br, bc, fi, fj, new);
        modifications.push(((block_idx * 64) + coef_idx) as u32);
    }
    (modifications, stego)
}

#[test]
fn t32_dirty_recost_bench() {
    let bytes = match std::fs::read(FIXTURE_PATH) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("fixture {FIXTURE_PATH} unavailable — skipping bench");
            return;
        }
    };
    let img = JpegImage::from_bytes(&bytes).expect("load 12MP fixture");
    let bw = img.dct_grid(0).blocks_wide();
    let bt = img.dct_grid(0).blocks_tall();
    let img_w = bw * 8;
    let img_h = bt * 8;
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).expect("Y qt");
    let cover = img.dct_grid(0).clone();

    eprintln!(
        "fixture: {}x{} pixels ({} MP), {} blocks",
        img_w,
        img_h,
        (img_w * img_h) as f64 / 1_048_576.0,
        bt * bw
    );

    // Synthesize a stego grid with ~0.13% modifications (typical Ghost).
    let (modifications, stego) = synthesize_modifications(&cover);
    eprintln!(
        "modifications: {} ({:.3}% of {}-coef grid)",
        modifications.len(),
        modifications.len() as f64 / ((bt * bw * 64) as f64) * 100.0,
        bt * bw * 64
    );

    // Upfront cover wavelet cost — amortized across cascade iterations.
    let t_w = Instant::now();
    let cover_wavelets = compute_cover_wavelets(&cover, qt);
    let upfront_ms = t_w.elapsed().as_millis();
    eprintln!("[upfront] compute_cover_wavelets: {upfront_ms} ms");

    // Upfront cover verify-positions (no SI) — also amortized.
    let t_p = Instant::now();
    let cover_verify_positions =
        compute_positions_from_wavelets(&cover, qt, &cover_wavelets, None).unwrap();
    let upfront_pos_ms = t_p.elapsed().as_millis();
    eprintln!(
        "[upfront] compute_positions_from_wavelets (cover, no SI): {upfront_pos_ms} ms — {} positions",
        cover_verify_positions.len()
    );

    // Warm both paths once before timing (allocator + cache warmup).
    let _ = compute_positions_streaming(&stego, qt, None).unwrap();
    let warm_stego_w = apply_dct_modifications_to_wavelets(
        &cover_wavelets, &modifications, &cover, &stego, qt,
    );
    let _ = compute_positions_with_dirty_recost(
        &stego, qt, &cover_verify_positions, &modifications, &warm_stego_w, None,
    ).unwrap();
    drop(warm_stego_w);

    // BASELINE: compute_positions_streaming on stego (what the code
    // did pre-T3.2.E per cascade iteration).
    let mut baseline_total = 0u128;
    let runs = 3;
    for _ in 0..runs {
        let t = Instant::now();
        let _ = compute_positions_streaming(&stego, qt, None).unwrap();
        baseline_total += t.elapsed().as_millis();
    }
    let baseline_ms = baseline_total / runs;
    eprintln!("[baseline] compute_positions_streaming(stego) avg: {baseline_ms} ms (over {runs} runs)");

    // OPTIMIZED: per-iter cost = apply_dct + dirty_recost.
    let mut opt_apply_total = 0u128;
    let mut opt_recost_total = 0u128;
    for _ in 0..runs {
        let t1 = Instant::now();
        let stego_w = apply_dct_modifications_to_wavelets(
            &cover_wavelets, &modifications, &cover, &stego, qt,
        );
        opt_apply_total += t1.elapsed().as_millis();
        let t2 = Instant::now();
        let _ = compute_positions_with_dirty_recost(
            &stego, qt, &cover_verify_positions, &modifications, &stego_w, None,
        ).unwrap();
        opt_recost_total += t2.elapsed().as_millis();
        drop(stego_w);
    }
    let opt_apply_ms = opt_apply_total / runs;
    let opt_recost_ms = opt_recost_total / runs;
    let opt_total_ms = opt_apply_ms + opt_recost_ms;
    eprintln!(
        "[optimized] apply_dct_modifications: {opt_apply_ms} ms / dirty_recost: {opt_recost_ms} ms / TOTAL: {opt_total_ms} ms (over {runs} runs)"
    );

    if opt_total_ms > 0 {
        let speedup = baseline_ms as f64 / opt_total_ms as f64;
        eprintln!("[T3.2 per-iter speedup] {speedup:.2}x");
    }

    // Always passes — this is a benchmark, not a correctness gate.
    // Bit-exact correctness is gated by the lib tests in uniward.rs.
}
