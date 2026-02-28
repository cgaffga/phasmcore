// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Diagnostic tool: analyze why a stego image decodes slowly.
//!
//! Usage: cargo run -p phasm-core --example diagnose_decode -- <stego.jpg>
//!
//! Analyzes the candidate ordering, header extraction, and predicts
//! which step would succeed during the decode sweep.

use phasm_core::jpeg::JpegImage;
use phasm_core::stego::armor::embedding;
use phasm_core::stego::armor::ecc;
use phasm_core::stego::armor::repetition;
use phasm_core::stego::armor::selection::compute_stability_map;
use phasm_core::stego::armor::spreading::{generate_spreading_vectors, SPREAD_LEN};
use phasm_core::stego::crypto;
use phasm_core::stego::frame;
use phasm_core::stego::permute;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: diagnose_decode <stego.jpg>");
        std::process::exit(1);
    }

    let stego = fs::read(&args[1]).expect("Could not read stego image");

    // Parse JPEG
    let img = JpegImage::from_bytes(&stego).expect("Invalid JPEG");
    let fi = img.frame_info();
    println!("=== Image Properties ===");
    println!("Dimensions: {}x{}", fi.width, fi.height);
    println!("Components: {}", fi.components.len());
    println!("File size: {} bytes ({:.1} MB)", stego.len(), stego.len() as f64 / 1_048_576.0);

    if img.num_components() == 0 {
        eprintln!("No luminance channel");
        return;
    }

    // Get QT
    let qt_id = fi.components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).expect("No QT");
    println!("\nQuantization table (natural order):");
    for row in 0..8 {
        for col in 0..8 {
            print!("{:4}", qt.values[row * 8 + col]);
        }
        println!();
    }

    // Compute mean QT (same as encode path)
    let actual_mean_qt = embedding::compute_mean_qt(&qt.values);
    let header_byte_from_actual = embedding::encode_mean_qt(actual_mean_qt);
    println!("\n=== Mean QT Analysis ===");
    println!("Actual mean QT (from image's QT table): {:.6}", actual_mean_qt);
    println!("Header byte that encode would produce: {} (0x{:02X})", header_byte_from_actual, header_byte_from_actual);
    println!("Decoded back: {:.6}", embedding::decode_mean_qt(header_byte_from_actual));

    // Compute stability map and positions (using empty passphrase as default)
    // We need to try with the actual passphrase, but we don't have it.
    // Instead, use empty passphrase to get position count (close enough for analysis).
    let cost_map = compute_stability_map(img.dct_grid(0), qt);
    let structural_key = crypto::derive_armor_structural_key("");
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let spread_seed: [u8; 32] = structural_key[32..].try_into().unwrap();
    let positions = permute::select_and_permute(&cost_map, &perm_seed);
    let num_units = positions.len() / SPREAD_LEN;
    let payload_units = if num_units > 56 { num_units - 56 } else { 0 };
    println!("\n=== Capacity ===");
    println!("Stable positions: {}", positions.len());
    println!("Embedding units: {}", num_units);
    println!("Payload units (after 56 header units): {}", payload_units);

    // Extract header byte to see what the decode path actually reads
    let vectors = generate_spreading_vectors(&spread_seed, num_units);
    let grid = img.dct_grid(0);

    // Extract header at bootstrap delta
    let header_byte = extract_header_byte_soft(grid, &positions, &vectors, embedding::BOOTSTRAP_DELTA, 0);
    let header_mean_qt = embedding::decode_mean_qt(header_byte);

    println!("\n=== Header Extraction ===");
    println!("Extracted header byte: {} (0x{:02X})", header_byte, header_byte);
    println!("Header mean QT: {:.6}", header_mean_qt);
    println!("Expected header byte: {} (0x{:02X})", header_byte_from_actual, header_byte_from_actual);
    if header_byte != header_byte_from_actual {
        println!("*** HEADER MISMATCH! Extracted != expected.");
        println!("    This means the header was corrupted or passphrase differs.");
    } else {
        println!("Header matches expected value.");
    }

    // Build candidate list (exactly as try_armor_decode does)
    let current_mean_qt = actual_mean_qt;

    let mut raw_candidates = Vec::with_capacity(24);
    raw_candidates.push(header_mean_qt);
    raw_candidates.push(current_mean_qt);
    for step in 1..=10 {
        let factor = step as f64 * 0.03;
        raw_candidates.push(header_mean_qt * (1.0 - factor));
        raw_candidates.push(header_mean_qt * (1.0 + factor));
    }

    // Deduplicate (within 0.1 tolerance)
    let mut candidates: Vec<f64> = Vec::with_capacity(raw_candidates.len());
    for &c in &raw_candidates {
        if c > 0.1 && !candidates.iter().any(|&existing| (existing - c).abs() < 0.1) {
            candidates.push(c);
        }
    }

    let nc = candidates.len();
    let total_steps = 2 * (1 + nc + nc) + 1 + 1;

    println!("\n=== Progress Step Map ===");
    println!("Candidate count (nc): {}", nc);
    println!("Total progress steps: {}", total_steps);
    println!();
    println!("Step layout:");
    println!("  Step 1: Fortress check");
    println!("  Steps 2..{}: Phase 1 candidates (fast, r=1)", 1 + nc);
    println!("  Steps {}..{}: Phase 2 candidates (brute-force r/parity)", 2 + nc, 1 + 2 * nc);
    println!("  --- First try_armor_decode ends at step {} ---", 1 + 2 * nc);
    println!("  Step {}: Phase 3 geometric recovery (FFT/template/resample)", 2 + 2 * nc);
    println!("  Steps {}..{}: Inner fortress + Phase 1 (inside Phase 3)", 3 + 2 * nc, 2 + 3 * nc);
    println!("  Steps {}..{}: Inner Phase 2 (inside Phase 3)", 3 + 3 * nc, 2 + 4 * nc);
    println!("  Step {}: Ghost decode", 3 + 4 * nc);

    // Show candidate order with deltas
    println!("\n=== Candidate List ===");
    println!("{:>4} {:>10} {:>12} {:>8} {:>8}", "Idx", "mean_qt", "delta(r=1)", "P1_step", "P2_step");
    for (i, &c) in candidates.iter().enumerate() {
        let delta = embedding::compute_delta_from_mean_qt(c, 1);
        let p1_step = 2 + i;
        let p2_step = 2 + nc + i;
        let marker = if (c - actual_mean_qt).abs() < 0.01 { " <-- ACTUAL" } else { "" };
        println!("{:>4} {:>10.4} {:>12.4} {:>8} {:>8}{}", i, c, delta, p1_step, p2_step, marker);
    }

    // Check which candidate matches the actual encoding delta
    let encode_delta = embedding::compute_delta_from_mean_qt(actual_mean_qt, 1);
    println!("\nEncode Phase 1 delta (r=1): {:.4}", encode_delta);

    // Show Phase 2 candidate r values
    println!("\n=== Phase 2 (r, parity) Candidates per mean_qt ===");
    let parity_tiers: [usize; 4] = [64, 128, 192, 240];
    for (ci, &mqt) in candidates.iter().enumerate() {
        let mut cands_for_mqt = Vec::new();
        for &parity in &parity_tiers {
            // Inline compute_candidate_rs logic
            let min_frame = frame::FRAME_OVERHEAD;
            let max_frame = frame::MAX_FRAME_BYTES;
            let mut rs_set = std::collections::BTreeSet::new();
            for frame_len in min_frame..=max_frame {
                let rs_encoded_len = ecc::rs_encoded_len_with_parity(frame_len, parity);
                let rs_bits = rs_encoded_len * 8;
                if rs_bits > payload_units {
                    break;
                }
                let r = repetition::compute_r(rs_bits, payload_units);
                if r >= 3 {
                    rs_set.insert(r);
                }
            }
            for r in &rs_set {
                let delta = embedding::compute_delta_from_mean_qt(mqt, *r);
                cands_for_mqt.push((parity, *r, delta));
            }
        }
        if ci == 0 || ci == 1 {
            println!("  Candidate {} (mean_qt={:.4}): {} (parity,r) combos", ci, mqt, cands_for_mqt.len());
            for (p, r, d) in &cands_for_mqt {
                println!("    parity={:>3}, r={:>3}, delta={:.4}", p, r, d);
            }
        }
    }

    // Analyze what happens at step 89/92
    println!("\n=== Step 89/92 Analysis ===");
    if total_steps == 92 {
        println!("Confirmed: total_steps = 92, nc = {}", nc);
        let first_run_end = 1 + 2 * nc;
        let phase3_step = first_run_end + 1;
        let inner_start = phase3_step + 1;
        let inner_p1_end = inner_start + nc - 1;
        let inner_p2_start = inner_p1_end + 1;
        let inner_p2_end = inner_p2_start + nc - 1;
        let ghost_step = inner_p2_end + 1;

        println!("First run: steps 1..{}", first_run_end);
        println!("Phase 3 step: {}", phase3_step);
        println!("Inner Phase 1: steps {}..{}", inner_start, inner_p1_end);
        println!("Inner Phase 2: steps {}..{}", inner_p2_start, inner_p2_end);
        println!("Ghost: step {}", ghost_step);

        if 89 >= inner_p2_start as usize && 89 <= inner_p2_end as usize {
            let inner_idx = 89 - inner_p2_start;
            println!("\nStep 89 falls in Inner Phase 2, candidate index {} (0-indexed)", inner_idx);
            if (inner_idx as usize) < candidates.len() {
                let mqt = candidates[inner_idx as usize];
                println!("mean_qt = {:.4}", mqt);
            }
        } else if 89 >= inner_start as usize && 89 <= inner_p1_end as usize {
            let inner_idx = 89 - inner_start;
            println!("\nStep 89 falls in Inner Phase 1, candidate index {}", inner_idx);
        } else if 89 >= (2 + nc) && 89 <= first_run_end {
            let idx = 89 - (2 + nc);
            println!("\nStep 89 falls in first-run Phase 2, candidate index {}", idx);
            if idx < candidates.len() {
                let mqt = candidates[idx];
                println!("mean_qt = {:.4}", mqt);
            }
        }
    } else {
        println!("total_steps = {} (not 92). nc = {}", total_steps, nc);
        // Still analyze where step 89 falls
        let first_run_end = 1 + 2 * nc;
        println!("First run ends at step {}", first_run_end);
        if 89 <= first_run_end {
            if 89 <= 1 + nc {
                println!("Step 89 is in Phase 1, candidate index {}", 89 - 2);
            } else {
                let p2_idx = 89 - (2 + nc);
                println!("Step 89 is in Phase 2, candidate index {}", p2_idx);
                if p2_idx < candidates.len() {
                    let mqt = candidates[p2_idx];
                    println!("mean_qt = {:.4}", mqt);
                }
            }
        } else {
            println!("Step 89 is past the first run (geometric recovery or ghost)");
        }
    }

    // Key insight: if the image was NOT recompressed and NOT rotated,
    // Phase 1 candidate[0] (header_mean_qt) should match perfectly.
    // If it doesn't, either:
    // 1. The passphrase is wrong (wrong structural key -> wrong positions/vectors)
    // 2. The header was corrupted
    // 3. The QT changed (recompression)
    println!("\n=== Key Insights ===");
    let delta_match = embedding::compute_delta_from_mean_qt(header_mean_qt, 1);
    let delta_actual = embedding::compute_delta_from_mean_qt(actual_mean_qt, 1);
    let delta_diff_pct = ((delta_match - delta_actual) / delta_actual * 100.0).abs();
    println!("Delta from header: {:.4}", delta_match);
    println!("Delta from actual QT: {:.4}", delta_actual);
    println!("Difference: {:.2}%", delta_diff_pct);

    if delta_diff_pct < 1.0 {
        println!("Deltas match well. Phase 1 candidate[0] should work for pristine images.");
        println!("If decode still fails at Phase 1, the image may be Phase 2 encoded (r>=3).");
        println!("Phase 2 encode uses a DIFFERENT delta (higher multiplier for larger r).");
        println!("Phase 1 decode with r=1 delta won't match a Phase 2 encode's delta.");
    }

    // Explain the Phase 1 vs Phase 2 mismatch
    println!("\n=== Phase 1 vs Phase 2 Delta Mismatch ===");
    println!("Phase 1 decode uses delta = compute_delta_from_mean_qt(mean_qt, r=1) = mean_qt * 3.0");
    println!("Phase 2 encode might use r>=3, giving delta = mean_qt * 6.0 (r=3..4) or * 7.0 (r=5..6) or * 8.0 (r>=7)");
    println!();
    println!("If the message was encoded with Phase 2 (r>=3), Phase 1 decode WILL FAIL");
    println!("because the delta is wrong. Only Phase 2's brute-force (r, parity) search");
    println!("will find the correct delta.");
    println!();
    println!("For this image:");
    println!("  Actual mean_qt = {:.4}", actual_mean_qt);
    println!("  Phase 1 delta (r=1): {:.4}  (mult=3.0)", actual_mean_qt * 3.0);
    println!("  Phase 2 delta (r=3): {:.4}  (mult=6.0)", actual_mean_qt * 6.0);
    println!("  Phase 2 delta (r=5): {:.4}  (mult=7.0)", actual_mean_qt * 7.0);
    println!("  Phase 2 delta (r=7): {:.4}  (mult=8.0)", actual_mean_qt * 8.0);
}

/// Read a coefficient by flat index (same as pipeline::flat_get).
fn flat_get(grid: &phasm_core::DctGrid, flat_idx: usize) -> i16 {
    grid.coeffs()[flat_idx]
}

/// Soft-extract header byte (reimplemented for diagnostic access).
fn extract_header_byte_soft(
    grid: &phasm_core::DctGrid,
    positions: &[permute::CoeffPos],
    vectors: &[[f64; SPREAD_LEN]],
    delta: f64,
    offset: usize,
) -> u8 {
    let mut header_llrs = [0.0f64; 56];
    for i in 0..56 {
        let unit_idx = offset + i;
        let group_start = unit_idx * SPREAD_LEN;
        let group = &positions[group_start..group_start + SPREAD_LEN];

        let mut coeffs = [0.0f64; SPREAD_LEN];
        for (k, pos) in group.iter().enumerate() {
            coeffs[k] = flat_get(grid, pos.flat_idx) as f64;
        }

        header_llrs[i] = embedding::stdm_extract_soft(&coeffs, &vectors[unit_idx], delta);
    }

    let mut byte = 0u8;
    for bit_pos in 0..8 {
        let mut total = 0.0;
        for copy in 0..7 {
            total += header_llrs[copy * 8 + bit_pos];
        }
        if total < 0.0 {
            byte |= 1 << (7 - bit_pos);
        }
    }
    byte
}
