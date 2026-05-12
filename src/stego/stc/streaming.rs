// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 6E-C1a — Streaming sliding-window Viterbi prototype + K sweep.
//!
//! This module is a **research prototype** for Task #18 (empirical
//! K choice). It is NOT the production §6E-C1a implementation.
//! It exists to:
//!
//! 1. Validate that sliding-window Viterbi converges to the
//!    full-Viterbi reference (`stc_embed_inline`) at large enough K.
//! 2. Measure bias (Hamming distance vs reference) and per-position
//!    flip-rate divergence at varying K.
//! 3. Produce the K sweep table that goes into
//!    `docs/design/video/h264/encoder-algorithms/streaming-viterbi.md`.
//!
//! Once K is chosen, §6E-C1a will reimplement streaming Viterbi
//! with the chosen K baked in, integrated with ChaCha20 H-column
//! generation, multi-IDR support, and the GOP-parallel Pass 1 +
//! Pass 3 pipeline. This prototype takes pre-built H-hat and works
//! on a single in-memory cover stream.
//!
//! ## Algorithm
//!
//! Sliding-window (delay-line) Viterbi:
//!
//! - Forward pass identical to inline Viterbi (compute curr_cost +
//!   back_ptr at each position).
//! - When the window reaches K back_ptr entries, commit the oldest
//!   position's bit by tracing back from the current best state
//!   through the K-position window.
//! - Commits are final — the streaming variant cannot revisit them
//!   even if a later position would have changed the optimum.
//! - At end of stream, drain the remaining window with full
//!   traceback from the final best state (terminal syndrome = 0).
//!
//! Bias source: positions where the global optimum requires a
//! non-local flip whose payoff propagates beyond K positions.

use super::embed::{stc_embed, EmbedResult};
use super::extract::stc_extract;
use super::hhat;

/// Sliding-window streaming Viterbi.
///
/// Same input semantics as [`stc_embed`] plus a `window_k` parameter.
/// When `window_k >= cover_bits.len()` the result is equivalent to
/// the full-Viterbi reference (just slower due to inefficient
/// per-commit traceback).
///
/// Returns `None` if embedding is infeasible (e.g., terminal state
/// not reachable — occurs when WET coverage forces an unreachable
/// syndrome) or if input parameters are out of range (h > 7, w == 0).
pub fn stc_embed_streaming(
    cover_bits: &[u8],
    costs: &[f32],
    message: &[u8],
    hhat_matrix: &[Vec<u32>],
    h: usize,
    w: usize,
    window_k: usize,
) -> Option<EmbedResult> {
    if w == 0 || h > 7 {
        return None;
    }

    let n = cover_bits.len();
    let m = message.len();
    let num_states = 1usize << h;
    let inf = f64::INFINITY;

    if m == 0 {
        return Some(EmbedResult {
            stego_bits: cover_bits.to_vec(),
            total_cost: 0.0,
            num_modifications: 0,
        });
    }

    // Pre-compute H-hat columns.
    let columns: Vec<usize> = (0..w)
        .map(|c| hhat::column_packed(hhat_matrix, c) as usize)
        .collect();

    // Forward pass state.
    let mut prev_cost = vec![inf; num_states];
    prev_cost[0] = 0.0;
    let mut curr_cost = vec![0.0f64; num_states];
    let mut shifted_cost = vec![inf; num_states];

    // back_ptr buffer — full storage (research prototype). Production
    // will use a circular buffer of size K.
    let mut back_ptr: Vec<u128> = Vec::with_capacity(n);
    // Per-position record of "is this a message-bit boundary?" + the
    // message bit at that boundary. Used during sliding-window
    // commit traceback.
    let mut boundary: Vec<Option<u8>> = Vec::with_capacity(n);

    let mut msg_idx = 0;
    let mut stego_bits = vec![0u8; n];

    for j in 0..n {
        let col_idx = j % w;
        let col = columns[col_idx];
        let flip_cost = costs[j] as f64;
        let cover_bit = cover_bits[j] & 1;

        let (cost_s0, cost_s1) = if cover_bit == 0 {
            (0.0, flip_cost)
        } else {
            (flip_cost, 0.0)
        };

        let mut packed_bp = 0u128;
        for s in 0..num_states {
            let cost_0 = prev_cost[s] + cost_s0;
            let cost_1 = prev_cost[s ^ col] + cost_s1;
            if cost_1 < cost_0 {
                curr_cost[s] = cost_1;
                packed_bp |= 1u128 << s;
            } else {
                curr_cost[s] = cost_0;
            }
        }

        back_ptr.push(packed_bp);

        let is_msg_boundary = col_idx == w - 1 && msg_idx < m;
        if is_msg_boundary {
            let required_bit = message[msg_idx] as usize;
            shifted_cost.fill(inf);
            for s in 0..num_states {
                if curr_cost[s] == inf { continue; }
                if (s & 1) != required_bit { continue; }
                let s_shifted = s >> 1;
                if curr_cost[s] < shifted_cost[s_shifted] {
                    shifted_cost[s_shifted] = curr_cost[s];
                }
            }
            std::mem::swap(&mut prev_cost, &mut shifted_cost);
            boundary.push(Some(required_bit as u8));
            msg_idx += 1;
        } else {
            std::mem::swap(&mut prev_cost, &mut curr_cost);
            boundary.push(None);
        }

        // Sliding-window commit: when we've processed (window_k)
        // positions BEYOND the oldest uncommitted one, commit it.
        // The position to commit is `j - window_k` if that index is
        // still uncommitted.
        if window_k < n && j + 1 > window_k {
            let commit_j = j - window_k;
            commit_position(
                commit_j, j, &back_ptr, &boundary, &columns, w, m,
                num_states, &prev_cost, message, &mut stego_bits,
            );
        }
    }

    // End-of-stream: drain remaining window with full traceback from
    // the final best state.
    let (best_state, best_cost) = find_best_state(&prev_cost);
    if best_cost == inf {
        return None;
    }

    // Determine the range that still needs committing. If window_k < n,
    // positions [n - window_k .. n) are uncommitted. Otherwise [0 .. n).
    let drain_start = n.saturating_sub(window_k);

    // Walk back from the terminal best state through the remaining
    // positions [drain_start .. n).
    let mut s = best_state;
    for j in (drain_start..n).rev() {
        let col_idx = j % w;
        if let Some(msg_bit) = boundary[j] {
            s = ((s << 1) | msg_bit as usize) & (num_states - 1);
        }
        let bit = ((back_ptr[j] >> s) & 1) as u8;
        stego_bits[j] = bit;
        if bit == 1 {
            s ^= columns[col_idx];
        }
    }

    let num_modifications = stego_bits
        .iter()
        .zip(cover_bits.iter())
        .filter(|(s, c)| s != c)
        .count();

    // Compute total cost as sum of flip costs at modified positions.
    let total_cost: f64 = stego_bits
        .iter()
        .zip(cover_bits.iter())
        .zip(costs.iter())
        .filter_map(|((s, c), cost)| if s != c { Some(*cost as f64) } else { None })
        .sum();

    Some(EmbedResult { stego_bits, total_cost, num_modifications })
}

/// Walk back from current best state at position `look_ahead_j`
/// through `back_ptr[commit_j ..= look_ahead_j]` to determine the
/// committed bit at position `commit_j`. Updates `stego_bits[commit_j]`.
#[allow(clippy::too_many_arguments)]
fn commit_position(
    commit_j: usize,
    look_ahead_j: usize,
    back_ptr: &[u128],
    boundary: &[Option<u8>],
    columns: &[usize],
    _w: usize,
    _m: usize,
    num_states: usize,
    prev_cost: &[f64],
    _message: &[u8],
    stego_bits: &mut [u8],
) {
    // Best state at look_ahead_j (after forward step at look_ahead_j).
    let (mut s, _) = find_best_state(prev_cost);
    let _ = num_states;

    // Walk back from look_ahead_j to commit_j.
    for j in (commit_j..=look_ahead_j).rev() {
        let col_idx = j % columns.len();
        if let Some(msg_bit) = boundary[j] {
            s = ((s << 1) | msg_bit as usize) & (num_states - 1);
        }
        let bit = ((back_ptr[j] >> s) & 1) as u8;
        if j == commit_j {
            stego_bits[commit_j] = bit;
            return;
        }
        if bit == 1 {
            s ^= columns[col_idx];
        }
    }
}

fn find_best_state(costs: &[f64]) -> (usize, f64) {
    let mut best = 0;
    let mut best_cost = f64::INFINITY;
    for (s, &c) in costs.iter().enumerate() {
        if c < best_cost {
            best_cost = c;
            best = s;
        }
    }
    (best, best_cost)
}

/// Empirical K sweep: compares the full-Viterbi reference to
/// sliding-window plans at various K, reports bias metrics.
///
/// Returns a `Vec<KSweepEntry>` with one entry per K tested.
pub fn run_k_sweep(
    cover_bits: &[u8],
    costs: &[f32],
    message: &[u8],
    hhat_matrix: &[Vec<u32>],
    h: usize,
    w: usize,
    k_values: &[usize],
) -> KSweepReport {
    let reference = stc_embed(cover_bits, costs, message, hhat_matrix, h, w)
        .expect("reference Viterbi failed");
    let n = cover_bits.len();

    // Per-position cover→stego flip indicator for the reference.
    let ref_flips: Vec<bool> = reference
        .stego_bits
        .iter()
        .zip(cover_bits.iter())
        .map(|(s, c)| s != c)
        .collect();
    let ref_total_flips = ref_flips.iter().filter(|f| **f).count();

    let mut entries = Vec::new();
    for &k in k_values {
        let start = std::time::Instant::now();
        let sw = stc_embed_streaming(
            cover_bits, costs, message, hhat_matrix, h, w, k,
        ).expect("streaming Viterbi failed");
        let wall_clock_ms = start.elapsed().as_millis();

        // Hamming distance between reference and streaming stego_bits.
        let hamming: usize = sw
            .stego_bits
            .iter()
            .zip(reference.stego_bits.iter())
            .filter(|(a, b)| a != b)
            .count();
        let hamming_pct = (hamming as f64) / (n as f64) * 100.0;

        // Per-position flip-rate divergence:
        // |sw_flip_rate - ref_flip_rate|, averaged in 1000-position
        // bins for stability.
        let bin_size = (n / 100).max(1);
        let mut max_bin_divergence: f64 = 0.0;
        let mut sum_bin_divergence: f64 = 0.0;
        let mut bin_count = 0;
        let mut start_idx = 0;
        while start_idx < n {
            let end = (start_idx + bin_size).min(n);
            let ref_bin: usize = ref_flips[start_idx..end].iter().filter(|f| **f).count();
            let sw_bin: usize = sw.stego_bits[start_idx..end]
                .iter()
                .zip(cover_bits[start_idx..end].iter())
                .filter(|(s, c)| s != c)
                .count();
            let len = end - start_idx;
            let div = ((ref_bin as f64) - (sw_bin as f64)).abs() / (len as f64);
            if div > max_bin_divergence { max_bin_divergence = div; }
            sum_bin_divergence += div;
            bin_count += 1;
            start_idx = end;
        }
        let avg_bin_divergence = sum_bin_divergence / (bin_count as f64);

        // Syndrome validity check: extract from streaming stego_bits
        // and compare to original message. STC delay-line may NOT
        // satisfy H·s=m because committed bits aren't constrained by
        // the terminal-state requirement.
        let extracted = stc_extract(&sw.stego_bits, hhat_matrix, w);
        let syndrome_match_bits: usize = extracted[..message.len()]
            .iter()
            .zip(message.iter())
            .filter(|(a, b)| a == b)
            .count();
        let syndrome_valid = syndrome_match_bits == message.len();
        let syndrome_match_pct = (syndrome_match_bits as f64) / (message.len() as f64) * 100.0;

        entries.push(KSweepEntry {
            k,
            n,
            hamming,
            hamming_pct,
            sw_total_flips: sw.num_modifications,
            ref_total_flips,
            sw_total_cost: sw.total_cost,
            ref_total_cost: reference.total_cost,
            max_bin_flip_rate_divergence: max_bin_divergence,
            avg_bin_flip_rate_divergence: avg_bin_divergence,
            wall_clock_ms,
            syndrome_valid,
            syndrome_match_pct,
        });
    }

    KSweepReport {
        n,
        m: message.len(),
        h,
        w,
        ref_total_flips,
        ref_total_cost: reference.total_cost,
        entries,
    }
}

#[derive(Debug, Clone)]
pub struct KSweepEntry {
    pub k: usize,
    pub n: usize,
    pub hamming: usize,
    pub hamming_pct: f64,
    pub sw_total_flips: usize,
    pub ref_total_flips: usize,
    pub sw_total_cost: f64,
    pub ref_total_cost: f64,
    pub max_bin_flip_rate_divergence: f64,
    pub avg_bin_flip_rate_divergence: f64,
    pub wall_clock_ms: u128,
    /// Does H·streaming_stego == m? CRITICAL for STC streaming —
    /// a pure delay-line Viterbi violates this because mid-stream
    /// commits aren't constrained by the terminal-state requirement.
    pub syndrome_valid: bool,
    /// Percentage of message bits that the streaming output decodes
    /// to correctly. 100% = syndrome valid.
    pub syndrome_match_pct: f64,
}

#[derive(Debug, Clone)]
pub struct KSweepReport {
    pub n: usize,
    pub m: usize,
    pub h: usize,
    pub w: usize,
    pub ref_total_flips: usize,
    pub ref_total_cost: f64,
    pub entries: Vec<KSweepEntry>,
}

impl KSweepReport {
    /// Format the sweep report as a markdown table suitable for
    /// pasting into the design note.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "## K sweep results\n\n\
             Cover length: n = {}, message: m = {} bits, h = {}, w = {}, \
             reference flips: {} ({:.4}% of n), reference cost: {:.2}\n\n",
            self.n, self.m, self.h, self.w,
            self.ref_total_flips,
            (self.ref_total_flips as f64) / (self.n as f64) * 100.0,
            self.ref_total_cost,
        ));
        out.push_str("| K | syndrome | Hamming | Hamming% | sw flips | ref flips | cost-Δ | max bin Δ | avg bin Δ | wall ms |\n");
        out.push_str("|---|---|---|---|---|---|---|---|---|---|\n");
        for e in &self.entries {
            let cost_delta = e.sw_total_cost - e.ref_total_cost;
            let syn = if e.syndrome_valid {
                "✓ valid".to_string()
            } else {
                format!("✗ {:.2}%", e.syndrome_match_pct)
            };
            out.push_str(&format!(
                "| {} | {} | {} | {:.4}% | {} | {} | {:+.2} | {:.6} | {:.6} | {} |\n",
                e.k, syn, e.hamming, e.hamming_pct, e.sw_total_flips,
                e.ref_total_flips, cost_delta,
                e.max_bin_flip_rate_divergence,
                e.avg_bin_flip_rate_divergence,
                e.wall_clock_ms,
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::hhat::generate_hhat;

    /// Sanity: streaming with K >= n must equal the full Viterbi.
    #[test]
    fn streaming_with_full_window_matches_inline() {
        let h = 7;
        let m = 100;
        let w = 10;
        let n = m * w;
        let seed = [42u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| ((i * 31 + 17) % 2) as u8).collect();
        let costs: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let message: Vec<u8> = (0..m).map(|i| ((i * 13 + 7) % 2) as u8).collect();

        let inline = stc_embed(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        let stream = stc_embed_streaming(&cover_bits, &costs, &message, &hhat, h, w, n).unwrap();

        assert_eq!(inline.stego_bits, stream.stego_bits,
            "K=n (full window) must equal inline Viterbi");
    }

    /// Streaming with K=1 should produce some output (correct
    /// syndrome from end-of-stream traceback) but with higher
    /// distortion / different stego bits than reference.
    #[test]
    fn streaming_with_tiny_window_runs_without_error() {
        let h = 4;
        let m = 50;
        let w = 5;
        let n = m * w;
        let seed = [13u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| ((i * 7) % 2) as u8).collect();
        let costs: Vec<f32> = vec![1.0; n];
        let message: Vec<u8> = (0..m).map(|i| (i % 2) as u8).collect();

        let stream = stc_embed_streaming(&cover_bits, &costs, &message, &hhat, h, w, 5).unwrap();
        assert_eq!(stream.stego_bits.len(), n);
    }

    /// **Task #18 — empirical K sweep.** Runs the sliding-window
    /// Viterbi at various K against a full-Viterbi reference,
    /// prints a markdown table for inclusion in the streaming-Viterbi
    /// design note. Marked `#[ignore]` because the largest K value
    /// triggers minutes of compute; run explicitly with:
    /// `cargo test --features cabac-stego --lib streaming::tests::k_sweep_report -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn k_sweep_report() {
        let h = 7;
        let m = 1000;
        let w = 100;
        let n = m * w; // 100K positions
        let seed = [77u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        // Synthetic cover: pseudo-random bits with mostly-uniform
        // costs (small variance) — representative of typical phasm
        // bypass-bin coverage.
        let cover_bits: Vec<u8> = (0..n)
            .map(|i| {
                // simple LCG → bit
                let mut x = i as u64;
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((x >> 33) & 1) as u8
            })
            .collect();
        let costs: Vec<f32> = (0..n)
            .map(|i| {
                let mut x = (i as u64).wrapping_add(0xcafef00d);
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let frac = ((x >> 32) & 0xffff) as f32 / 65536.0; // [0, 1)
                0.5 + frac * 0.5 // [0.5, 1.0)
            })
            .collect();
        let message: Vec<u8> = (0..m).map(|i| ((i * 19 + 11) % 2) as u8).collect();

        let k_values = vec![100, 500, 1000, 5000, 10000, 50000];
        let report = run_k_sweep(&cover_bits, &costs, &message, &hhat, h, w, &k_values);

        println!("\n{}\n", report.to_markdown());
    }
}
