// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Viterbi-based STC embedding.
//!
//! Implements the forward (Viterbi) and backward (traceback) passes of the
//! STC embedding algorithm. The encoder finds the minimum-cost stego bit
//! sequence whose syndrome (under the H-hat matrix) matches the message.
//!
//! Two internal paths:
//! - **Inline** (n ≤ 1M): stores all back pointers in one pass — fastest.
//! - **Segmented** (n > 1M): checkpoint/recompute approach — O(√n) memory,
//!   2× compute. Enables 48 MP+ images on memory-constrained devices.

use super::hhat;
use super::extract::stc_extract;
use crate::stego::progress;

/// Result of STC embedding: the stego bit sequence and its total distortion cost.
pub struct EmbedResult {
    pub stego_bits: Vec<u8>,
    pub total_cost: f64,
    /// Number of positions where cover bit != stego bit.
    pub num_modifications: usize,
}

/// Number of progress steps reported during STC Viterbi embedding.
///
/// Distributed across the forward pass(es). STC is typically ~20-30% of
/// total encode time on large images.
pub const STC_PROGRESS_STEPS: u32 = 50;

/// Back-pointer memory threshold (in cover positions). Above this, the
/// segmented path is used. 1M positions × 16 bytes/u128 = 16 MB.
const SEGMENTED_THRESHOLD: usize = 1_000_000;

/// Embed a message into cover bits using the Viterbi-based STC algorithm.
///
/// Automatically selects the inline path (single-pass, O(n) memory) for
/// small inputs, or the segmented path (checkpoint/recompute, O(√n) memory)
/// for large inputs. Both paths produce identical output.
///
/// - `cover_bits`: LSBs of the cover coefficients (length n)
/// - `costs`: cost of flipping each cover bit (length n). Use f32::INFINITY for WET.
///   Promoted to f64 internally for accumulation precision.
/// - `message`: message bits to embed (length m)
/// - `hhat_matrix`: the H-hat submatrix (h rows × w columns)
/// - `h`: constraint length (must be ≤ 7 so 2^h fits in u128)
/// - `w`: submatrix width (should be ceil(n/m))
///
/// Returns the stego bit sequence that encodes the message with minimum
/// distortion, or `None` if embedding is infeasible.
///
/// Reports [`STC_PROGRESS_STEPS`] progress sub-steps via [`progress::advance`]
/// during the Viterbi forward pass(es).
pub fn stc_embed(
    cover_bits: &[u8],
    costs: &[f32],
    message: &[u8],
    hhat_matrix: &[Vec<u32>],
    h: usize,
    w: usize,
) -> Option<EmbedResult> {
    // h ≤ 7 required: 2^7 = 128 states fit exactly in u128.
    if w == 0 || h > 7 {
        return None;
    }

    let n = cover_bits.len();
    let m = message.len();

    if m == 0 {
        return Some(EmbedResult {
            stego_bits: cover_bits.to_vec(),
            total_cost: 0.0,
            num_modifications: 0,
        });
    }

    if n > SEGMENTED_THRESHOLD {
        stc_embed_segmented(cover_bits, costs, message, hhat_matrix, h, w)
    } else {
        stc_embed_inline(cover_bits, costs, message, hhat_matrix, h, w)
    }
}

// ---------------------------------------------------------------------------
// Inline path: single forward pass, stores all back pointers.
// Best for small/medium images where O(n) memory is acceptable.
// ---------------------------------------------------------------------------

fn stc_embed_inline(
    cover_bits: &[u8],
    costs: &[f32],
    message: &[u8],
    hhat_matrix: &[Vec<u32>],
    h: usize,
    w: usize,
) -> Option<EmbedResult> {
    let n = cover_bits.len();
    let m = message.len();
    let num_states = 1usize << h;
    let inf = f64::INFINITY;

    // Pre-compute H-hat columns (avoids repeated column_packed calls).
    let columns: Vec<usize> = (0..w)
        .map(|c| hhat::column_packed(hhat_matrix, c) as usize)
        .collect();

    // Progress: advance every n/STC_PROGRESS_STEPS elements.
    let progress_interval = (n / STC_PROGRESS_STEPS as usize).max(1);

    // Forward Viterbi pass with 1-bit packed back pointers.
    // back_ptr[j] is a u128: bit s = 1 means stego_bit=1 was chosen for
    // target state s (predecessor = s ^ col). 16 bytes per step.
    //
    // Pre-allocated cost buffers avoid per-iteration heap allocations.
    // The target-state iteration writes every entry, so no fill needed
    // for curr_cost. Only shifted_cost needs fill (sparse writes).
    let mut prev_cost = vec![inf; num_states];
    prev_cost[0] = 0.0;
    let mut curr_cost = vec![0.0f64; num_states];
    let mut shifted_cost = vec![inf; num_states];

    let mut back_ptr: Vec<u128> = Vec::with_capacity(n);
    let mut msg_idx = 0;

    for j in 0..n {
        let col_idx = j % w;
        let col = columns[col_idx];
        let flip_cost = costs[j] as f64; // promote f32→f64 for accumulation
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

        if col_idx == w - 1 && msg_idx < m {
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
            msg_idx += 1;
        } else {
            std::mem::swap(&mut prev_cost, &mut curr_cost);
        }

        if (j + 1) % progress_interval == 0 {
            if progress::is_cancelled() { return None; }
            progress::advance();
        }
    }

    // Find terminal state with minimum cost.
    let (best_state, best_cost) = find_best_state(&prev_cost);
    if best_cost == inf { return None; }

    // Backward traceback.
    let mut stego_bits = vec![0u8; n];
    let mut s = best_state;

    for j in (0..n).rev() {
        let col_idx = j % w;

        if col_idx == w - 1 && (j / w) < m {
            let msg_bit = message[j / w] as usize;
            s = ((s << 1) | msg_bit) & (num_states - 1);
        }

        let bit = ((back_ptr[j] >> s) & 1) as u8;
        stego_bits[j] = bit;

        if bit == 1 {
            s ^= columns[col_idx];
        }
    }

    debug_assert_eq!(s, 0, "traceback did not return to initial state 0");
    debug_assert_eq!(
        stc_extract(&stego_bits, hhat_matrix, w)[..m],
        message[..m],
    );

    let num_modifications = stego_bits.iter().zip(cover_bits.iter())
        .filter(|(s, c)| s != c).count();

    Some(EmbedResult { stego_bits, total_cost: best_cost, num_modifications })
}

// ---------------------------------------------------------------------------
// Segmented path: checkpoint/recompute for O(√n) memory.
// Two forward passes: one to save checkpoints, one to recompute segments.
// ---------------------------------------------------------------------------

fn stc_embed_segmented(
    cover_bits: &[u8],
    costs: &[f32],
    message: &[u8],
    hhat_matrix: &[Vec<u32>],
    h: usize,
    w: usize,
) -> Option<EmbedResult> {
    let n = cover_bits.len();
    let m = message.len();
    let num_states = 1usize << h;
    let inf = f64::INFINITY;

    // Pre-compute H-hat columns (avoids repeated column_packed calls
    // across Phase A, Phase B forward, and Phase B traceback).
    let columns: Vec<usize> = (0..w)
        .map(|c| hhat::column_packed(hhat_matrix, c) as usize)
        .collect();

    // Checkpoint interval: K message blocks per segment.
    // sqrt(m) balances checkpoint memory (K × 1 KB) with segment back_ptr
    // memory (K × w × 16 bytes).
    let k = ((m as f64).sqrt().ceil() as usize).max(1);
    let num_segments = (m + k - 1) / k;

    // --- Phase A: forward scan, save checkpoints, no back_ptr ---
    // Reports half the STC progress sub-steps.
    let phase_a_steps = STC_PROGRESS_STEPS / 2;
    let progress_interval_a = (n / phase_a_steps as usize).max(1);

    // Pre-allocated cost buffers reused across all iterations.
    let mut prev_cost = vec![inf; num_states];
    prev_cost[0] = 0.0;
    let mut curr_cost = vec![0.0f64; num_states];
    let mut shifted_cost = vec![inf; num_states];

    // checkpoint[s] = cost array at the START of segment s (post-shift from
    // the preceding block, or the initial state for s=0).
    let mut checkpoints: Vec<Vec<f64>> = Vec::with_capacity(num_segments);
    checkpoints.push(prev_cost.clone());

    let mut msg_idx = 0;

    for j in 0..n {
        let col_idx = j % w;
        let col = columns[col_idx];
        let flip_cost = costs[j] as f64; // promote f32→f64
        let cover_bit = cover_bits[j] & 1;

        let (cost_s0, cost_s1) = if cover_bit == 0 {
            (0.0, flip_cost)
        } else {
            (flip_cost, 0.0)
        };

        for s in 0..num_states {
            let cost_0 = prev_cost[s] + cost_s0;
            let cost_1 = prev_cost[s ^ col] + cost_s1;
            curr_cost[s] = if cost_1 < cost_0 { cost_1 } else { cost_0 };
        }

        if col_idx == w - 1 && msg_idx < m {
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
            msg_idx += 1;

            // Save checkpoint at segment boundaries.
            if msg_idx % k == 0 && msg_idx < m {
                checkpoints.push(prev_cost.clone());
            }
        } else {
            std::mem::swap(&mut prev_cost, &mut curr_cost);
        }

        if (j + 1) % progress_interval_a == 0 {
            if progress::is_cancelled() { return None; }
            progress::advance();
        }
    }

    // Find terminal state with minimum cost.
    let (best_state, best_cost) = find_best_state(&prev_cost);
    if best_cost == inf { return None; }

    // --- Phase B: segment-by-segment recomputation + traceback ---
    // Reports the remaining STC progress sub-steps.
    let phase_b_steps = STC_PROGRESS_STEPS - phase_a_steps;
    let progress_interval_b = (n / phase_b_steps as usize).max(1);
    let mut progress_counter = 0usize;

    let mut stego_bits = vec![0u8; n];
    let mut entry_state = best_state;

    for seg in (0..num_segments).rev() {
        let block_start = seg * k;
        let block_end = ((seg + 1) * k).min(m);
        let j_start = block_start * w;
        let j_end = block_end * w;
        let seg_len = j_end - j_start;

        // Reset prev_cost from checkpoint (reuses the same buffer).
        prev_cost.copy_from_slice(&checkpoints[seg]);

        // Recompute forward Viterbi for this segment, storing back_ptr.
        let mut seg_back_ptr: Vec<u128> = Vec::with_capacity(seg_len);
        let mut seg_msg_idx = block_start;

        for j in j_start..j_end {
            let col_idx = j % w;
            let col = columns[col_idx];
            let flip_cost = costs[j] as f64; // promote f32→f64
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

            seg_back_ptr.push(packed_bp);

            if col_idx == w - 1 && seg_msg_idx < m {
                let required_bit = message[seg_msg_idx] as usize;
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
                seg_msg_idx += 1;
            } else {
                std::mem::swap(&mut prev_cost, &mut curr_cost);
            }

            progress_counter += 1;
            if progress_counter % progress_interval_b == 0 {
                if progress::is_cancelled() { return None; }
                progress::advance();
            }
        }

        // Traceback within this segment.
        let mut s = entry_state;
        for local_j in (0..seg_len).rev() {
            let j = j_start + local_j;
            let col_idx = j % w;

            if col_idx == w - 1 && (j / w) < m {
                let msg_bit = message[j / w] as usize;
                s = ((s << 1) | msg_bit) & (num_states - 1);
            }

            let bit = ((seg_back_ptr[local_j] >> s) & 1) as u8;
            stego_bits[j] = bit;

            if bit == 1 {
                s ^= columns[col_idx];
            }
        }

        // State at the start of this segment = entry state for previous segment.
        entry_state = s;
        // seg_back_ptr is dropped here, freeing the segment's memory.
    }

    debug_assert_eq!(entry_state, 0, "traceback did not return to initial state 0");
    debug_assert_eq!(
        stc_extract(&stego_bits, hhat_matrix, w)[..m],
        message[..m],
    );

    let num_modifications = stego_bits.iter().zip(cover_bits.iter())
        .filter(|(s, c)| s != c).count();

    Some(EmbedResult { stego_bits, total_cost: best_cost, num_modifications })
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Find the state with minimum cost. Returns (state, cost).
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::hhat::generate_hhat;
    use super::super::extract::stc_extract;

    #[test]
    fn embed_extract_roundtrip_tiny() {
        let h = 3;
        let n = 20;
        let m = 4;
        let w = (n + m - 1) / m; // ceil(20/4) = 5
        let seed = [42u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let costs: Vec<f32> = vec![1.0; n];
        let message = vec![1u8, 0, 1, 1];

        let result = stc_embed(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        assert_eq!(result.stego_bits.len(), n);

        let extracted = stc_extract(&result.stego_bits, &hhat, w);
        assert_eq!(&extracted[..m], &message[..]);
    }

    #[test]
    fn embed_extract_roundtrip_h7() {
        let h = 7;
        let n = 500;
        let m = 50;
        let w = (n + m - 1) / m;
        let seed = [13u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| ((i * 7 + 3) % 2) as u8).collect();
        let costs: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let message: Vec<u8> = (0..m).map(|i| (i % 2) as u8).collect();

        let result = stc_embed(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        let extracted = stc_extract(&result.stego_bits, &hhat, w);
        assert_eq!(&extracted[..m], &message[..]);
    }

    #[test]
    fn wet_coefficients_not_modified() {
        let h = 3;
        let n = 20;
        let m = 4;
        let w = (n + m - 1) / m;
        let seed = [55u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = vec![0; n];
        let mut costs: Vec<f32> = vec![1.0; n];
        // Make positions 0, 5, 10, 15 WET
        for i in (0..n).step_by(5) {
            costs[i] = 1e13;
        }
        let message = vec![0u8, 1, 0, 1];

        let result = stc_embed(&cover_bits, &costs, &message, &hhat, h, w).unwrap();

        // WET positions must not change
        for i in (0..n).step_by(5) {
            assert_eq!(
                result.stego_bits[i], cover_bits[i],
                "WET position {i} was modified"
            );
        }

        // Message still recoverable
        let extracted = stc_extract(&result.stego_bits, &hhat, w);
        assert_eq!(&extracted[..m], &message[..]);
    }

    #[test]
    fn empty_message() {
        let h = 3;
        let n = 10;
        let w = 5;
        let seed = [0u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = vec![1; n];
        let costs: Vec<f32> = vec![1.0; n];
        let message: Vec<u8> = vec![];

        let result = stc_embed(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        assert_eq!(result.stego_bits, cover_bits);
        assert_eq!(result.total_cost, 0.0);
    }

    /// Large synthetic test to verify 1-bit packed back pointers at scale.
    #[test]
    fn embed_extract_roundtrip_large() {
        let h = 7;
        let m = 10_000;
        let w = 10;
        let n = m * w; // 100K cover elements
        let seed = [77u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| ((i * 31 + 17) % 2) as u8).collect();
        let costs: Vec<f32> = (0..n).map(|i| {
            let base = 0.5 + (i % 100) as f32 * 0.02;
            if i % 500 == 0 { f32::INFINITY } else { base }
        }).collect();
        let message: Vec<u8> = (0..m).map(|i| ((i * 13 + 7) % 2) as u8).collect();

        let result = stc_embed(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        assert_eq!(result.stego_bits.len(), n);

        let extracted = stc_extract(&result.stego_bits, &hhat, w);
        assert_eq!(&extracted[..m], &message[..]);

        for i in (0..n).step_by(500) {
            assert_eq!(
                result.stego_bits[i], cover_bits[i],
                "WET position {i} was modified"
            );
        }
    }

    /// Verify that inline and segmented paths produce identical output.
    #[test]
    fn inline_segmented_equivalence() {
        let h = 7;
        let m = 500;
        let w = 10;
        let n = m * w; // 5000 cover elements
        let seed = [99u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| ((i * 31 + 17) % 2) as u8).collect();
        let costs: Vec<f32> = (0..n).map(|i| {
            let base = 0.5 + (i % 100) as f32 * 0.02;
            if i % 500 == 0 { f32::INFINITY } else { base }
        }).collect();
        let message: Vec<u8> = (0..m).map(|i| ((i * 13 + 7) % 2) as u8).collect();

        let inline = stc_embed_inline(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        let segmented = stc_embed_segmented(&cover_bits, &costs, &message, &hhat, h, w).unwrap();

        assert_eq!(inline.stego_bits, segmented.stego_bits, "stego bits differ");
        assert_eq!(inline.total_cost, segmented.total_cost, "total cost differs");
    }

    /// Equivalence test with a larger input covering multiple segments.
    #[test]
    fn inline_segmented_equivalence_large() {
        let h = 7;
        let m = 10_000;
        let w = 10;
        let n = m * w; // 100K cover elements, K ≈ 100 → ~100 segments
        let seed = [88u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| ((i * 37 + 11) % 2) as u8).collect();
        let costs: Vec<f32> = (0..n).map(|i| {
            let base = 0.3 + (i % 200) as f32 * 0.01;
            if i % 1000 == 0 { f32::INFINITY } else { base }
        }).collect();
        let message: Vec<u8> = (0..m).map(|i| ((i * 19 + 3) % 2) as u8).collect();

        let inline = stc_embed_inline(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        let segmented = stc_embed_segmented(&cover_bits, &costs, &message, &hhat, h, w).unwrap();

        assert_eq!(inline.stego_bits, segmented.stego_bits, "stego bits differ");
        assert_eq!(inline.total_cost, segmented.total_cost, "total cost differs");
    }

    /// Segmented path with a single segment (m ≤ K).
    #[test]
    fn segmented_single_segment() {
        let h = 7;
        let m = 4;
        let w = 5;
        let n = m * w;
        let seed = [33u8; 32];
        let hhat = generate_hhat(h, w, &seed);

        let cover_bits: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let costs: Vec<f32> = vec![1.0; n];
        let message: Vec<u8> = vec![1, 0, 1, 1];

        let inline = stc_embed_inline(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        let segmented = stc_embed_segmented(&cover_bits, &costs, &message, &hhat, h, w).unwrap();

        assert_eq!(inline.stego_bits, segmented.stego_bits);
        assert_eq!(inline.total_cost, segmented.total_cost);
    }
}
