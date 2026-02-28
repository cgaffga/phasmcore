// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Viterbi-based STC embedding.
//!
//! Implements the forward (Viterbi) and backward (traceback) passes of the
//! STC embedding algorithm. The encoder finds the minimum-cost stego bit
//! sequence whose syndrome (under the H-hat matrix) matches the message.

use super::hhat;
use super::extract::stc_extract;

/// Result of STC embedding: the stego bit sequence and its total distortion cost.
pub struct EmbedResult {
    pub stego_bits: Vec<u8>,
    pub total_cost: f64,
}

/// Embed a message into cover bits using the Viterbi-based STC algorithm.
///
/// The STC trellis has 2^h states representing an h-bit syndrome window.
/// Cover elements are processed left to right. Every `w` elements, one message
/// bit is emitted (the bottom bit of the state), and the state shifts right by 1.
///
/// - `cover_bits`: LSBs of the cover coefficients (length n)
/// - `costs`: cost of flipping each cover bit (length n). Use `WET_COST` (f64::INFINITY) for WET.
/// - `message`: message bits to embed (length m)
/// - `hhat_matrix`: the H-hat submatrix (h rows × w columns)
/// - `h`: constraint length
/// - `w`: submatrix width (should be ceil(n/m))
///
/// Returns the stego bit sequence that encodes the message with minimum distortion,
/// or `None` if embedding is infeasible.
pub fn stc_embed(
    cover_bits: &[u8],
    costs: &[f64],
    message: &[u8],
    hhat_matrix: &[Vec<u32>],
    h: usize,
    w: usize,
) -> Option<EmbedResult> {
    let n = cover_bits.len();
    let m = message.len();

    if m == 0 {
        return Some(EmbedResult {
            stego_bits: cover_bits.to_vec(),
            total_cost: 0.0,
        });
    }

    let num_states = 1usize << h;
    let inf = f64::INFINITY;

    // Forward Viterbi pass.
    // prev_cost[s] = minimum cost to reach state s after the previous element.
    let mut prev_cost = vec![inf; num_states];
    prev_cost[0] = 0.0; // start in state 0

    // Back pointers: back_ptr[j][s] = previous state for each reachable state at step j.
    let mut back_ptr: Vec<Vec<u32>> = Vec::with_capacity(n);

    let mut msg_idx = 0; // index into message bits

    for j in 0..n {
        let mut curr_cost = vec![inf; num_states];
        let mut bp = vec![0u32; num_states];

        let col_idx = j % w;
        let col = hhat::column_packed(hhat_matrix, col_idx);

        let flip_cost = costs[j];
        let cover_bit = cover_bits[j] & 1;

        for s_prev in 0..num_states {
            if prev_cost[s_prev] == inf {
                continue;
            }

            // Option 1: keep cover bit (stego_bit = cover_bit).
            let s_keep = if cover_bit == 1 {
                s_prev ^ (col as usize)
            } else {
                s_prev
            };

            // Option 2: flip cover bit (stego_bit = 1 - cover_bit).
            let s_flip = if cover_bit == 0 {
                s_prev ^ (col as usize)
            } else {
                s_prev
            };

            let cost_keep = prev_cost[s_prev];
            if cost_keep < curr_cost[s_keep] {
                curr_cost[s_keep] = cost_keep;
                bp[s_keep] = s_prev as u32;
            }

            // Skip flip option for WET (infinite cost) positions — these
            // coefficients must not be modified.
            if flip_cost.is_finite() {
                let cost_flip = prev_cost[s_prev] + flip_cost;
                if cost_flip < curr_cost[s_flip] {
                    curr_cost[s_flip] = cost_flip;
                    bp[s_flip] = s_prev as u32;
                }
            }
        }

        // At message-bit boundaries (every w elements), constrain and shift.
        if col_idx == w - 1 && msg_idx < m {
            let required_bit = message[msg_idx] as usize;

            // After shift, the new state will be curr_state >> 1.
            // Before shift, bit 0 of curr_state must equal the message bit.
            // So we prune states whose bit 0 doesn't match, then shift all surviving states.
            let mut shifted_cost = vec![inf; num_states];
            let mut shifted_bp = vec![0u32; num_states]; // store pre-shift state

            for s in 0..num_states {
                if curr_cost[s] == inf {
                    continue;
                }
                if (s & 1) != required_bit {
                    continue; // doesn't match message bit
                }
                let s_shifted = s >> 1;
                if curr_cost[s] < shifted_cost[s_shifted] {
                    shifted_cost[s_shifted] = curr_cost[s];
                    shifted_bp[s_shifted] = s as u32; // remember pre-shift state
                }
            }

            // We need to store both: the normal back pointer for this step,
            // and the shift mapping. We'll encode the shift into the back_ptr
            // by storing the pre-shift state. During traceback, we'll handle
            // shift boundaries specially.

            // Store the regular bp first (for the transition within this step).
            back_ptr.push(bp);

            // Add a virtual "shift" step: back_ptr entry mapping shifted state -> pre-shift state.
            back_ptr.push(shifted_bp);

            prev_cost = shifted_cost;
            msg_idx += 1;
        } else {
            back_ptr.push(bp);
            prev_cost = curr_cost;
        }
    }

    // Handle case where last block is incomplete (no shift was applied).
    // If n is not a multiple of w, the last partial block may not have emitted
    // a message bit. For the padded message approach (m_max), this should be fine
    // since w = ceil(n/m) guarantees n >= (m-1)*w + 1.

    // Find terminal state with minimum cost.
    let mut best_state = 0;
    let mut best_cost = inf;
    for s in 0..num_states {
        if prev_cost[s] < best_cost {
            best_cost = prev_cost[s];
            best_state = s;
        }
    }

    if best_cost == inf {
        return None;
    }

    // Backward traceback through back_ptr (which includes virtual shift steps).
    // Total entries in back_ptr = n + number_of_shifts.
    let total_steps = back_ptr.len();
    let mut states = vec![0usize; total_steps + 1];
    states[total_steps] = best_state;

    for step in (0..total_steps).rev() {
        states[step] = back_ptr[step][states[step + 1]] as usize;
    }

    // Now reconstruct stego bits. We need to map back_ptr steps to cover element indices.
    // Shift steps are virtual — they don't correspond to cover elements.
    // Build a mapping: for each back_ptr step, is it a cover step or a shift step?
    let mut step_is_shift = vec![false; total_steps];
    {
        let mut mi = 0;
        let mut step = 0;
        for j in 0..n {
            step += 1; // cover step
            let col_idx = j % w;
            if col_idx == w - 1 && mi < m {
                step_is_shift[step] = true;
                step += 1; // shift step
                mi += 1;
            }
        }
    }

    // Extract stego bits from state transitions (only cover steps).
    let mut stego_bits = Vec::with_capacity(n);
    let mut step = 0;
    for j in 0..n {
        let s_before = states[step];
        let s_after = states[step + 1];

        let col_idx = j % w;
        let col = hhat::column_packed(hhat_matrix, col_idx) as usize;

        // Determine the applied stego bit.
        // If stego bit = 1: s_after = s_before ^ col
        // If stego bit = 0: s_after = s_before
        let applied_bit = if s_after == s_before ^ col { 1u8 } else { 0u8 };
        stego_bits.push(applied_bit);

        step += 1;
        // Skip virtual shift step if present.
        if col_idx == w - 1 && step < total_steps && step_is_shift[step] {
            step += 1;
        }
    }

    // Verify correctness.
    debug_assert_eq!(
        stc_extract(&stego_bits, hhat_matrix, w)[..m],
        message[..m],
    );

    Some(EmbedResult {
        stego_bits,
        total_cost: best_cost,
    })
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
        let costs: Vec<f64> = vec![1.0; n];
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
        let costs: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.01).collect();
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
        let mut costs: Vec<f64> = vec![1.0; n];
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
        let costs: Vec<f64> = vec![1.0; n];
        let message: Vec<u8> = vec![];

        let result = stc_embed(&cover_bits, &costs, &message, &hhat, h, w).unwrap();
        assert_eq!(result.stego_bits, cover_bits);
        assert_eq!(result.total_cost, 0.0);
    }
}
