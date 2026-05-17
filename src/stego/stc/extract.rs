// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! STC message extraction via syndrome computation.
//!
//! Extracts the embedded message from a stego bit sequence by computing
//! the syndrome under the H-hat matrix. This is the decoder counterpart
//! to the Viterbi embedding in [`super::embed`].

use super::hhat;

/// Extract a message from stego bits using the STC syndrome computation.
///
/// The extraction walks the banded H matrix structure: for each message bit i,
/// it accumulates the syndrome over cover elements [i*w, (i+1)*w) using H-hat
/// columns, then reads the bottom bit of the state as message[i] and shifts
/// the state right by 1 to advance the syndrome window.
pub fn stc_extract(stego_bits: &[u8], hhat: &[Vec<u32>], w: usize) -> Vec<u8> {
    if w == 0 {
        return Vec::new();
    }

    let n = stego_bits.len();
    let m = n.div_ceil(w); // ceil(n / w)
    stc_extract_prefix(stego_bits, hhat, w, m)
}

/// Partial extraction — compute only the first `k_msg_bits` message
/// bits and stop early. Same syndrome-walking logic as
/// [`stc_extract`] but bounded.
///
/// **Use case (#516.2):** brute-force `m_total` decoders try many
/// candidate message lengths. For each candidate, the first 16
/// syndrome bits encode the phasm v1 frame's `u16 plaintext_len`
/// header. The decoder can verify
/// `(FRAME_OVERHEAD + plaintext_len) * 8 == m_total` after just
/// 16-bit partial extract — for the WRONG `m_total` the prefix is
/// essentially random and the equality holds with probability
/// 1/65536. Almost every wrong candidate rejects in `O(16 * w)`
/// work instead of the full `O(m_total * w)`.
///
/// On a 1080p × 30f IPPPP cover with 55 candidate tries, this drops
/// the brute-force search from ~106 ms (55 full extracts) to ~4 ms
/// (55 prefix extracts + 1 full extract on the winner).
pub fn stc_extract_prefix(
    stego_bits: &[u8],
    hhat: &[Vec<u32>],
    w: usize,
    k_msg_bits: usize,
) -> Vec<u8> {
    if w == 0 || k_msg_bits == 0 {
        return Vec::new();
    }

    let n = stego_bits.len();
    let max_i = k_msg_bits.min(n.div_ceil(w));

    let mut message = Vec::with_capacity(max_i);
    let mut state = 0u32;

    for i in 0..max_i {
        let start = i * w;
        let end = (start + w).min(n);

        // Accumulate syndrome contributions from this block.
        for j in start..end {
            if stego_bits[j] & 1 != 0 {
                let col_idx = j - start; // = j % w
                state ^= hhat::column_packed(hhat, col_idx);
            }
        }

        // The bottom bit of the state is the fully-determined syndrome bit = message[i].
        message.push((state & 1) as u8);

        // Shift the syndrome window: bit 0 is consumed, bits 1..h-1 carry forward.
        state >>= 1;
    }

    message
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::hhat::generate_hhat;

    #[test]
    fn known_syndrome() {
        // Trivial case: all-zero stego bits → all-zero message
        let h = 3;
        let w = 4;
        let hhat = generate_hhat(h, w, &[99u8; 32]);
        let stego = vec![0u8; 12]; // n=12, m=3
        let msg = stc_extract(&stego, &hhat, w);
        assert_eq!(msg, vec![0, 0, 0]);
    }
}
