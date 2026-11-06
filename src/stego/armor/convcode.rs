// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Terminated convolutional code with soft-decision Viterbi decoding
//! (prototype for Fortress v2, 2026-10-31).
//!
//! Constraint length K = 7 (64 states), generators in octal, the newest
//! input bit is the most significant register bit. The encoder appends
//! K - 1 zero tail bits. Soft inputs: positive favours coded bit 0
//! (the sign convention of every Armor LLR). Pure integer/f64 arithmetic,
//! deterministic on all platforms.

/// K = 7, rate 1/2: the CCSDS / NASA code (133, 171), free distance 10.
pub const K7_R2: (usize, &[u32]) = (7, &[0o133, 0o171]);
/// K = 7, rate 1/4 (135, 135, 147, 163), free distance 20.
pub const K7_R4: (usize, &[u32]) = (7, &[0o135, 0o135, 0o147, 0o163]);

/// Number of coded bits for `info_bits` input bits.
pub fn coded_len(code: (usize, &[u32]), info_bits: usize) -> usize {
    (info_bits + code.0 - 1) * code.1.len()
}

#[inline]
fn parity(x: u32) -> u8 {
    (x.count_ones() & 1) as u8
}

/// Encode `bits` (0/1), terminated with K - 1 zero bits.
pub fn encode(code: (usize, &[u32]), bits: &[u8]) -> Vec<u8> {
    let (k, gens) = code;
    let mut out = Vec::with_capacity(coded_len(code, bits.len()));
    let mut state: u32 = 0; // K - 1 previous bits, newest at bit K - 2
    for &b in bits.iter().chain(std::iter::repeat_n(&0u8, k - 1)) {
        let reg = ((b as u32) << (k - 1)) | state;
        for &g in gens {
            out.push(parity(reg & g));
        }
        state = reg >> 1;
    }
    out
}

/// Soft Viterbi: `soft` holds `coded_len(code, info_bits)` values (+ = 0).
/// Returns the `info_bits` most likely input bits.
pub fn decode(code: (usize, &[u32]), soft: &[f64], info_bits: usize) -> Vec<u8> {
    let (k, gens) = code;
    let n = gens.len();
    let s_count = 1usize << (k - 1);
    let steps = info_bits + k - 1;
    assert!(soft.len() >= steps * n, "convcode: short soft input");
    // Branch outputs: for state s and input b, sign of each coded bit.
    let mut sign = vec![0f64; s_count * 2 * n];
    for s in 0..s_count {
        for b in 0..2 {
            let reg = ((b as u32) << (k - 1)) | s as u32;
            for (j, &g) in gens.iter().enumerate() {
                sign[(s * 2 + b) * n + j] = if parity(reg & g) == 0 { 1.0 } else { -1.0 };
            }
        }
    }
    let mut pm = vec![f64::NEG_INFINITY; s_count];
    pm[0] = 0.0;
    let mut next = vec![0f64; s_count];
    // One bit per (step, state): which predecessor won.
    let mut choice = vec![0u64; steps * s_count.div_ceil(64)];
    let words = s_count.div_ceil(64);
    for t in 0..steps {
        let y = &soft[t * n..t * n + n];
        for d in 0..s_count {
            let b = d >> (k - 2);
            let p0 = (d << 1) & (s_count - 1);
            let p1 = p0 | 1;
            let bm = |p: usize| -> f64 {
                let base = (p * 2 + b) * n;
                (0..n).map(|j| sign[base + j] * y[j]).sum()
            };
            let c0 = pm[p0] + bm(p0);
            let c1 = pm[p1] + bm(p1);
            let take1 = c1 > c0;
            next[d] = if take1 { c1 } else { c0 };
            if t >= info_bits && b == 1 {
                next[d] = f64::NEG_INFINITY; // tail: zero inputs only
            }
            if take1 {
                choice[t * words + d / 64] |= 1u64 << (d % 64);
            }
        }
        std::mem::swap(&mut pm, &mut next);
    }
    let mut bits = vec![0u8; steps];
    let mut s = 0usize;
    for t in (0..steps).rev() {
        bits[t] = (s >> (k - 2)) as u8;
        let c = (choice[t * words + s / 64] >> (s % 64)) & 1;
        s = ((s << 1) & (s_count - 1)) | c as usize;
    }
    bits.truncate(info_bits);
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_noiseless_and_noisy() {
        let mut x = 0x9e37_79b9u32;
        let bits: Vec<u8> = (0..300).map(|_| { x ^= x << 13; x ^= x >> 17; x ^= x << 5; (x & 1) as u8 }).collect();
        for code in [K7_R2, K7_R4] {
            let c = encode(code, &bits);
            assert_eq!(c.len(), coded_len(code, bits.len()));
            let soft: Vec<f64> = c.iter().map(|&b| if b == 0 { 1.0 } else { -1.0 }).collect();
            assert_eq!(decode(code, &soft, bits.len()), bits);
            // flip every 9th coded bit: well inside the correction power
            let noisy: Vec<f64> = soft.iter().enumerate().map(|(i, &v)| if i % 9 == 4 { -0.6 * v } else { v }).collect();
            assert_eq!(decode(code, &noisy, bits.len()), bits);
        }
    }
}
