// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! D19 instrument — Monte Carlo of inter-shadow collisions with the REAL RS
//! block layout, to derive the per-shadow capacity the N-aware HUD bar shows.
//!
//! Model (each part mirrors the encoder):
//!   * A shadow's frame is `payload + SHADOW_FRAME_OVERHEAD_V1` bytes, RS-coded
//!     by `rs_encode_blocks_with_parity` in 255-byte blocks carrying
//!     `parity_len` parity bytes EACH, correcting `parity_len / 2` byte errors
//!     per block. Its bits go, in order, to its selected positions.
//!   * Positions are chosen by a per-passphrase hash priority, so each
//!     shadow's selection is a uniformly random ordered subset of the pool;
//!     a higher parity tier takes a longer PREFIX of the same order.
//!   * Shadows are stamped in order (the CLI and apps sort smallest first);
//!     a later shadow overwrites an earlier one, and an overwritten bit is
//!     wrong with probability 1/2 (independent random ciphertext bits). The
//!     primary never touches shadow positions (∞ cost).
//!   * The encoder tries tiers 4, 8, …, 128 and keeps the first at which EVERY
//!     shadow verifies.
//!
//! What it does NOT model: anything correlated in the priorities (they are
//! independent hashes) and cascades (the eligible domains are cascade-free by
//! construction). It is checked against the real-encode observations from the
//! D10 instruments before it is trusted.
//!
//! Run:
//!   cargo test --release --features h264-encoder --test d19_shadow_collision_mc \
//!     -- --ignored --nocapture --test-threads=1

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const TIERS: [usize; 6] = [4, 8, 16, 32, 64, 128];
const OVERHEAD: usize = 46; // SHADOW_FRAME_OVERHEAD_V1
const RS_N: usize = 255;

/// RS frame length in bytes for a `frame`-byte payload frame at `p` parity.
fn rs_len(frame: usize, p: usize) -> usize {
    let k = RS_N - p;
    let blocks = frame.div_ceil(k);
    frame + blocks * p
}

/// Byte index → RS block index (every block but the last is 255 bytes).
fn block_of(byte: usize) -> usize {
    byte / RS_N
}

/// One trial: `n` shadows of equal `payload` bytes in a pool of `pool`
/// positions. Returns whether SOME tier lets every shadow decode.
fn trial(pool: usize, n: usize, payload: usize, rng: &mut ChaCha8Rng, last: &mut [u8]) -> bool {
    let frame = payload + OVERHEAD;
    let max_bits = 8 * rs_len(frame, *TIERS.last().unwrap());
    if n * max_bits > pool {
        return false;
    }
    // Each shadow's priority order: max_bits distinct random positions.
    let mut taken = vec![false; pool];
    let orders: Vec<Vec<u32>> = (0..n)
        .map(|_| {
            for t in taken.iter_mut() {
                *t = false;
            }
            let mut v = Vec::with_capacity(max_bits);
            while v.len() < max_bits {
                let x = rng.gen_range(0..pool);
                if !taken[x] {
                    taken[x] = true;
                    v.push(x as u32);
                }
            }
            v
        })
        .collect();
    for &p in &TIERS {
        let bits = 8 * rs_len(frame, p);
        let blocks = rs_len(frame, p).div_ceil(RS_N);
        let t = p / 2;
        // Stamp in order; record the last writer per position.
        for s in 0..n {
            for &pos in &orders[s][..bits] {
                last[pos as usize] = s as u8;
            }
        }
        let mut ok = true;
        for s in 0..n {
            let mut errs = vec![0usize; blocks];
            let mut last_byte_marked = usize::MAX;
            for (i, &pos) in orders[s][..bits].iter().enumerate() {
                if last[pos as usize] != s as u8 && rng.gen_bool(0.5) {
                    let byte = i / 8;
                    if byte != last_byte_marked {
                        errs[block_of(byte)] += 1;
                        last_byte_marked = byte;
                    }
                }
            }
            if errs.iter().any(|&e| e > t) {
                ok = false;
                break;
            }
        }
        // Reset the positions this tier touched.
        for s in 0..n {
            for &pos in &orders[s][..bits] {
                last[pos as usize] = u8::MAX;
            }
        }
        if ok {
            return true;
        }
    }
    false
}

/// Success rate over `k` trials.
fn rate(pool: usize, n: usize, payload: usize, k: usize, seed: u64) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ (pool as u64) ^ ((n as u64) << 40) ^ ((payload as u64) << 20));
    let mut last = vec![u8::MAX; pool];
    let mut ok = 0;
    for _ in 0..k {
        if trial(pool, n, payload, &mut rng, &mut last) {
            ok += 1;
        }
    }
    ok as f64 / k as f64
}

#[test]
#[ignore = "instrument: Monte Carlo, ~minutes; run with --ignored --nocapture"]
fn d19_validate_against_real_encodes() {
    // Real-encode observations (synthetic 320x192x60 clip, pool 1 725 747 bits,
    // 2026-10-23): framed payload bytes -> outcome.
    let pool = 1_725_747;
    let obs: &[(usize, usize, bool)] = &[
        (2, 5080, true), (2, 5334, false), (2, 5588, false),
        (3, 3434, false), (3, 3293, false), (3, 3009, false),
        (3, 2655, true), (3, 2124, true), (3, 1593, true),
        (4, 2773, false), (4, 2658, false), (4, 2430, false),
        (4, 2144, true), (4, 1715, true), (4, 1286, true),
    ];
    for &(n, payload, ok) in obs {
        let r = rate(pool, n, payload, 200, 0xD19);
        eprintln!(
            "MC n={n} {payload:5} B: success {:5.1}%   observed {}",
            100.0 * r,
            if ok { "decode" } else { "refused" },
        );
    }
}
