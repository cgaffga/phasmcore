// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! `rand_chacha::ChaCha20Rng`'s output stream with a WASM SIMD128 block
//! function.
//!
//! `rand_chacha` computes ChaCha through `ppv_lite86`, which has no
//! wasm32 SIMD backend: on the website the keyed Fisher–Yates shuffle of a
//! 12 MP cover (12 M draws) spent more than half its time generating
//! keystream. [`ChaCha20Stream`] produces exactly the same `u32` sequence
//! (djb ChaCha20, 64-bit block counter from 0, stream 0, four blocks per
//! refill, words in order), computing the four blocks lane-parallel with
//! SIMD128 on wasm32 and with the scalar block function elsewhere.
//! `tests::matches_rand_chacha` pins the scalar path to `ChaCha20Rng`;
//! `chacha_stream_test_hash_hex` pins the WASM bundle to native.

use rand::RngCore;

const CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];
const BUF_WORDS: usize = 64;

/// `ChaCha20Rng::from_seed(seed)`'s stream (see the module docs). Only the
/// `RngCore` methods are provided; `gen_range` etc. come from `rand::Rng`.
pub struct ChaCha20Stream {
    key: [u32; 8],
    counter: u64,
    buf: [u32; BUF_WORDS],
    idx: usize,
}

impl ChaCha20Stream {
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let mut key = [0u32; 8];
        for (k, c) in key.iter_mut().zip(seed.chunks_exact(4)) {
            *k = u32::from_le_bytes(c.try_into().unwrap());
        }
        Self { key, counter: 0, buf: [0; BUF_WORDS], idx: BUF_WORDS }
    }

    /// Next four blocks (counters `counter .. counter + 4`) into `buf`.
    fn refill(&mut self) {
        blocks4(&self.key, self.counter, &mut self.buf);
        self.counter = self.counter.wrapping_add(4);
    }
}

impl RngCore for ChaCha20Stream {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.idx >= BUF_WORDS {
            self.refill();
            self.idx = 0;
        }
        let v = self.buf[self.idx];
        self.idx += 1;
        v
    }

    /// As `rand_core::block::BlockRng::next_u64`.
    fn next_u64(&mut self) -> u64 {
        let i = self.idx;
        if i < BUF_WORDS - 1 {
            self.idx += 2;
            (u64::from(self.buf[i + 1]) << 32) | u64::from(self.buf[i])
        } else if i >= BUF_WORDS {
            self.refill();
            self.idx = 2;
            (u64::from(self.buf[1]) << 32) | u64::from(self.buf[0])
        } else {
            let x = u64::from(self.buf[BUF_WORDS - 1]);
            self.refill();
            self.idx = 1;
            (u64::from(self.buf[0]) << 32) | x
        }
    }

    /// Whole words, little-endian (as `rand_core`'s `fill_via_u32_chunks`).
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(4) {
            let w = self.next_u32().to_le_bytes();
            chunk.copy_from_slice(&w[..chunk.len()]);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

#[inline(always)]
fn quarter(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(16);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(12);
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(8);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(7);
}

fn initial_state(key: &[u32; 8], counter: u64) -> [u32; 16] {
    let mut s = [0u32; 16];
    s[..4].copy_from_slice(&CONSTANTS);
    s[4..12].copy_from_slice(key);
    s[12] = counter as u32;
    s[13] = (counter >> 32) as u32;
    s
}

/// One ChaCha20 block (scalar reference).
fn block(key: &[u32; 8], counter: u64, out: &mut [u32]) {
    let s = initial_state(key, counter);
    let mut x = s;
    for _ in 0..10 {
        quarter(&mut x, 0, 4, 8, 12);
        quarter(&mut x, 1, 5, 9, 13);
        quarter(&mut x, 2, 6, 10, 14);
        quarter(&mut x, 3, 7, 11, 15);
        quarter(&mut x, 0, 5, 10, 15);
        quarter(&mut x, 1, 6, 11, 12);
        quarter(&mut x, 2, 7, 8, 13);
        quarter(&mut x, 3, 4, 9, 14);
    }
    for i in 0..16 {
        out[i] = x[i].wrapping_add(s[i]);
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn blocks4(key: &[u32; 8], counter: u64, out: &mut [u32; BUF_WORDS]) {
    for b in 0..4 {
        block(key, counter.wrapping_add(b as u64), &mut out[b * 16..(b + 1) * 16]);
    }
}

/// Four blocks at once, one per SIMD lane (lane `b` = counter + `b`).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn blocks4(key: &[u32; 8], counter: u64, out: &mut [u32; BUF_WORDS]) {
    use core::arch::wasm32::*;
    #[inline(always)]
    fn rotl(v: v128, n: u32) -> v128 {
        v128_or(i32x4_shl(v, n), u32x4_shr(v, 32 - n))
    }
    #[inline(always)]
    fn rotl16(v: v128) -> v128 {
        i8x16_shuffle::<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>(v, v)
    }
    #[inline(always)]
    fn rotl8(v: v128) -> v128 {
        i8x16_shuffle::<3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14>(v, v)
    }
    #[inline(always)]
    fn qr(x: &mut [v128; 16], a: usize, b: usize, c: usize, d: usize) {
        x[a] = i32x4_add(x[a], x[b]);
        x[d] = rotl16(v128_xor(x[d], x[a]));
        x[c] = i32x4_add(x[c], x[d]);
        x[b] = rotl(v128_xor(x[b], x[c]), 12);
        x[a] = i32x4_add(x[a], x[b]);
        x[d] = rotl8(v128_xor(x[d], x[a]));
        x[c] = i32x4_add(x[c], x[d]);
        x[b] = rotl(v128_xor(x[b], x[c]), 7);
    }
    let base = initial_state(key, counter);
    let mut s = [u32x4_splat(0); 16];
    for i in 0..16 {
        s[i] = u32x4_splat(base[i]);
    }
    let c = |b: u64| counter.wrapping_add(b);
    s[12] = u32x4(c(0) as u32, c(1) as u32, c(2) as u32, c(3) as u32);
    s[13] = u32x4((c(0) >> 32) as u32, (c(1) >> 32) as u32, (c(2) >> 32) as u32, (c(3) >> 32) as u32);
    let mut x = s;
    for _ in 0..10 {
        qr(&mut x, 0, 4, 8, 12);
        qr(&mut x, 1, 5, 9, 13);
        qr(&mut x, 2, 6, 10, 14);
        qr(&mut x, 3, 7, 11, 15);
        qr(&mut x, 0, 5, 10, 15);
        qr(&mut x, 1, 6, 11, 12);
        qr(&mut x, 2, 7, 8, 13);
        qr(&mut x, 3, 4, 9, 14);
    }
    for i in 0..16 {
        let v = i32x4_add(x[i], s[i]);
        out[i] = u32x4_extract_lane::<0>(v);
        out[16 + i] = u32x4_extract_lane::<1>(v);
        out[32 + i] = u32x4_extract_lane::<2>(v);
        out[48 + i] = u32x4_extract_lane::<3>(v);
    }
}

/// SHA-256 of 100 000 `next_u32` and 1 000 `gen_range` draws (mixed with
/// `next_u64` across buffer edges) of [`ChaCha20Stream`] — pinned in the
/// tests to `ChaCha20Rng`'s value; the WASM bundle exposes it so Node can
/// check the SIMD128 block function.
#[doc(hidden)]
pub fn chacha_stream_test_hash_hex() -> String {
    use rand::Rng;
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    let mut r = ChaCha20Stream::from_seed(*b"phasm chacha stream test seed 01");
    for _ in 0..100_000 {
        h.update(r.next_u32().to_le_bytes());
    }
    for i in 0..1000u32 {
        h.update(r.gen_range(0..=i * 7919 + 1).to_le_bytes());
        if i % 7 == 3 {
            h.update(r.next_u64().to_le_bytes());
        }
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn matches_rand_chacha() {
        for seed in [[0u8; 32], [7u8; 32], *b"phasm chacha stream test seed 01"] {
            let mut a = ChaCha20Stream::from_seed(seed);
            let mut b = rand_chacha::ChaCha20Rng::from_seed(seed);
            for i in 0..20_000u32 {
                match i % 11 {
                    3 => assert_eq!(a.next_u64(), b.next_u64(), "u64 at {i}"),
                    5 => assert_eq!(a.gen_range(0..=i), b.gen_range(0..=i), "range at {i}"),
                    _ => assert_eq!(a.next_u32(), b.next_u32(), "u32 at {i}"),
                }
            }
        }
        // The same hash as ChaCha20Rng's.
        let mut h = <sha2::Sha256 as sha2::Digest>::new();
        let mut r = rand_chacha::ChaCha20Rng::from_seed(*b"phasm chacha stream test seed 01");
        for _ in 0..100_000 {
            sha2::Digest::update(&mut h, r.next_u32().to_le_bytes());
        }
        for i in 0..1000u32 {
            sha2::Digest::update(&mut h, r.gen_range(0..=i * 7919 + 1).to_le_bytes());
            if i % 7 == 3 {
                sha2::Digest::update(&mut h, r.next_u64().to_le_bytes());
            }
        }
        let want: String = sha2::Digest::finalize(h).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(chacha_stream_test_hash_hex(), want);
        // What the WASM bundle's `__test_chacha_hash` must return.
        assert_eq!(want, "aa197740aecc5e8f0304f9be3d554e7a09694095c033a94b2288df31698003f8");
    }
}
