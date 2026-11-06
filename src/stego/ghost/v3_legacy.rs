// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! **Decode-only, deprecated: Ghost v3 primary frames** (core 0.5 /
//! apps 1.9). Kept for backwards compatibility with images those versions
//! wrote; nothing encodes v3 any more. **Never extend this module and never
//! use it for new code** — the current Ghost generation is
//! [`super::v31`] (`docs/design/image/ghost-embedding-v3.md`, "v3.1").
//!
//! Everything the v3 decoder depends on is a frozen copy here (seeds, width
//! ladder, H-hat, bit planes, lane geometry, frame-length search), so a
//! change to the v3.1 code can never break it. The only shared pieces are
//! the keyed order's shuffle ([`crate::stego::permute::shuffle_u32`], pinned
//! by its own tests and shared with the v1 decoder) and the end-set H-hat
//! generator (`stc::hhat`, pinned likewise).
//!
//! v3 wire format (frozen): keyed order of every Y AC coefficient
//! (`SHA-256("phasm-ghost-v3/perm\0" ‖ K)`), lanes `clamp(min(⌊n/2²⁰⌋,
//! ⌊m/4096⌋), 1, 64)`, per lane layer 1 = `c mod 2` (first `c − ⌊c/2⌋` bits),
//! layer 2 = `⌊c/2⌋ mod 2` (last `⌊c/2⌋` bits), each a binary STC of height 9
//! and constant width = the widest value of the ladder 1..16, ×1.2 with
//! `bits · w ≤ n_k` over the lane's first `bits · w` positions, end-set
//! H-hat seeded by `SHA-256("phasm-ghost-v3/hhat1" ‖ K)` / `…hhat2`.
//!
//! The v3 *shadow* layout is unchanged in v3.1 and is read by the current
//! shadow decoder (`shadow.rs`).

#[cfg(test)]
use crate::codec::jpeg::JpegImage;
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload::PayloadData;
use crate::stego::stc::hhat;
use sha2::{Digest, Sha256};

/// STC constraint height of v3 (frozen).
const V3_STC_H: usize = 9;
/// Lane geometry (frozen).
const LANE_POSITIONS: usize = 1 << 20;
const LANE_MIN_BITS: usize = 1 << 12;
const MAX_LANES: usize = 64;
/// Smallest frame the v3 encoder wrote (bits).
const MIN_FRAME_BITS: usize = (frame::FRAME_OVERHEAD + 1) * 8;

fn label_hash(label: &[u8], key: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(label);
    h.update(key);
    h.finalize().into()
}

/// The v3 order seed (`SHA-256("phasm-ghost-v3/perm\0" ‖ K)`). v3.1 keeps
/// this order on purpose (one shuffle per decode reads both generations);
/// this copy is the v3 decoder's own.
pub fn perm_seed(structural_key: &[u8; 64]) -> [u8; 32] {
    label_hash(b"phasm-ghost-v3/perm\0", structural_key)
}

fn hhat_seeds(structural_key: &[u8; 64]) -> ([u8; 32], [u8; 32]) {
    (label_hash(b"phasm-ghost-v3/hhat1", structural_key), label_hash(b"phasm-ghost-v3/hhat2", structural_key))
}

/// v3's width ladder (frozen): 1..=16, then `w + ⌊w/5⌋`.
fn ladder() -> impl Iterator<Item = usize> {
    let mut w = 1usize;
    std::iter::from_fn(move || {
        let cur = w;
        w = if w < 16 { w + 1 } else { w + w / 5 };
        Some(cur)
    })
}

fn choose_w(n: usize, m: usize) -> Option<usize> {
    if m == 0 || m > n {
        return None;
    }
    ladder().take_while(|&w| m.saturating_mul(w) <= n).last()
}

fn lanes(n: usize, m: usize) -> usize {
    (n / LANE_POSITIONS).min(m / LANE_MIN_BITS).clamp(1, MAX_LANES)
}

fn share(total: usize, l: usize, k: usize) -> std::ops::Range<usize> {
    let a = (k as u64 * total as u64 / l as u64) as usize;
    let b = ((k as u64 + 1) * total as u64 / l as u64) as usize;
    a..b
}

fn layer_split(c: usize) -> (usize, usize) {
    (c - c / 2, c / 2)
}

fn packed_cols(w: usize, seed: &[u8; 32]) -> Vec<u32> {
    let hh = hhat::generate_hhat_end_set(V3_STC_H, w, seed);
    (0..w).map(|c| hhat::column_packed(&hh, c)).collect()
}

/// The packed H-hat columns of v3 layer 1 / 2 at width `w` (the RNG golden
/// test pins them).
#[cfg(test)]
pub(crate) fn layer_cols_for_test(structural_key: &[u8; 64], layer: u8, w: usize) -> Vec<u32> {
    let (s1, s2) = hhat_seeds(structural_key);
    packed_cols(w, if layer == 1 { &s1 } else { &s2 })
}

/// v3 bit plane 2 of a coefficient value: `⌊c/2⌋ mod 2`.
#[inline]
pub fn plane2(c: i16) -> u8 {
    ((c >> 1) & 1) as u8
}

fn bits_to_usize(bits: &[u8]) -> usize {
    bits.iter().fold(0usize, |acc, &b| (acc << 1) | (b & 1) as usize)
}

fn frame_bits_from_prefix(fb: &[u8]) -> Option<usize> {
    let v1 = bits_to_usize(&fb[..16]);
    if v1 > 0 {
        return Some((frame::FRAME_OVERHEAD + v1) * 8);
    }
    let v2 = bits_to_usize(&fb[16..48]);
    if v2 > u16::MAX as usize {
        return Some((frame::FRAME_OVERHEAD_EXT + v2) * 8);
    }
    None
}

/// The first `k` syndrome bits of the constant-width STC over the
/// bit-packed positions `[start, start + len)` of `words`.
fn extract_bits(words: &[u64], start: usize, len: usize, cols: &[u32], k: usize) -> Vec<u8> {
    let w = cols.len();
    if w == 0 || k == 0 {
        return Vec::new();
    }
    let max_i = k.min(len.div_ceil(w));
    let mut out = Vec::with_capacity(max_i);
    let mut state = 0u32;
    for i in 0..max_i {
        let a = start + i * w;
        let b = (a + w).min(start + len);
        for (c, j) in (a..b).enumerate() {
            if (words[j >> 6] >> (j & 63)) & 1 != 0 {
                state ^= cols[c];
            }
        }
        out.push((state & 1) as u8);
        state >>= 1;
    }
    out
}

fn lane_m_range(n: usize, l: usize) -> Option<(usize, usize)> {
    if n < MIN_FRAME_BITS {
        return None;
    }
    let first_at_least = |target: usize| {
        let (mut lo, mut hi) = (MIN_FRAME_BITS, n + 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if lanes(n, mid) >= target { hi = mid } else { lo = mid + 1 }
        }
        lo
    };
    let lo = first_at_least(l);
    let hi = first_at_least(l + 1) - 1;
    (lo <= hi && lo <= n && lanes(n, lo) == l).then_some((lo, hi.min(n)))
}

/// Decode a v3 primary from its two bit planes along the v3 order
/// (`p1` = `c mod 2`, `p2` = [`plane2`], bit-packed, `n` positions). `None`
/// = no v3 frame for this passphrase; `Some(Err(DecryptionFailed))` = a frame
/// with a valid CRC whose ciphertext did not authenticate.
pub fn decode_planes(
    p1: &[u64],
    p2: &[u64],
    n: usize,
    passphrase: &str,
    structural_key: &[u8; 64],
    parse_and_decrypt: &dyn Fn(&[u8], &str) -> Result<PayloadData, StegoError>,
) -> Option<Result<PayloadData, StegoError>> {
    if n == 0 {
        return None;
    }
    let (s1, s2) = hhat_seeds(structural_key);
    let mut saw_decrypt_fail = false;
    let l_max = (n / LANE_POSITIONS).clamp(1, MAX_LANES);
    for l in 1..=l_max {
        let Some((m_lo, m_hi)) = lane_m_range(n, l) else { continue };
        let r0 = share(n, l, 0);
        let n0 = r0.len();
        let m1_of = |m: usize| layer_split(share(m, l, 0).len()).0;
        let (Some(w_max), w_min) = (choose_w(n0, m1_of(m_lo)), choose_w(n0, m1_of(m_hi)).unwrap_or(1)) else {
            continue;
        };
        for w in ladder().skip_while(|&w| w < w_min).take_while(|&w| w <= w_max) {
            if n0 / w < 16 {
                break;
            }
            let lim = (n0 / w) * w;
            let pre = extract_bits(p1, r0.start, lim, &packed_cols(w, &s1), 48.min(n0 / w));
            let m = if pre.len() >= 48 {
                match frame_bits_from_prefix(&pre) {
                    Some(m) => m,
                    None => continue,
                }
            } else {
                let v1 = bits_to_usize(&pre[..16]);
                if v1 == 0 {
                    continue;
                }
                (frame::FRAME_OVERHEAD + v1) * 8
            };
            if m > n || lanes(n, m) != l {
                continue;
            }
            let (m1_0, _) = layer_split(share(m, l, 0).len());
            if choose_w(n0, m1_0) != Some(w) {
                continue;
            }
            let mut fbits: Vec<u8> = Vec::with_capacity(m);
            let mut ok = true;
            for k in 0..l {
                let rk = share(n, l, k);
                let nk = rk.len();
                let (m1, m2) = layer_split(share(m, l, k).len());
                let w1 = choose_w(nk, m1);
                let w2 = if m2 > 0 { choose_w(nk, m2) } else { Some(0) };
                let (Some(w1), Some(w2)) = (w1, w2) else {
                    ok = false;
                    break;
                };
                fbits.extend(extract_bits(p1, rk.start, m1 * w1, &packed_cols(w1, &s1), m1));
                if m2 > 0 {
                    fbits.extend(extract_bits(p2, rk.start, m2 * w2, &packed_cols(w2, &s2), m2));
                }
            }
            if !ok {
                continue;
            }
            match parse_and_decrypt(&frame::bits_to_bytes(&fbits), passphrase) {
                Ok(p) => return Some(Ok(p)),
                Err(StegoError::DecryptionFailed) => saw_decrypt_fail = true,
                Err(_) => {}
            }
        }
    }
    if saw_decrypt_fail { Some(Err(StegoError::DecryptionFailed)) } else { None }
}

/// [`decode_planes`] from the image (its own order and planes). The Ghost
/// decoder normally reads both generations from one shuffle
/// (`v31::decode_all`); this is for tests and tools.
#[cfg(test)]
pub fn decode(
    img: &JpegImage,
    passphrase: &str,
    structural_key: &[u8; 64],
    parse_and_decrypt: &dyn Fn(&[u8], &str) -> Result<PayloadData, StegoError>,
) -> Option<Result<PayloadData, StegoError>> {
    if img.num_components() == 0 {
        return None;
    }
    let grid = img.dct_grid(0);
    let nb = grid.blocks_wide() * grid.blocks_tall();
    let mut ord: Vec<u32> = (0..nb as u32).flat_map(|b| (1..64u32).map(move |k| b * 64 + k)).collect();
    crate::stego::permute::shuffle_u32(&mut ord, &perm_seed(structural_key));
    let n = ord.len();
    let mut p1 = vec![0u64; n.div_ceil(64)];
    let mut p2 = vec![0u64; n.div_ceil(64)];
    for (i, &fi) in ord.iter().enumerate() {
        let c = grid.coeffs()[fi as usize];
        p1[i >> 6] |= ((c & 1) as u64) << (i & 63);
        p2[i >> 6] |= (plane2(c) as u64) << (i & 63);
    }
    drop(ord);
    decode_planes(&p1, &p2, n, passphrase, structural_key, parse_and_decrypt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frozen decoder reads the core 0.5.1 fixture on its own (and not
    /// a v3.1 image of the same cover, message and passphrase).
    #[test]
    fn legacy_decoder_reads_the_v3_fixture_only() {
        let pass = "wire-format-cross-platform-test";
        let key = crate::stego::crypto::derive_structural_key(pass).unwrap();
        let read = |name: &str| {
            let b = std::fs::read(format!("test-vectors/image/wire-format/{name}")).unwrap();
            let img = JpegImage::from_bytes(&b).unwrap();
            decode(&img, pass, &key, &super::super::pipeline::try_parse_and_decrypt)
        };
        let got = read("ghost_v3_2026_11_03.jpg").expect("a v3 frame").unwrap();
        assert_eq!(got.text, "Hello from the wire-format gate!");
        assert!(read("ghost_v31_2026_11_03.jpg").is_none());
    }
}
