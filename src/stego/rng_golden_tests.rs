// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Golden streams for every keyed layout and every random sampler phasm uses
//! (fix list C15).
//!
//! The expected values were recorded with `rand` 0.8.8 + `rand_chacha` 0.3.1,
//! the versions every released image and video was made with. **Never edit an
//! expected value.** A mismatch means a keyed layout changed, and every
//! released message whose layout depends on it no longer decodes. The fix is
//! in the code (the sampler or the stream), not here.
//!
//! Two kinds of goldens:
//! - samplers: SHA-256 over many draws of each pattern the code uses (integer
//!   ranges over `u32` / `i32` / `u8` / `usize`, inclusive and exclusive,
//!   small, large, non-power-of-two and full; `f64` ranges; `gen::<T>()`;
//!   `gen_bool`), plus the raw ChaCha20 / ChaCha8 stream (`next_u32`,
//!   `next_u64` at odd word alignment, `fill_bytes` of odd lengths,
//!   `set_word_pos`, `set_stream`, `seed_from_u64`);
//! - layouts: SHA-256 of the real keyed orders built from fixed passphrases
//!   and a real JPEG (Ghost v3 order and H-hat columns, shadow orders, Armor
//!   STDM positions and spreading vectors, Fortress block permutation,
//!   Fortress T/TL tile key, video GOP cover order, STC H-hat).

use crate::stego::rng_compat::WordRng as _;
use rand::SeedableRng;
use rand_chacha::{ChaCha20Rng, ChaCha8Rng};
use sha2::{Digest, Sha256};

/// The sampler calls under test, one function per pattern used in the code.
mod s {
    use crate::stego::rng_compat::Rand08;
    use rand_chacha::ChaCha20Rng;

    pub fn u32_incl(r: &mut ChaCha20Rng, lo: u32, hi: u32) -> u32 {
        r.gen_range08(lo..=hi)
    }
    pub fn u32_excl(r: &mut ChaCha20Rng, lo: u32, hi: u32) -> u32 {
        r.gen_range08(lo..hi)
    }
    pub fn i32_incl(r: &mut ChaCha20Rng, lo: i32, hi: i32) -> i32 {
        r.gen_range08(lo..=hi)
    }
    pub fn u8_excl(r: &mut ChaCha20Rng, lo: u8, hi: u8) -> u8 {
        r.gen_range08(lo..hi)
    }
    pub fn usize_excl(r: &mut ChaCha20Rng, lo: usize, hi: usize) -> usize {
        r.gen_range08(lo..hi)
    }
    pub fn f64_excl(r: &mut ChaCha20Rng, lo: f64, hi: f64) -> f64 {
        r.gen_range08(lo..hi)
    }
    pub fn gen_u32(r: &mut ChaCha20Rng) -> u32 {
        r.gen08::<u32>()
    }
    pub fn gen_u8(r: &mut ChaCha20Rng) -> u8 {
        r.gen08::<u8>()
    }
    pub fn gen_usize(r: &mut ChaCha20Rng) -> usize {
        r.gen08::<usize>()
    }
    pub fn gen_f32(r: &mut ChaCha20Rng) -> f32 {
        r.gen08::<f32>()
    }
    pub fn gen_f64(r: &mut ChaCha20Rng) -> f64 {
        r.gen08::<f64>()
    }
    pub fn gen_bool(r: &mut ChaCha20Rng, p: f64) -> bool {
        r.gen_bool08(p)
    }
}

struct H(Sha256);

impl H {
    fn new() -> Self {
        H(Sha256::new())
    }
    fn b(&mut self, x: &[u8]) {
        self.0.update(x);
    }
    fn u32(&mut self, x: u32) {
        self.0.update(x.to_le_bytes());
    }
    fn u64(&mut self, x: u64) {
        self.0.update(x.to_le_bytes());
    }
    fn hex(self) -> String {
        self.0.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }
}

fn seed(i: u8) -> [u8; 32] {
    let mut s = [0u8; 32];
    for (k, b) in s.iter_mut().enumerate() {
        *b = (k as u8).wrapping_mul(37).wrapping_add(i.wrapping_mul(101)).wrapping_add(7);
    }
    s
}

fn rng(i: u8) -> ChaCha20Rng {
    ChaCha20Rng::from_seed(seed(i))
}

/// Compares every `(name, got)` with the table and reports all mismatches
/// (and every value, so a new golden can be read off a failing run).
fn check(got: &[(&str, String)], want: &[(&str, &str)]) {
    let mut bad = Vec::new();
    for (name, g) in got {
        let w = want.iter().find(|(n, _)| n == name).map(|(_, w)| *w);
        if w != Some(g.as_str()) {
            bad.push(format!("(\"{name}\", \"{g}\"), // want {w:?}"));
        }
    }
    assert!(bad.is_empty(), "golden mismatches:\n{}", bad.join("\n"));
    assert_eq!(got.len(), want.len(), "golden table and computed list differ in length");
}

// ---------------------------------------------------------------------------
// Raw stream
// ---------------------------------------------------------------------------

fn raw_streams() -> Vec<(&'static str, String)> {
    let mut out = Vec::new();

    let mut h = H::new();
    let mut first = Vec::new();
    for i in 0..3u8 {
        let mut r = rng(i);
        for k in 0..5000 {
            let x = r.next_u32();
            if i == 0 && k < 4 {
                first.push(x.to_string());
            }
            h.u32(x);
        }
    }
    out.push(("chacha20_next_u32_first", first.join(",")));
    out.push(("chacha20_next_u32", h.hex()));

    // next_u64 at odd word alignment and across the 64-word buffer edge.
    let mut h = H::new();
    for i in 0..3u8 {
        let mut r = rng(i);
        for k in 0..3000 {
            if k % 3 == 0 {
                h.u32(r.next_u32());
            } else {
                h.u64(r.next_u64());
            }
        }
    }
    out.push(("chacha20_mixed_u32_u64", h.hex()));

    // fill_bytes of lengths that are not multiples of 4.
    let mut h = H::new();
    let mut r = rng(3);
    for i in 0..400usize {
        let mut buf = vec![0u8; (i * 7 + 1) % 41];
        r.fill_bytes(&mut buf);
        h.b(&buf);
        h.u32(r.next_u32());
        if i % 5 == 0 {
            h.u64(r.next_u64());
        }
    }
    let mut buf = [0u8; 512];
    for _ in 0..8 {
        r.fill_bytes(&mut buf);
        h.b(&buf);
    }
    out.push(("chacha20_fill_bytes", h.hex()));

    // set_word_pos (as the shadow / spreading / video code seeks), set_stream.
    let mut h = H::new();
    let mut r = rng(4);
    for p in [0u128, 1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 1000, 1001, 123_456_789, (1u128 << 36) + 3] {
        r.set_word_pos(p);
        h.u32(r.next_u32());
        r.set_word_pos(p);
        h.u64(r.next_u64());
        r.set_word_pos(p);
        let mut b = [0u8; 512];
        r.fill_bytes(&mut b);
        h.b(&b);
    }
    let mut r = rng(5);
    r.set_stream(1);
    let mut b = [0u8; 512];
    r.fill_bytes(&mut b);
    h.b(&b);
    h.u64(r.next_u64());
    out.push(("chacha20_seek_stream", h.hex()));

    // seed_from_u64 (ChaCha20 in tests; ChaCha8 for PHASM_DETERMINISTIC_SEED's
    // salt + nonce in crypto::encrypt).
    let mut h = H::new();
    for s in [0u64, 1, 42, 0xDEAD_BEEF, u64::MAX] {
        let mut r = ChaCha20Rng::seed_from_u64(s);
        h.u32(r.next_u32());
        h.u64(r.next_u64());
        let mut r8 = ChaCha8Rng::seed_from_u64(s);
        let mut salt = [0u8; crate::stego::crypto::SALT_LEN];
        let mut nonce = [0u8; crate::stego::crypto::NONCE_LEN];
        r8.fill_bytes(&mut salt);
        r8.fill_bytes(&mut nonce);
        h.b(&salt);
        h.b(&nonce);
        h.u32(r8.next_u32());
    }
    out.push(("seed_from_u64_chacha20_chacha8", h.hex()));
    out
}

// ---------------------------------------------------------------------------
// Samplers
// ---------------------------------------------------------------------------

fn samplers() -> Vec<(&'static str, String)> {
    let mut out = Vec::new();

    // Fisher-Yates `gen_range(0..=i as u32)`: permute, cover_order, fortress,
    // fortress_tile.
    let mut h = H::new();
    let mut first = Vec::new();
    let mut r = rng(10);
    for i in (1..=20_000u32).rev() {
        let j = s::u32_incl(&mut r, 0, i);
        if i > 20_000 - 6 {
            first.push(j.to_string());
        }
        h.u32(j);
    }
    for rep in 0..500u32 {
        for i in (1..=19u32).rev() {
            h.u32(s::u32_incl(&mut r, 0, i + rep % 3));
        }
    }
    out.push(("u32_incl_fy_first", first.join(",")));
    out.push(("u32_incl_fy", h.hex()));

    // Large and awkward bounds: the rejection zone matters here.
    let mut h = H::new();
    let mut r = rng(11);
    for i in 0..5000u32 {
        h.u32(s::u32_incl(&mut r, 0, 2_100_000_000u32.wrapping_add(i.wrapping_mul(214_033)) | 1));
        h.u32(s::u32_incl(&mut r, 0, 3_000_000_000 + i * 99_991));
        h.u32(s::u32_incl(&mut r, 0, (1u32 << 31) + i));
        h.u32(s::u32_incl(&mut r, 0, (1u32 << 31) - 1 - i));
        h.u32(s::u32_incl(&mut r, 0, u32::MAX - 1 - i));
        h.u32(s::u32_incl(&mut r, 0, i * 7919 + 1));
        h.u32(s::u32_incl(&mut r, i, i + 12_345_678));
    }
    for _ in 0..100 {
        h.u32(s::u32_incl(&mut r, 0, u32::MAX)); // full range
        h.u32(s::u32_incl(&mut r, 5, 5)); // one value
    }
    out.push(("u32_incl_large", h.hex()));

    // `gen_range(i as u32..len as u32)` (fortress_tile header cells) and other
    // exclusive ranges.
    let mut h = H::new();
    let mut r = rng(12);
    for i in 0..9u32 {
        h.u32(s::u32_excl(&mut r, i, 64));
    }
    for i in 0..5000u32 {
        h.u32(s::u32_excl(&mut r, i, 10_001 + i));
        h.u32(s::u32_excl(&mut r, 0, 3 + i % 61));
        h.u32(s::u32_excl(&mut r, i, u32::MAX));
    }
    out.push(("u32_excl", h.hex()));

    // Small integer ranges of the tests (i32 / u8 / usize).
    let mut h = H::new();
    let mut r = rng(13);
    for _ in 0..5000 {
        h.u32(s::i32_incl(&mut r, -100, 100) as u32);
        h.u32(s::i32_incl(&mut r, -4, 4) as u32);
        h.u32(s::i32_incl(&mut r, 0, 255) as u32);
        h.b(&[s::u8_excl(&mut r, 0, 26)]);
        h.u64(s::usize_excl(&mut r, 0, 64) as u64);
        h.u64(s::usize_excl(&mut r, 0, 1_234_567) as u64);
    }
    out.push(("small_int_ranges", h.hex()));

    // f64 ranges: spreading (-1..1), and (0..1).
    let mut h = H::new();
    let mut first = Vec::new();
    let mut r = rng(14);
    for k in 0..10_000 {
        let a = s::f64_excl(&mut r, -1.0, 1.0);
        if k < 3 {
            first.push(format!("{:016x}", a.to_bits()));
        }
        h.u64(a.to_bits());
        h.u64(s::f64_excl(&mut r, 0.0, 1.0).to_bits());
    }
    out.push(("f64_range_first", first.join(",")));
    out.push(("f64_range", h.hex()));

    // `gen::<T>()` and `gen_bool`.
    let mut h = H::new();
    let mut r = rng(15);
    for _ in 0..5000 {
        h.u32(s::gen_u32(&mut r));
        h.b(&[s::gen_u8(&mut r)]);
        h.u64(s::gen_usize(&mut r) as u64);
        h.u32(s::gen_f32(&mut r).to_bits());
        h.u64(s::gen_f64(&mut r).to_bits());
        h.b(&[s::gen_bool(&mut r, 0.1) as u8, s::gen_bool(&mut r, 0.5) as u8]);
    }
    out.push(("gen_and_bool", h.hex()));
    out
}

// ---------------------------------------------------------------------------
// Keyed layouts
// ---------------------------------------------------------------------------

const PHOTO: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/test-vectors/image/photo_320x240_q75_420.jpg"));

fn hash_u32s(v: impl IntoIterator<Item = u32>) -> String {
    let mut h = H::new();
    for x in v {
        h.u32(x);
    }
    h.hex()
}

fn layouts() -> Vec<(&'static str, String)> {
    use crate::stego::armor::{fortress, fortress_tile, selection, spreading};
    use crate::stego::ghost::{shadow, v31, v3_legacy};
    use crate::stego::permute::CoeffPos;
    use crate::stego::{cover_order, crypto, permute, stc};

    let mut out = Vec::new();
    let img = crate::JpegImage::from_bytes(PHOTO).expect("test JPEG");
    let grid = img.dct_grid(0);

    // Shared Fisher-Yates (Ghost v3, Armor positions, legacy Ghost v1).
    let mut v: Vec<u32> = (0..100_000).collect();
    permute::shuffle_u32(&mut v, &seed(20));
    out.push(("permute_shuffle_u32", hash_u32s(v)));
    let mut p: Vec<CoeffPos> = (0..5000).map(|i| CoeffPos { flat_idx: i * 3, cost: i as f32 }).collect();
    permute::permute_positions(&mut p, &seed(21));
    out.push(("permute_positions", hash_u32s(p.iter().map(|c| c.flat_idx))));

    // Ghost v3 (decode-only) and v3.1: the shared order, the H-hat columns
    // and v3.1's pair bits from a passphrase.
    let key = crypto::derive_structural_key("golden ghost passphrase").unwrap();
    let seeds = v31::v31_seeds(&key);
    out.push(("ghost_v3_order", hash_u32s(v31::order(grid, &seeds))));
    let mut cols = Vec::new();
    for w in [1usize, 2, 7, 16, 23] {
        cols.extend(v3_legacy::layer_cols_for_test(&key, 1, w));
        cols.extend(v3_legacy::layer_cols_for_test(&key, 2, w));
    }
    out.push(("ghost_v3_hhat_cols", hash_u32s(cols)));
    let mut cols = Vec::new();
    for (nk, m) in [(1000usize, 900usize), (5000, 700), (100_000, 2345)] {
        let g = v31::LaneGeom::new(nk, 2 * m).unwrap();
        for layer in [1u8, 2] {
            let code = v31::layer_code(&seeds, layer, &g, v31::V31_STC_H);
            for set in code.column_sets() {
                cols.extend_from_slice(set);
            }
        }
    }
    out.push(("ghost_v31_hhat_cols", hash_u32s(cols)));
    let pairs = v31::PairKey::new(&seeds.pair, 100_000);
    out.push(("ghost_v31_pairs", hash_u32s((0..100_000u32).map(|fi| pairs.get(fi) as u32))));

    // STC H-hat (Ghost v1, shadows, video).
    let mut h = H::new();
    for (hh, w) in [(7usize, 1usize), (7, 10), (10, 50), (9, 3000)] {
        for row in stc::hhat::generate_hhat(hh, w, &seed(22)) {
            for x in row {
                h.u32(x);
            }
        }
    }
    out.push(("stc_hhat", h.hex()));

    // Shadow orders (raw ChaCha20 priorities).
    let skey = crypto::derive_shadow_structural_key("golden shadow passphrase").unwrap();
    out.push(("shadow_ac_order", hash_u32s(shadow::ac_hash_order_prefix(grid, &skey, 5000))));
    out.push(("shadow_nzac_order", hash_u32s(shadow::nzac_hash_order_prefix(grid, &skey, 2000))));
    let pos: Vec<CoeffPos> = (0..3000).map(|i| CoeffPos { flat_idx: i * 5 + 1, cost: 1.0 }).collect();
    out.push(("shadow_hash_order", hash_u32s(shadow::hash_order(&pos, &skey).iter().map(|c| c.flat_idx))));

    // Armor STDM: position order and spreading vectors.
    let akey = crypto::derive_armor_structural_key("golden armor passphrase").unwrap();
    let perm_seed: [u8; 32] = akey[..32].try_into().unwrap();
    let spread_seed: [u8; 32] = akey[32..].try_into().unwrap();
    out.push(("armor_positions", hash_u32s(selection::armor_positions(40, 30, &perm_seed))));
    let mut h = H::new();
    for vec in spreading::generate_spreading_vectors(&spread_seed, 2000) {
        for x in vec {
            h.u64(x.to_bits());
        }
    }
    for vec in spreading::spreading_vectors_range(&spread_seed, 1234, 10) {
        for x in vec {
            h.u64(x.to_bits());
        }
    }
    out.push(("armor_spreading", h.hex()));

    // Fortress (RS / CC) block permutation and the Fortress T / TL tile key.
    let fkey = crypto::derive_fortress_structural_key("golden fortress passphrase").unwrap();
    out.push(("fortress_blocks", hash_u32s(fortress::permute_blocks(12_345, &fkey).into_iter().map(|x| x as u32))));
    let mut h = H::new();
    for pass in ["golden fortress passphrase", ""] {
        let tk = fortress_tile::TileKey::new(pass).unwrap();
        let (rank, pilot_bit) = tk.keyed_parts();
        for &x in rank {
            h.u32(x);
        }
        h.b(pilot_bit);
    }
    out.push(("fortress_tile_key", h.hex()));

    // Video GOP cover order.
    out.push(("video_gop_cover_order", hash_u32s(cover_order::gop_cover_order(77_777, &seed(23)))));

    // WASM ChaCha20 stream (pinned there too, by the Node test).
    out.push(("chacha_stream_test_hash", crate::stego::chacha_stream::chacha_stream_test_hash_hex()));
    out
}

const RAW_GOLDEN: &[(&str, &str)] = &[
    ("chacha20_next_u32_first", "2929549541,2701884826,2340975408,2613552312"),
    ("chacha20_next_u32", "fe3feb7c3c45b8312977e03d69b33ced0a8171fdffe3eae18e8267fbc8b795b8"),
    // The same words as above: next_u64 is two consecutive words, low first,
    // at any alignment and across the buffer edge.
    ("chacha20_mixed_u32_u64", "fe3feb7c3c45b8312977e03d69b33ced0a8171fdffe3eae18e8267fbc8b795b8"),
    ("chacha20_fill_bytes", "74ea0a00527ae010e7530b7a4eef48acd7e626b8806bfbf3ddddc5f3d5e9cc71"),
    ("chacha20_seek_stream", "c7f00f34cf29b55513b9a871c1edb96819fda9122b582a1f9c20db930605e825"),
    ("seed_from_u64_chacha20_chacha8", "23543a86cbbdad332e743c3f0fecfd09b2c4265d8db4ba1f5fa363fa66b5dc88"),
];
const SAMPLER_GOLDEN: &[(&str, &str)] = &[
    ("u32_incl_fy_first", "5186,14943,14265,14709,843,1306"),
    ("u32_incl_fy", "220990b6bff0069a61467ffd25a08d8d2335759d8e7c36ce7fd7cccf4a4907f4"),
    ("u32_incl_large", "a80ed1fe960f11297dba70c21fc2b1369d5ddff987fbc353025cb129785e6643"),
    ("u32_excl", "03b4f2cadd4b5726ec1a422ed8efbeb52b93c2a2f52c8d41744aa57bf1f83beb"),
    ("small_int_ranges", "9034c4b0cfd2549761305e3891381e513172599348d376d3ad8752ca719c2f1c"),
    ("f64_range_first", "bfe677a22dacb308,bfdd1847b82cb5e0,3fe256c28cf2cc20"),
    ("f64_range", "ba0255fd856ef38297fd7bd03be980a94ef126afc79d872511c24996fdfae5e2"),
    ("gen_and_bool", "3806e4624e584f7602ff8f056bf4d0599f92547e30b4cf9fa6e8bda2f8fd3260"),
];
const LAYOUT_GOLDEN: &[(&str, &str)] = &[
    ("permute_shuffle_u32", "7fe29cbe1ae99af52b1cdaae4ce7093c6d76022f8008d70e132905fc4d82466a"),
    ("permute_positions", "75ace8cfd0846e12d80e004b0199e5c6a0292843daae6d1b7a2929ca6a8c8192"),
    ("ghost_v3_order", "f9f44c5239b4001ec13a4b059f539cb65fd3b9c17b0830568a4c8048c816f6a4"),
    ("ghost_v3_hhat_cols", "c00511995af3a798969eee5fd8d5c392066bda740f3941ede2257dfbd9111679"),
    ("ghost_v31_hhat_cols", "c5df7e32077d385fe71a05eb7228e330a356578440f2b0380b5a8a26e588a54a"),
    ("ghost_v31_pairs", "b649ae8d975623c7482d3a378ac899d41989ad8a6f849ee4b5c7151226de2147"),
    ("stc_hhat", "527f72e224c4524b76460b8b2ecb9a95076f1a5080cedfaf98ac8aac4d176474"),
    ("shadow_ac_order", "ea25c34d4699bb7d0f4917656a0683a1a9730edac9fdfcc1d455fc381debb15c"),
    ("shadow_nzac_order", "8c576802efc99134a63a5bb803ed2b1270e859f8c3feead57cb394af6e5dba9f"),
    ("shadow_hash_order", "960d4a48fbe975b319aa167c4764e1fadaeaf1d71e2bcea8b701aab6cd794bf9"),
    ("armor_positions", "147b16dd20551907538ea2406952bf9fc4fdf335e774a88d9bf9a4cd772dd112"),
    ("armor_spreading", "507267d6374b5c1a87968aac5991d9c7548ddf4486bc72254c5eb81684ba7e16"),
    ("fortress_blocks", "f2c08f6e8ecc1d4c3dd7c02488b9320545db540ef6ca8fef359da31777d0078d"),
    ("fortress_tile_key", "99292a6d1c69ad39aa2e3fdb9751160547d914be50ae897594fc1a7ba92d4138"),
    ("video_gop_cover_order", "f820c75893397c178eabade3f1abad1248fa6b35e71265d2de5ab7526dbd3ba2"),
    ("chacha_stream_test_hash", "aa197740aecc5e8f0304f9be3d554e7a09694095c033a94b2288df31698003f8"),
];

#[test]
fn golden_raw_chacha_streams() {
    check(&raw_streams(), RAW_GOLDEN);
}

#[test]
fn golden_samplers() {
    check(&samplers(), SAMPLER_GOLDEN);
}

#[test]
fn golden_keyed_layouts() {
    check(&layouts(), LAYOUT_GOLDEN);
}
