// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Shadow verify must check exactly what a decoder will see.
//!
//! The Ghost shadow encoder accepts a shadow only after a decoder-side
//! verify: recompute the J-UNIWARD costs of the stego image, re-select
//! the shadow's cost pool from them, RS-decode + decrypt. Until
//! 2026-10 the cached-wavelet rungs (`FullParallel` = no memory budget,
//! as in the CLI; `CappedParallel`, picked by the mobile apps for many
//! photos) verified on an approximation of the stego costs:
//!
//! 1. the dirty set held only the STC flips, not the shadow LSB writes,
//!    so every block around a shadow modification kept its COVER cost;
//! 2. the stego wavelets were cover wavelets plus f64 deltas, which are
//!    not bit-identical to a decomposition of the stego pixels (and
//!    summed in HashSet order, so not even run-to-run deterministic).
//!
//! The verify then accepted shadows whose cost pool the decoder sees
//! differently: `FrameCorrupted` on decode, i.e. silent shadow loss
//! (2 of 16 seeds on the 12 MP photo with a 24 B shadow).
//!
//! The gates here:
//! * every accepted shadow decodes (12 MP repro, when the gitignored
//!   cover is present);
//! * the encode output is byte-identical on every memory rung, i.e. the
//!   cached rungs take exactly the streaming rung's accept/reject
//!   decisions — for plain and SI ("Deep Cover") encodes;
//! * the cached-rung stego positions equal the decoder's
//!   `compute_positions_streaming(stego, qt, None)` bit for bit under
//!   random modifications, including image-edge blocks.
//!
//! The memory budget and `PHASM_DETERMINISTIC_SEED` are process-global,
//! so every test here takes `GLOBAL_LOCK`.

use phasm_core::stego::cost::uniward::{
    compute_cover_wavelets, compute_positions_from_wavelets, compute_positions_streaming,
    compute_positions_with_dirty_recost, StegoWavelets,
};
use phasm_core::stego::permute::CoeffPos;
use phasm_core::{
    ghost_decode, ghost_encode_si_with_shadows, ghost_encode_with_shadows, ghost_shadow_decode,
    DctGrid, JpegImage, ShadowLayer, StegoError,
};
use std::sync::Mutex;

static GLOBAL_LOCK: Mutex<()> = Mutex::new(());

const PHOTO_12MP: &str = "photo_original_3557.jpg";

fn cover_path(name: &str) -> String {
    format!("test-vectors/image/{name}")
}

fn load_cover(name: &str) -> Option<Vec<u8>> {
    std::fs::read(cover_path(name)).ok()
}

/// Deterministic printable text of exactly `len` bytes.
fn text(len: usize, tag: usize) -> String {
    let prefix = format!("shadow{tag}-");
    let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789";
    (0..len)
        .map(|i| {
            if i < prefix.len() {
                prefix.as_bytes()[i] as char
            } else {
                alphabet[((i - prefix.len()) * 7 + tag * 13) % alphabet.len()] as char
            }
        })
        .collect()
}

fn shadows_for(n: usize, size: usize) -> Vec<ShadowLayer> {
    (0..n)
        .map(|k| ShadowLayer {
            message: text(size, k),
            passphrase: format!("shadow-pass-{k}"),
            files: vec![],
        })
        .collect()
}

/// Primary strictly larger than every shadow (no auto-sort swap).
fn primary_for(size: usize) -> String {
    text(size * 2 + 40, 99)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
    /// No memory budget → `FullParallel` (CLI, tests).
    Full,
    /// Budget of 1 byte → `MinimalClones` (streaming verify, no cache).
    Stream,
}

impl Path {
    fn budget(self) -> Option<usize> {
        match self {
            Path::Full => None,
            Path::Stream => Some(1),
        }
    }
}

/// Side information for an SI encode: the cover's own decoded RGB
/// pixels stand in for the "raw" source (as the CLI does for a PNG).
struct SiInput {
    rgb: Vec<u8>,
    w: u32,
    h: u32,
}

fn si_input(cover: &[u8]) -> SiInput {
    let img = image::load_from_memory_with_format(cover, image::ImageFormat::Jpeg)
        .expect("decode cover")
        .to_rgb8();
    let (w, h) = img.dimensions();
    SiInput { rgb: img.into_raw(), w, h }
}

/// Encode under a forced rung + deterministic salt/nonce. Caller holds
/// `GLOBAL_LOCK`.
fn encode(
    cover: &[u8],
    seed: u64,
    path: Path,
    primary: &str,
    shadows: &[ShadowLayer],
    si: Option<&SiInput>,
) -> Result<Vec<u8>, StegoError> {
    // SAFETY: process-global env + budget writes are serialized by
    // GLOBAL_LOCK (held by every test in this binary).
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", seed.to_string()) };
    phasm_core::set_memory_budget(path.budget());
    let res = match si {
        None => ghost_encode_with_shadows(cover, primary, &[], "primary-pass", shadows, None),
        Some(s) => ghost_encode_si_with_shadows(
            cover, &s.rgb, s.w, s.h, primary, &[], "primary-pass", shadows,
        ),
    };
    phasm_core::set_memory_budget(None);
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };
    res
}

/// (primary decodes, number of shadows that do NOT decode).
fn decode_all(stego: &[u8], primary: &str, shadows: &[ShadowLayer]) -> (bool, usize) {
    let primary_ok = matches!(ghost_decode(stego, "primary-pass"), Ok(p) if p.text == primary);
    let lost = shadows
        .iter()
        .filter(|s| !matches!(ghost_shadow_decode(stego, &s.passphrase), Ok(p) if p.text == s.message))
        .count();
    (primary_ok, lost)
}

fn lock() -> std::sync::MutexGuard<'static, ()> {
    GLOBAL_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

// ---------------------------------------------------------------------
// 1. The silent-loss repro (12 MP photo, gitignored → skips if absent).
// ---------------------------------------------------------------------

/// Seeds 25 and 26 were accepted-but-unreadable before the fix (24 B
/// shadow, `FullParallel`). Both must now decode, and the output must
/// equal the streaming rung's byte for byte.
#[test]
fn shadow_full_parallel_12mp_repro() {
    let Some(cover) = load_cover(PHOTO_12MP) else {
        eprintln!("SKIP: test-vectors/image/{PHOTO_12MP} not present (gitignored)");
        return;
    };
    let _g = lock();
    let primary = primary_for(24);
    let shadows = shadows_for(1, 24);
    for seed in [25u64, 26] {
        let full = encode(&cover, seed, Path::Full, &primary, &shadows, None)
            .expect("FullParallel encode");
        let (primary_ok, lost) = decode_all(&full, &primary, &shadows);
        assert!(primary_ok, "seed {seed}: primary must decode");
        assert_eq!(lost, 0, "seed {seed}: FullParallel accepted a shadow the decoder cannot read");
        let stream = encode(&cover, seed, Path::Stream, &primary, &shadows, None)
            .expect("streaming encode");
        assert!(full == stream, "seed {seed}: FullParallel and streaming rung must emit identical bytes");
    }
}

// ---------------------------------------------------------------------
// 2. Rung independence on the committed covers (fast, always runs).
// ---------------------------------------------------------------------

fn assert_rung_independent(cover_name: &str, si: bool, seeds: &[u64], n: usize, size: usize) {
    let cover = load_cover(cover_name).expect("committed cover");
    let si_in = si.then(|| si_input(&cover));
    let primary = primary_for(size);
    let shadows = shadows_for(n, size);
    for &seed in seeds {
        let full = encode(&cover, seed, Path::Full, &primary, &shadows, si_in.as_ref());
        let stream = encode(&cover, seed, Path::Stream, &primary, &shadows, si_in.as_ref());
        match (&full, &stream) {
            (Ok(f), Ok(s)) => {
                assert!(f == s, "{cover_name} si={si} seed={seed}: rungs emitted different bytes");
                let (primary_ok, lost) = decode_all(f, &primary, &shadows);
                assert!(primary_ok, "{cover_name} si={si} seed={seed}: primary lost");
                assert_eq!(lost, 0, "{cover_name} si={si} seed={seed}: shadow lost");
            }
            (Err(a), Err(b)) => assert_eq!(
                std::mem::discriminant(a),
                std::mem::discriminant(b),
                "{cover_name} si={si} seed={seed}: rungs rejected differently"
            ),
            _ => panic!(
                "{cover_name} si={si} seed={seed}: rungs disagree on accept/reject: \
                 full={:?} stream={:?}",
                full.as_ref().map(|b| b.len()),
                stream.as_ref().map(|b| b.len())
            ),
        }
    }
}

#[test]
fn rung_independent_output_plain() {
    let _g = lock();
    assert_rung_independent("photo_640x480_q75_420.jpg", false, &[1, 2, 3, 4], 2, 24);
    assert_rung_independent("photo_640x480_q75_420.jpg", false, &[5, 6], 1, 120);
}

#[test]
fn rung_independent_output_si() {
    let _g = lock();
    assert_rung_independent("photo_640x480_q75_420.jpg", true, &[1, 2, 3, 4], 2, 24);
    assert_rung_independent("photo_640x480_q75_420.jpg", true, &[5, 6], 1, 120);
}

// ---------------------------------------------------------------------
// 3. Cached-rung stego positions == decoder positions, bit for bit.
// ---------------------------------------------------------------------

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// nsF5-style ±1 on a non-zero AC coefficient (|1| → |2|), the only
/// kind of change the shadow LSB writer and the STC flip produce.
fn nsf5(c: i16) -> i16 {
    match c {
        1 => 2,
        -1 => -2,
        c if c > 1 => c - 1,
        c => c + 1,
    }
}

/// Mirrors the cached-rung verify in `ghost/pipeline.rs`
/// (`decoder_view_positions_sorted`, rungs 0/1): full coefficient diff
/// as the dirty set, exact stego wavelets, dirty re-cost over the cover
/// positions.
fn cached_rung_positions(cover: &DctGrid, stego: &DctGrid, qt: &phasm_core::QuantTable) -> Vec<CoeffPos> {
    let cw = compute_cover_wavelets(cover, qt);
    let cover_pos = compute_positions_from_wavelets(cover, qt, &cw, None).unwrap();
    drop(cw);
    let (bw, bt) = (cover.blocks_wide(), cover.blocks_tall());
    let mut mods = Vec::new();
    for br in 0..bt {
        for bc in 0..bw {
            for k in 0..64 {
                if cover.get(br, bc, k / 8, k % 8) != stego.get(br, bc, k / 8, k % 8) {
                    mods.push(((br * bw + bc) * 64 + k) as u32);
                }
            }
        }
    }
    let sw = StegoWavelets::compute_exact(stego, qt);
    compute_positions_with_dirty_recost(stego, qt, &cover_pos, &mods, &sw, None).unwrap()
}

fn assert_bit_identical(a: &[CoeffPos], b: &[CoeffPos], ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: position count");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.flat_idx, y.flat_idx, "{ctx}: flat_idx at {i}");
        assert_eq!(
            x.cost.to_bits(),
            y.cost.to_bits(),
            "{ctx}: cost at flat_idx {} ({} vs {})",
            x.flat_idx, x.cost, y.cost
        );
    }
}

fn check_decoder_view(cover_name: &str, n_mods: usize, seed: u64) {
    let Some(bytes) = load_cover(cover_name) else {
        eprintln!("SKIP: {cover_name} not present");
        return;
    };
    let img = JpegImage::from_bytes(&bytes).unwrap();
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).unwrap();
    let cover = img.dct_grid(0);
    let mut stego = cover.clone();
    let (bw, bt) = (cover.blocks_wide(), cover.blocks_tall());
    let mut rng = XorShift(seed | 1);
    // Random interior + every edge/corner block row/column.
    let mut blocks: Vec<(usize, usize)> = (0..n_mods).map(|_| (rng.below(bt), rng.below(bw))).collect();
    for bc in [0, bw / 2, bw - 1] {
        blocks.push((0, bc));
        blocks.push((bt - 1, bc));
    }
    for br in [bt / 3, bt - 2] {
        blocks.push((br, 0));
        blocks.push((br, bw - 1));
    }
    for (br, bc) in blocks {
        let k = 1 + rng.below(63);
        let c = stego.get(br, bc, k / 8, k % 8);
        if c != 0 {
            stego.set(br, bc, k / 8, k % 8, nsf5(c));
        }
    }
    let decoder = compute_positions_streaming(&stego, qt, None).unwrap();
    let cached = cached_rung_positions(cover, &stego, qt);
    assert_bit_identical(&cached, &decoder, &format!("{cover_name} n_mods={n_mods} seed={seed}"));
}

#[test]
fn cached_rung_positions_match_decoder_640() {
    for (n, seed) in [(10, 1), (300, 2), (3000, 3), (20000, 4)] {
        check_decoder_view("photo_640x480_q75_420.jpg", n, seed);
    }
    check_decoder_view("photo_320x240_q75_420.jpg", 500, 5);
    // Non-8-aligned / tiny geometries exercise the mirror boundaries.
    check_decoder_view("fractal_100x75_q85_420.jpg", 200, 6);
    check_decoder_view("nonaligned_13x13_q80.jpg", 20, 7);
}

#[test]
fn cached_rung_positions_match_decoder_large() {
    // Gitignored covers; skipped when absent.
    check_decoder_view("progressive_whatsapp_1200x1600.jpg", 5000, 8);
    check_decoder_view(PHOTO_12MP, 3000, 9);
}

// ---------------------------------------------------------------------
// 4. Measurement sweep (ignored; env-driven).
// ---------------------------------------------------------------------

/// PARITY_COVERS=a.jpg,b.jpg PARITY_SEEDS=21-36 PARITY_SIZES=8,24
/// PARITY_NSHADOWS=1,2 PARITY_PATHS=full,stream PARITY_SI=0|1
#[test]
#[ignore]
fn parity_sweep() {
    let _g = lock();
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let covers: Vec<String> = env("PARITY_COVERS", "photo_640x480_q75_420.jpg")
        .split(',').map(String::from).collect();
    let seeds: Vec<u64> = {
        let s = env("PARITY_SEEDS", "21-36");
        let (a, b) = s.split_once('-').unwrap();
        (a.parse().unwrap()..=b.parse().unwrap()).collect()
    };
    let list = |k: &str, d: &str| -> Vec<usize> {
        env(k, d).split(',').map(|x| x.parse().unwrap()).collect()
    };
    let sizes = list("PARITY_SIZES", "24");
    let nshadows = list("PARITY_NSHADOWS", "1");
    let paths: Vec<Path> = env("PARITY_PATHS", "full,stream")
        .split(',')
        .map(|p| if p == "stream" { Path::Stream } else { Path::Full })
        .collect();
    let si = env("PARITY_SI", "0") == "1";
    for cname in &covers {
        let cover = load_cover(cname).expect("cover");
        let si_in = si.then(|| si_input(&cover));
        for &n in &nshadows {
            for &size in &sizes {
                let primary = primary_for(size);
                let shadows = shadows_for(n, size);
                let mut outputs: Vec<Vec<Option<u64>>> = Vec::new();
                for &path in &paths {
                    let (mut acc, mut rej, mut lost_enc, mut lost, mut pfail) = (0, 0, 0, 0, 0);
                    let mut bad = Vec::new();
                    let mut hashes = Vec::new();
                    for &seed in &seeds {
                        match encode(&cover, seed, path, &primary, &shadows, si_in.as_ref()) {
                            Err(_) => {
                                rej += 1;
                                hashes.push(None);
                            }
                            Ok(st) => {
                                acc += 1;
                                hashes.push(Some(fnv(&st)));
                                let (pok, l) = decode_all(&st, &primary, &shadows);
                                if l > 0 {
                                    lost_enc += 1;
                                    lost += l;
                                    bad.push(seed);
                                }
                                if !pok {
                                    pfail += 1;
                                }
                            }
                        }
                    }
                    println!(
                        "RESULT cover={cname} si={si} n={n} size={size} path={path:?} seeds={} \
                         accepted={acc} rejected={rej} enc_with_lost_shadow={lost_enc} \
                         lost_shadows={lost} primary_fail={pfail} bad_seeds={bad:?}",
                        seeds.len()
                    );
                    outputs.push(hashes);
                }
                if outputs.len() == 2 {
                    let diff = outputs[0].iter().zip(&outputs[1]).filter(|(a, b)| a != b).count();
                    println!("RESULT-IDENT cover={cname} si={si} n={n} size={size} differing_outputs={diff}");
                }
            }
        }
    }
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
}
