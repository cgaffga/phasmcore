// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// The gate that decides whether the shadow prefix cache is usable at all:
// **the emitted stego bytes must not depend on whether the cache was on.**
//
// The cache replaces a recomputation of the clean encode → CABAC walk →
// J-UNIWARD costs → safe-MSL derivation that sweep A already did. If a cached
// prefix differs from a recomputed one in any way that reaches STC — a cost
// value, an ∞ flag, a cover bit, a position key — the encoder picks different
// flips and emits a different file. That file would still encode without
// error and would still *look* fine; it would simply decode to garbage for the
// shadow layers. So "it round-trips" is not sufficient evidence here, and this
// test compares bytes.
//
// `PHASM_DETERMINISTIC_SEED` is mandatory: `crypto::encrypt` draws a random
// salt+nonce per call, so without it two runs differ for reasons that have
// nothing to do with the cache.

#![cfg(feature = "h264-encoder")]

mod common;

use phasm_core::codec::h264::openh264_stego::{
    h264_encode_with_shadows_streaming, EncodeOpts, GopYuvSource,
};
use phasm_core::codec::h264::stego::cost_weights::CostWeights;
use phasm_core::stego::shadow_layer::ShadowLayer;

use std::sync::{Mutex, OnceLock};

// The OH264 fork keeps process-global encoder state; these must not overlap.
static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

const W: u32 = 320;
const H: u32 = 192;
const GOP: u32 = 10;
/// 3 GOPs: enough that sweep A caches more than one entry and sweep B fetches
/// them, which a single-GOP clip would not exercise.
const N: u32 = 30;
const QP: i32 = 26;

/// Deterministic synthetic clip with real spatial and temporal structure —
/// flat regions, a moving edge and noise — so the cover has a non-trivial mix
/// of flippable and wet positions.
fn synth_yuv(w: u32, h: u32, n: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut out = Vec::with_capacity(w * h * 3 / 2 * n as usize);
    let mut s: u32 = 0x1234_5678;
    for f in 0..n as usize {
        for y in 0..h {
            for x in 0..w {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let edge = ((x + f * 3) % w) < w / 2;
                let v = match (x * 4 / w, edge) {
                    (0, _) => 16u8,
                    (1, _) => 235,
                    (2, true) => ((x * 7 + y * 3 + f) % 256) as u8,
                    _ => (s >> 24) as u8,
                };
                out.push(v);
            }
        }
        for _ in 0..(w * h / 2) {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            out.push(128u8.wrapping_add(((s >> 26) as u8) & 0x0F));
        }
    }
    out
}

struct SliceSource {
    yuv: Vec<u8>,
    frame_bytes: usize,
    gop: u32,
    n_frames: u32,
}

impl GopYuvSource for SliceSource {
    fn gop_yuv(&mut self, gop_idx: u32) -> Result<Vec<u8>, phasm_core::stego::error::StegoError> {
        let start = (gop_idx * self.gop) as usize;
        let count = ((self.n_frames - gop_idx * self.gop).min(self.gop)) as usize;
        let a = start * self.frame_bytes;
        let b = a + count * self.frame_bytes;
        Ok(self.yuv[a..b].to_vec())
    }
    fn rewind(&mut self) -> Result<(), phasm_core::stego::error::StegoError> {
        Ok(())
    }
}

fn encode_with(budget_env: Option<(&str, &str)>) -> Vec<u8> {
    // SAFETY: single-threaded here; the session guard serializes the tests and
    // `--test-threads=1` is the convention for every OH264 test.
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        match budget_env {
            Some((k, v)) => std::env::set_var(k, v),
            None => {}
        }
    }

    let yuv = synth_yuv(W, H, N);
    let frame_bytes = (W * H * 3 / 2) as usize;
    let mut src = SliceSource { yuv, frame_bytes, gop: GOP, n_frames: N };

    let opts = EncodeOpts { qp: QP, intra_period: GOP as i32 };
    let shadows = vec![ShadowLayer { passphrase: "s1", message: "shadow one", files: &[] }];

    let weights = CostWeights::default();
    h264_encode_with_shadows_streaming(
        &mut src, W, H, N, opts, "primary message", &[], "primary-pass", &shadows, &weights, None, None,
    )
    .expect("shadow encode")
}

/// Cache ON vs cache OFF must produce the same bytes.
///
/// Driven through the env override rather than the parameter so the test
/// exercises the production path end to end — the orchestrator building the
/// cache, sweep A inserting, every sweep-B parity tier fetching.
#[test]
fn shadow_output_is_byte_identical_with_and_without_the_prefix_cache() {
    let _g = guard().lock().unwrap_or_else(|e| e.into_inner());

    // 0 MB disables both tiers → `build()` returns None → no cache at all.
    let without = encode_with(Some(("PHASM_H264_SHADOW_CACHE_MB", "0")));
    // Generous RAM budget: every GOP of this clip fits, so sweep B is served
    // entirely from cache and the recompute path never runs.
    let with_ram = encode_with(Some(("PHASM_H264_SHADOW_CACHE_MB", "256")));

    assert!(!without.is_empty(), "encode produced nothing");
    assert_eq!(
        without.len(),
        with_ram.len(),
        "cache changed the output LENGTH ({} vs {}) — the cached prefix is not \
         equivalent to a recomputed one",
        without.len(),
        with_ram.len()
    );
    let first_diff = without.iter().zip(with_ram.iter()).position(|(a, b)| a != b);
    assert_eq!(
        first_diff, None,
        "cache changed the output at byte {:?}. The cached prefix must be \
         indistinguishable from a recomputed one; a difference here means the \
         encoder chose different flips, which emits a file that encodes fine \
         and decodes to garbage.",
        first_diff
    );
}

/// Same, through the DISK tier. A RAM budget of 0 with a disk budget forces
/// every entry through flat_pack/flat_unpack, so this is the gate on the
/// serializer specifically — the RAM tier only clones and would not catch a
/// pack/unpack bug.
#[test]
fn shadow_output_is_byte_identical_when_the_cache_spills_to_disk() {
    let _g = guard().lock().unwrap_or_else(|e| e.into_inner());

    let without = encode_with(Some(("PHASM_H264_SHADOW_CACHE_MB", "0")));

    // SAFETY: as above.
    unsafe {
        std::env::set_var("PHASM_H264_SHADOW_CACHE_MB", "0");
        std::env::set_var("PHASM_H264_SHADOW_CACHE_DISK_MB", "256");
    }
    let spilled = encode_with(None);
    unsafe {
        std::env::remove_var("PHASM_H264_SHADOW_CACHE_DISK_MB");
    }

    assert_eq!(without.len(), spilled.len(), "disk-spilled cache changed output length");
    assert_eq!(
        without.iter().zip(spilled.iter()).position(|(a, b)| a != b),
        None,
        "disk-spilled cache changed the output — flat_pack/flat_unpack is not \
         round-tripping the prefix exactly"
    );
}

/// A cache too small to hold anything must behave exactly like no cache: every
/// GOP overflows, sweep B recomputes. This is the overflow-to-recompute
/// fallthrough the whole design rests on, exercised through the real encoder
/// rather than a unit test.
#[test]
fn a_cache_too_small_to_hold_anything_still_produces_identical_output() {
    let _g = guard().lock().unwrap_or_else(|e| e.into_inner());

    let without = encode_with(Some(("PHASM_H264_SHADOW_CACHE_MB", "0")));
    // 1 MB of RAM cannot hold even one GOP's prefix at this resolution's
    // per-position cost, so most or all GOPs fall through.
    let tiny = encode_with(Some(("PHASM_H264_SHADOW_CACHE_MB", "1")));

    assert_eq!(without.len(), tiny.len(), "undersized cache changed output length");
    assert_eq!(
        without.iter().zip(tiny.iter()).position(|(a, b)| a != b),
        None,
        "undersized cache changed the output — overflow must fall through to \
         recompute, not to a different result"
    );
}
