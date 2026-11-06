// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! EXPERIMENTAL measurement knobs for the H.264 v2 probe-deliverability
//! study (`docs/design/video/h264/intra-drift-diagnosis.md`, follow-up).
//! Every knob defaults to the shipped behaviour; nothing here changes a
//! default. Each is read once per process (the decoder builds an H-hat per
//! `m_total` candidate, so a per-call env read would be measurable).
//!
//! * `PHASM_H264_HHAT_ENDS=1` — Path A: the primary STC uses
//!   [`generate_hhat_end_set`] (first + last bit of every H-hat column set)
//!   on the encoder, the decoder and the capacity probe.
//! * `PHASM_H264_CAP_TRIALS=k` — the probe takes the median of `k` fixed
//!   trial seed pairs per GOP instead of one (default 1).
//! * `PHASM_H264_CAP_DERATE=n/d` — the probe's derate (default 5/6).
//! * `PHASM_H264_SEED_VARIANTS=V` — Path B: the encoder may use seed
//!   variant `a ∈ 0..V` per GOP ([`super::keys::H264StcKeys::gop_variant`]);
//!   the decoder tries each in order (default 1 = variant 0 only).
//!
//! The knobs are a WIRE change when set: an encode made with one setting of
//! `PHASM_H264_HHAT_ENDS` / `PHASM_H264_SEED_VARIANTS` must be decoded with
//! the same setting.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::stego::stc::hhat::{generate_hhat, generate_hhat_end_set};

fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

/// Whether the H.264 primary STC uses the end-set H-hat (Path A). **Shipped
/// default: on** (maintainer decision, 2026-10-25). `PHASM_H264_HHAT_ENDS=0`
/// forces the plain `generate_hhat` for an A/B comparison; any other value
/// (or unset) keeps the shipped end-set H-hat.
pub fn hhat_end_set() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_str("PHASM_H264_HHAT_ENDS").as_deref() != Some("0"))
}

/// The H-hat of the H.264 primary STC (encode, decode, probe). Ships the
/// end-set H-hat (`generate_hhat_end_set`, Path A); `generate_hhat` stays
/// untouched for Ghost images and AV1.
pub fn h264_hhat(h: usize, w: usize, seed: &[u8; 32]) -> Vec<Vec<u32>> {
    if hhat_end_set() {
        generate_hhat_end_set(h, w, seed)
    } else {
        generate_hhat(h, w, seed)
    }
}

/// `PHASM_H264_SEED_VARIANTS` (1..=4, default 1).
pub fn seed_variants() -> u8 {
    static V: OnceLock<u8> = OnceLock::new();
    *V.get_or_init(|| {
        env_str("PHASM_H264_SEED_VARIANTS")
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(1)
            .clamp(1, 4)
    })
}

/// `PHASM_H264_CAP_TRIALS` (1..=15, default 1).
pub fn cap_trials() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        env_str("PHASM_H264_CAP_TRIALS")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 15)
    })
}

/// `PHASM_H264_CAP_DERATE=n/d`, or `None` for the shipped 5/6.
pub fn cap_derate() -> Option<(usize, usize)> {
    static V: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    *V.get_or_init(|| {
        let s = env_str("PHASM_H264_CAP_DERATE")?;
        let (n, d) = s.split_once('/')?;
        let (n, d) = (n.trim().parse().ok()?, d.trim().parse::<usize>().ok()?);
        (d > 0).then_some((n, d))
    })
}

/// Variant statistics (Path B measurement): per variant index, how often the
/// encoder tried it, how often it was the one embedded, and how many decoder
/// GOP searches ran with it.
static ENC_TRIES: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];
static ENC_WINS: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];
static DEC_TRIES: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];

#[cfg(feature = "h264-encoder")]
pub(crate) fn note_enc_try(a: u8) {
    ENC_TRIES[a as usize & 3].fetch_add(1, Ordering::Relaxed);
}
#[cfg(feature = "h264-encoder")]
pub(crate) fn note_enc_win(a: u8) {
    ENC_WINS[a as usize & 3].fetch_add(1, Ordering::Relaxed);
}
pub(crate) fn note_dec_try(a: u8) {
    DEC_TRIES[a as usize & 3].fetch_add(1, Ordering::Relaxed);
}

/// `[encoder tries, encoder wins, decoder searches]`, each per variant 0..4,
/// and reset them.
#[doc(hidden)]
pub fn take_variant_stats() -> [[usize; 4]; 3] {
    let take = |a: &[AtomicUsize; 4]| std::array::from_fn(|i| a[i].swap(0, Ordering::Relaxed));
    [take(&ENC_TRIES), take(&ENC_WINS), take(&DEC_TRIES)]
}
