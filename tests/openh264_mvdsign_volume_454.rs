// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Christoph Gaffga
// https://github.com/cgaffga/phasmcore

//! Task #454 — empirical MvdSign cascade-leak measurement at volume.
//!
//! C.8.7 ships the MvdSign cascade-break (encoder shifts `pDecPic` by
//! `(CLEAN_MC − STEGO_MC)` so next-frame ME stays on the clean
//! reference). The math is correct and round-trip green, but the
//! synthetic `synth_yuv_frame` fixtures used by C.3.2 produce few
//! MvdSign override sites (most MBs predict to MV=0 → MVD=0 → no
//! override site). The leak-closure number was never measured at
//! volume.
//!
//! This file fills the measurement gap with three fixture variants
//! as suggested in `phase-c8-visual-recon-plan.md:185-204`:
//!
//! 1. **Sinusoidal per-MB shift** — synthetic content where every
//!    frame translates by a per-frame offset. Every inter-MB has a
//!    non-zero predicted MV → MvdSign exercises at maximum density.
//!
//! 2. **Real-corpus motion-rich YUV** — carplane fixture (1072×1920,
//!    10 frames) from the C.3.x corpus. High natural motion content.
//!
//! 3. **Flip-every-bit override mode** — diagnostic control. Override
//!    hook flips every MvdSign bit regardless of the cascade-safety
//!    predicate. Measures the leak-rate that would occur WITHOUT the
//!    safe-mask filter — i.e. what C.8.7 alone closes vs what the
//!    consumer-side safe-mask additionally closes.
//!
//! Per-fixture metrics:
//!   - `n_mvd`         total MvdSign override sites in baseline walk
//!   - `n_safe`        positions flagged cascade-safe by predicate
//!   - `n_flipped`     positions where the override fired
//!   - `Δstruct`       (flipped emit count − baseline emit count) →
//!                     **primary cascade-leak signal**. If the
//!                     cascade is fully closed, the flipped encode
//!                     produces the same number of MvdSign sites as
//!                     baseline; Δstruct ≠ 0 means contaminated
//!                     mode-decision shifted in subsequent frames.
//!   - `n_intended_landed` positions in the common prefix where the
//!                     override actually landed in the wire.
//!   - `n_unintended`  positions outside the flip set whose bit
//!                     differs in the common prefix. NOTE: when
//!                     Δstruct ≠ 0 this metric is contaminated by
//!                     position-shift noise (a single insertion
//!                     causes all subsequent bits to read as
//!                     "different"), so treat it as an upper bound
//!                     not a tight measurement.
//!   - `n_mag_drift`   |MVD| differences in the common prefix. Same
//!                     position-shift caveat as `n_unintended`.
//!
//! Marked `#[ignore]`: research measurement, not a CI ship-gate. Run
//! manually:
//!   cargo test --release --features "h264-encoder" \
//!     --test openh264_mvdsign_volume_454 -- --ignored --nocapture

#![cfg(feature = "h264-encoder")]

use core_openh264_sys::PhasmStegoDomain;
use phasm_core::codec::h264::cabac::bin_decoder::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};
use phasm_core::codec::h264::openh264::{
    set_frame_num, Encoder, StegoHandlers, StegoSession,
};
use phasm_core::codec::h264::stego::cascade_safety::analyze_safe_mvd_subset;
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

static SESSION_GUARD: Mutex<()> = Mutex::new(());

const QP: i32 = 26;
const GOP_SIZE: i32 = 30;

// ─────────────────────────────────────────────────────────────────────
// Fixture helpers
// ─────────────────────────────────────────────────────────────────────

/// Sinusoidal per-frame translation synthetic YUV. Each frame is the
/// previous frame translated by a small per-frame offset, so every
/// inter-MB has a non-zero predicted MV in the natural direction.
/// Produces MvdSign override sites at maximum density.
///
/// Frame `t`'s pixel `(x, y)` = base_texture(`x − dx(t)`, `y − dy(t)`)
/// where `dx(t) = 6 * sin(2π * t / 8)`, `dy(t) = 6 * cos(2π * t / 8)`.
fn synth_yuv_sinusoidal_motion(width: u32, height: u32, n_frames: u32) -> Vec<u8> {
    let w = width as i32;
    let h = height as i32;
    let frame_size = (width * height * 3 / 2) as usize;
    let mut out = Vec::with_capacity(frame_size * n_frames as usize);

    let base_y = |x: i32, y: i32| -> u8 {
        let xi = x.rem_euclid(w);
        let yi = y.rem_euclid(h);
        let v = ((xi ^ yi) as u32).wrapping_mul(7) ^ ((xi.wrapping_mul(13) + yi.wrapping_mul(17)) as u32);
        (v & 0xFF) as u8
    };
    let base_c = |x: i32, y: i32, plane: u8| -> u8 {
        let xi = x.rem_euclid(w / 2);
        let yi = y.rem_euclid(h / 2);
        let mix = ((xi.wrapping_mul(11) + yi.wrapping_mul(19)) as u32).wrapping_add(plane as u32 * 41);
        ((mix >> 1) & 0xFF) as u8 + 64
    };

    for t in 0..n_frames {
        let theta = (t as f64) * std::f64::consts::TAU / 8.0;
        let dx = (6.0 * theta.sin()).round() as i32;
        let dy = (6.0 * theta.cos()).round() as i32;
        // Y plane
        for y in 0..h {
            for x in 0..w {
                out.push(base_y(x - dx, y - dy));
            }
        }
        // U + V planes (4:2:0, half-rate in both dims)
        for plane in 0..2u8 {
            for y in 0..(h / 2) {
                for x in 0..(w / 2) {
                    out.push(base_c(x - dx / 2, y - dy / 2, plane));
                }
            }
        }
    }
    out
}

fn read_yuv(path: &std::path::Path) -> Option<Vec<u8>> {
    let mut buf = Vec::new();
    File::open(path).ok()?.read_to_end(&mut buf).ok()?;
    Some(buf)
}

// ─────────────────────────────────────────────────────────────────────
// Encode helpers
// ─────────────────────────────────────────────────────────────────────

fn encode_clean(yuv: &[u8], width: i32, height: i32, n_frames: u32) -> Vec<u8> {
    let luma_plane = (width as usize) * (height as usize);
    let chroma_plane = (width as usize / 2) * (height as usize / 2);
    let frame_bytes = luma_plane + 2 * chroma_plane;

    set_frame_num(0);
    let mut enc = Encoder::new(width, height, QP, GOP_SIZE).expect("Encoder::new");
    let mut annex_b = Vec::new();
    let mut nal_buf = vec![0u8; frame_bytes * 2];
    for i in 0..n_frames {
        let off = (i as usize) * frame_bytes;
        let y = &yuv[off..off + luma_plane];
        let u = &yuv[off + luma_plane..off + luma_plane + chroma_plane];
        let v = &yuv[off + luma_plane + chroma_plane..off + frame_bytes];
        let (_ft, n) = enc
            .encode_frame(y, u, v, (i as i64) * 33, &mut nal_buf)
            .expect("encode_frame");
        annex_b.extend_from_slice(&nal_buf[..n]);
    }
    annex_b
}

/// Encode with an enc_pre_emit hook that flips MvdSign bin `i` if
/// `flip_mask[i]` is true. Counter inside the hook tracks the bin
/// index in raster MB scan order, matching walker semantics.
fn encode_with_mvd_overrides(
    yuv: &[u8],
    width: i32,
    height: i32,
    n_frames: u32,
    flip_mask: &[bool],
) -> Vec<u8> {
    let flip_set: HashSet<usize> = flip_mask
        .iter()
        .enumerate()
        .filter(|&(_, &b)| b)
        .map(|(i, _)| i)
        .collect();
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_for_hook = counter.clone();
    let handlers = StegoHandlers {
        enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
            if pos.domain != PhasmStegoDomain::MvdSign as u8 {
                return None;
            }
            let idx = counter_for_hook.fetch_add(1, Ordering::SeqCst);
            if flip_set.contains(&idx) {
                Some(1 - original)
            } else {
                None
            }
        })),
        ..Default::default()
    };
    let _session = StegoSession::register(handlers).expect("register");
    encode_clean(yuv, width, height, n_frames)
}

// ─────────────────────────────────────────────────────────────────────
// Measurement core
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct LeakReport {
    n_mvd_baseline: usize,
    n_safe: usize,
    n_flipped: usize,
    n_intended_landed: usize,
    n_unintended_outside_set: usize,
    n_mag_drift: usize,
    structural_delta: i64, // flipped_n - baseline_n (negative = lost MVD sites)
}

/// Run baseline + flipped encode, walk both, compute cascade-leak
/// metrics. `flip_strategy` builds the per-bin flip mask from the
/// baseline walker output.
fn measure_cascade_leak(
    label: &str,
    yuv: &[u8],
    width: i32,
    height: i32,
    n_frames: u32,
    flip_strategy: impl FnOnce(&[bool], usize) -> Vec<bool>,
) -> LeakReport {
    // Baseline encode + walk.
    let baseline_h264 = encode_clean(yuv, width, height, n_frames);
    let baseline = walk_annex_b_for_cover_with_options(
        &baseline_h264,
        WalkOptions { record_mvd: true },
    )
    .unwrap_or_else(|e| panic!("{label}: baseline walker: {e:?}"));
    let n_mvd = baseline.mvd_meta.len();
    let safe_mask = analyze_safe_mvd_subset(&baseline.mvd_meta, baseline.mb_w, baseline.mb_h);
    let n_safe = safe_mask.iter().filter(|&&b| b).count();

    let flip_mask = flip_strategy(&safe_mask, n_mvd);
    let n_flipped = flip_mask.iter().filter(|&&b| b).count();

    if n_flipped == 0 {
        return LeakReport {
            n_mvd_baseline: n_mvd, n_safe, n_flipped: 0,
            n_intended_landed: 0,
            n_unintended_outside_set: 0,
            n_mag_drift: 0,
            structural_delta: 0,
        };
    }

    let flipped_h264 = encode_with_mvd_overrides(yuv, width, height, n_frames, &flip_mask);
    let flipped = walk_annex_b_for_cover_with_options(
        &flipped_h264,
        WalkOptions { record_mvd: true },
    )
    .unwrap_or_else(|e| panic!("{label}: flipped walker: {e:?}"));

    let baseline_n = baseline.mvd_meta.len();
    let flipped_n = flipped.mvd_meta.len();
    let structural_delta = flipped_n as i64 - baseline_n as i64;

    // Compare up to the shared prefix so we get a usable per-bit number
    // even when the structures diverge (which they will whenever the
    // overrides shift mode decision on subsequent frames).
    let common_n = baseline_n.min(flipped_n);
    let mut n_mag_drift = 0usize;
    let mut n_intended_landed = 0usize;
    let mut n_unintended_outside_set = 0usize;
    for i in 0..common_n {
        if baseline.mvd_meta[i].magnitude != flipped.mvd_meta[i].magnitude {
            n_mag_drift += 1;
        }
    }
    let common_bits = baseline.cover.mvd_sign_bypass.bits.len()
        .min(flipped.cover.mvd_sign_bypass.bits.len());
    for i in 0..common_bits {
        let b = baseline.cover.mvd_sign_bypass.bits[i];
        let f = flipped.cover.mvd_sign_bypass.bits[i];
        let in_flip_set = flip_mask.get(i).copied().unwrap_or(false);
        if in_flip_set {
            if f == 1 - b {
                n_intended_landed += 1;
            }
        } else if f != b {
            n_unintended_outside_set += 1;
        }
    }

    LeakReport {
        n_mvd_baseline: baseline_n,
        n_safe, n_flipped,
        n_intended_landed,
        n_unintended_outside_set,
        n_mag_drift,
        structural_delta,
    }
}

fn report(label: &str, r: &LeakReport) {
    let safe_pct = if r.n_mvd_baseline == 0 { 0.0 } else { 100.0 * r.n_safe as f64 / r.n_mvd_baseline as f64 };
    let landed_pct = if r.n_flipped == 0 { 0.0 } else { 100.0 * r.n_intended_landed as f64 / r.n_flipped as f64 };
    let leak_pct = if r.n_mvd_baseline == 0 || r.n_flipped == 0 {
        0.0
    } else {
        100.0 * r.n_unintended_outside_set as f64 / r.n_mvd_baseline as f64
    };
    eprintln!(
        "  {:<28} n_mvd={:>6}  safe={:>5} ({:>4.1}%)  Δstruct={:>+6}  flipped={:>5}  landed={:>5} ({:>5.1}%)  unintended={:>5} ({:>5.2}%)  mag_drift={:>5}",
        label, r.n_mvd_baseline, r.n_safe, safe_pct,
        r.structural_delta,
        r.n_flipped, r.n_intended_landed, landed_pct,
        r.n_unintended_outside_set, leak_pct,
        r.n_mag_drift,
    );
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

/// Approach 1: synthetic sinusoidal-motion fixture, exercise MvdSign at
/// maximum density. Bulk-flip cascade-safe positions; expect near-zero
/// unintended changes (C.8.7 closes the cascade).
#[test]
#[ignore]
fn mvdsign_volume_sinusoidal_synth() {
    let _g = SESSION_GUARD.lock().unwrap();
    eprintln!(
        "\n=== #454 measurement — sinusoidal-motion synth YUV ===\n"
    );
    eprintln!("Fixture: 320×240 × 12 frames, per-frame translation, QP={QP}, GOP={GOP_SIZE}");
    eprintln!();
    eprintln!("  {:<28} {}", "label", "metrics");
    eprintln!("  {}", "─".repeat(132));

    let width = 320i32;
    let height = 240i32;
    let n_frames = 12u32;
    let yuv = synth_yuv_sinusoidal_motion(width as u32, height as u32, n_frames);

    // Approach 1a: flip cascade-safe positions only (the production path).
    let r_safe = measure_cascade_leak(
        "synth_motion_safe_flip", &yuv, width, height, n_frames,
        |safe_mask, _n| safe_mask.to_vec(),
    );
    report("synth_motion_safe_flip", &r_safe);

    eprintln!();
    eprintln!(
        "  interpretation: low `unintended` and `mag_drift` ⇒ cascade-safe predicate"
    );
    eprintln!(
        "  + C.8.7 dual-MC cascade-break close the leak together. With C.8.7 alone"
    );
    eprintln!(
        "  but no safe-mask, unsafe positions would still cause downstream"
    );
    eprintln!(
        "  mode-decision shifts (measured by `flip_every_bit` test)."
    );
}

/// Approach 2: real-corpus motion-rich YUV (carplane). Bulk-flip
/// cascade-safe positions on actual iPhone footage with significant
/// motion content. Expects similar zero-unintended result.
#[test]
#[ignore]
fn mvdsign_volume_real_corpus() {
    let _g = SESSION_GUARD.lock().unwrap();
    eprintln!(
        "\n=== #454 measurement — real-corpus motion-rich YUV ===\n"
    );

    // Use whichever real-corpus fixtures are present in /tmp.
    // regen_openh264_baseline.sh produces them.
    let fixtures: &[(&str, i32, i32)] = &[
        ("carplane",  1072, 1920),
        ("img4138",   1920, 1072),
        ("dji_mini2", 1920, 1072),
    ];
    let n_frames = 10u32;
    let mut any = false;
    eprintln!("Fixture suite (QP={QP}, GOP={GOP_SIZE}, {n_frames} frames):");
    eprintln!();
    eprintln!("  {:<28} {}", "label", "metrics");
    eprintln!("  {}", "─".repeat(132));
    for &(tag, width, height) in fixtures {
        let path = std::path::PathBuf::from(format!(
            "/tmp/openh264_baseline_{tag}_{width}x{height}_f{n_frames}.yuv"
        ));
        let yuv = match read_yuv(&path) {
            Some(b) => b,
            None => {
                eprintln!("  {:<28} (skip: {} missing)", tag, path.display());
                continue;
            }
        };
        let expected_size = (width as usize) * (height as usize) * 3 / 2 * (n_frames as usize);
        if yuv.len() < expected_size {
            eprintln!("  {:<28} (skip: YUV too short)", tag);
            continue;
        }
        any = true;

        let r = measure_cascade_leak(
            tag, &yuv, width, height, n_frames,
            |safe_mask, _n| safe_mask.to_vec(),
        );
        report(tag, &r);
    }
    if !any {
        eprintln!();
        eprintln!("  (no real-corpus fixtures available — run");
        eprintln!("   core/test-vectors/video/h264/real-world/source/regen_openh264_baseline.sh)");
    }
}

/// Approach 3: flip-every-bit override mode. Diagnostic control — flip
/// EVERY MvdSign bin regardless of cascade-safety predicate. Measures
/// the leak that would occur without the consumer-side safe-mask filter.
///
/// Expected: `n_unintended` substantially larger than the safe-flip
/// case, demonstrating the safe-mask is necessary even with C.8.7.
#[test]
#[ignore]
fn mvdsign_volume_flip_every_bit() {
    let _g = SESSION_GUARD.lock().unwrap();
    eprintln!(
        "\n=== #454 measurement — flip-every-bit control (no safe-mask) ===\n"
    );

    let width = 320i32;
    let height = 240i32;
    let n_frames = 12u32;
    let synth_yuv = synth_yuv_sinusoidal_motion(width as u32, height as u32, n_frames);

    eprintln!("Fixture: 320×240 × 12 sinusoidal-motion, QP={QP}, GOP={GOP_SIZE}");
    eprintln!();
    eprintln!("  {:<28} {}", "label", "metrics");
    eprintln!("  {}", "─".repeat(132));

    let r_all = measure_cascade_leak(
        "synth_motion_flip_every", &synth_yuv, width, height, n_frames,
        |_safe_mask, n| vec![true; n],
    );
    report("synth_motion_flip_every", &r_all);

    // NOTE: real-corpus flip-every-bit was attempted and segfaults inside
    // the OpenH264 fork — flipping every MvdSign bin on 1080p footage
    // destabilises mode decision past structural integrity (encoder
    // produces a different number of MVD sites every frame, and at
    // some point Skip pre-checks or neighbour-state ref_idx commits
    // hit an invalid path). The safe-mask filter explicitly avoids
    // this pathological state — that's the point.
    //
    // The synth case above is the controlled lab measurement. For real-
    // corpus volume, use `mvdsign_volume_real_corpus` (safe-mask only).
    eprintln!();
    eprintln!(
        "  interpretation: `Δstruct` ≠ 0 here is EXPECTED — flip-every-bit"
    );
    eprintln!(
        "  includes cascade-UNSAFE positions whose flips shift downstream"
    );
    eprintln!(
        "  mode decision and change MVD-site emission count. Magnitude of"
    );
    eprintln!(
        "  Δstruct quantifies the cascade leak without the safe-mask filter."
    );
    eprintln!(
        "  Compare against `mvdsign_volume_real_corpus` (safe-mask only)"
    );
    eprintln!(
        "  to see what the consumer-side filter contributes on top of"
    );
    eprintln!(
        "  C.8.7's dual-MC cascade-break."
    );
}
