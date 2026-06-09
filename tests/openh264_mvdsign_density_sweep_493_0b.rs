// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #493.0b Phase 0.5 — MvdSign density sweep.
//!
//! Empirically bound the MvdSign cliff at 0.5/2/5/10/20% utilisation
//! on the legacy 4-domain OH264 path (no per-GOP code yet). For each
//! (fixture, utilisation) pair, force a fraction of cascade-safe MvdSign
//! positions to flip and measure cascade leak per #454's protocol.
//!
//! Result feeds Phase 2 (#493.2) `CostWeights.mvd_sign` default. The
//! ship target is "stay below the cliff at >2× safety margin under
//! realistic STC operating density." The cliff is the utilisation
//! where `unintended` (cascade-leak bits outside the flip set) or
//! `Δstruct` (structural emit-count delta) crosses an unacceptable
//! threshold.
//!
//! Operating thresholds (proposed):
//! - GREEN: `unintended` ≤ 0.5% of n_mvd, `|Δstruct|` ≤ 0.5% of n_mvd.
//!   STC density safe at this util level for production.
//! - YELLOW: `unintended` ∈ (0.5%, 2%], `|Δstruct|` ∈ (0.5%, 2%].
//!   Marginal; only safe if cost vector keeps STC well below this
//!   level with headroom.
//! - RED: anything above. Definitely outside the safe operating
//!   regime; `CostWeights.mvd_sign` must keep utilisation below this
//!   point.
//!
//! Reused from #454: `analyze_safe_mvd_subset` predicate +
//! `encode_with_mvd_overrides` hook + the LeakReport metric set.
//! Primitives are intentionally duplicated rather than refactored into
//! a shared helper module — both files are #[ignore]'d measurement
//! research, and 454 is the canonical implementation.
//!
//! Run with:
//!   cargo test --release --features "h264-encoder" \
//!     --test openh264_mvdsign_density_sweep_493_0b -- --ignored --nocapture

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

/// Target utilisations to sweep. 0.5% is the current production
/// regime (#454 measured 2.2-25.6% leak there). 2% is the v1.0
/// validated upper bound. 5/10/20% probes the cliff.
const DENSITIES: &[f64] = &[0.005, 0.02, 0.05, 0.10, 0.20];

// ─────────────────────────────────────────────────────────────────────
// Fixture helpers (mirror #454)
// ─────────────────────────────────────────────────────────────────────

fn synth_yuv_sinusoidal_motion(width: u32, height: u32, n_frames: u32) -> Vec<u8> {
    let w = width as i32;
    let h = height as i32;
    let frame_size = (width * height * 3 / 2) as usize;
    let mut out = Vec::with_capacity(frame_size * n_frames as usize);

    let base_y = |x: i32, y: i32| -> u8 {
        let xi = x.rem_euclid(w);
        let yi = y.rem_euclid(h);
        let v = ((xi ^ yi) as u32).wrapping_mul(7)
            ^ ((xi.wrapping_mul(13) + yi.wrapping_mul(17)) as u32);
        (v & 0xFF) as u8
    };
    let base_c = |x: i32, y: i32, plane: u8| -> u8 {
        let xi = x.rem_euclid(w / 2);
        let yi = y.rem_euclid(h / 2);
        let mix = ((xi.wrapping_mul(11) + yi.wrapping_mul(19)) as u32)
            .wrapping_add(plane as u32 * 41);
        ((mix >> 1) & 0xFF) as u8 + 64
    };

    for t in 0..n_frames {
        let theta = (t as f64) * std::f64::consts::TAU / 8.0;
        let dx = (6.0 * theta.sin()).round() as i32;
        let dy = (6.0 * theta.cos()).round() as i32;
        for y in 0..h {
            for x in 0..w {
                out.push(base_y(x - dx, y - dy));
            }
        }
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
    let mut f = File::open(path).ok()?;
    let mut out = Vec::new();
    f.read_to_end(&mut out).ok()?;
    Some(out)
}

// ─────────────────────────────────────────────────────────────────────
// Encode helpers (mirror #454)
// ─────────────────────────────────────────────────────────────────────

fn encode_clean(yuv: &[u8], width: i32, height: i32, n_frames: u32) -> Vec<u8> {
    let mut enc = Encoder::new(width, height, QP, GOP_SIZE).expect("enc");
    let frame_y = (width * height) as usize;
    let frame_uv = (width * height / 4) as usize;
    let frame_total = frame_y + 2 * frame_uv;
    let mut out = vec![0u8; 8 * 1024 * 1024];
    let mut annex_b = Vec::with_capacity(4 * 1024 * 1024);
    for frame in 0..n_frames {
        set_frame_num(frame);
        let base = (frame as usize) * frame_total;
        let (_, n) = enc
            .encode_frame(
                &yuv[base..base + frame_y],
                &yuv[base + frame_y..base + frame_y + frame_uv],
                &yuv[base + frame_y + frame_uv..base + frame_total],
                (frame as i64) * 33,
                &mut out,
            )
            .expect("encode");
        annex_b.extend_from_slice(&out[..n]);
    }
    annex_b
}

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
// Density-sweep measurement
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct DensityResult {
    density: f64,
    n_mvd_baseline: usize,
    n_safe: usize,
    n_flipped: usize,
    actual_density_of_safe: f64,
    actual_density_of_total: f64,
    structural_delta: i64,
    n_unintended_outside_set: usize,
    leak_pct_of_total: f64,
    cliff_band: CliffBand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliffBand {
    Green,
    Yellow,
    Red,
    Skipped,
}

/// Pick approximately `target_density × n_safe` safe positions to flip
/// using a deterministic stride. Reproducible across runs.
fn density_flip_mask(safe_mask: &[bool], target_density: f64) -> Vec<bool> {
    let safe_indices: Vec<usize> = safe_mask
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s)
        .map(|(i, _)| i)
        .collect();
    let n_safe = safe_indices.len();
    if n_safe == 0 {
        return vec![false; safe_mask.len()];
    }
    let target_count = ((n_safe as f64) * target_density).round() as usize;
    let mut out = vec![false; safe_mask.len()];
    if target_count == 0 {
        return out;
    }
    let stride = (n_safe as f64) / (target_count as f64);
    for k in 0..target_count {
        let idx = ((k as f64 + 0.5) * stride) as usize;
        if let Some(&pos) = safe_indices.get(idx.min(n_safe - 1)) {
            out[pos] = true;
        }
    }
    out
}

fn classify(unintended_pct_of_total: f64, structural_pct_of_total: f64) -> CliffBand {
    let max_metric = unintended_pct_of_total.max(structural_pct_of_total.abs());
    if max_metric <= 0.5 {
        CliffBand::Green
    } else if max_metric <= 2.0 {
        CliffBand::Yellow
    } else {
        CliffBand::Red
    }
}

fn run_density_sweep(
    label: &str,
    yuv: &[u8],
    width: i32,
    height: i32,
    n_frames: u32,
) -> Vec<DensityResult> {
    let baseline_h264 = encode_clean(yuv, width, height, n_frames);
    let baseline = walk_annex_b_for_cover_with_options(
        &baseline_h264,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .unwrap_or_else(|e| panic!("{label}: baseline walker: {e:?}"));
    let n_mvd = baseline.mvd_meta.len();
    let safe_mask = analyze_safe_mvd_subset(&baseline.mvd_meta, baseline.mb_w, baseline.mb_h);
    let n_safe = safe_mask.iter().filter(|&&b| b).count();

    let mut results = Vec::with_capacity(DENSITIES.len());
    for &density in DENSITIES {
        if n_safe == 0 {
            results.push(DensityResult {
                density,
                n_mvd_baseline: n_mvd,
                n_safe,
                n_flipped: 0,
                actual_density_of_safe: 0.0,
                actual_density_of_total: 0.0,
                structural_delta: 0,
                n_unintended_outside_set: 0,
                leak_pct_of_total: 0.0,
                cliff_band: CliffBand::Skipped,
            });
            continue;
        }
        let flip_mask = density_flip_mask(&safe_mask, density);
        let n_flipped = flip_mask.iter().filter(|&&b| b).count();
        let actual_density_of_safe = if n_safe > 0 {
            n_flipped as f64 / n_safe as f64
        } else { 0.0 };
        let actual_density_of_total = if n_mvd > 0 {
            n_flipped as f64 / n_mvd as f64
        } else { 0.0 };

        let flipped_h264 =
            encode_with_mvd_overrides(yuv, width, height, n_frames, &flip_mask);
        let flipped = walk_annex_b_for_cover_with_options(
            &flipped_h264,
            WalkOptions { record_mvd: true, ..Default::default() },
        )
        .unwrap_or_else(|e| panic!("{label}: flipped walker @{density}: {e:?}"));
        let structural_delta = flipped.mvd_meta.len() as i64 - baseline.mvd_meta.len() as i64;

        let common_bits = baseline
            .cover
            .mvd_sign_bypass
            .bits
            .len()
            .min(flipped.cover.mvd_sign_bypass.bits.len());
        let mut n_unintended = 0usize;
        for i in 0..common_bits {
            let b = baseline.cover.mvd_sign_bypass.bits[i];
            let f = flipped.cover.mvd_sign_bypass.bits[i];
            let in_flip_set = flip_mask.get(i).copied().unwrap_or(false);
            if !in_flip_set && f != b {
                n_unintended += 1;
            }
        }
        let leak_pct = if n_mvd > 0 {
            100.0 * n_unintended as f64 / n_mvd as f64
        } else { 0.0 };
        let structural_pct = if n_mvd > 0 {
            100.0 * structural_delta as f64 / n_mvd as f64
        } else { 0.0 };
        let cliff_band = classify(leak_pct, structural_pct);

        results.push(DensityResult {
            density,
            n_mvd_baseline: n_mvd,
            n_safe,
            n_flipped,
            actual_density_of_safe,
            actual_density_of_total,
            structural_delta,
            n_unintended_outside_set: n_unintended,
            leak_pct_of_total: leak_pct,
            cliff_band,
        });
    }
    results
}

fn report_fixture(label: &str, results: &[DensityResult]) {
    if results.is_empty() {
        eprintln!("  {label}: no results (fixture missing or 0 MVD positions)");
        return;
    }
    let n_mvd = results[0].n_mvd_baseline;
    let n_safe = results[0].n_safe;
    eprintln!(
        "  {label} (n_mvd={n_mvd}, safe={n_safe} = {:.1}%):",
        if n_mvd > 0 { 100.0 * n_safe as f64 / n_mvd as f64 } else { 0.0 }
    );
    eprintln!(
        "    {:>7}  {:>10}  {:>8}  {:>9}  {:>10}  {:>10}  {}",
        "target", "flipped", "act/safe", "act/total", "Δstruct%", "leak%", "band"
    );
    for r in results {
        let band = match r.cliff_band {
            CliffBand::Green => "GREEN",
            CliffBand::Yellow => "YELLOW",
            CliffBand::Red => "RED",
            CliffBand::Skipped => "skipped",
        };
        let struct_pct = if r.n_mvd_baseline > 0 {
            100.0 * r.structural_delta as f64 / r.n_mvd_baseline as f64
        } else { 0.0 };
        eprintln!(
            "    {:>6.1}%  {:>10}  {:>7.2}%  {:>8.2}%  {:>+9.2}  {:>9.2}  {}",
            r.density * 100.0,
            r.n_flipped,
            r.actual_density_of_safe * 100.0,
            r.actual_density_of_total * 100.0,
            struct_pct,
            r.leak_pct_of_total,
            band,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests — each fixture is a separate test so they isolate cleanly.
// ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "slow research measurement: density sweep on synth motion"]
fn density_sweep_synth_motion() {
    let _g = SESSION_GUARD.lock().unwrap();
    eprintln!("\n=== #493.0b density sweep — synth sinusoidal motion ===");
    eprintln!("Fixture: 320×240 × 12 frames, QP={QP}, GOP={GOP_SIZE}");
    let yuv = synth_yuv_sinusoidal_motion(320, 240, 12);
    let results = run_density_sweep("synth_motion", &yuv, 320, 240, 12);
    report_fixture("synth_motion", &results);
}

#[test]
#[ignore = "slow research measurement: density sweep on corpus fixtures"]
fn density_sweep_real_corpus() {
    let _g = SESSION_GUARD.lock().unwrap();
    eprintln!("\n=== #493.0b density sweep — real corpus 1080p × 10f ===");
    eprintln!("Fixtures expected at /tmp/openh264_baseline_*_f10.yuv");
    eprintln!("(regen via core/test-vectors/video/h264/real-world/source/regen_openh264_baseline.sh)\n");

    let fixtures: &[(&str, i32, i32)] = &[
        ("carplane",  1072, 1920),
        ("img4138",   1920, 1072),
        ("dji_mini2", 1920, 1072),
    ];
    let n_frames = 10u32;
    let mut any = false;
    for &(tag, width, height) in fixtures {
        let path = std::path::PathBuf::from(format!(
            "/tmp/openh264_baseline_{tag}_{width}x{height}_f{n_frames}.yuv"
        ));
        let yuv = match read_yuv(&path) {
            Some(b) => b,
            None => {
                eprintln!("  {tag}: SKIP (fixture missing: {})", path.display());
                continue;
            }
        };
        let expected = (width as usize) * (height as usize) * 3 / 2 * (n_frames as usize);
        if yuv.len() < expected {
            eprintln!("  {tag}: SKIP (YUV too short)");
            continue;
        }
        any = true;
        let results = run_density_sweep(tag, &yuv, width, height, n_frames);
        report_fixture(tag, &results);
        eprintln!();
    }
    if !any {
        eprintln!("(no real-corpus fixtures available)");
    }
}

#[test]
#[ignore = "slow research measurement: density sweep cliff recommendation"]
fn density_sweep_recommendation_summary() {
    let _g = SESSION_GUARD.lock().unwrap();
    eprintln!("\n=== #493.0b CostWeights.mvd_sign recommendation summary ===");
    eprintln!();
    eprintln!("Per locked decision (3): cost-weighted STC discovers allocation;");
    eprintln!("we set the weight to keep STC's natural utilisation in the GREEN");
    eprintln!("band with >2× safety margin. Highest GREEN density across fixtures");
    eprintln!("→ /2 → STC operating point → solve for weight against CoeffSign=1.");
    eprintln!();
    eprintln!("This test is a stub for the recommendation synthesis — actual");
    eprintln!("recommendation must be derived from the per-fixture cliff data");
    eprintln!("produced by density_sweep_synth_motion + density_sweep_real_corpus.");
    eprintln!();
    eprintln!("Run those two tests first, then update");
    eprintln!("  docs/design/video/h264/d07-streaming-4domain.md");
    eprintln!("'CostWeights v1.1 ship defaults' table with the measured weight.");
}
