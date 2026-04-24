// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! H.264 Phase 3c cross-domain stealth retrospective.
//!
//! Takes a cover and a stego MP4 (same dimensions, same encoder, same
//! bitstream layout — i.e. the output of `h264_ghost_encode` applied to the
//! cover). Re-parses both through the real encode/decode scanner, pairs up
//! positions, and reports where the flips landed and whether they perturb
//! first-order statistics.
//!
//! This is an exploratory measurement tool — it exists to answer "did the
//! Phase 3c cross-domain split actually spread the payload, or is one domain
//! soaking it all up?" Not a production path.
//!
//! Usage:
//! ```
//! cargo run --features video --release --example h264_stealth_retrospective -- \
//!     cover.mp4 stego.mp4
//! ```

use std::env;
use std::fs;

use phasm_core::codec::h264::cavlc::{EmbedDomain, EmbeddablePosition};
use phasm_core::codec::mp4;
use phasm_core::stego::video::h264_pipeline::scan_frames_for_stealth_analysis;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <cover.mp4> <stego.mp4>", args[0]);
        std::process::exit(2);
    }
    let cover_path = &args[1];
    let stego_path = &args[2];

    let cover = fs::read(cover_path).expect("read cover");
    let stego = fs::read(stego_path).expect("read stego");

    if cover.len() != stego.len() {
        eprintln!(
            "WARN: sizes differ ({} vs {}) — positions may not align byte-for-byte",
            cover.len(),
            stego.len()
        );
    }

    println!("==== H.264 Phase 3c stealth retrospective ====");
    println!("Cover: {cover_path} ({} bytes)", cover.len());
    println!("Stego: {stego_path} ({} bytes)", stego.len());
    println!();

    // Sample offsets (frame_idx -> byte in full MP4 buffer). Phase 3c preserves
    // layout so cover and stego offsets should match byte-for-byte; we load
    // them from the cover and fall back to stego-read only for sanity checks.
    let cover_offsets = sample_offsets(&cover);
    let stego_offsets = sample_offsets(&stego);
    assert_eq!(cover_offsets, stego_offsets, "Phase 3c should preserve MP4 layout");

    // Parse both. Positions are deterministic from the bitstream, but we
    // re-parse the stego independently because the MVD path's `coeff_value`
    // field encodes the *post-flip* MVD value — we need both the cover- and
    // stego-side parses to bin MVD magnitude distributions honestly.
    let (cover_positions, _, _) =
        scan_frames_for_stealth_analysis(&cover).expect("parse cover");
    let (stego_positions, _, _) =
        scan_frames_for_stealth_analysis(&stego).expect("parse stego");

    println!("Cover positions: {}", cover_positions.len());
    println!("Stego positions: {}", stego_positions.len());
    if cover_positions.len() != stego_positions.len() {
        eprintln!("WARN: position counts differ — flips may have shifted codeword length");
    }
    let n = cover_positions.len().min(stego_positions.len());

    // =========================================================================
    // 1. Flip counts per domain
    // =========================================================================
    #[derive(Default)]
    struct DomainStats {
        positions: usize,
        flips: usize,
    }
    let mut t1_sign = DomainStats::default();
    let mut lvl_mag = DomainStats::default();
    let mut lvl_sign = DomainStats::default();
    let mut mvd_lsb = DomainStats::default();

    for i in 0..n {
        let cpos = &cover_positions[i];
        let spos = &stego_positions[i];
        let cbit = read_bit(&cover, cpos, &cover_offsets);
        let sbit = read_bit(&stego, spos, &stego_offsets);
        let flipped = cbit != sbit;
        let bucket = match cpos.domain {
            EmbedDomain::T1Sign => &mut t1_sign,
            EmbedDomain::LevelSuffixMag => &mut lvl_mag,
            EmbedDomain::LevelSuffixSign => &mut lvl_sign,
            EmbedDomain::MvdLsb => &mut mvd_lsb,
        };
        bucket.positions += 1;
        if flipped {
            bucket.flips += 1;
        }
    }

    let total_flips =
        t1_sign.flips + lvl_mag.flips + lvl_sign.flips + mvd_lsb.flips;
    println!();
    println!("---- Flips per domain ----");
    println!(
        "  {:<18}  {:>10}  {:>8}  {:>8}  {:>10}",
        "domain", "pool", "flips", "%total", "%of-pool"
    );
    for (name, s) in [
        ("T1Sign", &t1_sign),
        ("LevelSuffixMag", &lvl_mag),
        ("LevelSuffixSign", &lvl_sign),
        ("MvdLsb", &mvd_lsb),
    ] {
        let pct_total = pct(s.flips, total_flips);
        let pct_pool = pct(s.flips, s.positions);
        println!(
            "  {:<18}  {:>10}  {:>8}  {:>7.2}%  {:>9.4}%",
            name, s.positions, s.flips, pct_total, pct_pool
        );
    }
    println!("  {:<18}  {:>10}  {:>8}", "TOTAL", n, total_flips);

    // Phase 3c cover-pool partition (mirrors pipeline filter):
    let coeff_pool = t1_sign.positions + lvl_mag.positions + lvl_sign.positions;
    let mvd_pool = mvd_lsb.positions;
    let coeff_flips = t1_sign.flips + lvl_mag.flips + lvl_sign.flips;
    let mvd_flips = mvd_lsb.flips;
    println!();
    println!("---- Cross-domain split (Phase 3c) ----");
    println!(
        "  coeff pool size: {coeff_pool}   mvd pool size: {mvd_pool}   (ratio {:.2} : 1)",
        coeff_pool as f64 / mvd_pool.max(1) as f64,
    );
    println!(
        "  expected coeff share of flips (pool-proportional): {:.2}%",
        100.0 * coeff_pool as f64 / (coeff_pool + mvd_pool).max(1) as f64,
    );
    println!(
        "  observed coeff flips: {coeff_flips} ({:.2}%)   observed mvd flips: {mvd_flips} ({:.2}%)",
        pct(coeff_flips, total_flips),
        pct(mvd_flips, total_flips),
    );

    // =========================================================================
    // 2. T1 sign bit balance (cover + stego) — expected 50/50
    // =========================================================================
    let (t1_c0, t1_c1, t1_s0, t1_s1) = bit_balance(
        &cover,
        &stego,
        &cover_positions,
        &stego_positions,
        &cover_offsets,
        &stego_offsets,
        n,
        EmbedDomain::T1Sign,
    );
    println!();
    println!("---- T1 sign bit balance (expected ~50/50) ----");
    report_balance("cover", t1_c0, t1_c1);
    report_balance("stego", t1_s0, t1_s1);

    // =========================================================================
    // 3. MVD LSB bit balance — expected 50/50 (codeNum signed mapping is
    //    symmetric once you condition on codeNum >= 1).
    // =========================================================================
    let (m_c0, m_c1, m_s0, m_s1) = bit_balance(
        &cover,
        &stego,
        &cover_positions,
        &stego_positions,
        &cover_offsets,
        &stego_offsets,
        n,
        EmbedDomain::MvdLsb,
    );
    println!();
    println!("---- MVD LSB bit balance (expected ~50/50) ----");
    report_balance("cover", m_c0, m_c1);
    report_balance("stego", m_s0, m_s1);

    // =========================================================================
    // 4. MVD |value| distribution: cover vs stego goodness-of-fit
    // =========================================================================
    let bins = &[
        (0, 0, "0"),
        (1, 1, "1"),
        (2, 2, "2"),
        (3, 4, "3-4"),
        (5, 8, "5-8"),
        (9, 16, "9-16"),
        (17, i32::MAX, "17+"),
    ];
    let mut cover_bins = vec![0u64; bins.len()];
    let mut stego_bins = vec![0u64; bins.len()];
    for i in 0..n {
        if cover_positions[i].domain != EmbedDomain::MvdLsb {
            continue;
        }
        let cv = cover_positions[i].coeff_value.abs();
        let sv = stego_positions[i].coeff_value.abs();
        if let Some(bi) = bin_idx(cv, bins) {
            cover_bins[bi] += 1;
        }
        if let Some(bi) = bin_idx(sv, bins) {
            stego_bins[bi] += 1;
        }
    }
    println!();
    println!("---- MVD |value| distribution (cover vs stego) ----");
    println!(
        "  {:<8}  {:>10}  {:>10}  {:>10}",
        "bin", "cover", "stego", "delta"
    );
    let tc: u64 = cover_bins.iter().sum();
    let ts: u64 = stego_bins.iter().sum();
    for (i, (_, _, label)) in bins.iter().enumerate() {
        let c = cover_bins[i];
        let s = stego_bins[i];
        let delta = s as i64 - c as i64;
        println!(
            "  {:<8}  {:>10}  {:>10}  {:>+10}",
            label, c, s, delta
        );
    }
    println!("  {:<8}  {:>10}  {:>10}", "total", tc, ts);

    // χ² goodness-of-fit: treat (scaled) cover distribution as the expected,
    // stego as observed. Only bins with expected > 0.5 contribute; a bin with
    // 0 cover but non-zero stego would be an infinite chi-square, so we
    // conservatively skip it and warn.
    let mut chi2 = 0.0f64;
    let mut df_active = 0usize;
    let scale = ts as f64 / tc.max(1) as f64;
    let mut skipped = 0usize;
    for i in 0..bins.len() {
        let expected = cover_bins[i] as f64 * scale;
        let observed = stego_bins[i] as f64;
        if expected > 0.5 {
            let d = observed - expected;
            chi2 += d * d / expected;
            df_active += 1;
        } else if observed > 0.0 {
            skipped += 1;
        }
    }
    let df = df_active.saturating_sub(1);
    println!(
        "  chi^2 = {:.4}   df = {}   (skipped {} empty-expected bins)",
        chi2, df, skipped
    );

    // =========================================================================
    // 5. chi² on the bit-balance tests themselves (expected 50/50)
    // =========================================================================
    println!();
    println!("---- chi^2 vs expected 50/50 (1 df) ----");
    println!(
        "  T1Sign cover: chi^2 = {:.4}   stego: chi^2 = {:.4}",
        chi2_half(t1_c0, t1_c1),
        chi2_half(t1_s0, t1_s1),
    );
    println!(
        "  MvdLsb cover: chi^2 = {:.4}   stego: chi^2 = {:.4}",
        chi2_half(m_c0, m_c1),
        chi2_half(m_s0, m_s1),
    );
    println!(
        "  (critical values at 1 df: 3.84 at p=0.05, 10.83 at p=0.001)"
    );

    // =========================================================================
    // 6. MVD flip-position diagnostics: what magnitudes get flipped, and do
    //    the stego-side magnitudes actually change?
    // =========================================================================
    let mut mvd_flip_same_mag = 0usize;
    let mut mvd_flip_diff_mag = 0usize;
    let mut mvd_flip_magnitude_hist: Vec<(i32, i32)> = Vec::new();
    for i in 0..n {
        if cover_positions[i].domain != EmbedDomain::MvdLsb {
            continue;
        }
        let cb = read_bit(&cover, &cover_positions[i], &cover_offsets);
        let sb = read_bit(&stego, &stego_positions[i], &stego_offsets);
        if cb == sb {
            continue;
        }
        let cv = cover_positions[i].coeff_value;
        let sv = stego_positions[i].coeff_value;
        mvd_flip_magnitude_hist.push((cv, sv));
        if cv.abs() == sv.abs() {
            mvd_flip_same_mag += 1;
        } else {
            mvd_flip_diff_mag += 1;
        }
    }
    println!();
    println!("---- MVD flip diagnostics ----");
    println!(
        "  MVD flips that preserve |value|: {mvd_flip_same_mag}"
    );
    println!(
        "  MVD flips that change    |value|: {mvd_flip_diff_mag}"
    );
    println!("  first 20 (cover_mvd, stego_mvd) pairs:");
    for (cv, sv) in mvd_flip_magnitude_hist.iter().take(20) {
        println!("    cover={cv:>+5}   stego={sv:>+5}");
    }
}

fn sample_offsets(mp4_bytes: &[u8]) -> Vec<usize> {
    let mp4_file = mp4::demux::demux(mp4_bytes).expect("demux for sample offsets");
    let track_idx = mp4_file.video_track_idx.expect("video track");
    let track = &mp4_file.tracks[track_idx];
    track.samples.iter().map(|s| s.offset as usize).collect()
}

fn read_bit(bytes: &[u8], pos: &EmbeddablePosition, sample_offsets: &[usize]) -> u8 {
    let frame = pos.frame_idx as usize;
    let sample_off = sample_offsets.get(frame).copied().unwrap_or(0);
    let abs = sample_off + pos.raw_byte_offset;
    if abs >= bytes.len() {
        return 0;
    }
    (bytes[abs] >> (7 - pos.bit_offset)) & 1
}

fn bit_balance(
    cover: &[u8],
    stego: &[u8],
    cover_positions: &[EmbeddablePosition],
    stego_positions: &[EmbeddablePosition],
    cover_offsets: &[usize],
    stego_offsets: &[usize],
    n: usize,
    domain: EmbedDomain,
) -> (u64, u64, u64, u64) {
    let mut c0 = 0u64;
    let mut c1 = 0u64;
    let mut s0 = 0u64;
    let mut s1 = 0u64;
    for i in 0..n {
        let cpos = &cover_positions[i];
        if cpos.domain != domain {
            continue;
        }
        let spos = &stego_positions[i];
        let cb = read_bit(cover, cpos, cover_offsets);
        let sb = read_bit(stego, spos, stego_offsets);
        if cb == 0 {
            c0 += 1;
        } else {
            c1 += 1;
        }
        if sb == 0 {
            s0 += 1;
        } else {
            s1 += 1;
        }
    }
    (c0, c1, s0, s1)
}

fn report_balance(label: &str, zeros: u64, ones: u64) {
    let total = zeros + ones;
    if total == 0 {
        println!("  {label}: empty");
        return;
    }
    println!(
        "  {label}: 0s={zeros} ({:.3}%)  1s={ones} ({:.3}%)  total={total}",
        100.0 * zeros as f64 / total as f64,
        100.0 * ones as f64 / total as f64,
    );
}

fn chi2_half(zeros: u64, ones: u64) -> f64 {
    let total = (zeros + ones) as f64;
    if total == 0.0 {
        return 0.0;
    }
    let expected = total / 2.0;
    let d0 = zeros as f64 - expected;
    let d1 = ones as f64 - expected;
    (d0 * d0 + d1 * d1) / expected
}

fn bin_idx(v: i32, bins: &[(i32, i32, &str)]) -> Option<usize> {
    bins.iter().position(|(lo, hi, _)| v >= *lo && v <= *hi)
}

fn pct(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        100.0 * num as f64 / den as f64
    }
}
