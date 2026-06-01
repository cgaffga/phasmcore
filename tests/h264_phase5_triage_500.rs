// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase 5 (#500) triage harness — per-domain stealth A/B between
// CS-only baseline and 4-domain test, both via the streaming-v2
// production entry point.
//
// **The comparison Phase 5 actually needs.**
//
// The pre-#493 ship was CS-only on the streaming path. #493 wired the
// other 3 domains (CSL + MVDs + MVDsl) into the same streaming
// orchestrator under a unified `StealthAllocator`. The Phase 5 design
// doc (`docs/design/video/h264/d07-streaming-4domain.md` §"Phase 5
// measurement protocol detail") calls this a HARD SHIP GATE: 4-domain
// per-domain stats must be no worse than CS-only at every (payload,
// content) cell, otherwise we re-tune `StealthAllocator` or escalate.
//
// **CS-only baseline mechanism (no API churn).**
//
// `StealthAllocator::v1_default()` honours `PHASM_STEALTH_ABLATE=cs`
// by zeroing the other 3 domain weights, forcing all message bits to
// route through CoeffSign. On the wire this is byte-equivalent to the
// pre-#493 CS-only ship: CSL / MVDs / MVDsl positions are still
// enrolled in the cover (so a 4-domain-aware classifier sees natural
// statistics there), but STC never picks them for flipping.
//
// **What we measure.**
//
// Per-domain on the stego output (no clean reference — see #511 for the
// pre-existing clean-vs-stego cover-shape regression that's orthogonal
// to Phase 5):
//
// - `n_total` — domain cover-bit count.
// - `sign_balance` — P(bit=0). Natural CABAC bypass ≈ 0.5; stego flips
//   preserve this only if STC syndrome bits are themselves balanced.
// - `shannon_entropy` — base-2 entropy of the cover bit stream. 1.0
//   bit is uniform; any drift below is a marginal-distribution
//   fingerprint.
//
// **Pass criterion (this harness — triage only).**
//
// Both runs round-trip green, 4-domain run actually exercises MVD
// domains (`n_flipped > 0` in MVDs + MVDsl by construction since they
// have nonzero allocator weight), per-domain entropies stay within
// `EPS = 0.02` bits of the CS-only baseline. The hard ship gate
// (#500.3) uses tighter thresholds + full corpus + multiple payload
// sizes; this triage is just an early-signal pass on one fixture.
//
// **Knob.** `PHASM_PHASE5_PAYLOAD_BYTES` overrides the payload length
// (defaults to 32 bytes — small enough to fit the 128×80×10f fixture's
// capacity at either allocator).

#![cfg(feature = "cabac-stego")]

use phasm_core::{
    h264_stego_encode_yuv_string_4domain_multigop_streaming_v2,
    h264_stego_smart_decode_video,
};
use phasm_core::codec::h264::cabac::bin_decoder::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};

fn load_yuv(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/video/h264/real-world/{name}"))
        .unwrap_or_else(|_| panic!("missing fixture: {name}"))
}

/// Shannon entropy of a bit stream (base-2). Uniform 50/50 → 1.0.
fn shannon_entropy(bits: &[u8]) -> f64 {
    if bits.is_empty() {
        return 0.0;
    }
    let mut n0 = 0u64;
    let mut n1 = 0u64;
    for &b in bits {
        match b {
            0 => n0 += 1,
            1 => n1 += 1,
            _ => panic!("non-bit value: {b}"),
        }
    }
    let n = bits.len() as f64;
    let p0 = n0 as f64 / n;
    let p1 = n1 as f64 / n;
    let term = |p: f64| if p > 0.0 { -p * p.log2() } else { 0.0 };
    term(p0) + term(p1)
}

/// P(bit=0). Natural CABAC bypass ≈ 0.5.
fn sign_balance(bits: &[u8]) -> f64 {
    if bits.is_empty() {
        return 0.5;
    }
    let n0 = bits.iter().filter(|&&b| b == 0).count();
    n0 as f64 / bits.len() as f64
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PerDomainStats {
    n_total: usize,
    sign_balance: f64,
    entropy: f64,
}

impl PerDomainStats {
    fn from_bits(bits: &[u8]) -> Self {
        Self {
            n_total: bits.len(),
            sign_balance: sign_balance(bits),
            entropy: shannon_entropy(bits),
        }
    }
}

#[derive(Debug)]
struct StealthSnapshot {
    cs: PerDomainStats,
    csl: PerDomainStats,
    mvds: PerDomainStats,
    mvdsl: PerDomainStats,
    bitstream_len: usize,
}

fn snapshot_stealth(stego: &[u8]) -> StealthSnapshot {
    let opts = WalkOptions { record_mvd: true, record_offsets: false };
    let walk = walk_annex_b_for_cover_with_options(stego, opts)
        .expect("walk_annex_b_for_cover_with_options");
    StealthSnapshot {
        cs: PerDomainStats::from_bits(&walk.cover.coeff_sign_bypass.bits),
        csl: PerDomainStats::from_bits(&walk.cover.coeff_suffix_lsb.bits),
        mvds: PerDomainStats::from_bits(&walk.cover.mvd_sign_bypass.bits),
        mvdsl: PerDomainStats::from_bits(&walk.cover.mvd_suffix_lsb.bits),
        bitstream_len: stego.len(),
    }
}

/// Set / unset `PHASM_STEALTH_ABLATE` and return its prior value.
/// SAFETY: env vars are process-global. Must run with
/// `--test-threads=1`.
fn set_ablate(ablate_cs_only: bool) {
    unsafe {
        if ablate_cs_only {
            std::env::set_var("PHASM_STEALTH_ABLATE", "cs");
        } else {
            std::env::remove_var("PHASM_STEALTH_ABLATE");
        }
    }
}

fn clear_ablate() {
    unsafe {
        std::env::remove_var("PHASM_STEALTH_ABLATE");
    }
}

/// Encode + decode in one call, with `PHASM_STEALTH_ABLATE` set
/// consistently across BOTH operations. The decoder also reads
/// `StealthAllocator::v1_default()` (see
/// `core/src/codec/h264/stego/decode_pixels.rs:225`), so if we ablate
/// the encoder but not the decoder the per-domain bit allocations
/// disagree → syndrome mismatch → `FrameCorrupted` at payload parse.
fn encode_and_decode(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    message: &str,
    passphrase: &str,
    ablate_cs_only: bool,
) -> (Vec<u8>, String) {
    set_ablate(ablate_cs_only);
    let stego = h264_stego_encode_yuv_string_4domain_multigop_streaming_v2(
        yuv, width, height, n_frames, gop_size, message, passphrase,
    )
    .expect("streaming-v2 encode");
    let recovered = h264_stego_smart_decode_video(&stego, passphrase)
        .expect("smart_decode");
    clear_ablate();
    (stego, recovered)
}

/// Build a deterministic ASCII payload of the given length. Printable
/// content is irrelevant for stealth (the STC syndrome is encrypted
/// and effectively random).
fn make_message(payload_bytes: usize) -> String {
    (0..payload_bytes)
        .map(|i| (b'a' + (i % 26) as u8) as char)
        .collect()
}

/// One A/B run + table print + entropy-delta check. Returns
/// `max |dH|` across the 4 domains.
fn run_triage(
    fixture_label: &str,
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    payload_bytes: usize,
    passphrase: &str,
) -> f64 {
    let message = make_message(payload_bytes);
    eprintln!("\n=== Phase 5 triage: {fixture_label}, {width}×{height}×{n_frames}f, payload={payload_bytes}B ===");

    let (stego_4d, recovered_4d) = encode_and_decode(
        yuv, width, height, n_frames, gop_size, &message, passphrase, false,
    );
    assert_eq!(recovered_4d, message, "{fixture_label}/{payload_bytes}B 4D round-trip");
    let snap_4d = snapshot_stealth(&stego_4d);

    let (stego_cs, recovered_cs) = encode_and_decode(
        yuv, width, height, n_frames, gop_size, &message, passphrase, true,
    );
    assert_eq!(recovered_cs, message, "{fixture_label}/{payload_bytes}B CS round-trip");
    let snap_cs = snapshot_stealth(&stego_cs);

    eprintln!("bitstream len: 4D={}B  CS={}B  delta={:+}B",
        snap_4d.bitstream_len, snap_cs.bitstream_len,
        snap_4d.bitstream_len as i64 - snap_cs.bitstream_len as i64);

    eprintln!("{:<24} {:>10} {:>10}   {:>10} {:>10}   {:>10} {:>10}",
        "domain", "n_4D", "n_CS", "bal_4D", "bal_CS", "H_4D", "H_CS");
    eprintln!("{}", "-".repeat(96));
    let rows = [
        ("coeff_sign", &snap_4d.cs, &snap_cs.cs),
        ("coeff_suffix_lsb", &snap_4d.csl, &snap_cs.csl),
        ("mvd_sign", &snap_4d.mvds, &snap_cs.mvds),
        ("mvd_suffix_lsb", &snap_4d.mvdsl, &snap_cs.mvdsl),
    ];
    for (name, a, b) in &rows {
        eprintln!("{:<24} {:>10} {:>10}   {:>10.4} {:>10.4}   {:>10.5} {:>10.5}",
            name, a.n_total, b.n_total,
            a.sign_balance, b.sign_balance,
            a.entropy, b.entropy);
    }

    assert_eq!(snap_4d.cs.n_total, snap_cs.cs.n_total, "{fixture_label} CS cover length");
    assert_eq!(snap_4d.csl.n_total, snap_cs.csl.n_total, "{fixture_label} CSL cover length");
    assert_eq!(snap_4d.mvds.n_total, snap_cs.mvds.n_total, "{fixture_label} MVDs cover length");
    assert_eq!(snap_4d.mvdsl.n_total, snap_cs.mvdsl.n_total, "{fixture_label} MVDsl cover length");

    let domain_deltas = [
        ("coeff_sign", snap_4d.cs.entropy - snap_cs.cs.entropy),
        ("coeff_suffix_lsb", snap_4d.csl.entropy - snap_cs.csl.entropy),
        ("mvd_sign", snap_4d.mvds.entropy - snap_cs.mvds.entropy),
        ("mvd_suffix_lsb", snap_4d.mvdsl.entropy - snap_cs.mvdsl.entropy),
    ];
    let max_dh = domain_deltas.iter().map(|(_, d)| d.abs()).fold(0.0f64, f64::max);
    eprintln!("max |dH| = {:.5}", max_dh);
    max_dh
}

// STEGO.B.P3 (2026-05-23): the per-domain A/B mechanism this test
// exercised (PHASM_STEALTH_ABLATE=cs zeroing other 3 domain weights)
// applied to Scheme B's StealthAllocator. After P3, streaming_v2 is
// Scheme A (single combined STC, no per-domain allocator). The
// 128×80×10 fixture also hits Scheme A's stricter combined-cover
// capacity (32-byte payload triggers MessageTooLarge). Stealth
// measurement at production fixture sizes is in
// `h264_stego_stealth_measurement`.
#[test]
#[ignore = "STEGO.B.P3: Scheme B per-domain A/B no longer applicable; PHASM_STEALTH_ABLATE removed in spirit"]
fn phase5_triage_4domain_vs_cs_only_128x80_10f() {
    let yuv = load_yuv("img4138_128x80_f10.yuv");
    let payload_bytes = std::env::var("PHASM_PHASE5_PAYLOAD_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32);
    let max_dh = run_triage(
        "img4138_128x80_f10",
        &yuv, 128, 80, 10, 5, payload_bytes, "phase5-triage-500",
    );
    let eps: f64 = 0.02;
    eprintln!("\ntriage status: max |dH| = {:.5} (EPS = {:.5}) — {}",
        max_dh, eps,
        if max_dh <= eps { "GREEN" } else { "YELLOW (within #500.3 scope)" });
}

/// 1080p sweep across 4 representative fixtures × 3 payload sizes.
/// Fixtures live in `/tmp` (generated by `regen_real_world_histograms.sh`
/// and friends — not committed to the repo). Marked `#[ignore]` because
/// the fixtures are local-only and a single full run takes ~15-30 min.
///
/// Coverage:
///   - img4138 (iPhone, mixed-motion still-life) — the canonical fixture
///   - dji_mini2 (drone, smooth-motion 2.7K downscaled to 1080p)
///   - carplane (Artlist, complex motion + detail)
///   - lumix_g9 (mirrorless camera, varied content)
///
/// Payload tier: 100B / 500B / 1KB — matches Phase 5 protocol cells.
/// 4KB tier is included in #500.3 full gate; this triage stops at 1KB
/// to keep runtime under 15 minutes per cycle.
#[test]
#[ignore = "needs /tmp 1080p YUV fixtures + ~15 min; run with --include-ignored"]
fn phase5_triage_4domain_vs_cs_only_1080p_sweep() {
    // (width, height, n_frames, file_path, label)
    // Note: the /tmp fixture set encodes at 1920x1072 (16-aligned crop
    // of the source 1920×1080). Matches the rest of the test harness.
    let fixtures = [
        (1920, 1072, 10, "/tmp/openh264_baseline_img4138_1920x1072_f10.yuv", "img4138_1080p_f10"),
        (1920, 1072, 10, "/tmp/openh264_baseline_dji_mini2_1920x1072_f10.yuv", "dji_mini2_1080p_f10"),
        (1072, 1920, 10, "/tmp/openh264_baseline_carplane_1072x1920_f10.yuv", "carplane_portrait_f10"),
        (1920, 1072, 10, "/tmp/openh264_baseline_lumix_g9_1080p_1920x1072_f10.yuv", "lumix_g9_1080p_f10"),
    ];
    let payloads = [100usize, 500, 1024];
    let gop_size = 5usize;
    let passphrase = "phase5-triage-500-1080p";

    let mut all_max_dh = Vec::new();
    for &(width, height, n_frames, path, label) in &fixtures {
        let Ok(yuv) = std::fs::read(path) else {
            eprintln!("=== SKIP {label}: {path} not found ===");
            continue;
        };
        let expected_len = (width as usize * height as usize * 3 / 2) * n_frames;
        if yuv.len() != expected_len {
            eprintln!("=== SKIP {label}: size mismatch {} vs expected {expected_len} ===",
                yuv.len());
            continue;
        }
        for &payload_bytes in &payloads {
            let max_dh = run_triage(
                label, &yuv, width, height, n_frames, gop_size,
                payload_bytes, passphrase,
            );
            all_max_dh.push((label, payload_bytes, max_dh));
        }
    }

    eprintln!("\n=== Phase 5 1080p triage summary ===");
    eprintln!("{:<32} {:>8} {:>12}", "fixture", "payload", "max |dH|");
    eprintln!("{}", "-".repeat(56));
    let eps: f64 = 0.02;
    let mut yellow_cells = 0usize;
    for (label, payload, max_dh) in &all_max_dh {
        let flag = if *max_dh > eps { " ⚠" } else { "" };
        eprintln!("{:<32} {:>8} {:>12.5}{}", label, payload, max_dh, flag);
        if *max_dh > eps { yellow_cells += 1; }
    }
    eprintln!("\n{} cells measured, {} yellow (> EPS = {:.5})",
        all_max_dh.len(), yellow_cells, eps);
    eprintln!("triage status: {}",
        if yellow_cells == 0 { "GREEN" } else { "YELLOW (within #500.3 scope)" });
}
