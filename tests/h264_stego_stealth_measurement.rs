// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase 6F.2(k).5 — per-domain stealth measurement harness.
//
// For a given real-world fixture and payload, this harness:
//
// 1. Encodes the YUV with stego (the public 4-domain orchestrator).
// 2. Encodes the same YUV WITHOUT stego (clean reference, same
//    encoder config + mode-decision shape — `enable_mvd_stego_hook=
//    true` to keep P_SKIP disabled so the bitstream shape matches).
// 3. Walks both Annex-B streams via `walk_annex_b_for_cover` to
//    extract per-domain bypass-bin sequences.
// 4. Computes per-domain metrics:
//    - cover-bit count (capacity exposed)
//    - modification count (Σ clean_bit ⊕ stego_bit) and rate
//    - Shannon entropy of bits before / after, and the delta
// 5. Asserts:
//    - Per-domain bypass-bin entropy stays uniform within ε
//      (bypass bins are 50/50 random; XOR with random STC syndrome
//      preserves uniformity).
//    - MVD modifications are non-zero (bitstream-mod path active).
//    - MVD share ≤ drift budget × total modifications.
//    - Total modification count > 0 (stego actually injected bits).
//
// Output is printed via `eprintln!` for human review. The harness
// is the v1.0 stealth-fingerprint inspector for Phase 6F.2(k);
// PSNR-via-spec-decoder + SRNet detector eval are tracked as
// follow-ons in deferred-items.md §37.

#![cfg(feature = "cabac-stego")]

use phasm_core::{
    h264_stego_encode_yuv_string_4domain_multigop,
    h264_stego_decode_yuv_string_4domain,
};
use phasm_core::codec::h264::cabac::bin_decoder::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};
use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};

fn load_real_world(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/video/h264/real-world/{name}"))
        .unwrap_or_else(|_| panic!("missing real-world fixture: {name}"))
}

/// Encode YUV with NO stego. Mirrors the §30D-C orchestrator's
/// encoder config + GOP pattern + `enable_mvd_stego_hook=true`
/// (which disables post-ME P_SKIP, keeping mode-decision shape
/// identical to the stego pipeline). With no stego hook installed,
/// no bin overrides happen — the output is the unmodified CABAC
/// bitstream the stego encoder would produce on a zero payload.
///
/// §6E-A.deploy.3 (2026-04-30): the §30D-C orchestrator now emits
/// IBPBP by default (`GopPattern::Ibpbp { gop, b_count: 1 }`), so
/// the clean reference must do the same — otherwise the per-domain
/// cover shapes diverge between clean and stego and stealth-delta
/// measurements are meaningless.
fn encode_clean_reference(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
) -> Vec<u8> {
    use phasm_core::codec::h264::stego::gop_pattern::{
        iter_encode_order, FrameType, GopPattern,
    };
    let frame_size = (width * height * 3 / 2) as usize;
    let quality = Some(26);
    let mut enc = Encoder::new(width, height, quality)
        .expect("clean encoder");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = false;
    enc.enable_mvd_stego_hook = true; // disable P_SKIP for shape parity
    let pattern = GopPattern::Ibpbp { gop: gop_size, b_count: 1 };
    enc.enable_b_frames = pattern.has_b_frames();
    let mut out = Vec::new();
    for meta in iter_encode_order(n_frames, pattern) {
        let d = meta.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match meta.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame).expect("clean i-frame"),
            FrameType::P => enc.encode_p_frame(frame).expect("clean p-frame"),
            FrameType::B => enc.encode_b_frame(frame).expect("clean b-frame"),
        };
        out.extend_from_slice(&bytes);
    }
    out
}

/// Per-domain Shannon entropy of a bit stream (in bits, base-2).
/// Uniform 50/50 → 1.0 bit; constant 0 or 1 → 0.0 bits.
fn shannon_entropy(bits: &[u8]) -> f64 {
    if bits.is_empty() {
        return 0.0;
    }
    let mut n0: usize = 0;
    let mut n1: usize = 0;
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

/// Count positions where `a[i] != b[i]`. Both slices must have
/// the same length (encoder + clean reference produce identical
/// cover shapes by §6F.2(k) construction — no cascade, so cover
/// length is the same).
fn modification_count(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len(), "cover shape mismatch");
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

#[derive(Debug)]
#[allow(dead_code)]
struct DomainMetrics {
    n_total: usize,
    n_flipped: usize,
    h_clean: f64,
    h_stego: f64,
}

impl DomainMetrics {
    fn flip_rate(&self) -> f64 {
        if self.n_total == 0 { 0.0 } else { self.n_flipped as f64 / self.n_total as f64 }
    }
    fn h_delta(&self) -> f64 {
        self.h_stego - self.h_clean
    }
}

fn measure(clean: &[u8], stego: &[u8]) -> DomainMetrics {
    DomainMetrics {
        n_total: clean.len(),
        n_flipped: modification_count(clean, stego),
        h_clean: shannon_entropy(clean),
        h_stego: shannon_entropy(stego),
    }
}

#[test]
fn stealth_measurement_real_world_64x48_5f() {
    // Fixture is the same one §6F.2(k).3+.4 round-trip-validated.
    let yuv = load_real_world("img4138_64x48_f5.yuv");
    let width = 64u32;
    let height = 48u32;
    let n_frames = 5usize;
    let gop_size = 5usize;
    // Same fixture/passphrase combination as the §6F.2(k).3+.4
    // round-trip gate — known to plan + decode successfully.
    let msg = "h";
    let pass = "test-pass-64";

    // 1. Stego encode.
    let stego_bytes = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, width, height, n_frames, gop_size, msg, pass,
    )
    .expect("stego encode 64x48 5f");

    // Round-trip sanity: payload must decode (otherwise our
    // measurements are over a broken bitstream).
    let recovered = h264_stego_decode_yuv_string_4domain(&stego_bytes, pass)
        .expect("stego round-trip");
    assert_eq!(recovered, msg, "round-trip preserves payload");

    // 2. Clean-reference encode.
    let clean_bytes = encode_clean_reference(&yuv, width, height, n_frames, gop_size);

    // 3. Walk both streams. `record_mvd: true` matches the decoder's
    // walker config (decode_pixels.rs) so MVD bins land in the
    // recorded cover.
    let opts = WalkOptions { record_mvd: true };
    let clean_walk = walk_annex_b_for_cover_with_options(&clean_bytes, opts)
        .expect("clean walk");
    let stego_walk = walk_annex_b_for_cover_with_options(&stego_bytes, opts)
        .expect("stego walk");

    // 4. Per-domain metrics.
    let cs = measure(
        &clean_walk.cover.coeff_sign_bypass.bits,
        &stego_walk.cover.coeff_sign_bypass.bits,
    );
    let cl = measure(
        &clean_walk.cover.coeff_suffix_lsb.bits,
        &stego_walk.cover.coeff_suffix_lsb.bits,
    );
    let ms = measure(
        &clean_walk.cover.mvd_sign_bypass.bits,
        &stego_walk.cover.mvd_sign_bypass.bits,
    );
    let ml = measure(
        &clean_walk.cover.mvd_suffix_lsb.bits,
        &stego_walk.cover.mvd_suffix_lsb.bits,
    );

    eprintln!("\n=== §6F.2(k).5 stealth measurement: img4138_64x48_f5.yuv ===");
    eprintln!("payload: {} byte(s)", msg.len());
    eprintln!("\nper-domain metrics (n / flipped / rate / H_clean / H_stego / ΔH):");
    eprintln!("  coeff_sign_bypass : n={:5} flipped={:4} rate={:.4} H_c={:.4} H_s={:.4} dH={:+.5}",
        cs.n_total, cs.n_flipped, cs.flip_rate(), cs.h_clean, cs.h_stego, cs.h_delta());
    eprintln!("  coeff_suffix_lsb  : n={:5} flipped={:4} rate={:.4} H_c={:.4} H_s={:.4} dH={:+.5}",
        cl.n_total, cl.n_flipped, cl.flip_rate(), cl.h_clean, cl.h_stego, cl.h_delta());
    eprintln!("  mvd_sign_bypass   : n={:5} flipped={:4} rate={:.4} H_c={:.4} H_s={:.4} dH={:+.5}",
        ms.n_total, ms.n_flipped, ms.flip_rate(), ms.h_clean, ms.h_stego, ms.h_delta());
    eprintln!("  mvd_suffix_lsb    : n={:5} flipped={:4} rate={:.4} H_c={:.4} H_s={:.4} dH={:+.5}",
        ml.n_total, ml.n_flipped, ml.flip_rate(), ml.h_clean, ml.h_stego, ml.h_delta());

    let total_flipped = cs.n_flipped + cl.n_flipped + ms.n_flipped + ml.n_flipped;
    let mvd_share = if total_flipped > 0 {
        (ms.n_flipped + ml.n_flipped) as f64 / total_flipped as f64
    } else { 0.0 };
    eprintln!("\ntotal modifications: {total_flipped}");
    eprintln!("MVD share of mods : {:.3} (drift budget cap = 0.20)", mvd_share);

    // 5. Assertions.
    assert!(total_flipped > 0,
        "stego encoder must produce at least one bit modification");

    // §6F.2(k) wires MVD-sign bitstream-mod; verify the path is
    // actually exercised. (mvd_suffix_lsb stays disabled — its
    // flip count must be zero by §6F.2(k).2 contract.)
    assert!(ms.n_flipped > 0,
        "MVD-sign bitstream-mod path must be exercised (got 0 flips)");
    assert_eq!(ml.n_flipped, 0,
        "MVD suffix-LSB injection is disabled in v1.0 (cascades); \
         got {} flips", ml.n_flipped);

    // Drift-budget check. Allocator caps MVD share at ≤ 0.20 of
    // M, but the actual fraction of total modifications can drift
    // slightly because STC-syndrome flips on residual coeffs may
    // introduce additional collateral flips per bit (whereas MVD
    // is one flip per planned bit). We allow a generous 0.30 cap
    // here as the operational ceiling — the strict 0.20 is on
    // m_d / M, not on flips / total_flips.
    assert!(mvd_share <= 0.30,
        "MVD modification share {mvd_share:.3} exceeds operational ceiling 0.30");

    // Per-domain entropy preservation: bypass bins are uniform
    // 50/50 by spec; STC flips at random positions preserve that
    // marginal distribution. ΔH within ±0.01 bit is well below
    // any practical detector's discrimination threshold for
    // small-cover fixtures (this 64x48 fixture is at the small-
    // sample noise floor — looser bound here, tighter for larger
    // fixtures in follow-on harness runs).
    let h_eps: f64 = 0.05;
    for (name, m) in [("coeff_sign", &cs), ("coeff_suffix", &cl), ("mvd_sign", &ms)] {
        if m.n_total > 100 {
            assert!(m.h_delta().abs() < h_eps,
                "domain {name}: entropy delta {:.5} exceeds {h_eps} \
                 (clean H={:.4}, stego H={:.4})",
                m.h_delta(), m.h_clean, m.h_stego);
        }
    }
}

#[test]
fn stealth_measurement_real_world_128x80_10f() {
    // Larger fixture — gives the per-domain entropy estimator more
    // samples for tighter statistical bounds.
    let yuv = load_real_world("img4138_128x80_f10.yuv");
    let width = 128u32;
    let height = 80u32;
    let n_frames = 10usize;
    let gop_size = 10usize;
    let msg = "h";
    let pass = "test-pass-128";

    let stego_bytes = h264_stego_encode_yuv_string_4domain_multigop(
        &yuv, width, height, n_frames, gop_size, msg, pass,
    )
    .expect("stego encode 128x80 10f");

    let recovered = h264_stego_decode_yuv_string_4domain(&stego_bytes, pass)
        .expect("stego round-trip");
    assert_eq!(recovered, msg);

    let clean_bytes = encode_clean_reference(&yuv, width, height, n_frames, gop_size);
    let opts = WalkOptions { record_mvd: true };
    let clean_walk = walk_annex_b_for_cover_with_options(&clean_bytes, opts)
        .expect("clean walk");
    let stego_walk = walk_annex_b_for_cover_with_options(&stego_bytes, opts)
        .expect("stego walk");

    let cs = measure(
        &clean_walk.cover.coeff_sign_bypass.bits,
        &stego_walk.cover.coeff_sign_bypass.bits,
    );
    let cl = measure(
        &clean_walk.cover.coeff_suffix_lsb.bits,
        &stego_walk.cover.coeff_suffix_lsb.bits,
    );
    let ms = measure(
        &clean_walk.cover.mvd_sign_bypass.bits,
        &stego_walk.cover.mvd_sign_bypass.bits,
    );
    let ml = measure(
        &clean_walk.cover.mvd_suffix_lsb.bits,
        &stego_walk.cover.mvd_suffix_lsb.bits,
    );

    eprintln!("\n=== §6F.2(k).5 stealth measurement: img4138_128x80_f10.yuv ===");
    eprintln!("payload: \"{}\" ({} bytes)", msg, msg.len());
    eprintln!("\nper-domain metrics (n / flipped / rate / H_clean / H_stego / ΔH):");
    eprintln!("  coeff_sign_bypass : n={:5} flipped={:4} rate={:.4} H_c={:.4} H_s={:.4} dH={:+.5}",
        cs.n_total, cs.n_flipped, cs.flip_rate(), cs.h_clean, cs.h_stego, cs.h_delta());
    eprintln!("  coeff_suffix_lsb  : n={:5} flipped={:4} rate={:.4} H_c={:.4} H_s={:.4} dH={:+.5}",
        cl.n_total, cl.n_flipped, cl.flip_rate(), cl.h_clean, cl.h_stego, cl.h_delta());
    eprintln!("  mvd_sign_bypass   : n={:5} flipped={:4} rate={:.4} H_c={:.4} H_s={:.4} dH={:+.5}",
        ms.n_total, ms.n_flipped, ms.flip_rate(), ms.h_clean, ms.h_stego, ms.h_delta());
    eprintln!("  mvd_suffix_lsb    : n={:5} flipped={:4} rate={:.4} H_c={:.4} H_s={:.4} dH={:+.5}",
        ml.n_total, ml.n_flipped, ml.flip_rate(), ml.h_clean, ml.h_stego, ml.h_delta());

    let total_flipped = cs.n_flipped + cl.n_flipped + ms.n_flipped + ml.n_flipped;
    let mvd_share = if total_flipped > 0 {
        (ms.n_flipped + ml.n_flipped) as f64 / total_flipped as f64
    } else { 0.0 };
    eprintln!("\ntotal modifications: {total_flipped}");
    eprintln!("MVD share of mods : {:.3} (drift budget cap = 0.20)", mvd_share);

    assert!(total_flipped > 0);
    assert!(ms.n_flipped > 0,
        "MVD-sign path must be exercised on real-world content");
    assert_eq!(ml.n_flipped, 0);
    assert!(mvd_share <= 0.30,
        "MVD share {mvd_share:.3} exceeds operational ceiling");

    // Tighter entropy bounds at this fixture size (more samples).
    let h_eps: f64 = 0.02;
    for (name, m) in [("coeff_sign", &cs), ("coeff_suffix", &cl), ("mvd_sign", &ms)] {
        if m.n_total > 100 {
            assert!(m.h_delta().abs() < h_eps,
                "domain {name}: entropy delta {:.5} exceeds {h_eps}",
                m.h_delta());
        }
    }
}
