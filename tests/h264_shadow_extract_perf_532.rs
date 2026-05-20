// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #532 — shadow_extract perf + correctness gate.
//!
//! Two things:
//!
//! 1. **Correctness**: a 1 KB shadow payload must decode. Pre-#532
//!    this was *impossible*: `peek_fdl_from_first_block` read u16 from
//!    a u32 BE plaintext_len → always returned `fdl =
//!    SHADOW_FRAME_OVERHEAD = 48` (header-only) → AES rejected. The
//!    brute-force fallback only scanned `fdl ∈ [48..=k-1] = [48..=250]`
//!    at parity=4, so any `fdl > 250` (plaintext > ~200 B) failed to
//!    decode entirely.
//!
//! 2. **Perf**: decode wall-clock across small/medium/large payloads.
//!    Pre-#532 the typical decode path was a brute-force scan over
//!    every `fdl ∈ [48..=250]`, each calling RS-decode + AES-GCM-SIV
//!    on a guess. Post-#532 the peek path lands the correct `fdl`
//!    first try via the u32 BE prefix (+ the consistency gate inside
//!    `try_single_fdl` rejects wrong-`fdl` brute-force candidates
//!    cheaply, before AES).
//!
//! Run with: `cargo test --release -p phasm-core --features
//! "video,h264-encoder,cabac-stego" --test
//! h264_shadow_extract_perf_532 -- --nocapture`

#![cfg(all(feature = "video", feature = "cabac-stego"))]

use phasm_core::codec::h264::stego::shadow as shadow_h264;
use phasm_core::codec::h264::stego::{DomainCover, DomainBits, EmbedDomain, PositionKey,
    SyntaxPath, Axis, BinKind};
use std::time::Instant;

/// Build a synthetic 4-domain cover with `n_per_domain` positions
/// across all four bypass-bin domains. Bits + position keys derived
/// from a linear-congruential PRNG for determinism.
fn synth_cover(n_per_domain: usize) -> DomainCover {
    let mut cover = DomainCover::default();
    let mut s: u32 = 0xDEAD_BEEF;
    let mut next_key = || {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        s
    };
    let mut push_bits = |bits: &mut DomainBits, domain: EmbedDomain| {
        for _ in 0..n_per_domain {
            let raw = next_key();
            let bit = (raw & 1) as u8;
            let path = SyntaxPath::Mvd {
                list: 0,
                partition: 0,
                axis: Axis::X,
                kind: BinKind::Sign,
            };
            let key = PositionKey::new(
                (raw >> 16) & 0xFF,
                (raw >> 8) & 0xFFFF,
                domain,
                path,
            );
            bits.bits.push(bit);
            bits.positions.push(key);
        }
    };
    push_bits(&mut cover.coeff_sign_bypass, EmbedDomain::CoeffSignBypass);
    push_bits(&mut cover.coeff_suffix_lsb, EmbedDomain::CoeffSuffixLsb);
    push_bits(&mut cover.mvd_sign_bypass, EmbedDomain::MvdSignBypass);
    push_bits(&mut cover.mvd_suffix_lsb, EmbedDomain::MvdSuffixLsb);
    cover
}

/// Run one prepare→inject→extract cycle with `payload_size_bytes`
/// random-looking payload. Returns (extract_wall_clock_ns).
fn run_one(payload_size_bytes: usize, cover_positions_per_domain: usize) -> u128 {
    let mut cover = synth_cover(cover_positions_per_domain);

    // Build a deterministic payload of the requested size.
    let msg: String = (0..payload_size_bytes)
        .map(|i| (b'a' + (i as u8 % 26)) as char)
        .collect();

    let state = shadow_h264::prepare_shadow_all4(
        &cover,
        "perf-test-pass",
        &msg,
        &[],
        4, // smallest parity tier — encoder's natural default
    )
    .expect("prepare_shadow_all4");

    // Inject into all 4 domain bit arrays.
    shadow_h264::embed_shadow_lsb_all4(
        &mut cover.coeff_sign_bypass.bits,
        &mut cover.coeff_suffix_lsb.bits,
        &mut cover.mvd_sign_bypass.bits,
        &mut cover.mvd_suffix_lsb.bits,
        &state,
    );

    // Extract — time only this.
    let t0 = Instant::now();
    let recovered = shadow_h264::shadow_extract_all4(&cover, "perf-test-pass")
        .expect("shadow_extract_all4 must succeed (#532 correctness)");
    let elapsed = t0.elapsed().as_nanos();

    assert_eq!(recovered.text, msg, "round-trip text must match");
    elapsed
}

/// Decode time on no-shadow cover (worst case — brute-force scans
/// every fdl across every parity tier).
fn run_no_shadow(cover_positions_per_domain: usize) -> u128 {
    let cover = synth_cover(cover_positions_per_domain);
    let t0 = Instant::now();
    let r = shadow_h264::shadow_extract_all4(&cover, "perf-test-pass");
    let elapsed = t0.elapsed().as_nanos();
    assert!(r.is_err(), "no shadow → must fail to extract");
    elapsed
}

/// #532 perf table — payload-size sweep.
///
/// Pre-#532, the 1024 B + 4096 B rows would NOT decode at all
/// (`fdl > 250`, brute-force unreachable). The test deliberately
/// includes those sizes to assert correctness as part of the gate.
#[test]
fn shadow_extract_perf_sweep() {
    // Sized to comfortably fit the 4 KB payload (12k bits/payload @
    // parity=4 ≈ 33 k cover bits needed; 50 k × 4 = 200 k available).
    let cover_n = 50_000;

    println!("\n=== #532 shadow_extract_all4 perf sweep ===");
    println!("Cover: {cover_n} positions per domain × 4 domains = {} bits", cover_n * 4);
    println!();
    println!("{:>14}  {:>12}  notes", "payload bytes", "extract µs");
    println!("{:-^14}  {:-^12}  -----", "", "");

    // Smallest payload (well within brute-force range pre-#532).
    let t12 = run_one(12, cover_n);
    println!("{:>14}  {:>12.1}  small (fdl ≈ 60)",
        12, t12 as f64 / 1000.0);

    // Medium payload — still within pre-#532 brute-force range.
    let t256 = run_one(256, cover_n);
    println!("{:>14}  {:>12.1}  medium (fdl ≈ 310, just past k=251)",
        256, t256 as f64 / 1000.0);

    // Large payload — pre-#532 CANNOT decode (fdl > k-1).
    let t1024 = run_one(1024, cover_n);
    println!("{:>14}  {:>12.1}  large — UNDECODABLE PRE-#532",
        1024, t1024 as f64 / 1000.0);

    // Very large payload — pre-#532 CANNOT decode.
    let t4096 = run_one(4096, cover_n);
    println!("{:>14}  {:>12.1}  v.large — UNDECODABLE PRE-#532",
        4096, t4096 as f64 / 1000.0);

    // No-shadow brute-force — worst case (scans all tiers × all fdl).
    let t_none = run_no_shadow(cover_n);
    println!("{:>14}  {:>12.1}  no shadow (worst-case full brute-force)",
        0, t_none as f64 / 1000.0);

    println!();
    println!("Notes:");
    println!("- Pre-#532 the 1024 B and 4096 B payload extracts would FAIL with");
    println!("  FrameCorrupted regardless of how long they ran. The post-#532");
    println!("  peek path lands the correct fdl in 1 RS-decode + 1 AES.");
    println!("- The no-shadow case still must scan all fdl candidates across all");
    println!("  6 parity tiers to confirm 'no valid shadow' — its wall-clock is");
    println!("  the perf floor we live with (only architectural changes — e.g.,");
    println!("  v2 wire format with explicit tier hint — could cut it further).");

    // Sanity: extract must be well under 1 second on synthetic cover.
    assert!(t12 < 1_000_000_000, "12B extract too slow: {} ns", t12);
    assert!(t4096 < 5_000_000_000, "4KB extract too slow: {} ns", t4096);
}
