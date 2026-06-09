// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! #493.4 Phase 3 — OH264 per-GOP 4-domain primitive integration test.
//!
//! Mirror of Phase 2's `h264_4domain_primitive_493_2.rs` for the OH264
//! backend. Verifies the new `h264_encode_gop_framed_bits_auto`
//! at the primitive level: encode → walk → combine_cover_4domain →
//! stc_extract → compare to original frame bits.
//!
//! Phase 3 is decoupled from the streaming session integration
//! (`oh264_finish` still calls the CS-only `encode_yuv_with_pre_framed_bits`);
//! Phase 4 / #493.5 will swap the call site once the matching combined-
//! cover decoder lands.

// DISABLED — do not reactivate without the source fix. Exercises the known
// 4-domain capacity gap (#493.4 / #505, deferred to retirement Phase 6). It
// was previously dead-gated on the removed pure-Rust `h264-encoder` feature;
// the openh264-backend→h264-encoder rename (Phase 5) would have silently
// reactivated it, so it is now explicitly disabled via `cfg(any())`.
#![cfg(any())]

use phasm_core::codec::h264::cabac::bin_decoder::slice::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};
use phasm_core::codec::h264::openh264_stego::{
    h264_encode_gop_framed_bits_auto, EncodeOpts,
};
use phasm_core::codec::h264::stego::{combine_cover_4domain, CostWeights};
use phasm_core::codec::h264::stego::orchestrate::DomainCosts;
use phasm_core::stego::stc::extract::stc_extract;
use phasm_core::stego::stc::hhat::generate_hhat;

const STC_H: usize = 4;

fn full_walk_options() -> WalkOptions {
    WalkOptions {
        record_mvd: true,
        ..Default::default()
    }
}

/// Same textured synth as Phase 2 — produces realistic 4-domain
/// cover density on small fixtures.
fn synth_yuv(width: u32, height: u32, frame_idx: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3 / 2) as usize);
    let w = width as i32;
    let h = height as i32;
    let half_w = w / 2;
    let half_h = h / 2;
    for j in 0..h {
        for i in 0..w {
            let val = ((i + frame_idx as i32 * 2) ^ (j + frame_idx as i32 * 3)) as u8;
            out.push(val);
        }
    }
    let mut s: u32 = 0xCAFE_F00D ^ frame_idx;
    for _plane in 0..2 {
        for j in 0..half_h {
            for i in 0..half_w {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let tex = (s >> 16) as u8;
                let pos = (i + j + frame_idx as i32) as u8;
                out.push(tex.wrapping_add(pos));
            }
        }
    }
    out
}

fn build_yuv_clip(width: u32, height: u32, n_frames: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for f in 0..n_frames {
        out.extend_from_slice(&synth_yuv(width, height, f));
    }
    out
}

/// Encode via OH264 4-domain primitive, walk the result with
/// `record_mvd: true`, combine the walker cover with the same
/// weights, and run stc_extract to recover frame bits.
fn encode_and_extract(
    width: u32,
    height: u32,
    n_frames: u32,
    frame_bits: &[u8],
    hhat_seed: &[u8; 32],
    weights: &CostWeights,
    qp: i32,
) -> Vec<u8> {
    let yuv = build_yuv_clip(width, height, n_frames);
    let opts = EncodeOpts {
        qp,
        intra_period: n_frames as i32,
    };
    let annex_b = h264_encode_gop_framed_bits_auto(
        &yuv, width, height, n_frames, opts, frame_bits, hhat_seed, weights,
    )
    .expect("OH264 4-domain encode");

    let walked = walk_annex_b_for_cover_with_options(&annex_b, full_walk_options())
        .expect("walk_annex_b_for_cover");

    let dummy_costs = DomainCosts::default();
    let (combined_cover, _, _boundaries) =
        combine_cover_4domain(&walked.cover, &dummy_costs, weights);

    let n_cover = combined_cover.len();
    let m_total = frame_bits.len();
    assert!(n_cover >= m_total, "n_cover {n_cover} < m_total {m_total} — fixture too small");
    let w = n_cover / m_total;
    assert!(w >= 1);
    let hhat = generate_hhat(STC_H, w, hhat_seed);
    let used_cover = m_total * w;
    let extracted = stc_extract(&combined_cover[..used_cover], &hhat, w);
    extracted[..m_total].to_vec()
}

#[test]
fn oh264_4domain_primitive_roundtrip_tiny() {
    let frame_bits = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
    let hhat_seed = [0x42u8; 32];
    let weights = CostWeights::default();
    let extracted = encode_and_extract(
        128, 80, 4, &frame_bits, &hhat_seed, &weights, /*qp*/ 26,
    );
    assert_eq!(extracted, frame_bits, "round-trip mismatch on tiny OH264 4-domain");
}

#[test]
fn oh264_4domain_primitive_roundtrip_default_weights() {
    let frame_bits: Vec<u8> = (0..64).map(|i| (i & 1) as u8).collect();
    let hhat_seed = [0x99u8; 32];
    let weights = CostWeights::default();
    let extracted = encode_and_extract(
        160, 96, 4, &frame_bits, &hhat_seed, &weights, /*qp*/ 26,
    );
    assert_eq!(extracted, frame_bits, "round-trip mismatch — default weights");
}

#[test]
fn oh264_4domain_primitive_roundtrip_uniform_weights() {
    // Force STC to genuinely spread across all 4 domains. This is
    // the load-bearing test — proves the OH264 override-map
    // dispatch handles all 4 domains uniformly when STC picks
    // multi-domain positions.
    let frame_bits: Vec<u8> = (0..48).map(|i| ((i * 7 + 3) & 1) as u8).collect();
    let hhat_seed = [0xAAu8; 32];
    let weights = CostWeights {
        coeff_sign: 1.0,
        coeff_suffix: 1.0,
        mvd_sign: 1.0,
        mvd_suffix: 1.0,
    };
    let extracted = encode_and_extract(
        160, 96, 4, &frame_bits, &hhat_seed, &weights, /*qp*/ 26,
    );
    assert_eq!(extracted, frame_bits, "round-trip mismatch — uniform weights");
}

#[test]
fn oh264_4domain_primitive_roundtrip_streaming_fixture_default_weights() {
    // Mirror the streaming-session fixture that's failing in lib tests:
    // 320×240 × 2f, ~456 frame_bits (= 57-byte chunk_frame), default
    // weights. If this passes, the bug is in the streaming-session
    // wiring, not in the OH264 4-domain primitive.
    //
    // 57 bytes = 4-byte chunk_frame header + 53-byte payload (=
    // phasm v1 frame for "hi" plaintext).
    let frame_bits: Vec<u8> = (0..456).map(|i| ((i * 13 + 7) & 1) as u8).collect();
    let hhat_seed = [0x33u8; 32];
    let weights = CostWeights::default();
    let extracted = encode_and_extract(
        320, 240, 2, &frame_bits, &hhat_seed, &weights, /*qp*/ 26,
    );
    assert_eq!(extracted, frame_bits,
        "round-trip mismatch — streaming fixture (320×240×2f, 456 bits, default weights)");
}

/// #505 diagnostic (currently #[ignore]'d) — pinpoint the EXACT cover-position(s)
/// where override application fails. Pass-1 clean → STC plan → Pass-2 stego
/// primitive → walker. Diff: at each cover position i, compare `plan.stego_bits[i]`
/// (what STC wanted) to `pass2_walker[i]` (what the walker actually reads after
/// override). Used to isolate the chroma DC C.8.5 cascade-break leak documented
/// in `memory/h264_chroma_csl_cascade_gap_504.md`.
#[test]
#[ignore = "diagnostic harness for #505 chroma cascade-break leak — keep available for re-run"]
fn oh264_4domain_primitive_diagnose_plan_vs_walker_pass2() {
    use phasm_core::codec::h264::openh264::Encoder;
    use phasm_core::codec::h264::stego::orchestrate::DomainCosts as Costs;
    use phasm_core::stego::stc::embed::stc_embed;
    use phasm_core::codec::h264::stego::hook::SyntaxPath;

    let weights = CostWeights {
        coeff_sign: 1.0,
        coeff_suffix: 1.0,
        mvd_sign: 1.0,
        mvd_suffix: 1.0,
    };
    let frame_bits: Vec<u8> = (0..456).map(|i| ((i * 17 + 11) & 1) as u8).collect();
    let hhat_seed = [0x77u8; 32];
    let m_total = frame_bits.len();
    let (w_dim, h_dim, n_dim) = (320u32, 240u32, 2u32);
    let yuv = build_yuv_clip(w_dim, h_dim, n_dim);
    let opts = EncodeOpts { qp: 26, intra_period: n_dim as i32 };

    // ── Pass-1: clean OH264 encode → walker ──
    let mut enc = Encoder::new(w_dim as i32, h_dim as i32, opts.qp, opts.intra_period)
        .expect("OH264 Encoder::new");
    let frame_y = (w_dim * h_dim) as usize;
    let frame_uv = (w_dim * h_dim / 4) as usize;
    let frame_total = frame_y + 2 * frame_uv;
    let mut out = vec![0u8; 4 * 1024 * 1024];
    let mut clean_annex_b = Vec::with_capacity(2 * 1024 * 1024);
    for f in 0..n_dim as usize {
        let base = f * frame_total;
        let (_ft, n) = enc.encode_frame(
            &yuv[base..base + frame_y],
            &yuv[base + frame_y..base + frame_y + frame_uv],
            &yuv[base + frame_y + frame_uv..base + frame_total],
            (f as i64) * 33,
            &mut out,
        ).expect("clean encode_frame");
        clean_annex_b.extend_from_slice(&out[..n]);
    }
    drop(enc);
    let clean_walked = walk_annex_b_for_cover_with_options(&clean_annex_b, full_walk_options())
        .expect("walk clean");
    let (clean_combined, clean_costs, boundaries) =
        combine_cover_4domain(&clean_walked.cover, &Costs::default(), &weights);

    let n_cover = clean_combined.len();
    let w = n_cover / m_total;
    let used = m_total * w;
    eprintln!("[DIFF] n_cover={n_cover} m_total={m_total} w={w} used={used}");
    eprintln!("  bounds: cs={} csl={} mvds={} mvdsl={}",
        boundaries.n_coeff_sign, boundaries.n_coeff_suffix,
        boundaries.n_mvd_sign, boundaries.n_mvd_suffix);

    // ── STC plan from clean Pass-1 cover ──
    let hhat = generate_hhat(STC_H, w, &hhat_seed);
    let plan = stc_embed(
        &clean_combined[..used],
        &clean_costs[..used],
        &frame_bits,
        &hhat, STC_H, w,
    ).expect("STC plan");
    let planned_flips: Vec<usize> = (0..used)
        .filter(|&i| plan.stego_bits[i] != clean_combined[i])
        .collect();
    eprintln!("  plan: {} positions to flip", planned_flips.len());

    // ── Pass-2: stego encode via primitive ──
    let stego_annex_b = h264_encode_gop_framed_bits_auto(
        &yuv, w_dim, h_dim, n_dim, opts, &frame_bits, &hhat_seed, &weights,
    ).expect("primitive encode");
    let stego_walked = walk_annex_b_for_cover_with_options(&stego_annex_b, full_walk_options())
        .expect("walk stego");
    let (stego_combined, _, stego_bounds) =
        combine_cover_4domain(&stego_walked.cover, &Costs::default(), &weights);
    eprintln!("[DIFF] Pass-2 bounds: cs={} csl={} mvds={} mvdsl={} total={}",
        stego_bounds.n_coeff_sign, stego_bounds.n_coeff_suffix,
        stego_bounds.n_mvd_sign, stego_bounds.n_mvd_suffix, stego_combined.len());
    eprintln!("[DIFF] per-domain Δ: cs Δ={} csl Δ={} mvds Δ={} mvdsl Δ={}",
        stego_bounds.n_coeff_sign as i64 - boundaries.n_coeff_sign as i64,
        stego_bounds.n_coeff_suffix as i64 - boundaries.n_coeff_suffix as i64,
        stego_bounds.n_mvd_sign as i64 - boundaries.n_mvd_sign as i64,
        stego_bounds.n_mvd_suffix as i64 - boundaries.n_mvd_suffix as i64);
    if stego_combined.len() != n_cover {
        // Find the per-position divergence in CSL between Pass-1 and Pass-2.
        let clean_csl = &clean_walked.cover.coeff_suffix_lsb.positions;
        let stego_csl = &stego_walked.cover.coeff_suffix_lsb.positions;
        let mut clean_i = 0usize;
        let mut stego_i = 0usize;
        while clean_i < clean_csl.len() && stego_i < stego_csl.len() {
            if clean_csl[clean_i].raw() == stego_csl[stego_i].raw() {
                clean_i += 1; stego_i += 1;
            } else {
                // Position present in clean but missing in stego (skip clean only)
                eprintln!("[CSL DIVERGE] clean[{clean_i}] frame={} mb_addr={} {:?} — missing in stego",
                    clean_csl[clean_i].frame_idx(),
                    clean_csl[clean_i].mb_addr(), clean_csl[clean_i].syntax_path());
                clean_i += 1;
                break;
            }
        }
        eprintln!("[DIFF] cover length mismatch — exiting before bit-level diff");
        return;
    }

    // ── Diff per cover position ──
    let mut miss_positions: Vec<(usize, u8, u8)> = vec![];
    for i in 0..used {
        if plan.stego_bits[i] != stego_combined[i] {
            miss_positions.push((i, plan.stego_bits[i], stego_combined[i]));
        }
    }
    eprintln!("[DIFF RESULT] {} override-miss cover positions (vs {} planned flips)",
        miss_positions.len(), planned_flips.len());
    for (i, planned, actual) in miss_positions.iter().take(30) {
        let n_cs = boundaries.n_coeff_sign;
        let n_csl = boundaries.n_coeff_suffix;
        let n_mvds = boundaries.n_mvd_sign;
        let (dom, dom_idx, path_info) = if *i < n_cs {
            let idx = *i;
            let pk = clean_walked.cover.coeff_sign_bypass.positions[idx];
            ("CS", idx, format!("mb_addr={} {:?}", pk.mb_addr(), pk.syntax_path()))
        } else if *i < n_cs + n_csl {
            let idx = *i - n_cs;
            let pk = clean_walked.cover.coeff_suffix_lsb.positions[idx];
            ("CSL", idx, format!("mb_addr={} {:?}", pk.mb_addr(), pk.syntax_path()))
        } else if *i < n_cs + n_csl + n_mvds {
            let idx = *i - n_cs - n_csl;
            let pk = clean_walked.cover.mvd_sign_bypass.positions[idx];
            ("MVDs", idx, format!("mb_addr={} {:?}", pk.mb_addr(), pk.syntax_path()))
        } else {
            let idx = *i - n_cs - n_csl - n_mvds;
            let pk = clean_walked.cover.mvd_suffix_lsb.positions[idx];
            ("MVDsl", idx, format!("mb_addr={} {:?}", pk.mb_addr(), pk.syntax_path()))
        };
        let was_planned_flip = clean_combined[*i] != *planned;
        eprintln!("  miss pos {i}: planned={planned} actual={actual} clean={} (planned_flip={was_planned_flip}) — {dom}[{dom_idx}] {path_info}",
            clean_combined[*i]);
        let _ = path_info; // silence warning
    }

    // Also classify: was the override map supposed to have this position?
    // For each miss: planned != actual AND planned != clean → plan changed bit, walker didn't see change → override miss
    //                planned != actual AND planned == clean → plan didn't change bit, walker sees different bit → walker drift
    let mut override_misses = 0;
    let mut walker_drifts = 0;
    for (i, planned, _) in &miss_positions {
        if clean_combined[*i] != *planned {
            override_misses += 1;
        } else {
            walker_drifts += 1;
        }
    }
    eprintln!("  classified: {override_misses} override-misses, {walker_drifts} walker-drifts");
    let _ = SyntaxPath::Mvd { list: 0, partition: 0, axis: phasm_core::codec::h264::stego::Axis::X, kind: phasm_core::codec::h264::stego::hook::BinKind::Sign };
}

#[test]
#[ignore = "diagnostic harness for #505 chroma cascade-break leak — keep available for re-run"]
fn oh264_4domain_primitive_diagnose_uniform_weights() {
    use phasm_core::codec::h264::stego::orchestrate::DomainCosts as Costs;
    let yuv = build_yuv_clip(320, 240, 2);
    let opts = EncodeOpts { qp: 26, intra_period: 2 };
    let weights = CostWeights {
        coeff_sign: 1.0,
        coeff_suffix: 1.0,
        mvd_sign: 1.0,
        mvd_suffix: 1.0,
    };
    let frame_bits: Vec<u8> = (0..456).map(|i| ((i * 17 + 11) & 1) as u8).collect();
    let hhat_seed = [0x77u8; 32];
    let m_total = frame_bits.len();

    let annex_b = h264_encode_gop_framed_bits_auto(
        &yuv, 320, 240, 2, opts, &frame_bits, &hhat_seed, &weights,
    ).expect("encode");
    let walked = walk_annex_b_for_cover_with_options(&annex_b, full_walk_options())
        .expect("walk");
    let (combined_cover, _, boundaries) =
        combine_cover_4domain(&walked.cover, &Costs::default(), &weights);
    let n_cover = combined_cover.len();
    let w = n_cover / m_total;
    let used_cover = m_total * w;
    let hhat = generate_hhat(STC_H, w, &hhat_seed);
    let extracted = stc_extract(&combined_cover[..used_cover], &hhat, w);

    let mismatches: Vec<usize> = (0..m_total)
        .filter(|&i| extracted[i] != frame_bits[i]).collect();
    let n_cs = boundaries.n_coeff_sign;
    let n_csl = boundaries.n_coeff_suffix;
    let n_mvds = boundaries.n_mvd_sign;
    let n_mvdsl = boundaries.n_mvd_suffix;
    eprintln!("[DIAG] n_cover={n_cover} m_total={m_total} w={w} used={used_cover}");
    eprintln!("  boundaries: n_cs={n_cs} n_csl={n_csl} n_mvds={n_mvds} n_mvdsl={n_mvdsl}");
    eprintln!("  mismatches: {} bits", mismatches.len());
    for &b in mismatches.iter().take(20) {
        let col_start = b * w;
        let dom = if col_start < n_cs { "CS" }
            else if col_start < n_cs + n_csl { "CSL" }
            else if col_start < n_cs + n_csl + n_mvds { "MVDs" }
            else { "MVDsl" };
        // For CSL positions: print the walker's SyntaxPath + mb_addr.
        let walker_info = if dom == "CSL" {
            let csl_idx = col_start - n_cs;
            let pk = walked.cover.coeff_suffix_lsb.positions[csl_idx];
            format!("mb_addr={} {:?}", pk.mb_addr(), pk.syntax_path())
        } else {
            String::from("(n/a)")
        };
        eprintln!("  bit {b}: STC col abs [{col_start},{}) starts in {dom} — {walker_info}",
            col_start + STC_H.min(used_cover - col_start));
    }

    // Dump all walker CSL chroma positions (mb_addr + path) for cross-ref
    // against fork's [FIRE_CSL_CHROMA] log lines emitted from the encoder.
    eprintln!("[WALKER_CSL_CHROMA_POSITIONS]");
    for (i, pk) in walked.cover.coeff_suffix_lsb.positions.iter().enumerate() {
        use phasm_core::codec::h264::stego::hook::SyntaxPath;
        match pk.syntax_path() {
            SyntaxPath::ChromaDc { .. } | SyntaxPath::ChromaAc { .. } => {
                eprintln!("  walker_csl[{i}]: mb_addr={} {:?}",
                    pk.mb_addr(), pk.syntax_path());
            }
            _ => {}
        }
    }
}

#[test]
fn oh264_4domain_primitive_roundtrip_streaming_fixture_uniform_weights() {
    // Uniform weights force STC across all 4 domains. Phase 4.5 (#504)
    // extended `encoder_pos_to_phasm_position_key` to handle MVD domains
    // and dropped the CS+CSL filter in `encode_once::enc_pre_emit`; the
    // fork's HOOK-H1..H7 + `phasm_apply_mvd_hooks` already fire the
    // callback with `pos.domain ∈ {MvdSign, MvdSuffixLsb}` and apply
    // overrides by mutating the qpel MV in place + re-running MC. C.8.7
    // cascade-break keeps pDecPic/pVisualRecPic consistent.
    //
    // BLOCKED: surfaces a separate latent C.8.5 chroma DC cascade-break
    // leak (#505). 4 syndrome bits fail because a chroma DC SuffixLsb
    // position vanishes between Pass-1 and Pass-2 at MB(17,5) Cb
    // coeff_idx=1, shifting the cover layout by 1. See
    // `memory/h264_chroma_csl_cascade_gap_504.md`. Re-enable when #505
    // ships a fork-side patch.
    let frame_bits: Vec<u8> = (0..456).map(|i| ((i * 17 + 11) & 1) as u8).collect();
    let hhat_seed = [0x77u8; 32];
    let weights = CostWeights {
        coeff_sign: 1.0,
        coeff_suffix: 1.0,
        mvd_sign: 1.0,
        mvd_suffix: 1.0,
    };
    let extracted = encode_and_extract(
        320, 240, 2, &frame_bits, &hhat_seed, &weights, /*qp*/ 26,
    );
    assert_eq!(extracted, frame_bits,
        "round-trip mismatch — streaming fixture (320×240×2f, 456 bits, uniform)");
}

#[test]
fn oh264_4domain_primitive_rejects_empty_frame_bits() {
    let yuv = build_yuv_clip(128, 80, 2);
    let opts = EncodeOpts { qp: 26, intra_period: 2 };
    let weights = CostWeights::default();
    let err = h264_encode_gop_framed_bits_auto(
        &yuv, 128, 80, 2, opts, &[], &[0u8; 32], &weights,
    )
    .expect_err("empty frame_bits should error");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("empty") || msg.contains("frame"),
        "unexpected error message: {msg}"
    );
}
