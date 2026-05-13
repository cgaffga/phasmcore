// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.8.13(b) (#455) — cascade-break gap audit.
//
// Reproduces the 2/76,384 wire-bit divergence observed in
// `openh264_stego_encode_yuv_string` when STC plans have 100+ flips, and
// reports per-diverging-position which `SyntaxPath` (Luma4x4 / Luma8x8 /
// ChromaAc / ChromaDc / LumaDcIntra16x16) the leaking position belongs
// to. The SyntaxPath identifies the C.8.x dual-recon hook that should
// have prevented the leak.
//
// Designed for ad-hoc invocation:
//   cargo test --release --features "h264-encoder openh264-backend" \
//       --test openh264_cascade_gap_audit -- --ignored --nocapture

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use core_openh264_sys::{
    encoder_pos_to_phasm_position_key, phasm_get_hook_dual_applied,
    phasm_get_hook_dual_bail_level_a_zero, phasm_get_hook_dual_bail_level_mismatch,
    phasm_get_hook_dual_fires_total, phasm_get_hook_single_applied,
    phasm_get_hook_single_bail_level_zero, phasm_get_hook_single_fires_total,
    phasm_reset_hook_dual_counters, PhasmStegoDomain, PHASM_MB_TYPE_OTHER,
};
use phasm_core::codec::h264::cabac::bin_decoder::slice::walk_annex_b_for_cover;
use phasm_core::codec::h264::openh264::{
    set_frame_num, Encoder, StegoHandlers, StegoSession,
};
use phasm_core::codec::h264::stego::hook::{EmbedDomain, SyntaxPath};
use phasm_core::stego::stc::embed::stc_embed;
use phasm_core::stego::stc::hhat::generate_hhat;

const STC_H: usize = 4;

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

fn synth_yuv(w: u32, h: u32, n_frames: u32) -> Vec<u8> {
    let frame_size = (w * h * 3 / 2) as usize;
    let mut out = Vec::with_capacity(frame_size * n_frames as usize);
    let mut s: u32 = 0xCAFE_F00D;
    for frame in 0..n_frames {
        for j in 0..h {
            for i in 0..w {
                let v = ((i + frame * 2) ^ (j + frame * 3)) as u8;
                out.push(v);
            }
        }
        for _plane in 0..2 {
            for j in 0..(h / 2) {
                for i in 0..(w / 2) {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    let texture = (s >> 16) as u8;
                    let pos = (i + j + frame) as u8;
                    out.push(texture.wrapping_add(pos));
                }
            }
        }
    }
    out
}

fn encode_once(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    qp: i32,
    override_map: Arc<Mutex<HashMap<u64, u8>>>,
    mb_type_table: Arc<Mutex<Vec<u8>>>,
    mb_width: u32,
    mb_per_frame: usize,
) -> Vec<u8> {
    {
        let mut t = mb_type_table.lock().unwrap();
        for x in t.iter_mut() {
            *x = 0xff;
        }
    }
    let map_for_hook = override_map.clone();
    let mb_type_for_hook = mb_type_table.clone();
    let mb_type_for_md = mb_type_table.clone();
    let handlers = StegoHandlers {
        enc_pre_emit: Some(Box::new(move |pos, _orig| {
            if pos.domain != PhasmStegoDomain::CoeffSign as u8 {
                return None;
            }
            let map = map_for_hook.lock().ok()?;
            if map.is_empty() {
                return None;
            }
            let mb_addr = (pos.mb_y as usize) * (mb_width as usize) + (pos.mb_x as usize);
            if mb_addr >= mb_per_frame {
                return None;
            }
            let mt = mb_type_for_hook.lock().ok()?[mb_addr];
            let mt_for_key = if mt == 0xff { PHASM_MB_TYPE_OTHER } else { mt };
            let key = encoder_pos_to_phasm_position_key(pos, mt_for_key, mb_width)?;
            map.get(&key).map(|&v| v as i32)
        })),
        md_cost: Some(Box::new(move |cost| {
            let mb_addr = (cost.mb_y as usize) * (mb_width as usize) + (cost.mb_x as usize);
            if mb_addr < mb_per_frame {
                if let Ok(mut t) = mb_type_for_md.lock() {
                    t[mb_addr] = cost.mb_type;
                }
            }
        })),
        ..Default::default()
    };
    let _sess = StegoSession::register(handlers).expect("register");
    let mut enc = Encoder::new(width as i32, height as i32, qp, 60).expect("enc");
    let frame_y = (width * height) as usize;
    let frame_uv = (width * height / 4) as usize;
    let frame_total = frame_y + 2 * frame_uv;
    let mut out = vec![0u8; 4 * 1024 * 1024];
    let mut bs = Vec::with_capacity(2 * 1024 * 1024);
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
        bs.extend_from_slice(&out[..n]);
    }
    bs
}

fn syntax_path_name(sp: &SyntaxPath) -> String {
    match sp {
        SyntaxPath::Luma4x4 { block_idx, coeff_idx, kind } =>
            format!("Luma4x4 (BC=2) block_idx={} coeff_idx={} kind={:?}", block_idx, coeff_idx, kind),
        SyntaxPath::Luma8x8 { block_idx, coeff_idx, kind } =>
            format!("Luma8x8 (BC=5) block_idx={} coeff_idx={} kind={:?}", block_idx, coeff_idx, kind),
        SyntaxPath::ChromaAc { plane, block_idx, coeff_idx, kind } =>
            format!("ChromaAc (BC=4) plane={} block_idx={} coeff_idx={} kind={:?}", plane, block_idx, coeff_idx, kind),
        SyntaxPath::ChromaDc { plane, coeff_idx, kind } =>
            format!("ChromaDc (BC=3) plane={} coeff_idx={} kind={:?}", plane, coeff_idx, kind),
        SyntaxPath::LumaDcIntra16x16 { coeff_idx, kind } =>
            format!("LumaDcIntra16x16 (BC=0) coeff_idx={} kind={:?}", coeff_idx, kind),
        SyntaxPath::Mvd { list, partition, axis, kind } =>
            format!("Mvd list={:?} part={} axis={:?} kind={:?}", list, partition, axis, kind),
    }
}

/// Single-flip probe: pick one walker position that was diverging in
/// the multi-seed audit, build an override map with ONLY that
/// position, encode pass 2, walk, check if the override landed.
///
/// If it lands → bug is COMBINATORIAL (only manifests at high flip
/// count, e.g. RDO interactions between multiple overrides).
/// If it doesn't → bug is per-position structural (some property of
/// that specific position prevents the override from reaching the wire).
#[test]
#[ignore]
fn audit_b_single_flip_probe() {
    let _g = session_guard().lock().unwrap();
    const W: u32 = 320;
    const H: u32 = 240;
    const N: u32 = 2;
    const QP: i32 = 22;

    let yuv = synth_yuv(W, H, N);
    let mb_width = W / 16;
    let mb_per_frame = (mb_width * (H / 16)) as usize;
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));

    // Baseline encode + walk.
    let baseline_bs = encode_once(
        &yuv, W, H, N, QP,
        override_map.clone(), mb_type_table.clone(),
        mb_width, mb_per_frame,
    );
    let baseline_walk = walk_annex_b_for_cover(&baseline_bs).expect("walker");
    let baseline_positions = baseline_walk.cover.coeff_sign_bypass.positions.clone();
    let baseline_bits = baseline_walk.cover.coeff_sign_bypass.bits.clone();

    // Walker_idx values from the multi-seed audit's first dirty seed (seed#0).
    // Each one was a missed flip at: frame=1, P-frame Luma 4x4 BC=2.
    let probe_walker_indices: &[usize] = &[74764, 75025, 75315, 74762, 49434];

    let mut summary = Vec::new();
    for &widx in probe_walker_indices {
        if widx >= baseline_positions.len() {
            continue;
        }
        let key = baseline_positions[widx];
        let target_bit = 1u8 ^ baseline_bits[widx];

        // Build single-override map.
        {
            let mut map = override_map.lock().unwrap();
            map.clear();
            map.insert(key.raw(), target_bit);
        }
        unsafe { phasm_reset_hook_dual_counters() };
        let stego_bs = encode_once(
            &yuv, W, H, N, QP,
            override_map.clone(), mb_type_table.clone(),
            mb_width, mb_per_frame,
        );
        let dual_applied = unsafe { phasm_get_hook_dual_applied() };
        let dual_bail = unsafe { phasm_get_hook_dual_bail_level_mismatch() };
        let dual_fires = unsafe { phasm_get_hook_dual_fires_total() };
        let single_applied = unsafe { phasm_get_hook_single_applied() };
        let single_fires = unsafe { phasm_get_hook_single_fires_total() };

        let p2_walk = walk_annex_b_for_cover(&stego_bs).expect("p2 walker");
        let p2_bits = &p2_walk.cover.coeff_sign_bypass.bits;
        let p2_positions = &p2_walk.cover.coeff_sign_bypass.positions;

        let landed = if widx < p2_positions.len()
            && p2_positions[widx].raw() == key.raw()
        {
            p2_bits[widx] == target_bit
        } else {
            false
        };
        // Also count any wire bit that differs in the entire stream.
        let mut total_wire_changes = 0u32;
        let cmp_len = baseline_bits.len().min(p2_bits.len());
        for i in 0..cmp_len {
            if p2_bits[i] != baseline_bits[i] {
                total_wire_changes += 1;
            }
        }

        let sp = key.syntax_path();
        eprintln!(
            "probe walker_idx={} mb_addr={} {}\n  baseline_bit={} target_bit={} → landed={} total_wire_changes={}\n  dual:   fires={} applied={} bail_mismatch={}\n  single: fires={} applied={}",
            widx, key.mb_addr(), syntax_path_name(&sp),
            baseline_bits[widx], target_bit, landed, total_wire_changes,
            dual_fires, dual_applied, dual_bail,
            single_fires, single_applied,
        );
        summary.push((widx, landed, dual_applied + single_applied, total_wire_changes));
    }

    let landed_count = summary.iter().filter(|(_, l, _, _)| *l).count();
    let total = summary.len();
    eprintln!(
        "single-flip probe summary: {}/{} positions landed cleanly",
        landed_count, total,
    );
}

/// Reproduce the cascade-break gap with a deterministic seed and report
/// which SyntaxPath the leaking positions belong to. This identifies
/// the C.8.x hook (HOOK-A/B/C/D/E/F/G/H1) that should be cascade-
/// breaking the position but isn't.
#[test]
#[ignore]
fn audit_b_cascade_gap_synth() {
    let _g = session_guard().lock().unwrap();
    const W: u32 = 320;
    const H: u32 = 240;
    const N: u32 = 2;
    const QP: i32 = 22;
    const M_TOTAL: usize = 800; // 100 bytes of "frame" — enough to force many flips
    // Try a few seeds — cascade-break holds on most but leaks on some.
    // Run multiple seeds until we hit one that diverges, then catalogue.
    let seeds: Vec<[u8; 32]> = (0u8..32u8).map(|s| [s.wrapping_mul(31).wrapping_add(7); 32]).collect();

    let yuv = synth_yuv(W, H, N);
    let mb_width = W / 16;
    let mb_per_frame = (mb_width * (H / 16)) as usize;
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));

    // Baseline encode + walk
    let baseline_bs = encode_once(
        &yuv, W, H, N, QP,
        override_map.clone(), mb_type_table.clone(),
        mb_width, mb_per_frame,
    );
    let baseline_walk = walk_annex_b_for_cover(&baseline_bs).expect("walker");
    let baseline_positions = baseline_walk.cover.coeff_sign_bypass.positions.clone();
    let baseline_bits = baseline_walk.cover.coeff_sign_bypass.bits.clone();
    let n_cover = baseline_bits.len();
    eprintln!("audit: n_cover={}", n_cover);
    assert!(n_cover >= M_TOTAL, "cover too small for M_TOTAL={}", M_TOTAL);

    let frame_bits: Vec<u8> = (0..M_TOTAL).map(|i| ((i * 17 + 3) % 2) as u8).collect();
    let w = n_cover / M_TOTAL;
    assert!(w >= 2, "w too small");
    let used = M_TOTAL * w;
    let costs: Vec<f32> = vec![1.0; used];

    // Aggregate divergence across all seeds.
    let mut total_diverge = 0u32;
    let mut clean_seeds = 0u32;
    let mut dirty_seeds = 0u32;
    let mut by_path_total: HashMap<String, u32> = HashMap::new();
    let mut diverges_sample: Vec<(usize, u32, u32, EmbedDomain, SyntaxPath)> = Vec::new();

    let mut total_bail_mismatch = 0u64;
    let mut total_applied = 0u64;
    let mut total_fires = 0u64;
    for (seed_idx, seed) in seeds.iter().enumerate() {
        let hhat = generate_hhat(STC_H, w, seed);
        let plan = stc_embed(&baseline_bits[..used], &costs, &frame_bits, &hhat, STC_H, w)
            .expect("stc_embed");

        let mut overrides: HashMap<u64, u8> = HashMap::new();
        for i in 0..used {
            if plan.stego_bits[i] != baseline_bits[i] {
                overrides.insert(baseline_positions[i].raw(), plan.stego_bits[i]);
            }
        }
        {
            let mut m = override_map.lock().unwrap();
            m.clear();
            for (k, v) in overrides.iter() {
                m.insert(*k, *v);
            }
        }
        // Reset fork-side counters right before the pass-2 encode so we
        // attribute bail counts to THIS seed's overrides only.
        unsafe { phasm_reset_hook_dual_counters() };
        let stego_bs = encode_once(
            &yuv, W, H, N, QP,
            override_map.clone(), mb_type_table.clone(),
            mb_width, mb_per_frame,
        );
        let seed_fires = unsafe { phasm_get_hook_dual_fires_total() };
        let seed_bail_zero = unsafe { phasm_get_hook_dual_bail_level_a_zero() };
        let seed_bail_mismatch = unsafe { phasm_get_hook_dual_bail_level_mismatch() };
        let seed_applied = unsafe { phasm_get_hook_dual_applied() };
        total_fires += seed_fires;
        total_bail_mismatch += seed_bail_mismatch;
        total_applied += seed_applied;
        let _ = seed_bail_zero;
        let p2_walk = walk_annex_b_for_cover(&stego_bs).expect("p2 walker");
        let p2_bits = &p2_walk.cover.coeff_sign_bypass.bits;
        let p2_positions = &p2_walk.cover.coeff_sign_bypass.positions;
        if p2_bits.len() != n_cover {
            eprintln!("seed#{}: position count diverged {} != {} (skipped)", seed_idx, p2_bits.len(), n_cover);
            continue;
        }

        let mut diverge_n = 0u32;
        let mut missed_flip = 0u32;    // plan said flip, wire didn't change
        let mut collateral = 0u32;     // plan said no flip, wire changed anyway
        for i in 0..used {
            if p2_bits[i] != plan.stego_bits[i] {
                diverge_n += 1;
                let k = p2_positions[i];
                let sp = k.syntax_path();
                *by_path_total.entry(syntax_path_name(&sp).to_string()).or_insert(0) += 1;
                // Classify: was this position a planned flip?
                let planned_flip = plan.stego_bits[i] != baseline_bits[i];
                if planned_flip {
                    missed_flip += 1;
                } else {
                    collateral += 1;
                }
                if diverges_sample.len() < 40 {
                    diverges_sample.push((i, k.frame_idx(), k.mb_addr(), k.domain(), sp));
                }
            }
        }
        if diverge_n > 0 {
            // augment per-seed report with the classification
            eprintln!(
                "  seed#{}: missed_flip={} collateral={}",
                seed_idx, missed_flip, collateral
            );
        }
        if diverge_n > 0 {
            dirty_seeds += 1;
            eprintln!(
                "seed#{:>2} (byte0={:02x}): planned_flips={} diverge={}",
                seed_idx, seed[0], plan.num_modifications, diverge_n
            );
        } else {
            clean_seeds += 1;
        }
        total_diverge += diverge_n;
    }

    eprintln!(
        "audit summary: seeds={} clean={} dirty={} total_diverge={}",
        seeds.len(), clean_seeds, dirty_seeds, total_diverge
    );
    eprintln!(
        "hook_dual counters (summed across all seeds): fires_total={} applied={} bail_mismatch={}",
        total_fires, total_applied, total_bail_mismatch
    );
    eprintln!("audit by SyntaxPath (across all dirty seeds):");
    let mut paths: Vec<_> = by_path_total.iter().collect();
    paths.sort_by(|a, b| b.1.cmp(a.1));
    for (path, count) in &paths {
        eprintln!("  {:>30}: {}", path, count);
    }
    eprintln!("audit diverging position sample (max 40):");
    for (idx, frame, mb_addr, dom, sp) in diverges_sample.iter() {
        eprintln!(
            "  walker_idx={:>6}  frame={}  mb_addr={:>4}  domain={:?}  path={}",
            idx, frame, mb_addr, dom, syntax_path_name(sp)
        );
    }
}
