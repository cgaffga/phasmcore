// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase 2.7 (#272) — per-MB spatial-direct trace harness for the
// v1.1 §B-cascade-real bug. Localizes which (frame, mb_x, mb_y) MB
// has the worst encoder.visual_recon vs ffmpeg.decode divergence,
// and dumps phasm's spatial-direct derivation inputs (neighbour
// A/B/C MV+ref state, colocated cell, colZeroFlag) + outputs for
// that MB so we can hand-trace ITU spec § 8.4.1.2.2 and identify
// where phasm's algorithm diverges from spec.
//
// Triggered by Option C diagnostic finding: disabling
// `skip_cbp_is_zero` pre-check on carplane drops mismatch_max from
// 199 to 29, indicating the Skip path's spatial-direct derivation
// has a real divergence vs ffmpeg-decoder spatial-direct on motion
// content. Phase 2.5 static audit said "matches modulo inactive
// clause-2 colZeroFlag gap" — empirical 199 contradicts that.
//
// **How to run** (carplane fixture must already exist; produced
// by the bisect harness or by running ensure_yuv on the corpus):
//
//   PHASM_DETERMINISTIC_SEED=42 PHASM_B_RDO=1 PHASM_B_RESIDUAL=1 \
//     cargo test -p phasm-core --release --features cabac-stego \
//     --test h264_phase_2_7_spatial_direct_trace -- --ignored --nocapture

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::b_direct_predictor::{
    drain_spatial_direct_traces, SpatialDirectTrace,
};
use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::encoder::mb_decision_b::{drain_b_mb_records, BMbRecord};
use phasm_core::codec::h264::stego::gop_pattern::{
    iter_encode_order, FrameType, GopPattern,
};

fn mode_label(mode_id: u8) -> &'static str {
    match mode_id {
        0 => "Skip",
        1 => "Direct",
        2 => "L0",
        3 => "L1",
        4 => "Bi",
        5 => "Partitioned",
        6 => "B_8x8",
        _ => "?",
    }
}

const CARPLANE_W: u32 = 1072;
const CARPLANE_H: u32 = 1920;

#[test]
#[ignore]
fn phase_2_7_spatial_direct_trace_carplane() {
    // Phase 2.13 (#270) — pick f15 / f30 fixture based on n_frames.
    let want_n: usize = std::env::var("PHASM_PHASE_2_7_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let fixture_n = if want_n <= 15 { 15 } else { 30 };
    let carplane_yuv = format!(
        "/tmp/phasm_corpus_artlist_carplane_1072x1920_f{}.yuv",
        fixture_n
    );
    let yuv = match std::fs::read(&carplane_yuv) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "SKIP: missing {} ({}). Run the bisect harness first to\n\
                 produce the carplane fixture.",
                carplane_yuv, e
            );
            return;
        }
    };
    // Phase 2.13 (#270 / #273, 2026-05-08) — extend to 30-frame run to
    // observe cascade behaviour at the same length as the corpus harness.
    // Override via PHASM_PHASE_2_7_FRAMES=N if needed.
    let n_frames: usize = std::env::var("PHASM_PHASE_2_7_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let frame_size = (CARPLANE_W * CARPLANE_H * 3 / 2) as usize;
    assert!(
        yuv.len() >= n_frames * frame_size,
        "carplane yuv has {} bytes, need {}",
        yuv.len(),
        n_frames * frame_size,
    );

    unsafe {
        // Honour external PHASM_B_RDO if set; default to 1 otherwise.
        if std::env::var_os("PHASM_B_RDO").is_none() {
            std::env::set_var("PHASM_B_RDO", "1");
        }
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::set_var("PHASM_B_SPATIAL_DIRECT_TRACE", "1");
        std::env::set_var("PHASM_B_INSTRUMENT", "1");
        // Phase 2.17 (#286, 2026-05-09) — preserve PHASM_B_FORCE_MODE
        // when explicitly set; otherwise remove to default to RDO mix.
        if std::env::var_os("PHASM_B_FORCE_MODE").is_none() {
            std::env::remove_var("PHASM_B_FORCE_MODE");
        }
    }

    // Drain any leftover records from earlier tests in this process
    // before encoding so our run starts with a clean record vec.
    let _ = drain_spatial_direct_traces();
    let _ = drain_b_mb_records();

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    let mut enc = Encoder::new(CARPLANE_W, CARPLANE_H, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    // Phase 2.13 (#273) — toggle transform_8x8 via env to bisect.
    enc.enable_transform_8x8 = std::env::var_os("PHASM_NO_T8X8").is_none();
    enc.enable_b_frames = true;

    // Encode + capture visual_recon Y for each frame.
    let mut bitstream = Vec::new();
    let mut recon_dumps: Vec<(u32, FrameType, Vec<u8>)> = Vec::new();
    // Map B-frame ordinal (1, 2, ...) → display_idx.
    let mut b_ordinal_to_display: Vec<u32> = Vec::new();
    let mut b_counter = 0u32;
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame),
            FrameType::P => enc.encode_p_frame(frame),
            FrameType::B => enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("encode (encode_idx={}): {e}", eo.encode_idx));
        bitstream.extend_from_slice(&bytes);
        recon_dumps.push((eo.display_idx, eo.frame_type, enc.visual_recon.y.clone()));
        if eo.frame_type == FrameType::B {
            b_counter += 1;
            if b_ordinal_to_display.len() < b_counter as usize {
                b_ordinal_to_display.resize(b_counter as usize, eo.display_idx);
            }
            b_ordinal_to_display[(b_counter - 1) as usize] = eo.display_idx;
        }
    }

    let traces = drain_spatial_direct_traces();
    let mb_records = drain_b_mb_records();
    eprintln!(
        "captured {} spatial-direct trace rows + {} BMb records across {} B-frames",
        traces.len(),
        mb_records.len(),
        b_ordinal_to_display.len()
    );

    // Decode the bitstream via ffmpeg.
    let h264_path = std::env::temp_dir().join("phasm_phase_2_7_spatial_direct.h264");
    let dec_path = std::env::temp_dir().join("phasm_phase_2_7_spatial_direct.dec.yuv");
    std::fs::write(&h264_path, &bitstream).expect("write h264");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .status()
        .expect("ffmpeg run");
    assert!(status.success(), "ffmpeg decode failed");

    let decoded = std::fs::read(&dec_path).expect("read decoded yuv");
    let y_size = (CARPLANE_W * CARPLANE_H) as usize;

    // Group traces by frame_idx — multiple calls per MB during RDO
    // mean we can have several rows per (frame, mb_x, mb_y); take
    // the LAST trace as it represents the latest derivation in
    // that frame's RDO sweep (closest to the actual emit choice
    // for that MB).
    use std::collections::HashMap;
    let mut last_trace: HashMap<(u32, u16, u16), SpatialDirectTrace> = HashMap::new();
    for t in &traces {
        last_trace.insert((t.frame_idx, t.mb_x, t.mb_y), *t);
    }
    let mut mb_record: HashMap<(u32, u16, u16), BMbRecord> = HashMap::new();
    for r in &mb_records {
        mb_record.insert((r.frame_idx, r.mb_x, r.mb_y), *r);
    }

    eprintln!("\n=== Per-frame mismatch + worst-MB spatial-direct trace ===");
    let mb_w = (CARPLANE_W / 16) as usize;
    let mb_h = (CARPLANE_H / 16) as usize;

    for (display_idx, ft, enc_y) in &recon_dumps {
        let off = (*display_idx as usize) * frame_size;
        let dec_y = &decoded[off..off + y_size];

        let mut frame_diff: u64 = 0;
        let mut frame_max: u32 = 0;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            frame_diff += d as u64;
            if d > frame_max {
                frame_max = d;
            }
        }
        eprintln!(
            "\n--- display={}  type={:?}  Σ|Δ|={}  max|Δ|={} ---",
            display_idx, ft, frame_diff, frame_max
        );

        // Skip non-B-frames for trace lookup.
        if *ft != FrameType::B {
            continue;
        }

        // Find this B-frame's ordinal (1-based) in encode order.
        let b_ordinal = match b_ordinal_to_display
            .iter()
            .position(|&d| d == *display_idx)
        {
            Some(p) => (p + 1) as u32,
            None => continue,
        };

        // Per-MB max|Δ| ranking.
        let mut mb_diverge: Vec<(usize, usize, u32, u32)> = Vec::new();
        for mby in 0..mb_h {
            for mbx in 0..mb_w {
                let mut max_d = 0u32;
                let mut sum_d = 0u32;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let row = mby * 16 + dy;
                        let col = mbx * 16 + dx;
                        let idx = row * (CARPLANE_W as usize) + col;
                        let d = (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
                        sum_d += d;
                        if d > max_d {
                            max_d = d;
                        }
                    }
                }
                if sum_d > 0 {
                    mb_diverge.push((mbx, mby, max_d, sum_d));
                }
            }
        }

        if mb_diverge.is_empty() {
            eprintln!("  no diverging MBs (clean B-frame)");
            continue;
        }

        // Phase 2.10 (#272 follow-on) — find FIRST raster-order MB
        // with ANY non-zero divergence (not just >=4). This exposes
        // sub-threshold cascade origins that the >=4 filter hides.
        let mut all_raster: Vec<(usize, usize, u32, u32)> = mb_diverge.clone();
        all_raster.sort_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));
        if let Some((mbx, mby, max_d, sum_d)) = all_raster.first() {
            let key = (b_ordinal, *mbx as u16, *mby as u16);
            let mode_str = match mb_record.get(&key) {
                Some(r) => format!(
                    "mode={} actual_mvL0=({},{}) actual_mvL1=({},{})",
                    mode_label(r.mode_id),
                    r.mv_l0_x, r.mv_l0_y, r.mv_l1_x, r.mv_l1_y
                ),
                None => "(no BMbRecord — likely Partitioned/B_8x8)".to_string(),
            };
            eprintln!(
                "\n  first non-zero diff in raster order: mb=({:>3},{:>3}) max|Δ|={} Σ|Δ|={} {}",
                mbx, mby, max_d, sum_d, mode_str
            );
            // Show first 10 raster-order MBs with max|Δ|>=1.
            eprintln!("  first 10 ANY-divergence MBs (max|Δ|>=1):");
            for (i, (mbx, mby, max_d, sum_d)) in all_raster.iter().take(10).enumerate() {
                let key = (b_ordinal, *mbx as u16, *mby as u16);
                let mode_str = match mb_record.get(&key) {
                    Some(r) => format!(
                        "mode={} mvL0=({},{}) mvL1=({},{})",
                        mode_label(r.mode_id),
                        r.mv_l0_x, r.mv_l0_y, r.mv_l1_x, r.mv_l1_y
                    ),
                    None => "(no BMbRecord)".to_string(),
                };
                eprintln!(
                    "    [{:>2}] mb=({:>3},{:>3}) max|Δ|={} Σ|Δ|={} {}",
                    i, mbx, mby, max_d, sum_d, mode_str
                );
            }
        }

        // First-diverging MB in raster order. This is the cascade
        // root: phasm's encoder grid first disagrees with ffmpeg's
        // decoder grid at this MB. Subsequent MBs may show worse
        // max|Δ| because they read this MB as a neighbour and amp
        // the divergence further.
        let mut raster = mb_diverge.clone();
        raster.sort_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));
        let mut first_significant: Vec<(usize, usize, u32, u32)> = raster
            .iter()
            .filter(|(_, _, max_d, _)| *max_d >= 4)
            .copied()
            .collect();
        if first_significant.is_empty() {
            // Fall back to all-diverging if none meet the threshold.
            first_significant = raster.clone();
        }
        eprintln!("  diverging MBs in this frame: {}", mb_diverge.len());
        eprintln!("  first 15 diverging MBs (max|Δ|>=4) in raster order — cascade-root candidates:");
        for (i, (mbx, mby, max_d, sum_d)) in first_significant.iter().take(15).enumerate() {
            let key = (b_ordinal, *mbx as u16, *mby as u16);
            let mode_str = match mb_record.get(&key) {
                Some(r) => format!(
                    "mode={} actual_mvL0=({},{}) actual_mvL1=({},{})",
                    mode_label(r.mode_id),
                    r.mv_l0_x, r.mv_l0_y, r.mv_l1_x, r.mv_l1_y
                ),
                None => "mode=? (BMbRecord missing — Partitioned/B_8x8 likely)".to_string(),
            };
            let trace_str = match last_trace.get(&key) {
                Some(t) => format!(
                    "  direct_draft_L0=({},{}) direct_draft_L1=({},{}) colMb=({},{}) colZF={}",
                    t.out_mv_l0_x, t.out_mv_l0_y,
                    t.out_mv_l1_x, t.out_mv_l1_y,
                    t.col_mv_l0_x, t.col_mv_l0_y,
                    t.col_zero_flag,
                ),
                None => "  (no spatial-direct trace)".to_string(),
            };
            eprintln!(
                "    [{:>2}] mb=({:>3},{:>3})  max|Δ|={}  Σ|Δ|={}  {}",
                i, mbx, mby, max_d, sum_d, mode_str
            );
            eprintln!("      {}", trace_str);
        }

        // Mode-frequency summary across the whole frame.
        let mut mode_freq: [u32; 7] = [0; 7];
        let mut diverge_by_mode: [u32; 7] = [0; 7];
        for r in &mb_records {
            if r.frame_idx == b_ordinal {
                let m = r.mode_id.min(6) as usize;
                mode_freq[m] += 1;
            }
        }
        for (mbx, mby, max_d, _) in &mb_diverge {
            let key = (b_ordinal, *mbx as u16, *mby as u16);
            if let Some(r) = mb_record.get(&key) {
                if *max_d >= 4 {
                    let m = r.mode_id.min(6) as usize;
                    diverge_by_mode[m] += 1;
                }
            }
        }
        eprintln!("  mode mix this frame:");
        for i in 0..=4u8 {
            let total = mode_freq[i as usize];
            let div = diverge_by_mode[i as usize];
            if total > 0 {
                eprintln!(
                    "    {:<10} total={:>5}  diverging(>=4)={:>4}  divergence_rate={:.1}%",
                    mode_label(i),
                    total,
                    div,
                    if total > 0 { 100.0 * div as f64 / total as f64 } else { 0.0 }
                );
            }
        }

        // Full neighbour-state dump for the FIRST cascade-root MB.
        if let Some((mbx, mby, _, _)) = first_significant.first() {
            let trace_key = (b_ordinal, *mbx as u16, *mby as u16);
            if let Some(t) = last_trace.get(&trace_key) {
                eprintln!(
                    "\n  --- FIRST cascade-root MB ({},{}) full spatial-direct trace ---",
                    mbx, mby
                );
                eprintln!(
                    "      kind={}  refIdxL0={} refIdxL1={}",
                    if t.kind == 0 { "spatial" } else { "temporal" },
                    t.ref_idx_l0,
                    t.ref_idx_l1,
                );
                eprintln!(
                    "      out_mvL0=({},{})  out_mvL1=({},{})",
                    t.out_mv_l0_x, t.out_mv_l0_y, t.out_mv_l1_x, t.out_mv_l1_y
                );
                eprintln!(
                    "      A_L0=({},{},ref={})  B_L0=({},{},ref={})  C_L0=({},{},ref={}, c_was_c={})",
                    t.a_l0_mv_x, t.a_l0_mv_y, t.a_l0_ref,
                    t.b_l0_mv_x, t.b_l0_mv_y, t.b_l0_ref,
                    t.c_l0_mv_x, t.c_l0_mv_y, t.c_l0_ref,
                    t.c_was_c
                );
                eprintln!(
                    "      A_L1=({},{},ref={})  B_L1=({},{},ref={})  C_L1=({},{},ref={})",
                    t.a_l1_mv_x, t.a_l1_mv_y, t.a_l1_ref,
                    t.b_l1_mv_x, t.b_l1_mv_y, t.b_l1_ref,
                    t.c_l1_mv_x, t.c_l1_mv_y, t.c_l1_ref
                );
                eprintln!(
                    "      colMb refIdxL0={} mvL0=({},{})  colZeroFlag={} (grid_present={})",
                    t.col_ref_idx_l0,
                    t.col_mv_l0_x,
                    t.col_mv_l0_y,
                    t.col_zero_flag,
                    t.col_grid_present
                );
            } else {
                eprintln!(
                    "\n  --- FIRST cascade-root MB ({},{}): no spatial-direct trace; \
                     RDO picked L0/L1/Bi/Skip-without-Direct here. PMV cascade likely. ---",
                    mbx, mby
                );
            }
        }

        // Phase 2.9 (#274 follow-on) — pixel diff dump for the FIRST
        // raster-order cascade-root MB. Helps identify whether the
        // divergence is sub-pel rounding (small uniform offset) or
        // structural (e.g., wrong reference patch).
        if let Some((mbx, mby, _, _)) = first_significant.first() {
            eprintln!("\n  --- pixel diff at raster-first cascade-root MB ({},{}) ---", mbx, mby);
            let off = (*display_idx as usize) * frame_size;
            let dec_y = &decoded[off..off + y_size];
            eprintln!("    enc.visual_recon (16x16 Y):");
            for dy in 0..16 {
                eprint!("      ");
                for dx in 0..16 {
                    let idx = (mby * 16 + dy) * CARPLANE_W as usize + mbx * 16 + dx;
                    eprint!("{:>4}", enc_y[idx]);
                }
                eprintln!();
            }
            eprintln!("    diff (enc - dec):");
            for dy in 0..16 {
                eprint!("      ");
                for dx in 0..16 {
                    let idx = (mby * 16 + dy) * CARPLANE_W as usize + mbx * 16 + dx;
                    let d = enc_y[idx] as i32 - dec_y[idx] as i32;
                    eprint!("{:>+5}", d);
                }
                eprintln!();
            }
        }

        // Sort by max|Δ| descending.
        mb_diverge.sort_by(|a, b| b.2.cmp(&a.2));
        eprintln!("\n  top-5 worst MBs by max|Δ| (downstream effects):");
        for (i, (mbx, mby, max_d, sum_d)) in mb_diverge.iter().take(5).enumerate() {
            let trace_key = (b_ordinal, *mbx as u16, *mby as u16);
            match last_trace.get(&trace_key) {
                Some(t) => {
                    let kind_str = if t.kind == 0 { "spatial" } else { "temporal" };
                    eprintln!(
                        "    [{:>2}] mb=({:>3},{:>3})  max|Δ|={}  Σ|Δ|={}  kind={}",
                        i, mbx, mby, max_d, sum_d, kind_str
                    );
                    eprintln!(
                        "         derived refIdxL0={} refIdxL1={}  out_mvL0=({},{}) out_mvL1=({},{})",
                        t.ref_idx_l0,
                        t.ref_idx_l1,
                        t.out_mv_l0_x,
                        t.out_mv_l0_y,
                        t.out_mv_l1_x,
                        t.out_mv_l1_y
                    );
                    eprintln!(
                        "         A_L0=({},{},ref={})  B_L0=({},{},ref={})  C_L0=({},{},ref={}, c_was_c={})",
                        t.a_l0_mv_x, t.a_l0_mv_y, t.a_l0_ref,
                        t.b_l0_mv_x, t.b_l0_mv_y, t.b_l0_ref,
                        t.c_l0_mv_x, t.c_l0_mv_y, t.c_l0_ref,
                        t.c_was_c
                    );
                    eprintln!(
                        "         A_L1=({},{},ref={})  B_L1=({},{},ref={})  C_L1=({},{},ref={})",
                        t.a_l1_mv_x, t.a_l1_mv_y, t.a_l1_ref,
                        t.b_l1_mv_x, t.b_l1_mv_y, t.b_l1_ref,
                        t.c_l1_mv_x, t.c_l1_mv_y, t.c_l1_ref
                    );
                    eprintln!(
                        "         colMb refIdxL0={} mvL0=({},{})  colZeroFlag={} (grid_present={})",
                        t.col_ref_idx_l0,
                        t.col_mv_l0_x,
                        t.col_mv_l0_y,
                        t.col_zero_flag,
                        t.col_grid_present
                    );
                }
                None => {
                    eprintln!(
                        "    [{:>2}] mb=({:>3},{:>3})  max|Δ|={}  Σ|Δ|={}  (no spatial-direct trace — \
                         RDO picked non-Direct mode)",
                        i, mbx, mby, max_d, sum_d
                    );
                }
            }
        }
    }

    unsafe {
        std::env::remove_var("PHASM_B_SPATIAL_DIRECT_TRACE");
        std::env::remove_var("PHASM_B_INSTRUMENT");
    }
    eprintln!(
        "\nphase_2_7 trace dump complete. Hand-trace the worst MB's row\n\
         through ITU spec § 8.4.1.2.2 to find where phasm diverges."
    );
}

/// Phase 2.8 (#272 follow-on, 2026-05-08) — port the Phase 2.6
/// `invest_b_force_mode_bisect_1080p` force-mode harness to the
/// CARPLANE fixture. Phase 2.6's claim "all 16×16 modes byte-exact"
/// was measured on iphone7. Phase 2.7 found L0/L1/Bi modes diverge
/// at 5-11% on carplane (default RDO mix). Force_mode this same
/// fixture and see if the bug fires when an isolated mode is used,
/// or only under cross-mode RDO interaction.
///
/// Knobs: pass force mode via `PHASM_B_FORCE_MODE` env var.
/// Defaults to `l0_16x16` if unset.
#[test]
#[ignore]
fn phase_2_8_force_mode_bisect_carplane() {
    let carplane_yuv = "/tmp/phasm_corpus_artlist_carplane_1072x1920_f15.yuv";
    let yuv = match std::fs::read(carplane_yuv) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: missing {} ({})", carplane_yuv, e);
            return;
        }
    };
    let n_frames = 5usize;
    let frame_size = (CARPLANE_W * CARPLANE_H * 3 / 2) as usize;
    assert!(yuv.len() >= n_frames * frame_size);

    let force_mode = std::env::var("PHASM_B_FORCE_MODE").unwrap_or("l0_16x16".into());
    eprintln!("=== CARPLANE force-mode bisect: PHASM_B_FORCE_MODE={force_mode} ===");

    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::set_var("PHASM_B_FORCE_MODE", &force_mode);
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
    }

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    let mut enc = Encoder::new(CARPLANE_W, CARPLANE_H, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    // Phase 2.13 (#273) — toggle transform_8x8 via env to bisect.
    enc.enable_transform_8x8 = std::env::var_os("PHASM_NO_T8X8").is_none();
    enc.enable_b_frames = true;

    let mut bitstream = Vec::new();
    let mut recon_dumps: Vec<(u32, FrameType, Vec<u8>)> = Vec::new();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame),
            FrameType::P => enc.encode_p_frame(frame),
            FrameType::B => enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bitstream.extend_from_slice(&bytes);
        recon_dumps.push((eo.display_idx, eo.frame_type, enc.visual_recon.y.clone()));
    }

    let h264 = std::env::temp_dir().join(format!("phasm_2_8_force_carplane_{force_mode}.h264"));
    let dec_path = std::env::temp_dir().join(format!("phasm_2_8_force_carplane_{force_mode}.dec.yuv"));
    std::fs::write(&h264, &bitstream).unwrap();
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec_path)
        .status().expect("ffmpeg");
    assert!(status.success(), "ffmpeg decode failed");
    let decoded = std::fs::read(&dec_path).unwrap();
    let y_size = (CARPLANE_W * CARPLANE_H) as usize;

    let mut sorted: Vec<_> = recon_dumps.iter().collect();
    sorted.sort_by_key(|(d, _, _)| *d);
    eprintln!("=== mode={force_mode} divergence summary (carplane) ===");
    let mb_w = (CARPLANE_W / 16) as usize;
    let mut first_b_dump: Option<(u32, usize, usize)> = None;
    for tup in &sorted {
        let display_idx: u32 = tup.0;
        let ft = tup.1;
        let enc_y: &Vec<u8> = &tup.2;
        let off = (display_idx as usize) * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut diff_sum = 0u64;
        let mut max_abs = 0u32;
        let mut nz = 0u32;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            diff_sum += d as u64;
            if d > 0 { nz += 1; if d > max_abs { max_abs = d; } }
        }
        let avg = diff_sum as f64 / y_size as f64;
        let pct = nz as f64 / y_size as f64 * 100.0;
        eprintln!("  mode={force_mode}  d={display_idx}  type={ft:?}  avg|Δ|={avg:.3}  max|Δ|={max_abs}  nz%={pct:.2}");
        // Capture the first diverging B-frame's first per-MB hotspot
        // (max|Δ|>=2) for diff dump.
        if first_b_dump.is_none() && matches!(ft, FrameType::B) && max_abs >= 2 {
            'outer: for mby in 0..(CARPLANE_H / 16) as usize {
                for mbx in 0..mb_w {
                    let mut max_in_mb = 0u32;
                    for dy in 0..16 {
                        for dx in 0..16 {
                            let idx = (mby * 16 + dy) * CARPLANE_W as usize + mbx * 16 + dx;
                            let d = (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
                            if d > max_in_mb { max_in_mb = d; }
                        }
                    }
                    if max_in_mb >= 2 {
                        first_b_dump = Some((display_idx, mbx, mby));
                        break 'outer;
                    }
                }
            }
        }
    }
    if let Some((display_idx, mbx, mby)) = first_b_dump {
        eprintln!("\n=== diff dump: mode={force_mode} d={display_idx} mb=({mbx},{mby}) ===");
        let enc_y: &Vec<u8> = &sorted.iter().find(|t| t.0 == display_idx).unwrap().2;
        let off = display_idx as usize * frame_size;
        let dec_y = &decoded[off..off + y_size];
        eprintln!("  enc.visual_recon (16x16):");
        for dy in 0..16 {
            eprint!("    ");
            for dx in 0..16 {
                let idx = (mby * 16 + dy) * CARPLANE_W as usize + mbx * 16 + dx;
                eprint!("{:>4}", enc_y[idx]);
            }
            eprintln!();
        }
        eprintln!("  ffmpeg.decode (16x16):");
        for dy in 0..16 {
            eprint!("    ");
            for dx in 0..16 {
                let idx = (mby * 16 + dy) * CARPLANE_W as usize + mbx * 16 + dx;
                eprint!("{:>4}", dec_y[idx]);
            }
            eprintln!();
        }
        eprintln!("  diff (enc - dec):");
        for dy in 0..16 {
            eprint!("    ");
            for dx in 0..16 {
                let idx = (mby * 16 + dy) * CARPLANE_W as usize + mbx * 16 + dx;
                let d = enc_y[idx] as i32 - dec_y[idx] as i32;
                eprint!("{:>+5}", d);
            }
            eprintln!();
        }
    }
    unsafe { std::env::remove_var("PHASM_B_FORCE_MODE"); }
}

