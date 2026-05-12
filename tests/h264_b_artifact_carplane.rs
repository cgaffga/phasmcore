// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// v1.5 §B-fast-motion (#317) — diagnose visible blocky artifacts in
// fast-moving B-frame content (carplane road foreground).
//
// User reported persistent visible block-aligned artifacts on the
// fast-moving road in carplane B-frames across multiple sessions.
// This test:
// 1. Encodes carplane with PHASM_B_INSTRUMENT=1.
// 2. Decodes via ffmpeg + computes per-MB max|delta| vs source.
// 3. Filters to "road area" (bottom 1/3 of frame, mb_y ≥ 80 for the
//    1072×1920 portrait fixture).
// 4. Sorts road MBs by max_dev and reports top 30 with their picked
//    mode + MV info.
// 5. Aggregates mode histogram across all high-deviation road MBs.
//
// Expected output: identifies which mode (Direct/Skip/L0/L1/Bi/etc)
// dominates the artifact set on fast-moving content. That tells us
// the fix class (Direct MV inheritance, ME quality, residual quant,
// etc.).
//
// Run: cargo test --release -p phasm-core --features cabac-stego \
//        --test h264_b_artifact_carplane -- --ignored --nocapture

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::encoder::mb_decision_b::{
    drain_b_intra_fallback_count, drain_b_mb_records, BRdoConfig,
    B_INSTRUMENT_FRAME_IDX,
};
use phasm_core::codec::h264::stego::gop_pattern::{iter_encode_order, FrameType, GopPattern};

const FIXTURE_W: u32 = 1072;
const FIXTURE_H: u32 = 1920;
const N_FRAMES: usize = 30;
const QP: u8 = 26;
// Road area = bottom 1/3 of frame in portrait orientation.
// 1920 / 16 = 120 mb_h. Bottom 1/3 = mb_y >= 80.
const ROAD_MB_Y_MIN: usize = 80;

#[test]
#[ignore]
fn carplane_b_fast_motion_artifact_dump() {
    let yuv_path = format!("/tmp/phasm_v14_carplane_{}x{}_f{}.yuv",
                            FIXTURE_W, FIXTURE_H, N_FRAMES);
    let yuv = std::fs::read(&yuv_path).expect(
        "carplane YUV missing — run h264_v14_multi_ref_carplane_demo first to populate cache",
    );
    let frame_size = (FIXTURE_W * FIXTURE_H * 3 / 2) as usize;
    assert!(yuv.len() >= frame_size * N_FRAMES, "YUV cache too small");
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        // Match production-visual demo settings.
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_CAND");
        std::env::remove_var("PHASM_B_MULTI_REFINE");
        std::env::remove_var("PHASM_B_DIRECT_VALIDATE");
        std::env::set_var("PHASM_DISABLE_SCENECUT", "1");
        std::env::set_var("PHASM_B_INSTRUMENT", "1");
    }

    let _ = drain_b_mb_records();
    let _ = drain_b_intra_fallback_count();
    B_INSTRUMENT_FRAME_IDX.store(0, std::sync::atomic::Ordering::Relaxed);

    let mut enc = Encoder::new(FIXTURE_W, FIXTURE_H, Some(QP)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config = BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== Carplane B-frame fast-motion artifact dump ===");
    eprintln!("Fixture: {}x{} × {}f IBPBP gop=30 b=1 QP={} PRODUCTION_VISUAL",
              FIXTURE_W, FIXTURE_H, N_FRAMES, QP);
    eprintln!("Road area: mb_y ≥ {} (bottom 1/3)", ROAD_MB_Y_MIN);

    let mut bs = Vec::new();
    let mut display_to_b_idx: std::collections::HashMap<u32, u32> =
        std::collections::HashMap::new();
    let mut b_counter = 0u32;
    // §B-fast-motion (#317) — per-B-frame visual_recon snapshot for
    // encoder-vs-decoder divergence bisect.
    let mut enc_recon_y: std::collections::HashMap<u32, Vec<u8>> =
        std::collections::HashMap::new();
    for eo in iter_encode_order(N_FRAMES, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => {
                b_counter += 1;
                display_to_b_idx.insert(eo.display_idx, b_counter);
                enc.encode_b_frame(f)
            }
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
        if matches!(eo.frame_type, FrameType::B) {
            enc_recon_y.insert(eo.display_idx, enc.visual_recon.y.clone());
        }
    }

    let records = drain_b_mb_records();
    let intra_fallback_count = drain_b_intra_fallback_count();
    eprintln!("captured {} B-MB records across {} B-frames",
              records.len(), b_counter);
    eprintln!("§intra-in-B (#319) Phase 2-diagnostic: {} B-MBs ({:.2}%) would qualify for intra-fallback (best_inter_satd > 25000)",
              intra_fallback_count,
              (intra_fallback_count as f32 / records.len().max(1) as f32) * 100.0);

    let h264 = std::env::temp_dir().join("phasm_carplane_artifact.h264");
    let dec_path = std::env::temp_dir().join("phasm_carplane_artifact.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let st = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec_path)
        .status().expect("ffmpeg");
    assert!(st.success(), "ffmpeg decode failed");
    let decoded = std::fs::read(&dec_path).unwrap();
    let y_size = (FIXTURE_W * FIXTURE_H) as usize;
    let c_size = y_size / 4;

    #[derive(Debug, Clone)]
    struct MbDev {
        display_idx: u32,
        b_idx: u32,
        mb_x: usize,
        mb_y: usize,
        max_y: u8,
        sum_y: u32,
        max_u: u8,
        max_v: u8,
        src_var: u32,  // proxy for content texture
    }
    let mb_w = (FIXTURE_W / 16) as usize;
    let mb_h = (FIXTURE_H / 16) as usize;
    let mut road_devs: Vec<MbDev> = Vec::new();
    let mut all_devs_for_hist: Vec<MbDev> = Vec::new();

    for (&display_idx, &b_idx) in &display_to_b_idx {
        let off = display_idx as usize * frame_size;
        let src_y = &yuv[off..off + y_size];
        let src_u = &yuv[off + y_size..off + y_size + c_size];
        let src_v = &yuv[off + y_size + c_size..off + frame_size];
        let dec_y = &decoded[off..off + y_size];
        let dec_u = &decoded[off + y_size..off + y_size + c_size];
        let dec_v = &decoded[off + y_size + c_size..off + frame_size];

        for mby in 0..mb_h {
            for mbx in 0..mb_w {
                let mut max_y = 0u8;
                let mut sum_y = 0u32;
                let mut sum_src = 0u32;
                let mut sum_src_sq = 0u64;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = (mby * 16 + dy) * FIXTURE_W as usize + mbx * 16 + dx;
                        let s = src_y[idx];
                        let d = (s as i32 - dec_y[idx] as i32).unsigned_abs();
                        sum_y += d;
                        sum_src += s as u32;
                        sum_src_sq += (s as u64) * (s as u64);
                        if (d as u8) > max_y { max_y = d as u8; }
                    }
                }
                let n = 256u32;
                let mean = sum_src / n;
                let var = ((sum_src_sq / n as u64) as u32)
                    .saturating_sub(mean * mean);

                let mut max_u = 0u8;
                let mut max_v = 0u8;
                let cw = (FIXTURE_W / 2) as usize;
                for dy in 0..8 {
                    for dx in 0..8 {
                        let cidx = (mby * 8 + dy) * cw + mbx * 8 + dx;
                        let du = (src_u[cidx] as i32 - dec_u[cidx] as i32).unsigned_abs();
                        let dv = (src_v[cidx] as i32 - dec_v[cidx] as i32).unsigned_abs();
                        if (du as u8) > max_u { max_u = du as u8; }
                        if (dv as u8) > max_v { max_v = dv as u8; }
                    }
                }

                let dev = MbDev {
                    display_idx,
                    b_idx,
                    mb_x: mbx,
                    mb_y: mby,
                    max_y,
                    sum_y,
                    max_u,
                    max_v,
                    src_var: var,
                };
                if max_y > 20 || max_u > 20 || max_v > 20 {
                    all_devs_for_hist.push(dev.clone());
                }
                if mby >= ROAD_MB_Y_MIN && (max_y > 20 || max_u > 20 || max_v > 20) {
                    road_devs.push(dev);
                }
            }
        }
    }

    eprintln!("\n=== Road area B-MBs (mb_y ≥ {}) ===", ROAD_MB_Y_MIN);
    eprintln!("Total deviated MBs (max|Δ|>20 anywhere): {}", road_devs.len());

    let mut all_y_devs: Vec<MbDev> = all_devs_for_hist.iter()
        .filter(|d| d.max_y >= 12)
        .cloned().collect();
    // Sort by sum_y (mean-abs-delta proxy) which captures DC quilt
    // (uniform per-MB tone shift = visible block). max_y captures
    // pixel spikes which are NOT visible as blockiness.
    all_y_devs.sort_by_key(|d| std::cmp::Reverse(d.sum_y));
    eprintln!("\n=== TOP 30 highest mean-Y-deviation B-MBs (DC-quilt fingerprint) ===");
    eprintln!("Total Y-deviated MBs (max_y >= 12): {}", all_y_devs.len());
    eprintln!("Sort key: sum_y (= sum |src-dec| over 256 pixels). Avg/pixel = sum_y/256.");
    let mut all_y20_count = 0;
    let mut all_y15_count = 0;
    let mut all_y12_count = 0;
    for d in &all_devs_for_hist {
        if d.max_y >= 20 { all_y20_count += 1; }
        if d.max_y >= 15 { all_y15_count += 1; }
        if d.max_y >= 12 { all_y12_count += 1; }
    }
    eprintln!("  max_y >= 20: {}   max_y >= 15: {}   max_y >= 12: {}",
              all_y20_count, all_y15_count, all_y12_count);
    let mode_names_g = ["Skip", "Direct", "L0", "L1", "Bi"];
    for (rank, dev) in all_y_devs.iter().take(30).enumerate() {
        let rec = records.iter().find(|r| {
            r.frame_idx == dev.b_idx
                && r.mb_x as usize == dev.mb_x
                && r.mb_y as usize == dev.mb_y
        });
        let mode_name = match rec.map(|r| r.mode_id).unwrap_or(255) {
            x if (x as usize) < mode_names_g.len() => mode_names_g[x as usize],
            _ => "?(no rec)",
        };
        eprintln!(
            "#{:>2} d={:>2} mb=({:>3},{:>3}) src_var={:>5} max|Δ|: Y={:>3} U={:>3} V={:>3} sumY={:>5}  mode={}",
            rank + 1, dev.display_idx, dev.mb_x, dev.mb_y,
            dev.src_var, dev.max_y, dev.max_u, dev.max_v, dev.sum_y, mode_name,
        );
        if let Some(r) = rec {
            eprintln!(
                "    mvL0=({:>4},{:>4}) mvL1=({:>4},{:>4}) directL0=({:>4},{:>4}) directL1=({:>4},{:>4}) src_min={} src_max={} src_mean={}",
                r.mv_l0_x, r.mv_l0_y, r.mv_l1_x, r.mv_l1_y,
                r.direct_mv_l0_x, r.direct_mv_l0_y,
                r.direct_mv_l1_x, r.direct_mv_l1_y,
                r.src_min, r.src_max, r.src_mean,
            );
            eprintln!(
                "    SATDs: SkipOrDirect={:>5} L0={:>5} L1={:>5} Bi={:>5}",
                r.satd_skip_or_direct, r.satd_l0, r.satd_l1, r.satd_bi,
            );
        }
    }

    // Spatial heatmap: count Y-deviated MBs per (mb_y_band) row, accumulated across all B-frames
    // §B-fast-motion (#317) — for the top 30 worst MBs, also compute
    // encoder's own recon vs source. If encoder-delta is small but
    // decoder-delta is large → encoder/decoder divergence bug. If both
    // are large → genuine prediction failure.
    eprintln!("\n=== ENC vs DEC divergence bisect (top 30 worst-by-max_y) ===");
    eprintln!("enc_max_y = max|src - enc.visual_recon|, dec_max_y = max|src - ffmpeg.decode|");
    eprintln!("if enc_max_y << dec_max_y → encoder/decoder bitstream divergence");
    for (rank, dev) in all_y_devs.iter().take(30).enumerate() {
        let off = dev.display_idx as usize * frame_size;
        let src_y_full = &yuv[off..off + y_size];
        let enc_y_full = match enc_recon_y.get(&dev.display_idx) {
            Some(v) => v,
            None => continue,
        };
        let mut enc_max = 0u8;
        let mut dec_max = 0u8;
        let mut enc_sum = 0u32;
        let mut dec_sum = 0u32;
        let dec_y_full = &decoded[off..off + y_size];
        for dy in 0..16 {
            for dx in 0..16 {
                let idx = (dev.mb_y * 16 + dy) * FIXTURE_W as usize + dev.mb_x * 16 + dx;
                let s = src_y_full[idx] as i32;
                let e = enc_y_full[idx] as i32;
                let d = dec_y_full[idx] as i32;
                let de = (s - e).unsigned_abs();
                let dd = (s - d).unsigned_abs();
                if (de as u8) > enc_max { enc_max = de as u8; }
                if (dd as u8) > dec_max { dec_max = dd as u8; }
                enc_sum += de;
                dec_sum += dd;
            }
        }
        // Also compare enc vs dec directly (should be 0 if identical)
        let mut enc_dec_max = 0u8;
        for dy in 0..16 {
            for dx in 0..16 {
                let idx = (dev.mb_y * 16 + dy) * FIXTURE_W as usize + dev.mb_x * 16 + dx;
                let dd = (enc_y_full[idx] as i32 - dec_y_full[idx] as i32).unsigned_abs();
                if (dd as u8) > enc_dec_max { enc_dec_max = dd as u8; }
            }
        }
        eprintln!(
            "#{:>2} d={:>2} mb=({:>3},{:>3}) | enc_max={:>3} (sum={:>5}) | dec_max={:>3} (sum={:>5}) | enc-vs-dec={:>3}",
            rank + 1, dev.display_idx, dev.mb_x, dev.mb_y,
            enc_max, enc_sum, dec_max, dec_sum, enc_dec_max,
        );
    }

    eprintln!("\n=== Spatial Y-deviation density (max_y>=12, all frames) ===");
    let mb_h_bands = 12;
    let band_height = (mb_h + mb_h_bands - 1) / mb_h_bands;
    let mut bands: Vec<(u32, u32)> = vec![(0u32, 0u32); mb_h_bands]; // (count, max_y_sum)
    for d in &all_y_devs {
        let band = (d.mb_y / band_height).min(mb_h_bands - 1);
        bands[band].0 += 1;
        bands[band].1 += d.max_y as u32;
    }
    for (i, (count, sum)) in bands.iter().enumerate() {
        let lo = i * band_height;
        let hi = ((i + 1) * band_height).min(mb_h);
        let avg = if *count > 0 { *sum as f32 / *count as f32 } else { 0.0 };
        eprintln!("  mb_y band [{:>3}..{:>3}]: count={:>5}  avg_max_y={:.1}",
                  lo, hi, count, avg);
    }

    // Mode histogram for FULL FRAME deviated MBs
    let mut all_mode_hist: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    for d in &all_y_devs {
        let rec = records.iter().find(|r| {
            r.frame_idx == d.b_idx
                && r.mb_x as usize == d.mb_x
                && r.mb_y as usize == d.mb_y
        });
        let name = match rec.map(|r| r.mode_id).unwrap_or(255) {
            x if (x as usize) < mode_names_g.len() => mode_names_g[x as usize],
            _ => "Skip-fast",
        };
        *all_mode_hist.entry(name).or_insert(0) += 1;
    }
    eprintln!("\n=== FULL FRAME Mode histogram across {} Y-deviated MBs ===", all_y_devs.len());
    let mut all_hist_sorted: Vec<_> = all_mode_hist.iter().collect();
    all_hist_sorted.sort_by_key(|&(_, n)| std::cmp::Reverse(*n));
    for (mode, count) in all_hist_sorted {
        let pct = (*count as f32 / all_y_devs.len().max(1) as f32) * 100.0;
        eprintln!("  {:>20}: {:>5} ({:>5.1}%)", mode, count, pct);
    }

    road_devs.sort_by_key(|d| std::cmp::Reverse(d.max_y));
    eprintln!("\n=== TOP 30 worst-luma-deviation road B-MBs ===");
    let mode_names = ["Skip", "Direct", "L0", "L1", "Bi"];
    for (rank, dev) in road_devs.iter().take(30).enumerate() {
        let rec = records.iter().find(|r| {
            r.frame_idx == dev.b_idx
                && r.mb_x as usize == dev.mb_x
                && r.mb_y as usize == dev.mb_y
        });
        let mode_name = match rec.map(|r| r.mode_id).unwrap_or(255) {
            x if (x as usize) < mode_names.len() => mode_names[x as usize],
            _ => "?(no record/Skip-fast)",
        };
        eprintln!(
            "#{:>2} d={:>2} mb=({:>3},{:>3}) src_var={:>5} max|Δ|: Y={:>3} U={:>3} V={:>3} sumY={:>5}  mode={}",
            rank + 1, dev.display_idx, dev.mb_x, dev.mb_y,
            dev.src_var, dev.max_y, dev.max_u, dev.max_v, dev.sum_y, mode_name,
        );
        if let Some(r) = rec {
            eprintln!(
                "    mvL0=({:>4},{:>4}) mvL1=({:>4},{:>4}) directL0=({:>4},{:>4}) directL1=({:>4},{:>4}) bdy={}",
                r.mv_l0_x, r.mv_l0_y, r.mv_l1_x, r.mv_l1_y,
                r.direct_mv_l0_x, r.direct_mv_l0_y,
                r.direct_mv_l1_x, r.direct_mv_l1_y,
                r.at_boundary,
            );
            eprintln!(
                "    SATDs: SkipOrDirect={:>5} L0={:>5} L1={:>5} Bi={:>5}",
                r.satd_skip_or_direct, r.satd_l0, r.satd_l1, r.satd_bi,
            );
        }
    }

    // Mode histogram across ALL deviated road MBs (not just top-30).
    let mut mode_hist: std::collections::HashMap<&str, u32> =
        std::collections::HashMap::new();
    let mut mode_max_y: std::collections::HashMap<&str, u32> =
        std::collections::HashMap::new();
    for dev in &road_devs {
        let rec = records.iter().find(|r| {
            r.frame_idx == dev.b_idx
                && r.mb_x as usize == dev.mb_x
                && r.mb_y as usize == dev.mb_y
        });
        let name = match rec.map(|r| r.mode_id).unwrap_or(255) {
            x if (x as usize) < mode_names.len() => mode_names[x as usize],
            _ => "Skip-fast/no-rec",
        };
        *mode_hist.entry(name).or_insert(0) += 1;
        let entry = mode_max_y.entry(name).or_insert(0);
        if dev.max_y as u32 > *entry { *entry = dev.max_y as u32; }
    }
    eprintln!("\n=== Mode histogram across all {} deviated road MBs ===", road_devs.len());
    let mut hist_sorted: Vec<_> = mode_hist.iter().collect();
    hist_sorted.sort_by_key(|&(_, n)| std::cmp::Reverse(*n));
    for (mode, count) in hist_sorted {
        let pct = (*count as f32 / road_devs.len().max(1) as f32) * 100.0;
        let max_y = mode_max_y.get(mode).unwrap_or(&0);
        eprintln!("  {:>20}: {:>5} ({:>5.1}%) | max|Δ|_Y in this mode: {}",
                  mode, count, pct, max_y);
    }

    // Cross-frame distribution
    let mut frame_hist: std::collections::HashMap<u32, u32> =
        std::collections::HashMap::new();
    for dev in &road_devs {
        *frame_hist.entry(dev.display_idx).or_insert(0) += 1;
    }
    let mut frame_sorted: Vec<_> = frame_hist.iter().collect();
    frame_sorted.sort_by_key(|&(d, _)| *d);
    eprintln!("\n=== Per-frame deviated-road-MB count ===");
    for (display_idx, count) in frame_sorted {
        eprintln!("  display={:>2}: {:>4} deviated road MBs", display_idx, count);
    }

    // Compare road vs non-road density
    let non_road_count = all_devs_for_hist.iter()
        .filter(|d| d.mb_y < ROAD_MB_Y_MIN).count();
    eprintln!("\n=== Spatial split ===");
    eprintln!("  road area    (mb_y >= {}): {} deviated MBs", ROAD_MB_Y_MIN, road_devs.len());
    eprintln!("  non-road     (mb_y < {}):  {} deviated MBs", ROAD_MB_Y_MIN, non_road_count);

    unsafe {
        std::env::remove_var("PHASM_B_INSTRUMENT");
    }

    let _ = std::fs::remove_file(&h264);
    let _ = std::fs::remove_file(&dec_path);
    eprintln!("\n=== END artifact dump ===");
}
