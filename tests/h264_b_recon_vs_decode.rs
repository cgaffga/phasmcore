// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §B-RDO.debug.4 (2026-05-04) — compare phasm encoder's internal
// recon for each B-frame to ffmpeg's decode of the same bitstream.
//
// Phase 1.1.C (2026-05-05): the right comparand is now
// `enc.visual_recon`, not `enc.recon`. After Phase 1.1.B,
// encoder.recon stays PRE-flip (used as reference for next-MB
// mode-decision; preserves cover-capture invariant), while
// visual_recon tracks the POST-flip + deblocked reconstruction that
// a downstream player reproduces. So visual_recon is the buffer
// that should match ffmpeg.decode for stego-active runs.
//
// If enc.visual_recon == ffmpeg.decode → encoder is honest; visual
// artifact is content/mode-decision-driven, not encoder bug.
// If enc.visual_recon != ffmpeg.decode → encoder/decoder semantic
// divergence (post-flip residual emit math doesn't match the spec
// decode side).

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::stego::gop_pattern::{
    iter_encode_order, FrameType, GopPattern,
};

#[test]
#[ignore]
fn dump_b_frame_recon_vs_decode() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 12usize; // 2026-05-06: bumped from 6 to 12 to reach d=10 P explosion
    let frame_size = (width * height * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;

    // Build the bitstream + capture enc.recon Y plane after every frame in encode order.
    let mut bitstream = Vec::new();
    let mut recon_dumps: Vec<(u32, FrameType, Vec<u8>)> = Vec::new();
    let mut intra_dumps: Vec<(u32, Vec<bool>)> = Vec::new();
    let mut mv_dumps: Vec<(
        u32,
        Vec<Option<(phasm_core::codec::h264::encoder::motion_estimation::MotionVector, i8)>>,
    )> = Vec::new();
    // §B-direct-fix Stage 2 (#232) follow-up — capture DPB last_ref Y
    // plane after each frame's encode. Used to verify the encoder's
    // reference for the NEXT P/B frame matches ffmpeg's reference.
    let mut dpb_dumps: Vec<(u32, Option<Vec<u8>>)> = Vec::new();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame),
            FrameType::P => enc.encode_p_frame(frame),
            FrameType::B => enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("encode (encode_idx={}, display={}): {e}", eo.encode_idx, eo.display_idx));
        bitstream.extend_from_slice(&bytes);
        // 2026-05-07 sanity check: enc.recon == enc.visual_recon when no
        // stego is active? P-frames=true, B-frames=false.
        // The proper comparand for ffmpeg.decode is enc.visual_recon
        // (POST-flip, post-residual). enc.recon stays at cover-pristine
        // for B-frames per Phase 1.1.B/C.
        let recon_eq_visual = enc.recon.y == enc.visual_recon.y;
        eprintln!("    [sanity] display={}  enc.recon == enc.visual_recon ? {}",
            eo.display_idx, recon_eq_visual);
        // 2026-05-07: USE visual_recon as the comparand. B-frames
        // diverge between enc.recon and enc.visual_recon even without
        // stego — comparing the wrong buffer was the source of the
        // "encoder/decoder divergence" measurement artifact.
        recon_dumps.push((eo.display_idx, eo.frame_type, enc.visual_recon.y.clone()));
        intra_dumps.push((eo.display_idx, enc.intra_grid().to_vec()));
        let mb_w = (width / 16) as u32;
        let mb_h = (height / 16) as u32;
        let mut mv_grid_snap = Vec::with_capacity((mb_w * mb_h) as usize);
        for mby in 0..mb_h {
            for mbx in 0..mb_w {
                mv_grid_snap.push(enc.mv_l0_at_mb(mbx, mby));
            }
        }
        mv_dumps.push((eo.display_idx, mv_grid_snap));
        // Snapshot of dpb.last_ref Y plane AFTER this frame's encode.
        // For checking what reference NEXT P/B frame will see.
        let dpb_y = enc.dpb.last_ref.as_ref().map(|r| r.y.clone());
        dpb_dumps.push((eo.display_idx, dpb_y));
    }

    // Write the bitstream + decode it via ffmpeg.
    let h264_path = std::env::temp_dir().join("phasm_b_recon_probe.h264");
    let dec_path = std::env::temp_dir().join("phasm_b_recon_probe.dec.yuv");
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
    let y_size = (width * height) as usize;

    eprintln!("\n=== Encoder.recon vs ffmpeg.decode (Y plane) ===");
    for (display_idx, ft, enc_y) in &recon_dumps {
        let off = (*display_idx as usize) * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut diff = 0u64;
        let mut max_abs = 0u32;
        let mut nz_pixels = 0u32;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            diff += d as u64;
            if d > 0 {
                nz_pixels += 1;
                if d > max_abs { max_abs = d; }
            }
        }
        let avg = (diff as f64) / (y_size as f64);
        eprintln!(
            "  display={}  type={:?}  avg|Δ|={:.3}  max|Δ|={}  nonzero_pixels={} ({:.2}%)",
            display_idx, ft, avg, max_abs, nz_pixels,
            100.0 * (nz_pixels as f64) / (y_size as f64)
        );

        // §B-direct-fix Stage 2 (#232) follow-up — per-MB hotspot map
        // for diverging frames. P4 / B5 still ~47% divergent after the
        // B_Skip/B_Direct/P_Skip fix; localize which MBs diverge to
        // identify the remaining recon-only path.
        // 2026-05-07: lowered from 0.20 to 0.02. After bin-6 fix +
        // visual_recon comparand fix, the remaining B-frame tail
        // diverges with nz% in 3-10% range — was being missed by the
        // 0.20 trigger.
        if nz_pixels as f64 / (y_size as f64) > 0.02 {
            let mb_w = (width / 16) as usize;
            let mb_h = (height / 16) as usize;
            let mut mb_diverge: Vec<(usize, usize, u32)> = Vec::new();
            for mby in 0..mb_h {
                for mbx in 0..mb_w {
                    let mut mb_diff = 0u32;
                    for dy in 0..16 {
                        for dx in 0..16 {
                            let row = mby * 16 + dy;
                            let col = mbx * 16 + dx;
                            let idx = row * (width as usize) + col;
                            let d = (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
                            mb_diff += d;
                        }
                    }
                    if mb_diff > 0 {
                        mb_diverge.push((mbx, mby, mb_diff));
                    }
                }
            }
            // Find FIRST diverging MB in raster scan order (before
            // sorting by magnitude). If CABAC cascade causes the
            // divergence, the first MB to diverge is the actual root.
            let mut first_div: Vec<(usize, usize, u32)> = mb_diverge.clone();
            first_div.sort_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));
            eprintln!("    first 5 diverging MBs in raster order (mbx, mby, sum|Δ|):");
            for (i, (mbx, mby, d)) in first_div.iter().take(5).enumerate() {
                eprintln!("      [{:>2}] mb=({:>3},{:>3})  Σ|Δ|={}", i, mbx, mby, d);
            }
            mb_diverge.sort_by(|a, b| b.2.cmp(&a.2));
            let total_mbs = mb_w * mb_h;
            let div_count = mb_diverge.len();
            eprintln!("    diverging MBs: {}/{} ({:.1}%)",
                div_count, total_mbs, 100.0 * div_count as f64 / total_mbs as f64);
            // Find this frame's intra grid for mode-tagging hotspots.
            let intra_g_opt = intra_dumps.iter()
                .find(|(d, _)| d == display_idx)
                .map(|(_, g)| g);
            // Find this frame's MV grid for MV-tagging hotspots.
            let mv_g_opt = mv_dumps.iter()
                .find(|(d, _)| d == display_idx)
                .map(|(_, g)| g);
            eprintln!("    top-10 hotspot MBs (mbx, mby, sum|Δ|, mode, mv):");
            for (i, (mbx, mby, d)) in mb_diverge.iter().take(10).enumerate() {
                let idx = mby * mb_w + mbx;
                let mode = intra_g_opt.map(|g| {
                    if idx < g.len() && g[idx] { "INTRA" } else { "inter" }
                }).unwrap_or("?");
                let mv_str = mv_g_opt.and_then(|g| {
                    if idx < g.len() {
                        Some(match g[idx] {
                            Some((mv, ref_idx)) => format!("({},{}) refIdx={}", mv.mv_x, mv.mv_y, ref_idx),
                            None => "intra/no-MV".to_string(),
                        })
                    } else { None }
                }).unwrap_or_else(|| "?".to_string());
                eprintln!("      [{:>2}] mb=({:>3},{:>3})  Σ|Δ|={}  mode={}  mv={}",
                    i, mbx, mby, d, mode, mv_str);
            }
            // Count intra vs inter among diverging MBs.
            if let Some(g) = intra_g_opt {
                let mut intra_div = 0u32;
                let mut inter_div = 0u32;
                for (mbx, mby, _) in &mb_diverge {
                    let idx = mby * mb_w + mbx;
                    if idx < g.len() && g[idx] { intra_div += 1; } else { inter_div += 1; }
                }
                eprintln!("    of {} diverging MBs: {} INTRA, {} inter",
                    div_count, intra_div, inter_div);
            }
            // §B-direct-fix Stage 2 (#232) follow-up — dump pixel
            // diff for the #1 hotspot MB. If diff is roughly constant
            // → DC offset (rare). If diff has pixel-shift pattern →
            // MV mismatch (encoder used MV_X, emitted MVD that
            // decodes to MV_Y). If diff has texture but smaller
            // magnitude → residual rounding mismatch. If diff is
            // sharp at edges → deblocking divergence.
            if let Some(&(mbx, mby, _)) = first_div.first() {
                eprintln!("\n    --- FIRST diverging MB diff dump: mb=({},{}) ---", mbx, mby);
                eprintln!("    enc.recon (Y, 16x16):");
                for dy in 0..16 {
                    let row = mby * 16 + dy;
                    eprint!("      ");
                    for dx in 0..16 {
                        let col = mbx * 16 + dx;
                        let idx = row * (width as usize) + col;
                        eprint!("{:>3} ", enc_y[idx]);
                    }
                    eprintln!();
                }
                eprintln!("    ffmpeg.decode (Y, 16x16):");
                for dy in 0..16 {
                    let row = mby * 16 + dy;
                    eprint!("      ");
                    for dx in 0..16 {
                        let col = mbx * 16 + dx;
                        let idx = row * (width as usize) + col;
                        eprint!("{:>3} ", dec_y[idx]);
                    }
                    eprintln!();
                }
                eprintln!("    diff (enc - dec):");
                for dy in 0..16 {
                    let row = mby * 16 + dy;
                    eprint!("      ");
                    for dx in 0..16 {
                        let col = mbx * 16 + dx;
                        let idx = row * (width as usize) + col;
                        let d = enc_y[idx] as i32 - dec_y[idx] as i32;
                        eprint!("{:>+4} ", d);
                    }
                    eprintln!();
                }
                // §B-direct-fix Stage 2 (#232) follow-up — for inter
                // hotspots, dump the L0-reference patch the encoder
                // sampled (DPB.last_ref at MV offset) and compare to
                // ffmpeg's view (= ffmpeg.decode[P2] at the same offset)
                // to verify the references match.
                let mv_g = mv_g_opt.expect("mv_g for hotspot frame");
                let mb_idx = mby * mb_w + mbx;
                if let Some(Some((mv, ref_idx))) = mv_g.get(mb_idx).copied() {
                    let _ = ref_idx;
                    // Find the previous frame's DPB snapshot (the L0
                    // reference for this frame). For P-frames in IBPBP,
                    // prev anchor is display - 2 (P4 → P2 → display=2;
                    // P2 → I0 → display=0). For B-frames in IBPBP, L0
                    // is the prev anchor at display - 1.
                    let prev_anchor_display = match ft {
                        FrameType::P => display_idx.saturating_sub(2),
                        FrameType::B => display_idx.saturating_sub(1),
                        FrameType::Idr => 0,
                    };
                    let dpb_snap = dpb_dumps.iter()
                        .find(|(d, _)| *d == prev_anchor_display)
                        .and_then(|(_, dpb_y)| dpb_y.as_ref());
                    // MV is in quarter-pels; convert to integer-pixel offset.
                    // Block top-left in pixels: (mbx*16, mby*16).
                    // Predicted block top-left: (mbx*16 + mv.x/4, mby*16 + mv.y/4).
                    let pred_x = (mbx as i32) * 16 + (mv.mv_x as i32) / 4;
                    let pred_y_pos = (mby as i32) * 16 + (mv.mv_y as i32) / 4;
                    eprintln!("    encoder MV=({},{}) qpel → pred patch top-left in P2 = ({},{})",
                        mv.mv_x, mv.mv_y, pred_x, pred_y_pos);
                    if let Some(dpb_y) = dpb_snap {
                        eprintln!("    encoder.dpb.last_ref @ pred patch (Y, 16x16):");
                        for dy in 0..16 {
                            let row = pred_y_pos + dy;
                            eprint!("      ");
                            for dx in 0..16 {
                                let col = pred_x + dx;
                                if row < 0 || row >= height as i32 || col < 0 || col >= width as i32 {
                                    eprint!("  -- ");
                                } else {
                                    let idx = (row as usize) * (width as usize) + (col as usize);
                                    eprint!("{:>3} ", dpb_y[idx]);
                                }
                            }
                            eprintln!();
                        }
                    }
                    // Also dump ffmpeg.decode[P2] at the same predicted patch.
                    let p2_off = (prev_anchor_display as usize) * frame_size;
                    let p2_y = &decoded[p2_off..p2_off + y_size];
                    eprintln!("    ffmpeg.decode[display={}] @ pred patch (Y, 16x16):", prev_anchor_display);
                    for dy in 0..16 {
                        let row = pred_y_pos + dy;
                        eprint!("      ");
                        for dx in 0..16 {
                            let col = pred_x + dx;
                            if row < 0 || row >= height as i32 || col < 0 || col >= width as i32 {
                                eprint!("  -- ");
                            } else {
                                let idx = (row as usize) * (width as usize) + (col as usize);
                                eprint!("{:>3} ", p2_y[idx]);
                            }
                        }
                        eprintln!();
                    }
                }
            }
        }
    }
}

/// 2026-05-05 §B-cascade-real Bug #2 verification: does P-frame
/// encoder.recon diverge from ffmpeg.decode when stego flips happen?
///
/// The hypothesis (per `memory/h264_stego_b_cascade_2026_05_05.md`):
/// `encoder.rs:3623` dequants from `levels_8x8` (pre-flip) while CABAC
/// emits from `scan` (post-flip). This test runs the orchestrator with
/// stego ON, captures encoder's per-frame internal recon (via the
/// no-stego transcode path's intermediate recon — same code), and
/// compares to ffmpeg.decode. If P-frames diverge under stego (when
/// they didn't under no-stego per the existing test), Bug #2 is
/// confirmed and that's where the fix lands.
///
/// NOTE: this test cannot easily expose the stego orchestrator's
/// internal recon (it's a 3-pass pipeline, recon at end of pass 3 is
/// the final state). What we CAN compare is the cumulative recon
/// drift via the per-frame Y-PSNR pattern: V4 cliff at frame 5 is the
/// observable signature of the bug. Documenting via cross-config diff
/// rather than direct recon dump.
#[test]
#[ignore]
fn cascade_bug2_marker() {
    eprintln!("see fast_stego_cascade_probe per-frame data for V4 cliff signature");
}

/// 2026-05-05 fast cascade probe via the actual stego orchestrator.
///
/// Runs `h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files`
/// directly on a small YUV fixture (no CLI demux/mux), decodes via
/// ffmpeg, computes per-frame Y-PSNR vs source, plus the no-stego
/// reference. Tens-of-seconds total — fast enough to iterate.
///
/// Reads `PHASM_B_RDO` and `PHASM_B_RESIDUAL` from the environment.
///
/// **Gated until #219 lands** — depends on `h264_transcode_yuv_no_stego`
/// (encoder-only baseline) which was reverted along with the §B-cascade-real
/// write-back work on 2026-05-05. Re-enable when the v1.1 cascade architecture
/// is in place.
#[cfg(feature = "_disabled_until_no_stego_helper")]
#[test]
#[ignore]
fn fast_stego_cascade_probe() {
    use phasm_core::{
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files,
        h264_transcode_yuv_no_stego,
    };
    let yuv = std::fs::read("/tmp/iphone7_1920x1072_f60.yuv")
        .expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 12usize;
    let frame_size = (width * height * 3 / 2) as usize;
    let yuv = &yuv[..frame_size * n_frames];
    let pattern = GopPattern::Ibpbp { gop: 12, b_count: 1 };

    let rdo = std::env::var("PHASM_B_RDO").unwrap_or_else(|_| "(unset)".into());
    let res = std::env::var("PHASM_B_RESIDUAL").unwrap_or_else(|_| "(unset)".into());
    eprintln!("\n=== cascade probe: PHASM_B_RDO={rdo}  PHASM_B_RESIDUAL={res} ===");

    let t0 = std::time::Instant::now();
    let stego_bs = h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
        yuv, width, height, n_frames, pattern, "probe", &[], "passprobe",
    ).expect("stego encode");
    let t_stego = t0.elapsed();
    eprintln!("stego encode: {:?}, {} bytes", t_stego, stego_bs.len());

    let t1 = std::time::Instant::now();
    let clean_bs = h264_transcode_yuv_no_stego(yuv, width, height, n_frames, pattern)
        .expect("clean encode");
    let t_clean = t1.elapsed();
    eprintln!("clean encode: {:?}, {} bytes", t_clean, clean_bs.len());

    // Decode each via ffmpeg → raw YUV → per-frame PSNR vs source.
    let psnr_each = |bs: &[u8], label: &str| {
        let h264 = std::env::temp_dir().join(format!("phasm_cascade_{label}.h264"));
        let dec = std::env::temp_dir().join(format!("phasm_cascade_{label}.dec.yuv"));
        std::fs::write(&h264, bs).unwrap();
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(&h264)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&dec)
            .status()
            .expect("ffmpeg run");
        assert!(status.success(), "ffmpeg decode failed for {label}");
        let decoded = std::fs::read(&dec).unwrap();
        let y_size = (width * height) as usize;
        let mut psnrs = Vec::new();
        for i in 0..n_frames {
            let off = i * frame_size;
            let src_y = &yuv[off..off + y_size];
            let dec_y = &decoded[off..off + y_size];
            let mut sum_sq: u64 = 0;
            for (a, b) in src_y.iter().zip(dec_y.iter()) {
                let d = (*a as i32 - *b as i32) as i64;
                sum_sq += (d * d) as u64;
            }
            let mse = (sum_sq as f64) / (y_size as f64);
            let psnr = if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() };
            psnrs.push(psnr);
        }
        psnrs
    };

    let stego_psnrs = psnr_each(&stego_bs, "stego");
    let clean_psnrs = psnr_each(&clean_bs, "clean");

    eprintln!("\n=== Per-frame Y-PSNR vs source (12-frame gop=12 single GOP) ===");
    eprintln!("{:>4} {:>10} {:>10} {:>8}", "n", "stego", "clean", "Δ");
    for i in 0..n_frames {
        eprintln!(
            "{:>4} {:>10.2} {:>10.2} {:>+8.2}",
            i + 1, stego_psnrs[i], clean_psnrs[i],
            stego_psnrs[i] - clean_psnrs[i]
        );
    }
    let avg_stego = stego_psnrs.iter().sum::<f64>() / n_frames as f64;
    let avg_clean = clean_psnrs.iter().sum::<f64>() / n_frames as f64;
    eprintln!(" avg {:>10.2} {:>10.2} {:>+8.2}", avg_stego, avg_clean, avg_stego - avg_clean);
}

/// 2026-05-05 cascade-source bisect — same as
/// `dump_b_frame_recon_vs_decode` but reads `PHASM_B_RDO` /
/// `PHASM_B_RESIDUAL` from the actual environment instead of forcing
/// them to "1". Use to test all 4 (rdo, residual) combinations and
/// see which one(s) produce the encoder.recon ≠ ffmpeg.decode
/// divergence.
#[test]
#[ignore]
fn dump_b_frame_recon_vs_decode_env() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 12usize; // bigger window — should expose any P-frame cascade
    let frame_size = (width * height * 3 / 2) as usize;

    // Caller's PHASM_B_RDO / PHASM_B_RESIDUAL pass through unchanged.
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }
    let rdo = std::env::var("PHASM_B_RDO").unwrap_or_else(|_| "(unset)".into());
    let res = std::env::var("PHASM_B_RESIDUAL").unwrap_or_else(|_| "(unset)".into());
    eprintln!("\n=== run config: PHASM_B_RDO={rdo}  PHASM_B_RESIDUAL={res} ===");

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
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
        .unwrap_or_else(|e| panic!("encode (encode_idx={}, display={}): {e}", eo.encode_idx, eo.display_idx));
        bitstream.extend_from_slice(&bytes);
        // §B-cascade-real Phase 1.1.C: capture visual_recon (post-flip
        // + deblocked, mirrors what ffmpeg.decode produces). Was
        // enc.recon (pre-flip) before Phase 1.1.B — comparison is
        // only meaningful against visual_recon now.
        recon_dumps.push((eo.display_idx, eo.frame_type, enc.visual_recon.y.clone()));
    }

    let h264_path = std::env::temp_dir().join("phasm_b_recon_probe_env.h264");
    let dec_path = std::env::temp_dir().join("phasm_b_recon_probe_env.dec.yuv");
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
    let y_size = (width * height) as usize;

    eprintln!("\n=== Encoder.recon vs ffmpeg.decode (Y plane), 12-frame window ===");
    // Sort by display_idx for clearer output
    let mut sorted: Vec<_> = recon_dumps.iter().collect();
    sorted.sort_by_key(|(d, _, _)| *d);
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    for (display_idx, ft, enc_y) in sorted {
        let off = (*display_idx as usize) * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut diff = 0u64;
        let mut max_abs = 0u32;
        let mut nz_pixels = 0u32;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            diff += d as u64;
            if d > 0 {
                nz_pixels += 1;
                if d > max_abs { max_abs = d; }
            }
        }
        let avg = (diff as f64) / (y_size as f64);
        eprintln!(
            "  display={:>2}  type={:?}  avg|Δ|={:.3}  max|Δ|={}  nonzero_pixels={} ({:.2}%)",
            display_idx, ft, avg, max_abs, nz_pixels,
            100.0 * (nz_pixels as f64) / (y_size as f64)
        );
        // §B-direct-fix Stage 2 (#232) follow-up — first diverging MB +
        // a 16×16 dump for any frame with >20% divergence (= the d=10
        // explosion fingerprint). This shows whether the bug is a
        // CABAC desync (sharp boundary, garbage after) or per-MB drift.
        if (nz_pixels as f64) / (y_size as f64) > 0.20 {
            let mut first: Option<(usize, usize, u32)> = None;
            'outer: for mby in 0..mb_h {
                for mbx in 0..mb_w {
                    let mut s = 0u32;
                    for dy in 0..16 {
                        for dx in 0..16 {
                            let idx = (mby * 16 + dy) * (width as usize) + mbx * 16 + dx;
                            s += (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
                        }
                    }
                    if s > 0 {
                        first = Some((mbx, mby, s));
                        break 'outer;
                    }
                }
            }
            if let Some((mbx, mby, s)) = first {
                eprintln!("    first diverging MB: mb=({},{})  Σ|Δ|={}", mbx, mby, s);
                eprintln!("    enc.visual_recon (Y, 16x16):");
                for dy in 0..16 {
                    let row = mby * 16 + dy;
                    eprint!("      ");
                    for dx in 0..16 {
                        let col = mbx * 16 + dx;
                        let idx = row * (width as usize) + col;
                        eprint!("{:>3} ", enc_y[idx]);
                    }
                    eprintln!();
                }
                eprintln!("    ffmpeg.decode (Y, 16x16):");
                for dy in 0..16 {
                    let row = mby * 16 + dy;
                    eprint!("      ");
                    for dx in 0..16 {
                        let col = mbx * 16 + dx;
                        let idx = row * (width as usize) + col;
                        eprint!("{:>3} ", dec_y[idx]);
                    }
                    eprintln!();
                }
            }
            // Where in the frame does divergence start? Find first MB.
            // Then check: do MBs BEFORE the first diverging one match,
            // and MBs AFTER all diverge (= CABAC desync), or scattered?
            let mut total_div_mbs = 0u32;
            let mut clean_after_first = 0u32;
            let mut dirty_after_first = 0u32;
            let mut seen_first = false;
            for mby in 0..mb_h {
                for mbx in 0..mb_w {
                    let mut s = 0u32;
                    for dy in 0..16 {
                        for dx in 0..16 {
                            let idx = (mby * 16 + dy) * (width as usize) + mbx * 16 + dx;
                            s += (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
                        }
                    }
                    let dirty = s > 0;
                    if dirty { total_div_mbs += 1; }
                    if seen_first {
                        if dirty { dirty_after_first += 1; } else { clean_after_first += 1; }
                    } else if dirty {
                        seen_first = true;
                    }
                }
            }
            eprintln!(
                "    {} of {} MBs diverge ({:.1}%); after first diverging MB: {} dirty / {} clean → {}",
                total_div_mbs,
                mb_w * mb_h,
                100.0 * total_div_mbs as f64 / (mb_w * mb_h) as f64,
                dirty_after_first, clean_after_first,
                if clean_after_first == 0 {
                    "all dirty after first → CABAC DESYNC"
                } else {
                    "scattered → per-MB drift"
                }
            );
        }
    }
}

/// **§B-cascade-real Phase 1.1.D — V4 cascade probe** (2026-05-05).
///
/// Validation gate for Phase 1.1.B+C. Runs the full 4-domain stego
/// pipeline on iPhone7 1080p × 10f IBPBP (gop=10, b_count=1, M=2 — the
/// production default). Decodes both stego + encoder-only baselines via
/// ffmpeg, computes per-frame Y-PSNR vs source. Target: avg stego cost
/// ≤ −2 dB matches the audit decision rule "ship as-is" threshold.
///
/// **Setup**: requires `/tmp/iphone7_1920x1072_f10.yuv` (raw YUV420p,
/// 10 frames at 1920×1072 — produced via `ffmpeg -i source.mov -vf
/// "scale=1920:1072,format=yuv420p" -frames 10 -f rawvideo
/// /tmp/iphone7_1920x1072_f10.yuv`) and ffmpeg in PATH. Test is
/// `#[ignore]`'d by default — invoke explicitly via
/// `cargo test --lib --features h264-encoder,cabac-stego --test
/// h264_b_recon_vs_decode phase_1_1_d_v4_cascade_probe_30f --
/// --ignored --nocapture`. Originally written for 30f but reduced
/// to 10f due to streaming-Viterbi pass-1 super-linear scaling at
/// 1080p single-thread debug build (30f probe ran 6h without
/// completing).
#[test]
#[ignore]
fn phase_1_1_d_v4_cascade_probe_30f() {
    use phasm_core::codec::h264::stego::encode_pixels::
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files;

    // Phase 1.1.D pragmatic gate (2026-05-05): single-thread debug
    // build + streaming-Viterbi pass-1 capture scales super-linearly
    // with frame count at 1080p. 30f probe ran 6h and was still in
    // pass1_capture; aborted. 10f gives a meaningful cliff-detection
    // signal (the §B-cascade-real cliff manifests at frame 5+ in
    // prior 12f probes) while completing in ~30-60 min.
    let yuv_path = "/tmp/iphone7_1920x1072_f10.yuv";
    let yuv = std::fs::read(yuv_path)
        .expect("missing /tmp/iphone7_1920x1072_f10.yuv (see test docstring)");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 10usize;
    let frame_size = (width * height * 3 / 2) as usize;
    assert_eq!(
        yuv.len(),
        frame_size * n_frames,
        "fixture size mismatch: expected {} bytes for 1920x1072x30 YUV420p, got {}",
        frame_size * n_frames,
        yuv.len()
    );
    let yuv = &yuv[..frame_size * n_frames];

    let pattern = GopPattern::Ibpbp { gop: 10, b_count: 1 };

    eprintln!("\n=== Phase 1.1.D V4 cascade probe — iPhone7 1080p × 10f IBPBP ===");
    eprintln!("PHASM_B_RDO={}  PHASM_B_RESIDUAL={}",
        std::env::var("PHASM_B_RDO").unwrap_or_else(|_| "(unset)".into()),
        std::env::var("PHASM_B_RESIDUAL").unwrap_or_else(|_| "(unset)".into()),
    );

    // ── 1. Stego encode (full 4-domain orchestrator) ──────────────
    let t0 = std::time::Instant::now();
    let stego_bs =
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
            yuv,
            width,
            height,
            n_frames,
            pattern,
            "phase-1.1.d-probe",
            &[],
            "passprobe-1.1.d",
        )
        .expect("stego encode");
    let t_stego = t0.elapsed();
    eprintln!("stego encode: {:?}, {} bytes", t_stego, stego_bs.len());

    // ── 2. Encoder-only no-stego baseline (no orchestrator hooks) ──
    // Mirrors dump_b_frame_recon_vs_decode_env's encode loop. Same
    // GOP pattern; same QP; just no stego hooks so the bitstream
    // contains pristine pre-flip residuals. Use PRODUCTION_VISUAL
    // (NOT default SAFE) so the comparison isolates "stego on/off"
    // rather than mixing in a config delta — the orchestrator forces
    // PRODUCTION_VISUAL on its encoder (encode_pixels.rs:264).
    let mut clean_enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    clean_enc.entropy_mode = EntropyMode::Cabac;
    clean_enc.enable_transform_8x8 = true;
    clean_enc.enable_b_frames = true;
    clean_enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;
    let mut clean_bs = Vec::new();
    let t1 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => clean_enc.encode_i_frame(frame),
            FrameType::P => clean_enc.encode_p_frame(frame),
            FrameType::B => clean_enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("clean encode (encode_idx={}, display={}): {e}", eo.encode_idx, eo.display_idx));
        clean_bs.extend_from_slice(&bytes);
    }
    let t_clean = t1.elapsed();
    eprintln!("clean encode: {:?}, {} bytes", t_clean, clean_bs.len());

    // ── 3. ffmpeg-decode both, per-frame Y-PSNR vs source ──────────
    let psnr_each = |bs: &[u8], label: &str| -> Vec<f64> {
        let h264 = std::env::temp_dir().join(format!("phasm_v4_probe_{label}.h264"));
        let dec = std::env::temp_dir().join(format!("phasm_v4_probe_{label}.dec.yuv"));
        std::fs::write(&h264, bs).unwrap();
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(&h264)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&dec)
            .status()
            .expect("ffmpeg run");
        assert!(status.success(), "ffmpeg decode failed for {label}");
        let decoded = std::fs::read(&dec).unwrap();
        let y_size = (width * height) as usize;
        let mut psnrs = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let off = i * frame_size;
            let src_y = &yuv[off..off + y_size];
            let dec_y = &decoded[off..off + y_size];
            let mut sum_sq: u64 = 0;
            for (a, b) in src_y.iter().zip(dec_y.iter()) {
                let d = (*a as i32 - *b as i32) as i64;
                sum_sq += (d * d) as u64;
            }
            let mse = (sum_sq as f64) / (y_size as f64);
            let psnr = if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() };
            psnrs.push(psnr);
        }
        psnrs
    };

    let stego_psnrs = psnr_each(&stego_bs, "stego");
    let clean_psnrs = psnr_each(&clean_bs, "clean");

    eprintln!("\n=== Per-frame Y-PSNR vs source ({}f IBPBP gop={}) ===", n_frames, n_frames);
    eprintln!("{:>4} {:>10} {:>10} {:>8}", "n", "stego", "clean", "Δ");
    for i in 0..n_frames {
        eprintln!(
            "{:>4} {:>10.2} {:>10.2} {:>+8.2}",
            i + 1,
            stego_psnrs[i],
            clean_psnrs[i],
            stego_psnrs[i] - clean_psnrs[i],
        );
    }
    let avg_stego = stego_psnrs.iter().sum::<f64>() / n_frames as f64;
    let avg_clean = clean_psnrs.iter().sum::<f64>() / n_frames as f64;
    let avg_cost = avg_stego - avg_clean;
    eprintln!(
        " avg {:>10.2} {:>10.2} {:>+8.2}",
        avg_stego, avg_clean, avg_cost,
    );

    // Per audit decision rule: ship-as-is threshold is avg ≥ -2 dB.
    // This is the gate Phase 1.1.D validates after Phase 1.1.B+C
    // landed the post-flip visual_recon path. Pre-Phase-1.1.B
    // measurements showed -7.37 dB on V4 = (B_RDO=1, B_RES=1) at
    // 12f gop=12 (memory/h264_stego_b_cascade_2026_05_05.md).
    // §B-direct-fix Stage 2 ROOT-CAUSE FIX 2026-05-06 — also mux
    // 1080p stego + clean to Desktop for visual inspection.
    // #234 (2026-05-16): muxed through phasm's HandbrakeX264 profile
    // for proper ctts box / display-order PTS (replacing the legacy
    // bare `ffmpeg -c:v copy` which couldn't derive PTS from raw .h264).
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};
    let timing = FrameTiming::FPS_30;
    let stego_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &stego_bs, width, height, timing, pattern, n_frames,
    ).expect("stego mp4 mux");
    let clean_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &clean_bs, width, height, timing, pattern, n_frames,
    ).expect("clean mp4 mux");
    let stego_mp4_path = "/Users/cgaffga/Desktop/phasm_v4_cascade_1080p_stego.mp4";
    let clean_mp4_path = "/Users/cgaffga/Desktop/phasm_v4_cascade_1080p_clean.mp4";
    std::fs::write(stego_mp4_path, &stego_mp4).expect("write stego mp4");
    std::fs::write(clean_mp4_path, &clean_mp4).expect("write clean mp4");
    eprintln!("muxed stego mp4 → {}", stego_mp4_path);
    eprintln!("muxed clean mp4 → {}", clean_mp4_path);

    eprintln!(
        "\nPhase 1.1.D verdict: avg cost = {:+.2} dB, target ≥ -2.00 dB, {}",
        avg_cost,
        if avg_cost >= -2.0 { "PASS" } else { "FAIL" },
    );
    assert!(
        avg_cost >= -2.0,
        "Phase 1.1.D gate fail: avg stego cost {:.2} dB exceeds -2 dB threshold",
        avg_cost,
    );
}

/// 2026-05-06 — fast Phase 1.1.D cascade probe at 480×272 × 12f IBPBP.
///
/// Same orchestrator + PRODUCTION_VISUAL config as the 1080p×10f probe
/// above, just at 1/16 the pixel count. Goal: sub-hour cycle on visual
/// confirmation of the §B-cascade-real fix without waiting on the
/// 1080p path that runs 12+ hours single-thread debug.
///
/// Generate fixture once (writes to /tmp; ~2 MB):
///     ffmpeg -y -f rawvideo -pix_fmt yuv420p -s 1920x1072 \
///         -i /tmp/iphone7_1920x1072_f60.yuv -frames:v 12 \
///         -vf scale=480:272 -f rawvideo -pix_fmt yuv420p \
///         /tmp/iphone7_480x272_f12.yuv
///
/// At end of run, also writes the decoded stego YUV back as an mp4 to
/// /Users/cgaffga/Desktop/phasm_v4_cascade_480p.mp4 for visual review.
#[test]
#[ignore]
fn phase_1_1_d_v4_cascade_probe_480p_12f() {
    use phasm_core::codec::h264::stego::encode_pixels::
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files;

    let yuv_path = "/tmp/iphone7_480x272_f12.yuv";
    let yuv = std::fs::read(yuv_path)
        .expect("missing /tmp/iphone7_480x272_f12.yuv (see test docstring for ffmpeg recipe)");
    let width = 480u32;
    let height = 272u32;
    let n_frames = 12usize;
    let frame_size = (width * height * 3 / 2) as usize;
    assert_eq!(
        yuv.len(),
        frame_size * n_frames,
        "fixture size mismatch: expected {} bytes for 480x272x12 YUV420p, got {}",
        frame_size * n_frames,
        yuv.len()
    );

    let pattern = GopPattern::Ibpbp { gop: 12, b_count: 1 };

    // §B-direct-fix Stage 2 (#232) follow-up: lock seed so probe is
    // reproducible. Without this, crypto::encrypt's per-call salt+nonce
    // gives ~5-10 dB run-to-run variance on stego cost, which
    // confounded the 2026-05-06 fix-vs-baseline comparison.
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
    }

    eprintln!("\n=== Phase 1.1.D V4 cascade probe — 480×272 × 12f IBPBP ===");
    eprintln!("PHASM_B_RDO={}  PHASM_B_RESIDUAL={}",
        std::env::var("PHASM_B_RDO").unwrap_or_else(|_| "(unset)".into()),
        std::env::var("PHASM_B_RESIDUAL").unwrap_or_else(|_| "(unset)".into()),
    );

    let t0 = std::time::Instant::now();
    let stego_bs =
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
            &yuv,
            width,
            height,
            n_frames,
            pattern,
            "phase-1.1.d-probe-480p",
            &[],
            "passprobe-1.1.d",
        )
        .expect("stego encode");
    let t_stego = t0.elapsed();
    eprintln!("stego encode: {:?}, {} bytes", t_stego, stego_bs.len());

    let mut clean_enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    clean_enc.entropy_mode = EntropyMode::Cabac;
    clean_enc.enable_transform_8x8 = true;
    clean_enc.enable_b_frames = true;
    clean_enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;
    let mut clean_bs = Vec::new();
    let t1 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => clean_enc.encode_i_frame(frame),
            FrameType::P => clean_enc.encode_p_frame(frame),
            FrameType::B => clean_enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("clean encode (encode_idx={}, display={}): {e}", eo.encode_idx, eo.display_idx));
        clean_bs.extend_from_slice(&bytes);
    }
    let t_clean = t1.elapsed();
    eprintln!("clean encode: {:?}, {} bytes", t_clean, clean_bs.len());

    let psnr_each = |bs: &[u8], label: &str| -> Vec<f64> {
        let h264 = std::env::temp_dir().join(format!("phasm_v4_probe_480p_{label}.h264"));
        let dec = std::env::temp_dir().join(format!("phasm_v4_probe_480p_{label}.dec.yuv"));
        std::fs::write(&h264, bs).unwrap();
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(&h264)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&dec)
            .status()
            .expect("ffmpeg run");
        assert!(status.success(), "ffmpeg decode failed for {label}");
        let decoded = std::fs::read(&dec).unwrap();
        let y_size = (width * height) as usize;
        let mut psnrs = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let off = i * frame_size;
            let src_y = &yuv[off..off + y_size];
            let dec_y = &decoded[off..off + y_size];
            let mut sum_sq: u64 = 0;
            for (a, b) in src_y.iter().zip(dec_y.iter()) {
                let d = (*a as i32 - *b as i32) as i64;
                sum_sq += (d * d) as u64;
            }
            let mse = (sum_sq as f64) / (y_size as f64);
            let psnr = if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() };
            psnrs.push(psnr);
        }
        psnrs
    };

    let stego_psnrs = psnr_each(&stego_bs, "stego");
    let clean_psnrs = psnr_each(&clean_bs, "clean");

    eprintln!("\n=== Per-frame Y-PSNR vs source (12f IBPBP gop=12, 480×272) ===");
    eprintln!("{:>4} {:>10} {:>10} {:>8}", "n", "stego", "clean", "Δ");
    for i in 0..n_frames {
        eprintln!(
            "{:>4} {:>10.2} {:>10.2} {:>+8.2}",
            i + 1,
            stego_psnrs[i],
            clean_psnrs[i],
            stego_psnrs[i] - clean_psnrs[i],
        );
    }
    let avg_stego = stego_psnrs.iter().sum::<f64>() / n_frames as f64;
    let avg_clean = clean_psnrs.iter().sum::<f64>() / n_frames as f64;
    let avg_cost = avg_stego - avg_clean;
    eprintln!(
        " avg {:>10.2} {:>10.2} {:>+8.2}",
        avg_stego, avg_clean, avg_cost,
    );

    // #234 (2026-05-16): mux through phasm's HandbrakeX264 profile so the
    // Desktop demo gets proper ctts box / display-order PTS — the legacy
    // bare `ffmpeg -c:v copy` couldn't derive PTS from raw .h264 and
    // produced first-frame artifacts in players.
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};
    let timing = FrameTiming::FPS_30;
    let stego_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &stego_bs, width, height, timing, pattern, n_frames,
    ).expect("stego mp4 mux");
    let clean_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &clean_bs, width, height, timing, pattern, n_frames,
    ).expect("clean mp4 mux");
    let stego_mp4_path = "/Users/cgaffga/Desktop/phasm_v4_cascade_480p_stego.mp4";
    let clean_mp4_path = "/Users/cgaffga/Desktop/phasm_v4_cascade_480p_clean.mp4";
    std::fs::write(stego_mp4_path, &stego_mp4).expect("write stego mp4");
    std::fs::write(clean_mp4_path, &clean_mp4).expect("write clean mp4");
    eprintln!("muxed stego mp4 → {}", stego_mp4_path);
    eprintln!("muxed clean mp4 → {}", clean_mp4_path);

    eprintln!(
        "\nPhase 1.1.D 480p verdict: avg cost = {:+.2} dB, target ≥ -2.00 dB, {}",
        avg_cost,
        if avg_cost >= -2.0 { "PASS" } else { "FAIL" },
    );
    assert!(
        avg_cost >= -2.0,
        "Phase 1.1.D 480p gate fail: avg stego cost {:.2} dB exceeds -2 dB threshold",
        avg_cost,
    );
}

/// **§B-cascade-real fix #233 follow-up** (2026-05-06). Fast clean-only
/// 480p × 12f IBPBP encode + per-frame ffmpeg PSNR. No stego, no
/// orchestrator. Runs in seconds (debug build), unlike
/// `phase_1_1_d_v4_cascade_probe_480p_12f` which takes 25+ min via
/// the streaming-Viterbi path. Purpose: confirm that the per-frame
/// PSNR for the CLEAN encoder is high (matches ffmpeg decode of source).
///
/// Also writes a Desktop mp4 demo via HandbrakeX264 (#234) so visual
/// inspection is artifact-free — the historical first-frame glitches
/// in `phasm_v4_cascade_480p_clean.mp4` came from bare `ffmpeg -c:v copy`
/// over raw .h264 (no PTS / no ctts box), which the V4 probe and this
/// test now both bypass.
///
/// Generate fixture once (writes to /tmp; ~2 MB):
///     ffmpeg -y -f rawvideo -pix_fmt yuv420p -s 1920x1072 \
///         -i /tmp/iphone7_1920x1072_f60.yuv -frames:v 12 \
///         -vf scale=480:272 -f rawvideo -pix_fmt yuv420p \
///         /tmp/iphone7_480x272_f12.yuv
#[test]
#[ignore]
fn phase_1_1_d_clean_480p_12f_psnr() {
    let yuv_path = "/tmp/iphone7_480x272_f12.yuv";
    let yuv = std::fs::read(yuv_path)
        .expect("missing /tmp/iphone7_480x272_f12.yuv (see test docstring for ffmpeg recipe)");
    let width = 480u32;
    let height = 272u32;
    let n_frames = 12usize;
    let frame_size = (width * height * 3 / 2) as usize;
    assert_eq!(yuv.len(), frame_size * n_frames);

    let pattern = GopPattern::Ibpbp { gop: 12, b_count: 1 };

    let mut clean_enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    clean_enc.entropy_mode = EntropyMode::Cabac;
    clean_enc.enable_transform_8x8 = true;
    clean_enc.enable_b_frames = true;
    clean_enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;
    let mut clean_bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => clean_enc.encode_i_frame(frame),
            FrameType::P => clean_enc.encode_p_frame(frame),
            FrameType::B => clean_enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("clean encode (encode_idx={}, display={}): {e}", eo.encode_idx, eo.display_idx));
        clean_bs.extend_from_slice(&bytes);
    }
    eprintln!("clean encode: {:?}, {} bytes", t0.elapsed(), clean_bs.len());

    let h264 = std::env::temp_dir().join("phasm_clean_480p_12f.h264");
    let dec = std::env::temp_dir().join("phasm_clean_480p_12f.dec.yuv");
    std::fs::write(&h264, &clean_bs).unwrap();
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec)
        .status()
        .expect("ffmpeg run");
    assert!(status.success(), "ffmpeg decode failed");
    let decoded = std::fs::read(&dec).unwrap();
    let y_size = (width * height) as usize;

    eprintln!(
        "\n=== Per-frame Y-PSNR vs source — CLEAN 480×272 × 12f IBPBP gop=12 ==="
    );
    eprintln!("{:>4} {:>10}", "n", "psnr");
    let mut sum_psnr = 0.0;
    for i in 0..n_frames {
        let off = i * frame_size;
        let src_y = &yuv[off..off + y_size];
        let dec_y = &decoded[off..off + y_size];
        let mut sum_sq: u64 = 0;
        for (a, b) in src_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32) as i64;
            sum_sq += (d * d) as u64;
        }
        let mse = (sum_sq as f64) / (y_size as f64);
        let psnr = if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() };
        sum_psnr += psnr;
        eprintln!("{:>4} {:>10.2}", i + 1, psnr);
    }
    let avg_psnr = sum_psnr / n_frames as f64;
    eprintln!(" avg {:>10.2}", avg_psnr);

    // Sanity gate: clean encode at QP=26 should keep avg Y-PSNR >= 32 dB
    // on real video content. If this fails, encoder has a real bug.
    assert!(
        avg_psnr >= 32.0,
        "clean encoder avg Y-PSNR {:.2} dB < 32 dB threshold — encoder regression",
        avg_psnr,
    );

    // #234 (2026-05-16): mux Desktop demo through HandbrakeX264 so the
    // resulting mp4 has proper ctts / display-order PTS. Pre-#234 this
    // test deliberately skipped the mp4 write because bare `ffmpeg -c:v
    // copy` over raw .h264 produced first-frame artifacts; the phasm
    // muxer emits the right ctts box from the encoder's frame pattern.
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};
    let timing = FrameTiming::FPS_30;
    let clean_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &clean_bs, width, height, timing, pattern, n_frames,
    ).expect("clean mp4 mux");
    let clean_mp4_path = "/Users/cgaffga/Desktop/phasm_v4_cascade_480p_clean.mp4";
    std::fs::write(clean_mp4_path, &clean_mp4).expect("write clean mp4");
    eprintln!("muxed clean mp4 → {}", clean_mp4_path);
}

/// **§B-cascade-real bisect (2026-05-06)** — IDR-only single-frame stego cost.
///
/// V4 480p × 12f probe shows per-frame stego cost of -12.72 dB (target -2 dB).
/// Frame 1 (IDR) alone has -11.42 dB cost despite having no past frames.
/// This isolates the IDR-direct stego damage to confirm whether it's
/// per-frame damage (high) or whether the IDR damage shown in V4 is
/// somehow the result of B-frame interaction. Single IDR + IPPPP pattern.
///
/// Runs in seconds (debug build) — no orchestrator multi-pass overhead.
#[test]
#[ignore]
fn phase_1_1_d_idr_only_stego_480p() {
    use phasm_core::codec::h264::stego::encode_pixels::
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files;

    let yuv_path = "/tmp/iphone7_480x272_f12.yuv";
    let yuv = std::fs::read(yuv_path)
        .expect("missing /tmp/iphone7_480x272_f12.yuv");
    let width = 480u32;
    let height = 272u32;
    let n_frames = 1usize; // IDR only
    let frame_size = (width * height * 3 / 2) as usize;

    // IPPPP pattern with gop=1 → just one IDR.
    let pattern = GopPattern::Ipppp { gop: 1 };

    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
    }

    eprintln!("\n=== IDR-only stego cost — 480x272 single frame ===");
    let t0 = std::time::Instant::now();
    let stego_bs =
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
            &yuv[..frame_size],
            width, height, n_frames, pattern,
            "idr-only-bisect-480p",
            &[],
            "passprobe-idr-only",
        )
        .expect("stego encode");
    let t_stego = t0.elapsed();
    eprintln!("stego encode (msg='idr-only-bisect-480p'): {:?}, {} bytes", t_stego, stego_bs.len());

    // Empty-message bisect: if stego cost is still nonzero with empty
    // message, the cost is from the ENCODER PIPELINE (cover-state
    // capture, RNG-driven mode selection, etc.), not from STC flips.
    let stego_empty_bs =
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
            &yuv[..frame_size],
            width, height, n_frames, pattern,
            "",  // empty message
            &[],
            "passprobe-idr-only",
        )
        .expect("stego encode (empty)");
    eprintln!("stego encode (msg=''): {} bytes", stego_empty_bs.len());

    let mut clean_enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    clean_enc.entropy_mode = EntropyMode::Cabac;
    clean_enc.enable_transform_8x8 = true;
    let frame = &yuv[..frame_size];
    let clean_bs = clean_enc.encode_i_frame(frame).expect("clean IDR encode");
    eprintln!("clean encode: {} bytes", clean_bs.len());

    let psnr_each = |bs: &[u8], label: &str| -> f64 {
        let h264 = std::env::temp_dir().join(format!("phasm_idr_only_{label}.h264"));
        let dec = std::env::temp_dir().join(format!("phasm_idr_only_{label}.dec.yuv"));
        std::fs::write(&h264, bs).unwrap();
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(&h264)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&dec)
            .status()
            .expect("ffmpeg run");
        assert!(status.success(), "ffmpeg decode failed for {label}");
        let decoded = std::fs::read(&dec).unwrap();
        let y_size = (width * height) as usize;
        let src_y = &yuv[..y_size];
        let dec_y = &decoded[..y_size];
        let mut sum_sq: u64 = 0;
        for (a, b) in src_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32) as i64;
            sum_sq += (d * d) as u64;
        }
        let mse = (sum_sq as f64) / (y_size as f64);
        if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
    };

    let stego_psnr = psnr_each(&stego_bs, "stego");
    let stego_empty_psnr = psnr_each(&stego_empty_bs, "stego_empty");
    let clean_psnr = psnr_each(&clean_bs, "clean");

    eprintln!("\n=== IDR-only Y-PSNR ===");
    eprintln!("stego (msg='idr-only-bisect-480p'): {:>6.2} dB  Δ={:+6.2} dB",
        stego_psnr, stego_psnr - clean_psnr);
    eprintln!("stego (msg=''):                     {:>6.2} dB  Δ={:+6.2} dB",
        stego_empty_psnr, stego_empty_psnr - clean_psnr);
    eprintln!("clean:                              {:>6.2} dB", clean_psnr);

    // No assert — diagnostic test for §B-cascade-real bisect.
}

/// **§B-cascade-real Bug #1a fix verification (#235)** — V6 production probe
/// with per-GOP STC orchestrator (2026-05-07).
///
/// Same fixture as V5 (1080p × 45f IBPBP gop=30) but uses the new
/// `h264_stego_encode_yuv_string_4domain_per_gop_v3` entry point that
/// runs per-GOP STC instead of the streaming-Viterbi clip-wide trellis.
///
/// Verifies:
/// 1. Encode + decode round-trip works (message recovers exactly).
/// 2. GOP-boundary catastrophic damage closes (V5 had GOP 1 -14 dB Y;
///    expected here: GOP 1 should land in same envelope as GOP 0).
/// 3. mp4 mux via HandbrakeX264 is clean (proper ctts).
#[test]
#[ignore]
fn phase_1_1_d_v6_per_gop_1080p_45f() {
    use phasm_core::codec::h264::stego::encode_pixels::
        h264_stego_encode_yuv_string_4domain_per_gop_v3;
    use phasm_core::codec::h264::stego::decode_pixels::
        h264_stego_decode_yuv_string_4domain_per_gop_v3;
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = yuv.len() / frame_size;
    eprintln!("fixture: 1920x1072 × {} frames", n_frames);

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42"); }

    let msg = "Hello from phasm — this is a real production-style stego payload, ~70 chars.";

    eprintln!("\n=== V6 per-GOP STC orchestrator (Bug #1a fix) ===");
    let t0 = std::time::Instant::now();
    let stego_bs = h264_stego_encode_yuv_string_4domain_per_gop_v3(
        &yuv, width, height, n_frames, pattern, msg, &[], "v6-per-gop-pass",
    ).expect("v3 encode");
    eprintln!("v3 encode: {:?}, {} bytes", t0.elapsed(), stego_bs.len());

    // Round-trip: decode the stego bytes.
    let recovered = h264_stego_decode_yuv_string_4domain_per_gop_v3(
        &stego_bs, "v6-per-gop-pass",
    ).expect("v3 decode");
    eprintln!("decoded: {:?}", recovered);
    assert_eq!(recovered, msg, "round-trip mismatch: bug #1a fix breaks decode");

    // Clean baseline.
    let mut clean_enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    clean_enc.entropy_mode = EntropyMode::Cabac;
    clean_enc.enable_transform_8x8 = true;
    clean_enc.enable_b_frames = true;
    clean_enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;
    let mut clean_bs = Vec::new();
    let t1 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => clean_enc.encode_i_frame(frame),
            FrameType::P => clean_enc.encode_p_frame(frame),
            FrameType::B => clean_enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("clean encode error: {e}"));
        clean_bs.extend_from_slice(&bytes);
    }
    eprintln!("clean encode: {:?}, {} bytes", t1.elapsed(), clean_bs.len());

    // Mux via HandbrakeX264.
    let timing = FrameTiming::FPS_30;
    let stego_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &stego_bs, width, height, timing, pattern, n_frames,
    ).expect("stego mp4 mux");
    let clean_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &clean_bs, width, height, timing, pattern, n_frames,
    ).expect("clean mp4 mux");
    let stego_mp4_path = "/Users/cgaffga/Desktop/phasm_v6_per_gop_1080p_stego.mp4";
    let clean_mp4_path = "/Users/cgaffga/Desktop/phasm_v6_per_gop_1080p_clean.mp4";
    std::fs::write(stego_mp4_path, &stego_mp4).expect("write stego mp4");
    std::fs::write(clean_mp4_path, &clean_mp4).expect("write clean mp4");
    eprintln!("muxed stego mp4 -> {}", stego_mp4_path);
    eprintln!("muxed clean mp4 -> {}", clean_mp4_path);

    // Per-frame Y/U/V PSNR.
    let psnr_each = |bs: &[u8], label: &str| -> Vec<(f64, f64, f64)> {
        let h264 = std::env::temp_dir().join(format!("phasm_v6_{label}.h264"));
        let dec = std::env::temp_dir().join(format!("phasm_v6_{label}.dec.yuv"));
        std::fs::write(&h264, bs).unwrap();
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
            .status().expect("ffmpeg run");
        assert!(status.success());
        let decoded = std::fs::read(&dec).unwrap();
        let y_size = (width * height) as usize;
        let c_size = (width / 2 * height / 2) as usize;
        let mut psnrs = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let off = i * frame_size;
            let calc = |src: &[u8], dec: &[u8]| -> f64 {
                let mut sum_sq: u64 = 0;
                for (a, b) in src.iter().zip(dec.iter()) {
                    let d = (*a as i32 - *b as i32) as i64;
                    sum_sq += (d * d) as u64;
                }
                let mse = (sum_sq as f64) / (src.len() as f64);
                if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
            };
            let py = calc(&yuv[off..off+y_size], &decoded[off..off+y_size]);
            let pu = calc(&yuv[off+y_size..off+y_size+c_size],
                          &decoded[off+y_size..off+y_size+c_size]);
            let pv = calc(&yuv[off+y_size+c_size..off+frame_size],
                          &decoded[off+y_size+c_size..off+frame_size]);
            psnrs.push((py, pu, pv));
        }
        psnrs
    };

    let stego_psnrs = psnr_each(&stego_bs, "stego");
    let clean_psnrs = psnr_each(&clean_bs, "clean");

    eprintln!("\n=== V6 Per-frame Y/U/V PSNR ===");
    eprintln!("{:>4} {:>10} {:>10} {:>10}  {:>10} {:>10} {:>10}  {:>9} {:>9} {:>9}",
        "n", "Y stego", "Y clean", "Y Δ", "U stego", "U clean", "U Δ",
        "V stego", "V clean", "V Δ");
    let (mut sum_dy, mut sum_du, mut sum_dv) = (0.0f64, 0.0f64, 0.0f64);
    let (mut sum_sy, mut sum_su, mut sum_sv) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n_frames {
        let (sy, su, sv) = stego_psnrs[i];
        let (cy, cu, cv) = clean_psnrs[i];
        eprintln!("{:>4} {:>10.2} {:>10.2} {:>+10.2}  {:>10.2} {:>10.2} {:>+10.2}  {:>9.2} {:>9.2} {:>+9.2}",
            i+1, sy, cy, sy-cy, su, cu, su-cu, sv, cv, sv-cv);
        sum_dy += sy - cy; sum_du += su - cu; sum_dv += sv - cv;
        sum_sy += sy; sum_su += su; sum_sv += sv;
    }
    let n = n_frames as f64;
    eprintln!("\navg  Y: stego={:.2}  Δ={:+.2} dB", sum_sy/n, sum_dy/n);
    eprintln!("avg  U: stego={:.2}  Δ={:+.2} dB", sum_su/n, sum_du/n);
    eprintln!("avg  V: stego={:.2}  Δ={:+.2} dB", sum_sv/n, sum_dv/n);

    // V5 had GOP 1 frames at -10 to -14 dB Y. Compute GOP 1 max-damage.
    let mut max_y_gop1: f64 = 0.0;
    for i in 30..n_frames {
        let dy = stego_psnrs[i].0 - clean_psnrs[i].0;
        if dy < max_y_gop1 || max_y_gop1 == 0.0 { max_y_gop1 = dy; }
    }
    eprintln!("\nGOP 1 worst-frame Y stego cost: {:+.2} dB (V5 baseline: -14.19 dB)", max_y_gop1);
}

/// **§B-direct-fix.v3 V7 — spatial-direct vs temporal-direct A/B** (2026-05-07).
///
/// User screenshot 2026-05-07 shows visible motion-boundary artifacts on the
/// V6 (spatial-direct default) demo. This test produces TWO clean (no stego)
/// 1080p × 45f IBPBP mp4s on Desktop:
/// - `phasm_v7_spatial_clean.mp4` — current default (PHASM_B_TEMPORAL_DIRECT unset)
/// - `phasm_v7_temporal_override_clean.mp4` — temporal-direct + new near-zero
///   colMb override (PHASM_B_TEMPORAL_DIRECT=1, set inside the test)
///
/// Both encoded from /tmp/iphone7_1920x1072_f60.yuv with identical encoder
/// config. The only difference is the direct-mode dispatch in encode_b_frame.
///
/// Visual verdict: open both in QuickTime side-by-side. The artifacts on
/// pants/wall boundary should be GONE (or substantially reduced) on the
/// temporal-override variant. PSNR is logged per-frame for both modes for
/// quantitative confirmation.
#[test]
#[ignore]
fn phase_1_1_d_v7_direct_mode_ab_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = yuv.len() / frame_size;
    let n_frames = n_frames.min(45);
    eprintln!("fixture: 1920x1072 × {} frames", n_frames);

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    // Encode helper. `temporal` = whether to set PHASM_B_TEMPORAL_DIRECT
    // for this run. Returns the bitstream.
    let encode_clean = |temporal: bool| -> Vec<u8> {
        unsafe {
            if temporal {
                std::env::set_var("PHASM_B_TEMPORAL_DIRECT", "1");
            } else {
                std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
            }
        }
        let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = true;
        enc.enable_b_frames = true;
        enc.b_rdo_config =
            phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;
        let mut bs = Vec::new();
        for eo in iter_encode_order(n_frames, pattern) {
            let d = eo.display_idx as usize;
            let frame = &yuv[d * frame_size..(d + 1) * frame_size];
            let bytes = match eo.frame_type {
                FrameType::Idr => enc.encode_i_frame(frame),
                FrameType::P => enc.encode_p_frame(frame),
                FrameType::B => enc.encode_b_frame(frame),
            }
            .unwrap_or_else(|e| panic!("clean encode error: {e}"));
            bs.extend_from_slice(&bytes);
        }
        bs
    };

    eprintln!("\n=== V7-A: spatial-direct (legacy default) ===");
    let t0 = std::time::Instant::now();
    let spatial_bs = encode_clean(false);
    eprintln!("encode: {:?}, {} bytes", t0.elapsed(), spatial_bs.len());

    eprintln!("\n=== V7-B: temporal-direct + near-zero colMb override (new) ===");
    let t1 = std::time::Instant::now();
    let temporal_bs = encode_clean(true);
    eprintln!("encode: {:?}, {} bytes", t1.elapsed(), temporal_bs.len());

    // Mux both via HandbrakeX264.
    let timing = FrameTiming::FPS_30;
    let spatial_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &spatial_bs, width, height, timing, pattern, n_frames,
    ).expect("spatial mp4 mux");
    let temporal_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &temporal_bs, width, height, timing, pattern, n_frames,
    ).expect("temporal mp4 mux");

    let spatial_path = "/Users/cgaffga/Desktop/phasm_v7_spatial_clean.mp4";
    let temporal_path = "/Users/cgaffga/Desktop/phasm_v7_temporal_override_clean.mp4";
    std::fs::write(spatial_path, &spatial_mp4).expect("write spatial mp4");
    std::fs::write(temporal_path, &temporal_mp4).expect("write temporal mp4");
    eprintln!("\nA spatial   -> {}", spatial_path);
    eprintln!("B temporal  -> {}", temporal_path);

    // Per-frame Y PSNR for both.
    let psnr_each = |bs: &[u8], label: &str| -> Vec<f64> {
        let h264 = std::env::temp_dir().join(format!("phasm_v7_{label}.h264"));
        let dec = std::env::temp_dir().join(format!("phasm_v7_{label}.dec.yuv"));
        std::fs::write(&h264, bs).unwrap();
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
            .status().expect("ffmpeg run");
        assert!(status.success(), "ffmpeg decode failed for {label}");
        let decoded = std::fs::read(&dec).unwrap();
        let y_size = (width * height) as usize;
        let mut psnrs = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let off = i * frame_size;
            let mut sum_sq: u64 = 0;
            for (a, b) in yuv[off..off+y_size].iter().zip(decoded[off..off+y_size].iter()) {
                let d = (*a as i32 - *b as i32) as i64;
                sum_sq += (d * d) as u64;
            }
            let mse = sum_sq as f64 / y_size as f64;
            let psnr = if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() };
            psnrs.push(psnr);
        }
        psnrs
    };

    let s_psnrs = psnr_each(&spatial_bs, "spatial");
    let t_psnrs = psnr_each(&temporal_bs, "temporal");

    eprintln!("\n=== V7 Per-frame Y PSNR comparison ===");
    eprintln!("{:>4} {:>10} {:>10} {:>+10}", "n", "spatial", "temporal", "Δ");
    let (mut sum_s, mut sum_t) = (0.0f64, 0.0f64);
    for i in 0..n_frames {
        eprintln!("{:>4} {:>10.2} {:>10.2} {:>+10.2}",
            i+1, s_psnrs[i], t_psnrs[i], t_psnrs[i] - s_psnrs[i]);
        sum_s += s_psnrs[i];
        sum_t += t_psnrs[i];
    }
    let n = n_frames as f64;
    eprintln!("\navg Y: spatial={:.2} dB  temporal={:.2} dB  Δ={:+.2} dB",
        sum_s/n, sum_t/n, (sum_t - sum_s)/n);

    // Cleanup env var so we leave no test side-effects.
    unsafe { std::env::remove_var("PHASM_B_TEMPORAL_DIRECT"); }
}

/// **§B-direct-fix.v3 V8 — Path 4.1 boundary-penalty A/B** (2026-05-07).
///
/// User visual verdict on V7 (spatial vs temporal+override): both modes
/// have visible motion-boundary artifacts, just on opposite sides
/// (spatial inside pants, temporal outside on wall). PSNR delta was
/// misleading; the visual artifact is unchanged in severity.
///
/// Path 4.1: surgical Direct rate-cost penalty when neighbour MVs are
/// bimodal (sharp motion boundary). Pushes RDO to L0/L1/Bi for those
/// MBs, where explicit ME produces content-correct prediction. Direct
/// stays for homogeneous-motion MBs where it's reliable.
///
/// This test produces TWO mp4s on Desktop with versioned filenames
/// (no overwrites — versioned per fix-tag):
/// - `phasm_v8_baseline_clean.mp4` — current default (spatial-direct,
///   PHASM_B_BOUNDARY_PENALTY unset). Matches V7-spatial baseline.
/// - `phasm_v8_path4_clean.mp4` — Path 4.1 active (spatial-direct +
///   bimodal-MV penalty on Direct cost).
///
/// Visual gate: open both in QuickTime side-by-side. The Path 4.1 mp4
/// should show no flickering blocks at the pants/wall boundary if the
/// fix routes those MBs to L0/L1/Bi. Compare with V7 mp4s for the full
/// 4-way picture.
#[test]
#[ignore]
fn phase_1_1_d_v8_path4_boundary_penalty_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = yuv.len() / frame_size;
    let n_frames = n_frames.min(45);
    eprintln!("fixture: 1920x1072 × {} frames", n_frames);

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    // Encode helper. `with_path4` toggles PHASM_B_BOUNDARY_PENALTY.
    let encode_clean = |with_path4: bool| -> Vec<u8> {
        unsafe {
            // Always use spatial-direct as the base (legacy default;
            // V7 showed temporal-direct alone shifts artifact to outside
            // the pants without closing it).
            std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
            if with_path4 {
                std::env::set_var("PHASM_B_BOUNDARY_PENALTY", "1");
            } else {
                std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
            }
        }
        let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = true;
        enc.enable_b_frames = true;
        enc.b_rdo_config =
            phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;
        let mut bs = Vec::new();
        for eo in iter_encode_order(n_frames, pattern) {
            let d = eo.display_idx as usize;
            let frame = &yuv[d * frame_size..(d + 1) * frame_size];
            let bytes = match eo.frame_type {
                FrameType::Idr => enc.encode_i_frame(frame),
                FrameType::P => enc.encode_p_frame(frame),
                FrameType::B => enc.encode_b_frame(frame),
            }
            .unwrap_or_else(|e| panic!("clean encode error: {e}"));
            bs.extend_from_slice(&bytes);
        }
        bs
    };

    eprintln!("\n=== V8-A: baseline (spatial-direct, no boundary penalty) ===");
    let t0 = std::time::Instant::now();
    let baseline_bs = encode_clean(false);
    eprintln!("encode: {:?}, {} bytes", t0.elapsed(), baseline_bs.len());

    eprintln!("\n=== V8-B: Path 4.1 (spatial-direct + boundary penalty) ===");
    let t1 = std::time::Instant::now();
    let path4_bs = encode_clean(true);
    eprintln!("encode: {:?}, {} bytes", t1.elapsed(), path4_bs.len());

    let timing = FrameTiming::FPS_30;
    let baseline_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &baseline_bs, width, height, timing, pattern, n_frames,
    ).expect("baseline mp4 mux");
    let path4_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &path4_bs, width, height, timing, pattern, n_frames,
    ).expect("path4 mp4 mux");

    let baseline_path = "/Users/cgaffga/Desktop/phasm_v8_baseline_clean.mp4";
    let path4_path = "/Users/cgaffga/Desktop/phasm_v8_path4_clean.mp4";
    std::fs::write(baseline_path, &baseline_mp4).expect("write baseline mp4");
    std::fs::write(path4_path, &path4_mp4).expect("write path4 mp4");
    eprintln!("\nA baseline -> {}", baseline_path);
    eprintln!("B path4    -> {}", path4_path);

    let psnr_each = |bs: &[u8], label: &str| -> Vec<f64> {
        let h264 = std::env::temp_dir().join(format!("phasm_v8_{label}.h264"));
        let dec = std::env::temp_dir().join(format!("phasm_v8_{label}.dec.yuv"));
        std::fs::write(&h264, bs).unwrap();
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
            .status().expect("ffmpeg run");
        assert!(status.success(), "ffmpeg decode failed for {label}");
        let decoded = std::fs::read(&dec).unwrap();
        let y_size = (width * height) as usize;
        let mut psnrs = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let off = i * frame_size;
            let mut sum_sq: u64 = 0;
            for (a, b) in yuv[off..off+y_size].iter().zip(decoded[off..off+y_size].iter()) {
                let d = (*a as i32 - *b as i32) as i64;
                sum_sq += (d * d) as u64;
            }
            let mse = sum_sq as f64 / y_size as f64;
            let psnr = if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() };
            psnrs.push(psnr);
        }
        psnrs
    };

    let b_psnrs = psnr_each(&baseline_bs, "baseline");
    let p4_psnrs = psnr_each(&path4_bs, "path4");

    eprintln!("\n=== V8 Per-frame Y PSNR comparison ===");
    eprintln!("{:>4} {:>10} {:>10} {:>+10}", "n", "baseline", "path4", "Δ");
    let (mut sum_b, mut sum_p4) = (0.0f64, 0.0f64);
    for i in 0..n_frames {
        eprintln!("{:>4} {:>10.2} {:>10.2} {:>+10.2}",
            i+1, b_psnrs[i], p4_psnrs[i], p4_psnrs[i] - b_psnrs[i]);
        sum_b += b_psnrs[i];
        sum_p4 += p4_psnrs[i];
    }
    let n = n_frames as f64;
    eprintln!("\navg Y: baseline={:.2} dB  path4={:.2} dB  Δ={:+.2} dB",
        sum_b/n, sum_p4/n, (sum_p4 - sum_b)/n);

    unsafe {
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
    }
}

/// **§B-direct-fix.v3 INVESTIGATION — pure-static fixture** (2026-05-07).
///
/// User visual feedback shows blue jacket pixels appearing in wall regions
/// where the source is white throughout. White vs blue luma differs by
/// >100, so ME cannot confuse them. The bug must be downstream of ME's
/// SAD/SATD search.
///
/// To prove the bug, encode IDR + 1 B-frame from a SYNTHETIC fixture
/// where source[0] == source[1] exactly (no motion at all). On pure
/// static input:
/// - ME must find MV=(0,0) for every MB (SAD at zero is perfect zero).
/// - Reconstruction MUST equal source.
/// - Any pixel deviation = encoder bug (not content limitation).
///
/// Use a small fixture (480x272) for fast iteration. Fixture: real
/// 1080p frame downscaled to 480p (matches V4 cascade fixture content).
#[test]
#[ignore]
fn phase_1_1_d_invest_static_b_frame_480p() {
    let yuv_path = "/tmp/iphone7_480x272_f12.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_480x272_f12.yuv");
    let width = 480u32;
    let height = 272u32;
    let frame_size = (width * height * 3 / 2) as usize;

    // Take ONE frame from the fixture, replicate it as both IDR and B-source.
    // ME on identical source/reference must find zero MVs everywhere.
    let single_frame = &yuv[..frame_size];
    let mut static_yuv = Vec::new();
    static_yuv.extend_from_slice(single_frame);  // frame 0 (IDR)
    static_yuv.extend_from_slice(single_frame);  // frame 1 (P, same content)
    static_yuv.extend_from_slice(single_frame);  // frame 2 (B between, same content)

    // IBPBP M=2 with gop=2: frame 0 IDR, frame 2 P, frame 1 B
    let pattern = GopPattern::Ibpbp { gop: 2, b_count: 1 };
    let n_frames = 3;

    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    // §INVEST 2026-05-07 — toggle B-RDO config via env to bisect bug locus.
    // PHASM_INVEST_B_RDO=0 = legacy Skip/Direct bucket path. Default
    // PRODUCTION_VISUAL.
    let b_rdo = std::env::var("PHASM_INVEST_B_RDO").ok();
    enc.b_rdo_config = match b_rdo.as_deref() {
        Some("0") => {
            eprintln!("(B_RDO=SAFE — legacy bucket Skip/Direct only)");
            phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::SAFE
        }
        _ => {
            eprintln!("(B_RDO=PRODUCTION_VISUAL — full RDO + residual)");
            phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL
        }
    };

    eprintln!("=== INVEST: 480p × 3f static (frame[0]=frame[1]=frame[2]) ===");
    let mut bs = Vec::new();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &static_yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame),
            FrameType::P => enc.encode_p_frame(frame),
            FrameType::B => enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {} bytes", bs.len());

    // Decode via ffmpeg + per-frame Y PSNR vs source.
    let h264 = std::env::temp_dir().join("phasm_invest_static.h264");
    let dec = std::env::temp_dir().join("phasm_invest_static.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    assert!(status.success(), "ffmpeg decode failed");
    let decoded = std::fs::read(&dec).unwrap();
    let y_size = (width * height) as usize;
    let c_size = (width / 2 * height / 2) as usize;
    eprintln!("\n=== Per-frame Y PSNR (source vs ffmpeg-decoded stego) ===");
    for i in 0..n_frames {
        let off = i * frame_size;
        let mut sum_sq: u64 = 0;
        let mut max_diff: u8 = 0;
        let mut count_nonzero: u32 = 0;
        for (a, b) in static_yuv[off..off+y_size].iter().zip(decoded[off..off+y_size].iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs() as u8;
            if d > 0 { count_nonzero += 1; }
            if d > max_diff { max_diff = d; }
            sum_sq += (d as u64) * (d as u64);
        }
        let mse = sum_sq as f64 / y_size as f64;
        let psnr = if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() };
        let frame_label = match i { 0 => "IDR", 1 => "B  ", 2 => "P  ", _ => "?" };
        eprintln!("frame {i} ({frame_label}): Y PSNR={psnr:.2} dB, max_pixel_diff={max_diff}, nonzero_pixels={count_nonzero}/{y_size} ({:.2}%)",
            (count_nonzero as f64 / y_size as f64) * 100.0);

        // §INVEST 2026-05-07 — wall-region check: top-left 32×32 pixels
        // are pure wall in iPhone7 fixture. Sample Y / Cb / Cr and report
        // per-channel max deviation. Wall is luma ~235, chroma ~128.
        let c_off = off + y_size;
        let cr_off = c_off + c_size;
        let mut wall_y_min = 255u8;
        let mut wall_y_max = 0u8;
        let mut wall_cb_min = 255u8;
        let mut wall_cb_max = 0u8;
        let mut wall_cr_min = 255u8;
        let mut wall_cr_max = 0u8;
        // Wall region in iPhone7 fixture: assume top-left 32x32 pure wall
        // (chroma plane is 16x16 there since 4:2:0).
        for y in 0..32 {
            for x in 0..32 {
                let v = decoded[off + y * width as usize + x];
                if v < wall_y_min { wall_y_min = v; }
                if v > wall_y_max { wall_y_max = v; }
            }
        }
        for y in 0..16 {
            for x in 0..16 {
                let cb = decoded[c_off + y * (width / 2) as usize + x];
                let cr = decoded[cr_off + y * (width / 2) as usize + x];
                if cb < wall_cb_min { wall_cb_min = cb; }
                if cb > wall_cb_max { wall_cb_max = cb; }
                if cr < wall_cr_min { wall_cr_min = cr; }
                if cr > wall_cr_max { wall_cr_max = cr; }
            }
        }
        eprintln!("  wall TL 32x32: Y[{wall_y_min},{wall_y_max}], Cb[{wall_cb_min},{wall_cb_max}], Cr[{wall_cr_min},{wall_cr_max}]");
    }

    // Also write decoded mp4 for visual inspection.
    let mp4 = phasm_core::codec::mp4::build::build_mp4_with_pattern(
        phasm_core::codec::mp4::build::MuxerProfile::HandbrakeX264,
        &bs, width, height, phasm_core::codec::mp4::build::FrameTiming::FPS_30, pattern, n_frames,
    ).expect("mp4 mux");
    let mp4_path = "/Users/cgaffga/Desktop/phasm_invest_static_480p.mp4";
    std::fs::write(mp4_path, &mp4).expect("write mp4");
    eprintln!("\nmp4 -> {}", mp4_path);
}

/// **§B-direct-fix.v3 V9 WALL-AREA AUDIT** (2026-05-07).
///
/// Decode user's V9 mp4 and the source YUV. For specific wall MBs the
/// user observed blue blocks in, compare per-pixel: source vs phasm-V9-
/// decoded vs source-encoded-by-x264 (sanity reference).
///
/// Reads pre-decoded YUV from /tmp/v9_decoded.yuv (must be regenerated
/// before running: `ffmpeg -y -i ~/Desktop/phasm_v9_path1plus4_clean.mp4
/// -f rawvideo -pix_fmt yuv420p /tmp/v9_decoded.yuv`).
#[test]
#[ignore]
fn phase_1_1_d_invest_v9_wall_pixel_audit() {
    let src = std::fs::read("/tmp/iphone7_1920x1072_f60.yuv").expect("missing source");
    // Pass dec path via env var so we can audit V7/V8/V9 in turn.
    let dec_path = std::env::var("PHASM_AUDIT_YUV").unwrap_or("/tmp/v9_decoded.yuv".into());
    eprintln!("auditing: {dec_path}");
    let dec = std::fs::read(&dec_path).unwrap_or_else(|_| panic!("missing {dec_path}"));
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let y_size = (width * height) as usize;
    let n_frames = src.len() / frame_size;
    let n_dec_frames = dec.len() / frame_size;
    eprintln!("source: {n_frames} frames, decoded: {n_dec_frames} frames");

    // Scan all 16x16 MBs. For each MB whose source is uniform pure
    // wall (mean Y > 220, std < 5), check the decoded MB for pixels
    // that deviate by > 30. Those are the "blue blocks in white wall"
    // anomalies the user reported.
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let limit = n_frames.min(n_dec_frames);

    for frame_idx in 0..limit.min(15) {
        let off = frame_idx * frame_size;
        let frame_label = if frame_idx == 0 || frame_idx == 30 {
            "I"
        } else if frame_idx % 2 == 0 {
            "P"
        } else {
            "B"
        };
        let mut anomaly_count = 0;
        let mut wall_mb_count = 0;
        let mut worst: (u8, usize, usize, u8) = (0, 0, 0, 0);  // (max_dev, mb_x, mb_y, src_mean)
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_sum: u32 = 0;
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let px_idx =
                            ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = src[off + px_idx];
                        src_sum += s as u32;
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                    }
                }
                let src_mean = (src_sum / 256) as u8;
                let src_range = src_max - src_min;
                // Flat pure-wall MB: mean > 220, range < 10 (no edge).
                if src_mean < 220 || src_range > 10 { continue; }
                wall_mb_count += 1;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let px_idx =
                            ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = src[off + px_idx];
                        let d = dec[off + px_idx];
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                if max_dev > 30 {
                    anomaly_count += 1;
                    if max_dev > worst.0 {
                        worst = (max_dev, mb_x, mb_y, src_mean);
                    }
                }
            }
        }
        eprintln!("[{frame_label}f{frame_idx}] wall_MBs={wall_mb_count} anomalies={anomaly_count} worst MB({},{}): max_dev={} src_mean={}",
            worst.1, worst.2, worst.0, worst.3);
    }
}

/// **§B-direct-fix.v3 SYNTHETIC INVESTIGATION** (2026-05-07).
///
/// Build a 96x80 synthetic YUV: left half white-wall (Y=235), right
/// half blue-jacket-like (Y=80, Cb=170, Cr=80). Pure flat regions with
/// a single sharp vertical boundary. Encode 3 frames identical (no
/// motion). On static input:
/// - White-wall MBs (luma 235 throughout) MUST reconstruct to ~235.
/// - Any pixel in the wall area showing values < 200 or > 250 = bug.
/// - Blue-jacket MBs (luma 80 throughout) MUST reconstruct to ~80.
///
/// If wall pixels show blue jacket values OR vice versa, we've
/// reproduced the user-reported bug at minimal scale.
#[test]
#[ignore]
fn phase_1_1_d_invest_synthetic_static_2color() {
    let width = 96u32;
    let height = 80u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let y_size = (width * height) as usize;
    let c_size = (width / 2 * height / 2) as usize;

    let mut frame = vec![0u8; frame_size];
    // Y plane: white wall left half (Y=235), blue jacket right half (Y=80).
    for y in 0..height as usize {
        for x in 0..width as usize {
            frame[y * width as usize + x] = if x < (width / 2) as usize { 235 } else { 80 };
        }
    }
    // Cb plane (white=128, blue jacket=170).
    for y in 0..(height / 2) as usize {
        for x in 0..(width / 2) as usize {
            let cb_off = y_size + y * (width / 2) as usize + x;
            frame[cb_off] = if x < (width / 4) as usize { 128 } else { 170 };
        }
    }
    // Cr plane (white=128, blue jacket=80).
    for y in 0..(height / 2) as usize {
        for x in 0..(width / 2) as usize {
            let cr_off = y_size + c_size + y * (width / 2) as usize + x;
            frame[cr_off] = if x < (width / 4) as usize { 128 } else { 80 };
        }
    }

    let mut static_yuv = Vec::new();
    static_yuv.extend_from_slice(&frame);
    static_yuv.extend_from_slice(&frame);
    static_yuv.extend_from_slice(&frame);

    let pattern = GopPattern::Ibpbp { gop: 2, b_count: 1 };
    let n_frames = 3;

    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== SYNTHETIC INVEST: 96x80 white|blue split, 3f static ===");
    let mut bs = Vec::new();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &static_yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }

    let h264 = std::env::temp_dir().join("phasm_invest_synth.h264");
    let dec = std::env::temp_dir().join("phasm_invest_synth.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    assert!(status.success());
    let decoded = std::fs::read(&dec).unwrap();

    eprintln!("\n=== Per-frame check (wall=Y=235 left, blue=Y=80 right) ===");
    for i in 0..n_frames {
        let off = i * frame_size;
        let frame_label = match i { 0 => "IDR", 1 => "B  ", 2 => "P  ", _ => "?" };

        let mut wall_min: u8 = 255;
        let mut wall_max: u8 = 0;
        let mut blue_min: u8 = 255;
        let mut blue_max: u8 = 0;
        // Sample MBs squarely inside each region (avoid the boundary at x=48).
        // White wall: x ∈ [0, 32). Blue jacket: x ∈ [64, 96).
        for y in 0..height as usize {
            for x in 0..32 {
                let v = decoded[off + y * width as usize + x];
                if v < wall_min { wall_min = v; }
                if v > wall_max { wall_max = v; }
            }
            for x in 64..96 {
                let v = decoded[off + y * width as usize + x];
                if v < blue_min { blue_min = v; }
                if v > blue_max { blue_max = v; }
            }
        }
        eprintln!("frame {i} ({frame_label}): wall Y range [{wall_min}, {wall_max}] (expect 235), blue Y range [{blue_min}, {blue_max}] (expect 80)");
        // Bug detection: wall pixel < 200 OR blue pixel > 130 = leak from other region.
        if wall_min < 200 {
            eprintln!("  ⚠ WALL CONTAINS NON-WALL PIXELS (min={wall_min}, expect ≥200) — BUG");
        }
        if blue_max > 130 {
            eprintln!("  ⚠ BLUE CONTAINS NON-BLUE PIXELS (max={blue_max}, expect ≤130) — BUG");
        }
    }

    // Also write decoded mp4 for visual inspection.
    let mp4 = phasm_core::codec::mp4::build::build_mp4_with_pattern(
        phasm_core::codec::mp4::build::MuxerProfile::HandbrakeX264,
        &bs, width, height, phasm_core::codec::mp4::build::FrameTiming::FPS_30, pattern, n_frames,
    ).expect("mp4 mux");
    let mp4_path = "/Users/cgaffga/Desktop/phasm_invest_synthetic_2color.mp4";
    std::fs::write(mp4_path, &mp4).expect("write mp4");
    eprintln!("\nmp4 -> {}", mp4_path);
}

/// **§B-encoder-decoder-divergence per-mode bisect** (2026-05-07).
///
/// dump_b_frame_recon_vs_decode showed B-frame visual_recon diverges
/// from ffmpeg.decode by max|Δ|=61-90 with hotspot MBs at mv=(0, 0)
/// inter mode and Σ|Δ|=15000-35000. The bug is in B-frame inter
/// residual emission/decoding for some specific mb_type.
///
/// This test runs with PHASM_B_FORCE_MODE=<mode> (passed via env)
/// and reports per-frame divergence. By running for each mode in turn
/// (skip, direct, l0_16x16, l1_16x16, bi_16x16, partitioned_4..21,
/// b_8x8_uniform_*), we can localise which mb_type's residual path
/// is broken.
///
/// Output format: one per-frame summary line per B-frame so callers
/// can compare modes side-by-side. Reads PHASM_B_FORCE_MODE; defaults
/// to "skip" if unset.
#[test]
#[ignore]
fn invest_b_force_mode_bisect_1080p() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = 6usize; // 1 IDR + 2 P + 3 B in IBPBP M=2
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    let force_mode = std::env::var("PHASM_B_FORCE_MODE").unwrap_or("skip".into());
    eprintln!("=== INVEST: PHASM_B_FORCE_MODE={force_mode} ===");

    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::set_var("PHASM_B_FORCE_MODE", &force_mode);
        // §B-encoder-decoder-divergence bisect: enable RDO+RESIDUAL so the
        // forced mode actually emits residual coefficients (the suspected
        // broken path). Without these, residual emission is off so all
        // modes appear clean — masking the bug.
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
    }

    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
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

    let h264 = std::env::temp_dir().join(format!("phasm_invest_force_{force_mode}.h264"));
    let dec_path = std::env::temp_dir().join(format!("phasm_invest_force_{force_mode}.dec.yuv"));
    std::fs::write(&h264, &bitstream).unwrap();
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec_path)
        .status().expect("ffmpeg");
    assert!(status.success(), "ffmpeg decode failed");
    let decoded = std::fs::read(&dec_path).unwrap();
    let y_size = (width * height) as usize;

    let mut sorted: Vec<_> = recon_dumps.iter().collect();
    sorted.sort_by_key(|(d, _, _)| *d);
    eprintln!("=== mode={force_mode} divergence summary ===");
    let mb_w = (width / 16) as usize;
    let mut first_b_div_mb: Option<(u32, usize, usize)> = None;
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
        // Capture the first B-frame's first diverging MB for diff dump.
        // Lowered threshold to 4 so even baseline-clean partition modes
        // dump their first divergent MB for comparison.
        if first_b_div_mb.is_none() && matches!(ft, FrameType::B) && max_abs > 4 {
            for mby in 0..(height / 16) as usize {
                for mbx in 0..mb_w {
                    let mut max_in_mb = 0u32;
                    for dy in 0..16 {
                        for dx in 0..16 {
                            let idx = (mby * 16 + dy) * width as usize + mbx * 16 + dx;
                            let d = (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
                            if d > max_in_mb { max_in_mb = d; }
                        }
                    }
                    if max_in_mb > 4 {
                        first_b_div_mb = Some((display_idx, mbx, mby));
                        break;
                    }
                }
                if first_b_div_mb.is_some() { break; }
            }
        }
    }
    if let Some((display_idx, mbx, mby)) = first_b_div_mb {
        eprintln!("\n=== diff dump: mode={force_mode} d={display_idx} mb=({mbx},{mby}) ===");
        let enc_y: &Vec<u8> = &sorted.iter().find(|t| t.0 == display_idx).unwrap().2;
        let off = display_idx as usize * frame_size;
        let dec_y = &decoded[off..off + y_size];
        eprintln!("  enc.visual_recon (16x16):");
        for dy in 0..16 {
            eprint!("    ");
            for dx in 0..16 {
                let idx = (mby * 16 + dy) * width as usize + mbx * 16 + dx;
                eprint!("{:>4}", enc_y[idx]);
            }
            eprintln!();
        }
        eprintln!("  ffmpeg.decode (16x16):");
        for dy in 0..16 {
            eprint!("    ");
            for dx in 0..16 {
                let idx = (mby * 16 + dy) * width as usize + mbx * 16 + dx;
                eprint!("{:>4}", dec_y[idx]);
            }
            eprintln!();
        }
        eprintln!("  diff (enc - dec):");
        for dy in 0..16 {
            eprint!("    ");
            for dx in 0..16 {
                let idx = (mby * 16 + dy) * width as usize + mbx * 16 + dx;
                let d = enc_y[idx] as i32 - dec_y[idx] as i32;
                eprint!("{:>+5}", d);
            }
            eprintln!();
        }
    }
    unsafe { std::env::remove_var("PHASM_B_FORCE_MODE"); }
}

/// **§B-encoder-decoder-divergence V14 — bin-6 ctxIdxInc fix** (2026-05-07).
///
/// V13 left ~1000 body anomalies because the wall/body fixes (Path 1
/// colMb override, Path 4 boundary penalty, anchor clamp, multi-cand
/// ME) were all symptom-treating. The actual root cause of B-frame
/// visual artifacts on partitioned MBs was in
/// `core/src/codec/h264/cabac/encoder.rs` `ctx_idx_inc_mb_type_bin`:
/// B-slice mb_type prefix bin 6 was falling through to default
/// ctxIdxInc=0, but the spec's ctxIdxInc table for B-slice mb_type
/// prefix bin 6 (H.264 spec § 9.3.3.1.1.6 Table 9-39 row for prior
/// decoded bin sequence "1 1 1 1 1 1") specifies 5. Bin 6 is only
/// emitted for mb_types 12..21 (the v∈{8..12} branch), so the bug
/// only fired when RDO picked one of those — explaining why the
/// bisect localized to "10 broken mb_types, all with Bi sub-partition".
///
/// This V14 demo runs the same fixture as V13 with the bin-6 fix
/// applied. Audit should report wall=0, body~0 (only quantization
/// noise) — none of the chain-propagated chaos that V11/V12/V13
/// were chasing.
///
/// Output: `phasm_v14_bin6_ctx_clean.mp4` on Desktop.
#[test]
#[ignore]
fn phase_1_1_d_v14_bin6_ctx_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_INVEST_DUMP");
        std::env::remove_var("PHASM_INVEST_DUMP_ALL");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V14: bin-6 ctxIdxInc fix (5 instead of 0 for B-slice mb_type prefix) ===");
    let _ = "V15 below sets PHASM_B_BOUNDARY_PENALTY=1 to validate Class-2 (mode-decision quality) hypothesis.";
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v14_bin6_ctx_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V14 -> {}", path);

    let h264 = std::env::temp_dir().join("phasm_v14.h264");
    let dec = std::env::temp_dir().join("phasm_v14.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V14 audit (15 frames, all MBs) ===");
    eprintln!("wall_anomalies={wall_anom}  body_anomalies={body_anom}  max_dev_overall={max_dev_overall}");
    eprintln!("V11 wall=2, body=1063  V12 wall=3, body=1048  V13 (anchor clamp): symptom-treating");
    eprintln!("V14 (bin-6 ctxIdxInc fix): wall={wall_anom}, body={body_anom}");
}

/// **V17 — Phase 2.7 list-major MVD emit/parse** (2026-05-08).
///
/// Renders the same fixture as V14 with the Phase 2.7 list-major
/// refactor in place (commit c607815). Forced-mode bisect closed
/// partitioned 14/15/18/19 + b_8x8_uniform_bi from max|Δ|=180-191
/// to 0; b_8x8_mixed from 170 to 8. Real-content default-RDO at
/// this fixture didn't move (body=1005 identical to V14), but the
/// visual demo may still differ subtly because per-MB CABAC ctx
/// state evolves differently when emit order changes for the small
/// percentage of partitioned/B_8x8 MBs RDO does pick.
///
/// Output: `phasm_v17_listmajor_clean.mp4` on Desktop. User-facing
/// comparison vs V14 will show whether the visible artifacts (which
/// persist in V14 audit at body=1005) differ in pattern post-fix.
///
/// Body audit: identical to V14 in numbers; goal is visual parity
/// inspection, NOT a numeric improvement.
#[test]
#[ignore]
fn phase_1_1_d_v17_listmajor_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_INVEST_DUMP");
        std::env::remove_var("PHASM_INVEST_DUMP_ALL");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V17: Phase 2.7 list-major MVD emit/parse (post c607815) ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v17_listmajor_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V17 -> {}", path);

    let h264 = std::env::temp_dir().join("phasm_v17.h264");
    let dec = std::env::temp_dir().join("phasm_v17.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V17 audit (15 frames, all MBs) ===");
    eprintln!("V14 (pre-Phase 2.7): wall=1, body=1005");
    eprintln!("V17 (post c607815):  wall={wall_anom}, body={body_anom}, max_dev={max_dev_overall}");
}

/// **V18 — Phase 4 (#251) multi-cand B-frame ME with temporal candidate**
/// (2026-05-08).
///
/// Adds the L1-reference's collocated MV (scaled by POC distance per
/// spec § 8.4.1.2.3) as a candidate to L0+L1 ME at every B-MB. At
/// motion-boundary MBs, the spatial-only candidate set collapses to
/// wall-direction MVs since neighbour MBs are wall MBs. The temporal
/// candidate points toward where the moving content was in the
/// previous frame — content-correct seed for ME refinement.
///
/// Output: `phasm_v18_temporal_cand_clean.mp4`. User-facing A/B vs
/// V14/V17 should show whether the wall-grey-on-jacket artifact closes.
#[test]
#[ignore]
fn phase_1_1_d_v18_temporal_cand_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_INVEST_DUMP");
        std::env::remove_var("PHASM_INVEST_DUMP_ALL");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V18: Phase 4 — temporal MV candidate in B-frame ME ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v18_temporal_cand_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V18 -> {}", path);

    let h264 = std::env::temp_dir().join("phasm_v18.h264");
    let dec = std::env::temp_dir().join("phasm_v18.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V18 audit (15 frames, all MBs) ===");
    eprintln!("V14 (pre-Phase 2.7):           wall=1, body=1005, max_dev=164");
    eprintln!("V17 (post Phase 2.7):          wall=1, body=1005, max_dev=164");
    eprintln!("V18 (post Phase 4 temporal):   wall={wall_anom}, body={body_anom}, max_dev={max_dev_overall}");
}

/// **V19 — Phase 4 temporal candidate + V15 boundary penalty combined**
/// (2026-05-08).
///
/// V18 alone (temporal-cand) gave ~neutral effect (body=1037 vs V14's
/// 1005). V15 alone (boundary penalty) closed body 1005→817 in
/// earlier session. V19 stacks both — if they target different MB
/// classes, the gains compose. The boundary penalty redirects
/// boundary MBs from Direct→explicit-MV; the temporal candidate
/// gives explicit-MV ME a content-correct seed at those MBs.
///
/// Output: `phasm_v19_temporal_plus_penalty_clean.mp4`. Audit target:
/// body < 800.
#[test]
#[ignore]
fn phase_1_1_d_v19_temporal_plus_penalty_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_INVEST_DUMP");
        std::env::remove_var("PHASM_INVEST_DUMP_ALL");
        std::env::remove_var("PHASM_B_FORCE_MODE");
        std::env::set_var("PHASM_B_BOUNDARY_PENALTY", "1");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V19: Phase 4 temporal + V15 boundary penalty stacked ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v19_temporal_plus_penalty_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V19 -> {}", path);

    let h264 = std::env::temp_dir().join("phasm_v19.h264");
    let dec = std::env::temp_dir().join("phasm_v19.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V19 audit (15 frames, all MBs) ===");
    eprintln!("V14 (baseline):                  wall=1, body=1005, max_dev=164");
    eprintln!("V15 (boundary penalty alone):    wall=3, body=817,  max_dev=157");
    eprintln!("V18 (temporal cand alone):       wall=2, body=1037, max_dev=169");
    eprintln!("V19 (temporal + penalty stacked): wall={wall_anom}, body={body_anom}, max_dev={max_dev_overall}");
    unsafe { std::env::remove_var("PHASM_B_BOUNDARY_PENALTY"); }
}

/// **V20 — Phase 4.3 multi-refine-from-each-candidate ME**
/// (2026-05-08).
///
/// V18 (temporal-cand alone) was net-neutral because the existing
/// `search_block_with_candidates` is "best-seed-then-refine": it
/// picks the candidate with lowest *initial* SAD+λ·rate then refines
/// once. A temporal candidate with marginally lower initial cost but
/// a worse refinement basin displaces ME from a better basin.
///
/// V20 enables the new `search_block_multi_refine` path via
/// `PHASM_B_MULTI_REFINE=1`: it runs integer-pel hex search from
/// EVERY candidate independently, picks the absolute best
/// post-integer-refinement, then sub-pel refines the winner. Cost
/// ~2× ME at N=6 candidates.
///
/// V20 also enables `PHASM_B_TEMPORAL_CAND=1` so the temporal
/// candidate is in the set. This is the experiment to determine
/// whether the visible blocky-grey-on-jacket artifact is ME-quality
/// (closes with V20) or something else (V20 looks like V18/V17 →
/// pivot to investigate mode-decision cost or motion-grid integrity).
///
/// Output: `phasm_v20_multi_refine_clean.mp4`. Audit + visual A/B
/// vs V14/V17/V18/V19.
#[test]
#[ignore]
fn phase_1_1_d_v20_multi_refine_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_INVEST_DUMP");
        std::env::remove_var("PHASM_INVEST_DUMP_ALL");
        std::env::remove_var("PHASM_B_FORCE_MODE");
        std::env::set_var("PHASM_B_MULTI_REFINE", "1");
        std::env::set_var("PHASM_B_TEMPORAL_CAND", "1");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V20: Phase 4.3 — multi-refine-from-each-cand + temporal cand ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v20_multi_refine_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V20 -> {}", path);

    let h264 = std::env::temp_dir().join("phasm_v20.h264");
    let dec = std::env::temp_dir().join("phasm_v20.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V20 audit (15 frames, all MBs) ===");
    eprintln!("V14 (baseline):                         wall=1, body=1005, max_dev=164");
    eprintln!("V15 (boundary penalty alone):           wall=3, body=817,  max_dev=157");
    eprintln!("V18 (temporal cand alone):              wall=2, body=1037, max_dev=169");
    eprintln!("V19 (temporal + penalty):               wall=2, body=856,  max_dev=172");
    eprintln!("V20 (multi-refine + temporal cand):     wall={wall_anom}, body={body_anom}, max_dev={max_dev_overall}");
    unsafe {
        std::env::remove_var("PHASM_B_MULTI_REFINE");
        std::env::remove_var("PHASM_B_TEMPORAL_CAND");
    }
}

/// **V22 — IPPPP diagnostic** (2026-05-08).
///
/// Diagnostic experiment to localize the visible v1.0 BLOCKER
/// artifact. V14-V20 all showed wall-grey-on-jacket blocky
/// artifacts at motion-boundary MBs in IBPBP M=2. Phase 4.3 multi-
/// refine ME didn't close it (V20 body=942 vs V14 1005, only -6%).
///
/// V22 encodes the SAME iPhone7 1080p × 45f source with **IPPPP
/// pattern** (no B-frames at all, gop=30). If V22 looks visually
/// clean → the artifact is exclusive to B-frames and IPPPP-default
/// is a viable v1.0 ship path. If V22 also shows the artifact →
/// the bug is deeper (I or P-frame mode-decision / motion-grid
/// initialization) and B-frame work was barking up the wrong tree.
///
/// Output: `phasm_v22_ipppp_clean.mp4`. User-facing visual A/B
/// vs V14/V17/V18/V20.
#[test]
#[ignore]
fn phase_1_1_d_v22_ipppp_diagnostic_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ipppp { gop: 30 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_CAND");
        std::env::remove_var("PHASM_B_MULTI_REFINE");
        std::env::remove_var("PHASM_INVEST_DUMP");
        std::env::remove_var("PHASM_INVEST_DUMP_ALL");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = false;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V22: IPPPP diagnostic (no B-frames) ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => panic!("IPPPP should not produce B frames"),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v22_ipppp_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V22 -> {}", path);

    let h264 = std::env::temp_dir().join("phasm_v22.h264");
    let dec = std::env::temp_dir().join("phasm_v22.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V22 audit (15 frames, IPPPP, all MBs) ===");
    eprintln!("V14 IBPBP baseline:               wall=1, body=1005, max_dev=164");
    eprintln!("V20 IBPBP multi-refine:           wall=6, body=942,  max_dev=168");
    eprintln!("V22 IPPPP (no B-frames):          wall={wall_anom}, body={body_anom}, max_dev={max_dev_overall}");
}

/// **V21 — force-L0_16x16 in IBPBP diagnostic** (2026-05-08).
///
/// V20 multi-refine ME didn't close the visible v1.0 BLOCKER artifact
/// (-6%). V22 IPPPP is clean — bug is exclusive to B-frames. V21
/// localizes WITHIN the B-frame path: force every B-MB to
/// L0_16x16 (no Direct, no Skip, no Bi, no partitioned, no B_8x8)
/// with default ME finding the L0 MV.
///
/// Outcomes:
/// - If V21 is visually clean (body close to V22's 21):
///     RDO mode-decision is the bug. Direct/Skip selection picks
///     spatial-direct-derived MVs that don't track moving content.
///     v1.0 fix: redesign B-frame mode-decision cost function.
/// - If V21 is dirty (body close to V14's 1005):
///     ME quality OR L0 explicit-mode recon path is broken even
///     with mandatory L0. v1.0 fix: deeper investigation into
///     ME starting points / refinement / L0 recon math.
///
/// Output: `phasm_v21_force_l0_clean.mp4`. Decisive A/B vs V14/V22.
#[test]
#[ignore]
fn phase_1_1_d_v21_force_l0_diagnostic_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_CAND");
        std::env::remove_var("PHASM_B_MULTI_REFINE");
        std::env::remove_var("PHASM_INVEST_DUMP");
        std::env::remove_var("PHASM_INVEST_DUMP_ALL");
        std::env::remove_var("PHASM_B_FORCE_MV");
        std::env::remove_var("PHASM_B_FORCE_MV_L1");
        std::env::set_var("PHASM_B_FORCE_MODE", "l0_16x16");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V21: force L0_16x16 in IBPBP (no Direct/Skip/Bi) ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v21_force_l0_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V21 -> {}", path);

    let h264 = std::env::temp_dir().join("phasm_v21.h264");
    let dec = std::env::temp_dir().join("phasm_v21.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V21 audit (15 frames, IBPBP force-L0, all MBs) ===");
    eprintln!("V14 IBPBP default RDO:           wall=1, body=1005, max_dev=164");
    eprintln!("V20 IBPBP multi-refine ME:       wall=6, body=942,  max_dev=168");
    eprintln!("V22 IPPPP (no B-frames):         wall=0, body=21,   max_dev=47");
    eprintln!("V21 IBPBP force-L0 (no Direct):  wall={wall_anom}, body={body_anom}, max_dev={max_dev_overall}");
    eprintln!();
    eprintln!("If body close to V22's 21 → bug is in RDO mode-decision");
    eprintln!("If body close to V14's 1005 → bug is in ME or L0 recon path");
    unsafe { std::env::remove_var("PHASM_B_FORCE_MODE"); }
}

/// **V23 — distortion-validated Direct selection** (2026-05-08).
///
/// Phase 4.4 fix: in `mb_decision_b_rdo`, refuse Direct/Skip when
/// its raw SATD > 1.5× best-explicit-mode SATD. At motion-boundary
/// MBs where Direct's spatial-derived MV points to wall background,
/// SATD ratio is typically > 2.0 → Direct refused → RDO falls to
/// L0/L1/Bi explicit modes. At uniform-content MBs Direct's MV
/// matches ME's MV → SATDs match → no penalty.
///
/// Default ON (no env var needed). Disable for ablation via
/// `PHASM_B_DIRECT_NO_VALIDATE=1`.
///
/// Output: `phasm_v23_direct_validate_clean.mp4`. Compare visually
/// against V21 (force-L0) and V14/V20 (default RDO with the bug).
#[test]
#[ignore]
fn phase_1_1_d_v23_direct_validate_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_CAND");
        std::env::remove_var("PHASM_B_MULTI_REFINE");
        std::env::remove_var("PHASM_B_DIRECT_NO_VALIDATE");
        std::env::remove_var("PHASM_INVEST_DUMP");
        std::env::remove_var("PHASM_INVEST_DUMP_ALL");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V23: distortion-validated Direct (refuse if SATD > 1.5× best explicit) ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v23_direct_validate_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V23 -> {}", path);

    let h264 = std::env::temp_dir().join("phasm_v23.h264");
    let dec = std::env::temp_dir().join("phasm_v23.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V23 audit (15 frames, all MBs) ===");
    eprintln!("V14 default RDO:                     wall=1, body=1005, max_dev=164");
    eprintln!("V20 multi-refine ME:                 wall=6, body=942,  max_dev=168");
    eprintln!("V21 force-L0 (clean):                wall=6, body=570,  max_dev=51");
    eprintln!("V22 IPPPP (clean ref):               wall=0, body=21,   max_dev=47");
    eprintln!("V23 distortion-validated Direct:     wall={wall_anom}, body={body_anom}, max_dev={max_dev_overall}");
    eprintln!();
    eprintln!("Target: max_dev close to V21/V22 (< 60). Body < 600 ideal.");
    eprintln!("If max_dev > 100: Direct still being picked at boundary MBs;");
    eprintln!("                  threshold k=1.5 too lenient; try k=1.2.");
}

/// **V24 — boundary-anchor override + Direct refusal** (2026-05-08).
///
/// Default-ON fix targeting the actual root cause identified via
/// V21/V22 cleanness vs V14/V20/V23 dirtiness:
///
/// At motion-boundary B-MBs (neighbour L0 MVs span > 1 px),
///   1. Refuse Direct in mb_decision_b_rdo (force RDO to L0/L1/Bi).
///   2. Override the ME rate-cost anchor from spatial-median to
///      MV=(0,0). This breaks the ME bias toward predictor-
///      direction MVs that pulls ME to wall-direction at boundary
///      MBs even when (0,0) has lower SAD.
///
/// Combined effect: at boundary MBs, encoder emits L0_16x16 with
/// MV near (0,0) and a residual — same path V21 demonstrated as
/// clean.
///
/// Output: `phasm_v24_boundary_anchor_clean.mp4`. Audit + visual
/// A/B vs V14/V21/V22.
#[test]
#[ignore]
fn phase_1_1_d_v24_boundary_anchor_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_CAND");
        std::env::remove_var("PHASM_B_MULTI_REFINE");
        std::env::remove_var("PHASM_B_DIRECT_VALIDATE");
        std::env::remove_var("PHASM_B_NO_BOUNDARY_REFUSE");
        std::env::remove_var("PHASM_INVEST_DUMP");
        std::env::remove_var("PHASM_INVEST_DUMP_ALL");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V24: boundary-anchor override + Direct refusal at boundary ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v24_boundary_anchor_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V24 -> {}", path);

    let h264 = std::env::temp_dir().join("phasm_v24.h264");
    let dec = std::env::temp_dir().join("phasm_v24.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V24 audit (15 frames, all MBs) ===");
    eprintln!("V14 default RDO:                     wall=1, body=1005, max_dev=164");
    eprintln!("V15 boundary penalty alone:          wall=3, body=817,  max_dev=157");
    eprintln!("V21 force-L0 (MV=0,0 magic):         wall=6, body=570,  max_dev=51");
    eprintln!("V22 IPPPP (clean ref):               wall=0, body=21,   max_dev=47");
    eprintln!("V24 boundary anchor + refuse:        wall={wall_anom}, body={body_anom}, max_dev={max_dev_overall}");
    eprintln!();
    eprintln!("Target: max_dev close to V21/V22 (< 60). Body < 200 ideal.");
}

/// **Phase 2.10 — per-MB artifact dump** (2026-05-08 #251).
///
/// After V14-V24 luma-ME / mode-decision attempts all failed to
/// move max_dev from 164, this test gathers the empirical data
/// the next fix attempt actually needs. For each B-frame:
///
/// 1. Records every MB's (mode, MVs, SATDs, source stats,
///    boundary flag) via the encoder-side recorder gated by
///    PHASM_B_INSTRUMENT=1.
/// 2. After encode + ffmpeg decode, computes per-MB max\|Δ\|
///    against source.
/// 3. Sorts and dumps the TOP 10 worst-deviation B-MBs with:
///       - encoder-recorded data (mode, MVs, SATDs, boundary)
///       - 16×16 luma source / encoder.visual_recon /
///         ffmpeg.decode side-by-side
///       - 8×8 chroma source / decoded U+V
///       - residual signature (sum |source - decode|)
/// 4. Aggregates: mode histogram of worst-10, chroma-zero rate
///    in worst-10, boundary-flag rate.
///
/// The data tells us which non-luma-ME hypothesis (P1 chroma /
/// P3 trellis / P5 CBP / P6 mux mismatch — see
/// `memory/h264_b_frame_v1_blocker_diagnostic_2026_05_08.md`)
/// matches the bug class.
#[test]
#[ignore]
fn phase_2_10_b_mb_artifact_dump_1080p_45f() {
    use phasm_core::codec::h264::encoder::mb_decision_b::{
        drain_b_mb_records, BMbRecord, B_INSTRUMENT_FRAME_IDX,
    };

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(15);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_CAND");
        std::env::remove_var("PHASM_B_MULTI_REFINE");
        std::env::remove_var("PHASM_B_DIRECT_VALIDATE");
        // Disable V24's boundary refusal to capture V14-baseline behaviour.
        std::env::set_var("PHASM_B_NO_BOUNDARY_REFUSE", "1");
        std::env::set_var("PHASM_B_INSTRUMENT", "1");
    }

    // Drain any records from prior tests + reset frame counter.
    let _ = drain_b_mb_records();
    B_INSTRUMENT_FRAME_IDX.store(0, std::sync::atomic::Ordering::Relaxed);

    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== Phase 2.10: per-MB artifact dump (V14-baseline behaviour) ===");
    let mut bs = Vec::new();
    let mut display_to_b_frame_idx: std::collections::HashMap<u32, u32> =
        std::collections::HashMap::new();
    let mut b_frame_counter = 0u32;
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => {
                b_frame_counter += 1;
                display_to_b_frame_idx.insert(eo.display_idx, b_frame_counter);
                enc.encode_b_frame(f)
            }
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }

    let records = drain_b_mb_records();
    eprintln!("captured {} per-MB records across {} B-frames",
              records.len(), b_frame_counter);

    let h264 = std::env::temp_dir().join("phasm_p210.h264");
    let dec = std::env::temp_dir().join("phasm_p210.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let y_size = (width * height) as usize;
    let c_size = y_size / 4;

    // Compute per-MB max|Δ| (luma) for every B-frame display_idx.
    // Sort by max|Δ| desc.
    #[derive(Debug, Clone)]
    struct MbDev {
        display_idx: u32,
        b_frame_idx: u32,
        mb_x: usize,
        mb_y: usize,
        max_luma_dev: u8,
        sum_luma_dev: u32,
        max_u_dev: u8,
        max_v_dev: u8,
        sum_u_dev: u32,
        sum_v_dev: u32,
        src_mean: u8,
    }
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut all_devs: Vec<MbDev> = Vec::new();
    for (&display_idx, &b_idx) in &display_to_b_frame_idx {
        let off = display_idx as usize * frame_size;
        let src_y = &yuv[off..off + y_size];
        let src_u = &yuv[off + y_size..off + y_size + c_size];
        let src_v = &yuv[off + y_size + c_size..off + frame_size];
        let dec_y = &decoded[off..off + y_size];
        let dec_u = &decoded[off + y_size..off + y_size + c_size];
        let dec_v = &decoded[off + y_size + c_size..off + frame_size];

        for mby in 0..mb_h {
            for mbx in 0..mb_w {
                // Luma 16×16
                let mut max_l = 0u8;
                let mut sum_l = 0u32;
                let mut src_sum = 0u32;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = (mby * 16 + dy) * width as usize + mbx * 16 + dx;
                        let d = (src_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
                        sum_l += d;
                        src_sum += src_y[idx] as u32;
                        if (d as u8) > max_l { max_l = d as u8; }
                    }
                }
                // Chroma 8×8 (each)
                let mut max_u = 0u8;
                let mut max_v = 0u8;
                let mut sum_u = 0u32;
                let mut sum_v = 0u32;
                let cw = (width / 2) as usize;
                for dy in 0..8 {
                    for dx in 0..8 {
                        let cidx = (mby * 8 + dy) * cw + mbx * 8 + dx;
                        let du = (src_u[cidx] as i32 - dec_u[cidx] as i32).unsigned_abs();
                        let dv = (src_v[cidx] as i32 - dec_v[cidx] as i32).unsigned_abs();
                        sum_u += du; sum_v += dv;
                        if (du as u8) > max_u { max_u = du as u8; }
                        if (dv as u8) > max_v { max_v = dv as u8; }
                    }
                }
                if max_l > 30 || max_u > 30 || max_v > 30 {
                    all_devs.push(MbDev {
                        display_idx,
                        b_frame_idx: b_idx,
                        mb_x: mbx,
                        mb_y: mby,
                        max_luma_dev: max_l,
                        sum_luma_dev: sum_l,
                        max_u_dev: max_u,
                        max_v_dev: max_v,
                        sum_u_dev: sum_u,
                        sum_v_dev: sum_v,
                        src_mean: (src_sum / 256) as u8,
                    });
                }
            }
        }
    }
    all_devs.sort_by_key(|d| std::cmp::Reverse(d.max_luma_dev));
    let top_n = all_devs.iter().take(10).cloned().collect::<Vec<_>>();

    eprintln!("\n=== TOP 10 worst-luma-deviation B-MBs ===");
    for (rank, dev) in top_n.iter().enumerate() {
        // Find matching encoder record (frame_idx, mb_x, mb_y).
        let rec = records.iter().find(|r| {
            r.frame_idx == dev.b_frame_idx
                && r.mb_x as usize == dev.mb_x
                && r.mb_y as usize == dev.mb_y
        });
        let mode_name = match rec.map(|r| r.mode_id).unwrap_or(255) {
            1 => "Direct",
            2 => "L0",
            3 => "L1",
            4 => "Bi",
            _ => "?",
        };
        eprintln!(
            "#{} d={} mb=({:>3},{:>3}) src_mean={:>3} max|Δ|: Y={:>3} U={:>3} V={:>3} \
             sum|Δ|: Y={:>5} U={:>4} V={:>4}",
            rank + 1, dev.display_idx, dev.mb_x, dev.mb_y, dev.src_mean,
            dev.max_luma_dev, dev.max_u_dev, dev.max_v_dev,
            dev.sum_luma_dev, dev.sum_u_dev, dev.sum_v_dev,
        );
        if let Some(r) = rec {
            eprintln!(
                "    encoder: mode={} mvL0=({:>4},{:>4}) mvL1=({:>4},{:>4}) \
                 direct.L0=({:>4},{:>4}) direct.L1=({:>4},{:>4}) boundary={}",
                mode_name,
                r.mv_l0_x, r.mv_l0_y, r.mv_l1_x, r.mv_l1_y,
                r.direct_mv_l0_x, r.direct_mv_l0_y,
                r.direct_mv_l1_x, r.direct_mv_l1_y,
                r.at_boundary,
            );
            eprintln!(
                "    SATDs: SkipOrDirect={:>5} L0={:>5} L1={:>5} Bi={:>5}",
                r.satd_skip_or_direct, r.satd_l0, r.satd_l1, r.satd_bi,
            );
        } else {
            eprintln!("    (no encoder record — Skip pre-check fired or non-RDO path?)");
        }
    }

    // Aggregate stats.
    let mut mode_hist: std::collections::HashMap<u8, u32> = std::collections::HashMap::new();
    let mut boundary_hits = 0u32;
    let mut chroma_zero_count = 0u32;
    for dev in &top_n {
        let rec = records.iter().find(|r| {
            r.frame_idx == dev.b_frame_idx
                && r.mb_x as usize == dev.mb_x
                && r.mb_y as usize == dev.mb_y
        });
        if let Some(r) = rec {
            *mode_hist.entry(r.mode_id).or_insert(0) += 1;
            if r.at_boundary { boundary_hits += 1; }
        }
        // Chroma "zero" = decoded chroma sum is much smaller than source variation
        // i.e. chroma residual was mostly dropped. Heuristic: max chroma dev > 30
        // suggests chroma is being lost.
        if dev.max_u_dev > 30 || dev.max_v_dev > 30 {
            chroma_zero_count += 1;
        }
    }
    eprintln!("\n=== Aggregate (top 10) ===");
    eprintln!("Mode histogram:");
    let names = ["?", "Direct", "L0", "L1", "Bi"];
    for (mode, count) in &mode_hist {
        let name = names.get(*mode as usize).unwrap_or(&"?");
        eprintln!("  {} ({}): {}", name, mode, count);
    }
    eprintln!("Boundary-flag fired: {}/10", boundary_hits);
    eprintln!("Chroma-deviated (max U or V > 30): {}/10", chroma_zero_count);

    unsafe {
        std::env::remove_var("PHASM_B_INSTRUMENT");
        std::env::remove_var("PHASM_B_NO_BOUNDARY_REFUSE");
    }
}

/// **Phase 2.11 — V25-state per-MB artifact dump with chroma axis**
/// (2026-05-08 #251). Twin of phase_2_10 but runs with V25's
/// magnitude clamp ACTIVE (default state, no env override). Adds
/// chroma-sorted top-10 to separately localise the chroma-stripe
/// artifact class user identified in V25 frame analysis.
#[test]
#[ignore]
fn phase_2_11_b_mb_artifact_dump_v25_state() {
    use phasm_core::codec::h264::encoder::mb_decision_b::{
        drain_b_mb_records, B_INSTRUMENT_FRAME_IDX,
    };

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(15);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_CAND");
        std::env::remove_var("PHASM_B_MULTI_REFINE");
        std::env::remove_var("PHASM_B_DIRECT_VALIDATE");
        std::env::remove_var("PHASM_B_NO_BOUNDARY_REFUSE");
        std::env::remove_var("PHASM_B_NO_DIRECT_MAGCLAMP");
        std::env::set_var("PHASM_B_INSTRUMENT", "1");
    }

    let _ = drain_b_mb_records();
    B_INSTRUMENT_FRAME_IDX.store(0, std::sync::atomic::Ordering::Relaxed);

    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== Phase 2.11: V25-state per-MB dump (mag-clamp + boundary refuse ACTIVE) ===");
    let mut bs = Vec::new();
    let mut display_to_b_frame_idx: std::collections::HashMap<u32, u32> =
        std::collections::HashMap::new();
    let mut b_frame_counter = 0u32;
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => {
                b_frame_counter += 1;
                display_to_b_frame_idx.insert(eo.display_idx, b_frame_counter);
                enc.encode_b_frame(f)
            }
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }

    let records = drain_b_mb_records();
    eprintln!("captured {} records / {} B-frames",
              records.len(), b_frame_counter);

    let h264 = std::env::temp_dir().join("phasm_p211.h264");
    let dec = std::env::temp_dir().join("phasm_p211.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let y_size = (width * height) as usize;
    let c_size = y_size / 4;

    #[derive(Debug, Clone)]
    struct MbDev {
        display_idx: u32,
        b_frame_idx: u32,
        mb_x: usize,
        mb_y: usize,
        max_l: u8,
        max_u: u8,
        max_v: u8,
        sum_l: u32,
        sum_u: u32,
        sum_v: u32,
        src_mean: u8,
    }
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut all_devs: Vec<MbDev> = Vec::new();
    for (&display_idx, &b_idx) in &display_to_b_frame_idx {
        let off = display_idx as usize * frame_size;
        let src_y = &yuv[off..off + y_size];
        let src_u = &yuv[off + y_size..off + y_size + c_size];
        let src_v = &yuv[off + y_size + c_size..off + frame_size];
        let dec_y = &decoded[off..off + y_size];
        let dec_u = &decoded[off + y_size..off + y_size + c_size];
        let dec_v = &decoded[off + y_size + c_size..off + frame_size];
        for mby in 0..mb_h {
            for mbx in 0..mb_w {
                let mut max_l = 0u8;
                let mut sum_l = 0u32;
                let mut src_sum = 0u32;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = (mby * 16 + dy) * width as usize + mbx * 16 + dx;
                        let d = (src_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
                        sum_l += d;
                        src_sum += src_y[idx] as u32;
                        if (d as u8) > max_l { max_l = d as u8; }
                    }
                }
                let mut max_u = 0u8; let mut max_v = 0u8;
                let mut sum_u = 0u32; let mut sum_v = 0u32;
                let cw = (width / 2) as usize;
                for dy in 0..8 {
                    for dx in 0..8 {
                        let cidx = (mby * 8 + dy) * cw + mbx * 8 + dx;
                        let du = (src_u[cidx] as i32 - dec_u[cidx] as i32).unsigned_abs();
                        let dv = (src_v[cidx] as i32 - dec_v[cidx] as i32).unsigned_abs();
                        sum_u += du; sum_v += dv;
                        if (du as u8) > max_u { max_u = du as u8; }
                        if (dv as u8) > max_v { max_v = dv as u8; }
                    }
                }
                if max_l > 30 || max_u > 20 || max_v > 20 {
                    all_devs.push(MbDev {
                        display_idx, b_frame_idx: b_idx, mb_x: mbx, mb_y: mby,
                        max_l, max_u, max_v, sum_l, sum_u, sum_v,
                        src_mean: (src_sum / 256) as u8,
                    });
                }
            }
        }
    }

    let dump_top = |label: &str, devs: &[MbDev]| {
        eprintln!("\n=== TOP 10 worst by {label} ===");
        for (rank, dev) in devs.iter().take(10).enumerate() {
            let rec = records.iter().find(|r| {
                r.frame_idx == dev.b_frame_idx
                    && r.mb_x as usize == dev.mb_x
                    && r.mb_y as usize == dev.mb_y
            });
            let mode_name = match rec.map(|r| r.mode_id).unwrap_or(255) {
                1 => "Direct", 2 => "L0", 3 => "L1", 4 => "Bi", _ => "?",
            };
            eprintln!(
                "#{} d={} mb=({:>3},{:>3}) src_mean={:>3} \
                 max|Δ|: Y={:>3} U={:>3} V={:>3}  sum: Y={:>5} U={:>4} V={:>4}",
                rank + 1, dev.display_idx, dev.mb_x, dev.mb_y, dev.src_mean,
                dev.max_l, dev.max_u, dev.max_v,
                dev.sum_l, dev.sum_u, dev.sum_v,
            );
            if let Some(r) = rec {
                eprintln!(
                    "    mode={} mvL0=({:>4},{:>4}) mvL1=({:>4},{:>4}) \
                     direct.L0=({:>4},{:>4}) direct.L1=({:>4},{:>4}) bndry={} \
                     SATDs: D={} L0={} L1={} Bi={}",
                    mode_name,
                    r.mv_l0_x, r.mv_l0_y, r.mv_l1_x, r.mv_l1_y,
                    r.direct_mv_l0_x, r.direct_mv_l0_y,
                    r.direct_mv_l1_x, r.direct_mv_l1_y,
                    r.at_boundary,
                    r.satd_skip_or_direct, r.satd_l0, r.satd_l1, r.satd_bi,
                );
            } else {
                eprintln!("    (no record — Skip pre-check or non-RDO path)");
            }
        }
    };

    let mut by_l = all_devs.clone();
    by_l.sort_by_key(|d| std::cmp::Reverse(d.max_l));
    dump_top("LUMA max|Δ|", &by_l);

    let mut by_u = all_devs.clone();
    by_u.sort_by_key(|d| std::cmp::Reverse(d.max_u));
    dump_top("U-CHROMA max|Δ|", &by_u);

    let mut by_v = all_devs.clone();
    by_v.sort_by_key(|d| std::cmp::Reverse(d.max_v));
    dump_top("V-CHROMA max|Δ|", &by_v);

    // Combined: weighted Y+U+V to see "all axes"
    let mut by_combined = all_devs.clone();
    by_combined.sort_by_key(|d| std::cmp::Reverse(
        d.max_l as u32 + 2 * (d.max_u as u32 + d.max_v as u32)
    ));
    dump_top("COMBINED (Y + 2×U + 2×V)", &by_combined);

    eprintln!("\n=== summary ===");
    eprintln!("MBs above threshold: {}", all_devs.len());
    eprintln!("Worst luma max|Δ|: {}", by_l.first().map(|d| d.max_l).unwrap_or(0));
    eprintln!("Worst U-chroma max|Δ|: {}", by_u.first().map(|d| d.max_u).unwrap_or(0));
    eprintln!("Worst V-chroma max|Δ|: {}", by_v.first().map(|d| d.max_v).unwrap_or(0));

    unsafe { std::env::remove_var("PHASM_B_INSTRUMENT"); }
}

/// **V25 — magnitude-based Direct refusal + ME anchor override**
/// (2026-05-08 #251).
///
/// Phase 2.10 instrumentation found 8/10 worst MBs were mode=Direct
/// with chain-propagated huge MVs (25-50 px). V24's variance-based
/// boundary detector missed them because chain MBs all inherit the
/// same huge MV (low variance). V25 adds an absolute-magnitude
/// trigger: refuse Direct + override ME anchor when |spatial_direct
/// _mv| or |predicted_mv| > 32 qpel (8 px). Default ON.
///
/// Output: `phasm_v25_magnitude_clamp_clean.mp4`. Audit + visual
/// A/B vs V14/V21/V22.
#[test]
#[ignore]
fn phase_1_1_d_v25_magnitude_clamp_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_CAND");
        std::env::remove_var("PHASM_B_MULTI_REFINE");
        std::env::remove_var("PHASM_B_DIRECT_VALIDATE");
        std::env::remove_var("PHASM_B_NO_BOUNDARY_REFUSE");
        std::env::remove_var("PHASM_B_NO_DIRECT_MAGCLAMP");
        std::env::remove_var("PHASM_B_INSTRUMENT");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V25: magnitude-clamp Direct + ME anchor override (Phase 2.10 finding) ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v25_magnitude_clamp_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V25 -> {}", path);

    let h264 = std::env::temp_dir().join("phasm_v25.h264");
    let dec = std::env::temp_dir().join("phasm_v25.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V25 audit (15 frames, all MBs) ===");
    eprintln!("V14 default RDO:                     wall=1, body=1005, max_dev=164");
    eprintln!("V21 force-L0 (MV=0,0 magic):         wall=6, body=570,  max_dev=51");
    eprintln!("V22 IPPPP (clean ref):               wall=0, body=21,   max_dev=47");
    eprintln!("V24 boundary anchor + refuse:        wall=1, body=798,  max_dev=166");
    eprintln!("V25 magnitude clamp + anchor:        wall={wall_anom}, body={body_anom}, max_dev={max_dev_overall}");
}

/// **V26 — V25 + ME-result hard clamp** (2026-05-08 #251).
///
/// Phase 2.11 instrumentation found post-V25 the worst MBs are
/// mode=L1 with HUGE ME-derived MVs (e.g. mvL1=(132,-295)). V25's
/// clamp only refused Direct/spatial-direct; ME runaway in L0/L1
/// search remained. V26 hard-clamps ME RESULT to 64 qpel = 16 px:
/// if |result.mv| exceeds, snap to (0,0). Default ON.
///
/// Output: `phasm_v26_me_clamp_clean.mp4`.
#[test]
#[ignore]
fn phase_1_1_d_v26_me_clamp_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_CAND");
        std::env::remove_var("PHASM_B_MULTI_REFINE");
        std::env::remove_var("PHASM_B_DIRECT_VALIDATE");
        std::env::remove_var("PHASM_B_NO_BOUNDARY_REFUSE");
        std::env::remove_var("PHASM_B_NO_DIRECT_MAGCLAMP");
        std::env::remove_var("PHASM_B_NO_ME_RESULT_CLAMP");
        std::env::remove_var("PHASM_B_INSTRUMENT");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V26: V25 + ME-result hard clamp at 64 qpel (16 px) ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v26_me_clamp_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V26 -> {}", path);

    let h264 = std::env::temp_dir().join("phasm_v26.h264");
    let dec = std::env::temp_dir().join("phasm_v26.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V26 audit (15 frames, all MBs) ===");
    eprintln!("V14 default RDO:                       wall=1, body=1005, max_dev=164");
    eprintln!("V21 force-L0 (MV=0,0 magic):           wall=6, body=570,  max_dev=51");
    eprintln!("V22 IPPPP (clean ref):                 wall=0, body=21,   max_dev=47");
    eprintln!("V25 magnitude clamp + anchor:          wall=1, body=707,  max_dev=149");
    eprintln!("V26 V25 + ME result clamp 64 qpel:     wall={wall_anom}, body={body_anom}, max_dev={max_dev_overall}");
}

/// **V15 — PHASM_B_BOUNDARY_PENALTY validation** (2026-05-07 evening).
///
/// Hypothesis: visible artifacts on body MBs are mode-decision quality
/// (encoder picks B_Direct with derived spatial-direct MV ≈ 0 from wall
/// neighbours; resulting prediction = wall pixels at body position →
/// visible "wall texture invading pants"). Bitstream is faithful;
/// player decodes faithfully; bug is at RDO selection.
///
/// V15 enables `PHASM_B_BOUNDARY_PENALTY=1` (already implemented in
/// `mb_decision_b_rdo`) which detects motion-boundary MBs by
/// neighbour-MV span and inflates Direct's RDO cost so explicit-MV
/// modes win for those MBs.
///
/// Output: `phasm_v15_boundary_penalty_clean.mp4` on Desktop.
///
/// Expected: body_anomalies should drop substantially if hypothesis
/// is correct. If unchanged, the artifact's root cause is elsewhere
/// (e.g., ME quality, secondary CABAC residual bug, or the penalty
/// threshold needs tuning).
#[test]
#[ignore]
fn phase_1_1_d_v15_boundary_penalty_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_INVEST_DUMP");
        std::env::remove_var("PHASM_INVEST_DUMP_ALL");
        std::env::remove_var("PHASM_B_FORCE_MODE");
        // V15: ENABLE boundary penalty (V14 explicitly removed it).
        std::env::set_var("PHASM_B_BOUNDARY_PENALTY", "1");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V15: PHASM_B_BOUNDARY_PENALTY=1 (motion-boundary Direct cost inflation) ===");
    let mut bs = Vec::new();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v15_boundary_penalty_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V15 -> {} ({} bytes)", path, mp4.len());

    let h264 = std::env::temp_dir().join("phasm_v15.h264");
    let dec = std::env::temp_dir().join("phasm_v15.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V15 audit (15 frames, all MBs) ===");
    eprintln!("V14 (no penalty): wall=1, body=1005, max_dev=164");
    eprintln!("V15 (with penalty): wall={wall_anom}, body={body_anom}, max_dev={max_dev_overall}");
    unsafe { std::env::remove_var("PHASM_B_BOUNDARY_PENALTY"); }
}

/// **V16 — Direct+residual emission validation** (2026-05-07 evening).
///
/// V15 (boundary penalty) addressed Class-2a (motion-boundary Direct
/// MV-derivation errors). V16 adds Class-2b fix: allow Direct mode
/// to emit residual coefficients (CBP > 0) instead of the legacy
/// hardcoded CBP=0. Per spec § 7.4.5.1, B_Direct_16x16 supports
/// non-zero CBP. Real-world consumer encoders use this routinely on
/// textured B-MBs to capture high-freq detail (e.g., the scarf
/// colour-stripe artifact visible on V14/V15 demos).
///
/// Combined with V15's penalty, V16 should close most remaining
/// visible body anomalies.
///
/// Output: `phasm_v16_direct_residual_clean.mp4` on Desktop.
#[test]
#[ignore]
fn phase_1_1_d_v16_direct_residual_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_INVEST_DUMP");
        std::env::remove_var("PHASM_INVEST_DUMP_ALL");
        std::env::remove_var("PHASM_B_FORCE_MODE");
        // V16: ENABLE boundary penalty (carry over V15's gain).
        std::env::set_var("PHASM_B_BOUNDARY_PENALTY", "1");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V16: V15's boundary penalty + Direct+residual emission (CBP > 0) ===");
    let mut bs = Vec::new();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v16_direct_residual_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V16 -> {} ({} bytes)", path, mp4.len());

    let h264 = std::env::temp_dir().join("phasm_v16.h264");
    let dec = std::env::temp_dir().join("phasm_v16.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V16 audit (15 frames, all MBs) ===");
    eprintln!("V14 (no penalty, no Direct+res): wall=1, body=1005, max_dev=164");
    eprintln!("V15 (penalty only):              wall=3, body=817,  max_dev=157");
    eprintln!("V16 (penalty + Direct+res):      wall={wall_anom}, body={body_anom}, max_dev={max_dev_overall}");
    unsafe { std::env::remove_var("PHASM_B_BOUNDARY_PENALTY"); }
}

/// **§B-direct-fix.v3 V13 — predictor magnitude clamp** (2026-05-07).
///
/// V11 closed wall MBs. V12 added neighbour candidates but only fixed
/// 1.5% of body anomalies (1063 → 1048). User insight: not just
/// adjusting little stuff — needs structural fix.
///
/// Per-MB dump revealed body MBs picking MVs like (-205, 112) — chain-
/// propagated wrong MVs that ME's bit-cost-anchor=median made
/// preferable to ZERO. dump_b_frame_recon_vs_decode test confirmed
/// encoder.recon ≡ ffmpeg.decode → encoder is honest, the bug is ME's
/// cost function biasing toward wrong-direction propagated predictors.
///
/// V13 fix: clamp ME's bit-cost anchor to ZERO when predicted_mv
/// magnitude > 32 quarter-pel (8 px). Real motion has small predicted;
/// chain-propagated bad MVs have huge predicted. The clamp breaks the
/// chain by removing the rate-cost bias toward continuing the chain.
///
/// Output: `phasm_v13_anchor_clamp_clean.mp4`. Audit reports body
/// anomaly count reduction from V12's 1048.
#[test]
#[ignore]
fn phase_1_1_d_v13_anchor_clamp_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_INVEST_DUMP");
        std::env::remove_var("PHASM_INVEST_DUMP_ALL");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V13: predictor magnitude clamp (anchor=ZERO if |pred|>32 qpel) ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v13_anchor_clamp_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V13 -> {}", path);

    let h264 = std::env::temp_dir().join("phasm_v13.h264");
    let dec = std::env::temp_dir().join("phasm_v13.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V13 audit (15 frames, all MBs) ===");
    eprintln!("wall_anomalies={wall_anom}  body_anomalies={body_anom}  max_dev_overall={max_dev_overall}");
    eprintln!("V11 wall=2, body=1063  V12 wall=3, body=1048");
    eprintln!("V13 (anchor clamp): wall={wall_anom}, body={body_anom}");
}

/// **§B-direct-fix.v3 V12 — full-symmetry P-style ME for B-frames** (2026-05-07).
///
/// V11 added `[predicted, ZERO]` to B-frame ME and closed wall-MB
/// anomalies. User visual confirmed wall is clean. But body-internal
/// boundary MBs (pants/skin, jacket/scarf junctions) still show
/// colored block artifacts.
///
/// Cause: heterogeneous-source MBs need a non-zero correct MV (1-2 px
/// from camera shake / body micro-motion). Predicted_mv may inherit a
/// wrong-direction body MV via raster propagation; zero is wrong
/// because the source genuinely moved. Real best MV is what actual
/// neighbour MBs picked — not the median, not zero.
///
/// V12 extends V11 by adding A/B/C neighbour MVs (left, top, top-right)
/// from the current frame's L0/L1 grids — full symmetry with
/// P-frame's `build_me_candidates`. Body MBs in raster scan inherit
/// the real motion their neighbour body MBs picked.
///
/// Output: `phasm_v12_full_mepred_clean.mp4` on Desktop.
#[test]
#[ignore]
fn phase_1_1_d_v12_full_mepred_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        // §INVEST 2026-05-07 — allow PHASM_INVEST_DUMP through if set externally.
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V12: legacy default + full-symmetry P-style ME (predicted+ZERO+A+B+C) ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v12_full_mepred_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V12 -> {}", path);

    // Audit ALL MBs (not just wall) for body-area artifacts.
    let h264 = std::env::temp_dir().join("phasm_v12.h264");
    let dec = std::env::temp_dir().join("phasm_v12.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut wall_anom = 0;
    let mut body_anom = 0;
    let mut all_max_dev = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev { max_dev = dev; }
                    }
                }
                let is_wall = src_min >= 220 && src_max - src_min <= 10;
                if max_dev > 30 {
                    if is_wall { wall_anom += 1; } else { body_anom += 1; }
                    if max_dev > all_max_dev { all_max_dev = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V12 audit (15 frames, all MBs) ===");
    eprintln!("wall_anomalies={wall_anom}  body_anomalies={body_anom}  max_dev_overall={all_max_dev}");
    eprintln!("V11 wall: 2, body unmeasured. V12 should drop body too.");
}

/// **§B-direct-fix.v3 V11 — multi-pred ME root-cause fix** (2026-05-07).
///
/// User screenshots showed blue blocks in white-wall area on V7/V8/V9.
/// V10 deblock-bisect ruled out deblock filter spread. Per-MB MV dump
/// of B-frames revealed wall MBs getting MV=(-85, 0) to (-404, -32) due
/// to spatial-direct's median predictor propagating body MVs through
/// raster-scan neighbours — and B-frame ME's `me.search_block(...)`
/// passes only `[predicted_mv]` so hex-search stayed in the wrong basin.
/// P-frame ME (`build_me_candidates`) always includes ZERO + neighbour
/// MVs as candidates, which is why P never has wall anomalies.
///
/// Fix shipped in this session: B-frame ME now passes
/// `[predicted_mv, ZERO]` to `search_block_with_candidates`. ME tests
/// both seeds; zero wins for wall content where SATD at zero is
/// near-perfect. RDO picks L0 (cheap zero-MV) over Direct (expensive
/// wrong-MV).
///
/// V11 tests legacy default (no Path 1, no Path 4) to confirm the ME
/// fix alone closes the wall anomalies.
#[test]
#[ignore]
fn phase_1_1_d_v11_me_zero_fix_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_INVEST_DUMP");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("=== V11: legacy default + ME zero-MV fix ===");
    let mut bs = Vec::new();
    let t0 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encoded: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v11_me_zero_fix_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("V11 -> {}", path);

    // Audit wall anomalies.
    let h264 = std::env::temp_dir().join("phasm_v11.h264");
    let dec = std::env::temp_dir().join("phasm_v11.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    let decoded = std::fs::read(&dec).unwrap();
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let mut total_anomalies = 0;
    let mut max_dev_overall = 0u8;
    for frame_idx in 0..n_frames.min(15) {
        let off = frame_idx * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut src_sum: u32 = 0;
                let mut src_min = 255u8;
                let mut src_max = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let s = yuv[off + idx];
                        src_sum += s as u32;
                        if s < src_min { src_min = s; }
                        if s > src_max { src_max = s; }
                    }
                }
                if src_sum / 256 < 220 || src_max - src_min > 10 { continue; }
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                        let d = (yuv[off+idx] as i32 - decoded[off+idx] as i32).unsigned_abs() as u8;
                        if d > max_dev { max_dev = d; }
                    }
                }
                if max_dev > 30 {
                    total_anomalies += 1;
                    if max_dev > max_dev_overall { max_dev_overall = max_dev; }
                }
            }
        }
    }
    eprintln!("\n=== V11 wall-anomaly audit (15 frames) ===");
    eprintln!("total: {total_anomalies}, max_dev: {max_dev_overall}");
    eprintln!("V7-spatial baseline: 3 anomalies, max_dev 79");
    eprintln!("V8-path4: 7 anomalies, max_dev 73");
    eprintln!("V9-path1+4: 13 anomalies, max_dev 78");
    eprintln!("V11 (ME zero fix only): {total_anomalies} anomalies, max_dev {max_dev_overall}");
}

/// **§B-direct-fix.v3 V10 — disable deblock to bisect** (2026-05-07).
///
/// Deblock-disable A/B test. Renders TWO clean mp4s on Desktop:
/// - `phasm_v10_baseline_clean.mp4` — default deblock-on (legacy V7-spatial)
/// - `phasm_v10_no_deblock_clean.mp4` — deblock disabled via PHASM_DISABLE_DEBLOCK
///
/// If wall-area artifacts disappear in the no-deblock variant, deblock
/// spread from body→wall is the root cause.
#[test]
#[ignore]
fn phase_1_1_d_v10_deblock_bisect_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(45);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    let encode = |disable_deblock: bool| -> Vec<u8> {
        unsafe {
            std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
            std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
            if disable_deblock {
                std::env::set_var("PHASM_DISABLE_DEBLOCK", "1");
            } else {
                std::env::remove_var("PHASM_DISABLE_DEBLOCK");
            }
        }
        let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = true;
        enc.enable_b_frames = true;
        enc.b_rdo_config =
            phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;
        let mut bs = Vec::new();
        for eo in iter_encode_order(n_frames, pattern) {
            let d = eo.display_idx as usize;
            let f = &yuv[d * frame_size..(d + 1) * frame_size];
            let bytes = match eo.frame_type {
                FrameType::Idr => enc.encode_i_frame(f),
                FrameType::P => enc.encode_p_frame(f),
                FrameType::B => enc.encode_b_frame(f),
            }
            .unwrap_or_else(|e| panic!("encode error: {e}"));
            bs.extend_from_slice(&bytes);
        }
        bs
    };

    eprintln!("=== V10-A: baseline (deblock ON, legacy default) ===");
    let baseline_bs = encode(false);
    eprintln!("  encoded: {} bytes", baseline_bs.len());

    eprintln!("=== V10-B: no-deblock (PHASM_DISABLE_DEBLOCK=1) ===");
    let nodeblock_bs = encode(true);
    eprintln!("  encoded: {} bytes", nodeblock_bs.len());

    let timing = FrameTiming::FPS_30;
    let baseline_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &baseline_bs, width, height, timing, pattern, n_frames,
    ).expect("baseline mp4 mux");
    let nodeblock_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &nodeblock_bs, width, height, timing, pattern, n_frames,
    ).expect("nodeblock mp4 mux");

    std::fs::write("/Users/cgaffga/Desktop/phasm_v10_baseline_clean.mp4", &baseline_mp4).unwrap();
    std::fs::write("/Users/cgaffga/Desktop/phasm_v10_no_deblock_clean.mp4", &nodeblock_mp4).unwrap();

    // Audit wall anomalies in both.
    let audit = |bs: &[u8], label: &str| {
        let h264 = std::env::temp_dir().join(format!("phasm_v10_{label}.h264"));
        let dec = std::env::temp_dir().join(format!("phasm_v10_{label}.dec.yuv"));
        std::fs::write(&h264, bs).unwrap();
        let _ = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
            .status().expect("ffmpeg run");
        let decoded = std::fs::read(&dec).unwrap();
        let mb_w = (width / 16) as usize;
        let mb_h = (height / 16) as usize;
        let mut total_anomalies = 0;
        let mut max_max_dev = 0u8;
        for frame_idx in 0..n_frames.min(15) {
            let off = frame_idx * frame_size;
            for mb_y in 0..mb_h {
                for mb_x in 0..mb_w {
                    let mut src_sum: u32 = 0;
                    let mut src_min = 255u8;
                    let mut src_max = 0u8;
                    for dy in 0..16 {
                        for dx in 0..16 {
                            let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                            let s = yuv[off + idx];
                            src_sum += s as u32;
                            if s < src_min { src_min = s; }
                            if s > src_max { src_max = s; }
                        }
                    }
                    if src_sum / 256 < 220 || src_max - src_min > 10 { continue; }
                    let mut max_dev = 0u8;
                    for dy in 0..16 {
                        for dx in 0..16 {
                            let idx = ((mb_y * 16 + dy) * width as usize + mb_x * 16 + dx) as usize;
                            let d = (yuv[off+idx] as i32 - decoded[off+idx] as i32).unsigned_abs() as u8;
                            if d > max_dev { max_dev = d; }
                        }
                    }
                    if max_dev > 30 {
                        total_anomalies += 1;
                        if max_dev > max_max_dev { max_max_dev = max_dev; }
                    }
                }
            }
        }
        eprintln!("  [{label}] total wall anomalies (15 frames): {total_anomalies}, max_max_dev: {max_max_dev}");
    };

    audit(&baseline_bs, "baseline");
    audit(&nodeblock_bs, "no_deblock");

    unsafe { std::env::remove_var("PHASM_DISABLE_DEBLOCK"); }
}

/// **§B-direct-fix.v3 V9 — Path 1 + Path 4 combined** (2026-05-07).
///
/// User V7 visual feedback: temporal-direct + override is subjectively
/// better than spatial baseline (+0.44 dB), but doesn't fully close the
/// artifact. V8 showed Path 4 (spatial + boundary penalty) gives a
/// similar +0.40 dB on different frames. V9 stacks both — temporal-
/// direct base + near-zero override + Path 4 boundary penalty — to see
/// if the wins compound.
///
/// Output: `phasm_v9_path1plus4_clean.mp4` on Desktop. Compare against
/// V7-temporal-override (Path 1 alone) and V8-path4 (Path 4 alone) for
/// the full A/B picture. If V9 closes the visible artifact, ship the
/// combined config as v1.0.
#[test]
#[ignore]
fn phase_1_1_d_v9_path1_plus_path4_1080p_45f() {
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = yuv.len() / frame_size;
    let n_frames = n_frames.min(45);
    eprintln!("fixture: 1920x1072 × {} frames", n_frames);

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe {
        std::env::set_var("PHASM_B_TEMPORAL_DIRECT", "1");
        std::env::set_var("PHASM_B_BOUNDARY_PENALTY", "1");
    }
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

    eprintln!("\n=== V9: Path 1 (temporal+override) + Path 4 (boundary penalty) ===");
    let t0 = std::time::Instant::now();
    let mut bs = Vec::new();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame),
            FrameType::P => enc.encode_p_frame(frame),
            FrameType::B => enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("encode error: {e}"));
        bs.extend_from_slice(&bytes);
    }
    eprintln!("encode: {:?}, {} bytes", t0.elapsed(), bs.len());

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &bs, width, height, timing, pattern, n_frames,
    ).expect("mp4 mux");
    let path = "/Users/cgaffga/Desktop/phasm_v9_path1plus4_clean.mp4";
    std::fs::write(path, &mp4).expect("write mp4");
    eprintln!("\nV9 -> {}", path);

    // PSNR.
    let h264 = std::env::temp_dir().join("phasm_v9_path1plus4.h264");
    let dec = std::env::temp_dir().join("phasm_v9_path1plus4.dec.yuv");
    std::fs::write(&h264, &bs).unwrap();
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
        .status().expect("ffmpeg run");
    assert!(status.success(), "ffmpeg decode failed");
    let decoded = std::fs::read(&dec).unwrap();
    let y_size = (width * height) as usize;
    let mut sum_psnr = 0.0f64;
    eprintln!("\n=== V9 Per-frame Y PSNR (vs source) ===");
    for i in 0..n_frames {
        let off = i * frame_size;
        let mut sum_sq: u64 = 0;
        for (a, b) in yuv[off..off+y_size].iter().zip(decoded[off..off+y_size].iter()) {
            let d = (*a as i32 - *b as i32) as i64;
            sum_sq += (d * d) as u64;
        }
        let mse = sum_sq as f64 / y_size as f64;
        let psnr = if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() };
        sum_psnr += psnr;
        eprintln!("{:>4} {:>10.2}", i+1, psnr);
    }
    let n = n_frames as f64;
    eprintln!("\navg Y: {:.2} dB", sum_psnr/n);
    eprintln!("V7-spatial baseline: 40.93 dB");
    eprintln!("V7-temporal-override (Path 1 alone): 41.37 dB");
    eprintln!("V8-path4 (Path 4 alone): 41.33 dB");

    unsafe {
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
    }
}

/// **§B-cascade-real V5 — production-realistic 1080p × 60f cascade probe** (2026-05-07).
///
/// Run on a real 1080p × 60f IBPBP fixture using:
/// 1. The streaming-Viterbi orchestrator with a real text message
/// 2. The HandbrakeX264 mp4 muxer (proper ctts + edts → no first-frame
///    visual artifacts from B-frame display reordering)
/// 3. Per-frame Y/U/V PSNR (so we can see chroma vs luma damage split)
///
/// Writes both stego + clean mp4s to /Users/cgaffga/Desktop/ for visual
/// inspection.
///
/// Built for release-mode runs (overnight) — the streaming orchestrator
/// in debug build at 1080p × 60f would take many hours.
#[test]
#[ignore]
fn phase_1_1_d_v5_production_1080p_60f() {
    use phasm_core::codec::h264::stego::encode_pixels::
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files;
    use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let n_frames = yuv.len() / frame_size;
    assert!(n_frames >= 45,
        "fixture has only {} frames, need ≥45 for production probe", n_frames);
    eprintln!("fixture: 1920x1072 × {} frames", n_frames);

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42"); }

    eprintln!("\n=== §B-cascade-real V5 — production-realistic 1080p × 60f IBPBP ===");
    let t0 = std::time::Instant::now();
    let stego_bs =
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
            &yuv, width, height, n_frames, pattern,
            "Hello from phasm — this is a real production-style stego payload, ~70 chars.",
            &[],
            "v5-production-pass",
        ).expect("stego encode");
    eprintln!("stego encode: {:?}, {} bytes", t0.elapsed(), stego_bs.len());

    let mut clean_enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    clean_enc.entropy_mode = EntropyMode::Cabac;
    clean_enc.enable_transform_8x8 = true;
    clean_enc.enable_b_frames = true;
    clean_enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;
    let mut clean_bs = Vec::new();
    let t1 = std::time::Instant::now();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => clean_enc.encode_i_frame(frame),
            FrameType::P => clean_enc.encode_p_frame(frame),
            FrameType::B => clean_enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("clean encode error: {e}"));
        clean_bs.extend_from_slice(&bytes);
    }
    eprintln!("clean encode: {:?}, {} bytes", t1.elapsed(), clean_bs.len());

    // Mux both via HandbrakeX264 (proper ctts + edts).
    let timing = FrameTiming::FPS_30;
    let stego_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &stego_bs, width, height, timing, pattern, n_frames,
    ).expect("stego mp4 mux");
    let clean_mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264, &clean_bs, width, height, timing, pattern, n_frames,
    ).expect("clean mp4 mux");
    let stego_mp4_path = "/Users/cgaffga/Desktop/phasm_v5_production_1080p_stego.mp4";
    let clean_mp4_path = "/Users/cgaffga/Desktop/phasm_v5_production_1080p_clean.mp4";
    std::fs::write(stego_mp4_path, &stego_mp4).expect("write stego mp4");
    std::fs::write(clean_mp4_path, &clean_mp4).expect("write clean mp4");
    eprintln!("muxed stego mp4 -> {} ({} bytes)", stego_mp4_path, stego_mp4.len());
    eprintln!("muxed clean mp4 -> {} ({} bytes)", clean_mp4_path, clean_mp4.len());

    // Per-frame Y/U/V PSNR for both bitstreams.
    let psnr_each = |bs: &[u8], label: &str| -> Vec<(f64, f64, f64)> {
        let h264 = std::env::temp_dir().join(format!("phasm_v5_{label}.h264"));
        let dec = std::env::temp_dir().join(format!("phasm_v5_{label}.dec.yuv"));
        std::fs::write(&h264, bs).unwrap();
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
            .status().expect("ffmpeg run");
        assert!(status.success(), "ffmpeg decode failed for {label}");
        let decoded = std::fs::read(&dec).unwrap();
        let y_size = (width * height) as usize;
        let c_size = (width / 2 * height / 2) as usize;
        let mut psnrs = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let off = i * frame_size;
            let calc = |src: &[u8], dec: &[u8]| -> f64 {
                let mut sum_sq: u64 = 0;
                for (a, b) in src.iter().zip(dec.iter()) {
                    let d = (*a as i32 - *b as i32) as i64;
                    sum_sq += (d * d) as u64;
                }
                let mse = (sum_sq as f64) / (src.len() as f64);
                if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
            };
            let py = calc(&yuv[off..off+y_size], &decoded[off..off+y_size]);
            let pu = calc(&yuv[off+y_size..off+y_size+c_size],
                          &decoded[off+y_size..off+y_size+c_size]);
            let pv = calc(&yuv[off+y_size+c_size..off+frame_size],
                          &decoded[off+y_size+c_size..off+frame_size]);
            psnrs.push((py, pu, pv));
        }
        psnrs
    };

    let stego_psnrs = psnr_each(&stego_bs, "stego");
    let clean_psnrs = psnr_each(&clean_bs, "clean");

    eprintln!("\n=== Per-frame Y/U/V PSNR (1080p × 60f IBPBP gop=30) ===");
    eprintln!("{:>4} {:>10} {:>10} {:>10}  {:>10} {:>10} {:>10}  {:>9} {:>9} {:>9}",
        "n", "Y stego", "Y clean", "Y Δ",
        "U stego", "U clean", "U Δ",
        "V stego", "V clean", "V Δ");
    let (mut sum_dy, mut sum_du, mut sum_dv) = (0.0f64, 0.0f64, 0.0f64);
    let (mut sum_sy, mut sum_su, mut sum_sv) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n_frames {
        let (sy, su, sv) = stego_psnrs[i];
        let (cy, cu, cv) = clean_psnrs[i];
        eprintln!("{:>4} {:>10.2} {:>10.2} {:>+10.2}  {:>10.2} {:>10.2} {:>+10.2}  {:>9.2} {:>9.2} {:>+9.2}",
            i+1, sy, cy, sy-cy, su, cu, su-cu, sv, cv, sv-cv);
        sum_dy += sy - cy; sum_du += su - cu; sum_dv += sv - cv;
        sum_sy += sy; sum_su += su; sum_sv += sv;
    }
    let n = n_frames as f64;
    eprintln!("{:>4} {:>10} {:>10} {:>10}  {:>10} {:>10} {:>10}  {:>9} {:>9} {:>9}",
        "----", "------", "------", "------", "------", "------", "------", "-----", "-----", "-----");
    eprintln!("{:>4} {:>10.2} {:>10}  {:>+9.2}  {:>10.2} {:>10}  {:>+9.2}  {:>9.2} {:>9}  {:>+8.2}",
        "avg", sum_sy/n, "", sum_dy/n,
                sum_su/n, "", sum_du/n,
                sum_sv/n, "", sum_dv/n);

    eprintln!("\n=== §B-cascade-real V5 verdict ===");
    eprintln!("avg Y stego cost: {:>+.2} dB", sum_dy/n);
    eprintln!("avg U stego cost: {:>+.2} dB", sum_du/n);
    eprintln!("avg V stego cost: {:>+.2} dB", sum_dv/n);
    let max_chroma_cost = ((sum_du/n).abs()).max((sum_dv/n).abs());
    let luma_cost_abs = (sum_dy/n).abs();
    if max_chroma_cost > luma_cost_abs * 1.5 {
        eprintln!("⚠ Chroma damage > luma damage by >50% — flip distribution biased");
    } else {
        eprintln!("✓ Chroma damage proportional to luma (within ~1.5×)");
    }
}

/// **§B-cascade-real bisect (2026-05-06)** — byte-diff clean encoder vs
/// empty-message stego encoder at 480p × 1f IDR. Empty msg = 0 stego bits
/// = should produce byte-identical bitstream as clean. If bytes differ,
/// the orchestrator is diverging from the clean encoder pipeline at
/// the mode-decision/configuration level, not at flip injection.
#[test]
#[ignore]
fn phase_1_1_d_empty_stego_vs_clean_bytewise_480p() {
    use phasm_core::codec::h264::stego::encode_pixels::
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files;

    let yuv_path = "/tmp/iphone7_480x272_f12.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing fixture");
    let width = 480u32;
    let height = 272u32;
    let frame_size = (width * height * 3 / 2) as usize;
    let pattern = GopPattern::Ipppp { gop: 1 };

    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42"); }

    let stego_empty_bs =
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
            &yuv[..frame_size],
            width, height, 1, pattern,
            "",
            &[],
            "passprobe-bytewise",
        ).expect("stego encode (empty)");

    let mut clean_enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    clean_enc.entropy_mode = EntropyMode::Cabac;
    clean_enc.enable_transform_8x8 = true;
    let clean_bs = clean_enc.encode_i_frame(&yuv[..frame_size]).expect("clean encode");

    eprintln!("\n=== Byte diff: clean vs empty-message stego (IDR 480p) ===");
    eprintln!("clean bytes:       {}", clean_bs.len());
    eprintln!("stego_empty bytes: {}", stego_empty_bs.len());

    let same_size = clean_bs.len() == stego_empty_bs.len();
    let same_bytes = clean_bs == stego_empty_bs;
    eprintln!("identical size:    {}", same_size);
    eprintln!("identical bytes:   {}", same_bytes);

    if !same_bytes && same_size {
        let mut diff_count = 0;
        let mut first_diff = None;
        for (i, (a, b)) in clean_bs.iter().zip(stego_empty_bs.iter()).enumerate() {
            if a != b {
                if first_diff.is_none() { first_diff = Some(i); }
                diff_count += 1;
            }
        }
        let total = clean_bs.len();
        eprintln!("byte differences:  {} of {} bytes ({:.1}%)",
            diff_count, total, 100.0 * diff_count as f64 / total as f64);
        if let Some(off) = first_diff {
            eprintln!("first diff at:     byte offset {} (0x{:x})", off, off);
            // Show 32-byte windows around first diff
            let start = off.saturating_sub(8);
            let end = (off + 24).min(total);
            eprint!("clean[{:#06x}..{:#06x}]:  ", start, end);
            for b in &clean_bs[start..end] { eprint!("{:02x} ", b); }
            eprintln!();
            eprint!("stego[{:#06x}..{:#06x}]:  ", start, end);
            for b in &stego_empty_bs[start..end] { eprint!("{:02x} ", b); }
            eprintln!();
        }
    }
}

/// **§B-cascade-real per-domain ablation (2026-05-06)** — IDR-only stego cost
/// across each of the 4 stego domains in isolation. Uses
/// `PHASM_STEALTH_ABLATE=cs|cl|ms|ml` env knob to set 3 weights to 0,
/// exercising one domain at a time.
///
/// Hypothesis: one domain is the CABAC bin-context desync source.
/// Domain causing >2 dB IDR damage on its own = the desync source.
///
/// Domains:
/// - cs: coeff_sign_bypass    (sign of dequantized coefficients)
/// - cl: coeff_suffix_lsb     (magnitude LSB of coefficients with |level|>=15)
/// - ms: mvd_sign_bypass      (sign bit of MVD components)
/// - ml: mvd_suffix_lsb       (magnitude LSB of MVD with |val|>=9)
#[test]
#[ignore]
fn phase_1_1_d_idr_per_domain_ablation_480p() {
    use phasm_core::codec::h264::stego::encode_pixels::
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files;

    let yuv_path = "/tmp/iphone7_480x272_f12.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing fixture");
    let width = 480u32;
    let height = 272u32;
    let n_frames = 1usize;
    let frame_size = (width * height * 3 / 2) as usize;
    let pattern = GopPattern::Ipppp { gop: 1 };

    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42"); }

    let psnr_one = |bs: &[u8], label: &str| -> f64 {
        let h264 = std::env::temp_dir().join(format!("phasm_ablate_{label}.h264"));
        let dec = std::env::temp_dir().join(format!("phasm_ablate_{label}.dec.yuv"));
        std::fs::write(&h264, bs).unwrap();
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"]).arg(&h264)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(&dec)
            .status().expect("ffmpeg run");
        assert!(status.success());
        let decoded = std::fs::read(&dec).unwrap();
        let y_size = (width * height) as usize;
        let mut sum_sq: u64 = 0;
        for (a, b) in yuv[..y_size].iter().zip(decoded[..y_size].iter()) {
            let d = (*a as i32 - *b as i32) as i64;
            sum_sq += (d * d) as u64;
        }
        let mse = (sum_sq as f64) / (y_size as f64);
        if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
    };

    let mut clean_enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    clean_enc.entropy_mode = EntropyMode::Cabac;
    clean_enc.enable_transform_8x8 = true;
    let clean_bs = clean_enc.encode_i_frame(&yuv[..frame_size]).expect("clean encode");
    let clean_psnr = psnr_one(&clean_bs, "clean");

    eprintln!("\n=== §B-cascade-real per-domain ablation (IDR-only 480x272) ===");
    eprintln!("clean baseline: {:>6.2} dB", clean_psnr);
    eprintln!("\n{:<8} {:>10} {:>10} {:>10}", "ablate", "stego", "Δ vs clean", "label");

    for (knob, label) in [("", "all-4"), ("cs", "coeff_sign"), ("cl", "coeff_suffix"),
                          ("ms", "mvd_sign"), ("ml", "mvd_suffix")] {
        unsafe {
            if knob.is_empty() { std::env::remove_var("PHASM_STEALTH_ABLATE"); }
            else { std::env::set_var("PHASM_STEALTH_ABLATE", knob); }
        }
        let result =
            h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
                &yuv[..frame_size],
                width, height, n_frames, pattern,
                "ablate-bisect-480p",
                &[],
                "passprobe-ablate",
            );
        match result {
            Ok(stego_bs) => {
                let stego_psnr = psnr_one(&stego_bs, label);
                eprintln!("{:<8} {:>10.2} {:>+10.2} dB  {}",
                    knob, stego_psnr, stego_psnr - clean_psnr, label);
            }
            Err(e) => {
                eprintln!("{:<8} ERROR: {}  {}", knob, e, label);
            }
        }
    }
    unsafe { std::env::remove_var("PHASM_STEALTH_ABLATE"); }
}

/// **§B-cascade-real bisect (2026-05-06)** — IDR-only single-frame stego cost
/// at 1080p. Resolution-scaling counterpart to `phase_1_1_d_idr_only_stego_480p`.
///
/// Same QP=26, same single IDR. If 1080p damage is significantly smaller
/// than 480p (~-8 dB), then the 480p damage is dominated by per-position
/// cost spread over fewer total positions. If similar damage, it's a
/// universal stego flip pattern issue.
#[test]
#[ignore]
fn phase_1_1_d_idr_only_stego_1080p() {
    use phasm_core::codec::h264::stego::encode_pixels::
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files;

    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path)
        .expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 1usize;
    let frame_size = (width * height * 3 / 2) as usize;

    let pattern = GopPattern::Ipppp { gop: 1 };
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42"); }

    eprintln!("\n=== IDR-only stego cost — 1920x1072 single frame ===");
    let t0 = std::time::Instant::now();
    let stego_bs =
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
            &yuv[..frame_size],
            width, height, n_frames, pattern,
            "idr-only-bisect-1080p",
            &[],
            "passprobe-idr-only",
        )
        .expect("stego encode");
    let t_stego = t0.elapsed();
    eprintln!("stego encode: {:?}, {} bytes", t_stego, stego_bs.len());

    let stego_empty_bs =
        h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
            &yuv[..frame_size],
            width, height, n_frames, pattern,
            "",
            &[],
            "passprobe-idr-only",
        )
        .expect("stego encode (empty)");
    eprintln!("stego encode (msg=''): {} bytes", stego_empty_bs.len());

    let mut clean_enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    clean_enc.entropy_mode = EntropyMode::Cabac;
    clean_enc.enable_transform_8x8 = true;
    let frame = &yuv[..frame_size];
    let clean_bs = clean_enc.encode_i_frame(frame).expect("clean IDR encode");
    eprintln!("clean encode: {} bytes", clean_bs.len());

    let psnr_each = |bs: &[u8], label: &str| -> f64 {
        let h264 = std::env::temp_dir().join(format!("phasm_idr_only_1080p_{label}.h264"));
        let dec = std::env::temp_dir().join(format!("phasm_idr_only_1080p_{label}.dec.yuv"));
        std::fs::write(&h264, bs).unwrap();
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(&h264)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&dec)
            .status()
            .expect("ffmpeg run");
        assert!(status.success(), "ffmpeg decode failed for {label}");
        let decoded = std::fs::read(&dec).unwrap();
        let y_size = (width * height) as usize;
        let src_y = &yuv[..y_size];
        let dec_y = &decoded[..y_size];
        let mut sum_sq: u64 = 0;
        for (a, b) in src_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32) as i64;
            sum_sq += (d * d) as u64;
        }
        let mse = (sum_sq as f64) / (y_size as f64);
        if mse == 0.0 { 100.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
    };

    let stego_psnr = psnr_each(&stego_bs, "stego");
    let stego_empty_psnr = psnr_each(&stego_empty_bs, "stego_empty");
    let clean_psnr = psnr_each(&clean_bs, "clean");

    eprintln!("\n=== IDR-only Y-PSNR (1080p) ===");
    eprintln!("stego (msg='idr-only-bisect-1080p'): {:>6.2} dB  Δ={:+6.2} dB",
        stego_psnr, stego_psnr - clean_psnr);
    eprintln!("stego (msg=''):                      {:>6.2} dB  Δ={:+6.2} dB",
        stego_empty_psnr, stego_empty_psnr - clean_psnr);
    eprintln!("clean:                               {:>6.2} dB", clean_psnr);
}

/// **§B-direct-fix Stage 2 (#232) — IPPPP-vs-IBPBP disambiguation** (2026-05-06).
///
/// `dump_b_frame_recon_vs_decode` shows P4 in IBPBP (encode order I0, P2,
/// B1, P4, ...) has 47% pixel divergence at mb=(0,0) between encoder.recon
/// and ffmpeg.decode, while P2 (right after IDR) is 0% diverged. The
/// difference between P2 and P4: P4 is the encoder's SECOND P-frame, and
/// there is an intervening B-frame.
///
/// This test isolates which condition triggers the bug by encoding I0 → P
/// → P (3 frames, no B between) and reading the second P-frame's recon.
///
/// Outcomes:
/// - If P (encode_idx=2) STILL has 47% divergence at mb=(0,0)  → bug is
///   P-frame-after-P intrinsic, not B-frame-induced. Investigate P-frame
///   state mutation (qp_grid, ref state, prev_mb_qp, etc.).
/// - If P (encode_idx=2) is CLEAN  → bug is triggered by B-frame
///   encoding mutating shared state that P4 then reads. Investigate
///   what `encode_b_frame` mutates that `encode_p_frame`'s reset block
///   doesn't reset.
#[test]
#[ignore]
fn ippp_vs_ibpbp_disambiguation() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 3usize; // I0 P1 P2 (encode order = display order in IPPPP)
    let frame_size = (width * height * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }

    let pattern = GopPattern::Ipppp { gop: 30 };
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = false;

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
        .unwrap_or_else(|e| {
            panic!("encode (encode_idx={}, display={}): {e}", eo.encode_idx, eo.display_idx)
        });
        bitstream.extend_from_slice(&bytes);
        recon_dumps.push((eo.display_idx, eo.frame_type, enc.recon.y.clone()));
    }

    let h264_path = std::env::temp_dir().join("phasm_ippp_disambig.h264");
    let dec_path = std::env::temp_dir().join("phasm_ippp_disambig.dec.yuv");
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
    let y_size = (width * height) as usize;

    eprintln!("\n=== IPPPP disambiguation: encoder.recon vs ffmpeg.decode (Y plane) ===");
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    for (display_idx, ft, enc_y) in &recon_dumps {
        let off = (*display_idx as usize) * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut diff = 0u64;
        let mut max_abs = 0u32;
        let mut nz_pixels = 0u32;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            diff += d as u64;
            if d > 0 {
                nz_pixels += 1;
                if d > max_abs {
                    max_abs = d;
                }
            }
        }
        let avg = (diff as f64) / (y_size as f64);
        eprintln!(
            "  display={:>2}  type={:?}  avg|Δ|={:.3}  max|Δ|={}  nonzero_pixels={} ({:.2}%)",
            display_idx,
            ft,
            avg,
            max_abs,
            nz_pixels,
            100.0 * (nz_pixels as f64) / (y_size as f64)
        );
        // mb=(0,0) sum-of-abs-diff (the IBPBP P4 hotspot).
        let mut mb00_sum = 0u32;
        for dy in 0..16 {
            for dx in 0..16 {
                let row = dy;
                let col = dx;
                let idx = row * (width as usize) + col;
                let d = (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
                mb00_sum += d;
            }
        }
        eprintln!("    mb=(0,0) Σ|Δ| = {}", mb00_sum);
        // First diverging MB in raster order.
        let mut first_div: Option<(usize, usize, u32)> = None;
        'outer: for mby in 0..mb_h {
            for mbx in 0..mb_w {
                let mut s = 0u32;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let row = mby * 16 + dy;
                        let col = mbx * 16 + dx;
                        let idx = row * (width as usize) + col;
                        s += (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
                    }
                }
                if s > 0 {
                    first_div = Some((mbx, mby, s));
                    break 'outer;
                }
            }
        }
        if let Some((mbx, mby, s)) = first_div {
            eprintln!("    first diverging MB: mb=({},{})  Σ|Δ|={}", mbx, mby, s);
        } else {
            eprintln!("    NO diverging MBs (clean frame)");
        }
    }

    // Verdict: read the encoder's 2nd P-frame line. If it has any nonzero
    // pixels at mb=(0,0), the bug fires in IPPPP too → P-frame intrinsic.
    let p2 = recon_dumps
        .iter()
        .filter(|(_, ft, _)| matches!(ft, FrameType::P))
        .nth(1);
    if let Some((display_idx, _, enc_y)) = p2 {
        let off = (*display_idx as usize) * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut mb00_sum = 0u32;
        for dy in 0..16 {
            for dx in 0..16 {
                let idx = dy * (width as usize) + dx;
                mb00_sum += (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
            }
        }
        eprintln!(
            "\n*** VERDICT: 2nd P-frame mb=(0,0) Σ|Δ| = {} ({}) ***",
            mb00_sum,
            if mb00_sum == 0 {
                "CLEAN — bug is B-frame-induced; investigate encode_b_frame mutations"
            } else {
                "DIRTY — bug is P-frame intrinsic; investigate P-after-P state"
            }
        );
    }
}

/// **§B-direct-fix Stage 2 (#232) — truncated IBPBP-skip d=10 vs encoder.**
///
/// Tests if the trailing B at d=9, B at d=11 specifically corrupt
/// ffmpeg's d=10 output. Encodes 12f IBPBP-skip, strips trailing B's
/// from bitstream, decodes via ffmpeg, compares ffmpeg.decode[d=10]
/// to encoder.visual_recon[d=10].
///
/// If clean → trailing B's are the trigger (ffmpeg DPB/reorder quirk).
/// If dirty → bug is in PRECEDING B-frames (4 B's before P at d=10).
#[test]
#[ignore]
fn truncated_d10_vs_encoder() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 12usize;
    let frame_size = (width * height * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::set_var("PHASM_B_FORCE_MODE", "skip");
    }

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;

    let mut bitstream = Vec::new();
    let mut p_d10_visual_recon: Option<Vec<u8>> = None;
    let mut frame_byte_offsets: Vec<(u32, FrameType, usize, usize)> = Vec::new();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes_before = bitstream.len();
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame),
            FrameType::P => enc.encode_p_frame(frame),
            FrameType::B => enc.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| {
            panic!("encode (encode_idx={}, display={}): {e}", eo.encode_idx, eo.display_idx)
        });
        bitstream.extend_from_slice(&bytes);
        let bytes_after = bitstream.len();
        frame_byte_offsets.push((eo.display_idx, eo.frame_type, bytes_before, bytes_after));
        if eo.display_idx == 10 && matches!(eo.frame_type, FrameType::P) {
            p_d10_visual_recon = Some(enc.visual_recon.y.clone());
        }
    }
    unsafe {
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }

    let p_d10_recon = p_d10_visual_recon.expect("d=10 P");

    // Find last frame that is P at d=10 — truncate bitstream there.
    let p_d10_offset = frame_byte_offsets
        .iter()
        .find(|(d, ft, _, _)| *d == 10 && matches!(ft, FrameType::P))
        .map(|(_, _, _, end)| *end)
        .expect("P at d=10 in stream");
    let truncated_bitstream = &bitstream[..p_d10_offset];

    eprintln!("truncated bitstream: {} bytes (was {} full)", truncated_bitstream.len(), bitstream.len());

    // Decode truncated via ffmpeg.
    let h264_path = std::env::temp_dir().join("phasm_trunc_d10_test.h264");
    let dec_path = std::env::temp_dir().join("phasm_trunc_d10_test.dec.yuv");
    std::fs::write(&h264_path, truncated_bitstream).expect("write");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .status()
        .expect("ffmpeg run");
    assert!(status.success(), "ffmpeg decode failed");

    let decoded = std::fs::read(&dec_path).expect("read dec");
    let n_decoded = decoded.len() / frame_size;
    eprintln!("truncated decoded: {} frames", n_decoded);

    // ffmpeg outputs frames in display order. Truncated has display
    // 0,1,2,3,4,5,6,7,8,10 (NO B at d=9 or d=11 since they got
    // truncated). d=10 is at frame_idx 9 in output.
    let y_size = (width * height) as usize;
    let d10_off = 9 * frame_size; // frame_idx=9 = d=10 in truncated
    if d10_off + y_size > decoded.len() {
        eprintln!("not enough frames decoded ({} < {})", decoded.len(), d10_off + y_size);
        return;
    }
    let dec_d10 = &decoded[d10_off..d10_off + y_size];
    let mut diff = 0u64;
    let mut nz = 0u32;
    let mut max_abs = 0u32;
    for (a, b) in p_d10_recon.iter().zip(dec_d10.iter()) {
        let d = (*a as i32 - *b as i32).unsigned_abs();
        diff += d as u64;
        if d > 0 {
            nz += 1;
            if d > max_abs { max_abs = d; }
        }
    }
    eprintln!(
        "\n*** truncated ffmpeg.decode[d=10] vs encoder.visual_recon[d=10]: avg|Δ|={:.3}, max|Δ|={}, nonzero {} ({:.2}%) ***",
        (diff as f64) / (y_size as f64),
        max_abs,
        nz,
        100.0 * (nz as f64) / (y_size as f64),
    );
}

/// **§B-direct-fix Stage 2 (#232) — bytewise diff P at d=10 slice between
/// IBPBP-no-B-call (clean) and IBPBP-12f-skip (broken).**
///
/// Definitive question: does encoder produce different P at d=10 bytes
/// in these two scenarios (= encoder state mutation we haven't found),
/// or same bytes that ffmpeg interprets differently (= ffmpeg quirk)?
#[test]
#[ignore]
fn p_d10_byte_diff_ibpbp_skip_vs_no_b_call() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
    }

    // ── Scenario A: IBPBP-no-B-call (clean d=10) ──────────────────
    unsafe {
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }
    let mut enc_a = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc_a.entropy_mode = EntropyMode::Cabac;
    enc_a.enable_transform_8x8 = true;
    enc_a.enable_b_frames = true;
    let displays_a = [0usize, 2, 4, 6, 8, 10];
    let mut bytes_per_frame_a: Vec<Vec<u8>> = Vec::new();
    for (i, &d) in displays_a.iter().enumerate() {
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = if i == 0 {
            enc_a.encode_i_frame(frame).expect("idr A")
        } else {
            enc_a.encode_p_frame(frame).expect("p A")
        };
        bytes_per_frame_a.push(bytes);
    }
    let p_d10_a = bytes_per_frame_a.last().unwrap().clone();

    // ── Scenario B: IBPBP-12f with B-FORCE-MODE=skip (broken d=10) ─
    unsafe {
        std::env::set_var("PHASM_B_FORCE_MODE", "skip");
    }
    let mut enc_b = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc_b.entropy_mode = EntropyMode::Cabac;
    enc_b.enable_transform_8x8 = true;
    enc_b.enable_b_frames = true;
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    let mut bytes_per_frame_b: Vec<(u32, FrameType, Vec<u8>)> = Vec::new();
    for eo in iter_encode_order(12, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc_b.encode_i_frame(frame),
            FrameType::P => enc_b.encode_p_frame(frame),
            FrameType::B => enc_b.encode_b_frame(frame),
        }
        .unwrap_or_else(|e| panic!("encode B (idx={}, d={}): {e}", eo.encode_idx, eo.display_idx));
        bytes_per_frame_b.push((eo.display_idx, eo.frame_type, bytes));
    }
    unsafe {
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }
    let p_d10_b = bytes_per_frame_b
        .iter()
        .find(|(d, ft, _)| *d == 10 && matches!(ft, FrameType::P))
        .expect("P at d=10 in scenario B")
        .2
        .clone();

    eprintln!("\n=== P at d=10 slice byte comparison ===");
    eprintln!("scenario A (no B calls):  {} bytes", p_d10_a.len());
    eprintln!("scenario B (B-skip):      {} bytes", p_d10_b.len());

    if p_d10_a == p_d10_b {
        eprintln!("\n*** BYTE-IDENTICAL → ffmpeg interpretation differs based on prior B-slice parsing ***");
    } else {
        eprintln!("\n*** BYTES DIFFER → encoder state at start of P at d=10 differs between scenarios ***");
        // Find first diverging byte.
        let mut first_diff = None;
        for (i, (a, b)) in p_d10_a.iter().zip(p_d10_b.iter()).enumerate() {
            if a != b {
                first_diff = Some(i);
                break;
            }
        }
        if let Some(i) = first_diff {
            eprintln!("first diverging byte index: {} (out of {} / {})",
                i, p_d10_a.len(), p_d10_b.len());
            // Dump 32 bytes around the divergence.
            let start = i.saturating_sub(16);
            let end_a = (i + 16).min(p_d10_a.len());
            let end_b = (i + 16).min(p_d10_b.len());
            eprint!("  A[{:04}..]: ", start);
            for byte in &p_d10_a[start..end_a] {
                eprint!("{:02x} ", byte);
            }
            eprintln!();
            eprint!("  B[{:04}..]: ", start);
            for byte in &p_d10_b[start..end_b] {
                eprint!("{:02x} ", byte);
            }
            eprintln!();
        } else {
            // One is a prefix of the other.
            eprintln!("one is a prefix of the other (one shorter)");
        }
    }
    // Also dump prefix lengths for context.
    eprintln!("\nFrame sizes per scenario:");
    eprintln!("  A: {:?}", bytes_per_frame_a.iter().map(|b| b.len()).collect::<Vec<_>>());
    eprintln!("  B: {:?}", bytes_per_frame_b.iter().map(|(d, ft, b)| (*d, *ft, b.len())).collect::<Vec<_>>());
}

/// **§B-direct-fix Stage 2 (#232) — enable_b_frames=true but skip encode_b_frame.**
///
/// IBPBP-skip d=10 still breaks; pure IPPPP {0,2,4,6,8,10} is clean.
/// What's different: enable_b_frames flag (toggles SPS pic_order_cnt_type +
/// max_num_ref_frames + slice header poc_lsb), AND actual encode_b_frame
/// calls between Ps.
///
/// This test: enable_b_frames=true (so SPS differs from IPPPP), but
/// SKIP the actual encode_b_frame calls — only encode P at d=0,2,4,6,8,10.
///
/// If d=10 BREAKS → bug is in SPS / P-slice header differences when
/// enable_b_frames=true (poc_lsb, max_num_ref_frames=3, etc.).
/// If d=10 CLEAN → bug is in encode_b_frame's state mutations even
/// when its slice content is trivial (all-Skip).
#[test]
#[ignore]
fn ibpbp_enable_no_b_call_d10_check() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }

    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true; // ← KEY: SPS config differs from IPPPP

    // Encode I0 at d=0 + P at d=2,4,6,8,10. NO encode_b_frame calls.
    let displays = [0usize, 2, 4, 6, 8, 10];
    let mut bitstream = Vec::new();
    let mut recon_dumps: Vec<(usize, Vec<u8>)> = Vec::new();
    for (i, &d) in displays.iter().enumerate() {
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = if i == 0 {
            enc.encode_i_frame(frame).expect("idr")
        } else {
            enc.encode_p_frame(frame).expect("p")
        };
        bitstream.extend_from_slice(&bytes);
        recon_dumps.push((d, enc.visual_recon.y.clone()));
    }

    let h264_path = std::env::temp_dir().join("phasm_ibpbp_enable_nob.h264");
    let dec_path = std::env::temp_dir().join("phasm_ibpbp_enable_nob.dec.yuv");
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
    let y_size = (width * height) as usize;

    eprintln!("\n=== enable_b_frames=true, no encode_b_frame, P at d=0,2,4,6,8,10 ===");
    for (i, (display_idx, enc_y)) in recon_dumps.iter().enumerate() {
        let off = i * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut diff = 0u64;
        let mut nz = 0u32;
        let mut max_abs = 0u32;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            diff += d as u64;
            if d > 0 {
                nz += 1;
                if d > max_abs { max_abs = d; }
            }
        }
        eprintln!(
            "  encode_idx={} display={:>2}  avg|Δ|={:.3}  max|Δ|={}  nonzero={} ({:.2}%)",
            i,
            display_idx,
            (diff as f64) / (y_size as f64),
            max_abs,
            nz,
            100.0 * (nz as f64) / (y_size as f64)
        );
    }
    let last = recon_dumps.last().unwrap();
    let last_off = (recon_dumps.len() - 1) * frame_size;
    let dec_last = &decoded[last_off..last_off + y_size];
    let mut nz = 0u32;
    for (a, b) in last.1.iter().zip(dec_last.iter()) {
        if (*a as i32 - *b as i32) != 0 { nz += 1; }
    }
    let pct = 100.0 * (nz as f64) / (y_size as f64);
    eprintln!(
        "\n*** VERDICT: d=10 P with enable_b_frames=true (no B calls) {:.2}% diverge → {} ***",
        pct,
        if pct < 1.0 {
            "CLEAN — encode_b_frame call mutates state that breaks d=10"
        } else {
            "DIRTY — enable_b_frames=true SPS/slice config alone breaks d=10"
        }
    );
}

/// **§B-direct-fix Stage 2 (#232) — IBPBP 12-frame with B forced to skip.**
///
/// 12-frame IBPBP env test shows d=10 P explodes 45.91% diverge.
/// Pure IPPPP {0,2,4,6,8,10} is clean. So B-frames cause the d=10
/// explosion. This test isolates: with B forced to all-Skip (no
/// residual emission, no self.recon writes from B-MBs), does d=10 P
/// still explode?
///
/// If d=10 CLEAN here → B-frame's L0/L1/Bi residual writes to
/// self.recon contaminate state P10 reads from.
/// If d=10 DIRTY here → B-frame state mutations OUTSIDE residual
/// writes (mv_grid populate, neighbor commit, frame_num/POC
/// counters) cause the cascade.
#[test]
#[ignore]
fn ibpbp_12f_b_force_skip_d10_check() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 12usize;
    let frame_size = (width * height * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::set_var("PHASM_B_FORCE_MODE", "skip");
    }

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
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
        .unwrap_or_else(|e| {
            panic!("encode (encode_idx={}, display={}): {e}", eo.encode_idx, eo.display_idx)
        });
        bitstream.extend_from_slice(&bytes);
        recon_dumps.push((eo.display_idx, eo.frame_type, enc.visual_recon.y.clone()));
    }
    unsafe {
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }

    let h264_path = std::env::temp_dir().join("phasm_ibpbp_12f_skip.h264");
    let dec_path = std::env::temp_dir().join("phasm_ibpbp_12f_skip.dec.yuv");
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
    let y_size = (width * height) as usize;

    eprintln!("\n=== IBPBP 12f B-force-skip: enc.visual_recon vs ffmpeg.decode ===");
    let mut sorted: Vec<_> = recon_dumps.iter().collect();
    sorted.sort_by_key(|(d, _, _)| *d);
    for (display_idx, ft, enc_y) in sorted {
        let off = (*display_idx as usize) * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut diff = 0u64;
        let mut nz_pixels = 0u32;
        let mut max_abs = 0u32;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            diff += d as u64;
            if d > 0 {
                nz_pixels += 1;
                if d > max_abs {
                    max_abs = d;
                }
            }
        }
        eprintln!(
            "  display={:>2}  type={:?}  avg|Δ|={:.3}  max|Δ|={}  nonzero_pixels={} ({:.2}%)",
            display_idx,
            ft,
            (diff as f64) / (y_size as f64),
            max_abs,
            nz_pixels,
            100.0 * (nz_pixels as f64) / (y_size as f64)
        );
    }

    // Find d=10 P specifically.
    let d10 = recon_dumps.iter().find(|(d, ft, _)| *d == 10 && matches!(ft, FrameType::P));
    if let Some((_, _, enc_y)) = d10 {
        let off = 10 * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut nz = 0u32;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            if (*a as i32 - *b as i32) != 0 {
                nz += 1;
            }
        }
        let pct = 100.0 * (nz as f64) / (y_size as f64);
        eprintln!(
            "\n*** VERDICT: d=10 P with B-force-skip {:.2}% diverge ({}) ***",
            pct,
            if pct < 1.0 {
                "CLEAN — B residual writes contaminate; d=10 broken by B's L0/L1/Bi self.recon writes"
            } else {
                "DIRTY — even with B all-Skip, d=10 P breaks; bug in B's mv_grid/counter mutations"
            }
        );
    }
}

/// **§B-direct-fix Stage 2 (#232) — IPPPP {0,2,4,6,8,10} bisect of d=10 onset.**
///
/// 12-frame IBPBP env test shows P at d=2..8 clean, d=10 P explodes 46%
/// (visual_recon vs ffmpeg). This test asks: does the explosion happen
/// in pure IPPPP at the same displays (no B-frame between), or does
/// it require B-frames?
///
/// If d=10 P CLEAN here → B-frames cause the explosion.
/// If d=10 P DIRTY here → P-state itself has issue with 5+ P-frames.
#[test]
#[ignore]
fn ipppp_d0_to_d10_bisect() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }

    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = false;

    let displays = [0usize, 2, 4, 6, 8, 10];
    let mut bitstream = Vec::new();
    let mut recon_dumps: Vec<(usize, Vec<u8>)> = Vec::new();
    for (i, &d) in displays.iter().enumerate() {
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = if i == 0 {
            enc.encode_i_frame(frame).expect("idr")
        } else {
            enc.encode_p_frame(frame).expect("p")
        };
        bitstream.extend_from_slice(&bytes);
        recon_dumps.push((d, enc.visual_recon.y.clone()));
    }

    let h264_path = std::env::temp_dir().join("phasm_ipppp_d0_d10.h264");
    let dec_path = std::env::temp_dir().join("phasm_ipppp_d0_d10.dec.yuv");
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
    let y_size = (width * height) as usize;

    eprintln!("\n=== IPPPP {{d=0,2,4,6,8,10}} encoder.visual_recon vs ffmpeg.decode ===");
    for (i, (display_idx, enc_y)) in recon_dumps.iter().enumerate() {
        // ffmpeg outputs frames in order — slot i corresponds to our i'th frame.
        let off = i * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut diff = 0u64;
        let mut max_abs = 0u32;
        let mut nz_pixels = 0u32;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            diff += d as u64;
            if d > 0 {
                nz_pixels += 1;
                if d > max_abs {
                    max_abs = d;
                }
            }
        }
        let mut mb00 = 0u32;
        for dy in 0..16 {
            for dx in 0..16 {
                let idx = dy * (width as usize) + dx;
                mb00 += (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
            }
        }
        eprintln!(
            "  encode_idx={} display={:>2}  avg|Δ|={:.3}  max|Δ|={}  nonzero={} ({:.2}%)  mb=(0,0) Σ|Δ|={}",
            i,
            display_idx,
            (diff as f64) / (y_size as f64),
            max_abs,
            nz_pixels,
            100.0 * (nz_pixels as f64) / (y_size as f64),
            mb00,
        );
    }

    let last = recon_dumps.last().unwrap();
    let last_off = (recon_dumps.len() - 1) * frame_size;
    let dec_last = &decoded[last_off..last_off + y_size];
    let mut nz = 0u32;
    for (a, b) in last.1.iter().zip(dec_last.iter()) {
        if (*a as i32 - *b as i32) != 0 {
            nz += 1;
        }
    }
    eprintln!(
        "\n*** VERDICT: pure IPPPP P at d={} {} {}",
        last.0,
        if nz == 0 { "CLEAN" } else { "DIRTY" },
        if nz == 0 {
            "→ d=10 explosion is B-frame-induced, not P-frame intrinsic"
        } else {
            "→ d=10 explosion is P-frame intrinsic at 5th P-frame after IDR"
        }
    );
}

/// **§B-direct-fix Stage 2 (#232) — IPPPP-{d=0,d=2,d=4} pixel diff.**
///
/// Confirms that the 47% divergence at the second P-frame (display=4)
/// happens in pure IPPPP encoding (no B-frame at all). If true, this
/// closes out the "B-frame mutates P state" hypothesis definitively
/// and pivots the investigation to encoder MC / residual paths.
///
/// Encodes I0(d=0), P(d=2), P(d=4) with enable_b_frames=false,
/// decodes via ffmpeg, computes pixel divergence per frame.
#[test]
#[ignore]
fn ipppp_d0_d2_d4_pixel_diff() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }

    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = false;

    // Manually feed display 0, 2, 4 — same source frames as IBPBP's
    // I0/P2/P4 anchors but no B-frame at all.
    let displays = [0usize, 2, 4];
    let mut bitstream = Vec::new();
    let mut recon_dumps: Vec<(usize, Vec<u8>)> = Vec::new();
    for (i, &d) in displays.iter().enumerate() {
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = if i == 0 {
            enc.encode_i_frame(frame).expect("idr")
        } else {
            enc.encode_p_frame(frame).expect("p")
        };
        bitstream.extend_from_slice(&bytes);
        recon_dumps.push((d, enc.recon.y.clone()));
    }

    let h264_path = std::env::temp_dir().join("phasm_ipppp_024.h264");
    let dec_path = std::env::temp_dir().join("phasm_ipppp_024.dec.yuv");
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
    let y_size = (width * height) as usize;

    eprintln!("\n=== IPPPP {{d=0, d=2, d=4}} encoder.recon vs ffmpeg.decode ===");
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    for (i, (display_idx, enc_y)) in recon_dumps.iter().enumerate() {
        // ffmpeg outputs decoded frames in DISPLAY order. Our bitstream
        // has 3 frames at displays 0, 2, 4. ffmpeg will renumber them
        // to 0, 1, 2 in its output (since they come in order). So
        // decoded[i] corresponds to enc.recon at our i'th encode.
        let off = i * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut diff = 0u64;
        let mut nz_pixels = 0u32;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            diff += d as u64;
            if d > 0 {
                nz_pixels += 1;
            }
        }
        // mb=(0,0) Σ|Δ|
        let mut mb00 = 0u32;
        for dy in 0..16 {
            for dx in 0..16 {
                let idx = dy * (width as usize) + dx;
                mb00 += (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
            }
        }
        eprintln!(
            "  encode_idx={} display={}  avg|Δ|={:.3}  nonzero={} ({:.2}%)  mb=(0,0) Σ|Δ|={}",
            i,
            display_idx,
            (diff as f64) / (y_size as f64),
            nz_pixels,
            100.0 * (nz_pixels as f64) / (y_size as f64),
            mb00,
        );
        // First diverging MB in raster order (for context).
        if (nz_pixels as f64 / y_size as f64) > 0.10 {
            'outer: for mby in 0..mb_h {
                for mbx in 0..mb_w {
                    let mut s = 0u32;
                    for dy in 0..16 {
                        for dx in 0..16 {
                            let row = mby * 16 + dy;
                            let col = mbx * 16 + dx;
                            let idx = row * (width as usize) + col;
                            s += (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
                        }
                    }
                    if s > 0 {
                        eprintln!("    first diverging MB: mb=({},{})  Σ|Δ|={}", mbx, mby, s);
                        break 'outer;
                    }
                }
            }
        }
    }
}

/// **§B-direct-fix Stage 2 (#232) — CABAC trace diff at second P-frame.**
///
/// Encodes a 2-P sequence in BOTH IPPPP and IBPBP-force-skip scenarios,
/// captures CABAC bin trace ONLY for the second P-frame, then prints
/// the first 30 lines of each trace side-by-side. First divergence
/// between the two = the bug at mb=(0,0).
///
/// IPPPP: I0, P1, P2 (3 frames). Trace P2.
/// IBPBP-skip: I0, P2, B1, P4 (4 frames; B1 forced all-Skip). Trace P4.
///
/// Both should hit mb=(0,0) of a P-slice with same source pixel data
/// (display=2 for IPPPP, display=4 for IBPBP) so the prediction MV
/// will differ but ctx/range/low evolution from slice start should be
/// identical IF the encoder state is clean.
#[test]
#[ignore]
fn cabac_trace_diff_second_p() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let frame_size = (width * height * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
    }

    // ── 1. IPPPP control: I0(d=0), P(d=2), P(d=4) ─ same source
    //    frames as IBPBP's P-anchors, just no B in between ────────
    let mut enc_ippp = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc_ippp.entropy_mode = EntropyMode::Cabac;
    enc_ippp.enable_transform_8x8 = true;
    enc_ippp.enable_b_frames = false;

    unsafe {
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }

    // Manually feed display=0, 2, 4 (skip 1 and 3) so the IPPPP path
    // matches IBPBP's P-anchor source frames. iter_encode_order can't
    // express this — its IPPPP iterates 0,1,2,...
    let _ = enc_ippp
        .encode_i_frame(&yuv[0 * frame_size..1 * frame_size])
        .expect("idr enc");
    let _ = enc_ippp
        .encode_p_frame(&yuv[2 * frame_size..3 * frame_size])
        .expect("p1 enc (d=2)");

    enc_ippp.enable_cabac_trace();
    let _ = enc_ippp
        .encode_p_frame(&yuv[4 * frame_size..5 * frame_size])
        .expect("p2 enc (d=4)");
    let trace_ippp = enc_ippp.take_cabac_trace();

    // ── 2. IBPBP B-skip: I0, P2, B1 (forced skip), P4 ────────────
    unsafe {
        std::env::set_var("PHASM_B_FORCE_MODE", "skip");
    }
    let mut enc_ib = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc_ib.entropy_mode = EntropyMode::Cabac;
    enc_ib.enable_transform_8x8 = true;
    enc_ib.enable_b_frames = true;
    let pattern_ib = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    let mut eo_iter = iter_encode_order(5, pattern_ib);
    // encode_idx=0 IDR (display=0)
    let eo = eo_iter.next().expect("idr");
    enc_ib.encode_i_frame(&yuv[(eo.display_idx as usize) * frame_size..(eo.display_idx as usize + 1) * frame_size]).expect("idr");
    // encode_idx=1 P (display=2)
    let eo = eo_iter.next().expect("p2");
    enc_ib.encode_p_frame(&yuv[(eo.display_idx as usize) * frame_size..(eo.display_idx as usize + 1) * frame_size]).expect("p2");
    // encode_idx=2 B (display=1)
    let eo = eo_iter.next().expect("b1");
    enc_ib.encode_b_frame(&yuv[(eo.display_idx as usize) * frame_size..(eo.display_idx as usize + 1) * frame_size]).expect("b1");

    enc_ib.enable_cabac_trace();
    // encode_idx=3 P (display=4) — this is the buggy one
    let eo = eo_iter.next().expect("p4");
    enc_ib.encode_p_frame(&yuv[(eo.display_idx as usize) * frame_size..(eo.display_idx as usize + 1) * frame_size]).expect("p4");
    let trace_ibpbp = enc_ib.take_cabac_trace();

    unsafe {
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }

    // ── 3. Diff: print first 30 bins of each + first divergence ──
    eprintln!("\n=== CABAC trace diff: 2nd P-frame mb=(0,0) bins ===");
    eprintln!("IPPPP trace lines: {}", trace_ippp.len());
    eprintln!("IBPBP-skip trace lines: {}", trace_ibpbp.len());

    let n = std::cmp::min(trace_ippp.len(), trace_ibpbp.len());
    let mut first_div: Option<usize> = None;
    for i in 0..n {
        if trace_ippp[i] != trace_ibpbp[i] {
            first_div = Some(i);
            break;
        }
    }

    eprintln!("\n--- First 30 bins (IPPPP control) ---");
    for line in trace_ippp.iter().take(30) {
        eprintln!("  {}", line);
    }
    eprintln!("\n--- First 30 bins (IBPBP force-skip) ---");
    for line in trace_ibpbp.iter().take(30) {
        eprintln!("  {}", line);
    }

    if let Some(i) = first_div {
        eprintln!("\n*** FIRST DIVERGENCE at bin index {} ***", i);
        eprintln!("  IPPPP:      {}", trace_ippp[i]);
        eprintln!("  IBPBP-skip: {}", trace_ibpbp[i]);
        // Show some context around the divergence
        eprintln!("\n--- 5 bins before divergence ---");
        let start = i.saturating_sub(5);
        for j in start..i {
            eprintln!("  [{:>4}] IPPPP:      {}", j, trace_ippp[j]);
            eprintln!("  [{:>4}] IBPBP-skip: {}", j, trace_ibpbp[j]);
        }
    } else {
        eprintln!("\n*** NO DIVERGENCE in first {} bins ***", n);
    }
}

/// **§B-direct-fix Stage 2 (#232) — IBPBP with B-frame forced to all-Skip.**
///
/// Same fixture + setup as `dump_b_frame_recon_vs_decode` but with
/// `PHASM_B_FORCE_MODE=skip` so every B-MB takes the trivial Skip path
/// (no residual emission, no L0/L1/Bi mode-decision, no joint bipred ME).
/// Reads encoder's P4 mb=(0,0) recon vs ffmpeg.decode.
///
/// Outcomes:
/// - P4 mb=(0,0) STILL diverges  → bug lives in slice header / setup
///   path that Skip cannot avoid (POC, frame_num, deblock, mv_grid
///   commit). Look for mutations SHARED with non-skip path.
/// - P4 mb=(0,0) CLEAN  → bug specifically in B's L0/L1/Bi residual
///   emission path (likely in `write_b_inter_residual_macroblock_cabac`
///   or `populate_b_direct_grid` when residual_enabled).
#[test]
#[ignore]
fn ibpbp_b_force_skip_disambiguation() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 6usize;
    let frame_size = (width * height * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::set_var("PHASM_B_FORCE_MODE", "skip");
    }

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
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
        .unwrap_or_else(|e| {
            panic!("encode (encode_idx={}, display={}): {e}", eo.encode_idx, eo.display_idx)
        });
        bitstream.extend_from_slice(&bytes);
        recon_dumps.push((eo.display_idx, eo.frame_type, enc.recon.y.clone()));
    }
    unsafe {
        std::env::remove_var("PHASM_B_FORCE_MODE");
    }

    let h264_path = std::env::temp_dir().join("phasm_b_force_skip.h264");
    let dec_path = std::env::temp_dir().join("phasm_b_force_skip.dec.yuv");
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
    let y_size = (width * height) as usize;

    eprintln!("\n=== B-force-Skip: encoder.recon vs ffmpeg.decode ===");
    let mut sorted: Vec<_> = recon_dumps.iter().collect();
    sorted.sort_by_key(|(d, _, _)| *d);
    for (display_idx, ft, enc_y) in sorted {
        let off = (*display_idx as usize) * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut diff = 0u64;
        let mut nz_pixels = 0u32;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            diff += d as u64;
            if d > 0 {
                nz_pixels += 1;
            }
        }
        let mut mb00_sum = 0u32;
        for dy in 0..16 {
            for dx in 0..16 {
                let idx = dy * (width as usize) + dx;
                mb00_sum += (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
            }
        }
        eprintln!(
            "  display={:>2}  type={:?}  avg|Δ|={:.3}  nonzero={} ({:.2}%)  mb=(0,0) Σ|Δ|={}",
            display_idx,
            ft,
            (diff as f64) / (y_size as f64),
            nz_pixels,
            100.0 * (nz_pixels as f64) / (y_size as f64),
            mb00_sum,
        );
    }
}

/// **§B-encoder-decoder-divergence Phase 2.6 — per-MB diff classification**
/// (2026-05-08).
///
/// Phase 2.5's static audit found phasm's `derive_b_direct_spatial_with_col`
/// matches the spec § 8.4.1.2.2 derivation byte-for-byte (modulo the
/// inactive clause-2 colZeroFlag gap). Yet B-frame `enc.visual_recon`
/// diverges from a spec-compliant decoder at 1080p (max|Δ|=158 on
/// display=5, 1.60% pixels per `dump_b_frame_recon_vs_decode_env`
/// 2026-05-08 run). Phase 3 (#245) Direct+residual retry was a 3×
/// regression with no static-analysis smoking gun.
///
/// Phase 2.6 dumps the 4 worst-diverging 16×16 luma MBs per B-frame and
/// classifies the diff pattern into one of:
///
/// - **uniform_shift_16x16**: same Δ across the whole MB → MV (pred surface)
///   off by N pixels in same direction → pred-derivation bug (inter MV or
///   `derive_b_direct_*` logic).
/// - **4x4_block_aligned**: diffs concentrate inside individual 4×4 sub-
///   blocks with sharp boundaries between them → 4×4 IT/quant residual
///   bug or a coefficient-index miscount (e.g., wrong scan order).
/// - **8x8_block_aligned**: 8×8 grid → 8×8 transform path or 8×8 quant
///   scale bug.
/// - **edge_only**: rows/cols 0/15 only → deblock filter bug.
/// - **scattered**: no obvious structure → likely CABAC desync / wrong
///   residual coefficients decoded from phasm's bitstream by a
///   spec-compliant decoder.
///
/// Output guides the next session: 4x4_block_aligned → bisect residual
/// emit + CABAC residual encode; uniform_shift → bisect MV-derivation;
/// scattered → CABAC ctx state divergence.
#[test]
#[ignore]
fn phase_2_6_b_mb_diff_classify() {
    let yuv_path = "/tmp/iphone7_1920x1072_f60.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/iphone7_1920x1072_f60.yuv");
    let width = 1920u32;
    let height = 1072u32;
    let n_frames = 12usize;
    let frame_size = (width * height * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
        std::env::remove_var("PHASM_B_FORCE_MODE");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
    }

    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::PRODUCTION_VISUAL;

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
        .unwrap_or_else(|e| panic!(
            "encode (encode_idx={}, display={}): {e}", eo.encode_idx, eo.display_idx
        ));
        bitstream.extend_from_slice(&bytes);
        recon_dumps.push((eo.display_idx, eo.frame_type, enc.visual_recon.y.clone()));
    }

    let h264_path = std::env::temp_dir().join("phasm_phase26.h264");
    let dec_path = std::env::temp_dir().join("phasm_phase26.dec.yuv");
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
    let y_size = (width * height) as usize;
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;

    let mut sorted: Vec<_> = recon_dumps.iter().collect();
    sorted.sort_by_key(|(d, _, _)| *d);

    let classify = |enc_y: &[u8], dec_y: &[u8], mbx: usize, mby: usize| -> (String, [[i32; 16]; 16]) {
        let mut diff = [[0i32; 16]; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                let idx = (mby * 16 + dy) * (width as usize) + mbx * 16 + dx;
                diff[dy][dx] = enc_y[idx] as i32 - dec_y[idx] as i32;
            }
        }
        let nonzero: i32 = diff.iter().flatten().filter(|&&d| d != 0).count() as i32;
        if nonzero == 0 {
            return ("ZERO".into(), diff);
        }

        // ── Tier 0: uniform shift (whole MB shifted by N).
        let nz_vals: Vec<i32> = diff.iter().flatten().copied().filter(|&d| d != 0).collect();
        let first_nz = nz_vals[0];
        if nz_vals.iter().all(|&d| d == first_nz) && nonzero >= 200 {
            return (format!("uniform_shift_16x16 (Δ={})", first_nz), diff);
        }

        // ── Tier 1: 16×8 split (top half vs bottom half).
        let mut top_sum: i64 = 0; let mut top_cnt = 0;
        let mut bot_sum: i64 = 0; let mut bot_cnt = 0;
        for dy in 0..16 {
            for dx in 0..16 {
                if diff[dy][dx] != 0 {
                    if dy < 8 { top_sum += diff[dy][dx] as i64; top_cnt += 1; }
                    else { bot_sum += diff[dy][dx] as i64; bot_cnt += 1; }
                }
            }
        }
        let top_avg = if top_cnt > 0 { top_sum / top_cnt as i64 } else { 0 };
        let bot_avg = if bot_cnt > 0 { bot_sum / bot_cnt as i64 } else { 0 };
        let top_dom = top_cnt >= 8 && bot_cnt <= 16;
        let bot_dom = bot_cnt >= 8 && top_cnt <= 16;
        if top_dom && !bot_dom {
            return (format!("split_16x8_top_only (top_avg={top_avg})"), diff);
        }
        if bot_dom && !top_dom {
            return (format!("split_16x8_bot_only (bot_avg={bot_avg})"), diff);
        }

        // ── Tier 1b: 8×16 split (left vs right).
        let mut l_sum: i64 = 0; let mut l_cnt = 0;
        let mut r_sum: i64 = 0; let mut r_cnt = 0;
        for dy in 0..16 {
            for dx in 0..16 {
                if diff[dy][dx] != 0 {
                    if dx < 8 { l_sum += diff[dy][dx] as i64; l_cnt += 1; }
                    else { r_sum += diff[dy][dx] as i64; r_cnt += 1; }
                }
            }
        }
        let l_dom = l_cnt >= 8 && r_cnt <= 16;
        let r_dom = r_cnt >= 8 && l_cnt <= 16;
        if l_dom && !r_dom {
            return (format!("split_8x16_left_only (l_avg={})", l_sum / l_cnt.max(1) as i64), diff);
        }
        if r_dom && !l_dom {
            return (format!("split_8x16_right_only (r_avg={})", r_sum / r_cnt.max(1) as i64), diff);
        }

        // ── Tier 2: edge-only (deblock).
        let mut edge = 0; let mut interior = 0;
        for dy in 0..16 {
            for dx in 0..16 {
                if diff[dy][dx] != 0 {
                    if dy == 0 || dy == 15 || dx == 0 || dx == 15 { edge += 1; }
                    else { interior += 1; }
                }
            }
        }
        if interior == 0 && edge > 0 {
            return ("edge_only (deblock?)".into(), diff);
        }

        // ── Tier 3: smooth gradient (MV-offset signature).
        // Compute per-row average diff. If row averages monotonically
        // shift across the MB AND row-by-row variance is small relative
        // to row-to-row delta, it's a smooth gradient → MV/ref bug.
        let mut row_avg = [0i32; 16];
        let mut row_cnt = [0i32; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                if diff[dy][dx] != 0 { row_avg[dy] += diff[dy][dx]; row_cnt[dy] += 1; }
            }
        }
        for dy in 0..16 {
            if row_cnt[dy] > 0 { row_avg[dy] /= row_cnt[dy]; }
        }
        let mut col_avg = [0i32; 16];
        let mut col_cnt = [0i32; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                if diff[dy][dx] != 0 { col_avg[dx] += diff[dy][dx]; col_cnt[dx] += 1; }
            }
        }
        for dx in 0..16 {
            if col_cnt[dx] > 0 { col_avg[dx] /= col_cnt[dx]; }
        }
        let row_min = row_avg.iter().min().copied().unwrap_or(0);
        let row_max = row_avg.iter().max().copied().unwrap_or(0);
        let col_min = col_avg.iter().min().copied().unwrap_or(0);
        let col_max = col_avg.iter().max().copied().unwrap_or(0);
        let row_range = row_max - row_min;
        let col_range = col_max - col_min;
        if (row_range >= 30 || col_range >= 30) && nonzero >= 100 {
            let dir = if col_range > row_range * 2 {
                "horizontal"
            } else if row_range > col_range * 2 {
                "vertical"
            } else {
                "diagonal"
            };
            return (
                format!("smooth_gradient_{dir} (row Δ {row_min}..{row_max}, col Δ {col_min}..{col_max})"),
                diff,
            );
        }

        // ── Tier 4: 4×4 block-aligned residual.
        let mut sb4_homogeneous = 0; let mut sb4_mixed = 0;
        for sby in 0..4 {
            for sbx in 0..4 {
                let mut nz_in_sb = 0;
                for ddy in 0..4 {
                    for ddx in 0..4 {
                        if diff[sby * 4 + ddy][sbx * 4 + ddx] != 0 { nz_in_sb += 1; }
                    }
                }
                if nz_in_sb == 0 || nz_in_sb == 16 { sb4_homogeneous += 1; }
                else { sb4_mixed += 1; }
            }
        }
        let mut sb8_homogeneous = 0; let mut sb8_mixed = 0;
        for sby in 0..2 {
            for sbx in 0..2 {
                let mut nz = 0;
                for ddy in 0..8 {
                    for ddx in 0..8 {
                        if diff[sby * 8 + ddy][sbx * 8 + ddx] != 0 { nz += 1; }
                    }
                }
                if nz == 0 || nz == 64 { sb8_homogeneous += 1; }
                else { sb8_mixed += 1; }
            }
        }
        let label = if sb4_homogeneous >= 12 && sb4_mixed <= 4 {
            "4x4_block_aligned (residual coeff bug)"
        } else if sb8_homogeneous >= 3 && sb8_mixed <= 1 {
            "8x8_block_aligned (8x8 transform bug)"
        } else {
            "scattered (CABAC desync? coeff parse mismatch?)"
        };
        (label.into(), diff)
    };

    eprintln!("\n=== Phase 2.6 — top-4 worst diverging B-MBs per B-frame ===");
    let mut frame_summary: Vec<(u32, FrameType, u32, u32, Vec<(usize, usize, u32, String)>)> = Vec::new();
    for (display_idx, ft, enc_y) in &sorted {
        if *ft != FrameType::B {
            continue;
        }
        let off = (*display_idx as usize) * frame_size;
        let dec_y = &decoded[off..off + y_size];
        let mut diff_total = 0u64;
        let mut nz_pixels = 0u32;
        for (a, b) in enc_y.iter().zip(dec_y.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            diff_total += d as u64;
            if d > 0 { nz_pixels += 1; }
        }
        if nz_pixels == 0 {
            continue;
        }

        // Score every MB and keep top 4.
        let mut mb_scores: Vec<(usize, usize, u32)> = Vec::new();
        for mby in 0..mb_h {
            for mbx in 0..mb_w {
                let mut s = 0u32;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = (mby * 16 + dy) * (width as usize) + mbx * 16 + dx;
                        s += (enc_y[idx] as i32 - dec_y[idx] as i32).unsigned_abs();
                    }
                }
                if s > 0 {
                    mb_scores.push((mbx, mby, s));
                }
            }
        }
        mb_scores.sort_by_key(|(_, _, s)| std::cmp::Reverse(*s));
        let top4: Vec<_> = mb_scores.iter().take(4).cloned().collect();

        eprintln!(
            "\n--- display={}  type=B  Σ|Δ|={}  nz_pixels={} ({:.3}%) ---",
            display_idx, diff_total, nz_pixels,
            100.0 * (nz_pixels as f64) / (y_size as f64),
        );

        let mut classifications: Vec<(usize, usize, u32, String)> = Vec::new();
        for (mbx, mby, s) in &top4 {
            let (label, diff) = classify(enc_y, dec_y, *mbx, *mby);
            classifications.push((*mbx, *mby, *s, label.clone()));
            // Compact per-MB report
            let mut max_abs = 0i32;
            for row in &diff {
                for &v in row {
                    if v.abs() > max_abs { max_abs = v.abs(); }
                }
            }
            eprintln!(
                "  MB=({:>3},{:>3})  Σ|Δ|={:>5}  max|Δ|={:>3}  → {}",
                mbx, mby, s, max_abs, label,
            );
        }

        // For the single worst MB on the worst-overall B-frame, dump the
        // full 16×16 diff grid.
        if !top4.is_empty() && nz_pixels > 5_000 {
            let (mbx, mby, _) = top4[0];
            let (_, diff) = classify(enc_y, dec_y, mbx, mby);
            eprintln!("    full diff grid for MB=({},{}):", mbx, mby);
            for row in &diff {
                eprint!("      ");
                for &v in row {
                    if v == 0 {
                        eprint!("  . ");
                    } else {
                        eprint!("{:>3} ", v);
                    }
                }
                eprintln!();
            }
        }

        frame_summary.push((*display_idx, *ft, nz_pixels, diff_total as u32, classifications));
    }

    eprintln!("\n=== Phase 2.6 summary (B-frames only) ===");
    let mut class_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for (_, _, _, _, classes) in &frame_summary {
        for (_, _, _, label) in classes {
            // Strip parenthetical detail for aggregate counting.
            let key = label.split(" (").next().unwrap_or(label).to_string();
            *class_counts.entry(key).or_insert(0) += 1;
        }
    }
    let mut counts: Vec<_> = class_counts.iter().collect();
    counts.sort_by_key(|&(_, n)| std::cmp::Reverse(*n));
    eprintln!("Diff-pattern counts across {} top-4 B-MBs:", frame_summary.len() * 4);
    for (label, n) in counts {
        eprintln!("  {:>4}  {}", n, label);
    }
    eprintln!("\nDominant pattern indicates the bug class:");
    eprintln!("  uniform_shift_16x16   → MV/pred-derivation bug (Phase 2.5 was wrong)");
    eprintln!("  4x4_block_aligned     → residual coeff emit/parse bug → CABAC residual code path");
    eprintln!("  8x8_block_aligned     → 8x8 transform bug");
    eprintln!("  edge_only             → deblock filter bug");
    eprintln!("  scattered             → CABAC desync / mb_type or CBP miscoded");
}
