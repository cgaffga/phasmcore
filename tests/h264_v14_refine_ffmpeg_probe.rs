// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// v1.4 Phase 4.5 (#316) DIAGNOSTIC PROBE — fast ffmpeg-validated
// fixture exercising ref_idx=1 emission via refine_p firing.
//
// Fixture: 32×32 (2×2 MB) IPP under DUAL_REF_L0. pattern_a as frame 0
// IDR; pattern_b (luma+13) as frame 1 P; pattern_a (original) as
// frame 2 P. On frame 2, refine_p should pick ref_idx=1 (= pattern_a
// = I0 = past_anchor) over ref_idx=0 (= pattern_b = last_ref).
//
// The test runs ffmpeg on the encoded bitstream and reports first
// stderr line containing "Reference" or "error". Use as a fast
// iteration loop while bisecting the structural ref_idx=1 bug.
//
// Run: cargo test --features h264-encoder,cabac-stego \
//        --test h264_v14_refine_ffmpeg_probe \
//        -- --nocapture --ignored

#![cfg(all(feature = "h264-encoder", feature = "cabac-stego"))]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode, MultiRefConfig};

fn make_pattern_a(w: u32, h: u32) -> Vec<u8> {
    // y_size + 2 chroma planes (yuv420p).
    let y_size = (w * h) as usize;
    let c_size = (w / 2 * h / 2) as usize;
    let mut buf = vec![0u8; y_size + 2 * c_size];
    // Luma: gradient. Chroma: 128.
    for y in 0..h {
        for x in 0..w {
            buf[(y * w + x) as usize] = ((x.wrapping_add(y * 7)) & 0xFF) as u8;
        }
    }
    for v in &mut buf[y_size..] {
        *v = 128;
    }
    buf
}

fn make_pattern_b(a: &[u8], y_size: usize) -> Vec<u8> {
    let mut b = a.to_vec();
    // Shift luma by +13 → ref_0 (P=B) is worse match for source
    // (=A, frame 2) than ref_1 (I=A) would be. refine_p should
    // pick ref_idx=1.
    for v in &mut b[..y_size] {
        *v = v.wrapping_add(13);
    }
    b
}

fn run_probe(label: &str, w: u32, h: u32, with_b: bool) -> Vec<String> {
    let pattern_a = make_pattern_a(w, h);
    let pattern_b = make_pattern_b(&pattern_a, (w * h) as usize);

    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    enc.multi_ref_config = MultiRefConfig::DUAL_REF_L0;
    enc.enable_b_frames = with_b;

    let mut all = Vec::new();
    if with_b {
        // IBPBP encode order: I0 (display 0), P0 (display 2),
        // B0 (display 1), P1 (display 4), B1 (display 3).
        // We pass display-order frames and let encoder reorder.
        // 5 frames = first B has past_anchor=I, pre_past_anchor=None.
        // Second B has past_anchor=P0, pre_past_anchor=I → refine
        // can fire.
        let pa = pattern_a.clone();
        let pb = pattern_b.clone();
        all.extend_from_slice(&enc.encode_i_frame(&pa).unwrap()); // disp 0 = I
        all.extend_from_slice(&enc.encode_p_frame(&pb).unwrap()); // disp 2 = P
        all.extend_from_slice(&enc.encode_b_frame(&pa).unwrap()); // disp 1 = B
        all.extend_from_slice(&enc.encode_p_frame(&pa).unwrap()); // disp 4 = P
        all.extend_from_slice(&enc.encode_b_frame(&pb).unwrap()); // disp 3 = B
    } else {
        all.extend_from_slice(&enc.encode_i_frame(&pattern_a).unwrap());
        all.extend_from_slice(&enc.encode_p_frame(&pattern_b).unwrap());
        all.extend_from_slice(&enc.encode_p_frame(&pattern_a).unwrap());
    }

    eprintln!("[probe-{}] bitstream {} bytes", label, all.len());

    let h264_path = std::env::temp_dir().join(format!("phasm_v14_refine_probe_{}.h264", label));
    let yuv_path = std::env::temp_dir().join(format!("phasm_v14_refine_probe_{}.yuv", label));
    std::fs::write(&h264_path, &all).unwrap();

    // Run ffmpeg decode and capture stderr.
    let output = std::process::Command::new("ffmpeg")
        .args(["-y", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .output()
        .expect("ffmpeg subprocess");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut error_lines = Vec::new();
    for line in stderr.lines() {
        if line.contains("Reference") || line.contains("error while decoding")
            || line.contains("Missing reference")
        {
            error_lines.push(line.to_string());
        }
    }

    if error_lines.is_empty() {
        eprintln!("[probe-{}] ffmpeg decode CLEAN", label);
    } else {
        eprintln!("[probe-{}] ffmpeg ERRORS ({}):", label, error_lines.len());
        for l in &error_lines {
            eprintln!("  {}", l);
        }
    }

    let _ = std::fs::remove_file(&h264_path);
    let _ = std::fs::remove_file(&yuv_path);
    error_lines
}

// All probes require PHASM_V14_REFINE_ON=1 to actually exercise
// refine; without it they all pass trivially (refine no-op, ref_idx=0
// only). Run with: PHASM_V14_REFINE_ON=1 cargo test --features ...
//                  --test h264_v14_refine_ffmpeg_probe -- --ignored

#[test]
#[ignore]
fn v14_refine_probe_ipp_32x32() {
    let errs = run_probe("ipp_32x32", 32, 32, false);
    assert!(errs.is_empty(), "ffmpeg errors: {:?}", errs);
}

#[test]
#[ignore]
fn v14_refine_probe_ibpbp_32x32() {
    let errs = run_probe("ibpbp_32x32", 32, 32, true);
    assert!(errs.is_empty(), "ffmpeg errors: {:?}", errs);
}

#[test]
#[ignore]
fn v14_refine_probe_ipp_128x96() {
    // Larger fixture (8x6 MBs) — bug fires here at MB (1,1) when
    // PHASM_V14_REFINE_ON=1. Use this to bisect ref_idx=1 emission
    // CABAC drift.
    let errs = run_probe("ipp_128x96", 128, 96, false);
    assert!(errs.is_empty(), "ffmpeg errors: {:?}", errs);
}

#[test]
#[ignore]
fn v14_refine_probe_ibpbp_128x96() {
    let errs = run_probe("ibpbp_128x96", 128, 96, true);
    assert!(errs.is_empty(), "ffmpeg errors: {:?}", errs);
}

/// v1.4 Phase 4.5 follow-up — encoder/decoder recon parity probe.
///
/// Encodes IPP at 128x96 with refine ON, then compares phasm's
/// `enc.visual_recon` (= what encoder thinks the decoded picture is)
/// against ffmpeg's decoded YUV (= what spec-conforming decoder
/// reconstructs). If max|Δ| > 0 → encoder/decoder disagree → cliff
/// source confirmed at the recon layer. Reports per-MB max|Δ| so
/// we can see WHICH MBs diverge.
#[test]
#[ignore]
fn v14_refine_recon_parity_128x96_ipp() {
    use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode, MultiRefConfig};
    let w = 128u32;
    let h = 96u32;
    let pattern_a = make_pattern_a(w, h);
    let pattern_b = make_pattern_b(&pattern_a, (w * h) as usize);

    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    enc.multi_ref_config = MultiRefConfig::DUAL_REF_L0;

    let mut all = Vec::new();
    all.extend_from_slice(&enc.encode_i_frame(&pattern_a).unwrap());
    all.extend_from_slice(&enc.encode_p_frame(&pattern_b).unwrap());
    all.extend_from_slice(&enc.encode_p_frame(&pattern_a).unwrap());

    // Snapshot phasm's visual_recon AFTER frame 2 (the second P-frame
    // = where refine fired).
    let y_size = (w * h) as usize;
    let c_size = ((w / 2) * (h / 2)) as usize;
    let phasm_y = enc.visual_recon.y.clone();
    let phasm_cb = enc.visual_recon.cb.clone();
    let phasm_cr = enc.visual_recon.cr.clone();
    assert_eq!(phasm_y.len(), y_size);
    assert_eq!(phasm_cb.len(), c_size);

    // Decode through ffmpeg to YUV.
    let h264_path = std::env::temp_dir().join("phasm_v14_recon_parity_ipp.h264");
    let yuv_path = std::env::temp_dir().join("phasm_v14_recon_parity_ipp.yuv");
    std::fs::write(&h264_path, &all).unwrap();
    let st = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("ffmpeg");
    assert!(st.success(), "ffmpeg decode failed");
    let ffmpeg_yuv = std::fs::read(&yuv_path).expect("read yuv");
    let frame_size = y_size + 2 * c_size;
    assert_eq!(ffmpeg_yuv.len(), 3 * frame_size, "expect 3 frames in YUV");

    // Frame 2 starts at offset 2 * frame_size.
    let off = 2 * frame_size;
    let ff_y = &ffmpeg_yuv[off..off + y_size];
    let ff_cb = &ffmpeg_yuv[off + y_size..off + y_size + c_size];
    let ff_cr = &ffmpeg_yuv[off + y_size + c_size..off + y_size + 2 * c_size];

    // Per-MB max|Δ| on luma.
    let mb_w = (w / 16) as usize;
    let mb_h = (h / 16) as usize;
    let mut total_max_dev = 0i32;
    let mut total_diverge = 0usize;
    let mut bad_mbs: Vec<(usize, usize, i32)> = Vec::new();
    for mby in 0..mb_h {
        for mbx in 0..mb_w {
            let mut mb_max = 0i32;
            for dy in 0..16 {
                for dx in 0..16 {
                    let py = mby * 16 + dy;
                    let px = mbx * 16 + dx;
                    let i = py * (w as usize) + px;
                    let d = phasm_y[i] as i32 - ff_y[i] as i32;
                    let abs = d.abs();
                    if abs > mb_max {
                        mb_max = abs;
                    }
                }
            }
            if mb_max > 0 {
                total_diverge += 1;
                bad_mbs.push((mbx, mby, mb_max));
            }
            if mb_max > total_max_dev {
                total_max_dev = mb_max;
            }
        }
    }

    eprintln!("=== Recon parity probe (frame 2 = second P) ===");
    eprintln!("  total MBs: {}, diverge: {}, max|Δ|_luma: {}",
              mb_w * mb_h, total_diverge, total_max_dev);
    eprintln!("  diverging MBs (mb_x, mb_y, max|Δ|):");
    for (i, &(x, y, m)) in bad_mbs.iter().enumerate().take(20) {
        eprintln!("    [{}] ({}, {}) max={}", i, x, y, m);
    }
    if bad_mbs.len() > 20 {
        eprintln!("    ... and {} more", bad_mbs.len() - 20);
    }

    // Dump pixel-level diff for the first divergent MB.
    if let Some(&(mbx, mby, _)) = bad_mbs.first() {
        eprintln!("=== Per-pixel dump for MB ({},{}) ===", mbx, mby);
        eprintln!("  phasm visual_recon (left) | ffmpeg decode (right) | Δ (= phasm - ffmpeg):");
        for dy in 0..16 {
            let py = mby * 16 + dy;
            let mut phasm_row = String::new();
            let mut ff_row = String::new();
            let mut delta_row = String::new();
            for dx in 0..16 {
                let px = mbx * 16 + dx;
                let idx = py * (w as usize) + px;
                phasm_row.push_str(&format!("{:3} ", phasm_y[idx]));
                ff_row.push_str(&format!("{:3} ", ff_y[idx]));
                delta_row.push_str(&format!("{:+4} ", phasm_y[idx] as i32 - ff_y[idx] as i32));
            }
            eprintln!("  row{:2}: P=[{}] F=[{}]", dy, phasm_row.trim(), ff_row.trim());
            eprintln!("         Δ=[{}]", delta_row.trim());
        }
        // Source pattern for this MB (frame 2 input = pattern_a).
        eprintln!("  source pattern_a for MB ({},{}) [for reference]:", mbx, mby);
        for dy in 0..16 {
            let py = mby * 16 + dy;
            let mut row = String::new();
            for dx in 0..16 {
                let px = mbx * 16 + dx;
                row.push_str(&format!("{:3} ",
                    pattern_a[py * (w as usize) + px]));
            }
            eprintln!("  row{:2}: S=[{}]", dy, row.trim());
        }
    }

    // Chroma divergence summary.
    let mut c_max = 0i32;
    let mut c_diverge = 0usize;
    for i in 0..c_size {
        let dcb = (phasm_cb[i] as i32 - ff_cb[i] as i32).abs();
        let dcr = (phasm_cr[i] as i32 - ff_cr[i] as i32).abs();
        if dcb > 0 || dcr > 0 {
            c_diverge += 1;
        }
        c_max = c_max.max(dcb).max(dcr);
    }
    eprintln!("  chroma: diverge_pixels={}, max|Δ|_chroma={}", c_diverge, c_max);

    // Preserve the bitstream for external analysis.
    eprintln!("  bitstream preserved at: {}", h264_path.display());

    // Don't assert — we want to see the data first. Test always passes
    // (the goal is the diagnostic output).
    eprintln!("=== END recon parity probe ===");
}
