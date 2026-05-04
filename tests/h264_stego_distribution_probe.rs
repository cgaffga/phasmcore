// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §6E-A6.5 calibration probe (#156) — measure records-per-MB
// coefficients for each forced B-MB mode under
// `scripts/h264_mb_partition_histogram.py`.
//
// The histogram counts unique `(src_x, src_y)` anchors per frame
// (one per partition top-left in pixel coords). The records-per-MB
// coefficient depends on the mb_type's partition shape AND on
// libavcodec's spatial-direct derivation behaviour for B_Direct/Skip.
//
// **1080p empirical results** (this fixture):
//
//   force_mode             | rec/MB | shape distribution
//   --------------------------------------------------------
//   skip                   | 1.000  | 100% 16x16
//   direct_16x16           | 1.749  | 98.8% 16x16, 1.0% 8x8 (leakage)
//   l0_16x16               | 1.000  | 100% 16x16
//   l1_16x16               | 1.000  | 100% 16x16
//   bi_16x16               | 1.000  | 100% 16x16
//   partitioned_4 (16x8 L0/L0)  | 2.000 | 100% 16x8 ✓
//   partitioned_5 (8x16 L0/L0)  | 2.000 | 100% 8x16 ✓
//   partitioned_6 (16x8 L1/L1)  | 2.000 | 100% 16x8 ✓
//   partitioned_7 (8x16 L1/L1)  | 2.000 | 100% 8x16 ✓
//   partitioned_12 (Bi-mixed)   | 0.716 | 39.6% 16x16 + 30.4% 8x8 + ...
//   partitioned_15 (Bi-mixed)   | 0.718 | 39.9% 16x16 + 30.2% 8x8 + ...
//   b_8x8_uniform_l0/l1         | 4.000 | 100% 8x8 ✓
//
// **Key finding**: Pure-direction Partitioned (mb_type 4/5/6/7) and
// B_8x8 (sub=1/2) have stable record-per-MB coefficients. Bi-mixed
// Partitioned (mb_type 12+) breaks at 1080p — libavcodec's MV
// exporter produces a mixed shape distribution that doesn't match
// the spec partition shape. AVOID Bi-mixed Partitioned for any
// shape calibration.
//
// **Caveat for #156 calibration**: real-world (mixed mb_type) gate
// runs do NOT obey these single-mode coefficients. Empirical evidence
// (3 calibration variants tested 2026-05-03) shows that introducing
// Partitioned/B_8x8 modes into a Skip+Direct-dominant mix triggers
// libavcodec's spatial-direct cascade analysis on neighbouring MBs,
// producing 2-3× more non-16x16 records than the linear coefficient
// model predicts. Single-coefficient calibration via bucket math
// fundamentally doesn't work — converging requires either ffmpeg
// instrumentation or iterative empirical bucket-search.
//
// `#[ignore]` because this needs `/tmp/img4138_1080p_f10.yuv` +
// python3 + PyAV + ~10 min of force-mode encodes.

#![cfg(feature = "cabac-stego")]

use phasm_core::{
    h264_stego_encode_yuv_string_4domain_multigop_with_pattern,
    GopPattern,
};

const FORCE_MODES: &[&str] = &[
    "skip",
    "direct_16x16",
    "l0_16x16",
    "l1_16x16",
    "bi_16x16",
    "partitioned_4",  // 16x8 L0/L0 — clean 2.0 r/MB, 100% 16x8
    "partitioned_5",  // 8x16 L0/L0 — clean 2.0 r/MB, 100% 8x16
    "partitioned_6",  // 16x8 L1/L1 — clean 2.0 r/MB, 100% 16x8
    "partitioned_7",  // 8x16 L1/L1 — clean 2.0 r/MB, 100% 8x16
    "partitioned_12", // 16x8 L0/Bi — broken at 1080p, AVOID
    "partitioned_15", // 8x16 L1/Bi — broken at 1080p, AVOID
    "b_8x8_uniform_direct",  // sub=0 — spatial-direct per sub-MB
    "b_8x8_uniform_l0",
    "b_8x8_uniform_l1",
    "b_8x8_uniform_bi",      // sub=3 — explicit Bi per sub-MB
    "b_8x8_mixed",           // sub=0..3 different per sub-MB
];

#[test]
#[ignore = "calibration probe; needs /tmp/img4138_128x80_f10.yuv + python3+PyAV"]
fn probe_force_mode_record_coefficients() {
    // Use 1080p fixture to match the gate's measurement context;
    // 128x80 fixture coefficients shift relative to 1080p due to
    // libavcodec's spatial-direct + content-dependent MV emission.
    let yuv = match std::fs::read("/tmp/img4138_1080p_f10.yuv") {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: /tmp/img4138_1080p_f10.yuv missing ({e})");
            return;
        }
    };

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent()
        .expect("manifest dir has parent");
    let script = workspace_root.join("scripts/h264_mb_partition_histogram.py");
    if !script.exists() {
        panic!("missing histogram script: {script:?}");
    }

    eprintln!("\n{:<22} | {:>7} | {:>10} | {:>10} | {}",
        "force_mode", "B-MBs", "B-records", "rec/MB", "B partition shape distribution");
    eprintln!("{}", "-".repeat(96));

    for &mode in FORCE_MODES {
        // SAFETY: lock-serialized.
        unsafe { std::env::set_var("PHASM_B_FORCE_MODE", mode); }
        let result = std::panic::catch_unwind(|| {
            let pattern = GopPattern::Ibpbp { gop: 5, b_count: 1 };
            let stego = h264_stego_encode_yuv_string_4domain_multigop_with_pattern(
                &yuv, 1920, 1072, 10, pattern, "x", "probe",
            )?;
            let path = std::env::temp_dir().join(format!("phasm_probe_{mode}.h264"));
            std::fs::write(&path, &stego).map_err(|e| {
                phasm_core::stego::StegoError::InvalidVideo(format!("write: {e}"))
            })?;
            let out = std::process::Command::new("python3")
                .arg(&script)
                .arg(&path)
                .output()
                .map_err(|e| phasm_core::stego::StegoError::InvalidVideo(format!("py: {e}")))?;
            let _ = std::fs::remove_file(&path);
            if !out.status.success() {
                return Err(phasm_core::stego::StegoError::InvalidVideo(format!(
                    "histogram script: {}", String::from_utf8_lossy(&out.stderr),
                )));
            }
            Ok(String::from_utf8_lossy(&out.stdout).into_owned())
        });
        unsafe { std::env::remove_var("PHASM_B_FORCE_MODE"); }
        let hist_text = match result {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                eprintln!("{:<22} | (encode failed: {e})", mode);
                continue;
            }
            Err(_) => {
                eprintln!("{:<22} | (panicked)", mode);
                continue;
            }
        };

        // Parse B-frame block.
        let (mbs, records, shape_summary) = parse_b_block(&hist_text);
        let rec_per_mb = if mbs > 0 { records as f64 / mbs as f64 } else { 0.0 };
        eprintln!("{:<22} | {:>7} | {:>10} | {:>10.3} | {}",
            mode, mbs, records, rec_per_mb, shape_summary);
    }
}

fn parse_b_block(text: &str) -> (u64, u64, String) {
    let mut in_b = false;
    let mut mbs = 0u64;
    let mut records = 0u64;
    let mut shapes: Vec<(String, f64)> = Vec::new();
    let mut directions: Vec<(String, f64)> = Vec::new();
    let mut in_direction = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("## B-frame") {
            in_b = true;
            in_direction = false;
            if let Some(open) = line.find('(') {
                if let Some(close) = line[open..].find(')') {
                    let inner = &line[open + 1..open + close];
                    for part in inner.split(',') {
                        let part = part.trim();
                        if let Some(n) = part.strip_suffix(" MBs") {
                            mbs = n.trim().parse().unwrap_or(0);
                        }
                    }
                }
            }
        } else if trimmed.starts_with("## ") {
            in_b = false;
            in_direction = false;
        } else if in_b {
            if trimmed.starts_with("partition shapes (over ") {
                let rest = &trimmed["partition shapes (over ".len()..];
                if let Some(end) = rest.find(' ') {
                    records = rest[..end].parse().unwrap_or(0);
                }
                in_direction = false;
            } else if trimmed.starts_with("direction (over") {
                in_direction = true;
            } else if let Some((label, _val, pct)) = parse_shape_or_dir_line(trimmed) {
                if !in_direction
                    && (label == "16x16" || label == "8x8" || label == "16x8"
                        || label == "8x16" || label == "8x4" || label == "4x8"
                        || label == "4x4")
                {
                    shapes.push((label, pct));
                } else if in_direction
                    && (label == "L0" || label == "L1" || label == "Bi")
                {
                    directions.push((label, pct));
                }
            }
        }
    }
    let mut summary = shapes.iter()
        .map(|(l, p)| format!("{l}={p:.1}%"))
        .collect::<Vec<_>>()
        .join(" ");
    if !directions.is_empty() {
        summary.push_str(" |");
        for (l, p) in &directions {
            summary.push_str(&format!(" {l}={p:.1}%"));
        }
    }
    (mbs, records, summary)
}

fn parse_shape_or_dir_line(line: &str) -> Option<(String, u64, f64)> {
    // "16x16 :    55548 ( 98.74%)" or "L0 :   26458 ( 47.03%)"
    let colon = line.find(':')?;
    let label = line[..colon].trim().to_string();
    let rest = &line[colon + 1..];
    let open = rest.find('(')?;
    let val: u64 = rest[..open].trim().parse().ok()?;
    let pct_str = rest[open + 1..].trim_start();
    let pct_end = pct_str.find('%')?;
    let pct: f64 = pct_str[..pct_end].trim().parse().ok()?;
    Some((label, val, pct))
}
