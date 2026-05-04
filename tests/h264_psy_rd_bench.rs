// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// PSY-RD prototype perf+stealth measurement spike. Encodes 1080p × 10f
// IBPBP fixture twice (PSY off vs on); reports wall-clock + writes
// outputs for histogram comparison.
//
// Run with:
//   cargo test --release --features cabac-stego --test h264_psy_rd_bench -- --ignored --nocapture

#![cfg(feature = "cabac-stego")]

use phasm_core::{
    h264_stego_encode_yuv_string_4domain_multigop_with_pattern, GopPattern,
};
use std::time::Instant;

fn run(label: &str, psy_on: bool) -> (std::time::Duration, Vec<u8>) {
    let yuv_path = "/tmp/img4138_1080p_f10.yuv";
    let yuv = std::fs::read(yuv_path).expect("missing /tmp/img4138_1080p_f10.yuv");

    // SAFETY: single-threaded test harness; no concurrent env access.
    unsafe {
        if psy_on {
            std::env::set_var("PHASM_PSY_RD", "1");
        } else {
            std::env::remove_var("PHASM_PSY_RD");
        }
        // Both runs use B-RDO.
        std::env::set_var("PHASM_B_RDO", "1");
    }

    let t0 = Instant::now();
    let stego = h264_stego_encode_yuv_string_4domain_multigop_with_pattern(
        &yuv,
        1920,
        1072,
        10,
        GopPattern::Ibpbp { gop: 5, b_count: 1 },
        "psy-bench",
        "psy-bench-pass",
    )
    .expect("phasm encode");
    let dur = t0.elapsed();
    eprintln!("{}: {:?} ({} bytes)", label, dur, stego.len());
    (dur, stego)
}

#[test]
#[ignore]
fn bench_psy_off_vs_on() {
    eprintln!("=== PSY-RD perf measurement spike ===");
    let (off_dur, off_bytes) = run("PSY=off", false);
    let (on_dur, on_bytes) = run("PSY=on (shift=4)", true);

    let off_path = std::env::temp_dir().join("phasm_psy_off.h264");
    let on_path = std::env::temp_dir().join("phasm_psy_on.h264");
    std::fs::write(&off_path, &off_bytes).expect("write off");
    std::fs::write(&on_path, &on_bytes).expect("write on");

    let off_ms = off_dur.as_millis() as f64;
    let on_ms = on_dur.as_millis() as f64;
    let delta_pct = (on_ms - off_ms) / off_ms * 100.0;

    eprintln!();
    eprintln!("PSY off: {} ms", off_ms as u64);
    eprintln!("PSY on : {} ms ({:+.1}% vs off)", on_ms as u64, delta_pct);
    eprintln!();
    eprintln!("Outputs:");
    eprintln!("  off: {}", off_path.display());
    eprintln!("  on : {}", on_path.display());
    eprintln!();
    eprintln!("To compare distributions:");
    eprintln!(
        "  python3 scripts/h264_mb_partition_histogram.py {}",
        off_path.display()
    );
    eprintln!(
        "  python3 scripts/h264_mb_partition_histogram.py {}",
        on_path.display()
    );
}
