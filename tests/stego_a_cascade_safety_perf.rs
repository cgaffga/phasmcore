// Perf benchmark for analyze_safe_mvd_subset on user's real 1024×576
// stego video. Pre-optimization baseline: 64s. Target: <1s.

#![cfg(feature = "h264-decoder")]

use phasm_core::codec::h264::cabac::bin_decoder::{walk_annex_b_for_cover_with_options, WalkOptions};
use phasm_core::codec::h264::stego::cascade_safety::analyze_safe_mvd_subset;

#[test]
fn cascade_safety_perf_user_video_1024x576() {
    let annex_b = match std::fs::read("/tmp/stego.h264") {
        Ok(b) => b,
        Err(e) => { eprintln!("skip: /tmp/stego.h264 missing ({e})"); return; }
    };
    let walk = walk_annex_b_for_cover_with_options(
        &annex_b,
        WalkOptions { record_mvd: true },
    ).expect("walk");

    eprintln!(
        "Input: {}×{} mb_w×mb_h, n_mvd_positions={}",
        walk.mb_w, walk.mb_h, walk.mvd_meta.len(),
    );

    let start = std::time::Instant::now();
    let safe = analyze_safe_mvd_subset(&walk.mvd_meta, walk.mb_w, walk.mb_h);
    let dt = start.elapsed();

    let n_safe = safe.iter().filter(|&&b| b).count();
    eprintln!(
        "analyze_safe_mvd_subset: {} safe / {} total ({:.1}%) in {:?}",
        n_safe, safe.len(),
        100.0 * n_safe as f64 / safe.len() as f64,
        dt,
    );

    // Soft assertion — we're tracking the perf trend. Production
    // target: <1s for 1024×576. Pre-optimization baseline was 64s.
    if dt.as_secs() > 5 {
        eprintln!("WARN: analyze_safe_mvd_subset slower than 5s target");
    }
}
