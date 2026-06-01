// Diagnose user's new stego video (VID_20260523_185551.mp4): does
// primary + shadow actually decode locally? Also measure where time
// goes (walk + cascade-safety + shadow extract + Scheme A extract).

#![cfg(all(
    feature = "h264-encoder",
    feature = "openh264-backend",
    feature = "cabac-stego",
))]

use phasm_core::codec::h264::cabac::bin_decoder::{walk_annex_b_for_cover_with_options, WalkOptions};
use phasm_core::codec::h264::stego::cascade_safety::{analyze_safe_mvd_subset, derive_msl_safe_from_msb};
use phasm_core::codec::h264::stego::shadow::shadow_extract_all4_safe;
use phasm_core::h264_stego_smart_decode_video;

#[test]
fn diagnose_user_video2() {
    let annex_b = match std::fs::read("/tmp/stego2.h264") {
        Ok(b) => b,
        Err(e) => { eprintln!("skip: {e}"); return; }
    };
    eprintln!("Annex-B: {} bytes", annex_b.len());

    let t0 = std::time::Instant::now();
    let walk = walk_annex_b_for_cover_with_options(
        &annex_b,
        WalkOptions { record_mvd: true, record_offsets: false },
    ).expect("walk");
    let dt_walk = t0.elapsed();
    eprintln!(
        "walk: {:?}   CSB={} CSL={} MSB={} MSL={}",
        dt_walk,
        walk.cover.coeff_sign_bypass.bits.len(),
        walk.cover.coeff_suffix_lsb.bits.len(),
        walk.cover.mvd_sign_bypass.bits.len(),
        walk.cover.mvd_suffix_lsb.bits.len(),
    );

    let t0 = std::time::Instant::now();
    let safe_msb = analyze_safe_mvd_subset(&walk.mvd_meta, walk.mb_w, walk.mb_h);
    let safe_msl = derive_msl_safe_from_msb(
        &walk.cover.mvd_sign_bypass.positions,
        &safe_msb,
        &walk.cover.mvd_suffix_lsb.positions,
    );
    let dt_safety = t0.elapsed();
    eprintln!(
        "cascade-safety: {:?}   safe_msl = {}/{}",
        dt_safety,
        safe_msl.iter().filter(|&&b| b).count(),
        safe_msl.len(),
    );

    // Shadow extract: fast yes/no on whether shadow is embedded for "s".
    let t0 = std::time::Instant::now();
    let res_s = shadow_extract_all4_safe(&walk.cover, "s", None, Some(&safe_msl));
    eprintln!(
        "shadow_extract(\"s\"): {:?} in {:?}",
        res_s.as_ref().map(|p| &p.text),
        t0.elapsed(),
    );

    let t0 = std::time::Instant::now();
    let res_empty = shadow_extract_all4_safe(&walk.cover, "", None, Some(&safe_msl));
    eprintln!(
        "shadow_extract(\"\"): {:?} in {:?}",
        res_empty.as_ref().map(|p| &p.text),
        t0.elapsed(),
    );

    // Full smart_decode_video — including primary path (slow Scheme A
    // brute-force expected). Measures end-to-end decode time the iOS
    // app actually sees.
    let t0 = std::time::Instant::now();
    let primary = h264_stego_smart_decode_video(&annex_b, "");
    eprintln!(
        "smart_decode(\"\"): {:?} in {:?}",
        primary.as_ref().map(|s| s.as_str()),
        t0.elapsed(),
    );
}
