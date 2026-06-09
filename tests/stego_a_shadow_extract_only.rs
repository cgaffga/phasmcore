// Fast diagnostic: ONLY try shadow_extract on the user's stego
// video. Skips the slow Scheme A/B brute-force. If shadow IS embedded
// for passphrase "s", this returns the message in <1s. If shadow is
// NOT embedded, returns FrameCorrupted in <1s.

#![cfg(feature = "h264-decoder")]

use phasm_core::codec::h264::cabac::bin_decoder::{walk_annex_b_for_cover_with_options, WalkOptions};
use phasm_core::codec::h264::stego::cascade_safety::{analyze_safe_mvd_subset, derive_msl_safe_from_msb};
use phasm_core::codec::h264::stego::shadow::shadow_extract;

#[test]
fn shadow_extract_only_on_user_video() {
    let annex_b = match std::fs::read("/tmp/stego.h264") {
        Ok(b) => b,
        Err(e) => { eprintln!("skip: /tmp/stego.h264 missing ({e})"); return; }
    };
    eprintln!("Annex-B size: {} bytes", annex_b.len());

    let walk_start = std::time::Instant::now();
    let walk = walk_annex_b_for_cover_with_options(
        &annex_b,
        WalkOptions { record_mvd: true },
    ).expect("walk");
    let dt_walk = walk_start.elapsed();
    eprintln!(
        "Walk: {} mb_w×{} mb_h, cover CSB={} CSL={} MSB={} MSL={} ({:?})",
        walk.mb_w, walk.mb_h,
        walk.cover.coeff_sign_bypass.bits.len(),
        walk.cover.coeff_suffix_lsb.bits.len(),
        walk.cover.mvd_sign_bypass.bits.len(),
        walk.cover.mvd_suffix_lsb.bits.len(),
        dt_walk,
    );

    let safety_start = std::time::Instant::now();
    let safe_msb = analyze_safe_mvd_subset(&walk.mvd_meta, walk.mb_w, walk.mb_h);
    let safe_msl = derive_msl_safe_from_msb(
        &walk.cover.mvd_sign_bypass.positions,
        &safe_msb,
        &walk.cover.mvd_suffix_lsb.positions,
    );
    let dt_safety = safety_start.elapsed();
    let n_safe_msl = safe_msl.iter().filter(|&&b| b).count();
    eprintln!(
        "Cascade-safety: {}/{} MSL safe ({:?})",
        n_safe_msl, safe_msl.len(), dt_safety,
    );

    // Try shadow extract with "s"
    let s_start = std::time::Instant::now();
    let result_s = shadow_extract(&walk.cover, "s", None, Some(&safe_msl));
    let dt_s = s_start.elapsed();
    eprintln!("shadow_extract(\"s\"): {:?} ({:?})", result_s.as_ref().map(|p| &p.text), dt_s);

    // Try with empty pass too
    let e_start = std::time::Instant::now();
    let result_empty = shadow_extract(&walk.cover, "", None, Some(&safe_msl));
    let dt_empty = e_start.elapsed();
    eprintln!("shadow_extract(\"\"): {:?} ({:?})", result_empty.as_ref().map(|p| &p.text), dt_empty);

    // Try common variations in case the iOS UI added/stripped whitespace
    for variation in &["s ", " s", "S"] {
        let start = std::time::Instant::now();
        let result = shadow_extract(&walk.cover, variation, None, Some(&safe_msl));
        eprintln!("shadow_extract(\"{}\"): {:?} ({:?})", variation, result.as_ref().map(|p| &p.text), start.elapsed());
    }
}
