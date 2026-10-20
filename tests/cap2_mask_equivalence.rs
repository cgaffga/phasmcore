// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// CAP2 — the capacity probe's structural-cost-mask equivalence GATE.
//
// The per-GOP capacity probe (`openh264_stego::h264_gop_capacity`) used to
// compute full Tier 3 J-UNIWARD content costs before its STC trials. It no
// longer does: the trials only ask `stc_embed(...).is_some()`, which fails on
// exactly one condition — no finite-cost path through the Viterbi trellis — so
// the answer is a function of the cost vector's ∞ pattern and nothing else.
// That pattern is structural (syntax path + frame range), so the probe now
// builds it directly via `content_costs::compute_cost_mask`, skipping the YUV
// read, the 3-subband wavelet and the per-position 19×19×3 reduction. On 1080p
// that was 41% of probe wall time (15.8 s of 38.5 s, `PHASM_CAP_PROFILE=1`,
// 2026-10-18).
//
// "The ∞ pattern is structural" is an argument, not a proof: the real cost
// path has a data-dependent tail that maps a computed cost to ∞ unless it is
// finite and `> 0.0`. This test is what turns the argument into evidence. It
// runs the probe BOTH ways over the real-world fixture corpus and asserts
// identical `per_tier_payload` — all 5 tiers, every fixture.
//
// **If this gate fires, the probe is reporting the wrong capacity.** Fix it;
// do not re-baseline it. Capacity over-report is the #797/#804/#814 bug class
// (silent payload loss), which is why the comparison is exact rather than
// tolerance-based.
//
// Skip-graceful: fixture `.MOV` / `.mp4` sources are gitignored and may be
// absent on a fresh clone. A missing fixture SKIPs; a present fixture that
// disagrees FAILs. At least one fixture must be present.

#![cfg(feature = "h264-encoder")]

mod common;
use phasm_core::codec::h264::openh264_stego::{
    h264_gop_capacity_per_tier_with_costs, h264_gop_cost_mask_divergence,
};
use phasm_core::codec::h264::stego::cost_weights::CostWeights;
use phasm_core::codec::h264::openh264_stego::EncodeOpts;

use std::sync::{Mutex, OnceLock};

// The OH264 fork keeps process-global encoder state, so probe calls must not
// overlap. Same convention as every other OH264 test (`--test-threads=1`).
static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

/// Scale a corpus source to a cached raw-YUV file. `None` when the source is
/// absent (gitignored on a fresh clone) or ffmpeg fails.
fn try_scale_yuv(source: &str, tag: &str, w: u32, h: u32, n_frames: u32) -> Option<Vec<u8>> {
    let src = corpus_root().join(source);
    if !src.exists() {
        return None;
    }
    let yuv_path = format!("{}/phasm_cap2mask_{}_{}x{}_f{}.yuv", common::scratch_dir(), tag, w, h, n_frames);
    // Exactly `need` bytes: `validate_dims` inside the probe is length-strict,
    // and a cached file from an earlier run with more frames would be longer.
    let need = (w * h * 3 / 2) as usize * n_frames as usize;
    if let Ok(mut data) = std::fs::read(&yuv_path) {
        if data.len() >= need {
            data.truncate(need);
            return Some(data);
        }
    }
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &n_frames.to_string()])
        .args(["-an", "-vf", &format!("scale={}:{}", w, h)])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .ok()?;
    if !status.success() {
        eprintln!("cap2-mask: ffmpeg scale failed for {source}");
        return None;
    }
    let mut data = std::fs::read(&yuv_path).ok()?;
    data.truncate(need);
    Some(data)
}

struct Fixture {
    tag: &'static str,
    source: &'static str,
}

const CORPUS: &[Fixture] = &[
    Fixture { tag: "iphone5", source: "iphone5_1080p_30fps_h264_high.mov" },
    Fixture { tag: "iphone7", source: "iphone7_1080p_30fps_h264_high.mov" },
    Fixture { tag: "lumix", source: "lumix_g9_1080p_30fps_h264_high.mp4" },
    Fixture { tag: "dji", source: "dji_mini2_2_7k_24fps_h264_high.mp4" },
    Fixture { tag: "img4138", source: "IMG_4138.MOV" },
    Fixture { tag: "asia_bottle", source: "Artlist_AsiaBottle.mp4" },
    Fixture { tag: "carplane", source: "Artlist_CarPlane.mp4" },
    Fixture { tag: "handbag", source: "Artlist_Handbag.mp4" },
    Fixture { tag: "horse_flag", source: "Artlist_HorseFlag.mp4" },
    Fixture { tag: "phone_booth", source: "Artlist_PhoneBooth.mp4" },
    Fixture { tag: "pirate_battle", source: "Artlist_PirateBattle.mp4" },
    Fixture { tag: "school_fight", source: "Artlist_SchoolFight.mp4" },
    Fixture { tag: "woman_subway", source: "Artlist_WomanSubway.mp4" },
];

/// Per-fixture probe both ways. `full_tiers = true` so all 5 tiers are really
/// computed — the tier filter is the one step that rewrites costs to ∞ after
/// the base vector is built, so it is exactly where a mask/full divergence
/// would show up, and the cheap `full_tiers = false` path would hide it by
/// copying tier 0 into tiers 1..4.
fn opts_for(qp: i32) -> EncodeOpts {
    EncodeOpts { qp, ..Default::default() }
}

fn compare_fixture(
    yuv: &[u8],
    w: u32,
    h: u32,
    n: u32,
    qp: i32,
) -> Result<([usize; 5], [usize; 5]), String> {
    let opts = opts_for(qp);
    let weights = CostWeights::default();

    let masked =
        h264_gop_capacity_per_tier_with_costs(yuv, w, h, n, opts, &weights, true, false)
            .map_err(|e| format!("mask probe: {e}"))?;
    let full =
        h264_gop_capacity_per_tier_with_costs(yuv, w, h, n, opts, &weights, true, true)
            .map_err(|e| format!("full probe: {e}"))?;
    Ok((masked, full))
}

#[test]
fn cap2_structural_mask_matches_full_juniward_on_corpus() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());

    // 480×272 × 8 frames, QP 22 — the same working point the other corpus
    // gates use. Capacity equivalence is a property of the ∞ pattern, not of
    // resolution, and 13 fixtures at 1080p would make this a multi-minute test.
    // `cap2_structural_mask_matches_full_juniward_1080p` covers high-res.
    const W: u32 = 480;
    const H: u32 = 272;
    const N: u32 = 8;
    const QP: i32 = 22;

    let mut tested = 0usize;
    let mut skipped = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for fx in CORPUS {
        let Some(yuv) = try_scale_yuv(fx.source, fx.tag, W, H, N) else {
            skipped.push(fx.tag);
            continue;
        };
        // The ∞ pattern first — it is the property the optimization rests on.
        // Capacity equality follows from it, so checking both localizes a
        // failure: pattern-only means the probe's trial space happened to be
        // insensitive; capacity-only would mean something else moved.
        match h264_gop_cost_mask_divergence(&yuv, W, H, N, opts_for(QP)) {
            Err(e) => failures.push(format!("{}: divergence probe: {e}", fx.tag)),
            Ok(div) if div != [0, 0, 0, 0] => failures.push(format!(
                "{}: ∞-pattern diverges — [csb, csl, msb, msl] mismatches {:?}",
                fx.tag, div
            )),
            Ok(_) => {}
        }

        match compare_fixture(&yuv, W, H, N, QP) {
            Err(e) => failures.push(format!("{}: {e}", fx.tag)),
            Ok((masked, full)) => {
                if masked != full {
                    failures.push(format!(
                        "{}: per-tier capacity differs — mask {:?} vs full J-UNIWARD {:?}",
                        fx.tag, masked, full
                    ));
                } else {
                    eprintln!(
                        "cap2-mask {:<14} per-tier {:?} — identical, ∞ pattern identical",
                        fx.tag, masked
                    );
                }
                tested += 1;
            }
        }
    }

    if !skipped.is_empty() {
        eprintln!("cap2-mask: SKIPPED (absent sources): {skipped:?}");
    }
    assert!(
        tested > 0,
        "no corpus fixtures present — cannot assert mask equivalence. Place sources in {}",
        corpus_root().display()
    );
    assert!(
        failures.is_empty(),
        "capacity-probe cost mask is NOT equivalent to full J-UNIWARD costs \
         ({} of {} fixtures disagree):\n  {}",
        failures.len(),
        tested,
        failures.join("\n  ")
    );
    eprintln!("cap2-mask: {tested} fixtures, per-tier capacity identical both ways");
}

/// The 1080p arm: one fixture, native-ish resolution, a full 30-frame GOP.
/// Resolution changes the cover size by ~20×, and with it the `w` the STC
/// trials pick and how deep the bisection goes — so it exercises a different
/// region of the trial space than the 480p sweep. `#[ignore]` because it is
/// the slow one (the full-J-UNIWARD arm pays the wavelet this change exists
/// to remove); run it with `--include-ignored` before shipping a probe change.
#[test]
#[ignore = "slow (1080p × 30f, both cost paths) — run with --include-ignored"]
fn cap2_structural_mask_matches_full_juniward_1080p() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());

    const W: u32 = 1920;
    const H: u32 = 1072;
    const N: u32 = 30;
    const QP: i32 = 26;

    let Some(yuv) = try_scale_yuv("iphone7_1080p_30fps_h264_high.mov", "iphone7", W, H, N)
    else {
        eprintln!("cap2-mask 1080p: iphone7 source absent — SKIP");
        return;
    };

    let (masked, full) = compare_fixture(&yuv, W, H, N, QP).expect("probe failed");
    assert_eq!(
        masked, full,
        "1080p per-tier capacity differs — mask {masked:?} vs full J-UNIWARD {full:?}"
    );
    eprintln!("cap2-mask 1080p: per-tier {masked:?} — identical");
}

/// The **segmented-STC** arm. `stc_embed` routes to
/// `stc_embed_streaming_segmented` once the combined cover exceeds
/// `SEGMENTED_THRESHOLD` (1 M bits); below it the inline Viterbi runs. The 480p
/// sweep and the 1080p arm both sit under that threshold (~283 k cover bits for
/// a 1080p GOP), so neither of them exercises the segmented branch at all — and
/// shipping the mask on the strength of gates that only cover one of the two STC
/// implementations would be shipping an unproven branch.
///
/// The segmented path's only cost-dependent failure is the same
/// `best_cost == inf` test as the inline path (`streaming_segmented.rs`), so the
/// equivalence argument carries over unchanged; this test is what turns that
/// reading into evidence.
///
/// 4K × 30 frames is what it takes to clear 1 M combined cover bits on real
/// content, which makes this the slowest arm by a wide margin — hence
/// `#[ignore]`. Run it with `--include-ignored` before shipping any change to
/// the probe's cost path.
#[test]
#[ignore = "very slow (4K × 30f, both cost paths) — run with --include-ignored"]
fn cap2_structural_mask_matches_full_juniward_segmented_path() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());

    // The DJI source is natively 2720×1530, so upscaling to 4K keeps real
    // high-frequency content rather than inventing it from a 1080p source.
    const W: u32 = 3840;
    const H: u32 = 2160;
    const N: u32 = 30;
    const QP: i32 = 26;

    let Some(yuv) =
        try_scale_yuv("dji_mini2_2_7k_24fps_h264_high.mp4", "dji4k", W, H, N)
    else {
        eprintln!("cap2-mask segmented: dji source absent — SKIP");
        return;
    };

    let div = h264_gop_cost_mask_divergence(&yuv, W, H, N, opts_for(QP))
        .expect("divergence probe failed");
    assert_eq!(
        div,
        [0, 0, 0, 0],
        "4K ∞-pattern diverges — [csb, csl, msb, msl] mismatches {div:?}"
    );

    let (masked, full) = compare_fixture(&yuv, W, H, N, QP).expect("probe failed");
    assert_eq!(
        masked, full,
        "4K per-tier capacity differs — mask {masked:?} vs full J-UNIWARD {full:?}"
    );
    eprintln!("cap2-mask segmented (4K): per-tier {masked:?} — identical");
}
