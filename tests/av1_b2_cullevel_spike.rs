// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase B.2.1 — Q1 spike: GolombTailLsb capacity and cascade-risk
//! quantification on the W4 corpus.
//!
//! Informs the cost-function design for B.2.2 (`channel-design.md`
//! § 4.2 + `phase-b2-golombtaillsb.md` § 3.3):
//!   - Hard-exclusion (cost = inf at risky positions) — if very few
//!     positions are risky, hard-exclusion suffices.
//!   - Soft-penalty (cost multiplied by culLevel proximity) — only
//!     worth the complexity if many positions sit close to the cap.
//!
//! # MVP scope
//!
//! Without extending AcSignMeta to track coefficient level, we can
//! measure capacity (positions per frame) but NOT the per-position
//! culLevel proximity. This spike ships the capacity measurement;
//! culLevel proximity is a v0.6+ extension that needs a fork-side
//! meta addition.

#![cfg(all(feature = "av1-encoder", feature = "av1-backend"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
    PHASM_TAG_AC_COEFF_SIGN, PHASM_TAG_GOLOMB_TAIL_LSB, PHASM_TAG_OTHER,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

struct SeekFixture {
    name: &'static str,
    source: &'static str,
    duration_s: f32,
}

const FIXTURES: &[SeekFixture] = &[
    SeekFixture { name: "iphone_img4138", source: "IMG_4138.MOV", duration_s: 20.0 },
    SeekFixture { name: "carplane",       source: "Artlist_CarPlane.mp4", duration_s: 8.0 },
    SeekFixture { name: "iphone5_1080p",  source: "iphone5_1080p_30fps_h264_high.mov", duration_s: 12.0 },
];

const SEEK_POINTS_PER_FIXTURE: usize = 10;
const WIDTH: u32 = 256;
const HEIGHT: u32 = 144;
const QUANTIZER: usize = 30;

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv_frame(source: &str, seek_s: f32) -> Vec<u8> {
    let src = corpus_root().join(source);
    assert!(src.exists(), "missing corpus source: {}", src.display());
    let vf = format!("scale={}:{}:force_original_aspect_ratio=disable", WIDTH, HEIGHT);
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args(["-frames:v", "1", "-vf", &vf, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output()
        .expect("ffmpeg");
    assert!(out.status.success());
    out.stdout
}

fn encode_natural(yuv: &[u8]) -> (usize, usize, usize) {
    let config = Arc::new(EncoderConfig {
        width: WIDTH as usize,
        height: HEIGHT as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: QUANTIZER,
        ..Default::default()
    });
    let mut sequence = Sequence::new(&config);
    sequence.enable_large_lru = false;
    let mut fi = FrameInvariants::<u8>::new_key_frame(
        config.clone(),
        Arc::new(sequence),
        0,
        Box::new([]),
    );
    fi.enable_segmentation = false;

    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let mut frame = make_frame::<u8>(w, h, ChromaSampling::Cs420);
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    frame.planes[0].copy_from_raw_u8(&yuv[..y_size], w, 1);
    frame.planes[1].copy_from_raw_u8(&yuv[y_size..y_size + uv_size], w / 2, 1);
    frame.planes[2].copy_from_raw_u8(&yuv[y_size + uv_size..y_size + 2 * uv_size], w / 2, 1);

    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame));
    let inter_cfg = make_inter_config(&config);
    let (_, recording) = encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);

    let tile = &recording.tiles[0];
    let mut n_ac = 0;
    let mut n_golomb = 0;
    let mut n_other = 0;
    for &tag in &tile.bit_tags {
        match tag {
            t if t == PHASM_TAG_AC_COEFF_SIGN => n_ac += 1,
            t if t == PHASM_TAG_GOLOMB_TAIL_LSB => n_golomb += 1,
            t if t == PHASM_TAG_OTHER => n_other += 1,
            _ => {}
        }
    }
    (n_ac, n_golomb, n_other)
}

#[test]
fn b2_1_golomb_tail_capacity_spike() {
    eprintln!(
        "[B.2.1] corpus: {} fixtures x {} seek points = {} samples",
        FIXTURES.len(),
        SEEK_POINTS_PER_FIXTURE,
        FIXTURES.len() * SEEK_POINTS_PER_FIXTURE
    );

    let mut totals_ac = 0usize;
    let mut totals_golomb = 0usize;
    let mut totals_other = 0usize;
    let mut per_fixture: Vec<(String, f64, f64, f64)> = Vec::new();

    for fx in FIXTURES {
        let mut fx_ac = 0usize;
        let mut fx_golomb = 0usize;
        let mut fx_other = 0usize;
        let usable_range = fx.duration_s - 1.0;
        let step = usable_range / SEEK_POINTS_PER_FIXTURE as f32;
        for i in 0..SEEK_POINTS_PER_FIXTURE {
            let seek_s = 0.5 + step * i as f32;
            let yuv = extract_yuv_frame(fx.source, seek_s);
            let (n_ac, n_golomb, n_other) = encode_natural(&yuv);
            fx_ac += n_ac;
            fx_golomb += n_golomb;
            fx_other += n_other;
        }
        let avg_ac = fx_ac as f64 / SEEK_POINTS_PER_FIXTURE as f64;
        let avg_golomb = fx_golomb as f64 / SEEK_POINTS_PER_FIXTURE as f64;
        let avg_other = fx_other as f64 / SEEK_POINTS_PER_FIXTURE as f64;
        per_fixture.push((
            fx.name.to_string(),
            avg_ac,
            avg_golomb,
            avg_other,
        ));
        totals_ac += fx_ac;
        totals_golomb += fx_golomb;
        totals_other += fx_other;
    }

    eprintln!("[B.2.1] Per-fixture avg bits/frame (256x144 @ QP=30, key frames):");
    eprintln!(
        "[B.2.1]   {:<20} {:>10} {:>10} {:>10} {:>10}",
        "fixture", "AC_sign", "GolombTL", "OTHER", "GolombTL%"
    );
    for (name, ac, golomb, other) in &per_fixture {
        let pct = if (ac + golomb) > 0.0 {
            100.0 * golomb / (ac + golomb)
        } else {
            0.0
        };
        eprintln!(
            "[B.2.1]   {:<20} {:>10.0} {:>10.0} {:>10.0} {:>9.2}%",
            name, ac, golomb, other, pct
        );
    }

    let n_samples = (FIXTURES.len() * SEEK_POINTS_PER_FIXTURE) as f64;
    let avg_ac = totals_ac as f64 / n_samples;
    let avg_golomb = totals_golomb as f64 / n_samples;
    let avg_other = totals_other as f64 / n_samples;
    let overall_pct = if (avg_ac + avg_golomb) > 0.0 {
        100.0 * avg_golomb / (avg_ac + avg_golomb)
    } else {
        0.0
    };

    eprintln!("[B.2.1] CORPUS AVG:");
    eprintln!("[B.2.1]   AC sign bits per frame:       {:.0}", avg_ac);
    eprintln!("[B.2.1]   Golomb tail bits per frame:   {:.0}", avg_golomb);
    eprintln!("[B.2.1]   OTHER (non-Tier1) per frame:  {:.0}", avg_other);
    eprintln!(
        "[B.2.1]   Tier-1 diversity capacity:    {:.2}% golomb / ({:.0}% AC)",
        overall_pct,
        100.0 - overall_pct
    );

    // Sanity: GolombTailLsb path should produce SOME bits at QP=30
    // on real cover content. If 0 across the whole corpus, the tag
    // wiring is broken.
    assert!(
        totals_golomb > 0,
        "expected GolombTailLsb bits on real cover at QP=30 — tag wiring may be broken"
    );

    // Order-of-magnitude expectations from channel-design.md:
    // - AC sign: ~tens of thousands per 1080p frame; tens of
    //   thousands proportional at 256x144 too (~hundreds-thousands).
    // - Golomb tail: ~hundreds per 1080p frame; ~tens at 256x144.
    eprintln!(
        "[B.2.1] EXPECTED v0.5: hard-exclusion cost (cost=inf at risk-1 positions) suffices since"
    );
    eprintln!(
        "[B.2.1]                AC sign provides >>{:.0}x more capacity than golomb tail.",
        avg_ac / avg_golomb.max(1.0)
    );
}
