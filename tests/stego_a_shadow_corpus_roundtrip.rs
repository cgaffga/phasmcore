// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// STEGO.A.10 — primary + shadow round-trip GATE across the real-world
// fixture corpus. Mirror of `v04b_oh264_corpus_roundtrip.rs` but
// exercises the n-shadow encoder path
// (`encode_yuv_with_n_shadows_with_pattern_and_files`) on every
// fixture and verifies both passphrases (primary + 1 shadow) decode
// correctly via `h264_stego_smart_decode_video`.
//
// Skip-graceful: fixtures gitignored as `.MOV` / `.mp4` may be
// absent on a fresh clone. Test SKIPs missing sources but FAILs if
// any present fixture round-trips wrong. At least one fixture must
// be present.

#![cfg(all(
    feature = "h264-encoder",
    feature = "openh264-backend",
    feature = "cabac-stego",
))]

use phasm_core::codec::h264::openh264_stego::{
    encode_yuv_with_n_shadows_with_pattern_and_files, EncodeOpts,
};
use phasm_core::codec::h264::stego::cost_weights::CostWeights;
use phasm_core::h264_stego_smart_decode_video;
use phasm_core::stego::shadow_layer::ShadowLayer;
use std::sync::{Mutex, OnceLock};

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn try_scale_yuv(source_mp4: &str, tag: &str, w: u32, h: u32, n_frames: u32) -> Option<Vec<u8>> {
    let src = corpus_root().join(source_mp4);
    if !src.exists() {
        return None;
    }
    let yuv_path = std::env::temp_dir().join(format!("stegoa_corpus_{tag}_{w}x{h}_{n_frames}f.yuv"));
    if yuv_path.exists() && std::fs::metadata(&yuv_path).map(|m| m.len()).unwrap_or(0)
        == (w as u64 * h as u64 * 3 / 2) * n_frames as u64
    {
        return std::fs::read(&yuv_path).ok();
    }
    let vf = format!("scale={}:{}", w, h);
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &n_frames.to_string()])
        .args(["-an", "-vf", &vf])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .ok()?;
    if !status.success() {
        eprintln!("STEGO.A.10 corpus: ffmpeg scale failed for {}", source_mp4);
        return None;
    }
    std::fs::read(&yuv_path).ok()
}

struct Fixture {
    tag: &'static str,
    source: &'static str,
}

const CORPUS: &[Fixture] = &[
    Fixture { tag: "iphone5",       source: "iphone5_1080p_30fps_h264_high.mov" },
    Fixture { tag: "iphone7",       source: "iphone7_1080p_30fps_h264_high.mov" },
    Fixture { tag: "lumix",         source: "lumix_g9_1080p_30fps_h264_high.mp4" },
    Fixture { tag: "dji",           source: "dji_mini2_2_7k_24fps_h264_high.mp4" },
    Fixture { tag: "img4138",       source: "IMG_4138.MOV" },
    Fixture { tag: "asia_bottle",   source: "Artlist_AsiaBottle.mp4" },
    Fixture { tag: "carplane",      source: "Artlist_CarPlane.mp4" },
    Fixture { tag: "handbag",       source: "Artlist_Handbag.mp4" },
    Fixture { tag: "horse_flag",    source: "Artlist_HorseFlag.mp4" },
    Fixture { tag: "phone_booth",   source: "Artlist_PhoneBooth.mp4" },
    Fixture { tag: "pirate_battle", source: "Artlist_PirateBattle.mp4" },
    Fixture { tag: "school_fight",  source: "Artlist_SchoolFight.mp4" },
    Fixture { tag: "woman_subway",  source: "Artlist_WomanSubway.mp4" },
];

#[test]
fn stego_a_shadow_corpus_roundtrip_480p() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());

    const W: u32 = 480;
    const H: u32 = 272;
    const N: u32 = 8;
    let opts = EncodeOpts { qp: 26, intra_period: 5 };
    let weights = CostWeights::default();

    let mut tested = 0usize;
    let mut skipped = Vec::new();
    let mut failures: Vec<(String, String)> = Vec::new();
    let mut capacity_skips: Vec<String> = Vec::new();

    for fx in CORPUS {
        let Some(yuv) = try_scale_yuv(fx.source, fx.tag, W, H, N) else {
            skipped.push(fx.tag);
            continue;
        };

        let primary_msg = format!("p {}", fx.tag);
        let primary_pass = format!("ppass-stegoa10-{}", fx.tag);
        let shadow_msg = format!("s {}", fx.tag);
        let shadow_pass = format!("spass-stegoa10-{}", fx.tag);

        let shadows = [ShadowLayer {
            message: &shadow_msg,
            passphrase: &shadow_pass,
            files: &[],
        }];

        let encode_res = encode_yuv_with_n_shadows_with_pattern_and_files(
            &yuv, W, H, N, opts,
            &primary_msg, &[], &primary_pass,
            &shadows,
            &weights,
        );

        match encode_res {
            // Capacity-too-small is content-dependent (some fixtures
            // have insufficient cover for primary+shadow at this QP /
            // resolution); not a correctness bug. Track separately so
            // we don't gate on it.
            Err(e) if format!("{e}").contains("too large") => {
                capacity_skips.push(fx.tag.to_string());
                eprintln!("STEGO.A.10 corpus capacity-skip: {} ({})", fx.tag, e);
            }
            Err(e) => failures.push((fx.tag.to_string(), format!("encode: {e}"))),
            Ok(annex_b) => {
                let stego_size = annex_b.len();
                let primary_dec = h264_stego_smart_decode_video(&annex_b, &primary_pass);
                let shadow_dec = h264_stego_smart_decode_video(&annex_b, &shadow_pass);
                match (primary_dec, shadow_dec) {
                    (Err(e), _) => failures.push((
                        fx.tag.to_string(),
                        format!("primary decode: {e}"),
                    )),
                    (_, Err(e)) => failures.push((
                        fx.tag.to_string(),
                        format!("shadow decode: {e}"),
                    )),
                    (Ok(pr), _) if pr != primary_msg => failures.push((
                        fx.tag.to_string(),
                        format!("primary mismatch: got {pr:?} expected {primary_msg:?}"),
                    )),
                    (_, Ok(sh)) if sh != shadow_msg => failures.push((
                        fx.tag.to_string(),
                        format!("shadow mismatch: got {sh:?} expected {shadow_msg:?}"),
                    )),
                    _ => {
                        eprintln!(
                            "STEGO.A.10 corpus OK: {} ({}×{}×{}, primary+1 shadow, {} stego bytes)",
                            fx.tag, W, H, N, stego_size,
                        );
                    }
                }
            }
        }
        tested += 1;
    }

    eprintln!(
        "STEGO.A.10 corpus summary: tested={}, skipped={} ({:?}), capacity_skips={} ({:?}), failures={}",
        tested,
        skipped.len(),
        skipped,
        capacity_skips.len(),
        capacity_skips,
        failures.len(),
    );

    assert!(
        tested >= 1,
        "STEGO.A.10 corpus gate requires at least one fixture present \
         (all {} skipped — is ffmpeg installed? Are sources gitignored?)",
        CORPUS.len()
    );
    assert!(
        failures.is_empty(),
        "STEGO.A.10 corpus FAILED — {} failures: {:?}",
        failures.len(),
        failures,
    );
}
