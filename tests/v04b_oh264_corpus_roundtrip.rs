// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// V0.4.B — OH264 stego round-trip GATE across the real-world fixture
// corpus.
//
// `openh264_stego_roundtrip.rs::c813_roundtrip_iphone7_1080p` already
// covers iPhone7 at 1080p. This test runs the SAME round-trip primitive
// across every other fixture in
// `core/test-vectors/video/h264/real-world/source/` at a uniform 480p
// resize. A single failure on any fixture fails the gate.
//
// **Scale-down rationale**: full 1080p × N per-fixture is too slow for
// the live lane (~2-5 s each × 9 fixtures = 20+ s test). 480p × 8 frames
// reproduces the encoder/decoder behaviour we care about — different
// content drives different mode-decision + ME paths — without paying
// the 1080p tax. The iPhone7 1080p test already exists for high-res
// coverage.
//
// **Skip-graceful**: fixtures gitignored as `.MOV` / `.mp4` may be
// absent on a fresh clone. The test SKIPs (eprintln + continue) when a
// source is missing, but FAILs if any present fixture round-trips wrong.
// At least one fixture must be present.

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use phasm_core::codec::h264::openh264_stego::{
    openh264_stego_decode_yuv_string, openh264_stego_encode_yuv_text, EncodeOpts,
};
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

/// Scale the source MP4 to a YUV cache file at /tmp. Returns `Some(yuv)`
/// if the source exists and ffmpeg succeeds, `None` if the source is
/// absent (gitignored MOVs on a fresh clone).
fn try_scale_yuv(source_mp4: &str, tag: &str, w: u32, h: u32, n_frames: u32) -> Option<Vec<u8>> {
    let src = corpus_root().join(source_mp4);
    if !src.exists() {
        return None;
    }
    let yuv_path = format!("/tmp/phasm_v04b_corpus_{}_{}x{}_f{}.yuv", tag, w, h, n_frames);
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * (n_frames as usize);
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need {
            return Some(data);
        }
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
        eprintln!("V0.4.B corpus: ffmpeg scale failed for {}", source_mp4);
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
fn v04b_oh264_corpus_roundtrip_480p() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());

    const W: u32 = 480;
    const H: u32 = 272;
    const N: u32 = 8;
    let opts = EncodeOpts { qp: 22, intra_period: 60 };

    let mut tested = 0usize;
    let mut skipped = Vec::new();
    let mut failures: Vec<(String, String)> = Vec::new();

    for fx in CORPUS {
        let Some(yuv) = try_scale_yuv(fx.source, fx.tag, W, H, N) else {
            skipped.push(fx.tag);
            continue;
        };
        // Per-fixture message — embeds the tag so a payload mix-up
        // surfaces in the assertion message.
        let msg = format!("v04b corpus round-trip — {}", fx.tag);
        let pass = "v04b-corpus-pass";

        match openh264_stego_encode_yuv_text(&yuv, W, H, N, opts, &msg, pass) {
            Err(e) => failures.push((fx.tag.to_string(), format!("encode: {e}"))),
            Ok(stego) => {
                match openh264_stego_decode_yuv_string(&stego, pass) {
                    Err(e) => {
                        failures.push((fx.tag.to_string(), format!("decode: {e}")))
                    }
                    Ok(recovered) if recovered != msg => failures.push((
                        fx.tag.to_string(),
                        format!("payload mismatch: got {recovered:?} expected {msg:?}"),
                    )),
                    Ok(_) => {
                        eprintln!(
                            "V0.4.B corpus OK: {} ({}×{}×{}, {} stego bytes)",
                            fx.tag, W, H, N, stego.len()
                        );
                    }
                }
            }
        }
        tested += 1;
    }

    eprintln!(
        "V0.4.B corpus summary: tested={}, skipped={} ({:?}), failures={}",
        tested,
        skipped.len(),
        skipped,
        failures.len()
    );

    assert!(
        tested >= 1,
        "V0.4.B corpus gate requires at least one fixture present \
         (all {} skipped — is ffmpeg installed? Are sources gitignored?)",
        CORPUS.len()
    );
    assert!(
        failures.is_empty(),
        "V0.4.B corpus round-trip failures: {:?}",
        failures
    );
}
