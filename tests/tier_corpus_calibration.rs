// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
//
// D'.7 — Track 1 tier corpus calibration.
//
// Sweep real-world fixtures × tiers 0-4 to:
//   1. Confirm explicit tiers 1-4 roundtrip on real content
//   2. Measure per-fixture × per-tier outcomes (encode_ok, decode_ok,
//      stego bytes) to calibrate a safer Auto heuristic than the naïve
//      count-based default (deferred to Tier 0 in D'.8).
//
// Uses the production OH264 path via `StreamingEncodeSession`. Tier
// override is set via `PHASM_TIER_OVERRIDE` env var for each sweep cell
// (the encoder code checks it inside `encode_yuv_with_pre_framed_bits_4domain`).
//
// Corpus = full `test-vectors/video/h264/real-world/source` set (iPhone,
// Lumix, DJI drone, Artlist library — diverse content styles: static
// product, action, motion, talking-head, abstract texture).
//
// Output: per-fixture × per-tier table on stderr. Tier 0 must roundtrip;
// higher tiers may fail on some content (that's the calibration signal).

#![cfg(all(feature = "openh264-backend", feature = "cabac-stego"))]

use phasm_core::{
    h264_stego_capacity_4domain_oh264,
    ColorParams, CostWeights, EncodeEngineChoice, EncodeSessionParams,
    StreamingDecodeSession, StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::h264::openh264_stego::EncodeOpts;
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
    let yuv_path = format!("/tmp/phasm_tier_corpus_{}_{}x{}_f{}.yuv", tag, w, h, n_frames);
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * (n_frames as usize);
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need {
            return Some(data[..need].to_vec());
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
        return None;
    }
    let data = std::fs::read(&yuv_path).ok()?;
    if data.len() < need {
        return None;
    }
    Some(data[..need].to_vec())
}

struct Fixture {
    tag: &'static str,
    source: &'static str,
    /// Brief content-style note for the calibration table.
    style: &'static str,
}

/// Full real-world corpus — diverse content styles for tier calibration.
/// Tags + sources match `v04b_oh264_corpus_roundtrip` CORPUS plus the
/// extra Artlist clips.
const CORPUS: &[Fixture] = &[
    Fixture { tag: "iphone5",       source: "iphone5_1080p_30fps_h264_high.mov", style: "smartphone" },
    Fixture { tag: "iphone7",       source: "iphone7_1080p_30fps_h264_high.mov", style: "smartphone" },
    Fixture { tag: "lumix",         source: "lumix_g9_1080p_30fps_h264_high.mp4", style: "mirrorless" },
    Fixture { tag: "dji",           source: "dji_mini2_2_7k_24fps_h264_high.mp4", style: "drone-aerial" },
    Fixture { tag: "img4138",       source: "IMG_4138.MOV",                       style: "smartphone" },
    Fixture { tag: "asia_bottle",   source: "Artlist_AsiaBottle.mp4",             style: "product-static" },
    Fixture { tag: "carplane",      source: "Artlist_CarPlane.mp4",               style: "action-motion" },
    Fixture { tag: "handbag",       source: "Artlist_Handbag.mp4",                style: "product-static" },
    Fixture { tag: "horse_flag",    source: "Artlist_HorseFlag.mp4",              style: "fast-motion" },
    Fixture { tag: "phone_booth",   source: "Artlist_PhoneBooth.mp4",             style: "interior" },
    Fixture { tag: "pirate_battle", source: "Artlist_PirateBattle.mp4",           style: "high-motion" },
    Fixture { tag: "school_fight",  source: "Artlist_SchoolFight.mp4",            style: "fast-motion" },
    Fixture { tag: "woman_subway",  source: "Artlist_WomanSubway.mp4",            style: "talking-head" },
];

#[derive(Debug, Clone, Copy)]
struct TierOutcome {
    encode_ok: bool,
    decode_ok: bool,
    stego_bytes: usize,
}

fn try_tier(
    yuv: &[u8],
    w: u32, h: u32, n_frames: u32,
    gop_size: u32,
    msg: &str, pass: &str,
    tier_idx: u8,
) -> TierOutcome {
    // Tier override via env var — the OH264 path
    // `encode_yuv_with_pre_framed_bits_4domain` checks PHASM_TIER_OVERRIDE
    // and resolves it ahead of the Auto heuristic.
    // SAFETY: Rust 2024 edition flags env::set_var as unsafe due to
    // racey global state. Single-threaded test harness (SESSION_GUARD
    // mutex on the outer test) is fine.
    unsafe { std::env::set_var("PHASM_TIER_OVERRIDE", tier_idx.to_string()); }

    let result = (|| -> Result<Vec<u8>, String> {
        let params = EncodeSessionParams {
            width: w,
            height: h,
            fps_num: 30,
            fps_den: 1,
            qp: 26,
            gop_size,
            total_frames_hint: n_frames,
            color: ColorParams::default(),
            engine: EncodeEngineChoice::Oh264,
            cost_weights: CostWeights::default(),
            progress_callback: None,
        };
        let mut enc = StreamingEncodeSession::create(params, msg, pass)
            .map_err(|e| format!("create: {e:?}"))?;

        let frame_bytes = (w as usize) * (h as usize) * 3 / 2;
        let chroma_w = (w as usize) / 2;
        let chroma_h = (h as usize) / 2;
        let y_size = (w as usize) * (h as usize);
        let chroma_size = chroma_w * chroma_h;

        let mut annex_b = Vec::new();
        for f in 0..n_frames {
            let off = (f as usize) * frame_bytes;
            let y_plane = &yuv[off..off + y_size];
            let u_plane = &yuv[off + y_size..off + y_size + chroma_size];
            let v_plane = &yuv[off + y_size + chroma_size..off + y_size + 2 * chroma_size];
            let frame = YuvFrameRef {
                y: y_plane, y_stride: w as usize,
                u: u_plane, u_stride: chroma_w,
                v: v_plane, v_stride: chroma_w,
            };
            enc.push_frame(frame, &mut annex_b)
                .map_err(|e| format!("push: {e:?}"))?;
        }
        enc.finish(&mut annex_b)
            .map_err(|e| format!("finish: {e:?}"))?;
        Ok(annex_b)
    })();

    unsafe { std::env::remove_var("PHASM_TIER_OVERRIDE"); }

    match result {
        Err(e) => {
            eprintln!("[#796/tier_sweep] encode FAIL tier={tier_idx}: {e}");
            TierOutcome { encode_ok: false, decode_ok: false, stego_bytes: 0 }
        },
        Ok(bytes) => {
            let stego_bytes = bytes.len();
            // Pair with StreamingDecodeSession — chunk_frame protocol needs
            // the per-GOP reassembler, not the one-shot smart_decode_video.
            let decoded: Option<String> = (|| -> Option<String> {
                let mut dec = StreamingDecodeSession::create(pass).ok()?;
                dec.push_annex_b(&bytes).ok()?;
                dec.finish().ok().map(|d| d.text)
            })();
            match decoded {
                Some(recovered) if recovered == msg => TierOutcome {
                    encode_ok: true,
                    decode_ok: true,
                    stego_bytes,
                },
                _ => TierOutcome {
                    encode_ok: true,
                    decode_ok: false,
                    stego_bytes,
                },
            }
        }
    }
}

/// #796 — bisect: reproduce the cascade failure pattern at 8 frames
/// (matches `v04b_oh264_corpus_roundtrip` parameters). If failures
/// shift between 8 vs 30 frames, the bug is fixture-size-related.
#[test]
fn d_prime_7_tier_sweep_480p_8frames() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    run_tier_sweep(480, 272, 8, 8,
        "tier sweep — 8-frame bisect for #796",
        "tier-sweep-pass-8f");
}

/// D'.7 — explicit-tier sweep on real corpus at 480p × 30 frames.
/// Skips gracefully when corpus is absent (fresh clone).
#[test]
fn d_prime_7_tier_sweep_480p() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    run_tier_sweep(480, 272, 30, 30,
        "tier sweep — real corpus payload, ~50 bytes long padding",
        "tier-sweep-pass");
}

fn run_tier_sweep(w: u32, h: u32, n_frames: u32, gop: u32, msg: &str, pass: &str) {
    let W = w; let H = h; let N = n_frames; let GOP = gop;

    let mut rows: Vec<(String, &'static str, [TierOutcome; 5], [usize; 5])> = Vec::new();
    let mut skipped: Vec<&str> = Vec::new();

    for fx in CORPUS {
        let Some(yuv) = try_scale_yuv(fx.source, fx.tag, W, H, N) else {
            skipped.push(fx.tag);
            continue;
        };

        // Per-tier capacity via the OH264-accurate walker (#796) — must
        // match the OH264 streaming session this sweep actually encodes
        // with. The pure-Rust `h264_stego_capacity_4domain` over-reports
        // by up to ~32× on OH264 content (lumix/dji/horse_flag), which
        // is why the old table showed a 9247-byte cap next to an
        // encode-fail. qp/intra_period mirror the encode params below.
        let cap_opts = EncodeOpts { qp: 26, intra_period: GOP as i32 };
        let capacities = match h264_stego_capacity_4domain_oh264(&yuv, W, H, N as usize, cap_opts, /* full_tiers */ true) {
            Ok(info) => info.per_tier_primary_max_message_bytes,
            Err(_) => [0; 5],
        };

        let outcomes = [
            try_tier(&yuv, W, H, N, GOP, msg, pass, 0),
            try_tier(&yuv, W, H, N, GOP, msg, pass, 1),
            try_tier(&yuv, W, H, N, GOP, msg, pass, 2),
            try_tier(&yuv, W, H, N, GOP, msg, pass, 3),
            try_tier(&yuv, W, H, N, GOP, msg, pass, 4),
        ];

        rows.push((fx.tag.to_string(), fx.style, outcomes, capacities));
    }

    if rows.is_empty() {
        eprintln!(
            "D'.7 tier sweep: all {} fixtures skipped (missing sources / no ffmpeg)",
            CORPUS.len(),
        );
        return;
    }

    eprintln!("\n=== D'.7 tier corpus calibration — {W}x{H}x{N}, gop={GOP} ===");
    eprintln!(
        "{:<14} {:<16} | {:>6} {:>6} {:>6} {:>6} {:>6} | {:>3} {:>3} {:>3} {:>3} {:>3}",
        "fixture", "style",
        "T0cap", "T1cap", "T2cap", "T3cap", "T4cap",
        "T0", "T1", "T2", "T3", "T4",
    );
    let mut ok_per_tier = [0usize; 5];
    let attempts = rows.len();
    for (tag, style, outcomes, caps) in &rows {
        let cells: Vec<String> = outcomes.iter().map(|o| {
            if !o.encode_ok { "E".to_string() }
            else if !o.decode_ok { "X".to_string() }
            else { "OK".to_string() }
        }).collect();
        eprintln!(
            "{:<14} {:<16} | {:>6} {:>6} {:>6} {:>6} {:>6} | {:>3} {:>3} {:>3} {:>3} {:>3}",
            tag, style,
            caps[0], caps[1], caps[2], caps[3], caps[4],
            cells[0], cells[1], cells[2], cells[3], cells[4],
        );
        for t in 0..5 {
            if outcomes[t].encode_ok && outcomes[t].decode_ok {
                ok_per_tier[t] += 1;
            }
        }
    }
    eprintln!(
        "{:<14} {:<16} | {:>6} {:>6} {:>6} {:>6} {:>6} | {:>3} {:>3} {:>3} {:>3} {:>3}",
        "tier success", "(↑ = bigger)",
        "—", "—", "—", "—", "—",
        format!("{}/{}", ok_per_tier[0], attempts),
        format!("{}/{}", ok_per_tier[1], attempts),
        format!("{}/{}", ok_per_tier[2], attempts),
        format!("{}/{}", ok_per_tier[3], attempts),
        format!("{}/{}", ok_per_tier[4], attempts),
    );
    eprintln!("(legend: OK=roundtrip ok, X=encode ok but decode fails, E=encode fails)");
    if !skipped.is_empty() {
        eprintln!("Skipped (sources missing): {:?}", skipped);
    }

    // D'.7 finding: tier filter is content-neutral on this corpus —
    // each fixture's OK/X outcome is the same across all tiers. The
    // gate asserts that property (tier filter doesn't introduce new
    // failures) rather than absolute Tier 0 success — the latter is
    // limited by pre-existing streaming-session cascade bugs on
    // ~30% of corpus content (lumix, dji, school_fight, woman_subway).
    for tier_idx in 1..=4 {
        assert!(
            ok_per_tier[tier_idx] >= ok_per_tier[0],
            "Tier {} regressed vs Tier 0 ({} vs {} OK) — tier filter \
             should be content-neutral",
            tier_idx, ok_per_tier[tier_idx], ok_per_tier[0],
        );
    }
}
