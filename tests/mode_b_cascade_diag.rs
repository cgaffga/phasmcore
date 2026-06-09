// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
//
// Mode-B cascade diagnostic (#796/#778). D'.7 found dji / school_fight /
// woman_subway fail decode at every tier via StreamingEncodeSession +
// StreamingDecodeSession, while CarPlane (control) round-trips.
//
// This harness measures the FAILURE CHARACTER before any fix:
//   1. Determinism — encode the same fixture TWICE, compare Annex-B
//      byte-for-byte. Identical => deterministic content cascade.
//      Different => OH264 inter-call non-determinism is in play (the
//      "1 residual diff" hypothesis from the wire_only measurement).
//   2. Decode outcome — exact result/error per fixture.
//   3. Frame-count sweep — does it fail at 10 frames or only 30?
//      (isolates GOP-boundary / multi-GOP dependence).
//
// Tier 0 forced (PHASM_TIER_OVERRIDE=0) to isolate from Track 1.
//
// #[ignore] by default — needs ffmpeg + gitignored corpus fixtures.
// Run: PHASM_TIER_OVERRIDE=0 cargo test --test mode_b_cascade_diag \
//        --features h264-encoder,cabac-stego --release -- --ignored --nocapture

#![cfg(feature = "h264-encoder")]

use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingDecodeSession,
    StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::h264::stego::CostWeights;
use std::sync::{Mutex, OnceLock};

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn guard() -> &'static Mutex<()> { SESSION_GUARD.get_or_init(|| Mutex::new(())) }

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn try_scale_yuv(src_mp4: &str, tag: &str, w: u32, h: u32, n: u32) -> Option<Vec<u8>> {
    let src = corpus_root().join(src_mp4);
    if !src.exists() { return None; }
    let yuv_path = format!("/tmp/phasm_modeb_{}_{}x{}_f{}.yuv", tag, w, h, n);
    let need = (w * h * 3 / 2) as usize * n as usize;
    if let Ok(d) = std::fs::read(&yuv_path) { if d.len() >= need { return Some(d[..need].to_vec()); } }
    let vf = format!("scale={}:{}", w, h);
    let ok = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"]).arg(&src)
        .args(["-frames:v", &n.to_string(), "-an", "-vf", &vf,
               "-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path).status().ok()?.success();
    if !ok { return None; }
    let d = std::fs::read(&yuv_path).ok()?;
    if d.len() < need { return None; }
    Some(d[..need].to_vec())
}

fn encode(yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, msg: &str, pass: &str) -> Option<Vec<u8>> {
    let params = EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut enc = StreamingEncodeSession::create(params, msg, pass).ok()?;
    let fb = (w * h * 3 / 2) as usize;
    let (cw, ch) = ((w / 2) as usize, (h / 2) as usize);
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let mut out = Vec::new();
    for f in 0..n as usize {
        let off = f * fb;
        let frame = YuvFrameRef {
            y: &yuv[off..off + ys], y_stride: w as usize,
            u: &yuv[off + ys..off + ys + cs], u_stride: cw,
            v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
        };
        let _ = ch;
        enc.push_frame(frame, &mut out).ok()?;
    }
    enc.finish(&mut out).ok()?;
    Some(out)
}

fn decode(annex_b: &[u8], pass: &str) -> Result<String, String> {
    let mut dec = StreamingDecodeSession::create(pass).map_err(|e| format!("create: {e:?}"))?;
    dec.push_annex_b(annex_b).map_err(|e| format!("push: {e:?}"))?;
    dec.finish().map(|d| d.text).map_err(|e| format!("finish: {e:?}"))
}

/// #778 — walk the encoded Annex-B with the decoder-side walker and
/// print per-domain cover counts. Compare to the encoder's symmetry
/// trace (e.g. school_fight@10f: CS=15290 CSL=8 MVDs=3166 MVDsl=475).
/// A mismatch means the decoder reconstructs a different cover than the
/// encoder planned against → STC misalignment.
fn walk_counts(annex_b: &[u8]) -> String {
    use phasm_core::codec::h264::cabac::bin_decoder::{
        walk_annex_b_for_cover_with_options, WalkOptions,
    };
    match walk_annex_b_for_cover_with_options(
        annex_b, WalkOptions { record_mvd: true, ..Default::default() },
    ) {
        Ok(w) => format!("CS={} CSL={} MVDs={} MVDsl={}",
            w.cover.coeff_sign_bypass.positions.len(),
            w.cover.coeff_suffix_lsb.positions.len(),
            w.cover.mvd_sign_bypass.positions.len(),
            w.cover.mvd_suffix_lsb.positions.len()),
        Err(e) => format!("WALK-ERR {e}"),
    }
}

struct Fx { tag: &'static str, src: &'static str, expect_ok: bool }

// #800: after the chunk_frame payload_len fix, ALL fixtures that encode
// successfully must decode (the m_total w-class ambiguity is gone). The
// gate below asserts NO fixture returns FrameCorrupted.
const FIXTURES: &[Fx] = &[
    Fx { tag: "carplane",     src: "Artlist_CarPlane.mp4",                  expect_ok: true }, // control
    Fx { tag: "school_fight", src: "Artlist_SchoolFight.mp4",              expect_ok: true },
    Fx { tag: "woman_subway", src: "Artlist_WomanSubway.mp4",             expect_ok: true },
    Fx { tag: "dji",          src: "dji_mini2_2_7k_24fps_h264_high.mp4",   expect_ok: true },
];

#[test]
#[ignore]
fn mode_b_diagnose() {
    let _g = guard().lock().unwrap_or_else(|e| e.into_inner());
    unsafe { std::env::set_var("PHASM_TIER_OVERRIDE", "0"); } // isolate from Track 1
    const W: u32 = 480; const H: u32 = 272;
    let msg = "mode-b diag payload — short";
    let pass = "modeb-pass";

    // #800 gate: any FrameCorrupted is a regression of the m_total fix.
    let mut corrupted: Vec<String> = Vec::new();

    for &n in &[10u32, 30u32] {
        let gop = n; // single-GOP at n=10; gop=30 multi at n=30
        eprintln!("\n========== {}x{} x {} frames, gop={} ==========", W, H, n, gop);
        for fx in FIXTURES {
            let Some(yuv) = try_scale_yuv(fx.src, fx.tag, W, H, n) else {
                eprintln!("{:<14} SKIP (no fixture/ffmpeg)", fx.tag); continue;
            };
            let enc1 = encode(&yuv, W, H, n, gop, msg, pass);
            let enc2 = encode(&yuv, W, H, n, gop, msg, pass);
            let (det, bytes) = match (&enc1, &enc2) {
                (Some(a), Some(b)) => (
                    if a == b { "deterministic".to_string() }
                    else {
                        let fd = a.iter().zip(b.iter()).position(|(x, y)| x != y);
                        format!("NON-DET (len {}/{}, first_diff @ {:?})", a.len(), b.len(), fd)
                    },
                    a.len(),
                ),
                _ => ("ENCODE-FAIL".to_string(), 0),
            };
            let dec = match &enc1 {
                Some(b) => match decode(b, pass) {
                    Ok(t) if t == msg => "OK".to_string(),
                    Ok(t) => format!("WRONG ({:?})", &t[..t.len().min(20)]),
                    Err(e) => format!("DECODE-FAIL ({e})"),
                },
                None => "n/a".to_string(),
            };
            let flag = if (dec == "OK") == fx.expect_ok { "" } else { "  <-- UNEXPECTED" };
            let dwalk = enc1.as_ref().map(|b| walk_counts(b)).unwrap_or_default();
            eprintln!("{:<14} {:>9}B  enc:{:<20}  dec:{:<32}  decoder-walk:[{}]{}",
                fx.tag, bytes, det, dec, dwalk, flag);
            if dec.contains("FrameCorrupted") {
                corrupted.push(format!("{}@{}f", fx.tag, n));
            }
        }
    }
    unsafe { std::env::remove_var("PHASM_TIER_OVERRIDE"); }
    assert!(
        corrupted.is_empty(),
        "#800 regression — FrameCorrupted on: {corrupted:?} (decoder m_total ambiguity returned)"
    );
}
