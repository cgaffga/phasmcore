// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
//
// #797 — capacity-accuracy diagnostic on horse_flag (fast-motion).
//
// D'.9 found horse_flag encode-fails (push: MessageTooLarge) on a ~52-byte
// message at 480×272×30f, yet the OH264 capacity walker reports thousands
// of bytes. This harness:
//   1. measures the ACTUAL OH264 cover (n_cover) the encode sees vs the
//      reported capacity, at increasing frame counts;
//   2. confirms the user's hypothesis — with more frames (more cover) the
//      same message DOES encode + round-trip — which proves the 30f
//      failure is pure capacity, not a correctness bug.
//
// Run (needs ffmpeg + gitignored corpus):
//   PHASM_PERF_TRACE=1 cargo test --test horse_flag_capacity_797 \
//     --features openh264-backend,cabac-stego --release -- --ignored --nocapture

#![cfg(all(feature = "openh264-backend", feature = "cabac-stego"))]

use phasm_core::{
    h264_stego_capacity_4domain_oh264,
    ColorParams, CostWeights, EncodeEngineChoice, EncodeSessionParams,
    StreamingDecodeSession, StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::h264::openh264_stego::EncodeOpts;
use std::sync::{Mutex, OnceLock};

static G: OnceLock<Mutex<()>> = OnceLock::new();
fn guard() -> &'static Mutex<()> { G.get_or_init(|| Mutex::new(())) }

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn try_scale_yuv(src_mp4: &str, w: u32, h: u32, n: u32) -> Option<Vec<u8>> {
    let src = corpus_root().join(src_mp4);
    if !src.exists() { return None; }
    // Cache key MUST include the source — otherwise different fixtures
    // at the same WxHxN collide and reuse the first-scaled YUV.
    let key: String = src_mp4.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    let yuv_path = format!("/tmp/phasm_hf797_{}_{}x{}_f{}.yuv", key, w, h, n);
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

fn encode(yuv: &[u8], w: u32, h: u32, n: u32, gop: u32, msg: &str, pass: &str) -> Result<Vec<u8>, String> {
    let params = EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: 26, gop_size: gop,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    };
    let mut enc = StreamingEncodeSession::create(params, msg, pass)
        .map_err(|e| format!("create: {e:?}"))?;
    let fb = (w * h * 3 / 2) as usize;
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let (cw, _ch) = ((w / 2) as usize, (h / 2) as usize);
    let mut out = Vec::new();
    for f in 0..n as usize {
        let off = f * fb;
        let frame = YuvFrameRef {
            y: &yuv[off..off + ys], y_stride: w as usize,
            u: &yuv[off + ys..off + ys + cs], u_stride: cw,
            v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
        };
        enc.push_frame(frame, &mut out).map_err(|e| format!("push: {e:?}"))?;
    }
    enc.finish(&mut out).map_err(|e| format!("finish: {e:?}"))?;
    Ok(out)
}

fn decode(annex_b: &[u8], pass: &str) -> Result<String, String> {
    let mut dec = StreamingDecodeSession::create(pass).map_err(|e| format!("{e:?}"))?;
    dec.push_annex_b(annex_b).map_err(|e| format!("{e:?}"))?;
    dec.finish().map(|d| d.text).map_err(|e| format!("{e:?}"))
}

/// Binary-search the largest message that actually encodes (OK) for a
/// fixture, to compare the TRUE encodable capacity against the reported
/// `h264_stego_capacity_4domain_oh264` value. This is the real
/// accuracy probe for #797.
/// Deterministic ~incompressible printable-ASCII string (LCG-driven).
/// Using "x".repeat(len) would measure Brotli compression, not capacity
/// (the payload layer compresses before STC embed).
fn incompressible(len: usize) -> String {
    let mut s = String::with_capacity(len);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let c = 0x21u8 + ((state >> 33) as usize % 0x5d) as u8; // printable 0x21..0x7d
        s.push(c as char);
    }
    s
}

fn max_encodable(yuv: &[u8], w: u32, h: u32, n: u32, gop: u32) -> usize {
    let pass = "hf797-cap";
    let fits = |len: usize| -> bool {
        let msg = incompressible(len);
        encode(yuv, w, h, n, gop, &msg, pass).is_ok()
    };
    // Exponential probe for an upper bound that fails, then bisect.
    let mut hi = 64usize;
    while fits(hi) && hi < 1 << 20 { hi *= 2; }
    let mut lo = hi / 2; // fits(lo) true (or lo=32 floor)
    if !fits(lo) { lo = 0; }
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if fits(mid) { lo = mid; } else { hi = mid; }
    }
    lo
}

#[test]
#[ignore]
fn horse_flag_oversized_check() {
    // Ground truth: does an obviously-oversized message (50 KB into a
    // ~15 KB cover) get REJECTED at encode, and if it "encodes", does it
    // actually round-trip? Resolves the binary-search 1 MB anomaly.
    let _g = guard().lock().unwrap_or_else(|e| e.into_inner());
    const W: u32 = 480; const H: u32 = 272; const GOP: u32 = 30;
    let Some(yuv) = try_scale_yuv("Artlist_HorseFlag.mp4", W, H, 30) else {
        eprintln!("SKIP (no fixture)"); return;
    };
    for len in [58usize, 5000, 50000] {
        let msg: String = core::iter::repeat('x').take(len).collect();
        match encode(&yuv, W, H, 30, GOP, &msg, "ovz") {
            Ok(b) => {
                let rt = match decode(&b, "ovz") {
                    Ok(t) if t == msg => "ROUND-TRIP-OK",
                    Ok(t) => { let _ = t; "DECODE-WRONG/TRUNCATED" }
                    Err(_) => "DECODE-ERR",
                };
                eprintln!("len={:>6}  encode=OK ({} B annexb)  {}", len, b.len(), rt);
            }
            Err(e) => eprintln!("len={:>6}  encode=ERR({})", len, e),
        }
    }
}

#[test]
#[ignore]
fn horse_flag_true_vs_reported_capacity() {
    let _g = guard().lock().unwrap_or_else(|e| e.into_inner());
    const W: u32 = 480; const H: u32 = 272; const GOP: u32 = 30;
    eprintln!("\n=== #797 true vs reported capacity (OH264) ===");
    eprintln!("fixture       frames | reported(T0) | true_max | ratio");
    for (src, tag) in [("Artlist_HorseFlag.mp4", "horse_flag"),
                       ("lumix_g9_1080p_30fps_h264_high.mp4", "lumix"),
                       ("Artlist_CarPlane.mp4", "carplane")] {
        for &n in &[30u32, 60] {
            let Some(yuv) = try_scale_yuv(src, W, H, n) else {
                eprintln!("{:<13} {:>6} | SKIP", tag, n); continue;
            };
            let reported = h264_stego_capacity_4domain_oh264(&yuv, W, H, n as usize,
                EncodeOpts { qp: 26, intra_period: GOP as i32 })
                .map(|i| i.per_tier_primary_max_message_bytes[0]).unwrap_or(0);
            let true_max = max_encodable(&yuv, W, H, n, GOP);
            let ratio = if true_max > 0 { reported as f64 / true_max as f64 } else { f64::NAN };
            eprintln!("{:<13} {:>6} | {:>12} | {:>8} | {:.2}x", tag, n, reported, true_max, ratio);
        }
    }
}

#[test]
#[ignore]
fn horse_flag_capacity_vs_frames() {
    let _g = guard().lock().unwrap_or_else(|e| e.into_inner());
    const W: u32 = 480; const H: u32 = 272; const GOP: u32 = 30;
    let msg = "tier sweep — real corpus payload, ~50 bytes long padding"; // same as D'.9 sweep
    let pass = "hf797-pass";
    eprintln!("\n=== #797 horse_flag capacity vs frame count (msg={} B) ===", msg.len());
    eprintln!("frames |  reported_cap(OH264 T0)  |  encode            |  decode");
    for &n in &[30u32, 60, 90, 120, 150] {
        let Some(yuv) = try_scale_yuv("Artlist_HorseFlag.mp4", W, H, n) else {
            eprintln!("{:>6} | SKIP (no fixture/ffmpeg)", n); continue;
        };
        let cap = h264_stego_capacity_4domain_oh264(&yuv, W, H, n as usize,
            EncodeOpts { qp: 26, intra_period: GOP as i32 })
            .map(|i| i.per_tier_primary_max_message_bytes[0] as i64)
            .unwrap_or(-1);
        let enc = encode(&yuv, W, H, n, GOP, msg, pass);
        let (enc_s, dec_s) = match &enc {
            Ok(b) => ("OK".to_string(), match decode(b, pass) {
                Ok(t) if t == msg => "OK".to_string(),
                Ok(t) => format!("WRONG({:?})", &t[..t.len().min(16)]),
                Err(e) => format!("FAIL({e})"),
            }),
            Err(e) => (format!("FAIL({e})"), "n/a".to_string()),
        };
        eprintln!("{:>6} | {:>22} | {:<18} | {}", n, cap, enc_s, dec_s);
    }
}

/// #797 — realistic-cover capacity check. The 480×272 gate is a
/// deliberately pathological small cover; real clips are 720p/1080p and
/// seconds long. Confirms lumix (which reads 0 at 480×272) has real
/// capacity at realistic size, and that capacity scales with duration.
#[test]
#[ignore]
fn realistic_capacity_check() {
    let _g = guard().lock().unwrap_or_else(|e| e.into_inner());
    const W: u32 = 1280; // 720p, 16-aligned (1280%16==0, 720%16==0)
    const H: u32 = 720;
    const GOP: u32 = 30;
    let rep = |yuv: &[u8], n: u32| -> usize {
        h264_stego_capacity_4domain_oh264(yuv, W, H, n as usize,
            EncodeOpts { qp: 26, intra_period: GOP as i32 })
            .map(|i| i.per_tier_primary_max_message_bytes[0]).unwrap_or(0)
    };

    eprintln!("\n=== #797 realistic capacity (1280x720, gop=30) ===");
    eprintln!("fixture       frames |  sec | reported(T0) | true_max | ratio");
    for (src, tag) in [("lumix_g9_1080p_30fps_h264_high.mp4", "lumix"),
                       ("Artlist_HorseFlag.mp4", "horse_flag"),
                       ("Artlist_CarPlane.mp4", "carplane")] {
        let n = 90u32;
        let Some(yuv) = try_scale_yuv(src, W, H, n) else {
            eprintln!("{:<13} {:>6} | SKIP (no fixture)", tag, n); continue;
        };
        let reported = rep(&yuv, n);
        let true_max = max_encodable(&yuv, W, H, n, GOP);
        let ratio = if true_max > 0 { reported as f64 / true_max as f64 } else { f64::NAN };
        eprintln!("{:<13} {:>6} | {:>4.1} | {:>12} | {:>8} | {:.2}x",
            tag, n, n as f64 / 30.0, reported, true_max, ratio);
    }

    eprintln!("\n=== lumix capacity vs duration (1280x720, reported T0) ===");
    eprintln!("frames |  sec | reported(T0)");
    for &n in &[30u32, 90, 150, 300] {
        let Some(yuv) = try_scale_yuv("lumix_g9_1080p_30fps_h264_high.mp4", W, H, n) else {
            eprintln!("{:>6} | SKIP", n); continue;
        };
        eprintln!("{:>6} | {:>4.1} | {:>12}", n, n as f64 / 30.0, rep(&yuv, n));
    }
}
