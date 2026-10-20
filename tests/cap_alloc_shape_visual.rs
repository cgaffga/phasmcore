// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Allocation-SHAPE **visual fidelity** comparison — per-GOP, per-arm.
//!
//! ## Why this exists separately from the stealth sweep
//!
//! `h264_alloc_shape_sweep` answered a *steganalysis* question: can a W6
//! detector tell cover from stego under different allocation shapes? Answer:
//! no, flat at AUC ~0.502 even when an entire payload lands in ONE GOP
//! (gini 0.70, max share 1.00).
//!
//! That says nothing about how the video **looks**. Those are different
//! questions and a flat AUC is not evidence of visual transparency:
//!
//! - A sign flip is a coefficient change, so its pixel cost is
//!   `delta = 2 x |coeff| x Q-step x IDCT_basis`. Concentrating payload
//!   concentrates that distortion.
//! - Worse, it concentrates it *in time*. One GOP is ~1 s at 30 fps. A
//!   single second of elevated distortion between clean neighbours reads to
//!   a human as pumping / breathing — temporal inconsistency is far more
//!   noticeable than a uniform small penalty, and whole-clip average PSNR
//!   averages it away completely.
//!
//! The project treats this as correctness, not polish: visual fidelity is a
//! ship gate, and "encoder reconstructs the cover" IS correctness.
//!
//! ## The hypothesis being tested
//!
//! Per-GOP capacity tracks the usable bypass-bin population, which tracks
//! texture and motion — and texture is exactly what *masks* artifacts
//! (the J-UNIWARD premise). That predicts an ordering:
//!
//! - **uniform** forces a smooth, low-capacity GOP to carry the same BYTE
//!   count as a textured one. Smooth regions mask worst, so uniform should
//!   be perceptually the *worst* despite distributing bytes "evenly".
//! - **proportional** fills every GOP to the same FRACTION of its own cap,
//!   which equalises the per-GOP distortion budget rather than the byte
//!   count. Predicted best.
//! - **concentrate** maximises both total and temporal concentration.
//!   Predicted worst on the per-GOP spread metric even if its average is
//!   fine.
//!
//! Equalising bytes is the wrong invariant; equalising the fraction of each
//! GOP's own cost budget is the right one. This gate measures whether that
//! reasoning survives contact with real video.
//!
//! ## Method
//!
//! Matched payload across arms, sized to 80% of `n_gops x min(cap)` so the
//! uniform arm can actually carry it (above that it fails outright — the
//! #841 defect — and there would be nothing to compare). Each arm installs
//! an explicit plan, encodes, is decoded by ffmpeg as an independent
//! reference, and is measured per GOP with the SAME four metrics the
//! shipped visual gate uses (`common::visual_metrics`).
//!
//! Headline number is **`spread`**: max minus min of per-GOP mean Y-PSNR
//! within a clip. That is the pumping metric. Averages are reported too,
//! for comparability with the shipped gate.
//!
//! Run:
//!   cargo test --test cap_alloc_shape_visual --features h264-encoder \
//!     --release -- --ignored --nocapture --test-threads=1

#![cfg(feature = "h264-encoder")]

mod common;
use common::visual_metrics::{count_bad_pixels, psnr_y, ssim_y, worst_mb_psnr_y};

use phasm_core::codec::h264::openh264_stego::EncodeOpts;
use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingEncodeSession,
    StreamingProbeSession, YuvFrameRef,
};
use phasm_core::codec::h264::stego::CostWeights;
use phasm_core::stego::chunk_frame::{allocate_chunks_proportional, CAP2_DEFAULT_R_TARGET};

const GOP: u32 = 30;
const QP: i32 = 26;
const BAD_PIXEL_DELTA: u8 = 80;

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn crop_yuv(src: &str, w: u32, h: u32, n: u32) -> Option<Vec<u8>> {
    let src_path = corpus_root().join(src);
    if !src_path.exists() {
        return None;
    }
    let key: String = src
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let yuv_path = format!(
        "{}/phasm_cap810_{}_{}x{}_f{}.yuv",
        common::scratch_dir(),
        key,
        w,
        h,
        n
    );
    let need = (w * h * 3 / 2) as usize * n as usize;
    if let Ok(d) = std::fs::read(&yuv_path) {
        if d.len() >= need {
            return Some(d[..need].to_vec());
        }
    }
    let vf = format!("crop={}:{}:0:0", w, h);
    let ok = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src_path)
        .args([
            "-frames:v", &n.to_string(), "-an", "-vf", &vf, "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
        ])
        .arg(&yuv_path)
        .status()
        .ok()?
        .success();
    if !ok {
        return None;
    }
    let d = std::fs::read(&yuv_path).ok()?;
    (d.len() >= need).then(|| d[..need].to_vec())
}

fn params(w: u32, h: u32, n: u32) -> EncodeSessionParams {
    EncodeSessionParams {
        width: w, height: h, fps_num: 30, fps_den: 1, qp: QP, gop_size: GOP,
        total_frames_hint: n, color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264, cost_weights: CostWeights::default(),
        progress_callback: None,
    }
}

fn probe_caps(yuv: &[u8], w: u32, h: u32, n: u32) -> Option<Vec<usize>> {
    let mut p = StreamingProbeSession::create(params(w, h, n)).ok()?;
    let fb = (w * h * 3 / 2) as usize;
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let cw = (w / 2) as usize;
    for f in 0..n as usize {
        let off = f * fb;
        p.push_frame(YuvFrameRef {
            y: &yuv[off..off + ys], y_stride: w as usize,
            u: &yuv[off + ys..off + ys + cs], u_stride: cw,
            v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
        })
        .ok()?;
    }
    p.finish_with_per_gop().ok().map(|(_r, caps)| caps)
}

/// Deterministic, effectively incompressible payload, so the framed length
/// tracks the raw length and the arms genuinely carry the same bytes.
fn payload(len: usize) -> String {
    let mut s = String::with_capacity(len);
    let mut x: u64 = 0x5DEECE66D;
    for _ in 0..len {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s.push((33 + ((x >> 33) % 94) as u8) as char);
    }
    s
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Shape { Uniform, Proportional, Concentrate }

impl Shape {
    fn label(self) -> &'static str {
        match self {
            Shape::Uniform => "uniform",
            Shape::Proportional => "proportional",
            Shape::Concentrate => "concentrate",
        }
    }
}

/// Build each shape's explicit per-GOP byte plan over `framed` bytes.
/// Explicit for every arm, including uniform: that pins what is being
/// compared instead of relying on the drain's reactive even-split-with-carry.
fn plan_for(shape: Shape, caps: &[usize], framed: usize) -> Option<Vec<usize>> {
    let n = caps.len();
    match shape {
        Shape::Uniform => {
            let base = framed / n;
            let rem = framed % n;
            Some((0..n).map(|i| base + usize::from(i < rem)).collect())
        }
        Shape::Proportional => {
            allocate_chunks_proportional(caps, framed, CAP2_DEFAULT_R_TARGET)
        }
        Shape::Concentrate => {
            // Mirror of `allocate_chunks_concentrate_tail`: fill each GOP in
            // order to floor(r*cap), widening to full cap past the soft
            // ceiling, leaving a plain tail.
            let soft: usize = caps.iter().map(|&c| (CAP2_DEFAULT_R_TARGET * c as f64) as usize).sum();
            let use_full = framed > soft;
            let mut plan = vec![0usize; n];
            let mut left = framed;
            for (i, &c) in caps.iter().enumerate() {
                if left == 0 { break; }
                let room = if use_full { c } else { (CAP2_DEFAULT_R_TARGET * c as f64) as usize };
                let take = room.min(left);
                plan[i] = take;
                left -= take;
            }
            (left == 0).then_some(plan)
        }
    }
}

fn encode_with_plan(
    yuv: &[u8], w: u32, h: u32, n: u32, msg: &str, pass: &str, plan: Vec<usize>,
) -> Result<Vec<u8>, String> {
    let mut enc = StreamingEncodeSession::create(params(w, h, n), msg, pass)
        .map_err(|e| format!("create: {e:?}"))?;
    enc.set_gop_alloc_plan(plan)
        .map_err(|e| format!("set_plan: {e:?}"))?;
    let fb = (w * h * 3 / 2) as usize;
    let (ys, cs) = ((w * h) as usize, (w / 2 * h / 2) as usize);
    let cw = (w / 2) as usize;
    let mut out = Vec::new();
    for f in 0..n as usize {
        let off = f * fb;
        enc.push_frame(
            YuvFrameRef {
                y: &yuv[off..off + ys], y_stride: w as usize,
                u: &yuv[off + ys..off + ys + cs], u_stride: cw,
                v: &yuv[off + ys + cs..off + ys + 2 * cs], v_stride: cw,
            },
            &mut out,
        )
        .map_err(|e| format!("push #{f}: {e:?}"))?;
    }
    enc.finish(&mut out).map_err(|e| format!("finish: {e:?}"))?;
    Ok(out)
}

/// ffmpeg-decode the Annex-B as an independent reference. Scratch goes to
/// `$PHASM_TMP` and is removed afterwards — a decoded 1080p YUV is ~90 MB
/// per arm per fixture and this gate writes one for each.
fn ffmpeg_decode(annex_b: &[u8], tag: &str, expect_len: usize) -> Option<Vec<u8>> {
    let base = common::scratch_dir();
    let h264 = format!("{base}/phasm_shape_vis_{tag}.h264");
    let yuv = format!("{base}/phasm_shape_vis_{tag}.dec.yuv");
    std::fs::write(&h264, annex_b).ok()?;
    let ok = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate", "30", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv)
        .status()
        .ok()?
        .success();
    let decoded = if ok { std::fs::read(&yuv).ok() } else { None };
    let _ = std::fs::remove_file(&h264);
    let _ = std::fs::remove_file(&yuv);
    decoded.filter(|d| d.len() == expect_len)
}

struct GopMetrics {
    mean_psnr: f64,
    min_psnr: f64,
    mean_ssim: f64,
    min_worst_mb: f64,
    max_bad: u32,
    planned: usize,
}

#[test]
#[ignore]
fn cap_alloc_shape_visual_fidelity() {
    // iphone7 first (the #841 fixture, 17 GOPs, min/mu = 0.358) then two
    // shorter clips so the run stays affordable.
    let fixtures: [(&str, &str, u32, u32, u32); 3] = [
        ("iphone7", "iphone7_1080p_30fps_h264_high.mov", 1920, 1072, 510),
        ("carplane", "Artlist_CarPlane.mp4", 1072, 1920, 180),
        ("woman_subway", "Artlist_WomanSubway.mp4", 1280, 720, 360),
    ];
    let mut failures: Vec<String> = Vec::new();
    let mut verdicts: Vec<String> = Vec::new();

    for (tag, file, w, h, n) in fixtures {
        let Some(yuv) = crop_yuv(file, w, h, n) else {
            eprintln!("SKIP {tag} — no fixture");
            continue;
        };
        let Some(caps) = probe_caps(&yuv, w, h, n) else {
            failures.push(format!("{tag}: probe failed"));
            continue;
        };
        let n_gops = caps.len();
        let min_cap = caps.iter().copied().min().unwrap_or(0);
        let sigma: usize = caps.iter().sum();

        // Matched payload all three arms can carry: 80% of n x min.
        let target = (n_gops * min_cap) * 80 / 100;
        if target < 64 {
            eprintln!("SKIP {tag} — n x min too small ({target} B)");
            continue;
        }
        let msg = payload(target);
        let pass = format!("shape_vis_{tag}");

        let framed = match StreamingEncodeSession::create(params(w, h, n), &msg, &pass) {
            Ok(s) => s.framed_message_len().unwrap_or(0),
            Err(e) => { failures.push(format!("{tag}: framing: {e:?}")); continue; }
        };

        eprintln!(
            "\n=== {tag} {w}x{h}x{n}f | {n_gops} GOPs | Sigma={sigma} min={min_cap} \
             min/mu={:.3} | payload framed={framed} B ({:.0}% of Sigma) ===",
            min_cap as f64 / (sigma as f64 / n_gops as f64),
            100.0 * framed as f64 / sigma as f64,
        );

        for shape in [Shape::Uniform, Shape::Proportional, Shape::Concentrate] {
            let Some(plan) = plan_for(shape, &caps, framed) else {
                failures.push(format!("{tag}/{}: no plan for {framed} B", shape.label()));
                continue;
            };
            let over: Vec<String> = plan.iter().zip(caps.iter()).enumerate()
                .filter(|&(_, (p, c))| p > c)
                .map(|(i, (p, c))| format!("GOP{i} {p}>{c}"))
                .collect();

            let stego = match encode_with_plan(&yuv, w, h, n, &msg, &pass, plan.clone()) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "  {:<13} ENCODE FAILED: {e}{}",
                        shape.label(),
                        if over.is_empty() { String::new() }
                        else { format!("  [over-allocated: {}]", over.join(", ")) },
                    );
                    continue;
                }
            };
            let Some(dec) = ffmpeg_decode(&stego, &format!("{tag}_{}", shape.label()), yuv.len())
            else {
                failures.push(format!("{tag}/{}: ffmpeg decode failed", shape.label()));
                continue;
            };

            // Per-GOP metrics over the luma plane.
            let fb = (w * h * 3 / 2) as usize;
            let ys = (w * h) as usize;
            let mut per_gop: Vec<GopMetrics> = Vec::new();
            for g in 0..n_gops {
                let f0 = g * GOP as usize;
                let f1 = ((g + 1) * GOP as usize).min(n as usize);
                if f0 >= f1 { break; }
                let (mut sp, mut ss) = (0.0f64, 0.0f64);
                let mut mn = f64::INFINITY;
                let mut wmb = f64::INFINITY;
                let mut bad = 0u32;
                for f in f0..f1 {
                    let off = f * fb;
                    let a = &yuv[off..off + ys];
                    let b = &dec[off..off + ys];
                    let p = psnr_y(a, b);
                    sp += p;
                    mn = mn.min(p);
                    ss += ssim_y(a, b, w, h);
                    wmb = wmb.min(worst_mb_psnr_y(a, b, w, h).0);
                    bad = bad.max(count_bad_pixels(a, b, BAD_PIXEL_DELTA));
                }
                let k = (f1 - f0) as f64;
                per_gop.push(GopMetrics {
                    mean_psnr: sp / k, min_psnr: mn, mean_ssim: ss / k,
                    min_worst_mb: wmb, max_bad: bad,
                    planned: plan.get(g).copied().unwrap_or(0),
                });
            }

            let means: Vec<f64> = per_gop.iter().map(|m| m.mean_psnr).collect();
            let hi = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lo = means.iter().cloned().fold(f64::INFINITY, f64::min);
            let spread = hi - lo;
            let avg = means.iter().sum::<f64>() / means.len() as f64;
            let worst_mb = per_gop.iter().map(|m| m.min_worst_mb)
                .fold(f64::INFINITY, f64::min);
            let min_ssim = per_gop.iter().map(|m| m.mean_ssim)
                .fold(f64::INFINITY, f64::min);
            let max_bad = per_gop.iter().map(|m| m.max_bad).max().unwrap_or(0);

            eprintln!(
                "  {:<13} avgPSNR={avg:6.2}  SPREAD={spread:5.2} dB  minGOP={lo:6.2}  \
                 worstMB={worst_mb:5.2}  minSSIM={min_ssim:.4}  maxBad={max_bad}",
                shape.label(),
            );
            eprintln!(
                "  {:<13}   per-GOP mean PSNR: {}",
                "",
                per_gop.iter()
                    .map(|m| format!("{:.1}", m.mean_psnr))
                    .collect::<Vec<_>>().join(" "),
            );
            eprintln!(
                "  {:<13}   per-GOP bytes:     {}",
                "",
                per_gop.iter().map(|m| m.planned.to_string())
                    .collect::<Vec<_>>().join(" "),
            );
            verdicts.push(format!(
                "{tag:<14} {:<13} avg={avg:.2} spread={spread:.2} worstMB={worst_mb:.2}",
                shape.label(),
            ));
        }
    }

    eprintln!("\n=== summary (SPREAD = max-min of per-GOP mean PSNR = the pumping metric) ===");
    for v in &verdicts { eprintln!("  {v}"); }
    if !failures.is_empty() {
        eprintln!("\nFAILURES:");
        for f in &failures { eprintln!("  {f}"); }
    }
    assert!(
        !verdicts.is_empty(),
        "no arm produced measurements:\n  {}",
        failures.join("\n  "),
    );
}
