// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// v1.4 Phase 8 (#311) — multi-ref carplane visual gate.
//
// Encodes the carplane fixture (1072×1920 IBPBP × 30 frames at QP=26)
// twice:
//   * SINGLE_REF baseline (= v1.3 ship behaviour, multi_ref_config
//     = SINGLE_REF, no post-pass).
//   * DUAL_REF_L0 (= Path B v1.4-cut1 post-pass active on P + B-frame
//     L0_16x16/Bi_16x16).
//
// Both runs use BRdoConfig::PRODUCTION_VISUAL (full RDO + residual)
// because Path B's multi-ref post-pass only matters on content-driven
// mode-decision; SAFE_L0_ZERO (current lib default) forces every
// B-MB to L0_16x16/MV=(0,0) and would heavily limit upgrade rate.
//
// Per-frame Y/U/V PSNR + SSIM via ffmpeg psnr/ssim filters. Renders
// both MP4s to Desktop. Reports avg/min per frame and per-frame
// breakdown across the 30-frame audit window so user can A/B
// visually + pinpoint where multi-ref helps (high-motion mid-GOP B
// frames are the predicted-benefit class).
//
// Usage:
//   cargo test --release -p phasm-core --features cabac-stego \
//     --test h264_v14_multi_ref_carplane_demo \
//     -- --ignored --nocapture

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode, MultiRefConfig};
use phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig;
use phasm_core::codec::h264::stego::gop_pattern::{
    iter_encode_order, FrameType, GopPattern,
};
use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

const N_FRAMES: usize = 30;
const QP: u8 = 26;
const FIXTURE_W: u32 = 1072;
const FIXTURE_H: u32 = 1920;
const FIXTURE_NAME: &str = "carplane";
const SOURCE_MP4: &str = "Artlist_CarPlane.mp4";

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn ensure_yuv() -> Vec<u8> {
    let yuv_path = format!(
        "/tmp/phasm_v14_carplane_{}x{}_f{}.yuv",
        FIXTURE_W, FIXTURE_H, N_FRAMES
    );
    let frame_size = (FIXTURE_W * FIXTURE_H * 3 / 2) as usize;
    let need_bytes = frame_size * N_FRAMES;
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need_bytes {
            return data;
        }
    }
    let src = corpus_root().join(SOURCE_MP4);
    let crop = format!("crop={}:{}:0:0", FIXTURE_W, FIXTURE_H);
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &N_FRAMES.to_string()])
        .args(["-an", "-vf", &crop])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("ffmpeg launch");
    assert!(status.success(), "ffmpeg crop→yuv failed for {SOURCE_MP4}");
    std::fs::read(&yuv_path).expect("read yuv after regen")
}

fn clear_b_env() {
    unsafe {
        std::env::remove_var("PHASM_B_TEMPORAL_DIRECT");
        std::env::remove_var("PHASM_B_BOUNDARY_PENALTY");
        std::env::remove_var("PHASM_B_TEMPORAL_CAND");
        std::env::remove_var("PHASM_B_MULTI_REFINE");
        std::env::remove_var("PHASM_B_DIRECT_VALIDATE");
        std::env::remove_var("PHASM_B_NO_BOUNDARY_REFUSE");
        std::env::remove_var("PHASM_B_NO_DIRECT_MAGCLAMP");
        std::env::remove_var("PHASM_B_NO_ME_RESULT_CLAMP");
        std::env::remove_var("PHASM_B_INSTRUMENT");
        std::env::remove_var("PHASM_B_FORCE_MODE");
        std::env::set_var("PHASM_DISABLE_SCENECUT", "1");
    }
}

struct Run {
    label: &'static str,
    multi_ref: MultiRefConfig,
}

const SINGLE_REF_RUN: Run = Run {
    label: "single_ref",
    multi_ref: MultiRefConfig::SINGLE_REF,
};
const DUAL_REF_L0_RUN: Run = Run {
    label: "dual_ref_l0",
    multi_ref: MultiRefConfig::DUAL_REF_L0,
};

struct RunResult {
    label: &'static str,
    encode_ms: u128,
    bitstream_bytes: usize,
    psnr_per_frame: Vec<(f64, f64, f64)>, // (y, u, v)
    ssim_y_per_frame: Vec<f64>,
    desktop_mp4: String,
    decoded_yuv: Vec<u8>,
}

fn run_encode(yuv: &[u8], run: &Run) -> RunResult {
    clear_b_env();
    let frame_size = (FIXTURE_W * FIXTURE_H * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(N_FRAMES);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    let mut enc = Encoder::new(FIXTURE_W, FIXTURE_H, Some(QP)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    // PRODUCTION_VISUAL = full RDO + residual. Path B's multi-ref
    // post-pass needs content-driven mode-decision to have an
    // upgradeable L0 MV. SAFE_L0_ZERO would force MV=(0,0) and the
    // post-pass would still fire, but with a fixed-point seed the
    // upgrade rate would be artificially low.
    enc.b_rdo_config = BRdoConfig::PRODUCTION_VISUAL;
    enc.multi_ref_config = run.multi_ref;
    // §B-fast-motion (#317) — AQ-mode-3 (auto-variance + dark bias)
    // strength 1.0 (Q10=1024) gives +1.99 dB Y-PSNR / +0.016 SSIM /
    // 5.3× fewer max|Δ|≥12 deviated MBs vs default AQ-1 strength 1.0
    // on this 1080p×30f IBPBP fixture (carplane). Trade-off is +50%
    // bitrate. Acceptable for visual-quality demos. Library default
    // stays AQ-1 (matches x264-medium centroid for L4 stealth).
    unsafe {
        std::env::set_var("PHASM_AQ_MODE", "3");
        std::env::set_var("PHASM_AQ_STRENGTH_Q10", "1024");
    }

    // §v1.7 Phase 3 (#325) — opt-in CRF mode. PHASM_CRF=N anchors
    // the per-frame base CRF target. Lookahead offset (if enabled)
    // modulates it per-frame; frame-type +0/+1/+2 applied downstream.
    if let Some(v) = std::env::var_os("PHASM_CRF") {
        let crf: u8 = v
            .to_string_lossy()
            .parse()
            .expect("PHASM_CRF must be an integer 0..=51");
        enc.crf = Some(crf);
        eprintln!("[{}] §CRF: anchor={}", run.label, crf);
    }

    // §v1.7 Phase 2.1 (#324) — opt-in lookahead frame-level QP offset.
    if std::env::var_os("PHASM_LOOKAHEAD").is_some() {
        let strength: u32 = std::env::var("PHASM_LOOKAHEAD_STRENGTH")
            .ok().and_then(|s| s.parse().ok())
            .unwrap_or(phasm_core::codec::h264::encoder::lookahead::DEFAULT_STRENGTH);
        let la = phasm_core::codec::h264::encoder::lookahead::analyze_lookahead_window(
            yuv, FIXTURE_W, FIXTURE_H, n_frames, strength,
        );
        eprintln!("[{}] §Lookahead: mean_complexity={} per-frame offsets={:?}",
                  run.label, la.mean_complexity, la.per_frame_qp_offset);
        enc.lookahead = Some(la);
    }

    // §v1.7 Phase 1.1 (#323) — opt-in MB-tree per-MB QP offset.
    // PHASM_MBTREE=1 enables; PHASM_MBTREE_STRENGTH=N overrides default.
    if std::env::var_os("PHASM_MBTREE").is_some() {
        let strength: u32 = std::env::var("PHASM_MBTREE_STRENGTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(phasm_core::codec::h264::encoder::mb_tree::DEFAULT_STRENGTH);
        let lookahead: usize = std::env::var("PHASM_MBTREE_LOOKAHEAD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(phasm_core::codec::h264::encoder::mb_tree::DEFAULT_LOOKAHEAD);
        let mb_tree = phasm_core::codec::h264::encoder::mb_tree::compute_mb_tree_qp_offsets(
            yuv, FIXTURE_W, FIXTURE_H, n_frames, strength, lookahead,
        );
        let hist = phasm_core::codec::h264::encoder::mb_tree::offset_histogram(&mb_tree);
        eprintln!("[{}] §MB-tree: {} non-zero offset buckets, distribution:",
                  run.label, hist.iter().filter(|&(&k, _)| k != 0).count());
        for (offset, count) in &hist {
            eprintln!("  offset={:+3} count={}", offset, count);
        }
        enc.mb_tree = Some(mb_tree);
    }

    let t0 = std::time::Instant::now();
    let mut bs = Vec::new();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        // §v1.7 Phase 1.1 — feed display index to encoder for MB-tree lookup.
        enc.mb_tree_display_idx = d;
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("[{}] encode error: {e}", run.label));
        bs.extend_from_slice(&bytes);
    }
    let encode_ms = t0.elapsed().as_millis();

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264,
        &bs,
        FIXTURE_W,
        FIXTURE_H,
        timing,
        pattern,
        n_frames,
    )
    .expect("mp4 mux");
    let desktop_mp4 = format!(
        "/Users/cgaffga/Desktop/phasm_v14_{}_{}_qp{}.mp4",
        FIXTURE_NAME, run.label, QP
    );
    std::fs::write(&desktop_mp4, &mp4).expect("write demo mp4");

    let h264 = std::env::temp_dir()
        .join(format!("phasm_v14_{}_{}.h264", FIXTURE_NAME, run.label));
    let dec = std::env::temp_dir()
        .join(format!("phasm_v14_{}_{}.dec.yuv", FIXTURE_NAME, run.label));
    std::fs::write(&h264, &bs).unwrap();
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec)
        .status()
        .expect("ffmpeg decode");
    assert!(status.success(), "ffmpeg decode failed for {}", run.label);
    let decoded = std::fs::read(&dec).expect("read decoded yuv");

    let bitstream_bytes = bs.len();
    let (psnr_per_frame, ssim_y_per_frame) =
        compute_psnr_ssim(yuv, &decoded, n_frames, run.label);

    let _ = std::fs::remove_file(&h264);
    // Keep decoded yuv for cross-run frame comparison; cleaned by caller.
    RunResult {
        label: run.label,
        encode_ms,
        bitstream_bytes,
        psnr_per_frame,
        ssim_y_per_frame,
        desktop_mp4,
        decoded_yuv: decoded,
    }
}

fn compute_psnr_ssim(
    src_yuv: &[u8],
    dec_yuv: &[u8],
    audit_frames: usize,
    label: &str,
) -> (Vec<(f64, f64, f64)>, Vec<f64>) {
    let frame_size = (FIXTURE_W * FIXTURE_H * 3 / 2) as usize;
    let need = audit_frames * frame_size;
    let src_path = std::env::temp_dir()
        .join(format!("phasm_v14_{}_{}_src_metrics.yuv", FIXTURE_NAME, label));
    let dec_path = std::env::temp_dir()
        .join(format!("phasm_v14_{}_{}_dec_metrics.yuv", FIXTURE_NAME, label));
    std::fs::write(&src_path, &src_yuv[..need.min(src_yuv.len())])
        .expect("write src yuv for metrics");
    std::fs::write(&dec_path, &dec_yuv[..need.min(dec_yuv.len())])
        .expect("write dec yuv for metrics");

    let psnr_log = std::env::temp_dir()
        .join(format!("phasm_v14_{}_{}_psnr.log", FIXTURE_NAME, label));
    let ssim_log = std::env::temp_dir()
        .join(format!("phasm_v14_{}_{}_ssim.log", FIXTURE_NAME, label));
    let _ = std::fs::remove_file(&psnr_log);
    let _ = std::fs::remove_file(&ssim_log);

    let size_arg = format!("{}x{}", FIXTURE_W, FIXTURE_H);
    let lavfi = format!(
        "[0:v][1:v]psnr=stats_file={};[0:v][1:v]ssim=stats_file={}",
        psnr_log.display(),
        ssim_log.display(),
    );
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "yuv420p",
               "-s", &size_arg])
        .arg("-i").arg(&dec_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p",
               "-s", &size_arg])
        .arg("-i").arg(&src_path)
        .args(["-lavfi", &lavfi])
        .args(["-f", "null", "-"])
        .status()
        .expect("ffmpeg psnr/ssim launch");

    let psnr_text = std::fs::read_to_string(&psnr_log).unwrap_or_default();
    let ssim_text = std::fs::read_to_string(&ssim_log).unwrap_or_default();

    fn parse_field(line: &str, key: &str) -> Option<f64> {
        for tok in line.split_whitespace() {
            if let Some(rest) = tok.strip_prefix(key) {
                if rest == "inf" {
                    return Some(99.0);
                }
                return rest.parse().ok();
            }
        }
        None
    }

    let psnr_per_frame: Vec<(f64, f64, f64)> = psnr_text
        .lines()
        .filter_map(|line| {
            let y = parse_field(line, "psnr_y:")?;
            let u = parse_field(line, "psnr_u:")?;
            let v = parse_field(line, "psnr_v:")?;
            Some((y, u, v))
        })
        .collect();

    let ssim_per_frame: Vec<f64> = ssim_text
        .lines()
        .filter_map(|line| parse_field(line, "Y:"))
        .collect();

    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&dec_path);
    let _ = std::fs::remove_file(&psnr_log);
    let _ = std::fs::remove_file(&ssim_log);

    (psnr_per_frame, ssim_per_frame)
}

fn aggregate_psnr(per_frame: &[(f64, f64, f64)]) -> (f64, f64, f64, f64) {
    if per_frame.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let n = per_frame.len() as f64;
    let y_mean = per_frame.iter().map(|p| p.0).sum::<f64>() / n;
    let u_mean = per_frame.iter().map(|p| p.1).sum::<f64>() / n;
    let v_mean = per_frame.iter().map(|p| p.2).sum::<f64>() / n;
    let y_min = per_frame.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    (y_mean, u_mean, v_mean, y_min)
}

#[test]
#[ignore]
fn v14_multi_ref_carplane_demo() {
    eprintln!("\n=== v1.4 Phase 8 (#311) — multi-ref carplane visual gate ===");
    eprintln!("Fixture: {} ({}x{} × {} frames, IBPBP gop=30 b=1, QP={}, PRODUCTION_VISUAL)",
              FIXTURE_NAME, FIXTURE_W, FIXTURE_H, N_FRAMES, QP);

    let yuv = ensure_yuv();
    let frame_size = (FIXTURE_W * FIXTURE_H * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(N_FRAMES);

    eprintln!("\n--- run 1: SINGLE_REF (v1.3 baseline) ---");
    let r0 = run_encode(&yuv, &SINGLE_REF_RUN);
    eprintln!("  encode: {}ms, bitstream: {} bytes", r0.encode_ms, r0.bitstream_bytes);
    eprintln!("  desktop: {}", r0.desktop_mp4);

    eprintln!("\n--- run 2: DUAL_REF_L0 (Path B v1.4-cut1) ---");
    let r1 = run_encode(&yuv, &DUAL_REF_L0_RUN);
    eprintln!("  encode: {}ms, bitstream: {} bytes", r1.encode_ms, r1.bitstream_bytes);
    eprintln!("  desktop: {}", r1.desktop_mp4);

    let (y0, u0, v0, ymin0) = aggregate_psnr(&r0.psnr_per_frame);
    let (y1, u1, v1, ymin1) = aggregate_psnr(&r1.psnr_per_frame);
    let s0 = if r0.ssim_y_per_frame.is_empty() {
        f64::NAN
    } else {
        r0.ssim_y_per_frame.iter().sum::<f64>() / r0.ssim_y_per_frame.len() as f64
    };
    let s1 = if r1.ssim_y_per_frame.is_empty() {
        f64::NAN
    } else {
        r1.ssim_y_per_frame.iter().sum::<f64>() / r1.ssim_y_per_frame.len() as f64
    };

    eprintln!("\n=== Comparison ===");
    eprintln!(
        "{:<14}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}  {:>9}",
        "run", "y_mean", "u_mean", "v_mean", "y_min", "ssim_y", "encode_ms", "bytes"
    );
    eprintln!(
        "{:<14}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.5}  {:>10}  {:>9}",
        r0.label, y0, u0, v0, ymin0, s0, r0.encode_ms, r0.bitstream_bytes
    );
    eprintln!(
        "{:<14}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.5}  {:>10}  {:>9}",
        r1.label, y1, u1, v1, ymin1, s1, r1.encode_ms, r1.bitstream_bytes
    );
    let dy = y1 - y0;
    let du = u1 - u0;
    let dv = v1 - v0;
    let dymin = ymin1 - ymin0;
    let dssim = s1 - s0;
    let dwall = r1.encode_ms as f64 / r0.encode_ms.max(1) as f64;
    let dbytes = r1.bitstream_bytes as f64 / r0.bitstream_bytes.max(1) as f64;
    eprintln!(
        "{:<14}  {:>+8.3}  {:>+8.3}  {:>+8.3}  {:>+8.3}  {:>+8.5}  {:>9.2}x  {:>8.2}x",
        "Δ (B−A)", dy, du, dv, dymin, dssim, dwall, dbytes
    );

    eprintln!("\n=== Per-frame Y-PSNR (frame: SINGLE_REF / DUAL_REF_L0 / Δ) ===");
    let nf = r0.psnr_per_frame.len().min(r1.psnr_per_frame.len()).min(n_frames);
    for i in 0..nf {
        let y_a = r0.psnr_per_frame[i].0;
        let y_b = r1.psnr_per_frame[i].0;
        let delta = y_b - y_a;
        let marker = if delta > 0.05 {
            "+"
        } else if delta < -0.05 {
            "-"
        } else {
            " "
        };
        eprintln!(
            "  frame {:2}: {:6.2}  {:6.2}  {:>+6.3}  {}",
            i, y_a, y_b, delta, marker
        );
    }

    // Frame inspection: write three side-by-side comparison PNGs at
    // the worst-Δ frame (= frame where DUAL_REF_L0 helped or hurt
    // most). Source / SINGLE_REF / DUAL_REF_L0 — three PNGs to
    // /tmp so user can pixel-compare without re-running.
    let mut worst_helped_idx = 0usize;
    let mut worst_helped_delta = f64::NEG_INFINITY;
    let mut worst_hurt_idx = 0usize;
    let mut worst_hurt_delta = f64::INFINITY;
    for i in 0..nf {
        let d = r1.psnr_per_frame[i].0 - r0.psnr_per_frame[i].0;
        if d > worst_helped_delta {
            worst_helped_delta = d;
            worst_helped_idx = i;
        }
        if d < worst_hurt_delta {
            worst_hurt_delta = d;
            worst_hurt_idx = i;
        }
    }
    eprintln!("\n=== Worst-Δ frame inspection ===");
    eprintln!(
        "  most-helped frame {} (DUAL_REF_L0 +{:.3} dB on Y)",
        worst_helped_idx, worst_helped_delta
    );
    eprintln!(
        "  most-hurt   frame {} (DUAL_REF_L0 {:.3} dB on Y)",
        worst_hurt_idx, worst_hurt_delta
    );
    for (label, idx) in [("helped", worst_helped_idx), ("hurt", worst_hurt_idx)] {
        for (rrun_label, rrun_yuv) in [
            ("source", &yuv),
            (r0.label, &r0.decoded_yuv),
            (r1.label, &r1.decoded_yuv),
        ] {
            let png_path = format!(
                "/tmp/phasm_v14_{}_{}_frame{}_{}.png",
                FIXTURE_NAME, label, idx, rrun_label
            );
            let off = idx * frame_size;
            let frame = &rrun_yuv[off..off + frame_size];
            let yuv_path = format!("/tmp/phasm_v14_frame_extract.yuv");
            std::fs::write(&yuv_path, frame).expect("write frame yuv");
            let size_arg = format!("{}x{}", FIXTURE_W, FIXTURE_H);
            let _ = std::process::Command::new("ffmpeg")
                .args(["-y", "-loglevel", "error",
                       "-f", "rawvideo", "-pix_fmt", "yuv420p",
                       "-s", &size_arg])
                .arg("-i").arg(&yuv_path)
                .args(["-frames:v", "1"])
                .arg(&png_path)
                .status()
                .expect("ffmpeg yuv→png");
            eprintln!("  → {}", png_path);
        }
    }

    eprintln!("\n=== Desktop demos ===");
    eprintln!("  SINGLE_REF:  {}", r0.desktop_mp4);
    eprintln!("  DUAL_REF_L0: {}", r1.desktop_mp4);
    eprintln!("\nPath B verdict gate — Δ Y ≥ +0.3 dB → ship as v1.4. \
0.1–0.3 dB → trade-off call. <0.1 dB → consider Path A.\n");
}
