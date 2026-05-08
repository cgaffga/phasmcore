// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// #255 v1.0 ship-readiness — V26 validation on broader fixture corpus.
//
// One #[test] per fixture so cargo's parallel test runner spreads the
// encode workload across all CPU cores. Each fixture:
//   1. Crops MP4 → YUV at largest 16-aligned dim (cached at /tmp).
//   2. Encodes 30 frames IBPBP gop=30 at QP=26 with V26 default.
//   3. Builds mp4 + writes demo to ~/Desktop/phasm_corpus_<name>_v26.mp4
//      for visual review.
//   4. Decodes via ffmpeg and audits per-MB max-deviation.
//   5. Writes /tmp/phasm_corpus_<name>_report.txt with one-line summary.
//
// `cargo test --release -p phasm-core --features cabac-stego \
//      --test h264_corpus_v26_validation -- --ignored --nocapture`
//
// To aggregate results after the corpus run, invoke the summary test:
//   `cargo test --release ... corpus_v26_zzz_summary -- --ignored --nocapture`

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::stego::gop_pattern::{
    iter_encode_order, FrameType, GopPattern,
};
use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

const N_FRAMES: usize = 30;
const QP: u8 = 26;

struct FixtureSpec {
    name: &'static str,
    source_mp4: &'static str,
    encode_w: u32,
    encode_h: u32,
    content_tags: &'static str,
}

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn ensure_yuv(spec: &FixtureSpec) -> Vec<u8> {
    let yuv_path = format!(
        "/tmp/phasm_corpus_{}_{}x{}_f{}.yuv",
        spec.name, spec.encode_w, spec.encode_h, N_FRAMES
    );
    let frame_size = (spec.encode_w * spec.encode_h * 3 / 2) as usize;
    let need_bytes = frame_size * N_FRAMES;
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need_bytes {
            return data;
        }
    }
    let src = corpus_root().join(spec.source_mp4);
    let crop = format!("crop={}:{}:0:0", spec.encode_w, spec.encode_h);
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &N_FRAMES.to_string()])
        .args(["-an", "-vf", &crop])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("ffmpeg launch");
    assert!(
        status.success(),
        "ffmpeg crop→yuv failed for {}",
        spec.source_mp4
    );
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
        // Phase A (#257) finding: encode_p_frame auto-promotes P→IDR
        // when scene-cut SAD exceeds 20 gray levels. Fast-motion
        // content (CarPlane) trips this, breaking the predictable
        // IBPBP pattern + producing duplicate-DTS at GOP boundaries
        // that confuse downstream players. The corpus harness wants
        // a fixed shape; disable scenecut for the duration.
        std::env::set_var("PHASM_DISABLE_SCENECUT", "1");
    }
}

/// Compute Y/U/V PSNR + SSIM-Y per frame via ffmpeg's psnr/ssim filters,
/// aggregating mean+min across the audit window.
fn compute_psnr_ssim(
    src_yuv: &[u8],
    dec_yuv: &[u8],
    width: u32,
    height: u32,
    audit_frames: usize,
    name: &str,
) -> (Vec<(f64, f64, f64)>, Vec<f64>) {
    let frame_size = (width * height * 3 / 2) as usize;
    let need = audit_frames * frame_size;
    let src_path = std::env::temp_dir()
        .join(format!("phasm_corpus_{}_src_metrics.yuv", name));
    let dec_path = std::env::temp_dir()
        .join(format!("phasm_corpus_{}_dec_metrics.yuv", name));
    std::fs::write(&src_path, &src_yuv[..need.min(src_yuv.len())])
        .expect("write src yuv for metrics");
    std::fs::write(&dec_path, &dec_yuv[..need.min(dec_yuv.len())])
        .expect("write dec yuv for metrics");

    let psnr_log = std::env::temp_dir()
        .join(format!("phasm_corpus_{}_psnr.log", name));
    let ssim_log = std::env::temp_dir()
        .join(format!("phasm_corpus_{}_ssim.log", name));
    let _ = std::fs::remove_file(&psnr_log);
    let _ = std::fs::remove_file(&ssim_log);

    let size_arg = format!("{}x{}", width, height);
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
                    return Some(99.0); // perfect-match cap
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

struct MbStats {
    mean_abs_dev_y: f64,
    edge_contrast_y: f64,
    mb_dc_drift_count_y: u32,
    mb_total_y: u32,
    // chroma plane drift (8x8 blocks because chroma is 4:2:0 subsampled)
    mb_dc_drift_count_u: u32,
    mb_dc_drift_count_v: u32,
    mb_total_chroma: u32,
    mean_abs_dev_u: f64,
    mean_abs_dev_v: f64,
}

/// Compute Rust-side per-MB Y-plane statistics + chroma 8×8 block drift.
fn compute_mb_stats(
    src_yuv: &[u8],
    dec_yuv: &[u8],
    width: u32,
    height: u32,
    audit_frames: usize,
) -> MbStats {
    let frame_size = (width * height * 3 / 2) as usize;
    let y_size = (width * height) as usize;
    let chroma_w = (width / 2) as usize;
    let chroma_h = (height / 2) as usize;
    let chroma_size = chroma_w * chroma_h;
    let stride = width as usize;
    let mb_w = (width / 16) as usize;
    let mb_h = (height / 16) as usize;
    let cmb_w = (width / 16) as usize;  // chroma 8x8 per luma MB; same MB grid
    let cmb_h = (height / 16) as usize;

    let mut total_abs_y: u64 = 0;
    let mut total_count_y: u64 = 0;
    let mut edge_abs: u64 = 0;
    let mut edge_count: u64 = 0;
    let mut interior_abs: u64 = 0;
    let mut interior_count: u64 = 0;
    let mut mb_dc_drift_count_y: u32 = 0;
    let mut mb_total_y: u32 = 0;
    let mut total_abs_u: u64 = 0;
    let mut total_count_u: u64 = 0;
    let mut total_abs_v: u64 = 0;
    let mut total_count_v: u64 = 0;
    let mut mb_dc_drift_count_u: u32 = 0;
    let mut mb_dc_drift_count_v: u32 = 0;
    let mut mb_total_chroma: u32 = 0;

    for f in 0..audit_frames {
        let off = f * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut mb_sum: i64 = 0;
                for dy in 0..16usize {
                    for dx in 0..16usize {
                        let idx = (mb_y * 16 + dy) * stride + mb_x * 16 + dx;
                        let s = src_yuv[off + idx] as i32;
                        let d = dec_yuv[off + idx] as i32;
                        let dev = (s - d) as i64;
                        let abs_dev = dev.unsigned_abs();
                        total_abs_y += abs_dev;
                        total_count_y += 1;
                        mb_sum += dev;
                        let is_edge = dy == 0 || dy == 15 || dx == 0 || dx == 15;
                        if is_edge {
                            edge_abs += abs_dev;
                            edge_count += 1;
                        } else {
                            interior_abs += abs_dev;
                            interior_count += 1;
                        }
                    }
                }
                mb_total_y += 1;
                let mb_dc_drift = (mb_sum / 256).unsigned_abs();
                if mb_dc_drift > 5 {
                    mb_dc_drift_count_y += 1;
                }
            }
        }
        // Chroma planes: U at off+y_size..+y_size+chroma_size, V after.
        let u_off = off + y_size;
        let v_off = off + y_size + chroma_size;
        for cmb_y in 0..cmb_h {
            for cmb_x in 0..cmb_w {
                let mut u_sum: i64 = 0;
                let mut v_sum: i64 = 0;
                for dy in 0..8usize {
                    for dx in 0..8usize {
                        let idx = (cmb_y * 8 + dy) * chroma_w + cmb_x * 8 + dx;
                        let su = src_yuv[u_off + idx] as i32;
                        let du = dec_yuv[u_off + idx] as i32;
                        let sv = src_yuv[v_off + idx] as i32;
                        let dv = dec_yuv[v_off + idx] as i32;
                        let udev = (su - du) as i64;
                        let vdev = (sv - dv) as i64;
                        u_sum += udev;
                        v_sum += vdev;
                        total_abs_u += udev.unsigned_abs();
                        total_count_u += 1;
                        total_abs_v += vdev.unsigned_abs();
                        total_count_v += 1;
                    }
                }
                mb_total_chroma += 1;
                if (u_sum / 64).unsigned_abs() > 3 {
                    mb_dc_drift_count_u += 1;
                }
                if (v_sum / 64).unsigned_abs() > 3 {
                    mb_dc_drift_count_v += 1;
                }
            }
        }
    }
    let mean_abs_dev_y = total_abs_y as f64 / total_count_y.max(1) as f64;
    let mean_abs_dev_u = total_abs_u as f64 / total_count_u.max(1) as f64;
    let mean_abs_dev_v = total_abs_v as f64 / total_count_v.max(1) as f64;
    let edge_mean = edge_abs as f64 / edge_count.max(1) as f64;
    let interior_mean = interior_abs as f64 / interior_count.max(1) as f64;
    let edge_contrast_y = if interior_mean > 1e-6 {
        edge_mean / interior_mean
    } else {
        0.0
    };
    MbStats {
        mean_abs_dev_y,
        edge_contrast_y,
        mb_dc_drift_count_y,
        mb_total_y,
        mb_dc_drift_count_u,
        mb_dc_drift_count_v,
        mb_total_chroma,
        mean_abs_dev_u,
        mean_abs_dev_v,
    }
}

/// Phase C — encode same audit-window YUV via libx264 medium at QP=26
/// and compute Y-PSNR vs source. Returns (y_psnr_mean, ssim_y_mean).
/// Lets us measure phasm−x264 delta to separate "encoder broken" from
/// "content hard to compress at QP=26".
fn compute_x264_reference(
    src_yuv: &[u8],
    width: u32,
    height: u32,
    audit_frames: usize,
    name: &str,
) -> (f64, f64) {
    let frame_size = (width * height * 3 / 2) as usize;
    let need = audit_frames * frame_size;
    let src_path = std::env::temp_dir()
        .join(format!("phasm_corpus_{}_x264_src.yuv", name));
    let mp4_path = std::env::temp_dir()
        .join(format!("phasm_corpus_{}_x264_ref.mp4", name));
    let dec_path = std::env::temp_dir()
        .join(format!("phasm_corpus_{}_x264_dec.yuv", name));
    std::fs::write(&src_path, &src_yuv[..need.min(src_yuv.len())])
        .expect("write x264 src");

    let size_arg = format!("{}x{}", width, height);
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "yuv420p",
               "-s", &size_arg, "-r", "30"])
        .arg("-i").arg(&src_path)
        .args(["-c:v", "libx264", "-preset", "medium",
               "-qp", "26",
               "-x264-params", "keyint=30:min-keyint=30:scenecut=0:bframes=1"])
        .arg(&mp4_path)
        .status()
        .expect("ffmpeg x264 encode");

    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&mp4_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .status()
        .expect("ffmpeg x264 decode");
    let dec_yuv = std::fs::read(&dec_path).unwrap_or_default();
    if dec_yuv.is_empty() {
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&mp4_path);
        let _ = std::fs::remove_file(&dec_path);
        return (f64::NAN, f64::NAN);
    }

    let (psnr_per_frame, ssim_per_frame) = compute_psnr_ssim(
        src_yuv,
        &dec_yuv,
        width,
        height,
        audit_frames,
        &format!("{}_x264", name),
    );
    let y_mean = if psnr_per_frame.is_empty() {
        f64::NAN
    } else {
        psnr_per_frame.iter().map(|p| p.0).sum::<f64>() / psnr_per_frame.len() as f64
    };
    let ssim_mean = if ssim_per_frame.is_empty() {
        f64::NAN
    } else {
        ssim_per_frame.iter().sum::<f64>() / ssim_per_frame.len() as f64
    };

    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&mp4_path);
    let _ = std::fs::remove_file(&dec_path);
    (y_mean, ssim_mean)
}

fn run_fixture(spec: &FixtureSpec) {
    clear_b_env();
    let yuv = ensure_yuv(spec);
    let frame_size = (spec.encode_w * spec.encode_h * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(N_FRAMES);
    let pattern = GopPattern::Ibpbp { gop: 30, b_count: 1 };

    let mut enc = Encoder::new(spec.encode_w, spec.encode_h, Some(QP)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;
    // Phase F (#262, 2026-05-08, v1.0 ship path): SAFE_L0_ZERO
    // overrides every B-MB to L0_16x16+MV=(0,0). Bypasses derived-MV
    // paths so encoder/decoder MC agrees. Closes the corpus
    // ghost-image / blocky-motion bug from Phase D bisect.
    enc.b_rdo_config =
        phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig::SAFE_L0_ZERO;

    let t0 = std::time::Instant::now();
    let mut bs = Vec::new();
    for eo in iter_encode_order(n_frames, pattern) {
        let d = eo.display_idx as usize;
        let f = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(f),
            FrameType::P => enc.encode_p_frame(f),
            FrameType::B => enc.encode_b_frame(f),
        }
        .unwrap_or_else(|e| panic!("[{}] encode error: {e}", spec.name));
        bs.extend_from_slice(&bytes);
    }
    let encode_ms = t0.elapsed().as_millis();

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264,
        &bs,
        spec.encode_w,
        spec.encode_h,
        timing,
        pattern,
        n_frames,
    )
    .expect("mp4 mux");
    let demo = format!(
        "/Users/cgaffga/Desktop/phasm_corpus_{}_v26.mp4",
        spec.name
    );
    std::fs::write(&demo, &mp4).expect("write demo mp4");

    let h264 = std::env::temp_dir().join(format!("phasm_corpus_{}.h264", spec.name));
    let dec = std::env::temp_dir().join(format!("phasm_corpus_{}.dec.yuv", spec.name));
    std::fs::write(&h264, &bs).unwrap();
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec)
        .status()
        .expect("ffmpeg decode");
    assert!(status.success(), "ffmpeg decode failed for {}", spec.name);
    let decoded = std::fs::read(&dec).unwrap();

    let mb_w = (spec.encode_w / 16) as usize;
    let mb_h = (spec.encode_h / 16) as usize;
    let audit_frames = n_frames.min(15);
    let mut total_anom = 0u32;
    let mut max_dev_overall = 0u8;
    let mut worst_frame = 0;
    let mut hist = [0u32; 5];
    let stride = spec.encode_w as usize;
    for frame_idx in 0..audit_frames {
        let off = frame_idx * frame_size;
        let mut frame_max = 0u8;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut max_dev = 0u8;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = (mb_y * 16 + dy) * stride + mb_x * 16 + dx;
                        let s = yuv[off + idx];
                        let d = decoded[off + idx];
                        let dev = (s as i32 - d as i32).unsigned_abs() as u8;
                        if dev > max_dev {
                            max_dev = dev;
                        }
                    }
                }
                if max_dev > 30 {
                    total_anom += 1;
                    let bin = match max_dev {
                        31..=60 => 0,
                        61..=100 => 1,
                        101..=150 => 2,
                        151..=200 => 3,
                        _ => 4,
                    };
                    hist[bin] += 1;
                }
                if max_dev > frame_max {
                    frame_max = max_dev;
                }
            }
        }
        if frame_max > max_dev_overall {
            max_dev_overall = frame_max;
            worst_frame = frame_idx;
        }
    }

    // Phase B metrics (#258): PSNR, SSIM, mean-abs-dev, MB-edge-contrast,
    // MB DC-drift count (Y + chroma). Phase C (#259): x264 reference.
    let (psnr_per_frame, ssim_per_frame) =
        compute_psnr_ssim(&yuv, &decoded, spec.encode_w, spec.encode_h, audit_frames, spec.name);
    let mb = compute_mb_stats(&yuv, &decoded, spec.encode_w, spec.encode_h, audit_frames);
    let (x264_y_psnr, x264_ssim_y) =
        compute_x264_reference(&yuv, spec.encode_w, spec.encode_h, audit_frames, spec.name);

    let mut y_psnr_mean = f64::NAN;
    let mut y_psnr_min = f64::NAN;
    let mut u_psnr_mean = f64::NAN;
    let mut v_psnr_mean = f64::NAN;
    let mut ssim_y_mean = f64::NAN;
    let mut ssim_y_min = f64::NAN;
    if !psnr_per_frame.is_empty() {
        let n = psnr_per_frame.len() as f64;
        y_psnr_mean = psnr_per_frame.iter().map(|p| p.0).sum::<f64>() / n;
        u_psnr_mean = psnr_per_frame.iter().map(|p| p.1).sum::<f64>() / n;
        v_psnr_mean = psnr_per_frame.iter().map(|p| p.2).sum::<f64>() / n;
        y_psnr_min = psnr_per_frame
            .iter()
            .map(|p| p.0)
            .fold(f64::INFINITY, f64::min);
    }
    if !ssim_per_frame.is_empty() {
        let n = ssim_per_frame.len() as f64;
        ssim_y_mean = ssim_per_frame.iter().sum::<f64>() / n;
        ssim_y_min = ssim_per_frame.iter().fold(f64::INFINITY, |a, b| a.min(*b));
    }

    let _ = std::fs::remove_file(&h264);
    let _ = std::fs::remove_file(&dec);

    let y_psnr_delta = y_psnr_mean - x264_y_psnr;
    let ssim_y_delta = ssim_y_mean - x264_ssim_y;

    let report = format!(
        "{name}\t{w}x{h}\t{tags}\tframes={frames}\tencode_ms={ems}\tbs_bytes={bsb}\t\
         total_anom={anom}\tmax_dev={mdev}\tworst_frame={wf}\t\
         hist_31_60={h0}\thist_61_100={h1}\thist_101_150={h2}\thist_151_200={h3}\thist_200plus={h4}\t\
         y_psnr_mean={ypm:.2}\ty_psnr_min={ypn:.2}\tu_psnr_mean={upm:.2}\tv_psnr_mean={vpm:.2}\t\
         ssim_y_mean={sym:.4}\tssim_y_min={syn:.4}\t\
         mean_abs_dev_y={mady:.2}\tmean_abs_dev_u={madu:.2}\tmean_abs_dev_v={madv:.2}\t\
         mb_edge_contrast={mec:.3}\t\
         mb_dc_drift_y={mdy}\tmb_dc_drift_u={mdu}\tmb_dc_drift_v={mdv}\t\
         mb_total_y={mbty}\tmb_total_chroma={mbtc}\t\
         x264_y_psnr={x264y:.2}\tx264_ssim_y={x264s:.4}\t\
         y_psnr_delta={ypd:.2}\tssim_y_delta={ssd:.4}\n",
        name = spec.name,
        w = spec.encode_w,
        h = spec.encode_h,
        tags = spec.content_tags,
        frames = audit_frames,
        ems = encode_ms,
        bsb = bs.len(),
        anom = total_anom,
        mdev = max_dev_overall,
        wf = worst_frame,
        h0 = hist[0],
        h1 = hist[1],
        h2 = hist[2],
        h3 = hist[3],
        h4 = hist[4],
        ypm = y_psnr_mean,
        ypn = y_psnr_min,
        upm = u_psnr_mean,
        vpm = v_psnr_mean,
        sym = ssim_y_mean,
        syn = ssim_y_min,
        mady = mb.mean_abs_dev_y,
        madu = mb.mean_abs_dev_u,
        madv = mb.mean_abs_dev_v,
        mec = mb.edge_contrast_y,
        mdy = mb.mb_dc_drift_count_y,
        mdu = mb.mb_dc_drift_count_u,
        mdv = mb.mb_dc_drift_count_v,
        mbty = mb.mb_total_y,
        mbtc = mb.mb_total_chroma,
        x264y = x264_y_psnr,
        x264s = x264_ssim_y,
        ypd = y_psnr_delta,
        ssd = ssim_y_delta,
    );
    let report_path = format!("/tmp/phasm_corpus_{}_report.txt", spec.name);
    std::fs::write(&report_path, &report).expect("write report");

    eprintln!(
        "[{}] {}x{} -- y_psnr={:.2}(Δ{:+.2}) ssim={:.4}(Δ{:+.4}) drift y/u/v={}/{}/{} mean_abs y/u/v={:.2}/{:.2}/{:.2} demo={}",
        spec.name,
        spec.encode_w,
        spec.encode_h,
        y_psnr_mean,
        y_psnr_delta,
        ssim_y_mean,
        ssim_y_delta,
        mb.mb_dc_drift_count_y,
        mb.mb_dc_drift_count_u,
        mb.mb_dc_drift_count_v,
        mb.mean_abs_dev_y,
        mb.mean_abs_dev_u,
        mb.mean_abs_dev_v,
        demo,
    );
}

const SPEC_IPHONE7_LANDSCAPE: FixtureSpec = FixtureSpec {
    name: "iphone7_landscape",
    source_mp4: "IMG_4138.MOV",
    encode_w: 1920,
    encode_h: 1072,
    content_tags: "iphone7-baseline,kid+wall+jacket",
};

const SPEC_HORSEFLAG: FixtureSpec = FixtureSpec {
    name: "artlist_horseflag",
    source_mp4: "Artlist_HorseFlag.mp4",
    encode_w: 1920,
    encode_h: 1072,
    content_tags: "high-texture,outdoor-motion",
};

const SPEC_PHONEBOOTH: FixtureSpec = FixtureSpec {
    name: "artlist_phonebooth",
    source_mp4: "Artlist_PhoneBooth.mp4",
    encode_w: 1920,
    encode_h: 1072,
    content_tags: "saturated-color,mid-motion",
};

const SPEC_PIRATEBATTLE: FixtureSpec = FixtureSpec {
    name: "artlist_piratebattle",
    source_mp4: "Artlist_PirateBattle.mp4",
    encode_w: 1920,
    encode_h: 1072,
    content_tags: "fast-motion,multi-subject",
};

const SPEC_SCHOOLFIGHT: FixtureSpec = FixtureSpec {
    name: "artlist_schoolfight",
    source_mp4: "Artlist_SchoolFight.mp4",
    encode_w: 1280,
    encode_h: 720,
    content_tags: "indoor-fast,many-subjects,720p",
};

const SPEC_WOMANSUBWAY: FixtureSpec = FixtureSpec {
    name: "artlist_womansubway",
    source_mp4: "Artlist_WomanSubway.mp4",
    encode_w: 1280,
    encode_h: 720,
    content_tags: "low-light,motion-blur,720p",
};

const SPEC_ASIABOTTLE: FixtureSpec = FixtureSpec {
    name: "artlist_asiabottle",
    source_mp4: "Artlist_AsiaBottle.mp4",
    encode_w: 1072,
    encode_h: 1920,
    content_tags: "portrait,slow,scene-switches",
};

const SPEC_CARPLANE: FixtureSpec = FixtureSpec {
    name: "artlist_carplane",
    source_mp4: "Artlist_CarPlane.mp4",
    encode_w: 1072,
    encode_h: 1920,
    content_tags: "portrait,fast-outdoor",
};

const SPEC_HANDBAG: FixtureSpec = FixtureSpec {
    name: "artlist_handbag",
    source_mp4: "Artlist_Handbag.mp4",
    encode_w: 1072,
    encode_h: 1904,
    content_tags: "portrait,skin-closeup",
};

const SPEC_DJI_2K: FixtureSpec = FixtureSpec {
    name: "dji_mini2_2k",
    source_mp4: "dji_mini2_2_7k_24fps_h264_high.mp4",
    encode_w: 2720,
    encode_h: 1520,
    content_tags: "drone-aerial,2.7k",
};

#[test]
#[ignore]
fn corpus_v26_iphone7_landscape() {
    run_fixture(&SPEC_IPHONE7_LANDSCAPE);
}

#[test]
#[ignore]
fn corpus_v26_artlist_horseflag() {
    run_fixture(&SPEC_HORSEFLAG);
}

#[test]
#[ignore]
fn corpus_v26_artlist_phonebooth() {
    run_fixture(&SPEC_PHONEBOOTH);
}

#[test]
#[ignore]
fn corpus_v26_artlist_piratebattle() {
    run_fixture(&SPEC_PIRATEBATTLE);
}

#[test]
#[ignore]
fn corpus_v26_artlist_schoolfight() {
    run_fixture(&SPEC_SCHOOLFIGHT);
}

#[test]
#[ignore]
fn corpus_v26_artlist_womansubway() {
    run_fixture(&SPEC_WOMANSUBWAY);
}

#[test]
#[ignore]
fn corpus_v26_artlist_asiabottle() {
    run_fixture(&SPEC_ASIABOTTLE);
}

#[test]
#[ignore]
fn corpus_v26_artlist_carplane() {
    run_fixture(&SPEC_CARPLANE);
}

#[test]
#[ignore]
fn corpus_v26_artlist_handbag() {
    run_fixture(&SPEC_HANDBAG);
}

#[test]
#[ignore]
fn corpus_v26_dji_mini2_2k() {
    run_fixture(&SPEC_DJI_2K);
}

/// Aggregator. Runs after the per-fixture tests have written their
/// `/tmp/phasm_corpus_<name>_report.txt` files. Reads, sorts, prints
/// a pass/fail table.
///
/// Pass gates (V26 default):
///   max_dev ≤ 100  — soft visual ceiling
///   total_anom relative to 1080p iPhone7 baseline (~486)
#[test]
#[ignore]
fn corpus_v26_zzz_summary() {
    let names: &[&str] = &[
        "iphone7_landscape",
        "artlist_horseflag",
        "artlist_phonebooth",
        "artlist_piratebattle",
        "artlist_schoolfight",
        "artlist_womansubway",
        "artlist_asiabottle",
        "artlist_carplane",
        "artlist_handbag",
        "dji_mini2_2k",
    ];

    eprintln!("\n=== V26 corpus quality table — phasm vs x264-medium reference ===");
    eprintln!(
        "{:<22}  {:>9}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>9}  {:>9}  {:>7}  {}",
        "fixture",
        "dim",
        "y_psnr",
        "Δx264",
        "ssim",
        "Δx264",
        "drift_y",
        "drift_uv",
        "mb_total",
        "mb_chr",
        "max_dev",
        "verdict",
    );
    let mut found = 0;
    for name in names {
        let path = format!("/tmp/phasm_corpus_{}_report.txt", name);
        let Ok(line) = std::fs::read_to_string(&path) else {
            eprintln!("{:<22}  (no report — skip or rerun)", name);
            continue;
        };
        found += 1;
        let mut fields = std::collections::HashMap::new();
        for tok in line.trim().split('\t') {
            if let Some((k, v)) = tok.split_once('=') {
                fields.insert(k.to_string(), v.to_string());
            }
        }
        let dim_pos = line.find('\t').unwrap_or(0) + 1;
        let dim = line[dim_pos..]
            .split('\t')
            .next()
            .unwrap_or("?x?")
            .to_string();

        let f64_field = |k: &str| -> f64 {
            fields.get(k).and_then(|s| s.parse().ok()).unwrap_or(f64::NAN)
        };
        let u64_field = |k: &str| -> u64 {
            fields.get(k).and_then(|s| s.parse().ok()).unwrap_or(0)
        };

        let y_psnr_mean = f64_field("y_psnr_mean");
        let y_psnr_delta = f64_field("y_psnr_delta");
        let ssim_y_mean = f64_field("ssim_y_mean");
        let ssim_y_delta = f64_field("ssim_y_delta");
        let drift_y = u64_field("mb_dc_drift_y");
        let drift_u = u64_field("mb_dc_drift_u");
        let drift_v = u64_field("mb_dc_drift_v");
        let drift_uv = drift_u + drift_v;
        let mb_total_y = u64_field("mb_total_y");
        let mb_total_c = u64_field("mb_total_chroma");
        let max_dev = u64_field("max_dev");

        // Defensible v1.0 thresholds:
        // - phasm should be within ~2 dB of x264 medium QP=26
        // - mb_dc_drift_y rate < 1% (gives a defensible blockiness signal)
        let pass_psnr = y_psnr_delta >= -2.0 || y_psnr_delta.is_nan();
        let drift_rate = if mb_total_y > 0 {
            drift_y as f64 / mb_total_y as f64
        } else {
            0.0
        };
        let pass_drift = drift_rate < 0.01;
        let verdict = if pass_psnr && pass_drift {
            "PASS"
        } else if !pass_drift && pass_psnr {
            "BLOCKY"
        } else if !pass_psnr && pass_drift {
            "PSNR_GAP"
        } else {
            "REVIEW"
        };

        eprintln!(
            "{:<22}  {:>9}  {:>7.2}  {:>+7.2}  {:>7.4}  {:>+7.4}  {:>7}  {:>7}  {:>9}  {:>9}  {:>7}  {}",
            name,
            dim,
            y_psnr_mean,
            y_psnr_delta,
            ssim_y_mean,
            ssim_y_delta,
            drift_y,
            drift_uv,
            mb_total_y,
            mb_total_c,
            max_dev,
            verdict,
        );
    }
    eprintln!(
        "\n{} reports. Demos: ~/Desktop/phasm_corpus_*_v26.mp4. Δx264 = phasm − x264-medium QP26.",
        found
    );
    eprintln!(
        "Verdicts: PASS = ΔY-PSNR ≥ -2dB AND drift_y rate < 1%. \
         BLOCKY = drift dominant. PSNR_GAP = encoder broken vs reference."
    );
}
