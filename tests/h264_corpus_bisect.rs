// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase D (#260) — bisect harness: encode the worst-case fixtures
// with each of several encoder knobs, identify which knob's
// ablation eliminates the visible blocky/staircase artifacts.
//
// Knobs are environment-variable + encoder-field gated. They are
// process-global, so this harness runs serially within one cargo
// test invocation (--test-threads=1 enforced by --ignored alone is
// not enough — but we use a single #[test] that loops internally,
// so parallelism doesn't apply).
//
// Output: /tmp/phasm_bisect_<fixture>_<knob>_report.txt + demo
// MP4 ~/Desktop/phasm_bisect_<fixture>_<knob>.mp4. Aggregator
// produces a comparison table.
//
// Run:
//   cargo test --release --features cabac-stego \
//     --test h264_corpus_bisect bisect_run -- --ignored --nocapture

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::encoder::mb_decision_b::BRdoConfig;
use phasm_core::codec::h264::stego::gop_pattern::{
    iter_encode_order, FrameType, GopPattern,
};
use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};

const N_FRAMES: usize = 15;
const QP: u8 = 26;

#[derive(Clone, Copy, Debug)]
enum Knob {
    /// Default V26 production (clamps + RDO + residual). Reference.
    BaselineV26,
    /// IPPPP no-B-frames. Eliminates B path entirely.
    Ipppp,
    /// All B-MBs forced to Skip.
    ForceSkip,
    /// All B-MBs forced to Direct_16x16.
    ForceDirect,
    /// All B-MBs forced to L0_16x16 with MV=(0,0).
    ForceL0Zero,
    /// All B-MBs forced to Bi_16x16 with MVs=(0,0).
    ForceBiZero,
    /// Disable transform_8x8 (force 4x4 only).
    No8x8Transform,
    /// Disable B-RDO entirely (mode-decision via fast hash mix only).
    NoBRdo,
}

impl Knob {
    fn name(&self) -> &'static str {
        match self {
            Knob::BaselineV26 => "baseline_v26",
            Knob::Ipppp => "ipppp",
            Knob::ForceSkip => "force_skip",
            Knob::ForceDirect => "force_direct",
            Knob::ForceL0Zero => "force_l0_zero",
            Knob::ForceBiZero => "force_bi_zero",
            Knob::No8x8Transform => "no_8x8_transform",
            Knob::NoBRdo => "no_b_rdo",
        }
    }

    fn apply_env(&self) {
        unsafe {
            // Reset all relevant env first so each knob is hermetic.
            for k in [
                "PHASM_B_FORCE_MODE",
                "PHASM_B_FORCE_MV",
                "PHASM_B_FORCE_MV_L1",
                "PHASM_B_NO_BOUNDARY_REFUSE",
                "PHASM_B_NO_DIRECT_MAGCLAMP",
                "PHASM_B_NO_ME_RESULT_CLAMP",
                "PHASM_B_INSTRUMENT",
            ] {
                std::env::remove_var(k);
            }
            std::env::set_var("PHASM_DISABLE_SCENECUT", "1");
            match self {
                Knob::BaselineV26 => {}
                Knob::Ipppp => {} // applied via encoder field below
                Knob::ForceSkip => std::env::set_var("PHASM_B_FORCE_MODE", "skip"),
                Knob::ForceDirect => std::env::set_var("PHASM_B_FORCE_MODE", "direct"),
                Knob::ForceL0Zero => {
                    std::env::set_var("PHASM_B_FORCE_MODE", "l0_16x16");
                    std::env::set_var("PHASM_B_FORCE_MV", "0,0");
                }
                Knob::ForceBiZero => {
                    std::env::set_var("PHASM_B_FORCE_MODE", "bi_16x16");
                    std::env::set_var("PHASM_B_FORCE_MV", "0,0");
                    std::env::set_var("PHASM_B_FORCE_MV_L1", "0,0");
                }
                Knob::No8x8Transform => {} // applied via encoder field
                Knob::NoBRdo => {} // applied via b_rdo_config
            }
        }
    }

    fn configure_encoder(&self, enc: &mut Encoder) {
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = !matches!(self, Knob::No8x8Transform);
        enc.enable_b_frames = !matches!(self, Knob::Ipppp);
        enc.b_rdo_config = match self {
            Knob::NoBRdo => BRdoConfig::SAFE,
            _ => BRdoConfig::PRODUCTION_VISUAL,
        };
    }

    fn pattern(&self) -> GopPattern {
        match self {
            Knob::Ipppp => GopPattern::Ipppp { gop: 30 },
            _ => GopPattern::Ibpbp { gop: 30, b_count: 1 },
        }
    }
}

const KNOBS: &[Knob] = &[
    Knob::BaselineV26,
    Knob::Ipppp,
    Knob::ForceSkip,
    Knob::ForceDirect,
    Knob::ForceL0Zero,
    Knob::ForceBiZero,
    Knob::No8x8Transform,
    Knob::NoBRdo,
];

struct Fixture {
    name: &'static str,
    source_mp4: &'static str,
    encode_w: u32,
    encode_h: u32,
}

// Three most-broken fixtures by drift_y rate (CarPlane 3.18%,
// SchoolFight 1.27%, HorseFlag 0.99%) + iPhone7 (different
// failure mode = isolated high-mag artifacts on textured fine
// detail). Skipping piratebattle for now; if any other knob makes
// these 4 clean, re-test against piratebattle.
const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "iphone7",
        source_mp4: "IMG_4138.MOV",
        encode_w: 1920,
        encode_h: 1072,
    },
    Fixture {
        name: "horseflag",
        source_mp4: "Artlist_HorseFlag.mp4",
        encode_w: 1920,
        encode_h: 1072,
    },
    Fixture {
        name: "schoolfight",
        source_mp4: "Artlist_SchoolFight.mp4",
        encode_w: 1280,
        encode_h: 720,
    },
    Fixture {
        name: "carplane",
        source_mp4: "Artlist_CarPlane.mp4",
        encode_w: 1072,
        encode_h: 1920,
    },
];

fn corpus_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn ensure_yuv(spec: &Fixture) -> Vec<u8> {
    let yuv_path = format!(
        "/tmp/phasm_corpus_artlist_{}_{}x{}_f{}.yuv",
        spec.name, spec.encode_w, spec.encode_h, N_FRAMES
    );
    // For iphone7, the corpus harness uses a different cache name.
    let yuv_alt = format!(
        "/tmp/phasm_corpus_iphone7_landscape_{}x{}_f{}.yuv",
        spec.encode_w, spec.encode_h, N_FRAMES
    );
    if let Ok(data) = std::fs::read(&yuv_path) {
        return data;
    }
    if spec.name == "iphone7" {
        if let Ok(data) = std::fs::read(&yuv_alt) {
            return data;
        }
    }
    // Regen via ffmpeg.
    let target = if spec.name == "iphone7" { &yuv_alt } else { &yuv_path };
    let src = corpus_root().join(spec.source_mp4);
    let crop = format!("crop={}:{}:0:0", spec.encode_w, spec.encode_h);
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &N_FRAMES.to_string()])
        .args(["-an", "-vf", &crop])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(target)
        .status()
        .expect("ffmpeg launch");
    assert!(status.success(), "ffmpeg failed for {}", spec.name);
    std::fs::read(target).expect("read yuv")
}

struct BisectMetrics {
    encode_ms: u128,
    bs_bytes: usize,
    y_psnr_mean: f64,
    y_psnr_min: f64,
    ssim_y_mean: f64,
    drift_y: u32,
    drift_uv: u32,
    mean_abs_y: f64,
    max_dev: u8,
}

fn run_one(fixture: &Fixture, knob: Knob) -> BisectMetrics {
    knob.apply_env();
    let yuv = ensure_yuv(fixture);
    let frame_size = (fixture.encode_w * fixture.encode_h * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(N_FRAMES);

    let mut enc = Encoder::new(fixture.encode_w, fixture.encode_h, Some(QP)).expect("encoder new");
    knob.configure_encoder(&mut enc);
    let pattern = knob.pattern();

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
        .unwrap_or_else(|e| panic!("[{}/{}] encode error: {e}", fixture.name, knob.name()));
        bs.extend_from_slice(&bytes);
    }
    let encode_ms = t0.elapsed().as_millis();

    let timing = FrameTiming::FPS_30;
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264,
        &bs,
        fixture.encode_w,
        fixture.encode_h,
        timing,
        pattern,
        n_frames,
    )
    .expect("mp4 mux");
    let demo = format!(
        "/Users/cgaffga/Desktop/phasm_bisect_{}_{}.mp4",
        fixture.name,
        knob.name()
    );
    std::fs::write(&demo, &mp4).expect("write demo");

    let h264 = std::env::temp_dir().join(format!("phasm_bisect_{}_{}.h264", fixture.name, knob.name()));
    let dec = std::env::temp_dir().join(format!("phasm_bisect_{}_{}.dec.yuv", fixture.name, knob.name()));
    std::fs::write(&h264, &bs).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec)
        .status()
        .expect("ffmpeg decode");
    let decoded = std::fs::read(&dec).unwrap();

    // Audit (15 frames). Y-plane drift, chroma drift, max_dev.
    let stride = fixture.encode_w as usize;
    let mb_w = (fixture.encode_w / 16) as usize;
    let mb_h = (fixture.encode_h / 16) as usize;
    let y_size = (fixture.encode_w * fixture.encode_h) as usize;
    let chroma_w = (fixture.encode_w / 2) as usize;
    let chroma_size = chroma_w * (fixture.encode_h / 2) as usize;
    let audit_frames = n_frames.min(15);
    let mut total_abs_y: u64 = 0;
    let mut count_y: u64 = 0;
    let mut drift_y: u32 = 0;
    let mut drift_uv: u32 = 0;
    let mut max_dev: u8 = 0;
    for f in 0..audit_frames {
        let off = f * frame_size;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut mb_sum: i64 = 0;
                for dy in 0..16 {
                    for dx in 0..16 {
                        let idx = (mb_y * 16 + dy) * stride + mb_x * 16 + dx;
                        let s = yuv[off + idx] as i32;
                        let d = decoded[off + idx] as i32;
                        let dev = s - d;
                        let adev = dev.unsigned_abs() as u8;
                        if adev > max_dev {
                            max_dev = adev;
                        }
                        total_abs_y += adev as u64;
                        count_y += 1;
                        mb_sum += dev as i64;
                    }
                }
                if (mb_sum / 256).unsigned_abs() > 5 {
                    drift_y += 1;
                }
            }
        }
        let u_off = off + y_size;
        let v_off = off + y_size + chroma_size;
        let cmb_w = (fixture.encode_w / 16) as usize;
        let cmb_h = (fixture.encode_h / 16) as usize;
        for cmb_y in 0..cmb_h {
            for cmb_x in 0..cmb_w {
                let mut u_sum: i64 = 0;
                let mut v_sum: i64 = 0;
                for dy in 0..8 {
                    for dx in 0..8 {
                        let idx = (cmb_y * 8 + dy) * chroma_w + cmb_x * 8 + dx;
                        u_sum += yuv[u_off + idx] as i64 - decoded[u_off + idx] as i64;
                        v_sum += yuv[v_off + idx] as i64 - decoded[v_off + idx] as i64;
                    }
                }
                if (u_sum / 64).unsigned_abs() > 3 {
                    drift_uv += 1;
                }
                if (v_sum / 64).unsigned_abs() > 3 {
                    drift_uv += 1;
                }
            }
        }
    }
    let mean_abs_y = total_abs_y as f64 / count_y.max(1) as f64;

    // PSNR + SSIM via ffmpeg. Truncate src to audit frames.
    let need = audit_frames * frame_size;
    let src_path = std::env::temp_dir().join(format!("phasm_bisect_{}_{}_src.yuv", fixture.name, knob.name()));
    let dec_short = std::env::temp_dir().join(format!("phasm_bisect_{}_{}_dec.yuv", fixture.name, knob.name()));
    std::fs::write(&src_path, &yuv[..need.min(yuv.len())]).unwrap();
    std::fs::write(&dec_short, &decoded[..need.min(decoded.len())]).unwrap();
    let psnr_log = std::env::temp_dir().join(format!("phasm_bisect_{}_{}_psnr.log", fixture.name, knob.name()));
    let ssim_log = std::env::temp_dir().join(format!("phasm_bisect_{}_{}_ssim.log", fixture.name, knob.name()));
    let _ = std::fs::remove_file(&psnr_log);
    let _ = std::fs::remove_file(&ssim_log);
    let size = format!("{}x{}", fixture.encode_w, fixture.encode_h);
    let lavfi = format!(
        "[0:v][1:v]psnr=stats_file={};[0:v][1:v]ssim=stats_file={}",
        psnr_log.display(), ssim_log.display()
    );
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "yuv420p",
               "-s", &size])
        .arg("-i").arg(&dec_short)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", &size])
        .arg("-i").arg(&src_path)
        .args(["-lavfi", &lavfi])
        .args(["-f", "null", "-"])
        .status();
    let psnr_text = std::fs::read_to_string(&psnr_log).unwrap_or_default();
    let ssim_text = std::fs::read_to_string(&ssim_log).unwrap_or_default();
    let parse = |line: &str, key: &str| -> Option<f64> {
        for tok in line.split_whitespace() {
            if let Some(rest) = tok.strip_prefix(key) {
                if rest == "inf" {
                    return Some(99.0);
                }
                return rest.parse().ok();
            }
        }
        None
    };
    let psnr_y_per: Vec<f64> = psnr_text.lines().filter_map(|l| parse(l, "psnr_y:")).collect();
    let ssim_y_per: Vec<f64> = ssim_text.lines().filter_map(|l| parse(l, "Y:")).collect();
    let y_psnr_mean = if psnr_y_per.is_empty() { f64::NAN } else { psnr_y_per.iter().sum::<f64>() / psnr_y_per.len() as f64 };
    let y_psnr_min = psnr_y_per.iter().fold(f64::INFINITY, |a, b| a.min(*b));
    let ssim_y_mean = if ssim_y_per.is_empty() { f64::NAN } else { ssim_y_per.iter().sum::<f64>() / ssim_y_per.len() as f64 };

    for p in [&h264, &dec, &src_path, &dec_short, &psnr_log, &ssim_log] {
        let _ = std::fs::remove_file(p);
    }

    let m = BisectMetrics {
        encode_ms,
        bs_bytes: bs.len(),
        y_psnr_mean,
        y_psnr_min,
        ssim_y_mean,
        drift_y,
        drift_uv,
        mean_abs_y,
        max_dev,
    };

    let report_path = format!(
        "/tmp/phasm_bisect_{}_{}_report.txt",
        fixture.name,
        knob.name()
    );
    std::fs::write(
        &report_path,
        format!(
            "fixture={}\tknob={}\tencode_ms={}\tbs={}\ty_psnr={:.2}\ty_psnr_min={:.2}\tssim_y={:.4}\tdrift_y={}\tdrift_uv={}\tmean_abs_y={:.2}\tmax_dev={}\n",
            fixture.name, knob.name(),
            m.encode_ms, m.bs_bytes, m.y_psnr_mean, m.y_psnr_min, m.ssim_y_mean,
            m.drift_y, m.drift_uv, m.mean_abs_y, m.max_dev
        ),
    )
    .unwrap();

    eprintln!(
        "[{}/{:>16}] enc={}ms y_psnr={:.2}/{:.2} ssim={:.4} drift y/uv={}/{} mean_abs_y={:.2} max_dev={} demo={}",
        fixture.name, knob.name(),
        m.encode_ms, m.y_psnr_mean, m.y_psnr_min, m.ssim_y_mean,
        m.drift_y, m.drift_uv, m.mean_abs_y, m.max_dev, demo
    );
    m
}

#[test]
#[ignore]
fn bisect_run() {
    eprintln!("=== Phase D bisect: {} fixtures × {} knobs ===", FIXTURES.len(), KNOBS.len());
    let t0 = std::time::Instant::now();
    for fixture in FIXTURES {
        eprintln!("\n--- fixture {} ---", fixture.name);
        for knob in KNOBS {
            let _ = run_one(fixture, *knob);
        }
    }
    eprintln!("\n=== bisect complete in {:?} ===", t0.elapsed());
}

#[test]
#[ignore]
fn bisect_summary() {
    eprintln!("\n=== Phase D bisect summary ===");
    eprintln!(
        "{:<14} {:<18} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8}",
        "fixture", "knob", "y_psnr", "y_min", "ssim_y", "drift_y", "drift_uv", "mean_abs", "max_dev"
    );
    for fixture in FIXTURES {
        for knob in KNOBS {
            let path = format!(
                "/tmp/phasm_bisect_{}_{}_report.txt",
                fixture.name,
                knob.name()
            );
            let Ok(line) = std::fs::read_to_string(&path) else {
                continue;
            };
            let mut fields = std::collections::HashMap::new();
            for tok in line.trim().split('\t') {
                if let Some((k, v)) = tok.split_once('=') {
                    fields.insert(k.to_string(), v.to_string());
                }
            }
            let f = |k: &str| fields.get(k).cloned().unwrap_or_default();
            eprintln!(
                "{:<14} {:<18} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8}",
                fixture.name,
                knob.name(),
                f("y_psnr"),
                f("y_psnr_min"),
                f("ssim_y"),
                f("drift_y"),
                f("drift_uv"),
                f("mean_abs_y"),
                f("max_dev"),
            );
        }
        eprintln!("");
    }
}
