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
    /// Phase 3 diagnostic: same as `BaselineV26` but with the
    /// `skip_cbp_is_zero` early-Skip pre-check disabled. Forces every
    /// B-MB through full RDO sweep. Used to measure how much of the
    /// drift / mismatch on motion content is driven by Skip-without-
    /// residual emission.
    NoSkipPrecheck,
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
            Knob::NoSkipPrecheck => "no_skip_precheck",
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
                "PHASM_B_SKIP_NO_CBP_PRECHECK",
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
                Knob::NoSkipPrecheck => {
                    std::env::set_var("PHASM_B_SKIP_NO_CBP_PRECHECK", "1");
                }
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
    Knob::NoSkipPrecheck,
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
    /// Phase 1 (#266) — encoder/decoder agreement metric. Pixels in
    /// `enc.visual_recon.y` that don't match `ffmpeg.decode` of the
    /// same bitstream. mismatch_y > 0 = real divergence (Bug B class).
    /// mismatch_y ≈ 0 with drift_y > 0 = mode-quality drift (Bug A).
    mismatch_y: u64,
    /// Worst per-pixel mismatch between visual_recon and ffmpeg.decode.
    mismatch_max: u8,
}

fn run_one(fixture: &Fixture, knob: Knob) -> BisectMetrics {
    // Knob env is applied ONCE per knob on the main thread in
    // bisect_run before spawning the fixture-parallel workers.
    // Re-applying here from each worker would race during the
    // remove_var/set_var transition.
    let yuv = ensure_yuv(fixture);
    let frame_size = (fixture.encode_w * fixture.encode_h * 3 / 2) as usize;
    let n_frames = (yuv.len() / frame_size).min(N_FRAMES);

    let mut enc = Encoder::new(fixture.encode_w, fixture.encode_h, Some(QP)).expect("encoder new");
    knob.configure_encoder(&mut enc);
    let pattern = knob.pattern();

    // Phase 1 (#266): capture enc.visual_recon.y per frame. Index by
    // display_idx so we can pair with ffmpeg.decode in display order.
    let y_size = (fixture.encode_w * fixture.encode_h) as usize;
    let mut visual_recon_per_display: Vec<Option<Vec<u8>>> = vec![None; n_frames];

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
        // Capture encoder's claim of post-flip recon for this frame.
        // Only Y plane (we measure mismatch_y).
        visual_recon_per_display[d] = Some(enc.visual_recon.y[..y_size].to_vec());
    }
    let encode_ms = t0.elapsed().as_millis();

    let timing = FrameTiming::FPS_30;
    // Gate the Desktop demo MP4 behind PHASM_BISECT_WRITE_DEMOS=1 so
    // diagnostic bisect runs don't spam ~/Desktop/. Set this var when
    // doing a visual-review pass; leave unset for metric-only runs.
    let demo = if std::env::var("PHASM_BISECT_WRITE_DEMOS").is_ok() {
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
        let path = format!(
            "/Users/cgaffga/Desktop/phasm_bisect_{}_{}.mp4",
            fixture.name,
            knob.name()
        );
        std::fs::write(&path, &mp4).expect("write demo");
        path
    } else {
        "(demos gated; set PHASM_BISECT_WRITE_DEMOS=1)".to_string()
    };
    let _ = timing;

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

    // Phase 1 (#266) — encoder/decoder agreement metric. For each
    // captured visual_recon, diff against the corresponding ffmpeg
    // .decode frame (Y plane only). mismatch_y > 0 = real divergence
    // (Bug B class). mismatch_y ≈ 0 with drift_y > 0 = mode-quality
    // drift (Bug A — encoder + decoder agree on poor output).
    let mut mismatch_y: u64 = 0;
    let mut mismatch_max: u8 = 0;
    for (display_idx, vr_opt) in visual_recon_per_display.iter().enumerate().take(audit_frames) {
        let Some(vr) = vr_opt else { continue };
        let off = display_idx * frame_size;
        for i in 0..y_size.min(vr.len()) {
            let v = vr[i] as i32;
            let d = decoded[off + i] as i32;
            let diff = (v - d).unsigned_abs() as u8;
            if diff > 0 {
                mismatch_y += 1;
            }
            if diff > mismatch_max {
                mismatch_max = diff;
            }
        }
    }

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
        mismatch_y,
        mismatch_max,
    };

    let report_path = format!(
        "/tmp/phasm_bisect_{}_{}_report.txt",
        fixture.name,
        knob.name()
    );
    std::fs::write(
        &report_path,
        format!(
            "fixture={}\tknob={}\tencode_ms={}\tbs={}\ty_psnr={:.2}\ty_psnr_min={:.2}\tssim_y={:.4}\tdrift_y={}\tdrift_uv={}\tmean_abs_y={:.2}\tmax_dev={}\tmismatch_y={}\tmismatch_max={}\n",
            fixture.name, knob.name(),
            m.encode_ms, m.bs_bytes, m.y_psnr_mean, m.y_psnr_min, m.ssim_y_mean,
            m.drift_y, m.drift_uv, m.mean_abs_y, m.max_dev,
            m.mismatch_y, m.mismatch_max,
        ),
    )
    .unwrap();

    eprintln!(
        "[{}/{:>16}] enc={}ms y_psnr={:.2}/{:.2} ssim={:.4} drift y/uv={}/{} mean_abs_y={:.2} max_dev={} mismatch_y={} mismatch_max={} demo={}",
        fixture.name, knob.name(),
        m.encode_ms, m.y_psnr_mean, m.y_psnr_min, m.ssim_y_mean,
        m.drift_y, m.drift_uv, m.mean_abs_y, m.max_dev,
        m.mismatch_y, m.mismatch_max, demo
    );
    m
}

#[test]
#[ignore]
fn bisect_run() {
    eprintln!(
        "=== bisect: {} fixtures × {} knobs (fixtures-within-knob parallel) ===",
        FIXTURES.len(), KNOBS.len()
    );
    let t0 = std::time::Instant::now();
    // Knobs sequential (env vars are process-global; changing between
    // knobs would race). Fixtures within a knob run in parallel —
    // they share the SAME env state for that knob, so no race.
    for knob in KNOBS {
        eprintln!("--- knob {} ---", knob.name());
        // Apply env once on the main thread before spawning workers.
        knob.apply_env();
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(FIXTURES.len());
            for fixture in FIXTURES {
                handles.push(s.spawn(move || run_one(fixture, *knob)));
            }
            for h in handles {
                let _ = h.join();
            }
        });
    }
    eprintln!("\n=== bisect complete in {:?} ===", t0.elapsed());
}

#[test]
#[ignore]
fn bisect_summary() {
    eprintln!("\n=== Phase D bisect summary (with Phase 1 #266 mismatch_y) ===");
    eprintln!(
        "{:<14} {:<18} {:>8} {:>8} {:>8} {:>8} {:>10} {:>11}",
        "fixture", "knob", "y_psnr", "drift_y", "drift_uv", "max_dev", "mismatch_y", "mismatch_max"
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
                "{:<14} {:<18} {:>8} {:>8} {:>8} {:>8} {:>10} {:>11}",
                fixture.name,
                knob.name(),
                f("y_psnr"),
                f("drift_y"),
                f("drift_uv"),
                f("max_dev"),
                f("mismatch_y"),
                f("mismatch_max"),
            );
        }
        eprintln!("");
    }
    eprintln!(
        "Reframe predictions:\n\
         - mismatch_y ≈ 0 with drift_y > 0 = Bug A (mode-quality, encoder/decoder agree)\n\
         - mismatch_y > 0 = Bug B (real divergence)\n\
         - force_skip / force_direct / no_b_rdo: predicted mismatch_y ≈ 0\n\
         - baseline_v26 (RDO with partitioned modes): predicted mismatch_y > 0"
    );
}
