// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.4 (#405-#409) — stealth L3 fingerprint gate on OH264 backend.
//
//   C.4.1 (#406) — run distribution gate against OH264-backend output.
//   C.4.2 (#407) — gate thresholds tuned for the OH264 cover story
//                  (see `docs/design/video/h264/openh264-stealth-report.md`).
//
// The cover story (locked in `memory/openh264_cover_story_decision.md`
// after C.4-pre.3 / #391) is: phasm OH264 stego is "OpenH264-style
// stego, no external-encoder match attempted". The L3 stealth gate is
// therefore a SELF-SIMILARITY gate: stego output's marginal mb_type /
// partition / direction / Skip-Intra-split / QP histograms must be
// within ε of the SAME-FIXTURE CLEAN OpenH264 baseline.
//
// Comparison reference: `core/tests/data/openh264_baseline_*_
// reference_mb_histogram.txt` (committed by #390 C.4-pre.2). Each
// baseline file holds the histogram of a clean OH264 encode of the
// corresponding fixture at QP=26, 10 frames, IPPPP, intra_period=30.
//
// Per fixture this test:
//   1. Scales the source MOV/MP4 → YUV at the same dims the baseline
//      used (longest side = 1920, both axes 16-aligned).
//   2. Encodes STEGO via the OH264 backend at QP=26, intra_period=30.
//   3. Pipes stego Annex-B through `scripts/h264_mb_partition_histogram.py`.
//   4. Parses the resulting histogram + the committed clean baseline.
//   5. Asserts each marginal share is within ε of clean.
//
// Smoke variant (`stealth_oh264_smoke_img4138`) runs by default.
// Full corpus variants (`stealth_oh264_*_1080p`) are `#[ignore]`'d.
//
// Run full corpus:
//   cargo test --release --features "h264-encoder" \
//     --test openh264_stealth_distribution_gate -- --ignored --nocapture
//
// Depends on: ffmpeg, ffprobe, python3, PyAV (for the histogram script).

#![cfg(feature = "h264-encoder")]

mod common;
use common::oh264_stream;

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn data_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/data");
    p
}

fn script_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("..");
    p.push("scripts");
    p.push("h264_mb_partition_histogram.py");
    p
}

struct Fixture {
    name: &'static str,
    source_mp4: &'static str,
    /// Tag matches `openh264_baseline_<tag>_reference_mb_histogram.txt`
    baseline_tag: &'static str,
    n_frames: u32,
    qp: i32,
    intra_period: i32,
}

/// Mirror `regen_openh264_baseline.sh::get_target_dims` — fit longest
/// side to 1920, round both axes down to 16. The baseline histograms
/// were generated at these dims; the gate must match.
fn probe_baseline_dims(spec: &Fixture) -> (u32, u32) {
    let src = corpus_root().join(spec.source_mp4);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let out = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height",
               "-of", "csv=p=0"])
        .arg(&src)
        .output()
        .expect("ffprobe");
    let s = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<&str> = s.trim().split(',').collect();
    let iw: u32 = parts[0].parse().expect("source width");
    let ih: u32 = parts[1].parse().expect("source height");
    let (tw, th) = if iw >= ih {
        (1920, ((1920 * ih) / iw / 16) * 16)
    } else {
        (((1920 * iw) / ih / 16) * 16, 1920)
    };
    (tw, th)
}

fn ensure_yuv(spec: &Fixture, w: u32, h: u32) -> Vec<u8> {
    let yuv_path = format!(
        "/tmp/openh264_baseline_{}_{}x{}_f{}.yuv",
        spec.baseline_tag, w, h, spec.n_frames
    );
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * spec.n_frames as usize;
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need { return data; }
    }
    let src = corpus_root().join(spec.source_mp4);
    let vf = format!("scale={w}:{h},format=yuv420p");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &spec.n_frames.to_string(),
               "-vf", &vf, "-f", "rawvideo"])
        .arg(&yuv_path)
        .status()
        .expect("ffmpeg launch");
    assert!(status.success(), "ffmpeg scale failed for {}", spec.source_mp4);
    std::fs::read(&yuv_path).expect("read yuv")
}

// ─────────────────────────── histogram parser ─────────────────────────

/// Subset of the histogram script's per-pict-type output that matters
/// for the L3 stealth gate. All values are percentages over the
/// per-type MB count.
#[derive(Debug, Default, Clone)]
struct PerPicTypeHist {
    // Decoder-synthesized partition shapes (from PyAV MV side-data).
    p16x16_pct: f64,
    p8x8_pct: f64,
    p16x8_pct: f64,
    p8x16_pct: f64,
    /// Decoder-synthesized direction. OH264 default is IPPPP so we
    /// expect ~100% L0 on P-frames, none on I-frames.
    l0_pct: f64,
    l1_pct: f64,
    bi_pct: f64,
    /// Encoder ground truth from ffmpeg-debug mb_type.
    skip_count_pct: f64,
    direct_count_pct: f64,
    intra_count_pct: f64,
    /// Intra-mode split (intra MBs only; matters more on I-frames).
    intra_4x4_pct: f64,
    intra_16x16_pct: f64,
}

#[derive(Debug, Default)]
struct ParsedHist {
    i: PerPicTypeHist,
    p: PerPicTypeHist,
}

fn parse_histogram(text: &str) -> ParsedHist {
    let mut out = ParsedHist::default();
    enum Section { None, I, P, B }
    let mut sec = Section::None;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("## I-frame") {
            sec = Section::I;
            continue;
        }
        if t.starts_with("## P-frame") {
            sec = Section::P;
            continue;
        }
        if t.starts_with("## B-frame") {
            sec = Section::B;
            continue;
        }
        let h = match sec {
            Section::I => &mut out.i,
            Section::P => &mut out.p,
            _ => continue,
        };
        let pct = extract_pct(t);
        if let Some(p) = pct {
            // Order matters: tighter prefixes first so "8x16" doesn't
            // match the "8x" prefix of "8x8" etc.
            if t.starts_with("16x16") { h.p16x16_pct = p; }
            else if t.starts_with("16x8") { h.p16x8_pct = p; }
            else if t.starts_with("8x16") { h.p8x16_pct = p; }
            else if t.starts_with("8x8") { h.p8x8_pct = p; }
            else if t.starts_with("L0") { h.l0_pct = p; }
            else if t.starts_with("L1") { h.l1_pct = p; }
            else if t.starts_with("Bi") { h.bi_pct = p; }
            else if t.starts_with("Skip_count") { h.skip_count_pct = p; }
            else if t.starts_with("Direct_count") { h.direct_count_pct = p; }
            else if t.starts_with("Intra_count") { h.intra_count_pct = p; }
            else if t.starts_with("Intra_4x4") { h.intra_4x4_pct = p; }
            else if t.starts_with("Intra_16x16") { h.intra_16x16_pct = p; }
        }
    }
    out
}

fn extract_pct(line: &str) -> Option<f64> {
    let open = line.find('(')?;
    let close = line[open..].find('%')?;
    line[open + 1..open + close].trim().parse().ok()
}

fn delta(name: &str, stego: f64, clean: f64, eps: f64, fails: &mut Vec<String>) {
    let d = (stego - clean).abs();
    let mark = if d <= eps { "✓" } else { "✗" };
    eprintln!("    {mark} {name:<22} stego={stego:>6.2}%  clean={clean:>6.2}%  Δ={d:>5.2}pp  (ε={eps}pp)");
    if d > eps {
        fails.push(format!("{name}: stego {stego:.2}% vs clean {clean:.2}% (Δ={d:.2}pp > ε={eps}pp)"));
    }
}

// ─────────────────────────── thresholds ───────────────────────────────

/// Partition shape epsilon (pp). Stego flips can shift the marginal
/// shape distribution slightly because flipping CABAC sign bits at
/// |coeff|=16 boundary occasionally changes how the decoder reads
/// subsequent shape bins. Empirically observed: <3pp drift on
/// production fixtures.
const EPS_PARTITION: f64 = 5.0;

/// Direction epsilon (pp). OH264 is IPPPP — P-frames are 100% L0,
/// I-frames have no MVs. Direction drift should be essentially zero.
const EPS_DIRECTION: f64 = 2.0;

/// Skip / Direct / Intra split epsilon (pp). Stego flipping nudges
/// the boundary between Skip and Inter_fwd via MvdSign overrides;
/// 5pp tolerates this without losing the bug-detection signal
/// (cascade leaks would shift these by 20pp+).
const EPS_SKIP_INTRA: f64 = 5.0;

/// Intra mode mix epsilon (pp). Intra_4x4 vs Intra_16x16 are very
/// content-sensitive; CoeffSign flips on I_4x4 sub-blocks can shift
/// the proportion. Tolerance is wider (10pp) because the impact on
/// L3 detectability is small — joint steganalysis features look at
/// shape distribution, not intra-mode submix.
const EPS_INTRA_MIX: f64 = 10.0;

// ─────────────────────────── core harness ─────────────────────────────

fn run_gate(spec: &Fixture, msg: &str, pass: &str) {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());

    let (w, h) = probe_baseline_dims(spec);
    let yuv = ensure_yuv(spec, w, h);

    let stego = oh264_stream::encode(
        &yuv, w, h, spec.n_frames, spec.qp, msg, pass,
    ).expect("oh264 stego encode");

    // Write stego Annex-B + invoke histogram script.
    let stego_h264 = std::env::temp_dir().join(format!("oh264_stealth_{}.h264", spec.name));
    let stego_hist_path = std::env::temp_dir().join(format!("oh264_stealth_{}.hist.txt", spec.name));
    std::fs::write(&stego_h264, &stego).expect("write annex-b");

    let status = std::process::Command::new("python3")
        .arg(script_path())
        .arg(&stego_h264)
        .arg(&stego_hist_path)
        .status()
        .expect("invoke histogram script");
    assert!(status.success(), "histogram script failed for {}", spec.name);

    let stego_hist_text = std::fs::read_to_string(&stego_hist_path)
        .expect("read stego histogram");
    let baseline_path = data_root().join(format!(
        "openh264_baseline_{}_reference_mb_histogram.txt",
        spec.baseline_tag
    ));
    let baseline_text = std::fs::read_to_string(&baseline_path).unwrap_or_else(|e| {
        panic!("missing baseline {}: {e} — run regen_openh264_baseline.sh", baseline_path.display())
    });

    let stego_h = parse_histogram(&stego_hist_text);
    let clean_h = parse_histogram(&baseline_text);

    eprintln!("\n[{}] {w}×{h}×{n} qp={qp} intra={ip} stealth gate vs clean OH264 baseline:",
        spec.name, w = w, h = h, n = spec.n_frames, qp = spec.qp, ip = spec.intra_period);

    let mut fails: Vec<String> = Vec::new();

    eprintln!("  I-frame:");
    delta("intra_4x4",   stego_h.i.intra_4x4_pct,   clean_h.i.intra_4x4_pct,   EPS_INTRA_MIX,  &mut fails);
    delta("intra_16x16", stego_h.i.intra_16x16_pct, clean_h.i.intra_16x16_pct, EPS_INTRA_MIX,  &mut fails);
    delta("intra_count", stego_h.i.intra_count_pct, clean_h.i.intra_count_pct, EPS_SKIP_INTRA, &mut fails);

    eprintln!("  P-frame:");
    delta("part 16x16",  stego_h.p.p16x16_pct, clean_h.p.p16x16_pct, EPS_PARTITION,  &mut fails);
    delta("part 16x8",   stego_h.p.p16x8_pct,  clean_h.p.p16x8_pct,  EPS_PARTITION,  &mut fails);
    delta("part 8x16",   stego_h.p.p8x16_pct,  clean_h.p.p8x16_pct,  EPS_PARTITION,  &mut fails);
    delta("part 8x8",    stego_h.p.p8x8_pct,   clean_h.p.p8x8_pct,   EPS_PARTITION,  &mut fails);
    delta("dir L0",      stego_h.p.l0_pct,     clean_h.p.l0_pct,     EPS_DIRECTION,  &mut fails);
    delta("dir L1",      stego_h.p.l1_pct,     clean_h.p.l1_pct,     EPS_DIRECTION,  &mut fails);
    delta("dir Bi",      stego_h.p.bi_pct,     clean_h.p.bi_pct,     EPS_DIRECTION,  &mut fails);
    delta("skip_count",  stego_h.p.skip_count_pct, clean_h.p.skip_count_pct, EPS_SKIP_INTRA, &mut fails);
    delta("intra_count", stego_h.p.intra_count_pct, clean_h.p.intra_count_pct, EPS_SKIP_INTRA, &mut fails);

    assert!(
        fails.is_empty(),
        "[{}] L3 stealth gate failed on {} marginal(s):\n  {}",
        spec.name, fails.len(), fails.join("\n  ")
    );
}

// ─────────────────────────── smoke ────────────────────────────────────

/// Default-CI smoke gate. Uses iPhone7 fixture at the same baseline
/// dims as the committed reference. Single fixture, ~3s for the
/// encode + histogram script.
#[test]
fn stealth_oh264_smoke_img4138() {
    run_gate(
        &Fixture {
            name: "img4138_smoke",
            source_mp4: "IMG_4138.MOV",
            baseline_tag: "img4138",
            n_frames: 10,
            qp: 26,
            intra_period: 30,
        },
        "C.4 stealth gate smoke", "stealth-pass",
    );
}

// ─────────────────────────── production ───────────────────────────────

#[test]
#[ignore]
fn stealth_oh264_img4138_1080p() {
    run_gate(
        &Fixture {
            name: "img4138_1080p",
            source_mp4: "IMG_4138.MOV",
            baseline_tag: "img4138",
            n_frames: 10, qp: 26, intra_period: 30,
        },
        "C.4 stealth gate — img4138", "stealth-pass",
    );
}

#[test]
#[ignore]
fn stealth_oh264_img4273_1080p() {
    run_gate(
        &Fixture {
            name: "img4273_1080p",
            source_mp4: "IMG_4273.MOV",
            baseline_tag: "img4273",
            n_frames: 10, qp: 26, intra_period: 30,
        },
        "C.4 stealth gate — img4273", "stealth-pass",
    );
}

#[test]
#[ignore]
fn stealth_oh264_carplane_1080p() {
    run_gate(
        &Fixture {
            name: "carplane_1080p",
            source_mp4: "Artlist_CarPlane.mp4",
            baseline_tag: "carplane",
            n_frames: 10, qp: 26, intra_period: 30,
        },
        "C.4 stealth gate — carplane", "stealth-pass",
    );
}

#[test]
#[ignore]
fn stealth_oh264_dji_mini2_1080p() {
    run_gate(
        &Fixture {
            name: "dji_mini2_1080p",
            source_mp4: "dji_mini2_2_7k_24fps_h264_high.mp4",
            baseline_tag: "dji_mini2",
            n_frames: 10, qp: 26, intra_period: 30,
        },
        "C.4 stealth gate — dji_mini2", "stealth-pass",
    );
}

#[test]
#[ignore]
fn stealth_oh264_lumix_g9_1080p() {
    run_gate(
        &Fixture {
            name: "lumix_g9_1080p",
            source_mp4: "lumix_g9_1080p_30fps_h264_high.mp4",
            baseline_tag: "lumix_g9",
            n_frames: 10, qp: 26, intra_period: 30,
        },
        "C.4 stealth gate — lumix_g9", "stealth-pass",
    );
}
