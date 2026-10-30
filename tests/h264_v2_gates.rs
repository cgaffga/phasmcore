// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// ── H.264 video stego v2 gates (cover order + damage costs + shadows) ─────
//
// Spec: `docs/design/video/stego-cover-order-and-damage-costs.md` § 11.
// Grew out of the drift-diagnosis harness (`diag_h264_intra_drift.rs` on the
// diagnosis branch); this is the committed, env-gated version, built so the
// SAME binary source measures the build before the v2 change and after it:
// it only uses the public encode / decode / probe / walker API, never
// encoder-internal hooks.
//
// Per encode it reports, against the codec's NATURAL encode of the same
// source at the same QP and GOP (`oh264_stream::encode_clean`, no stego
// handlers), so every number is the stego term only:
//
// * visual: mean, worst-frame and worst-MB PSNR-Y; the flood signature (a
//   frame whose error is a near-constant offset over most of its area:
//   damaged MBs > half the frame and |mean e| / mean |e| > 0.8);
// * stealth: per-domain flips vs the natural cover (both walked with the
//   production CABAC walker, positions matched by `PositionKey`), the change
//   rate per admitted position per domain and its ratio to the overall rate,
//   CoeffSign sign balance, and |c| / frame-in-GOP / MB-row / MB-col
//   histograms of the flipped positions next to the admitted ones;
// * capacity: the probe's achievable bytes, primary and shadow;
// * performance: probe, encode and decode wall time;
// * correctness: every payload is decoded back and compared.
//
// Frames are paired BY INDEX: the decoder runs with `-fps_mode passthrough`
// and the frame count is asserted equal to the source's. `index_alignment`
// shows the pairing on a 1080p clip: PSNR(stego[i], natural[i]) against the
// off-by-one pairings.
//
// Knobs (all optional):
//   PHASM_GATE_FIXTURES=src:w:h[,...]  corpus clip (or absolute path) at w×h
//                                      (default IMG_4138.MOV:1920:1080)
//   PHASM_GATE_FRAMES=30               frames per clip
//   PHASM_GATE_GOP=30                  GOP length (30 = production)
//   PHASM_GATE_QP=26
//   PHASM_GATE_FILLS=32b,1k,50%,90%    payload sizes: bytes (b / k) or % of the
//                                      probe's achievable_max_message_bytes
//   PHASM_GATE_SEEDS=1,2               PHASM_DETERMINISTIC_SEED values
//   PHASM_GATE_PASSES=gate-a           passphrases (they key order + H-hat)
//   PHASM_GATE_SHADOWS=0               n shadow layers (each shadow carries
//                                      the fill's payload; the primary 32 B)
//   PHASM_GATE_TAG=before              label in every row (before / after)
//   PHASM_GATE_OUT=<file>              append rows to this TSV as well
//
// Two calibration instruments ride along (same knobs where they apply):
// `per_passphrase_gop_fit` (largest payload one GOP really embeds + verifies
// for 8 passphrases vs the probe's per-GOP figure) and `shadow_damage_hist`
// (the distribution of the shadow damage feature log2(D / D_ref) over a
// clean cover, which sets the shadow bucket edges). `per_passphrase_gop_fit`
// also takes PHASM_GATE_FIT_PASSES=8 (how many of p1..p8),
// PHASM_GATE_TRIAL_SEEDS=0 (extra fixed trial seed pairs) and
// PHASM_GATE_TRIAL_VARY=both|order|hhat (which half of a trial pair moves).
// `natural_temporal_background` dumps the natural encode's per-frame
// admitted / |c| = 1 CoeffSign counts: the background for the § 13
// temporal-concentration bound.
//
// Run (minutes per fixture; --test-threads=1 is mandatory, OH264 is global):
//   PHASM_TMP=/Volumes/EXTDISK/tmp/phasm cargo test --release -p phasm-core \
//     --features h264-encoder --test h264_v2_gates -- --ignored --nocapture \
//     --test-threads=1 gate_sweep
//
// Primary-only κ sweeps: `PHASM_DIAG_DAMAGE=kx_num/kx_den,edge_gain,kt16`
// (see `stego::damage::DamageParams::primary`), `PHASM_H264_DAMAGE_COSTS=0`
// (order only, no damage term).

#![cfg(feature = "h264-encoder")]

mod common;
use common::oh264_stream;
use common::video_fixture::{corpus_root, ensure_scaled_yuv, session_guard, ScaleRecipe};
use common::visual_metrics::psnr_y;

use phasm_core::codec::h264::cabac::bin_decoder::{walk_annex_b_for_cover_with_options, WalkOptions};
use phasm_core::codec::h264::openh264_stego::{
    h264_encode_with_shadows_streaming, EncodeOpts, SliceYuvSource,
};
use phasm_core::codec::h264::stego::cascade_safety::{analyze_safe_mvd_subset, derive_msl_safe_from_msb};
use phasm_core::codec::h264::stego::{CostWeights, DomainBits, PositionKey, SyntaxPath};
use phasm_core::stego::payload::FileEntry;
use phasm_core::stego::shadow_layer::ShadowLayer;
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| default.to_string())
}

fn list<T: std::str::FromStr>(s: &str) -> Vec<T> {
    s.split(',').filter_map(|t| t.trim().parse().ok()).collect()
}

/// Deterministic incompressible bytes (the encoder embeds ciphertext, so a
/// random-looking payload is the representative one).
fn blob(len: usize, salt: u64) -> Vec<u8> {
    let mut s = 0xDEAD_BEEF_u64 ^ (len as u64) << 17 ^ salt;
    (0..len)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) as u8
        })
        .collect()
}

/// Parse a fill: `32b`, `1k`, `50%`. Returns the payload (blob) byte count.
fn fill_bytes(spec: &str, achievable: usize) -> usize {
    let s = spec.trim();
    if let Some(p) = s.strip_suffix('%') {
        let f: f64 = p.parse().unwrap();
        // The file entry adds a small envelope (name, length fields).
        ((achievable as f64 * f / 100.0) as usize).saturating_sub(16).max(1)
    } else if let Some(k) = s.strip_suffix('k') {
        k.parse::<usize>().unwrap() * 1024
    } else if let Some(b) = s.strip_suffix('b') {
        b.parse().unwrap()
    } else {
        s.parse().unwrap()
    }
}

fn source_path(src: &str) -> String {
    if src.starts_with('/') {
        src.to_string()
    } else {
        corpus_root().join(src).to_string_lossy().into_owned()
    }
}

fn load_yuv(src: &str, w: u32, h: u32, n: u32) -> Vec<u8> {
    let key = std::path::Path::new(src)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .replace('.', "_");
    // `ensure_scaled_yuv` resolves relative names under the corpus; absolute
    // paths (e.g. the land-horseflag bench clip) are scaled here the same way.
    if src.starts_with('/') {
        let path = common::scratch_path(&format!("phasm_v2gate_{key}_{w}x{h}_f{n}.yuv"));
        let need = (w * h * 3 / 2) as usize * n as usize;
        if let Ok(d) = std::fs::read(&path) {
            if d.len() >= need {
                return d[..need].to_vec();
            }
        }
        let st = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i", src, "-frames:v", &n.to_string(), "-an"])
            .args(["-vf", &format!("scale={w}:{h}"), "-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&path)
            .status()
            .expect("ffmpeg");
        assert!(st.success());
        let d = std::fs::read(&path).unwrap();
        assert!(d.len() >= need, "{src}: only {} of {need} bytes", d.len());
        return d[..need].to_vec();
    }
    let d = ensure_scaled_yuv(src, "phasm_v2gate", &key, w, h, n, ScaleRecipe::PixFmt);
    let need = (w * h * 3 / 2) as usize * n as usize;
    assert!(d.len() >= need, "{src}: short clip");
    d[..need].to_vec()
}

/// Decode Annex-B to I420 with frames paired by index: `-fps_mode
/// passthrough` emits every decoded frame exactly once (no timestamp-driven
/// duplication or dropping), and the count is asserted.
fn decode_by_index(annex_b: &[u8], tag: &str, frame_bytes: usize, n: u32) -> Vec<u8> {
    let dir = format!("{}/phasm-h264-v2", common::scratch_dir());
    std::fs::create_dir_all(&dir).unwrap();
    // Per-process names: several sweeps may run side by side.
    let pid = std::process::id();
    let inp = format!("{dir}/{tag}.{pid}.h264");
    let out = format!("{dir}/{tag}.{pid}.yuv");
    std::fs::write(&inp, annex_b).unwrap();
    let res = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i", &inp, "-fps_mode", "passthrough"])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", &out])
        .output()
        .expect("ffmpeg decode");
    assert!(res.status.success(), "ffmpeg decode {tag}: {}", String::from_utf8_lossy(&res.stderr));
    let err = String::from_utf8_lossy(&res.stderr);
    assert!(!err.contains("concealing"), "{tag}: decoder concealment: {err}");
    let y = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&inp);
    assert_eq!(y.len(), frame_bytes * n as usize, "{tag}: decoded frame count != {n}");
    y
}

#[derive(Default, Clone)]
struct ClipVisual {
    /// Mean of the per-frame PSNRs (identical frames count as 99 dB).
    mean: f64,
    /// PSNR of the clip's mean squared error (frames weighted equally).
    clip: f64,
    worst_frame: f64,
    worst_mb: f64,
    floods: usize,
    /// Largest |mean e| / mean |e| over frames with > half the MBs damaged.
    max_flood_ratio: f64,
    max_damaged_frac: f64,
}

fn clip_visual(natural: &[u8], stego: &[u8], w: usize, h: usize, n: usize) -> ClipVisual {
    let fsz = w * h * 3 / 2;
    let ysz = w * h;
    let (mbw, mbh) = (w.div_ceil(16), h.div_ceil(16));
    let mut v = ClipVisual { worst_frame: f64::INFINITY, worst_mb: f64::INFINITY, ..Default::default() };
    let mut sum = 0.0;
    let mut mse_sum = 0.0f64;
    for f in 0..n {
        let a = &natural[f * fsz..f * fsz + ysz];
        let b = &stego[f * fsz..f * fsz + ysz];
        let p = psnr_y(a, b);
        sum += p;
        mse_sum += a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / ysz as f64;
        v.worst_frame = v.worst_frame.min(p);
        let (mut abs_sum, mut sgn_sum) = (0.0f64, 0.0f64);
        let mut damaged = 0usize;
        for my in 0..mbh {
            for mx in 0..mbw {
                let (rows, cols) = ((h - my * 16).min(16), (w - mx * 16).min(16));
                let mut se = 0.0f64;
                for dy in 0..rows {
                    for dx in 0..cols {
                        let o = (my * 16 + dy) * w + mx * 16 + dx;
                        let d = b[o] as f64 - a[o] as f64;
                        se += d * d;
                        abs_sum += d.abs();
                        sgn_sum += d;
                    }
                }
                let mse = se / (rows * cols) as f64;
                if mse > 4.0 {
                    damaged += 1;
                }
                let pm = if mse == 0.0 { 99.0 } else { 10.0 * (65025.0 / mse).log10() };
                v.worst_mb = v.worst_mb.min(pm);
            }
        }
        let frac = damaged as f64 / (mbw * mbh) as f64;
        v.max_damaged_frac = v.max_damaged_frac.max(frac);
        if frac > 0.5 && abs_sum > 0.0 {
            let ratio = sgn_sum.abs() / abs_sum;
            v.max_flood_ratio = v.max_flood_ratio.max(ratio);
            if ratio > 0.8 {
                v.floods += 1;
            }
        }
    }
    v.mean = sum / n as f64;
    let mse = mse_sum / n as f64;
    v.clip = if mse == 0.0 { 99.0 } else { 10.0 * (65025.0 / mse).log10() };
    v
}

// ─── per-domain flip statistics (stealth § 4.1 / 4.2) ──────────────────

const DOMAINS: [&str; 4] = ["CS", "CSL", "MSB", "MSL"];

/// AC coefficient positions are the admitted coefficient positions in both
/// builds (DC / chroma DC / Intra16x16 luma DC / 8x8 are wet in both).
fn coeff_admitted(k: PositionKey) -> bool {
    match k.syntax_path() {
        SyntaxPath::Luma4x4 { coeff_idx, .. } | SyntaxPath::ChromaAc { coeff_idx, .. } => coeff_idx != 0,
        _ => false,
    }
}

fn mag_bucket(m: u16) -> usize {
    match m {
        0..=1 => 0,
        2 => 1,
        3 => 2,
        4..=7 => 3,
        8..=15 => 4,
        _ => 5,
    }
}

#[derive(Default)]
struct DomainStats {
    positions: [usize; 4],
    admitted: [usize; 4],
    flips: [usize; 4],
    unmatched: usize,
    /// CoeffSign flips 0→1 / 1→0.
    sign_01: usize,
    sign_10: usize,
    /// [flipped, admitted] per |c| bucket (CS + CSL).
    mag: [[usize; 6]; 2],
    /// [flipped, admitted] per frame-in-GOP (coefficient domains).
    frame: Vec<[usize; 2]>,
    /// [flipped, admitted] per MB-row / MB-col decile (coefficient domains).
    row: [[usize; 2]; 10],
    col: [[usize; 2]; 10],
    /// Flips in the IDR's top MB row or left MB column.
    idr_edge_flips: usize,
}

fn domain_stats(natural: &[u8], stego: &[u8], gop: u32, mvd_admitted: bool) -> DomainStats {
    let walk = |b: &[u8]| {
        walk_annex_b_for_cover_with_options(b, WalkOptions { record_mvd: true, ..Default::default() })
            .expect("walk")
    };
    let nat = walk(natural);
    let stg = walk(stego);
    let (mb_w, mb_h) = (nat.mb_w.max(1), nat.mb_h.max(1));
    let mut s = DomainStats { frame: vec![[0, 0]; gop as usize], ..Default::default() };
    let safe_msb = analyze_safe_mvd_subset(&nat.mvd_meta, nat.mb_w, nat.mb_h);
    let safe_msl = derive_msl_safe_from_msb(
        &nat.cover.mvd_sign_bypass.positions,
        &safe_msb,
        &nat.cover.mvd_suffix_lsb.positions,
    );
    let nd: [&DomainBits; 4] = [
        &nat.cover.coeff_sign_bypass,
        &nat.cover.coeff_suffix_lsb,
        &nat.cover.mvd_sign_bypass,
        &nat.cover.mvd_suffix_lsb,
    ];
    let sd: [&DomainBits; 4] = [
        &stg.cover.coeff_sign_bypass,
        &stg.cover.coeff_suffix_lsb,
        &stg.cover.mvd_sign_bypass,
        &stg.cover.mvd_suffix_lsb,
    ];
    for d in 0..4 {
        let map: HashMap<u64, u8> = sd[d].positions.iter().zip(&sd[d].bits).map(|(k, &b)| (k.raw(), b)).collect();
        for i in 0..nd[d].positions.len() {
            let k = nd[d].positions[i];
            s.positions[d] += 1;
            let admitted = match d {
                0 | 1 => coeff_admitted(k),
                2 => mvd_admitted,
                _ => mvd_admitted && safe_msl.get(i).copied().unwrap_or(false),
            };
            let Some(&sb) = map.get(&k.raw()) else {
                s.unmatched += 1;
                continue;
            };
            let flipped = sb != nd[d].bits[i];
            if admitted {
                s.admitted[d] += 1;
            }
            if flipped {
                s.flips[d] += 1;
            }
            if d < 2 && admitted {
                let fi = (k.frame_idx() % gop) as usize;
                let (mx, my) = (k.mb_addr() % mb_w, k.mb_addr() / mb_w);
                let rd = ((my * 10) / mb_h).min(9) as usize;
                let cd = ((mx * 10) / mb_w).min(9) as usize;
                let mb = mag_bucket(nd[d].magnitudes[i]);
                let f = flipped as usize;
                s.mag[0][mb] += f;
                s.mag[1][mb] += 1;
                s.frame[fi][0] += f;
                s.frame[fi][1] += 1;
                s.row[rd][0] += f;
                s.row[rd][1] += 1;
                s.col[cd][0] += f;
                s.col[cd][1] += 1;
                if flipped && fi == 0 && (mx == 0 || my == 0) {
                    s.idr_edge_flips += 1;
                }
            }
            if d == 0 && flipped {
                if nd[d].bits[i] == 0 {
                    s.sign_01 += 1;
                } else {
                    s.sign_10 += 1;
                }
            }
        }
    }
    s
}

fn hist(pairs: &[[usize; 2]]) -> String {
    pairs.iter().map(|p| format!("{}/{}", p[0], p[1])).collect::<Vec<_>>().join(" ")
}

fn stats_cols(s: &DomainStats) -> String {
    let tot_f: usize = s.flips.iter().sum();
    let tot_a: usize = s.admitted.iter().sum();
    let overall = if tot_a > 0 { tot_f as f64 / tot_a as f64 } else { 0.0 };
    let rate = |d: usize| if s.admitted[d] > 0 { s.flips[d] as f64 / s.admitted[d] as f64 } else { 0.0 };
    let mut cols = Vec::new();
    for d in 0..4 {
        let r = rate(d);
        let k = if overall > 0.0 { r / overall } else { 0.0 };
        cols.push(format!("{}:{}/{}/{}({:.2}x)", DOMAINS[d], s.flips[d], s.admitted[d], s.positions[d], k));
    }
    let mag: Vec<[usize; 2]> = (0..6).map(|b| [s.mag[0][b], s.mag[1][b]]).collect();
    format!(
        "{}\trate={overall:.5}\tunmatched={}\tsign01/10={}/{}\tidr_edge_flips={}\tmag[{}]\tframe[{}]\trow[{}]\tcol[{}]",
        cols.join(" "),
        s.unmatched,
        s.sign_01,
        s.sign_10,
        s.idr_edge_flips,
        hist(&mag),
        hist(&s.frame),
        hist(&s.row),
        hist(&s.col),
    )
}

fn emit(line: &str) {
    eprintln!("{line}");
    if let Ok(p) = std::env::var("PHASM_GATE_OUT") {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let _ = writeln!(f, "{line}");
        }
    }
}

struct Fixture {
    src: String,
    w: u32,
    h: u32,
}

fn fixtures() -> Vec<Fixture> {
    env_or("PHASM_GATE_FIXTURES", "IMG_4138.MOV:1920:1080")
        .split(',')
        .map(|fx| {
            let p: Vec<&str> = fx.split(':').collect();
            Fixture { src: p[0].to_string(), w: p[1].parse().unwrap(), h: p[2].parse().unwrap() }
        })
        .collect()
}

#[test]
#[ignore]
fn gate_sweep() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let n: u32 = env_or("PHASM_GATE_FRAMES", "30").parse().unwrap();
    let gop: u32 = env_or("PHASM_GATE_GOP", "30").parse().unwrap();
    let qp: i32 = env_or("PHASM_GATE_QP", "26").parse().unwrap();
    let fills: Vec<String> = env_or("PHASM_GATE_FILLS", "32b,1k,50%,90%").split(',').map(String::from).collect();
    let seeds: Vec<u64> = list(&env_or("PHASM_GATE_SEEDS", "1,2"));
    let passes: Vec<String> = env_or("PHASM_GATE_PASSES", "gate-a").split(',').map(String::from).collect();
    let n_shadows: usize = env_or("PHASM_GATE_SHADOWS", "0").parse().unwrap();
    let tag = env_or("PHASM_GATE_TAG", "run");
    let mvd_admitted = env_or("PHASM_GATE_MVD_ADMITTED", "0") == "1";
    // PHASM_GATE_NO_VISUAL=1: skip the decodes + PSNR (stealth columns only;
    // the PSNR columns read 0).
    let no_visual = env_or("PHASM_GATE_NO_VISUAL", "0") == "1";

    emit(&format!(
        "GATE_HDR\ttag\tclip\tsize\tframes\tgop\tfill\tbytes\tpass\tseed\tshadows\tdecoded\tenc_ms\tdec_ms\tpsnr_mean\tpsnr_clip\tpsnr_worst_frame\tpsnr_worst_mb\tfloods\tmax_flood_ratio\tmax_damaged_frac\tdomains..."
    ));
    for fx in fixtures() {
        let (w, h) = (fx.w, fx.h);
        let path = source_path(&fx.src);
        let clip = std::path::Path::new(&path).file_name().unwrap().to_string_lossy().into_owned();
        let yuv = load_yuv(&fx.src, w, h, n);
        let fsz = (w * h * 3 / 2) as usize;
        let natural = oh264_stream::encode_clean(&yuv, w, h, n, qp, gop);
        let nat_dec = if no_visual {
            Vec::new()
        } else {
            decode_by_index(&natural, &format!("nat_{clip}_{w}x{h}"), fsz, n)
        };
        let transcode: f64 = if no_visual {
            0.0
        } else {
            (0..n as usize)
                .map(|f| psnr_y(&yuv[f * fsz..f * fsz + (w * h) as usize], &nat_dec[f * fsz..f * fsz + (w * h) as usize]))
                .sum::<f64>()
                / n as f64
        };
        let t = Instant::now();
        let (probe, caps) = oh264_stream::probe_capacity(&yuv, w, h, n, qp, gop).expect("probe");
        let probe_ms = t.elapsed().as_secs_f64() * 1e3;
        let achievable = probe.achievable_max_message_bytes;
        let shadow_cap = probe.shadow_max_message_bytes(n_shadows.max(1) as u32);
        emit(&format!(
            "GATE_CLIP\t{tag}\t{clip}\t{w}x{h}\t{n}\t{gop}\tachievable={achievable}\tprimary_max={}\tshadow_cap(n={})={shadow_cap}\tprobe_ms={probe_ms:.0}\tcaps={caps:?}\ttranscode_psnr={transcode:.2}\tdry_bits={}\tmax_bytes={}\tcomfort_bytes={}",
            probe.primary_max_message_bytes(),
            n_shadows.max(1),
            probe.dry_cover_bits,
            probe.max_message_bytes(),
            probe.comfort_message_bytes(),
        ));
        for fill in &fills {
            let base = if n_shadows > 0 { shadow_cap } else { achievable };
            let bytes = fill_bytes(fill, base);
            for pass in &passes {
                for &seed in &seeds {
                    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", seed.to_string()) };
                    let payload = blob(bytes, seed);
                    let files = vec![FileEntry { filename: "p".into(), content: payload.clone() }];
                    let rowtag = format!("{clip}_{w}x{h}_g{gop}_{fill}_{pass}_s{seed}_sh{n_shadows}");
                    let t = Instant::now();
                    let mut tier = String::from("-");
                    let res = if n_shadows == 0 {
                        oh264_stream::encode_files_planned(
                            &yuv, w, h, n, qp, gop, Some(&caps), "", &files, pass,
                        )
                        .map(|r| {
                            tier = format!("{:?}", r.resolved_tier);
                            r.annex_b
                        })
                    } else {
                        let spass: Vec<String> = (0..n_shadows).map(|i| format!("{pass}-shadow{i}")).collect();
                        let layers: Vec<ShadowLayer> = spass
                            .iter()
                            .map(|p| ShadowLayer { message: "", passphrase: p, files: &files })
                            .collect();
                        let mut src = SliceYuvSource::new(&yuv, w, h, n, gop);
                        h264_encode_with_shadows_streaming(
                            &mut src,
                            w,
                            h,
                            n,
                            EncodeOpts { qp, intra_period: gop as i32 },
                            "phasm v2 gate primary 32 bytes!",
                            &[],
                            pass,
                            &layers,
                            &CostWeights::default(),
                            None,
                            None,
                        )
                    };
                    let enc_ms = t.elapsed().as_secs_f64() * 1e3;
                    let stego = match res {
                        Ok(s) => s,
                        Err(e) => {
                            emit(&format!(
                                "GATE\t{tag}\t{clip}\t{w}x{h}\t{n}\t{gop}\t{fill}\t{bytes}\t{pass}\t{seed}\t{n_shadows}\tENCODE_ERR {e:?}\t{enc_ms:.0}"
                            ));
                            continue;
                        }
                    };
                    let t = Instant::now();
                    let decoded = if n_shadows == 0 {
                        match oh264_stream::decode_full(&stego, pass) {
                            Ok(r) => r.files.first().map(|f| f.content == payload).unwrap_or(false).to_string(),
                            Err(e) => format!("err:{e:?}"),
                        }
                    } else {
                        let p_ok = oh264_stream::decode_text(&stego, pass)
                            .map(|t| t == "phasm v2 gate primary 32 bytes!")
                            .unwrap_or(false);
                        let mut ok = p_ok;
                        for i in 0..n_shadows {
                            let sp = format!("{pass}-shadow{i}");
                            ok &= oh264_stream::decode_full(&stego, &sp)
                                .map(|r| r.files.first().map(|f| f.content == payload).unwrap_or(false))
                                .unwrap_or(false);
                        }
                        ok.to_string()
                    };
                    let dec_ms = t.elapsed().as_secs_f64() * 1e3;
                    let v = if no_visual {
                        ClipVisual::default()
                    } else {
                        let st_dec = decode_by_index(&stego, &format!("st_{rowtag}"), fsz, n);
                        clip_visual(&nat_dec, &st_dec, w as usize, h as usize, n as usize)
                    };
                    let ds = domain_stats(&natural, &stego, gop, mvd_admitted);
                    emit(&format!(
                        "GATE\t{tag}\t{clip}\t{w}x{h}\t{n}\t{gop}\t{fill}\t{bytes}\t{pass}\t{seed}\t{n_shadows}\t{decoded}\t{enc_ms:.0}\t{dec_ms:.0}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{}\t{:.3}\t{:.3}\t{}\ttier={tier}",
                        v.mean,
                        v.clip,
                        v.worst_frame,
                        v.worst_mb,
                        v.floods,
                        v.max_flood_ratio,
                        v.max_damaged_frac,
                        stats_cols(&ds),
                    ));
                }
            }
        }
    }
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };
}

/// Frames are paired by index: for a stego encode, PSNR against the natural
/// encode's SAME frame index is far above the off-by-one pairings (a
/// timestamp-driven duplicate or drop anywhere would show as one of the
/// shifted pairings winning from that frame on).
#[test]
#[ignore]
fn index_alignment() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let n: u32 = env_or("PHASM_GATE_FRAMES", "30").parse().unwrap();
    let gop: u32 = env_or("PHASM_GATE_GOP", "30").parse().unwrap();
    for fx in fixtures() {
        let (w, h) = (fx.w, fx.h);
        let yuv = load_yuv(&fx.src, w, h, n);
        let fsz = (w * h * 3 / 2) as usize;
        let ysz = (w * h) as usize;
        let natural = oh264_stream::encode_clean(&yuv, w, h, n, 26, gop);
        let nat = decode_by_index(&natural, "align_nat", fsz, n);
        unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "1") };
        let files = vec![FileEntry { filename: "p".into(), content: blob(1024, 1) }];
        let stego = oh264_stream::encode_files_planned(&yuv, w, h, n, 26, gop, None, "", &files, "gate-a")
            .expect("encode")
            .annex_b;
        let st = decode_by_index(&stego, "align_st", fsz, n);
        let fr = |b: &[u8], i: usize| b[i * fsz..i * fsz + ysz].to_vec();
        let mut worse = 0;
        for i in 0..n as usize {
            let same = psnr_y(&fr(&nat, i), &fr(&st, i));
            let prev = if i > 0 { psnr_y(&fr(&nat, i - 1), &fr(&st, i)) } else { f64::NAN };
            let next = if i + 1 < n as usize { psnr_y(&fr(&nat, i + 1), &fr(&st, i)) } else { f64::NAN };
            eprintln!("ALIGN\t{}\tframe={i}\tsame={same:.2}\tprev={prev:.2}\tnext={next:.2}", fx.src);
            if !(same > prev.max(next) || (prev.is_nan() && next.is_nan())) {
                worse += 1;
            }
        }
        assert_eq!(worse, 0, "{}: some frame pairs better off-index", fx.src);
    }
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };
}

/// Calibration instrument: for one GOP per fixture, the largest payload that
/// embeds AND decodes for each of 8 passphrases (each keys its own order and
/// H-hat), next to the probe's per-GOP figure (fixed trial seeds, 5/6
/// derated). `PHASM_GATE_FIXTURES` entries are `src:w:h` (frames =
/// `PHASM_GATE_FRAMES`, default 10, one GOP).
#[test]
#[ignore]
fn per_passphrase_gop_fit() {
    use phasm_core::codec::h264::stego::keys::H264StcKeys;
    use phasm_core::codec::h264::streaming_session::h264_diag_gop_emit_once;
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let n: u32 = env_or("PHASM_GATE_FRAMES", "10").parse().unwrap();
    let payload = |len: usize| blob(len, 7);
    for fx in fixtures() {
        let (w, h) = (fx.w, fx.h);
        let yuv = load_yuv(&fx.src, w, h, n);
        let (_probe, caps) = oh264_stream::probe_capacity(&yuv, w, h, n, 26, n).expect("probe");
        let cap = caps[0];
        let mut fits = Vec::new();
        // Passphrase-keyed seeds, then (PHASM_GATE_TRIAL_SEEDS=k) k fixed
        // trial seed pairs like the probe's, to see the probe's own spread.
        let n_trial: u8 = env_or("PHASM_GATE_TRIAL_SEEDS", "0").parse().unwrap();
        let n_pass: usize = env_or("PHASM_GATE_FIT_PASSES", "8").parse().unwrap();
        let mut all_seeds: Vec<phasm_core::stego::cover_order::GopSeeds> = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
            .iter()
            .take(n_pass)
            .map(|p| H264StcKeys::derive(p).unwrap().gop(0))
            .collect();
        // PHASM_GATE_TRIAL_VARY=both|order|hhat: which half of the trial
        // seed pair moves with t (the other stays at the probe's).
        let vary = env_or("PHASM_GATE_TRIAL_VARY", "both");
        for t in 0..n_trial {
            let o = if vary == "hhat" { 0x5b } else { 0x5b ^ t };
            let hh = if vary == "order" { 0x2c } else { 0x2c ^ t };
            all_seeds.push(phasm_core::stego::cover_order::GopSeeds { order: [o; 32], hhat: [hh; 32] });
        }
        for seeds in all_seeds {
            let ok = |len: usize| -> bool {
                h264_diag_gop_emit_once(
                    &yuv, w, h, n, 26, n, &payload(len), 0, len.max(1) as u32, &seeds,
                    &CostWeights::default(),
                )
                .map(|(_, ok)| ok)
                .unwrap_or(false)
            };
            let (mut lo, mut hi) = (0usize, cap * 3 + 64);
            while lo + 8 < hi {
                let mid = (lo + hi) / 2;
                if ok(mid) { lo = mid } else { hi = mid }
            }
            fits.push(lo);
        }
        let mn = *fits.iter().min().unwrap();
        emit(&format!(
            "FIT\t{}\t{w}x{h}x{n}\tprobe_cap={cap}\tfits={fits:?}\tmin/probe={:.2}",
            fx.src,
            mn as f64 / cap.max(1) as f64
        ));
    }
}

/// Calibration instrument: histogram of the shadow damage feature
/// `log2(D / D_ref(26))` (`stego::damage`, the shadow `|c|` cap) over a
/// clean encode's CoeffSign + CoeffSuffixLsb positions, cumulative share per
/// octave. Bucket 0 (`SHADOW_LOW_LOG2`) must hold the bulk (the shadow
/// collision model only holds inside one bucket).
#[test]
#[ignore]
fn shadow_damage_hist() {
    use phasm_core::codec::h264::stego::damage::{DamageModel, DAMAGE_PARAMS, SHADOW_MAG_CAP};
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let n: u32 = env_or("PHASM_GATE_FRAMES", "30").parse().unwrap();
    for fx in fixtures() {
        let (w, h) = (fx.w, fx.h);
        let yuv = load_yuv(&fx.src, w, h, n);
        let clean = oh264_stream::encode_clean(&yuv, w, h, n, 26, n);
        let walk = walk_annex_b_for_cover_with_options(&clean, WalkOptions { record_mvd: true, ..Default::default() })
            .expect("walk");
        let model = DamageModel::from_cover(&walk.cover);
        let r = DamageModel::reference(26, &DAMAGE_PARAMS).ilog2() as i64;
        let mut hist = std::collections::BTreeMap::<i64, usize>::new();
        let mut unpriced = 0usize;
        for bits in [&walk.cover.coeff_sign_bypass, &walk.cover.coeff_suffix_lsb] {
            for (i, &k) in bits.positions.iter().enumerate() {
                match model.damage(k, bits.magnitudes[i], SHADOW_MAG_CAP) {
                    None => unpriced += 1,
                    Some(d) => *hist.entry(d.max(1).ilog2() as i64 - r).or_default() += 1,
                }
            }
        }
        let tot: usize = hist.values().sum();
        let mut cum = 0usize;
        let rows: Vec<String> = hist
            .iter()
            .map(|(l, c)| {
                cum += c;
                format!("{l}:{c}({:.0}%)", 100.0 * cum as f64 / tot.max(1) as f64)
            })
            .collect();
        emit(&format!("DHIST\t{}\t{w}x{h}x{n}\tunpriced={unpriced}\tpriced={tot}\t{}", fx.src, rows.join(" ")));
    }
}

/// Calibration instrument (§ 13 temporal bound): the natural encode's
/// per-frame coefficient statistics, the background a temporal detector
/// sees. Per frame: frame index, frame-in-GOP, the admitted (AC) CoeffSign
/// positions, how many of them have `|c| = 1`, and the frame's MB count.
/// Same knobs as `gate_sweep` (`PHASM_GATE_FRAMES`, `PHASM_GATE_GOP`,
/// `PHASM_GATE_QP`).
#[test]
#[ignore]
fn natural_temporal_background() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let n: u32 = env_or("PHASM_GATE_FRAMES", "120").parse().unwrap();
    let gop: u32 = env_or("PHASM_GATE_GOP", "30").parse().unwrap();
    let qp: i32 = env_or("PHASM_GATE_QP", "26").parse().unwrap();
    for fx in fixtures() {
        let (w, h) = (fx.w, fx.h);
        let yuv = load_yuv(&fx.src, w, h, n);
        let natural = oh264_stream::encode_clean(&yuv, w, h, n, qp, gop);
        let walk = walk_annex_b_for_cover_with_options(&natural, WalkOptions { record_mvd: true, ..Default::default() })
            .expect("walk");
        let cs = &walk.cover.coeff_sign_bypass;
        // [admitted, |c| = 1 among admitted, all CS positions]
        let mut per = vec![[0usize; 3]; n as usize];
        for (i, &k) in cs.positions.iter().enumerate() {
            let f = k.frame_idx() as usize;
            if f >= per.len() {
                continue;
            }
            per[f][2] += 1;
            if coeff_admitted(k) {
                per[f][0] += 1;
                if cs.magnitudes[i] == 1 {
                    per[f][1] += 1;
                }
            }
        }
        let rows: Vec<String> = per
            .iter()
            .enumerate()
            .map(|(f, c)| format!("{f}:{}:{}/{}/{}", f as u32 % gop, c[0], c[1], c[2]))
            .collect();
        emit(&format!(
            "NATBG\t{}\t{w}x{h}x{n}\tgop={gop}\tmbs={}\t{}",
            fx.src,
            walk.mb_w * walk.mb_h,
            rows.join(" ")
        ));
    }
}

/// Probe-deliverability measurement (follow-up, Paths A / B): per fixture,
/// the probe's displayed figure (`achievable_max_message_bytes`) and probe
/// time, then for each passphrase of `PHASM_GATE_PASSES` (default p1..p8) an
/// encode at `PHASM_GATE_FILL_PCT` (default 100) % of that figure on the
/// production planned path, decoded back, plus one wrong-passphrase decode.
/// The payload is a random attachment sized so the framed message is exactly
/// `figure × pct + FRAME_OVERHEAD` bytes, i.e. a message of the displayed
/// size. The variant under test is selected by the `PHASM_H264_*` knobs
/// (`codec::h264::stego::v2_knobs`); rows carry the Path B variant counters.
#[test]
#[ignore]
fn deliverability() {
    use phasm_core::codec::h264::stego::v2_knobs::take_variant_stats;
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let n: u32 = env_or("PHASM_GATE_FRAMES", "60").parse().unwrap();
    let gop: u32 = env_or("PHASM_GATE_GOP", "30").parse().unwrap();
    let qp: i32 = env_or("PHASM_GATE_QP", "26").parse().unwrap();
    let pct: usize = env_or("PHASM_GATE_FILL_PCT", "100").parse().unwrap();
    let tag = env_or("PHASM_GATE_TAG", "run");
    let passes: Vec<String> =
        env_or("PHASM_GATE_PASSES", "p1,p2,p3,p4,p5,p6,p7,p8").split(',').map(String::from).collect();
    for fx in fixtures() {
        let (w, h) = (fx.w, fx.h);
        let path = source_path(&fx.src);
        let clip = std::path::Path::new(&path).file_name().unwrap().to_string_lossy().into_owned();
        let yuv = load_yuv(&fx.src, w, h, n);
        let t = Instant::now();
        let (probe, caps) = oh264_stream::probe_capacity(&yuv, w, h, n, qp, gop).expect("probe");
        let probe_ms = t.elapsed().as_secs_f64() * 1e3;
        let shown = probe.achievable_max_message_bytes;
        emit(&format!("DELIV_CLIP\t{tag}\t{clip}\t{w}x{h}\t{n}\t{gop}\tshown={shown}\tcaps={caps:?}\tprobe_ms={probe_ms:.0}"));
        // PHASM_GATE_PROBE_REPS=k: time k more probes and skip the encodes.
        let probe_reps: usize = env_or("PHASM_GATE_PROBE_REPS", "0").parse().unwrap();
        if probe_reps > 0 {
            let ms: Vec<String> = (0..probe_reps)
                .map(|_| {
                    let t = Instant::now();
                    let _ = oh264_stream::probe_capacity(&yuv, w, h, n, qp, gop).expect("probe");
                    format!("{:.0}", t.elapsed().as_secs_f64() * 1e3)
                })
                .collect();
            emit(&format!("DELIV_PROBE\t{tag}\t{clip}\t{w}x{h}\tprobe_ms=[{}]", ms.join(",")));
            continue;
        }
        let target_framed = shown * pct / 100 + phasm_core::stego::frame::FRAME_OVERHEAD;
        let mut last_stego: Option<Vec<u8>> = None;
        for (pi, pass) in passes.iter().enumerate() {
            let probe_len = target_framed.saturating_sub(phasm_core::stego::frame::FRAME_OVERHEAD).max(1);
            let probe_files = vec![FileEntry { filename: "p".into(), content: blob(probe_len, pi as u64) }];
            let framed0 = oh264_stream::framed_len_files(w, h, n, qp, gop, "", &probe_files, pass).expect("framed len");
            let len = (probe_len + target_framed).saturating_sub(framed0).max(1);
            let payload = blob(len, pi as u64);
            let files = vec![FileEntry { filename: "p".into(), content: payload.clone() }];
            let framed = oh264_stream::framed_len_files(w, h, n, qp, gop, "", &files, pass).expect("framed len");
            let _ = take_variant_stats();
            let t = Instant::now();
            let res = oh264_stream::encode_files_planned(&yuv, w, h, n, qp, gop, Some(&caps), "", &files, pass);
            let enc_ms = t.elapsed().as_secs_f64() * 1e3;
            let enc_stats = take_variant_stats();
            let (ok, dec_ms, dec_stats) = match &res {
                Ok(r) => {
                    let t = Instant::now();
                    let ok = oh264_stream::decode_full(&r.annex_b, pass)
                        .map(|d| d.files.first().map(|f| f.content == payload).unwrap_or(false))
                        .unwrap_or(false);
                    let dms = t.elapsed().as_secs_f64() * 1e3;
                    (ok.to_string(), dms, take_variant_stats())
                }
                Err(e) => (format!("ENCODE_ERR {e:?}"), 0.0, take_variant_stats()),
            };
            emit(&format!(
                "DELIV\t{tag}\t{clip}\t{w}x{h}\t{pass}\tframed={framed}\ttarget={target_framed}\t{ok}\tenc_ms={enc_ms:.0}\tdec_ms={dec_ms:.0}\tenc_tries={:?}\tenc_wins={:?}\tdec_tries={:?}",
                enc_stats[0], enc_stats[1], dec_stats[2],
            ));
            if let Ok(r) = res {
                last_stego = Some(r.annex_b);
            }
        }
        if let Some(stego) = last_stego {
            let _ = take_variant_stats();
            let t = Instant::now();
            let r = oh264_stream::decode_full(&stego, "not-the-passphrase");
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let st = take_variant_stats();
            emit(&format!(
                "DELIV_WRONG\t{tag}\t{clip}\t{w}x{h}\trefused={}\tdec_ms={ms:.0}\tdec_tries={:?}",
                r.is_err(),
                st[2],
            ));
        }
    }
}

/// Calibration instrument for the per-clip capacity maximum (spec § 4a,
/// adaptive `max_message_bytes`): for each fixture, encode at a grid of
/// change rates α (framed payload bits per syntax-dry STC position), planned
/// the way the apps plan (`plan_safe_balanced`, every GOP sampled), and
/// measure the stego term against the natural encode (index-aligned frames).
/// Rows `ACAL` carry the measured clip / worst-frame / worst-MB PSNR-Y, floods
/// and the probe's predicted figures; `ACAL_CLIP` the probe's capacity
/// policy for the cell.
///
/// `PHASM_GATE_DUMP=<dir>` also writes, per cell, the encoder's per-position
/// view of every GOP (`diag_gop_dry_positions`: key, |c|, luma, intra,
/// key_pos, seg_len, area, content cost, final cost, own energy, edge
/// energy — 36 B/position, little-endian) and per encode the per-MB MSE map
/// (f32, frame-major) and the flipped CoeffSign / CoeffSuffixLsb keys (u64),
/// for the offline predictor fit.
///
/// Knobs: PHASM_GATE_FIXTURES, PHASM_GATE_FRAMES (60), PHASM_GATE_GOP (30),
/// PHASM_GATE_QP (26), PHASM_GATE_ALPHAS (comma list of α or `max`,
/// `max<pct>` = that % of the probe's `max_message_bytes`),
/// PHASM_GATE_PASSES (p1,p2,p3), PHASM_GATE_TAG, PHASM_GATE_OUT.
#[test]
#[ignore]
fn adaptmax_calib() {
    use phasm_core::codec::h264::openh264_stego::diag_gop_dry_positions;
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let n: u32 = env_or("PHASM_GATE_FRAMES", "60").parse().unwrap();
    let gop: u32 = env_or("PHASM_GATE_GOP", "30").parse().unwrap();
    let qp: i32 = env_or("PHASM_GATE_QP", "26").parse().unwrap();
    let alphas: Vec<String> =
        env_or("PHASM_GATE_ALPHAS", "0.0625,0.125,0.1875,0.25,0.3125,0.375,0.5").split(',').map(String::from).collect();
    let passes: Vec<String> = env_or("PHASM_GATE_PASSES", "p1,p2,p3").split(',').map(String::from).collect();
    let tag = env_or("PHASM_GATE_TAG", "acal");
    let dump = std::env::var("PHASM_GATE_DUMP").ok().filter(|s| !s.is_empty());
    if let Some(d) = &dump {
        std::fs::create_dir_all(d).unwrap();
    }
    let overhead = phasm_core::stego::frame::FRAME_OVERHEAD;
    for fx in fixtures() {
        let (w, h) = (fx.w, fx.h);
        let path = source_path(&fx.src);
        let clip = std::path::Path::new(&path).file_name().unwrap().to_string_lossy().into_owned();
        let cell = format!("{}_{w}x{h}", clip.replace('.', "_"));
        let yuv = load_yuv(&fx.src, w, h, n);
        let fsz = (w * h * 3 / 2) as usize;
        let natural = oh264_stream::encode_clean(&yuv, w, h, n, qp, gop);
        let nat_dec = decode_by_index(&natural, &format!("acal_nat_{cell}"), fsz, n);
        let t = Instant::now();
        let (probe, caps, curves) = oh264_stream::probe_capacity_curves(&yuv, w, h, n, qp, gop).expect("probe");
        let probe_ms = t.elapsed().as_secs_f64() * 1e3;
        let dry = probe.dry_cover_bits;
        let curve_txt: Vec<String> = curves
            .iter()
            .map(|c| format!("{}:{}:{}", c.n_dry, c.n_frames, c.energy.iter().map(|e| format!("{e:.6e}")).collect::<Vec<_>>().join(",")))
            .collect();
        let ip_txt: Vec<String> = curves
            .iter()
            .map(|c| {
                let q = c.intra_p_share.iter().map(|q| format!("{q:.4}")).collect::<Vec<_>>().join(",");
                let pk = c.intra_p_per_mb.iter().map(|q| format!("{q:.4}")).collect::<Vec<_>>().join(",");
                format!("{q}/{pk}")
            })
            .collect();
        emit(&format!(
            "ACAL_CLIP\t{tag}\t{cell}\t{n}\t{gop}\tdry_bits={dry}\tachievable={}\tmax_bytes={}\tcomfort_bytes={}\tprobe_ms={probe_ms:.0}\tcaps={caps:?}\tmax_alpha_64={}\tip={}\tcurves={}",
            probe.achievable_max_message_bytes,
            probe.max_message_bytes(),
            probe.comfort_message_bytes(),
            probe.max_alpha_64,
            ip_txt.join(";"),
            curve_txt.join(";"),
        ));
        let predict = |framed: usize| -> (f64, f64) {
            use phasm_core::stego::balanced_allocation::{plan_safe_balanced, PlanOutcome};
            let samples: Vec<(usize, usize)> = caps.iter().copied().enumerate().collect();
            let cal = phasm_core::stego::calibration::AllocationCalibration::H264_1080P_QP26;
            match plan_safe_balanced(&samples, caps.len(), framed, gop as usize, &cal) {
                PlanOutcome::Plan { plan, .. } => {
                    phasm_core::codec::h264::stego::quality_predict::predict_quality(&curves, &plan, w, h).unwrap_or((99.0, 99.0))
                }
                _ => (f64::NAN, f64::NAN),
            }
        };
        if let Some(d) = &dump {
            let n_gops = n.div_ceil(gop);
            for g in 0..n_gops {
                let f0 = (g * gop) as usize;
                let gn = gop.min(n - g * gop);
                let slab = &yuv[f0 * fsz..(f0 + gn as usize) * fsz];
                let pos = diag_gop_dry_positions(slab, w, h, gn, EncodeOpts { qp, intra_period: gop as i32 }, &CostWeights::default())
                    .expect("diag positions");
                let mut buf = Vec::with_capacity(pos.len() * 36);
                for p in &pos {
                    let t = p.terms.expect("dry positions are priced");
                    buf.extend_from_slice(&p.key.to_le_bytes());
                    buf.extend_from_slice(&p.magnitude.to_le_bytes());
                    buf.push(p.luma as u8);
                    buf.push(t.intra as u8);
                    buf.extend_from_slice(&(t.key_pos as u16).to_le_bytes());
                    buf.extend_from_slice(&(t.seg_len as u16).to_le_bytes());
                    buf.extend_from_slice(&t.area.to_le_bytes());
                    buf.extend_from_slice(&p.content_cost.to_le_bytes());
                    buf.extend_from_slice(&p.cost.to_le_bytes());
                    buf.extend_from_slice(&(t.e_px2 as f32).to_le_bytes());
                    buf.extend_from_slice(&(t.edge_px2 as f32).to_le_bytes());
                }
                std::fs::write(format!("{d}/{cell}_g{g}.pos"), buf).unwrap();
            }
        }
        let walk = |b: &[u8]| {
            walk_annex_b_for_cover_with_options(b, WalkOptions { record_mvd: false, ..Default::default() }).expect("walk")
        };
        let nat_walk = if dump.is_some() { Some(walk(&natural)) } else { None };
        if env_or("PHASM_GATE_SKIP_ENCODE", "0") == "1" {
            continue;
        }
        for a_spec in &alphas {
            let target_framed = if let Some(p) = a_spec.strip_prefix("max") {
                let pct: usize = if p.is_empty() { 100 } else { p.parse().unwrap() };
                probe.max_message_bytes() * pct / 100 + overhead
            } else {
                let a: f64 = a_spec.parse().unwrap();
                (a * dry as f64 / 8.0) as usize
            };
            for (pi, pass) in passes.iter().enumerate() {
                let probe_len = target_framed.saturating_sub(overhead).max(1);
                let probe_files = vec![FileEntry { filename: "p".into(), content: blob(probe_len, pi as u64) }];
                let framed0 = oh264_stream::framed_len_files(w, h, n, qp, gop, "", &probe_files, pass).expect("framed len");
                let len = (probe_len + target_framed).saturating_sub(framed0).max(1);
                let payload = blob(len, pi as u64);
                let files = vec![FileEntry { filename: "p".into(), content: payload.clone() }];
                let t = Instant::now();
                let res = oh264_stream::encode_files_balanced(&yuv, w, h, n, qp, gop, &caps, "", &files, pass);
                let enc_ms = t.elapsed().as_secs_f64() * 1e3;
                let r = match res {
                    Ok(r) => r,
                    Err(e) => {
                        emit(&format!("ACAL\t{tag}\t{cell}\t{a_spec}\t{pass}\t{target_framed}\tENCODE_ERR {e:?}\t{enc_ms:.0}"));
                        continue;
                    }
                };
                let decoded = oh264_stream::decode_full(&r.annex_b, pass)
                    .map(|d| d.files.first().map(|f| f.content == payload).unwrap_or(false))
                    .unwrap_or(false);
                let st_dec = decode_by_index(&r.annex_b, &format!("acal_st_{cell}_{pi}"), fsz, n);
                let v = clip_visual(&nat_dec, &st_dec, w as usize, h as usize, n as usize);
                let mut n_flips = 0usize;
                if let (Some(d), Some(nw)) = (&dump, &nat_walk) {
                    let sw = walk(&r.annex_b);
                    let mut flips: Vec<u8> = Vec::new();
                    for (nd, sd) in [
                        (&nw.cover.coeff_sign_bypass, &sw.cover.coeff_sign_bypass),
                        (&nw.cover.coeff_suffix_lsb, &sw.cover.coeff_suffix_lsb),
                    ] {
                        let map: HashMap<u64, u8> = sd.positions.iter().zip(&sd.bits).map(|(k, &b)| (k.raw(), b)).collect();
                        for (k, &b) in nd.positions.iter().zip(&nd.bits) {
                            if map.get(&k.raw()).is_some_and(|&sb| sb != b) {
                                flips.extend_from_slice(&k.raw().to_le_bytes());
                                n_flips += 1;
                            }
                        }
                    }
                    let stem = format!("{d}/{cell}_a{a_spec}_{pass}");
                    std::fs::write(format!("{stem}.flips"), flips).unwrap();
                    std::fs::write(format!("{stem}.mb"), mb_mse_map(&nat_dec, &st_dec, w as usize, h as usize, n as usize)).unwrap();
                }
                let (pc, pw) = predict(r.framed_len);
                emit(&format!(
                    "ACAL\t{tag}\t{cell}\t{a_spec}\t{pass}\t{target_framed}\t{}\t{decoded}\t{enc_ms:.0}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{}\tflips={n_flips}\tpred_clip={pc:.2}\tpred_wmb={pw:.2}",
                    r.framed_len, v.mean, v.clip, v.worst_frame, v.worst_mb, v.floods,
                ));
            }
        }
    }
}

/// Per-MB luma MSE (f32 LE, frame-major, MB raster) of `stego` vs `natural`.
fn mb_mse_map(natural: &[u8], stego: &[u8], w: usize, h: usize, n: usize) -> Vec<u8> {
    let fsz = w * h * 3 / 2;
    let (mbw, mbh) = (w.div_ceil(16), h.div_ceil(16));
    let mut out = Vec::with_capacity(n * mbw * mbh * 4);
    for f in 0..n {
        let a = &natural[f * fsz..];
        let b = &stego[f * fsz..];
        for my in 0..mbh {
            for mx in 0..mbw {
                let (rows, cols) = ((h - my * 16).min(16), (w - mx * 16).min(16));
                let mut se = 0.0f64;
                for dy in 0..rows {
                    for dx in 0..cols {
                        let o = (my * 16 + dy) * w + mx * 16 + dx;
                        let d = b[o] as f64 - a[o] as f64;
                        se += d * d;
                    }
                }
                out.extend_from_slice(&((se / (rows * cols) as f64) as f32).to_le_bytes());
            }
        }
    }
    out
}
