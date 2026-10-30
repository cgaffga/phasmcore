// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AV1 v2 quality gates (spec `docs/design/video/stego-cover-order-and-damage-costs.md`
//! § 11): stego against the NATURAL encode of the same source and params,
//! per frame and per 16×16 luma block, plus the flood signature and the
//! per-plane change rate. Results go to
//! `docs/design/video/av1/v2-gate-results.md`.
//!
//! Corpus clips are pulled through ffmpeg (CLI, test-only) and decoded the
//! same way. Clips: `PHASM_GATE_CLIPS` (comma-separated paths) or the
//! default corpus below; missing clips are skipped with a message.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder", feature = "video"))]

mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use phasm_core::codec::av1::stego::decoder::decode_with_recording;
use phasm_core::codec::av1::stego::primary_plan::gop_primary_caps;
use phasm_core::codec::av1::stego::session::Av1StreamingDecodeSession;
use phasm_core::codec::av1::stego::whole_video::{
    av1_encode_natural, av1_encode_streaming, Av1GopYuvSource, Av1SliceYuvSource, Pass1CacheBudget,
};
use phasm_core::stego::{crypto, frame, payload};
use phasm_core::{av1_probe_gops_at, Av1StreamingEncodeParams};

const QUANTIZER: usize = 100; // production (CLI av1_quantizer, iOS defaultQuantizer)
const GOP: u32 = 30; // production (PhasmAv1Transcoder.defaultGopSize)

#[derive(Debug, Clone, Copy)]
enum Fill {
    Bytes32,
    Kb1,
    Half,
    NinetyPct,
    /// α in thousandths: payload bits per cover position (spec § 4a).
    AlphaMilli(u32),
    /// Thousandths of the primary capacity (deliverability margin).
    PctMilli(u32),
    /// The production red end: the stratified probe's `max_message_bytes`.
    ProbeMax,
}

#[derive(Debug, Default)]
struct GateReport {
    mean_psnr_y: f64,
    worst_frame_psnr_y: f64,
    worst_block_psnr_y: f64,
    worst_frame_psnr_u: f64,
    worst_frame_psnr_v: f64,
    flooded_frames: usize,
    flips: u64,
    flips_by_plane: [u64; 3],
    positions_by_plane: [u64; 3],
    encode_ms: u128,
    decode_ms: u128,
    primary_capacity: usize,
    framed_len: usize,
    /// Cover positions over the clip (the probe's count): α and the change
    /// rate are per cover position.
    cover_positions: usize,
}

fn psnr(mse: f64) -> f64 {
    if mse <= 0.0 {
        99.0
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

/// A flooded frame carries a near-constant offset over most of its area:
/// |mean error| is most of mean |error|, and not negligible.
fn is_flooded(stego: &[u8], natural: &[u8]) -> bool {
    let n = stego.len().max(1) as f64;
    let (mut sum, mut sum_abs) = (0f64, 0f64);
    for (&s, &c) in stego.iter().zip(natural) {
        let e = s as f64 - c as f64;
        sum += e;
        sum_abs += e.abs();
    }
    let (mean, mean_abs) = (sum / n, sum_abs / n);
    mean.abs() >= 1.0 && mean.abs() >= 0.8 * mean_abs
}

fn clip_yuv(clip: &Path, w: u32, h: u32, n: u32) -> Option<Vec<u8>> {
    let vf = format!("scale={w}:{h}:flags=area");
    let out = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(clip)
        .args(["-frames:v", &n.to_string(), "-vf", &vf, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output()
        .ok()?;
    let want = (w * h * 3 / 2 * n) as usize;
    (out.status.success() && out.stdout.len() == want).then_some(out.stdout)
}

/// Decode an AV1 OBU stream to tight I420, muxed by the core MP4 writer as
/// the apps ship it (the raw stream has no temporal delimiters).
fn decode_obu(bytes: &[u8], w: u32, h: u32, tag: &str) -> Vec<u8> {
    use phasm_core::codec::mp4::build::{build_mp4_av1, FrameTiming};
    let dir = format!("{}/av1-v2", common::scratch_dir());
    std::fs::create_dir_all(&dir).unwrap();
    let path = format!("{dir}/{tag}.mp4");
    let mp4 = build_mp4_av1(bytes, w, h, FrameTiming { fps_num: 30, fps_den: 1 }).expect("mux");
    std::fs::write(&path, mp4).unwrap();
    let out = std::process::Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i", &path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .output()
        .expect("ffmpeg decode");
    assert!(out.status.success(), "decode {tag} failed");
    out.stdout
}

fn framed_primary(text: &str, pass: &str) -> Vec<u8> {
    let p = payload::encode_payload(text, &[]).expect("payload");
    let (ct, nonce, salt) = crypto::encrypt(&p, pass).expect("encrypt");
    frame::build_frame(p.len(), &salt, &nonce, &ct)
}

/// Deterministic text whose framed primary is at most `target` bytes.
fn text_for_framed_len(target: usize) -> String {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let pool: String = (0..target * 2 + 64)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (33 + (s % 94) as u8) as char
        })
        .collect();
    let framed_len = |n: usize| payload::encode_payload(&pool[..n], &[]).unwrap().len() + frame::FRAME_OVERHEAD;
    let (mut lo, mut hi) = (0usize, pool.len());
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if framed_len(mid) <= target {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    pool[..lo].to_string()
}

fn gate_case(yuv: &[u8], w: u32, h: u32, n: u32, fill: Fill, tag: &str, pass: &str) -> Option<GateReport> {
    let p = Av1StreamingEncodeParams { width: w, height: h, quantizer: QUANTIZER, gop_size: GOP, total_frames_hint: n, colour: Default::default() };
    let n_gops = n.div_ceil(GOP);
    let mut src = Av1SliceYuvSource::new(yuv, w, h, n, GOP);
    src.rewind().unwrap();
    let cover: Vec<usize> = av1_probe_gops_at(&mut src, n, p, &(0..n_gops).collect::<Vec<_>>())
        .expect("probe")
        .into_iter()
        .map(|(_, cb)| cb)
        .collect();
    let capacity: usize = gop_primary_caps(&cover, &vec![0; cover.len()]).iter().map(|c| c.payload_bytes).sum();
    let target = match fill {
        Fill::Bytes32 => 32,
        Fill::Kb1 => 1024,
        Fill::Half => capacity / 2,
        Fill::NinetyPct => capacity * 9 / 10,
        Fill::AlphaMilli(a) => (cover.iter().sum::<usize>() as u64 * a as u64 / 1000 / 8) as usize,
        Fill::PctMilli(m) => capacity * m as usize / 1000,
        Fill::ProbeMax => {
            src.rewind().unwrap();
            let cal = phasm_core::stego::calibration::AllocationCalibration::AV1_1080P_QP30;
            let est = phasm_core::codec::av1::stego::capacity::av1_capacity_stratified(&mut src, n, p, &cal).expect("stratified probe");
            est.max_message_bytes() + frame::FRAME_OVERHEAD
        }
    };
    if (target > capacity && !matches!(fill, Fill::AlphaMilli(_) | Fill::PctMilli(_) | Fill::ProbeMax)) || target < 32 {
        eprintln!("{tag}: skip {fill:?} (capacity {capacity} B)");
        return None;
    }
    // The length estimate can overshoot the real frame by a few bytes on
    // long texts; trim until the frame fits the target.
    let mut text = text_for_framed_len(target);
    let mut framed = framed_primary(&text, pass);
    while framed.len() > target && !text.is_empty() {
        text.truncate(text.len().saturating_sub(framed.len() - target));
        framed = framed_primary(&text, pass);
    }
    // PHASM_GATE_DELIVERY_ONLY=1: encode + round trip only, one row per
    // case, failures reported instead of asserted (deliverability runs).
    let delivery_only = std::env::var("PHASM_GATE_DELIVERY_ONLY").is_ok_and(|v| v == "1");

    src.rewind().unwrap();
    let natural = if delivery_only { Vec::new() } else { av1_encode_natural(&mut src, n, p).expect("natural encode") };
    src.rewind().unwrap();
    let t0 = Instant::now();
    // PHASM_GATE_SHADOW=<bytes> adds one shadow of that size (spec § 11:
    // a shadow's damage on top of the primary's).
    let shadow_bytes: usize = std::env::var("PHASM_GATE_SHADOW").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let shadow = payload::encode_payload(&"z".repeat(shadow_bytes.max(1)), &[]).unwrap();
    let encoded = if shadow_bytes > 0 {
        phasm_core::codec::av1::stego::whole_video::av1_encode_with_shadows_streaming(
            &mut src, n, p, &framed, pass, &[("gate-shadow", &shadow)], 4, Pass1CacheBudget::from_env_or_default(),
        )
    } else {
        av1_encode_streaming(&mut src, n, p, &framed, pass, Pass1CacheBudget::from_env_or_default(), None)
    };
    let stego = match encoded {
        Ok(s) => s,
        Err(e) => {
            // The probe promised more than the encode can embed: a finding,
            // not a harness failure.
            println!("| {tag} | {pass} | {fill:?} | {} | {capacity} | ENCODE FAILED: {e:?} |", framed.len());
            return None;
        }
    };
    let encode_ms = t0.elapsed().as_millis();
    if delivery_only {
        let mut dec = Av1StreamingDecodeSession::create(pass);
        dec.push_bytes(&stego);
        let outcome = match dec.finish_primary_payload() {
            Ok(got) if got.text == text => "DELIVERED".to_string(),
            Ok(_) => "WRONG TEXT".to_string(),
            Err(e) => format!("DECODE FAILED: {e:?}"),
        };
        println!("| {tag} | {pass} | {fill:?} | {} | {capacity} | {outcome} | {encode_ms} |", framed.len());
        return None;
    }
    let t1 = Instant::now();
    let mut dec = Av1StreamingDecodeSession::create(pass);
    dec.push_bytes(&stego);
    let got = dec.finish_primary_payload().expect("decode");
    let decode_ms = t1.elapsed().as_millis();
    assert_eq!(got.text, text, "{tag} {fill:?}: round trip");

    // Flips: natural and stego share one position list; only values differ.
    let nat_pos = decode_with_recording(&natural).expect("record natural");
    let st_pos = decode_with_recording(&stego).expect("record stego");
    assert_eq!(nat_pos.len(), st_pos.len(), "{tag}: position lists differ");
    let mut r = GateReport {
        encode_ms,
        decode_ms,
        primary_capacity: capacity,
        framed_len: framed.len(),
        cover_positions: cover.iter().sum(),
        ..Default::default()
    };
    for (a, b) in nat_pos.iter().zip(st_pos.iter()) {
        let plane = (b.meta.plane as usize).min(2);
        r.positions_by_plane[plane] += 1;
        if a.decoded_value != b.decoded_value {
            r.flips += 1;
            r.flips_by_plane[plane] += 1;
        }
    }

    let nat_px = decode_obu(&natural, w, h, &format!("{tag}-natural"));
    let st_px = decode_obu(&stego, w, h, &format!("{tag}-stego"));
    let (ys, cs) = ((w * h) as usize, ((w / 2) * (h / 2)) as usize);
    let fsz = ys + 2 * cs;
    assert_eq!(nat_px.len(), st_px.len());
    let frames = nat_px.len() / fsz;
    let (mut mse_sum, mut worst_y, mut worst_u, mut worst_v, mut worst_block) = (0f64, 99f64, 99f64, 99f64, 99f64);
    // PHASM_GATE_PER_GOP=1: worst 16x16 luma block per GOP (display order),
    // to find GOPs the plan overloads (damage-aware allocation, spec § 4a).
    let per_gop = std::env::var("PHASM_GATE_PER_GOP").is_ok_and(|v| v == "1");
    let mut gop_worst = vec![99f64; frames.div_ceil(GOP as usize)];
    let mse = |a: &[u8], b: &[u8]| a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.len().max(1) as f64;
    for f in 0..frames {
        let (a, b) = (&st_px[f * fsz..(f + 1) * fsz], &nat_px[f * fsz..(f + 1) * fsz]);
        let my = mse(&a[..ys], &b[..ys]);
        mse_sum += my;
        worst_y = worst_y.min(psnr(my));
        worst_u = worst_u.min(psnr(mse(&a[ys..ys + cs], &b[ys..ys + cs])));
        worst_v = worst_v.min(psnr(mse(&a[ys + cs..], &b[ys + cs..])));
        if is_flooded(&a[..ys], &b[..ys]) {
            r.flooded_frames += 1;
        }
        for by in (0..h as usize).step_by(16) {
            for bx in (0..w as usize).step_by(16) {
                let (mut s, mut c) = (0f64, 0usize);
                for y in by..(by + 16).min(h as usize) {
                    for x in bx..(bx + 16).min(w as usize) {
                        let i = y * w as usize + x;
                        s += (a[i] as f64 - b[i] as f64).powi(2);
                        c += 1;
                    }
                }
                worst_block = worst_block.min(psnr(s / c as f64));
                let g = f / GOP as usize;
                gop_worst[g] = gop_worst[g].min(psnr(s / c as f64));
            }
        }
    }
    if per_gop {
        let mut gop_flips = vec![0usize; gop_worst.len()];
        let frame_gop = |fr: u16| (fr as usize / GOP as usize).min(gop_worst.len() - 1);
        for (a, b) in nat_pos.iter().zip(st_pos.iter()) {
            if a.decoded_value != b.decoded_value {
                gop_flips[frame_gop(b.frame)] += 1;
            }
        }
        let cells: Vec<String> = gop_worst.iter().zip(&gop_flips).map(|(w, f)| format!("{w:.1}/{f}")).collect();
        println!("PER_GOP {tag} {pass} {fill:?} worst-block-Y/flips (flips by decode frame): {}", cells.join(" "));
    }
    r.mean_psnr_y = psnr(mse_sum / frames.max(1) as f64);
    r.worst_frame_psnr_y = worst_y;
    r.worst_frame_psnr_u = worst_u;
    r.worst_frame_psnr_v = worst_v;
    r.worst_block_psnr_y = worst_block;
    Some(r)
}

fn corpus() -> Vec<PathBuf> {
    if let Ok(list) = std::env::var("PHASM_GATE_CLIPS") {
        return list.split(',').map(PathBuf::from).collect();
    }
    let bench = "/Volumes/EXTDISK/tmp/phasm-bench";
    let mut v: Vec<PathBuf> = ["port-iphone5-3s.mov", "rot180-iphone7-3s.mov", "land-horseflag.mp4"]
        .iter()
        .map(|c| PathBuf::from(bench).join(c))
        .collect();
    v.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-vectors/video/h264/real-world/source/IMG_4138.MOV"));
    v
}

/// The corpus at 360p and 720p (all clips) and 1080p (IMG_4138), 2 GOPs of
/// 30 frames, at 32 B, 1 KB, 50 % and 90 % of the primary capacity.
#[test]
#[ignore = "AV1 encodes over the corpus, minutes; --ignored --nocapture --test-threads=1"]
fn v2_gate_corpus() {
    // PHASM_GATE_GOPS=N sets the clip length (default 2 GOPs).
    let n = GOP * std::env::var("PHASM_GATE_GOPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    println!("| clip | size | fill | framed B | cap B | flips | Y/U/V flips | mean Y | worst frame Y | worst block Y | worst U | worst V | flooded | enc ms | dec ms | cover |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for clip in corpus() {
        if !clip.exists() {
            eprintln!("skip missing clip {}", clip.display());
            continue;
        }
        let name = clip.file_stem().unwrap().to_string_lossy().to_string();
        let default_sizes: &[(u32, u32)] = if name == "IMG_4138" { &[(640, 360), (1280, 720), (1920, 1080)] } else { &[(640, 360), (1280, 720)] };
        // PHASM_GATE_SIZES="640x360,1280x720" narrows the sizes (calibration).
        let sizes: Vec<(u32, u32)> = match std::env::var("PHASM_GATE_SIZES") {
            Ok(v) => v.split(',').filter_map(|s| s.split_once('x')).map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())).collect(),
            Err(_) => default_sizes.to_vec(),
        };
        for &(w, h) in &sizes {
            let Some(yuv) = clip_yuv(&clip, w, h, n) else {
                eprintln!("skip {name} {w}x{h}: could not extract {n} frames");
                continue;
            };
            // PHASM_GATE_FILLS="Kb1,Half" narrows the fills (calibration).
            // PHASM_GATE_ALPHAS="20,40,80" (thousandths) sweeps α instead (§ 4a).
            // PHASM_GATE_PCTS="1000,950" (thousandths of capacity) measures
            // deliverability; PHASM_GATE_REPS=N repeats each case with
            // passphrases gate-0..gate-N.
            let fills: Vec<Fill> = match (std::env::var("PHASM_GATE_ALPHAS"), std::env::var("PHASM_GATE_PCTS")) {
                (Ok(v), _) => v.split(',').filter_map(|a| a.trim().parse().ok()).map(Fill::AlphaMilli).collect(),
                (_, Ok(v)) => v.split(',').filter_map(|a| a.trim().parse().ok()).map(Fill::PctMilli).collect(),
                _ if std::env::var("PHASM_GATE_FILLS").is_ok_and(|v| v == "ProbeMax") => vec![Fill::ProbeMax],
                _ => [Fill::Bytes32, Fill::Kb1, Fill::Half, Fill::NinetyPct]
                    .into_iter()
                    .filter(|f| std::env::var("PHASM_GATE_FILLS").map_or(true, |v| v.split(',').any(|x| x == format!("{f:?}"))))
                    .collect(),
            };
            let reps: usize = std::env::var("PHASM_GATE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            for (fill, rep) in fills.into_iter().flat_map(|f| (0..reps).map(move |r| (f, r))) {
                let tag = format!("{name}-{h}p");
                let pass = if reps > 1 { format!("gate-{rep}") } else { "gate".to_string() };
                let Some(r) = gate_case(&yuv, w, h, n, fill, &tag, &pass) else { continue };
                println!(
                    "| {name} | {h}p | {fill:?} | {} | {} | {} | {}/{}/{} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {} | {} | {} | {} |",
                    r.framed_len, r.primary_capacity, r.flips,
                    r.flips_by_plane[0], r.flips_by_plane[1], r.flips_by_plane[2],
                    r.mean_psnr_y, r.worst_frame_psnr_y, r.worst_block_psnr_y,
                    r.worst_frame_psnr_u, r.worst_frame_psnr_v, r.flooded_frames, r.encode_ms, r.decode_ms,
                    r.cover_positions
                );
            }
        }
    }
}

#[test]
fn flood_signature_detects_a_constant_offset() {
    let natural: Vec<u8> = (0..4096).map(|i| (64 + (i * 7) % 128) as u8).collect();
    let shifted: Vec<u8> = natural.iter().map(|&v| v + 6).collect();
    assert!(is_flooded(&shifted, &natural));
    let mut s = 0x1234_5678u32;
    let noisy: Vec<u8> = natural
        .iter()
        .map(|&v| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if s & 1 == 0 { v + 6 } else { v - 6 }
        })
        .collect();
    assert!(!is_flooded(&noisy, &natural));
}

/// A v1 (pre-v2 wire) stego stream for Task 4's `v1_video_does_not_decode`:
/// 256×144, 2 GOPs, 32 B, passphrase `v1-fixture`. Run once BEFORE the
/// wire changes.
#[test]
#[ignore = "writes the v1 fixture; run once before the v2 wire lands"]
fn write_v1_fixture() {
    let clip = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-vectors/video/h264/real-world/source/IMG_4138.MOV");
    let (w, h, gop, n) = (256u32, 144u32, 8u32, 16u32);
    let yuv = clip_yuv(&clip, w, h, n).expect("corpus");
    let p = Av1StreamingEncodeParams { width: w, height: h, quantizer: 30, gop_size: gop, total_frames_hint: n, colour: Default::default() };
    let framed = framed_primary(&text_for_framed_len(32), "v1-fixture");
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let stego = av1_encode_streaming(&mut src, n, p, &framed, "v1-fixture", Pass1CacheBudget::from_env_or_default(), None)
        .expect("encode");
    let dir = format!("{}/av1-v2", common::scratch_dir());
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(format!("{dir}/v1-fixture.obu"), &stego).unwrap();
    let mut dec = Av1StreamingDecodeSession::create("v1-fixture");
    dec.push_bytes(&stego);
    assert!(dec.finish_primary_payload().is_ok(), "v1 fixture must decode under v1");
}

/// Spec § 11 "the probe matches the encode": at 360p and 720p, an encode at
/// exactly the stratified probe's primary capacity, with one shadow at its
/// shadow capacity, succeeds and round-trips both messages.
#[test]
#[ignore = "AV1 encodes over the corpus, minutes; --ignored --nocapture --test-threads=1"]
fn probe_capacity_encodes_on_the_corpus() {
    use phasm_core::codec::av1::stego::whole_video::av1_encode_with_shadows_streaming;
    use phasm_core::stego::calibration::AllocationCalibration;
    use phasm_core::av1_capacity_stratified;
    let n = 2 * GOP;
    let mut failures = Vec::new();
    for clip in corpus() {
        if !clip.exists() {
            continue;
        }
        let name = clip.file_stem().unwrap().to_string_lossy().to_string();
        for (w, h) in [(640u32, 360u32), (1280, 720)] {
            let Some(yuv) = clip_yuv(&clip, w, h, n) else { continue };
            let p = Av1StreamingEncodeParams { width: w, height: h, quantizer: QUANTIZER, gop_size: GOP, total_frames_hint: n, colour: Default::default() };
            let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, GOP);
            src.rewind().unwrap();
            let cap = av1_capacity_stratified(&mut src, n, p, &AllocationCalibration::AV1_1080P_QP30).expect("probe");
            let text = text_for_framed_len(cap.primary_max_message_bytes + frame::FRAME_OVERHEAD);
            let framed = framed_primary(&text, "probe-p");
            let shadow_text = "s".repeat(cap.shadow_max_message_bytes_n1.saturating_sub(16).max(1));
            let shadow = payload::encode_payload(&shadow_text, &[]).unwrap();
            src.rewind().unwrap();
            let got = av1_encode_with_shadows_streaming(
                &mut src, n, p, &framed, "probe-p", &[("probe-s", &shadow)], 4,
                Pass1CacheBudget::from_env_or_default(),
            );
            let tag = format!("{name} {h}p primary {} B shadow {} B", framed.len(), shadow.len());
            match got {
                Ok(stego) => {
                    let mut dec = Av1StreamingDecodeSession::create("probe-p");
                    dec.push_bytes(&stego);
                    let ok_p = dec.finish_shadow_whole_video("probe-s").map(|s| s.text == shadow_text).unwrap_or(false);
                    let ok_s = dec.finish_primary_payload().map(|d| d.text == text).unwrap_or(false);
                    println!("{tag}: encoded, primary {ok_s}, shadow {ok_p}");
                    if !(ok_p && ok_s) {
                        failures.push(format!("{tag}: round trip failed"));
                    }
                }
                Err(e) => {
                    println!("{tag}: ENCODE FAILED {e:?}");
                    failures.push(format!("{tag}: {e:?}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Diagnostic: list the flipped cover positions of one frame (decode
/// order) between two OBU streams, with their syntax and v2 damage.
/// `PHASM_DIAG_NATURAL`, `PHASM_DIAG_STEGO`, `PHASM_DIAG_FRAME`.
#[test]
#[ignore = "diagnostic; needs PHASM_DIAG_* env"]
fn diag_flips_in_frame() {
    use phasm_core::stego::cost::av1_damage::{damage, DamageInputs, DamageParams};
    let nat = std::fs::read(std::env::var("PHASM_DIAG_NATURAL").unwrap()).unwrap();
    let st = std::fs::read(std::env::var("PHASM_DIAG_STEGO").unwrap()).unwrap();
    let want: u16 = std::env::var("PHASM_DIAG_FRAME").unwrap().parse().unwrap();
    let a = decode_with_recording(&nat).unwrap();
    let b = decode_with_recording(&st).unwrap();
    let frames = b.frames().to_vec();
    let gop = std::env::var("PHASM_DIAG_GOP").ok().and_then(|v| v.parse().ok()).unwrap_or(30u16);
    let mut rows = Vec::new();
    for (x, y) in a.iter().zip(b.iter()) {
        if x.decoded_value == y.decoded_value || y.frame != want {
            continue;
        }
        let f = frames[y.frame as usize];
        let (w, h) = if y.meta.plane == 0 { (f.width, f.height) } else { (f.width.div_ceil(2), f.height.div_ceil(2)) };
        let d = damage(
            &DamageInputs {
                plane: y.meta.plane, px_x: y.meta.plane_px_x, px_y: y.meta.plane_px_y, plane_w: w, plane_h: h,
                tx_width_log2: y.meta.tx_width_log2, tx_height_log2: y.meta.tx_height_log2, tx_type: y.meta.tx_type,
                scan_pos: y.meta.scan_pos, abs_c: y.coeff_magnitude, qindex: f.qindex,
                frame_idx: y.frame % gop, gop_len: gop, is_key_frame: y.frame % gop == 0,
            },
            &DamageParams::from_env(),
        );
        rows.push((d, y.tag, y.meta, y.coeff_magnitude));
    }
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("{} flips in frame {want}", rows.len());
    for (d, tag, m, c) in rows.iter().take(40) {
        println!(
            "D {d:>12.0} tag {tag} plane {} px ({},{}) tx {}x{} type {} scan {} |c| {c}",
            m.plane, m.plane_px_x, m.plane_px_y, 1 << m.tx_width_log2, 1 << m.tx_height_log2, m.tx_type, m.scan_pos
        );
    }
}

/// The stratified probe alone (the production capacity figure) on
/// `PHASM_GATE_CLIPS` at `PHASM_GATE_SIZES`, `PHASM_GATE_GOPS` GOPs.
#[test]
#[ignore = "AV1 probe encodes, minutes; --ignored --nocapture"]
fn v2_probe_report() {
    let n = GOP * std::env::var("PHASM_GATE_GOPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let sizes: Vec<(u32, u32)> = std::env::var("PHASM_GATE_SIZES")
        .unwrap_or_else(|_| "1280x720".into())
        .split(',')
        .filter_map(|s| s.split_once('x'))
        .map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()))
        .collect();
    for clip in corpus() {
        for &(w, h) in &sizes {
            let Some(yuv) = clip_yuv(&clip, w, h, n) else { continue };
            let p = Av1StreamingEncodeParams { width: w, height: h, quantizer: QUANTIZER, gop_size: GOP, total_frames_hint: n, colour: Default::default() };
            let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, GOP);
            src.rewind().unwrap();
            let cal = phasm_core::stego::calibration::AllocationCalibration::AV1_1080P_QP30;
            let est = phasm_core::codec::av1::stego::capacity::av1_capacity_stratified(&mut src, n, p, &cal).expect("probe");
            println!("PROBE {} {h}p max {} comfort {} {est:?}", clip.display(), est.max_message_bytes(), est.comfort_message_bytes());
        }
    }
}
