// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Measurement harness: does a primary payload at the capacity the mobile
//! HUD advertises actually fit through the AV1 production encoder?
//!
//! Written 2026-10-23 to measure the even split (it failed at short
//! trailing GOPs); kept as the end-to-end check of the proportional plan
//! that replaced it. See
//! `docs/design/video/av1/capacity-split-and-payload-limits-2026-10-23.md`.
//!
//! Mirrors the mobile path exactly:
//!   - HUD number = `av1_capacity_stratified(.., AV1_1080P_QP30)`
//!     `.primary_max_message_bytes` (iOS `probeAv1CapacityStratified`,
//!     Android `probeStratifiedCapacity`, both with `kMin == 0` → 8).
//!   - Encode = `av1_encode_streaming` (the no-shadow entry both bridges
//!     call), quantizer 100, gop_size 30 (`PhasmAv1Transcoder` defaults).
//!   - Native resolution, cropped down to a multiple of 8 (AV1 alignment).
//!
//! Driven by env vars so one binary covers the corpus:
//!   PHASM_AV1_CAPCHECK_CLIP    source clip path (required)
//!   PHASM_AV1_CAPCHECK_FRAMES  optional frame cap (trim, e.g. to drop a
//!                              partial trailing GOP)
//!   PHASM_AV1_CAPCHECK_FILLS   comma list of fractions of the advertised
//!                              capacity, tried in order until one passes
//!                              (default "0.9,0.5")
//!   PHASM_AV1_CAPCHECK_OUT     optional file to append the summary line to
//!   PHASM_TMP                  scratch base for the decoded y4m
//!
//! ```bash
//! PHASM_AV1_CAPCHECK_CLIP=/path/clip.mp4 cargo test --release \
//!   --features av1-encoder,av1-decoder --test av1_capacity_probe \
//!   -- --ignored --nocapture
//! ```

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::io::Write as _;
use std::time::Instant;

use phasm_core::codec::av1::stego::session::Av1StreamingDecodeSession;
use phasm_core::codec::av1::stego::whole_video::{
    av1_encode_streaming, Av1GopYuvSource, Av1SliceYuvSource,
};
use phasm_core::stego::calibration::AllocationCalibration;
use phasm_core::stego::{crypto, frame, payload};
use phasm_core::{av1_capacity_stratified, Av1StreamingEncodeParams};

const QUANTIZER: usize = 100;
const GOP_SIZE: u32 = 30;
const PASSPHRASE: &str = "av1-even-split-capacity-probe";
/// The largest framed primary a video decoder accepts
/// (`frame::VIDEO_MAX_FRAME_BYTES`: the 16 MiB inner-payload limit plus
/// framing). The 256 KiB `frame::MAX_FRAME_BYTES` bounds image frames only;
/// video decoders stopped using it on 2026-10-23. The encode refuses a
/// frame over this bound, so a flagged fill is expected to fail.
const VIDEO_DECODE_MAX_FRAME_BYTES: usize = frame::VIDEO_MAX_FRAME_BYTES;

/// Decode `clip` to a cropped-to-8 I420 y4m in scratch (ffmpeg applies
/// autorotation, so the y4m header carries the dims the phone would see),
/// then return `(w, h, n_frames, concatenated I420)`.
fn load_clip(clip: &str, max_frames: Option<u32>) -> (u32, u32, u32, Vec<u8>) {
    let base = std::env::var("PHASM_TMP")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "/tmp".into());
    let dir = format!("{base}/phasm-av1-evensplit");
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let stem = std::path::Path::new(clip)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let tag = max_frames.map(|n| format!("_{n}f")).unwrap_or_default();
    let y4m = format!("{dir}/{stem}{tag}.y4m");

    let mut args: Vec<String> = vec!["-y".into(), "-loglevel".into(), "error".into()];
    args.extend(["-i".into(), clip.into()]);
    if let Some(n) = max_frames {
        args.extend(["-frames:v".into(), n.to_string()]);
    }
    args.extend([
        "-vf".into(),
        "crop=trunc(iw/8)*8:trunc(ih/8)*8:0:0".into(),
        "-pix_fmt".into(),
        "yuv420p".into(),
        "-strict".into(),
        "-1".into(),
        y4m.clone(),
    ]);
    let st = std::process::Command::new("ffmpeg")
        .args(&args)
        .status()
        .expect("ffmpeg launch");
    assert!(st.success(), "ffmpeg y4m decode failed for {clip}");

    let raw = std::fs::read(&y4m).expect("read y4m");
    let hdr_end = raw.iter().position(|&b| b == b'\n').expect("y4m header");
    let hdr = std::str::from_utf8(&raw[..hdr_end]).unwrap();
    let mut w = 0u32;
    let mut h = 0u32;
    for tok in hdr.split(' ') {
        if let Some(v) = tok.strip_prefix('W') {
            w = v.parse().unwrap();
        } else if let Some(v) = tok.strip_prefix('H') {
            h = v.parse().unwrap();
        }
    }
    assert!(w > 0 && h > 0, "bad y4m header: {hdr}");
    let frame_bytes = (w * h * 3 / 2) as usize;
    let mut yuv = Vec::with_capacity(raw.len());
    let mut p = hdr_end + 1;
    let mut n = 0u32;
    while p < raw.len() {
        let fh_end = p + raw[p..].iter().position(|&b| b == b'\n').expect("FRAME hdr");
        p = fh_end + 1;
        yuv.extend_from_slice(&raw[p..p + frame_bytes]);
        p += frame_bytes;
        n += 1;
    }
    let _ = std::fs::remove_file(&y4m);
    (w, h, n, yuv)
}

/// Deterministic high-entropy printable text whose `encode_payload` output
/// is as close to `target` bytes as possible without exceeding it.
fn text_for_payload_len(target: usize) -> String {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let pool: String = (0..target * 2 + 64)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (33 + (s % 94) as u8) as char
        })
        .collect();
    let plen = |n: usize| payload::encode_payload(&pool[..n], &[]).unwrap().len();
    let (mut lo, mut hi) = (0usize, pool.len());
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if plen(mid) <= target {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    pool[..lo].to_string()
}

#[test]
#[ignore = "measurement harness: minutes-to-hours of AV1 encode; needs PHASM_AV1_CAPCHECK_CLIP"]
fn av1_capacity_probe() {
    let clip = std::env::var("PHASM_AV1_CAPCHECK_CLIP").expect("set PHASM_AV1_CAPCHECK_CLIP");
    let max_frames = std::env::var("PHASM_AV1_CAPCHECK_FRAMES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok());
    let fills: Vec<f64> = std::env::var("PHASM_AV1_CAPCHECK_FILLS")
        .unwrap_or_else(|_| "0.9,0.5".into())
        .split(',')
        .map(|s| s.trim().parse().expect("fill"))
        .collect();
    let clip_name = std::path::Path::new(&clip)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();

    let (w, h, n_frames, yuv) = load_clip(&clip, max_frames);
    let n_gops = n_frames.div_ceil(GOP_SIZE);
    let tail_frames = n_frames - (n_gops - 1) * GOP_SIZE;
    println!(
        "[capcheck] {clip_name}: {w}x{h} × {n_frames} frames → {n_gops} GOPs \
         (trailing GOP {tail_frames} frames)"
    );

    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: QUANTIZER,
        gop_size: GOP_SIZE,
        total_frames_hint: n_frames,
    };

    // HUD number, exactly as the mobile probe computes it. The probe is
    // deterministic, so PHASM_AV1_CAPCHECK_ADVERTISED can replay a number
    // measured by an earlier run and skip the probe's wall-clock.
    let advertised = match std::env::var("PHASM_AV1_CAPCHECK_ADVERTISED")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(a) => {
            println!("[capcheck] stratified SKIPPED: advertised primary {a} B (replayed)");
            a
        }
        None => {
            let t0 = Instant::now();
            let mut src = Av1SliceYuvSource::new(&yuv, w, h, n_frames, GOP_SIZE);
            src.rewind().expect("rewind");
            let cap = av1_capacity_stratified(
                &mut src,
                n_frames,
                params,
                &AllocationCalibration::AV1_1080P_QP30,
            )
            .expect("stratified capacity");
            let probe_s = t0.elapsed().as_secs_f64();
            let mean = cap.cover_bits_mean_per_gop.max(1) as f64;
            println!(
                "[capcheck] stratified ({probe_s:.0}s): mean {} cb/GOP, lower99.9 {}, safe {}, \
                 advertised primary {} B",
                cap.cover_bits_mean_per_gop,
                cap.cover_bits_lower_99_9_per_gop,
                cap.cover_bits_pessimism_adjusted_per_gop,
                cap.primary_max_message_bytes,
            );
            for &(g, cb) in &cap.samples {
                println!(
                    "[capcheck]   sample gop {g:3}: {cb:>10} cb ({:.2}× mean)",
                    cb as f64 / mean
                );
            }
            cap.primary_max_message_bytes
        }
    };

    let mut summary = format!(
        "{clip_name}\t{w}x{h}\tframes={n_frames}\tgops={n_gops}\ttail={tail_frames}\t\
         advertised={advertised}"
    );

    for &fill in &fills {
        let target = ((advertised as f64) * fill).floor() as usize;
        if target == 0 {
            println!("[capcheck] fill {fill}: advertised capacity is 0 — nothing to encode");
            summary.push_str(&format!("\tfill={fill}:ZERO_CAP"));
            break;
        }
        let text = text_for_payload_len(target);
        let primary_payload = payload::encode_payload(&text, &[]).unwrap();
        let (ct, nonce, salt) = crypto::encrypt(&primary_payload, PASSPHRASE).unwrap();
        let framed = frame::build_frame(primary_payload.len(), &salt, &nonce, &ct);
        let per_gop = framed.len() as f64 / n_gops as f64;
        println!(
            "[capcheck] fill {fill}: payload {} B (target {target}), framed {} B, \
             even share {per_gop:.0} B/GOP{}",
            primary_payload.len(),
            framed.len(),
            if framed.len() > VIDEO_DECODE_MAX_FRAME_BYTES {
                " — OVER the video decode bound (frame::VIDEO_MAX_FRAME_BYTES)"
            } else {
                ""
            }
        );

        let mut src = Av1SliceYuvSource::new(&yuv, w, h, n_frames, GOP_SIZE);
        // GOPs embedded so far (Pass 2): a failure is at the next one.
        let mut done_gops = 0u32;
        let mut tick = |e: phasm_core::codec::av1::stego::encode_progress::Av1EncodeProgress| {
            if e.phase == phasm_core::codec::av1::stego::encode_progress::Av1EncodePhase::Embed {
                done_gops = e.gop;
            }
            println!(
                "[capcheck]   {:?} gop {}/{} ok ({:.1} %)",
                e.phase,
                e.gop,
                e.n_gops,
                e.fraction * 100.0
            );
        };
        let t1 = Instant::now();
        let res = av1_encode_streaming(
            &mut src,
            n_frames,
            params,
            &framed,
            PASSPHRASE,
            phasm_core::codec::av1::stego::whole_video::Pass1CacheBudget::from_env_or_default(),
            Some(&mut tick),
        );
        let enc_s = t1.elapsed().as_secs_f64();
        match res {
            Ok(obus) => {
                let mut dec = Av1StreamingDecodeSession::create(PASSPHRASE);
                dec.push_bytes(&obus);
                let verdict = match dec.finish_primary_payload() {
                    Ok(p) if p.text == text => "PASS".to_string(),
                    Ok(_) => "DECODE_MISMATCH".to_string(),
                    Err(e) => format!("DECODE_ERR({e:?})"),
                };
                println!("[capcheck] fill {fill}: {verdict} ({enc_s:.0}s encode)");
                summary.push_str(&format!("\tfill={fill}:{verdict}:{enc_s:.0}s"));
                if verdict == "PASS" {
                    break;
                }
            }
            Err(e) => {
                println!(
                    "[capcheck] fill {fill}: FAIL at gop {done_gops} of {n_gops} after \
                     {enc_s:.0}s — {e:?}"
                );
                summary.push_str(&format!(
                    "\tfill={fill}:FAIL@gop{done_gops}/{n_gops}:{enc_s:.0}s:{e:?}"
                ));
            }
        }
    }

    println!("[capcheck] SUMMARY {summary}");
    if let Ok(out) = std::env::var("PHASM_AV1_CAPCHECK_OUT") {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(out)
            .expect("open summary out");
        writeln!(f, "{summary}").unwrap();
    }
}
