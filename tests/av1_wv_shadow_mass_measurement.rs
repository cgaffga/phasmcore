// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! WV.5b — direct stealth-mass measurement comparing per-GOP vs
//! whole-video shadow scopes.
//!
//! Instead of running the 12-feature AoSO detector (which is single-
//! frame and would require reconstructing per-frame stego pixels from
//! the assembled output), this test measures the load-bearing axis
//! directly:
//!
//! **The structural stealth advantage of whole-video shadow is
//! that the shadow message is carried ONCE across the union cover,
//! not replicated N_GOPs times.** For the same payload + passphrase,
//! per-GOP scope produces ~N_GOPs × more shadow-related cover-bit
//! flips than whole-video scope.
//!
//! Direct measurement: Hamming distance between the assembled stego
//! output's union cover and a natural (no-stego) encode of the same
//! YUV. Whole-video scope should produce SUBSTANTIALLY FEWER flips
//! than per-GOP scope on the same multi-GOP fixture.
//!
//! This isn't a steganalysis detector AUC, but it's a more direct
//! claim than AUC: detectability scales with the magnitude of stego
//! signal; fewer flips = less signal to detect. Smaller is strictly
//! better.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use phasm_core::codec::av1::stego::orchestrator::harvest_cover_bits_from_stego;
use phasm_core::codec::av1::stego::session::{
    Av1ShadowSpec, Av1StreamingEncodeParams, Av1StreamingEncodeSession,
};
use phasm_core::stego::payload;
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::phasm_stego::{
    encode_gop_with_phasm_tee, make_frame,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

const W: u32 = 256;
const H: u32 = 144;
const Q: usize = 30;
const SOURCE: &str = "IMG_4138.MOV";

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_n_yuv420_frames(n: usize) -> Vec<Vec<u8>> {
    let src = corpus_root().join(SOURCE);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!("scale={W}:{H}:force_original_aspect_ratio=disable");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args([
            "-frames:v",
            &n.to_string(),
            "-vf",
            &vf,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .expect("ffmpeg launch");
    assert!(out.status.success(), "ffmpeg yuv extract failed");
    let frame_size = (W * H * 3 / 2) as usize;
    out.stdout
        .chunks(frame_size)
        .map(|s| s.to_vec())
        .collect()
}

/// Encode N frames as a single GOP-chain naturally (no stego). The
/// concatenated per-frame OBU output mirrors the stego sessions' wire
/// format so `harvest_cover_bits_from_stego` returns a comparable
/// union cover vector.
fn encode_natural_multi_frame(yuvs: &[Vec<u8>], params: Av1StreamingEncodeParams) -> Vec<u8> {
    let gop_size = params.gop_size.max(1) as usize;
    let frame_size = (params.width as usize) * (params.height as usize) * 3 / 2;
    let y_size = (params.width as usize) * (params.height as usize);
    let uv_size = ((params.width as usize) / 2) * ((params.height as usize) / 2);

    let mut config = EncoderConfig {
        width: params.width as usize,
        height: params.height as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: params.quantizer,
        ..Default::default()
    };
    config.low_latency = true;
    config.speed_settings.multiref = false;
    let config = Arc::new(config);
    let mut sequence = Sequence::new(&config);
    sequence.enable_large_lru = false;
    let sequence = Arc::new(sequence);

    let mut output = Vec::new();
    let mut frame_idx = 0;
    while frame_idx < yuvs.len() {
        let end_idx = (frame_idx + gop_size).min(yuvs.len());
        let frames_in_gop: Vec<Arc<phasm_rav1e::Frame<u8>>> = (frame_idx..end_idx)
            .map(|i| {
                let yuv = &yuvs[i];
                let mut frame_in = make_frame::<u8>(
                    params.width as usize,
                    params.height as usize,
                    ChromaSampling::Cs420,
                );
                frame_in.planes[0].copy_from_raw_u8(&yuv[..y_size], params.width as usize, 1);
                frame_in.planes[1].copy_from_raw_u8(
                    &yuv[y_size..y_size + uv_size],
                    (params.width as usize) / 2,
                    1,
                );
                frame_in.planes[2].copy_from_raw_u8(
                    &yuv[y_size + uv_size..frame_size],
                    (params.width as usize) / 2,
                    1,
                );
                Arc::new(frame_in)
            })
            .collect();
        let pf = encode_gop_with_phasm_tee::<u8>(&frames_in_gop, config.clone(), sequence.clone());
        for (packet, _) in pf {
            output.extend_from_slice(&packet);
        }
        frame_idx = end_idx;
    }
    output
}

fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

#[test]
fn wv_total_shadow_mass_scales_inversely_with_n_gops() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260605");
    }

    let primary_text = "wv-mass primary message — substantive";
    let shadow_text = "wv-mass shadow — also substantive";
    let primary_pass = "wv-mass-primary";
    let shadow_pass = "wv-mass-shadow";

    let primary_payload = payload::encode_payload(primary_text, &[]).expect("encode primary");
    let shadow_payload = payload::encode_payload(shadow_text, &[]).expect("encode shadow");

    let n_frames = 12usize;
    let gop_size = 4u32;
    let n_gops = (n_frames as u32).div_ceil(gop_size) as usize;
    assert_eq!(n_gops, 3, "test geometry assumes 3 GOPs");

    let yuvs = extract_n_yuv420_frames(n_frames);

    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size,
        total_frames_hint: n_frames as u32,
    };

    // 1. Natural encode baseline.
    let natural_bytes = encode_natural_multi_frame(&yuvs, params);
    let natural_cover = harvest_cover_bits_from_stego(&natural_bytes).expect("harvest natural");

    // 2. Per-GOP shadow scope encode.
    let shadows_pg = vec![Av1ShadowSpec {
        passphrase: shadow_pass.to_string(),
        message: shadow_payload.clone(),
    }];
    let mut sess_pg = Av1StreamingEncodeSession::create_with_shadows(
        primary_pass,
        &primary_payload,
        params,
        shadows_pg,
        16,
    )
    .expect("create per-GOP session");
    let mut pg_bytes = Vec::new();
    for yuv in &yuvs {
        sess_pg.push_frame(yuv, &mut pg_bytes).expect("push per-GOP");
    }
    sess_pg.finish(&mut pg_bytes).expect("finish per-GOP");
    let pg_cover = harvest_cover_bits_from_stego(&pg_bytes).expect("harvest per-GOP");

    // 3. Whole-video shadow scope encode.
    let shadows_wv = vec![Av1ShadowSpec {
        passphrase: shadow_pass.to_string(),
        message: shadow_payload,
    }];
    let mut sess_wv = Av1StreamingEncodeSession::create_whole_video_with_shadows(
        primary_pass,
        &primary_payload,
        params,
        shadows_wv,
        16,
    )
    .expect("create WV session");
    let mut wv_bytes = Vec::new();
    for yuv in &yuvs {
        sess_wv.push_frame(yuv, &mut wv_bytes).expect("push WV");
    }
    sess_wv.finish(&mut wv_bytes).expect("finish WV");
    let wv_cover = harvest_cover_bits_from_stego(&wv_bytes).expect("harvest WV");

    // 4. Both stego covers should have the SAME length as natural
    //    (we're flipping bits, not adding them).
    eprintln!(
        "[WV.5b] cover sizes — natural: {} bits | per-GOP stego: {} | WV stego: {}",
        natural_cover.len(),
        pg_cover.len(),
        wv_cover.len()
    );
    let min_len = natural_cover.len().min(pg_cover.len()).min(wv_cover.len());

    let pg_flips = hamming_distance(&natural_cover[..min_len], &pg_cover[..min_len]);
    let wv_flips = hamming_distance(&natural_cover[..min_len], &wv_cover[..min_len]);

    eprintln!(
        "[WV.5b] flips vs natural: per-GOP scope = {} | whole-video scope = {} | ratio = {:.2}x",
        pg_flips,
        wv_flips,
        pg_flips as f64 / wv_flips.max(1) as f64,
    );

    // 5. Load-bearing assertion: whole-video scope should produce
    //    substantially fewer flips than per-GOP scope on the same
    //    fixture. The structural ratio is ~N_GOPs (here = 3) since
    //    per-GOP replicates the shadow N times. Real encodes are
    //    noisy (primary STC overlap with shadow positions, RS
    //    expansion granularity), so we use a conservative threshold:
    //    WV must be at least 1.4× LESS than per-GOP. Higher gop counts
    //    + larger shadow payloads make the gap bigger.
    let min_ratio = 1.4;
    let actual_ratio = pg_flips as f64 / wv_flips.max(1) as f64;
    assert!(
        actual_ratio >= min_ratio,
        "whole-video shadow scope must produce significantly fewer flips than per-GOP scope. \
         per-GOP {} flips vs WV {} flips (ratio {:.2}x, expected >= {:.1}x). \
         Either per-GOP is over-spreading or WV is over-concentrating shadow mass.",
        pg_flips,
        wv_flips,
        actual_ratio,
        min_ratio,
    );

    // 6. Both must still produce non-zero flips (sanity: the shadow
    //    actually got embedded in both modes).
    assert!(
        pg_flips > 0 && wv_flips > 0,
        "both modes must have non-zero flips (shadow + primary should always produce some)"
    );
}
