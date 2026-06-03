// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! TG-2 — GOP-boundary probe.
//!
//! Catches V5-class cliffs: stego damage varying wildly across
//! independent STC units. In H.264 this was a -14 dB Y-PSNR drop
//! at GOP 1 frame 31 — caught only by the V5 production probe,
//! AFTER all round-trip + cover-bit + drift + AUC gates passed.
//! See `memory/h264_v5_production_finding_2026_05_07.md` +
//! `memory/h264_v6_per_gop_stc_2026_05_07.md` (the V6 fix).
//!
//! ## How AV1 v0.5 simulates "multi-GOP" today
//!
//! AV1 today emits one keyframe + no inter frames per encode session
//! — each session is structurally a single GOP with a single STC
//! unit. To stress-test the property V5 was supposed to have but
//! didn't (consistent stego damage across independent STC units),
//! we run N encode-and-embed cycles on consecutive frames extracted
//! from the same video at incrementing seek times. Each cycle is
//! independent. If STC plan determinism, cost compute, or any other
//! per-session machinery diverges across cycles, the gate trips.
//!
//! When multi-GOP scale-up lands
//! ([`phase-c-streaming-session-v6.md`](../../docs/design/video/av1/phase-c-streaming-session-v6.md)),
//! this test extends naturally to a single-encode multi-GOP probe
//! without changing the assertion shape.
//!
//! ## What it asserts
//!
//! 1. **Damage cliff** — `max(stego_damage_dB_i) -
//!    min(stego_damage_dB_i) <= 2.0 dB`. The V5 cliff was 12+ dB.
//!    Natural per-frame damage variance is well under 1 dB on the
//!    same content.
//! 2. **Capacity bound** — `max(cover_bits_i) <= 5 ×
//!    min(cover_bits_i)`. Natural per-frame motion variance on the
//!    same source rarely exceeds 3×; catastrophic encoder config
//!    drift produces 10×+.
//!
//! Both metrics are CONTENT-INVARIANT: they measure variance across
//! sessions, not absolute values, so they tolerate natural per-frame
//! content variation without false-firing on it.
//!
//! See [`phase-c-test-gates.md`](../../docs/design/video/av1/phase-c-test-gates.md)
//! § 2 for the full design rationale.

#![cfg(all(feature = "av1-encoder", feature = "av1-backend"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use phasm_core::codec::av1::stego::orchestrator::{av1_stego_embed, av1_stego_extract};
use phasm_core::codec::av1::stego::writer::CoverPositions;
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::ec::PhasmFrameRecording;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
    PHASM_TAG_AC_COEFF_SIGN,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

/// Number of consecutive frames probed per fixture. 3 matches the
/// frame count used elsewhere in the AV1 corpus + bench suite.
const NUM_FRAMES: usize = 3;

/// Damage cliff allowed across frames (in dB). V5 saw a 12+ dB drop;
/// natural per-frame stego damage variance on consecutive frames of
/// the same content is well under 1 dB. 2.0 dB is the threshold that
/// catches catastrophic without false-firing on natural variance.
const PSNR_DAMAGE_CLIFF_DB: f64 = 2.0;

/// Capacity bound: `max(cover_bits_i) <= MULTIPLIER × min(cover_bits_i)`.
/// Natural content variance on 3 consecutive frames of the same
/// video rarely exceeds 3×; catastrophic encoder config drift would
/// produce 10×+. 5.0 sits cleanly between the two regimes.
const CAPACITY_RATIO_MAX: f64 = 5.0;

struct Fixture {
    name: &'static str,
    source: &'static str,
    width: u32,
    height: u32,
    /// First frame seek (sec). Subsequent frames step by `seek_step_s`.
    first_seek_s: f32,
    seek_step_s: f32,
    quantizer: usize,
}

/// Single fixture for now: carplane (1080×1920 portrait, motion-heavy).
/// Add more fixtures here when needed; the harness is content-blind.
const FIXTURE: Fixture = Fixture {
    name: "carplane",
    source: "Artlist_CarPlane.mp4",
    width: 144,
    height: 256,
    first_seek_s: 0.5,
    seek_step_s: 0.5,
    quantizer: 30,
};

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv420_frame(spec: &Fixture, seek_s: f32) -> Vec<u8> {
    let src = corpus_root().join(spec.source);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!(
        "scale={}:{}:force_original_aspect_ratio=disable",
        spec.width, spec.height
    );
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args([
            "-frames:v", "1", "-vf", &vf, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
        ])
        .output()
        .expect("ffmpeg launch");
    assert!(out.status.success(), "ffmpeg yuv extract failed");
    let expected = (spec.width * spec.height * 3 / 2) as usize;
    assert_eq!(out.stdout.len(), expected);
    out.stdout
}

fn encode_natural_with_recording(
    yuv: &[u8],
    spec: &Fixture,
) -> (Vec<u8>, PhasmFrameRecording<u8>) {
    let config = Arc::new(EncoderConfig {
        width: spec.width as usize,
        height: spec.height as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: spec.quantizer,
        ..Default::default()
    });
    let mut sequence = Sequence::new(&config);
    sequence.enable_large_lru = false;
    let mut fi = FrameInvariants::<u8>::new_key_frame(
        config.clone(),
        Arc::new(sequence),
        0,
        Box::new([]),
    );
    fi.enable_segmentation = false;
    let mut frame =
        make_frame::<u8>(spec.width as usize, spec.height as usize, ChromaSampling::Cs420);
    let w = spec.width as usize;
    let h = spec.height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    frame.planes[0].copy_from_raw_u8(&yuv[..y_size], w, 1);
    frame.planes[1].copy_from_raw_u8(&yuv[y_size..y_size + uv_size], w / 2, 1);
    frame.planes[2].copy_from_raw_u8(
        &yuv[y_size + uv_size..y_size + 2 * uv_size],
        w / 2,
        1,
    );
    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame));
    let inter_cfg = make_inter_config(&config);
    encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg)
}

fn build_ivf_single_frame(obus: &[u8], width: u16, height: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + 12 + obus.len());
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"AV01");
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(obus.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(obus);
    out
}

fn decode_av1_to_yuv(av1_bytes: &[u8], width: u32, height: u32) -> Vec<u8> {
    let ivf = build_ivf_single_frame(av1_bytes, width as u16, height as u16);
    let ivf_path = std::env::temp_dir().join(format!(
        "tg2_decode_{}_{}.ivf",
        std::process::id(),
        width
    ));
    std::fs::write(&ivf_path, &ivf).expect("write ivf");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&ivf_path)
        .args([
            "-frames:v", "1", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
        ])
        .output()
        .expect("ffmpeg decode launch");
    std::fs::remove_file(&ivf_path).ok();
    assert!(out.status.success(), "ffmpeg decode failed");
    out.stdout
}

fn compute_psnr_y(source: &[u8], reconstructed: &[u8], width: u32, height: u32) -> f64 {
    let y_size = (width * height) as usize;
    assert!(source.len() >= y_size);
    assert!(reconstructed.len() >= y_size);
    let mut sum_sq_err: u64 = 0;
    for i in 0..y_size {
        let diff = source[i] as i32 - reconstructed[i] as i32;
        sum_sq_err += (diff * diff) as u64;
    }
    let mse = sum_sq_err as f64 / y_size as f64;
    if mse < 0.001 {
        return 100.0;
    }
    10.0 * (65025.0 / mse).log10()
}

#[test]
fn tg2_gop_boundary_probe_carplane() {
    // Force deterministic crypto salt+nonce so the cliff measurement
    // is reproducible. Without this, every embed call gets a random
    // salt+nonce → different payload bits → different STC plan →
    // different stego pixels. Production paths leave this unset.
    // Diagnosed B.1.5.6 session 2026-06-03 — diagnostic-only env var
    // already wired in `src/stego/crypto.rs:198`.
    // SAFETY: TG-2 is a single test in its own binary; no other
    // tests share this process. Safe to set process-wide env.
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260603");
    }

    let spec = &FIXTURE;
    let message = b"TG-2 GOP boundary probe";
    let passphrase = "tg2-gop-boundary-2026-06-03";

    let mut natural_psnr = Vec::with_capacity(NUM_FRAMES);
    let mut stego_psnr = Vec::with_capacity(NUM_FRAMES);
    let mut stego_damage = Vec::with_capacity(NUM_FRAMES);
    let mut cover_bits = Vec::with_capacity(NUM_FRAMES);

    for frame_idx in 0..NUM_FRAMES {
        let seek_s = spec.first_seek_s + frame_idx as f32 * spec.seek_step_s;
        let source_yuv = extract_yuv420_frame(spec, seek_s);
        let (natural_packet, recording) = encode_natural_with_recording(&source_yuv, spec);

        // Natural Y-PSNR.
        let natural_decoded = decode_av1_to_yuv(&natural_packet, spec.width, spec.height);
        let n_psnr = compute_psnr_y(&source_yuv, &natural_decoded, spec.width, spec.height);

        // Cover-bit count (AC_COEFF_SIGN positions in the recording).
        let tile = &recording.tiles[0];
        let positions = CoverPositions::from_recorder(&tile.bit_positions, &tile.bit_tags);
        let n_cover = positions.count_by_tag(PHASM_TAG_AC_COEFF_SIGN);

        // Embed + stego Y-PSNR.
        let stego_packet =
            av1_stego_embed(natural_packet.clone(), recording.clone(), message, passphrase)
                .unwrap_or_else(|e| {
                    panic!(
                        "[TG-2] frame {} (seek {:.1}s): av1_stego_embed failed: {:?}",
                        frame_idx, seek_s, e
                    )
                });
        let stego_decoded = decode_av1_to_yuv(&stego_packet, spec.width, spec.height);
        let s_psnr = compute_psnr_y(&source_yuv, &stego_decoded, spec.width, spec.height);

        // Round-trip sanity per frame.
        let extracted = av1_stego_extract(&stego_packet, passphrase).unwrap_or_else(|e| {
            panic!(
                "[TG-2] frame {} (seek {:.1}s): av1_stego_extract failed: {:?}",
                frame_idx, seek_s, e
            )
        });
        assert_eq!(
            extracted.as_slice(),
            message,
            "[TG-2] frame {} (seek {:.1}s) round-trip mismatch",
            frame_idx, seek_s
        );

        let damage = n_psnr - s_psnr;
        let s_vs_n_psnr = compute_psnr_y(&natural_decoded, &stego_decoded, spec.width, spec.height);
        let len_delta = stego_packet.len() as i64 - natural_packet.len() as i64;
        let rebuild_fired = len_delta != 0;
        eprintln!(
            "[TG-2] {} frame {} (seek {:.1}s): natural Y-PSNR {:.2} dB, stego Y-PSNR {:.2} dB, \
             damage {:.3} dB, stego-vs-natural-decoded {:.2} dB, packet {} vs {} (Δ {:+}, OBU-rebuild={}), \
             cover_bits_ac_sign {}",
            spec.name, frame_idx, seek_s, n_psnr, s_psnr, damage, s_vs_n_psnr,
            stego_packet.len(), natural_packet.len(), len_delta, rebuild_fired,
            n_cover
        );

        // Dump YUVs to /tmp for visual inspection when TG-2 is debugged.
        let dump_dir = std::env::temp_dir();
        let src_path = dump_dir.join(format!("tg2_{}_f{}_source.yuv", spec.name, frame_idx));
        let nat_path = dump_dir.join(format!("tg2_{}_f{}_natural.yuv", spec.name, frame_idx));
        let stg_path = dump_dir.join(format!("tg2_{}_f{}_stego.yuv", spec.name, frame_idx));
        std::fs::write(&src_path, &source_yuv).ok();
        std::fs::write(&nat_path, &natural_decoded).ok();
        std::fs::write(&stg_path, &stego_decoded).ok();

        natural_psnr.push(n_psnr);
        stego_psnr.push(s_psnr);
        stego_damage.push(damage);
        cover_bits.push(n_cover);
    }

    // ---- 1. Damage cliff gate ----
    let damage_max = stego_damage.iter().cloned().fold(f64::MIN, f64::max);
    let damage_min = stego_damage.iter().cloned().fold(f64::MAX, f64::min);
    let damage_cliff = damage_max - damage_min;

    assert!(
        damage_cliff <= PSNR_DAMAGE_CLIFF_DB,
        "[{}] GOP-boundary cliff: stego_damage variance {:.3} dB exceeds threshold {} dB. \
         Per-frame damage values: {:?}. This means independent STC units (each frame == one GOP) \
         produced wildly different visual quality — the V5-class bug class. See \
         `memory/h264_v5_production_finding_2026_05_07.md`.",
        spec.name, damage_cliff, PSNR_DAMAGE_CLIFF_DB, stego_damage
    );

    // ---- 2. Capacity bound gate ----
    let cap_max = *cover_bits.iter().max().unwrap();
    let cap_min = cover_bits.iter().copied().min().unwrap().max(1);
    let cap_ratio = cap_max as f64 / cap_min as f64;

    assert!(
        cap_ratio <= CAPACITY_RATIO_MAX,
        "[{}] Capacity bound: cover_bits varies {:.2}× across frames (max {} / min {}, values {:?}). \
         Catastrophic encoder config drift would produce > {}× variance.",
        spec.name, cap_ratio, cap_max, cap_min, cover_bits, CAPACITY_RATIO_MAX
    );
}
