// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! W4.5 — Visual-fidelity gate. Asserts the natural AV1 encode
//! reconstructs to approximately the source YUV (PSNR_Y > floor).
//!
//! **Why this exists** (2026-05-21):
//! W3.10.5 round-trip + W4 byte/cover-bit assertions + W5 ±5% drift +
//! W6 distribution-AUC all passed on a phasm-rav1e fork path that
//! was silently encoding only the top-left quadrant of 256×144 frames.
//! The remaining ~75% of every cover frame was being reconstructed as
//! uniform gray (luma 0x82). Round-trip tests didn't catch it because
//! the decoder consumes the same partial-encoded stream symmetrically —
//! bits round-trip correctly through a destroyed cover.
//!
//! For steganography, "the encoder reconstructs the cover" is a
//! correctness gate, not a nice-to-have. A half-gray output isn't
//! hiding a message in a cover image — it's hiding a message in a
//! corrupted image. See `memory/feedback_visual_fidelity_is_correctness`.
//!
//! This file is the regression guard that *would have* caught the
//! bug. Currently it fails on the bug — that's intentional. Once the
//! root cause is fixed in the phasm-rav1e fork, it becomes a
//! permanent gate against regressions of this class.
//!
//! # What it tests
//! For each W4 fixture: source YUV → rav1e encode (via phasm_tee
//! path) → ffmpeg decode → compute Y-PSNR vs source. Assert
//! Y-PSNR ≥ `PSNR_FLOOR_DB`.
//!
//! # Floor justification
//! - Working AV1 at q=30 on natural content: ~35-40 dB typical
//! - Half-gray bug pattern: ~10-15 dB (mean abs error ~50-60 in
//!   gray regions, MSE ~3000-5000)
//! - Floor 25 dB cleanly separates the two regimes
//!
//! Tighten to 30 dB once the bug is fixed (the post-fix baseline
//! should easily exceed that).

#![cfg(all(feature = "av1-encoder", feature = "av1-backend"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::{Config, EncoderConfig, EncoderStatus};

struct Fixture {
    name: &'static str,
    source: &'static str,
    width: u32,
    height: u32,
    seek_s: f32,
    quantizer: usize,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "iphone_img4138",
        source: "IMG_4138.MOV",
        width: 256,
        height: 144,
        seek_s: 1.0,
        quantizer: 30,
    },
    Fixture {
        // Source is 1080×1920 PORTRAIT — encode at portrait dims
        // (v0.4 hygiene re-baseline).
        name: "carplane",
        source: "Artlist_CarPlane.mp4",
        width: 144,
        height: 256,
        seek_s: 2.0,
        quantizer: 30,
    },
    Fixture {
        name: "iphone5_1080p",
        source: "iphone5_1080p_30fps_h264_high.mov",
        width: 256,
        height: 144,
        seek_s: 1.0,
        quantizer: 30,
    },
];

/// PSNR floor in dB. Encoder + decoder are deterministic + lossless-
/// roundtrippable at this q range; below 25 dB means the encoder is
/// dropping content (half-gray output etc.) — that's a correctness
/// bug, not a quality knob.
const PSNR_FLOOR_DB: f64 = 25.0;

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv420_frame(spec: &Fixture) -> Vec<u8> {
    let src = corpus_root().join(spec.source);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!(
        "scale={}:{}:force_original_aspect_ratio=disable",
        spec.width, spec.height
    );
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(spec.seek_s.to_string())
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

fn encode_natural(yuv: &[u8], spec: &Fixture) -> Vec<u8> {
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
    let mut frame = make_frame::<u8>(spec.width as usize, spec.height as usize, ChromaSampling::Cs420);
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
    let (packet, _recording) = encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);
    packet
}

/// Build a single-frame IVF container around raw AV1 OBU bytes so
/// ffmpeg can decode them (it refuses raw OBU input).
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

/// Decode AV1 OBU bytes to YUV4:2:0 by wrapping in IVF + piping
/// through ffmpeg. Returns raw YUV bytes (Y plane then U plane then
/// V plane, no stride padding).
fn decode_av1_to_yuv(av1_bytes: &[u8], width: u32, height: u32) -> Vec<u8> {
    let ivf = build_ivf_single_frame(av1_bytes, width as u16, height as u16);
    let ivf_path = std::env::temp_dir().join(format!(
        "av1_psnr_decode_{}_{}.ivf",
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
    assert!(
        out.status.success(),
        "ffmpeg decode failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let expected = (width * height * 3 / 2) as usize;
    assert_eq!(out.stdout.len(), expected, "decoded YUV size mismatch");
    out.stdout
}

/// Y-plane PSNR. Both YUVs must be 4:2:0 planar with the same dims.
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

/// Count what fraction of luma pixels are essentially "neutral gray"
/// (luma in [0x7e..0x86] band). For natural content this should be
/// small; near 1.0 means the encoder is filling with default gray.
fn fraction_neutral_gray(yuv: &[u8], width: u32, height: u32) -> f64 {
    let y_size = (width * height) as usize;
    let neutral = yuv[..y_size]
        .iter()
        .filter(|&&p| (0x7e..=0x86).contains(&p))
        .count();
    neutral as f64 / y_size as f64
}

fn run_psnr_check(spec: &Fixture) -> (f64, f64, f64) {
    let source_yuv = extract_yuv420_frame(spec);
    let natural_av1 = encode_natural(&source_yuv, spec);
    let reconstructed_yuv = decode_av1_to_yuv(&natural_av1, spec.width, spec.height);

    let psnr = compute_psnr_y(&source_yuv, &reconstructed_yuv, spec.width, spec.height);
    let src_gray = fraction_neutral_gray(&source_yuv, spec.width, spec.height);
    let rec_gray = fraction_neutral_gray(&reconstructed_yuv, spec.width, spec.height);

    eprintln!(
        "[psnr] {}: Y-PSNR={:.2} dB, neutral-gray frac: source={:.3} reconstructed={:.3}",
        spec.name, psnr, src_gray, rec_gray
    );

    // On failure, dump the source + reconstructed YUVs to /tmp for
    // post-mortem inspection (ffplay -f rawvideo -pixel_format yuv420p
    // -video_size WxH /tmp/path.yuv).
    if psnr < PSNR_FLOOR_DB {
        let src_dump = std::env::temp_dir()
            .join(format!("psnr_fail_{}_source_{}x{}.yuv", spec.name, spec.width, spec.height));
        let rec_dump = std::env::temp_dir()
            .join(format!("psnr_fail_{}_recon_{}x{}.yuv", spec.name, spec.width, spec.height));
        std::fs::write(&src_dump, &source_yuv).ok();
        std::fs::write(&rec_dump, &reconstructed_yuv).ok();
        eprintln!("  → dumped source YUV: {}", src_dump.display());
        eprintln!("  → dumped reconstructed YUV: {}", rec_dump.display());
    }

    (psnr, src_gray, rec_gray)
}

/// Encode the same source YUV via the STANDARD rav1e Context API
/// (send_frame + receive_packet) — no phasm_tee shortcut. Returns
/// the raw AV1 OBU bytes from the first emitted packet.
fn encode_natural_standard_context(yuv: &[u8], spec: &Fixture) -> Vec<u8> {
    // Start from with_speed_preset(0) (best-quality preset) instead of
    // ::default() (speed=6) to rule out speed-preset-induced skip-mode
    // RDO behavior. Override only the dims + chroma + quantizer.
    let mut enc_config = EncoderConfig::with_speed_preset(0);
    enc_config.width = spec.width as usize;
    enc_config.height = spec.height as usize;
    enc_config.bit_depth = 8;
    enc_config.chroma_sampling = ChromaSampling::Cs420;
    enc_config.quantizer = spec.quantizer;
    enc_config.still_picture = true;
    let cfg = Config::new().with_encoder_config(enc_config.clone());
    let mut ctx: phasm_rav1e::Context<u8> =
        cfg.new_context().expect("standard Context creation");

    // Build the Frame via the CANONICAL constructor `ctx.new_frame()`
    // rather than `make_frame`. If THIS produces a clean encode, the
    // bug is in the phasm-stego helper's frame construction.
    let mut frame = ctx.new_frame();
    let w = spec.width as usize;
    let h = spec.height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    // Use the canonical `Plane::copy_from_raw_u8` — handles
    // rav1e's filter-tap padding rows correctly. Manual chunks_mut
    // fill puts content in the padding region instead of visible.
    frame.planes[0].copy_from_raw_u8(&yuv[..y_size], w, 1);
    frame.planes[1].copy_from_raw_u8(&yuv[y_size..y_size + uv_size], w / 2, 1);
    frame.planes[2].copy_from_raw_u8(
        &yuv[y_size + uv_size..y_size + 2 * uv_size],
        w / 2,
        1,
    );
    ctx.send_frame(std::sync::Arc::new(frame)).expect("send_frame");
    ctx.flush();

    // Pull packets until we get the first frame's OBU bytes.
    loop {
        match ctx.receive_packet() {
            Ok(pkt) => return pkt.data,
            Err(EncoderStatus::Encoded) => continue,
            Err(EncoderStatus::NeedMoreData) => continue,
            Err(EncoderStatus::LimitReached) => panic!("standard Context produced no packet"),
            Err(e) => panic!("receive_packet error: {:?}", e),
        }
    }
}

#[test]
fn diagnostic_phasm_tee_vs_standard_path() {
    let spec = &FIXTURES[1]; // carplane
    let source_yuv = extract_yuv420_frame(spec);

    // Path A: phasm_tee (the path that's broken).
    let av1_tee = encode_natural(&source_yuv, spec);
    let recon_tee = decode_av1_to_yuv(&av1_tee, spec.width, spec.height);
    let psnr_tee = compute_psnr_y(&source_yuv, &recon_tee, spec.width, spec.height);
    let gray_tee = fraction_neutral_gray(&recon_tee, spec.width, spec.height);

    // Path B: standard rav1e Context API.
    let av1_std = encode_natural_standard_context(&source_yuv, spec);
    let recon_std = decode_av1_to_yuv(&av1_std, spec.width, spec.height);
    let psnr_std = compute_psnr_y(&source_yuv, &recon_std, spec.width, spec.height);
    let gray_std = fraction_neutral_gray(&recon_std, spec.width, spec.height);

    eprintln!(
        "[isolation] phasm_tee:        bytes={}  Y-PSNR={:.2} dB  gray-frac={:.3}",
        av1_tee.len(),
        psnr_tee,
        gray_tee
    );
    eprintln!(
        "[isolation] standard Context: bytes={}  Y-PSNR={:.2} dB  gray-frac={:.3}",
        av1_std.len(),
        psnr_std,
        gray_std
    );

    // If standard >> phasm_tee, the bug is in the phasm_tee fork
    // path. If both low, the bug is in our shared EncoderConfig /
    // Frame setup. Either way, the diagnostic prints both numbers.
    if psnr_std >= 25.0 && psnr_tee < 25.0 {
        eprintln!("[isolation] DIAGNOSIS: bug is in phasm_tee fork path; standard path is fine.");
    } else if psnr_std < 25.0 && psnr_tee < 25.0 {
        eprintln!("[isolation] DIAGNOSIS: both paths broken — bug is in shared setup.");
    } else if psnr_std >= 25.0 && psnr_tee >= 25.0 {
        eprintln!("[isolation] DIAGNOSIS: both paths fine — bug fixed?");
    }
}

#[test]
fn corpus_av1_visual_psnr_gate() {
    let mut failures: Vec<String> = Vec::new();
    for spec in FIXTURES {
        let (psnr, _src_gray, rec_gray) = run_psnr_check(spec);
        if psnr < PSNR_FLOOR_DB {
            failures.push(format!(
                "{} Y-PSNR={:.2}dB <{:.0}dB (reconstructed-gray-frac={:.3})",
                spec.name, psnr, PSNR_FLOOR_DB, rec_gray
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Visual-fidelity gate FAILED. The encoder is destroying cover content. \
         For stego this is a correctness bug — the 'stego' output isn't hiding a \
         message in the original image, it's hiding it in a corrupted image. \
         Failures:\n  {}",
        failures.join("\n  ")
    );
}
