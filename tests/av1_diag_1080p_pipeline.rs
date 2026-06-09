// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! F.7-diag — root-cause ladder for the 1088×1920 (~1080p) primary
//! extract failure surfaced by F.7's deferred `f7_full_1080p_2shadow_at_capacity`.
//!
//! No existing AV1 stego test exercises native 1080p — all the
//! corpus_validation fixtures are downscaled to 256×144. F.7's
//! attempt at native resolution was the first, and the primary
//! extract failed. This diagnostic ladder isolates which pipeline
//! stage breaks.
//!
//! Ladder (each stage builds on the prior):
//!
//! 1. `diag_1080p_natural_round_trip` — rav1e encode + dav1d
//!    decode + ffmpeg PSNR. NO embed at all. Confirms the codec
//!    layer alone works at 1088×1920.
//!
//! 2. `diag_1080p_primary_embed_extract_no_shadow` — full primary
//!    flow with `av1_stego_embed` (no shadows). Confirms STC +
//!    replay + extract work at the larger cover.
//!
//! 3. `diag_1080p_primary_plus_one_shadow` — primary + 1 shadow.
//!    Confirms the F.6 shadow integration works at scale.
//!
//! 4. `diag_1080p_primary_plus_two_shadows` — F.7's failing case.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;

use phasm_core::codec::av1::stego::orchestrator::{
    av1_stego_embed, av1_stego_embed_payload_bits_with_shadows_parity, av1_stego_extract,
    av1_stego_extract_shadow,
};
use phasm_core::stego::{crypto, frame, payload};
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::ec::PhasmFrameRecording;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

// AV1 width must be multiple of 8 for the rav1e tile-grid encoder.
// 1088×1920 (close to standard 1080p portrait) is the natural
// rotated framing for the iphone5 vertical-portrait corpus fixture.
const W: u32 = 1088;
const H: u32 = 1920;
const Q: usize = 30;

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv(seek_s: f32) -> Vec<u8> {
    let src = corpus_root().join("iphone5_1080p_30fps_h264_high.mov");
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!("scale={W}:{H}:force_original_aspect_ratio=disable");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args([
            "-frames:v",
            "1",
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
    let expected = (W * H * 3 / 2) as usize;
    assert_eq!(out.stdout.len(), expected);
    out.stdout
}

fn encode_natural(yuv: &[u8]) -> (Vec<u8>, PhasmFrameRecording<u8>) {
    let config = Arc::new(EncoderConfig {
        width: W as usize,
        height: H as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: Q,
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

    let mut frame_in = make_frame::<u8>(W as usize, H as usize, ChromaSampling::Cs420);
    let w = W as usize;
    let h = H as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    frame_in.planes[0].copy_from_raw_u8(&yuv[..y_size], w, 1);
    frame_in.planes[1].copy_from_raw_u8(&yuv[y_size..y_size + uv_size], w / 2, 1);
    frame_in.planes[2].copy_from_raw_u8(
        &yuv[y_size + uv_size..y_size + 2 * uv_size],
        w / 2,
        1,
    );

    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame_in));
    let inter_cfg = make_inter_config(&config);
    encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg)
}

fn build_ivf(obus: &[u8], width: u16, height: u16) -> Vec<u8> {
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

fn decode_with_ffmpeg(av1_bytes: &[u8]) -> Vec<u8> {
    let ivf = build_ivf(av1_bytes, W as u16, H as u16);
    let mut child = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i", "pipe:0"])
        .args([
            "-frames:v", "1", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("ffmpeg decode spawn");
    child
        .stdin
        .as_mut()
        .expect("ffmpeg stdin")
        .write_all(&ivf)
        .expect("write ivf");
    let out = child.wait_with_output().expect("ffmpeg wait");
    assert!(out.status.success(), "ffmpeg decode failed: {:?}", out);
    out.stdout
}

fn psnr_y(source: &[u8], reconstructed: &[u8]) -> f64 {
    let y_size = (W * H) as usize;
    let mut sum_sq: u64 = 0;
    for i in 0..y_size {
        let d = source[i] as i32 - reconstructed[i] as i32;
        sum_sq += (d * d) as u64;
    }
    let mse = sum_sq as f64 / y_size as f64;
    if mse < 0.001 {
        return 100.0;
    }
    10.0 * (65025.0 / mse).log10()
}

/// Stage 1: pure codec round-trip. No embed at all.
#[test]
fn diag_1080p_natural_round_trip() {
    let yuv = extract_yuv(1.0);
    let (packet, recording) = encode_natural(&yuv);
    let tile = &recording.tiles[0];
    let total_bit_positions = tile.bit_positions.len();
    let n_ac_sign = tile
        .bit_tags
        .iter()
        .filter(|&&t| t == phasm_rav1e::phasm_stego::PHASM_TAG_AC_COEFF_SIGN)
        .count();
    let n_golomb_tail = tile
        .bit_tags
        .iter()
        .filter(|&&t| t == phasm_rav1e::phasm_stego::PHASM_TAG_GOLOMB_TAIL_LSB)
        .count();
    eprintln!(
        "[diag-1080p Stage 1] natural packet {} bytes, total bit_positions {}, AC sign {}, golomb tail {}",
        packet.len(), total_bit_positions, n_ac_sign, n_golomb_tail
    );
    let yuv_back = decode_with_ffmpeg(&packet);
    let psnr = psnr_y(&yuv, &yuv_back);
    eprintln!("[diag-1080p Stage 1] natural Y-PSNR vs source: {psnr:.2} dB");
    assert!(
        psnr > 35.0,
        "natural Y-PSNR at 1088×1920 should exceed 35 dB; got {psnr}"
    );
}

/// Stage 2: primary embed + extract via the legacy single-frame API.
/// No shadows.
#[test]
fn diag_1080p_primary_embed_extract_no_shadow() {
    let yuv = extract_yuv(1.0);
    let (natural_packet, recording) = encode_natural(&yuv);
    let n_ac_sign = recording.tiles[0]
        .bit_tags
        .iter()
        .filter(|&&t| t == phasm_rav1e::phasm_stego::PHASM_TAG_AC_COEFF_SIGN)
        .count();
    eprintln!(
        "[diag-1080p Stage 2] natural packet {} bytes, n_ac_sign {}",
        natural_packet.len(),
        n_ac_sign
    );

    let message = b"diag 1080p primary only";
    let passphrase = "diag-1080p";
    let stego = av1_stego_embed(natural_packet, recording, message, passphrase)
        .expect("primary embed (1080p, no shadows)");
    eprintln!("[diag-1080p Stage 2] stego packet {} bytes", stego.len());

    let recovered = av1_stego_extract(&stego, passphrase).expect("primary extract (1080p)");
    assert_eq!(recovered.as_slice(), message);
}

/// Stage 3: primary + 1 shadow.
#[test]
fn diag_1080p_primary_plus_one_shadow() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260605");
    }
    let yuv = extract_yuv(1.0);
    let (natural_packet, recording) = encode_natural(&yuv);

    let primary_msg = b"primary 1080p+1shadow";
    let primary_pass = "primary-1080p-1s";
    let shadow_msg = "shadow 1080p test";
    let shadow_payload = payload::encode_payload(shadow_msg, &[]).unwrap();
    let shadow_pass = "shadow-1080p-1s";

    let structural_key = crypto::derive_structural_key(primary_pass).unwrap();
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();
    let (ct, nonce, salt) = crypto::encrypt(primary_msg, primary_pass).unwrap();
    let primary_frame = frame::build_frame(primary_msg.len(), &salt, &nonce, &ct);
    let primary_bits = frame::bytes_to_bits(&primary_frame);

    let stego = av1_stego_embed_payload_bits_with_shadows_parity(
        natural_packet,
        recording,
        &primary_bits,
        &hhat_seed,
        &[(shadow_pass, &shadow_payload)],
        16,
    )
    .expect("embed primary + 1 shadow at 1080p");
    eprintln!("[diag-1080p Stage 3] stego packet {} bytes", stego.len());

    let recovered_primary =
        av1_stego_extract(&stego, primary_pass).expect("primary extract (1080p + 1 shadow)");
    assert_eq!(recovered_primary.as_slice(), primary_msg);

    let recovered_shadow =
        av1_stego_extract_shadow(&stego, shadow_pass).expect("shadow extract (1080p + 1 shadow)");
    assert_eq!(recovered_shadow.text, shadow_msg);
}

/// Stage 4: primary + 2 shadows (F.7's failing scenario, reduced).
#[test]
fn diag_1080p_primary_plus_two_shadows() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260605");
    }
    let yuv = extract_yuv(1.0);
    let (natural_packet, recording) = encode_natural(&yuv);

    let primary_msg = b"primary 1080p+2shadows";
    let primary_pass = "primary-1080p-2s";

    let shadow_msg_a = "shadow A 1080p";
    let shadow_msg_b = "shadow B 1080p";
    let payload_a = payload::encode_payload(shadow_msg_a, &[]).unwrap();
    let payload_b = payload::encode_payload(shadow_msg_b, &[]).unwrap();

    let structural_key = crypto::derive_structural_key(primary_pass).unwrap();
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();
    let (ct, nonce, salt) = crypto::encrypt(primary_msg, primary_pass).unwrap();
    let primary_frame = frame::build_frame(primary_msg.len(), &salt, &nonce, &ct);
    let primary_bits = frame::bytes_to_bits(&primary_frame);

    let stego = av1_stego_embed_payload_bits_with_shadows_parity(
        natural_packet,
        recording,
        &primary_bits,
        &hhat_seed,
        &[("pass-a", &payload_a), ("pass-b", &payload_b)],
        32,
    )
    .expect("embed primary + 2 shadows at 1080p");
    eprintln!("[diag-1080p Stage 4] stego packet {} bytes", stego.len());

    let recovered_primary =
        av1_stego_extract(&stego, primary_pass).expect("primary extract (1080p + 2 shadows)");
    assert_eq!(recovered_primary.as_slice(), primary_msg);

    let recovered_a =
        av1_stego_extract_shadow(&stego, "pass-a").expect("shadow A (1080p + 2 shadows)");
    assert_eq!(recovered_a.text, shadow_msg_a);
    let recovered_b =
        av1_stego_extract_shadow(&stego, "pass-b").expect("shadow B (1080p + 2 shadows)");
    assert_eq!(recovered_b.text, shadow_msg_b);
}
