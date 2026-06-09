// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! YUV-level entry points for mobile bridges.
//!
//! The existing AV1 stego API takes a `PhasmFrameRecording` (the
//! result of running rav1e's `encode_frame_with_phasm_tee`), but
//! every mobile bridge needs a thinner contract: "here is a YUV
//! I420 frame, here is a message + passphrase — produce stego AV1
//! bytes". This module is the codec-layer helper that hides the
//! per-frame `FrameInvariants` / `FrameState` / `EncoderConfig`
//! ceremony.
//!
//! Three top-level functions mirror the H.264 yuv-string FFI shape
//! (see `core/src/codec/h264/stego/encode_pixels.rs` lines 3118+):
//!
//! - `av1_stego_encode_yuv` — single-frame primary
//! - `av1_stego_encode_yuv_with_shadows` — primary + N shadows
//! - `av1_stego_capacity_yuv` — capacity probe over the same YUV
//!
//! All three are single-frame (legacy `av1_stego_embed` semantics)
//! suitable for image-mode AV1 stego. Multi-frame video uses
//! `Av1StreamingEncodeSession` directly (the session API already
//! takes YUV per `push_frame`).
//!
//! ## Scope
//!
//! - Single keyframe, single tile (matches the rest of image-mode AV1).
//! - I420 / 8-bit YUV only (sufficient for mobile photo capture).
//! - Quantizer is a parameter so the bridge can vary quality.
//! - Decode side is symmetric — `av1_stego_decode_yuv` returns the
//!   primary message, `av1_stego_decode_yuv_shadow` returns a shadow
//!   payload.

use std::sync::Arc;

use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::ec::PhasmFrameRecording;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

use super::capacity::Av1CapacityInfo;
use super::orchestrator::{
    av1_stego_embed, av1_stego_embed_with_shadows, av1_stego_extract, av1_stego_extract_shadow,
    Av1StegoError,
};

/// I420 YUV frame size in bytes for given dimensions.
fn yuv_size(width: u32, height: u32) -> usize {
    let y = (width as usize) * (height as usize);
    let uv = ((width as usize) / 2) * ((height as usize) / 2);
    y + 2 * uv
}

/// Natural-encode a single keyframe from tight I420 YUV bytes.
/// Internal helper — also used by `av1_stego_capacity_yuv` (probe)
/// and the embed entry points.
fn encode_natural_keyframe(
    yuv_i420: &[u8],
    width: u32,
    height: u32,
    quantizer: usize,
) -> Result<(Vec<u8>, PhasmFrameRecording<u8>), Av1StegoError> {
    let expected = yuv_size(width, height);
    if yuv_i420.len() != expected {
        return Err(Av1StegoError::InvalidPacket(format!(
            "yuv length {} != expected {} for {}×{}",
            yuv_i420.len(),
            expected,
            width,
            height,
        )));
    }
    let config = Arc::new(EncoderConfig {
        width: width as usize,
        height: height as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer,
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

    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    let mut frame_in = make_frame::<u8>(w, h, ChromaSampling::Cs420);
    frame_in.planes[0].copy_from_raw_u8(&yuv_i420[..y_size], w, 1);
    frame_in.planes[1].copy_from_raw_u8(&yuv_i420[y_size..y_size + uv_size], w / 2, 1);
    frame_in.planes[2].copy_from_raw_u8(
        &yuv_i420[y_size + uv_size..y_size + 2 * uv_size],
        w / 2,
        1,
    );

    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame_in));
    let inter_cfg = make_inter_config(&config);
    Ok(encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg))
}

/// Encode an I420 YUV frame as stego AV1 bytes carrying `message`.
///
/// `quantizer` is the rav1e qindex (0-255). Typical values: 30 for
/// high-quality stego, 50 for smaller files. iOS / Android bridges
/// pick this based on the device's storage / share-target target.
pub fn av1_stego_encode_yuv(
    yuv_i420: &[u8],
    width: u32,
    height: u32,
    quantizer: usize,
    message: &[u8],
    passphrase: &str,
) -> Result<Vec<u8>, Av1StegoError> {
    let (natural_packet, recording) =
        encode_natural_keyframe(yuv_i420, width, height, quantizer)?;
    av1_stego_embed(natural_packet, recording, message, passphrase)
}

/// Encode an I420 YUV frame as stego AV1 bytes carrying `message`
/// as the primary plus N shadows. Each shadow is `(passphrase,
/// payload_bytes)` — payload_bytes is the output of
/// `crate::stego::payload::encode_payload(text, files)` if the shadow
/// carries text + file attachments, or just raw bytes.
pub fn av1_stego_encode_yuv_with_shadows(
    yuv_i420: &[u8],
    width: u32,
    height: u32,
    quantizer: usize,
    message: &[u8],
    passphrase: &str,
    shadows: &[(&str, &[u8])],
) -> Result<Vec<u8>, Av1StegoError> {
    let (natural_packet, recording) =
        encode_natural_keyframe(yuv_i420, width, height, quantizer)?;
    av1_stego_embed_with_shadows(natural_packet, recording, message, passphrase, shadows)
}

/// Decode the primary message from stego AV1 bytes. Returns the
/// plaintext bytes on success. Forwarded to `av1_stego_extract`.
pub fn av1_stego_decode_yuv(
    stego_av1_bytes: &[u8],
    passphrase: &str,
) -> Result<Vec<u8>, Av1StegoError> {
    av1_stego_extract(stego_av1_bytes, passphrase)
}

/// Decode a shadow message from stego AV1 bytes. Returns the
/// `PayloadData` (text + files) on success. Wrong passphrase yields
/// `StegoError::FrameCorrupted`. Forwarded to `av1_stego_extract_shadow`.
pub fn av1_stego_decode_yuv_shadow(
    stego_av1_bytes: &[u8],
    shadow_passphrase: &str,
) -> Result<crate::stego::payload::PayloadData, crate::stego::error::StegoError> {
    av1_stego_extract_shadow(stego_av1_bytes, shadow_passphrase)
}

/// Capacity probe over an I420 YUV frame.
///
/// Returns the full `Av1CapacityInfo` struct (cover bits, primary
/// max bytes, per-domain breakdown, n_gops=1, shadow_n=1 cap).
/// Wrapper around `crate::codec::av1::stego::capacity::av1_capacity`
/// adapted to single-frame YUV input. UI shows
/// `primary_max_message_bytes` as the "remaining bytes" counter.
pub fn av1_stego_capacity_yuv(
    yuv_i420: &[u8],
    width: u32,
    height: u32,
    quantizer: usize,
) -> Result<Av1CapacityInfo, Av1StegoError> {
    use super::session::Av1StreamingEncodeParams;
    let params = Av1StreamingEncodeParams {
        width,
        height,
        quantizer,
        gop_size: 1,
        total_frames_hint: 1,
    };
    super::capacity::av1_capacity(yuv_i420, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_yuv(width: u32, height: u32) -> Vec<u8> {
        let mut yuv = vec![0u8; yuv_size(width, height)];
        // Deterministic gradient so cover bits are non-trivial.
        for i in 0..yuv.len() {
            yuv[i] = ((i * 7) ^ (i >> 3)) as u8;
        }
        yuv
    }

    #[test]
    fn yuv_round_trip_primary_only() {
        let _seed = crate::stego::crypto::DeterministicSeedGuard::set("20260605");
        let yuv = synthetic_yuv(144, 256);
        let stego = av1_stego_encode_yuv(&yuv, 144, 256, 30, b"BR test", "br-pass")
            .expect("encode_yuv");
        let recovered = av1_stego_decode_yuv(&stego, "br-pass").expect("decode_yuv");
        assert_eq!(recovered.as_slice(), b"BR test");
    }

    #[test]
    fn yuv_round_trip_with_shadow() {
        let _seed = crate::stego::crypto::DeterministicSeedGuard::set("20260605");
        let yuv = synthetic_yuv(144, 256);
        let shadow_payload = crate::stego::payload::encode_payload("shadow body", &[]).unwrap();
        let stego = av1_stego_encode_yuv_with_shadows(
            &yuv,
            144,
            256,
            30,
            b"primary body",
            "primary-br",
            &[("shadow-br", &shadow_payload)],
        )
        .expect("encode_yuv_with_shadows");

        let primary = av1_stego_decode_yuv(&stego, "primary-br").expect("primary");
        assert_eq!(primary.as_slice(), b"primary body");

        let shadow = av1_stego_decode_yuv_shadow(&stego, "shadow-br").expect("shadow");
        assert_eq!(shadow.text, "shadow body");
    }

    #[test]
    fn yuv_capacity_returns_sensible_bounds() {
        let yuv = synthetic_yuv(144, 256);
        let cap = av1_stego_capacity_yuv(&yuv, 144, 256, 30).expect("capacity");
        assert!(cap.primary_max_message_bytes > 0);
        assert_eq!(cap.n_gops, 1);
        // Per-domain breakdown sums to total.
        assert_eq!(
            cap.per_domain_bits.ac_sign + cap.per_domain_bits.golomb_tail,
            cap.cover_size_bits
        );
    }

    #[test]
    fn yuv_encode_rejects_wrong_yuv_size() {
        let undersized = vec![0u8; 100];
        let err =
            av1_stego_encode_yuv(&undersized, 144, 256, 30, b"x", "p").expect_err("size error");
        let msg = format!("{err:?}");
        assert!(msg.contains("yuv length"));
    }
}
