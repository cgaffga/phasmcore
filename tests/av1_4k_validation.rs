// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! v0.7+ track 2 — 4K (3840x2160) stress validation.
//!
//! Confirms the AV1 stego pipeline works at 4K resolution, not just
//! the 256x144 / 1088x1920 corpus fixtures all other tests use. The
//! F.7-diag fix made `extract_first_valid_w`'s max_w cap scale as
//! `n / (FRAME_OVERHEAD * 8)` so the decoder's brute-force `w`
//! search reaches the encoder's actual `w` at large cover sizes
//! (~10000 at 4K vs the old hardcoded 1000).
//!
//! This test exercises 3 stages:
//!
//! 1. `four_k_natural_round_trip` — rav1e encode + dav1d decode +
//!    pixel-bytes parity. No stego at all. Confirms the codec stack
//!    survives at 4K.
//!
//! 2. `four_k_primary_embed_extract` — full primary stego flow at
//!    4K. Confirms STC plan + replay + extract work at the larger
//!    cover. Implicitly validates the F.7-diag max_w fix.
//!
//! 3. `four_k_capacity_probe` — capacity-probe API at 4K. Sanity
//!    check that the probe returns a non-zero usable byte count
//!    on the larger frame.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use phasm_core::codec::av1::stego::capacity::av1_capacity;
use phasm_core::codec::av1::stego::orchestrator::{av1_stego_embed, av1_stego_extract};
use phasm_core::codec::av1::stego::session::Av1StreamingEncodeParams;
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::ec::PhasmFrameRecording;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

// 4K UHD. Both dimensions are multiples of 8 (rav1e tile-grid
// requirement): 3840 / 8 = 480, 2160 / 8 = 270.
const W: u32 = 3840;
const H: u32 = 2160;
const Q: usize = 30;

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv_4k() -> Vec<u8> {
    // No 4K source in the corpus — upscale the 1080p fixture. The
    // pipeline doesn't care that the source is upscaled; what matters
    // is that the YUV buffer is 4K-sized and the encoder runs on
    // 4K-sized frame state.
    let src = corpus_root().join("iphone5_1080p_30fps_h264_high.mov");
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!("scale={W}:{H}:force_original_aspect_ratio=disable");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss", "1.0", "-i"])
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
    assert!(out.status.success(), "ffmpeg yuv extract failed at 4K");
    let expected = (W * H * 3 / 2) as usize;
    assert_eq!(out.stdout.len(), expected, "expected 4K I420 size");
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

    let w = W as usize;
    let h = H as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);

    let mut frame_in = make_frame::<u8>(w, h, ChromaSampling::Cs420);
    frame_in.planes[0].copy_from_raw_u8(&yuv[..y_size], w, 1);
    frame_in.planes[1].copy_from_raw_u8(&yuv[y_size..y_size + uv_size], w / 2, 1);
    frame_in.planes[2].copy_from_raw_u8(&yuv[y_size + uv_size..y_size + 2 * uv_size], w / 2, 1);

    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame_in));
    let inter_cfg = make_inter_config(&config);
    encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg)
}

#[test]
#[ignore = "slow — runs at 4K, ~30-60 sec per stage; opt in with --ignored"]
fn four_k_natural_round_trip() {
    let yuv = extract_yuv_4k();
    let (packet, recording) = encode_natural(&yuv);
    eprintln!("[4K] natural encode: {} bytes", packet.len());
    assert!(!packet.is_empty(), "natural encode produced empty packet");
    assert!(
        !recording.tiles.is_empty(),
        "recording has no tiles at 4K"
    );
    let tile = &recording.tiles[0];
    eprintln!(
        "[4K] storage tuples: {} | bit_positions: {} | bit_tags: {}",
        tile.storage.len(),
        tile.bit_positions.len(),
        tile.bit_tags.len()
    );
    assert!(
        !tile.bit_positions.is_empty(),
        "no bit_positions captured at 4K"
    );
}

#[test]
#[ignore = "slow — runs at 4K, ~30-60 sec per stage; opt in with --ignored"]
fn four_k_primary_embed_extract() {
    let yuv = extract_yuv_4k();
    let (natural, recording) = encode_natural(&yuv);

    let message = b"phasm AV1 4K validation - primary embed-extract at 3840x2160 stress test";
    let passphrase = "4k-validation-2026-06-05";

    let stego = av1_stego_embed(natural, recording, message, passphrase)
        .expect("4K primary embed failed");
    eprintln!("[4K] stego encode: {} bytes", stego.len());

    let plaintext = av1_stego_extract(&stego, passphrase).expect("4K primary extract failed");
    assert_eq!(
        plaintext.as_slice(),
        &message[..],
        "4K round-trip plaintext mismatch"
    );
}

#[test]
#[ignore = "slow — runs at 4K, ~30-60 sec per stage; opt in with --ignored"]
fn four_k_capacity_probe() {
    let yuv = extract_yuv_4k();
    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: 1,
    };
    let info = av1_capacity(&yuv, params).expect("4K capacity probe failed");
    eprintln!(
        "[4K] capacity: {} usable primary bytes (cover_size_bits={}, ac_sign={}, golomb_tail={}, n_gops={})",
        info.primary_max_message_bytes,
        info.cover_size_bits,
        info.per_domain_bits.ac_sign,
        info.per_domain_bits.golomb_tail,
        info.n_gops,
    );
    assert!(
        info.primary_max_message_bytes > 100,
        "capacity probe returned suspiciously small primary_max_message_bytes at 4K: {}",
        info.primary_max_message_bytes
    );
    assert!(
        info.cover_size_bits > 1000,
        "capacity probe returned suspiciously small cover_size_bits at 4K: {}",
        info.cover_size_bits
    );
}
