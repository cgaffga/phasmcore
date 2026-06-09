// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! TG-4 — Encoder recon-buffer separation invariant.
//!
//! Asserts `av1_stego_embed` does NOT mutate the encoder's
//! `reconstructed_planes` buffer
//! ([`phasm_rav1e::ec::PhasmFrameRecording::reconstructed_planes`]).
//! The recon buffer is the reference path for future inter-frame ME
//! AND the input to J-UNIWARD cost compute — stego flips must stay
//! WIRE-ONLY (at the MSAC bypass-bin emit site), per the V6
//! streaming-session architecture lifted from H.264 #540.
//!
//! ## Why this exists
//!
//! H.264's dump tests compared `enc.recon` thinking it was
//! `visual_recon`, but `enc.recon` was the pre-flip cover state.
//! Multi-week diagnostic confusion ensued before the buffer split
//! was recognised. See
//! `memory/h264_b_encoder_decoder_divergence_2026_05_07.md` +
//! `docs/design/video/av1/phase-c-test-gates.md` § 4.
//!
//! For AV1 v0.5 (single-frame, wire-only override at MSAC emit), the
//! encoder's `fs.rec` is never touched by stego — that property is
//! structural (the override fires during entropy coding, not during
//! reconstruction). This test locks it. Any future Tier 2 work that
//! mutates recon during stego will trip this gate before shipping.
//!
//! ## What it asserts
//!
//! 1. Snapshot `reconstructed_planes` BEFORE running embed.
//! 2. Run `av1_stego_embed` (consumes a `clone` of the recording —
//!    Arc-share, no data copy; both clones point to the same Frame).
//! 3. Snapshot `reconstructed_planes` AFTER embed via the original
//!    Arc reference.
//! 4. Assert: snapshots byte-identical per plane (Y/Cb/Cr).
//! 5. Sanity: extract round-trips the embedded message.
//!
//! See [`phase-c-test-gates.md`](../../docs/design/video/av1/phase-c-test-gates.md)
//! § 4 for the full design rationale.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use phasm_core::codec::av1::stego::orchestrator::{av1_stego_embed, av1_stego_extract};
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::ec::PhasmFrameRecording;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

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

/// Pack the visible region of each plane of the recon `Frame` into
/// contiguous `Vec<u8>` (strip filter-tap padding at xorigin/yorigin).
/// Mirrors `orchestrator::pack_visible_planes` but local to avoid
/// pulling in non-pub helpers.
fn snapshot_recon_planes(rec: &Arc<phasm_rav1e::Frame<u8>>) -> [Vec<u8>; 3] {
    let pack = |plane_idx: usize| -> Vec<u8> {
        let p = &rec.planes[plane_idx];
        let w = p.cfg.width;
        let h = p.cfg.height;
        let stride = p.cfg.stride;
        let start = p.cfg.yorigin * stride + p.cfg.xorigin;
        let mut out = Vec::with_capacity(w * h);
        for row in 0..h {
            let row_start = start + row * stride;
            out.extend_from_slice(&p.data[row_start..row_start + w]);
        }
        out
    };
    [pack(0), pack(1), pack(2)]
}

fn assert_recon_unchanged_by_embed(spec: &Fixture) {
    let yuv = extract_yuv420_frame(spec);
    let (natural_packet, recording) = encode_natural_with_recording(&yuv, spec);

    // Snapshot the recon planes (Y/Cb/Cr) BEFORE running embed.
    let recon_before = snapshot_recon_planes(&recording.reconstructed_planes);

    // Hold a separate Arc on the recon Frame so we can re-snapshot AFTER
    // embed consumes its clone of the recording. PhasmFrameRecording is
    // Clone (cheap — all heavy fields are Arc).
    let recon_arc: Arc<phasm_rav1e::Frame<u8>> = Arc::clone(&recording.reconstructed_planes);

    let message = b"tg-4 recon-separation invariant";
    let passphrase = "tg4-test-passphrase-2026-06-03";

    let stego_packet = av1_stego_embed(
        natural_packet.clone(),
        recording.clone(),
        message,
        passphrase,
    )
    .unwrap_or_else(|e| panic!("[{}] av1_stego_embed failed: {:?}", spec.name, e));

    // Snapshot AFTER embed via the Arc we preserved.
    let recon_after = snapshot_recon_planes(&recon_arc);

    // ---- Core TG-4 invariant: recon byte-identical before vs after embed.
    for (plane_idx, (b, a)) in recon_before.iter().zip(recon_after.iter()).enumerate() {
        assert_eq!(
            b.len(),
            a.len(),
            "[{}] plane {} length changed by embed (before {} bytes, after {} bytes) — \
             severe recon-buffer corruption",
            spec.name,
            plane_idx,
            b.len(),
            a.len()
        );
        if b != a {
            // Find first diverging pixel for a useful failure message.
            let first_diff = b.iter().zip(a.iter()).position(|(x, y)| x != y);
            panic!(
                "[{}] plane {} mutated by av1_stego_embed at byte {:?} — \
                 wire-only invariant VIOLATED. The encoder's recon buffer is the \
                 reference path for inter-frame ME and the input to J-UNIWARD cost; \
                 stego flips MUST stay at the MSAC bypass-bin emit site only. See \
                 `docs/design/video/av1/phase-c-streaming-session-v6.md` § 6 + \
                 `phase-c-test-gates.md` § 4.",
                spec.name, plane_idx, first_diff,
            );
        }
    }

    // ---- Sanity: stego packet round-trips the embedded message.
    let extracted = av1_stego_extract(&stego_packet, passphrase)
        .unwrap_or_else(|e| panic!("[{}] av1_stego_extract failed: {:?}", spec.name, e));
    assert_eq!(
        extracted.as_slice(),
        message,
        "[{}] round-trip extraction did not match embedded message",
        spec.name
    );
}

#[test]
fn iphone_img4138_recon_unchanged_by_embed() {
    assert_recon_unchanged_by_embed(&FIXTURES[0]);
}

#[test]
fn carplane_recon_unchanged_by_embed() {
    assert_recon_unchanged_by_embed(&FIXTURES[1]);
}

#[test]
fn iphone5_1080p_recon_unchanged_by_embed() {
    assert_recon_unchanged_by_embed(&FIXTURES[2]);
}
