// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! W4 — AV1 stego corpus validation on real-world content.
//!
//! Reuses the H.264 real-world source corpus
//! (`core/test-vectors/video/h264/real-world/source/`) — same iPhone /
//! drone / pro-camera MOVs/MP4s, gitignored, see that directory's
//! README for provenance. Each fixture: ffmpeg extracts a single
//! frame as YUV4:2:0, rav1e encodes it as an AV1 key frame via
//! `encode_frame_with_phasm_tee`, the orchestrator embeds an
//! encrypted message, dav1d decodes + extracts, plaintext is asserted
//! to round-trip.
//!
//! v0.3-AV1 scope: single-tile, single key frame, AC_COEFF_SIGN only.
//!
//! Run:
//!     cargo test --features av1-encoder,av1-backend --test av1_corpus_validation

#![cfg(all(feature = "av1-encoder", feature = "av1-backend"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use phasm_core::codec::av1::stego::orchestrator::{av1_stego_embed, av1_stego_extract};
use phasm_core::codec::av1::stego::writer::CoverPositions;
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
    PhasmFrameRecording, PHASM_TAG_AC_COEFF_SIGN,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

/// Per-fixture spec. Same source MOVs as H.264 corpus.
struct Fixture {
    /// Short id used in test name + manifest.
    name: &'static str,
    /// Filename under `core/test-vectors/video/h264/real-world/source/`.
    source: &'static str,
    /// Encode dims (must be multiples of 8). Aspect-preserving downscale
    /// from source is applied by ffmpeg's scale filter.
    width: u32,
    height: u32,
    /// Seek time in seconds. Picks a frame mid-clip so static intro
    /// scenes don't bias the texture content.
    seek_s: f32,
    /// AV1 quantizer (rav1e EncoderConfig::quantizer). 30 ≈ visually
    /// transparent for 8-bit content.
    quantizer: usize,
}

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

/// Extract one frame as planar YUV4:2:0 raw bytes via ffmpeg.
fn extract_yuv420_frame(spec: &Fixture) -> Vec<u8> {
    let src = corpus_root().join(spec.source);
    assert!(
        src.exists(),
        "corpus fixture missing: {}\n\
         Symlink or copy the source MOV/MP4 from a checkout that has it.\n\
         The file is gitignored per project convention.",
        src.display()
    );

    let vf = format!("scale={}:{}:force_original_aspect_ratio=disable", spec.width, spec.height);
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(spec.seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args(["-frames:v", "1", "-vf", &vf, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output()
        .expect("ffmpeg failed to launch");

    assert!(
        out.status.success(),
        "ffmpeg failed on {}: {}",
        spec.source,
        String::from_utf8_lossy(&out.stderr)
    );

    let expected = (spec.width * spec.height * 3 / 2) as usize;
    assert_eq!(
        out.stdout.len(),
        expected,
        "ffmpeg yuv output size mismatch for {} ({}×{})",
        spec.source,
        spec.width,
        spec.height
    );
    out.stdout
}

/// Build a rav1e Frame from planar YUV4:2:0 raw bytes and run the
/// natural (no stego) encode via `encode_frame_with_phasm_tee`.
fn encode_natural(yuv: &[u8], spec: &Fixture) -> (Vec<u8>, PhasmFrameRecording) {
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
    // Disable segmentation: kmeans underflows when bypassing the
    // Context API's spatiotemporal_scores prepass (same fix as
    // W3.10.5 orchestrator test).
    fi.enable_segmentation = false;

    let w = spec.width as usize;
    let h = spec.height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);

    let mut frame = make_frame::<u8>(w, h, ChromaSampling::Cs420);

    for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
        let (src, src_stride) = match plane_idx {
            0 => (&yuv[..y_size], w),
            1 => (&yuv[y_size..y_size + uv_size], w / 2),
            2 => (&yuv[y_size + uv_size..y_size + 2 * uv_size], w / 2),
            _ => unreachable!(),
        };
        let dst_stride = plane.cfg.stride;
        for (dst_row, src_row) in plane.data.chunks_mut(dst_stride).zip(src.chunks(src_stride)) {
            dst_row[..src_row.len()].copy_from_slice(src_row);
        }
    }

    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame));
    let inter_cfg = make_inter_config(&config);
    encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg)
}

struct CorpusMetrics {
    natural_bytes: usize,
    cover_bits_ac_sign: usize,
    cover_bits_total: usize,
    encode_ms: u128,
    embed_ms: u128,
    extract_ms: u128,
}

fn run_corpus_roundtrip(spec: &Fixture, message: &[u8], passphrase: &str) -> CorpusMetrics {
    eprintln!(
        "[av1-corpus] {} ({}×{} q{} from {})",
        spec.name, spec.width, spec.height, spec.quantizer, spec.source
    );

    let yuv = extract_yuv420_frame(spec);

    let t0 = Instant::now();
    let (natural_packet, recording) = encode_natural(&yuv, spec);
    let encode_ms = t0.elapsed().as_millis();
    let natural_bytes = natural_packet.len();
    assert!(!natural_packet.is_empty(), "natural packet empty for {}", spec.name);

    // Count cover bits before embed consumes the recording.
    let tile = &recording.tiles[0];
    let positions = CoverPositions::from_recorder(&tile.bit_positions, &tile.bit_tags);
    let cover_bits_total = positions.len();
    let cover_bits_ac_sign = positions.count_by_tag(PHASM_TAG_AC_COEFF_SIGN);

    let t1 = Instant::now();
    let stego_packet = av1_stego_embed(natural_packet.clone(), recording, message, passphrase)
        .unwrap_or_else(|e| panic!("av1_stego_embed failed for {}: {:?}", spec.name, e));
    let embed_ms = t1.elapsed().as_millis();
    assert_eq!(
        stego_packet.len(),
        natural_bytes,
        "stego packet length must equal natural packet length"
    );

    let t2 = Instant::now();
    let extracted = av1_stego_extract(&stego_packet, passphrase)
        .unwrap_or_else(|e| panic!("av1_stego_extract failed for {}: {:?}", spec.name, e));
    let extract_ms = t2.elapsed().as_millis();
    assert_eq!(
        extracted.as_slice(),
        message,
        "extracted plaintext mismatch for {}",
        spec.name
    );

    let metrics = CorpusMetrics {
        natural_bytes,
        cover_bits_ac_sign,
        cover_bits_total,
        encode_ms,
        embed_ms,
        extract_ms,
    };
    eprintln!(
        "[av1-corpus] {} OK — encode {} ms, embed {} ms, extract {} ms, packet {} bytes, \
         cover bits AC_SIGN {} / total {}",
        spec.name,
        metrics.encode_ms,
        metrics.embed_ms,
        metrics.extract_ms,
        metrics.natural_bytes,
        metrics.cover_bits_ac_sign,
        metrics.cover_bits_total
    );
    metrics
}

/// Mid-clip frame from iPhone-shot footage. Natural noise + real
/// camera grain — the canonical "does it survive real content?" check.
#[test]
fn corpus_av1_iphone_img4138() {
    let spec = Fixture {
        name: "iphone_img4138",
        source: "IMG_4138.MOV",
        width: 256,
        height: 144,
        seek_s: 1.0,
        quantizer: 30,
    };
    run_corpus_roundtrip(&spec, b"hi from av1 stego (iphone)", "corpus-pass-1");
}

/// Mid-clip from Artlist_CarPlane — high-motion, mixed texture
/// (sky + plane + car + foliage). Heavy test for diverse coefficient
/// distributions.
#[test]
fn corpus_av1_carplane() {
    let spec = Fixture {
        name: "carplane",
        source: "Artlist_CarPlane.mp4",
        width: 256,
        height: 144,
        seek_s: 2.0,
        quantizer: 30,
    };
    run_corpus_roundtrip(&spec, b"hi from av1 stego (carplane)", "corpus-pass-2");
}

/// Older iPhone 5 footage — different sensor noise profile and a
/// different generation of camera ISP. Cross-device variance check.
#[test]
fn corpus_av1_iphone5_1080p() {
    let spec = Fixture {
        name: "iphone5_1080p",
        source: "iphone5_1080p_30fps_h264_high.mov",
        width: 256,
        height: 144,
        seek_s: 1.0,
        quantizer: 30,
    };
    run_corpus_roundtrip(&spec, b"hi from av1 stego (iphone5)", "corpus-pass-3");
}
