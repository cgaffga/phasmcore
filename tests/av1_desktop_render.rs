// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! One-shot visual inspection: render natural + stego AV1 carplane
//! frames to ~/Desktop as MP4 for eyeball comparison.
//!
//! Not part of the regular test suite (#[ignore]'d). Run on demand:
//!
//! ```bash
//! cargo test --features av1-encoder,av1-backend \
//!     --test av1_desktop_render --release -- --ignored --nocapture
//! ```
//!
//! Outputs to ~/Desktop:
//!   - phasm_av1_carplane_natural.mp4  (clean encode, no stego)
//!   - phasm_av1_carplane_stego.mp4    (with hidden encrypted message)

#![cfg(all(feature = "av1-encoder", feature = "av1-backend"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use phasm_core::codec::av1::stego::orchestrator::av1_stego_embed;
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

#[test]
#[ignore]
fn render_carplane_stego_to_desktop() {
    // Source is 1080×1920 portrait. 720×1280 = exact 9:16, fills
    // QuickTime player nicely. Note: encode_frame_with_phasm_tee
    // has a known v0.3 partial-encode bug that leaves a thin gray
    // strip on the rightmost ~1/6 of the frame; at 720×1280 it's
    // small enough to live with for the visual demo.
    let width: u32 = 720;
    let height: u32 = 1280;
    let seek_s: f32 = 2.0;
    let quantizer: usize = 30;
    let source = "Artlist_CarPlane.mp4";
    // Short payload for v0.3 visual demo. The previous 153-byte
    // message at this resolution caused ~12% AC sign flip rate and
    // visible blocky artifacts under uniform STC costs — that's the
    // cascade leakage cost-model.md / cascade-safety.md target with
    // J-UNIWARD in v0.5+. Short payload → ~0.2% flip rate → visually
    // indistinguishable cover/stego pair, which is what we want for
    // the v0.3 "look, here's the stego output" demo.
    let message: &[u8] = b"hi";
    let passphrase = "desktop-demo-pass";

    // 1. Extract YUV4:2:0 at chosen dims from carplane.
    let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors/video/h264/real-world/source")
        .join(source);
    assert!(src_path.exists(), "missing source: {}", src_path.display());

    let vf = format!("scale={}:{}", width, height);
    let yuv_out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src_path)
        .args(["-frames:v", "1", "-vf", &vf, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output()
        .expect("ffmpeg yuv extract");
    assert!(yuv_out.status.success(), "yuv extract failed: {}", String::from_utf8_lossy(&yuv_out.stderr));
    let yuv = yuv_out.stdout;
    let expected = (width * height * 3 / 2) as usize;
    assert_eq!(yuv.len(), expected, "yuv size mismatch");

    // 2. Natural encode via encode_frame_with_phasm_tee.
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
    let mut fi =
        FrameInvariants::<u8>::new_key_frame(config.clone(), Arc::new(sequence), 0, Box::new([]));
    fi.enable_segmentation = false;
    let mut frame = make_frame::<u8>(width as usize, height as usize, ChromaSampling::Cs420);

    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    // CRITICAL: use `Plane::copy_from_raw_u8` — see feedback memory.
    frame.planes[0].copy_from_raw_u8(&yuv[..y_size], w, 1);
    frame.planes[1].copy_from_raw_u8(&yuv[y_size..y_size + uv_size], w / 2, 1);
    frame.planes[2].copy_from_raw_u8(
        &yuv[y_size + uv_size..y_size + 2 * uv_size],
        w / 2,
        1,
    );

    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame));
    let inter_cfg = make_inter_config(&config);
    let (natural_packet, recording) = encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);
    eprintln!(
        "[av1-desktop] natural packet: {} bytes at {}×{} q{}",
        natural_packet.len(),
        width,
        height,
        quantizer
    );

    // 3. Embed stego.
    let stego_packet = av1_stego_embed(natural_packet.clone(), recording, message, passphrase)
        .expect("av1_stego_embed");
    let byte_diff = natural_packet
        .iter()
        .zip(stego_packet.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "[av1-desktop] stego packet: {} bytes (differs from natural in {} bytes)",
        stego_packet.len(),
        byte_diff
    );

    // 4. Wrap each packet in IVF (32-byte header + per-frame header)
    // and mux to MP4 with ffmpeg. ffmpeg refuses to ingest raw OBU
    // without a recognized container, but IVF is trivial to write
    // by hand and ffmpeg understands it natively.
    let desktop = PathBuf::from(env!("HOME")).join("Desktop");
    let tmp_dir = std::env::temp_dir();
    for (packet, label) in [(&natural_packet, "natural"), (&stego_packet, "stego")] {
        let ivf_path = tmp_dir.join(format!("phasm_av1_carplane_{}.ivf", label));
        let mp4_path = desktop.join(format!("phasm_av1_carplane_{}.mp4", label));

        let ivf = build_ivf_single_frame(packet, width as u16, height as u16);
        std::fs::write(&ivf_path, &ivf).expect("write ivf");

        let mux = Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(&ivf_path)
            .args(["-c", "copy"])
            .arg(&mp4_path)
            .status()
            .expect("ffmpeg mux launch");
        assert!(mux.success(), "ffmpeg mux failed for {}", label);

        let _ = std::fs::remove_file(&ivf_path);
        eprintln!("[av1-desktop] wrote {}", mp4_path.display());
    }

    eprintln!(
        "[av1-desktop] DONE. Open in QuickTime or VLC. Files:\n  \
         ~/Desktop/phasm_av1_carplane_natural.mp4 (clean encode)\n  \
         ~/Desktop/phasm_av1_carplane_stego.mp4   (with hidden encrypted message)"
    );
}

/// Build a minimal IVF container around a single AV1 frame's OBU
/// bytes. IVF format ref:
///   - 32-byte file header (DKIF magic + AV01 fourcc + dims + fps)
///   - 12-byte per-frame header (size + pts)
///   - frame OBU bytes
fn build_ivf_single_frame(obus: &[u8], width: u16, height: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + 12 + obus.len());
    // File header.
    out.extend_from_slice(b"DKIF");           // 0..4   magic
    out.extend_from_slice(&0u16.to_le_bytes()); // 4..6   version
    out.extend_from_slice(&32u16.to_le_bytes()); // 6..8   header size
    out.extend_from_slice(b"AV01");           // 8..12  fourcc
    out.extend_from_slice(&width.to_le_bytes()); // 12..14 width
    out.extend_from_slice(&height.to_le_bytes()); // 14..16 height
    out.extend_from_slice(&30u32.to_le_bytes()); // 16..20 fps numerator
    out.extend_from_slice(&1u32.to_le_bytes());  // 20..24 fps denominator
    out.extend_from_slice(&1u32.to_le_bytes());  // 24..28 frame count
    out.extend_from_slice(&0u32.to_le_bytes());  // 28..32 reserved
    // Per-frame header.
    out.extend_from_slice(&(obus.len() as u32).to_le_bytes()); // 0..4 size
    out.extend_from_slice(&0u64.to_le_bytes());                // 4..12 pts
    // OBU payload.
    out.extend_from_slice(obus);
    out
}
