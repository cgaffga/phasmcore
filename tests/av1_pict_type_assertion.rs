// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! TG-3 — Container-layer guard.
//!
//! Asserts the MP4 muxed from a phasm-av1 stego encode:
//!   - decode-order frame 0 is a KEYFRAME (pict_type == I)
//!   - PTS[0] == 0 (no offset injected by an edit list)
//!   - no edit list (`edts/elst`) box silently shifting decode order
//!
//! ## Why this exists
//!
//! H.264's encoder-quality audit on 2026-05-05 found a phantom 7.2 dB
//! Y-PSNR "B-frame encoder cliff" that vanished when a custom
//! `edts/elst` box (media_time=+pre_roll) was removed: ffmpeg's PSNR
//! pairing had been comparing source[k] against stego[k+1], not the
//! encoder making poor mode decisions. The "encoder is broken"
//! diagnosis ran for weeks before the container layer was suspected.
//! See `docs/design/video/h264/encoder-quality-perf-gap-2026-05-04.md`
//! lines 380-388 + `memory/h264_elst_bug_2026_05_05.md`.
//!
//! phasm-av1 today only emits keyframes (single-frame encode) and uses
//! ffmpeg's default MP4 muxer (no custom elst), so this test is a
//! cheap structural gate that locks today's clean behavior and
//! catches the failure class when multi-GOP scale-up (`phase-c-
//! streaming-session-v6.md`) or a custom muxer lands in Phase D.
//!
//! See [`phase-c-test-gates.md`](../../docs/design/video/av1/phase-c-test-gates.md)
//! § 3 for the full design rationale.

#![cfg(all(feature = "av1-encoder", feature = "av1-backend"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use phasm_rav1e::color::ChromaSampling;
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
        // Source is 1080×1920 PORTRAIT — encode at portrait dims.
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
    let (packet, _recording) = encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);
    packet
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

fn mux_ivf_to_mp4(av1_bytes: &[u8], spec: &Fixture) -> PathBuf {
    let ivf_path = std::env::temp_dir().join(format!(
        "tg3_{}_{}.ivf",
        spec.name,
        std::process::id()
    ));
    let mp4_path = std::env::temp_dir().join(format!(
        "tg3_{}_{}.mp4",
        spec.name,
        std::process::id()
    ));
    let ivf = build_ivf_single_frame(av1_bytes, spec.width as u16, spec.height as u16);
    std::fs::write(&ivf_path, &ivf).expect("write ivf");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&ivf_path)
        .args(["-c:v", "copy"])
        .arg(&mp4_path)
        .output()
        .expect("ffmpeg mux launch");
    let _ = std::fs::remove_file(&ivf_path);
    assert!(
        out.status.success(),
        "ffmpeg mux failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    mp4_path
}

fn ffprobe_one(mp4: &Path, entries: &str, stream: bool) -> String {
    let prefix = if stream { "stream=" } else { "frame=" };
    let mut cmd = Command::new("ffprobe");
    cmd.args(["-loglevel", "error", "-select_streams", "v:0", "-show_entries"]);
    cmd.arg(format!("{}{}", prefix, entries));
    if !stream {
        cmd.args(["-read_intervals", "%+#1"]);
    }
    cmd.args(["-of", "csv=p=0"]).arg(mp4);
    let out = cmd.output().expect("ffprobe launch");
    assert!(
        out.status.success(),
        "ffprobe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn first_frame_pict_type(mp4: &Path) -> String {
    ffprobe_one(mp4, "pict_type", false)
}

fn first_packet_pts(mp4: &Path) -> String {
    // packet entries need a different prefix; do it inline.
    let out = Command::new("ffprobe")
        .args([
            "-loglevel", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts",
            "-read_intervals", "%+#1",
            "-of", "csv=p=0",
        ])
        .arg(mp4)
        .output()
        .expect("ffprobe launch");
    assert!(out.status.success(), "ffprobe packet pts failed");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn stream_start_time(mp4: &Path) -> String {
    ffprobe_one(mp4, "start_time", true)
}

fn assert_clean_container(spec: &Fixture) {
    let yuv = extract_yuv420_frame(spec);
    let av1 = encode_natural(&yuv, spec);
    let mp4 = mux_ivf_to_mp4(&av1, spec);

    let pict_type = first_frame_pict_type(&mp4);
    let pts = first_packet_pts(&mp4);
    let start_time = stream_start_time(&mp4);

    let _ = std::fs::remove_file(&mp4);

    // ---- 1. Decode-order frame 0 MUST be a keyframe.
    assert_eq!(
        pict_type, "I",
        "[{}] decode-order frame 0 pict_type must be I (keyframe), got {:?}. \
         A non-I first frame means the container lists a non-keyframe in decode position 0 — \
         either the encoder produced a non-key first frame, or an edit list reorders decode. \
         See `docs/design/video/h264/encoder-quality-perf-gap-2026-05-04.md` for the H.264 \
         lesson where a similar misordering looked like a 7.2 dB B-frame encoder cliff.",
        spec.name, pict_type
    );

    // ---- 2. First packet PTS MUST be 0.
    // A non-zero PTS[0] means the container shifted the time origin —
    // any PSNR-paired comparison (src[k] vs stego[k+1]) silently
    // measures a one-frame offset instead of encoder quality.
    let pts_ok = pts == "0" || pts == "N/A";
    assert!(
        pts_ok,
        "[{}] first packet PTS must be 0 (or N/A on stride-less streams), got {:?}. \
         Non-zero PTS[0] indicates an edit list (`edts/elst`) shifting decode order. \
         This silently breaks PSNR-paired comparison and looks like an encoder cliff.",
        spec.name, pts
    );

    // ---- 3. Stream start_time MUST be 0 (no `edts/elst`).
    // ffprobe reports start_time = 0.000000 for a stream with no edit
    // list; if an elst injects a positive media_time, start_time
    // reflects the offset.
    let elst_clean = start_time == "0.000000" || start_time.is_empty() || start_time == "N/A";
    assert!(
        elst_clean,
        "[{}] MP4 stream has start_time={:?} (expected 0.000000 / N/A). \
         Non-zero start_time indicates an `edts/elst` box. phasm-av1 uses ffmpeg's default \
         muxer with `-c:v copy` — if this trips, either ffmpeg's defaults regressed for AV1 \
         or a custom muxer with an elst landed. See \
         `memory/h264_elst_bug_2026_05_05.md` for the H.264 incident pattern.",
        spec.name, start_time
    );
}

#[test]
fn iphone_img4138_first_frame_clean_container() {
    assert_clean_container(&FIXTURES[0]);
}

#[test]
fn carplane_first_frame_clean_container() {
    assert_clean_container(&FIXTURES[1]);
}

#[test]
fn iphone5_1080p_first_frame_clean_container() {
    assert_clean_container(&FIXTURES[2]);
}
