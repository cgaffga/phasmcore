// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Shared corpus-fixture scaffolding for the OH264 video visual gates.
//!
//! Extracted from `openh264_visual_psnr_regression.rs` when
//! `oh264_quality_vs_fill.rs` was added (2026-10-20). Both gates need the
//! identical loader chain — ffprobe the source dims, 16-align them, scale to a
//! cached raw I420 clip, then ffmpeg-decode a stego Annex-B back to I420 and
//! count concealment events — and a *second copy* of that chain is exactly how
//! two measurements of "the same thing" start disagreeing. The metric kernels
//! already live in one place (`common::visual_metrics`) for the same reason;
//! this is the loader half.
//!
//! Scratch policy: every intermediate goes through `common::scratch_dir()`
//! (i.e. `$PHASM_TMP`), never `std::env::temp_dir()`. The decoded-YUV caches
//! here are hundreds of MB per fixture/resolution; see `common::scratch_dir`
//! for the history.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// OH264 serialization guard.
///
/// The OpenH264 FFI is driven through process-global state, so the corpus
/// gates take this before touching an encode session. Shared here so every
/// gate in a given test binary contends on one mutex.
///
/// Callers should recover from poisoning (`unwrap_or_else(|e| e.into_inner())`)
/// — one failing fixture is an independent measurement and must not
/// cascade-fail the rest of the corpus.
static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();

pub fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

/// Root of the real-world H.264 source corpus (gitignored, large).
pub fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

/// One corpus clip at one working resolution / frame count / QP.
pub struct VideoFixture {
    pub name: &'static str,
    pub source_mp4: &'static str,
    pub long_side_cap: Option<u32>,
    pub n_frames: u32,
    pub qp: i32,
}

/// Probe source W×H via ffprobe and return aspect-preserving 16-aligned
/// dims (H.264 macroblock alignment). Capped at `long_side_cap` if set.
pub fn probe_aligned_dims(spec: &VideoFixture) -> (u32, u32) {
    let src = corpus_root().join(spec.source_mp4);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(&src)
        .output()
        .expect("ffprobe");
    let s = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<&str> = s.trim().split(',').collect();
    let src_w: u32 = parts[0].parse().expect("source width");
    let src_h: u32 = parts[1].parse().expect("source height");

    let (mut tw, mut th) = (src_w, src_h);
    if let Some(cap) = spec.long_side_cap {
        let long = src_w.max(src_h) as f64;
        if (long as u32) > cap {
            let scale = cap as f64 / long;
            tw = (src_w as f64 * scale) as u32;
            th = (src_h as f64 * scale) as u32;
        }
    }
    let aw = (tw / 16) * 16;
    let ah = (th / 16) * 16;
    (aw.max(16), ah.max(16))
}

/// Scale the corpus clip to a flat I420 buffer of `spec.n_frames` frames at
/// `w`×`h`, caching the result under `$PHASM_TMP`.
///
/// The cache key (and filename prefix) is deliberately unchanged from the
/// original `openh264_visual_psnr_regression` implementation so existing
/// caches on a developer machine keep being reused.
pub fn ensure_yuv(spec: &VideoFixture, w: u32, h: u32) -> Vec<u8> {
    let yuv_path = super::scratch_path(&format!(
        "phasm_oh264_psnr_{}_{}x{}_f{}.yuv",
        spec.name, w, h, spec.n_frames
    ));
    let frame_size = (w * h * 3 / 2) as usize;
    let need = frame_size * (spec.n_frames as usize);
    if let Ok(data) = std::fs::read(&yuv_path) {
        if data.len() >= need {
            return data;
        }
    }
    let src = corpus_root().join(spec.source_mp4);
    let vf = format!("scale={w}:{h}");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args(["-frames:v", &spec.n_frames.to_string()])
        .args(["-an", "-vf", &vf])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("ffmpeg launch");
    assert!(status.success(), "ffmpeg scale failed for {}", spec.source_mp4);
    std::fs::read(&yuv_path).expect("read yuv")
}

/// An independently-decoded clip plus the decoder's own complaint count.
pub struct DecodedClip {
    /// Flat I420, `n_frames × (w*h*3/2)` bytes.
    pub yuv: Vec<u8>,
    /// Number of ffmpeg log lines containing "concealing" — any non-zero
    /// value means the bitstream itself is broken, independent of PSNR.
    pub n_conceal: usize,
    /// Where the ffmpeg stderr log was written (for failure messages).
    pub log_path: String,
    /// Where the raw decoded I420 was written. At 1080p×30 this is ~93 MB per
    /// call, so a sweep that decodes many variants should
    /// [`discard_decoded_yuv`] each one once it has been measured — the bytes
    /// are already in `yuv`, and the Annex-B stays on disk for eyeballing.
    pub yuv_path: String,
    /// Where the stego Annex-B was written — small, and the useful artifact to
    /// keep when a measurement looks wrong (play it, or re-decode by hand).
    pub annex_b_path: String,
}

impl DecodedClip {
    /// Delete the big raw-I420 intermediate, keeping the Annex-B and the log.
    /// Best-effort: a failure to unlink is not a test failure.
    pub fn discard_decoded_yuv(&self) {
        let _ = std::fs::remove_file(&self.yuv_path);
    }
}

/// Decode stego Annex-B with an independent decoder (ffmpeg subprocess) back
/// to flat I420.
///
/// `tag` names the intermediate files; pass something unique per
/// (fixture, variant) so parallel-ish callers don't collide.
pub fn ffmpeg_decode_annex_b(annex_b: &[u8], tag: &str) -> DecodedClip {
    let h264_path = super::scratch_path(&format!("phasm_oh264_{tag}.h264"));
    let dec_path = super::scratch_path(&format!("phasm_oh264_{tag}.dec.yuv"));
    let log_path = super::scratch_path(&format!("phasm_oh264_{tag}.ffmpeg.log"));
    std::fs::write(&h264_path, annex_b).expect("write annex-b");
    let log_file = std::fs::File::create(&log_path).expect("create log");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "info", "-framerate", "30", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .stderr(log_file)
        .status()
        .expect("ffmpeg decode");
    assert!(status.success(), "ffmpeg decode failed for {tag}");

    let log = std::fs::read_to_string(&log_path).expect("read ffmpeg log");
    let n_conceal = log.lines().filter(|l| l.contains("concealing")).count();
    let yuv = std::fs::read(&dec_path).expect("read decoded");
    DecodedClip {
        yuv,
        n_conceal,
        log_path,
        yuv_path: dec_path,
        annex_b_path: h264_path,
    }
}
