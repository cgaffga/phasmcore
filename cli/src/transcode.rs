// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 4b: shell-out to ffmpeg to transcode non-Baseline inputs into
//! H.264 Baseline CAVLC MP4, which is the only format the Rust H.264
//! pipeline accepts. Used by the CLI `video-encode` and `video-capacity`
//! subcommands when the input codec is a profile other than Baseline
//! CAVLC (e.g. Main/High-profile CABAC out of iPhone recordings).
//!
//! ffmpeg is assumed to be installed and on `$PATH`. If it isn't, we
//! surface a clear error telling the user how to install it or transcode
//! manually.

use crate::error::CliError;
use phasm_core::StegoError;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Shell command we suggest in user-facing error messages when auto-transcode
/// can't run (ffmpeg missing, transcode failed, etc.). Kept as a module
/// constant so encode/capacity/decode paths all suggest the identical
/// invocation.
pub const TRANSCODE_SUGGESTION: &str =
    "ffmpeg -i input.mp4 -c:v libx264 -profile:v baseline -level 3.1 -crf 18 -c:a copy output.mp4";

/// Heuristic: did a `StegoError::InvalidVideo` from the H.264 pipeline
/// indicate the input needs re-encoding to Baseline CAVLC? Matches the
/// error text produced by `h264_ghost_encode/capacity/decode` when the
/// PPS says CABAC or when the SPS is a non-Baseline profile we don't
/// support yet.
pub fn is_non_baseline_error(err: &StegoError) -> bool {
    if let StegoError::InvalidVideo(msg) = err {
        let lower = msg.to_lowercase();
        lower.contains("cabac") || lower.contains("baseline")
    } else {
        false
    }
}

/// Check that `ffmpeg` is runnable (returns exit 0 on `ffmpeg -version`).
pub fn ensure_ffmpeg_available() -> Result<(), CliError> {
    let status = Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(_) | Err(_) => Err(CliError::InvalidArgs(format!(
            "ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg` on macOS, \
             `apt install ffmpeg` on Debian/Ubuntu) or pre-transcode the input manually:\n  {TRANSCODE_SUGGESTION}"
        ))),
    }
}

/// Transcode `input` to H.264 Baseline CAVLC MP4 at `output`. Uses
/// libx264 with `-profile:v baseline -level 3.1 -crf 18`. Audio is
/// stream-copied (`-c:a copy`) to preserve the original track.
///
/// Returns an error if ffmpeg exits non-zero; captures stderr so the
/// user sees ffmpeg's diagnostic instead of a bare exit code.
pub fn transcode_to_baseline(input: &Path, output: &Path) -> Result<(), CliError> {
    let result = Command::new("ffmpeg")
        .arg("-y") // overwrite output
        .arg("-i")
        .arg(input)
        .args([
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-crf", "18",
            "-c:a", "copy",
        ])
        .arg(output)
        .output()
        .map_err(|e| {
            CliError::InvalidArgs(format!(
                "failed to spawn ffmpeg: {e}\n\nInstall ffmpeg or transcode manually:\n  {TRANSCODE_SUGGESTION}"
            ))
        })?;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        // ffmpeg writes a lot of info to stderr even on success; the last
        // ~25 lines are usually where the actual error lives.
        let tail: String = stderr
            .lines()
            .rev()
            .take(25)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Err(CliError::InvalidArgs(format!(
            "ffmpeg transcode failed (exit {}). Last ffmpeg output:\n\n{tail}",
            result.status.code().unwrap_or(-1),
        )));
    }
    Ok(())
}

/// Build a transcode-output tempfile path next to the system temp dir
/// with a unique name. Caller is responsible for cleaning it up.
pub fn temp_transcode_path(original: &Path) -> PathBuf {
    let stem = original
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("phasm_transcode");
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("{stem}_phasm_baseline_{pid}_{ts}.mp4"))
}

// ─── Streaming-session ffmpeg helpers (video feature) ────────────
//
// CLI's `video-encode` / `video-capacity` / `decode` subcommands route
// through the OpenH264 streaming session (matching the iOS + Android
// bridges). The Rust core consumes raw YUV in / Annex-B out; ffmpeg
// handles the MP4 demux + remux, since phasm-core has no built-in
// H.264 → YUV decoder. These helpers are shared by the OH264 encode
// path (`h264-encoder`) and the decode path (`h264-decoder`), so
// they are gated on the `video` umbrella that both imply.

/// Probed video metadata from `ffprobe`.
#[cfg(feature = "video")]
#[derive(Debug, Clone)]
pub struct VideoProbe {
    pub width: u32,
    pub height: u32,
    pub n_frames: usize,
    /// Frame rate as a "num/den" string (e.g. "30/1", "30000/1001"),
    /// passed verbatim to ffmpeg `-framerate` for the mux step.
    pub frame_rate: String,
    /// True if the input has at least one audio stream we should
    /// preserve when muxing the stego Annex-B back to MP4.
    pub has_audio: bool,
}

/// Run `ffprobe` to read width/height/frame-count/rate from `input`.
///
/// Reports the RAW source dimensions (any size — no alignment enforced here).
/// The encoder needs macroblock/transform alignment (H.264 = 16, AV1 = 8);
/// callers pass the probe to [`decode_to_yuv_aligned`], which CROPS the
/// decoded frames (top-left anchored) down to the nearest valid grid so any
/// input size encodes without a manual pre-crop. Cropping (not scaling)
/// matches the mobile floor-crop in `MediaPlaneExtractor`, and avoids the
/// resample artifacts that would weaken stego stealth.
#[cfg(feature = "video")]
pub fn probe_video(input: &Path) -> Result<VideoProbe, CliError> {
    use std::process::Command;
    let result = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=0",
        ])
        .arg(input)
        .output()
        .map_err(|e| CliError::InvalidArgs(format!(
            "failed to spawn ffprobe: {e}\n\nInstall ffmpeg/ffprobe or transcode manually:\n  {TRANSCODE_SUGGESTION}"
        )))?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(CliError::InvalidArgs(format!(
            "ffprobe failed (exit {}): {stderr}",
            result.status.code().unwrap_or(-1),
        )));
    }
    let out = String::from_utf8_lossy(&result.stdout);
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut n_frames: Option<usize> = None;
    let mut frame_rate: Option<String> = None;
    for line in out.lines() {
        if let Some(rest) = line.strip_prefix("width=") {
            width = rest.trim().parse().ok();
        } else if let Some(rest) = line.strip_prefix("height=") {
            height = rest.trim().parse().ok();
        } else if let Some(rest) = line.strip_prefix("nb_frames=") {
            n_frames = rest.trim().parse().ok();
        } else if let Some(rest) = line.strip_prefix("r_frame_rate=") {
            frame_rate = Some(rest.trim().to_string());
        }
    }
    let width = width.ok_or_else(|| CliError::InvalidArgs(
        "ffprobe did not report video width".into()
    ))?;
    let height = height.ok_or_else(|| CliError::InvalidArgs(
        "ffprobe did not report video height".into()
    ))?;
    let frame_rate = frame_rate.unwrap_or_else(|| "30/1".to_string());

    // nb_frames is sometimes "N/A" for streams without an explicit
    // frame count; fall back to a duration*fps estimate.
    let n_frames = if let Some(n) = n_frames {
        n
    } else {
        probe_n_frames_via_packet_count(input)?
    };

    // Any size is accepted here — `decode_to_yuv_aligned` CROPS the decode
    // (top-left) to the encoder's grid (H.264 = 16, AV1 = 8) so the caller
    // never has to pre-crop. Reports the RAW source dims.
    let has_audio = probe_has_audio(input);
    Ok(VideoProbe { width, height, n_frames, frame_rate, has_audio })
}

#[cfg(feature = "video")]
fn probe_n_frames_via_packet_count(input: &Path) -> Result<usize, CliError> {
    use std::process::Command;
    let result = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "default=nokey=1:noprint_wrappers=1",
        ])
        .arg(input)
        .output()
        .map_err(|e| CliError::InvalidArgs(format!(
            "failed to spawn ffprobe (packet count): {e}"
        )))?;
    let out = String::from_utf8_lossy(&result.stdout);
    out.trim()
        .parse::<usize>()
        .map_err(|_| CliError::InvalidArgs(format!(
            "ffprobe could not determine frame count for {}",
            input.display()
        )))
}

#[cfg(feature = "video")]
fn probe_has_audio(input: &Path) -> bool {
    use std::process::Command;
    Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=nokey=1:noprint_wrappers=1",
        ])
        .arg(input)
        .output()
        .ok()
        .filter(|r| r.status.success())
        .map(|r| !r.stdout.is_empty())
        .unwrap_or(false)
}

/// Largest multiple of `align` that is `<= d` (and `>= align`). The H.264
/// encoder needs 16-aligned dimensions, AV1 8-aligned; [`decode_to_yuv_aligned`]
/// CROPS the source to this grid so any input size encodes.
#[cfg(feature = "video")]
pub fn aligned_dim(d: u32, align: u32) -> u32 {
    let a = align.max(1);
    ((d / a) * a).max(a)
}

/// Decode the input MP4 to raw YUV420p planar frames at `output`.
/// Used to feed the v2 streaming orchestrator, which takes raw YUV.
/// Prefer [`decode_to_yuv_aligned`] for the encode/capacity paths — it CROPS
/// any non-grid-aligned source so the encoder accepts it.
#[cfg(feature = "video")]
pub fn decode_to_yuv(input: &Path, output: &Path) -> Result<(), CliError> {
    decode_to_yuv_cropped(input, output, None)
}

/// Decode + CROP to the encoder's macroblock/transform grid: drops the
/// right/bottom edge strip (at most `align - 1` px per axis) down to the nearest
/// `align`-multiple so ANY source size encodes without a manual pre-crop. This
/// matches the iOS/Android decode path (which floor-crops the same way) — no
/// resampling, no aspect change, no padding. Mutates `probe.width`/
/// `probe.height` to the encoded dims — the caller MUST use these for the
/// session + frame_size + mux. `align` is 16 for H.264, 8 for AV1. No-op (clean
/// passthrough decode) when the source is already aligned.
#[cfg(feature = "video")]
pub fn decode_to_yuv_aligned(
    input: &Path,
    output: &Path,
    probe: &mut VideoProbe,
    align: u32,
) -> Result<(), CliError> {
    let aw = aligned_dim(probe.width, align);
    let ah = aligned_dim(probe.height, align);
    let crop = (aw != probe.width || ah != probe.height).then_some((aw, ah));
    if let Some((w, h)) = crop {
        eprintln!(
            "note: cropping {}x{} -> {w}x{h} ({align}-aligned for the encoder)",
            probe.width, probe.height,
        );
    }
    decode_to_yuv_cropped(input, output, crop)?;
    probe.width = aw;
    probe.height = ah;
    Ok(())
}

/// Inner ffmpeg decode with an optional top-left `-vf crop=W:H:0:0`.
#[cfg(feature = "video")]
fn decode_to_yuv_cropped(
    input: &Path,
    output: &Path,
    crop: Option<(u32, u32)>,
) -> Result<(), CliError> {
    use std::process::Command;
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y").arg("-i").arg(input);
    if let Some((w, h)) = crop {
        // CROP (not scale): keep the top-left WxH, drop the right/bottom
        // alignment strip (<= align-1 px). Anchored at 0,0 to match the mobile
        // floor-crop exactly — no resampling, no aspect distortion, no padding.
        cmd.args(["-vf", &format!("crop={w}:{h}:0:0")]);
    }
    cmd.args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(output);
    let result = cmd.output().map_err(|e| {
        CliError::InvalidArgs(format!(
            "failed to spawn ffmpeg (decode_to_yuv): {e}\n\nInstall ffmpeg or pre-extract YUV manually:\n  {TRANSCODE_SUGGESTION}"
        ))
    })?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        let tail: String = stderr.lines().rev().take(15)
            .collect::<Vec<_>>().into_iter().rev()
            .collect::<Vec<_>>().join("\n");
        return Err(CliError::InvalidArgs(format!(
            "ffmpeg YUV decode failed (exit {}). Last output:\n\n{tail}",
            result.status.code().unwrap_or(-1),
        )));
    }
    Ok(())
}

/// §Stealth.L4.* — output container profile selection. Drives whether
/// the CLI muxes via the phasm-owned HandBrake/x264-medium builder or
/// via ffmpeg passthrough. Default is `HandbrakeX264` because the
/// strategy doc (`docs/design/video/h264/stealth-strategy.md`) names it as
/// the v1.0 ship target — output lands inside the libx264 container
/// metaclass instead of phasm's own (per Yang/EVA + Altinisik).
#[cfg(feature = "video")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum MuxProfile {
    /// HandBrake/x264-medium container shape (default). Video-only —
    /// drops audio from the input. Lands at the Yang/EVA libx264-class
    /// leaf; topology matches the Altinisik tree-hash reference.
    #[default]
    #[value(name = "handbrake-x264")]
    HandbrakeX264,
    /// FFmpeg native mux with audio passthrough. Larger L4
    /// fingerprint distance from libx264 (ffmpeg-class instead of
    /// HandBrake-class) but preserves audio.
    #[value(name = "ffmpeg")]
    Ffmpeg,
}

/// Parse an `r_frame_rate` string ("30/1", "30000/1001") into
/// `(num, den)`. Returns `(30, 1)` on any parse failure — the
/// fallback fps phasm uses for the stealth-measurement fixtures.
#[cfg(feature = "video")]
pub fn parse_frame_rate(s: &str) -> (u32, u32) {
    let mut parts = s.splitn(2, '/');
    let num = parts.next().and_then(|x| x.trim().parse().ok());
    let den = parts.next().and_then(|x| x.trim().parse().ok()).unwrap_or(1u32);
    match num {
        Some(n) if n > 0 && den > 0 => (n, den),
        _ => (30, 1),
    }
}

/// Branching mux: handbrake-x264 → phasm's clean-room HandBrake mux
/// (`phasm_core::codec::mp4::build::build_mp4_with_pattern_audio` when
/// audio is present, `_with_pattern` otherwise); ffmpeg → the legacy
/// shell-out path with audio passthrough via `-c:a copy`.
///
/// Returns whether the chosen profile silently dropped audio.
/// Currently `false` for both branches — HandBrake mux now passes
/// audio through (§Stealth.L4.5) and ffmpeg always has. Kept as a
/// `bool` return for forward compatibility with future profiles
/// that may yet drop audio (e.g. an Apple-rotation profile that
/// replaces the audio track entirely).
#[cfg(feature = "h264-encoder")]
#[allow(clippy::too_many_arguments)]
pub fn mux_annexb_to_mp4_with_profile(
    annex_b_path: &Path,
    audio_source: Option<&Path>,
    frame_rate: &str,
    width: u32,
    height: u32,
    n_display_frames: usize,
    gop_pattern: phasm_core::GopPattern,
    profile: MuxProfile,
    output: &Path,
) -> Result<bool, CliError> {
    let dropped_audio = match profile {
        MuxProfile::HandbrakeX264 => {
            let annex_b = std::fs::read(annex_b_path)?;
            let (fps_num, fps_den) = parse_frame_rate(frame_rate);
            let timing = phasm_core::codec::mp4::build::FrameTiming { fps_num, fps_den };
            let mp4 = if let Some(src_path) = audio_source {
                let source_mp4 = std::fs::read(src_path).map_err(|e| {
                    CliError::InvalidArgs(format!(
                        "failed to read source MP4 for audio passthrough: {e}"
                    ))
                })?;
                phasm_core::codec::mp4::build::build_mp4_with_pattern_audio(
                    phasm_core::codec::mp4::build::MuxerProfile::HandbrakeX264,
                    &annex_b,
                    width, height, timing, gop_pattern, n_display_frames,
                    &source_mp4,
                )
                .map_err(|e| CliError::InvalidArgs(format!(
                    "HandBrake mux with audio failed: {e}"
                )))?
            } else {
                phasm_core::codec::mp4::build::build_mp4_with_pattern(
                    phasm_core::codec::mp4::build::MuxerProfile::HandbrakeX264,
                    &annex_b,
                    width, height, timing, gop_pattern, n_display_frames,
                )
                .map_err(|e| CliError::InvalidArgs(format!(
                    "HandBrake mux failed: {e}"
                )))?
            };
            std::fs::write(output, mp4)?;
            false // §Stealth.L4.5: HandBrake profile now passes audio through.
        }
        MuxProfile::Ffmpeg => {
            mux_annexb_to_mp4(annex_b_path, audio_source, frame_rate, output)?;
            false
        }
    };
    Ok(dropped_audio)
}

/// Mux a raw H.264 Annex-B stream into MP4. If `audio_source` is
/// `Some`, the audio track is copied from there (the original input
/// file). The video track uses the supplied frame rate.
#[cfg(feature = "h264-encoder")]
pub fn mux_annexb_to_mp4(
    annex_b: &Path,
    audio_source: Option<&Path>,
    frame_rate: &str,
    output: &Path,
) -> Result<(), CliError> {
    use std::process::Command;
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y").arg("-loglevel").arg("error")
        .args(["-framerate", frame_rate, "-f", "h264"])
        .arg("-i").arg(annex_b);
    if let Some(audio) = audio_source {
        cmd.arg("-i").arg(audio)
            .args(["-map", "0:v:0", "-map", "1:a:0?"])
            .args(["-c:v", "copy", "-c:a", "copy"]);
    } else {
        cmd.args(["-c:v", "copy"]);
    }
    cmd.arg(output);

    let result = cmd.output().map_err(|e| CliError::InvalidArgs(format!(
        "failed to spawn ffmpeg (mux_annexb_to_mp4): {e}"
    )))?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(CliError::InvalidArgs(format!(
            "ffmpeg mux failed (exit {}): {stderr}",
            result.status.code().unwrap_or(-1),
        )));
    }
    Ok(())
}

/// Extract the H.264 video track from an MP4 as raw Annex-B bytes
/// (start-code-prefixed NAL stream, the format the CABAC decoder
/// consumes). Uses ffmpeg's `h264_mp4toannexb` bitstream filter.
#[cfg(feature = "video")]
pub fn extract_annexb_from_mp4(input: &Path, output: &Path) -> Result<(), CliError> {
    use std::process::Command;
    let result = Command::new("ffmpeg")
        .arg("-y").arg("-loglevel").arg("error")
        .arg("-i").arg(input)
        .args(["-c:v", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "h264"])
        .arg(output)
        .output()
        .map_err(|e| CliError::InvalidArgs(format!(
            "failed to spawn ffmpeg (extract_annexb): {e}"
        )))?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(CliError::InvalidArgs(format!(
            "ffmpeg Annex-B extract failed (exit {}): {stderr}",
            result.status.code().unwrap_or(-1),
        )));
    }
    Ok(())
}

/// Produce a unique tempfile path with a custom extension. Caller is
/// responsible for cleaning it up.
#[cfg(feature = "video")]
pub fn temp_path_with_ext(original: &Path, ext: &str) -> PathBuf {
    let stem = original
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("phasm_temp");
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("{stem}_phasm_cabac_{pid}_{ts}.{ext}"))
}
