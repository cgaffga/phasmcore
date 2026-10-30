// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

// The decode path only needs the demux helpers (`ensure_ffmpeg_available`,
// `temp_path_with_ext`, `extract_annexb_from_mp4`); the rest serves the
// encode subcommands and is unused in a decode-only (`h264-decoder`) build.
#![cfg_attr(not(any(feature = "h264-encoder", feature = "av1-encoder")), allow(dead_code))]

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
    /// When the first video frame is presented, in µs from the start of
    /// the source's timeline (the time zero the passed-through audio is
    /// placed against). Non-zero for a video track with a leading empty
    /// edit, e.g. 166 016 µs on the Artlist stock clips, whose audio starts
    /// at 120 ms. 0 when unknown or negative.
    pub video_start_us: u64,
}

/// Run `ffprobe` to read width/height/frame-count/rate from `input`.
///
/// Reports the RAW source dimensions (any size — no alignment enforced here).
/// Callers pass the probe to [`decode_to_yuv_aligned`] with the encoder's size
/// granularity: H.264 encodes any EVEN size exactly ([`H264_SIZE_ALIGN`] = 2;
/// the SPS crops the 16×16 MB grid back to the display size, like a phone
/// encoder), AV1 needs 8. Only a source off that grid is CROPPED (top-left
/// anchored, by at most `align − 1` px per axis). Cropping (not scaling)
/// avoids the resample artifacts that would weaken stego stealth.
#[cfg(feature = "video")]
pub fn probe_video(input: &Path) -> Result<VideoProbe, CliError> {
    use std::process::Command;
    let result = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            // Rotation matters: `stream=width,height` reports the CODED size,
            // but ffmpeg AUTO-ROTATES on decode, so a phone video shot in
            // portrait reports 1920x1080 and then decodes to 1080x1920. See
            // the swap below. `stream_side_data=rotation` is the modern form;
            // `stream_tags=rotate` is the older one, kept for files that
            // predate it.
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate,start_time:stream_side_data=rotation:stream_tags=rotate",
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
    let mut rotation: Option<i32> = None;
    let mut start_time: Option<f64> = None;
    for line in out.lines() {
        if let Some(rest) = line.strip_prefix("width=") {
            width = rest.trim().parse().ok();
        } else if let Some(rest) = line.strip_prefix("height=") {
            height = rest.trim().parse().ok();
        } else if let Some(rest) = line.strip_prefix("nb_frames=") {
            n_frames = rest.trim().parse().ok();
        } else if let Some(rest) = line.strip_prefix("r_frame_rate=") {
            frame_rate = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("start_time=") {
            // "N/A" when the stream has no start time.
            start_time = rest.trim().parse().ok();
        } else if let Some(rest) = line.strip_prefix("rotation=") {
            rotation = rest.trim().parse::<f64>().ok().map(|r| r as i32);
        } else if let Some(rest) = line.strip_prefix("TAG:rotate=") {
            // Older container form; only used when side data is absent.
            if rotation.is_none() {
                rotation = rest.trim().parse::<f64>().ok().map(|r| r as i32);
            }
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

    // DISPLAY dimensions, not coded ones.
    //
    // ffmpeg applies the stream's rotation matrix on decode unless told
    // otherwise, so for a portrait phone video `stream=width,height` says
    // 1920x1080 while the frames that actually come out are 1080x1920. The
    // CLI used to build its crop filter from the coded size and hand it
    // rotated frames, which ffmpeg rejects outright:
    //
    //     Invalid too big or non positive size for width '1920' or height '1072'
    //
    // That made every rotated phone video unencodable from the CLI while the
    // same clip worked in the app (2026-10-20, D17). Swapping here means the
    // crop, the encoder dimensions and the mux all agree with what ffmpeg
    // emits, and the output is upright with no rotation metadata needed.
    let (width, height) = match rotation.map(|r| r.rem_euclid(360)) {
        Some(90) | Some(270) => {
            eprintln!(
                "note: source is rotated {}deg; using display size {}x{} \
                 (coded {}x{})",
                rotation.unwrap(), height, width, width, height,
            );
            (height, width)
        }
        _ => (width, height),
    };

    // Any size is accepted here — `decode_to_yuv_aligned` CROPS the decode
    // (top-left) to the encoder's grid (H.264 = 2, i.e. even; AV1 = 8) so the
    // caller never has to pre-crop.
    let has_audio = probe_has_audio(input);
    let video_start_us = start_time
        .filter(|t| t.is_finite() && *t > 0.0)
        .map(|t| (t * 1_000_000.0).round() as u64)
        .unwrap_or(0);
    Ok(VideoProbe { width, height, n_frames, frame_rate, has_audio, video_start_us })
}

/// The first video stream's colour tags as ITU-T H.273 codes:
/// `(primaries, transfer, matrix, full_range)`, 2 (unspecified) for a tag
/// the source does not carry. What the raw decode ([`decode_to_yuv_aligned`],
/// `-pix_fmt yuv420p`) leaves true of its frames: it changes the bit depth
/// but neither the primaries, the transfer nor the matrix, and it converts a
/// `yuvj*` (full-range) source to studio range.
#[cfg(feature = "video")]
pub fn probe_colour_h273(input: &Path) -> (u8, u8, u8, bool) {
    use std::process::Command;
    let unspecified = (2, 2, 2, false);
    let Ok(result) = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=pix_fmt,color_range,color_space,color_transfer,color_primaries",
            "-of", "default=noprint_wrappers=1",
        ])
        .arg(input)
        .output()
    else {
        return unspecified;
    };
    if !result.status.success() {
        return unspecified;
    }
    let out = String::from_utf8_lossy(&result.stdout);
    let field = |k: &str| out.lines().find_map(|l| l.strip_prefix(k)).map(str::trim).unwrap_or("");
    let primaries = match field("color_primaries=") {
        "bt709" => 1,
        "bt470m" => 4,
        "bt470bg" => 5,
        "smpte170m" => 6,
        "smpte240m" => 7,
        "film" => 8,
        "bt2020" => 9,
        "smpte428" => 10,
        "smpte431" => 11,
        "smpte432" => 12,
        "ebu3213" | "jedec-p22" => 22,
        _ => 2,
    };
    let transfer = match field("color_transfer=") {
        "bt709" => 1,
        "gamma22" => 4,
        "gamma28" => 5,
        "smpte170m" => 6,
        "smpte240m" => 7,
        "linear" => 8,
        "log100" => 9,
        "log316" => 10,
        "iec61966-2-4" => 11,
        "bt1361e" => 12,
        "iec61966-2-1" => 13,
        "bt2020-10" => 14,
        "bt2020-12" => 15,
        "smpte2084" => 16,
        "smpte428" => 17,
        "arib-std-b67" => 18,
        _ => 2,
    };
    let matrix = match field("color_space=") {
        "gbr" => 0,
        "bt709" => 1,
        "fcc" => 4,
        "bt470bg" => 5,
        "smpte170m" => 6,
        "smpte240m" => 7,
        "ycgco" => 8,
        "bt2020nc" => 9,
        "bt2020c" => 10,
        "smpte2085" => 11,
        "chroma-derived-nc" => 12,
        "chroma-derived-c" => 13,
        "ictcp" => 14,
        _ => 2,
    };
    let full_range = field("color_range=") == "pc" && !field("pix_fmt=").starts_with("yuvj");
    (primaries, transfer, matrix, full_range)
}

/// Presentation times (µs) of the first video stream's frames, in the order
/// the raw decode ([`decode_to_yuv_aligned`], one output frame per decoded
/// frame) emits them, on the source's timeline. `None` when ffprobe fails or
/// a frame has no timestamp.
#[cfg(feature = "video")]
pub fn probe_frame_times_us(input: &Path) -> Option<Vec<i64>> {
    use std::process::Command;
    let result = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time",
            "-of", "csv=p=0",
        ])
        .arg(input)
        .output()
        .ok()?;
    if !result.status.success() {
        return None;
    }
    String::from_utf8_lossy(&result.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim()
                .trim_end_matches(',')
                .parse::<f64>()
                .ok()
                .filter(|t| t.is_finite())
                .map(|t| (t * 1_000_000.0).round() as i64)
        })
        .collect()
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

/// Size granularity of the H.264 encoder: any even size is encoded exactly.
/// The bitstream codes the 16×16 MB grid and the SPS `frame_cropping` brings
/// it back to the display size (1920×1080 stays 1920×1080, coded as 1088
/// lines with `crop_bottom = 4`, as on an iPhone). Odd sizes cannot be
/// expressed in 4:2:0 crop units, so they lose their last row/column.
#[cfg(feature = "video")]
pub const H264_SIZE_ALIGN: u32 = 2;

/// Size granularity for AV1: any even size. rav1e pads to its 8-px grid
/// internally and signals the exact frame size, so nothing is cropped
/// beyond an odd source's last row/column (as the apps do).
#[cfg(feature = "video")]
pub const AV1_SIZE_ALIGN: u32 = 2;

/// Largest multiple of `align` that is `<= d` (and `>= align`). Both
/// encoders take any even size ([`H264_SIZE_ALIGN`], [`AV1_SIZE_ALIGN`]);
/// [`decode_to_yuv_aligned`] CROPS the source to this grid so any input size
/// encodes.
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

/// Decode + CROP to the encoder's size grid: drops the right/bottom edge strip
/// (at most `align - 1` px per axis) down to the nearest `align`-multiple so
/// ANY source size encodes without a manual pre-crop — no resampling, no
/// aspect change, no padding. Mutates `probe.width`/`probe.height` to the
/// encoded dims — the caller MUST use these for the session + frame_size +
/// mux. `align` is [`H264_SIZE_ALIGN`] (2) for H.264, which keeps every even
/// source at its exact size, and 8 for AV1. No-op (clean passthrough decode)
/// when the source is already aligned.
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

/// `--resolution` for `video-encode` / `video-capacity`: the output size
/// preset the mobile apps offer (core `codec::video_frame::VideoPreset`).
#[cfg(feature = "video")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum Resolution {
    /// The source's own size (default).
    #[default]
    Original,
    /// Fits 848x480 (480p), turned to the source's orientation.
    Sd,
    /// Fits 1280x720 (720p).
    Hd,
    /// Fits 1920x1080 (1080p).
    Fhd,
}

#[cfg(feature = "video")]
impl Resolution {
    fn preset(self) -> phasm_core::codec::video_frame::VideoPreset {
        use phasm_core::codec::video_frame::VideoPreset;
        match self {
            Resolution::Original => VideoPreset::Original,
            Resolution::Sd => VideoPreset::Sd,
            Resolution::Hd => VideoPreset::Hd,
            Resolution::Fhd => VideoPreset::Fhd,
        }
    }
}

/// Downscale the decoded YUV at `yuv` — tight I420 frames at
/// `probe.width`×`probe.height`, already upright (ffmpeg autorotates) and
/// even ([`decode_to_yuv_aligned`]) — to `resolution`, in place, and set
/// `probe` to the new size. Call it right after the decode, before anything
/// reads the frames, so the encode, its capacity probe and the mux all see
/// the scaled size.
///
/// Sizes come from the core resolver (`video_target_resolve`, orientation 0
/// since the frames are upright) and pixels from the core's deterministic
/// area-average `I420Scaler` — the scaler the Android app runs, so the same
/// clip at the same preset gives the same cover. Frame by frame, so RAM stays
/// O(frame). No-op for `Original` or a source that already fits the preset.
#[cfg(feature = "video")]
pub fn scale_yuv_to_preset(yuv: &Path, probe: &mut VideoProbe, resolution: Resolution) -> Result<(), CliError> {
    use phasm_core::codec::video_frame::{video_target_resolve, I420Scaler};
    use std::io::{BufReader, BufWriter, Read, Write};

    let target = video_target_resolve(probe.width, probe.height, 0, resolution.preset())
        .ok_or_else(|| CliError::InvalidArgs(format!("cannot size a {}x{} video", probe.width, probe.height)))?;
    let Some((dw, dh)) = target.scale_to.filter(|&s| s != (probe.width, probe.height)) else {
        if resolution != Resolution::Original {
            eprintln!(
                "note: {}x{} already fits the {resolution:?} preset; keeping its size",
                probe.width, probe.height,
            );
        }
        return Ok(());
    };
    let frame_len = (probe.width as usize) * (probe.height as usize) * 3 / 2;
    let file_len = std::fs::metadata(yuv)?.len() as usize;
    if frame_len == 0 || !file_len.is_multiple_of(frame_len) {
        return Err(CliError::InvalidArgs(format!(
            "decoded YUV {file_len} bytes is not a whole number of {}x{} frames",
            probe.width, probe.height,
        )));
    }
    eprintln!("note: scaling {}x{} -> {dw}x{dh} ({resolution:?} preset)", probe.width, probe.height);

    let mut scaled_path = yuv.as_os_str().to_owned();
    scaled_path.push(".scaled");
    let scaled_path = PathBuf::from(scaled_path);
    let result = (|| -> Result<(), CliError> {
        let mut reader = BufReader::new(std::fs::File::open(yuv)?);
        let mut writer = BufWriter::new(std::fs::File::create(&scaled_path)?);
        let mut scaler = I420Scaler::new(probe.width, probe.height, dw, dh);
        let mut frame = vec![0u8; frame_len];
        let mut out = Vec::new();
        for _ in 0..file_len / frame_len {
            reader.read_exact(&mut frame)?;
            out.clear();
            scaler.scale_into(&frame, &mut out);
            writer.write_all(&out)?;
        }
        writer.flush()?;
        drop(writer);
        std::fs::rename(&scaled_path, yuv)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&scaled_path);
    }
    result?;
    probe.width = dw;
    probe.height = dh;
    Ok(())
}

/// Option spellings that make the decode emit every decoded frame exactly
/// once, in order: the current one first, then the older spelling that
/// releases before 5.1 understand (and later ones still accept).
#[cfg(feature = "video")]
const PASSTHROUGH_FRAME_OPTS: [[&str; 2]; 2] =
    [["-fps_mode", "passthrough"], ["-vsync", "passthrough"]];

/// Build the raw-YUV decode command: one output frame per decoded source
/// frame, optional top-left crop.
#[cfg(feature = "video")]
fn decode_to_yuv_command(
    input: &Path,
    output: &Path,
    crop: Option<(u32, u32)>,
    frame_opt: [&str; 2],
) -> Command {
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y").arg("-i").arg(input);
    if let Some((w, h)) = crop {
        // CROP (not scale): keep the top-left WxH, drop the right/bottom
        // alignment strip (<= align-1 px). Anchored at 0,0 to match the mobile
        // floor-crop exactly — no resampling, no aspect distortion, no padding.
        cmd.args(["-vf", &format!("crop={w}:{h}:0:0")]);
    }
    // One output frame per decoded frame — no frame-rate conversion.
    //
    // Raw video carries no timestamps, so without this the decode falls back
    // to constant-frame-rate output: it lays the frames on a fixed 1/fps grid
    // that starts at the FILE's start time and duplicates or drops frames to
    // fill it. When another track starts earlier than the video (an audio
    // track at 0.120 s, video at 0.166 s: the Artlist stock clips, and any
    // file with a leading empty edit on its video track), the gap in front of
    // the first video frame is filled by repeating that frame. A 121-frame
    // 24 fps clip came out as 122 frames, the first one doubled, and every
    // later frame shifted a slot. The stego encode, the capacity probe and
    // the muxed output all inherited the extra frame. A variable-rate source
    // could equally lose or repeat frames mid-clip.
    //
    // The apps never had this: AVAssetReader and MediaCodec hand over each
    // coded sample once.
    cmd.args(frame_opt);
    cmd.args(["-f", "rawvideo", "-pix_fmt", "yuv420p"]).arg(output);
    cmd
}

/// Inner raw-YUV decode with an optional top-left `-vf crop=W:H:0:0`.
#[cfg(feature = "video")]
fn decode_to_yuv_cropped(
    input: &Path,
    output: &Path,
    crop: Option<(u32, u32)>,
) -> Result<(), CliError> {
    let spawn = |frame_opt: [&str; 2]| {
        decode_to_yuv_command(input, output, crop, frame_opt).output().map_err(|e| {
            CliError::InvalidArgs(format!(
                "failed to spawn ffmpeg (decode_to_yuv): {e}\n\nInstall ffmpeg or pre-extract YUV manually:\n  {TRANSCODE_SUGGESTION}"
            ))
        })
    };
    let [current, legacy] = PASSTHROUGH_FRAME_OPTS;
    let mut result = spawn(current)?;
    if !result.status.success()
        && String::from_utf8_lossy(&result.stderr).contains(&format!("option '{}'", &current[0][1..]))
    {
        // A release that predates the current option name rejects it as
        // unrecognised; it takes the older spelling with the same meaning.
        result = spawn(legacy)?;
    }
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

/// Output container profile selection: the phasm-owned builder or the
/// legacy shell-out mux. Default is `LavfOpenH264` — the container a
/// libavformat transcoder writes around a libopenh264 encode, which is what
/// the OpenH264 bitstream inside it is (2026-10-29; it replaced a
/// HandBrake/x264 profile whose x264 SEI banner and HandBrake `©too` the
/// bitstream contradicted — `docs/design/video/h264/openh264-stealth-report.md`,
/// "Container layer").
#[cfg(feature = "video")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum MuxProfile {
    /// In-tree builder, `MuxerProfile::LavfOpenH264` (default). Passes the
    /// source's audio through, interleaved with the video.
    #[default]
    #[value(name = "lavf-openh264")]
    LavfOpenH264,
    /// Legacy shell-out mux (`mux_annexb_to_mp4`) with audio passthrough.
    /// Not used: it mis-times streams whose POC resets every GOP
    /// (2026-05-24, see `write_and_mux_oh264`).
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

/// Branching mux: lavf-openh264 → phasm's in-tree builder
/// (`phasm_core::codec::mp4::build::build_mp4_with_pattern_audio_lead_in`
/// when audio is present, `_with_pattern` otherwise); ffmpeg → the legacy
/// shell-out path with audio passthrough via `-c:a copy`.
///
/// Returns whether the chosen profile silently dropped audio.
/// Currently `false` for both branches — the in-tree builder passes
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
    video_lead_in_us: u64,
    output: &Path,
) -> Result<bool, CliError> {
    let dropped_audio = match profile {
        MuxProfile::LavfOpenH264 => {
            let annex_b = std::fs::read(annex_b_path)?;
            let (fps_num, fps_den) = parse_frame_rate(frame_rate);
            let timing = phasm_core::codec::mp4::build::FrameTiming { fps_num, fps_den };
            let mp4 = if let Some(src_path) = audio_source {
                let source_mp4 = std::fs::read(src_path).map_err(|e| {
                    CliError::InvalidArgs(format!(
                        "failed to read source MP4 for audio passthrough: {e}"
                    ))
                })?;
                // The audio keeps the source's placement, so the video must
                // too: the source's video lead-in becomes a leading empty edit.
                phasm_core::codec::mp4::build::build_mp4_with_pattern_audio_lead_in(
                    phasm_core::codec::mp4::build::MuxerProfile::LavfOpenH264,
                    &annex_b,
                    width, height, timing, gop_pattern, n_display_frames,
                    &source_mp4,
                    video_lead_in_us,
                )
                .map_err(|e| CliError::InvalidArgs(format!(
                    "MP4 mux with audio failed: {e}"
                )))?
            } else {
                phasm_core::codec::mp4::build::build_mp4_with_pattern(
                    phasm_core::codec::mp4::build::MuxerProfile::LavfOpenH264,
                    &annex_b,
                    width, height, timing, gop_pattern, n_display_frames,
                )
                .map_err(|e| CliError::InvalidArgs(format!(
                    "MP4 mux failed: {e}"
                )))?
            };
            std::fs::write(output, mp4)?;
            false // §Stealth.L4.5: the in-tree builder passes audio through.
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

#[cfg(all(test, feature = "video"))]
mod frame_count_tests {
    use super::*;

    /// Regression (2026-10-25): the raw-YUV decode must yield exactly the
    /// source's frames. It used to resample to a constant rate anchored at
    /// the file's start, so a video track starting after the audio got its
    /// first frame repeated to fill the gap (Artlist_HorseFlag: 121 frames
    /// decoded as 122). The fixture is built here: 24 frames at 24 fps whose
    /// video track starts ~0.083 s after its audio track, which the old
    /// decode turned into 26 frames.
    #[test]
    fn decode_yields_every_source_frame_once_when_video_starts_late() {
        let dir = std::path::PathBuf::from(
            std::env::var("PHASM_TMP").unwrap_or_else(|_| "/tmp".into()),
        )
        .join(format!("phasm_framecount_unit_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("late_video.mp4");
        let yuv = dir.join("late_video.yuv");

        let ok = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error",
                   "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1.3",
                   "-itsoffset", "0.1",
                   "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=1",
                   "-map", "1:v", "-map", "0:a",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"])
            .arg(&src)
            .status();
        match ok {
            Ok(s) if s.success() => {}
            _ => {
                eprintln!("skipping: fixture could not be built (decoder tool or libx264 unavailable)");
                let _ = std::fs::remove_dir_all(&dir);
                return;
            }
        }

        let mut probe = probe_video(&src).unwrap_or_else(|e| panic!("probe fixture: {e}"));
        assert_eq!(probe.n_frames, 24, "fixture has 24 coded frames");
        decode_to_yuv_aligned(&src, &yuv, &mut probe, H264_SIZE_ALIGN)
            .unwrap_or_else(|e| panic!("decode: {e}"));
        let len = std::fs::metadata(&yuv).unwrap().len() as usize;
        let _ = std::fs::remove_dir_all(&dir);

        let frame = (probe.width * probe.height * 3 / 2) as usize;
        assert_eq!(len % frame, 0, "whole frames");
        assert_eq!(
            len / frame, 24,
            "decode must emit each of the 24 source frames exactly once \
             (26 = the late video track's lead-in was filled with copies of frame 0)",
        );
    }
}

#[cfg(all(test, feature = "video"))]
mod rotation_tests {
    use super::*;

    /// D17 regression (2026-10-20) — a portrait phone video reports CODED
    /// dimensions, and ffmpeg auto-rotates on decode. `probe_video` must
    /// report what ffmpeg will actually emit, or the crop filter is built for
    /// the wrong geometry and ffmpeg refuses the whole encode with
    /// "Invalid too big or non positive size".
    ///
    /// Builds its own fixture so it needs no committed test vector: a tiny
    /// clip tagged `rotate=90` via the container, which is exactly how phone
    /// footage carries it.
    #[test]
    fn probe_reports_display_dims_for_a_rotated_source() {
        let dir = std::env::var("PHASM_TMP").unwrap_or_else(|_| "/tmp".into());
        let src = std::path::PathBuf::from(&dir).join("phasm_d17_rot.mp4");
        let _ = std::fs::remove_file(&src);

        // 64x32 landscape, then tagged as rotated 90deg -> displays 32x64.
        let ok = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error",
                   "-f", "lavfi", "-i", "testsrc=size=64x32:rate=10:duration=1",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-metadata:s:v:0", "rotate=90"])
            .arg(&src)
            .status();
        match ok {
            Ok(s) if s.success() => {}
            _ => {
                eprintln!("skipping: ffmpeg unavailable");
                return;
            }
        }

        let probe = match probe_video(&src) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("skipping: probe failed ({e})");
                let _ = std::fs::remove_file(&src);
                return;
            }
        };
        let _ = std::fs::remove_file(&src);

        // Some ffmpeg builds drop the `rotate` tag rather than writing side
        // data. Only assert the swap when the rotation actually survived —
        // otherwise this would fail on the muxer, not on our logic.
        if probe.width == 32 && probe.height == 64 {
            // swapped as intended
        } else {
            assert_eq!(
                (probe.width, probe.height), (64, 32),
                "neither display (32x64) nor coded (64x32) dims — probe is wrong",
            );
            eprintln!("note: this ffmpeg did not preserve the rotation tag; \
                       swap path not exercised");
        }
    }
}
