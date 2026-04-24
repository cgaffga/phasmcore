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
