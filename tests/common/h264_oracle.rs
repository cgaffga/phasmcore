// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 decode-oracle harness (Phase 6.0c).
//!
//! Runs phasm-encoded H.264 output through a third-party decoder
//! (ffmpeg) and reports whether the bitstream decodes cleanly and
//! whether the decoded pixels match a reference reconstruction.
//!
//! Design doc: `docs/design/video/h264/encoder-algorithms/oracle-harness.md`
//!
//! Three layers:
//!   - Layer 0: ffmpeg parses the file (no crash, no errors).
//!   - Layer 1: ffmpeg decodes to YUV without truncation.
//!   - Layer 2: decoded YUV matches a caller-provided reference.
//!
//! This file ships Layer 0 + Layer 1 immediately. Layer 2 activates
//! once Phase 6A.6 (reconstruction) lands and can produce an
//! expected YUV buffer.

use std::io::Write;
use std::process::{Command, Stdio};

/// Result of a successful oracle decode.
#[derive(Debug)]
pub struct OracleResult {
    /// Decoded yuv420p bytes, frame-major, packed per plane.
    pub decoded_yuv: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub frame_count: u32,
    pub ffmpeg_stderr: String,
}

/// Errors from the oracle harness.
#[derive(Debug)]
pub enum OracleError {
    /// `ffmpeg` binary not found on PATH. Tests should `skip` on this.
    FfmpegNotInstalled,
    /// Spawning/reading the ffmpeg subprocess failed.
    Io(std::io::Error),
    /// ffmpeg returned a non-zero exit code.
    FfmpegExitCode { code: i32, stderr: String },
    /// Decoded YUV is shorter than `width*height*1.5*frame_count`.
    ShortDecode { expected: usize, got: usize },
    /// Layer 2: decoded pixels did not match reference.
    PixelMismatch { frame: u32, mse: f64 },
    /// ffprobe couldn't determine stream dimensions.
    ProbeFailed(String),
}

impl From<std::io::Error> for OracleError {
    fn from(e: std::io::Error) -> Self {
        OracleError::Io(e)
    }
}

/// Return true iff `ffmpeg` is available on PATH.
///
/// Use this at the top of oracle-dependent tests and `return` early
/// (or call `skip!()`) if missing — oracle tests are expected to be
/// skipped silently in CI environments without ffmpeg.
pub fn system_has_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Layer 0 + Layer 1: feed `mp4_bytes` into ffmpeg via stdin, pull
/// decoded yuv420p bytes from stdout, return them.
///
/// Uses ffprobe first to read the stream dimensions — we don't trust
/// ffmpeg's stderr format for this.
pub fn decode_via_ffmpeg(mp4_bytes: &[u8]) -> Result<OracleResult, OracleError> {
    decode_via_ffmpeg_with_format(mp4_bytes, None)
}

/// Same as `decode_via_ffmpeg` but with a user-specified ffmpeg input
/// format (e.g. `"h264"` for raw Annex B streams).
pub fn decode_via_ffmpeg_with_format(
    bytes: &[u8],
    input_format: Option<&str>,
) -> Result<OracleResult, OracleError> {
    if !system_has_ffmpeg() {
        return Err(OracleError::FfmpegNotInstalled);
    }

    let (width, height, frame_count) = match input_format {
        Some(fmt) => probe_annexb_dimensions(bytes, fmt)?,
        None => ffprobe_dimensions(bytes)?,
    };

    let mut args: Vec<String> = vec![
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
    ];
    if let Some(fmt) = input_format {
        args.push("-f".into());
        args.push(fmt.into());
    }
    args.extend(
        [
            "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1",
        ]
        .iter()
        .map(std::string::ToString::to_string),
    );
    let mut child = Command::new("ffmpeg")
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Write input on a background thread so we can read stdout
    // concurrently — ffmpeg's pipe buffering would otherwise deadlock.
    let mut stdin = child.stdin.take().expect("stdin is piped");
    let bytes_copy = bytes.to_vec();
    let writer_thread = std::thread::spawn(move || stdin.write_all(&bytes_copy));

    let output = child.wait_with_output()?;
    writer_thread.join().expect("stdin writer panicked")?;

    if !output.status.success() {
        return Err(OracleError::FfmpegExitCode {
            code: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        });
    }

    // yuv420p = 1.5 bytes per pixel per frame.
    let expected = (width as usize) * (height as usize) * 3 / 2 * (frame_count as usize);
    if output.stdout.len() < expected {
        return Err(OracleError::ShortDecode {
            expected,
            got: output.stdout.len(),
        });
    }

    Ok(OracleResult {
        decoded_yuv: output.stdout,
        width,
        height,
        frame_count,
        ffmpeg_stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

/// Probe dimensions of a raw Annex B / H.264 byte stream by asking
/// ffprobe with an explicit format flag.
fn probe_annexb_dimensions(
    bytes: &[u8],
    input_format: &str,
) -> Result<(u32, u32, u32), OracleError> {
    let mut child = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-f",
            input_format,
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            "pipe:0",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| OracleError::FfmpegNotInstalled)?;

    let mut stdin = child.stdin.take().expect("stdin is piped");
    let bytes_copy = bytes.to_vec();
    let writer = std::thread::spawn(move || stdin.write_all(&bytes_copy));
    let output = child.wait_with_output()?;
    writer.join().expect("probe stdin writer panicked").ok();

    if !output.status.success() {
        return Err(OracleError::ProbeFailed(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let first = text.lines().next().unwrap_or("").trim();
    let parts: Vec<&str> = first.split(',').collect();
    if parts.len() < 2 {
        return Err(OracleError::ProbeFailed(format!(
            "unparseable annexb probe: '{first}'"
        )));
    }
    let w: u32 = parts[0]
        .parse()
        .map_err(|_| OracleError::ProbeFailed(format!("bad width: '{}'", parts[0])))?;
    let h: u32 = parts[1]
        .parse()
        .map_err(|_| OracleError::ProbeFailed(format!("bad height: '{}'", parts[1])))?;
    // Frame count isn't reliable from an Annex B stream without decoding;
    // caller should rely on decoded-bytes length / frame-size instead.
    Ok((w, h, 1))
}

/// Probe stream dimensions via ffprobe.
///
/// Returns `(width, height, frame_count)`. ffprobe is part of every
/// ffmpeg install so we don't need a separate `system_has_ffprobe`.
fn ffprobe_dimensions(mp4_bytes: &[u8]) -> Result<(u32, u32, u32), OracleError> {
    let mut child = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "csv=p=0",
            "pipe:0",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| OracleError::FfmpegNotInstalled)?;

    let mut stdin = child.stdin.take().expect("stdin is piped");
    let bytes = mp4_bytes.to_vec();
    let writer = std::thread::spawn(move || stdin.write_all(&bytes));
    let output = child.wait_with_output()?;
    writer.join().expect("probe stdin writer panicked").ok();

    if !output.status.success() {
        return Err(OracleError::ProbeFailed(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    // Expected: "320,240,15\n" (or similar). nb_frames may be N/A for
    // streaming MP4s — we fall back on 1 so callers don't over-check.
    let first = text.lines().next().unwrap_or("").trim();
    let parts: Vec<&str> = first.split(',').collect();
    if parts.len() < 2 {
        return Err(OracleError::ProbeFailed(format!("unparseable: '{first}'")));
    }
    let w: u32 = parts[0].parse().map_err(|_| {
        OracleError::ProbeFailed(format!("bad width: '{}'", parts[0]))
    })?;
    let h: u32 = parts[1].parse().map_err(|_| {
        OracleError::ProbeFailed(format!("bad height: '{}'", parts[1]))
    })?;
    let n: u32 = parts
        .get(2)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1);
    Ok((w, h, n))
}

/// Layer 2: assert decoded pixels match a reference reconstruction.
///
/// Phase 6.0c stub — activates once Phase 6A.6 ships the encoder's
/// own reconstruction loop and can feed us a reference buffer.
pub fn assert_decoder_matches_recon(
    mp4_bytes: &[u8],
    expected_yuv: &[u8],
    tolerance_mse: f64,
) -> Result<(), OracleError> {
    let r = decode_via_ffmpeg(mp4_bytes)?;
    if r.decoded_yuv.len() != expected_yuv.len() {
        return Err(OracleError::ShortDecode {
            expected: expected_yuv.len(),
            got: r.decoded_yuv.len(),
        });
    }
    // Per-frame MSE. H.264 IDCT is integer so MSE should be 0.0 for
    // a spec-compliant encoder; any nonzero MSE means a bug.
    let frame_bytes = (r.width as usize) * (r.height as usize) * 3 / 2;
    for frame_idx in 0..r.frame_count {
        let start = (frame_idx as usize) * frame_bytes;
        let end = start + frame_bytes;
        let d = &r.decoded_yuv[start..end];
        let e = &expected_yuv[start..end];
        let mse = mean_square_error(d, e);
        if mse > tolerance_mse {
            return Err(OracleError::PixelMismatch {
                frame: frame_idx,
                mse,
            });
        }
    }
    Ok(())
}

fn mean_square_error(a: &[u8], b: &[u8]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64) - (*y as f64);
        acc += d * d;
    }
    acc / a.len() as f64
}
