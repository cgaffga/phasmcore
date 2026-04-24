// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::output::{self, OutputMode};
use crate::passphrase::get_passphrase;
use crate::progress::spawn_progress_bar;
use crate::transcode;
use phasm_core::{detect_video_codec, h264_ghost_encode, VideoCodec};
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

#[derive(Parser)]
pub struct VideoEncodeArgs {
    /// Cover MP4 video file
    pub video: PathBuf,

    /// Message text (reads from stdin if omitted in pipe mode)
    #[arg(short = 'm')]
    pub message: Option<String>,

    /// Passphrase (interactive prompt if omitted)
    #[arg(short = 'p')]
    pub passphrase: Option<String>,

    /// Output file (default: <name>.stego.mp4)
    #[arg(short = 'o')]
    pub output: Option<String>,

    /// Minimal output
    #[arg(long)]
    pub quiet: bool,

    /// Detailed output
    #[arg(long)]
    pub verbose: bool,

    /// Show progress bar
    #[arg(long)]
    pub progress: bool,

    /// JSON output
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: VideoEncodeArgs) -> Result<(), CliError> {
    let message = get_message(&args.message)?;
    let passphrase = get_passphrase(args.passphrase.as_deref(), "Passphrase", true)?;

    let output_path = match &args.output {
        Some(p) => PathBuf::from(p),
        None => {
            let stem = args.video
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "output".to_string());
            let parent = args.video.parent().unwrap_or(std::path::Path::new("."));
            parent.join(format!("{stem}.stego.mp4"))
        }
    };

    let progress_handle = if args.progress {
        Some(spawn_progress_bar())
    } else {
        None
    };

    let start = Instant::now();

    // Phase 4b: H.264 Baseline CAVLC only. Non-Baseline H.264 inputs
    // (Main/High profile, CABAC entropy) are auto-transcoded via an
    // ffmpeg shell-out. HEVC inputs are rejected with an ffmpeg suggestion
    // — HEVC support is archived behind the `hevc-archive` cargo feature
    // and will not re-ship in Phase 4.
    let file_bytes = std::fs::read(&args.video)?;
    let mut transcode_tempfile: Option<PathBuf> = None;
    match detect_video_codec(&file_bytes) {
        VideoCodec::H264 => {
            // Try encoding the file as-is; if the pipeline rejects it as
            // non-Baseline, transcode to Baseline CAVLC and retry once.
            match h264_ghost_encode(&file_bytes, &message, &passphrase) {
                Ok(stego) => {
                    std::fs::write(&output_path, &stego)?;
                }
                Err(e) if transcode::is_non_baseline_error(&e) => {
                    transcode::ensure_ffmpeg_available()?;
                    let temp = transcode::temp_transcode_path(&args.video);
                    eprintln!(
                        "Input is H.264 but not Baseline CAVLC — transcoding via ffmpeg..."
                    );
                    transcode::transcode_to_baseline(&args.video, &temp)?;
                    transcode_tempfile = Some(temp.clone());
                    let transcoded = std::fs::read(&temp)?;
                    let stego = h264_ghost_encode(&transcoded, &message, &passphrase)?;
                    std::fs::write(&output_path, &stego)?;
                }
                Err(e) => return Err(CliError::from(e)),
            }
        }
        VideoCodec::Hevc => {
            return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                format!(
                    "HEVC/H.265 video stego is archived. Transcode first:\n  {}",
                    transcode::TRANSCODE_SUGGESTION
                ),
            )));
        }
        VideoCodec::Unknown => {
            return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                "unsupported or unrecognised video codec (expected H.264 Baseline CAVLC)".into(),
            )));
        }
    }
    // Clean up the transcode tempfile, if we created one.
    if let Some(temp) = transcode_tempfile {
        let _ = std::fs::remove_file(&temp);
    }
    let elapsed = start.elapsed();

    if let Some(handle) = progress_handle {
        let _ = handle.join();
    }

    let out_mode = if args.json {
        OutputMode::Json
    } else if args.quiet {
        OutputMode::Quiet
    } else if args.verbose {
        OutputMode::Verbose
    } else {
        OutputMode::Default
    };

    let out_str = output_path.to_string_lossy().to_string();
    output::print_video_encode_result(&out_str, elapsed, out_mode);

    Ok(())
}

fn get_message(flag: &Option<String>) -> Result<String, CliError> {
    if let Some(m) = flag {
        return Ok(m.clone());
    }
    if !io::stdin().is_terminal() {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        let trimmed = buf.trim_end_matches('\n').trim_end_matches('\r');
        if trimmed.is_empty() {
            return Err(CliError::NoMessage);
        }
        return Ok(trimmed.to_string());
    }
    Err(CliError::NoMessage)
}
