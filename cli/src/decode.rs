// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::output::{self, OutputMode};
use crate::passphrase::get_passphrase;
use crate::progress::spawn_progress_bar;
use phasm_core::{
    detect_video_codec, h264_ghost_decode, is_mp4, smart_decode, DecodeQuality, VideoCodec,
};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

#[derive(Parser)]
pub struct DecodeArgs {
    /// Stego image or video file
    pub image: PathBuf,

    /// Passphrase (interactive prompt if omitted)
    #[arg(short = 'p')]
    pub passphrase: Option<String>,

    /// Extract file attachments to directory
    #[arg(long)]
    pub extract: Option<PathBuf>,

    /// Minimal output (message text only)
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

pub fn run(args: DecodeArgs) -> Result<(), CliError> {
    let file_bytes = std::fs::read(&args.image)?;
    let passphrase = get_passphrase(args.passphrase.as_deref(), "Passphrase", false)?;

    let progress_handle = if args.progress {
        Some(spawn_progress_bar())
    } else {
        None
    };

    let start = Instant::now();

    // Auto-detect format: H.264 MP4 vs JPEG image. HEVC is archived behind
    // `hevc-archive`. Phase 4b note: we deliberately do NOT auto-transcode
    // stego videos on decode — transcoding re-encodes pixels which destroys
    // the embedded message. The stego must already be Baseline CAVLC
    // (which it would be, since it was produced by `phasm video-encode`).
    let (payload, quality) = if is_mp4(&file_bytes) {
        let payload = match detect_video_codec(&file_bytes) {
            VideoCodec::H264 => h264_ghost_decode(&file_bytes, &passphrase)?,
            VideoCodec::Hevc => {
                return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                    "HEVC/H.265 decode is not supported. This file was encoded outside \
                     Phasm's H.264 pipeline — no hidden message will be recoverable."
                        .into(),
                )));
            }
            VideoCodec::Unknown => {
                return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                    "unsupported or unrecognised video codec (expected H.264 Baseline CAVLC)"
                        .into(),
                )));
            }
        };
        (payload, DecodeQuality::ghost())
    } else {
        smart_decode(&file_bytes, &passphrase)?
    };

    let elapsed = start.elapsed();

    if let Some(handle) = progress_handle {
        let _ = handle.join();
    }

    // Extract files if requested
    let extract_dir = if let Some(ref dir) = args.extract {
        if !payload.files.is_empty() {
            std::fs::create_dir_all(dir)?;
            for f in &payload.files {
                let out_path = dir.join(&f.filename);
                std::fs::write(&out_path, &f.content)?;
            }
        }
        Some(dir.to_string_lossy().to_string())
    } else {
        None
    };

    let out_mode = if args.json {
        OutputMode::Json
    } else if args.quiet {
        OutputMode::Quiet
    } else if args.verbose {
        OutputMode::Verbose
    } else {
        OutputMode::Default
    };

    output::print_decode_result(&payload, &quality, extract_dir.as_deref(), elapsed, out_mode);

    Ok(())
}
