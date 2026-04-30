// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::output::{self, CliVideoCapacityInfo};
use crate::transcode;
use phasm_core::{detect_video_codec, h264_ghost_capacity, h264_ghost_capacity_max, VideoCodec};
use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
pub struct VideoCapacityArgs {
    /// MP4 video file
    pub video: PathBuf,

    /// JSON output
    #[arg(long)]
    pub json: bool,

    /// Report the maximum capacity (STC `w = 1`, less stealth) instead
    /// of the conservative default (STC `w = 5`, default stealth).
    /// Useful for power users who want the absolute upper bound on
    /// what the encoder can fit. The default value is what the mobile
    /// apps display.
    #[arg(long)]
    pub max: bool,
}

pub fn run(args: VideoCapacityArgs) -> Result<(), CliError> {
    let mp4_bytes = std::fs::read(&args.video)?;

    // Phase 4b: H.264 Baseline CAVLC only. Non-Baseline H.264 inputs
    // (Main/High profile, CABAC) are auto-transcoded to Baseline via
    // ffmpeg and the capacity is measured on the transcoded copy —
    // because the capacity of the eventual stego file depends on what
    // Baseline positions land after re-encoding.
    let mut transcode_tempfile: Option<std::path::PathBuf> = None;
    let cap_fn: fn(&[u8]) -> Result<usize, phasm_core::StegoError> = if args.max {
        h264_ghost_capacity_max
    } else {
        h264_ghost_capacity
    };
    let info = match detect_video_codec(&mp4_bytes) {
        VideoCodec::H264 => {
            let cap = match cap_fn(&mp4_bytes) {
                Ok(c) => c,
                Err(e) if transcode::is_non_baseline_error(&e) => {
                    transcode::ensure_ffmpeg_available()?;
                    let temp = transcode::temp_transcode_path(&args.video);
                    eprintln!(
                        "Input is H.264 but not Baseline CAVLC — transcoding via ffmpeg..."
                    );
                    transcode::transcode_to_baseline(&args.video, &temp)?;
                    transcode_tempfile = Some(temp.clone());
                    let transcoded = std::fs::read(&temp)?;
                    cap_fn(&transcoded)?
                }
                Err(e) => return Err(CliError::from(e)),
            };
            CliVideoCapacityInfo {
                total_positions: 0,
                num_i_frames: 0,
                capacity_bytes: cap,
                frame_positions: Vec::new(),
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
    };
    if let Some(temp) = transcode_tempfile {
        let _ = std::fs::remove_file(&temp);
    }

    output::print_video_capacity(&info, args.json);

    Ok(())
}
