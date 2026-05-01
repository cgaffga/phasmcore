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
    /// apps display. Ignored when `--cabac-v2` is set.
    #[arg(long)]
    pub max: bool,

    /// Task #96 — Use the CABAC v2 4-domain capacity formula instead
    /// of the legacy CAVLC `h264_ghost_capacity` path. Requires the
    /// `cabac-stego` build feature. Reports both primary and shadow
    /// (single-shadow) capacities. Decodes the input to YUV first
    /// via ffmpeg (slow on long clips) since the CABAC v2 pipeline
    /// works on raw pixels not parsed Annex-B.
    #[cfg(feature = "cabac-stego")]
    #[arg(long)]
    pub cabac_v2: bool,

    /// Task #96 + #98 — GOP size used by the CABAC v2 capacity
    /// estimate. Defaults to 30 (matches mobile encoder default at
    /// 30 fps = 1 IDR/sec). Only honored under `--cabac-v2`.
    #[cfg(feature = "cabac-stego")]
    #[arg(long, default_value_t = 30)]
    pub gop_size: usize,
}

pub fn run(args: VideoCapacityArgs) -> Result<(), CliError> {
    // Task #96 — CABAC v2 capacity branch.
    #[cfg(feature = "cabac-stego")]
    if args.cabac_v2 {
        return run_cabac_v2(args);
    }

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

/// Task #96 — CABAC v2 capacity branch.
///
/// Decodes the input MP4 to a temp YUV via ffmpeg (since CABAC v2's
/// capacity API works on raw pixels), runs Pass-1 cover counting,
/// and reports primary + shadow capacity. The decode + Pass 1
/// together take O(seconds) on 1080p × 10 frames; long clips are
/// proportionally slower.
#[cfg(feature = "cabac-stego")]
fn run_cabac_v2(args: VideoCapacityArgs) -> Result<(), CliError> {
    use phasm_core::h264_stego_capacity_4domain;

    transcode::ensure_ffmpeg_available()?;
    let probe = transcode::probe_video(&args.video)?;
    let yuv_path = transcode::temp_path_with_ext(&args.video, "yuv");
    transcode::decode_to_yuv(&args.video, &yuv_path)?;
    let yuv = std::fs::read(&yuv_path).map_err(|e| {
        CliError::InvalidArgs(format!("read decoded YUV {}: {e}", yuv_path.display()))
    })?;
    let _ = std::fs::remove_file(&yuv_path);

    let gop_size = args.gop_size.min(probe.n_frames).max(1);
    let info = h264_stego_capacity_4domain(
        &yuv, probe.width, probe.height, probe.n_frames, gop_size,
    )
    .map_err(CliError::from)?;

    if args.json {
        println!(
            r#"{{"engine":"cabac-v2","width":{},"height":{},"n_frames":{},"gop_size":{},"cover_size_bits":{},"primary_max_message_bytes":{},"shadow_max_message_bytes":{}}}"#,
            probe.width, probe.height, probe.n_frames, gop_size,
            info.cover_size_bits,
            info.primary_max_message_bytes,
            info.shadow_max_message_bytes,
        );
    } else {
        println!("CABAC v2 capacity ({}x{}, {} frames, gop={}):",
            probe.width, probe.height, probe.n_frames, gop_size);
        println!("  cover bits (3 injectable domains): {}", info.cover_size_bits);
        println!("  primary max message bytes:         {}", info.primary_max_message_bytes);
        println!("  shadow max message bytes (n=1):    {}", info.shadow_max_message_bytes);
    }
    Ok(())
}
