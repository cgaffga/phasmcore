// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
#[cfg(feature = "h264-encoder")]
use crate::transcode;
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

    /// GOP size used by the streaming-session capacity
    /// estimate. Defaults to 30 (matches mobile encoder default at
    /// 30 fps = 1 IDR/sec).
    #[cfg(feature = "h264-encoder")]
    #[arg(long, default_value_t = 30)]
    pub gop_size: usize,
}

pub fn run(args: VideoCapacityArgs) -> Result<(), CliError> {
    // H.264 video capacity is reported by the streaming-session 4-domain
    // formula (the production OpenH264 encoder path). The legacy CAVLC
    // capacity scan was retired with the CAVLC stego subsystem (it also
    // reported wildly incorrect numbers — it counted Baseline-CAVLC
    // positions, not what the OH264 streaming-session encoder accepts).
    #[cfg(feature = "h264-encoder")]
    {
        return run_video_capacity(args);
    }

    // Without the OpenH264 backend there is no video-encode capacity to
    // report — the only H.264 surface in a decode-only build is the
    // walker. Tell the user to rebuild with the encoder feature.
    #[cfg(not(feature = "h264-encoder"))]
    {
        let _ = args;
        Err(CliError::from(phasm_core::StegoError::InvalidVideo(
            "H.264 video capacity requires the OpenH264 backend. \
             Rebuild from source: \
             `cargo install phasm-cli --features h264-encoder`."
                .into(),
        )))
    }
}

/// Streaming-session capacity branch.
///
/// Decodes the input MP4 to a temp YUV via ffmpeg (since the
/// capacity API works on raw pixels), runs the OH264 walker cover
/// count, and reports primary + shadow capacity. The decode + count
/// together take O(seconds) on 1080p × 10 frames; long clips are
/// proportionally slower.
#[cfg(feature = "h264-encoder")]
fn run_video_capacity(args: VideoCapacityArgs) -> Result<(), CliError> {
    transcode::ensure_ffmpeg_available()?;
    let probe = transcode::probe_video(&args.video)?;
    let yuv_path = transcode::temp_path_with_ext(&args.video, "yuv");
    transcode::decode_to_yuv(&args.video, &yuv_path)?;
    let yuv = std::fs::read(&yuv_path).map_err(|e| {
        CliError::InvalidArgs(format!("read decoded YUV {}: {e}", yuv_path.display()))
    })?;
    let _ = std::fs::remove_file(&yuv_path);

    let gop_size = args.gop_size.min(probe.n_frames).max(1);
    // OH264-accurate capacity, matching what
    // `StreamingEncodeSession` actually accepts.
    let info = {
        use phasm_core::h264_video_capacity;
        use phasm_core::codec::h264::openh264_stego::EncodeOpts;
        let opts = EncodeOpts { qp: 26, intra_period: gop_size as i32 };
        h264_video_capacity(
            &yuv, probe.width, probe.height, probe.n_frames, opts,
            /* full_tiers — CLI shows the per-tier breakdown */ true,
        )
        .map_err(CliError::from)?
    };

    if args.json {
        println!(
            r#"{{"engine":"video","width":{},"height":{},"n_frames":{},"gop_size":{},"cover_size_bits":{},"primary_max_message_bytes":{},"shadow_max_message_bytes":{},"per_tier_primary_max_message_bytes":[{},{},{},{},{}]}}"#,
            probe.width, probe.height, probe.n_frames, gop_size,
            info.cover_size_bits,
            info.primary_max_message_bytes,
            info.shadow_max_message_bytes,
            info.per_tier_primary_max_message_bytes[0],
            info.per_tier_primary_max_message_bytes[1],
            info.per_tier_primary_max_message_bytes[2],
            info.per_tier_primary_max_message_bytes[3],
            info.per_tier_primary_max_message_bytes[4],
        );
    } else {
        println!("CABAC v2 capacity ({}x{}, {} frames, gop={}):",
            probe.width, probe.height, probe.n_frames, gop_size);
        println!("  cover bits (3 injectable domains): {}", info.cover_size_bits);
        println!("  primary max message bytes:         {}", info.primary_max_message_bytes);
        println!("  shadow max message bytes (n=1):    {}", info.shadow_max_message_bytes);
        println!();
        println!("  D'.2 per-tier primary capacity (cascade-safety quality):");
        let tier_names = ["Max Capacity", "Balanced", "Quality", "High Quality", "Best Quality"];
        for (tier, name) in tier_names.iter().enumerate() {
            println!("    Tier {} ({:14}): {:>10} bytes",
                tier, name, info.per_tier_primary_max_message_bytes[tier]);
        }
    }
    Ok(())
}
