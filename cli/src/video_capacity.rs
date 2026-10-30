// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::transcode;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum CodecChoice {
    #[cfg(feature = "h264-encoder")]
    H264,
    #[cfg(feature = "av1-encoder")]
    Av1,
}

impl CodecChoice {
    /// H.264 when compiled in (backwards-compatible default), else AV1.
    const fn default_choice() -> Self {
        #[cfg(feature = "h264-encoder")]
        { Self::H264 }
        #[cfg(not(feature = "h264-encoder"))]
        { Self::Av1 }
    }
}

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
    /// apps display. H.264 only — AV1 capacity is single-tier.
    #[arg(long)]
    pub max: bool,

    /// Codec to estimate capacity for. Defaults to H.264 for
    /// backwards compatibility with `phasm video-capacity` invocations
    /// from before AV1 landed.
    #[arg(long, value_enum, default_value_t = CodecChoice::default_choice())]
    pub codec: CodecChoice,

    /// GOP size used by the streaming-session capacity
    /// estimate. Defaults to 30 (matches mobile encoder default at
    /// 30 fps = 1 IDR/sec). Applies to both H.264 and AV1.
    #[arg(long, default_value_t = 30)]
    pub gop_size: usize,

    /// Output resolution preset of the encode to size for (`original`,
    /// `sd`, `hd`, `fhd` — see `phasm video-encode --resolution`). Capacity
    /// is measured on the scaled frames, so it only holds for an encode at
    /// the same preset.
    #[arg(long, value_enum, default_value_t = crate::transcode::Resolution::Original)]
    pub resolution: crate::transcode::Resolution,
}

pub fn run(args: VideoCapacityArgs) -> Result<(), CliError> {
    match args.codec {
        #[cfg(feature = "h264-encoder")]
        CodecChoice::H264 => run_h264_video_capacity(args),
        #[cfg(feature = "av1-encoder")]
        CodecChoice::Av1 => run_av1_video_capacity(args),
    }
}

/// H.264 streaming-session capacity branch.
///
/// Decodes the input MP4 to a temp YUV via ffmpeg (since the
/// capacity API works on raw pixels), runs the OH264 walker cover
/// count, and reports primary + shadow capacity. The decode + count
/// together take O(seconds) on 1080p × 10 frames; long clips are
/// proportionally slower.
#[cfg(feature = "h264-encoder")]
fn run_h264_video_capacity(args: VideoCapacityArgs) -> Result<(), CliError> {
    transcode::ensure_ffmpeg_available()?;
    let mut probe = transcode::probe_video(&args.video)?;
    let yuv_path = transcode::temp_path_with_ext(&args.video, "yuv");
    // Same size rule as encode (exact even size; an odd side loses one px),
    // so capacity is measured on exactly the frames encode will embed into.
    transcode::decode_to_yuv_aligned(&args.video, &yuv_path, &mut probe, transcode::H264_SIZE_ALIGN)?;
    // Same `--resolution` scaling as the encode.
    if let Err(e) = transcode::scale_yuv_to_preset(&yuv_path, &mut probe, args.resolution) {
        let _ = std::fs::remove_file(&yuv_path);
        return Err(e);
    }
    let yuv = std::fs::read(&yuv_path).map_err(|e| {
        CliError::InvalidArgs(format!("read decoded YUV {}: {e}", yuv_path.display()))
    })?;
    let _ = std::fs::remove_file(&yuv_path);

    // Container `nb_frames` is unreliable (CarPlane: meta=193 vs decoded=194);
    // the decoded buffer is authoritative. Derive the real frame count from it.
    let frame_size = (probe.width as usize) * (probe.height as usize) * 3 / 2;
    if frame_size == 0 || yuv.len() % frame_size != 0 {
        return Err(CliError::InvalidArgs(format!(
            "decoded YUV {} bytes is not a whole number of {}x{} frames",
            yuv.len(), probe.width, probe.height,
        )));
    }
    probe.n_frames = yuv.len() / frame_size;

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
        // `primary_max_message_bytes` keeps its meaning (the STC ceiling) for
        // compatibility; `max_message_bytes` / `comfort_message_bytes` are the
        // apps' bands, `achievable_max_message_bytes` the planner's accept limit.
        println!(
            r#"{{"engine":"video","codec":"h264","width":{},"height":{},"n_frames":{},"gop_size":{},"cover_size_bits":{},"comfort_message_bytes":{},"max_message_bytes":{},"achievable_max_message_bytes":{},"primary_max_message_bytes":{},"shadow_max_message_bytes":{},"per_tier_primary_max_message_bytes":[{},{},{},{},{}]}}"#,
            probe.width, probe.height, probe.n_frames, gop_size,
            info.cover_size_bits,
            info.comfort_message_bytes,
            info.max_message_bytes,
            info.achievable_max_message_bytes,
            info.primary_max_message_bytes,
            info.shadow_max_message_bytes,
            info.per_tier_primary_max_message_bytes[0],
            info.per_tier_primary_max_message_bytes[1],
            info.per_tier_primary_max_message_bytes[2],
            info.per_tier_primary_max_message_bytes[3],
            info.per_tier_primary_max_message_bytes[4],
        );
    } else {
        println!("H.264 capacity ({}x{}, {} frames, gop={}):",
            probe.width, probe.height, probe.n_frames, gop_size);
        // The figures the apps show. The CLI does not enforce them: a larger
        // payload still encodes (with a warning) at the cost of visible damage.
        println!("  comfort message bytes:             {}", info.comfort_message_bytes);
        println!("  max message bytes (red end):       {}", info.max_message_bytes);
        println!("  technical ceiling (STC limit):     {}   (visible damage above the max)",
            info.primary_max_message_bytes);
        println!("  shadow max message bytes (n=1):    {}", info.shadow_max_message_bytes);
        println!();
        // CSB-only, and a raw position count (infinity-cost positions included) —
        // NOT a capacity and NOT the 3-domain shadow pool, which the old
        // label here wrongly claimed. See `H264StegoCapacityInfo`.
        println!("  cover bits (CoeffSign domain):     {}", info.cover_size_bits);
        println!("  planner accept limit:              {}", info.achievable_max_message_bytes);
        println!();
        println!("  D'.2 per-tier primary capacity (cascade-safety quality, technical ceiling):");
        let tier_names = ["Max Capacity", "Balanced", "Quality", "High Quality", "Best Quality"];
        for (tier, name) in tier_names.iter().enumerate() {
            println!("    Tier {} ({:14}): {:>10} bytes",
                tier, name, info.per_tier_primary_max_message_bytes[tier]);
        }
    }
    Ok(())
}

/// AV1 capacity branch: ffmpeg → temp YUV → `av1_capacity`, which probes
/// every GOP of the clip at the encode's GOP size and quantizer and reports
/// what the apps show: the v2 cover after the damage-aware per-GOP caps, the
/// maximum (red end) and the comfort figure (start of the yellow band). The
/// apps sample a few GOPs and show a conservative estimate of the same
/// figures, so the CLI's can be a little higher. AV1 has no per-tier
/// breakdown; `--max` has no effect here and is reported as such.
#[cfg(feature = "av1-encoder")]
fn run_av1_video_capacity(args: VideoCapacityArgs) -> Result<(), CliError> {
    transcode::ensure_ffmpeg_available()?;
    let mut probe = transcode::probe_video(&args.video)?;
    let yuv_path = transcode::temp_path_with_ext(&args.video, "yuv");
    // CROP an odd source to even, exactly as the encode does.
    transcode::decode_to_yuv_aligned(&args.video, &yuv_path, &mut probe, transcode::AV1_SIZE_ALIGN)?;
    // Same `--resolution` scaling as the encode.
    if let Err(e) = transcode::scale_yuv_to_preset(&yuv_path, &mut probe, args.resolution) {
        let _ = std::fs::remove_file(&yuv_path);
        return Err(e);
    }
    let yuv = std::fs::read(&yuv_path).map_err(|e| {
        CliError::InvalidArgs(format!("read decoded YUV {}: {e}", yuv_path.display()))
    })?;
    let _ = std::fs::remove_file(&yuv_path);

    // Mirror the H.264 branch's container-vs-decoded reconciliation —
    // `decode_to_yuv` is authoritative for `n_frames`.
    let frame_size = (probe.width as usize) * (probe.height as usize) * 3 / 2;
    if frame_size == 0 || yuv.len() % frame_size != 0 {
        return Err(CliError::InvalidArgs(format!(
            "decoded YUV {} bytes is not a whole number of {}x{} frames",
            yuv.len(), probe.width, probe.height,
        )));
    }
    probe.n_frames = yuv.len() / frame_size;

    let gop_size = args.gop_size.min(probe.n_frames).max(1);

    let info = {
        use phasm_core::{av1_capacity, Av1StreamingEncodeParams};
        // Probe the encode's shape: its GOP size (`--gop-size`) and its
        // quantizer. A gop_size of 1 makes every frame a key frame, which
        // carries about twice the cover of the P-frames the encode writes,
        // and a lower quantizer adds cover too; either over-reports what
        // fits.
        let params = Av1StreamingEncodeParams {
            width: probe.width,
            height: probe.height,
            quantizer: crate::video_encode::av1_quantizer(),
            gop_size: gop_size as u32,
            total_frames_hint: probe.n_frames as u32,
            colour: Default::default(),
        };
        av1_capacity(&yuv, params).map_err(|e| {
            CliError::from(phasm_core::StegoError::InvalidVideo(format!(
                "AV1 capacity probe failed: {e:?}"
            )))
        })?
    };

    if args.max {
        eprintln!("note: --max has no effect on AV1 (single-tier capacity); ignored.");
    }

    if args.json {
        println!(
            r#"{{"engine":"video","codec":"av1","width":{},"height":{},"n_frames":{},"gop_size":{},"cover_size_bits":{},"primary_max_message_bytes":{},"comfort_message_bytes":{},"shadow_max_message_bytes_n1":{}}}"#,
            probe.width, probe.height, probe.n_frames, gop_size,
            info.cover_size_bits,
            info.primary_max_message_bytes,
            info.comfort_message_bytes,
            info.shadow_max_message_bytes_n1,
        );
    } else {
        println!("AV1 capacity ({}x{}, {} frames, encode gop={}):",
            probe.width, probe.height, probe.n_frames, gop_size);
        println!("  cover positions (damage-aware):   {}", info.cover_size_bits);
        println!("  max message bytes:                {}", info.primary_max_message_bytes);
        println!("  comfort message bytes:            {}", info.comfort_message_bytes);
        println!("  shadow max message bytes (n=1):   {}", info.shadow_max_message_bytes_n1);
    }
    Ok(())
}
