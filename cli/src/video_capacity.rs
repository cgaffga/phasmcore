// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
#[cfg(any(feature = "h264-encoder", feature = "av1-encoder"))]
use crate::transcode;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum CodecChoice {
    H264,
    Av1,
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
    #[arg(long, value_enum, default_value_t = CodecChoice::H264)]
    pub codec: CodecChoice,

    /// GOP size used by the streaming-session capacity
    /// estimate. Defaults to 30 (matches mobile encoder default at
    /// 30 fps = 1 IDR/sec). Applies to both H.264 and AV1.
    #[arg(long, default_value_t = 30)]
    pub gop_size: usize,
}

pub fn run(args: VideoCapacityArgs) -> Result<(), CliError> {
    match args.codec {
        CodecChoice::H264 => run_h264(args),
        CodecChoice::Av1 => run_av1(args),
    }
}

fn run_h264(args: VideoCapacityArgs) -> Result<(), CliError> {
    // H.264 video capacity is reported by the streaming-session 4-domain
    // formula (the production OpenH264 encoder path). The legacy CAVLC
    // capacity scan was retired with the CAVLC stego subsystem (it also
    // reported wildly incorrect numbers — it counted Baseline-CAVLC
    // positions, not what the OH264 streaming-session encoder accepts).
    #[cfg(feature = "h264-encoder")]
    {
        return run_h264_video_capacity(args);
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

fn run_av1(args: VideoCapacityArgs) -> Result<(), CliError> {
    #[cfg(feature = "av1-encoder")]
    {
        return run_av1_video_capacity(args);
    }
    #[cfg(not(feature = "av1-encoder"))]
    {
        let _ = args;
        Err(CliError::from(phasm_core::StegoError::InvalidVideo(
            "AV1 video capacity requires the AV1 backend. \
             Rebuild from source: \
             `cargo install phasm-cli --features av1-encoder`."
                .into(),
        )))
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
    // CROP any non-16-aligned source so capacity matches what encode accepts.
    transcode::decode_to_yuv_aligned(&args.video, &yuv_path, &mut probe, 16)?;
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
        println!(
            r#"{{"engine":"video","codec":"h264","width":{},"height":{},"n_frames":{},"gop_size":{},"cover_size_bits":{},"primary_max_message_bytes":{},"shadow_max_message_bytes":{},"per_tier_primary_max_message_bytes":[{},{},{},{},{}]}}"#,
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

/// AV1 streaming-session capacity branch (v0.6 closed-form).
///
/// Mirrors `run_h264_video_capacity`: ffmpeg → temp YUV →
/// `Av1StreamingProbeSession`. The probe runs at `gop_size=1`
/// (per session-API contract — each frame is its own GOP for
/// cover counting), then closed-form math projects the conservative
/// primary + shadow caps via the 0.15 STC ratio. There is no per-tier
/// breakdown on AV1 v0.6 (single joint Tier 1
/// `AC_COEFF_SIGN + GOLOMB_TAIL_LSB`); the `--max` flag has no effect
/// here and is reported as such.
///
/// The `--gop-size` flag scales the per-GOP overhead deduction: the
/// real encode uses `gop_size` (default 30, matching mobile), so
/// the closed-form math reflects that GOP count. Capacity itself is
/// per-frame-density × frame-count, independent of how those frames
/// are grouped — the deduction is the chunk_frame v3 wire overhead.
#[cfg(feature = "av1-encoder")]
fn run_av1_video_capacity(args: VideoCapacityArgs) -> Result<(), CliError> {
    transcode::ensure_ffmpeg_available()?;
    let mut probe = transcode::probe_video(&args.video)?;
    let yuv_path = transcode::temp_path_with_ext(&args.video, "yuv");
    // CROP any non-8-aligned source so capacity matches what encode accepts.
    transcode::decode_to_yuv_aligned(&args.video, &yuv_path, &mut probe, 8)?;
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
        // The AV1 probe API requires gop_size=1 (D-phase contract:
        // each frame is its own GOP for cover counting). The user-
        // facing `--gop-size` is the encode GOP size; the closed-form
        // capacity formula doesn't depend on the encode GOP — only on
        // total cover bits — so we hand 1 to the params struct.
        // `quantizer=80` is the rav1e default-quality target that
        // `Av1StreamingEncodeParams::default()` documents (CLI uses
        // the same default as mobile; matches the encode params
        // selected by `run_av1_encode`).
        let params = Av1StreamingEncodeParams {
            width: probe.width,
            height: probe.height,
            quantizer: 80,
            gop_size: 1,
            total_frames_hint: probe.n_frames as u32,
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
            r#"{{"engine":"video","codec":"av1","width":{},"height":{},"n_frames":{},"gop_size":{},"cover_size_bits":{},"primary_max_message_bytes":{},"shadow_max_message_bytes_n1":{},"ac_sign_bits":{},"golomb_tail_bits":{}}}"#,
            probe.width, probe.height, probe.n_frames, gop_size,
            info.cover_size_bits,
            info.primary_max_message_bytes,
            info.shadow_max_message_bytes_n1,
            info.per_domain_bits.ac_sign,
            info.per_domain_bits.golomb_tail,
        );
    } else {
        println!("AV1 streaming-session capacity ({}x{}, {} frames, encode gop={}):",
            probe.width, probe.height, probe.n_frames, gop_size);
        println!("  cover bits (Tier 1 joint domain): {}", info.cover_size_bits);
        println!("    AC_COEFF_SIGN:    {} bits", info.per_domain_bits.ac_sign);
        println!("    GOLOMB_TAIL_LSB:  {} bits", info.per_domain_bits.golomb_tail);
        println!("  primary max message bytes:        {}", info.primary_max_message_bytes);
        println!("  shadow max message bytes (n=1):   {}", info.shadow_max_message_bytes_n1);
    }
    Ok(())
}
