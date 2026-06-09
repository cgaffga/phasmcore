// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::output::{self, OutputMode};
use crate::passphrase::get_passphrase;
use crate::progress::spawn_progress_bar;
#[cfg(feature = "h264-encoder")]
use crate::transcode;
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

/// Resolve + report the Auto tier using the OH264 walker, so the
/// CLI can show "Cascade tier: High Quality (auto)" before encoding.
/// Falls back silently if the resolve probe fails (encode will then
/// resolve internally with the same logic).
#[cfg(feature = "h264-encoder")]
fn report_resolved_auto_tier(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    message_bytes: usize,
    headroom: f32,
) {
    use phasm_core::{h264_resolve_auto_tier, CascadeTier};
    use phasm_core::codec::h264::openh264_stego::EncodeOpts;
    let opts = EncodeOpts { qp: 26, intra_period: gop_size as i32 };
    // Framed message size ≈ message + ~64 (crypto envelope) + ~4 (chunk).
    let estimated_framed_bytes = message_bytes.saturating_add(70);
    match h264_resolve_auto_tier(
        yuv, width, height, n_frames, opts, estimated_framed_bytes, headroom,
    ) {
        Ok(tier) if tier != CascadeTier::Auto => {
            eprintln!("Cascade tier: {} (auto, payload {} B)",
                tier.ui_name(), message_bytes);
        }
        _ => {}
    }
}

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

    /// VID-OPT (2026-05-24) — texture-adaptive cover optimizer toggle.
    /// Wiring only in this commit: the flag is threaded to the encoder
    /// dispatch site but the actual optimizer pass for video lands in
    /// a follow-on. Mirrors the image-side `--optimize` flag in
    /// `encode.rs` (line 81-83).
    #[arg(long)]
    pub optimize: bool,
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

    // Encoder routing.
    //
    // The OpenH264 path uses ffmpeg to demux + decode the source to
    // raw YUV before the encode pipeline ever sees it, so it accepts
    // any ffmpeg-decodable codec (H.264, HEVC, ProRes, VP9, AV1, …).
    // The legacy CAVLC bitstream-mod encode path was retired with the
    // CAVLC stego subsystem; video stego encode now requires the
    // OpenH264 backend feature.
    #[cfg(feature = "h264-encoder")]
    {
        run_oh264_encode(&args.video, &message, &passphrase, &output_path, args.optimize)?;
    }
    // Without the OpenH264 backend there is no video-encode path — the
    // legacy CAVLC bitstream-mod encoder was retired. Bail with a clean
    // rebuild instruction (the early return makes the result-reporting
    // tail below openh264-only).
    #[cfg(not(feature = "h264-encoder"))]
    {
        let _ = (&message, &passphrase, &output_path, &start, progress_handle);
        return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
            "H.264 video stego encode requires the OpenH264 backend. \
             Rebuild from source: \
             `cargo install phasm-cli --features h264-encoder`."
                .into(),
        )));
    }

    #[cfg(feature = "h264-encoder")]
    {
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
}

/// Production OpenH264-backend stego encode path.
///
/// MP4 → ffmpeg → YUV → `StreamingEncodeSession` (OpenH264 backend,
/// visual-reconstruction stage) → Annex-B → in-tree HandBrake mux → MP4.
///
/// Uses ffmpeg only for demux + decode (`transcode::probe_video`,
/// `transcode::decode_to_yuv`); the encode step routes through
/// `phasm_core::codec::h264::openh264_stego`, the sole H.264 encoder
/// since the pure-Rust encoder retirement (see
/// `docs/design/video/h264/phase-c8-visual-recon-plan.md` §C.8.14).
///
/// v1.0 limitations:
///  - no file attachments yet on this CLI path; shadows + attachments
///    are a follow-on.
///  - mux uses the in-tree HandBrake/x264 profile
///    (`MuxProfile::HandbrakeX264`); see the mux-site comment for why the
///    earlier ffmpeg `-c copy` shell-out was dropped (2026-05-24).
#[cfg(feature = "h264-encoder")]
fn run_oh264_encode(
    input: &PathBuf,
    message: &str,
    passphrase: &str,
    output: &PathBuf,
    // VID-OPT (2026-05-24) — wiring only. Reaches the dispatch site but
    // not yet plumbed into EncodeSessionParams (the eventual home for
    // the optimizer toggle). Functional optimizer pass for video lands
    // as a follow-on.
    optimize: bool,
) -> Result<(), CliError> {
    let _ = optimize; // TODO: thread into EncodeSessionParams when video optimizer ships
    use phasm_core::{
        ColorParams, CostWeights, EncodeEngineChoice, EncodeSessionParams,
        StreamingEncodeSession, StreamingProbeSession, YuvFrameRef,
    };

    transcode::ensure_ffmpeg_available()?;
    let probe = transcode::probe_video(input)?;
    eprintln!(
        "Encoding via OpenH264 (streaming, production v1.0): {}x{} × {} frames @ {} fps{}",
        probe.width, probe.height, probe.n_frames, probe.frame_rate,
        if probe.has_audio { " (with audio)" } else { "" },
    );

    let yuv_temp = transcode::temp_path_with_ext(input, "yuv");
    let annex_b_temp = transcode::temp_path_with_ext(input, "h264");
    let mut cleanup_paths = vec![yuv_temp.clone(), annex_b_temp.clone()];

    let result = (|| -> Result<(), CliError> {
        // ffmpeg writes the whole decoded YUV to a temp file. Memory-
        // optimal streaming via stdin-pipe is tracked as a v1.1 polish;
        // for v1.0 we mmap-read the YUV file once and iterate per-frame
        // slices into the streaming session, which keeps the encoder's
        // own working set bounded to O(gop_size × frame).
        transcode::decode_to_yuv(input, &yuv_temp)?;
        let yuv_bytes = std::fs::read(&yuv_temp)?;
        let expected =
            (probe.width as usize) * (probe.height as usize) * 3 / 2 * probe.n_frames;
        if yuv_bytes.len() != expected {
            return Err(CliError::InvalidArgs(format!(
                "ffmpeg-decoded YUV is {} bytes; expected {} ({}x{} × 1.5 × {})",
                yuv_bytes.len(), expected, probe.width, probe.height, probe.n_frames,
            )));
        }

        // Parse fps string ("30/1", "30000/1001") into rational
        // numerator/denominator for the SPS time_scale fields.
        let (fps_num, fps_den) = transcode::parse_frame_rate(&probe.frame_rate);

        // Per-GOP STC isolation. Match mobile's DEFAULT_GOP=30 so
        // CLI- and mobile-encoded files have identical structure.
        let qp: i32 = std::env::var("PHASM_DEBUG_QP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(26);
        const GOP_SIZE: u32 = 30;

        // Report the resolved auto-tier (mirrors the mobile success-
        // screen "Quality mode: High Quality (auto)" UX). The encoder
        // resolves the same tier internally; this is purely the
        // user-facing announcement before the (slower) encode runs.
        report_resolved_auto_tier(
            &yuv_bytes, probe.width, probe.height, probe.n_frames,
            GOP_SIZE as usize, message.len(),
            phasm_core::CASCADE_DEFAULT_HEADROOM,
        );

        let params = EncodeSessionParams {
            width: probe.width,
            height: probe.height,
            fps_num,
            fps_den,
            qp,
            gop_size: GOP_SIZE,
            total_frames_hint: probe.n_frames as u32,
            // Default BT.709 limited; CLI doesn't probe VUI from ffprobe
            // yet — v1.1 polish to add `color_primaries` / `transfer` /
            // `matrix` / `range` parsing from ffprobe output.
            color: ColorParams::default(),
            engine: EncodeEngineChoice::Oh264,
            cost_weights: match std::env::var("PHASM_DEBUG_DOMAIN").as_deref() {
                Ok("mvd_sign") => CostWeights::debug_mvd_sign_only(),
                Ok("mvd_suffix") => CostWeights::debug_mvd_suffix_only(),
                Ok("coeff_sign") => CostWeights::debug_coeff_sign_only(),
                Ok("coeff_suffix") => CostWeights::debug_coeff_suffix_only(),
                Ok("cs_csl") => CostWeights::conservative_cs_csl_only(),
                Ok("mvd_pair") => CostWeights::debug_mvd_pair_only(),
                Ok("bias_coeff") => CostWeights::debug_bias_coeff_pair(),
                Ok("bias_mvd") => CostWeights::debug_bias_mvd_pair(),
                Ok("bias_csb") => CostWeights::debug_bias_coeff_sign(),
                Ok("bias_csl") => CostWeights::debug_bias_coeff_suffix(),
                Ok("bias_msb") => CostWeights::debug_bias_mvd_sign(),
                Ok("bias_msl") => CostWeights::debug_bias_mvd_suffix(),
                _ => CostWeights::default(),
            },
                progress_callback: None,
        };
        let frame_bytes = (probe.width as usize) * (probe.height as usize) * 3 / 2;
        let chroma_w = (probe.width as usize) / 2;
        let chroma_h = (probe.height as usize) / 2;
        let y_plane_size = (probe.width as usize) * (probe.height as usize);
        let chroma_plane_size = chroma_w * chroma_h;

        // Proportional allocation (CLI two-pass). `yuv_bytes` holds
        // the whole clip (re-readable), so run a capacity probe FIRST to learn
        // each GOP's cap, then size each GOP's chunk PROPORTIONAL to its cap
        // (uniform flip density — stealthier than the default online
        // uniform-BYTES spread, which over-fills low-capacity GOPs into
        // detectability hotspots). The per-GOP round-trip verify + shrink-carry
        // inside the encode is the backstop, so an over-optimistic plan degrades
        // to carry-forward (never data loss). This doubles encode wall-clock;
        // an easy-fit fast path (skip the probe for tiny payloads) is a
        // perf follow-on.
        // Concentrate+tail at the shared stealth-leaning default
        // (0.5, provisional — pending stealth-corpus calibration). Supersedes the 1.0
        // pure-proportional spread, whose r_target was inert.
        use phasm_core::codec::h264::stego::chunk_frame::CAP2_DEFAULT_R_TARGET;
        let per_gop_caps: Vec<usize> = {
            let mut probe_session = StreamingProbeSession::create(params.clone())?;
            for frame_idx in 0..probe.n_frames {
                let frame_off = frame_idx * frame_bytes;
                let y_start = frame_off;
                let y_end = y_start + y_plane_size;
                let u_end = y_end + chroma_plane_size;
                let v_end = u_end + chroma_plane_size;
                probe_session.push_frame(YuvFrameRef {
                    y: &yuv_bytes[y_start..y_end],
                    y_stride: probe.width as usize,
                    u: &yuv_bytes[y_end..u_end],
                    u_stride: chroma_w,
                    v: &yuv_bytes[u_end..v_end],
                    v_stride: chroma_w,
                })?;
            }
            let (_probe_result, caps) = probe_session.finish_with_per_gop()?;
            caps
        };

        let mut session = StreamingEncodeSession::create(params, message, passphrase)?;
        if session.plan_proportional(&per_gop_caps, CAP2_DEFAULT_R_TARGET)? {
            eprintln!(
                "  concentrate+tail across {} GOPs (r_target={})",
                per_gop_caps.len(),
                CAP2_DEFAULT_R_TARGET
            );
        }

        let mut annex_b: Vec<u8> = Vec::with_capacity(yuv_bytes.len() / 8);
        for frame_idx in 0..probe.n_frames {
            let frame_off = frame_idx * frame_bytes;
            let y_start = frame_off;
            let y_end = y_start + y_plane_size;
            let u_end = y_end + chroma_plane_size;
            let v_end = u_end + chroma_plane_size;
            let frame = YuvFrameRef {
                y: &yuv_bytes[y_start..y_end],
                y_stride: probe.width as usize,
                u: &yuv_bytes[y_end..u_end],
                u_stride: chroma_w,
                v: &yuv_bytes[u_end..v_end],
                v_stride: chroma_w,
            };
            session.push_frame(frame, &mut annex_b)?;
        }
        // Drain the final partial GOP (consumes the session).
        session.finish(&mut annex_b)?;
        drop(yuv_bytes); // free YUV before mux

        std::fs::write(&annex_b_temp, &annex_b)?;

        // Mux annex-B → MP4 via the in-tree HandBrake mux. The ffmpeg
        // shell-out previously used here mis-synthesised PTS for raw
        // H.264 input on streams with per-GOP POC reset — output had
        // 60/1 r_frame_rate and ~1.7 s duration regardless of true
        // length, breaking downstream tools (DECODE-FAST follow-up
        // 2026-05-24). HandBrake-style mux supplies explicit PTS per
        // frame from the encoder's known fps_num/fps_den, so the
        // failure mode is bypassed entirely. Same path the CABAC v2
        // CLI flow already uses.
        let audio_source = probe.has_audio.then_some(input.as_path());
        let pattern = phasm_core::GopPattern::Ipppp { gop: GOP_SIZE as usize };
        let mux_profile = transcode::MuxProfile::HandbrakeX264;
        let _dropped_audio = transcode::mux_annexb_to_mp4_with_profile(
            &annex_b_temp, audio_source, &probe.frame_rate,
            probe.width, probe.height, probe.n_frames, pattern,
            mux_profile, output,
        )?;
        Ok(())
    })();

    for p in cleanup_paths.drain(..) {
        let _ = std::fs::remove_file(p);
    }
    result
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
