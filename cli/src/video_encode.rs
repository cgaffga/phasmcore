// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::output::{self, OutputMode};
use crate::passphrase::get_passphrase;
use crate::progress::spawn_progress_bar;
#[cfg(any(feature = "h264-encoder", feature = "av1-encoder"))]
use crate::transcode;
use std::io::{self, IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum CodecChoice {
    H264,
    Av1,
}

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

    /// Codec to use for the stego encode. Defaults to H.264 to preserve
    /// the pre-AV1 `phasm video-encode` invocation shape.
    #[arg(long, value_enum, default_value_t = CodecChoice::H264)]
    pub codec: CodecChoice,

    /// GOP size (intra period) for the encoder. Mobile defaults are
    /// 30 (1 IDR/sec at 30 fps); CLI follows the same.
    #[arg(long, default_value_t = 30)]
    pub gop_size: usize,

    /// File attachment for the primary message (repeatable). Shadow video only.
    #[arg(long, action = clap::ArgAction::Append)]
    pub attach: Vec<PathBuf>,

    // WV.6.g — shadow messages (slots 2..9): plausible-deniability layers, each
    // with its own passphrase. When any are set, the video routes through the
    // O(GOP) streaming shadow encode. Mirrors the image `encode` flags. H.264
    // only — AV1 shadow CLI flags ship alongside the AV1 streaming-shadow port.
    #[arg(long)] pub shadow_message_2: Option<String>,
    #[arg(long)] pub shadow_message_3: Option<String>,
    #[arg(long)] pub shadow_message_4: Option<String>,
    #[arg(long)] pub shadow_message_5: Option<String>,
    #[arg(long)] pub shadow_message_6: Option<String>,
    #[arg(long)] pub shadow_message_7: Option<String>,
    #[arg(long)] pub shadow_message_8: Option<String>,
    #[arg(long)] pub shadow_message_9: Option<String>,

    // Shadow passphrases (slots 2..9; primary passphrase is `-p` above)
    #[arg(long)] pub shadow_passphrase_2: Option<String>,
    #[arg(long)] pub shadow_passphrase_3: Option<String>,
    #[arg(long)] pub shadow_passphrase_4: Option<String>,
    #[arg(long)] pub shadow_passphrase_5: Option<String>,
    #[arg(long)] pub shadow_passphrase_6: Option<String>,
    #[arg(long)] pub shadow_passphrase_7: Option<String>,
    #[arg(long)] pub shadow_passphrase_8: Option<String>,
    #[arg(long)] pub shadow_passphrase_9: Option<String>,

    // Shadow attachments (slots 2..9; repeatable per slot)
    #[arg(long, action = clap::ArgAction::Append)] pub shadow_attach_2: Vec<PathBuf>,
    #[arg(long, action = clap::ArgAction::Append)] pub shadow_attach_3: Vec<PathBuf>,
    #[arg(long, action = clap::ArgAction::Append)] pub shadow_attach_4: Vec<PathBuf>,
    #[arg(long, action = clap::ArgAction::Append)] pub shadow_attach_5: Vec<PathBuf>,
    #[arg(long, action = clap::ArgAction::Append)] pub shadow_attach_6: Vec<PathBuf>,
    #[arg(long, action = clap::ArgAction::Append)] pub shadow_attach_7: Vec<PathBuf>,
    #[arg(long, action = clap::ArgAction::Append)] pub shadow_attach_8: Vec<PathBuf>,
    #[arg(long, action = clap::ArgAction::Append)] pub shadow_attach_9: Vec<PathBuf>,
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

    // Codec dispatch. Both encoders consume YUV produced by ffmpeg's
    // demux + decode (`transcode::probe_video` / `transcode::decode_to_yuv`),
    // so any ffmpeg-decodable source (H.264, HEVC, AV1, ProRes, VP9, …)
    // works regardless of the chosen stego codec.
    //
    // H.264 sub-dispatch (WV.6.g): when shadow flags
    // (`--shadow-message-2..--shadow-message-9`) are set, route through the
    // O(GOP) streaming shadow pull encode; otherwise the (unchanged)
    // non-shadow push session. The AV1 shadow CLI flags ship alongside the
    // AV1 streaming-shadow port (after H.264's bridges land).
    match args.codec {
        CodecChoice::H264 => {
            #[cfg(feature = "h264-encoder")]
            {
                let primary_files = crate::encode::load_files(&args.attach)?;
                let shadows = collect_video_shadows(&args)?;
                if shadows.is_empty() {
                    run_oh264_encode(
                        &args.video, &message, &passphrase, &output_path,
                        args.gop_size, args.optimize,
                    )?;
                } else {
                    run_oh264_shadow_encode(
                        &args.video, &message, &primary_files, &passphrase,
                        &shadows, &output_path,
                    )?;
                }
            }
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
        }
        CodecChoice::Av1 => {
            #[cfg(feature = "av1-encoder")]
            {
                let primary_files = crate::encode::load_files(&args.attach)?;
                let shadows = collect_video_shadows(&args)?;
                if shadows.is_empty() {
                    // No-shadow per-GOP path — the in-memory whole-clip
                    // session is fine here (short clips, primary only).
                    if !primary_files.is_empty() {
                        eprintln!(
                            "note: --attach is shadow-CLI / mobile only — AV1 no-shadow CLI \
                             is text-only for now; files ignored."
                        );
                    }
                    run_av1_encode(
                        &args.video, &message, &passphrase, &output_path,
                        args.gop_size, args.optimize,
                    )?;
                } else {
                    // WV.7.14 — streaming shadow path. O(GOP) working
                    // set via Av1FileYuvSource over the ffmpeg temp.
                    run_av1_shadow_encode_streaming(
                        &args.video, &message, &primary_files, &passphrase,
                        &shadows, &output_path, args.gop_size,
                    )?;
                }
            }
            #[cfg(not(feature = "av1-encoder"))]
            {
                let _ = (&message, &passphrase, &output_path, &start, progress_handle);
                return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                    "AV1 video stego encode requires the AV1 backend. \
                     Rebuild from source: \
                     `cargo install phasm-cli --features av1-encoder`."
                        .into(),
                )));
            }
        }
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

/// Write the encoded annex-B to a temp file, then mux to MP4 via the in-tree
/// HandBrake mux. Explicit per-frame PTS (from the encoder's fps) avoids the
/// ffmpeg shell-out's mis-synthesised PTS on per-GOP-POC-reset streams — the
/// DECODE-FAST 2026-05-24 failure (60/1 r_frame_rate, ~1.7 s duration
/// regardless of true length). Shared by the shadow + non-shadow OH264 paths.
#[cfg(feature = "h264-encoder")]
fn write_and_mux_oh264(
    annex_b: &[u8],
    annex_b_temp: &Path,
    input: &Path,
    probe: &transcode::VideoProbe,
    n_frames: usize,
    gop_size: u32,
    output: &Path,
) -> Result<(), CliError> {
    std::fs::write(annex_b_temp, annex_b)?;
    let audio_source = probe.has_audio.then_some(input);
    let pattern = phasm_core::GopPattern::Ipppp { gop: gop_size as usize };
    let _dropped_audio = transcode::mux_annexb_to_mp4_with_profile(
        annex_b_temp, audio_source, &probe.frame_rate,
        probe.width, probe.height, n_frames, pattern,
        transcode::MuxProfile::HandbrakeX264, output,
    )?;
    Ok(())
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
/// This is the NON-shadow path (primary message only). Shadow messages +
/// attachments on the CLI route through `run_oh264_shadow_encode` (WV.6.g — the
/// O(GOP) streaming pull encode), dispatched from `run()` when
/// `--shadow-message-2..--shadow-message-9` are set.
///
/// v1.0 notes:
///  - mux uses the in-tree HandBrake/x264 profile
///    (`MuxProfile::HandbrakeX264`); see the mux-site comment for why the
///    earlier ffmpeg `-c copy` shell-out was dropped (2026-05-24).
#[cfg(feature = "h264-encoder")]
fn run_oh264_encode(
    input: &PathBuf,
    message: &str,
    passphrase: &str,
    output: &PathBuf,
    cli_gop_size: usize,
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
    let mut probe = transcode::probe_video(input)?;
    eprintln!(
        "Encoding via OpenH264 (streaming, production v1.0): {}x{} × {} frames @ {} fps{}",
        probe.width, probe.height, probe.n_frames, probe.frame_rate,
        if probe.has_audio { " (with audio)" } else { "" },
    );

    let yuv_temp = transcode::temp_path_with_ext(input, "yuv");
    let annex_b_temp = transcode::temp_path_with_ext(input, "h264");
    let mut cleanup_paths = vec![yuv_temp.clone(), annex_b_temp.clone()];

    let result = (|| -> Result<(), CliError> {
        // ffmpeg writes the whole decoded YUV to a temp file, which we mmap:
        // the OS pages it in on demand and can evict clean (file-backed) pages
        // under pressure, so a long clip no longer forces a non-evictable
        // O(clip) heap allocation. (The shadow path's FileYuvSource is fully
        // O(GOP); the non-shadow probe + tier-report + encode passes still
        // index the whole clip, so this is the OOM-safe equivalent — true
        // O(GOP) here would need a streaming `h264_resolve_auto_tier`.)
        transcode::decode_to_yuv(input, &yuv_temp)?;
        let yuv_file = std::fs::File::open(&yuv_temp)?;
        // SAFETY: `yuv_temp` is our own freshly-written temp file (above), not
        // modified concurrently for the lifetime of this map.
        let yuv_bytes = unsafe { memmap2::Mmap::map(&yuv_file)? };
        // The container's `nb_frames` metadata is unreliable — e.g.
        // Artlist_CarPlane.mp4 reports nb_frames=193 but the decoder yields
        // 194. The decoded buffer is authoritative: derive the real frame
        // count from it rather than trusting the probe, and only hard-error
        // if the buffer isn't a whole number of frames (genuinely corrupt).
        let frame_size = (probe.width as usize) * (probe.height as usize) * 3 / 2;
        if frame_size == 0 || yuv_bytes.len() % frame_size != 0 {
            return Err(CliError::InvalidArgs(format!(
                "decoded YUV {} bytes is not a whole number of {}x{} frames ({} B/frame)",
                yuv_bytes.len(), probe.width, probe.height, frame_size,
            )));
        }
        let decoded_frames = yuv_bytes.len() / frame_size;
        if decoded_frames != probe.n_frames {
            eprintln!(
                "note: container nb_frames={} but decoder produced {} frames — using the decoded count",
                probe.n_frames, decoded_frames,
            );
            probe.n_frames = decoded_frames;
        }

        // Parse fps string ("30/1", "30000/1001") into rational
        // numerator/denominator for the SPS time_scale fields.
        let (fps_num, fps_den) = transcode::parse_frame_rate(&probe.frame_rate);

        // Per-GOP STC isolation. CLI default 30 matches mobile's
        // DEFAULT_GOP=30 so CLI- and mobile-encoded files share
        // structure (1 IDR/sec at 30 fps). User override via
        // `--gop-size`.
        let qp: i32 = std::env::var("PHASM_DEBUG_QP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(26);
        let gop_size: u32 = (cli_gop_size.max(1)) as u32;

        // Report the resolved auto-tier (mirrors the mobile success-
        // screen "Quality mode: High Quality (auto)" UX). The encoder
        // resolves the same tier internally; this is purely the
        // user-facing announcement before the (slower) encode runs.
        report_resolved_auto_tier(
            &yuv_bytes, probe.width, probe.height, probe.n_frames,
            gop_size as usize, message.len(),
            phasm_core::CASCADE_DEFAULT_HEADROOM,
        );

        let params = EncodeSessionParams {
            width: probe.width,
            height: probe.height,
            fps_num,
            fps_den,
            qp,
            gop_size: gop_size,
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

        write_and_mux_oh264(
            &annex_b, &annex_b_temp, input, &probe, probe.n_frames, gop_size, output,
        )?;
        Ok(())
    })();

    for p in cleanup_paths.drain(..) {
        let _ = std::fs::remove_file(p);
    }
    result
}

/// WV.6.g — collect shadow layers from `--shadow-message-2..--shadow-message-9` /
/// `--shadow-passphrase-2..--shadow-passphrase-9` /
/// `--shadow-attach-2..--shadow-attach-9`. Mirrors the image
/// `encode::collect_shadows`. Owned `ShadowLayer`s;
/// [`run_oh264_shadow_encode`] borrows from them.
///
/// Shared by both encoders' shadow CLI paths (H.264 `run_oh264_shadow_encode`
/// + AV1 `run_av1_shadow_encode_streaming`), so it's gated on either encoder
/// — not `h264-encoder` alone — else the AV1-only build can't see it.
#[cfg(any(feature = "h264-encoder", feature = "av1-encoder"))]
fn collect_video_shadows(args: &VideoEncodeArgs) -> Result<Vec<phasm_core::ShadowLayer>, CliError> {
    use crate::encode::load_files;
    let msgs = [
        &args.shadow_message_2, &args.shadow_message_3,
        &args.shadow_message_4, &args.shadow_message_5,
        &args.shadow_message_6, &args.shadow_message_7,
        &args.shadow_message_8, &args.shadow_message_9,
    ];
    let passes = [
        &args.shadow_passphrase_2, &args.shadow_passphrase_3,
        &args.shadow_passphrase_4, &args.shadow_passphrase_5,
        &args.shadow_passphrase_6, &args.shadow_passphrase_7,
        &args.shadow_passphrase_8, &args.shadow_passphrase_9,
    ];
    let attaches = [
        &args.shadow_attach_2, &args.shadow_attach_3,
        &args.shadow_attach_4, &args.shadow_attach_5,
        &args.shadow_attach_6, &args.shadow_attach_7,
        &args.shadow_attach_8, &args.shadow_attach_9,
    ];
    let mut shadows = Vec::new();
    for i in 0..8 {
        if let Some(msg) = msgs[i] {
            let label = format!("Shadow {} passphrase", i + 2);
            let pass = get_passphrase(passes[i].as_deref(), &label, true)?;
            let files = load_files(attaches[i])?;
            shadows.push(phasm_core::ShadowLayer { message: msg.clone(), passphrase: pass, files });
        }
    }
    // Smallest payload first — matches the mobile app + image CLI ordering.
    shadows.sort_by_key(|s| phasm_core::compressed_payload_size(&s.message, &s.files));
    Ok(shadows)
}

/// WV.6.g — streaming SHADOW video encode (O(GOP) working set). ffmpeg decodes the
/// source to a temp YUV file; a [`FileYuvSource`] then feeds the streaming encoder
/// ONE GOP at a time (the whole decoded YUV is never loaded into RAM, unlike the
/// non-shadow `run_oh264_encode`). Same in-tree HandBrake mux.
#[cfg(feature = "h264-encoder")]
fn run_oh264_shadow_encode(
    input: &PathBuf,
    message: &str,
    primary_files: &[phasm_core::FileEntry],
    passphrase: &str,
    shadows: &[phasm_core::ShadowLayer],
    output: &PathBuf,
) -> Result<(), CliError> {
    use phasm_core::codec::h264::openh264_stego::{
        h264_encode_with_shadows_streaming, EncodeOpts, FileYuvSource,
    };
    use phasm_core::stego::shadow_layer::ShadowLayer as VideoShadowLayer;
    use phasm_core::CostWeights;

    transcode::ensure_ffmpeg_available()?;
    let probe = transcode::probe_video(input)?;
    eprintln!(
        "Encoding via OpenH264 (streaming shadow, {} shadow{}): {}x{} @ {} fps{}",
        shadows.len(),
        if shadows.len() == 1 { "" } else { "s" },
        probe.width, probe.height, probe.frame_rate,
        if probe.has_audio { " (with audio)" } else { "" },
    );

    let yuv_temp = transcode::temp_path_with_ext(input, "yuv");
    let annex_b_temp = transcode::temp_path_with_ext(input, "h264");
    let mut cleanup_paths = vec![yuv_temp.clone(), annex_b_temp.clone()];

    let result = (|| -> Result<(), CliError> {
        transcode::decode_to_yuv(input, &yuv_temp)?;

        // Exact frame count from the file SIZE — no whole-clip load (O(GOP) RAM).
        let frame_size = (probe.width as usize) * (probe.height as usize) * 3 / 2;
        let file_len = std::fs::metadata(&yuv_temp)?.len() as usize;
        if frame_size == 0 || file_len % frame_size != 0 {
            return Err(CliError::InvalidArgs(format!(
                "decoded YUV {file_len} bytes is not a whole number of {}x{} frames",
                probe.width, probe.height,
            )));
        }
        let n_frames = (file_len / frame_size) as u32;

        let qp: i32 = std::env::var("PHASM_DEBUG_QP").ok().and_then(|s| s.parse().ok()).unwrap_or(26);
        const GOP_SIZE: u32 = 30;
        let opts = EncodeOpts { qp, intra_period: GOP_SIZE as i32 };
        let weights = CostWeights::default();

        // owned ShadowLayer (String) → borrow ShadowLayer<'a> (&str), as the
        // mobile bridges do.
        let video_shadows: Vec<VideoShadowLayer> = shadows
            .iter()
            .map(|s| VideoShadowLayer {
                message: &s.message,
                passphrase: &s.passphrase,
                files: &s.files,
            })
            .collect();

        let mut source =
            FileYuvSource::open(&yuv_temp, probe.width, probe.height, n_frames, GOP_SIZE)?;
        // WV.6.g.progress — throttled determinate progress line. The streaming
        // shadow encode does ~4 forward passes over the GOPs; the fraction moves
        // continuously (off 0% on the first GOP, 100% only at the very end).
        let mut last_pct = -1i32;
        let mut progress_cb = |frac: f32| {
            let pct = (frac * 100.0).round() as i32;
            if pct != last_pct {
                last_pct = pct;
                eprint!("\r  shadow encode: {pct:>3}%");
                let _ = std::io::Write::flush(&mut std::io::stderr());
            }
        };
        let annex_b = h264_encode_with_shadows_streaming(
            &mut source, probe.width, probe.height, n_frames, opts,
            message, primary_files, passphrase, &video_shadows, &weights,
            Some(&mut progress_cb),
        )?;
        eprintln!(); // terminate the progress line

        write_and_mux_oh264(
            &annex_b, &annex_b_temp, input, &probe, n_frames as usize, GOP_SIZE, output,
        )?;
        Ok(())
    })();

    for p in cleanup_paths.drain(..) {
        let _ = std::fs::remove_file(p);
    }
    result
}

/// WV.7.14 — STREAMING AV1 shadow video encode (O(GOP) working set,
/// CLI / desktop side). Mirror of `run_oh264_shadow_encode`. ffmpeg
/// decodes the source to a temp YUV file; an [`Av1FileYuvSource`] then
/// feeds the streaming AV1 shadow encoder ONE GOP at a time (the whole
/// decoded YUV is never loaded into RAM, unlike the in-memory whole-
/// clip session that `run_av1_encode` uses for the no-shadow path).
///
/// Shadow file attachments are still deferred (mirror of mobile + H.264
/// CLI). `--attach` is honoured for the PRIMARY message only.
#[cfg(feature = "av1-encoder")]
fn run_av1_shadow_encode_streaming(
    input: &PathBuf,
    message: &str,
    primary_files: &[phasm_core::FileEntry],
    passphrase: &str,
    shadows: &[phasm_core::ShadowLayer],
    output: &PathBuf,
    cli_gop_size: usize,
) -> Result<(), CliError> {
    use phasm_core::codec::av1::stego::whole_video::{
        av1_encode_with_shadows_streaming, Av1FileYuvSource,
    };
    use phasm_core::codec::mp4::build::{
        build_mp4_av1, build_mp4_av1_with_audio, FrameTiming,
    };
    use phasm_core::stego::{crypto, frame, payload};
    use phasm_core::Av1StreamingEncodeParams;

    transcode::ensure_ffmpeg_available()?;
    let probe = transcode::probe_video(input)?;
    eprintln!(
        "Encoding via phasm-rav1e (AV1 streaming shadow, {} shadow{}): {}x{} × {} frames @ {} fps{}",
        shadows.len(),
        if shadows.len() == 1 { "" } else { "s" },
        probe.width, probe.height, probe.n_frames, probe.frame_rate,
        if probe.has_audio { " (with audio)" } else { "" },
    );
    if probe.width % 8 != 0 || probe.height % 8 != 0 {
        return Err(CliError::InvalidArgs(format!(
            "AV1 encode requires 8-aligned dimensions; got {}x{} \
             (crop / pre-scale with ffmpeg first)",
            probe.width, probe.height,
        )));
    }
    if shadows.iter().any(|s| !s.files.is_empty()) {
        eprintln!(
            "note: shadow --attach flags are not yet supported on the AV1 CLI \
             (parity with mobile bridges); shadow files ignored."
        );
    }

    let yuv_temp = transcode::temp_path_with_ext(input, "yuv");
    let mut cleanup_paths = vec![yuv_temp.clone()];

    let result = (|| -> Result<(), CliError> {
        transcode::decode_to_yuv(input, &yuv_temp)?;

        // Exact frame count from file SIZE — no whole-clip load.
        let frame_size = (probe.width as usize) * (probe.height as usize) * 3 / 2;
        let file_len = std::fs::metadata(&yuv_temp)?.len() as usize;
        if frame_size == 0 || file_len % frame_size != 0 {
            return Err(CliError::InvalidArgs(format!(
                "decoded YUV {file_len} bytes is not a whole number of {}x{} frames",
                probe.width, probe.height,
            )));
        }
        let n_frames = (file_len / frame_size) as u32;

        // Two-stage primary framing — must match what the mobile FFI
        // bridges do (iOS WV.7.9 + Android WV.7.10):
        //   (a) `payload::encode_payload(text, files)` — INNER framing
        //       (compression flags + text + files); the decoder's
        //       `finish_primary_payload` reverses this.
        //   (b) `crypto::encrypt + frame::build_frame` — OUTER stego
        //       framing that `av1_encode_with_shadows_streaming`
        //       expects (the session API does this internally, the
        //       streaming entry doesn't — see whole_video.rs:62).
        let primary_payload = payload::encode_payload(message, primary_files)
            .map_err(CliError::from)?;
        let (ciphertext, nonce, salt) = crypto::encrypt(&primary_payload, passphrase)
            .map_err(CliError::from)?;
        let primary_framed = frame::build_frame(primary_payload.len(), &salt, &nonce, &ciphertext);

        let mut quantizer: usize = std::env::var("PHASM_AV1_QUANTIZER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);
        if quantizer == 0 {
            quantizer = 100;
        }
        let encode_gop_size: u32 = cli_gop_size.max(1) as u32;
        let params = Av1StreamingEncodeParams {
            width: probe.width,
            height: probe.height,
            quantizer,
            gop_size: encode_gop_size,
            total_frames_hint: n_frames,
        };

        // Shadow framing: same payload::encode_payload INNER framing as
        // primary (text + files prefix) — the encoder then does its
        // own encrypt+frame OUTER layer inside
        // `prepare_shadow_pre_selection_av1`. Shadow file attachments
        // are deferred (mobile parity); pass an empty files list.
        let shadow_payloads: Vec<Vec<u8>> = shadows
            .iter()
            .map(|s| payload::encode_payload(&s.message, &[]).map_err(CliError::from))
            .collect::<Result<_, _>>()?;
        let shadow_borrows: Vec<(&str, &[u8])> = shadows
            .iter()
            .zip(shadow_payloads.iter())
            .map(|(s, p)| (s.passphrase.as_str(), p.as_slice()))
            .collect();

        // Shadow parity floor — matches mobile bridge + WV.7.0 gate
        // (16). The cascade tries [16, 32, 64, 128]; lower floors
        // expose tiny-fdl edge cases in the small-message path that
        // production callers haven't hit yet (mobile + bridges always
        // ship 16).
        const SHADOW_PARITY_LEN_FLOOR: usize = 16;

        let mut source = Av1FileYuvSource::open(
            &yuv_temp, probe.width as u32, probe.height as u32,
            n_frames, encode_gop_size,
        )
        .map_err(av1_err_to_cli)?;
        let obus = av1_encode_with_shadows_streaming(
            &mut source, n_frames, params,
            &primary_framed, passphrase,
            &shadow_borrows,
            SHADOW_PARITY_LEN_FLOOR,
        )
        .map_err(av1_err_to_cli)?;

        if obus.is_empty() {
            return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                "AV1 streaming shadow encoder produced no OBU bytes".into(),
            )));
        }

        // Mux. Mirror of `run_av1_encode`'s mux step — audio passthrough
        // best-effort via the in-tree muxer's source-MP4 path.
        let (fps_num, fps_den) = transcode::parse_frame_rate(&probe.frame_rate);
        let timing = FrameTiming { fps_num, fps_den };
        let mp4 = if probe.has_audio {
            let source_bytes = std::fs::read(input).map_err(CliError::from)?;
            build_mp4_av1_with_audio(&obus, probe.width, probe.height, timing, &source_bytes)
        } else {
            build_mp4_av1(&obus, probe.width, probe.height, timing)
        }
        .map_err(|e| CliError::InvalidArgs(format!("AV1 MP4 mux failed: {e}")))?;
        std::fs::write(output, &mp4).map_err(CliError::from)?;
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

/// AV1 stego encode path. Mirrors `run_oh264_encode`'s shape:
///
/// MP4 → ffmpeg → YUV → `Av1StreamingProbeSession` (per-GOP cap probe)
/// → `Av1StreamingEncodeSession` (planned via `plan_safe_balanced` with
/// `AllocationCalibration::AV1_1080P_QP30`) → OBU stream → in-tree
/// `build_mp4_av1_with_audio` mux → MP4.
///
/// v1.0 CLI limitations (text-only, same as H.264 CLI):
///   - no file attachments (mobile + the FFI surface support them, see
///     AV1.FILE1/2; the CLI flag wiring is a small follow-on).
///   - no shadow messages (whole-video shadow API is on the FFI; same
///     follow-on adds CLI flags for it).
///   - container always video-only / audio passthrough (no audio
///     drop flag).
#[cfg(feature = "av1-encoder")]
fn run_av1_encode(
    input: &PathBuf,
    message: &str,
    passphrase: &str,
    output: &PathBuf,
    cli_gop_size: usize,
    optimize: bool,
) -> Result<(), CliError> {
    let _ = optimize; // matches the H.264 wiring-only TODO; video optimizer is a follow-on
    use phasm_core::{
        av1_capacity, Av1StegoError, Av1StreamingEncodeParams, Av1StreamingEncodeSession,
        Av1StreamingProbeSession,
    };
    use phasm_core::codec::av1::stego::session::Av1PlanOutcome;
    use phasm_core::codec::mp4::build::{
        build_mp4_av1, build_mp4_av1_with_audio, FrameTiming,
    };
    use phasm_core::stego::calibration::AllocationCalibration;
    use phasm_core::stego::payload;

    transcode::ensure_ffmpeg_available()?;
    let mut probe = transcode::probe_video(input)?;
    eprintln!(
        "Encoding via phasm-rav1e (AV1 streaming-session): {}x{} × {} frames @ {} fps{}",
        probe.width, probe.height, probe.n_frames, probe.frame_rate,
        if probe.has_audio { " (with audio)" } else { "" },
    );

    // AV1 requires 8-byte alignment (vs H.264's 16-byte). Most real-world
    // sources are 8-aligned (1920×1080, 1280×720, 640×480, …); flag a
    // clear error rather than silently cropping.
    if probe.width % 8 != 0 || probe.height % 8 != 0 {
        return Err(CliError::InvalidArgs(format!(
            "AV1 encode requires 8-aligned dimensions; got {}x{} \
             (crop / pre-scale with ffmpeg first)",
            probe.width, probe.height
        )));
    }

    let yuv_temp = transcode::temp_path_with_ext(input, "yuv");
    let mut cleanup_paths = vec![yuv_temp.clone()];

    let result = (|| -> Result<(), CliError> {
        transcode::decode_to_yuv(input, &yuv_temp)?;
        let yuv_bytes = std::fs::read(&yuv_temp)?;

        // Reconcile container nb_frames vs decoded buffer (same gotcha
        // as the H.264 path — CarPlane reports 193 but decode yields 194).
        let frame_size = (probe.width as usize) * (probe.height as usize) * 3 / 2;
        if frame_size == 0 || yuv_bytes.len() % frame_size != 0 {
            return Err(CliError::InvalidArgs(format!(
                "decoded YUV {} bytes is not a whole number of {}x{} frames ({} B/frame)",
                yuv_bytes.len(), probe.width, probe.height, frame_size,
            )));
        }
        let decoded_frames = yuv_bytes.len() / frame_size;
        if decoded_frames != probe.n_frames {
            eprintln!(
                "note: container nb_frames={} but decoder produced {} frames — using the decoded count",
                probe.n_frames, decoded_frames,
            );
            probe.n_frames = decoded_frames;
        }

        // Frame primary `(text, files)` into the canonical
        // `payload::encode_payload` byte sequence so the bytes round-
        // trip through the decoder's `finish_primary_payload`. CLI v1.0
        // ships text-only — files is always `&[]` — matching the H.264
        // CLI's surface. The decoder's "no payload frame" fallback
        // keeps legacy text-only streams parsing too.
        let primary_bytes = payload::encode_payload(message, &[]).map_err(CliError::from)?;

        let mut quantizer: usize = std::env::var("PHASM_AV1_QUANTIZER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);
        // The calibration is locked at QP=30 corpus measurements
        // (`AllocationCalibration::AV1_1080P_QP30`); the production
        // rav1e quantizer is on a separate 0..255 scale. The mobile
        // bridges and this CLI ship quantizer=100 as the production
        // default — keep them consistent.
        if quantizer == 0 {
            quantizer = 100;
        }
        let encode_gop_size: u32 = cli_gop_size.max(1) as u32;
        let probe_params = Av1StreamingEncodeParams {
            width: probe.width,
            height: probe.height,
            quantizer,
            // Probe API contract: gop_size=1 (D-phase per-frame cover
            // count). Encode session below uses the real encode_gop_size.
            gop_size: 1,
            total_frames_hint: probe.n_frames as u32,
        };
        let encode_params = Av1StreamingEncodeParams {
            gop_size: encode_gop_size,
            ..probe_params
        };

        // Mobile reports a per-MB cap before encode runs. CLI mirrors
        // that: closed-form cap from `av1_capacity` so the user sees
        // "will this fit?" without a separate `phasm video-capacity`
        // invocation. Failure is best-effort (skip the announcement,
        // proceed with encode — the encode itself surfaces real
        // overflow as `MessageTooLarge`).
        if let Ok(info) = av1_capacity(&yuv_bytes, probe_params) {
            eprintln!(
                "  Tier 1 cover: {} bits   primary cap: {} B   payload: {} B",
                info.cover_size_bits, info.primary_max_message_bytes, primary_bytes.len(),
            );
        }

        // ── G.2.P4 per-GOP probe + plan_safe_balanced.
        // Mirror of the Android transcoder's flow
        // (`PhasmAv1Transcoder.kt::probePerGopCaps` + the
        // `av1SessionPlanSafeBalanced` step). Runs the probe at
        // gop_size=1, aggregates per-frame caps into per-encode-GOP
        // caps, then installs the balanced plan in one
        // `plan_safe_balanced` iteration with full-clip samples.
        let per_frame_cover_bits: Vec<usize> = {
            let mut probe_session = Av1StreamingProbeSession::create(probe_params)
                .map_err(av1_err_to_cli)?;
            for frame_idx in 0..probe.n_frames {
                let off = frame_idx * frame_size;
                probe_session
                    .push_frame(&yuv_bytes[off..off + frame_size])
                    .map_err(av1_err_to_cli)?;
            }
            probe_session.finish().per_gop_byte_caps()
        };

        // Aggregate the per-frame byte caps into per-encode-GOP caps.
        // The probe yields one entry per frame (each frame = its own
        // probe GOP); the encode uses `encode_gop_size`-frame GOPs.
        let g = encode_gop_size as usize;
        let n_encode_gops = (per_frame_cover_bits.len() + g - 1) / g.max(1);
        let mut per_gop_caps: Vec<usize> = vec![0; n_encode_gops];
        for (i, &c) in per_frame_cover_bits.iter().enumerate() {
            per_gop_caps[i / g] = per_gop_caps[i / g].saturating_add(c);
        }

        let mut session = Av1StreamingEncodeSession::create(passphrase, &primary_bytes, encode_params)
            .map_err(av1_err_to_cli)?;

        let samples: Vec<(usize, usize)> = per_gop_caps
            .iter()
            .copied()
            .enumerate()
            .collect();
        match session.plan_safe_balanced(
            &samples, g, &AllocationCalibration::AV1_1080P_QP30,
        ).map_err(av1_err_to_cli)? {
            Av1PlanOutcome::Installed { window } => {
                eprintln!(
                    "  balanced plan: W={} / n_gops={}",
                    window, per_gop_caps.len()
                );
            }
            Av1PlanOutcome::NeedMoreSamples { total_target, .. } => {
                // Should not happen with full-clip samples — the planner
                // has all caps available. Log + fall through to the
                // default even-split (always safe).
                eprintln!(
                    "note: plan_safe_balanced unexpectedly NeedMoreSamples \
                     (total_target={}) with full-clip samples; using even-split",
                    total_target
                );
            }
            Av1PlanOutcome::MessageTooLarge => {
                return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                    "message exceeds AV1 cover-bit capacity at this video size \
                     (run `phasm video-capacity --codec av1` to see the cap)"
                        .into(),
                )));
            }
        }

        let mut obus: Vec<u8> = Vec::with_capacity(yuv_bytes.len() / 8);
        for frame_idx in 0..probe.n_frames {
            let off = frame_idx * frame_size;
            session
                .push_frame(&yuv_bytes[off..off + frame_size], &mut obus)
                .map_err(av1_err_to_cli)?;
        }
        session.finish(&mut obus).map_err(av1_err_to_cli)?;

        if obus.is_empty() {
            return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                "AV1 encoder produced no OBU bytes".into(),
            )));
        }

        let (fps_num, fps_den) = transcode::parse_frame_rate(&probe.frame_rate);
        let timing = FrameTiming { fps_num, fps_den };

        // Audio passthrough via the in-tree muxer's source-MP4 path
        // (mirror of `phasm_av1_build_mp4` in ios-bridge). Read failure
        // falls back to video-only mux — never blocks on audio.
        let mp4 = if probe.has_audio {
            let source_bytes = std::fs::read(input).map_err(CliError::from)?;
            build_mp4_av1_with_audio(&obus, probe.width, probe.height, timing, &source_bytes)
        } else {
            build_mp4_av1(&obus, probe.width, probe.height, timing)
        }
        .map_err(|e| {
            CliError::from(phasm_core::StegoError::InvalidVideo(format!(
                "AV1 MP4 mux failed: {e}"
            )))
        })?;
        drop(yuv_bytes); // YUV is large; free before writing the (smaller) MP4

        std::fs::write(output, &mp4).map_err(CliError::from)?;
        Ok(())
    })();

    for p in cleanup_paths.drain(..) {
        let _ = std::fs::remove_file(p);
    }
    result
}

#[cfg(feature = "av1-encoder")]
fn av1_err_to_cli(e: phasm_core::Av1StegoError) -> CliError {
    CliError::from(phasm_core::StegoError::InvalidVideo(format!(
        "AV1 encode failed: {e:?}"
    )))
}
