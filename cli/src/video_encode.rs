// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::output::{self, OutputMode};
use crate::passphrase::get_passphrase;
use crate::progress::spawn_progress_bar;
use crate::transcode;
#[cfg(not(any(feature = "openh264-backend", feature = "cabac-stego")))]
use phasm_core::{detect_video_codec, VideoCodec};
#[cfg(not(feature = "cabac-stego"))]
use phasm_core::h264_ghost_encode;
#[cfg(feature = "cabac-stego")]
use phasm_core::h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files_with_tier;
#[cfg(feature = "cabac-stego")]
use phasm_core::h264_stego_encode_yuv_string_with_n_shadows_with_pattern_and_files;
#[cfg(feature = "cabac-stego")]
use phasm_core::{FileEntry, ShadowLayer};
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;
use std::time::Instant;

use clap::{ArgAction, Parser, ValueEnum};

/// D.0.1 (2026-05-14) — encoder backend selection.
///
/// `open-h264` (default when `--features openh264-backend`) is the
/// production v1.0 pipeline: vendored OpenH264 fork with visual_recon
/// dual-recon hooks. Closed Phase C.
///
/// `rust-h264` is the pure-Rust 4-domain CABAC H.264 stego pipeline.
/// EXPERIMENTAL: ~1000-4000× slower than OpenH264 at production
/// resolutions (1080p × 30 ≈ 60 min). Kept as opt-in for research +
/// patent-free distribution. Default when only `--features
/// cabac-stego` is on.
///
/// (The legacy CAVLC bitstream-mod path is only present when neither
/// stego feature is enabled, in which case this flag isn't exposed.)
#[cfg(any(feature = "openh264-backend", feature = "cabac-stego"))]
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum EncoderChoice {
    /// Production OpenH264-backend (default). Phase C.8 visual_recon.
    #[cfg(feature = "openh264-backend")]
    #[clap(name = "open-h264")]
    OpenH264,
    /// EXPERIMENTAL pure-Rust H.264 4-domain CABAC stego. ~1000-4000× slower.
    #[clap(name = "rust-h264")]
    RustH264,
}

#[cfg(any(feature = "openh264-backend", feature = "cabac-stego"))]
impl Default for EncoderChoice {
    fn default() -> Self {
        #[cfg(feature = "openh264-backend")]
        return EncoderChoice::OpenH264;
        #[cfg(not(feature = "openh264-backend"))]
        return EncoderChoice::RustH264;
    }
}

/// D'.5 — CLI surface for the cascade-safety tier (Track 1).
///
/// Mirrors `phasm_core::CascadeTier` with kebab-case CLI names matching
/// the mobile UI ladder. `auto` picks the highest tier whose capacity
/// covers the message + headroom.
#[cfg(feature = "cabac-stego")]
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum CliCascadeTier {
    /// Pick highest-quality tier that fits the message (default).
    Auto,
    /// Tier 0 — no filter, full capacity.
    #[clap(name = "max-capacity")]
    MaxCapacity,
    /// Tier 1 — mild filter, +1-2 dB.
    Balanced,
    /// Tier 2 — moderate filter, +3-4 dB.
    Quality,
    /// Tier 3 — heavy filter, +5-7 dB.
    #[clap(name = "high-quality")]
    HighQuality,
    /// Tier 4 — maximum filter, +8-12 dB. Best for small messages.
    #[clap(name = "best-quality")]
    BestQuality,
}

#[cfg(feature = "cabac-stego")]
impl CliCascadeTier {
    pub fn to_lib(self) -> phasm_core::CascadeTier {
        match self {
            CliCascadeTier::Auto => phasm_core::CascadeTier::Auto,
            CliCascadeTier::MaxCapacity => phasm_core::CascadeTier::Tier0,
            CliCascadeTier::Balanced => phasm_core::CascadeTier::Tier1,
            CliCascadeTier::Quality => phasm_core::CascadeTier::Tier2,
            CliCascadeTier::HighQuality => phasm_core::CascadeTier::Tier3,
            CliCascadeTier::BestQuality => phasm_core::CascadeTier::Tier4,
        }
    }
}

/// D'.5 — print the user-facing cascade-tier line. Mirrors the mobile
/// success-screen "Quality mode: X (auto)" UX. For explicit (non-Auto)
/// selection, the suffix is omitted. The actual resolved tier under
/// Auto isn't known until inside the encoder; this just announces the
/// user's selection.
#[cfg(feature = "cabac-stego")]
fn report_cascade_tier(tier: phasm_core::CascadeTier, headroom: f32) {
    use phasm_core::CascadeTier;
    let name = tier.ui_name();
    match tier {
        CascadeTier::Auto => {
            let pct = ((headroom - 1.0) * 100.0).round() as i32;
            eprintln!("Cascade tier: Auto (headroom {pct}%) — encoder picks highest-quality fit");
        }
        _ => eprintln!("Cascade tier: {name}"),
    }
}

/// #796 — resolve + report the Auto tier using the OH264 walker, so the
/// CLI can show "Cascade tier: High Quality (auto)" before encoding.
/// Falls back silently if the resolve probe fails (encode will then
/// resolve internally with the same logic).
#[cfg(feature = "openh264-backend")]
fn report_resolved_auto_tier(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    message_bytes: usize,
    headroom: f32,
) {
    use phasm_core::{h264_resolve_auto_tier_oh264, CascadeTier};
    use phasm_core::codec::h264::openh264_stego::EncodeOpts;
    let opts = EncodeOpts { qp: 26, intra_period: gop_size as i32 };
    // Framed message size ≈ message + ~64 (crypto envelope) + ~4 (chunk).
    let estimated_framed_bytes = message_bytes.saturating_add(70);
    match h264_resolve_auto_tier_oh264(
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

    /// §6E-A5(c.4) — Shadow message (plausible deniability layer).
    /// Legacy alias for `--m2`. When set, embeds a primary message
    /// plus ONE shadow message into the same video. Each layer
    /// decodes only with its own passphrase. Requires `--shadow-pass`.
    /// CABAC-stego feature only.
    #[cfg(feature = "cabac-stego")]
    #[arg(long = "shadow-msg")]
    pub shadow_message: Option<String>,

    /// Shadow passphrase. Must differ from primary passphrase. Legacy
    /// alias for `--p2`.
    #[cfg(feature = "cabac-stego")]
    #[arg(long = "shadow-pass")]
    pub shadow_passphrase: Option<String>,

    // Shadow messages (m2..m9) — parity with image stego.
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub m2: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub m3: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub m4: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub m5: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub m6: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub m7: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub m8: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub m9: Option<String>,

    // Shadow passphrases (p2..p9).
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub p2: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub p3: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub p4: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub p5: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub p6: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub p7: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub p8: Option<String>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long)] pub p9: Option<String>,

    /// File attachment for primary message (repeatable). CABAC v2 only.
    #[cfg(feature = "cabac-stego")]
    #[arg(long, action = ArgAction::Append)]
    pub attach: Vec<PathBuf>,

    /// IDR period (gop_size) in frames. Range 1..=n_frames. Default
    /// 30 (1 s @ 30fps), matching typical phone-source cadence.
    /// Lower values reduce in-flight memory at long clips (more
    /// streaming-pipeline lockstep) at the cost of bytestream
    /// overhead from extra SPS/PPS/AUD/IDR emissions; higher values
    /// reduce overhead but extend the P/B chain. CABAC v2 path only.
    #[cfg(feature = "cabac-stego")]
    #[arg(long = "gop-size")]
    pub gop_size: Option<usize>,

    /// §Stealth.L4 — output container profile. Default
    /// `handbrake-x264` lands the file in the libx264 container
    /// metaclass (Yang/EVA + Altinisik decision-tree leaf), matching
    /// the v1.0 strategy doc target. `ffmpeg` keeps the legacy
    /// shell-out muxer with audio passthrough — less stealthy
    /// (ffmpeg-class container fingerprint) but uses ffmpeg for the
    /// final mux step. CABAC v2 path only.
    ///
    /// **Audio**: both profiles pass the source audio track through
    /// verbatim (§Stealth.L4.5). The codec configuration record
    /// (esds for AAC, etc.) is preserved byte-exact; only chunk
    /// offsets are rewritten.
    #[cfg(feature = "cabac-stego")]
    #[arg(long = "mux-profile", default_value = "handbrake-x264")]
    pub mux_profile: transcode::MuxProfile,

    /// D.0.1 — encoder backend (default: `open-h264` when built with
    /// `--features openh264-backend`, the production v1.0 path).
    /// Override with `--encoder rust-h264` for the slow (EXPERIMENTAL)
    /// pure-Rust pipeline.
    #[cfg(any(feature = "openh264-backend", feature = "cabac-stego"))]
    #[arg(long = "encoder", value_enum, default_value_t = EncoderChoice::default())]
    pub encoder: EncoderChoice,

    /// D'.5 — cascade-safety tier (Track 1 tunable quality).
    ///
    /// Default `auto` picks the highest tier whose capacity covers
    /// `msg_bytes × (1 + cascade_headroom)`. Higher tier = better
    /// visual quality, lower capacity. See `phasm video-capacity` for
    /// per-tier capacity. Currently only honored on `--encoder rust-h264`;
    /// OH264 path integration is task #795.
    #[cfg(feature = "cabac-stego")]
    #[arg(long = "cascade-tier", value_enum, default_value_t = CliCascadeTier::Auto)]
    pub cascade_tier: CliCascadeTier,

    /// D'.5 — auto-tier headroom (fractional). Encoder picks highest
    /// tier where `capacity ≥ msg_bytes × (1 + headroom)`. Default 0.2
    /// (20%) covers AES envelope + STC w-slack + per-GOP variance.
    /// Only honored when `--cascade-tier auto`.
    #[cfg(feature = "cabac-stego")]
    #[arg(long = "cascade-headroom", default_value_t = 0.2)]
    pub cascade_headroom: f32,

    /// VID-OPT (2026-05-24) — texture-adaptive cover optimizer toggle.
    /// Wiring only in this commit: the flag is threaded to the encoder
    /// dispatch site but the actual optimizer pass for video lands in
    /// a follow-on. Mirrors the image-side `--optimize` flag in
    /// `encode.rs` (line 81-83).
    #[arg(long)]
    pub optimize: bool,

    // Shadow attachments (attach2..attach9).
    #[cfg(feature = "cabac-stego")]
    #[arg(long, action = ArgAction::Append)] pub attach2: Vec<PathBuf>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long, action = ArgAction::Append)] pub attach3: Vec<PathBuf>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long, action = ArgAction::Append)] pub attach4: Vec<PathBuf>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long, action = ArgAction::Append)] pub attach5: Vec<PathBuf>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long, action = ArgAction::Append)] pub attach6: Vec<PathBuf>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long, action = ArgAction::Append)] pub attach7: Vec<PathBuf>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long, action = ArgAction::Append)] pub attach8: Vec<PathBuf>,
    #[cfg(feature = "cabac-stego")]
    #[arg(long, action = ArgAction::Append)] pub attach9: Vec<PathBuf>,
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

    // D.0.7.14 — encoder routing.
    //
    // OH264 and CABAC v2 paths use ffmpeg to demux + decode the
    // source to raw YUV before the encode pipeline ever sees it, so
    // they accept any ffmpeg-decodable codec (H.264, HEVC, ProRes,
    // VP9, AV1, …). The legacy CAVLC bitstream-mod path
    // (`run_cavlc_encode`) edits the parsed Annex-B in place and
    // therefore still requires an H.264 input.
    #[cfg(any(feature = "openh264-backend", feature = "cabac-stego"))]
    {
        let encoder = args.encoder;
        let optimize = args.optimize;
        match encoder {
            #[cfg(feature = "openh264-backend")]
            EncoderChoice::OpenH264 => {
                run_oh264_encode(&args.video, &message, &passphrase, &output_path, optimize)?;
            }
            EncoderChoice::RustH264 => {
                let primary_files = load_files(&args.attach)?;
                let shadows = collect_shadows(&args)?;
                let gop_override = args.gop_size;
                let cascade_tier = args.cascade_tier.to_lib();
                let cascade_headroom = 1.0 + args.cascade_headroom;
                if shadows.is_empty() {
                    run_cabac_encode(
                        &args.video, &message, &primary_files, &passphrase,
                        gop_override, args.mux_profile, &output_path, optimize,
                        cascade_tier, cascade_headroom,
                    )?;
                } else {
                    run_cabac_encode_with_shadows(
                        &args.video, &message, &primary_files, &passphrase,
                        &shadows, gop_override, args.mux_profile, &output_path, optimize,
                    )?;
                }
            }
        }
    }
    #[cfg(not(any(feature = "openh264-backend", feature = "cabac-stego")))]
    {
        let file_bytes = std::fs::read(&args.video)?;
        match detect_video_codec(&file_bytes) {
            VideoCodec::H264 => {
                run_cavlc_encode(&args.video, &file_bytes, &message, &passphrase, &output_path)?;
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
                    "unsupported or unrecognised video codec (expected H.264)".into(),
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

/// Phase 6.5 wiring — CABAC v2 streaming-orchestrator path. Mirrors
/// what the iOS + Android bridges call (`h264_stego_encode_yuv_string
/// _4domain_multigop_streaming_v2`). Uses ffmpeg to demux MP4 → YUV
/// and to remux Annex-B + audio → MP4, since phasm-core has no
/// built-in H.264 → YUV decoder.
#[cfg(feature = "cabac-stego")]
#[allow(clippy::too_many_arguments)]
fn run_cabac_encode(
    input: &PathBuf,
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    gop_override: Option<usize>,
    mux_profile: transcode::MuxProfile,
    output: &PathBuf,
    // VID-OPT (2026-05-24) — wiring only. The flag reaches this dispatch
    // site but isn't yet plumbed into the Scheme A encoder API. Functional
    // optimizer pass for video lands as a follow-on.
    optimize: bool,
    // D'.5 — Track 1 cascade-safety tier + auto-tier headroom.
    cascade_tier: phasm_core::CascadeTier,
    cascade_headroom: f32,
) -> Result<(), CliError> {
    let _ = optimize; // TODO: thread to encoder when video optimizer ships
    transcode::ensure_ffmpeg_available()?;
    let probe = transcode::probe_video(input)?;
    eprintln!(
        "Encoding via CABAC v2: {}x{} × {} frames @ {} fps{}",
        probe.width, probe.height, probe.n_frames, probe.frame_rate,
        if probe.has_audio { " (with audio)" } else { "" },
    );

    // gop_size default 30 = 1s @ 30fps, matching real iPhone
    // capture cadence. Override via `--gop-size N` (#98). Real-
    // world phone GOP is typically 30-300; 30 keeps re-keying
    // frequent enough for efficient lockstep memory bounds without
    // ballooning IDR overhead.
    let gop_size = resolve_gop_size(gop_override, probe.n_frames)?;

    let yuv_temp = transcode::temp_path_with_ext(input, "yuv");
    let annex_b_temp = transcode::temp_path_with_ext(input, "h264");
    let mut cleanup_paths = vec![yuv_temp.clone(), annex_b_temp.clone()];

    let result = (|| -> Result<(), CliError> {
        transcode::decode_to_yuv(input, &yuv_temp)?;
        let yuv_bytes = std::fs::read(&yuv_temp)?;
        let expected = (probe.width as usize) * (probe.height as usize) * 3 / 2 * probe.n_frames;
        if yuv_bytes.len() != expected {
            return Err(CliError::InvalidArgs(format!(
                "ffmpeg-decoded YUV is {} bytes; expected {} ({}x{} × 1.5 × {})",
                yuv_bytes.len(), expected, probe.width, probe.height, probe.n_frames,
            )));
        }
        // §Stealth.L3.2 — auto-select GOP pattern from source MP4
        // cadence. IPPPP-source clips emit IPPPP (no B-frames, no
        // ctts); IBPBP-source clips emit IBPBP. HEVC/AV1/non-H.264
        // sources fall through to the HandBrake/x264-medium centroid.
        // Caller's --gop-size override wins over detected GOP size
        // while preserving the detected B-frame structure.
        let pattern = {
            let detected = std::fs::read(input)
                .ok()
                .as_deref()
                .map(|b| phasm_core::GopPattern::auto_select(Some(b)))
                .unwrap_or_else(phasm_core::GopPattern::handbrake_x264_centroid);
            match detected {
                phasm_core::GopPattern::Ipppp { .. } => {
                    phasm_core::GopPattern::Ipppp { gop: gop_size }
                }
                phasm_core::GopPattern::Ibpbp { b_count, .. } => {
                    phasm_core::GopPattern::Ibpbp { gop: gop_size, b_count }
                }
            }
        };
        eprintln!("Auto-selected GopPattern: {pattern:?}");

        // D'.5 — report cascade tier. For `Auto`, resolve here so the
        // user sees what the encoder picked (matches mobile success-
        // screen "Quality mode: High Quality (auto)" UX).
        report_cascade_tier(cascade_tier, cascade_headroom);
        #[cfg(feature = "openh264-backend")]
        if matches!(cascade_tier, phasm_core::CascadeTier::Auto) {
            report_resolved_auto_tier(
                &yuv_bytes, probe.width, probe.height, probe.n_frames,
                pattern.gop_size(),
                message.len(),
                cascade_headroom,
            );
        }

        let annex_b = h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files_with_tier(
            &yuv_bytes, probe.width, probe.height, probe.n_frames, pattern,
            message, files, passphrase,
            cascade_tier, cascade_headroom,
        )?;
        std::fs::write(&annex_b_temp, &annex_b)?;
        let audio_source = probe.has_audio.then_some(input.as_path());
        let dropped_audio = transcode::mux_annexb_to_mp4_with_profile(
            &annex_b_temp, audio_source, &probe.frame_rate,
            probe.width, probe.height, probe.n_frames, pattern,
            mux_profile, output,
        )?;
        if dropped_audio {
            eprintln!(
                "warning: chosen --mux-profile dropped audio from input.",
            );
        }
        Ok(())
    })();

    for p in cleanup_paths.drain(..) {
        let _ = std::fs::remove_file(p);
    }
    result
}

/// Task #120 — CABAC v2 + N shadows + per-layer file attachments.
/// Mirrors image stego's `--m2..m9 / --p2..p9 / --attach2..attach9`
/// surface for video. Routes through
/// `h264_stego_encode_yuv_string_with_n_shadows_with_pattern_and_files`
/// with `GopPattern::Ibpbp { gop, b_count: 1 }` (Apple-iPhone canonical).
#[cfg(feature = "cabac-stego")]
#[allow(clippy::too_many_arguments)]
fn run_cabac_encode_with_shadows(
    input: &PathBuf,
    primary_message: &str,
    primary_files: &[FileEntry],
    primary_passphrase: &str,
    shadows: &[ShadowLayer],
    gop_override: Option<usize>,
    mux_profile: transcode::MuxProfile,
    output: &PathBuf,
    // VID-OPT (2026-05-24) — wiring only; see sister run_cabac_encode.
    optimize: bool,
) -> Result<(), CliError> {
    let _ = optimize; // TODO: thread to encoder when video optimizer ships
    use phasm_core::stego::shadow_layer::ShadowLayer as VideoShadowLayer;

    transcode::ensure_ffmpeg_available()?;
    let probe = transcode::probe_video(input)?;
    eprintln!(
        "Encoding via CABAC v2 + {} shadow{}: {}x{} × {} frames @ {} fps{}",
        shadows.len(),
        if shadows.len() == 1 { "" } else { "s" },
        probe.width, probe.height, probe.n_frames, probe.frame_rate,
        if probe.has_audio { " (with audio)" } else { "" },
    );

    let gop_size = resolve_gop_size(gop_override, probe.n_frames)?;
    // §Stealth.L3.2 — source-adaptive GOP selection. If the source
    // clip is H.264 with detectable cadence, mimic it. HEVC/AV1
    // sources / unreadable input fall through to the
    // HandBrake/x264-medium centroid (the strategy doc default).
    // gop-size override still wins over auto-detected GOP.
    let pattern = {
        let detected = std::fs::read(input)
            .ok()
            .as_deref()
            .map(|b| phasm_core::GopPattern::auto_select(Some(b)))
            .unwrap_or_else(phasm_core::GopPattern::handbrake_x264_centroid);
        // Substitute caller's gop_size override for the detected one
        // while preserving the detected B-frame structure.
        match detected {
            phasm_core::GopPattern::Ipppp { .. } => {
                phasm_core::GopPattern::Ipppp { gop: gop_size }
            }
            phasm_core::GopPattern::Ibpbp { b_count, .. } => {
                phasm_core::GopPattern::Ibpbp { gop: gop_size, b_count }
            }
        }
    };
    eprintln!("Auto-selected GopPattern: {pattern:?}");

    let yuv_temp = transcode::temp_path_with_ext(input, "yuv");
    let annex_b_temp = transcode::temp_path_with_ext(input, "h264");
    let mut cleanup_paths = vec![yuv_temp.clone(), annex_b_temp.clone()];

    let result = (|| -> Result<(), CliError> {
        transcode::decode_to_yuv(input, &yuv_temp)?;
        let yuv_bytes = std::fs::read(&yuv_temp)?;
        let expected = (probe.width as usize) * (probe.height as usize) * 3 / 2 * probe.n_frames;
        if yuv_bytes.len() != expected {
            return Err(CliError::InvalidArgs(format!(
                "ffmpeg-decoded YUV is {} bytes; expected {} ({}x{} × 1.5 × {})",
                yuv_bytes.len(), expected, probe.width, probe.height, probe.n_frames,
            )));
        }
        // Convert public owned `ShadowLayer` to the internal borrowed
        // `shadow_layer::ShadowLayer<'a>` expected by the video API.
        let video_shadows: Vec<VideoShadowLayer> = shadows
            .iter()
            .map(|s| VideoShadowLayer {
                message: &s.message,
                passphrase: &s.passphrase,
                files: &s.files,
            })
            .collect();
        let annex_b = h264_stego_encode_yuv_string_with_n_shadows_with_pattern_and_files(
            &yuv_bytes, probe.width, probe.height, probe.n_frames, pattern,
            primary_message, primary_files, primary_passphrase,
            &video_shadows,
        )?;
        std::fs::write(&annex_b_temp, &annex_b)?;
        let audio_source = probe.has_audio.then_some(input.as_path());
        let dropped_audio = transcode::mux_annexb_to_mp4_with_profile(
            &annex_b_temp, audio_source, &probe.frame_rate,
            probe.width, probe.height, probe.n_frames, pattern,
            mux_profile, output,
        )?;
        if dropped_audio {
            eprintln!(
                "warning: chosen --mux-profile dropped audio from input.",
            );
        }
        Ok(())
    })();

    for p in cleanup_paths.drain(..) {
        let _ = std::fs::remove_file(p);
    }
    result
}

/// D.0.1 — production OpenH264-backend stego encode path.
///
/// MP4 → ffmpeg → YUV → `openh264_stego_encode_yuv_text` (Phase C.8
/// visual_recon) → Annex-B → ffmpeg copy-mux → MP4.
///
/// Sister of `run_cabac_encode`. Uses the same ffmpeg-based demux +
/// remux infrastructure (`transcode::probe_video`,
/// `transcode::decode_to_yuv`), but the encode step routes through
/// `phasm_core::codec::h264::openh264_stego` instead of the pure-Rust
/// multigop streaming v2 orchestrator — 1000-4000× faster (see
/// `docs/design/video/h264/phase-c8-visual-recon-plan.md` §C.8.14).
///
/// v1.0 limitations:
///  - single-domain (CoeffSign only); other 3 domains stay opt-in.
///  - no file attachments yet (mirrors current `openh264_stego_encode_
///    yuv_text` surface). Shadows + attachments arrive with D.0.4.
///  - mux is a simple ffmpeg `-c copy` for now (no L4 stealth profile);
///    HandBrake/x264 mux profile parity tracked as v1.1 polish.
#[cfg(feature = "openh264-backend")]
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

        // CAP2.2 §12 — proportional allocation (CLI two-pass). `yuv_bytes` holds
        // the whole clip (re-readable), so run a capacity probe FIRST to learn
        // each GOP's cap, then size each GOP's chunk PROPORTIONAL to its cap
        // (uniform flip density — stealthier than the default online
        // uniform-BYTES spread, which over-fills low-capacity GOPs into
        // detectability hotspots). The per-GOP round-trip verify + shrink-carry
        // inside the encode is the backstop, so an over-optimistic plan degrades
        // to carry-forward (never data loss). This doubles encode wall-clock;
        // the §13 easy-fit fast path (skip the probe for tiny payloads) is a
        // perf follow-on.
        // CAP2.3: concentrate+tail at the shared stealth-leaning default
        // (0.5, provisional — #806 calibrates). Supersedes the 1.0
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

#[cfg(not(feature = "cabac-stego"))]
fn run_cavlc_encode(
    input: &PathBuf,
    file_bytes: &[u8],
    message: &str,
    passphrase: &str,
    output: &PathBuf,
) -> Result<(), CliError> {
    let mut transcode_tempfile: Option<PathBuf> = None;
    match h264_ghost_encode(file_bytes, message, passphrase) {
        Ok(stego) => {
            std::fs::write(output, &stego)?;
        }
        Err(e) if transcode::is_non_baseline_error(&e) => {
            transcode::ensure_ffmpeg_available()?;
            let temp = transcode::temp_transcode_path(input);
            eprintln!(
                "Input is H.264 but not Baseline CAVLC — transcoding via ffmpeg..."
            );
            transcode::transcode_to_baseline(input, &temp)?;
            transcode_tempfile = Some(temp.clone());
            let transcoded = std::fs::read(&temp)?;
            let stego = h264_ghost_encode(&transcoded, message, passphrase)?;
            std::fs::write(output, &stego)?;
        }
        Err(e) => return Err(CliError::from(e)),
    }
    if let Some(temp) = transcode_tempfile {
        let _ = std::fs::remove_file(&temp);
    }
    Ok(())
}

/// Task #120 — assemble shadow layers from CLI args. Mirrors the
/// image-stego `collect_shadows` in `encode.rs`. Accepts both the
/// legacy `--shadow-msg / --shadow-pass` pair (treated as `--m2 /
/// --p2`) and the new `--m2..m9 / --p2..p9 / --attach2..attach9`
/// flags. Reorders by compressed payload size (smallest first) so
/// shadow-priority hashing is stable across reruns.
#[cfg(feature = "cabac-stego")]
fn collect_shadows(args: &VideoEncodeArgs) -> Result<Vec<ShadowLayer>, CliError> {
    use phasm_core::compressed_payload_size;

    // Legacy --shadow-msg / --shadow-pass become an implicit slot 2.
    let legacy_msg = args.shadow_message.clone();
    let legacy_pass = args.shadow_passphrase.clone();
    let m2_effective = args.m2.clone().or(legacy_msg);
    let p2_effective = args.p2.clone().or(legacy_pass);
    if m2_effective.is_some() != p2_effective.is_some()
        && (args.shadow_message.is_some() || args.shadow_passphrase.is_some())
    {
        // One of --shadow-msg / --shadow-pass set without the other.
        return Err(CliError::InvalidArgs(
            "--shadow-msg and --shadow-pass must both be set together".into(),
        ));
    }

    let shadow_msgs: Vec<Option<&String>> = vec![
        m2_effective.as_ref(), args.m3.as_ref(), args.m4.as_ref(), args.m5.as_ref(),
        args.m6.as_ref(), args.m7.as_ref(), args.m8.as_ref(), args.m9.as_ref(),
    ];
    let shadow_passes: Vec<Option<&String>> = vec![
        p2_effective.as_ref(), args.p3.as_ref(), args.p4.as_ref(), args.p5.as_ref(),
        args.p6.as_ref(), args.p7.as_ref(), args.p8.as_ref(), args.p9.as_ref(),
    ];
    let shadow_attaches: Vec<&Vec<PathBuf>> = vec![
        &args.attach2, &args.attach3, &args.attach4, &args.attach5,
        &args.attach6, &args.attach7, &args.attach8, &args.attach9,
    ];

    let mut shadows = Vec::new();
    for i in 0..8 {
        if let Some(msg) = shadow_msgs[i] {
            let label = format!("Shadow {} passphrase", i + 2);
            let pass = crate::passphrase::get_passphrase(
                shadow_passes[i].map(|s| s.as_str()),
                &label,
                true,
            )?;
            let files = load_files(shadow_attaches[i])?;
            shadows.push(ShadowLayer {
                message: msg.clone(),
                passphrase: pass,
                files,
            });
        }
    }

    shadows.sort_by_key(|s| compressed_payload_size(&s.message, &s.files));
    Ok(shadows)
}

/// #98 — resolve `--gop-size N` override against the probed
/// `n_frames`. Defaults to 30 (1 s @ 30 fps phone-capture
/// cadence). Validates `1 <= gop_size <= n_frames`; returns a
/// `CliError::InvalidArgs` with a clear message on out-of-range.
#[cfg(feature = "cabac-stego")]
fn resolve_gop_size(override_: Option<usize>, n_frames: usize) -> Result<usize, CliError> {
    const DEFAULT_GOP_SIZE: usize = 30;
    let n_frames = n_frames.max(1);
    let gop = override_.unwrap_or(DEFAULT_GOP_SIZE);
    if gop == 0 {
        return Err(CliError::InvalidArgs(
            "--gop-size must be >= 1".into(),
        ));
    }
    if gop > n_frames {
        return Err(CliError::InvalidArgs(format!(
            "--gop-size {gop} exceeds video length ({n_frames} frames). \
             Use a value <= {n_frames} or omit the flag for default {DEFAULT_GOP_SIZE}."
        )));
    }
    Ok(gop)
}

#[cfg(feature = "cabac-stego")]
fn load_files(paths: &[PathBuf]) -> Result<Vec<FileEntry>, CliError> {
    let mut entries = Vec::new();
    for path in paths {
        let content = std::fs::read(path)?;
        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "file".to_string());
        entries.push(FileEntry { filename, content });
    }
    Ok(entries)
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
