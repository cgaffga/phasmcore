// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::output::{self, OutputMode};
use crate::passphrase::get_passphrase;
use crate::progress::spawn_progress_bar;
use phasm_core::smart_decode;
#[cfg(feature = "video")]
use phasm_core::{detect_video_codec, is_mp4, DecodeQuality, VideoCodec};
#[cfg(all(feature = "video", not(feature = "cabac-stego")))]
use phasm_core::h264_ghost_decode;
#[cfg(feature = "cabac-stego")]
use phasm_core::h264_stego_smart_decode_video_with_payload;
#[cfg(feature = "cabac-stego")]
use crate::transcode;
use std::path::PathBuf;
use std::time::Instant;

/// v0.2.9 — `is_mp4` lives behind the `video` feature on phasm-core,
/// so when this binary is built image-only we need a tiny local
/// magic-byte sniff to give a clean "rebuild with --features video"
/// error instead of letting `smart_decode` fail with a confusing
/// not-a-valid-JPEG error.
#[cfg(not(feature = "video"))]
fn looks_like_mp4(bytes: &[u8]) -> bool {
    // ISO BMFF: bytes 4..8 are the `ftyp` box type marker.
    bytes.len() >= 12 && &bytes[4..8] == b"ftyp"
}

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
    //
    // v0.2.9 — when this binary is built without `video`, video-stego
    // decoding is unreachable code; we sniff the MP4 magic and return a
    // clean rebuild-instruction error instead of letting smart_decode
    // emit a confusing "not a valid JPEG" message.
    #[cfg(not(feature = "video"))]
    if looks_like_mp4(&file_bytes) {
        return Err(CliError::UnsupportedFormat(
            "MP4 / video stego decoding is not built into this binary. \
             The public CLI release ships image-stego-only for AVC patent \
             reasons. To decode video stego, rebuild from source: \
             `cargo install phasm-cli --features video`."
                .into(),
        ));
    }
    let (payload, quality) = {
        #[cfg(feature = "video")]
        {
            if is_mp4(&file_bytes) {
                let payload = match detect_video_codec(&file_bytes) {
                    VideoCodec::H264 => {
                        #[cfg(feature = "cabac-stego")]
                        { decode_h264_cabac(&args.image, &passphrase)? }
                        #[cfg(not(feature = "cabac-stego"))]
                        { h264_ghost_decode(&file_bytes, &passphrase)? }
                    }
                    VideoCodec::Hevc => {
                        return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                            "HEVC/H.265 decode is not supported. This file was encoded outside \
                             Phasm's H.264 pipeline — no hidden message will be recoverable."
                                .into(),
                        )));
                    }
                    VideoCodec::Unknown => {
                        return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                            "unsupported or unrecognised video codec (expected H.264)".into(),
                        )));
                    }
                };
                (payload, DecodeQuality::ghost())
            } else {
                smart_decode(&file_bytes, &passphrase)?
            }
        }
        #[cfg(not(feature = "video"))]
        {
            smart_decode(&file_bytes, &passphrase)?
        }
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

/// Phase 6.5 + D.0.1 — H.264 decode dispatch. ffmpeg demuxes MP4 →
/// Annex-B; tries OpenH264-backend decode first (production v1.0
/// path), falls back to pure-Rust CABAC v2 decode (`open-h264` +
/// `rust-h264` produced files are bitstream-distinct — different
/// stego cover sets — so the right decoder must be selected per-file).
///
/// Auto-detection rationale: the user shouldn't have to remember
/// which encoder produced a file. Try OH264 first (default encoder
/// in v1.0), then CABAC v2 if that fails with a FrameCorrupted /
/// invalid-stego error. The decoders fail fast on cross-encoder
/// input — wrong-cover walks always trip CRC well before any
/// expensive work.
#[cfg(feature = "cabac-stego")]
fn decode_h264_cabac(
    input: &PathBuf,
    passphrase: &str,
) -> Result<phasm_core::PayloadData, CliError> {
    transcode::ensure_ffmpeg_available()?;
    let annex_b_temp = transcode::temp_path_with_ext(input, "h264");
    let result = (|| -> Result<phasm_core::PayloadData, CliError> {
        transcode::extract_annexb_from_mp4(input, &annex_b_temp)?;
        let annex_b = std::fs::read(&annex_b_temp)?;

        // D.0.7.13 — try the streaming decode session first. This
        // handles the new chunk_frame wire format produced by
        // `run_oh264_encode` (and by the mobile bridges). Falls
        // through to the legacy OH264 + CABAC v2 decoders on
        // failure, so files produced by older versions or by the
        // experimental `--encoder rust-h264` path still decode.
        {
            use phasm_core::StreamingDecodeSession;
            if let Ok(mut session) = StreamingDecodeSession::create(passphrase) {
                if session.push_annex_b(&annex_b).is_ok() {
                    if let Ok(res) = session.finish() {
                        // Streaming decode currently returns text +
                        // mode_id; surface as a PayloadData with no
                        // attachments. Attached-file streaming-decode
                        // recovery tracked as v1.1 polish.
                        return Ok(phasm_core::PayloadData {
                            text: res.text,
                            files: Vec::new(),
                        });
                    }
                }
            }
        }

        // D.0.1 — fall back to the legacy OpenH264-backend decoder
        // (handles older OH264-produced stego streams from before
        // the streaming chunk_frame format landed).
        #[cfg(feature = "openh264-backend")]
        {
            use phasm_core::codec::h264::openh264_stego::openh264_stego_decode_yuv;
            match openh264_stego_decode_yuv(&annex_b, passphrase) {
                Ok(payload) => return Ok(payload),
                Err(_) => {
                    // Not an OH264-produced stego stream — fall through
                    // to CABAC v2 decode.
                }
            }
        }

        Ok(h264_stego_smart_decode_video_with_payload(&annex_b, passphrase)?)
    })();
    let _ = std::fs::remove_file(&annex_b_temp);
    result
}
