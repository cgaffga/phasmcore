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
#[cfg(all(feature = "video", not(feature = "h264-decoder")))]
use phasm_core::h264_ghost_decode;
#[cfg(feature = "h264-decoder")]
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

    // Auto-detect format: H.264 MP4 vs JPEG image. (HEVC stego unsupported —
    // HEVC pipeline removed 2026-06-04.) Note: we do NOT auto-transcode
    // stego videos on decode — transcoding re-encodes pixels which destroys
    // the embedded message. The stego must already be CABAC High-profile
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
                        #[cfg(feature = "h264-decoder")]
                        { decode_h264_cabac(&args.image, &passphrase)? }
                        #[cfg(not(feature = "h264-decoder"))]
                        { h264_ghost_decode(&file_bytes, &passphrase)? }
                    }
                    VideoCodec::Hevc => {
                        return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                            "HEVC/H.265 decode is not supported. This file was encoded outside \
                             Phasm's H.264 pipeline — no hidden message will be recoverable."
                                .into(),
                        )));
                    }
                    VideoCodec::Av1 => {
                        #[cfg(feature = "av1-encoder")]
                        { decode_av1_streaming(&file_bytes, &passphrase)? }
                        #[cfg(not(feature = "av1-encoder"))]
                        {
                            return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                                "AV1 video stego decode is not built into this binary. \
                                 Rebuild from source: \
                                 `cargo install phasm-cli --features av1-encoder`."
                                    .into(),
                            )));
                        }
                    }
                    VideoCodec::Unknown => {
                        return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
                            "unsupported or unrecognised video codec \
                             (expected H.264 or AV1)".into(),
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

/// H.264 stego decode dispatch. ffmpeg demuxes MP4 → Annex-B, then the
/// engine-agnostic `StreamingDecodeSession` (the pure-Rust walker — the
/// sole production decoder) recovers the message. The streaming session
/// owns the v1.0 chunk_frame wire format produced by `run_oh264_encode`
/// and the mobile bridges. It fails fast on input it doesn't own
/// (wrong-cover walks trip CRC well before any expensive work).
#[cfg(feature = "h264-decoder")]
fn decode_h264_cabac(
    input: &PathBuf,
    passphrase: &str,
) -> Result<phasm_core::PayloadData, CliError> {
    transcode::ensure_ffmpeg_available()?;
    let annex_b_temp = transcode::temp_path_with_ext(input, "h264");
    let result = (|| -> Result<phasm_core::PayloadData, CliError> {
        transcode::extract_annexb_from_mp4(input, &annex_b_temp)?;
        let annex_b = std::fs::read(&annex_b_temp)?;

        // Streaming decode session: handles the chunk_frame wire format
        // produced by `run_oh264_encode` and the mobile bridges.
        use phasm_core::StreamingDecodeSession;
        let mut session = StreamingDecodeSession::create(passphrase)?;
        session.push_annex_b(&annex_b)?;
        let res = session.finish()?;
        // `DecodeSessionResult` also carries `res.files`, but this CLI path
        // does not yet surface attached files — it returns a PayloadData
        // with `text` only. Wiring `res.files` through is v1.1 polish.
        Ok(phasm_core::PayloadData {
            text: res.text,
            files: Vec::new(),
        })
    })();
    let _ = std::fs::remove_file(&annex_b_temp);
    result
}

/// AV1 stego decode dispatch.
///
/// Walks the MP4 via the in-tree `codec::mp4::demux` (no ffmpeg
/// shell-out), pulls each AV1 sample's OBU bytes from the av01 video
/// track, and streams them into `Av1StreamingDecodeSession`. Finalization
/// goes through `finish_primary_payload`, which decodes both the
/// `payload::encode_payload`-framed shape (CLI v1, mobile) AND legacy
/// plain-text streams via its internal fallback.
///
/// Returns `text + files` as a `PayloadData` — the same shape as the
/// H.264 dispatch with the file follow-on already wired through.
#[cfg(feature = "av1-encoder")]
fn decode_av1_streaming(
    file_bytes: &[u8],
    passphrase: &str,
) -> Result<phasm_core::PayloadData, CliError> {
    use phasm_core::Av1StreamingDecodeSession;
    let mp4 = phasm_core::codec::mp4::demux::demux(file_bytes).map_err(|e| {
        CliError::from(phasm_core::StegoError::InvalidVideo(format!(
            "AV1 MP4 demux failed: {e}"
        )))
    })?;
    let idx = mp4.video_track_idx.ok_or_else(|| {
        CliError::from(phasm_core::StegoError::InvalidVideo(
            "AV1 decode: no video track found in MP4".into(),
        ))
    })?;
    let track = &mp4.tracks[idx];
    if !track.is_av1() {
        return Err(CliError::from(phasm_core::StegoError::InvalidVideo(
            "AV1 decode: selected video track is not av01".into(),
        )));
    }

    let mut session = Av1StreamingDecodeSession::create(passphrase);
    // muxer-sh-in-band-fix (2026-06-29) — phasm's MP4 muxer
    // (`split_av1_into_samples`) now emits the `sequence_header_obu`
    // IN-BAND at the start of every sync (keyframe) sample. The
    // streaming decode session's `split_av1_into_gops` walker splits
    // on those in-band SHs to recover per-GOP slabs. av1C still
    // carries a redundant SH (mandatory per AV1-ISOBMFF § 2.3.1), but
    // we no longer re-inject it before each sample — that would
    // produce SH-only slabs interleaved with real GOPs and trip
    // `NoCoverPositions`. The earlier MOBOOM.T3.11 workaround is
    // retired here.
    for sample in &track.samples {
        session.push_bytes(&sample.data);
    }
    let payload = session.finish_primary_payload().map_err(|e| {
        CliError::from(phasm_core::StegoError::InvalidVideo(format!(
            "AV1 decode failed: {e:?}"
        )))
    })?;
    Ok(payload)
}
