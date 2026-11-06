// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::image_conv::prepare_image;
use crate::output::{self, OutputMode};
use crate::passphrase::get_passphrase;
use crate::progress::spawn_progress_bar;
use phasm_core::{
    compressed_payload_size, ghost_bands, ghost_bands_si, ghost_encode_si_with_shadows_quality,
    ghost_encode_with_shadows_quality, armor_encode_with_quality,
    ghost_encode_si_with_files_quality, ghost_encode_with_files_quality,
    EncodeQuality, FileEntry, JpegImage, ShadowLayer, StegoError, ARMOR_TARGET_DIMENSION,
};
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ArgAction};

/// Ghost encodes tried (each with a fresh salt/nonce) before reporting
/// `MessageTooLarge`.
const GHOST_ENCODE_ATTEMPTS: u32 = 3;

#[derive(Parser)]
pub struct EncodeArgs {
    /// Cover image file
    pub image: PathBuf,

    /// Message text (reads from stdin if omitted in pipe mode)
    #[arg(short = 'm')]
    pub message: Option<String>,

    /// Passphrase (interactive prompt if omitted)
    #[arg(short = 'p')]
    pub passphrase: Option<String>,

    // Shadow messages (slots 2..9; primary message is `-m` above)
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

    /// File attachment for primary message (repeatable)
    #[arg(long, action = ArgAction::Append)]
    pub attach: Vec<PathBuf>,

    // Shadow attachments (slots 2..9; repeatable per slot)
    #[arg(long, action = ArgAction::Append)] pub shadow_attach_2: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub shadow_attach_3: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub shadow_attach_4: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub shadow_attach_5: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub shadow_attach_6: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub shadow_attach_7: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub shadow_attach_8: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub shadow_attach_9: Vec<PathBuf>,

    /// Stego mode: ghost or armor
    #[arg(long, default_value = "armor")]
    pub mode: String,

    /// Ghost: JPEG quality of the cover phasm writes (default 92). On a
    /// JPEG kept at its own size it forces a re-encode (a weaker cover).
    /// Armor covers are always QF65.
    #[arg(long)]
    pub qf: Option<u8>,

    /// Ghost: shrink the longest edge to at most this many pixels. Armor:
    /// replaces the channel preset with this size (QF65; not
    /// recompression-robust above 1600 px)
    #[arg(long)]
    pub resize: Option<u32>,

    /// Armor: keep the original resolution instead of the default 1600 px
    /// cover re-encoded at QF65. The message still decodes, but will not
    /// survive recompression or resizing (e.g. by messengers).
    #[arg(long, conflicts_with = "resize")]
    pub full_res: bool,

    /// Armor: the channel the message is meant for, which sets the cover
    /// size: standard (1600 px long side: WhatsApp, Signal, e-mail),
    /// telegram (1280 px) or instagram (1080 px wide)
    #[arg(long, value_parser = crate::image_conv::parse_armor_preset, default_value = "standard",
          conflicts_with_all = ["resize", "full_res"])]
    pub armor_preset: phasm_core::ArmorCoverPreset,

    /// Strip the metadata, like the apps' magic wand: a JPEG that is used as
    /// it is gets cleaned losslessly (EXIF orientation applied); a cover
    /// phasm writes gets none. Without it the output keeps the source's
    /// metadata (a written cover: its EXIF, cleaned) and `encode` lists what
    /// is kept. No pixel filter.
    #[arg(long)]
    pub optimize: bool,

    /// Output file (default: <name>.stego.jpg, use "-" for stdout)
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
}

pub fn run(args: EncodeArgs) -> Result<(), CliError> {
    let mode = match args.mode.to_lowercase().as_str() {
        "ghost" | "g" | "1" => "ghost",
        "armor" | "a" | "2" => "armor",
        other => return Err(CliError::InvalidArgs(format!("Unknown mode: {other}. Use ghost or armor."))),
    };

    // 1. Get message
    let message = get_message(&args.message)?;

    // 2. Get passphrase
    let passphrase = get_passphrase(args.passphrase.as_deref(), "Passphrase", true)?;

    // 3. Collect shadows
    let shadows = collect_shadows(&args)?;

    // 4. Load primary file attachments
    let files = load_files(&args.attach)?;

    if args.full_res && mode != "armor" {
        return Err(CliError::InvalidArgs("--full-res applies to Armor only.".into()));
    }
    if args.qf.is_some() && mode == "armor" {
        return Err(CliError::InvalidArgs(
            "--qf applies to Ghost only: Armor covers are always written at QF65 (what its robustness was measured on).".into(),
        ));
    }

    // 5. Prepare image (format conversion, resize, optimize; Armor's 1600 px cover)
    let prepared = prepare_image(
        &args.image, mode, args.qf, args.resize, args.optimize, args.full_res, args.armor_preset,
    )?;
    // Deep Cover is side-informed Ghost; Armor ignores the raw pixels. Core
    // drops the side information for pixels that are a decoded JPEG on the
    // cover's grid (e.g. a PNG saved from a JPEG at the same size).
    let decoded_jpeg = mode == "ghost"
        && prepared
            .raw_pixels
            .as_deref()
            .is_some_and(|px| phasm_core::is_aligned_jpeg_decode(px, prepared.width, prepared.height));
    let deep_cover = mode == "ghost" && prepared.raw_pixels.is_some() && !decoded_jpeg;
    if let (Some(note), false) = (&prepared.color_note, args.quiet) {
        eprintln!("note: {note}");
    }
    if mode == "ghost" && !args.quiet {
        if prepared.jpeg_reencoded_same_size {
            eprintln!(
                "warning: --qf re-encodes this JPEG at its own size, which makes a double-compressed \
                 cover; it is embedded without Deep Cover side information. Without --qf the \
                 original file is used as it is, which is the better cover."
            );
        }
        if prepared.lossless_clean {
            eprintln!(
                "note: --optimize on a JPEG: metadata removed and EXIF orientation applied without \
                 re-encoding (the original coefficients are the cover)."
            );
        }
        if decoded_jpeg {
            eprintln!(
                "note: these pixels are a decoded JPEG at its own size (same 8x8 grid), so they \
                 carry no usable side information; embedding as plain Ghost."
            );
        }
    }
    // (An Instagram cover may be taller than 1600 px; Instagram keeps it.)
    if mode == "armor" && (args.full_res || args.resize.is_some()) && prepared.exceeds_armor_target() && !args.quiet {
        let why = if args.full_res {
            "--full-res".to_string()
        } else {
            format!("--resize {}", args.resize.unwrap_or_default())
        };
        eprintln!(
            "warning: {why}: the Armor cover is {}x{}, larger than {ARMOR_TARGET_DIMENSION} px. \
             The message decodes, but will not survive recompression or resizing \
             (e.g. by messengers).",
            prepared.width, prepared.height,
        );
    }

    // Ghost: show the apps' capacity bands and warn above the red end. Display
    // only: core enforces technical feasibility, so the encode still runs.
    if mode == "ghost" {
        announce_ghost_bands(&prepared, &message, &files, &shadows, args.quiet);
    }

    // 6. Progress bar
    let progress_handle = if args.progress {
        Some(spawn_progress_bar())
    } else {
        None
    };

    // 7. Encode. A Ghost payload far above the bands' maximum (over ~2/3 of
    // the Y AC positions) leaves the trellis so little room that it can be
    // infeasible for one random salt/nonce and feasible for the next: retry
    // with fresh ones (the decoder does not care) before giving up.
    let start = Instant::now();
    let attempts = if mode == "ghost" { GHOST_ENCODE_ATTEMPTS } else { 1 };
    let mut attempt = 1;
    let (stego_bytes, quality) = loop {
        let result = do_encode(
            mode,
            &prepared.jpeg_bytes,
            prepared.raw_pixels.as_deref(),
            prepared.width,
            prepared.height,
            &message,
            &files,
            &passphrase,
            &shadows,
        );
        match result {
            Err(CliError::Stego(StegoError::MessageTooLarge)) if attempt < attempts => {
                if !args.quiet {
                    eprintln!("note: the message barely fits this cover; retrying with a fresh salt ({attempt}/{attempts})");
                }
                attempt += 1;
            }
            other => break other?,
        }
    };
    let elapsed = start.elapsed();

    // 8. Join progress thread
    if let Some(handle) = progress_handle {
        let _ = handle.join();
    }

    // 9. Write output
    let output_path = write_output(&args.image, args.output.as_deref(), &stego_bytes)?;

    // The source's metadata the output keeps (fix list R3.10: kept by
    // default, but said; `--optimize` strips it).
    let kept_metadata = crate::metadata::identifying_metadata(&stego_bytes);
    if !kept_metadata.is_empty() && !args.quiet {
        eprintln!(
            "warning: the output keeps the source's metadata: {}. Add --optimize to strip it.",
            kept_metadata.join(", ")
        );
    }

    // 10. Display result
    let out_mode = output_mode(&args);
    let mode_name = if mode == "ghost" { "Ghost" } else { "Armor" };
    let cover = output::CoverInfo {
        source: (prepared.source_width, prepared.source_height),
        encoded: (prepared.width, prepared.height),
        qf: prepared.reencoded_qf,
    };
    output::print_encode_result(&output_path, &quality, deep_cover, mode_name, &cover, &kept_metadata, elapsed, out_mode);

    Ok(())
}

/// Print the bands the apps show for this Ghost embed (stderr, like the other
/// progress lines) and warn when the primary payload is above the red end.
/// The bands are those of the cover as `encode` embeds it (Deep Cover with
/// the pixels, plain otherwise) next to the shadows, whose plaintext sizes are
/// counted the way the apps count them. A failure to measure is not reported
/// here: the encode reports its own error.
fn announce_ghost_bands(
    prepared: &crate::image_conv::PreparedImage,
    message: &str,
    files: &[FileEntry],
    shadows: &[ShadowLayer],
    quiet: bool,
) {
    if quiet {
        return;
    }
    let shadow_total: usize = shadows
        .iter()
        .map(|s| s.message.len() + 1 + s.files.iter().map(|f| f.content.len()).sum::<usize>())
        .sum();
    let bands = JpegImage::from_bytes(&prepared.jpeg_bytes).ok().and_then(|img| {
        match &prepared.raw_pixels {
            Some(px) => ghost_bands_si(&img, px, prepared.width, prepared.height, shadows.len(), shadow_total),
            None => ghost_bands(&img, shadows.len(), shadow_total),
        }
        .ok()
    });
    let Some(b) = bands else { return };
    let payload = compressed_payload_size(message, files);
    eprintln!("  max: {} B   comfort: {} B   payload: {payload} B", b.max_bytes, b.comfort_bytes);
    warn_over_max(payload, b.max_bytes);
}

/// The CLI shows the capacity bands but does not enforce them: a payload
/// above the red end still encodes, with a warning on stderr.
fn warn_over_max(payload: usize, max: usize) {
    if payload <= max {
        return;
    }
    eprintln!(
        "warning: the payload ({payload} B) exceeds the recommended maximum for this image \
         ({max} B); expect a detectable embedding. `phasm capacity` shows the bands."
    );
}

fn get_message(flag: &Option<String>) -> Result<String, CliError> {
    if let Some(m) = flag {
        return Ok(m.clone());
    }
    // Try reading from stdin if it's a pipe
    if !io::stdin().is_terminal() {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        // Trim trailing newline from pipe
        let trimmed = buf.trim_end_matches('\n').trim_end_matches('\r');
        if trimmed.is_empty() {
            return Err(CliError::NoMessage);
        }
        return Ok(trimmed.to_string());
    }
    Err(CliError::NoMessage)
}

fn collect_shadows(args: &EncodeArgs) -> Result<Vec<ShadowLayer>, CliError> {
    let shadow_msgs: Vec<Option<&String>> = vec![
        args.shadow_message_2.as_ref(), args.shadow_message_3.as_ref(),
        args.shadow_message_4.as_ref(), args.shadow_message_5.as_ref(),
        args.shadow_message_6.as_ref(), args.shadow_message_7.as_ref(),
        args.shadow_message_8.as_ref(), args.shadow_message_9.as_ref(),
    ];
    let shadow_passes: Vec<Option<&String>> = vec![
        args.shadow_passphrase_2.as_ref(), args.shadow_passphrase_3.as_ref(),
        args.shadow_passphrase_4.as_ref(), args.shadow_passphrase_5.as_ref(),
        args.shadow_passphrase_6.as_ref(), args.shadow_passphrase_7.as_ref(),
        args.shadow_passphrase_8.as_ref(), args.shadow_passphrase_9.as_ref(),
    ];
    let shadow_attaches: Vec<&Vec<PathBuf>> = vec![
        &args.shadow_attach_2, &args.shadow_attach_3,
        &args.shadow_attach_4, &args.shadow_attach_5,
        &args.shadow_attach_6, &args.shadow_attach_7,
        &args.shadow_attach_8, &args.shadow_attach_9,
    ];

    let mut shadows = Vec::new();
    for i in 0..8 {
        if let Some(msg) = shadow_msgs[i] {
            let label = format!("Shadow {} passphrase", i + 2);
            let pass = get_passphrase(shadow_passes[i].map(|s| s.as_str()), &label, true)?;
            let files = load_files(shadow_attaches[i])?;
            shadows.push(ShadowLayer {
                message: msg.clone(),
                passphrase: pass,
                files,
            });
        }
    }

    // Sort by compressed payload size (smallest first) — matches mobile app
    shadows.sort_by_key(|s| compressed_payload_size(&s.message, &s.files));

    Ok(shadows)
}

pub(crate) fn load_files(paths: &[PathBuf]) -> Result<Vec<FileEntry>, CliError> {
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

#[allow(clippy::too_many_arguments)]
fn do_encode(
    mode: &str,
    jpeg_bytes: &[u8],
    raw_pixels: Option<&[u8]>,
    width: u32,
    height: u32,
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    shadows: &[ShadowLayer],
) -> Result<(Vec<u8>, EncodeQuality), CliError> {
    let has_shadows = !shadows.is_empty();
    let has_si = raw_pixels.is_some();

    match mode {
        "ghost" if has_shadows && has_si => {
            let px = raw_pixels.unwrap();
            let (bytes, q) = ghost_encode_si_with_shadows_quality(
                jpeg_bytes, px, width, height, message, files, passphrase, shadows,
            )?;
            Ok((bytes, q))
        }
        "ghost" if has_shadows => {
            let (bytes, q) = ghost_encode_with_shadows_quality(
                jpeg_bytes, message, files, passphrase, shadows, None,
            )?;
            Ok((bytes, q))
        }
        "ghost" if has_si => {
            let px = raw_pixels.unwrap();
            let (bytes, q) = ghost_encode_si_with_files_quality(
                jpeg_bytes, px, width, height, message, files, passphrase,
            )?;
            Ok((bytes, q))
        }
        "ghost" => {
            let (bytes, q) = ghost_encode_with_files_quality(jpeg_bytes, message, files, passphrase)?;
            Ok((bytes, q))
        }
        _ => {
            // Armor — no shadows, no SI, no files
            if has_shadows {
                return Err(CliError::InvalidArgs(
                    "Shadow messages are only supported in Ghost mode.".into(),
                ));
            }
            if !files.is_empty() {
                return Err(CliError::InvalidArgs(
                    "File attachments are only supported in Ghost mode.".into(),
                ));
            }
            let (bytes, q) = armor_encode_with_quality(jpeg_bytes, message, passphrase)?;
            Ok((bytes, q))
        }
    }
}

fn write_output(
    input_path: &PathBuf,
    output_flag: Option<&str>,
    stego_bytes: &[u8],
) -> Result<String, CliError> {
    match output_flag {
        Some("-") => {
            io::stdout().write_all(stego_bytes)?;
            Ok("-".to_string())
        }
        Some(path) => {
            std::fs::write(path, stego_bytes)?;
            Ok(path.to_string())
        }
        None => {
            let stem = input_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "output".to_string());
            let parent = input_path.parent().unwrap_or(std::path::Path::new("."));
            let out_path = parent.join(format!("{stem}.stego.jpg"));
            let out_str = out_path.to_string_lossy().to_string();
            std::fs::write(&out_path, stego_bytes)?;
            Ok(out_str)
        }
    }
}

fn output_mode(args: &EncodeArgs) -> OutputMode {
    if args.json {
        OutputMode::Json
    } else if args.quiet {
        OutputMode::Quiet
    } else if args.verbose {
        OutputMode::Verbose
    } else {
        OutputMode::Default
    }
}
