// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::image_conv::prepare_image;
use crate::output::{self, OutputMode};
use crate::passphrase::get_passphrase;
use crate::progress::spawn_progress_bar;
use phasm_core::{
    compressed_payload_size, ghost_encode_si_with_shadows_quality,
    ghost_encode_with_shadows_quality, armor_encode_with_quality,
    ghost_encode_si_with_files_quality, ghost_encode_with_files_quality,
    EncodeQuality, FileEntry, ShadowLayer,
};
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ArgAction};

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

    // Shadow messages (m2..m9)
    #[arg(long)] pub m2: Option<String>,
    #[arg(long)] pub m3: Option<String>,
    #[arg(long)] pub m4: Option<String>,
    #[arg(long)] pub m5: Option<String>,
    #[arg(long)] pub m6: Option<String>,
    #[arg(long)] pub m7: Option<String>,
    #[arg(long)] pub m8: Option<String>,
    #[arg(long)] pub m9: Option<String>,

    // Shadow passphrases (p2..p9)
    #[arg(long)] pub p2: Option<String>,
    #[arg(long)] pub p3: Option<String>,
    #[arg(long)] pub p4: Option<String>,
    #[arg(long)] pub p5: Option<String>,
    #[arg(long)] pub p6: Option<String>,
    #[arg(long)] pub p7: Option<String>,
    #[arg(long)] pub p8: Option<String>,
    #[arg(long)] pub p9: Option<String>,

    /// File attachment for primary message (repeatable)
    #[arg(long, action = ArgAction::Append)]
    pub attach: Vec<PathBuf>,

    // Shadow attachments (attach2..attach9)
    #[arg(long, action = ArgAction::Append)] pub attach2: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub attach3: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub attach4: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub attach5: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub attach6: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub attach7: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub attach8: Vec<PathBuf>,
    #[arg(long, action = ArgAction::Append)] pub attach9: Vec<PathBuf>,

    /// Stego mode: ghost or armor
    #[arg(long, default_value = "armor")]
    pub mode: String,

    /// JPEG quality factor (default: 92 Ghost, 65 Armor)
    #[arg(long)]
    pub qf: Option<u8>,

    /// Resize longest edge to max pixels
    #[arg(long)]
    pub resize: Option<u32>,

    /// Run texture-adaptive cover optimizer
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

    // 5. Prepare image (format conversion, resize, optimize)
    let prepared = prepare_image(&args.image, mode, args.qf, args.resize, args.optimize)?;
    let deep_cover = prepared.raw_pixels.is_some();

    // 6. Progress bar
    let progress_handle = if args.progress {
        Some(spawn_progress_bar())
    } else {
        None
    };

    // 7. Encode
    let start = Instant::now();
    let (stego_bytes, quality) = do_encode(
        mode,
        &prepared.jpeg_bytes,
        prepared.raw_pixels.as_deref(),
        prepared.width,
        prepared.height,
        &message,
        &files,
        &passphrase,
        &shadows,
    )?;
    let elapsed = start.elapsed();

    // 8. Join progress thread
    if let Some(handle) = progress_handle {
        let _ = handle.join();
    }

    // 9. Write output
    let output_path = write_output(&args.image, args.output.as_deref(), &stego_bytes)?;

    // 10. Display result
    let out_mode = output_mode(&args);
    let mode_name = if mode == "ghost" { "Ghost" } else { "Armor" };
    output::print_encode_result(&output_path, &quality, deep_cover, mode_name, elapsed, out_mode);

    Ok(())
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
        args.m2.as_ref(), args.m3.as_ref(), args.m4.as_ref(), args.m5.as_ref(),
        args.m6.as_ref(), args.m7.as_ref(), args.m8.as_ref(), args.m9.as_ref(),
    ];
    let shadow_passes: Vec<Option<&String>> = vec![
        args.p2.as_ref(), args.p3.as_ref(), args.p4.as_ref(), args.p5.as_ref(),
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
