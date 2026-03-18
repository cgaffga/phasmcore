// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::output::{self, OutputMode};
use crate::passphrase::get_passphrase;
use crate::progress::spawn_progress_bar;
use phasm_core::smart_decode;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

#[derive(Parser)]
pub struct DecodeArgs {
    /// Stego image file
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
    let image_bytes = std::fs::read(&args.image)?;
    let passphrase = get_passphrase(args.passphrase.as_deref(), "Passphrase", false)?;

    let progress_handle = if args.progress {
        Some(spawn_progress_bar())
    } else {
        None
    };

    let start = Instant::now();
    let (payload, quality) = smart_decode(&image_bytes, &passphrase)?;
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
