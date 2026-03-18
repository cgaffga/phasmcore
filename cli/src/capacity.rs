// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::output;
use phasm_core::{
    armor_capacity, armor_capacity_info, estimate_shadow_capacity, ghost_capacity,
    ghost_capacity_si, JpegImage,
};
use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
pub struct CapacityArgs {
    /// Image file
    pub image: PathBuf,

    /// Show specific mode only (ghost or armor)
    #[arg(long)]
    pub mode: Option<String>,

    /// JSON output
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: CapacityArgs) -> Result<(), CliError> {
    let jpeg_bytes = std::fs::read(&args.image)?;
    let img = JpegImage::from_bytes(&jpeg_bytes)?;

    let ghost = ghost_capacity(&img)?;
    let ghost_si = ghost_capacity_si(&img)?;
    let armor = armor_capacity(&img)?;
    let shadow = estimate_shadow_capacity(&img)?;

    let fortress = match armor_capacity_info(&jpeg_bytes) {
        Ok(info) => info.fortress_capacity,
        Err(_) => 0,
    };

    output::print_capacity(
        ghost,
        ghost_si,
        armor,
        fortress,
        shadow,
        args.mode.as_deref(),
        args.json,
    );

    Ok(())
}
