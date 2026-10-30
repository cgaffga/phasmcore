// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use crate::error::CliError;
use crate::image_conv::prepare_image;
use crate::output;
use phasm_core::{
    armor_capacity, armor_capacity_info, estimate_shadow_capacity, estimate_shadow_capacity_si,
    ghost_bands, ghost_bands_si, ghost_capacity, ghost_capacity_si, JpegImage,
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

    /// Armor: measure the original resolution instead of the 1600 px cover
    /// `encode` uses by default
    #[arg(long)]
    pub full_res: bool,

    /// JSON output
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: CapacityArgs) -> Result<(), CliError> {
    // Ghost is measured on the cover `encode --mode ghost` embeds into: a
    // JPEG as is; any other input as the Deep Cover cover phasm writes from
    // its pixels, whose exact side information earns the larger maximum
    // (nzAC / 4 instead of nzAC / 5 frame bits).
    let ghost_cover = prepare_image(&args.image, "ghost", None, None, false, false)?;
    let img = JpegImage::from_bytes(&ghost_cover.jpeg_bytes)?;
    let ghost = ghost_capacity(&img)?;
    // The bands the apps show (no shadows): for a Deep Cover cover, with the
    // pixels, so core grants the exact-SI maximum. Display only; `encode`
    // does not enforce them.
    let (ghost_si, shadow, bands) = match &ghost_cover.raw_pixels {
        Some(px) => {
            let (w, h) = (ghost_cover.width, ghost_cover.height);
            (
                ghost_capacity_si(&img, px, w, h)?,
                estimate_shadow_capacity_si(&img, px, w, h)?,
                ghost_bands_si(&img, px, w, h, 0, 0)?,
            )
        }
        None => (ghost, estimate_shadow_capacity(&img)?, ghost_bands(&img, 0, 0)?),
    };

    // Armor is measured on the cover `encode` would actually embed into.
    let cover = prepare_image(&args.image, "armor", None, None, false, args.full_res)?;
    let armor_img = JpegImage::from_bytes(&cover.jpeg_bytes)?;
    let armor = armor_capacity(&armor_img)?;
    let fortress = match armor_capacity_info(&cover.jpeg_bytes) {
        Ok(info) => info.fortress_capacity,
        Err(_) => 0,
    };

    output::print_capacity(
        ghost,
        ghost_si,
        &bands,
        ghost_cover.raw_pixels.is_some(),
        armor,
        fortress,
        shadow,
        (cover.width, cover.height),
        args.mode.as_deref(),
        args.json,
    );

    Ok(())
}
