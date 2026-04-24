// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

mod capacity;
mod decode;
mod encode;
mod error;
mod image_conv;
mod output;
mod passphrase;
mod progress;
mod transcode;
mod video_capacity;
mod video_encode;

use clap::{Parser, Subcommand};
use std::process;

#[derive(Parser)]
#[command(name = "phasm", version, about = "Steganography — hide encrypted messages in photos and videos")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode a secret message into a cover image
    Encode(encode::EncodeArgs),
    /// Decode a secret message from a stego image or video
    Decode(decode::DecodeArgs),
    /// Show how much message data fits in an image
    Capacity(capacity::CapacityArgs),
    /// Encode a secret message into an MP4 video (H.264 Baseline CAVLC)
    VideoEncode(video_encode::VideoEncodeArgs),
    /// Show video embedding capacity
    VideoCapacity(video_capacity::VideoCapacityArgs),
    // Phase 4a: `VideoTranscode` (HEVC pixel-domain transcode) removed.
    // Phase 4b will add an ffmpeg-based transcode flow on `VideoEncode`.
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Encode(args) => encode::run(args),
        Commands::Decode(args) => decode::run(args),
        Commands::Capacity(args) => capacity::run(args),
        Commands::VideoEncode(args) => video_encode::run(args),
        Commands::VideoCapacity(args) => video_capacity::run(args),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        process::exit(e.exit_code());
    }
}
