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
#[cfg(feature = "video")]
mod transcode;
// Encode-side video subcommands exist only when a video encoder is compiled in;
// a decode-only build (`h264-decoder`) decodes videos via `phasm decode`.
#[cfg(any(feature = "h264-encoder", feature = "av1-encoder"))]
mod video_capacity;
#[cfg(any(feature = "h264-encoder", feature = "av1-encoder"))]
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
    #[cfg_attr(feature = "video", doc = "Decode a secret message from a stego image or video")]
    #[cfg_attr(not(feature = "video"), doc = "Decode a secret message from a stego image")]
    Decode(decode::DecodeArgs),
    /// Show how much message data fits in an image
    Capacity(capacity::CapacityArgs),
    /// Encode a secret message into an MP4 video (H.264, or AV1 with --codec av1)
    #[cfg(any(feature = "h264-encoder", feature = "av1-encoder"))]
    VideoEncode(video_encode::VideoEncodeArgs),
    /// Show video embedding capacity
    #[cfg(any(feature = "h264-encoder", feature = "av1-encoder"))]
    VideoCapacity(video_capacity::VideoCapacityArgs),
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Encode(args) => encode::run(args),
        Commands::Decode(args) => decode::run(args),
        Commands::Capacity(args) => capacity::run(args),
        #[cfg(any(feature = "h264-encoder", feature = "av1-encoder"))]
        Commands::VideoEncode(args) => video_encode::run(args),
        #[cfg(any(feature = "h264-encoder", feature = "av1-encoder"))]
        Commands::VideoCapacity(args) => video_capacity::run(args),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        process::exit(e.exit_code());
    }
}
