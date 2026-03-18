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

use clap::{Parser, Subcommand};
use std::process;

#[derive(Parser)]
#[command(name = "phasm", version, about = "Steganography — hide encrypted messages in photos")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode a secret message into a cover image
    Encode(encode::EncodeArgs),
    /// Decode a secret message from a stego image
    Decode(decode::DecodeArgs),
    /// Show how much message data fits in an image
    Capacity(capacity::CapacityArgs),
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Encode(args) => encode::run(args),
        Commands::Decode(args) => decode::run(args),
        Commands::Capacity(args) => capacity::run(args),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        process::exit(e.exit_code());
    }
}
