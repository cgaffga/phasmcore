// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use indicatif::{ProgressBar, ProgressStyle};
use std::thread;
use std::time::Duration;

/// Spawn a background thread that polls phasm_core::progress::get() and
/// renders an indicatif progress bar. Returns a handle to join when done.
pub fn spawn_progress_bar() -> thread::JoinHandle<()> {
    thread::spawn(|| {
        let pb = ProgressBar::new(0);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len}")
                .unwrap()
                .progress_chars("=>-"),
        );

        loop {
            let (step, total) = phasm_core::progress::get();
            if total > 0 {
                pb.set_length(total as u64);
                pb.set_position(step as u64);
            }
            if total > 0 && step >= total {
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }

        pb.finish_and_clear();
    })
}
