// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

use phasm_core::{DecodeQuality, EncodeQuality, PayloadData};
use std::time::Duration;

/// Output mode determined by CLI flags.
#[derive(Clone, Copy)]
pub enum OutputMode {
    Default,
    Quiet,
    Verbose,
    Json,
}

// ── Encode output ──

pub fn print_encode_result(
    output_path: &str,
    quality: &EncodeQuality,
    deep_cover: bool,
    mode_name: &str,
    elapsed: Duration,
    mode: OutputMode,
) {
    match mode {
        OutputMode::Quiet => println!("{output_path}"),
        OutputMode::Json => {
            let dc = if deep_cover { "true" } else { "false" };
            println!(
                "{{\"output\":\"{}\",\"mode\":\"{}\",\"deepCover\":{},\"score\":{},\"hint\":\"{}\"}}",
                json_escape(output_path),
                json_escape(mode_name),
                dc,
                quality.score,
                json_escape(&quality.hint_key),
            );
        }
        OutputMode::Verbose => {
            let dc_label = if deep_cover { " (Deep Cover)" } else { "" };
            println!("Encoded: {output_path}");
            println!("Mode: {mode_name}{dc_label}");
            let label = if quality.mode == 1 { "Stealth" } else { "Robustness" };
            println!("{label}: {}% — {}", quality.score, score_word(quality.score));
            println!("Hint: {}", quality.hint_key);
            println!("Time: {:.1}s", elapsed.as_secs_f64());
        }
        OutputMode::Default => {
            let dc_label = if deep_cover { " (Deep Cover)" } else { "" };
            println!("Encoded: {output_path}");
            println!("Mode: {mode_name}{dc_label}");
            let label = if quality.mode == 1 { "Stealth" } else { "Robustness" };
            println!("{label}: {}% — {}", quality.score, score_word(quality.score));
        }
    }
}

// ── Decode output ──

pub fn print_decode_result(
    payload: &PayloadData,
    quality: &DecodeQuality,
    extract_dir: Option<&str>,
    elapsed: Duration,
    mode: OutputMode,
) {
    match mode {
        OutputMode::Quiet => print!("{}", payload.text),
        OutputMode::Json => {
            let mode_name = decode_mode_name(quality);
            let mut json = format!(
                "{{\"message\":\"{}\",\"mode\":\"{}\",\"integrity\":{}",
                json_escape(&payload.text),
                json_escape(mode_name),
                quality.integrity_percent,
            );
            if quality.fortress_used {
                json.push_str(",\"fortress\":true");
            }
            if quality.rs_errors_corrected > 0 {
                json.push_str(&format!(
                    ",\"rsErrors\":{},\"rsCapacity\":{}",
                    quality.rs_errors_corrected, quality.rs_error_capacity
                ));
            }
            if !payload.files.is_empty() {
                json.push_str(",\"files\":[");
                for (i, f) in payload.files.iter().enumerate() {
                    if i > 0 {
                        json.push(',');
                    }
                    json.push_str(&format!(
                        "{{\"name\":\"{}\",\"size\":{}}}",
                        json_escape(&f.filename),
                        f.content.len()
                    ));
                }
                json.push(']');
            }
            json.push('}');
            println!("{json}");
        }
        OutputMode::Verbose => {
            println!("{}", payload.text);
            println!();
            let mode_name = decode_mode_name(quality);
            println!("Mode: {mode_name}");
            println!("Integrity: {}%", quality.integrity_percent);
            if quality.fortress_used {
                println!("Fortress: yes");
            }
            if quality.rs_errors_corrected > 0 {
                println!(
                    "RS errors: {}/{}",
                    quality.rs_errors_corrected, quality.rs_error_capacity
                );
            }
            println!("Time: {:.1}s", elapsed.as_secs_f64());
            print_attachments(&payload.files, extract_dir);
        }
        OutputMode::Default => {
            println!("{}", payload.text);
            print_attachments(&payload.files, extract_dir);
        }
    }
}

fn print_attachments(files: &[phasm_core::FileEntry], extract_dir: Option<&str>) {
    if files.is_empty() {
        return;
    }
    println!();
    match extract_dir {
        Some(dir) => {
            for f in files {
                let path = format!("{}/{}", dir, f.filename);
                println!("Extracted: {} ({})", path, format_size(f.content.len()));
            }
        }
        None => {
            println!("Attachments (use --extract <dir> to save):");
            for f in files {
                println!("  {} ({})", f.filename, format_size(f.content.len()));
            }
        }
    }
}

// ── Capacity output ──

pub fn print_capacity(
    ghost: usize,
    ghost_si: usize,
    armor: usize,
    fortress: usize,
    shadow: usize,
    mode_filter: Option<&str>,
    json: bool,
) {
    if json {
        println!(
            "{{\"ghost\":{ghost},\"ghostDeepCover\":{ghost_si},\"armor\":{armor},\"fortress\":{fortress},\"shadow\":{shadow}}}"
        );
        return;
    }
    match mode_filter {
        Some("ghost") => {
            println!("Ghost:   {ghost} bytes (Deep Cover: {ghost_si} bytes)");
            println!("Shadows: {shadow} bytes per layer");
        }
        Some("armor") => {
            println!("Armor:   {armor} bytes (Fortress: {fortress} bytes)");
        }
        _ => {
            println!("Ghost:   {ghost} bytes (Deep Cover: {ghost_si} bytes)");
            println!("Armor:   {armor} bytes (Fortress: {fortress} bytes)");
            println!("Shadows: {shadow} bytes per layer");
        }
    }
}

// ── Helpers ──

fn score_word(score: u8) -> &'static str {
    match score {
        90..=100 => "Excellent",
        70..=89 => "Good",
        50..=69 => "Fair",
        _ => "Low",
    }
}

fn decode_mode_name(q: &DecodeQuality) -> &'static str {
    if q.fortress_used {
        "Armor (Fortress)"
    } else if q.mode == 1 {
        "Ghost"
    } else {
        "Armor"
    }
}

fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}
