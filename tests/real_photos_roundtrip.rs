// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Auto-discovery round-trip tests for real JPEG photos.
//!
//! Scans `tests/real_photos/` for `.jpg`/`.jpeg` files and runs Ghost + Armor
//! encode/decode round-trips on each. Passes gracefully if the directory is
//! empty or missing (0 iterations). Skips photos that fail to parse.

use phasm_core::{ghost_encode, ghost_decode, armor_encode, armor_decode, JpegImage};
use std::path::Path;

fn discover_photos() -> Vec<std::path::PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/real_photos");
    if !dir.exists() {
        return Vec::new();
    }
    let mut photos: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("jpg" | "jpeg")
            )
        })
        .collect();
    photos.sort();
    photos
}

#[test]
fn ghost_roundtrip_real_photos() {
    let photos = discover_photos();
    eprintln!("Ghost round-trip: found {} photos", photos.len());
    for photo in &photos {
        let name = photo.file_name().unwrap().to_string_lossy();
        let jpeg_bytes = std::fs::read(photo).unwrap();
        let image = match JpegImage::from_bytes(&jpeg_bytes) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("  {name}: skipped (parse error: {e:?})");
                continue;
            }
        };
        let cap = match phasm_core::ghost_capacity(&image) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  {name}: skipped (capacity error: {e:?})");
                continue;
            }
        };
        let msg = "Ghost test message";
        if msg.len() > cap {
            eprintln!("  {name}: skipped (capacity {cap} < {})", msg.len());
            continue;
        }
        let stego = ghost_encode(&jpeg_bytes, msg, "test-pass").unwrap();
        let decoded = ghost_decode(&stego, "test-pass").unwrap();
        assert_eq!(decoded.text, msg, "Ghost mismatch on {name}");
        eprintln!("  {name}: OK (capacity {cap})");
    }
}

#[test]
fn armor_roundtrip_real_photos() {
    let photos = discover_photos();
    eprintln!("Armor round-trip: found {} photos", photos.len());
    for photo in &photos {
        let name = photo.file_name().unwrap().to_string_lossy();
        let jpeg_bytes = std::fs::read(photo).unwrap();
        let image = match JpegImage::from_bytes(&jpeg_bytes) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("  {name}: skipped (parse error: {e:?})");
                continue;
            }
        };
        let cap = match phasm_core::armor_capacity(&image) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  {name}: skipped (capacity error: {e:?})");
                continue;
            }
        };
        let msg = "Armor test message";
        if msg.len() > cap {
            eprintln!("  {name}: skipped (capacity {cap} < {})", msg.len());
            continue;
        }
        let stego = armor_encode(&jpeg_bytes, msg, "test-pass").unwrap();
        let (decoded, _quality) = armor_decode(&stego, "test-pass").unwrap();
        assert_eq!(decoded.text, msg, "Armor mismatch on {name}");
        eprintln!("  {name}: OK (capacity {cap})");
    }
}
