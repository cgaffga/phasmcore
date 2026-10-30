// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! `phasm capacity` and `phasm encode --mode ghost` show the same image Ghost
//! bands the apps show (`ghost_bands` / `ghost_bands_si`), and only show them:
//! a payload above the max warns on stderr but still encodes and decodes.

use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, ImageEncoder, RgbImage};
use phasm_core::{deep_cover_jpeg, ghost_bands, ghost_bands_si, JpegImage};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const PASS: &str = "bands-pass";

fn scratch(tag: &str) -> PathBuf {
    let base = std::env::var("PHASM_TMP").unwrap_or_else(|_| "/tmp".into());
    let dir = PathBuf::from(base).join(format!("phasm_cli_ghostbands_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A textured 320x240 photo-like image (deterministic).
fn cover() -> RgbImage {
    let (w, h) = (320u32, 240u32);
    let mut state = 0x1234_5678u32;
    RgbImage::from_fn(w, h, |x, y| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let n = ((state >> 24) % 40) as i32 - 20;
        let v = |base: i32| (base + n).clamp(0, 255) as u8;
        image::Rgb([
            v(((x * 255) / w) as i32),
            v(((y * 255) / h) as i32),
            v((((x + y) * 255) / (w + h)) as i32),
        ])
    })
}

fn write_jpeg(path: &Path, img: &RgbImage) {
    let mut buf = Vec::new();
    JpegEncoder::new_with_quality(&mut buf, 85)
        .write_image(img.as_raw(), img.width(), img.height(), ExtendedColorType::Rgb8)
        .unwrap();
    std::fs::write(path, buf).unwrap();
}

fn phasm(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_phasm")).args(args).output().unwrap()
}

/// The unsigned integer after `"key":` in a flat JSON object.
fn json_num(json: &str, key: &str) -> usize {
    let tag = format!("\"{key}\":");
    let at = json.find(&tag).unwrap_or_else(|| panic!("no {key} in {json}")) + tag.len();
    json[at..].chars().take_while(char::is_ascii_digit).collect::<String>().parse().unwrap()
}

fn check_json(json: &str, b: phasm_core::GhostBands) {
    assert_eq!(json_num(json, "ghostMax"), b.max_bytes, "{json}");
    assert_eq!(json_num(json, "ghostCaution"), b.caution_bytes, "{json}");
    assert_eq!(json_num(json, "ghostComfort"), b.comfort_bytes, "{json}");
    assert!(b.comfort_bytes <= b.caution_bytes && b.caution_bytes <= b.max_bytes);
}

#[test]
fn capacity_json_matches_core_bands_jpeg() {
    let dir = scratch("cap_jpeg");
    let path = dir.join("c.jpg");
    write_jpeg(&path, &cover());
    let out = phasm(&["capacity", path.to_str().unwrap(), "--json"]);
    assert!(out.status.success());
    let json = String::from_utf8(out.stdout).unwrap();
    let img = JpegImage::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    check_json(&json, ghost_bands(&img, 0, 0).unwrap());
    assert!(json.contains("\"ghostDeepCoverBands\":false"));
}

#[test]
fn capacity_json_matches_deep_cover_bands_png() {
    let dir = scratch("cap_png");
    let path = dir.join("c.png");
    let rgb = cover();
    rgb.save(&path).unwrap();
    let out = phasm(&["capacity", path.to_str().unwrap(), "--json"]);
    assert!(out.status.success());
    let json = String::from_utf8(out.stdout).unwrap();
    // The cover `encode --mode ghost` writes for this PNG, with its pixels.
    let jpeg = deep_cover_jpeg(rgb.as_raw(), rgb.width(), rgb.height(), 92).unwrap();
    let img = JpegImage::from_bytes(&jpeg).unwrap();
    let b = ghost_bands_si(&img, rgb.as_raw(), rgb.width(), rgb.height(), 0, 0).unwrap();
    check_json(&json, b);
    assert!(json.contains("\"ghostDeepCoverBands\":true"));
    // Exact side information: larger maximum than the plain bands.
    assert!(b.max_bytes > ghost_bands(&img, 0, 0).unwrap().max_bytes);
    // The text output carries the same numbers.
    let text = phasm(&["capacity", path.to_str().unwrap(), "--mode", "ghost"]);
    let text = String::from_utf8(text.stdout).unwrap();
    assert!(text.contains(&format!("max {} bytes", b.max_bytes)), "{text}");
    assert!(text.contains(&format!("comfort {} bytes", b.comfort_bytes)), "{text}");
    assert!(text.contains("Deep Cover"), "{text}");
}

fn encode_decode(msg: &str, extra: &[&str], tag: &str) -> (Output, String) {
    let dir = scratch(tag);
    let src = dir.join("c.jpg");
    write_jpeg(&src, &cover());
    let out_path = dir.join("o.jpg");
    let mut args = vec![
        "encode", src.to_str().unwrap(), "--mode", "ghost", "-m", msg, "-p", PASS,
        "-o", out_path.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    let enc = phasm(&args);
    let decoded = if out_path.exists() {
        let dec = phasm(&["decode", out_path.to_str().unwrap(), "-p", PASS, "--quiet"]);
        String::from_utf8(dec.stdout).unwrap()
    } else {
        String::new()
    };
    (enc, decoded)
}

#[test]
fn encode_below_max_shows_bands_without_warning() {
    let (enc, decoded) = encode_decode("hello", &[], "enc_below");
    assert!(enc.status.success());
    let err = String::from_utf8(enc.stderr).unwrap();
    assert!(err.contains("max: ") && err.contains("comfort: ") && err.contains("payload: "), "{err}");
    assert!(!err.contains("warning"), "{err}");
    assert_eq!(decoded, "hello");
}

#[test]
fn encode_above_max_warns_but_still_encodes() {
    let img_dir = scratch("enc_above_probe");
    let probe = img_dir.join("c.jpg");
    write_jpeg(&probe, &cover());
    let img = JpegImage::from_bytes(&std::fs::read(&probe).unwrap()).unwrap();
    let max = ghost_bands(&img, 0, 0).unwrap().max_bytes;
    // Incompressible-ish text, grown until its compressed payload is just above
    // the max (compression must not hide it).
    let mut state = 0xdead_beefu32;
    let mut msg = String::new();
    while phasm_core::compressed_payload_size(&msg, &[]) <= max + 16 {
        for _ in 0..32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            msg.push(char::from(b'!' + ((state >> 24) % 90) as u8));
        }
    }

    let (enc, decoded) = encode_decode(&msg, &[], "enc_above");
    let err = String::from_utf8(enc.stderr.clone()).unwrap();
    assert!(err.contains("warning: the payload"), "{err}");
    assert!(err.contains(&format!("({max} B)")), "{err}");
    assert!(enc.status.success(), "above max still encodes: {err}");
    assert_eq!(decoded, msg);

    // --quiet suppresses it, --json keeps stdout clean (valid JSON object).
    let (quiet, _) = encode_decode(&msg, &["--quiet"], "enc_above_q");
    assert!(quiet.status.success());
    assert!(!String::from_utf8(quiet.stderr).unwrap().contains("warning"));
    let (json, _) = encode_decode(&msg, &["--json"], "enc_above_j");
    let out = String::from_utf8(json.stdout).unwrap();
    assert!(out.trim().starts_with('{') && out.trim().ends_with('}'), "{out}");
    assert!(String::from_utf8(json.stderr).unwrap().contains("warning"));
}
