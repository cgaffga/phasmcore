// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! `phasm encode --fortress auto|t|tl` (issue cgaffga/phasmcore#6): the Armor
//! payload a message Fortress takes is written with, reported in the output;
//! `tl` fails clearly on an image with too little texture.

use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, ImageEncoder, RgbImage};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const PASS: &str = "fortress-pass";
const MSG: &str = "Meet at the old bridge at seven.";

fn scratch(tag: &str) -> PathBuf {
    let base = std::env::var("PHASM_TMP").unwrap_or_else(|_| "/tmp".into());
    let dir = PathBuf::from(base).join(format!("phasm_cli_fortress_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// 1024 x 768, textured at block scale everywhere (multi-octave value noise,
/// like foliage or stone): Fortress TL's activity comes from the block means,
/// so per-pixel noise would not count as texture.
fn textured() -> RgbImage {
    let (w, h) = (1024usize, 768usize);
    let mut s: u64 = 0x5DEE_CE66D;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut lum = vec![128.0f64; w * h];
    for (cell, amp) in [(32usize, 30.0f64), (16, 18.0), (8, 12.0), (4, 7.0)] {
        let (gw, gh) = (w / cell + 2, h / cell + 2);
        let g: Vec<f64> = (0..gw * gh).map(|_| next() - 0.5).collect();
        for y in 0..h {
            let (fy, ty) = (y / cell, (y % cell) as f64 / cell as f64);
            for x in 0..w {
                let (fx, tx) = (x / cell, (x % cell) as f64 / cell as f64);
                let a = g[fy * gw + fx] * (1.0 - tx) + g[fy * gw + fx + 1] * tx;
                let b = g[(fy + 1) * gw + fx] * (1.0 - tx) + g[(fy + 1) * gw + fx + 1] * tx;
                lum[y * w + x] += amp * 2.0 * (a * (1.0 - ty) + b * ty);
            }
        }
    }
    RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let g = lum[y as usize * w + x as usize].clamp(8.0, 247.0) as u8;
        image::Rgb([g, g.saturating_sub(10), g.saturating_add(6)])
    })
}

/// 1024 x 768 smooth gradient: Fortress T takes a short message, Fortress TL nothing.
fn flat() -> RgbImage {
    RgbImage::from_fn(1024, 768, |x, y| {
        let g = (60 + x * 120 / 1024 + y * 40 / 768) as u8;
        image::Rgb([g, g, g.saturating_add(20)])
    })
}

fn write_jpeg(path: &Path, img: &RgbImage) {
    let mut buf = Vec::new();
    JpegEncoder::new_with_quality(&mut buf, 90)
        .write_image(img.as_raw(), img.width(), img.height(), ExtendedColorType::Rgb8)
        .unwrap();
    std::fs::write(path, buf).unwrap();
}

fn phasm(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_phasm")).args(args).output().unwrap()
}

fn encode_json(src: &Path, out: &Path, fortress: &str) -> Output {
    phasm(&["encode", src.to_str().unwrap(), "-m", MSG, "-p", PASS, "--fortress", fortress, "--json", "-o", out.to_str().unwrap()])
}

fn decodes(path: &Path) -> bool {
    let o = phasm(&["decode", path.to_str().unwrap(), "-p", PASS, "--quiet"]);
    o.status.success() && String::from_utf8_lossy(&o.stdout).trim_end() == MSG
}

#[test]
fn fortress_flag_chooses_the_payload_and_reports_it() {
    let dir = scratch("choose");
    let src = dir.join("textured.jpg");
    write_jpeg(&src, &textured());
    for (flag, written) in [("auto", "\"fortress\":\"T\""), ("t", "\"fortress\":\"T\""), ("tl", "\"fortress\":\"TL\"")] {
        let out = dir.join(format!("out_{flag}.jpg"));
        let o = encode_json(&src, &out, flag);
        let json = String::from_utf8_lossy(&o.stdout);
        assert!(o.status.success(), "--fortress {flag}: {}", String::from_utf8_lossy(&o.stderr));
        assert!(json.contains(written) && json.contains("\"mode\":\"Armor\""), "--fortress {flag}: {json}");
        assert!(decodes(&out), "--fortress {flag}: decode");
    }
    // The text output names the sub-mode too.
    let out = dir.join("out_text.jpg");
    let o = phasm(&["encode", src.to_str().unwrap(), "-m", MSG, "-p", PASS, "--fortress", "tl", "-o", out.to_str().unwrap()]);
    assert!(String::from_utf8_lossy(&o.stdout).contains("Mode: Armor (Fortress TL)"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fortress_tl_on_a_flat_image_fails_clearly() {
    let dir = scratch("flat");
    let src = dir.join("flat.jpg");
    write_jpeg(&src, &flat());
    let o = encode_json(&src, &dir.join("tl.jpg"), "tl");
    assert_eq!(o.status.code(), Some(4), "exit code: message too large");
    assert!(String::from_utf8_lossy(&o.stderr).contains("Fortress TL cannot hold this message"), "{}", String::from_utf8_lossy(&o.stderr));
    let out = dir.join("auto.jpg");
    let o = encode_json(&src, &out, "auto");
    assert!(o.status.success() && String::from_utf8_lossy(&o.stdout).contains("\"fortress\":\"T\""));
    assert!(decodes(&out));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fortress_flag_is_armor_only() {
    let dir = scratch("ghost");
    let src = dir.join("textured.jpg");
    write_jpeg(&src, &textured());
    let o = phasm(&["encode", src.to_str().unwrap(), "-m", MSG, "-p", PASS, "--mode", "ghost", "--fortress", "tl", "-o", dir.join("g.jpg").to_str().unwrap()]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--fortress applies to Armor only"));
    let _ = std::fs::remove_dir_all(&dir);
}
