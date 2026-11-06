// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Rescale decode on scale-friendly Armor covers (`armor_cover_dimensions`):
//! an odd-aspect source (3:2, whose 1600 px short side would be 1067) through
//! simulated messenger downscales that round, floor or ceil the short side.
//! Measurement: `eval/runs/2026-10-31-armor-v2/RESULTS.md`, Part 4.

use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};
use phasm_core::stego::armor::pipeline::{armor_decode_opts, armor_encode_opts, ArmorDecodeOpts, ArmorEncodeOpts};
use phasm_core::{armor_cover_dimensions, armor_decode, armor_encode, deep_cover_jpeg, resample_rgb, ArmorCoverPreset};

/// A textured upright source: smooth random-phase waves plus fine noise.
fn source(w: u32, h: u32) -> RgbImage {
    let mut s: u32 = 0x9e37_79b9;
    let mut rnd = move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 8) as f64 / (1u32 << 24) as f64
    };
    let waves: Vec<(f64, f64, f64)> = (0..12).map(|_| (rnd() * 0.08, rnd() * 0.08, rnd() * 6.283)).collect();
    let mut img = RgbImage::new(w, h);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let v: f64 = waves.iter().map(|&(fx, fy, ph)| (fx * x as f64 + fy * y as f64 + ph).sin()).sum::<f64>();
        let l = (128.0 + 22.0 * v + 30.0 * (rnd() - 0.5)).clamp(0.0, 255.0) as u8;
        *p = image::Rgb([l, l.saturating_add(8), l.saturating_sub(8)]);
    }
    img
}

/// The Armor cover the frontends make: scale, center-crop, QF65.
fn cover(src: &RgbImage, preset: ArmorCoverPreset) -> (Vec<u8>, u32, u32) {
    let d = armor_cover_dimensions(src.width(), src.height(), preset);
    let scaled = resample_rgb(src.as_raw(), src.width(), src.height(), d.scaled_width, d.scaled_height).unwrap();
    let scaled = RgbImage::from_raw(d.scaled_width, d.scaled_height, scaled).unwrap();
    let px = DynamicImage::ImageRgb8(scaled).crop_imm(d.crop_x, d.crop_y, d.width, d.height).to_rgb8();
    (deep_cover_jpeg(px.as_raw(), d.width, d.height, 65).unwrap(), d.width, d.height)
}

/// A messenger: long side (or width) to `size`, the other side rounded with
/// `round`, Lanczos, re-encoded at `q`.
fn channel(stego: &[u8], size: u32, width_pinned: bool, round: fn(f64) -> f64, q: u8) -> Vec<u8> {
    let img = image::load_from_memory(stego).unwrap();
    let (w, h) = (img.width(), img.height());
    let (nw, nh) = if width_pinned || w >= h {
        (size, round(h as f64 * size as f64 / w as f64) as u32)
    } else {
        (round(w as f64 * size as f64 / h as f64) as u32, size)
    };
    let small = img.resize_exact(nw, nh, FilterType::Lanczos3).to_rgb8();
    let mut out = Vec::new();
    JpegEncoder::new_with_quality(&mut out, q).encode_image(&small).unwrap();
    out
}

const ROUNDINGS: [(&str, fn(f64) -> f64); 3] = [("round", f64::round), ("floor", f64::floor), ("ceil", f64::ceil)];

#[test]
fn odd_aspect_cover_is_trimmed_to_the_grid() {
    let d = armor_cover_dimensions(2400, 1600, ArmorCoverPreset::Standard);
    assert_eq!((d.width, d.height, d.scaled_height, d.crop_y), (1600, 1065, 1067, 1));
}

/// v2 STDM (the format that needs the exact cover size) at 256 B through
/// 1280 px (x 4/5, exact on a trimmed cover) at QF87 (Telegram) and QF75,
/// with each rounding mode. (v2 does not survive 1080 px wide reliably even
/// at the exact size; Fortress does, see below.)
#[test]
fn v2_stdm_survives_downscales_of_a_trimmed_odd_aspect_cover() {
    let (cover_jpeg, w, h) = cover(&source(2400, 1600), ArmorCoverPreset::Standard);
    assert_eq!((w, h), (1600, 1065));
    let msg: String = (0..256).map(|i| (b'!' + ((i * 37 + 11) % 90) as u8) as char).collect();
    let opts = ArmorEncodeOpts { force_stdm: true, ..ArmorEncodeOpts::default() };
    let (stego, info) = armor_encode_opts(&cover_jpeg, &msg, "pass-rescale", &opts).unwrap();
    assert!(!info.fortress);
    for (name, round) in ROUNDINGS {
        for (size, q) in [(1280, 87), (1280, 75)] {
            let rx = channel(&stego, size, false, round, q);
            let (p, _) = armor_decode(&rx, "pass-rescale").unwrap_or_else(|e| panic!("{size} {name}: {e}"));
            assert_eq!(p.text, msg, "{size} {name}");
        }
    }
}

/// Fortress CC (the production short-message path) through Instagram's
/// 1080 px width: a Telegram cover (1280 px preset, x 27/32) and a standard
/// one (1600 px, x 27/40), each rounding mode.
#[test]
fn fortress_survives_instagram_downscales() {
    for (src, preset, dims) in [
        ((3000, 2000), ArmorCoverPreset::Telegram, (1280, 850)),
        ((2400, 1600), ArmorCoverPreset::Standard, (1600, 1065)),
    ] {
        let (cover_jpeg, w, h) = cover(&source(src.0, src.1), preset);
        assert_eq!((w, h), dims);
        let stego = armor_encode(&cover_jpeg, "meet at the old bridge", "pass-ig").unwrap();
        for (name, round) in ROUNDINGS {
            let rx = channel(&stego, 1080, true, round, 75);
            let (p, _) = armor_decode(&rx, "pass-ig").unwrap_or_else(|e| panic!("{preset:?} {name}: {e}"));
            assert_eq!(p.text, "meet at the old bridge", "{preset:?} {name}");
        }
    }
}

/// The snap is decoder-only and runs only in the rescale attempt: a received
/// image at the cover's own size decodes the same with and without it, and a
/// downscaled one decodes through the "rescaled" path.
#[test]
fn snap_only_affects_the_rescale_attempt() {
    let (cover_jpeg, _, _) = cover(&source(2400, 1600), ArmorCoverPreset::Standard);
    let msg: String = (0..200).map(|i| (b'#' + ((i * 13 + 5) % 80) as u8) as char).collect();
    let opts = ArmorEncodeOpts { force_stdm: true, ..ArmorEncodeOpts::default() };
    let (stego, _) = armor_encode_opts(&cover_jpeg, &msg, "pw", &opts).unwrap();
    let same_size = channel(&stego, 1600, false, f64::round, 75);
    let no_snap = ArmorDecodeOpts { rescale_snap: 1, ..ArmorDecodeOpts::default() };
    for opts in [ArmorDecodeOpts::default(), no_snap] {
        let (p, _, i) = armor_decode_opts(&same_size, "pw", &opts).unwrap();
        assert_eq!((p.text.as_str(), i.path), (msg.as_str(), "v2"));
    }
    let down = channel(&stego, 1280, false, f64::floor, 87);
    let (p, _, i) = armor_decode_opts(&down, "pw", &ArmorDecodeOpts::default()).unwrap();
    assert_eq!((p.text.as_str(), i.path), (msg.as_str(), "rescaled"));
}

/// The Instagram preset pre-crops a 2:3 portrait to 4:5 (1080 x 1350), which
/// Instagram keeps: the message survives its re-encode at that size.
#[test]
fn instagram_portrait_cover_is_four_fifths_and_decodes() {
    let (cover_jpeg, w, h) = cover(&source(1600, 2400), ArmorCoverPreset::Instagram);
    assert_eq!((w, h), (1080, 1350));
    let stego = armor_encode(&cover_jpeg, "posted as is", "pass-ig45").unwrap();
    let rx = channel(&stego, 1080, true, f64::round, 75);
    assert_eq!(image::load_from_memory(&rx).unwrap().height(), 1350);
    assert_eq!(armor_decode(&rx, "pass-ig45").unwrap().0.text, "posted as is");
}
