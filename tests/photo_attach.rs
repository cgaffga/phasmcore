// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Photo attachments: fitting (libaom) and showing (dav1d). Needs `photo-encoder` and `av1-decoder`.
#![cfg(all(feature = "photo-encoder", feature = "av1-decoder"))]

use phasm_core::photo::avif::{is_avif, read_avif};
use phasm_core::photo::decode::decode_avif;
use phasm_core::photo::fit::{fit_photo, FLOOR_QUANTIZER};
use phasm_core::photo::PhotoError;
use phasm_core::{compressed_payload_size, resample_rgb, FileEntry};

/// A 1600 x 1200 test picture: smooth sky, a textured lower half, a few edges (deterministic).
fn picture() -> (Vec<u8>, u32, u32) {
    let (w, h) = (1600u32, 1200u32);
    let mut s = 0x1234_5678u32;
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let n = (s % 40) as i32 - 20;
            let px = if y < h / 2 {
                [120 + (y / 10) as i32, 160 + (y / 12) as i32, 220]
            } else {
                let t = ((x / 37 + y / 23) % 7) as i32 * 18;
                [70 + t + n, 110 + t / 2 + n, 40 + n]
            };
            let edge = (x / 200 + y / 150) % 5 == 0 && x % 200 < 8;
            for c in px {
                rgb.push(if edge { 20 } else { c.clamp(0, 255) as u8 });
            }
        }
    }
    (rgb, w, h)
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
}

#[test]
fn fits_each_budget_and_decodes() {
    let (rgb, w, h) = picture();
    let text = "Meet me at the old bridge at nine.";
    let mut last_w = 0;
    for budget in [1_700usize, 4_000, 14_000] {
        let mut calls = 0;
        let fitted = fit_photo(&rgb, w, h, text, &[], "photo-1.avif", budget, &mut |p| {
            assert!((0.0..=1.0).contains(&p));
            calls += 1;
            true
        })
        .expect("fit");
        let files = [FileEntry { filename: "photo-1.avif".into(), content: fitted.avif.clone() }];
        let used = compressed_payload_size(text, &files);
        assert!(used <= budget, "budget {budget}: payload {used}");
        // a fit that leaves most of the budget unused would be a bug in the search
        assert!(used * 10 >= budget * 7, "budget {budget}: payload only {used}");
        assert!(fitted.width >= last_w, "a larger budget never gives a smaller photo");
        last_w = fitted.width;
        assert!(calls >= 3);

        assert!(is_avif(&fitted.avif));
        let img = read_avif(&fitted.avif).expect("read");
        assert_eq!((img.width, img.height), (fitted.width, fitted.height));
        let dec = decode_avif(&fitted.avif).expect("decode");
        assert_eq!((dec.width, dec.height), (fitted.width, fitted.height));
        let reference = resample_rgb(&rgb, w, h, fitted.width, fitted.height).unwrap();
        let p = psnr(&reference, &dec.rgb);
        assert!(p > 18.0, "budget {budget}: {}x{} q{} PSNR {p:.1}", fitted.width, fitted.height, fitted.quantizer);
        eprintln!(
            "budget {budget}: {}x{} q{} {} B, payload {used}, PSNR {p:.1}, below floor {}",
            fitted.width, fitted.height, fitted.quantizer, fitted.avif.len(), fitted.below_floor
        );
        if !fitted.below_floor {
            assert!(fitted.quantizer as i32 <= FLOOR_QUANTIZER + 4);
        }
    }
}

#[test]
fn shares_the_budget_with_other_files() {
    let (rgb, w, h) = picture();
    let other = [FileEntry { filename: "note.txt".into(), content: vec![b'x'; 900] }];
    let first = fit_photo(&rgb, w, h, "", &other, "photo-2.avif", 4_000, &mut |_| true).unwrap();
    let all = [other[0].clone(), FileEntry { filename: "photo-2.avif".into(), content: first.avif }];
    assert!(compressed_payload_size("", &all) <= 4_000);
}

#[test]
fn too_small_a_budget_does_not_fit() {
    let (rgb, w, h) = picture();
    let err = fit_photo(&rgb, w, h, "", &[], "photo-1.avif", 120, &mut |_| true).unwrap_err();
    assert_eq!(err, PhotoError::DoesNotFit);
}

#[test]
fn cancel_stops_the_fit() {
    let (rgb, w, h) = picture();
    let mut n = 0;
    let err = fit_photo(&rgb, w, h, "", &[], "photo-1.avif", 14_000, &mut |_| {
        n += 1;
        n < 3
    })
    .unwrap_err();
    assert_eq!(err, PhotoError::Cancelled);
}

#[test]
fn bad_input_is_refused() {
    assert_eq!(fit_photo(&[0; 10], 4, 4, "", &[], "p.avif", 4_000, &mut |_| true).unwrap_err(), PhotoError::InvalidImage);
    assert!(read_avif(b"not an avif file at all").is_err());
    assert!(decode_avif(&[0u8; 64]).is_err());
}

/// Same planes through `avifenc` and through us: the encoder setup must match the measurement's
/// (sizes within a few percent). Needs a 4:2:0 y4m and its avifenc files:
/// PHASM_PHOTO_Y4M=…/valley512.y4m PHASM_PHOTO_AVIFENC="58:…/av_8.avif,50:…/av_20.avif"
#[test]
#[ignore]
fn matches_avifenc() {
    use phasm_core::photo::fit::encode_i420;
    let y4m = std::fs::read(std::env::var("PHASM_PHOTO_Y4M").unwrap()).unwrap();
    let head_end = y4m.iter().position(|&b| b == b'\n').unwrap();
    let head = std::str::from_utf8(&y4m[..head_end]).unwrap();
    let dim = |k: char| head.split(' ').find(|t| t.starts_with(k)).unwrap()[1..].parse::<usize>().unwrap();
    let (w, h) = (dim('W'), dim('H'));
    let frame = head_end + 1 + y4m[head_end + 1..].iter().position(|&b| b == b'\n').unwrap() + 1;
    let (y, rest) = y4m[frame..].split_at(w * h);
    let (u, v) = rest.split_at(w * h / 4);
    for pair in std::env::var("PHASM_PHOTO_AVIFENC").unwrap().split(',') {
        let (q, path) = pair.split_once(':').unwrap();
        let theirs = std::fs::metadata(path).unwrap().len() as f64;
        let file = encode_i420(y, u, &v[..w * h / 4], w as u32, h as u32, q.parse().unwrap(), 6, 1).unwrap();
        // kept next to the reference for a look with other decoders (avifdec, sips, browsers)
        std::fs::write(format!("{path}.ours.avif"), &file).unwrap();
        let ours = file.len() as f64;
        eprintln!("q{q}: avifenc {theirs} B, ours {ours} B ({:+.1} %)", (ours / theirs - 1.0) * 100.0);
        assert!((ours / theirs - 1.0).abs() < 0.05);
    }
}
