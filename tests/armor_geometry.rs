// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Armor geometry resilience: Fortress T (tiled layout + pilots + checkerboard
//! sync) survives rotation, rescale and crops of any offset, each followed by
//! a JPEG re-save.
//!
//! The DFT template of core <= 0.5.x is not tested: it is not in the file (the
//! template code placed its peaks around index (W/2, H/2) of an unshifted
//! spectrum, i.e. at 0.33-0.47 cycles/px, where the QF65 re-quantization of
//! the encoder itself removes them) and never rescued an image
//! (eval/runs/2026-11-02-armor-geometry/RESULTS.md).

use phasm_core::stego::armor::pipeline::{armor_encode_opts, ArmorEncodeOpts};
use phasm_core::{armor_decode, armor_encode, deep_cover_jpeg};

fn rgb_of(jpeg: &[u8]) -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(jpeg).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

fn jpeg(px: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    let mut b = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut b, q)
        .encode(px, w, h, image::ExtendedColorType::Rgb8)
        .unwrap();
    b
}

/// An Armor cover as the apps write it (the 1280 px channel preset, QF65).
fn cover() -> Vec<u8> {
    let src = std::fs::read("test-vectors/image/photo_640x480_q75_420.jpg").unwrap();
    let (px, w, h) = rgb_of(&src);
    let up = phasm_core::resample_rgb(&px, w, h, 1280, 960).unwrap();
    deep_cover_jpeg(&up, 1280, 960, 65).unwrap()
}

fn encode_t(message: &str, passphrase: &str) -> Vec<u8> {
    let opts = ArmorEncodeOpts { fortress_tiled: true, ..ArmorEncodeOpts::default() };
    let (bytes, info) = armor_encode_opts(&cover(), message, passphrase, &opts).unwrap();
    assert!(info.fortress, "the message must take Fortress T");
    bytes
}

/// Rotate by `deg` about the centre, keeping the canvas (edge-clamped, bilinear).
fn rotate(px: &[u8], w: u32, h: u32, deg: f64) -> Vec<u8> {
    let (s, c) = deg.to_radians().sin_cos();
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let mut out = vec![0u8; px.len()];
    let get = |x: f64, y: f64, ch: usize| {
        let (xi, yi) = (x.clamp(0.0, w as f64 - 1.0) as usize, y.clamp(0.0, h as f64 - 1.0) as usize);
        px[(yi * w as usize + xi) * 3 + ch] as f64
    };
    for y in 0..h as usize {
        for x in 0..w as usize {
            let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
            let sx = c * dx + s * dy + cx - 0.5;
            let sy = -s * dx + c * dy + cy - 0.5;
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            for ch in 0..3 {
                let v = get(x0, y0, ch) * (1.0 - fx) * (1.0 - fy)
                    + get(x0 + 1.0, y0, ch) * fx * (1.0 - fy)
                    + get(x0, y0 + 1.0, ch) * (1.0 - fx) * fy
                    + get(x0 + 1.0, y0 + 1.0, ch) * fx * fy;
                out[(y * w as usize + x) * 3 + ch] = v.round() as u8;
            }
        }
    }
    out
}

fn crop(px: &[u8], w: u32, x0: u32, y0: u32, cw: u32, ch: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((cw * ch * 3) as usize);
    for y in y0..y0 + ch {
        let r = ((y * w + x0) * 3) as usize;
        out.extend_from_slice(&px[r..r + (cw * 3) as usize]);
    }
    out
}

#[test]
fn armor_geometry_backward_compat_no_transform() {
    let cover = std::fs::read("test-vectors/image/photo_320x240_q75_420.jpg").unwrap();
    let message = "Geometry resilience test - no transform";
    let stego = armor_encode(&cover, message, "geo-test-pass").unwrap();
    let (decoded, quality) = armor_decode(&stego, "geo-test-pass").unwrap();
    assert_eq!(decoded.text, message);
    assert!(!quality.geometry_corrected, "should decode via fast path");
}

#[test]
fn fortress_t_untouched_and_recompressed() {
    let msg = "tiled fortress, untouched";
    let stego = encode_t(msg, "t-pass");
    let (d, q) = armor_decode(&stego, "t-pass").unwrap();
    assert_eq!(d.text, msg);
    assert!(q.fortress_used && !q.geometry_corrected);
    let (px, w, h) = rgb_of(&stego);
    let (d, _) = armor_decode(&jpeg(&px, w, h, 75), "t-pass").unwrap();
    assert_eq!(d.text, msg);
}

#[test]
fn fortress_t_rotation() {
    let msg = "survive a 3 degree rotation";
    let stego = encode_t(msg, "rot-pass");
    let (px, w, h) = rgb_of(&stego);
    let rotated = jpeg(&rotate(&px, w, h, 3.0), w, h, 85);
    let (d, q) = armor_decode(&rotated, "rot-pass").unwrap();
    assert_eq!(d.text, msg);
    assert!(q.geometry_corrected);
    assert!((q.estimated_rotation_deg - 3.0).abs() < 0.05, "rotation {}", q.estimated_rotation_deg);
    assert!((q.estimated_scale - 1.0).abs() < 0.002, "scale {}", q.estimated_scale);
}

#[test]
fn fortress_t_crop_any_offset() {
    let msg = "crop me anywhere";
    let stego = encode_t(msg, "crop-pass");
    let (px, w, h) = rgb_of(&stego);
    // Whole blocks (aligned decoder) and an odd offset (grid sync).
    for (x0, y0) in [(16u32, 8u32), (37, 21)] {
        let (cw, ch) = (w - x0 - 11, h - y0 - 5);
        let c = jpeg(&crop(&px, w, x0, y0, cw, ch), cw, ch, 85);
        let (d, _) = armor_decode(&c, "crop-pass").unwrap_or_else(|e| panic!("crop ({x0},{y0}): {e}"));
        assert_eq!(d.text, msg);
    }
}

#[test]
fn fortress_t_rescale() {
    let msg = "survive a rescale";
    let stego = encode_t(msg, "scale-pass");
    let (px, w, h) = rgb_of(&stego);
    for s in [0.75f64, 1.3] {
        let (tw, th) = ((w as f64 * s).round() as u32, (h as f64 * s).round() as u32);
        let r = phasm_core::resample_rgb(&px, w, h, tw, th).unwrap();
        let (d, q) = armor_decode(&jpeg(&r, tw, th, 85), "scale-pass").unwrap_or_else(|e| panic!("scale {s}: {e}"));
        assert_eq!(d.text, msg);
        assert!((q.estimated_scale as f64 - s).abs() < 0.01, "scale {} vs {s}", q.estimated_scale);
    }
}

#[test]
fn fortress_t_empty_passphrase_rotated() {
    let msg = "compact frame";
    let stego = encode_t(msg, "");
    let (px, w, h) = rgb_of(&stego);
    let (d, _) = armor_decode(&jpeg(&rotate(&px, w, h, -2.0), w, h, 85), "").unwrap();
    assert_eq!(d.text, msg);
}

#[test]
fn fortress_t_wrong_passphrase_fails() {
    let stego = encode_t("secret", "correct-pass");
    let (px, w, h) = rgb_of(&stego);
    assert!(armor_decode(&stego, "wrong-pass").is_err());
    let rotated = jpeg(&rotate(&px, w, h, 2.0), w, h, 85);
    assert!(armor_decode(&rotated, "wrong-pass").is_err());
}

/// Quarter turns (0-3, counter-clockwise) and the mirror image, pixel-exact.
fn turn(px: &[u8], w: u32, h: u32, quarters: u32, mirror: bool) -> (Vec<u8>, u32, u32) {
    let mut img = image::RgbImage::from_raw(w, h, px.to_vec()).unwrap();
    if mirror {
        img = image::imageops::flip_horizontal(&img);
    }
    img = match quarters % 4 {
        1 => image::imageops::rotate270(&img),
        2 => image::imageops::rotate180(&img),
        3 => image::imageops::rotate90(&img),
        _ => img,
    };
    let (tw, th) = img.dimensions();
    (img.into_raw(), tw, th)
}

/// Turned by quarter turns, mirrored, and both, each re-saved; also with a
/// few degrees and a crop on top (the checkerboard looks the same in every
/// orientation, the pilots tell which one).
#[test]
fn fortress_t_quarter_turns_and_mirror() {
    let msg = "turn me around";
    let st = encode_t(msg, "pw");
    let (px, w, h) = rgb_of(&st);
    for mirror in [false, true] {
        for q in 0..4 {
            if q == 0 && !mirror {
                continue;
            }
            let (t, tw, th) = turn(&px, w, h, q, mirror);
            let out = jpeg(&t, tw, th, 85);
            assert_eq!(armor_decode(&out, "pw").unwrap().0.text, msg, "{q} quarter turns, mirror {mirror}");
            let r = rotate(&t, tw, th, 2.5);
            let c = crop(&r, tw, 37, 21, tw - 101, th - 77);
            let out = jpeg(&c, tw - 101, th - 77, 85);
            assert_eq!(armor_decode(&out, "pw").unwrap().0.text, msg, "{q} quarter turns + 2.5 deg + crop, mirror {mirror}");
        }
    }
    assert!(armor_decode(&jpeg(&turn(&px, w, h, 1, false).0, h, w, 85), "wrong").is_err());
}

/// The FFT search over all orientations agrees with the direct search on an
/// untouched image: orientation 0, the same offset, the same (exact) score.
#[test]
fn fft_pilot_search_matches_direct() {
    use phasm_core::codec::jpeg::JpegImage;
    use phasm_core::stego::armor::fortress_tile::{find_offset, find_offset_any, BlockField, Dihedral, TileKey};
    let st = encode_t("pinned", "pw");
    let field = BlockField::from_image(&JpegImage::from_bytes(&st).unwrap()).unwrap();
    let key = TileKey::new("pw").unwrap();
    let (d, ox, oy, z) = find_offset_any(&field, &key);
    let (ox1, oy1, z1) = find_offset(&field, &key);
    assert_eq!((d, ox, oy), (Dihedral(0), ox1, oy1));
    assert_eq!(z.to_bits(), z1.to_bits());
    assert!(z > 6.0);
}

// ---------------------------------------------------------------------------
// Fortress TL: messages longer than Fortress T, on the AC lanes.
// ---------------------------------------------------------------------------

/// A 1280 x 960 Armor cover textured everywhere: multi-octave value noise
/// (random values on 4 / 8 / 16 / 32 px grids, bilinearly interpolated), like
/// foliage or stone, without the line spectra a sum of sine waves would add
/// (those can pull the grid sync onto a wrong scale). Fortress TL carries its
/// payload only in textured blocks and needs texture in every large crop
/// window, which the repo's test photos (texture in one region) do not have.
fn textured_cover() -> Vec<u8> {
    let (w, h) = (1280usize, 960usize);
    let mut s: u64 = 0x5DEECE66D;
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
    let px: Vec<u8> = lum
        .iter()
        .flat_map(|&v| {
            let g = v.clamp(8.0, 247.0) as u8;
            [g, g.saturating_sub(10), g.saturating_add(6)]
        })
        .collect();
    deep_cover_jpeg(&px, w as u32, h as u32, 65).unwrap()
}

/// Random text whose payload is `target` bytes or just below.
fn text_of_payload(target: usize, seed: u64) -> String {
    let mut s = seed | 1;
    let mut out = String::new();
    loop {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.push((b'a' + (s % 26) as u8) as char);
        if phasm_core::stego::payload::compressed_payload_size(&out, &[]) > target {
            out.pop();
            return out;
        }
    }
}

/// The capacities: Fortress T and the Fortress maximum (TL) of a cover.
fn t_and_fortress_caps(cover: &[u8]) -> (usize, usize) {
    let img = phasm_core::codec::jpeg::JpegImage::from_bytes(cover).unwrap();
    let t = phasm_core::stego::armor::fortress_tile::t_max_frame_bytes(&img).saturating_sub(50);
    let all = phasm_core::stego::armor::pipeline::armor_capacity_info(cover).unwrap().fortress_capacity;
    (t, all)
}

#[test]
fn fortress_tl_takes_long_messages_and_survives_geometry() {
    let cover = textured_cover();
    let (t_cap, cap) = t_and_fortress_caps(&cover);
    assert!(cap > t_cap + 100, "Fortress TL holds more than Fortress T ({cap} vs {t_cap})");
    let msg = text_of_payload((t_cap + cap) / 2, 7);
    let (st, info) = armor_encode_opts(&cover, &msg, "pw", &ArmorEncodeOpts::default()).unwrap();
    assert!(info.fortress && info.fortress_tl, "a message between the two capacities takes Fortress TL");
    let (px, w, h) = rgb_of(&st);
    let ok = |b: &[u8], what: &str| {
        let r = armor_decode(b, "pw");
        assert!(matches!(&r, Ok((p, _)) if p.text == msg), "{what}: {:?}", r.as_ref().err());
    };
    ok(&st, "untouched");
    ok(&jpeg(&px, w, h, 75), "re-save q75");
    ok(&jpeg(&rotate(&px, w, h, 3.0), w, h, 85), "rotate 3 deg");
    ok(&jpeg(&crop(&px, w, 37, 21, w - 200, h - 150), w - 200, h - 150, 85), "crop at (37, 21)");
    let (sw, sh) = (w * 3 / 4, h * 3 / 4);
    ok(&jpeg(&phasm_core::resample_rgb(&px, w, h, sw, sh).unwrap(), sw, sh, 85), "rescale 0.75");
    for mirror in [false, true] {
        for q in 0..4 {
            if q == 0 && !mirror {
                continue;
            }
            let (t, tw, th) = turn(&px, w, h, q, mirror);
            ok(&jpeg(&t, tw, th, 85), &format!("{q} quarter turns, mirror {mirror}"));
        }
    }
    assert!(armor_decode(&st, "wrong").is_err());
}

/// The Fortress capacity (T or TL) is one number per cover, the exact
/// Fortress / Standard boundary for every passphrase, the compact frame too:
/// the apps' HUD shows what core writes.
#[test]
fn fortress_tl_capacity_is_the_boundary_for_every_passphrase() {
    let cover = textured_cover();
    let info = phasm_core::stego::armor::pipeline::armor_capacity_info(&cover).unwrap();
    for (pass, cap) in [("pw", info.fortress_capacity), ("x", info.fortress_capacity), ("a longer passphrase", info.fortress_capacity), ("", info.fortress_capacity_compact)] {
        // Payload exactly the capacity: search seeds until one hits it.
        let at = (1..400u64).map(|s| text_of_payload(cap, s)).find(|m| phasm_core::stego::payload::compressed_payload_size(m, &[]) == cap).unwrap();
        let over = (1..400u64).map(|s| text_of_payload(cap + 1, s)).find(|m| phasm_core::stego::payload::compressed_payload_size(m, &[]) == cap + 1).unwrap();
        let (_, i_at) = armor_encode_opts(&cover, &at, pass, &ArmorEncodeOpts::default()).unwrap();
        let (_, i_over) = armor_encode_opts(&cover, &over, pass, &ArmorEncodeOpts::default()).unwrap();
        assert!(i_at.fortress && i_at.fortress_tl, "{pass:?}: at capacity -> Fortress TL");
        assert!(!i_over.fortress, "{pass:?}: one byte over -> Standard");
    }
}

/// A JPEG whose chroma components use the luma quantization table (some
/// encoders write one table): the Fortress pre-settle must requantize every
/// component with the one QF75 table its slot then holds, or the colours
/// come out wrong.
#[test]
fn fortress_keeps_colours_when_chroma_shares_the_luma_table() {
    let (w, h) = (640u32, 480u32);
    let px: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let g = (128.0 + 60.0 * ((x as f64 / 23.0).sin() + (y as f64 / 17.0).cos())) as u8;
            [g, 255 - g, ((x * 3 + y) % 200) as u8 + 30]
        })
        .collect();
    let mut cover = deep_cover_jpeg(&px, w, h, 90).unwrap();
    // SOF0: FF C0, length, precision, height, width, Nf, then (id, HV, Tq) per component.
    let sof = cover.windows(2).position(|m| m == [0xFF, 0xC0]).unwrap();
    assert_eq!(cover[sof + 9], 3);
    let tq_y = cover[sof + 12];
    cover[sof + 15] = tq_y;
    cover[sof + 18] = tq_y;
    let stego = armor_encode(&cover, "shared table", "pw").unwrap();
    assert_eq!(armor_decode(&stego, "pw").unwrap().0.text, "shared table");
    let (a, _, _) = rgb_of(&cover);
    let (b, _, _) = rgb_of(&stego);
    // Mean |R - G| difference: chroma decoded with the wrong table shifts it.
    let chroma = |p: &[u8]| -> Vec<f64> { p.chunks(3).map(|c| c[0] as f64 - c[1] as f64).collect() };
    let (ca, cb) = (chroma(&a), chroma(&b));
    let mad = ca.iter().zip(&cb).map(|(x, y)| (x - y).abs()).sum::<f64>() / ca.len() as f64;
    assert!(mad < 4.0, "mean |R - G| change {mad:.2}");
}

/// Rows `y0 .. y0 + ch` and columns `x0 .. x0 + cw` of an RGB image `w` wide.
fn crop_rgb(px: &[u8], w: u32, x0: u32, y0: u32, cw: u32, ch: u32) -> Vec<u8> {
    (y0..y0 + ch)
        .flat_map(|y| px[((y * w + x0) * 3) as usize..((y * w + x0 + cw) * 3) as usize].to_vec())
        .collect()
}

/// A cover side under 32 blocks (one header sub-tile) would leave header bits
/// without a copy: Fortress (T and TL) has no capacity there. From 32 blocks
/// on, a message at the Fortress capacity decodes.
#[test]
fn fortress_needs_32_blocks_on_both_sides() {
    let (px, w, h) = rgb_of(&textured_cover());
    for (side, fits) in [(248u32, false), (256, true)] {
        for (cw, ch) in [(w, side), (side, h)] {
            let cover = deep_cover_jpeg(&crop_rgb(&px, w, 0, 0, cw, ch), cw, ch, 65).unwrap();
            let info = phasm_core::stego::armor::pipeline::armor_capacity_info(&cover).unwrap();
            if !fits {
                assert_eq!((info.fortress_capacity, info.fortress_capacity_compact), (0, 0), "{cw}x{ch}");
                let (_, ei) = armor_encode_opts(&cover, "hi", "pw", &ArmorEncodeOpts::default()).unwrap();
                assert!(!ei.fortress, "{cw}x{ch}: Standard Armor");
                continue;
            }
            for (pass, cap) in [("pw", info.fortress_capacity), ("", info.fortress_capacity_compact)] {
                assert!(cap > 0, "{cw}x{ch} {pass:?}");
                let msg = text_of_payload(cap, 11);
                let (st, ei) = armor_encode_opts(&cover, &msg, pass, &ArmorEncodeOpts::default()).unwrap();
                assert!(ei.fortress, "{cw}x{ch} {pass:?}: at capacity -> Fortress");
                assert_eq!(armor_decode(&st, pass).unwrap().0.text, msg, "{cw}x{ch} {pass:?}");
            }
        }
    }
}

/// A crop narrower than a header sub-tile (32 blocks) can miss every copy of
/// some header bits; the decoder tries them both ways.
#[test]
fn fortress_t_survives_a_strip_crop() {
    let msg = "strip";
    let opts = ArmorEncodeOpts { fortress_tiled: true, ..ArmorEncodeOpts::default() };
    let (st, info) = armor_encode_opts(&textured_cover(), msg, "pw", &opts).unwrap();
    assert!(info.fortress && !info.fortress_tl);
    let (px, w, h) = rgb_of(&st);
    for (x0, y0, cw, ch) in [(0, 80, w, 200), (256, 0, 192, h), (8, 488, w - 16, 176)] {
        let crop = jpeg(&crop_rgb(&px, w, x0, y0, cw, ch), cw, ch, 90);
        assert_eq!(armor_decode(&crop, "pw").unwrap().0.text, msg, "crop {cw}x{ch} at ({x0}, {y0})");
    }
}

/// Padding (dummy) blocks that break libjpeg's rule: AC non-zero, or a DC
/// other than the left neighbour's (right edge) or the last block of the
/// MCU's row above (bottom edge).
fn padding_rule_violations(bytes: &[u8]) -> (usize, usize) {
    let img = phasm_core::JpegImage::from_bytes(bytes).unwrap();
    let fi = img.frame_info().clone();
    let (mut n, mut bad) = (0, 0);
    for ci in 0..fi.components.len() {
        let g = img.dct_grid(ci);
        let hs = fi.components[ci].h_sampling as usize;
        let (rw, rt) = (fi.comp_blocks_wide(ci), fi.comp_blocks_tall(ci));
        for br in 0..g.blocks_tall() {
            for bc in 0..g.blocks_wide() {
                if br < rt && bc < rw {
                    continue;
                }
                n += 1;
                let dc = if br < rt { g.get(br, bc - 1, 0, 0) } else { g.get(br - 1, bc / hs * hs + hs - 1, 0, 0) };
                if g.get(br, bc, 0, 0) != dc || (1..64).any(|k| g.get(br, bc, k / 8, k % 8) != 0) {
                    bad += 1;
                }
            }
        }
    }
    (n, bad)
}

/// R3.27: Fortress moves real block DCs; the padding blocks must still be
/// what libjpeg writes (an odd luma block count on both sides gives padding
/// on the right and at the bottom).
#[test]
fn fortress_padding_blocks_follow_libjpeg() {
    let src = std::fs::read("test-vectors/image/photo_640x480_q75_420.jpg").unwrap();
    let (px, w, h) = rgb_of(&src);
    let (cw, ch) = (1272, 868);
    let up = phasm_core::resample_rgb(&px, w, h, cw, ch).unwrap();
    let cover = deep_cover_jpeg(&up, cw, ch, 65).unwrap();
    let (n, bad) = padding_rule_violations(&cover);
    assert!(n > 0, "the cover must have padding blocks");
    assert_eq!(bad, 0, "the cover writer follows libjpeg");
    let opts = ArmorEncodeOpts { fortress_tiled: true, ..ArmorEncodeOpts::default() };
    let (stego, info) = armor_encode_opts(&cover, "padding stays libjpeg", "pw", &opts).unwrap();
    assert!(info.fortress);
    assert_eq!(padding_rule_violations(&stego), (n, 0));
    assert_eq!(armor_decode(&stego, "pw").unwrap().0.text, "padding stays libjpeg");
}
