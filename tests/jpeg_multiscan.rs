// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Multi-scan JPEGs (fix list R3.1 / R3.2, 2026-11-03): a scan with one
//! component is non-interleaved and codes `ceil(xᵢ/8) × ceil(yᵢ/8)` blocks
//! (T.81 A.2.2), not the MCU-padded grid. Before the fix, progressive files
//! with an odd luma block count failed to parse (`HuffmanDecode`), a
//! single-component DC scan put the luma DCs into the wrong blocks, and a
//! 2x2-sampled grayscale baseline file failed.
//!
//! Vectors in `test-vectors/image/multiscan/` (libjpeg-turbo 3.1 `cjpeg` /
//! `jpegtran`; crops of `photo_640x480_q75_420.jpg` decoded with `djpeg`).
//! Every multi-scan file has a baseline twin with the *same coefficients*
//! written by libjpeg's own transcoder, the reference for the parse:
//!
//! ```sh
//! djpeg -ppm photo_640x480_q75_420.jpg > src.ppm
//! convert src.ppm -crop 264x200+101+77 +repage c264.ppm   # 33x25 luma blocks
//! convert src.ppm -crop 216x270+301+101 +repage c216.ppm  # 27x34 (1080x1350 / 5)
//! convert c264.ppm -colorspace Gray pgm:g264.pgm
//! cjpeg -quality 85 -sample 2x2 -progressive c264.ppm > prog420_264x200.jpg
//! # (`jpegtran prog420_264x200.jpg` gives base420_264x200.jpg byte for byte)
//! cjpeg -quality 85 -sample 2x2 -progressive c216.ppm > prog420_216x270.jpg
//! jpegtran prog420_216x270.jpg > prog420_216x270_base.jpg
//! cjpeg -quality 85 -sample 2x2 c264.ppm > base420_264x200.jpg
//! # one DC scan per component, then one AC scan per component:
//! printf '0: 0 0 0 0;\n1: 0 0 0 0;\n2: 0 0 0 0;\n0: 1 63 0 0;\n1: 1 63 0 0;\n2: 1 63 0 0;\n' > sepdc.scans
//! jpegtran -scans sepdc.scans base420_264x200.jpg > sepdc420_264x200.jpg
//! jpegtran -restart 7B -progressive base420_264x200.jpg > prog420rst_264x200.jpg
//! jpegtran -restart 7B base420_264x200.jpg > prog420rst_264x200_base.jpg
//! # sequential (SOF0) with two scans: Y alone, then Cb + Cr interleaved:
//! printf '0;\n1 2;\n' > seq.scans
//! jpegtran -scans seq.scans base420_264x200.jpg > seq420_264x200.jpg
//! cjpeg -quality 85 -sample 2x2 g264.pgm > gray22_264x200.jpg     # 1 component, 2x2
//! jpegtran -progressive gray22_264x200.jpg > gray22prog_264x200.jpg  # (libjpeg makes it 1x1)
//! ```
//!
//! The Armor vectors (`armor_fortress_488x392_*.jpg`): a Fortress stego of
//! a 488x392 cover (61x49 luma blocks) from [`regenerate_armor_vector`],
//! re-saved as progressive losslessly and lossily:
//!
//! ```sh
//! jpegtran -progressive armor_fortress_488x392.jpg > armor_fortress_488x392_tranprog.jpg
//! jpegtran armor_fortress_488x392_tranprog.jpg > armor_fortress_488x392_tranprog_base.jpg  # padding reference
//! djpeg -ppm armor_fortress_488x392.jpg | cjpeg -quality 85 -progressive > armor_fortress_488x392_prog85.jpg
//! ```

use phasm_core::{JpegImage, smart_decode};
use std::path::Path;

const DIR: &str = "test-vectors/image/multiscan";

fn read(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DIR).join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// The coefficients of every block a decoder shows (the non-interleaved
/// block counts, T.81 A.2.2) agree; the MCU padding may differ (libjpeg
/// copies a DC into it, a progressive file does not code it).
fn assert_same_real_coefficients(a: &JpegImage, b: &JpegImage, what: &str) {
    let (fa, fb) = (a.frame_info(), b.frame_info());
    assert_eq!((fa.width, fa.height), (fb.width, fb.height), "{what}: size");
    assert_eq!(a.num_components(), b.num_components(), "{what}: components");
    for c in 0..a.num_components() {
        let (rw, rt) = (fa.comp_blocks_wide(c), fa.comp_blocks_tall(c));
        assert_eq!((rw, rt), (fb.comp_blocks_wide(c), fb.comp_blocks_tall(c)), "{what}: comp {c} block counts");
        let (ga, gb) = (a.dct_grid(c), b.dct_grid(c));
        for br in 0..rt {
            for bc in 0..rw {
                assert_eq!(ga.block(br, bc), gb.block(br, bc), "{what}: comp {c} block ({br}, {bc})");
            }
        }
    }
}

/// Blocks outside the coded area of a non-interleaved scan have AC zero (the
/// DC too if `include_dc`; a multi-scan parse fills the DC like libjpeg).
fn assert_dummy_blocks_zero(img: &JpegImage, include_dc: bool, what: &str) {
    let fi = img.frame_info();
    for c in 0..img.num_components() {
        let g = img.dct_grid(c);
        let (rw, rt) = (fi.comp_blocks_wide(c), fi.comp_blocks_tall(c));
        for br in 0..g.blocks_tall() {
            for bc in 0..g.blocks_wide() {
                if br >= rt || bc >= rw {
                    let from = if include_dc { 0 } else { 1 };
                    assert!(g.block(br, bc)[from..].iter().all(|&v| v == 0), "{what}: comp {c} dummy block ({br}, {bc}) not zero");
                }
            }
        }
    }
}

/// RGB pixels as an independent decoder (the `image` crate) sees them.
fn pixels(jpeg: &[u8]) -> Vec<u8> {
    image::load_from_memory(jpeg).expect("reference decode").to_rgb8().into_raw()
}

/// A multi-scan file parses to libjpeg's coefficients, and phasm's
/// (baseline) re-write decodes to the same pixels as libjpeg's baseline twin.
/// `reference_reads_it`: the `image` crate's decoder reads the multi-scan
/// file itself correctly (it reads only the first scan of a sequential
/// multi-scan file).
fn check_against_twin(multi: &str, twin: &str, expect_dummy_luma: bool, reference_reads_it: bool) {
    let (mb, tb) = (read(multi), read(twin));
    let m = JpegImage::from_bytes(&mb).unwrap_or_else(|e| panic!("{multi}: parse failed: {e}"));
    let t = JpegImage::from_bytes(&tb).unwrap_or_else(|e| panic!("{twin}: parse failed: {e}"));
    assert_same_real_coefficients(&m, &t, multi);
    let fi = m.frame_info();
    let g = m.dct_grid(0);
    assert_eq!(
        (g.blocks_wide(), g.blocks_tall()) != (fi.comp_blocks_wide(0), fi.comp_blocks_tall(0)),
        expect_dummy_luma,
        "{multi}: the vector should exercise an odd luma block count"
    );
    assert_dummy_blocks_zero(&m, false, multi);
    // The padding is what libjpeg's transcoder writes into it (jctrans.c:
    // AC 0, DC of the last real block / of the MCU's row above): the whole
    // MCU-padded grid equals the `jpegtran` baseline twin's, which codes it.
    for c in 0..m.num_components() {
        assert_eq!(m.dct_grid(c).coeffs(), t.dct_grid(c).coeffs(), "{multi}: comp {c} MCU-padded grid vs libjpeg's twin");
    }
    // Phasm writes baseline: same coefficients after a re-parse, same pixels.
    let out = m.to_bytes().unwrap();
    let r = JpegImage::from_bytes(&out).unwrap();
    for c in 0..m.num_components() {
        assert_eq!(m.dct_grid(c).coeffs(), r.dct_grid(c).coeffs(), "{multi}: re-write comp {c}");
    }
    let reference = pixels(&tb);
    if reference_reads_it {
        assert!(pixels(&mb) == reference, "{multi}: reference decoder disagrees with libjpeg's twin");
    }
    assert!(pixels(&out) == reference, "{multi}: phasm's re-write decodes differently");
}

#[test]
fn progressive_420_odd_block_counts_match_libjpeg() {
    // 264x200: 33x25 luma blocks in a 34x26 MCU grid (both sides odd).
    check_against_twin("prog420_264x200.jpg", "base420_264x200.jpg", true, true);
    // 216x270 (1080x1350 / 5): 27x34 luma blocks in a 28x34 grid.
    check_against_twin("prog420_216x270.jpg", "prog420_216x270_base.jpg", true, true);
    let p = JpegImage::from_bytes(&read("prog420_264x200.jpg")).unwrap();
    assert_dummy_blocks_zero(&p, false, "prog420_264x200.jpg");
}

#[test]
fn progressive_with_restart_intervals() {
    // Restart intervals count blocks in non-interleaved scans.
    check_against_twin("prog420rst_264x200.jpg", "prog420rst_264x200_base.jpg", true, true);
}

#[test]
fn separate_dc_scans_are_non_interleaved() {
    // R3.2: one DC scan per component (jpegtran -scans). Before the fix the
    // luma DCs landed in the wrong blocks without any error.
    check_against_twin("sepdc420_264x200.jpg", "base420_264x200.jpg", true, true);
}

#[test]
fn sequential_multi_scan_decodes_every_component() {
    // SOF0 with a luma scan and a chroma scan: before, only the first scan
    // was read.
    check_against_twin("seq420_264x200.jpg", "base420_264x200.jpg", true, false);
}

#[test]
fn grayscale_with_2x2_sampling() {
    // One component, sampling 2x2: the MCU grid is 34x26 blocks, the scan
    // codes 33x25. jpegtran's progressive copy (1x1) holds the same
    // coefficients.
    let b = read("gray22_264x200.jpg");
    let img = JpegImage::from_bytes(&b).expect("2x2 grayscale parses");
    let fi = img.frame_info();
    assert_eq!((fi.components[0].h_sampling, fi.components[0].v_sampling), (2, 2));
    assert_eq!((fi.comp_blocks_wide(0), fi.comp_blocks_tall(0)), (33, 25));
    assert_eq!((img.dct_grid(0).blocks_wide(), img.dct_grid(0).blocks_tall()), (34, 26));
    assert_dummy_blocks_zero(&img, true, "gray22");
    let p = JpegImage::from_bytes(&read("gray22prog_264x200.jpg")).unwrap();
    assert_same_real_coefficients(&img, &p, "gray22 vs its progressive copy");
    // The re-write stays a valid 2x2 grayscale file (non-interleaved, no
    // padding blocks coded) with the same pixels.
    let out = img.to_bytes().unwrap();
    let r = JpegImage::from_bytes(&out).unwrap();
    assert_eq!(img.dct_grid(0).coeffs(), r.dct_grid(0).coeffs());
    assert!(pixels(&out) == pixels(&b));
    assert!(pixels(&b) == pixels(&read("gray22prog_264x200.jpg")));
}

/// The padding rule on the Armor vector (odd x odd luma block counts, so
/// both the right and the bottom padding are dummy): a lossless progressive
/// re-save (`jpegtran -progressive`) parses to the same MCU-padded grid as
/// libjpeg's own baseline conversion of it
/// (`jpegtran armor_fortress_488x392_tranprog.jpg > ..._tranprog_base.jpg`).
#[test]
fn progressive_padding_matches_libjpeg_transcode() {
    let prog = JpegImage::from_bytes(&read("armor_fortress_488x392_tranprog.jpg")).unwrap();
    let twin = JpegImage::from_bytes(&read("armor_fortress_488x392_tranprog_base.jpg")).unwrap();
    let g = prog.dct_grid(0);
    assert_eq!((g.blocks_wide(), g.blocks_tall()), (62, 50));
    for c in 0..prog.num_components() {
        assert_eq!(prog.dct_grid(c).coeffs(), twin.dct_grid(c).coeffs(), "comp {c}");
    }
    // Some padding DC is non-zero (a DC-0 padding would be the tell).
    assert!((0..g.blocks_tall()).any(|br| g.block(br, 61)[0] != 0), "right padding column has DC");
    assert!((0..g.blocks_wide()).any(|bc| g.block(49, bc)[0] != 0), "bottom padding row has DC");
    // Explicit rule: right column = DC of the real block to the left; the
    // bottom padding row = DC of the last block of the MCU's row above
    // (the corner block included).
    for br in 0..49 {
        assert_eq!(g.block(br, 61)[0], g.block(br, 60)[0], "right padding ({br}, 61)");
    }
    for bc in (0..62).step_by(2) {
        assert_eq!(g.block(49, bc)[0], g.block(48, bc + 1)[0], "bottom padding (49, {bc})");
        assert_eq!(g.block(49, bc + 1)[0], g.block(48, bc + 1)[0], "bottom padding (49, {})", bc + 1);
    }
}

#[test]
fn ghost_round_trips_on_multi_scan_covers() {
    for name in ["prog420_264x200.jpg", "sepdc420_264x200.jpg", "seq420_264x200.jpg", "gray22_264x200.jpg"] {
        let cover = read(name);
        let stego = phasm_core::ghost_encode(&cover, "multi-scan", "pw").unwrap_or_else(|e| panic!("{name}: {e}"));
        let decoded = phasm_core::ghost_decode(&stego, "pw").unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(decoded.text, "multi-scan", "{name}");
    }
}

const ARMOR_MESSAGE: &str = "R3.1 prog";
const ARMOR_PASS: &str = "r3-armor";

#[test]
fn armor_survives_a_progressive_re_save_at_an_odd_block_count() {
    let base = read("armor_fortress_488x392.jpg");
    let img = JpegImage::from_bytes(&base).unwrap();
    let fi = img.frame_info();
    assert_eq!((fi.comp_blocks_wide(0), fi.comp_blocks_tall(0)), (61, 49));
    for name in ["armor_fortress_488x392.jpg", "armor_fortress_488x392_tranprog.jpg", "armor_fortress_488x392_prog85.jpg"] {
        let b = read(name);
        assert!(name == "armor_fortress_488x392.jpg" || JpegImage::from_bytes(&b).is_ok(), "{name}: parse");
        let (payload, q) = smart_decode(&b, ARMOR_PASS).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(payload.text, ARMOR_MESSAGE, "{name}");
        assert_eq!(q.mode, 2, "{name}: Armor");
        assert!(q.fortress_used, "{name}: Fortress");
    }
}

/// Writes `armor_fortress_488x392.jpg` (a Fortress stego of a 488x392 crop
/// of `photo_640x480_q75_420.jpg`, cover written by core at QF65) to
/// `$PHASM_TMP/phasm-multiscan/`; the progressive copies are made from it
/// with the commands in the module docs.
#[test]
#[ignore = "generator for the checked-in Armor vectors"]
fn regenerate_armor_vector() {
    let src = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("test-vectors/image/photo_640x480_q75_420.jpg")).unwrap();
    let rgb = image::load_from_memory(&src).unwrap().to_rgb8();
    let (w, h) = (488u32, 392u32);
    let (x0, y0) = (71u32, 41u32);
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            px.extend_from_slice(&rgb.get_pixel(x, y).0);
        }
    }
    let cover = phasm_core::deep_cover_jpeg(&px, w, h, phasm_core::ARMOR_COVER_QUALITY).unwrap();
    println!("capacity {:?}", phasm_core::armor_capacity_info(&cover).map(|c| (c.fortress_capacity, c.stdm_capacity)));
    let (stego, q) = phasm_core::armor_encode_with_options(&cover, ARMOR_MESSAGE, ARMOR_PASS, false).unwrap();
    assert!(q.fortress, "the message must take Fortress");
    let dir = Path::new(&std::env::var("PHASM_TMP").unwrap_or_else(|_| "/tmp".into())).join("phasm-multiscan");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("armor_fortress_488x392.jpg"), &stego).unwrap();
    println!("wrote {}", dir.join("armor_fortress_488x392.jpg").display());
}
