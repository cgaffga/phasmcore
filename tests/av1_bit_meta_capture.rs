// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase B.1.1.a — encoder-side AcSignMeta capture validation.
//!
//! Verifies that `encode_frame_with_phasm_tee` populates the new
//! `bit_meta` field on `PhasmTileRecording` and the new
//! `reconstructed_planes` + `frame_qindex` fields on
//! `PhasmFrameRecording`. Each AC-sign-tagged emission's metadata
//! is checked for plausibility: plane in [0,2], pixel coords within
//! frame dims, TX dims in valid range, scan_pos within TX area.
//!
//! Dav1d-side symmetric meta capture + cross-side strict-parity
//! test land in B.1.1.b (separate commit). The strict-parity test
//! is what guards against encoder/decoder spec drift on the meta
//! contract (mirror of the W3.10.4-fix cursor parity test).

#![cfg(all(feature = "av1-encoder", feature = "av1-backend"))]

use std::sync::Arc;

use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
    PHASM_TAG_AC_COEFF_SIGN, PHASM_TAG_GOLOMB_TAIL_LSB, PHASM_TAG_OTHER,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

fn build_test_frame_state(width: u32, height: u32) -> (FrameInvariants<u8>, FrameState<u8>) {
    let config = Arc::new(EncoderConfig {
        width: width as usize,
        height: height as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: 30,
        ..Default::default()
    });
    let mut sequence = Sequence::new(&config);
    sequence.enable_large_lru = false;
    let mut fi = FrameInvariants::<u8>::new_key_frame(
        config.clone(),
        Arc::new(sequence),
        0,
        Box::new([]),
    );
    fi.enable_segmentation = false;

    // Gradient pattern with enough AC content to ensure non-trivial
    // sign emissions across multiple TX blocks.
    let mut frame = make_frame::<u8>(width as usize, height as usize, ChromaSampling::Cs420);
    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    let mut yuv = vec![0u8; y_size + 2 * uv_size];
    for row in 0..h {
        for col in 0..w {
            yuv[row * w + col] = ((row.wrapping_mul(7) + col.wrapping_mul(3)) & 0xff) as u8;
        }
    }
    for row in 0..(h / 2) {
        for col in 0..(w / 2) {
            yuv[y_size + row * (w / 2) + col] =
                ((row.wrapping_mul(11) + col.wrapping_mul(5) + 13) & 0xff) as u8;
            yuv[y_size + uv_size + row * (w / 2) + col] =
                ((row.wrapping_mul(13) + col.wrapping_mul(7) + 31) & 0xff) as u8;
        }
    }
    frame.planes[0].copy_from_raw_u8(&yuv[..y_size], w, 1);
    frame.planes[1].copy_from_raw_u8(&yuv[y_size..y_size + uv_size], w / 2, 1);
    frame.planes[2].copy_from_raw_u8(
        &yuv[y_size + uv_size..y_size + 2 * uv_size],
        w / 2,
        1,
    );

    let fs = FrameState::new_with_frame(&fi, Arc::new(frame));
    (fi, fs)
}

#[test]
fn b1_1a_bit_meta_invariants_at_128x128() {
    let (fi, mut fs) = build_test_frame_state(128, 128);
    let inter_cfg = make_inter_config(fi.config.as_ref());
    let (packet, recording) = encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);

    assert!(!packet.is_empty(), "packet empty");
    assert_eq!(recording.tiles.len(), 1, "expected single tile for 128x128");
    assert_eq!(recording.frame_qindex, 30, "frame_qindex should match EncoderConfig.quantizer");

    let tile = &recording.tiles[0];
    let n_positions = tile.bit_positions.len();
    assert!(n_positions > 0, "no emissions captured");

    // The three parallel Vecs must stay length-aligned.
    assert_eq!(
        tile.bit_meta.len(),
        n_positions,
        "bit_meta length must match bit_positions"
    );
    assert_eq!(
        tile.bit_tags.len(),
        n_positions,
        "bit_tags length must match bit_positions"
    );

    // Count AC_COEFF_SIGN-tagged emissions; verify each has plausible
    // metadata (non-zero pixel coords, valid TX shape, valid scan_pos).
    let mut ac_count = 0usize;
    let mut other_count = 0usize;
    let mut golomb_count = 0usize;
    for i in 0..n_positions {
        let tag = tile.bit_tags[i];
        let meta = tile.bit_meta[i];
        match tag {
            PHASM_TAG_AC_COEFF_SIGN => {
                ac_count += 1;
                // Plane must be 0 (Y), 1 (U), or 2 (V).
                assert!(meta.plane <= 2, "AC entry {i}: plane {} > 2", meta.plane);
                // Pixel coords within 128x128 luma or 64x64 chroma.
                let plane_w = if meta.plane == 0 { 128 } else { 64 };
                let plane_h = if meta.plane == 0 { 128 } else { 64 };
                assert!(
                    meta.plane_px_x < plane_w as u16,
                    "AC entry {i}: plane_px_x {} >= plane width {}",
                    meta.plane_px_x,
                    plane_w
                );
                assert!(
                    meta.plane_px_y < plane_h as u16,
                    "AC entry {i}: plane_px_y {} >= plane height {}",
                    meta.plane_px_y,
                    plane_h
                );
                // TX dims log2 in 2..=6 (AV1 TX sizes 4 to 64).
                assert!(
                    (2..=6).contains(&meta.tx_width_log2),
                    "AC entry {i}: tx_width_log2 {} not in 2..=6",
                    meta.tx_width_log2
                );
                assert!(
                    (2..=6).contains(&meta.tx_height_log2),
                    "AC entry {i}: tx_height_log2 {} not in 2..=6",
                    meta.tx_height_log2
                );
                // scan_pos within TX area.
                let tx_area = 1u32
                    << (meta.tx_width_log2 as u32 + meta.tx_height_log2 as u32);
                assert!(
                    (meta.scan_pos as u32) < tx_area,
                    "AC entry {i}: scan_pos {} >= tx_area {} (tx {}x{})",
                    meta.scan_pos,
                    tx_area,
                    1 << meta.tx_width_log2,
                    1 << meta.tx_height_log2
                );
                // scan_pos > 0 because AC = non-DC = scan position > 0.
                assert!(
                    meta.scan_pos > 0,
                    "AC entry {i}: scan_pos must be > 0 (DC is not AC)"
                );
            }
            PHASM_TAG_GOLOMB_TAIL_LSB | PHASM_TAG_OTHER => {
                if tag == PHASM_TAG_GOLOMB_TAIL_LSB {
                    golomb_count += 1;
                } else {
                    other_count += 1;
                }
                // Non-AC tags inherit whatever meta was last set
                // (sticky state, mirror of tag stickiness). phasm-core
                // never reads meta for non-AC entries, so the stale
                // value is harmless. No assertion on meta content.
            }
            unknown => panic!("unexpected tag value {} at entry {}", unknown, i),
        }
    }

    eprintln!(
        "[b1_1a] 128x128 captured: {} AC_COEFF_SIGN + {} GOLOMB_TAIL_LSB + {} OTHER = {} total",
        ac_count, golomb_count, other_count, n_positions
    );

    assert!(
        ac_count > 100,
        "expected >100 AC_COEFF_SIGN emissions on 128x128 gradient frame, got {}",
        ac_count
    );
}

#[test]
fn b1_1a_reconstructed_planes_captured() {
    let (fi, mut fs) = build_test_frame_state(128, 128);
    let inter_cfg = make_inter_config(fi.config.as_ref());
    let (_packet, recording) = encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);

    let rec = &recording.reconstructed_planes;
    assert_eq!(rec.planes[0].cfg.width, 128, "Y plane width");
    assert_eq!(rec.planes[0].cfg.height, 128, "Y plane height");
    assert_eq!(rec.planes[1].cfg.width, 64, "U plane width (4:2:0)");
    assert_eq!(rec.planes[1].cfg.height, 64, "U plane height (4:2:0)");
    assert_eq!(rec.planes[2].cfg.width, 64, "V plane width");
    assert_eq!(rec.planes[2].cfg.height, 64, "V plane height");

    // The reconstructed Y plane should contain real content (not all
    // zero, not all neutral-gray) — the encoder went through deblock +
    // CDEF + LR. Read from the visible region: plane.data is laid
    // out with top/left padding rows for filter taps; the visible
    // top-left pixel is at offset (yorigin * stride + xorigin).
    let cfg = &rec.planes[0].cfg;
    let visible_w = cfg.width;
    let visible_h = cfg.height;
    let visible_start = cfg.yorigin * cfg.stride + cfg.xorigin;
    let mid_y = visible_h / 2;
    let mid_row_offset = visible_start + mid_y * cfg.stride;
    let mut nonzero = 0;
    let mut non_gray = 0;
    for x in 0..visible_w {
        let v = rec.planes[0].data[mid_row_offset + x];
        if v != 0 {
            nonzero += 1;
        }
        if !(0x7e..=0x86).contains(&v) {
            non_gray += 1;
        }
    }
    assert!(
        nonzero > visible_w / 4,
        "reconstructed Y mid-row is mostly zero ({} of {} non-zero)",
        nonzero,
        visible_w
    );
    assert!(
        non_gray > visible_w / 4,
        "reconstructed Y mid-row is mostly neutral-gray ({} of {} non-gray) — encoder may have produced empty output",
        non_gray,
        visible_w
    );
}
