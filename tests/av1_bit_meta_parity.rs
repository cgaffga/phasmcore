// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase B.1.1.b — cross-side strict-parity test for AcSignMeta.
//!
//! Encodes a frame via phasm-rav1e's `encode_frame_with_phasm_tee`,
//! capturing encoder-side `bit_meta`. Decodes the same bytes via
//! phasm-dav1d's `decode_with_recording`, capturing decoder-side
//! `meta` on each AC sign emission. Asserts the two parallel
//! sequences (filtered to AC_COEFF_SIGN tag) are bit-identical.
//!
//! This is the gate that catches encoder/decoder spec divergence on
//! the meta contract — mirror of W3.10.4-fix's strict cursor parity
//! test that caught the DC golomb tag drift bug.
//!
//! If this test ever fails, the bug is one of:
//!   (a) Encoder + decoder disagree on which TX shape applies (e.g.,
//!       tx_size_log2 differs because we computed it from a different
//!       source on one side)
//!   (b) Pixel-coord arithmetic differs (e.g., chroma subsampling
//!       handled one way encoder-side, another way decoder-side)
//!   (c) Scan-order interpretation differs (e.g., scan_pos uses raster
//!       index on encoder but scan-step-index on decoder)
//!   (d) Tag-firing-order divergence (encoder fires AC_COEFF_SIGN
//!       at a different bit than decoder)
//!
//! Any (a)-(d) breaks J-UNIWARD cost evaluation on the decode side
//! and prevents B.1.4 self-steganalyzer's AoSO adapter from running
//! coherently.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::sync::Arc;

use phasm_core::codec::av1::stego::decoder::decode_with_recording;
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
    PHASM_TAG_AC_COEFF_SIGN,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

fn build_test_packet_128x128() -> (Vec<u8>, phasm_rav1e::phasm_stego::PhasmFrameRecording<u8>) {
    let config = Arc::new(EncoderConfig {
        width: 128,
        height: 128,
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

    let mut frame = make_frame::<u8>(128, 128, ChromaSampling::Cs420);
    let w = 128;
    let h = 128;
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

    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame));
    let inter_cfg = make_inter_config(&config);
    encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg)
}

/// Compact equivalent of phasm-rav1e's `AcSignMeta` for cross-side
/// comparison. The encoder side comes from `phasm_rav1e::AcSignMeta`
/// (via `PhasmTileRecording.bit_meta`); the decoder side from
/// `core_dav1d_sys::Dav1dPhasmAcSignMeta` (via `DecodedCoverPosition.
/// meta`). We project both into this neutral struct so the assertion
/// doesn't depend on which side's type it sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompareMeta {
    plane: u8,
    plane_px_x: u16,
    plane_px_y: u16,
    tx_width_log2: u8,
    tx_height_log2: u8,
    tx_type: u8,
    scan_pos: u16,
}

fn enc_meta_to_compare(m: &phasm_rav1e::phasm_stego::AcSignMeta) -> CompareMeta {
    CompareMeta {
        plane: m.plane,
        plane_px_x: m.plane_px_x,
        plane_px_y: m.plane_px_y,
        tx_width_log2: m.tx_width_log2,
        tx_height_log2: m.tx_height_log2,
        tx_type: m.tx_type,
        scan_pos: m.scan_pos,
    }
}

fn dec_meta_to_compare(m: &core_dav1d_sys::Dav1dPhasmAcSignMeta) -> CompareMeta {
    CompareMeta {
        plane: m.plane,
        plane_px_x: m.plane_px_x,
        plane_px_y: m.plane_px_y,
        tx_width_log2: m.tx_width_log2,
        tx_height_log2: m.tx_height_log2,
        tx_type: m.tx_type,
        scan_pos: m.scan_pos,
    }
}

#[test]
fn b1_1b_ac_sign_meta_strict_parity() {
    use core_dav1d_sys::DAV1D_PHASM_TAG_AC_COEFF_SIGN;

    let (packet, recording) = build_test_packet_128x128();
    assert_eq!(recording.tiles.len(), 1);

    // Encoder-side: walk the tile recorder and pull (meta, tag) for
    // every AC_COEFF_SIGN-tagged emission.
    let tile = &recording.tiles[0];
    let enc_ac_metas: Vec<CompareMeta> = tile
        .bit_meta
        .iter()
        .zip(tile.bit_tags.iter())
        .filter_map(|(m, &t)| {
            if t == PHASM_TAG_AC_COEFF_SIGN {
                Some(enc_meta_to_compare(m))
            } else {
                None
            }
        })
        .collect();

    // Decoder-side: decode the same bytes, capture (meta, tag) for
    // every AC_COEFF_SIGN-tagged emission via the new meta_hook.
    let decoded = decode_with_recording(&packet).expect("decode");
    let dec_ac_metas: Vec<CompareMeta> = decoded
        .iter()
        .filter_map(|p| {
            if p.tag == DAV1D_PHASM_TAG_AC_COEFF_SIGN {
                Some(dec_meta_to_compare(&p.meta))
            } else {
                None
            }
        })
        .collect();

    eprintln!(
        "[b1_1b] enc AC count: {} | dec AC count: {}",
        enc_ac_metas.len(),
        dec_ac_metas.len()
    );

    assert_eq!(
        enc_ac_metas.len(),
        dec_ac_metas.len(),
        "AC_COEFF_SIGN count mismatch: encoder captured {} vs decoder {}. \
         Cursor parity is the prerequisite for meta parity — if THIS fails, \
         W3.10.4-fix's tag-parity guarantee has broken.",
        enc_ac_metas.len(),
        dec_ac_metas.len()
    );

    // Compare every entry. On failure, dump the first divergence
    // with both sides' values so the gap is obvious.
    for (i, (enc, dec)) in enc_ac_metas.iter().zip(dec_ac_metas.iter()).enumerate() {
        if enc != dec {
            panic!(
                "B.1.1.b STRICT PARITY FAILURE at AC position {}:\n  \
                 encoder: {:?}\n  decoder: {:?}\n  \
                 (continuing past this would compare {} more entries — \
                 first failure dumped above)",
                i,
                enc,
                dec,
                enc_ac_metas.len() - i - 1
            );
        }
    }

    assert!(
        enc_ac_metas.len() > 100,
        "expected >100 AC entries; cross-side parity gate is trivial otherwise"
    );

    eprintln!(
        "[b1_1b] STRICT PARITY GREEN — {} AC entries match exactly across encoder + decoder",
        enc_ac_metas.len()
    );
}
