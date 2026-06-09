// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Phase B.2.2/B.2.3 diagnostic: strict joint parity test for the
//! Tier 1 channel set (AC sign + golomb tail) between encoder and
//! decoder.
//!
//! After enrolling GOLOMB_TAIL_LSB in the orchestrator's combined
//! cover vector (B.2.3), the W3.10.5 round-trip started failing
//! with ExtractionFailed. This test isolates whether the failure
//! is at the cursor-parity level (counts/order disagree between
//! sides) or downstream.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::sync::Arc;

use phasm_core::codec::av1::stego::decoder::decode_with_recording;
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
    PHASM_TAG_AC_COEFF_SIGN, PHASM_TAG_GOLOMB_TAIL_LSB,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;
const QUANTIZER: usize = 30;

fn make_gradient() -> Vec<u8> {
    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let mut yuv = vec![128u8; w * h * 3 / 2];
    for y in 0..h {
        for x in 0..w {
            yuv[y * w + x] = ((x + y * 2) & 0xff) as u8;
        }
    }
    yuv
}

#[test]
fn b2_2_joint_tier1_parity_diagnostic() {
    let yuv = make_gradient();
    let config = Arc::new(EncoderConfig {
        width: WIDTH as usize,
        height: HEIGHT as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: QUANTIZER,
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

    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let mut frame = make_frame::<u8>(w, h, ChromaSampling::Cs420);
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    frame.planes[0].copy_from_raw_u8(&yuv[..y_size], w, 1);
    frame.planes[1].copy_from_raw_u8(&yuv[y_size..y_size + uv_size], w / 2, 1);
    frame.planes[2]
        .copy_from_raw_u8(&yuv[y_size + uv_size..y_size + 2 * uv_size], w / 2, 1);

    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame));
    let inter_cfg = make_inter_config(&config);
    let (av1_bytes, recording) = encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);

    let tile = &recording.tiles[0];
    let enc_joint: Vec<(u8, u8)> = tile
        .bit_positions
        .iter()
        .zip(&tile.bit_tags)
        .filter(|&(_, &tag)| {
            tag == PHASM_TAG_AC_COEFF_SIGN || tag == PHASM_TAG_GOLOMB_TAIL_LSB
        })
        .map(|(&(_, value), &tag)| (value as u8, tag))
        .collect();

    let decoded = decode_with_recording(&av1_bytes).expect("decode");
    use core_dav1d_sys::{
        DAV1D_PHASM_TAG_AC_COEFF_SIGN, DAV1D_PHASM_TAG_GOLOMB_TAIL_LSB,
    };
    let dec_joint: Vec<(u8, u8)> = decoded
        .iter()
        .filter(|p| {
            p.tag == DAV1D_PHASM_TAG_AC_COEFF_SIGN
                || p.tag == DAV1D_PHASM_TAG_GOLOMB_TAIL_LSB
        })
        .map(|p| (p.decoded_value, p.tag))
        .collect();

    let enc_ac = enc_joint
        .iter()
        .filter(|(_, t)| *t == PHASM_TAG_AC_COEFF_SIGN)
        .count();
    let enc_gol = enc_joint
        .iter()
        .filter(|(_, t)| *t == PHASM_TAG_GOLOMB_TAIL_LSB)
        .count();
    let dec_ac = dec_joint
        .iter()
        .filter(|(_, t)| *t == DAV1D_PHASM_TAG_AC_COEFF_SIGN)
        .count();
    let dec_gol = dec_joint
        .iter()
        .filter(|(_, t)| *t == DAV1D_PHASM_TAG_GOLOMB_TAIL_LSB)
        .count();

    eprintln!(
        "[b2_2_parity] ENC: total joint = {} (AC = {} + GOLOMB = {})",
        enc_joint.len(),
        enc_ac,
        enc_gol
    );
    eprintln!(
        "[b2_2_parity] DEC: total joint = {} (AC = {} + GOLOMB = {})",
        dec_joint.len(),
        dec_ac,
        dec_gol
    );

    assert_eq!(enc_ac, dec_ac, "AC_COEFF_SIGN count mismatch");
    assert_eq!(enc_gol, dec_gol, "GOLOMB_TAIL_LSB count mismatch");
    assert_eq!(enc_joint.len(), dec_joint.len(), "joint count mismatch");

    // Now check ORDER + VALUE: enc[i] should equal dec[i] for every i.
    let mut first_mismatch: Option<usize> = None;
    for i in 0..enc_joint.len() {
        if enc_joint[i] != dec_joint[i] {
            first_mismatch = Some(i);
            break;
        }
    }
    match first_mismatch {
        None => eprintln!(
            "[b2_2_parity] STRICT JOINT PARITY GREEN — {} entries match",
            enc_joint.len()
        ),
        Some(i) => {
            // Show context: 10 entries around the first mismatch.
            let lo = i.saturating_sub(5);
            let hi = (i + 5).min(enc_joint.len() - 1);
            eprintln!(
                "[b2_2_parity] MISMATCH at index {} of {}:",
                i,
                enc_joint.len()
            );
            for j in lo..=hi {
                let marker = if j == i { " <==" } else { "" };
                eprintln!(
                    "[b2_2_parity]   [{}] ENC=(v={}, tag={}) DEC=(v={}, tag={}){}",
                    j,
                    enc_joint[j].0,
                    enc_joint[j].1,
                    dec_joint[j].0,
                    dec_joint[j].1,
                    marker
                );
            }
            panic!("joint parity broken at index {}", i);
        }
    }
}
