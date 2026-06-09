// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! VP.8 — integration tests for the AV1 sequence_header_obu parser.
//!
//! Validates:
//!
//! 1. **Synthetic test** — hand-construct an SH OBU with known field
//!    values via a bit-writer helper; verify the parser extracts each
//!    field correctly.
//!
//! 2. **Real rav1e round-trip** — encode a small YUV via
//!    `encode_frame_with_phasm_tee` (the actual production path),
//!    extract the SH OBU via the VP.M.2 splitter, parse it, and
//!    verify the fields make sense (dimensions match input, default
//!    profile 4:2:0 8-bit).

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder", feature = "video"))]

use std::sync::Arc;

use phasm_core::codec::mp4::av1_obu_split::{
    parse_sequence_header_obu, split_av1_into_samples, OBU_SEQUENCE_HEADER,
};
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

// ───────────────────────────────────────────────────────────────────
//  Synthetic SH OBU bit-writer
// ───────────────────────────────────────────────────────────────────

/// MSB-first bit writer. AV1 syntax (§ 4.10.1) is defined this way.
struct BitWriter {
    bytes: Vec<u8>,
    bit_pos: usize, // bit offset from start of `bytes`
}

impl BitWriter {
    fn new() -> Self {
        Self { bytes: Vec::new(), bit_pos: 0 }
    }

    fn write_bits(&mut self, value: u64, n: usize) {
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            let byte_idx = self.bit_pos >> 3;
            let bit_idx_in_byte = 7 - (self.bit_pos & 7);
            if byte_idx >= self.bytes.len() {
                self.bytes.push(0);
            }
            self.bytes[byte_idx] |= bit << bit_idx_in_byte;
            self.bit_pos += 1;
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Build a minimum-viable SH OBU per spec § 5.5.1: single operating
/// point, no timing info, no decoder model, no display delay, no
/// frame ID numbers, no order hint, etc. Sufficient to exercise the
/// parser's field extraction without the conditional bit fields the
/// rav1e output also avoids.
fn make_minimal_sh_obu(
    seq_profile: u8,
    seq_level_idx_0: u8,
    width: u32,
    height: u32,
    high_bitdepth: bool,
) -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.write_bits(seq_profile as u64, 3);
    bw.write_bits(0, 1); // still_picture
    bw.write_bits(0, 1); // reduced_still_picture_header
    bw.write_bits(0, 1); // timing_info_present_flag
    bw.write_bits(0, 1); // initial_display_delay_present_flag
    bw.write_bits(0, 5); // operating_points_cnt_minus_1 = 0 (single op)
    bw.write_bits(0, 12); // operating_point_idc[0]
    bw.write_bits(seq_level_idx_0 as u64, 5);
    if seq_level_idx_0 > 7 {
        bw.write_bits(0, 1); // seq_tier[0]
    }
    // initial_display_delay_present_flag=0 → skip its conditional bit

    // frame_width_bits / frame_height_bits — choose 11 (covers up to 2048 dims)
    let fwb = 11;
    let fhb = 11;
    bw.write_bits((fwb - 1) as u64, 4);
    bw.write_bits((fhb - 1) as u64, 4);
    bw.write_bits((width - 1) as u64, fwb);
    bw.write_bits((height - 1) as u64, fhb);

    bw.write_bits(0, 1); // frame_id_numbers_present_flag
    bw.write_bits(0, 1); // use_128x128_superblock
    bw.write_bits(0, 1); // enable_filter_intra
    bw.write_bits(0, 1); // enable_intra_edge_filter
    bw.write_bits(0, 1); // enable_interintra_compound
    bw.write_bits(0, 1); // enable_masked_compound
    bw.write_bits(0, 1); // enable_warped_motion
    bw.write_bits(0, 1); // enable_dual_filter
    bw.write_bits(0, 1); // enable_order_hint = 0 → no enable_jnt_comp / mvs / order_hint_bits
    bw.write_bits(1, 1); // seq_choose_screen_content_tools = 1
                          // → seq_force_screen_content_tools = SELECT (sentinel = 2 > 0)
                          // Spec § 5.5.1 requires the integer_mv block in that case:
    bw.write_bits(1, 1); // seq_choose_integer_mv = 1
                          // → seq_force_integer_mv = SELECT, no extra bit needed

    bw.write_bits(0, 1); // enable_superres
    bw.write_bits(0, 1); // enable_cdef
    bw.write_bits(0, 1); // enable_restoration

    // color_config()
    bw.write_bits(high_bitdepth as u64, 1);
    if seq_profile == 2 && high_bitdepth {
        bw.write_bits(0, 1); // twelve_bit = 0
    }
    if seq_profile != 1 {
        bw.write_bits(0, 1); // monochrome = 0
    }
    bw.write_bits(0, 1); // color_description_present_flag = 0 (UNSPECIFIED)
    bw.write_bits(0, 1); // color_range
    // For seq_profile=0: subsampling fixed at 1/1. chroma_sample_position u(2):
    if seq_profile == 0 || seq_profile == 2 {
        bw.write_bits(0, 2); // chroma_sample_position = 0 (UNKNOWN)
    }
    bw.write_bits(0, 1); // separate_uv_deltas

    // Wrap the bit-payload in an OBU header + ULEB size.
    let payload = bw.into_bytes();
    let mut obu = Vec::new();
    // OBU header: forbidden=0 | type=SH(4 bits) | extension=0 | has_size=1 | reserved=0
    obu.push(((OBU_SEQUENCE_HEADER & 0x0F) << 3) | 0b010);
    // ULEB128(payload.len())
    let mut n = payload.len() as u64;
    loop {
        let mut byte = (n & 0x7F) as u8;
        n >>= 7;
        if n != 0 {
            byte |= 0x80;
        }
        obu.push(byte);
        if n == 0 {
            break;
        }
    }
    obu.extend(payload);
    obu
}

// ───────────────────────────────────────────────────────────────────
//  Tests
// ───────────────────────────────────────────────────────────────────

#[test]
fn parse_synthetic_sh_yuv420_8bit_profile0() {
    let sh = make_minimal_sh_obu(0, 8, 1920, 1080, false);
    let info = parse_sequence_header_obu(&sh).expect("parser must accept minimal SH");

    assert_eq!(info.seq_profile, 0);
    assert_eq!(info.seq_level_idx_0, 8);
    assert_eq!(info.seq_tier_0, 0);
    assert_eq!(info.max_frame_width, 1920);
    assert_eq!(info.max_frame_height, 1080);
    assert!(!info.high_bitdepth);
    assert!(!info.twelve_bit);
    assert!(!info.monochrome);
    assert!(info.chroma_subsampling_x);
    assert!(info.chroma_subsampling_y);
    assert_eq!(info.chroma_sample_position, 0);
}

#[test]
fn parse_synthetic_sh_yuv420_10bit_profile0() {
    let sh = make_minimal_sh_obu(0, 8, 1920, 1080, true);
    let info = parse_sequence_header_obu(&sh).expect("parser must accept high-bitdepth SH");
    assert!(info.high_bitdepth);
    assert!(!info.twelve_bit);
}

#[test]
fn parse_synthetic_sh_low_level_no_tier_bit() {
    // seq_level_idx_0 = 4 → no seq_tier_0 bit emitted (per spec).
    let sh = make_minimal_sh_obu(0, 4, 1280, 720, false);
    let info = parse_sequence_header_obu(&sh).expect("parser must accept level <= 7");
    assert_eq!(info.seq_level_idx_0, 4);
    assert_eq!(info.seq_tier_0, 0);
    assert_eq!(info.max_frame_width, 1280);
    assert_eq!(info.max_frame_height, 720);
}

#[test]
fn parser_rejects_empty_input() {
    assert!(parse_sequence_header_obu(&[]).is_none());
}

#[test]
fn parser_rejects_wrong_obu_type() {
    // Build a FRAME OBU instead of SH — type=6.
    let frame_obu = vec![
        ((6u8 & 0x0F) << 3) | 0b010, // header
        0x00,                         // size = 0
    ];
    assert!(parse_sequence_header_obu(&frame_obu).is_none());
}

#[test]
fn parser_handles_real_rav1e_sh_obu() {
    // Encode a tiny synthetic YUV through phasm-rav1e (the actual
    // production path), extract the SH OBU, parse it.
    let w = 64usize;
    let h = 64usize;

    let mut config = EncoderConfig {
        width: w,
        height: h,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: 30,
        ..Default::default()
    };
    config.low_latency = true;
    config.speed_settings.multiref = false;
    let config = Arc::new(config);
    let mut sequence = Sequence::new(&config);
    sequence.enable_large_lru = false;
    let mut fi = FrameInvariants::<u8>::new_key_frame(
        config.clone(),
        Arc::new(sequence),
        0,
        Box::new([]),
    );
    fi.enable_segmentation = false;

    let mut frame_in = make_frame::<u8>(w, h, ChromaSampling::Cs420);
    // Fill with a gradient so the encoder emits something non-trivial.
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    let mut y = vec![0u8; y_size];
    for (i, b) in y.iter_mut().enumerate() {
        *b = (i % 256) as u8;
    }
    frame_in.planes[0].copy_from_raw_u8(&y, w, 1);
    frame_in.planes[1].copy_from_raw_u8(&vec![128u8; uv_size], w / 2, 1);
    frame_in.planes[2].copy_from_raw_u8(&vec![128u8; uv_size], w / 2, 1);

    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame_in));
    let inter_cfg = make_inter_config(&config);
    let (av1_bytes, _recording) = encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);

    // Split + extract SH.
    let split = split_av1_into_samples(&av1_bytes).unwrap();
    assert!(!split.sequence_header_obu.is_empty(), "rav1e must emit an SH");

    // Parse it via VP.8.
    let info = parse_sequence_header_obu(&split.sequence_header_obu)
        .expect("VP.8 parser must accept rav1e's SH OBU");

    // Verify against the encoder config + sane defaults.
    assert_eq!(info.max_frame_width, w as u32, "parsed width matches encoder config");
    assert_eq!(info.max_frame_height, h as u32, "parsed height matches encoder config");
    assert_eq!(info.seq_profile, 0, "rav1e default = profile 0");
    assert!(!info.high_bitdepth, "8-bit encoder config");
    assert!(!info.monochrome, "Cs420 = not monochrome");
    assert!(info.chroma_subsampling_x, "Cs420 has x-subsampling");
    assert!(info.chroma_subsampling_y, "Cs420 has y-subsampling");
}
