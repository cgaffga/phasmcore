// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 slice header parsing.
//!
//! Implements ITU-T H.264 Section 7.3.3. Parses the slice header to extract
//! slice type, QP, frame number, and — critically — the `data_bit_offset`
//! where macroblock data begins.

use super::bitstream::RbspReader;
use super::sps::{Pps, Sps};
use super::{H264Error, NalType};

/// H.264 slice type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    P = 0,
    B = 1,
    I = 2,
    SP = 3,
    SI = 4,
}

impl SliceType {
    /// Parse from the raw slice_type value (0-9).
    /// Values 5-9 mean "all MBs in this slice are this type" (same mapping).
    pub fn from_raw(raw: u32) -> Result<Self, H264Error> {
        match raw % 5 {
            0 => Ok(Self::P),
            1 => Ok(Self::B),
            2 => Ok(Self::I),
            3 => Ok(Self::SP),
            4 => Ok(Self::SI),
            _ => Err(H264Error::InvalidParameterSet(format!(
                "invalid slice_type: {raw}"
            ))),
        }
    }

    pub fn is_intra(self) -> bool {
        matches!(self, Self::I | Self::SI)
    }

    pub fn is_inter(self) -> bool {
        matches!(self, Self::P | Self::B | Self::SP)
    }
}

impl std::fmt::Display for SliceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::P => write!(f, "P"),
            Self::B => write!(f, "B"),
            Self::I => write!(f, "I"),
            Self::SP => write!(f, "SP"),
            Self::SI => write!(f, "SI"),
        }
    }
}

/// Parsed H.264 slice header.
#[derive(Debug, Clone)]
pub struct SliceHeader {
    /// Address of the first macroblock in this slice.
    pub first_mb_in_slice: u32,
    /// Slice type (I, P, B, SP, SI).
    pub slice_type: SliceType,
    /// PPS referenced by this slice.
    pub pps_id: u8,
    /// Frame number (modulo max_frame_num).
    pub frame_num: u32,
    /// True if this is a field picture (not frame).
    pub field_pic_flag: bool,
    /// True if this is the bottom field.
    pub bottom_field_flag: bool,
    /// IDR picture ID (IDR slices only).
    pub idr_pic_id: u32,
    /// Picture order count LSB (poc_type == 0 only).
    pub pic_order_cnt_lsb: u32,
    /// Active number of L0 reference indices (after slice override).
    /// For P/SP/B slices, may override `pps.num_ref_idx_l0_default`.
    pub num_ref_idx_l0_active: u8,
    /// Active number of L1 reference indices (B-slices only).
    pub num_ref_idx_l1_active: u8,
    /// Direct mode predictor selection for B-slices (spec § 7.4.3).
    /// `true`  → spatial direct (mobile encoder default).
    /// `false` → temporal direct.
    /// Phase 6E-A locks `true`; the decoder needs the parsed value
    /// for compliance with arbitrary B-slice streams. Always
    /// `false` for non-B slices.
    pub direct_spatial_mv_pred_flag: bool,
    /// `cabac_init_idc` from the slice header (spec § 7.4.3) — selects
    /// which P/B context-init column to use (slot 1, 2, or 3 in the
    /// internal `CabacInitSlot::PIdc{0,1,2}` enum). 0 for I/SI slices.
    /// Phase 6E-A captures this so B-slice CABAC contexts initialize
    /// correctly; the encoder side already emits a fixed value (we
    /// continue to use idc=0 by default for stealth-via-mimicry of
    /// mobile encoder defaults).
    pub cabac_init_idc: u8,
    /// Slice QP delta (relative to PPS init_qp).
    pub slice_qp_delta: i32,
    /// Deblocking filter control.
    pub disable_deblocking_filter_idc: u8,
    pub slice_alpha_c0_offset_div2: i32,
    pub slice_beta_offset_div2: i32,
    /// Bit offset in the RBSP where macroblock data starts.
    /// The CAVLC decoder reads from `rbsp[data_bit_offset / 8..]` at
    /// bit position `data_bit_offset % 8`.
    pub data_bit_offset: usize,
    // Derived
    /// Slice QP = 26 + pic_init_qp_minus26 + slice_qp_delta.
    pub slice_qp: i32,
}

/// Parse an H.264 slice header from RBSP data.
///
/// Reference: ITU-T H.264 Section 7.3.3
pub fn parse_slice_header(
    rbsp: &[u8],
    sps: &Sps,
    pps: &Pps,
    nal_type: NalType,
    nal_ref_idc: u8,
) -> Result<SliceHeader, H264Error> {
    let mut r = RbspReader::new(rbsp);

    let first_mb_in_slice = r.read_ue()?;
    let slice_type_raw = r.read_ue()?;
    let slice_type = SliceType::from_raw(slice_type_raw)?;
    let pps_id = r.read_ue()? as u8;

    // colour_plane_id (if separate_colour_plane_flag)
    if sps.separate_colour_plane_flag {
        r.skip_bits(2)?;
    }

    let frame_num = r.read_bits(sps.log2_max_frame_num)?;

    let mut field_pic_flag = false;
    let mut bottom_field_flag = false;
    if !sps.frame_mbs_only_flag {
        field_pic_flag = r.read_bit()?;
        if field_pic_flag {
            bottom_field_flag = r.read_bit()?;
        }
    }

    let idr_pic_id = if nal_type.is_idr() {
        r.read_ue()?
    } else {
        0
    };

    let mut pic_order_cnt_lsb = 0u32;
    if sps.pic_order_cnt_type == 0 {
        pic_order_cnt_lsb = r.read_bits(sps.log2_max_pic_order_cnt_lsb)?;
        if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
            let _delta_pic_order_cnt_bottom = r.read_se()?;
        }
    }

    if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
        let _delta_pic_order_cnt_0 = r.read_se()?;
        if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
            let _delta_pic_order_cnt_1 = r.read_se()?;
        }
    }

    if pps.redundant_pic_cnt_present_flag {
        let _redundant_pic_cnt = r.read_ue()?;
    }

    // Slice type-dependent header fields. § 6E-A2 captures
    // direct_spatial_mv_pred_flag (was read+discarded). False for
    // non-B slices.
    let direct_spatial_mv_pred_flag = if slice_type == SliceType::B {
        r.read_bit()?
    } else {
        false
    };

    // num_ref_idx_active_override: defaults from PPS, overridden in slice header.
    let mut num_ref_idx_l0_active = pps.num_ref_idx_l0_default;
    let mut num_ref_idx_l1_active = pps.num_ref_idx_l1_default;

    if slice_type == SliceType::P
        || slice_type == SliceType::SP
        || slice_type == SliceType::B
    {
        let num_ref_idx_active_override = r.read_bit()?;
        if num_ref_idx_active_override {
            num_ref_idx_l0_active = r.read_ue()? as u8 + 1;
            if slice_type == SliceType::B {
                num_ref_idx_l1_active = r.read_ue()? as u8 + 1;
            }
        }
    }

    // ref_pic_list_modification (skip)
    if slice_type != SliceType::I && slice_type != SliceType::SI {
        skip_ref_pic_list_modification(&mut r, slice_type)?;
    }

    // §Stealth.L4.6.4 — pred_weight_table (spec § 7.3.3.2). Only present
    // when (PPS weighted_pred_flag && (P|SP)) ||
    //      (PPS weighted_bipred_idc == 1 && B).
    // Phasm uses weighted_bipred_idc = 2 (implicit, no explicit table)
    // and weighted_pred_flag is configurable in the PPS.
    let pwt_p = pps.weighted_pred_flag
        && (slice_type == SliceType::P || slice_type == SliceType::SP);
    let pwt_b = pps.weighted_bipred_idc == 1 && slice_type == SliceType::B;
    if pwt_p || pwt_b {
        skip_pred_weight_table(
            &mut r,
            sps,
            num_ref_idx_l0_active,
            num_ref_idx_l1_active,
            slice_type,
        )?;
    }

    // dec_ref_pic_marking (skip) — only present for reference pictures
    if nal_ref_idc > 0 {
        if nal_type.is_idr() {
            let _no_output_of_prior_pics = r.read_bit()?;
            let _long_term_reference = r.read_bit()?;
        } else {
            let adaptive_ref_pic_marking = r.read_bit()?;
            if adaptive_ref_pic_marking {
                loop {
                    let op = r.read_ue()?;
                    if op == 0 {
                        break;
                    }
                    match op {
                        1 | 3 => {
                            let _ = r.read_ue()?;
                        }
                        2 => {
                            let _ = r.read_ue()?;
                        }
                        4 | 5 => {}
                        6 => {
                            let _ = r.read_ue()?;
                        }
                        _ => break,
                    }
                }
            }
        }
    }

    // CABAC init — § 6E-A2 captures cabac_init_idc (was discarded).
    // Selects which of the three P/B context-init columns to use
    // (CabacInitSlot::PIdc{0,1,2}).
    let mut cabac_init_idc = 0u8;
    if pps.entropy_coding_mode_flag
        && slice_type != SliceType::I
        && slice_type != SliceType::SI
    {
        cabac_init_idc = r.read_ue()? as u8;
    }

    let slice_qp_delta = r.read_se()?;

    if slice_type == SliceType::SP || slice_type == SliceType::SI {
        if slice_type == SliceType::SP {
            r.skip_bits(1)?; // sp_for_switch_flag
        }
        let _slice_qs_delta = r.read_se()?;
    }

    let mut disable_deblocking_filter_idc = 0u8;
    let mut slice_alpha_c0_offset_div2 = 0i32;
    let mut slice_beta_offset_div2 = 0i32;

    if pps.deblocking_filter_control_present_flag {
        disable_deblocking_filter_idc = r.read_ue()? as u8;
        if disable_deblocking_filter_idc != 1 {
            slice_alpha_c0_offset_div2 = r.read_se()?;
            slice_beta_offset_div2 = r.read_se()?;
        }
    }

    // data_bit_offset: where macroblock data begins
    // For CAVLC, the slice data starts right after the header (byte-aligned
    // only if entropy_coding_mode_flag is set, which it isn't for CAVLC).
    // Actually, for CAVLC the data is NOT byte-aligned — it continues from
    // the current bit position.
    let data_bit_offset = r.bits_read();

    let slice_qp = 26 + pps.pic_init_qp_minus26 + slice_qp_delta;

    Ok(SliceHeader {
        first_mb_in_slice,
        slice_type,
        pps_id,
        frame_num,
        field_pic_flag,
        bottom_field_flag,
        idr_pic_id,
        pic_order_cnt_lsb,
        num_ref_idx_l0_active,
        num_ref_idx_l1_active,
        direct_spatial_mv_pred_flag,
        cabac_init_idc,
        slice_qp_delta,
        disable_deblocking_filter_idc,
        slice_alpha_c0_offset_div2,
        slice_beta_offset_div2,
        data_bit_offset,
        slice_qp,
    })
}

/// Skip ref_pic_list_modification syntax (H.264 Section 7.3.3.1).
/// §Stealth.L4.6.4 — Skip the `pred_weight_table` block per spec §
/// 7.3.3.2 without applying any weights (phasm uses unweighted MC).
///
/// Layout (4:2:0, ChromaArrayType=1):
/// - `luma_log2_weight_denom`             : ue(v)
/// - `chroma_log2_weight_denom`           : ue(v)
/// - For each i in 0..num_ref_idx_l0_active:
///   - `luma_weight_l0_flag`              : u(1)
///   - if flag: `luma_weight_l0[i]`       : se(v) + `luma_offset_l0[i]` : se(v)
///   - `chroma_weight_l0_flag`            : u(1)
///   - if flag: 2× (`chroma_weight_l0[i][j]` : se(v) + `chroma_offset_l0[i][j]` : se(v))
/// - For B-slices, repeat the loop for L1.
///
/// For our 4:2:0 streams ChromaArrayType is always 1, so chroma fields
/// are always present. The skipper does not apply any weights — phasm
/// reconstruction uses the unweighted formula either way (the table is
/// emitted purely as an L4 fingerprint match against x264-medium).
fn skip_pred_weight_table(
    r: &mut RbspReader<'_>,
    sps: &Sps,
    num_ref_idx_l0_active: u8,
    num_ref_idx_l1_active: u8,
    slice_type: SliceType,
) -> Result<(), H264Error> {
    let chroma_array_type = if sps.separate_colour_plane_flag {
        0
    } else {
        sps.chroma_format_idc
    };

    let _ = r.read_ue()?; // luma_log2_weight_denom
    if chroma_array_type != 0 {
        let _ = r.read_ue()?; // chroma_log2_weight_denom
    }

    let skip_list_for = |r: &mut RbspReader<'_>, n_active: u8| -> Result<(), H264Error> {
        for _ in 0..n_active {
            let luma_flag = r.read_bit()?;
            if luma_flag {
                let _ = r.read_se()?; // luma_weight
                let _ = r.read_se()?; // luma_offset
            }
            if chroma_array_type != 0 {
                let chroma_flag = r.read_bit()?;
                if chroma_flag {
                    for _ in 0..2 {
                        let _ = r.read_se()?; // chroma_weight
                        let _ = r.read_se()?; // chroma_offset
                    }
                }
            }
        }
        Ok(())
    };

    skip_list_for(r, num_ref_idx_l0_active)?;
    if slice_type == SliceType::B {
        skip_list_for(r, num_ref_idx_l1_active)?;
    }
    Ok(())
}

fn skip_ref_pic_list_modification(
    r: &mut RbspReader<'_>,
    slice_type: SliceType,
) -> Result<(), H264Error> {
    // ref_pic_list_modification_flag_l0
    if slice_type != SliceType::I && slice_type != SliceType::SI {
        let flag_l0 = r.read_bit()?;
        if flag_l0 {
            loop {
                let op = r.read_ue()?;
                if op == 3 {
                    break;
                }
                let _ = r.read_ue()?; // abs_diff_pic_num_minus1 or long_term_pic_num
            }
        }
    }
    // ref_pic_list_modification_flag_l1 (B slices only)
    if slice_type == SliceType::B {
        let flag_l1 = r.read_bit()?;
        if flag_l1 {
            loop {
                let op = r.read_ue()?;
                if op == 3 {
                    break;
                }
                let _ = r.read_ue()?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BitWriter for constructing test RBSP data.
    struct BitWriter {
        data: Vec<u8>,
        current: u8,
        bit_pos: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self { data: Vec::new(), current: 0, bit_pos: 0 }
        }
        fn write_bit(&mut self, val: bool) {
            if val { self.current |= 1 << (7 - self.bit_pos); }
            self.bit_pos += 1;
            if self.bit_pos == 8 { self.data.push(self.current); self.current = 0; self.bit_pos = 0; }
        }
        fn write_bits(&mut self, val: u32, n: u8) {
            for i in (0..n).rev() { self.write_bit((val >> i) & 1 != 0); }
        }
        fn write_ue(&mut self, val: u32) {
            if val == 0 { self.write_bit(true); return; }
            let n = 32 - (val + 1).leading_zeros();
            for _ in 0..n - 1 { self.write_bit(false); }
            self.write_bits(val + 1, n as u8);
        }
        fn write_se(&mut self, val: i32) {
            let ue = if val > 0 { (2 * val - 1) as u32 } else if val < 0 { (2 * (-val)) as u32 } else { 0 };
            self.write_ue(ue);
        }
        fn align(&mut self) {
            if self.bit_pos > 0 { self.data.push(self.current); self.current = 0; self.bit_pos = 0; }
        }
    }

    fn make_test_sps() -> Sps {
        Sps {
            profile_idc: 66,
            constraint_set_flags: 0xC0,
            level_idc: 30,
            sps_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            qpprime_y_zero_transform_bypass_flag: false,
            log2_max_frame_num: 4,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb: 6,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            max_num_ref_frames: 1,
            gaps_in_frame_num_allowed: false,
            pic_width_in_mbs: 20,
            pic_height_in_map_units: 15,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: false,
            frame_cropping_flag: false,
            crop_left: 0, crop_right: 0, crop_top: 0, crop_bottom: 0,
            width_in_pixels: 320,
            height_in_pixels: 240,
            pic_size_in_mbs: 300,
        }
    }

    fn make_test_pps() -> Pps {
        Pps {
            pps_id: 0,
            sps_id: 0,
            entropy_coding_mode_flag: false,
            bottom_field_pic_order_in_frame_present_flag: false,
            num_slice_groups_minus1: 0,
            num_ref_idx_l0_default: 1,
            num_ref_idx_l1_default: 1,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26: 0,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            deblocking_filter_control_present_flag: true,
            constrained_intra_pred_flag: false,
            redundant_pic_cnt_present_flag: false,
            transform_8x8_mode_flag: false,
            second_chroma_qp_index_offset: 0,
        }
    }

    #[test]
    fn parse_idr_i_slice_header() {
        let sps = make_test_sps();
        let pps = make_test_pps();

        let mut bits = BitWriter::new();
        bits.write_ue(0); // first_mb_in_slice = 0
        bits.write_ue(7); // slice_type = 7 → I (all MBs)
        bits.write_ue(0); // pps_id = 0
        bits.write_bits(0, 4); // frame_num (log2_max=4 → 4 bits) = 0
        // frame_mbs_only=true → no field_pic_flag
        // IDR: idr_pic_id
        bits.write_ue(0); // idr_pic_id = 0
        // poc_type=0: pic_order_cnt_lsb (log2_max=6 → 6 bits)
        bits.write_bits(0, 6);
        // dec_ref_pic_marking (IDR)
        bits.write_bit(false); // no_output_of_prior_pics
        bits.write_bit(false); // long_term_reference
        // slice_qp_delta
        bits.write_se(2); // qp_delta = +2
        // deblocking filter control
        bits.write_ue(0); // disable_deblocking_filter_idc = 0
        bits.write_se(0); // alpha offset
        bits.write_se(0); // beta offset
        bits.align();

        let hdr = parse_slice_header(&bits.data, &sps, &pps, NalType::SLICE_IDR, 3).unwrap();
        assert_eq!(hdr.first_mb_in_slice, 0);
        assert_eq!(hdr.slice_type, SliceType::I);
        assert_eq!(hdr.pps_id, 0);
        assert_eq!(hdr.frame_num, 0);
        assert_eq!(hdr.idr_pic_id, 0);
        assert_eq!(hdr.slice_qp_delta, 2);
        assert_eq!(hdr.slice_qp, 28); // 26 + 0 + 2
        assert!(hdr.data_bit_offset > 0);
    }

    #[test]
    fn parse_p_slice_header() {
        let sps = make_test_sps();
        let pps = make_test_pps();

        let mut bits = BitWriter::new();
        bits.write_ue(0); // first_mb_in_slice
        bits.write_ue(5); // slice_type = 5 → P (all MBs)
        bits.write_ue(0); // pps_id
        bits.write_bits(1, 4); // frame_num = 1
        bits.write_bits(2, 6); // pic_order_cnt_lsb = 2
        // ref_pic_list_modification
        bits.write_bit(false); // ref_pic_list_modification_flag_l0 = 0
        // dec_ref_pic_marking (non-IDR)
        bits.write_bit(false); // adaptive_ref_pic_marking = 0
        // slice_qp_delta
        bits.write_se(-1);
        // deblocking
        bits.write_ue(0);
        bits.write_se(0);
        bits.write_se(0);
        bits.align();

        // nal_ref_idc=0: non-reference P-slice (no dec_ref_pic_marking)
        let hdr = parse_slice_header(&bits.data, &sps, &pps, NalType::SLICE, 0).unwrap();
        assert_eq!(hdr.slice_type, SliceType::P);
        assert_eq!(hdr.frame_num, 1);
        assert_eq!(hdr.slice_qp_delta, -1);
        assert_eq!(hdr.slice_qp, 25); // 26 + 0 + (-1)
    }

    #[test]
    fn slice_type_from_raw() {
        assert_eq!(SliceType::from_raw(0).unwrap(), SliceType::P);
        assert_eq!(SliceType::from_raw(2).unwrap(), SliceType::I);
        assert_eq!(SliceType::from_raw(5).unwrap(), SliceType::P); // 5 = all-P
        assert_eq!(SliceType::from_raw(7).unwrap(), SliceType::I); // 7 = all-I
    }

    #[test]
    fn slice_type_properties() {
        assert!(SliceType::I.is_intra());
        assert!(!SliceType::I.is_inter());
        assert!(!SliceType::P.is_intra());
        assert!(SliceType::P.is_inter());
    }

    /// §6E-A2 — B-slice header parsing captures direct_spatial_mv_pred_flag
    /// + cabac_init_idc + num_ref_idx_l1_active.
    #[test]
    fn parse_b_slice_header_captures_b_specific_fields() {
        let sps = make_test_sps();
        let mut pps = make_test_pps();
        pps.entropy_coding_mode_flag = true; // CABAC for cabac_init_idc

        let mut bits = BitWriter::new();
        bits.write_ue(0); // first_mb_in_slice
        bits.write_ue(6); // slice_type = 6 → B (all MBs)
        bits.write_ue(0); // pps_id
        bits.write_bits(2, 4); // frame_num = 2
        bits.write_bits(4, 6); // pic_order_cnt_lsb = 4
        // direct_spatial_mv_pred_flag = 1 (Phase 6E-A default)
        bits.write_bit(true);
        // num_ref_idx_active_override = 0 (use PPS defaults)
        bits.write_bit(false);
        // ref_pic_list_modification: B-slice has both L0 + L1 flags
        bits.write_bit(false); // ref_pic_list_modification_flag_l0
        bits.write_bit(false); // ref_pic_list_modification_flag_l1
        // dec_ref_pic_marking — nal_ref_idc=0 (B is non-reference) so
        // skip
        // cabac_init_idc = 1
        bits.write_ue(1);
        // slice_qp_delta
        bits.write_se(0);
        // deblocking
        bits.write_ue(0);
        bits.write_se(0);
        bits.write_se(0);
        bits.align();

        let hdr = parse_slice_header(
            &bits.data, &sps, &pps,
            NalType::SLICE,
            /* nal_ref_idc */ 0,
        ).unwrap();
        assert_eq!(hdr.slice_type, SliceType::B);
        assert!(hdr.direct_spatial_mv_pred_flag);
        assert_eq!(hdr.cabac_init_idc, 1);
        assert_eq!(hdr.num_ref_idx_l0_active, pps.num_ref_idx_l0_default);
        assert_eq!(hdr.num_ref_idx_l1_active, pps.num_ref_idx_l1_default);
        assert_eq!(hdr.frame_num, 2);
        assert_eq!(hdr.pic_order_cnt_lsb, 4);
    }

    /// §6E-A2 — non-B slices have direct_spatial_mv_pred_flag = false
    /// (the field is meaningless for them; we set it to false to avoid
    /// any surprise).
    #[test]
    fn non_b_slice_has_direct_spatial_false() {
        let sps = make_test_sps();
        let pps = make_test_pps();

        let mut bits = BitWriter::new();
        bits.write_ue(0);
        bits.write_ue(7); // I (all MBs)
        bits.write_ue(0);
        bits.write_bits(0, 4);
        bits.write_ue(0);
        bits.write_bits(0, 6);
        bits.write_bit(false);
        bits.write_bit(false);
        bits.write_se(0);
        bits.write_ue(0);
        bits.write_se(0);
        bits.write_se(0);
        bits.align();

        let hdr = parse_slice_header(
            &bits.data, &sps, &pps,
            NalType::SLICE_IDR, 3,
        ).unwrap();
        assert_eq!(hdr.slice_type, SliceType::I);
        assert!(!hdr.direct_spatial_mv_pred_flag);
        // I-slice has no cabac_init_idc field; defaults to 0.
        assert_eq!(hdr.cabac_init_idc, 0);
    }
}
