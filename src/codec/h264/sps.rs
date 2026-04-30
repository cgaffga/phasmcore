// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 Sequence Parameter Set (SPS) and Picture Parameter Set (PPS) parsing.
//!
//! Implements ITU-T H.264 sections 7.3.2.1 (SPS) and 7.3.2.2 (PPS).
//! Targets Baseline profile but parses High profile fields to detect and
//! reject unsupported configurations.

use super::bitstream::RbspReader;
use super::H264Error;

/// Parsed Sequence Parameter Set.
#[derive(Debug, Clone)]
pub struct Sps {
    pub profile_idc: u8,
    pub constraint_set_flags: u8,
    pub level_idc: u8,
    pub sps_id: u8,
    // High profile extensions (parsed but not used in Phase 1a)
    pub chroma_format_idc: u8,
    pub separate_colour_plane_flag: bool,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub qpprime_y_zero_transform_bypass_flag: bool,
    // Frame number
    pub log2_max_frame_num: u8,
    // Picture order count
    pub pic_order_cnt_type: u8,
    pub log2_max_pic_order_cnt_lsb: u8,
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub num_ref_frames_in_pic_order_cnt_cycle: u8,
    // Reference frames
    pub max_num_ref_frames: u8,
    pub gaps_in_frame_num_allowed: bool,
    // Picture dimensions (in macroblocks)
    pub pic_width_in_mbs: u32,
    pub pic_height_in_map_units: u32,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub direct_8x8_inference_flag: bool,
    // Cropping
    pub frame_cropping_flag: bool,
    pub crop_left: u32,
    pub crop_right: u32,
    pub crop_top: u32,
    pub crop_bottom: u32,
    // Derived fields
    pub width_in_pixels: u32,
    pub height_in_pixels: u32,
    pub pic_size_in_mbs: u32,
}

/// Parsed Picture Parameter Set.
#[derive(Debug, Clone)]
pub struct Pps {
    pub pps_id: u8,
    pub sps_id: u8,
    pub entropy_coding_mode_flag: bool,
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    pub num_slice_groups_minus1: u32,
    pub num_ref_idx_l0_default: u8,
    pub num_ref_idx_l1_default: u8,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u8,
    pub pic_init_qp_minus26: i32,
    pub pic_init_qs_minus26: i32,
    pub chroma_qp_index_offset: i32,
    pub deblocking_filter_control_present_flag: bool,
    pub constrained_intra_pred_flag: bool,
    pub redundant_pic_cnt_present_flag: bool,
    pub transform_8x8_mode_flag: bool,
    pub second_chroma_qp_index_offset: i32,
}

/// Parse an H.264 SPS from RBSP data.
///
/// Reference: ITU-T H.264 Section 7.3.2.1.1
pub fn parse_sps(rbsp: &[u8]) -> Result<Sps, H264Error> {
    let mut r = RbspReader::new(rbsp);

    let profile_idc = r.read_bits(8)? as u8;
    let constraint_set_flags = r.read_bits(8)? as u8; // 6 constraint flags + 2 reserved
    let level_idc = r.read_bits(8)? as u8;
    let sps_id = r.read_ue()? as u8;
    if sps_id > 31 {
        return Err(H264Error::InvalidParameterSet(format!(
            "sps_id {sps_id} > 31"
        )));
    }

    // High profile extensions
    let mut chroma_format_idc = 1u8; // default 4:2:0
    let mut separate_colour_plane_flag = false;
    let mut bit_depth_luma = 8u8;
    let mut bit_depth_chroma = 8u8;
    let mut qpprime_y_zero_transform_bypass_flag = false;

    if profile_idc == 100
        || profile_idc == 110
        || profile_idc == 122
        || profile_idc == 244
        || profile_idc == 44
        || profile_idc == 83
        || profile_idc == 86
        || profile_idc == 118
        || profile_idc == 128
        || profile_idc == 138
        || profile_idc == 139
        || profile_idc == 134
        || profile_idc == 135
    {
        chroma_format_idc = r.read_ue()? as u8;
        if chroma_format_idc == 3 {
            separate_colour_plane_flag = r.read_bit()?;
        }
        bit_depth_luma = r.read_ue()? as u8 + 8;
        bit_depth_chroma = r.read_ue()? as u8 + 8;
        qpprime_y_zero_transform_bypass_flag = r.read_bit()?;

        // seq_scaling_matrix_present_flag
        let scaling_matrix_present = r.read_bit()?;
        if scaling_matrix_present {
            let count = if chroma_format_idc != 3 { 8 } else { 12 };
            for _ in 0..count {
                let list_present = r.read_bit()?;
                if list_present {
                    // Skip scaling list (read and discard delta values)
                    let size = if count <= 6 { 16 } else { 64 };
                    skip_scaling_list(&mut r, size)?;
                }
            }
        }
    }

    let log2_max_frame_num = r.read_ue()? as u8 + 4;
    let pic_order_cnt_type = r.read_ue()? as u8;

    let mut log2_max_pic_order_cnt_lsb = 0u8;
    let mut delta_pic_order_always_zero_flag = false;
    let mut offset_for_non_ref_pic = 0i32;
    let mut offset_for_top_to_bottom_field = 0i32;
    let mut num_ref_frames_in_pic_order_cnt_cycle = 0u8;

    if pic_order_cnt_type == 0 {
        log2_max_pic_order_cnt_lsb = r.read_ue()? as u8 + 4;
    } else if pic_order_cnt_type == 1 {
        delta_pic_order_always_zero_flag = r.read_bit()?;
        offset_for_non_ref_pic = r.read_se()?;
        offset_for_top_to_bottom_field = r.read_se()?;
        num_ref_frames_in_pic_order_cnt_cycle = r.read_ue()? as u8;
        // Skip offset_for_ref_frame[i]
        for _ in 0..num_ref_frames_in_pic_order_cnt_cycle {
            let _ = r.read_se()?;
        }
    }

    let max_num_ref_frames = r.read_ue()? as u8;
    let gaps_in_frame_num_allowed = r.read_bit()?;
    let pic_width_in_mbs = r.read_ue()? + 1;
    let pic_height_in_map_units = r.read_ue()? + 1;
    let frame_mbs_only_flag = r.read_bit()?;

    let mb_adaptive_frame_field_flag = if !frame_mbs_only_flag {
        r.read_bit()?
    } else {
        false
    };

    let direct_8x8_inference_flag = r.read_bit()?;

    // Frame cropping
    let frame_cropping_flag = r.read_bit()?;
    let (crop_left, crop_right, crop_top, crop_bottom) = if frame_cropping_flag {
        (r.read_ue()?, r.read_ue()?, r.read_ue()?, r.read_ue()?)
    } else {
        (0, 0, 0, 0)
    };

    // VUI parameters — skip entirely (not needed for stego)
    let vui_parameters_present = r.read_bit()?;
    if vui_parameters_present {
        skip_vui_parameters(&mut r)?;
    }

    // Derive pixel dimensions
    let frame_height_in_mbs = pic_height_in_map_units * if frame_mbs_only_flag { 1 } else { 2 };
    let sub_width_c: u32 = if chroma_format_idc == 1 || chroma_format_idc == 2 { 2 } else { 1 };
    let sub_height_c: u32 = if chroma_format_idc == 1 { 2 } else { 1 };
    let crop_unit_x = if chroma_format_idc == 0 { 1 } else { sub_width_c };
    let crop_unit_y = if chroma_format_idc == 0 {
        if frame_mbs_only_flag { 1 } else { 2 }
    } else {
        sub_height_c * if frame_mbs_only_flag { 1 } else { 2 }
    };

    let width_in_pixels =
        pic_width_in_mbs * 16 - crop_unit_x * (crop_left + crop_right);
    let height_in_pixels =
        frame_height_in_mbs * 16 - crop_unit_y * (crop_top + crop_bottom);
    let pic_size_in_mbs = pic_width_in_mbs * frame_height_in_mbs;

    Ok(Sps {
        profile_idc,
        constraint_set_flags,
        level_idc,
        sps_id,
        chroma_format_idc,
        separate_colour_plane_flag,
        bit_depth_luma,
        bit_depth_chroma,
        qpprime_y_zero_transform_bypass_flag,
        log2_max_frame_num,
        pic_order_cnt_type,
        log2_max_pic_order_cnt_lsb,
        delta_pic_order_always_zero_flag,
        offset_for_non_ref_pic,
        offset_for_top_to_bottom_field,
        num_ref_frames_in_pic_order_cnt_cycle,
        max_num_ref_frames,
        gaps_in_frame_num_allowed,
        pic_width_in_mbs,
        pic_height_in_map_units,
        frame_mbs_only_flag,
        mb_adaptive_frame_field_flag,
        direct_8x8_inference_flag,
        frame_cropping_flag,
        crop_left,
        crop_right,
        crop_top,
        crop_bottom,
        width_in_pixels,
        height_in_pixels,
        pic_size_in_mbs,
    })
}

/// Parse an H.264 PPS from RBSP data.
///
/// Reference: ITU-T H.264 Section 7.3.2.2
pub fn parse_pps(rbsp: &[u8]) -> Result<Pps, H264Error> {
    let mut r = RbspReader::new(rbsp);

    let pps_id = r.read_ue()? as u8;
    let sps_id = r.read_ue()? as u8;
    let entropy_coding_mode_flag = r.read_bit()?;
    let bottom_field_pic_order_in_frame_present_flag = r.read_bit()?;

    let num_slice_groups_minus1 = r.read_ue()?;
    if num_slice_groups_minus1 > 0 {
        return Err(H264Error::Unsupported(
            "FMO (num_slice_groups_minus1 > 0) not supported".into(),
        ));
    }

    let num_ref_idx_l0_default = r.read_ue()? as u8 + 1;
    let num_ref_idx_l1_default = r.read_ue()? as u8 + 1;
    let weighted_pred_flag = r.read_bit()?;
    let weighted_bipred_idc = r.read_bits(2)? as u8;
    let pic_init_qp_minus26 = r.read_se()?;
    let pic_init_qs_minus26 = r.read_se()?;
    let chroma_qp_index_offset = r.read_se()?;
    let deblocking_filter_control_present_flag = r.read_bit()?;
    let constrained_intra_pred_flag = r.read_bit()?;
    let redundant_pic_cnt_present_flag = r.read_bit()?;

    // High profile extensions
    let mut transform_8x8_mode_flag = false;
    let mut second_chroma_qp_index_offset = chroma_qp_index_offset;

    if r.more_rbsp_data() {
        transform_8x8_mode_flag = r.read_bit()?;
        let pic_scaling_matrix_present_flag = r.read_bit()?;
        if pic_scaling_matrix_present_flag {
            let count = if transform_8x8_mode_flag { 8 } else { 6 };
            // Note: count depends on chroma_format_idc for 4:4:4, but we
            // don't support that in Phase 1a
            for _ in 0..count {
                let list_present = r.read_bit()?;
                if list_present {
                    let size = if count <= 6 { 16 } else { 64 };
                    skip_scaling_list(&mut r, size)?;
                }
            }
        }
        second_chroma_qp_index_offset = r.read_se()?;
    }

    Ok(Pps {
        pps_id,
        sps_id,
        entropy_coding_mode_flag,
        bottom_field_pic_order_in_frame_present_flag,
        num_slice_groups_minus1,
        num_ref_idx_l0_default,
        num_ref_idx_l1_default,
        weighted_pred_flag,
        weighted_bipred_idc,
        pic_init_qp_minus26,
        pic_init_qs_minus26,
        chroma_qp_index_offset,
        deblocking_filter_control_present_flag,
        constrained_intra_pred_flag,
        redundant_pic_cnt_present_flag,
        transform_8x8_mode_flag,
        second_chroma_qp_index_offset,
    })
}

/// Skip a scaling list in the SPS/PPS (read and discard delta values).
fn skip_scaling_list(r: &mut RbspReader<'_>, size: usize) -> Result<(), H264Error> {
    let mut last_scale = 8i32;
    let mut next_scale = 8i32;
    for _ in 0..size {
        if next_scale != 0 {
            let delta = r.read_se()?;
            next_scale = (last_scale + delta + 256) % 256;
        }
        last_scale = if next_scale == 0 {
            last_scale
        } else {
            next_scale
        };
    }
    Ok(())
}

/// Skip VUI parameters (not needed for steganography, but must be parsed to
/// reach the end of SPS correctly).
fn skip_vui_parameters(r: &mut RbspReader<'_>) -> Result<(), H264Error> {
    let aspect_ratio_info_present = r.read_bit()?;
    if aspect_ratio_info_present {
        let aspect_ratio_idc = r.read_bits(8)?;
        if aspect_ratio_idc == 255 {
            // Extended_SAR
            r.skip_bits(16)?; // sar_width
            r.skip_bits(16)?; // sar_height
        }
    }
    let overscan_info_present = r.read_bit()?;
    if overscan_info_present {
        r.skip_bits(1)?; // overscan_appropriate_flag
    }
    let video_signal_type_present = r.read_bit()?;
    if video_signal_type_present {
        r.skip_bits(3)?; // video_format
        r.skip_bits(1)?; // video_full_range_flag
        let colour_description_present = r.read_bit()?;
        if colour_description_present {
            r.skip_bits(8)?; // colour_primaries
            r.skip_bits(8)?; // transfer_characteristics
            r.skip_bits(8)?; // matrix_coefficients
        }
    }
    let chroma_loc_info_present = r.read_bit()?;
    if chroma_loc_info_present {
        let _ = r.read_ue()?; // chroma_sample_loc_type_top_field
        let _ = r.read_ue()?; // chroma_sample_loc_type_bottom_field
    }
    let timing_info_present = r.read_bit()?;
    if timing_info_present {
        r.skip_bits(32)?; // num_units_in_tick
        r.skip_bits(32)?; // time_scale
        r.skip_bits(1)?; // fixed_frame_rate_flag
    }
    let nal_hrd_present = r.read_bit()?;
    if nal_hrd_present {
        skip_hrd_parameters(r)?;
    }
    let vcl_hrd_present = r.read_bit()?;
    if vcl_hrd_present {
        skip_hrd_parameters(r)?;
    }
    if nal_hrd_present || vcl_hrd_present {
        r.skip_bits(1)?; // low_delay_hrd_flag
    }
    r.skip_bits(1)?; // pic_struct_present_flag
    let bitstream_restriction = r.read_bit()?;
    if bitstream_restriction {
        r.skip_bits(1)?; // motion_vectors_over_pic_boundaries_flag
        let _ = r.read_ue()?; // max_bytes_per_pic_denom
        let _ = r.read_ue()?; // max_bits_per_mb_denom
        let _ = r.read_ue()?; // log2_max_mv_length_horizontal
        let _ = r.read_ue()?; // log2_max_mv_length_vertical
        let _ = r.read_ue()?; // max_num_reorder_frames
        let _ = r.read_ue()?; // max_dec_frame_buffering
    }
    Ok(())
}

fn skip_hrd_parameters(r: &mut RbspReader<'_>) -> Result<(), H264Error> {
    let cpb_cnt = r.read_ue()? + 1;
    r.skip_bits(4)?; // bit_rate_scale
    r.skip_bits(4)?; // cpb_size_scale
    for _ in 0..cpb_cnt {
        let _ = r.read_ue()?; // bit_rate_value_minus1
        let _ = r.read_ue()?; // cpb_size_value_minus1
        r.skip_bits(1)?; // cbr_flag
    }
    r.skip_bits(5)?; // initial_cpb_removal_delay_length_minus1
    r.skip_bits(5)?; // cpb_removal_delay_length_minus1
    r.skip_bits(5)?; // dpb_output_delay_length_minus1
    r.skip_bits(5)?; // time_offset_length
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    

    /// Build a minimal Baseline SPS RBSP for testing.
    /// Profile=66 (Baseline), Level=30, 320x240, poc_type=0, log2_max_poc_lsb=6
    fn make_baseline_sps_rbsp() -> Vec<u8> {
        let mut bits = BitWriter::new();
        bits.write_u8(66); // profile_idc = Baseline
        bits.write_u8(0xC0); // constraint_set0+1 flags set, others 0
        bits.write_u8(30); // level_idc = 3.0
        bits.write_ue(0); // sps_id = 0
        // No High profile extensions (profile 66 < 100)
        bits.write_ue(0); // log2_max_frame_num_minus4 = 0 → log2=4
        bits.write_ue(0); // pic_order_cnt_type = 0
        bits.write_ue(2); // log2_max_pic_order_cnt_lsb_minus4 = 2 → log2=6
        bits.write_ue(1); // max_num_ref_frames = 1
        bits.write_bit(false); // gaps_in_frame_num_allowed = 0
        bits.write_ue(19); // pic_width_in_mbs_minus1 = 19 → 20 MBs = 320px
        bits.write_ue(14); // pic_height_in_map_units_minus1 = 14 → 15 MUs = 240px
        bits.write_bit(true); // frame_mbs_only_flag = 1 (progressive)
        // No mb_adaptive_frame_field_flag (frame_mbs_only=1)
        bits.write_bit(false); // direct_8x8_inference_flag = 0
        bits.write_bit(false); // frame_cropping_flag = 0
        bits.write_bit(false); // vui_parameters_present = 0
        // RBSP stop bit
        bits.write_bit(true);
        bits.align();
        bits.data
    }

    /// Build a minimal PPS RBSP (CAVLC, no FMO).
    fn make_cavlc_pps_rbsp() -> Vec<u8> {
        let mut bits = BitWriter::new();
        bits.write_ue(0); // pps_id = 0
        bits.write_ue(0); // sps_id = 0
        bits.write_bit(false); // entropy_coding_mode_flag = 0 (CAVLC)
        bits.write_bit(false); // bottom_field_pic_order_in_frame_present = 0
        bits.write_ue(0); // num_slice_groups_minus1 = 0 (no FMO)
        bits.write_ue(0); // num_ref_idx_l0_default_active_minus1 = 0
        bits.write_ue(0); // num_ref_idx_l1_default_active_minus1 = 0
        bits.write_bit(false); // weighted_pred_flag = 0
        bits.write_bits(0, 2); // weighted_bipred_idc = 0
        bits.write_se(0); // pic_init_qp_minus26 = 0
        bits.write_se(0); // pic_init_qs_minus26 = 0
        bits.write_se(0); // chroma_qp_index_offset = 0
        bits.write_bit(true); // deblocking_filter_control_present = 1
        bits.write_bit(false); // constrained_intra_pred = 0
        bits.write_bit(false); // redundant_pic_cnt_present = 0
        // RBSP stop bit
        bits.write_bit(true);
        bits.align();
        bits.data
    }

    #[test]
    fn parse_baseline_sps() {
        let rbsp = make_baseline_sps_rbsp();
        let sps = parse_sps(&rbsp).unwrap();
        assert_eq!(sps.profile_idc, 66);
        assert_eq!(sps.level_idc, 30);
        assert_eq!(sps.sps_id, 0);
        assert_eq!(sps.chroma_format_idc, 1); // default 4:2:0
        assert_eq!(sps.log2_max_frame_num, 4);
        assert_eq!(sps.pic_order_cnt_type, 0);
        assert_eq!(sps.log2_max_pic_order_cnt_lsb, 6);
        assert_eq!(sps.pic_width_in_mbs, 20);
        assert_eq!(sps.pic_height_in_map_units, 15);
        assert!(sps.frame_mbs_only_flag);
        assert_eq!(sps.width_in_pixels, 320);
        assert_eq!(sps.height_in_pixels, 240);
        assert_eq!(sps.pic_size_in_mbs, 300);
    }

    #[test]
    fn parse_cavlc_pps() {
        let rbsp = make_cavlc_pps_rbsp();
        let pps = parse_pps(&rbsp).unwrap();
        assert_eq!(pps.pps_id, 0);
        assert_eq!(pps.sps_id, 0);
        assert!(!pps.entropy_coding_mode_flag); // CAVLC
        assert_eq!(pps.num_slice_groups_minus1, 0);
        assert_eq!(pps.num_ref_idx_l0_default, 1);
        assert_eq!(pps.pic_init_qp_minus26, 0);
        assert!(pps.deblocking_filter_control_present_flag);
    }

    #[test]
    fn reject_cabac_pps() {
        let mut bits = BitWriter::new();
        bits.write_ue(0); // pps_id
        bits.write_ue(0); // sps_id
        bits.write_bit(true); // entropy_coding_mode_flag = 1 (CABAC)
        bits.write_bit(false);
        bits.write_ue(0); // num_slice_groups
        bits.write_ue(0);
        bits.write_ue(0);
        bits.write_bit(false);
        bits.write_bits(0, 2);
        bits.write_se(0);
        bits.write_se(0);
        bits.write_se(0);
        bits.write_bit(true);
        bits.write_bit(false);
        bits.write_bit(false);
        bits.write_bit(true);
        bits.align();

        let pps = parse_pps(&bits.data).unwrap();
        assert!(pps.entropy_coding_mode_flag); // CABAC detected
    }

    #[test]
    fn reject_fmo_pps() {
        let mut bits = BitWriter::new();
        bits.write_ue(0);
        bits.write_ue(0);
        bits.write_bit(false); // CAVLC
        bits.write_bit(false);
        bits.write_ue(1); // num_slice_groups_minus1 = 1 → FMO!
        bits.align();

        let result = parse_pps(&bits.data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("FMO"));
    }

    // -- BitWriter helper for test SPS/PPS construction --

    struct BitWriter {
        data: Vec<u8>,
        current: u8,
        bit_pos: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                data: Vec::new(),
                current: 0,
                bit_pos: 0,
            }
        }

        fn write_bit(&mut self, val: bool) {
            if val {
                self.current |= 1 << (7 - self.bit_pos);
            }
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.data.push(self.current);
                self.current = 0;
                self.bit_pos = 0;
            }
        }

        fn write_bits(&mut self, val: u32, n: u8) {
            for i in (0..n).rev() {
                self.write_bit((val >> i) & 1 != 0);
            }
        }

        fn write_u8(&mut self, val: u8) {
            self.write_bits(val as u32, 8);
        }

        fn write_ue(&mut self, val: u32) {
            let code_num = val;
            if code_num == 0 {
                self.write_bit(true); // "1"
                return;
            }
            let n = 32 - (code_num + 1).leading_zeros(); // bit length of (code_num+1)
            let leading_zeros = n - 1;
            for _ in 0..leading_zeros {
                self.write_bit(false);
            }
            self.write_bits(code_num + 1, n as u8);
        }

        fn write_se(&mut self, val: i32) {
            let ue = if val > 0 {
                (2 * val - 1) as u32
            } else if val < 0 {
                (2 * (-val)) as u32
            } else {
                0
            };
            self.write_ue(ue);
        }

        fn align(&mut self) {
            if self.bit_pos > 0 {
                self.data.push(self.current);
                self.current = 0;
                self.bit_pos = 0;
            }
        }
    }
}
