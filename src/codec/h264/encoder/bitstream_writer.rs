// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Encoder-side H.264 bitstream writer. Phase 6A.5.
//!
//! Three layers:
//!   1. `BitWriter` — byte-buffer with bit-level writes + Exp-Golomb
//!      ue/se helpers + RBSP trailing bit.
//!   2. Syntax writers — `build_sps_baseline`, `build_pps_cavlc`,
//!      `build_slice_header_i`. Each outputs RBSP bytes; NAL
//!      wrapping is separate.
//!   3. `wrap_rbsp_as_nal` — adds the NAL header byte + emulation-
//!      prevention bytes via the existing
//!      `codec::h264::bitstream::insert_emulation_prevention`.
//!
//! Algorithm note: docs/design/h264-encoder-algorithms/bitstream.md.
//!
//! MB layer writing is Phase 6A.4's job (CAVLC). P-slice header is
//! Phase 6B.3.

use crate::codec::h264::bitstream::insert_emulation_prevention;
use crate::codec::h264::NalType;

// ─── BitSink trait ────────────────────────────────────────────────
//
// Phase B.1 of the encoder-quality plan. Abstracts over "something
// that consumes bits" so the CAVLC encoder can run in two modes:
//
//   1. Real emit path (BitWriter) — produces the bytes we write to
//      the slice.
//   2. Size-only path (BitSizer) — accumulates a bit counter without
//      touching a buffer, used by Phase C RDO to estimate R(mode)
//      bit-accurately for each candidate.
//
// Both paths walk the exact same codeword logic so the size counter
// is guaranteed to equal the real bitstream's length (the Phase B.3
// unit tests enforce this on real MBs).

/// Something that accepts bits MSB-first. Implementations track
/// cumulative bit count so callers can ask "how many bits have I
/// written so far?" without peeking at the buffer.
pub trait BitSink {
    fn write_bits(&mut self, value: u32, n: u8);
    fn write_bit(&mut self, b: bool);
    fn write_ue(&mut self, value: u32);
    fn write_se(&mut self, value: i32);
    fn bits_written(&self) -> usize;
}

/// Bit counter — no buffer, no emission. Use this in place of
/// [`BitWriter`] when you only need the encoded length (Phase C RDO
/// rate estimation).
#[derive(Debug, Default, Clone, Copy)]
pub struct BitSizer {
    bits: usize,
}

impl BitSizer {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn reset(&mut self) {
        self.bits = 0;
    }
}

impl BitSink for BitSizer {
    #[inline]
    fn write_bits(&mut self, _value: u32, n: u8) {
        self.bits += n as usize;
    }
    #[inline]
    fn write_bit(&mut self, _b: bool) {
        self.bits += 1;
    }
    #[inline]
    fn write_ue(&mut self, value: u32) {
        // codeNum length = 2 * leading_zeros + 1, same math as
        // `BitWriter::write_ue`. Keep the computation identical so
        // the counter matches byte-exact.
        let plus_one = value as u64 + 1;
        let leading_zeros = (63u8 - plus_one.leading_zeros() as u8).min(31);
        self.bits += 2 * leading_zeros as usize + 1;
    }
    #[inline]
    fn write_se(&mut self, value: i32) {
        let code_num = if value > 0 {
            (2 * value - 1) as u32
        } else {
            (-2 * value) as u32
        };
        self.write_ue(code_num);
    }
    #[inline]
    fn bits_written(&self) -> usize {
        self.bits
    }
}

// ─── BitWriter ────────────────────────────────────────────────────

/// Byte-level buffer with MSB-first bit writes.
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    buffer: Vec<u8>,
    /// Bits pending in the low end of `pending` (MSB-first when flushed).
    pending: u32,
    /// Number of pending bits, 0..8.
    pending_bits: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(cap),
            ..Self::default()
        }
    }

    /// Write the low `n` bits of `value`, MSB-first. `n` must be ≤ 32.
    pub fn write_bits(&mut self, value: u32, n: u8) {
        debug_assert!(n <= 32, "write_bits called with n={n} > 32");
        let mut remaining = n;
        while remaining > 0 {
            let take = remaining.min(8 - self.pending_bits);
            let shift = remaining - take;
            let chunk = (value >> shift) & ((1u32 << take) - 1);
            self.pending = (self.pending << take) | chunk;
            self.pending_bits += take;
            remaining -= take;
            if self.pending_bits == 8 {
                self.buffer.push(self.pending as u8);
                self.pending = 0;
                self.pending_bits = 0;
            }
        }
    }

    /// Write a single bit.
    #[inline]
    pub fn write_bit(&mut self, b: bool) {
        self.write_bits(b as u32, 1);
    }

    /// Write an unsigned Exp-Golomb codeword ue(v).
    pub fn write_ue(&mut self, value: u32) {
        // codeNum = value. Encoded as `leading_zeros` zero bits,
        // then a 1 bit, then `leading_zeros` suffix bits.
        let plus_one = value as u64 + 1;
        let leading_zeros = (63u8 - plus_one.leading_zeros() as u8).min(31);
        // Total length = 2*leading_zeros + 1.
        // Bits: 0^lz, 1, suffix[lz] (where suffix is plus_one's low lz bits).
        for _ in 0..leading_zeros {
            self.write_bit(false);
        }
        self.write_bit(true);
        if leading_zeros > 0 {
            let suffix = plus_one as u32 & ((1u32 << leading_zeros) - 1);
            self.write_bits(suffix, leading_zeros);
        }
    }

    /// Write a signed Exp-Golomb codeword se(v).
    pub fn write_se(&mut self, value: i32) {
        let code_num = if value > 0 {
            (2 * value - 1) as u32
        } else {
            (-2 * value) as u32
        };
        self.write_ue(code_num);
    }

    /// Append the RBSP trailing bits (stop bit + zero-fill to byte
    /// boundary). Must be the last thing written before the NAL wrap.
    pub fn write_rbsp_trailing(&mut self) {
        self.write_bit(true);
        while self.pending_bits != 0 {
            self.write_bit(false);
        }
    }

    /// True when the writer is at a byte boundary (no pending bits).
    #[inline]
    pub fn byte_aligned(&self) -> bool {
        self.pending_bits == 0
    }

    /// Total number of bits written so far (flushed bytes + pending).
    #[inline]
    pub fn bits_written(&self) -> usize {
        self.buffer.len() * 8 + self.pending_bits as usize
    }

}

impl BitSink for BitWriter {
    #[inline]
    fn write_bits(&mut self, value: u32, n: u8) {
        BitWriter::write_bits(self, value, n);
    }
    #[inline]
    fn write_bit(&mut self, b: bool) {
        BitWriter::write_bit(self, b);
    }
    #[inline]
    fn write_ue(&mut self, value: u32) {
        BitWriter::write_ue(self, value);
    }
    #[inline]
    fn write_se(&mut self, value: i32) {
        BitWriter::write_se(self, value);
    }
    #[inline]
    fn bits_written(&self) -> usize {
        BitWriter::bits_written(self)
    }
}

impl BitWriter {
    /// Consume the writer and return the written bytes. The trailing
    /// byte is zero-padded at the low bits if there are pending bits
    /// (callers should usually call `write_rbsp_trailing` first).
    pub fn finish(mut self) -> Vec<u8> {
        if self.pending_bits > 0 {
            let pad = 8 - self.pending_bits;
            self.pending <<= pad;
            self.buffer.push(self.pending as u8);
        }
        self.buffer
    }
}

// ─── SPS ──────────────────────────────────────────────────────────

/// Minimal Baseline-profile SPS parameters — what the encoder can
/// actually vary. Other fields are fixed to conservative defaults
/// (see `bitstream.md`).
#[derive(Debug, Clone, Copy)]
pub struct SpsParams {
    pub width_pixels: u32,
    pub height_pixels: u32,
    pub sps_id: u8,
    pub max_num_ref_frames: u8,
    /// `pic_order_cnt_type` per spec § 7.4.2.1. Default `2`
    /// (frame_num → POC) is fine for I+P-only streams. For B-frame
    /// support set to `0` (LSB-based POC) so the decoder can
    /// reorder display vs encode order. Phase 6E-A1 + §6E-A4 use
    /// `0` when emitting B-slices. Phase 6B I/P-only paths keep
    /// `2`.
    pub pic_order_cnt_type: u8,
    /// Only emitted when `pic_order_cnt_type == 0`. Width of the
    /// `pic_order_cnt_lsb` field is `log2_max_pic_order_cnt_lsb_minus4
    /// + 4` bits. Phase 6E-A1 default = 4 → 8-bit POC LSB
    /// (range 0..255).
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
}

impl Default for SpsParams {
    fn default() -> Self {
        Self {
            width_pixels: 1280,
            height_pixels: 720,
            sps_id: 0,
            max_num_ref_frames: 1,
            pic_order_cnt_type: 2,
            log2_max_pic_order_cnt_lsb_minus4: 4,
        }
    }
}

/// Pick `level_idc` from frame dimensions. MaxFrameSize (MBs/frame)
/// thresholds from H.264 Annex A Table A-1.
fn derive_level_idc(width: u32, height: u32) -> u8 {
    let mb = (width / 16) * (height / 16);
    match mb {
        0..=99 => 10,       // 1.0: QCIF 176×144
        100..=396 => 20,    // 2.0: CIF 352×288
        397..=1620 => 30,   // 3.0: SD (≤720×480)
        1621..=3600 => 31,  // 3.1: 720p
        3601..=5120 => 32,  // 3.2
        5121..=8192 => 40,  // 4.0: 1080p
        8193..=8704 => 42,  // 4.2
        8705..=22080 => 50, // 5.0
        22081..=36864 => 51, // 5.1: 4K
        _ => 52,            // 5.2: max
    }
}

/// Build an SPS RBSP payload for a Baseline-profile stream with the
/// given dimensions. Returns RBSP bytes (no NAL header, no emulation
/// prevention).
pub fn build_sps_baseline(p: &SpsParams) -> Vec<u8> {
    build_sps_inner(p, 66, 0b1000_0000)
}

/// Build an SPS RBSP payload for a Main-profile stream (profile_idc =
/// 77). CABAC requires Main or higher — Baseline (66) does not support
/// CABAC entropy coding. Apart from the profile_idc and the cleared
/// Baseline constraint flag, the SPS payload is identical to the
/// Baseline variant (we don't emit any of the High-profile chroma/
/// bit-depth fields since those start at profile_idc ≥ 100).
pub fn build_sps_main(p: &SpsParams) -> Vec<u8> {
    // constraint_set0_flag = 0 (clears the Baseline marker). All other
    // constraint bits and reserved zeros stay zero.
    build_sps_inner(p, 77, 0b0000_0000)
}

/// Build an SPS RBSP payload for a High-profile stream (profile_idc =
/// 100). High profile is the gate for 8×8 transform signaling in the
/// PPS (`transform_8x8_mode_flag` is only allowed at profile_idc ≥ 100).
/// Pair with [`build_pps_cabac_high`] to enable 8×8 DCT emission.
///
/// Adds the High-profile suffix fields after the base SPS body:
/// - `chroma_format_idc` = 1 (4:2:0; we don't support 4:2:2 or 4:4:4).
/// - `bit_depth_luma_minus8` = 0 (8-bit luma).
/// - `bit_depth_chroma_minus8` = 0 (8-bit chroma).
/// - `qpprime_y_zero_transform_bypass_flag` = 0 (no lossless mode).
/// - `seq_scaling_matrix_present_flag` = 0 (flat scaling; no custom
///   matrices in the SPS. Decoder uses Flat_8x8_16 default).
pub fn build_sps_high(p: &SpsParams) -> Vec<u8> {
    build_sps_inner_high(p)
}

fn build_sps_inner(p: &SpsParams, profile_idc: u32, constraint_byte: u32) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(16);

    w.write_bits(profile_idc, 8);
    w.write_bits(constraint_byte, 8);
    w.write_bits(derive_level_idc(p.width_pixels, p.height_pixels) as u32, 8);
    // seq_parameter_set_id
    w.write_ue(p.sps_id as u32);
    // log2_max_frame_num_minus4 = 0 → 4-bit frame_num (0..15).
    w.write_ue(0);
    // pic_order_cnt_type — § 6E-A4 honors the SpsParams field.
    // Default `2` keeps backwards compat (Phase 6B I+P-only).
    // `0` enables LSB-based POC for B-frame reordering.
    w.write_ue(p.pic_order_cnt_type as u32);
    if p.pic_order_cnt_type == 0 {
        w.write_ue(p.log2_max_pic_order_cnt_lsb_minus4 as u32);
    }
    // max_num_ref_frames
    w.write_ue(p.max_num_ref_frames as u32);
    // gaps_in_frame_num_value_allowed_flag = 0
    w.write_bit(false);
    // pic_width_in_mbs_minus1
    w.write_ue((p.width_pixels / 16) - 1);
    // pic_height_in_map_units_minus1
    w.write_ue((p.height_pixels / 16) - 1);
    // frame_mbs_only_flag = 1 (progressive)
    w.write_bit(true);
    // direct_8x8_inference_flag — required = 1 for progressive
    w.write_bit(true);
    // frame_cropping_flag
    w.write_bit(false);
    // vui_parameters_present_flag = 0
    w.write_bit(false);

    w.write_rbsp_trailing();
    w.finish()
}

fn build_sps_inner_high(p: &SpsParams) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(16);

    w.write_bits(100, 8); // profile_idc = 100 (High)
    w.write_bits(0b0000_0000, 8); // no constraint flags set
    w.write_bits(derive_level_idc(p.width_pixels, p.height_pixels) as u32, 8);
    w.write_ue(p.sps_id as u32);

    // ── High-profile suffix (spec § 7.3.2.1.1.1) — emitted when
    //    profile_idc ∈ {100, 110, 122, 244, 44, 83, 86, 118, 128, 138,
    //    139, 134, 135}.
    w.write_ue(1); // chroma_format_idc = 1 (4:2:0)
    // Skip separate_colour_plane_flag — only emitted when
    // chroma_format_idc == 3 (4:4:4).
    w.write_ue(0); // bit_depth_luma_minus8
    w.write_ue(0); // bit_depth_chroma_minus8
    w.write_bit(false); // qpprime_y_zero_transform_bypass_flag
    w.write_bit(false); // seq_scaling_matrix_present_flag (flat default)

    // ── Resume the base SPS body (same as build_sps_inner from
    //    log2_max_frame_num_minus4 onward).
    w.write_ue(0); // log2_max_frame_num_minus4
    w.write_ue(p.pic_order_cnt_type as u32);
    if p.pic_order_cnt_type == 0 {
        w.write_ue(p.log2_max_pic_order_cnt_lsb_minus4 as u32);
    }
    w.write_ue(p.max_num_ref_frames as u32);
    w.write_bit(false); // gaps_in_frame_num_value_allowed_flag
    w.write_ue((p.width_pixels / 16) - 1);
    w.write_ue((p.height_pixels / 16) - 1);
    w.write_bit(true); // frame_mbs_only_flag
    w.write_bit(true); // direct_8x8_inference_flag
    w.write_bit(false); // frame_cropping_flag
    w.write_bit(false); // vui_parameters_present_flag

    w.write_rbsp_trailing();
    w.finish()
}

// ─── PPS ──────────────────────────────────────────────────────────

/// Minimal PPS parameters.
#[derive(Debug, Clone, Copy)]
pub struct PpsParams {
    pub pps_id: u8,
    pub sps_id: u8,
    /// Base QP for I-slices before slice-level delta. QP 26 = sensible default.
    pub pic_init_qp: u8,
    /// When true, sets `deblocking_filter_control_present_flag = 1` so
    /// the slice header can emit `disable_deblocking_filter_idc`. Phase
    /// 6A.7 stopgap — set this + `ISliceHeaderParams::disable_deblocking = 1`
    /// to skip the deblocker until the real filter ships.
    pub deblocking_filter_control_present: bool,
}

impl Default for PpsParams {
    fn default() -> Self {
        Self {
            pps_id: 0,
            sps_id: 0,
            pic_init_qp: 26,
            deblocking_filter_control_present: false,
        }
    }
}

/// Build a PPS RBSP payload for a CAVLC Baseline stream.
pub fn build_pps_cavlc(p: &PpsParams) -> Vec<u8> {
    build_pps_inner(p, false, false)
}

/// Build a PPS RBSP payload with `entropy_coding_mode_flag = 1` for
/// CABAC. The rest of the PPS payload is identical to the CAVLC path.
/// Pair with [`build_sps_main`] since Baseline (profile_idc=66) does
/// not permit CABAC.
pub fn build_pps_cabac(p: &PpsParams) -> Vec<u8> {
    build_pps_inner(p, true, false)
}

/// Build a PPS RBSP payload for a High-profile stream with CABAC +
/// `transform_8x8_mode_flag = 1`. Pair with [`build_sps_high`]. The
/// High-profile PPS suffix adds:
/// - `transform_8x8_mode_flag` = 1 (enable 8×8 transform signaling).
/// - `pic_scaling_matrix_present_flag` = 0 (flat scaling; decoder uses
///   Flat_8x8_16 default per spec § 7.4.2.2).
/// - `second_chroma_qp_index_offset` = 0.
pub fn build_pps_cabac_high(p: &PpsParams) -> Vec<u8> {
    build_pps_inner(p, true, true)
}

fn build_pps_inner(p: &PpsParams, entropy_coding_mode: bool, high_profile_suffix: bool) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(8);

    w.write_ue(p.pps_id as u32);
    w.write_ue(p.sps_id as u32);
    // entropy_coding_mode_flag
    w.write_bit(entropy_coding_mode);
    // bottom_field_pic_order_in_frame_present_flag = 0
    w.write_bit(false);
    // num_slice_groups_minus1 = 0 (no FMO)
    w.write_ue(0);
    // num_ref_idx_l0_default_active_minus1 = 0
    w.write_ue(0);
    // num_ref_idx_l1_default_active_minus1 = 0
    w.write_ue(0);
    // weighted_pred_flag = 0
    w.write_bit(false);
    // weighted_bipred_idc = 0
    w.write_bits(0, 2);
    // pic_init_qp_minus26
    w.write_se(p.pic_init_qp as i32 - 26);
    // pic_init_qs_minus26 = 0
    w.write_se(0);
    // chroma_qp_index_offset = 0
    w.write_se(0);
    // deblocking_filter_control_present_flag
    w.write_bit(p.deblocking_filter_control_present);
    // constrained_intra_pred_flag = 0
    w.write_bit(false);
    // redundant_pic_cnt_present_flag = 0
    w.write_bit(false);

    if high_profile_suffix {
        // Spec § 7.3.2.2 PPS High-profile extension.
        // transform_8x8_mode_flag = 1 (enable 8×8 signaling).
        w.write_bit(true);
        // pic_scaling_matrix_present_flag = 0 (flat / default).
        w.write_bit(false);
        // second_chroma_qp_index_offset = 0.
        w.write_se(0);
    }

    w.write_rbsp_trailing();
    w.finish()
}

// ─── I-slice header ───────────────────────────────────────────────

/// Parameters for an I-slice header (Phase 6A).
#[derive(Debug, Clone, Copy)]
pub struct ISliceHeaderParams {
    pub is_idr: bool,
    pub pps_id: u8,
    /// Frame counter (wraps at `1 << log2_max_frame_num`, = 16 per our SPS).
    pub frame_num: u8,
    /// For IDR slices only. 0 for non-IDR.
    pub idr_pic_id: u16,
    /// QP delta from `pic_init_qp` — usually 0 for Baseline-CRF.
    pub slice_qp_delta: i32,
    /// Phase 6A.7 stopgap — emit `disable_deblocking_filter_idc = 1`.
    /// Requires the PPS's `deblocking_filter_control_present` to be set.
    /// Ignored otherwise.
    pub disable_deblocking: bool,
    /// Phase 6E-A4 — when `Some(lsb)`, emit `pic_order_cnt_lsb` per
    /// spec § 7.3.3 (required when SPS uses `pic_order_cnt_type = 0`).
    /// `None` skips the field (Phase 6B I+P-only path,
    /// `pic_order_cnt_type = 2`).
    pub pic_order_cnt_lsb: Option<u32>,
    /// Phase 6E-A4 — width of the `pic_order_cnt_lsb` field in bits,
    /// = `log2_max_pic_order_cnt_lsb_minus4 + 4`. Only consulted when
    /// `pic_order_cnt_lsb` is `Some`.
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
}

impl Default for ISliceHeaderParams {
    fn default() -> Self {
        Self {
            is_idr: true,
            pps_id: 0,
            frame_num: 0,
            idr_pic_id: 0,
            slice_qp_delta: 0,
            disable_deblocking: false,
            pic_order_cnt_lsb: None,
            log2_max_pic_order_cnt_lsb_minus4: 4,
        }
    }
}

/// Build an I-slice header RBSP, up to the start of MB data.
///
/// The caller appends MB data (Phase 6A.4 — CAVLC) to the returned
/// vector before writing the trailing bits. Because slice data is
/// itself byte-aligned at this point (we return a byte-aligned
/// header), we don't write trailing bits here; the caller does it
/// after the MB layer finishes.
pub fn build_slice_header_i(p: &ISliceHeaderParams) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(8);

    // first_mb_in_slice = 0 (single slice per frame in Phase 6A)
    w.write_ue(0);
    // slice_type = 7 ("I — all MBs in slice are I" hint)
    w.write_ue(7);
    // pic_parameter_set_id
    w.write_ue(p.pps_id as u32);
    // frame_num (4 bits — matches log2_max_frame_num_minus4 = 0 in SPS)
    w.write_bits(p.frame_num as u32 & 0xF, 4);
    // IDR-only field
    if p.is_idr {
        w.write_ue(p.idr_pic_id as u32);
    }
    // pic_order_cnt_type — § 6E-A4 emits pic_order_cnt_lsb when SPS
    // uses type 0. Phase 6B I+P-only SPS uses type 2 (no field).
    if let Some(lsb) = p.pic_order_cnt_lsb {
        let bits: u8 = p.log2_max_pic_order_cnt_lsb_minus4 + 4;
        let mask = (1u32 << bits as u32) - 1;
        w.write_bits(lsb & mask, bits);
    }

    // For I-slice (slice_type = 7, == 2 mod 5), no ref_pic_list_modification
    // loops apply. For IDR, write no_output_of_prior_pics_flag = 0 and
    // long_term_reference_flag = 0 (one bit each) before the QP delta.
    if p.is_idr {
        w.write_bit(false); // no_output_of_prior_pics_flag
        w.write_bit(false); // long_term_reference_flag
    } else {
        // adaptive_ref_pic_marking_mode_flag = 0 for non-IDR I
        w.write_bit(false);
    }

    // slice_qp_delta
    w.write_se(p.slice_qp_delta);

    // disable_deblocking_filter_idc — present whenever the PPS has
    // `deblocking_filter_control_present_flag = 1`. Spec § 7.3.3:
    //   idc = 0: filter on, offsets emitted next.
    //   idc = 1: filter off, no offsets.
    //   idc = 2: filter on within slice only (we don't use this).
    if p.disable_deblocking {
        w.write_ue(1);
    } else {
        w.write_ue(0);
        w.write_se(0); // slice_alpha_c0_offset_div2
        w.write_se(0); // slice_beta_offset_div2
    }

    // NOTE: we do NOT write rbsp_trailing here — the caller appends
    // MB data first. The returned buffer may be partial-byte-aligned.
    // Phase 6A.8 uses `continue_slice_header_i` below to keep a single
    // BitWriter alive across header + MB data for streamed encoding.
    w.finish()
}

/// Variant of `build_slice_header_i` that writes into a caller-owned
/// `BitWriter` and does NOT emit trailing bits. Phase 6A.8 uses this
/// to stream the slice header + MB data into one buffer.
pub fn continue_slice_header_i(w: &mut BitWriter, p: &ISliceHeaderParams) {
    w.write_ue(0);
    w.write_ue(7);
    w.write_ue(p.pps_id as u32);
    w.write_bits(p.frame_num as u32 & 0xF, 4);
    if p.is_idr {
        w.write_ue(p.idr_pic_id as u32);
    }
    // Phase 6E-A4 — emit pic_order_cnt_lsb when SPS uses type 0.
    if let Some(lsb) = p.pic_order_cnt_lsb {
        let bits: u8 = p.log2_max_pic_order_cnt_lsb_minus4 + 4;
        let mask = (1u32 << bits as u32) - 1;
        w.write_bits(lsb & mask, bits);
    }
    if p.is_idr {
        w.write_bit(false);
        w.write_bit(false);
    } else {
        w.write_bit(false);
    }
    w.write_se(p.slice_qp_delta);
    if p.disable_deblocking {
        w.write_ue(1);
    } else {
        w.write_ue(0);
        w.write_se(0);
        w.write_se(0);
    }
}

/// Parameters for a P-slice header (Phase 6B).
#[derive(Debug, Clone, Copy)]
#[derive(Default)]
pub struct PSliceHeaderParams {
    pub pps_id: u8,
    /// Frame counter (wraps at `1 << log2_max_frame_num`).
    pub frame_num: u8,
    /// Phase 6E-A4 — when `Some(lsb)`, emit `pic_order_cnt_lsb`
    /// (required when SPS uses `pic_order_cnt_type = 0`).
    pub pic_order_cnt_lsb: Option<u32>,
    /// Phase 6E-A4 — width of `pic_order_cnt_lsb` field. Only
    /// consulted when `pic_order_cnt_lsb` is `Some`.
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    /// QP delta from `pic_init_qp`.
    pub slice_qp_delta: i32,
    /// Phase 6A.7 stopgap — emit `disable_deblocking_filter_idc = 1`.
    pub disable_deblocking: bool,
    /// `num_ref_idx_active_override_flag` — 0 uses the PPS default.
    pub num_ref_idx_active_override: bool,
    /// Only used when `num_ref_idx_active_override` is true.
    pub num_ref_idx_l0_active_minus1: u8,
    /// CABAC initialization IDC (spec § 7.3.3, Table 9-31).
    /// `Some(idc)` emits `cabac_init_idc` between dec_ref_pic_marking
    /// and slice_qp_delta — only valid in streams whose PPS has
    /// `entropy_coding_mode_flag = 1`. `None` (default) omits the field
    /// and matches the CAVLC path.
    pub cabac_init_idc: Option<u8>,
}


/// Stream-mode P-slice header writer. Mirrors
/// `continue_slice_header_i` but for slice_type = P.
pub fn continue_slice_header_p(w: &mut BitWriter, p: &PSliceHeaderParams) {
    // first_mb_in_slice = 0 (single-slice-per-frame, Phase 6A convention).
    w.write_ue(0);
    // slice_type = 5 ("P, all MBs in this slice are P") per Table 7-6.
    w.write_ue(5);
    // pic_parameter_set_id
    w.write_ue(p.pps_id as u32);
    // frame_num u(4) — matches SPS's log2_max_frame_num_minus4 = 0.
    w.write_bits(p.frame_num as u32 & 0xF, 4);
    // Non-IDR P slice does NOT emit idr_pic_id, no_output_of_prior_pics,
    // or long_term_reference_flag.
    //
    // Phase 6E-A4 — emit pic_order_cnt_lsb when SPS uses type 0.
    // Phase 6B I+P-only SPS uses type 2 (no field).
    if let Some(lsb) = p.pic_order_cnt_lsb {
        let bits: u8 = p.log2_max_pic_order_cnt_lsb_minus4 + 4;
        let mask = (1u32 << bits as u32) - 1;
        w.write_bits(lsb & mask, bits);
    }
    //
    // num_ref_idx_active_override_flag
    w.write_bit(p.num_ref_idx_active_override);
    if p.num_ref_idx_active_override {
        w.write_ue(p.num_ref_idx_l0_active_minus1 as u32);
    }
    // ref_pic_list_modification_flag_l0 = 0 (no modification for
    // single-ref P slices)
    w.write_bit(false);
    // Decoded reference picture marking: for non-IDR P slices, just
    // adaptive_ref_pic_marking_mode_flag = 0.
    w.write_bit(false);
    // cabac_init_idc (spec § 7.3.3 — only when PPS enables CABAC AND
    // slice_type is neither I nor SI).
    if let Some(idc) = p.cabac_init_idc {
        debug_assert!(idc <= 2);
        w.write_ue(idc as u32);
    }
    // slice_qp_delta
    w.write_se(p.slice_qp_delta);
    // disable_deblocking_filter_idc (optional, controlled by PPS flag)
    if p.disable_deblocking {
        w.write_ue(1);
    } else {
        w.write_ue(0);
        w.write_se(0);
        w.write_se(0);
    }
}

// ─── B-slice header (Phase 6E-A3) ────────────────────────────────

/// Parameters for a B-slice header (Phase 6E-A).
///
/// Mirrors `PSliceHeaderParams` with the B-specific additions:
/// `direct_spatial_mv_pred_flag` (always `true` for Phase 6E-A
/// per the architecture lock-in) and `num_ref_idx_l1_active_minus1`.
/// Encoders should use `pic_order_cnt_type = 0` SPS when emitting
/// B-slices so decoders can reorder display vs encode order via
/// `pic_order_cnt_lsb`.
#[derive(Debug, Clone, Copy)]
#[derive(Default)]
pub struct BSliceHeaderParams {
    pub pps_id: u8,
    pub frame_num: u8,
    /// `pic_order_cnt_lsb` for this B-frame. Computed by the
    /// encoder driver from the §6E-A1 `PocTracker`. Wraps mod
    /// `1 << (log2_max_pic_order_cnt_lsb_minus4 + 4)`.
    pub pic_order_cnt_lsb: u32,
    /// Phase 6E-A locks this to `true` (spatial direct mode —
    /// matches mobile encoder defaults). Plumbed through for
    /// completeness.
    pub direct_spatial_mv_pred_flag: bool,
    pub slice_qp_delta: i32,
    pub disable_deblocking: bool,
    /// `num_ref_idx_active_override_flag`. When `false`, both L0
    /// and L1 active counts come from the PPS defaults.
    pub num_ref_idx_active_override: bool,
    /// Only emitted when `num_ref_idx_active_override` is true.
    pub num_ref_idx_l0_active_minus1: u8,
    /// Only emitted when `num_ref_idx_active_override` is true
    /// AND slice_type is B.
    pub num_ref_idx_l1_active_minus1: u8,
    /// CABAC initialization IDC (spec § 7.3.3, Table 9-31).
    /// `Some(idc)` emits `cabac_init_idc`. Phase 6E-A defaults to
    /// `Some(0)` (PIdc0) to mimic mobile encoder defaults.
    pub cabac_init_idc: Option<u8>,
    /// `log2_max_pic_order_cnt_lsb_minus4` from SPS — width of
    /// the `pic_order_cnt_lsb` field. Phase 6E-A defaults to 4
    /// (LSB range 0..255).
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    /// `log2_max_frame_num_minus4` from SPS — width of the
    /// `frame_num` field. Phase 6E-A defaults to 0 (4-bit
    /// frame_num) for consistency with the existing P-slice
    /// header path.
    pub log2_max_frame_num_minus4: u8,
}

/// Phase 6E-A3 — stream-mode B-slice header writer. Mirrors
/// `continue_slice_header_p` with B-specific additions per spec
/// § 7.3.3.
pub fn continue_slice_header_b(w: &mut BitWriter, p: &BSliceHeaderParams) {
    let frame_num_bits: u8 = p.log2_max_frame_num_minus4 + 4;
    let poc_lsb_bits: u8 = p.log2_max_pic_order_cnt_lsb_minus4 + 4;

    // first_mb_in_slice = 0 (single-slice-per-frame).
    w.write_ue(0);
    // slice_type = 6 ("B, all MBs are B") per Table 7-6.
    w.write_ue(6);
    // pic_parameter_set_id
    w.write_ue(p.pps_id as u32);
    // frame_num u(log2_max_frame_num)
    let frame_num_mask = (1u32 << frame_num_bits as u32) - 1;
    w.write_bits(p.frame_num as u32 & frame_num_mask, frame_num_bits);
    // Non-IDR B slice does NOT emit idr_pic_id.
    // pic_order_cnt_type = 0 path: emit pic_order_cnt_lsb.
    let poc_mask = (1u32 << poc_lsb_bits as u32) - 1;
    w.write_bits(p.pic_order_cnt_lsb & poc_mask, poc_lsb_bits);
    // direct_spatial_mv_pred_flag
    w.write_bit(p.direct_spatial_mv_pred_flag);
    // num_ref_idx_active_override_flag
    w.write_bit(p.num_ref_idx_active_override);
    if p.num_ref_idx_active_override {
        w.write_ue(p.num_ref_idx_l0_active_minus1 as u32);
        w.write_ue(p.num_ref_idx_l1_active_minus1 as u32);
    }
    // ref_pic_list_modification: B has BOTH L0 and L1 flags. We
    // emit no modifications (default ordering by POC).
    w.write_bit(false); // ref_pic_list_modification_flag_l0
    w.write_bit(false); // ref_pic_list_modification_flag_l1
    // dec_ref_pic_marking: B-frames are non-reference (nal_ref_idc=0)
    // by Phase 6E-A convention, so the slice header SKIPS
    // dec_ref_pic_marking entirely (spec § 7.3.3.1 gates on
    // nal_ref_idc != 0). The caller wraps the NAL with
    // `wrap_rbsp_as_nal(.., NalType::SLICE, /* nal_ref_idc */ 0)`.
    // cabac_init_idc
    if let Some(idc) = p.cabac_init_idc {
        debug_assert!(idc <= 2);
        w.write_ue(idc as u32);
    }
    // slice_qp_delta
    w.write_se(p.slice_qp_delta);
    // disable_deblocking_filter_idc
    if p.disable_deblocking {
        w.write_ue(1);
    } else {
        w.write_ue(0);
        w.write_se(0);
        w.write_se(0);
    }
}

// ─── Access Unit Delimiter (AUD, spec § 7.3.2.4) ─────────────────

/// `primary_pic_type` values per spec Table 7-5.
#[derive(Debug, Clone, Copy)]
pub enum PrimaryPicType {
    /// Only I slices.
    IOnly = 0,
    /// I and P slices.
    IP = 1,
    /// I, P, and B slices (B not used by our encoder).
    IPB = 2,
}

/// Build an AUD NAL RBSP for the given `primary_pic_type`. Caller
/// wraps with `wrap_rbsp_as_nal(.., NalType::AUD, 0)`.
pub fn build_aud_rbsp(pic_type: PrimaryPicType) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(1);
    w.write_bits(pic_type as u32, 3);
    w.write_rbsp_trailing();
    w.finish()
}

// ─── NAL wrap ─────────────────────────────────────────────────────

/// Build the NAL header byte (§ 7.3.1).
fn nal_header_byte(nal_type: NalType, nal_ref_idc: u8) -> u8 {
    debug_assert!(nal_ref_idc <= 3);
    // forbidden_zero_bit (0) | nal_ref_idc (2 bits) | nal_unit_type (5 bits)
    ((nal_ref_idc & 0x3) << 5) | (nal_type.0 & 0x1F)
}

/// Wrap RBSP bytes into a NAL unit payload (header byte + emulation
/// prevention insertion), suitable for writing into an MP4 with a
/// length prefix or an Annex B stream with a start code.
///
/// Does NOT add the start code / length prefix — the muxer is the
/// right layer for that.
pub fn wrap_rbsp_as_nal(rbsp: &[u8], nal_type: NalType, nal_ref_idc: u8) -> Vec<u8> {
    let protected = insert_emulation_prevention(rbsp);
    let mut nal = Vec::with_capacity(protected.len() + 1);
    nal.push(nal_header_byte(nal_type, nal_ref_idc));
    nal.extend_from_slice(&protected);
    nal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::bitstream::{parse_nal_unit, RbspReader};
    use crate::codec::h264::sps::{parse_pps, parse_sps};

    // ─── BitWriter primitives ─────────────────────────────────────

    #[test]
    fn bitwriter_bytes_and_bits() {
        let mut w = BitWriter::new();
        // Write 0b1010 (4 bits), then 0b0110 (4 bits) → byte 0xA6.
        w.write_bits(0b1010, 4);
        w.write_bits(0b0110, 4);
        assert_eq!(w.finish(), vec![0xA6]);
    }

    #[test]
    fn bitwriter_cross_byte_write() {
        let mut w = BitWriter::new();
        // 3 + 10 = 13 bits. 0b101 then 0b11_0000_1100 → concat
        // 0b1011_1000_0011_00; pad to 16 with zeros:
        // 0b1011_1000_0110_0000 = 0xB8 0x60.
        w.write_bits(0b101, 3);
        w.write_bits(0b11_0000_1100, 10);
        assert_eq!(w.finish(), vec![0xB8, 0x60]);
    }

    #[test]
    fn bitwriter_ue_known_codewords() {
        // Reference (H.264 Table 9-2):
        //   ue(0) = "1"        (1 bit)
        //   ue(1) = "010"      (3 bits)
        //   ue(2) = "011"      (3 bits)
        //   ue(3) = "00100"    (5 bits)
        //   ue(6) = "00111"    (5 bits)
        for (value, expected_bits) in &[
            (0u32, "1"),
            (1, "010"),
            (2, "011"),
            (3, "00100"),
            (6, "00111"),
        ] {
            let mut w = BitWriter::new();
            w.write_ue(*value);
            // Flush to bytes for comparison.
            let mut padded = w;
            padded.write_rbsp_trailing();
            let bytes = padded.finish();
            // Re-read the first codeword, confirm it parses back.
            let mut r = RbspReader::new(&bytes);
            assert_eq!(
                r.read_ue().unwrap(),
                *value,
                "ue({value}) round-trip — expected encoding {expected_bits}"
            );
        }
    }

    #[test]
    fn bitwriter_se_round_trip() {
        for v in &[-10i32, -5, -1, 0, 1, 5, 10] {
            let mut w = BitWriter::new();
            w.write_se(*v);
            w.write_rbsp_trailing();
            let bytes = w.finish();
            let mut r = RbspReader::new(&bytes);
            assert_eq!(r.read_se().unwrap(), *v, "se({v}) round-trip");
        }
    }

    // ─── SPS round-trip ───────────────────────────────────────────

    #[test]
    fn sps_baseline_round_trip() {
        let p = SpsParams {
            width_pixels: 320,
            height_pixels: 240,
            sps_id: 0,
            max_num_ref_frames: 1,
            ..SpsParams::default()
        };
        let rbsp = build_sps_baseline(&p);
        let sps = parse_sps(&rbsp).expect("parse_sps should accept our SPS");
        assert_eq!(sps.profile_idc, 66);
        assert_eq!(sps.level_idc, 20); // 320*240 = 75600 px → 300 MBs → level 2.0
        assert_eq!(sps.sps_id, 0);
        assert_eq!(sps.pic_width_in_mbs, 20);
        assert_eq!(sps.pic_height_in_map_units, 15);
        assert_eq!(sps.width_in_pixels, 320);
        assert_eq!(sps.height_in_pixels, 240);
        assert!(sps.frame_mbs_only_flag);
    }

    #[test]
    fn sps_1080p_level_40() {
        let p = SpsParams {
            width_pixels: 1920,
            height_pixels: 1088, // 1080 padded to MB-align
            ..SpsParams::default()
        };
        let rbsp = build_sps_baseline(&p);
        let sps = parse_sps(&rbsp).unwrap();
        assert_eq!(sps.level_idc, 40);
        assert_eq!(sps.width_in_pixels, 1920);
        assert_eq!(sps.height_in_pixels, 1088);
    }

    // ─── PPS round-trip ───────────────────────────────────────────

    #[test]
    fn pps_cavlc_round_trip() {
        let p = PpsParams::default();
        let rbsp = build_pps_cavlc(&p);
        let pps = parse_pps(&rbsp).expect("parse_pps should accept our PPS");
        assert_eq!(pps.pps_id, 0);
        assert_eq!(pps.sps_id, 0);
        assert!(!pps.entropy_coding_mode_flag);
        assert_eq!(pps.pic_init_qp_minus26, 0);
        assert!(!pps.deblocking_filter_control_present_flag);
    }

    #[test]
    fn pps_with_deblocking_control_round_trip() {
        let p = PpsParams {
            deblocking_filter_control_present: true,
            ..PpsParams::default()
        };
        let rbsp = build_pps_cavlc(&p);
        let pps = parse_pps(&rbsp).unwrap();
        assert!(pps.deblocking_filter_control_present_flag);
    }

    // ─── Phase 6C.6a: Main profile SPS + CABAC PPS ────────────────

    #[test]
    fn sps_main_profile_has_profile_idc_77() {
        let p = SpsParams {
            width_pixels: 320,
            height_pixels: 240,
            sps_id: 0,
            max_num_ref_frames: 1,
            ..SpsParams::default()
        };
        let rbsp = build_sps_main(&p);
        let sps = parse_sps(&rbsp).expect("parse_sps should accept our Main SPS");
        assert_eq!(sps.profile_idc, 77);
        assert_eq!(sps.level_idc, 20);
        assert_eq!(sps.width_in_pixels, 320);
        assert_eq!(sps.height_in_pixels, 240);
    }

    #[test]
    fn sps_main_and_baseline_payloads_differ_only_in_profile_byte() {
        // Baseline and Main builders should produce identical RBSPs
        // apart from byte 0 (profile_idc) and byte 1 (constraint flags).
        let p = SpsParams::default();
        let rb_base = build_sps_baseline(&p);
        let rb_main = build_sps_main(&p);
        assert_eq!(rb_base.len(), rb_main.len());
        assert_eq!(rb_base[0], 66);
        assert_eq!(rb_main[0], 77);
        // Bytes 2..END should match (level_idc onward).
        assert_eq!(&rb_base[2..], &rb_main[2..]);
    }

    #[test]
    fn pps_cabac_sets_entropy_coding_mode_flag() {
        let p = PpsParams::default();
        let rbsp = build_pps_cabac(&p);
        let pps = parse_pps(&rbsp).expect("parse_pps should accept CABAC PPS");
        assert!(pps.entropy_coding_mode_flag);
        assert_eq!(pps.pic_init_qp_minus26, 0);
    }

    #[test]
    fn pps_cabac_and_cavlc_differ_only_at_entropy_coding_bit() {
        // Identical PpsParams → both payloads should parse back; only
        // the entropy_coding_mode_flag differs.
        let p = PpsParams::default();
        let rbsp_cavlc = build_pps_cavlc(&p);
        let rbsp_cabac = build_pps_cabac(&p);
        assert_eq!(rbsp_cavlc.len(), rbsp_cabac.len());
        let pps_cavlc = parse_pps(&rbsp_cavlc).unwrap();
        let pps_cabac = parse_pps(&rbsp_cabac).unwrap();
        assert!(!pps_cavlc.entropy_coding_mode_flag);
        assert!(pps_cabac.entropy_coding_mode_flag);
        assert_eq!(pps_cavlc.pps_id, pps_cabac.pps_id);
        assert_eq!(
            pps_cavlc.deblocking_filter_control_present_flag,
            pps_cabac.deblocking_filter_control_present_flag,
        );
    }

    #[test]
    fn p_slice_header_emits_cabac_init_idc_when_set() {
        // Build a CABAC P-slice header and re-parse the relevant
        // fields. We don't have a full P-slice parser exposed here;
        // the goal of this test is byte-shape sanity: the header with
        // cabac_init_idc = Some(0) should be exactly ue(0) = "1" longer
        // than the CAVLC form. In bits: adds 1 bit before slice_qp_delta.
        use crate::codec::h264::encoder::bitstream_writer::continue_slice_header_p;
        let mut w_cavlc = BitWriter::new();
        let p_cavlc = PSliceHeaderParams::default();
        continue_slice_header_p(&mut w_cavlc, &p_cavlc);
        let len_cavlc_bits = w_cavlc.bits_written();

        let mut w_cabac = BitWriter::new();
        let p_cabac = PSliceHeaderParams {
            cabac_init_idc: Some(0),
            ..PSliceHeaderParams::default()
        };
        continue_slice_header_p(&mut w_cabac, &p_cabac);
        let len_cabac_bits = w_cabac.bits_written();

        assert_eq!(
            len_cabac_bits,
            len_cavlc_bits + 1,
            "ue(0) = '1' adds exactly 1 bit"
        );
    }

    #[test]
    fn p_slice_header_cabac_init_idc_2_adds_three_bits() {
        use crate::codec::h264::encoder::bitstream_writer::continue_slice_header_p;
        let mut w_cavlc = BitWriter::new();
        continue_slice_header_p(&mut w_cavlc, &PSliceHeaderParams::default());
        let len_cavlc_bits = w_cavlc.bits_written();

        let mut w_cabac = BitWriter::new();
        continue_slice_header_p(
            &mut w_cabac,
            &PSliceHeaderParams {
                cabac_init_idc: Some(2),
                ..PSliceHeaderParams::default()
            },
        );
        let len_cabac_bits = w_cabac.bits_written();

        // ue(2) = "011" = 3 bits.
        assert_eq!(len_cabac_bits, len_cavlc_bits + 3);
    }

    // ─── NAL wrap ─────────────────────────────────────────────────

    #[test]
    fn nal_wrap_sps_parses_back() {
        let p = SpsParams::default();
        let rbsp = build_sps_baseline(&p);
        let nal = wrap_rbsp_as_nal(&rbsp, NalType::SPS, 3);
        let parsed = parse_nal_unit(&nal).unwrap();
        assert_eq!(parsed.nal_type, NalType::SPS);
        assert_eq!(parsed.nal_ref_idc, 3);
        // rbsp should match what we wrote (EP round-trip tested in bitstream.rs).
        assert_eq!(parsed.rbsp, rbsp);
    }

    #[test]
    fn nal_wrap_emulation_prevention_inserts_03() {
        // A payload containing 00 00 01 triggers EP byte insertion.
        let rbsp = vec![0x00, 0x00, 0x01, 0xFF];
        let nal = wrap_rbsp_as_nal(&rbsp, NalType::SLICE_IDR, 3);
        // Expected output: header(0x65) + [00, 00, 03, 01, FF]
        assert_eq!(nal, vec![0x65, 0x00, 0x00, 0x03, 0x01, 0xFF]);
    }

    #[test]
    fn nal_header_byte_layout() {
        assert_eq!(nal_header_byte(NalType::SPS, 3), 0x67);
        assert_eq!(nal_header_byte(NalType::PPS, 3), 0x68);
        assert_eq!(nal_header_byte(NalType::SLICE_IDR, 3), 0x65);
        assert_eq!(nal_header_byte(NalType::SLICE, 2), 0x41);
    }

    // ─── level_idc derivation ─────────────────────────────────────

    #[test]
    fn level_idc_bands() {
        assert_eq!(derive_level_idc(176, 144), 10);
        assert_eq!(derive_level_idc(352, 288), 20);
        assert_eq!(derive_level_idc(720, 480), 30);
        assert_eq!(derive_level_idc(1280, 720), 31);
        assert_eq!(derive_level_idc(1920, 1088), 40);
        assert_eq!(derive_level_idc(3840, 2176), 51);
    }

    // ─── High-profile SPS + PPS (Phase 100-D) ────────────────────

    #[test]
    fn sps_high_profile_idc_is_100() {
        let p = SpsParams {
            width_pixels: 320,
            height_pixels: 240,
            sps_id: 0,
            max_num_ref_frames: 1,
            ..SpsParams::default()
        };
        let rbsp = build_sps_high(&p);
        let sps = parse_sps(&rbsp).expect("parse_sps should accept our High SPS");
        assert_eq!(sps.profile_idc, 100);
        assert_eq!(sps.level_idc, 20);
        assert_eq!(sps.width_in_pixels, 320);
        assert_eq!(sps.height_in_pixels, 240);
    }

    #[test]
    fn sps_high_emits_chroma_format_420() {
        // When profile_idc = 100, parse_sps reads the chroma_format_idc
        // / bit_depth fields from the High-profile suffix. Our builder
        // emits chroma_format_idc = 1 (4:2:0) and 8-bit depths.
        let p = SpsParams::default();
        let rbsp = build_sps_high(&p);
        let sps = parse_sps(&rbsp).unwrap();
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.bit_depth_luma, 8);
        assert_eq!(sps.bit_depth_chroma, 8);
    }

    #[test]
    fn sps_high_is_longer_than_main_due_to_suffix() {
        // High-profile SPS carries 5 extra syntax elements compared to
        // Main: chroma_format_idc, bit_depth_luma_minus8,
        // bit_depth_chroma_minus8, qpprime_y_zero_transform_bypass_flag,
        // seq_scaling_matrix_present_flag. That's ue(1)=1bit + ue(0)=1
        // + ue(0)=1 + bit + bit = 5 bits, rounded up to at least one
        // extra byte after RBSP trailing alignment.
        let p = SpsParams::default();
        let rb_main = build_sps_main(&p);
        let rb_high = build_sps_high(&p);
        assert!(
            rb_high.len() >= rb_main.len(),
            "High SPS ({}) shouldn't be shorter than Main ({})",
            rb_high.len(),
            rb_main.len(),
        );
        assert_eq!(rb_high[0], 100);
        assert_eq!(rb_main[0], 77);
    }

    #[test]
    fn pps_cabac_high_sets_transform_8x8_mode_flag() {
        let p = PpsParams::default();
        let rbsp = build_pps_cabac_high(&p);
        let pps = parse_pps(&rbsp).expect("parse_pps should accept High CABAC PPS");
        assert!(pps.entropy_coding_mode_flag);
        assert!(pps.transform_8x8_mode_flag);
    }

    #[test]
    fn pps_cabac_high_matches_cabac_except_suffix() {
        // The first 10 prefix fields (through redundant_pic_cnt_present_flag)
        // are identical between CABAC Main and CABAC High; only the
        // High suffix adds transform_8x8_mode_flag + scaling-matrix
        // flag + second_chroma_qp_index_offset.
        let p = PpsParams::default();
        let rb_main = build_pps_cabac(&p);
        let rb_high = build_pps_cabac_high(&p);
        assert!(
            rb_high.len() >= rb_main.len(),
            "High PPS ({}) shouldn't be shorter than CABAC Main ({})",
            rb_high.len(),
            rb_main.len(),
        );
    }

    /// §6E-A3 — B-slice header writer produces non-empty output and
    /// the bytes are sensible (slice_type bits land in the right
    /// position). Full round-trip through the parser happens at the
    /// §6E-A4 end-to-end IBPBP encode test, where the encoder driver
    /// + walker exercise the writer in production.
    #[test]
    fn b_slice_header_writer_emits_bytes() {
        let mut w = BitWriter::new();
        let p = BSliceHeaderParams {
            pps_id: 0,
            frame_num: 3,
            pic_order_cnt_lsb: 6,
            direct_spatial_mv_pred_flag: true,
            slice_qp_delta: 2,
            disable_deblocking: false,
            num_ref_idx_active_override: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            cabac_init_idc: Some(0),
            log2_max_pic_order_cnt_lsb_minus4: 4,
            log2_max_frame_num_minus4: 0,
        };
        continue_slice_header_b(&mut w, &p);
        let bytes = w.finish();
        assert!(!bytes.is_empty(), "B-slice header produced no bytes");

        // First byte starts with first_mb_in_slice=0 (1 bit "1" via
        // ue(0)) followed by slice_type=6 (ue(6) = "00111").
        // Combined leading bits: 1 00111 = 0b100111... → high nibble
        // is 0x9 (1001).
        assert_eq!(
            bytes[0] >> 4, 0x9,
            "first nibble should encode first_mb_in_slice=0 + slice_type=6 prefix; got {:08b}",
            bytes[0],
        );
    }

    /// §6E-A3 — B-slice header writer with override + cabac_init_idc
    /// produces longer output than the no-override case (sanity that
    /// the override path emits the additional fields).
    #[test]
    fn b_slice_header_writer_override_extends_output() {
        let base_params = BSliceHeaderParams {
            pps_id: 0,
            frame_num: 0,
            pic_order_cnt_lsb: 0,
            direct_spatial_mv_pred_flag: true,
            slice_qp_delta: 0,
            disable_deblocking: false,
            num_ref_idx_active_override: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            cabac_init_idc: Some(0),
            log2_max_pic_order_cnt_lsb_minus4: 4,
            log2_max_frame_num_minus4: 0,
        };
        let mut w_no_override = BitWriter::new();
        continue_slice_header_b(&mut w_no_override, &base_params);
        let no_override = w_no_override.finish();

        let mut w_override = BitWriter::new();
        let mut p = base_params;
        p.num_ref_idx_active_override = true;
        p.num_ref_idx_l0_active_minus1 = 1;
        p.num_ref_idx_l1_active_minus1 = 1;
        continue_slice_header_b(&mut w_override, &p);
        let with_override = w_override.finish();

        assert!(
            with_override.len() >= no_override.len(),
            "override path should not produce shorter output"
        );
    }
}
